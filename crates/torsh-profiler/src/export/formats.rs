//! Export format implementations
//!
//! This module provides various export formats for profiling data,
//! extracted from the massive lib.rs implementation.

use crate::{core::profiler::global_profiler, ProfileEvent, TorshResult};
use std::fs::File;
use std::io::{BufWriter, Write};
use torsh_core::TorshError;

// Re-export from existing modules for backward compatibility
pub use crate::chrome_trace::*;
pub use crate::custom_export::*;
pub use crate::tensorboard::*;

/// Export formats supported by the profiler
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportFormat {
    /// Chrome trace JSON format
    ChromeTrace,
    /// Standard JSON format
    Json,
    /// Comma-separated values format
    Csv,
    /// TensorBoard format
    TensorBoard,
    /// Custom compact JSON
    CompactJson,
    /// Performance-focused CSV
    PerformanceCsv,
    /// Simple text format
    SimpleText,
}

impl ExportFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::ChromeTrace => "json",
            ExportFormat::Json => "json",
            ExportFormat::Csv => "csv",
            ExportFormat::TensorBoard => "pb",
            ExportFormat::CompactJson => "json",
            ExportFormat::PerformanceCsv => "csv",
            ExportFormat::SimpleText => "txt",
        }
    }

    /// Get MIME type for this format
    pub fn mime_type(&self) -> &'static str {
        match self {
            ExportFormat::ChromeTrace | ExportFormat::Json | ExportFormat::CompactJson => {
                "application/json"
            }
            ExportFormat::Csv | ExportFormat::PerformanceCsv => "text/csv",
            ExportFormat::TensorBoard => "application/x-protobuf",
            ExportFormat::SimpleText => "text/plain",
        }
    }
}

/// Export events to Chrome trace format
pub fn export_chrome_trace_format(events: &[ProfileEvent], path: &str) -> TorshResult<()> {
    let file = File::create(path).map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to create file {}: {}", path, e))
    })?;

    let mut chrome_events = Vec::new();

    for event in events {
        let chrome_event = serde_json::json!({
            "name": event.name,
            "cat": event.category,
            "ph": "X", // Complete event
            "ts": event.start_us,
            "dur": event.duration_us,
            "pid": 1,
            "tid": event.thread_id,
            "args": {
                "operation_count": event.operation_count,
                "flops": event.flops,
                "bytes_transferred": event.bytes_transferred,
                "stack_trace": event.stack_trace
            }
        });
        chrome_events.push(chrome_event);
    }

    let output = serde_json::json!({
        "traceEvents": chrome_events,
        "displayTimeUnit": "ms",
        "systemTraceEvents": "SystemTraceData",
        "otherData": {
            "version": "torsh-profiler-1.0"
        }
    });

    serde_json::to_writer_pretty(file, &output)
        .map_err(|e| TorshError::InvalidArgument(format!("Failed to write Chrome trace: {}", e)))
}

/// Export events to standard JSON format
pub fn export_json_format(events: &[ProfileEvent], path: &str) -> TorshResult<()> {
    let file = File::create(path).map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to create file {}: {}", path, e))
    })?;

    serde_json::to_writer_pretty(file, events)
        .map_err(|e| TorshError::InvalidArgument(format!("Failed to write JSON: {}", e)))
}

/// Export events to CSV format
pub fn export_csv_format(events: &[ProfileEvent], path: &str) -> TorshResult<()> {
    let file = File::create(path).map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to create file {}: {}", path, e))
    })?;

    let mut writer = BufWriter::new(file);

    // Write CSV header
    writeln!(
        writer,
        "name,category,start_us,duration_us,thread_id,operation_count,flops,bytes_transferred,stack_trace"
    ).map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to write CSV header: {}", e))
    })?;

    // Write events
    for event in events {
        let stack_trace_str = event.stack_trace.as_deref().unwrap_or("");
        // Escape stack trace for CSV (replace newlines with \\n and quotes with "")
        let escaped_stack_trace = stack_trace_str.replace('\n', "\\n").replace('"', "\"\"");

        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},\"{}\"",
            event.name,
            event.category,
            event.start_us,
            event.duration_us,
            event.thread_id,
            event.operation_count.unwrap_or(0),
            event.flops.unwrap_or(0),
            event.bytes_transferred.unwrap_or(0),
            escaped_stack_trace
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Failed to write CSV row: {}", e)))?;
    }

    writer
        .flush()
        .map_err(|e| TorshError::InvalidArgument(format!("Failed to flush CSV writer: {}", e)))
}

/// Generic export function that dispatches to appropriate format handler
pub fn export_events(events: &[ProfileEvent], format: ExportFormat, path: &str) -> TorshResult<()> {
    match format {
        ExportFormat::ChromeTrace => export_chrome_trace_format(events, path),
        ExportFormat::Json => export_json_format(events, path),
        ExportFormat::Csv => export_csv_format(events, path),
        ExportFormat::CompactJson => export_json_format(events, path), // For now, same as JSON
        ExportFormat::PerformanceCsv => export_csv_format(events, path), // For now, same as CSV
        ExportFormat::SimpleText => export_csv_format(events, path),   // For now, same as CSV
        ExportFormat::TensorBoard => Err(TorshError::InvalidArgument(
            "TensorBoard export not yet implemented".to_string(),
        )),
    }
}

/// Export global profiler events
pub fn export_global_events(format: ExportFormat, path: &str) -> TorshResult<()> {
    let profiler_guard = global_profiler();
    let events = {
        let profiler = profiler_guard.lock();
        profiler.events().to_vec()
    };
    export_events(&events, format, path)
}

/// Get available export format names
pub fn available_format_names() -> Vec<String> {
    vec![
        "chrome_trace".to_string(),
        "json".to_string(),
        "csv".to_string(),
        "compact_json".to_string(),
        "performance_csv".to_string(),
        "simple_text".to_string(),
    ]
}

/// Parse format name to ExportFormat
pub fn parse_format(name: &str) -> Option<ExportFormat> {
    match name.to_lowercase().as_str() {
        "chrome_trace" | "chrome" => Some(ExportFormat::ChromeTrace),
        "json" => Some(ExportFormat::Json),
        "csv" => Some(ExportFormat::Csv),
        "compact_json" | "compact" => Some(ExportFormat::CompactJson),
        "performance_csv" | "performance" => Some(ExportFormat::PerformanceCsv),
        "simple_text" | "text" => Some(ExportFormat::SimpleText),
        "tensorboard" | "tb" => Some(ExportFormat::TensorBoard),
        _ => None,
    }
}
