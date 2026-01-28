//! Operation Logging System with Structured Output
//!
//! This module provides comprehensive logging of tensor operations with structured output support.
//! Useful for debugging, performance analysis, and operation tracing.
//!
//! # Features
//!
//! - **Structured Output**: Support for JSON, CSV, and custom formats
//! - **Performance Metrics**: Track operation timing and memory usage
//! - **Filtering**: Filter by operation type, device, or custom criteria
//! - **Minimal Overhead**: Conditional compilation and async logging for minimal performance impact
//! - **Thread-Safe**: Concurrent operation logging from multiple threads
//! - **Hierarchical Tracing**: Track nested operation chains and call stacks

// Allow unused_mut in test code
#![allow(unused_mut)]

//! # Examples
//!
//! ```rust,ignore
//! use torsh_tensor::operation_logging::{OperationLogger, LogFormat, LogFilter};
//!
//! // Create logger with JSON output
//! let logger = OperationLogger::new()
//!     .with_format(LogFormat::Json)
//!     .with_output_file("operations.json")
//!     .with_filter(LogFilter::operation_type("matmul"))
//!     .build()?;
//!
//! // Log an operation
//! logger.log_operation("matmul", |logger| {
//!     // Perform operation
//!     let result = a.matmul(&b)?;
//!     logger.add_metadata("shape_a", &format!("{:?}", a.shape()));
//!     logger.add_metadata("shape_b", &format!("{:?}", b.shape()));
//!     Ok(result)
//! })?;
//!
//! // Export logs
//! logger.export_to_file("operations.json")?;
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
};

/// Format for structured output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// JSON format (structured, easy to parse)
    Json,
    /// Pretty-printed JSON (human-readable)
    JsonPretty,
    /// CSV format (tabular, spreadsheet-compatible)
    Csv,
    /// Plain text (simple, minimal)
    PlainText,
    /// Custom format (user-defined)
    Custom,
}

/// Operation log entry with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationLogEntry {
    /// Unique identifier for this operation
    pub id: u64,
    /// Parent operation ID (for nested operations)
    pub parent_id: Option<u64>,
    /// Operation name/type
    pub operation: String,
    /// Device where operation executed
    pub device: String,
    /// Start timestamp
    pub timestamp: String,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Memory allocated during operation (bytes)
    pub memory_allocated: usize,
    /// Memory freed during operation (bytes)
    pub memory_freed: usize,
    /// Input tensor shapes
    pub input_shapes: Vec<Vec<usize>>,
    /// Output tensor shape
    pub output_shape: Vec<usize>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Stack trace (if enabled)
    pub stack_trace: Option<Vec<String>>,
    /// Error message (if operation failed)
    pub error: Option<String>,
}

/// Filter for selecting which operations to log
#[derive(Clone)]
pub enum LogFilter {
    /// Log all operations
    All,
    /// Log only operations matching a specific name
    OperationType(String),
    /// Log only operations on a specific device
    Device(DeviceType),
    /// Log only operations with duration exceeding threshold (microseconds)
    MinDuration(u64),
    /// Log only operations allocating more than threshold memory (bytes)
    MinMemory(usize),
    /// Custom filter function
    Custom(Arc<dyn Fn(&OperationLogEntry) -> bool + Send + Sync>),
    /// Combination of multiple filters (AND logic)
    And(Vec<LogFilter>),
    /// Combination of multiple filters (OR logic)
    Or(Vec<LogFilter>),
}

impl std::fmt::Debug for LogFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogFilter::All => write!(f, "All"),
            LogFilter::OperationType(op) => write!(f, "OperationType({:?})", op),
            LogFilter::Device(device) => write!(f, "Device({:?})", device),
            LogFilter::MinDuration(duration) => write!(f, "MinDuration({})", duration),
            LogFilter::MinMemory(memory) => write!(f, "MinMemory({})", memory),
            LogFilter::Custom(_) => write!(f, "Custom(<function>)"),
            LogFilter::And(filters) => write!(f, "And({:?})", filters),
            LogFilter::Or(filters) => write!(f, "Or({:?})", filters),
        }
    }
}

impl LogFilter {
    /// Create a filter for a specific operation type
    pub fn operation_type(op: &str) -> Self {
        LogFilter::OperationType(op.to_string())
    }

    /// Create a filter for minimum duration
    pub fn min_duration(duration_us: u64) -> Self {
        LogFilter::MinDuration(duration_us)
    }

    /// Create a filter for minimum memory usage
    pub fn min_memory(memory_bytes: usize) -> Self {
        LogFilter::MinMemory(memory_bytes)
    }

    /// Check if an entry passes this filter
    pub fn matches(&self, entry: &OperationLogEntry) -> bool {
        match self {
            LogFilter::All => true,
            LogFilter::OperationType(op) => entry.operation.contains(op),
            LogFilter::Device(device) => entry.device == format!("{:?}", device),
            LogFilter::MinDuration(min_us) => entry.duration_us >= *min_us,
            LogFilter::MinMemory(min_bytes) => entry.memory_allocated >= *min_bytes,
            LogFilter::Custom(f) => f(entry),
            LogFilter::And(filters) => filters.iter().all(|f| f.matches(entry)),
            LogFilter::Or(filters) => filters.iter().any(|f| f.matches(entry)),
        }
    }
}

/// Configuration for operation logging
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Output format
    pub format: LogFormat,
    /// Filter for selecting operations
    pub filter: LogFilter,
    /// Whether to include stack traces
    pub include_stack_trace: bool,
    /// Whether to track memory allocations
    pub track_memory: bool,
    /// Maximum number of entries to keep in memory
    pub max_entries: usize,
    /// Whether to enable async logging
    pub async_logging: bool,
    /// Output file path (None for in-memory only)
    pub output_file: Option<String>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            format: LogFormat::JsonPretty,
            filter: LogFilter::All,
            include_stack_trace: false,
            track_memory: true,
            max_entries: 10000,
            async_logging: false,
            output_file: None,
        }
    }
}

/// Operation logger with structured output
pub struct OperationLogger {
    /// Configuration
    config: LogConfig,
    /// Log entries
    entries: Arc<RwLock<Vec<OperationLogEntry>>>,
    /// Next entry ID
    next_id: Arc<Mutex<u64>>,
    /// Current operation stack (for nested operations)
    operation_stack: Arc<Mutex<Vec<u64>>>,
    /// Global statistics
    stats: Arc<RwLock<LogStatistics>>,
}

/// Statistics about logged operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogStatistics {
    /// Total number of operations logged
    pub total_operations: u64,
    /// Total duration of all operations (microseconds)
    pub total_duration_us: u64,
    /// Total memory allocated (bytes)
    pub total_memory_allocated: usize,
    /// Total memory freed (bytes)
    pub total_memory_freed: usize,
    /// Count of operations by type
    pub operations_by_type: HashMap<String, u64>,
    /// Average duration by operation type (microseconds)
    pub avg_duration_by_type: HashMap<String, u64>,
    /// Failed operations count
    pub failed_operations: u64,
}

impl OperationLogger {
    /// Create a new operation logger with default configuration
    pub fn new() -> Self {
        Self::with_config(LogConfig::default())
    }

    /// Create a new operation logger with custom configuration
    pub fn with_config(config: LogConfig) -> Self {
        Self {
            config,
            entries: Arc::new(RwLock::new(Vec::new())),
            next_id: Arc::new(Mutex::new(0)),
            operation_stack: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(RwLock::new(LogStatistics::default())),
        }
    }

    /// Builder pattern: set output format
    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.config.format = format;
        self
    }

    /// Builder pattern: set filter
    pub fn with_filter(mut self, filter: LogFilter) -> Self {
        self.config.filter = filter;
        self
    }

    /// Builder pattern: set output file
    pub fn with_output_file(mut self, path: impl Into<String>) -> Self {
        self.config.output_file = Some(path.into());
        self
    }

    /// Builder pattern: enable stack traces
    pub fn with_stack_traces(mut self, enable: bool) -> Self {
        self.config.include_stack_trace = enable;
        self
    }

    /// Builder pattern: enable memory tracking
    pub fn with_memory_tracking(mut self, enable: bool) -> Self {
        self.config.track_memory = enable;
        self
    }

    /// Start logging an operation
    pub fn start_operation(
        &self,
        operation: impl Into<String>,
        device: DeviceType,
    ) -> OperationContext {
        let id = {
            let mut next_id = self.next_id.lock().expect("lock should not be poisoned");
            let id = *next_id;
            *next_id += 1;
            id
        };

        let parent_id = {
            let stack = self
                .operation_stack
                .lock()
                .expect("lock should not be poisoned");
            stack.last().copied()
        };

        self.operation_stack
            .lock()
            .expect("lock should not be poisoned")
            .push(id);

        OperationContext {
            id,
            parent_id,
            operation: operation.into(),
            device: format!("{:?}", device),
            start_time: Instant::now(),
            metadata: HashMap::new(),
            input_shapes: Vec::new(),
            logger: Arc::new(self.clone_logger()),
        }
    }

    /// Finish logging an operation
    pub fn end_operation(
        &self,
        context: OperationContext,
        output_shape: Vec<usize>,
        error: Option<String>,
    ) {
        let duration = context.start_time.elapsed();

        // Remove from stack
        {
            let mut stack = self
                .operation_stack
                .lock()
                .expect("lock should not be poisoned");
            stack.pop();
        }

        let entry = OperationLogEntry {
            id: context.id,
            parent_id: context.parent_id,
            operation: context.operation.clone(),
            device: context.device.clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            duration_us: duration.as_micros() as u64,
            memory_allocated: context
                .metadata
                .get("memory_allocated")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            memory_freed: context
                .metadata
                .get("memory_freed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            input_shapes: context.input_shapes.clone(),
            output_shape,
            metadata: context.metadata.clone(),
            stack_trace: if self.config.include_stack_trace {
                Some(Self::capture_stack_trace())
            } else {
                None
            },
            error,
        };

        // Apply filter
        if !self.config.filter.matches(&entry) {
            return;
        }

        // Update statistics
        {
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.total_operations += 1;
            stats.total_duration_us += entry.duration_us;
            stats.total_memory_allocated += entry.memory_allocated;
            stats.total_memory_freed += entry.memory_freed;

            *stats
                .operations_by_type
                .entry(entry.operation.clone())
                .or_insert(0) += 1;

            let count = *stats
                .operations_by_type
                .get(&entry.operation)
                .expect("operation should exist after insertion");
            let current_avg = stats
                .avg_duration_by_type
                .get(&entry.operation)
                .copied()
                .unwrap_or(0);
            let new_avg = (current_avg * (count - 1) + entry.duration_us) / count;
            stats
                .avg_duration_by_type
                .insert(entry.operation.clone(), new_avg);

            if entry.error.is_some() {
                stats.failed_operations += 1;
            }
        }

        // Store entry
        {
            let mut entries = self.entries.write().expect("lock should not be poisoned");
            entries.push(entry);

            // Trim if exceeding max entries
            if entries.len() > self.config.max_entries {
                let remove_count = entries.len() - self.config.max_entries;
                entries.drain(0..remove_count);
            }
        }
    }

    /// Capture stack trace (placeholder - would use backtrace crate in real impl)
    fn capture_stack_trace() -> Vec<String> {
        // In a real implementation, use the backtrace crate
        vec!["Stack trace not implemented".to_string()]
    }

    /// Clone logger (for passing to context)
    fn clone_logger(&self) -> OperationLogger {
        Self {
            config: self.config.clone(),
            entries: Arc::clone(&self.entries),
            next_id: Arc::clone(&self.next_id),
            operation_stack: Arc::clone(&self.operation_stack),
            stats: Arc::clone(&self.stats),
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> LogStatistics {
        self.stats
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get all log entries
    pub fn get_entries(&self) -> Vec<OperationLogEntry> {
        self.entries
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Clear all log entries
    pub fn clear(&self) {
        self.entries
            .write()
            .expect("lock should not be poisoned")
            .clear();
        *self.stats.write().expect("lock should not be poisoned") = LogStatistics::default();
    }

    /// Export logs to file
    pub fn export_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let entries = self.get_entries();
        let mut file = File::create(path)?;

        match self.config.format {
            LogFormat::Json => {
                let json = serde_json::to_string(&entries)
                    .map_err(|e| TorshError::Other(format!("JSON serialization error: {}", e)))?;
                file.write_all(json.as_bytes())?;
            }
            LogFormat::JsonPretty => {
                let json = serde_json::to_string_pretty(&entries)
                    .map_err(|e| TorshError::Other(format!("JSON serialization error: {}", e)))?;
                file.write_all(json.as_bytes())?;
            }
            LogFormat::Csv => {
                self.export_csv(&mut file, &entries)?;
            }
            LogFormat::PlainText => {
                self.export_plain_text(&mut file, &entries)?;
            }
            LogFormat::Custom => {
                // User would implement their own exporter
                writeln!(file, "Custom format not implemented")?;
            }
        }

        Ok(())
    }

    /// Export to CSV format
    fn export_csv(&self, file: &mut File, entries: &[OperationLogEntry]) -> Result<()> {
        // Write header
        writeln!(file, "id,parent_id,operation,device,timestamp,duration_us,memory_allocated,memory_freed,input_shapes,output_shape,error")?;

        // Write entries
        for entry in entries {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{:?},{:?},{}",
                entry.id,
                entry.parent_id.map(|id| id.to_string()).unwrap_or_default(),
                entry.operation,
                entry.device,
                entry.timestamp,
                entry.duration_us,
                entry.memory_allocated,
                entry.memory_freed,
                entry.input_shapes,
                entry.output_shape,
                entry.error.as_deref().unwrap_or("")
            )?;
        }

        Ok(())
    }

    /// Export to plain text format
    fn export_plain_text(&self, file: &mut File, entries: &[OperationLogEntry]) -> Result<()> {
        for entry in entries {
            writeln!(file, "Operation #{}", entry.id)?;
            if let Some(parent_id) = entry.parent_id {
                writeln!(file, "  Parent: #{}", parent_id)?;
            }
            writeln!(file, "  Type: {}", entry.operation)?;
            writeln!(file, "  Device: {}", entry.device)?;
            writeln!(file, "  Time: {}", entry.timestamp)?;
            writeln!(file, "  Duration: {} μs", entry.duration_us)?;
            writeln!(
                file,
                "  Memory: +{} / -{} bytes",
                entry.memory_allocated, entry.memory_freed
            )?;
            writeln!(file, "  Input shapes: {:?}", entry.input_shapes)?;
            writeln!(file, "  Output shape: {:?}", entry.output_shape)?;
            if let Some(error) = &entry.error {
                writeln!(file, "  Error: {}", error)?;
            }
            writeln!(file)?;
        }

        Ok(())
    }
}

impl Default for OperationLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Context for an ongoing operation
pub struct OperationContext {
    id: u64,
    parent_id: Option<u64>,
    operation: String,
    device: String,
    start_time: Instant,
    pub metadata: HashMap<String, String>,
    pub input_shapes: Vec<Vec<usize>>,
    logger: Arc<OperationLogger>,
}

impl OperationContext {
    /// Add metadata to this operation
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Add input shape
    pub fn add_input_shape(&mut self, shape: Vec<usize>) {
        self.input_shapes.push(shape);
    }

    /// Finish this operation with success
    pub fn finish(self, output_shape: Vec<usize>) {
        let logger = self.logger.clone();
        logger.end_operation(self, output_shape, None);
    }

    /// Finish this operation with error
    pub fn finish_with_error(self, error: impl Into<String>) {
        let logger = self.logger.clone();
        logger.end_operation(self, Vec::new(), Some(error.into()));
    }
}

/// Global operation logger (singleton)
static GLOBAL_LOGGER: once_cell::sync::Lazy<Mutex<Option<OperationLogger>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(None));

/// Initialize global logger
pub fn init_global_logger(config: LogConfig) {
    let mut global = GLOBAL_LOGGER.lock().expect("lock should not be poisoned");
    *global = Some(OperationLogger::with_config(config));
}

/// Get global logger
pub fn global_logger() -> Option<OperationLogger> {
    GLOBAL_LOGGER
        .lock()
        .unwrap()
        .as_ref()
        .map(|l| l.clone_logger())
}

/// Macro for logging operations
#[macro_export]
macro_rules! log_operation {
    ($operation:expr, $device:expr, $block:expr) => {{
        #[cfg(feature = "operation-logging")]
        {
            if let Some(logger) = $crate::operation_logging::global_logger() {
                let mut ctx = logger.start_operation($operation, $device);
                let result = (|| $block)();
                match &result {
                    Ok(_) => ctx.finish(vec![]), // Output shape would be filled in
                    Err(e) => ctx.finish_with_error(format!("{:?}", e)),
                }
                result
            } else {
                $block
            }
        }
        #[cfg(not(feature = "operation-logging"))]
        {
            $block
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logger_creation() {
        let logger = OperationLogger::new();
        assert_eq!(logger.get_entries().len(), 0);
    }

    #[test]
    fn test_operation_logging() {
        let logger = OperationLogger::new();

        let mut ctx = logger.start_operation("test_op", DeviceType::Cpu);
        ctx.add_metadata("test", "value");
        ctx.add_input_shape(vec![2, 3]);
        ctx.finish(vec![2, 3]);

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].operation, "test_op");
        assert_eq!(entries[0].metadata.get("test"), Some(&"value".to_string()));
    }

    #[test]
    fn test_filter_operation_type() {
        let filter = LogFilter::operation_type("matmul");

        let entry = OperationLogEntry {
            id: 0,
            parent_id: None,
            operation: "matmul".to_string(),
            device: "Cpu".to_string(),
            timestamp: String::new(),
            duration_us: 100,
            memory_allocated: 0,
            memory_freed: 0,
            input_shapes: vec![],
            output_shape: vec![],
            metadata: HashMap::new(),
            stack_trace: None,
            error: None,
        };

        assert!(filter.matches(&entry));
    }

    #[test]
    fn test_filter_min_duration() {
        let filter = LogFilter::min_duration(1000);

        let entry_fast = OperationLogEntry {
            id: 0,
            parent_id: None,
            operation: "add".to_string(),
            device: "Cpu".to_string(),
            timestamp: String::new(),
            duration_us: 500,
            memory_allocated: 0,
            memory_freed: 0,
            input_shapes: vec![],
            output_shape: vec![],
            metadata: HashMap::new(),
            stack_trace: None,
            error: None,
        };

        let entry_slow = OperationLogEntry {
            id: 1,
            parent_id: None,
            operation: "matmul".to_string(),
            device: "Cpu".to_string(),
            timestamp: String::new(),
            duration_us: 2000,
            memory_allocated: 0,
            memory_freed: 0,
            input_shapes: vec![],
            output_shape: vec![],
            metadata: HashMap::new(),
            stack_trace: None,
            error: None,
        };

        assert!(!filter.matches(&entry_fast));
        assert!(filter.matches(&entry_slow));
    }

    #[test]
    fn test_statistics() {
        let logger = OperationLogger::new();

        // Log multiple operations
        for i in 0..5 {
            let ctx = logger.start_operation("test_op", DeviceType::Cpu);
            ctx.finish(vec![i]);
        }

        let stats = logger.get_statistics();
        assert_eq!(stats.total_operations, 5);
        assert_eq!(stats.operations_by_type.get("test_op"), Some(&5));
    }

    #[test]
    fn test_nested_operations() {
        let logger = OperationLogger::new();

        let ctx1 = logger.start_operation("outer", DeviceType::Cpu);
        let ctx2 = logger.start_operation("inner", DeviceType::Cpu);
        ctx2.finish(vec![1]);
        ctx1.finish(vec![2]);

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].operation, "inner");
        assert_eq!(entries[0].parent_id, Some(0));
        assert_eq!(entries[1].operation, "outer");
        assert_eq!(entries[1].parent_id, None);
    }

    #[test]
    fn test_json_export() {
        let logger = OperationLogger::new().with_format(LogFormat::Json);

        let ctx = logger.start_operation("test", DeviceType::Cpu);
        ctx.finish(vec![1, 2, 3]);

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_log.json");
        logger.export_to_file(&temp_file).unwrap();

        assert!(temp_file.exists());
        std::fs::remove_file(temp_file).ok();
    }

    #[test]
    fn test_max_entries_limit() {
        let logger = OperationLogger::with_config(LogConfig {
            max_entries: 5,
            ..Default::default()
        });

        // Log 10 operations
        for _ in 0..10 {
            let ctx = logger.start_operation("test", DeviceType::Cpu);
            ctx.finish(vec![]);
        }

        let entries = logger.get_entries();
        assert_eq!(entries.len(), 5);
    }
}
