//! Performance data export functionality

use crate::TorshResult;

#[cfg(feature = "serde_json")]
use std::collections::HashMap;
#[cfg(feature = "serde_json")]
use std::time::SystemTime;
#[cfg(feature = "serde_json")]
use torsh_core::TorshError;

use super::super::core::PerformanceMeasurement;
use super::super::reporting::PerformanceReport;
use super::plot_data::PlotData;

/// Export utilities for performance data in various formats
///
/// The `PerformanceExporter` provides methods to export performance reports
/// and measurements to different formats suitable for analysis, visualization,
/// and integration with external tools.
pub struct PerformanceExporter;

impl PerformanceExporter {
    /// Export performance report to CSV format
    ///
    /// Creates a CSV representation of the performance report with operation
    /// statistics, timing information, and memory usage data.
    ///
    /// # Arguments
    ///
    /// * `report` - Performance report to export
    ///
    /// # Returns
    ///
    /// CSV string representation of the report
    pub fn to_csv(report: &PerformanceReport) -> TorshResult<String> {
        let mut csv = String::new();

        // Header
        csv.push_str(
            "operation,count,total_time_ms,min_time_ms,max_time_ms,avg_time_ms,avg_memory_bytes,ops_per_sec,consistency_score\n",
        );

        // Data rows
        for (operation, stats) in &report.operation_statistics {
            csv.push_str(&format!(
                "{},{},{:.3},{:.3},{:.3},{:.3},{:.1},{:.2},{:.3}\n",
                operation,
                stats.count,
                stats.total_time.as_secs_f64() * 1000.0,
                stats.min_time.as_secs_f64() * 1000.0,
                stats.max_time.as_secs_f64() * 1000.0,
                stats.avg_time().as_secs_f64() * 1000.0,
                stats.avg_memory,
                stats.operations_per_second(),
                1.0 - stats.timing_consistency() // Convert coefficient of variation to consistency score
            ));
        }

        Ok(csv)
    }

    /// Export performance report to JSON format
    ///
    /// Creates a structured JSON representation with detailed performance
    /// metrics, memory analysis, and metadata.
    #[cfg(feature = "serde_json")]
    pub fn to_json(report: &PerformanceReport) -> TorshResult<String> {
        use serde_json;

        #[derive(serde::Serialize)]
        struct JsonReport<'a> {
            total_measurements: usize,
            operation_count: usize,
            operations: Vec<JsonOperation<'a>>,
            #[serde(skip)]
            memory_analyses_count: usize,
            generated_at: SystemTime,
            metadata: &'a HashMap<String, String>,
        }

        #[derive(serde::Serialize)]
        struct JsonOperation<'a> {
            operation: &'a str,
            count: usize,
            total_time_ms: f64,
            min_time_ms: f64,
            max_time_ms: f64,
            avg_time_ms: f64,
            avg_memory: f64,
            ops_per_sec: f64,
            consistency_score: f64,
            memory_efficiency: f64,
        }

        let operations: Vec<JsonOperation> = report
            .operation_statistics
            .iter()
            .map(|(name, stats)| JsonOperation {
                operation: name,
                count: stats.count,
                total_time_ms: stats.total_time.as_secs_f64() * 1000.0,
                min_time_ms: stats.min_time.as_secs_f64() * 1000.0,
                max_time_ms: stats.max_time.as_secs_f64() * 1000.0,
                avg_time_ms: stats.avg_time().as_secs_f64() * 1000.0,
                avg_memory: stats.avg_memory,
                ops_per_sec: stats.operations_per_second(),
                consistency_score: 1.0 - stats.timing_consistency(),
                memory_efficiency: stats.memory_efficiency(),
            })
            .collect();

        let json_report = JsonReport {
            total_measurements: report.total_measurements,
            operation_count: report.operation_count,
            operations,
            memory_analyses_count: report.memory_analyses.len(),
            generated_at: report.generated_at,
            metadata: &report.metadata,
        };

        serde_json::to_string_pretty(&json_report)
            .map_err(|e| TorshError::InvalidArgument(format!("JSON serialization failed: {}", e)))
    }

    /// Export performance report to JSON format (fallback without serde)
    #[cfg(not(feature = "serde_json"))]
    pub fn to_json(report: &PerformanceReport) -> TorshResult<String> {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!(
            "  \"total_measurements\": {},\n",
            report.total_measurements
        ));
        json.push_str(&format!(
            "  \"operation_count\": {},\n",
            report.operation_count
        ));
        json.push_str("  \"operations\": [\n");

        let operations: Vec<String> = report
            .operation_statistics
            .iter()
            .map(|(name, stats)| {
                format!(
                    "    {{\n      \"operation\": \"{}\",\n      \"count\": {},\n      \"avg_time_ms\": {:.3},\n      \"ops_per_sec\": {:.2}\n    }}",
                    name,
                    stats.count,
                    stats.avg_time().as_secs_f64() * 1000.0,
                    stats.operations_per_second()
                )
            })
            .collect();

        json.push_str(&operations.join(",\n"));
        json.push_str("\n  ]\n");
        json.push_str("}\n");

        Ok(json)
    }

    /// Export performance measurements to CSV format
    pub fn measurements_to_csv(measurements: &[PerformanceMeasurement]) -> TorshResult<String> {
        let mut csv = String::new();

        // Header
        csv.push_str("operation,duration_ms,memory_before,memory_after,peak_memory,memory_delta\n");

        // Data rows
        for measurement in measurements {
            csv.push_str(&format!(
                "{},{:.3},{},{},{},{}\n",
                measurement.operation,
                measurement.duration.as_secs_f64() * 1000.0,
                measurement.memory_before,
                measurement.memory_after,
                measurement.peak_memory,
                measurement.memory_delta()
            ));
        }

        Ok(csv)
    }

    /// Generate plot-ready data from performance report
    pub fn generate_plot_data(report: &PerformanceReport) -> PlotData {
        let mut operation_names = Vec::new();
        let mut avg_times = Vec::new();
        let mut throughputs = Vec::new();
        let mut memory_usage = Vec::new();

        for (name, stats) in &report.operation_statistics {
            operation_names.push(name.clone());
            avg_times.push(stats.avg_time().as_secs_f64() * 1000.0);
            throughputs.push(stats.operations_per_second());
            memory_usage.push(stats.avg_memory);
        }

        PlotData {
            operation_names,
            avg_times,
            throughputs,
            memory_usage,
            timestamps: Vec::new(), // Not applicable for summary data
        }
    }
}
