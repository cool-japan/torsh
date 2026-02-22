//! Performance Monitoring and Metrics for FX Graph Execution
//!
//! This module provides comprehensive performance monitoring capabilities for FX graph execution.
//! It includes execution metrics collection, performance analysis, and reporting functionality.

use std::collections::HashMap;

/// Performance monitoring for graph execution
///
/// Collects and analyzes performance metrics during graph execution, including
/// execution times, operation counts, and memory usage statistics.
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// Total execution time in milliseconds
    pub total_time_ms: f64,
    /// Time spent on each operation type
    pub operation_times: HashMap<String, f64>,
    /// Number of operations executed
    pub operation_count: usize,
    /// Memory usage during execution
    pub peak_memory_mb: f64,
}

impl ExecutionMetrics {
    /// Create new execution metrics
    ///
    /// # Returns
    /// * `Self` - New empty execution metrics instance
    pub fn new() -> Self {
        Self {
            total_time_ms: 0.0,
            operation_times: HashMap::new(),
            operation_count: 0,
            peak_memory_mb: 0.0,
        }
    }

    /// Add execution time for an operation
    ///
    /// Records the execution time for a specific operation type and updates
    /// the total operation count.
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation
    /// * `time_ms` - Execution time in milliseconds
    pub fn add_operation_time(&mut self, op_name: &str, time_ms: f64) {
        *self
            .operation_times
            .entry(op_name.to_string())
            .or_insert(0.0) += time_ms;
        self.operation_count += 1;
    }

    /// Set total execution time
    ///
    /// # Arguments
    /// * `time_ms` - Total execution time in milliseconds
    pub fn set_total_time(&mut self, time_ms: f64) {
        self.total_time_ms = time_ms;
    }

    /// Set peak memory usage
    ///
    /// # Arguments
    /// * `memory_mb` - Peak memory usage in megabytes
    pub fn set_peak_memory(&mut self, memory_mb: f64) {
        self.peak_memory_mb = memory_mb;
    }

    /// Get average execution time per operation
    ///
    /// # Returns
    /// * `f64` - Average time per operation in milliseconds
    pub fn average_time_per_op(&self) -> f64 {
        if self.operation_count > 0 {
            self.total_time_ms / self.operation_count as f64
        } else {
            0.0
        }
    }

    /// Get the slowest operation type
    ///
    /// # Returns
    /// * `Option<(&String, &f64)>` - Operation name and time for slowest operation
    pub fn slowest_operation(&self) -> Option<(&String, &f64)> {
        self.operation_times
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the fastest operation type
    ///
    /// # Returns
    /// * `Option<(&String, &f64)>` - Operation name and time for fastest operation
    pub fn fastest_operation(&self) -> Option<(&String, &f64)> {
        self.operation_times
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get total time for a specific operation type
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation
    ///
    /// # Returns
    /// * `f64` - Total time spent on this operation type
    pub fn get_operation_time(&self, op_name: &str) -> f64 {
        self.operation_times.get(op_name).copied().unwrap_or(0.0)
    }

    /// Get percentage of total time spent on a specific operation
    ///
    /// # Arguments
    /// * `op_name` - Name of the operation
    ///
    /// # Returns
    /// * `f64` - Percentage of total time (0.0 to 100.0)
    pub fn get_operation_percentage(&self, op_name: &str) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.get_operation_time(op_name) / self.total_time_ms) * 100.0
        } else {
            0.0
        }
    }

    /// Get number of unique operation types
    ///
    /// # Returns
    /// * `usize` - Number of different operation types executed
    pub fn operation_type_count(&self) -> usize {
        self.operation_times.len()
    }

    /// Check if metrics collection is empty
    ///
    /// # Returns
    /// * `bool` - True if no metrics have been collected
    pub fn is_empty(&self) -> bool {
        self.operation_count == 0 && self.total_time_ms == 0.0
    }

    /// Merge metrics from another instance
    ///
    /// Combines metrics from another ExecutionMetrics instance, useful for
    /// accumulating metrics across multiple execution runs.
    ///
    /// # Arguments
    /// * `other` - Other metrics instance to merge
    pub fn merge(&mut self, other: &ExecutionMetrics) {
        self.total_time_ms += other.total_time_ms;
        self.operation_count += other.operation_count;
        self.peak_memory_mb = self.peak_memory_mb.max(other.peak_memory_mb);

        for (op_name, time) in &other.operation_times {
            *self.operation_times.entry(op_name.clone()).or_insert(0.0) += time;
        }
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.total_time_ms = 0.0;
        self.operation_times.clear();
        self.operation_count = 0;
        self.peak_memory_mb = 0.0;
    }

    /// Generate a comprehensive performance report
    ///
    /// Creates a detailed text report of all collected performance metrics,
    /// including timing breakdowns and operation analysis.
    ///
    /// # Returns
    /// * `String` - Formatted performance report
    pub fn generate_report(&self) -> String {
        let mut report = format!(
            "Execution Performance Report:\n\
             Total Time: {:.2} ms\n\
             Operations Executed: {}\n\
             Average Time/Op: {:.2} ms\n\
             Peak Memory: {:.2} MB\n\
             Operation Types: {}\n\n\
             Operation Breakdown:",
            self.total_time_ms,
            self.operation_count,
            self.average_time_per_op(),
            self.peak_memory_mb,
            self.operation_type_count()
        );

        let mut sorted_ops: Vec<_> = self.operation_times.iter().collect();
        sorted_ops.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (op_name, time) in sorted_ops {
            let percentage = self.get_operation_percentage(op_name);
            report.push_str(&format!(
                "\n  {}: {:.2} ms ({:.1}%)",
                op_name, time, percentage
            ));
        }

        // Add performance insights
        if let Some((slowest_op, slowest_time)) = self.slowest_operation() {
            report.push_str(&format!(
                "\n\nPerformance Insights:\n\
                 Slowest Operation: {} ({:.2} ms)\n",
                slowest_op, slowest_time
            ));
        }

        if self.peak_memory_mb > 1000.0 {
            report.push_str(&format!(
                " Memory Usage: High ({:.2} MB)\n",
                self.peak_memory_mb
            ));
        }

        if self.average_time_per_op() > 100.0 {
            report.push_str(" Average Operation Time: High (>100ms per operation)\n");
        }

        report
    }

    /// Generate a compact summary report
    ///
    /// Creates a brief summary of the most important performance metrics.
    ///
    /// # Returns
    /// * `String` - Compact performance summary
    pub fn generate_summary(&self) -> String {
        format!(
            "Performance Summary: {:.2}ms total, {} ops, {:.2}ms/op avg, {:.2}MB peak",
            self.total_time_ms,
            self.operation_count,
            self.average_time_per_op(),
            self.peak_memory_mb
        )
    }

    /// Generate JSON representation of metrics
    ///
    /// Creates a JSON string containing all metrics data, useful for
    /// integration with external monitoring systems.
    ///
    /// # Returns
    /// * `String` - JSON representation of metrics
    pub fn to_json(&self) -> String {
        let mut json = format!(
            r#"{{
  "total_time_ms": {},
  "operation_count": {},
  "peak_memory_mb": {},
  "average_time_per_op": {},
  "operation_times": {{"#,
            self.total_time_ms,
            self.operation_count,
            self.peak_memory_mb,
            self.average_time_per_op()
        );

        let op_entries: Vec<String> = self
            .operation_times
            .iter()
            .map(|(name, time)| format!(r#"    "{}": {}"#, name, time))
            .collect();

        json.push_str(&op_entries.join(",\n"));
        json.push_str("\n  }\n}");

        json
    }
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer utility for measuring execution time
///
/// Provides convenient timing functionality for measuring operation execution times.
pub struct ExecutionTimer {
    start_time: std::time::Instant,
}

impl ExecutionTimer {
    /// Start a new timer
    ///
    /// # Returns
    /// * `Self` - New timer instance started at current time
    pub fn start() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }

    /// Get elapsed time in milliseconds
    ///
    /// # Returns
    /// * `f64` - Elapsed time since timer start in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() * 1000.0
    }

    /// Reset the timer to current time
    pub fn reset(&mut self) {
        self.start_time = std::time::Instant::now();
    }
}

/// Metrics collector for tracking multiple execution runs
///
/// Aggregates metrics across multiple graph execution runs, providing
/// statistical analysis and trend detection.
pub struct MetricsCollector {
    runs: Vec<ExecutionMetrics>,
    total_metrics: ExecutionMetrics,
}

impl MetricsCollector {
    /// Create a new metrics collector
    ///
    /// # Returns
    /// * `Self` - New empty metrics collector
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            total_metrics: ExecutionMetrics::new(),
        }
    }

    /// Add metrics from a single execution run
    ///
    /// # Arguments
    /// * `metrics` - Execution metrics from a single run
    pub fn add_run(&mut self, metrics: ExecutionMetrics) {
        self.total_metrics.merge(&metrics);
        self.runs.push(metrics);
    }

    /// Get total aggregated metrics
    ///
    /// # Returns
    /// * `&ExecutionMetrics` - Reference to aggregated metrics across all runs
    pub fn total_metrics(&self) -> &ExecutionMetrics {
        &self.total_metrics
    }

    /// Get metrics for a specific run
    ///
    /// # Arguments
    /// * `run_index` - Index of the run to retrieve
    ///
    /// # Returns
    /// * `Option<&ExecutionMetrics>` - Metrics for the specified run if it exists
    pub fn get_run(&self, run_index: usize) -> Option<&ExecutionMetrics> {
        self.runs.get(run_index)
    }

    /// Get number of recorded runs
    ///
    /// # Returns
    /// * `usize` - Number of execution runs recorded
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Calculate average execution time across all runs
    ///
    /// # Returns
    /// * `f64` - Average total execution time in milliseconds
    pub fn average_execution_time(&self) -> f64 {
        if self.runs.is_empty() {
            0.0
        } else {
            self.runs.iter().map(|r| r.total_time_ms).sum::<f64>() / self.runs.len() as f64
        }
    }

    /// Calculate execution time variance
    ///
    /// # Returns
    /// * `f64` - Variance in execution times across runs
    pub fn execution_time_variance(&self) -> f64 {
        if self.runs.len() <= 1 {
            return 0.0;
        }

        let avg = self.average_execution_time();
        let variance = self
            .runs
            .iter()
            .map(|r| (r.total_time_ms - avg).powi(2))
            .sum::<f64>()
            / (self.runs.len() - 1) as f64;

        variance
    }

    /// Get the fastest execution time
    ///
    /// # Returns
    /// * `Option<f64>` - Fastest execution time in milliseconds
    pub fn fastest_execution(&self) -> Option<f64> {
        self.runs
            .iter()
            .map(|r| r.total_time_ms)
            .fold(None, |acc, time| match acc {
                None => Some(time),
                Some(min_time) => Some(min_time.min(time)),
            })
    }

    /// Get the slowest execution time
    ///
    /// # Returns
    /// * `Option<f64>` - Slowest execution time in milliseconds
    pub fn slowest_execution(&self) -> Option<f64> {
        self.runs
            .iter()
            .map(|r| r.total_time_ms)
            .fold(None, |acc, time| match acc {
                None => Some(time),
                Some(max_time) => Some(max_time.max(time)),
            })
    }

    /// Clear all collected metrics
    pub fn clear(&mut self) {
        self.runs.clear();
        self.total_metrics.clear();
    }

    /// Generate statistical report across all runs
    ///
    /// # Returns
    /// * `String` - Statistical analysis report
    pub fn generate_statistical_report(&self) -> String {
        if self.runs.is_empty() {
            return "No execution runs recorded".to_string();
        }

        format!(
            "Statistical Performance Report:\n\
             Total Runs: {}\n\
             Average Execution Time: {:.2} ms\n\
             Fastest Execution: {:.2} ms\n\
             Slowest Execution: {:.2} ms\n\
             Execution Time Variance: {:.2}\n\
             Standard Deviation: {:.2} ms\n\n\
             {}",
            self.run_count(),
            self.average_execution_time(),
            self.fastest_execution().unwrap_or(0.0),
            self.slowest_execution().unwrap_or(0.0),
            self.execution_time_variance(),
            self.execution_time_variance().sqrt(),
            self.total_metrics.generate_report()
        )
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}
