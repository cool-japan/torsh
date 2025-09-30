//! # Reporting and Statistics Module
//!
//! This module provides comprehensive reporting and statistical analysis capabilities
//! for sparse tensor performance data. It aggregates operation statistics, generates
//! detailed reports, and provides analysis tools for performance optimization.
//!
//! ## Key Components
//!
//! - **OperationStatistics**: Detailed statistics for individual operations
//! - **PerformanceReport**: Comprehensive performance reports with multiple metrics
//! - **StatisticsCollector**: Utility for collecting and aggregating performance data
//! - **Report generation and analysis tools**: Various utilities for report creation and analysis
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_sparse::performance_tools::reporting::{OperationStatistics, PerformanceReport};
//!
//! // Create operation statistics
//! let stats = OperationStatistics::new("sparse_matmul".to_string());
//!
//! // Generate comprehensive report
//! let report = PerformanceReport::new();
//! let fastest_op = report.find_fastest_operation("matmul");
//! ```

use std::collections::HashMap;
use std::time::Duration;

use super::core::PerformanceMeasurement;
use super::memory::MemoryAnalysis;

/// Detailed statistics for a specific operation type
///
/// This struct aggregates multiple performance measurements for a single operation
/// type to provide comprehensive statistical analysis including timing, memory usage,
/// and performance trends.
#[derive(Debug, Clone)]
pub struct OperationStatistics {
    /// Operation name identifier
    pub operation: String,
    /// Total number of measurements
    pub count: usize,
    /// Total execution time across all measurements
    pub total_time: Duration,
    /// Minimum execution time observed
    pub min_time: Duration,
    /// Maximum execution time observed
    pub max_time: Duration,
    /// Average memory usage across measurements
    pub avg_memory: f64,
    /// Standard deviation of execution times
    pub time_std_dev: f64,
    /// Memory usage statistics
    pub memory_stats: MemoryStatistics,
    /// Custom operation metrics
    pub custom_metrics: HashMap<String, MetricStatistics>,
}

/// Memory usage statistics aggregation
#[derive(Debug, Clone, Default)]
pub struct MemoryStatistics {
    /// Average memory usage before operations
    pub avg_memory_before: f64,
    /// Average memory usage after operations
    pub avg_memory_after: f64,
    /// Average peak memory usage
    pub avg_peak_memory: f64,
    /// Average memory delta per operation
    pub avg_memory_delta: f64,
    /// Maximum memory delta observed
    pub max_memory_delta: i64,
    /// Minimum memory delta observed
    pub min_memory_delta: i64,
}

/// Statistics for custom metrics
#[derive(Debug, Clone)]
pub struct MetricStatistics {
    /// Metric name
    pub name: String,
    /// Number of data points
    pub count: usize,
    /// Sum of all values
    pub sum: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Standard deviation
    pub std_dev: f64,
}

impl OperationStatistics {
    /// Create new operation statistics for the given operation
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            count: 0,
            total_time: Duration::new(0, 0),
            min_time: Duration::MAX,
            max_time: Duration::new(0, 0),
            avg_memory: 0.0,
            time_std_dev: 0.0,
            memory_stats: MemoryStatistics::default(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Add a measurement to these statistics
    pub fn add_measurement(&mut self, measurement: &PerformanceMeasurement) {
        self.count += 1;
        self.total_time += measurement.duration;
        self.min_time = self.min_time.min(measurement.duration);
        self.max_time = self.max_time.max(measurement.duration);

        // Update memory statistics
        self.update_memory_stats(measurement);

        // Update custom metrics
        for (key, value) in &measurement.metrics {
            self.update_custom_metric(key.clone(), *value);
        }

        // Recalculate derived statistics
        self.recalculate_statistics();
    }

    /// Get average execution time
    pub fn avg_time(&self) -> Duration {
        if self.count > 0 {
            self.total_time / self.count as u32
        } else {
            Duration::new(0, 0)
        }
    }

    /// Get operations per second
    pub fn operations_per_second(&self) -> f64 {
        if self.total_time.as_secs_f64() > 0.0 {
            self.count as f64 / self.total_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get coefficient of variation for timing (std_dev / mean)
    pub fn timing_consistency(&self) -> f64 {
        let avg_time_ms = self.avg_time().as_secs_f64() * 1000.0;
        if avg_time_ms > 0.0 {
            self.time_std_dev / avg_time_ms
        } else {
            0.0
        }
    }

    /// Check if performance is consistent (low coefficient of variation)
    pub fn is_consistent(&self) -> bool {
        self.timing_consistency() < 0.2 // Less than 20% variation
    }

    /// Get memory efficiency score (0-1, higher is better)
    pub fn memory_efficiency(&self) -> f64 {
        if self.memory_stats.avg_peak_memory > self.memory_stats.avg_memory_after {
            self.memory_stats.avg_memory_after / self.memory_stats.avg_peak_memory
        } else {
            1.0
        }
    }

    /// Update memory statistics with new measurement
    fn update_memory_stats(&mut self, measurement: &PerformanceMeasurement) {
        let n = self.count as f64;
        let prev_n = (self.count - 1) as f64;

        // Running average updates
        self.memory_stats.avg_memory_before =
            (self.memory_stats.avg_memory_before * prev_n + measurement.memory_before as f64) / n;
        self.memory_stats.avg_memory_after =
            (self.memory_stats.avg_memory_after * prev_n + measurement.memory_after as f64) / n;
        self.memory_stats.avg_peak_memory =
            (self.memory_stats.avg_peak_memory * prev_n + measurement.peak_memory as f64) / n;

        let memory_delta = measurement.memory_delta();
        self.memory_stats.avg_memory_delta =
            (self.memory_stats.avg_memory_delta * prev_n + memory_delta as f64) / n;
        self.memory_stats.max_memory_delta = self.memory_stats.max_memory_delta.max(memory_delta);
        self.memory_stats.min_memory_delta = self.memory_stats.min_memory_delta.min(memory_delta);
    }

    /// Update custom metric statistics
    fn update_custom_metric(&mut self, metric_name: String, value: f64) {
        let metric_stats = self
            .custom_metrics
            .entry(metric_name.clone())
            .or_insert_with(|| MetricStatistics {
                name: metric_name,
                count: 0,
                sum: 0.0,
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                std_dev: 0.0,
            });

        metric_stats.count += 1;
        metric_stats.sum += value;
        metric_stats.min = metric_stats.min.min(value);
        metric_stats.max = metric_stats.max.max(value);
    }

    /// Recalculate derived statistics (standard deviations, etc.)
    fn recalculate_statistics(&mut self) {
        // This is a simplified calculation - in practice, you'd track
        // sum of squares for more accurate standard deviation calculation
        let time_range = self.max_time.as_secs_f64() - self.min_time.as_secs_f64();
        self.time_std_dev = time_range * 1000.0 / 4.0; // Rough approximation in milliseconds

        // Update custom metric standard deviations
        for metric_stats in self.custom_metrics.values_mut() {
            if metric_stats.count > 1 {
                let range = metric_stats.max - metric_stats.min;
                metric_stats.std_dev = range / 4.0; // Rough approximation
            }
        }
    }
}

impl MetricStatistics {
    /// Get average value
    pub fn average(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    /// Get range (max - min)
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Get coefficient of variation
    pub fn coefficient_of_variation(&self) -> f64 {
        let avg = self.average();
        if avg != 0.0 {
            self.std_dev / avg.abs()
        } else {
            0.0
        }
    }
}

/// Comprehensive performance report containing aggregated statistics
///
/// This struct provides a complete overview of performance across all operations,
/// including summaries, comparisons, and analysis capabilities.
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Total number of measurements across all operations
    pub total_measurements: usize,
    /// Number of unique operations measured
    pub operation_count: usize,
    /// Statistics for each operation
    pub operation_statistics: HashMap<String, OperationStatistics>,
    /// Overall memory analysis results
    pub memory_analyses: Vec<MemoryAnalysis>,
    /// Report generation timestamp
    pub generated_at: std::time::SystemTime,
    /// Additional report metadata
    pub metadata: HashMap<String, String>,
}

impl std::fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Sparse Tensor Performance Report ===")?;
        writeln!(f, "Generated at: {:?}", self.generated_at)?;
        writeln!(f, "Total measurements: {}", self.total_measurements)?;
        writeln!(f, "Unique operations: {}", self.operation_count)?;
        writeln!(f)?;

        writeln!(f, "Operation Statistics:")?;
        writeln!(
            f,
            "{:<30} {:<8} {:<12} {:<12} {:<12} {:<10}",
            "Operation", "Count", "Avg Time", "Min Time", "Max Time", "Ops/Sec"
        )?;
        writeln!(f, "{}", "-".repeat(90))?;

        for (operation, stats) in &self.operation_statistics {
            writeln!(
                f,
                "{:<30} {:<8} {:<12.3} {:<12.3} {:<12.3} {:<10.2}",
                operation,
                stats.count,
                stats.avg_time().as_secs_f64() * 1000.0,
                stats.min_time.as_secs_f64() * 1000.0,
                stats.max_time.as_secs_f64() * 1000.0,
                stats.operations_per_second()
            )?;
        }

        if !self.memory_analyses.is_empty() {
            writeln!(f)?;
            writeln!(f, "Memory Analysis Summary:")?;
            for analysis in &self.memory_analyses {
                writeln!(
                    f,
                    "  Format: {:?}, Compression: {:.1}x, Efficiency: {}",
                    analysis.format,
                    analysis.compression_ratio,
                    analysis.memory_efficiency_rating()
                )?;
            }
        }

        Ok(())
    }
}

impl PerformanceReport {
    /// Create a new empty performance report
    pub fn new() -> Self {
        Self {
            total_measurements: 0,
            operation_count: 0,
            operation_statistics: HashMap::new(),
            memory_analyses: Vec::new(),
            generated_at: std::time::SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add measurements to the report
    pub fn add_measurements(&mut self, measurements: &[PerformanceMeasurement]) {
        for measurement in measurements {
            self.add_measurement(measurement);
        }
    }

    /// Add a single measurement to the report
    pub fn add_measurement(&mut self, measurement: &PerformanceMeasurement) {
        let stats = self
            .operation_statistics
            .entry(measurement.operation.clone())
            .or_insert_with(|| OperationStatistics::new(measurement.operation.clone()));

        stats.add_measurement(measurement);
        self.total_measurements += 1;
        self.operation_count = self.operation_statistics.len();
    }

    /// Add memory analysis to the report
    pub fn add_memory_analysis(&mut self, analysis: MemoryAnalysis) {
        self.memory_analyses.push(analysis);
    }

    /// Find the fastest operation matching a pattern
    pub fn find_fastest_operation(&self, operation_pattern: &str) -> Option<&OperationStatistics> {
        self.operation_statistics
            .values()
            .filter(|stats| stats.operation.contains(operation_pattern))
            .min_by_key(|stats| stats.avg_time())
    }

    /// Find the most memory-efficient operation matching a pattern
    pub fn find_memory_efficient_operation(
        &self,
        operation_pattern: &str,
    ) -> Option<&OperationStatistics> {
        self.operation_statistics
            .values()
            .filter(|stats| stats.operation.contains(operation_pattern))
            .max_by(|a, b| {
                a.memory_efficiency()
                    .partial_cmp(&b.memory_efficiency())
                    .unwrap()
            })
    }

    /// Get top N operations by throughput (operations per second)
    pub fn top_operations_by_throughput(&self, n: usize) -> Vec<&OperationStatistics> {
        let mut operations: Vec<&OperationStatistics> =
            self.operation_statistics.values().collect();
        operations.sort_by(|a, b| {
            b.operations_per_second()
                .partial_cmp(&a.operations_per_second())
                .unwrap()
        });
        operations.into_iter().take(n).collect()
    }

    /// Get operations with inconsistent performance
    pub fn inconsistent_operations(&self) -> Vec<&OperationStatistics> {
        self.operation_statistics
            .values()
            .filter(|stats| !stats.is_consistent())
            .collect()
    }

    /// Get overall performance summary
    pub fn performance_summary(&self) -> PerformanceSummary {
        let total_time: Duration = self
            .operation_statistics
            .values()
            .map(|stats| stats.total_time)
            .sum();

        let avg_operations_per_second: f64 = self
            .operation_statistics
            .values()
            .map(|stats| stats.operations_per_second())
            .sum::<f64>()
            / self.operation_count.max(1) as f64;

        let memory_efficiency: f64 = self
            .operation_statistics
            .values()
            .map(|stats| stats.memory_efficiency())
            .sum::<f64>()
            / self.operation_count.max(1) as f64;

        PerformanceSummary {
            total_operations: self.total_measurements,
            total_time,
            avg_throughput: avg_operations_per_second,
            avg_memory_efficiency: memory_efficiency,
            consistency_score: self.calculate_consistency_score(),
        }
    }

    /// Add metadata to the report
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get recommendations based on performance analysis
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for inconsistent operations
        let inconsistent = self.inconsistent_operations();
        if !inconsistent.is_empty() {
            recommendations.push(format!(
                "Found {} operations with inconsistent performance - consider investigating: {}",
                inconsistent.len(),
                inconsistent
                    .iter()
                    .map(|op| op.operation.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        // Check for memory efficiency
        let inefficient_ops: Vec<&OperationStatistics> = self
            .operation_statistics
            .values()
            .filter(|stats| stats.memory_efficiency() < 0.7)
            .collect();

        if !inefficient_ops.is_empty() {
            recommendations.push(format!(
                "Found {} memory-inefficient operations - consider optimization",
                inefficient_ops.len()
            ));
        }

        // Check for slow operations
        let slow_ops: Vec<&OperationStatistics> = self
            .operation_statistics
            .values()
            .filter(|stats| stats.operations_per_second() < 100.0)
            .collect();

        if !slow_ops.is_empty() {
            recommendations.push(format!(
                "Found {} slow operations (< 100 ops/sec) - consider algorithmic improvements",
                slow_ops.len()
            ));
        }

        // Memory analysis recommendations
        for analysis in &self.memory_analyses {
            if !analysis.is_memory_efficient() {
                recommendations.push(format!(
                    "Format {:?} has poor compression ratio ({:.1}x) - consider alternative format",
                    analysis.format, analysis.compression_ratio
                ));
            }
        }

        if recommendations.is_empty() {
            recommendations.push("Performance appears optimal across all metrics".to_string());
        }

        recommendations
    }

    /// Calculate overall consistency score
    fn calculate_consistency_score(&self) -> f64 {
        if self.operation_statistics.is_empty() {
            return 1.0;
        }

        let consistent_count = self
            .operation_statistics
            .values()
            .filter(|stats| stats.is_consistent())
            .count();

        consistent_count as f64 / self.operation_statistics.len() as f64
    }
}

impl Default for PerformanceReport {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level performance summary
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Total number of operations measured
    pub total_operations: usize,
    /// Total execution time across all operations
    pub total_time: Duration,
    /// Average throughput across all operations
    pub avg_throughput: f64,
    /// Average memory efficiency score
    pub avg_memory_efficiency: f64,
    /// Consistency score (0-1, higher is better)
    pub consistency_score: f64,
}

impl PerformanceSummary {
    /// Get overall performance grade (A-F)
    pub fn performance_grade(&self) -> String {
        let score =
            (self.avg_throughput.log10() + self.avg_memory_efficiency + self.consistency_score)
                / 3.0;

        match score {
            s if s >= 0.9 => "A".to_string(),
            s if s >= 0.8 => "B".to_string(),
            s if s >= 0.7 => "C".to_string(),
            s if s >= 0.6 => "D".to_string(),
            _ => "F".to_string(),
        }
    }
}

/// Utility for collecting and aggregating performance statistics
#[derive(Debug)]
pub struct StatisticsCollector {
    /// Collected measurements
    measurements: Vec<PerformanceMeasurement>,
    /// Memory analyses
    memory_analyses: Vec<MemoryAnalysis>,
    /// Collection metadata
    metadata: HashMap<String, String>,
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticsCollector {
    /// Create a new statistics collector
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            memory_analyses: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a measurement to the collection
    pub fn add_measurement(&mut self, measurement: PerformanceMeasurement) {
        self.measurements.push(measurement);
    }

    /// Add multiple measurements
    pub fn add_measurements(&mut self, measurements: Vec<PerformanceMeasurement>) {
        self.measurements.extend(measurements);
    }

    /// Add a memory analysis
    pub fn add_memory_analysis(&mut self, analysis: MemoryAnalysis) {
        self.memory_analyses.push(analysis);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Generate a comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut report = PerformanceReport::new();

        // Add all measurements
        report.add_measurements(&self.measurements);

        // Add memory analyses
        for analysis in &self.memory_analyses {
            report.add_memory_analysis(analysis.clone());
        }

        // Add metadata
        for (key, value) in &self.metadata {
            report.add_metadata(key.clone(), value.clone());
        }

        report
    }

    /// Clear all collected data
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.memory_analyses.clear();
        self.metadata.clear();
    }

    /// Get number of collected measurements
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Get measurements for a specific operation
    pub fn get_measurements_for_operation(
        &self,
        operation_pattern: &str,
    ) -> Vec<&PerformanceMeasurement> {
        self.measurements
            .iter()
            .filter(|m| m.operation.contains(operation_pattern))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_measurement(operation: &str, duration_ms: u64) -> PerformanceMeasurement {
        let mut measurement = PerformanceMeasurement::new(operation.to_string());
        measurement.duration = Duration::from_millis(duration_ms);
        measurement.memory_before = 1000;
        measurement.memory_after = 1100;
        measurement.peak_memory = 1200;
        measurement
    }

    #[test]
    fn test_operation_statistics_creation() {
        let stats = OperationStatistics::new("test_operation".to_string());

        assert_eq!(stats.operation, "test_operation");
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total_time, Duration::new(0, 0));
        assert_eq!(stats.min_time, Duration::MAX);
        assert_eq!(stats.max_time, Duration::new(0, 0));
    }

    #[test]
    fn test_operation_statistics_add_measurement() {
        let mut stats = OperationStatistics::new("test".to_string());
        let measurement = create_test_measurement("test", 100);

        stats.add_measurement(&measurement);

        assert_eq!(stats.count, 1);
        assert_eq!(stats.avg_time(), Duration::from_millis(100));
        assert_eq!(stats.min_time, Duration::from_millis(100));
        assert_eq!(stats.max_time, Duration::from_millis(100));
    }

    #[test]
    fn test_operation_statistics_multiple_measurements() {
        let mut stats = OperationStatistics::new("test".to_string());

        stats.add_measurement(&create_test_measurement("test", 100));
        stats.add_measurement(&create_test_measurement("test", 200));
        stats.add_measurement(&create_test_measurement("test", 300));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.avg_time(), Duration::from_millis(200)); // (100+200+300)/3
        assert_eq!(stats.min_time, Duration::from_millis(100));
        assert_eq!(stats.max_time, Duration::from_millis(300));
        assert!(stats.operations_per_second() > 0.0);
    }

    #[test]
    fn test_operation_statistics_consistency() {
        let mut consistent_stats = OperationStatistics::new("consistent".to_string());
        // Add measurements with similar timing
        for _ in 0..5 {
            consistent_stats.add_measurement(&create_test_measurement("consistent", 100));
        }

        let mut inconsistent_stats = OperationStatistics::new("inconsistent".to_string());
        // Add measurements with varying timing
        inconsistent_stats.add_measurement(&create_test_measurement("inconsistent", 50));
        inconsistent_stats.add_measurement(&create_test_measurement("inconsistent", 200));
        inconsistent_stats.add_measurement(&create_test_measurement("inconsistent", 500));

        assert!(consistent_stats.is_consistent());
        assert!(!inconsistent_stats.is_consistent());
    }

    #[test]
    fn test_memory_statistics() {
        let mut stats = OperationStatistics::new("test".to_string());
        let measurement = create_test_measurement("test", 100);

        stats.add_measurement(&measurement);

        assert_eq!(stats.memory_stats.avg_memory_before, 1000.0);
        assert_eq!(stats.memory_stats.avg_memory_after, 1100.0);
        assert_eq!(stats.memory_stats.avg_peak_memory, 1200.0);
        assert_eq!(stats.memory_stats.avg_memory_delta, 100.0);
    }

    #[test]
    fn test_performance_report_creation() {
        let report = PerformanceReport::new();

        assert_eq!(report.total_measurements, 0);
        assert_eq!(report.operation_count, 0);
        assert!(report.operation_statistics.is_empty());
        assert!(report.memory_analyses.is_empty());
    }

    #[test]
    fn test_performance_report_add_measurements() {
        let mut report = PerformanceReport::new();
        let measurements = vec![
            create_test_measurement("op1", 100),
            create_test_measurement("op2", 200),
            create_test_measurement("op1", 150),
        ];

        report.add_measurements(&measurements);

        assert_eq!(report.total_measurements, 3);
        assert_eq!(report.operation_count, 2);
        assert!(report.operation_statistics.contains_key("op1"));
        assert!(report.operation_statistics.contains_key("op2"));

        let op1_stats = &report.operation_statistics["op1"];
        assert_eq!(op1_stats.count, 2);
        assert_eq!(op1_stats.avg_time(), Duration::from_millis(125)); // (100+150)/2
    }

    #[test]
    fn test_performance_report_find_operations() {
        let mut report = PerformanceReport::new();
        report.add_measurement(&create_test_measurement("fast_operation", 50));
        report.add_measurement(&create_test_measurement("slow_operation", 500));
        report.add_measurement(&create_test_measurement("medium_operation", 200));

        let fastest = report.find_fastest_operation("operation");
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().operation, "fast_operation");

        let top_ops = report.top_operations_by_throughput(2);
        assert_eq!(top_ops.len(), 2);
        // fastest operation should have highest throughput
        assert_eq!(top_ops[0].operation, "fast_operation");
    }

    #[test]
    fn test_performance_report_recommendations() {
        let mut report = PerformanceReport::new();

        // Add inconsistent operation
        report.add_measurement(&create_test_measurement("inconsistent_op", 50));
        report.add_measurement(&create_test_measurement("inconsistent_op", 500));

        // Add slow operation
        let mut slow_measurement = create_test_measurement("slow_op", 10000);
        slow_measurement.add_metric("custom_metric".to_string(), 1.0);
        report.add_measurement(&slow_measurement);

        let recommendations = report.get_recommendations();
        assert!(!recommendations.is_empty());

        // Should have recommendations about inconsistent and slow operations
        let rec_text = recommendations.join(" ");
        assert!(rec_text.contains("inconsistent") || rec_text.contains("slow"));
    }

    #[test]
    fn test_performance_summary() {
        let mut report = PerformanceReport::new();
        report.add_measurement(&create_test_measurement("op1", 100));
        report.add_measurement(&create_test_measurement("op2", 200));

        let summary = report.performance_summary();
        assert_eq!(summary.total_operations, 2);
        assert!(summary.avg_throughput > 0.0);
        assert!(summary.avg_memory_efficiency >= 0.0 && summary.avg_memory_efficiency <= 1.0);
        assert!(summary.consistency_score >= 0.0 && summary.consistency_score <= 1.0);

        let grade = summary.performance_grade();
        assert!(["A", "B", "C", "D", "F"].contains(&grade.as_str()));
    }

    #[test]
    fn test_statistics_collector() {
        let mut collector = StatisticsCollector::new();

        collector.add_measurement(create_test_measurement("op1", 100));
        collector.add_measurement(create_test_measurement("op2", 200));
        collector.add_metadata("test_key".to_string(), "test_value".to_string());

        assert_eq!(collector.measurement_count(), 2);

        let report = collector.generate_report();
        assert_eq!(report.total_measurements, 2);
        assert_eq!(report.operation_count, 2);
        assert!(report.metadata.contains_key("test_key"));

        collector.clear();
        assert_eq!(collector.measurement_count(), 0);
    }

    #[test]
    fn test_metric_statistics() {
        let mut metric_stats = MetricStatistics {
            name: "test_metric".to_string(),
            count: 3,
            sum: 15.0,
            min: 2.0,
            max: 8.0,
            std_dev: 2.5,
        };

        assert_eq!(metric_stats.average(), 5.0);
        assert_eq!(metric_stats.range(), 6.0);
        assert_eq!(metric_stats.coefficient_of_variation(), 0.5); // 2.5 / 5.0
    }

    #[test]
    fn test_performance_report_display() {
        let mut report = PerformanceReport::new();
        report.add_measurement(&create_test_measurement("test_op", 100));

        let display_string = format!("{}", report);
        assert!(display_string.contains("Sparse Tensor Performance Report"));
        assert!(display_string.contains("test_op"));
        assert!(display_string.contains("Total measurements: 1"));
    }

    #[test]
    fn test_memory_statistics_defaults() {
        let memory_stats = MemoryStatistics::default();
        assert_eq!(memory_stats.avg_memory_before, 0.0);
        assert_eq!(memory_stats.avg_memory_after, 0.0);
        assert_eq!(memory_stats.avg_peak_memory, 0.0);
        assert_eq!(memory_stats.avg_memory_delta, 0.0);
        assert_eq!(memory_stats.max_memory_delta, 0);
        assert_eq!(memory_stats.min_memory_delta, 0);
    }
}
