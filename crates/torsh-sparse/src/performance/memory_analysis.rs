//! Memory analysis and reporting tools for sparse tensor operations
//!
//! This module provides comprehensive memory usage analysis, operation statistics tracking,
//! and performance reporting capabilities for sparse tensor systems.

use crate::SparseFormat;
use super::core::PerformanceMeasurement;
use std::collections::HashMap;
use std::time::Duration;

/// Detailed memory analysis for sparse tensors
///
/// MemoryAnalysis provides comprehensive information about memory usage patterns
/// for different sparse formats, including compression ratios and overhead calculations.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::MemoryAnalysis;
/// use torsh_sparse::SparseFormat;
///
/// let analysis = MemoryAnalysis {
///     format: SparseFormat::Csr,
///     nnz: 1000,
///     estimated_memory: 12000,
///     dense_memory: 1000000,
///     compression_ratio: 83.33,
///     overhead_per_nnz: 12.0,
///     matrix_dimensions: (1000, 1000),
/// };
///
/// println!("Compression ratio: {:.2}x", analysis.compression_ratio);
/// println!("Memory overhead per NNZ: {:.1} bytes", analysis.overhead_per_nnz);
/// ```
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    /// Sparse format being analyzed
    pub format: SparseFormat,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Estimated memory usage (bytes)
    pub estimated_memory: usize,
    /// Memory usage if stored as dense (bytes)
    pub dense_memory: usize,
    /// Compression ratio (dense/sparse)
    pub compression_ratio: f32,
    /// Memory overhead per non-zero element
    pub overhead_per_nnz: f32,
    /// Matrix dimensions (rows, columns)
    pub matrix_dimensions: (usize, usize),
}

impl MemoryAnalysis {
    /// Create a new memory analysis
    pub fn new(
        format: SparseFormat,
        nnz: usize,
        matrix_dimensions: (usize, usize),
    ) -> Self {
        let (rows, cols) = matrix_dimensions;
        let dense_memory = rows * cols * std::mem::size_of::<f32>();

        let estimated_memory = Self::calculate_sparse_memory(format, nnz, matrix_dimensions);
        let compression_ratio = if estimated_memory > 0 {
            dense_memory as f32 / estimated_memory as f32
        } else {
            1.0
        };
        let overhead_per_nnz = if nnz > 0 {
            estimated_memory as f32 / nnz as f32
        } else {
            0.0
        };

        Self {
            format,
            nnz,
            estimated_memory,
            dense_memory,
            compression_ratio,
            overhead_per_nnz,
            matrix_dimensions,
        }
    }

    /// Calculate memory usage for a specific sparse format
    pub fn calculate_sparse_memory(
        format: SparseFormat,
        nnz: usize,
        (rows, cols): (usize, usize),
    ) -> usize {
        match format {
            SparseFormat::Coo => {
                // Row indices + column indices + values (all as 32-bit)
                nnz * (4 + 4 + 4)
            }
            SparseFormat::Csr => {
                // Values + column indices + row pointers
                nnz * 8 + (rows + 1) * 4
            }
            SparseFormat::Csc => {
                // Values + row indices + column pointers
                nnz * 8 + (cols + 1) * 4
            }
            _ => nnz * 12, // Default estimate
        }
    }

    /// Get memory efficiency (0.0 to 1.0, higher is better)
    pub fn memory_efficiency(&self) -> f32 {
        if self.dense_memory == 0 {
            1.0
        } else {
            1.0 - (self.estimated_memory as f32 / self.dense_memory as f32)
        }
    }

    /// Get sparsity level (0.0 to 1.0)
    pub fn sparsity_level(&self) -> f32 {
        let total_elements = self.matrix_dimensions.0 * self.matrix_dimensions.1;
        if total_elements == 0 {
            0.0
        } else {
            1.0 - (self.nnz as f32 / total_elements as f32)
        }
    }

    /// Check if this format provides good compression
    pub fn is_efficiently_compressed(&self) -> bool {
        self.compression_ratio > 2.0 && self.memory_efficiency() > 0.5
    }

    /// Compare with another memory analysis
    pub fn compare_with(&self, other: &MemoryAnalysis) -> MemoryComparison {
        MemoryComparison {
            format_a: self.format,
            format_b: other.format,
            memory_ratio: self.estimated_memory as f32 / other.estimated_memory as f32,
            compression_ratio_diff: self.compression_ratio - other.compression_ratio,
            overhead_diff: self.overhead_per_nnz - other.overhead_per_nnz,
            better_format: if self.estimated_memory < other.estimated_memory {
                self.format
            } else {
                other.format
            },
        }
    }
}

/// Comparison between two memory analyses
#[derive(Debug, Clone)]
pub struct MemoryComparison {
    pub format_a: SparseFormat,
    pub format_b: SparseFormat,
    pub memory_ratio: f32,
    pub compression_ratio_diff: f32,
    pub overhead_diff: f32,
    pub better_format: SparseFormat,
}

/// Statistics for a specific operation type
///
/// OperationStatistics aggregates performance data across multiple runs
/// of the same operation, providing statistical summaries and insights.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::OperationStatistics;
/// use std::time::Duration;
///
/// let stats = OperationStatistics {
///     operation: "sparse_matmul_csr_csr".to_string(),
///     count: 10,
///     total_time: Duration::from_millis(500),
///     min_time: Duration::from_millis(45),
///     max_time: Duration::from_millis(55),
///     avg_memory: 1024.0 * 1024.0,
/// };
///
/// println!("Average time: {:?}", stats.average_time());
/// println!("Time std dev: {:?}", stats.time_variance());
/// ```
#[derive(Debug, Clone)]
pub struct OperationStatistics {
    /// Operation name
    pub operation: String,
    /// Number of measurements
    pub count: usize,
    /// Total execution time across all measurements
    pub total_time: Duration,
    /// Minimum execution time
    pub min_time: Duration,
    /// Maximum execution time
    pub max_time: Duration,
    /// Average memory usage
    pub avg_memory: f64,
}

impl OperationStatistics {
    /// Create new operation statistics
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            count: 0,
            total_time: Duration::new(0, 0),
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::new(0, 0),
            avg_memory: 0.0,
        }
    }

    /// Add a measurement to the statistics
    pub fn add_measurement(&mut self, measurement: &PerformanceMeasurement) {
        self.count += 1;
        self.total_time += measurement.duration;
        self.min_time = std::cmp::min(self.min_time, measurement.duration);
        self.max_time = std::cmp::max(self.max_time, measurement.duration);

        // Update running average for memory
        let prev_avg = self.avg_memory;
        self.avg_memory = prev_avg + (measurement.peak_memory as f64 - prev_avg) / self.count as f64;
    }

    /// Get average execution time
    pub fn average_time(&self) -> Duration {
        if self.count == 0 {
            Duration::new(0, 0)
        } else {
            self.total_time / self.count as u32
        }
    }

    /// Get time variance (max - min)
    pub fn time_variance(&self) -> Duration {
        if self.count == 0 {
            Duration::new(0, 0)
        } else {
            self.max_time - self.min_time
        }
    }

    /// Get operations per second (if FLOPS data is available)
    pub fn operations_per_second(&self) -> Option<f64> {
        // This would require FLOPS data from measurements
        // For now, return None - could be enhanced with measurement analysis
        None
    }

    /// Check if this operation is consistent (low variance)
    pub fn is_consistent(&self) -> bool {
        if self.count < 2 {
            return true;
        }

        let avg = self.average_time();
        let variance_ratio = self.time_variance().as_nanos() as f64 / avg.as_nanos() as f64;
        variance_ratio < 0.1 // Less than 10% variance
    }
}

/// Comprehensive performance report
///
/// PerformanceReport aggregates all performance measurements and provides
/// analysis tools for understanding operation performance characteristics.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::{PerformanceReport, OperationStatistics};
/// use std::collections::HashMap;
///
/// let mut operation_stats = HashMap::new();
/// operation_stats.insert("conversion_coo_to_csr".to_string(),
///     OperationStatistics::new("conversion_coo_to_csr".to_string()));
///
/// let report = PerformanceReport {
///     total_measurements: 10,
///     operation_statistics: operation_stats,
///     measurements: vec![],
/// };
///
/// println!("{}", report);
/// ```
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Total number of measurements collected
    pub total_measurements: usize,
    /// Statistics grouped by operation type
    pub operation_statistics: HashMap<String, OperationStatistics>,
    /// All individual measurements for detailed analysis
    pub measurements: Vec<PerformanceMeasurement>,
}

impl std::fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Sparse Tensor Performance Report ===")?;
        writeln!(f)?;
        writeln!(f, "Total measurements: {}", self.total_measurements)?;
        writeln!(f, "Operation types: {}", self.operation_statistics.len())?;
        writeln!(f)?;

        for (operation, stats) in &self.operation_statistics {
            writeln!(f, "Operation: {operation}")?;
            writeln!(f, "  Count: {}", stats.count)?;
            writeln!(f, "  Total time: {:?}", stats.total_time)?;
            writeln!(f, "  Min time: {:?}", stats.min_time)?;
            writeln!(f, "  Max time: {:?}", stats.max_time)?;
            writeln!(f, "  Avg time: {:?}", stats.average_time())?;
            writeln!(f, "  Avg memory: {:.1} bytes", stats.avg_memory)?;
            writeln!(f, "  Consistent: {}", stats.is_consistent())?;
            writeln!(f)?;
        }

        Ok(())
    }
}

impl PerformanceReport {
    /// Create a new empty performance report
    pub fn new() -> Self {
        Self {
            total_measurements: 0,
            operation_statistics: HashMap::new(),
            measurements: Vec::new(),
        }
    }

    /// Add a measurement to the report
    pub fn add_measurement(&mut self, measurement: PerformanceMeasurement) {
        self.total_measurements += 1;

        let stats = self.operation_statistics
            .entry(measurement.operation.clone())
            .or_insert_with(|| OperationStatistics::new(measurement.operation.clone()));

        stats.add_measurement(&measurement);
        self.measurements.push(measurement);
    }

    /// Find the fastest operation for a given pattern
    pub fn find_fastest_operation(&self, operation_pattern: &str) -> Option<&OperationStatistics> {
        self.operation_statistics
            .values()
            .filter(|stats| stats.operation.contains(operation_pattern))
            .min_by_key(|stats| stats.average_time())
    }

    /// Find the most memory-efficient operation for a given pattern
    pub fn find_memory_efficient_operation(
        &self,
        operation_pattern: &str,
    ) -> Option<&OperationStatistics> {
        self.operation_statistics
            .values()
            .filter(|stats| stats.operation.contains(operation_pattern))
            .min_by(|a, b| a.avg_memory.partial_cmp(&b.avg_memory).expect("memory comparison should succeed"))
    }

    /// Get all operations matching a pattern
    pub fn get_operations_by_pattern(&self, pattern: &str) -> Vec<&OperationStatistics> {
        self.operation_statistics
            .values()
            .filter(|stats| stats.operation.contains(pattern))
            .collect()
    }

    /// Get summary statistics across all operations
    pub fn get_summary(&self) -> ReportSummary {
        if self.operation_statistics.is_empty() {
            return ReportSummary::default();
        }

        let total_time: Duration = self.operation_statistics
            .values()
            .map(|stats| stats.total_time)
            .sum();

        let avg_memory: f64 = self.operation_statistics
            .values()
            .map(|stats| stats.avg_memory)
            .sum::<f64>() / self.operation_statistics.len() as f64;

        let fastest_op = self.operation_statistics
            .values()
            .min_by_key(|stats| stats.average_time())
            .map(|stats| stats.operation.clone());

        let slowest_op = self.operation_statistics
            .values()
            .max_by_key(|stats| stats.average_time())
            .map(|stats| stats.operation.clone());

        ReportSummary {
            total_operations: self.operation_statistics.len(),
            total_time,
            average_memory: avg_memory,
            fastest_operation: fastest_op,
            slowest_operation: slowest_op,
        }
    }

    /// Export measurements for external analysis
    pub fn export_measurements(&self) -> &[PerformanceMeasurement] {
        &self.measurements
    }

    /// Clear all data from the report
    pub fn clear(&mut self) {
        self.total_measurements = 0;
        self.operation_statistics.clear();
        self.measurements.clear();
    }
}

impl Default for PerformanceReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for a performance report
#[derive(Debug, Clone, Default)]
pub struct ReportSummary {
    pub total_operations: usize,
    pub total_time: Duration,
    pub average_memory: f64,
    pub fastest_operation: Option<String>,
    pub slowest_operation: Option<String>,
}

impl std::fmt::Display for ReportSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Performance Report Summary ===")?;
        writeln!(f, "Total operation types: {}", self.total_operations)?;
        writeln!(f, "Total execution time: {:?}", self.total_time)?;
        writeln!(f, "Average memory usage: {:.1} bytes", self.average_memory)?;

        if let Some(ref fastest) = self.fastest_operation {
            writeln!(f, "Fastest operation: {}", fastest)?;
        }

        if let Some(ref slowest) = self.slowest_operation {
            writeln!(f, "Slowest operation: {}", slowest)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_memory_analysis_creation() {
        let analysis = MemoryAnalysis::new(SparseFormat::Csr, 1000, (100, 100));

        assert_eq!(analysis.format, SparseFormat::Csr);
        assert_eq!(analysis.nnz, 1000);
        assert_eq!(analysis.matrix_dimensions, (100, 100));
        assert!(analysis.compression_ratio > 1.0);
        assert!(analysis.overhead_per_nnz > 0.0);
    }

    #[test]
    fn test_memory_analysis_calculations() {
        let analysis = MemoryAnalysis::new(SparseFormat::Coo, 500, (1000, 1000));

        // COO format: 12 bytes per NNZ
        assert_eq!(analysis.estimated_memory, 500 * 12);

        // Dense: 1000 * 1000 * 4 = 4MB
        assert_eq!(analysis.dense_memory, 1000 * 1000 * 4);

        // Should have high compression ratio
        assert!(analysis.compression_ratio > 500.0);
        assert!(analysis.is_efficiently_compressed());
    }

    #[test]
    fn test_operation_statistics() {
        let mut stats = OperationStatistics::new("test_op".to_string());

        let measurement = PerformanceMeasurement {
            operation: "test_op".to_string(),
            duration: Duration::from_millis(100),
            memory_before: 1000,
            memory_after: 2000,
            peak_memory: 2500,
            metrics: HashMap::new(),
        };

        stats.add_measurement(&measurement);

        assert_eq!(stats.count, 1);
        assert_eq!(stats.average_time(), Duration::from_millis(100));
        assert_eq!(stats.avg_memory, 2500.0);
        assert!(stats.is_consistent());
    }

    #[test]
    fn test_performance_report() {
        let mut report = PerformanceReport::new();

        let measurement = PerformanceMeasurement {
            operation: "conversion_coo_to_csr".to_string(),
            duration: Duration::from_millis(50),
            memory_before: 1000,
            memory_after: 1500,
            peak_memory: 2000,
            metrics: HashMap::new(),
        };

        report.add_measurement(measurement);

        assert_eq!(report.total_measurements, 1);
        assert_eq!(report.operation_statistics.len(), 1);

        let fastest = report.find_fastest_operation("conversion");
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().operation, "conversion_coo_to_csr");
    }

    #[test]
    fn test_report_summary() {
        let mut report = PerformanceReport::new();

        // Add multiple measurements
        for i in 0..3 {
            let measurement = PerformanceMeasurement {
                operation: format!("test_op_{}", i),
                duration: Duration::from_millis(100 + i as u64 * 10),
                memory_before: 1000,
                memory_after: 1500,
                peak_memory: 2000,
                metrics: HashMap::new(),
            };
            report.add_measurement(measurement);
        }

        let summary = report.get_summary();
        assert_eq!(summary.total_operations, 3);
        assert_eq!(summary.fastest_operation, Some("test_op_0".to_string()));
        assert_eq!(summary.slowest_operation, Some("test_op_2".to_string()));
    }

    #[test]
    fn test_memory_comparison() {
        let coo_analysis = MemoryAnalysis::new(SparseFormat::Coo, 1000, (100, 100));
        let csr_analysis = MemoryAnalysis::new(SparseFormat::Csr, 1000, (100, 100));

        let comparison = coo_analysis.compare_with(&csr_analysis);

        assert_eq!(comparison.format_a, SparseFormat::Coo);
        assert_eq!(comparison.format_b, SparseFormat::Csr);
        assert!(comparison.memory_ratio > 0.0);
    }
}