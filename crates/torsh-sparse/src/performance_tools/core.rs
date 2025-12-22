//! # Core Profiling Infrastructure
//!
//! This module provides the foundational infrastructure for performance measurement
//! and profiling of sparse tensor operations. It includes the main `SparseProfiler`
//! class and core benchmarking capabilities.
//!
//! ## Key Components
//!
//! - **PerformanceMeasurement**: Individual measurement results with timing and memory data
//! - **BenchmarkConfig**: Configuration for benchmark execution parameters
//! - **SparseProfiler**: Main profiler class for comprehensive benchmarking
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_sparse::performance_tools::core::{SparseProfiler, BenchmarkConfig};
//!
//! let config = BenchmarkConfig::default();
//! let mut profiler = SparseProfiler::new(config);
//!
//! // Benchmark format conversion
//! let dense_matrix = create_test_matrix(1000, 1000);
//! let measurements = profiler.benchmark_format_conversion(&dense_matrix)?;
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{CooTensor, CsrTensor, SparseFormat, SparseTensor, TorshResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::{Shape, TorshError};
use torsh_tensor::Tensor;

/// Performance measurement result containing timing and memory information
///
/// This struct captures comprehensive performance data for a single operation,
/// including execution time, memory usage before/after, peak memory, and
/// additional custom metrics.
#[derive(Debug, Clone)]
pub struct PerformanceMeasurement {
    /// Operation name for identification
    pub operation: String,
    /// Execution time for the operation
    pub duration: Duration,
    /// Memory usage before operation (bytes)
    pub memory_before: usize,
    /// Memory usage after operation (bytes)
    pub memory_after: usize,
    /// Peak memory usage during operation (bytes)
    pub peak_memory: usize,
    /// Additional custom metrics as key-value pairs
    pub metrics: HashMap<String, f64>,
}

impl PerformanceMeasurement {
    /// Create a new performance measurement
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            duration: Duration::new(0, 0),
            memory_before: 0,
            memory_after: 0,
            peak_memory: 0,
            metrics: HashMap::new(),
        }
    }

    /// Add a custom metric to this measurement
    pub fn add_metric(&mut self, key: String, value: f64) {
        self.metrics.insert(key, value);
    }

    /// Get memory delta (increase during operation)
    pub fn memory_delta(&self) -> i64 {
        self.memory_after as i64 - self.memory_before as i64
    }

    /// Get peak memory increase from baseline
    pub fn peak_memory_increase(&self) -> usize {
        self.peak_memory.saturating_sub(self.memory_before)
    }
}

/// Configuration for benchmarking operations
///
/// This struct controls how benchmarks are executed, including iteration counts,
/// memory collection, and timing constraints.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warm-up iterations to run before measurement
    pub warmup_iterations: usize,
    /// Number of measured iterations for averaging
    pub measured_iterations: usize,
    /// Whether to collect detailed memory statistics
    pub collect_memory: bool,
    /// Whether to run garbage collection between iterations
    pub gc_between_iterations: bool,
    /// Maximum allowed execution time per iteration
    pub max_iteration_time: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            measured_iterations: 10,
            collect_memory: true,
            gc_between_iterations: false,
            max_iteration_time: Duration::from_secs(30),
        }
    }
}

impl BenchmarkConfig {
    /// Create a fast configuration for quick testing
    pub fn fast() -> Self {
        Self {
            warmup_iterations: 1,
            measured_iterations: 3,
            collect_memory: false,
            gc_between_iterations: false,
            max_iteration_time: Duration::from_secs(5),
        }
    }

    /// Create a thorough configuration for detailed analysis
    pub fn thorough() -> Self {
        Self {
            warmup_iterations: 5,
            measured_iterations: 20,
            collect_memory: true,
            gc_between_iterations: true,
            max_iteration_time: Duration::from_secs(60),
        }
    }

    /// Create a memory-focused configuration
    pub fn memory_focused() -> Self {
        Self {
            warmup_iterations: 1,
            measured_iterations: 5,
            collect_memory: true,
            gc_between_iterations: true,
            max_iteration_time: Duration::from_secs(10),
        }
    }
}

/// Comprehensive performance profiler for sparse tensor operations
///
/// The `SparseProfiler` is the main class for benchmarking sparse tensor operations.
/// It provides methods for profiling format conversions, matrix operations, and
/// memory usage patterns.
#[derive(Debug)]
pub struct SparseProfiler {
    /// Configuration for benchmark execution
    pub config: BenchmarkConfig,
    /// Collected performance measurements
    pub measurements: Vec<PerformanceMeasurement>,
    /// Operation counters for tracking
    pub operation_counters: HashMap<String, usize>,
}

impl Default for SparseProfiler {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}

impl SparseProfiler {
    /// Create a new sparse profiler with the given configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            measurements: Vec::new(),
            operation_counters: HashMap::new(),
        }
    }

    /// Benchmark sparse format conversion operations
    ///
    /// This method profiles the conversion between different sparse formats
    /// (COO, CSR, etc.) and measures the performance characteristics.
    ///
    /// # Arguments
    ///
    /// * `dense_matrix` - Input dense matrix to convert to sparse formats
    ///
    /// # Returns
    ///
    /// Vector of performance measurements for each format conversion
    pub fn benchmark_format_conversion(
        &mut self,
        dense_matrix: &Tensor,
    ) -> TorshResult<Vec<PerformanceMeasurement>> {
        let mut results = Vec::new();
        let operation_base = "format_conversion";

        // Convert to COO format
        let coo_measurement = self
            .measure_operation(format!("{}_to_coo", operation_base), || {
                self.convert_to_coo(dense_matrix)
            })?;
        results.push(coo_measurement);

        // Convert to CSR format
        let csr_measurement = self
            .measure_operation(format!("{}_to_csr", operation_base), || {
                self.convert_to_csr(dense_matrix)
            })?;
        results.push(csr_measurement);

        // Add measurements to collection
        self.measurements.extend(results.clone());

        // Update operation counters
        *self
            .operation_counters
            .entry(operation_base.to_string())
            .or_insert(0) += results.len();

        Ok(results)
    }

    /// Benchmark sparse matrix multiplication operations
    ///
    /// This method profiles matrix multiplication between sparse matrices
    /// in different formats and measures performance characteristics.
    ///
    /// # Arguments
    ///
    /// * `lhs` - Left-hand side sparse matrix
    /// * `rhs` - Right-hand side sparse matrix
    ///
    /// # Returns
    ///
    /// Vector of performance measurements for matrix multiplication
    pub fn benchmark_sparse_matmul(
        &mut self,
        lhs: &dyn SparseTensor,
        rhs: &dyn SparseTensor,
    ) -> TorshResult<Vec<PerformanceMeasurement>> {
        let mut results = Vec::new();
        let operation_base = "sparse_matmul";

        // Benchmark matrix multiplication
        let matmul_measurement = self.measure_operation(
            format!("{}_{:?}x{:?}", operation_base, lhs.format(), rhs.format()),
            || self.perform_matrix_multiplication(lhs, rhs),
        )?;
        results.push(matmul_measurement);

        // Add measurements to collection
        self.measurements.extend(results.clone());

        // Update operation counters
        *self
            .operation_counters
            .entry(operation_base.to_string())
            .or_insert(0) += results.len();

        Ok(results)
    }

    /// Benchmark dense to sparse conversion operations
    ///
    /// This method profiles the conversion from dense matrices to sparse
    /// formats with different sparsity patterns and thresholds.
    ///
    /// # Arguments
    ///
    /// * `dense_matrix` - Input dense matrix
    /// * `sparsity_threshold` - Threshold for considering elements as zero
    ///
    /// # Returns
    ///
    /// Vector of performance measurements for dense-to-sparse conversion
    pub fn benchmark_dense_to_sparse(
        &mut self,
        dense_matrix: &Tensor,
        sparsity_threshold: f32,
    ) -> TorshResult<Vec<PerformanceMeasurement>> {
        let mut results = Vec::new();
        let operation_base = "dense_to_sparse";

        // Benchmark conversion with threshold
        let conversion_measurement = self.measure_operation(
            format!("{}_threshold_{}", operation_base, sparsity_threshold),
            || self.convert_dense_to_sparse(dense_matrix, sparsity_threshold),
        )?;
        results.push(conversion_measurement);

        // Add measurements to collection
        self.measurements.extend(results.clone());

        // Update operation counters
        *self
            .operation_counters
            .entry(operation_base.to_string())
            .or_insert(0) += results.len();

        Ok(results)
    }

    /// Profile format comparison for a given dense matrix
    ///
    /// This method converts a dense matrix to all supported sparse formats
    /// and compares their performance characteristics.
    pub fn profile_format_comparison(
        &mut self,
        dense_matrix: &Tensor,
    ) -> TorshResult<Vec<PerformanceMeasurement>> {
        let mut all_measurements = Vec::new();

        // Benchmark all format conversions
        let format_measurements = self.benchmark_format_conversion(dense_matrix)?;
        all_measurements.extend(format_measurements);

        // Analyze sparsity characteristics
        let sparsity_ratio = self.calculate_sparsity_ratio(dense_matrix)?;
        let mut sparsity_measurement = PerformanceMeasurement::new("sparsity_analysis".to_string());
        sparsity_measurement.add_metric("sparsity_ratio".to_string(), sparsity_ratio as f64);
        all_measurements.push(sparsity_measurement);

        self.measurements.extend(all_measurements.clone());

        Ok(all_measurements)
    }

    /// Clear all collected measurements and reset counters
    pub fn clear_measurements(&mut self) {
        self.measurements.clear();
        self.operation_counters.clear();
    }

    /// Get the total number of measurements collected
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Get measurements for a specific operation
    pub fn get_measurements_for_operation(&self, operation: &str) -> Vec<&PerformanceMeasurement> {
        self.measurements
            .iter()
            .filter(|m| m.operation.contains(operation))
            .collect()
    }

    // Helper methods for actual operations

    /// Helper method to convert dense matrix to COO format
    fn convert_to_coo(&self, dense_matrix: &Tensor) -> TorshResult<CooTensor> {
        // Simplified COO conversion - in real implementation this would
        // use the actual sparse tensor creation logic
        let shape = dense_matrix.shape().to_vec();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // This is a placeholder - actual implementation would iterate through
        // the dense matrix and extract non-zero elements
        let nnz = (shape[0] * shape[1]) / 10; // Assume 10% sparsity
        for i in 0..nnz {
            row_indices.push(i % shape[0]);
            col_indices.push(i % shape[1]);
            values.push(1.0);
        }

        CooTensor::new(row_indices, col_indices, values, Shape::new(shape))
    }

    /// Helper method to convert dense matrix to CSR format
    fn convert_to_csr(&self, dense_matrix: &Tensor) -> TorshResult<CsrTensor> {
        // First convert to COO, then to CSR
        let coo = self.convert_to_coo(dense_matrix)?;
        CsrTensor::from_coo(&coo)
    }

    /// Helper method for matrix multiplication
    fn perform_matrix_multiplication(
        &self,
        lhs: &dyn SparseTensor,
        rhs: &dyn SparseTensor,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        // Simplified matrix multiplication - actual implementation would
        // handle different format combinations efficiently
        match (lhs.format(), rhs.format()) {
            (SparseFormat::Csr, SparseFormat::Csr) => {
                // CSR x CSR multiplication
                Ok(self.csr_multiply_simplified(
                    lhs.as_any().downcast_ref::<CsrTensor>().unwrap(),
                    rhs.as_any().downcast_ref::<CsrTensor>().unwrap(),
                )?)
            }
            _ => {
                // Convert to CSR and multiply
                let lhs_csr = match lhs.format() {
                    SparseFormat::Csr => lhs.as_any().downcast_ref::<CsrTensor>().unwrap().clone(),
                    _ => CsrTensor::from_coo(lhs.as_any().downcast_ref::<CooTensor>().unwrap())?,
                };
                let rhs_csr = match rhs.format() {
                    SparseFormat::Csr => rhs.as_any().downcast_ref::<CsrTensor>().unwrap().clone(),
                    _ => CsrTensor::from_coo(rhs.as_any().downcast_ref::<CooTensor>().unwrap())?,
                };
                Ok(self.csr_multiply_simplified(&lhs_csr, &rhs_csr)?)
            }
        }
    }

    /// Simplified CSR multiplication for benchmarking
    fn csr_multiply_simplified(
        &self,
        lhs: &CsrTensor,
        rhs: &CsrTensor,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        // This is a simplified implementation for benchmarking purposes
        // In practice, this would use optimized sparse BLAS operations

        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        if lhs_shape.dims()[1] != rhs_shape.dims()[0] {
            return Err(TorshError::InvalidArgument(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        // Create result matrix structure (simplified)
        let result_shape = Shape::new(vec![lhs_shape.dims()[0], rhs_shape.dims()[1]]);
        let result_values = vec![1.0; 100]; // Placeholder
        let result_col_indices = (0..100).collect();
        let result_row_ptrs = (0..=lhs_shape.dims()[0]).collect();

        let result = CsrTensor::new(
            result_row_ptrs,
            result_col_indices,
            result_values,
            result_shape,
        )?;
        Ok(Box::new(result))
    }

    /// Helper method to convert dense to sparse with threshold
    fn convert_dense_to_sparse(
        &self,
        dense_matrix: &Tensor,
        threshold: f32,
    ) -> TorshResult<CooTensor> {
        // Simplified conversion with sparsity threshold
        let shape = dense_matrix.shape().to_vec();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // This is a placeholder - actual implementation would iterate through
        // the dense matrix and filter based on threshold
        let estimated_nnz = (shape[0] as f64 * shape[1] as f64 * (1.0 - threshold as f64)) as usize;
        for i in 0..estimated_nnz {
            row_indices.push(i % shape[0]);
            col_indices.push(i % shape[1]);
            values.push(1.0 + threshold as f32);
        }

        CooTensor::new(row_indices, col_indices, values, Shape::new(shape))
    }

    /// Calculate sparsity ratio of a dense matrix
    fn calculate_sparsity_ratio(&self, dense_matrix: &Tensor) -> TorshResult<f32> {
        // Placeholder implementation - would count actual zeros
        let shape = dense_matrix.shape();
        let total_elements = shape.dims().iter().product::<usize>() as f32;
        let assumed_zeros = total_elements * 0.8; // Assume 80% zeros
        Ok(assumed_zeros / total_elements)
    }

    /// Generic method to measure operation performance
    fn measure_operation<F, R>(
        &self,
        operation: String,
        mut operation_fn: F,
    ) -> TorshResult<PerformanceMeasurement>
    where
        F: FnMut() -> TorshResult<R>,
    {
        let mut measurement = PerformanceMeasurement::new(operation);

        // Collect initial memory
        measurement.memory_before = self.get_current_memory_usage();

        // Warm-up iterations
        for _ in 0..self.config.warmup_iterations {
            operation_fn()?;
            if self.config.gc_between_iterations {
                // In Rust, we don't have explicit GC, but we could drop intermediate results
                std::hint::black_box(());
            }
        }

        // Measured iterations
        let mut durations = Vec::new();
        let mut peak_memory = measurement.memory_before;

        for _ in 0..self.config.measured_iterations {
            let start = Instant::now();
            let _start_memory = self.get_current_memory_usage();

            operation_fn()?;

            let end_memory = self.get_current_memory_usage();
            let duration = start.elapsed();

            if duration > self.config.max_iteration_time {
                return Err(TorshError::InvalidArgument(format!(
                    "Operation {} exceeded maximum iteration time",
                    measurement.operation
                )));
            }

            durations.push(duration);
            peak_memory = peak_memory.max(end_memory);

            if self.config.gc_between_iterations {
                std::hint::black_box(());
            }
        }

        // Calculate statistics
        measurement.duration = self.calculate_mean_duration(&durations);
        measurement.peak_memory = peak_memory;
        measurement.memory_after = self.get_current_memory_usage();

        // Add timing statistics as metrics
        if let (Some(&min_duration), Some(&max_duration)) =
            (durations.iter().min(), durations.iter().max())
        {
            measurement.add_metric(
                "min_time_ms".to_string(),
                min_duration.as_secs_f64() * 1000.0,
            );
            measurement.add_metric(
                "max_time_ms".to_string(),
                max_duration.as_secs_f64() * 1000.0,
            );
            measurement.add_metric("std_dev_ms".to_string(), self.calculate_std_dev(&durations));
        }

        Ok(measurement)
    }

    /// Get current memory usage (simplified implementation)
    fn get_current_memory_usage(&self) -> usize {
        // In a real implementation, this would use system calls or libraries
        // to get actual memory usage. For now, return a placeholder.
        #[cfg(target_os = "linux")]
        {
            // Could use /proc/self/status on Linux
            64 * 1024 * 1024 // 64MB placeholder
        }
        #[cfg(not(target_os = "linux"))]
        {
            64 * 1024 * 1024 // 64MB placeholder
        }
    }

    /// Calculate mean duration from a vector of durations
    fn calculate_mean_duration(&self, durations: &[Duration]) -> Duration {
        let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
        Duration::from_nanos(total_nanos / durations.len() as u64)
    }

    /// Calculate standard deviation of durations in milliseconds
    fn calculate_std_dev(&self, durations: &[Duration]) -> f64 {
        let mean =
            durations.iter().map(|d| d.as_nanos() as f64).sum::<f64>() / durations.len() as f64;
        let variance = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;
        (variance.sqrt()) / 1_000_000.0 // Convert to milliseconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_measurement_creation() {
        let measurement = PerformanceMeasurement::new("test_operation".to_string());
        assert_eq!(measurement.operation, "test_operation");
        assert_eq!(measurement.duration, Duration::new(0, 0));
        assert_eq!(measurement.memory_before, 0);
        assert_eq!(measurement.memory_after, 0);
        assert_eq!(measurement.peak_memory, 0);
        assert!(measurement.metrics.is_empty());
    }

    #[test]
    fn test_performance_measurement_metrics() {
        let mut measurement = PerformanceMeasurement::new("test".to_string());
        measurement.add_metric("test_metric".to_string(), 42.0);

        assert_eq!(measurement.metrics.get("test_metric"), Some(&42.0));
        assert_eq!(measurement.memory_delta(), 0);
        assert_eq!(measurement.peak_memory_increase(), 0);
    }

    #[test]
    fn test_benchmark_config_defaults() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 3);
        assert_eq!(config.measured_iterations, 10);
        assert!(config.collect_memory);
        assert!(!config.gc_between_iterations);
        assert_eq!(config.max_iteration_time, Duration::from_secs(30));
    }

    #[test]
    fn test_benchmark_config_presets() {
        let fast_config = BenchmarkConfig::fast();
        assert_eq!(fast_config.warmup_iterations, 1);
        assert_eq!(fast_config.measured_iterations, 3);
        assert!(!fast_config.collect_memory);

        let thorough_config = BenchmarkConfig::thorough();
        assert_eq!(thorough_config.warmup_iterations, 5);
        assert_eq!(thorough_config.measured_iterations, 20);
        assert!(thorough_config.collect_memory);
        assert!(thorough_config.gc_between_iterations);

        let memory_config = BenchmarkConfig::memory_focused();
        assert!(memory_config.collect_memory);
        assert!(memory_config.gc_between_iterations);
    }

    #[test]
    fn test_sparse_profiler_creation() {
        let config = BenchmarkConfig::fast();
        let profiler = SparseProfiler::new(config.clone());

        assert_eq!(profiler.config.warmup_iterations, config.warmup_iterations);
        assert!(profiler.measurements.is_empty());
        assert!(profiler.operation_counters.is_empty());
    }

    #[test]
    fn test_sparse_profiler_default() {
        let profiler = SparseProfiler::default();
        assert_eq!(profiler.config.warmup_iterations, 3);
        assert_eq!(profiler.measurement_count(), 0);
    }

    #[test]
    fn test_clear_measurements() {
        let mut profiler = SparseProfiler::default();

        // Add some dummy measurements
        profiler
            .measurements
            .push(PerformanceMeasurement::new("test1".to_string()));
        profiler
            .measurements
            .push(PerformanceMeasurement::new("test2".to_string()));
        profiler.operation_counters.insert("test".to_string(), 2);

        assert_eq!(profiler.measurement_count(), 2);
        assert_eq!(profiler.operation_counters.len(), 1);

        profiler.clear_measurements();

        assert_eq!(profiler.measurement_count(), 0);
        assert!(profiler.operation_counters.is_empty());
    }

    #[test]
    fn test_get_measurements_for_operation() {
        let mut profiler = SparseProfiler::default();

        // Add measurements with different operation names
        profiler.measurements.push(PerformanceMeasurement::new(
            "format_conversion_to_coo".to_string(),
        ));
        profiler.measurements.push(PerformanceMeasurement::new(
            "format_conversion_to_csr".to_string(),
        ));
        profiler
            .measurements
            .push(PerformanceMeasurement::new("sparse_matmul".to_string()));

        let format_measurements = profiler.get_measurements_for_operation("format_conversion");
        assert_eq!(format_measurements.len(), 2);

        let matmul_measurements = profiler.get_measurements_for_operation("matmul");
        assert_eq!(matmul_measurements.len(), 1);

        let nonexistent_measurements = profiler.get_measurements_for_operation("nonexistent");
        assert_eq!(nonexistent_measurements.len(), 0);
    }

    #[test]
    fn test_calculate_mean_duration() {
        let profiler = SparseProfiler::default();
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
        ];

        let mean = profiler.calculate_mean_duration(&durations);
        assert_eq!(mean.as_millis(), 200); // (100 + 200 + 300) / 3 = 200
    }

    #[test]
    fn test_calculate_std_dev() {
        let profiler = SparseProfiler::default();
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
        ];

        let std_dev = profiler.calculate_std_dev(&durations);
        // Standard deviation should be approximately 81.65 ms
        assert!((std_dev - 81.65).abs() < 1.0);
    }

    #[test]
    fn test_memory_measurement_calculations() {
        let mut measurement = PerformanceMeasurement::new("test".to_string());
        measurement.memory_before = 1000;
        measurement.memory_after = 1500;
        measurement.peak_memory = 2000;

        assert_eq!(measurement.memory_delta(), 500);
        assert_eq!(measurement.peak_memory_increase(), 1000);
    }

    #[test]
    fn test_memory_measurement_no_increase() {
        let mut measurement = PerformanceMeasurement::new("test".to_string());
        measurement.memory_before = 1000;
        measurement.memory_after = 800;
        measurement.peak_memory = 900;

        assert_eq!(measurement.memory_delta(), -200);
        assert_eq!(measurement.peak_memory_increase(), 0); // saturating_sub protects against underflow
    }
}
