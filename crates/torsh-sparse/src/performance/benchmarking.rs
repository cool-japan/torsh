//! Benchmarking and profiling tools for sparse tensor operations
//!
//! This module provides comprehensive benchmarking capabilities for sparse tensors,
//! including format conversion timing, matrix multiplication performance analysis,
//! and memory usage profiling.

use crate::{CooTensor, CsrTensor, SparseFormat, SparseTensor, TorshResult};
use super::core::{BenchmarkConfig, PerformanceMeasurement};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::{Shape, TorshError};
use torsh_tensor::Tensor;

/// Comprehensive performance profiler for sparse operations
///
/// SparseProfiler provides a unified interface for benchmarking various sparse tensor
/// operations including format conversions, matrix multiplications, and memory analysis.
/// It supports configurable benchmark parameters and collects detailed performance metrics.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::{SparseProfiler, BenchmarkConfig};
/// use torsh_sparse::{CooTensor, SparseFormat};
/// use torsh_core::Shape;
///
/// let config = BenchmarkConfig::fast();
/// let mut profiler = SparseProfiler::new(config);
///
/// // Create a test sparse tensor
/// let rows = vec![0, 0, 1, 2];
/// let cols = vec![0, 2, 1, 2];
/// let vals = vec![1.0, 2.0, 3.0, 4.0];
/// let shape = Shape::new(vec![3, 3]);
/// let coo = CooTensor::new(rows, cols, vals, shape)?;
///
/// // Benchmark format conversion
/// let measurement = profiler.benchmark_format_conversion(&coo, SparseFormat::Csr)?;
/// println!("Conversion took: {:?}", measurement.duration);
/// ```
pub struct SparseProfiler {
    /// Configuration for benchmarks
    config: BenchmarkConfig,
    /// Collected measurements
    measurements: Vec<PerformanceMeasurement>,
    /// Operation counters
    operation_counts: HashMap<String, usize>,
}

impl Default for SparseProfiler {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}

impl SparseProfiler {
    /// Create a new sparse profiler
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            measurements: Vec::new(),
            operation_counts: HashMap::new(),
        }
    }

    /// Benchmark sparse format conversion
    ///
    /// Measures the time and memory usage for converting between different sparse formats.
    /// The benchmark includes warm-up iterations to ensure stable measurements and
    /// collects various performance metrics.
    ///
    /// # Arguments
    ///
    /// * `sparse` - The sparse tensor to convert from
    /// * `target_format` - The target sparse format to convert to
    ///
    /// # Returns
    ///
    /// Returns a `PerformanceMeasurement` containing timing, memory, and operation-specific metrics.
    pub fn benchmark_format_conversion(
        &mut self,
        sparse: &dyn SparseTensor,
        target_format: SparseFormat,
    ) -> TorshResult<PerformanceMeasurement> {
        let operation = format!("convert_{:?}_to_{:?}", sparse.format(), target_format);

        // Warm-up iterations
        for _ in 0..self.config.warmup_iterations {
            let _ = self.convert_format(sparse, target_format)?;
        }

        let mut durations = Vec::new();
        let mut memory_before = 0;
        let mut memory_after = 0;
        let mut peak_memory = 0;

        // Measured iterations
        for _ in 0..self.config.measured_iterations {
            if self.config.collect_memory {
                memory_before = self.estimate_memory_usage(sparse);
            }

            let start = Instant::now();
            let result = self.convert_format(sparse, target_format)?;
            let duration = start.elapsed();

            if self.config.collect_memory {
                memory_after = memory_before + self.estimate_memory_usage(&*result);
                peak_memory = std::cmp::max(peak_memory, memory_after);
            }

            durations.push(duration);

            if duration > self.config.max_iteration_time {
                return Err(TorshError::ComputeError(format!(
                    "Operation exceeded maximum time limit: {duration:?}"
                )));
            }
        }

        let avg_duration = Duration::from_nanos(
            (durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128) as u64,
        );

        let mut metrics = HashMap::new();
        metrics.insert(
            "min_time_ns".to_string(),
            durations.iter().min().expect("durations should not be empty").as_nanos() as f64,
        );
        metrics.insert(
            "max_time_ns".to_string(),
            durations.iter().max().expect("durations should not be empty").as_nanos() as f64,
        );
        metrics.insert("std_dev_ns".to_string(), self.compute_std_dev(&durations));
        metrics.insert("nnz".to_string(), sparse.nnz() as f64);
        metrics.insert("sparsity".to_string(), sparse.sparsity() as f64);

        let measurement = PerformanceMeasurement {
            operation: operation.clone(),
            duration: avg_duration,
            memory_before,
            memory_after,
            peak_memory,
            metrics,
        };

        self.measurements.push(measurement.clone());
        *self.operation_counts.entry(operation).or_insert(0) += 1;

        Ok(measurement)
    }

    /// Benchmark sparse matrix multiplication
    ///
    /// Measures the performance of sparse matrix multiplication between two sparse tensors.
    /// This benchmark estimates FLOPS and analyzes the efficiency of the multiplication algorithm.
    ///
    /// # Arguments
    ///
    /// * `a` - The left sparse matrix
    /// * `b` - The right sparse matrix
    ///
    /// # Returns
    ///
    /// Returns a `PerformanceMeasurement` with timing and FLOPS estimates.
    pub fn benchmark_sparse_matmul(
        &mut self,
        a: &dyn SparseTensor,
        b: &dyn SparseTensor,
    ) -> TorshResult<PerformanceMeasurement> {
        let operation = format!("matmul_{:?}_{:?}", a.format(), b.format());

        // Warm-up iterations
        for _ in 0..self.config.warmup_iterations {
            let _ = self.perform_matmul(a, b)?;
        }

        let mut durations = Vec::new();
        let memory_before = if self.config.collect_memory {
            self.estimate_memory_usage(a) + self.estimate_memory_usage(b)
        } else {
            0
        };

        // Measured iterations
        for _ in 0..self.config.measured_iterations {
            let start = Instant::now();
            let _result = self.perform_matmul(a, b)?;
            let duration = start.elapsed();
            durations.push(duration);
        }

        let avg_duration = Duration::from_nanos(
            (durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128) as u64,
        );

        let mut metrics = HashMap::new();
        metrics.insert("a_nnz".to_string(), a.nnz() as f64);
        metrics.insert("b_nnz".to_string(), b.nnz() as f64);
        metrics.insert("flops_estimate".to_string(), (2 * a.nnz() * b.nnz()) as f64);
        metrics.insert(
            "min_time_ns".to_string(),
            durations.iter().min().expect("durations should not be empty").as_nanos() as f64,
        );
        metrics.insert(
            "max_time_ns".to_string(),
            durations.iter().max().expect("durations should not be empty").as_nanos() as f64,
        );

        let measurement = PerformanceMeasurement {
            operation: operation.clone(),
            duration: avg_duration,
            memory_before,
            memory_after: memory_before, // Approximation
            peak_memory: memory_before,
            metrics,
        };

        self.measurements.push(measurement.clone());
        *self.operation_counts.entry(operation).or_insert(0) += 1;

        Ok(measurement)
    }

    /// Benchmark dense to sparse conversion
    ///
    /// Measures the performance of converting a dense tensor to a sparse representation.
    /// This includes timing the conversion process and analyzing compression ratios.
    ///
    /// # Arguments
    ///
    /// * `dense` - The dense tensor to convert
    /// * `format` - The target sparse format
    /// * `threshold` - Values below this threshold are considered zero
    ///
    /// # Returns
    ///
    /// Returns a `PerformanceMeasurement` with conversion timing and compression metrics.
    pub fn benchmark_dense_to_sparse(
        &mut self,
        dense: &Tensor,
        format: SparseFormat,
        threshold: f32,
    ) -> TorshResult<PerformanceMeasurement> {
        let operation = format!("dense_to_{format:?}");

        // Warm-up iterations
        for _ in 0..self.config.warmup_iterations {
            let _ = self.convert_dense_to_sparse(dense, format, threshold)?;
        }

        let mut durations = Vec::new();
        let memory_before = if self.config.collect_memory {
            dense.shape().numel() * std::mem::size_of::<f32>()
        } else {
            0
        };

        // Measured iterations
        for _ in 0..self.config.measured_iterations {
            let start = Instant::now();
            let _result = self.convert_dense_to_sparse(dense, format, threshold)?;
            let duration = start.elapsed();
            durations.push(duration);
        }

        let avg_duration = Duration::from_nanos(
            (durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128) as u64,
        );

        let mut metrics = HashMap::new();
        metrics.insert("dense_elements".to_string(), dense.numel() as f64);
        metrics.insert(
            "compression_ratio".to_string(),
            dense.numel() as f64 / (self.count_nonzeros(dense, threshold)? as f64),
        );

        let measurement = PerformanceMeasurement {
            operation: operation.clone(),
            duration: avg_duration,
            memory_before,
            memory_after: memory_before, // Approximation
            peak_memory: memory_before,
            metrics,
        };

        self.measurements.push(measurement.clone());
        Ok(measurement)
    }

    /// Profile all supported formats for a given dense matrix
    ///
    /// Compares the performance of converting a dense matrix to different sparse formats,
    /// helping to identify the most efficient format for a given matrix pattern.
    ///
    /// # Arguments
    ///
    /// * `dense` - The dense matrix to analyze
    /// * `threshold` - Sparsity threshold for conversion
    ///
    /// # Returns
    ///
    /// Returns a map of sparse formats to their corresponding performance measurements.
    pub fn profile_format_comparison(
        &mut self,
        dense: &Tensor,
        threshold: f32,
    ) -> TorshResult<HashMap<SparseFormat, PerformanceMeasurement>> {
        let formats = vec![SparseFormat::Coo, SparseFormat::Csr, SparseFormat::Csc];

        let mut results = HashMap::new();

        for format in formats {
            let measurement = self.benchmark_dense_to_sparse(dense, format, threshold)?;
            results.insert(format, measurement);
        }

        Ok(results)
    }

    /// Get all collected measurements
    pub fn measurements(&self) -> &[PerformanceMeasurement] {
        &self.measurements
    }

    /// Get operation counts
    pub fn operation_counts(&self) -> &HashMap<String, usize> {
        &self.operation_counts
    }

    /// Clear all collected measurements
    pub fn clear_measurements(&mut self) {
        self.measurements.clear();
        self.operation_counts.clear();
    }

    /// Get benchmark configuration
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Update benchmark configuration
    pub fn set_config(&mut self, config: BenchmarkConfig) {
        self.config = config;
    }

    // Private helper methods

    /// Helper method to convert between formats
    fn convert_format(
        &self,
        sparse: &dyn SparseTensor,
        target_format: SparseFormat,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        match target_format {
            SparseFormat::Coo => Ok(Box::new(sparse.to_coo()?)),
            SparseFormat::Csr => Ok(Box::new(sparse.to_csr()?)),
            SparseFormat::Csc => Ok(Box::new(sparse.to_csc()?)),
            _ => Err(TorshError::UnsupportedOperation {
                op: format!("Conversion to {target_format:?}"),
                dtype: "sparse_tensor".to_string(),
            }),
        }
    }

    /// Helper method for matrix multiplication
    fn perform_matmul(
        &self,
        a: &dyn SparseTensor,
        b: &dyn SparseTensor,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert both to CSR for multiplication
        let a_csr = a.to_csr()?;
        let b_csr = b.to_csr()?;

        // Perform multiplication (simplified implementation)
        let result_coo = self.csr_multiply(&a_csr, &b_csr)?;
        Ok(Box::new(result_coo))
    }

    /// Simplified CSR multiplication
    fn csr_multiply(&self, a: &CsrTensor, b: &CsrTensor) -> TorshResult<CooTensor> {
        // For now, convert to COO and perform basic multiplication
        let a_coo = a.to_coo()?;
        let b_coo = b.to_coo()?;

        // This is a simplified implementation - in practice, you'd use optimized CSR multiplication
        let mut result_triplets = Vec::new();
        let a_triplets = a_coo.triplets();
        let b_triplets = b_coo.triplets();

        // Build column map for B
        let mut b_col_map: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (row, col, val) in b_triplets {
            b_col_map.entry(row).or_default().push((col, val));
        }

        // Perform multiplication
        for (a_row, a_col, a_val) in a_triplets {
            if let Some(b_entries) = b_col_map.get(&a_col) {
                for &(b_col, b_val) in b_entries {
                    result_triplets.push((a_row, b_col, a_val * b_val));
                }
            }
        }

        // Aggregate duplicates (simplified)
        result_triplets.sort_by_key(|&(r, c, _)| (r, c));
        let mut final_triplets = Vec::new();
        let mut current_sum = 0.0;
        let mut current_pos = (usize::MAX, usize::MAX);

        for (r, c, v) in result_triplets {
            if (r, c) == current_pos {
                current_sum += v;
            } else {
                if current_pos != (usize::MAX, usize::MAX) && current_sum.abs() > 1e-12 {
                    final_triplets.push((current_pos.0, current_pos.1, current_sum));
                }
                current_pos = (r, c);
                current_sum = v;
            }
        }

        if current_pos != (usize::MAX, usize::MAX) && current_sum.abs() > 1e-12 {
            final_triplets.push((current_pos.0, current_pos.1, current_sum));
        }

        let (rows, cols, vals): (Vec<_>, Vec<_>, Vec<_>) = final_triplets.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rs, mut cs, mut vs), (r, c, v)| {
                rs.push(r);
                cs.push(c);
                vs.push(v);
                (rs, cs, vs)
            },
        );

        let result_shape = Shape::new(vec![a.shape().dims()[0], b.shape().dims()[1]]);
        CooTensor::new(rows, cols, vals, result_shape)
    }

    /// Helper method to convert dense to sparse
    fn convert_dense_to_sparse(
        &self,
        dense: &Tensor,
        format: SparseFormat,
        threshold: f32,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        let coo = CooTensor::from_dense(dense, threshold)?;
        self.convert_format(&coo, format)
    }

    /// Estimate memory usage for a sparse tensor
    fn estimate_memory_usage(&self, sparse: &dyn SparseTensor) -> usize {
        let nnz = sparse.nnz();
        match sparse.format() {
            SparseFormat::Coo => nnz * 12, // row, col, val
            SparseFormat::Csr => nnz * 8 + sparse.shape().dims()[0] * 4,
            SparseFormat::Csc => nnz * 8 + sparse.shape().dims()[1] * 4,
            _ => nnz * 12,
        }
    }

    /// Count non-zero elements in dense tensor
    fn count_nonzeros(&self, dense: &Tensor, threshold: f32) -> TorshResult<usize> {
        let mut count = 0;
        let shape = dense.shape();

        for i in 0..shape.dims()[0] {
            for j in 0..shape.dims()[1] {
                let val = dense.get(&[i, j])?;
                if val.abs() > threshold {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Compute standard deviation of durations
    fn compute_std_dev(&self, durations: &[Duration]) -> f64 {
        let mean =
            durations.iter().map(|d| d.as_nanos()).sum::<u128>() as f64 / durations.len() as f64;
        let variance = durations
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / durations.len() as f64;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CooTensor;
    use torsh_core::Shape;

    fn create_test_sparse_tensor() -> TorshResult<CooTensor> {
        let rows = vec![0, 0, 1, 2];
        let cols = vec![0, 2, 1, 2];
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let shape = Shape::new(vec![3, 3]);
        CooTensor::new(rows, cols, vals, shape)
    }

    #[test]
    fn test_sparse_profiler_creation() {
        let config = BenchmarkConfig::fast();
        let profiler = SparseProfiler::new(config);

        assert_eq!(profiler.measurements().len(), 0);
        assert_eq!(profiler.operation_counts().len(), 0);
        assert_eq!(profiler.config().warmup_iterations, 1);
    }

    #[test]
    fn test_format_conversion_benchmark() -> TorshResult<()> {
        let mut profiler = SparseProfiler::new(BenchmarkConfig::fast());
        let coo = create_test_sparse_tensor()?;

        let measurement = profiler.benchmark_format_conversion(&coo, SparseFormat::Csr)?;

        assert_eq!(measurement.operation, "convert_Coo_to_Csr");
        assert!(measurement.duration.as_nanos() > 0);
        assert!(measurement.metrics.contains_key("nnz"));
        assert!(measurement.metrics.contains_key("sparsity"));

        Ok(())
    }

    #[test]
    fn test_profiler_measurements_collection() -> TorshResult<()> {
        let mut profiler = SparseProfiler::new(BenchmarkConfig::fast());
        let coo = create_test_sparse_tensor()?;

        // Run multiple benchmarks
        profiler.benchmark_format_conversion(&coo, SparseFormat::Csr)?;
        profiler.benchmark_format_conversion(&coo, SparseFormat::Csc)?;

        assert_eq!(profiler.measurements().len(), 2);
        assert_eq!(profiler.operation_counts().len(), 2);
        assert_eq!(profiler.operation_counts()["convert_Coo_to_Csr"], 1);
        assert_eq!(profiler.operation_counts()["convert_Coo_to_Csc"], 1);

        Ok(())
    }

    #[test]
    fn test_clear_measurements() -> TorshResult<()> {
        let mut profiler = SparseProfiler::new(BenchmarkConfig::fast());
        let coo = create_test_sparse_tensor()?;

        profiler.benchmark_format_conversion(&coo, SparseFormat::Csr)?;
        assert_eq!(profiler.measurements().len(), 1);

        profiler.clear_measurements();
        assert_eq!(profiler.measurements().len(), 0);
        assert_eq!(profiler.operation_counts().len(), 0);

        Ok(())
    }

    #[test]
    fn test_memory_estimation() {
        let profiler = SparseProfiler::new(BenchmarkConfig::default());
        let coo = create_test_sparse_tensor().unwrap();

        let memory_usage = profiler.estimate_memory_usage(&coo);
        assert_eq!(memory_usage, 4 * 12); // 4 non-zeros * 12 bytes each (COO format)
    }

    #[test]
    fn test_std_dev_computation() {
        let profiler = SparseProfiler::new(BenchmarkConfig::default());
        let durations = vec![
            Duration::from_nanos(100),
            Duration::from_nanos(110),
            Duration::from_nanos(90),
            Duration::from_nanos(105),
        ];

        let std_dev = profiler.compute_std_dev(&durations);
        assert!(std_dev > 0.0);
        assert!(std_dev < 10.0); // Should be reasonable for this small variance
    }
}