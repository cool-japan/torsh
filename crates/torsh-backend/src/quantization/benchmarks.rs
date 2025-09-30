//! Performance benchmarking and measurement utilities for quantization operations
//!
//! This module provides comprehensive benchmarking tools for measuring and analyzing
//! the performance of quantization operations. It includes utilities for measuring
//! throughput, latency, memory usage, and comparing different quantization configurations.

use crate::quantization::{
    hardware::QuantizationHardwareFeatures, QuantizationOps, QuantizationParams, QuantizedDType,
    QuantizedTensor,
};
use crate::{BackendResult, Device};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Comprehensive benchmarking suite for quantization operations
///
/// This struct provides a framework for systematically benchmarking different
/// quantization configurations and operations to measure their performance
/// characteristics across various hardware platforms.
#[derive(Debug)]
pub struct QuantizationBenchmarkSuite {
    /// Device for benchmarking
    device: Device,
    /// Hardware features available
    hw_features: QuantizationHardwareFeatures,
    /// Benchmark results storage
    results: HashMap<String, Vec<BenchmarkResult>>,
    /// Configuration for benchmark runs
    config: BenchmarkConfig,
}

impl QuantizationBenchmarkSuite {
    /// Create a new benchmark suite
    ///
    /// # Arguments
    ///
    /// * `device` - Target device for benchmarking
    /// * `config` - Configuration parameters for benchmarks
    pub fn new(device: Device, config: BenchmarkConfig) -> Self {
        let hw_features = QuantizationHardwareFeatures::detect_for_device(&device);

        Self {
            device,
            hw_features,
            results: HashMap::new(),
            config,
        }
    }

    /// Benchmark quantization operations across different data types and sizes
    ///
    /// Runs comprehensive benchmarks for quantization and dequantization operations
    /// using various quantization parameters and data sizes.
    pub fn benchmark_quantization_ops<T: QuantizationOps>(
        &mut self,
        ops: &T,
    ) -> BackendResult<BenchmarkSummary> {
        let mut summary = BenchmarkSummary::new("quantization_ops".to_string());

        // Test different quantization data types
        let dtypes = vec![
            QuantizedDType::Int8,
            QuantizedDType::UInt8,
            QuantizedDType::Int4,
            QuantizedDType::UInt4,
        ];

        // Test different data sizes
        let sizes = vec![1024, 4096, 16384, 65536, 262144];

        for dtype in dtypes {
            for size in &sizes {
                // Create test data
                let test_data: Vec<f32> = (0..*size)
                    .map(|i| (i as f32 / *size as f32) * 10.0 - 5.0)
                    .collect();

                let params = self.create_test_params(dtype.clone());

                // Benchmark quantization
                let quant_result = self
                    .benchmark_operation(&format!("quantize_{:?}_{}", dtype, size), || {
                        ops.quantize_f32(&test_data, &params)
                    })?;

                summary.add_result(quant_result);

                // Benchmark dequantization if quantization succeeded
                if let Ok(quantized_data) = ops.quantize_f32(&test_data, &params) {
                    let dequant_result = self
                        .benchmark_operation(&format!("dequantize_{:?}_{}", dtype, size), || {
                            ops.dequantize_f32(&quantized_data, &params)
                        })?;

                    summary.add_result(dequant_result);
                }
            }
        }

        Ok(summary)
    }

    /// Benchmark matrix operations with different configurations
    pub fn benchmark_matrix_ops<T: QuantizationOps>(
        &mut self,
        ops: &T,
    ) -> BackendResult<BenchmarkSummary> {
        let mut summary = BenchmarkSummary::new("matrix_ops".to_string());

        // Test different matrix sizes
        let matrix_sizes = vec![
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ];

        let dtypes = vec![QuantizedDType::Int8, QuantizedDType::UInt8];

        for (m, k, n) in matrix_sizes {
            for dtype in &dtypes {
                let params = self.create_test_params(dtype.clone());

                // Create test matrices
                let a_data = vec![100u8; m * k];
                let b_data = vec![100u8; k * n];

                let a_tensor = QuantizedTensor {
                    data: a_data,
                    shape: vec![m, k],
                    params: params.clone(),
                    device: self.device.clone(),
                };

                let b_tensor = QuantizedTensor {
                    data: b_data,
                    shape: vec![k, n],
                    params: params.clone(),
                    device: self.device.clone(),
                };

                // Benchmark matrix multiplication
                let matmul_result = self.benchmark_operation(
                    &format!("qmatmul_{:?}_{}x{}x{}", dtype, m, k, n),
                    || ops.qmatmul(&a_tensor, &b_tensor),
                )?;

                summary.add_result(matmul_result);
            }
        }

        Ok(summary)
    }

    /// Benchmark element-wise operations
    pub fn benchmark_elementwise_ops<T: QuantizationOps>(
        &mut self,
        ops: &T,
    ) -> BackendResult<BenchmarkSummary> {
        let mut summary = BenchmarkSummary::new("elementwise_ops".to_string());

        let sizes = vec![1024, 4096, 16384, 65536];
        let dtypes = vec![QuantizedDType::Int8, QuantizedDType::UInt8];

        for size in sizes {
            for dtype in &dtypes {
                let params = self.create_test_params(dtype.clone());

                // Create test tensors
                let a_data = vec![100u8; size];
                let b_data = vec![50u8; size];

                let a_tensor = QuantizedTensor {
                    data: a_data,
                    shape: vec![size],
                    params: params.clone(),
                    device: self.device.clone(),
                };

                let b_tensor = QuantizedTensor {
                    data: b_data.clone(),
                    shape: vec![size],
                    params: params.clone(),
                    device: self.device.clone(),
                };

                // Benchmark addition
                let add_result = self
                    .benchmark_operation(&format!("qadd_{:?}_{}", dtype, size), || {
                        ops.qadd(&a_tensor, &b_tensor)
                    })?;

                summary.add_result(add_result);

                // Benchmark ReLU
                let relu_result = self
                    .benchmark_operation(&format!("qrelu_{:?}_{}", dtype, size), || {
                        ops.qrelu(&a_tensor)
                    })?;

                summary.add_result(relu_result);
            }
        }

        Ok(summary)
    }

    /// Benchmark memory usage for different quantization configurations
    pub fn benchmark_memory_usage(&mut self) -> BackendResult<MemoryBenchmarkResults> {
        let mut results = MemoryBenchmarkResults::new();

        // Test memory usage for different data types
        let sizes = vec![1024, 4096, 16384, 65536, 262144];
        let dtypes = vec![
            QuantizedDType::Int8,
            QuantizedDType::UInt8,
            QuantizedDType::Int4,
            QuantizedDType::UInt4,
            QuantizedDType::Binary,
        ];

        for size in sizes {
            for dtype in &dtypes {
                let params = self.create_test_params(dtype.clone());

                // Create tensor and measure memory usage
                let tensor = QuantizedTensor {
                    data: vec![0u8; Self::calculate_quantized_size(size, dtype)],
                    shape: vec![size],
                    params,
                    device: self.device.clone(),
                };

                let memory_usage = MemoryUsage {
                    dtype: dtype.clone(),
                    elements: size,
                    bytes_used: tensor.memory_usage(),
                    compression_ratio: Self::calculate_compression_ratio(size, dtype),
                    memory_efficiency: Self::calculate_memory_efficiency(dtype),
                };

                results.add_measurement(memory_usage);
            }
        }

        Ok(results)
    }

    /// Perform a comparative benchmark between different quantization methods
    pub fn comparative_benchmark<T: QuantizationOps>(
        &mut self,
        ops: &T,
        test_data: &[f32],
    ) -> BackendResult<ComparativeBenchmarkResult> {
        let mut result = ComparativeBenchmarkResult::new();

        // Test different quantization configurations
        let configs = vec![
            ("int8_symmetric", QuantizationParams::int8_symmetric()),
            ("uint8_asymmetric", QuantizationParams::uint8_asymmetric()),
            ("int4_symmetric", QuantizationParams::int4_symmetric()),
        ];

        for (name, params) in configs {
            // Benchmark quantization
            let quant_benchmark = self
                .benchmark_operation(&format!("{}_quantize", name), || {
                    ops.quantize_f32(test_data, &params)
                })?;

            // Benchmark dequantization
            let quantized_data = ops.quantize_f32(test_data, &params)?;
            let dequant_benchmark = self
                .benchmark_operation(&format!("{}_dequantize", name), || {
                    ops.dequantize_f32(&quantized_data, &params)
                })?;

            // Calculate accuracy (MSE between original and dequantized)
            let dequantized_data = ops.dequantize_f32(&quantized_data, &params)?;
            let mse = Self::calculate_mse(test_data, &dequantized_data);

            let config_result = ConfigurationBenchmarkResult {
                name: name.to_string(),
                params,
                quantization_performance: quant_benchmark,
                dequantization_performance: dequant_benchmark,
                accuracy_mse: mse,
                memory_usage: quantized_data.len(),
                compression_ratio: test_data.len() * 4 / quantized_data.len(), // f32 vs quantized
            };

            result.add_configuration(config_result);
        }

        Ok(result)
    }

    /// Benchmark a single operation with timing and memory measurement
    fn benchmark_operation<F, R>(&self, name: &str, operation: F) -> BackendResult<BenchmarkResult>
    where
        F: Fn() -> BackendResult<R>,
    {
        let mut durations = Vec::new();
        let mut memory_measurements = Vec::new();

        // Warm-up runs
        for _ in 0..self.config.warmup_iterations {
            let _ = operation();
        }

        // Actual benchmark runs
        for _ in 0..self.config.benchmark_iterations {
            let memory_before = Self::get_memory_usage();
            let start_time = Instant::now();

            match operation() {
                Ok(_) => {
                    let duration = start_time.elapsed();
                    let memory_after = Self::get_memory_usage();

                    durations.push(duration);
                    memory_measurements.push(memory_after.saturating_sub(memory_before));
                }
                Err(e) => return Err(e),
            }
        }

        // Calculate statistics
        let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
        let min_duration = durations.iter().min().copied().unwrap_or_default();
        let max_duration = durations.iter().max().copied().unwrap_or_default();

        let avg_memory = memory_measurements.iter().sum::<usize>() / memory_measurements.len();

        Ok(BenchmarkResult {
            operation_name: name.to_string(),
            avg_duration,
            min_duration,
            max_duration,
            throughput: Self::calculate_throughput(&durations),
            memory_usage: avg_memory,
            iterations: self.config.benchmark_iterations,
        })
    }

    /// Create test quantization parameters for a given data type
    fn create_test_params(&self, dtype: QuantizedDType) -> QuantizationParams {
        match dtype {
            QuantizedDType::Int8 => QuantizationParams::int8_symmetric(),
            QuantizedDType::UInt8 => QuantizationParams::uint8_asymmetric(),
            QuantizedDType::Int4 => QuantizationParams::int4_symmetric(),
            QuantizedDType::UInt4 => {
                let mut params = QuantizationParams::int4_symmetric();
                params.dtype = QuantizedDType::UInt4;
                params
            }
            _ => QuantizationParams::default(),
        }
    }

    /// Calculate the size needed for quantized data
    fn calculate_quantized_size(elements: usize, dtype: &QuantizedDType) -> usize {
        match dtype {
            QuantizedDType::Int4 | QuantizedDType::UInt4 => (elements + 1) / 2,
            QuantizedDType::Binary => (elements + 7) / 8,
            QuantizedDType::Int8 | QuantizedDType::UInt8 => elements,
            QuantizedDType::Int16 | QuantizedDType::UInt16 => elements * 2,
            QuantizedDType::Mixed(_) => elements, // Conservative estimate
        }
    }

    /// Calculate compression ratio compared to FP32
    fn calculate_compression_ratio(elements: usize, dtype: &QuantizedDType) -> f32 {
        let fp32_size = elements * 4; // 4 bytes per f32
        let quantized_size = Self::calculate_quantized_size(elements, dtype);
        fp32_size as f32 / quantized_size as f32
    }

    /// Calculate memory efficiency (values per byte)
    fn calculate_memory_efficiency(dtype: &QuantizedDType) -> f32 {
        match dtype {
            QuantizedDType::Int8 | QuantizedDType::UInt8 => 1.0,
            QuantizedDType::Int16 | QuantizedDType::UInt16 => 0.5,
            QuantizedDType::Int4 | QuantizedDType::UInt4 => 2.0,
            QuantizedDType::Binary => 8.0,
            QuantizedDType::Mixed(_) => 1.0, // Conservative estimate
        }
    }

    /// Calculate MSE between original and reconstructed data
    fn calculate_mse(original: &[f32], reconstructed: &[f32]) -> f32 {
        if original.len() != reconstructed.len() {
            return f32::INFINITY;
        }

        let mse: f64 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            / original.len() as f64;

        mse as f32
    }

    /// Calculate throughput (operations per second)
    fn calculate_throughput(durations: &[Duration]) -> f64 {
        if durations.is_empty() {
            return 0.0;
        }

        let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
        if avg_duration.as_secs_f64() > 0.0 {
            1.0 / avg_duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get current memory usage (simplified implementation)
    fn get_memory_usage() -> usize {
        // In a real implementation, this would use OS-specific APIs
        // to get actual memory usage. For now, return 0 as placeholder.
        0
    }

    /// Clear all benchmark results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Get all benchmark results
    pub fn get_results(&self) -> &HashMap<String, Vec<BenchmarkResult>> {
        &self.results
    }

    /// Save benchmark results to storage
    pub fn save_results(&self, path: &str) -> BackendResult<()> {
        // In a real implementation, this would serialize results to a file
        // For now, just return Ok
        Ok(())
    }
}

/// Configuration for benchmark runs
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations before actual benchmarking
    pub warmup_iterations: usize,
    /// Number of benchmark iterations for timing
    pub benchmark_iterations: usize,
    /// Whether to include memory measurements
    pub measure_memory: bool,
    /// Whether to include accuracy measurements
    pub measure_accuracy: bool,
    /// Timeout for individual operations (milliseconds)
    pub operation_timeout_ms: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            benchmark_iterations: 10,
            measure_memory: true,
            measure_accuracy: true,
            operation_timeout_ms: 10000, // 10 seconds
        }
    }
}

/// Result of a single benchmark operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the operation
    pub operation_name: String,
    /// Average execution duration
    pub avg_duration: Duration,
    /// Minimum execution duration
    pub min_duration: Duration,
    /// Maximum execution duration
    pub max_duration: Duration,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Number of iterations run
    pub iterations: usize,
}

impl BenchmarkResult {
    /// Get duration in milliseconds
    pub fn avg_duration_ms(&self) -> f64 {
        self.avg_duration.as_secs_f64() * 1000.0
    }

    /// Get minimum duration in milliseconds
    pub fn min_duration_ms(&self) -> f64 {
        self.min_duration.as_secs_f64() * 1000.0
    }

    /// Get maximum duration in milliseconds
    pub fn max_duration_ms(&self) -> f64 {
        self.max_duration.as_secs_f64() * 1000.0
    }
}

/// Summary of multiple benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Name of the benchmark suite
    pub suite_name: String,
    /// Individual benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Overall statistics
    pub statistics: SummaryStatistics,
}

impl BenchmarkSummary {
    /// Create a new benchmark summary
    pub fn new(suite_name: String) -> Self {
        Self {
            suite_name,
            results: Vec::new(),
            statistics: SummaryStatistics::default(),
        }
    }

    /// Add a benchmark result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
        self.update_statistics();
    }

    /// Update summary statistics
    fn update_statistics(&mut self) {
        if self.results.is_empty() {
            return;
        }

        let total_ops = self.results.len();
        let avg_throughput =
            self.results.iter().map(|r| r.throughput).sum::<f64>() / total_ops as f64;
        let total_memory = self.results.iter().map(|r| r.memory_usage).sum::<usize>();

        let fastest_op = self
            .results
            .iter()
            .min_by(|a, b| a.min_duration.cmp(&b.min_duration))
            .map(|r| r.operation_name.clone())
            .unwrap_or_default();

        let slowest_op = self
            .results
            .iter()
            .max_by(|a, b| a.max_duration.cmp(&b.max_duration))
            .map(|r| r.operation_name.clone())
            .unwrap_or_default();

        self.statistics = SummaryStatistics {
            total_operations: total_ops,
            avg_throughput,
            total_memory_usage: total_memory,
            fastest_operation: fastest_op,
            slowest_operation: slowest_op,
        };
    }
}

/// Summary statistics for benchmark results
#[derive(Debug, Clone, Default)]
pub struct SummaryStatistics {
    /// Total number of operations benchmarked
    pub total_operations: usize,
    /// Average throughput across all operations
    pub avg_throughput: f64,
    /// Total memory usage across all operations
    pub total_memory_usage: usize,
    /// Name of the fastest operation
    pub fastest_operation: String,
    /// Name of the slowest operation
    pub slowest_operation: String,
}

/// Memory usage measurement for quantization
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Quantization data type
    pub dtype: QuantizedDType,
    /// Number of elements
    pub elements: usize,
    /// Actual bytes used
    pub bytes_used: usize,
    /// Compression ratio vs FP32
    pub compression_ratio: f32,
    /// Memory efficiency (values per byte)
    pub memory_efficiency: f32,
}

/// Collection of memory benchmark results
#[derive(Debug, Clone)]
pub struct MemoryBenchmarkResults {
    /// Individual memory measurements
    pub measurements: Vec<MemoryUsage>,
}

impl MemoryBenchmarkResults {
    /// Create new memory benchmark results
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
        }
    }

    /// Add a memory measurement
    pub fn add_measurement(&mut self, measurement: MemoryUsage) {
        self.measurements.push(measurement);
    }

    /// Get the most memory-efficient configuration
    pub fn most_efficient(&self) -> Option<&MemoryUsage> {
        self.measurements.iter().max_by(|a, b| {
            a.memory_efficiency
                .partial_cmp(&b.memory_efficiency)
                .unwrap()
        })
    }

    /// Get the highest compression ratio
    pub fn highest_compression(&self) -> Option<&MemoryUsage> {
        self.measurements.iter().max_by(|a, b| {
            a.compression_ratio
                .partial_cmp(&b.compression_ratio)
                .unwrap()
        })
    }
}

/// Result of comparing different quantization configurations
#[derive(Debug, Clone)]
pub struct ComparativeBenchmarkResult {
    /// Results for each configuration tested
    pub configurations: Vec<ConfigurationBenchmarkResult>,
}

impl ComparativeBenchmarkResult {
    /// Create new comparative benchmark result
    pub fn new() -> Self {
        Self {
            configurations: Vec::new(),
        }
    }

    /// Add a configuration result
    pub fn add_configuration(&mut self, config: ConfigurationBenchmarkResult) {
        self.configurations.push(config);
    }

    /// Get the fastest configuration for quantization
    pub fn fastest_quantization(&self) -> Option<&ConfigurationBenchmarkResult> {
        self.configurations.iter().min_by(|a, b| {
            a.quantization_performance
                .avg_duration
                .cmp(&b.quantization_performance.avg_duration)
        })
    }

    /// Get the most accurate configuration (lowest MSE)
    pub fn most_accurate(&self) -> Option<&ConfigurationBenchmarkResult> {
        self.configurations
            .iter()
            .min_by(|a, b| a.accuracy_mse.partial_cmp(&b.accuracy_mse).unwrap())
    }

    /// Get the best compression ratio
    pub fn best_compression(&self) -> Option<&ConfigurationBenchmarkResult> {
        self.configurations
            .iter()
            .max_by(|a, b| a.compression_ratio.cmp(&b.compression_ratio))
    }
}

/// Benchmark result for a specific quantization configuration
#[derive(Debug, Clone)]
pub struct ConfigurationBenchmarkResult {
    /// Configuration name
    pub name: String,
    /// Quantization parameters used
    pub params: QuantizationParams,
    /// Quantization performance
    pub quantization_performance: BenchmarkResult,
    /// Dequantization performance
    pub dequantization_performance: BenchmarkResult,
    /// Accuracy (MSE)
    pub accuracy_mse: f32,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Compression ratio vs original data
    pub compression_ratio: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::ops::CpuQuantizationOps;

    #[test]
    fn test_benchmark_suite_creation() {
        let device = Device::cpu().unwrap();
        let config = BenchmarkConfig::default();
        let suite = QuantizationBenchmarkSuite::new(device, config);

        assert!(suite.results.is_empty());
        assert_eq!(suite.config.benchmark_iterations, 10);
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig {
            warmup_iterations: 5,
            benchmark_iterations: 20,
            measure_memory: false,
            measure_accuracy: false,
            operation_timeout_ms: 5000,
        };

        assert_eq!(config.warmup_iterations, 5);
        assert_eq!(config.benchmark_iterations, 20);
        assert!(!config.measure_memory);
    }

    #[test]
    fn test_memory_usage_calculation() {
        // Test different data types
        assert_eq!(
            QuantizationBenchmarkSuite::calculate_quantized_size(1000, &QuantizedDType::Int8),
            1000
        );
        assert_eq!(
            QuantizationBenchmarkSuite::calculate_quantized_size(1000, &QuantizedDType::Int4),
            500
        );
        assert_eq!(
            QuantizationBenchmarkSuite::calculate_quantized_size(1000, &QuantizedDType::Binary),
            125
        );
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let ratio_int8 =
            QuantizationBenchmarkSuite::calculate_compression_ratio(1000, &QuantizedDType::Int8);
        assert_eq!(ratio_int8, 4.0); // FP32 is 4x larger than INT8

        let ratio_int4 =
            QuantizationBenchmarkSuite::calculate_compression_ratio(1000, &QuantizedDType::Int4);
        assert_eq!(ratio_int4, 8.0); // FP32 is 8x larger than INT4

        let ratio_binary =
            QuantizationBenchmarkSuite::calculate_compression_ratio(1000, &QuantizedDType::Binary);
        assert_eq!(ratio_binary, 32.0); // FP32 is 32x larger than binary
    }

    #[test]
    fn test_memory_efficiency_calculation() {
        assert_eq!(
            QuantizationBenchmarkSuite::calculate_memory_efficiency(&QuantizedDType::Int8),
            1.0
        );
        assert_eq!(
            QuantizationBenchmarkSuite::calculate_memory_efficiency(&QuantizedDType::Int4),
            2.0
        );
        assert_eq!(
            QuantizationBenchmarkSuite::calculate_memory_efficiency(&QuantizedDType::Binary),
            8.0
        );
    }

    #[test]
    fn test_mse_calculation() {
        let original = vec![1.0, 2.0, 3.0, 4.0];
        let reconstructed = vec![1.1, 1.9, 3.1, 3.9];

        let mse = QuantizationBenchmarkSuite::calculate_mse(&original, &reconstructed);
        assert!(mse > 0.0 && mse < 1.0); // Should be small but non-zero

        // Test perfect reconstruction
        let mse_perfect = QuantizationBenchmarkSuite::calculate_mse(&original, &original);
        assert_eq!(mse_perfect, 0.0);

        // Test mismatched lengths
        let short_vec = vec![1.0, 2.0];
        let mse_mismatch = QuantizationBenchmarkSuite::calculate_mse(&original, &short_vec);
        assert!(mse_mismatch.is_infinite());
    }

    #[test]
    fn test_throughput_calculation() {
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(150),
        ];

        let throughput = QuantizationBenchmarkSuite::calculate_throughput(&durations);
        assert!(throughput > 0.0);

        // Test empty durations
        let empty_durations = vec![];
        let empty_throughput = QuantizationBenchmarkSuite::calculate_throughput(&empty_durations);
        assert_eq!(empty_throughput, 0.0);
    }

    #[test]
    fn test_benchmark_result_methods() {
        let result = BenchmarkResult {
            operation_name: "test_op".to_string(),
            avg_duration: Duration::from_millis(150),
            min_duration: Duration::from_millis(100),
            max_duration: Duration::from_millis(200),
            throughput: 6.67,
            memory_usage: 1024,
            iterations: 10,
        };

        assert_eq!(result.avg_duration_ms(), 150.0);
        assert_eq!(result.min_duration_ms(), 100.0);
        assert_eq!(result.max_duration_ms(), 200.0);
    }

    #[test]
    fn test_benchmark_summary() {
        let mut summary = BenchmarkSummary::new("test_suite".to_string());
        assert_eq!(summary.suite_name, "test_suite");
        assert!(summary.results.is_empty());

        let result = BenchmarkResult {
            operation_name: "op1".to_string(),
            avg_duration: Duration::from_millis(100),
            min_duration: Duration::from_millis(80),
            max_duration: Duration::from_millis(120),
            throughput: 10.0,
            memory_usage: 512,
            iterations: 5,
        };

        summary.add_result(result);
        assert_eq!(summary.results.len(), 1);
        assert_eq!(summary.statistics.total_operations, 1);
    }

    #[test]
    fn test_memory_benchmark_results() {
        let mut results = MemoryBenchmarkResults::new();
        assert!(results.measurements.is_empty());

        let measurement1 = MemoryUsage {
            dtype: QuantizedDType::Int8,
            elements: 1000,
            bytes_used: 1000,
            compression_ratio: 4.0,
            memory_efficiency: 1.0,
        };

        let measurement2 = MemoryUsage {
            dtype: QuantizedDType::Int4,
            elements: 1000,
            bytes_used: 500,
            compression_ratio: 8.0,
            memory_efficiency: 2.0,
        };

        results.add_measurement(measurement1);
        results.add_measurement(measurement2);

        assert_eq!(results.measurements.len(), 2);

        // Test finding most efficient
        let most_efficient = results.most_efficient();
        assert!(most_efficient.is_some());
        assert_eq!(most_efficient.unwrap().dtype, QuantizedDType::Int4);

        // Test finding highest compression
        let highest_compression = results.highest_compression();
        assert!(highest_compression.is_some());
        assert_eq!(highest_compression.unwrap().compression_ratio, 8.0);
    }

    #[test]
    fn test_comparative_benchmark_result() {
        let mut result = ComparativeBenchmarkResult::new();
        assert!(result.configurations.is_empty());

        let config_result = ConfigurationBenchmarkResult {
            name: "int8_config".to_string(),
            params: QuantizationParams::int8_symmetric(),
            quantization_performance: BenchmarkResult {
                operation_name: "quantize".to_string(),
                avg_duration: Duration::from_millis(100),
                min_duration: Duration::from_millis(90),
                max_duration: Duration::from_millis(110),
                throughput: 10.0,
                memory_usage: 1000,
                iterations: 5,
            },
            dequantization_performance: BenchmarkResult {
                operation_name: "dequantize".to_string(),
                avg_duration: Duration::from_millis(80),
                min_duration: Duration::from_millis(70),
                max_duration: Duration::from_millis(90),
                throughput: 12.5,
                memory_usage: 1000,
                iterations: 5,
            },
            accuracy_mse: 0.01,
            memory_usage: 1000,
            compression_ratio: 4,
        };

        result.add_configuration(config_result);
        assert_eq!(result.configurations.len(), 1);

        // Test analysis methods
        let fastest = result.fastest_quantization();
        assert!(fastest.is_some());

        let most_accurate = result.most_accurate();
        assert!(most_accurate.is_some());

        let best_compression = result.best_compression();
        assert!(best_compression.is_some());
    }

    #[test]
    fn test_quantization_ops_benchmark() {
        let device = Device::cpu().unwrap();
        let config = BenchmarkConfig {
            warmup_iterations: 1,
            benchmark_iterations: 2,
            measure_memory: true,
            measure_accuracy: true,
            operation_timeout_ms: 1000,
        };

        let mut suite = QuantizationBenchmarkSuite::new(device.clone(), config);
        let ops = CpuQuantizationOps::new();

        // This test just ensures the benchmark runs without errors
        let result = suite.benchmark_quantization_ops(&ops);
        assert!(result.is_ok());

        let summary = result.unwrap();
        assert!(!summary.results.is_empty());
    }
}
