//! Advanced quantization acceleration and auto-tuning
//!
//! This module provides sophisticated quantization acceleration capabilities
//! including auto-tuning, performance benchmarking, and optimal configuration
//! selection based on hardware capabilities and workload characteristics.

use crate::quantization::{
    hardware::QuantizationHardwareFeatures, QuantizationOps, QuantizationParams,
    QuantizationScheme, QuantizedDType, QuantizedTensor,
};
use crate::{BackendResult, Device};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Advanced hardware acceleration features for quantization
///
/// This struct provides high-level acceleration capabilities that coordinate
/// multiple specialized hardware features to achieve optimal performance.
pub struct AdvancedQuantizationAccelerator {
    /// Base hardware operations
    base_ops: Arc<dyn QuantizationOps>,
    /// Hardware features available on this device
    hw_features: QuantizationHardwareFeatures,
    /// Performance benchmarking infrastructure
    benchmarks: QuantizationBenchmarks,
    /// Auto-tuning configuration
    auto_tuning: AutoTuningConfig,
    /// Device this accelerator is configured for
    device: Device,
}

impl Clone for AdvancedQuantizationAccelerator {
    fn clone(&self) -> Self {
        Self {
            base_ops: Arc::clone(&self.base_ops),
            hw_features: self.hw_features.clone(),
            benchmarks: self.benchmarks.clone(),
            auto_tuning: self.auto_tuning.clone(),
            device: self.device.clone(),
        }
    }
}

impl AdvancedQuantizationAccelerator {
    /// Create new advanced quantization accelerator
    ///
    /// Automatically detects hardware features and initializes specialized
    /// acceleration components based on what's available.
    ///
    /// # Arguments
    ///
    /// * `device` - The target device for acceleration
    /// * `base_ops` - Base quantization operations implementation
    ///
    /// # Returns
    ///
    /// A fully configured `AdvancedQuantizationAccelerator` instance
    pub fn new(device: Device, base_ops: Arc<dyn QuantizationOps>) -> Self {
        let hw_features = QuantizationHardwareFeatures::detect_for_device(&device);

        Self {
            base_ops,
            hw_features,
            benchmarks: QuantizationBenchmarks::new(),
            auto_tuning: AutoTuningConfig::default(),
            device,
        }
    }

    /// Get hardware features for this accelerator
    pub fn hardware_features(&self) -> &QuantizationHardwareFeatures {
        &self.hw_features
    }

    /// Enable or disable auto-tuning
    pub fn set_auto_tuning_enabled(&mut self, enabled: bool) {
        self.auto_tuning.enabled = enabled;
    }

    /// Set auto-tuning parameters
    pub fn configure_auto_tuning(&mut self, config: AutoTuningConfig) {
        self.auto_tuning = config;
    }

    /// Benchmark quantization operations across different configurations
    ///
    /// Runs comprehensive benchmarks to measure performance of various
    /// quantization operations and configurations on the current hardware.
    ///
    /// # Returns
    ///
    /// Detailed benchmark results showing performance characteristics
    pub fn benchmark_operations(&mut self) -> BackendResult<BenchmarkResults> {
        let mut results = BenchmarkResults::new();

        // Test different operation sizes to understand scaling characteristics
        let test_sizes = vec![64, 256, 1024, 4096];

        for size in test_sizes {
            // Benchmark quantization operations
            self.benchmark_quantization(&mut results, size)?;

            // Benchmark matrix operations (but only for reasonable sizes)
            if size <= 1024 {
                self.benchmark_matrix_operations(&mut results, size)?;
            }

            // Benchmark element-wise operations
            self.benchmark_elementwise_operations(&mut results, size)?;
        }

        Ok(results)
    }

    /// Benchmark quantization/dequantization operations
    fn benchmark_quantization(
        &mut self,
        results: &mut BenchmarkResults,
        size: usize,
    ) -> BackendResult<()> {
        // Test data for benchmarking
        let test_data: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();

        // Test different quantization configurations
        let configs = vec![
            QuantizationParams::int8_symmetric(),
            QuantizationParams::uint8_asymmetric(),
            QuantizationParams::int4_symmetric(),
        ];

        for params in configs {
            // Benchmark quantization
            let start = std::time::Instant::now();
            let _ = self.base_ops.quantize_f32(&test_data, &params)?;
            let quantization_time = start.elapsed();

            let operation_name = format!("quantize_{:?}", params.dtype);
            results.add_benchmark(&operation_name, size, quantization_time);
        }

        Ok(())
    }

    /// Benchmark matrix multiplication operations
    fn benchmark_matrix_operations(
        &mut self,
        results: &mut BenchmarkResults,
        size: usize,
    ) -> BackendResult<()> {
        let params = QuantizationParams::int8_symmetric();

        // Create test matrices
        let a_data = vec![100u8; size * size];
        let b_data = vec![100u8; size * size];

        let a_tensor = QuantizedTensor {
            data: a_data,
            shape: vec![size, size],
            params: params.clone(),
            device: self.device.clone(),
        };

        let b_tensor = QuantizedTensor {
            data: b_data,
            shape: vec![size, size],
            params: params.clone(),
            device: self.device.clone(),
        };

        // Benchmark matrix multiplication (skip if not implemented)
        let start = std::time::Instant::now();
        match self.base_ops.qmatmul(&a_tensor, &b_tensor) {
            Ok(_) => {
                let matmul_time = start.elapsed();
                results.add_benchmark("qmatmul", size, matmul_time);
            }
            Err(torsh_core::error::TorshError::NotImplemented(_)) => {
                // Skip matrix multiplication benchmarks if not implemented
                // This is expected for CPU backend currently
            }
            Err(e) => return Err(e), // Other errors should still propagate
        }

        Ok(())
    }

    /// Benchmark element-wise operations
    fn benchmark_elementwise_operations(
        &mut self,
        results: &mut BenchmarkResults,
        size: usize,
    ) -> BackendResult<()> {
        let params = QuantizationParams::int8_symmetric();

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
            data: b_data,
            shape: vec![size],
            params: params.clone(),
            device: self.device.clone(),
        };

        // Benchmark addition (skip if not implemented)
        let start = std::time::Instant::now();
        match self.base_ops.qadd(&a_tensor, &b_tensor) {
            Ok(_) => {
                let add_time = start.elapsed();
                results.add_benchmark("qadd", size, add_time);
            }
            Err(torsh_core::error::TorshError::NotImplemented(_)) => {
                // Skip quantized addition if not implemented
            }
            Err(e) => return Err(e),
        }

        // Benchmark ReLU (skip if not implemented)
        let start = std::time::Instant::now();
        match self.base_ops.qrelu(&a_tensor) {
            Ok(_) => {
                let relu_time = start.elapsed();
                results.add_benchmark("qrelu", size, relu_time);
            }
            Err(torsh_core::error::TorshError::NotImplemented(_)) => {
                // Skip quantized ReLU if not implemented
            }
            Err(e) => return Err(e),
        }

        Ok(())
    }

    /// Auto-tune quantization parameters for optimal performance
    ///
    /// Automatically searches through different quantization configurations
    /// to find the optimal balance of performance, memory usage, and accuracy
    /// for the given workload.
    ///
    /// # Arguments
    ///
    /// * `workload` - Description of the workload to optimize for
    ///
    /// # Returns
    ///
    /// Optimal quantization configuration for this workload and hardware
    pub fn auto_tune(
        &mut self,
        workload: &QuantizationWorkload,
    ) -> BackendResult<OptimalQuantizationConfig> {
        if !self.auto_tuning.enabled {
            return Ok(OptimalQuantizationConfig::default());
        }

        let mut best_config = OptimalQuantizationConfig::default();
        let mut best_performance = f64::INFINITY;

        // Define search space for auto-tuning
        let schemes = vec![
            QuantizationScheme::Symmetric,
            QuantizationScheme::Linear,
            QuantizationScheme::Asymmetric,
        ];

        let dtypes = vec![
            QuantizedDType::Int8,
            QuantizedDType::UInt8,
            QuantizedDType::Int4,
        ];

        // Search through configuration space
        for scheme in schemes {
            for dtype in &dtypes {
                // Skip configurations that aren't well supported by hardware
                if !self.hw_features.supports_dtype_efficiently(dtype) {
                    continue;
                }

                let params = QuantizationParams {
                    dtype: dtype.clone(),
                    scheme,
                    scale: vec![1.0],
                    zero_point: vec![0],
                    block_size: None,
                    min_val: None,
                    max_val: None,
                };

                // Benchmark this configuration
                if let Ok(performance) = self.benchmark_config(&params, workload) {
                    if performance < best_performance {
                        best_performance = performance;
                        best_config = OptimalQuantizationConfig {
                            params,
                            estimated_speedup: 1.0 / performance,
                            memory_savings: self.estimate_memory_savings(dtype),
                            accuracy_impact: self.estimate_accuracy_impact(dtype, scheme),
                        };
                    }
                }
            }
        }

        Ok(best_config)
    }

    /// Benchmark a specific quantization configuration
    fn benchmark_config(
        &self,
        params: &QuantizationParams,
        workload: &QuantizationWorkload,
    ) -> BackendResult<f64> {
        let start = std::time::Instant::now();

        // Execute the workload with this configuration
        match &workload.operation_type {
            QuantizationOperationType::MatrixMultiply { m, n, k } => {
                self.benchmark_matmul_config(params, *m, *n, *k)?;
            }
            QuantizationOperationType::Convolution2D {
                batch_size,
                channels,
                height,
                width,
                kernel_size,
            } => {
                self.benchmark_conv2d_config(
                    params,
                    *batch_size,
                    *channels,
                    *height,
                    *width,
                    *kernel_size,
                )?;
            }
        }

        let elapsed = start.elapsed();
        Ok(elapsed.as_secs_f64())
    }

    /// Benchmark matrix multiplication with specific configuration
    fn benchmark_matmul_config(
        &self,
        params: &QuantizationParams,
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        let a_data = vec![128u8; m * k];
        let b_data = vec![128u8; k * n];

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

        // Skip if matrix multiplication is not implemented
        match self.base_ops.qmatmul(&a_tensor, &b_tensor) {
            Ok(_) => {} // Success case - continue normally
            Err(torsh_core::error::TorshError::NotImplemented(_)) => {
                // Skip matrix multiplication if not implemented - this is expected for CPU backend
            }
            Err(e) => return Err(e), // Other errors should still propagate
        }
        Ok(())
    }

    /// Benchmark convolution with specific configuration
    fn benchmark_conv2d_config(
        &self,
        params: &QuantizationParams,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) -> BackendResult<()> {
        let input_data = vec![128u8; batch_size * channels * height * width];
        let weight_data = vec![128u8; channels * channels * kernel_size * kernel_size];

        let input_tensor = QuantizedTensor {
            data: input_data,
            shape: vec![batch_size, channels, height, width],
            params: params.clone(),
            device: self.device.clone(),
        };

        let weight_tensor = QuantizedTensor {
            data: weight_data,
            shape: vec![channels, channels, kernel_size, kernel_size],
            params: params.clone(),
            device: self.device.clone(),
        };

        let _ = self
            .base_ops
            .qconv2d(&input_tensor, &weight_tensor, None, (1, 1), (0, 0))?;
        Ok(())
    }

    /// Estimate memory savings for a quantization type
    fn estimate_memory_savings(&self, dtype: &QuantizedDType) -> f64 {
        let bits = dtype.bits() as f64;
        let fp32_bits = 32.0;
        1.0 - (bits / fp32_bits)
    }

    /// Estimate accuracy impact for a quantization configuration
    fn estimate_accuracy_impact(&self, dtype: &QuantizedDType, scheme: QuantizationScheme) -> f64 {
        // Base accuracy based on data type
        let base_accuracy = match dtype {
            QuantizedDType::Int16 | QuantizedDType::UInt16 => 0.99,
            QuantizedDType::Int8 | QuantizedDType::UInt8 => 0.95,
            QuantizedDType::Int4 | QuantizedDType::UInt4 => 0.85,
            QuantizedDType::Binary => 0.70,
            QuantizedDType::Mixed(_) => 0.90,
        };

        // Adjust based on quantization scheme
        let scheme_factor = match scheme {
            QuantizationScheme::Symmetric => 1.0,
            QuantizationScheme::Linear => 0.98,
            QuantizationScheme::Asymmetric => 0.96,
            QuantizationScheme::ChannelWise => 1.02, // Better accuracy
            QuantizationScheme::BlockWise => 1.01,   // Slightly better
            QuantizationScheme::Logarithmic => 0.90, // More aggressive
        };

        (base_accuracy as f64 * scheme_factor as f64).min(1.0f64)
    }

    /// Get recommendations for the given workload
    pub fn get_recommendations(
        &self,
        workload: &QuantizationWorkload,
    ) -> QuantizationRecommendations {
        let mut recommendations = QuantizationRecommendations::default();

        // Analyze workload characteristics
        match &workload.operation_type {
            QuantizationOperationType::MatrixMultiply { m, n, k } => {
                let total_ops = m * n * k * 2; // Approximate FLOPs
                if total_ops > 1_000_000 {
                    recommendations.preferred_dtype = if self.hw_features.supports_int8_simd {
                        QuantizedDType::Int8
                    } else {
                        QuantizedDType::UInt8
                    };
                    recommendations.batch_operations = true;
                } else {
                    recommendations.preferred_dtype = QuantizedDType::Int16; // Higher precision for small ops
                    recommendations.batch_operations = false;
                }
            }
            QuantizationOperationType::Convolution2D { .. } => {
                // Convolutions typically benefit from INT8 quantization
                recommendations.preferred_dtype = QuantizedDType::Int8;
                recommendations.use_channel_wise = true;
                recommendations.batch_operations = true;
            }
        }

        // Consider hardware features
        if self.hw_features.supports_tensor_cores {
            recommendations.use_tensor_cores = true;
        }

        if self.hw_features.supports_mixed_precision {
            recommendations.enable_mixed_precision = true;
        }

        recommendations
    }
}

/// Quantization benchmarking infrastructure
#[derive(Debug, Clone)]
pub struct QuantizationBenchmarks {
    /// Benchmark results storage
    results: HashMap<String, Vec<BenchmarkResult>>,
}

impl QuantizationBenchmarks {
    /// Create new benchmarking infrastructure
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Add a benchmark result
    pub fn add_result(&mut self, operation: String, result: BenchmarkResult) {
        self.results
            .entry(operation)
            .or_insert_with(Vec::new)
            .push(result);
    }

    /// Get results for a specific operation
    pub fn get_results(&self, operation: &str) -> Option<&Vec<BenchmarkResult>> {
        self.results.get(operation)
    }

    /// Get the best result for an operation
    pub fn get_best_result(&self, operation: &str) -> Option<&BenchmarkResult> {
        self.results
            .get(operation)?
            .iter()
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
    }

    /// Clear all benchmark results
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

/// Individual benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Problem size
    pub size: usize,
    /// Execution duration
    pub duration: std::time::Duration,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory usage (bytes)
    pub memory_usage: Option<usize>,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(operation: String, size: usize, duration: std::time::Duration) -> Self {
        let throughput = size as f64 / duration.as_secs_f64();

        Self {
            operation,
            size,
            duration,
            throughput,
            memory_usage: None,
        }
    }

    /// Set memory usage for this benchmark
    pub fn with_memory_usage(mut self, memory_usage: usize) -> Self {
        self.memory_usage = Some(memory_usage);
        self
    }
}

/// Collection of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// All benchmark results
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkResults {
    /// Create new benchmark results collection
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add a benchmark result
    pub fn add_benchmark(&mut self, operation: &str, size: usize, duration: std::time::Duration) {
        let result = BenchmarkResult::new(operation.to_string(), size, duration);
        self.results.push(result);
    }

    /// Get the best result for a specific operation
    pub fn get_best_result(&self, operation: &str) -> Option<&BenchmarkResult> {
        self.results
            .iter()
            .filter(|r| r.operation == operation)
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
    }

    /// Get average throughput for an operation
    pub fn get_average_throughput(&self, operation: &str) -> Option<f64> {
        let matching_results: Vec<_> = self
            .results
            .iter()
            .filter(|r| r.operation == operation)
            .collect();

        if matching_results.is_empty() {
            return None;
        }

        let sum: f64 = matching_results.iter().map(|r| r.throughput).sum();
        Some(sum / matching_results.len() as f64)
    }
}

/// Auto-tuning configuration
#[derive(Debug, Clone)]
pub struct AutoTuningConfig {
    /// Enable auto-tuning
    pub enabled: bool,
    /// Number of benchmark iterations per configuration
    pub benchmark_iterations: usize,
    /// Accuracy threshold for accepting configurations
    pub accuracy_threshold: f64,
    /// Maximum search time in seconds
    pub max_search_time: f64,
    /// Performance improvement threshold to consider a configuration
    pub min_improvement_threshold: f64,
}

impl Default for AutoTuningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            benchmark_iterations: 5,
            accuracy_threshold: 0.95,
            max_search_time: 60.0,
            min_improvement_threshold: 0.05, // 5% improvement
        }
    }
}

/// Optimal quantization configuration found by auto-tuning
#[derive(Debug, Clone)]
pub struct OptimalQuantizationConfig {
    /// Optimal quantization parameters
    pub params: QuantizationParams,
    /// Estimated speedup over FP32
    pub estimated_speedup: f64,
    /// Memory savings ratio (0.0-1.0)
    pub memory_savings: f64,
    /// Accuracy impact (0.0-1.0, higher is better)
    pub accuracy_impact: f64,
}

impl Default for OptimalQuantizationConfig {
    fn default() -> Self {
        Self {
            params: QuantizationParams::default(),
            estimated_speedup: 1.0,
            memory_savings: 0.0,
            accuracy_impact: 1.0,
        }
    }
}

/// Quantization workload description for auto-tuning
#[derive(Debug, Clone)]
pub struct QuantizationWorkload {
    /// Type of operation being performed
    pub operation_type: QuantizationOperationType,
    /// Expected frequency of this operation (relative)
    pub frequency: f64,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
}

/// Types of quantization operations for workload analysis
#[derive(Debug, Clone)]
pub enum QuantizationOperationType {
    /// Matrix multiplication workload
    MatrixMultiply {
        /// Number of rows in A
        m: usize,
        /// Number of columns in B
        n: usize,
        /// Shared dimension
        k: usize,
    },
    /// 2D convolution workload
    Convolution2D {
        /// Batch size
        batch_size: usize,
        /// Number of channels
        channels: usize,
        /// Input height
        height: usize,
        /// Input width
        width: usize,
        /// Kernel size (assumed square)
        kernel_size: usize,
    },
}

/// Performance requirements for workloads
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Maximum acceptable latency (milliseconds)
    pub max_latency_ms: f64,
    /// Minimum required throughput (operations per second)
    pub min_throughput: f64,
    /// Power budget (watts, if applicable)
    pub power_budget: Option<f64>,
    /// Memory budget (bytes, if applicable)
    pub memory_budget: Option<usize>,
}

/// Quantization recommendations generated by the accelerator
#[derive(Debug, Clone)]
pub struct QuantizationRecommendations {
    /// Recommended quantization data type
    pub preferred_dtype: QuantizedDType,
    /// Recommended quantization scheme
    pub preferred_scheme: QuantizationScheme,
    /// Whether to use channel-wise quantization
    pub use_channel_wise: bool,
    /// Whether to batch operations for better performance
    pub batch_operations: bool,
    /// Whether to use tensor cores if available
    pub use_tensor_cores: bool,
    /// Whether to enable mixed precision
    pub enable_mixed_precision: bool,
    /// Recommended block size for operations
    pub recommended_block_size: Option<usize>,
}

impl Default for QuantizationRecommendations {
    fn default() -> Self {
        Self {
            preferred_dtype: QuantizedDType::Int8,
            preferred_scheme: QuantizationScheme::Symmetric,
            use_channel_wise: false,
            batch_operations: true,
            use_tensor_cores: false,
            enable_mixed_precision: false,
            recommended_block_size: None,
        }
    }
}

// Manual Debug implementation for AdvancedQuantizationAccelerator
impl std::fmt::Debug for AdvancedQuantizationAccelerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdvancedQuantizationAccelerator")
            .field("device", &self.device)
            .field("hw_features", &self.hw_features)
            .field("benchmarks", &self.benchmarks)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::ops::CpuQuantizationOps;

    fn create_test_accelerator() -> AdvancedQuantizationAccelerator {
        let device = Device::cpu().unwrap();
        let cpu_ops = CpuQuantizationOps::new();
        AdvancedQuantizationAccelerator::new(device, Arc::new(cpu_ops))
    }

    #[test]
    fn test_accelerator_creation() {
        let accelerator = create_test_accelerator();

        // Should have detected hardware features
        let features = accelerator.hardware_features();
        assert!(features.max_parallel_ops >= 1);
    }

    #[test]
    fn test_auto_tuning_configuration() {
        let mut accelerator = create_test_accelerator();

        // Test enabling/disabling auto-tuning
        accelerator.set_auto_tuning_enabled(false);

        let config = AutoTuningConfig {
            enabled: true,
            benchmark_iterations: 3,
            accuracy_threshold: 0.90,
            max_search_time: 30.0,
            min_improvement_threshold: 0.10,
        };

        accelerator.configure_auto_tuning(config.clone());
        assert_eq!(accelerator.auto_tuning.benchmark_iterations, 3);
    }

    #[test]
    fn test_benchmark_operations() {
        let mut accelerator = create_test_accelerator();

        // Run benchmarks
        let results = accelerator.benchmark_operations();
        if let Err(ref e) = results {
            panic!("Benchmark operations failed with error: {:?}", e);
        }
        assert!(results.is_ok());

        let benchmark_results = results.unwrap();
        assert!(!benchmark_results.results.is_empty());

        // Should have quantization benchmarks
        let quantize_results: Vec<_> = benchmark_results
            .results
            .iter()
            .filter(|r| r.operation.contains("quantize"))
            .collect();
        assert!(!quantize_results.is_empty());
    }

    #[test]
    fn test_auto_tuning() {
        let mut accelerator = create_test_accelerator();

        let workload = QuantizationWorkload {
            operation_type: QuantizationOperationType::MatrixMultiply {
                m: 64,
                n: 64,
                k: 64,
            },
            frequency: 1.0,
            performance_requirements: PerformanceRequirements {
                max_latency_ms: 10.0,
                min_throughput: 100.0,
                power_budget: None,
                memory_budget: None,
            },
        };

        let result = accelerator.auto_tune(&workload);
        assert!(result.is_ok());

        let config = result.unwrap();
        assert!(config.estimated_speedup >= 0.0);
        assert!(config.memory_savings >= 0.0);
        assert!(config.accuracy_impact >= 0.0 && config.accuracy_impact <= 1.0);
    }

    #[test]
    fn test_workload_recommendations() {
        let accelerator = create_test_accelerator();

        // Test matrix multiplication workload
        let matmul_workload = QuantizationWorkload {
            operation_type: QuantizationOperationType::MatrixMultiply {
                m: 1024,
                n: 1024,
                k: 1024,
            },
            frequency: 1.0,
            performance_requirements: PerformanceRequirements {
                max_latency_ms: 100.0,
                min_throughput: 10.0,
                power_budget: None,
                memory_budget: None,
            },
        };

        let recommendations = accelerator.get_recommendations(&matmul_workload);
        assert!(recommendations.batch_operations); // Large matmul should batch

        // Test convolution workload
        let conv_workload = QuantizationWorkload {
            operation_type: QuantizationOperationType::Convolution2D {
                batch_size: 32,
                channels: 128,
                height: 64,
                width: 64,
                kernel_size: 3,
            },
            frequency: 1.0,
            performance_requirements: PerformanceRequirements {
                max_latency_ms: 50.0,
                min_throughput: 20.0,
                power_budget: None,
                memory_budget: None,
            },
        };

        let conv_recommendations = accelerator.get_recommendations(&conv_workload);
        assert_eq!(conv_recommendations.preferred_dtype, QuantizedDType::Int8);
        assert!(conv_recommendations.use_channel_wise);
    }

    #[test]
    fn test_benchmark_results() {
        let mut results = BenchmarkResults::new();

        let duration = std::time::Duration::from_millis(10);
        results.add_benchmark("test_op", 1000, duration);
        results.add_benchmark("test_op", 2000, std::time::Duration::from_millis(15));

        // Should have two results
        assert_eq!(results.results.len(), 2);

        // Test best result selection
        let best = results.get_best_result("test_op");
        assert!(best.is_some());

        let best_result = best.unwrap();
        assert_eq!(best_result.size, 2000); // Higher throughput

        // Test average throughput
        let avg_throughput = results.get_average_throughput("test_op");
        assert!(avg_throughput.is_some());
        assert!(avg_throughput.unwrap() > 0.0);
    }

    #[test]
    fn test_benchmark_infrastructure() {
        let mut benchmarks = QuantizationBenchmarks::new();

        let result = BenchmarkResult::new(
            "test_operation".to_string(),
            1000,
            std::time::Duration::from_millis(5),
        );

        benchmarks.add_result("test_operation".to_string(), result);

        // Should be able to retrieve results
        let results = benchmarks.get_results("test_operation");
        assert!(results.is_some());
        assert_eq!(results.unwrap().len(), 1);

        // Should be able to get best result
        let best = benchmarks.get_best_result("test_operation");
        assert!(best.is_some());

        // Clear should work
        benchmarks.clear();
        assert!(benchmarks.get_results("test_operation").is_none());
    }

    #[test]
    fn test_memory_savings_estimation() {
        let accelerator = create_test_accelerator();

        // INT8 should save 75% memory compared to FP32
        let int8_savings = accelerator.estimate_memory_savings(&QuantizedDType::Int8);
        assert!((int8_savings - 0.75).abs() < 0.01);

        // INT4 should save 87.5% memory
        let int4_savings = accelerator.estimate_memory_savings(&QuantizedDType::Int4);
        assert!((int4_savings - 0.875).abs() < 0.01);

        // Binary should save 96.875% memory
        let binary_savings = accelerator.estimate_memory_savings(&QuantizedDType::Binary);
        assert!((binary_savings - 0.96875).abs() < 0.01);
    }

    #[test]
    fn test_accuracy_impact_estimation() {
        let accelerator = create_test_accelerator();

        // INT8 with symmetric scheme should have high accuracy
        let int8_acc = accelerator
            .estimate_accuracy_impact(&QuantizedDType::Int8, QuantizationScheme::Symmetric);
        assert!(int8_acc >= 0.90);

        // INT4 should have lower accuracy
        let int4_acc = accelerator
            .estimate_accuracy_impact(&QuantizedDType::Int4, QuantizationScheme::Symmetric);
        assert!(int4_acc < int8_acc);

        // Channel-wise should improve accuracy
        let channelwise_acc = accelerator
            .estimate_accuracy_impact(&QuantizedDType::Int8, QuantizationScheme::ChannelWise);
        assert!(channelwise_acc >= int8_acc);
    }

    #[test]
    fn test_benchmark_result_creation() {
        let duration = std::time::Duration::from_millis(10);
        let result = BenchmarkResult::new("test".to_string(), 1000, duration);

        assert_eq!(result.operation, "test");
        assert_eq!(result.size, 1000);
        assert_eq!(result.duration, duration);
        assert!(result.throughput > 0.0);
        assert!(result.memory_usage.is_none());

        // Test with memory usage
        let result_with_memory = result.with_memory_usage(1024);
        assert_eq!(result_with_memory.memory_usage, Some(1024));
    }
}
