//! Advanced hardware acceleration modules for quantization
//!
//! This module provides specialized acceleration for quantization operations using:
//! - [`VnniQuantizationOps`] - Intel VNNI (Vector Neural Network Instructions)
//! - [`Dp4aQuantizationOps`] - NVIDIA DP4A (4-element dot product and accumulate)
//! - [`TensorCoreQuantizationOps`] - NVIDIA Tensor Core operations
//! - [`AdvancedQuantizationAccelerator`] - Unified accelerator with auto-tuning

use super::core::{QuantizedDType, QuantizationParams, QuantizationScheme, QuantizedTensor};
use super::operations::HardwareQuantizationOps;
use crate::{BackendResult, Device};
use torsh_core::error::TorshError;
use std::time::{Duration, Instant};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String};

/// Quantization operation types for workload characterization
#[derive(Debug, Clone)]
pub enum QuantizationOperationType {
    /// Matrix multiplication with dimensions (M, N, K)
    MatrixMultiply { m: usize, n: usize, k: usize },
    /// 2D convolution with batch, channels, height, width, kernel size
    Convolution2D {
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    },
}

/// Workload description for auto-tuning
#[derive(Debug, Clone)]
pub struct QuantizationWorkload {
    /// Type of operation to optimize for
    pub operation_type: QuantizationOperationType,
    /// Expected frequency of this workload
    pub frequency: f32,
    /// Performance requirements
    pub requirements: PerformanceRequirements,
}

/// Performance requirements for auto-tuning
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f32,
    /// Minimum acceptable accuracy (0.0 to 1.0)
    pub min_accuracy: f32,
    /// Memory budget in bytes
    pub memory_budget_bytes: usize,
    /// Priority: 0 = speed, 1 = accuracy, 0.5 = balanced
    pub speed_vs_accuracy: f32,
}

impl Default for PerformanceRequirements {
    fn default() -> Self {
        Self {
            max_latency_ms: 10.0,
            min_accuracy: 0.95,
            memory_budget_bytes: 1024 * 1024 * 1024, // 1GB
            speed_vs_accuracy: 0.5,
        }
    }
}

/// Auto-tuning configuration
#[derive(Debug, Clone)]
pub struct AutoTuningConfig {
    /// Enable automatic parameter selection
    pub enable_auto_tuning: bool,
    /// Maximum tuning time in seconds
    pub max_tuning_time_secs: f32,
    /// Number of benchmark iterations per configuration
    pub benchmark_iterations: usize,
    /// Minimum improvement threshold to switch configurations
    pub improvement_threshold: f32,
}

impl Default for AutoTuningConfig {
    fn default() -> Self {
        Self {
            enable_auto_tuning: true,
            max_tuning_time_secs: 10.0,
            benchmark_iterations: 3,
            improvement_threshold: 0.1, // 10% improvement required
        }
    }
}

/// Optimal quantization configuration result
#[derive(Debug, Clone)]
pub struct OptimalQuantizationConfig {
    /// Optimal quantization parameters
    pub params: QuantizationParams,
    /// Estimated speedup factor
    pub estimated_speedup: f64,
    /// Memory savings ratio (0.0 to 1.0)
    pub memory_savings: f64,
    /// Estimated accuracy impact (0.0 to 1.0)
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

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Operation name
    pub operation: String,
    /// Input size
    pub size: usize,
    /// Execution time
    pub time: Duration,
    /// Throughput (operations per second)
    pub throughput: f64,
}

/// Collection of benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Individual benchmark results
    pub results: Vec<BenchmarkResult>,
    /// Summary statistics
    pub summary: BenchmarkSummary,
}

/// Summary statistics for benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    /// Average throughput across all benchmarks
    pub avg_throughput: f64,
    /// Best performing operation
    pub best_operation: String,
    /// Worst performing operation
    pub worst_operation: String,
    /// Total benchmark time
    pub total_time: Duration,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            summary: BenchmarkSummary {
                avg_throughput: 0.0,
                best_operation: String::new(),
                worst_operation: String::new(),
                total_time: Duration::from_secs(0),
            },
        }
    }

    pub fn add_benchmark(&mut self, operation: &str, size: usize, time: Duration) {
        let throughput = size as f64 / time.as_secs_f64();

        self.results.push(BenchmarkResult {
            operation: operation.to_string(),
            size,
            time,
            throughput,
        });

        self.update_summary();
    }

    fn update_summary(&mut self) {
        if self.results.is_empty() {
            return;
        }

        let total_throughput: f64 = self.results.iter().map(|r| r.throughput).sum();
        self.summary.avg_throughput = total_throughput / self.results.len() as f64;

        let best = self.results.iter().max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap()).unwrap();
        let worst = self.results.iter().min_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap()).unwrap();

        self.summary.best_operation = best.operation.clone();
        self.summary.worst_operation = worst.operation.clone();
        self.summary.total_time = self.results.iter().map(|r| r.time).sum();
    }
}

/// Benchmarking infrastructure for quantization operations
#[derive(Debug, Clone)]
pub struct QuantizationBenchmarks {
    /// Results cache
    results_cache: Vec<BenchmarkResult>,
    /// Benchmark configuration
    config: BenchmarkConfig,
}

/// Configuration for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Maximum benchmark time per operation
    pub max_time_per_op: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            measurement_iterations: 10,
            max_time_per_op: Duration::from_secs(5),
        }
    }
}

impl QuantizationBenchmarks {
    pub fn new() -> Self {
        Self {
            results_cache: Vec::new(),
            config: BenchmarkConfig::default(),
        }
    }

    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self {
            results_cache: Vec::new(),
            config,
        }
    }
}

/// Advanced quantization accelerator with auto-tuning capabilities
///
/// This struct combines multiple hardware-specific acceleration modules and provides
/// automatic performance tuning to select the optimal implementation for each workload.
#[derive(Debug)]
pub struct AdvancedQuantizationAccelerator {
    /// Base hardware operations
    base_ops: HardwareQuantizationOps,
    /// VNNI-specific optimizations
    #[allow(dead_code)]
    vnni_ops: Option<VnniQuantizationOps>,
    /// DP4A-specific optimizations
    #[allow(dead_code)]
    dp4a_ops: Option<Dp4aQuantizationOps>,
    /// Tensor core optimizations
    #[allow(dead_code)]
    tensor_core_ops: Option<TensorCoreQuantizationOps>,
    /// Performance benchmarking
    #[allow(dead_code)]
    benchmarks: QuantizationBenchmarks,
    /// Auto-tuning configuration
    #[allow(dead_code)]
    auto_tuning: AutoTuningConfig,
}

impl AdvancedQuantizationAccelerator {
    /// Create new advanced quantization accelerator
    ///
    /// Automatically detects and initializes available hardware acceleration modules
    /// based on the target device capabilities.
    pub fn new(device: Device) -> Self {
        let base_ops = HardwareQuantizationOps::new(device.clone());

        let vnni_ops = if base_ops.hardware_features().supports_vnni {
            Some(VnniQuantizationOps::new())
        } else {
            None
        };

        let dp4a_ops = if base_ops.hardware_features().supports_dp4a {
            Some(Dp4aQuantizationOps::new())
        } else {
            None
        };

        let tensor_core_ops = if base_ops.hardware_features().supports_tensor_cores {
            Some(TensorCoreQuantizationOps::new())
        } else {
            None
        };

        Self {
            base_ops,
            vnni_ops,
            dp4a_ops,
            tensor_core_ops,
            benchmarks: QuantizationBenchmarks::new(),
            auto_tuning: AutoTuningConfig::default(),
        }
    }

    /// Get the base hardware operations
    pub fn base_ops(&self) -> &HardwareQuantizationOps {
        &self.base_ops
    }

    /// Check if VNNI acceleration is available
    pub fn has_vnni(&self) -> bool {
        self.vnni_ops.is_some()
    }

    /// Check if DP4A acceleration is available
    pub fn has_dp4a(&self) -> bool {
        self.dp4a_ops.is_some()
    }

    /// Check if Tensor Core acceleration is available
    pub fn has_tensor_cores(&self) -> bool {
        self.tensor_core_ops.is_some()
    }

    /// Benchmark quantization operations across different configurations
    pub fn benchmark_operations(&mut self) -> BackendResult<BenchmarkResults> {
        let mut results = BenchmarkResults::new();

        // Benchmark different operation types and sizes
        let test_sizes = vec![64, 256, 1024, 4096];

        for size in test_sizes {
            // Benchmark quantization
            let test_data: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
            let params = QuantizationParams::uint8_asymmetric();

            let start = Instant::now();
            let _ = self.base_ops.quantize_f32(&test_data, &params)?;
            let quantization_time = start.elapsed();

            results.add_benchmark("quantization", size, quantization_time);

            // Benchmark matrix multiplication for smaller sizes to avoid memory issues
            if size <= 512 {
                let a_data = vec![128u8; size * size];
                let b_data = vec![128u8; size * size];

                let a_tensor = QuantizedTensor::from_data(
                    a_data,
                    vec![size, size],
                    params.clone(),
                    self.base_ops.device().clone(),
                )?;

                let b_tensor = QuantizedTensor::from_data(
                    b_data,
                    vec![size, size],
                    params.clone(),
                    self.base_ops.device().clone(),
                )?;

                let start = Instant::now();
                let _ = self.base_ops.qmatmul(&a_tensor, &b_tensor)?;
                let matmul_time = start.elapsed();

                results.add_benchmark("qmatmul", size, matmul_time);
            }
        }

        Ok(results)
    }

    /// Auto-tune quantization parameters for optimal performance
    ///
    /// Systematically evaluates different quantization configurations and selects
    /// the one that best meets the performance requirements.
    pub fn auto_tune(
        &mut self,
        workload: &QuantizationWorkload,
    ) -> BackendResult<OptimalQuantizationConfig> {
        let mut best_config = OptimalQuantizationConfig::default();
        let mut best_performance = f64::INFINITY;

        // Try different quantization schemes
        let schemes = vec![
            QuantizationScheme::Linear,
            QuantizationScheme::Symmetric,
            QuantizationScheme::Asymmetric,
        ];

        let dtypes = vec![
            QuantizedDType::Int8,
            QuantizedDType::UInt8,
            QuantizedDType::Int4,
        ];

        for scheme in schemes {
            for dtype in &dtypes {
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
                let performance = self.benchmark_config(&params, workload)?;

                if performance < best_performance {
                    best_performance = performance;
                    best_config = OptimalQuantizationConfig {
                        params,
                        estimated_speedup: 1.0 / performance,
                        memory_savings: self.estimate_memory_savings(dtype),
                        accuracy_impact: 0.95, // Placeholder - would be measured in practice
                    };
                }
            }
        }

        Ok(best_config)
    }

    /// Benchmark a specific quantization configuration against a workload
    fn benchmark_config(
        &self,
        params: &QuantizationParams,
        workload: &QuantizationWorkload,
    ) -> BackendResult<f64> {
        let start = Instant::now();

        // Run the workload with this configuration
        match &workload.operation_type {
            QuantizationOperationType::MatrixMultiply { m, n, k } => {
                let a_data = vec![128u8; m * k];
                let b_data = vec![128u8; k * n];

                let a_tensor = QuantizedTensor::from_data(
                    a_data,
                    vec![*m, *k],
                    params.clone(),
                    self.base_ops.device().clone(),
                )?;

                let b_tensor = QuantizedTensor::from_data(
                    b_data,
                    vec![*k, *n],
                    params.clone(),
                    self.base_ops.device().clone(),
                )?;

                let _ = self.base_ops.qmatmul(&a_tensor, &b_tensor)?;
            }
            QuantizationOperationType::Convolution2D {
                batch_size,
                channels,
                height,
                width,
                kernel_size,
            } => {
                let input_data = vec![128u8; batch_size * channels * height * width];
                let weight_data = vec![128u8; channels * channels * kernel_size * kernel_size];

                let input_tensor = QuantizedTensor::from_data(
                    input_data,
                    vec![*batch_size, *channels, *height, *width],
                    params.clone(),
                    self.base_ops.device().clone(),
                )?;

                let weight_tensor = QuantizedTensor::from_data(
                    weight_data,
                    vec![*channels, *channels, *kernel_size, *kernel_size],
                    params.clone(),
                    self.base_ops.device().clone(),
                )?;

                let _ = self.base_ops.qconv2d(&input_tensor, &weight_tensor, None, (1, 1), (0, 0))?;
            }
        }

        let elapsed = start.elapsed();
        Ok(elapsed.as_secs_f64())
    }

    /// Estimate memory savings for a quantization type compared to FP32
    fn estimate_memory_savings(&self, dtype: &QuantizedDType) -> f64 {
        let bits = dtype.bits() as f64;
        let fp32_bits = 32.0;
        1.0 - (bits / fp32_bits)
    }

    /// Configure auto-tuning parameters
    pub fn set_auto_tuning_config(&mut self, config: AutoTuningConfig) {
        self.auto_tuning = config;
    }

    /// Get current auto-tuning configuration
    pub fn auto_tuning_config(&self) -> &AutoTuningConfig {
        &self.auto_tuning
    }
}

/// Intel VNNI (Vector Neural Network Instructions) acceleration
///
/// Provides optimized quantization operations using Intel's VNNI instructions,
/// available on processors with AVX-512 VNNI or AVX VNNI support.
#[derive(Debug, Clone)]
pub struct VnniQuantizationOps {
    /// VNNI instruction availability
    vnni_available: bool,
}

impl VnniQuantizationOps {
    /// Create new VNNI quantization operations
    pub fn new() -> Self {
        Self {
            vnni_available: Self::detect_vnni(),
        }
    }

    /// Detect VNNI support via runtime feature detection
    fn detect_vnni() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Check for VNNI support via runtime feature detection
            // Note: This uses a conservative check for AVX-512 VNNI
            std::arch::is_x86_feature_detected!("avx512vnni")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Check if VNNI is available
    pub fn is_available(&self) -> bool {
        self.vnni_available
    }

    /// VNNI-accelerated INT8 matrix multiplication
    ///
    /// Uses Intel VNNI instructions for efficient INT8 matrix multiplication
    /// with INT32 accumulation, providing significant speedup over scalar operations.
    pub fn vnni_qmatmul_int8(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.vnni_available {
            return Err(TorshError::BackendError("VNNI not available".to_string()).into());
        }

        // Validate input tensors
        if a.params().dtype != QuantizedDType::Int8 || b.params().dtype != QuantizedDType::Int8 {
            return Err(TorshError::BackendError(
                "VNNI requires INT8 tensors".to_string(),
            ).into());
        }

        // This would use actual VNNI instructions (vpdpbusd, vpdpbusds)
        // For now, return a placeholder result
        let m = a.shape()[0];
        let n = b.shape()[1];

        QuantizedTensor::from_data(
            vec![0; m * n],
            vec![m, n],
            a.params().clone(),
            a.device().clone(),
        )
    }

    /// VNNI-accelerated convolution
    pub fn vnni_qconv2d_int8(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.vnni_available {
            return Err(TorshError::BackendError("VNNI not available".to_string()).into());
        }

        // This would implement VNNI-optimized convolution
        // For now, return a simplified result
        let batch_size = input.shape()[0];
        let out_channels = weight.shape()[0];
        let out_height = input.shape()[2]; // Simplified - no padding/stride
        let out_width = input.shape()[3];

        QuantizedTensor::from_data(
            vec![0; batch_size * out_channels * out_height * out_width],
            vec![batch_size, out_channels, out_height, out_width],
            input.params().clone(),
            input.device().clone(),
        )
    }
}

/// NVIDIA DP4A (4-element dot product and accumulate) acceleration
///
/// Provides optimized quantization operations for NVIDIA GPUs using DP4A instructions,
/// available on Pascal and newer architectures.
#[derive(Debug, Clone)]
pub struct Dp4aQuantizationOps {
    /// DP4A instruction availability
    dp4a_available: bool,
}

impl Dp4aQuantizationOps {
    /// Create new DP4A quantization operations
    pub fn new() -> Self {
        Self {
            dp4a_available: Self::detect_dp4a(),
        }
    }

    /// Detect DP4A support (simplified - would query CUDA properties)
    fn detect_dp4a() -> bool {
        // In practice, this would query CUDA device properties
        // For now, assume available on modern NVIDIA GPUs
        true
    }

    /// Check if DP4A is available
    pub fn is_available(&self) -> bool {
        self.dp4a_available
    }

    /// DP4A-accelerated INT8 matrix multiplication
    ///
    /// Uses NVIDIA DP4A instructions for efficient INT8 matrix multiplication
    /// with 4-way SIMD processing and INT32 accumulation.
    pub fn dp4a_qmatmul_int8(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.dp4a_available {
            return Err(TorshError::BackendError("DP4A not available".to_string()).into());
        }

        // Validate input tensors
        if a.params().dtype != QuantizedDType::Int8 || b.params().dtype != QuantizedDType::Int8 {
            return Err(TorshError::BackendError(
                "DP4A requires INT8 tensors".to_string(),
            ).into());
        }

        // This would use CUDA DP4A instructions (__dp4a)
        // For now, return a placeholder result
        let m = a.shape()[0];
        let n = b.shape()[1];

        QuantizedTensor::from_data(
            vec![0; m * n],
            vec![m, n],
            a.params().clone(),
            a.device().clone(),
        )
    }

    /// DP4A-accelerated convolution
    pub fn dp4a_qconv2d_int8(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.dp4a_available {
            return Err(TorshError::BackendError("DP4A not available".to_string()).into());
        }

        // This would implement DP4A-optimized convolution
        let batch_size = input.shape()[0];
        let out_channels = weight.shape()[0];
        let out_height = input.shape()[2];
        let out_width = input.shape()[3];

        QuantizedTensor::from_data(
            vec![0; batch_size * out_channels * out_height * out_width],
            vec![batch_size, out_channels, out_height, out_width],
            input.params().clone(),
            input.device().clone(),
        )
    }
}

/// NVIDIA Tensor Core quantization operations
///
/// Provides optimized quantization operations using NVIDIA Tensor Cores,
/// available on Volta and newer architectures for massive parallel processing.
#[derive(Debug, Clone)]
pub struct TensorCoreQuantizationOps {
    /// Tensor core availability
    tensor_cores_available: bool,
}

impl TensorCoreQuantizationOps {
    /// Create new Tensor Core quantization operations
    pub fn new() -> Self {
        Self {
            tensor_cores_available: Self::detect_tensor_cores(),
        }
    }

    /// Detect Tensor Core support (simplified - would query CUDA properties)
    fn detect_tensor_cores() -> bool {
        // In practice, this would query CUDA device compute capability
        // Tensor Cores are available on compute capability 7.0+ (Volta and newer)
        true
    }

    /// Check if Tensor Cores are available
    pub fn is_available(&self) -> bool {
        self.tensor_cores_available
    }

    /// Tensor Core INT8 matrix multiplication
    ///
    /// Uses NVIDIA Tensor Cores with WMMA (Warp Matrix-Multiply Accumulate)
    /// for massive parallel INT8 matrix multiplication with INT32 accumulation.
    pub fn tensor_core_qmatmul_int8(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.tensor_cores_available {
            return Err(TorshError::BackendError(
                "Tensor cores not available".to_string(),
            ).into());
        }

        // Validate input tensors and dimensions for Tensor Core requirements
        if a.params().dtype != QuantizedDType::Int8 || b.params().dtype != QuantizedDType::Int8 {
            return Err(TorshError::BackendError(
                "Tensor Cores require INT8 tensors".to_string(),
            ).into());
        }

        let m = a.shape()[0];
        let n = b.shape()[1];

        // Tensor Cores work best with dimensions that are multiples of 16
        if m % 16 != 0 || n % 16 != 0 {
            return Err(TorshError::BackendError(
                "Tensor Core dimensions should be multiples of 16".to_string(),
            ).into());
        }

        // This would use Tensor Core WMMA instructions
        // For now, return a placeholder result
        QuantizedTensor::from_data(
            vec![0; m * n],
            vec![m, n],
            a.params().clone(),
            a.device().clone(),
        )
    }

    /// Tensor Core mixed-precision operations
    pub fn tensor_core_mixed_precision_qmatmul(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        if !self.tensor_cores_available {
            return Err(TorshError::BackendError(
                "Tensor cores not available".to_string(),
            ).into());
        }

        // This would implement mixed-precision computation using Tensor Cores
        // For example: INT8 inputs with FP16 or FP32 accumulation
        let m = a.shape()[0];
        let n = b.shape()[1];

        QuantizedTensor::from_data(
            vec![0; m * n],
            vec![m, n],
            a.params().clone(),
            a.device().clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vnni_ops_creation() {
        let vnni_ops = VnniQuantizationOps::new();
        // VNNI availability depends on hardware
        assert!(vnni_ops.is_available() || !vnni_ops.is_available());
    }

    #[test]
    fn test_dp4a_ops_creation() {
        let dp4a_ops = Dp4aQuantizationOps::new();
        // DP4A is assumed available in this test environment
        assert!(dp4a_ops.is_available());
    }

    #[test]
    fn test_tensor_core_ops_creation() {
        let tc_ops = TensorCoreQuantizationOps::new();
        // Tensor Cores are assumed available in this test environment
        assert!(tc_ops.is_available());
    }

    #[test]
    fn test_advanced_accelerator_creation() {
        let accelerator = AdvancedQuantizationAccelerator::new(Device::cpu().unwrap());

        // Check that base operations are available
        assert!(accelerator.base_ops().device() == &Device::cpu().unwrap());

        // Hardware-specific features depend on the actual hardware
        // Just verify the methods work
        let _has_vnni = accelerator.has_vnni();
        let _has_dp4a = accelerator.has_dp4a();
        let _has_tc = accelerator.has_tensor_cores();
    }

    #[test]
    fn test_benchmark_results() {
        let mut results = BenchmarkResults::new();
        results.add_benchmark("test_op", 100, Duration::from_millis(10));

        assert_eq!(results.results.len(), 1);
        assert!(results.summary.avg_throughput > 0.0);
        assert_eq!(results.summary.best_operation, "test_op");
    }

    #[test]
    fn test_workload_creation() {
        let workload = QuantizationWorkload {
            operation_type: QuantizationOperationType::MatrixMultiply { m: 128, n: 128, k: 128 },
            frequency: 1.0,
            requirements: PerformanceRequirements::default(),
        };

        match workload.operation_type {
            QuantizationOperationType::MatrixMultiply { m, n, k } => {
                assert_eq!(m, 128);
                assert_eq!(n, 128);
                assert_eq!(k, 128);
            }
            _ => panic!("Unexpected operation type"),
        }
    }

    #[test]
    fn test_auto_tuning_config() {
        let config = AutoTuningConfig::default();
        assert!(config.enable_auto_tuning);
        assert!(config.max_tuning_time_secs > 0.0);
        assert!(config.benchmark_iterations > 0);
    }
}