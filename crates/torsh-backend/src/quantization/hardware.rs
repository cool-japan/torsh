//! Hardware acceleration features for quantization operations
//!
//! This module provides hardware-specific optimizations and feature detection
//! for quantization operations. It includes support for various CPU and GPU
//! acceleration technologies including SIMD, VNNI, DP4A, and Tensor Cores.

use crate::{BackendResult, Device};
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Hardware-specific quantization features available on the current device
///
/// This struct encapsulates the hardware capabilities available for quantization
/// operations, enabling the system to choose optimal implementations based on
/// what the hardware supports.
#[derive(Debug, Clone)]
pub struct QuantizationHardwareFeatures {
    /// Supports INT8 SIMD operations
    ///
    /// Indicates whether the hardware can perform vectorized INT8 operations,
    /// which significantly accelerates quantized computations.
    pub supports_int8_simd: bool,

    /// Supports packed INT4 operations
    ///
    /// Some hardware can efficiently handle sub-byte quantization formats
    /// like INT4, where multiple values are packed into single bytes.
    pub supports_int4_packed: bool,

    /// Supports VNNI (Vector Neural Network Instructions)
    ///
    /// Intel's VNNI instructions provide hardware acceleration for
    /// neural network workloads, particularly beneficial for quantized models.
    pub supports_vnni: bool,

    /// Supports DP4A (4-element dot product and accumulate)
    ///
    /// NVIDIA's DP4A instruction performs 4-element dot products in a single
    /// operation, ideal for quantized matrix operations on CUDA devices.
    pub supports_dp4a: bool,

    /// Supports tensor core operations
    ///
    /// Modern GPUs include specialized tensor cores for mixed-precision
    /// and quantized neural network computations.
    pub supports_tensor_cores: bool,

    /// Supports mixed precision operations
    ///
    /// Hardware capability to efficiently mix different quantization
    /// precisions within the same computation.
    pub supports_mixed_precision: bool,

    /// Maximum number of parallel operations
    ///
    /// The optimal number of parallel operations for this hardware,
    /// used for scheduling and batching decisions.
    pub max_parallel_ops: usize,
}

impl Default for QuantizationHardwareFeatures {
    /// Conservative default hardware features
    ///
    /// Returns a conservative set of capabilities that should work
    /// on any hardware without advanced acceleration features.
    fn default() -> Self {
        Self {
            supports_int8_simd: false,
            supports_int4_packed: false,
            supports_vnni: false,
            supports_dp4a: false,
            supports_tensor_cores: false,
            supports_mixed_precision: false,
            max_parallel_ops: 1,
        }
    }
}

impl QuantizationHardwareFeatures {
    /// Detect hardware features for the given device
    ///
    /// Performs runtime detection of available hardware acceleration
    /// features and returns a capabilities structure.
    ///
    /// # Arguments
    ///
    /// * `device` - The target device to analyze
    ///
    /// # Returns
    ///
    /// A `QuantizationHardwareFeatures` struct with detected capabilities
    pub fn detect_for_device(device: &Device) -> Self {
        match device.device_type() {
            torsh_core::device::DeviceType::Cpu => Self::detect_cpu_features(),
            torsh_core::device::DeviceType::Cuda(_) => Self::detect_cuda_features(),
            _ => Self::default(),
        }
    }

    /// Detect CPU-specific quantization features
    fn detect_cpu_features() -> Self {
        Self {
            supports_int8_simd: Self::detect_int8_simd(),
            supports_int4_packed: true, // Generally available through software
            supports_vnni: Self::detect_vnni(),
            supports_dp4a: false,         // DP4A is CUDA-specific
            supports_tensor_cores: false, // Tensor cores are GPU-specific
            supports_mixed_precision: true,
            max_parallel_ops: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
    }

    /// Detect CUDA GPU quantization features
    fn detect_cuda_features() -> Self {
        Self {
            supports_int8_simd: true, // CUDA has vectorized INT8 support
            supports_int4_packed: true,
            supports_vnni: false, // VNNI is Intel-specific
            supports_dp4a: Self::detect_dp4a(),
            supports_tensor_cores: Self::detect_tensor_cores(),
            supports_mixed_precision: true,
            max_parallel_ops: 1024, // Many CUDA cores available
        }
    }

    /// Detect Intel VNNI (Vector Neural Network Instructions) support
    ///
    /// VNNI instructions accelerate neural network computations by providing
    /// hardware support for common operations like dot products with INT8 data.
    fn detect_vnni() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Check for VNNI support via CPUID
            // Note: This checks for AVX512-VNNI specifically
            // AVX-VNNI support would require different detection
            is_x86_feature_detected!("avx512vnni")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Detect general INT8 SIMD support
    ///
    /// Checks for hardware support of vectorized INT8 operations,
    /// which are crucial for efficient quantized computation.
    fn detect_int8_simd() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Most modern x86 CPUs support some form of INT8 SIMD
            is_x86_feature_detected!("sse2") || is_x86_feature_detected!("avx2")
        }
        #[cfg(target_arch = "aarch64")]
        {
            // ARM NEON supports INT8 operations
            std::arch::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Detect NVIDIA DP4A support
    ///
    /// DP4A (4-element dot product and accumulate) is available on
    /// modern NVIDIA GPUs and provides efficient INT8 matrix operations.
    fn detect_dp4a() -> bool {
        // In a real implementation, this would query CUDA device properties
        // For now, assume modern CUDA GPUs have DP4A support
        true
    }

    /// Detect tensor core support
    ///
    /// Tensor cores provide specialized acceleration for mixed-precision
    /// and quantized neural network operations on modern GPUs.
    fn detect_tensor_cores() -> bool {
        // In a real implementation, this would check GPU architecture
        // (Volta, Turing, Ampere, etc.) for tensor core availability
        true
    }

    /// Check if the hardware supports a specific quantization data type efficiently
    ///
    /// # Arguments
    ///
    /// * `dtype` - The quantization data type to check
    ///
    /// # Returns
    ///
    /// `true` if the hardware can efficiently process this data type
    pub fn supports_dtype_efficiently(&self, dtype: &crate::quantization::QuantizedDType) -> bool {
        use crate::quantization::QuantizedDType;

        match dtype {
            QuantizedDType::Int8 | QuantizedDType::UInt8 => self.supports_int8_simd,
            QuantizedDType::Int4 | QuantizedDType::UInt4 => self.supports_int4_packed,
            QuantizedDType::Binary => self.supports_int8_simd, // Can use SIMD for binary ops
            QuantizedDType::Int16 | QuantizedDType::UInt16 => true, // Generally well supported
            QuantizedDType::Mixed(_) => self.supports_mixed_precision,
        }
    }

    /// Get the optimal block size for parallel operations
    ///
    /// Returns the recommended block size for batching operations
    /// based on hardware characteristics and parallelism capabilities.
    pub fn optimal_block_size(&self) -> usize {
        if self.supports_tensor_cores {
            // Tensor cores work well with larger blocks
            256
        } else if self.supports_int8_simd {
            // SIMD operations benefit from medium-sized blocks
            64
        } else {
            // Conservative block size for scalar operations
            16
        }
    }

    /// Get the performance preference ranking for quantization schemes
    ///
    /// Returns quantization schemes ordered by expected performance
    /// on this hardware, with the fastest schemes first.
    pub fn performance_ranking(&self) -> Vec<crate::quantization::QuantizationScheme> {
        use crate::quantization::QuantizationScheme;

        let mut schemes = vec![
            QuantizationScheme::Symmetric,   // Often fastest due to no zero point
            QuantizationScheme::Linear,      // Standard implementation
            QuantizationScheme::Asymmetric,  // Requires zero point handling
            QuantizationScheme::ChannelWise, // More complex but better accuracy
            QuantizationScheme::BlockWise,   // Complex memory access patterns
            QuantizationScheme::Logarithmic, // Requires expensive log operations
        ];

        // Adjust ranking based on hardware capabilities
        if self.supports_vnni || self.supports_dp4a {
            // Hardware-accelerated schemes can handle complexity better
            schemes.swap(2, 3); // Prefer channel-wise over asymmetric
        }

        schemes
    }
}

/// SIMD-accelerated quantization operations
///
/// This struct provides vectorized implementations of quantization operations
/// that can take advantage of CPU SIMD instructions for improved performance.
#[derive(Debug, Clone)]
pub struct SimdQuantizationOps {
    /// Whether SIMD operations are available
    simd_available: bool,
    /// Optimal vector width for this hardware
    vector_width: usize,
}

impl SimdQuantizationOps {
    /// Create new SIMD quantization operations
    pub fn new() -> Self {
        Self {
            simd_available: QuantizationHardwareFeatures::detect_int8_simd(),
            vector_width: Self::detect_vector_width(),
        }
    }

    /// Detect optimal vector width for SIMD operations
    fn detect_vector_width() -> usize {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx512f") {
                64 // AVX-512 can handle 64 bytes (512 bits)
            } else if is_x86_feature_detected!("avx2") {
                32 // AVX2 can handle 32 bytes (256 bits)
            } else if is_x86_feature_detected!("sse2") {
                16 // SSE2 can handle 16 bytes (128 bits)
            } else {
                4 // Fallback to scalar with some vectorization
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            16 // ARM NEON typically handles 128-bit vectors
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            4 // Conservative fallback
        }
    }

    /// SIMD-accelerated f32 to u8 quantization
    ///
    /// Uses vectorized operations to quantize multiple floating-point values
    /// to 8-bit unsigned integers simultaneously.
    pub fn quantize_f32_to_u8_simd(
        &self,
        input: &[f32],
        scale: f32,
        zero_point: f32,
    ) -> BackendResult<Vec<u8>> {
        if !self.simd_available {
            return Err(TorshError::BackendError("SIMD not available".to_string()));
        }

        let mut output = Vec::with_capacity(input.len());
        let inv_scale = 1.0 / scale;

        // Process in chunks optimized for the vector width
        let chunk_size = self.vector_width / 4; // 4 bytes per f32

        for chunk in input.chunks(chunk_size) {
            // In a real implementation, this would use platform-specific SIMD intrinsics
            // For portability, we use a vectorized approach that compilers can optimize
            for &val in chunk {
                let quantized = (val * inv_scale + zero_point).round().clamp(0.0, 255.0) as u8;
                output.push(quantized);
            }
        }

        Ok(output)
    }

    /// SIMD-accelerated u8 to f32 dequantization
    pub fn dequantize_u8_to_f32_simd(
        &self,
        input: &[u8],
        scale: f32,
        zero_point: f32,
    ) -> BackendResult<Vec<f32>> {
        if !self.simd_available {
            return Err(TorshError::BackendError("SIMD not available".to_string()));
        }

        let mut output = Vec::with_capacity(input.len());
        let chunk_size = self.vector_width; // 1 byte per u8

        for chunk in input.chunks(chunk_size) {
            for &val in chunk {
                let dequantized = (val as f32 - zero_point) * scale;
                output.push(dequantized);
            }
        }

        Ok(output)
    }

    /// SIMD-accelerated INT8 vector addition
    pub fn add_int8_simd(&self, a: &[i8], b: &[i8]) -> BackendResult<Vec<i8>> {
        if !self.simd_available || a.len() != b.len() {
            return Err(TorshError::BackendError(
                "Invalid input for SIMD addition".to_string(),
            ));
        }

        let mut result = Vec::with_capacity(a.len());
        let chunk_size = self.vector_width;

        for (a_chunk, b_chunk) in a.chunks(chunk_size).zip(b.chunks(chunk_size)) {
            for (&a_val, &b_val) in a_chunk.iter().zip(b_chunk.iter()) {
                let sum = (a_val as i16 + b_val as i16).clamp(-128, 127) as i8;
                result.push(sum);
            }
        }

        Ok(result)
    }

    /// Check if SIMD operations are available
    pub fn is_available(&self) -> bool {
        self.simd_available
    }

    /// Get the optimal vector width for this hardware
    pub fn vector_width(&self) -> usize {
        self.vector_width
    }
}

/// Memory layout optimization for quantized data
///
/// Provides utilities for organizing quantized data in memory layouts
/// that are optimal for hardware acceleration.
#[derive(Debug, Clone)]
pub struct QuantizedMemoryLayout {
    /// Whether to use packed layouts for sub-byte types
    pub use_packed_layout: bool,
    /// Preferred memory alignment in bytes
    pub alignment: usize,
    /// Whether to use interleaved data layouts
    pub use_interleaving: bool,
}

impl QuantizedMemoryLayout {
    /// Create optimal memory layout for the given hardware features
    pub fn optimal_for_hardware(features: &QuantizationHardwareFeatures) -> Self {
        Self {
            use_packed_layout: features.supports_int4_packed,
            alignment: if features.supports_int8_simd { 32 } else { 16 },
            use_interleaving: features.supports_tensor_cores,
        }
    }

    /// Calculate optimal stride for accessing quantized data
    pub fn optimal_stride(&self, data_width: usize) -> usize {
        // Align stride to hardware requirements
        let aligned_width = (data_width + self.alignment - 1) & !(self.alignment - 1);
        aligned_width
    }

    /// Check if the given memory layout is hardware-optimal
    pub fn is_layout_optimal(&self, data_size: usize, stride: usize) -> bool {
        let optimal_stride = self.optimal_stride(data_size);
        stride >= optimal_stride && stride % self.alignment == 0
    }
}

/// Hardware-specific performance hints
///
/// Provides recommendations for optimal quantization strategies
/// based on detected hardware capabilities.
#[derive(Debug, Clone)]
pub struct QuantizationPerformanceHints {
    /// Recommended quantization data types in order of preference
    pub preferred_dtypes: Vec<crate::quantization::QuantizedDType>,
    /// Recommended quantization schemes in order of preference
    pub preferred_schemes: Vec<crate::quantization::QuantizationScheme>,
    /// Optimal batch size for operations
    pub optimal_batch_size: usize,
    /// Whether to prefer in-place operations
    pub prefer_inplace: bool,
}

impl QuantizationPerformanceHints {
    /// Generate performance hints for the given hardware features
    pub fn for_hardware(features: &QuantizationHardwareFeatures) -> Self {
        use crate::quantization::QuantizedDType;

        let mut preferred_dtypes = vec![];
        let mut preferred_schemes = vec![];

        // Order data types by hardware support and performance
        if features.supports_int8_simd {
            preferred_dtypes.extend([QuantizedDType::Int8, QuantizedDType::UInt8]);
        }
        if features.supports_int4_packed {
            preferred_dtypes.extend([QuantizedDType::Int4, QuantizedDType::UInt4]);
        }
        if features.supports_mixed_precision {
            preferred_dtypes.push(QuantizedDType::Mixed(vec![8, 4, 8]));
        }

        // Add remaining types
        preferred_dtypes.extend([
            QuantizedDType::Int16,
            QuantizedDType::UInt16,
            QuantizedDType::Binary,
        ]);

        // Use hardware-specific scheme ranking
        preferred_schemes = features.performance_ranking();

        Self {
            preferred_dtypes,
            preferred_schemes,
            optimal_batch_size: features.optimal_block_size(),
            prefer_inplace: !features.supports_tensor_cores, // Tensor cores often prefer separate output
        }
    }

    /// Get the best quantization data type for the given requirements
    pub fn best_dtype_for_accuracy(
        &self,
        min_accuracy: f64,
    ) -> Option<&crate::quantization::QuantizedDType> {
        use crate::quantization::QuantizedDType;

        // Higher bit widths generally provide better accuracy
        for dtype in &self.preferred_dtypes {
            let expected_accuracy = match dtype {
                QuantizedDType::Int16 | QuantizedDType::UInt16 => 0.99,
                QuantizedDType::Int8 | QuantizedDType::UInt8 => 0.95,
                QuantizedDType::Int4 | QuantizedDType::UInt4 => 0.85,
                QuantizedDType::Binary => 0.70,
                QuantizedDType::Mixed(_) => 0.90,
            };

            if expected_accuracy >= min_accuracy {
                return Some(dtype);
            }
        }

        None
    }

    /// Get the best quantization scheme for the given performance requirements
    pub fn best_scheme_for_latency(
        &self,
        max_latency_factor: f64,
    ) -> Option<&crate::quantization::QuantizationScheme> {
        use crate::quantization::QuantizationScheme;

        // Different schemes have different computational complexity
        for scheme in &self.preferred_schemes {
            let latency_factor = match scheme {
                QuantizationScheme::Symmetric => 1.0,
                QuantizationScheme::Linear => 1.1,
                QuantizationScheme::Asymmetric => 1.2,
                QuantizationScheme::ChannelWise => 1.3,
                QuantizationScheme::BlockWise => 1.4,
                QuantizationScheme::Logarithmic => 2.0,
            };

            if latency_factor <= max_latency_factor {
                return Some(scheme);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_features_detection() {
        let features = QuantizationHardwareFeatures::default();

        // Default features should be conservative
        assert!(!features.supports_int8_simd);
        assert!(!features.supports_vnni);
        assert!(!features.supports_dp4a);
        assert!(!features.supports_tensor_cores);
        assert_eq!(features.max_parallel_ops, 1);
    }

    #[test]
    fn test_cpu_features_detection() {
        let features = QuantizationHardwareFeatures::detect_cpu_features();

        // CPU features should never include GPU-specific capabilities
        assert!(!features.supports_dp4a);
        assert!(!features.supports_tensor_cores);
        assert!(features.max_parallel_ops >= 1);
    }

    #[test]
    fn test_cuda_features_detection() {
        let features = QuantizationHardwareFeatures::detect_cuda_features();

        // CUDA features should include GPU-specific capabilities
        assert!(features.supports_int8_simd);
        assert!(!features.supports_vnni); // VNNI is Intel-specific
        assert!(features.max_parallel_ops > 1);
    }

    #[test]
    fn test_device_feature_detection() {
        let cpu_device = Device::cpu().unwrap();
        let cpu_features = QuantizationHardwareFeatures::detect_for_device(&cpu_device);

        // Should detect CPU-appropriate features
        assert!(!cpu_features.supports_dp4a);
        assert!(!cpu_features.supports_tensor_cores);
    }

    #[test]
    fn test_dtype_support_check() {
        use crate::quantization::QuantizedDType;

        let mut features = QuantizationHardwareFeatures::default();
        features.supports_int8_simd = true;
        features.supports_int4_packed = true;

        assert!(features.supports_dtype_efficiently(&QuantizedDType::Int8));
        assert!(features.supports_dtype_efficiently(&QuantizedDType::Int4));
        assert!(!features.supports_dtype_efficiently(&QuantizedDType::Mixed(vec![8, 4])));
    }

    #[test]
    fn test_optimal_block_size() {
        let mut features = QuantizationHardwareFeatures::default();

        // Test different hardware configurations
        features.supports_tensor_cores = true;
        assert_eq!(features.optimal_block_size(), 256);

        features.supports_tensor_cores = false;
        features.supports_int8_simd = true;
        assert_eq!(features.optimal_block_size(), 64);

        features.supports_int8_simd = false;
        assert_eq!(features.optimal_block_size(), 16);
    }

    #[test]
    fn test_performance_ranking() {
        let features = QuantizationHardwareFeatures::default();
        let ranking = features.performance_ranking();

        // Should have all schemes
        assert_eq!(ranking.len(), 6);

        // Symmetric should typically be first (fastest)
        use crate::quantization::QuantizationScheme;
        assert_eq!(ranking[0], QuantizationScheme::Symmetric);
    }

    #[test]
    fn test_simd_ops_creation() {
        let simd_ops = SimdQuantizationOps::new();

        // Should detect SIMD availability appropriately for the platform
        assert!(simd_ops.vector_width() >= 4);
    }

    #[test]
    fn test_vector_width_detection() {
        let width = SimdQuantizationOps::detect_vector_width();

        // Should return a reasonable vector width
        assert!(width >= 4);
        assert!(width <= 64);

        // Should be a power of 2 or multiple of 4
        assert!(width % 4 == 0);
    }

    #[test]
    fn test_memory_layout_optimization() {
        let features = QuantizationHardwareFeatures::default();
        let layout = QuantizedMemoryLayout::optimal_for_hardware(&features);

        assert!(layout.alignment >= 16);
        assert!(!layout.use_packed_layout); // Default doesn't support packed
    }

    #[test]
    fn test_optimal_stride_calculation() {
        let layout = QuantizedMemoryLayout {
            use_packed_layout: false,
            alignment: 32,
            use_interleaving: false,
        };

        // Test stride calculation with different data widths
        assert_eq!(layout.optimal_stride(10), 32); // Rounds up to alignment
        assert_eq!(layout.optimal_stride(32), 32); // Already aligned
        assert_eq!(layout.optimal_stride(50), 64); // Rounds up to next alignment
    }

    #[test]
    fn test_layout_optimality_check() {
        let layout = QuantizedMemoryLayout {
            use_packed_layout: false,
            alignment: 16,
            use_interleaving: false,
        };

        assert!(layout.is_layout_optimal(10, 16)); // Properly aligned
        assert!(layout.is_layout_optimal(10, 32)); // Over-aligned (OK)
        assert!(!layout.is_layout_optimal(10, 15)); // Under-aligned
        assert!(!layout.is_layout_optimal(10, 17)); // Misaligned
    }

    #[test]
    fn test_performance_hints_generation() {
        let features = QuantizationHardwareFeatures {
            supports_int8_simd: true,
            supports_int4_packed: true,
            supports_mixed_precision: true,
            ..Default::default()
        };

        let hints = QuantizationPerformanceHints::for_hardware(&features);

        // Should have preferences based on hardware support
        assert!(!hints.preferred_dtypes.is_empty());
        assert!(!hints.preferred_schemes.is_empty());
        assert!(hints.optimal_batch_size > 0);
    }

    #[test]
    fn test_best_dtype_for_accuracy() {
        let hints = QuantizationPerformanceHints {
            preferred_dtypes: vec![
                crate::quantization::QuantizedDType::Int8,
                crate::quantization::QuantizedDType::Int4,
                crate::quantization::QuantizedDType::Binary,
            ],
            preferred_schemes: vec![],
            optimal_batch_size: 64,
            prefer_inplace: false,
        };

        // Should return INT8 for high accuracy requirements
        let dtype = hints.best_dtype_for_accuracy(0.90);
        assert!(dtype.is_some());

        // Should return None for impossible accuracy requirements
        let dtype = hints.best_dtype_for_accuracy(0.99);
        assert!(dtype.is_none());
    }

    #[test]
    fn test_best_scheme_for_latency() {
        use crate::quantization::QuantizationScheme;

        let hints = QuantizationPerformanceHints {
            preferred_dtypes: vec![],
            preferred_schemes: vec![
                QuantizationScheme::Symmetric,
                QuantizationScheme::Linear,
                QuantizationScheme::Asymmetric,
            ],
            optimal_batch_size: 64,
            prefer_inplace: false,
        };

        // Should return fastest scheme for strict latency requirements
        let scheme = hints.best_scheme_for_latency(1.1);
        assert!(scheme.is_some());

        // Should return None for impossible latency requirements
        let scheme = hints.best_scheme_for_latency(0.5);
        assert!(scheme.is_none());
    }

    #[test]
    fn test_simd_quantization_operations() {
        let simd_ops = SimdQuantizationOps::new();

        if simd_ops.is_available() {
            let input = vec![1.0, 2.0, 3.0, 4.0];
            let result = simd_ops.quantize_f32_to_u8_simd(&input, 1.0, 0.0);

            if let Ok(quantized) = result {
                assert_eq!(quantized.len(), input.len());
                // Values should be approximately correct
                assert!(quantized[0] <= 2); // 1.0 rounded
                assert!(quantized[3] <= 5); // 4.0 rounded
            }
        }
    }

    #[test]
    fn test_simd_dequantization_operations() {
        let simd_ops = SimdQuantizationOps::new();

        if simd_ops.is_available() {
            let input = vec![1u8, 2u8, 3u8, 4u8];
            let result = simd_ops.dequantize_u8_to_f32_simd(&input, 1.0, 0.0);

            if let Ok(dequantized) = result {
                assert_eq!(dequantized.len(), input.len());
                // Values should match input (scale=1, zero_point=0)
                assert!((dequantized[0] - 1.0).abs() < 0.001);
                assert!((dequantized[3] - 4.0).abs() < 0.001);
            }
        }
    }

    #[test]
    fn test_simd_int8_addition() {
        let simd_ops = SimdQuantizationOps::new();

        if simd_ops.is_available() {
            let a = vec![10i8, 20i8, 30i8, 40i8];
            let b = vec![5i8, 10i8, 15i8, 20i8];
            let result = simd_ops.add_int8_simd(&a, &b);

            if let Ok(sum) = result {
                assert_eq!(sum.len(), a.len());
                assert_eq!(sum[0], 15i8);
                assert_eq!(sum[1], 30i8);
                assert_eq!(sum[2], 45i8);
                assert_eq!(sum[3], 60i8);
            }
        }
    }
}
