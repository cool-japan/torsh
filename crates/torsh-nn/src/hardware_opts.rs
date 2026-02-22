//! Hardware-Specific Layer Optimizations
//!
//! This module provides hardware-aware layer implementations that automatically
//! select the best implementation based on available hardware features:
//! - SIMD optimizations (AVX2, AVX-512, NEON)
//! - GPU tensor core utilization
//! - Cache-optimized memory layouts
//! - Quantized inference kernels
//!
//! # SciRS2 Policy Compliance
//!
//! All hardware-specific operations use scirs2-core abstractions:
//! - SIMD: `scirs2_core::simd_ops::SimdUnifiedOps`
//! - GPU: `scirs2_core::gpu` (when available)
//! - Parallel: `scirs2_core::parallel_ops`
//!
//! # Examples
//!
//! ```ignore
//! use torsh_nn::hardware_opts::{HardwareLinear, HardwareContext};
//!
//! // Auto-detect hardware capabilities
//! let ctx = HardwareContext::auto_detect();
//!
//! // Create hardware-optimized linear layer
//! let layer = HardwareLinear::new(784, 128, true, &ctx)?;
//!
//! // Forward pass automatically uses best available implementation
//! let output = layer.forward(&input)?;
//! ```

use crate::{Module, ModuleBase, Parameter};
use std::collections::HashMap;
use torsh_core::error::Result;
#[cfg(not(feature = "cuda"))]
use torsh_core::error::TorshError;
use torsh_tensor::{creation::*, Tensor};

// ================================================================================================
// Hardware Detection and Context
// ================================================================================================

/// Hardware capabilities detected at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareCapabilities {
    /// AVX2 support (x86/x86_64)
    pub has_avx2: bool,
    /// AVX-512 support (x86/x86_64)
    pub has_avx512: bool,
    /// NEON support (ARM)
    pub has_neon: bool,
    /// CUDA GPU available
    pub has_cuda: bool,
    /// ROCm GPU available
    pub has_rocm: bool,
    /// Metal GPU available (Apple Silicon)
    pub has_metal: bool,
    /// Tensor cores available (NVIDIA)
    pub has_tensor_cores: bool,
    /// Number of CPU cores
    pub num_cores: usize,
    /// L1 cache size (bytes)
    pub l1_cache_size: usize,
    /// L2 cache size (bytes)
    pub l2_cache_size: usize,
    /// L3 cache size (bytes)
    pub l3_cache_size: usize,
}

impl HardwareCapabilities {
    /// Detect hardware capabilities automatically
    pub fn detect() -> Self {
        // Platform-specific detection
        #[cfg(target_arch = "x86_64")]
        let (has_avx2, has_avx512) = {
            #[cfg(target_feature = "avx2")]
            let avx2 = true;
            #[cfg(not(target_feature = "avx2"))]
            let avx2 = is_x86_feature_detected!("avx2");

            #[cfg(target_feature = "avx512f")]
            let avx512 = true;
            #[cfg(not(target_feature = "avx512f"))]
            let avx512 = is_x86_feature_detected!("avx512f");

            (avx2, avx512)
        };

        #[cfg(not(target_arch = "x86_64"))]
        let (has_avx2, has_avx512) = (false, false);

        #[cfg(target_arch = "aarch64")]
        let has_neon = {
            #[cfg(target_feature = "neon")]
            {
                true
            }
            #[cfg(not(target_feature = "neon"))]
            {
                // NEON is mandatory on AArch64
                true
            }
        };

        #[cfg(not(target_arch = "aarch64"))]
        let has_neon = false;

        // GPU detection (simplified - would need actual GPU query in production)
        #[cfg(feature = "cuda")]
        let has_cuda = true;
        #[cfg(not(feature = "cuda"))]
        let has_cuda = false;

        #[cfg(feature = "rocm")]
        let has_rocm = true;
        #[cfg(not(feature = "rocm"))]
        let has_rocm = false;

        #[cfg(all(target_vendor = "apple", feature = "metal"))]
        let has_metal = true;
        #[cfg(not(all(target_vendor = "apple", feature = "metal")))]
        let has_metal = false;

        // Tensor cores (NVIDIA Volta+)
        let has_tensor_cores = has_cuda; // Simplified check

        // CPU cores
        let num_cores = num_cpus::get();

        // Cache sizes (platform-specific estimates)
        let (l1_cache_size, l2_cache_size, l3_cache_size) = {
            #[cfg(target_arch = "x86_64")]
            {
                (32 * 1024, 256 * 1024, 8 * 1024 * 1024) // Typical x86_64
            }
            #[cfg(target_arch = "aarch64")]
            {
                (64 * 1024, 512 * 1024, 4 * 1024 * 1024) // Typical ARM
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                (32 * 1024, 256 * 1024, 2 * 1024 * 1024) // Generic
            }
        };

        Self {
            has_avx2,
            has_avx512,
            has_neon,
            has_cuda,
            has_rocm,
            has_metal,
            has_tensor_cores,
            num_cores,
            l1_cache_size,
            l2_cache_size,
            l3_cache_size,
        }
    }

    /// Get best SIMD width for this hardware (in f32 elements)
    pub fn simd_width(&self) -> usize {
        if self.has_avx512 {
            16 // AVX-512: 512 bits / 32 bits = 16 floats
        } else if self.has_avx2 {
            8 // AVX2: 256 bits / 32 bits = 8 floats
        } else if self.has_neon {
            4 // NEON: 128 bits / 32 bits = 4 floats
        } else {
            1 // Scalar fallback
        }
    }

    /// Recommended tile size for matrix operations based on cache
    pub fn matrix_tile_size(&self) -> usize {
        // Aim to fit tiles in L1 cache
        // For A[tile x k] @ B[k x tile], we need 2*tile*k*4 bytes
        // Target: tile*tile*4 <= L1_cache / 3 (leave room for other data)
        let target_bytes = self.l1_cache_size / 3;
        let tile = (target_bytes / 4).isqrt(); // Integer square root

        // Round down to SIMD width multiple
        let simd = self.simd_width();
        (tile / simd) * simd
    }
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

/// Hardware execution context
#[derive(Debug, Clone)]
pub struct HardwareContext {
    capabilities: HardwareCapabilities,
    prefer_gpu: bool,
    force_cpu: bool,
    tile_size_override: Option<usize>,
}

impl HardwareContext {
    /// Auto-detect hardware and create context
    pub fn auto_detect() -> Self {
        Self {
            capabilities: HardwareCapabilities::detect(),
            prefer_gpu: false,
            force_cpu: false,
            tile_size_override: None,
        }
    }

    /// Create CPU-only context
    pub fn cpu_only() -> Self {
        Self {
            capabilities: HardwareCapabilities::detect(),
            prefer_gpu: false,
            force_cpu: true,
            tile_size_override: None,
        }
    }

    /// Create GPU-preferred context
    pub fn gpu_preferred() -> Self {
        Self {
            capabilities: HardwareCapabilities::detect(),
            prefer_gpu: true,
            force_cpu: false,
            tile_size_override: None,
        }
    }

    /// Set custom tile size
    pub fn with_tile_size(mut self, size: usize) -> Self {
        self.tile_size_override = Some(size);
        self
    }

    /// Get effective tile size
    pub fn tile_size(&self) -> usize {
        self.tile_size_override
            .unwrap_or_else(|| self.capabilities.matrix_tile_size())
    }

    /// Check if GPU should be used
    pub fn use_gpu(&self) -> bool {
        !self.force_cpu
            && self.prefer_gpu
            && (self.capabilities.has_cuda
                || self.capabilities.has_rocm
                || self.capabilities.has_metal)
    }

    /// Get SIMD width
    pub fn simd_width(&self) -> usize {
        self.capabilities.simd_width()
    }

    /// Get capabilities
    pub fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }
}

impl Default for HardwareContext {
    fn default() -> Self {
        Self::auto_detect()
    }
}

// ================================================================================================
// Hardware-Optimized Linear Layer
// ================================================================================================

/// Hardware-optimized linear layer
///
/// Automatically selects the best implementation based on:
/// - Available SIMD instructions (AVX2, AVX-512, NEON)
/// - GPU availability and tensor core support
/// - Cache sizes for tiling strategies
///
/// # Examples
///
/// ```ignore
/// let ctx = HardwareContext::auto_detect();
/// let layer = HardwareLinear::new(1024, 512, true, &ctx)?;
/// ```
#[derive(Debug)]
pub struct HardwareLinear {
    base: ModuleBase,
    in_features: usize,
    out_features: usize,
    use_bias: bool,
    context: HardwareContext,
}

impl HardwareLinear {
    /// Create new hardware-optimized linear layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        context: &HardwareContext,
    ) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Initialize weight with shape [in_features, out_features] for direct matmul
        let weight = crate::init::kaiming_uniform(&[in_features, out_features], "fan_in")?;
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        // Initialize bias if enabled
        if use_bias {
            let bias = zeros(&[out_features])?;
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            in_features,
            out_features,
            use_bias,
            context: context.clone(),
        })
    }

    /// Forward pass with hardware-specific optimizations
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Select implementation based on hardware
        if self.context.use_gpu() {
            self.forward_gpu(input)
        } else if self.context.capabilities().has_avx512 {
            self.forward_avx512(input)
        } else if self.context.capabilities().has_avx2 {
            self.forward_avx2(input)
        } else if self.context.capabilities().has_neon {
            self.forward_neon(input)
        } else {
            self.forward_generic(input)
        }
    }

    /// GPU-accelerated forward pass
    #[cfg(feature = "cuda")]
    fn forward_gpu(&self, input: &Tensor) -> Result<Tensor> {
        // Use GPU backend if available
        let weight = self.base.parameters["weight"].tensor().read().clone();
        let bias_opt = if self.use_bias {
            Some(self.base.parameters["bias"].tensor().read().clone())
        } else {
            None
        };

        // TODO: Use actual GPU kernel through scirs2_core::gpu
        // For now, fall back to generic
        crate::functional::linear(input, &weight, bias_opt.as_ref())
    }

    #[cfg(not(feature = "cuda"))]
    fn forward_gpu(&self, _input: &Tensor) -> Result<Tensor> {
        Err(TorshError::Other(
            "GPU support not enabled (cuda feature required)".to_string(),
        ))
    }

    /// AVX-512 optimized forward pass
    #[cfg(target_arch = "x86_64")]
    fn forward_avx512(&self, input: &Tensor) -> Result<Tensor> {
        // Use SIMD operations through scirs2_core
        #[cfg(feature = "simd")]
        {
            // TODO: Use scirs2_core::simd_ops::SimdUnifiedOps for AVX-512 intrinsics
            let weight = self.base.parameters["weight"].tensor().read().clone();

            // For AVX-512, process 16 floats at a time
            // TODO: Implement tiled matmul with AVX-512 intrinsics through scirs2
            // For now, use generic with hint that SIMD is available

            let result = input.matmul(&weight)?;

            if self.use_bias {
                let bias = self.base.parameters["bias"].tensor().read().clone();
                result.add(&bias)
            } else {
                Ok(result)
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            self.forward_generic(input)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn forward_avx512(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_generic(input)
    }

    /// AVX2 optimized forward pass
    #[cfg(target_arch = "x86_64")]
    fn forward_avx2(&self, input: &Tensor) -> Result<Tensor> {
        // Use SIMD operations through scirs2_core
        #[cfg(feature = "simd")]
        {
            // TODO: Use scirs2_core::simd_ops::SimdUnifiedOps for AVX2 intrinsics
            let weight = self.base.parameters["weight"].tensor().read().clone();

            // For AVX2, process 8 floats at a time
            // TODO: Implement tiled matmul with AVX2 intrinsics through scirs2
            // For now, use generic with hint that SIMD is available

            let result = input.matmul(&weight)?;

            if self.use_bias {
                let bias = self.base.parameters["bias"].tensor().read().clone();
                result.add(&bias)
            } else {
                Ok(result)
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            self.forward_generic(input)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn forward_avx2(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_generic(input)
    }

    /// NEON optimized forward pass (ARM)
    #[cfg(target_arch = "aarch64")]
    fn forward_neon(&self, input: &Tensor) -> Result<Tensor> {
        // Use SIMD operations through scirs2_core
        #[cfg(feature = "simd")]
        {
            // TODO: Use scirs2_core::simd_ops::SimdUnifiedOps for NEON intrinsics
            let weight = self.base.parameters["weight"].tensor().read().clone();

            // For NEON, process 4 floats at a time
            // TODO: Implement tiled matmul with NEON intrinsics through scirs2

            let result = input.matmul(&weight)?;

            if self.use_bias {
                let bias = self.base.parameters["bias"].tensor().read().clone();
                result.add(&bias)
            } else {
                Ok(result)
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            self.forward_generic(input)
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn forward_neon(&self, _input: &Tensor) -> Result<Tensor> {
        self.forward_generic(_input)
    }

    /// Generic (portable) forward pass
    fn forward_generic(&self, input: &Tensor) -> Result<Tensor> {
        let weight = self.base.parameters["weight"].tensor().read().clone();
        let bias_opt = if self.use_bias {
            Some(self.base.parameters["bias"].tensor().read().clone())
        } else {
            None
        };

        crate::functional::linear(input, &weight, bias_opt.as_ref())
    }

    /// Get input features
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Check if bias is enabled
    pub fn has_bias(&self) -> bool {
        self.use_bias
    }

    /// Get hardware context
    pub fn context(&self) -> &HardwareContext {
        &self.context
    }
}

impl Module for HardwareLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn training(&self) -> bool {
        self.base.training()
    }
}

// ================================================================================================
// Hardware Information Utilities
// ================================================================================================

/// Print hardware capabilities summary
pub fn print_hardware_info() {
    let caps = HardwareCapabilities::detect();
    println!("=== Hardware Capabilities ===");
    println!("CPU:");
    println!("  Cores: {}", caps.num_cores);
    println!("  AVX2: {}", caps.has_avx2);
    println!("  AVX-512: {}", caps.has_avx512);
    println!("  NEON: {}", caps.has_neon);
    println!("  SIMD Width: {} floats", caps.simd_width());
    println!("Cache:");
    println!("  L1: {} KB", caps.l1_cache_size / 1024);
    println!("  L2: {} KB", caps.l2_cache_size / 1024);
    println!("  L3: {} KB", caps.l3_cache_size / 1024);
    println!("  Recommended tile size: {}", caps.matrix_tile_size());
    println!("GPU:");
    println!("  CUDA: {}", caps.has_cuda);
    println!("  ROCm: {}", caps.has_rocm);
    println!("  Metal: {}", caps.has_metal);
    println!("  Tensor Cores: {}", caps.has_tensor_cores);
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let caps = HardwareCapabilities::detect();
        // Should detect at least one core
        assert!(caps.num_cores > 0);
        // SIMD width should be valid
        assert!(caps.simd_width() >= 1);
        assert!(caps.simd_width() <= 16);
        // Cache sizes should be reasonable
        assert!(caps.l1_cache_size > 0);
        assert!(caps.l2_cache_size >= caps.l1_cache_size);
    }

    #[test]
    fn test_hardware_context() {
        let ctx = HardwareContext::auto_detect();
        assert!(ctx.tile_size() > 0);
        assert!(ctx.simd_width() >= 1);

        let cpu_ctx = HardwareContext::cpu_only();
        assert!(!cpu_ctx.use_gpu());

        let gpu_ctx = HardwareContext::gpu_preferred();
        // May or may not have GPU
        let _ = gpu_ctx.use_gpu();
    }

    #[test]
    fn test_hardware_linear_creation() {
        let ctx = HardwareContext::auto_detect();
        let layer = HardwareLinear::new(10, 5, true, &ctx);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert!(layer.has_bias());
    }

    #[test]
    fn test_hardware_linear_forward() {
        let ctx = HardwareContext::cpu_only(); // Force CPU for deterministic test
        let layer = HardwareLinear::new(10, 5, true, &ctx).unwrap();
        let input = randn(&[2, 10]).unwrap();
        let output = layer.forward(&input);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);
    }

    #[test]
    fn test_custom_tile_size() {
        let ctx = HardwareContext::auto_detect().with_tile_size(64);
        assert_eq!(ctx.tile_size(), 64);
    }

    #[test]
    fn test_simd_width_bounds() {
        let caps = HardwareCapabilities::detect();
        let width = caps.simd_width();

        // SIMD width should be power of 2 and reasonable
        assert!(width == 1 || width == 2 || width == 4 || width == 8 || width == 16);
    }
}
