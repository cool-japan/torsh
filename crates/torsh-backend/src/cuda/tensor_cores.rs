//! CUDA Tensor Core acceleration for mixed precision training
//!
//! This module provides high-performance tensor operations using NVIDIA Tensor Cores
//! available on Volta, Turing, Ampere, and later GPU architectures.

use crate::cuda::stream::CudaStream;
use crate::error::BackendResult;
use half::f16;
// Note: scirs2-core provides GPU support through gpu::backends::cuda, not a top-level cuda module
// use scirs2_core::gpu::backends::cuda;
// use scirs2_core::tensor_cores;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Tensor Core compute capability requirements
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorCoreCapability {
    /// Volta architecture (7.0, 7.2) - First generation Tensor Cores
    Volta,
    /// Turing architecture (7.5) - Second generation Tensor Cores
    Turing,
    /// Ampere architecture (8.0, 8.6) - Third generation Tensor Cores
    Ampere,
    /// Hopper architecture (9.0) - Fourth generation Tensor Cores
    Hopper,
    /// Ada Lovelace architecture (8.9) - Ada generation Tensor Cores
    AdaLovelace,
    /// Unsupported architecture
    Unsupported,
}

impl TensorCoreCapability {
    /// Determine capability from compute capability version
    pub fn from_compute_capability(major: i32, minor: i32) -> Self {
        match (major, minor) {
            (7, 0) | (7, 2) => Self::Volta,
            (7, 5) => Self::Turing,
            (8, 0) | (8, 6) => Self::Ampere,
            (8, 9) => Self::AdaLovelace,
            (9, 0) => Self::Hopper,
            _ if major >= 9 => Self::Hopper, // Future architectures
            _ => Self::Unsupported,
        }
    }

    /// Check if Tensor Cores are supported
    pub fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    /// Get supported data types for this capability
    pub fn supported_dtypes(&self) -> Vec<TensorCoreDType> {
        match self {
            Self::Volta => vec![TensorCoreDType::F16],
            Self::Turing => vec![
                TensorCoreDType::F16,
                TensorCoreDType::Int8,
                TensorCoreDType::Int4,
            ],
            Self::Ampere => vec![
                TensorCoreDType::F16,
                TensorCoreDType::BF16,
                TensorCoreDType::TF32,
                TensorCoreDType::Int8,
                TensorCoreDType::Int4,
                TensorCoreDType::Int1,
            ],
            Self::AdaLovelace => vec![
                TensorCoreDType::F16,
                TensorCoreDType::BF16,
                TensorCoreDType::TF32,
                TensorCoreDType::Int8,
                TensorCoreDType::Int4,
                TensorCoreDType::FP8E4M3,
                TensorCoreDType::FP8E5M2,
            ],
            Self::Hopper => vec![
                TensorCoreDType::F16,
                TensorCoreDType::BF16,
                TensorCoreDType::TF32,
                TensorCoreDType::Int8,
                TensorCoreDType::Int4,
                TensorCoreDType::FP8E4M3,
                TensorCoreDType::FP8E5M2,
            ],
            Self::Unsupported => vec![],
        }
    }

    /// Get maximum matrix dimensions for optimal performance
    pub fn optimal_dimensions(&self) -> (usize, usize, usize) {
        match self {
            Self::Volta => (16, 16, 16),       // 16x16x16 tiles
            Self::Turing => (16, 16, 16),      // 16x16x16 tiles
            Self::Ampere => (16, 16, 16),      // 16x16x16 tiles, some ops support larger
            Self::AdaLovelace => (16, 16, 16), // 16x16x16 tiles
            Self::Hopper => (16, 16, 16),      // 16x16x16 tiles with Hopper extensions
            Self::Unsupported => (1, 1, 1),
        }
    }
}

/// Supported Tensor Core data types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorCoreDType {
    /// IEEE 754 half precision (16-bit)
    F16,
    /// Brain floating point (16-bit)
    BF16,
    /// TensorFloat-32 (Ampere and later)
    TF32,
    /// 8-bit integer
    Int8,
    /// 4-bit integer
    Int4,
    /// 1-bit integer (binary)
    Int1,
    /// FP8 E4M3 format (Hopper and later)
    FP8E4M3,
    /// FP8 E5M2 format (Hopper and later)  
    FP8E5M2,
}

impl TensorCoreDType {
    /// Get the size in bits for this data type
    pub fn size_bits(&self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 16,
            Self::TF32 => 32,
            Self::Int8 | Self::FP8E4M3 | Self::FP8E5M2 => 8,
            Self::Int4 => 4,
            Self::Int1 => 1,
        }
    }

    /// Get the size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        (self.size_bits() + 7) / 8 // Round up to nearest byte
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            Self::F16 | Self::BF16 | Self::TF32 | Self::FP8E4M3 | Self::FP8E5M2
        )
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::Int8 | Self::Int4 | Self::Int1)
    }
}

impl From<TensorCoreDType> for tensor::DType {
    fn from(dtype: TensorCoreDType) -> Self {
        match dtype {
            TensorCoreDType::F16 => tensor::DType::F16,
            TensorCoreDType::BF16 => tensor::DType::BF16,
            TensorCoreDType::TF32 => tensor::DType::F32, // Map TF32 to F32 for compatibility
            TensorCoreDType::Int8 => tensor::DType::I8,
            TensorCoreDType::Int4 => tensor::DType::I8, // Map Int4 to I8 for compatibility
            TensorCoreDType::Int1 => tensor::DType::U8, // Map Int1 to U8 for compatibility
            TensorCoreDType::FP8E4M3 => tensor::DType::F16, // Map FP8 to F16 for compatibility
            TensorCoreDType::FP8E5M2 => tensor::DType::F16, // Map FP8 to F16 for compatibility
        }
    }
}

/// Tensor Core operation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorCoreOp {
    /// Matrix multiplication: C = A * B
    MatMul,
    /// Fused matrix multiplication and addition: C = A * B + C
    MatMulAdd,
    /// Convolution operation
    Convolution,
    /// Multi-head attention
    Attention,
    /// Custom user-defined operation
    Custom,
}

/// Tensor Core matrix multiplication configuration
#[derive(Debug, Clone)]
pub struct TensorCoreGemmConfig {
    /// Matrix A dimensions (M, K)
    pub a_shape: (usize, usize),
    /// Matrix B dimensions (K, N)
    pub b_shape: (usize, usize),
    /// Matrix C dimensions (M, N)
    pub c_shape: (usize, usize),
    /// Leading dimension of A
    pub lda: usize,
    /// Leading dimension of B
    pub ldb: usize,
    /// Leading dimension of C
    pub ldc: usize,
    /// Data type for computation
    pub dtype: TensorCoreDType,
    /// Whether to transpose matrix A
    pub trans_a: bool,
    /// Whether to transpose matrix B
    pub trans_b: bool,
    /// Alpha scaling factor
    pub alpha: f32,
    /// Beta scaling factor (for C = alpha * A * B + beta * C)
    pub beta: f32,
}

impl Default for TensorCoreGemmConfig {
    fn default() -> Self {
        Self {
            a_shape: (16, 16),
            b_shape: (16, 16),
            c_shape: (16, 16),
            lda: 16,
            ldb: 16,
            ldc: 16,
            dtype: TensorCoreDType::F16,
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

/// Tensor Core execution context
pub struct TensorCoreContext {
    /// GPU capability
    capability: TensorCoreCapability,
    /// Whether Tensor Cores are enabled
    enabled: bool,
    /// Performance statistics
    stats: Arc<Mutex<TensorCoreStats>>,
    /// Operation cache for optimization
    op_cache: HashMap<String, TensorCoreGemmConfig>,
    /// SciRS2 CUDA device for actual computation
    scirs2_device: Option<Arc<SciRs2CudaDevice>>,
}

/// Performance statistics for Tensor Core operations
#[derive(Debug, Default)]
pub struct TensorCoreStats {
    /// Total number of Tensor Core operations executed
    pub total_ops: u64,
    /// Total compute time in microseconds
    pub total_compute_time_us: u64,
    /// Total FLOPS computed
    pub total_flops: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
}

impl TensorCoreStats {
    /// Calculate average FLOPS per second
    pub fn avg_flops_per_second(&self) -> f64 {
        if self.total_compute_time_us == 0 {
            0.0
        } else {
            (self.total_flops as f64) / (self.total_compute_time_us as f64 / 1_000_000.0)
        }
    }

    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses == 0 {
            0.0
        } else {
            (self.cache_hits as f64) / (total_accesses as f64)
        }
    }

    /// Calculate average operation time
    pub fn avg_op_time_us(&self) -> f64 {
        if self.total_ops == 0 {
            0.0
        } else {
            (self.total_compute_time_us as f64) / (self.total_ops as f64)
        }
    }
}

impl TensorCoreContext {
    /// Create new Tensor Core context
    pub fn new(compute_major: i32, compute_minor: i32) -> Self {
        let capability =
            TensorCoreCapability::from_compute_capability(compute_major, compute_minor);
        let enabled = capability.is_supported();

        // Try to initialize SciRS2 CUDA device
        let scirs2_device = cuda::get_device(0).ok().map(Arc::new);

        Self {
            capability,
            enabled: enabled && scirs2_device.is_some(),
            stats: Arc::new(Mutex::new(TensorCoreStats::default())),
            op_cache: HashMap::new(),
            scirs2_device,
        }
    }

    /// Create new Tensor Core context with specific device
    pub fn with_device(compute_major: i32, compute_minor: i32, device_id: u32) -> Self {
        let capability =
            TensorCoreCapability::from_compute_capability(compute_major, compute_minor);
        let enabled = capability.is_supported();

        // Try to initialize SciRS2 CUDA device with specific device ID
        let scirs2_device = cuda::get_device(device_id).ok().map(Arc::new);

        Self {
            capability,
            enabled: enabled && scirs2_device.is_some(),
            stats: Arc::new(Mutex::new(TensorCoreStats::default())),
            op_cache: HashMap::new(),
            scirs2_device,
        }
    }

    /// Check if Tensor Cores are available and enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable Tensor Core usage
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled && self.capability.is_supported();
    }

    /// Get the GPU capability
    pub fn capability(&self) -> TensorCoreCapability {
        self.capability
    }

    /// Get performance statistics
    pub fn stats(&self) -> TensorCoreStats {
        self.stats.lock().unwrap().clone()
    }

    /// Reset performance statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = TensorCoreStats::default();
    }

    /// Check if a data type is supported on this GPU
    pub fn supports_dtype(&self, dtype: TensorCoreDType) -> bool {
        self.capability.supported_dtypes().contains(&dtype)
    }

    /// Get optimal tile size for matrix operations
    pub fn optimal_tile_size(&self) -> (usize, usize, usize) {
        self.capability.optimal_dimensions()
    }

    /// Perform Tensor Core matrix multiplication
    pub fn gemm(
        &mut self,
        config: &TensorCoreGemmConfig,
        a_ptr: *const f16,
        b_ptr: *const f16,
        c_ptr: *mut f32,
        stream: &CudaStream,
    ) -> BackendResult<()> {
        if !self.enabled {
            return Err(BackendError::BackendError(
                "Tensor Cores not available or not enabled".to_string(),
            ));
        }

        if !self.supports_dtype(config.dtype) {
            return Err(BackendError::InvalidArgument(format!(
                "Data type {:?} not supported on {:?}",
                config.dtype, self.capability
            )));
        }

        let start_time = std::time::Instant::now();

        // Validate matrix dimensions
        self.validate_gemm_config(config)?;

        // Use cuBLAS with Tensor Core acceleration or custom kernels
        let result = self.launch_tensor_core_gemm(config, a_ptr, b_ptr, c_ptr, stream);

        // Update statistics
        let elapsed_us = start_time.elapsed().as_micros() as u64;
        let flops = 2 * config.a_shape.0 * config.a_shape.1 * config.b_shape.1;

        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_ops += 1;
            stats.total_compute_time_us += elapsed_us;
            stats.total_flops += flops as u64;
        }

        result
    }

    /// Launch Tensor Core GEMM operation
    fn launch_tensor_core_gemm(
        &self,
        config: &TensorCoreGemmConfig,
        a_ptr: *const f16,
        b_ptr: *const f16,
        c_ptr: *mut f32,
        stream: &CudaStream,
    ) -> BackendResult<()> {
        let device = self.scirs2_device.as_ref().ok_or_else(|| {
            BackendError::BackendError("SciRS2 CUDA device not initialized".to_string())
        })?;

        // Create tensor views for SciRS2
        let a_shape = [config.a_shape.0, config.a_shape.1];
        let b_shape = [config.b_shape.0, config.b_shape.1];
        let c_shape = [config.c_shape.0, config.c_shape.1];

        // Use SciRS2's CUDA tensor core GEMM implementation
        unsafe {
            // Create SciRS2 tensor views from raw pointers
            let a_tensor = tensor::from_raw_ptr(
                a_ptr as *const u8,
                &a_shape,
                &[config.a_shape.1, 1], // Row-major strides
                tensor::DType::F16,
            )
            .map_err(|e| BackendError::BackendError(format!("Failed to create tensor A: {}", e)))?;

            let b_tensor = tensor::from_raw_ptr(
                b_ptr as *const u8,
                &b_shape,
                &[config.b_shape.1, 1], // Row-major strides
                tensor::DType::F16,
            )
            .map_err(|e| BackendError::BackendError(format!("Failed to create tensor B: {}", e)))?;

            let mut c_tensor = tensor::from_raw_ptr_mut(
                c_ptr as *mut u8,
                &c_shape,
                &[config.c_shape.1, 1], // Row-major strides
                tensor::DType::F32,
            )
            .map_err(|e| BackendError::BackendError(format!("Failed to create tensor C: {}", e)))?;

            // Perform tensor core GEMM operation via SciRS2
            cuda::tensor_ops::gemm_tensor_core(
                device.as_ref(),
                &a_tensor,
                &b_tensor,
                &mut c_tensor,
                config.alpha,
                config.beta,
                config.trans_a,
                config.trans_b,
                config.dtype.into(),
            )
            .map_err(|e| {
                BackendError::BackendError(format!("SciRS2 tensor core GEMM failed: {}", e))
            })?;
        }

        Ok(())
    }

    /// Validate GEMM configuration
    fn validate_gemm_config(&self, config: &TensorCoreGemmConfig) -> BackendResult<()> {
        // Check matrix dimension compatibility
        if config.a_shape.1 != config.b_shape.0 {
            return Err(BackendError::InvalidArgument(format!(
                "Matrix dimension mismatch: A.cols ({}) != B.rows ({})",
                config.a_shape.1, config.b_shape.0
            )));
        }

        if config.c_shape.0 != config.a_shape.0 || config.c_shape.1 != config.b_shape.1 {
            return Err(BackendError::InvalidArgument(format!(
                "Output matrix dimensions incorrect: expected ({}, {}), got ({}, {})",
                config.a_shape.0, config.b_shape.1, config.c_shape.0, config.c_shape.1
            )));
        }

        // Check alignment requirements for optimal performance
        let (opt_m, opt_n, opt_k) = self.optimal_tile_size();
        if config.a_shape.0 % opt_m != 0
            || config.a_shape.1 % opt_k != 0
            || config.b_shape.1 % opt_n != 0
        {
            eprintln!(
                "Warning: Matrix dimensions not optimally aligned for Tensor Cores. \
                 Consider padding to multiples of {}x{}x{} for best performance.",
                opt_m, opt_n, opt_k
            );
        }

        Ok(())
    }

    /// Perform Tensor Core convolution
    pub fn convolution(
        &mut self,
        input: *const f16,
        weight: *const f16,
        output: *mut f32,
        input_shape: (usize, usize, usize, usize), // (N, C, H, W)
        weight_shape: (usize, usize, usize, usize), // (K, C, H, W)
        output_shape: (usize, usize, usize, usize), // (N, K, H, W)
        padding: (usize, usize),
        stride: (usize, usize),
        stream: &CudaStream,
    ) -> BackendResult<()> {
        if !self.enabled {
            return Err(BackendError::BackendError(
                "Tensor Cores not available or not enabled".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        // Use cuDNN with Tensor Core acceleration
        let result = self.launch_tensor_core_conv(
            input,
            weight,
            output,
            input_shape,
            weight_shape,
            output_shape,
            padding,
            stride,
            stream,
        );

        // Update statistics
        let elapsed_us = start_time.elapsed().as_micros() as u64;
        let flops = 2
            * output_shape.0
            * output_shape.1
            * output_shape.2
            * output_shape.3
            * weight_shape.1
            * weight_shape.2
            * weight_shape.3;

        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_ops += 1;
            stats.total_compute_time_us += elapsed_us;
            stats.total_flops += flops as u64;
        }

        result
    }

    /// Launch Tensor Core convolution
    fn launch_tensor_core_conv(
        &self,
        input: *const f16,
        weight: *const f16,
        output: *mut f32,
        input_shape: (usize, usize, usize, usize),
        weight_shape: (usize, usize, usize, usize),
        output_shape: (usize, usize, usize, usize),
        padding: (usize, usize),
        stride: (usize, usize),
        stream: &CudaStream,
    ) -> BackendResult<()> {
        let device = self.scirs2_device.as_ref().ok_or_else(|| {
            BackendError::BackendError("SciRS2 CUDA device not initialized".to_string())
        })?;

        // Create tensor views for SciRS2
        let input_shape_arr = [input_shape.0, input_shape.1, input_shape.2, input_shape.3];
        let weight_shape_arr = [
            weight_shape.0,
            weight_shape.1,
            weight_shape.2,
            weight_shape.3,
        ];
        let output_shape_arr = [
            output_shape.0,
            output_shape.1,
            output_shape.2,
            output_shape.3,
        ];

        // Use SciRS2's CUDA tensor core convolution implementation
        unsafe {
            // Create SciRS2 tensor views from raw pointers
            let input_tensor = tensor::from_raw_ptr(
                input as *const u8,
                &input_shape_arr,
                &[
                    input_shape.1 * input_shape.2 * input_shape.3,
                    input_shape.2 * input_shape.3,
                    input_shape.3,
                    1,
                ], // NCHW strides
                tensor::DType::F16,
            )
            .map_err(|e| {
                BackendError::BackendError(format!("Failed to create input tensor: {}", e))
            })?;

            let weight_tensor = tensor::from_raw_ptr(
                weight as *const u8,
                &weight_shape_arr,
                &[
                    weight_shape.1 * weight_shape.2 * weight_shape.3,
                    weight_shape.2 * weight_shape.3,
                    weight_shape.3,
                    1,
                ], // KCHW strides
                tensor::DType::F16,
            )
            .map_err(|e| {
                BackendError::BackendError(format!("Failed to create weight tensor: {}", e))
            })?;

            let mut output_tensor = tensor::from_raw_ptr_mut(
                output as *mut u8,
                &output_shape_arr,
                &[
                    output_shape.1 * output_shape.2 * output_shape.3,
                    output_shape.2 * output_shape.3,
                    output_shape.3,
                    1,
                ], // NCHW strides
                tensor::DType::F32,
            )
            .map_err(|e| {
                BackendError::BackendError(format!("Failed to create output tensor: {}", e))
            })?;

            // Perform tensor core convolution operation via SciRS2
            cuda::neural_ops::conv2d_tensor_core(
                device.as_ref(),
                &input_tensor,
                &weight_tensor,
                &mut output_tensor,
                &[padding.0, padding.1, padding.0, padding.1], // [pad_left, pad_right, pad_top, pad_bottom]
                &[stride.0, stride.1],                         // [stride_h, stride_w]
                &[1, 1], // [dilation_h, dilation_w] - default to 1
                1,       // groups - default to 1
            )
            .map_err(|e| {
                BackendError::BackendError(format!("SciRS2 tensor core convolution failed: {}", e))
            })?;
        }

        Ok(())
    }

    /// Auto-tune Tensor Core operations for optimal performance
    pub fn auto_tune(
        &mut self,
        operation: TensorCoreOp,
        input_shapes: &[(usize, usize)],
        stream: &CudaStream,
    ) -> BackendResult<TensorCoreGemmConfig> {
        if !self.enabled {
            return Err(BackendError::BackendError(
                "Tensor Cores not available or not enabled".to_string(),
            ));
        }

        let mut best_config = TensorCoreGemmConfig::default();
        let mut best_time = f64::INFINITY;

        // Test different configurations
        let dtypes = self.capability.supported_dtypes();
        let tile_sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)];

        for &dtype in &dtypes {
            if !dtype.is_float() {
                continue; // Skip integer types for auto-tuning
            }

            for &(m, n, k) in &tile_sizes {
                for &(rows, cols) in input_shapes {
                    // Skip if dimensions don't fit
                    if rows < m || cols < k {
                        continue;
                    }

                    let config = TensorCoreGemmConfig {
                        a_shape: (rows, cols),
                        b_shape: (cols, rows),
                        c_shape: (rows, rows),
                        lda: rows,
                        ldb: cols,
                        ldc: rows,
                        dtype,
                        trans_a: false,
                        trans_b: false,
                        alpha: 1.0,
                        beta: 0.0,
                    };

                    // Benchmark this configuration
                    if let Ok(time) = self.benchmark_config(&config, stream) {
                        if time < best_time {
                            best_time = time;
                            best_config = config;
                        }
                    }
                }
            }
        }

        // Cache the best configuration
        let cache_key = format!("{:?}_{:?}", operation, input_shapes);
        self.op_cache.insert(cache_key, best_config.clone());

        Ok(best_config)
    }

    /// Benchmark a specific configuration
    fn benchmark_config(
        &self,
        config: &TensorCoreGemmConfig,
        stream: &CudaStream,
    ) -> BackendResult<f64> {
        let _ = (config, stream); // Suppress unused warnings

        // In a real implementation, this would:
        // 1. Allocate test matrices
        // 2. Run the operation multiple times
        // 3. Measure average execution time
        // 4. Return the time in seconds

        Ok(0.001) // Placeholder: 1ms
    }

    /// Get or create cached configuration for an operation
    pub fn get_cached_config(
        &mut self,
        operation: TensorCoreOp,
        input_shapes: &[(usize, usize)],
    ) -> Option<&TensorCoreGemmConfig> {
        let cache_key = format!("{:?}_{:?}", operation, input_shapes);

        if self.op_cache.contains_key(&cache_key) {
            {
                let mut stats = self.stats.lock().unwrap();
                stats.cache_hits += 1;
            }
            self.op_cache.get(&cache_key)
        } else {
            {
                let mut stats = self.stats.lock().unwrap();
                stats.cache_misses += 1;
            }
            None
        }
    }
}

/// Utility functions for Tensor Core operations
pub mod utils {
    use super::*;

    /// Check if given dimensions are optimal for Tensor Cores
    pub fn is_tensor_core_optimal(
        m: usize,
        n: usize,
        k: usize,
        capability: TensorCoreCapability,
    ) -> bool {
        let (opt_m, opt_n, opt_k) = capability.optimal_dimensions();
        m % opt_m == 0 && n % opt_n == 0 && k % opt_k == 0
    }

    /// Pad matrix dimensions to be Tensor Core optimal
    pub fn pad_for_tensor_cores(
        original: (usize, usize),
        capability: TensorCoreCapability,
    ) -> (usize, usize) {
        let (opt_m, opt_n, _) = capability.optimal_dimensions();
        let padded_m = ((original.0 + opt_m - 1) / opt_m) * opt_m;
        let padded_n = ((original.1 + opt_n - 1) / opt_n) * opt_n;
        (padded_m, padded_n)
    }

    /// Calculate FLOPS for a matrix multiplication
    pub fn calculate_gemm_flops(m: usize, n: usize, k: usize) -> u64 {
        2 * (m as u64) * (n as u64) * (k as u64)
    }

    /// Get the theoretical peak FLOPS for Tensor Cores on different architectures
    pub fn theoretical_peak_flops(
        capability: TensorCoreCapability,
        clock_mhz: f32,
        sm_count: u32,
    ) -> f64 {
        let flops_per_sm_per_cycle = match capability {
            TensorCoreCapability::Volta => 4096.0,       // FP16 operations
            TensorCoreCapability::Turing => 4096.0,      // FP16 operations
            TensorCoreCapability::Ampere => 8192.0,      // BF16/TF32 operations
            TensorCoreCapability::AdaLovelace => 8192.0, // Enhanced Tensor Cores
            TensorCoreCapability::Hopper => 16384.0,     // FP8 and enhanced operations
            TensorCoreCapability::Unsupported => 0.0,
        };

        (sm_count as f64) * flops_per_sm_per_cycle * (clock_mhz as f64) * 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_core_capability_detection() {
        assert_eq!(
            TensorCoreCapability::from_compute_capability(7, 0),
            TensorCoreCapability::Volta
        );
        assert_eq!(
            TensorCoreCapability::from_compute_capability(7, 5),
            TensorCoreCapability::Turing
        );
        assert_eq!(
            TensorCoreCapability::from_compute_capability(8, 0),
            TensorCoreCapability::Ampere
        );
        assert_eq!(
            TensorCoreCapability::from_compute_capability(6, 1),
            TensorCoreCapability::Unsupported
        );
    }

    #[test]
    fn test_tensor_core_dtype_properties() {
        assert_eq!(TensorCoreDType::F16.size_bits(), 16);
        assert_eq!(TensorCoreDType::BF16.size_bytes(), 2);
        assert!(TensorCoreDType::F16.is_float());
        assert!(TensorCoreDType::Int8.is_integer());
        assert!(!TensorCoreDType::F16.is_integer());
    }

    #[test]
    fn test_tensor_core_context_creation() {
        let mut context = TensorCoreContext::new(8, 0); // Ampere
        assert_eq!(context.capability(), TensorCoreCapability::Ampere);
        assert!(context.is_enabled());
        assert!(context.supports_dtype(TensorCoreDType::F16));
        assert!(context.supports_dtype(TensorCoreDType::BF16));
        assert!(context.supports_dtype(TensorCoreDType::TF32));

        context.set_enabled(false);
        assert!(!context.is_enabled());
    }

    #[test]
    fn test_gemm_config_validation() {
        let context = TensorCoreContext::new(8, 0);

        let valid_config = TensorCoreGemmConfig {
            a_shape: (64, 32),
            b_shape: (32, 64),
            c_shape: (64, 64),
            ..Default::default()
        };

        assert!(context.validate_gemm_config(&valid_config).is_ok());

        let invalid_config = TensorCoreGemmConfig {
            a_shape: (64, 32),
            b_shape: (16, 64), // Incompatible K dimension
            c_shape: (64, 64),
            ..Default::default()
        };

        assert!(context.validate_gemm_config(&invalid_config).is_err());
    }

    #[test]
    fn test_tensor_core_stats() {
        let mut stats = TensorCoreStats::default();
        stats.total_ops = 100;
        stats.total_compute_time_us = 1_000_000; // 1 second
        stats.total_flops = 10_000_000_000; // 10 GFLOPS
        stats.cache_hits = 80;
        stats.cache_misses = 20;

        assert_eq!(stats.avg_flops_per_second(), 10_000_000_000.0);
        assert_eq!(stats.cache_hit_ratio(), 0.8);
        assert_eq!(stats.avg_op_time_us(), 10_000.0);
    }

    #[test]
    fn test_tensor_core_utils() {
        let capability = TensorCoreCapability::Ampere;

        assert!(utils::is_tensor_core_optimal(16, 16, 16, capability));
        assert!(utils::is_tensor_core_optimal(32, 32, 32, capability));
        assert!(!utils::is_tensor_core_optimal(15, 16, 16, capability));

        assert_eq!(utils::pad_for_tensor_cores((15, 17), capability), (16, 32));
        assert_eq!(utils::calculate_gemm_flops(16, 16, 16), 8192);

        let peak_flops = utils::theoretical_peak_flops(capability, 1500.0, 108);
        assert!(peak_flops > 0.0);
    }
}
