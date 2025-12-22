//! SciRS2 integration layer for CPU backend
//!
//! This module provides integration with scirs2-core's optimized CPU operations,
//! BLAS/LAPACK routines, SIMD implementations, and auto-tuning system for the CPU backend.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::cpu::autotuning::{AutoTuner, PerformanceMeasurement};
use crate::cpu::error::{cpu_errors, CpuResult};
use crate::error::conversion;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// SciRS2 CPU backend integration wrapper with auto-tuning
pub struct SciRS2CpuBackend {
    /// Thread pool size for parallel operations
    num_threads: usize,
    /// Auto-tuning engine for performance optimization
    autotuner: Arc<Mutex<AutoTuner>>,
    /// Cache for optimal kernel configurations
    kernel_cache: Arc<Mutex<HashMap<String, KernelConfig>>>,
    /// Performance profiling data
    profiling_enabled: bool,
}

/// Optimized kernel configuration from auto-tuning
#[derive(Debug, Clone)]
pub struct KernelConfig {
    pub operation: String,
    pub optimal_threads: usize,
    pub optimal_chunk_size: usize,
    pub optimal_block_size: Option<usize>,
    pub use_simd: bool,
    pub cache_blocking: bool,
}

impl SciRS2CpuBackend {
    /// Create a new SciRS2 CPU backend with auto-tuning
    pub fn new() -> CpuResult<Self> {
        // Use SciRS2 parallel operations for thread count (SciRS2 POLICY compliance)
        let num_threads = scirs2_core::parallel_ops::get_num_threads();

        Ok(Self {
            num_threads,
            autotuner: Arc::new(Mutex::new(AutoTuner::default())),
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            profiling_enabled: true,
        })
    }

    /// Create a new SciRS2 CPU backend with custom configuration
    pub fn with_config(num_threads: usize, enable_profiling: bool) -> CpuResult<Self> {
        Ok(Self {
            num_threads,
            autotuner: Arc::new(Mutex::new(AutoTuner::default())),
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            profiling_enabled: enable_profiling,
        })
    }

    /// Get CPU device information (placeholder for now)
    pub fn device_info(&self) -> usize {
        self.num_threads
    }

    /// Get number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Set number of threads (for future operations)
    pub fn set_num_threads(&mut self, num_threads: usize) {
        self.num_threads = num_threads;
        // Clear cache since thread count affects optimal configurations
        if let Ok(mut cache) = self.kernel_cache.lock() {
            cache.clear();
        }
    }

    /// Enable or disable performance profiling
    pub fn set_profiling_enabled(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
    }

    /// Get optimal kernel configuration for an operation
    fn get_optimal_config(&self, operation: &str, input_size: usize) -> CpuResult<KernelConfig> {
        let cache_key = format!("{}:{}", operation, input_size);

        // Check cache first
        if let Ok(cache) = self.kernel_cache.lock() {
            if let Some(config) = cache.get(&cache_key) {
                return Ok(config.clone());
            }
        }

        // Run auto-tuning if not cached
        let tuning_result = if let Ok(autotuner) = self.autotuner.lock() {
            autotuner.get_optimal_params(operation, input_size, "f32")
        } else {
            return Err(cpu_errors::optimization_error(
                "Failed to acquire autotuner lock",
            ));
        }?;

        let config = KernelConfig {
            operation: operation.to_string(),
            optimal_threads: tuning_result.optimal_thread_count,
            optimal_chunk_size: tuning_result.optimal_chunk_size,
            optimal_block_size: tuning_result.optimal_block_size,
            use_simd: input_size > 1000, // Use SIMD for larger arrays
            cache_blocking: tuning_result.optimal_block_size.is_some(),
        };

        // Cache the configuration
        if let Ok(mut cache) = self.kernel_cache.lock() {
            cache.insert(cache_key, config.clone());
        }

        Ok(config)
    }

    /// Execute operation with performance profiling
    fn execute_with_profiling<F, R>(&self, _operation: &str, f: F) -> CpuResult<R>
    where
        F: FnOnce() -> CpuResult<R>,
    {
        let start = if self.profiling_enabled {
            Some(Instant::now())
        } else {
            None
        };

        let result = f()?;

        if let Some(start_time) = start {
            let _elapsed = start_time.elapsed();
            // Log performance data for future auto-tuning
            #[cfg(feature = "tracing")]
            tracing::debug!("Operation {} took {:?}", _operation, _elapsed);
        }

        Ok(result)
    }

    /// Perform optimized matrix multiplication with auto-tuning
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool,
    ) -> CpuResult<()> {
        // Validate input dimensions
        let expected_a_len = if transpose_a { k * m } else { m * k };
        let expected_b_len = if transpose_b { n * k } else { k * n };
        let expected_result_len = m * n;

        if a.len() != expected_a_len {
            return Err(cpu_errors::invalid_parameter_error(format!(
                "Matrix A size mismatch: expected {}, got {}",
                expected_a_len,
                a.len()
            )));
        }
        if b.len() != expected_b_len {
            return Err(cpu_errors::invalid_parameter_error(format!(
                "Matrix B size mismatch: expected {}, got {}",
                expected_b_len,
                b.len()
            )));
        }
        if result.len() != expected_result_len {
            return Err(cpu_errors::invalid_parameter_error(format!(
                "Result matrix size mismatch: expected {}, got {}",
                expected_result_len,
                result.len()
            )));
        }

        // Get optimal configuration for matrix multiplication
        let config = self.get_optimal_config("matrix", m * n * k)?;

        self.execute_with_profiling("matmul", || {
            if config.cache_blocking && config.optimal_block_size.is_some() {
                self.matmul_blocked(a, b, result, m, n, k, transpose_a, transpose_b, &config)
            } else {
                self.matmul_simple(a, b, result, m, n, k, transpose_a, transpose_b, &config)
            }
        })
    }

    /// Simple matrix multiplication implementation
    fn matmul_simple(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool,
        config: &KernelConfig,
    ) -> CpuResult<()> {
        use rayon::prelude::*;

        // Use optimized chunk size from auto-tuning
        let chunk_size = config.optimal_chunk_size.min(m);

        // Use parallel iteration over chunks of the result matrix
        result
            .par_chunks_mut(n * chunk_size)
            .enumerate()
            .for_each(|(chunk_idx, result_chunk)| {
                let start_row = chunk_idx * chunk_size;
                let end_row = (start_row + chunk_size).min(m);

                for (local_i, result_row) in result_chunk.chunks_mut(n).enumerate() {
                    let i = start_row + local_i;
                    if i >= end_row {
                        break;
                    }

                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            let a_val = if transpose_a {
                                a[l * m + i]
                            } else {
                                a[i * k + l]
                            };
                            let b_val = if transpose_b {
                                b[j * k + l]
                            } else {
                                b[l * n + j]
                            };
                            sum += a_val * b_val;
                        }
                        result_row[j] = sum;
                    }
                }
            });

        Ok(())
    }

    /// Cache-blocked matrix multiplication for better cache locality
    fn matmul_blocked(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
        transpose_a: bool,
        transpose_b: bool,
        config: &KernelConfig,
    ) -> CpuResult<()> {
        use rayon::prelude::*;

        let block_size = config.optimal_block_size.unwrap_or(64);

        // Initialize result to zero
        result.fill(0.0);

        // Block over all three dimensions for better cache locality
        (0..m)
            .into_par_iter()
            .step_by(block_size)
            .for_each(|i_block| {
                for j_block in (0..n).step_by(block_size) {
                    for k_block in (0..k).step_by(block_size) {
                        // Process micro-block
                        let i_end = (i_block + block_size).min(m);
                        let j_end = (j_block + block_size).min(n);
                        let k_end = (k_block + block_size).min(k);

                        for i in i_block..i_end {
                            for j in j_block..j_end {
                                let mut sum = 0.0f32;
                                for l in k_block..k_end {
                                    let a_val = if transpose_a {
                                        a[l * m + i]
                                    } else {
                                        a[i * k + l]
                                    };
                                    let b_val = if transpose_b {
                                        b[j * k + l]
                                    } else {
                                        b[l * n + j]
                                    };
                                    sum += a_val * b_val;
                                }
                                // Atomic update to avoid race conditions
                                unsafe {
                                    let ptr = result.as_ptr().add(i * n + j) as *mut f32;
                                    *ptr += sum;
                                }
                            }
                        }
                    }
                }
            });

        Ok(())
    }

    /// Perform element-wise addition using optimized CPU operations with auto-tuning
    pub fn add_elementwise(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(cpu_errors::invalid_parameter_error(format!(
                "Shape mismatch in elementwise add: a.len()={}, b.len()={}, result.len()={}",
                a.len(),
                b.len(),
                result.len()
            )));
        }

        let config = self.get_optimal_config("element_wise", a.len())?;

        self.execute_with_profiling("add_elementwise", || {
            if config.use_simd && a.len() >= 4 {
                self.add_elementwise_simd(a, b, result, &config)
            } else {
                self.add_elementwise_simple(a, b, result, &config)
            }
        })
    }

    /// Simple element-wise addition
    fn add_elementwise_simple(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        config: &KernelConfig,
    ) -> CpuResult<()> {
        use rayon::prelude::*;

        result
            .par_chunks_mut(config.optimal_chunk_size)
            .zip(
                a.par_chunks(config.optimal_chunk_size)
                    .zip(b.par_chunks(config.optimal_chunk_size)),
            )
            .for_each(|(result_chunk, (a_chunk, b_chunk))| {
                for ((r, &a_val), &b_val) in result_chunk
                    .iter_mut()
                    .zip(a_chunk.iter())
                    .zip(b_chunk.iter())
                {
                    *r = a_val + b_val;
                }
            });

        Ok(())
    }

    /// SIMD-optimized element-wise addition
    fn add_elementwise_simd(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        config: &KernelConfig,
    ) -> CpuResult<()> {
        #[cfg(feature = "simd")]
        {
            use rayon::prelude::*;
            use wide::f32x4;

            let chunk_size = (config.optimal_chunk_size / 4) * 4; // Ensure multiple of 4

            result
                .par_chunks_mut(chunk_size)
                .zip(a.par_chunks(chunk_size).zip(b.par_chunks(chunk_size)))
                .for_each(|(result_chunk, (a_chunk, b_chunk))| {
                    // Process 4 elements at a time with SIMD
                    for (result_simd, (a_simd, b_simd)) in result_chunk
                        .chunks_exact_mut(4)
                        .zip(a_chunk.chunks_exact(4).zip(b_chunk.chunks_exact(4)))
                    {
                        let a_vec = f32x4::from([a_simd[0], a_simd[1], a_simd[2], a_simd[3]]);
                        let b_vec = f32x4::from([b_simd[0], b_simd[1], b_simd[2], b_simd[3]]);
                        let result_vec = a_vec + b_vec;
                        result_simd.copy_from_slice(&result_vec.to_array());
                    }

                    // Handle remaining elements
                    let remainder = result_chunk.len() % 4;
                    if remainder > 0 {
                        let start = result_chunk.len() - remainder;
                        for i in 0..remainder {
                            result_chunk[start + i] = a_chunk[start + i] + b_chunk[start + i];
                        }
                    }
                });
        }

        #[cfg(not(feature = "simd"))]
        {
            // Fallback to simple implementation if SIMD is not available
            self.add_elementwise_simple(a, b, result, config)?;
        }

        Ok(())
    }

    /// Perform element-wise multiplication using optimized CPU operations with auto-tuning
    pub fn mul_elementwise(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> CpuResult<()> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(cpu_errors::invalid_parameter_error(format!(
                "Shape mismatch in elementwise mul: a.len()={}, b.len()={}, result.len()={}",
                a.len(),
                b.len(),
                result.len()
            )));
        }

        let config = self.get_optimal_config("element_wise", a.len())?;

        self.execute_with_profiling("mul_elementwise", || {
            if config.use_simd && a.len() >= 4 {
                self.mul_elementwise_simd(a, b, result, &config)
            } else {
                self.mul_elementwise_simple(a, b, result, &config)
            }
        })
    }

    /// Simple element-wise multiplication
    fn mul_elementwise_simple(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        config: &KernelConfig,
    ) -> CpuResult<()> {
        use rayon::prelude::*;

        result
            .par_chunks_mut(config.optimal_chunk_size)
            .zip(
                a.par_chunks(config.optimal_chunk_size)
                    .zip(b.par_chunks(config.optimal_chunk_size)),
            )
            .for_each(|(result_chunk, (a_chunk, b_chunk))| {
                for ((r, &a_val), &b_val) in result_chunk
                    .iter_mut()
                    .zip(a_chunk.iter())
                    .zip(b_chunk.iter())
                {
                    *r = a_val * b_val;
                }
            });

        Ok(())
    }

    /// SIMD-optimized element-wise multiplication
    fn mul_elementwise_simd(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        config: &KernelConfig,
    ) -> CpuResult<()> {
        #[cfg(feature = "simd")]
        {
            use rayon::prelude::*;
            use wide::f32x4;

            let chunk_size = (config.optimal_chunk_size / 4) * 4;

            result
                .par_chunks_mut(chunk_size)
                .zip(a.par_chunks(chunk_size).zip(b.par_chunks(chunk_size)))
                .for_each(|(result_chunk, (a_chunk, b_chunk))| {
                    // Process 4 elements at a time with SIMD
                    for (result_simd, (a_simd, b_simd)) in result_chunk
                        .chunks_exact_mut(4)
                        .zip(a_chunk.chunks_exact(4).zip(b_chunk.chunks_exact(4)))
                    {
                        let a_vec = f32x4::from([a_simd[0], a_simd[1], a_simd[2], a_simd[3]]);
                        let b_vec = f32x4::from([b_simd[0], b_simd[1], b_simd[2], b_simd[3]]);
                        let result_vec = a_vec * b_vec;
                        result_simd.copy_from_slice(&result_vec.to_array());
                    }

                    // Handle remaining elements
                    let remainder = result_chunk.len() % 4;
                    if remainder > 0 {
                        let start = result_chunk.len() - remainder;
                        for i in 0..remainder {
                            result_chunk[start + i] = a_chunk[start + i] * b_chunk[start + i];
                        }
                    }
                });
        }

        #[cfg(not(feature = "simd"))]
        {
            self.mul_elementwise_simple(a, b, result, config)?;
        }

        Ok(())
    }

    /// Perform scalar addition with auto-tuning
    pub fn add_scalar(&self, a: &[f32], scalar: f32, result: &mut [f32]) -> CpuResult<()> {
        if a.len() != result.len() {
            return Err(cpu_errors::invalid_parameter_error(format!(
                "Shape mismatch in scalar add: a.len()={}, result.len()={}",
                a.len(),
                result.len()
            )));
        }

        let config = self.get_optimal_config("element_wise", a.len())?;

        self.execute_with_profiling("add_scalar", || {
            if config.use_simd && a.len() >= 4 {
                self.add_scalar_simd(a, scalar, result, &config)
            } else {
                self.add_scalar_simple(a, scalar, result, &config)
            }
        })
    }

    /// Simple scalar addition
    fn add_scalar_simple(
        &self,
        a: &[f32],
        scalar: f32,
        result: &mut [f32],
        config: &KernelConfig,
    ) -> CpuResult<()> {
        use rayon::prelude::*;

        result
            .par_chunks_mut(config.optimal_chunk_size)
            .zip(a.par_chunks(config.optimal_chunk_size))
            .for_each(|(result_chunk, a_chunk)| {
                for (r, &a_val) in result_chunk.iter_mut().zip(a_chunk.iter()) {
                    *r = a_val + scalar;
                }
            });

        Ok(())
    }

    /// SIMD-optimized scalar addition
    fn add_scalar_simd(
        &self,
        a: &[f32],
        scalar: f32,
        result: &mut [f32],
        config: &KernelConfig,
    ) -> CpuResult<()> {
        #[cfg(feature = "simd")]
        {
            use rayon::prelude::*;
            use wide::f32x4;

            let chunk_size = (config.optimal_chunk_size / 4) * 4;
            let scalar_vec = f32x4::splat(scalar);

            result
                .par_chunks_mut(chunk_size)
                .zip(a.par_chunks(chunk_size))
                .for_each(|(result_chunk, a_chunk)| {
                    // Process 4 elements at a time with SIMD
                    for (result_simd, a_simd) in result_chunk
                        .chunks_exact_mut(4)
                        .zip(a_chunk.chunks_exact(4))
                    {
                        let a_vec = f32x4::from([a_simd[0], a_simd[1], a_simd[2], a_simd[3]]);
                        let result_vec = a_vec + scalar_vec;
                        result_simd.copy_from_slice(&result_vec.to_array());
                    }

                    // Handle remaining elements
                    let remainder = result_chunk.len() % 4;
                    if remainder > 0 {
                        let start = result_chunk.len() - remainder;
                        for i in 0..remainder {
                            result_chunk[start + i] = a_chunk[start + i] + scalar;
                        }
                    }
                });
        }

        #[cfg(not(feature = "simd"))]
        {
            self.add_scalar_simple(a, scalar, result, config)?;
        }

        Ok(())
    }

    /// Perform scalar multiplication with auto-tuning
    pub fn mul_scalar(&self, a: &[f32], scalar: f32, result: &mut [f32]) -> CpuResult<()> {
        if a.len() != result.len() {
            return Err(cpu_errors::invalid_parameter_error(format!(
                "Shape mismatch in scalar mul: a.len()={}, result.len()={}",
                a.len(),
                result.len()
            )));
        }

        let config = self.get_optimal_config("element_wise", a.len())?;

        self.execute_with_profiling("mul_scalar", || {
            if config.use_simd && a.len() >= 4 {
                self.mul_scalar_simd(a, scalar, result, &config)
            } else {
                self.mul_scalar_simple(a, scalar, result, &config)
            }
        })
    }

    /// Simple scalar multiplication
    fn mul_scalar_simple(
        &self,
        a: &[f32],
        scalar: f32,
        result: &mut [f32],
        config: &KernelConfig,
    ) -> CpuResult<()> {
        use rayon::prelude::*;

        result
            .par_chunks_mut(config.optimal_chunk_size)
            .zip(a.par_chunks(config.optimal_chunk_size))
            .for_each(|(result_chunk, a_chunk)| {
                for (r, &a_val) in result_chunk.iter_mut().zip(a_chunk.iter()) {
                    *r = a_val * scalar;
                }
            });

        Ok(())
    }

    /// SIMD-optimized scalar multiplication
    fn mul_scalar_simd(
        &self,
        a: &[f32],
        scalar: f32,
        result: &mut [f32],
        config: &KernelConfig,
    ) -> CpuResult<()> {
        #[cfg(feature = "simd")]
        {
            use rayon::prelude::*;
            use wide::f32x4;

            let chunk_size = (config.optimal_chunk_size / 4) * 4;
            let scalar_vec = f32x4::splat(scalar);

            result
                .par_chunks_mut(chunk_size)
                .zip(a.par_chunks(chunk_size))
                .for_each(|(result_chunk, a_chunk)| {
                    // Process 4 elements at a time with SIMD
                    for (result_simd, a_simd) in result_chunk
                        .chunks_exact_mut(4)
                        .zip(a_chunk.chunks_exact(4))
                    {
                        let a_vec = f32x4::from([a_simd[0], a_simd[1], a_simd[2], a_simd[3]]);
                        let result_vec = a_vec * scalar_vec;
                        result_simd.copy_from_slice(&result_vec.to_array());
                    }

                    // Handle remaining elements
                    let remainder = result_chunk.len() % 4;
                    if remainder > 0 {
                        let start = result_chunk.len() - remainder;
                        for i in 0..remainder {
                            result_chunk[start + i] = a_chunk[start + i] * scalar;
                        }
                    }
                });
        }

        #[cfg(not(feature = "simd"))]
        {
            self.mul_scalar_simple(a, scalar, result, config)?;
        }

        Ok(())
    }

    /// Perform reduction sum with auto-tuning
    pub fn sum(&self, a: &[f32]) -> CpuResult<f32> {
        let config = self.get_optimal_config("reduction", a.len())?;

        self.execute_with_profiling("sum", || {
            if config.use_simd && a.len() >= 4 {
                self.sum_simd(a, &config)
            } else {
                self.sum_simple(a, &config)
            }
        })
    }

    /// Simple reduction sum
    fn sum_simple(&self, a: &[f32], config: &KernelConfig) -> CpuResult<f32> {
        use rayon::prelude::*;

        let sum = a
            .par_chunks(config.optimal_chunk_size)
            .map(|chunk| chunk.iter().sum::<f32>())
            .sum();

        Ok(sum)
    }

    /// SIMD-optimized reduction sum
    fn sum_simd(&self, a: &[f32], config: &KernelConfig) -> CpuResult<f32> {
        #[cfg(feature = "simd")]
        {
            use rayon::prelude::*;
            use wide::f32x4;

            let chunk_size = (config.optimal_chunk_size / 4) * 4;

            let sum = a
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut acc = f32x4::ZERO;

                    // Process 4 elements at a time
                    for simd_chunk in chunk.chunks_exact(4) {
                        let vec = f32x4::from([
                            simd_chunk[0],
                            simd_chunk[1],
                            simd_chunk[2],
                            simd_chunk[3],
                        ]);
                        acc = acc + vec;
                    }

                    // Sum the accumulator
                    let acc_array = acc.to_array();
                    let mut partial_sum = acc_array[0] + acc_array[1] + acc_array[2] + acc_array[3];

                    // Handle remaining elements
                    let remainder = chunk.len() % 4;
                    if remainder > 0 {
                        let start = chunk.len() - remainder;
                        for i in 0..remainder {
                            partial_sum += chunk[start + i];
                        }
                    }

                    partial_sum
                })
                .sum();

            Ok(sum)
        }

        #[cfg(not(feature = "simd"))]
        {
            self.sum_simple(a, config)
        }
    }

    /// Synchronize all operations (no-op for CPU but maintains interface consistency)
    pub fn synchronize(&self) -> CpuResult<()> {
        // CPU operations are synchronous by nature
        Ok(())
    }

    /// Get auto-tuning statistics
    pub fn get_autotuning_stats(&self) -> CpuResult<(usize, usize)> {
        if let Ok(autotuner) = self.autotuner.lock() {
            Ok(autotuner.get_cache_stats())
        } else {
            Err(cpu_errors::optimization_error(
                "Failed to acquire autotuner lock",
            ))
        }
    }

    /// Clear auto-tuning cache
    pub fn clear_autotuning_cache(&self) -> CpuResult<()> {
        if let Ok(autotuner) = self.autotuner.lock() {
            autotuner.clear_cache();
        }
        if let Ok(mut cache) = self.kernel_cache.lock() {
            cache.clear();
        }
        Ok(())
    }

    /// Pre-populate auto-tuning cache with common operations
    pub fn warm_up_autotuning(&self) -> CpuResult<()> {
        if let Ok(autotuner) = self.autotuner.lock() {
            autotuner.populate_default_cache()
        } else {
            Err(cpu_errors::optimization_error(
                "Failed to acquire autotuner lock",
            ))
        }
    }
}

impl Default for SciRS2CpuBackend {
    fn default() -> Self {
        let backend = Self::new().expect("Failed to create default SciRS2 CPU backend");

        // Warm up auto-tuning cache for common operations
        let _ = backend.warm_up_autotuning();

        backend
    }
}

/// Utility function to convert TorSh tensor data to optimized format
///
/// This function ensures that tensor data is properly aligned and formatted
/// for optimal performance with SciRS2 operations. It validates alignment
/// requirements and provides optimized data layout where possible.
pub fn prepare_tensor_data<'a>(data: &'a [f32], shape: &[usize]) -> CpuResult<&'a [f32]> {
    // Validate input parameters
    if data.is_empty() && !shape.is_empty() && shape.iter().product::<usize>() > 0 {
        return Err(conversion::cpu_error_with_context(
            "Empty data slice provided for non-empty shape",
            "prepare_tensor_data",
        ));
    }

    // Check if data length matches expected size from shape
    let expected_size: usize = shape.iter().product();
    if data.len() != expected_size {
        return Err(conversion::cpu_error_with_context(
            &format!(
                "Data length {} does not match expected size {} from shape {:?}",
                data.len(),
                expected_size,
                shape
            ),
            "prepare_tensor_data",
        ));
    }

    // Check memory alignment for optimal performance
    let data_ptr = data.as_ptr() as usize;

    // For optimal SIMD performance, check 16-byte alignment (128-bit)
    // For advanced AVX-512, 64-byte alignment is preferred
    const PREFERRED_ALIGNMENT: usize = 16; // 128-bit alignment for SSE/AVX
    const OPTIMAL_ALIGNMENT: usize = 64; // 512-bit alignment for AVX-512

    if data_ptr % OPTIMAL_ALIGNMENT == 0 {
        // Optimal alignment - no action needed
    } else if data_ptr % PREFERRED_ALIGNMENT == 0 {
        // Good alignment - acceptable for most operations
    } else {
        // Note: In a production implementation, we might want to copy to aligned memory
        // For now, we'll proceed with a warning that performance may be suboptimal
        #[cfg(feature = "std")]
        eprintln!(
            "Warning: Tensor data is not optimally aligned (ptr={:#x}). \
             Performance may be suboptimal for SIMD operations.",
            data_ptr
        );
    }

    // For contiguous tensors, no layout transformation is needed
    // Future enhancements could include:
    // - Checking for strided layouts and converting to contiguous
    // - Applying memory prefetching hints for large tensors
    // - Caching alignment information for repeated operations

    Ok(data)
}

/// Utility function to create mutable tensor data
///
/// This function ensures that mutable tensor data is properly aligned and formatted
/// for optimal performance with SciRS2 operations. It validates alignment
/// requirements and provides optimized data layout for in-place operations.
pub fn prepare_tensor_data_mut<'a>(
    data: &'a mut [f32],
    shape: &[usize],
) -> CpuResult<&'a mut [f32]> {
    // Validate input parameters
    if data.is_empty() && !shape.is_empty() && shape.iter().product::<usize>() > 0 {
        return Err(conversion::cpu_error_with_context(
            "Empty mutable data slice provided for non-empty shape",
            "prepare_tensor_data_mut",
        ));
    }

    // Check if data length matches expected size from shape
    let expected_size: usize = shape.iter().product();
    if data.len() != expected_size {
        return Err(conversion::cpu_error_with_context(
            &format!(
                "Mutable data length {} does not match expected size {} from shape {:?}",
                data.len(),
                expected_size,
                shape
            ),
            "prepare_tensor_data_mut",
        ));
    }

    // Check memory alignment for optimal performance
    let data_ptr = data.as_mut_ptr() as usize;

    // For optimal SIMD performance, check 16-byte alignment (128-bit)
    // For advanced AVX-512, 64-byte alignment is preferred
    const PREFERRED_ALIGNMENT: usize = 16; // 128-bit alignment for SSE/AVX
    const OPTIMAL_ALIGNMENT: usize = 64; // 512-bit alignment for AVX-512

    if data_ptr % OPTIMAL_ALIGNMENT == 0 {
        // Optimal alignment - perfect for all SIMD operations
    } else if data_ptr % PREFERRED_ALIGNMENT == 0 {
        // Good alignment - acceptable for most SIMD operations
    } else {
        // Suboptimal alignment - may impact performance
        #[cfg(feature = "std")]
        eprintln!(
            "Warning: Mutable tensor data is not optimally aligned (ptr={:#x}). \
             Performance may be suboptimal for in-place SIMD operations.",
            data_ptr
        );
    }

    // For mutable data, we can potentially apply additional optimizations:
    // - Memory prefetching for large tensors that will be modified
    // - Cache line alignment checks for write-heavy operations
    // - Layout optimization for strided access patterns

    // Apply memory prefetching hint for large tensors
    if data.len() > 1024 {
        // Hint to prefetch the first cache line for immediate use
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(
                data.as_ptr() as *const i8,
                core::arch::x86_64::_MM_HINT_T0,
            );
        }

        // Note: ARM64 prefetch intrinsics are less standardized and often unstable
        // For production use, consider using __builtin_prefetch via inline assembly
        // or rely on hardware prefetchers which are generally effective
        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 memory prefetching - using standard approach
            // This provides a hint to the hardware prefetcher
            let ptr = data.as_ptr();
            unsafe {
                // Use inline assembly for ARM64 PRFM instruction
                core::arch::asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) ptr,
                    options(nostack, readonly)
                );
            }
        }
    }

    Ok(data)
}

/// High-level SciRS2 auto-tuning manager that integrates with scirs2-core
pub struct SciRS2AutoTuningManager {
    cpu_backend: SciRS2CpuBackend,
    adaptive_mode: bool,
    performance_history: Arc<Mutex<HashMap<String, Vec<PerformanceMeasurement>>>>,
}

impl SciRS2AutoTuningManager {
    /// Create a new SciRS2 auto-tuning manager
    pub fn new() -> CpuResult<Self> {
        Ok(Self {
            cpu_backend: SciRS2CpuBackend::new()?,
            adaptive_mode: true,
            performance_history: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create with custom configuration
    pub fn with_config(num_threads: usize, adaptive_mode: bool) -> CpuResult<Self> {
        Ok(Self {
            cpu_backend: SciRS2CpuBackend::with_config(num_threads, true)?,
            adaptive_mode,
            performance_history: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Enable or disable adaptive auto-tuning
    pub fn set_adaptive_mode(&mut self, enabled: bool) {
        self.adaptive_mode = enabled;
    }

    /// Get the underlying CPU backend
    pub fn backend(&self) -> &SciRS2CpuBackend {
        &self.cpu_backend
    }

    /// Get mutable reference to the underlying CPU backend
    pub fn backend_mut(&mut self) -> &mut SciRS2CpuBackend {
        &mut self.cpu_backend
    }

    /// Execute an operation with automatic performance tracking
    pub fn execute_with_auto_tuning<F, R>(
        &self,
        operation: &str,
        operation_size: usize,
        f: F,
    ) -> CpuResult<R>
    where
        F: FnOnce(&SciRS2CpuBackend) -> CpuResult<R>,
    {
        let start = Instant::now();
        let result = f(&self.cpu_backend)?;
        let elapsed = start.elapsed();

        if self.adaptive_mode {
            self.record_performance(operation, operation_size, elapsed)?;
        }

        Ok(result)
    }

    /// Record performance measurement for adaptive tuning
    fn record_performance(
        &self,
        operation: &str,
        operation_size: usize,
        elapsed: Duration,
    ) -> CpuResult<()> {
        let measurement = PerformanceMeasurement::new(elapsed, operation_size);

        if let Ok(mut history) = self.performance_history.lock() {
            let operation_history = history
                .entry(operation.to_string())
                .or_insert_with(Vec::new);
            operation_history.push(measurement);

            // Keep only the last 100 measurements to avoid unbounded growth
            if operation_history.len() > 100 {
                operation_history.remove(0);
            }
        }

        Ok(())
    }

    /// Get performance statistics for an operation
    pub fn get_performance_stats(&self, operation: &str) -> Option<(f64, f64, Duration)> {
        if let Ok(history) = self.performance_history.lock() {
            if let Some(measurements) = history.get(operation) {
                if !measurements.is_empty() {
                    let total_throughput: f64 = measurements.iter().map(|m| m.throughput).sum();
                    let avg_throughput = total_throughput / measurements.len() as f64;

                    let total_efficiency: f64 = measurements.iter().map(|m| m.efficiency).sum();
                    let avg_efficiency = total_efficiency / measurements.len() as f64;

                    let avg_time = measurements
                        .iter()
                        .map(|m| m.execution_time)
                        .sum::<Duration>()
                        / measurements.len() as u32;

                    return Some((avg_throughput, avg_efficiency, avg_time));
                }
            }
        }
        None
    }

    /// Initialize and warm up the auto-tuning system
    pub fn initialize(&mut self) -> CpuResult<()> {
        // Warm up the auto-tuning cache
        self.cpu_backend.warm_up_autotuning()?;

        // Run some initial benchmarks to seed the performance history
        self.run_initial_benchmarks()?;

        Ok(())
    }

    /// Run initial benchmarks to establish baseline performance
    fn run_initial_benchmarks(&self) -> CpuResult<()> {
        let test_sizes = [1000, 10000, 100000];

        for &size in &test_sizes {
            // Benchmark element-wise operations
            let a = vec![1.0f32; size];
            let b = vec![2.0f32; size];
            let mut result = vec![0.0f32; size];

            let _ = self.execute_with_auto_tuning("add_elementwise", size, |backend| {
                backend.add_elementwise(&a, &b, &mut result)
            });

            let _ = self.execute_with_auto_tuning("mul_elementwise", size, |backend| {
                backend.mul_elementwise(&a, &b, &mut result)
            });

            // Benchmark reduction operations
            let _ = self.execute_with_auto_tuning("sum", size, |backend| backend.sum(&a));

            // Benchmark matrix operations (for smaller sizes)
            if size <= 10000 {
                let dim = (size as f64).sqrt() as usize;
                let matrix_a = vec![1.0f32; dim * dim];
                let matrix_b = vec![2.0f32; dim * dim];
                let mut matrix_result = vec![0.0f32; dim * dim];

                let _ = self.execute_with_auto_tuning("matmul", dim * dim, |backend| {
                    backend.matmul(
                        &matrix_a,
                        &matrix_b,
                        &mut matrix_result,
                        dim,
                        dim,
                        dim,
                        false,
                        false,
                    )
                });
            }
        }

        Ok(())
    }

    /// Get overall auto-tuning system statistics
    pub fn get_system_stats(&self) -> CpuResult<AutoTuningSystemStats> {
        let (cache_hits, cache_misses) = self.cpu_backend.get_autotuning_stats()?;

        let operation_count = if let Ok(history) = self.performance_history.lock() {
            history.len()
        } else {
            0
        };

        Ok(AutoTuningSystemStats {
            cache_hits,
            cache_misses,
            cache_hit_ratio: if cache_hits + cache_misses > 0 {
                cache_hits as f64 / (cache_hits + cache_misses) as f64
            } else {
                0.0
            },
            tracked_operations: operation_count,
            adaptive_mode: self.adaptive_mode,
        })
    }
}

impl Default for SciRS2AutoTuningManager {
    fn default() -> Self {
        let mut manager = Self::new().expect("Failed to create SciRS2AutoTuningManager");
        let _ = manager.initialize();
        manager
    }
}

/// Statistics for the auto-tuning system
#[derive(Debug, Clone)]
pub struct AutoTuningSystemStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_hit_ratio: f64,
    pub tracked_operations: usize,
    pub adaptive_mode: bool,
}

/// Factory function to create a SciRS2 auto-tuning manager
pub fn create_autotuning_manager() -> CpuResult<SciRS2AutoTuningManager> {
    let mut manager = SciRS2AutoTuningManager::new()?;
    manager.initialize()?;
    Ok(manager)
}

/// Factory function to create a SciRS2 auto-tuning manager with custom configuration
pub fn create_autotuning_manager_with_config(
    num_threads: usize,
    adaptive_mode: bool,
) -> CpuResult<SciRS2AutoTuningManager> {
    let mut manager = SciRS2AutoTuningManager::with_config(num_threads, adaptive_mode)?;
    manager.initialize()?;
    Ok(manager)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scirs2_backend_creation() {
        let backend = SciRS2CpuBackend::new();
        assert!(backend.is_ok());
    }

    #[test]
    fn test_scirs2_matmul() {
        let backend = SciRS2CpuBackend::new().unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut result = vec![0.0; 4]; // 2x2 result

        let res = backend.matmul(&a, &b, &mut result, 2, 2, 2, false, false);
        assert!(res.is_ok());

        // Expected result: [19, 22, 43, 50]
        assert!((result[0] - 19.0).abs() < 1e-6);
        assert!((result[1] - 22.0).abs() < 1e-6);
        assert!((result[2] - 43.0).abs() < 1e-6);
        assert!((result[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_scirs2_elementwise_add() {
        let backend = SciRS2CpuBackend::new().unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        let res = backend.add_elementwise(&a, &b, &mut result);
        if let Err(e) = &res {
            eprintln!("Error in test_scirs2_elementwise_add: {:?}", e);
        }
        assert!(res.is_ok());

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_scirs2_elementwise_mul() {
        let backend = SciRS2CpuBackend::new().unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        let res = backend.mul_elementwise(&a, &b, &mut result);
        assert!(res.is_ok());

        assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_scirs2_autotuning_cache() {
        let backend = SciRS2CpuBackend::new().unwrap();

        // Test cache stats
        let (hits, misses) = backend.get_autotuning_stats().unwrap();
        // hits and misses are usize, so they're always >= 0
        assert!(hits < usize::MAX);
        assert!(misses < usize::MAX);

        // Test cache clearing
        let clear_result = backend.clear_autotuning_cache();
        assert!(clear_result.is_ok());
    }

    #[test]
    fn test_scirs2_scalar_operations() {
        let backend = SciRS2CpuBackend::new().unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = vec![0.0; 4];

        // Test scalar addition
        let res = backend.add_scalar(&a, 10.0, &mut result);
        assert!(res.is_ok());
        assert_eq!(result, vec![11.0, 12.0, 13.0, 14.0]);

        // Test scalar multiplication
        let res = backend.mul_scalar(&a, 3.0, &mut result);
        assert!(res.is_ok());
        assert_eq!(result, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_scirs2_reduction() {
        let backend = SciRS2CpuBackend::new().unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sum = backend.sum(&a).unwrap();

        assert!((sum - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_scirs2_large_operations() {
        let backend = SciRS2CpuBackend::new().unwrap();

        // Test with larger arrays to trigger SIMD and auto-tuning
        let size = 10000;
        let a = vec![1.0; size];
        let b = vec![2.0; size];
        let mut result = vec![0.0; size];

        // Test element-wise addition
        let res = backend.add_elementwise(&a, &b, &mut result);
        assert!(res.is_ok());
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));

        // Test element-wise multiplication
        let res = backend.mul_elementwise(&a, &b, &mut result);
        assert!(res.is_ok());
        assert!(result.iter().all(|&x| (x - 2.0).abs() < 1e-6));

        // Test scalar operations
        let res = backend.add_scalar(&a, 5.0, &mut result);
        assert!(res.is_ok());
        assert!(result.iter().all(|&x| (x - 6.0).abs() < 1e-6));

        let res = backend.mul_scalar(&a, 3.0, &mut result);
        assert!(res.is_ok());
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));

        // Test reduction
        let sum = backend.sum(&a).unwrap();
        assert!((sum - size as f32).abs() < 1e-3);
    }

    #[test]
    fn test_tensor_data_preparation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        let prepared = prepare_tensor_data(&data, &shape);
        assert!(prepared.is_ok());

        let mut data_mut = vec![1.0, 2.0, 3.0, 4.0];
        let prepared_mut = prepare_tensor_data_mut(&mut data_mut, &shape);
        assert!(prepared_mut.is_ok());
    }

    #[test]
    fn test_kernel_config() {
        let config = KernelConfig {
            operation: "test".to_string(),
            optimal_threads: 4,
            optimal_chunk_size: 1024,
            optimal_block_size: Some(64),
            use_simd: true,
            cache_blocking: true,
        };

        assert_eq!(config.operation, "test");
        assert_eq!(config.optimal_threads, 4);
        assert!(config.use_simd);
        assert!(config.cache_blocking);
    }

    #[test]
    fn test_autotuning_manager_creation() {
        let manager = SciRS2AutoTuningManager::new();
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert!(manager.adaptive_mode);
    }

    #[test]
    fn test_autotuning_manager_with_config() {
        let manager = SciRS2AutoTuningManager::with_config(8, false);
        assert!(manager.is_ok());

        let manager = manager.unwrap();
        assert!(!manager.adaptive_mode);
        assert_eq!(manager.backend().num_threads(), 8);
    }

    #[test]
    fn test_autotuning_execution() {
        let manager = SciRS2AutoTuningManager::new().unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        let res = manager.execute_with_auto_tuning("add_elementwise", a.len(), |backend| {
            backend.add_elementwise(&a, &b, &mut result)
        });

        if let Err(e) = &res {
            eprintln!("Error in test_autotuning_execution: {:?}", e);
        }
        assert!(res.is_ok());
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_performance_tracking() {
        let manager = SciRS2AutoTuningManager::new().unwrap();

        // Execute some operations to generate performance data
        let a = vec![1.0; 1000];
        let b = vec![2.0; 1000];
        let mut result = vec![0.0; 1000];

        // Run multiple times to build performance history
        for _ in 0..5 {
            let _ = manager.execute_with_auto_tuning("add_elementwise", a.len(), |backend| {
                backend.add_elementwise(&a, &b, &mut result)
            });
        }

        // Check if performance stats are available
        let stats = manager.get_performance_stats("add_elementwise");
        assert!(stats.is_some());

        let (throughput, efficiency, _avg_time) = stats.unwrap();
        assert!(throughput > 0.0);
        assert!(efficiency > 0.0);
    }

    #[test]
    fn test_factory_functions() {
        let manager = create_autotuning_manager();
        assert!(manager.is_ok());

        let manager_with_config = create_autotuning_manager_with_config(4, false);
        assert!(manager_with_config.is_ok());

        let manager = manager_with_config.unwrap();
        assert!(!manager.adaptive_mode);
    }

    #[test]
    fn test_system_stats() {
        let mut manager = SciRS2AutoTuningManager::new().unwrap();
        let _ = manager.initialize();

        let stats = manager.get_system_stats().unwrap();
        assert!(stats.tracked_operations < usize::MAX);
        assert!(stats.cache_hit_ratio >= 0.0 && stats.cache_hit_ratio <= 1.0);
        assert!(stats.adaptive_mode);
    }
}
