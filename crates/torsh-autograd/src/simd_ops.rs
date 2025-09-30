//! SIMD optimized gradient operations for high-performance automatic differentiation
//!
//! This module provides vectorized implementations of common gradient operations
//! using SIMD instructions to achieve maximum performance on modern CPUs.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};

/// SIMD optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD optimization
    None,
    /// SSE2 optimizations (x86_64)
    Sse2,
    /// AVX optimizations (x86_64)
    Avx,
    /// AVX2 optimizations (x86_64)
    Avx2,
    /// AVX512 optimizations (x86_64)
    Avx512,
    /// NEON optimizations (ARM)
    Neon,
}

/// Detect available SIMD features on the current CPU
pub fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            SimdLevel::Avx512
        } else if is_x86_feature_detected!("avx2") {
            SimdLevel::Avx2
        } else if is_x86_feature_detected!("avx") {
            SimdLevel::Avx
        } else if is_x86_feature_detected!("sse2") {
            SimdLevel::Sse2
        } else {
            SimdLevel::None
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if cfg!(target_feature = "neon") {
            SimdLevel::Neon
        } else {
            SimdLevel::None
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        SimdLevel::None
    }
}

/// SIMD-optimized gradient accumulation
pub trait SimdGradAccumulator<T: FloatElement> {
    /// Accumulate gradients with SIMD optimization
    fn accumulate_simd(&mut self, gradients: &[T], weights: Option<&[T]>) -> Result<()>;

    /// Get accumulated gradient values
    fn get_accumulated(&self) -> &[T];

    /// Reset accumulator
    fn reset(&mut self);

    /// Get the SIMD level being used
    fn simd_level(&self) -> SimdLevel;
}

/// F32 SIMD gradient accumulator
pub struct F32SimdAccumulator {
    accumulated: Vec<f32>,
    simd_level: SimdLevel,
}

impl F32SimdAccumulator {
    /// Create a new F32 SIMD accumulator
    pub fn new(size: usize) -> Self {
        Self {
            accumulated: vec![0.0; size],
            simd_level: detect_simd_level(),
        }
    }

    /// Create with specific SIMD level
    pub fn with_simd_level(size: usize, simd_level: SimdLevel) -> Self {
        Self {
            accumulated: vec![0.0; size],
            simd_level,
        }
    }
}

impl SimdGradAccumulator<f32> for F32SimdAccumulator {
    fn accumulate_simd(&mut self, gradients: &[f32], weights: Option<&[f32]>) -> Result<()> {
        if gradients.len() != self.accumulated.len() {
            return Err(TorshError::AutogradError(format!(
                "Gradient size mismatch: expected {}, got {}",
                self.accumulated.len(),
                gradients.len()
            )));
        }

        match self.simd_level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => unsafe { self.accumulate_avx2(gradients, weights) },
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx => unsafe { self.accumulate_avx(gradients, weights) },
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Sse2 => unsafe { self.accumulate_sse2(gradients, weights) },
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => unsafe { self.accumulate_neon(gradients, weights) },
            _ => self.accumulate_scalar(gradients, weights),
        }
    }

    fn get_accumulated(&self) -> &[f32] {
        &self.accumulated
    }

    fn reset(&mut self) {
        self.accumulated.fill(0.0);
    }

    fn simd_level(&self) -> SimdLevel {
        self.simd_level
    }
}

impl F32SimdAccumulator {
    /// Scalar fallback implementation
    fn accumulate_scalar(&mut self, gradients: &[f32], weights: Option<&[f32]>) -> Result<()> {
        match weights {
            Some(w) => {
                if w.len() != gradients.len() {
                    return Err(TorshError::AutogradError(
                        "Weight size mismatch".to_string(),
                    ));
                }
                for ((acc, &grad), &weight) in self.accumulated.iter_mut().zip(gradients).zip(w) {
                    *acc += grad * weight;
                }
            }
            None => {
                for (acc, &grad) in self.accumulated.iter_mut().zip(gradients) {
                    *acc += grad;
                }
            }
        }
        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn accumulate_sse2(&mut self, gradients: &[f32], weights: Option<&[f32]>) -> Result<()> {
        let chunks = self.accumulated.len() / 4;
        let remainder = self.accumulated.len() % 4;

        match weights {
            Some(w) => {
                if w.len() != gradients.len() {
                    return Err(TorshError::AutogradError(
                        "Weight size mismatch".to_string(),
                    ));
                }

                for i in 0..chunks {
                    let idx = i * 4;
                    let acc_vec = _mm_loadu_ps(self.accumulated.as_ptr().add(idx));
                    let grad_vec = _mm_loadu_ps(gradients.as_ptr().add(idx));
                    let weight_vec = _mm_loadu_ps(w.as_ptr().add(idx));

                    let weighted_grad = _mm_mul_ps(grad_vec, weight_vec);
                    let result = _mm_add_ps(acc_vec, weighted_grad);

                    _mm_storeu_ps(self.accumulated.as_mut_ptr().add(idx), result);
                }

                // Handle remainder
                #[allow(clippy::needless_range_loop)]
                for i in (chunks * 4)..(chunks * 4 + remainder) {
                    self.accumulated[i] += gradients[i] * w[i];
                }
            }
            None => {
                for i in 0..chunks {
                    let idx = i * 4;
                    let acc_vec = _mm_loadu_ps(self.accumulated.as_ptr().add(idx));
                    let grad_vec = _mm_loadu_ps(gradients.as_ptr().add(idx));
                    let result = _mm_add_ps(acc_vec, grad_vec);

                    _mm_storeu_ps(self.accumulated.as_mut_ptr().add(idx), result);
                }

                // Handle remainder
                #[allow(clippy::needless_range_loop)]
                for i in (chunks * 4)..(chunks * 4 + remainder) {
                    self.accumulated[i] += gradients[i];
                }
            }
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn accumulate_avx(&mut self, gradients: &[f32], weights: Option<&[f32]>) -> Result<()> {
        let chunks = self.accumulated.len() / 8;
        let remainder = self.accumulated.len() % 8;

        match weights {
            Some(w) => {
                if w.len() != gradients.len() {
                    return Err(TorshError::AutogradError(
                        "Weight size mismatch".to_string(),
                    ));
                }

                for i in 0..chunks {
                    let idx = i * 8;
                    let acc_vec = _mm256_loadu_ps(self.accumulated.as_ptr().add(idx));
                    let grad_vec = _mm256_loadu_ps(gradients.as_ptr().add(idx));
                    let weight_vec = _mm256_loadu_ps(w.as_ptr().add(idx));

                    let weighted_grad = _mm256_mul_ps(grad_vec, weight_vec);
                    let result = _mm256_add_ps(acc_vec, weighted_grad);

                    _mm256_storeu_ps(self.accumulated.as_mut_ptr().add(idx), result);
                }

                // Handle remainder with scalar operations
                #[allow(clippy::needless_range_loop)]
                for i in (chunks * 8)..(chunks * 8 + remainder) {
                    self.accumulated[i] += gradients[i] * w[i];
                }
            }
            None => {
                for i in 0..chunks {
                    let idx = i * 8;
                    let acc_vec = _mm256_loadu_ps(self.accumulated.as_ptr().add(idx));
                    let grad_vec = _mm256_loadu_ps(gradients.as_ptr().add(idx));
                    let result = _mm256_add_ps(acc_vec, grad_vec);

                    _mm256_storeu_ps(self.accumulated.as_mut_ptr().add(idx), result);
                }

                // Handle remainder with scalar operations
                #[allow(clippy::needless_range_loop)]
                for i in (chunks * 8)..(chunks * 8 + remainder) {
                    self.accumulated[i] += gradients[i];
                }
            }
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn accumulate_avx2(&mut self, gradients: &[f32], weights: Option<&[f32]>) -> Result<()> {
        // AVX2 has same vector width as AVX for f32, but may have better instruction scheduling
        self.accumulate_avx(gradients, weights)
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn accumulate_neon(&mut self, gradients: &[f32], weights: Option<&[f32]>) -> Result<()> {
        let chunks = self.accumulated.len() / 4;
        let remainder = self.accumulated.len() % 4;

        match weights {
            Some(w) => {
                if w.len() != gradients.len() {
                    return Err(TorshError::AutogradError(
                        "Weight size mismatch".to_string(),
                    ));
                }

                for i in 0..chunks {
                    let idx = i * 4;
                    let acc_vec = vld1q_f32(self.accumulated.as_ptr().add(idx));
                    let grad_vec = vld1q_f32(gradients.as_ptr().add(idx));
                    let weight_vec = vld1q_f32(w.as_ptr().add(idx));

                    let weighted_grad = vmulq_f32(grad_vec, weight_vec);
                    let result = vaddq_f32(acc_vec, weighted_grad);

                    vst1q_f32(self.accumulated.as_mut_ptr().add(idx), result);
                }

                // Handle remainder
                #[allow(clippy::needless_range_loop)]
                for i in (chunks * 4)..(chunks * 4 + remainder) {
                    self.accumulated[i] += gradients[i] * w[i];
                }
            }
            None => {
                for i in 0..chunks {
                    let idx = i * 4;
                    let acc_vec = vld1q_f32(self.accumulated.as_ptr().add(idx));
                    let grad_vec = vld1q_f32(gradients.as_ptr().add(idx));
                    let result = vaddq_f32(acc_vec, grad_vec);

                    vst1q_f32(self.accumulated.as_mut_ptr().add(idx), result);
                }

                // Handle remainder
                #[allow(clippy::needless_range_loop)]
                for i in (chunks * 4)..(chunks * 4 + remainder) {
                    self.accumulated[i] += gradients[i];
                }
            }
        }

        Ok(())
    }
}

/// SIMD-optimized gradient operations
pub mod ops {
    use super::*;

    /// Element-wise gradient multiplication with SIMD optimization
    pub fn grad_mul_simd<T: FloatElement>(a: &[T], b: &[T], output: &mut [T]) -> Result<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(TorshError::AutogradError("Array size mismatch".to_string()));
        }

        // Dispatch to type-specific implementations
        match std::any::TypeId::of::<T>() {
            id if id == std::any::TypeId::of::<f32>() => {
                let a_f32 =
                    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f32, a.len()) };
                let b_f32 =
                    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len()) };
                let output_f32 = unsafe {
                    std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len())
                };
                grad_mul_f32_simd(a_f32, b_f32, output_f32)
            }
            id if id == std::any::TypeId::of::<f64>() => {
                let a_f64 =
                    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const f64, a.len()) };
                let b_f64 =
                    unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f64, b.len()) };
                let output_f64 = unsafe {
                    std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, output.len())
                };
                grad_mul_f64_simd(a_f64, b_f64, output_f64)
            }
            _ => {
                // Fallback to scalar operations
                for ((a_val, b_val), out) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
                    *out = *a_val * *b_val;
                }
                Ok(())
            }
        }
    }

    /// F32 gradient multiplication with automatic SIMD dispatch
    pub fn grad_mul_f32_simd(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        let simd_level = detect_simd_level();

        match simd_level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx => unsafe { grad_mul_f32_avx(a, b, output) },
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Sse2 => unsafe { grad_mul_f32_sse2(a, b, output) },
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => unsafe { grad_mul_f32_neon(a, b, output) },
            _ => grad_mul_f32_scalar(a, b, output),
        }
    }

    /// F64 gradient multiplication with automatic SIMD dispatch
    pub fn grad_mul_f64_simd(a: &[f64], b: &[f64], output: &mut [f64]) -> Result<()> {
        let simd_level = detect_simd_level();

        match simd_level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 | SimdLevel::Avx => unsafe { grad_mul_f64_avx(a, b, output) },
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Sse2 => unsafe { grad_mul_f64_sse2(a, b, output) },
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => unsafe { grad_mul_f64_neon(a, b, output) },
            _ => grad_mul_f64_scalar(a, b, output),
        }
    }

    // Scalar implementations
    fn grad_mul_f32_scalar(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        for ((a_val, b_val), out) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *out = a_val * b_val;
        }
        Ok(())
    }

    fn grad_mul_f64_scalar(a: &[f64], b: &[f64], output: &mut [f64]) -> Result<()> {
        for ((a_val, b_val), out) in a.iter().zip(b.iter()).zip(output.iter_mut()) {
            *out = a_val * b_val;
        }
        Ok(())
    }

    // SIMD implementations for x86_64
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn grad_mul_f32_sse2(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = _mm_loadu_ps(a.as_ptr().add(idx));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(idx));
            let result = _mm_mul_ps(a_vec, b_vec);
            _mm_storeu_ps(output.as_mut_ptr().add(idx), result);
        }

        // Handle remainder
        #[allow(clippy::needless_range_loop)]
        for i in (chunks * 4)..(chunks * 4 + remainder) {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn grad_mul_f32_avx(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        let chunks = a.len() / 8;
        let remainder = a.len() % 8;

        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(idx));
            let result = _mm256_mul_ps(a_vec, b_vec);
            _mm256_storeu_ps(output.as_mut_ptr().add(idx), result);
        }

        // Handle remainder
        #[allow(clippy::needless_range_loop)]
        for i in (chunks * 8)..(chunks * 8 + remainder) {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn grad_mul_f64_sse2(a: &[f64], b: &[f64], output: &mut [f64]) -> Result<()> {
        let chunks = a.len() / 2;
        let remainder = a.len() % 2;

        for i in 0..chunks {
            let idx = i * 2;
            let a_vec = _mm_loadu_pd(a.as_ptr().add(idx));
            let b_vec = _mm_loadu_pd(b.as_ptr().add(idx));
            let result = _mm_mul_pd(a_vec, b_vec);
            _mm_storeu_pd(output.as_mut_ptr().add(idx), result);
        }

        // Handle remainder
        for i in (chunks * 2)..(chunks * 2 + remainder) {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn grad_mul_f64_avx(a: &[f64], b: &[f64], output: &mut [f64]) -> Result<()> {
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(idx));
            let result = _mm256_mul_pd(a_vec, b_vec);
            _mm256_storeu_pd(output.as_mut_ptr().add(idx), result);
        }

        // Handle remainder
        #[allow(clippy::needless_range_loop)]
        for i in (chunks * 4)..(chunks * 4 + remainder) {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }

    // SIMD implementations for ARM NEON
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn grad_mul_f32_neon(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<()> {
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = vld1q_f32(a.as_ptr().add(idx));
            let b_vec = vld1q_f32(b.as_ptr().add(idx));
            let result = vmulq_f32(a_vec, b_vec);
            vst1q_f32(output.as_mut_ptr().add(idx), result);
        }

        // Handle remainder
        #[allow(clippy::needless_range_loop)]
        for i in (chunks * 4)..(chunks * 4 + remainder) {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn grad_mul_f64_neon(a: &[f64], b: &[f64], output: &mut [f64]) -> Result<()> {
        let chunks = a.len() / 2;
        let remainder = a.len() % 2;

        for i in 0..chunks {
            let idx = i * 2;
            let a_vec = vld1q_f64(a.as_ptr().add(idx));
            let b_vec = vld1q_f64(b.as_ptr().add(idx));
            let result = vmulq_f64(a_vec, b_vec);
            vst1q_f64(output.as_mut_ptr().add(idx), result);
        }

        // Handle remainder
        for i in (chunks * 2)..(chunks * 2 + remainder) {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }
}

/// Performance benchmarking utilities for SIMD operations
pub mod bench {
    use super::*;
    use std::time::Instant;

    /// Benchmark results for SIMD operations
    #[derive(Debug, Clone)]
    pub struct BenchResult {
        pub simd_level: SimdLevel,
        pub duration_ns: u64,
        pub throughput_gflops: f64,
        pub operations_count: usize,
    }

    /// Benchmark gradient accumulation performance
    pub fn benchmark_grad_accumulation(size: usize, iterations: usize) -> Vec<BenchResult> {
        let mut results = Vec::new();
        let gradients: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();

        // Test different SIMD levels
        let simd_levels = vec![
            SimdLevel::None,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Sse2,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2,
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon,
        ];

        for &simd_level in &simd_levels {
            // Skip unavailable SIMD levels
            if simd_level != SimdLevel::None && simd_level != detect_simd_level() {
                continue;
            }

            let mut accumulator = F32SimdAccumulator::with_simd_level(size, simd_level);

            let start = Instant::now();
            for _ in 0..iterations {
                accumulator.accumulate_simd(&gradients, None).unwrap();
            }
            let duration = start.elapsed();

            let ops_count = size * iterations;
            let duration_ns = duration.as_nanos() as u64;
            let throughput_gflops = (ops_count as f64) / (duration.as_secs_f64() * 1e9);

            results.push(BenchResult {
                simd_level,
                duration_ns,
                throughput_gflops,
                operations_count: ops_count,
            });
        }

        results
    }

    /// Print benchmark results in a formatted table
    pub fn print_benchmark_results(results: &[BenchResult]) {
        println!("SIMD Gradient Accumulation Benchmark Results:");
        println!("┌─────────────┬─────────────┬────────────────┬─────────────────┐");
        println!("│ SIMD Level  │ Duration ns │ Throughput     │ Operations      │");
        println!("├─────────────┼─────────────┼────────────────┼─────────────────┤");

        for result in results {
            println!(
                "│ {:11} │ {:11} │ {:8.2} GFLOPS │ {:15} │",
                format!("{:?}", result.simd_level),
                result.duration_ns,
                result.throughput_gflops,
                result.operations_count
            );
        }

        println!("└─────────────┴─────────────┴────────────────┴─────────────────┘");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_detection() {
        let level = detect_simd_level();
        println!("Detected SIMD level: {:?}", level);
        // Just ensure it doesn't panic
        assert!(true);
    }

    #[test]
    fn test_f32_accumulator() {
        let mut accumulator = F32SimdAccumulator::new(100);
        let gradients: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();

        // Test basic accumulation
        accumulator.accumulate_simd(&gradients, None).unwrap();
        let result = accumulator.get_accumulated();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, i as f32 * 0.01, epsilon = 1e-6);
        }

        // Test weighted accumulation
        let weights: Vec<f32> = vec![2.0; 100];
        accumulator.reset();
        accumulator
            .accumulate_simd(&gradients, Some(&weights))
            .unwrap();
        let result = accumulator.get_accumulated();

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, i as f32 * 0.01 * 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gradient_multiplication() {
        let size = 1000;
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.002).collect();
        let mut output = vec![0.0f32; size];

        ops::grad_mul_simd(&a, &b, &mut output).unwrap();

        // Verify results
        for (i, &val) in output.iter().enumerate() {
            let expected = (i as f32 * 0.001) * ((i as f32 + 1.0) * 0.002);
            assert_relative_eq!(val, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_benchmark() {
        let results = bench::benchmark_grad_accumulation(1000, 100);
        assert!(!results.is_empty());

        // Print results for manual inspection
        bench::print_benchmark_results(&results);
    }
}
