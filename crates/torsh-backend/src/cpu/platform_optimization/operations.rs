//! Platform-optimized operations with SIMD implementations
//!
//! This module provides optimized implementations of common operations
//! using CPU-specific SIMD instructions for maximum performance.

use super::detection::CpuInfo;
use crate::error::{BackendError, BackendResult};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Platform-specific optimized operations
#[derive(Debug)]
pub struct PlatformOptimizedOps {
    cpu_info: &'static CpuInfo,
}

impl PlatformOptimizedOps {
    /// Create new platform-optimized operations
    pub fn new() -> Self {
        Self {
            cpu_info: CpuInfo::get(),
        }
    }

    /// Get the detected CPU information
    pub fn cpu_info(&self) -> &CpuInfo {
        self.cpu_info
    }

    /// Optimized vector dot product
    pub fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> BackendResult<f32> {
        if a.len() != b.len() {
            return Err(BackendError::InvalidArgument(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_info.features.avx2 {
                return Ok(self.dot_product_avx2(a, b));
            } else if self.cpu_info.features.sse {
                return Ok(self.dot_product_sse(a, b));
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.cpu_info.features.neon {
                return Ok(self.dot_product_neon(a, b));
            }
        }

        // Fallback to scalar implementation
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    /// AVX2 optimized dot product
    #[cfg(target_arch = "x86_64")]
    fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let mut sum = _mm256_setzero_ps();
            let len = a.len();
            let simd_len = len & !7; // Round down to multiple of 8

            // Process 8 elements at a time with AVX2
            for i in (0..simd_len).step_by(8) {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));

                if self.cpu_info.features.fma {
                    sum = _mm256_fmadd_ps(va, vb, sum);
                } else {
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
                }
            }

            // Horizontal sum of vector
            let sum_low = _mm256_castps256_ps128(sum);
            let sum_high = _mm256_extractf128_ps(sum, 1);
            let sum128 = _mm_add_ps(sum_low, sum_high);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remaining elements
            for i in simd_len..len {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// SSE optimized dot product
    #[cfg(target_arch = "x86_64")]
    fn dot_product_sse(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let mut sum = _mm_setzero_ps();
            let len = a.len();
            let simd_len = len & !3; // Round down to multiple of 4

            // Process 4 elements at a time with SSE
            for i in (0..simd_len).step_by(4) {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vb = _mm_loadu_ps(b.as_ptr().add(i));
                sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
            }

            // Horizontal sum
            let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remaining elements
            for i in simd_len..len {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// NEON optimized dot product
    #[cfg(target_arch = "aarch64")]
    fn dot_product_neon(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum = vdupq_n_f32(0.0);
            let len = a.len();
            let simd_len = len & !3; // Round down to multiple of 4

            // Process 4 elements at a time with NEON
            for i in (0..simd_len).step_by(4) {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                sum = vfmaq_f32(sum, va, vb);
            }

            // Horizontal sum
            let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
            let mut result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

            // Handle remaining elements
            for i in simd_len..len {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// Optimized matrix multiplication with microarchitecture-specific blocking
    pub fn matrix_multiply_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(BackendError::InvalidArgument(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        let block_size = self.cpu_info.optimization.matrix_block_size;

        // Use cache-blocked matrix multiplication
        for ii in (0..m).step_by(block_size) {
            for jj in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let i_end = (ii + block_size).min(m);
                    let j_end = (jj + block_size).min(n);
                    let k_end = (kk + block_size).min(k);

                    self.matrix_multiply_block(a, b, c, ii, i_end, jj, j_end, kk, k_end, m, n, k);
                }
            }
        }

        Ok(())
    }

    /// Optimized block matrix multiplication
    fn matrix_multiply_block(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
        k_start: usize,
        k_end: usize,
        _m: usize,
        n: usize,
        k: usize,
    ) {
        let unroll = self.cpu_info.optimization.unroll_factor;

        for i in i_start..i_end {
            for j in (j_start..j_end).step_by(unroll) {
                let j_limit = (j + unroll).min(j_end);

                for kk in k_start..k_end {
                    let a_val = a[i * k + kk];

                    // Unrolled inner loop
                    for jj in j..j_limit {
                        c[i * n + jj] += a_val * b[kk * n + jj];
                    }
                }
            }
        }
    }

    /// Get optimal chunk size for parallel operations
    pub fn get_optimal_parallel_chunk_size(&self, total_elements: usize) -> usize {
        let base_chunk = self.cpu_info.optimization.parallel_chunk_size;
        let num_threads = self.cpu_info.logical_cores;

        // Ensure we have enough work per thread
        let min_chunk = total_elements / (num_threads * 4);

        base_chunk.max(min_chunk).min(total_elements)
    }

    /// Check if operation should use software prefetching
    pub fn should_use_prefetch(&self, data_size: usize) -> bool {
        self.cpu_info.optimization.software_prefetch && data_size > self.cpu_info.cache.l3_size
    }

    /// Get optimal memory alignment for current platform
    pub fn get_memory_alignment(&self) -> usize {
        self.cpu_info.optimization.memory_alignment
    }

    /// Print detailed CPU information
    pub fn print_cpu_info(&self) {
        let info = self.cpu_info;

        println!("CPU Information:");
        println!("  Vendor: {}", info.vendor);
        println!("  Model: {}", info.model_name);
        println!(
            "  Cores: {} physical, {} logical",
            info.physical_cores, info.logical_cores
        );
        println!(
            "  Frequency: {:.1} MHz base, {:.1} MHz max",
            info.base_frequency, info.max_frequency
        );

        if let Some(arch) = info.x86_microarch {
            println!("  Microarchitecture: {:?}", arch);
        }
        if let Some(arch) = info.arm_microarch {
            println!("  Microarchitecture: {:?}", arch);
        }

        println!(
            "  Cache: L1D={}KB, L1I={}KB, L2={}KB, L3={}KB",
            info.cache.l1d_size / 1024,
            info.cache.l1i_size / 1024,
            info.cache.l2_size / 1024,
            info.cache.l3_size / 1024
        );

        println!("Features:");
        #[cfg(target_arch = "x86_64")]
        {
            if info.features.sse {
                print!(" SSE");
            }
            if info.features.sse2 {
                print!(" SSE2");
            }
            if info.features.sse3 {
                print!(" SSE3");
            }
            if info.features.ssse3 {
                print!(" SSSE3");
            }
            if info.features.sse4_1 {
                print!(" SSE4.1");
            }
            if info.features.sse4_2 {
                print!(" SSE4.2");
            }
            if info.features.avx {
                print!(" AVX");
            }
            if info.features.avx2 {
                print!(" AVX2");
            }
            if info.features.avx512f {
                print!(" AVX-512F");
            }
            if info.features.fma {
                print!(" FMA");
            }
            if info.features.bmi1 {
                print!(" BMI1");
            }
            if info.features.bmi2 {
                print!(" BMI2");
            }
            if info.features.aes {
                print!(" AES");
            }
            if info.features.sha {
                print!(" SHA");
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if info.features.neon {
                print!(" NEON");
            }
            if info.features.fp {
                print!(" FP");
            }
            if info.features.asimd {
                print!(" ASIMD");
            }
            if info.features.aes_arm {
                print!(" AES");
            }
            if info.features.sha1 {
                print!(" SHA1");
            }
            if info.features.sha256 {
                print!(" SHA256");
            }
            if info.features.crc32 {
                print!(" CRC32");
            }
            if info.features.sve {
                print!(" SVE");
            }
        }
        println!();

        println!("Optimizations:");
        println!(
            "  Vector width: {} bytes",
            info.optimization.optimal_vector_width
        );
        println!("  Unroll factor: {}", info.optimization.unroll_factor);
        println!(
            "  Matrix block size: {}",
            info.optimization.matrix_block_size
        );
        println!(
            "  Memory alignment: {} bytes",
            info.optimization.memory_alignment
        );
        println!(
            "  Parallel chunk size: {}",
            info.optimization.parallel_chunk_size
        );
        println!(
            "  Software prefetch: {}",
            info.optimization.software_prefetch
        );
        println!("  HT aware: {}", info.optimization.ht_aware);
        println!("  NUMA aware: {}", info.optimization.numa_aware);
    }
}

impl Default for PlatformOptimizedOps {
    fn default() -> Self {
        Self::new()
    }
}
