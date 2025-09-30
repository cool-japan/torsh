//! SIMD Operations for CPU Backend - Clean Modular Interface
//!
//! This module provides a unified interface to SIMD-accelerated implementations of common tensor operations
//! when the "simd" feature is enabled. Supports x86 AVX/AVX2/AVX-512 and ARM NEON.
//!
//! # Architecture
//!
//! The SIMD system is organized into specialized modules:
//!
//! - **core**: Feature detection, infrastructure, and SIMD capabilities
//! - **arithmetic**: Basic arithmetic operations (add, sub, mul, div, dot, sum, pow, sqrt)
//! - **activation**: Neural network activation functions (relu, sigmoid, tanh, gelu)
//! - **math**: Mathematical transcendental functions (exp, log, approximations)
//! - **reduction**: Min/max operations and reduction functions
//! - **specialized**: Matrix operations, complex numbers, quantized operations, adaptive SIMD
//!
//! All SIMD operations are unified through re-exports for full backward compatibility.

// Modular SIMD system
pub mod activation;
pub mod arithmetic;
pub mod core;
pub mod math;
pub mod reduction;
pub mod specialized;

// Re-export the complete modular interface
pub use core::{
    alignment, f32_vector_width, f64_vector_width, has_avx2, has_avx512, has_neon,
    i32_vector_width, optimal_simd_chunk_size, should_use_simd, vector_width,
};

// Re-export arithmetic operations
pub use arithmetic::{
    simd_add_f32, simd_add_scalar_f32, simd_div_f32, simd_dot_f32, simd_mul_f32,
    simd_mul_scalar_f32, simd_pow_f32, simd_sqrt_f32, simd_sub_f32, simd_sum_f32,
};

// Re-export activation functions
pub use activation::{simd_gelu_f32, simd_relu_f32, simd_sigmoid_f32, simd_tanh_f32};

// Re-export mathematical functions
pub use math::{simd_exp_f32, simd_log_f32};

// Re-export SIMD-specific functions only when feature is enabled
#[cfg(feature = "simd")]
pub use math::{simd_fast_exp_f32x8, simd_fast_log_f32x8};

// Re-export reduction operations
pub use reduction::{simd_max_f32, simd_min_f32, simd_reduce_max_f32, simd_reduce_min_f32};

// Re-export specialized operations
pub use specialized::{
    adaptive, simd_add_i8, simd_add_u8, simd_complex_add_f32, simd_complex_mul_f32,
    simd_matmul_f32, simd_quantized_mul_u8,
};

// Re-export adaptive SIMD functions for convenience
pub use specialized::adaptive::{
    adaptive_simd_add_f32, adaptive_simd_dot_f32, adaptive_simd_matmul_f32, adaptive_simd_mul_f32,
    adaptive_simd_relu_f32, adaptive_simd_sigmoid_f32, adaptive_simd_sum_f32,
};

// Re-export SIMD types from core for convenience
#[cfg(feature = "simd")]
pub use core::{f32x8, f64x4, i16x16, i32x8, i8x32, u16x16, u32x8, u8x32};

/// SIMD trait for common operations on arrays
pub trait SimdArray<T> {
    /// Element-wise addition with another array
    fn simd_add(&self, other: &[T], result: &mut [T]);

    /// Element-wise multiplication with another array
    fn simd_mul(&self, other: &[T], result: &mut [T]);

    /// Dot product with another array
    fn simd_dot(&self, other: &[T]) -> T;

    /// Sum reduction of the array
    fn simd_sum(&self) -> T;

    /// Element-wise addition with a scalar
    fn simd_add_scalar(&self, scalar: T, result: &mut [T]);

    /// Element-wise multiplication with a scalar
    fn simd_mul_scalar(&self, scalar: T, result: &mut [T]);

    /// Apply ReLU activation
    fn simd_relu(&self, result: &mut [T]);

    /// Apply Sigmoid activation
    fn simd_sigmoid(&self, result: &mut [T]);

    /// Apply Tanh activation
    fn simd_tanh(&self, result: &mut [T]);

    /// Apply GELU activation
    fn simd_gelu(&self, result: &mut [T]);

    /// Find minimum value
    fn simd_reduce_min(&self) -> T;

    /// Find maximum value
    fn simd_reduce_max(&self) -> T;
}

impl SimdArray<f32> for [f32] {
    fn simd_add(&self, other: &[f32], result: &mut [f32]) {
        simd_add_f32(self, other, result);
    }

    fn simd_mul(&self, other: &[f32], result: &mut [f32]) {
        simd_mul_f32(self, other, result);
    }

    fn simd_dot(&self, other: &[f32]) -> f32 {
        simd_dot_f32(self, other)
    }

    fn simd_sum(&self) -> f32 {
        simd_sum_f32(self)
    }

    fn simd_add_scalar(&self, scalar: f32, result: &mut [f32]) {
        simd_add_scalar_f32(self, scalar, result);
    }

    fn simd_mul_scalar(&self, scalar: f32, result: &mut [f32]) {
        simd_mul_scalar_f32(self, scalar, result);
    }

    fn simd_relu(&self, result: &mut [f32]) {
        simd_relu_f32(self, result);
    }

    fn simd_sigmoid(&self, result: &mut [f32]) {
        simd_sigmoid_f32(self, result);
    }

    fn simd_tanh(&self, result: &mut [f32]) {
        simd_tanh_f32(self, result);
    }

    fn simd_gelu(&self, result: &mut [f32]) {
        simd_gelu_f32(self, result);
    }

    fn simd_reduce_min(&self) -> f32 {
        simd_reduce_min_f32(self)
    }

    fn simd_reduce_max(&self) -> f32 {
        simd_reduce_max_f32(self)
    }
}

/// Performance benchmarking utilities for SIMD operations
pub mod benchmarks {
    use super::*;

    /// Benchmark SIMD vs scalar addition performance
    pub fn benchmark_add_performance(size: usize) -> (f64, f64) {
        use std::time::Instant;

        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut result = vec![0.0; size];

        // Benchmark SIMD version
        let start = Instant::now();
        for _ in 0..100 {
            simd_add_f32(&a, &b, &mut result);
        }
        let simd_time = start.elapsed().as_secs_f64();

        // Benchmark scalar version
        let start = Instant::now();
        for _ in 0..100 {
            for i in 0..size {
                result[i] = a[i] + b[i];
            }
        }
        let scalar_time = start.elapsed().as_secs_f64();

        (simd_time, scalar_time)
    }

    /// Get SIMD speedup factor for various operations
    pub fn get_simd_speedup_factor(size: usize) -> f64 {
        let (simd_time, scalar_time) = benchmark_add_performance(size);
        scalar_time / simd_time
    }
}

/// SIMD operation selection utilities
pub mod selection {
    use super::*;

    /// Automatically select the best SIMD implementation based on array size and CPU features
    pub fn auto_select_add(a: &[f32], b: &[f32], result: &mut [f32]) {
        if should_use_simd(a.len()) {
            adaptive_simd_add_f32(a, b, result);
        } else {
            let len = a.len().min(b.len()).min(result.len());
            for i in 0..len {
                result[i] = a[i] + b[i];
            }
        }
    }

    /// Automatically select the best SIMD implementation for dot product
    pub fn auto_select_dot(a: &[f32], b: &[f32]) -> f32 {
        if should_use_simd(a.len()) {
            adaptive_simd_dot_f32(a, b)
        } else {
            let len = a.len().min(b.len());
            let mut sum = 0.0;
            for i in 0..len {
                sum += a[i] * b[i];
            }
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_simd_system() {
        // Test that all modules are accessible and working

        // Test feature detection
        let _has_avx2 = has_avx2();
        let _has_avx512 = has_avx512();
        let _has_neon = has_neon();

        // Test arithmetic operations
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 1.0, 1.0, 1.0];
        let mut result = [0.0; 4];

        simd_add_f32(&a, &b, &mut result);
        assert_eq!(result, [2.0, 3.0, 4.0, 5.0]);

        // Test activation functions
        let input = [1.0, -1.0, 0.0, 2.0];
        let mut output = [0.0; 4];

        simd_relu_f32(&input, &mut output);
        assert_eq!(output, [1.0, 0.0, 0.0, 2.0]);

        // Test reduction operations
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let min_val = simd_reduce_min_f32(&input);
        let max_val = simd_reduce_max_f32(&input);

        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);
    }

    #[test]
    fn test_simd_trait() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 2.0, 2.0, 2.0];
        let mut result = [0.0f32; 4];

        // Test trait methods
        a.simd_add(&b, &mut result);
        assert_eq!(result, [3.0, 4.0, 5.0, 6.0]);

        let dot_product = a.simd_dot(&b);
        assert_eq!(dot_product, 20.0); // 1*2 + 2*2 + 3*2 + 4*2

        let sum = a.simd_sum();
        assert_eq!(sum, 10.0); // 1 + 2 + 3 + 4
    }

    #[test]
    fn test_backward_compatibility() {
        // Ensure that the modular system maintains full backward compatibility
        let input = [-1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0; 4];

        // These functions should work exactly as before
        simd_relu_f32(&input, &mut output);
        simd_sigmoid_f32(&input, &mut output);
        simd_tanh_f32(&input, &mut output);

        // Arithmetic operations should work as before
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let mut result = [0.0; 4];

        simd_add_f32(&a, &b, &mut result);
        simd_mul_f32(&a, &b, &mut result);
        simd_sub_f32(&a, &b, &mut result);
        simd_div_f32(&a, &b, &mut result);

        let _dot = simd_dot_f32(&a, &b);
        let _sum = simd_sum_f32(&a);
    }

    #[test]
    fn test_selection_utilities() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        let mut result = [0.0; 4];

        // Test auto-selection functions
        selection::auto_select_add(&a, &b, &mut result);
        assert_eq!(result, [5.0, 5.0, 5.0, 5.0]);

        let dot = selection::auto_select_dot(&a, &b);
        assert_eq!(dot, 20.0); // 1*4 + 2*3 + 3*2 + 4*1
    }

    #[test]
    fn test_vector_widths() {
        // Test that vector width functions return reasonable values
        let f32_width = f32_vector_width();
        let f64_width = f64_vector_width();
        let i32_width = i32_vector_width();

        assert!(f32_width >= 1);
        assert!(f64_width >= 1);
        assert!(i32_width >= 1);

        // f32 should have at least as many elements as f64 (same or double)
        assert!(f32_width >= f64_width);
    }
}
