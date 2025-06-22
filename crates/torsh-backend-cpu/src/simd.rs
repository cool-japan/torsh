//! SIMD Operations for CPU Backend
//!
//! This module provides SIMD-accelerated implementations of common tensor operations
//! when the "simd" feature is enabled.

#[cfg(feature = "simd")]
use wide::*;

/// SIMD-accelerated element-wise addition for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());
    let simd_len = len / 8; // f32x8 vectors
    let remainder_start = simd_len * 8;

    // Process 8 elements at a time using SIMD
    for i in 0..simd_len {
        let idx = i * 8;
        let a_simd = f32x8::from([
            a[idx],
            a[idx + 1],
            a[idx + 2],
            a[idx + 3],
            a[idx + 4],
            a[idx + 5],
            a[idx + 6],
            a[idx + 7],
        ]);
        let b_simd = f32x8::from([
            b[idx],
            b[idx + 1],
            b[idx + 2],
            b[idx + 3],
            b[idx + 4],
            b[idx + 5],
            b[idx + 6],
            b[idx + 7],
        ]);
        let result_simd = a_simd + b_simd;
        let result_array: [f32; 8] = result_simd.into();

        result[idx..idx + 8].copy_from_slice(&result_array);
    }

    // Handle remaining elements
    for i in remainder_start..len {
        result[i] = a[i] + b[i];
    }
}

/// SIMD-accelerated element-wise multiplication for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    for i in 0..simd_len {
        let idx = i * 8;
        let a_simd = f32x8::from([
            a[idx],
            a[idx + 1],
            a[idx + 2],
            a[idx + 3],
            a[idx + 4],
            a[idx + 5],
            a[idx + 6],
            a[idx + 7],
        ]);
        let b_simd = f32x8::from([
            b[idx],
            b[idx + 1],
            b[idx + 2],
            b[idx + 3],
            b[idx + 4],
            b[idx + 5],
            b[idx + 6],
            b[idx + 7],
        ]);
        let result_simd = a_simd * b_simd;
        let result_array: [f32; 8] = result_simd.into();

        result[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        result[i] = a[i] * b[i];
    }
}

/// SIMD-accelerated ReLU activation for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_relu_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let zero = f32x8::splat(0.0);

    for i in 0..simd_len {
        let idx = i * 8;
        let input_simd = f32x8::from([
            input[idx],
            input[idx + 1],
            input[idx + 2],
            input[idx + 3],
            input[idx + 4],
            input[idx + 5],
            input[idx + 6],
            input[idx + 7],
        ]);
        let result_simd = input_simd.max(zero);
        let result_array: [f32; 8] = result_simd.into();

        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = input[i].max(0.0);
    }
}

/// SIMD-accelerated sum reduction for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_sum_f32(input: &[f32]) -> f32 {
    let len = input.len();
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let mut sum_simd = f32x8::splat(0.0);

    // Process 8 elements at a time
    for i in 0..simd_len {
        let idx = i * 8;
        let input_simd = f32x8::from([
            input[idx],
            input[idx + 1],
            input[idx + 2],
            input[idx + 3],
            input[idx + 4],
            input[idx + 5],
            input[idx + 6],
            input[idx + 7],
        ]);
        sum_simd = sum_simd + input_simd;
    }

    // Sum the SIMD vector
    let sum_array: [f32; 8] = sum_simd.into();
    let mut total = sum_array.iter().sum::<f32>();

    // Handle remaining elements
    for i in remainder_start..len {
        total += input[i];
    }

    total
}

/// SIMD-accelerated dot product for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let mut dot_simd = f32x8::splat(0.0);

    for i in 0..simd_len {
        let idx = i * 8;
        let a_simd = f32x8::from([
            a[idx],
            a[idx + 1],
            a[idx + 2],
            a[idx + 3],
            a[idx + 4],
            a[idx + 5],
            a[idx + 6],
            a[idx + 7],
        ]);
        let b_simd = f32x8::from([
            b[idx],
            b[idx + 1],
            b[idx + 2],
            b[idx + 3],
            b[idx + 4],
            b[idx + 5],
            b[idx + 6],
            b[idx + 7],
        ]);
        dot_simd = dot_simd + (a_simd * b_simd);
    }

    let dot_array: [f32; 8] = dot_simd.into();
    let mut total = dot_array.iter().sum::<f32>();

    for i in remainder_start..len {
        total += a[i] * b[i];
    }

    total
}

// Fallback implementations when SIMD is not available
#[cfg(not(feature = "simd"))]
pub fn simd_add_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        result[i] = a[i] + b[i];
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_mul_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        result[i] = a[i] * b[i];
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_relu_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].max(0.0);
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_sum_f32(input: &[f32]) -> f32 {
    input.iter().sum()
}

#[cfg(not(feature = "simd"))]
pub fn simd_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Trait for SIMD-accelerated operations
pub trait SimdOps<T> {
    fn simd_add(&self, other: &[T], result: &mut [T]);
    fn simd_mul(&self, other: &[T], result: &mut [T]);
    fn simd_relu(&self, result: &mut [T]);
    fn simd_sum(&self) -> T;
    fn simd_dot(&self, other: &[T]) -> T;
}

impl SimdOps<f32> for [f32] {
    fn simd_add(&self, other: &[f32], result: &mut [f32]) {
        simd_add_f32(self, other, result);
    }

    fn simd_mul(&self, other: &[f32], result: &mut [f32]) {
        simd_mul_f32(self, other, result);
    }

    fn simd_relu(&self, result: &mut [f32]) {
        simd_relu_f32(self, result);
    }

    fn simd_sum(&self) -> f32 {
        simd_sum_f32(self)
    }

    fn simd_dot(&self, other: &[f32]) -> f32 {
        simd_dot_f32(self, other)
    }
}

/// Determine if SIMD is available and beneficial for the given size
pub fn should_use_simd(size: usize) -> bool {
    #[cfg(feature = "simd")]
    {
        size >= 64 // Only use SIMD for reasonably large arrays
    }
    #[cfg(not(feature = "simd"))]
    {
        let _ = size; // Silence unused variable warning
        false
    }
}

/// Get the optimal chunk size for SIMD operations
pub fn optimal_simd_chunk_size<T>() -> usize {
    #[cfg(feature = "simd")]
    {
        match std::mem::size_of::<T>() {
            4 => 8, // f32: 8 elements per SIMD register
            8 => 4, // f64: 4 elements per SIMD register
            _ => 1,
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_add_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut result = vec![0.0; 9];
        let expected = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0];

        simd_add_f32(&a, &b, &mut result);

        for i in 0..9 {
            assert_relative_eq!(result[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_mul_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 8];
        let expected = vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0];

        simd_mul_f32(&a, &b, &mut result);

        for i in 0..8 {
            assert_relative_eq!(result[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_relu_f32() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 4.0, -5.0];
        let mut output = vec![0.0; 8];
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 4.0, 0.0];

        simd_relu_f32(&input, &mut output);

        for i in 0..8 {
            assert_relative_eq!(output[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_sum_f32() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let expected = 45.0;

        let result = simd_sum_f32(&input);
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_dot_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0; // 70.0

        let result = simd_dot_f32(&a, &b);
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_ops_trait() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        a.simd_add(&b, &mut result);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

        let sum = a.simd_sum();
        assert_relative_eq!(sum, 10.0, epsilon = 1e-6);

        let dot = a.simd_dot(&b);
        assert_relative_eq!(dot, 70.0, epsilon = 1e-6);
    }

    #[test]
    fn test_should_use_simd() {
        assert!(!should_use_simd(10));
        assert!(should_use_simd(100) || !cfg!(feature = "simd"));
    }

    #[test]
    fn test_optimal_simd_chunk_size() {
        assert_eq!(
            optimal_simd_chunk_size::<f32>(),
            if cfg!(feature = "simd") { 8 } else { 1 }
        );
        assert_eq!(
            optimal_simd_chunk_size::<f64>(),
            if cfg!(feature = "simd") { 4 } else { 1 }
        );
    }
}
