//! SIMD reduction operations
//!
//! This module provides SIMD-accelerated reduction operations
//! including min, max, and their element-wise variants.

#[cfg(feature = "simd")]
use wide::f32x8;

/// SIMD-accelerated element-wise minimum for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_min_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
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
        let result_simd = a_simd.min(b_simd);
        let result_array: [f32; 8] = result_simd.into();
        result[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        result[i] = a[i].min(b[i]);
    }
}

/// SIMD-accelerated element-wise maximum for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_max_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
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
        let result_simd = a_simd.max(b_simd);
        let result_array: [f32; 8] = result_simd.into();
        result[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        result[i] = a[i].max(b[i]);
    }
}

/// SIMD-accelerated reduction to find minimum value
#[cfg(feature = "simd")]
pub fn simd_reduce_min_f32(input: &[f32]) -> f32 {
    if input.is_empty() {
        return f32::INFINITY;
    }

    let len = input.len();
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let mut min_simd = f32x8::splat(f32::INFINITY);

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
        min_simd = min_simd.min(input_simd);
    }

    let min_array: [f32; 8] = min_simd.into();
    let mut result = min_array.iter().fold(f32::INFINITY, |a, &b| a.min(b));

    for &value in input.iter().take(len).skip(remainder_start) {
        result = result.min(value);
    }

    result
}

/// SIMD-accelerated reduction to find maximum value
#[cfg(feature = "simd")]
pub fn simd_reduce_max_f32(input: &[f32]) -> f32 {
    if input.is_empty() {
        return f32::NEG_INFINITY;
    }

    let len = input.len();
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let mut max_simd = f32x8::splat(f32::NEG_INFINITY);

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
        max_simd = max_simd.max(input_simd);
    }

    let max_array: [f32; 8] = max_simd.into();
    let mut result = max_array.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    for &value in input.iter().take(len).skip(remainder_start) {
        result = result.max(value);
    }

    result
}

// Fallback implementations when SIMD is not available
#[cfg(not(feature = "simd"))]
pub fn simd_min_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        result[i] = a[i].min(b[i]);
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_max_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        result[i] = a[i].max(b[i]);
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_reduce_min_f32(input: &[f32]) -> f32 {
    input.iter().fold(f32::INFINITY, |a, &b| a.min(b))
}

#[cfg(not(feature = "simd"))]
pub fn simd_reduce_max_f32(input: &[f32]) -> f32 {
    input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_min_f32() {
        let a = [1.0, 5.0, 3.0, 8.0];
        let b = [2.0, 4.0, 6.0, 1.0];
        let mut result = [0.0; 4];

        simd_min_f32(&a, &b, &mut result);

        assert_eq!(result[0], 1.0); // min(1.0, 2.0)
        assert_eq!(result[1], 4.0); // min(5.0, 4.0)
        assert_eq!(result[2], 3.0); // min(3.0, 6.0)
        assert_eq!(result[3], 1.0); // min(8.0, 1.0)
    }

    #[test]
    fn test_simd_max_f32() {
        let a = [1.0, 5.0, 3.0, 8.0];
        let b = [2.0, 4.0, 6.0, 1.0];
        let mut result = [0.0; 4];

        simd_max_f32(&a, &b, &mut result);

        assert_eq!(result[0], 2.0); // max(1.0, 2.0)
        assert_eq!(result[1], 5.0); // max(5.0, 4.0)
        assert_eq!(result[2], 6.0); // max(3.0, 6.0)
        assert_eq!(result[3], 8.0); // max(8.0, 1.0)
    }

    #[test]
    fn test_simd_reduce_min_f32() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let result = simd_reduce_min_f32(&input);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_simd_reduce_max_f32() {
        let input = [3.0, 1.0, 4.0, 1.0, 5.0];
        let result = simd_reduce_max_f32(&input);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_empty_arrays() {
        let input: &[f32] = &[];
        assert_eq!(simd_reduce_min_f32(input), f32::INFINITY);
        assert_eq!(simd_reduce_max_f32(input), f32::NEG_INFINITY);
    }
}
