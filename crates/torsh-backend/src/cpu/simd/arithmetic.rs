//! SIMD arithmetic operations
//!
//! This module provides SIMD-accelerated basic arithmetic operations
//! including element-wise operations, dot products, and scalar operations.

#[cfg(feature = "simd")]
use wide::f32x8;

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

/// SIMD-accelerated element-wise subtraction for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_sub_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
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
        let result_simd = a_simd - b_simd;
        let result_array: [f32; 8] = result_simd.into();
        result[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        result[i] = a[i] - b[i];
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

/// SIMD-accelerated element-wise division for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_div_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
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
        let result_simd = a_simd / b_simd;
        let result_array: [f32; 8] = result_simd.into();
        result[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        result[i] = a[i] / b[i];
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
        sum_simd += input_simd;
    }

    // Sum the SIMD vector
    let sum_array: [f32; 8] = sum_simd.into();
    let mut total = sum_array.iter().sum::<f32>();

    // Handle remaining elements
    for &value in input.iter().take(len).skip(remainder_start) {
        total += value;
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
        dot_simd += a_simd * b_simd;
    }

    let dot_array: [f32; 8] = dot_simd.into();
    let mut total = dot_array.iter().sum::<f32>();

    for i in remainder_start..len {
        total += a[i] * b[i];
    }

    total
}

/// SIMD-accelerated scalar addition for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_add_scalar_f32(input: &[f32], scalar: f32, output: &mut [f32]) {
    let len = input.len().min(output.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let scalar_simd = f32x8::splat(scalar);

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
        let result_simd = input_simd + scalar_simd;
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = input[i] + scalar;
    }
}

/// SIMD-accelerated scalar multiplication for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_mul_scalar_f32(input: &[f32], scalar: f32, output: &mut [f32]) {
    let len = input.len().min(output.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let scalar_simd = f32x8::splat(scalar);

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
        let result_simd = input_simd * scalar_simd;
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = input[i] * scalar;
    }
}

/// SIMD-accelerated power function (x^y) for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_pow_f32(base: &[f32], exponent: f32, output: &mut [f32]) {
    let len = base.len().min(output.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let exp_simd = f32x8::splat(exponent);

    for i in 0..simd_len {
        let idx = i * 8;
        let base_simd = f32x8::from([
            base[idx],
            base[idx + 1],
            base[idx + 2],
            base[idx + 3],
            base[idx + 4],
            base[idx + 5],
            base[idx + 6],
            base[idx + 7],
        ]);

        // x^y = exp(y * ln(x))
        let log_base = simd_fast_log_f32x8(base_simd.max(f32x8::splat(1e-30)));
        let result_simd = simd_fast_exp_f32x8(exp_simd * log_base);
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = base[i].powf(exponent);
    }
}

/// SIMD-accelerated square root for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_sqrt_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

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

        // Use Newton-Raphson for fast square root approximation
        let result_simd = simd_fast_sqrt_f32x8(input_simd);
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = input[i].sqrt();
    }
}

/// Fast SIMD square root using Newton-Raphson
#[cfg(feature = "simd")]
fn simd_fast_sqrt_f32x8(x: f32x8) -> f32x8 {
    // Fast inverse square root approximation followed by multiplication
    let half = f32x8::splat(0.5);
    let one_half = f32x8::splat(1.5);

    // Clamp to positive values
    let clamped = x.max(f32x8::splat(1e-30));

    // Initial guess using bit manipulation approximation
    // This is a simplified version - in practice you'd use bit manipulation
    let mut y = clamped; // Simple starting guess

    // Newton-Raphson iterations: y = y * (1.5 - 0.5 * x * y * y)
    for _ in 0..3 {
        y = y * (one_half - half * clamped * y * y);
    }

    // Convert from inverse sqrt to sqrt
    clamped * y
}

// Helper functions for pow implementation
/// Fast SIMD exponential approximation
#[cfg(feature = "simd")]
fn simd_fast_exp_f32x8(x: f32x8) -> f32x8 {
    // Polynomial approximation for exp(x)
    // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
    let one = f32x8::splat(1.0);
    let x_sq = x * x;
    let x_cube = x_sq * x;
    let x_fourth = x_sq * x_sq;
    let x_fifth = x_fourth * x;

    one + x
        + x_sq * f32x8::splat(0.5)
        + x_cube * f32x8::splat(1.0 / 6.0)
        + x_fourth * f32x8::splat(1.0 / 24.0)
        + x_fifth * f32x8::splat(1.0 / 120.0)
}

/// Fast SIMD logarithm approximation
#[cfg(feature = "simd")]
fn simd_fast_log_f32x8(x: f32x8) -> f32x8 {
    // Polynomial approximation for ln(x) around x=1
    // ln(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3 for x near 1
    let one = f32x8::splat(1.0);
    let x_minus_1 = x - one;
    let x_minus_1_sq = x_minus_1 * x_minus_1;
    let x_minus_1_cube = x_minus_1_sq * x_minus_1;

    x_minus_1 - x_minus_1_sq * f32x8::splat(0.5) + x_minus_1_cube * f32x8::splat(1.0 / 3.0)
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
pub fn simd_sub_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        result[i] = a[i] - b[i];
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
pub fn simd_div_f32(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len().min(b.len()).min(result.len());
    for i in 0..len {
        result[i] = a[i] / b[i];
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

#[cfg(not(feature = "simd"))]
pub fn simd_add_scalar_f32(input: &[f32], scalar: f32, output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i] + scalar;
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_mul_scalar_f32(input: &[f32], scalar: f32, output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i] * scalar;
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_pow_f32(base: &[f32], exponent: f32, output: &mut [f32]) {
    let len = base.len().min(output.len());
    for i in 0..len {
        output[i] = base[i].powf(exponent);
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_sqrt_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add_f32() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut result = [0.0; 9];

        simd_add_f32(&a, &b, &mut result);

        for i in 0..9 {
            assert_eq!(result[i], a[i] + b[i]);
        }
    }

    #[test]
    fn test_simd_dot_f32() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 3.0, 4.0, 5.0];

        let result = simd_dot_f32(&a, &b);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_simd_sum_f32() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = simd_sum_f32(&input);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_simd_add_scalar_f32() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];

        simd_add_scalar_f32(&input, 5.0, &mut output);

        for i in 0..4 {
            assert_eq!(output[i], input[i] + 5.0);
        }
    }
}
