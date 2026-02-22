//! SIMD mathematical functions
//!
//! This module provides SIMD-accelerated transcendental mathematical functions
//! including exp, log, and other advanced math operations.

#[cfg(feature = "simd")]
use wide::f32x8;

/// SIMD-accelerated exponential function for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_exp_f32(input: &[f32], output: &mut [f32]) {
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

        let result_simd = simd_fast_exp_f32x8(input_simd);
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = input[i].exp();
    }
}

/// SIMD-accelerated natural logarithm for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_log_f32(input: &[f32], output: &mut [f32]) {
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

        let result_simd = simd_fast_log_f32x8(input_simd.max(f32x8::splat(1e-30)));
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = input[i].ln();
    }
}

/// Fast SIMD exponential approximation
#[cfg(feature = "simd")]
pub fn simd_fast_exp_f32x8(x: f32x8) -> f32x8 {
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
pub fn simd_fast_log_f32x8(x: f32x8) -> f32x8 {
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
pub fn simd_exp_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].exp();
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_log_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].ln();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_exp_f32() {
        let input = [0.0, 1.0];
        let mut output = [0.0; 2];

        simd_exp_f32(&input, &mut output);

        // exp(0) ≈ 1, exp(1) ≈ 2.718
        assert!((output[0] - 1.0).abs() < 0.1);
        assert!((output[1] - 2.718).abs() < 0.5);
    }

    #[test]
    fn test_simd_log_f32() {
        let input = [1.0, 2.718];
        let mut output = [0.0; 2];

        simd_log_f32(&input, &mut output);

        // ln(1) = 0, ln(e) ≈ 1
        assert!(output[0].abs() < 0.1);
        assert!((output[1] - 1.0).abs() < 0.5);
    }
}
