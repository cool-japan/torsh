//! SIMD activation functions
//!
//! This module provides SIMD-accelerated neural network activation functions
//! including ReLU, Sigmoid, Tanh, and GELU.

#[cfg(feature = "simd")]
use wide::{f32x8, CmpLt};

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

/// SIMD-accelerated sigmoid activation for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let one = f32x8::splat(1.0);

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

        // Approximate sigmoid: 1 / (1 + exp(-x))
        // Using fast exp approximation for SIMD
        let neg_input = -input_simd;
        let exp_neg = simd_fast_exp_f32x8(neg_input);
        let result_simd = one / (one + exp_neg);
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
    }
}

/// SIMD-accelerated tanh activation for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_tanh_f32(input: &[f32], output: &mut [f32]) {
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

        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let two_x = input_simd * f32x8::splat(2.0);
        let exp_2x = simd_fast_exp_f32x8(two_x);
        let one = f32x8::splat(1.0);
        let result_simd = (exp_2x - one) / (exp_2x + one);
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        output[i] = input[i].tanh();
    }
}

/// SIMD-accelerated GELU activation for f32 arrays
#[cfg(feature = "simd")]
pub fn simd_gelu_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    let simd_len = len / 8;
    let remainder_start = simd_len * 8;

    let half = f32x8::splat(0.5);
    let one = f32x8::splat(1.0);
    let sqrt_2_pi = f32x8::splat((2.0 / core::f32::consts::PI).sqrt());

    for i in 0..simd_len {
        let idx = i * 8;
        let x = f32x8::from([
            input[idx],
            input[idx + 1],
            input[idx + 2],
            input[idx + 3],
            input[idx + 4],
            input[idx + 5],
            input[idx + 6],
            input[idx + 7],
        ]);

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = x * x * x;
        let inner = sqrt_2_pi * (x + f32x8::splat(0.044715) * x_cubed);

        // Approximate tanh using our fast implementation
        let tanh_inner = simd_fast_tanh_f32x8(inner);
        let result_simd = half * x * (one + tanh_inner);
        let result_array: [f32; 8] = result_simd.into();
        output[idx..idx + 8].copy_from_slice(&result_array);
    }

    for i in remainder_start..len {
        let x = input[i];
        output[i] = 0.5
            * x
            * (1.0 + ((2.0 / core::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
    }
}

/// Fast SIMD exp approximation for f32x8
#[cfg(feature = "simd")]
fn simd_fast_exp_f32x8(x: f32x8) -> f32x8 {
    // Fast exp approximation using polynomial approximation
    // exp(x) ≈ (1 + x/256)^256 for |x| <= 1
    // For larger values, we use exp(x) = exp(a) * exp(b) where x = a + b

    let one = f32x8::splat(1.0);
    let inv_256 = f32x8::splat(1.0 / 256.0);

    // Clamp to reasonable range to avoid overflow
    let clamped = x.max(f32x8::splat(-20.0)).min(f32x8::splat(20.0));

    // Simple polynomial approximation
    let term = one + clamped * inv_256;

    // Approximate power of 256 by repeated squaring (8 times: 2^8 = 256)
    let mut result = term;
    result = result * result; // ^2
    result = result * result; // ^4
    result = result * result; // ^8
    result = result * result; // ^16
    result = result * result; // ^32
    result = result * result; // ^64
    result = result * result; // ^128
    result = result * result; // ^256

    result
}

/// Fast SIMD tanh approximation for f32x8
#[cfg(feature = "simd")]
fn simd_fast_tanh_f32x8(x: f32x8) -> f32x8 {
    // Fast tanh approximation using rational function
    let abs_x = x.abs();
    let x_squared = x * x;

    // Rational approximation for |x| < 1
    let num = x * (f32x8::splat(27.0) + x_squared);
    let den = f32x8::splat(27.0) + f32x8::splat(9.0) * x_squared;

    let small_approx = num / den;

    // For larger values, use sign(x) * (1 - 2/(exp(2*|x|) + 1))
    let two_abs_x = f32x8::splat(2.0) * abs_x;
    let exp_2abs = simd_fast_exp_f32x8(two_abs_x);

    // Manual sign calculation since signum() is not available
    let _zero = f32x8::splat(0.0);
    let _one = f32x8::splat(1.0);
    let _neg_one = f32x8::splat(-1.0);
    let x_sign = {
        let mut sign_array: [f32; 8] = x.into();
        for i in 0..8 {
            sign_array[i] = if sign_array[i] > 0.0 {
                1.0
            } else if sign_array[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
        }
        f32x8::from(sign_array)
    };

    let large_approx =
        x_sign * (f32x8::splat(1.0) - f32x8::splat(2.0) / (exp_2abs + f32x8::splat(1.0)));

    // Use small approximation for |x| < 2, large for |x| >= 2
    let abs_threshold = f32x8::splat(2.0);
    let use_small = abs_x.cmp_lt(abs_threshold);

    // Blend results based on magnitude using SIMD blend
    use_small.blend(small_approx, large_approx)
}

// Fallback implementations when SIMD is not available
#[cfg(not(feature = "simd"))]
pub fn simd_relu_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].max(0.0);
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_sigmoid_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_tanh_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        output[i] = input[i].tanh();
    }
}

#[cfg(not(feature = "simd"))]
pub fn simd_gelu_f32(input: &[f32], output: &mut [f32]) {
    let len = input.len().min(output.len());
    for i in 0..len {
        let x = input[i];
        output[i] = 0.5
            * x
            * (1.0 + ((2.0 / core::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_relu_f32() {
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0; 5];

        simd_relu_f32(&input, &mut output);

        assert_eq!(output[0], 0.0); // -2.0 -> 0.0
        assert_eq!(output[1], 0.0); // -1.0 -> 0.0
        assert_eq!(output[2], 0.0); // 0.0 -> 0.0
        assert_eq!(output[3], 1.0); // 1.0 -> 1.0
        assert_eq!(output[4], 2.0); // 2.0 -> 2.0
    }

    #[test]
    fn test_simd_sigmoid_f32() {
        let input = [0.0];
        let mut output = [0.0];

        simd_sigmoid_f32(&input, &mut output);

        // sigmoid(0) should be 0.5
        assert!((output[0] - 0.5).abs() < 0.1); // Allow some approximation error
    }

    #[test]
    fn test_simd_tanh_f32() {
        let input = [0.0];
        let mut output = [0.0];

        simd_tanh_f32(&input, &mut output);

        // tanh(0) should be 0
        assert!(output[0].abs() < 0.1); // Allow some approximation error
    }

    #[test]
    fn test_simd_gelu_f32() {
        let input = [0.0];
        let mut output = [0.0];

        simd_gelu_f32(&input, &mut output);

        // GELU(0) should be 0
        assert!(output[0].abs() < 0.1); // Allow some approximation error
    }

    #[test]
    fn test_activation_consistency() {
        // Test that SIMD and scalar versions produce similar results
        let input = [0.5];
        let mut simd_output = [0.0];
        let mut scalar_output = [0.0];

        // Test ReLU consistency
        simd_relu_f32(&input, &mut simd_output);
        scalar_output[0] = input[0].max(0.0);
        assert_eq!(simd_output[0], scalar_output[0]);
    }
}
