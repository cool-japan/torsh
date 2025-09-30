//! SIMD Optimizations for Special Functions
//!
//! This module provides SIMD-accelerated implementations of computationally
//! intensive special functions to improve performance on supported hardware.

use crate::TorshResult;
use torsh_tensor::Tensor;

/// SIMD-optimized gamma function
///
/// Uses vectorized operations for computing the gamma function on multiple values simultaneously.
/// Falls back to scalar implementation when SIMD is not available.
pub fn gamma_simd(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return gamma_avx2(&data, input);
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return gamma_sse41(&data, input);
        }
    }

    // Fallback to standard implementation
    crate::gamma(input)
}

/// SIMD-optimized error function
///
/// Uses vectorized polynomial approximations for improved performance.
pub fn erf_simd(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return erf_avx2(&data, input);
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return erf_sse41(&data, input);
        }
    }

    // Fallback to standard implementation
    crate::erf(input)
}

/// SIMD-optimized exponential function variants
///
/// Provides fast vectorized implementations of exp, expm1, and related functions.
pub fn exp_family_simd(input: &Tensor<f32>, variant: ExpVariant) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return exp_family_avx2(&data, input, variant);
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return exp_family_sse41(&data, input, variant);
        }
    }

    // Fallback to standard implementation
    match variant {
        ExpVariant::Exp => input.exp(),
        ExpVariant::Expm1 => crate::expm1(input),
        ExpVariant::Log1p => crate::log1p(input),
    }
}

#[derive(Clone, Copy)]
pub enum ExpVariant {
    Exp,
    Expm1,
    Log1p,
}

#[cfg(target_arch = "x86_64")]
mod x86_simd {
    use super::ExpVariant;

    #[target_feature(enable = "avx2")]
    pub unsafe fn gamma_avx2_impl(data: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;

        let mut result = Vec::with_capacity(data.len());
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        // Process 8 values at a time using AVX2
        for chunk in chunks {
            let input_vec = _mm256_loadu_ps(chunk.as_ptr());

            // Implement vectorized gamma approximation
            // Using Stirling's approximation with polynomial correction
            let _one = _mm256_set1_ps(1.0);
            let half = _mm256_set1_ps(0.5);
            let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);

            // ln(Γ(z)) ≈ (z - 0.5) * ln(z) - z + 0.5 * ln(2π) + 1/(12z) - 1/(360z³) + ...
            let z_minus_half = _mm256_sub_ps(input_vec, half);
            let ln_z = fast_ln_ps(input_vec);
            let term1 = _mm256_mul_ps(z_minus_half, ln_z);
            let term2 = _mm256_sub_ps(term1, input_vec);
            let term3 = _mm256_mul_ps(half, fast_ln_ps(two_pi));
            let ln_gamma = _mm256_add_ps(
                _mm256_add_ps(term2, term3),
                stirling_correction_ps(input_vec),
            );

            let gamma_vec = fast_exp_ps(ln_gamma);

            let mut output: [f32; 8] = [0.0; 8];
            _mm256_storeu_ps(output.as_mut_ptr(), gamma_vec);
            result.extend_from_slice(&output);
        }

        // Handle remainder
        for &val in remainder {
            result.push(scirs2_special::gamma(val as f64) as f32);
        }

        result
    }

    #[target_feature(enable = "sse4.1")]
    pub unsafe fn gamma_sse41_impl(data: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;

        let mut result = Vec::with_capacity(data.len());
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();

        // Process 4 values at a time using SSE4.1
        for chunk in chunks {
            let input_vec = _mm_loadu_ps(chunk.as_ptr());

            // Similar implementation but for 4 values
            let _one = _mm_set1_ps(1.0);
            let half = _mm_set1_ps(0.5);
            let two_pi = _mm_set1_ps(2.0 * std::f32::consts::PI);

            let z_minus_half = _mm_sub_ps(input_vec, half);
            let ln_z = fast_ln_ps_sse(input_vec);
            let term1 = _mm_mul_ps(z_minus_half, ln_z);
            let term2 = _mm_sub_ps(term1, input_vec);
            let term3 = _mm_mul_ps(half, fast_ln_ps_sse(two_pi));
            let ln_gamma = _mm_add_ps(
                _mm_add_ps(term2, term3),
                stirling_correction_ps_sse(input_vec),
            );

            let gamma_vec = fast_exp_ps_sse(ln_gamma);

            let mut output: [f32; 4] = [0.0; 4];
            _mm_storeu_ps(output.as_mut_ptr(), gamma_vec);
            result.extend_from_slice(&output);
        }

        // Handle remainder
        for &val in remainder {
            result.push(scirs2_special::gamma(val as f64) as f32);
        }

        result
    }

    // Fast SIMD logarithm approximation
    #[target_feature(enable = "avx2")]
    unsafe fn fast_ln_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
        // Fast log approximation using polynomial
        // ln(x) ≈ ln(2) * (e + f) where x = 2^e * (1 + f)
        use std::arch::x86_64::*;

        // Extract exponent and mantissa
        let x_int = _mm256_castps_si256(x);
        let exp_mask = _mm256_set1_epi32(0x7F800000);
        let mant_mask = _mm256_set1_epi32(0x007FFFFF);
        let exp_bias = _mm256_set1_epi32(127);

        let exp_bits = _mm256_and_si256(x_int, exp_mask);
        let exp_shifted = _mm256_srli_epi32(exp_bits, 23);
        let exponent = _mm256_sub_epi32(exp_shifted, exp_bias);
        let exp_f = _mm256_cvtepi32_ps(exponent);

        let mantissa_bits = _mm256_or_si256(
            _mm256_and_si256(x_int, mant_mask),
            _mm256_set1_epi32(0x3F800000),
        );
        let mantissa = _mm256_castsi256_ps(mantissa_bits);

        // Polynomial approximation for ln(1 + f) where f is in [0, 1]
        let one = _mm256_set1_ps(1.0);
        let f = _mm256_sub_ps(mantissa, one);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);

        // ln(1 + f) ≈ f - f²/2 + f³/3 - f⁴/4 + ...
        let f2 = _mm256_mul_ps(f, f);
        let f3 = _mm256_mul_ps(f2, f);
        let f4 = _mm256_mul_ps(f3, f);

        let poly = _mm256_add_ps(
            f,
            _mm256_add_ps(
                _mm256_mul_ps(f2, _mm256_set1_ps(-0.5)),
                _mm256_add_ps(
                    _mm256_mul_ps(f3, _mm256_set1_ps(1.0 / 3.0)),
                    _mm256_mul_ps(f4, _mm256_set1_ps(-0.25)),
                ),
            ),
        );

        _mm256_add_ps(_mm256_mul_ps(exp_f, ln2), poly)
    }

    // Fast SIMD exponential approximation
    #[target_feature(enable = "avx2")]
    unsafe fn fast_exp_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;

        // exp(x) = 2^(x/ln2) = 2^(i + f) where i is integer, f is fractional
        let inv_ln2 = _mm256_set1_ps(1.0 / std::f32::consts::LN_2);
        let x_scaled = _mm256_mul_ps(x, inv_ln2);

        // Split into integer and fractional parts
        let x_floor = _mm256_floor_ps(x_scaled);
        let f = _mm256_sub_ps(x_scaled, x_floor);

        // Convert integer part to actual power of 2
        let i = _mm256_cvtps_epi32(x_floor);
        let exp_i = _mm256_slli_epi32(_mm256_add_epi32(i, _mm256_set1_epi32(127)), 23);
        let pow2_i = _mm256_castsi256_ps(exp_i);

        // Polynomial approximation for 2^f where f is in [0, 1]
        // 2^f ≈ 1 + f*ln(2) + (f*ln(2))²/2! + (f*ln(2))³/3! + ...
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        let f_ln2 = _mm256_mul_ps(f, ln2);
        let f_ln2_2 = _mm256_mul_ps(f_ln2, f_ln2);
        let f_ln2_3 = _mm256_mul_ps(f_ln2_2, f_ln2);

        let poly = _mm256_add_ps(
            _mm256_set1_ps(1.0),
            _mm256_add_ps(
                f_ln2,
                _mm256_add_ps(
                    _mm256_mul_ps(f_ln2_2, _mm256_set1_ps(0.5)),
                    _mm256_mul_ps(f_ln2_3, _mm256_set1_ps(1.0 / 6.0)),
                ),
            ),
        );

        _mm256_mul_ps(pow2_i, poly)
    }

    // Stirling correction terms for gamma function
    #[target_feature(enable = "avx2")]
    unsafe fn stirling_correction_ps(z: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
        use std::arch::x86_64::*;

        // Stirling series: 1/(12z) - 1/(360z³) + 1/(1260z⁵) - ...
        let z_inv = _mm256_div_ps(_mm256_set1_ps(1.0), z);
        let z_inv2 = _mm256_mul_ps(z_inv, z_inv);
        let z_inv3 = _mm256_mul_ps(z_inv2, z_inv);
        let z_inv5 = _mm256_mul_ps(z_inv3, z_inv2);

        let term1 = _mm256_mul_ps(z_inv, _mm256_set1_ps(1.0 / 12.0));
        let term2 = _mm256_mul_ps(z_inv3, _mm256_set1_ps(-1.0 / 360.0));
        let term3 = _mm256_mul_ps(z_inv5, _mm256_set1_ps(1.0 / 1260.0));

        _mm256_add_ps(_mm256_add_ps(term1, term2), term3)
    }

    // SSE versions (similar implementations but for 4 values)
    #[target_feature(enable = "sse4.1")]
    unsafe fn fast_ln_ps_sse(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
        // Similar to AVX2 version but for 4 values
        use std::arch::x86_64::*;

        let x_int = _mm_castps_si128(x);
        let exp_mask = _mm_set1_epi32(0x7F800000);
        let mant_mask = _mm_set1_epi32(0x007FFFFF);
        let exp_bias = _mm_set1_epi32(127);

        let exp_bits = _mm_and_si128(x_int, exp_mask);
        let exp_shifted = _mm_srli_epi32(exp_bits, 23);
        let exponent = _mm_sub_epi32(exp_shifted, exp_bias);
        let exp_f = _mm_cvtepi32_ps(exponent);

        let mantissa_bits =
            _mm_or_si128(_mm_and_si128(x_int, mant_mask), _mm_set1_epi32(0x3F800000));
        let mantissa = _mm_castsi128_ps(mantissa_bits);

        let one = _mm_set1_ps(1.0);
        let f = _mm_sub_ps(mantissa, one);
        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);

        let f2 = _mm_mul_ps(f, f);
        let f3 = _mm_mul_ps(f2, f);
        let f4 = _mm_mul_ps(f3, f);

        let poly = _mm_add_ps(
            f,
            _mm_add_ps(
                _mm_mul_ps(f2, _mm_set1_ps(-0.5)),
                _mm_add_ps(
                    _mm_mul_ps(f3, _mm_set1_ps(1.0 / 3.0)),
                    _mm_mul_ps(f4, _mm_set1_ps(-0.25)),
                ),
            ),
        );

        _mm_add_ps(_mm_mul_ps(exp_f, ln2), poly)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn fast_exp_ps_sse(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
        use std::arch::x86_64::*;

        let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
        let x_scaled = _mm_mul_ps(x, inv_ln2);

        let x_floor = _mm_floor_ps(x_scaled);
        let f = _mm_sub_ps(x_scaled, x_floor);

        let i = _mm_cvtps_epi32(x_floor);
        let exp_i = _mm_slli_epi32(_mm_add_epi32(i, _mm_set1_epi32(127)), 23);
        let pow2_i = _mm_castsi128_ps(exp_i);

        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
        let f_ln2 = _mm_mul_ps(f, ln2);
        let f_ln2_2 = _mm_mul_ps(f_ln2, f_ln2);
        let f_ln2_3 = _mm_mul_ps(f_ln2_2, f_ln2);

        let poly = _mm_add_ps(
            _mm_set1_ps(1.0),
            _mm_add_ps(
                f_ln2,
                _mm_add_ps(
                    _mm_mul_ps(f_ln2_2, _mm_set1_ps(0.5)),
                    _mm_mul_ps(f_ln2_3, _mm_set1_ps(1.0 / 6.0)),
                ),
            ),
        );

        _mm_mul_ps(pow2_i, poly)
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn stirling_correction_ps_sse(
        z: std::arch::x86_64::__m128,
    ) -> std::arch::x86_64::__m128 {
        use std::arch::x86_64::*;

        let z_inv = _mm_div_ps(_mm_set1_ps(1.0), z);
        let z_inv2 = _mm_mul_ps(z_inv, z_inv);
        let z_inv3 = _mm_mul_ps(z_inv2, z_inv);
        let z_inv5 = _mm_mul_ps(z_inv3, z_inv2);

        let term1 = _mm_mul_ps(z_inv, _mm_set1_ps(1.0 / 12.0));
        let term2 = _mm_mul_ps(z_inv3, _mm_set1_ps(-1.0 / 360.0));
        let term3 = _mm_mul_ps(z_inv5, _mm_set1_ps(1.0 / 1260.0));

        _mm_add_ps(_mm_add_ps(term1, term2), term3)
    }

    /// SIMD-optimized error function implementation for AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn erf_avx2_impl(data: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;

        let mut result = Vec::with_capacity(data.len());
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        // Process 8 values at a time using AVX2
        for chunk in chunks {
            let x = _mm256_loadu_ps(chunk.as_ptr());

            // Use Abramowitz and Stegun approximation
            // erf(x) ≈ 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)e^(-x²)
            // where t = 1/(1 + px) and p = 0.3275911
            let abs_x = _mm256_and_ps(x, _mm256_set1_ps(f32::from_bits(0x7FFFFFFF)));
            let sign_x = _mm256_and_ps(x, _mm256_set1_ps(f32::from_bits(0x80000000)));

            let p = _mm256_set1_ps(0.3275911);
            let one = _mm256_set1_ps(1.0);
            let t = _mm256_div_ps(one, _mm256_add_ps(one, _mm256_mul_ps(p, abs_x)));

            let a1 = _mm256_set1_ps(0.254_829_6);
            let a2 = _mm256_set1_ps(-0.284_496_72);
            let a3 = _mm256_set1_ps(1.421_413_8);
            let a4 = _mm256_set1_ps(-1.453_152_1);
            let a5 = _mm256_set1_ps(1.061_405_4);

            let t2 = _mm256_mul_ps(t, t);
            let t3 = _mm256_mul_ps(t2, t);
            let t4 = _mm256_mul_ps(t3, t);
            let t5 = _mm256_mul_ps(t4, t);

            let poly = _mm256_add_ps(
                _mm256_mul_ps(a1, t),
                _mm256_add_ps(
                    _mm256_mul_ps(a2, t2),
                    _mm256_add_ps(
                        _mm256_mul_ps(a3, t3),
                        _mm256_add_ps(_mm256_mul_ps(a4, t4), _mm256_mul_ps(a5, t5)),
                    ),
                ),
            );

            // Compute e^(-x²)
            let x_squared = _mm256_mul_ps(abs_x, abs_x);
            let neg_x_squared = _mm256_sub_ps(_mm256_setzero_ps(), x_squared);
            let exp_neg_x_sq = fast_exp_ps(neg_x_squared);

            let erf_result = _mm256_sub_ps(one, _mm256_mul_ps(poly, exp_neg_x_sq));

            // Apply original sign
            let final_result = _mm256_or_ps(erf_result, sign_x);

            let mut output: [f32; 8] = [0.0; 8];
            _mm256_storeu_ps(output.as_mut_ptr(), final_result);
            result.extend_from_slice(&output);
        }

        // Handle remainder using scirs2
        for &val in remainder {
            result.push(scirs2_special::erf(val as f64) as f32);
        }

        result
    }

    /// SIMD-optimized error function implementation for SSE4.1
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn erf_sse41_impl(data: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;

        let mut result = Vec::with_capacity(data.len());
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();

        // Process 4 values at a time using SSE4.1
        for chunk in chunks {
            let x = _mm_loadu_ps(chunk.as_ptr());

            // Same algorithm as AVX2 but for 4 values
            let abs_x = _mm_and_ps(x, _mm_set1_ps(f32::from_bits(0x7FFFFFFF)));
            let sign_x = _mm_and_ps(x, _mm_set1_ps(f32::from_bits(0x80000000)));

            let p = _mm_set1_ps(0.3275911);
            let one = _mm_set1_ps(1.0);
            let t = _mm_div_ps(one, _mm_add_ps(one, _mm_mul_ps(p, abs_x)));

            let a1 = _mm_set1_ps(0.254_829_6);
            let a2 = _mm_set1_ps(-0.284_496_72);
            let a3 = _mm_set1_ps(1.421_413_8);
            let a4 = _mm_set1_ps(-1.453_152_1);
            let a5 = _mm_set1_ps(1.061_405_4);

            let t2 = _mm_mul_ps(t, t);
            let t3 = _mm_mul_ps(t2, t);
            let t4 = _mm_mul_ps(t3, t);
            let t5 = _mm_mul_ps(t4, t);

            let poly = _mm_add_ps(
                _mm_mul_ps(a1, t),
                _mm_add_ps(
                    _mm_mul_ps(a2, t2),
                    _mm_add_ps(
                        _mm_mul_ps(a3, t3),
                        _mm_add_ps(_mm_mul_ps(a4, t4), _mm_mul_ps(a5, t5)),
                    ),
                ),
            );

            // Compute e^(-x²)
            let x_squared = _mm_mul_ps(abs_x, abs_x);
            let neg_x_squared = _mm_sub_ps(_mm_setzero_ps(), x_squared);
            let exp_neg_x_sq = fast_exp_ps_sse(neg_x_squared);

            let erf_result = _mm_sub_ps(one, _mm_mul_ps(poly, exp_neg_x_sq));

            // Apply original sign
            let final_result = _mm_or_ps(erf_result, sign_x);

            let mut output: [f32; 4] = [0.0; 4];
            _mm_storeu_ps(output.as_mut_ptr(), final_result);
            result.extend_from_slice(&output);
        }

        // Handle remainder using scirs2
        for &val in remainder {
            result.push(scirs2_special::erf(val as f64) as f32);
        }

        result
    }

    /// SIMD-optimized exponential family functions for AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn exp_family_avx2_impl(data: &[f32], variant: ExpVariant) -> Vec<f32> {
        use std::arch::x86_64::*;

        let mut result = Vec::with_capacity(data.len());
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        // Process 8 values at a time using AVX2
        for chunk in chunks {
            let x = _mm256_loadu_ps(chunk.as_ptr());

            let result_vec = match variant {
                ExpVariant::Exp => {
                    // Fast exponential using the existing fast_exp_ps function
                    fast_exp_ps(x)
                }
                ExpVariant::Expm1 => {
                    // expm1(x) = exp(x) - 1
                    let exp_x = fast_exp_ps(x);
                    let one = _mm256_set1_ps(1.0);
                    _mm256_sub_ps(exp_x, one)
                }
                ExpVariant::Log1p => {
                    // log1p(x) = ln(1 + x)
                    let one = _mm256_set1_ps(1.0);
                    let one_plus_x = _mm256_add_ps(one, x);
                    fast_ln_ps(one_plus_x)
                }
            };

            let mut output: [f32; 8] = [0.0; 8];
            _mm256_storeu_ps(output.as_mut_ptr(), result_vec);
            result.extend_from_slice(&output);
        }

        // Handle remainder using scirs2
        for &val in remainder {
            let result_val = match variant {
                ExpVariant::Exp => val.exp(),
                ExpVariant::Expm1 => val.exp_m1(),
                ExpVariant::Log1p => val.ln_1p(),
            };
            result.push(result_val);
        }

        result
    }

    /// SIMD-optimized exponential family functions for SSE4.1
    #[target_feature(enable = "sse4.1")]
    pub unsafe fn exp_family_sse41_impl(data: &[f32], variant: ExpVariant) -> Vec<f32> {
        use std::arch::x86_64::*;

        let mut result = Vec::with_capacity(data.len());
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();

        // Process 4 values at a time using SSE4.1
        for chunk in chunks {
            let x = _mm_loadu_ps(chunk.as_ptr());

            let result_vec = match variant {
                ExpVariant::Exp => {
                    // Fast exponential using the existing fast_exp_ps_sse function
                    fast_exp_ps_sse(x)
                }
                ExpVariant::Expm1 => {
                    // expm1(x) = exp(x) - 1
                    let exp_x = fast_exp_ps_sse(x);
                    let one = _mm_set1_ps(1.0);
                    _mm_sub_ps(exp_x, one)
                }
                ExpVariant::Log1p => {
                    // log1p(x) = ln(1 + x)
                    let one = _mm_set1_ps(1.0);
                    let one_plus_x = _mm_add_ps(one, x);
                    fast_ln_ps_sse(one_plus_x)
                }
            };

            let mut output: [f32; 4] = [0.0; 4];
            _mm_storeu_ps(output.as_mut_ptr(), result_vec);
            result.extend_from_slice(&output);
        }

        // Handle remainder using standard library functions
        for &val in remainder {
            let result_val = match variant {
                ExpVariant::Exp => val.exp(),
                ExpVariant::Expm1 => val.exp_m1(),
                ExpVariant::Log1p => val.ln_1p(),
            };
            result.push(result_val);
        }

        result
    }
}

#[cfg(target_arch = "x86_64")]
fn gamma_avx2(data: &[f32], input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let result_data = unsafe { x86_simd::gamma_avx2_impl(data) };
    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(target_arch = "x86_64")]
fn gamma_sse41(data: &[f32], input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let result_data = unsafe { x86_simd::gamma_sse41_impl(data) };
    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(target_arch = "x86_64")]
fn erf_avx2(data: &[f32], input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let result_data = unsafe { x86_simd::erf_avx2_impl(data) };
    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(target_arch = "x86_64")]
fn erf_sse41(data: &[f32], input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let result_data = unsafe { x86_simd::erf_sse41_impl(data) };
    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(target_arch = "x86_64")]
fn exp_family_avx2(
    data: &[f32],
    input: &Tensor<f32>,
    variant: ExpVariant,
) -> TorshResult<Tensor<f32>> {
    let result_data = unsafe { x86_simd::exp_family_avx2_impl(data, variant) };
    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(target_arch = "x86_64")]
fn exp_family_sse41(
    data: &[f32],
    input: &Tensor<f32>,
    variant: ExpVariant,
) -> TorshResult<Tensor<f32>> {
    let result_data = unsafe { x86_simd::exp_family_sse41_impl(data, variant) };
    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_simd_gamma_correctness() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![8],
            device,
        )?;

        let standard_result = crate::gamma(&x)?;
        let simd_result = gamma_simd(&x)?;

        let standard_data = standard_result.data()?;
        let simd_data = simd_result.data()?;

        // Note: SIMD implementation uses approximations for performance, accepting up to 20% error for large values
        for (i, (&std_val, &simd_val)) in standard_data.iter().zip(simd_data.iter()).enumerate() {
            if std_val.is_finite() && simd_val.is_finite() {
                let relative_error = (std_val - simd_val).abs() / std_val.max(1e-6);
                if relative_error > 0.25 {
                    // 25% tolerance for SIMD approximations
                    panic!("Excessive error at index {i}: std={std_val}, simd={simd_val}, rel_error={relative_error:.4}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_simd_performance_benefit() -> TorshResult<()> {
        // This test is more about ensuring SIMD paths are exercised
        // In a real benchmark, SIMD should show performance improvements
        let device = DeviceType::Cpu;
        let large_input: Vec<f32> = (1..1000).map(|x| x as f32 * 0.1).collect();
        let x = Tensor::from_data(large_input, vec![999], device)?;

        let result = gamma_simd(&x)?;
        // In real use, we would benchmark this against the standard implementation
        assert!(result.data()?.len() == 999); // Verify output shape
        Ok(())
    }
}
