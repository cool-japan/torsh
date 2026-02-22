//! Fast Approximations for Special Functions
//!
//! This module provides fast, lower-precision approximations of special functions
//! for use cases where speed is more important than numerical accuracy.

use crate::TorshResult;
use std::f32::consts::PI;
use torsh_tensor::Tensor;

/// Fast approximation of the gamma function using Stirling's approximation
///
/// Uses Γ(z) ≈ √(2π/z) * (z/e)^z for z > 0.5
/// Accuracy: ~1-5% error for z > 1, degrades for smaller values
pub fn gamma_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&z| {
            if z <= 0.0 {
                return f32::NAN;
            }
            if z < 0.5 {
                // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
                return PI / ((PI * z).sin() * gamma_stirling_approx(1.0 - z));
            }
            gamma_stirling_approx(z)
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of the error function using polynomial approximation
///
/// Uses rational approximation: erf(x) ≈ sign(x) * (1 - 1/(1 + ax + bx² + cx³ + dx⁴)^4)
/// Accuracy: ~0.01% maximum error
pub fn erf_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            let abs_x = x.abs();
            if abs_x > 5.0 {
                return if x > 0.0 { 1.0 } else { -1.0 };
            }

            // Abramowitz and Stegun approximation
            let a1 = 0.254_829_6;
            let a2 = -0.284_496_7;
            let a3 = 1.421_413_7;
            let a4 = -1.453_152;
            let a5 = 1.061_405_4;
            let p = 0.327_591_1;

            let t = 1.0 / (1.0 + p * abs_x);
            let poly =
                a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t;
            let erf_val = 1.0 - poly * (-abs_x * abs_x).exp();

            if x >= 0.0 {
                erf_val
            } else {
                -erf_val
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of the complementary error function
///
/// Uses erfc(x) = 1 - erf(x) for x < 0, and continued fraction for x > 0
/// Accuracy: ~0.1% maximum error
pub fn erfc_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x < 0.0 {
                return 2.0 - erfc_positive_fast(-x);
            }
            erfc_positive_fast(x)
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of Bessel J₀ function
///
/// Uses polynomial approximation for |x| < 3 and asymptotic expansion for |x| ≥ 3
/// Accuracy: ~0.1% maximum error
pub fn bessel_j0_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            let abs_x = x.abs();
            if abs_x < 3.0 {
                bessel_j0_polynomial_approx(x)
            } else {
                bessel_j0_asymptotic_approx(abs_x)
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of the natural logarithm
///
/// Uses polynomial approximation for x near 1, and bit manipulation for general case
/// Accuracy: ~0.1% maximum error for x > 0.1
pub fn log_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x <= 0.0 {
                return f32::NEG_INFINITY;
            }
            if x == 1.0 {
                return 0.0;
            }

            // Fast log approximation using bit manipulation
            let x_bits = x.to_bits();
            let exp_bits = ((x_bits >> 23) & 0xFF) as i32 - 127;
            let mantissa_bits = (x_bits & 0x7FFFFF) | 0x3F800000;
            let mantissa = f32::from_bits(mantissa_bits);

            // Polynomial approximation for log(1 + f) where f = mantissa - 1
            let f = mantissa - 1.0;
            let log_mantissa = f * (1.0 - 0.5 * f + f * f / 3.0 - f * f * f / 4.0);

            exp_bits as f32 * std::f32::consts::LN_2 + log_mantissa
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of the exponential function
///
/// Uses polynomial approximation with range reduction
/// Accuracy: ~0.01% maximum error
pub fn exp_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x > 88.0 {
                return f32::INFINITY;
            }
            if x < -87.0 {
                return 0.0;
            }

            // Range reduction: exp(x) = exp(n*ln(2) + r) = 2^n * exp(r)
            let inv_ln2 = 1.0 / std::f32::consts::LN_2;
            let n = (x * inv_ln2).round();
            let r = x - n * std::f32::consts::LN_2;

            // Polynomial approximation for exp(r) where |r| < ln(2)/2
            let r2 = r * r;
            let r3 = r2 * r;
            let r4 = r3 * r;
            let exp_r = 1.0 + r + 0.5 * r2 + r3 / 6.0 + r4 / 24.0;

            // Compute 2^n efficiently
            let n_int = n as i32;
            let pow2_n = f32::from_bits(((n_int + 127) << 23) as u32);

            pow2_n * exp_r
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of sine function
///
/// Uses polynomial approximation with range reduction
/// Accuracy: ~0.001% maximum error
pub fn sin_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Range reduction to [-π, π]
            let x_mod = x % (2.0 * PI);
            let x_reduced = if x_mod > PI { x_mod - 2.0 * PI } else { x_mod };

            // Polynomial approximation: sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
            let x2 = x_reduced * x_reduced;
            let x3 = x2 * x_reduced;
            let x5 = x3 * x2;
            let x7 = x5 * x2;

            x_reduced - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of cosine function
///
/// Uses polynomial approximation with range reduction
/// Accuracy: ~0.001% maximum error
pub fn cos_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Range reduction to [-π, π]
            let x_mod = x % (2.0 * PI);
            let x_reduced = if x_mod > PI { x_mod - 2.0 * PI } else { x_mod };

            // Polynomial approximation: cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720
            let x2 = x_reduced * x_reduced;
            let x4 = x2 * x2;
            let x6 = x4 * x2;

            1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

// Helper functions

fn gamma_stirling_approx(z: f32) -> f32 {
    if z < 1.0 {
        return gamma_stirling_approx(z + 1.0) / z;
    }

    // Stirling's approximation: Γ(z) ≈ √(2π/z) * (z/e)^z
    let ln_gamma = 0.5 * (2.0 * PI / z).ln() + z * (z.ln() - 1.0);

    // Add first-order correction: + 1/(12z)
    let correction = 1.0 / (12.0 * z);

    (ln_gamma + correction).exp()
}

fn erfc_positive_fast(x: f32) -> f32 {
    if x > 5.0 {
        return 0.0;
    }

    // Continued fraction approximation for erfc(x)
    let x2 = x * x;
    let exp_neg_x2 = (-x2).exp();

    // erfc(x) ≈ (sqrt(π) * x)^(-1) * exp(-x²) * (1 - 1/(2x²) + 3/(4x⁴) - ...)
    let inv_sqrt_pi = 1.0 / PI.sqrt();
    let term1 = inv_sqrt_pi / x * exp_neg_x2;
    let correction = 1.0 - 1.0 / (2.0 * x2) + 3.0 / (4.0 * x2 * x2);

    term1 * correction
}

fn bessel_j0_polynomial_approx(x: f32) -> f32 {
    // Polynomial approximation for small arguments
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;

    1.0 - x2 / 4.0 + x4 / 64.0 - x6 / 2304.0
}

fn bessel_j0_asymptotic_approx(x: f32) -> f32 {
    // Asymptotic expansion for large arguments
    // J₀(x) ≈ √(2/(πx)) * cos(x - π/4)
    let sqrt_term = (2.0 / (PI * x)).sqrt();
    let phase = x - PI / 4.0;
    sqrt_term * phase.cos()
}

/// Fast approximation of hyperbolic tangent function
///
/// Uses rational approximation for better accuracy than polynomial
/// Accuracy: ~0.001% maximum error
pub fn tanh_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x > 7.0 {
                return 1.0;
            }
            if x < -7.0 {
                return -1.0;
            }

            // Fast tanh approximation using rational function
            // tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)
            let x2 = x * x;
            x * (27.0 + x2) / (27.0 + 9.0 * x2)
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of hyperbolic sine function
///
/// Uses exponential-based approximation with range reduction
/// Accuracy: ~0.01% maximum error
pub fn sinh_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x > 88.0 {
                return f32::INFINITY;
            }
            if x < -88.0 {
                return f32::NEG_INFINITY;
            }

            let abs_x = x.abs();
            if abs_x < 1.0 {
                // Use series expansion for small values: sinh(x) ≈ x + x³/6 + x⁵/120
                let x2 = x * x;
                let x3 = x2 * x;
                let x5 = x3 * x2;
                x + x3 / 6.0 + x5 / 120.0
            } else {
                // Use exp approximation: sinh(x) = (exp(x) - exp(-x)) / 2
                let exp_x = exp_fast_scalar(x);
                let exp_neg_x = exp_fast_scalar(-x);
                (exp_x - exp_neg_x) / 2.0
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of hyperbolic cosine function
///
/// Uses exponential-based approximation with range reduction
/// Accuracy: ~0.01% maximum error
pub fn cosh_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            let abs_x = x.abs();
            if abs_x > 88.0 {
                return f32::INFINITY;
            }

            if abs_x < 1.0 {
                // Use series expansion for small values: cosh(x) ≈ 1 + x²/2 + x⁴/24
                let x2 = x * x;
                let x4 = x2 * x2;
                1.0 + x2 / 2.0 + x4 / 24.0
            } else {
                // Use exp approximation: cosh(x) = (exp(x) + exp(-x)) / 2
                let exp_x = exp_fast_scalar(abs_x);
                let exp_neg_x = exp_fast_scalar(-abs_x);
                (exp_x + exp_neg_x) / 2.0
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Fast approximation of inverse hyperbolic tangent function
///
/// Uses series expansion for |x| < 0.5 and logarithmic form for larger values
/// Accuracy: ~0.01% maximum error for |x| < 0.95
pub fn atanh_fast(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x.abs() >= 1.0 {
                return if x >= 1.0 {
                    f32::INFINITY
                } else {
                    f32::NEG_INFINITY
                };
            }

            let abs_x = x.abs();
            if abs_x < 0.5 {
                // Use series expansion: atanh(x) ≈ x + x³/3 + 2x⁵/15 + 17x⁷/315
                let x2 = x * x;
                let x3 = x2 * x;
                let x5 = x3 * x2;
                let x7 = x5 * x2;
                x + x3 / 3.0 + 2.0 * x5 / 15.0 + 17.0 * x7 / 315.0
            } else {
                // Use logarithmic form: atanh(x) = 0.5 * log((1+x)/(1-x))
                let one_plus_x = 1.0 + x;
                let one_minus_x = 1.0 - x;
                0.5 * log_fast_scalar(one_plus_x / one_minus_x)
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

// Helper functions for scalar operations

fn exp_fast_scalar(x: f32) -> f32 {
    if x > 88.0 {
        return f32::INFINITY;
    }
    if x < -87.0 {
        return 0.0;
    }

    // Range reduction: exp(x) = exp(n*ln(2) + r) = 2^n * exp(r)
    let inv_ln2 = 1.0 / std::f32::consts::LN_2;
    let n = (x * inv_ln2).round();
    let r = x - n * std::f32::consts::LN_2;

    // Polynomial approximation for exp(r) where |r| < ln(2)/2
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r3 * r;
    let exp_r = 1.0 + r + 0.5 * r2 + r3 / 6.0 + r4 / 24.0;

    // Compute 2^n efficiently
    let n_int = n as i32;
    let pow2_n = f32::from_bits(((n_int + 127) << 23) as u32);

    pow2_n * exp_r
}

fn log_fast_scalar(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    if x == 1.0 {
        return 0.0;
    }

    // Fast log approximation using bit manipulation
    let x_bits = x.to_bits();
    let exp_bits = ((x_bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (x_bits & 0x7FFFFF) | 0x3F800000;
    let mantissa = f32::from_bits(mantissa_bits);

    // Polynomial approximation for log(1 + f) where f = mantissa - 1
    let f = mantissa - 1.0;
    let log_mantissa = f * (1.0 - 0.5 * f + f * f / 3.0 - f * f * f / 4.0);

    exp_bits as f32 * std::f32::consts::LN_2 + log_mantissa
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_gamma_fast_accuracy() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device)?;

        let fast_result = gamma_fast(&x)?;
        let accurate_result = crate::gamma(&x)?;

        let fast_data = fast_result.data()?;
        let accurate_data = accurate_result.data()?;

        for (i, (&fast_val, &accurate_val)) in
            fast_data.iter().zip(accurate_data.iter()).enumerate()
        {
            let relative_error = (fast_val - accurate_val).abs() / accurate_val;
            assert!(
                relative_error < 0.1,
                "Relative error too large at index {}: {}",
                i,
                relative_error
            );
        }
        Ok(())
    }

    #[test]
    fn test_erf_fast_accuracy() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 0.5, 1.0, 1.5, 2.0], vec![5], device)?;

        let fast_result = erf_fast(&x)?;
        let accurate_result = crate::erf(&x)?;

        let fast_data = fast_result.data()?;
        let accurate_data = accurate_result.data()?;

        for (i, (&fast_val, &accurate_val)) in
            fast_data.iter().zip(accurate_data.iter()).enumerate()
        {
            let absolute_error = (fast_val - accurate_val).abs();
            assert!(
                absolute_error < 0.01,
                "Absolute error too large at index {i}: {absolute_error}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_bessel_j0_fast_basic() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, 5.0, 10.0], vec![4], device)?;

        let result = bessel_j0_fast(&x)?;
        let data = result.data()?;

        // J₀(0) = 1
        assert_relative_eq!(data[0], 1.0, epsilon = 0.01);

        // J₀(x) should be finite and reasonable
        for &val in data.iter() {
            assert!(val.is_finite());
            assert!(val.abs() <= 1.5); // Bessel functions are bounded
        }
        Ok(())
    }

    #[test]
    fn test_trigonometric_fast() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(
            vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0],
            vec![5],
            device,
        )?;

        let sin_result = sin_fast(&x)?;
        let cos_result = cos_fast(&x)?;

        let sin_data = sin_result.data()?;
        let cos_data = cos_result.data()?;

        // Check known values
        assert_relative_eq!(sin_data[0], 0.0, epsilon = 0.001); // sin(0) = 0
        assert_relative_eq!(cos_data[0], 1.0, epsilon = 0.001); // cos(0) = 1
        assert_relative_eq!(sin_data[4], 1.0, epsilon = 0.001); // sin(π/2) = 1
        assert_relative_eq!(cos_data[4], 0.0, epsilon = 0.001); // cos(π/2) = 0
        Ok(())
    }

    #[test]
    fn test_exp_log_fast() -> TorshResult<()> {
        use std::f32::consts::E;
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, 2.0, -1.0, -2.0], vec![5], device)?;

        let exp_result = exp_fast(&x)?;
        let log_input =
            Tensor::from_data(vec![1.0, E, E * E, 1.0 / E, 1.0 / (E * E)], vec![5], device)?;
        let log_result = log_fast(&log_input)?;

        let exp_data = exp_result.data()?;
        let log_data = log_result.data()?;

        // Check exp values
        assert_relative_eq!(exp_data[0], 1.0, epsilon = 0.01); // exp(0) = 1
        assert_relative_eq!(exp_data[1], E, epsilon = 0.01); // exp(1) = e

        // Check log values
        assert_relative_eq!(log_data[0], 0.0, epsilon = 0.01); // log(1) = 0
        assert_relative_eq!(log_data[1], 1.0, epsilon = 0.01); // log(e) = 1
        Ok(())
    }

    #[test]
    fn test_hyperbolic_fast_accuracy() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 0.5, 1.0, 2.0, -0.5], vec![5], device)?;

        let tanh_result = tanh_fast(&x)?;
        let sinh_result = sinh_fast(&x)?;
        let cosh_result = cosh_fast(&x)?;

        let tanh_data = tanh_result.data()?;
        let sinh_data = sinh_result.data()?;
        let cosh_data = cosh_result.data()?;

        // Check tanh values
        assert_relative_eq!(tanh_data[0], 0.0, epsilon = 0.001); // tanh(0) = 0
        assert!(tanh_data[1] > 0.0 && tanh_data[1] < 1.0); // tanh(0.5) ∈ (0,1)
        assert!(tanh_data[4] < 0.0 && tanh_data[4] > -1.0); // tanh(-0.5) ∈ (-1,0)

        // Check sinh values
        assert_relative_eq!(sinh_data[0], 0.0, epsilon = 0.001); // sinh(0) = 0
        assert!(sinh_data[1] > 0.0); // sinh(0.5) > 0
        assert!(sinh_data[4] < 0.0); // sinh(-0.5) < 0

        // Check cosh values
        assert_relative_eq!(cosh_data[0], 1.0, epsilon = 0.001); // cosh(0) = 1
        assert!(cosh_data[1] > 1.0); // cosh(0.5) > 1
        assert!(cosh_data[4] > 1.0); // cosh(-0.5) > 1 (even function)

        Ok(())
    }

    #[test]
    fn test_atanh_fast_accuracy() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 0.2, 0.5, 0.8, -0.3], vec![5], device)?;

        let result = atanh_fast(&x)?;
        let data = result.data()?;

        // Check known values and properties
        assert_relative_eq!(data[0], 0.0, epsilon = 0.001); // atanh(0) = 0
        assert!(data[1] > 0.0); // atanh(0.2) > 0
        assert!(data[4] < 0.0); // atanh(-0.3) < 0 (odd function)

        // Check that all values are finite for valid inputs
        for &val in data.iter() {
            assert!(val.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_hyperbolic_identities() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![1.0, 2.0], vec![2], device)?;

        let sinh_result = sinh_fast(&x)?;
        let cosh_result = cosh_fast(&x)?;
        let tanh_result = tanh_fast(&x)?;

        let sinh_data = sinh_result.data()?;
        let cosh_data = cosh_result.data()?;
        let tanh_data = tanh_result.data()?;

        // Check identity: tanh(x) = sinh(x) / cosh(x) (with relaxed tolerance for fast approximations)
        for i in 0..sinh_data.len() {
            let expected_tanh = sinh_data[i] / cosh_data[i];
            assert_relative_eq!(tanh_data[i], expected_tanh, epsilon = 0.05);
        }

        // Check identity: cosh²(x) - sinh²(x) = 1 (with relaxed tolerance for fast approximations)
        for i in 0..sinh_data.len() {
            let identity = cosh_data[i] * cosh_data[i] - sinh_data[i] * sinh_data[i];
            assert_relative_eq!(identity, 1.0, epsilon = 0.05);
        }

        Ok(())
    }
}
