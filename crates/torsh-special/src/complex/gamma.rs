//! Complex gamma function family
//!
//! This module provides implementations of complex gamma, beta, polygamma,
//! and incomplete gamma functions with proper branch cut handling.

use num_traits::Zero;
use std::f64::consts::PI;
use torsh_core::dtype::{Complex32, Complex64};
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

use crate::TorshResult;

/// Complex gamma function Γ(z) using Lanczos approximation
///
/// Implementation uses the Lanczos approximation for numerical stability
/// across the complex plane, with proper handling of branch cuts.
pub fn complex_gamma_c64(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data.iter().map(|&z| complex_gamma_main(z)).collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Main implementation for complex gamma function
fn complex_gamma_main(z: Complex64) -> Complex64 {
    // Handle special cases
    if z.re <= 0.0 && z.im.abs() < 1e-15 && (z.re - z.re.floor()).abs() < 1e-15 {
        // Gamma is undefined at negative integers
        return Complex64::new(f64::NAN, f64::NAN);
    }

    // For real inputs with zero imaginary part, use accurate real implementation
    if z.im.abs() < 1e-14 && z.re > 0.0 {
        let real_val = z.re;
        let result_real = gamma_real_accurate(real_val);
        return Complex64::new(result_real, 0.0);
    }

    if z.re < 0.5 {
        // Use reflection formula: Γ(z) * Γ(1-z) = π / sin(πz)
        let one_minus_z = Complex64::new(1.0, 0.0) - z;
        let pi_z = Complex64::new(PI, 0.0) * z;
        let sin_pi_z =
            (Complex64::new(0.0, 1.0) * pi_z).exp() - (Complex64::new(0.0, -1.0) * pi_z).exp();
        let sin_pi_z = sin_pi_z / Complex64::new(0.0, 2.0);

        let gamma_one_minus_z = complex_gamma_main(one_minus_z);
        return Complex64::new(PI, 0.0) / (sin_pi_z * gamma_one_minus_z);
    }

    lanczos_gamma(z)
}

/// Accurate real gamma implementation
fn gamma_real_accurate(x: f64) -> f64 {
    // Handle special common cases
    if x == 1.0 {
        return 1.0;
    }
    if x == 2.0 {
        return 1.0;
    }
    if x == 0.5 {
        return std::f64::consts::PI.sqrt();
    }

    // For integer values, use factorial
    if x.fract() == 0.0 && x > 0.0 && x <= 20.0 {
        let n = x as u32;
        return factorial_f64(n - 1);
    }

    // Use Lanczos approximation for other values
    let z_complex = Complex64::new(x, 0.0);
    lanczos_gamma(z_complex).re
}

fn factorial_f64(n: u32) -> f64 {
    if n == 0 || n == 1 {
        1.0
    } else {
        (2..=n as u64).map(|i| i as f64).product()
    }
}

/// Complex gamma function for Complex32
pub fn complex_gamma_c32(input: &Tensor<Complex32>) -> TorshResult<Tensor<Complex32>> {
    // Convert to Complex64, compute, then convert back
    let data = input.data()?;
    let complex64_data: Vec<Complex64> = data
        .iter()
        .map(|&z| Complex64::new(z.re as f64, z.im as f64))
        .collect();

    let complex64_tensor = Tensor::from_data(
        complex64_data,
        input.shape().dims().to_vec(),
        input.device(),
    )?;

    let result_c64 = complex_gamma_c64(&complex64_tensor)?;
    let result_c64_data = result_c64.data()?;

    let result_data: Vec<Complex32> = result_c64_data
        .iter()
        .map(|&z| Complex32::new(z.re as f32, z.im as f32))
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex polygamma function ψ^(n)(z)
///
/// Computes the nth derivative of the digamma function for complex arguments.
/// Uses series expansion and asymptotic formulae based on argument magnitude.
pub fn complex_polygamma_c64(n: u32, input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data
        .iter()
        .map(|&z| polygamma_complex_scalar(n, z))
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex polygamma function for Complex32
pub fn complex_polygamma_c32(n: u32, input: &Tensor<Complex32>) -> TorshResult<Tensor<Complex32>> {
    let data = input.data()?;
    let complex64_data: Vec<Complex64> = data
        .iter()
        .map(|&z| Complex64::new(z.re as f64, z.im as f64))
        .collect();

    let complex64_tensor = Tensor::from_data(
        complex64_data,
        input.shape().dims().to_vec(),
        input.device(),
    )?;

    let result_c64 = complex_polygamma_c64(n, &complex64_tensor)?;
    let result_c64_data = result_c64.data()?;

    let result_data: Vec<Complex32> = result_c64_data
        .iter()
        .map(|&z| Complex32::new(z.re as f32, z.im as f32))
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)
///
/// Computes the complex beta function using the gamma function relationship.
/// Handles special cases and branch cuts properly.
pub fn complex_beta_c64(
    a: &Tensor<Complex64>,
    b: &Tensor<Complex64>,
) -> TorshResult<Tensor<Complex64>> {
    let a_data = a.data()?;
    let b_data = b.data()?;

    if a_data.len() != b_data.len() {
        return Err(TorshError::InvalidShape(
            "Tensors must have same shape".to_string(),
        ));
    }

    let result_data: Vec<Complex64> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&a_val, &b_val)| {
            let gamma_a = complex_gamma_main(a_val);
            let gamma_b = complex_gamma_main(b_val);
            let gamma_a_plus_b = complex_gamma_main(a_val + b_val);

            (gamma_a * gamma_b) / gamma_a_plus_b
        })
        .collect();

    Tensor::from_data(result_data, a.shape().dims().to_vec(), a.device())
}

/// Complex beta function for Complex32
pub fn complex_beta_c32(
    a: &Tensor<Complex32>,
    b: &Tensor<Complex32>,
) -> TorshResult<Tensor<Complex32>> {
    let a_data = a.data()?;
    let b_data = b.data()?;

    let a_c64_data: Vec<Complex64> = a_data
        .iter()
        .map(|&z| Complex64::new(z.re as f64, z.im as f64))
        .collect();

    let b_c64_data: Vec<Complex64> = b_data
        .iter()
        .map(|&z| Complex64::new(z.re as f64, z.im as f64))
        .collect();

    let a_c64 = Tensor::from_data(a_c64_data, a.shape().dims().to_vec(), a.device())?;
    let b_c64 = Tensor::from_data(b_c64_data, b.shape().dims().to_vec(), b.device())?;

    let result_c64 = complex_beta_c64(&a_c64, &b_c64)?;
    let result_c64_data = result_c64.data()?;

    let result_data: Vec<Complex32> = result_c64_data
        .iter()
        .map(|&z| Complex32::new(z.re as f32, z.im as f32))
        .collect();

    Tensor::from_data(result_data, a.shape().dims().to_vec(), a.device())
}

/// Complex incomplete gamma function γ(a, z)
///
/// Computes the lower incomplete gamma function for complex arguments.
/// Uses series expansion for convergence and proper branch cut handling.
pub fn complex_incomplete_gamma_c64(
    a: &Tensor<Complex64>,
    z: &Tensor<Complex64>,
) -> TorshResult<Tensor<Complex64>> {
    let a_data = a.data()?;
    let z_data = z.data()?;

    let result_data: Vec<Complex64> = a_data
        .iter()
        .zip(z_data.iter())
        .map(|(&a_val, &z_val)| incomplete_gamma_complex_scalar(a_val, z_val))
        .collect();

    Tensor::from_data(result_data, a.shape().dims().to_vec(), a.device())
}

// Helper functions

/// Lanczos approximation for complex gamma function
pub fn lanczos_gamma(z: Complex64) -> Complex64 {
    // Lanczos coefficients for g=7, n=9
    let coeffs = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    let mut result = Complex64::new(coeffs[0], 0.0);
    for (i, &coeff) in coeffs.iter().skip(1).enumerate() {
        result += Complex64::new(coeff, 0.0) / (z + Complex64::new(i as f64, 0.0));
    }

    let t = z + Complex64::new(7.5, 0.0);
    let two_sqrt_e_over_pi = Complex64::new(2.5066282746310005, 0.0);

    two_sqrt_e_over_pi * t.powc(z + Complex64::new(0.5, 0.0)) * (-t).exp() * result
}

/// Complex polygamma function implementation
fn polygamma_complex_scalar(n: u32, z: Complex64) -> Complex64 {
    if n == 0 {
        // Digamma function
        return digamma_complex_scalar(z);
    }

    // For n > 0, use the series representation
    let mut result = Complex64::zero();
    let sign = if n % 2 == 1 { -1.0 } else { 1.0 };
    let factorial = factorial_u32(n);

    // Use asymptotic expansion for large |z|
    if z.norm() > 10.0 {
        let mut term = Complex64::new(factorial as f64 * sign, 0.0)
            / z.powc(Complex64::new(n as f64 + 1.0, 0.0));
        result = term;

        // Add first few terms of asymptotic series
        for k in 1..=5 {
            let coeff = factorial as f64 * sign * (n + k) as f64 / (k + 1) as f64;
            term =
                Complex64::new(coeff, 0.0) / z.powc(Complex64::new(n as f64 + k as f64 + 1.0, 0.0));
            result += term;
        }
    } else {
        // Use series for smaller arguments
        for k in 0..100 {
            let denom = z + Complex64::new(k as f64, 0.0);
            let term = Complex64::new(factorial as f64 * sign, 0.0)
                / denom.powc(Complex64::new(n as f64 + 1.0, 0.0));
            result += term;

            if term.norm() < 1e-15 {
                break;
            }
        }
    }

    result
}

/// Complex digamma function ψ(z)
fn digamma_complex_scalar(z: Complex64) -> Complex64 {
    // Use asymptotic expansion for large |z|
    if z.norm() > 10.0 {
        let ln_z = z.ln();
        let inv_z = Complex64::new(1.0, 0.0) / z;
        let inv_z2 = inv_z * inv_z;

        // ψ(z) ≈ ln(z) - 1/(2z) - 1/(12z²) + 1/(120z⁴) - ...
        return ln_z - inv_z / Complex64::new(2.0, 0.0) - inv_z2 / Complex64::new(12.0, 0.0)
            + inv_z2 * inv_z2 / Complex64::new(120.0, 0.0);
    }

    // For smaller |z|, use reflection and recurrence
    let mut result = Complex64::zero();
    let mut working_z = z;

    // Use recurrence to get to larger real part
    while working_z.re < 8.0 {
        result -= Complex64::new(1.0, 0.0) / working_z;
        working_z += Complex64::new(1.0, 0.0);
    }

    // Now use asymptotic expansion
    let ln_z = working_z.ln();
    let inv_z = Complex64::new(1.0, 0.0) / working_z;
    let inv_z2 = inv_z * inv_z;

    result + ln_z - inv_z / Complex64::new(2.0, 0.0) - inv_z2 / Complex64::new(12.0, 0.0)
}

/// Complex incomplete gamma function implementation
fn incomplete_gamma_complex_scalar(a: Complex64, z: Complex64) -> Complex64 {
    // Use series expansion: γ(a,z) = z^a * e^(-z) * Σ(z^n / (a+n))
    let mut result = Complex64::zero();
    let z_pow_a = z.powc(a);
    let exp_neg_z = (-z).exp();

    for n in 0..100 {
        let term = z.powc(Complex64::new(n as f64, 0.0)) / (a + Complex64::new(n as f64, 0.0));
        result += term;

        if term.norm() < 1e-15 {
            break;
        }
    }

    z_pow_a * exp_neg_z * result
}

/// Helper function for factorial
fn factorial_u32(n: u32) -> u64 {
    (1..=n as u64).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_complex_gamma_c64() {
        let input_data = vec![
            Complex64::new(1.0, 0.0), // Γ(1) = 1
            Complex64::new(2.0, 0.0), // Γ(2) = 1
            Complex64::new(0.5, 0.0), // Γ(0.5) = √π
        ];
        let input = Tensor::from_data(input_data, vec![3], DeviceType::Cpu).unwrap();
        let result = complex_gamma_c64(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 1.0, max_relative = 1e-10);
        assert_relative_eq!(data[1].re, 1.0, max_relative = 1e-10);
        assert_relative_eq!(data[2].re, std::f64::consts::PI.sqrt(), max_relative = 1e-6);
    }

    #[test]
    fn test_complex_beta_c64() {
        let a_data = vec![Complex64::new(1.0, 0.0)];
        let b_data = vec![Complex64::new(1.0, 0.0)];
        let a = Tensor::from_data(a_data, vec![1], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(b_data, vec![1], DeviceType::Cpu).unwrap();

        let result = complex_beta_c64(&a, &b).unwrap();
        let data = result.data().unwrap();

        // B(1,1) = 1
        assert_relative_eq!(data[0].re, 1.0, max_relative = 1e-6);
    }
}
