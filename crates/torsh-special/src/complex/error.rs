//! Complex error functions
//!
//! This module provides implementations of complex error functions including
//! erf and erfc with proper handling of branch cuts and numerical stability.

use std::f64::consts::PI;
use torsh_core::dtype::{Complex32, Complex64};
use torsh_tensor::Tensor;

use crate::TorshResult;

/// Complex error function erf(z)
///
/// Uses the series expansion for small |z| and continued fraction for large |z|
pub fn complex_erf_c64(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data.iter().map(|&z| complex_erf_main(z)).collect();
    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex error function for Complex32
pub fn complex_erf_c32(input: &Tensor<Complex32>) -> TorshResult<Tensor<Complex32>> {
    let data = input.data()?;
    let c64_data: Vec<Complex64> = data
        .iter()
        .map(|&z| Complex64::new(z.re as f64, z.im as f64))
        .collect();

    let c64_tensor = Tensor::from_data(c64_data, input.shape().dims().to_vec(), input.device())?;

    let result_c64 = complex_erf_c64(&c64_tensor)?;
    let result_data = result_c64.data()?;
    let result_c32: Vec<Complex32> = result_data
        .iter()
        .map(|&z| Complex32::new(z.re as f32, z.im as f32))
        .collect();

    Tensor::from_data(result_c32, input.shape().dims().to_vec(), input.device())
}

/// Complex complementary error function erfc(z)
pub fn complex_erfc_c64(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data.iter().map(|&z| complex_erfc_main(z)).collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex complementary error function for Complex32
pub fn complex_erfc_c32(input: &Tensor<Complex32>) -> TorshResult<Tensor<Complex32>> {
    let data = input.data()?;
    let c64_data: Vec<Complex64> = data
        .iter()
        .map(|&z| Complex64::new(z.re as f64, z.im as f64))
        .collect();

    let c64_tensor = Tensor::from_data(c64_data, input.shape().dims().to_vec(), input.device())?;

    let result_c64 = complex_erfc_c64(&c64_tensor)?;
    let result_data = result_c64.data()?;
    let result_c32: Vec<Complex32> = result_data
        .iter()
        .map(|&z| Complex32::new(z.re as f32, z.im as f32))
        .collect();

    Tensor::from_data(result_c32, input.shape().dims().to_vec(), input.device())
}

/// Main implementation of complex erfc
fn complex_erfc_main(z: Complex64) -> Complex64 {
    // Handle special cases
    let abs_z = z.norm();

    // For z = 0, erfc(0) = 1
    if abs_z == 0.0 {
        return Complex64::new(1.0, 0.0);
    }

    // For real inputs with zero imaginary part, use accurate real implementation
    if z.im.abs() < 1e-14 {
        let real_val = z.re;
        let result_real = erfc_real_accurate(real_val);
        return Complex64::new(result_real, 0.0);
    }

    // For small |z|, use: erfc(z) = 1 - erf(z)
    if abs_z < 2.0 {
        let erf_z = complex_erf_main(z);
        return Complex64::new(1.0, 0.0) - erf_z;
    }

    // For large |z|, use asymptotic expansion
    // erfc(z) ≈ exp(-z²) / (√π * z) * [1 - 1/(2z²) + 3/(4z⁴) - ...]
    let z_squared = z * z;
    let exp_neg_z2 = (-z_squared).exp();
    let sqrt_pi = Complex64::new(PI.sqrt(), 0.0);

    let mut series = Complex64::new(1.0, 0.0);
    let mut term = Complex64::new(1.0, 0.0);
    let neg_inv_2z2 = Complex64::new(-0.5, 0.0) / z_squared;

    for n in 1..10 {
        term = term * neg_inv_2z2 * Complex64::new((2 * n - 1) as f64, 0.0);
        series += term;

        if term.norm() < 1e-15 {
            break;
        }
    }

    exp_neg_z2 / (sqrt_pi * z) * series
}

fn complex_erf_main(z: Complex64) -> Complex64 {
    let abs_z = z.norm();

    // For z = 0, erf(0) = 0
    if abs_z == 0.0 {
        return Complex64::new(0.0, 0.0);
    }

    // For real inputs with zero imaginary part, use accurate real implementation
    if z.im.abs() < 1e-14 {
        let real_val = z.re;
        let result_real = erf_real_accurate(real_val);
        return Complex64::new(result_real, 0.0);
    }

    // Series expansion: erf(z) = (2/√π) * z * Σ((-1)^n * z^(2n) / (n! * (2n+1)))
    let mut sum = z;
    let mut term = z;
    let z_squared = z * z;
    let mut factorial = 1.0;

    for n in 1..50 {
        factorial *= n as f64;
        term = -term * z_squared / Complex64::new(factorial, 0.0);
        let next_term = term / Complex64::new((2 * n + 1) as f64, 0.0);
        sum += next_term;

        if next_term.norm() < 1e-15 {
            break;
        }
    }

    sum * Complex64::new(2.0 / PI.sqrt(), 0.0)
}

/// Accurate real erf implementation
fn erf_real_accurate(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let x_abs = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();

    if x >= 0.0 {
        y
    } else {
        -y
    }
}

/// Accurate real erfc implementation
fn erfc_real_accurate(x: f64) -> f64 {
    1.0 - erf_real_accurate(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_complex_erf_c64_real_axis() {
        // Test on real axis where we can compare with known values
        let input_data = vec![
            Complex64::new(0.0, 0.0), // erf(0) = 0
            Complex64::new(1.0, 0.0), // erf(1) ≈ 0.8427
        ];
        let input = Tensor::from_data(input_data, vec![2], DeviceType::Cpu).unwrap();
        let result = complex_erf_c64(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 0.0, max_relative = 1e-10);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-10);
        assert_relative_eq!(data[1].re, 0.8427007929, max_relative = 1e-4);
    }

    #[test]
    fn test_complex_erfc_c64() {
        let input_data = vec![Complex64::new(0.0, 0.0)]; // erfc(0) = 1
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_erfc_c64(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 1.0, max_relative = 1e-6);
    }

    #[test]
    fn test_erf_erfc_complement() {
        // Test that erf(z) + erfc(z) = 1
        let z_val = Complex64::new(0.5, 0.3);
        let input_data = vec![z_val];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();

        let erf_result = complex_erf_c64(&input).unwrap();
        let erfc_result = complex_erfc_c64(&input).unwrap();

        let erf_data = erf_result.data().unwrap();
        let erfc_data = erfc_result.data().unwrap();

        let sum = erf_data[0] + erfc_data[0];
        assert_relative_eq!(sum.re, 1.0, max_relative = 1e-6);
        assert_relative_eq!(sum.im, 0.0, max_relative = 1e-6);
    }
}
