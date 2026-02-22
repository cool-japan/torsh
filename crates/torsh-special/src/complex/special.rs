//! Complex special functions
//!
//! This module provides implementations of complex Airy functions and
//! other special functions that don't fit in the main categories.

use std::f64::consts::PI;
use torsh_core::dtype::Complex64;
use torsh_tensor::Tensor;

use crate::TorshResult;

/// Complex Airy function Ai(z)
///
/// Computes the first solution to the Airy differential equation w'' - zw = 0.
/// Uses asymptotic expansions for large |z| and series expansion for moderate values.
pub fn complex_airy_ai_c64(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data.iter().map(|&z| complex_airy_ai_impl(z)).collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex Airy function Bi(z)
///
/// Computes the second solution to the Airy differential equation w'' - zw = 0.
/// Uses asymptotic expansions for large |z| and series expansion for moderate values.
pub fn complex_airy_bi_c64(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data.iter().map(|&z| complex_airy_bi_impl(z)).collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Implementation of complex Airy function Ai(z)
fn complex_airy_ai_impl(z: Complex64) -> Complex64 {
    // Use integral representation or series expansion
    // For large |z|, use asymptotic expansion

    if z.norm() > 5.0 {
        // Asymptotic expansion for large |z|
        let zeta = z * z.sqrt() * Complex64::new(2.0 / 3.0, 0.0);
        let factor = Complex64::new(1.0 / (2.0 * PI.sqrt()), 0.0) / z.sqrt().sqrt();

        if z.re > 0.0 {
            return factor * (-zeta).exp();
        } else {
            // For negative real part, use oscillatory form
            let cos_term = (zeta.re).cos();
            let sin_term = (zeta.re).sin();
            return factor * Complex64::new(cos_term, -sin_term);
        }
    }

    // Use series expansion for moderate values
    // Ai(z) approximated by truncated series
    let mut result = Complex64::new(0.355_028_053_887_817_2, 0.0); // Ai(0)
    let mut z_power = z;
    let mut factorial = 1.0;

    for n in 1..20 {
        if n % 3 == 1 {
            factorial *= (n as f64) * ((n + 1) as f64) * ((n + 2) as f64);
            let term = z_power / Complex64::new(factorial, 0.0);
            if n % 6 == 1 {
                result += term;
            } else {
                result -= term;
            }
        }
        z_power *= z;
    }

    result
}

/// Implementation of complex Airy function Bi(z)
fn complex_airy_bi_impl(z: Complex64) -> Complex64 {
    // Similar to Ai but with different normalization
    // Bi(z) = (1/π) * ∫₀^∞ [exp(-t³/3 + zt) + sin(t³/3 + zt)] dt

    if z.norm() > 5.0 {
        let zeta = z * z.sqrt() * Complex64::new(2.0 / 3.0, 0.0);
        let factor = Complex64::new(1.0 / PI.sqrt(), 0.0) / z.sqrt().sqrt();

        if z.re > 0.0 {
            return factor * zeta.exp();
        } else {
            let cos_term = (zeta.re).cos();
            let sin_term = (zeta.re).sin();
            return factor * Complex64::new(sin_term, cos_term);
        }
    }

    // Series expansion
    let mut result = Complex64::new(0.614_926_627_446_000_7, 0.0); // Bi(0)
    let mut z_power = z;
    let mut factorial = 1.0;

    for n in 1..20 {
        if n % 3 == 2 {
            factorial *= (n as f64) * ((n + 1) as f64) * ((n + 2) as f64);
            let term = z_power / Complex64::new(factorial, 0.0);
            result += term; // Bi series has same sign terms
        }
        z_power *= z;
    }

    // Handle special case for z = 0
    if z.norm() < 1e-14 {
        return Complex64::new(0.614_926_627_446_000_7, 0.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_complex_airy_ai_c64_at_origin() {
        // Ai(0) ≈ 0.35503
        let input_data = vec![Complex64::new(0.0, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_airy_ai_c64(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 0.355028, max_relative = 1e-4);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_complex_airy_bi_c64_at_origin() {
        // Bi(0) ≈ 0.61493
        let input_data = vec![Complex64::new(0.0, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_airy_bi_c64(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 0.61493, max_relative = 1e-3);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_complex_airy_functions_finite() {
        // Test that functions return finite values for various inputs
        let input_data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(-1.0, 0.0),
        ];
        let input = Tensor::from_data(input_data, vec![4], DeviceType::Cpu).unwrap();

        let ai_result = complex_airy_ai_c64(&input).unwrap();
        let bi_result = complex_airy_bi_c64(&input).unwrap();

        let ai_data = ai_result.data().unwrap();
        let bi_data = bi_result.data().unwrap();

        for i in 0..4 {
            assert!(ai_data[i].re.is_finite() && ai_data[i].im.is_finite());
            assert!(bi_data[i].re.is_finite() && bi_data[i].im.is_finite());
        }
    }
}
