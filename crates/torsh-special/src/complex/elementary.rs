//! Complex elementary functions
//!
//! This module provides implementations of complex elementary functions including
//! logarithm, square root, and power functions with proper branch cut handling.

use scirs2_core::numeric::Zero; // SciRS2 POLICY compliant
use torsh_core::dtype::Complex64;
use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

use crate::TorshResult;

/// Principal branch of complex logarithm
///
/// Branch cut along negative real axis, with the branch cut discontinuity
/// chosen to give -π < arg(log(z)) ≤ π.
pub fn complex_log_principal(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data
        .iter()
        .map(|&z| {
            if z.norm() == 0.0 {
                Complex64::new(f64::NEG_INFINITY, 0.0)
            } else {
                Complex64::new(z.norm().ln(), z.arg())
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Principal branch of complex square root
///
/// Branch cut along negative real axis, chosen so that
/// Re(sqrt(z)) ≥ 0 for all z.
pub fn complex_sqrt_principal(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data
        .iter()
        .map(|&z| {
            if z.norm() == 0.0 {
                Complex64::zero()
            } else {
                let r = z.norm().sqrt();
                let theta = z.arg() / 2.0;
                Complex64::new(r * theta.cos(), r * theta.sin())
            }
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Principal branch of complex power z^w
///
/// Computes z^w = exp(w * log(z)) using the principal branch of the logarithm.
/// Branch cuts follow those of complex logarithm.
pub fn complex_pow_principal(
    base: &Tensor<Complex64>,
    exponent: &Tensor<Complex64>,
) -> TorshResult<Tensor<Complex64>> {
    // z^w = exp(w * log(z))
    let log_base = complex_log_principal(base)?;
    let exponent_data = exponent.data()?;
    let log_base_data = log_base.data()?;

    if exponent_data.len() != log_base_data.len() {
        return Err(TorshError::InvalidShape(
            "Tensors must have same shape".to_string(),
        ));
    }

    let result_data: Vec<Complex64> = exponent_data
        .iter()
        .zip(log_base_data.iter())
        .map(|(&w, &log_z)| (w * log_z).exp())
        .collect();

    Tensor::from_data(result_data, base.shape().dims().to_vec(), base.device())
}

/// Complex exponential function
///
/// Computes exp(z) = e^z = e^(x+iy) = e^x * (cos(y) + i*sin(y))
/// This is entire (analytic everywhere) so no branch cuts are needed.
pub fn complex_exp(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data.iter().map(|&z| z.exp()).collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex sine function
///
/// Computes sin(z) = (e^(iz) - e^(-iz))/(2i)
/// This is entire (analytic everywhere) so no branch cuts are needed.
pub fn complex_sin(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data
        .iter()
        .map(|&z| {
            let i = Complex64::new(0.0, 1.0);
            let exp_iz = (i * z).exp();
            let exp_neg_iz = (-i * z).exp();
            (exp_iz - exp_neg_iz) / (2.0 * i)
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Complex cosine function
///
/// Computes cos(z) = (e^(iz) + e^(-iz))/2
/// This is entire (analytic everywhere) so no branch cuts are needed.
pub fn complex_cos(input: &Tensor<Complex64>) -> TorshResult<Tensor<Complex64>> {
    let data = input.data()?;
    let result_data: Vec<Complex64> = data
        .iter()
        .map(|&z| {
            let i = Complex64::new(0.0, 1.0);
            let exp_iz = (i * z).exp();
            let exp_neg_iz = (-i * z).exp();
            (exp_iz + exp_neg_iz) / Complex64::new(2.0, 0.0)
        })
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_complex_log_principal_real_positive() {
        // log(e) = 1
        let input_data = vec![Complex64::new(std::f64::consts::E, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_log_principal(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 1.0, max_relative = 1e-10);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_complex_log_principal_imaginary_unit() {
        // log(i) = iπ/2
        let input_data = vec![Complex64::new(0.0, 1.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_log_principal(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 0.0, max_relative = 1e-10);
        assert_relative_eq!(data[0].im, PI / 2.0, max_relative = 1e-10);
    }

    #[test]
    fn test_complex_sqrt_principal() {
        // sqrt(4) = 2
        let input_data = vec![Complex64::new(4.0, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_sqrt_principal(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 2.0, max_relative = 1e-10);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_complex_sqrt_principal_negative_real() {
        // sqrt(-1) = i
        let input_data = vec![Complex64::new(-1.0, 0.0)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_sqrt_principal(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 0.0, max_relative = 1e-10);
        assert_relative_eq!(data[0].im, 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_complex_pow_principal() {
        // 2^3 = 8
        let base_data = vec![Complex64::new(2.0, 0.0)];
        let exp_data = vec![Complex64::new(3.0, 0.0)];
        let base = Tensor::from_data(base_data, vec![1], DeviceType::Cpu).unwrap();
        let exponent = Tensor::from_data(exp_data, vec![1], DeviceType::Cpu).unwrap();

        let result = complex_pow_principal(&base, &exponent).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, 8.0, max_relative = 1e-10);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_complex_exp() {
        // exp(iπ) = -1 (Euler's identity)
        let input_data = vec![Complex64::new(0.0, PI)];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();
        let result = complex_exp(&input).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0].re, -1.0, max_relative = 1e-10);
        assert_relative_eq!(data[0].im, 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_complex_trigonometric_identity() {
        // sin²(z) + cos²(z) = 1
        let z = Complex64::new(0.5, 0.3);
        let input_data = vec![z];
        let input = Tensor::from_data(input_data, vec![1], DeviceType::Cpu).unwrap();

        let sin_result = complex_sin(&input).unwrap();
        let cos_result = complex_cos(&input).unwrap();

        let sin_data = sin_result.data().unwrap();
        let cos_data = cos_result.data().unwrap();

        let sin_squared = sin_data[0] * sin_data[0];
        let cos_squared = cos_data[0] * cos_data[0];
        let sum = sin_squared + cos_squared;

        assert_relative_eq!(sum.re, 1.0, max_relative = 1e-10);
        assert_relative_eq!(sum.im, 0.0, max_relative = 1e-10);
    }
}
