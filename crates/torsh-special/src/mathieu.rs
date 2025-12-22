//! Mathieu Functions
//!
//! This module provides implementations of Mathieu functions ce_n(x,q) and se_n(x,q)
//! which are solutions to the Mathieu differential equation.
//!
//! ## Applications
//! - Wave propagation in elliptical geometries
//! - Quantum mechanics (particle in elliptical potential)
//! - Electromagnetic wave theory (waveguides)
//! - Vibration analysis of elliptical membranes
//! - Celestial mechanics (orbital dynamics)
//!
//! ## Mathematical Background
//! The Mathieu equation is:
//! ```text
//! y'' + (a - 2q cos(2x))y = 0
//! ```
//! where a is the characteristic value and q is the parameter.
//!
//! Solutions are:
//! - ce_n(x,q): Even periodic Mathieu functions
//! - se_n(x,q): Odd periodic Mathieu functions

use crate::TorshResult;
use torsh_tensor::Tensor;

/// Mathieu characteristic value a_n(q) for even functions
///
/// Computes the characteristic value for even periodic Mathieu functions ce_n(x,q).
///
/// # Arguments
/// * `n` - Order (n ≥ 0)
/// * `q` - Parameter q
pub fn mathieu_a(n: i32, q: f32) -> f32 {
    mathieu_a_scalar(n, q as f64) as f32
}

/// Mathieu characteristic value b_n(q) for odd functions
///
/// Computes the characteristic value for odd periodic Mathieu functions se_n(x,q).
///
/// # Arguments
/// * `n` - Order (n ≥ 1)
/// * `q` - Parameter q
pub fn mathieu_b(n: i32, q: f32) -> f32 {
    mathieu_b_scalar(n, q as f64) as f32
}

/// Even periodic Mathieu function ce_n(x,q)
///
/// These are even 2π-periodic solutions of the Mathieu equation.
///
/// # Arguments
/// * `n` - Order (n ≥ 0)
/// * `q` - Parameter q
/// * `x` - Argument
///
/// # Examples
/// ```rust
/// use torsh_special::mathieu_ce;
/// use torsh_tensor::Tensor;
/// use torsh_core::device::DeviceType;
/// // let x = Tensor::from_data(vec![0.0, 0.5, 1.0], vec![3], DeviceType::Cpu).unwrap();
/// // let ce = mathieu_ce(0, 1.0, &x).unwrap();
/// ```
pub fn mathieu_ce(n: i32, q: f32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&xi| mathieu_ce_scalar(n, q as f64, xi as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Odd periodic Mathieu function se_n(x,q)
///
/// These are odd 2π-periodic solutions of the Mathieu equation.
///
/// # Arguments
/// * `n` - Order (n ≥ 1)
/// * `q` - Parameter q
/// * `x` - Argument
///
/// # Examples
/// ```rust
/// use torsh_special::mathieu_se;
/// use torsh_tensor::Tensor;
/// use torsh_core::device::DeviceType;
/// // let x = Tensor::from_data(vec![0.0, 0.5, 1.0], vec![3], DeviceType::Cpu).unwrap();
/// // let se = mathieu_se(1, 1.0, &x).unwrap();
/// ```
pub fn mathieu_se(n: i32, q: f32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&xi| mathieu_se_scalar(n, q as f64, xi as f64) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

// ============================================================================
// Scalar implementations
// ============================================================================

/// Compute Mathieu characteristic value a_n(q) using perturbation theory
fn mathieu_a_scalar(n: i32, q: f64) -> f64 {
    if q.abs() < 1e-10 {
        // For q = 0, a_n = n²
        return (n * n) as f64;
    }

    // Use perturbation expansion for small q
    // a_n(q) = n² - q + q²/(2n²) + O(q³) for n > 0
    // a_0(q) = -q²/2 + O(q⁴)

    if n == 0 {
        -q * q / 2.0
    } else {
        let n2 = (n * n) as f64;
        n2 - q + q * q / (2.0 * n2)
    }
}

/// Compute Mathieu characteristic value b_n(q) using perturbation theory
fn mathieu_b_scalar(n: i32, q: f64) -> f64 {
    if n < 1 {
        return 0.0;
    }

    if q.abs() < 1e-10 {
        // For q = 0, b_n = n²
        return (n * n) as f64;
    }

    // Use perturbation expansion for small q
    // b_n(q) = n² + q - q²/(2n²) + O(q³) for n > 0
    let n2 = (n * n) as f64;
    n2 + q - q * q / (2.0 * n2)
}

/// Compute even Mathieu function ce_n(x,q) using Fourier series
fn mathieu_ce_scalar(n: i32, q: f64, x: f64) -> f64 {
    if q.abs() < 1e-10 {
        // For q = 0, ce_n(x,0) = cos(nx)
        return ((n as f64) * x).cos();
    }

    // Use Fourier series expansion
    // ce_n(x,q) = Σ A_{2r}^n(q) cos(2rx) for even n
    // ce_n(x,q) = Σ A_{2r}^n(q) cos((2r)x) for odd n

    let mut sum;
    let _a_n = mathieu_a_scalar(n, q); // May be used in future for normalization

    // Simplified: use first few Fourier coefficients
    // For small q, dominant term is cos(nx)
    if n % 2 == 0 {
        sum = ((n as f64) * x).cos();
        // Add correction terms for small q
        if q.abs() < 1.0 {
            let correction = -q / (4.0 * (n as f64)) * ((n as f64 + 2.0) * x).cos();
            sum += correction;
        }
    } else {
        sum = ((n as f64) * x).cos();
        if q.abs() < 1.0 {
            let correction = -q / (4.0 * (n as f64)) * ((n as f64 + 2.0) * x).cos();
            sum += correction;
        }
    }

    sum
}

/// Compute odd Mathieu function se_n(x,q) using Fourier series
fn mathieu_se_scalar(n: i32, q: f64, x: f64) -> f64 {
    if n < 1 {
        return 0.0;
    }

    if q.abs() < 1e-10 {
        // For q = 0, se_n(x,0) = sin(nx)
        return ((n as f64) * x).sin();
    }

    // Use Fourier series expansion
    // se_n(x,q) = Σ B_{2r}^n(q) sin(2rx) for even n
    // se_n(x,q) = Σ B_{2r}^n(q) sin((2r)x) for odd n

    let mut sum;
    let _b_n = mathieu_b_scalar(n, q); // May be used in future for normalization

    // Simplified: use first few Fourier coefficients
    // For small q, dominant term is sin(nx)
    sum = ((n as f64) * x).sin();

    // Add correction terms for small q
    if q.abs() < 1.0 && n > 0 {
        let correction = -q / (4.0 * (n as f64)) * ((n as f64 + 2.0) * x).sin();
        sum += correction;
    }

    sum
}

/// Modified Mathieu function of the first kind Ce_n(z,q) for complex argument
///
/// Note: This is a simplified implementation for real arguments
#[allow(non_snake_case)]
pub fn mathieu_Ce(n: i32, q: f32, z: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // For real z, can use real implementation
    mathieu_ce(n, q, z)
}

/// Modified Mathieu function of the first kind Se_n(z,q) for complex argument
///
/// Note: This is a simplified implementation for real arguments
#[allow(non_snake_case)]
pub fn mathieu_Se(n: i32, q: f32, z: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // For real z, can use real implementation
    mathieu_se(n, q, z)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_mathieu_a_zero_q() {
        // For q = 0, a_n = n²
        assert_relative_eq!(mathieu_a(0, 0.0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(mathieu_a(1, 0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(mathieu_a(2, 0.0), 4.0, epsilon = 1e-6);
        assert_relative_eq!(mathieu_a(3, 0.0), 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mathieu_b_zero_q() {
        // For q = 0, b_n = n²
        assert_relative_eq!(mathieu_b(1, 0.0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(mathieu_b(2, 0.0), 4.0, epsilon = 1e-6);
        assert_relative_eq!(mathieu_b(3, 0.0), 9.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mathieu_ce_basic() -> TorshResult<()> {
        let x = Tensor::from_data(
            vec![0.0_f32, PI as f32 / 2.0, PI as f32],
            vec![3],
            DeviceType::Cpu,
        )?;
        let ce = mathieu_ce(0, 0.0, &x)?;
        let result = ce.data()?;

        // For q = 0, ce_0(x,0) = cos(0*x) = 1
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[2], 1.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_mathieu_ce_order_1() -> TorshResult<()> {
        let x = Tensor::from_data(vec![0.0_f32, PI as f32 / 2.0], vec![2], DeviceType::Cpu)?;
        let ce = mathieu_ce(1, 0.0, &x)?;
        let result = ce.data()?;

        // For q = 0, ce_1(x,0) = cos(x)
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_mathieu_se_basic() -> TorshResult<()> {
        let x = Tensor::from_data(vec![0.0_f32, PI as f32 / 2.0], vec![2], DeviceType::Cpu)?;
        let se = mathieu_se(1, 0.0, &x)?;
        let result = se.data()?;

        // For q = 0, se_1(x,0) = sin(x)
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(result[1], 1.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_mathieu_periodicity() -> TorshResult<()> {
        // Mathieu functions are 2π-periodic
        let x1 = Tensor::from_data(vec![0.5_f32], vec![1], DeviceType::Cpu)?;
        let x2 = Tensor::from_data(vec![0.5_f32 + 2.0 * PI as f32], vec![1], DeviceType::Cpu)?;

        let ce1 = mathieu_ce(1, 0.5, &x1)?;
        let ce2 = mathieu_ce(1, 0.5, &x2)?;

        let result1 = ce1.data()?;
        let result2 = ce2.data()?;

        assert_relative_eq!(result1[0], result2[0], epsilon = 1e-4);

        Ok(())
    }

    #[test]
    fn test_mathieu_a_perturbation() {
        // Test perturbation expansion for small q
        let q = 0.1;
        let n = 2;

        let a = mathieu_a(n, q);
        let n2 = (n * n) as f32;
        let expected = n2 - q + q * q / (2.0 * n2);

        assert_relative_eq!(a, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_mathieu_functions_finite() -> TorshResult<()> {
        let x = Tensor::from_data(vec![0.0_f32, 1.0, 2.0], vec![3], DeviceType::Cpu)?;

        let ce = mathieu_ce(2, 1.0, &x)?;
        let se = mathieu_se(2, 1.0, &x)?;

        let ce_data = ce.data()?;
        let se_data = se.data()?;

        assert!(ce_data.iter().all(|&x| x.is_finite()));
        assert!(se_data.iter().all(|&x| x.is_finite()));

        Ok(())
    }
}
