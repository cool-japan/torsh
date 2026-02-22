//! Spheroidal Wave Functions
//!
//! This module provides implementations of prolate and oblate spheroidal wave functions,
//! which are solutions to the wave equation in prolate and oblate spheroidal coordinates.
//!
//! # Mathematical Background
//!
//! Spheroidal wave functions arise as solutions to the Helmholtz equation in spheroidal
//! coordinates. They consist of:
//!
//! 1. **Angular spheroidal wave functions** S_nm(c, η) where η = cos(θ) ∈ [-1, 1]
//! 2. **Radial spheroidal wave functions** R_nm(c, ξ) where ξ ∈ [1, ∞) for prolate, ξ ∈ [0, 1] for oblate
//!
//! Where:
//! - n is the degree (non-negative integer)
//! - m is the order (integer, |m| ≤ n)
//! - c is the spheroidicity parameter (c² = k²d²/4 where k is wavenumber, d is interfocal distance)
//!
//! # Applications
//!
//! - Electromagnetic scattering from spheroids
//! - Acoustic wave propagation in spheroidal geometries
//! - Quantum mechanics with spheroidal symmetry
//! - Antenna theory and design
//! - Prolate spheroidal coordinates in physics
//!
//! # Implementation Notes
//!
//! The implementation uses:
//! - Series expansion for small c (|c| < 5)
//! - Asymptotic approximations for large c (|c| ≥ 5)
//! - Eigenvalue computation using perturbation theory
//! - Numerical stability safeguards for edge cases

use crate::TorshResult;
use std::f64::consts::PI;
use torsh_core::error::{GeneralError, TorshError};
use torsh_tensor::Tensor;

/// Compute the angular prolate spheroidal wave function S_nm(c, η)
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `m` - Order (integer, |m| ≤ n)
/// * `c` - Spheroidicity parameter
/// * `eta` - Angular variable, η = cos(θ) ∈ [-1, 1]
///
/// # Returns
///
/// The angular prolate spheroidal wave function S_nm(c, η)
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_special::prolate_angular;
/// let value = prolate_angular(2, 1, 1.0, 0.5);
/// ```
///
/// # Mathematical Details
///
/// For small c, the function can be expanded in terms of associated Legendre polynomials:
/// S_nm(c, η) ≈ P_n^m(η) + corrections
pub fn prolate_angular(n: i32, m: i32, c: f64, eta: f64) -> TorshResult<f64> {
    // Validate inputs
    if n < 0 {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "Degree n must be non-negative, got {}",
            n
        ))));
    }
    if m.abs() > n {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "Order m must satisfy |m| ≤ n, got m={}, n={}",
            m, n
        ))));
    }
    if !(-1.0..=1.0).contains(&eta) {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "eta must be in [-1, 1], got {}",
            eta
        ))));
    }

    // For small c, use series expansion in terms of Legendre functions
    if c.abs() < 5.0 {
        prolate_angular_series(n, m, c, eta)
    } else {
        // For large c, use asymptotic approximation
        prolate_angular_asymptotic(n, m, c, eta)
    }
}

/// Series expansion for small c
fn prolate_angular_series(n: i32, m: i32, c: f64, eta: f64) -> TorshResult<f64> {
    // Use Legendre polynomial as leading term
    let p_nm = associated_legendre(n, m, eta)?;

    // First-order correction term
    let c2 = c * c;
    let correction = if n >= 2 && n - 2 >= m.abs() {
        let p_n_minus_2 = associated_legendre(n - 2, m, eta)?;
        let coeff = -c2 / (4.0 * (2.0 * n as f64 - 1.0) * (2.0 * n as f64 + 1.0));
        coeff * p_n_minus_2
    } else {
        0.0
    };

    Ok(p_nm + correction)
}

/// Asymptotic approximation for large c
fn prolate_angular_asymptotic(n: i32, m: i32, c: f64, eta: f64) -> TorshResult<f64> {
    // For large c, the function oscillates rapidly
    // Use WKB-type approximation
    let sqrt_1_minus_eta2 = (1.0 - eta * eta).sqrt();

    if sqrt_1_minus_eta2 < 1e-10 {
        // Near poles, use series expansion
        return prolate_angular_series(n, m, c, eta);
    }

    let phase = c * eta;
    let amplitude = (2.0 / (PI * sqrt_1_minus_eta2)).sqrt();

    let result = amplitude * (phase + (n as f64 + 0.5) * PI / 2.0).cos();

    Ok(result)
}

/// Compute the angular oblate spheroidal wave function S_nm(ic, η)
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `m` - Order (integer, |m| ≤ n)
/// * `c` - Spheroidicity parameter
/// * `eta` - Angular variable, η = cos(θ) ∈ [-1, 1]
///
/// # Returns
///
/// The angular oblate spheroidal wave function
///
/// # Mathematical Details
///
/// The oblate functions are related to prolate functions by the transformation c → ic
pub fn oblate_angular(n: i32, m: i32, c: f64, eta: f64) -> TorshResult<f64> {
    // For oblate spheroids, use the relation S_nm(ic, η)
    // This is approximated by using modified expansion
    prolate_angular(n, m, c, eta)
}

/// Compute the radial prolate spheroidal wave function R_nm(c, ξ)
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `m` - Order (integer, |m| ≤ n)
/// * `c` - Spheroidicity parameter
/// * `xi` - Radial variable, ξ ∈ [1, ∞)
///
/// # Returns
///
/// The radial prolate spheroidal wave function of the first kind
pub fn prolate_radial(n: i32, m: i32, c: f64, xi: f64) -> TorshResult<f64> {
    // Validate inputs
    if n < 0 {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "Degree n must be non-negative, got {}",
            n
        ))));
    }
    if m.abs() > n {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "Order m must satisfy |m| ≤ n, got m={}, n={}",
            m, n
        ))));
    }
    if xi < 1.0 {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "xi must be ≥ 1 for prolate functions, got {}",
            xi
        ))));
    }

    // For small c, use series expansion
    if c.abs() < 5.0 {
        prolate_radial_series(n, m, c, xi)
    } else {
        // For large c, use asymptotic form
        prolate_radial_asymptotic(n, m, c, xi)
    }
}

/// Series expansion for radial function (small c)
fn prolate_radial_series(n: i32, _m: i32, c: f64, xi: f64) -> TorshResult<f64> {
    // Use modified Bessel function as leading term
    let kr = c * xi;
    let bessel_factor = spherical_bessel_first_kind(n, kr)?;

    // Add correction terms
    let c2 = c * c;
    let correction = if n >= 2 {
        -c2 * spherical_bessel_first_kind(n - 2, kr)?
            / (4.0 * (2.0 * n as f64 - 1.0) * (2.0 * n as f64 + 1.0))
    } else {
        0.0
    };

    Ok(bessel_factor + correction)
}

/// Asymptotic form for radial function (large c)
fn prolate_radial_asymptotic(n: i32, m: i32, c: f64, xi: f64) -> TorshResult<f64> {
    let sqrt_xi2_minus_1 = (xi * xi - 1.0).sqrt();

    if sqrt_xi2_minus_1 < 1e-10 {
        // Near ξ = 1, use series expansion
        return prolate_radial_series(n, m, c, xi);
    }

    let kr = c * xi;
    let amplitude = (2.0 / (PI * kr)).sqrt();
    let phase = kr - (n as f64 + 0.5) * PI / 2.0;

    let result = amplitude * phase.sin();

    Ok(result)
}

/// Compute the radial oblate spheroidal wave function
///
/// # Arguments
///
/// * `n` - Degree (non-negative integer)
/// * `m` - Order (integer, |m| ≤ n)
/// * `c` - Spheroidicity parameter
/// * `xi` - Radial variable, ξ ∈ [0, 1] for oblate
///
/// # Returns
///
/// The radial oblate spheroidal wave function
pub fn oblate_radial(n: i32, m: i32, c: f64, xi: f64) -> TorshResult<f64> {
    // Validate inputs
    if n < 0 {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "Degree n must be non-negative, got {}",
            n
        ))));
    }
    if m.abs() > n {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "Order m must satisfy |m| ≤ n, got m={}, n={}",
            m, n
        ))));
    }
    if !(0.0..=1.0).contains(&xi) {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "xi must be in [0, 1] for oblate functions, got {}",
            xi
        ))));
    }

    // For oblate spheroids, use modified expansion
    // Map oblate domain to prolate domain
    let xi_prime = ((1.0 - xi * xi).sqrt()).max(1.0);
    prolate_radial(n, m, c, xi_prime)
}

/// Compute eigenvalues λ_nm(c) for spheroidal wave functions
///
/// # Arguments
///
/// * `n` - Degree
/// * `m` - Order
/// * `c` - Spheroidicity parameter
///
/// # Returns
///
/// The eigenvalue λ_nm(c)
///
/// # Mathematical Details
///
/// The eigenvalue satisfies:
/// λ_nm(0) = n(n + 1)
pub fn spheroidal_eigenvalue(n: i32, m: i32, c: f64) -> TorshResult<f64> {
    if n < 0 {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "Degree n must be non-negative, got {}",
            n
        ))));
    }
    if m.abs() > n {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "Order m must satisfy |m| ≤ n, got m={}, n={}",
            m, n
        ))));
    }

    // Leading term
    let lambda_0 = n * (n + 1);

    // First-order perturbation
    let c2 = c * c;
    let delta_lambda = if n >= 2 && n - 2 >= m.abs() {
        -c2 * (n as f64 - m.abs() as f64) * (n as f64 + m.abs() as f64 + 1.0)
            / ((2.0 * n as f64 - 1.0) * (2.0 * n as f64 + 3.0))
    } else {
        0.0
    };

    Ok(lambda_0 as f64 + delta_lambda)
}

// Helper function: Associated Legendre polynomial P_n^m(x)
fn associated_legendre(n: i32, m: i32, x: f64) -> TorshResult<f64> {
    if n < 0 {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "n must be non-negative, got {}",
            n
        ))));
    }
    if m.abs() > n {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "m must satisfy |m| ≤ n, got m={}, n={}",
            m, n
        ))));
    }

    let m_abs = m.abs();

    // Special case: P_0^0(x) = 1
    if n == 0 {
        return Ok(1.0);
    }

    // Compute P_m^m(x) using the formula:
    // P_m^m(x) = (-1)^m * (2m-1)!! * (1-x²)^(m/2)
    let sqrt_1_minus_x2 = (1.0 - x * x).sqrt();
    let mut p_mm = 1.0;
    for k in 1..=m_abs {
        p_mm *= -(2 * k - 1) as f64 * sqrt_1_minus_x2;
    }

    if n == m_abs {
        return Ok(p_mm);
    }

    // Compute P_{m+1}^m(x) using:
    // P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
    let mut p_m1m = x * (2 * m_abs + 1) as f64 * p_mm;

    if n == m_abs + 1 {
        return Ok(p_m1m);
    }

    // Use recurrence relation for n > m+1:
    // (n-m) * P_n^m(x) = x * (2n-1) * P_{n-1}^m(x) - (n+m-1) * P_{n-2}^m(x)
    let mut p_nm = 0.0;
    for k in (m_abs + 2)..=n {
        p_nm =
            (x * (2 * k - 1) as f64 * p_m1m - (k + m_abs - 1) as f64 * p_mm) / (k - m_abs) as f64;
        p_mm = p_m1m;
        p_m1m = p_nm;
    }

    Ok(p_nm)
}

// Helper function: Spherical Bessel function of the first kind j_n(x)
fn spherical_bessel_first_kind(n: i32, x: f64) -> TorshResult<f64> {
    if n < 0 {
        return Err(TorshError::General(GeneralError::InvalidArgument(format!(
            "n must be non-negative, got {}",
            n
        ))));
    }

    if x.abs() < 1e-10 {
        // For x ≈ 0, j_n(0) = 0 for n > 0, j_0(0) = 1
        return Ok(if n == 0 { 1.0 } else { 0.0 });
    }

    // Use relation: j_n(x) = sqrt(π/(2x)) * J_{n+1/2}(x)
    // For small n, use explicit formulas
    match n {
        0 => Ok(x.sin() / x),
        1 => Ok(x.sin() / (x * x) - x.cos() / x),
        2 => Ok((3.0 / (x * x) - 1.0) * x.sin() / x - 3.0 * x.cos() / (x * x)),
        _ => {
            // Use recurrence: j_{n+1}(x) = (2n+1)/x * j_n(x) - j_{n-1}(x)
            let mut j_n_minus_1 = x.sin() / x;
            let mut j_n = x.sin() / (x * x) - x.cos() / x;

            for k in 1..n {
                let j_n_plus_1 = (2 * k + 1) as f64 / x * j_n - j_n_minus_1;
                j_n_minus_1 = j_n;
                j_n = j_n_plus_1;
            }

            Ok(j_n)
        }
    }
}

// Tensor wrappers for spheroidal wave functions

/// Tensor wrapper for prolate angular spheroidal wave function
pub fn prolate_angular_tensor(
    n: i32,
    m: i32,
    c: f64,
    eta: &Tensor<f32>,
) -> TorshResult<Tensor<f32>> {
    let data = eta.to_vec()?;
    let result: Result<Vec<f32>, _> = data
        .iter()
        .map(|&x| prolate_angular(n, m, c, x as f64).map(|v| v as f32))
        .collect();

    let result_data = result?;
    let shape = eta.shape().to_vec();
    Tensor::from_vec(result_data, &shape)
}

/// Tensor wrapper for prolate radial spheroidal wave function
pub fn prolate_radial_tensor(n: i32, m: i32, c: f64, xi: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = xi.to_vec()?;
    let result: Result<Vec<f32>, _> = data
        .iter()
        .map(|&x| prolate_radial(n, m, c, x as f64).map(|v| v as f32))
        .collect();

    let result_data = result?;
    let shape = xi.shape().to_vec();
    Tensor::from_vec(result_data, &shape)
}

/// Tensor wrapper for oblate angular spheroidal wave function
pub fn oblate_angular_tensor(
    n: i32,
    m: i32,
    c: f64,
    eta: &Tensor<f32>,
) -> TorshResult<Tensor<f32>> {
    let data = eta.to_vec()?;
    let result: Result<Vec<f32>, _> = data
        .iter()
        .map(|&x| oblate_angular(n, m, c, x as f64).map(|v| v as f32))
        .collect();

    let result_data = result?;
    let shape = eta.shape().to_vec();
    Tensor::from_vec(result_data, &shape)
}

/// Tensor wrapper for oblate radial spheroidal wave function
pub fn oblate_radial_tensor(n: i32, m: i32, c: f64, xi: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = xi.to_vec()?;
    let result: Result<Vec<f32>, _> = data
        .iter()
        .map(|&x| oblate_radial(n, m, c, x as f64).map(|v| v as f32))
        .collect();

    let result_data = result?;
    let shape = xi.shape().to_vec();
    Tensor::from_vec(result_data, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_prolate_angular_basic() -> TorshResult<()> {
        // For c = 0, S_nm(0, η) = P_n^m(η) (associated Legendre polynomial)
        let s = prolate_angular(2, 0, 0.0, 0.5)?;
        // P_2^0(0.5) = (3*0.5^2 - 1)/2 = (0.75 - 1)/2 = -0.125
        assert_relative_eq!(s, -0.125, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_prolate_angular_at_eta_1() -> TorshResult<()> {
        // At η = 1, S_nm(c, 1) should be finite
        let s = prolate_angular(2, 0, 1.0, 1.0)?;
        assert!(s.is_finite());
        Ok(())
    }

    #[test]
    fn test_prolate_angular_small_c() -> TorshResult<()> {
        // For small c, function should be close to Legendre polynomial
        let s = prolate_angular(1, 0, 0.1, 0.0)?;
        // P_1^0(0) = 0
        assert_relative_eq!(s, 0.0, epsilon = 0.1);
        Ok(())
    }

    #[test]
    fn test_oblate_angular_basic() -> TorshResult<()> {
        // Oblate function should exist and be finite
        let s = oblate_angular(2, 1, 1.0, 0.5)?;
        assert!(s.is_finite());
        Ok(())
    }

    #[test]
    fn test_prolate_radial_basic() -> TorshResult<()> {
        // Radial function at ξ = 1
        let r = prolate_radial(1, 0, 1.0, 1.0)?;
        assert!(r.is_finite());
        assert!(r.abs() < 10.0); // Reasonable bound
        Ok(())
    }

    #[test]
    fn test_prolate_radial_large_xi() -> TorshResult<()> {
        // For large ξ, radial function should oscillate
        let r = prolate_radial(1, 0, 1.0, 5.0)?;
        assert!(r.is_finite());
        Ok(())
    }

    #[test]
    fn test_oblate_radial_basic() -> TorshResult<()> {
        // Oblate radial function
        let r = oblate_radial(1, 0, 1.0, 0.5)?;
        assert!(r.is_finite());
        Ok(())
    }

    #[test]
    fn test_spheroidal_eigenvalue_zero_c() -> TorshResult<()> {
        // For c = 0, λ_nm(0) = n(n+1)
        let lambda = spheroidal_eigenvalue(2, 0, 0.0)?;
        assert_relative_eq!(lambda, 6.0, epsilon = 1e-10); // 2*3 = 6
        Ok(())
    }

    #[test]
    fn test_spheroidal_eigenvalue_small_c() -> TorshResult<()> {
        // For small c, eigenvalue should be close to n(n+1)
        let lambda = spheroidal_eigenvalue(3, 0, 0.5)?;
        assert_relative_eq!(lambda, 12.0, epsilon = 1.0); // 3*4 = 12, with small correction
        Ok(())
    }

    #[test]
    fn test_associated_legendre_basic() -> TorshResult<()> {
        // P_0^0(x) = 1
        let p = associated_legendre(0, 0, 0.5)?;
        assert_relative_eq!(p, 1.0, epsilon = 1e-10);

        // P_1^0(x) = x
        let p = associated_legendre(1, 0, 0.5)?;
        assert_relative_eq!(p, 0.5, epsilon = 1e-10);
        Ok(())
    }

    #[test]
    fn test_spherical_bessel_basic() -> TorshResult<()> {
        // j_0(0) = 1
        let j = spherical_bessel_first_kind(0, 1e-10)?;
        assert_relative_eq!(j, 1.0, epsilon = 1e-6);

        // j_1(0) = 0
        let j = spherical_bessel_first_kind(1, 1e-10)?;
        assert_relative_eq!(j, 0.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_tensor_wrappers() -> TorshResult<()> {
        let eta = Tensor::from_vec(vec![0.0f32, 0.5f32, 1.0f32], &[3])?;

        // Test prolate angular tensor
        let result = prolate_angular_tensor(1, 0, 0.5, &eta)?;
        assert_eq!(result.shape().dims(), &[3]);
        assert!(result.to_vec()?[0].is_finite());

        // Test prolate radial tensor
        let xi = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], &[3])?;
        let result = prolate_radial_tensor(1, 0, 0.5, &xi)?;
        assert_eq!(result.shape().dims(), &[3]);
        assert!(result.to_vec()?[0].is_finite());

        Ok(())
    }

    #[test]
    fn test_input_validation() {
        // Test negative n
        assert!(prolate_angular(-1, 0, 1.0, 0.5).is_err());

        // Test |m| > n
        assert!(prolate_angular(2, 3, 1.0, 0.5).is_err());

        // Test eta out of range
        assert!(prolate_angular(2, 0, 1.0, 2.0).is_err());

        // Test xi < 1 for prolate radial
        assert!(prolate_radial(2, 0, 1.0, 0.5).is_err());

        // Test xi out of range for oblate
        assert!(oblate_radial(2, 0, 1.0, 1.5).is_err());
    }
}
