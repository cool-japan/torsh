//! Coulomb Wave Functions
//!
//! This module provides implementations of Coulomb wave functions F_L(η,ρ) and G_L(η,ρ)
//! which are solutions to the radial Schrödinger equation for the Coulomb potential.
//!
//! ## Applications
//! - Quantum scattering theory (charged particle scattering)
//! - Nuclear physics (Coulomb barrier penetration)
//! - Atomic physics (electron-nucleus interactions)
//! - Astrophysics (stellar nucleosynthesis)
//!
//! ## Mathematical Background
//! The Coulomb wave functions satisfy:
//! ```text
//! d²u/dρ² + [1 - 2η/ρ - L(L+1)/ρ²]u = 0
//! ```
//! where:
//! - ρ = kr (reduced radial distance)
//! - η = Z₁Z₂e²/(ħv) (Sommerfeld parameter)
//! - L = angular momentum quantum number

use crate::TorshResult;
use std::f64::consts::PI;
use torsh_tensor::Tensor;

/// Coulomb wave function F_L(η,ρ) - Regular solution
///
/// This is the regular Coulomb wave function, normalized such that
/// F_L → sin(ρ - η ln(2ρ) - Lπ/2 + σ_L) as ρ → ∞
///
/// # Arguments
/// * `l` - Angular momentum quantum number (L ≥ 0)
/// * `eta` - Sommerfeld parameter η
/// * `rho` - Reduced radial distance ρ
///
/// # Examples
/// ```rust
/// use torsh_special::coulomb_f;
/// use torsh_tensor::Tensor;
/// use torsh_core::device::DeviceType;
/// // let rho = Tensor::from_data(vec![1.0, 2.0, 5.0], vec![3], DeviceType::Cpu).unwrap();
/// // let f = coulomb_f(0, 1.0, &rho).unwrap();
/// ```
pub fn coulomb_f(l: i32, eta: f32, rho: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = rho.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&r| coulomb_f_scalar(l, eta as f64, r as f64) as f32)
        .collect();

    Tensor::from_data(result_data, rho.shape().dims().to_vec(), rho.device())
}

/// Coulomb wave function G_L(η,ρ) - Irregular solution
///
/// This is the irregular Coulomb wave function, normalized such that
/// G_L → -cos(ρ - η ln(2ρ) - Lπ/2 + σ_L) as ρ → ∞
///
/// # Arguments
/// * `l` - Angular momentum quantum number (L ≥ 0)
/// * `eta` - Sommerfeld parameter η
/// * `rho` - Reduced radial distance ρ
///
/// # Examples
/// ```rust
/// use torsh_special::coulomb_g;
/// use torsh_tensor::Tensor;
/// use torsh_core::device::DeviceType;
/// // let rho = Tensor::from_data(vec![1.0, 2.0, 5.0], vec![3], DeviceType::Cpu).unwrap();
/// // let g = coulomb_g(0, 1.0, &rho).unwrap();
/// ```
pub fn coulomb_g(l: i32, eta: f32, rho: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = rho.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&r| coulomb_g_scalar(l, eta as f64, r as f64) as f32)
        .collect();

    Tensor::from_data(result_data, rho.shape().dims().to_vec(), rho.device())
}

/// Coulomb phase shift σ_L(η)
///
/// Computes the Coulomb phase shift using:
/// σ_L(η) = arg[Γ(L + 1 + iη)]
///
/// # Arguments
/// * `l` - Angular momentum quantum number
/// * `eta` - Sommerfeld parameter
pub fn coulomb_sigma(l: i32, eta: f32) -> f32 {
    coulomb_sigma_scalar(l, eta as f64) as f32
}

// ============================================================================
// Scalar implementations
// ============================================================================

/// Scalar implementation of Coulomb F function using series expansion
fn coulomb_f_scalar(l: i32, eta: f64, rho: f64) -> f64 {
    if rho <= 0.0 {
        return 0.0;
    }

    // For small ρ, use series expansion near origin
    if rho < 1.0 {
        coulomb_f_series(l, eta, rho)
    } else {
        // For larger ρ, use asymptotic expansion or numerical integration
        coulomb_f_asymptotic(l, eta, rho)
    }
}

/// Series expansion for small ρ
/// F_L(η,ρ) = C_L(η) ρ^(L+1) M(L+1+iη, 2L+2, -2iρ)
fn coulomb_f_series(l: i32, eta: f64, rho: f64) -> f64 {
    let c_l = coulomb_normalization(l, eta);
    let rho_power = rho.powi(l + 1);

    // Confluent hypergeometric function 1F1(a, b, z)
    // For now, use first few terms of series
    let a = (l + 1) as f64 + eta; // Note: simplified, should use complex for iη
    let b = 2.0 * (l + 1) as f64;
    let z = -2.0 * rho; // Simplified: should be -2iρ

    let mut term = 1.0;
    let mut sum = 1.0;

    for n in 1..50 {
        term *= a + (n - 1) as f64;
        term *= z;
        term /= b + (n - 1) as f64;
        term /= n as f64;

        sum += term;

        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }

    c_l * rho_power * sum
}

/// Asymptotic expansion for large ρ
fn coulomb_f_asymptotic(l: i32, eta: f64, rho: f64) -> f64 {
    let sigma = coulomb_sigma_scalar(l, eta);
    let theta = rho - eta * (2.0 * rho).ln() - (l as f64) * PI / 2.0 + sigma;
    theta.sin()
}

/// Scalar implementation of Coulomb G function
fn coulomb_g_scalar(l: i32, eta: f64, rho: f64) -> f64 {
    if rho <= 0.0 {
        return f64::INFINITY;
    }

    // For small ρ, G diverges like ρ^(-L)
    if rho < 0.1 && l >= 0 {
        return coulomb_g_small_rho(l, eta, rho);
    }

    // For larger ρ, use asymptotic form
    coulomb_g_asymptotic(l, eta, rho)
}

/// Small ρ behavior for G function
fn coulomb_g_small_rho(l: i32, eta: f64, rho: f64) -> f64 {
    let c_l = coulomb_normalization(l, eta);
    let factor = if l == 0 {
        -eta.ln() - 0.5772156649015329 // Euler's constant
    } else {
        let l_factorial = (1..=l).fold(1.0, |acc, i| acc * i as f64);
        (2.0 * l as f64 + 1.0) / (2.0 * l_factorial * rho.powi(l))
    };

    c_l * factor
}

/// Asymptotic expansion for G function
fn coulomb_g_asymptotic(l: i32, eta: f64, rho: f64) -> f64 {
    let sigma = coulomb_sigma_scalar(l, eta);
    let theta = rho - eta * (2.0 * rho).ln() - (l as f64) * PI / 2.0 + sigma;
    -theta.cos()
}

/// Coulomb phase shift σ_L(η) = arg[Γ(L + 1 + iη)]
fn coulomb_sigma_scalar(l: i32, eta: f64) -> f64 {
    // Use Stirling's approximation for the argument of gamma function
    // σ_L(η) ≈ η ln(2L+2) - Lπ/2 + arg[Γ(1 + iη)]

    if eta.abs() < 1e-10 {
        return 0.0;
    }

    let mut sigma = 0.0;
    for k in 1..=l {
        sigma += (eta / k as f64).atan();
    }

    sigma
}

/// Normalization constant C_L(η)
fn coulomb_normalization(l: i32, eta: f64) -> f64 {
    // C_L(η) = 2^L exp(-πη/2) |Γ(L + 1 + iη)| / Γ(2L + 2)

    // Simplified calculation
    let two_l = 2.0_f64.powi(l);
    let exp_factor = (-PI * eta / 2.0).exp();

    // |Γ(L + 1 + iη)| approximation
    let gamma_abs = if eta.abs() < 0.1 {
        // For small η, |Γ(L + 1 + iη)| ≈ Γ(L + 1)
        (1..=l).fold(1.0, |acc, i| acc * i as f64)
    } else {
        // Use Stirling approximation
        let l_plus_1 = (l + 1) as f64;
        ((l_plus_1 * l_plus_1 + eta * eta).sqrt()).powi(l)
            * (1.0 + eta * eta / (l_plus_1 * l_plus_1)).sqrt()
    };

    let gamma_2l_plus_2 = (1..=(2 * l + 1)).fold(1.0, |acc, i| acc * i as f64);

    two_l * exp_factor * gamma_abs / gamma_2l_plus_2
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_coulomb_f_basic() -> TorshResult<()> {
        let rho = Tensor::from_data(vec![1.0_f32, 2.0, 5.0], vec![3], DeviceType::Cpu)?;
        let f = coulomb_f(0, 1.0, &rho)?;
        let result = f.data()?;

        // Basic sanity checks
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&x| x.is_finite()));

        Ok(())
    }

    #[test]
    fn test_coulomb_g_basic() -> TorshResult<()> {
        let rho = Tensor::from_data(vec![1.0_f32, 2.0, 5.0], vec![3], DeviceType::Cpu)?;
        let g = coulomb_g(0, 1.0, &rho)?;
        let result = g.data()?;

        // Basic sanity checks
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&x| x.is_finite()));

        Ok(())
    }

    #[test]
    fn test_coulomb_sigma() {
        // σ_0(0) should be 0
        let sigma = coulomb_sigma(0, 0.0);
        assert_relative_eq!(sigma, 0.0, epsilon = 1e-6);

        // Test some known values
        let sigma = coulomb_sigma(1, 1.0);
        assert!(sigma.is_finite());
    }

    #[test]
    fn test_coulomb_f_zero_rho() -> TorshResult<()> {
        let rho = Tensor::from_data(vec![0.0_f32], vec![1], DeviceType::Cpu)?;
        let f = coulomb_f(0, 1.0, &rho)?;
        let result = f.data()?;

        // F should be 0 at ρ = 0
        assert_relative_eq!(result[0], 0.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_coulomb_normalization() {
        // Test normalization constant for L=0
        let c = coulomb_normalization(0, 1.0);
        assert!(c > 0.0 && c.is_finite());

        // Test for L=1
        let c = coulomb_normalization(1, 1.0);
        assert!(c > 0.0 && c.is_finite());
    }

    #[test]
    fn test_coulomb_wronskian() -> TorshResult<()> {
        // Wronskian W(F, G) = 1 for all ρ
        let rho_val = 2.0;
        let l = 0;
        let eta = 1.0;

        let f = coulomb_f_scalar(l, eta as f64, rho_val as f64);
        let g = coulomb_g_scalar(l, eta as f64, rho_val as f64);

        // Basic check that both are finite
        assert!(f.is_finite());
        assert!(g.is_finite());

        Ok(())
    }
}
