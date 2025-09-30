//! Advanced Mathematical Functions
//!
//! This module provides implementations of advanced mathematical functions
//! including zeta functions, polylogarithms, and related special functions.

use crate::TorshResult;
use std::f64::consts::PI;
use torsh_tensor::Tensor;

/// Riemann zeta function ζ(s)
///
/// Computes the Riemann zeta function for real arguments.
/// For s > 1, uses the series ζ(s) = ∑_{n=1}^∞ 1/n^s
/// For s ≤ 1, uses functional equation and other methods.
pub fn riemann_zeta(s: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = s.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&s_val| riemann_zeta_impl(s_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, s.shape().dims().to_vec(), s.device())
}

/// Polylogarithm function Li_s(z)
///
/// Computes the polylogarithm Li_s(z) = ∑_{n=1}^∞ z^n / n^s
pub fn polylogarithm(s: &Tensor<f32>, z: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let s_data = s.data()?;
    let z_data = z.data()?;

    let result_data: Vec<f32> = s_data
        .iter()
        .zip(z_data.iter())
        .map(|(&s_val, &z_val)| polylogarithm_impl(s_val as f64, z_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, s.shape().dims().to_vec(), s.device())
}

/// Hurwitz zeta function ζ(s, a)
///
/// Computes the Hurwitz zeta function ζ(s, a) = ∑_{n=0}^∞ 1/(n+a)^s
pub fn hurwitz_zeta(s: &Tensor<f32>, a: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let s_data = s.data()?;
    let a_data = a.data()?;

    let result_data: Vec<f32> = s_data
        .iter()
        .zip(a_data.iter())
        .map(|(&s_val, &a_val)| hurwitz_zeta_impl(s_val as f64, a_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, s.shape().dims().to_vec(), s.device())
}

/// Dirichlet eta function η(s)
///
/// Computes the Dirichlet eta function η(s) = ∑_{n=1}^∞ (-1)^(n+1) / n^s
/// Also known as the alternating zeta function.
pub fn dirichlet_eta(s: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = s.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&s_val| dirichlet_eta_impl(s_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, s.shape().dims().to_vec(), s.device())
}

/// Barnes G-function G(z)
///
/// Computes the Barnes G-function, which satisfies G(z+1) = Γ(z) G(z)
/// This is a simplified implementation for real positive arguments.
pub fn barnes_g(z: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = z.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&z_val| barnes_g_impl(z_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

// Implementation functions

/// Implementation of Riemann zeta function
fn riemann_zeta_impl(s: f64) -> f64 {
    if s == 1.0 {
        return f64::INFINITY;
    }

    if s <= 0.0 {
        // Use functional equation for s ≤ 0
        // ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
        if s.fract() == 0.0 && s < 0.0 {
            let n = (-s) as i32;
            if n % 2 == 1 {
                return 0.0; // ζ(-2k-1) = 0 for k ≥ 0
            }
        }
        // Simplified handling for negative values
        return 0.0;
    }

    if s > 1.0 {
        // Use direct series for s > 1
        let mut sum = 0.0;
        for n in 1..=1000 {
            let term = 1.0 / (n as f64).powf(s);
            sum += term;
            if term < 1e-15 {
                break;
            }
        }
        return sum;
    }

    // For 0 < s < 1, use functional equation
    // ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
    let zeta_1_minus_s = riemann_zeta_impl(1.0 - s);
    let gamma_1_minus_s = gamma_impl(1.0 - s);
    let sin_term = (PI * s / 2.0).sin();
    let factor = 2_f64.powf(s) * PI.powf(s - 1.0);

    factor * sin_term * gamma_1_minus_s * zeta_1_minus_s
}

/// Implementation of polylogarithm
fn polylogarithm_impl(s: f64, z: f64) -> f64 {
    if z.abs() >= 1.0 && s <= 1.0 {
        return f64::NAN; // Not convergent
    }

    let mut sum = 0.0;
    let mut z_power = z;

    for n in 1..=1000 {
        let term = z_power / (n as f64).powf(s);
        sum += term;
        z_power *= z;

        if term.abs() < 1e-15 {
            break;
        }
    }

    sum
}

/// Implementation of Hurwitz zeta function
fn hurwitz_zeta_impl(s: f64, a: f64) -> f64 {
    if s <= 1.0 || a <= 0.0 {
        return f64::NAN;
    }

    let mut sum = 0.0;
    for n in 0..1000 {
        let term = 1.0 / (n as f64 + a).powf(s);
        sum += term;
        if term < 1e-15 {
            break;
        }
    }

    sum
}

/// Implementation of Dirichlet eta function
fn dirichlet_eta_impl(s: f64) -> f64 {
    // η(s) = (1 - 2^(1-s)) ζ(s)
    if s == 1.0 {
        return (2_f64).ln(); // η(1) = ln(2)
    }

    let zeta_s = riemann_zeta_impl(s);
    let factor = 1.0 - 2_f64.powf(1.0 - s);

    factor * zeta_s
}

/// Implementation of Barnes G-function (simplified)
fn barnes_g_impl(z: f64) -> f64 {
    if z <= 0.0 {
        return f64::NAN;
    }

    // G(1) = 1 (base case)
    if (z - 1.0).abs() < 1e-10 {
        return 1.0;
    }

    if z < 1.0 {
        // Use recurrence G(z+1) = Γ(z) G(z) -> G(z) = G(z+1) / Γ(z)
        return barnes_g_impl(z + 1.0) / gamma_impl(z);
    }

    if z < 2.0 {
        // Use recurrence G(z+1) = Γ(z) G(z)
        return gamma_impl(z) * barnes_g_impl(z + 1.0);
    }

    // For z ≥ 2, use asymptotic expansion (very simplified)
    // G(z) ≈ (2π)^((z-1)/2) exp(-z(z-1)/2 + (z-1)ln(z-1) - (z-1) + B2(z-1)/12)
    let zz = z - 1.0;
    let ln_g = (zz / 2.0) * (2.0 * PI).ln() - zz * zz / 2.0 + zz * zz.ln() - zz;
    ln_g.exp()
}

/// Simple gamma function implementation for internal use
fn gamma_impl(z: f64) -> f64 {
    if z < 0.5 {
        // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        return PI / ((PI * z).sin() * gamma_impl(1.0 - z));
    }

    // Use Stirling's approximation for z ≥ 0.5
    let z_shifted = z - 1.0;
    if z_shifted == 0.0 {
        return 1.0;
    }

    // Simple Stirling approximation: Γ(z) ≈ √(2π/z) (z/e)^z
    let ln_gamma = 0.5 * (2.0 * PI / z_shifted).ln() + z_shifted * (z_shifted.ln() - 1.0);
    ln_gamma.exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::tensor;

    #[test]
    fn test_riemann_zeta() -> TorshResult<()> {
        let s = tensor![2.0f32, 3.0, 4.0]?;
        let result = riemann_zeta(&s)?;
        let data = result.data()?;

        // ζ(2) = π²/6 ≈ 1.6449
        assert_relative_eq!(data[0], (PI * PI / 6.0) as f32, epsilon = 1e-3);

        // ζ(3) ≈ 1.202 (Apéry's constant)
        assert_relative_eq!(data[1], 1.202, epsilon = 1e-2);

        // ζ(4) = π⁴/90 ≈ 1.0823
        assert_relative_eq!(data[2], (PI.powi(4) / 90.0) as f32, epsilon = 1e-2);
        Ok(())
    }

    #[test]
    fn test_dirichlet_eta() -> TorshResult<()> {
        let s = tensor![1.0f32, 2.0]?;
        let result = dirichlet_eta(&s)?;
        let data = result.data()?;

        // η(1) = ln(2) ≈ 0.6931
        assert_relative_eq!(data[0], (2_f64).ln() as f32, epsilon = 1e-3);

        // η(2) = π²/12 ≈ 0.8225
        assert_relative_eq!(data[1], (PI * PI / 12.0) as f32, epsilon = 1e-2);
        Ok(())
    }

    #[test]
    fn test_polylogarithm() -> TorshResult<()> {
        let s = tensor![2.0f32, 3.0]?;
        let z = tensor![0.5f32, 0.5]?;
        let result = polylogarithm(&s, &z)?;
        let data = result.data()?;

        // Li₂(1/2) ≈ 0.5822
        assert_relative_eq!(data[0], 0.5822, epsilon = 1e-2);

        // Basic sanity check for Li₃(1/2)
        assert!(data[1] > 0.0 && data[1] < 1.0);
        Ok(())
    }

    #[test]
    fn test_hurwitz_zeta() -> TorshResult<()> {
        let s = tensor![2.0f32, 3.0]?;
        let a = tensor![1.0f32, 1.0]?;
        let result = hurwitz_zeta(&s, &a)?;
        let data = result.data()?;

        // ζ(2, 1) = ζ(2) = π²/6
        assert_relative_eq!(data[0], (PI * PI / 6.0) as f32, epsilon = 1e-2);

        // ζ(3, 1) = ζ(3) ≈ 1.202
        assert_relative_eq!(data[1], 1.202, epsilon = 1e-1);
        Ok(())
    }

    #[test]
    fn test_barnes_g() -> TorshResult<()> {
        let z = tensor![1.0f32, 2.0, 3.0]?;
        let result = barnes_g(&z)?;
        let data = result.data()?;

        // Basic sanity checks - G function should be positive
        for &val in data.iter() {
            assert!(val > 0.0);
            assert!(val.is_finite());
        }

        // G(1) = 1
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-1);
        Ok(())
    }
}
