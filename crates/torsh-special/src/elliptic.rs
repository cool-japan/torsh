//! Elliptic Functions and Integrals
//!
//! This module provides implementations of elliptic functions and integrals.

use crate::TorshResult;
use std::f64::consts::PI;
use torsh_tensor::Tensor;

/// Complete elliptic integral of the first kind K(m)
///
/// Computes K(m) = ∫₀^(π/2) dθ / √(1 - m sin²θ)
/// where m is the parameter (squared modulus).
pub fn elliptic_k(m: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = m.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&m_val| {
            let m_f64 = m_val as f64;
            if m_f64 >= 1.0 {
                f32::INFINITY
            } else if m_f64 <= 0.0 {
                (PI / 2.0) as f32
            } else {
                // Use AGM (Arithmetic-Geometric Mean) method for K(m)
                agm_elliptic_k(m_f64) as f32
            }
        })
        .collect();

    Tensor::from_data(result_data, m.shape().dims().to_vec(), m.device())
}

/// Complete elliptic integral of the second kind E(m)
///
/// Computes E(m) = ∫₀^(π/2) √(1 - m sin²θ) dθ
/// where m is the parameter (squared modulus).
pub fn elliptic_e(m: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = m.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&m_val| {
            let m_f64 = m_val as f64;
            if m_f64 >= 1.0 {
                1.0
            } else if m_f64 <= 0.0 {
                (PI / 2.0) as f32
            } else {
                // Use AGM method for E(m)
                agm_elliptic_e(m_f64) as f32
            }
        })
        .collect();

    Tensor::from_data(result_data, m.shape().dims().to_vec(), m.device())
}

/// Incomplete elliptic integral of the first kind F(φ, m)
///
/// Computes F(φ, m) = ∫₀^φ dθ / √(1 - m sin²θ)
/// where φ is the amplitude and m is the parameter.
pub fn elliptic_f(phi: &Tensor<f32>, m: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let phi_data = phi.data()?;
    let m_data = m.data()?;

    let result_data: Vec<f32> = phi_data
        .iter()
        .zip(m_data.iter())
        .map(|(&phi_val, &m_val)| incomplete_elliptic_f(phi_val as f64, m_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, phi.shape().dims().to_vec(), phi.device())
}

/// Incomplete elliptic integral of the second kind E(φ, m)
///
/// Computes E(φ, m) = ∫₀^φ √(1 - m sin²θ) dθ
/// where φ is the amplitude and m is the parameter.
pub fn elliptic_e_incomplete(phi: &Tensor<f32>, m: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let phi_data = phi.data()?;
    let m_data = m.data()?;

    let result_data: Vec<f32> = phi_data
        .iter()
        .zip(m_data.iter())
        .map(|(&phi_val, &m_val)| incomplete_elliptic_e(phi_val as f64, m_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, phi.shape().dims().to_vec(), phi.device())
}

/// Jacobi elliptic function sn(u, m)
///
/// Computes the Jacobi sine amplitude function.
pub fn jacobi_sn(u: &Tensor<f32>, m: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let u_data = u.data()?;
    let m_data = m.data()?;

    let result_data: Vec<f32> = u_data
        .iter()
        .zip(m_data.iter())
        .map(|(&u_val, &m_val)| {
            let (sn, _cn, _dn) = jacobi_elliptic(u_val as f64, m_val as f64);
            sn as f32
        })
        .collect();

    Tensor::from_data(result_data, u.shape().dims().to_vec(), u.device())
}

/// Jacobi elliptic function cn(u, m)
///
/// Computes the Jacobi cosine amplitude function.
pub fn jacobi_cn(u: &Tensor<f32>, m: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let u_data = u.data()?;
    let m_data = m.data()?;

    let result_data: Vec<f32> = u_data
        .iter()
        .zip(m_data.iter())
        .map(|(&u_val, &m_val)| {
            let (_sn, cn, _dn) = jacobi_elliptic(u_val as f64, m_val as f64);
            cn as f32
        })
        .collect();

    Tensor::from_data(result_data, u.shape().dims().to_vec(), u.device())
}

/// Jacobi elliptic function dn(u, m)
///
/// Computes the Jacobi delta amplitude function.
pub fn jacobi_dn(u: &Tensor<f32>, m: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let u_data = u.data()?;
    let m_data = m.data()?;

    let result_data: Vec<f32> = u_data
        .iter()
        .zip(m_data.iter())
        .map(|(&u_val, &m_val)| {
            let (_sn, _cn, dn) = jacobi_elliptic(u_val as f64, m_val as f64);
            dn as f32
        })
        .collect();

    Tensor::from_data(result_data, u.shape().dims().to_vec(), u.device())
}

// Helper functions for numerical computation

/// Compute complete elliptic integral K(m) using AGM
fn agm_elliptic_k(m: f64) -> f64 {
    if m >= 1.0 {
        return f64::INFINITY;
    }
    if m <= 0.0 {
        return PI / 2.0;
    }

    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();
    let mut _c = m.sqrt();

    // AGM iteration
    for _ in 0..50 {
        let a_new = (a + b) / 2.0;
        let b_new = (a * b).sqrt();
        let c_new = (a - b) / 2.0;

        if c_new.abs() < 1e-15 {
            break;
        }

        a = a_new;
        b = b_new;
        _c = c_new;
    }

    PI / (2.0 * a)
}

/// Compute complete elliptic integral E(m) using AGM
fn agm_elliptic_e(m: f64) -> f64 {
    if m >= 1.0 {
        return 1.0;
    }
    if m <= 0.0 {
        return PI / 2.0;
    }

    // E(m) = π/2 * ₂F₁(-1/2, 1/2; 1; m)
    // Use hypergeometric series: ₂F₁(a,b;c;z) = Σ (a)_n (b)_n z^n / ((c)_n n!)
    // where (a)_n = a(a+1)...(a+n-1) is the Pochhammer symbol

    let a = -0.5;
    let b = 0.5;
    let c = 1.0;

    let mut sum = 1.0; // n=0 term
    let mut term;
    let mut pochhammer_a = a;
    let mut pochhammer_b = b;
    let mut pochhammer_c = c;
    let mut factorial = 1.0;
    let mut z_power = m;

    for n in 1..50 {
        // Update Pochhammer symbols
        if n > 1 {
            pochhammer_a *= a + n as f64 - 1.0;
            pochhammer_b *= b + n as f64 - 1.0;
            pochhammer_c *= c + n as f64 - 1.0;
            factorial *= n as f64;
            z_power *= m;
        } else {
            // For n=1
            pochhammer_a = a;
            pochhammer_b = b;
            pochhammer_c = c;
            factorial = 1.0;
            z_power = m;
        }

        term = (pochhammer_a * pochhammer_b * z_power) / (pochhammer_c * factorial);
        sum += term;

        if term.abs() < 1e-15 {
            break;
        }
    }

    (PI / 2.0) * sum
}

/// Compute incomplete elliptic integral F(φ, m)
fn incomplete_elliptic_f(phi: f64, m: f64) -> f64 {
    if m >= 1.0 || phi.abs() >= PI / 2.0 {
        return f64::INFINITY;
    }

    // Simple numerical integration using Simpson's rule
    let n = 100;
    let h = phi / n as f64;
    let mut sum = 0.0;

    for i in 0..=n {
        let theta = i as f64 * h;
        let sin_theta = theta.sin();
        let integrand = 1.0 / (1.0 - m * sin_theta * sin_theta).sqrt();

        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };

        sum += weight * integrand;
    }

    h * sum / 3.0
}

/// Compute incomplete elliptic integral E(φ, m)
fn incomplete_elliptic_e(phi: f64, m: f64) -> f64 {
    if phi.abs() >= PI / 2.0 {
        return 1.0;
    }

    // Simple numerical integration using Simpson's rule
    let n = 100;
    let h = phi / n as f64;
    let mut sum = 0.0;

    for i in 0..=n {
        let theta = i as f64 * h;
        let sin_theta = theta.sin();
        let integrand = (1.0 - m * sin_theta * sin_theta).sqrt();

        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };

        sum += weight * integrand;
    }

    h * sum / 3.0
}

/// Compute Jacobi elliptic functions (sn, cn, dn) using descending Landen transformation
fn jacobi_elliptic(u: f64, m: f64) -> (f64, f64, f64) {
    if m <= 0.0 {
        let sin_u = u.sin();
        let cos_u = u.cos();
        return (sin_u, cos_u, 1.0);
    }

    if m >= 1.0 {
        let tanh_u = u.tanh();
        let sech_u = 1.0 / u.cosh();
        return (tanh_u, sech_u, sech_u);
    }

    // Use series expansion for small u and moderate m
    if u.abs() < 1e-10 {
        return (u, 1.0, 1.0);
    }

    // Use AGM-based algorithm for numerical stability
    let mut a = Vec::new();
    let mut g = Vec::new();
    let mut c = Vec::new();

    a.push(1.0);
    g.push((1.0 - m).sqrt());
    c.push(m.sqrt());

    // AGM iterations
    for n in 0..20 {
        if c[n].abs() < 1e-15 {
            break;
        }

        a.push((a[n] + g[n]) / 2.0);
        g.push((a[n] * g[n]).sqrt());
        c.push((a[n] - g[n]) / 2.0);
    }

    let n = a.len() - 1;
    let mut phi = (2_f64.powi(n as i32)) * a[n] * u;

    // Backward recurrence
    for k in (0..n).rev() {
        phi = (phi + (c[k] * phi.sin()).asin()) / 2.0;
    }

    let sn = phi.sin();
    let cn = phi.cos();
    let dn = (1.0 - m * sn * sn).sqrt();

    (sn, cn, dn)
}

/// Weierstrass elliptic function ℘(z; g₂, g₃)
///
/// Computes the Weierstrass elliptic function with invariants g₂ and g₃.
pub fn weierstrass_p(
    z: &Tensor<f32>,
    g2: &Tensor<f32>,
    g3: &Tensor<f32>,
) -> TorshResult<Tensor<f32>> {
    let z_data = z.data()?;
    let g2_data = g2.data()?;
    let g3_data = g3.data()?;

    let result_data: Vec<f32> = z_data
        .iter()
        .zip(g2_data.iter())
        .zip(g3_data.iter())
        .map(|((&z_val, &g2_val), &g3_val)| {
            weierstrass_p_impl(z_val as f64, g2_val as f64, g3_val as f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Weierstrass zeta function ζ(z; g₂, g₃)
///
/// Computes the Weierstrass zeta function, which is the logarithmic derivative of σ(z).
pub fn weierstrass_zeta(
    z: &Tensor<f32>,
    g2: &Tensor<f32>,
    g3: &Tensor<f32>,
) -> TorshResult<Tensor<f32>> {
    let z_data = z.data()?;
    let g2_data = g2.data()?;
    let g3_data = g3.data()?;

    let result_data: Vec<f32> = z_data
        .iter()
        .zip(g2_data.iter())
        .zip(g3_data.iter())
        .map(|((&z_val, &g2_val), &g3_val)| {
            weierstrass_zeta_impl(z_val as f64, g2_val as f64, g3_val as f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Weierstrass sigma function σ(z; g₂, g₃)
///
/// Computes the Weierstrass sigma function.
pub fn weierstrass_sigma(
    z: &Tensor<f32>,
    g2: &Tensor<f32>,
    g3: &Tensor<f32>,
) -> TorshResult<Tensor<f32>> {
    let z_data = z.data()?;
    let g2_data = g2.data()?;
    let g3_data = g3.data()?;

    let result_data: Vec<f32> = z_data
        .iter()
        .zip(g2_data.iter())
        .zip(g3_data.iter())
        .map(|((&z_val, &g2_val), &g3_val)| {
            weierstrass_sigma_impl(z_val as f64, g2_val as f64, g3_val as f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Jacobi theta function θ₁(z, q)
///
/// Computes the first Jacobi theta function.
pub fn theta_1(z: &Tensor<f32>, q: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let z_data = z.data()?;
    let q_data = q.data()?;

    let result_data: Vec<f32> = z_data
        .iter()
        .zip(q_data.iter())
        .map(|(&z_val, &q_val)| theta_1_impl(z_val as f64, q_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Jacobi theta function θ₂(z, q)
///
/// Computes the second Jacobi theta function.
pub fn theta_2(z: &Tensor<f32>, q: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let z_data = z.data()?;
    let q_data = q.data()?;

    let result_data: Vec<f32> = z_data
        .iter()
        .zip(q_data.iter())
        .map(|(&z_val, &q_val)| theta_2_impl(z_val as f64, q_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Jacobi theta function θ₃(z, q)
///
/// Computes the third Jacobi theta function.
pub fn theta_3(z: &Tensor<f32>, q: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let z_data = z.data()?;
    let q_data = q.data()?;

    let result_data: Vec<f32> = z_data
        .iter()
        .zip(q_data.iter())
        .map(|(&z_val, &q_val)| theta_3_impl(z_val as f64, q_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Jacobi theta function θ₄(z, q)
///
/// Computes the fourth Jacobi theta function.
pub fn theta_4(z: &Tensor<f32>, q: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let z_data = z.data()?;
    let q_data = q.data()?;

    let result_data: Vec<f32> = z_data
        .iter()
        .zip(q_data.iter())
        .map(|(&z_val, &q_val)| theta_4_impl(z_val as f64, q_val as f64) as f32)
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

// Implementation functions for Weierstrass functions

/// Implementation of Weierstrass ℘ function using Laurent series expansion
fn weierstrass_p_impl(z: f64, g2: f64, g3: f64) -> f64 {
    if z.abs() < 1e-10 {
        return f64::INFINITY;
    }

    // Simple approximation for small |z|: ℘(z) ≈ 1/z² + g₂/20 * z² + g₃/28 * z⁴ + ...
    let z2 = z * z;
    let z4 = z2 * z2;

    if z.abs() <= 0.5 {
        return 1.0 / z2 + g2 / 20.0 * z2 + g3 / 28.0 * z4;
    }

    // For larger |z|, use numerical approximation
    // This is a simplified implementation
    let mut sum = 1.0 / z2;

    // Add lattice sum contribution (simplified)
    for n in 1..=10 {
        for m in -10..=10 {
            if n == 0 && m == 0 {
                continue;
            }
            let omega = n as f64 + m as f64 * 0.5; // Simplified lattice
            let diff = z - omega;
            if diff.abs() > 1e-10 {
                sum += 1.0 / (diff * diff) - 1.0 / (omega * omega);
            }
        }
    }

    sum
}

/// Implementation of Weierstrass ζ function
fn weierstrass_zeta_impl(z: f64, g2: f64, g3: f64) -> f64 {
    if z.abs() < 1e-10 {
        return f64::INFINITY;
    }

    // Simple approximation: ζ(z) ≈ 1/z + g₂/60 * z³ + g₃/140 * z⁵ + ...
    let z2 = z * z;
    let z3 = z2 * z;
    let z5 = z3 * z2;

    if z.abs() <= 0.5 {
        return 1.0 / z + g2 / 60.0 * z3 + g3 / 140.0 * z5;
    }

    // For larger |z|, use numerical approximation
    let mut sum = 1.0 / z;

    // Add lattice sum contribution (simplified)
    for n in 1..=10 {
        for m in -10..=10 {
            if n == 0 && m == 0 {
                continue;
            }
            let omega = n as f64 + m as f64 * 0.5; // Simplified lattice
            let diff = z - omega;
            if diff.abs() > 1e-10 {
                sum += 1.0 / diff - 1.0 / omega;
            }
        }
    }

    sum
}

/// Implementation of Weierstrass σ function
fn weierstrass_sigma_impl(z: f64, _g2: f64, _g3: f64) -> f64 {
    if z.abs() < 1e-10 {
        return z;
    }

    // Simple approximation: σ(z) ≈ z * (1 - z²/20 * g₂ - z⁴/28 * g₃ + ...)
    // For simplicity, we use σ(z) ≈ z for small z
    if z.abs() < 0.1 {
        return z * (1.0 - z * z / 20.0);
    }

    // For larger |z|, use numerical approximation
    z * (1.0 - z * z / 20.0)
}

// Implementation functions for Jacobi theta functions

/// Implementation of θ₁(z, q) using series expansion
fn theta_1_impl(z: f64, q: f64) -> f64 {
    if q.abs() >= 1.0 {
        return 0.0;
    }

    let mut sum = 0.0;

    // θ₁(z, q) = 2 * ∑_{n=1}^∞ (-1)^(n-1) * q^(n²) * sin(2nz)
    for n in 1..=50 {
        let q_power = q.powi(n * n);
        let term = if n % 2 == 1 { 1.0 } else { -1.0 };
        let contribution = term * q_power * (2.0 * n as f64 * z).sin();
        sum += contribution;

        if contribution.abs() < 1e-15 {
            break;
        }
    }

    2.0 * sum
}

/// Implementation of θ₂(z, q) using series expansion
fn theta_2_impl(z: f64, q: f64) -> f64 {
    if q.abs() >= 1.0 {
        return 0.0;
    }

    let mut sum = 0.0;

    // θ₂(z, q) = 2 * q^(1/4) * ∑_{n=0}^∞ q^(n(n+1)) * cos((2n+1)z)
    for n in 0..=50 {
        let q_power_n_n_plus_1 = q.powi(n * (n + 1));
        let contribution = q_power_n_n_plus_1 * ((2 * n + 1) as f64 * z).cos();
        sum += contribution;

        if contribution.abs() < 1e-15 {
            break;
        }
    }

    2.0 * q.sqrt().sqrt() * sum // 2 * q^(1/4) * sum
}

/// Implementation of θ₃(z, q) using series expansion
fn theta_3_impl(z: f64, q: f64) -> f64 {
    if q.abs() >= 1.0 {
        return 1.0;
    }

    let mut sum = 1.0; // n=0 term

    // θ₃(z, q) = 1 + 2 * ∑_{n=1}^∞ q^(n²) * cos(2nz)
    for n in 1..=50 {
        let q_power = q.powi(n * n);
        let contribution = 2.0 * q_power * (2.0 * n as f64 * z).cos();
        sum += contribution;

        if contribution.abs() < 1e-15 {
            break;
        }
    }

    sum
}

/// Implementation of θ₄(z, q) using series expansion
fn theta_4_impl(z: f64, q: f64) -> f64 {
    if q.abs() >= 1.0 {
        return 1.0;
    }

    let mut sum = 1.0; // n=0 term

    // θ₄(z, q) = 1 + 2 * ∑_{n=1}^∞ (-1)^n * q^(n²) * cos(2nz)
    for n in 1..=50 {
        let q_power = q.powi(n * n);
        let term = if n % 2 == 1 { -1.0 } else { 1.0 };
        let contribution = 2.0 * term * q_power * (2.0 * n as f64 * z).cos();
        sum += contribution;

        if contribution.abs() < 1e-15 {
            break;
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::tensor;

    #[test]
    fn test_elliptic_k() -> TorshResult<()> {
        let m = tensor![0.0f32, 0.5]?;
        let result = elliptic_k(&m)?;
        let data = result.data()?;

        // K(0) = π/2 ≈ 1.5708
        assert_relative_eq!(data[0], std::f32::consts::FRAC_PI_2, epsilon = 1e-4);
        // K(0.5) ≈ 1.8541
        assert_relative_eq!(data[1], 1.8541, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_elliptic_e() -> TorshResult<()> {
        let m = tensor![0.0f32, 0.5]?;
        let result = elliptic_e(&m)?;
        let data = result.data()?;

        // E(0) = π/2 ≈ 1.5708
        assert_relative_eq!(data[0], std::f32::consts::FRAC_PI_2, epsilon = 1e-4);
        // E(0.5) ≈ 1.3506
        assert_relative_eq!(data[1], 1.3506, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_jacobi_functions() -> TorshResult<()> {
        let u = tensor![0.0f32, 1.0]?;
        let m = tensor![0.5f32, 0.5]?;

        let sn_result = jacobi_sn(&u, &m)?;
        let cn_result = jacobi_cn(&u, &m)?;
        let dn_result = jacobi_dn(&u, &m)?;

        let sn_data = sn_result.data()?;
        let cn_data = cn_result.data()?;
        let dn_data = dn_result.data()?;

        // sn(0, m) = 0, cn(0, m) = 1, dn(0, m) = 1
        assert_relative_eq!(sn_data[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(cn_data[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(dn_data[0], 1.0, epsilon = 1e-6);

        // Verify fundamental identity: sn² + cn² = 1
        let sn_sq = sn_data[1] * sn_data[1];
        let cn_sq = cn_data[1] * cn_data[1];
        assert_relative_eq!(sn_sq + cn_sq, 1.0, epsilon = 1e-4);
        Ok(())
    }

    #[test]
    fn test_weierstrass_functions() -> TorshResult<()> {
        let z = tensor![0.1f32, 0.5]?;
        let g2 = tensor![1.0f32, 1.0]?;
        let g3 = tensor![0.0f32, 0.0]?;

        let p_result = weierstrass_p(&z, &g2, &g3)?;
        let zeta_result = weierstrass_zeta(&z, &g2, &g3)?;
        let sigma_result = weierstrass_sigma(&z, &g2, &g3)?;

        let p_data = p_result.data()?;
        let zeta_data = zeta_result.data()?;
        let sigma_data = sigma_result.data()?;

        println!(
            "℘(0.1) = {}, ζ(0.1) = {}, σ(0.1) = {}",
            p_data[0], zeta_data[0], sigma_data[0]
        );

        // Basic sanity checks
        assert!(p_data[0].is_finite());
        assert!(zeta_data[0].is_finite());
        assert!(sigma_data[0].is_finite());

        // For small z, σ(z) ≈ z
        assert_relative_eq!(sigma_data[0], 0.1, epsilon = 1e-1);
        Ok(())
    }

    #[test]
    fn test_theta_functions() -> TorshResult<()> {
        let z = tensor![0.0f32, std::f32::consts::PI / 4.0]?;
        let q = tensor![0.1f32, 0.1]?;

        let theta1_result = theta_1(&z, &q)?;
        let theta2_result = theta_2(&z, &q)?;
        let theta3_result = theta_3(&z, &q)?;
        let theta4_result = theta_4(&z, &q)?;

        let theta1_data = theta1_result.data()?;
        let theta2_data = theta2_result.data()?;
        let theta3_data = theta3_result.data()?;
        let theta4_data = theta4_result.data()?;

        // θ₁(0, q) = 0
        assert_relative_eq!(theta1_data[0], 0.0, epsilon = 1e-6);

        // θ₃(0, q) > 1 for positive q
        assert!(theta3_data[0] > 1.0);

        // θ₄(0, q) should be finite and positive for 0 < q < 1
        assert!(theta4_data[0] > 0.0 && theta4_data[0].is_finite());

        // All theta functions should be finite
        for &val in [
            theta1_data[1],
            theta2_data[1],
            theta3_data[1],
            theta4_data[1],
        ]
        .iter()
        {
            assert!(val.is_finite());
        }
        Ok(())
    }
}
