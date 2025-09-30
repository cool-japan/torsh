//! Orthogonal Polynomials
//!
//! This module provides implementations of classical orthogonal polynomials
//! including Legendre, Chebyshev, Hermite, Laguerre, and Jacobi polynomials.

use crate::TorshResult;
use torsh_tensor::Tensor;

#[allow(dead_code)]
/// Legendre polynomial Pₙ(x)
///
/// Computes the Legendre polynomial of degree n at x.
/// These are orthogonal on [-1, 1] with weight function w(x) = 1.
pub fn legendre_p(n: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            legendre_p_impl(n, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Associated Legendre polynomial Pₙᵐ(x)
///
/// Computes the associated Legendre polynomial of degree n and order m at x.
pub fn legendre_p_associated(n: i32, m: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            legendre_p_associated_impl(n, m, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Chebyshev polynomial of the first kind Tₙ(x)
///
/// Computes the Chebyshev polynomial of the first kind of degree n at x.
/// These are orthogonal on [-1, 1] with weight function w(x) = 1/√(1-x²).
pub fn chebyshev_t(n: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            chebyshev_t_impl(n, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Chebyshev polynomial of the second kind Uₙ(x)
///
/// Computes the Chebyshev polynomial of the second kind of degree n at x.
/// These are orthogonal on [-1, 1] with weight function w(x) = √(1-x²).
pub fn chebyshev_u(n: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            chebyshev_u_impl(n, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Hermite polynomial Hₙ(x) (physicist's convention)
///
/// Computes the Hermite polynomial of degree n at x using the physicist's convention.
/// These are orthogonal on (-∞, ∞) with weight function w(x) = e^(-x²).
pub fn hermite_h(n: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            hermite_h_impl(n, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Hermite polynomial Heₙ(x) (probabilist's convention)
///
/// Computes the Hermite polynomial of degree n at x using the probabilist's convention.
/// These are orthogonal on (-∞, ∞) with weight function w(x) = e^(-x²/2).
pub fn hermite_he(n: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            hermite_he_impl(n, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Laguerre polynomial Lₙ(x)
///
/// Computes the Laguerre polynomial of degree n at x.
/// These are orthogonal on [0, ∞) with weight function w(x) = e^(-x).
pub fn laguerre_l(n: i32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            laguerre_l_impl(n, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Associated Laguerre polynomial Lₙᵅ(x)
///
/// Computes the associated Laguerre polynomial of degree n and parameter α at x.
/// These are orthogonal on [0, ∞) with weight function w(x) = x^α e^(-x).
pub fn laguerre_l_associated(n: i32, alpha: f32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            laguerre_l_associated_impl(n, alpha as f64, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Jacobi polynomial Pₙ^(α,β)(x)
///
/// Computes the Jacobi polynomial of degree n with parameters α and β at x.
/// These are orthogonal on [-1, 1] with weight function w(x) = (1-x)^α (1+x)^β.
pub fn jacobi_p(n: i32, alpha: f32, beta: f32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            jacobi_p_impl(n, alpha as f64, beta as f64, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Gegenbauer (ultraspherical) polynomial Cₙ^(λ)(x)
///
/// Computes the Gegenbauer polynomial of degree n with parameter λ at x.
/// These are orthogonal on [-1, 1] with weight function w(x) = (1-x²)^(λ-1/2).
pub fn gegenbauer_c(n: i32, lambda: f32, x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| {
            let x_f64 = x_val as f64;
            gegenbauer_c_impl(n, lambda as f64, x_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

// Helper functions for numerical computation

/// Compute Legendre polynomial Pₙ(x) using recurrence relation
fn legendre_p_impl(n: i32, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p0 = 1.0;
            let mut p1 = x;

            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            p1
        }
    }
}

/// Compute associated Legendre polynomial Pₙᵐ(x)
fn legendre_p_associated_impl(n: i32, m: i32, x: f64) -> f64 {
    if m < 0 || m > n {
        return 0.0;
    }

    if m == 0 {
        return legendre_p_impl(n, x);
    }

    // Use the formula for associated Legendre polynomials
    let factor = (1.0 - x * x).powf(m as f64 / 2.0);
    let mut pn = legendre_p_impl(n, x);

    // Apply derivative m times
    for _ in 0..m {
        pn = derivative_legendre(n, x, pn);
    }

    factor * pn
}

/// Simple derivative approximation for Legendre polynomials
fn derivative_legendre(n: i32, x: f64, pn: f64) -> f64 {
    if n == 0 {
        0.0
    } else {
        n as f64 * (x * pn - legendre_p_impl(n - 1, x)) / (x * x - 1.0)
    }
}

/// Compute Chebyshev polynomial Tₙ(x) using recurrence relation
fn chebyshev_t_impl(n: i32, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut t0 = 1.0;
            let mut t1 = x;

            for _ in 2..=n {
                let t2 = 2.0 * x * t1 - t0;
                t0 = t1;
                t1 = t2;
            }
            t1
        }
    }
}

/// Compute Chebyshev polynomial Uₙ(x) using recurrence relation
fn chebyshev_u_impl(n: i32, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    match n {
        0 => 1.0,
        1 => 2.0 * x,
        _ => {
            let mut u0 = 1.0;
            let mut u1 = 2.0 * x;

            for _ in 2..=n {
                let u2 = 2.0 * x * u1 - u0;
                u0 = u1;
                u1 = u2;
            }
            u1
        }
    }
}

/// Compute Hermite polynomial Hₙ(x) using recurrence relation (physicist's convention)
fn hermite_h_impl(n: i32, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    match n {
        0 => 1.0,
        1 => 2.0 * x,
        _ => {
            let mut h0 = 1.0;
            let mut h1 = 2.0 * x;

            for k in 2..=n {
                let h2 = 2.0 * x * h1 - 2.0 * (k - 1) as f64 * h0;
                h0 = h1;
                h1 = h2;
            }
            h1
        }
    }
}

/// Compute Hermite polynomial Heₙ(x) using recurrence relation (probabilist's convention)
fn hermite_he_impl(n: i32, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut he0 = 1.0;
            let mut he1 = x;

            for k in 2..=n {
                let he2 = x * he1 - (k - 1) as f64 * he0;
                he0 = he1;
                he1 = he2;
            }
            he1
        }
    }
}

/// Compute Laguerre polynomial Lₙ(x) using recurrence relation
fn laguerre_l_impl(n: i32, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    match n {
        0 => 1.0,
        1 => 1.0 - x,
        _ => {
            let mut l0 = 1.0;
            let mut l1 = 1.0 - x;

            for k in 2..=n {
                let l2 = ((2 * k - 1) as f64 - x) * l1 / k as f64 - (k - 1) as f64 * l0 / k as f64;
                l0 = l1;
                l1 = l2;
            }
            l1
        }
    }
}

/// Compute associated Laguerre polynomial Lₙᵅ(x) using recurrence relation
fn laguerre_l_associated_impl(n: i32, alpha: f64, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    match n {
        0 => 1.0,
        1 => 1.0 + alpha - x,
        _ => {
            let mut l0 = 1.0;
            let mut l1 = 1.0 + alpha - x;

            for k in 2..=n {
                let l2 = ((2 * k - 1) as f64 + alpha - x) * l1 / k as f64
                    - (k - 1 + alpha as i32) as f64 * l0 / k as f64;
                l0 = l1;
                l1 = l2;
            }
            l1
        }
    }
}

/// Compute Jacobi polynomial Pₙ^(α,β)(x) using recurrence relation
fn jacobi_p_impl(n: i32, alpha: f64, beta: f64, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    match n {
        0 => 1.0,
        1 => (alpha - beta + (alpha + beta + 2.0) * x) / 2.0,
        _ => {
            let mut p0 = 1.0;
            let mut p1 = (alpha - beta + (alpha + beta + 2.0) * x) / 2.0;

            for k in 2..=n {
                let k_f64 = k as f64;
                let a1 = 2.0 * k_f64 * (k_f64 + alpha + beta) * (2.0 * k_f64 + alpha + beta - 2.0);
                let a2 = (2.0 * k_f64 + alpha + beta - 1.0) * (alpha * alpha - beta * beta);
                let a3 = (2.0 * k_f64 + alpha + beta - 1.0)
                    * (2.0 * k_f64 + alpha + beta)
                    * (2.0 * k_f64 + alpha + beta - 2.0);
                let a4 = 2.0
                    * (k_f64 + alpha - 1.0)
                    * (k_f64 + beta - 1.0)
                    * (2.0 * k_f64 + alpha + beta);

                let p2 = ((a2 + a3 * x) * p1 - a4 * p0) / a1;
                p0 = p1;
                p1 = p2;
            }
            p1
        }
    }
}

/// Compute Gegenbauer polynomial Cₙ^(λ)(x) using recurrence relation
fn gegenbauer_c_impl(n: i32, lambda: f64, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }

    if lambda == 0.0 {
        // Degenerate case: C_n^(0)(x) = (2/n) T_n(x) for n > 0
        if n == 0 {
            1.0
        } else {
            2.0 * chebyshev_t_impl(n, x) / n as f64
        }
    } else {
        match n {
            0 => 1.0,
            1 => 2.0 * lambda * x,
            _ => {
                let mut c0 = 1.0;
                let mut c1 = 2.0 * lambda * x;

                for k in 2..=n {
                    let c2 = (2.0 * (k as f64 + lambda - 1.0) * x * c1
                        - (k as f64 + 2.0 * lambda - 2.0) * c0)
                        / k as f64;
                    c0 = c1;
                    c1 = c2;
                }
                c1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::tensor;

    #[test]
    fn test_legendre_polynomials() -> TorshResult<()> {
        let x = tensor![-1.0f32, 0.0, 1.0]?;

        // P₀(x) = 1
        let p0 = legendre_p(0, &x)?;
        let data0 = p0.data()?;
        assert_relative_eq!(data0[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data0[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data0[2], 1.0, epsilon = 1e-6);

        // P₁(x) = x
        let p1 = legendre_p(1, &x)?;
        let data1 = p1.data()?;
        assert_relative_eq!(data1[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(data1[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(data1[2], 1.0, epsilon = 1e-6);

        // P₂(x) = (3x² - 1)/2
        let p2 = legendre_p(2, &x)?;
        let data2 = p2.data()?;
        assert_relative_eq!(data2[0], 1.0, epsilon = 1e-6); // (3*1 - 1)/2 = 1
        assert_relative_eq!(data2[1], -0.5, epsilon = 1e-6); // (3*0 - 1)/2 = -0.5
        assert_relative_eq!(data2[2], 1.0, epsilon = 1e-6); // (3*1 - 1)/2 = 1
        Ok(())
    }

    #[test]
    fn test_chebyshev_polynomials() -> TorshResult<()> {
        let x = tensor![-1.0f32, 0.0, 1.0]?;

        // T₀(x) = 1
        let t0 = chebyshev_t(0, &x)?;
        let data0 = t0.data()?;
        assert_relative_eq!(data0[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data0[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data0[2], 1.0, epsilon = 1e-6);

        // T₁(x) = x
        let t1 = chebyshev_t(1, &x)?;
        let data1 = t1.data()?;
        assert_relative_eq!(data1[0], -1.0, epsilon = 1e-6);
        assert_relative_eq!(data1[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(data1[2], 1.0, epsilon = 1e-6);

        // T₂(x) = 2x² - 1
        let t2 = chebyshev_t(2, &x)?;
        let data2 = t2.data()?;
        assert_relative_eq!(data2[0], 1.0, epsilon = 1e-6); // 2*1 - 1 = 1
        assert_relative_eq!(data2[1], -1.0, epsilon = 1e-6); // 2*0 - 1 = -1
        assert_relative_eq!(data2[2], 1.0, epsilon = 1e-6); // 2*1 - 1 = 1
        Ok(())
    }

    #[test]
    fn test_hermite_polynomials() -> TorshResult<()> {
        let x = tensor![0.0f32, 1.0]?;

        // H₀(x) = 1
        let h0 = hermite_h(0, &x)?;
        let data0 = h0.data()?;
        assert_relative_eq!(data0[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data0[1], 1.0, epsilon = 1e-6);

        // H₁(x) = 2x
        let h1 = hermite_h(1, &x)?;
        let data1 = h1.data()?;
        assert_relative_eq!(data1[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(data1[1], 2.0, epsilon = 1e-6);

        // H₂(x) = 4x² - 2
        let h2 = hermite_h(2, &x)?;
        let data2 = h2.data()?;
        assert_relative_eq!(data2[0], -2.0, epsilon = 1e-6); // 4*0 - 2 = -2
        assert_relative_eq!(data2[1], 2.0, epsilon = 1e-6); // 4*1 - 2 = 2
        Ok(())
    }

    #[test]
    fn test_laguerre_polynomials() -> TorshResult<()> {
        let x = tensor![0.0f32, 1.0]?;

        // L₀(x) = 1
        let l0 = laguerre_l(0, &x)?;
        let data0 = l0.data()?;
        assert_relative_eq!(data0[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data0[1], 1.0, epsilon = 1e-6);

        // L₁(x) = 1 - x
        let l1 = laguerre_l(1, &x)?;
        let data1 = l1.data()?;
        assert_relative_eq!(data1[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data1[1], 0.0, epsilon = 1e-6);
        Ok(())
    }

    #[test]
    fn test_jacobi_polynomials() -> TorshResult<()> {
        let x = tensor![-1.0f32, 0.0, 1.0]?;

        // P₀^(α,β)(x) = 1 for any α, β
        let p0 = jacobi_p(0, 1.0, 1.0, &x)?;
        let data0 = p0.data()?;
        assert_relative_eq!(data0[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data0[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(data0[2], 1.0, epsilon = 1e-6);

        // For α = β = 0, Jacobi polynomials reduce to Legendre polynomials
        let p1_jacobi = jacobi_p(1, 0.0, 0.0, &x)?;
        let p1_legendre = legendre_p(1, &x)?;
        let data1_j = p1_jacobi.data()?;
        let data1_l = p1_legendre.data()?;

        for i in 0..3 {
            assert_relative_eq!(data1_j[i], data1_l[i], epsilon = 1e-5);
        }
        Ok(())
    }
}
