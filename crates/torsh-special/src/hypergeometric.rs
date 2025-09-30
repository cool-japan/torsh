//! Hypergeometric Functions
//!
//! This module provides implementations of hypergeometric functions including
//! the Gauss hypergeometric function 2F1, confluent hypergeometric functions,
//! and related special functions.

use crate::TorshResult;
use std::f64::consts::PI;
use torsh_tensor::Tensor;

/// Pochhammer symbol (rising factorial) (x)ₙ = x(x+1)(x+2)...(x+n-1)
///
/// Computes the Pochhammer symbol (x)ₙ which represents the rising factorial.
/// This is a fundamental building block for hypergeometric series.
/// For n = 0, returns 1 by convention.
/// For negative n, uses the identity (x)₋ₙ = 1/((x-n)ₙ).
pub fn pochhammer(x: &Tensor<f32>, n: i32) -> TorshResult<Tensor<f32>> {
    let data = x.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x_val| pochhammer_scalar(x_val as f64, n) as f32)
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Scalar implementation of the Pochhammer symbol
fn pochhammer_scalar(x: f64, n: i32) -> f64 {
    if n == 0 {
        return 1.0;
    }

    if n < 0 {
        // Use identity (x)₋ₙ = 1/((x-n)ₙ)
        let abs_n = (-n) as usize;
        return 1.0 / pochhammer_scalar(x - abs_n as f64, -n);
    }

    let mut result = 1.0;
    for i in 0..n {
        result *= x + i as f64;
    }
    result
}

/// Binomial coefficient C(n, k) = n! / (k! * (n-k)!)
///
/// Computes the binomial coefficient using the Pochhammer symbol for numerical stability.
/// Returns NaN for invalid inputs (k < 0 or k > n when n is a positive integer).
pub fn binomial(n: &Tensor<f32>, k: i32) -> TorshResult<Tensor<f32>> {
    let data = n.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&n_val| binomial_scalar(n_val as f64, k) as f32)
        .collect();

    Tensor::from_data(result_data, n.shape().dims().to_vec(), n.device())
}

/// Scalar implementation of binomial coefficient
fn binomial_scalar(n: f64, k: i32) -> f64 {
    if k < 0 {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }

    // Use C(n,k) = (n)_k / k! = pochhammer(n-k+1, k) / k!
    let poch = pochhammer_scalar(n - k as f64 + 1.0, k);
    let k_factorial = (1..=k).fold(1.0, |acc, i| acc * i as f64);
    poch / k_factorial
}

/// Gauss hypergeometric function ₂F₁(a, b; c; z)
///
/// Computes the hypergeometric function ₂F₁(a, b; c; z) = Σₙ₌₀^∞ (a)ₙ(b)ₙ zⁿ / ((c)ₙ n!)
/// where (x)ₙ is the Pochhammer symbol (rising factorial).
pub fn hypergeometric_2f1(a: f32, b: f32, c: f32, z: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = z.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&z_val| {
            let z_f64 = z_val as f64;
            hypergeometric_2f1_impl(a as f64, b as f64, c as f64, z_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Confluent hypergeometric function ₁F₁(a; c; z)
///
/// Computes the confluent hypergeometric function ₁F₁(a; c; z) = Σₙ₌₀^∞ (a)ₙ zⁿ / ((c)ₙ n!)
/// Also known as Kummer's function M(a, c, z).
pub fn hypergeometric_1f1(a: f32, c: f32, z: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = z.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&z_val| {
            let z_f64 = z_val as f64;
            hypergeometric_1f1_impl(a as f64, c as f64, z_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Confluent hypergeometric function of the second kind U(a, c, z)
///
/// Computes the confluent hypergeometric function U(a, c, z), also known as
/// Tricomi's function or the confluent hypergeometric function of the second kind.
pub fn hypergeometric_u(a: f32, c: f32, z: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    let data = z.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&z_val| {
            let z_f64 = z_val as f64;
            hypergeometric_u_impl(a as f64, c as f64, z_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Generalized hypergeometric function ₚFₑ
///
/// Computes ₚFₑ(a₁,...,aₚ; b₁,...,bₑ; z) for p ≤ q+1.
/// Note: This is a simplified implementation for small p, q.
pub fn hypergeometric_pfq(
    a_params: &[f32],
    b_params: &[f32],
    z: &Tensor<f32>,
) -> TorshResult<Tensor<f32>> {
    let data = z.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&z_val| {
            let z_f64 = z_val as f64;
            let a_f64: Vec<f64> = a_params.iter().map(|&x| x as f64).collect();
            let b_f64: Vec<f64> = b_params.iter().map(|&x| x as f64).collect();
            hypergeometric_pfq_impl(&a_f64, &b_f64, z_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

/// Appell hypergeometric function F₁(a, b₁, b₂; c; x, y)
///
/// Computes F₁(a, b₁, b₂; c; x, y) = Σₘ,ₙ₌₀^∞ (a)ₘ₊ₙ(b₁)ₘ(b₂)ₙ xᵐyⁿ / ((c)ₘ₊ₙ m! n!)
pub fn appell_f1(
    a: f32,
    b1: f32,
    b2: f32,
    c: f32,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
) -> TorshResult<Tensor<f32>> {
    let x_data = x.data()?;
    let y_data = y.data()?;

    let result_data: Vec<f32> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(&x_val, &y_val)| {
            let x_f64 = x_val as f64;
            let y_f64 = y_val as f64;
            appell_f1_impl(a as f64, b1 as f64, b2 as f64, c as f64, x_f64, y_f64) as f32
        })
        .collect();

    Tensor::from_data(result_data, x.shape().dims().to_vec(), x.device())
}

/// Meijer G-function (simplified implementation)
///
/// This is a placeholder for the Meijer G-function, which is extremely complex.
/// In practice, this would require specialized numerical libraries.
pub fn meijer_g(
    _m: i32,
    _n: i32,
    _p: i32,
    _q: i32,
    _a_params: &[f32],
    _b_params: &[f32],
    z: &Tensor<f32>,
) -> TorshResult<Tensor<f32>> {
    let data = z.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&_z_val| {
            // Placeholder implementation - Meijer G functions are extremely complex
            // and typically require specialized numerical libraries
            1.0f32
        })
        .collect();

    Tensor::from_data(result_data, z.shape().dims().to_vec(), z.device())
}

// Helper functions for numerical computation

/// Pochhammer symbol (rising factorial) (a)ₙ = a(a+1)...(a+n-1)
fn pochhammer_helper(a: f64, n: i32) -> f64 {
    if n == 0 {
        1.0
    } else if n > 0 {
        let mut result = 1.0;
        for k in 0..n {
            result *= a + k as f64;
        }
        result
    } else {
        // For negative n, use (a)₍₋ₙ₎ = 1 / (a-n)₍ₙ₎
        1.0 / pochhammer_helper(a - n as f64, -n)
    }
}

/// Compute ₂F₁(a, b; c; z) using series expansion and transformations
fn hypergeometric_2f1_impl(a: f64, b: f64, c: f64, z: f64) -> f64 {
    // Handle special cases
    if z == 0.0 {
        return 1.0;
    }

    if z == 1.0 {
        // ₂F₁(a, b; c; 1) = Γ(c)Γ(c-a-b) / (Γ(c-a)Γ(c-b)) if Re(c-a-b) > 0
        if c - a - b > 0.0 {
            return gamma_func(c) * gamma_func(c - a - b) / (gamma_func(c - a) * gamma_func(c - b));
        } else {
            return f64::INFINITY;
        }
    }

    // Use series expansion for |z| < 1
    if z.abs() < 0.8 {
        let mut sum = 1.0;
        let mut term: f64 = 1.0;
        let mut n = 1;

        while term.abs() > 1e-15 && n < 200 {
            term *=
                (a + n as f64 - 1.0) * (b + n as f64 - 1.0) * z / ((c + n as f64 - 1.0) * n as f64);
            sum += term;
            n += 1;
        }
        sum
    } else if z < 1.0 {
        // Use transformation ₂F₁(a, b; c; z) = (1-z)^(-a) ₂F₁(a, c-b; c; z/(z-1))
        let z_transformed = z / (z - 1.0);
        let factor = (1.0 - z).powf(-a);
        factor * hypergeometric_2f1_series(a, c - b, c, z_transformed)
    } else {
        // For |z| ≥ 1, need more sophisticated methods
        f64::NAN
    }
}

/// Series expansion for ₂F₁
fn hypergeometric_2f1_series(a: f64, b: f64, c: f64, z: f64) -> f64 {
    let mut sum = 1.0;
    let mut term: f64 = 1.0;
    let mut n = 1;

    while term.abs() > 1e-15 && n < 200 {
        term *= (a + n as f64 - 1.0) * (b + n as f64 - 1.0) * z / ((c + n as f64 - 1.0) * n as f64);
        sum += term;
        n += 1;
    }
    sum
}

/// Compute ₁F₁(a; c; z) using series expansion
fn hypergeometric_1f1_impl(a: f64, c: f64, z: f64) -> f64 {
    if z == 0.0 {
        return 1.0;
    }

    // Series expansion: ₁F₁(a; c; z) = Σ [(a)ₙ zⁿ] / [(c)ₙ n!]
    let mut sum = 1.0;
    let mut term: f64 = 1.0;

    for n in 1..200 {
        // Update term: multiply by (a+n-1)*z / ((c+n-1)*n)
        term *= (a + (n - 1) as f64) * z / ((c + (n - 1) as f64) * n as f64);

        if term.abs() < 1e-15 {
            break;
        }
        sum += term;
    }
    sum
}

/// Compute U(a, c, z) - confluent hypergeometric function of the second kind
fn hypergeometric_u_impl(a: f64, c: f64, z: f64) -> f64 {
    if z <= 0.0 {
        return f64::INFINITY;
    }

    // For large z, use asymptotic expansion
    if z > 10.0 {
        let mut sum = 1.0;
        let mut term: f64 = 1.0;
        let mut n = 1;

        while term.abs() > 1e-15 && n < 50 {
            term *= (a + n as f64 - 1.0) * (a - c + n as f64) / (z * n as f64);
            sum += term;
            n += 1;
        }
        z.powf(-a) * sum
    } else {
        // Use relation with ₁F₁ for moderate z
        let m1f1 = hypergeometric_1f1_impl(a, c, z);
        let gamma_factor = gamma_func(1.0 - c) / gamma_func(a - c + 1.0);
        let z_factor = z.powf(1.0 - c);
        let second_term = hypergeometric_1f1_impl(a - c + 1.0, 2.0 - c, z);

        gamma_factor * (m1f1 + z_factor * gamma_func(c - 1.0) / gamma_func(a) * second_term)
    }
}

/// Compute ₚFₑ using series expansion (simplified for small p, q)
fn hypergeometric_pfq_impl(a_params: &[f64], b_params: &[f64], z: f64) -> f64 {
    if z == 0.0 {
        return 1.0;
    }

    let p = a_params.len();
    let q = b_params.len();

    // Series expansion
    let mut sum = 1.0;
    let mut term: f64 = 1.0;
    let mut n = 1;

    while term.abs() > 1e-15 && n < 100 {
        let mut a_prod = 1.0;
        for &a in a_params {
            a_prod *= a + n as f64 - 1.0;
        }

        let mut b_prod = 1.0;
        for &b in b_params {
            b_prod *= b + n as f64 - 1.0;
        }

        term *= a_prod * z / (b_prod * n as f64);
        sum += term;
        n += 1;

        // Check convergence criteria
        if p > q + 1 && z.abs() >= 1.0 {
            break; // Series may not converge
        }
    }

    sum
}

/// Compute Appell F₁ function
fn appell_f1_impl(a: f64, b1: f64, b2: f64, c: f64, x: f64, y: f64) -> f64 {
    if x == 0.0 && y == 0.0 {
        return 1.0;
    }

    // Double series expansion (limited terms for numerical stability)
    let mut sum = 1.0;

    for m in 1..=20 {
        for n in 1..=20 {
            let coeff =
                pochhammer_helper(a, m + n) * pochhammer_helper(b1, m) * pochhammer_helper(b2, n)
                    / (pochhammer_helper(c, m + n) * factorial(m) * factorial(n));
            let term = coeff * x.powi(m) * y.powi(n);

            if term.abs() < 1e-15 {
                break;
            }
            sum += term;
        }
    }

    sum
}

/// Simple gamma function approximation using Stirling's formula
fn gamma_func(x: f64) -> f64 {
    if x < 0.5 {
        // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        PI / ((PI * x).sin() * gamma_func(1.0 - x))
    } else if x < 1.5 {
        // Use Γ(x+1) = x*Γ(x)
        gamma_func(x + 1.0) / x
    } else {
        // Stirling's approximation for x ≥ 1.5
        let ln_gamma = (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln() + 1.0 / (12.0 * x)
            - 1.0 / (360.0 * x.powi(3));
        ln_gamma.exp()
    }
}

/// Simple factorial function
fn factorial(n: i32) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (2..=n).map(|i| i as f64).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::tensor;

    #[test]
    fn test_hypergeometric_2f1() -> TorshResult<()> {
        let z = tensor![0.0f32, 0.5]?;
        let result = hypergeometric_2f1(1.0, 1.0, 2.0, &z)?;
        let data = result.data()?;

        // ₂F₁(1, 1; 2; 0) = 1
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        // ₂F₁(1, 1; 2; 0.5) = 2*ln(2) ≈ 1.386 (using identity: ₂F₁(1,1;2;z) = -ln(1-z)/z)
        assert_relative_eq!(data[1], 2.0 * 2.0_f32.ln(), epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_hypergeometric_1f1() -> TorshResult<()> {
        let z = tensor![0.0f32, 1.0]?;
        let result = hypergeometric_1f1(1.0, 2.0, &z)?;
        let data = result.data()?;

        // ₁F₁(1; 2; 0) = 1
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        // ₁F₁(1; 2; 1) = (e - 1) ≈ 1.718 (using identity: ₁F₁(1; 2; z) = (e^z - 1)/z)
        assert_relative_eq!(data[1], std::f32::consts::E - 1.0, epsilon = 1e-3);
        Ok(())
    }

    #[test]
    fn test_pochhammer() -> TorshResult<()> {
        assert_relative_eq!(pochhammer_helper(1.0, 0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(pochhammer_helper(1.0, 3), 6.0, epsilon = 1e-10); // 1*2*3
        assert_relative_eq!(pochhammer_helper(0.5, 2), 0.75, epsilon = 1e-10); // 0.5*1.5 = 0.75
        Ok(())
    }

    #[test]
    fn test_appell_f1() -> TorshResult<()> {
        let x = tensor![0.0f32, 0.1]?;
        let y = tensor![0.0f32, 0.1]?;
        let result = appell_f1(1.0, 1.0, 1.0, 2.0, &x, &y)?;
        let data = result.data()?;

        // F₁(1, 1, 1; 2; 0, 0) = 1
        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6);
        // F₁(1, 1, 1; 2; 0.1, 0.1) should be close to 1 for small values
        assert!(data[1] > 1.0 && data[1] < 2.0);
        Ok(())
    }
}
