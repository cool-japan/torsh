//! Utility Functions for Special Mathematical Operations
//!
//! This module provides utility functions that support special function computations
//! and mathematical analysis. These functions are building blocks for more complex
//! mathematical operations.

use crate::TorshResult;
use torsh_tensor::Tensor;

/// Evaluates a continued fraction using the Lentz method
///
/// This function evaluates continued fractions of the form:
/// a₀ + b₁/(a₁ + b₂/(a₂ + b₃/(a₃ + ...)))
///
/// The Lentz method is a numerically stable algorithm for continued fraction evaluation.
/// It's particularly useful for special functions like hypergeometric functions,
/// Bessel functions, and many others.
///
/// # Arguments
/// * `a0` - The constant term
/// * `coeffs` - A vector of (aₙ, bₙ) coefficient pairs
/// * `max_terms` - Maximum number of terms to evaluate
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// The value of the continued fraction
pub fn continued_fraction(a0: f64, coeffs: &[(f64, f64)], max_terms: usize, tolerance: f64) -> f64 {
    if coeffs.is_empty() {
        return a0;
    }

    let tiny = 1e-30;

    // Handle first term specially
    let (a1, b1) = coeffs[0];
    let mut d = a1;

    if d.abs() < tiny {
        d = tiny;
    }
    let mut c = a0 + b1 / d;
    d = 1.0 / d;
    let mut result = c * d;

    // Lentz algorithm for remaining terms
    for i in 1..max_terms.min(coeffs.len()) {
        let (a_n, b_n) = coeffs[i % coeffs.len()];

        // Update d_n = a_n + b_n/d_{n-1}
        d = a_n + b_n * d;
        if d.abs() < tiny {
            d = tiny;
        }

        // Update c_n = a_n + b_n/c_{n-1}
        c = a_n + b_n / c;
        if c.abs() < tiny {
            c = tiny;
        }

        d = 1.0 / d;
        let delta = c * d;
        result *= delta;

        if (delta - 1.0).abs() < tolerance {
            break;
        }
    }

    result
}

/// Computes the Padé approximant ratio P_m(x)/Q_n(x) for a function
///
/// Padé approximants are rational function approximations that often provide
/// better convergence than Taylor series, especially near poles or singularities.
/// This function computes a simple [m/n] Padé approximant using provided coefficients.
///
/// # Arguments
/// * `x` - The point at which to evaluate the approximant
/// * `p_coeffs` - Coefficients for the numerator polynomial P_m(x)
/// * `q_coeffs` - Coefficients for the denominator polynomial Q_n(x)
///
/// # Returns
/// The value of the Padé approximant P_m(x)/Q_n(x)
pub fn pade_approximant(x: f64, p_coeffs: &[f64], q_coeffs: &[f64]) -> f64 {
    if q_coeffs.is_empty() {
        return f64::NAN;
    }

    // Evaluate numerator polynomial P_m(x)
    let mut numerator = 0.0;
    let mut x_power = 1.0;
    for &coeff in p_coeffs {
        numerator += coeff * x_power;
        x_power *= x;
    }

    // Evaluate denominator polynomial Q_n(x)
    let mut denominator = 0.0;
    x_power = 1.0;
    for &coeff in q_coeffs {
        denominator += coeff * x_power;
        x_power *= x;
    }

    if denominator.abs() < 1e-15 {
        return f64::NAN;
    }

    numerator / denominator
}

/// Evaluates a Chebyshev polynomial expansion
///
/// Chebyshev polynomials provide excellent approximation properties and are
/// widely used in numerical analysis. This function evaluates a series of the form:
/// f(x) ≈ Σ cₙ Tₙ(x) where Tₙ is the nth Chebyshev polynomial of the first kind.
///
/// # Arguments
/// * `x` - The point at which to evaluate (should be in [-1, 1])
/// * `coeffs` - Chebyshev coefficients
///
/// # Returns
/// The value of the Chebyshev expansion
pub fn chebyshev_expansion(x: f64, coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }

    if coeffs.len() == 1 {
        return coeffs[0];
    }

    // Use Clenshaw's recurrence algorithm for numerical stability
    let mut b_k_plus_2 = 0.0;
    let mut b_k_plus_1 = 0.0;

    // Iterate backwards through coefficients
    for &c_k in coeffs.iter().rev() {
        let b_k = c_k + 2.0 * x * b_k_plus_1 - b_k_plus_2;
        b_k_plus_2 = b_k_plus_1;
        b_k_plus_1 = b_k;
    }

    // Final step: T_0(x) = 1, so we subtract the extra T_0 term
    b_k_plus_1 - x * b_k_plus_2
}

/// Computes factorial using an iterative approach with overflow protection
///
/// This function provides a safe factorial computation for moderate values of n.
/// For large factorials, use the gamma function or Stirling's approximation instead.
///
/// # Arguments
/// * `n` - Non-negative integer
///
/// # Returns
/// n! if n is reasonable, or f64::INFINITY for overflow
pub fn factorial_safe(n: u32) -> f64 {
    if n > 170 {
        // Above this, f64 overflows
        return f64::INFINITY;
    }

    if n <= 1 {
        return 1.0;
    }

    let mut result = 1.0;
    for i in 2..=n {
        result *= i as f64;
        if !result.is_finite() {
            return f64::INFINITY;
        }
    }

    result
}

/// Computes double factorial n!! = n × (n-2) × (n-4) × ...
///
/// Double factorial is used in many combinatorial and physics applications.
/// For even n: n!! = n × (n-2) × ... × 2
/// For odd n: n!! = n × (n-2) × ... × 1
///
/// # Arguments
/// * `n` - Non-negative integer
///
/// # Returns
/// n!! if computable, or f64::INFINITY for overflow
pub fn double_factorial(n: u32) -> f64 {
    if n <= 1 {
        return 1.0;
    }

    let mut result = n as f64;
    let mut current = n.saturating_sub(2);

    while current > 0 {
        result *= current as f64;
        if !result.is_finite() {
            return f64::INFINITY;
        }
        current = current.saturating_sub(2);
    }

    result
}

/// Tensor wrapper for continued fraction evaluation
///
/// Evaluates a continued fraction for each element in a tensor using the same
/// coefficient structure.
pub fn continued_fraction_tensor(
    input: &Tensor<f32>,
    coeffs: &[(f64, f64)],
    max_terms: usize,
    tolerance: f64,
) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| continued_fraction(x as f64, coeffs, max_terms, tolerance) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Tensor wrapper for Padé approximant evaluation
///
/// Evaluates a Padé approximant for each element in a tensor.
pub fn pade_approximant_tensor(
    input: &Tensor<f32>,
    p_coeffs: &[f64],
    q_coeffs: &[f64],
) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| pade_approximant(x as f64, p_coeffs, q_coeffs) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Tensor wrapper for Chebyshev expansion evaluation
///
/// Evaluates a Chebyshev expansion for each element in a tensor.
pub fn chebyshev_expansion_tensor(input: &Tensor<f32>, coeffs: &[f64]) -> TorshResult<Tensor<f32>> {
    let data = input.data()?;
    let result_data: Vec<f32> = data
        .iter()
        .map(|&x| chebyshev_expansion(x as f64, coeffs) as f32)
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_continued_fraction_simple() {
        // Test the continued fraction for the golden ratio: 1 + 1/(1 + 1/(1 + ...))
        let coeffs = vec![(1.0, 1.0); 10];
        let result = continued_fraction(1.0, &coeffs, 10, 1e-10);
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;

        assert_relative_eq!(result, golden_ratio, max_relative = 1e-4);
    }

    #[test]
    fn test_pade_approximant_simple() {
        // Test simple rational function: (1 + x) / (1 - x) at x = 0.5
        let p_coeffs = [1.0, 1.0]; // 1 + x
        let q_coeffs = [1.0, -1.0]; // 1 - x
        let result = pade_approximant(0.5, &p_coeffs, &q_coeffs);
        let expected = 1.5 / 0.5; // = 3.0

        assert_relative_eq!(result, expected, max_relative = 1e-10);
    }

    #[test]
    fn test_chebyshev_expansion() {
        // Test T_2(x) = 2x² - 1 at x = 0.5
        let coeffs = [-1.0, 0.0, 2.0]; // Coefficients for T_0, T_1, T_2
        let result = chebyshev_expansion(0.5, &coeffs);
        let expected = -1.0 + 0.0 + 2.0 * (2.0 * 0.5 * 0.5 - 1.0); // = -0.5

        assert_relative_eq!(result, expected, max_relative = 1e-10);
    }

    #[test]
    fn test_factorial_safe() {
        assert_eq!(factorial_safe(0), 1.0);
        assert_eq!(factorial_safe(1), 1.0);
        assert_eq!(factorial_safe(5), 120.0);
        assert_eq!(factorial_safe(10), 3628800.0);
        assert!(factorial_safe(200).is_infinite());
    }

    #[test]
    fn test_double_factorial() {
        assert_eq!(double_factorial(0), 1.0);
        assert_eq!(double_factorial(1), 1.0);
        assert_eq!(double_factorial(4), 8.0); // 4 × 2 = 8
        assert_eq!(double_factorial(5), 15.0); // 5 × 3 × 1 = 15
        assert_eq!(double_factorial(6), 48.0); // 6 × 4 × 2 = 48
    }

    #[test]
    fn test_tensor_wrappers() {
        let input = Tensor::from_data(vec![0.5, 1.0], vec![2], DeviceType::Cpu).unwrap();

        // Test Padé approximant tensor
        let p_coeffs = [1.0, 1.0];
        let q_coeffs = [1.0, -1.0];
        let result = pade_approximant_tensor(&input, &p_coeffs, &q_coeffs).unwrap();
        let data = result.data().unwrap();

        assert_relative_eq!(data[0], 3.0, max_relative = 1e-6); // (1+0.5)/(1-0.5) = 3
                                                                // For (1+1)/(1-1), denominator is 0, so result should be large or infinite
        assert!(data[1].abs() > 1e10 || data[1].is_infinite() || data[1].is_nan());
    }
}
