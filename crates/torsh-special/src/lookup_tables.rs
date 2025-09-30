//! Lookup Tables for Special Functions
//!
//! This module provides precomputed lookup tables for commonly used special function values
//! to improve performance for repeated calculations of the same inputs.

use crate::TorshResult;
use lazy_static::lazy_static;
use std::collections::HashMap;
use torsh_tensor::Tensor;

lazy_static! {
    /// Precomputed gamma function values for integer inputs
    static ref GAMMA_INTEGER_TABLE: HashMap<i32, f64> = {
    let mut table = HashMap::new();

    // Γ(n) = (n-1)! for positive integers n
    table.insert(1, 1.0);  // Γ(1) = 0! = 1
    table.insert(2, 1.0);  // Γ(2) = 1! = 1
    table.insert(3, 2.0);  // Γ(3) = 2! = 2
    table.insert(4, 6.0);  // Γ(4) = 3! = 6
    table.insert(5, 24.0); // Γ(5) = 4! = 24
    table.insert(6, 120.0); // Γ(6) = 5! = 120
    table.insert(7, 720.0); // Γ(7) = 6! = 720
    table.insert(8, 5040.0); // Γ(8) = 7! = 5040
    table.insert(9, 40320.0); // Γ(9) = 8! = 40320
    table.insert(10, 362880.0); // Γ(10) = 9! = 362880

    // Half-integer values: Γ(n + 1/2) = (2n-1)!! * √π / 2^n
    table.insert(-1, f64::INFINITY); // Γ(1/2) = √π but using half-integer key

    table
    };
}

lazy_static! {
    /// Precomputed error function values for common inputs
    static ref ERF_COMMON_VALUES: HashMap<i32, f64> = {
    let mut table = HashMap::new();

    // Common erf values (scaled by 1000 for integer keys)
    table.insert(0, 0.0);              // erf(0) = 0
    table.insert(1000, 0.8427007929);  // erf(1) ≈ 0.8427
    table.insert(2000, 0.9953222650);  // erf(2) ≈ 0.9953
    table.insert(3000, 0.9999779095);  // erf(3) ≈ 0.99998
    table.insert(-1000, -0.8427007929); // erf(-1) ≈ -0.8427
    table.insert(-2000, -0.9953222650); // erf(-2) ≈ -0.9953
    table.insert(-3000, -0.9999779095); // erf(-3) ≈ -0.99998

    table
    };
}

lazy_static! {
    /// Precomputed Bessel J₀ values for common inputs
    static ref BESSEL_J0_VALUES: HashMap<i32, f64> = {
    let mut table = HashMap::new();

    // Common J₀ values (scaled by 1000 for integer keys)
    table.insert(0, 1.0);               // J₀(0) = 1
    table.insert(1000, 0.7651976866);   // J₀(1) ≈ 0.7652
    table.insert(2000, 0.2238907791);   // J₀(2) ≈ 0.2239
    table.insert(3000, -0.2600519549);  // J₀(3) ≈ -0.2601
    table.insert(5000, -0.1775967713);  // J₀(5) ≈ -0.1776
    table.insert(10000, -0.2459357645); // J₀(10) ≈ -0.2459

    table
    };
}

lazy_static! {
    /// Precomputed factorial values
    static ref FACTORIAL_TABLE: Vec<f64> = {
    vec![
        1.0,          // 0!
        1.0,          // 1!
        2.0,          // 2!
        6.0,          // 3!
        24.0,         // 4!
        120.0,        // 5!
        720.0,        // 6!
        5040.0,       // 7!
        40320.0,      // 8!
        362880.0,     // 9!
        3628800.0,    // 10!
        39916800.0,   // 11!
        479001600.0,  // 12!
        6227020800.0, // 13!
        87178291200.0, // 14!
        1307674368000.0, // 15!
        20922789888000.0, // 16!
        355687428096000.0, // 17!
        6402373705728000.0, // 18!
        121645100408832000.0, // 19!
        2432902008176640000.0, // 20!
    ]
    };
}

/// Fast lookup for gamma function values
pub fn gamma_lookup(input: &Tensor<f32>) -> Option<Tensor<f32>> {
    let data = input.data().ok()?;
    let mut result_data = Vec::with_capacity(data.len());

    for &val in data.iter() {
        let int_val = val.round() as i32;
        if (val - int_val as f32).abs() < 1e-6 && (1..=10).contains(&int_val) {
            if let Some(&gamma_val) = GAMMA_INTEGER_TABLE.get(&int_val) {
                result_data.push(gamma_val as f32);
            } else {
                return None; // Miss in lookup table
            }
        } else {
            return None; // Not suitable for lookup
        }
    }

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device()).ok()
}

/// Fast lookup for error function values
pub fn erf_lookup(input: &Tensor<f32>) -> Option<Tensor<f32>> {
    let data = input.data().ok()?;
    let mut result_data = Vec::with_capacity(data.len());

    for &val in data.iter() {
        let scaled_val = (val * 1000.0).round() as i32;
        if (val - scaled_val as f32 / 1000.0).abs() < 1e-6 {
            if let Some(&erf_val) = ERF_COMMON_VALUES.get(&scaled_val) {
                result_data.push(erf_val as f32);
            } else {
                return None; // Miss in lookup table
            }
        } else {
            return None; // Not suitable for lookup
        }
    }

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device()).ok()
}

/// Fast lookup for Bessel J₀ values
pub fn bessel_j0_lookup(input: &Tensor<f32>) -> Option<Tensor<f32>> {
    let data = input.data().ok()?;
    let mut result_data = Vec::with_capacity(data.len());

    for &val in data.iter() {
        let scaled_val = (val * 1000.0).round() as i32;
        if (val - scaled_val as f32 / 1000.0).abs() < 1e-6 {
            if let Some(&j0_val) = BESSEL_J0_VALUES.get(&scaled_val) {
                result_data.push(j0_val as f32);
            } else {
                return None; // Miss in lookup table
            }
        } else {
            return None; // Not suitable for lookup
        }
    }

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device()).ok()
}

/// Fast factorial computation using lookup table
pub fn factorial(n: usize) -> Option<f64> {
    FACTORIAL_TABLE.get(n).copied()
}

/// Enhanced gamma function with lookup optimization
pub fn gamma_optimized(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // Try lookup first
    if let Some(result) = gamma_lookup(input) {
        return Ok(result);
    }

    // Fall back to computation
    crate::gamma(input)
}

/// Enhanced error function with lookup optimization
pub fn erf_optimized(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // Try lookup first
    if let Some(result) = erf_lookup(input) {
        return Ok(result);
    }

    // Fall back to computation
    crate::erf(input)
}

/// Enhanced Bessel J₀ function with lookup optimization
pub fn bessel_j0_optimized(input: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
    // Try lookup first
    if let Some(result) = bessel_j0_lookup(input) {
        return Ok(result);
    }

    // Fall back to computation
    crate::bessel_j0(input)
}

/// Polynomial coefficient lookup tables for approximations
pub struct PolynomialCoeffs {
    pub gamma_stirling: &'static [f64],
    pub erf_rational: &'static [f64],
    pub bessel_asymptotic: &'static [f64],
}

/// Precomputed polynomial coefficients for various approximations
pub static POLY_COEFFS: PolynomialCoeffs = PolynomialCoeffs {
    // Stirling approximation coefficients for gamma function
    gamma_stirling: &[
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
    ],

    // Rational approximation coefficients for error function
    erf_rational: &[
        1.26551223,
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277,
    ],

    // Asymptotic expansion coefficients for Bessel functions
    bessel_asymptotic: &[
        1.0,
        -0.125,
        0.0703125,
        -0.0732421875,
        0.1123046875,
        -0.2271080017,
        0.5725014209,
        -1.7277275562,
        6.0740420012,
        -24.3805296324,
    ],
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_gamma_lookup() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4], device)?;

        let lookup_result = gamma_lookup(&x).unwrap();
        let data = lookup_result.data()?;

        assert_relative_eq!(data[0], 1.0, epsilon = 1e-6); // Γ(1) = 1
        assert_relative_eq!(data[1], 1.0, epsilon = 1e-6); // Γ(2) = 1
        assert_relative_eq!(data[2], 2.0, epsilon = 1e-6); // Γ(3) = 2
        assert_relative_eq!(data[3], 6.0, epsilon = 1e-6); // Γ(4) = 6
        Ok(())
    }

    #[test]
    fn test_erf_lookup() -> TorshResult<()> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], device)?;

        let lookup_result = erf_lookup(&x).unwrap();
        let data = lookup_result.data()?;

        assert_relative_eq!(data[0], 0.0, epsilon = 1e-6); // erf(0) = 0
        assert_relative_eq!(data[1], 0.842_700_8, epsilon = 1e-6); // erf(1)
        assert_relative_eq!(data[2], -0.842_700_8, epsilon = 1e-6); // erf(-1)
        Ok(())
    }

    #[test]
    fn test_factorial_lookup() -> TorshResult<()> {
        assert_eq!(factorial(0), Some(1.0));
        assert_eq!(factorial(1), Some(1.0));
        assert_eq!(factorial(5), Some(120.0));
        assert_eq!(factorial(10), Some(3628800.0));
        assert_eq!(factorial(25), None); // Beyond table
        Ok(())
    }

    #[test]
    fn test_optimized_functions() -> TorshResult<()> {
        let device = DeviceType::Cpu;

        // Test gamma optimization
        let x = Tensor::from_data(vec![3.0, 4.0], vec![2], device)?;
        let result = gamma_optimized(&x)?;
        let data = result.data()?;
        assert_relative_eq!(data[0], 2.0, epsilon = 1e-5);
        assert_relative_eq!(data[1], 6.0, epsilon = 1e-5);

        // Test erf optimization
        let x = Tensor::from_data(vec![0.0, 1.0], vec![2], device)?;
        let result = erf_optimized(&x)?;
        let data = result.data()?;
        assert_relative_eq!(data[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(data[1], 0.842_700_8, epsilon = 1e-3);
        Ok(())
    }
}
