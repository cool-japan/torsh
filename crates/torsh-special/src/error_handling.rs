//! Enhanced error handling and edge case coverage for special functions
//!
//! This module provides comprehensive error handling, input validation,
//! and special case handling for all special functions.

use crate::TorshResult;
use std::f32;
use std::f64;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Comprehensive input validation for special functions
pub trait InputValidation<T> {
    /// Validate input and provide detailed error messages
    fn validate_input(&self, function_name: &str) -> Result<()>;

    /// Check for NaN values
    fn check_nan(&self, function_name: &str) -> Result<()>;

    /// Check for infinite values
    fn check_infinite(&self, function_name: &str) -> Result<()>;

    /// Check for domain-specific constraints
    fn check_domain(&self, function_name: &str, constraints: &DomainConstraints) -> Result<()>;
}

/// Domain constraints for special functions
#[derive(Debug, Clone)]
pub struct DomainConstraints {
    /// Minimum allowed value (inclusive)
    pub min_value: Option<f64>,
    /// Maximum allowed value (inclusive)
    pub max_value: Option<f64>,
    /// Whether zero is allowed
    pub allow_zero: bool,
    /// Whether negative values are allowed
    pub allow_negative: bool,
    /// Whether infinite values are allowed
    pub allow_infinite: bool,
    /// Custom validation function
    pub custom_validator: Option<fn(f64) -> bool>,
    /// Custom error message for domain violations
    pub custom_error_message: Option<String>,
}

impl Default for DomainConstraints {
    fn default() -> Self {
        DomainConstraints {
            min_value: None,
            max_value: None,
            allow_zero: true,
            allow_negative: true,
            allow_infinite: false,
            custom_validator: None,
            custom_error_message: None,
        }
    }
}

impl DomainConstraints {
    /// Create constraints for gamma function
    pub fn gamma() -> Self {
        DomainConstraints {
            min_value: None,
            max_value: Some(170.0), // Avoid overflow
            allow_zero: false,
            allow_negative: true, // Gamma is defined for negative non-integers
            allow_infinite: false,
            custom_validator: Some(|x| {
                // Check if x is a negative integer
                if x < 0.0 && x.fract() == 0.0 {
                    false // Gamma is undefined at negative integers
                } else {
                    true
                }
            }),
            custom_error_message: Some(
                "Gamma function is undefined at negative integers and zero".to_string(),
            ),
        }
    }

    /// Create constraints for logarithmic functions
    pub fn logarithmic() -> Self {
        DomainConstraints {
            min_value: Some(0.0),
            max_value: None,
            allow_zero: false,
            allow_negative: false,
            allow_infinite: true,
            custom_validator: None,
            custom_error_message: Some(
                "Logarithmic functions require positive arguments".to_string(),
            ),
        }
    }

    /// Create constraints for Bessel functions
    pub fn bessel() -> Self {
        DomainConstraints {
            min_value: None,
            max_value: Some(700.0), // Avoid numerical overflow
            allow_zero: true,
            allow_negative: true,
            allow_infinite: false,
            custom_validator: None,
            custom_error_message: Some(
                "Bessel function argument too large, may cause numerical overflow".to_string(),
            ),
        }
    }

    /// Create constraints for inverse functions
    pub fn inverse_function(min: f64, max: f64) -> Self {
        DomainConstraints {
            min_value: Some(min),
            max_value: Some(max),
            allow_zero: true,
            allow_negative: min < 0.0,
            allow_infinite: false,
            custom_validator: None,
            custom_error_message: Some(format!("Input must be in range [{min}, {max}]")),
        }
    }
}

impl InputValidation<f32> for Tensor<f32> {
    fn validate_input(&self, function_name: &str) -> Result<()> {
        self.check_nan(function_name)?;
        self.check_infinite(function_name)?;
        Ok(())
    }

    fn check_nan(&self, function_name: &str) -> Result<()> {
        let data = self.data()?;
        for (i, &value) in data.iter().enumerate() {
            if value.is_nan() {
                return Err(TorshError::InvalidArgument(format!(
                    "{function_name}: Input contains NaN at index {i}"
                )));
            }
        }
        Ok(())
    }

    fn check_infinite(&self, function_name: &str) -> Result<()> {
        let data = self.data()?;
        for (i, &value) in data.iter().enumerate() {
            if value.is_infinite() {
                return Err(TorshError::InvalidArgument(format!(
                    "{function_name}: Input contains infinite value at index {i}"
                )));
            }
        }
        Ok(())
    }

    fn check_domain(&self, function_name: &str, constraints: &DomainConstraints) -> Result<()> {
        let data = self.data()?;

        for (i, &value) in data.iter().enumerate() {
            let value_f64 = value as f64;

            // Check minimum constraint
            if let Some(min) = constraints.min_value {
                if value_f64 < min {
                    return Err(TorshError::InvalidArgument(format!(
                        "{function_name}: Value {value} at index {i} is below minimum {min}"
                    )));
                }
            }

            // Check maximum constraint
            if let Some(max) = constraints.max_value {
                if value_f64 > max {
                    return Err(TorshError::InvalidArgument(format!(
                        "{function_name}: Value {value} at index {i} exceeds maximum {max}"
                    )));
                }
            }

            // Check zero constraint
            if !constraints.allow_zero && value_f64 == 0.0 {
                return Err(TorshError::InvalidArgument(format!(
                    "{function_name}: Zero value not allowed at index {i}"
                )));
            }

            // Check negative constraint
            if !constraints.allow_negative && value_f64 < 0.0 {
                return Err(TorshError::InvalidArgument(format!(
                    "{function_name}: Negative value {value} not allowed at index {i}"
                )));
            }

            // Check infinite constraint
            if !constraints.allow_infinite && value.is_infinite() {
                return Err(TorshError::InvalidArgument(format!(
                    "{function_name}: Infinite value not allowed at index {i}"
                )));
            }

            // Check custom validator
            if let Some(validator) = constraints.custom_validator {
                if !validator(value_f64) {
                    let message = constraints
                        .custom_error_message
                        .clone()
                        .unwrap_or_else(|| format!("Custom validation failed for value {value}"));
                    return Err(TorshError::InvalidArgument(format!(
                        "{function_name}: {message} at index {i}"
                    )));
                }
            }
        }

        Ok(())
    }
}

/// Enhanced wrapper functions with comprehensive error handling
pub mod safe_functions {
    use super::*;
    use crate::{bessel_j0_scirs2, bessel_j1_scirs2, bessel_y0_scirs2, bessel_y1_scirs2};
    use crate::{erf as erf_impl, erfc as erfc_impl, gamma as gamma_impl, lgamma as lgamma_impl};

    /// Safe gamma function with comprehensive error handling
    pub fn safe_gamma(input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.validate_input("gamma")?;
        input.check_domain("gamma", &DomainConstraints::gamma())?;

        // Handle special cases
        let data = input.data()?;
        let mut has_special_cases = false;
        for &value in data.iter() {
            if value < 0.0 && (value - value.round()).abs() < 1e-10 {
                has_special_cases = true;
                break;
            }
        }

        if has_special_cases {
            return Err(TorshError::InvalidArgument(
                "gamma: Cannot compute gamma for negative integers".to_string(),
            ));
        }

        gamma_impl(input)
    }

    /// Safe log gamma function with comprehensive error handling
    pub fn safe_lgamma(input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.validate_input("lgamma")?;
        input.check_domain("lgamma", &DomainConstraints::gamma())?;

        lgamma_impl(input)
    }

    /// Safe error function with comprehensive error handling
    pub fn safe_erf(input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.validate_input("erf")?;

        // Check for extremely large values that might cause numerical issues
        let data = input.data()?;
        for (i, &value) in data.iter().enumerate() {
            if value.abs() > 100.0 {
                return Err(TorshError::InvalidArgument(format!(
                    "erf: Input value {value} at index {i} may cause numerical instability"
                )));
            }
        }

        erf_impl(input)
    }

    /// Safe complementary error function with comprehensive error handling
    pub fn safe_erfc(input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.validate_input("erfc")?;

        // Check for extremely large positive values that might underflow
        let data = input.data()?;
        for (i, &value) in data.iter().enumerate() {
            if value > 100.0 {
                return Err(TorshError::InvalidArgument(format!(
                    "erfc: Input value {value} at index {i} may cause underflow to zero"
                )));
            }
            if value < -100.0 {
                return Err(TorshError::InvalidArgument(format!(
                    "erfc: Input value {value} at index {i} may cause overflow"
                )));
            }
        }

        erfc_impl(input)
    }

    /// Safe Bessel J0 function with comprehensive error handling
    pub fn safe_bessel_j0(input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.validate_input("bessel_j0")?;
        input.check_domain("bessel_j0", &DomainConstraints::bessel())?;

        bessel_j0_scirs2(input)
    }

    /// Safe Bessel J1 function with comprehensive error handling
    pub fn safe_bessel_j1(input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.validate_input("bessel_j1")?;
        input.check_domain("bessel_j1", &DomainConstraints::bessel())?;

        bessel_j1_scirs2(input)
    }

    /// Safe Bessel Y0 function with comprehensive error handling
    pub fn safe_bessel_y0(input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.validate_input("bessel_y0")?;

        // Y0 is undefined for x <= 0
        let constraints = DomainConstraints {
            min_value: Some(f64::EPSILON),
            max_value: Some(700.0),
            allow_zero: false,
            allow_negative: false,
            allow_infinite: false,
            custom_validator: None,
            custom_error_message: Some("Bessel Y0 requires positive arguments".to_string()),
        };
        input.check_domain("bessel_y0", &constraints)?;

        bessel_y0_scirs2(input)
    }

    /// Safe Bessel Y1 function with comprehensive error handling
    pub fn safe_bessel_y1(input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.validate_input("bessel_y1")?;

        // Y1 is undefined for x <= 0
        let constraints = DomainConstraints {
            min_value: Some(f64::EPSILON),
            max_value: Some(700.0),
            allow_zero: false,
            allow_negative: false,
            allow_infinite: false,
            custom_validator: None,
            custom_error_message: Some("Bessel Y1 requires positive arguments".to_string()),
        };
        input.check_domain("bessel_y1", &constraints)?;

        bessel_y1_scirs2(input)
    }
}

/// Error recovery strategies for numerical edge cases
pub mod error_recovery {
    use super::*;

    /// Attempt to recover from numerical overflow by clamping values
    pub fn clamp_to_finite(tensor: &Tensor<f32>, max_value: f32) -> TorshResult<Tensor<f32>> {
        let data = tensor.data()?;
        let clamped_data: Vec<f32> = data
            .iter()
            .map(|&x| {
                if x.is_infinite() {
                    if x.is_sign_positive() {
                        max_value
                    } else {
                        -max_value
                    }
                } else if x.is_nan() {
                    0.0
                } else {
                    x.clamp(-max_value, max_value)
                }
            })
            .collect();

        Tensor::from_data(
            clamped_data,
            tensor.shape().dims().to_vec(),
            tensor.device(),
        )
    }

    /// Replace problematic values with safe defaults
    pub fn replace_problematic_values(
        tensor: &Tensor<f32>,
        default_value: f32,
    ) -> TorshResult<Tensor<f32>> {
        let data = tensor.data()?;
        let safe_data: Vec<f32> = data
            .iter()
            .map(|&x| if x.is_finite() { x } else { default_value })
            .collect();

        Tensor::from_data(safe_data, tensor.shape().dims().to_vec(), tensor.device())
    }

    /// Apply gradual clamping for overflow protection
    pub fn gradual_clamp(
        tensor: &Tensor<f32>,
        warning_threshold: f32,
        max_value: f32,
    ) -> TorshResult<(Tensor<f32>, Vec<usize>)> {
        let data = tensor.data()?;
        let mut clamped_indices = Vec::new();

        let clamped_data: Vec<f32> = data
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if x.abs() > warning_threshold {
                    clamped_indices.push(i);
                    if x > max_value {
                        max_value
                    } else if x < -max_value {
                        -max_value
                    } else {
                        x
                    }
                } else {
                    x
                }
            })
            .collect();

        let result = Tensor::from_data(
            clamped_data,
            tensor.shape().dims().to_vec(),
            tensor.device(),
        )?;
        Ok((result, clamped_indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_input_validation() -> TorshResult<()> {
        // Test NaN detection
        let input_with_nan =
            Tensor::from_data(vec![1.0f32, f32::NAN, 3.0], vec![3], DeviceType::Cpu)?;
        assert!(input_with_nan.check_nan("test").is_err());

        // Test infinite detection
        let input_with_inf =
            Tensor::from_data(vec![1.0f32, f32::INFINITY, 3.0], vec![3], DeviceType::Cpu)?;
        assert!(input_with_inf.check_infinite("test").is_err());

        // Test valid input
        let valid_input = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
        assert!(valid_input.validate_input("test").is_ok());
        Ok(())
    }

    #[test]
    fn test_domain_constraints() -> TorshResult<()> {
        let input = Tensor::from_data(vec![-1.0f32, 0.0, 1.0], vec![3], DeviceType::Cpu)?;

        // Test logarithmic constraints (positive only)
        let log_constraints = DomainConstraints::logarithmic();
        assert!(input.check_domain("test", &log_constraints).is_err());

        // Test gamma constraints (allows negative non-integers)
        let gamma_constraints = DomainConstraints::gamma();
        assert!(input.check_domain("test", &gamma_constraints).is_err()); // zero not allowed
        Ok(())
    }

    #[test]
    fn test_safe_functions() -> TorshResult<()> {
        use super::safe_functions::*;

        // Test safe gamma with valid input
        let valid_input = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
        assert!(safe_gamma(&valid_input).is_ok());

        // Test safe gamma with invalid input (negative integer)
        let invalid_input = Tensor::from_data(vec![-1.0f32], vec![1], DeviceType::Cpu)?;
        assert!(safe_gamma(&invalid_input).is_err());
        Ok(())
    }

    #[test]
    fn test_error_recovery() -> TorshResult<()> {
        use super::error_recovery::*;

        let problematic_input = Tensor::from_data(
            vec![1.0f32, f32::INFINITY, f32::NAN, -f32::INFINITY],
            vec![4],
            DeviceType::Cpu,
        )?;

        // Test clamping
        let clamped = clamp_to_finite(&problematic_input, 100.0)?;
        let clamped_data = clamped.data()?;
        assert!(clamped_data.iter().all(|&x| x.is_finite()));

        // Test replacement
        let replaced = replace_problematic_values(&problematic_input, 0.0)?;
        let replaced_data = replaced.data()?;
        assert!(replaced_data.iter().all(|&x| x.is_finite()));
        Ok(())
    }
}
