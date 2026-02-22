//! Comprehensive gradient validation for shape and type checking
//!
//! This module provides validation utilities to ensure gradient correctness
//! across tensor operations, preventing common bugs and providing detailed
//! error reporting.

use crate::{AutogradTensor, Result};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use std::collections::HashMap;
use torsh_core::dtype::TensorElement;
use torsh_core::error::TorshError;
use torsh_core::shape::Shape;

/// Configuration for gradient validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable shape validation
    pub validate_shapes: bool,
    /// Enable type validation  
    pub validate_types: bool,
    /// Enable value range validation
    pub validate_ranges: bool,
    /// Enable finite value checking (no NaN, Inf)
    pub validate_finite: bool,
    /// Tolerance for shape mismatches
    pub shape_tolerance: f64,
    /// Maximum allowed gradient magnitude
    pub max_gradient_magnitude: Option<f64>,
    /// Minimum allowed gradient magnitude
    pub min_gradient_magnitude: Option<f64>,
    /// Whether to throw exceptions on validation failures
    pub strict_mode: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validate_shapes: true,
            validate_types: true,
            validate_ranges: true,
            validate_finite: true,
            shape_tolerance: 1e-10,
            max_gradient_magnitude: Some(1e6),
            min_gradient_magnitude: Some(1e-12),
            strict_mode: true,
        }
    }
}

/// Result of gradient validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Specific validation errors
    pub errors: Vec<ValidationError>,
    /// Warnings (non-fatal issues)
    pub warnings: Vec<ValidationWarning>,
    /// Statistics about the validated gradients
    pub statistics: GradientStatistics,
    /// Detailed per-tensor validation results
    pub tensor_results: HashMap<String, TensorValidationResult>,
}

/// Validation error with detailed information
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,
    /// Tensor or parameter name
    pub tensor_name: String,
    /// Expected value/shape
    pub expected: String,
    /// Actual value/shape found
    pub actual: String,
    /// Additional context
    pub context: String,
    /// Severity level
    pub severity: ErrorSeverity,
}

/// Types of validation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationErrorType {
    /// Shape mismatch between tensor and gradient
    ShapeMismatch,
    /// Type mismatch between tensor and gradient
    TypeMismatch,
    /// Invalid gradient values (NaN, Inf)
    InvalidValues,
    /// Gradient magnitude out of acceptable range
    MagnitudeOutOfRange,
    /// Gradient shape incompatible with operation
    IncompatibleShape,
    /// Missing gradient for tensor requiring grad
    MissingGradient,
    /// Unexpected gradient for tensor not requiring grad
    UnexpectedGradient,
    /// Dimension mismatch in gradient computation
    DimensionMismatch,
    /// Edge case: empty tensor
    EmptyTensor,
    /// Edge case: zero-dimensional tensor
    ZeroDimensional,
    /// Edge case: tensor with zero elements but non-zero dimensions
    ZeroElements,
    /// Edge case: extremely large tensor that may cause memory issues
    OversizedTensor,
    /// Edge case: negative dimensions
    NegativeDimensions,
    /// Edge case: singular matrices in operations requiring inversion
    SingularMatrix,
    /// Edge case: broadcast incompatibility
    BroadcastIncompatible,
}

/// Validation warning (non-fatal)
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning type
    pub warning_type: ValidationWarningType,
    /// Tensor name
    pub tensor_name: String,
    /// Warning message
    pub message: String,
    /// Suggested action
    pub suggestion: String,
}

/// Types of validation warnings
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationWarningType {
    /// Gradient magnitude is suspiciously large
    LargeGradientMagnitude,
    /// Gradient magnitude is suspiciously small
    SmallGradientMagnitude,
    /// Gradient contains many zeros
    SparseGradient,
    /// Gradient has unusual distribution
    UnusualDistribution,
    /// Performance concern
    PerformanceConcern,
    /// Edge case detected but handled
    EdgeCaseHandled,
    /// Potential numerical instability
    NumericalInstability,
    /// Memory usage concern
    MemoryConcern,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Statistics about validated gradients
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    /// Total number of tensors validated
    pub total_tensors: usize,
    /// Number of tensors with valid gradients
    pub valid_gradients: usize,
    /// Number of tensors with invalid gradients
    pub invalid_gradients: usize,
    /// Total gradient elements
    pub total_elements: usize,
    /// Number of zero gradients
    pub zero_elements: usize,
    /// Number of infinite gradients
    pub infinite_elements: usize,
    /// Number of NaN gradients
    pub nan_elements: usize,
    /// Mean gradient magnitude
    pub mean_magnitude: f64,
    /// Maximum gradient magnitude
    pub max_magnitude: f64,
    /// Minimum gradient magnitude
    pub min_magnitude: f64,
    /// Standard deviation of gradient magnitudes
    pub std_magnitude: f64,
}

/// Validation result for a single tensor
#[derive(Debug, Clone)]
pub struct TensorValidationResult {
    /// Tensor name
    pub tensor_name: String,
    /// Whether this tensor passed validation
    pub passed: bool,
    /// Shape validation result
    pub shape_valid: bool,
    /// Type validation result
    pub type_valid: bool,
    /// Value validation result
    pub values_valid: bool,
    /// Specific errors for this tensor
    pub errors: Vec<ValidationError>,
    /// Warnings for this tensor
    pub warnings: Vec<ValidationWarning>,
    /// Gradient statistics for this tensor
    pub gradient_stats: TensorGradientStats,
}

/// Gradient statistics for a single tensor
#[derive(Debug, Clone)]
pub struct TensorGradientStats {
    /// Tensor shape
    pub shape: Shape,
    /// Number of elements
    pub num_elements: usize,
    /// Number of zero elements
    pub zero_elements: usize,
    /// Number of finite elements
    pub finite_elements: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min_value: f64,
    /// Maximum value
    pub max_value: f64,
    /// L1 norm
    pub l1_norm: f64,
    /// L2 norm
    pub l2_norm: f64,
}

/// Comprehensive gradient validator
pub struct GradientValidator {
    config: ValidationConfig,
}

impl GradientValidator {
    /// Create a new gradient validator with default configuration
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create a new gradient validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate gradients for a set of tensors
    pub fn validate_gradients<T>(
        &self,
        tensors: &[(&str, &dyn AutogradTensor<T>)],
        gradients: &[(&str, Option<&[T]>)],
    ) -> Result<ValidationResult>
    where
        T: TensorElement + Float + ToPrimitive + FromPrimitive + std::fmt::Debug,
    {
        let mut result = ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            statistics: GradientStatistics {
                total_tensors: tensors.len(),
                valid_gradients: 0,
                invalid_gradients: 0,
                total_elements: 0,
                zero_elements: 0,
                infinite_elements: 0,
                nan_elements: 0,
                mean_magnitude: 0.0,
                max_magnitude: 0.0,
                min_magnitude: f64::INFINITY,
                std_magnitude: 0.0,
            },
            tensor_results: HashMap::new(),
        };

        let mut all_magnitudes = Vec::new();

        // Validate each tensor-gradient pair
        for ((tensor_name, tensor), (grad_name, gradient)) in tensors.iter().zip(gradients.iter()) {
            if tensor_name != grad_name {
                result.errors.push(ValidationError {
                    error_type: ValidationErrorType::MissingGradient,
                    tensor_name: tensor_name.to_string(),
                    expected: tensor_name.to_string(),
                    actual: grad_name.to_string(),
                    context: "Tensor and gradient names do not match".to_string(),
                    severity: ErrorSeverity::High,
                });
                result.passed = false;
                continue;
            }

            let tensor_result = self.validate_single_tensor(tensor_name, *tensor, gradient)?;

            if !tensor_result.passed {
                result.passed = false;
                result.statistics.invalid_gradients += 1;
            } else {
                result.statistics.valid_gradients += 1;
            }

            // Accumulate statistics
            result.statistics.total_elements += tensor_result.gradient_stats.num_elements;
            result.statistics.zero_elements += tensor_result.gradient_stats.zero_elements;

            // Collect magnitude statistics
            if tensor_result.gradient_stats.finite_elements > 0 {
                all_magnitudes.push(tensor_result.gradient_stats.mean.abs());
                result.statistics.max_magnitude = result
                    .statistics
                    .max_magnitude
                    .max(tensor_result.gradient_stats.max_value.abs());
                result.statistics.min_magnitude = result
                    .statistics
                    .min_magnitude
                    .min(tensor_result.gradient_stats.min_value.abs());
            }

            // Collect errors and warnings
            result.errors.extend(tensor_result.errors.clone());
            result.warnings.extend(tensor_result.warnings.clone());

            result
                .tensor_results
                .insert(tensor_name.to_string(), tensor_result);
        }

        // Compute overall statistics
        if !all_magnitudes.is_empty() {
            result.statistics.mean_magnitude =
                all_magnitudes.iter().sum::<f64>() / all_magnitudes.len() as f64;

            let variance = all_magnitudes
                .iter()
                .map(|&mag| (mag - result.statistics.mean_magnitude).powi(2))
                .sum::<f64>()
                / all_magnitudes.len() as f64;
            result.statistics.std_magnitude = variance.sqrt();
        }

        // Validate overall consistency
        self.validate_overall_consistency(&mut result)?;

        // Apply strict mode if enabled
        if self.config.strict_mode && !result.passed {
            return Err(TorshError::AutogradError(format!(
                "Gradient validation failed: {} errors found",
                result.errors.len()
            )));
        }

        Ok(result)
    }

    /// Validate a single tensor and its gradient
    fn validate_single_tensor<T>(
        &self,
        tensor_name: &str,
        tensor: &dyn AutogradTensor<T>,
        gradient: &Option<&[T]>,
    ) -> Result<TensorValidationResult>
    where
        T: TensorElement + Float + ToPrimitive + FromPrimitive + std::fmt::Debug,
    {
        let mut result = TensorValidationResult {
            tensor_name: tensor_name.to_string(),
            passed: true,
            shape_valid: true,
            type_valid: true,
            values_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            gradient_stats: TensorGradientStats {
                shape: tensor.shape(),
                num_elements: 0,
                zero_elements: 0,
                finite_elements: 0,
                mean: 0.0,
                std_dev: 0.0,
                min_value: f64::INFINITY,
                max_value: f64::NEG_INFINITY,
                l1_norm: 0.0,
                l2_norm: 0.0,
            },
        };

        // Check if gradient exists when it should
        if tensor.requires_grad() && gradient.is_none() {
            result.errors.push(ValidationError {
                error_type: ValidationErrorType::MissingGradient,
                tensor_name: tensor_name.to_string(),
                expected: "gradient present".to_string(),
                actual: "gradient missing".to_string(),
                context: "Tensor requires gradient but none provided".to_string(),
                severity: ErrorSeverity::High,
            });
            result.passed = false;
            return Ok(result);
        }

        // Check if gradient exists when it shouldn't
        if !tensor.requires_grad() && gradient.is_some() {
            result.warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::PerformanceConcern,
                tensor_name: tensor_name.to_string(),
                message: "Gradient provided for tensor that doesn't require grad".to_string(),
                suggestion: "Consider not computing gradients for this tensor".to_string(),
            });
        }

        if let Some(grad_data) = gradient {
            // Validate shape
            if self.config.validate_shapes {
                self.validate_shape(tensor, grad_data, &mut result)?;
            }

            // Validate types
            if self.config.validate_types {
                self.validate_types(tensor, grad_data, &mut result)?;
            }

            // Validate values
            if self.config.validate_ranges || self.config.validate_finite {
                self.validate_values(grad_data, &mut result)?;
            }

            // Compute gradient statistics
            self.compute_gradient_statistics(grad_data, &mut result)?;
        }

        Ok(result)
    }

    /// Validate gradient shape matches tensor shape
    fn validate_shape<T>(
        &self,
        tensor: &dyn AutogradTensor<T>,
        gradient: &[T],
        result: &mut TensorValidationResult,
    ) -> Result<()>
    where
        T: TensorElement,
    {
        let tensor_shape = tensor.shape();
        let expected_elements = tensor_shape.numel();
        let actual_elements = gradient.len();

        if expected_elements != actual_elements {
            result.errors.push(ValidationError {
                error_type: ValidationErrorType::ShapeMismatch,
                tensor_name: result.tensor_name.clone(),
                expected: format!("{} elements (shape: {:?})", expected_elements, tensor_shape),
                actual: format!("{} elements", actual_elements),
                context: "Gradient size doesn't match tensor size".to_string(),
                severity: ErrorSeverity::Critical,
            });
            result.shape_valid = false;
            result.passed = false;
        }

        Ok(())
    }

    /// Validate gradient types are compatible
    fn validate_types<T>(
        &self,
        _tensor: &dyn AutogradTensor<T>,
        _gradient: &[T],
        result: &mut TensorValidationResult,
    ) -> Result<()>
    where
        T: TensorElement,
    {
        // Type validation - in a real implementation, this would check
        // type compatibility between tensor and gradient
        result.type_valid = true;
        Ok(())
    }

    /// Validate gradient values are reasonable
    fn validate_values<T>(&self, gradient: &[T], result: &mut TensorValidationResult) -> Result<()>
    where
        T: TensorElement + Float + ToPrimitive,
    {
        for (i, &val) in gradient.iter().enumerate() {
            let val_f64 = <T as ToPrimitive>::to_f64(&val).unwrap_or(0.0);

            // Check for NaN and infinity
            if self.config.validate_finite {
                if val_f64.is_nan() {
                    result.errors.push(ValidationError {
                        error_type: ValidationErrorType::InvalidValues,
                        tensor_name: result.tensor_name.clone(),
                        expected: "finite value".to_string(),
                        actual: "NaN".to_string(),
                        context: format!("NaN found at index {}", i),
                        severity: ErrorSeverity::Critical,
                    });
                    result.values_valid = false;
                    result.passed = false;
                }

                if val_f64.is_infinite() {
                    result.errors.push(ValidationError {
                        error_type: ValidationErrorType::InvalidValues,
                        tensor_name: result.tensor_name.clone(),
                        expected: "finite value".to_string(),
                        actual: if val_f64.is_sign_positive() {
                            "+Inf"
                        } else {
                            "-Inf"
                        }
                        .to_string(),
                        context: format!("Infinite value found at index {}", i),
                        severity: ErrorSeverity::High,
                    });
                    result.values_valid = false;
                    result.passed = false;
                }
            }

            // Check magnitude bounds
            if self.config.validate_ranges {
                let magnitude = val_f64.abs();

                if let Some(max_mag) = self.config.max_gradient_magnitude {
                    if magnitude > max_mag {
                        result.errors.push(ValidationError {
                            error_type: ValidationErrorType::MagnitudeOutOfRange,
                            tensor_name: result.tensor_name.clone(),
                            expected: format!("magnitude <= {}", max_mag),
                            actual: format!("magnitude = {}", magnitude),
                            context: format!("Large gradient at index {}", i),
                            severity: ErrorSeverity::Medium,
                        });
                        result.values_valid = false;
                        result.passed = false;
                    }
                }

                if let Some(min_mag) = self.config.min_gradient_magnitude {
                    if magnitude > 0.0 && magnitude < min_mag {
                        result.warnings.push(ValidationWarning {
                            warning_type: ValidationWarningType::SmallGradientMagnitude,
                            tensor_name: result.tensor_name.clone(),
                            message: format!("Very small gradient magnitude: {}", magnitude),
                            suggestion: "Check for vanishing gradient problems".to_string(),
                        });
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute detailed statistics for gradient values
    fn compute_gradient_statistics<T>(
        &self,
        gradient: &[T],
        result: &mut TensorValidationResult,
    ) -> Result<()>
    where
        T: TensorElement + Float + ToPrimitive,
    {
        result.gradient_stats.num_elements = gradient.len();

        if gradient.is_empty() {
            return Ok(());
        }

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut l1_norm = 0.0;
        let mut l2_norm = 0.0;
        let mut zero_count = 0;
        let mut finite_count = 0;
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &val in gradient {
            let val_f64 = <T as ToPrimitive>::to_f64(&val).unwrap_or(0.0);

            if val_f64.abs() < f64::EPSILON {
                zero_count += 1;
            }

            if val_f64.is_finite() {
                finite_count += 1;
                sum += val_f64;
                sum_sq += val_f64 * val_f64;
                l1_norm += val_f64.abs();
                l2_norm += val_f64 * val_f64;
                min_val = min_val.min(val_f64);
                max_val = max_val.max(val_f64);
            }
        }

        result.gradient_stats.zero_elements = zero_count;
        result.gradient_stats.finite_elements = finite_count;
        result.gradient_stats.l1_norm = l1_norm;
        result.gradient_stats.l2_norm = l2_norm.sqrt();
        result.gradient_stats.min_value = if min_val.is_finite() { min_val } else { 0.0 };
        result.gradient_stats.max_value = if max_val.is_finite() { max_val } else { 0.0 };

        if finite_count > 0 {
            result.gradient_stats.mean = sum / finite_count as f64;
            let variance = (sum_sq / finite_count as f64) - result.gradient_stats.mean.powi(2);
            result.gradient_stats.std_dev = variance.max(0.0).sqrt();
        }

        // Generate warnings based on statistics
        let stats_clone = result.gradient_stats.clone();
        self.analyze_gradient_statistics(&stats_clone, &mut *result);

        Ok(())
    }

    /// Analyze gradient statistics and generate warnings
    fn analyze_gradient_statistics(
        &self,
        stats: &TensorGradientStats,
        result: &mut TensorValidationResult,
    ) {
        // Check for sparse gradients
        let sparsity_ratio = stats.zero_elements as f64 / stats.num_elements as f64;
        if sparsity_ratio > 0.9 {
            result.warnings.push(ValidationWarning {
                warning_type: ValidationWarningType::SparseGradient,
                tensor_name: result.tensor_name.clone(),
                message: format!("Gradient is {:.1}% sparse", sparsity_ratio * 100.0),
                suggestion: "Consider using sparse gradient optimization".to_string(),
            });
        }

        // Check for unusual distributions
        if stats.finite_elements > 1 && stats.std_dev > 0.0 {
            let coefficient_of_variation = stats.std_dev / stats.mean.abs();
            if coefficient_of_variation > 10.0 {
                result.warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::UnusualDistribution,
                    tensor_name: result.tensor_name.clone(),
                    message: "Gradient has high coefficient of variation".to_string(),
                    suggestion: "Check for gradient scaling issues".to_string(),
                });
            }
        }

        // Check for suspiciously large gradients
        if let Some(max_mag) = self.config.max_gradient_magnitude {
            if stats.max_value.abs() > max_mag * 0.8 {
                result.warnings.push(ValidationWarning {
                    warning_type: ValidationWarningType::LargeGradientMagnitude,
                    tensor_name: result.tensor_name.clone(),
                    message: format!("Large gradient magnitude: {}", stats.max_value.abs()),
                    suggestion: "Consider gradient clipping".to_string(),
                });
            }
        }
    }

    /// Validate overall consistency across all gradients
    fn validate_overall_consistency(&self, result: &mut ValidationResult) -> Result<()> {
        // Check for global issues
        let total_elements = result.statistics.total_elements;
        let nan_ratio = result.statistics.nan_elements as f64 / total_elements as f64;
        let inf_ratio = result.statistics.infinite_elements as f64 / total_elements as f64;

        if nan_ratio > 0.01 {
            result.errors.push(ValidationError {
                error_type: ValidationErrorType::InvalidValues,
                tensor_name: "global".to_string(),
                expected: "< 1% NaN values".to_string(),
                actual: format!("{:.2}% NaN values", nan_ratio * 100.0),
                context: "High proportion of NaN gradients detected".to_string(),
                severity: ErrorSeverity::Critical,
            });
            result.passed = false;
        }

        if inf_ratio > 0.001 {
            result.errors.push(ValidationError {
                error_type: ValidationErrorType::InvalidValues,
                tensor_name: "global".to_string(),
                expected: "< 0.1% infinite values".to_string(),
                actual: format!("{:.3}% infinite values", inf_ratio * 100.0),
                context: "High proportion of infinite gradients detected".to_string(),
                severity: ErrorSeverity::High,
            });
            result.passed = false;
        }

        Ok(())
    }

    /// Quick validation check for a single gradient
    pub fn quick_validate<T>(&self, _tensor_name: &str, gradient: &[T]) -> Result<bool>
    where
        T: TensorElement + Float + ToPrimitive,
    {
        for &val in gradient {
            let val_f64 = <T as ToPrimitive>::to_f64(&val).unwrap_or(0.0);

            if val_f64.is_nan() || val_f64.is_infinite() {
                return Ok(false);
            }

            if let Some(max_mag) = self.config.max_gradient_magnitude {
                if val_f64.abs() > max_mag {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    /// Generate a validation report
    pub fn generate_report(&self, result: &ValidationResult) -> String {
        let mut report = String::new();

        report.push_str("=== Gradient Validation Report ===\n\n");

        // Overall status
        report.push_str(&format!(
            "Status: {}\n",
            if result.passed { "PASSED" } else { "FAILED" }
        ));
        report.push_str(&format!(
            "Total tensors: {}\n",
            result.statistics.total_tensors
        ));
        report.push_str(&format!(
            "Valid gradients: {}\n",
            result.statistics.valid_gradients
        ));
        report.push_str(&format!(
            "Invalid gradients: {}\n",
            result.statistics.invalid_gradients
        ));

        // Statistics
        report.push_str("\n=== Statistics ===\n");
        report.push_str(&format!(
            "Total elements: {}\n",
            result.statistics.total_elements
        ));
        report.push_str(&format!(
            "Zero elements: {}\n",
            result.statistics.zero_elements
        ));
        report.push_str(&format!(
            "Mean magnitude: {:.6}\n",
            result.statistics.mean_magnitude
        ));
        report.push_str(&format!(
            "Max magnitude: {:.6}\n",
            result.statistics.max_magnitude
        ));
        report.push_str(&format!(
            "Min magnitude: {:.6}\n",
            result.statistics.min_magnitude
        ));

        // Errors
        if !result.errors.is_empty() {
            report.push_str("\n=== Errors ===\n");
            for error in &result.errors {
                report.push_str(&format!(
                    "- {}: {} (expected: {}, actual: {})\n",
                    error.tensor_name, error.context, error.expected, error.actual
                ));
            }
        }

        // Warnings
        if !result.warnings.is_empty() {
            report.push_str("\n=== Warnings ===\n");
            for warning in &result.warnings {
                report.push_str(&format!(
                    "- {}: {} ({})\n",
                    warning.tensor_name, warning.message, warning.suggestion
                ));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::CpuDevice;
    use torsh_core::shape::Shape;

    // Mock tensor for testing
    struct MockTensor {
        data: Vec<f32>,
        shape: Shape,
        requires_grad: bool,
        device: CpuDevice,
    }

    impl MockTensor {
        fn new(data: Vec<f32>, shape: Shape, requires_grad: bool) -> Self {
            Self {
                data,
                shape,
                requires_grad,
                device: CpuDevice::new(),
            }
        }
    }

    impl AutogradTensor<f32> for MockTensor {
        fn shape(&self) -> Shape {
            self.shape.clone()
        }

        fn requires_grad(&self) -> bool {
            self.requires_grad
        }

        fn data(&self) -> Box<dyn std::ops::Deref<Target = [f32]> + '_> {
            Box::new(self.data.as_slice())
        }

        fn clone_tensor(&self) -> Box<dyn AutogradTensor<f32>> {
            Box::new(MockTensor::new(
                self.data.clone(),
                self.shape.clone(),
                self.requires_grad,
            ))
        }

        fn to_vec(&self) -> Vec<f32> {
            self.data.clone()
        }

        fn device(&self) -> &dyn torsh_core::Device {
            &self.device
        }

        fn ones_like(&self) -> Box<dyn AutogradTensor<f32>> {
            let ones_data = vec![1.0; self.data.len()];
            Box::new(MockTensor::new(
                ones_data,
                self.shape.clone(),
                self.requires_grad,
            ))
        }

        fn zeros_like(&self) -> Box<dyn AutogradTensor<f32>> {
            let zeros_data = vec![0.0; self.data.len()];
            Box::new(MockTensor::new(
                zeros_data,
                self.shape.clone(),
                self.requires_grad,
            ))
        }
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(config.validate_shapes);
        assert!(config.validate_types);
        assert!(config.validate_ranges);
        assert!(config.validate_finite);
        assert!(config.strict_mode);
    }

    #[test]
    fn test_validator_creation() {
        let validator = GradientValidator::new();
        assert!(validator.config.validate_shapes);

        let custom_config = ValidationConfig {
            validate_shapes: false,
            ..Default::default()
        };
        let custom_validator = GradientValidator::with_config(custom_config);
        assert!(!custom_validator.config.validate_shapes);
    }

    #[test]
    fn test_shape_validation_success() {
        let validator = GradientValidator::new();
        let tensor = MockTensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]), true);
        let gradient = vec![0.1, 0.2, 0.3];

        let tensors = vec![("test", &tensor as &dyn AutogradTensor<f32>)];
        let gradients = vec![("test", Some(gradient.as_slice()))];

        let result = validator.validate_gradients(&tensors, &gradients).unwrap();
        assert!(result.passed);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_shape_validation_failure() {
        let _validator = GradientValidator::new();
        let tensor = MockTensor::new(vec![1.0, 2.0, 3.0], Shape::new(vec![3]), true);
        let gradient = vec![0.1, 0.2]; // Wrong size

        let tensors = vec![("test", &tensor as &dyn AutogradTensor<f32>)];
        let gradients = vec![("test", Some(gradient.as_slice()))];

        let config = ValidationConfig {
            strict_mode: false,
            ..Default::default()
        };
        let validator = GradientValidator::with_config(config);

        let result = validator.validate_gradients(&tensors, &gradients).unwrap();
        assert!(!result.passed);
        assert!(!result.errors.is_empty());
        assert_eq!(
            result.errors[0].error_type,
            ValidationErrorType::ShapeMismatch
        );
    }

    #[test]
    fn test_nan_detection() {
        let _validator = GradientValidator::new();
        let tensor = MockTensor::new(vec![1.0, 2.0], Shape::new(vec![2]), true);
        let gradient = vec![f32::NAN, 0.2];

        let tensors = vec![("test", &tensor as &dyn AutogradTensor<f32>)];
        let gradients = vec![("test", Some(gradient.as_slice()))];

        let config = ValidationConfig {
            strict_mode: false,
            ..Default::default()
        };
        let validator = GradientValidator::with_config(config);

        let result = validator.validate_gradients(&tensors, &gradients).unwrap();
        assert!(!result.passed);

        let has_nan_error = result
            .errors
            .iter()
            .any(|e| e.error_type == ValidationErrorType::InvalidValues && e.actual == "NaN");
        assert!(has_nan_error);
    }

    #[test]
    fn test_missing_gradient_detection() {
        let _validator = GradientValidator::new();
        let tensor = MockTensor::new(vec![1.0, 2.0], Shape::new(vec![2]), true);

        let tensors = vec![("test", &tensor as &dyn AutogradTensor<f32>)];
        let gradients = vec![("test", None)];

        let config = ValidationConfig {
            strict_mode: false,
            ..Default::default()
        };
        let validator = GradientValidator::with_config(config);

        let result = validator.validate_gradients(&tensors, &gradients).unwrap();
        assert!(!result.passed);

        let has_missing_error = result
            .errors
            .iter()
            .any(|e| e.error_type == ValidationErrorType::MissingGradient);
        assert!(has_missing_error);
    }

    #[test]
    fn test_quick_validate() {
        let validator = GradientValidator::new();

        // Valid gradient
        let valid_gradient = vec![0.1, 0.2, 0.3];
        assert!(validator.quick_validate("test", &valid_gradient).unwrap());

        // Invalid gradient with NaN
        let invalid_gradient = vec![0.1, f32::NAN, 0.3];
        assert!(!validator.quick_validate("test", &invalid_gradient).unwrap());
    }

    #[test]
    fn test_statistics_computation() {
        let validator = GradientValidator::new();
        let tensor = MockTensor::new(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]), true);
        let gradient = vec![0.0, 1.0, -1.0, 2.0];

        let tensors = vec![("test", &tensor as &dyn AutogradTensor<f32>)];
        let gradients = vec![("test", Some(gradient.as_slice()))];

        let result = validator.validate_gradients(&tensors, &gradients).unwrap();
        assert!(result.passed);

        let tensor_result = result.tensor_results.get("test").unwrap();
        assert_eq!(tensor_result.gradient_stats.num_elements, 4);
        assert_eq!(tensor_result.gradient_stats.zero_elements, 1);
        assert_eq!(tensor_result.gradient_stats.finite_elements, 4);
    }

    #[test]
    fn test_report_generation() {
        let validator = GradientValidator::new();
        let tensor = MockTensor::new(vec![1.0], Shape::new(vec![1]), true);
        let gradient = vec![0.5];

        let tensors = vec![("test", &tensor as &dyn AutogradTensor<f32>)];
        let gradients = vec![("test", Some(gradient.as_slice()))];

        let result = validator.validate_gradients(&tensors, &gradients).unwrap();
        let report = validator.generate_report(&result);

        assert!(report.contains("=== Gradient Validation Report ==="));
        assert!(report.contains("Status: PASSED"));
        assert!(report.contains("Total tensors: 1"));
    }
}
