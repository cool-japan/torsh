use crate::{Result, VisionError};
use std::backtrace::Backtrace;
use std::fmt;
use torsh_tensor::Tensor;

/// Enhanced error context for better debugging and error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub location: Option<ErrorLocation>,
    pub tensor_info: Option<TensorInfo>,
    pub suggestions: Vec<String>,
    pub related_errors: Vec<String>,
}

/// Location information for errors
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    pub file: String,
    pub line: u32,
    pub function: String,
}

/// Tensor information for debugging tensor-related errors
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub requires_grad: bool,
    pub element_count: usize,
}

impl TensorInfo {
    pub fn from_tensor(tensor: &Tensor<f32>) -> Self {
        let shape = tensor.shape().dims().to_vec();
        let element_count = shape.iter().product();

        Self {
            shape,
            dtype: "f32".to_string(),
            device: "cpu".to_string(), // Default for now
            requires_grad: false,      // Would need to check autograd context
            element_count,
        }
    }
}

impl ErrorContext {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            location: None,
            tensor_info: None,
            suggestions: Vec::new(),
            related_errors: Vec::new(),
        }
    }

    pub fn with_location(mut self, file: &str, line: u32, function: &str) -> Self {
        self.location = Some(ErrorLocation {
            file: file.to_string(),
            line,
            function: function.to_string(),
        });
        self
    }

    pub fn with_tensor_info(mut self, tensor: &Tensor<f32>) -> Self {
        self.tensor_info = Some(TensorInfo::from_tensor(tensor));
        self
    }

    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }

    pub fn with_related_error(mut self, error: &str) -> Self {
        self.related_errors.push(error.to_string());
        self
    }
}

/// Enhanced VisionError with better context and suggestions
#[derive(Debug)]
pub enum EnhancedVisionError {
    /// Shape mismatch with detailed information
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        operation: String,
        context: ErrorContext,
    },

    /// Transform error with recovery suggestions
    TransformError {
        transform_name: String,
        message: String,
        context: ErrorContext,
    },

    /// Model error with detailed diagnostics
    ModelError {
        model_name: String,
        layer: Option<String>,
        message: String,
        context: ErrorContext,
    },

    /// Device compatibility error
    DeviceError {
        required_device: String,
        actual_device: String,
        operation: String,
        context: ErrorContext,
    },

    /// Memory error with suggestions for optimization
    MemoryError {
        required_memory: Option<usize>,
        available_memory: Option<usize>,
        operation: String,
        context: ErrorContext,
    },

    /// Validation error with specific field information
    ValidationError {
        field: String,
        value: String,
        constraint: String,
        context: ErrorContext,
    },

    /// Wrapped original error with enhanced context
    Wrapped {
        original: VisionError,
        context: ErrorContext,
    },
}

impl fmt::Display for EnhancedVisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnhancedVisionError::ShapeMismatch {
                expected,
                actual,
                operation,
                context,
            } => {
                writeln!(f, "Shape mismatch in operation '{}'", operation)?;
                writeln!(f, "  Expected shape: {:?}", expected)?;
                writeln!(f, "  Actual shape:   {:?}", actual)?;
                self.write_context(f, context)?;
            }
            EnhancedVisionError::TransformError {
                transform_name,
                message,
                context,
            } => {
                writeln!(f, "Transform '{}' failed: {}", transform_name, message)?;
                self.write_context(f, context)?;
            }
            EnhancedVisionError::ModelError {
                model_name,
                layer,
                message,
                context,
            } => {
                write!(f, "Model '{}' error", model_name)?;
                if let Some(layer) = layer {
                    write!(f, " in layer '{}'", layer)?;
                }
                writeln!(f, ": {}", message)?;
                self.write_context(f, context)?;
            }
            EnhancedVisionError::DeviceError {
                required_device,
                actual_device,
                operation,
                context,
            } => {
                writeln!(f, "Device incompatibility in operation '{}'", operation)?;
                writeln!(f, "  Required device: {}", required_device)?;
                writeln!(f, "  Actual device:   {}", actual_device)?;
                self.write_context(f, context)?;
            }
            EnhancedVisionError::MemoryError {
                required_memory,
                available_memory,
                operation,
                context,
            } => {
                writeln!(f, "Memory error in operation '{}'", operation)?;
                if let (Some(req), Some(avail)) = (required_memory, available_memory) {
                    writeln!(f, "  Required memory: {} bytes", req)?;
                    writeln!(f, "  Available memory: {} bytes", avail)?;
                }
                self.write_context(f, context)?;
            }
            EnhancedVisionError::ValidationError {
                field,
                value,
                constraint,
                context,
            } => {
                writeln!(f, "Validation error for field '{}'", field)?;
                writeln!(f, "  Value: {}", value)?;
                writeln!(f, "  Constraint: {}", constraint)?;
                self.write_context(f, context)?;
            }
            EnhancedVisionError::Wrapped { original, context } => {
                writeln!(f, "Enhanced error context for: {}", original)?;
                self.write_context(f, context)?;
            }
        }
        Ok(())
    }
}

impl EnhancedVisionError {
    fn write_context(&self, f: &mut fmt::Formatter<'_>, context: &ErrorContext) -> fmt::Result {
        if !context.operation.is_empty() {
            writeln!(f, "  Operation: {}", context.operation)?;
        }

        if let Some(location) = &context.location {
            writeln!(
                f,
                "  Location: {}:{} in {}",
                location.file, location.line, location.function
            )?;
        }

        if let Some(tensor_info) = &context.tensor_info {
            writeln!(f, "  Tensor info:")?;
            writeln!(f, "    Shape: {:?}", tensor_info.shape)?;
            writeln!(f, "    Type: {}", tensor_info.dtype)?;
            writeln!(f, "    Device: {}", tensor_info.device)?;
            writeln!(f, "    Elements: {}", tensor_info.element_count)?;
        }

        if !context.suggestions.is_empty() {
            writeln!(f, "  Suggestions:")?;
            for suggestion in &context.suggestions {
                writeln!(f, "    - {}", suggestion)?;
            }
        }

        if !context.related_errors.is_empty() {
            writeln!(f, "  Related errors:")?;
            for error in &context.related_errors {
                writeln!(f, "    - {}", error)?;
            }
        }

        Ok(())
    }
}

impl std::error::Error for EnhancedVisionError {}

/// Error recovery strategies for common vision operations
pub enum ErrorRecovery {
    /// Retry the operation with modified parameters
    Retry { max_attempts: u32, backoff_ms: u64 },
    /// Fall back to an alternative implementation
    Fallback { alternative: String },
    /// Use default values for missing parameters
    UseDefaults,
    /// Skip the operation and continue
    Skip,
    /// Abort the entire pipeline
    Abort,
}

/// Enhanced error builder for fluent error construction
pub struct ErrorBuilder {
    error_type: String,
    message: String,
    context: ErrorContext,
}

impl ErrorBuilder {
    pub fn new(error_type: &str, message: &str) -> Self {
        Self {
            error_type: error_type.to_string(),
            message: message.to_string(),
            context: ErrorContext::new(error_type),
        }
    }

    pub fn operation(mut self, operation: &str) -> Self {
        self.context.operation = operation.to_string();
        self
    }

    pub fn location(self, file: &str, line: u32, function: &str) -> Self {
        Self {
            context: self.context.with_location(file, line, function),
            ..self
        }
    }

    pub fn tensor(self, tensor: &Tensor<f32>) -> Self {
        Self {
            context: self.context.with_tensor_info(tensor),
            ..self
        }
    }

    pub fn suggestion(self, suggestion: &str) -> Self {
        Self {
            context: self.context.with_suggestion(suggestion),
            ..self
        }
    }

    pub fn related_error(self, error: &str) -> Self {
        Self {
            context: self.context.with_related_error(error),
            ..self
        }
    }

    pub fn build_shape_mismatch(
        self,
        expected: Vec<usize>,
        actual: Vec<usize>,
    ) -> EnhancedVisionError {
        EnhancedVisionError::ShapeMismatch {
            expected,
            actual,
            operation: self.context.operation.clone(),
            context: self.context,
        }
    }

    pub fn build_transform_error(self, transform_name: &str) -> EnhancedVisionError {
        EnhancedVisionError::TransformError {
            transform_name: transform_name.to_string(),
            message: self.message,
            context: self.context,
        }
    }

    pub fn build_model_error(self, model_name: &str, layer: Option<&str>) -> EnhancedVisionError {
        EnhancedVisionError::ModelError {
            model_name: model_name.to_string(),
            layer: layer.map(|s| s.to_string()),
            message: self.message,
            context: self.context,
        }
    }

    pub fn build_validation_error(
        self,
        field: &str,
        value: &str,
        constraint: &str,
    ) -> EnhancedVisionError {
        EnhancedVisionError::ValidationError {
            field: field.to_string(),
            value: value.to_string(),
            constraint: constraint.to_string(),
            context: self.context,
        }
    }

    pub fn wrap(self, original: VisionError) -> EnhancedVisionError {
        EnhancedVisionError::Wrapped {
            original,
            context: self.context,
        }
    }
}

/// Macros for convenient error creation
#[macro_export]
macro_rules! shape_mismatch {
    ($expected:expr, $actual:expr, $op:expr) => {
        $crate::error_handling::ErrorBuilder::new("ShapeMismatch", "Shape mismatch detected")
            .operation($op)
            .location(file!(), line!(), std::any::type_name::<()>())
            .suggestion("Check input tensor dimensions")
            .build_shape_mismatch($expected, $actual)
    };
}

#[macro_export]
macro_rules! transform_error {
    ($transform:expr, $msg:expr) => {
        $crate::error_handling::ErrorBuilder::new("TransformError", $msg)
            .location(file!(), line!(), std::any::type_name::<()>())
            .build_transform_error($transform)
    };
}

#[macro_export]
macro_rules! model_error {
    ($model:expr, $layer:expr, $msg:expr) => {
        $crate::error_handling::ErrorBuilder::new("ModelError", $msg)
            .location(file!(), line!(), std::any::type_name::<()>())
            .build_model_error($model, $layer)
    };
}

/// Error handling utilities
pub struct ErrorHandler;

impl ErrorHandler {
    /// Validate tensor shape with detailed error reporting
    pub fn validate_shape(
        tensor: &Tensor<f32>,
        expected_dims: usize,
        operation: &str,
    ) -> std::result::Result<(), EnhancedVisionError> {
        let actual_dims = tensor.shape().dims().len();
        if actual_dims != expected_dims {
            return Err(
                ErrorBuilder::new("ShapeValidation", "Invalid tensor dimensionality")
                    .operation(operation)
                    .location(file!(), line!(), "validate_shape")
                    .tensor(tensor)
                    .suggestion(&format!(
                        "Expected {}D tensor, got {}D",
                        expected_dims, actual_dims
                    ))
                    .suggestion("Check your input preprocessing pipeline")
                    .build_validation_error(
                        "tensor_dims",
                        &actual_dims.to_string(),
                        &format!("Must be exactly {} dimensions", expected_dims),
                    ),
            );
        }
        Ok(())
    }

    /// Validate tensor shape matches expected shape
    pub fn validate_exact_shape(
        tensor: &Tensor<f32>,
        expected_shape: &[usize],
        operation: &str,
    ) -> std::result::Result<(), EnhancedVisionError> {
        let shape_binding = tensor.shape();
        let actual_shape = shape_binding.dims();
        if actual_shape != expected_shape {
            return Err(
                ErrorBuilder::new("ShapeValidation", "Tensor shape mismatch")
                    .operation(operation)
                    .location(file!(), line!(), "validate_exact_shape")
                    .tensor(tensor)
                    .suggestion("Resize or reshape your input tensor")
                    .suggestion("Check if you need to add/remove batch dimension")
                    .build_shape_mismatch(expected_shape.to_vec(), actual_shape.to_vec()),
            );
        }
        Ok(())
    }

    /// Validate parameter ranges with suggestions
    pub fn validate_range<T: PartialOrd + fmt::Display + Copy>(
        value: T,
        min: T,
        max: T,
        param_name: &str,
    ) -> std::result::Result<(), EnhancedVisionError> {
        if value < min || value > max {
            return Err(
                ErrorBuilder::new("ParameterValidation", "Parameter out of valid range")
                    .location(file!(), line!(), "validate_range")
                    .suggestion(&format!("Valid range is [{}, {}]", min, max))
                    .suggestion("Consider using parameter clipping or validation in your pipeline")
                    .build_validation_error(
                        param_name,
                        &value.to_string(),
                        &format!("Must be in range [{}, {}]", min, max),
                    ),
            );
        }
        Ok(())
    }

    /// Provide recovery suggestions for common errors
    pub fn suggest_recovery(error: &VisionError) -> Vec<String> {
        let mut suggestions = Vec::new();

        match error {
            VisionError::InvalidShape(_) => {
                suggestions.push("Check input tensor dimensions".to_string());
                suggestions.push("Verify your data preprocessing pipeline".to_string());
                suggestions.push("Add explicit reshaping transforms".to_string());
            }
            VisionError::TransformError(_) => {
                suggestions.push("Verify transform parameters are valid".to_string());
                suggestions
                    .push("Check if input tensor is in expected format (CHW vs HWC)".to_string());
                suggestions.push("Ensure tensor values are in expected range".to_string());
            }
            VisionError::ModelError(_) => {
                suggestions.push("Check model input requirements".to_string());
                suggestions.push("Verify model is properly initialized".to_string());
                suggestions.push("Ensure input preprocessing matches training".to_string());
            }
            VisionError::TensorError(_) => {
                suggestions
                    .push("Check tensor operations are valid for current device".to_string());
                suggestions.push("Verify tensor dtypes are compatible".to_string());
                suggestions.push("Ensure sufficient memory is available".to_string());
            }
            _ => {
                suggestions.push("Check the error message for specific details".to_string());
                suggestions.push("Verify all inputs are valid".to_string());
            }
        }

        suggestions
    }

    /// Convert standard VisionError to enhanced error with context
    pub fn enhance_error(error: VisionError, operation: &str) -> EnhancedVisionError {
        let suggestions = Self::suggest_recovery(&error);
        let mut builder = ErrorBuilder::new("Enhanced", "Enhanced error with context")
            .operation(operation)
            .location(file!(), line!(), "enhance_error");

        for suggestion in suggestions {
            builder = builder.suggestion(&suggestion);
        }

        builder.wrap(error)
    }
}

/// Result type alias for enhanced errors
pub type EnhancedResult<T> = std::result::Result<T, EnhancedVisionError>;

/// Trait for operations that can provide enhanced error handling
pub trait EnhancedErrorHandling {
    type Output;

    fn with_error_context(self, operation: &str) -> EnhancedResult<Self::Output>;
}

impl<T> EnhancedErrorHandling for Result<T> {
    type Output = T;

    fn with_error_context(self, operation: &str) -> EnhancedResult<T> {
        self.map_err(|e| ErrorHandler::enhance_error(e, operation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_error_builder() {
        let error = ErrorBuilder::new("Test", "Test error")
            .operation("test_operation")
            .suggestion("This is a test suggestion")
            .build_validation_error("test_field", "invalid_value", "must be valid");

        assert!(format!("{}", error).contains("test_operation"));
        assert!(format!("{}", error).contains("This is a test suggestion"));
    }

    #[test]
    fn test_shape_validation() {
        let tensor = zeros(&[3, 224, 224]).unwrap();
        let result = ErrorHandler::validate_shape(&tensor, 3, "test_operation");
        assert!(result.is_ok());

        let result = ErrorHandler::validate_shape(&tensor, 4, "test_operation");
        assert!(result.is_err());
    }

    #[test]
    fn test_parameter_validation() {
        let result = ErrorHandler::validate_range(0.5f32, 0.0, 1.0, "probability");
        assert!(result.is_ok());

        let result = ErrorHandler::validate_range(1.5f32, 0.0, 1.0, "probability");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_enhancement() {
        let original_error = VisionError::InvalidShape("test shape error".to_string());
        let enhanced = ErrorHandler::enhance_error(original_error, "test_operation");

        let error_str = format!("{}", enhanced);
        assert!(error_str.contains("test_operation"));
        assert!(error_str.contains("Suggestions:"));
    }
}
