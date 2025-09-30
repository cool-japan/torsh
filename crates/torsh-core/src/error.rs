//! Error types for ToRSh - Clean Modular Interface
//!
//! This module provides a unified interface to the ToRSh error system.
//! All error implementations have been organized into specialized modules
//! for better maintainability and categorization.
//!
//! # Architecture
//!
//! The error system is organized into specialized modules:
//!
//! - **core**: Error infrastructure, location tracking, debug context
//! - **shape_errors**: Shape mismatches, broadcasting, tensor operations
//! - **index_errors**: Index bounds checking and access violations
//! - **general_errors**: I/O, configuration, runtime, and miscellaneous errors
//!
//! All error types are unified through the main `TorshError` enum which provides
//! backward compatibility while enabling modular error handling.

// Modular error system
mod core;
mod general_errors;
mod index_errors;
mod shape_errors;

// Re-export the complete modular interface
pub use core::{
    capture_minimal_stack_trace, capture_stack_trace, format_shape, ErrorCategory,
    ErrorDebugContext, ErrorLocation, ErrorSeverity, ShapeDisplay, ThreadInfo,
};

pub use general_errors::GeneralError;
pub use index_errors::IndexError;
pub use shape_errors::ShapeError;

// Re-export the unified error type and result
pub use thiserror::Error;

/// Main ToRSh error enum - unified interface to all error types
#[derive(Error, Debug, Clone)]
pub enum TorshError {
    // Modular error variants
    #[error(transparent)]
    Shape(#[from] ShapeError),

    #[error(transparent)]
    Index(#[from] IndexError),

    #[error(transparent)]
    General(#[from] GeneralError),

    // Error with enhanced context information
    #[error("{message}")]
    WithContext {
        message: String,
        error_category: ErrorCategory,
        severity: ErrorSeverity,
        debug_context: Box<ErrorDebugContext>,
        #[source]
        source: Option<Box<TorshError>>,
    },

    // Legacy compatibility variants (for backward compatibility)
    #[error(
        "Shape mismatch: expected {}, got {}",
        format_shape(expected),
        format_shape(got)
    )]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error(
        "Broadcasting error: incompatible shapes {} and {}",
        format_shape(shape1),
        format_shape(shape2)
    )]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    #[error("Index out of bounds: index {index} is out of bounds for dimension with size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Device mismatch: tensors must be on the same device")]
    DeviceMismatch,

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    // Additional legacy compatibility variants
    #[error("Thread synchronization error: {0}")]
    SynchronizationError(String),

    #[error("Memory allocation failed: {0}")]
    AllocationError(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Numeric conversion error: {0}")]
    ConversionError(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Unsupported operation '{op}' for data type '{dtype}'")]
    UnsupportedOperation { op: String, dtype: String },

    #[error("Autograd error: {0}")]
    AutogradError(String),

    #[error("Compute error: {0}")]
    ComputeError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Index out of bounds: index {index} is out of bounds for dimension with size {size}")]
    IndexError { index: usize, size: usize },

    #[error("Iteration error: {0}")]
    IterationError(String),

    #[error("Other error: {0}")]
    Other(String),
}

/// Result type alias for ToRSh operations
pub type Result<T> = std::result::Result<T, TorshError>;

impl TorshError {
    /// Create a shape mismatch error (backward compatibility)
    pub fn shape_mismatch(expected: &[usize], got: &[usize]) -> Self {
        Self::Shape(ShapeError::shape_mismatch(expected, got))
    }

    /// Create a dimension error during operation
    pub fn dimension_error(msg: &str, operation: &str) -> Self {
        Self::General(GeneralError::DimensionError(format!(
            "{msg} during {operation}"
        )))
    }

    /// Create an index error
    pub fn index_error(index: usize, size: usize) -> Self {
        Self::Index(IndexError::out_of_bounds(index, size))
    }

    /// Create a type mismatch error
    pub fn type_mismatch(expected: &str, actual: &str) -> Self {
        Self::General(GeneralError::TypeMismatch {
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }

    /// Create a dimension error with context (backward compatibility)
    pub fn dimension_error_with_context(msg: &str, operation: &str) -> Self {
        Self::General(GeneralError::DimensionError(format!(
            "{msg} during {operation}"
        )))
    }

    /// Create a synchronization error (backward compatibility)
    pub fn synchronization_error(msg: &str) -> Self {
        Self::SynchronizationError(msg.to_string())
    }

    /// Create an allocation error (backward compatibility)
    pub fn allocation_error(msg: &str) -> Self {
        Self::AllocationError(msg.to_string())
    }

    /// Create an invalid operation error (backward compatibility)
    pub fn invalid_operation(msg: &str) -> Self {
        Self::InvalidOperation(msg.to_string())
    }

    /// Create a conversion error (backward compatibility)
    pub fn conversion_error(msg: &str) -> Self {
        Self::ConversionError(msg.to_string())
    }

    /// Create an invalid argument error with context (backward compatibility)
    pub fn invalid_argument_with_context(msg: &str, context: &str) -> Self {
        Self::InvalidArgument(format!("{msg} (context: {context})"))
    }

    /// Create a config error with context (backward compatibility)
    pub fn config_error_with_context(msg: &str, context: &str) -> Self {
        Self::ConfigError(format!("{msg} (context: {context})"))
    }

    /// Create a dimension error (backward compatibility)
    pub fn dimension_error_simple(msg: String) -> Self {
        Self::InvalidShape(msg)
    }

    /// Create a formatted shape mismatch error (backward compatibility)
    pub fn shape_mismatch_formatted(expected: &str, got: &str) -> Self {
        Self::InvalidShape(format!("Shape mismatch: expected {expected}, got {got}"))
    }

    /// Create an operation error (backward compatibility)
    pub fn operation_error(msg: &str) -> Self {
        Self::InvalidOperation(msg.to_string())
    }

    /// Wrap an error with location information (backward compatibility)
    pub fn wrap_with_location(self, location: String) -> Self {
        // For backward compatibility, just add context
        self.with_context(&location)
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Shape(e) => e.category(),
            Self::Index(e) => e.category(),
            Self::General(e) => e.category(),
            Self::WithContext { error_category, .. } => error_category.clone(),
            Self::ShapeMismatch { .. } | Self::BroadcastError { .. } => ErrorCategory::Shape,
            Self::IndexOutOfBounds { .. } => ErrorCategory::UserInput,
            Self::InvalidArgument(_) => ErrorCategory::UserInput,
            Self::IoError(_) => ErrorCategory::Io,
            Self::DeviceMismatch => ErrorCategory::Device,
            Self::NotImplemented(_) => ErrorCategory::Internal,
            Self::SynchronizationError(_) => ErrorCategory::Threading,
            Self::AllocationError(_) => ErrorCategory::Memory,
            Self::InvalidOperation(_) => ErrorCategory::UserInput,
            Self::ConversionError(_) => ErrorCategory::DataType,
            Self::BackendError(_) => ErrorCategory::Device,
            Self::InvalidShape(_) => ErrorCategory::Shape,
            Self::RuntimeError(_) => ErrorCategory::Internal,
            Self::DeviceError(_) => ErrorCategory::Device,
            Self::ConfigError(_) => ErrorCategory::Configuration,
            Self::InvalidState(_) => ErrorCategory::Internal,
            Self::UnsupportedOperation { .. } => ErrorCategory::UserInput,
            Self::AutogradError(_) => ErrorCategory::Internal,
            Self::ComputeError(_) => ErrorCategory::Internal,
            Self::SerializationError(_) => ErrorCategory::Io,
            Self::IndexError { .. } => ErrorCategory::UserInput,
            Self::IterationError(_) => ErrorCategory::Internal,
            Self::Other(_) => ErrorCategory::Internal,
        }
    }

    /// Get the error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Shape(e) => e.severity(),
            Self::Index(_) => ErrorSeverity::Medium,
            Self::General(_) => ErrorSeverity::Low,
            Self::WithContext { severity, .. } => severity.clone(),
            Self::ShapeMismatch { .. } | Self::BroadcastError { .. } => ErrorSeverity::High,
            Self::IndexOutOfBounds { .. } => ErrorSeverity::Medium,
            Self::DeviceMismatch => ErrorSeverity::High,
            Self::SynchronizationError(_) => ErrorSeverity::Medium,
            Self::AllocationError(_) => ErrorSeverity::High,
            Self::InvalidOperation(_) => ErrorSeverity::Medium,
            Self::ConversionError(_) => ErrorSeverity::Medium,
            Self::BackendError(_) => ErrorSeverity::High,
            Self::InvalidShape(_) => ErrorSeverity::High,
            Self::RuntimeError(_) => ErrorSeverity::Medium,
            Self::DeviceError(_) => ErrorSeverity::High,
            Self::ConfigError(_) => ErrorSeverity::Medium,
            Self::InvalidState(_) => ErrorSeverity::Medium,
            Self::UnsupportedOperation { .. } => ErrorSeverity::Medium,
            Self::AutogradError(_) => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }

    /// Add context to an error
    pub fn with_context(self, message: &str) -> Self {
        let category = self.category();
        let severity = self.severity();

        Self::WithContext {
            message: message.to_string(),
            error_category: category,
            severity,
            debug_context: Box::new(ErrorDebugContext::minimal()),
            source: Some(Box::new(self)),
        }
    }
}

// Standard library error conversions
impl From<std::io::Error> for TorshError {
    fn from(err: std::io::Error) -> Self {
        Self::General(GeneralError::IoError(err.to_string()))
    }
}

#[cfg(feature = "serialize")]
impl From<serde_json::Error> for TorshError {
    fn from(err: serde_json::Error) -> Self {
        Self::General(GeneralError::SerializationError(err.to_string()))
    }
}

impl<T> From<std::sync::PoisonError<T>> for TorshError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        Self::General(GeneralError::SynchronizationError(format!(
            "Mutex poisoned: {err}"
        )))
    }
}

impl From<std::num::TryFromIntError> for TorshError {
    fn from(err: std::num::TryFromIntError) -> Self {
        Self::General(GeneralError::ConversionError(format!(
            "Integer conversion failed: {err}"
        )))
    }
}

impl From<std::num::ParseIntError> for TorshError {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::General(GeneralError::ConversionError(format!(
            "Integer parsing failed: {err}"
        )))
    }
}

impl From<std::num::ParseFloatError> for TorshError {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::General(GeneralError::ConversionError(format!(
            "Float parsing failed: {err}"
        )))
    }
}

/// Convenience macros for error creation with location information
#[macro_export]
macro_rules! torsh_error_with_location {
    ($error_type:expr) => {
        $crate::error::TorshError::WithContext {
            message: format!("{}", $error_type),
            error_category: $error_type.category(),
            severity: $error_type.severity(),
            debug_context: $crate::error::ErrorDebugContext::minimal(),
            source: Some(Box::new($error_type.into())),
        }
    };
    ($message:expr) => {
        $crate::error::TorshError::WithContext {
            message: $message.to_string(),
            error_category: $crate::error::ErrorCategory::Internal,
            severity: $crate::error::ErrorSeverity::Medium,
            debug_context: $crate::error::ErrorDebugContext::minimal(),
            source: None,
        }
    };
}

/// Convenience macro for shape mismatch errors
#[macro_export]
macro_rules! shape_mismatch_error {
    ($expected:expr, $got:expr) => {
        $crate::error::TorshError::shape_mismatch($expected, $got)
    };
}

/// Convenience macro for index errors
#[macro_export]
macro_rules! index_error {
    ($index:expr, $size:expr) => {
        $crate::error::TorshError::index_error($index, $size)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_error_system() {
        // Test shape error conversion
        let shape_err = ShapeError::shape_mismatch(&[2, 3], &[3, 2]);
        let torsh_err: TorshError = shape_err.into();
        assert_eq!(torsh_err.category(), ErrorCategory::Shape);

        // Test index error conversion
        let index_err = IndexError::out_of_bounds(5, 3);
        let torsh_err: TorshError = index_err.into();
        assert_eq!(torsh_err.category(), ErrorCategory::UserInput);

        // Test general error conversion
        let general_err = GeneralError::InvalidArgument("test".to_string());
        let torsh_err: TorshError = general_err.into();
        assert_eq!(torsh_err.category(), ErrorCategory::UserInput);
    }

    #[test]
    fn test_backward_compatibility() {
        let error = TorshError::shape_mismatch(&[2, 3], &[3, 2]);
        assert_eq!(error.category(), ErrorCategory::Shape);
        assert_eq!(error.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_error_context() {
        let base_error = TorshError::InvalidArgument("test".to_string());
        let contextual_error = base_error.with_context("During tensor operation");

        match contextual_error {
            TorshError::WithContext { message, .. } => {
                assert_eq!(message, "During tensor operation");
            }
            _ => panic!("Expected WithContext error"),
        }
    }

    #[test]
    fn test_convenience_macros() {
        let shape_error = shape_mismatch_error!(&[2, 3], &[3, 2]);
        assert_eq!(shape_error.category(), ErrorCategory::Shape);

        let idx_error = index_error!(5, 3);
        assert_eq!(idx_error.category(), ErrorCategory::UserInput);
    }

    #[test]
    fn test_standard_conversions() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let torsh_err: TorshError = io_err.into();
        assert_eq!(torsh_err.category(), ErrorCategory::Io);

        #[cfg(feature = "serialize")]
        {
            let json_err = serde_json::from_str::<i32>("invalid json").unwrap_err();
            let torsh_err: TorshError = json_err.into();
            assert_eq!(torsh_err.category(), ErrorCategory::Internal);
        }
    }

    #[test]
    fn test_error_severity_ordering() {
        let low_error = TorshError::NotImplemented("test".to_string());
        let high_error = TorshError::shape_mismatch(&[2, 3], &[3, 2]);

        assert!(low_error.severity() < high_error.severity());
    }
}
