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

    #[error(
        "Invalid dimension: dimension {dim} is out of bounds for tensor with {ndim} dimensions"
    )]
    InvalidDimension { dim: usize, ndim: usize },

    #[error("Iteration error: {0}")]
    IterationError(String),

    #[error("Other error: {0}")]
    Other(String),

    // CUDA/GPU Backend compatibility variants
    #[error("Context error: {message}")]
    Context { message: String },

    #[error("Invalid device: device {device_id}")]
    InvalidDevice { device_id: usize },

    #[error("Backend operation failed: {0}")]
    Backend(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Memory error: {message}")]
    Memory { message: String },

    #[error("cuDNN error: {0}")]
    CudnnError(String),

    #[error("Unimplemented: {0}")]
    Unimplemented(String),

    #[error("Initialization error: {0}")]
    InitializationError(String),
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
            Self::InvalidDimension { .. } => ErrorCategory::UserInput,
            Self::IterationError(_) => ErrorCategory::Internal,
            Self::Other(_) => ErrorCategory::Internal,
            // CUDA/GPU Backend compatibility variants
            Self::Context { .. } => ErrorCategory::Device,
            Self::InvalidDevice { .. } => ErrorCategory::Device,
            Self::Backend(_) => ErrorCategory::Device,
            Self::InvalidValue(_) => ErrorCategory::UserInput,
            Self::Memory { .. } => ErrorCategory::Memory,
            Self::CudnnError(_) => ErrorCategory::Device,
            Self::Unimplemented(_) => ErrorCategory::Internal,
            Self::InitializationError(_) => ErrorCategory::Internal,
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
            Self::InvalidDimension { .. } => ErrorSeverity::Medium,
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

    /// Add minimal context to an error (lightweight, no backtrace)
    ///
    /// Use this for performance-critical paths where error context
    /// is helpful but backtrace overhead is not justified.
    ///
    /// # Example
    /// ```
    /// use torsh_core::error::{TorshError, Result};
    ///
    /// fn tensor_operation() -> Result<()> {
    ///     let error = TorshError::InvalidShape("invalid dimensions".to_string())
    ///         .with_context("during tensor reshape");
    ///     Err(error)
    /// }
    /// ```
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

    /// Add rich context to an error (includes full backtrace)
    ///
    /// Use this for debugging and development environments where
    /// detailed error information is valuable. Captures full backtrace
    /// and thread information.
    ///
    /// # Example
    /// ```
    /// use torsh_core::error::{TorshError, Result};
    ///
    /// fn critical_operation() -> Result<()> {
    ///     let error = TorshError::InvalidShape("invalid dimensions".to_string())
    ///         .with_rich_context("during critical tensor operation");
    ///     Err(error)
    /// }
    /// ```
    pub fn with_rich_context(self, message: &str) -> Self {
        let category = self.category();
        let severity = self.severity();

        Self::WithContext {
            message: message.to_string(),
            error_category: category,
            severity,
            debug_context: Box::new(ErrorDebugContext::new()),
            source: Some(Box::new(self)),
        }
    }

    /// Add context with custom metadata (minimal backtrace)
    ///
    /// Use this to add structured metadata without the overhead
    /// of a full backtrace. Ideal for operation tracking and debugging.
    ///
    /// # Example
    /// ```
    /// use torsh_core::error::{TorshError, Result};
    ///
    /// fn tensor_add(shape1: &[usize], shape2: &[usize]) -> Result<()> {
    ///     let error = TorshError::InvalidShape("incompatible shapes".to_string())
    ///         .with_metadata("during tensor addition")
    ///         .add_metadata("shape1", &format!("{:?}", shape1))
    ///         .add_metadata("shape2", &format!("{:?}", shape2));
    ///     Err(error)
    /// }
    /// ```
    pub fn with_metadata(self, message: &str) -> Self {
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

    /// Add metadata to an existing error
    ///
    /// This method allows adding key-value metadata to enrich error context
    /// without creating a new error wrapper. If the error is not already
    /// a `WithContext` variant, it will be converted to one.
    ///
    /// # Example
    /// ```
    /// use torsh_core::error::{TorshError, Result};
    ///
    /// fn process_tensor(name: &str, size: usize) -> Result<()> {
    ///     let error = TorshError::AllocationError("out of memory".to_string())
    ///         .add_metadata("tensor_name", name)
    ///         .add_metadata("requested_size", &size.to_string());
    ///     Err(error)
    /// }
    /// ```
    pub fn add_metadata(self, key: &str, value: &str) -> Self {
        match self {
            Self::WithContext {
                message,
                error_category,
                severity,
                mut debug_context,
                source,
            } => {
                debug_context
                    .metadata
                    .insert(key.to_string(), value.to_string());
                Self::WithContext {
                    message,
                    error_category,
                    severity,
                    debug_context,
                    source,
                }
            }
            other => {
                let category = other.category();
                let severity = other.severity();
                let mut context = ErrorDebugContext::minimal();
                context.metadata.insert(key.to_string(), value.to_string());

                Self::WithContext {
                    message: format!("{other}"),
                    error_category: category,
                    severity,
                    debug_context: Box::new(context),
                    source: Some(Box::new(other)),
                }
            }
        }
    }

    /// Add shape information as metadata
    ///
    /// Convenience method for adding tensor shape information to errors.
    ///
    /// # Example
    /// ```
    /// use torsh_core::error::{TorshError, Result};
    ///
    /// fn validate_shape(actual: &[usize], expected: &[usize]) -> Result<()> {
    ///     let error = TorshError::shape_mismatch(expected, actual)
    ///         .add_shape_metadata("actual_shape", actual)
    ///         .add_shape_metadata("expected_shape", expected);
    ///     Err(error)
    /// }
    /// ```
    pub fn add_shape_metadata(self, key: &str, shape: &[usize]) -> Self {
        self.add_metadata(key, &format!("{:?}", shape))
    }

    /// Add operation name as metadata
    ///
    /// Convenience method for tracking which operation caused the error.
    ///
    /// # Example
    /// ```
    /// use torsh_core::error::{TorshError, Result};
    ///
    /// fn matmul(a_shape: &[usize], b_shape: &[usize]) -> Result<()> {
    ///     let error = TorshError::shape_mismatch(a_shape, b_shape)
    ///         .with_operation("matmul")
    ///         .add_shape_metadata("lhs_shape", a_shape)
    ///         .add_shape_metadata("rhs_shape", b_shape);
    ///     Err(error)
    /// }
    /// ```
    pub fn with_operation(self, operation: &str) -> Self {
        self.add_metadata("operation", operation)
    }

    /// Add device information as metadata
    ///
    /// Convenience method for tracking device-related errors.
    ///
    /// # Example
    /// ```
    /// use torsh_core::error::{TorshError, Result};
    ///
    /// fn allocate_on_device(device_id: usize) -> Result<()> {
    ///     let error = TorshError::DeviceError("allocation failed".to_string())
    ///         .with_device(device_id)
    ///         .add_metadata("allocation_type", "tensor");
    ///     Err(error)
    /// }
    /// ```
    pub fn with_device(self, device_id: usize) -> Self {
        self.add_metadata("device_id", &device_id.to_string())
    }

    /// Add dtype information as metadata
    ///
    /// Convenience method for tracking data type-related errors.
    ///
    /// # Example
    /// ```
    /// use torsh_core::error::{TorshError, Result};
    ///
    /// fn convert_dtype(from: &str, to: &str) -> Result<()> {
    ///     let error = TorshError::ConversionError("unsupported conversion".to_string())
    ///         .add_metadata("from_dtype", from)
    ///         .add_metadata("to_dtype", to);
    ///     Err(error)
    /// }
    /// ```
    pub fn with_dtype(self, dtype: &str) -> Self {
        self.add_metadata("dtype", dtype)
    }

    /// Get all metadata from the error
    ///
    /// Returns an empty map if the error doesn't have metadata.
    pub fn metadata(&self) -> std::collections::HashMap<String, String> {
        match self {
            Self::WithContext { debug_context, .. } => debug_context.metadata.clone(),
            _ => std::collections::HashMap::new(),
        }
    }

    /// Get the error's debug context if available
    ///
    /// Returns None if the error is not a `WithContext` variant.
    pub fn debug_context(&self) -> Option<&ErrorDebugContext> {
        match self {
            Self::WithContext { debug_context, .. } => Some(debug_context),
            _ => None,
        }
    }

    /// Format the error with full debug information
    ///
    /// This includes metadata, backtrace, and thread information when available.
    pub fn format_debug(&self) -> String {
        let mut output = format!("Error: {self}\n");

        if let Some(context) = self.debug_context() {
            output.push_str("\n");
            output.push_str(&context.format_debug_info());
        }

        if let Self::WithContext {
            source: Some(source),
            ..
        } = self
        {
            output.push_str("\nCaused by:\n");
            output.push_str(&format!("  {source}"));
        }

        output
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

    #[test]
    fn test_rich_context() {
        let error = TorshError::InvalidShape("test error".to_string())
            .with_rich_context("during tensor operation");

        match error {
            TorshError::WithContext {
                message,
                debug_context,
                ..
            } => {
                assert_eq!(message, "during tensor operation");
                // Rich context should have backtrace (or a message about it)
                assert!(debug_context.backtrace.is_some());
            }
            _ => panic!("Expected WithContext error"),
        }
    }

    #[test]
    fn test_add_metadata() {
        let error = TorshError::InvalidShape("test".to_string())
            .add_metadata("key1", "value1")
            .add_metadata("key2", "value2");

        let metadata = error.metadata();
        assert_eq!(metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(metadata.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_add_shape_metadata() {
        let shape1 = vec![2, 3, 4];
        let shape2 = vec![4, 5];

        let error = TorshError::shape_mismatch(&shape1, &shape2)
            .add_shape_metadata("tensor_a", &shape1)
            .add_shape_metadata("tensor_b", &shape2);

        let metadata = error.metadata();
        assert!(metadata.contains_key("tensor_a"));
        assert!(metadata.contains_key("tensor_b"));
        assert!(metadata["tensor_a"].contains("2"));
        assert!(metadata["tensor_b"].contains("4"));
    }

    #[test]
    fn test_with_operation() {
        let error = TorshError::InvalidShape("test".to_string()).with_operation("matmul");

        let metadata = error.metadata();
        assert_eq!(metadata.get("operation"), Some(&"matmul".to_string()));
    }

    #[test]
    fn test_with_device() {
        let error = TorshError::DeviceError("allocation failed".to_string()).with_device(42);

        let metadata = error.metadata();
        assert_eq!(metadata.get("device_id"), Some(&"42".to_string()));
    }

    #[test]
    fn test_with_dtype() {
        let error = TorshError::ConversionError("unsupported".to_string()).with_dtype("f32");

        let metadata = error.metadata();
        assert_eq!(metadata.get("dtype"), Some(&"f32".to_string()));
    }

    #[test]
    fn test_chained_metadata() {
        let error = TorshError::InvalidShape("test".to_string())
            .with_operation("conv2d")
            .add_metadata("batch_size", "32")
            .add_shape_metadata("input_shape", &[32, 3, 224, 224])
            .with_device(0)
            .with_dtype("f32");

        let metadata = error.metadata();
        assert_eq!(metadata.get("operation"), Some(&"conv2d".to_string()));
        assert_eq!(metadata.get("batch_size"), Some(&"32".to_string()));
        assert_eq!(metadata.get("device_id"), Some(&"0".to_string()));
        assert_eq!(metadata.get("dtype"), Some(&"f32".to_string()));
        assert!(metadata.contains_key("input_shape"));
    }

    #[test]
    fn test_format_debug() {
        let error = TorshError::InvalidShape("test error".to_string())
            .with_operation("test_op")
            .add_metadata("key", "value");

        let debug_output = error.format_debug();
        assert!(debug_output.contains("Error:"));
        assert!(debug_output.contains("test error"));
        assert!(debug_output.contains("operation: test_op"));
        assert!(debug_output.contains("key: value"));
    }

    #[test]
    fn test_metadata_on_non_context_error() {
        // Test that adding metadata to a non-WithContext error converts it
        let error = TorshError::InvalidArgument("test".to_string());
        let metadata_before = error.metadata();
        assert!(metadata_before.is_empty());

        let error_with_metadata = error.add_metadata("new_key", "new_value");
        let metadata_after = error_with_metadata.metadata();
        assert_eq!(
            metadata_after.get("new_key"),
            Some(&"new_value".to_string())
        );
    }

    #[test]
    fn test_debug_context_availability() {
        let error_without_context = TorshError::InvalidShape("test".to_string());
        assert!(error_without_context.debug_context().is_none());

        let error_with_context =
            TorshError::InvalidShape("test".to_string()).with_context("during operation");
        assert!(error_with_context.debug_context().is_some());
    }
}
