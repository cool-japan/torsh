//! Enhanced error types for FFI operations
//!
//! This module provides a comprehensive error handling system for FFI operations
//! with structured error codes, context, severity levels, and recovery suggestions.
//!
//! # Features
//!
//! - **Structured Error Codes**: Machine-readable error codes for automated handling
//! - **Error Context**: File, line, column, and operation context
//! - **Severity Levels**: Critical, Error, Warning, Info
//! - **Recovery Suggestions**: Actionable suggestions for error recovery
//! - **Error Categories**: Organized by domain (Tensor, Memory, Type, etc.)
//! - **Serialization**: JSON serialization for logging and debugging
//!
//! # Example
//!
//! ```rust,ignore
//! use torsh_ffi::error::{FfiError, ErrorBuilder, ErrorCode, Severity};
//!
//! // Create a detailed error with context
//! let error = ErrorBuilder::new(ErrorCode::ShapeMismatch)
//!     .message("Incompatible tensor shapes")
//!     .context("operation", "matmul")
//!     .context("expected_shape", "[2, 3]")
//!     .context("actual_shape", "[3, 2]")
//!     .source_location(file!(), line!(), column!())
//!     .severity(Severity::Error)
//!     .suggestion("Transpose one of the tensors or use broadcasting")
//!     .build();
//! ```

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// Structured error codes for machine-readable error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum ErrorCode {
    // Tensor errors (1000-1999)
    TensorCreationFailed = 1000,
    TensorOperationFailed = 1001,
    ShapeMismatch = 1002,
    DTypeMismatch = 1003,
    DeviceMismatch = 1004,
    TensorNotFound = 1005,

    // Memory errors (2000-2999)
    AllocationFailed = 2000,
    MemoryPoolExhausted = 2001,
    OutOfMemory = 2002,
    MemoryLeakDetected = 2003,
    InvalidPointer = 2004,
    DanglingPointer = 2005,

    // Type conversion errors (3000-3999)
    InvalidConversion = 3000,
    TypeNotSupported = 3001,
    PrecisionLoss = 3002,
    OverflowDetected = 3003,
    UnderflowDetected = 3004,

    // Parameter validation errors (4000-4999)
    InvalidParameter = 4000,
    ParameterOutOfRange = 4001,
    NullPointer = 4002,
    InvalidShape = 4003,
    InvalidDType = 4004,
    InvalidDevice = 4005,

    // Operation errors (5000-5999)
    OperationNotSupported = 5000,
    OperationFailed = 5001,
    BroadcastingFailed = 5002,
    MatrixNotInvertible = 5003,
    DivisionByZero = 5004,

    // Language binding errors (6000-6999)
    PythonError = 6000,
    JavaError = 6001,
    CSharpError = 6002,
    GoError = 6003,
    SwiftError = 6004,
    WasmError = 6005,

    // I/O errors (7000-7999)
    FileNotFound = 7000,
    FileReadError = 7001,
    FileWriteError = 7002,
    SerializationFailed = 7003,
    DeserializationFailed = 7004,

    // Module errors (8000-8999)
    ModuleNotFound = 8000,
    ModuleInitFailed = 8001,
    ModuleLoadError = 8002,

    // Cross-language errors (9000-9999)
    OwnershipConflict = 9000,
    DataRace = 9001,
    DeadlockDetected = 9002,

    // Unknown/Other errors (10000+)
    Unknown = 10000,
    Internal = 10001,
}

impl ErrorCode {
    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match *self as u32 {
            1000..=1999 => ErrorCategory::Tensor,
            2000..=2999 => ErrorCategory::Memory,
            3000..=3999 => ErrorCategory::TypeConversion,
            4000..=4999 => ErrorCategory::Validation,
            5000..=5999 => ErrorCategory::Operation,
            6000..=6999 => ErrorCategory::LanguageBinding,
            7000..=7999 => ErrorCategory::IO,
            8000..=8999 => ErrorCategory::Module,
            9000..=9999 => ErrorCategory::CrossLanguage,
            _ => ErrorCategory::Unknown,
        }
    }

    /// Get default severity for this error code
    pub fn default_severity(&self) -> Severity {
        match self {
            Self::AllocationFailed
            | Self::OutOfMemory
            | Self::DeadlockDetected
            | Self::DataRace => Severity::Critical,

            Self::ShapeMismatch
            | Self::DTypeMismatch
            | Self::InvalidParameter
            | Self::OperationFailed => Severity::Error,

            Self::PrecisionLoss | Self::MemoryLeakDetected => Severity::Warning,

            _ => Severity::Error,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} ({})", self, *self as u32)
    }
}

/// Error category for grouping related errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    Tensor,
    Memory,
    TypeConversion,
    Validation,
    Operation,
    LanguageBinding,
    IO,
    Module,
    CrossLanguage,
    Unknown,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor => write!(f, "Tensor"),
            Self::Memory => write!(f, "Memory"),
            Self::TypeConversion => write!(f, "Type Conversion"),
            Self::Validation => write!(f, "Validation"),
            Self::Operation => write!(f, "Operation"),
            Self::LanguageBinding => write!(f, "Language Binding"),
            Self::IO => write!(f, "I/O"),
            Self::Module => write!(f, "Module"),
            Self::CrossLanguage => write!(f, "Cross-Language"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Error severity level (ordered from lowest to highest severity)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Info - informational message
    Info,
    /// Warning - potential issue detected
    Warning,
    /// Error - operation failed but system can continue
    Error,
    /// Critical error - system cannot continue
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "CRITICAL"),
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// Source code location for error tracking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Enhanced error with structured information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedError {
    /// Error code for machine-readable identification
    pub code: ErrorCode,

    /// Human-readable error message
    pub message: String,

    /// Error severity level
    pub severity: Severity,

    /// Error category
    pub category: ErrorCategory,

    /// Source code location where error occurred
    pub location: Option<SourceLocation>,

    /// Additional context as key-value pairs
    pub context: HashMap<String, String>,

    /// Recovery suggestions
    pub suggestions: Vec<String>,

    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Chain of underlying errors
    pub causes: Vec<String>,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            severity: code.default_severity(),
            category: code.category(),
            location: None,
            context: HashMap::new(),
            suggestions: Vec::new(),
            timestamp: chrono::Utc::now(),
            causes: Vec::new(),
        }
    }

    /// Convert to legacy FfiError for backward compatibility
    pub fn to_ffi_error(&self) -> FfiError {
        FfiError::Enhanced(self.clone())
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Check if error is recoverable (not critical)
    pub fn is_recoverable(&self) -> bool {
        self.severity < Severity::Critical
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[{}] {} - {}", self.severity, self.code, self.message)?;

        if let Some(ref loc) = self.location {
            writeln!(f, "  at {}", loc)?;
        }

        if !self.context.is_empty() {
            writeln!(f, "  Context:")?;
            for (key, value) in &self.context {
                writeln!(f, "    {}: {}", key, value)?;
            }
        }

        if !self.suggestions.is_empty() {
            writeln!(f, "  Suggestions:")?;
            for suggestion in &self.suggestions {
                writeln!(f, "    - {}", suggestion)?;
            }
        }

        if !self.causes.is_empty() {
            writeln!(f, "  Caused by:")?;
            for cause in &self.causes {
                writeln!(f, "    {}", cause)?;
            }
        }

        Ok(())
    }
}

impl std::error::Error for EnhancedError {}

/// Builder for creating enhanced errors with fluent API
pub struct ErrorBuilder {
    error: EnhancedError,
}

impl ErrorBuilder {
    /// Create a new error builder with error code
    pub fn new(code: ErrorCode) -> Self {
        Self {
            error: EnhancedError::new(code, ""),
        }
    }

    /// Set error message
    pub fn message(mut self, message: impl Into<String>) -> Self {
        self.error.message = message.into();
        self
    }

    /// Set severity level
    pub fn severity(mut self, severity: Severity) -> Self {
        self.error.severity = severity;
        self
    }

    /// Add source code location
    pub fn source_location(mut self, file: &str, line: u32, column: u32) -> Self {
        self.error.location = Some(SourceLocation {
            file: file.to_string(),
            line,
            column,
        });
        self
    }

    /// Add context key-value pair
    pub fn context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.error.context.insert(key.into(), value.into());
        self
    }

    /// Add multiple context entries
    pub fn contexts(mut self, contexts: HashMap<String, String>) -> Self {
        self.error.context.extend(contexts);
        self
    }

    /// Add recovery suggestion
    pub fn suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.error.suggestions.push(suggestion.into());
        self
    }

    /// Add multiple suggestions
    pub fn suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.error.suggestions.extend(suggestions);
        self
    }

    /// Add underlying cause
    pub fn cause(mut self, cause: impl std::fmt::Display) -> Self {
        self.error.causes.push(cause.to_string());
        self
    }

    /// Build the enhanced error
    pub fn build(self) -> EnhancedError {
        self.error
    }

    /// Build and convert to FfiError
    pub fn build_ffi(self) -> FfiError {
        FfiError::Enhanced(self.error)
    }
}

/// FFI-specific error types (backward compatible)
#[derive(Error, Debug, Clone)]
pub enum FfiError {
    /// Enhanced error with full context and structure
    #[error("{0}")]
    Enhanced(EnhancedError),

    #[error("Tensor error: {message}")]
    Tensor { message: String },

    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Data type mismatch: expected {expected}, got {actual}")]
    DTypeMismatch { expected: String, actual: String },

    #[error("Invalid conversion: {message}")]
    InvalidConversion { message: String },

    #[error("Python error: {message}")]
    Python { message: String },

    #[error("NumPy error: {message}")]
    NumPy { message: String },

    #[error("Memory allocation failed: {message}")]
    AllocationFailed { message: String },

    #[error("Invalid parameter: {parameter} = {value}")]
    InvalidParameter { parameter: String, value: String },

    #[error("Operation not supported: {operation}")]
    UnsupportedOperation { operation: String },

    #[error("Module error: {message}")]
    Module { message: String },

    #[error("Memory pool error: {message}")]
    MemoryPool { message: String },

    #[error("Cross-language ownership conflict: {message}")]
    OwnershipConflict { message: String },

    #[error("Device transfer error: {message}")]
    DeviceTransfer { message: String },
}

#[cfg(feature = "python")]
impl From<FfiError> for pyo3::PyErr {
    fn from(err: FfiError) -> Self {
        match err {
            FfiError::Enhanced(enhanced) => {
                // Use the appropriate Python exception based on error category
                use pyo3::exceptions::*;
                let msg = enhanced.to_string();
                match enhanced.category {
                    ErrorCategory::Memory => PyMemoryError::new_err(msg),
                    ErrorCategory::TypeConversion => PyTypeError::new_err(msg),
                    ErrorCategory::Validation => PyValueError::new_err(msg),
                    ErrorCategory::IO => PyIOError::new_err(msg),
                    ErrorCategory::Module => PyModuleNotFoundError::new_err(msg),
                    _ => PyRuntimeError::new_err(msg),
                }
            }
            FfiError::Tensor { message } => pyo3::exceptions::PyRuntimeError::new_err(message),
            FfiError::ShapeMismatch { expected, actual } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                ))
            }
            FfiError::DTypeMismatch { expected, actual } => pyo3::exceptions::PyTypeError::new_err(
                format!("Data type mismatch: expected {}, got {}", expected, actual),
            ),
            FfiError::InvalidConversion { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            }
            FfiError::Python { message } => pyo3::exceptions::PyRuntimeError::new_err(message),
            FfiError::NumPy { message } => {
                pyo3::exceptions::PyRuntimeError::new_err(format!("NumPy error: {}", message))
            }
            FfiError::AllocationFailed { message } => {
                pyo3::exceptions::PyMemoryError::new_err(message)
            }
            FfiError::InvalidParameter { parameter, value } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid parameter: {} = {}",
                    parameter, value
                ))
            }
            FfiError::UnsupportedOperation { operation } => {
                pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "Operation not supported: {}",
                    operation
                ))
            }
            FfiError::Module { message } => {
                pyo3::exceptions::PyModuleNotFoundError::new_err(message)
            }
            FfiError::MemoryPool { message } => {
                pyo3::exceptions::PyMemoryError::new_err(format!("Memory pool error: {}", message))
            }
            FfiError::OwnershipConflict { message } => pyo3::exceptions::PyRuntimeError::new_err(
                format!("Ownership conflict: {}", message),
            ),
            FfiError::DeviceTransfer { message } => pyo3::exceptions::PyRuntimeError::new_err(
                format!("Device transfer error: {}", message),
            ),
        }
    }
}

// Error conversions
impl From<std::fmt::Error> for FfiError {
    fn from(err: std::fmt::Error) -> Self {
        FfiError::InvalidConversion {
            message: format!("Formatting error: {}", err),
        }
    }
}

impl From<std::io::Error> for FfiError {
    fn from(err: std::io::Error) -> Self {
        FfiError::InvalidConversion {
            message: format!("IO error: {}", err),
        }
    }
}

impl From<torsh_core::error::TorshError> for FfiError {
    fn from(err: torsh_core::error::TorshError) -> Self {
        FfiError::Tensor {
            message: format!("{}", err),
        }
    }
}

#[cfg(feature = "python")]
pub fn fmt_error_to_pyerr(err: std::fmt::Error) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Formatting error: {}", err))
}

#[cfg(feature = "python")]
pub fn torsh_error_to_pyerr(err: torsh_core::error::TorshError) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor error: {}", err))
}

/// Result type for FFI operations
pub type FfiResult<T> = Result<T, FfiError>;

#[cfg(feature = "python")]
pub mod python_exceptions {
    //! Custom Python exception classes for better error handling

    use pyo3::exceptions::PyException;
    use pyo3::prelude::*;
    use pyo3::types::PyAny;
    use pyo3::{create_exception, Py, PyErr, Python};

    // Custom exception types for ToRSh-specific errors
    create_exception!(
        torsh,
        TorshError,
        PyException,
        "Base exception for ToRSh operations"
    );
    create_exception!(torsh, TensorError, TorshError, "Tensor operation error");
    create_exception!(torsh, ShapeError, TorshError, "Tensor shape related error");
    create_exception!(torsh, DeviceError, TorshError, "Device operation error");
    create_exception!(
        torsh,
        NumericalError,
        TorshError,
        "Numerical computation error"
    );
    create_exception!(torsh, MemoryError, TorshError, "Memory management error");

    /// Enhanced error context for Python exceptions
    #[derive(Debug, Clone)]
    pub struct ErrorContext {
        pub operation: String,
        pub file: Option<String>,
        pub line: Option<u32>,
        pub suggestion: Option<String>,
        pub error_code: Option<i32>,
        pub recoverable: bool,
    }

    impl ErrorContext {
        pub fn new(operation: &str) -> Self {
            Self {
                operation: operation.to_string(),
                file: None,
                line: None,
                suggestion: None,
                error_code: None,
                recoverable: false,
            }
        }

        pub fn with_location(mut self, file: &str, line: u32) -> Self {
            self.file = Some(file.to_string());
            self.line = Some(line);
            self
        }

        pub fn with_suggestion(mut self, suggestion: &str) -> Self {
            self.suggestion = Some(suggestion.to_string());
            self
        }

        pub fn with_error_code(mut self, code: i32) -> Self {
            self.error_code = Some(code);
            self
        }

        pub fn recoverable(mut self) -> Self {
            self.recoverable = true;
            self
        }
    }

    /// Create enhanced Python exception with context
    pub fn create_enhanced_exception(
        py: Python<'_>,
        exc_type: &Py<PyAny>,
        message: &str,
        context: ErrorContext,
    ) -> PyErr {
        let exc = PyErr::new::<PyException, _>((message.to_string(),));

        // Add context attributes to the exception
        if let Ok(exception_obj) = exc_type.call1(py, (message.to_string(),)) {
            let _ = exception_obj.setattr(py, "operation", &context.operation);
            let _ = exception_obj.setattr(py, "recoverable", context.recoverable);

            if let Some(file) = &context.file {
                let _ = exception_obj.setattr(py, "source_file", file);
            }
            if let Some(line) = context.line {
                let _ = exception_obj.setattr(py, "source_line", line);
            }
            if let Some(suggestion) = &context.suggestion {
                let _ = exception_obj.setattr(py, "suggestion", suggestion);
            }
            if let Some(code) = context.error_code {
                let _ = exception_obj.setattr(py, "error_code", code);
            }
        }

        exc
    }

    /// Register exception types with Python module
    pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("TorshError", m.py().get_type::<TorshError>())?;
        m.add("TensorError", m.py().get_type::<TensorError>())?;
        m.add("ShapeError", m.py().get_type::<ShapeError>())?;
        m.add("DeviceError", m.py().get_type::<DeviceError>())?;
        m.add("NumericalError", m.py().get_type::<NumericalError>())?;
        m.add("MemoryError", m.py().get_type::<MemoryError>())?;
        Ok(())
    }

    /// Utility function to create tensor shape errors with helpful suggestions
    pub fn create_shape_error(expected: &[usize], actual: &[usize], operation: &str) -> PyErr {
        let message = format!(
            "Shape mismatch in {}: expected {:?}, got {:?}",
            operation, expected, actual
        );

        let suggestion = if expected.len() != actual.len() {
            Some(format!(
                "Expected tensor with {} dimensions, but got {} dimensions. Consider using reshape() or unsqueeze()/squeeze() operations.",
                expected.len(), actual.len()
            ))
        } else {
            let mismatched_dims: Vec<_> = expected
                .iter()
                .zip(actual.iter())
                .enumerate()
                .filter(|(_, (e, a))| e != a)
                .collect();

            if mismatched_dims.len() == 1 {
                let (dim, (exp, act)) = mismatched_dims[0];
                Some(format!(
                    "Dimension {} mismatch: expected {}, got {}. Consider using reshape(), transpose(), or broadcasting operations.",
                    dim, exp, act
                ))
            } else {
                Some("Multiple dimensions don't match. Verify tensor shapes and consider using broadcasting or reshape operations.".to_string())
            }
        };

        let context = ErrorContext::new(operation)
            .with_suggestion(&suggestion.unwrap_or_default())
            .recoverable();

        Python::attach(|py| {
            create_enhanced_exception(py, &py.get_type::<ShapeError>().into(), &message, context)
        })
    }

    /// Utility function to create numerical errors with recovery suggestions
    pub fn create_numerical_error(message: &str, operation: &str) -> PyErr {
        let suggestion = if message.contains("NaN") {
            Some("Check for division by zero, invalid operations, or uninitialized values. Consider using torch.isnan() to detect NaN values.".to_string())
        } else if message.contains("inf") || message.contains("infinity") {
            Some("Values became infinite. Consider gradient clipping, smaller learning rates, or numerical stability improvements.".to_string())
        } else if message.contains("overflow") {
            Some("Numerical overflow detected. Try using smaller values, different data types, or scaling your data.".to_string())
        } else {
            Some(
                "Numerical computation failed. Check input values and operation parameters."
                    .to_string(),
            )
        };

        let context = ErrorContext::new(operation)
            .with_suggestion(&suggestion.unwrap_or_default())
            .recoverable();

        Python::attach(|py| {
            create_enhanced_exception(
                py,
                &py.get_type::<NumericalError>().into(),
                message,
                context,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_category() {
        assert_eq!(ErrorCode::ShapeMismatch.category(), ErrorCategory::Tensor);
        assert_eq!(
            ErrorCode::AllocationFailed.category(),
            ErrorCategory::Memory
        );
        assert_eq!(
            ErrorCode::InvalidConversion.category(),
            ErrorCategory::TypeConversion
        );
        assert_eq!(
            ErrorCode::InvalidParameter.category(),
            ErrorCategory::Validation
        );
    }

    #[test]
    fn test_error_code_severity() {
        assert_eq!(
            ErrorCode::OutOfMemory.default_severity(),
            Severity::Critical
        );
        assert_eq!(ErrorCode::ShapeMismatch.default_severity(), Severity::Error);
        assert_eq!(
            ErrorCode::PrecisionLoss.default_severity(),
            Severity::Warning
        );
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::Error);
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Warning > Severity::Info);
    }

    #[test]
    fn test_enhanced_error_creation() {
        let error = EnhancedError::new(ErrorCode::ShapeMismatch, "Test error");
        assert_eq!(error.code, ErrorCode::ShapeMismatch);
        assert_eq!(error.message, "Test error");
        assert_eq!(error.severity, Severity::Error);
        assert_eq!(error.category, ErrorCategory::Tensor);
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_error_builder() {
        let error = ErrorBuilder::new(ErrorCode::ShapeMismatch)
            .message("Incompatible shapes")
            .context("operation", "matmul")
            .context("expected", "[2, 3]")
            .context("actual", "[3, 2]")
            .suggestion("Transpose one of the tensors")
            .severity(Severity::Error)
            .build();

        assert_eq!(error.message, "Incompatible shapes");
        assert_eq!(error.context.get("operation"), Some(&"matmul".to_string()));
        assert_eq!(error.suggestions.len(), 1);
        assert!(error.is_recoverable());
    }

    #[test]
    fn test_error_with_location() {
        let error = ErrorBuilder::new(ErrorCode::InvalidParameter)
            .message("Invalid value")
            .source_location("test.rs", 42, 10)
            .build();

        assert!(error.location.is_some());
        let loc = error.location.unwrap();
        assert_eq!(loc.file, "test.rs");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.column, 10);
    }

    #[test]
    fn test_error_with_causes() {
        let error = ErrorBuilder::new(ErrorCode::OperationFailed)
            .message("Operation failed")
            .cause("Underlying IO error")
            .cause("File not found")
            .build();

        assert_eq!(error.causes.len(), 2);
        assert!(error.causes.contains(&"Underlying IO error".to_string()));
    }

    #[test]
    fn test_error_display() {
        let error = ErrorBuilder::new(ErrorCode::ShapeMismatch)
            .message("Test error")
            .context("op", "test")
            .suggestion("Fix it")
            .build();

        let display = error.to_string();
        assert!(display.contains("ShapeMismatch"));
        assert!(display.contains("Test error"));
        assert!(display.contains("op: test"));
        assert!(display.contains("Fix it"));
    }

    #[test]
    fn test_error_json_serialization() {
        let error = EnhancedError::new(ErrorCode::InvalidConversion, "Test");
        let json = error.to_json();
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("InvalidConversion"));
        assert!(json_str.contains("Test"));
    }

    #[test]
    fn test_error_recoverability() {
        let critical = EnhancedError::new(ErrorCode::OutOfMemory, "OOM");
        assert!(!critical.is_recoverable());

        let warning = EnhancedError::new(ErrorCode::PrecisionLoss, "Precision");
        assert!(warning.is_recoverable());
    }

    #[test]
    fn test_ffi_error_enhanced_variant() {
        let enhanced = EnhancedError::new(ErrorCode::TensorCreationFailed, "Failed");
        let ffi_error = FfiError::Enhanced(enhanced);

        match ffi_error {
            FfiError::Enhanced(e) => {
                assert_eq!(e.code, ErrorCode::TensorCreationFailed);
            }
            _ => panic!("Expected Enhanced variant"),
        }
    }

    #[test]
    fn test_error_builder_fluent_api() {
        let error = ErrorBuilder::new(ErrorCode::MatrixNotInvertible)
            .message("Matrix is singular")
            .context("determinant", "0.0")
            .context("condition_number", "inf")
            .suggestion("Check matrix values")
            .suggestion("Try adding regularization")
            .source_location(file!(), line!(), column!())
            .build();

        assert_eq!(error.suggestions.len(), 2);
        assert_eq!(error.context.len(), 2);
        assert!(error.location.is_some());
    }
}
