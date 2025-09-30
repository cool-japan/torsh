//! Error types for FFI operations

use thiserror::Error;

/// FFI-specific error types
#[derive(Error, Debug)]
pub enum FfiError {
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

    use pyo3::exceptions::{PyException, PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::{create_exception, PyErr, Python};

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
        exc_type: &PyObject,
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

        Python::with_gil(|py| {
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

        Python::with_gil(|py| {
            create_enhanced_exception(
                py,
                &py.get_type::<NumericalError>().into(),
                message,
                context,
            )
        })
    }
}
