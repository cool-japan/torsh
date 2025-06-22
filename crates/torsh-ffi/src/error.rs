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
    DTypeMismatch {
        expected: String,
        actual: String,
    },
    
    #[error("Invalid conversion: {message}")]
    InvalidConversion { message: String },
    
    #[error("Python error: {message}")]
    Python { message: String },
    
    #[error("NumPy error: {message}")]
    NumPy { message: String },
    
    #[error("Memory allocation failed: {message}")]
    AllocationFailed { message: String },
    
    #[error("Invalid parameter: {parameter} = {value}")]
    InvalidParameter {
        parameter: String,
        value: String,
    },
    
    #[error("Operation not supported: {operation}")]
    UnsupportedOperation { operation: String },
    
    #[error("Module error: {message}")]
    Module { message: String },
}

#[cfg(feature = "python")]
impl From<FfiError> for pyo3::PyErr {
    fn from(err: FfiError) -> Self {
        match err {
            FfiError::Tensor { message } => pyo3::exceptions::PyRuntimeError::new_err(message),
            FfiError::ShapeMismatch { expected, actual } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Shape mismatch: expected {:?}, got {:?}", expected, actual
                ))
            },
            FfiError::DTypeMismatch { expected, actual } => {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Data type mismatch: expected {}, got {}", expected, actual
                ))
            },
            FfiError::InvalidConversion { message } => {
                pyo3::exceptions::PyValueError::new_err(message)
            },
            FfiError::Python { message } => {
                pyo3::exceptions::PyRuntimeError::new_err(message)
            },
            FfiError::NumPy { message } => {
                pyo3::exceptions::PyRuntimeError::new_err(format!("NumPy error: {}", message))
            },
            FfiError::AllocationFailed { message } => {
                pyo3::exceptions::PyMemoryError::new_err(message)
            },
            FfiError::InvalidParameter { parameter, value } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid parameter: {} = {}", parameter, value
                ))
            },
            FfiError::UnsupportedOperation { operation } => {
                pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "Operation not supported: {}", operation
                ))
            },
            FfiError::Module { message } => {
                pyo3::exceptions::PyModuleNotFoundError::new_err(message)
            },
        }
    }
}

/// Result type for FFI operations
pub type FfiResult<T> = Result<T, FfiError>;