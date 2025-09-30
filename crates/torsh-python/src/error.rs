//! Error handling for PyTorch Python bindings

use pyo3::exceptions::{PyIndexError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use torsh_core::error::TorshError;

/// Python exception wrapper for ToRSh errors
#[pyclass]
pub struct TorshPyError {
    message: String,
}

#[pymethods]
impl TorshPyError {
    #[new]
    fn new(message: String) -> Self {
        Self { message }
    }

    fn __str__(&self) -> String {
        self.message.clone()
    }

    fn __repr__(&self) -> String {
        format!("TorshError('{}')", self.message)
    }
}

/// Convert ToRSh errors to Python exceptions
pub fn torsh_error_to_py_err(err: TorshError) -> PyErr {
    match err {
        // Modular error variants
        TorshError::Shape(shape_err) => {
            PyValueError::new_err(format!("Shape error: {}", shape_err))
        }
        TorshError::Index(index_err) => {
            PyIndexError::new_err(format!("Index error: {}", index_err))
        }
        TorshError::General(general_err) => {
            PyRuntimeError::new_err(format!("General error: {}", general_err))
        }

        // Error with context
        TorshError::WithContext { message, .. } => {
            PyRuntimeError::new_err(format!("ToRSh error: {}", message))
        }

        // Legacy compatibility variants
        TorshError::ShapeMismatch { expected, got } => PyValueError::new_err(format!(
            "Shape mismatch: expected {:?}, got {:?}",
            expected, got
        )),
        TorshError::BroadcastError { shape1, shape2 } => PyValueError::new_err(format!(
            "Broadcasting error: incompatible shapes {:?} and {:?}",
            shape1, shape2
        )),
        TorshError::IndexOutOfBounds { index, size } => {
            PyIndexError::new_err(format!("Index {} out of bounds for size {}", index, size))
        }
        TorshError::InvalidArgument(msg) => {
            PyValueError::new_err(format!("Invalid argument: {}", msg))
        }
        TorshError::IoError(msg) => PyOSError::new_err(format!("IO error: {}", msg)),
        TorshError::DeviceMismatch => {
            PyOSError::new_err("Device mismatch: tensors must be on the same device")
        }
        TorshError::NotImplemented(msg) => {
            PyRuntimeError::new_err(format!("Not implemented: {}", msg))
        }
        TorshError::SynchronizationError(msg) => {
            PyRuntimeError::new_err(format!("Synchronization error: {}", msg))
        }
        TorshError::AllocationError(msg) => {
            PyRuntimeError::new_err(format!("Memory allocation failed: {}", msg))
        }
        TorshError::InvalidOperation(msg) => {
            PyRuntimeError::new_err(format!("Invalid operation: {}", msg))
        }
        TorshError::ConversionError(msg) => {
            PyValueError::new_err(format!("Numeric conversion error: {}", msg))
        }
        TorshError::BackendError(msg) => PyRuntimeError::new_err(format!("Backend error: {}", msg)),
        TorshError::InvalidShape(msg) => PyValueError::new_err(format!("Invalid shape: {}", msg)),
        TorshError::RuntimeError(msg) => PyRuntimeError::new_err(format!("Runtime error: {}", msg)),

        // Additional missing variants - handle all with catch-all
        _ => PyRuntimeError::new_err(format!("ToRSh error: {}", err)),
    }
}

/// Result type for Python operations
pub type PyResult<T> = Result<T, PyErr>;

/// Convert ToRSh Result to Python Result
pub fn to_py_result<T>(result: torsh_core::error::Result<T>) -> PyResult<T> {
    result.map_err(torsh_error_to_py_err)
}

/// Macro for easy error conversion
#[macro_export]
macro_rules! py_result {
    ($expr:expr) => {
        $crate::error::to_py_result($expr)
    };
}

/// Register error types with Python module
pub fn register_error_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("TorshError", m.py().get_type::<TorshPyError>())?;
    Ok(())
}
