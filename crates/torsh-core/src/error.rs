//! Error types for ToRSh

use thiserror::Error;

/// Main error type for ToRSh operations
#[derive(Error, Debug)]
pub enum TorshError {
    /// Shape mismatch in tensor operations
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Invalid shape specification
    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    /// Index out of bounds
    #[error("Index out of bounds: index {index} is out of bounds for dimension with size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    /// Index error
    #[error("Index error: index {index} is out of bounds for size {size}")]
    IndexError { index: usize, size: usize },

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Device mismatch between tensors
    #[error("Device mismatch: tensors must be on the same device")]
    DeviceMismatch,

    /// Unsupported operation for the given dtype
    #[error("Unsupported operation {op} for dtype {dtype}")]
    UnsupportedOperation { op: String, dtype: String },

    /// Backend-specific error
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Autograd error
    #[error("Autograd error: {0}")]
    AutogradError(String),

    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    AllocationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Generic error from scirs2
    #[error("SciRS2 error: {0}")]
    SciRS2Error(String),

    /// Compute error
    #[error("Compute error: {0}")]
    ComputeError(String),

    /// Other errors
    #[error("{0}")]
    Other(String),
}

/// Result type alias for ToRSh operations
pub type Result<T> = std::result::Result<T, TorshError>;

// Standard library error conversions
impl From<std::io::Error> for TorshError {
    fn from(err: std::io::Error) -> Self {
        TorshError::IoError(err.to_string())
    }
}

// Serialization error conversions (when serde_json is available)
#[cfg(feature = "serde_json")]
impl From<serde_json::Error> for TorshError {
    fn from(err: serde_json::Error) -> Self {
        TorshError::SerializationError(err.to_string())
    }
}
