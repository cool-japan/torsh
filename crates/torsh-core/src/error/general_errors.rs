//! General error types for ToRSh
//!
//! This module contains miscellaneous error variants including I/O errors,
//! configuration errors, runtime errors, and other general-purpose errors.

use thiserror::Error;

/// General error variants
#[derive(Error, Debug, Clone)]
pub enum GeneralError {
    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Iteration error for iterative algorithms
    #[error("Iteration error: {0}")]
    IterationError(String),

    /// Not implemented error for placeholder functionality
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Invalid operation error
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Invalid state error
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Other errors
    #[error("{0}")]
    Other(String),

    /// Device mismatch between tensors
    #[error("Device mismatch: tensors must be on the same device")]
    DeviceMismatch,

    /// Unsupported operation for given data type
    #[error("Unsupported operation '{op}' for data type '{dtype}'")]
    UnsupportedOperation { op: String, dtype: String },

    /// Backend-specific error
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Device error
    #[error("Device error: {0}")]
    DeviceError(String),

    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    AllocationError(String),

    /// Autograd error
    #[error("Autograd error: {0}")]
    AutogradError(String),

    /// Type mismatch error
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    /// Dimension error for tensors
    #[error("Dimension error: {0}")]
    DimensionError(String),

    /// Threading error
    #[error("Threading error: {0}")]
    ThreadError(String),

    /// Thread synchronization error (mutex poisoned, channel error, etc.)
    #[error("Thread synchronization error: {0}")]
    SynchronizationError(String),

    /// Numeric conversion error
    #[error("Numeric conversion error: {0}")]
    ConversionError(String),

    /// Data loading error
    #[error("Data loading error: {0}")]
    DataError(String),

    /// Model error
    #[error("Model error: {0}")]
    ModelError(String),

    /// Security error
    #[error("Security error: {0}")]
    SecurityError(String),

    /// Generic error from scirs2
    #[error("SciRS2 error: {0}")]
    SciRS2Error(String),

    /// Compute error
    #[error("Compute error: {0}")]
    ComputeError(String),
}

impl GeneralError {
    pub fn category(&self) -> crate::error::core::ErrorCategory {
        match self {
            Self::IoError(_) => crate::error::core::ErrorCategory::Io,
            Self::ConfigError(_) => crate::error::core::ErrorCategory::Configuration,
            Self::DeviceMismatch | Self::DeviceError(_) | Self::BackendError(_) => {
                crate::error::core::ErrorCategory::Device
            }
            Self::AllocationError(_) => crate::error::core::ErrorCategory::Memory,
            Self::TypeMismatch { .. } | Self::ConversionError(_) => {
                crate::error::core::ErrorCategory::DataType
            }
            Self::ThreadError(_) | Self::SynchronizationError(_) => {
                crate::error::core::ErrorCategory::Threading
            }
            Self::InvalidArgument(_) => crate::error::core::ErrorCategory::UserInput,
            _ => crate::error::core::ErrorCategory::Internal,
        }
    }
}
