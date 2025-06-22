//! Error types for CPU backend

use thiserror::Error;
use torsh_backends::BackendError;

/// CPU backend specific error types
#[derive(Error, Debug)]
pub enum CpuBackendError {
    #[error("Memory allocation failed: {message}")]
    MemoryAllocation { message: String },

    #[error("Buffer error: {message}")]
    Buffer { message: String },

    #[error("Kernel execution error: {message}")]
    KernelExecution { message: String },

    #[error("Thread pool error: {message}")]
    ThreadPool { message: String },

    #[error("SIMD operation error: {message}")]
    SimdError { message: String },

    #[error("Optimization error: {message}")]
    Optimization { message: String },

    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },

    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<CpuBackendError> for BackendError {
    fn from(err: CpuBackendError) -> Self {
        BackendError::ComputeError {
            reason: err.to_string(),
        }
    }
}

/// Result type for CPU backend operations
pub type CpuResult<T> = Result<T, CpuBackendError>;
