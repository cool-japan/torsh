//! Error types for CUDA backend

use thiserror::Error;
use torsh_backends::BackendError;

/// CUDA-specific error types
#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA runtime error: {0}")]
    Runtime(#[from] cust::CudaError),
    
    #[error("CUDA device error: {message}")]
    Device { message: String },
    
    #[error("CUDA memory error: {message}")]
    Memory { message: String },
    
    #[error("CUDA kernel launch error: {message}")]
    KernelLaunch { message: String },
    
    #[error("CUDA stream error: {message}")]
    Stream { message: String },
    
    #[error("cuDNN error: {0}")]
    CudnnError(String),
    
    #[error("cuBLAS error: {message}")]
    CublasError { message: String },
    
    #[error("NCCL error: {message}")]
    NcclError { message: String },
    
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
    
    #[error("Invalid device ID: {device_id}")]
    InvalidDevice { device_id: usize },
    
    #[error("Out of memory: requested {requested} bytes")]
    OutOfMemory { requested: usize },
    
    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation { operation: String },
    
    #[error("Context error: {message}")]
    Context { message: String },
}

impl From<CudaError> for BackendError {
    fn from(err: CudaError) -> Self {
        BackendError::ComputeError {
            reason: err.to_string(),
        }
    }
}

/// Result type for CUDA operations
pub type CudaResult<T> = Result<T, CudaError>;