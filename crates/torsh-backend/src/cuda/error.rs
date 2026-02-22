//! Error handling for CUDA backend using unified TorshError

pub use crate::error::{BackendError, BackendResult as CudaResult, ErrorContext, ErrorContextExt};
pub use torsh_core::error::TorshError;

/// Type alias for CUDA errors (uses unified TorshError)
pub type CudaError = TorshError;

/// Helper functions for CUDA-specific error creation
pub mod cuda_errors {
    use super::*;

    /// Create a CUDA runtime error with context
    pub fn runtime_error(
        cuda_error: impl std::fmt::Display,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("cuda_runtime")
            .with_backend("CUDA")
            .with_details(cuda_error.to_string());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a CUDA device error with context
    pub fn device_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("cuda_device")
            .with_backend("CUDA")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::BackendError(context.format())
    }

    /// Create a CUDA memory error with context
    pub fn memory_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("cuda_memory")
            .with_backend("CUDA")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::AllocationError(context.format())
    }

    /// Create a CUDA kernel launch error with context
    pub fn kernel_launch_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("cuda_kernel_launch")
            .with_backend("CUDA")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a CUDA stream error with context
    pub fn stream_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("cuda_stream")
            .with_backend("CUDA")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a cuDNN error with context
    pub fn cudnn_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("cudnn")
            .with_backend("CUDA")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a cuBLAS error with context
    pub fn cublas_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("cublas")
            .with_backend("CUDA")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create an NCCL error with context
    pub fn nccl_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("nccl")
            .with_backend("CUDA")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create an invalid device error with context
    pub fn invalid_device_error(device_id: usize) -> TorshError {
        let context = ErrorContext::new("device_validation")
            .with_backend("CUDA")
            .with_device(format!("cuda:{}", device_id))
            .with_details(format!("Invalid device ID: {}", device_id));

        TorshError::InvalidArgument(context.format())
    }

    /// Create an out of memory error with context
    pub fn out_of_memory_error(requested: usize, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("memory_allocation")
            .with_backend("CUDA")
            .with_details(format!("Out of memory: requested {} bytes", requested));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::AllocationError(context.format())
    }

    /// Create an unsupported operation error with context
    pub fn unsupported_operation_error(
        operation: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("operation_validation")
            .with_backend("CUDA")
            .with_details(format!("Unsupported operation: {}", operation.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::InvalidArgument(context.format())
    }

    /// Create a CUDA context error with context
    pub fn context_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("cuda_context")
            .with_backend("CUDA")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("cuda:{}", device_id));
        }

        TorshError::BackendError(context.format())
    }
}

/// Helper function to convert cust::error::CudaError to TorshError
/// Use .map_err(cuda_error_to_torsh) instead of ? for cust functions
pub fn cuda_error_to_torsh(error: cust::error::CudaError) -> TorshError {
    TorshError::ComputeError(format!("CUDA error: {}", error))
}

/// Helper function to convert cust::error::CudaError to BackendError
pub fn cuda_error_to_backend(error: cust::error::CudaError) -> crate::BackendError {
    crate::BackendError::Runtime {
        message: format!("CUDA error: {}", error),
    }
}

/// Extension trait for converting cust errors to TorshError
pub trait CustResultExt<T> {
    /// Convert cust error to TorshError
    fn cuda_err(self) -> Result<T, TorshError>;
    /// Convert cust error to BackendError
    fn backend_err(self) -> Result<T, crate::BackendError>;
    /// Convert cust error to CudaResult
    fn cuda_result(self) -> CudaResult<T>;
}

impl<T> CustResultExt<T> for Result<T, cust::error::CudaError> {
    fn cuda_err(self) -> Result<T, TorshError> {
        self.map_err(cuda_error_to_torsh)
    }

    fn backend_err(self) -> Result<T, crate::BackendError> {
        self.map_err(cuda_error_to_backend)
    }

    fn cuda_result(self) -> CudaResult<T> {
        self.map_err(cuda_error_to_torsh)
    }
}
