//! Error handling for CPU backend using unified TorshError

pub use crate::error::{BackendError, BackendResult as CpuResult, ErrorContext, ErrorContextExt};
pub use torsh_core::error::TorshError;

/// Helper functions for CPU-specific error creation
pub mod cpu_errors {
    use super::*;

    /// Create a memory allocation error with CPU backend context
    pub fn memory_allocation_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("memory_allocation")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::AllocationError(context.format())
    }

    /// Create a buffer error with CPU backend context
    pub fn buffer_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("buffer_operation")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::BackendError(context.format())
    }

    /// Create a kernel execution error with CPU backend context
    pub fn kernel_execution_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("kernel_execution")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::ComputeError(context.format())
    }

    /// Create a thread pool error with CPU backend context
    pub fn thread_pool_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("thread_pool")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::BackendError(context.format())
    }

    /// Create a SIMD operation error with CPU backend context
    pub fn simd_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("simd_operation")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::ComputeError(context.format())
    }

    /// Create an optimization error with CPU backend context
    pub fn optimization_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("optimization")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::BackendError(context.format())
    }

    /// Create an invalid parameter error with CPU backend context
    pub fn invalid_parameter_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("parameter_validation")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::InvalidArgument(context.format())
    }

    /// Create a parsing error with CPU backend context
    pub fn parsing_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("parsing")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::BackendError(context.format())
    }

    /// Create a serialization error with CPU backend context
    pub fn serialization_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("serialization")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::BackendError(context.format())
    }

    /// Create an IO error with CPU backend context (string message version)
    pub fn io_error(message: impl Into<String>) -> TorshError {
        let context = ErrorContext::new("io_operation")
            .with_backend("CPU")
            .with_details(message.into());
        TorshError::BackendError(context.format())
    }

    /// Convert IO errors to TorshError with CPU backend context
    pub fn io_error_from_std(err: std::io::Error, operation: &str) -> TorshError {
        let context = ErrorContext::new(operation)
            .with_backend("CPU")
            .with_details(err.to_string());
        TorshError::BackendError(context.format())
    }
}
