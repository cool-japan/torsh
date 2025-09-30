//! Error handling for Metal backend using unified TorshError

pub use crate::error::{BackendError, BackendResult, ErrorContext, ErrorContextExt};
pub use torsh_core::error::TorshError;

/// Metal-specific error type alias
pub type MetalError = BackendError;

/// Metal-specific result type alias
pub type Result<T> = BackendResult<T>;

/// Helper functions for Metal-specific error creation
pub mod metal_errors {
    use super::*;
    use torsh_core::device::DeviceType;

    /// Create a Metal API error with context
    pub fn metal_api_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("metal_api")
            .with_backend("Metal")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create an invalid device error with context
    pub fn invalid_device_error(device_id: usize) -> TorshError {
        let context = ErrorContext::new("device_validation")
            .with_backend("Metal")
            .with_device(format!("metal:{}", device_id))
            .with_details(format!("Invalid Metal device index: {}", device_id));

        TorshError::InvalidArgument(context.format())
    }

    /// Create an unsupported device type error with context
    pub fn unsupported_device_error(device_type: DeviceType) -> TorshError {
        let context = ErrorContext::new("device_type_validation")
            .with_backend("Metal")
            .with_details(format!("Unsupported device type: {:?}", device_type));

        TorshError::InvalidArgument(context.format())
    }

    /// Create a buffer allocation error with context
    pub fn buffer_allocation_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("buffer_allocation")
            .with_backend("Metal")
            .with_details(format!(
                "Failed to allocate Metal buffer: {}",
                message.into()
            ));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::AllocationError(context.format())
    }

    /// Create a shader compilation error with context
    pub fn shader_compilation_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("shader_compilation")
            .with_backend("Metal")
            .with_details(format!(
                "Metal shader compilation error: {}",
                message.into()
            ));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a kernel execution error with context
    pub fn kernel_execution_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("kernel_execution")
            .with_backend("Metal")
            .with_details(format!("Metal kernel execution error: {}", message.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create an unsupported operation error with context
    pub fn unsupported_operation_error(
        operation: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("operation_validation")
            .with_backend("Metal")
            .with_details(format!(
                "Operation not supported on Metal backend: {}",
                operation.into()
            ));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::InvalidArgument(context.format())
    }

    /// Create an unsupported data type error with context
    pub fn unsupported_dtype_error(
        dtype: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("dtype_validation")
            .with_backend("Metal")
            .with_details(format!("Unsupported data type for Metal: {}", dtype.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::InvalidArgument(context.format())
    }

    /// Create a shape mismatch error with context
    pub fn shape_mismatch_error(
        expected: Vec<usize>,
        got: Vec<usize>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("shape_validation")
            .with_backend("Metal")
            .with_details(format!(
                "Shape mismatch: expected {:?}, got {:?}",
                expected, got
            ));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::InvalidShape(context.format())
    }

    /// Create a Metal Performance Shaders error with context
    pub fn mps_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("mps")
            .with_backend("Metal")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a Metal command buffer error with context
    pub fn command_buffer_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("command_buffer")
            .with_backend("Metal")
            .with_details(message.into());

        if let Some(device_id) = device_id {
            context = context.with_device(format!("metal:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }
}
