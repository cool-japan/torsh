//! Error handling for WebGPU backend using unified TorshError

#[cfg(feature = "webgpu")]
use crate::webgpu::wgpu;

use std::fmt;

pub use crate::error::{BackendError, ErrorContext, ErrorContextExt};
pub use torsh_core::error::TorshError;

/// WebGPU-specific error type
#[derive(Debug, Clone)]
pub enum WebGpuError {
    /// No WebGPU adapter found
    NoAdapterFound,
    /// WebGPU not available on this platform
    NotAvailable,
    /// Device creation failed
    DeviceCreation(String),
    /// Buffer creation failed
    BufferCreation(String),
    /// Pipeline creation failed
    PipelineCreation(String),
    /// Shader compilation failed
    ShaderCompilation(String),
    /// Compute pass execution failed
    ComputePassError(String),
    /// Buffer mapping failed
    BufferMapping(String),
    /// Memory allocation failed
    MemoryAllocation(String),
    /// Invalid buffer usage
    InvalidBufferUsage(String),
    /// Invalid workgroup size
    InvalidWorkgroupSize(String),
    /// Unsupported data type
    UnsupportedDataType(String),
    /// Resource not found
    ResourceNotFound(String),
    /// Operation timeout
    OperationTimeout(String),
    /// Device lost
    DeviceLost(String),
    /// Validation failed
    ValidationFailed(String),
    /// Unsupported feature
    UnsupportedFeature(String),
    /// Invalid shader source
    InvalidShaderSource(String),
    /// Invalid argument
    InvalidArgument(String),
    /// Runtime error
    RuntimeError(String),
    /// Initialization error
    InitializationError(String),
    /// Generic backend error
    Backend(String),
}

impl fmt::Display for WebGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WebGpuError::NoAdapterFound => write!(f, "No WebGPU adapter found"),
            WebGpuError::NotAvailable => write!(f, "WebGPU not available on this platform"),
            WebGpuError::DeviceCreation(msg) => write!(f, "Device creation failed: {}", msg),
            WebGpuError::BufferCreation(msg) => write!(f, "Buffer creation failed: {}", msg),
            WebGpuError::PipelineCreation(msg) => write!(f, "Pipeline creation failed: {}", msg),
            WebGpuError::ShaderCompilation(msg) => write!(f, "Shader compilation failed: {}", msg),
            WebGpuError::ComputePassError(msg) => {
                write!(f, "Compute pass execution failed: {}", msg)
            }
            WebGpuError::BufferMapping(msg) => write!(f, "Buffer mapping failed: {}", msg),
            WebGpuError::MemoryAllocation(msg) => write!(f, "Memory allocation failed: {}", msg),
            WebGpuError::InvalidBufferUsage(msg) => write!(f, "Invalid buffer usage: {}", msg),
            WebGpuError::InvalidWorkgroupSize(msg) => write!(f, "Invalid workgroup size: {}", msg),
            WebGpuError::UnsupportedDataType(msg) => write!(f, "Unsupported data type: {}", msg),
            WebGpuError::ResourceNotFound(msg) => write!(f, "Resource not found: {}", msg),
            WebGpuError::OperationTimeout(msg) => write!(f, "Operation timeout: {}", msg),
            WebGpuError::DeviceLost(msg) => write!(f, "Device lost: {}", msg),
            WebGpuError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            WebGpuError::UnsupportedFeature(msg) => write!(f, "Unsupported feature: {}", msg),
            WebGpuError::InvalidShaderSource(msg) => write!(f, "Invalid shader source: {}", msg),
            WebGpuError::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            WebGpuError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            WebGpuError::InitializationError(msg) => write!(f, "Initialization error: {}", msg),
            WebGpuError::Backend(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for WebGpuError {}

impl From<WebGpuError> for TorshError {
    fn from(err: WebGpuError) -> Self {
        match err {
            WebGpuError::NoAdapterFound => {
                TorshError::BackendError("No WebGPU adapter found".to_string())
            }
            WebGpuError::NotAvailable => {
                TorshError::BackendError("WebGPU not available".to_string())
            }
            WebGpuError::DeviceCreation(msg) => {
                TorshError::BackendError(format!("Device creation failed: {}", msg))
            }
            WebGpuError::BufferCreation(msg) => {
                TorshError::AllocationError(format!("Buffer creation failed: {}", msg))
            }
            WebGpuError::PipelineCreation(msg) => {
                TorshError::ComputeError(format!("Pipeline creation failed: {}", msg))
            }
            WebGpuError::ShaderCompilation(msg) => {
                TorshError::ComputeError(format!("Shader compilation failed: {}", msg))
            }
            WebGpuError::ComputePassError(msg) => {
                TorshError::ComputeError(format!("Compute pass execution failed: {}", msg))
            }
            WebGpuError::BufferMapping(msg) => {
                TorshError::ComputeError(format!("Buffer mapping failed: {}", msg))
            }
            WebGpuError::MemoryAllocation(msg) => {
                TorshError::AllocationError(format!("Memory allocation failed: {}", msg))
            }
            WebGpuError::InvalidBufferUsage(msg) => {
                TorshError::InvalidArgument(format!("Invalid buffer usage: {}", msg))
            }
            WebGpuError::InvalidWorkgroupSize(msg) => {
                TorshError::InvalidArgument(format!("Invalid workgroup size: {}", msg))
            }
            WebGpuError::UnsupportedDataType(msg) => {
                TorshError::InvalidArgument(format!("Unsupported data type: {}", msg))
            }
            WebGpuError::ResourceNotFound(msg) => {
                TorshError::BackendError(format!("Resource not found: {}", msg))
            }
            WebGpuError::OperationTimeout(msg) => {
                TorshError::BackendError(format!("Operation timeout: {}", msg))
            }
            WebGpuError::DeviceLost(msg) => {
                TorshError::BackendError(format!("Device lost: {}", msg))
            }
            WebGpuError::ValidationFailed(msg) => {
                TorshError::InvalidArgument(format!("Validation failed: {}", msg))
            }
            WebGpuError::UnsupportedFeature(msg) => {
                TorshError::InvalidArgument(format!("Unsupported feature: {}", msg))
            }
            WebGpuError::InvalidShaderSource(msg) => {
                TorshError::ComputeError(format!("Invalid shader source: {}", msg))
            }
            WebGpuError::InvalidArgument(msg) => {
                TorshError::InvalidArgument(format!("Invalid argument: {}", msg))
            }
            WebGpuError::RuntimeError(msg) => {
                TorshError::BackendError(format!("Runtime error: {}", msg))
            }
            WebGpuError::InitializationError(msg) => {
                TorshError::BackendError(format!("Initialization error: {}", msg))
            }
            WebGpuError::Backend(msg) => {
                TorshError::BackendError(format!("Backend error: {}", msg))
            }
        }
    }
}

/// Result type for WebGPU operations
pub type WebGpuResult<T> = Result<T, WebGpuError>;

/// Validate workgroup size against device limits
pub fn validate_workgroup_size(
    workgroup_size: (u32, u32, u32),
    limits: &wgpu::Limits,
) -> WebGpuResult<()> {
    let (x, y, z) = workgroup_size;

    if x > limits.max_compute_workgroup_size_x {
        return Err(WebGpuError::InvalidWorkgroupSize(format!(
            "Workgroup size X ({}) exceeds limit ({})",
            x, limits.max_compute_workgroup_size_x
        )));
    }

    if y > limits.max_compute_workgroup_size_y {
        return Err(WebGpuError::InvalidWorkgroupSize(format!(
            "Workgroup size Y ({}) exceeds limit ({})",
            y, limits.max_compute_workgroup_size_y
        )));
    }

    if z > limits.max_compute_workgroup_size_z {
        return Err(WebGpuError::InvalidWorkgroupSize(format!(
            "Workgroup size Z ({}) exceeds limit ({})",
            z, limits.max_compute_workgroup_size_z
        )));
    }

    let total_size = x as u64 * y as u64 * z as u64;
    if total_size > limits.max_compute_invocations_per_workgroup as u64 {
        return Err(WebGpuError::InvalidWorkgroupSize(format!(
            "Total workgroup invocations ({}) exceeds limit ({})",
            total_size, limits.max_compute_invocations_per_workgroup
        )));
    }

    Ok(())
}

/// Helper functions for WebGPU-specific error creation
pub mod webgpu_errors {
    use super::*;
    use torsh_core::DType;

    /// Create a "no adapter found" error with context
    pub fn no_adapter_found_error() -> TorshError {
        let context = ErrorContext::new("adapter_enumeration")
            .with_backend("WebGPU")
            .with_details("No WebGPU adapter found");

        TorshError::BackendError(context.format())
    }

    /// Create a device creation error with context
    pub fn device_creation_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("device_creation")
            .with_backend("WebGPU")
            .with_details(format!("Device creation failed: {}", message.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::BackendError(context.format())
    }

    /// Create a buffer creation error with context
    pub fn buffer_creation_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("buffer_creation")
            .with_backend("WebGPU")
            .with_details(format!("Buffer creation failed: {}", message.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::AllocationError(context.format())
    }

    /// Create a pipeline creation error with context
    pub fn pipeline_creation_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("pipeline_creation")
            .with_backend("WebGPU")
            .with_details(format!("Pipeline creation failed: {}", message.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a shader compilation error with context
    pub fn shader_compilation_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("shader_compilation")
            .with_backend("WebGPU")
            .with_details(format!("Shader compilation failed: {}", message.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a compute pass execution error with context
    pub fn compute_pass_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("compute_pass")
            .with_backend("WebGPU")
            .with_details(format!("Compute pass execution failed: {}", message.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a buffer mapping error with context
    pub fn buffer_mapping_error(
        message: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("buffer_mapping")
            .with_backend("WebGPU")
            .with_details(format!("Buffer mapping failed: {}", message.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::ComputeError(context.format())
    }

    /// Create a memory allocation error with context
    pub fn memory_allocation_error(
        size: u64,
        alignment: u64,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("memory_allocation")
            .with_backend("WebGPU")
            .with_details(format!(
                "Memory allocation failed: size={}, alignment={}",
                size, alignment
            ));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::AllocationError(context.format())
    }

    /// Create an invalid buffer usage error with context
    pub fn invalid_buffer_usage_error(
        usage: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("buffer_usage_validation")
            .with_backend("WebGPU")
            .with_details(format!("Invalid buffer usage: {}", usage.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::InvalidArgument(context.format())
    }

    /// Create an invalid workgroup size error with context
    pub fn invalid_workgroup_size_error(
        size: (u32, u32, u32),
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("workgroup_size_validation")
            .with_backend("WebGPU")
            .with_details(format!("Invalid compute workgroup size: {:?}", size));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::InvalidArgument(context.format())
    }

    /// Create an unsupported data type error with context
    pub fn unsupported_datatype_error(dtype: DType, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("datatype_validation")
            .with_backend("WebGPU")
            .with_details(format!("Unsupported data type: {:?}", dtype));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::InvalidArgument(context.format())
    }

    /// Create a resource not found error with context
    pub fn resource_not_found_error(
        resource: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("resource_lookup")
            .with_backend("WebGPU")
            .with_details(format!("Resource not found: {}", resource.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::BackendError(context.format())
    }

    /// Create an operation timeout error with context
    pub fn operation_timeout_error(
        operation: impl Into<String>,
        device_id: Option<usize>,
    ) -> TorshError {
        let mut context = ErrorContext::new("operation_timeout")
            .with_backend("WebGPU")
            .with_details(format!("Operation timeout: {}", operation.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::BackendError(context.format())
    }

    /// Create a device lost error with context
    pub fn device_lost_error(message: impl Into<String>, device_id: Option<usize>) -> TorshError {
        let mut context = ErrorContext::new("device_lost")
            .with_backend("WebGPU")
            .with_details(format!("Device lost: {}", message.into()));

        if let Some(device_id) = device_id {
            context = context.with_device(format!("wgpu:{}", device_id));
        }

        TorshError::BackendError(context.format())
    }
}
