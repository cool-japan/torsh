//! Core backend trait and implementations

use crate::{Buffer, BufferDescriptor, Device, Kernel, KernelDescriptor, MemoryManager, Profiler};
use torsh_core::{
    device::DeviceType,
    dtype::DType,
    error::{Result, TorshError},
};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Result type for backend operations
pub type BackendResult<T> = Result<T>;

/// Error types specific to backend operations
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Backend not available: {reason}")]
    NotAvailable { reason: String },

    #[error("Device not found: {device}")]
    DeviceNotFound { device: String },

    #[error("Kernel compilation failed: {error}")]
    KernelCompilationFailed { error: String },

    #[error("Buffer allocation failed: {size} bytes")]
    BufferAllocationFailed { size: usize },

    #[error("Synchronization failed: {reason}")]
    SynchronizationFailed { reason: String },

    #[error("Unsupported operation: {operation}")]
    UnsupportedOperation { operation: String },

    #[error("Memory error: {reason}")]
    MemoryError { reason: String },

    #[error("Compute error: {reason}")]
    ComputeError { reason: String },
}

impl From<BackendError> for TorshError {
    fn from(err: BackendError) -> Self {
        TorshError::BackendError(err.to_string())
    }
}

/// The main backend trait that all compute backends must implement
#[async_trait::async_trait]
pub trait Backend: Send + Sync {
    /// Get the device type this backend supports
    fn device_type(&self) -> DeviceType;

    /// Get the name of this backend
    fn name(&self) -> &str;

    /// Check if this backend is available on the current system
    fn is_available(&self) -> BackendResult<bool>;

    /// Initialize the backend
    async fn initialize(&mut self) -> BackendResult<()>;

    /// Shutdown the backend and cleanup resources
    async fn shutdown(&mut self) -> BackendResult<()>;

    /// Get available devices for this backend
    fn devices(&self) -> BackendResult<Vec<Device>>;

    /// Get the default device
    fn default_device(&self) -> BackendResult<Device>;

    /// Create a device from an ID
    fn create_device(&self, device_id: usize) -> BackendResult<Device>;

    /// Create a buffer on the specified device
    fn create_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer>;

    /// Create a kernel from source or bytecode
    fn create_kernel(
        &self,
        device: &Device,
        descriptor: &KernelDescriptor,
    ) -> BackendResult<Kernel>;

    /// Get the memory manager for a device
    fn memory_manager(&self, device: &Device) -> BackendResult<Box<dyn MemoryManager>>;

    /// Get the profiler for performance analysis
    fn profiler(&self) -> BackendResult<Box<dyn Profiler>>;

    /// Synchronize operations on a device (wait for completion)
    async fn synchronize(&self, device: &Device) -> BackendResult<()>;

    /// Copy data between buffers
    async fn copy_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> BackendResult<()>;

    /// Copy data from host memory to device buffer
    async fn copy_to_device(
        &self,
        src: &[u8],
        dst: &Buffer,
        dst_offset: usize,
    ) -> BackendResult<()>;

    /// Copy data from device buffer to host memory
    async fn copy_from_device(
        &self,
        src: &Buffer,
        dst: &mut [u8],
        src_offset: usize,
    ) -> BackendResult<()>;

    /// Execute a kernel with the given parameters
    async fn execute_kernel(
        &self,
        kernel: &Kernel,
        buffers: &[&Buffer],
        uniform_data: &[u8],
        workgroup_size: (u32, u32, u32),
        workgroup_count: (u32, u32, u32),
    ) -> BackendResult<()>;

    /// Get backend-specific capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Get performance hints for optimization
    fn performance_hints(&self) -> PerformanceHints;
}

/// Backend capabilities description
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Maximum buffer size in bytes
    pub max_buffer_size: usize,

    /// Maximum number of compute units
    pub max_compute_units: usize,

    /// Maximum workgroup size
    pub max_workgroup_size: (u32, u32, u32),

    /// Supported data types
    pub supported_dtypes: Vec<DType>,

    /// Whether the backend supports async operations
    pub supports_async: bool,

    /// Whether the backend supports unified memory
    pub supports_unified_memory: bool,

    /// Whether the backend supports sub-buffers
    pub supports_sub_buffers: bool,

    /// Whether the backend supports kernel compilation caching
    pub supports_kernel_caching: bool,

    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,

    /// Compute throughput in GFLOPS
    pub compute_throughput_gflops: f32,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            max_buffer_size: 1024 * 1024 * 1024, // 1GB
            max_compute_units: 1,
            max_workgroup_size: (256, 1, 1),
            supported_dtypes: vec![DType::F32, DType::F64, DType::I32, DType::I64],
            supports_async: false,
            supports_unified_memory: false,
            supports_sub_buffers: false,
            supports_kernel_caching: false,
            memory_bandwidth_gbps: 10.0,
            compute_throughput_gflops: 100.0,
        }
    }
}

/// Performance optimization hints
#[derive(Debug, Clone)]
pub struct PerformanceHints {
    /// Preferred workgroup size for compute kernels
    pub preferred_workgroup_size: (u32, u32, u32),

    /// Optimal memory alignment in bytes
    pub memory_alignment: usize,

    /// Whether to prefer vectorized operations
    pub prefer_vectorized: bool,

    /// Whether to use asynchronous operations when possible
    pub prefer_async: bool,

    /// Optimal batch size for operations
    pub optimal_batch_size: usize,

    /// Whether to cache compiled kernels
    pub cache_kernels: bool,
}

impl Default for PerformanceHints {
    fn default() -> Self {
        Self {
            preferred_workgroup_size: (64, 1, 1),
            memory_alignment: 16,
            prefer_vectorized: true,
            prefer_async: false,
            optimal_batch_size: 32,
            cache_kernels: true,
        }
    }
}

/// Trait for backend factories
pub trait BackendFactory: Send + Sync {
    /// Create a new backend instance
    fn create(&self) -> BackendResult<Box<dyn Backend>>;

    /// Get the device type this factory creates backends for
    fn device_type(&self) -> DeviceType;

    /// Check if this backend type is available
    fn is_available(&self) -> bool;
}
