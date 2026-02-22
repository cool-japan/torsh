//! Core backend trait and implementations

use crate::memory::MemoryManager;
use crate::profiler::Profiler;
use crate::{Buffer, BufferDescriptor, Device, Kernel, KernelDescriptor};
use torsh_core::{device::DeviceType, dtype::DType, error::TorshError};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Result type for backend operations  
pub type BackendResult<T> = Result<T, TorshError>;

// BackendError removed - use TorshError directly

/// Core backend information and capabilities
pub trait BackendCore: Send + Sync + std::fmt::Debug {
    /// Get the device type this backend supports
    fn device_type(&self) -> DeviceType;

    /// Get the name of this backend
    fn name(&self) -> &str;

    /// Check if this backend is available on the current system
    fn is_available(&self) -> BackendResult<bool>;

    /// Get backend-specific capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Get performance hints for optimization
    fn performance_hints(&self) -> PerformanceHints;
}

/// Backend lifecycle management
#[async_trait::async_trait]
pub trait BackendLifecycle: Send + Sync {
    /// Initialize the backend
    async fn initialize(&mut self) -> BackendResult<()>;

    /// Shutdown the backend and cleanup resources
    async fn shutdown(&mut self) -> BackendResult<()>;

    /// Check if the backend is initialized
    fn is_initialized(&self) -> bool;
}

/// Device management operations
pub trait BackendDeviceManager: Send + Sync {
    /// Get available devices for this backend
    fn devices(&self) -> BackendResult<Vec<Device>>;

    /// Get the default device
    fn default_device(&self) -> BackendResult<Device>;

    /// Create a device from an ID
    fn create_device(&self, device_id: usize) -> BackendResult<Device>;

    /// Get device count
    fn device_count(&self) -> BackendResult<usize>;

    /// Check if a device is available
    fn is_device_available(&self, device_id: usize) -> bool;
}

/// Resource creation and management with enhanced lifetime safety
pub trait BackendResourceManager: Send + Sync {
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

    /// Get the memory manager for a device with better lifetime bounds
    fn memory_manager(
        &self,
        device: &Device,
    ) -> BackendResult<Box<dyn MemoryManager + Send + Sync>>;

    /// Get the profiler for performance analysis with better lifetime bounds
    fn profiler(&self) -> BackendResult<Box<dyn Profiler + Send + Sync>>;

    /// Create a buffer with automatic cleanup (replaces generic scoped resource for object safety)
    fn create_scoped_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer>;
}

/// Advanced resource management operations with generic support (separate trait for type safety)
pub trait BackendAdvancedResourceManager: Send + Sync {
    /// Create a resource with automatic cleanup and type safety
    fn create_resource_with_cleanup<T, F>(
        &self,
        device: &Device,
        factory: F,
        cleanup: impl FnOnce(&T) + Send + 'static,
    ) -> BackendResult<ManagedResource<T>>
    where
        T: Send + Sync + 'static,
        F: FnOnce(&Device) -> BackendResult<T>;
}

/// Execution operations
#[async_trait::async_trait]
pub trait BackendExecutor: Send + Sync {
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
}

/// Specialized operations support
pub trait BackendOperations: Send + Sync {
    /// Get FFT operations for this backend
    fn fft_ops(&self) -> Box<dyn crate::fft::FftOps>;

    /// Get convolution operations for this backend
    fn convolution_ops(&self) -> Box<dyn crate::convolution::ConvolutionOps>;

    /// Get RNN operations for this backend
    fn rnn_ops(&self) -> Box<dyn crate::rnn::RnnOps>;

    /// Get sparse operations for this backend
    fn sparse_ops(&self) -> Box<dyn crate::sparse_ops::SparseOps<f32>>;

    /// Get quantization operations for this backend
    fn quantization_ops(&self) -> Box<dyn crate::quantization::QuantizationOps>;

    /// Get operations bundle for efficient access to multiple operation types
    fn operations_bundle(&self) -> OperationsBundle;
}

/// The main backend trait that combines all backend functionality
pub trait Backend:
    BackendCore
    + BackendLifecycle
    + BackendDeviceManager
    + BackendResourceManager
    + BackendExecutor
    + BackendOperations
    + BackendOps
{
    /// Get a reference to this backend as BackendCore
    fn as_core(&self) -> &dyn BackendCore;

    /// Get a mutable reference to this backend as BackendLifecycle
    fn as_lifecycle(&mut self) -> &mut dyn BackendLifecycle;

    /// Get a reference to this backend as BackendDeviceManager
    fn as_device_manager(&self) -> &dyn BackendDeviceManager;

    /// Get a reference to this backend as BackendResourceManager
    fn as_resource_manager(&self) -> &dyn BackendResourceManager;

    /// Get a reference to this backend as BackendExecutor
    fn as_executor(&self) -> &dyn BackendExecutor;

    /// Get a reference to this backend as BackendOperations
    fn as_operations(&self) -> &dyn BackendOperations;
}

/// RAII wrapper for backend resources with automatic cleanup and better type safety
pub struct ScopedResource<'a, T> {
    resource: Option<T>,
    cleanup: Option<Box<dyn FnOnce(T) + Send + 'a>>,
}

impl<'a, T> ScopedResource<'a, T> {
    /// Create a new scoped resource
    pub fn new(resource: T) -> Self {
        Self {
            resource: Some(resource),
            cleanup: None,
        }
    }

    /// Create a new scoped resource with custom cleanup
    pub fn new_with_cleanup<F>(resource: T, cleanup: F) -> Self
    where
        F: FnOnce(T) + Send + 'a,
    {
        Self {
            resource: Some(resource),
            cleanup: Some(Box::new(cleanup)),
        }
    }

    /// Get a reference to the resource
    pub fn get(&self) -> Option<&T> {
        self.resource.as_ref()
    }

    /// Get a mutable reference to the resource
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.resource.as_mut()
    }

    /// Take ownership of the resource (prevents cleanup)
    pub fn take(mut self) -> Option<T> {
        self.cleanup = None; // Prevent cleanup
        self.resource.take()
    }

    /// Execute a function with the resource, ensuring cleanup happens even if function panics
    pub fn with_resource<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&T) -> R,
    {
        self.resource.as_ref().map(f)
    }

    /// Check if the resource is available
    pub fn is_available(&self) -> bool {
        self.resource.is_some()
    }
}

impl<'a, T> Drop for ScopedResource<'a, T> {
    fn drop(&mut self) {
        if let (Some(resource), Some(cleanup)) = (self.resource.take(), self.cleanup.take()) {
            cleanup(resource);
        }
    }
}

/// Managed resource with automatic cleanup (no lifetime parameter for better usability)
pub struct ManagedResource<T> {
    resource: Option<T>,
    cleanup: Option<Box<dyn FnOnce(&T) + Send + 'static>>,
}

impl<T> ManagedResource<T> {
    /// Create a new managed resource
    pub fn new(resource: T) -> Self {
        Self {
            resource: Some(resource),
            cleanup: None,
        }
    }

    /// Create a new managed resource with custom cleanup
    pub fn new_with_cleanup<F>(resource: T, cleanup: F) -> Self
    where
        F: FnOnce(&T) + Send + 'static,
    {
        Self {
            resource: Some(resource),
            cleanup: Some(Box::new(cleanup)),
        }
    }

    /// Get a reference to the resource
    pub fn get(&self) -> Option<&T> {
        self.resource.as_ref()
    }

    /// Get a mutable reference to the resource
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.resource.as_mut()
    }

    /// Take ownership of the resource (prevents cleanup)
    pub fn take(mut self) -> Option<T> {
        self.cleanup = None; // Prevent cleanup
        self.resource.take()
    }

    /// Execute a function with the resource, ensuring cleanup happens even if function panics
    pub fn with_resource<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&T) -> R,
    {
        self.resource.as_ref().map(f)
    }

    /// Check if the resource is available
    pub fn is_available(&self) -> bool {
        self.resource.is_some()
    }
}

impl<T> Drop for ManagedResource<T> {
    fn drop(&mut self) {
        if let (Some(resource), Some(cleanup)) = (self.resource.as_ref(), self.cleanup.take()) {
            cleanup(resource);
        }
    }
}

// Ensure ManagedResource is Send and Sync when T is Send and Sync
unsafe impl<T: Send> Send for ManagedResource<T> {}
unsafe impl<T: Sync> Sync for ManagedResource<T> {}

/// Bundle of operations for efficient access
pub struct OperationsBundle {
    pub fft: Box<dyn crate::fft::FftOps>,
    pub convolution: Box<dyn crate::convolution::ConvolutionOps>,
    pub rnn: Box<dyn crate::rnn::RnnOps>,
    pub sparse: Box<dyn crate::sparse_ops::SparseOps<f32>>,
    pub quantization: Box<dyn crate::quantization::QuantizationOps>,
}

impl OperationsBundle {
    /// Create a new operations bundle
    pub fn new(
        fft: Box<dyn crate::fft::FftOps>,
        convolution: Box<dyn crate::convolution::ConvolutionOps>,
        rnn: Box<dyn crate::rnn::RnnOps>,
        sparse: Box<dyn crate::sparse_ops::SparseOps<f32>>,
        quantization: Box<dyn crate::quantization::QuantizationOps>,
    ) -> Self {
        Self {
            fft,
            convolution,
            rnn,
            sparse,
            quantization,
        }
    }
}

/// Backend capabilities description with enhanced extensibility
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

    /// Extended capabilities for better extensibility
    pub extended_capabilities: ExtendedCapabilities,
}

/// Extended capabilities for future extensibility
#[derive(Debug, Clone)]
pub struct ExtendedCapabilities {
    /// Supported tensor shapes (None means no limits)
    pub max_tensor_dims: Option<usize>,

    /// Supported precision modes
    pub precision_modes: Vec<PrecisionMode>,

    /// Hardware-specific features
    pub hardware_features: Vec<HardwareFeature>,

    /// Memory hierarchy information
    pub memory_hierarchy: MemoryHierarchy,

    /// Execution model capabilities
    pub execution_model: ExecutionModel,

    /// Custom capabilities for backend-specific features
    pub custom_capabilities: std::collections::HashMap<String, CapabilityValue>,
}

/// Precision modes supported by the backend
#[derive(Debug, Clone, PartialEq)]
pub enum PrecisionMode {
    /// 16-bit floating point
    F16,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// Mixed precision (automatic)
    Mixed,
    /// Custom precision with bits
    Custom(u8),
}

/// Hardware-specific features
#[derive(Debug, Clone, PartialEq)]
pub enum HardwareFeature {
    /// Tensor cores (like CUDA Tensor Cores)
    TensorCores,
    /// Vector processing units
    VectorUnits,
    /// Shared memory
    SharedMemory,
    /// Constant memory
    ConstantMemory,
    /// Atomic operations
    AtomicOperations,
    /// Cooperative groups
    CooperativeGroups,
    /// Dynamic parallelism
    DynamicParallelism,
    /// Custom feature
    Custom(String),
}

/// Memory hierarchy information
#[derive(Debug, Clone, Default)]
pub struct MemoryHierarchy {
    /// L1 cache size in bytes
    pub l1_cache_size: Option<usize>,
    /// L2 cache size in bytes
    pub l2_cache_size: Option<usize>,
    /// L3 cache size in bytes
    pub l3_cache_size: Option<usize>,
    /// Shared memory size in bytes
    pub shared_memory_size: Option<usize>,
    /// Memory access latency in cycles
    pub memory_latency_cycles: Option<u32>,
    /// Memory bandwidth per core in GB/s
    pub memory_bandwidth_per_core: Option<f32>,
}

/// Execution model capabilities
#[derive(Debug, Clone)]
pub struct ExecutionModel {
    /// Whether the backend supports SIMD operations
    pub supports_simd: bool,
    /// Whether the backend supports SIMT operations
    pub supports_simt: bool,
    /// Whether the backend supports task parallelism
    pub supports_task_parallelism: bool,
    /// Whether the backend supports data parallelism
    pub supports_data_parallelism: bool,
    /// Maximum concurrent streams/queues
    pub max_concurrent_streams: Option<u32>,
    /// Whether the backend supports out-of-order execution
    pub supports_out_of_order: bool,
}

/// Capability value for custom capabilities
#[derive(Debug, Clone)]
pub enum CapabilityValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<CapabilityValue>),
}

impl Default for ExtendedCapabilities {
    fn default() -> Self {
        Self {
            max_tensor_dims: Some(8),
            precision_modes: vec![PrecisionMode::F32],
            hardware_features: vec![],
            memory_hierarchy: MemoryHierarchy::default(),
            execution_model: ExecutionModel::default(),
            custom_capabilities: std::collections::HashMap::new(),
        }
    }
}

impl Default for ExecutionModel {
    fn default() -> Self {
        Self {
            supports_simd: false,
            supports_simt: false,
            supports_task_parallelism: true,
            supports_data_parallelism: true,
            max_concurrent_streams: Some(1),
            supports_out_of_order: false,
        }
    }
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
            extended_capabilities: ExtendedCapabilities::default(),
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

/// Backend type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BackendType {
    /// Automatic backend selection
    Auto,
    /// CPU backend
    Cpu,
    /// CUDA GPU backend
    Cuda,
    /// Metal GPU backend
    Metal,
    /// ROCm/HIP GPU backend
    Rocm,
    /// WebGPU backend
    WebGpu,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Auto => write!(f, "Auto"),
            BackendType::Cpu => write!(f, "CPU"),
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::Rocm => write!(f, "ROCm"),
            BackendType::WebGpu => write!(f, "WebGPU"),
        }
    }
}

/// Backend operations trait - groups of related operations
pub trait BackendOps: Send + Sync {
    /// Get the backend type
    fn backend_type(&self) -> BackendType;

    /// Get available operations for this backend
    fn available_ops(&self) -> Vec<&str>;

    /// Check if an operation is supported
    fn supports_op(&self, op_name: &str) -> bool;

    /// Check if FFT operations are supported
    fn supports_fft(&self) -> bool;

    /// Check if convolution operations are supported
    fn supports_convolution(&self) -> bool;

    /// Check if RNN operations are supported
    fn supports_rnn(&self) -> bool;

    /// Check if sparse operations are supported
    fn supports_sparse(&self) -> bool;

    /// Check if quantization operations are supported
    fn supports_quantization(&self) -> bool;

    /// Get operation-specific capabilities
    fn operation_capabilities(
        &self,
        op_name: &str,
    ) -> Option<std::collections::HashMap<String, CapabilityValue>>;
}

/// Backend extensibility trait for adding custom functionality
pub trait BackendExtension: Send + Sync {
    /// Get the extension name
    fn extension_name(&self) -> &str;

    /// Get the extension version
    fn extension_version(&self) -> &str;

    /// Check if this extension is compatible with the backend
    fn is_compatible_with(&self, backend: &dyn BackendCore) -> bool;

    /// Initialize the extension with the backend
    fn initialize(&mut self, backend: &dyn Backend) -> BackendResult<()>;

    /// Shutdown the extension
    fn shutdown(&mut self) -> BackendResult<()>;

    /// Get extension-specific capabilities
    fn capabilities(&self) -> std::collections::HashMap<String, CapabilityValue>;

    /// Handle custom operations
    fn handle_operation(
        &self,
        op_name: &str,
        args: &[CapabilityValue],
    ) -> BackendResult<CapabilityValue>;
}

/// Backend registry for managing extensions with enhanced lifetime safety
pub struct BackendExtensionRegistry {
    extensions: std::collections::HashMap<String, Box<dyn BackendExtension>>,
    /// Track initialization state for proper cleanup
    initialized_extensions: std::collections::HashSet<String>,
}

impl BackendExtensionRegistry {
    /// Create a new extension registry
    pub fn new() -> Self {
        Self {
            extensions: std::collections::HashMap::new(),
            initialized_extensions: std::collections::HashSet::new(),
        }
    }

    /// Register a new extension with ownership transfer
    pub fn register_extension(
        &mut self,
        extension: Box<dyn BackendExtension>,
    ) -> BackendResult<()> {
        let name = extension.extension_name().to_string();
        if self.extensions.contains_key(&name) {
            return Err(TorshError::BackendError(format!(
                "Extension '{}' is already registered",
                name
            )));
        }
        self.extensions.insert(name, extension);
        Ok(())
    }

    /// Get an extension by name with proper lifetime bounds
    pub fn get_extension(&self, name: &str) -> Option<&dyn BackendExtension> {
        self.extensions.get(name).map(|e| e.as_ref())
    }

    /// Get a mutable extension by name with proper lifetime bounds
    pub fn get_extension_mut(&mut self, name: &str) -> Option<&mut Box<dyn BackendExtension>> {
        self.extensions.get_mut(name)
    }

    /// Get all registered extensions
    pub fn extensions(&self) -> Vec<&str> {
        self.extensions.keys().map(|s| s.as_str()).collect()
    }

    /// Initialize all extensions with a backend with proper error handling
    pub fn initialize_all(&mut self, backend: &dyn Backend) -> BackendResult<Vec<String>> {
        let mut failed_extensions = Vec::new();

        for (name, extension) in self.extensions.iter_mut() {
            if extension.is_compatible_with(backend.as_core()) {
                match extension.initialize(backend) {
                    Ok(()) => {
                        self.initialized_extensions.insert(name.clone());
                    }
                    Err(e) => {
                        failed_extensions.push(format!("{}: {}", name, e));
                    }
                }
            }
        }

        if failed_extensions.is_empty() {
            Ok(vec![])
        } else {
            Err(TorshError::BackendError(format!(
                "Failed to initialize extensions: {}",
                failed_extensions.join(", ")
            )))
        }
    }

    /// Shutdown all extensions with proper error handling
    pub fn shutdown_all(&mut self) -> BackendResult<Vec<String>> {
        let mut failed_extensions = Vec::new();

        // Only shutdown initialized extensions
        for (name, extension) in self.extensions.iter_mut() {
            if self.initialized_extensions.contains(name) {
                if let Err(e) = extension.shutdown() {
                    failed_extensions.push(format!("{}: {}", name, e));
                } else {
                    self.initialized_extensions.remove(name);
                }
            }
        }

        if failed_extensions.is_empty() {
            Ok(vec![])
        } else {
            Err(TorshError::BackendError(format!(
                "Failed to shutdown extensions: {}",
                failed_extensions.join(", ")
            )))
        }
    }

    /// Remove an extension by name with proper cleanup
    pub fn remove_extension(&mut self, name: &str) -> Option<Box<dyn BackendExtension>> {
        // Ensure extension is shutdown before removal
        if let Some(extension) = self.extensions.get_mut(name) {
            if self.initialized_extensions.contains(name) {
                let _ = extension.shutdown(); // Ignore errors during removal
                self.initialized_extensions.remove(name);
            }
        }
        self.extensions.remove(name)
    }

    /// Check if an extension is registered
    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.contains_key(name)
    }

    /// Get the number of registered extensions
    pub fn len(&self) -> usize {
        self.extensions.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }
}

impl Default for BackendExtensionRegistry {
    fn default() -> Self {
        Self::new()
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

    /// Get the priority of this backend (higher is better)
    fn priority(&self) -> u32;

    /// Get the capabilities of backends created by this factory
    fn capabilities(&self) -> BackendCapabilities;
}

/// Device enumeration and selection utilities
pub struct DeviceEnumerator;

impl DeviceEnumerator {
    /// Enumerate all available devices across all backends
    pub fn enumerate_all_devices() -> BackendResult<Vec<(DeviceType, Vec<Device>)>> {
        let mut all_devices = Vec::new();

        // CPU devices are always available
        #[cfg(feature = "cpu")]
        {
            if let Ok(cpu_backend) = crate::cpu::CpuBackend::new() {
                if let Ok(devices) = cpu_backend.devices() {
                    all_devices.push((DeviceType::Cpu, devices));
                }
            }
        }

        // CUDA devices
        #[cfg(feature = "cuda")]
        {
            // Since this is a fallback CUDA implementation, no devices are available
            // This avoids type mismatch issues with the fallback implementation
        }

        // Metal devices
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            if let Ok(metal_backend) = crate::metal::MetalBackend::new() {
                if let Ok(devices) = metal_backend.devices() {
                    all_devices.push((DeviceType::Metal(0), devices));
                }
            }
        }

        // WebGPU devices
        #[cfg(feature = "webgpu")]
        {
            let webgpu_backend = crate::webgpu::WebGpuBackend::with_default_config();
            if let Ok(devices) = webgpu_backend.devices() {
                all_devices.push((DeviceType::Wgpu(0), devices));
            }
        }

        Ok(all_devices)
    }

    /// Find the best available device based on performance characteristics
    pub fn find_best_device() -> BackendResult<(DeviceType, Device)> {
        let all_devices = Self::enumerate_all_devices()?;

        if all_devices.is_empty() {
            return Err(TorshError::BackendError("No devices available".to_string()));
        }

        // Prioritize backends: CUDA > Metal > ROCm > WebGPU > CPU
        let backend_priorities = [
            DeviceType::Cuda(0),
            DeviceType::Metal(0),
            DeviceType::Wgpu(0),
            DeviceType::Cpu,
        ];

        for preferred_type in &backend_priorities {
            for (device_type, devices) in &all_devices {
                if Self::device_types_match(device_type, preferred_type) && !devices.is_empty() {
                    // Find the device with highest compute performance
                    let best_device = devices
                        .iter()
                        .max_by(|a, b| {
                            a.info()
                                .peak_gflops
                                .partial_cmp(&b.info().peak_gflops)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .cloned()
                        .expect("devices should not be empty after is_empty check");

                    return Ok((*device_type, best_device));
                }
            }
        }

        // Fallback to first available device
        let (device_type, devices) = &all_devices[0];
        if !devices.is_empty() {
            Ok((*device_type, devices[0].clone()))
        } else {
            Err(TorshError::BackendError(
                "No usable devices found".to_string(),
            ))
        }
    }

    /// Helper to match device types ignoring IDs
    fn device_types_match(a: &DeviceType, b: &DeviceType) -> bool {
        matches!(
            (a, b),
            (DeviceType::Cpu, DeviceType::Cpu)
                | (DeviceType::Cuda(_), DeviceType::Cuda(_))
                | (DeviceType::Metal(_), DeviceType::Metal(_))
                | (DeviceType::Wgpu(_), DeviceType::Wgpu(_))
        )
    }

    /// Get devices by type
    pub fn get_devices_by_type(device_type: DeviceType) -> BackendResult<Vec<Device>> {
        match device_type {
            #[cfg(feature = "cpu")]
            DeviceType::Cpu => {
                let cpu_backend = crate::cpu::CpuBackend::new()?;
                cpu_backend.devices()
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_device_id) => {
                // Since this is a fallback CUDA implementation, return empty vector
                Ok(vec![])
            }
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            DeviceType::Metal(_) => {
                let metal_backend = crate::metal::MetalBackend::new()?;
                metal_backend.devices()
            }
            #[cfg(feature = "webgpu")]
            DeviceType::Wgpu(_) => {
                let webgpu_backend = crate::webgpu::WebGpuBackend::with_default_config();
                webgpu_backend.devices()
            }
            #[allow(unreachable_patterns)]
            _ => Err(TorshError::BackendError(format!(
                "Backend type {device_type:?} not available"
            ))),
        }
    }

    /// Check if a specific device type is available
    pub fn is_device_type_available(device_type: DeviceType) -> bool {
        match device_type {
            #[cfg(feature = "cpu")]
            DeviceType::Cpu => true,
            #[cfg(cuda_available)]
            DeviceType::Cuda(device_id) => {
                crate::cuda::CudaBackend::new(crate::cuda::CudaBackendConfig {
                    device_id: device_id as usize,
                    ..Default::default()
                })
                .is_ok()
            }
            #[cfg(all(feature = "cuda", not(cuda_available)))]
            DeviceType::Cuda(_) => false, // CUDA feature enabled but not available on this platform
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            DeviceType::Metal(_) => crate::metal::MetalBackend::new().is_ok(),
            #[cfg(feature = "webgpu")]
            DeviceType::Wgpu(_) => true, // WebGPU backend with default config is always available
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}

/// Backend plugin system for dynamic backend loading
pub trait BackendPlugin: Send + Sync + std::fmt::Debug {
    /// Get the name of this plugin
    fn name(&self) -> &str;

    /// Get the version of this plugin
    fn version(&self) -> &str;

    /// Create a backend instance from this plugin
    fn create_backend(&self) -> BackendResult<Box<dyn Backend>>;

    /// Check if this plugin is compatible with the current system
    fn is_compatible(&self) -> bool;

    /// Get the device types this plugin supports
    fn supported_device_types(&self) -> Vec<DeviceType>;

    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;
}

/// Plugin metadata information
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub supported_architectures: Vec<String>,
    pub required_features: Vec<String>,
    pub optional_features: Vec<String>,
}

/// Backend resource monitoring and management trait for better RAII patterns
pub trait BackendResourceMonitor: Send + Sync {
    /// Get the current resource usage
    fn resource_usage(&self) -> ResourceUsage;

    /// Set resource limits
    fn set_resource_limits(&mut self, limits: ResourceLimits) -> BackendResult<()>;

    /// Get resource limits
    fn resource_limits(&self) -> ResourceLimits;

    /// Cleanup unused resources
    fn cleanup_resources(&mut self) -> BackendResult<()>;

    /// Get resource statistics
    fn resource_statistics(&self) -> ResourceStatistics;

    /// Enable resource monitoring
    fn enable_monitoring(&mut self) -> BackendResult<()>;

    /// Disable resource monitoring
    fn disable_monitoring(&mut self) -> BackendResult<()>;

    /// Check if monitoring is enabled
    fn is_monitoring_enabled(&self) -> bool;
}

/// Current resource usage information
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_used: usize,
    pub buffers_allocated: usize,
    pub kernels_cached: usize,
    pub active_streams: usize,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: f32,
}

/// Resource limits configuration
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_memory: Option<usize>,
    pub max_buffers: Option<usize>,
    pub max_kernels: Option<usize>,
    pub max_streams: Option<usize>,
    pub memory_pressure_threshold: f32,
}

/// Resource statistics over time
#[derive(Debug, Clone)]
pub struct ResourceStatistics {
    pub peak_memory_usage: usize,
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub average_buffer_size: f32,
    pub cache_hit_rate: f32,
    pub allocation_failure_count: u32,
}

/// Backend registry for managing multiple backends and plugins
pub struct BackendRegistry {
    backends: std::collections::HashMap<String, Box<dyn BackendPlugin>>,
    default_backend: Option<String>,
}

impl BackendRegistry {
    /// Create a new backend registry
    pub fn new() -> Self {
        Self {
            backends: std::collections::HashMap::new(),
            default_backend: None,
        }
    }

    /// Register a new backend plugin
    pub fn register_plugin(&mut self, plugin: Box<dyn BackendPlugin>) -> BackendResult<()> {
        let name = plugin.name().to_string();

        // Check if plugin is compatible
        if !plugin.is_compatible() {
            return Err(TorshError::BackendError(format!(
                "Plugin {name} is not compatible with current system"
            )));
        }

        self.backends.insert(name.clone(), plugin);

        // Set as default if this is the first compatible plugin
        if self.default_backend.is_none() {
            self.default_backend = Some(name);
        }

        Ok(())
    }

    /// Get available backend names
    pub fn available_backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }

    /// Create a backend by name
    pub fn create_backend(&self, name: &str) -> BackendResult<Box<dyn Backend>> {
        if let Some(plugin) = self.backends.get(name) {
            plugin.create_backend()
        } else {
            Err(TorshError::BackendError(format!(
                "Backend {name} not found"
            )))
        }
    }

    /// Create the default backend
    pub fn create_default_backend(&self) -> BackendResult<Box<dyn Backend>> {
        if let Some(default_name) = &self.default_backend {
            self.create_backend(default_name)
        } else {
            Err(TorshError::BackendError(
                "No default backend available".to_string(),
            ))
        }
    }

    /// Set the default backend
    pub fn set_default_backend(&mut self, name: &str) -> BackendResult<()> {
        if self.backends.contains_key(name) {
            self.default_backend = Some(name.to_string());
            Ok(())
        } else {
            Err(TorshError::BackendError(format!(
                "Backend {name} not found"
            )))
        }
    }

    /// Get plugin metadata
    pub fn get_plugin_metadata(&self, name: &str) -> Option<PluginMetadata> {
        self.backends.get(name).map(|plugin| plugin.metadata())
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend configuration trait for customizing backend behavior
pub trait BackendConfig: Send + Sync + Clone {
    /// Get the backend type this configuration is for
    fn backend_type(&self) -> BackendType;

    /// Validate the configuration
    fn validate(&self) -> BackendResult<()>;

    /// Get configuration as key-value pairs
    fn as_properties(&self) -> std::collections::HashMap<String, CapabilityValue>;

    /// Set configuration from key-value pairs
    fn from_properties(
        properties: &std::collections::HashMap<String, CapabilityValue>,
    ) -> BackendResult<Self>
    where
        Self: Sized;

    /// Merge with another configuration
    fn merge(&mut self, other: &Self) -> BackendResult<()>;

    /// Get default configuration
    fn default_config() -> Self
    where
        Self: Sized;
}

/// Backend builder trait for creating configured backends
pub trait BackendBuilder<T: BackendConfig>: Send + Sync {
    /// Create a new builder with default configuration
    fn new() -> Self;

    /// Set configuration
    fn with_config(self, config: T) -> Self;

    /// Build the backend
    fn build(self) -> BackendResult<Box<dyn Backend>>;

    /// Get the current configuration
    fn config(&self) -> &T;

    /// Get a mutable reference to the configuration
    fn config_mut(&mut self) -> &mut T;
}

/// Backend error handling trait for better error context
pub trait BackendErrorHandler: Send + Sync {
    /// Handle a backend error and provide context
    fn handle_error(&self, error: TorshError, context: &str) -> TorshError;

    /// Convert a backend-specific error to TorshError
    fn convert_error(&self, error: Box<dyn std::error::Error + Send + Sync>) -> TorshError;

    /// Get error recovery suggestions
    fn recovery_suggestions(&self, error: &TorshError) -> Vec<String>;

    /// Log error with appropriate level
    fn log_error(&self, error: &TorshError, context: &str);
}

/// Default error handler implementation
pub struct DefaultBackendErrorHandler {
    backend_name: String,
}

impl DefaultBackendErrorHandler {
    pub fn new(backend_name: String) -> Self {
        Self { backend_name }
    }
}

impl BackendErrorHandler for DefaultBackendErrorHandler {
    fn handle_error(&self, error: TorshError, context: &str) -> TorshError {
        // Add backend context to error
        match error {
            TorshError::BackendError(msg) => TorshError::BackendError(format!(
                "{}: {} (context: {})",
                self.backend_name, msg, context
            )),
            other => other,
        }
    }

    fn convert_error(&self, error: Box<dyn std::error::Error + Send + Sync>) -> TorshError {
        TorshError::BackendError(format!("{}: {}", self.backend_name, error))
    }

    fn recovery_suggestions(&self, error: &TorshError) -> Vec<String> {
        match error {
            TorshError::BackendError(msg) if msg.contains("not available") => {
                vec![
                    "Check if the backend is properly installed".to_string(),
                    "Verify system compatibility".to_string(),
                    "Try a different backend".to_string(),
                ]
            }
            TorshError::BackendError(msg) if msg.contains("memory") => {
                vec![
                    "Reduce batch size or tensor dimensions".to_string(),
                    "Enable memory optimization".to_string(),
                    "Check available memory".to_string(),
                ]
            }
            _ => vec!["Contact support with error details".to_string()],
        }
    }

    fn log_error(&self, error: &TorshError, context: &str) {
        eprintln!("[{}] Error in {}: {}", self.backend_name, context, error);
    }
}

/// Extended Backend trait with device factory methods
impl dyn Backend {
    /// Create the best available backend automatically
    pub fn auto() -> BackendResult<Box<dyn Backend>> {
        let (device_type, _device) = DeviceEnumerator::find_best_device()?;

        match device_type {
            #[cfg(feature = "cpu")]
            DeviceType::Cpu => Ok(Box::new(crate::cpu::CpuBackend::new()?)),
            #[cfg(cuda_available)]
            DeviceType::Cuda(device_id) => Ok(Box::new(crate::cuda::CudaBackend::new(
                crate::cuda::CudaBackendConfig {
                    device_id: device_id as usize,
                    ..Default::default()
                },
            )?)),
            #[cfg(all(feature = "cuda", not(cuda_available)))]
            DeviceType::Cuda(_) => Err(TorshError::BackendError(
                "CUDA backend not available on this platform".to_string(),
            )),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            DeviceType::Metal(_) => Ok(Box::new(crate::metal::MetalBackend::new()?)),
            #[cfg(feature = "webgpu")]
            DeviceType::Wgpu(_) => {
                Ok(Box::new(crate::webgpu::WebGpuBackend::with_default_config()))
            }
            #[allow(unreachable_patterns)]
            _ => Err(TorshError::BackendError(
                "No suitable backend found".to_string(),
            )),
        }
    }
}
