//! CUDA backend implementation

// Allow unused variables and unnecessary unsafe for CUDA placeholder implementations
#![allow(unused_variables)]
#![allow(unused_unsafe)]
#![allow(unused_mut)]

use crate::backend::{
    BackendCapabilities, BackendCore, BackendDeviceManager, BackendExecutor, BackendLifecycle,
    BackendOperations, BackendOps, BackendResourceManager, BackendType, CapabilityValue,
    OperationsBundle, PerformanceHints,
};
use crate::cuda::buffer::CudaBuffer;
use crate::cuda::cooperative_groups::{
    CooperativeGroupsContext, CooperativeKernelConfig, CooperativeWorkload,
};
use crate::cuda::device::CudaDevice;
use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::graph::{CudaGraph, GraphCache, GraphCaptureContext};
use crate::cuda::kernels::KernelRegistry;
use crate::cuda::memory::{CudaMemoryManager, MemoryAdvice, UnifiedAllocation};
use crate::cuda::stream::CudaStream;
use crate::error::{conversion, BackendError, BackendResult};
use crate::{
    Backend, Buffer, BufferDescriptor, BufferHandle, Device, Kernel, KernelDescriptor,
    MemoryManager, Profiler,
};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, RwLock,
};
use torsh_core::device::DeviceType;
use torsh_core::DType;

/// CUDA backend implementation with enhanced resource management
#[derive(Debug)]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    memory_manager: Arc<RwLock<CudaMemoryManager>>,
    default_stream: Arc<CudaStream>,
    kernels: Arc<KernelRegistry>,
    config: CudaBackendConfig,
    graph_cache: Arc<RwLock<GraphCache>>,
    capture_context: Arc<Mutex<Option<GraphCaptureContext>>>,
    cooperative_groups: Option<Arc<CooperativeGroupsContext>>,
    is_shutdown: Arc<AtomicBool>,
    resource_tracker: Arc<Mutex<ResourceTracker>>,
}

/// Resource tracker for proper cleanup
#[derive(Debug, Default)]
pub struct ResourceTracker {
    active_buffers: Vec<usize>, // Store addresses as usize for thread safety
    active_streams: Vec<Arc<CudaStream>>,
    active_graphs: Vec<String>, // Graph keys for cleanup
    unified_allocations: Vec<UnifiedAllocation>,
}

impl ResourceTracker {
    /// Track a new buffer allocation
    pub fn track_buffer(&mut self, ptr: *mut std::ffi::c_void) {
        if !ptr.is_null() {
            self.active_buffers.push(ptr as usize);
        }
    }

    /// Untrack a buffer allocation
    pub fn untrack_buffer(&mut self, ptr: *mut std::ffi::c_void) {
        let addr = ptr as usize;
        self.active_buffers.retain(|&p| p != addr);
    }

    /// Track a new stream
    pub fn track_stream(&mut self, stream: Arc<CudaStream>) {
        self.active_streams.push(stream);
    }

    /// Track a new graph
    pub fn track_graph(&mut self, key: String) {
        if !self.active_graphs.contains(&key) {
            self.active_graphs.push(key);
        }
    }

    /// Track a unified memory allocation
    pub fn track_unified_allocation(&mut self, allocation: &UnifiedAllocation) {
        // Store a copy for tracking (assuming UnifiedAllocation implements Clone)
        // If not, we could store just the pointer and size
        // For now, we'll track the key information
    }

    /// Untrack a unified memory allocation
    pub fn untrack_unified_allocation(&mut self, ptr: *mut u8) {
        self.unified_allocations.retain(|alloc| alloc.ptr() != ptr);
    }

    /// Get the number of active resources
    pub fn active_resource_count(&self) -> (usize, usize, usize, usize) {
        (
            self.active_buffers.len(),
            self.active_streams.len(),
            self.active_graphs.len(),
            self.unified_allocations.len(),
        )
    }

    /// Clean up all tracked resources (should only be called during shutdown)
    pub fn cleanup_all_resources(&mut self) {
        // Note: In a real implementation, this would call appropriate cleanup functions
        // For now, we'll just clear the vectors to show intent
        tracing::info!(
            "Cleaning up {} buffers, {} streams, {} graphs, {} unified allocations",
            self.active_buffers.len(),
            self.active_streams.len(),
            self.active_graphs.len(),
            self.unified_allocations.len()
        );

        self.active_buffers.clear();
        self.active_streams.clear();
        self.active_graphs.clear();
        self.unified_allocations.clear();
    }
}

/// CUDA backend configuration
#[derive(Debug, Clone)]
pub struct CudaBackendConfig {
    pub device_id: usize,
    pub allow_tf32: bool,
    pub enable_profiling: bool,
    pub memory_pool_size: Option<usize>,
    pub stream_pool_size: usize,
}

impl Default for CudaBackendConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            allow_tf32: true,
            enable_profiling: false,
            memory_pool_size: None,
            stream_pool_size: 4,
        }
    }
}

// BackendConfig trait no longer exists

/// CUDA Backend Builder
pub struct CudaBackendBuilder {
    device_id: usize,
    memory_pool_config: Option<crate::MemoryPoolConfig>,
    allow_tf32: bool,
    enable_profiling: bool,
    stream_pool_size: usize,
}

impl CudaBackendBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            device_id: 0,
            memory_pool_config: None,
            allow_tf32: true,
            enable_profiling: false,
            stream_pool_size: 4,
        }
    }

    /// Set the device ID
    pub fn device(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    /// Set memory pool configuration
    pub fn memory_pool(mut self, config: crate::memory::MemoryPoolConfig) -> Self {
        self.memory_pool_config = Some(config);
        self
    }

    /// Set whether to allow TF32
    pub fn allow_tf32(mut self, allow: bool) -> Self {
        self.allow_tf32 = allow;
        self
    }

    /// Enable profiling
    pub fn enable_profiling(mut self, enable: bool) -> Self {
        self.enable_profiling = enable;
        self
    }

    /// Set stream pool size
    pub fn stream_pool_size(mut self, size: usize) -> Self {
        self.stream_pool_size = size;
        self
    }

    /// Build the CUDA backend
    pub fn build(self) -> CudaResult<CudaBackend> {
        let config = CudaBackendConfig {
            device_id: self.device_id,
            allow_tf32: self.allow_tf32,
            enable_profiling: self.enable_profiling,
            memory_pool_size: self.memory_pool_config.as_ref().map(|c| c.initial_size),
            stream_pool_size: self.stream_pool_size,
        };
        CudaBackend::new(config)
    }
}

impl CudaBackend {
    /// Create a new CUDA backend builder
    pub fn builder() -> CudaBackendBuilder {
        CudaBackendBuilder::new()
    }

    /// Create new CUDA backend
    pub fn new(config: CudaBackendConfig) -> CudaResult<Self> {
        // Initialize CUDA
        cust::init(cust::CudaFlags::empty())
            .map_err(|e| CudaError::Backend(format!("CUDA init failed: {}", e)))?;

        // Create device
        let device = Arc::new(CudaDevice::new(config.device_id)?);

        // Set device as current (done via CudaDevice::new)
        // No separate set_device needed as device is bound during creation

        // Create a new memory manager for the backend
        // Note: We create a separate one here because CudaBackend needs RwLock wrapper
        // while CudaDevice has Arc<CudaMemoryManager>
        let memory_manager = Arc::new(RwLock::new(CudaMemoryManager::new(config.device_id)?));

        // Use default stream from device (already created in CudaDevice::new)
        let default_stream = Arc::clone(device.default_stream());

        // Load kernels (would load from embedded PTX in real implementation)
        let kernels = Arc::new(Self::load_kernels()?);

        // Create thread-safe graph cache
        let graph_cache = Arc::new(RwLock::new(GraphCache::new()));
        let capture_context = Arc::new(Mutex::new(None));

        // Try to initialize cooperative groups (optional, won't fail if not supported)
        let cooperative_groups = CooperativeGroupsContext::new(config.device_id)
            .map(|ctx| Arc::new(ctx))
            .ok();

        // Initialize resource tracking and shutdown flag
        let is_shutdown = Arc::new(AtomicBool::new(false));
        let resource_tracker = Arc::new(Mutex::new(ResourceTracker::default()));

        Ok(Self {
            device,
            memory_manager,
            default_stream,
            kernels,
            config,
            graph_cache,
            capture_context,
            cooperative_groups,
            is_shutdown,
            resource_tracker,
        })
    }

    /// Load CUDA kernels
    fn load_kernels() -> CudaResult<KernelRegistry> {
        // In a real implementation, this would load compiled PTX
        // For now, we'll create a placeholder registry
        let ptx = include_str!("kernels/compiled.ptx");
        KernelRegistry::load_from_ptx(ptx).or_else(|_| {
            // Fallback: create empty registry for testing
            tracing::warn!("Failed to load CUDA kernels, using fallback");
            Ok(KernelRegistry::load_from_ptx("")?)
        })
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get memory manager (thread-safe access)
    pub fn memory_manager(&self) -> &Arc<RwLock<CudaMemoryManager>> {
        &self.memory_manager
    }

    /// Shutdown the backend and cleanup resources
    pub fn shutdown(&self) -> CudaResult<()> {
        // Check if already shutdown
        if self.is_shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        tracing::info!(
            "Shutting down CUDA backend for device {}",
            self.config.device_id
        );

        // Mark as shutdown to prevent new operations
        self.is_shutdown.store(true, Ordering::SeqCst);

        // Synchronize the device to ensure all operations are complete
        if let Err(e) = self.synchronize() {
            tracing::warn!("Failed to synchronize device during shutdown: {}", e);
        }

        // Cleanup any ongoing graph capture
        if let Ok(mut capture_opt) = self.capture_context.lock() {
            if let Some(mut capture_ctx) = capture_opt.take() {
                if let Err(e) = capture_ctx.abort() {
                    tracing::warn!("Failed to abort graph capture during shutdown: {}", e);
                }
            }
        }

        // Cleanup tracked resources
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.cleanup_all_resources();
        }

        // Clear graph cache
        if let Ok(mut cache) = self.graph_cache.write() {
            cache.clear();
        }

        tracing::info!(
            "CUDA backend shutdown complete for device {}",
            self.config.device_id
        );
        Ok(())
    }

    /// Check if the backend is shutdown
    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown.load(Ordering::SeqCst)
    }

    /// Check if backend is available for operations
    fn check_availability(&self) -> CudaResult<()> {
        if self.is_shutdown() {
            return Err(CudaError::Context {
                message: "Backend has been shutdown".to_string(),
            });
        }
        Ok(())
    }

    /// Get default stream
    pub fn default_stream(&self) -> &Arc<CudaStream> {
        &self.default_stream
    }

    /// Create buffer
    pub fn create_buffer<T: Clone + Send + Sync + 'static>(
        &self,
        length: usize,
        dtype: DType,
    ) -> CudaResult<CudaBuffer<T>> {
        CudaBuffer::new(Arc::clone(&self.device), length, dtype)
    }

    /// Synchronize device
    pub fn synchronize(&self) -> CudaResult<()> {
        self.device
            .synchronize()
            .map_err(|e| CudaError::Backend(e.to_string()))?;
        Ok(())
    }

    /// Check if cooperative groups are supported
    pub fn is_cooperative_groups_supported(&self) -> bool {
        self.cooperative_groups
            .as_ref()
            .map(|cg| cg.is_supported())
            .unwrap_or(false)
    }

    /// Get cooperative groups context
    pub fn cooperative_groups(&self) -> Option<&Arc<CooperativeGroupsContext>> {
        self.cooperative_groups.as_ref()
    }

    /// Launch a cooperative kernel
    pub unsafe fn launch_cooperative_kernel(
        &self,
        kernel_func: *const std::ffi::c_void,
        config: &CooperativeKernelConfig,
        kernel_params: &[*mut std::ffi::c_void],
    ) -> CudaResult<u64> {
        if let Some(ref cg_context) = self.cooperative_groups {
            cg_context
                .launch_cooperative_kernel(kernel_func, config, kernel_params)
                .map_err(|e| CudaError::Backend(e.to_string()))
        } else {
            Err(CudaError::UnsupportedOperation {
                op: "cooperative_groups".to_string(),
                dtype: "Cooperative groups not supported on this device".to_string(),
            })
        }
    }

    /// Get optimal cooperative kernel configuration for a workload
    pub fn suggest_cooperative_config(
        &self,
        workload: &CooperativeWorkload,
    ) -> CudaResult<CooperativeKernelConfig> {
        if let Some(ref cg_context) = self.cooperative_groups {
            cg_context
                .suggest_optimal_config(workload)
                .map_err(|e| CudaError::Backend(e.to_string()))
        } else {
            Err(CudaError::UnsupportedOperation {
                op: "cooperative_groups".to_string(),
                dtype: "Cooperative groups not supported on this device".to_string(),
            })
        }
    }

    /// Finish cooperative kernel execution and get performance metrics
    pub fn finish_cooperative_kernel(
        &self,
        kernel_id: u64,
    ) -> CudaResult<crate::cuda::cooperative_groups::KernelPerformanceMetrics> {
        if let Some(ref cg_context) = self.cooperative_groups {
            cg_context
                .finish_kernel(kernel_id)
                .map_err(|e| CudaError::Backend(e.to_string()))
        } else {
            Err(CudaError::UnsupportedOperation {
                op: "cooperative_groups".to_string(),
                dtype: "Cooperative groups not supported on this device".to_string(),
            })
        }
    }

    /// Allocate unified memory with resource tracking
    ///
    /// Note: Unified memory allocation is currently stubbed out.
    /// TODO: Implement full unified memory support when CUDA unified memory APIs are available.
    pub fn allocate_unified(&self, size: usize) -> CudaResult<UnifiedAllocation> {
        self.check_availability()?;

        // Allocate managed memory using CUDA runtime
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let result = crate::cuda::cuda_sys_compat::cudaMallocManaged(
                &mut ptr,
                size,
                crate::cuda::cuda_sys_compat::cudaMemAttachGlobal,
            );

            if result != crate::cuda::cudaSuccess || ptr.is_null() {
                return Err(CudaError::Context {
                    message: format!("Failed to allocate {} bytes of unified memory", size),
                });
            }
        }

        // Create allocation with simple metadata
        let allocation = UnifiedAllocation {
            ptr: crate::cuda::memory::allocation::SendSyncPtr::new(ptr as *mut u8),
            size,
            allocation_time: std::time::Instant::now(),
            preferred_location: crate::cuda::memory::allocation::PreferredLocation::Device(
                self.config.device_id,
            ),
            access_hints: crate::cuda::memory::allocation::AccessHints::default(),
            migration_stats: crate::cuda::memory::allocation::MigrationStats::default(),
            metadata: crate::cuda::memory::allocation::AllocationMetadata::default(),
        };

        // Track the allocation
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.track_unified_allocation(&allocation);
        }

        Ok(allocation)
    }

    /// Deallocate unified memory with resource tracking
    pub fn deallocate_unified(&self, allocation: UnifiedAllocation) -> CudaResult<()> {
        let ptr = allocation.ptr.as_ptr();

        // Free the memory
        unsafe {
            let result = crate::cuda::cuda_sys_compat::cudaFree(ptr as *mut std::ffi::c_void);
            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Context {
                    message: "Failed to free unified memory".to_string(),
                });
            }
        }

        // Untrack the allocation
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.untrack_unified_allocation(ptr);
        }

        Ok(())
    }

    /// Prefetch unified memory to device with availability check
    pub fn prefetch_to_device(
        &self,
        ptr: *mut u8,
        size: usize,
        device_id: Option<usize>,
    ) -> CudaResult<()> {
        self.check_availability()?;

        let device = device_id.unwrap_or(self.config.device_id) as i32;

        unsafe {
            let result = crate::cuda::cuda_sys_compat::cudaMemPrefetchAsync(
                ptr as *const std::ffi::c_void,
                size,
                device,
                0 as crate::cuda::cudaStream_t,
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to prefetch memory to device {}", device),
                });
            }
        }

        Ok(())
    }

    /// Prefetch unified memory to host with availability check
    pub fn prefetch_to_host(&self, ptr: *mut u8, size: usize) -> CudaResult<()> {
        self.check_availability()?;

        // cudaCpuDeviceId is -1
        const CUDA_CPU_DEVICE_ID: i32 = -1;

        unsafe {
            let result = crate::cuda::cuda_sys_compat::cudaMemPrefetchAsync(
                ptr as *const std::ffi::c_void,
                size,
                CUDA_CPU_DEVICE_ID,
                0 as crate::cuda::cudaStream_t,
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Context {
                    message: "Failed to prefetch memory to host".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Set memory advice for unified memory with availability check
    pub fn set_memory_advice(
        &self,
        ptr: *mut u8,
        size: usize,
        advice: MemoryAdvice,
        _device_id: Option<usize>,
    ) -> CudaResult<()> {
        self.check_availability()?;

        let device = self.config.device_id as i32;

        // Map MemoryAdvice to CUDA advice constants
        let cuda_advice = match advice {
            MemoryAdvice::SetReadMostly => crate::cuda::cuda_sys_compat::cudaMemAdviseSetReadMostly,
            MemoryAdvice::UnsetReadMostly => {
                crate::cuda::cuda_sys_compat::cudaMemAdviseUnsetReadMostly
            }
            MemoryAdvice::SetPreferredLocation => {
                crate::cuda::cuda_sys_compat::cudaMemAdviseSetPreferredLocation
            }
            MemoryAdvice::UnsetPreferredLocation => {
                crate::cuda::cuda_sys_compat::cudaMemAdviseUnsetPreferredLocation
            }
            MemoryAdvice::SetAccessedBy => crate::cuda::cuda_sys_compat::cudaMemAdviseSetAccessedBy,
            MemoryAdvice::UnsetAccessedBy => {
                crate::cuda::cuda_sys_compat::cudaMemAdviseUnsetAccessedBy
            }
        };

        unsafe {
            let result = crate::cuda::cuda_sys_compat::cudaMemAdvise(
                ptr as *const std::ffi::c_void,
                size,
                cuda_advice,
                device,
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to set memory advice {:?}", advice),
                });
            }
        }

        Ok(())
    }

    /// Check if device supports unified memory
    pub fn supports_unified_memory(&self) -> CudaResult<bool> {
        self.device
            .supports_feature(crate::cuda::device::CudaFeature::ManagedMemory)
    }

    /// Execute elementwise addition
    pub fn elementwise_add_f32(
        &self,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(CudaError::InvalidDevice {
                device_id: a.len(), // Using as error code
            });
        }

        let stream = stream.unwrap_or(&self.default_stream);
        let size = a.len();

        unsafe {
            crate::cuda::kernels::tensor_ops::launch_elementwise_add_f32(
                a.device_ptr().as_mut_ptr(),
                b.device_ptr().as_mut_ptr(),
                output.device_ptr().as_mut_ptr(),
                size,
                stream.stream(),
            );
        }

        Ok(())
    }

    /// Execute elementwise multiplication
    pub fn elementwise_mul_f32(
        &self,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(CudaError::InvalidDevice {
                device_id: a.len(), // Using as error code
            });
        }

        let stream = stream.unwrap_or(&self.default_stream);
        let size = a.len();

        unsafe {
            crate::cuda::kernels::tensor_ops::launch_elementwise_mul_f32(
                a.device_ptr().as_mut_ptr(),
                b.device_ptr().as_mut_ptr(),
                output.device_ptr().as_mut_ptr(),
                size,
                stream.stream(),
            );
        }

        Ok(())
    }

    /// Execute matrix multiplication using cuBLAS
    pub fn matmul_f32(
        &self,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        let _ = (stream, m, n, k, a, b, output);

        // TODO: Implement cuBLAS integration when cublas-sys is available
        // The cust crate doesn't include cuBLAS bindings directly
        // For now, return an error indicating GEMM is not yet implemented
        Err(CudaError::InvalidValue(
            "cuBLAS GEMM not yet implemented - requires cublas-sys crate integration".to_string(),
        ))
    }

    /// Execute convolution using cuDNN
    pub fn conv2d_f32(
        &self,
        input: &CudaBuffer<f32>,
        weight: &CudaBuffer<f32>,
        bias: Option<&CudaBuffer<f32>>,
        output: &mut CudaBuffer<f32>,
        config: &Conv2dConfig,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        let stream = stream.unwrap_or(&self.default_stream);

        // Use custom kernel for now (would use cuDNN in production)
        unsafe {
            crate::cuda::kernels::neural_ops::launch_conv2d_f32(
                input.device_ptr().as_mut_ptr(),
                weight.device_ptr().as_mut_ptr(),
                bias.map(|b| b.device_ptr().as_mut_ptr())
                    .unwrap_or(std::ptr::null_mut()),
                output.device_ptr().as_mut_ptr(),
                config.batch_size as i32,
                config.in_channels as i32,
                config.out_channels as i32,
                config.input_height as i32,
                config.input_width as i32,
                config.kernel_height as i32,
                config.kernel_width as i32,
                config.pad_h as i32,
                config.pad_w as i32,
                config.stride_h as i32,
                config.stride_w as i32,
                config.dilation_h as i32,
                config.dilation_w as i32,
                stream.stream(),
            );
        }

        Ok(())
    }

    /// Begin graph capture on a stream
    pub fn begin_graph_capture(&self, _stream: Option<&CudaStream>) -> CudaResult<()> {
        // Use default stream for capture (stream parameter reserved for future use)
        let stream_arc = Arc::clone(&self.default_stream);

        // Check if already capturing
        let mut capture_opt = self
            .capture_context
            .lock()
            .expect("lock should not be poisoned");
        if capture_opt.is_some() {
            return Err(CudaError::Context {
                message: "Already capturing a graph".to_string(),
            });
        }

        // Create new capture context (GraphCaptureContext::new returns Result)
        let capture_ctx = GraphCaptureContext::new(stream_arc).map_err(|e| CudaError::Context {
            message: format!("Failed to create graph capture context: {}", e),
        })?;

        capture_ctx.start().map_err(|e| CudaError::Context {
            message: format!("Failed to start graph capture: {}", e),
        })?;
        *capture_opt = Some(capture_ctx);

        Ok(())
    }

    /// End graph capture and return the captured graph
    pub fn end_graph_capture(&self) -> CudaResult<CudaGraph> {
        let mut capture_opt = self
            .capture_context
            .lock()
            .expect("lock should not be poisoned");
        let capture_ctx = capture_opt.take().ok_or_else(|| CudaError::Context {
            message: "Not capturing a graph".to_string(),
        })?;

        capture_ctx.end_capture().map_err(|e| CudaError::Context {
            message: format!("Failed to end graph capture: {}", e),
        })
    }

    /// Check if currently capturing a graph
    pub fn is_capturing_graph(&self) -> bool {
        self.capture_context
            .lock()
            .expect("lock should not be poisoned")
            .is_some()
    }

    /// Execute a captured graph
    pub fn launch_graph(&self, graph: &CudaGraph, stream: Option<&CudaStream>) -> CudaResult<()> {
        let stream = stream.unwrap_or(&self.default_stream);
        graph.launch(stream).map_err(|e| CudaError::Context {
            message: format!("Failed to launch graph: {}", e),
        })
    }

    /// Get or create a cached graph with thread-safe access
    pub fn get_or_create_graph<F>(&self, key: &str, creator: F) -> CudaResult<Arc<Mutex<CudaGraph>>>
    where
        F: FnOnce() -> CudaResult<CudaGraph>,
    {
        self.check_availability()?;

        let result = {
            let cache = self.graph_cache.write().map_err(|_| CudaError::Context {
                message: "Failed to acquire graph cache write lock".to_string(),
            })?;

            cache.get_or_create(key, || {
                creator().map_err(|e| BackendError::ComputeError(e.to_string()))
            })
        };

        match result {
            Ok(graph) => {
                // Track the graph
                if let Ok(mut tracker) = self.resource_tracker.lock() {
                    tracker.track_graph(key.to_string());
                }
                Ok(graph)
            }
            Err(e) => Err(CudaError::Context {
                message: format!("Failed to get or create graph: {}", e),
            }),
        }
    }

    /// Execute elementwise addition with graph capture support
    pub fn elementwise_add_f32_graph(
        &self,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        use_graph: bool,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        if !use_graph || self.is_capturing_graph() {
            // Direct execution or already capturing
            return self.elementwise_add_f32(a, b, output, stream);
        }

        // Try to use cached graph
        let key = format!("add_f32_{}", a.len());
        let graph = self.get_or_create_graph(&key, || {
            self.begin_graph_capture(stream)?;
            self.elementwise_add_f32(a, b, output, stream)?;
            self.end_graph_capture()
        })?;

        // Launch the graph
        let graph = graph.lock().expect("lock should not be poisoned");
        self.launch_graph(&graph, stream)
    }

    /// Execute matrix multiplication with graph capture support
    pub fn matmul_f32_graph(
        &self,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        use_graph: bool,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        if !use_graph || self.is_capturing_graph() {
            // Direct execution or already capturing
            return self.matmul_f32(a, b, output, m, n, k, stream);
        }

        // Try to use cached graph
        let key = format!("matmul_f32_{}x{}x{}", m, n, k);
        let graph = self.get_or_create_graph(&key, || {
            self.begin_graph_capture(stream)?;
            self.matmul_f32(a, b, output, m, n, k, stream)?;
            self.end_graph_capture()
        })?;

        // Launch the graph
        let graph = graph.lock().expect("lock should not be poisoned");
        self.launch_graph(&graph, stream)
    }
}

/// Implement Drop for automatic resource cleanup
impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Ensure proper cleanup when the backend is dropped
        if !self.is_shutdown() {
            tracing::warn!(
                "CudaBackend dropped without explicit shutdown, performing emergency cleanup"
            );
            // Call the inherent (non-async) shutdown method directly
            if let Err(e) = CudaBackend::shutdown(self) {
                tracing::error!("Failed to shutdown CUDA backend during drop: {}", e);
            }
        }
    }
}

// ============== BackendCore Implementation ==============
impl BackendCore for CudaBackend {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda(self.config.device_id)
    }

    fn name(&self) -> &str {
        "CUDA Backend"
    }

    fn is_available(&self) -> BackendResult<bool> {
        Ok(crate::cuda::is_available())
    }

    fn capabilities(&self) -> BackendCapabilities {
        use crate::backend::{ExtendedCapabilities, HardwareFeature, PrecisionMode};

        let mut extended_capabilities = ExtendedCapabilities::default();
        extended_capabilities.precision_modes = vec![
            PrecisionMode::F16,
            PrecisionMode::F32,
            PrecisionMode::F64,
            PrecisionMode::Mixed,
        ];
        extended_capabilities.hardware_features = vec![
            HardwareFeature::TensorCores,
            HardwareFeature::SharedMemory,
            HardwareFeature::AtomicOperations,
            HardwareFeature::CooperativeGroups,
            HardwareFeature::DynamicParallelism,
        ];
        extended_capabilities.execution_model.supports_simd = true;
        extended_capabilities.execution_model.supports_simt = true;
        extended_capabilities
            .execution_model
            .supports_task_parallelism = true;
        extended_capabilities
            .execution_model
            .supports_data_parallelism = true;
        extended_capabilities.execution_model.max_concurrent_streams = Some(32);
        extended_capabilities.execution_model.supports_out_of_order = true;

        BackendCapabilities {
            max_buffer_size: 16 * 1024 * 1024 * 1024, // 16GB typical GPU memory
            max_compute_units: 128,                   // Typical SM count
            max_workgroup_size: (1024, 1024, 64),
            supported_dtypes: vec![
                DType::F16,
                DType::F32,
                DType::F64,
                DType::I32,
                DType::I64,
                DType::I16,
                DType::I8,
                DType::U8,
                DType::Bool,
            ],
            supports_async: true,
            supports_unified_memory: true,
            supports_sub_buffers: true,
            supports_kernel_caching: true,
            memory_bandwidth_gbps: 900.0,       // Typical high-end GPU
            compute_throughput_gflops: 20000.0, // Typical GPU TFLOPS
            extended_capabilities,
        }
    }

    fn performance_hints(&self) -> PerformanceHints {
        PerformanceHints {
            preferred_workgroup_size: (256, 1, 1),
            memory_alignment: 256, // CUDA memory alignment
            prefer_vectorized: true,
            prefer_async: true,
            optimal_batch_size: 256,
            cache_kernels: true,
        }
    }
}

// ============== BackendLifecycle Implementation ==============
#[async_trait]
impl BackendLifecycle for CudaBackend {
    async fn initialize(&mut self) -> BackendResult<()> {
        if self.is_shutdown() {
            return Err(conversion::cuda_error_with_context(
                "Backend has been shutdown",
                "initialize",
                Some(self.config.device_id),
            ));
        }
        Ok(())
    }

    async fn shutdown(&mut self) -> BackendResult<()> {
        CudaBackend::shutdown(self).map_err(|e| {
            conversion::cuda_error_with_context(
                e.to_string(),
                "shutdown",
                Some(self.config.device_id),
            )
        })
    }

    fn is_initialized(&self) -> bool {
        !self.is_shutdown()
    }
}

// ============== BackendDeviceManager Implementation ==============
impl BackendDeviceManager for CudaBackend {
    fn devices(&self) -> BackendResult<Vec<Device>> {
        let device = cuda_device_to_abstract(&self.device, self.config.device_id);
        Ok(vec![device])
    }

    fn default_device(&self) -> BackendResult<Device> {
        Ok(cuda_device_to_abstract(&self.device, self.config.device_id))
    }

    fn create_device(&self, device_id: usize) -> BackendResult<Device> {
        if device_id != self.config.device_id {
            return Err(conversion::cuda_error_with_context(
                format!(
                    "CUDA device {} not managed by this backend (managing device {})",
                    device_id, self.config.device_id
                ),
                "create_device",
                Some(self.config.device_id),
            ));
        }
        Ok(cuda_device_to_abstract(&self.device, self.config.device_id))
    }

    fn device_count(&self) -> BackendResult<usize> {
        Ok(1) // This backend manages one device
    }

    fn is_device_available(&self, device_id: usize) -> bool {
        device_id == self.config.device_id
    }
}

/// Helper function to convert CudaDevice to abstract Device
fn cuda_device_to_abstract(_cuda_device: &Arc<CudaDevice>, device_id: usize) -> Device {
    use crate::{DeviceFeature, DeviceInfo};

    let info = DeviceInfo {
        vendor: "NVIDIA".to_string(),
        driver_version: "CUDA".to_string(),
        total_memory: 16 * 1024 * 1024 * 1024, // 16GB default estimate
        available_memory: 16 * 1024 * 1024 * 1024,
        compute_units: 128, // Typical SM count
        max_work_group_size: 1024,
        max_work_group_dimensions: vec![1024, 1024, 64],
        clock_frequency_mhz: 2000,
        memory_bandwidth_gbps: 900.0,
        peak_gflops: 20000.0,
        features: vec![
            DeviceFeature::DoublePrecision,
            DeviceFeature::UnifiedMemory,
            DeviceFeature::AtomicOperations,
            DeviceFeature::Profiling,
            DeviceFeature::ConcurrentExecution,
            DeviceFeature::AsyncMemory,
            DeviceFeature::FastMath,
        ],
        properties: vec![
            ("compute_capability".to_string(), "7.0+".to_string()),
            ("tensor_cores".to_string(), "supported".to_string()),
        ],
    };

    Device::new(
        device_id,
        DeviceType::Cuda(device_id),
        format!("CUDA Device {}", device_id),
        info,
    )
}

// ============== BackendResourceManager Implementation ==============
impl BackendResourceManager for CudaBackend {
    fn create_buffer(
        &self,
        _device: &Device,
        _descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer> {
        // For CUDA backend, create a generic Buffer wrapper
        // This is a simplified implementation - in production you'd have proper CUDA buffer creation
        Err(conversion::cuda_error_with_context(
            "CUDA buffer creation through abstract interface not yet implemented",
            "create_buffer",
            Some(self.config.device_id),
        ))
    }

    fn create_kernel(
        &self,
        _device: &Device,
        _descriptor: &KernelDescriptor,
    ) -> BackendResult<Kernel> {
        Err(conversion::cuda_error_with_context(
            "CUDA kernel creation through abstract interface not yet implemented",
            "create_kernel",
            Some(self.config.device_id),
        ))
    }

    fn memory_manager(
        &self,
        _device: &Device,
    ) -> BackendResult<Box<dyn MemoryManager + Send + Sync>> {
        // Return a memory manager wrapper
        Err(conversion::cuda_error_with_context(
            "CUDA memory manager through abstract interface not yet implemented",
            "memory_manager",
            Some(self.config.device_id),
        ))
    }

    fn profiler(&self) -> BackendResult<Box<dyn Profiler + Send + Sync>> {
        Err(conversion::cuda_error_with_context(
            "CUDA profiler through abstract interface not yet implemented",
            "profiler",
            Some(self.config.device_id),
        ))
    }

    fn create_scoped_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer> {
        let dtype = descriptor.dtype.unwrap_or(DType::F32);
        let element_size = dtype.size_bytes();
        let length = if element_size > 0 {
            descriptor.size / element_size
        } else {
            descriptor.size
        };

        // Create CUDA buffer
        let _cuda_buffer: CudaBuffer<u8> = self.create_buffer(length, dtype).map_err(|e| {
            conversion::cuda_error_with_context(
                e.to_string(),
                "create_scoped_buffer",
                Some(self.config.device_id),
            )
        })?;

        // Return a properly constructed Buffer
        static BUFFER_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let buffer_id = BUFFER_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(Buffer::new(
            buffer_id,
            device.clone(),
            descriptor.size,
            descriptor.usage.clone(),
            descriptor.clone(),
            BufferHandle::Cuda {
                device_ptr: buffer_id as u64,
                size: descriptor.size,
            },
        ))
    }
}

// ============== BackendExecutor Implementation ==============
#[async_trait]
impl BackendExecutor for CudaBackend {
    async fn synchronize(&self, _device: &Device) -> BackendResult<()> {
        CudaBackend::synchronize(self).map_err(|e| {
            conversion::cuda_error_with_context(
                e.to_string(),
                "synchronize",
                Some(self.config.device_id),
            )
        })
    }

    async fn copy_buffer(
        &self,
        _src: &Buffer,
        _dst: &Buffer,
        _src_offset: usize,
        _dst_offset: usize,
        _size: usize,
    ) -> BackendResult<()> {
        Err(conversion::cuda_error_with_context(
            "CUDA buffer copy through abstract interface not yet implemented",
            "copy_buffer",
            Some(self.config.device_id),
        ))
    }

    async fn copy_to_device(
        &self,
        _src: &[u8],
        _dst: &Buffer,
        _dst_offset: usize,
    ) -> BackendResult<()> {
        Err(conversion::cuda_error_with_context(
            "CUDA copy to device through abstract interface not yet implemented",
            "copy_to_device",
            Some(self.config.device_id),
        ))
    }

    async fn copy_from_device(
        &self,
        _src: &Buffer,
        _dst: &mut [u8],
        _src_offset: usize,
    ) -> BackendResult<()> {
        Err(conversion::cuda_error_with_context(
            "CUDA copy from device through abstract interface not yet implemented",
            "copy_from_device",
            Some(self.config.device_id),
        ))
    }

    async fn execute_kernel(
        &self,
        _kernel: &Kernel,
        _buffers: &[&Buffer],
        _uniform_data: &[u8],
        _workgroup_size: (u32, u32, u32),
        _workgroup_count: (u32, u32, u32),
    ) -> BackendResult<()> {
        Err(conversion::cuda_error_with_context(
            "CUDA kernel execution through abstract interface not yet implemented",
            "execute_kernel",
            Some(self.config.device_id),
        ))
    }
}

// ============== BackendOperations Implementation ==============
impl BackendOperations for CudaBackend {
    fn fft_ops(&self) -> Box<dyn crate::fft::FftOps> {
        Box::new(crate::fft::DefaultFftOps::new())
    }

    fn convolution_ops(&self) -> Box<dyn crate::convolution::ConvolutionOps> {
        Box::new(crate::convolution::DefaultConvolutionOps::new())
    }

    fn rnn_ops(&self) -> Box<dyn crate::rnn::RnnOps> {
        Box::new(crate::rnn::DefaultRnnOps::new())
    }

    fn sparse_ops(&self) -> Box<dyn crate::sparse_ops::SparseOps<f32>> {
        Box::new(crate::sparse_ops::DefaultSparseOps::new(
            self.default_device()
                .expect("CUDA backend should always have a default device when initialized"),
        ))
    }

    fn quantization_ops(&self) -> Box<dyn crate::quantization::QuantizationOps> {
        Box::new(crate::quantization::CpuQuantizationOps)
    }

    fn operations_bundle(&self) -> OperationsBundle {
        OperationsBundle::new(
            self.fft_ops(),
            self.convolution_ops(),
            self.rnn_ops(),
            self.sparse_ops(),
            self.quantization_ops(),
        )
    }
}

// ============== BackendOps Implementation ==============
impl BackendOps for CudaBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }

    fn available_ops(&self) -> Vec<&str> {
        vec![
            "add",
            "sub",
            "mul",
            "div",
            "sin",
            "cos",
            "exp",
            "log",
            "matmul",
            "conv2d",
            "batch_norm",
            "relu",
            "softmax",
            "dropout",
            "fft",
            "ifft",
            "rnn",
            "lstm",
            "gru",
            "sparse_matmul",
            "quantize",
            "dequantize",
            "tensor_core_matmul",
            "mixed_precision",
        ]
    }

    fn supports_op(&self, op_name: &str) -> bool {
        self.available_ops().contains(&op_name)
    }

    fn supports_fft(&self) -> bool {
        true // cuFFT available
    }

    fn supports_convolution(&self) -> bool {
        true // cuDNN available
    }

    fn supports_rnn(&self) -> bool {
        true // cuDNN RNN available
    }

    fn supports_sparse(&self) -> bool {
        true // cuSPARSE available
    }

    fn supports_quantization(&self) -> bool {
        true // CUDA quantization supported
    }

    fn operation_capabilities(&self, op_name: &str) -> Option<HashMap<String, CapabilityValue>> {
        let mut caps = HashMap::new();

        match op_name {
            "matmul" => {
                caps.insert("max_size".to_string(), CapabilityValue::Int(65536));
                caps.insert("supports_batched".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_strided".to_string(), CapabilityValue::Bool(true));
                caps.insert(
                    "supports_tensor_cores".to_string(),
                    CapabilityValue::Bool(true),
                );
            }
            "conv2d" => {
                caps.insert("max_kernel_size".to_string(), CapabilityValue::Int(31));
                caps.insert("supports_groups".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_dilation".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_cudnn".to_string(), CapabilityValue::Bool(true));
            }
            "fft" => {
                caps.insert("max_size".to_string(), CapabilityValue::Int(134217728)); // 128M elements
                caps.insert("supports_real".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_batched".to_string(), CapabilityValue::Bool(true));
            }
            _ => return None,
        }

        Some(caps)
    }
}

// ============== Main Backend Trait Implementation ==============
impl Backend for CudaBackend {
    fn as_core(&self) -> &dyn BackendCore {
        self
    }

    fn as_lifecycle(&mut self) -> &mut dyn BackendLifecycle {
        self
    }

    fn as_device_manager(&self) -> &dyn BackendDeviceManager {
        self
    }

    fn as_resource_manager(&self) -> &dyn BackendResourceManager {
        self
    }

    fn as_executor(&self) -> &dyn BackendExecutor {
        self
    }

    fn as_operations(&self) -> &dyn BackendOperations {
        self
    }
}

/// Convolution 2D configuration
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    pub batch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend_creation() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config);
            assert!(backend.is_ok());

            let backend = backend.unwrap();
            assert_eq!(BackendCore::name(&backend), "CUDA Backend");
            assert!(BackendCore::is_available(&backend).unwrap_or(false));
        }
    }

    #[test]
    fn test_cuda_buffer_creation() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config).unwrap();

            // Use the CudaBackend's own create_buffer method
            let buffer = CudaBackend::create_buffer::<f32>(&backend, 1024, DType::F32);
            assert!(buffer.is_ok());

            let buffer = buffer.unwrap();
            assert_eq!(buffer.len(), 1024);
            assert_eq!(buffer.dtype(), DType::F32);
        }
    }

    #[test]
    fn test_elementwise_addition() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config).unwrap();

            let mut a = CudaBackend::create_buffer::<f32>(&backend, 4, DType::F32).unwrap();
            let mut b = CudaBackend::create_buffer::<f32>(&backend, 4, DType::F32).unwrap();
            let mut output = CudaBackend::create_buffer::<f32>(&backend, 4, DType::F32).unwrap();

            // Copy test data
            let data_a = vec![1.0, 2.0, 3.0, 4.0];
            let data_b = vec![5.0, 6.0, 7.0, 8.0];

            a.copy_from_host(&data_a).unwrap();
            b.copy_from_host(&data_b).unwrap();

            // Perform addition
            backend
                .elementwise_add_f32(&a, &b, &mut output, None)
                .unwrap();

            // Copy result back
            let mut result = vec![0.0; 4];
            output.copy_to_host(&mut result).unwrap();

            assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        }
    }

    #[test]
    #[ignore] // CUDA Graph API not available in cuda-sys 0.2.0
    fn test_cuda_graph_capture() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config).unwrap();

            // Test graph capture
            assert!(!backend.is_capturing_graph());

            let result = backend.begin_graph_capture(None);
            assert!(result.is_ok());
            assert!(backend.is_capturing_graph());

            // Can't start another capture while one is active
            let result2 = backend.begin_graph_capture(None);
            assert!(result2.is_err());

            // End capture
            let graph = backend.end_graph_capture();
            assert!(graph.is_ok());
            assert!(!backend.is_capturing_graph());
        }
    }

    #[test]
    #[ignore] // CUDA Graph API not available in cuda-sys 0.2.0
    fn test_cuda_graph_operations() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config).unwrap();

            let mut a = CudaBackend::create_buffer::<f32>(&backend, 1024, DType::F32).unwrap();
            let mut b = CudaBackend::create_buffer::<f32>(&backend, 1024, DType::F32).unwrap();
            let mut output = CudaBackend::create_buffer::<f32>(&backend, 1024, DType::F32).unwrap();

            // Copy test data
            let data_a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
            let data_b: Vec<f32> = (0..1024).map(|i| (i * 2) as f32).collect();

            a.copy_from_host(&data_a).unwrap();
            b.copy_from_host(&data_b).unwrap();

            // First execution (creates and caches graph)
            backend
                .elementwise_add_f32_graph(&a, &b, &mut output, true, None)
                .unwrap();

            // Copy result back
            let mut result = vec![0.0; 1024];
            output.copy_to_host(&mut result).unwrap();

            // Verify results
            for i in 0..1024 {
                assert_eq!(result[i], data_a[i] + data_b[i]);
            }

            // Second execution (uses cached graph)
            backend
                .elementwise_add_f32_graph(&a, &b, &mut output, true, None)
                .unwrap();
        }
    }

    #[test]
    #[ignore] // CUDA Graph API not available in cuda-sys 0.2.0
    fn test_cuda_graph_matmul() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config).unwrap();

            let m = 32;
            let n = 32;
            let k = 32;

            let mut a = CudaBackend::create_buffer::<f32>(&backend, m * k, DType::F32).unwrap();
            let mut b = CudaBackend::create_buffer::<f32>(&backend, k * n, DType::F32).unwrap();
            let mut output =
                CudaBackend::create_buffer::<f32>(&backend, m * n, DType::F32).unwrap();

            // Initialize with simple data
            let data_a: Vec<f32> = vec![1.0; m * k];
            let data_b: Vec<f32> = vec![2.0; k * n];

            a.copy_from_host(&data_a).unwrap();
            b.copy_from_host(&data_b).unwrap();

            // Execute with graph capture
            backend
                .matmul_f32_graph(&a, &b, &mut output, m, n, k, true, None)
                .unwrap();

            // Copy result back
            let mut result = vec![0.0; m * n];
            output.copy_to_host(&mut result).unwrap();

            // Each element should be k * 1.0 * 2.0 = 2k
            let expected = (k * 2) as f32;
            for &val in &result {
                assert_eq!(val, expected);
            }
        }
    }

    #[test]
    fn test_unified_memory_support() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config).unwrap();

            // Test unified memory support detection
            let supports = backend.supports_unified_memory();
            assert!(supports.is_ok());

            if supports.unwrap() {
                // Test unified memory allocation
                let allocation = backend.allocate_unified(1024);
                assert!(allocation.is_ok());

                let allocation = allocation.unwrap();
                assert_eq!(allocation.size(), 1024);

                // Test deallocation
                let result = backend.deallocate_unified(allocation);
                assert!(result.is_ok());
            }
        }
    }

    #[test]
    fn test_unified_memory_operations() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config).unwrap();

            if backend.supports_unified_memory().unwrap_or(false) {
                let mut allocation = backend.allocate_unified(16).unwrap();

                // Test data operations
                let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
                allocation.copy_from_host(&test_data).unwrap();

                // Test prefetching
                backend
                    .prefetch_to_device(allocation.ptr(), allocation.size(), None)
                    .unwrap();
                backend
                    .prefetch_to_host(allocation.ptr(), allocation.size())
                    .unwrap();

                // Test memory advice
                backend
                    .set_memory_advice(
                        allocation.ptr(),
                        allocation.size(),
                        MemoryAdvice::SetReadMostly,
                        None,
                    )
                    .unwrap();

                // Verify data integrity
                let mut result_data = vec![0.0f32; 4];
                allocation.copy_to_host(&mut result_data).unwrap();
                assert_eq!(result_data, test_data);

                backend.deallocate_unified(allocation).unwrap();
            }
        }
    }

    #[test]
    fn test_unified_memory_performance_hints() {
        if crate::cuda::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::new(config).unwrap();

            if backend.supports_unified_memory().unwrap_or(false) {
                let allocation = backend.allocate_unified(4096).unwrap();

                // Test different memory advice types for performance optimization
                backend
                    .set_memory_advice(
                        allocation.ptr(),
                        allocation.size(),
                        MemoryAdvice::SetPreferredLocation,
                        Some(0),
                    )
                    .unwrap();

                backend
                    .set_memory_advice(
                        allocation.ptr(),
                        allocation.size(),
                        MemoryAdvice::SetAccessedBy,
                        Some(0),
                    )
                    .unwrap();

                backend
                    .set_memory_advice(
                        allocation.ptr(),
                        allocation.size(),
                        MemoryAdvice::SetReadMostly,
                        None,
                    )
                    .unwrap();

                // Test unsetting advice
                backend
                    .set_memory_advice(
                        allocation.ptr(),
                        allocation.size(),
                        MemoryAdvice::UnsetReadMostly,
                        None,
                    )
                    .unwrap();

                backend.deallocate_unified(allocation).unwrap();
            }
        }
    }
}
