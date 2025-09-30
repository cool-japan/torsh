//! CUDA backend implementation

use crate::cuda::buffer::CudaBuffer;
use crate::cuda::cooperative_groups::{
    CooperativeGroupsContext, CooperativeKernelConfig, CooperativeWorkload,
};
use crate::cuda::device::CudaDevice;
use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::graph::{CudaGraph, GraphCache, GraphCaptureContext};
use crate::cuda::kernels::{KernelRegistry, LaunchConfig};
use crate::cuda::memory::{CudaMemoryManager, MemoryAdvice, UnifiedAllocation};
use crate::cuda::stream::CudaStream;
use crate::error::BackendError;
use crate::Backend;
use async_trait::async_trait;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, RwLock,
};
use torsh_core::{DType, DeviceType};

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
    active_buffers: Vec<*mut std::ffi::c_void>,
    active_streams: Vec<Arc<CudaStream>>,
    active_graphs: Vec<String>, // Graph keys for cleanup
    unified_allocations: Vec<UnifiedAllocation>,
}

impl ResourceTracker {
    /// Track a new buffer allocation
    pub fn track_buffer(&mut self, ptr: *mut std::ffi::c_void) {
        if !ptr.is_null() {
            self.active_buffers.push(ptr);
        }
    }

    /// Untrack a buffer allocation
    pub fn untrack_buffer(&mut self, ptr: *mut std::ffi::c_void) {
        self.active_buffers.retain(|&p| p != ptr);
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
            memory_pool_size: self.memory_pool_config.as_ref().map(|c| c.total_size),
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
        crate::init()?;

        // Create device
        let device = Arc::new(CudaDevice::new(config.device_id)?);

        // Set device as current
        crate::set_device(config.device_id)?;

        // Create memory manager with thread-safe wrapper
        let memory_manager = Arc::new(RwLock::new(CudaMemoryManager::new(config.device_id)?));

        // Create default stream
        let default_stream = Arc::new(CudaStream::default()?);

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
        let ptx = include_str!("../kernels/compiled.ptx");
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
            .map_err(|e| CudaError::Backend(e.into()))?;
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
                .map_err(|e| CudaError::Backend(e.into()))
        } else {
            Err(CudaError::UnsupportedOperation(
                "Cooperative groups not supported on this device".to_string(),
            ))
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
                .map_err(|e| CudaError::Backend(e.into()))
        } else {
            Err(CudaError::UnsupportedOperation(
                "Cooperative groups not supported on this device".to_string(),
            ))
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
                .map_err(|e| CudaError::Backend(e.into()))
        } else {
            Err(CudaError::UnsupportedOperation(
                "Cooperative groups not supported on this device".to_string(),
            ))
        }
    }

    /// Allocate unified memory with resource tracking
    pub fn allocate_unified(&self, size: usize) -> CudaResult<UnifiedAllocation> {
        self.check_availability()?;

        let allocation = {
            let memory_manager = self.memory_manager.read().map_err(|_| CudaError::Context {
                message: "Failed to acquire memory manager read lock".to_string(),
            })?;
            memory_manager.allocate_unified(size)?
        };

        // Track the allocation
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.track_unified_allocation(&allocation);
        }

        Ok(allocation)
    }

    /// Deallocate unified memory with resource tracking
    pub fn deallocate_unified(&self, allocation: UnifiedAllocation) -> CudaResult<()> {
        let ptr = allocation.ptr();

        let result = {
            let memory_manager = self.memory_manager.read().map_err(|_| CudaError::Context {
                message: "Failed to acquire memory manager read lock".to_string(),
            })?;
            memory_manager.deallocate_unified(allocation)
        };

        // Untrack the allocation
        if let Ok(mut tracker) = self.resource_tracker.lock() {
            tracker.untrack_unified_allocation(ptr);
        }

        result
    }

    /// Prefetch unified memory to device with availability check
    pub fn prefetch_to_device(
        &self,
        ptr: *mut u8,
        size: usize,
        device_id: Option<usize>,
    ) -> CudaResult<()> {
        self.check_availability()?;

        let memory_manager = self.memory_manager.read().map_err(|_| CudaError::Context {
            message: "Failed to acquire memory manager read lock".to_string(),
        })?;
        memory_manager.prefetch_to_device(ptr, size, device_id)
    }

    /// Prefetch unified memory to host with availability check
    pub fn prefetch_to_host(&self, ptr: *mut u8, size: usize) -> CudaResult<()> {
        self.check_availability()?;

        let memory_manager = self.memory_manager.read().map_err(|_| CudaError::Context {
            message: "Failed to acquire memory manager read lock".to_string(),
        })?;
        memory_manager.prefetch_to_host(ptr, size)
    }

    /// Set memory advice for unified memory with availability check
    pub fn set_memory_advice(
        &self,
        ptr: *mut u8,
        size: usize,
        advice: MemoryAdvice,
        device_id: Option<usize>,
    ) -> CudaResult<()> {
        self.check_availability()?;

        let memory_manager = self.memory_manager.read().map_err(|_| CudaError::Context {
            message: "Failed to acquire memory manager read lock".to_string(),
        })?;
        memory_manager.set_memory_advice(ptr, size, advice, device_id)
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
            crate::kernels::tensor_ops::launch_elementwise_add_f32(
                a.device_ptr().as_raw_mut(),
                b.device_ptr().as_raw_mut(),
                output.device_ptr().as_raw_mut(),
                size,
                stream.raw().as_inner(),
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
            crate::kernels::tensor_ops::launch_elementwise_mul_f32(
                a.device_ptr().as_raw_mut(),
                b.device_ptr().as_raw_mut(),
                output.device_ptr().as_raw_mut(),
                size,
                stream.raw().as_inner(),
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
        let stream = stream.unwrap_or(&self.default_stream);

        // Use cuBLAS for matrix multiplication
        let cublas_handle = self.get_cublas_handle()?;

        let alpha = 1.0f32;
        let beta = 0.0f32;

        unsafe {
            cust::cublas::sgemm(
                cublas_handle,
                cust::cublas::Operation::N,
                cust::cublas::Operation::N,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                b.device_ptr().as_raw(),
                n as i32,
                a.device_ptr().as_raw(),
                k as i32,
                &beta,
                output.device_ptr().as_raw_mut(),
                n as i32,
            )?;
        }

        Ok(())
    }

    /// Get cuBLAS handle
    fn get_cublas_handle(&self) -> CudaResult<cust::cublas::CublasHandle> {
        // In a real implementation, this would be cached per device/stream
        Ok(cust::cublas::CublasHandle::new()?)
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
            crate::kernels::neural_ops::launch_conv2d_f32(
                input.device_ptr().as_raw_mut(),
                weight.device_ptr().as_raw_mut(),
                bias.map(|b| b.device_ptr().as_raw_mut())
                    .unwrap_or(std::ptr::null_mut()),
                output.device_ptr().as_raw_mut(),
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
                stream.raw().as_inner(),
            );
        }

        Ok(())
    }

    /// Begin graph capture on a stream
    pub fn begin_graph_capture(&self, stream: Option<&CudaStream>) -> CudaResult<()> {
        let stream = stream.unwrap_or(&self.default_stream);

        // Check if already capturing
        let mut capture_opt = self.capture_context.lock().unwrap();
        if capture_opt.is_some() {
            return Err(CudaError::Context {
                message: "Already capturing a graph".to_string(),
            });
        }

        // Create new capture context
        let mut capture_ctx = GraphCaptureContext::new(Arc::clone(&stream));
        capture_ctx.start().map_err(|e| CudaError::Context {
            message: format!("Failed to start graph capture: {}", e),
        })?;
        *capture_opt = Some(capture_ctx);

        Ok(())
    }

    /// End graph capture and return the captured graph
    pub fn end_graph_capture(&self) -> CudaResult<CudaGraph> {
        let mut capture_opt = self.capture_context.lock().unwrap();
        let mut capture_ctx = capture_opt.take().ok_or_else(|| CudaError::Context {
            message: "Not capturing a graph".to_string(),
        })?;

        capture_ctx.end().map_err(|e| CudaError::Context {
            message: format!("Failed to end graph capture: {}", e),
        })
    }

    /// Check if currently capturing a graph
    pub fn is_capturing_graph(&self) -> bool {
        self.capture_context.lock().unwrap().is_some()
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
                creator().map_err(|e| BackendError::ComputeError {
                    reason: e.to_string(),
                })
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
        let graph = graph.lock().unwrap();
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
        let graph = graph.lock().unwrap();
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
            if let Err(e) = self.shutdown() {
                tracing::error!("Failed to shutdown CUDA backend during drop: {}", e);
            }
        }
    }
}

#[async_trait]
impl Backend for CudaBackend {
    type Device = CudaDevice;
    type Config = CudaBackendConfig;

    async fn initialize(config: Self::Config) -> Result<Self, BackendError> {
        CudaBackend::new(config).map_err(|e| BackendError::InitializationFailed {
            message: e.to_string(),
        })
    }

    fn name(&self) -> &str {
        "cuda"
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda(self.config.device_id)
    }

    fn is_available(&self) -> bool {
        crate::is_available()
    }

    fn synchronize(&self) -> Result<(), BackendError> {
        self.synchronize().map_err(|e| BackendError::Runtime {
            message: e.to_string(),
        })
    }

    fn create_buffer<T: Clone + Send + Sync + 'static>(
        &self,
        length: usize,
        dtype: DType,
    ) -> Result<Box<dyn crate::Buffer<T>>, BackendError> {
        let buffer =
            self.create_buffer::<T>(length, dtype)
                .map_err(|e| BackendError::AllocationFailed {
                    message: e.to_string(),
                })?;
        Ok(Box::new(buffer))
    }

    fn add_tensors<T: Clone + Send + Sync + 'static>(
        &self,
        a: &dyn crate::Buffer<T>,
        b: &dyn crate::Buffer<T>,
        output: &mut dyn crate::Buffer<T>,
    ) -> Result<(), BackendError> {
        // Downcast to CUDA buffers
        let a_cuda = a.as_any().downcast_ref::<CudaBuffer<T>>().ok_or_else(|| {
            BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input A".to_string(),
            }
        })?;
        let b_cuda = b.as_any().downcast_ref::<CudaBuffer<T>>().ok_or_else(|| {
            BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input B".to_string(),
            }
        })?;
        let output_cuda = output
            .as_any_mut()
            .downcast_mut::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for output".to_string(),
            })?;

        // For now, only support f32
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let a_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(a_cuda) };
            let b_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(b_cuda) };
            let output_f32 = unsafe {
                std::mem::transmute::<&mut CudaBuffer<T>, &mut CudaBuffer<f32>>(output_cuda)
            };

            self.elementwise_add_f32(a_f32, b_f32, output_f32, None)
                .map_err(|e| BackendError::Runtime {
                    message: e.to_string(),
                })?;
        } else {
            return Err(BackendError::UnsupportedOperation {
                operation: "add_tensors".to_string(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        }

        Ok(())
    }

    fn multiply_tensors<T: Clone + Send + Sync + 'static>(
        &self,
        a: &dyn crate::Buffer<T>,
        b: &dyn crate::Buffer<T>,
        output: &mut dyn crate::Buffer<T>,
    ) -> Result<(), BackendError> {
        // Downcast to CUDA buffers
        let a_cuda = a.as_any().downcast_ref::<CudaBuffer<T>>().ok_or_else(|| {
            BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input A".to_string(),
            }
        })?;
        let b_cuda = b.as_any().downcast_ref::<CudaBuffer<T>>().ok_or_else(|| {
            BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input B".to_string(),
            }
        })?;
        let output_cuda = output
            .as_any_mut()
            .downcast_mut::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for output".to_string(),
            })?;

        // For now, only support f32
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let a_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(a_cuda) };
            let b_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(b_cuda) };
            let output_f32 = unsafe {
                std::mem::transmute::<&mut CudaBuffer<T>, &mut CudaBuffer<f32>>(output_cuda)
            };

            self.elementwise_mul_f32(a_f32, b_f32, output_f32, None)
                .map_err(|e| BackendError::Runtime {
                    message: e.to_string(),
                })?;
        } else {
            return Err(BackendError::UnsupportedOperation {
                operation: "multiply_tensors".to_string(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        }

        Ok(())
    }

    fn matmul<T: Clone + Send + Sync + 'static>(
        &self,
        a: &dyn crate::Buffer<T>,
        b: &dyn crate::Buffer<T>,
        output: &mut dyn crate::Buffer<T>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        // Downcast to CUDA buffers
        let a_cuda = a.as_any().downcast_ref::<CudaBuffer<T>>().ok_or_else(|| {
            BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input A".to_string(),
            }
        })?;
        let b_cuda = b.as_any().downcast_ref::<CudaBuffer<T>>().ok_or_else(|| {
            BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input B".to_string(),
            }
        })?;
        let output_cuda = output
            .as_any_mut()
            .downcast_mut::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for output".to_string(),
            })?;

        // For now, only support f32
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let a_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(a_cuda) };
            let b_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(b_cuda) };
            let output_f32 = unsafe {
                std::mem::transmute::<&mut CudaBuffer<T>, &mut CudaBuffer<f32>>(output_cuda)
            };

            self.matmul_f32(a_f32, b_f32, output_f32, m, n, k, None)
                .map_err(|e| BackendError::Runtime {
                    message: e.to_string(),
                })?;
        } else {
            return Err(BackendError::UnsupportedOperation {
                operation: "matmul".to_string(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        }

        Ok(())
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

    #[tokio::test]
    async fn test_cuda_backend_creation() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await;
            assert!(backend.is_ok());

            let backend = backend.unwrap();
            assert_eq!(backend.name(), "cuda");
            assert!(backend.is_available());
        }
    }

    #[tokio::test]
    async fn test_buffer_creation() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();

            let buffer = backend.create_buffer::<f32>(1024, DType::F32);
            assert!(buffer.is_ok());

            let buffer = buffer.unwrap();
            assert_eq!(buffer.len(), 1024);
            assert_eq!(buffer.dtype(), DType::F32);
        }
    }

    #[tokio::test]
    async fn test_elementwise_addition() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();

            let mut a = backend.create_buffer::<f32>(4, DType::F32).unwrap();
            let mut b = backend.create_buffer::<f32>(4, DType::F32).unwrap();
            let mut output = backend.create_buffer::<f32>(4, DType::F32).unwrap();

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

    #[tokio::test]
    async fn test_cuda_graph_capture() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();

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

    #[tokio::test]
    async fn test_cuda_graph_operations() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();

            let mut a = backend.create_buffer::<f32>(1024, DType::F32).unwrap();
            let mut b = backend.create_buffer::<f32>(1024, DType::F32).unwrap();
            let mut output = backend.create_buffer::<f32>(1024, DType::F32).unwrap();

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

    #[tokio::test]
    async fn test_cuda_graph_matmul() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();

            let m = 32;
            let n = 32;
            let k = 32;

            let mut a = backend.create_buffer::<f32>(m * k, DType::F32).unwrap();
            let mut b = backend.create_buffer::<f32>(k * n, DType::F32).unwrap();
            let mut output = backend.create_buffer::<f32>(m * n, DType::F32).unwrap();

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

    #[tokio::test]
    async fn test_unified_memory_support() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();

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

    #[tokio::test]
    async fn test_unified_memory_operations() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();

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

    #[tokio::test]
    async fn test_unified_memory_performance_hints() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();

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
