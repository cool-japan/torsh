//! CPU Backend Implementation

use crate::backend::{
    BackendCapabilities, BackendCore, BackendDeviceManager, BackendExecutor, BackendLifecycle,
    BackendOperations, BackendOps, BackendResourceManager, BackendType, CapabilityValue,
    PerformanceHints,
};
use crate::cpu::buffer::{BufferCpuExt, CpuBuffer};
use crate::cpu::{
    CpuConvolutionOps, CpuDevice, CpuFftOps, CpuKernel, CpuMemoryManager, CpuProfiler, CpuRnnOps,
    PlatformOptimizer, WasmSimdOps,
};
use crate::error::{conversion, BackendResult};
use crate::{
    Backend, Buffer, BufferDescriptor, Device, Kernel, KernelDescriptor, MemoryManager,
    MemoryPoolConfig, Profiler,
};
use async_trait::async_trait;
use std::sync::Once;
use torsh_core::device::DeviceType;
use torsh_core::DType;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Static initialization for global thread pool
static THREAD_POOL_INIT: Once = Once::new();

/// Thread pool management with better error handling
struct ThreadPoolManager;

impl ThreadPoolManager {
    /// Safely initialize the global thread pool
    fn initialize_global_pool(num_threads: Option<usize>) -> BackendResult<()> {
        use std::sync::Mutex;
        static INIT_ERROR: Mutex<Option<String>> = Mutex::new(None);

        THREAD_POOL_INIT.call_once(|| {
            let mut builder = rayon::ThreadPoolBuilder::new();

            if let Some(threads) = num_threads {
                // Cap thread count to reasonable limits to avoid OS resource exhaustion
                let max_threads = (num_cpus::get() * 8).max(64); // Max 8x CPU cores or 64, whichever is higher
                let capped_threads = threads.min(max_threads);

                builder = builder.num_threads(capped_threads);
            }

            if let Err(e) = builder.build_global() {
                // Store the error for later retrieval
                if let Ok(mut error_opt) = INIT_ERROR.lock() {
                    *error_opt = Some(e.to_string());
                }
            }
        });

        // Check if there was an initialization error
        if let Ok(error_opt) = INIT_ERROR.lock() {
            if let Some(ref error_msg) = *error_opt {
                // Check if the error is about the thread pool already being initialized
                if error_msg.contains("already been initialized") {
                    // This is expected in tests, just return Ok
                    return Ok(());
                } else {
                    return Err(conversion::cpu_error_with_context(
                        format!("Failed to initialize global thread pool: {}", error_msg),
                        "thread_pool_init",
                    ));
                }
            }
        }

        Ok(())
    }

    /// Check if the global thread pool is already initialized
    fn is_initialized() -> bool {
        THREAD_POOL_INIT.is_completed()
    }

    /// Get the current number of threads in the pool
    fn current_num_threads() -> usize {
        rayon::current_num_threads()
    }
}

/// CPU Backend Builder
#[derive(Debug)]
pub struct CpuBackendBuilder {
    num_threads: Option<usize>,
    memory_pool_config: Option<MemoryPoolConfig>,
    enable_platform_optimization: bool,
}

impl Default for CpuBackendBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackendBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            num_threads: None,
            memory_pool_config: None,
            enable_platform_optimization: true,
        }
    }

    /// Set the number of threads
    pub fn num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Set memory pool configuration
    pub fn memory_pool(mut self, config: MemoryPoolConfig) -> Self {
        self.memory_pool_config = Some(config);
        self
    }

    /// Enable or disable platform-specific optimization
    pub fn platform_optimization(mut self, enable: bool) -> Self {
        self.enable_platform_optimization = enable;
        self
    }

    /// Build the CPU backend
    pub fn build(self) -> BackendResult<CpuBackend> {
        let num_cores = CpuBackend::detect_cpu_cores();

        // Initialize thread pool safely
        if !ThreadPoolManager::is_initialized() {
            ThreadPoolManager::initialize_global_pool(self.num_threads)?;
        } else if let Some(requested_threads) = self.num_threads {
            // Warn if thread pool is already initialized with different settings
            let current_threads = ThreadPoolManager::current_num_threads();
            if current_threads != requested_threads {
                // Silently use existing thread pool configuration to avoid stderr pollution
                // The thread pool is global and can only be initialized once in Rayon
                // Note: Users can query the actual thread count with current_num_threads()
            }
        }

        // For CPU, we typically have one logical device that represents all cores
        let device = CpuDevice::new(0, num_cores)?;

        // Create memory manager with optional pool config
        let memory_manager = if let Some(config) = self.memory_pool_config {
            CpuMemoryManager::with_config(config)
        } else {
            CpuMemoryManager::new()
        };

        // Initialize platform optimizer if enabled
        let platform_optimizer = if self.enable_platform_optimization {
            Some(PlatformOptimizer::new().map_err(|e| {
                conversion::cpu_error_with_context(
                    format!("Failed to initialize platform optimizer: {}", e),
                    "platform_optimizer_init",
                )
            })?)
        } else {
            None
        };

        Ok(CpuBackend {
            devices: vec![device],
            memory_manager,
            profiler: CpuProfiler::new(),
            platform_optimizer,
            wasm_simd: WasmSimdOps::new(),
            fft_ops: CpuFftOps::new(self.num_threads),
            convolution_ops: CpuConvolutionOps::new(self.num_threads),
            rnn_ops: CpuRnnOps::new(self.num_threads),
            initialized: false,
        })
    }
}

/// CPU compute backend implementation
#[derive(Debug)]
pub struct CpuBackend {
    devices: Vec<CpuDevice>,
    memory_manager: CpuMemoryManager,
    profiler: CpuProfiler,
    platform_optimizer: Option<PlatformOptimizer>,
    wasm_simd: WasmSimdOps,
    fft_ops: CpuFftOps,
    convolution_ops: CpuConvolutionOps,
    rnn_ops: CpuRnnOps,
    initialized: bool,
}

impl CpuBackend {
    /// Create a new CPU backend builder
    pub fn builder() -> CpuBackendBuilder {
        CpuBackendBuilder::new()
    }

    /// Create a new CPU backend
    pub fn new() -> BackendResult<Self> {
        let num_cores = Self::detect_cpu_cores();

        // For CPU, we typically have one logical device that represents all cores
        let device = CpuDevice::new(0, num_cores)?;

        // Initialize platform optimizer by default
        let platform_optimizer = Some(PlatformOptimizer::new().map_err(|e| {
            conversion::cpu_error_with_context(
                format!("Failed to initialize platform optimizer: {}", e),
                "platform_optimizer_init",
            )
        })?);

        Ok(Self {
            devices: vec![device],
            memory_manager: CpuMemoryManager::new(),
            profiler: CpuProfiler::new(),
            platform_optimizer,
            wasm_simd: WasmSimdOps::new(),
            fft_ops: CpuFftOps::new(None),
            convolution_ops: CpuConvolutionOps::new(None),
            rnn_ops: CpuRnnOps::new(None),
            initialized: false,
        })
    }

    /// Get access to WASM SIMD operations
    pub fn wasm_simd(&self) -> &WasmSimdOps {
        &self.wasm_simd
    }

    /// Get the number of available CPU cores
    pub fn num_cores(&self) -> usize {
        Self::detect_cpu_cores()
    }

    /// Detect the number of CPU cores
    fn detect_cpu_cores() -> usize {
        #[cfg(feature = "std")]
        {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }
        #[cfg(not(feature = "std"))]
        {
            // Fallback for no-std
            4 // Default assumption
        }
    }

    /// Get access to the platform optimizer if available
    pub fn platform_optimizer(&self) -> Option<&PlatformOptimizer> {
        self.platform_optimizer.as_ref()
    }

    /// Check if platform optimization is enabled
    pub fn has_platform_optimization(&self) -> bool {
        self.platform_optimizer.is_some()
    }

    /// Get CPU features detected by the platform optimizer
    pub fn cpu_features(&self) -> BackendResult<String> {
        if let Some(optimizer) = &self.platform_optimizer {
            Ok(optimizer.get_cpu_info())
        } else {
            Ok("Platform optimization disabled".to_string())
        }
    }
}

// Implement individual traits for better modularity
impl BackendCore for CpuBackend {
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn name(&self) -> &str {
        "CPU Backend"
    }

    fn is_available(&self) -> BackendResult<bool> {
        Ok(true) // CPU backend is always available
    }

    fn capabilities(&self) -> BackendCapabilities {
        use crate::backend::{ExtendedCapabilities, HardwareFeature, PrecisionMode};

        let mut extended_capabilities = ExtendedCapabilities::default();
        extended_capabilities.precision_modes = vec![PrecisionMode::F32, PrecisionMode::F64];
        extended_capabilities.hardware_features = vec![
            HardwareFeature::VectorUnits,  // CPU has SIMD/vector units
            HardwareFeature::SharedMemory, // CPU has cache hierarchy
        ];
        extended_capabilities.execution_model.supports_simd = true;
        extended_capabilities
            .execution_model
            .supports_task_parallelism = true;
        extended_capabilities
            .execution_model
            .supports_data_parallelism = true;
        extended_capabilities.execution_model.max_concurrent_streams =
            Some(self.num_cores() as u32);

        BackendCapabilities {
            max_buffer_size: usize::MAX, // Limited by available system memory
            max_compute_units: self.num_cores(),
            max_workgroup_size: (u32::MAX, 1, 1), // CPU doesn't have workgroup size limits
            supported_dtypes: vec![
                DType::F32,
                DType::F64,
                DType::I32,
                DType::I64,
                DType::I16,
                DType::I8,
                DType::U8,
                DType::Bool,
            ],
            supports_async: false,         // CPU operations are synchronous
            supports_unified_memory: true, // CPU uses system memory
            supports_sub_buffers: true,
            supports_kernel_caching: true,
            memory_bandwidth_gbps: 50.0, // Typical DDR4 bandwidth
            compute_throughput_gflops: self.num_cores() as f32 * 10.0, // Rough estimate
            extended_capabilities,
        }
    }

    fn performance_hints(&self) -> PerformanceHints {
        PerformanceHints {
            preferred_workgroup_size: (self.num_cores() as u32, 1, 1),
            memory_alignment: 64, // Cache line size
            prefer_vectorized: cfg!(feature = "simd"),
            prefer_async: false, // CPU operations are synchronous
            optimal_batch_size: 1024,
            cache_kernels: true,
        }
    }
}

#[async_trait]
impl BackendLifecycle for CpuBackend {
    async fn initialize(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Ensure thread pool is initialized safely
        if !ThreadPoolManager::is_initialized() {
            // Initialize with default thread count if not already done
            ThreadPoolManager::initialize_global_pool(None)?;
        }

        self.initialized = true;
        Ok(())
    }

    async fn shutdown(&mut self) -> BackendResult<()> {
        self.initialized = false;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl BackendDeviceManager for CpuBackend {
    fn devices(&self) -> BackendResult<Vec<Device>> {
        Ok(self.devices.iter().map(|d| d.to_device()).collect())
    }

    fn default_device(&self) -> BackendResult<Device> {
        self.devices
            .first()
            .ok_or_else(|| {
                conversion::cpu_error_with_context("No CPU device available", "default_device")
            })
            .map(|d| d.to_device())
    }

    fn create_device(&self, device_id: usize) -> BackendResult<Device> {
        if device_id >= self.devices.len() {
            return Err(conversion::cpu_error_with_context(
                format!(
                    "CPU device {} not found (only {} available)",
                    device_id,
                    self.devices.len()
                ),
                "create_device",
            ));
        }
        Ok(self.devices[device_id].to_device())
    }

    fn device_count(&self) -> BackendResult<usize> {
        Ok(self.devices.len())
    }

    fn is_device_available(&self, device_id: usize) -> bool {
        device_id < self.devices.len()
    }
}

impl BackendResourceManager for CpuBackend {
    fn create_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer> {
        CpuBuffer::new_buffer(device.clone(), descriptor)
    }

    fn create_kernel(
        &self,
        device: &Device,
        descriptor: &KernelDescriptor,
    ) -> BackendResult<Kernel> {
        CpuKernel::new_kernel(device.clone(), descriptor)
    }

    fn memory_manager(
        &self,
        _device: &Device,
    ) -> BackendResult<Box<dyn MemoryManager + Send + Sync>> {
        Ok(Box::new(self.memory_manager.clone()))
    }

    fn profiler(&self) -> BackendResult<Box<dyn Profiler + Send + Sync>> {
        Ok(Box::new(self.profiler.clone()))
    }

    fn create_scoped_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer> {
        // For CPU backend, scoped buffer is just a regular buffer since CPU memory is managed by the OS
        self.create_buffer(device, descriptor)
    }
}

impl crate::backend::BackendAdvancedResourceManager for CpuBackend {
    fn create_resource_with_cleanup<T, F>(
        &self,
        device: &Device,
        factory: F,
        cleanup: impl FnOnce(&T) + Send + 'static,
    ) -> BackendResult<crate::backend::ManagedResource<T>>
    where
        T: Send + Sync + 'static,
        F: FnOnce(&Device) -> BackendResult<T>,
    {
        let resource = factory(device)?;
        Ok(crate::backend::ManagedResource::new_with_cleanup(
            resource, cleanup,
        ))
    }
}

#[async_trait]
impl BackendExecutor for CpuBackend {
    async fn synchronize(&self, _device: &Device) -> BackendResult<()> {
        // CPU operations are synchronous, so nothing to do
        Ok(())
    }

    async fn copy_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> BackendResult<()> {
        if !src.is_cpu() || !dst.is_cpu() {
            return Err(conversion::cpu_error_with_context(
                "Both source and destination buffers must be CPU buffers",
                "copy_buffer",
            ));
        }

        // Use the safer CpuBuffer access methods
        let src_buffer = src.as_cpu_buffer().ok_or_else(|| {
            conversion::cpu_error_with_context("Failed to get source CPU buffer", "copy_buffer")
        })?;
        let dst_buffer = dst.as_cpu_buffer().ok_or_else(|| {
            conversion::cpu_error_with_context(
                "Failed to get destination CPU buffer",
                "copy_buffer",
            )
        })?;

        // Validate bounds
        if src_offset + size > src.size {
            return Err(conversion::cpu_error_with_context(
                format!(
                    "Source offset {} + size {} exceeds buffer size {}",
                    src_offset, size, src.size
                ),
                "copy_buffer",
            ));
        }
        if dst_offset + size > dst.size {
            return Err(conversion::cpu_error_with_context(
                format!(
                    "Destination offset {} + size {} exceeds buffer size {}",
                    dst_offset, size, dst.size
                ),
                "copy_buffer",
            ));
        }

        // Use the safe copy_to method from CpuBuffer
        src_buffer
            .copy_to(dst_buffer, src_offset, dst_offset, size)
            .map_err(|e| {
                conversion::cpu_error_with_context(
                    format!("Buffer copy failed: {}", e),
                    "copy_buffer",
                )
            })?;

        Ok(())
    }

    async fn copy_to_device(
        &self,
        src: &[u8],
        dst: &Buffer,
        dst_offset: usize,
    ) -> BackendResult<()> {
        if !dst.is_cpu() {
            return Err(conversion::cpu_error_with_context(
                "Destination buffer must be a CPU buffer",
                "copy_to_device",
            ));
        }

        let dst_buffer = dst.as_cpu_buffer().ok_or_else(|| {
            conversion::cpu_error_with_context(
                "Failed to get destination CPU buffer",
                "copy_to_device",
            )
        })?;

        // Validate bounds
        if dst_offset + src.len() > dst.size {
            return Err(conversion::cpu_error_with_context(
                format!(
                    "Destination offset {} + source size {} exceeds buffer size {}",
                    dst_offset,
                    src.len(),
                    dst.size
                ),
                "copy_to_device",
            ));
        }

        // Use the safe write_bytes method from CpuBuffer
        dst_buffer.write_bytes(src, dst_offset).map_err(|e| {
            conversion::cpu_error_with_context(
                format!("Copy to device failed: {}", e),
                "copy_to_device",
            )
        })?;

        Ok(())
    }

    async fn copy_from_device(
        &self,
        src: &Buffer,
        dst: &mut [u8],
        src_offset: usize,
    ) -> BackendResult<()> {
        if !src.is_cpu() {
            return Err(conversion::cpu_error_with_context(
                "Source buffer must be a CPU buffer",
                "copy_from_device",
            ));
        }

        let src_buffer = src.as_cpu_buffer().ok_or_else(|| {
            conversion::cpu_error_with_context(
                "Failed to get source CPU buffer",
                "copy_from_device",
            )
        })?;

        // Validate bounds
        if src_offset + dst.len() > src.size {
            return Err(conversion::cpu_error_with_context(
                format!(
                    "Source offset {} + destination size {} exceeds buffer size {}",
                    src_offset,
                    dst.len(),
                    src.size
                ),
                "copy_from_device",
            ));
        }

        // Use the safe read_bytes method from CpuBuffer
        src_buffer.read_bytes(dst, src_offset).map_err(|e| {
            conversion::cpu_error_with_context(
                format!("Copy from device failed: {}", e),
                "copy_from_device",
            )
        })?;

        Ok(())
    }

    async fn execute_kernel(
        &self,
        kernel: &Kernel,
        _buffers: &[&Buffer],
        _uniform_data: &[u8],
        _workgroup_size: (u32, u32, u32),
        _workgroup_count: (u32, u32, u32),
    ) -> BackendResult<()> {
        // For now, use the kernel name to dispatch to appropriate function
        // In a real implementation, you'd extract the CPU kernel implementation
        // from the abstract Kernel
        return Err(conversion::kernel_error_with_context(
            "Abstract kernel execution not yet implemented for CPU backend",
            &kernel.name,
            "CPU",
            Some("cpu:0"),
        ));
    }
}

impl BackendOperations for CpuBackend {
    fn fft_ops(&self) -> Box<dyn crate::fft::FftOps> {
        Box::new(self.fft_ops.clone())
    }

    fn convolution_ops(&self) -> Box<dyn crate::convolution::ConvolutionOps> {
        Box::new(self.convolution_ops.clone())
    }

    fn rnn_ops(&self) -> Box<dyn crate::rnn::RnnOps> {
        Box::new(self.rnn_ops.clone())
    }

    fn sparse_ops(&self) -> Box<dyn crate::sparse_ops::SparseOps<f32>> {
        Box::new(crate::sparse_ops::DefaultSparseOps::new(
            self.default_device().unwrap(),
        ))
    }

    fn quantization_ops(&self) -> Box<dyn crate::quantization::QuantizationOps> {
        Box::new(crate::quantization::CpuQuantizationOps::new())
    }

    fn operations_bundle(&self) -> crate::backend::OperationsBundle {
        crate::backend::OperationsBundle::new(
            self.fft_ops(),
            self.convolution_ops(),
            self.rnn_ops(),
            self.sparse_ops(),
            self.quantization_ops(),
        )
    }
}

impl BackendOps for CpuBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::Cpu
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
        ]
    }

    fn supports_op(&self, op_name: &str) -> bool {
        self.available_ops().contains(&op_name)
    }

    fn supports_fft(&self) -> bool {
        true
    }

    fn supports_convolution(&self) -> bool {
        true
    }

    fn supports_rnn(&self) -> bool {
        true
    }

    fn supports_sparse(&self) -> bool {
        true
    }

    fn supports_quantization(&self) -> bool {
        true
    }

    fn operation_capabilities(
        &self,
        op_name: &str,
    ) -> Option<std::collections::HashMap<String, CapabilityValue>> {
        let mut caps = std::collections::HashMap::new();

        match op_name {
            "matmul" => {
                caps.insert("max_size".to_string(), CapabilityValue::Int(16384));
                caps.insert("supports_batched".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_strided".to_string(), CapabilityValue::Bool(true));
            }
            "conv2d" => {
                caps.insert("max_kernel_size".to_string(), CapabilityValue::Int(15));
                caps.insert("supports_groups".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_dilation".to_string(), CapabilityValue::Bool(true));
            }
            "fft" => {
                caps.insert("max_size".to_string(), CapabilityValue::Int(65536));
                caps.insert("supports_real".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_batched".to_string(), CapabilityValue::Bool(true));
            }
            _ => return None,
        }

        Some(caps)
    }
}

// Main Backend trait implementation using composition
impl Backend for CpuBackend {
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

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create CPU backend")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_cpu_backend_initialization() {
        let mut backend = CpuBackend::new().unwrap();
        assert!(!backend.initialized);

        backend.initialize().await.unwrap();
        assert!(backend.initialized);

        backend.shutdown().await.unwrap();
        assert!(!backend.initialized);
    }

    #[tokio::test]
    async fn test_cpu_backend_devices() {
        let backend = CpuBackend::new().unwrap();

        let devices = backend.devices().unwrap();
        assert!(!devices.is_empty());

        let default_device = backend.default_device().unwrap();
        assert_eq!(default_device.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_cpu_backend_capabilities() {
        let backend = CpuBackend::new().unwrap();
        let caps = backend.capabilities();

        assert!(caps.max_compute_units > 0);
        assert!(caps.supports_unified_memory);
        assert!(!caps.supported_dtypes.is_empty());
    }

    #[test]
    fn test_thread_pool_safety() {
        // Test that multiple backend initializations don't cause panics
        let backend1 = CpuBackend::builder().num_threads(2).build();
        assert!(backend1.is_ok());

        // Second initialization with different thread count should work (with warning)
        let backend2 = CpuBackend::builder().num_threads(4).build();
        assert!(backend2.is_ok());

        // Check that thread pool is initialized
        assert!(ThreadPoolManager::is_initialized());
        assert!(ThreadPoolManager::current_num_threads() > 0);
    }

    #[tokio::test]
    async fn test_multiple_initialization() {
        // Test that multiple initialize() calls are safe
        let mut backend1 = CpuBackend::new().unwrap();
        let mut backend2 = CpuBackend::new().unwrap();

        let result1 = backend1.initialize().await;
        let result2 = backend2.initialize().await;

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(backend1.initialized);
        assert!(backend2.initialized);
    }

    #[tokio::test]
    async fn test_thread_pool_manager() {
        // Test thread pool manager functions
        let mut backend = CpuBackend::new().unwrap();

        // Initialize the backend to set up the thread pool
        backend.initialize().await.unwrap();

        // After initializing the backend, thread pool should be initialized
        assert!(ThreadPoolManager::is_initialized());

        // Current thread count should be reasonable
        let thread_count = ThreadPoolManager::current_num_threads();
        assert!(thread_count > 0);
        assert!(thread_count <= num_cpus::get() * 2); // Sanity check
    }

    // ========== EDGE CASE AND ERROR CONDITION TESTS ==========

    #[test]
    fn test_platform_optimization_features() {
        // Test platform optimization integration
        let backend = CpuBackend::new().unwrap();

        // Should have platform optimization enabled by default
        assert!(backend.has_platform_optimization());

        // Should be able to get CPU features info
        let cpu_info = backend.cpu_features().unwrap();
        assert!(!cpu_info.is_empty());

        if let Some(optimizer) = backend.platform_optimizer() {
            // Should have detected some CPU features
            let info = optimizer.get_cpu_info();
            assert!(!info.is_empty());
        }
    }

    #[test]
    fn test_platform_optimization_disabled() {
        // Test backend without platform optimization
        let backend = CpuBackend::builder()
            .platform_optimization(false)
            .build()
            .unwrap();

        assert!(!backend.has_platform_optimization());

        let cpu_info = backend.cpu_features().unwrap();
        assert!(cpu_info.contains("disabled"));
    }

    #[test]
    fn test_cpu_core_detection_edge_cases() {
        // Test CPU core detection
        let cores = CpuBackend::detect_cpu_cores();
        assert!(cores > 0);
        assert!(cores <= 1024); // Sanity check for reasonable upper bound

        // Test that backend reports same core count
        let backend = CpuBackend::new().unwrap();
        assert_eq!(backend.num_cores(), cores);
    }

    #[tokio::test]
    async fn test_invalid_device_operations() {
        let backend = CpuBackend::new().unwrap();

        // Test creating device with invalid ID
        let result = backend.create_device(999);
        assert!(result.is_err());

        let error_str = result.unwrap_err().to_string();
        assert!(error_str.contains("999"));
        assert!(error_str.contains("not found"));
    }

    #[tokio::test]
    async fn test_buffer_operations_edge_cases() {
        let backend = CpuBackend::new().unwrap();
        let device = backend.default_device().unwrap();

        // Test buffer creation with zero size
        let desc = BufferDescriptor {
            size: 0,
            usage: crate::BufferUsage::STORAGE,
            location: crate::MemoryLocation::Device,
            dtype: None,
            shape: None,
            initial_data: None,
            alignment: Some(8),
            zero_init: false,
        };

        let result = backend.create_buffer(&device, &desc);
        // Should handle gracefully (may succeed with 0-byte allocation)
        if let Err(e) = result {
            // Error message should be descriptive
            let error_str = e.to_string();
            assert!(!error_str.is_empty());
        }

        // Test buffer creation with extremely large size
        let desc = BufferDescriptor {
            size: usize::MAX,
            usage: crate::BufferUsage::STORAGE,
            location: crate::MemoryLocation::Device,
            dtype: None,
            shape: None,
            initial_data: None,
            alignment: Some(8),
            zero_init: false,
        };

        let result = backend.create_buffer(&device, &desc);
        // Should fail gracefully
        assert!(result.is_err());
        let error_str = result.unwrap_err().to_string();
        assert!(!error_str.is_empty());
    }

    #[tokio::test]
    async fn test_memory_manager_edge_cases() {
        let backend = CpuBackend::new().unwrap();
        let device = backend.default_device().unwrap();

        let memory_manager = backend.memory_manager(&device).unwrap();

        // Test memory statistics
        let stats = memory_manager.stats();
        // Should have reasonable values
        assert!(stats.total_memory >= stats.allocated_memory);
        assert!(stats.total_allocations >= stats.active_allocations);
    }

    #[tokio::test]
    async fn test_concurrent_initialization() {
        use std::sync::Arc;
        use std::thread;

        // Test concurrent backend initialization
        let mut handles = vec![];
        let results = Arc::new(std::sync::Mutex::new(Vec::new()));

        for i in 0..5 {
            let results_clone = Arc::clone(&results);
            let handle = thread::spawn(move || {
                let backend = CpuBackend::builder()
                    .num_threads(2 + i) // Different thread counts
                    .build();

                let mut results = results_clone.lock().unwrap();
                results.push(backend.is_ok());
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let results = results.lock().unwrap();
        // At least some should succeed
        assert!(results.iter().any(|&success| success));
    }

    #[tokio::test]
    async fn test_kernel_creation_edge_cases() {
        let backend = CpuBackend::new().unwrap();
        let device = backend.default_device().unwrap();

        // Test kernel creation with empty name
        let desc = KernelDescriptor {
            name: "".to_string(),
            source: crate::kernel::KernelSource::Source {
                code: "test_source".to_string(),
                language: crate::kernel::KernelLanguage::Glsl,
            },
            compile_options: vec![],
            parameters: vec![],
            workgroup_size_hint: None,
            cache: true,
        };

        let result = backend.create_kernel(&device, &desc);
        // Should handle gracefully
        if let Err(e) = result {
            let error_str = e.to_string();
            assert!(!error_str.is_empty());
        }

        // Test kernel creation with very long name
        let long_name = "x".repeat(10000);
        let desc = KernelDescriptor {
            name: long_name.clone(),
            source: crate::kernel::KernelSource::Source {
                code: "test_source".to_string(),
                language: crate::kernel::KernelLanguage::Glsl,
            },
            compile_options: vec![],
            parameters: vec![],
            workgroup_size_hint: None,
            cache: true,
        };

        let result = backend.create_kernel(&device, &desc);
        // Should handle gracefully
        if let Err(e) = result {
            let error_str = e.to_string();
            assert!(!error_str.is_empty());
        }
    }

    #[tokio::test]
    async fn test_profiler_edge_cases() {
        let backend = CpuBackend::new().unwrap();

        let profiler = backend.profiler().unwrap();

        // Test profiler basic functionality
        // Note: Actual profiling tests would need more setup
        // This just tests that we can get a profiler instance
        drop(profiler); // Should not panic
    }

    #[tokio::test]
    async fn test_synchronization_edge_cases() {
        let backend = CpuBackend::new().unwrap();
        let device = backend.default_device().unwrap();

        // CPU synchronization should always succeed immediately
        let result = backend.synchronize(&device).await;
        assert!(result.is_ok());

        // Multiple synchronizations should be safe
        for _ in 0..10 {
            let result = backend.synchronize(&device).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_buffer_copy_edge_cases() {
        let backend = CpuBackend::new().unwrap();
        let device = backend.default_device().unwrap();

        // Create small test buffers with proper alignment
        let desc = BufferDescriptor {
            size: 100,
            usage: crate::BufferUsage::STORAGE,
            location: crate::MemoryLocation::Device,
            dtype: None,
            shape: None,
            initial_data: None,
            alignment: Some(8), // Use 8-byte alignment instead of 1-byte
            zero_init: false,
        };

        let src_buffer = backend.create_buffer(&device, &desc);
        let dst_buffer = backend.create_buffer(&device, &desc);

        if let (Ok(src), Ok(dst)) = (src_buffer, dst_buffer) {
            // Test copy with invalid offset
            let result = backend.copy_buffer(&src, &dst, 150, 0, 10).await;
            assert!(result.is_err());

            // Test copy with invalid size
            let result = backend.copy_buffer(&src, &dst, 0, 0, 150).await;
            assert!(result.is_err());

            // Test valid copy
            let result = backend.copy_buffer(&src, &dst, 0, 0, 50).await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_host_device_transfer_edge_cases() {
        let backend = CpuBackend::new().unwrap();
        let device = backend.default_device().unwrap();

        let desc = BufferDescriptor {
            size: 100,
            usage: crate::BufferUsage::STORAGE,
            location: crate::MemoryLocation::Device,
            dtype: None,
            shape: None,
            initial_data: None,
            alignment: Some(8),
            zero_init: false,
        };

        if let Ok(buffer) = backend.create_buffer(&device, &desc) {
            // Test copy to device with invalid offset
            let src_data = vec![1u8; 50];
            let result = backend.copy_to_device(&src_data, &buffer, 60).await;
            assert!(result.is_err());

            // Test copy from device with invalid size
            let mut dst_data = vec![0u8; 200];
            let result = backend.copy_from_device(&buffer, &mut dst_data, 0).await;
            assert!(result.is_err());
        }
    }
}
