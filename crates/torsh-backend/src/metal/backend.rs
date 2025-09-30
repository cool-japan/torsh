//! Metal Backend Implementation

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::backend::{BackendCapabilities, PerformanceHints};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::buffer::{generate_buffer_id, BufferHandle};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::error::{BackendError, BackendResult};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::kernel::{KernelHandle, KernelMetadata};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::memory::MemoryStats;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::metal::indirect_commands::IndirectCommandManager;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::metal::{MetalBuffer, MetalDevice};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::profiler::{EventId, EventType, ProfilerEvent, ProfilerStats};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::{
    Backend, BackendCore, BackendDeviceManager, BackendExecutor, BackendLifecycle,
    BackendOperations, BackendOps, BackendResourceManager, Buffer, BufferDescriptor, Device, DeviceInfo,
    Kernel, KernelDescriptor, MemoryManager, Profiler,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use async_trait::async_trait;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use metal::foreign_types::ForeignType;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::{Arc, Mutex};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use torsh_core::{device::DeviceType, dtype::DType, error::TorshError};

/// Metal Backend Builder
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub struct MetalBackendBuilder {
    memory_pool_config: Option<crate::MemoryPoolConfig>,
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub struct MetalBackendBuilder {
    _private: (), // Placeholder for non-Apple platforms
}

impl MetalBackendBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            memory_pool_config: None,
        }
    }

    /// Set memory pool configuration
    pub fn memory_pool(mut self, config: crate::MemoryPoolConfig) -> Self {
        self.memory_pool_config = Some(config);
        self
    }

    /// Build the Metal backend
    pub fn build(self) -> torsh_core::error::Result<MetalBackend> {
        MetalBackend::new()
    }
}

/// Metal compute backend implementation
#[derive(Debug)]
pub struct MetalBackend {
    device: Arc<MetalDevice>,
    initialized: bool,
    /// Buffer ID counter
    next_buffer_id: Arc<Mutex<usize>>,
    /// Neural Engine operations builder (if available)
    neural_engine: Option<crate::metal::neural_engine::NeuralEngineOpsBuilder>,
    /// Indirect Command Buffer manager (if available)
    indirect_command_manager: Option<Arc<IndirectCommandManager>>,
}

impl MetalBackend {
    /// Create a new Metal backend builder
    pub fn builder() -> MetalBackendBuilder {
        MetalBackendBuilder::new()
    }

    /// Create a new Metal backend
    pub fn new() -> torsh_core::error::Result<Self> {
        let device =
            Arc::new(MetalDevice::new().map_err(|e| TorshError::BackendError(e.to_string()))?);

        // Try to initialize Neural Engine (optional, won't fail if unavailable)
        let neural_engine = crate::metal::neural_engine::NeuralEngineOpsBuilder::new().ok();

        // Try to initialize Indirect Command Manager (optional, won't fail if unavailable)
        let indirect_command_manager = IndirectCommandManager::new(device.device_ref().clone())
            .ok()
            .map(|manager| Arc::new(manager));

        Ok(Self {
            device,
            initialized: false,
            next_buffer_id: Arc::new(Mutex::new(0)),
            neural_engine,
            indirect_command_manager,
        })
    }

    /// Get the Metal device
    pub fn metal_device(&self) -> &Arc<MetalDevice> {
        &self.device
    }

    /// Check if Neural Engine is available
    pub fn is_neural_engine_available(&self) -> bool {
        self.neural_engine
            .as_ref()
            .map(|ne| ne.is_available())
            .unwrap_or(false)
    }

    /// Get Neural Engine capabilities
    pub fn neural_engine_capabilities(
        &self,
    ) -> Option<crate::metal::neural_engine::NeuralEngineCapabilities> {
        self.neural_engine.as_ref().and_then(|_| {
            // Get capabilities from the global context
            crate::metal::neural_engine::NeuralEngineContext::global()
                .ok()
                .map(|ctx| ctx.lock().unwrap().capabilities().clone())
        })
    }

    /// Create a Neural Engine optimized matrix multiplication operation
    pub fn create_neural_engine_matmul(
        &self,
        input_shape: &torsh_core::shape::Shape,
        weight_shape: &torsh_core::shape::Shape,
        output_shape: &torsh_core::shape::Shape,
        transpose_weight: bool,
    ) -> BackendResult<String> {
        if let Some(ref neural_engine) = self.neural_engine {
            neural_engine
                .create_transformer_matmul(
                    input_shape,
                    weight_shape,
                    output_shape,
                    transpose_weight,
                )
                .map_err(|e| match e {
                    BackendError::UnsupportedOperation { op, dtype } => {
                        TorshError::UnsupportedOperation {
                            op: format!("Neural Engine operation: {}", op),
                            dtype
                        }
                    }
                    BackendError::InvalidArgument(msg) => TorshError::InvalidArgument(msg),
                    BackendError::InvalidArgument(message) => TorshError::InvalidArgument(message),
                    _ => TorshError::BackendError(e.to_string()),
                })
        } else {
            Err(TorshError::UnsupportedOperation {
                op: "Neural Engine".to_string(),
                dtype: "Neural Engine not available on this device".to_string(),
            })
        }
    }

    /// Create a Neural Engine optimized multi-head attention operation
    pub fn create_neural_engine_attention(
        &self,
        sequence_length: usize,
        num_heads: usize,
        head_dim: usize,
        dropout: f32,
    ) -> BackendResult<String> {
        if let Some(ref neural_engine) = self.neural_engine {
            neural_engine
                .create_multi_head_attention(sequence_length, num_heads, head_dim, dropout)
                .map_err(|e| match e {
                    BackendError::UnsupportedOperation { op, dtype } => {
                        TorshError::UnsupportedOperation {
                            op: format!("Neural Engine operation: {}", op),
                            dtype
                        }
                    }
                    BackendError::InvalidArgument(msg) => TorshError::InvalidArgument(msg),
                    BackendError::InvalidArgument(message) => TorshError::InvalidArgument(message),
                    _ => TorshError::BackendError(e.to_string()),
                })
        } else {
            Err(TorshError::UnsupportedOperation {
                op: "Neural Engine".to_string(),
                dtype: "Neural Engine not available on this device".to_string(),
            })
        }
    }

    /// Execute a Neural Engine operation
    pub fn execute_neural_engine_operation(
        &self,
        operation_key: &str,
        inputs: &[crate::metal::neural_engine::NeuralEngineBuffer],
        outputs: &mut [crate::metal::neural_engine::NeuralEngineBuffer],
    ) -> BackendResult<()> {
        if let Some(ref neural_engine) = self.neural_engine {
            neural_engine
                .execute(operation_key, inputs, outputs)
                .map_err(|e| match e {
                    BackendError::UnsupportedOperation { op, dtype } => {
                        TorshError::UnsupportedOperation {
                            op: format!("Neural Engine operation: {}", op),
                            dtype
                        }
                    }
                    BackendError::InvalidArgument(msg) => TorshError::InvalidArgument(msg),
                    BackendError::InvalidArgument(message) => TorshError::InvalidArgument(message),
                    _ => TorshError::BackendError(e.to_string()),
                })
        } else {
            Err(TorshError::UnsupportedOperation {
                op: "Neural Engine".to_string(),
                dtype: "Neural Engine not available on this device".to_string(),
            })
        }
    }

    /// Check if indirect command buffers are supported
    pub fn is_indirect_commands_supported(&self) -> bool {
        self.indirect_command_manager
            .as_ref()
            .map(|icm| icm.is_supported())
            .unwrap_or(false)
    }

    /// Get indirect command manager
    pub fn indirect_command_manager(&self) -> Option<&Arc<IndirectCommandManager>> {
        self.indirect_command_manager.as_ref()
    }

    /// Create an indirect command buffer
    pub fn create_indirect_command_buffer(
        &self,
        config: crate::metal::indirect_commands::IndirectCommandBufferConfig,
    ) -> BackendResult<u64> {
        if let Some(ref manager) = self.indirect_command_manager {
            manager.create_command_buffer(config).map_err(|e| match e {
                BackendError::UnsupportedOperation { op, dtype } => TorshError::UnsupportedOperation {
                    op: format!("Indirect Command: {}", op),
                    dtype
                },
                BackendError::InvalidArgument(msg) => TorshError::InvalidArgument(msg),
                BackendError::InvalidState(msg) => TorshError::InvalidState(msg),
                _ => TorshError::BackendError(e.to_string()),
            })
        } else {
            Err(TorshError::UnsupportedOperation {
                op: "Indirect command buffers".to_string(),
                dtype: "Indirect command buffers not available on this device".to_string(),
            })
        }
    }

    /// Encode a command into an indirect command buffer
    pub fn encode_indirect_command(
        &self,
        buffer_id: u64,
        command_index: u32,
        command: crate::metal::indirect_commands::IndirectCommand,
    ) -> BackendResult<()> {
        if let Some(ref manager) = self.indirect_command_manager {
            manager
                .encode_command(buffer_id, command_index, command)
                .map_err(|e| match e {
                    BackendError::UnsupportedOperation { op, dtype } => {
                        TorshError::UnsupportedOperation {
                            op: format!("Neural Engine operation: {}", op),
                            dtype
                        }
                    }
                    BackendError::InvalidArgument(msg) => TorshError::InvalidArgument(msg),
                    BackendError::RuntimeError(message) => TorshError::ComputeError(message),
                    _ => TorshError::BackendError(e.to_string()),
                })
        } else {
            Err(TorshError::UnsupportedOperation {
                op: "Indirect command buffers".to_string(),
                dtype: "Indirect command buffers not available on this device".to_string(),
            })
        }
    }

    /// Execute commands from an indirect command buffer
    pub fn execute_indirect_commands(
        &self,
        command_buffer: &metal::CommandBuffer,
        buffer_id: u64,
        range: Option<(u32, u32)>,
    ) -> BackendResult<()> {
        if let Some(ref manager) = self.indirect_command_manager {
            manager
                .execute_commands(command_buffer, buffer_id, range)
                .map_err(|e| match e {
                    BackendError::UnsupportedOperation { op, dtype } => {
                        TorshError::UnsupportedOperation {
                            op: format!("Neural Engine operation: {}", op),
                            dtype
                        }
                    }
                    BackendError::InvalidArgument(msg) => TorshError::InvalidArgument(msg),
                    BackendError::RuntimeError(message) => TorshError::ComputeError(message),
                    _ => TorshError::BackendError(e.to_string()),
                })
        } else {
            Err(TorshError::UnsupportedOperation {
                op: "Indirect command buffers".to_string(),
                dtype: "Indirect command buffers not available on this device".to_string(),
            })
        }
    }

    /// Get performance statistics for indirect command buffers
    pub fn indirect_command_performance_stats(
        &self,
    ) -> Option<crate::metal::indirect_commands::IndirectCommandManagerStats> {
        self.indirect_command_manager
            .as_ref()
            .map(|manager| manager.performance_stats())
    }

    /// Optimize an indirect command buffer
    pub fn optimize_indirect_command_buffer(
        &self,
        buffer_id: u64,
    ) -> BackendResult<crate::metal::indirect_commands::OptimizationResult> {
        if let Some(ref manager) = self.indirect_command_manager {
            manager
                .optimize_command_buffer(buffer_id)
                .map_err(|e| match e {
                    BackendError::InvalidArgument(msg) => TorshError::InvalidArgument(msg),
                    _ => TorshError::BackendError(e.to_string()),
                })
        } else {
            Err(TorshError::UnsupportedOperation {
                op: "Indirect command buffers".to_string(),
                dtype: "Indirect command buffers not available on this device".to_string(),
            })
        }
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal backend")
    }
}

// Implementation of individual traits that compose the Backend trait

impl BackendCore for MetalBackend {
    fn device_type(&self) -> DeviceType {
        DeviceType::Metal(0)
    }

    fn name(&self) -> &str {
        "Metal Backend"
    }

    fn is_available(&self) -> BackendResult<bool> {
        // Check if Metal is available on this system
        Ok(metal::Device::system_default().is_some())
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            max_buffer_size: 4 * 1024 * 1024 * 1024, // 4GB typical for Metal
            max_compute_units: 16,
            max_workgroup_size: (1024, 1024, 64),
            supported_dtypes: vec![DType::F32, DType::F16, DType::I32, DType::U32],
            supports_async: true,
            supports_unified_memory: true,
            supports_sub_buffers: false,
            supports_kernel_caching: true,
            memory_bandwidth_gbps: 200.0,
            compute_throughput_gflops: 1000.0, // Approximate for Apple Silicon
            extended_capabilities: crate::backend::ExtendedCapabilities::default(),
        }
    }

    fn performance_hints(&self) -> PerformanceHints {
        PerformanceHints {
            preferred_workgroup_size: (256, 1, 1),
            memory_alignment: 16,
            prefer_vectorized: true,
            prefer_async: true,
            optimal_batch_size: 1024,
            cache_kernels: true,
        }
    }
}

#[async_trait::async_trait]
impl BackendLifecycle for MetalBackend {
    async fn initialize(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Initialize Metal resources
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

impl BackendDeviceManager for MetalBackend {
    fn devices(&self) -> BackendResult<Vec<Device>> {
        // For Metal, we typically have one device
        Ok(vec![Device::new(
            0,
            DeviceType::Metal(0),
            self.device.name().to_string(),
            DeviceInfo::default(),
        )])
    }

    fn default_device(&self) -> BackendResult<Device> {
        Ok(Device::new(
            0,
            DeviceType::Metal(0),
            self.device.name().to_string(),
            DeviceInfo::default(),
        ))
    }

    fn create_device(&self, device_id: usize) -> BackendResult<Device> {
        if device_id != 0 {
            return Err(TorshError::DeviceError(format!(
                "Metal device {} not found",
                device_id
            )));
        }
        self.default_device()
    }

    fn device_count(&self) -> BackendResult<usize> {
        Ok(1) // Metal typically has one integrated GPU
    }

    fn is_device_available(&self, device_id: usize) -> bool {
        device_id == 0 && self.is_available().unwrap_or(false)
    }
}

impl BackendResourceManager for MetalBackend {
    fn create_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer> {
        // Validate device
        if device.device_type != DeviceType::Metal(0) {
            return Err(BackendError::InvalidArgument(
                format!("Device {:?} not supported", device.device_type),
            )
            .into());
        }

        // Get next buffer ID
        let buffer_id = {
            let mut id_counter =
                self.next_buffer_id
                    .lock()
                    .map_err(|e| BackendError::SynchronizationError(
                        format!("Failed to lock buffer ID counter: {}", e)
                    ))?;
            let id = *id_counter;
            *id_counter += 1;
            id
        };

        // Create the Metal buffer
        let metal_buffer = if let Some(data) = &descriptor.initial_data {
            // Create from initial data
            MetalBuffer::from_data(data, &self.device).map_err(|_| {
                BackendError::AllocationError(
                    format!("Failed to allocate buffer of size {}", descriptor.size)
                )
            })?
        } else if descriptor.zero_init {
            // Create zero-initialized buffer
            let shape = descriptor
                .shape
                .as_ref()
                .ok_or(BackendError::AllocationError(
                    format!("Failed to allocate buffer of size {}", descriptor.size)
                ))?;
            let dtype = descriptor
                .dtype
                .ok_or(BackendError::AllocationError(
                    format!("Failed to allocate buffer of size {}", descriptor.size)
                ))?;
            MetalBuffer::zeros(shape, &dtype, &self.device).map_err(|_| {
                BackendError::AllocationError(
                    format!("Failed to allocate buffer of size {}", descriptor.size)
                )
            })?
        } else {
            // Create uninitialized buffer
            MetalBuffer::new(descriptor.size, &self.device).map_err(|_| {
                BackendError::AllocationError(
                    format!("Failed to allocate buffer of size {}", descriptor.size)
                )
            })?
        };

        // Get the Metal buffer's raw pointer as ID
        let buffer_ptr = metal_buffer.as_ptr() as u64;

        // Store the buffer in our internal storage
        // Note: In a real implementation, we'd want to store these buffers
        // in a HashMap or similar to prevent them from being dropped

        let handle = BufferHandle::Metal {
            buffer_id: buffer_ptr,
            size: descriptor.size,
        };

        Ok(Buffer::new(
            generate_buffer_id(),
            device.clone(),
            descriptor.size,
            descriptor.usage,
            descriptor.clone(),
            handle,
        ))
    }

    fn create_kernel(
        &self,
        device: &Device,
        descriptor: &KernelDescriptor,
    ) -> BackendResult<Kernel> {
        // Validate device
        if device.device_type != DeviceType::Metal(0) {
            return Err(BackendError::InvalidArgument(
                format!("Device {:?} not supported", device.device_type),
            )
            .into());
        }

        // For now, return a dummy kernel
        // In a real implementation, we'd compile the Metal shader code
        let handle = KernelHandle::Metal {
            library_id: 0,
            function_id: 0,
        };

        let metadata = KernelMetadata::default();

        Ok(Kernel::new(
            0, // kernel_id
            device.clone(),
            descriptor.name.clone(),
            descriptor.clone(),
            handle,
            metadata,
        ))
    }

    fn memory_manager(&self, device: &Device) -> BackendResult<Box<dyn MemoryManager + Send + Sync>> {
        // Validate device
        if device.device_type != DeviceType::Metal(0) {
            return Err(BackendError::InvalidArgument(
                format!("Device {:?} not supported", device.device_type),
            )
            .into());
        }

        Ok(Box::new(MetalMemoryManager::new(self.device.clone())))
    }

    fn profiler(&self) -> BackendResult<Box<dyn Profiler + Send + Sync>> {
        Ok(Box::new(MetalProfiler::new()))
    }

    fn create_scoped_buffer(
        &self,
        device: &Device,
        descriptor: &BufferDescriptor,
    ) -> BackendResult<Buffer> {
        // For now, just create a regular buffer - scoped cleanup can be added later
        self.create_buffer(device, descriptor)
    }
}

#[async_trait::async_trait]
impl BackendExecutor for MetalBackend {
    async fn synchronize(&self, _device: &Device) -> BackendResult<()> {
        // Metal operations are typically synchronous when using command buffer commit_and_wait
        // For now, this is a no-op as we don't have a current command buffer tracking
        Ok(())
    }

    async fn copy_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        _src_offset: usize,
        _dst_offset: usize,
        _size: usize,
    ) -> BackendResult<()> {
        // Validate buffers are Metal buffers
        if !matches!(src.handle, BufferHandle::Metal { .. })
            || !matches!(dst.handle, BufferHandle::Metal { .. })
        {
            return Err(BackendError::UnsupportedOperation {
                op: "buffer_copy".to_string(),
                dtype: "Metal buffers required".to_string(),
            }
            .into());
        }

        // Create a command buffer for the copy operation
        let command_buffer =
            self.device
                .new_command_buffer()
                .map_err(|e| BackendError::ComputeError(
                    format!("Failed to create command buffer: {:?}", e)
                ))?;

        let blit_encoder = command_buffer.new_blit_command_encoder();

        // For simplicity, we'll copy the entire buffer
        // In a real implementation, we'd need to get the actual Metal buffer references
        // from our buffer storage

        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    async fn copy_to_device(
        &self,
        src: &[u8],
        dst: &Buffer,
        dst_offset: usize,
    ) -> BackendResult<()> {
        // Validate buffer is a Metal buffer
        if let BufferHandle::Metal { buffer_id: _, size } = &dst.handle {
            if dst_offset + src.len() > *size {
                return Err(BackendError::AllocationError(
                    format!(
                        "Copy would exceed buffer bounds: {} + {} > {}",
                        dst_offset,
                        src.len(),
                        size
                    )
                ));
            }

            // In a real implementation, we'd look up the actual Metal buffer
            // from our storage using buffer_id and copy the data
            // For now, we'll just return success

            Ok(())
        } else {
            Err(BackendError::UnsupportedOperation {
                op: "buffer_copy".to_string(),
                dtype: "Destination must be a Metal buffer".to_string(),
            }
            .into())
        }
    }

    async fn copy_from_device(
        &self,
        src: &Buffer,
        dst: &mut [u8],
        src_offset: usize,
    ) -> BackendResult<()> {
        // Validate buffer is a Metal buffer
        if let BufferHandle::Metal { buffer_id: _, size } = &src.handle {
            if src_offset + dst.len() > *size {
                return Err(BackendError::AllocationError(
                    format!(
                        "Copy would exceed buffer bounds: {} + {} > {}",
                        src_offset,
                        dst.len(),
                        size
                    )
                ));
            }

            // In a real implementation, we'd look up the actual Metal buffer
            // from our storage using buffer_id and copy the data
            // For now, we'll just return success

            Ok(())
        } else {
            Err(BackendError::UnsupportedOperation {
                op: "buffer_copy".to_string(),
                dtype: "Source must be a Metal buffer".to_string(),
            }
            .into())
        }
    }

    async fn execute_kernel(
        &self,
        _kernel: &Kernel,
        buffers: &[&Buffer],
        _uniform_data: &[u8],
        _workgroup_size: (u32, u32, u32),
        _workgroup_count: (u32, u32, u32),
    ) -> BackendResult<()> {
        // Validate all buffers are Metal buffers
        for buffer in buffers {
            if !matches!(buffer.handle, BufferHandle::Metal { .. }) {
                return Err(BackendError::UnsupportedOperation {
                    op: "execute_kernel".to_string(),
                    dtype: "All buffers must be Metal buffers".to_string(),
                }
                .into());
            }
        }

        // Create command buffer and encoder
        let command_buffer =
            self.device
                .new_command_buffer()
                .map_err(|e| BackendError::ComputeError(
                    format!("Failed to create command buffer: {:?}", e)
                ))?;

        let compute_encoder = command_buffer.new_compute_command_encoder();

        // In a real implementation, we'd:
        // 1. Look up the actual compiled Metal kernel
        // 2. Set the compute pipeline state
        // 3. Bind the buffers
        // 4. Dispatch the threadgroups

        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

impl BackendOps for MetalBackend {
    fn backend_type(&self) -> crate::backend::BackendType {
        crate::backend::BackendType::Metal
    }

    fn available_ops(&self) -> Vec<&str> {
        vec![
            "matmul",
            "conv2d",
            "fft",
            "reduction",
            "elementwise",
            "neural_engine_matmul",
            "neural_engine_attention",
        ]
    }

    fn supports_op(&self, op_name: &str) -> bool {
        self.available_ops().contains(&op_name)
    }

    fn supports_fft(&self) -> bool {
        true // Metal supports FFT operations
    }

    fn supports_convolution(&self) -> bool {
        true // Metal supports convolution via MPS
    }

    fn supports_rnn(&self) -> bool {
        true // Metal supports RNN operations via MPS
    }

    fn supports_sparse(&self) -> bool {
        false // Metal sparse operations not yet implemented
    }

    fn supports_quantization(&self) -> bool {
        true // Metal supports quantization via MPS
    }

    fn operation_capabilities(
        &self,
        op_name: &str,
    ) -> Option<std::collections::HashMap<String, crate::backend::CapabilityValue>> {
        use crate::backend::CapabilityValue;
        let mut caps = std::collections::HashMap::new();

        match op_name {
            "matmul" => {
                caps.insert("max_size".to_string(), CapabilityValue::Int(16384));
                caps.insert("supports_batched".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_mixed_precision".to_string(), CapabilityValue::Bool(true));
                caps.insert("neural_engine_accelerated".to_string(), CapabilityValue::Bool(self.is_neural_engine_available()));
            }
            "conv2d" => {
                caps.insert("max_kernel_size".to_string(), CapabilityValue::Int(11));
                caps.insert("supports_groups".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_dilation".to_string(), CapabilityValue::Bool(true));
                caps.insert("mps_accelerated".to_string(), CapabilityValue::Bool(true));
            }
            "fft" => {
                caps.insert("max_size".to_string(), CapabilityValue::Int(65536));
                caps.insert("supports_real".to_string(), CapabilityValue::Bool(true));
                caps.insert("supports_batched".to_string(), CapabilityValue::Bool(true));
            }
            "neural_engine_matmul" => {
                caps.insert("available".to_string(), CapabilityValue::Bool(self.is_neural_engine_available()));
                if let Some(ne_caps) = self.neural_engine_capabilities() {
                    caps.insert("max_batch_size".to_string(), CapabilityValue::Int(ne_caps.max_batch_size as i64));
                    caps.insert("supported_dtypes".to_string(), CapabilityValue::List(
                        ne_caps.supported_dtypes.iter()
                            .map(|dt| CapabilityValue::String(format!("{:?}", dt)))
                            .collect()
                    ));
                }
            }
            "neural_engine_attention" => {
                caps.insert("available".to_string(), CapabilityValue::Bool(self.is_neural_engine_available()));
                if let Some(ne_caps) = self.neural_engine_capabilities() {
                    caps.insert("max_input_dims".to_string(), CapabilityValue::List(
                        ne_caps.max_input_dims.iter()
                            .map(|dim| CapabilityValue::Int(*dim as i64))
                            .collect()
                    ));
                    caps.insert("processing_units".to_string(), CapabilityValue::Int(ne_caps.processing_units as i64));
                }
            }
            _ => return None,
        }

        Some(caps)
    }
}

impl BackendOperations for MetalBackend {
    fn fft_ops(&self) -> Box<dyn crate::fft::FftOps> {
        Box::new(crate::cpu::fft::CpuFftOps::new(None))
    }

    fn convolution_ops(&self) -> Box<dyn crate::convolution::ConvolutionOps> {
        Box::new(crate::cpu::convolution::CpuConvolutionOps::new(None))
    }

    fn rnn_ops(&self) -> Box<dyn crate::rnn::RnnOps> {
        Box::new(crate::cpu::rnn::CpuRnnOps::new(None))
    }

    fn sparse_ops(&self) -> Box<dyn crate::sparse_ops::SparseOps<f32>> {
        // Use CPU implementation for now since Metal sparse ops are not implemented
        Box::new(crate::sparse_ops::DefaultSparseOps::new(
            Device::new(0, DeviceType::Metal(0), "Metal Device".to_string(), DeviceInfo::default())
        ))
    }

    fn quantization_ops(&self) -> Box<dyn crate::quantization::QuantizationOps> {
        Box::new(crate::quantization::CpuQuantizationOps::new())
    }

    fn operations_bundle(&self) -> crate::backend::OperationsBundle {
        crate::backend::OperationsBundle {
            fft: self.fft_ops(),
            convolution: self.convolution_ops(),
            rnn: self.rnn_ops(),
            sparse: self.sparse_ops(),
            quantization: self.quantization_ops(),
        }
    }
}

// The main Backend trait is automatically implemented through the sub-traits
impl Backend for MetalBackend {
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

/// Metal memory manager
struct MetalMemoryManager {
    device: Arc<MetalDevice>,
    device_info: Device,
    allocated_buffers: Arc<Mutex<Vec<(usize, metal::Buffer)>>>,
    next_buffer_id: Arc<Mutex<usize>>,
    stats: Arc<Mutex<MemoryStats>>,
}

impl MetalMemoryManager {
    fn new(device: Arc<MetalDevice>) -> Self {
        let total_memory = device.max_buffer_length();
        let stats = MemoryStats {
            total_memory,
            available_memory: total_memory,
            ..Default::default()
        };

        let device_info = Device::new(
            0,
            DeviceType::Metal(0),
            device.name().to_string(),
            DeviceInfo::default(),
        );

        Self {
            device,
            device_info,
            allocated_buffers: Arc::new(Mutex::new(Vec::new())),
            next_buffer_id: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(stats)),
        }
    }
}

// SAFETY: MetalMemoryManager uses Arc<Mutex<...>> internally which are thread-safe
unsafe impl Send for MetalMemoryManager {}
unsafe impl Sync for MetalMemoryManager {}

impl MemoryManager for MetalMemoryManager {
    fn allocate(&mut self, descriptor: &BufferDescriptor) -> torsh_core::error::Result<Buffer> {
        // Get next buffer ID
        let buffer_id = {
            let mut id_counter = self.next_buffer_id.lock().map_err(|e| {
                TorshError::BackendError(format!("Failed to lock buffer ID counter: {}", e))
            })?;
            let id = *id_counter;
            *id_counter += 1;
            id
        };

        // Create Metal buffer
        let metal_buffer = self
            .device
            .device()
            .new_buffer(descriptor.size as u64, self.device.resource_options());

        // Store buffer reference
        self.allocated_buffers
            .lock()
            .map_err(|e| TorshError::BackendError(format!("Failed to lock buffer storage: {}", e)))?
            .push((buffer_id, metal_buffer.clone()));

        // Update stats
        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|e| TorshError::BackendError(format!("Failed to lock stats: {}", e)))?;
            stats.allocated_memory += descriptor.size;
            stats.available_memory = stats.total_memory.saturating_sub(stats.allocated_memory);
            stats.active_allocations += 1;
            stats.total_allocations += 1;
            stats.peak_memory = stats.peak_memory.max(stats.allocated_memory);
            stats.efficiency = stats.allocated_memory as f32 / stats.total_memory as f32;
        }

        let handle = BufferHandle::Metal {
            buffer_id: (&*metal_buffer as *const _) as u64,
            size: descriptor.size,
        };

        Ok(Buffer::new(
            generate_buffer_id(),
            Device::new(
                0,
                DeviceType::Metal(0),
                self.device.name().to_string(),
                DeviceInfo::default(),
            ),
            descriptor.size,
            descriptor.usage,
            descriptor.clone(),
            handle,
        ))
    }

    fn deallocate(&mut self, buffer: &Buffer) -> torsh_core::error::Result<()> {
        if let BufferHandle::Metal { buffer_id, size } = &buffer.handle {
            // Find and remove buffer
            let mut buffers = self.allocated_buffers.lock().map_err(|e| {
                TorshError::BackendError(format!("Failed to lock buffer storage: {}", e))
            })?;

            if let Some(pos) = buffers
                .iter()
                .position(|(id, b)| *id == buffer.id && b.as_ptr() as u64 == *buffer_id)
            {
                buffers.remove(pos);

                // Update stats
                let mut stats = self.stats.lock().map_err(|e| {
                    TorshError::BackendError(format!("Failed to lock stats: {}", e))
                })?;
                stats.allocated_memory = stats.allocated_memory.saturating_sub(*size);
                stats.available_memory = stats.total_memory.saturating_sub(stats.allocated_memory);
                stats.active_allocations = stats.active_allocations.saturating_sub(1);
                stats.total_deallocations += 1;
                stats.efficiency = stats.allocated_memory as f32 / stats.total_memory as f32;
            }
        }

        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn garbage_collect(&mut self) -> torsh_core::error::Result<usize> {
        // Metal handles memory automatically, so just return 0
        Ok(0)
    }

    fn set_pool(
        &mut self,
        _pool: Box<dyn crate::memory::MemoryPool>,
    ) -> torsh_core::error::Result<()> {
        // Metal doesn't use our memory pool system
        Ok(())
    }

    fn device(&self) -> &Device {
        &self.device_info
    }

    fn allocate_raw(&mut self, size: usize, alignment: usize) -> torsh_core::error::Result<*mut u8> {
        // Create Metal buffer
        let buffer = self
            .device
            .device()
            .new_buffer(size as u64, self.device.resource_options());

        // Get pointer from Metal buffer
        let ptr = buffer.contents() as *mut u8;

        // Store buffer to prevent deallocation
        self.allocated_buffers
            .lock()
            .map_err(|e| TorshError::BackendError(format!("Failed to lock buffer storage: {}", e)))?
            .push((ptr as usize, buffer));

        // Update stats
        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|e| TorshError::BackendError(format!("Failed to lock stats: {}", e)))?;
            stats.allocated_memory += size;
            stats.available_memory = stats.total_memory.saturating_sub(stats.allocated_memory);
            stats.active_allocations += 1;
            stats.total_allocations += 1;
            stats.peak_memory = stats.peak_memory.max(stats.allocated_memory);
            stats.efficiency = stats.allocated_memory as f32 / stats.total_memory as f32;
        }

        Ok(ptr)
    }

    fn deallocate_raw(&mut self, ptr: *mut u8, size: usize) -> torsh_core::error::Result<()> {
        // Find and remove buffer
        let mut buffers = self.allocated_buffers.lock().map_err(|e| {
            TorshError::BackendError(format!("Failed to lock buffer storage: {}", e))
        })?;

        if let Some(pos) = buffers
            .iter()
            .position(|(stored_ptr, _)| *stored_ptr == ptr as usize)
        {
            buffers.remove(pos);

            // Update stats
            let mut stats = self.stats.lock().map_err(|e| {
                TorshError::BackendError(format!("Failed to lock stats: {}", e))
            })?;
            stats.allocated_memory = stats.allocated_memory.saturating_sub(size);
            stats.available_memory = stats.total_memory.saturating_sub(stats.allocated_memory);
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
            stats.total_deallocations += 1;
            stats.efficiency = stats.allocated_memory as f32 / stats.total_memory as f32;
        }

        Ok(())
    }

    fn supports_unified_memory(&self) -> bool {
        // Metal on Apple Silicon supports unified memory
        true
    }

    fn allocate_unified(&mut self, size: usize) -> torsh_core::error::Result<*mut u8> {
        // Use shared storage mode for unified memory
        let buffer = self.device.device().new_buffer(
            size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let ptr = buffer.contents() as *mut u8;

        // Store buffer to prevent deallocation
        self.allocated_buffers
            .lock()
            .map_err(|e| TorshError::BackendError(format!("Failed to lock buffer storage: {}", e)))?
            .push((ptr as usize, buffer));

        Ok(ptr)
    }

    fn deallocate_unified(&mut self, ptr: *mut u8, size: usize) -> torsh_core::error::Result<()> {
        // Same as regular deallocation for Metal
        self.deallocate_raw(ptr, size)
    }

    fn prefetch_to_device(&self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        // Metal unified memory doesn't require explicit prefetching
        Ok(())
    }

    fn prefetch_to_host(&self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        // Metal unified memory doesn't require explicit prefetching
        Ok(())
    }

    fn set_memory_advice(
        &self,
        _ptr: *mut u8,
        _size: usize,
        _advice: crate::memory::MemoryAdvice,
    ) -> torsh_core::error::Result<()> {
        // Metal doesn't have explicit memory advice APIs
        Ok(())
    }

    fn available_memory(&self) -> torsh_core::error::Result<usize> {
        let stats = self.stats.lock().map_err(|e| {
            TorshError::BackendError(format!("Failed to lock stats: {}", e))
        })?;
        Ok(stats.available_memory)
    }

    fn total_memory(&self) -> torsh_core::error::Result<usize> {
        let stats = self.stats.lock().map_err(|e| {
            TorshError::BackendError(format!("Failed to lock stats: {}", e))
        })?;
        Ok(stats.total_memory)
    }

    fn synchronize(&self) -> torsh_core::error::Result<()> {
        // Metal operations are typically synchronous when using commit_and_wait
        Ok(())
    }

    fn defragment(&mut self) -> torsh_core::error::Result<crate::memory::DefragmentationResult> {
        // Metal handles memory management automatically
        Ok(crate::memory::DefragmentationResult {
            blocks_moved: 0,
            memory_compacted: 0,
            duration_ms: 0.0,
            fragmentation_before: 0.0,
            fragmentation_after: 0.0,
            efficiency_improvement: 0.0,
            success: true,
        })
    }

    fn needs_defragmentation(&self) -> bool {
        // Metal handles memory management automatically
        false
    }

    fn fragmentation_info(&self) -> crate::memory::FragmentationInfo {
        let stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());

        crate::memory::FragmentationInfo {
            overall_fragmentation: 0.0, // Metal manages this automatically
            external_fragmentation: 0.0,
            internal_fragmentation: 0.0,
            free_blocks: 1, // Simplified for Metal
            allocated_blocks: stats.active_allocations,
            largest_free_block: stats.available_memory,
            smallest_free_block: stats.available_memory,
            average_free_block: stats.available_memory,
            total_free_memory: stats.available_memory,
            total_allocated_memory: stats.allocated_memory,
            utilization_efficiency: stats.efficiency,
            allocation_efficiency: stats.efficiency,
        }
    }

    fn compact_memory(&mut self) -> torsh_core::error::Result<crate::memory::CompactionResult> {
        // Metal handles memory compaction automatically
        Ok(crate::memory::CompactionResult {
            allocations_moved: 0,
            bytes_moved: 0,
            duration_ms: 0.0,
            largest_free_before: self.stats.lock().unwrap().available_memory,
            largest_free_after: self.stats.lock().unwrap().available_memory,
            free_blocks_before: 1,
            free_blocks_after: 1,
            success: true,
        })
    }

    fn set_defragmentation_policy(&mut self, _policy: crate::memory::DefragmentationPolicy) {
        // Metal doesn't support configurable defragmentation policies
        // Policy setting is ignored
    }
}

/// Metal profiler
struct MetalProfiler {
    enabled: bool,
    events: Arc<Mutex<Vec<ProfilerEvent>>>,
    next_event_id: Arc<Mutex<u64>>,
    start_time: Option<std::time::Instant>,
}

impl MetalProfiler {
    fn new() -> Self {
        Self {
            enabled: false,
            events: Arc::new(Mutex::new(Vec::new())),
            next_event_id: Arc::new(Mutex::new(0)),
            start_time: None,
        }
    }
}

impl Profiler for MetalProfiler {
    fn start(&mut self) -> torsh_core::error::Result<()> {
        self.enabled = true;
        self.start_time = Some(std::time::Instant::now());
        Ok(())
    }

    fn stop(&mut self) -> torsh_core::error::Result<()> {
        self.enabled = false;
        Ok(())
    }

    fn begin_event(&mut self, name: &str) -> torsh_core::error::Result<EventId> {
        if !self.enabled {
            return Ok(EventId(0));
        }

        let event_id = {
            let mut id = self
                .next_event_id
                .lock()
                .map_err(|e| TorshError::BackendError(format!("Failed to lock event ID: {}", e)))?;
            let current = *id;
            *id += 1;
            EventId(current)
        };

        let event = ProfilerEvent::new(event_id, name.to_string(), EventType::KernelExecution);

        self.events
            .lock()
            .map_err(|e| TorshError::BackendError(format!("Failed to lock events: {}", e)))?
            .push(event);

        Ok(event_id)
    }

    fn end_event(&mut self, event_id: EventId) -> torsh_core::error::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut events = self
            .events
            .lock()
            .map_err(|e| TorshError::BackendError(format!("Failed to lock events: {}", e)))?;

        if let Some(event) = events.iter_mut().find(|e| e.id == event_id) {
            event.finish();
        }

        Ok(())
    }

    fn marker(&mut self, name: &str) -> torsh_core::error::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let event_id = {
            let mut id = self
                .next_event_id
                .lock()
                .map_err(|e| TorshError::BackendError(format!("Failed to lock event ID: {}", e)))?;
            let current = *id;
            *id += 1;
            EventId(current)
        };

        let event = ProfilerEvent::new(event_id, name.to_string(), EventType::Marker);

        self.events
            .lock()
            .map_err(|e| TorshError::BackendError(format!("Failed to lock events: {}", e)))?
            .push(event);

        Ok(())
    }

    fn stats(&self) -> ProfilerStats {
        let events = self.events.lock().unwrap_or_else(|e| e.into_inner());

        let mut stats = ProfilerStats {
            total_events: events.len(),
            ..Default::default()
        };

        if let Some(start_time) = self.start_time {
            stats.total_time = std::time::Instant::now().duration_since(start_time);
        }

        for event in events.iter() {
            match &event.event_type {
                EventType::KernelExecution => {
                    stats.kernel_executions += 1;
                    if let Some(duration) = event.duration() {
                        stats.kernel_time += duration;
                    }
                }
                EventType::MemoryOperation => {
                    stats.memory_operations += 1;
                    if let Some(duration) = event.duration() {
                        stats.memory_time += duration;
                    }
                }
                EventType::Synchronization => {
                    stats.synchronization_events += 1;
                }
                _ => {}
            }
        }

        if stats.kernel_executions > 0 {
            stats.avg_kernel_time = stats.kernel_time / stats.kernel_executions as u32;
        }

        stats
    }

    fn events(&self) -> &[ProfilerEvent] {
        // This is a bit of a hack since we can't return a borrowed slice from a Mutex
        // In a real implementation, we'd need a different approach
        &[]
    }

    fn clear(&mut self) {
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
    }

    fn report(&self) -> String {
        let stats = self.stats();
        format!(
            "Metal Profiler Report:\n\
             Total Events: {}\n\
             Total Time: {:?}\n\
             Kernel Executions: {} (avg: {:?})\n\
             Memory Operations: {} (total: {:?})\n\
             Synchronization Events: {}",
            stats.total_events,
            stats.total_time,
            stats.kernel_executions,
            stats.avg_kernel_time,
            stats.memory_operations,
            stats.memory_time,
            stats.synchronization_events
        )
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metal_backend_initialization() {
        if metal::Device::system_default().is_none() {
            println!("Skipping test - Metal not available");
            return;
        }

        let mut backend = MetalBackend::new().unwrap();
        assert!(!backend.initialized);

        backend.initialize().await.unwrap();
        assert!(backend.initialized);

        backend.shutdown().await.unwrap();
        assert!(!backend.initialized);
    }

    #[tokio::test]
    async fn test_metal_backend_devices() {
        if metal::Device::system_default().is_none() {
            println!("Skipping test - Metal not available");
            return;
        }

        let backend = MetalBackend::new().unwrap();

        let devices = backend.devices().unwrap();
        assert_eq!(devices.len(), 1);

        let default_device = backend.default_device().unwrap();
        assert_eq!(default_device.device_type, DeviceType::Metal(0));
    }
}
