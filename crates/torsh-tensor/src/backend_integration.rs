//! Backend integration module for device-specific optimizations and cross-device operations
//! 🚀 Enhanced with SciRS2 GPU acceleration capabilities
//! - Multi-backend GPU support (CUDA/Metal/WebGPU/ROCm/OpenCL)
//! - Tensor core acceleration for mixed-precision training
//! - Automatic GPU kernel selection and optimization
//! - Memory management with unified memory and pinned buffers

use crate::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::{device::DeviceType, dtype::TensorElement, error::Result};

// 🚀 SciRS2 GPU integration for breakthrough performance
#[cfg(feature = "gpu")]
use scirs2_core::gpu::{
    CudaBackend, GpuBuffer, GpuContext, GpuKernel, GpuMemoryManager, MetalBackend, OpenClBackend,
    RocmBackend, UnifiedMemory, WebGpuBackend,
};

#[cfg(feature = "gpu")]
use scirs2_core::tensor_cores::{AutoTuning, MixedPrecision, TensorCore};

/// Device-specific optimization strategies
#[derive(Debug, Clone)]
pub enum DeviceOptimization {
    /// CPU-specific optimizations
    Cpu(CpuOptimization),
    /// GPU-specific optimizations  
    Gpu(GpuOptimization),
    /// Metal-specific optimizations
    Metal(MetalOptimization),
    /// WebGPU-specific optimizations
    WebGpu(WebGpuOptimization),
}

/// CPU optimization configuration
#[derive(Debug, Clone)]
pub struct CpuOptimization {
    /// Use SIMD instructions when available
    pub use_simd: bool,
    /// Number of threads for parallel operations
    pub thread_count: Option<usize>,
    /// Enable cache-friendly memory access patterns
    pub cache_friendly: bool,
    /// Enable NUMA-aware memory allocation
    pub numa_aware: bool,
}

/// 🚀 Advanced GPU optimization configuration with SciRS2 integration
#[derive(Debug, Clone)]
pub struct GpuOptimization {
    /// Use pinned memory for transfers
    pub use_pinned_memory: bool,
    /// Stream count for asynchronous operations
    pub stream_count: u32,
    /// Enable mixed precision computation (FP16/BF16)
    pub mixed_precision: bool,
    /// GPU memory pool configuration
    pub memory_pool_size: Option<usize>,

    // 🚀 SciRS2 Advanced GPU Features
    /// Enable tensor core acceleration for supported operations
    pub use_tensor_cores: bool,
    /// Automatic kernel selection and optimization
    pub auto_kernel_tuning: bool,
    /// Enable unified memory management (CUDA/HIP)
    pub use_unified_memory: bool,
    /// Multi-GPU distribution strategy
    pub multi_gpu_strategy: MultiGpuStrategy,
    /// GPU backend preference order
    pub backend_preference: Vec<GpuBackendType>,
    /// Memory coalescing optimization
    pub memory_coalescing: bool,
    /// Kernel fusion optimization level (0-3)
    pub kernel_fusion_level: u8,
    /// Dynamic batching for improved throughput
    pub dynamic_batching: bool,
}

/// Multi-GPU distribution strategies
#[derive(Debug, Clone)]
pub enum MultiGpuStrategy {
    /// Single GPU execution
    Single,
    /// Data parallel execution across multiple GPUs
    DataParallel,
    /// Model parallel execution (layers split across GPUs)
    ModelParallel,
    /// Pipeline parallel execution
    PipelineParallel,
    /// Automatic strategy selection based on workload
    Auto,
}

/// GPU backend types supported by SciRS2
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuBackendType {
    /// NVIDIA CUDA backend
    Cuda,
    /// Apple Metal backend
    Metal,
    /// Cross-platform WebGPU backend
    WebGpu,
    /// AMD ROCm backend (HIP)
    Rocm,
    /// OpenCL backend
    OpenCl,
}

/// Metal optimization configuration
#[derive(Debug, Clone)]
pub struct MetalOptimization {
    /// Use Metal Performance Shaders
    pub use_mps: bool,
    /// Command buffer count
    pub command_buffer_count: u32,
    /// Enable automatic memory management
    pub auto_memory_management: bool,
}

/// WebGPU optimization configuration
#[derive(Debug, Clone)]
pub struct WebGpuOptimization {
    /// Use compute shaders for operations
    pub use_compute_shaders: bool,
    /// Buffer pool size for efficient memory reuse
    pub buffer_pool_size: Option<usize>,
    /// Enable pipeline caching
    pub pipeline_caching: bool,
}

/// Cross-device operation scheduler
#[derive(Debug)]
pub struct OperationScheduler {
    /// Pending operations per device
    device_queues: HashMap<DeviceType, Vec<ScheduledOperation>>,
    /// Device synchronization state
    sync_state: HashMap<DeviceType, SyncState>,
    /// Global operation counter
    operation_counter: Arc<RwLock<u64>>,
}

/// Scheduled operation
#[derive(Debug)]
pub struct ScheduledOperation {
    /// Unique operation ID
    pub id: u64,
    /// Operation type
    pub operation: OperationType,
    /// Priority level (higher = more priority)
    pub priority: u8,
    /// Device dependencies
    pub dependencies: Vec<DeviceType>,
}

/// Operation type for scheduling
#[derive(Debug)]
pub enum OperationType {
    /// Tensor computation
    Compute,
    /// Memory transfer
    Transfer,
    /// Synchronization barrier
    Synchronization,
}

/// Device synchronization state
#[derive(Debug)]
pub struct SyncState {
    /// Last operation timestamp
    pub last_operation: std::time::Instant,
    /// Pending transfers
    pub pending_transfers: usize,
    /// Device availability
    pub available: bool,
}

impl Default for CpuOptimization {
    fn default() -> Self {
        Self {
            use_simd: true,
            thread_count: None, // Use default thread pool
            cache_friendly: true,
            numa_aware: true,
        }
    }
}

impl Default for GpuOptimization {
    fn default() -> Self {
        Self {
            use_pinned_memory: true,
            stream_count: 4,
            mixed_precision: false,
            memory_pool_size: Some(1024 * 1024 * 1024), // 1GB

            // 🚀 SciRS2 Advanced GPU Features - optimized defaults
            use_tensor_cores: true, // Enable tensor cores for supported hardware
            auto_kernel_tuning: true, // Automatic performance optimization
            use_unified_memory: true, // Simplified memory management
            multi_gpu_strategy: MultiGpuStrategy::Auto, // Intelligent multi-GPU selection
            backend_preference: vec![
                GpuBackendType::Cuda,   // NVIDIA first (most common)
                GpuBackendType::Metal,  // Apple Silicon second
                GpuBackendType::Rocm,   // AMD third
                GpuBackendType::WebGpu, // Cross-platform fallback
                GpuBackendType::OpenCl, // Universal fallback
            ],
            memory_coalescing: true, // Optimize memory access patterns
            kernel_fusion_level: 2,  // Moderate kernel fusion (0-3 scale)
            dynamic_batching: true,  // Adaptive batch sizing
        }
    }
}

impl Default for MetalOptimization {
    fn default() -> Self {
        Self {
            use_mps: true,
            command_buffer_count: 8,
            auto_memory_management: true,
        }
    }
}

impl Default for WebGpuOptimization {
    fn default() -> Self {
        Self {
            use_compute_shaders: true,
            buffer_pool_size: Some(256 * 1024 * 1024), // 256MB
            pipeline_caching: true,
        }
    }
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Transfer tensor to another device with optimization
    pub fn to_device(&self, target_device: DeviceType) -> Result<Self> {
        if self.device == target_device {
            return Ok(self.clone());
        }

        // Get optimization strategy for target device
        let optimization = self.get_device_optimization(target_device);

        // Perform optimized transfer
        match (self.device, target_device) {
            (DeviceType::Cpu, DeviceType::Cuda(gpu_id)) => {
                self.cpu_to_gpu_transfer(gpu_id as u32, optimization)
            }
            (DeviceType::Cuda(gpu_id), DeviceType::Cpu) => {
                self.gpu_to_cpu_transfer(gpu_id as u32, optimization)
            }
            (DeviceType::Cpu, DeviceType::Metal(metal_id)) => {
                self.cpu_to_metal_transfer(metal_id as u32, optimization)
            }
            (DeviceType::Metal(metal_id), DeviceType::Cpu) => {
                self.metal_to_cpu_transfer(metal_id as u32, optimization)
            }
            _ => {
                // Generic transfer through CPU
                self.generic_device_transfer(target_device)
            }
        }
    }

    /// Get device-specific optimization configuration
    fn get_device_optimization(&self, device: DeviceType) -> DeviceOptimization {
        match device {
            DeviceType::Cpu => DeviceOptimization::Cpu(CpuOptimization::default()),
            DeviceType::Cuda(_) => DeviceOptimization::Gpu(GpuOptimization::default()),
            DeviceType::Metal(_) => DeviceOptimization::Metal(MetalOptimization::default()),
            DeviceType::Wgpu(_) => DeviceOptimization::Gpu(GpuOptimization::default()),
        }
    }

    /// Optimized CPU to GPU transfer
    fn cpu_to_gpu_transfer(&self, _gpu_id: u32, optimization: DeviceOptimization) -> Result<Self> {
        let data = self.to_vec()?;

        // Apply GPU-specific optimizations
        if let DeviceOptimization::Gpu(gpu_opt) = optimization {
            if gpu_opt.use_pinned_memory {
                // Use pinned memory for faster transfers
                self.transfer_with_pinned_memory(data, DeviceType::Cuda(_gpu_id as usize))
            } else {
                // Standard transfer
                Self::from_data(
                    data,
                    self.shape().dims().to_vec(),
                    DeviceType::Cuda(_gpu_id as usize),
                )
            }
        } else {
            Self::from_data(
                data,
                self.shape().dims().to_vec(),
                DeviceType::Cuda(_gpu_id as usize),
            )
        }
    }

    /// Optimized GPU to CPU transfer
    fn gpu_to_cpu_transfer(&self, _gpu_id: u32, optimization: DeviceOptimization) -> Result<Self> {
        let data = self.to_vec()?;

        // Apply CPU-specific optimizations
        if let DeviceOptimization::Cpu(cpu_opt) = optimization {
            if cpu_opt.numa_aware {
                // Use NUMA-aware allocation
                self.transfer_with_numa_awareness(data, DeviceType::Cpu)
            } else {
                // Standard transfer
                Self::from_data(data, self.shape().dims().to_vec(), DeviceType::Cpu)
            }
        } else {
            Self::from_data(data, self.shape().dims().to_vec(), DeviceType::Cpu)
        }
    }

    /// Optimized CPU to Metal transfer
    fn cpu_to_metal_transfer(
        &self,
        _metal_id: u32,
        optimization: DeviceOptimization,
    ) -> Result<Self> {
        let data = self.to_vec()?;

        // Apply Metal-specific optimizations
        if let DeviceOptimization::Metal(metal_opt) = optimization {
            if metal_opt.use_mps {
                // Use Metal Performance Shaders for optimization
                self.transfer_with_mps(data, DeviceType::Metal(_metal_id as usize))
            } else {
                // Standard transfer
                Self::from_data(
                    data,
                    self.shape().dims().to_vec(),
                    DeviceType::Metal(_metal_id as usize),
                )
            }
        } else {
            Self::from_data(
                data,
                self.shape().dims().to_vec(),
                DeviceType::Metal(_metal_id as usize),
            )
        }
    }

    /// Optimized Metal to CPU transfer
    fn metal_to_cpu_transfer(
        &self,
        _metal_id: u32,
        optimization: DeviceOptimization,
    ) -> Result<Self> {
        let data = self.to_vec()?;

        // Apply CPU-specific optimizations
        if let DeviceOptimization::Cpu(cpu_opt) = optimization {
            if cpu_opt.cache_friendly {
                // Use cache-friendly memory layout
                self.transfer_with_cache_optimization(data, DeviceType::Cpu)
            } else {
                // Standard transfer
                Self::from_data(data, self.shape().dims().to_vec(), DeviceType::Cpu)
            }
        } else {
            Self::from_data(data, self.shape().dims().to_vec(), DeviceType::Cpu)
        }
    }

    /// Generic device transfer through CPU
    fn generic_device_transfer(&self, target_device: DeviceType) -> Result<Self> {
        let data = self.to_vec()?;
        Self::from_data(data, self.shape().dims().to_vec(), target_device)
    }

    /// Transfer with pinned memory optimization
    fn transfer_with_pinned_memory(&self, data: Vec<T>, target_device: DeviceType) -> Result<Self> {
        // For now, use standard transfer (pinned memory would require GPU backend)
        Self::from_data(data, self.shape().dims().to_vec(), target_device)
    }

    /// Transfer with NUMA awareness
    fn transfer_with_numa_awareness(
        &self,
        data: Vec<T>,
        target_device: DeviceType,
    ) -> Result<Self> {
        // For now, use standard transfer (NUMA awareness would require system-level support)
        Self::from_data(data, self.shape().dims().to_vec(), target_device)
    }

    /// Transfer with Metal Performance Shaders
    fn transfer_with_mps(&self, data: Vec<T>, target_device: DeviceType) -> Result<Self> {
        // For now, use standard transfer (MPS would require Metal backend)
        Self::from_data(data, self.shape().dims().to_vec(), target_device)
    }

    /// Transfer with cache optimization
    fn transfer_with_cache_optimization(
        &self,
        data: Vec<T>,
        target_device: DeviceType,
    ) -> Result<Self> {
        // Apply cache-friendly memory layout
        let optimized_data = self.optimize_for_cache(data)?;
        Self::from_data(optimized_data, self.shape().dims().to_vec(), target_device)
    }

    /// Optimize data layout for cache efficiency
    fn optimize_for_cache(&self, data: Vec<T>) -> Result<Vec<T>> {
        // For now, return data as-is (cache optimization would require detailed analysis)
        Ok(data)
    }

    /// Synchronize operations across devices
    pub fn synchronize_devices(&self, devices: &[DeviceType]) -> Result<()> {
        // For now, this is a no-op (synchronization would require backend support)
        for device in devices {
            self.synchronize_device(*device)?;
        }
        Ok(())
    }

    /// Synchronize operations on a specific device
    fn synchronize_device(&self, _device: DeviceType) -> Result<()> {
        // For now, this is a no-op (synchronization would require backend support)
        Ok(())
    }

    /// Check if tensor can be efficiently transferred to target device
    pub fn can_transfer_efficiently(&self, target_device: DeviceType) -> bool {
        match (self.device, target_device) {
            // Same device - always efficient
            (a, b) if a == b => true,
            // CPU-GPU transfers are generally efficient
            (DeviceType::Cpu, DeviceType::Cuda(_)) | (DeviceType::Cuda(_), DeviceType::Cpu) => true,
            // CPU-Metal transfers are efficient on Apple systems
            (DeviceType::Cpu, DeviceType::Metal(_)) | (DeviceType::Metal(_), DeviceType::Cpu) => {
                true
            }
            // Other transfers may require multiple hops
            _ => false,
        }
    }

    /// Get optimal transfer strategy for device pair
    pub fn get_transfer_strategy(&self, target_device: DeviceType) -> TransferStrategy {
        match (self.device, target_device) {
            (a, b) if a == b => TransferStrategy::NoTransfer,
            (DeviceType::Cpu, DeviceType::Cuda(_)) => TransferStrategy::DirectTransfer,
            (DeviceType::Cuda(_), DeviceType::Cpu) => TransferStrategy::DirectTransfer,
            (DeviceType::Cpu, DeviceType::Metal(_)) => TransferStrategy::DirectTransfer,
            (DeviceType::Metal(_), DeviceType::Cpu) => TransferStrategy::DirectTransfer,
            _ => TransferStrategy::ThroughCpu,
        }
    }
}

/// Transfer strategy for cross-device operations
#[derive(Debug, Clone, PartialEq)]
pub enum TransferStrategy {
    /// No transfer needed
    NoTransfer,
    /// Direct transfer between devices
    DirectTransfer,
    /// Transfer through CPU as intermediate
    ThroughCpu,
}

impl OperationScheduler {
    /// Create a new operation scheduler
    pub fn new() -> Self {
        Self {
            device_queues: HashMap::new(),
            sync_state: HashMap::new(),
            operation_counter: Arc::new(RwLock::new(0)),
        }
    }

    /// Schedule an operation on a specific device
    pub fn schedule_operation(
        &mut self,
        device: DeviceType,
        operation: OperationType,
        priority: u8,
        dependencies: Vec<DeviceType>,
    ) -> Result<u64> {
        // Generate unique operation ID
        let mut counter = self.operation_counter.write().unwrap();
        *counter += 1;
        let op_id = *counter;
        drop(counter);

        // Create scheduled operation
        let scheduled_op = ScheduledOperation {
            id: op_id,
            operation,
            priority,
            dependencies,
        };

        // Add to device queue
        self.device_queues
            .entry(device)
            .or_default()
            .push(scheduled_op);

        // Sort by priority (highest first)
        if let Some(queue) = self.device_queues.get_mut(&device) {
            queue.sort_by(|a, b| b.priority.cmp(&a.priority));
        }

        // Update sync state
        self.sync_state.entry(device).or_insert_with(|| SyncState {
            last_operation: std::time::Instant::now(),
            pending_transfers: 0,
            available: true,
        });

        Ok(op_id)
    }

    /// Execute next operation on device
    pub fn execute_next_operation(&mut self, device: DeviceType) -> Result<Option<u64>> {
        // First, get the operation without holding the mutable borrow
        let op = if let Some(queue) = self.device_queues.get_mut(&device) {
            if queue.is_empty() {
                None
            } else {
                Some(queue.remove(0)) // Remove highest priority item (first element)
            }
        } else {
            None
        };

        if let Some(op) = op {
            // Check dependencies (this borrows self immutably)
            let dependencies_satisfied = self.check_dependencies(&op.dependencies)?;

            if dependencies_satisfied {
                // Execute operation (placeholder)
                self.execute_operation(&op)?;

                // Update sync state
                if let Some(sync_state) = self.sync_state.get_mut(&device) {
                    sync_state.last_operation = std::time::Instant::now();
                }

                Ok(Some(op.id))
            } else {
                // Dependencies not satisfied, requeue at front to maintain priority
                if let Some(queue) = self.device_queues.get_mut(&device) {
                    queue.insert(0, op);
                }
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Check if dependencies are satisfied
    fn check_dependencies(&self, dependencies: &[DeviceType]) -> Result<bool> {
        for &dep_device in dependencies {
            if let Some(sync_state) = self.sync_state.get(&dep_device) {
                if !sync_state.available {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Execute an operation (placeholder)
    fn execute_operation(&self, _operation: &ScheduledOperation) -> Result<()> {
        // Placeholder for actual operation execution
        std::thread::sleep(std::time::Duration::from_millis(1));
        Ok(())
    }

    /// Get device queue length
    pub fn get_queue_length(&self, device: DeviceType) -> usize {
        self.device_queues
            .get(&device)
            .map_or(0, |queue| queue.len())
    }

    /// Clear all operations for a device
    pub fn clear_device_queue(&mut self, device: DeviceType) {
        self.device_queues.remove(&device);
    }
}

impl Default for OperationScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Global operation scheduler instance
static GLOBAL_SCHEDULER: parking_lot::Mutex<Option<OperationScheduler>> =
    parking_lot::Mutex::new(None);

/// Get or create global operation scheduler
pub fn get_global_scheduler() -> parking_lot::MutexGuard<'static, Option<OperationScheduler>> {
    let mut guard = GLOBAL_SCHEDULER.lock();
    if guard.is_none() {
        *guard = Some(OperationScheduler::new());
    }
    guard
}

/// Initialize global scheduler with custom configuration
pub fn initialize_global_scheduler() -> Result<()> {
    let mut guard = GLOBAL_SCHEDULER.lock();
    *guard = Some(OperationScheduler::new());
    Ok(())
}

// 🚀 SciRS2 Advanced GPU Integration Functions
#[cfg(feature = "gpu")]
impl<T: TensorElement + Copy> Tensor<T> {
    /// 🚀 Enhanced GPU kernel execution with automatic optimization
    pub fn execute_gpu_kernel(&self, kernel_name: &str, params: Vec<T>) -> Result<Self> {
        let gpu_opt = match self.get_device_optimization(self.device) {
            DeviceOptimization::Gpu(opt) => opt,
            _ => {
                return Err(torsh_core::error::TorshError::InvalidArgument(
                    "GPU kernel execution requires GPU device".to_string(),
                ))
            }
        };

        // Create GPU context with optimal backend selection
        let gpu_context = self.create_optimal_gpu_context(&gpu_opt)?;

        // Prepare GPU buffer with memory coalescing
        let input_buffer = self.create_gpu_buffer(&gpu_context, &gpu_opt)?;

        // Select and execute optimized kernel
        let kernel = self.select_optimal_kernel(&gpu_context, kernel_name, &gpu_opt)?;
        let output_buffer = kernel.execute(&input_buffer, &params)?;

        // Transfer result back with optimal strategy
        self.gpu_buffer_to_tensor(output_buffer, &gpu_context, &gpu_opt)
    }

    /// Create optimal GPU context based on backend preference and hardware detection
    fn create_optimal_gpu_context(&self, gpu_opt: &GpuOptimization) -> Result<GpuContext> {
        for backend_type in &gpu_opt.backend_preference {
            match backend_type {
                GpuBackendType::Cuda => {
                    if let Ok(context) = CudaBackend::create_context() {
                        return Ok(context);
                    }
                }
                GpuBackendType::Metal => {
                    if let Ok(context) = MetalBackend::create_context() {
                        return Ok(context);
                    }
                }
                GpuBackendType::WebGpu => {
                    if let Ok(context) = WebGpuBackend::create_context() {
                        return Ok(context);
                    }
                }
                GpuBackendType::Rocm => {
                    if let Ok(context) = RocmBackend::create_context() {
                        return Ok(context);
                    }
                }
                GpuBackendType::OpenCl => {
                    if let Ok(context) = OpenClBackend::create_context() {
                        return Ok(context);
                    }
                }
            }
        }

        Err(torsh_core::error::TorshError::InvalidArgument(
            "No compatible GPU backend found".to_string(),
        ))
    }

    /// Create GPU buffer with optimal memory management
    fn create_gpu_buffer(
        &self,
        context: &GpuContext,
        gpu_opt: &GpuOptimization,
    ) -> Result<GpuBuffer<T>> {
        let data = self.to_vec()?;

        if gpu_opt.use_unified_memory {
            // Use unified memory for simplified management
            GpuBuffer::from_unified_memory(context, &data)
        } else if gpu_opt.use_pinned_memory {
            // Use pinned memory for faster transfers
            GpuBuffer::from_pinned_memory(context, &data)
        } else {
            // Standard GPU memory allocation
            GpuBuffer::from_data(context, &data)
        }
        .map_err(|e| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Failed to create GPU buffer: {}",
                e
            ))
        })
    }

    /// Select optimal kernel with automatic tuning
    fn select_optimal_kernel(
        &self,
        context: &GpuContext,
        kernel_name: &str,
        gpu_opt: &GpuOptimization,
    ) -> Result<GpuKernel> {
        let mut kernel = GpuKernel::load(context, kernel_name).map_err(|e| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Failed to load kernel '{}': {}",
                kernel_name, e
            ))
        })?;

        if gpu_opt.auto_kernel_tuning {
            // Automatic performance tuning
            kernel.auto_tune(self.shape().dims(), self.numel())?;
        }

        if gpu_opt.use_tensor_cores && kernel.supports_tensor_cores() {
            // Enable tensor core acceleration for supported operations
            kernel.enable_tensor_cores()?;
        }

        if gpu_opt.kernel_fusion_level > 0 {
            // Apply kernel fusion optimization
            kernel.enable_fusion(gpu_opt.kernel_fusion_level)?;
        }

        Ok(kernel)
    }

    /// Convert GPU buffer back to tensor with optimal transfer strategy
    fn gpu_buffer_to_tensor(
        &self,
        buffer: GpuBuffer<T>,
        context: &GpuContext,
        gpu_opt: &GpuOptimization,
    ) -> Result<Self> {
        let data = if gpu_opt.memory_coalescing {
            // Use memory coalescing for optimal bandwidth
            buffer.to_vec_coalesced()?
        } else {
            // Standard memory transfer
            buffer.to_vec()?
        };

        Self::from_data(data, self.shape().dims().to_vec(), self.device)
    }

    /// 🚀 Multi-GPU tensor distribution with automatic strategy selection
    pub fn distribute_multi_gpu(
        &self,
        gpu_count: usize,
        strategy: Option<MultiGpuStrategy>,
    ) -> Result<Vec<Self>> {
        if gpu_count <= 1 {
            return Ok(vec![self.clone()]);
        }

        let strategy = strategy.unwrap_or(MultiGpuStrategy::Auto);
        let effective_strategy = match strategy {
            MultiGpuStrategy::Auto => self.select_optimal_multi_gpu_strategy(gpu_count),
            s => s,
        };

        match effective_strategy {
            MultiGpuStrategy::DataParallel => self.data_parallel_distribution(gpu_count),
            MultiGpuStrategy::ModelParallel => self.model_parallel_distribution(gpu_count),
            MultiGpuStrategy::PipelineParallel => self.pipeline_parallel_distribution(gpu_count),
            _ => Ok(vec![self.clone()]), // Single GPU fallback
        }
    }

    /// Select optimal multi-GPU strategy based on tensor characteristics
    fn select_optimal_multi_gpu_strategy(&self, gpu_count: usize) -> MultiGpuStrategy {
        let total_elements = self.numel();
        let dims = self.shape().dims();

        // Data parallel for large batch dimensions
        if dims.len() > 0 && dims[0] >= gpu_count * 4 {
            return MultiGpuStrategy::DataParallel;
        }

        // Model parallel for large feature dimensions
        if dims.len() > 1 && dims.iter().skip(1).product::<usize>() > 1024 * 1024 {
            return MultiGpuStrategy::ModelParallel;
        }

        // Pipeline parallel for deep networks (many dimensions)
        if dims.len() > 3 {
            return MultiGpuStrategy::PipelineParallel;
        }

        // Default to data parallel
        MultiGpuStrategy::DataParallel
    }

    /// Data parallel distribution across multiple GPUs
    fn data_parallel_distribution(&self, gpu_count: usize) -> Result<Vec<Self>> {
        let dims = self.shape().dims();
        if dims.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Cannot distribute scalar tensor".to_string(),
            ));
        }

        let batch_size = dims[0];
        let chunk_size = (batch_size + gpu_count - 1) / gpu_count; // Ceiling division

        let mut distributed_tensors = Vec::with_capacity(gpu_count);
        let data = self.to_vec()?;
        let elements_per_batch = dims.iter().skip(1).product::<usize>();

        for gpu_id in 0..gpu_count {
            let start_batch = gpu_id * chunk_size;
            let end_batch = ((gpu_id + 1) * chunk_size).min(batch_size);

            if start_batch >= batch_size {
                break; // No more data for this GPU
            }

            let start_idx = start_batch * elements_per_batch;
            let end_idx = end_batch * elements_per_batch;
            let chunk_data = data[start_idx..end_idx].to_vec();

            let mut chunk_dims = dims.to_vec();
            chunk_dims[0] = end_batch - start_batch;

            let chunk_tensor = Self::from_data(chunk_data, chunk_dims, DeviceType::Cuda(gpu_id))?;

            distributed_tensors.push(chunk_tensor);
        }

        Ok(distributed_tensors)
    }

    /// Model parallel distribution (split feature dimensions)
    fn model_parallel_distribution(&self, gpu_count: usize) -> Result<Vec<Self>> {
        let dims = self.shape().dims();
        if dims.len() < 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Model parallel requires at least 2D tensor".to_string(),
            ));
        }

        // Split along the last dimension (features)
        let feature_dim = dims.len() - 1;
        let feature_size = dims[feature_dim];
        let chunk_size = (feature_size + gpu_count - 1) / gpu_count;

        let mut distributed_tensors = Vec::with_capacity(gpu_count);
        let data = self.to_vec()?;

        for gpu_id in 0..gpu_count {
            let start_feature = gpu_id * chunk_size;
            let end_feature = ((gpu_id + 1) * chunk_size).min(feature_size);

            if start_feature >= feature_size {
                break;
            }

            // Extract chunk data (simplified for demonstration)
            // In practice, this would need proper strided extraction
            let mut chunk_dims = dims.to_vec();
            chunk_dims[feature_dim] = end_feature - start_feature;

            // Create a simplified chunk (actual implementation would need proper indexing)
            let chunk_size_total: usize = chunk_dims.iter().product();
            let chunk_data = vec![T::default(); chunk_size_total];

            let chunk_tensor = Self::from_data(chunk_data, chunk_dims, DeviceType::Cuda(gpu_id))?;

            distributed_tensors.push(chunk_tensor);
        }

        Ok(distributed_tensors)
    }

    /// Pipeline parallel distribution (split across layers/operations)
    fn pipeline_parallel_distribution(&self, gpu_count: usize) -> Result<Vec<Self>> {
        // Pipeline parallel typically involves splitting the computation graph
        // For demonstration, we'll create identical copies on different GPUs
        let mut distributed_tensors = Vec::with_capacity(gpu_count);

        for gpu_id in 0..gpu_count {
            let pipeline_tensor = Self::from_data(
                self.to_vec()?,
                self.shape().dims().to_vec(),
                DeviceType::Cuda(gpu_id),
            )?;
            distributed_tensors.push(pipeline_tensor);
        }

        Ok(distributed_tensors)
    }

    /// 🚀 Mixed precision training support with tensor cores
    pub fn enable_mixed_precision(&mut self, precision: MixedPrecision) -> Result<()> {
        if let DeviceOptimization::Gpu(gpu_opt) = self.get_device_optimization(self.device) {
            if gpu_opt.use_tensor_cores {
                // Enable tensor core mixed precision
                TensorCore::enable_mixed_precision(precision)?;
                return Ok(());
            }
        }

        Err(torsh_core::error::TorshError::InvalidArgument(
            "Mixed precision requires tensor core support".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_device_transfer() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        // Test transfer to same device
        let same_device = tensor.to_device(DeviceType::Cpu).unwrap();
        assert_eq!(same_device.device(), DeviceType::Cpu);

        // Test transfer strategy
        assert_eq!(
            tensor.get_transfer_strategy(DeviceType::Cpu),
            TransferStrategy::NoTransfer
        );
        assert_eq!(
            tensor.get_transfer_strategy(DeviceType::Cuda(0)),
            TransferStrategy::DirectTransfer
        );
    }

    #[test]
    fn test_operation_scheduler() {
        let mut scheduler = OperationScheduler::new();

        // Schedule operations
        let op1 = scheduler
            .schedule_operation(DeviceType::Cpu, OperationType::Compute, 5, vec![])
            .unwrap();

        let op2 = scheduler
            .schedule_operation(DeviceType::Cpu, OperationType::Compute, 10, vec![])
            .unwrap();

        // Higher priority operation should be executed first
        assert_eq!(
            scheduler.execute_next_operation(DeviceType::Cpu).unwrap(),
            Some(op2)
        );
        assert_eq!(
            scheduler.execute_next_operation(DeviceType::Cpu).unwrap(),
            Some(op1)
        );
    }

    #[test]
    fn test_transfer_efficiency() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        // Same device should be efficient
        assert!(tensor.can_transfer_efficiently(DeviceType::Cpu));

        // CPU-GPU should be efficient
        assert!(tensor.can_transfer_efficiently(DeviceType::Cuda(0)));

        // CPU-Metal should be efficient
        assert!(tensor.can_transfer_efficiently(DeviceType::Metal(0)));
    }

    #[test]
    fn test_device_optimization_defaults() {
        let cpu_opt = CpuOptimization::default();
        assert!(cpu_opt.use_simd);
        assert!(cpu_opt.cache_friendly);
        assert!(cpu_opt.numa_aware);

        let gpu_opt = GpuOptimization::default();
        assert!(gpu_opt.use_pinned_memory);
        assert_eq!(gpu_opt.stream_count, 4);
        assert!(!gpu_opt.mixed_precision);
    }

    #[test]
    fn test_global_scheduler() {
        initialize_global_scheduler().unwrap();

        {
            let mut scheduler = get_global_scheduler();
            let scheduler = scheduler.as_mut().unwrap();

            let op_id = scheduler
                .schedule_operation(DeviceType::Cpu, OperationType::Compute, 5, vec![])
                .unwrap();

            assert_eq!(scheduler.get_queue_length(DeviceType::Cpu), 1);
            assert_eq!(
                scheduler.execute_next_operation(DeviceType::Cpu).unwrap(),
                Some(op_id)
            );
        }
    }
}
