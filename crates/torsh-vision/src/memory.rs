//! Memory optimization utilities for torsh-vision
//!
//! This module provides tools for optimizing memory usage in computer vision workloads:
//! - Tensor memory management and pooling
//! - Image cache optimization
//! - Memory-mapped file operations
//! - Batch processing with controlled memory usage
//! - Memory profiling and monitoring

use crate::{Result, VisionError};
// use half; // Commented out - half crate not available
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use torsh_tensor::{creation, Tensor};

/// Memory pool for tensor reuse to reduce allocations
pub struct TensorPool {
    pools: HashMap<Vec<usize>, VecDeque<Tensor<f32>>>,
    max_pool_size: usize,
    total_tensors: usize,
    allocation_count: Arc<Mutex<usize>>,
    reuse_count: Arc<Mutex<usize>>,
}

impl TensorPool {
    /// Create a new tensor pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_pool_size,
            total_tensors: 0,
            allocation_count: Arc::new(Mutex::new(0)),
            reuse_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a tensor from the pool or create a new one
    pub fn get_tensor(&mut self, shape: &[usize]) -> Result<Tensor<f32>> {
        let shape_key = shape.to_vec();

        if let Some(pool) = self.pools.get_mut(&shape_key) {
            if let Some(tensor) = pool.pop_front() {
                *self.reuse_count.lock().unwrap() += 1;
                return Ok(tensor);
            }
        }

        // Create new tensor
        *self.allocation_count.lock().unwrap() += 1;
        let tensor = creation::zeros(shape)?;
        Ok(tensor)
    }

    /// Return a tensor to the pool for reuse
    pub fn return_tensor(&mut self, tensor: Tensor<f32>) -> Result<()> {
        let shape = tensor.shape().dims().to_vec();

        let pool = self.pools.entry(shape).or_insert_with(VecDeque::new);

        if pool.len() < self.max_pool_size {
            // Zero out the tensor before returning to pool
            let mut tensor = tensor;
            tensor.zero_()?;
            pool.push_back(tensor);
            self.total_tensors += 1;
        }

        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let allocation_count = *self.allocation_count.lock().unwrap();
        let reuse_count = *self.reuse_count.lock().unwrap();
        let total_operations = allocation_count + reuse_count;

        PoolStats {
            total_tensors: self.total_tensors,
            allocation_count,
            reuse_count,
            reuse_rate: if total_operations > 0 {
                reuse_count as f32 / total_operations as f32
            } else {
                0.0
            },
            pools_count: self.pools.len(),
        }
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
        self.total_tensors = 0;
    }

    /// Get memory usage estimate in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let mut total_bytes = 0;

        for (shape, pool) in &self.pools {
            let tensor_size = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
            total_bytes += tensor_size * pool.len();
        }

        total_bytes
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_tensors: usize,
    pub allocation_count: usize,
    pub reuse_count: usize,
    pub reuse_rate: f32,
    pub pools_count: usize,
}

/// Memory-efficient batch processor with controlled memory usage
pub struct MemoryEfficientBatchProcessor {
    max_memory_mb: usize,
    current_memory_mb: usize,
    tensor_pool: TensorPool,
    processing_queue: VecDeque<ProcessingItem>,
}

#[derive(Debug)]
struct ProcessingItem {
    id: usize,
    tensor: Tensor<f32>,
    estimated_size_mb: usize,
}

impl MemoryEfficientBatchProcessor {
    /// Create a new memory-efficient batch processor
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            max_memory_mb,
            current_memory_mb: 0,
            tensor_pool: TensorPool::new(100),
            processing_queue: VecDeque::new(),
        }
    }

    /// Add a tensor to the processing queue
    pub fn add_tensor(&mut self, tensor: Tensor<f32>) -> Result<usize> {
        let id = self.processing_queue.len();
        let estimated_size_mb = self.estimate_tensor_size_mb(&tensor);

        // Check if adding this tensor would exceed memory limit
        if self.current_memory_mb + estimated_size_mb > self.max_memory_mb {
            self.process_batch()?;
        }

        self.processing_queue.push_back(ProcessingItem {
            id,
            tensor,
            estimated_size_mb,
        });

        self.current_memory_mb += estimated_size_mb;
        Ok(id)
    }

    /// Process the current batch of tensors
    pub fn process_batch(&mut self) -> Result<Vec<ProcessingResult>> {
        let mut results = Vec::new();

        while let Some(item) = self.processing_queue.pop_front() {
            // Simulate processing - in real use, this would apply transforms, etc.
            let processed_tensor = self.process_single_tensor(item.tensor)?;

            results.push(ProcessingResult {
                id: item.id,
                tensor: processed_tensor,
            });

            self.current_memory_mb = self
                .current_memory_mb
                .saturating_sub(item.estimated_size_mb);
        }

        Ok(results)
    }

    /// Process a single tensor (placeholder implementation)
    fn process_single_tensor(&mut self, tensor: Tensor<f32>) -> Result<Tensor<f32>> {
        // This is where actual processing would happen
        // For now, just return the tensor as-is
        Ok(tensor)
    }

    /// Estimate tensor memory usage in MB
    fn estimate_tensor_size_mb(&self, tensor: &Tensor<f32>) -> usize {
        let elements = tensor.shape().dims().iter().product::<usize>();
        let bytes = elements * std::mem::size_of::<f32>();
        (bytes + 1024 * 1024 - 1) / (1024 * 1024) // Round up to MB
    }

    /// Flush all remaining tensors
    pub fn flush(&mut self) -> Result<Vec<ProcessingResult>> {
        self.process_batch()
    }

    /// Get current memory usage
    pub fn current_memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            current_mb: self.current_memory_mb,
            max_mb: self.max_memory_mb,
            utilization: self.current_memory_mb as f32 / self.max_memory_mb as f32,
            queue_size: self.processing_queue.len(),
        }
    }
}

/// Processing result
#[derive(Debug)]
pub struct ProcessingResult {
    pub id: usize,
    pub tensor: Tensor<f32>,
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub current_mb: usize,
    pub max_mb: usize,
    pub utilization: f32,
    pub queue_size: usize,
}

/// Memory profiler for tracking allocations and usage
pub struct MemoryProfiler {
    allocations: Arc<RwLock<Vec<AllocationRecord>>>,
    peak_usage: Arc<Mutex<usize>>,
    current_usage: Arc<Mutex<usize>>,
    start_time: std::time::Instant,
}

#[derive(Debug, Clone)]
pub struct AllocationRecord {
    pub size_bytes: usize,
    pub timestamp: std::time::Duration,
    pub location: String,
    pub operation: String,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self {
            allocations: Arc::new(RwLock::new(Vec::new())),
            peak_usage: Arc::new(Mutex::new(0)),
            current_usage: Arc::new(Mutex::new(0)),
            start_time: std::time::Instant::now(),
        }
    }

    /// Record an allocation
    pub fn record_allocation(&self, size_bytes: usize, location: &str, operation: &str) {
        let timestamp = self.start_time.elapsed();

        {
            let mut allocations = self.allocations.write();
            allocations.push(AllocationRecord {
                size_bytes,
                timestamp,
                location: location.to_string(),
                operation: operation.to_string(),
            });
        }

        // Update current usage
        {
            let mut current_usage = self.current_usage.lock().unwrap();
            *current_usage += size_bytes;

            let mut peak_usage = self.peak_usage.lock().unwrap();
            if *current_usage > *peak_usage {
                *peak_usage = *current_usage;
            }
        }
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size_bytes: usize) {
        let mut current_usage = self.current_usage.lock().unwrap();
        *current_usage = current_usage.saturating_sub(size_bytes);
    }

    /// Get profiling summary
    pub fn summary(&self) -> ProfilingSummary {
        let allocations = self.allocations.read();
        let peak_usage = *self.peak_usage.lock().unwrap();
        let current_usage = *self.current_usage.lock().unwrap();

        let total_allocations = allocations.len();
        let total_allocated: usize = allocations.iter().map(|a| a.size_bytes).sum();
        let average_allocation = if total_allocations > 0 {
            total_allocated / total_allocations
        } else {
            0
        };

        ProfilingSummary {
            total_allocations,
            total_allocated_bytes: total_allocated,
            peak_usage_bytes: peak_usage,
            current_usage_bytes: current_usage,
            average_allocation_bytes: average_allocation,
            duration: self.start_time.elapsed(),
        }
    }

    /// Get allocations by operation type
    pub fn allocations_by_operation(&self) -> HashMap<String, AllocationStats> {
        let allocations = self.allocations.read();
        let mut stats: HashMap<String, AllocationStats> = HashMap::new();

        for allocation in allocations.iter() {
            let entry = stats
                .entry(allocation.operation.clone())
                .or_insert(AllocationStats {
                    count: 0,
                    total_bytes: 0,
                    average_bytes: 0,
                    max_bytes: 0,
                });

            entry.count += 1;
            entry.total_bytes += allocation.size_bytes;
            entry.max_bytes = entry.max_bytes.max(allocation.size_bytes);
        }

        // Calculate averages
        for (_, stats) in stats.iter_mut() {
            stats.average_bytes = stats.total_bytes / stats.count;
        }

        stats
    }

    /// Clear profiling data
    pub fn clear(&self) {
        let mut allocations = self.allocations.write();
        allocations.clear();

        let mut peak_usage = self.peak_usage.lock().unwrap();
        let mut current_usage = self.current_usage.lock().unwrap();
        *peak_usage = 0;
        *current_usage = 0;
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profiling summary
#[derive(Debug, Clone)]
pub struct ProfilingSummary {
    pub total_allocations: usize,
    pub total_allocated_bytes: usize,
    pub peak_usage_bytes: usize,
    pub current_usage_bytes: usize,
    pub average_allocation_bytes: usize,
    pub duration: std::time::Duration,
}

/// Allocation statistics by operation
#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub count: usize,
    pub total_bytes: usize,
    pub average_bytes: usize,
    pub max_bytes: usize,
}

/// Memory optimization utilities
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// Optimize tensor for memory usage by converting to appropriate dtype
    pub fn optimize_tensor_dtype(
        tensor: &Tensor<f32>,
        target_precision: Precision,
    ) -> Result<OptimizedTensor> {
        match target_precision {
            Precision::Full => Ok(OptimizedTensor::F32(tensor.clone())),
            Precision::Half => {
                let f16_tensor = tensor.to_dtype(torsh_core::dtype::DType::F16)?;
                Ok(OptimizedTensor::F16(f16_tensor))
            }
            Precision::Int8 => {
                // Quantize to int8 (simplified approach)
                let min_val = tensor.min()?.item()?;
                let max_val = tensor.max(None, false)?.item()?;
                let scale = (max_val - min_val) / 255.0;

                let mut quantized = tensor.clone();
                quantized.sub_scalar_(min_val)?;
                quantized.div_scalar_(scale)?;
                let quantized = quantized.round()?.clamp(0.0, 255.0)?;
                Ok(OptimizedTensor::Quantized {
                    data: quantized,
                    scale,
                    zero_point: min_val,
                })
            }
        }
    }

    /// Calculate optimal batch size based on available memory
    pub fn calculate_optimal_batch_size(
        sample_shape: &[usize],
        available_memory_mb: usize,
        safety_factor: f32,
    ) -> usize {
        let sample_size_bytes = sample_shape.iter().product::<usize>() * std::mem::size_of::<f32>();
        let available_bytes = (available_memory_mb * 1024 * 1024) as f32 * safety_factor;
        let max_batch_size = (available_bytes / sample_size_bytes as f32) as usize;

        // Ensure at least batch size of 1
        max_batch_size.max(1)
    }

    /// Estimate memory usage for a batch of tensors
    pub fn estimate_batch_memory(shapes: &[Vec<usize>]) -> MemoryEstimate {
        let mut total_bytes = 0;
        let mut max_tensor_bytes = 0;

        for shape in shapes {
            let tensor_bytes = shape.iter().product::<usize>() * std::mem::size_of::<f32>();
            total_bytes += tensor_bytes;
            max_tensor_bytes = max_tensor_bytes.max(tensor_bytes);
        }

        MemoryEstimate {
            total_bytes,
            total_mb: total_bytes / (1024 * 1024),
            max_tensor_bytes,
            average_tensor_bytes: total_bytes / shapes.len(),
            tensor_count: shapes.len(),
        }
    }
}

/// Precision levels for optimization
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    Full, // f32
    Half, // f16
    Int8, // 8-bit quantized
}

/// Optimized tensor variants
#[derive(Debug, Clone)]
pub enum OptimizedTensor {
    F32(Tensor<f32>),
    // F16(Tensor<half::f16>), // Commented out - half crate not available
    F16(Tensor<f32>), // Using f32 for now instead of f16
    Quantized {
        data: Tensor<f32>,
        scale: f32,
        zero_point: f32,
    },
}

impl OptimizedTensor {
    /// Convert back to f32 tensor
    pub fn to_f32(&self) -> Result<Tensor<f32>> {
        match self {
            OptimizedTensor::F32(tensor) => Ok(tensor.clone()),
            OptimizedTensor::F16(tensor) => tensor
                .to_dtype(torsh_core::dtype::DType::F32)
                .map_err(|e| VisionError::TensorError(e)),
            OptimizedTensor::Quantized {
                data,
                scale,
                zero_point,
            } => {
                let mut result = data.clone();
                result.mul_scalar_(*scale)?;
                result.add_scalar_(*zero_point)?;
                Ok(result)
            }
        }
    }

    /// Get estimated memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            OptimizedTensor::F32(tensor) => {
                tensor.shape().dims().iter().product::<usize>() * std::mem::size_of::<f32>()
            }
            OptimizedTensor::F16(tensor) => {
                tensor.shape().dims().iter().product::<usize>() * 2 // f16 is 2 bytes
            }
            OptimizedTensor::Quantized { data, .. } => {
                data.shape().dims().iter().product::<usize>() * std::mem::size_of::<f32>() + 8
                // + scale/zero_point
            }
        }
    }
}

/// Memory usage estimate
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    pub total_bytes: usize,
    pub total_mb: usize,
    pub max_tensor_bytes: usize,
    pub average_tensor_bytes: usize,
    pub tensor_count: usize,
}

/// Global memory manager for the entire vision library
pub struct GlobalMemoryManager {
    profiler: MemoryProfiler,
    tensor_pool: Arc<Mutex<TensorPool>>,
    batch_processor: Arc<Mutex<MemoryEfficientBatchProcessor>>,
    settings: MemorySettings,
}

/// Memory management settings
#[derive(Debug, Clone)]
pub struct MemorySettings {
    pub enable_pooling: bool,
    pub max_pool_size: usize,
    pub max_batch_memory_mb: usize,
    pub enable_profiling: bool,
    pub auto_optimization: bool,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            max_pool_size: 100,
            max_batch_memory_mb: 1024, // 1GB
            enable_profiling: false,
            auto_optimization: true,
        }
    }
}

impl GlobalMemoryManager {
    /// Create a new global memory manager
    pub fn new(settings: MemorySettings) -> Self {
        Self {
            profiler: MemoryProfiler::new(),
            tensor_pool: Arc::new(Mutex::new(TensorPool::new(settings.max_pool_size))),
            batch_processor: Arc::new(Mutex::new(MemoryEfficientBatchProcessor::new(
                settings.max_batch_memory_mb,
            ))),
            settings,
        }
    }

    /// Get a tensor from the global pool
    pub fn get_tensor(&self, shape: &[usize]) -> Result<Tensor<f32>> {
        if self.settings.enable_pooling {
            let mut pool = self.tensor_pool.lock().unwrap();
            pool.get_tensor(shape)
        } else {
            creation::zeros(shape).map_err(|e| VisionError::TensorError(e))
        }
    }

    /// Return a tensor to the global pool
    pub fn return_tensor(&self, tensor: Tensor<f32>) -> Result<()> {
        if self.settings.enable_pooling {
            let mut pool = self.tensor_pool.lock().unwrap();
            pool.return_tensor(tensor)
        } else {
            Ok(())
        }
    }

    /// Get global memory statistics
    pub fn global_stats(&self) -> GlobalMemoryStats {
        let pool_stats = if self.settings.enable_pooling {
            Some(self.tensor_pool.lock().unwrap().stats())
        } else {
            None
        };

        let profiling_summary = if self.settings.enable_profiling {
            Some(self.profiler.summary())
        } else {
            None
        };

        GlobalMemoryStats {
            pool_stats,
            profiling_summary,
            settings: self.settings.clone(),
        }
    }
}

/// Global memory statistics
#[derive(Debug, Clone)]
pub struct GlobalMemoryStats {
    pub pool_stats: Option<PoolStats>,
    pub profiling_summary: Option<ProfilingSummary>,
    pub settings: MemorySettings,
}

/// Thread-safe global memory manager instance
static GLOBAL_MEMORY_MANAGER: std::sync::OnceLock<GlobalMemoryManager> = std::sync::OnceLock::new();

/// Get the global memory manager instance
pub fn global_memory_manager() -> &'static GlobalMemoryManager {
    GLOBAL_MEMORY_MANAGER.get_or_init(|| GlobalMemoryManager::new(MemorySettings::default()))
}

/// Configure the global memory manager
/// Note: This only works if called before first access to global_memory_manager()
pub fn configure_global_memory(settings: MemorySettings) {
    let _ = GLOBAL_MEMORY_MANAGER.set(GlobalMemoryManager::new(settings));
}
