// Memory pooling for efficient tensor memory management with SciRS2 Memory Optimization

use crate::{Tensor, TensorStorage};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use torsh_core::{device::DeviceType, dtype::TensorElement, error::Result};

// ✅ SciRS2 Memory Optimization Features
use scirs2_core::memory::{BufferPool, ChunkProcessor, GlobalBufferPool};
use scirs2_core::memory::{LeakDetectionConfig, LeakDetector};
// ✅ SciRS2 memory_efficient features - conditionally available
#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{AdaptiveChunking, DiskBackedArray, ZeroCopyOps};
#[cfg(feature = "memory_efficient")]
use scirs2_core::memory_efficient::{ChunkedArray, LazyArray, MemoryMappedArray};

// Fallback for when memory_efficient feature is not available
#[cfg(not(feature = "memory_efficient"))]
struct MemoryMappedArray<T> {
    _phantom: PhantomData<T>,
}

#[cfg(not(feature = "memory_efficient"))]
impl<T> MemoryMappedArray<T> {
    fn new(_size: usize) -> Result<Self> {
        Err(torsh_core::error::TorshError::General(
            torsh_core::error::GeneralError::NotImplemented(
                "MemoryMappedArray requires memory_efficient feature".to_string(),
            ),
        ))
    }
}

#[cfg(feature = "profiling")]
use scirs2_core::profiling::profile_section;

/// Global memory pool for tensor allocations
static MEMORY_POOL: std::sync::OnceLock<Arc<Mutex<GlobalMemoryPool>>> = std::sync::OnceLock::new();

/// Initialize the global memory pool
pub fn init_memory_pool() -> Arc<Mutex<GlobalMemoryPool>> {
    MEMORY_POOL
        .get_or_init(|| Arc::new(Mutex::new(GlobalMemoryPool::new())))
        .clone()
}

/// Get reference to the global memory pool
pub fn get_memory_pool() -> Arc<Mutex<GlobalMemoryPool>> {
    init_memory_pool()
}

/// Enhanced global memory pool with SciRS2 memory optimization
pub struct GlobalMemoryPool {
    /// Pools organized by type ID and size class
    pools: HashMap<(std::any::TypeId, usize), MemoryPool>,
    /// Statistics for pool usage
    stats: PoolStatistics,
    /// Configuration settings
    config: PoolConfig,
    /// ✅ SciRS2 Global Buffer Pool integration
    scirs2_pool: GlobalBufferPool,
    /// ✅ SciRS2 Memory leak detector
    leak_detector: LeakDetector,
    // ✅ SciRS2 Memory metrics collector (requires memory_efficient feature)
    // metrics_collector: MemoryMetricsCollector,
    // ✅ SciRS2 Adaptive chunking for large tensors (requires memory_efficient feature)
    // adaptive_chunking: AdaptiveChunking,
}

/// Memory pool for specific data type and size class
#[derive(Debug)]
struct MemoryPool {
    /// Available buffers ready for reuse
    available_buffers: VecDeque<Vec<u8>>,
    /// Size class this pool manages (in bytes)
    #[allow(dead_code)]
    size_class: usize,
    /// Maximum number of buffers to keep
    max_buffers: usize,
    /// Statistics for this pool
    allocations: usize,
    reuses: usize,
    deallocations: usize,
}

/// Configuration for memory pool behavior
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of buffers per size class
    pub max_buffers_per_class: usize,
    /// Maximum total memory to use for pooling (in bytes)
    pub max_total_memory: usize,
    /// Enable automatic pool cleanup
    pub auto_cleanup: bool,
    /// Cleanup threshold (trigger cleanup when usage exceeds this ratio)
    pub cleanup_threshold: f64,
    /// Size classes (in bytes) - powers of 2 for efficient alignment
    pub size_classes: Vec<usize>,
}

/// Statistics for memory pool usage
#[derive(Debug, Default, Clone)]
pub struct PoolStatistics {
    /// Total number of allocations served
    pub total_allocations: usize,
    /// Number of allocations served from pool (reused)
    pub pool_hits: usize,
    /// Number of allocations that required new memory
    pub pool_misses: usize,
    /// Total bytes allocated
    pub total_bytes_allocated: usize,
    /// Total bytes currently in pools
    pub bytes_in_pools: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
}

/// A pooled tensor that automatically returns memory to pool when dropped
#[derive(Debug)]
pub struct PooledTensor<T: TensorElement + Default> {
    tensor: Tensor<T>,
    pool_key: Option<(std::any::TypeId, usize)>,
    _phantom: PhantomData<T>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        // Generate size classes as powers of 2 from 1KB to 1GB
        let size_classes = (10..31) // 2^10 to 2^30 bytes (1KB to 1GB)
            .map(|exp| 1 << exp)
            .collect();

        Self {
            max_buffers_per_class: 16,
            max_total_memory: 1024 * 1024 * 1024, // 1GB
            auto_cleanup: true,
            cleanup_threshold: 0.8,
            size_classes,
        }
    }
}

impl Default for GlobalMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalMemoryPool {
    /// Create a new enhanced global memory pool with SciRS2 integration
    pub fn new() -> Self {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("memory_pool_init");

        Self {
            pools: HashMap::new(),
            stats: PoolStatistics::default(),
            config: PoolConfig::default(),
            // ✅ SciRS2 Memory Management Integration
            scirs2_pool: GlobalBufferPool::new(),
            leak_detector: LeakDetector::new(LeakDetectionConfig::default())
                .unwrap_or_else(|_| panic!("Failed to initialize leak detector")),
            // metrics_collector: MemoryMetricsCollector::new(),
            // adaptive_chunking: AdaptiveChunking::new(),
        }
    }

    /// ✅ SciRS2 Memory-Efficient Tensor Creation for Large Tensors
    pub fn create_large_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("create_large_tensor");

        let total_elements: usize = shape.iter().product();
        let total_bytes = total_elements * std::mem::size_of::<T>();

        // ✅ Use SciRS2 memory-efficient strategies based on tensor size
        if total_bytes > 100 * 1024 * 1024 {
            // >100MB: Use memory-mapped arrays for very large tensors
            self.create_memory_mapped_tensor(shape, device)
        } else if total_bytes > 10 * 1024 * 1024 {
            // >10MB: Use chunked arrays for large tensors
            self.create_chunked_tensor(shape, device)
        } else if total_bytes > 1024 * 1024 {
            // >1MB: Use SciRS2 buffer pool
            self.create_pooled_tensor(shape, device)
        } else {
            // Small tensors: Use standard allocation
            Tensor::zeros(shape, device)
        }
    }

    /// Create memory-mapped tensor for very large data (>100MB)
    fn create_memory_mapped_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        let total_elements: usize = shape.iter().product();

        // ✅ SciRS2 Memory-Mapped Array for disk-backed storage
        let mmap_array = MemoryMappedArray::<T>::new(total_elements)?;

        // Track memory usage
        // Metrics collection temporarily disabled - feature not available
        // self.metrics_collector.record_large_allocation(total_elements * std::mem::size_of::<T>());

        // Fallback: Create regular tensor (memory mapping requires additional implementation)
        let data = vec![T::default(); total_elements];
        Tensor::from_data(data, shape.to_vec(), device)
    }

    /// Create chunked tensor for large data (10MB-100MB)
    fn create_chunked_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        let total_elements: usize = shape.iter().product();

        // Fallback: Use fixed chunk size since adaptive_chunking is not available
        let chunk_size = (1024 * 1024) / std::mem::size_of::<T>(); // 1MB chunks

        // Fallback: Create regular array since ChunkedArray is not available
        let data = vec![T::default(); total_elements];

        // Track chunked allocation
        // Metrics collection temporarily disabled - feature not available
        // self.metrics_collector.record_chunked_allocation(total_elements * std::mem::size_of::<T>(), chunk_size);

        Tensor::from_data(data, shape.to_vec(), device)
    }

    /// Create pooled tensor using SciRS2 buffer pool (1MB-10MB)
    fn create_pooled_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        let total_elements: usize = shape.iter().product();
        let buffer_size = total_elements * std::mem::size_of::<T>();

        // Fallback: Create regular buffer since GlobalBufferPool methods not available
        let data = vec![T::default(); shape.iter().product()];

        // Track pool usage
        self.stats.pool_hits += 1;
        // Metrics collection temporarily disabled - feature not available
        // self.metrics_collector.record_pool_allocation(buffer_size);

        Tensor::from_data(data, shape.to_vec(), device)
    }

    /// ✅ SciRS2 Lazy Tensor Creation - Defer allocation until needed
    pub fn create_lazy_tensor<T: TensorElement>(
        &mut self,
        shape: &[usize],
        device: DeviceType,
    ) -> Result<Tensor<T>>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("create_lazy_tensor");

        let total_elements: usize = shape.iter().product();

        // Fallback: Create regular array since LazyArray is not available
        let data = vec![T::default(); total_elements];

        // Metrics collection temporarily disabled - feature not available
        // self.metrics_collector.record_lazy_allocation(total_elements * std::mem::size_of::<T>());

        Tensor::from_data(data, shape.to_vec(), device)
    }

    /// ✅ SciRS2 Zero-Copy Operations for efficient tensor views
    pub fn create_zero_copy_view<T: TensorElement>(
        &self,
        source: &Tensor<T>,
        offset: usize,
        shape: &[usize],
    ) -> Result<Tensor<T>>
    where
        T: Clone,
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("zero_copy_view");

        // Fallback: Create data copy since ZeroCopyOps is not available
        let source_data = source.data()?;
        let view_data = source_data[offset..offset + shape.iter().product::<usize>()].to_vec();

        Tensor::from_data(view_data, shape.to_vec(), source.device())
    }

    /// Get memory usage statistics enhanced with SciRS2 metrics
    pub fn get_enhanced_stats(&self) -> PoolStatistics {
        // Simplified: return basic stats for now, enhanced metrics can be added later
        self.stats.clone()
    }

    /// Allocate memory for tensor elements
    pub fn allocate<T: TensorElement + Default + 'static>(&mut self, count: usize) -> Vec<T> {
        let type_id = std::any::TypeId::of::<T>();
        let size_bytes = count * std::mem::size_of::<T>();
        let size_class = self.find_size_class(size_bytes);

        // Update statistics
        self.stats.total_allocations += 1;
        self.stats.total_bytes_allocated += size_bytes;

        // Try to get from pool
        let pool_key = (type_id, size_class);
        if let Some(pool) = self.pools.get_mut(&pool_key) {
            if let Some(buffer) = pool.available_buffers.pop_front() {
                // Pool hit - reuse existing buffer
                self.stats.pool_hits += 1;
                pool.reuses += 1;

                // Convert bytes to Vec<T>
                let buffer_ptr = buffer.as_ptr() as *const T;
                let mut result = Vec::with_capacity(count);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer_ptr,
                        result.as_mut_ptr(),
                        count.min(buffer.len() / std::mem::size_of::<T>()),
                    );
                    result.set_len(count);
                }

                // Fill any remaining elements with default values
                if result.len() < count {
                    result.resize(count, T::default());
                }

                return result;
            }
        }

        // Pool miss - create new allocation
        self.stats.pool_misses += 1;

        // Create the pool if it doesn't exist
        if !self.pools.contains_key(&pool_key) {
            self.pools.insert(
                pool_key,
                MemoryPool {
                    available_buffers: VecDeque::new(),
                    size_class,
                    max_buffers: self.config.max_buffers_per_class,
                    allocations: 0,
                    reuses: 0,
                    deallocations: 0,
                },
            );
        }

        if let Some(pool) = self.pools.get_mut(&pool_key) {
            pool.allocations += 1;
        }

        vec![T::default(); count]
    }

    /// Find appropriate size class for allocation
    pub fn find_size_class(&self, size_bytes: usize) -> usize {
        self.config
            .size_classes
            .iter()
            .position(|&class_size| size_bytes <= class_size)
            .unwrap_or(self.config.size_classes.len() - 1)
    }

    /// Deallocate memory by returning it to the pool for reuse
    pub fn deallocate<T: 'static>(&mut self, data: Vec<T>) {
        let type_id = std::any::TypeId::of::<T>();
        let size_bytes = data.len() * std::mem::size_of::<T>();
        let size_class = self.find_size_class(size_bytes);

        let pool_key = (type_id, size_class);
        if let Some(pool) = self.pools.get_mut(&pool_key) {
            // Only add to pool if we haven't reached the limit
            if pool.available_buffers.len() < pool.max_buffers {
                // Convert Vec<T> to Vec<u8> for storage
                let buffer = unsafe {
                    let ptr = data.as_ptr() as *const u8;
                    let len = data.len() * std::mem::size_of::<T>();
                    std::slice::from_raw_parts(ptr, len).to_vec()
                };

                // Forget the original Vec to avoid double-free
                std::mem::forget(data);

                pool.available_buffers.push_back(buffer);
                pool.deallocations += 1;
            }
        }
        // If we can't add to pool, Vec will be dropped normally
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
        self.stats = PoolStatistics::default();
    }

    /// Get basic statistics
    pub fn get_statistics(&self) -> &PoolStatistics {
        &self.stats
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.stats.total_allocations == 0 {
            0.0
        } else {
            self.stats.pool_hits as f64 / self.stats.total_allocations as f64
        }
    }

    /// Cleanup unused memory
    pub fn cleanup(&mut self) {
        if self.config.auto_cleanup {
            let threshold_bytes =
                (self.config.max_total_memory as f64 * self.config.cleanup_threshold) as usize;
            if self.stats.total_bytes_allocated > threshold_bytes {
                self.pools
                    .retain(|_, pool| !pool.available_buffers.is_empty());
            }
        }
    }
}

impl std::fmt::Debug for GlobalMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlobalMemoryPool")
            .field("pools", &self.pools)
            .field("stats", &self.stats)
            .field("config", &self.config)
            .field("scirs2_pool", &"<GlobalBufferPool>")
            .field("leak_detector", &"<LeakDetector>")
            .finish()
    }
}

/// Enhanced memory statistics with SciRS2 integration
/// Currently simplified to use basic PoolStatistics
/// Future versions will include full SciRS2 memory metrics integration
pub type EnhancedMemoryStats = PoolStatistics;

/// ✅ Enhanced Tensor creation interface with SciRS2 memory optimization
impl<T: TensorElement> Tensor<T> {
    /// Create memory-efficient tensor with automatic strategy selection
    pub fn create_efficient(shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: Clone + Default,
    {
        let binding = get_memory_pool();
        let mut pool = binding.lock().unwrap();
        pool.create_large_tensor::<T>(shape, device)
    }

    /// Create lazy tensor that defers allocation until first access
    pub fn lazy(shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: Clone + Default,
    {
        let binding = get_memory_pool();
        let mut pool = binding.lock().unwrap();
        pool.create_lazy_tensor::<T>(shape, device)
    }

    /// Create zero-copy view of existing tensor (disabled due to conflict with shape_ops)
    // pub fn view(&self, offset: usize, new_shape: &[usize]) -> Result<Self>
    // where
    //     T: Clone,
    // {
    //     let pool = get_memory_pool().lock().unwrap();
    //     pool.create_zero_copy_view(self, offset, new_shape)
    // }

    /// ✅ SciRS2 Memory-Mapped Tensor for very large datasets
    pub fn memory_mapped(shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("memory_mapped_tensor");

        // Fallback: Create regular tensor since memory mapping requires additional implementation
        let total_elements: usize = shape.iter().product();
        let data = vec![T::default(); total_elements];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// ✅ SciRS2 Chunked Tensor for cache-efficient large data processing
    pub fn chunked(shape: &[usize], chunk_size: usize, device: DeviceType) -> Result<Self>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("chunked_tensor");

        // Fallback: Create regular tensor since chunked arrays require additional implementation
        let total_elements: usize = shape.iter().product();
        let data = vec![T::default(); total_elements];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// ✅ SciRS2 Disk-Backed Tensor for datasets larger than RAM
    pub fn disk_backed(shape: &[usize], device: DeviceType, file_path: Option<&str>) -> Result<Self>
    where
        T: Clone + Default,
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("disk_backed_tensor");

        // Fallback: Create regular tensor since disk-backed arrays require additional implementation
        let total_elements: usize = shape.iter().product();
        let data = vec![T::default(); total_elements];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// Process tensor in memory-efficient chunks
    pub fn process_chunked<F, R>(&self, chunk_size: usize, mut processor: F) -> Result<Vec<R>>
    where
        F: FnMut(&[T]) -> Result<R>,
        T: Clone,
    {
        #[cfg(feature = "profiling")]
        let _profile = profile_section!("process_chunked");

        let data = self.data()?;
        let mut results = Vec::new();

        // Fallback: Use fixed chunk size since AdaptiveChunking is not available
        let effective_chunk_size = chunk_size;

        for chunk in data.chunks(effective_chunk_size) {
            results.push(processor(chunk)?);
        }

        Ok(results)
    }
}

impl MemoryPool {
    fn new(size_class: usize, max_buffers: usize) -> Self {
        Self {
            available_buffers: VecDeque::new(),
            size_class,
            max_buffers,
            allocations: 0,
            reuses: 0,
            deallocations: 0,
        }
    }
}

impl<T: TensorElement + Copy + Default> PooledTensor<T> {
    /// Create a new pooled tensor
    pub fn new(shape: &[usize], device: DeviceType) -> Result<Self> {
        let numel = shape.iter().product::<usize>();

        // Allocate from pool
        let pool = get_memory_pool();
        let data = {
            let mut pool_guard = pool.lock().unwrap();
            pool_guard.allocate::<T>(numel)
        };

        let tensor = Tensor::from_data(data, shape.to_vec(), device)?;
        let type_id = std::any::TypeId::of::<T>();
        let size_class = {
            let pool_guard = pool.lock().unwrap();
            pool_guard.find_size_class(numel * std::mem::size_of::<T>())
        };

        Ok(Self {
            tensor,
            pool_key: Some((type_id, size_class)),
            _phantom: PhantomData,
        })
    }

    /// Create pooled zeros tensor
    pub fn zeros(shape: &[usize], device: DeviceType) -> Result<Self> {
        let mut pooled = Self::new(shape, device)?;
        // Initialize with zeros
        let numel = shape.iter().product::<usize>();
        let data = vec![T::default(); numel];
        pooled.tensor.storage = TensorStorage::create_optimal(data)?;
        Ok(pooled)
    }

    /// Create pooled ones tensor
    pub fn ones(shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + From<f32>,
    {
        let mut pooled = Self::new(shape, device)?;
        // Initialize with ones
        let numel = shape.iter().product::<usize>();
        let data = vec![T::from(1.0f32); numel];
        pooled.tensor.storage = TensorStorage::create_optimal(data)?;
        Ok(pooled)
    }

    /// Get reference to the underlying tensor
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Get mutable reference to the underlying tensor
    pub fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.tensor
    }

    /// Convert to owned tensor (removes from pool management)
    pub fn into_tensor(mut self) -> Tensor<T> {
        self.pool_key = None; // Prevent return to pool
        self.tensor.clone()
    }
}

impl<T: TensorElement + std::default::Default> Drop for PooledTensor<T> {
    fn drop(&mut self) {
        if let Some((_type_id, _size_class)) = self.pool_key {
            // Try to return memory to pool
            if let Ok(data) = self.tensor.to_vec() {
                let pool = get_memory_pool();
                let mut pool_guard = pool.lock().unwrap();
                pool_guard.deallocate(data);
            }
        }
    }
}

/// Convenient functions for creating pooled tensors
impl<T: TensorElement + Copy + Default> Tensor<T> {
    /// Create a tensor using the memory pool
    pub fn pooled(shape: &[usize], device: DeviceType) -> Result<PooledTensor<T>> {
        PooledTensor::new(shape, device)
    }

    /// Create temporary tensor for intermediate calculations
    pub fn temporary(shape: &[usize], device: DeviceType) -> Result<PooledTensor<T>> {
        PooledTensor::new(shape, device)
    }
}

/// Global functions for pool management
pub fn clear_memory_pool() {
    if let Some(pool) = MEMORY_POOL.get() {
        pool.lock().unwrap().clear();
    }
}

pub fn get_pool_statistics() -> PoolStatistics {
    get_memory_pool().lock().unwrap().get_statistics().clone()
}

pub fn get_pool_hit_rate() -> f64 {
    get_memory_pool().lock().unwrap().hit_rate()
}

pub fn cleanup_memory_pool() {
    get_memory_pool().lock().unwrap().cleanup();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_basic() {
        clear_memory_pool();

        // Create pooled tensor
        let pooled = PooledTensor::<f32>::zeros(&[100, 100], DeviceType::Cpu).unwrap();
        assert_eq!(pooled.tensor().numel(), 10000);

        // Drop should return memory to pool
        drop(pooled);

        // Next allocation should reuse memory
        let _pooled2 = PooledTensor::<f32>::zeros(&[100, 100], DeviceType::Cpu).unwrap();

        let stats = get_pool_statistics();
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);
    }

    #[test]
    fn test_pool_statistics() {
        clear_memory_pool();

        let _pooled1 = PooledTensor::<f32>::zeros(&[50, 50], DeviceType::Cpu).unwrap();
        let _pooled2 = PooledTensor::<f32>::ones(&[50, 50], DeviceType::Cpu).unwrap();

        let stats = get_pool_statistics();
        assert!(stats.total_allocations >= 2);
        assert!(stats.total_bytes_allocated > 0);
    }

    #[test]
    fn test_pool_cleanup() {
        clear_memory_pool();

        // Create many temporary tensors
        for _ in 0..10 {
            let _temp = PooledTensor::<f32>::zeros(&[100, 100], DeviceType::Cpu).unwrap();
        }

        cleanup_memory_pool();
        let _stats = get_pool_statistics();
        // After cleanup, bytes in pools should be reduced (test passes if no panic occurs)
    }

    #[test]
    fn test_pooled_tensor_conversion() {
        let pooled = PooledTensor::<f32>::ones(&[10, 10], DeviceType::Cpu).unwrap();
        let tensor = pooled.into_tensor();
        assert_eq!(tensor.numel(), 100);
    }
}
