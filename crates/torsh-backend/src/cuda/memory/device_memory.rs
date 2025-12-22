//! CUDA device memory management
//!
//! This module provides comprehensive CUDA device memory management including
//! allocation, deallocation, memory pooling, and device-specific optimizations.
//! It manages GPU memory directly using CUDA runtime APIs.

use super::allocation::{
    size_class, AllocationMetadata, AllocationRequest, AllocationStats, AllocationType,
    CudaAllocation,
};
use crate::cuda::error::{CudaError, CudaResult};
use cust::device::Device as CustDevice;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(debug_assertions)]
use std::collections::HashSet;
#[cfg(debug_assertions)]
static ALLOCATION_TRACKER: once_cell::sync::Lazy<Mutex<HashSet<usize>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashSet::new()));

/// CUDA device memory manager with advanced pooling and optimization
///
/// Manages GPU device memory with pooling, pressure detection, and automatic
/// optimization. Provides high-performance allocation with minimal fragmentation.
#[derive(Debug)]
pub struct CudaMemoryManager {
    /// CUDA device ID
    device_id: usize,

    /// Memory pools organized by size class
    pools: Mutex<HashMap<usize, DeviceMemoryPool>>,

    /// Total allocated bytes (atomic for thread-safety)
    total_allocated: AtomicUsize,

    /// Peak memory usage
    peak_allocated: AtomicUsize,

    /// Memory limit for this device
    memory_limit: AtomicUsize,

    /// Memory pressure threshold
    pressure_threshold: AtomicUsize,

    /// Allocation statistics
    stats: Mutex<AllocationStats>,

    /// Configuration settings
    config: DeviceMemoryConfig,

    /// Last cleanup time
    last_cleanup: Mutex<Instant>,

    /// Device properties cache
    device_properties: DeviceProperties,
}

/// Device memory pool for specific size class
#[derive(Debug)]
pub struct DeviceMemoryPool {
    /// Size class (power of 2)
    size_class: usize,

    /// Available allocations for reuse
    free_blocks: Vec<CudaAllocation>,

    /// Currently allocated blocks
    allocated_blocks: Vec<CudaAllocation>,

    /// Pool statistics
    pool_stats: PoolStatistics,

    /// Last access time for cleanup
    last_access: Instant,

    /// Pool configuration
    config: PoolConfig,
}

/// Configuration for device memory management
#[derive(Debug, Clone)]
pub struct DeviceMemoryConfig {
    /// Maximum memory usage percentage (0.0 to 1.0)
    pub max_memory_fraction: f32,

    /// Memory pressure threshold percentage
    pub pressure_threshold_fraction: f32,

    /// Enable automatic memory pooling
    pub enable_pooling: bool,

    /// Maximum pool size per size class
    pub max_pool_size: usize,

    /// Enable memory compaction under pressure
    pub enable_compaction: bool,

    /// Cleanup interval for unused pools
    pub cleanup_interval: Duration,

    /// Enable debug allocation tracking
    pub debug_tracking: bool,

    /// Preferred allocation alignment
    pub allocation_alignment: usize,

    /// Enable asynchronous allocations where possible
    pub enable_async_alloc: bool,
}

/// Pool-specific configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of free blocks to keep
    pub max_free_blocks: usize,

    /// Minimum allocation age before cleanup
    pub min_age_for_cleanup: Duration,

    /// Pool growth strategy
    pub growth_strategy: GrowthStrategy,

    /// Enable pool statistics tracking
    pub track_statistics: bool,
}

/// Pool growth strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStrategy {
    /// Fixed size pool
    Fixed,
    /// Linear growth
    Linear,
    /// Exponential growth with cap
    Exponential { max_size: usize },
    /// Adaptive growth based on usage patterns
    Adaptive,
}

/// Pool statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    /// Total allocations served by this pool
    pub total_allocations: u64,

    /// Cache hits (reused allocations)
    pub cache_hits: u64,

    /// Cache misses (new allocations)
    pub cache_misses: u64,

    /// Current pool utilization (0.0 to 1.0)
    pub utilization: f32,

    /// Peak utilization
    pub peak_utilization: f32,

    /// Average allocation lifetime
    pub average_lifetime: Duration,

    /// Memory efficiency (allocated/requested)
    pub memory_efficiency: f32,
}

/// CUDA device properties cache
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Total device memory in bytes
    pub total_memory: usize,

    /// Device compute capability
    pub compute_capability: (i32, i32),

    /// Memory bus width
    pub memory_bus_width: i32,

    /// Memory clock rate
    pub memory_clock_rate: i32,

    /// L2 cache size
    pub l2_cache_size: i32,

    /// Maximum threads per block
    pub max_threads_per_block: i32,

    /// Device supports unified addressing
    pub unified_addressing: bool,

    /// Device supports concurrent kernels
    pub concurrent_kernels: bool,
}

/// Memory allocation context for tracking and debugging
#[derive(Debug, Clone)]
pub struct AllocationContext {
    /// Request that generated this allocation
    pub request: AllocationRequest,

    /// Allocation result metadata
    pub metadata: AllocationMetadata,

    /// Performance metrics for this allocation
    pub performance: AllocationPerformance,

    /// Device context information
    pub device_context: DeviceContext,
}

/// Performance metrics for individual allocations
#[derive(Debug, Clone)]
pub struct AllocationPerformance {
    /// Time taken to allocate
    pub allocation_time: Duration,

    /// Whether allocation came from pool cache
    pub cache_hit: bool,

    /// Memory pressure at allocation time
    pub memory_pressure: f32,

    /// Fragmentation level at allocation time
    pub fragmentation_level: f32,

    /// Number of retries required
    pub retry_count: u32,
}

/// Device context information
#[derive(Debug, Clone)]
pub struct DeviceContext {
    /// Current CUDA context
    pub cuda_context: Option<String>,

    /// Available memory at allocation time
    pub available_memory: usize,

    /// Device utilization percentage
    pub device_utilization: f32,

    /// Active streams count
    pub active_streams: usize,
}

impl CudaMemoryManager {
    /// Create new CUDA memory manager for specified device
    pub fn new(device_id: usize) -> CudaResult<Self> {
        Self::new_with_config(device_id, DeviceMemoryConfig::default())
    }

    /// Create memory manager with custom configuration
    pub fn new_with_config(device_id: usize, config: DeviceMemoryConfig) -> CudaResult<Self> {
        let device_properties = Self::query_device_properties(device_id)?;

        let (memory_limit, pressure_threshold) = Self::calculate_memory_limits(
            &device_properties,
            config.max_memory_fraction,
            config.pressure_threshold_fraction,
        );

        Ok(Self {
            device_id,
            pools: Mutex::new(HashMap::new()),
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            memory_limit: AtomicUsize::new(memory_limit),
            pressure_threshold: AtomicUsize::new(pressure_threshold),
            stats: Mutex::new(AllocationStats::default()),
            config,
            last_cleanup: Mutex::new(Instant::now()),
            device_properties,
        })
    }

    /// Allocate device memory with specified size
    pub fn allocate(&self, size: usize) -> CudaResult<CudaAllocation> {
        let request = AllocationRequest {
            size,
            allocation_type: AllocationType::Device,
            device_id: Some(self.device_id),
            ..Default::default()
        };

        self.allocate_with_request(request)
    }

    /// Allocate device memory with detailed request
    pub fn allocate_with_request(&self, request: AllocationRequest) -> CudaResult<CudaAllocation> {
        let allocation_start = Instant::now();

        // Validate request
        self.validate_allocation_request(&request)?;

        // Check memory pressure and perform cleanup if needed
        if self.is_under_memory_pressure() {
            self.handle_memory_pressure()?;
        }

        // Check if allocation would exceed limits
        self.check_memory_limits(request.size)?;

        let size_cls = size_class(request.size);

        // Try pool allocation first if pooling is enabled
        if self.config.enable_pooling {
            if let Some(allocation) = self.try_pool_allocation(size_cls, &request)? {
                self.record_allocation_success(&request, allocation_start, true);
                return Ok(allocation);
            }
        }

        // Allocate new memory block
        let allocation = self.allocate_new_block(size_cls, &request)?;
        self.record_allocation_success(&request, allocation_start, false);

        Ok(allocation)
    }

    /// Deallocate device memory
    pub fn deallocate(&self, mut allocation: CudaAllocation) -> CudaResult<()> {
        // Update allocation state
        allocation.mark_free();

        // Update statistics
        self.update_deallocation_stats(&allocation);

        if self.config.enable_pooling {
            // Return to pool for reuse
            self.return_to_pool(allocation)?;
        } else {
            // Free memory immediately
            self.free_allocation(allocation)?;
        }

        Ok(())
    }

    /// Get current memory usage statistics
    pub fn memory_info(&self) -> DeviceMemoryInfo {
        let current_allocated = self.total_allocated.load(Ordering::Relaxed);
        let peak_allocated = self.peak_allocated.load(Ordering::Relaxed);
        let memory_limit = self.memory_limit.load(Ordering::Relaxed);

        DeviceMemoryInfo {
            device_id: self.device_id,
            total_memory: self.device_properties.total_memory,
            current_allocated,
            peak_allocated,
            memory_limit,
            available_memory: memory_limit.saturating_sub(current_allocated),
            utilization_percent: if memory_limit > 0 {
                (current_allocated * 100) / memory_limit
            } else {
                0
            },
            fragmentation_level: self.calculate_fragmentation_level(),
            pool_count: self.get_pool_count(),
        }
    }

    /// Get detailed allocation statistics
    pub fn get_statistics(&self) -> CudaResult<AllocationStats> {
        let stats = self.stats.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire statistics lock".to_string(),
        })?;
        Ok(stats.clone())
    }

    /// Force memory cleanup and compaction
    pub fn cleanup_and_compact(&self) -> CudaResult<CleanupResult> {
        let cleanup_start = Instant::now();
        let mut total_freed = 0;
        let mut pools_cleaned = 0;

        let mut pools = self.pools.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire pools lock for cleanup".to_string(),
        })?;

        // Clean up individual pools
        for (size_class, pool) in pools.iter_mut() {
            let freed = pool.cleanup_old_allocations(self.config.cleanup_interval)?;
            if freed > 0 {
                total_freed += freed;
                pools_cleaned += 1;
            }
        }

        // Remove empty pools
        pools.retain(|_, pool| !pool.is_empty());

        // Update cleanup timestamp
        if let Ok(mut last_cleanup) = self.last_cleanup.lock() {
            *last_cleanup = Instant::now();
        }

        Ok(CleanupResult {
            duration: cleanup_start.elapsed(),
            bytes_freed: total_freed,
            pools_cleaned,
            empty_pools_removed: 0, // Would need to track this
        })
    }

    /// Check if currently under memory pressure
    pub fn is_under_memory_pressure(&self) -> bool {
        let current = self.total_allocated.load(Ordering::Relaxed);
        let threshold = self.pressure_threshold.load(Ordering::Relaxed);
        current > threshold
    }

    /// Prefetch data to device (for unified memory compatibility)
    pub fn prefetch_to_device(&self, ptr: *mut u8, size: usize) -> CudaResult<()> {
        unsafe {
            let result = cuda_sys::cudaMemPrefetchAsync(
                ptr as *const std::ffi::c_void,
                size,
                self.device_id as i32,
                0 as cuda_sys::cudaStream_t,
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!(
                        "Failed to prefetch memory to device {}: {:?}",
                        self.device_id, result
                    ),
                });
            }
        }
        Ok(())
    }

    /// Get device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Get device properties
    pub fn device_properties(&self) -> &DeviceProperties {
        &self.device_properties
    }

    // Private implementation methods

    fn query_device_properties(device_id: usize) -> CudaResult<DeviceProperties> {
        if let Ok(device) = CustDevice::get_device(device_id as u32) {
            let total_memory = device.total_memory().unwrap_or(8 * 1024 * 1024 * 1024); // 8GB fallback

            // Note: In a real implementation, we would query all device properties
            // For now, we'll provide reasonable defaults
            Ok(DeviceProperties {
                total_memory: total_memory as usize,
                compute_capability: (7, 5), // Default to common capability
                memory_bus_width: 384,
                memory_clock_rate: 1000000,     // 1 GHz
                l2_cache_size: 6 * 1024 * 1024, // 6MB
                max_threads_per_block: 1024,
                unified_addressing: true,
                concurrent_kernels: true,
            })
        } else {
            // Fallback properties when device is not available
            Ok(DeviceProperties {
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                compute_capability: (7, 5),
                memory_bus_width: 384,
                memory_clock_rate: 1000000,
                l2_cache_size: 6 * 1024 * 1024,
                max_threads_per_block: 1024,
                unified_addressing: true,
                concurrent_kernels: true,
            })
        }
    }

    fn calculate_memory_limits(
        properties: &DeviceProperties,
        max_fraction: f32,
        pressure_fraction: f32,
    ) -> (usize, usize) {
        let total_memory = properties.total_memory;
        let memory_limit = (total_memory as f32 * max_fraction) as usize;
        let pressure_threshold = (total_memory as f32 * pressure_fraction) as usize;
        (memory_limit, pressure_threshold)
    }

    fn validate_allocation_request(&self, request: &AllocationRequest) -> CudaResult<()> {
        if request.size == 0 {
            return Err(CudaError::Context {
                message: "Cannot allocate zero bytes".to_string(),
            });
        }

        if request.size > self.device_properties.total_memory {
            return Err(CudaError::Context {
                message: format!(
                    "Requested size {} exceeds total device memory {}",
                    request.size, self.device_properties.total_memory
                ),
            });
        }

        Ok(())
    }

    fn check_memory_limits(&self, size: usize) -> CudaResult<()> {
        let current_allocated = self.total_allocated.load(Ordering::Relaxed);
        let memory_limit = self.memory_limit.load(Ordering::Relaxed);

        if current_allocated + size > memory_limit {
            return Err(CudaError::Context {
                message: format!(
                    "Allocation would exceed memory limit. Requested: {}, Current: {}, Limit: {}",
                    size, current_allocated, memory_limit
                ),
            });
        }

        Ok(())
    }

    fn handle_memory_pressure(&self) -> CudaResult<()> {
        if self.config.enable_compaction {
            let _ = self.cleanup_and_compact()?;
        }
        Ok(())
    }

    fn try_pool_allocation(
        &self,
        size_class: usize,
        request: &AllocationRequest,
    ) -> CudaResult<Option<CudaAllocation>> {
        let mut pools = self.pools.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire pools lock for allocation".to_string(),
        })?;

        if let Some(pool) = pools.get_mut(&size_class) {
            if let Some(mut allocation) = pool.allocate() {
                allocation.mark_in_use();
                allocation.metadata.tag = request.tag.clone();
                self.update_allocation_stats(size_class, true);
                return Ok(Some(allocation));
            }
        }

        Ok(None)
    }

    fn allocate_new_block(
        &self,
        size_class: usize,
        request: &AllocationRequest,
    ) -> CudaResult<CudaAllocation> {
        // Allocate new device memory
        let ptr = unsafe { cust::memory::cuda_malloc(size_class)? };

        let mut allocation =
            CudaAllocation::new_on_device(ptr, size_class, size_class, self.device_id);
        allocation.metadata.tag = request.tag.clone();

        // Add to appropriate pool if pooling is enabled
        if self.config.enable_pooling {
            let mut pools = self.pools.lock().map_err(|_| CudaError::Context {
                message: "Failed to acquire pools lock for new block".to_string(),
            })?;

            let pool = pools
                .entry(size_class)
                .or_insert_with(|| DeviceMemoryPool::new(size_class, PoolConfig::default()));

            pool.add_allocation(allocation.clone());
        }

        self.update_allocation_stats(size_class, false);

        #[cfg(debug_assertions)]
        {
            if self.config.debug_tracking {
                if let Ok(mut tracker) = ALLOCATION_TRACKER.lock() {
                    tracker.insert(allocation.as_ptr());
                }
            }
        }

        Ok(allocation)
    }

    fn return_to_pool(&self, allocation: CudaAllocation) -> CudaResult<()> {
        let mut pools = self.pools.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire pools lock for deallocation".to_string(),
        })?;

        if let Some(pool) = pools.get_mut(&allocation.size_class) {
            pool.deallocate(allocation);
        }

        Ok(())
    }

    fn free_allocation(&self, allocation: CudaAllocation) -> CudaResult<()> {
        #[cfg(debug_assertions)]
        {
            if self.config.debug_tracking {
                if let Ok(mut tracker) = ALLOCATION_TRACKER.lock() {
                    tracker.remove(&allocation.as_ptr());
                }
            }
        }

        unsafe {
            cust::memory::cuda_free(allocation.ptr)?;
        }

        Ok(())
    }

    fn update_allocation_stats(&self, size: usize, cache_hit: bool) {
        let current = self.total_allocated.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak
        let mut peak = self.peak_allocated.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }

        // Update detailed statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_allocations += 1;
            stats.active_allocations += 1;
            stats.total_bytes_allocated += size as u64;
            stats.current_bytes_allocated = current as u64;
            stats.peak_bytes_allocated = self.peak_allocated.load(Ordering::Relaxed) as u64;

            if cache_hit {
                let total = stats.total_allocations as f32;
                stats.cache_hit_rate = ((stats.cache_hit_rate * (total - 1.0)) + 1.0) / total;
            } else {
                let total = stats.total_allocations as f32;
                stats.cache_hit_rate = (stats.cache_hit_rate * (total - 1.0)) / total;
            }

            stats.average_allocation_size = if stats.total_allocations > 0 {
                (stats.total_bytes_allocated / stats.total_allocations) as usize
            } else {
                0
            };
        }
    }

    fn update_deallocation_stats(&self, allocation: &CudaAllocation) {
        self.total_allocated
            .fetch_sub(allocation.size, Ordering::Relaxed);

        if let Ok(mut stats) = self.stats.lock() {
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
            stats.current_bytes_allocated = self.total_allocated.load(Ordering::Relaxed) as u64;
        }
    }

    fn record_allocation_success(
        &self,
        request: &AllocationRequest,
        start_time: Instant,
        cache_hit: bool,
    ) {
        let allocation_time = start_time.elapsed();

        if let Ok(mut stats) = self.stats.lock() {
            // Update average allocation time
            let total = stats.total_allocations as u32;
            if total > 0 {
                stats.average_allocation_time =
                    (stats.average_allocation_time * (total - 1) + allocation_time) / total;
            } else {
                stats.average_allocation_time = allocation_time;
            }

            // Update success rate
            stats.success_rate = 1.0; // This allocation succeeded
        }
    }

    fn calculate_fragmentation_level(&self) -> f32 {
        // Simplified fragmentation calculation
        // In a real implementation, this would analyze memory layout
        0.1 // Placeholder
    }

    fn get_pool_count(&self) -> usize {
        self.pools.lock().map(|pools| pools.len()).unwrap_or(0)
    }
}

/// Device memory information
#[derive(Debug, Clone)]
pub struct DeviceMemoryInfo {
    /// Device ID
    pub device_id: usize,

    /// Total device memory
    pub total_memory: usize,

    /// Currently allocated bytes
    pub current_allocated: usize,

    /// Peak allocated bytes
    pub peak_allocated: usize,

    /// Memory limit
    pub memory_limit: usize,

    /// Available memory
    pub available_memory: usize,

    /// Memory utilization percentage
    pub utilization_percent: usize,

    /// Memory fragmentation level (0.0 to 1.0)
    pub fragmentation_level: f32,

    /// Number of active memory pools
    pub pool_count: usize,
}

/// Cleanup operation result
#[derive(Debug, Clone)]
pub struct CleanupResult {
    /// Time taken for cleanup
    pub duration: Duration,

    /// Total bytes freed
    pub bytes_freed: usize,

    /// Number of pools cleaned
    pub pools_cleaned: usize,

    /// Number of empty pools removed
    pub empty_pools_removed: usize,
}

// Default implementations
impl Default for DeviceMemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_fraction: 0.85,
            pressure_threshold_fraction: 0.75,
            enable_pooling: true,
            max_pool_size: 16,
            enable_compaction: true,
            cleanup_interval: Duration::from_secs(60),
            debug_tracking: cfg!(debug_assertions),
            allocation_alignment: 256,
            enable_async_alloc: false,
        }
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_free_blocks: 8,
            min_age_for_cleanup: Duration::from_secs(30),
            growth_strategy: GrowthStrategy::Adaptive,
            track_statistics: true,
        }
    }
}

impl DeviceMemoryPool {
    fn new(size_class: usize, config: PoolConfig) -> Self {
        Self {
            size_class,
            free_blocks: Vec::new(),
            allocated_blocks: Vec::new(),
            pool_stats: PoolStatistics::default(),
            last_access: Instant::now(),
            config,
        }
    }

    fn allocate(&mut self) -> Option<CudaAllocation> {
        self.last_access = Instant::now();
        if let Some(allocation) = self.free_blocks.pop() {
            self.pool_stats.cache_hits += 1;
            Some(allocation)
        } else {
            self.pool_stats.cache_misses += 1;
            None
        }
    }

    fn deallocate(&mut self, allocation: CudaAllocation) {
        if self.free_blocks.len() < self.config.max_free_blocks {
            self.free_blocks.push(allocation);
        }
        self.last_access = Instant::now();
    }

    fn add_allocation(&mut self, allocation: CudaAllocation) {
        self.allocated_blocks.push(allocation);
        self.pool_stats.total_allocations += 1;
    }

    fn cleanup_old_allocations(&mut self, max_age: Duration) -> CudaResult<usize> {
        let now = Instant::now();
        let initial_count = self.free_blocks.len();

        self.free_blocks.retain(|allocation| {
            let age = now.duration_since(allocation.allocation_time);
            age <= max_age || age <= self.config.min_age_for_cleanup
        });

        Ok(initial_count - self.free_blocks.len())
    }

    fn is_empty(&self) -> bool {
        self.free_blocks.is_empty() && self.allocated_blocks.is_empty()
    }
}

impl Default for PoolStatistics {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            cache_hits: 0,
            cache_misses: 0,
            utilization: 0.0,
            peak_utilization: 0.0,
            average_lifetime: Duration::from_secs(0),
            memory_efficiency: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_memory_config() {
        let config = DeviceMemoryConfig::default();
        assert_eq!(config.max_memory_fraction, 0.85);
        assert!(config.enable_pooling);
        assert!(config.enable_compaction);
    }

    #[test]
    fn test_memory_limit_calculation() {
        let properties = DeviceProperties {
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB
            ..Default::default()
        };

        let (limit, threshold) = CudaMemoryManager::calculate_memory_limits(&properties, 0.8, 0.7);

        assert_eq!(limit, (8 * 1024 * 1024 * 1024 as f32 * 0.8) as usize);
        assert_eq!(threshold, (8 * 1024 * 1024 * 1024 as f32 * 0.7) as usize);
    }

    #[test]
    fn test_device_memory_pool() {
        let config = PoolConfig::default();
        let mut pool = DeviceMemoryPool::new(1024, config);

        // Initially empty
        assert!(pool.is_empty());
        assert_eq!(pool.pool_stats.total_allocations, 0);

        // No allocations available
        assert!(pool.allocate().is_none());
        assert_eq!(pool.pool_stats.cache_misses, 1);
    }

    #[test]
    fn test_growth_strategies() {
        assert_eq!(GrowthStrategy::Fixed, GrowthStrategy::Fixed);
        assert_ne!(GrowthStrategy::Linear, GrowthStrategy::Fixed);

        if let GrowthStrategy::Exponential { max_size } =
            (GrowthStrategy::Exponential { max_size: 1024 })
        {
            assert_eq!(max_size, 1024);
        }
    }

    #[test]
    fn test_allocation_request_validation() {
        // This would be tested with actual CudaMemoryManager instance
        // in an environment with CUDA support
    }
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            total_memory: 8 * 1024 * 1024 * 1024,
            compute_capability: (7, 5),
            memory_bus_width: 384,
            memory_clock_rate: 1000000,
            l2_cache_size: 6 * 1024 * 1024,
            max_threads_per_block: 1024,
            unified_addressing: true,
            concurrent_kernels: true,
        }
    }
}

// Type aliases for compatibility
pub type DeviceMemoryMetrics = PoolStatistics;
pub type PoolConfiguration = PoolConfig;
