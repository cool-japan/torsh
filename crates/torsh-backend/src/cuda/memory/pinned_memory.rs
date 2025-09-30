//! CUDA pinned (page-locked) memory management
//!
//! This module provides comprehensive management of CUDA pinned memory allocations
//! that enable efficient data transfers between host and device by avoiding
//! page faults and providing direct memory access.

use super::allocation::{
    pinned_size_class, AllocationMetadata, AllocationRequest, AllocationStats, AllocationType,
    PinnedAllocation, PinnedMemoryFlags,
};
use crate::error::{CudaError, CudaResult};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

/// Global pinned memory manager instances per device
static PINNED_MANAGERS: once_cell::sync::Lazy<Mutex<HashMap<usize, Arc<PinnedMemoryManager>>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// CUDA pinned memory manager with advanced pooling and optimization
///
/// Manages page-locked host memory for efficient host-device data transfers.
/// Provides pooling, automatic cleanup, and performance optimization.
#[derive(Debug)]
pub struct PinnedMemoryManager {
    /// Device ID this manager is associated with
    device_id: usize,

    /// Memory pools organized by size class
    pools: Mutex<HashMap<usize, PinnedMemoryPool>>,

    /// Total allocated pinned memory
    total_pinned_memory: AtomicUsize,

    /// Peak pinned memory usage
    peak_pinned_memory: AtomicUsize,

    /// Pinned memory limit
    pinned_memory_limit: AtomicUsize,

    /// Allocation statistics
    allocation_stats: Mutex<PinnedAllocationStats>,

    /// Configuration settings
    config: PinnedMemoryConfig,

    /// Last cleanup operation time
    last_cleanup: Mutex<Instant>,

    /// Transfer performance metrics
    transfer_metrics: Mutex<TransferMetrics>,
}

/// Pinned memory pool for specific size class
#[derive(Debug)]
pub struct PinnedMemoryPool {
    /// Size class (power of 2, minimum 4KB)
    size_class: usize,

    /// Available allocations for reuse
    free_blocks: Vec<PinnedAllocation>,

    /// Currently allocated blocks
    allocated_blocks: Vec<PinnedAllocation>,

    /// Pool statistics
    total_allocations: usize,

    /// Peak usage count
    peak_usage: usize,

    /// Cache hit count
    cache_hits: usize,

    /// Cache miss count
    cache_misses: usize,

    /// Last access timestamp
    last_access: Instant,
}

/// Configuration for pinned memory management
#[derive(Debug, Clone)]
pub struct PinnedMemoryConfig {
    /// Maximum pinned memory to allocate (bytes)
    pub max_pinned_memory: usize,

    /// Maximum age for cached allocations
    pub max_cache_age: Duration,

    /// Enable automatic cleanup of old allocations
    pub enable_auto_cleanup: bool,

    /// Cleanup check interval
    pub cleanup_interval: Duration,

    /// Maximum free blocks to keep per pool
    pub max_free_blocks_per_pool: usize,

    /// Enable device mapping for pinned allocations
    pub enable_device_mapping: bool,

    /// Enable portable memory (accessible from all contexts)
    pub enable_portable_memory: bool,

    /// Enable write-combining optimization
    pub enable_write_combining: bool,

    /// Enable transfer performance tracking
    pub enable_transfer_tracking: bool,

    /// Preferred memory alignment
    pub memory_alignment: usize,
}

/// Pinned memory allocation request
#[derive(Debug, Clone)]
pub struct PinnedMemoryRequest {
    /// Requested size in bytes
    pub size: usize,

    /// Enable device mapping
    pub enable_mapping: bool,

    /// Pinned memory flags
    pub flags: PinnedMemoryFlags,

    /// Memory alignment requirement
    pub alignment: Option<usize>,

    /// Optional tag for debugging
    pub tag: Option<String>,

    /// Expected usage pattern
    pub usage_pattern: UsagePattern,

    /// Priority level
    pub priority: AllocationPriority,
}

/// Expected usage patterns for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsagePattern {
    /// Frequent host-to-device transfers
    HostToDevice,
    /// Frequent device-to-host transfers
    DeviceToHost,
    /// Bidirectional transfers
    Bidirectional,
    /// Temporary staging area
    Staging,
    /// Long-lived buffer
    Persistent,
}

/// Allocation priorities for pinned memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AllocationPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Extended statistics for pinned memory
#[derive(Debug, Clone)]
pub struct PinnedAllocationStats {
    /// Base allocation statistics
    pub base_stats: AllocationStats,

    /// Pinned-specific metrics
    pub mapping_success_rate: f32,

    /// Average allocation age at deallocation
    pub average_allocation_age: Duration,

    /// Total mapped allocations
    pub total_mapped_allocations: u64,

    /// Current mapped allocations
    pub current_mapped_allocations: u64,

    /// Memory pressure events
    pub memory_pressure_events: u64,

    /// Cleanup operations performed
    pub cleanup_operations: u64,

    /// Bytes freed by cleanup
    pub cleanup_bytes_freed: u64,
}

/// Data transfer performance metrics
#[derive(Debug, Clone)]
pub struct TransferMetrics {
    /// Host-to-device transfer statistics
    pub host_to_device: TransferStats,

    /// Device-to-host transfer statistics
    pub device_to_host: TransferStats,

    /// Bidirectional transfer statistics
    pub bidirectional: TransferStats,

    /// Overall transfer efficiency
    pub overall_efficiency: f32,

    /// Peak transfer bandwidth achieved
    pub peak_bandwidth: f64,

    /// Average transfer latency
    pub average_latency: Duration,
}

/// Statistics for specific transfer direction
#[derive(Debug, Clone)]
pub struct TransferStats {
    /// Total number of transfers
    pub total_transfers: u64,

    /// Total bytes transferred
    pub total_bytes: u64,

    /// Total transfer time
    pub total_time: Duration,

    /// Average bandwidth (bytes per second)
    pub average_bandwidth: f64,

    /// Peak bandwidth achieved
    pub peak_bandwidth: f64,

    /// Minimum bandwidth (for detecting issues)
    pub min_bandwidth: f64,

    /// Transfer efficiency (actual vs theoretical)
    pub efficiency: f32,
}

/// Information about current pinned memory usage
#[derive(Debug, Clone)]
pub struct PinnedMemoryInfo {
    /// Device ID
    pub device_id: usize,

    /// Currently allocated pinned memory
    pub current_allocated: usize,

    /// Peak allocated pinned memory
    pub peak_allocated: usize,

    /// Memory limit for pinned allocations
    pub memory_limit: usize,

    /// Utilization percentage
    pub utilization_percent: usize,

    /// Number of active pools
    pub active_pools: usize,

    /// Total cached allocations
    pub cached_allocations: usize,

    /// Memory fragmentation level
    pub fragmentation_level: f32,
}

/// Cleanup operation result
#[derive(Debug, Clone)]
pub struct PinnedCleanupResult {
    /// Number of allocations freed
    pub allocations_freed: usize,

    /// Total bytes freed
    pub bytes_freed: usize,

    /// Number of pools cleaned
    pub pools_cleaned: usize,

    /// Cleanup duration
    pub duration: Duration,

    /// Success status
    pub success: bool,
}

impl PinnedMemoryManager {
    /// Create new pinned memory manager
    pub fn new(config: PinnedMemoryConfig) -> CudaResult<Self> {
        Self::new_for_device(0, config)
    }

    /// Create pinned memory manager for specific device
    pub fn new_for_device(device_id: usize, config: PinnedMemoryConfig) -> CudaResult<Self> {
        Ok(Self {
            device_id,
            pools: Mutex::new(HashMap::new()),
            total_pinned_memory: AtomicUsize::new(0),
            peak_pinned_memory: AtomicUsize::new(0),
            pinned_memory_limit: AtomicUsize::new(config.max_pinned_memory),
            allocation_stats: Mutex::new(PinnedAllocationStats::default()),
            config,
            last_cleanup: Mutex::new(Instant::now()),
            transfer_metrics: Mutex::new(TransferMetrics::default()),
        })
    }

    /// Allocate pinned memory
    pub fn allocate_pinned(&self, request: PinnedMemoryRequest) -> CudaResult<PinnedAllocation> {
        let allocation_start = Instant::now();

        // Validate request
        self.validate_request(&request)?;

        // Check memory limits
        self.check_memory_limits(request.size)?;

        let size_class = pinned_size_class(request.size);

        // Try pool allocation first
        if let Some(allocation) = self.try_pool_allocation(size_class, &request)? {
            self.record_allocation_success(size_class, allocation_start, true);
            return Ok(allocation);
        }

        // Allocate new pinned memory
        let allocation = self.allocate_new_pinned_block(size_class, &request)?;
        self.record_allocation_success(size_class, allocation_start, false);

        Ok(allocation)
    }

    /// Deallocate pinned memory
    pub fn deallocate_pinned(&self, allocation: PinnedAllocation) -> CudaResult<()> {
        let size = allocation.size;

        // Update statistics
        self.update_deallocation_stats(&allocation);

        // Return to pool if within limits
        if self.should_cache_allocation(&allocation) {
            self.return_to_pool(allocation)?;
        } else {
            self.free_pinned_allocation(allocation)?;
        }

        Ok(())
    }

    /// Get current pinned memory information
    pub fn info(&self) -> PinnedMemoryInfo {
        let current_allocated = self.total_pinned_memory.load(Ordering::Relaxed);
        let peak_allocated = self.peak_pinned_memory.load(Ordering::Relaxed);
        let memory_limit = self.pinned_memory_limit.load(Ordering::Relaxed);

        let active_pools = self.pools.lock().map(|pools| pools.len()).unwrap_or(0);
        let cached_allocations = self.get_cached_allocation_count();

        PinnedMemoryInfo {
            device_id: self.device_id,
            current_allocated,
            peak_allocated,
            memory_limit,
            utilization_percent: if memory_limit > 0 {
                (current_allocated * 100) / memory_limit
            } else {
                0
            },
            active_pools,
            cached_allocations,
            fragmentation_level: self.calculate_fragmentation_level(),
        }
    }

    /// Get detailed allocation statistics
    pub fn stats(&self) -> CudaResult<PinnedAllocationStats> {
        let stats = self
            .allocation_stats
            .lock()
            .map_err(|_| CudaError::Context {
                message: "Failed to acquire statistics lock".to_string(),
            })?;
        Ok(stats.clone())
    }

    /// Force cleanup of old cached allocations
    pub fn cleanup(&self) -> CudaResult<PinnedCleanupResult> {
        let cleanup_start = Instant::now();
        let mut allocations_freed = 0;
        let mut bytes_freed = 0;
        let mut pools_cleaned = 0;

        let mut pools = self.pools.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire pools lock for cleanup".to_string(),
        })?;

        for (_, pool) in pools.iter_mut() {
            let result = pool.cleanup_old_allocations(Instant::now(), self.config.max_cache_age)?;

            if result.allocations_freed > 0 {
                allocations_freed += result.allocations_freed;
                bytes_freed += result.bytes_freed;
                pools_cleaned += 1;
            }
        }

        // Remove empty pools
        pools.retain(|_, pool| !pool.is_empty());

        // Update cleanup timestamp
        if let Ok(mut last_cleanup) = self.last_cleanup.lock() {
            *last_cleanup = Instant::now();
        }

        // Update statistics
        if let Ok(mut stats) = self.allocation_stats.lock() {
            stats.cleanup_operations += 1;
            stats.cleanup_bytes_freed += bytes_freed as u64;
        }

        Ok(PinnedCleanupResult {
            allocations_freed,
            bytes_freed,
            pools_cleaned,
            duration: cleanup_start.elapsed(),
            success: true,
        })
    }

    /// Record data transfer performance
    pub fn record_transfer(
        &self,
        direction: TransferDirection,
        bytes: usize,
        duration: Duration,
    ) -> CudaResult<()> {
        if !self.config.enable_transfer_tracking {
            return Ok(());
        }

        let mut metrics = self
            .transfer_metrics
            .lock()
            .map_err(|_| CudaError::Context {
                message: "Failed to acquire transfer metrics lock".to_string(),
            })?;

        let bandwidth = bytes as f64 / duration.as_secs_f64();

        match direction {
            TransferDirection::HostToDevice => {
                metrics
                    .host_to_device
                    .update_stats(bytes, duration, bandwidth);
            }
            TransferDirection::DeviceToHost => {
                metrics
                    .device_to_host
                    .update_stats(bytes, duration, bandwidth);
            }
            TransferDirection::Bidirectional => {
                metrics
                    .bidirectional
                    .update_stats(bytes, duration, bandwidth);
            }
        }

        // Update overall metrics
        if bandwidth > metrics.peak_bandwidth {
            metrics.peak_bandwidth = bandwidth;
        }

        let total_transfers = metrics.host_to_device.total_transfers
            + metrics.device_to_host.total_transfers
            + metrics.bidirectional.total_transfers;

        if total_transfers > 0 {
            let total_time = metrics.host_to_device.total_time
                + metrics.device_to_host.total_time
                + metrics.bidirectional.total_time;

            metrics.average_latency = total_time / total_transfers as u32;

            // Update overall efficiency (simplified calculation)
            metrics.overall_efficiency = (bandwidth / metrics.peak_bandwidth) as f32;
        }

        Ok(())
    }

    /// Get transfer performance metrics
    pub fn get_transfer_metrics(&self) -> CudaResult<TransferMetrics> {
        let metrics = self
            .transfer_metrics
            .lock()
            .map_err(|_| CudaError::Context {
                message: "Failed to acquire transfer metrics lock".to_string(),
            })?;
        Ok(metrics.clone())
    }

    /// Check if automatic cleanup should run
    pub fn should_run_cleanup(&self) -> bool {
        if !self.config.enable_auto_cleanup {
            return false;
        }

        if let Ok(last_cleanup) = self.last_cleanup.lock() {
            let age = Instant::now().duration_since(*last_cleanup);
            age >= self.config.cleanup_interval
        } else {
            false
        }
    }

    // Private implementation methods

    fn validate_request(&self, request: &PinnedMemoryRequest) -> CudaResult<()> {
        if request.size == 0 {
            return Err(CudaError::Context {
                message: "Cannot allocate zero bytes".to_string(),
            });
        }

        if request.size > self.config.max_pinned_memory {
            return Err(CudaError::Context {
                message: format!(
                    "Requested size {} exceeds maximum pinned memory {}",
                    request.size, self.config.max_pinned_memory
                ),
            });
        }

        Ok(())
    }

    fn check_memory_limits(&self, size: usize) -> CudaResult<()> {
        let current = self.total_pinned_memory.load(Ordering::Relaxed);
        let limit = self.pinned_memory_limit.load(Ordering::Relaxed);

        if current + size > limit {
            // Try cleanup first
            if self.config.enable_auto_cleanup {
                let _ = self.cleanup();

                // Check again after cleanup
                let current_after_cleanup = self.total_pinned_memory.load(Ordering::Relaxed);
                if current_after_cleanup + size > limit {
                    return Err(CudaError::Context {
                        message: format!(
                            "Pinned memory allocation would exceed limit. Requested: {}, Current: {}, Limit: {}",
                            size, current_after_cleanup, limit
                        ),
                    });
                }
            } else {
                return Err(CudaError::Context {
                    message: format!(
                        "Pinned memory allocation would exceed limit. Requested: {}, Current: {}, Limit: {}",
                        size, current, limit
                    ),
                });
            }
        }

        Ok(())
    }

    fn try_pool_allocation(
        &self,
        size_class: usize,
        request: &PinnedMemoryRequest,
    ) -> CudaResult<Option<PinnedAllocation>> {
        let mut pools = self.pools.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire pools lock".to_string(),
        })?;

        if let Some(pool) = pools.get_mut(&size_class) {
            if let Some(mut allocation) = pool.allocate() {
                // Update allocation metadata
                allocation.increment_usage();
                if let Some(tag) = &request.tag {
                    allocation.metadata.tag = Some(tag.clone());
                }

                return Ok(Some(allocation));
            }
        }

        Ok(None)
    }

    fn allocate_new_pinned_block(
        &self,
        size_class: usize,
        request: &PinnedMemoryRequest,
    ) -> CudaResult<PinnedAllocation> {
        let flags = self.calculate_cuda_flags(&request.flags);
        let ptr = self.allocate_cuda_pinned_memory(size_class, flags)?;

        let device_ptr = if request.enable_mapping || self.config.enable_device_mapping {
            self.map_pinned_memory_to_device(ptr, size_class)?
        } else {
            None
        };

        let mut allocation =
            PinnedAllocation::new_with_mapping(ptr, size_class, device_ptr, request.flags);

        if let Some(tag) = &request.tag {
            allocation.metadata.tag = Some(tag.clone());
        }

        // Add to pool for future reuse
        let mut pools = self.pools.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire pools lock".to_string(),
        })?;

        let pool = pools
            .entry(size_class)
            .or_insert_with(|| PinnedMemoryPool::new(size_class));

        pool.add_allocation(allocation.clone());

        // Update memory tracking
        self.update_allocation_stats(size_class);

        Ok(allocation)
    }

    fn calculate_cuda_flags(&self, flags: &PinnedMemoryFlags) -> u32 {
        let mut cuda_flags = 0u32;

        if flags.portable || self.config.enable_portable_memory {
            cuda_flags |= cuda_sys::cudaHostAllocPortable;
        }

        if flags.write_combining || self.config.enable_write_combining {
            cuda_flags |= cuda_sys::cudaHostAllocWriteCombined;
        }

        if flags.enable_mapping || self.config.enable_device_mapping {
            cuda_flags |= cuda_sys::cudaHostAllocMapped;
        }

        cuda_flags | flags.raw_flags
    }

    fn allocate_cuda_pinned_memory(&self, size: usize, flags: u32) -> CudaResult<*mut u8> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        unsafe {
            let result =
                cuda_sys::cudaHostAlloc(&mut ptr as *mut *mut std::ffi::c_void, size, flags);

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to allocate pinned memory: {:?}", result),
                });
            }
        }

        Ok(ptr as *mut u8)
    }

    fn map_pinned_memory_to_device(
        &self,
        host_ptr: *mut u8,
        size: usize,
    ) -> CudaResult<Option<cust::DevicePointer<u8>>> {
        let mut device_ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        unsafe {
            let result = cuda_sys::cudaHostGetDevicePointer(
                &mut device_ptr as *mut *mut std::ffi::c_void,
                host_ptr as *mut std::ffi::c_void,
                0, // flags
            );

            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to map pinned memory to device: {:?}", result),
                });
            }
        }

        if device_ptr.is_null() {
            Ok(None)
        } else {
            // Convert to cust::DevicePointer
            // This is a simplified conversion - in practice, we'd need proper handling
            Ok(Some(unsafe {
                cust::DevicePointer::wrap(device_ptr as *mut u8)
            }))
        }
    }

    fn should_cache_allocation(&self, allocation: &PinnedAllocation) -> bool {
        let pools = self.pools.lock().unwrap();
        if let Some(pool) = pools.get(&allocation.size) {
            pool.free_blocks.len() < self.config.max_free_blocks_per_pool
        } else {
            true // Always cache if pool doesn't exist yet
        }
    }

    fn return_to_pool(&self, allocation: PinnedAllocation) -> CudaResult<()> {
        let mut pools = self.pools.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire pools lock".to_string(),
        })?;

        if let Some(pool) = pools.get_mut(&allocation.size) {
            pool.deallocate(allocation);
        }

        Ok(())
    }

    fn free_pinned_allocation(&self, allocation: PinnedAllocation) -> CudaResult<()> {
        unsafe {
            let result = cuda_sys::cudaFreeHost(allocation.ptr as *mut std::ffi::c_void);
            if result != cuda_sys::cudaError_t::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!("Failed to free pinned memory: {:?}", result),
                });
            }
        }

        Ok(())
    }

    fn update_allocation_stats(&self, size: usize) {
        let current = self.total_pinned_memory.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak
        let mut peak = self.peak_pinned_memory.load(Ordering::Relaxed);
        while current > peak {
            match self.peak_pinned_memory.compare_exchange_weak(
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
        if let Ok(mut stats) = self.allocation_stats.lock() {
            stats.base_stats.total_allocations += 1;
            stats.base_stats.active_allocations += 1;
            stats.base_stats.total_bytes_allocated += size as u64;
            stats.base_stats.current_bytes_allocated = current as u64;
            stats.base_stats.peak_bytes_allocated = peak as u64;
        }
    }

    fn update_deallocation_stats(&self, allocation: &PinnedAllocation) {
        self.total_pinned_memory
            .fetch_sub(allocation.size, Ordering::Relaxed);

        if let Ok(mut stats) = self.allocation_stats.lock() {
            stats.base_stats.active_allocations =
                stats.base_stats.active_allocations.saturating_sub(1);
            stats.base_stats.current_bytes_allocated =
                self.total_pinned_memory.load(Ordering::Relaxed) as u64;

            // Update average age
            let age = allocation.age();
            let total_deallocations =
                stats.base_stats.total_allocations - stats.base_stats.active_allocations;

            if total_deallocations > 0 {
                let total_age =
                    stats.average_allocation_age * (total_deallocations - 1) as u32 + age;
                stats.average_allocation_age = total_age / total_deallocations as u32;
            }
        }
    }

    fn record_allocation_success(&self, size: usize, start_time: Instant, cache_hit: bool) {
        let allocation_time = start_time.elapsed();

        if let Ok(mut stats) = self.allocation_stats.lock() {
            // Update cache hit rate
            let total = stats.base_stats.total_allocations as f32;
            if total > 0.0 {
                if cache_hit {
                    stats.base_stats.cache_hit_rate =
                        ((stats.base_stats.cache_hit_rate * (total - 1.0)) + 1.0) / total;
                } else {
                    stats.base_stats.cache_hit_rate =
                        (stats.base_stats.cache_hit_rate * (total - 1.0)) / total;
                }
            }

            // Update average allocation time
            if total > 0.0 {
                stats.base_stats.average_allocation_time =
                    (stats.base_stats.average_allocation_time * (total - 1.0) as u32
                        + allocation_time)
                        / total as u32;
            }
        }
    }

    fn get_cached_allocation_count(&self) -> usize {
        self.pools
            .lock()
            .map(|pools| pools.values().map(|pool| pool.free_blocks.len()).sum())
            .unwrap_or(0)
    }

    fn calculate_fragmentation_level(&self) -> f32 {
        // Simplified fragmentation calculation
        // In practice, this would analyze pool distributions and allocation patterns
        0.05 // 5% fragmentation placeholder
    }
}

/// Data transfer directions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    Bidirectional,
}

/// Pool cleanup result for individual pools
#[derive(Debug, Clone)]
pub struct PoolCleanupResult {
    /// Number of allocations freed
    pub allocations_freed: usize,

    /// Bytes freed
    pub bytes_freed: usize,
}

impl PinnedMemoryPool {
    fn new(size_class: usize) -> Self {
        Self {
            size_class,
            free_blocks: Vec::new(),
            allocated_blocks: Vec::new(),
            total_allocations: 0,
            peak_usage: 0,
            cache_hits: 0,
            cache_misses: 0,
            last_access: Instant::now(),
        }
    }

    fn allocate(&mut self) -> Option<PinnedAllocation> {
        self.last_access = Instant::now();
        if let Some(allocation) = self.free_blocks.pop() {
            self.cache_hits += 1;
            Some(allocation)
        } else {
            self.cache_misses += 1;
            None
        }
    }

    fn deallocate(&mut self, allocation: PinnedAllocation) {
        self.free_blocks.push(allocation);
        self.last_access = Instant::now();
    }

    fn add_allocation(&mut self, allocation: PinnedAllocation) {
        self.allocated_blocks.push(allocation);
        self.total_allocations += 1;
        self.peak_usage = self.peak_usage.max(self.allocated_blocks.len());
    }

    fn cleanup_old_allocations(
        &mut self,
        now: Instant,
        max_age: Duration,
    ) -> CudaResult<PoolCleanupResult> {
        let initial_count = self.free_blocks.len();
        let mut bytes_freed = 0;

        self.free_blocks.retain(|allocation| {
            let age = now.duration_since(allocation.allocation_time);
            if age > max_age {
                bytes_freed += allocation.size;
                // Free the pinned memory
                unsafe {
                    let result = cuda_sys::cudaFreeHost(allocation.ptr as *mut std::ffi::c_void);
                    if result != cuda_sys::cudaError_t::cudaSuccess {
                        eprintln!(
                            "Warning: Failed to free pinned memory during cleanup: {:?}",
                            result
                        );
                    }
                }
                false
            } else {
                true
            }
        });

        Ok(PoolCleanupResult {
            allocations_freed: initial_count - self.free_blocks.len(),
            bytes_freed,
        })
    }

    fn is_empty(&self) -> bool {
        self.free_blocks.is_empty() && self.allocated_blocks.is_empty()
    }
}

impl TransferStats {
    fn update_stats(&mut self, bytes: usize, duration: Duration, bandwidth: f64) {
        self.total_transfers += 1;
        self.total_bytes += bytes as u64;
        self.total_time += duration;

        // Update bandwidth statistics
        if bandwidth > self.peak_bandwidth {
            self.peak_bandwidth = bandwidth;
        }

        if self.min_bandwidth == 0.0 || bandwidth < self.min_bandwidth {
            self.min_bandwidth = bandwidth;
        }

        // Recalculate average bandwidth
        if self.total_time.as_secs_f64() > 0.0 {
            self.average_bandwidth = self.total_bytes as f64 / self.total_time.as_secs_f64();
        }

        // Update efficiency (actual vs peak)
        if self.peak_bandwidth > 0.0 {
            self.efficiency = (self.average_bandwidth / self.peak_bandwidth) as f32;
        }
    }
}

/// Get global pinned memory manager for device
pub fn get_pinned_memory_manager(
    device_id: usize,
    config: Option<PinnedMemoryConfig>,
) -> CudaResult<Arc<PinnedMemoryManager>> {
    let mut managers = PINNED_MANAGERS.lock().map_err(|_| CudaError::Context {
        message: "Failed to acquire global managers lock".to_string(),
    })?;

    if let Some(manager) = managers.get(&device_id) {
        Ok(Arc::clone(manager))
    } else {
        let config = config.unwrap_or_default();
        let manager = Arc::new(PinnedMemoryManager::new_for_device(device_id, config)?);
        managers.insert(device_id, Arc::clone(&manager));
        Ok(manager)
    }
}

// Default implementations
impl Default for PinnedMemoryConfig {
    fn default() -> Self {
        Self {
            max_pinned_memory: 512 * 1024 * 1024,    // 512MB default
            max_cache_age: Duration::from_secs(300), // 5 minutes
            enable_auto_cleanup: true,
            cleanup_interval: Duration::from_secs(60), // 1 minute
            max_free_blocks_per_pool: 8,
            enable_device_mapping: false,
            enable_portable_memory: false,
            enable_write_combining: false,
            enable_transfer_tracking: true,
            memory_alignment: 256,
        }
    }
}

impl Default for PinnedAllocationStats {
    fn default() -> Self {
        Self {
            base_stats: AllocationStats::default(),
            mapping_success_rate: 1.0,
            average_allocation_age: Duration::from_secs(0),
            total_mapped_allocations: 0,
            current_mapped_allocations: 0,
            memory_pressure_events: 0,
            cleanup_operations: 0,
            cleanup_bytes_freed: 0,
        }
    }
}

impl Default for TransferMetrics {
    fn default() -> Self {
        Self {
            host_to_device: TransferStats::default(),
            device_to_host: TransferStats::default(),
            bidirectional: TransferStats::default(),
            overall_efficiency: 0.0,
            peak_bandwidth: 0.0,
            average_latency: Duration::from_secs(0),
        }
    }
}

impl Default for TransferStats {
    fn default() -> Self {
        Self {
            total_transfers: 0,
            total_bytes: 0,
            total_time: Duration::from_secs(0),
            average_bandwidth: 0.0,
            peak_bandwidth: 0.0,
            min_bandwidth: 0.0,
            efficiency: 0.0,
        }
    }
}

impl Default for PinnedMemoryRequest {
    fn default() -> Self {
        Self {
            size: 0,
            enable_mapping: false,
            flags: PinnedMemoryFlags::default(),
            alignment: None,
            tag: None,
            usage_pattern: UsagePattern::Bidirectional,
            priority: AllocationPriority::Normal,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_memory_config() {
        let config = PinnedMemoryConfig::default();
        assert_eq!(config.max_pinned_memory, 512 * 1024 * 1024);
        assert!(config.enable_auto_cleanup);
        assert!(config.enable_transfer_tracking);
    }

    #[test]
    fn test_pinned_memory_pool() {
        let mut pool = PinnedMemoryPool::new(4096);
        assert_eq!(pool.size_class, 4096);
        assert!(pool.free_blocks.is_empty());
        assert!(pool.allocated_blocks.is_empty());
        assert!(pool.is_empty());
    }

    #[test]
    fn test_transfer_stats() {
        let mut stats = TransferStats::default();

        stats.update_stats(1024, Duration::from_millis(10), 1024000.0);

        assert_eq!(stats.total_transfers, 1);
        assert_eq!(stats.total_bytes, 1024);
        assert_eq!(stats.peak_bandwidth, 1024000.0);
    }

    #[test]
    fn test_usage_patterns() {
        assert_eq!(UsagePattern::HostToDevice, UsagePattern::HostToDevice);
        assert_ne!(UsagePattern::HostToDevice, UsagePattern::DeviceToHost);
    }

    #[test]
    fn test_allocation_priorities() {
        assert!(AllocationPriority::Critical > AllocationPriority::High);
        assert!(AllocationPriority::High > AllocationPriority::Normal);
        assert!(AllocationPriority::Normal > AllocationPriority::Low);
    }

    #[test]
    fn test_pinned_memory_request() {
        let request = PinnedMemoryRequest {
            size: 4096,
            enable_mapping: true,
            usage_pattern: UsagePattern::HostToDevice,
            priority: AllocationPriority::High,
            ..Default::default()
        };

        assert_eq!(request.size, 4096);
        assert!(request.enable_mapping);
        assert_eq!(request.usage_pattern, UsagePattern::HostToDevice);
        assert_eq!(request.priority, AllocationPriority::High);
    }
}
