//! Advanced memory management for sparse operations
//!
//! This module provides efficient memory allocation, caching, and garbage collection
//! strategies specifically optimized for sparse matrix operations.

use crate::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use torsh_core::{DType, Result as TorshResult, Shape};

/// Global memory pool for sparse matrix operations
static MEMORY_POOL: std::sync::OnceLock<Arc<RwLock<SparseMemoryPool>>> = std::sync::OnceLock::new();

/// Get or initialize the global memory pool
fn get_memory_pool() -> &'static Arc<RwLock<SparseMemoryPool>> {
    MEMORY_POOL.get_or_init(|| Arc::new(RwLock::new(SparseMemoryPool::new())))
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Maximum total memory to use (in bytes)
    pub max_total_memory: usize,
    /// Maximum memory per allocation (in bytes)
    pub max_allocation_size: usize,
    /// Number of size buckets for memory allocation
    pub num_size_buckets: usize,
    /// Time after which unused memory is released
    pub memory_timeout: Duration,
    /// Enable memory usage tracking
    pub enable_tracking: bool,
    /// Garbage collection interval
    pub gc_interval: Duration,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_total_memory: 1024 * 1024 * 1024,   // 1GB
            max_allocation_size: 128 * 1024 * 1024, // 128MB
            num_size_buckets: 16,
            memory_timeout: Duration::from_secs(300), // 5 minutes
            enable_tracking: true,
            gc_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

/// Memory allocation bucket for specific size ranges
#[derive(Debug)]
struct MemoryBucket {
    /// Size range this bucket handles
    size_range: (usize, usize),
    /// Available memory blocks
    available_blocks: VecDeque<MemoryBlock>,
    /// Total allocated memory in this bucket
    total_allocated: usize,
    /// Number of active allocations
    active_allocations: usize,
}

/// Memory block with metadata
#[derive(Debug)]
struct MemoryBlock {
    /// Raw memory allocation
    memory: Vec<u8>,
    /// Allocation timestamp
    allocated_at: Instant,
    /// Last accessed timestamp
    last_accessed: Instant,
    /// Reference count
    ref_count: usize,
    /// Block ID for tracking
    #[allow(dead_code)]
    id: u64,
}

/// Sparse memory pool for efficient memory management
#[derive(Debug)]
pub struct SparseMemoryPool {
    /// Configuration
    config: MemoryPoolConfig,
    /// Memory buckets organized by size
    buckets: Vec<MemoryBucket>,
    /// Total memory currently allocated
    total_allocated: usize,
    /// Memory usage statistics
    stats: MemoryStatistics,
    /// Last garbage collection time
    last_gc: Instant,
    /// Unique ID counter for memory blocks
    next_block_id: u64,
    /// Cache for frequently used allocations
    allocation_cache: HashMap<String, Vec<u64>>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Total bytes allocated since start
    pub total_allocated_bytes: usize,
    /// Total bytes deallocated since start
    pub total_deallocated_bytes: usize,
    /// Current active allocations
    pub active_allocations: usize,
    /// Current memory usage
    pub current_memory_usage: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Number of allocation requests
    pub allocation_requests: usize,
    /// Number of successful reuses from pool
    pub pool_reuses: usize,
    /// Number of garbage collections performed
    pub garbage_collections: usize,
    /// Average allocation size
    pub average_allocation_size: f64,
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self {
            total_allocated_bytes: 0,
            total_deallocated_bytes: 0,
            active_allocations: 0,
            current_memory_usage: 0,
            peak_memory_usage: 0,
            allocation_requests: 0,
            pool_reuses: 0,
            garbage_collections: 0,
            average_allocation_size: 0.0,
        }
    }
}

impl MemoryBucket {
    fn new(size_range: (usize, usize)) -> Self {
        Self {
            size_range,
            available_blocks: VecDeque::new(),
            total_allocated: 0,
            active_allocations: 0,
        }
    }

    fn can_handle(&self, size: usize) -> bool {
        size >= self.size_range.0 && size <= self.size_range.1
    }

    fn allocate(&mut self, size: usize, block_id: u64) -> Option<MemoryBlock> {
        if !self.can_handle(size) {
            return None;
        }

        // Try to reuse existing block
        if let Some(mut block) = self.available_blocks.pop_front() {
            block.last_accessed = Instant::now();
            block.ref_count = 1;
            return Some(block);
        }

        // Allocate new block
        let actual_size = self.size_range.1; // Allocate at bucket maximum for reuse
        let memory = vec![0u8; actual_size];
        let now = Instant::now();

        self.total_allocated += actual_size;
        self.active_allocations += 1;

        Some(MemoryBlock {
            memory,
            allocated_at: now,
            last_accessed: now,
            ref_count: 1,
            id: block_id,
        })
    }

    fn deallocate(&mut self, block: MemoryBlock) {
        self.active_allocations = self.active_allocations.saturating_sub(1);
        self.available_blocks.push_back(block);
    }

    fn cleanup_expired(&mut self, timeout: Duration) -> usize {
        let now = Instant::now();
        let initial_count = self.available_blocks.len();

        self.available_blocks.retain(|block| {
            let should_keep = now.duration_since(block.last_accessed) < timeout;
            if !should_keep {
                self.total_allocated = self.total_allocated.saturating_sub(block.memory.len());
            }
            should_keep
        });

        initial_count - self.available_blocks.len()
    }
}

impl Default for SparseMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseMemoryPool {
    /// Create a new memory pool with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryPoolConfig::default())
    }

    /// Create a new memory pool with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        let buckets = Self::create_buckets(&config);

        Self {
            config,
            buckets,
            total_allocated: 0,
            stats: MemoryStatistics::default(),
            last_gc: Instant::now(),
            next_block_id: 1,
            allocation_cache: HashMap::new(),
        }
    }

    /// Create memory buckets based on configuration
    fn create_buckets(config: &MemoryPoolConfig) -> Vec<MemoryBucket> {
        let mut buckets = Vec::new();
        let max_size = config.max_allocation_size;
        let num_buckets = config.num_size_buckets;

        // Create exponentially sized buckets
        let mut size = 1024; // Start at 1KB
        for _ in 0..num_buckets {
            let next_size = (size * 2).min(max_size);
            buckets.push(MemoryBucket::new((size, next_size)));
            size = next_size;
            if size >= max_size {
                break;
            }
        }

        buckets
    }

    /// Allocate memory for sparse matrix data
    pub fn allocate(
        &mut self,
        size: usize,
        allocation_type: &str,
    ) -> TorshResult<SparseMemoryHandle> {
        if size > self.config.max_allocation_size {
            return Err(torsh_core::TorshError::Other(format!(
                "Allocation size {} exceeds maximum {}",
                size, self.config.max_allocation_size
            )));
        }

        if self.total_allocated + size > self.config.max_total_memory {
            // Try garbage collection first
            self.garbage_collect();

            if self.total_allocated + size > self.config.max_total_memory {
                return Err(torsh_core::TorshError::Other(
                    "Memory pool capacity exceeded".to_string(),
                ));
            }
        }

        // Find appropriate bucket
        let bucket_idx = self.find_bucket_for_size(size);
        let block_id = self.next_block_id;
        self.next_block_id += 1;

        let block = if let Some(idx) = bucket_idx {
            self.buckets[idx].allocate(size, block_id)
        } else {
            // Direct allocation for sizes that don't fit buckets
            let memory = vec![0u8; size];
            let now = Instant::now();
            Some(MemoryBlock {
                memory,
                allocated_at: now,
                last_accessed: now,
                ref_count: 1,
                id: block_id,
            })
        };

        if let Some(block) = block {
            self.total_allocated += block.memory.len();
            self.stats.total_allocated_bytes += block.memory.len();
            self.stats.allocation_requests += 1;
            self.stats.active_allocations += 1;
            self.stats.current_memory_usage = self.total_allocated;

            if self.total_allocated > self.stats.peak_memory_usage {
                self.stats.peak_memory_usage = self.total_allocated;
            }

            self.stats.average_allocation_size =
                self.stats.total_allocated_bytes as f64 / self.stats.allocation_requests as f64;

            // Cache allocation type for optimization
            self.allocation_cache
                .entry(allocation_type.to_string())
                .or_default()
                .push(block_id);

            Ok(SparseMemoryHandle::new(block, bucket_idx))
        } else {
            Err(torsh_core::TorshError::Other(
                "Failed to allocate memory".to_string(),
            ))
        }
    }

    /// Find the appropriate bucket for a given size
    fn find_bucket_for_size(&self, size: usize) -> Option<usize> {
        self.buckets
            .iter()
            .position(|bucket| bucket.can_handle(size))
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&mut self, handle: SparseMemoryHandle) {
        let (block, bucket_idx) = handle.into_parts();

        self.total_allocated = self.total_allocated.saturating_sub(block.memory.len());
        self.stats.total_deallocated_bytes += block.memory.len();
        self.stats.active_allocations = self.stats.active_allocations.saturating_sub(1);
        self.stats.current_memory_usage = self.total_allocated;

        if let Some(idx) = bucket_idx {
            if idx < self.buckets.len() {
                self.buckets[idx].deallocate(block);
                self.stats.pool_reuses += 1;
            }
        }
        // For direct allocations (no bucket), memory is automatically freed when block is dropped
    }

    /// Perform garbage collection
    pub fn garbage_collect(&mut self) {
        let now = Instant::now();

        if now.duration_since(self.last_gc) < self.config.gc_interval {
            return; // Too soon for GC
        }

        let mut _total_freed = 0;
        for bucket in &mut self.buckets {
            _total_freed += bucket.cleanup_expired(self.config.memory_timeout);
        }

        // Clear old allocation cache entries
        self.allocation_cache.retain(|_, ids| {
            ids.retain(|_| true); // Keep all for now, could implement smarter logic
            !ids.is_empty()
        });

        self.stats.garbage_collections += 1;
        self.last_gc = now;
    }

    /// Force garbage collection (ignore timing restrictions)
    pub fn force_garbage_collect(&mut self) {
        self.last_gc = Instant::now() - self.config.gc_interval;
        self.garbage_collect();
    }

    /// Get current memory statistics
    pub fn statistics(&self) -> MemoryStatistics {
        self.stats.clone()
    }

    /// Get memory usage by allocation type
    pub fn usage_by_type(&self) -> HashMap<String, usize> {
        // This would need more sophisticated tracking to implement properly
        // For now, return a placeholder
        let mut usage = HashMap::new();
        usage.insert("sparse_matrices".to_string(), self.total_allocated);
        usage
    }

    /// Check if memory pool is healthy (not near capacity)
    pub fn is_healthy(&self) -> bool {
        let usage_ratio = self.total_allocated as f64 / self.config.max_total_memory as f64;
        usage_ratio < 0.8 // Consider healthy if under 80% capacity
    }

    /// Get memory efficiency score (0.0 to 1.0)
    pub fn efficiency_score(&self) -> f64 {
        if self.stats.allocation_requests == 0 {
            return 1.0;
        }

        let reuse_ratio = self.stats.pool_reuses as f64 / self.stats.allocation_requests as f64;
        let fragmentation_ratio =
            1.0 - (self.stats.current_memory_usage as f64 / self.stats.peak_memory_usage as f64);

        (reuse_ratio + (1.0 - fragmentation_ratio)) / 2.0
    }
}

/// Handle for managed memory allocation
pub struct SparseMemoryHandle {
    block: Option<MemoryBlock>,
    bucket_idx: Option<usize>,
}

impl SparseMemoryHandle {
    fn new(block: MemoryBlock, bucket_idx: Option<usize>) -> Self {
        Self {
            block: Some(block),
            bucket_idx,
        }
    }

    /// Get mutable access to the memory
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if let Some(ref mut block) = self.block {
            block.last_accessed = Instant::now();
            &mut block.memory
        } else {
            &mut []
        }
    }

    /// Get immutable access to the memory
    pub fn as_slice(&self) -> &[u8] {
        if let Some(ref block) = self.block {
            &block.memory
        } else {
            &[]
        }
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.block.as_ref().map_or(0, |b| b.memory.len())
    }

    /// Get allocation age
    pub fn age(&self) -> Duration {
        self.block.as_ref().map_or(Duration::ZERO, |b| {
            Instant::now().duration_since(b.allocated_at)
        })
    }

    /// Check if the handle is valid
    pub fn is_valid(&self) -> bool {
        self.block.is_some()
    }

    /// Extract the internal parts (consumes the handle)
    fn into_parts(mut self) -> (MemoryBlock, Option<usize>) {
        let block = self.block.take().expect("Handle should have a block");
        (block, self.bucket_idx)
    }
}

impl Drop for SparseMemoryHandle {
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            if let Ok(mut pool) = get_memory_pool().write() {
                pool.deallocate(SparseMemoryHandle {
                    block: Some(block),
                    bucket_idx: self.bucket_idx,
                });
            }
        }
    }
}

/// Global memory management functions
pub struct SparseMemoryManager;

impl SparseMemoryManager {
    /// Allocate memory using the global pool
    pub fn allocate(size: usize, allocation_type: &str) -> TorshResult<SparseMemoryHandle> {
        get_memory_pool()
            .write()
            .unwrap()
            .allocate(size, allocation_type)
    }

    /// Get global memory statistics
    pub fn global_statistics() -> MemoryStatistics {
        get_memory_pool().read().unwrap().statistics()
    }

    /// Force global garbage collection
    pub fn force_garbage_collect() {
        get_memory_pool().write().unwrap().force_garbage_collect();
    }

    /// Check if global memory pool is healthy
    pub fn is_healthy() -> bool {
        get_memory_pool().read().unwrap().is_healthy()
    }

    /// Get global memory efficiency score
    pub fn efficiency_score() -> f64 {
        get_memory_pool().read().unwrap().efficiency_score()
    }

    /// Configure the global memory pool
    pub fn configure(config: MemoryPoolConfig) {
        let mut pool = get_memory_pool().write().unwrap();
        *pool = SparseMemoryPool::with_config(config);
    }

    /// Generate memory usage report
    pub fn generate_report() -> MemoryReport {
        let pool = get_memory_pool().read().unwrap();
        let stats = pool.statistics();
        let usage_by_type = pool.usage_by_type();
        let is_healthy = pool.is_healthy();
        let efficiency = pool.efficiency_score();

        MemoryReport {
            statistics: stats,
            usage_by_type,
            is_healthy,
            efficiency_score: efficiency,
            recommendations: Self::generate_recommendations(&pool),
        }
    }

    /// Generate memory optimization recommendations
    fn generate_recommendations(pool: &SparseMemoryPool) -> Vec<String> {
        let mut recommendations = Vec::new();
        let stats = &pool.stats;

        // Check memory usage
        let usage_ratio = pool.total_allocated as f64 / pool.config.max_total_memory as f64;
        if usage_ratio > 0.9 {
            recommendations.push("Memory usage is very high. Consider increasing pool size or optimizing allocations.".to_string());
        } else if usage_ratio > 0.8 {
            recommendations.push(
                "Memory usage is high. Monitor closely and consider optimization.".to_string(),
            );
        }

        // Check reuse efficiency
        let reuse_ratio = if stats.allocation_requests > 0 {
            stats.pool_reuses as f64 / stats.allocation_requests as f64
        } else {
            0.0
        };

        if reuse_ratio < 0.3 {
            recommendations.push("Low memory reuse detected. Consider adjusting bucket sizes or allocation patterns.".to_string());
        }

        // Check garbage collection frequency
        if stats.garbage_collections == 0 && stats.allocation_requests > 100 {
            recommendations.push(
                "No garbage collections performed. Consider enabling automatic GC.".to_string(),
            );
        }

        // Check fragmentation
        let fragmentation = if stats.peak_memory_usage > 0 {
            1.0 - (stats.current_memory_usage as f64 / stats.peak_memory_usage as f64)
        } else {
            0.0
        };

        if fragmentation > 0.5 {
            recommendations.push(
                "High memory fragmentation detected. Consider more frequent garbage collection."
                    .to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Memory management appears optimal.".to_string());
        }

        recommendations
    }
}

/// Comprehensive memory usage report
#[derive(Debug, Clone)]
pub struct MemoryReport {
    pub statistics: MemoryStatistics,
    pub usage_by_type: HashMap<String, usize>,
    pub is_healthy: bool,
    pub efficiency_score: f64,
    pub recommendations: Vec<String>,
}

/// Memory-aware sparse matrix builder
pub struct MemoryAwareSparseBuilder {
    format: SparseFormat,
    estimated_nnz: usize,
    memory_handles: Vec<SparseMemoryHandle>,
    optimization_hints: Vec<String>,
}

impl MemoryAwareSparseBuilder {
    /// Create a new memory-aware builder
    pub fn new(format: SparseFormat, estimated_nnz: usize) -> Self {
        Self {
            format,
            estimated_nnz,
            memory_handles: Vec::new(),
            optimization_hints: Vec::new(),
        }
    }

    /// Pre-allocate memory for efficient building
    pub fn pre_allocate(&mut self) -> TorshResult<()> {
        let memory_needed = self.estimate_memory_requirements();

        // Allocate in chunks for better memory management
        let chunk_size = 1024 * 1024; // 1MB chunks
        let num_chunks = memory_needed.div_ceil(chunk_size);

        for i in 0..num_chunks {
            let size = if i == num_chunks - 1 {
                memory_needed - i * chunk_size
            } else {
                chunk_size
            };

            let handle = SparseMemoryManager::allocate(size, "sparse_builder")?;
            self.memory_handles.push(handle);
        }

        Ok(())
    }

    /// Estimate memory requirements for the sparse matrix
    fn estimate_memory_requirements(&self) -> usize {
        match self.format {
            SparseFormat::Coo => {
                self.estimated_nnz * (2 * std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
            }
            SparseFormat::Csr | SparseFormat::Csc => {
                self.estimated_nnz * (std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
                    + 1000 * std::mem::size_of::<usize>() // Rough estimate for row/col pointers
            }
            _ => self.estimated_nnz * 3 * std::mem::size_of::<f32>(), // Conservative estimate
        }
    }

    /// Build the sparse matrix using pre-allocated memory
    pub fn build(
        self,
        data: &[(usize, usize, f32)],
        shape: Shape,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        // This would use the pre-allocated memory to build the matrix efficiently
        // For now, fall back to standard creation
        match self.format {
            SparseFormat::Coo => {
                let mut coo = CooTensor::empty(shape, DType::F32)?;
                for &(row, col, val) in data {
                    coo.insert(row, col, val)?;
                }
                Ok(Box::new(coo))
            }
            SparseFormat::Csr => {
                // Convert to CSR format using pre-allocated memory
                let coo = {
                    let mut coo = CooTensor::empty(shape, DType::F32)?;
                    for &(row, col, val) in data {
                        coo.insert(row, col, val)?;
                    }
                    coo
                };
                Ok(Box::new(coo.to_csr()?))
            }
            _ => {
                // Default to COO for other formats
                let mut coo = CooTensor::empty(shape, DType::F32)?;
                for &(row, col, val) in data {
                    coo.insert(row, col, val)?;
                }
                Ok(Box::new(coo))
            }
        }
    }

    /// Get optimization hints generated during building
    pub fn optimization_hints(&self) -> &[String] {
        &self.optimization_hints
    }
}

/// Convenience function to create memory-efficient sparse matrices
pub fn create_sparse_with_memory_management(
    data: &[(usize, usize, f32)],
    shape: Shape,
    format: SparseFormat,
) -> TorshResult<Box<dyn SparseTensor>> {
    let mut builder = MemoryAwareSparseBuilder::new(format, data.len());
    builder.pre_allocate()?;
    builder.build(data, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool = SparseMemoryPool::new();
        assert!(pool.is_healthy());
        assert_eq!(pool.total_allocated, 0);
    }

    #[test]
    fn test_memory_allocation() {
        let mut pool = SparseMemoryPool::new();
        let handle = pool.allocate(1024, "test").unwrap();

        assert!(handle.is_valid());
        assert!(handle.size() >= 1024); // Pool allocates bucket-sized blocks for efficiency
        assert!(pool.total_allocated > 0);
    }

    #[test]
    fn test_memory_statistics() {
        let mut pool = SparseMemoryPool::new();
        let _handle1 = pool.allocate(1024, "test").unwrap();
        let _handle2 = pool.allocate(2048, "test").unwrap();

        let stats = pool.statistics();
        assert_eq!(stats.allocation_requests, 2);
        assert_eq!(stats.active_allocations, 2);
        assert!(stats.current_memory_usage > 0);
    }

    #[test]
    fn test_garbage_collection() {
        let mut pool = SparseMemoryPool::with_config(MemoryPoolConfig {
            memory_timeout: Duration::from_millis(1),
            ..Default::default()
        });

        {
            let _handle = pool.allocate(1024, "test").unwrap();
        } // Handle dropped here

        std::thread::sleep(Duration::from_millis(10));
        pool.force_garbage_collect();

        let stats = pool.statistics();
        assert!(stats.garbage_collections > 0);
    }

    #[test]
    fn test_global_memory_manager() {
        let handle = SparseMemoryManager::allocate(1024, "test").unwrap();
        assert!(handle.is_valid());

        let stats = SparseMemoryManager::global_statistics();
        assert!(stats.allocation_requests > 0);

        assert!(SparseMemoryManager::is_healthy());
    }

    #[test]
    fn test_memory_report() {
        let _handle = SparseMemoryManager::allocate(1024, "test").unwrap();
        let report = SparseMemoryManager::generate_report();

        assert!(report.statistics.allocation_requests > 0);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_memory_aware_builder() {
        let mut builder = MemoryAwareSparseBuilder::new(SparseFormat::Coo, 10);
        builder.pre_allocate().unwrap();

        let data = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let shape = Shape::new(vec![3, 3]);

        let sparse = builder.build(&data, shape).unwrap();
        assert_eq!(sparse.nnz(), 3);
    }

    #[test]
    fn test_memory_handle_operations() {
        let mut handle = SparseMemoryManager::allocate(1024, "test").unwrap();

        // Test mutable access
        {
            let slice = handle.as_mut_slice();
            slice[0] = 42;
            slice[1] = 24;
        }

        // Test immutable access
        let slice = handle.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[1], 24);

        assert!(handle.age() >= Duration::ZERO);
        assert!(handle.size() >= 1024); // Pool allocates bucket-sized blocks
    }
}
