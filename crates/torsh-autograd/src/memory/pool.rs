//! Memory pool management for efficient allocation and reuse
//!
//! This module provides memory pooling functionality to reduce allocation overhead
//! and improve memory efficiency in autograd operations. Memory pools maintain
//! collections of pre-allocated memory chunks that can be reused across operations.
//!
//! # Overview
//!
//! Memory pooling is a critical optimization technique that:
//!
//! - **Reduces Allocation Overhead**: Reuses existing memory instead of allocating new
//! - **Improves Cache Locality**: Keeps frequently used memory "warm" in cache
//! - **Reduces Fragmentation**: Manages memory in predictable chunk sizes
//! - **Provides Statistics**: Tracks pool usage for optimization insights
//!
//! # Pool Management Strategy
//!
//! The memory pool system uses size-based pooling where:
//! - Different pools are maintained for different tensor sizes
//! - Pool size grows dynamically based on usage patterns
//! - Unused memory is periodically released to avoid unbounded growth
//! - Cache hit rates are tracked to optimize pool sizing
//!
//! # Examples
//!
//! ```rust
//! use crate::memory::pool::MemoryPool;
//! use torsh_core::dtype::f32;
//!
//! let mut pool: MemoryPool<f32> = MemoryPool::new();
//!
//! // Allocate memory from pool
//! let memory = pool.allocate(1000)?; // 1000 f32 elements
//!
//! // Use memory for computations
//! // ... gradient computation ...
//!
//! // Return memory to pool for reuse
//! pool.deallocate(memory);
//!
//! // Check pool statistics
//! let stats = pool.get_stats();
//! println!("Cache hit rate: {:.2}%", stats.cache_hit_rate());
//! ```

use std::collections::HashMap;
use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};

/// Memory pool for efficient allocation and reuse
///
/// Manages a collection of memory chunks of the same element type, providing
/// efficient allocation and deallocation through reuse. The pool maintains
/// statistics about usage patterns to optimize performance.
///
/// # Type Parameter
///
/// * `T` - The element type stored in the pool (must implement `FloatElement`)
///
/// # Thread Safety
///
/// This pool is not thread-safe. For concurrent access, wrap in appropriate
/// synchronization primitives like `Arc<Mutex<MemoryPool<T>>>`.
///
/// # Memory Management
///
/// The pool automatically manages memory lifecycle:
/// - Allocations first try to reuse existing chunks
/// - New chunks are allocated only when pool is empty
/// - Deallocated chunks are returned to the pool for reuse
/// - Pool size is limited to prevent unbounded growth
#[derive(Debug)]
pub struct MemoryPool<T: FloatElement> {
    /// Available memory chunks organized by size
    available_chunks: HashMap<usize, Vec<Vec<T>>>,
    /// Allocated chunks tracking for statistics
    allocated_chunks: HashMap<*mut T, usize>,
    /// Pool statistics
    stats: MemoryPoolStats,
    /// Maximum number of chunks to keep per size
    max_chunks_per_size: usize,
    /// Maximum total memory to keep in pool (bytes)
    max_pool_memory: usize,
}

/// Memory pool statistics
///
/// Comprehensive statistics about memory pool usage, including allocation
/// patterns, cache efficiency, and memory consumption. These statistics
/// are essential for optimizing pool configuration and understanding
/// memory usage patterns.
///
/// # Key Metrics
///
/// - **Allocations/Deallocations**: Total lifecycle operations
/// - **Cache Hits/Misses**: Efficiency of memory reuse
/// - **Pool Size**: Current and peak memory consumption
/// - **Hit Rate**: Percentage of allocations served from pool
///
/// # Performance Analysis
///
/// Use these metrics to optimize pool configuration:
/// - Low hit rate → Increase pool size or adjust chunk sizes
/// - High peak memory → Reduce max pool size or implement more aggressive cleanup
/// - High miss rate → Pool warming or preallocation strategies
#[derive(Debug, Clone, Default)]
pub struct MemoryPoolStats {
    /// Total allocations requested
    pub total_allocations: usize,
    /// Total deallocations processed
    pub total_deallocations: usize,
    /// Cache hits (allocations served from pool)
    pub cache_hits: usize,
    /// Cache misses (allocations requiring new memory)
    pub cache_misses: usize,
    /// Current pool size in bytes
    pub current_pool_size: usize,
    /// Peak pool size in bytes
    pub peak_pool_size: usize,
    /// Number of different chunk sizes in pool
    pub unique_chunk_sizes: usize,
    /// Total number of chunks currently in pool
    pub total_chunks_in_pool: usize,
}

impl<T: FloatElement> MemoryPool<T> {
    /// Create a new memory pool with default configuration
    ///
    /// Initializes an empty memory pool with sensible defaults:
    /// - Maximum 100 chunks per size class
    /// - Maximum 1GB total pool memory
    /// - All statistics reset to zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// let pool: MemoryPool<f32> = MemoryPool::new();
    /// assert_eq!(pool.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            available_chunks: HashMap::new(),
            allocated_chunks: HashMap::new(),
            stats: MemoryPoolStats::default(),
            max_chunks_per_size: 100,
            max_pool_memory: 1024 * 1024 * 1024, // 1GB
        }
    }

    /// Create a memory pool with custom configuration
    ///
    /// Allows fine-tuning of pool behavior for specific use cases.
    ///
    /// # Arguments
    ///
    /// * `max_chunks_per_size` - Maximum chunks to keep for each size class
    /// * `max_pool_memory` - Maximum total memory to keep in pool (bytes)
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Create a smaller pool for memory-constrained environments
    /// let pool: MemoryPool<f32> = MemoryPool::with_config(50, 512 * 1024 * 1024);
    /// ```
    pub fn with_config(max_chunks_per_size: usize, max_pool_memory: usize) -> Self {
        Self {
            available_chunks: HashMap::new(),
            allocated_chunks: HashMap::new(),
            stats: MemoryPoolStats::default(),
            max_chunks_per_size,
            max_pool_memory,
        }
    }

    /// Allocate memory from the pool
    ///
    /// Attempts to serve the allocation from existing pool memory first,
    /// falling back to new allocation if no suitable chunk is available.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of elements to allocate
    ///
    /// # Returns
    ///
    /// Vector of `T` elements with exactly `size` capacity.
    ///
    /// # Performance
    ///
    /// - **Cache Hit**: O(1) - returns existing chunk immediately
    /// - **Cache Miss**: O(n) - allocates new memory where n = size
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut pool: MemoryPool<f32> = MemoryPool::new();
    /// let memory = pool.allocate(1000)?;
    /// assert_eq!(memory.len(), 1000);
    /// ```
    pub fn allocate(&mut self, size: usize) -> Result<Vec<T>> {
        self.stats.total_allocations += 1;

        // Try to reuse existing chunk
        if let Some(chunks) = self.available_chunks.get_mut(&size) {
            if let Some(chunk) = chunks.pop() {
                self.stats.cache_hits += 1;
                self.stats.current_pool_size -= size * std::mem::size_of::<T>();
                self.stats.total_chunks_in_pool -= 1;

                // Track allocation for statistics
                let ptr = chunk.as_ptr() as *mut T;
                self.allocated_chunks.insert(ptr, size);

                return Ok(chunk);
            }
        }

        // Allocate new chunk
        self.stats.cache_misses += 1;
        let mut chunk = Vec::with_capacity(size);
        chunk.resize(size, <T as torsh_core::dtype::TensorElement>::zero()); // Allocate with correct length, not just capacity

        // Track allocation
        let ptr = chunk.as_ptr() as *mut T;
        self.allocated_chunks.insert(ptr, size);

        Ok(chunk)
    }

    /// Deallocate memory back to the pool
    ///
    /// Returns memory to the pool for reuse, subject to pool size limits.
    /// If the pool is at capacity, the memory is dropped instead of stored.
    ///
    /// # Arguments
    ///
    /// * `memory` - Vector to return to the pool
    ///
    /// # Pool Management
    ///
    /// The pool automatically manages its size:
    /// - Chunks are stored if pool has capacity
    /// - Oldest chunks are evicted when pool is full
    /// - Pool size is tracked to enforce memory limits
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut pool: MemoryPool<f32> = MemoryPool::new();
    /// let memory = pool.allocate(1000)?;
    /// // ... use memory ...
    /// pool.deallocate(memory); // Returns to pool for reuse
    /// ```
    pub fn deallocate(&mut self, mut memory: Vec<T>) {
        self.stats.total_deallocations += 1;

        let size = memory.capacity();
        let ptr = memory.as_ptr() as *mut T;

        // Remove from tracking
        self.allocated_chunks.remove(&ptr);

        // Check if we should keep this chunk in the pool
        if self.should_keep_chunk(size) {
            // Clear the vector but keep capacity
            memory.clear();

            // Add to available chunks
            let chunks = self.available_chunks.entry(size).or_insert_with(Vec::new);

            // Enforce per-size limit
            if chunks.len() < self.max_chunks_per_size {
                chunks.push(memory);
                self.stats.current_pool_size += size * std::mem::size_of::<T>();
                self.stats.total_chunks_in_pool += 1;

                // Update peak pool size
                if self.stats.current_pool_size > self.stats.peak_pool_size {
                    self.stats.peak_pool_size = self.stats.current_pool_size;
                }
            }
        }

        // Update unique chunk sizes count
        self.stats.unique_chunk_sizes = self.available_chunks.len();
    }

    /// Check if a chunk should be kept in the pool
    ///
    /// Determines whether a deallocated chunk should be stored in the pool
    /// based on current pool size and memory limits.
    fn should_keep_chunk(&self, size: usize) -> bool {
        let chunk_memory = size * std::mem::size_of::<T>();
        self.stats.current_pool_size + chunk_memory <= self.max_pool_memory
    }

    /// Get current pool statistics
    ///
    /// Returns a copy of current pool statistics for analysis and monitoring.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let stats = pool.get_stats();
    /// println!("Cache hit rate: {:.1}%", stats.cache_hit_rate());
    /// println!("Pool memory usage: {} bytes", stats.current_pool_size);
    /// ```
    pub fn get_stats(&self) -> MemoryPoolStats {
        self.stats.clone()
    }

    /// Get number of chunks currently in the pool
    ///
    /// Returns the total number of memory chunks available for reuse.
    pub fn len(&self) -> usize {
        self.stats.total_chunks_in_pool
    }

    /// Check if the pool is empty
    ///
    /// Returns true if no chunks are available for reuse.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all memory from the pool
    ///
    /// Removes all cached chunks from the pool, freeing memory.
    /// Statistics are preserved but current pool size is reset to zero.
    ///
    /// # Use Cases
    ///
    /// - Memory pressure response
    /// - End of training epoch cleanup
    /// - Manual memory management
    ///
    /// # Examples
    ///
    /// ```rust
    /// pool.clear();
    /// assert_eq!(pool.len(), 0);
    /// assert_eq!(pool.get_stats().current_pool_size, 0);
    /// ```
    pub fn clear(&mut self) {
        self.available_chunks.clear();
        self.stats.current_pool_size = 0;
        self.stats.total_chunks_in_pool = 0;
        self.stats.unique_chunk_sizes = 0;
    }

    /// Trim pool to reduce memory usage
    ///
    /// Removes least recently used chunks to reduce pool memory usage.
    /// Useful for responding to memory pressure without completely clearing the pool.
    ///
    /// # Arguments
    ///
    /// * `target_size` - Target pool size in bytes
    ///
    /// # Strategy
    ///
    /// - Removes entire size classes starting with largest chunks
    /// - Preserves smaller, more frequently used chunks
    /// - Updates statistics to reflect new pool state
    pub fn trim_to_size(&mut self, target_size: usize) {
        if self.stats.current_pool_size <= target_size {
            return;
        }

        // Collect and sort sizes by chunk size (largest first)
        let mut sizes: Vec<usize> = self.available_chunks.keys().cloned().collect();
        sizes.sort_by(|a, b| b.cmp(a));

        for size in sizes {
            if self.stats.current_pool_size <= target_size {
                break;
            }

            if let Some(chunks) = self.available_chunks.get_mut(&size) {
                let chunk_memory = size * std::mem::size_of::<T>();
                let chunks_to_remove = std::cmp::min(
                    chunks.len(),
                    (self.stats.current_pool_size - target_size) / chunk_memory + 1,
                );

                for _ in 0..chunks_to_remove {
                    if chunks.pop().is_some() {
                        self.stats.current_pool_size -= chunk_memory;
                        self.stats.total_chunks_in_pool -= 1;
                    }
                }

                if chunks.is_empty() {
                    self.available_chunks.remove(&size);
                }
            }
        }

        self.stats.unique_chunk_sizes = self.available_chunks.len();
    }
}

impl Default for MemoryPool<f32> {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPoolStats {
    /// Calculate cache hit rate as percentage
    ///
    /// Returns the percentage of allocations that were served from the pool
    /// rather than requiring new memory allocation.
    ///
    /// # Returns
    ///
    /// Hit rate as percentage (0.0 - 100.0), or 0.0 if no allocations.
    pub fn cache_hit_rate(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / self.total_allocations as f64) * 100.0
        }
    }

    /// Calculate memory efficiency ratio
    ///
    /// Returns the ratio of current pool size to peak pool size,
    /// indicating how efficiently pool memory is being used.
    ///
    /// # Returns
    ///
    /// Efficiency ratio (0.0 - 1.0), or 1.0 if no peak recorded.
    pub fn memory_efficiency(&self) -> f64 {
        if self.peak_pool_size == 0 {
            1.0
        } else {
            self.current_pool_size as f64 / self.peak_pool_size as f64
        }
    }

    /// Get average chunks per size class
    ///
    /// Returns the average number of chunks stored per unique size class.
    pub fn average_chunks_per_size(&self) -> f64 {
        if self.unique_chunk_sizes == 0 {
            0.0
        } else {
            self.total_chunks_in_pool as f64 / self.unique_chunk_sizes as f64
        }
    }

    /// Check if pool performance is healthy
    ///
    /// Returns true if pool statistics indicate good performance:
    /// - Cache hit rate > 70%
    /// - Memory efficiency > 50%
    /// - Reasonable number of size classes
    pub fn is_healthy(&self) -> bool {
        self.cache_hit_rate() > 70.0
            && self.memory_efficiency() > 0.5
            && self.unique_chunk_sizes < 100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool: MemoryPool<f32> = MemoryPool::new();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());

        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.current_pool_size, 0);
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let mut pool: MemoryPool<f32> = MemoryPool::new();

        // First allocation should be a cache miss
        let memory1 = pool.allocate(100).unwrap();
        assert_eq!(memory1.capacity(), 100);

        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 0);

        // Deallocate back to pool
        pool.deallocate(memory1);
        assert_eq!(pool.len(), 1);

        // Second allocation of same size should be a cache hit
        let memory2 = pool.allocate(100).unwrap();
        assert_eq!(memory2.capacity(), 100);

        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
    }

    #[test]
    fn test_different_sizes() {
        let mut pool: MemoryPool<f32> = MemoryPool::new();

        let mem1 = pool.allocate(100).unwrap();
        let mem2 = pool.allocate(200).unwrap();

        pool.deallocate(mem1);
        pool.deallocate(mem2);

        assert_eq!(pool.len(), 2);

        let stats = pool.get_stats();
        assert_eq!(stats.unique_chunk_sizes, 2);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut pool: MemoryPool<f32> = MemoryPool::new();

        // Allocate and deallocate to populate pool
        let mem = pool.allocate(100).unwrap();
        pool.deallocate(mem);

        // Multiple allocations of same size
        for _ in 0..5 {
            let mem = pool.allocate(100).unwrap();
            pool.deallocate(mem);
        }

        let stats = pool.get_stats();
        assert!(stats.cache_hit_rate() > 80.0); // Should have high hit rate
    }

    #[test]
    fn test_pool_limits() {
        let mut pool: MemoryPool<f32> = MemoryPool::with_config(2, 1000);

        // Fill pool to capacity
        for _ in 0..3 {
            let mem = pool.allocate(10).unwrap();
            pool.deallocate(mem);
        }

        // Pool should be limited to max_chunks_per_size
        assert!(pool.len() <= 2);
    }

    #[test]
    fn test_clear_pool() {
        let mut pool: MemoryPool<f32> = MemoryPool::new();

        let mem = pool.allocate(100).unwrap();
        pool.deallocate(mem);
        assert_eq!(pool.len(), 1);

        pool.clear();
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.get_stats().current_pool_size, 0);
    }

    #[test]
    fn test_trim_pool() {
        let mut pool: MemoryPool<f32> = MemoryPool::new();

        // Add chunks of different sizes
        for size in [100, 200, 400] {
            let mem = pool.allocate(size).unwrap();
            pool.deallocate(mem);
        }

        let initial_size = pool.get_stats().current_pool_size;
        assert!(initial_size > 0);

        // Trim to smaller size
        pool.trim_to_size(initial_size / 2);

        let final_size = pool.get_stats().current_pool_size;
        assert!(final_size <= initial_size / 2);
    }

    #[test]
    fn test_stats_calculations() {
        let mut stats = MemoryPoolStats::default();
        stats.total_allocations = 100;
        stats.cache_hits = 80;
        stats.current_pool_size = 600; // > 0.5 efficiency for health check
        stats.peak_pool_size = 1000;
        stats.unique_chunk_sizes = 5;
        stats.total_chunks_in_pool = 20;

        assert_eq!(stats.cache_hit_rate(), 80.0);
        assert_eq!(stats.memory_efficiency(), 0.6);
        assert_eq!(stats.average_chunks_per_size(), 4.0);
        assert!(stats.is_healthy());
    }

    #[test]
    fn test_custom_config() {
        let pool: MemoryPool<f32> = MemoryPool::with_config(50, 512 * 1024);
        assert_eq!(pool.max_chunks_per_size, 50);
        assert_eq!(pool.max_pool_memory, 512 * 1024);
    }
}
