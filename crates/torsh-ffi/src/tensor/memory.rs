use crate::error::FfiResult;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Memory pool for efficient tensor allocation
#[derive(Debug)]
pub struct MemoryPool {
    /// Available memory blocks organized by size
    free_blocks: Mutex<std::collections::HashMap<usize, VecDeque<Vec<f32>>>>,
    /// Statistics for monitoring (using atomics for lock-free counter updates)
    allocations: AtomicUsize,
    deallocations: AtomicUsize,
    pool_hits: AtomicUsize,
    pool_misses: AtomicUsize,
    max_pool_size: usize,
}

impl MemoryPool {
    /// Create a new memory pool with specified maximum size
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            free_blocks: Mutex::new(std::collections::HashMap::new()),
            allocations: AtomicUsize::new(0),
            deallocations: AtomicUsize::new(0),
            pool_hits: AtomicUsize::new(0),
            pool_misses: AtomicUsize::new(0),
            max_pool_size,
        }
    }

    /// Allocate a vector from the pool or create new
    pub fn allocate(&self, size: usize) -> FfiResult<Vec<f32>> {
        let mut free_blocks = self.free_blocks.lock();

        if let Some(blocks) = free_blocks.get_mut(&size) {
            if let Some(mut block) = blocks.pop_front() {
                // Reuse existing block
                block.clear();
                block.resize(size, 0.0);
                self.pool_hits.fetch_add(1, Ordering::Relaxed);
                self.allocations.fetch_add(1, Ordering::Relaxed);
                return Ok(block);
            }
        }

        // Create new block
        self.pool_misses.fetch_add(1, Ordering::Relaxed);
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(vec![0.0; size])
    }

    /// Return a vector to the pool for reuse
    pub fn deallocate(&self, mut data: Vec<f32>) -> FfiResult<()> {
        let size = data.capacity();

        // Only pool blocks that are reasonably sized and within our limits
        if size > 0 && size <= self.max_pool_size {
            let mut free_blocks = self.free_blocks.lock();

            let blocks = free_blocks.entry(size).or_insert_with(VecDeque::new);

            // Limit the number of blocks per size to prevent unbounded growth
            if blocks.len() < 10 {
                data.clear();
                blocks.push_back(data);
            }
        }

        self.deallocations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get memory pool statistics
    pub fn stats(&self) -> FfiResult<MemoryPoolStats> {
        Ok(MemoryPoolStats {
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            pool_hits: self.pool_hits.load(Ordering::Relaxed),
            pool_misses: self.pool_misses.load(Ordering::Relaxed),
            active_blocks: self.free_blocks.lock().values().map(|v| v.len()).sum(),
        })
    }

    /// Clear all pooled memory
    pub fn clear(&self) -> FfiResult<()> {
        self.free_blocks.lock().clear();
        Ok(())
    }
}

/// Statistics for memory pool usage
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub active_blocks: usize,
}

/// Global memory pool instance
pub static MEMORY_POOL: Lazy<MemoryPool> = Lazy::new(|| {
    MemoryPool::new(1024 * 1024) // 1MB max pool size
});
