use crate::error::{FfiError, FfiResult};
use once_cell::sync::Lazy;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Memory pool for efficient tensor allocation
#[derive(Debug)]
pub struct MemoryPool {
    /// Available memory blocks organized by size
    free_blocks: Mutex<std::collections::HashMap<usize, VecDeque<Vec<f32>>>>,
    /// Statistics for monitoring
    allocations: Mutex<usize>,
    deallocations: Mutex<usize>,
    pool_hits: Mutex<usize>,
    pool_misses: Mutex<usize>,
    max_pool_size: usize,
}

impl MemoryPool {
    /// Create a new memory pool with specified maximum size
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            free_blocks: Mutex::new(std::collections::HashMap::new()),
            allocations: Mutex::new(0),
            deallocations: Mutex::new(0),
            pool_hits: Mutex::new(0),
            pool_misses: Mutex::new(0),
            max_pool_size,
        }
    }

    /// Allocate a vector from the pool or create new
    pub fn allocate(&self, size: usize) -> FfiResult<Vec<f32>> {
        let mut free_blocks = self.free_blocks.lock().map_err(|_| FfiError::MemoryPool {
            message: "Failed to acquire pool lock".to_string(),
        })?;

        if let Some(blocks) = free_blocks.get_mut(&size) {
            if let Some(mut block) = blocks.pop_front() {
                // Reuse existing block
                block.clear();
                block.resize(size, 0.0);
                *self.pool_hits.lock().unwrap() += 1;
                *self.allocations.lock().unwrap() += 1;
                return Ok(block);
            }
        }

        // Create new block
        *self.pool_misses.lock().unwrap() += 1;
        *self.allocations.lock().unwrap() += 1;
        Ok(vec![0.0; size])
    }

    /// Return a vector to the pool for reuse
    pub fn deallocate(&self, mut data: Vec<f32>) -> FfiResult<()> {
        let size = data.capacity();

        // Only pool blocks that are reasonably sized and within our limits
        if size > 0 && size <= self.max_pool_size {
            let mut free_blocks = self.free_blocks.lock().map_err(|_| FfiError::MemoryPool {
                message: "Failed to acquire pool lock".to_string(),
            })?;

            let blocks = free_blocks.entry(size).or_insert_with(VecDeque::new);

            // Limit the number of blocks per size to prevent unbounded growth
            if blocks.len() < 10 {
                data.clear();
                blocks.push_back(data);
            }
        }

        *self.deallocations.lock().unwrap() += 1;
        Ok(())
    }

    /// Get memory pool statistics
    pub fn stats(&self) -> FfiResult<MemoryPoolStats> {
        Ok(MemoryPoolStats {
            allocations: *self.allocations.lock().unwrap(),
            deallocations: *self.deallocations.lock().unwrap(),
            pool_hits: *self.pool_hits.lock().unwrap(),
            pool_misses: *self.pool_misses.lock().unwrap(),
            active_blocks: self
                .free_blocks
                .lock()
                .unwrap()
                .values()
                .map(|v| v.len())
                .sum(),
        })
    }

    /// Clear all pooled memory
    pub fn clear(&self) -> FfiResult<()> {
        let mut free_blocks = self.free_blocks.lock().map_err(|_| FfiError::MemoryPool {
            message: "Failed to acquire pool lock".to_string(),
        })?;
        free_blocks.clear();
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
