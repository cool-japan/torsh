//! Memory pooling system for efficient small tensor allocations
//!
//! This module provides memory pooling to reduce allocation overhead and fragmentation
//! for small tensor operations. It includes thread-local pools for common data types.

use crate::dtype::TensorElement;
use std::collections::HashMap;

/// Memory pool for small tensor allocations to reduce fragmentation
///
/// This pool caches allocations of specific sizes to avoid frequent malloc/free calls
/// for small tensors. It's particularly useful for temporary tensors created during
/// computation.
#[derive(Debug)]
pub struct MemoryPool<T: TensorElement> {
    pools: HashMap<usize, Vec<Vec<T>>>,
    max_pool_size: usize,
    allocation_threshold: usize,
}

impl<T: TensorElement> Default for MemoryPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: TensorElement> MemoryPool<T> {
    /// Create a new memory pool
    pub fn new() -> Self {
        MemoryPool {
            pools: HashMap::new(),
            max_pool_size: 64, // Maximum number of cached allocations per size
            allocation_threshold: 1024, // Only pool allocations smaller than 1KB
        }
    }

    /// Set the maximum number of cached allocations per size
    pub fn with_max_pool_size(mut self, size: usize) -> Self {
        self.max_pool_size = size;
        self
    }

    /// Set the allocation threshold for pooling (in elements)
    pub fn with_allocation_threshold(mut self, threshold: usize) -> Self {
        self.allocation_threshold = threshold;
        self
    }

    /// Create a memory pool with custom configuration
    pub fn with_config(max_pool_size: usize, allocation_threshold: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_pool_size,
            allocation_threshold,
        }
    }

    /// Allocate from pool or create new
    ///
    /// If an allocation of the requested size is available in the pool, it will be reused.
    /// Otherwise, a new allocation will be created.
    pub fn allocate(&mut self, size: usize) -> Vec<T> {
        if size <= self.allocation_threshold {
            if let Some(pool) = self.pools.get_mut(&size) {
                if let Some(allocation) = pool.pop() {
                    return allocation;
                }
            }
        }

        // Create new allocation
        let mut vec = Vec::with_capacity(size);
        vec.resize(size, T::zero());
        vec
    }

    /// Allocate with specific initial value
    pub fn allocate_with_value(&mut self, size: usize, value: T) -> Vec<T> {
        if size <= self.allocation_threshold {
            if let Some(pool) = self.pools.get_mut(&size) {
                if let Some(mut allocation) = pool.pop() {
                    // Fill with the requested value
                    allocation.fill(value);
                    return allocation;
                }
            }
        }

        // Create new allocation with value
        vec![value; size]
    }

    /// Return allocation to pool
    ///
    /// The allocation will be cached for future reuse if it meets the pooling criteria.
    pub fn deallocate(&mut self, mut allocation: Vec<T>) {
        let size = allocation.len();

        if size <= self.allocation_threshold {
            let pool = self.pools.entry(size).or_default();

            if pool.len() < self.max_pool_size {
                // Clear the allocation but keep capacity
                allocation.clear();
                allocation.resize(size, T::zero());
                pool.push(allocation);
            }
        }
        // If not pooled, allocation will be dropped normally
    }

    /// Force deallocate without pooling (immediate free)
    pub fn deallocate_immediate(&mut self, allocation: Vec<T>) {
        drop(allocation); // Explicit drop for clarity
    }

    /// Clear all pools
    pub fn clear(&mut self) {
        self.pools.clear();
    }

    /// Clear pools for a specific size
    pub fn clear_size(&mut self, size: usize) {
        self.pools.remove(&size);
    }

    /// Shrink pools to target utilization
    pub fn shrink_to_fit(&mut self, target_utilization: f32) {
        let target_utilization = target_utilization.clamp(0.0, 1.0);

        for pool in self.pools.values_mut() {
            let target_size = (pool.len() as f32 * target_utilization) as usize;
            pool.truncate(target_size);
            pool.shrink_to_fit();
        }
    }

    /// Get statistics about pool usage
    pub fn stats(&self) -> PoolStats {
        let mut total_cached = 0;
        let mut total_sizes = 0;
        let mut largest_pool = 0;
        let mut memory_usage = 0;

        for (&size, pool) in &self.pools {
            total_cached += pool.len();
            total_sizes += size * pool.len();
            largest_pool = largest_pool.max(pool.len());
            memory_usage += size * pool.len() * std::mem::size_of::<T>();
        }

        PoolStats {
            pool_count: self.pools.len(),
            total_cached_allocations: total_cached,
            total_cached_elements: total_sizes,
            largest_pool_size: largest_pool,
            allocation_threshold: self.allocation_threshold,
            memory_usage_bytes: memory_usage,
            type_size: std::mem::size_of::<T>(),
        }
    }

    /// Get detailed statistics per pool size
    pub fn detailed_stats(&self) -> HashMap<usize, PoolSizeStats> {
        self.pools
            .iter()
            .map(|(&size, pool)| {
                let stats = PoolSizeStats {
                    element_size: size,
                    cached_allocations: pool.len(),
                    memory_usage_bytes: size * pool.len() * std::mem::size_of::<T>(),
                    utilization: pool.len() as f32 / self.max_pool_size as f32,
                };
                (size, stats)
            })
            .collect()
    }

    /// Warm up the pool with common allocation sizes
    pub fn warmup(&mut self, common_sizes: &[usize]) {
        for &size in common_sizes {
            if size <= self.allocation_threshold {
                let pool = self.pools.entry(size).or_default();

                // Pre-allocate a few common sizes
                let warmup_count = (self.max_pool_size / 4).max(1);
                for _ in 0..warmup_count {
                    if pool.len() < self.max_pool_size {
                        let mut vec = Vec::with_capacity(size);
                        vec.resize(size, T::zero());
                        pool.push(vec);
                    }
                }
            }
        }
    }

    /// Check if a size would be pooled
    pub fn would_pool(&self, size: usize) -> bool {
        size <= self.allocation_threshold
    }

    /// Get current pool capacity for a specific size
    pub fn pool_capacity(&self, size: usize) -> usize {
        self.pools.get(&size).map_or(0, |pool| pool.len())
    }

    /// Set maximum pool size (affects future allocations)
    pub fn set_max_pool_size(&mut self, max_size: usize) {
        self.max_pool_size = max_size;

        // Shrink existing pools if necessary
        for pool in self.pools.values_mut() {
            pool.truncate(max_size);
        }
    }

    /// Set allocation threshold (affects future allocations)
    pub fn set_allocation_threshold(&mut self, threshold: usize) {
        self.allocation_threshold = threshold;
    }
}

/// Statistics about memory pool usage
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub pool_count: usize,
    pub total_cached_allocations: usize,
    pub total_cached_elements: usize,
    pub largest_pool_size: usize,
    pub allocation_threshold: usize,
    pub memory_usage_bytes: usize,
    pub type_size: usize,
}

impl std::fmt::Display for PoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PoolStats(pools={}, cached={}, elements={}, memory={}KB, threshold={})",
            self.pool_count,
            self.total_cached_allocations,
            self.total_cached_elements,
            self.memory_usage_bytes / 1024,
            self.allocation_threshold
        )
    }
}

/// Statistics for a specific pool size
#[derive(Debug, Clone)]
pub struct PoolSizeStats {
    pub element_size: usize,
    pub cached_allocations: usize,
    pub memory_usage_bytes: usize,
    pub utilization: f32,
}

// Thread-local memory pools to reduce contention
thread_local! {
    static F32_POOL: std::cell::RefCell<MemoryPool<f32>> = std::cell::RefCell::new(MemoryPool::new());
    static F64_POOL: std::cell::RefCell<MemoryPool<f64>> = std::cell::RefCell::new(MemoryPool::new());
    static I32_POOL: std::cell::RefCell<MemoryPool<i32>> = std::cell::RefCell::new(MemoryPool::new());
    static I64_POOL: std::cell::RefCell<MemoryPool<i64>> = std::cell::RefCell::new(MemoryPool::new());
    static U32_POOL: std::cell::RefCell<MemoryPool<u32>> = std::cell::RefCell::new(MemoryPool::new());
    static U64_POOL: std::cell::RefCell<MemoryPool<u64>> = std::cell::RefCell::new(MemoryPool::new());
    static I8_POOL: std::cell::RefCell<MemoryPool<i8>> = std::cell::RefCell::new(MemoryPool::new());
    static U8_POOL: std::cell::RefCell<MemoryPool<u8>> = std::cell::RefCell::new(MemoryPool::new());
}

/// Allocate from thread-local pool
///
/// This function automatically dispatches to the appropriate thread-local pool
/// based on the element type.
pub fn allocate_pooled<T: TensorElement + 'static>(size: usize) -> Vec<T> {
    // Dispatch to appropriate thread-local pool based on type
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        F32_POOL.with(|pool| {
            let allocation = pool.borrow_mut().allocate(size);
            // Safety: We know T is f32 due to the TypeId check
            unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(allocation) }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        F64_POOL.with(|pool| {
            let allocation = pool.borrow_mut().allocate(size);
            // Safety: We know T is f64 due to the TypeId check
            unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(allocation) }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        I32_POOL.with(|pool| {
            let allocation = pool.borrow_mut().allocate(size);
            // Safety: We know T is i32 due to the TypeId check
            unsafe { std::mem::transmute::<Vec<i32>, Vec<T>>(allocation) }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        I64_POOL.with(|pool| {
            let allocation = pool.borrow_mut().allocate(size);
            // Safety: We know T is i64 due to the TypeId check
            unsafe { std::mem::transmute::<Vec<i64>, Vec<T>>(allocation) }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        U32_POOL.with(|pool| {
            let allocation = pool.borrow_mut().allocate(size);
            // Safety: We know T is u32 due to the TypeId check
            unsafe { std::mem::transmute::<Vec<u32>, Vec<T>>(allocation) }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        U64_POOL.with(|pool| {
            let allocation = pool.borrow_mut().allocate(size);
            // Safety: We know T is u64 due to the TypeId check
            unsafe { std::mem::transmute::<Vec<u64>, Vec<T>>(allocation) }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
        I8_POOL.with(|pool| {
            let allocation = pool.borrow_mut().allocate(size);
            // Safety: We know T is i8 due to the TypeId check
            unsafe { std::mem::transmute::<Vec<i8>, Vec<T>>(allocation) }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
        U8_POOL.with(|pool| {
            let allocation = pool.borrow_mut().allocate(size);
            // Safety: We know T is u8 due to the TypeId check
            unsafe { std::mem::transmute::<Vec<u8>, Vec<T>>(allocation) }
        })
    } else {
        // For types without pools, allocate normally
        let mut vec = Vec::with_capacity(size);
        vec.resize(size, T::zero());
        vec
    }
}

/// Allocate from thread-local pool with specific value
pub fn allocate_pooled_with_value<T: TensorElement + 'static>(size: usize, value: T) -> Vec<T> {
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        F32_POOL.with(|pool| {
            let value_f32 = unsafe { std::mem::transmute_copy::<T, f32>(&value) };
            let allocation = pool.borrow_mut().allocate_with_value(size, value_f32);
            unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(allocation) }
        })
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        F64_POOL.with(|pool| {
            let value_f64 = unsafe { std::mem::transmute_copy::<T, f64>(&value) };
            let allocation = pool.borrow_mut().allocate_with_value(size, value_f64);
            unsafe { std::mem::transmute::<Vec<f64>, Vec<T>>(allocation) }
        })
    } else {
        // For other types, use the generic function or regular allocation
        vec![value; size]
    }
}

/// Deallocate to thread-local pool
pub fn deallocate_pooled<T: TensorElement + 'static>(allocation: Vec<T>) {
    // Dispatch to appropriate thread-local pool based on type
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        F32_POOL.with(|pool| {
            // Safety: We know T is f32 due to the TypeId check
            let f32_allocation = unsafe { std::mem::transmute::<Vec<T>, Vec<f32>>(allocation) };
            pool.borrow_mut().deallocate(f32_allocation);
        });
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        F64_POOL.with(|pool| {
            // Safety: We know T is f64 due to the TypeId check
            let f64_allocation = unsafe { std::mem::transmute::<Vec<T>, Vec<f64>>(allocation) };
            pool.borrow_mut().deallocate(f64_allocation);
        });
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        I32_POOL.with(|pool| {
            // Safety: We know T is i32 due to the TypeId check
            let i32_allocation = unsafe { std::mem::transmute::<Vec<T>, Vec<i32>>(allocation) };
            pool.borrow_mut().deallocate(i32_allocation);
        });
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        I64_POOL.with(|pool| {
            // Safety: We know T is i64 due to the TypeId check
            let i64_allocation = unsafe { std::mem::transmute::<Vec<T>, Vec<i64>>(allocation) };
            pool.borrow_mut().deallocate(i64_allocation);
        });
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        U32_POOL.with(|pool| {
            let u32_allocation = unsafe { std::mem::transmute::<Vec<T>, Vec<u32>>(allocation) };
            pool.borrow_mut().deallocate(u32_allocation);
        });
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        U64_POOL.with(|pool| {
            let u64_allocation = unsafe { std::mem::transmute::<Vec<T>, Vec<u64>>(allocation) };
            pool.borrow_mut().deallocate(u64_allocation);
        });
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
        I8_POOL.with(|pool| {
            let i8_allocation = unsafe { std::mem::transmute::<Vec<T>, Vec<i8>>(allocation) };
            pool.borrow_mut().deallocate(i8_allocation);
        });
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
        U8_POOL.with(|pool| {
            let u8_allocation = unsafe { std::mem::transmute::<Vec<T>, Vec<u8>>(allocation) };
            pool.borrow_mut().deallocate(u8_allocation);
        });
    }
    // For types without pools, allocation is simply dropped
}

/// Get statistics for thread-local memory pools
pub fn pooled_memory_stats() -> HashMap<&'static str, PoolStats> {
    let mut stats = HashMap::new();

    stats.insert("f32", F32_POOL.with(|pool| pool.borrow().stats()));
    stats.insert("f64", F64_POOL.with(|pool| pool.borrow().stats()));
    stats.insert("i32", I32_POOL.with(|pool| pool.borrow().stats()));
    stats.insert("i64", I64_POOL.with(|pool| pool.borrow().stats()));
    stats.insert("u32", U32_POOL.with(|pool| pool.borrow().stats()));
    stats.insert("u64", U64_POOL.with(|pool| pool.borrow().stats()));
    stats.insert("i8", I8_POOL.with(|pool| pool.borrow().stats()));
    stats.insert("u8", U8_POOL.with(|pool| pool.borrow().stats()));

    stats
}

/// Clear all thread-local memory pools
pub fn clear_pooled_memory() {
    F32_POOL.with(|pool| pool.borrow_mut().clear());
    F64_POOL.with(|pool| pool.borrow_mut().clear());
    I32_POOL.with(|pool| pool.borrow_mut().clear());
    I64_POOL.with(|pool| pool.borrow_mut().clear());
    U32_POOL.with(|pool| pool.borrow_mut().clear());
    U64_POOL.with(|pool| pool.borrow_mut().clear());
    I8_POOL.with(|pool| pool.borrow_mut().clear());
    U8_POOL.with(|pool| pool.borrow_mut().clear());
}

/// Configure thread-local pools
pub fn configure_pools(config: PoolConfig) {
    F32_POOL.with(|pool| {
        let mut p = pool.borrow_mut();
        p.set_max_pool_size(config.max_pool_size);
        p.set_allocation_threshold(config.allocation_threshold);
    });
    F64_POOL.with(|pool| {
        let mut p = pool.borrow_mut();
        p.set_max_pool_size(config.max_pool_size);
        p.set_allocation_threshold(config.allocation_threshold);
    });
    // Apply to all other pools...
}

/// Configuration for memory pools
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_pool_size: usize,
    pub allocation_threshold: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 64,
            allocation_threshold: 1024,
        }
    }
}

/// Warm up all thread-local pools with common sizes
pub fn warmup_pools(common_sizes: &[usize]) {
    F32_POOL.with(|pool| pool.borrow_mut().warmup(common_sizes));
    F64_POOL.with(|pool| pool.borrow_mut().warmup(common_sizes));
    I32_POOL.with(|pool| pool.borrow_mut().warmup(common_sizes));
    I64_POOL.with(|pool| pool.borrow_mut().warmup(common_sizes));
    U32_POOL.with(|pool| pool.borrow_mut().warmup(common_sizes));
    U64_POOL.with(|pool| pool.borrow_mut().warmup(common_sizes));
    I8_POOL.with(|pool| pool.borrow_mut().warmup(common_sizes));
    U8_POOL.with(|pool| pool.borrow_mut().warmup(common_sizes));
}
