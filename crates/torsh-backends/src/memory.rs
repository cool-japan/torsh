//! Memory management abstractions

use crate::{Buffer, BufferDescriptor, Device};
use torsh_core::error::{Result, TorshError};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Memory management interface
pub trait MemoryManager: Send + Sync {
    /// Allocate a buffer
    fn allocate(&mut self, descriptor: &BufferDescriptor) -> Result<Buffer>;

    /// Deallocate a buffer
    fn deallocate(&mut self, buffer: &Buffer) -> Result<()>;

    /// Get memory statistics
    fn stats(&self) -> MemoryStats;

    /// Garbage collect unused memory
    fn garbage_collect(&mut self) -> Result<usize>;

    /// Set memory pool for efficient allocation
    fn set_pool(&mut self, pool: Box<dyn MemoryPool>) -> Result<()>;

    /// Get the device this manager is for
    fn device(&self) -> &Device;
}

/// Memory pool interface for efficient allocation
pub trait MemoryPool: Send + Sync {
    /// Allocate memory from the pool
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut u8>;

    /// Deallocate memory back to the pool
    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()>;

    /// Get pool statistics
    fn stats(&self) -> PoolStats;

    /// Reset the pool (deallocate all memory)
    fn reset(&mut self) -> Result<()>;

    /// Get total pool capacity
    fn capacity(&self) -> usize;

    /// Get available memory in pool
    fn available(&self) -> usize;
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total device memory in bytes
    pub total_memory: usize,

    /// Currently allocated memory in bytes
    pub allocated_memory: usize,

    /// Available memory in bytes
    pub available_memory: usize,

    /// Peak memory usage in bytes
    pub peak_memory: usize,

    /// Number of active allocations
    pub active_allocations: usize,

    /// Total number of allocations made
    pub total_allocations: usize,

    /// Total number of deallocations made
    pub total_deallocations: usize,

    /// Memory fragmentation ratio (0.0 to 1.0)
    pub fragmentation: f32,

    /// Allocation efficiency (allocated / total)
    pub efficiency: f32,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_memory: 0,
            allocated_memory: 0,
            available_memory: 0,
            peak_memory: 0,
            active_allocations: 0,
            total_allocations: 0,
            total_deallocations: 0,
            fragmentation: 0.0,
            efficiency: 0.0,
        }
    }
}

impl MemoryStats {
    /// Calculate utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.total_memory == 0 {
            0.0
        } else {
            (self.allocated_memory as f32 / self.total_memory as f32) * 100.0
        }
    }

    /// Check if memory pressure is high
    pub fn is_under_pressure(&self) -> bool {
        self.utilization() > 90.0 || self.fragmentation > 0.5
    }
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total pool capacity in bytes
    pub capacity: usize,

    /// Currently allocated bytes from pool
    pub allocated: usize,

    /// Available bytes in pool
    pub available: usize,

    /// Number of free blocks
    pub free_blocks: usize,

    /// Number of allocated blocks
    pub allocated_blocks: usize,

    /// Largest free block size
    pub largest_free_block: usize,

    /// Smallest free block size
    pub smallest_free_block: usize,

    /// Average free block size
    pub average_free_block: usize,
}

/// Simple memory pool implementation using a free list
pub struct FreeListPool {
    /// Device this pool is for
    #[allow(dead_code)]
    device: Device,

    /// Total capacity
    capacity: usize,

    /// Base pointer to memory region
    base_ptr: *mut u8,

    /// Free blocks (offset, size)
    free_blocks: Vec<(usize, usize)>,

    /// Allocated blocks (offset, size)
    allocated_blocks: Vec<(usize, usize)>,

    /// Statistics
    stats: PoolStats,
}

impl FreeListPool {
    /// Create a new free list pool
    pub fn new(device: Device, capacity: usize) -> Result<Self> {
        // This is a simplified implementation
        // Real backends would allocate actual device memory
        let base_ptr = std::ptr::null_mut(); // Placeholder

        let free_blocks = vec![(0, capacity)];

        let stats = PoolStats {
            capacity,
            available: capacity,
            free_blocks: 1,
            largest_free_block: capacity,
            smallest_free_block: capacity,
            average_free_block: capacity,
            ..Default::default()
        };

        Ok(Self {
            device,
            capacity,
            base_ptr,
            free_blocks,
            allocated_blocks: Vec::new(),
            stats,
        })
    }

    /// Update pool statistics
    fn update_stats(&mut self) {
        self.stats.free_blocks = self.free_blocks.len();
        self.stats.allocated_blocks = self.allocated_blocks.len();
        self.stats.allocated = self.allocated_blocks.iter().map(|(_, size)| *size).sum();
        self.stats.available = self.capacity - self.stats.allocated;

        if !self.free_blocks.is_empty() {
            self.stats.largest_free_block = self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .max()
                .unwrap_or(0);
            self.stats.smallest_free_block = self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .min()
                .unwrap_or(0);
            self.stats.average_free_block = self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .sum::<usize>()
                / self.free_blocks.len();
        } else {
            self.stats.largest_free_block = 0;
            self.stats.smallest_free_block = 0;
            self.stats.average_free_block = 0;
        }
    }

    /// Find a suitable free block
    fn find_free_block(&self, size: usize, alignment: usize) -> Option<usize> {
        for (i, &(offset, block_size)) in self.free_blocks.iter().enumerate() {
            let aligned_offset = (offset + alignment - 1) & !(alignment - 1);
            let required_size = aligned_offset - offset + size;

            if required_size <= block_size {
                return Some(i);
            }
        }
        None
    }
}

impl MemoryPool for FreeListPool {
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut u8> {
        if let Some(block_idx) = self.find_free_block(size, alignment) {
            let (offset, block_size) = self.free_blocks[block_idx];
            let aligned_offset = (offset + alignment - 1) & !(alignment - 1);
            let padding = aligned_offset - offset;
            let required_size = padding + size;

            // Remove the free block
            self.free_blocks.remove(block_idx);

            // Add padding as a new free block if needed
            if padding > 0 {
                self.free_blocks.push((offset, padding));
            }

            // Add remaining space as a new free block if any
            if required_size < block_size {
                let remaining_offset = offset + required_size;
                let remaining_size = block_size - required_size;
                self.free_blocks.push((remaining_offset, remaining_size));
            }

            // Record the allocation
            self.allocated_blocks.push((aligned_offset, size));

            // Update statistics
            self.update_stats();

            // Return aligned pointer (simplified - real implementation would use actual memory)
            Ok(unsafe { self.base_ptr.add(aligned_offset) })
        } else {
            Err(TorshError::AllocationError(format!(
                "Out of memory: requested {} bytes, largest free block is {} bytes",
                size, self.stats.largest_free_block
            )))
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        // Calculate offset from base pointer
        // Safety: This is unsafe because it operates on raw pointers
        let offset = unsafe { ptr.offset_from(self.base_ptr) } as usize;

        // Find and remove the allocation
        if let Some(pos) = self
            .allocated_blocks
            .iter()
            .position(|&(off, sz)| off == offset && sz == size)
        {
            self.allocated_blocks.remove(pos);

            // Add back to free list
            self.free_blocks.push((offset, size));

            // TODO: Coalesce adjacent free blocks

            // Update statistics
            self.update_stats();

            Ok(())
        } else {
            Err(TorshError::InvalidArgument(
                "Invalid deallocation: block not found".to_string(),
            ))
        }
    }

    fn stats(&self) -> PoolStats {
        self.stats.clone()
    }

    fn reset(&mut self) -> Result<()> {
        self.free_blocks.clear();
        self.allocated_blocks.clear();
        self.free_blocks.push((0, self.capacity));
        self.update_stats();
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn available(&self) -> usize {
        self.stats.available
    }
}

unsafe impl Send for FreeListPool {}
unsafe impl Sync for FreeListPool {}

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// First fit - use the first block that's large enough
    FirstFit,

    /// Best fit - use the smallest block that's large enough
    BestFit,

    /// Worst fit - use the largest available block
    WorstFit,

    /// Next fit - like first fit but start from last allocation
    NextFit,
}

/// Memory allocation hint
#[derive(Debug, Clone)]
pub struct AllocationHint {
    /// Expected lifetime of the allocation
    pub lifetime: AllocationLifetime,

    /// Access pattern hint
    pub access_pattern: AccessPattern,

    /// Preferred allocation strategy
    pub strategy: AllocationStrategy,

    /// Whether to use memory pool
    pub use_pool: bool,
}

/// Expected allocation lifetime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationLifetime {
    /// Very short-lived (microseconds to milliseconds)
    Temporary,

    /// Short-lived (milliseconds to seconds)
    Short,

    /// Medium-lived (seconds to minutes)
    Medium,

    /// Long-lived (minutes to hours)
    Long,

    /// Persistent (hours to application lifetime)
    Persistent,
}

/// Memory access pattern hint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Random access
    Random,

    /// Sequential access
    Sequential,

    /// Mostly read operations
    ReadMostly,

    /// Mostly write operations
    WriteMostly,

    /// Streaming (write once, read sequentially)
    Streaming,
}

impl Default for AllocationHint {
    fn default() -> Self {
        Self {
            lifetime: AllocationLifetime::Medium,
            access_pattern: AccessPattern::Random,
            strategy: AllocationStrategy::FirstFit,
            use_pool: true,
        }
    }
}
