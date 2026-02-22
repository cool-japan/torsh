//! Memory management abstractions

use crate::{Buffer, BufferDescriptor, Device};
use torsh_core::error::{Result, TorshError};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Unified memory management interface across all backends
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

    /// Allocate raw memory with specific alignment requirements
    fn allocate_raw(&mut self, size: usize, alignment: usize) -> Result<*mut u8>;

    /// Deallocate raw memory
    fn deallocate_raw(&mut self, ptr: *mut u8, size: usize) -> Result<()>;

    /// Check if memory manager supports unified memory
    fn supports_unified_memory(&self) -> bool;

    /// Allocate unified memory (host-device accessible)
    fn allocate_unified(&mut self, size: usize) -> Result<*mut u8>;

    /// Deallocate unified memory
    fn deallocate_unified(&mut self, ptr: *mut u8, size: usize) -> Result<()>;

    /// Prefetch memory to device (for unified memory)
    fn prefetch_to_device(&self, ptr: *mut u8, size: usize) -> Result<()>;

    /// Prefetch memory to host (for unified memory)
    fn prefetch_to_host(&self, ptr: *mut u8, size: usize) -> Result<()>;

    /// Set memory access advice (for unified memory optimization)
    fn set_memory_advice(&self, ptr: *mut u8, size: usize, advice: MemoryAdvice) -> Result<()>;

    /// Get available memory on device
    fn available_memory(&self) -> Result<usize>;

    /// Get total memory on device
    fn total_memory(&self) -> Result<usize>;

    /// Synchronize all pending memory operations
    fn synchronize(&self) -> Result<()>;

    /// Defragment memory to reduce fragmentation
    fn defragment(&mut self) -> Result<DefragmentationResult>;

    /// Check if defragmentation is needed
    fn needs_defragmentation(&self) -> bool;

    /// Get memory fragmentation information
    fn fragmentation_info(&self) -> FragmentationInfo;

    /// Compact memory by moving allocated blocks together
    fn compact_memory(&mut self) -> Result<CompactionResult>;

    /// Set defragmentation policy
    fn set_defragmentation_policy(&mut self, policy: DefragmentationPolicy);
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

    /// Defragment the memory pool
    fn defragment(&mut self) -> Result<DefragmentationResult>;

    /// Check if the pool needs defragmentation
    fn needs_defragmentation(&self) -> bool;

    /// Get pool fragmentation information
    fn fragmentation_info(&self) -> FragmentationInfo;

    /// Compact allocated blocks in the pool
    fn compact(&mut self) -> Result<CompactionResult>;
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

// Note: FreeListPool implementation removed due to unsafe null pointer usage.
// Backend-specific memory pools should be implemented by each backend using
// proper memory allocation for their respective devices.

/// FreeListPool memory allocator implementation
#[derive(Debug)]
pub struct FreeListPool {
    /// Base pointer to the memory region
    base_ptr: *mut u8,
    /// Total size of the memory pool
    total_size: usize,
    /// List of free blocks as (offset, size) pairs
    free_blocks: Vec<(usize, usize)>,
    /// List of allocated blocks as (offset, size) pairs
    allocated_blocks: Vec<(usize, usize)>,
    /// Memory statistics
    stats: MemoryStats,
}

impl FreeListPool {
    /// Create a new FreeListPool with the given base pointer and size
    pub fn new(base_ptr: *mut u8, total_size: usize) -> Self {
        let mut pool = Self {
            base_ptr,
            total_size,
            free_blocks: vec![(0, total_size)],
            allocated_blocks: Vec::new(),
            stats: MemoryStats::default(),
        };
        pool.update_stats();
        pool
    }

    /// Find a suitable free block for the given size and alignment using first-fit strategy
    fn find_free_block(&self, size: usize, alignment: usize) -> Option<usize> {
        self.find_free_block_with_strategy(size, alignment, AllocationStrategy::FirstFit)
    }

    /// Find a suitable free block using the specified allocation strategy
    fn find_free_block_with_strategy(
        &self,
        size: usize,
        alignment: usize,
        strategy: AllocationStrategy,
    ) -> Option<usize> {
        match strategy {
            AllocationStrategy::FirstFit => self
                .free_blocks
                .iter()
                .enumerate()
                .find(|(_, &(offset, block_size))| {
                    let aligned_offset = (offset + alignment - 1) & !(alignment - 1);
                    let padding = aligned_offset - offset;
                    padding + size <= block_size
                })
                .map(|(idx, _)| idx),
            AllocationStrategy::BestFit => {
                let mut best_idx = None;
                let mut best_size = usize::MAX;

                for (idx, &(offset, block_size)) in self.free_blocks.iter().enumerate() {
                    let aligned_offset = (offset + alignment - 1) & !(alignment - 1);
                    let padding = aligned_offset - offset;

                    if padding + size <= block_size && block_size < best_size {
                        best_idx = Some(idx);
                        best_size = block_size;
                    }
                }

                best_idx
            }
            AllocationStrategy::WorstFit => {
                let mut worst_idx = None;
                let mut worst_size = 0;

                for (idx, &(offset, block_size)) in self.free_blocks.iter().enumerate() {
                    let aligned_offset = (offset + alignment - 1) & !(alignment - 1);
                    let padding = aligned_offset - offset;

                    if padding + size <= block_size && block_size > worst_size {
                        worst_idx = Some(idx);
                        worst_size = block_size;
                    }
                }

                worst_idx
            }
            AllocationStrategy::NextFit => {
                // For simplicity, fall back to first fit
                // In a real implementation, we'd maintain a cursor for next fit
                self.find_free_block_with_strategy(size, alignment, AllocationStrategy::FirstFit)
            }
        }
    }

    /// Update memory statistics
    fn update_stats(&mut self) {
        let allocated: usize = self.allocated_blocks.iter().map(|(_, size)| size).sum();
        let available: usize = self.free_blocks.iter().map(|(_, size)| size).sum();

        self.stats.allocated_memory = allocated;
        self.stats.available_memory = available;
        self.stats.active_allocations = self.allocated_blocks.len();
        self.stats.total_memory = self.total_size;
        self.stats.efficiency = if self.total_size > 0 {
            allocated as f32 / self.total_size as f32
        } else {
            0.0
        };
        self.stats.fragmentation = if available > 0 {
            1.0 - (self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .max()
                .unwrap_or(0) as f32
                / available as f32)
        } else {
            0.0
        };
    }

    /// Get the total capacity of the memory pool
    pub fn capacity(&self) -> usize {
        self.total_size
    }

    /// Coalesce adjacent free blocks to reduce fragmentation
    fn coalesce_free_blocks(&mut self) {
        if self.free_blocks.len() <= 1 {
            return;
        }

        // Sort free blocks by offset
        self.free_blocks.sort_by_key(|(offset, _)| *offset);

        // Coalesce adjacent blocks
        let mut i = 0;
        while i < self.free_blocks.len().saturating_sub(1) {
            let (offset1, size1) = self.free_blocks[i];
            let (offset2, size2) = self.free_blocks[i + 1];

            // Check if blocks are adjacent
            if offset1 + size1 == offset2 {
                // Merge blocks
                self.free_blocks[i] = (offset1, size1 + size2);
                self.free_blocks.remove(i + 1);
                // Don't increment i to check if this merged block can be further coalesced
            } else {
                i += 1;
            }
        }
    }

    /// Detect potential memory leaks by finding long-lived allocations
    pub fn detect_leaks(&self) -> Vec<LeakReport> {
        // In a real implementation, we'd track allocation timestamps
        // For now, report allocations that seem suspiciously large or numerous
        let mut reports = Vec::new();

        if self.allocated_blocks.len() > 1000 {
            reports.push(LeakReport {
                leak_type: LeakType::TooManyAllocations,
                block_count: self.allocated_blocks.len(),
                total_size: self.stats.allocated_memory,
                severity: LeakSeverity::High,
                description: format!(
                    "Too many active allocations: {}",
                    self.allocated_blocks.len()
                ),
            });
        }

        // Check for very large allocations that might be leaks
        for &(offset, size) in &self.allocated_blocks {
            if size > self.total_size / 4 {
                // More than 25% of total memory
                reports.push(LeakReport {
                    leak_type: LeakType::LargeAllocation,
                    block_count: 1,
                    total_size: size,
                    severity: LeakSeverity::Medium,
                    description: format!("Large allocation at offset {}: {} bytes", offset, size),
                });
            }
        }

        reports
    }

    /// Validate internal consistency of the memory pool
    pub fn validate_consistency(&self) -> Result<()> {
        // Check for overlapping allocated blocks
        for i in 0..self.allocated_blocks.len() {
            for j in (i + 1)..self.allocated_blocks.len() {
                let (offset1, size1) = self.allocated_blocks[i];
                let (offset2, size2) = self.allocated_blocks[j];

                let end1 = offset1 + size1;
                let end2 = offset2 + size2;

                if offset1 < end2 && offset2 < end1 {
                    return Err(TorshError::AllocationError(format!(
                        "Overlapping allocations detected: [{}, {}) and [{}, {})",
                        offset1, end1, offset2, end2
                    )));
                }
            }
        }

        // Check for overlapping free blocks
        for i in 0..self.free_blocks.len() {
            for j in (i + 1)..self.free_blocks.len() {
                let (offset1, size1) = self.free_blocks[i];
                let (offset2, size2) = self.free_blocks[j];

                let end1 = offset1 + size1;
                let end2 = offset2 + size2;

                if offset1 < end2 && offset2 < end1 {
                    return Err(TorshError::AllocationError(format!(
                        "Overlapping free blocks detected: [{}, {}) and [{}, {})",
                        offset1, end1, offset2, end2
                    )));
                }
            }
        }

        // Check for out-of-bounds blocks
        for &(offset, size) in &self.allocated_blocks {
            if offset + size > self.total_size {
                return Err(TorshError::AllocationError(format!(
                    "Allocated block extends beyond pool: offset={}, size={}, pool_size={}",
                    offset, size, self.total_size
                )));
            }
        }

        for &(offset, size) in &self.free_blocks {
            if offset + size > self.total_size {
                return Err(TorshError::AllocationError(format!(
                    "Free block extends beyond pool: offset={}, size={}, pool_size={}",
                    offset, size, self.total_size
                )));
            }
        }

        Ok(())
    }
}

impl MemoryPool for FreeListPool {
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut u8> {
        // Input validation
        if size == 0 {
            return Err(TorshError::InvalidArgument(
                "Allocation size cannot be zero".to_string(),
            ));
        }

        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(TorshError::InvalidArgument(format!(
                "Alignment must be a power of two and non-zero, got: {}",
                alignment
            )));
        }

        // Check for potential overflow in size + alignment
        if size > self.total_size || alignment > self.total_size {
            return Err(TorshError::AllocationError(format!(
                "Requested size ({}) or alignment ({}) exceeds pool capacity ({})",
                size, alignment, self.total_size
            )));
        }

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
                "Out of memory: requested {} bytes, available memory is {} bytes",
                size, self.stats.available_memory
            )))
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn deallocate(&mut self, ptr: *mut u8, size: usize) -> Result<()> {
        // Input validation
        if ptr.is_null() {
            return Err(TorshError::InvalidArgument(
                "Cannot deallocate null pointer".to_string(),
            ));
        }

        if size == 0 {
            return Err(TorshError::InvalidArgument(
                "Cannot deallocate zero-sized block".to_string(),
            ));
        }

        // Safety check: ensure pointer is within our memory range
        if ptr < self.base_ptr || ptr >= unsafe { self.base_ptr.add(self.total_size) } {
            return Err(TorshError::InvalidArgument(
                "Pointer outside of memory pool range".to_string(),
            ));
        }

        // Calculate offset from base pointer
        // Safety: This is unsafe because it operates on raw pointers, but we've validated the bounds
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

            // Coalesce adjacent free blocks
            self.coalesce_free_blocks();

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
        PoolStats {
            capacity: self.total_size,
            allocated: self.stats.allocated_memory,
            available: self.stats.available_memory,
            free_blocks: self.free_blocks.len(),
            allocated_blocks: self.allocated_blocks.len(),
            largest_free_block: self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .max()
                .unwrap_or(0),
            smallest_free_block: self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .min()
                .unwrap_or(0),
            average_free_block: if self.free_blocks.is_empty() {
                0
            } else {
                self.stats.available_memory / self.free_blocks.len()
            },
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.free_blocks.clear();
        self.allocated_blocks.clear();
        self.free_blocks.push((0, self.total_size));
        self.update_stats();
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.total_size
    }

    fn available(&self) -> usize {
        self.stats.available_memory
    }

    fn defragment(&mut self) -> Result<DefragmentationResult> {
        // Simple stub implementation for FreeListPool
        Ok(DefragmentationResult {
            blocks_moved: 0,
            memory_compacted: 0,
            duration_ms: 0.0,
            fragmentation_before: 0.0,
            fragmentation_after: 0.0,
            efficiency_improvement: 0.0,
            success: true,
        })
    }

    fn needs_defragmentation(&self) -> bool {
        // Simple heuristic: check if we have many small free blocks
        self.free_blocks.len() > 10
    }

    fn fragmentation_info(&self) -> FragmentationInfo {
        let free_blocks = self.free_blocks.len();
        let allocated_blocks = self.allocated_blocks.len();
        let total_free = self.stats.available_memory;
        let total_allocated = self.stats.allocated_memory;

        let largest_free = self
            .free_blocks
            .iter()
            .map(|(_, size)| *size)
            .max()
            .unwrap_or(0);

        let smallest_free = self
            .free_blocks
            .iter()
            .map(|(_, size)| *size)
            .min()
            .unwrap_or(0);

        let average_free = if free_blocks > 0 {
            total_free / free_blocks
        } else {
            0
        };

        let fragmentation = if self.capacity() > 0 {
            free_blocks as f32 / (free_blocks + allocated_blocks) as f32
        } else {
            0.0
        };

        FragmentationInfo {
            overall_fragmentation: fragmentation,
            external_fragmentation: fragmentation * 0.8,
            internal_fragmentation: fragmentation * 0.2,
            free_blocks,
            allocated_blocks,
            largest_free_block: largest_free,
            smallest_free_block: smallest_free,
            average_free_block: average_free,
            total_free_memory: total_free,
            total_allocated_memory: total_allocated,
            utilization_efficiency: if self.capacity() > 0 {
                total_allocated as f32 / self.capacity() as f32
            } else {
                0.0
            },
            allocation_efficiency: if self.capacity() > 0 {
                total_allocated as f32 / self.capacity() as f32
            } else {
                0.0
            },
        }
    }

    fn compact(&mut self) -> Result<CompactionResult> {
        // Simple stub implementation for FreeListPool
        let free_blocks_before = self.free_blocks.len();

        // Sort free blocks by offset to help with coalescing
        self.free_blocks.sort_by_key(|(offset, _)| *offset);

        // Try to coalesce adjacent free blocks
        let mut i = 0;
        while i < self.free_blocks.len().saturating_sub(1) {
            let (offset1, size1) = self.free_blocks[i];
            let (offset2, size2) = self.free_blocks[i + 1];

            if offset1 + size1 == offset2 {
                // Adjacent blocks, coalesce them
                self.free_blocks[i] = (offset1, size1 + size2);
                self.free_blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }

        let free_blocks_after = self.free_blocks.len();

        Ok(CompactionResult {
            allocations_moved: 0,
            bytes_moved: 0,
            duration_ms: 0.0,
            largest_free_before: self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .max()
                .unwrap_or(0),
            largest_free_after: self
                .free_blocks
                .iter()
                .map(|(_, size)| *size)
                .max()
                .unwrap_or(0),
            free_blocks_before,
            free_blocks_after,
            success: true,
        })
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

/// Memory advice for unified memory optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAdvice {
    /// Set preferred location for memory
    SetPreferredLocation,
    /// Unset preferred location
    UnsetPreferredLocation,
    /// Set which device can access this memory
    SetAccessedBy,
    /// Unset device access
    UnsetAccessedBy,
    /// Mark memory as read-mostly
    SetReadMostly,
    /// Unmark memory as read-mostly
    UnsetReadMostly,
}

/// Extended memory manager factory for creating backend-specific managers
pub trait MemoryManagerFactory: Send + Sync {
    /// Create a memory manager for the given device
    fn create_manager(&self, device: &Device) -> Result<Box<dyn MemoryManager>>;

    /// Get the backend type this factory supports
    fn backend_type(&self) -> crate::BackendType;

    /// Check if factory supports the given device
    fn supports_device(&self, device: &Device) -> bool;
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    /// Initial pool size in bytes
    pub initial_size: usize,

    /// Maximum pool size in bytes (None for unlimited)
    pub max_size: Option<usize>,

    /// Growth factor when pool needs to expand
    pub growth_factor: f32,

    /// Allocation strategy to use
    pub strategy: AllocationStrategy,

    /// Whether to enable memory coalescing
    pub enable_coalescing: bool,

    /// Minimum block size for allocations
    pub min_block_size: usize,

    /// Memory alignment requirement
    pub alignment: usize,

    /// NUMA allocation strategy (CPU backend only)
    pub numa_strategy: Option<crate::cpu::memory::NumaAllocationStrategy>,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 64 * 1024 * 1024, // 64MB
            max_size: None,
            growth_factor: 1.5,
            strategy: AllocationStrategy::FirstFit,
            enable_coalescing: true,
            min_block_size: 256,
            alignment: 16,
            numa_strategy: None,
        }
    }
}

impl MemoryPoolConfig {
    /// Create a new memory pool configuration
    pub fn new(initial_size: usize) -> Self {
        Self {
            initial_size,
            ..Default::default()
        }
    }

    /// Set maximum pool size
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = Some(max_size);
        self
    }

    /// Set growth factor
    pub fn with_growth_factor(mut self, growth_factor: f32) -> Self {
        self.growth_factor = growth_factor;
        self
    }

    /// Set allocation strategy
    pub fn with_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }
}

/// Defragmentation result information
#[derive(Debug, Clone)]
pub struct DefragmentationResult {
    /// Number of blocks moved during defragmentation
    pub blocks_moved: usize,

    /// Amount of memory compacted in bytes
    pub memory_compacted: usize,

    /// Time taken for defragmentation in milliseconds
    pub duration_ms: f64,

    /// Fragmentation level before defragmentation (0.0 to 1.0)
    pub fragmentation_before: f32,

    /// Fragmentation level after defragmentation (0.0 to 1.0)
    pub fragmentation_after: f32,

    /// Memory efficiency improvement (0.0 to 1.0)
    pub efficiency_improvement: f32,

    /// Whether defragmentation was successful
    pub success: bool,
}

impl DefragmentationResult {
    /// Check if defragmentation provided significant improvement
    pub fn is_improvement_significant(&self) -> bool {
        self.success && self.efficiency_improvement > 0.1 // 10% improvement
    }

    /// Get compaction ratio (memory_compacted / total_memory)
    pub fn compaction_ratio(&self, total_memory: usize) -> f32 {
        if total_memory == 0 {
            0.0
        } else {
            self.memory_compacted as f32 / total_memory as f32
        }
    }
}

/// Compaction result information
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Number of allocations moved
    pub allocations_moved: usize,

    /// Total bytes moved during compaction
    pub bytes_moved: usize,

    /// Time taken for compaction in milliseconds
    pub duration_ms: f64,

    /// Largest contiguous free block size before compaction
    pub largest_free_before: usize,

    /// Largest contiguous free block size after compaction
    pub largest_free_after: usize,

    /// Number of free blocks before compaction
    pub free_blocks_before: usize,

    /// Number of free blocks after compaction
    pub free_blocks_after: usize,

    /// Whether compaction was successful
    pub success: bool,
}

impl CompactionResult {
    /// Calculate free space consolidation improvement
    pub fn consolidation_improvement(&self) -> f32 {
        if self.free_blocks_before == 0 {
            1.0
        } else {
            1.0 - (self.free_blocks_after as f32 / self.free_blocks_before as f32)
        }
    }

    /// Calculate largest block improvement ratio
    pub fn largest_block_improvement(&self) -> f32 {
        if self.largest_free_before == 0 {
            if self.largest_free_after > 0 {
                f32::INFINITY
            } else {
                0.0
            }
        } else {
            self.largest_free_after as f32 / self.largest_free_before as f32
        }
    }
}

/// Detailed memory fragmentation information
#[derive(Debug, Clone, Default)]
pub struct FragmentationInfo {
    /// Overall fragmentation level (0.0 = no fragmentation, 1.0 = maximum fragmentation)
    pub overall_fragmentation: f32,

    /// External fragmentation (unused space due to allocation patterns)
    pub external_fragmentation: f32,

    /// Internal fragmentation (wasted space within allocated blocks)
    pub internal_fragmentation: f32,

    /// Number of free blocks
    pub free_blocks: usize,

    /// Number of allocated blocks
    pub allocated_blocks: usize,

    /// Size of largest free block
    pub largest_free_block: usize,

    /// Size of smallest free block
    pub smallest_free_block: usize,

    /// Average free block size
    pub average_free_block: usize,

    /// Total free memory
    pub total_free_memory: usize,

    /// Total allocated memory
    pub total_allocated_memory: usize,

    /// Memory utilization efficiency (0.0 to 1.0)
    pub utilization_efficiency: f32,

    /// Allocation/deallocation pattern efficiency
    pub allocation_efficiency: f32,
}

impl FragmentationInfo {
    /// Check if memory is severely fragmented
    pub fn is_severely_fragmented(&self) -> bool {
        self.overall_fragmentation > 0.7 || self.external_fragmentation > 0.6
    }

    /// Check if defragmentation would be beneficial
    pub fn would_benefit_from_defragmentation(&self) -> bool {
        self.is_severely_fragmented()
            || (self.free_blocks > 10 && self.utilization_efficiency < 0.8)
    }

    /// Get fragmentation severity level
    pub fn severity_level(&self) -> FragmentationSeverity {
        if self.overall_fragmentation < 0.2 {
            FragmentationSeverity::Low
        } else if self.overall_fragmentation < 0.5 {
            FragmentationSeverity::Medium
        } else if self.overall_fragmentation < 0.8 {
            FragmentationSeverity::High
        } else {
            FragmentationSeverity::Critical
        }
    }
}

/// Fragmentation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FragmentationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Defragmentation policy configuration
#[derive(Debug, Clone)]
pub struct DefragmentationPolicy {
    /// Automatic defragmentation trigger threshold (fragmentation level 0.0 to 1.0)
    pub auto_trigger_threshold: f32,

    /// Minimum time between automatic defragmentations in milliseconds
    pub min_interval_ms: u64,

    /// Maximum time allowed for defragmentation in milliseconds
    pub max_duration_ms: u64,

    /// Defragmentation strategy to use
    pub strategy: DefragmentationStrategy,

    /// Whether to enable background defragmentation
    pub enable_background: bool,

    /// Priority of defragmentation process
    pub priority: DefragmentationPriority,

    /// Whether to pause allocations during defragmentation
    pub pause_allocations: bool,

    /// Memory pressure threshold to trigger emergency defragmentation
    pub emergency_threshold: f32,
}

impl Default for DefragmentationPolicy {
    fn default() -> Self {
        Self {
            auto_trigger_threshold: 0.6,
            min_interval_ms: 10_000, // 10 seconds
            max_duration_ms: 5_000,  // 5 seconds
            strategy: DefragmentationStrategy::Incremental,
            enable_background: true,
            priority: DefragmentationPriority::Low,
            pause_allocations: false,
            emergency_threshold: 0.9,
        }
    }
}

/// Defragmentation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefragmentationStrategy {
    /// Compact all memory in one operation
    FullCompaction,

    /// Incremental defragmentation over time
    Incremental,

    /// Only move smaller allocations
    SmallBlocksOnly,

    /// Focus on largest free blocks
    LargeBlocksFirst,

    /// Minimize movement, focus on coalescing
    CoalesceOnly,

    /// Use generational approach (move old allocations)
    Generational,
}

/// Defragmentation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DefragmentationPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Memory leak detection report
#[derive(Debug, Clone)]
pub struct LeakReport {
    /// Type of potential leak detected
    pub leak_type: LeakType,
    /// Number of blocks involved
    pub block_count: usize,
    /// Total size of potentially leaked memory
    pub total_size: usize,
    /// Severity of the potential leak
    pub severity: LeakSeverity,
    /// Human-readable description
    pub description: String,
}

/// Types of memory leaks that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeakType {
    /// Too many small allocations that haven't been freed
    TooManyAllocations,
    /// Large allocation that might be a leak
    LargeAllocation,
    /// Long-lived allocation that might be forgotten
    LongLivedAllocation,
    /// Fragmentation causing inefficient memory use
    Fragmentation,
}

/// Severity levels for memory leaks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LeakSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceInfo};
    use torsh_core::device::DeviceType;

    fn create_test_device() -> Device {
        let info = DeviceInfo::default();
        Device::new(0, DeviceType::Cpu, "Test CPU".to_string(), info)
    }

    #[test]
    fn test_memory_stats_default() {
        let stats = MemoryStats::default();

        assert_eq!(stats.total_memory, 0);
        assert_eq!(stats.allocated_memory, 0);
        assert_eq!(stats.available_memory, 0);
        assert_eq!(stats.peak_memory, 0);
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.total_allocations, 0);
        assert_eq!(stats.total_deallocations, 0);
        assert_eq!(stats.fragmentation, 0.0);
        assert_eq!(stats.efficiency, 0.0);
    }

    #[test]
    fn test_memory_stats_utilization() {
        let mut stats = MemoryStats {
            total_memory: 1000,
            allocated_memory: 300,
            ..Default::default()
        };

        assert!((stats.utilization() - 30.0).abs() < 0.001);

        stats.total_memory = 0;
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn test_memory_stats_pressure() {
        let mut stats = MemoryStats {
            total_memory: 1000,
            allocated_memory: 850, // 85% utilization
            fragmentation: 0.3,    // 30% fragmentation
            ..Default::default()
        };

        assert!(!stats.is_under_pressure()); // Not quite at 90%

        stats.allocated_memory = 950; // 95% utilization
        assert!(stats.is_under_pressure()); // Over 90%

        stats.allocated_memory = 500; // 50% utilization
        stats.fragmentation = 0.6; // 60% fragmentation
        assert!(stats.is_under_pressure()); // High fragmentation
    }

    #[test]
    fn test_pool_stats_default() {
        let stats = PoolStats::default();

        assert_eq!(stats.capacity, 0);
        assert_eq!(stats.allocated, 0);
        assert_eq!(stats.available, 0);
        assert_eq!(stats.free_blocks, 0);
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.largest_free_block, 0);
        assert_eq!(stats.smallest_free_block, 0);
        assert_eq!(stats.average_free_block, 0);
    }

    #[test]
    fn test_allocation_strategy_variants() {
        let strategies = [
            AllocationStrategy::FirstFit,
            AllocationStrategy::BestFit,
            AllocationStrategy::WorstFit,
            AllocationStrategy::NextFit,
        ];

        // Ensure all strategies are distinct
        for (i, strategy1) in strategies.iter().enumerate() {
            for (j, strategy2) in strategies.iter().enumerate() {
                if i != j {
                    assert_ne!(strategy1, strategy2);
                }
            }
        }
    }

    #[test]
    fn test_allocation_lifetime_variants() {
        let lifetimes = [
            AllocationLifetime::Temporary,
            AllocationLifetime::Short,
            AllocationLifetime::Medium,
            AllocationLifetime::Long,
            AllocationLifetime::Persistent,
        ];

        // Ensure all lifetimes are distinct
        for (i, lifetime1) in lifetimes.iter().enumerate() {
            for (j, lifetime2) in lifetimes.iter().enumerate() {
                if i != j {
                    assert_ne!(lifetime1, lifetime2);
                }
            }
        }
    }

    #[test]
    fn test_access_pattern_variants() {
        let patterns = [
            AccessPattern::Random,
            AccessPattern::Sequential,
            AccessPattern::ReadMostly,
            AccessPattern::WriteMostly,
            AccessPattern::Streaming,
        ];

        // Ensure all patterns are distinct
        for (i, pattern1) in patterns.iter().enumerate() {
            for (j, pattern2) in patterns.iter().enumerate() {
                if i != j {
                    assert_ne!(pattern1, pattern2);
                }
            }
        }
    }

    #[test]
    fn test_allocation_hint_default() {
        let hint = AllocationHint::default();

        assert_eq!(hint.lifetime, AllocationLifetime::Medium);
        assert_eq!(hint.access_pattern, AccessPattern::Random);
        assert_eq!(hint.strategy, AllocationStrategy::FirstFit);
        assert!(hint.use_pool);
    }

    #[test]
    fn test_memory_pool_config_default() {
        let config = MemoryPoolConfig::default();

        assert_eq!(config.initial_size, 64 * 1024 * 1024); // 64MB
        assert_eq!(config.max_size, None);
        assert_eq!(config.growth_factor, 1.5);
        assert_eq!(config.strategy, AllocationStrategy::FirstFit);
        assert!(config.enable_coalescing);
        assert_eq!(config.min_block_size, 256);
        assert_eq!(config.alignment, 16);
    }

    #[test]
    fn test_memory_pool_config_builder() {
        let config = MemoryPoolConfig::new(128 * 1024 * 1024) // 128MB
            .with_max_size(1024 * 1024 * 1024) // 1GB
            .with_growth_factor(2.0)
            .with_strategy(AllocationStrategy::BestFit)
            .with_alignment(64);

        assert_eq!(config.initial_size, 128 * 1024 * 1024);
        assert_eq!(config.max_size, Some(1024 * 1024 * 1024));
        assert_eq!(config.growth_factor, 2.0);
        assert_eq!(config.strategy, AllocationStrategy::BestFit);
        assert_eq!(config.alignment, 64);
    }

    #[test]
    fn test_free_list_pool_creation() {
        let _device = create_test_device();
        let capacity = 1024 * 1024; // 1MB

        // Allocate actual memory for the test
        let layout = std::alloc::Layout::from_size_align(capacity, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());

        let pool = FreeListPool::new(ptr, capacity);
        assert_eq!(pool.capacity(), capacity);
        assert_eq!(pool.available(), capacity);

        let stats = pool.stats();
        assert_eq!(stats.capacity, capacity);
        assert_eq!(stats.available, capacity);
        assert_eq!(stats.allocated, 0);
        assert_eq!(stats.free_blocks, 1);
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.largest_free_block, capacity);

        // Clean up allocated memory
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_free_list_pool_allocation() {
        let _device = create_test_device();
        let capacity = 1024;

        // Allocate actual memory for the test
        let layout = std::alloc::Layout::from_size_align(capacity, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());

        let mut pool = FreeListPool::new(ptr, capacity);

        // Allocate some memory
        let ptr1 = pool.allocate(256, 16);
        assert!(ptr1.is_ok());

        let stats = pool.stats();
        assert_eq!(stats.allocated, 256);
        assert!(stats.available < capacity); // Should be less due to alignment
        assert_eq!(stats.allocated_blocks, 1);

        // Allocate more memory
        let ptr2 = pool.allocate(128, 16);
        assert!(ptr2.is_ok());

        let stats = pool.stats();
        assert_eq!(stats.allocated, 256 + 128);
        assert_eq!(stats.allocated_blocks, 2);

        // Try to allocate more than available
        let ptr3 = pool.allocate(1024, 16);
        assert!(ptr3.is_err());

        // Clean up allocated memory
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_free_list_pool_deallocation() {
        let _device = create_test_device();
        let capacity = 1024;

        // Allocate actual memory for the test
        let layout = std::alloc::Layout::from_size_align(capacity, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());

        let mut pool = FreeListPool::new(ptr, capacity);

        // Allocate some memory
        let ptr1 = pool.allocate(256, 16).unwrap();
        let ptr2 = pool.allocate(128, 16).unwrap();

        assert_eq!(pool.stats().allocated_blocks, 2);

        // Deallocate first allocation
        let result = pool.deallocate(ptr1, 256);
        assert!(result.is_ok());

        let stats = pool.stats();
        assert_eq!(stats.allocated, 128);
        assert_eq!(stats.allocated_blocks, 1);

        // Deallocate second allocation
        let result = pool.deallocate(ptr2, 128);
        assert!(result.is_ok());

        let stats = pool.stats();
        assert_eq!(stats.allocated, 0);
        assert_eq!(stats.allocated_blocks, 0);

        // Clean up allocated memory
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_free_list_pool_reset() {
        let _device = create_test_device();
        let capacity = 1024;

        // Allocate actual memory for the test
        let layout = std::alloc::Layout::from_size_align(capacity, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());

        let mut pool = FreeListPool::new(ptr, capacity);

        // Allocate some memory
        let _ptr1 = pool.allocate(256, 16).unwrap();
        let _ptr2 = pool.allocate(128, 16).unwrap();

        assert_eq!(pool.stats().allocated_blocks, 2);

        // Reset the pool
        let result = pool.reset();
        assert!(result.is_ok());

        let stats = pool.stats();
        assert_eq!(stats.allocated, 0);
        assert_eq!(stats.allocated_blocks, 0);
        assert_eq!(stats.free_blocks, 1);
        assert_eq!(stats.available, capacity);
        assert_eq!(stats.largest_free_block, capacity);

        // Clean up allocated memory
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }

    #[test]
    fn test_free_list_pool_find_free_block() {
        let _device = create_test_device();
        let capacity = 1024;

        // Allocate actual memory for the test
        let layout = std::alloc::Layout::from_size_align(capacity, 8).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());

        let pool = FreeListPool::new(ptr, capacity);

        // Should find a block for reasonable allocation
        let block_idx = pool.find_free_block(256, 16);
        assert!(block_idx.is_some());
        assert_eq!(block_idx.unwrap(), 0); // First (and only) block

        // Should not find a block for oversized allocation
        let block_idx = pool.find_free_block(2048, 16);
        assert!(block_idx.is_none());

        // Clean up allocated memory
        unsafe {
            std::alloc::dealloc(ptr, layout);
        }
    }
}
