//! CPU Memory Management

use torsh_backends::buffer::BufferHandle;
use torsh_backends::memory::{MemoryManager, MemoryPool, MemoryStats, PoolStats};
use torsh_backends::{BackendError, BackendResult, Buffer, BufferDescriptor, Device};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::sync::{Arc, Mutex};

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap as HashMap, sync::Arc};
#[cfg(not(feature = "std"))]
use spin::Mutex;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// CPU memory manager implementation
#[derive(Debug, Clone)]
pub struct CpuMemoryManager {
    pools: Arc<Mutex<HashMap<usize, CpuMemoryPool>>>,
    stats: Arc<Mutex<MemoryStats>>,
}

impl CpuMemoryManager {
    /// Create a new CPU memory manager
    pub fn new() -> Self {
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(MemoryStats::default())),
        }
    }

    /// Get or create a memory pool for a specific size class
    fn get_or_create_pool(&self, size_class: usize) -> BackendResult<CpuMemoryPool> {
        let mut pools = self.pools.lock().map_err(|_| BackendError::MemoryError {
            reason: "Failed to acquire pools lock".to_string(),
        })?;

        if let Some(pool) = pools.get(&size_class) {
            Ok(pool.clone())
        } else {
            let pool = CpuMemoryPool::new(size_class);
            pools.insert(size_class, pool.clone());
            Ok(pool)
        }
    }

    /// Calculate size class for a given size (power of 2 rounding)
    fn calculate_size_class(size: usize) -> usize {
        if size <= 64 {
            64
        } else {
            size.next_power_of_two()
        }
    }
}

impl MemoryManager for CpuMemoryManager {
    fn allocate(&mut self, descriptor: &BufferDescriptor) -> Result<Buffer> {
        let size = descriptor.size;
        let alignment = descriptor.alignment.unwrap_or(std::mem::align_of::<u8>());
        let size_class = Self::calculate_size_class(size);
        let mut pool = self
            .get_or_create_pool(size_class)
            .map_err(|e| TorshError::AllocationError(e.to_string()))?;

        let ptr = pool
            .allocate(size, alignment)
            .map_err(|e| TorshError::AllocationError(e.to_string()))?;

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| TorshError::AllocationError("Failed to acquire stats lock".to_string()))?;
        stats.allocated_memory += size;
        stats.peak_memory = stats.peak_memory.max(stats.allocated_memory);
        stats.total_allocations += 1;
        stats.active_allocations += 1;

        Ok(Buffer::new(
            0, // ID - should be generated properly
            self.device().clone(),
            size,
            descriptor.usage,
            descriptor.clone(),
            BufferHandle::Cpu { ptr, size },
        ))
    }

    fn deallocate(&mut self, buffer: &Buffer) -> Result<()> {
        let size_class = Self::calculate_size_class(buffer.size);
        let mut pool = self
            .get_or_create_pool(size_class)
            .map_err(|e| TorshError::AllocationError(e.to_string()))?;

        // Extract pointer from buffer handle
        let ptr = match &buffer.handle {
            BufferHandle::Cpu { ptr, .. } => *ptr,
            _ => {
                return Err(TorshError::AllocationError(
                    "Invalid buffer handle type for CPU backend".to_string(),
                ))
            }
        };

        pool.deallocate(ptr, buffer.size)
            .map_err(|e| TorshError::AllocationError(e.to_string()))?;

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| TorshError::AllocationError("Failed to acquire stats lock".to_string()))?;
        stats.allocated_memory = stats.allocated_memory.saturating_sub(buffer.size);
        stats.total_deallocations += 1;
        stats.active_allocations = stats.active_allocations.saturating_sub(1);

        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    fn garbage_collect(&mut self) -> Result<usize> {
        // For CPU memory, we don't need explicit garbage collection
        Ok(0)
    }

    fn set_pool(&mut self, _pool: Box<dyn MemoryPool>) -> Result<()> {
        // Simple implementation - we manage our own pools
        Ok(())
    }

    fn device(&self) -> &Device {
        use std::sync::OnceLock;
        use torsh_backends::device::DeviceInfo;
        static CPU_DEVICE: OnceLock<Device> = OnceLock::new();
        CPU_DEVICE.get_or_init(|| Device {
            id: 0,
            device_type: DeviceType::Cpu,
            name: "CPU".to_string(),
            info: DeviceInfo {
                vendor: "CPU".to_string(),
                driver_version: "CPU Backend 1.0".to_string(),
                total_memory: 8 * 1024 * 1024 * 1024, // 8GB typical
                available_memory: 8 * 1024 * 1024 * 1024,
                compute_units: 1,
                max_work_group_size: u32::MAX as usize,
                max_work_group_dimensions: vec![u32::MAX as usize, 1, 1],
                clock_frequency_mhz: 3000, // 3GHz typical
                memory_bandwidth_gbps: 50.0,
                peak_gflops: 100.0,
                features: Vec::new(),
                properties: Vec::new(),
            },
        })
    }
}

impl Default for CpuMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU memory pool implementation
#[derive(Debug)]
pub struct CpuMemoryPool {
    size_class: usize,
    #[allow(clippy::arc_with_non_send_sync)]
    free_blocks: Arc<Mutex<Vec<*mut u8>>>,
    #[allow(clippy::arc_with_non_send_sync)]
    allocated_blocks: Arc<Mutex<HashMap<*mut u8, usize>>>,
}

// SAFETY: We ensure thread safety by using Mutex and proper synchronization
unsafe impl Send for CpuMemoryPool {}
unsafe impl Sync for CpuMemoryPool {}

impl Clone for CpuMemoryPool {
    fn clone(&self) -> Self {
        Self {
            size_class: self.size_class,
            free_blocks: Arc::clone(&self.free_blocks),
            allocated_blocks: Arc::clone(&self.allocated_blocks),
        }
    }
}

impl CpuMemoryPool {
    /// Create a new CPU memory pool
    pub fn new(size_class: usize) -> Self {
        #[allow(clippy::arc_with_non_send_sync)]
        let free_blocks = Arc::new(Mutex::new(Vec::new()));
        #[allow(clippy::arc_with_non_send_sync)]
        let allocated_blocks = Arc::new(Mutex::new(HashMap::new()));

        Self {
            size_class,
            free_blocks,
            allocated_blocks,
        }
    }
}

impl MemoryPool for CpuMemoryPool {
    fn allocate(&mut self, size: usize, alignment: usize) -> Result<*mut u8> {
        let mut free_blocks = self.free_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire free_blocks lock".to_string())
        })?;

        // Try to reuse a free block
        if let Some(ptr) = free_blocks.pop() {
            let mut allocated_blocks = self.allocated_blocks.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire allocated_blocks lock".to_string())
            })?;
            allocated_blocks.insert(ptr, size);
            return Ok(ptr);
        }

        // Allocate new block
        let layout = std::alloc::Layout::from_size_align(self.size_class, alignment)
            .map_err(|e| TorshError::AllocationError(format!("Invalid layout: {}", e)))?;

        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            return Err(TorshError::AllocationError(format!(
                "Failed to allocate {} bytes",
                size
            )));
        }

        let mut allocated_blocks = self.allocated_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocated_blocks lock".to_string())
        })?;
        allocated_blocks.insert(ptr, size);

        Ok(ptr)
    }

    fn deallocate(&mut self, ptr: *mut u8, _size: usize) -> Result<()> {
        let mut allocated_blocks = self.allocated_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocated_blocks lock".to_string())
        })?;

        if allocated_blocks.remove(&ptr).is_some() {
            let mut free_blocks = self.free_blocks.lock().map_err(|_| {
                TorshError::AllocationError("Failed to acquire free_blocks lock".to_string())
            })?;

            // Add to free list for reuse
            free_blocks.push(ptr);
            Ok(())
        } else {
            Err(TorshError::InvalidArgument(
                "Attempted to deallocate unknown pointer".to_string(),
            ))
        }
    }

    fn stats(&self) -> PoolStats {
        let allocated_blocks = self
            .allocated_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));
        let free_blocks = self
            .free_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));

        let allocated_bytes: usize = allocated_blocks.values().sum();
        let total_capacity = (allocated_blocks.len() + free_blocks.len()) * self.size_class;

        PoolStats {
            capacity: total_capacity,
            allocated: allocated_bytes,
            available: total_capacity - allocated_bytes,
            free_blocks: free_blocks.len(),
            allocated_blocks: allocated_blocks.len(),
            largest_free_block: if free_blocks.is_empty() {
                0
            } else {
                self.size_class
            },
            smallest_free_block: if free_blocks.is_empty() {
                0
            } else {
                self.size_class
            },
            average_free_block: if free_blocks.is_empty() {
                0
            } else {
                self.size_class
            },
        }
    }

    fn reset(&mut self) -> Result<()> {
        let mut allocated_blocks = self.allocated_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire allocated_blocks lock".to_string())
        })?;
        let mut free_blocks = self.free_blocks.lock().map_err(|_| {
            TorshError::AllocationError("Failed to acquire free_blocks lock".to_string())
        })?;

        // Deallocate all blocks
        for (&ptr, &size) in allocated_blocks.iter() {
            let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
            unsafe {
                std::alloc::dealloc(ptr, layout);
            }
        }

        for &ptr in free_blocks.iter() {
            let layout = std::alloc::Layout::from_size_align(self.size_class, 1).unwrap();
            unsafe {
                std::alloc::dealloc(ptr, layout);
            }
        }

        allocated_blocks.clear();
        free_blocks.clear();

        Ok(())
    }

    fn capacity(&self) -> usize {
        let allocated_blocks = self
            .allocated_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));
        let free_blocks = self
            .free_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));
        (allocated_blocks.len() + free_blocks.len()) * self.size_class
    }

    fn available(&self) -> usize {
        let free_blocks = self
            .free_blocks
            .lock()
            .unwrap_or_else(|_| panic!("Lock poisoned"));
        free_blocks.len() * self.size_class
    }
}

// Implement Drop for CpuMemoryPool to clean up allocated memory
impl Drop for CpuMemoryPool {
    fn drop(&mut self) {
        // Clean up any remaining allocated blocks
        if let Ok(allocated_blocks) = self.allocated_blocks.lock() {
            for (&ptr, &size) in allocated_blocks.iter() {
                let layout = std::alloc::Layout::from_size_align(size, 1).unwrap();
                unsafe {
                    std::alloc::dealloc(ptr, layout);
                }
            }
        }

        // Clean up free blocks
        if let Ok(free_blocks) = self.free_blocks.lock() {
            for &ptr in free_blocks.iter() {
                let layout = std::alloc::Layout::from_size_align(self.size_class, 1).unwrap();
                unsafe {
                    std::alloc::dealloc(ptr, layout);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_backends::buffer::BufferUsage;

    #[test]
    fn test_size_class_calculation() {
        assert_eq!(CpuMemoryManager::calculate_size_class(32), 64);
        assert_eq!(CpuMemoryManager::calculate_size_class(64), 64);
        assert_eq!(CpuMemoryManager::calculate_size_class(65), 128);
        assert_eq!(CpuMemoryManager::calculate_size_class(1000), 1024);
        assert_eq!(CpuMemoryManager::calculate_size_class(2048), 2048);
    }

    #[test]
    fn test_memory_manager_creation() {
        let manager = CpuMemoryManager::new();
        let stats = manager.stats();
        assert_eq!(stats.allocated_memory, 0);
    }
}
