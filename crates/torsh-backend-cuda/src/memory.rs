//! CUDA memory management

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::error::{CudaError, CudaResult};

/// CUDA memory manager with pooling
#[derive(Debug)]
pub struct CudaMemoryManager {
    device_id: usize,
    pools: Mutex<HashMap<usize, MemoryPool>>,
    total_allocated: std::sync::atomic::AtomicUsize,
    peak_allocated: std::sync::atomic::AtomicUsize,
}

impl CudaMemoryManager {
    /// Create new memory manager for device
    pub fn new(device_id: usize) -> CudaResult<Self> {
        Ok(Self {
            device_id,
            pools: Mutex::new(HashMap::new()),
            total_allocated: std::sync::atomic::AtomicUsize::new(0),
            peak_allocated: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    
    /// Allocate memory
    pub fn allocate(&self, size: usize) -> CudaResult<CudaAllocation> {
        let size_class = self.size_class(size);
        let mut pools = self.pools.lock().unwrap();
        
        let pool = pools.entry(size_class).or_insert_with(|| MemoryPool::new(size_class));
        
        match pool.allocate() {
            Some(allocation) => {
                self.update_stats(size);
                Ok(allocation)
            },
            None => {
                // Pool exhausted, allocate new block
                let ptr = unsafe { cust::memory::cuda_malloc(size)? };
                let allocation = CudaAllocation::new(ptr, size, size_class);
                pool.add_allocation(allocation.clone());
                self.update_stats(size);
                Ok(allocation)
            }
        }
    }
    
    /// Deallocate memory
    pub fn deallocate(&self, allocation: CudaAllocation) -> CudaResult<()> {
        let mut pools = self.pools.lock().unwrap();
        
        if let Some(pool) = pools.get_mut(&allocation.size_class) {
            pool.deallocate(allocation);
        }
        
        Ok(())
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.total_allocated.load(std::sync::atomic::Ordering::Relaxed),
            peak_allocated: self.peak_allocated.load(std::sync::atomic::Ordering::Relaxed),
            device_id: self.device_id,
        }
    }
    
    /// Clear all memory pools
    pub fn clear(&self) -> CudaResult<()> {
        let mut pools = self.pools.lock().unwrap();
        for pool in pools.values_mut() {
            pool.clear()?;
        }
        pools.clear();
        
        self.total_allocated.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    
    fn size_class(&self, size: usize) -> usize {
        // Round up to nearest power of 2, minimum 256 bytes
        let min_size = 256;
        if size <= min_size {
            min_size
        } else {
            let next_power = (size - 1).next_power_of_two();
            next_power.max(min_size)
        }
    }
    
    fn update_stats(&self, size: usize) {
        let current = self.total_allocated.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
        let mut peak = self.peak_allocated.load(std::sync::atomic::Ordering::Relaxed);
        
        while current > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak, current, 
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }
}

/// Memory pool for specific size class
#[derive(Debug)]
struct MemoryPool {
    size_class: usize,
    free_blocks: Vec<CudaAllocation>,
    allocated_blocks: Vec<CudaAllocation>,
}

impl MemoryPool {
    fn new(size_class: usize) -> Self {
        Self {
            size_class,
            free_blocks: Vec::new(),
            allocated_blocks: Vec::new(),
        }
    }
    
    fn allocate(&mut self) -> Option<CudaAllocation> {
        self.free_blocks.pop()
    }
    
    fn deallocate(&mut self, allocation: CudaAllocation) {
        self.free_blocks.push(allocation);
    }
    
    fn add_allocation(&mut self, allocation: CudaAllocation) {
        self.allocated_blocks.push(allocation);
    }
    
    fn clear(&mut self) -> CudaResult<()> {
        for allocation in &self.allocated_blocks {
            unsafe {
                cust::memory::cuda_free(allocation.ptr)?;
            }
        }
        
        for allocation in &self.free_blocks {
            unsafe {
                cust::memory::cuda_free(allocation.ptr)?;
            }
        }
        
        self.allocated_blocks.clear();
        self.free_blocks.clear();
        Ok(())
    }
}

/// CUDA memory allocation
#[derive(Debug, Clone)]
pub struct CudaAllocation {
    ptr: cust::DevicePointer<u8>,
    size: usize,
    size_class: usize,
}

impl CudaAllocation {
    fn new(ptr: cust::DevicePointer<u8>, size: usize, size_class: usize) -> Self {
        Self { ptr, size, size_class }
    }
    
    /// Get device pointer
    pub fn ptr(&self) -> cust::DevicePointer<u8> {
        self.ptr
    }
    
    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Cast to typed pointer
    pub fn as_ptr<T>(&self) -> cust::DevicePointer<T> {
        self.ptr.cast()
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub device_id: usize,
}

/// Global memory manager instance
static MEMORY_MANAGERS: once_cell::sync::Lazy<Mutex<HashMap<usize, Arc<CudaMemoryManager>>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(HashMap::new()));

/// Get memory manager for device
pub fn get_memory_manager(device_id: usize) -> CudaResult<Arc<CudaMemoryManager>> {
    let mut managers = MEMORY_MANAGERS.lock().unwrap();
    
    if let Some(manager) = managers.get(&device_id) {
        Ok(Arc::clone(manager))
    } else {
        let manager = Arc::new(CudaMemoryManager::new(device_id)?);
        managers.insert(device_id, Arc::clone(&manager));
        Ok(manager)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_manager_creation() {
        let manager = CudaMemoryManager::new(0);
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_size_class_calculation() {
        let manager = CudaMemoryManager::new(0).unwrap();
        
        assert_eq!(manager.size_class(100), 256);
        assert_eq!(manager.size_class(256), 256);
        assert_eq!(manager.size_class(300), 512);
        assert_eq!(manager.size_class(1000), 1024);
    }
    
    #[test]
    fn test_memory_stats() {
        let manager = CudaMemoryManager::new(0).unwrap();
        let stats = manager.stats();
        
        assert_eq!(stats.device_id, 0);
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.peak_allocated, 0);
    }
    
    #[test]
    fn test_global_memory_manager() {
        let manager1 = get_memory_manager(0);
        let manager2 = get_memory_manager(0);
        
        assert!(manager1.is_ok());
        assert!(manager2.is_ok());
        
        // Should be the same instance
        assert!(Arc::ptr_eq(&manager1.unwrap(), &manager2.unwrap()));
    }
}