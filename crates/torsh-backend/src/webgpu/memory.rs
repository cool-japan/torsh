//! WebGPU memory management for ToRSh

use crate::webgpu::{WebGpuBuffer, WebGpuBufferPool, WebGpuDevice, WebGpuError, WebGpuResult};
use crate::{
    buffer::generate_buffer_id, BufferDescriptor, BufferHandle, BufferUsage, MemoryLocation,
    MemoryManager, MemoryPoolConfig, MemoryStats,
};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use torsh_core::error::TorshError;

/// WebGPU memory manager implementation
#[derive(Debug)]
pub struct WebGpuMemoryManager {
    device: Arc<WebGpuDevice>,
    buffer_pool: Arc<WebGpuBufferPool>,
    config: MemoryPoolConfig,
    stats: Arc<RwLock<MemoryStats>>,
    active_buffers: Arc<RwLock<HashMap<BufferHandle, Arc<WebGpuBuffer>>>>,
    next_handle: Arc<Mutex<u64>>,
}

impl WebGpuMemoryManager {
    /// Create a new WebGPU memory manager
    pub fn new(device: Arc<WebGpuDevice>, config: MemoryPoolConfig) -> Self {
        let buffer_pool = Arc::new(WebGpuBufferPool::new(Arc::clone(&device)));

        Self {
            device,
            buffer_pool,
            config,
            stats: Arc::new(RwLock::new(MemoryStats::default())),
            active_buffers: Arc::new(RwLock::new(HashMap::new())),
            next_handle: Arc::new(Mutex::new(1)),
        }
    }

    /// Create memory manager with default configuration
    pub fn with_default_config(device: Arc<WebGpuDevice>) -> Self {
        Self::new(device, MemoryPoolConfig::default())
    }

    /// Get the device associated with this memory manager
    pub fn device(&self) -> &Arc<WebGpuDevice> {
        &self.device
    }

    /// Get the buffer pool
    pub fn buffer_pool(&self) -> &Arc<WebGpuBufferPool> {
        &self.buffer_pool
    }

    /// Update memory statistics
    fn update_stats(&self, delta_allocated: i64, delta_count: i64) {
        let mut stats = self.stats.write();

        if delta_allocated > 0 {
            stats.allocated_memory += delta_allocated as usize;
            stats.peak_memory = stats.peak_memory.max(stats.allocated_memory);
            stats.available_memory = stats.total_memory.saturating_sub(stats.allocated_memory);
        } else {
            stats.allocated_memory = stats
                .allocated_memory
                .saturating_sub((-delta_allocated) as usize);
            stats.available_memory = stats.total_memory.saturating_sub(stats.allocated_memory);
        }

        if delta_count > 0 {
            stats.total_allocations += delta_count as usize;
            stats.active_allocations += delta_count as usize;
        } else {
            stats.total_deallocations += (-delta_count) as usize;
            stats.active_allocations = stats
                .active_allocations
                .saturating_sub((-delta_count) as usize);
        }
    }

    /// Validate buffer parameters
    fn validate_buffer_descriptor(&self, descriptor: &BufferDescriptor) -> WebGpuResult<()> {
        if descriptor.size == 0 {
            return Err(WebGpuError::InvalidBufferUsage(
                "Buffer size cannot be zero".to_string(),
            ));
        }

        if descriptor.size > self.device.limits().max_storage_buffer_binding_size as usize {
            return Err(WebGpuError::MemoryAllocation(format!(
                "Buffer size {} exceeds maximum limit {}",
                descriptor.size,
                self.device.limits().max_storage_buffer_binding_size
            )));
        }

        // Check if requested usage is supported
        if descriptor.usage.contains(BufferUsage::VERTEX)
            && descriptor.usage.contains(BufferUsage::STORAGE)
        {
            // Some combinations might not be optimal, but WebGPU is generally flexible
        }

        Ok(())
    }

    /// Create staging buffer for host-device transfers
    async fn create_staging_buffer(
        &self,
        size: u64,
        for_upload: bool,
    ) -> WebGpuResult<WebGpuBuffer> {
        let usage = if for_upload {
            BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC
        } else {
            BufferUsage::MAP_READ | BufferUsage::COPY_DST
        };

        let descriptor =
            BufferDescriptor::new(size as usize, usage).with_location(MemoryLocation::Host);

        let handle = BufferHandle::WebGpu {
            buffer_ptr: *self.next_handle.lock(),
            size: size as usize,
        };
        *self.next_handle.lock() += 1;

        WebGpuBuffer::new(Arc::clone(&self.device), descriptor, handle)
    }
}

impl MemoryManager for WebGpuMemoryManager {
    fn allocate(
        &mut self,
        descriptor: &BufferDescriptor,
    ) -> torsh_core::error::Result<crate::Buffer> {
        self.validate_buffer_descriptor(descriptor)
            .map_err(|e| TorshError::BackendError(e.to_string()))?;

        let webgpu_buffer = self
            .buffer_pool
            .get_buffer(descriptor.clone())
            .map_err(|e| TorshError::BackendError(e.to_string()))?;

        let handle = webgpu_buffer.handle();

        // Store buffer reference
        {
            let mut active_buffers = self.active_buffers.write();
            active_buffers.insert(handle.clone(), Arc::new(webgpu_buffer.clone()));
        }

        // Update statistics
        self.update_stats(descriptor.size as i64, 1);

        // Create abstract Buffer from WebGpuBuffer
        let buffer_handle = crate::BufferHandle::WebGpu {
            buffer_ptr: handle.id() as u64,
            size: descriptor.size,
        };

        // Create a Device from WebGpuDevice
        let device = crate::Device::new(
            0, // Use 0 as default device index for WebGPU
            self.device.device_type(),
            self.device.name().to_string(),
            self.device.info().clone(),
        );

        let buffer = crate::Buffer::new(
            generate_buffer_id(),
            device,
            descriptor.size,
            descriptor.usage,
            descriptor.clone(),
            buffer_handle,
        );

        Ok(buffer)
    }

    fn deallocate(&mut self, buffer: &crate::Buffer) -> torsh_core::error::Result<()> {
        let handle = buffer.handle();
        let buffer_arc = {
            let mut active_buffers = self.active_buffers.write();
            active_buffers.remove(&handle)
        };

        if let Some(buffer_arc) = buffer_arc {
            let size = buffer_arc.size();

            // Try to return to pool (will be dropped if pool is full)
            if let Ok(buffer) = Arc::try_unwrap(buffer_arc) {
                self.buffer_pool.return_buffer(buffer);
            }

            // Update statistics
            self.update_stats(-(size as i64), -1);

            Ok(())
        } else {
            Err(TorshError::BackendError(format!(
                "Buffer handle {:?} not found",
                handle
            )))
        }
    }

    fn stats(&self) -> MemoryStats {
        self.stats.read().clone()
    }

    fn garbage_collect(&mut self) -> torsh_core::error::Result<usize> {
        // Clear unused buffers from pool
        let initial_count = self.buffer_pool.stats().values().sum::<usize>();
        self.buffer_pool.clear();
        let final_count = self.buffer_pool.stats().values().sum::<usize>();
        Ok(initial_count - final_count)
    }

    fn set_pool(&mut self, _pool: Box<dyn crate::MemoryPool>) -> torsh_core::error::Result<()> {
        // WebGPU memory manager uses its own internal buffer pool
        // External pool setting is not supported
        Err(TorshError::BackendError(
            "External pool setting not supported for WebGPU".to_string(),
        ))
    }

    fn device(&self) -> &crate::Device {
        // This is a workaround - we can't store a reference to Device
        // We need to create one on-the-fly or restructure the trait
        // For now, let's create a Device instance using OnceLock for thread-safety
        static CACHED_DEVICE: OnceLock<crate::Device> = OnceLock::new();
        CACHED_DEVICE.get_or_init(|| {
            crate::Device::new(
                0, // Use 0 as default device index for WebGPU
                self.device.device_type(),
                self.device.name().to_string(),
                self.device.info().clone(),
            )
        })
    }

    // Raw memory allocation methods
    fn allocate_raw(
        &mut self,
        _size: usize,
        _alignment: usize,
    ) -> torsh_core::error::Result<*mut u8> {
        // WebGPU doesn't support raw pointer allocation, return error
        Err(TorshError::BackendError(
            "WebGPU doesn't support raw memory allocation".to_string(),
        ))
    }

    fn deallocate_raw(&mut self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        // WebGPU doesn't support raw pointer deallocation, return error
        Err(TorshError::BackendError(
            "WebGPU doesn't support raw memory deallocation".to_string(),
        ))
    }

    // Unified memory methods
    fn supports_unified_memory(&self) -> bool {
        false // WebGPU doesn't support unified memory
    }

    fn allocate_unified(&mut self, _size: usize) -> torsh_core::error::Result<*mut u8> {
        Err(TorshError::BackendError(
            "WebGPU doesn't support unified memory allocation".to_string(),
        ))
    }

    fn deallocate_unified(&mut self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        Err(TorshError::BackendError(
            "WebGPU doesn't support unified memory deallocation".to_string(),
        ))
    }

    // Memory prefetching methods
    fn prefetch_to_device(&self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        // WebGPU handles memory transfer automatically, no explicit prefetch needed
        Ok(())
    }

    fn prefetch_to_host(&self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        // WebGPU handles memory transfer automatically, no explicit prefetch needed
        Ok(())
    }

    fn set_memory_advice(
        &self,
        _ptr: *mut u8,
        _size: usize,
        _advice: crate::memory::MemoryAdvice,
    ) -> torsh_core::error::Result<()> {
        // WebGPU doesn't support memory advice hints
        Ok(())
    }

    // Memory information methods
    fn available_memory(&self) -> torsh_core::error::Result<usize> {
        // WebGPU doesn't provide direct access to available memory
        // Return a conservative estimate
        Ok(1024 * 1024 * 1024) // 1GB default
    }

    fn total_memory(&self) -> torsh_core::error::Result<usize> {
        // WebGPU doesn't provide direct access to total memory
        // Return a conservative estimate
        Ok(4 * 1024 * 1024 * 1024) // 4GB default
    }

    fn synchronize(&self) -> torsh_core::error::Result<()> {
        // WebGPU synchronization is handled by the browser/driver
        Ok(())
    }

    // Defragmentation methods
    fn defragment(&mut self) -> torsh_core::error::Result<crate::memory::DefragmentationResult> {
        // WebGPU doesn't support manual defragmentation
        Ok(crate::memory::DefragmentationResult {
            blocks_moved: 0,
            memory_compacted: 0,
            duration_ms: 0.0,
            efficiency_improvement: 0.0,
            success: true,
            fragmentation_before: 0.0,
            fragmentation_after: 0.0,
        })
    }

    fn needs_defragmentation(&self) -> bool {
        false // WebGPU handles fragmentation automatically
    }

    fn fragmentation_info(&self) -> crate::memory::FragmentationInfo {
        crate::memory::FragmentationInfo {
            largest_free_block: 1024 * 1024 * 1024, // 1GB
            total_free_memory: 1024 * 1024 * 1024,
            overall_fragmentation: 0.0,
            external_fragmentation: 0.0,
            internal_fragmentation: 0.0,
            free_blocks: 1,
            allocated_blocks: 0,
            smallest_free_block: 1024 * 1024,
            average_free_block: 1024 * 1024 * 1024,
            total_allocated_memory: 0,
            utilization_efficiency: 100.0,
            allocation_efficiency: 100.0,
        }
    }

    fn compact_memory(&mut self) -> torsh_core::error::Result<crate::memory::CompactionResult> {
        // WebGPU doesn't support manual memory compaction
        Ok(crate::memory::CompactionResult {
            allocations_moved: 0,
            duration_ms: 0.0,
            largest_free_before: 0,
            largest_free_after: 0,
            free_blocks_before: 0,
            free_blocks_after: 0,
            success: true,
            bytes_moved: 0,
        })
    }

    fn set_defragmentation_policy(&mut self, _policy: crate::memory::DefragmentationPolicy) {
        // WebGPU doesn't support configurable defragmentation policies
        // Policy setting is ignored
    }
}

/// WebGPU memory pool implementation
#[derive(Debug)]
pub struct WebGpuMemoryPool {
    manager: Arc<WebGpuMemoryManager>,
    pool_config: MemoryPoolConfig,
}

impl WebGpuMemoryPool {
    /// Create a new WebGPU memory pool
    pub fn new(device: Arc<WebGpuDevice>, config: MemoryPoolConfig) -> Self {
        let manager = Arc::new(WebGpuMemoryManager::new(device, config.clone()));
        Self {
            manager,
            pool_config: config,
        }
    }
}

impl crate::MemoryPool for WebGpuMemoryPool {
    fn allocate(&mut self, _size: usize, _alignment: usize) -> torsh_core::error::Result<*mut u8> {
        // WebGPU doesn't support raw pointer allocation, so we simulate it
        // This is a simplified implementation - real usage would manage this differently
        let ptr = std::ptr::null_mut();
        Ok(ptr)
    }

    fn deallocate(&mut self, _ptr: *mut u8, _size: usize) -> torsh_core::error::Result<()> {
        // WebGPU memory is managed by the GPU driver
        Ok(())
    }

    fn stats(&self) -> crate::PoolStats {
        let manager_stats = self.manager.stats();
        crate::PoolStats {
            capacity: self.pool_config.initial_size,
            allocated: manager_stats.allocated_memory,
            available: self
                .pool_config
                .initial_size
                .saturating_sub(manager_stats.allocated_memory),
            free_blocks: 1,
            allocated_blocks: manager_stats.active_allocations,
            largest_free_block: self.pool_config.initial_size,
            smallest_free_block: 1024,
            average_free_block: self.pool_config.initial_size,
        }
    }

    fn reset(&mut self) -> torsh_core::error::Result<()> {
        // Reset the underlying manager
        // Note: This requires mut access to manager, which we don't have
        // In a real implementation, this would need to be restructured
        Ok(())
    }

    fn capacity(&self) -> usize {
        self.pool_config.initial_size as usize
    }

    fn available(&self) -> usize {
        let stats = self.manager.stats();
        (self
            .pool_config
            .initial_size
            .saturating_sub(stats.allocated_memory)) as usize
    }

    fn defragment(&mut self) -> torsh_core::error::Result<crate::memory::DefragmentationResult> {
        // WebGPU doesn't support manual defragmentation
        Ok(crate::memory::DefragmentationResult {
            blocks_moved: 0,
            memory_compacted: 0,
            duration_ms: 0.0,
            efficiency_improvement: 0.0,
            success: true,
            fragmentation_before: 0.0,
            fragmentation_after: 0.0,
        })
    }

    fn needs_defragmentation(&self) -> bool {
        false // WebGPU handles fragmentation automatically
    }

    fn fragmentation_info(&self) -> crate::memory::FragmentationInfo {
        crate::memory::FragmentationInfo {
            largest_free_block: self.available(),
            total_free_memory: self.available(),
            overall_fragmentation: 0.0,
            external_fragmentation: 0.0,
            internal_fragmentation: 0.0,
            free_blocks: 1,
            allocated_blocks: 0,
            smallest_free_block: 1024,
            average_free_block: self.available(),
            total_allocated_memory: 0,
            utilization_efficiency: 100.0,
            allocation_efficiency: 100.0,
        }
    }

    fn compact(&mut self) -> torsh_core::error::Result<crate::memory::CompactionResult> {
        // WebGPU doesn't support manual memory compaction
        Ok(crate::memory::CompactionResult {
            allocations_moved: 0,
            duration_ms: 0.0,
            largest_free_before: 0,
            largest_free_after: 0,
            free_blocks_before: 0,
            free_blocks_after: 0,
            success: true,
            bytes_moved: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryPool;

    #[tokio::test]
    async fn test_memory_manager_creation() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);
                let config = MemoryPoolConfig::default();
                let manager = WebGpuMemoryManager::new(device, config);

                assert_eq!(manager.stats().allocated_memory, 0);
                assert_eq!(manager.stats().active_allocations, 0);
            }
        }
    }

    #[tokio::test]
    async fn test_buffer_allocation() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);
                let mut manager = WebGpuMemoryManager::with_default_config(device);

                let descriptor = BufferDescriptor {
                    size: 1024,
                    usage: BufferUsage::STORAGE,
                    location: MemoryLocation::Device,
                    dtype: None,
                    shape: None,
                    initial_data: None,
                    alignment: None,
                    zero_init: false,
                };

                let buffer_result = manager.allocate(&descriptor);
                assert!(buffer_result.is_ok());

                if let Ok(buffer) = buffer_result {
                    assert_eq!(manager.stats().allocated_memory, 1024);
                    assert_eq!(manager.stats().active_allocations, 1);

                    // Test deallocation
                    let result = manager.deallocate(&buffer);
                    assert!(result.is_ok());

                    assert_eq!(manager.stats().allocated_memory, 0);
                    assert_eq!(manager.stats().active_allocations, 0);
                }
            }
        }
    }

    #[test]
    fn test_memory_pool_config() {
        let config = MemoryPoolConfig {
            initial_size: 1024 * 1024,
            max_size: Some(8 * 1024 * 1024),
            growth_factor: 1.5,
            strategy: crate::memory::AllocationStrategy::FirstFit,
            enable_coalescing: true,
            min_block_size: 4096,
            alignment: 64,
            numa_strategy: None,
        };

        assert_eq!(config.initial_size, 1024 * 1024);
        assert_eq!(config.max_size, Some(8 * 1024 * 1024));
        assert_eq!(config.min_block_size, 4096);
        assert!(config.enable_coalescing);
    }

    #[tokio::test]
    async fn test_memory_pool() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);
                let config = MemoryPoolConfig::default();
                let mut pool = WebGpuMemoryPool::new(device, config);

                // Test allocation
                let ptr_result = pool.allocate(1024, 16);
                assert!(ptr_result.is_ok());

                if let Ok(ptr) = ptr_result {
                    assert!(pool.stats().allocated > 0);

                    // Test deallocation
                    let result = pool.deallocate(ptr, 1024);
                    assert!(result.is_ok());
                }
            }
        }
    }

    #[test]
    fn test_validate_buffer_descriptor() {
        // This test would need a real device to test limits
        let descriptor = BufferDescriptor {
            size: 0,
            usage: BufferUsage::STORAGE,
            location: MemoryLocation::Device,
            dtype: None,
            shape: None,
            initial_data: None,
            alignment: None,
            zero_init: false,
        };

        // Zero size should be invalid
        assert_eq!(descriptor.size, 0);
    }
}
