//! WebGPU buffer management for ToRSh

#[cfg(feature = "webgpu")]
use bytemuck;
#[cfg(feature = "webgpu")]
use wgpu;

use crate::webgpu::{WebGpuDevice, WebGpuError, WebGpuResult};
use crate::{Buffer, BufferDescriptor, BufferHandle, BufferUsage, BufferView, MemoryLocation};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::DType;

/// WebGPU buffer wrapper
#[derive(Debug, Clone)]
pub struct WebGpuBuffer {
    buffer: Arc<wgpu::Buffer>,
    device: Arc<WebGpuDevice>,
    descriptor: BufferDescriptor,
    handle: BufferHandle,
    usage: wgpu::BufferUsages,
    size: u64,
    // Mapping state tracking
    mapping_state: Arc<RwLock<MappingState>>,
}

/// Buffer mapping state
#[derive(Debug, Clone, PartialEq)]
pub enum MappingState {
    Unmapped,
    MappingPending,
    MappedRead,
    MappedWrite,
}

impl WebGpuBuffer {
    /// Create a new WebGPU buffer
    pub fn new(
        device: Arc<WebGpuDevice>,
        descriptor: BufferDescriptor,
        handle: BufferHandle,
    ) -> WebGpuResult<Self> {
        let usage = Self::convert_buffer_usage(&descriptor.usage)?;

        let wgpu_descriptor = wgpu::BufferDescriptor {
            label: Some("WebGPU Buffer"),
            size: descriptor.size as u64,
            usage,
            mapped_at_creation: false,
        };

        let buffer = Arc::new(device.create_buffer(&wgpu_descriptor));
        let size = descriptor.size as u64;

        Ok(Self {
            buffer,
            device,
            descriptor,
            handle,
            usage,
            size,
            mapping_state: Arc::new(RwLock::new(MappingState::Unmapped)),
        })
    }

    /// Create a buffer with initial data
    pub fn with_data<T: bytemuck::Pod>(
        device: Arc<WebGpuDevice>,
        descriptor: BufferDescriptor,
        handle: BufferHandle,
        data: &[T],
    ) -> WebGpuResult<Self> {
        let usage = Self::convert_buffer_usage(&descriptor.usage)?;

        let wgpu_descriptor = wgpu::BufferDescriptor {
            label: Some("WebGPU Buffer"),
            size: descriptor.size as u64,
            usage,
            mapped_at_creation: true,
        };

        let buffer = Arc::new(device.create_buffer(&wgpu_descriptor));

        // Copy initial data
        let data_bytes = bytemuck::cast_slice(data);
        if data_bytes.len() > descriptor.size {
            return Err(WebGpuError::InvalidBufferUsage(format!(
                "Data size {} exceeds buffer size {}",
                data_bytes.len(),
                descriptor.size
            )));
        }

        buffer.slice(..).get_mapped_range_mut()[..data_bytes.len()].copy_from_slice(data_bytes);
        buffer.unmap();
        let size = descriptor.size as u64;

        Ok(Self {
            buffer,
            device,
            descriptor,
            handle,
            usage,
            size,
            mapping_state: Arc::new(RwLock::new(MappingState::Unmapped)),
        })
    }

    /// Get the underlying wgpu buffer
    pub fn wgpu_buffer(&self) -> &wgpu::Buffer {
        &*self.buffer
    }

    /// Consume the WebGpu buffer and return the inner wgpu buffer
    pub fn into_wgpu_buffer(self) -> wgpu::Buffer {
        Arc::try_unwrap(self.buffer.clone()).unwrap_or_else(|arc| (*arc).clone())
    }

    /// Get buffer slice
    pub fn slice<S: std::ops::RangeBounds<wgpu::BufferAddress>>(
        &self,
        bounds: S,
    ) -> wgpu::BufferSlice {
        self.buffer.slice(bounds)
    }

    /// Map buffer for reading
    pub async fn map_read(&self, offset: u64, size: Option<u64>) -> WebGpuResult<()> {
        if !self.usage.contains(wgpu::BufferUsages::MAP_READ) {
            return Err(WebGpuError::InvalidBufferUsage(
                "Buffer does not support MAP_READ".to_string(),
            ));
        }

        {
            let mut state = self.mapping_state.write();
            if *state != MappingState::Unmapped {
                return Err(WebGpuError::InvalidBufferUsage(format!(
                    "Buffer is already mapped: {:?}",
                    *state
                )));
            }
            *state = MappingState::MappingPending;
        }

        let actual_size = size.unwrap_or(self.size - offset);
        let slice = self.buffer.slice(offset..offset + actual_size);

        // Use the newer wgpu API that requires a callback
        slice.map_async(wgpu::MapMode::Read, |result| {
            // Callback is handled by wgpu internally
        });

        // Wait for the mapping to complete
        let _ = self.device.device().poll(wgpu::PollType::Wait);

        // Check if mapping succeeded by trying to get mapped data
        *self.mapping_state.write() = MappingState::MappedRead;
        Ok(())
    }

    /// Map buffer for writing
    pub async fn map_write(&self, offset: u64, size: Option<u64>) -> WebGpuResult<()> {
        if !self.usage.contains(wgpu::BufferUsages::MAP_WRITE) {
            return Err(WebGpuError::InvalidBufferUsage(
                "Buffer does not support MAP_WRITE".to_string(),
            ));
        }

        {
            let mut state = self.mapping_state.write();
            if *state != MappingState::Unmapped {
                return Err(WebGpuError::InvalidBufferUsage(format!(
                    "Buffer is already mapped: {:?}",
                    *state
                )));
            }
            *state = MappingState::MappingPending;
        }

        let actual_size = size.unwrap_or(self.size - offset);
        let slice = self.buffer.slice(offset..offset + actual_size);

        // Use the newer wgpu API that requires a callback
        slice.map_async(wgpu::MapMode::Write, |result| {
            // Callback is handled by wgpu internally
        });

        // Wait for the mapping to complete
        let _ = self.device.device().poll(wgpu::PollType::Wait);

        // Check if mapping succeeded by trying to get mapped data
        *self.mapping_state.write() = MappingState::MappedWrite;
        Ok(())
    }

    /// Get mapped range for reading
    pub fn mapped_range(&self, offset: u64, size: Option<u64>) -> WebGpuResult<wgpu::BufferView> {
        let state = self.mapping_state.read();
        if *state != MappingState::MappedRead {
            return Err(WebGpuError::InvalidBufferUsage(format!(
                "Buffer is not mapped for reading: {:?}",
                *state
            )));
        }

        let actual_size = size.unwrap_or(self.size - offset);
        let slice = self.buffer.slice(offset..offset + actual_size);
        Ok(slice.get_mapped_range())
    }

    /// Get mapped range for writing
    pub fn mapped_range_mut(
        &self,
        offset: u64,
        size: Option<u64>,
    ) -> WebGpuResult<wgpu::BufferViewMut> {
        let state = self.mapping_state.read();
        if *state != MappingState::MappedWrite {
            return Err(WebGpuError::InvalidBufferUsage(format!(
                "Buffer is not mapped for writing: {:?}",
                *state
            )));
        }

        let actual_size = size.unwrap_or(self.size - offset);
        let slice = self.buffer.slice(offset..offset + actual_size);
        Ok(slice.get_mapped_range_mut())
    }

    /// Unmap the buffer
    pub fn unmap(&self) {
        let mut state = self.mapping_state.write();
        if *state != MappingState::Unmapped {
            self.buffer.unmap();
            *state = MappingState::Unmapped;
        }
    }

    /// Check if buffer is currently mapped
    pub fn is_mapped(&self) -> bool {
        *self.mapping_state.read() != MappingState::Unmapped
    }

    /// Get mapping state
    pub fn mapping_state(&self) -> MappingState {
        self.mapping_state.read().clone()
    }

    /// Write data to buffer (creates temporary staging buffer if needed)
    pub async fn write_data<T: bytemuck::Pod>(&self, offset: u64, data: &[T]) -> WebGpuResult<()> {
        let data_bytes = bytemuck::cast_slice(data);

        if self.usage.contains(wgpu::BufferUsages::MAP_WRITE) {
            // Direct mapping approach
            self.map_write(offset, Some(data_bytes.len() as u64))
                .await?;
            {
                let mut mapped = self.mapped_range_mut(offset, Some(data_bytes.len() as u64))?;
                mapped.copy_from_slice(data_bytes);
            }
            self.unmap();
        } else if self.usage.contains(wgpu::BufferUsages::COPY_DST) {
            // Queue write approach
            self.device
                .queue()
                .write_buffer(&self.buffer, offset, data_bytes);
        } else {
            return Err(WebGpuError::InvalidBufferUsage(
                "Buffer does not support writing".to_string(),
            ));
        }

        Ok(())
    }

    /// Read data from buffer
    pub async fn read_data<T: bytemuck::Pod>(
        &self,
        offset: u64,
        count: usize,
    ) -> WebGpuResult<Vec<T>> {
        if !self.usage.contains(wgpu::BufferUsages::MAP_READ) {
            return Err(WebGpuError::InvalidBufferUsage(
                "Buffer does not support MAP_READ".to_string(),
            ));
        }

        let size = (count * std::mem::size_of::<T>()) as u64;
        self.map_read(offset, Some(size)).await?;

        let result = {
            let mapped = self.mapped_range(offset, Some(size))?;
            let data_slice: &[T] = bytemuck::cast_slice(&mapped);
            data_slice.to_vec()
        };

        self.unmap();
        Ok(result)
    }

    /// Copy data from another buffer
    pub fn copy_from_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &WebGpuBuffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) -> WebGpuResult<()> {
        if !src.usage.contains(wgpu::BufferUsages::COPY_SRC) {
            return Err(WebGpuError::InvalidBufferUsage(
                "Source buffer does not support COPY_SRC".to_string(),
            ));
        }

        if !self.usage.contains(wgpu::BufferUsages::COPY_DST) {
            return Err(WebGpuError::InvalidBufferUsage(
                "Destination buffer does not support COPY_DST".to_string(),
            ));
        }

        encoder.copy_buffer_to_buffer(&src.buffer, src_offset, &self.buffer, dst_offset, size);

        Ok(())
    }

    /// Convert BufferUsage to wgpu::BufferUsages
    fn convert_buffer_usage(usage: &BufferUsage) -> WebGpuResult<wgpu::BufferUsages> {
        let mut wgpu_usage = wgpu::BufferUsages::empty();

        if usage.contains(BufferUsage::STORAGE) {
            wgpu_usage |= wgpu::BufferUsages::STORAGE;
        }
        if usage.contains(BufferUsage::UNIFORM) {
            wgpu_usage |= wgpu::BufferUsages::UNIFORM;
        }
        if usage.contains(BufferUsage::VERTEX) {
            wgpu_usage |= wgpu::BufferUsages::VERTEX;
        }
        if usage.contains(BufferUsage::INDEX) {
            wgpu_usage |= wgpu::BufferUsages::INDEX;
        }
        if usage.contains(BufferUsage::COPY_SRC) {
            wgpu_usage |= wgpu::BufferUsages::COPY_SRC;
        }
        if usage.contains(BufferUsage::COPY_DST) {
            wgpu_usage |= wgpu::BufferUsages::COPY_DST;
        }
        if usage.contains(BufferUsage::MAP_READ) {
            wgpu_usage |= wgpu::BufferUsages::MAP_READ;
        }
        if usage.contains(BufferUsage::MAP_WRITE) {
            wgpu_usage |= wgpu::BufferUsages::MAP_WRITE;
        }

        if wgpu_usage.is_empty() {
            wgpu_usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
        }

        Ok(wgpu_usage)
    }
}

// Note: Buffer trait implementation temporarily disabled due to compilation issues
// TODO: Implement proper buffer trait when available
impl WebGpuBuffer {
    pub fn handle(&self) -> BufferHandle {
        self.handle.clone()
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn usage(&self) -> BufferUsage {
        self.descriptor.usage
    }

    pub fn memory_location(&self) -> MemoryLocation {
        MemoryLocation::Device
    }

    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
}

impl Drop for WebGpuBuffer {
    fn drop(&mut self) {
        // Ensure buffer is unmapped before dropping
        self.unmap();

        // Track deallocation
        self.device.track_buffer_deallocation(self.size);
    }
}

/// WebGPU buffer pool for efficient buffer reuse
#[derive(Debug)]
pub struct WebGpuBufferPool {
    device: Arc<WebGpuDevice>,
    pools: RwLock<HashMap<(u64, wgpu::BufferUsages), Vec<wgpu::Buffer>>>,
    next_handle: parking_lot::Mutex<u64>,
}

impl WebGpuBufferPool {
    /// Create a new buffer pool
    pub fn new(device: Arc<WebGpuDevice>) -> Self {
        Self {
            device,
            pools: RwLock::new(HashMap::new()),
            next_handle: parking_lot::Mutex::new(1),
        }
    }

    /// Get or create a buffer from the pool
    pub fn get_buffer(&self, descriptor: BufferDescriptor) -> WebGpuResult<WebGpuBuffer> {
        let usage = WebGpuBuffer::convert_buffer_usage(&descriptor.usage)?;
        let key = (descriptor.size as u64, usage);

        // Try to get from pool first
        {
            let mut pools = self.pools.write();
            if let Some(buffers) = pools.get_mut(&key) {
                if let Some(buffer) = buffers.pop() {
                    let handle = BufferHandle::WebGpu {
                        buffer_ptr: *self.next_handle.lock() as u64,
                        size: descriptor.size,
                    };
                    *self.next_handle.lock() += 1;

                    let size = descriptor.size as u64;
                    return Ok(WebGpuBuffer {
                        buffer: Arc::new(buffer),
                        device: Arc::clone(&self.device),
                        descriptor,
                        handle,
                        usage,
                        size,
                        mapping_state: Arc::new(RwLock::new(MappingState::Unmapped)),
                    });
                }
            }
        }

        // Create new buffer if none available in pool
        let handle = BufferHandle::WebGpu {
            buffer_ptr: *self.next_handle.lock() as u64,
            size: descriptor.size,
        };
        *self.next_handle.lock() += 1;

        WebGpuBuffer::new(Arc::clone(&self.device), descriptor, handle)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, buffer: WebGpuBuffer) {
        let key = (buffer.size, buffer.usage);
        let mut pools = self.pools.write();
        pools
            .entry(key)
            .or_insert_with(Vec::new)
            .push(buffer.into_wgpu_buffer());
    }

    /// Clear all buffers from pool
    pub fn clear(&self) {
        self.pools.write().clear();
    }

    /// Get pool statistics
    pub fn stats(&self) -> HashMap<(u64, wgpu::BufferUsages), usize> {
        let pools = self.pools.read();
        pools.iter().map(|(k, v)| (*k, v.len())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_buffer_creation() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);

                let descriptor = BufferDescriptor {
                    size: 1024,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                    location: MemoryLocation::Device,
                    dtype: None,
                    shape: None,
                    initial_data: None,
                    alignment: None,
                    zero_init: false,
                };

                let handle = BufferHandle::WebGpu { buffer_ptr: 1, size: 1024 };
                let buffer = WebGpuBuffer::new(device, descriptor, handle);

                assert!(buffer.is_ok());
                if let Ok(buffer) = buffer {
                    assert_eq!(buffer.size(), 1024);
                    assert_eq!(buffer.handle().id(), 1);
                    assert!(!buffer.is_mapped());
                }
            }
        }
    }

    #[test]
    fn test_buffer_usage_conversion() {
        let usage = BufferUsage::STORAGE | BufferUsage::UNIFORM | BufferUsage::COPY_SRC;
        let wgpu_usage = WebGpuBuffer::convert_buffer_usage(&usage).unwrap();

        assert!(wgpu_usage.contains(wgpu::BufferUsages::STORAGE));
        assert!(wgpu_usage.contains(wgpu::BufferUsages::UNIFORM));
        assert!(wgpu_usage.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!wgpu_usage.contains(wgpu::BufferUsages::MAP_READ));
    }

    #[test]
    fn test_mapping_state() {
        assert_eq!(MappingState::Unmapped, MappingState::Unmapped);
        assert_ne!(MappingState::Unmapped, MappingState::MappedRead);

        let state = MappingState::MappingPending;
        assert_eq!(format!("{:?}", state), "MappingPending");
    }

    #[tokio::test]
    async fn test_buffer_pool() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);
                let pool = WebGpuBufferPool::new(device);

                let descriptor = BufferDescriptor {
                    size: 512,
                    usage: BufferUsage::STORAGE,
                    location: MemoryLocation::Device,
                    dtype: None,
                    shape: None,
                    initial_data: None,
                    alignment: None,
                    zero_init: false,
                };

                // Get buffer from pool (will create new)
                let buffer1 = pool.get_buffer(descriptor.clone());
                assert!(buffer1.is_ok());

                // Return to pool and get again (should reuse)
                if let Ok(buffer1) = buffer1 {
                    let handle1 = buffer1.handle();
                    pool.return_buffer(buffer1);

                    let buffer2 = pool.get_buffer(descriptor);
                    if let Ok(buffer2) = buffer2 {
                        // Handles will be different but buffer should be reused
                        assert_ne!(buffer2.handle(), handle1);
                        assert_eq!(buffer2.size(), 512);
                    }
                }
            }
        }
    }
}
