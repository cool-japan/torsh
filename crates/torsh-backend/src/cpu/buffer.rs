//! CPU Buffer Implementation

use crate::buffer::{generate_buffer_id, BufferHandle};
use crate::error::{BackendError, BackendResult};
use crate::{Buffer, BufferDescriptor, BufferUsage, Device};

#[cfg(feature = "std")]
use std::sync::{Arc, RwLock};

#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
use spin::RwLock;

/// CPU buffer implementation using system memory
#[derive(Debug, Clone)]
pub struct CpuBuffer {
    data: Arc<RwLock<Vec<u8>>>,
    size: usize,
    usage: BufferUsage,
}

impl CpuBuffer {
    /// Create a new CPU buffer
    pub fn new(size: usize, usage: BufferUsage) -> BackendResult<Self> {
        // Check for reasonable size limits to avoid capacity overflow
        if size > isize::MAX as usize {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "Buffer size {} is too large (exceeds maximum allowed size)",
                size
            )));
        }

        let data = match size {
            0 => Vec::new(), // Handle zero-size case
            size => {
                // Try to allocate the vector, catching any allocation errors
                match std::panic::catch_unwind(|| vec![0u8; size]) {
                    Ok(vec) => vec,
                    Err(_) => {
                        return Err(torsh_core::error::TorshError::BackendError(format!(
                            "Failed to allocate {} bytes for buffer",
                            size
                        )));
                    }
                }
            }
        };

        Ok(Self {
            data: Arc::new(RwLock::new(data)),
            size,
            usage,
        })
    }

    /// Create a CPU buffer and return an abstract Buffer
    pub fn new_buffer(device: Device, descriptor: &BufferDescriptor) -> BackendResult<Buffer> {
        let cpu_buffer = Self::new(descriptor.size, descriptor.usage)?;

        // Store the CpuBuffer in the BufferHandle using Generic variant
        // This avoids the dangling pointer issue
        let handle = BufferHandle::Generic {
            handle: Box::new(cpu_buffer),
            size: descriptor.size,
        };

        let buffer = Buffer::new(
            generate_buffer_id(),
            device,
            descriptor.size,
            descriptor.usage,
            descriptor.clone(),
            handle,
        );

        Ok(buffer)
    }

    /// Create a CPU buffer from existing data
    pub fn from_data(data: Vec<u8>, usage: BufferUsage) -> Self {
        let size = data.len();
        Self {
            data: Arc::new(RwLock::new(data)),
            size,
            usage,
        }
    }

    /// Get the buffer size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the buffer usage flags
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Read data from the buffer
    pub fn read_bytes(&self, dst: &mut [u8], offset: usize) -> BackendResult<()> {
        let data = self.data.read().map_err(|_| {
            BackendError::AllocationError("Failed to acquire read lock".to_string())
        })?;

        if offset + dst.len() > data.len() {
            return Err(BackendError::AllocationError(format!(
                "Read bounds check failed: offset {} + size {} > buffer size {}",
                offset,
                dst.len(),
                data.len()
            )));
        }

        dst.copy_from_slice(&data[offset..offset + dst.len()]);
        Ok(())
    }

    /// Write data to the buffer
    pub fn write_bytes(&self, src: &[u8], offset: usize) -> BackendResult<()> {
        let mut data = self.data.write().map_err(|_| {
            BackendError::AllocationError("Failed to acquire write lock".to_string())
        })?;

        if offset + src.len() > data.len() {
            return Err(BackendError::AllocationError(format!(
                "Write bounds check failed: offset {} + size {} > buffer size {}",
                offset,
                src.len(),
                data.len()
            )));
        }

        data[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    /// Copy data from another CPU buffer
    pub fn copy_to(
        &self,
        dst: &CpuBuffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> BackendResult<()> {
        let src_data = self.data.read().map_err(|_| {
            BackendError::AllocationError("Failed to acquire source read lock".to_string())
        })?;

        let mut dst_data = dst.data.write().map_err(|_| {
            BackendError::AllocationError("Failed to acquire destination write lock".to_string())
        })?;

        if src_offset + size > src_data.len() {
            return Err(BackendError::AllocationError(format!(
                "Source bounds check failed: offset {} + size {} > buffer size {}",
                src_offset,
                size,
                src_data.len()
            )));
        }

        if dst_offset + size > dst_data.len() {
            return Err(BackendError::AllocationError(format!(
                "Destination bounds check failed: offset {} + size {} > buffer size {}",
                dst_offset,
                size,
                dst_data.len()
            )));
        }

        dst_data[dst_offset..dst_offset + size]
            .copy_from_slice(&src_data[src_offset..src_offset + size]);

        Ok(())
    }

    /// Get a reference to the underlying data (for zero-copy operations)
    pub fn data(&self) -> Arc<RwLock<Vec<u8>>> {
        self.data.clone()
    }

    /// Map the buffer for reading (returns a read guard)
    pub fn map_read(&self) -> BackendResult<std::sync::RwLockReadGuard<'_, Vec<u8>>> {
        self.data
            .read()
            .map_err(|_| BackendError::AllocationError("Failed to acquire read lock".to_string()))
    }

    /// Map the buffer for writing (returns a write guard)
    pub fn map_write(&self) -> BackendResult<std::sync::RwLockWriteGuard<'_, Vec<u8>>> {
        self.data
            .write()
            .map_err(|_| BackendError::AllocationError("Failed to acquire write lock".to_string()))
    }
}

// Extension trait for Buffer to work with CPU buffers
pub trait BufferCpuExt {
    fn is_cpu(&self) -> bool;
    fn as_cpu_ptr(&self) -> Option<*mut u8>;
    fn as_cpu_buffer(&self) -> Option<&CpuBuffer>;
}

impl BufferCpuExt for Buffer {
    fn is_cpu(&self) -> bool {
        match &self.handle {
            BufferHandle::Cpu { .. } => true,
            BufferHandle::Generic { handle, .. } => {
                // Check if the generic handle contains a CpuBuffer
                handle.downcast_ref::<CpuBuffer>().is_some()
            }
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { .. } => false,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            BufferHandle::Metal { .. } => false,
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { .. } => false,
        }
    }

    fn as_cpu_ptr(&self) -> Option<*mut u8> {
        match &self.handle {
            BufferHandle::Cpu { ptr, .. } => Some(*ptr),
            BufferHandle::Generic { handle, .. } => {
                // Safely extract pointer from CpuBuffer
                if let Some(cpu_buffer) = handle.downcast_ref::<CpuBuffer>() {
                    // Get pointer to the underlying data safely
                    let data_guard = cpu_buffer.data.read().ok()?;
                    Some(data_guard.as_ptr() as *mut u8)
                } else {
                    None
                }
            }
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { .. } => None,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            BufferHandle::Metal { .. } => None,
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { .. } => None,
        }
    }

    fn as_cpu_buffer(&self) -> Option<&CpuBuffer> {
        match &self.handle {
            BufferHandle::Generic { handle, .. } => handle.downcast_ref::<CpuBuffer>(),
            BufferHandle::Cpu { .. } => None, // Legacy CPU buffers don't have CpuBuffer reference
            #[cfg(feature = "cuda")]
            BufferHandle::Cuda { .. } => None,
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            BufferHandle::Metal { .. } => None,
            #[cfg(feature = "webgpu")]
            BufferHandle::WebGpu { .. } => None,
        }
    }
}

// Unsafe operations for performance-critical code
impl CpuBuffer {
    /// Get a raw pointer to the buffer data (unsafe)
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The returned pointer is not used after the buffer is dropped
    /// - No mutable references to the buffer exist when using this pointer
    /// - The buffer is not resized while using this pointer
    pub unsafe fn as_ptr(&self) -> *const u8 {
        let data = self.data.read().expect("lock should not be poisoned");
        data.as_ptr()
    }

    /// Get a raw mutable pointer to the buffer data (unsafe)
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The returned pointer is not used after the buffer is dropped
    /// - No other references to the buffer exist when using this pointer
    /// - The buffer is not resized while using this pointer
    pub unsafe fn as_mut_ptr(&self) -> *mut u8 {
        let mut data = self.data.write().expect("lock should not be poisoned");
        data.as_mut_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_buffer_creation() {
        let buffer = CpuBuffer::new(1024, BufferUsage::STORAGE).unwrap();
        assert_eq!(buffer.size(), 1024);
        assert_eq!(buffer.usage(), BufferUsage::STORAGE);
    }

    #[test]
    fn test_cpu_buffer_read_write() {
        let buffer = CpuBuffer::new(256, BufferUsage::STORAGE).unwrap();

        let write_data = vec![1, 2, 3, 4, 5];
        buffer.write_bytes(&write_data, 10).unwrap();

        let mut read_data = vec![0; 5];
        buffer.read_bytes(&mut read_data, 10).unwrap();

        assert_eq!(read_data, write_data);
    }

    #[test]
    fn test_cpu_buffer_copy() {
        let src_buffer = CpuBuffer::new(256, BufferUsage::STORAGE).unwrap();
        let dst_buffer = CpuBuffer::new(256, BufferUsage::STORAGE).unwrap();

        let test_data = vec![10, 20, 30, 40, 50];
        src_buffer.write_bytes(&test_data, 0).unwrap();

        src_buffer
            .copy_to(&dst_buffer, 0, 0, test_data.len())
            .unwrap();

        let mut read_data = vec![0; test_data.len()];
        dst_buffer.read_bytes(&mut read_data, 0).unwrap();

        assert_eq!(read_data, test_data);
    }

    #[test]
    fn test_buffer_bounds_checking() {
        let buffer = CpuBuffer::new(10, BufferUsage::STORAGE).unwrap();

        // Test read bounds
        let mut read_data = vec![0; 5];
        assert!(buffer.read_bytes(&mut read_data, 10).is_err()); // Out of bounds

        // Test write bounds
        let write_data = vec![1, 2, 3, 4, 5];
        assert!(buffer.write_bytes(&write_data, 10).is_err()); // Out of bounds
    }
}
