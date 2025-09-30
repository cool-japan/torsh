//! Unified memory buffer implementation for CUDA

use crate::cuda::device::CudaDevice;
use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::memory::{MemoryAdvice, UnifiedAllocation};
use crate::{Buffer, BufferOps};
use std::sync::Arc;
use torsh_core::DType;

/// Debug information for unified buffers
#[derive(Debug, Clone)]
pub struct UnifiedBufferDebugInfo {
    pub ptr: *mut u8,
    pub length: usize,
    pub size_bytes: usize,
    pub dtype: DType,
    pub device_id: usize,
}

/// Unified memory buffer that can be accessed from both CPU and GPU
#[derive(Debug)]
pub struct UnifiedBuffer<T> {
    allocation: UnifiedAllocation,
    length: usize,
    dtype: DType,
    device: Arc<CudaDevice>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Clone + Send + Sync + 'static> UnifiedBuffer<T> {
    /// Create a new unified buffer
    pub fn new(device: Arc<CudaDevice>, length: usize, dtype: DType) -> CudaResult<Self> {
        let byte_size = length * std::mem::size_of::<T>();
        let allocation = device.memory_manager().allocate_unified(byte_size)?;

        Ok(Self {
            allocation,
            length,
            dtype,
            device,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the underlying allocation
    pub fn allocation(&self) -> &UnifiedAllocation {
        &self.allocation
    }

    /// Get mutable reference to the underlying allocation
    pub fn allocation_mut(&mut self) -> &mut UnifiedAllocation {
        &mut self.allocation
    }

    /// Get raw pointer to the data
    pub fn as_ptr(&self) -> *const T {
        self.allocation.as_ptr()
    }

    /// Get mutable raw pointer to the data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.allocation.as_ptr() as *mut T
    }

    /// Get a slice view of the data (safe when accessed from CPU)
    pub unsafe fn as_slice(&self) -> &[T] {
        std::slice::from_raw_parts(self.as_ptr(), self.length)
    }

    /// Get a mutable slice view of the data (safe when accessed from CPU)
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.length)
    }

    /// Prefetch data to GPU
    pub fn prefetch_to_device(&self, device_id: Option<usize>) -> CudaResult<()> {
        let byte_size = self.length * std::mem::size_of::<T>();
        self.device
            .memory_manager()
            .prefetch_to_device(self.allocation.ptr(), byte_size, device_id)
    }

    /// Prefetch data to CPU
    pub fn prefetch_to_host(&self) -> CudaResult<()> {
        let byte_size = self.length * std::mem::size_of::<T>();
        self.device
            .memory_manager()
            .prefetch_to_host(self.allocation.ptr(), byte_size)
    }

    /// Set memory advice for performance optimization
    pub fn set_memory_advice(
        &self,
        advice: MemoryAdvice,
        device_id: Option<usize>,
    ) -> CudaResult<()> {
        let byte_size = self.length * std::mem::size_of::<T>();
        self.device.memory_manager().set_memory_advice(
            self.allocation.ptr(),
            byte_size,
            advice,
            device_id,
        )
    }

    /// Set data as read-mostly for optimization
    pub fn set_read_mostly(&self) -> CudaResult<()> {
        self.set_memory_advice(MemoryAdvice::SetReadMostly, None)
    }

    /// Set preferred location for the data
    pub fn set_preferred_location(&self, device_id: usize) -> CudaResult<()> {
        self.set_memory_advice(MemoryAdvice::SetPreferredLocation, Some(device_id))
    }

    /// Indicate which device will access this data
    pub fn set_accessed_by(&self, device_id: usize) -> CudaResult<()> {
        self.set_memory_advice(MemoryAdvice::SetAccessedBy, Some(device_id))
    }

    /// Get memory statistics for debugging
    pub fn debug_info(&self) -> UnifiedBufferDebugInfo {
        UnifiedBufferDebugInfo {
            ptr: self.allocation.ptr(),
            length: self.length,
            size_bytes: self.length * std::mem::size_of::<T>(),
            dtype: self.dtype,
            device_id: self.device.id(),
        }
    }
}

impl<T: Clone + Send + Sync + 'static> Buffer<T> for UnifiedBuffer<T> {
    fn len(&self) -> usize {
        self.length
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &dyn crate::Device {
        self.device.as_ref()
    }

    fn copy_from_host(&mut self, data: &[T]) -> Result<(), crate::BackendError> {
        if data.len() != self.length {
            return Err(crate::BackendError::InvalidBuffer {
                message: format!(
                    "Data length {} does not match buffer length {}",
                    data.len(),
                    self.length
                ),
            });
        }

        self.allocation
            .copy_from_host(data)
            .map_err(|e| crate::BackendError::Runtime {
                message: format!("Failed to copy from host: {}", e),
            })
    }

    fn copy_to_host(&self, data: &mut [T]) -> Result<(), crate::BackendError> {
        if data.len() != self.length {
            return Err(crate::BackendError::InvalidBuffer {
                message: format!(
                    "Data length {} does not match buffer length {}",
                    data.len(),
                    self.length
                ),
            });
        }

        self.allocation
            .copy_to_host(data)
            .map_err(|e| crate::BackendError::Runtime {
                message: format!("Failed to copy to host: {}", e),
            })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl<T: Clone + Send + Sync + 'static> BufferOps<T> for UnifiedBuffer<T> {
    fn fill(&mut self, value: T) -> Result<(), crate::BackendError> {
        unsafe {
            let slice = self.as_mut_slice();
            slice.fill(value);
        }
        Ok(())
    }

    fn copy_from_buffer(&mut self, src: &dyn Buffer<T>) -> Result<(), crate::BackendError> {
        if src.len() != self.length {
            return Err(crate::BackendError::InvalidBuffer {
                message: format!(
                    "Source buffer length {} does not match target length {}",
                    src.len(),
                    self.length
                ),
            });
        }

        // Try to copy directly if source is also a unified buffer
        if let Some(src_unified) = src.as_any().downcast_ref::<UnifiedBuffer<T>>() {
            unsafe {
                std::ptr::copy_nonoverlapping(src_unified.as_ptr(), self.as_mut_ptr(), self.length);
            }
            return Ok(());
        }

        // Otherwise, copy via host
        let mut temp_data = Vec::<T>::with_capacity(self.length);
        unsafe {
            temp_data.set_len(self.length);
        }
        src.copy_to_host(&mut temp_data)?;
        self.copy_from_host(&temp_data)
    }

    fn set_zero(&mut self) -> Result<(), crate::BackendError> {
        unsafe {
            std::ptr::write_bytes(
                self.as_mut_ptr() as *mut u8,
                0,
                self.length * std::mem::size_of::<T>(),
            );
        }
        Ok(())
    }
}

// UnifiedBuffer doesn't need an explicit Drop implementation
// The UnifiedAllocation field will be automatically dropped and
// its Drop implementation will free the CUDA unified memory

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::device::CudaDevice;
    use torsh_core::DType;

    #[test]
    fn test_unified_buffer_creation() {
        if crate::is_available() {
            let device = Arc::new(CudaDevice::new(0).unwrap());

            // Check if device supports unified memory
            if device
                .supports_feature(crate::cuda::device::CudaFeature::ManagedMemory)
                .unwrap_or(false)
            {
                let buffer = UnifiedBuffer::<f32>::new(device, 1024, DType::F32);
                assert!(buffer.is_ok());

                let buffer = buffer.unwrap();
                assert_eq!(buffer.len(), 1024);
                assert_eq!(buffer.dtype(), DType::F32);
            }
        }
    }

    #[test]
    fn test_unified_buffer_operations() {
        if crate::is_available() {
            let device = Arc::new(CudaDevice::new(0).unwrap());

            if device
                .supports_feature(crate::cuda::device::CudaFeature::ManagedMemory)
                .unwrap_or(false)
            {
                let mut buffer = UnifiedBuffer::<f32>::new(device, 4, DType::F32).unwrap();

                // Test copy operations
                let test_data = vec![1.0, 2.0, 3.0, 4.0];
                buffer.copy_from_host(&test_data).unwrap();

                let mut result_data = vec![0.0; 4];
                buffer.copy_to_host(&mut result_data).unwrap();
                assert_eq!(result_data, test_data);

                // Test prefetching
                buffer.prefetch_to_device(None).unwrap();
                buffer.prefetch_to_host().unwrap();

                // Test memory advice
                buffer.set_read_mostly().unwrap();
                buffer.set_preferred_location(0).unwrap();
                buffer.set_accessed_by(0).unwrap();
            }
        }
    }

    #[test]
    fn test_unified_buffer_slice_access() {
        if crate::is_available() {
            let device = Arc::new(CudaDevice::new(0).unwrap());

            if device
                .supports_feature(crate::cuda::device::CudaFeature::ManagedMemory)
                .unwrap_or(false)
            {
                let mut buffer = UnifiedBuffer::<i32>::new(device, 8, DType::I32).unwrap();

                // Fill buffer with test data
                let test_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
                buffer.copy_from_host(&test_data).unwrap();

                // Test slice access (after prefetching to host)
                buffer.prefetch_to_host().unwrap();
                unsafe {
                    let slice = buffer.as_slice();
                    assert_eq!(slice.len(), 8);
                    assert_eq!(slice[0], 1);
                    assert_eq!(slice[7], 8);
                }
            }
        }
    }

    #[test]
    fn test_unified_buffer_fill_and_zero() {
        if crate::is_available() {
            let device = Arc::new(CudaDevice::new(0).unwrap());

            if device
                .supports_feature(crate::cuda::device::CudaFeature::ManagedMemory)
                .unwrap_or(false)
            {
                let mut buffer = UnifiedBuffer::<f32>::new(device, 10, DType::F32).unwrap();

                // Test fill
                buffer.fill(3.14).unwrap();

                let mut result = vec![0.0; 10];
                buffer.copy_to_host(&mut result).unwrap();
                for &val in &result {
                    assert_eq!(val, 3.14);
                }

                // Test zero
                buffer.set_zero().unwrap();
                buffer.copy_to_host(&mut result).unwrap();
                for &val in &result {
                    assert_eq!(val, 0.0);
                }
            }
        }
    }
}
