//! CUDA buffer implementation

use crate::cuda::cuda_sys_compat as cuda_sys;
use crate::cuda::device::CudaDevice;
use crate::cuda::error::{CudaError, CudaResult};
use crate::cuda::memory::{CudaAllocation, SendSyncPtr};
use crate::cuda::stream::CudaStream;
#[allow(unused_imports)]
use crate::{Buffer, BufferError};
use std::ffi::c_void;
use std::sync::Arc;
use torsh_core::DType;

/// Device pointer type alias using our thread-safe wrapper
pub type BufferDevicePtr<T> = SendSyncPtr<T>;

/// CUDA buffer implementation
#[derive(Debug, Clone)]
pub struct CudaBuffer<T> {
    allocation: CudaAllocation,
    length: usize,
    dtype: DType,
    device: Arc<CudaDevice>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Clone + Send + Sync + 'static> CudaBuffer<T> {
    /// Create new CUDA buffer
    pub fn new(device: Arc<CudaDevice>, length: usize, dtype: DType) -> CudaResult<Self> {
        let size = length * std::mem::size_of::<T>();
        let allocation = device.memory_manager().allocate(size)?;

        Ok(Self {
            allocation,
            length,
            dtype,
            device,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create buffer from existing allocation
    pub fn from_allocation(
        device: Arc<CudaDevice>,
        allocation: CudaAllocation,
        length: usize,
        dtype: DType,
    ) -> Self {
        Self {
            allocation,
            length,
            dtype,
            device,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get device pointer as typed SendSyncPtr
    pub fn device_ptr(&self) -> BufferDevicePtr<T> {
        // Cast the u8 pointer to T pointer
        SendSyncPtr::new(self.allocation.as_ptr() as *mut T)
    }

    /// Get raw device pointer
    pub fn raw_ptr(&self) -> *mut u8 {
        self.allocation.as_ptr()
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, data: &[T]) -> CudaResult<()> {
        if data.len() != self.length {
            return Err(CudaError::Memory {
                message: format!(
                    "Data length mismatch: expected {}, got {}",
                    self.length,
                    data.len()
                ),
            });
        }

        unsafe {
            let result = cuda_sys::cudaMemcpy(
                self.allocation.as_ptr() as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len() * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyHostToDevice,
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Memory {
                    message: format!("Host-to-device copy failed: {:?}", result),
                });
            }
        }

        Ok(())
    }

    /// Copy data from host to device asynchronously
    pub fn copy_from_host_async(&mut self, data: &[T], stream: &CudaStream) -> CudaResult<()> {
        if data.len() != self.length {
            return Err(CudaError::Memory {
                message: format!(
                    "Data length mismatch: expected {}, got {}",
                    self.length,
                    data.len()
                ),
            });
        }

        unsafe {
            let result = cuda_sys::cudaMemcpyAsync(
                self.allocation.as_ptr() as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len() * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyHostToDevice,
                stream.stream(),
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Memory {
                    message: format!("Async host-to-device copy failed: {:?}", result),
                });
            }
        }

        Ok(())
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [T]) -> CudaResult<()> {
        if data.len() != self.length {
            return Err(CudaError::Memory {
                message: format!(
                    "Data length mismatch: expected {}, got {}",
                    self.length,
                    data.len()
                ),
            });
        }

        unsafe {
            let result = cuda_sys::cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.allocation.as_ptr() as *const c_void,
                data.len() * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyDeviceToHost,
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Memory {
                    message: format!("Device-to-host copy failed: {:?}", result),
                });
            }
        }

        Ok(())
    }

    /// Copy data from device to host asynchronously
    pub fn copy_to_host_async(&self, data: &mut [T], stream: &CudaStream) -> CudaResult<()> {
        if data.len() != self.length {
            return Err(CudaError::Memory {
                message: format!(
                    "Data length mismatch: expected {}, got {}",
                    self.length,
                    data.len()
                ),
            });
        }

        unsafe {
            let result = cuda_sys::cudaMemcpyAsync(
                data.as_mut_ptr() as *mut c_void,
                self.allocation.as_ptr() as *const c_void,
                data.len() * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyDeviceToHost,
                stream.stream(),
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Memory {
                    message: format!("Async device-to-host copy failed: {:?}", result),
                });
            }
        }

        Ok(())
    }

    /// Copy from another CUDA buffer
    pub fn copy_from_buffer(&mut self, src: &CudaBuffer<T>) -> CudaResult<()> {
        if self.length != src.length {
            return Err(CudaError::Memory {
                message: format!(
                    "Buffer length mismatch: expected {}, got {}",
                    self.length, src.length
                ),
            });
        }

        unsafe {
            let result = cuda_sys::cudaMemcpy(
                self.allocation.as_ptr() as *mut c_void,
                src.allocation.as_ptr() as *const c_void,
                self.length * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyDeviceToDevice,
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Memory {
                    message: format!("Device-to-device copy failed: {:?}", result),
                });
            }
        }

        Ok(())
    }

    /// Copy from another CUDA buffer (alias for copy_from_buffer)
    pub fn copy_from(&mut self, src: &CudaBuffer<T>) -> CudaResult<()> {
        self.copy_from_buffer(src)
    }

    /// Copy from another CUDA buffer asynchronously
    pub fn copy_from_buffer_async(
        &mut self,
        src: &CudaBuffer<T>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if self.length != src.length {
            return Err(CudaError::Memory {
                message: format!(
                    "Buffer length mismatch: expected {}, got {}",
                    self.length, src.length
                ),
            });
        }

        unsafe {
            let result = cuda_sys::cudaMemcpyAsync(
                self.allocation.as_ptr() as *mut c_void,
                src.allocation.as_ptr() as *const c_void,
                self.length * std::mem::size_of::<T>(),
                cuda_sys::cudaMemcpyKind_cudaMemcpyDeviceToDevice,
                stream.stream(),
            );

            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Memory {
                    message: format!("Async device-to-device copy failed: {:?}", result),
                });
            }
        }

        Ok(())
    }

    /// Fill buffer with value
    pub fn fill(&mut self, value: T) -> CudaResult<()>
    where
        T: Copy,
    {
        // For now, use host-side fill and copy
        let data = vec![value; self.length];
        self.copy_from_host(&data)
    }

    /// Get buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.allocation.size()
    }

    /// Get element count
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

// TODO: Implement Buffer trait when it's defined in the codebase
// Currently Buffer is a struct, not a trait, so this implementation is commented out
/*
impl<T: Clone + Send + Sync + 'static> Buffer<T> for CudaBuffer<T> {
    fn len(&self) -> usize {
        self.length
    }

    fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn capacity(&self) -> usize {
        self.allocation.size() / std::mem::size_of::<T>()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device_type(&self) -> crate::DeviceType {
        self.device.device_type()
    }

    fn clone_empty(&self, length: usize) -> Result<Box<dyn Buffer<T>>, BufferError> {
        let buffer =
            CudaBuffer::new(Arc::clone(&self.device), length, self.dtype).map_err(|e| {
                BufferError::AllocationFailed {
                    message: e.to_string(),
                }
            })?;

        Ok(Box::new(buffer))
    }

    fn copy_from_slice(&mut self, data: &[T]) -> Result<(), BufferError> {
        self.copy_from_host(data)
            .map_err(|e| BufferError::CopyFailed {
                message: e.to_string(),
            })
    }

    fn copy_to_slice(&self, data: &mut [T]) -> Result<(), BufferError> {
        self.copy_to_host(data)
            .map_err(|e| BufferError::CopyFailed {
                message: e.to_string(),
            })
    }

    fn copy_from_buffer(&mut self, src: &dyn Buffer<T>) -> Result<(), BufferError> {
        if let Some(cuda_src) = src.as_any().downcast_ref::<CudaBuffer<T>>() {
            self.copy_from_buffer(cuda_src)
                .map_err(|e| BufferError::CopyFailed {
                    message: e.to_string(),
                })
        } else {
            // Cross-device copy via host memory
            let mut temp = vec![T::default(); src.len()];
            src.copy_to_slice(&mut temp)?;
            self.copy_from_slice(&temp)
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
*/

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        // Memory will be returned to pool automatically when allocation is dropped
        if let Err(e) = self
            .device
            .memory_manager()
            .deallocate(self.allocation.clone())
        {
            tracing::warn!("Failed to deallocate CUDA buffer: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DType;

    #[test]
    fn test_cuda_buffer_creation() {
        if crate::is_available() {
            let device = Arc::new(CudaDevice::new(0).unwrap());
            let buffer = CudaBuffer::<f32>::new(device, 1024, DType::F32);

            assert!(buffer.is_ok());
            let buffer = buffer.unwrap();
            assert_eq!(buffer.len(), 1024);
            assert_eq!(buffer.dtype(), DType::F32);
        }
    }

    #[test]
    fn test_host_device_copy() {
        if crate::is_available() {
            let device = Arc::new(CudaDevice::new(0).unwrap());
            let mut buffer = CudaBuffer::<f32>::new(device, 4, DType::F32).unwrap();

            let host_data = vec![1.0, 2.0, 3.0, 4.0];
            buffer.copy_from_host(&host_data).unwrap();

            let mut result = vec![0.0; 4];
            buffer.copy_to_host(&mut result).unwrap();

            assert_eq!(host_data, result);
        }
    }

    #[test]
    fn test_buffer_copy() {
        if crate::is_available() {
            let device = Arc::new(CudaDevice::new(0).unwrap());
            let mut src = CudaBuffer::<f32>::new(Arc::clone(&device), 4, DType::F32).unwrap();
            let mut dst = CudaBuffer::<f32>::new(Arc::clone(&device), 4, DType::F32).unwrap();

            let host_data = vec![1.0, 2.0, 3.0, 4.0];
            src.copy_from_host(&host_data).unwrap();
            dst.copy_from_buffer(&src).unwrap();

            let mut result = vec![0.0; 4];
            dst.copy_to_host(&mut result).unwrap();

            assert_eq!(host_data, result);
        }
    }
}
