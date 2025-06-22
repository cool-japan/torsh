//! CUDA backend implementation

use std::sync::Arc;
use async_trait::async_trait;
use torsh_core::{DType, TensorError};
use torsh_backends::{Backend, BackendConfig, BackendError, DeviceType};
use crate::device::CudaDevice;
use crate::buffer::CudaBuffer;
use crate::memory::CudaMemoryManager;
use crate::stream::CudaStream;
use crate::error::{CudaError, CudaResult};
use crate::kernels::{LaunchConfig, KernelRegistry};

/// CUDA backend implementation
#[derive(Debug)]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    memory_manager: Arc<CudaMemoryManager>,
    default_stream: Arc<CudaStream>,
    kernels: Arc<KernelRegistry>,
    config: CudaBackendConfig,
}

/// CUDA backend configuration
#[derive(Debug, Clone)]
pub struct CudaBackendConfig {
    pub device_id: usize,
    pub allow_tf32: bool,
    pub enable_profiling: bool,
    pub memory_pool_size: Option<usize>,
    pub stream_pool_size: usize,
}

impl Default for CudaBackendConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            allow_tf32: true,
            enable_profiling: false,
            memory_pool_size: None,
            stream_pool_size: 4,
        }
    }
}

impl BackendConfig for CudaBackendConfig {}

impl CudaBackend {
    /// Create new CUDA backend
    pub fn new(config: CudaBackendConfig) -> CudaResult<Self> {
        // Initialize CUDA
        crate::init()?;
        
        // Create device
        let device = Arc::new(CudaDevice::new(config.device_id)?);
        
        // Set device as current
        crate::set_device(config.device_id)?;
        
        // Create memory manager
        let memory_manager = Arc::new(CudaMemoryManager::new(config.device_id)?);
        
        // Create default stream
        let default_stream = Arc::new(CudaStream::default()?);
        
        // Load kernels (would load from embedded PTX in real implementation)
        let kernels = Arc::new(Self::load_kernels()?);
        
        Ok(Self {
            device,
            memory_manager,
            default_stream,
            kernels,
            config,
        })
    }
    
    /// Load CUDA kernels
    fn load_kernels() -> CudaResult<KernelRegistry> {
        // In a real implementation, this would load compiled PTX
        // For now, we'll create a placeholder registry
        let ptx = include_str!("../kernels/compiled.ptx");
        KernelRegistry::load_from_ptx(ptx)
            .or_else(|_| {
                // Fallback: create empty registry for testing
                tracing::warn!("Failed to load CUDA kernels, using fallback");
                Ok(KernelRegistry::load_from_ptx("")?)
            })
    }
    
    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
    
    /// Get memory manager
    pub fn memory_manager(&self) -> &Arc<CudaMemoryManager> {
        &self.memory_manager
    }
    
    /// Get default stream
    pub fn default_stream(&self) -> &Arc<CudaStream> {
        &self.default_stream
    }
    
    /// Create buffer
    pub fn create_buffer<T: Clone + Send + Sync + 'static>(
        &self,
        length: usize,
        dtype: DType,
    ) -> CudaResult<CudaBuffer<T>> {
        CudaBuffer::new(Arc::clone(&self.device), length, dtype)
    }
    
    /// Synchronize device
    pub fn synchronize(&self) -> CudaResult<()> {
        self.device.synchronize()
            .map_err(|e| CudaError::Backend(e.into()))?;
        Ok(())
    }
    
    /// Execute elementwise addition
    pub fn elementwise_add_f32(
        &self,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(CudaError::InvalidDevice {
                device_id: a.len(), // Using as error code
            });
        }
        
        let stream = stream.unwrap_or(&self.default_stream);
        let size = a.len();
        
        unsafe {
            crate::kernels::tensor_ops::launch_elementwise_add_f32(
                a.device_ptr().as_raw_mut(),
                b.device_ptr().as_raw_mut(),
                output.device_ptr().as_raw_mut(),
                size,
                stream.raw().as_inner(),
            );
        }
        
        Ok(())
    }
    
    /// Execute elementwise multiplication
    pub fn elementwise_mul_f32(
        &self,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        if a.len() != b.len() || a.len() != output.len() {
            return Err(CudaError::InvalidDevice {
                device_id: a.len(), // Using as error code
            });
        }
        
        let stream = stream.unwrap_or(&self.default_stream);
        let size = a.len();
        
        unsafe {
            crate::kernels::tensor_ops::launch_elementwise_mul_f32(
                a.device_ptr().as_raw_mut(),
                b.device_ptr().as_raw_mut(),
                output.device_ptr().as_raw_mut(),
                size,
                stream.raw().as_inner(),
            );
        }
        
        Ok(())
    }
    
    /// Execute matrix multiplication using cuBLAS
    pub fn matmul_f32(
        &self,
        a: &CudaBuffer<f32>,
        b: &CudaBuffer<f32>,
        output: &mut CudaBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        let stream = stream.unwrap_or(&self.default_stream);
        
        // Use cuBLAS for matrix multiplication
        let cublas_handle = self.get_cublas_handle()?;
        
        let alpha = 1.0f32;
        let beta = 0.0f32;
        
        unsafe {
            cust::cublas::sgemm(
                cublas_handle,
                cust::cublas::Operation::N,
                cust::cublas::Operation::N,
                n as i32, m as i32, k as i32,
                &alpha,
                b.device_ptr().as_raw(), n as i32,
                a.device_ptr().as_raw(), k as i32,
                &beta,
                output.device_ptr().as_raw_mut(), n as i32,
            )?;
        }
        
        Ok(())
    }
    
    /// Get cuBLAS handle
    fn get_cublas_handle(&self) -> CudaResult<cust::cublas::CublasHandle> {
        // In a real implementation, this would be cached per device/stream
        Ok(cust::cublas::CublasHandle::new()?)
    }
    
    /// Execute convolution using cuDNN
    pub fn conv2d_f32(
        &self,
        input: &CudaBuffer<f32>,
        weight: &CudaBuffer<f32>,
        bias: Option<&CudaBuffer<f32>>,
        output: &mut CudaBuffer<f32>,
        config: &Conv2dConfig,
        stream: Option<&CudaStream>,
    ) -> CudaResult<()> {
        let stream = stream.unwrap_or(&self.default_stream);
        
        // Use custom kernel for now (would use cuDNN in production)
        unsafe {
            crate::kernels::neural_ops::launch_conv2d_f32(
                input.device_ptr().as_raw_mut(),
                weight.device_ptr().as_raw_mut(),
                bias.map(|b| b.device_ptr().as_raw_mut()).unwrap_or(std::ptr::null_mut()),
                output.device_ptr().as_raw_mut(),
                config.batch_size as i32,
                config.in_channels as i32,
                config.out_channels as i32,
                config.input_height as i32,
                config.input_width as i32,
                config.kernel_height as i32,
                config.kernel_width as i32,
                config.pad_h as i32,
                config.pad_w as i32,
                config.stride_h as i32,
                config.stride_w as i32,
                config.dilation_h as i32,
                config.dilation_w as i32,
                stream.raw().as_inner(),
            );
        }
        
        Ok(())
    }
}

#[async_trait]
impl Backend for CudaBackend {
    type Device = CudaDevice;
    type Config = CudaBackendConfig;
    
    async fn initialize(config: Self::Config) -> Result<Self, BackendError> {
        CudaBackend::new(config).map_err(|e| BackendError::InitializationFailed {
            message: e.to_string(),
        })
    }
    
    fn name(&self) -> &str {
        "cuda"
    }
    
    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda(self.config.device_id)
    }
    
    fn is_available(&self) -> bool {
        crate::is_available()
    }
    
    fn synchronize(&self) -> Result<(), BackendError> {
        self.synchronize().map_err(|e| BackendError::Runtime {
            message: e.to_string(),
        })
    }
    
    fn create_buffer<T: Clone + Send + Sync + 'static>(
        &self,
        length: usize,
        dtype: DType,
    ) -> Result<Box<dyn torsh_backends::Buffer<T>>, BackendError> {
        let buffer = self.create_buffer::<T>(length, dtype)
            .map_err(|e| BackendError::AllocationFailed {
                message: e.to_string(),
            })?;
        Ok(Box::new(buffer))
    }
    
    fn add_tensors<T: Clone + Send + Sync + 'static>(
        &self,
        a: &dyn torsh_backends::Buffer<T>,
        b: &dyn torsh_backends::Buffer<T>,
        output: &mut dyn torsh_backends::Buffer<T>,
    ) -> Result<(), BackendError> {
        // Downcast to CUDA buffers
        let a_cuda = a.as_any().downcast_ref::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input A".to_string(),
            })?;
        let b_cuda = b.as_any().downcast_ref::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input B".to_string(),
            })?;
        let output_cuda = output.as_any_mut().downcast_mut::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for output".to_string(),
            })?;
        
        // For now, only support f32
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let a_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(a_cuda) };
            let b_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(b_cuda) };
            let output_f32 = unsafe { std::mem::transmute::<&mut CudaBuffer<T>, &mut CudaBuffer<f32>>(output_cuda) };
            
            self.elementwise_add_f32(a_f32, b_f32, output_f32, None)
                .map_err(|e| BackendError::Runtime {
                    message: e.to_string(),
                })?;
        } else {
            return Err(BackendError::UnsupportedOperation {
                operation: "add_tensors".to_string(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        }
        
        Ok(())
    }
    
    fn multiply_tensors<T: Clone + Send + Sync + 'static>(
        &self,
        a: &dyn torsh_backends::Buffer<T>,
        b: &dyn torsh_backends::Buffer<T>,
        output: &mut dyn torsh_backends::Buffer<T>,
    ) -> Result<(), BackendError> {
        // Downcast to CUDA buffers
        let a_cuda = a.as_any().downcast_ref::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input A".to_string(),
            })?;
        let b_cuda = b.as_any().downcast_ref::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input B".to_string(),
            })?;
        let output_cuda = output.as_any_mut().downcast_mut::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for output".to_string(),
            })?;
        
        // For now, only support f32
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let a_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(a_cuda) };
            let b_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(b_cuda) };
            let output_f32 = unsafe { std::mem::transmute::<&mut CudaBuffer<T>, &mut CudaBuffer<f32>>(output_cuda) };
            
            self.elementwise_mul_f32(a_f32, b_f32, output_f32, None)
                .map_err(|e| BackendError::Runtime {
                    message: e.to_string(),
                })?;
        } else {
            return Err(BackendError::UnsupportedOperation {
                operation: "multiply_tensors".to_string(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        }
        
        Ok(())
    }
    
    fn matmul<T: Clone + Send + Sync + 'static>(
        &self,
        a: &dyn torsh_backends::Buffer<T>,
        b: &dyn torsh_backends::Buffer<T>,
        output: &mut dyn torsh_backends::Buffer<T>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), BackendError> {
        // Downcast to CUDA buffers
        let a_cuda = a.as_any().downcast_ref::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input A".to_string(),
            })?;
        let b_cuda = b.as_any().downcast_ref::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for input B".to_string(),
            })?;
        let output_cuda = output.as_any_mut().downcast_mut::<CudaBuffer<T>>()
            .ok_or_else(|| BackendError::InvalidBuffer {
                message: "Expected CUDA buffer for output".to_string(),
            })?;
        
        // For now, only support f32
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let a_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(a_cuda) };
            let b_f32 = unsafe { std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(b_cuda) };
            let output_f32 = unsafe { std::mem::transmute::<&mut CudaBuffer<T>, &mut CudaBuffer<f32>>(output_cuda) };
            
            self.matmul_f32(a_f32, b_f32, output_f32, m, n, k, None)
                .map_err(|e| BackendError::Runtime {
                    message: e.to_string(),
                })?;
        } else {
            return Err(BackendError::UnsupportedOperation {
                operation: "matmul".to_string(),
                dtype: std::any::type_name::<T>().to_string(),
            });
        }
        
        Ok(())
    }
}

/// Convolution 2D configuration
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    pub batch_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub kernel_height: usize,
    pub kernel_width: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cuda_backend_creation() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await;
            assert!(backend.is_ok());
            
            let backend = backend.unwrap();
            assert_eq!(backend.name(), "cuda");
            assert!(backend.is_available());
        }
    }
    
    #[tokio::test]
    async fn test_buffer_creation() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();
            
            let buffer = backend.create_buffer::<f32>(1024, DType::F32);
            assert!(buffer.is_ok());
            
            let buffer = buffer.unwrap();
            assert_eq!(buffer.len(), 1024);
            assert_eq!(buffer.dtype(), DType::F32);
        }
    }
    
    #[tokio::test]
    async fn test_elementwise_addition() {
        if crate::is_available() {
            let config = CudaBackendConfig::default();
            let backend = CudaBackend::initialize(config).await.unwrap();
            
            let mut a = backend.create_buffer::<f32>(4, DType::F32).unwrap();
            let mut b = backend.create_buffer::<f32>(4, DType::F32).unwrap();
            let mut output = backend.create_buffer::<f32>(4, DType::F32).unwrap();
            
            // Copy test data
            let data_a = vec![1.0, 2.0, 3.0, 4.0];
            let data_b = vec![5.0, 6.0, 7.0, 8.0];
            
            a.copy_from_host(&data_a).unwrap();
            b.copy_from_host(&data_b).unwrap();
            
            // Perform addition
            backend.elementwise_add_f32(&a, &b, &mut output, None).unwrap();
            
            // Copy result back
            let mut result = vec![0.0; 4];
            output.copy_to_host(&mut result).unwrap();
            
            assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        }
    }
}