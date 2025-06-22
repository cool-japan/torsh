//! cuDNN integration for high-performance deep learning operations

use std::sync::Arc;
use std::sync::Mutex;
use std::collections::HashMap;

use crate::error::{CudaError, CudaResult};
use crate::stream::CudaStream;
use torsh_core::{DType, TensorError};

#[cfg(feature = "cudnn")]
use cudnn_sys::*;

/// cuDNN handle wrapper
pub struct CudnnHandle {
    #[cfg(feature = "cudnn")]
    handle: cudnnHandle_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}

impl CudnnHandle {
    /// Create new cuDNN handle
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut handle: cudnnHandle_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreate(&mut handle) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to create cuDNN handle: {:?}", status)));
            }
            Ok(Self { handle })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
        }
    }

    /// Set stream for cuDNN operations
    pub fn set_stream(&mut self, stream: &CudaStream) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let status = unsafe { cudnnSetStream(self.handle, stream.raw() as cudaStream_t) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to set cuDNN stream: {:?}", status)));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = stream;
            Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
        }
    }

    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnHandle_t {
        self.handle
    }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            unsafe {
                cudnnDestroy(self.handle);
            }
        }
    }
}

/// Tensor descriptor for cuDNN operations
pub struct TensorDescriptor {
    #[cfg(feature = "cudnn")]
    desc: cudnnTensorDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl TensorDescriptor {
    /// Create new tensor descriptor
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut desc: cudnnTensorDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreateTensorDescriptor(&mut desc) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to create tensor descriptor: {:?}", status)));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self { _phantom: std::marker::PhantomData })
        }
    }

    /// Set tensor descriptor with NCHW format
    pub fn set_4d(&mut self, dtype: DType, n: i32, c: i32, h: i32, w: i32) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_type = match dtype {
                DType::F32 => cudnnDataType_t::CUDNN_DATA_FLOAT,
                DType::F64 => cudnnDataType_t::CUDNN_DATA_DOUBLE,
                DType::F16 => cudnnDataType_t::CUDNN_DATA_HALF,
                _ => return Err(CudaError::CudnnError(format!("Unsupported dtype for cuDNN: {:?}", dtype))),
            };

            let status = unsafe {
                cudnnSetTensor4dDescriptor(
                    self.desc,
                    cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    cudnn_type,
                    n, c, h, w
                )
            };
            
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to set tensor descriptor: {:?}", status)));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (dtype, n, c, h, w);
            Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
        }
    }

    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnTensorDescriptor_t {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            unsafe {
                cudnnDestroyTensorDescriptor(self.desc);
            }
        }
    }
}

/// Filter (kernel) descriptor for convolution operations
pub struct FilterDescriptor {
    #[cfg(feature = "cudnn")]
    desc: cudnnFilterDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl FilterDescriptor {
    /// Create new filter descriptor
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut desc: cudnnFilterDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreateFilterDescriptor(&mut desc) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to create filter descriptor: {:?}", status)));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self { _phantom: std::marker::PhantomData })
        }
    }

    /// Set filter descriptor for 2D convolution
    pub fn set_4d(&mut self, dtype: DType, k: i32, c: i32, h: i32, w: i32) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_type = match dtype {
                DType::F32 => cudnnDataType_t::CUDNN_DATA_FLOAT,
                DType::F64 => cudnnDataType_t::CUDNN_DATA_DOUBLE,
                DType::F16 => cudnnDataType_t::CUDNN_DATA_HALF,
                _ => return Err(CudaError::CudnnError(format!("Unsupported dtype for cuDNN: {:?}", dtype))),
            };

            let status = unsafe {
                cudnnSetFilter4dDescriptor(
                    self.desc,
                    cudnn_type,
                    cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    k, c, h, w
                )
            };
            
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to set filter descriptor: {:?}", status)));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (dtype, k, c, h, w);
            Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
        }
    }

    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnFilterDescriptor_t {
        self.desc
    }
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            unsafe {
                cudnnDestroyFilterDescriptor(self.desc);
            }
        }
    }
}

/// Convolution descriptor
pub struct ConvolutionDescriptor {
    #[cfg(feature = "cudnn")]
    desc: cudnnConvolutionDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl ConvolutionDescriptor {
    /// Create new convolution descriptor
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut desc: cudnnConvolutionDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreateConvolutionDescriptor(&mut desc) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to create convolution descriptor: {:?}", status)));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self { _phantom: std::marker::PhantomData })
        }
    }

    /// Set 2D convolution descriptor
    pub fn set_2d(&mut self, pad_h: i32, pad_w: i32, stride_h: i32, stride_w: i32, dilation_h: i32, dilation_w: i32, mode: ConvolutionMode) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_mode = match mode {
                ConvolutionMode::Convolution => cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
                ConvolutionMode::CrossCorrelation => cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            };

            let status = unsafe {
                cudnnSetConvolution2dDescriptor(
                    self.desc,
                    pad_h, pad_w,
                    stride_h, stride_w,
                    dilation_h, dilation_w,
                    cudnn_mode,
                    cudnnDataType_t::CUDNN_DATA_FLOAT
                )
            };
            
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to set convolution descriptor: {:?}", status)));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, mode);
            Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
        }
    }

    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnConvolutionDescriptor_t {
        self.desc
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            unsafe {
                cudnnDestroyConvolutionDescriptor(self.desc);
            }
        }
    }
}

/// Convolution mode
#[derive(Debug, Clone, Copy)]
pub enum ConvolutionMode {
    Convolution,
    CrossCorrelation,
}

/// Activation descriptor for activation functions
pub struct ActivationDescriptor {
    #[cfg(feature = "cudnn")]
    desc: cudnnActivationDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl ActivationDescriptor {
    /// Create new activation descriptor
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut desc: cudnnActivationDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreateActivationDescriptor(&mut desc) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to create activation descriptor: {:?}", status)));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self { _phantom: std::marker::PhantomData })
        }
    }

    /// Set activation descriptor
    pub fn set(&mut self, mode: ActivationMode, nan_opt: NanPropagation, coef: f64) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_mode = match mode {
                ActivationMode::Sigmoid => cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
                ActivationMode::Relu => cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
                ActivationMode::Tanh => cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
                ActivationMode::ClippedRelu => cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU,
                ActivationMode::Elu => cudnnActivationMode_t::CUDNN_ACTIVATION_ELU,
            };

            let cudnn_nan = match nan_opt {
                NanPropagation::NotPropagateNan => cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                NanPropagation::PropagateNan => cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
            };

            let status = unsafe {
                cudnnSetActivationDescriptor(
                    self.desc,
                    cudnn_mode,
                    cudnn_nan,
                    coef
                )
            };
            
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("Failed to set activation descriptor: {:?}", status)));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (mode, nan_opt, coef);
            Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
        }
    }

    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnActivationDescriptor_t {
        self.desc
    }
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            unsafe {
                cudnnDestroyActivationDescriptor(self.desc);
            }
        }
    }
}

/// Activation function modes
#[derive(Debug, Clone, Copy)]
pub enum ActivationMode {
    Sigmoid,
    Relu,
    Tanh,
    ClippedRelu,
    Elu,
}

/// NaN propagation modes
#[derive(Debug, Clone, Copy)]
pub enum NanPropagation {
    NotPropagateNan,
    PropagateNan,
}

/// cuDNN operations interface
pub struct CudnnOps {
    handle: Arc<Mutex<CudnnHandle>>,
    cache: Mutex<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>,
}

impl CudnnOps {
    /// Create new cuDNN operations
    pub fn new() -> CudaResult<Self> {
        let handle = CudnnHandle::new()?;
        Ok(Self {
            handle: Arc::new(Mutex::new(handle)),
            cache: Mutex::new(HashMap::new()), 
        })
    }

    /// Perform 2D convolution forward pass
    pub fn conv2d_forward(
        &self,
        input: cust::DevicePointer<f32>,
        weight: cust::DevicePointer<f32>,
        bias: Option<cust::DevicePointer<f32>>,
        output: cust::DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32), // (N, C, H, W)
        weight_shape: (i32, i32, i32, i32), // (K, C, H, W)
        output_shape: (i32, i32, i32, i32), // (N, K, H, W)
        padding: (i32, i32),
        stride: (i32, i32),
        dilation: (i32, i32),
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            // Create descriptors
            let mut input_desc = TensorDescriptor::new()?;
            input_desc.set_4d(DType::F32, input_shape.0, input_shape.1, input_shape.2, input_shape.3)?;

            let mut weight_desc = FilterDescriptor::new()?;
            weight_desc.set_4d(DType::F32, weight_shape.0, weight_shape.1, weight_shape.2, weight_shape.3)?;

            let mut output_desc = TensorDescriptor::new()?;
            output_desc.set_4d(DType::F32, output_shape.0, output_shape.1, output_shape.2, output_shape.3)?;

            let mut conv_desc = ConvolutionDescriptor::new()?;
            conv_desc.set_2d(padding.0, padding.1, stride.0, stride.1, dilation.0, dilation.1, ConvolutionMode::CrossCorrelation)?;

            // Perform convolution
            let alpha = 1.0f32;
            let beta = 0.0f32;

            let handle = self.handle.lock().unwrap();
            let status = unsafe {
                cudnnConvolutionForward(
                    handle.raw(),
                    &alpha as *const f32 as *const std::ffi::c_void,
                    input_desc.raw(),
                    input.as_raw_mut() as *const std::ffi::c_void,
                    weight_desc.raw(),
                    weight.as_raw_mut() as *const std::ffi::c_void,
                    conv_desc.raw(),
                    cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    std::ptr::null_mut(),
                    0,
                    &beta as *const f32 as *const std::ffi::c_void,
                    output_desc.raw(),
                    output.as_raw_mut() as *mut std::ffi::c_void,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("cuDNN convolution forward failed: {:?}", status)));
            }

            // Add bias if provided
            if let Some(bias_ptr) = bias {
                let mut bias_desc = TensorDescriptor::new()?;
                bias_desc.set_4d(DType::F32, 1, output_shape.1, 1, 1)?;

                let status = unsafe {
                    cudnnAddTensor(
                        handle.raw(),
                        &alpha as *const f32 as *const std::ffi::c_void,
                        bias_desc.raw(),
                        bias_ptr.as_raw_mut() as *const std::ffi::c_void,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        output_desc.raw(),
                        output.as_raw_mut() as *mut std::ffi::c_void,
                    )
                };

                if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                    return Err(CudaError::CudnnError(format!("cuDNN bias addition failed: {:?}", status)));
                }
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (input, weight, bias, output, input_shape, weight_shape, output_shape, padding, stride, dilation);
            Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
        }
    }

    /// Perform activation forward pass
    pub fn activation_forward(
        &self,
        mode: ActivationMode,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let mut input_desc = TensorDescriptor::new()?;
            input_desc.set_4d(DType::F32, shape.0, shape.1, shape.2, shape.3)?;

            let mut output_desc = TensorDescriptor::new()?;
            output_desc.set_4d(DType::F32, shape.0, shape.1, shape.2, shape.3)?;

            let mut act_desc = ActivationDescriptor::new()?;
            act_desc.set(mode, NanPropagation::NotPropagateNan, 0.0)?;

            let alpha = 1.0f32;
            let beta = 0.0f32;

            let handle = self.handle.lock().unwrap();
            let status = unsafe {
                cudnnActivationForward(
                    handle.raw(),
                    act_desc.raw(),
                    &alpha as *const f32 as *const std::ffi::c_void,
                    input_desc.raw(),
                    input.as_raw_mut() as *const std::ffi::c_void,
                    &beta as *const f32 as *const std::ffi::c_void,
                    output_desc.raw(),
                    output.as_raw_mut() as *mut std::ffi::c_void,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!("cuDNN activation forward failed: {:?}", status)));
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (mode, input, output, shape);
            Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
        }
    }
}

/// Initialize cuDNN library
pub fn init() -> CudaResult<()> {
    #[cfg(feature = "cudnn")]
    {
        // cuDNN initialization is handled per-handle
        Ok(())
    }
    #[cfg(not(feature = "cudnn"))]
    {
        Err(CudaError::CudnnError("cuDNN not available - feature not enabled".to_string()))
    }
}

/// Check if cuDNN is available
pub fn is_available() -> bool {
    #[cfg(feature = "cudnn")]
    {
        CudnnHandle::new().is_ok()
    }
    #[cfg(not(feature = "cudnn"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cudnn_availability() {
        // This test will pass if cuDNN is available, skip if not
        if is_available() {
            let _handle = CudnnHandle::new().unwrap();
        }
    }

    #[test]
    fn test_tensor_descriptor() {
        if is_available() {
            let mut desc = TensorDescriptor::new().unwrap();
            desc.set_4d(DType::F32, 1, 3, 224, 224).unwrap();
        }
    }

    #[test]
    fn test_convolution_descriptor() {
        if is_available() {
            let mut desc = ConvolutionDescriptor::new().unwrap();
            desc.set_2d(1, 1, 1, 1, 1, 1, ConvolutionMode::CrossCorrelation).unwrap();
        }
    }

    #[test]
    fn test_activation_descriptor() {
        if is_available() {
            let mut desc = ActivationDescriptor::new().unwrap();
            desc.set(ActivationMode::Relu, NanPropagation::NotPropagateNan, 0.0).unwrap();
        }
    }
}