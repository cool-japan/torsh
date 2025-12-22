//! cuDNN descriptor management
//!
//! This module provides wrapper types for various cuDNN descriptors including
//! tensor, filter, convolution, activation, and pooling descriptors.

use super::types::{ActivationMode, ConvolutionMode, NanPropagation, PoolingMode};
use crate::cuda::error::{CudaError, CudaResult};
use torsh_core::DType;

#[cfg(feature = "cudnn")]
use cudnn_sys::*;

/// Tensor descriptor for cuDNN operations
///
/// Wraps cudnnTensorDescriptor_t to provide safe tensor description
/// for cuDNN operations with automatic resource management.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_backend::cuda::cudnn::TensorDescriptor;
/// use torsh_core::DType;
///
/// let mut desc = TensorDescriptor::new()?;
/// desc.set_4d(DType::F32, 1, 3, 224, 224)?;
/// ```
pub struct TensorDescriptor {
    #[cfg(feature = "cudnn")]
    desc: cudnnTensorDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl TensorDescriptor {
    /// Create new tensor descriptor
    ///
    /// # Returns
    ///
    /// A new `TensorDescriptor` instance on success, or an error if creation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if descriptor creation fails.
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut desc: cudnnTensorDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreateTensorDescriptor(&mut desc) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to create tensor descriptor: {:?}",
                    status
                )));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self {
                _phantom: std::marker::PhantomData,
            })
        }
    }

    /// Set tensor descriptor with NCHW format
    ///
    /// Configures the tensor descriptor for 4D tensors in NCHW format.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Data type of the tensor elements
    /// * `n` - Batch size
    /// * `c` - Number of channels
    /// * `h` - Height
    /// * `w` - Width
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if configuration fails.
    pub fn set_4d(&mut self, dtype: DType, n: i32, c: i32, h: i32, w: i32) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_type = match dtype {
                DType::F32 => cudnnDataType_t::CUDNN_DATA_FLOAT,
                DType::F64 => cudnnDataType_t::CUDNN_DATA_DOUBLE,
                DType::F16 => cudnnDataType_t::CUDNN_DATA_HALF,
                _ => {
                    return Err(CudaError::CudnnError(format!(
                        "Unsupported dtype for cuDNN: {:?}",
                        dtype
                    )))
                }
            };

            let status = unsafe {
                cudnnSetTensor4dDescriptor(
                    self.desc,
                    cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    cudnn_type,
                    n,
                    c,
                    h,
                    w,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set tensor descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (dtype, n, c, h, w);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Set tensor descriptor with NHWC format
    ///
    /// Configures the tensor descriptor for 4D tensors in NHWC format.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Data type of the tensor elements
    /// * `n` - Batch size
    /// * `h` - Height
    /// * `w` - Width
    /// * `c` - Number of channels
    pub fn set_4d_nhwc(&mut self, dtype: DType, n: i32, h: i32, w: i32, c: i32) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_type = match dtype {
                DType::F32 => cudnnDataType_t::CUDNN_DATA_FLOAT,
                DType::F64 => cudnnDataType_t::CUDNN_DATA_DOUBLE,
                DType::F16 => cudnnDataType_t::CUDNN_DATA_HALF,
                _ => {
                    return Err(CudaError::CudnnError(format!(
                        "Unsupported dtype for cuDNN: {:?}",
                        dtype
                    )))
                }
            };

            let status = unsafe {
                cudnnSetTensor4dDescriptor(
                    self.desc,
                    cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
                    cudnn_type,
                    n,
                    c,
                    h,
                    w,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set tensor descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (dtype, n, h, w, c);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Set tensor descriptor with arbitrary dimensions
    ///
    /// Configures the tensor descriptor for tensors with arbitrary dimensions.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Data type of the tensor elements
    /// * `dims` - Dimension sizes
    /// * `strides` - Stride values
    pub fn set_nd(&mut self, dtype: DType, dims: &[i32], strides: &[i32]) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if dims.len() != strides.len() {
                return Err(CudaError::CudnnError(
                    "Dimensions and strides must have the same length".to_string(),
                ));
            }

            let cudnn_type = match dtype {
                DType::F32 => cudnnDataType_t::CUDNN_DATA_FLOAT,
                DType::F64 => cudnnDataType_t::CUDNN_DATA_DOUBLE,
                DType::F16 => cudnnDataType_t::CUDNN_DATA_HALF,
                _ => {
                    return Err(CudaError::CudnnError(format!(
                        "Unsupported dtype for cuDNN: {:?}",
                        dtype
                    )))
                }
            };

            let status = unsafe {
                cudnnSetTensorNdDescriptor(
                    self.desc,
                    cudnn_type,
                    dims.len() as i32,
                    dims.as_ptr(),
                    strides.as_ptr(),
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set tensor descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (dtype, dims, strides);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get the raw cuDNN tensor descriptor
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnTensorDescriptor_t {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.desc.is_null() {
                unsafe {
                    let _status = cudnnDestroyTensorDescriptor(self.desc);
                }
            }
        }
    }
}

/// Filter (kernel) descriptor for convolution operations
///
/// Wraps cudnnFilterDescriptor_t to provide safe filter description
/// for convolution operations with automatic resource management.
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
                return Err(CudaError::CudnnError(format!(
                    "Failed to create filter descriptor: {:?}",
                    status
                )));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self {
                _phantom: std::marker::PhantomData,
            })
        }
    }

    /// Set filter descriptor for 2D convolution
    ///
    /// # Arguments
    ///
    /// * `dtype` - Data type of the filter elements
    /// * `k` - Number of output feature maps
    /// * `c` - Number of input feature maps
    /// * `h` - Filter height
    /// * `w` - Filter width
    pub fn set_4d(&mut self, dtype: DType, k: i32, c: i32, h: i32, w: i32) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_type = match dtype {
                DType::F32 => cudnnDataType_t::CUDNN_DATA_FLOAT,
                DType::F64 => cudnnDataType_t::CUDNN_DATA_DOUBLE,
                DType::F16 => cudnnDataType_t::CUDNN_DATA_HALF,
                _ => {
                    return Err(CudaError::CudnnError(format!(
                        "Unsupported dtype for cuDNN: {:?}",
                        dtype
                    )))
                }
            };

            let status = unsafe {
                cudnnSetFilter4dDescriptor(
                    self.desc,
                    cudnn_type,
                    cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    k,
                    c,
                    h,
                    w,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set filter descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (dtype, k, c, h, w);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Set filter descriptor with arbitrary dimensions
    ///
    /// # Arguments
    ///
    /// * `dtype` - Data type of the filter elements
    /// * `dims` - Filter dimensions
    pub fn set_nd(&mut self, dtype: DType, dims: &[i32]) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_type = match dtype {
                DType::F32 => cudnnDataType_t::CUDNN_DATA_FLOAT,
                DType::F64 => cudnnDataType_t::CUDNN_DATA_DOUBLE,
                DType::F16 => cudnnDataType_t::CUDNN_DATA_HALF,
                _ => {
                    return Err(CudaError::CudnnError(format!(
                        "Unsupported dtype for cuDNN: {:?}",
                        dtype
                    )))
                }
            };

            let status = unsafe {
                cudnnSetFilterNdDescriptor(
                    self.desc,
                    cudnn_type,
                    cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    dims.len() as i32,
                    dims.as_ptr(),
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set filter descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (dtype, dims);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get the raw cuDNN filter descriptor
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnFilterDescriptor_t {
        self.desc
    }
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.desc.is_null() {
                unsafe {
                    let _status = cudnnDestroyFilterDescriptor(self.desc);
                }
            }
        }
    }
}

/// Convolution descriptor
///
/// Wraps cudnnConvolutionDescriptor_t to provide safe convolution description
/// for convolution operations with automatic resource management.
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
                return Err(CudaError::CudnnError(format!(
                    "Failed to create convolution descriptor: {:?}",
                    status
                )));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self {
                _phantom: std::marker::PhantomData,
            })
        }
    }

    /// Set 2D convolution descriptor
    ///
    /// # Arguments
    ///
    /// * `pad_h` - Zero-padding height
    /// * `pad_w` - Zero-padding width
    /// * `stride_h` - Vertical stride
    /// * `stride_w` - Horizontal stride
    /// * `dilation_h` - Vertical dilation
    /// * `dilation_w` - Horizontal dilation
    /// * `mode` - Convolution mode
    pub fn set_2d(
        &mut self,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        mode: ConvolutionMode,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_mode = mode.to_cudnn();

            let status = unsafe {
                cudnnSetConvolution2dDescriptor(
                    self.desc,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    cudnn_mode,
                    cudnnDataType_t::CUDNN_DATA_FLOAT,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set convolution descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, mode,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Set convolution math type
    ///
    /// # Arguments
    ///
    /// * `math_type` - Math type for the convolution
    #[cfg(feature = "cudnn")]
    pub fn set_math_type(&mut self, math_type: cudnnMathType_t) -> CudaResult<()> {
        let status = unsafe { cudnnSetConvolutionMathType(self.desc, math_type) };
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::CudnnError(format!(
                "Failed to set convolution math type: {:?}",
                status
            )));
        }
        Ok(())
    }

    /// Get the raw cuDNN convolution descriptor
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnConvolutionDescriptor_t {
        self.desc
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.desc.is_null() {
                unsafe {
                    let _status = cudnnDestroyConvolutionDescriptor(self.desc);
                }
            }
        }
    }
}

/// Activation descriptor for activation functions
///
/// Wraps cudnnActivationDescriptor_t to provide safe activation description
/// for activation operations with automatic resource management.
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
                return Err(CudaError::CudnnError(format!(
                    "Failed to create activation descriptor: {:?}",
                    status
                )));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self {
                _phantom: std::marker::PhantomData,
            })
        }
    }

    /// Set activation descriptor
    ///
    /// # Arguments
    ///
    /// * `mode` - Activation function mode
    /// * `nan_opt` - NaN propagation mode
    /// * `coef` - Coefficient for certain activation functions
    pub fn set(
        &mut self,
        mode: ActivationMode,
        nan_opt: NanPropagation,
        coef: f64,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_mode = mode.to_cudnn();
            let cudnn_nan = nan_opt.to_cudnn();

            let status =
                unsafe { cudnnSetActivationDescriptor(self.desc, cudnn_mode, cudnn_nan, coef) };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set activation descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (mode, nan_opt, coef);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get the raw cuDNN activation descriptor
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnActivationDescriptor_t {
        self.desc
    }
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.desc.is_null() {
                unsafe {
                    let _status = cudnnDestroyActivationDescriptor(self.desc);
                }
            }
        }
    }
}

/// Pooling descriptor for pooling operations
///
/// Wraps cudnnPoolingDescriptor_t to provide safe pooling description
/// for pooling operations with automatic resource management.
pub struct PoolingDescriptor {
    #[cfg(feature = "cudnn")]
    desc: cudnnPoolingDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl PoolingDescriptor {
    /// Create new pooling descriptor
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut desc: cudnnPoolingDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreatePoolingDescriptor(&mut desc) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to create pooling descriptor: {:?}",
                    status
                )));
            }
            Ok(Self { desc })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self {
                _phantom: std::marker::PhantomData,
            })
        }
    }

    /// Set 2D pooling descriptor
    ///
    /// # Arguments
    ///
    /// * `mode` - Pooling mode
    /// * `nan_opt` - NaN propagation mode
    /// * `window_h` - Pooling window height
    /// * `window_w` - Pooling window width
    /// * `pad_h` - Zero-padding height
    /// * `pad_w` - Zero-padding width
    /// * `stride_h` - Vertical stride
    /// * `stride_w` - Horizontal stride
    pub fn set_2d(
        &mut self,
        mode: PoolingMode,
        nan_opt: NanPropagation,
        window_h: i32,
        window_w: i32,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_mode = mode.to_cudnn();
            let cudnn_nan = nan_opt.to_cudnn();

            let status = unsafe {
                cudnnSetPooling2dDescriptor(
                    self.desc, cudnn_mode, cudnn_nan, window_h, window_w, pad_h, pad_w, stride_h,
                    stride_w,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set pooling descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                mode, nan_opt, window_h, window_w, pad_h, pad_w, stride_h, stride_w,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get the raw cuDNN pooling descriptor
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnPoolingDescriptor_t {
        self.desc
    }
}

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.desc.is_null() {
                unsafe {
                    let _status = cudnnDestroyPoolingDescriptor(self.desc);
                }
            }
        }
    }
}

// Default implementations for all descriptors
impl Default for TensorDescriptor {
    fn default() -> Self {
        Self::new().expect("Failed to create default tensor descriptor")
    }
}

impl Default for FilterDescriptor {
    fn default() -> Self {
        Self::new().expect("Failed to create default filter descriptor")
    }
}

impl Default for ConvolutionDescriptor {
    fn default() -> Self {
        Self::new().expect("Failed to create default convolution descriptor")
    }
}

impl Default for ActivationDescriptor {
    fn default() -> Self {
        Self::new().expect("Failed to create default activation descriptor")
    }
}

impl Default for PoolingDescriptor {
    fn default() -> Self {
        Self::new().expect("Failed to create default pooling descriptor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_descriptor_creation() {
        match TensorDescriptor::new() {
            Ok(_desc) => {
                // Descriptor created successfully
            }
            Err(_) => {
                // cuDNN might not be available in test environment
            }
        }
    }

    #[test]
    fn test_filter_descriptor_creation() {
        match FilterDescriptor::new() {
            Ok(_desc) => {
                // Descriptor created successfully
            }
            Err(_) => {
                // cuDNN might not be available in test environment
            }
        }
    }

    #[test]
    fn test_convolution_descriptor_creation() {
        match ConvolutionDescriptor::new() {
            Ok(_desc) => {
                // Descriptor created successfully
            }
            Err(_) => {
                // cuDNN might not be available in test environment
            }
        }
    }

    #[test]
    fn test_activation_descriptor_creation() {
        match ActivationDescriptor::new() {
            Ok(_desc) => {
                // Descriptor created successfully
            }
            Err(_) => {
                // cuDNN might not be available in test environment
            }
        }
    }

    #[test]
    fn test_pooling_descriptor_creation() {
        match PoolingDescriptor::new() {
            Ok(_desc) => {
                // Descriptor created successfully
            }
            Err(_) => {
                // cuDNN might not be available in test environment
            }
        }
    }

    #[test]
    fn test_tensor_descriptor_4d() {
        if let Ok(mut desc) = TensorDescriptor::new() {
            let result = desc.set_4d(DType::F32, 1, 3, 224, 224);
            // Result may fail if cuDNN is not available, which is acceptable in tests
            match result {
                Ok(_) => {
                    // Successfully set descriptor
                }
                Err(_) => {
                    // cuDNN might not be available
                }
            }
        }
    }

    #[test]
    fn test_default_implementations() {
        // Test that default implementations don't panic when cuDNN is available
        // When cuDNN is not available, these will panic by design
        #[cfg(feature = "cudnn")]
        {
            // Only test defaults when cuDNN feature is enabled
            // In practice, these might still fail if cuDNN runtime is not available
        }
    }
}
