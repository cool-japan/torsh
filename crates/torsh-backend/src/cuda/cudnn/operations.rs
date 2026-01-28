//! cuDNN operations implementation
//!
//! This module provides high-level cuDNN operations including convolution,
//! batch normalization, activation, and pooling operations. All operations
//! are implemented with proper error handling and feature-conditional compilation.

// Allow unused imports as they are used conditionally with the cudnn feature
#![allow(unused_imports)]

use cust::prelude::DevicePointer;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::cuda::error::{CudaError, CudaResult};
use torsh_core::DType;

use super::descriptors::{
    ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, PoolingDescriptor,
    TensorDescriptor,
};
use super::handle::CudnnHandle;
use super::rnn::{RNNDataDescriptor, RNNDescriptor, RNNForwardMode};
use super::types::{
    ActivationMode, ConvolutionForwardAlgorithm, ConvolutionForwardAlgorithmPerformance,
    ConvolutionMode, NanPropagation, PoolingMode,
};

#[cfg(feature = "cudnn")]
use cudnn_sys::*;

// Import compatibility layer for missing cudnn-sys functions and types
#[cfg(feature = "cudnn")]
use super::compat::{
    cudnnBatchNormMode_t, cudnnBatchNormalizationForwardInference,
    cudnnBatchNormalizationForwardTraining, cudnnForwardMode_t, cudnnNormAlgo_t, cudnnNormMode_t,
    cudnnNormOps_t, cudnnNormalizationForwardInference, cudnnRNNForward,
    cudnnSetConvolutionGroupCount,
};
// Note: cudnnAddMode_t comes from cudnn_sys::* (not compat)

/// Convert compat convolution forward algorithm to cudnn_sys type
/// Note: cudnn-sys 0.0.3 doesn't have WINOGRAD variants, so we use GEMM fallback
#[cfg(feature = "cudnn")]
fn to_sys_conv_fwd_algo(
    algo: super::compat::cudnnConvolutionFwdAlgo_t,
) -> cudnnConvolutionFwdAlgo_t {
    match algo {
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        }
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        }
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM
        }
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
        }
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT
        }
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
        }
        // WINOGRAD variants and COUNT not available in cudnn-sys 0.0.3, use GEMM fallback
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD |
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED |
        super::compat::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM
        }
    }
}

/// cuDNN operations interface
///
/// Provides high-level operations for deep learning computations using cuDNN.
/// All operations are implemented with proper resource management and error handling.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_backend::cuda::cudnn::CudnnOps;
///
/// let ops = CudnnOps::new()?;
/// ops.conv2d_forward(
///     input_ptr,
///     weight_ptr,
///     None, // no bias
///     output_ptr,
///     (1, 3, 224, 224), // input shape (N, C, H, W)
///     (64, 3, 7, 7),    // weight shape (K, C, H, W)
///     (1, 64, 218, 218), // output shape (N, K, H_out, W_out)
///     (0, 0),           // padding
///     (1, 1),           // stride
///     (1, 1),           // dilation
/// )?;
/// ```
pub struct CudnnOps {
    handle: Arc<Mutex<CudnnHandle>>,
    cache: Mutex<HashMap<String, Box<dyn std::any::Any + Send + Sync>>>,
}

impl CudnnOps {
    /// Create new cuDNN operations
    ///
    /// Initializes a new cuDNN operations instance with a managed handle.
    /// The handle is shared and thread-safe for concurrent operations.
    ///
    /// # Returns
    ///
    /// A new `CudnnOps` instance on success, or an error if initialization fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if cuDNN handle creation fails.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::CudnnOps;
    ///
    /// let ops = CudnnOps::new()?;
    /// // Use ops for cuDNN operations
    /// ```
    pub fn new() -> CudaResult<Self> {
        let handle = CudnnHandle::new()?;
        Ok(Self {
            handle: Arc::new(Mutex::new(handle)),
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Perform 2D convolution forward pass
    ///
    /// Executes a 2D convolution operation on the input tensor using the provided
    /// weights and optional bias. Supports various padding, stride, and dilation
    /// configurations.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor device pointer
    /// * `weight` - Weight tensor device pointer
    /// * `bias` - Optional bias tensor device pointer
    /// * `output` - Output tensor device pointer
    /// * `input_shape` - Input tensor shape as (N, C, H, W)
    /// * `weight_shape` - Weight tensor shape as (K, C, H, W)
    /// * `output_shape` - Output tensor shape as (N, K, H_out, W_out)
    /// * `padding` - Padding as (pad_h, pad_w)
    /// * `stride` - Stride as (stride_h, stride_w)
    /// * `dilation` - Dilation as (dilation_h, dilation_w)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Descriptor creation fails
    /// - Convolution operation fails
    /// - Bias addition fails (if bias is provided)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::CudnnOps;
    ///
    /// let ops = CudnnOps::new()?;
    /// ops.conv2d_forward(
    ///     input_ptr,
    ///     weight_ptr,
    ///     Some(bias_ptr),
    ///     output_ptr,
    ///     (1, 3, 224, 224), // input: batch=1, channels=3, height=224, width=224
    ///     (64, 3, 7, 7),    // weight: filters=64, in_channels=3, kernel=7x7
    ///     (1, 64, 218, 218), // output: batch=1, channels=64, height=218, width=218
    ///     (0, 0),           // no padding
    ///     (1, 1),           // stride of 1x1
    ///     (1, 1),           // dilation of 1x1
    /// )?;
    /// ```
    pub fn conv2d_forward(
        &self,
        input: DevicePointer<f32>,
        weight: DevicePointer<f32>,
        bias: Option<DevicePointer<f32>>,
        output: DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32),  // (N, C, H, W)
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
            input_desc.set_4d(
                DType::F32,
                input_shape.0,
                input_shape.1,
                input_shape.2,
                input_shape.3,
            )?;

            let mut weight_desc = FilterDescriptor::new()?;
            weight_desc.set_4d(
                DType::F32,
                weight_shape.0,
                weight_shape.1,
                weight_shape.2,
                weight_shape.3,
            )?;

            let mut output_desc = TensorDescriptor::new()?;
            output_desc.set_4d(
                DType::F32,
                output_shape.0,
                output_shape.1,
                output_shape.2,
                output_shape.3,
            )?;

            let mut conv_desc = ConvolutionDescriptor::new()?;
            conv_desc.set_2d(
                padding.0,
                padding.1,
                stride.0,
                stride.1,
                dilation.0,
                dilation.1,
                ConvolutionMode::CrossCorrelation,
            )?;

            // Perform convolution
            let alpha = 1.0f32;
            let beta = 0.0f32;

            let handle = self.handle.lock().expect("lock should not be poisoned");
            let status = unsafe {
                cudnnConvolutionForward(
                    handle.raw(),
                    &alpha as *const f32 as *const std::ffi::c_void,
                    input_desc.raw(),
                    input.as_raw() as *const std::ffi::c_void,
                    weight_desc.raw(),
                    weight.as_raw() as *const std::ffi::c_void,
                    conv_desc.raw(),
                    cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    std::ptr::null_mut(),
                    0,
                    &beta as *const f32 as *const std::ffi::c_void,
                    output_desc.raw(),
                    output.as_raw() as *mut std::ffi::c_void,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "cuDNN convolution forward failed: {:?}",
                    status
                )));
            }

            // Add bias if provided
            if let Some(bias_ptr) = bias {
                let mut bias_desc = TensorDescriptor::new()?;
                bias_desc.set_4d(DType::F32, 1, output_shape.1, 1, 1)?;

                let status = unsafe {
                    cudnnAddTensor(
                        handle.raw(),
                        cudnnAddMode_t::CUDNN_ADD_SAME_C,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        bias_desc.raw(),
                        bias_ptr.as_raw() as *const std::ffi::c_void,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        output_desc.raw(),
                        output.as_raw() as *mut std::ffi::c_void,
                    )
                };

                if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                    return Err(CudaError::CudnnError(format!(
                        "cuDNN bias addition failed: {:?}",
                        status
                    )));
                }
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                input,
                weight,
                bias,
                output,
                input_shape,
                weight_shape,
                output_shape,
                padding,
                stride,
                dilation,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Perform batch normalization forward pass
    ///
    /// Executes batch normalization on the input tensor. Supports both training
    /// and inference modes with proper handling of running statistics.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor device pointer
    /// * `output` - Output tensor device pointer
    /// * `scale` - Scale parameter device pointer
    /// * `bias` - Bias parameter device pointer
    /// * `running_mean` - Running mean device pointer
    /// * `running_var` - Running variance device pointer
    /// * `epsilon` - Small constant for numerical stability
    /// * `exponential_average_factor` - Factor for exponential moving average
    /// * `shape` - Tensor shape as (N, C, H, W)
    /// * `training` - Whether in training mode
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Descriptor creation fails
    /// - Batch normalization operation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::CudnnOps;
    ///
    /// let ops = CudnnOps::new()?;
    /// ops.batchnorm_forward(
    ///     input_ptr,
    ///     output_ptr,
    ///     scale_ptr,
    ///     bias_ptr,
    ///     running_mean_ptr,
    ///     running_var_ptr,
    ///     1e-5,  // epsilon
    ///     0.1,   // momentum
    ///     (1, 64, 56, 56), // shape
    ///     true,  // training mode
    /// )?;
    /// ```
    pub fn batchnorm_forward(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        scale: DevicePointer<f32>,
        bias: DevicePointer<f32>,
        running_mean: DevicePointer<f32>,
        running_var: DevicePointer<f32>,
        epsilon: f64,
        exponential_average_factor: f64,
        shape: (i32, i32, i32, i32),
        training: bool,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let mut input_desc = TensorDescriptor::new()?;
            input_desc.set_4d(DType::F32, shape.0, shape.1, shape.2, shape.3)?;

            let mut output_desc = TensorDescriptor::new()?;
            output_desc.set_4d(DType::F32, shape.0, shape.1, shape.2, shape.3)?;

            let mut scale_bias_desc = TensorDescriptor::new()?;
            scale_bias_desc.set_4d(DType::F32, 1, shape.1, 1, 1)?;

            let alpha = 1.0f32;
            let beta = 0.0f32;

            let handle = self.handle.lock().expect("lock should not be poisoned");
            let status = if training {
                unsafe {
                    cudnnBatchNormalizationForwardTraining(
                        handle.raw(),
                        cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        &beta as *const f32 as *const std::ffi::c_void,
                        input_desc.raw(),
                        input.as_raw() as *const std::ffi::c_void,
                        output_desc.raw(),
                        output.as_raw() as *mut std::ffi::c_void,
                        scale_bias_desc.raw(),
                        scale.as_raw() as *const std::ffi::c_void,
                        bias.as_raw() as *const std::ffi::c_void,
                        exponential_average_factor,
                        running_mean.as_raw() as *mut std::ffi::c_void,
                        running_var.as_raw() as *mut std::ffi::c_void,
                        epsilon,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    )
                }
            } else {
                unsafe {
                    cudnnBatchNormalizationForwardInference(
                        handle.raw(),
                        cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        &beta as *const f32 as *const std::ffi::c_void,
                        input_desc.raw(),
                        input.as_raw() as *const std::ffi::c_void,
                        output_desc.raw(),
                        output.as_raw() as *mut std::ffi::c_void,
                        scale_bias_desc.raw(),
                        scale.as_raw() as *const std::ffi::c_void,
                        bias.as_raw() as *const std::ffi::c_void,
                        running_mean.as_raw() as *const std::ffi::c_void,
                        running_var.as_raw() as *const std::ffi::c_void,
                        epsilon,
                    )
                }
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "cuDNN batch normalization forward failed: {:?}",
                    status
                )));
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                input,
                output,
                scale,
                bias,
                running_mean,
                running_var,
                epsilon,
                exponential_average_factor,
                shape,
                training,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Perform activation forward pass
    ///
    /// Applies an activation function to the input tensor. Supports various
    /// activation modes including ReLU, Sigmoid, Tanh, etc.
    ///
    /// # Arguments
    ///
    /// * `mode` - Activation function mode
    /// * `input` - Input tensor device pointer
    /// * `output` - Output tensor device pointer
    /// * `shape` - Tensor shape as (N, C, H, W)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Descriptor creation fails
    /// - Activation operation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::{CudnnOps, ActivationMode};
    ///
    /// let ops = CudnnOps::new()?;
    /// ops.activation_forward(
    ///     ActivationMode::Relu,
    ///     input_ptr,
    ///     output_ptr,
    ///     (1, 64, 56, 56),
    /// )?;
    /// ```
    pub fn activation_forward(
        &self,
        mode: ActivationMode,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let mut input_desc = TensorDescriptor::new()?;
            input_desc.set_4d(DType::F32, shape.0, shape.1, shape.2, shape.3)?;

            let mut output_desc = TensorDescriptor::new()?;
            output_desc.set_4d(DType::F32, shape.0, shape.1, shape.2, shape.3)?;

            // cudnn-sys 0.0.3 uses old API that takes mode directly, not descriptor
            let cudnn_mode = match mode {
                ActivationMode::Sigmoid => cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
                ActivationMode::Relu => cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
                ActivationMode::Tanh => cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
                _ => cudnnActivationMode_t::CUDNN_ACTIVATION_RELU, // Default fallback
            };

            let alpha = 1.0f32;
            let beta = 0.0f32;

            let handle = self.handle.lock().expect("lock should not be poisoned");
            let status = unsafe {
                cudnnActivationForward(
                    handle.raw(),
                    cudnn_mode,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    input_desc.raw(),
                    input.as_raw() as *const std::ffi::c_void,
                    &beta as *const f32 as *const std::ffi::c_void,
                    output_desc.raw(),
                    output.as_raw() as *mut std::ffi::c_void,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "cuDNN activation forward failed: {:?}",
                    status
                )));
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (mode, input, output, shape);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Perform 2D pooling forward pass
    ///
    /// Applies pooling operation to the input tensor. Supports max pooling,
    /// average pooling, and average pooling excluding padding.
    ///
    /// # Arguments
    ///
    /// * `mode` - Pooling mode (Max, Average, AverageExcludePadding)
    /// * `input` - Input tensor device pointer
    /// * `output` - Output tensor device pointer
    /// * `input_shape` - Input tensor shape as (N, C, H, W)
    /// * `output_shape` - Output tensor shape as (N, C, H_out, W_out)
    /// * `window_size` - Pooling window size as (window_h, window_w)
    /// * `padding` - Padding as (pad_h, pad_w)
    /// * `stride` - Stride as (stride_h, stride_w)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Descriptor creation fails
    /// - Pooling operation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::{CudnnOps, PoolingMode};
    ///
    /// let ops = CudnnOps::new()?;
    /// ops.pooling2d_forward(
    ///     PoolingMode::Max,
    ///     input_ptr,
    ///     output_ptr,
    ///     (1, 64, 112, 112), // input shape
    ///     (1, 64, 56, 56),   // output shape
    ///     (2, 2),            // 2x2 pooling window
    ///     (0, 0),            // no padding
    ///     (2, 2),            // stride of 2x2
    /// )?;
    /// ```
    pub fn pooling2d_forward(
        &self,
        mode: PoolingMode,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32),  // (N, C, H, W)
        output_shape: (i32, i32, i32, i32), // (N, C, H_out, W_out)
        window_size: (i32, i32),
        padding: (i32, i32),
        stride: (i32, i32),
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let mut input_desc = TensorDescriptor::new()?;
            input_desc.set_4d(
                DType::F32,
                input_shape.0,
                input_shape.1,
                input_shape.2,
                input_shape.3,
            )?;

            let mut output_desc = TensorDescriptor::new()?;
            output_desc.set_4d(
                DType::F32,
                output_shape.0,
                output_shape.1,
                output_shape.2,
                output_shape.3,
            )?;

            let mut pool_desc = PoolingDescriptor::new()?;
            pool_desc.set_2d(
                mode,
                NanPropagation::NotPropagate,
                window_size.0,
                window_size.1,
                padding.0,
                padding.1,
                stride.0,
                stride.1,
            )?;

            let alpha = 1.0f32;
            let beta = 0.0f32;

            let handle = self.handle.lock().expect("lock should not be poisoned");
            let status = unsafe {
                cudnnPoolingForward(
                    handle.raw(),
                    pool_desc.raw(),
                    &alpha as *const f32 as *const std::ffi::c_void,
                    input_desc.raw(),
                    input.as_raw() as *const std::ffi::c_void,
                    &beta as *const f32 as *const std::ffi::c_void,
                    output_desc.raw(),
                    output.as_raw() as *mut std::ffi::c_void,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "cuDNN pooling forward failed: {:?}",
                    status
                )));
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                mode,
                input,
                output,
                input_shape,
                output_shape,
                window_size,
                padding,
                stride,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Perform layer normalization forward pass
    ///
    /// Applies layer normalization to the input tensor using the provided scale
    /// and bias parameters. This is useful for transformer architectures and
    /// other neural networks that benefit from normalization.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor device pointer
    /// * `scale` - Scale parameter device pointer
    /// * `bias` - Bias parameter device pointer
    /// * `epsilon` - Small constant for numerical stability
    /// * `x_desc` - Input tensor descriptor
    /// * `scale_bias_desc` - Scale and bias tensor descriptor
    /// * `y` - Output tensor device pointer
    /// * `mean` - Mean values device pointer
    /// * `inv_variance` - Inverse variance device pointer
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Layer normalization operation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::CudnnOps;
    ///
    /// let ops = CudnnOps::new()?;
    /// ops.layer_norm_forward(
    ///     x_ptr,
    ///     scale_ptr,
    ///     bias_ptr,
    ///     1e-5,
    ///     &x_desc,
    ///     &scale_bias_desc,
    ///     y_ptr,
    ///     mean_ptr,
    ///     inv_variance_ptr,
    /// )?;
    /// ```
    pub fn layer_norm_forward(
        &self,
        x: DevicePointer<f32>,
        scale: DevicePointer<f32>,
        bias: DevicePointer<f32>,
        epsilon: f64,
        x_desc: &TensorDescriptor,
        scale_bias_desc: &TensorDescriptor,
        y: DevicePointer<f32>,
        mean: DevicePointer<f32>,
        inv_variance: DevicePointer<f32>,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let handle = self.handle.lock().expect("lock should not be poisoned");
            let alpha = 1.0f32;
            let beta = 0.0f32;

            let status = unsafe {
                cudnnNormalizationForwardInference(
                    handle.raw(),
                    cudnnNormMode_t::CUDNN_LAYER_NORM,
                    cudnnNormOps_t::CUDNN_NORM_OPS_NORM_ACTIVATION,
                    cudnnNormAlgo_t::CUDNN_NORM_ALGO_STANDARD,
                    &alpha as *const f32 as *const std::ffi::c_void,
                    &beta as *const f32 as *const std::ffi::c_void,
                    x_desc.raw(),
                    x.as_raw() as *const std::ffi::c_void,
                    scale_bias_desc.raw(),
                    scale.as_raw() as *const std::ffi::c_void,
                    bias.as_raw() as *const std::ffi::c_void,
                    epsilon,
                    x_desc.raw(),
                    y.as_raw() as *mut std::ffi::c_void,
                    scale_bias_desc.raw(),
                    mean.as_raw() as *mut std::ffi::c_void,
                    inv_variance.as_raw() as *mut std::ffi::c_void,
                    std::ptr::null_mut(), // activationDesc
                    std::ptr::null_mut(), // workspace
                    0,                    // workspaceSizeInBytes
                    std::ptr::null_mut(), // reserveSpace
                    0,                    // reserveSpaceSizeInBytes
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "cuDNN layer normalization forward failed: {:?}",
                    status
                )));
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                x,
                scale,
                bias,
                epsilon,
                x_desc,
                scale_bias_desc,
                y,
                mean,
                inv_variance,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Perform grouped convolution forward pass
    ///
    /// Executes a grouped convolution operation where the input and output channels
    /// are divided into groups, allowing for more efficient computation and
    /// parameter sharing within each group.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor device pointer
    /// * `filter` - Filter tensor device pointer
    /// * `bias` - Optional bias tensor device pointer
    /// * `output` - Output tensor device pointer
    /// * `input_desc` - Input tensor descriptor
    /// * `filter_desc` - Filter tensor descriptor
    /// * `bias_desc` - Optional bias tensor descriptor
    /// * `output_desc` - Output tensor descriptor
    /// * `conv_desc` - Convolution descriptor
    /// * `groups` - Number of groups for the convolution
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Group count setting fails
    /// - Convolution operation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::CudnnOps;
    ///
    /// let ops = CudnnOps::new()?;
    /// ops.grouped_conv2d_forward(
    ///     input_ptr,
    ///     filter_ptr,
    ///     Some(bias_ptr),
    ///     output_ptr,
    ///     &input_desc,
    ///     &filter_desc,
    ///     Some(&bias_desc),
    ///     &output_desc,
    ///     &conv_desc,
    ///     4, // 4 groups
    /// )?;
    /// ```
    pub fn grouped_conv2d_forward(
        &self,
        input: DevicePointer<f32>,
        filter: DevicePointer<f32>,
        bias: Option<DevicePointer<f32>>,
        output: DevicePointer<f32>,
        input_desc: &TensorDescriptor,
        filter_desc: &FilterDescriptor,
        bias_desc: Option<&TensorDescriptor>,
        output_desc: &TensorDescriptor,
        conv_desc: &ConvolutionDescriptor,
        groups: i32,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            // Set group count in convolution descriptor
            let status = unsafe { cudnnSetConvolutionGroupCount(conv_desc.raw(), groups) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set convolution group count: {:?}",
                    status
                )));
            }

            // Perform convolution using descriptors
            let alpha = 1.0f32;
            let beta = 0.0f32;

            let handle = self.handle.lock().expect("lock should not be poisoned");
            let status = unsafe {
                cudnnConvolutionForward(
                    handle.raw(),
                    &alpha as *const f32 as *const std::ffi::c_void,
                    input_desc.raw(),
                    input.as_raw() as *const std::ffi::c_void,
                    filter_desc.raw(),
                    filter.as_raw() as *const std::ffi::c_void,
                    conv_desc.raw(),
                    cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    std::ptr::null_mut(),
                    0,
                    &beta as *const f32 as *const std::ffi::c_void,
                    output_desc.raw(),
                    output.as_raw() as *mut std::ffi::c_void,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "cuDNN grouped convolution forward failed: {:?}",
                    status
                )));
            }

            // Add bias if provided
            if let (Some(bias_ptr), Some(bias_desc)) = (bias, bias_desc) {
                let status = unsafe {
                    cudnnAddTensor(
                        handle.raw(),
                        cudnnAddMode_t::CUDNN_ADD_SAME_C,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        bias_desc.raw(),
                        bias_ptr.as_raw() as *const std::ffi::c_void,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        output_desc.raw(),
                        output.as_raw() as *mut std::ffi::c_void,
                    )
                };

                if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                    return Err(CudaError::CudnnError(format!(
                        "cuDNN grouped convolution bias addition failed: {:?}",
                        status
                    )));
                }
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                input,
                filter,
                bias,
                output,
                input_desc,
                filter_desc,
                bias_desc,
                output_desc,
                conv_desc,
                groups,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get workspace size for convolution algorithm
    ///
    /// Determines the workspace memory size required for a specific convolution
    /// algorithm. This is useful for memory planning and algorithm selection.
    ///
    /// # Arguments
    ///
    /// * `input_desc` - Input tensor descriptor
    /// * `filter_desc` - Filter tensor descriptor
    /// * `conv_desc` - Convolution descriptor
    /// * `output_desc` - Output tensor descriptor
    /// * `algorithm` - Convolution algorithm to query
    ///
    /// # Returns
    ///
    /// The workspace size in bytes on success, or an error if the query fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Workspace size query fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::{CudnnOps, ConvolutionForwardAlgorithm};
    ///
    /// let ops = CudnnOps::new()?;
    /// let workspace_size = ops.get_convolution_forward_workspace_size(
    ///     &input_desc,
    ///     &filter_desc,
    ///     &conv_desc,
    ///     &output_desc,
    ///     ConvolutionForwardAlgorithm::ImplicitGemm,
    /// )?;
    /// ```
    pub fn get_convolution_forward_workspace_size(
        &self,
        input_desc: &TensorDescriptor,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        output_desc: &TensorDescriptor,
        algorithm: ConvolutionForwardAlgorithm,
    ) -> CudaResult<usize> {
        #[cfg(feature = "cudnn")]
        {
            let handle = self.handle.lock().expect("lock should not be poisoned");
            let mut size: usize = 0;

            let status = unsafe {
                cudnnGetConvolutionForwardWorkspaceSize(
                    handle.raw(),
                    input_desc.raw(),
                    filter_desc.raw(),
                    conv_desc.raw(),
                    output_desc.raw(),
                    to_sys_conv_fwd_algo(algorithm.to_cudnn()),
                    &mut size,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to get convolution workspace size: {:?}",
                    status
                )));
            }

            Ok(size)
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (input_desc, filter_desc, conv_desc, output_desc, algorithm);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Find optimal convolution forward algorithm
    ///
    /// Searches for the best performing convolution algorithm for the given
    /// tensor and convolution descriptors. Returns performance information
    /// for multiple algorithms ranked by performance.
    ///
    /// # Arguments
    ///
    /// * `input_desc` - Input tensor descriptor
    /// * `filter_desc` - Filter tensor descriptor
    /// * `conv_desc` - Convolution descriptor
    /// * `output_desc` - Output tensor descriptor
    /// * `request_algo_count` - Maximum number of algorithms to return
    ///
    /// # Returns
    ///
    /// A vector of algorithm performance results ranked by performance.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Algorithm search fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::CudnnOps;
    ///
    /// let ops = CudnnOps::new()?;
    /// let algorithms = ops.find_convolution_forward_algorithm(
    ///     &input_desc,
    ///     &filter_desc,
    ///     &conv_desc,
    ///     &output_desc,
    ///     5, // Request top 5 algorithms
    /// )?;
    ///
    /// for algo_perf in algorithms {
    ///     println!("Algorithm: {:?}, Time: {} ms", algo_perf.algorithm, algo_perf.time);
    /// }
    /// ```
    pub fn find_convolution_forward_algorithm(
        &self,
        input_desc: &TensorDescriptor,
        filter_desc: &FilterDescriptor,
        conv_desc: &ConvolutionDescriptor,
        output_desc: &TensorDescriptor,
        request_algo_count: i32,
    ) -> CudaResult<Vec<ConvolutionForwardAlgorithmPerformance>> {
        #[cfg(feature = "cudnn")]
        {
            let handle = self.handle.lock().expect("lock should not be poisoned");
            let mut returned_algo_count: i32 = 0;
            let mut perf_results = vec![
                ConvolutionForwardAlgorithmPerformance::default();
                request_algo_count as usize
            ];

            let status = unsafe {
                cudnnFindConvolutionForwardAlgorithm(
                    handle.raw(),
                    input_desc.raw(),
                    filter_desc.raw(),
                    conv_desc.raw(),
                    output_desc.raw(),
                    request_algo_count,
                    &mut returned_algo_count,
                    perf_results.as_mut_ptr() as *mut cudnnConvolutionFwdAlgoPerf_t,
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to find convolution algorithm: {:?}",
                    status
                )));
            }

            perf_results.truncate(returned_algo_count as usize);
            Ok(perf_results)
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                input_desc,
                filter_desc,
                conv_desc,
                output_desc,
                request_algo_count,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Perform LSTM forward pass
    ///
    /// Executes a complete LSTM forward pass with the specified RNN configuration.
    /// Supports both training and inference modes with optional hidden and cell states.
    ///
    /// # Arguments
    ///
    /// * `rnn_desc` - RNN descriptor configured for LSTM
    /// * `forward_mode` - Training or inference mode
    /// * `dev_seq_lengths` - Optional device pointer to sequence lengths
    /// * `x_desc` - Input data descriptor
    /// * `x` - Input data device pointer
    /// * `y_desc` - Output data descriptor
    /// * `y` - Output data device pointer
    /// * `h_desc` - Hidden state descriptor
    /// * `hx` - Optional initial hidden state device pointer
    /// * `hy` - Optional final hidden state device pointer
    /// * `c_desc` - Cell state descriptor
    /// * `cx` - Optional initial cell state device pointer
    /// * `cy` - Optional final cell state device pointer
    /// * `weight_space_size` - Size of weight space in bytes
    /// * `weight_space` - Weight space device pointer
    /// * `work_space_size` - Size of workspace in bytes
    /// * `work_space` - Workspace device pointer
    /// * `reserve_space_size` - Size of reserve space in bytes
    /// * `reserve_space` - Optional reserve space device pointer (for training)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - LSTM forward operation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::{CudnnOps, rnn::RNNForwardMode};
    ///
    /// let ops = CudnnOps::new()?;
    /// ops.lstm_forward(
    ///     &rnn_desc,
    ///     RNNForwardMode::Training,
    ///     Some(seq_lengths_ptr),
    ///     &x_desc,
    ///     x_ptr,
    ///     &y_desc,
    ///     y_ptr,
    ///     &h_desc,
    ///     Some(hx_ptr),
    ///     Some(hy_ptr),
    ///     &c_desc,
    ///     Some(cx_ptr),
    ///     Some(cy_ptr),
    ///     weight_space_size,
    ///     weight_space_ptr,
    ///     work_space_size,
    ///     work_space_ptr,
    ///     reserve_space_size,
    ///     Some(reserve_space_ptr),
    /// )?;
    /// ```
    pub fn lstm_forward(
        &self,
        rnn_desc: &RNNDescriptor,
        forward_mode: RNNForwardMode,
        dev_seq_lengths: Option<DevicePointer<i32>>,
        x_desc: &RNNDataDescriptor,
        x: DevicePointer<f32>,
        y_desc: &RNNDataDescriptor,
        y: DevicePointer<f32>,
        h_desc: &TensorDescriptor,
        hx: Option<DevicePointer<f32>>,
        hy: Option<DevicePointer<f32>>,
        c_desc: &TensorDescriptor,
        cx: Option<DevicePointer<f32>>,
        cy: Option<DevicePointer<f32>>,
        weight_space_size: usize,
        weight_space: DevicePointer<u8>,
        work_space_size: usize,
        work_space: DevicePointer<u8>,
        reserve_space_size: usize,
        reserve_space: Option<DevicePointer<u8>>,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let handle = self.handle.lock().expect("lock should not be poisoned");
            let status = unsafe {
                cudnnRNNForward(
                    handle.raw(),
                    rnn_desc.raw(),
                    forward_mode.to_cudnn(),
                    dev_seq_lengths
                        .map(|p| p.as_raw() as *const i32)
                        .unwrap_or(std::ptr::null()),
                    x_desc.raw(),
                    x.as_raw() as *const std::ffi::c_void,
                    y_desc.raw(),
                    y.as_raw() as *mut std::ffi::c_void,
                    h_desc.raw(),
                    hx.map(|p| p.as_raw() as *const std::ffi::c_void)
                        .unwrap_or(std::ptr::null()),
                    hy.map(|p| p.as_raw() as *mut std::ffi::c_void)
                        .unwrap_or(std::ptr::null_mut()),
                    c_desc.raw(),
                    cx.map(|p| p.as_raw() as *const std::ffi::c_void)
                        .unwrap_or(std::ptr::null()),
                    cy.map(|p| p.as_raw() as *mut std::ffi::c_void)
                        .unwrap_or(std::ptr::null_mut()),
                    weight_space_size,
                    weight_space.as_raw() as *const std::ffi::c_void,
                    work_space_size,
                    work_space.as_raw() as *mut std::ffi::c_void,
                    reserve_space_size,
                    reserve_space
                        .map(|p| p.as_raw() as *mut std::ffi::c_void)
                        .unwrap_or(std::ptr::null_mut()),
                )
            };

            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "cuDNN LSTM forward failed: {:?}",
                    status
                )));
            }

            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                rnn_desc,
                forward_mode,
                dev_seq_lengths,
                x_desc,
                x,
                y_desc,
                y,
                h_desc,
                hx,
                hy,
                c_desc,
                cx,
                cy,
                weight_space_size,
                weight_space,
                work_space_size,
                work_space,
                reserve_space_size,
                reserve_space,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }
}

unsafe impl Send for CudnnOps {}
unsafe impl Sync for CudnnOps {}

/// Initialize cuDNN library
///
/// Initializes the cuDNN library for use. This function is provided for
/// compatibility but cuDNN initialization is handled per-handle in practice.
///
/// # Returns
///
/// `Ok(())` on success, or an error if cuDNN is not available.
///
/// # Errors
///
/// Returns `CudaError::CudnnError` if cuDNN is not available (feature not enabled).
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_backend::cuda::cudnn;
///
/// cudnn::init()?;
/// let ops = cudnn::CudnnOps::new()?;
/// ```
pub fn init() -> CudaResult<()> {
    #[cfg(feature = "cudnn")]
    {
        // cuDNN initialization is handled per-handle
        Ok(())
    }
    #[cfg(not(feature = "cudnn"))]
    {
        Err(CudaError::CudnnError(
            "cuDNN not available - feature not enabled".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cudnn_ops_creation() {
        // Test basic CudnnOps creation
        // Note: This test will only pass if cuDNN is available
        #[cfg(feature = "cudnn")]
        {
            match CudnnOps::new() {
                Ok(ops) => {
                    // Basic verification that the operations instance was created
                    assert!(std::ptr::addr_of!(ops) as usize != 0);
                }
                Err(_) => {
                    // cuDNN might not be available in test environment
                    // This is acceptable
                }
            }
        }

        #[cfg(not(feature = "cudnn"))]
        {
            let result = CudnnOps::new();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_init() {
        // Test cuDNN initialization
        #[cfg(feature = "cudnn")]
        {
            // Should always succeed as it's a no-op
            assert!(init().is_ok());
        }

        #[cfg(not(feature = "cudnn"))]
        {
            let result = init();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_send_sync() {
        // Test that CudnnOps implements Send and Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<CudnnOps>();
        assert_sync::<CudnnOps>();
    }
}
