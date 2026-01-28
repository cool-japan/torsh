//! cuDNN compatibility layer for older cudnn-sys versions
//!
//! This module provides type definitions and stub functions for cuDNN features
//! that are not available in older versions of cudnn-sys (e.g., 0.0.3).
//! These stubs allow the code to compile while making unavailable features
//! return appropriate errors at runtime.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use cudnn_sys::cudnnStatus_t;

// ============================================================================
// Missing type definitions from cudnn-sys 0.0.3
// ============================================================================

/// Data type (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
}

/// Tensor format (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
}

/// Convolution mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnConvolutionMode_t {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
}

/// Activation mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnActivationMode_t {
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU = 1,
    CUDNN_ACTIVATION_TANH = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU = 4,
    CUDNN_ACTIVATION_IDENTITY = 5,
}

/// NaN propagation mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnNanPropagation_t {
    CUDNN_NOT_PROPAGATE_NAN = 0,
    CUDNN_PROPAGATE_NAN = 1,
}

/// Pooling mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnPoolingMode_t {
    CUDNN_POOLING_MAX = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
    CUDNN_POOLING_MAX_DETERMINISTIC = 3,
}

/// RNN input mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNInputMode_t {
    CUDNN_LINEAR_INPUT = 0,
    CUDNN_SKIP_INPUT = 1,
}

/// Direction mode for RNN (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnDirectionMode_t {
    CUDNN_UNIDIRECTIONAL = 0,
    CUDNN_BIDIRECTIONAL = 1,
}

/// RNN mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNMode_t {
    CUDNN_RNN_RELU = 0,
    CUDNN_RNN_TANH = 1,
    CUDNN_LSTM = 2,
    CUDNN_GRU = 3,
}

/// Batch normalization mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnBatchNormMode_t {
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL = 1,
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
}

/// Convolution forward algorithm (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
    CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
}

/// LRN mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnLRNMode_t {
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0,
}

/// RNN Data Layout type (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNDataLayout_t {
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 0,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 1,
    CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2,
}

/// Normalization mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnNormMode_t {
    CUDNN_NORM_PER_ACTIVATION = 0,
    CUDNN_NORM_PER_CHANNEL = 1,
    CUDNN_LAYER_NORM = 2,
}

/// Normalization operations (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnNormOps_t {
    CUDNN_NORM_OPS_NORM = 0,
    CUDNN_NORM_OPS_NORM_ACTIVATION = 1,
    CUDNN_NORM_OPS_NORM_ADD_ACTIVATION = 2,
}

/// Normalization algorithm (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnNormAlgo_t {
    CUDNN_NORM_ALGO_STANDARD = 0,
    CUDNN_NORM_ALGO_PERSIST = 1,
}

/// Forward mode (missing from cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnForwardMode_t {
    CUDNN_FWD_MODE_INFERENCE = 0,
    CUDNN_FWD_MODE_TRAINING = 1,
}

/// RNN Algorithm type (may be missing or incomplete in cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnRNNAlgo_t {
    CUDNN_STANDARD = 0,
    CUDNN_STATIC_PERSISTENT = 1,
    CUDNN_DYNAMIC_PERSISTENT = 2,
}

/// Math type for cuDNN operations (may be missing or incomplete in cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnMathType_t {
    CUDNN_DEFAULT_MATH = 0,
    CUDNN_TENSOR_OP_MATH = 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
}

/// Add mode for cudnnAddTensor (required in cudnn-sys 0.0.3)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cudnnAddMode_t {
    CUDNN_ADD_IMAGE = 0,
    CUDNN_ADD_FEATURE_MAP = 1,
    CUDNN_ADD_SAME_C = 2,
    CUDNN_ADD_FULL_TENSOR = 3,
}

/// RNN algorithm variants for LSTM (extended from cudnn-sys 0.0.3)
/// Note: cudnn-sys 0.0.3 only has CUDNN_RNN_ALGO_STANDARD, etc.
/// We add the LSTM-specific variants used in newer cuDNN versions
pub mod rnn_algo_ext {
    pub const CUDNN_STANDARD_LSTM: u32 = 0;
    pub const CUDNN_PERSIST_STATIC_LSTM: u32 = 1;
    pub const CUDNN_PERSIST_DYNAMIC_LSTM: u32 = 2;
}

// Opaque descriptor types (pointers to incomplete types)
pub type cudnnRNNDescriptor_t = *mut std::ffi::c_void;
pub type cudnnDropoutDescriptor_t = *mut std::ffi::c_void;
pub type cudnnRNNDataDescriptor_t = *mut std::ffi::c_void;
pub type cudnnActivationDescriptor_t = *mut std::ffi::c_void;

// ============================================================================
// Stub function declarations - These return NOT_SUPPORTED status
// ============================================================================

/// Stub: Set convolution math type (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnSetConvolutionMathType(
    _conv_desc: cudnn_sys::cudnnConvolutionDescriptor_t,
    _math_type: cudnnMathType_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Set convolution group count (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnSetConvolutionGroupCount(
    _conv_desc: cudnn_sys::cudnnConvolutionDescriptor_t,
    _group_count: i32,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Set math type on handle (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnSetMathType(
    _handle: cudnn_sys::cudnnHandle_t,
    _math_type: cudnnMathType_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Get math type from handle (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnGetMathType(
    _handle: cudnn_sys::cudnnHandle_t,
    _math_type: *mut cudnnMathType_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

// ============================================================================
// Activation descriptor stubs
// ============================================================================

/// Stub: Create activation descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnCreateActivationDescriptor(
    _activation_desc: *mut cudnnActivationDescriptor_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Set activation descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnSetActivationDescriptor(
    _activation_desc: cudnnActivationDescriptor_t,
    _mode: cudnnActivationMode_t,
    _relu_nan_opt: cudnnNanPropagation_t,
    _coef: f64,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Destroy activation descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnDestroyActivationDescriptor(
    _activation_desc: cudnnActivationDescriptor_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

// ============================================================================
// Batch normalization stubs
// ============================================================================

/// Stub: Batch normalization forward training (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnBatchNormalizationForwardTraining(
    _handle: cudnn_sys::cudnnHandle_t,
    _mode: cudnnBatchNormMode_t,
    _alpha: *const std::ffi::c_void,
    _beta: *const std::ffi::c_void,
    _x_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _x: *const std::ffi::c_void,
    _y_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _y: *mut std::ffi::c_void,
    _bn_scale_bias_mean_var_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _bn_scale: *const std::ffi::c_void,
    _bn_bias: *const std::ffi::c_void,
    _exponential_average_factor: f64,
    _result_running_mean: *mut std::ffi::c_void,
    _result_running_variance: *mut std::ffi::c_void,
    _epsilon: f64,
    _result_save_mean: *mut std::ffi::c_void,
    _result_save_inv_variance: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Batch normalization forward inference (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnBatchNormalizationForwardInference(
    _handle: cudnn_sys::cudnnHandle_t,
    _mode: cudnnBatchNormMode_t,
    _alpha: *const std::ffi::c_void,
    _beta: *const std::ffi::c_void,
    _x_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _x: *const std::ffi::c_void,
    _y_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _y: *mut std::ffi::c_void,
    _bn_scale_bias_mean_var_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _bn_scale: *const std::ffi::c_void,
    _bn_bias: *const std::ffi::c_void,
    _estimated_mean: *const std::ffi::c_void,
    _estimated_variance: *const std::ffi::c_void,
    _epsilon: f64,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Normalization forward inference (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnNormalizationForwardInference(
    _handle: cudnn_sys::cudnnHandle_t,
    _mode: cudnnNormMode_t,
    _norm_ops: cudnnNormOps_t,
    _algo: cudnnNormAlgo_t,
    _alpha: *const std::ffi::c_void,
    _beta: *const std::ffi::c_void,
    _x_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _x: *const std::ffi::c_void,
    _norm_scale_bias_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _norm_scale: *const std::ffi::c_void,
    _norm_bias: *const std::ffi::c_void,
    _epsilon: f64,
    _y_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _y: *mut std::ffi::c_void,
    _norm_mean_var_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _norm_mean: *mut std::ffi::c_void,
    _norm_inv_variance: *mut std::ffi::c_void,
    _activation_desc: cudnnActivationDescriptor_t,
    _workspace: *mut std::ffi::c_void,
    _workspace_size_in_bytes: usize,
    _reserve_space: *mut std::ffi::c_void,
    _reserve_space_size_in_bytes: usize,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

// ============================================================================
// RNN descriptor stubs
// ============================================================================

/// Stub: Create RNN descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnCreateRNNDescriptor(_rnn_desc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Set RNN descriptor v8 (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnSetRNNDescriptor_v8(
    _rnn_desc: cudnnRNNDescriptor_t,
    _algo: cudnnRNNAlgo_t,
    _direction: cudnnDirectionMode_t,
    _rnn_mode: cudnnRNNMode_t,
    _input_mode: cudnnRNNInputMode_t,
    _hidden_size: i32,
    _num_layers: i32,
    _dropout_desc: cudnnDropoutDescriptor_t,
    _aux_flags: u32,
    _math_type: cudnnMathType_t,
    _data_type: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Destroy RNN descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnDestroyRNNDescriptor(_rnn_desc: cudnnRNNDescriptor_t) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: RNN forward (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnRNNForward(
    _handle: cudnn_sys::cudnnHandle_t,
    _rnn_desc: cudnnRNNDescriptor_t,
    _fwd_mode: cudnnForwardMode_t,
    _dev_seq_lengths: *const i32,
    _x_desc: cudnnRNNDataDescriptor_t,
    _x: *const std::ffi::c_void,
    _y_desc: cudnnRNNDataDescriptor_t,
    _y: *mut std::ffi::c_void,
    _h_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _hx: *const std::ffi::c_void,
    _hy: *mut std::ffi::c_void,
    _c_desc: cudnn_sys::cudnnTensorDescriptor_t,
    _cx: *const std::ffi::c_void,
    _cy: *mut std::ffi::c_void,
    _weight_space_size: usize,
    _weight_space: *const std::ffi::c_void,
    _work_space_size: usize,
    _work_space: *mut std::ffi::c_void,
    _reserve_space_size: usize,
    _reserve_space: *mut std::ffi::c_void,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

// ============================================================================
// Dropout descriptor stubs
// ============================================================================

/// Stub: Create dropout descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnCreateDropoutDescriptor(
    _dropout_desc: *mut cudnnDropoutDescriptor_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Get dropout states size (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnDropoutGetStatesSize(
    _handle: cudnn_sys::cudnnHandle_t,
    _size_in_bytes: *mut usize,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Set dropout descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnSetDropoutDescriptor(
    _dropout_desc: cudnnDropoutDescriptor_t,
    _handle: cudnn_sys::cudnnHandle_t,
    _dropout: f32,
    _states: *mut std::ffi::c_void,
    _states_size_in_bytes: usize,
    _seed: u64,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Destroy dropout descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnDestroyDropoutDescriptor(
    _dropout_desc: cudnnDropoutDescriptor_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

// ============================================================================
// RNN data descriptor stubs
// ============================================================================

/// Stub: Create RNN data descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnCreateRNNDataDescriptor(
    _rnn_data_desc: *mut cudnnRNNDataDescriptor_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Set RNN data descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnSetRNNDataDescriptor(
    _rnn_data_desc: cudnnRNNDataDescriptor_t,
    _data_type: cudnnDataType_t,
    _layout: cudnnRNNDataLayout_t,
    _max_seq_length: i32,
    _batch_size: i32,
    _vector_size: i32,
    _seq_length_array: *const i32,
    _padding_fill: *const f32,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

/// Stub: Destroy RNN data descriptor (not available in cudnn-sys 0.0.3)
#[inline]
pub unsafe fn cudnnDestroyRNNDataDescriptor(
    _rnn_data_desc: cudnnRNNDataDescriptor_t,
) -> cudnnStatus_t {
    cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_functions_return_not_supported() {
        unsafe {
            // Test that stub functions return NOT_SUPPORTED
            let status = cudnnSetConvolutionMathType(
                std::ptr::null_mut(),
                cudnnMathType_t::CUDNN_DEFAULT_MATH,
            );
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED);

            let status = cudnnSetConvolutionGroupCount(std::ptr::null_mut(), 1);
            assert_eq!(status, cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED);
        }
    }

    #[test]
    fn test_enum_values() {
        // Verify enum values match expected cuDNN values
        assert_eq!(
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED as i32,
            0
        );
        assert_eq!(
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED as i32,
            1
        );
        assert_eq!(
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED as i32,
            2
        );
    }
}
