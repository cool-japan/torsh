//! C API for ToRSh
//!
//! This module provides a C-compatible API for integrating ToRSh with
//! other languages and systems that can call C functions.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_float, c_void};
use std::ptr;
use torsh_core::DType;
use crate::error::{FfiError, FfiResult};

/// Opaque handle for Tensor objects
#[repr(C)]
pub struct TorshTensor {
    _private: [u8; 0],
}

/// Opaque handle for Module objects  
#[repr(C)]
pub struct TorshModule {
    _private: [u8; 0],
}

/// Opaque handle for Optimizer objects
#[repr(C)]
pub struct TorshOptimizer {
    _private: [u8; 0],
}

/// Data types for C API
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum TorshDType {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I64 = 3,
    U8 = 4,
}

/// Error codes for C API
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum TorshError {
    Success = 0,
    InvalidArgument = 1,
    OutOfMemory = 2,
    ShapeMismatch = 3,
    TypeMismatch = 4,
    RuntimeError = 5,
    NotImplemented = 6,
}

impl From<DType> for TorshDType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F32 => TorshDType::F32,
            DType::F64 => TorshDType::F64,
            DType::I32 => TorshDType::I32,
            DType::I64 => TorshDType::I64,
            DType::U8 => TorshDType::U8,
            _ => TorshDType::F32, // Default fallback
        }
    }
}

impl From<TorshDType> for DType {
    fn from(dtype: TorshDType) -> Self {
        match dtype {
            TorshDType::F32 => DType::F32,
            TorshDType::F64 => DType::F64,
            TorshDType::I32 => DType::I32,
            TorshDType::I64 => DType::I64,
            TorshDType::U8 => DType::U8,
        }
    }
}

/// Convert Rust Result to C error code
fn to_c_error<T>(result: FfiResult<T>) -> (TorshError, Option<T>) {
    match result {
        Ok(value) => (TorshError::Success, Some(value)),
        Err(err) => {
            let error_code = match err {
                FfiError::InvalidConversion { .. } => TorshError::InvalidArgument,
                FfiError::ShapeMismatch { .. } => TorshError::ShapeMismatch,
                FfiError::DTypeMismatch { .. } => TorshError::TypeMismatch,
                FfiError::AllocationFailed { .. } => TorshError::OutOfMemory,
                FfiError::UnsupportedOperation { .. } => TorshError::NotImplemented,
                _ => TorshError::RuntimeError,
            };
            (error_code, None)
        }
    }
}

// Tensor operations
extern "C" {
    /// Create a new tensor from data
    pub fn torsh_tensor_new(
        data: *const c_void,
        shape: *const usize,
        ndim: usize,
        dtype: TorshDType,
    ) -> *mut TorshTensor;
    
    /// Free a tensor
    pub fn torsh_tensor_free(tensor: *mut TorshTensor);
    
    /// Get tensor data
    pub fn torsh_tensor_data(tensor: *const TorshTensor) -> *const c_void;
    
    /// Get tensor shape
    pub fn torsh_tensor_shape(
        tensor: *const TorshTensor,
        shape: *mut usize,
        ndim: *mut usize,
    ) -> TorshError;
    
    /// Get tensor data type
    pub fn torsh_tensor_dtype(tensor: *const TorshTensor) -> TorshDType;
    
    /// Add two tensors
    pub fn torsh_tensor_add(
        a: *const TorshTensor,
        b: *const TorshTensor,
        out: *mut TorshTensor,
    ) -> TorshError;
    
    /// Multiply two tensors
    pub fn torsh_tensor_mul(
        a: *const TorshTensor,
        b: *const TorshTensor,
        out: *mut TorshTensor,
    ) -> TorshError;
    
    /// Matrix multiplication
    pub fn torsh_tensor_matmul(
        a: *const TorshTensor,
        b: *const TorshTensor,
        out: *mut TorshTensor,
    ) -> TorshError;
    
    /// Apply ReLU activation
    pub fn torsh_tensor_relu(
        input: *const TorshTensor,
        out: *mut TorshTensor,
    ) -> TorshError;
}

// Neural network operations
extern "C" {
    /// Create a linear layer
    pub fn torsh_linear_new(
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> *mut TorshModule;
    
    /// Forward pass through linear layer
    pub fn torsh_linear_forward(
        module: *const TorshModule,
        input: *const TorshTensor,
        output: *mut TorshTensor,
    ) -> TorshError;
    
    /// Free a module
    pub fn torsh_module_free(module: *mut TorshModule);
}

// Optimizer operations
extern "C" {
    /// Create SGD optimizer
    pub fn torsh_sgd_new(
        learning_rate: c_float,
        momentum: c_float,
    ) -> *mut TorshOptimizer;
    
    /// Create Adam optimizer
    pub fn torsh_adam_new(
        learning_rate: c_float,
        beta1: c_float,
        beta2: c_float,
        epsilon: c_float,
    ) -> *mut TorshOptimizer;
    
    /// Optimizer step
    pub fn torsh_optimizer_step(optimizer: *mut TorshOptimizer) -> TorshError;
    
    /// Zero gradients
    pub fn torsh_optimizer_zero_grad(optimizer: *mut TorshOptimizer) -> TorshError;
    
    /// Free optimizer
    pub fn torsh_optimizer_free(optimizer: *mut TorshOptimizer);
}

// Utility functions
extern "C" {
    /// Get version string
    pub fn torsh_version() -> *const c_char;
    
    /// Initialize ToRSh library
    pub fn torsh_init() -> TorshError;
    
    /// Cleanup ToRSh library
    pub fn torsh_cleanup() -> TorshError;
    
    /// Set device
    pub fn torsh_set_device(device_type: c_int, device_id: c_int) -> TorshError;
    
    /// Get last error message
    pub fn torsh_get_last_error() -> *const c_char;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dtype_conversion() {
        let rust_dtype = DType::F32;
        let c_dtype = TorshDType::from(rust_dtype);
        let back_to_rust = DType::from(c_dtype);
        
        assert_eq!(rust_dtype, back_to_rust);
    }
    
    #[test]
    fn test_error_conversion() {
        let error = FfiError::ShapeMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
        };
        
        let (c_error, _) = to_c_error::<()>(Err(error));
        assert!(matches!(c_error, TorshError::ShapeMismatch));
    }
}