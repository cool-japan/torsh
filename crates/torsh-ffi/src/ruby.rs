//! Ruby FFI bindings for ToRSh
//!
//! This module provides Ruby-compatible FFI bindings using the existing C API.
//! Ruby can call C functions directly through the FFI gem.

#![allow(dead_code)]

use crate::c_api::*;
use std::os::raw::{c_char, c_float, c_int, c_void};

/// Ruby-specific wrapper for tensor creation
#[no_mangle]
pub unsafe extern "C" fn ruby_tensor_new(
    data: *const c_void,
    shape: *const usize,
    ndim: usize,
    dtype: TorshDType,
) -> *mut TorshTensor {
    // Delegate to the C API
    torsh_tensor_new(data, shape, ndim, dtype)
}

/// Ruby-specific wrapper for tensor addition
#[no_mangle]
pub unsafe extern "C" fn ruby_tensor_add(
    a: *mut TorshTensor,
    b: *mut TorshTensor,
    output: *mut TorshTensor,
) -> TorshError {
    torsh_tensor_add(a, b, output)
}

/// Ruby-specific wrapper for tensor multiplication
#[no_mangle]
pub unsafe extern "C" fn ruby_tensor_mul(
    a: *mut TorshTensor,
    b: *mut TorshTensor,
    output: *mut TorshTensor,
) -> TorshError {
    torsh_tensor_mul(a, b, output)
}

/// Ruby-specific wrapper for matrix multiplication
#[no_mangle]
pub unsafe extern "C" fn ruby_tensor_matmul(
    a: *mut TorshTensor,
    b: *mut TorshTensor,
    output: *mut TorshTensor,
) -> TorshError {
    torsh_tensor_matmul(a, b, output)
}

/// Ruby-specific wrapper for ReLU activation
#[no_mangle]
pub unsafe extern "C" fn ruby_tensor_relu(
    input: *mut TorshTensor,
    output: *mut TorshTensor,
) -> TorshError {
    torsh_tensor_relu(input, output)
}

/// Ruby-specific wrapper for getting tensor shape
#[no_mangle]
pub unsafe extern "C" fn ruby_tensor_shape(
    tensor: *mut TorshTensor,
    shape: *mut usize,
    ndim: *mut usize,
) -> TorshError {
    torsh_tensor_shape(tensor, shape, ndim)
}

/// Ruby-specific wrapper for getting tensor data
#[no_mangle]
pub unsafe extern "C" fn ruby_tensor_data(tensor: *mut TorshTensor) -> *const c_void {
    torsh_tensor_data(tensor)
}

/// Ruby-specific wrapper for tensor cleanup
#[no_mangle]
pub unsafe extern "C" fn ruby_tensor_free(tensor: *mut TorshTensor) {
    torsh_tensor_free(tensor)
}

/// Ruby-specific wrapper for creating a linear layer
#[no_mangle]
pub unsafe extern "C" fn ruby_linear_new(
    in_features: usize,
    out_features: usize,
    bias: c_int,
) -> *mut TorshModule {
    torsh_linear_new(in_features, out_features, bias != 0)
}

/// Ruby-specific wrapper for linear layer forward pass
#[no_mangle]
pub unsafe extern "C" fn ruby_linear_forward(
    module: *mut TorshModule,
    input: *mut TorshTensor,
) -> *mut TorshTensor {
    // For simplicity, use input tensor as output (in-place operation)
    let error = torsh_linear_forward(module, input, input);
    if error != TorshError::Success {
        return std::ptr::null_mut();
    }

    input
}

/// Ruby-specific wrapper for module cleanup
#[no_mangle]
pub unsafe extern "C" fn ruby_module_free(module: *mut TorshModule) {
    torsh_module_free(module)
}

/// Ruby-specific wrapper for SGD optimizer
#[no_mangle]
pub unsafe extern "C" fn ruby_sgd_new(
    learning_rate: c_float,
    momentum: c_float,
) -> *mut TorshOptimizer {
    torsh_sgd_new(learning_rate, momentum)
}

/// Ruby-specific wrapper for Adam optimizer
#[no_mangle]
pub unsafe extern "C" fn ruby_adam_new(
    learning_rate: c_float,
    beta1: c_float,
    beta2: c_float,
    epsilon: c_float,
) -> *mut TorshOptimizer {
    torsh_adam_new(learning_rate, beta1, beta2, epsilon)
}

/// Ruby-specific wrapper for optimizer step
#[no_mangle]
pub unsafe extern "C" fn ruby_optimizer_step(
    optimizer: *mut TorshOptimizer,
    _parameters: *mut *mut TorshTensor,
    _gradients: *mut *mut TorshTensor,
    _param_count: usize,
) -> TorshError {
    torsh_optimizer_step(optimizer)
}

/// Ruby-specific wrapper for optimizer cleanup
#[no_mangle]
pub unsafe extern "C" fn ruby_optimizer_free(optimizer: *mut TorshOptimizer) {
    torsh_optimizer_free(optimizer)
}

/// Ruby-specific wrapper for getting last error
#[no_mangle]
pub unsafe extern "C" fn ruby_get_last_error() -> *const c_char {
    torsh_get_last_error()
}

/// Ruby-specific wrapper for clearing last error
#[no_mangle]
pub unsafe extern "C" fn ruby_clear_last_error() {
    torsh_clear_last_error()
}

/// Ruby-specific wrapper for version information
#[no_mangle]
pub unsafe extern "C" fn ruby_version() -> *const c_char {
    torsh_version()
}

/// Ruby-specific wrapper for device information
#[no_mangle]
pub unsafe extern "C" fn ruby_cuda_is_available() -> c_int {
    torsh_cuda_is_available()
}

/// Ruby-specific wrapper for CUDA device count
#[no_mangle]
pub unsafe extern "C" fn ruby_cuda_device_count() -> c_int {
    torsh_cuda_device_count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_ruby_tensor_operations() {
        // Test basic tensor creation through Ruby API
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        unsafe {
            let tensor = ruby_tensor_new(
                data.as_ptr() as *const c_void,
                shape.as_ptr(),
                shape.len(),
                TorshDType::F32,
            );

            assert!(!tensor.is_null());

            // Test tensor shape retrieval
            let mut retrieved_shape = vec![0usize; 2];
            let mut ndim = 0;
            let result = ruby_tensor_shape(tensor, retrieved_shape.as_mut_ptr(), &mut ndim);
            assert_eq!(result, TorshError::Success);
            assert_eq!(ndim, 2);
            assert_eq!(retrieved_shape, shape);

            // Clean up
            ruby_tensor_free(tensor);
        }
    }

    #[test]
    fn test_ruby_module_operations() {
        unsafe {
            // Test linear layer creation
            let linear = ruby_linear_new(4, 2, 1);
            assert!(!linear.is_null());

            // Test forward pass with dummy input
            let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
            let input_shape = vec![1, 4];

            let input_tensor = ruby_tensor_new(
                input_data.as_ptr() as *const c_void,
                input_shape.as_ptr(),
                input_shape.len(),
                TorshDType::F32,
            );

            let output = ruby_linear_forward(linear, input_tensor);
            assert!(!output.is_null());

            // Clean up
            ruby_tensor_free(input_tensor);
            ruby_tensor_free(output);
            ruby_module_free(linear);
        }
    }

    #[test]
    fn test_ruby_optimizer_operations() {
        unsafe {
            // Test SGD optimizer creation
            let sgd = ruby_sgd_new(0.01, 0.9);
            assert!(!sgd.is_null());

            // Test Adam optimizer creation
            let adam = ruby_adam_new(0.001, 0.9, 0.999, 1e-8);
            assert!(!adam.is_null());

            // Clean up
            ruby_optimizer_free(sgd);
            ruby_optimizer_free(adam);
        }
    }
}
