//! Go CGO bindings for ToRSh
//!
//! This module provides Go-compatible FFI bindings using CGO.
//! Go can call these functions through import "C" and cgo directives.

#![allow(dead_code)]

use crate::c_api::*;
use std::os::raw::{c_char, c_float, c_long, c_void};

// Go-specific types
pub type GoInt = c_long;
pub type GoInt32 = i32;
pub type GoFloat32 = f32;
pub type GoFloat64 = f64;
pub type GoUintptr = usize;

/// Go-specific wrapper for tensor creation
/// //export go_tensor_new
#[no_mangle]
pub unsafe extern "C" fn go_tensor_new(
    data: *const c_void,
    shape: *const GoInt,
    ndim: GoInt,
    dtype: GoInt32,
) -> GoUintptr {
    // Convert Go types to C types
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let torsh_dtype = match dtype {
        0 => TorshDType::F32,
        1 => TorshDType::F64,
        2 => TorshDType::I32,
        3 => TorshDType::I64,
        4 => TorshDType::U8,
        _ => TorshDType::F32,
    };

    let tensor = torsh_tensor_new(data, shape_vec.as_ptr(), shape_vec.len(), torsh_dtype);
    tensor as GoUintptr
}

/// Go-specific wrapper for tensor creation from float slice
/// //export go_tensor_from_float32_slice
#[no_mangle]
pub unsafe extern "C" fn go_tensor_from_float32_slice(
    data: *const GoFloat32,
    _data_len: GoInt,
    shape: *const GoInt,
    ndim: GoInt,
) -> GoUintptr {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let tensor = torsh_tensor_new(
        data as *const c_void,
        shape_vec.as_ptr(),
        shape_vec.len(),
        TorshDType::F32,
    );
    tensor as GoUintptr
}

/// Go-specific wrapper for tensor creation from float64 slice
/// //export go_tensor_from_float64_slice
#[no_mangle]
pub unsafe extern "C" fn go_tensor_from_float64_slice(
    data: *const GoFloat64,
    _data_len: GoInt,
    shape: *const GoInt,
    ndim: GoInt,
) -> GoUintptr {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let tensor = torsh_tensor_new(
        data as *const c_void,
        shape_vec.as_ptr(),
        shape_vec.len(),
        TorshDType::F64,
    );
    tensor as GoUintptr
}

/// Go-specific wrapper for tensor creation from int slice
/// //export go_tensor_from_int_slice
#[no_mangle]
pub unsafe extern "C" fn go_tensor_from_int_slice(
    data: *const GoInt32,
    _data_len: GoInt,
    shape: *const GoInt,
    ndim: GoInt,
) -> GoUintptr {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let tensor = torsh_tensor_new(
        data as *const c_void,
        shape_vec.as_ptr(),
        shape_vec.len(),
        TorshDType::I32,
    );
    tensor as GoUintptr
}

/// Go-specific wrapper for tensor addition
/// //export go_tensor_add
#[no_mangle]
pub unsafe extern "C" fn go_tensor_add(a: GoUintptr, b: GoUintptr) -> GoUintptr {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    // For simplicity, we'll reuse tensor_a as output
    // In a real implementation, you might want to create a new tensor
    let result = torsh_tensor_add(
        tensor_a as *const TorshTensor,
        tensor_b as *const TorshTensor,
        tensor_a,
    );
    match result {
        crate::c_api::TorshError::Success => tensor_a as GoUintptr,
        _ => 0 as GoUintptr,
    }
}

/// Go-specific wrapper for tensor subtraction
/// //export go_tensor_sub
#[no_mangle]
pub unsafe extern "C" fn go_tensor_sub(a: GoUintptr, b: GoUintptr) -> GoUintptr {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    // For simplicity, we'll reuse tensor_a as output
    let result = torsh_tensor_sub(
        tensor_a as *const TorshTensor,
        tensor_b as *const TorshTensor,
        tensor_a,
    );
    match result {
        crate::c_api::TorshError::Success => tensor_a as GoUintptr,
        _ => 0 as GoUintptr,
    }
}

/// Go-specific wrapper for tensor multiplication
/// //export go_tensor_mul
#[no_mangle]
pub unsafe extern "C" fn go_tensor_mul(a: GoUintptr, b: GoUintptr) -> GoUintptr {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    // For simplicity, we'll reuse tensor_a as output
    let result = torsh_tensor_mul(
        tensor_a as *const TorshTensor,
        tensor_b as *const TorshTensor,
        tensor_a,
    );
    match result {
        crate::c_api::TorshError::Success => tensor_a as GoUintptr,
        _ => 0 as GoUintptr,
    }
}

/// Go-specific wrapper for matrix multiplication
/// //export go_tensor_matmul
#[no_mangle]
pub unsafe extern "C" fn go_tensor_matmul(a: GoUintptr, b: GoUintptr) -> GoUintptr {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    // For simplicity, we'll reuse tensor_a as output
    let result = torsh_tensor_matmul(
        tensor_a as *const TorshTensor,
        tensor_b as *const TorshTensor,
        tensor_a,
    );
    match result {
        crate::c_api::TorshError::Success => tensor_a as GoUintptr,
        _ => 0 as GoUintptr,
    }
}

/// Go-specific wrapper for ReLU activation
/// //export go_tensor_relu
#[no_mangle]
pub unsafe extern "C" fn go_tensor_relu(tensor: GoUintptr) -> GoUintptr {
    let input_tensor = tensor as *mut TorshTensor;
    // For simplicity, we'll reuse input_tensor as output
    let result = torsh_tensor_relu(input_tensor as *const TorshTensor, input_tensor);
    match result {
        crate::c_api::TorshError::Success => input_tensor as GoUintptr,
        _ => 0 as GoUintptr,
    }
}

/// Go-specific wrapper for getting tensor shape
/// //export go_tensor_get_shape
#[no_mangle]
pub unsafe extern "C" fn go_tensor_get_shape(
    tensor: GoUintptr,
    shape: *mut GoInt,
    max_dims: GoInt,
    actual_dims: *mut GoInt,
) -> GoInt32 {
    let input_tensor = tensor as *mut TorshTensor;
    let mut temp_shape = vec![0usize; max_dims as usize];
    let mut ndim = 0usize;

    let result = torsh_tensor_shape(input_tensor, temp_shape.as_mut_ptr(), &mut ndim);

    if result == TorshError::Success {
        *actual_dims = ndim as GoInt;

        // Copy shape data, converting usize to GoInt
        for i in 0..std::cmp::min(ndim, max_dims as usize) {
            *shape.add(i) = temp_shape[i] as GoInt;
        }

        0 // Success
    } else {
        1 // Error
    }
}

/// Go-specific wrapper for getting tensor data
/// //export go_tensor_get_data
#[no_mangle]
pub unsafe extern "C" fn go_tensor_get_data(
    tensor: GoUintptr,
    data: *mut c_void,
    max_elements: GoInt,
) -> GoInt32 {
    let input_tensor = tensor as *mut TorshTensor;
    let result = torsh_tensor_data(input_tensor as *const TorshTensor);

    if !result.is_null() {
        // Copy data from result to the output buffer
        let src_data = std::slice::from_raw_parts(result as *const c_float, max_elements as usize);
        let dst_data = std::slice::from_raw_parts_mut(data as *mut c_float, max_elements as usize);
        dst_data.copy_from_slice(src_data);
        0 // Success
    } else {
        1 // Error
    }
}

/// Go-specific wrapper for tensor disposal
/// //export go_tensor_free
#[no_mangle]
pub unsafe extern "C" fn go_tensor_free(tensor: GoUintptr) {
    let input_tensor = tensor as *mut TorshTensor;
    torsh_tensor_free(input_tensor);
}

/// Go-specific wrapper for creating a linear layer
/// //export go_linear_new
#[no_mangle]
pub unsafe extern "C" fn go_linear_new(
    in_features: GoInt,
    out_features: GoInt,
    bias: GoInt32, // 0 = false, 1 = true
) -> GoUintptr {
    let module = torsh_linear_new(in_features as usize, out_features as usize, bias != 0);
    module as GoUintptr
}

/// Go-specific wrapper for linear layer forward pass
/// //export go_linear_forward
#[no_mangle]
pub unsafe extern "C" fn go_linear_forward(module: GoUintptr, input: GoUintptr) -> GoUintptr {
    let linear_module = module as *mut TorshModule;
    let input_tensor = input as *mut TorshTensor;

    // For simplicity, we'll reuse input_tensor as output
    let result = torsh_linear_forward(
        linear_module as *const TorshModule,
        input_tensor as *const TorshTensor,
        input_tensor,
    );
    match result {
        crate::c_api::TorshError::Success => input_tensor as GoUintptr,
        _ => 0 as GoUintptr,
    }
}

/// Go-specific wrapper for module disposal
/// //export go_module_free
#[no_mangle]
pub unsafe extern "C" fn go_module_free(module: GoUintptr) {
    let input_module = module as *mut TorshModule;
    torsh_module_free(input_module);
}

/// Go-specific wrapper for SGD optimizer
/// //export go_sgd_new
#[no_mangle]
pub unsafe extern "C" fn go_sgd_new(learning_rate: GoFloat32, momentum: GoFloat32) -> GoUintptr {
    let optimizer = torsh_sgd_new(learning_rate, momentum);
    optimizer as GoUintptr
}

/// Go-specific wrapper for Adam optimizer
/// //export go_adam_new
#[no_mangle]
pub unsafe extern "C" fn go_adam_new(
    learning_rate: GoFloat32,
    beta1: GoFloat32,
    beta2: GoFloat32,
    epsilon: GoFloat32,
) -> GoUintptr {
    let optimizer = torsh_adam_new(learning_rate, beta1, beta2, epsilon);
    optimizer as GoUintptr
}

/// Go-specific wrapper for optimizer step
/// //export go_optimizer_step
#[no_mangle]
pub unsafe extern "C" fn go_optimizer_step(
    optimizer: GoUintptr,
    parameters: *const GoUintptr,
    gradients: *const GoUintptr,
    param_count: GoInt,
) -> GoInt32 {
    let opt = optimizer as *mut TorshOptimizer;

    // Convert Go uintptr arrays to C pointer arrays
    let _param_vec: Vec<*mut TorshTensor> =
        std::slice::from_raw_parts(parameters, param_count as usize)
            .iter()
            .map(|&ptr| ptr as *mut TorshTensor)
            .collect();

    let _grad_vec: Vec<*mut TorshTensor> =
        std::slice::from_raw_parts(gradients, param_count as usize)
            .iter()
            .map(|&ptr| ptr as *mut TorshTensor)
            .collect();

    // Note: C API only takes optimizer, additional parameters are ignored for now
    let result = torsh_optimizer_step(opt);

    match result {
        TorshError::Success => 0,
        _ => 1,
    }
}

/// Go-specific wrapper for optimizer disposal
/// //export go_optimizer_free
#[no_mangle]
pub unsafe extern "C" fn go_optimizer_free(optimizer: GoUintptr) {
    let opt = optimizer as *mut TorshOptimizer;
    torsh_optimizer_free(opt);
}

/// Go-specific wrapper for error handling
/// //export go_get_last_error
#[no_mangle]
pub unsafe extern "C" fn go_get_last_error() -> *const c_char {
    torsh_get_last_error()
}

/// Go-specific wrapper for clearing errors
/// //export go_clear_last_error
#[no_mangle]
pub unsafe extern "C" fn go_clear_last_error() {
    torsh_clear_last_error()
}

/// Go-specific wrapper for version information
/// //export go_get_version
#[no_mangle]
pub unsafe extern "C" fn go_get_version() -> *const c_char {
    torsh_version()
}

/// Go-specific wrapper for CUDA availability
/// //export go_cuda_is_available
#[no_mangle]
pub unsafe extern "C" fn go_cuda_is_available() -> GoInt32 {
    torsh_cuda_is_available()
}

/// Go-specific wrapper for CUDA device count
/// //export go_cuda_device_count
#[no_mangle]
pub unsafe extern "C" fn go_cuda_device_count() -> GoInt32 {
    torsh_cuda_device_count()
}

/// Go-specific helper for creating zero tensors
/// //export go_tensor_zeros
#[no_mangle]
pub unsafe extern "C" fn go_tensor_zeros(
    shape: *const GoInt,
    ndim: GoInt,
    dtype: GoInt32,
) -> GoUintptr {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let total_elements: usize = shape_vec.iter().product();
    let zeros_data = vec![0.0f32; total_elements];

    let torsh_dtype = match dtype {
        0 => TorshDType::F32,
        1 => TorshDType::F64,
        2 => TorshDType::I32,
        3 => TorshDType::I64,
        4 => TorshDType::U8,
        _ => TorshDType::F32,
    };

    let tensor = torsh_tensor_new(
        zeros_data.as_ptr() as *const c_void,
        shape_vec.as_ptr(),
        shape_vec.len(),
        torsh_dtype,
    );
    tensor as GoUintptr
}

/// Go-specific helper for creating ones tensors
/// //export go_tensor_ones
#[no_mangle]
pub unsafe extern "C" fn go_tensor_ones(
    shape: *const GoInt,
    ndim: GoInt,
    dtype: GoInt32,
) -> GoUintptr {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let total_elements: usize = shape_vec.iter().product();
    let ones_data = vec![1.0f32; total_elements];

    let torsh_dtype = match dtype {
        0 => TorshDType::F32,
        1 => TorshDType::F64,
        2 => TorshDType::I32,
        3 => TorshDType::I64,
        4 => TorshDType::U8,
        _ => TorshDType::F32,
    };

    let tensor = torsh_tensor_new(
        ones_data.as_ptr() as *const c_void,
        shape_vec.as_ptr(),
        shape_vec.len(),
        torsh_dtype,
    );
    tensor as GoUintptr
}

/// Go-specific helper for creating random tensors
/// //export go_tensor_randn
#[no_mangle]
pub unsafe extern "C" fn go_tensor_randn(
    shape: *const GoInt,
    ndim: GoInt,
    dtype: GoInt32,
) -> GoUintptr {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let total_elements: usize = shape_vec.iter().product();

    // Simple random number generation (in practice, use a proper RNG)
    let mut random_data = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        random_data.push((i as f32 * 0.1) % 1.0); // Placeholder random values
    }

    let torsh_dtype = match dtype {
        0 => TorshDType::F32,
        1 => TorshDType::F64,
        2 => TorshDType::I32,
        3 => TorshDType::I64,
        4 => TorshDType::U8,
        _ => TorshDType::F32,
    };

    let tensor = torsh_tensor_new(
        random_data.as_ptr() as *const c_void,
        shape_vec.as_ptr(),
        shape_vec.len(),
        torsh_dtype,
    );
    tensor as GoUintptr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_go_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2i64, 2i64]; // GoInt is c_long which is i64 on most systems

        unsafe {
            let tensor = go_tensor_from_float32_slice(
                data.as_ptr(),
                data.len() as GoInt,
                shape.as_ptr(),
                shape.len() as GoInt,
            );

            assert_ne!(tensor, 0);

            // Test shape retrieval
            let mut retrieved_shape = vec![0i64; 2];
            let mut actual_dims = 0i64;
            let result =
                go_tensor_get_shape(tensor, retrieved_shape.as_mut_ptr(), 2, &mut actual_dims);

            assert_eq!(result, 0); // Success
            assert_eq!(actual_dims, 2);
            assert_eq!(retrieved_shape, shape);

            // Clean up
            go_tensor_free(tensor);
        }
    }

    #[test]
    fn test_go_tensor_operations() {
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![5.0f32, 6.0, 7.0, 8.0];
        let shape = vec![2i64, 2i64];

        unsafe {
            let tensor1 = go_tensor_from_float32_slice(
                data1.as_ptr(),
                data1.len() as GoInt,
                shape.as_ptr(),
                shape.len() as GoInt,
            );

            let tensor2 = go_tensor_from_float32_slice(
                data2.as_ptr(),
                data2.len() as GoInt,
                shape.as_ptr(),
                shape.len() as GoInt,
            );

            assert_ne!(tensor1, 0);
            assert_ne!(tensor2, 0);

            // Test addition
            let result = go_tensor_add(tensor1, tensor2);
            assert_ne!(result, 0);

            // Test multiplication
            let mul_result = go_tensor_mul(tensor1, tensor2);
            assert_ne!(mul_result, 0);

            // Clean up
            go_tensor_free(tensor1);
            go_tensor_free(tensor2);
            go_tensor_free(result);
            go_tensor_free(mul_result);
        }
    }

    #[test]
    fn test_go_zeros_ones_randn() {
        let shape = vec![3i64, 3i64];

        unsafe {
            // Test zeros creation
            let zeros = go_tensor_zeros(shape.as_ptr(), shape.len() as GoInt, 0);
            assert_ne!(zeros, 0);

            // Test ones creation
            let ones = go_tensor_ones(shape.as_ptr(), shape.len() as GoInt, 0);
            assert_ne!(ones, 0);

            // Test randn creation
            let randn = go_tensor_randn(shape.as_ptr(), shape.len() as GoInt, 0);
            assert_ne!(randn, 0);

            // Clean up
            go_tensor_free(zeros);
            go_tensor_free(ones);
            go_tensor_free(randn);
        }
    }

    #[test]
    fn test_go_linear_layer() {
        unsafe {
            // Test linear layer creation
            let linear = go_linear_new(4, 2, 1);
            assert_ne!(linear, 0);

            // Test with dummy input
            let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
            let input_shape = vec![1i64, 4i64];

            let input_tensor = go_tensor_from_float32_slice(
                input_data.as_ptr(),
                input_data.len() as GoInt,
                input_shape.as_ptr(),
                input_shape.len() as GoInt,
            );

            let output = go_linear_forward(linear, input_tensor);
            assert_ne!(output, 0);

            // Clean up
            go_tensor_free(input_tensor);
            go_tensor_free(output);
            go_module_free(linear);
        }
    }

    #[test]
    fn test_go_optimizers() {
        unsafe {
            // Test SGD optimizer creation
            let sgd = go_sgd_new(0.01, 0.9);
            assert_ne!(sgd, 0);

            // Test Adam optimizer creation
            let adam = go_adam_new(0.001, 0.9, 0.999, 1e-8);
            assert_ne!(adam, 0);

            // Clean up
            go_optimizer_free(sgd);
            go_optimizer_free(adam);
        }
    }

    #[test]
    fn test_go_uintptr_conversions() {
        // Test that uintptr conversions work correctly
        let test_ptr = 0x123456789abcdef0u64 as *mut TorshTensor;
        let handle = test_ptr as GoUintptr;
        let converted_back = handle as *mut TorshTensor;
        assert_eq!(test_ptr, converted_back);
    }
}
