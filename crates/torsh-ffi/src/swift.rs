//! Swift C interop bindings for ToRSh
//!
//! This module provides Swift-compatible FFI bindings using Swift's C interop.
//! Swift can call these functions directly through bridging headers.

#![allow(dead_code)]

use crate::c_api::*;
use std::os::raw::{c_char, c_float, c_long, c_void};
use std::ptr;

// Swift-specific types (matching Swift C interop conventions)
pub type SwiftInt = c_long;
pub type SwiftInt32 = i32;
pub type SwiftFloat = f32;
pub type SwiftDouble = f64;
pub type SwiftBool = u8; // Swift Bool is mapped to UInt8
pub type SwiftUnsafeRawPointer = *const c_void;
pub type SwiftUnsafeMutableRawPointer = *mut c_void;

/// Swift-specific wrapper for tensor creation
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_new(
    data: SwiftUnsafeRawPointer,
    shape: *const SwiftInt,
    ndim: SwiftInt,
    dtype: SwiftInt32,
) -> SwiftUnsafeMutableRawPointer {
    // Convert Swift types to C types
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
    tensor as SwiftUnsafeMutableRawPointer
}

/// Swift-specific wrapper for tensor creation from Float array
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_from_float_array(
    data: *const SwiftFloat,
    _data_count: SwiftInt,
    shape: *const SwiftInt,
    ndim: SwiftInt,
) -> SwiftUnsafeMutableRawPointer {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let tensor = torsh_tensor_new(
        data as SwiftUnsafeRawPointer,
        shape_vec.as_ptr(),
        shape_vec.len(),
        TorshDType::F32,
    );
    tensor as SwiftUnsafeMutableRawPointer
}

/// Swift-specific wrapper for tensor creation from Double array
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_from_double_array(
    data: *const SwiftDouble,
    _data_count: SwiftInt,
    shape: *const SwiftInt,
    ndim: SwiftInt,
) -> SwiftUnsafeMutableRawPointer {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let tensor = torsh_tensor_new(
        data as SwiftUnsafeRawPointer,
        shape_vec.as_ptr(),
        shape_vec.len(),
        TorshDType::F64,
    );
    tensor as SwiftUnsafeMutableRawPointer
}

/// Swift-specific wrapper for tensor creation from Int32 array
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_from_int32_array(
    data: *const SwiftInt32,
    _data_count: SwiftInt,
    shape: *const SwiftInt,
    ndim: SwiftInt,
) -> SwiftUnsafeMutableRawPointer {
    let shape_vec: Vec<usize> = std::slice::from_raw_parts(shape, ndim as usize)
        .iter()
        .map(|&x| x as usize)
        .collect();

    let tensor = torsh_tensor_new(
        data as SwiftUnsafeRawPointer,
        shape_vec.as_ptr(),
        shape_vec.len(),
        TorshDType::I32,
    );
    tensor as SwiftUnsafeMutableRawPointer
}

/// Swift-specific wrapper for tensor addition
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_add(
    a: SwiftUnsafeMutableRawPointer,
    b: SwiftUnsafeMutableRawPointer,
) -> SwiftUnsafeMutableRawPointer {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    // For simplicity, we'll reuse tensor_a as output
    let result = torsh_tensor_add(
        tensor_a as *const TorshTensor,
        tensor_b as *const TorshTensor,
        tensor_a,
    );
    match result {
        crate::c_api::TorshError::Success => tensor_a as SwiftUnsafeMutableRawPointer,
        _ => std::ptr::null_mut(),
    }
}

/// Swift-specific wrapper for tensor subtraction
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_subtract(
    a: SwiftUnsafeMutableRawPointer,
    b: SwiftUnsafeMutableRawPointer,
) -> SwiftUnsafeMutableRawPointer {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    // For simplicity, we'll reuse tensor_a as output
    let result = torsh_tensor_sub(
        tensor_a as *const TorshTensor,
        tensor_b as *const TorshTensor,
        tensor_a,
    );
    match result {
        crate::c_api::TorshError::Success => tensor_a as SwiftUnsafeMutableRawPointer,
        _ => std::ptr::null_mut(),
    }
}

/// Swift-specific wrapper for tensor multiplication
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_multiply(
    a: SwiftUnsafeMutableRawPointer,
    b: SwiftUnsafeMutableRawPointer,
) -> SwiftUnsafeMutableRawPointer {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    // For simplicity, we'll reuse tensor_a as output
    let result = torsh_tensor_mul(
        tensor_a as *const TorshTensor,
        tensor_b as *const TorshTensor,
        tensor_a,
    );
    match result {
        crate::c_api::TorshError::Success => tensor_a as SwiftUnsafeMutableRawPointer,
        _ => std::ptr::null_mut(),
    }
}

/// Swift-specific wrapper for matrix multiplication
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_matmul(
    a: SwiftUnsafeMutableRawPointer,
    b: SwiftUnsafeMutableRawPointer,
) -> SwiftUnsafeMutableRawPointer {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    // For simplicity, we'll reuse tensor_a as output
    let result = torsh_tensor_matmul(
        tensor_a as *const TorshTensor,
        tensor_b as *const TorshTensor,
        tensor_a,
    );
    match result {
        crate::c_api::TorshError::Success => tensor_a as SwiftUnsafeMutableRawPointer,
        _ => std::ptr::null_mut(),
    }
}

/// Swift-specific wrapper for ReLU activation
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_relu(
    tensor: SwiftUnsafeMutableRawPointer,
) -> SwiftUnsafeMutableRawPointer {
    let input_tensor = tensor as *mut TorshTensor;
    // For simplicity, we'll reuse input_tensor as output
    let result = torsh_tensor_relu(input_tensor as *const TorshTensor, input_tensor);
    match result {
        crate::c_api::TorshError::Success => input_tensor as SwiftUnsafeMutableRawPointer,
        _ => std::ptr::null_mut(),
    }
}

/// Swift-specific wrapper for getting tensor shape
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_get_shape(
    tensor: SwiftUnsafeMutableRawPointer,
    shape: *mut SwiftInt,
    max_dims: SwiftInt,
    actual_dims: *mut SwiftInt,
) -> SwiftBool {
    let input_tensor = tensor as *mut TorshTensor;
    let mut temp_shape = vec![0usize; max_dims as usize];
    let mut ndim = 0usize;

    let result = torsh_tensor_shape(input_tensor, temp_shape.as_mut_ptr(), &mut ndim);

    if result == TorshError::Success {
        *actual_dims = ndim as SwiftInt;

        // Copy shape data, converting usize to SwiftInt
        for i in 0..std::cmp::min(ndim, max_dims as usize) {
            *shape.add(i) = temp_shape[i] as SwiftInt;
        }

        1 // true in Swift Bool
    } else {
        0 // false in Swift Bool
    }
}

/// Swift-specific wrapper for getting tensor dimensions count
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_get_ndim(tensor: SwiftUnsafeMutableRawPointer) -> SwiftInt {
    let input_tensor = tensor as *mut TorshTensor;
    let mut temp_shape = vec![0usize; 16]; // Max 16 dimensions
    let mut ndim = 0usize;

    let result = torsh_tensor_shape(input_tensor, temp_shape.as_mut_ptr(), &mut ndim);

    if result == TorshError::Success {
        ndim as SwiftInt
    } else {
        -1 // Error indicator
    }
}

/// Swift-specific wrapper for getting tensor element count
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_get_element_count(
    tensor: SwiftUnsafeMutableRawPointer,
) -> SwiftInt {
    let input_tensor = tensor as *mut TorshTensor;
    let mut temp_shape = vec![0usize; 16]; // Max 16 dimensions
    let mut ndim = 0usize;

    let result = torsh_tensor_shape(input_tensor, temp_shape.as_mut_ptr(), &mut ndim);

    if result == TorshError::Success {
        let count: usize = temp_shape[0..ndim].iter().product();
        count as SwiftInt
    } else {
        -1 // Error indicator
    }
}

/// Swift-specific wrapper for getting tensor data
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_get_data(
    tensor: SwiftUnsafeMutableRawPointer,
    data: SwiftUnsafeMutableRawPointer,
    max_elements: SwiftInt,
) -> SwiftBool {
    let input_tensor = tensor as *mut TorshTensor;
    let result = torsh_tensor_data(input_tensor as *const TorshTensor);

    if !result.is_null() {
        // Copy data from result to the output buffer
        let src_data = std::slice::from_raw_parts(result as *const c_float, max_elements as usize);
        let dst_data = std::slice::from_raw_parts_mut(data as *mut c_float, max_elements as usize);
        dst_data.copy_from_slice(src_data);
        1 // true
    } else {
        0 // false
    }
}

/// Swift-specific wrapper for tensor disposal (following Swift memory management patterns)
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_deallocate(tensor: SwiftUnsafeMutableRawPointer) {
    let input_tensor = tensor as *mut TorshTensor;
    torsh_tensor_free(input_tensor);
}

/// Swift-specific wrapper for creating a linear layer
#[no_mangle]
pub unsafe extern "C" fn swift_linear_new(
    in_features: SwiftInt,
    out_features: SwiftInt,
    has_bias: SwiftBool,
) -> SwiftUnsafeMutableRawPointer {
    let module = torsh_linear_new(in_features as usize, out_features as usize, has_bias != 0);
    module as SwiftUnsafeMutableRawPointer
}

/// Swift-specific wrapper for linear layer forward pass
#[no_mangle]
pub unsafe extern "C" fn swift_linear_forward(
    module: SwiftUnsafeMutableRawPointer,
    input: SwiftUnsafeMutableRawPointer,
) -> SwiftUnsafeMutableRawPointer {
    let linear_module = module as *mut TorshModule;
    let input_tensor = input as *mut TorshTensor;

    // For simplicity, we'll reuse input_tensor as output
    let result = torsh_linear_forward(
        linear_module as *const TorshModule,
        input_tensor as *const TorshTensor,
        input_tensor,
    );
    match result {
        crate::c_api::TorshError::Success => input_tensor as SwiftUnsafeMutableRawPointer,
        _ => std::ptr::null_mut(),
    }
}

/// Swift-specific wrapper for module disposal
#[no_mangle]
pub unsafe extern "C" fn swift_module_deallocate(module: SwiftUnsafeMutableRawPointer) {
    let input_module = module as *mut TorshModule;
    torsh_module_free(input_module);
}

/// Swift-specific wrapper for SGD optimizer
#[no_mangle]
pub unsafe extern "C" fn swift_sgd_new(
    learning_rate: SwiftFloat,
    momentum: SwiftFloat,
) -> SwiftUnsafeMutableRawPointer {
    let optimizer = torsh_sgd_new(learning_rate, momentum);
    optimizer as SwiftUnsafeMutableRawPointer
}

/// Swift-specific wrapper for Adam optimizer
#[no_mangle]
pub unsafe extern "C" fn swift_adam_new(
    learning_rate: SwiftFloat,
    beta1: SwiftFloat,
    beta2: SwiftFloat,
    epsilon: SwiftFloat,
) -> SwiftUnsafeMutableRawPointer {
    let optimizer = torsh_adam_new(learning_rate, beta1, beta2, epsilon);
    optimizer as SwiftUnsafeMutableRawPointer
}

/// Swift-specific wrapper for optimizer step
#[no_mangle]
pub unsafe extern "C" fn swift_optimizer_step(
    optimizer: SwiftUnsafeMutableRawPointer,
    parameters: *const SwiftUnsafeMutableRawPointer,
    gradients: *const SwiftUnsafeMutableRawPointer,
    param_count: SwiftInt,
) -> SwiftBool {
    let opt = optimizer as *mut TorshOptimizer;

    // Convert Swift pointer arrays to C pointer arrays
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
        TorshError::Success => 1, // true
        _ => 0,                   // false
    }
}

/// Swift-specific wrapper for optimizer disposal
#[no_mangle]
pub unsafe extern "C" fn swift_optimizer_deallocate(optimizer: SwiftUnsafeMutableRawPointer) {
    let opt = optimizer as *mut TorshOptimizer;
    torsh_optimizer_free(opt);
}

/// Swift-specific wrapper for error handling
#[no_mangle]
pub unsafe extern "C" fn swift_get_last_error() -> *const c_char {
    torsh_get_last_error()
}

/// Swift-specific wrapper for clearing errors
#[no_mangle]
pub unsafe extern "C" fn swift_clear_last_error() {
    torsh_clear_last_error()
}

/// Swift-specific wrapper for version information
#[no_mangle]
pub unsafe extern "C" fn swift_get_version() -> *const c_char {
    torsh_version()
}

/// Swift-specific wrapper for CUDA availability
#[no_mangle]
pub unsafe extern "C" fn swift_cuda_is_available() -> SwiftBool {
    let available = torsh_cuda_is_available();
    if available != 0 {
        1
    } else {
        0
    }
}

/// Swift-specific wrapper for CUDA device count
#[no_mangle]
pub unsafe extern "C" fn swift_cuda_device_count() -> SwiftInt32 {
    torsh_cuda_device_count()
}

/// Swift-specific helper for creating zero tensors
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_zeros(
    shape: *const SwiftInt,
    ndim: SwiftInt,
    dtype: SwiftInt32,
) -> SwiftUnsafeMutableRawPointer {
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
        zeros_data.as_ptr() as SwiftUnsafeRawPointer,
        shape_vec.as_ptr(),
        shape_vec.len(),
        torsh_dtype,
    );
    tensor as SwiftUnsafeMutableRawPointer
}

/// Swift-specific helper for creating ones tensors
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_ones(
    shape: *const SwiftInt,
    ndim: SwiftInt,
    dtype: SwiftInt32,
) -> SwiftUnsafeMutableRawPointer {
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
        ones_data.as_ptr() as SwiftUnsafeRawPointer,
        shape_vec.as_ptr(),
        shape_vec.len(),
        torsh_dtype,
    );
    tensor as SwiftUnsafeMutableRawPointer
}

/// Swift-specific helper for creating identity tensors
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_eye(
    size: SwiftInt,
    dtype: SwiftInt32,
) -> SwiftUnsafeMutableRawPointer {
    let size_usize = size as usize;
    let total_elements = size_usize * size_usize;
    let mut eye_data = vec![0.0f32; total_elements];

    // Set diagonal elements to 1
    for i in 0..size_usize {
        eye_data[i * size_usize + i] = 1.0;
    }

    let shape_vec = vec![size_usize, size_usize];

    let torsh_dtype = match dtype {
        0 => TorshDType::F32,
        1 => TorshDType::F64,
        2 => TorshDType::I32,
        3 => TorshDType::I64,
        4 => TorshDType::U8,
        _ => TorshDType::F32,
    };

    let tensor = torsh_tensor_new(
        eye_data.as_ptr() as SwiftUnsafeRawPointer,
        shape_vec.as_ptr(),
        shape_vec.len(),
        torsh_dtype,
    );
    tensor as SwiftUnsafeMutableRawPointer
}

/// Swift-specific helper for checking if pointer is valid
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_is_valid(tensor: SwiftUnsafeMutableRawPointer) -> SwiftBool {
    if tensor.is_null() {
        0 // false
    } else {
        1 // true
    }
}

/// Swift-specific helper for checking if two tensors have the same shape
#[no_mangle]
pub unsafe extern "C" fn swift_tensor_same_shape(
    a: SwiftUnsafeMutableRawPointer,
    b: SwiftUnsafeMutableRawPointer,
) -> SwiftBool {
    let tensor_a = a as *mut TorshTensor;
    let tensor_b = b as *mut TorshTensor;

    let mut shape_a = vec![0usize; 16];
    let mut shape_b = vec![0usize; 16];
    let mut ndim_a = 0usize;
    let mut ndim_b = 0usize;

    let result_a = torsh_tensor_shape(tensor_a, shape_a.as_mut_ptr(), &mut ndim_a);
    let result_b = torsh_tensor_shape(tensor_b, shape_b.as_mut_ptr(), &mut ndim_b);

    if result_a != TorshError::Success || result_b != TorshError::Success {
        return 0; // false - error getting shapes
    }

    if ndim_a != ndim_b {
        return 0; // false - different number of dimensions
    }

    for i in 0..ndim_a {
        if shape_a[i] != shape_b[i] {
            return 0; // false - different shape
        }
    }

    1 // true - same shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swift_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2i64, 2i64]; // SwiftInt is c_long which is i64 on most systems

        unsafe {
            let tensor = swift_tensor_from_float_array(
                data.as_ptr(),
                data.len() as SwiftInt,
                shape.as_ptr(),
                shape.len() as SwiftInt,
            );

            assert!(!tensor.is_null());

            // Test shape retrieval
            let mut retrieved_shape = vec![0i64; 2];
            let mut actual_dims = 0i64;
            let success =
                swift_tensor_get_shape(tensor, retrieved_shape.as_mut_ptr(), 2, &mut actual_dims);

            assert_eq!(success, 1); // true
            assert_eq!(actual_dims, 2);
            assert_eq!(retrieved_shape, shape);

            // Test element count
            let element_count = swift_tensor_get_element_count(tensor);
            assert_eq!(element_count, 4);

            // Test validity check
            let is_valid = swift_tensor_is_valid(tensor);
            assert_eq!(is_valid, 1); // true

            // Clean up
            swift_tensor_deallocate(tensor);
        }
    }

    #[test]
    fn test_swift_tensor_operations() {
        let data1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let data2 = vec![5.0f32, 6.0, 7.0, 8.0];
        let shape = vec![2i64, 2i64];

        unsafe {
            let tensor1 = swift_tensor_from_float_array(
                data1.as_ptr(),
                data1.len() as SwiftInt,
                shape.as_ptr(),
                shape.len() as SwiftInt,
            );

            let tensor2 = swift_tensor_from_float_array(
                data2.as_ptr(),
                data2.len() as SwiftInt,
                shape.as_ptr(),
                shape.len() as SwiftInt,
            );

            assert!(!tensor1.is_null());
            assert!(!tensor2.is_null());

            // Test same shape check
            let same_shape = swift_tensor_same_shape(tensor1, tensor2);
            assert_eq!(same_shape, 1); // true

            // Test addition
            let result = swift_tensor_add(tensor1, tensor2);
            assert!(!result.is_null());

            // Test ReLU
            let relu_result = swift_tensor_relu(tensor1);
            assert!(!relu_result.is_null());

            // Clean up
            swift_tensor_deallocate(tensor1);
            swift_tensor_deallocate(tensor2);
            swift_tensor_deallocate(result);
            swift_tensor_deallocate(relu_result);
        }
    }

    #[test]
    fn test_swift_creation_helpers() {
        let shape = vec![3i64, 3i64];

        unsafe {
            // Test zeros creation
            let zeros = swift_tensor_zeros(shape.as_ptr(), shape.len() as SwiftInt, 0);
            assert!(!zeros.is_null());

            // Test ones creation
            let ones = swift_tensor_ones(shape.as_ptr(), shape.len() as SwiftInt, 0);
            assert!(!ones.is_null());

            // Test eye creation
            let eye = swift_tensor_eye(3, 0);
            assert!(!eye.is_null());

            // Test element counts
            let zeros_count = swift_tensor_get_element_count(zeros);
            assert_eq!(zeros_count, 9);

            let eye_count = swift_tensor_get_element_count(eye);
            assert_eq!(eye_count, 9);

            // Clean up
            swift_tensor_deallocate(zeros);
            swift_tensor_deallocate(ones);
            swift_tensor_deallocate(eye);
        }
    }

    #[test]
    fn test_swift_linear_layer() {
        unsafe {
            // Test linear layer creation
            let linear = swift_linear_new(4, 2, 1); // has_bias = true
            assert!(!linear.is_null());

            // Test with dummy input
            let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
            let input_shape = vec![1i64, 4i64];

            let input_tensor = swift_tensor_from_float_array(
                input_data.as_ptr(),
                input_data.len() as SwiftInt,
                input_shape.as_ptr(),
                input_shape.len() as SwiftInt,
            );

            let output = swift_linear_forward(linear, input_tensor);
            assert!(!output.is_null());

            // Clean up
            swift_tensor_deallocate(input_tensor);
            swift_tensor_deallocate(output);
            swift_module_deallocate(linear);
        }
    }

    #[test]
    fn test_swift_optimizers() {
        unsafe {
            // Test SGD optimizer creation
            let sgd = swift_sgd_new(0.01, 0.9);
            assert!(!sgd.is_null());

            // Test Adam optimizer creation
            let adam = swift_adam_new(0.001, 0.9, 0.999, 1e-8);
            assert!(!adam.is_null());

            // Clean up
            swift_optimizer_deallocate(sgd);
            swift_optimizer_deallocate(adam);
        }
    }

    #[test]
    fn test_swift_pointer_conversions() {
        // Test that pointer conversions work correctly
        let test_ptr = 0x123456789abcdef0u64 as *mut TorshTensor;
        let handle = test_ptr as SwiftUnsafeMutableRawPointer;
        let converted_back = handle as *mut TorshTensor;
        assert_eq!(test_ptr, converted_back);
    }

    #[test]
    fn test_swift_null_pointer_handling() {
        unsafe {
            let null_ptr = ptr::null_mut();
            let is_valid = swift_tensor_is_valid(null_ptr);
            assert_eq!(is_valid, 0); // false
        }
    }
}
