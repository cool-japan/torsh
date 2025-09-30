//! Julia Language Bindings for ToRSh
//!
//! This module provides Julia language bindings using Julia's C API for high-performance scientific computing.
//! Julia bindings enable seamless integration with Julia's high-performance numerical computing environment.

#![allow(dead_code)]

use crate::c_api::*;
use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;

/// Julia wrapper for tensor operations
/// Julia uses jl_value_t* for all values (similar to PyObject*)
pub struct JuliaTensor {
    handle: *mut TorshTensor,
    jl_value: *mut c_void, // jl_value_t* pointer for Julia integration
    gc_tracked: bool,      // Whether this object is tracked by Julia's GC
}

impl JuliaTensor {
    pub fn new(handle: *mut TorshTensor) -> Self {
        Self {
            handle,
            jl_value: ptr::null_mut(),
            gc_tracked: false,
        }
    }

    pub fn handle(&self) -> *mut TorshTensor {
        self.handle
    }

    pub fn set_jl_value(&mut self, jl_value: *mut c_void) {
        self.jl_value = jl_value;
    }

    pub fn jl_value(&self) -> *mut c_void {
        self.jl_value
    }

    pub fn set_gc_tracked(&mut self, tracked: bool) {
        self.gc_tracked = tracked;
    }

    pub fn is_gc_tracked(&self) -> bool {
        self.gc_tracked
    }
}

/// Julia wrapper for neural network modules
pub struct JuliaModule {
    handle: *mut TorshModule,
    jl_value: *mut c_void,
    gc_tracked: bool,
}

impl JuliaModule {
    pub fn new(handle: *mut TorshModule) -> Self {
        Self {
            handle,
            jl_value: ptr::null_mut(),
            gc_tracked: false,
        }
    }

    pub fn handle(&self) -> *mut TorshModule {
        self.handle
    }

    pub fn set_jl_value(&mut self, jl_value: *mut c_void) {
        self.jl_value = jl_value;
    }

    pub fn jl_value(&self) -> *mut c_void {
        self.jl_value
    }
}

/// Julia wrapper for optimizers
pub struct JuliaOptimizer {
    handle: *mut TorshOptimizer,
    jl_value: *mut c_void,
    gc_tracked: bool,
}

impl JuliaOptimizer {
    pub fn new(handle: *mut TorshOptimizer) -> Self {
        Self {
            handle,
            jl_value: ptr::null_mut(),
            gc_tracked: false,
        }
    }

    pub fn handle(&self) -> *mut TorshOptimizer {
        self.handle
    }

    pub fn set_jl_value(&mut self, jl_value: *mut c_void) {
        self.jl_value = jl_value;
    }

    pub fn jl_value(&self) -> *mut c_void {
        self.jl_value
    }
}

/// Convert Julia Array{Float64} to ToRSh tensor
#[no_mangle]
pub unsafe extern "C" fn jl_array_float64_to_torsh_tensor(
    data: *const c_double,
    length: c_int,
    shape_data: *const c_int,
    shape_len: c_int,
) -> *mut JuliaTensor {
    if data.is_null() || shape_data.is_null() {
        return ptr::null_mut();
    }

    let data_slice = std::slice::from_raw_parts(data, length as usize);
    let shape_slice = std::slice::from_raw_parts(shape_data, shape_len as usize);

    // Convert f64 to f32 for ToRSh
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();

    let tensor_handle = torsh_tensor_from_data(
        f32_data.as_ptr(),
        f32_data.len(),
        shape.as_ptr(),
        shape.len(),
    );

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(tensor_handle)))
}

/// Convert Julia Array{Float32} to ToRSh tensor (native precision)
#[no_mangle]
pub unsafe extern "C" fn jl_array_float32_to_torsh_tensor(
    data: *const f32,
    length: c_int,
    shape_data: *const c_int,
    shape_len: c_int,
) -> *mut JuliaTensor {
    if data.is_null() || shape_data.is_null() {
        return ptr::null_mut();
    }

    let shape_slice = std::slice::from_raw_parts(shape_data, shape_len as usize);
    let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();

    let tensor_handle = torsh_tensor_from_data(data, length as usize, shape.as_ptr(), shape.len());

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(tensor_handle)))
}

/// Convert Julia Array{Int32} to ToRSh tensor
#[no_mangle]
pub unsafe extern "C" fn jl_array_int32_to_torsh_tensor(
    data: *const c_int,
    length: c_int,
    shape_data: *const c_int,
    shape_len: c_int,
) -> *mut JuliaTensor {
    if data.is_null() || shape_data.is_null() {
        return ptr::null_mut();
    }

    let data_slice = std::slice::from_raw_parts(data, length as usize);
    let shape_slice = std::slice::from_raw_parts(shape_data, shape_len as usize);

    // Convert i32 to f32 for ToRSh
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();

    let tensor_handle = torsh_tensor_from_data(
        f32_data.as_ptr(),
        f32_data.len(),
        shape.as_ptr(),
        shape.len(),
    );

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(tensor_handle)))
}

/// Convert ToRSh tensor to Julia Array{Float64}
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_to_jl_array_float64(
    jl_tensor: *const JuliaTensor,
    output_data: *mut c_double,
    output_length: *mut c_int,
    output_shape: *mut c_int,
    max_shape_len: c_int,
    actual_shape_len: *mut c_int,
) -> TorshError {
    if jl_tensor.is_null()
        || output_data.is_null()
        || output_length.is_null()
        || output_shape.is_null()
        || actual_shape_len.is_null()
    {
        return TorshError::InvalidArgument;
    }

    let jl_tensor_ref = &*jl_tensor;
    let tensor_handle = jl_tensor_ref.handle();

    // Get tensor data pointer (torsh_tensor_data only takes one parameter)
    let data_void_ptr = torsh_tensor_data(tensor_handle);
    if data_void_ptr.is_null() {
        return TorshError::InvalidArgument;
    }

    let data_ptr = data_void_ptr as *const f32;
    let length = torsh_tensor_numel(tensor_handle) as c_int;

    let result = TorshError::Success;

    if result != TorshError::Success {
        return result;
    }

    // Copy data and convert f32 to f64
    let data_slice = std::slice::from_raw_parts(data_ptr, length as usize);
    let output_slice = std::slice::from_raw_parts_mut(output_data, length as usize);

    for (i, &value) in data_slice.iter().enumerate() {
        output_slice[i] = value as c_double;
    }

    *output_length = length;

    // Get shape information
    let mut shape_buffer = [0usize; 16]; // Buffer for shape dimensions
    let mut ndim = 0usize;
    let shape_result = torsh_tensor_shape(tensor_handle, shape_buffer.as_mut_ptr(), &mut ndim);

    if shape_result != TorshError::Success {
        return shape_result;
    }

    // Copy shape information
    let copy_len = std::cmp::min(ndim, max_shape_len as usize);
    let output_shape_slice = std::slice::from_raw_parts_mut(output_shape, copy_len);

    for (i, &dim) in shape_buffer.iter().take(copy_len).enumerate() {
        output_shape_slice[i] = dim as c_int;
    }

    *actual_shape_len = ndim as c_int;

    TorshError::Success
}

/// Convert ToRSh tensor to Julia Array{Float32} (native precision)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_to_jl_array_float32(
    jl_tensor: *const JuliaTensor,
    output_data: *mut f32,
    output_length: *mut c_int,
    output_shape: *mut c_int,
    max_shape_len: c_int,
    actual_shape_len: *mut c_int,
) -> TorshError {
    if jl_tensor.is_null()
        || output_data.is_null()
        || output_length.is_null()
        || output_shape.is_null()
        || actual_shape_len.is_null()
    {
        return TorshError::InvalidArgument;
    }

    let jl_tensor_ref = &*jl_tensor;
    let tensor_handle = jl_tensor_ref.handle();

    // Get tensor data pointer (torsh_tensor_data only takes one parameter)
    let data_void_ptr = torsh_tensor_data(tensor_handle);
    if data_void_ptr.is_null() {
        return TorshError::InvalidArgument;
    }

    let data_ptr = data_void_ptr as *const f32;
    let length = torsh_tensor_numel(tensor_handle) as c_int;

    let result = TorshError::Success;

    if result != TorshError::Success {
        return result;
    }

    // Copy data directly (no conversion needed for f32)
    let data_slice = std::slice::from_raw_parts(data_ptr, length as usize);
    let output_slice = std::slice::from_raw_parts_mut(output_data, length as usize);
    output_slice.copy_from_slice(data_slice);

    *output_length = length;

    // Get shape information
    let mut shape_buffer = [0usize; 16]; // Buffer for shape dimensions
    let mut ndim = 0usize;
    let shape_result = torsh_tensor_shape(tensor_handle, shape_buffer.as_mut_ptr(), &mut ndim);

    if shape_result != TorshError::Success {
        return shape_result;
    }

    // Copy shape information
    let copy_len = std::cmp::min(ndim, max_shape_len as usize);
    let output_shape_slice = std::slice::from_raw_parts_mut(output_shape, copy_len);

    for (i, &dim) in shape_buffer.iter().take(copy_len).enumerate() {
        output_shape_slice[i] = dim as c_int;
    }

    *actual_shape_len = ndim as c_int;

    TorshError::Success
}

/// Create Julia tensor from data
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_create_float64(
    data: *const c_double,
    length: c_int,
    shape: *const c_int,
    shape_len: c_int,
) -> *mut JuliaTensor {
    jl_array_float64_to_torsh_tensor(data, length, shape, shape_len)
}

/// Create Julia tensor from Float32 data (native precision)
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_create_float32(
    data: *const f32,
    length: c_int,
    shape: *const c_int,
    shape_len: c_int,
) -> *mut JuliaTensor {
    jl_array_float32_to_torsh_tensor(data, length, shape, shape_len)
}

/// Create Julia tensor with zeros
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_zeros(
    shape: *const c_int,
    shape_len: c_int,
) -> *mut JuliaTensor {
    if shape.is_null() {
        return ptr::null_mut();
    }

    let shape_slice = std::slice::from_raw_parts(shape, shape_len as usize);
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let tensor_handle = torsh_tensor_zeros(shape_usize.as_ptr(), shape_usize.len());

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(tensor_handle)))
}

/// Create Julia tensor with ones
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_ones(shape: *const c_int, shape_len: c_int) -> *mut JuliaTensor {
    if shape.is_null() {
        return ptr::null_mut();
    }

    let shape_slice = std::slice::from_raw_parts(shape, shape_len as usize);
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let tensor_handle = torsh_tensor_ones(shape_usize.as_ptr(), shape_usize.len());

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(tensor_handle)))
}

/// Create Julia tensor with random normal distribution (like Julia's randn)
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_randn(
    shape: *const c_int,
    shape_len: c_int,
) -> *mut JuliaTensor {
    if shape.is_null() {
        return ptr::null_mut();
    }

    let shape_slice = std::slice::from_raw_parts(shape, shape_len as usize);
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let tensor_handle = torsh_tensor_randn(shape_usize.as_ptr(), shape_usize.len());

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(tensor_handle)))
}

/// Create Julia tensor with uniform random distribution (like Julia's rand)
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_rand(shape: *const c_int, shape_len: c_int) -> *mut JuliaTensor {
    if shape.is_null() {
        return ptr::null_mut();
    }

    let shape_slice = std::slice::from_raw_parts(shape, shape_len as usize);
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let tensor_handle = torsh_tensor_rand(shape_usize.as_ptr(), shape_usize.len());

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(tensor_handle)))
}

/// Julia tensor addition
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_add(
    a: *const JuliaTensor,
    b: *const JuliaTensor,
) -> *mut JuliaTensor {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let a_ref = &*a;
    let b_ref = &*b;

    // Get shape of tensor a to create output tensor
    let mut shape_dims: Vec<usize> = vec![0; 16];
    let mut ndim: usize = 0;
    let shape_result = torsh_tensor_shape(a_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);
    if shape_result != TorshError::Success {
        return ptr::null_mut();
    }

    // Create output tensor
    let result_handle = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
    if result_handle.is_null() {
        return ptr::null_mut();
    }

    // Perform addition
    let add_result = torsh_tensor_add(a_ref.handle(), b_ref.handle(), result_handle);
    if add_result != TorshError::Success {
        torsh_tensor_free(result_handle);
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(result_handle)))
}

/// Julia tensor subtraction
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_sub(
    a: *const JuliaTensor,
    b: *const JuliaTensor,
) -> *mut JuliaTensor {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let a_ref = &*a;
    let b_ref = &*b;

    // Get shape of tensor a to create output tensor
    let mut shape_dims: Vec<usize> = vec![0; 16];
    let mut ndim: usize = 0;
    let shape_result = torsh_tensor_shape(a_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);
    if shape_result != TorshError::Success {
        return ptr::null_mut();
    }

    // Create output tensor
    let result_handle = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
    if result_handle.is_null() {
        return ptr::null_mut();
    }

    // Perform subtraction
    let sub_result = torsh_tensor_sub(a_ref.handle(), b_ref.handle(), result_handle);
    if sub_result != TorshError::Success {
        torsh_tensor_free(result_handle);
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(result_handle)))
}

/// Julia tensor element-wise multiplication
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_mul(
    a: *const JuliaTensor,
    b: *const JuliaTensor,
) -> *mut JuliaTensor {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let a_ref = &*a;
    let b_ref = &*b;

    // Get shape of tensor a to create output tensor
    let mut shape_dims: Vec<usize> = vec![0; 16];
    let mut ndim: usize = 0;
    let shape_result = torsh_tensor_shape(a_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);
    if shape_result != TorshError::Success {
        return ptr::null_mut();
    }

    // Create output tensor
    let result_handle = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
    if result_handle.is_null() {
        return ptr::null_mut();
    }

    // Perform multiplication
    let mul_result = torsh_tensor_mul(a_ref.handle(), b_ref.handle(), result_handle);
    if mul_result != TorshError::Success {
        torsh_tensor_free(result_handle);
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(result_handle)))
}

/// Julia tensor matrix multiplication (like Julia's * operator for matrices)
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_matmul(
    a: *const JuliaTensor,
    b: *const JuliaTensor,
) -> *mut JuliaTensor {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let a_ref = &*a;
    let b_ref = &*b;

    // Get shape of tensor a to create output tensor
    let mut shape_dims: Vec<usize> = vec![0; 16];
    let mut ndim: usize = 0;
    let shape_result = torsh_tensor_shape(a_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);
    if shape_result != TorshError::Success {
        return ptr::null_mut();
    }

    // Create output tensor
    let result_handle = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
    if result_handle.is_null() {
        return ptr::null_mut();
    }

    // Perform matrix multiplication
    let matmul_result = torsh_tensor_matmul(a_ref.handle(), b_ref.handle(), result_handle);
    if matmul_result != TorshError::Success {
        torsh_tensor_free(result_handle);
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(result_handle)))
}

/// Julia tensor transpose (like Julia's transpose() or ')
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_transpose(tensor: *const JuliaTensor) -> *mut JuliaTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let tensor_ref = &*tensor;
    let result_handle = torsh_tensor_transpose(tensor_ref.handle());

    if result_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(result_handle)))
}

/// Apply mathematical functions to tensors (broadcasting)
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_apply_func(
    tensor: *const JuliaTensor,
    func_name: *const c_char,
) -> *mut JuliaTensor {
    if tensor.is_null() || func_name.is_null() {
        return ptr::null_mut();
    }

    let tensor_ref = &*tensor;
    let func_str = CStr::from_ptr(func_name).to_str().unwrap_or("");

    // Get input tensor shape
    let mut temp_dims = vec![0usize; 16]; // max 16 dims
    let mut ndim = 0usize;
    let shape_result = torsh_tensor_shape(tensor_ref.handle(), temp_dims.as_mut_ptr(), &mut ndim);

    if shape_result != TorshError::Success {
        return ptr::null_mut();
    }

    // Create output tensor with same shape as input
    let output_tensor = torsh_tensor_zeros(temp_dims.as_ptr(), ndim);
    if output_tensor.is_null() {
        return ptr::null_mut();
    }

    // Apply the activation function - relu has different signature
    let result = match func_str {
        "relu" => {
            let relu_result = torsh_tensor_relu(tensor_ref.handle(), output_tensor);
            if relu_result != TorshError::Success {
                torsh_tensor_free(output_tensor);
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(output_tensor)))
        }
        "sigmoid" => {
            torsh_tensor_free(output_tensor); // Free the pre-allocated tensor
            let result_tensor = torsh_tensor_sigmoid(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        "tanh" => {
            torsh_tensor_free(output_tensor);
            let result_tensor = torsh_tensor_tanh(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        "exp" => {
            torsh_tensor_free(output_tensor);
            let result_tensor = torsh_tensor_exp(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        "log" => {
            torsh_tensor_free(output_tensor);
            let result_tensor = torsh_tensor_log(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        "sqrt" => {
            torsh_tensor_free(output_tensor);
            let result_tensor = torsh_tensor_sqrt(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        "abs" => {
            torsh_tensor_free(output_tensor);
            let result_tensor = torsh_tensor_abs(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        "sin" => {
            torsh_tensor_free(output_tensor);
            let result_tensor = torsh_tensor_sin(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        "cos" => {
            torsh_tensor_free(output_tensor);
            let result_tensor = torsh_tensor_cos(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        "tan" => {
            torsh_tensor_free(output_tensor);
            let result_tensor = torsh_tensor_tan(tensor_ref.handle());
            if result_tensor.is_null() {
                return ptr::null_mut();
            }
            Box::into_raw(Box::new(JuliaTensor::new(result_tensor)))
        }
        _ => {
            torsh_tensor_free(output_tensor);
            return ptr::null_mut();
        }
    };

    result
}

/// Julia Linear layer creation
#[no_mangle]
pub unsafe extern "C" fn jl_linear_create(
    in_features: c_int,
    out_features: c_int,
    bias: c_int, // Julia Bool (0 = false, 1 = true)
) -> *mut JuliaModule {
    let module_handle = torsh_linear_create(in_features as usize, out_features as usize, bias != 0);

    if module_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaModule::new(module_handle)))
}

/// Julia Linear layer forward pass
#[no_mangle]
pub unsafe extern "C" fn jl_linear_forward(
    module: *const JuliaModule,
    input: *const JuliaTensor,
) -> *mut JuliaTensor {
    if module.is_null() || input.is_null() {
        return ptr::null_mut();
    }

    let module_ref = &*module;
    let input_ref = &*input;

    // Get input shape to determine batch dimensions
    let mut ndim = 0usize;
    let mut input_shape = vec![0usize; 16];
    let shape_result = torsh_tensor_shape(input_ref.handle(), input_shape.as_mut_ptr(), &mut ndim);

    if shape_result != TorshError::Success {
        return ptr::null_mut();
    }

    // Get actual output dimensions from the linear module
    let output_features = torsh_linear_get_output_features(module_ref.handle());
    if output_features == 0 {
        return ptr::null_mut();
    }

    // Create proper output tensor shape: [batch_size, output_features]
    let output_shape = if ndim >= 2 {
        vec![input_shape[0], output_features]
    } else if ndim == 1 {
        vec![output_features]
    } else {
        return ptr::null_mut();
    };

    let output_handle = torsh_tensor_zeros(output_shape.as_ptr(), output_shape.len());
    if output_handle.is_null() {
        return ptr::null_mut();
    }

    let result = torsh_linear_forward(module_ref.handle(), input_ref.handle(), output_handle);

    if result != TorshError::Success {
        torsh_tensor_free(output_handle);
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(output_handle)))
}

/// Create Julia SGD optimizer
#[no_mangle]
pub unsafe extern "C" fn jl_sgd_create(learning_rate: c_double) -> *mut JuliaOptimizer {
    let optimizer_handle = torsh_sgd_create(learning_rate as f32);

    if optimizer_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaOptimizer::new(optimizer_handle)))
}

/// Create Julia Adam optimizer
#[no_mangle]
pub unsafe extern "C" fn jl_adam_create(
    learning_rate: c_double,
    beta1: c_double,
    beta2: c_double,
    epsilon: c_double,
) -> *mut JuliaOptimizer {
    let optimizer_handle = torsh_adam_create(
        learning_rate as f32,
        beta1 as f32,
        beta2 as f32,
        epsilon as f32,
    );

    if optimizer_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaOptimizer::new(optimizer_handle)))
}

/// Julia optimizer step
#[no_mangle]
pub unsafe extern "C" fn jl_optimizer_step(optimizer: *const JuliaOptimizer) -> TorshError {
    if optimizer.is_null() {
        return TorshError::InvalidArgument;
    }

    let optimizer_ref = &*optimizer;

    // Note: This assumes the optimizer type is stored in the handle
    // In practice, you'd need to track optimizer type or use a different approach
    torsh_sgd_step(optimizer_ref.handle()) // Simplified
}

/// Get Julia tensor dimensions (like Julia's size() function)
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_size(
    tensor: *const JuliaTensor,
    dims: *mut c_int,
    max_dims: c_int,
    actual_dims: *mut c_int,
) -> TorshError {
    if tensor.is_null() || dims.is_null() || actual_dims.is_null() {
        return TorshError::InvalidArgument;
    }

    let tensor_ref = &*tensor;

    // Get the tensor shape using the correct function signature
    let mut temp_dims = vec![0usize; max_dims as usize];
    let mut ndim = 0usize;
    let result = torsh_tensor_shape(tensor_ref.handle(), temp_dims.as_mut_ptr(), &mut ndim);

    if result == TorshError::Success {
        // Convert usize to c_int for Julia compatibility
        let copy_len = std::cmp::min(ndim, max_dims as usize);
        for i in 0..copy_len {
            *dims.add(i) = temp_dims[i] as c_int;
        }
        *actual_dims = ndim as c_int;
    }

    result
}

/// Get Julia tensor length (like Julia's length() function)
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_length(tensor: *const JuliaTensor) -> c_int {
    if tensor.is_null() {
        return -1;
    }

    let tensor_ref = &*tensor;
    torsh_tensor_numel(tensor_ref.handle()) as c_int
}

/// Julia tensor reduction operations
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_reduce(
    tensor: *const JuliaTensor,
    operation: *const c_char,
    dim: c_int, // -1 for all dimensions
) -> *mut JuliaTensor {
    if tensor.is_null() || operation.is_null() {
        return ptr::null_mut();
    }

    let tensor_ref = &*tensor;
    let op_str = CStr::from_ptr(operation).to_str().unwrap_or("");

    let result_handle = match op_str {
        "sum" => {
            if dim == -1 {
                torsh_tensor_sum_all(tensor_ref.handle())
            } else {
                torsh_tensor_sum_dim(tensor_ref.handle(), dim)
            }
        }
        "mean" => {
            if dim == -1 {
                torsh_tensor_mean_all(tensor_ref.handle())
            } else {
                torsh_tensor_mean_dim(tensor_ref.handle(), dim)
            }
        }
        "max" => {
            if dim == -1 {
                torsh_tensor_max_all(tensor_ref.handle())
            } else {
                torsh_tensor_max_dim(tensor_ref.handle(), dim)
            }
        }
        "min" => {
            if dim == -1 {
                torsh_tensor_min_all(tensor_ref.handle())
            } else {
                torsh_tensor_min_dim(tensor_ref.handle(), dim)
            }
        }
        _ => ptr::null_mut(),
    };

    if result_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(result_handle)))
}

/// Julia tensor broadcasting operations
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_broadcast_scalar(
    tensor: *const JuliaTensor,
    scalar: c_double,
    operation: *const c_char,
) -> *mut JuliaTensor {
    if tensor.is_null() || operation.is_null() {
        return ptr::null_mut();
    }

    let tensor_ref = &*tensor;
    let op_str = CStr::from_ptr(operation).to_str().unwrap_or("");

    let result_handle = match op_str {
        "add" => torsh_tensor_add_scalar(tensor_ref.handle(), scalar as f32),
        "sub" => torsh_tensor_sub_scalar(tensor_ref.handle(), scalar as f32),
        "mul" => torsh_tensor_mul_scalar(tensor_ref.handle(), scalar as f32),
        "div" => torsh_tensor_div_scalar(tensor_ref.handle(), scalar as f32),
        _ => ptr::null_mut(),
    };

    if result_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(JuliaTensor::new(result_handle)))
}

/// Set Julia GC tracking for tensor
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_set_gc_tracking(
    tensor: *mut JuliaTensor,
    tracked: c_int,
) -> TorshError {
    if tensor.is_null() {
        return TorshError::InvalidArgument;
    }

    let tensor_ref = &mut *tensor;
    tensor_ref.set_gc_tracked(tracked != 0);

    TorshError::Success
}

/// Check if Julia tensor is GC tracked
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_is_gc_tracked(tensor: *const JuliaTensor) -> c_int {
    if tensor.is_null() {
        return -1;
    }

    let tensor_ref = &*tensor;
    if tensor_ref.is_gc_tracked() {
        1
    } else {
        0
    }
}

/// Free Julia tensor
#[no_mangle]
pub unsafe extern "C" fn jl_tensor_free(tensor: *mut JuliaTensor) {
    if !tensor.is_null() {
        let tensor_box = Box::from_raw(tensor);
        torsh_tensor_free(tensor_box.handle());
    }
}

/// Free Julia module
#[no_mangle]
pub unsafe extern "C" fn jl_module_free(module: *mut JuliaModule) {
    if !module.is_null() {
        let module_box = Box::from_raw(module);
        torsh_module_free(module_box.handle());
    }
}

/// Free Julia optimizer
#[no_mangle]
pub unsafe extern "C" fn jl_optimizer_free(optimizer: *mut JuliaOptimizer) {
    if !optimizer.is_null() {
        let optimizer_box = Box::from_raw(optimizer);
        torsh_optimizer_free(optimizer_box.handle());
    }
}

/// Get last error message for Julia
#[no_mangle]
pub unsafe extern "C" fn jl_get_last_error(buffer: *mut c_char, buffer_size: c_int) -> c_int {
    let error_ptr = torsh_get_last_error();
    if error_ptr.is_null() {
        return 0;
    }

    let error_str = std::ffi::CStr::from_ptr(error_ptr);
    let error_bytes = error_str.to_bytes();
    let copy_len = std::cmp::min(error_bytes.len(), (buffer_size - 1) as usize);

    std::ptr::copy_nonoverlapping(error_bytes.as_ptr(), buffer as *mut u8, copy_len);
    *buffer.add(copy_len) = 0; // null terminate

    copy_len as c_int
}

/// Clear last error for Julia
#[no_mangle]
pub unsafe extern "C" fn jl_clear_last_error() {
    torsh_clear_last_error();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_julia_tensor_wrapper() {
        let handle = 0x12345 as *mut TorshTensor;
        let jl_tensor = JuliaTensor::new(handle);
        assert_eq!(jl_tensor.handle(), handle);
        assert!(jl_tensor.jl_value().is_null());
        assert!(!jl_tensor.is_gc_tracked());
    }

    #[test]
    fn test_julia_module_wrapper() {
        let handle = 0x12345 as *mut TorshModule;
        let jl_module = JuliaModule::new(handle);
        assert_eq!(jl_module.handle(), handle);
        assert!(jl_module.jl_value().is_null());
    }

    #[test]
    fn test_julia_optimizer_wrapper() {
        let handle = 0x12345 as *mut TorshOptimizer;
        let jl_optimizer = JuliaOptimizer::new(handle);
        assert_eq!(jl_optimizer.handle(), handle);
        assert!(jl_optimizer.jl_value().is_null());
    }

    #[test]
    fn test_julia_gc_tracking() {
        let handle = 0x12345 as *mut TorshTensor;
        let mut jl_tensor = JuliaTensor::new(handle);

        assert!(!jl_tensor.is_gc_tracked());
        jl_tensor.set_gc_tracked(true);
        assert!(jl_tensor.is_gc_tracked());
        jl_tensor.set_gc_tracked(false);
        assert!(!jl_tensor.is_gc_tracked());
    }
}
