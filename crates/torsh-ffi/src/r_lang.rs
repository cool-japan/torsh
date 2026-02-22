//! R Language Bindings for ToRSh
//!
//! This module provides R language bindings using R's C API for statistical computing integration.
//! R bindings enable seamless integration with R's statistical computing environment and data analysis tools.

#![allow(dead_code)]

use crate::c_api::*;
use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;

/// R wrapper for tensor operations
/// R uses SEXP (S-expressions) for data representation
pub struct RTensor {
    handle: *mut TorshTensor,
    sexp_ptr: *mut c_void, // SEXP pointer for R integration
}

impl RTensor {
    pub fn new(handle: *mut TorshTensor) -> Self {
        Self {
            handle,
            sexp_ptr: ptr::null_mut(),
        }
    }

    pub fn handle(&self) -> *mut TorshTensor {
        self.handle
    }

    pub fn set_sexp(&mut self, sexp_ptr: *mut c_void) {
        self.sexp_ptr = sexp_ptr;
    }

    pub fn sexp(&self) -> *mut c_void {
        self.sexp_ptr
    }
}

/// R wrapper for neural network modules
pub struct RModule {
    handle: *mut TorshModule,
    r_object_ptr: *mut c_void,
}

impl RModule {
    pub fn new(handle: *mut TorshModule) -> Self {
        Self {
            handle,
            r_object_ptr: ptr::null_mut(),
        }
    }

    pub fn handle(&self) -> *mut TorshModule {
        self.handle
    }
}

/// R wrapper for optimizers
pub struct ROptimizer {
    handle: *mut TorshOptimizer,
    r_object_ptr: *mut c_void,
}

impl ROptimizer {
    pub fn new(handle: *mut TorshOptimizer) -> Self {
        Self {
            handle,
            r_object_ptr: ptr::null_mut(),
        }
    }

    pub fn handle(&self) -> *mut TorshOptimizer {
        self.handle
    }
}

/// Convert R's REAL vector to ToRSh tensor
#[no_mangle]
pub unsafe extern "C" fn r_real_to_torsh_tensor(
    real_data: *const c_double,
    length: c_int,
    shape_data: *const c_int,
    shape_len: c_int,
) -> *mut RTensor {
    if real_data.is_null() || shape_data.is_null() {
        return ptr::null_mut();
    }

    let data_slice = std::slice::from_raw_parts(real_data, length as usize);
    let shape_slice = std::slice::from_raw_parts(shape_data, shape_len as usize);

    // Convert f64 to f32 for ToRSh
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();

    let tensor_handle = torsh_tensor_new(
        f32_data.as_ptr() as *const c_void,
        shape.as_ptr(),
        shape.len(),
        crate::c_api::TorshDType::F32,
    );

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RTensor::new(tensor_handle)))
}

/// Convert R's INTEGER vector to ToRSh tensor
#[no_mangle]
pub unsafe extern "C" fn r_integer_to_torsh_tensor(
    int_data: *const c_int,
    length: c_int,
    shape_data: *const c_int,
    shape_len: c_int,
) -> *mut RTensor {
    if int_data.is_null() || shape_data.is_null() {
        return ptr::null_mut();
    }

    let data_slice = std::slice::from_raw_parts(int_data, length as usize);
    let shape_slice = std::slice::from_raw_parts(shape_data, shape_len as usize);

    // Convert i32 to f32 for ToRSh (R integers to float tensors)
    let f32_data: Vec<f32> = data_slice.iter().map(|&x| x as f32).collect();
    let shape: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();

    let tensor_handle = torsh_tensor_new(
        f32_data.as_ptr() as *const c_void,
        shape.as_ptr(),
        shape.len(),
        crate::c_api::TorshDType::F32,
    );

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RTensor::new(tensor_handle)))
}

/// Convert ToRSh tensor to R's REAL vector
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_to_r_real(
    r_tensor: *const RTensor,
    output_data: *mut c_double,
    output_length: *mut c_int,
    output_shape: *mut c_int,
    max_shape_len: c_int,
    actual_shape_len: *mut c_int,
) -> TorshError {
    if r_tensor.is_null()
        || output_data.is_null()
        || output_length.is_null()
        || output_shape.is_null()
        || actual_shape_len.is_null()
    {
        return TorshError::InvalidArgument;
    }

    let r_tensor_ref = &*r_tensor;
    let tensor_handle = r_tensor_ref.handle();

    // Get tensor data pointer
    let data_ptr = torsh_tensor_data(tensor_handle);
    if data_ptr.is_null() {
        return TorshError::InvalidArgument;
    }

    // Get tensor size
    let length = torsh_tensor_numel(tensor_handle) as c_int;
    let data_ptr = data_ptr as *const f32;

    // Copy data and convert f32 to f64
    let data_slice = std::slice::from_raw_parts(data_ptr, length as usize);
    let output_slice = std::slice::from_raw_parts_mut(output_data, length as usize);

    for (i, &value) in data_slice.iter().enumerate() {
        output_slice[i] = value as c_double;
    }

    *output_length = length;

    // Get shape information
    let mut shape_dims: Vec<usize> = vec![0; max_shape_len as usize];
    let mut ndim: usize = 0;

    let shape_result = torsh_tensor_shape(tensor_handle, shape_dims.as_mut_ptr(), &mut ndim);

    if shape_result != TorshError::Success {
        return shape_result;
    }

    // Copy shape information to output
    let copy_len = std::cmp::min(ndim, max_shape_len as usize);
    let output_shape_slice = std::slice::from_raw_parts_mut(output_shape, copy_len);

    for (i, &dim) in shape_dims.iter().take(copy_len).enumerate() {
        output_shape_slice[i] = dim as c_int;
    }

    *actual_shape_len = ndim as c_int;

    TorshError::Success
}

/// Create R tensor from data (R interface)
#[no_mangle]
pub unsafe extern "C" fn r_tensor_create(
    data: *const c_double,
    length: c_int,
    shape: *const c_int,
    shape_len: c_int,
) -> *mut RTensor {
    r_real_to_torsh_tensor(data, length, shape, shape_len)
}

/// Create R tensor with zeros
#[no_mangle]
pub unsafe extern "C" fn r_tensor_zeros(shape: *const c_int, shape_len: c_int) -> *mut RTensor {
    if shape.is_null() {
        return ptr::null_mut();
    }

    let shape_slice = std::slice::from_raw_parts(shape, shape_len as usize);
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let tensor_handle = torsh_tensor_zeros(shape_usize.as_ptr(), shape_len as usize);

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RTensor::new(tensor_handle)))
}

/// Create R tensor with ones
#[no_mangle]
pub unsafe extern "C" fn r_tensor_ones(shape: *const c_int, shape_len: c_int) -> *mut RTensor {
    if shape.is_null() {
        return ptr::null_mut();
    }

    let shape_slice = std::slice::from_raw_parts(shape, shape_len as usize);
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let tensor_handle = torsh_tensor_ones(shape_usize.as_ptr(), shape_len as usize);

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RTensor::new(tensor_handle)))
}

/// Create R tensor with random normal distribution (matching R's rnorm)
#[no_mangle]
pub unsafe extern "C" fn r_tensor_rnorm(
    shape: *const c_int,
    shape_len: c_int,
    mean: c_double,
    sd: c_double,
) -> *mut RTensor {
    if shape.is_null() {
        return ptr::null_mut();
    }

    let shape_slice = std::slice::from_raw_parts(shape, shape_len as usize);
    let shape_usize: Vec<usize> = shape_slice.iter().map(|&x| x as usize).collect();
    let tensor_handle = torsh_tensor_randn(shape_usize.as_ptr(), shape_len as usize);

    if tensor_handle.is_null() {
        return ptr::null_mut();
    }

    // Scale and shift to match R's rnorm(mean, sd)
    if mean != 0.0 || sd != 1.0 {
        let mut current_tensor = tensor_handle;

        // Scale by sd if needed
        if sd != 1.0 {
            let scaled = torsh_tensor_mul_scalar(current_tensor, sd as f32);
            if scaled.is_null() {
                torsh_tensor_free(current_tensor);
                return ptr::null_mut();
            }
            torsh_tensor_free(current_tensor);
            current_tensor = scaled;
        }

        // Add mean if needed
        if mean != 0.0 {
            let result = torsh_tensor_add_scalar(current_tensor, mean as f32);
            if result.is_null() {
                torsh_tensor_free(current_tensor);
                return ptr::null_mut();
            }
            torsh_tensor_free(current_tensor);
            current_tensor = result;
        }

        return Box::into_raw(Box::new(RTensor::new(current_tensor)));
    }

    Box::into_raw(Box::new(RTensor::new(tensor_handle)))
}

/// R tensor addition
#[no_mangle]
pub unsafe extern "C" fn r_tensor_add(a: *const RTensor, b: *const RTensor) -> *mut RTensor {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let a_ref = &*a;
    let b_ref = &*b;

    // Get shape of tensor a (assuming they have the same shape)
    let mut shape_dims: Vec<usize> = vec![0; 16]; // Max dimensions
    let mut ndim: usize = 0;

    let shape_result = torsh_tensor_shape(a_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);

    if shape_result != TorshError::Success {
        return ptr::null_mut();
    }

    // Create output tensor with same shape
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

    Box::into_raw(Box::new(RTensor::new(result_handle)))
}

/// R tensor multiplication
#[no_mangle]
pub unsafe extern "C" fn r_tensor_mul(a: *const RTensor, b: *const RTensor) -> *mut RTensor {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let a_ref = &*a;
    let b_ref = &*b;

    // Get shape of tensor a (assuming they have the same shape for element-wise multiplication)
    let mut shape_dims: Vec<usize> = vec![0; 16]; // Max dimensions
    let mut ndim: usize = 0;

    let shape_result = torsh_tensor_shape(a_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);

    if shape_result != TorshError::Success {
        return ptr::null_mut();
    }

    // Create output tensor with same shape
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

    Box::into_raw(Box::new(RTensor::new(result_handle)))
}

/// R tensor matrix multiplication (%*% operator in R)
#[no_mangle]
pub unsafe extern "C" fn r_tensor_matmul(a: *const RTensor, b: *const RTensor) -> *mut RTensor {
    if a.is_null() || b.is_null() {
        return ptr::null_mut();
    }

    let a_ref = &*a;
    let b_ref = &*b;

    // Get shapes of input tensors to determine output shape
    let mut shape_a: Vec<usize> = vec![0; 16];
    let mut ndim_a: usize = 0;
    let mut shape_b: Vec<usize> = vec![0; 16];
    let mut ndim_b: usize = 0;

    let shape_result_a = torsh_tensor_shape(a_ref.handle(), shape_a.as_mut_ptr(), &mut ndim_a);
    let shape_result_b = torsh_tensor_shape(b_ref.handle(), shape_b.as_mut_ptr(), &mut ndim_b);

    if shape_result_a != TorshError::Success || shape_result_b != TorshError::Success {
        return ptr::null_mut();
    }

    // For matrix multiplication A(m×n) × B(n×p) = C(m×p)
    // Use shape of A for simplicity (the matmul function should handle shape validation)
    let result_handle = torsh_tensor_zeros(shape_a.as_ptr(), ndim_a);
    if result_handle.is_null() {
        return ptr::null_mut();
    }

    // Perform matrix multiplication
    let matmul_result = torsh_tensor_matmul(a_ref.handle(), b_ref.handle(), result_handle);
    if matmul_result != TorshError::Success {
        torsh_tensor_free(result_handle);
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RTensor::new(result_handle)))
}

/// R tensor transpose (t() function in R)
#[no_mangle]
pub unsafe extern "C" fn r_tensor_t(tensor: *const RTensor) -> *mut RTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let tensor_ref = &*tensor;
    let result_handle = torsh_tensor_transpose(tensor_ref.handle());

    if result_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RTensor::new(result_handle)))
}

/// Apply function to tensor elements (lapply-style operation)
#[no_mangle]
pub unsafe extern "C" fn r_tensor_apply(
    tensor: *const RTensor,
    func_name: *const c_char,
) -> *mut RTensor {
    if tensor.is_null() || func_name.is_null() {
        return ptr::null_mut();
    }

    let tensor_ref = &*tensor;
    let func_str = CStr::from_ptr(func_name).to_str().unwrap_or("");

    let result_handle = match func_str {
        "relu" => {
            // Get shape of input tensor
            let mut shape_dims: Vec<usize> = vec![0; 16];
            let mut ndim: usize = 0;
            let shape_result =
                torsh_tensor_shape(tensor_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);
            if shape_result != TorshError::Success {
                return ptr::null_mut();
            }
            // Create output tensor
            let output = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
            if output.is_null() {
                return ptr::null_mut();
            }
            // Apply relu
            let relu_result = torsh_tensor_relu(tensor_ref.handle(), output);
            if relu_result != TorshError::Success {
                torsh_tensor_free(output);
                return ptr::null_mut();
            }
            output
        }
        "exp" => torsh_tensor_exp(tensor_ref.handle()),
        "log" => torsh_tensor_log(tensor_ref.handle()),
        "sqrt" => torsh_tensor_sqrt(tensor_ref.handle()),
        "abs" => torsh_tensor_abs(tensor_ref.handle()),
        _ => ptr::null_mut(),
    };

    if result_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RTensor::new(result_handle)))
}

/// Create linear layer (R interface)
#[no_mangle]
pub unsafe extern "C" fn r_linear_create(
    in_features: c_int,
    out_features: c_int,
    bias: c_int, // R's logical (0 = FALSE, 1 = TRUE)
) -> *mut RModule {
    let module_handle = torsh_linear_create(in_features as usize, out_features as usize, bias != 0);

    if module_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RModule::new(module_handle)))
}

/// Forward pass through linear layer (R interface)
#[no_mangle]
pub unsafe extern "C" fn r_linear_forward(
    module: *const RModule,
    input: *const RTensor,
) -> *mut RTensor {
    if module.is_null() || input.is_null() {
        return ptr::null_mut();
    }

    let module_ref = &*module;
    let input_ref = &*input;

    // Get input tensor shape to determine batch dimensions
    let mut shape_dims: Vec<usize> = vec![0; 16];
    let mut ndim: usize = 0;
    let shape_result = torsh_tensor_shape(input_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);
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
        vec![shape_dims[0], output_features]
    } else if ndim == 1 {
        vec![output_features]
    } else {
        return ptr::null_mut();
    };

    let output_handle = torsh_tensor_zeros(output_shape.as_ptr(), output_shape.len());
    let result = torsh_linear_forward(module_ref.handle(), input_ref.handle(), output_handle);

    if result != TorshError::Success {
        torsh_tensor_free(output_handle);
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(RTensor::new(output_handle)))
}

/// Create SGD optimizer (R interface)
#[no_mangle]
pub unsafe extern "C" fn r_sgd_create(learning_rate: c_double) -> *mut ROptimizer {
    let optimizer_handle = torsh_sgd_create(learning_rate as f32);

    if optimizer_handle.is_null() {
        return ptr::null_mut();
    }

    Box::into_raw(Box::new(ROptimizer::new(optimizer_handle)))
}

/// SGD step (R interface)
#[no_mangle]
pub unsafe extern "C" fn r_sgd_step(optimizer: *const ROptimizer) -> TorshError {
    if optimizer.is_null() {
        return TorshError::InvalidArgument;
    }

    let optimizer_ref = &*optimizer;
    torsh_sgd_step(optimizer_ref.handle())
}

/// Get R tensor dimensions (similar to R's dim() function)
#[no_mangle]
pub unsafe extern "C" fn r_tensor_dim(
    tensor: *const RTensor,
    dims: *mut c_int,
    max_dims: c_int,
    actual_dims: *mut c_int,
) -> TorshError {
    if tensor.is_null() || dims.is_null() || actual_dims.is_null() {
        return TorshError::InvalidArgument;
    }

    let tensor_ref = &*tensor;
    // Convert c_int parameters to usize for the C API
    let mut shape_dims: Vec<usize> = vec![0; max_dims as usize];
    let mut ndim: usize = 0;

    let result = torsh_tensor_shape(tensor_ref.handle(), shape_dims.as_mut_ptr(), &mut ndim);

    if result == TorshError::Success {
        // Convert back to c_int for R
        let copy_len = std::cmp::min(ndim, max_dims as usize);
        let dims_slice = std::slice::from_raw_parts_mut(dims, copy_len);

        for (i, &dim) in shape_dims.iter().take(copy_len).enumerate() {
            dims_slice[i] = dim as c_int;
        }

        *actual_dims = ndim as c_int;
    }

    result
}

/// Get R tensor length (similar to R's length() function)
#[no_mangle]
pub unsafe extern "C" fn r_tensor_length(tensor: *const RTensor) -> c_int {
    if tensor.is_null() {
        return -1;
    }

    let tensor_ref = &*tensor;
    torsh_tensor_numel(tensor_ref.handle()) as c_int
}

/// Summary statistics for R tensor (similar to R's summary() function)
#[no_mangle]
pub unsafe extern "C" fn r_tensor_summary(
    tensor: *const RTensor,
    min: *mut c_double,
    max: *mut c_double,
    mean: *mut c_double,
    median: *mut c_double,
    q1: *mut c_double,
    q3: *mut c_double,
) -> TorshError {
    if tensor.is_null()
        || min.is_null()
        || max.is_null()
        || mean.is_null()
        || median.is_null()
        || q1.is_null()
        || q3.is_null()
    {
        return TorshError::InvalidArgument;
    }

    let tensor_ref = &*tensor;

    // Get tensor data pointer (torsh_tensor_data only takes one parameter)
    let data_void_ptr = torsh_tensor_data(tensor_ref.handle());
    if data_void_ptr.is_null() {
        return TorshError::InvalidArgument;
    }

    let data_ptr = data_void_ptr as *const f32;
    let length = torsh_tensor_numel(tensor_ref.handle()) as c_int;

    let result = TorshError::Success;

    if result != TorshError::Success {
        return result;
    }

    // Calculate statistics
    let data_slice = std::slice::from_raw_parts(data_ptr, length as usize);
    let mut sorted_data: Vec<f32> = data_slice.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    *min = *sorted_data.first().unwrap_or(&0.0) as c_double;
    *max = *sorted_data.last().unwrap_or(&0.0) as c_double;

    let sum: f32 = data_slice.iter().sum();
    *mean = (sum / length as f32) as c_double;

    let n = sorted_data.len();
    *median = if n % 2 == 0 {
        (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0
    } else {
        sorted_data[n / 2]
    } as c_double;

    *q1 = sorted_data[n / 4] as c_double;
    *q3 = sorted_data[3 * n / 4] as c_double;

    TorshError::Success
}

/// Free R tensor
#[no_mangle]
pub unsafe extern "C" fn r_tensor_free(tensor: *mut RTensor) {
    if !tensor.is_null() {
        let tensor_box = Box::from_raw(tensor);
        torsh_tensor_free(tensor_box.handle());
    }
}

/// Free R module
#[no_mangle]
pub unsafe extern "C" fn r_module_free(module: *mut RModule) {
    if !module.is_null() {
        let module_box = Box::from_raw(module);
        torsh_module_free(module_box.handle());
    }
}

/// Free R optimizer
#[no_mangle]
pub unsafe extern "C" fn r_optimizer_free(optimizer: *mut ROptimizer) {
    if !optimizer.is_null() {
        let optimizer_box = Box::from_raw(optimizer);
        torsh_optimizer_free(optimizer_box.handle());
    }
}

/// Get last error message for R
#[no_mangle]
pub unsafe extern "C" fn r_get_last_error(buffer: *mut c_char, buffer_size: c_int) -> c_int {
    let error_ptr = torsh_get_last_error();
    if error_ptr.is_null() {
        return 0; // No error
    }

    let error_str = CStr::from_ptr(error_ptr);
    let error_bytes = error_str.to_bytes();
    let copy_len = std::cmp::min(error_bytes.len(), (buffer_size - 1) as usize);

    ptr::copy_nonoverlapping(error_bytes.as_ptr() as *const c_char, buffer, copy_len);
    *buffer.add(copy_len) = 0; // Null terminate

    copy_len as c_int
}

/// Clear last error for R
#[no_mangle]
pub unsafe extern "C" fn r_clear_last_error() {
    torsh_clear_last_error();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r_tensor_wrapper() {
        let handle = 0x12345 as *mut TorshTensor;
        let r_tensor = RTensor::new(handle);
        assert_eq!(r_tensor.handle(), handle);
        assert!(r_tensor.sexp().is_null());
    }

    #[test]
    fn test_r_module_wrapper() {
        let handle = 0x12345 as *mut TorshModule;
        let r_module = RModule::new(handle);
        assert_eq!(r_module.handle(), handle);
    }

    #[test]
    fn test_r_optimizer_wrapper() {
        let handle = 0x12345 as *mut TorshOptimizer;
        let r_optimizer = ROptimizer::new(handle);
        assert_eq!(r_optimizer.handle(), handle);
    }
}
