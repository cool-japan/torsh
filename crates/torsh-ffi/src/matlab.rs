//! MATLAB MEX interface for ToRSh tensors
//!
//! This module provides MATLAB integration through MEX (MATLAB Executable) functions.
//! It allows MATLAB users to create and manipulate ToRSh tensors directly from MATLAB code.

use crate::c_api::*;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};
use std::slice;

#[allow(dead_code)]
#[repr(C)]
struct MxArray {
    _private: [u8; 0],
}

// MATLAB MEX API declarations (simplified)
extern "C" {
    fn mxGetPr(array_ptr: *const MxArray) -> *mut c_double;
    fn mxGetM(array_ptr: *const MxArray) -> usize;
    fn mxGetN(array_ptr: *const MxArray) -> usize;
    fn mxGetNumberOfDimensions(array_ptr: *const MxArray) -> usize;
    fn mxGetDimensions(array_ptr: *const MxArray) -> *const usize;
    fn mxCreateDoubleMatrix(m: usize, n: usize, complexity: c_int) -> *mut MxArray;
    fn mxCreateNumericArray(
        ndims: usize,
        dims: *const usize,
        class_id: c_int,
        complexity: c_int,
    ) -> *mut MxArray;
    fn mexErrMsgIdAndTxt(msgid: *const c_char, msg: *const c_char);
    fn mexPrintf(fmt: *const c_char, ...);
}

// MATLAB class IDs
const MX_DOUBLE_CLASS: c_int = 6;
const MX_REAL: c_int = 0;

/// Convert MATLAB array to ToRSh tensor
#[no_mangle]
pub extern "C" fn matlab_to_torsh_tensor(mx_array: *const MxArray) -> TensorHandle {
    if mx_array.is_null() {
        error_and_return_null("Input MATLAB array is null");
        return std::ptr::null_mut();
    }

    unsafe {
        // Get array properties
        let data_ptr = mxGetPr(mx_array);
        let ndims = mxGetNumberOfDimensions(mx_array);
        let dims_ptr = mxGetDimensions(mx_array);

        if data_ptr.is_null() || dims_ptr.is_null() {
            error_and_return_null("Failed to get MATLAB array data or dimensions");
            return std::ptr::null_mut();
        }

        // Convert dimensions
        let dims = slice::from_raw_parts(dims_ptr, ndims);
        let total_elements: usize = dims.iter().product();

        // Convert data
        let matlab_data = slice::from_raw_parts(data_ptr, total_elements);
        let rust_data: Vec<f32> = matlab_data.iter().map(|&x| x as f32).collect();

        // Create ToRSh tensor
        let dims_usize: Vec<usize> = dims.to_vec();
        torsh_tensor_from_data(
            rust_data.as_ptr(),
            rust_data.len(),
            dims_usize.as_ptr(),
            dims_usize.len(),
        )
    }
}

/// Convert ToRSh tensor to MATLAB array
#[no_mangle]
pub extern "C" fn torsh_tensor_to_matlab(tensor: TensorHandle) -> *mut MxArray {
    if tensor.is_null() {
        error_and_return_null("Input tensor handle is null");
        return std::ptr::null_mut();
    }

    unsafe {
        // Get tensor properties
        let mut ndims = torsh_tensor_ndim(tensor);
        let mut dims = vec![0usize; ndims];
        let shape_result = torsh_tensor_shape(tensor, dims.as_mut_ptr(), &mut ndims);

        if shape_result != TorshError::Success {
            error_and_return_null("Failed to get tensor shape");
            return std::ptr::null_mut();
        }

        // Get tensor data
        let numel = torsh_tensor_numel(tensor);
        let mut data = vec![0.0f32; numel];
        let data_ptr = torsh_tensor_data(tensor);

        if data_ptr.is_null() {
            error_and_return_null("Failed to get tensor data");
            return std::ptr::null_mut();
        }

        // Copy data from the tensor's internal storage
        let src_data = slice::from_raw_parts(data_ptr as *const f32, numel);
        data.copy_from_slice(src_data);

        // Create MATLAB array
        let mx_array = if ndims <= 2 {
            // Use matrix creation for 2D or less
            let rows = if ndims >= 1 { dims[0] } else { 1 };
            let cols = if ndims >= 2 { dims[1] } else { 1 };
            mxCreateDoubleMatrix(rows, cols, MX_REAL)
        } else {
            // Use N-dimensional array creation
            mxCreateNumericArray(ndims, dims.as_ptr(), MX_DOUBLE_CLASS, MX_REAL)
        };

        if mx_array.is_null() {
            error_and_return_null("Failed to create MATLAB array");
            return std::ptr::null_mut();
        }

        // Copy data to MATLAB array
        let mx_data = mxGetPr(mx_array);
        if mx_data.is_null() {
            error_and_return_null("Failed to get MATLAB array data pointer");
            return std::ptr::null_mut();
        }

        let mx_slice = slice::from_raw_parts_mut(mx_data, numel);
        for (i, &value) in data.iter().enumerate() {
            mx_slice[i] = value as c_double;
        }

        mx_array
    }
}

/// MATLAB MEX function: Create tensor from data
#[no_mangle]
pub extern "C" fn mexFunction(
    nlhs: c_int,
    plhs: *mut *mut MxArray,
    nrhs: c_int,
    prhs: *const *const MxArray,
) {
    unsafe {
        if nrhs == 0 {
            mexPrintf(CString::new("ToRSh MATLAB Interface\n").expect("static string should not contain null bytes").as_ptr());
            mexPrintf(
                CString::new("Usage: tensor = torsh_tensor(data)\n")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            mexPrintf(
                CString::new("       data = torsh_data(tensor)\n")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return;
        }

        if nrhs != 1 || nlhs != 1 {
            let msg = CString::new("Exactly one input and one output required").expect("static string should not contain null bytes");
            mexErrMsgIdAndTxt(
                CString::new("ToRSh:InvalidArgs").expect("static string should not contain null bytes").as_ptr(),
                msg.as_ptr(),
            );
            return;
        }

        // Convert MATLAB array to ToRSh tensor and back
        let input_array = *prhs;
        let tensor = matlab_to_torsh_tensor(input_array);

        if tensor.is_null() {
            let msg = CString::new("Failed to create ToRSh tensor").expect("static string should not contain null bytes");
            mexErrMsgIdAndTxt(
                CString::new("ToRSh:CreationError").expect("static string should not contain null bytes").as_ptr(),
                msg.as_ptr(),
            );
            return;
        }

        // Convert back to MATLAB for output
        let output_array = torsh_tensor_to_matlab(tensor);
        if output_array.is_null() {
            torsh_tensor_free(tensor);
            let msg = CString::new("Failed to convert tensor to MATLAB array").expect("static string should not contain null bytes");
            mexErrMsgIdAndTxt(
                CString::new("ToRSh:ConversionError").expect("static string should not contain null bytes").as_ptr(),
                msg.as_ptr(),
            );
            return;
        }

        *plhs = output_array;
        torsh_tensor_free(tensor);
    }
}

/// MATLAB tensor operations
#[no_mangle]
pub extern "C" fn matlab_tensor_add(
    lhs_mx: *const MxArray,
    rhs_mx: *const MxArray,
) -> *mut MxArray {
    unsafe {
        let lhs_tensor = matlab_to_torsh_tensor(lhs_mx);
        let rhs_tensor = matlab_to_torsh_tensor(rhs_mx);

        if lhs_tensor.is_null() || rhs_tensor.is_null() {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Get shape of the first tensor for the output tensor
        let mut ndims = torsh_tensor_ndim(lhs_tensor);
        let mut dims = vec![0usize; ndims];
        let shape_result = torsh_tensor_shape(lhs_tensor, dims.as_mut_ptr(), &mut ndims);

        if shape_result != TorshError::Success {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Create output tensor with the same shape
        let result = torsh_tensor_zeros(dims.as_ptr(), ndims);
        if result.is_null() {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Perform the addition
        let add_result = torsh_tensor_add(lhs_tensor, rhs_tensor, result);
        if add_result != TorshError::Success {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            torsh_tensor_free(result);
            return std::ptr::null_mut();
        }

        let result_mx = torsh_tensor_to_matlab(result);

        torsh_tensor_free(lhs_tensor);
        torsh_tensor_free(rhs_tensor);
        torsh_tensor_free(result);

        result_mx
    }
}

#[no_mangle]
pub extern "C" fn matlab_tensor_mul(
    lhs_mx: *const MxArray,
    rhs_mx: *const MxArray,
) -> *mut MxArray {
    unsafe {
        let lhs_tensor = matlab_to_torsh_tensor(lhs_mx);
        let rhs_tensor = matlab_to_torsh_tensor(rhs_mx);

        if lhs_tensor.is_null() || rhs_tensor.is_null() {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Get shape of the first tensor for the output tensor
        let mut ndims = torsh_tensor_ndim(lhs_tensor);
        let mut dims = vec![0usize; ndims];
        let shape_result = torsh_tensor_shape(lhs_tensor, dims.as_mut_ptr(), &mut ndims);

        if shape_result != TorshError::Success {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Create output tensor with the same shape
        let result = torsh_tensor_zeros(dims.as_ptr(), ndims);
        if result.is_null() {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Perform the multiplication
        let mul_result = torsh_tensor_multiply(lhs_tensor, rhs_tensor, result);
        if mul_result != TorshError::Success {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            torsh_tensor_free(result);
            return std::ptr::null_mut();
        }

        let result_mx = torsh_tensor_to_matlab(result);

        torsh_tensor_free(lhs_tensor);
        torsh_tensor_free(rhs_tensor);
        torsh_tensor_free(result);

        result_mx
    }
}

#[no_mangle]
pub extern "C" fn matlab_tensor_matmul(
    lhs_mx: *const MxArray,
    rhs_mx: *const MxArray,
) -> *mut MxArray {
    unsafe {
        let lhs_tensor = matlab_to_torsh_tensor(lhs_mx);
        let rhs_tensor = matlab_to_torsh_tensor(rhs_mx);

        if lhs_tensor.is_null() || rhs_tensor.is_null() {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Get shapes of both tensors to calculate proper matmul output shape
        let mut lhs_ndims = torsh_tensor_ndim(lhs_tensor);
        let mut rhs_ndims = torsh_tensor_ndim(rhs_tensor);

        let mut lhs_dims = vec![0usize; lhs_ndims];
        let mut rhs_dims = vec![0usize; rhs_ndims];

        let lhs_shape_result =
            torsh_tensor_shape(lhs_tensor, lhs_dims.as_mut_ptr(), &mut lhs_ndims);
        let rhs_shape_result =
            torsh_tensor_shape(rhs_tensor, rhs_dims.as_mut_ptr(), &mut rhs_ndims);

        if lhs_shape_result != TorshError::Success || rhs_shape_result != TorshError::Success {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Calculate proper matmul output shape: (m, k) x (k, n) -> (m, n)
        let output_dims = if lhs_ndims >= 2 && rhs_ndims >= 2 {
            // For 2D matrices: [m, k] x [k, n] -> [m, n]
            vec![lhs_dims[lhs_ndims - 2], rhs_dims[rhs_ndims - 1]]
        } else if lhs_ndims == 1 && rhs_ndims == 2 {
            // Vector-matrix multiplication: [k] x [k, n] -> [n]
            vec![rhs_dims[1]]
        } else if lhs_ndims == 2 && rhs_ndims == 1 {
            // Matrix-vector multiplication: [m, k] x [k] -> [m]
            vec![lhs_dims[0]]
        } else {
            // Fallback to original shape for unsupported cases
            lhs_dims.clone()
        };

        let result = torsh_tensor_zeros(output_dims.as_ptr(), output_dims.len());
        if result.is_null() {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            return std::ptr::null_mut();
        }

        // Perform the matrix multiplication
        let matmul_result = torsh_tensor_matmul(lhs_tensor, rhs_tensor, result);
        if matmul_result != TorshError::Success {
            torsh_tensor_free(lhs_tensor);
            torsh_tensor_free(rhs_tensor);
            torsh_tensor_free(result);
            return std::ptr::null_mut();
        }

        let result_mx = torsh_tensor_to_matlab(result);

        torsh_tensor_free(lhs_tensor);
        torsh_tensor_free(rhs_tensor);
        torsh_tensor_free(result);

        result_mx
    }
}

#[no_mangle]
pub extern "C" fn matlab_tensor_relu(mx_array: *const MxArray) -> *mut MxArray {
    unsafe {
        let tensor = matlab_to_torsh_tensor(mx_array);
        if tensor.is_null() {
            return std::ptr::null_mut();
        }

        // Get input tensor shape
        let mut temp_dims = vec![0usize; 16]; // max 16 dims
        let mut ndim = 0usize;
        let shape_result = torsh_tensor_shape(tensor, temp_dims.as_mut_ptr(), &mut ndim);

        if shape_result != TorshError::Success {
            torsh_tensor_free(tensor);
            return std::ptr::null_mut();
        }

        // Create output tensor with same shape as input
        let output_tensor = torsh_tensor_zeros(temp_dims.as_ptr(), ndim);
        if output_tensor.is_null() {
            torsh_tensor_free(tensor);
            return std::ptr::null_mut();
        }

        // Apply ReLU
        let relu_result = torsh_tensor_relu(tensor, output_tensor);
        if relu_result != TorshError::Success {
            torsh_tensor_free(tensor);
            torsh_tensor_free(output_tensor);
            return std::ptr::null_mut();
        }

        let result_mx = torsh_tensor_to_matlab(output_tensor);

        torsh_tensor_free(tensor);
        torsh_tensor_free(output_tensor);

        result_mx
    }
}

/// Utility function for error handling
fn error_and_return_null(message: &str) {
    unsafe {
        let c_msg = CString::new(message)
            .unwrap_or_else(|_| CString::new("Error creating error message").expect("static fallback string should not contain null bytes"));
        mexErrMsgIdAndTxt(
            CString::new("ToRSh:Error").expect("static string should not contain null bytes").as_ptr(),
            c_msg.as_ptr(),
        );
    }
}

/// MATLAB tensor creation utilities
#[no_mangle]
pub extern "C" fn matlab_tensor_zeros(dims_mx: *const MxArray) -> *mut MxArray {
    unsafe {
        if dims_mx.is_null() {
            error_and_return_null("Dimensions array is null");
            return std::ptr::null_mut();
        }

        let dims_data = mxGetPr(dims_mx);
        let ndims = mxGetM(dims_mx) * mxGetN(dims_mx);

        if dims_data.is_null() || ndims == 0 {
            error_and_return_null("Invalid dimensions array");
            return std::ptr::null_mut();
        }

        let dims_slice = slice::from_raw_parts(dims_data, ndims);
        let dims_usize: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        let tensor = torsh_tensor_zeros(dims_usize.as_ptr(), dims_usize.len());
        let result_mx = torsh_tensor_to_matlab(tensor);

        torsh_tensor_free(tensor);
        result_mx
    }
}

#[no_mangle]
pub extern "C" fn matlab_tensor_ones(dims_mx: *const MxArray) -> *mut MxArray {
    unsafe {
        if dims_mx.is_null() {
            error_and_return_null("Dimensions array is null");
            return std::ptr::null_mut();
        }

        let dims_data = mxGetPr(dims_mx);
        let ndims = mxGetM(dims_mx) * mxGetN(dims_mx);

        if dims_data.is_null() || ndims == 0 {
            error_and_return_null("Invalid dimensions array");
            return std::ptr::null_mut();
        }

        let dims_slice = slice::from_raw_parts(dims_data, ndims);
        let dims_usize: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

        let tensor = torsh_tensor_ones(dims_usize.as_ptr(), dims_usize.len());
        let result_mx = torsh_tensor_to_matlab(tensor);

        torsh_tensor_free(tensor);
        result_mx
    }
}

/// Neural network operations for MATLAB
#[no_mangle]
pub extern "C" fn matlab_linear_forward(
    input_mx: *const MxArray,
    weight_mx: *const MxArray,
    bias_mx: *const MxArray,
) -> *mut MxArray {
    unsafe {
        let input_tensor = matlab_to_torsh_tensor(input_mx);
        let weight_tensor = matlab_to_torsh_tensor(weight_mx);
        let bias_tensor = if bias_mx.is_null() {
            std::ptr::null_mut()
        } else {
            matlab_to_torsh_tensor(bias_mx)
        };

        if input_tensor.is_null() || weight_tensor.is_null() {
            torsh_tensor_free(input_tensor);
            torsh_tensor_free(weight_tensor);
            torsh_tensor_free(bias_tensor);
            return std::ptr::null_mut();
        }

        // Get weight tensor shape to determine input/output features
        let mut weight_shape = vec![0usize; 16]; // max 16 dims
        let mut weight_ndim = 0usize;
        let shape_result = torsh_tensor_shape(weight_tensor, weight_shape.as_mut_ptr(), &mut weight_ndim);

        if shape_result != TorshError::Success || weight_ndim < 2 {
            torsh_tensor_free(input_tensor);
            torsh_tensor_free(weight_tensor);
            torsh_tensor_free(bias_tensor);
            return std::ptr::null_mut();
        }

        let layer = torsh_linear_new(
            weight_shape[1], // input features
            weight_shape[0], // output features
            !bias_tensor.is_null(),
        );

        if layer.is_null() {
            torsh_tensor_free(input_tensor);
            torsh_tensor_free(weight_tensor);
            torsh_tensor_free(bias_tensor);
            return std::ptr::null_mut();
        }

        // Get input shape to determine output shape
        let mut input_ndims = torsh_tensor_ndim(input_tensor);
        let mut input_dims = vec![0usize; input_ndims];
        let input_shape_result =
            torsh_tensor_shape(input_tensor, input_dims.as_mut_ptr(), &mut input_ndims);

        if input_shape_result != TorshError::Success {
            torsh_tensor_free(input_tensor);
            torsh_tensor_free(weight_tensor);
            torsh_tensor_free(bias_tensor);
            torsh_linear_free(layer);
            return std::ptr::null_mut();
        }

        // Calculate output shape: input (batch_size, in_features) -> output (batch_size, out_features)
        let out_features = weight_shape[0]; // Use the already retrieved weight shape
        let mut output_dims = input_dims.clone();
        if input_ndims >= 2 {
            output_dims[input_ndims - 1] = out_features; // Change last dimension to out_features
        }

        // Create output tensor
        let result = torsh_tensor_zeros(output_dims.as_ptr(), input_ndims);
        if result.is_null() {
            torsh_tensor_free(input_tensor);
            torsh_tensor_free(weight_tensor);
            torsh_tensor_free(bias_tensor);
            torsh_linear_free(layer);
            return std::ptr::null_mut();
        }

        // Perform linear forward pass
        let forward_result = torsh_linear_forward(layer, input_tensor, result);
        if forward_result != TorshError::Success {
            torsh_tensor_free(input_tensor);
            torsh_tensor_free(weight_tensor);
            torsh_tensor_free(bias_tensor);
            torsh_tensor_free(result);
            torsh_linear_free(layer);
            return std::ptr::null_mut();
        }

        let result_mx = torsh_tensor_to_matlab(result);

        torsh_tensor_free(input_tensor);
        torsh_tensor_free(weight_tensor);
        torsh_tensor_free(bias_tensor);
        torsh_tensor_free(result);
        torsh_linear_free(layer);

        result_mx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matlab_bindings_compilation() {
        // Test that the module compiles correctly
        // In practice, MATLAB integration would require actual MEX compilation
        assert!(true);
    }
}
