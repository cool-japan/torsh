//! Tensor operations for C API
//!
//! This module provides all tensor-related functionality for the ToRSh C API,
//! including tensor creation, manipulation, mathematical operations, and activations.

use scirs2_core::legacy::rng;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::os::raw::{c_float, c_int, c_void};
use std::ptr;
use std::slice;
use std::sync::{Mutex, OnceLock};
use torsh_core::DType;

use super::types::{TorshDType, TorshError, TorshTensor};

// Global tensor storage
static TENSOR_STORE: OnceLock<Mutex<HashMap<usize, Box<TensorImpl>>>> = OnceLock::new();
static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

/// Internal tensor implementation
#[derive(Clone)]
pub(crate) struct TensorImpl {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// Get the global tensor store
pub(crate) fn get_tensor_store() -> &'static Mutex<HashMap<usize, Box<TensorImpl>>> {
    TENSOR_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Generate next unique ID
pub(crate) fn get_next_id() -> usize {
    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

/// Set last error message (imported from parent module)
pub(crate) fn set_last_error(message: String) {
    if let Ok(mut last_error) = super::get_last_error().lock() {
        *last_error = Some(message);
    }
}

// =============================================================================
// Tensor Creation Operations
// =============================================================================

/// Create a new tensor from data
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_new(
    data: *const c_void,
    shape: *const usize,
    ndim: usize,
    dtype: TorshDType,
) -> *mut TorshTensor {
    if data.is_null() || shape.is_null() || ndim == 0 {
        set_last_error("Invalid arguments to torsh_tensor_new".to_string());
        return ptr::null_mut();
    }

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let total_elements: usize = shape_slice.iter().product();

    let tensor_data = match dtype {
        TorshDType::F32 => {
            let data_slice = slice::from_raw_parts(data as *const f32, total_elements);
            data_slice.to_vec()
        }
        TorshDType::F64 => {
            let data_slice = slice::from_raw_parts(data as *const f64, total_elements);
            data_slice.iter().map(|&x| x as f32).collect()
        }
        TorshDType::I32 => {
            let data_slice = slice::from_raw_parts(data as *const i32, total_elements);
            data_slice.iter().map(|&x| x as f32).collect()
        }
        TorshDType::I64 => {
            let data_slice = slice::from_raw_parts(data as *const i64, total_elements);
            data_slice.iter().map(|&x| x as f32).collect()
        }
        TorshDType::U8 => {
            let data_slice = slice::from_raw_parts(data as *const u8, total_elements);
            data_slice.iter().map(|&x| x as f32).collect()
        }
    };

    let tensor_impl = TensorImpl {
        data: tensor_data,
        shape: shape_slice.to_vec(),
        dtype: dtype.into(),
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        set_last_error("Failed to store tensor".to_string());
        ptr::null_mut()
    }
}

/// Create tensor filled with zeros
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_zeros(shape: *const usize, ndim: usize) -> *mut TorshTensor {
    if shape.is_null() || ndim == 0 {
        set_last_error("Invalid shape parameters".to_string());
        return ptr::null_mut();
    }

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let total_size: usize = shape_slice.iter().product();

    let data = vec![0.0f32; total_size];
    let tensor_impl = TensorImpl {
        data,
        shape: shape_slice.to_vec(),
        dtype: DType::F32,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        set_last_error("Failed to store tensor".to_string());
        ptr::null_mut()
    }
}

/// Create tensor filled with ones
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_ones(shape: *const usize, ndim: usize) -> *mut TorshTensor {
    if shape.is_null() || ndim == 0 {
        set_last_error("Invalid shape parameters".to_string());
        return ptr::null_mut();
    }

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let total_size: usize = shape_slice.iter().product();

    let data = vec![1.0f32; total_size];
    let tensor_impl = TensorImpl {
        data,
        shape: shape_slice.to_vec(),
        dtype: DType::F32,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        set_last_error("Failed to store tensor".to_string());
        ptr::null_mut()
    }
}

/// Create tensor with random normal distribution
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_randn(shape: *const usize, ndim: usize) -> *mut TorshTensor {
    if shape.is_null() || ndim == 0 {
        set_last_error("Invalid shape parameters".to_string());
        return ptr::null_mut();
    }

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let total_size: usize = shape_slice.iter().product();

    // Use SciRS2 for high-quality random normal generation
    let mut data = Vec::with_capacity(total_size);
    let mut random_gen = rng();
    let normal_dist = Normal::new(0.0, 1.0).expect("valid normal distribution parameters");

    for _ in 0..total_size {
        data.push(random_gen.sample(&normal_dist) as f32);
    }

    let tensor_impl = TensorImpl {
        data,
        shape: shape_slice.to_vec(),
        dtype: DType::F32,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        set_last_error("Failed to store tensor".to_string());
        ptr::null_mut()
    }
}

/// Random tensor generation
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_rand(shape: *const usize, ndim: usize) -> *mut TorshTensor {
    if shape.is_null() || ndim == 0 {
        return ptr::null_mut();
    }

    let shape_slice = slice::from_raw_parts(shape, ndim);
    let total_elements: usize = shape_slice.iter().product();

    let mut data = Vec::with_capacity(total_elements);
    let mut random_gen = rng();
    let uniform_dist = Uniform::new(0.0, 1.0).expect("valid uniform distribution parameters");
    for _ in 0..total_elements {
        data.push(random_gen.sample(&uniform_dist) as f32);
    }

    let tensor_impl = TensorImpl {
        data,
        shape: shape_slice.to_vec(),
        dtype: DType::F32,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        ptr::null_mut()
    }
}

/// Create scalar tensor
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_scalar(value: c_float) -> *mut TorshTensor {
    let data = vec![value];
    let tensor_impl = TensorImpl {
        data,
        shape: vec![1],
        dtype: DType::F32,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        set_last_error("Failed to store tensor".to_string());
        ptr::null_mut()
    }
}

/// Create a tensor from raw data array
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_from_data(
    data: *const c_float,
    data_len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut TorshTensor {
    if data.is_null() || shape.is_null() || ndim == 0 {
        return ptr::null_mut();
    }

    let data_slice = slice::from_raw_parts(data, data_len);
    let shape_slice = slice::from_raw_parts(shape, ndim);

    let tensor_impl = TensorImpl {
        data: data_slice.to_vec(),
        shape: shape_slice.to_vec(),
        dtype: DType::F32,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        ptr::null_mut()
    }
}

// =============================================================================
// Tensor Access Operations
// =============================================================================

/// Get tensor data
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_data(tensor: *const TorshTensor) -> *const c_void {
    if tensor.is_null() {
        return ptr::null();
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            return tensor_impl.data.as_ptr() as *const c_void;
        }
    }

    set_last_error("Invalid tensor handle".to_string());
    ptr::null()
}

/// Get tensor shape
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_shape(
    tensor: *const TorshTensor,
    shape: *mut usize,
    ndim: *mut usize,
) -> TorshError {
    if tensor.is_null() || shape.is_null() || ndim.is_null() {
        return TorshError::InvalidArgument;
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            *ndim = tensor_impl.shape.len();
            let shape_slice = slice::from_raw_parts_mut(shape, tensor_impl.shape.len());
            shape_slice.copy_from_slice(&tensor_impl.shape);
            return TorshError::Success;
        }
    }

    set_last_error("Invalid tensor handle".to_string());
    TorshError::InvalidArgument
}

/// Get tensor data type
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_dtype(tensor: *const TorshTensor) -> TorshDType {
    if tensor.is_null() {
        return TorshDType::F32; // Default fallback
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            return tensor_impl.dtype.into();
        }
    }

    set_last_error("Invalid tensor handle".to_string());
    TorshDType::F32 // Default fallback
}

/// Get the number of elements in a tensor
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_numel(tensor: *const TorshTensor) -> usize {
    if tensor.is_null() {
        return 0;
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            return tensor_impl.shape.iter().product();
        }
    }
    0
}

/// Get the number of dimensions of a tensor
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_ndim(tensor: *const TorshTensor) -> usize {
    if tensor.is_null() {
        return 0;
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            return tensor_impl.shape.len();
        }
    }
    0
}

/// Get tensor size (number of elements)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_size(tensor: *const TorshTensor) -> usize {
    torsh_tensor_numel(tensor)
}

// =============================================================================
// Basic Mathematical Operations
// =============================================================================

/// Add two tensors
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_add(
    a: *const TorshTensor,
    b: *const TorshTensor,
    out: *mut TorshTensor,
) -> TorshError {
    if a.is_null() || b.is_null() || out.is_null() {
        return TorshError::InvalidArgument;
    }

    let id_a = a as usize;
    let id_b = b as usize;
    let id_out = out as usize;

    if let Ok(mut store) = get_tensor_store().lock() {
        let tensor_a = store.get(&id_a).cloned();
        let tensor_b = store.get(&id_b).cloned();

        if let (Some(a_impl), Some(b_impl)) = (tensor_a, tensor_b) {
            if a_impl.shape != b_impl.shape {
                set_last_error("Shape mismatch in tensor addition".to_string());
                return TorshError::ShapeMismatch;
            }

            let result_data: Vec<f32> = a_impl
                .data
                .iter()
                .zip(b_impl.data.iter())
                .map(|(&x, &y)| x + y)
                .collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: a_impl.shape.clone(),
                dtype: a_impl.dtype,
            };

            store.insert(id_out, Box::new(result_tensor));
            return TorshError::Success;
        }
    }

    set_last_error("Invalid tensor handles".to_string());
    TorshError::InvalidArgument
}

/// Subtract two tensors
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_sub(
    a: *const TorshTensor,
    b: *const TorshTensor,
    out: *mut TorshTensor,
) -> TorshError {
    if a.is_null() || b.is_null() || out.is_null() {
        return TorshError::InvalidArgument;
    }

    let id_a = a as usize;
    let id_b = b as usize;
    let id_out = out as usize;

    if let Ok(mut store) = get_tensor_store().lock() {
        let tensor_a = store.get(&id_a).cloned();
        let tensor_b = store.get(&id_b).cloned();

        if let (Some(a_impl), Some(b_impl)) = (tensor_a, tensor_b) {
            if a_impl.shape != b_impl.shape {
                set_last_error("Shape mismatch in tensor subtraction".to_string());
                return TorshError::ShapeMismatch;
            }

            let result_data: Vec<f32> = a_impl
                .data
                .iter()
                .zip(b_impl.data.iter())
                .map(|(&x, &y)| x - y)
                .collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: a_impl.shape.clone(),
                dtype: a_impl.dtype,
            };

            store.insert(id_out, Box::new(result_tensor));
            return TorshError::Success;
        }
    }

    set_last_error("Invalid tensor handles".to_string());
    TorshError::InvalidArgument
}

/// Multiply two tensors
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_mul(
    a: *const TorshTensor,
    b: *const TorshTensor,
    out: *mut TorshTensor,
) -> TorshError {
    if a.is_null() || b.is_null() || out.is_null() {
        return TorshError::InvalidArgument;
    }

    let id_a = a as usize;
    let id_b = b as usize;
    let id_out = out as usize;

    if let Ok(mut store) = get_tensor_store().lock() {
        let tensor_a = store.get(&id_a).cloned();
        let tensor_b = store.get(&id_b).cloned();

        if let (Some(a_impl), Some(b_impl)) = (tensor_a, tensor_b) {
            if a_impl.shape != b_impl.shape {
                set_last_error("Shape mismatch in tensor multiplication".to_string());
                return TorshError::ShapeMismatch;
            }

            let result_data: Vec<f32> = a_impl
                .data
                .iter()
                .zip(b_impl.data.iter())
                .map(|(&x, &y)| x * y)
                .collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: a_impl.shape.clone(),
                dtype: a_impl.dtype,
            };

            store.insert(id_out, Box::new(result_tensor));
            return TorshError::Success;
        }
    }

    set_last_error("Invalid tensor handles".to_string());
    TorshError::InvalidArgument
}

/// Element-wise tensor multiplication (alias for torsh_tensor_mul)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_multiply(
    a: *const TorshTensor,
    b: *const TorshTensor,
    result: *mut TorshTensor,
) -> TorshError {
    torsh_tensor_mul(a, b, result)
}

/// Matrix multiplication
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_matmul(
    a: *const TorshTensor,
    b: *const TorshTensor,
    out: *mut TorshTensor,
) -> TorshError {
    if a.is_null() || b.is_null() || out.is_null() {
        return TorshError::InvalidArgument;
    }

    let id_a = a as usize;
    let id_b = b as usize;
    let id_out = out as usize;

    if let Ok(mut store) = get_tensor_store().lock() {
        let tensor_a = store.get(&id_a).cloned();
        let tensor_b = store.get(&id_b).cloned();

        if let (Some(a_impl), Some(b_impl)) = (tensor_a, tensor_b) {
            if a_impl.shape.len() != 2 || b_impl.shape.len() != 2 {
                set_last_error("Matrix multiplication requires 2D tensors".to_string());
                return TorshError::InvalidArgument;
            }

            let (m, k) = (a_impl.shape[0], a_impl.shape[1]);
            let (k2, n) = (b_impl.shape[0], b_impl.shape[1]);

            if k != k2 {
                set_last_error("Inner dimensions must match for matrix multiplication".to_string());
                return TorshError::ShapeMismatch;
            }

            let mut result_data = vec![0.0; m * n];

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += a_impl.data[i * k + l] * b_impl.data[l * n + j];
                    }
                    result_data[i * n + j] = sum;
                }
            }

            let result_tensor = TensorImpl {
                data: result_data,
                shape: vec![m, n],
                dtype: a_impl.dtype,
            };

            store.insert(id_out, Box::new(result_tensor));
            return TorshError::Success;
        }
    }

    set_last_error("Invalid tensor handles".to_string());
    TorshError::InvalidArgument
}

// =============================================================================
// Scalar Operations
// =============================================================================

/// Scalar addition: tensor + scalar
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_add_scalar(
    tensor: *mut TorshTensor,
    scalar: c_float,
) -> *mut TorshTensor {
    if tensor.is_null() {
        set_last_error("Null tensor pointer for scalar addition".to_string());
        return ptr::null_mut();
    }

    let tensor_id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&tensor_id) {
            // Create result tensor by adding scalar to all elements
            let result_data: Vec<f32> = tensor_impl.data.iter().map(|&x| x + scalar).collect();

            let result_impl = TensorImpl {
                data: result_data,
                shape: tensor_impl.shape.clone(),
                dtype: tensor_impl.dtype,
            };

            let result_id = get_next_id();

            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_impl));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to perform scalar addition".to_string());
    ptr::null_mut()
}

/// Scalar multiplication: tensor * scalar
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_mul_scalar(
    tensor: *mut TorshTensor,
    scalar: c_float,
) -> *mut TorshTensor {
    if tensor.is_null() {
        set_last_error("Null tensor pointer for scalar multiplication".to_string());
        return ptr::null_mut();
    }

    let tensor_id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&tensor_id) {
            // Create result tensor by multiplying all elements by scalar
            let result_data: Vec<f32> = tensor_impl.data.iter().map(|&x| x * scalar).collect();

            let result_impl = TensorImpl {
                data: result_data,
                shape: tensor_impl.shape.clone(),
                dtype: tensor_impl.dtype,
            };

            let result_id = get_next_id();

            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_impl));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to perform scalar multiplication".to_string());
    ptr::null_mut()
}

/// Tensor subtraction (scalar)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_sub_scalar(
    tensor: *const TorshTensor,
    scalar: c_float,
) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let result_data: Vec<f32> = tensor_impl.data.iter().map(|x| x - scalar).collect();

            let result_impl = TensorImpl {
                data: result_data,
                shape: tensor_impl.shape.clone(),
                dtype: tensor_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

/// Tensor division (scalar)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_div_scalar(
    tensor: *const TorshTensor,
    scalar: c_float,
) -> *mut TorshTensor {
    if tensor.is_null() || scalar == 0.0 {
        return ptr::null_mut();
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let result_data: Vec<f32> = tensor_impl.data.iter().map(|x| x / scalar).collect();

            let result_impl = TensorImpl {
                data: result_data,
                shape: tensor_impl.shape.clone(),
                dtype: tensor_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

// =============================================================================
// Activation Functions
// =============================================================================

/// Apply ReLU activation
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_relu(
    input: *const TorshTensor,
    out: *mut TorshTensor,
) -> TorshError {
    if input.is_null() || out.is_null() {
        return TorshError::InvalidArgument;
    }

    let id_input = input as usize;
    let id_out = out as usize;

    if let Ok(mut store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id_input).cloned() {
            let result_data: Vec<f32> = input_impl.data.iter().map(|&x| x.max(0.0)).collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            store.insert(id_out, Box::new(result_tensor));
            return TorshError::Success;
        }
    }

    set_last_error("Invalid tensor handle".to_string());
    TorshError::InvalidArgument
}

/// Apply sigmoid activation (for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_sigmoid(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer".to_string());
        return ptr::null_mut();
    }

    let id_input = input as usize;

    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id_input) {
            let result_data: Vec<f32> = input_impl
                .data
                .iter()
                .map(|&x| 1.0 / (1.0 + (-x).exp()))
                .collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_tensor));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to apply sigmoid".to_string());
    ptr::null_mut()
}

/// Apply tanh activation (for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_tanh(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer".to_string());
        return ptr::null_mut();
    }

    let id_input = input as usize;

    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id_input) {
            let result_data: Vec<f32> = input_impl.data.iter().map(|&x| x.tanh()).collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_tensor));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to apply tanh".to_string());
    ptr::null_mut()
}

// =============================================================================
// Mathematical Functions
// =============================================================================

/// Apply exp function (for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_exp(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer".to_string());
        return ptr::null_mut();
    }

    let id_input = input as usize;

    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id_input) {
            let result_data: Vec<f32> = input_impl.data.iter().map(|&x| x.exp()).collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_tensor));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to apply exp".to_string());
    ptr::null_mut()
}

/// Apply log function (for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_log(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer".to_string());
        return ptr::null_mut();
    }

    let id_input = input as usize;

    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id_input) {
            let result_data: Vec<f32> = input_impl.data.iter().map(|&x| x.ln()).collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_tensor));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to apply log".to_string());
    ptr::null_mut()
}

/// Apply sqrt function (for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_sqrt(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer".to_string());
        return ptr::null_mut();
    }

    let id_input = input as usize;

    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id_input) {
            let result_data: Vec<f32> = input_impl.data.iter().map(|&x| x.sqrt()).collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_tensor));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to apply sqrt".to_string());
    ptr::null_mut()
}

/// Apply abs function (for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_abs(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer".to_string());
        return ptr::null_mut();
    }

    let id_input = input as usize;

    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id_input) {
            let result_data: Vec<f32> = input_impl.data.iter().map(|&x| x.abs()).collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_tensor));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to apply abs".to_string());
    ptr::null_mut()
}

/// Trigonometric sine function
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_sin(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        return ptr::null_mut();
    }

    let id = input as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id) {
            let result_data: Vec<f32> = input_impl.data.iter().map(|x| x.sin()).collect();

            let result_impl = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

/// Trigonometric cosine function
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_cos(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        return ptr::null_mut();
    }

    let id = input as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id) {
            let result_data: Vec<f32> = input_impl.data.iter().map(|x| x.cos()).collect();

            let result_impl = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

/// Trigonometric tangent function
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_tan(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        return ptr::null_mut();
    }

    let id = input as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id) {
            let result_data: Vec<f32> = input_impl.data.iter().map(|x| x.tan()).collect();

            let result_impl = TensorImpl {
                data: result_data,
                shape: input_impl.shape.clone(),
                dtype: input_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

// =============================================================================
// Tensor Manipulation
// =============================================================================

/// Transpose tensor (for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_transpose(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer".to_string());
        return ptr::null_mut();
    }

    let id_input = input as usize;

    if let Ok(store) = get_tensor_store().lock() {
        if let Some(input_impl) = store.get(&id_input) {
            if input_impl.shape.len() != 2 {
                set_last_error("Transpose only supported for 2D tensors".to_string());
                return ptr::null_mut();
            }

            let rows = input_impl.shape[0];
            let cols = input_impl.shape[1];
            let mut result_data = vec![0.0f32; rows * cols];

            for i in 0..rows {
                for j in 0..cols {
                    result_data[j * rows + i] = input_impl.data[i * cols + j];
                }
            }

            let result_tensor = TensorImpl {
                data: result_data,
                shape: vec![cols, rows],
                dtype: input_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_tensor));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to transpose tensor".to_string());
    ptr::null_mut()
}

// =============================================================================
// Reduction Operations
// =============================================================================

/// Sum all elements in tensor
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_sum_all(tensor: *const TorshTensor) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let sum: f32 = tensor_impl.data.iter().sum();

            let result_impl = TensorImpl {
                data: vec![sum],
                shape: vec![1],
                dtype: tensor_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

/// Sum along a dimension
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_sum_dim(
    tensor: *const TorshTensor,
    _dim: c_int,
) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    // For simplicity, just return sum of all elements for now
    torsh_tensor_sum_all(tensor)
}

/// Mean of all elements
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_mean_all(tensor: *const TorshTensor) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let mean: f32 = tensor_impl.data.iter().sum::<f32>() / tensor_impl.data.len() as f32;

            let result_impl = TensorImpl {
                data: vec![mean],
                shape: vec![1],
                dtype: tensor_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

/// Mean along a dimension
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_mean_dim(
    tensor: *const TorshTensor,
    _dim: c_int,
) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    // For simplicity, just return mean of all elements for now
    torsh_tensor_mean_all(tensor)
}

/// Maximum of all elements
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_max_all(tensor: *const TorshTensor) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let max: f32 = tensor_impl
                .data
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            let result_impl = TensorImpl {
                data: vec![max],
                shape: vec![1],
                dtype: tensor_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

/// Maximum along a dimension
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_max_dim(
    tensor: *const TorshTensor,
    _dim: c_int,
) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    // For simplicity, just return max of all elements for now
    torsh_tensor_max_all(tensor)
}

/// Minimum of all elements
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_min_all(tensor: *const TorshTensor) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let min: f32 = tensor_impl
                .data
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);

            let result_impl = TensorImpl {
                data: vec![min],
                shape: vec![1],
                dtype: tensor_impl.dtype,
            };

            let id = get_next_id();

            drop(store);
            if let Ok(mut store) = get_tensor_store().lock() {
                store.insert(id, Box::new(result_impl));
                return id as *mut TorshTensor;
            }
        }
    }
    ptr::null_mut()
}

/// Minimum along a dimension
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_min_dim(
    tensor: *const TorshTensor,
    _dim: c_int,
) -> *mut TorshTensor {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    // For simplicity, just return min of all elements for now
    torsh_tensor_min_all(tensor)
}

// =============================================================================
// Additional Creation Operations
// =============================================================================

/// Create an identity matrix of size n×n
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_eye(n: usize) -> *mut TorshTensor {
    if n == 0 {
        set_last_error("Size n must be > 0 for identity matrix".to_string());
        return ptr::null_mut();
    }

    let total_size = n * n;
    let mut data = vec![0.0f32; total_size];
    for i in 0..n {
        data[i * n + i] = 1.0f32;
    }

    let tensor_impl = TensorImpl {
        data,
        shape: vec![n, n],
        dtype: DType::F32,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        set_last_error("Failed to store tensor".to_string());
        ptr::null_mut()
    }
}

/// Create a 1D tensor with linearly spaced values from start to end (inclusive)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_linspace(
    start: f64,
    end: f64,
    steps: usize,
) -> *mut TorshTensor {
    if steps == 0 {
        set_last_error("steps must be > 0 for linspace".to_string());
        return ptr::null_mut();
    }

    let data = if steps == 1 {
        vec![start as f32]
    } else {
        let step_size = (end - start) / (steps - 1) as f64;
        (0..steps)
            .map(|i| (start + step_size * i as f64) as f32)
            .collect()
    };

    let tensor_impl = TensorImpl {
        data,
        shape: vec![steps],
        dtype: DType::F32,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_tensor_store().lock() {
        store.insert(id, Box::new(tensor_impl));
        id as *mut TorshTensor
    } else {
        set_last_error("Failed to store tensor".to_string());
        ptr::null_mut()
    }
}

/// Reshape a tensor to a new shape (total elements must be the same)
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_reshape(
    tensor: *const TorshTensor,
    shape: *const usize,
    ndim: usize,
) -> *mut TorshTensor {
    if tensor.is_null() || shape.is_null() || ndim == 0 {
        set_last_error("Invalid arguments to torsh_tensor_reshape".to_string());
        return ptr::null_mut();
    }

    let new_shape = slice::from_raw_parts(shape, ndim);
    let new_total: usize = new_shape.iter().product();

    let id = tensor as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let old_total: usize = tensor_impl.shape.iter().product();
            if old_total != new_total {
                set_last_error(format!(
                    "Cannot reshape tensor of {} elements to shape with {} elements",
                    old_total, new_total
                ));
                return ptr::null_mut();
            }

            let result_impl = TensorImpl {
                data: tensor_impl.data.clone(),
                shape: new_shape.to_vec(),
                dtype: tensor_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_impl));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to reshape tensor".to_string());
    ptr::null_mut()
}

/// Apply softmax activation along the given dimension
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_softmax(
    input: *const TorshTensor,
    dim: i64,
) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer for softmax".to_string());
        return ptr::null_mut();
    }

    let id = input as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let shape = &tensor_impl.shape;
            let ndim = shape.len() as i64;

            // Normalise negative dim
            let axis = if dim < 0 { ndim + dim } else { dim };
            if axis < 0 || axis >= ndim {
                set_last_error(format!(
                    "Softmax dim {} out of range for tensor with {} dimensions",
                    dim, ndim
                ));
                return ptr::null_mut();
            }
            let axis = axis as usize;

            let total: usize = shape.iter().product();
            let mut result_data = tensor_impl.data.clone();

            // Compute outer, inner, and axis size for stride-based iteration
            let axis_size = shape[axis];
            let inner: usize = shape[axis + 1..].iter().product();
            let outer: usize = shape[..axis].iter().product();

            for o in 0..outer {
                for i in 0..inner {
                    // Numerically-stable softmax: subtract max before exp
                    let mut max_val = f32::NEG_INFINITY;
                    for a in 0..axis_size {
                        let idx = o * axis_size * inner + a * inner + i;
                        if idx < total && result_data[idx] > max_val {
                            max_val = result_data[idx];
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for a in 0..axis_size {
                        let idx = o * axis_size * inner + a * inner + i;
                        if idx < total {
                            result_data[idx] = (result_data[idx] - max_val).exp();
                            sum_exp += result_data[idx];
                        }
                    }
                    if sum_exp > 0.0 {
                        for a in 0..axis_size {
                            let idx = o * axis_size * inner + a * inner + i;
                            if idx < total {
                                result_data[idx] /= sum_exp;
                            }
                        }
                    }
                }
            }

            let result_impl = TensorImpl {
                data: result_data,
                shape: shape.clone(),
                dtype: tensor_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(result_impl));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to apply softmax".to_string());
    ptr::null_mut()
}

/// Deep-copy (clone) a tensor
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_clone(input: *const TorshTensor) -> *mut TorshTensor {
    if input.is_null() {
        set_last_error("Null tensor pointer for clone".to_string());
        return ptr::null_mut();
    }

    let id = input as usize;
    if let Ok(store) = get_tensor_store().lock() {
        if let Some(tensor_impl) = store.get(&id) {
            let cloned = TensorImpl {
                data: tensor_impl.data.clone(),
                shape: tensor_impl.shape.clone(),
                dtype: tensor_impl.dtype,
            };

            let result_id = get_next_id();
            drop(store);
            if let Ok(mut result_store) = get_tensor_store().lock() {
                result_store.insert(result_id, Box::new(cloned));
                return result_id as *mut TorshTensor;
            }
        }
    }

    set_last_error("Failed to clone tensor".to_string());
    ptr::null_mut()
}

/// Detach a tensor from the autograd graph (returns a copy with identical data)
///
/// In the current pure-Rust backend there is no autograd graph, so this is
/// semantically equivalent to `torsh_tensor_clone` — it returns a new tensor
/// handle containing the same data and shape.
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_detach(input: *const TorshTensor) -> *mut TorshTensor {
    torsh_tensor_clone(input)
}

// =============================================================================
// Element-wise tensor-tensor division
// =============================================================================

/// Divide two tensors element-wise (a / b).
///
/// Both tensors must have the same shape.  The result is written into the
/// pre-allocated `out` tensor.
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_div(
    a: *const TorshTensor,
    b: *const TorshTensor,
    out: *mut TorshTensor,
) -> TorshError {
    if a.is_null() || b.is_null() || out.is_null() {
        return TorshError::InvalidArgument;
    }

    let id_a = a as usize;
    let id_b = b as usize;
    let id_out = out as usize;

    if let Ok(mut store) = get_tensor_store().lock() {
        let tensor_a = store.get(&id_a).cloned();
        let tensor_b = store.get(&id_b).cloned();

        if let (Some(a_impl), Some(b_impl)) = (tensor_a, tensor_b) {
            if a_impl.shape != b_impl.shape {
                set_last_error("Shape mismatch in tensor division".to_string());
                return TorshError::ShapeMismatch;
            }

            let result_data: Vec<f32> = a_impl
                .data
                .iter()
                .zip(b_impl.data.iter())
                .map(|(&x, &y)| x / y)
                .collect();

            let result_tensor = TensorImpl {
                data: result_data,
                shape: a_impl.shape.clone(),
                dtype: a_impl.dtype,
            };

            store.insert(id_out, Box::new(result_tensor));
            return TorshError::Success;
        }
    }

    set_last_error("Invalid tensor handles".to_string());
    TorshError::InvalidArgument
}

// =============================================================================
// 2-D Convolution
// =============================================================================

/// 2-D convolution: NCHW input, OIHW weight, optional bias.
///
/// `bias` may be `null`; when present it must contain exactly `out_channels`
/// elements (one scalar per output channel).
///
/// Returns a new tensor of shape `[N, out_channels, H_out, W_out]`, or null
/// on invalid arguments.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn torsh_tensor_conv2d(
    input: *const TorshTensor,
    weight: *const TorshTensor,
    bias: *const TorshTensor, // nullable
    stride: usize,
    padding: usize,
) -> *mut TorshTensor {
    if input.is_null() || weight.is_null() {
        set_last_error("Null tensor pointer in conv2d".to_string());
        return ptr::null_mut();
    }

    let stride = stride.max(1);

    let id_in = input as usize;
    let id_w = weight as usize;

    if let Ok(store) = get_tensor_store().lock() {
        let inp = match store.get(&id_in) {
            Some(t) => t.clone(),
            None => {
                set_last_error("Invalid input tensor handle in conv2d".to_string());
                return ptr::null_mut();
            }
        };
        let wgt = match store.get(&id_w) {
            Some(t) => t.clone(),
            None => {
                set_last_error("Invalid weight tensor handle in conv2d".to_string());
                return ptr::null_mut();
            }
        };

        // Validate shapes
        if inp.shape.len() != 4 || wgt.shape.len() != 4 {
            set_last_error("conv2d requires 4-D input [N,C,H,W] and weight [O,I,kH,kW]".to_string());
            return ptr::null_mut();
        }

        let (n_batch, in_ch, h_in, w_in) = (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]);
        let (out_ch, in_ch_w, kh, kw) = (wgt.shape[0], wgt.shape[1], wgt.shape[2], wgt.shape[3]);

        if in_ch != in_ch_w {
            set_last_error(format!(
                "conv2d input channels {} != weight in-channels {}",
                in_ch, in_ch_w
            ));
            return ptr::null_mut();
        }

        let h_out = (h_in + 2 * padding).saturating_sub(kh) / stride + 1;
        let w_out = (w_in + 2 * padding).saturating_sub(kw) / stride + 1;

        // Optional bias: clone it out before dropping the lock
        let bias_data: Option<Vec<f32>> = if !bias.is_null() {
            let id_b = bias as usize;
            store.get(&id_b).map(|b| b.data.clone())
        } else {
            None
        };

        let total = n_batch * out_ch * h_out * w_out;
        let mut out_data = vec![0.0f32; total];

        for n in 0..n_batch {
            for oc in 0..out_ch {
                let bias_val = bias_data.as_ref().map(|b| b.get(oc).copied().unwrap_or(0.0)).unwrap_or(0.0);
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = bias_val;
                        for ic in 0..in_ch {
                            for khi in 0..kh {
                                let ih = oh * stride + khi;
                                if ih < padding || ih >= h_in + padding {
                                    continue;
                                }
                                let ih_real = ih - padding;
                                for kwi in 0..kw {
                                    let iw = ow * stride + kwi;
                                    if iw < padding || iw >= w_in + padding {
                                        continue;
                                    }
                                    let iw_real = iw - padding;
                                    let inp_idx = n * in_ch * h_in * w_in
                                        + ic * h_in * w_in
                                        + ih_real * w_in
                                        + iw_real;
                                    let w_idx = oc * in_ch * kh * kw
                                        + ic * kh * kw
                                        + khi * kw
                                        + kwi;
                                    acc += inp.data[inp_idx] * wgt.data[w_idx];
                                }
                            }
                        }
                        let out_idx = n * out_ch * h_out * w_out
                            + oc * h_out * w_out
                            + oh * w_out
                            + ow;
                        out_data[out_idx] = acc;
                    }
                }
            }
        }

        let result_impl = TensorImpl {
            data: out_data,
            shape: vec![n_batch, out_ch, h_out, w_out],
            dtype: inp.dtype,
        };

        let result_id = get_next_id();
        drop(store);
        if let Ok(mut result_store) = get_tensor_store().lock() {
            result_store.insert(result_id, Box::new(result_impl));
            return result_id as *mut TorshTensor;
        }
    }

    set_last_error("Failed to execute conv2d".to_string());
    ptr::null_mut()
}

// =============================================================================
// In-place optimizer primitives
// =============================================================================

/// In-place axpy: `target += alpha * source`.
///
/// Mutates `target` in place; returns `TorshError::Success` on success.
/// Both tensors must have the same total number of elements.
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_axpy_inplace(
    target: *mut TorshTensor,
    source: *const TorshTensor,
    alpha: f32,
) -> TorshError {
    if target.is_null() || source.is_null() {
        return TorshError::InvalidArgument;
    }

    let id_t = target as usize;
    let id_s = source as usize;

    if let Ok(mut store) = get_tensor_store().lock() {
        let src_data: Option<Vec<f32>> = store.get(&id_s).map(|s| s.data.clone());

        if let (Some(src), Some(tgt)) = (src_data, store.get_mut(&id_t)) {
            if tgt.data.len() != src.len() {
                set_last_error("axpy_inplace: element count mismatch".to_string());
                return TorshError::ShapeMismatch;
            }
            for (t, s) in tgt.data.iter_mut().zip(src.iter()) {
                *t += alpha * s;
            }
            return TorshError::Success;
        }
    }

    set_last_error("Invalid tensor handles in axpy_inplace".to_string());
    TorshError::InvalidArgument
}

/// In-place Adam optimizer step.
///
/// Updates `param`, `m`, and `v` in place following the standard Adam rule:
///   m   = beta1 * m + (1 - beta1) * grad
///   v   = beta2 * v + (1 - beta2) * grad²
///   p   -= lr * (m / (1 - beta1^step)) / (sqrt(v / (1 - beta2^step)) + eps)
///
/// All four tensors must have the same number of elements.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn torsh_tensor_adam_step_inplace(
    param: *mut TorshTensor,
    grad: *const TorshTensor,
    m: *mut TorshTensor,
    v: *mut TorshTensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: i64,
) -> TorshError {
    if param.is_null() || grad.is_null() || m.is_null() || v.is_null() {
        return TorshError::InvalidArgument;
    }

    let id_p = param as usize;
    let id_g = grad as usize;
    let id_m = m as usize;
    let id_v = v as usize;

    let bias_correction1 = 1.0 - beta1.powi(step as i32);
    let bias_correction2 = 1.0 - beta2.powi(step as i32);

    if let Ok(mut store) = get_tensor_store().lock() {
        let grad_data: Option<Vec<f32>> = store.get(&id_g).map(|g| g.data.clone());

        if let Some(g) = grad_data {
            // Borrow m, v, param mutably in sequence.
            if let Some(m_impl) = store.get_mut(&id_m) {
                if m_impl.data.len() != g.len() {
                    set_last_error("adam_step: element count mismatch (m)".to_string());
                    return TorshError::ShapeMismatch;
                }
                for (mi, &gi) in m_impl.data.iter_mut().zip(g.iter()) {
                    *mi = beta1 * *mi + (1.0 - beta1) * gi;
                }
            } else {
                set_last_error("Invalid m handle in adam_step".to_string());
                return TorshError::InvalidArgument;
            }

            if let Some(v_impl) = store.get_mut(&id_v) {
                if v_impl.data.len() != g.len() {
                    set_last_error("adam_step: element count mismatch (v)".to_string());
                    return TorshError::ShapeMismatch;
                }
                for (vi, &gi) in v_impl.data.iter_mut().zip(g.iter()) {
                    *vi = beta2 * *vi + (1.0 - beta2) * gi * gi;
                }
            } else {
                set_last_error("Invalid v handle in adam_step".to_string());
                return TorshError::InvalidArgument;
            }

            // Clone m and v for the param update (can't borrow both m and param mutably)
            let m_snap: Option<Vec<f32>> = store.get(&id_m).map(|mi| mi.data.clone());
            let v_snap: Option<Vec<f32>> = store.get(&id_v).map(|vi| vi.data.clone());

            if let (Some(m_d), Some(v_d)) = (m_snap, v_snap) {
                if let Some(p_impl) = store.get_mut(&id_p) {
                    if p_impl.data.len() != m_d.len() {
                        set_last_error("adam_step: element count mismatch (param)".to_string());
                        return TorshError::ShapeMismatch;
                    }
                    for ((pi, mi), vi) in p_impl.data.iter_mut().zip(m_d.iter()).zip(v_d.iter()) {
                        let m_hat = mi / bias_correction1;
                        let v_hat = vi / bias_correction2;
                        *pi -= lr * m_hat / (v_hat.sqrt() + eps);
                    }
                    return TorshError::Success;
                }
            }
        }
    }

    set_last_error("Failed to execute adam_step_inplace".to_string());
    TorshError::InvalidArgument
}

// =============================================================================
// Memory Management
// =============================================================================

/// Free a tensor
#[no_mangle]
pub unsafe extern "C" fn torsh_tensor_free(tensor: *mut TorshTensor) {
    if !tensor.is_null() {
        let id = tensor as usize;
        if let Ok(mut store) = get_tensor_store().lock() {
            store.remove(&id);
        }
    }
}

/// Clear all tensors from storage (for cleanup)
pub(crate) fn clear_tensor_store() {
    if let Ok(mut store) = get_tensor_store().lock() {
        store.clear();
    }
}
