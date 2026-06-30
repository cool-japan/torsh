//! Math, activation, reduction, convolution, and in-place optimizer operations
//! for the ToRSh C API tensor module.
//!
//! This file is included via `#[path]` from `tensor.rs`; all items from the
//! parent module are available through `use super::*;`.

use super::*;

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
// Softmax Activation
// =============================================================================

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
            set_last_error(
                "conv2d requires 4-D input [N,C,H,W] and weight [O,I,kH,kW]".to_string(),
            );
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
                let bias_val = bias_data
                    .as_ref()
                    .map(|b| b.get(oc).copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
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
                                    let w_idx =
                                        oc * in_ch * kh * kw + ic * kh * kw + khi * kw + kwi;
                                    acc += inp.data[inp_idx] * wgt.data[w_idx];
                                }
                            }
                        }
                        let out_idx =
                            n * out_ch * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
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
