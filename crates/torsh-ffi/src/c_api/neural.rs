//! Neural network operations for C API
//!
//! This module provides neural network functionality for the ToRSh C API,
//! including layer creation, forward passes, and module management.

use std::collections::HashMap;
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use super::tensor::{get_next_id, get_tensor_store, TensorImpl};
use super::types::{TorshError, TorshModule};

// Global module storage
static MODULE_STORE: OnceLock<Mutex<HashMap<usize, Box<ModuleImpl>>>> = OnceLock::new();

/// Internal module implementation
pub(crate) struct ModuleImpl {
    pub module_type: String,
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
    pub weight: Vec<f32>,
    pub bias_data: Option<Vec<f32>>,
}

/// Get the global module store
pub(crate) fn get_module_store() -> &'static Mutex<HashMap<usize, Box<ModuleImpl>>> {
    MODULE_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Set last error message (imported from parent module)
pub(crate) fn set_last_error(message: String) {
    if let Ok(mut last_error) = super::get_last_error().lock() {
        *last_error = Some(message);
    }
}

// =============================================================================
// Linear Layer Operations
// =============================================================================

/// Create a linear layer
#[no_mangle]
pub unsafe extern "C" fn torsh_linear_new(
    in_features: usize,
    out_features: usize,
    bias: bool,
) -> *mut TorshModule {
    if in_features == 0 || out_features == 0 {
        set_last_error("Invalid feature dimensions".to_string());
        return ptr::null_mut();
    }

    // Initialize weights with Xavier initialization
    let weight_count = in_features * out_features;
    let weight_std = (2.0 / in_features as f32).sqrt();
    let mut weight = Vec::with_capacity(weight_count);

    // Simple random initialization (in real implementation would use proper RNG)
    let mut seed = 12345u64;
    for _ in 0..weight_count {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let random_val = (seed as f32) / (u64::MAX as f32);
        weight.push((random_val - 0.5) * 2.0 * weight_std);
    }

    let bias_data = if bias {
        Some(vec![0.0; out_features])
    } else {
        None
    };

    let module_impl = ModuleImpl {
        module_type: "linear".to_string(),
        in_features,
        out_features,
        bias,
        weight,
        bias_data,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_module_store().lock() {
        store.insert(id, Box::new(module_impl));
        id as *mut TorshModule
    } else {
        set_last_error("Failed to store module".to_string());
        ptr::null_mut()
    }
}

/// Create linear layer (alias for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_linear_create(
    in_features: usize,
    out_features: usize,
    bias: bool,
) -> *mut TorshModule {
    torsh_linear_new(in_features, out_features, bias)
}

/// Forward pass through linear layer
#[no_mangle]
pub unsafe extern "C" fn torsh_linear_forward(
    module: *const TorshModule,
    input: *const super::types::TorshTensor,
    output: *mut super::types::TorshTensor,
) -> TorshError {
    if module.is_null() || input.is_null() || output.is_null() {
        return TorshError::InvalidArgument;
    }

    let module_id = module as usize;
    let input_id = input as usize;
    let output_id = output as usize;

    if let (Ok(module_store), Ok(mut tensor_store)) =
        (get_module_store().lock(), get_tensor_store().lock())
    {
        if let (Some(module_impl), Some(input_impl)) =
            (module_store.get(&module_id), tensor_store.get(&input_id))
        {
            if input_impl.shape.len() != 2 {
                set_last_error("Linear layer requires 2D input".to_string());
                return TorshError::ShapeMismatch;
            }

            let batch_size = input_impl.shape[0];
            let input_features = input_impl.shape[1];

            if input_features != module_impl.in_features {
                set_last_error("Input feature dimension mismatch".to_string());
                return TorshError::ShapeMismatch;
            }

            let mut output_data = vec![0.0; batch_size * module_impl.out_features];

            // Perform matrix multiplication: input @ weight.T + bias
            for batch_idx in 0..batch_size {
                for out_idx in 0..module_impl.out_features {
                    let mut sum = 0.0;

                    // Matrix multiplication
                    for in_idx in 0..module_impl.in_features {
                        let input_val = input_impl.data[batch_idx * input_features + in_idx];
                        let weight_val =
                            module_impl.weight[out_idx * module_impl.in_features + in_idx];
                        sum += input_val * weight_val;
                    }

                    // Add bias if present
                    if let Some(ref bias_data) = module_impl.bias_data {
                        sum += bias_data[out_idx];
                    }

                    output_data[batch_idx * module_impl.out_features + out_idx] = sum;
                }
            }

            let output_tensor = TensorImpl {
                data: output_data,
                shape: vec![batch_size, module_impl.out_features],
                dtype: input_impl.dtype,
            };

            tensor_store.insert(output_id, Box::new(output_tensor));
            return TorshError::Success;
        }
    }

    set_last_error("Invalid module or tensor handles".to_string());
    TorshError::InvalidArgument
}

// =============================================================================
// Module Management
// =============================================================================

/// Free a module
#[no_mangle]
pub unsafe extern "C" fn torsh_module_free(module: *mut TorshModule) {
    if !module.is_null() {
        let id = module as usize;
        if let Ok(mut store) = get_module_store().lock() {
            store.remove(&id);
        }
    }
}

/// Free linear layer (alias for module_free)
#[no_mangle]
pub unsafe extern "C" fn torsh_linear_free(module: *mut TorshModule) {
    torsh_module_free(module)
}

/// Get output dimensions of a linear module
#[no_mangle]
pub unsafe extern "C" fn torsh_linear_get_output_features(module: *const TorshModule) -> usize {
    if module.is_null() {
        return 0;
    }

    let module_id = module as usize;
    if let Ok(store) = get_module_store().lock() {
        if let Some(module_impl) = store.get(&module_id) {
            return module_impl.out_features;
        }
    }
    0
}

/// Get input dimensions of a linear module
#[no_mangle]
pub unsafe extern "C" fn torsh_linear_get_input_features(module: *const TorshModule) -> usize {
    if module.is_null() {
        return 0;
    }

    let module_id = module as usize;
    if let Ok(store) = get_module_store().lock() {
        if let Some(module_impl) = store.get(&module_id) {
            return module_impl.in_features;
        }
    }
    0
}

/// Clear all modules from storage (for cleanup)
pub(crate) fn clear_module_store() {
    if let Ok(mut store) = get_module_store().lock() {
        store.clear();
    }
}

// =============================================================================
// Future Neural Network Operations (Placeholders)
// =============================================================================

// The following operations can be added in future iterations:
//
// - Convolutional layers (torsh_conv2d_new, torsh_conv2d_forward)
// - Batch normalization (torsh_batchnorm_new, torsh_batchnorm_forward)
// - Dropout layers (torsh_dropout_new, torsh_dropout_forward)
// - LSTM/GRU layers (torsh_lstm_new, torsh_lstm_forward)
// - Attention mechanisms (torsh_attention_new, torsh_attention_forward)
// - Embedding layers (torsh_embedding_new, torsh_embedding_forward)
//
// Each would follow the same pattern:
// 1. Define module-specific parameters in ModuleImpl
// 2. Create initialization function with proper weight initialization
// 3. Implement forward pass with mathematical operations
// 4. Provide memory management functions

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_creation() {
        unsafe {
            let linear = torsh_linear_new(10, 5, true);
            assert!(!linear.is_null());

            // Clean up
            torsh_linear_free(linear);
        }
    }

    #[test]
    fn test_invalid_dimensions() {
        unsafe {
            let linear = torsh_linear_new(0, 5, true);
            assert!(linear.is_null());
        }
    }
}
