//! C API for ToRSh
//!
//! This module provides a C-compatible API for integrating ToRSh with
//! other languages and systems that can call C functions.
//!
//! The C API is organized into focused modules for maintainability:
//! - [`types`]: C-compatible types, handles, and error codes
//! - [`tensor`]: Tensor operations (creation, manipulation, math)
//! - [`neural`]: Neural network operations (layers, modules)
//! - [`optimizer`]: Optimization algorithms (SGD, Adam, etc.)
//! - [`utils`]: Utility functions (initialization, device management)
//!
//! ## Architecture
//!
//! The C API uses opaque handles to safely expose Rust objects to C code:
//! - `TorshTensor`: Handle to tensor objects with data and shape
//! - `TorshModule`: Handle to neural network modules (layers)
//! - `TorshOptimizer`: Handle to optimization algorithms
//!
//! Error handling is done through return codes and a global error state that
//! can be queried using `torsh_get_last_error()`.
//!
//! ## Memory Management
//!
//! All objects created through this API must be explicitly freed:
//! - Tensors: `torsh_tensor_free()`
//! - Modules: `torsh_module_free()` or `torsh_linear_free()`
//! - Optimizers: `torsh_optimizer_free()`
//!
//! ## Usage Example (C)
//!
//! ```c
//! #include "torsh.h"
//!
//! int main() {
//!     // Initialize library
//!     torsh_init();
//!
//!     // Create tensors
//!     size_t shape[] = {2, 3};
//!     TorshTensor* a = torsh_tensor_ones(shape, 2);
//!     TorshTensor* b = torsh_tensor_zeros(shape, 2);
//!     TorshTensor* result = torsh_tensor_zeros(shape, 2);
//!
//!     // Perform operation
//!     torsh_tensor_add(a, b, result);
//!
//!     // Create neural network
//!     TorshModule* linear = torsh_linear_new(3, 2, true);
//!     TorshOptimizer* optimizer = torsh_sgd_new(0.01, 0.9);
//!
//!     // Clean up
//!     torsh_tensor_free(a);
//!     torsh_tensor_free(b);
//!     torsh_tensor_free(result);
//!     torsh_linear_free(linear);
//!     torsh_optimizer_free(optimizer);
//!
//!     torsh_cleanup();
//!     return 0;
//! }
//! ```
//!
//! ## Thread Safety
//!
//! The C API is thread-safe through the use of internal mutexes. However,
//! individual tensor/module/optimizer handles should not be shared between
//! threads without external synchronization.

pub mod neural;
pub mod optimizer;
pub mod tensor;
pub mod types;
pub mod utils;

// Re-export all public types for backward compatibility
pub use types::{TensorHandle, TorshDType, TorshError, TorshModule, TorshOptimizer, TorshTensor};

// Re-export utility functions for error handling
pub use utils::{get_last_error, set_last_error};

// =============================================================================
// Public API Re-exports for Backward Compatibility
// =============================================================================

// Tensor operations
pub use tensor::{
    torsh_tensor_abs,
    // Basic math
    torsh_tensor_add,
    // Scalar operations
    torsh_tensor_add_scalar,
    torsh_tensor_cos,
    // Access
    torsh_tensor_data,
    torsh_tensor_div_scalar,

    torsh_tensor_dtype,
    // Mathematical functions
    torsh_tensor_exp,
    // Memory management
    torsh_tensor_free,
    torsh_tensor_from_data,

    torsh_tensor_log,
    torsh_tensor_matmul,

    torsh_tensor_max_all,
    torsh_tensor_max_dim,
    torsh_tensor_mean_all,
    torsh_tensor_mean_dim,
    torsh_tensor_min_all,
    torsh_tensor_min_dim,

    torsh_tensor_mul,
    torsh_tensor_mul_scalar,
    torsh_tensor_multiply,
    torsh_tensor_ndim,
    // Creation
    torsh_tensor_new,
    torsh_tensor_numel,
    torsh_tensor_ones,
    torsh_tensor_rand,
    torsh_tensor_randn,
    // Activations
    torsh_tensor_relu,
    torsh_tensor_scalar,
    torsh_tensor_shape,
    torsh_tensor_sigmoid,
    torsh_tensor_sin,
    torsh_tensor_size,

    torsh_tensor_sqrt,
    torsh_tensor_sub,
    torsh_tensor_sub_scalar,
    // Reductions
    torsh_tensor_sum_all,
    torsh_tensor_sum_dim,
    torsh_tensor_tan,

    torsh_tensor_tanh,

    // Manipulation
    torsh_tensor_transpose,

    torsh_tensor_zeros,
};

// Neural network operations
pub use neural::{
    torsh_linear_create, torsh_linear_forward, torsh_linear_free, torsh_linear_get_input_features,
    torsh_linear_get_output_features, torsh_linear_new, torsh_module_free,
};

// Optimizer operations
pub use optimizer::{
    torsh_adam_create,

    // Adam
    torsh_adam_new,
    torsh_optimizer_free,
    // Generic
    torsh_optimizer_step,
    torsh_optimizer_zero_grad,
    torsh_sgd_create,
    // SGD
    torsh_sgd_new,
    torsh_sgd_step,
};

// Utility operations
pub use utils::{
    torsh_cleanup,

    torsh_clear_last_error,
    torsh_cuda_device_count,
    torsh_cuda_device_memory,

    torsh_cuda_device_name,
    // CUDA support
    torsh_cuda_is_available,
    torsh_get_device_id,

    torsh_get_device_type,
    // Error handling
    torsh_get_last_error,
    // Memory and performance
    torsh_get_memory_usage,
    torsh_get_num_threads,
    torsh_get_peak_memory_usage,
    torsh_has_error,

    torsh_init,
    torsh_is_autocast_enabled,
    torsh_is_debug_mode_enabled,
    torsh_is_grad_enabled,

    torsh_reset_memory_tracking,
    // Configuration
    torsh_set_autocast,
    // Debug and profiling
    torsh_set_debug_mode,
    // Device management
    torsh_set_device,
    torsh_set_grad_enabled,
    torsh_set_num_threads,

    torsh_set_random_seed,
    torsh_start_profiling,
    torsh_stop_profiling,
    // Version and initialization
    torsh_version,
};

/// Global initialization and cleanup for all C API modules
pub fn initialize_c_api() -> Result<(), String> {
    // This function can be called from Rust code to ensure
    // all global state is properly initialized

    // Initialize stores
    let _ = tensor::get_tensor_store();
    let _ = neural::get_module_store();
    let _ = optimizer::get_optimizer_store();
    let _ = utils::get_last_error();

    Ok(())
}

/// Cleanup all C API global state
pub fn cleanup_c_api() {
    tensor::clear_tensor_store();
    neural::clear_module_store();
    optimizer::clear_optimizer_store();

    if let Ok(mut last_error) = utils::get_last_error().lock() {
        *last_error = None;
    }
}

/// C API module statistics for debugging
pub struct CApiStats {
    pub tensor_count: usize,
    pub module_count: usize,
    pub optimizer_count: usize,
    pub has_error: bool,
}

/// Get current C API statistics
pub fn get_c_api_stats() -> CApiStats {
    let tensor_count = if let Ok(store) = tensor::get_tensor_store().lock() {
        store.len()
    } else {
        0
    };

    let module_count = if let Ok(store) = neural::get_module_store().lock() {
        store.len()
    } else {
        0
    };

    let optimizer_count = if let Ok(store) = optimizer::get_optimizer_store().lock() {
        store.len()
    } else {
        0
    };

    let has_error = if let Ok(error) = utils::get_last_error().lock() {
        error.is_some()
    } else {
        false
    };

    CApiStats {
        tensor_count,
        module_count,
        optimizer_count,
        has_error,
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::slice;

    #[test]
    fn test_full_workflow() {
        unsafe {
            // Initialize
            assert_eq!(torsh_init(), TorshError::Success);

            // Create tensors
            let shape = [2usize, 3usize];
            let a = torsh_tensor_ones(shape.as_ptr(), 2);
            let b = torsh_tensor_zeros(shape.as_ptr(), 2);
            let result = torsh_tensor_zeros(shape.as_ptr(), 2);

            assert!(!a.is_null());
            assert!(!b.is_null());
            assert!(!result.is_null());

            // Perform operation
            assert_eq!(torsh_tensor_add(a, b, result), TorshError::Success);

            // Create neural network
            let linear = torsh_linear_new(3, 2, true);
            assert!(!linear.is_null());

            // Create optimizer
            let optimizer = torsh_sgd_new(0.01, 0.9);
            assert!(!optimizer.is_null());

            // Check API stats
            let stats = get_c_api_stats();
            assert_eq!(stats.tensor_count, 3);
            assert_eq!(stats.module_count, 1);
            assert_eq!(stats.optimizer_count, 1);
            assert!(!stats.has_error);

            // Clean up
            torsh_tensor_free(a);
            torsh_tensor_free(b);
            torsh_tensor_free(result);
            torsh_linear_free(linear);
            torsh_optimizer_free(optimizer);

            assert_eq!(torsh_cleanup(), TorshError::Success);
        }
    }

    #[test]
    fn test_error_handling() {
        unsafe {
            torsh_init();

            // Clear any existing errors
            torsh_clear_last_error();
            assert_eq!(torsh_has_error(), 0);

            // Trigger an error (invalid tensor operation)
            let result = torsh_tensor_add(std::ptr::null(), std::ptr::null(), std::ptr::null_mut());
            assert_eq!(result, TorshError::InvalidArgument);

            // Error should be cleared after init/cleanup
            torsh_cleanup();
        }
    }

    #[test]
    fn test_tensor_operations() {
        unsafe {
            torsh_init();

            // Create test data
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let shape = [2usize, 2usize];

            let tensor = torsh_tensor_from_data(data.as_ptr(), data.len(), shape.as_ptr(), 2);
            assert!(!tensor.is_null());

            // Test tensor properties
            assert_eq!(torsh_tensor_numel(tensor), 4);
            assert_eq!(torsh_tensor_ndim(tensor), 2);

            // Test mathematical operations
            let doubled = torsh_tensor_mul_scalar(tensor, 2.0);
            assert!(!doubled.is_null());

            let sqrt_tensor = torsh_tensor_sqrt(tensor);
            assert!(!sqrt_tensor.is_null());

            let transposed = torsh_tensor_transpose(tensor);
            assert!(!transposed.is_null());

            // Clean up
            torsh_tensor_free(tensor);
            torsh_tensor_free(doubled);
            torsh_tensor_free(sqrt_tensor);
            torsh_tensor_free(transposed);

            torsh_cleanup();
        }
    }

    #[test]
    fn test_neural_network() {
        unsafe {
            torsh_init();

            // Create linear layer
            let linear = torsh_linear_new(4, 2, true);
            assert!(!linear.is_null());

            // Create input tensor (batch_size=1, features=4)
            let input_data = [1.0f32, 2.0, 3.0, 4.0];
            let input_shape = [1usize, 4usize];
            let input = torsh_tensor_from_data(
                input_data.as_ptr(),
                input_data.len(),
                input_shape.as_ptr(),
                2,
            );
            assert!(!input.is_null());

            // Create output tensor (batch_size=1, features=2)
            let output_shape = [1usize, 2usize];
            let output = torsh_tensor_zeros(output_shape.as_ptr(), 2);
            assert!(!output.is_null());

            // Forward pass
            let result = torsh_linear_forward(linear, input, output);
            assert_eq!(result, TorshError::Success);

            // Clean up
            torsh_tensor_free(input);
            torsh_tensor_free(output);
            torsh_linear_free(linear);

            torsh_cleanup();
        }
    }

    #[test]
    fn test_optimizers() {
        unsafe {
            torsh_init();

            // Test SGD optimizer
            let sgd = torsh_sgd_new(0.01, 0.9);
            assert!(!sgd.is_null());

            assert_eq!(torsh_optimizer_zero_grad(sgd), TorshError::Success);
            assert_eq!(torsh_optimizer_step(sgd), TorshError::Success);

            torsh_optimizer_free(sgd);

            // Test Adam optimizer
            let adam = torsh_adam_new(0.001, 0.9, 0.999, 1e-8);
            assert!(!adam.is_null());

            assert_eq!(torsh_optimizer_zero_grad(adam), TorshError::Success);
            assert_eq!(torsh_optimizer_step(adam), TorshError::Success);

            torsh_optimizer_free(adam);

            torsh_cleanup();
        }
    }

    #[test]
    fn test_utilities() {
        unsafe {
            // Test version
            let version = torsh_version();
            assert!(!version.is_null());

            // Test device management
            assert_eq!(torsh_set_device(0, 0), TorshError::Success);
            assert_eq!(torsh_get_device_type(), 0);
            assert_eq!(torsh_get_device_id(), 0);

            // Test CUDA info
            assert_eq!(torsh_cuda_is_available(), 0);
            assert_eq!(torsh_cuda_device_count(), 0);

            // Test configuration
            assert_eq!(torsh_set_random_seed(12345), TorshError::Success);
            assert_eq!(torsh_set_num_threads(4), TorshError::Success);
        }
    }
}
