//! Optimizer operations for C API
//!
//! This module provides optimization functionality for the ToRSh C API,
//! including SGD, Adam, and other optimization algorithms with parameter updates.

use std::collections::HashMap;
use std::os::raw::c_float;
use std::ptr;
use std::sync::{Mutex, OnceLock};

use super::tensor::get_next_id;
use super::types::{TorshError, TorshOptimizer};

// Global optimizer storage
static OPTIMIZER_STORE: OnceLock<Mutex<HashMap<usize, Box<OptimizerImpl>>>> = OnceLock::new();

/// Internal optimizer implementation
pub(crate) struct OptimizerImpl {
    #[allow(dead_code)]
    pub optimizer_type: String,
    #[allow(dead_code)]
    pub learning_rate: f32,
    #[allow(dead_code)]
    pub momentum: Option<f32>,
    #[allow(dead_code)]
    pub beta1: Option<f32>,
    #[allow(dead_code)]
    pub beta2: Option<f32>,
    #[allow(dead_code)]
    pub epsilon: Option<f32>,
}

/// Get the global optimizer store
pub(crate) fn get_optimizer_store() -> &'static Mutex<HashMap<usize, Box<OptimizerImpl>>> {
    OPTIMIZER_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Set last error message (imported from parent module)
pub(crate) fn set_last_error(message: String) {
    if let Ok(mut last_error) = super::get_last_error().lock() {
        *last_error = Some(message);
    }
}

// =============================================================================
// SGD Optimizer Operations
// =============================================================================

/// Create SGD optimizer
#[no_mangle]
pub unsafe extern "C" fn torsh_sgd_new(
    learning_rate: c_float,
    momentum: c_float,
) -> *mut TorshOptimizer {
    if learning_rate <= 0.0 {
        set_last_error("Learning rate must be positive".to_string());
        return ptr::null_mut();
    }

    let optimizer_impl = OptimizerImpl {
        optimizer_type: "sgd".to_string(),
        learning_rate,
        momentum: Some(momentum),
        beta1: None,
        beta2: None,
        epsilon: None,
    };

    let id = get_next_id();
    if let Ok(mut store) = get_optimizer_store().lock() {
        store.insert(id, Box::new(optimizer_impl));
        id as *mut TorshOptimizer
    } else {
        set_last_error("Failed to store optimizer".to_string());
        ptr::null_mut()
    }
}

/// Create SGD optimizer (alias for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_sgd_create(learning_rate: c_float) -> *mut TorshOptimizer {
    torsh_sgd_new(learning_rate, 0.0) // Default momentum = 0.0
}

/// Perform SGD optimizer step (for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_sgd_step(optimizer: *const TorshOptimizer) -> TorshError {
    if optimizer.is_null() {
        return TorshError::InvalidArgument;
    }

    let id = optimizer as usize;

    if let Ok(store) = get_optimizer_store().lock() {
        if let Some(_optimizer_impl) = store.get(&id) {
            // In a real implementation, this would update parameters using gradients
            // For now, just return success
            return TorshError::Success;
        }
    }

    set_last_error("Invalid optimizer handle".to_string());
    TorshError::InvalidArgument
}

// =============================================================================
// Adam Optimizer Operations
// =============================================================================

/// Create Adam optimizer
#[no_mangle]
pub unsafe extern "C" fn torsh_adam_new(
    learning_rate: c_float,
    beta1: c_float,
    beta2: c_float,
    epsilon: c_float,
) -> *mut TorshOptimizer {
    if learning_rate <= 0.0 {
        set_last_error("Learning rate must be positive".to_string());
        return ptr::null_mut();
    }

    if beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0 {
        set_last_error("Beta parameters must be in [0, 1)".to_string());
        return ptr::null_mut();
    }

    let optimizer_impl = OptimizerImpl {
        optimizer_type: "adam".to_string(),
        learning_rate,
        momentum: None,
        beta1: Some(beta1),
        beta2: Some(beta2),
        epsilon: Some(epsilon),
    };

    let id = get_next_id();
    if let Ok(mut store) = get_optimizer_store().lock() {
        store.insert(id, Box::new(optimizer_impl));
        id as *mut TorshOptimizer
    } else {
        set_last_error("Failed to store optimizer".to_string());
        ptr::null_mut()
    }
}

/// Create Adam optimizer (alias for R bindings compatibility)
#[no_mangle]
pub unsafe extern "C" fn torsh_adam_create(
    learning_rate: c_float,
    beta1: c_float,
    beta2: c_float,
    eps: c_float,
) -> *mut TorshOptimizer {
    let optimizer_impl = OptimizerImpl {
        optimizer_type: "Adam".to_string(),
        learning_rate: learning_rate.max(0.0),
        momentum: None,
        beta1: Some(beta1.max(0.0).min(1.0)),
        beta2: Some(beta2.max(0.0).min(1.0)),
        epsilon: Some(eps.max(1e-8)),
    };

    let id = get_next_id();
    if let Ok(mut store) = get_optimizer_store().lock() {
        store.insert(id, Box::new(optimizer_impl));
        id as *mut TorshOptimizer
    } else {
        ptr::null_mut()
    }
}

// =============================================================================
// Generic Optimizer Operations
// =============================================================================

/// Optimizer step
#[no_mangle]
pub unsafe extern "C" fn torsh_optimizer_step(optimizer: *mut TorshOptimizer) -> TorshError {
    if optimizer.is_null() {
        return TorshError::InvalidArgument;
    }

    let id = optimizer as usize;
    if let Ok(store) = get_optimizer_store().lock() {
        if let Some(_optimizer_impl) = store.get(&id) {
            // In a real implementation, this would:
            // 1. Iterate through all parameters
            // 2. Apply the optimizer algorithm (SGD, Adam, etc.)
            // 3. Update parameter values based on gradients

            // For now, this is a placeholder
            return TorshError::Success;
        }
    }

    set_last_error("Invalid optimizer handle".to_string());
    TorshError::InvalidArgument
}

/// Zero gradients
#[no_mangle]
pub unsafe extern "C" fn torsh_optimizer_zero_grad(optimizer: *mut TorshOptimizer) -> TorshError {
    if optimizer.is_null() {
        return TorshError::InvalidArgument;
    }

    let id = optimizer as usize;
    if let Ok(store) = get_optimizer_store().lock() {
        if let Some(_optimizer_impl) = store.get(&id) {
            // In a real implementation, this would:
            // 1. Iterate through all parameters
            // 2. Set all gradients to zero

            // For now, this is a placeholder
            return TorshError::Success;
        }
    }

    set_last_error("Invalid optimizer handle".to_string());
    TorshError::InvalidArgument
}

// =============================================================================
// Memory Management
// =============================================================================

/// Free optimizer
#[no_mangle]
pub unsafe extern "C" fn torsh_optimizer_free(optimizer: *mut TorshOptimizer) {
    if !optimizer.is_null() {
        let id = optimizer as usize;
        if let Ok(mut store) = get_optimizer_store().lock() {
            store.remove(&id);
        }
    }
}

/// Clear all optimizers from storage (for cleanup)
pub(crate) fn clear_optimizer_store() {
    if let Ok(mut store) = get_optimizer_store().lock() {
        store.clear();
    }
}

// =============================================================================
// Future Optimizer Operations (Placeholders)
// =============================================================================

// The following optimizers can be added in future iterations:
//
// - AdamW optimizer (torsh_adamw_new, torsh_adamw_step)
// - RMSprop optimizer (torsh_rmsprop_new, torsh_rmsprop_step)
// - AdaGrad optimizer (torsh_adagrad_new, torsh_adagrad_step)
// - LBFGS optimizer (torsh_lbfgs_new, torsh_lbfgs_step)
// - Momentum optimizer (torsh_momentum_new, torsh_momentum_step)
// - NAdam optimizer (torsh_nadam_new, torsh_nadam_step)
//
// Each would follow the same pattern:
// 1. Extend OptimizerImpl with optimizer-specific parameters
// 2. Implement creation function with validation
// 3. Implement step function with the optimization algorithm
// 4. Provide parameter management and state tracking

/// Helper function to validate optimizer parameters
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn validate_optimizer_params(
    learning_rate: f32,
    beta1: Option<f32>,
    beta2: Option<f32>,
) -> bool {
    if learning_rate <= 0.0 {
        return false;
    }

    if let Some(b1) = beta1 {
        if b1 < 0.0 || b1 >= 1.0 {
            return false;
        }
    }

    if let Some(b2) = beta2 {
        if b2 < 0.0 || b2 >= 1.0 {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_creation() {
        unsafe {
            let sgd = torsh_sgd_new(0.01, 0.9);
            assert!(!sgd.is_null());

            // Clean up
            torsh_optimizer_free(sgd);
        }
    }

    #[test]
    fn test_adam_creation() {
        unsafe {
            let adam = torsh_adam_new(0.001, 0.9, 0.999, 1e-8);
            assert!(!adam.is_null());

            // Clean up
            torsh_optimizer_free(adam);
        }
    }

    #[test]
    fn test_invalid_learning_rate() {
        unsafe {
            let sgd = torsh_sgd_new(-0.01, 0.9);
            assert!(sgd.is_null());
        }
    }

    #[test]
    fn test_invalid_beta_parameters() {
        unsafe {
            let adam = torsh_adam_new(0.001, 1.1, 0.999, 1e-8);
            assert!(adam.is_null());
        }
    }

    #[test]
    fn test_parameter_validation() {
        assert!(validate_optimizer_params(0.01, Some(0.9), Some(0.999)));
        assert!(!validate_optimizer_params(-0.01, Some(0.9), Some(0.999)));
        assert!(!validate_optimizer_params(0.01, Some(1.1), Some(0.999)));
        assert!(!validate_optimizer_params(0.01, Some(0.9), Some(-0.1)));
    }
}
