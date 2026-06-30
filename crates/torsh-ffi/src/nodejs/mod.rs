//! Node.js N-API bindings for ToRSh tensors
//!
//! This module provides Node.js integration through N-API, allowing JavaScript/TypeScript
//! applications to use ToRSh tensors with native performance.
//!
//! Sub-modules:
//! - [`helpers`]: Low-level NAPI <-> Rust conversion utilities
//! - [`creation`]: Tensor creation functions (zeros, ones, randn, eye, linspace, …)
//! - [`ops`]: Element-wise, scalar, and matrix operations
//! - [`activations`]: Activation function handlers (sigmoid, tanh, softmax, …)
//! - [`reductions`]: Reduction operations (sum, mean)
//! - [`clone_detach`]: Clone / detach handlers
//! - [`utils_js`]: Utility handlers (shape, data, seed, CUDA info, save/load)

pub mod activations;
pub mod clone_detach;
pub mod creation;
pub mod helpers;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod reductions;
pub mod utils_js;

use std::ffi::CString;
use std::ptr;

// Re-export NAPI core types so submodules can use them without re-declaring.
pub use helpers::{
    NapiCallback, NapiCallbackInfo, NapiEnv, NapiFinalizeCallback, NapiPropertyDescriptor,
    NapiStatus, NapiValue,
};

// Re-export all handler functions referenced in the property table below.
use activations::{js_sigmoid, js_softmax, js_tanh};
use clone_detach::{js_clone, js_detach};
use creation::{js_create_tensor, js_eye, js_linspace, js_ones, js_randn, js_zeros};
use nn::{js_conv2d, js_cross_entropy_loss, js_log};
use ops::{
    js_add, js_add_scalar, js_div_scalar, js_divide, js_matmul, js_mul_scalar, js_multiply,
    js_relu, js_reshape, js_sub, js_sub_scalar, js_transpose,
};
use optim::{js_adam_step, js_sgd_step};
use reductions::{js_mean, js_sum};
use utils_js::{
    js_cuda_available, js_cuda_device_count, js_get_data, js_get_shape, js_load_tensor,
    js_manual_seed, js_save_tensor,
};

extern "C" {
    pub fn napi_define_properties(
        env: NapiEnv,
        object: NapiValue,
        property_count: usize,
        properties: *const NapiPropertyDescriptor,
    ) -> NapiStatus;
}

/// Module initialization entry-point called by Node.js on `require()`.
///
/// All NAPI handler functions are registered here as named properties on the
/// `exports` object so that `index.ts` can call them as `native.<name>(...)`.
#[no_mangle]
pub extern "C" fn napi_register_module_v1(env: NapiEnv, exports: NapiValue) -> NapiValue {
    unsafe {
        // Build name–method pairs.  The CStrings must outlive the descriptors.
        let pairs: &[(&str, NapiCallback)] = &[
            // ── Tensor creation ──────────────────────────────────────────────
            ("createTensor", js_create_tensor),
            ("zeros", js_zeros),
            ("ones", js_ones),
            ("randn", js_randn),
            ("eye", js_eye),
            ("linspace", js_linspace),
            // ── Element-wise / matrix ops ─────────────────────────────────────
            ("add", js_add),
            ("sub", js_sub),
            ("multiply", js_multiply),
            ("divide", js_divide),
            ("matmul", js_matmul),
            ("relu", js_relu),
            ("addScalar", js_add_scalar),
            ("subScalar", js_sub_scalar),
            ("mulScalar", js_mul_scalar),
            ("divScalar", js_div_scalar),
            ("transpose", js_transpose),
            ("reshape", js_reshape),
            // ── Activations ───────────────────────────────────────────────────
            ("sigmoid", js_sigmoid),
            ("tanh", js_tanh),
            ("softmax", js_softmax),
            // ── Math functions ────────────────────────────────────────────────
            ("log", js_log),
            // ── Neural network ops ────────────────────────────────────────────
            ("conv2d", js_conv2d),
            ("crossEntropyLoss", js_cross_entropy_loss),
            // ── Optimizer ops ─────────────────────────────────────────────────
            ("sgdStep", js_sgd_step),
            ("adamStep", js_adam_step),
            // ── Reductions ────────────────────────────────────────────────────
            ("sum", js_sum),
            ("mean", js_mean),
            // ── Clone / detach ────────────────────────────────────────────────
            ("clone", js_clone),
            ("detach", js_detach),
            // ── Data access ───────────────────────────────────────────────────
            ("getShape", js_get_shape),
            ("getData", js_get_data),
            // ── Utilities ─────────────────────────────────────────────────────
            ("manualSeed", js_manual_seed),
            ("cudaAvailable", js_cuda_available),
            ("cudaDeviceCount", js_cuda_device_count),
            ("saveTensor", js_save_tensor),
            ("loadTensor", js_load_tensor),
        ];

        // Keep the CStrings alive for the duration of `napi_define_properties`.
        let c_names: Vec<CString> = pairs
            .iter()
            .map(|(name, _)| {
                CString::new(*name).expect("static string should not contain null bytes")
            })
            .collect();

        // SAFETY: transmuting a null pointer to a function pointer is required
        // by the N-API ABI for unused getter/setter slots.
        #[allow(clippy::transmute_null_to_fn)]
        let properties: Vec<NapiPropertyDescriptor> = pairs
            .iter()
            .enumerate()
            .map(|(i, (_, method))| NapiPropertyDescriptor {
                utf8name: c_names[i].as_ptr(),
                name: ptr::null_mut(),
                method: *method,
                getter: std::mem::transmute(ptr::null::<()>()),
                setter: std::mem::transmute(ptr::null::<()>()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            })
            .collect();

        napi_define_properties(env, exports, properties.len(), properties.as_ptr());
        exports
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_nodejs_bindings_compilation() {
        // Verifies the module compiles correctly.
        // Full integration tests require a live Node.js runtime.
        assert!(true);
    }
}
