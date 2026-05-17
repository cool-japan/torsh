//! NAPI handlers for optimizer operations: sgdStep, adamStep.
//!
//! Both handlers require in-place mutation of parameter tensors.  They rely on
//! `torsh_tensor_axpy_inplace` (for SGD) and `torsh_tensor_adam_step_inplace`
//! (for Adam) which mutate tensors stored in the global tensor store.

use std::ptr;

use crate::c_api::{torsh_tensor_adam_step_inplace, torsh_tensor_axpy_inplace};
use crate::c_api::types::TorshError;

use super::helpers::{
    get_tensor_from_external, throw_error, NapiCallbackInfo, NapiEnv, NapiStatus, NapiValue,
};

extern "C" {
    fn napi_get_cb_info(
        env: NapiEnv,
        cbinfo: NapiCallbackInfo,
        argc: *mut usize,
        argv: *mut NapiValue,
        this_arg: *mut NapiValue,
        data: *mut *mut std::os::raw::c_void,
    ) -> NapiStatus;
    fn napi_get_value_double(env: NapiEnv, value: NapiValue, result: *mut f64) -> NapiStatus;
    fn napi_get_array_length(env: NapiEnv, value: NapiValue, result: *mut u32) -> NapiStatus;
    fn napi_get_element(
        env: NapiEnv,
        object: NapiValue,
        index: u32,
        result: *mut NapiValue,
    ) -> NapiStatus;
    fn napi_is_array(env: NapiEnv, value: NapiValue, result: *mut bool) -> NapiStatus;
    fn napi_get_undefined(env: NapiEnv, result: *mut NapiValue) -> NapiStatus;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Collect tensor handles from a JS array of externals.  Returns `None` on
/// any error (throws a NAPI error and leaves a message in `env`).
unsafe fn js_array_to_tensor_handles(
    env: NapiEnv,
    array: NapiValue,
    label: &str,
) -> Option<Vec<*mut crate::c_api::types::TorshTensor>> {
    let mut is_arr = false;
    if napi_is_array(env, array, &mut is_arr) != NapiStatus::Ok || !is_arr {
        let msg = format!("{} must be an array of tensors", label);
        throw_error(env, "INVALID_ARGS", &msg);
        return None;
    }
    let mut length = 0u32;
    if napi_get_array_length(env, array, &mut length) != NapiStatus::Ok {
        throw_error(env, "INVALID_ARGS", "Failed to get array length");
        return None;
    }
    let mut handles = Vec::with_capacity(length as usize);
    for i in 0..length {
        let mut elem = ptr::null_mut();
        if napi_get_element(env, array, i, &mut elem) != NapiStatus::Ok {
            throw_error(env, "INVALID_ARGS", "Failed to read array element");
            return None;
        }
        let h = get_tensor_from_external(env, elem);
        if h.is_null() {
            let msg = format!("{}: element {} is not a valid tensor", label, i);
            throw_error(env, "INVALID_TENSOR", &msg);
            return None;
        }
        handles.push(h);
    }
    Some(handles)
}

/// Helper to return `undefined` (JS void).
unsafe fn return_undefined(env: NapiEnv) -> NapiValue {
    let mut undef = ptr::null_mut();
    napi_get_undefined(env, &mut undef);
    undef
}

// ─────────────────────────────────────────────────────────────────────────────

/// JavaScript: `sgdStep(params, grads, lr)`.
///
/// Performs `param[i] -= lr * grad[i]` in-place for each parameter, using
/// `torsh_tensor_axpy_inplace(param, grad, -lr)`.
///
/// - `params` : JS array of tensor externals
/// - `grads`  : JS array of tensor externals (same length as params)
/// - `lr`     : learning-rate (number)
pub extern "C" fn js_sgd_step(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 3usize;
        let mut args = [ptr::null_mut(); 3];
        if napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
        ) != NapiStatus::Ok
        {
            return ptr::null_mut();
        }
        if argc != 3 {
            throw_error(env, "INVALID_ARGS", "sgdStep(params, grads, lr) requires 3 arguments");
            return ptr::null_mut();
        }

        let params = match js_array_to_tensor_handles(env, args[0], "params") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };
        let grads = match js_array_to_tensor_handles(env, args[1], "grads") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };

        if params.len() != grads.len() {
            throw_error(env, "INVALID_ARGS", "params and grads must have the same length");
            return ptr::null_mut();
        }

        let mut lr_d = 0.0f64;
        if napi_get_value_double(env, args[2], &mut lr_d) != NapiStatus::Ok {
            throw_error(env, "INVALID_ARGS", "lr must be a number");
            return ptr::null_mut();
        }
        let lr = lr_d as f32;

        for (param, grad) in params.iter().zip(grads.iter()) {
            // param += (-lr) * grad  ⟺  param -= lr * grad
            if torsh_tensor_axpy_inplace(*param, *grad, -lr) != TorshError::Success {
                throw_error(env, "OPERATION_FAILED", "sgdStep: axpy_inplace failed");
                return ptr::null_mut();
            }
        }

        return_undefined(env)
    }
}

/// JavaScript: `adamStep(params, grads, m, v, lr?, beta1?, beta2?, eps?, step?)`.
///
/// Applies the Adam update rule in-place for each parameter:
///   m[i]  = beta1 * m[i] + (1 - beta1) * grad[i]
///   v[i]  = beta2 * v[i] + (1 - beta2) * grad[i]²
///   param[i] -= lr * (m[i]/(1-beta1^step)) / (sqrt(v[i]/(1-beta2^step)) + eps)
///
/// - `params`, `grads`, `m`, `v` : JS arrays of tensor externals (equal length)
/// - `lr`    : default 0.001
/// - `beta1` : default 0.9
/// - `beta2` : default 0.999
/// - `eps`   : default 1e-8
/// - `step`  : default 1 (integer ≥ 1)
pub extern "C" fn js_adam_step(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 9usize;
        let mut args = [ptr::null_mut(); 9];
        if napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
        ) != NapiStatus::Ok
        {
            return ptr::null_mut();
        }
        if argc < 4 {
            throw_error(
                env,
                "INVALID_ARGS",
                "adamStep(params, grads, m, v [, lr, beta1, beta2, eps, step]) requires at least 4 arguments",
            );
            return ptr::null_mut();
        }

        let params = match js_array_to_tensor_handles(env, args[0], "params") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };
        let grads = match js_array_to_tensor_handles(env, args[1], "grads") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };
        let m_arr = match js_array_to_tensor_handles(env, args[2], "m") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };
        let v_arr = match js_array_to_tensor_handles(env, args[3], "v") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };

        if params.len() != grads.len() || params.len() != m_arr.len() || params.len() != v_arr.len() {
            throw_error(
                env,
                "INVALID_ARGS",
                "params, grads, m, and v must all have the same length",
            );
            return ptr::null_mut();
        }

        // Optional scalar hyperparameters
        let lr = if argc >= 5 {
            let mut d = 0.001f64;
            let _ = napi_get_value_double(env, args[4], &mut d);
            d as f32
        } else {
            0.001f32
        };

        let beta1 = if argc >= 6 {
            let mut d = 0.9f64;
            let _ = napi_get_value_double(env, args[5], &mut d);
            d as f32
        } else {
            0.9f32
        };

        let beta2 = if argc >= 7 {
            let mut d = 0.999f64;
            let _ = napi_get_value_double(env, args[6], &mut d);
            d as f32
        } else {
            0.999f32
        };

        let eps = if argc >= 8 {
            let mut d = 1e-8f64;
            let _ = napi_get_value_double(env, args[7], &mut d);
            d as f32
        } else {
            1e-8f32
        };

        let step: i64 = if argc >= 9 {
            let mut d = 1.0f64;
            let _ = napi_get_value_double(env, args[8], &mut d);
            (d as i64).max(1)
        } else {
            1i64
        };

        for i in 0..params.len() {
            if torsh_tensor_adam_step_inplace(
                params[i],
                grads[i],
                m_arr[i],
                v_arr[i],
                lr,
                beta1,
                beta2,
                eps,
                step,
            ) != TorshError::Success
            {
                throw_error(env, "OPERATION_FAILED", "adamStep: adam_step_inplace failed");
                return ptr::null_mut();
            }
        }

        return_undefined(env)
    }
}
