//! NAPI handlers for neural network operations: conv2d, log (needed for cross-entropy).

use std::ptr;

use crate::c_api::types::TorshError;
use crate::c_api::{torsh_tensor_conv2d, torsh_tensor_log, torsh_tensor_softmax};

use super::helpers::{
    create_tensor_external, get_tensor_from_external, throw_error, NapiCallbackInfo, NapiEnv,
    NapiStatus, NapiValue,
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
    fn napi_typeof(
        env: NapiEnv,
        value: NapiValue,
        result: *mut super::helpers::NapiValueType,
    ) -> NapiStatus;
    fn napi_get_value_int32(env: NapiEnv, value: NapiValue, result: *mut i32) -> NapiStatus;
}

use super::helpers::NapiValueType;

// ─────────────────────────────────────────────────────────────────────────────

/// JavaScript: `log(tensor)` — element-wise natural logarithm.
///
/// Exposed so TypeScript can compose cross-entropy from primitives.
pub extern "C" fn js_log(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 1usize;
        let mut args = [ptr::null_mut(); 1];
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
        if argc != 1 {
            throw_error(env, "INVALID_ARGS", "log(tensor) requires 1 argument");
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        let result = torsh_tensor_log(tensor);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "Log operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: `conv2d(input, weight, bias?, stride?, padding?)`.
///
/// - `input`  : 4-D tensor [N, C, H, W]
/// - `weight` : 4-D tensor [out_channels, in_channels, kH, kW]
/// - `bias`   : optional 1-D tensor [out_channels] or `null`
/// - `stride` : integer stride (default 1)
/// - `padding`: integer padding (default 0)
///
/// Returns a new 4-D tensor [N, out_channels, H_out, W_out].
pub extern "C" fn js_conv2d(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 5usize;
        let mut args = [ptr::null_mut(); 5];
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
        if argc < 2 {
            throw_error(
                env,
                "INVALID_ARGS",
                "conv2d(input, weight [, bias, stride, padding]) requires at least 2 arguments",
            );
            return ptr::null_mut();
        }

        let input = get_tensor_from_external(env, args[0]);
        if input.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid input tensor");
            return ptr::null_mut();
        }

        let weight = get_tensor_from_external(env, args[1]);
        if weight.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid weight tensor");
            return ptr::null_mut();
        }

        // Bias: optional (arg index 2), may be null / undefined
        let bias = if argc >= 3 {
            let mut vtype = NapiValueType::Undefined;
            let _ = napi_typeof(env, args[2], &mut vtype);
            if vtype == NapiValueType::External {
                get_tensor_from_external(env, args[2])
            } else {
                ptr::null_mut()
            }
        } else {
            ptr::null_mut()
        };

        // Stride (arg index 3, default 1)
        let stride: usize = if argc >= 4 {
            let mut s = 0i32;
            if napi_get_value_int32(env, args[3], &mut s) == NapiStatus::Ok && s > 0 {
                s as usize
            } else {
                1
            }
        } else {
            1
        };

        // Padding (arg index 4, default 0)
        let padding: usize = if argc >= 5 {
            let mut p = 0i32;
            if napi_get_value_int32(env, args[4], &mut p) == NapiStatus::Ok && p >= 0 {
                p as usize
            } else {
                0
            }
        } else {
            0
        };

        let result = torsh_tensor_conv2d(input, weight, bias, stride, padding);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "conv2d operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: `crossEntropyLoss(prediction, target)`.
///
/// Computes `-mean( sum(target * log(softmax(pred) + eps), dim=1) )`.
///
/// The `eps` (1e-7) avoids log(0).  `dim=1` assumes `pred` is shaped [N, C].
/// Both tensors must have the same shape.
///
/// This is a composed implementation using existing C API primitives.
pub extern "C" fn js_cross_entropy_loss(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 2usize;
        let mut args = [ptr::null_mut(); 2];
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
        if argc != 2 {
            throw_error(
                env,
                "INVALID_ARGS",
                "crossEntropyLoss(prediction, target) requires 2 arguments",
            );
            return ptr::null_mut();
        }

        let pred = get_tensor_from_external(env, args[0]);
        let target = get_tensor_from_external(env, args[1]);
        if pred.is_null() || target.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        // Step 1: softmax(pred, dim=1) → probs
        let probs = torsh_tensor_softmax(pred, 1);
        if probs.is_null() {
            throw_error(env, "OPERATION_FAILED", "crossEntropyLoss: softmax failed");
            return ptr::null_mut();
        }

        // Step 2: add eps scalar to probs (in-place-style via add_scalar)
        use crate::c_api::torsh_tensor_add_scalar;
        let probs_eps = torsh_tensor_add_scalar(probs, 1e-7f32);
        crate::c_api::torsh_tensor_free(probs);
        if probs_eps.is_null() {
            throw_error(env, "OPERATION_FAILED", "crossEntropyLoss: add eps failed");
            return ptr::null_mut();
        }

        // Step 3: log(probs_eps) → log_probs
        let log_probs = torsh_tensor_log(probs_eps);
        crate::c_api::torsh_tensor_free(probs_eps);
        if log_probs.is_null() {
            throw_error(env, "OPERATION_FAILED", "crossEntropyLoss: log failed");
            return ptr::null_mut();
        }

        // Step 4: target * log_probs (element-wise) → prod
        use super::helpers::tensor_shape_vec;
        use crate::c_api::{torsh_tensor_mul, torsh_tensor_zeros};
        let log_shape = match tensor_shape_vec(log_probs) {
            Some(s) => s,
            None => {
                crate::c_api::torsh_tensor_free(log_probs);
                throw_error(
                    env,
                    "OPERATION_FAILED",
                    "crossEntropyLoss: failed to get shape",
                );
                return ptr::null_mut();
            }
        };

        let prod_buf = torsh_tensor_zeros(log_shape.as_ptr(), log_shape.len());
        if prod_buf.is_null() {
            crate::c_api::torsh_tensor_free(log_probs);
            throw_error(env, "ALLOCATION_FAILED", "crossEntropyLoss: alloc failed");
            return ptr::null_mut();
        }

        if torsh_tensor_mul(target, log_probs, prod_buf) != TorshError::Success {
            crate::c_api::torsh_tensor_free(log_probs);
            crate::c_api::torsh_tensor_free(prod_buf);
            throw_error(env, "OPERATION_FAILED", "crossEntropyLoss: mul failed");
            return ptr::null_mut();
        }
        crate::c_api::torsh_tensor_free(log_probs);

        // Step 5: sum_all(prod) and negate and divide by N (mean)
        use crate::c_api::{torsh_tensor_mul_scalar, torsh_tensor_sum_all};
        let sum = torsh_tensor_sum_all(prod_buf);
        crate::c_api::torsh_tensor_free(prod_buf);
        if sum.is_null() {
            throw_error(env, "OPERATION_FAILED", "crossEntropyLoss: sum failed");
            return ptr::null_mut();
        }

        // Compute N = first dim (batch size)
        let n_batch = log_shape.first().copied().unwrap_or(1).max(1) as f32;

        // result = -sum / N_batch
        let result = torsh_tensor_mul_scalar(sum, -1.0 / n_batch);
        crate::c_api::torsh_tensor_free(sum);
        if result.is_null() {
            throw_error(
                env,
                "OPERATION_FAILED",
                "crossEntropyLoss: negate/scale failed",
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}
