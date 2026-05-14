//! NAPI handlers for reduction operations: sum, mean.

use std::ptr;

use crate::c_api::{
    torsh_tensor_mean_all, torsh_tensor_mean_dim, torsh_tensor_sum_all, torsh_tensor_sum_dim,
};

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
    fn napi_typeof(env: NapiEnv, value: NapiValue, result: *mut super::helpers::NapiValueType)
        -> NapiStatus;
}

use super::helpers::NapiValueType;

// ─────────────────────────────────────────────────────────────────────────────

/// JavaScript: `sum(tensor, dim?)` — reduce sum.
///
/// `dim` is optional; if omitted, sums all elements.
pub extern "C" fn js_sum(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
        if argc == 0 {
            throw_error(env, "INVALID_ARGS", "Expected at least 1 argument");
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        // Check if a dim argument was supplied
        let result = if argc >= 2 {
            let mut vtype = NapiValueType::Undefined;
            let _ = napi_typeof(env, args[1], &mut vtype);
            if vtype == NapiValueType::Number {
                let mut dim_d = 0.0f64;
                if napi_get_value_double(env, args[1], &mut dim_d) == NapiStatus::Ok {
                    torsh_tensor_sum_dim(tensor, dim_d as i32)
                } else {
                    torsh_tensor_sum_all(tensor)
                }
            } else {
                torsh_tensor_sum_all(tensor)
            }
        } else {
            torsh_tensor_sum_all(tensor)
        };

        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "Sum operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: `mean(tensor, dim?)` — reduce mean.
///
/// `dim` is optional; if omitted, averages all elements.
pub extern "C" fn js_mean(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
        if argc == 0 {
            throw_error(env, "INVALID_ARGS", "Expected at least 1 argument");
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        let result = if argc >= 2 {
            let mut vtype = NapiValueType::Undefined;
            let _ = napi_typeof(env, args[1], &mut vtype);
            if vtype == NapiValueType::Number {
                let mut dim_d = 0.0f64;
                if napi_get_value_double(env, args[1], &mut dim_d) == NapiStatus::Ok {
                    torsh_tensor_mean_dim(tensor, dim_d as i32)
                } else {
                    torsh_tensor_mean_all(tensor)
                }
            } else {
                torsh_tensor_mean_all(tensor)
            }
        } else {
            torsh_tensor_mean_all(tensor)
        };

        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "Mean operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}
