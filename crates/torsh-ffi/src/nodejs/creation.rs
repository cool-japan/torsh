//! NAPI handlers for tensor creation: zeros, ones, randn, eye, linspace, createTensor.

use std::ptr;

use crate::c_api::{
    torsh_tensor_eye, torsh_tensor_from_data, torsh_tensor_linspace, torsh_tensor_ones,
    torsh_tensor_randn, torsh_tensor_zeros,
};

use super::helpers::{
    create_tensor_external, parse_nested_array, throw_error, NapiCallbackInfo, NapiEnv, NapiStatus,
    NapiValue,
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
    fn napi_get_value_int32(env: NapiEnv, value: NapiValue, result: *mut i32) -> NapiStatus;
    fn napi_get_value_double(env: NapiEnv, value: NapiValue, result: *mut f64) -> NapiStatus;
    fn napi_throw_error(
        env: NapiEnv,
        code: *const std::os::raw::c_char,
        msg: *const std::os::raw::c_char,
    ) -> NapiStatus;
}

// ─────────────────────────────────────────────────────────────────────────────

/// JavaScript: `createTensor(data)` — create tensor from nested array.
pub extern "C" fn js_create_tensor(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
            throw_error(env, "INVALID_ARGS", "Expected 1 argument");
            return ptr::null_mut();
        }

        let (data, dims) = parse_nested_array(env, args[0]);
        if data.is_empty() || dims.is_empty() {
            throw_error(env, "INVALID_DATA", "Invalid tensor data");
            return ptr::null_mut();
        }

        let tensor = torsh_tensor_from_data(data.as_ptr(), data.len(), dims.as_ptr(), dims.len());
        if tensor.is_null() {
            throw_error(env, "CREATION_FAILED", "Failed to create tensor");
            return ptr::null_mut();
        }

        create_tensor_external(env, tensor)
    }
}

/// JavaScript: `zeros(...dims)` — tensor filled with 0s.
pub extern "C" fn js_zeros(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 10usize;
        let mut args = [ptr::null_mut(); 10];
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
            throw_error(env, "INVALID_ARGS", "Expected at least 1 dimension");
            return ptr::null_mut();
        }
        let mut dims = Vec::new();
        for i in 0..argc {
            let mut d = 0i32;
            if napi_get_value_int32(env, args[i], &mut d) != NapiStatus::Ok {
                throw_error(env, "INVALID_ARGS", "Dimensions must be integers");
                return ptr::null_mut();
            }
            dims.push(d as usize);
        }
        let tensor = torsh_tensor_zeros(dims.as_ptr(), dims.len());
        if tensor.is_null() {
            throw_error(env, "CREATION_FAILED", "Failed to create zeros tensor");
            return ptr::null_mut();
        }
        create_tensor_external(env, tensor)
    }
}

/// JavaScript: `ones(...dims)` — tensor filled with 1s.
pub extern "C" fn js_ones(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 10usize;
        let mut args = [ptr::null_mut(); 10];
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
            throw_error(env, "INVALID_ARGS", "Expected at least 1 dimension");
            return ptr::null_mut();
        }
        let mut dims = Vec::new();
        for i in 0..argc {
            let mut d = 0i32;
            if napi_get_value_int32(env, args[i], &mut d) != NapiStatus::Ok {
                throw_error(env, "INVALID_ARGS", "Dimensions must be integers");
                return ptr::null_mut();
            }
            dims.push(d as usize);
        }
        let tensor = torsh_tensor_ones(dims.as_ptr(), dims.len());
        if tensor.is_null() {
            throw_error(env, "CREATION_FAILED", "Failed to create ones tensor");
            return ptr::null_mut();
        }
        create_tensor_external(env, tensor)
    }
}

/// JavaScript: `randn(...dims)` — tensor with values drawn from N(0,1).
pub extern "C" fn js_randn(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 10usize;
        let mut args = [ptr::null_mut(); 10];
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
            throw_error(env, "INVALID_ARGS", "Expected at least 1 dimension");
            return ptr::null_mut();
        }
        let mut dims = Vec::new();
        for i in 0..argc {
            let mut d = 0i32;
            if napi_get_value_int32(env, args[i], &mut d) != NapiStatus::Ok {
                throw_error(env, "INVALID_ARGS", "Dimensions must be integers");
                return ptr::null_mut();
            }
            dims.push(d as usize);
        }
        let tensor = torsh_tensor_randn(dims.as_ptr(), dims.len());
        if tensor.is_null() {
            throw_error(env, "CREATION_FAILED", "Failed to create randn tensor");
            return ptr::null_mut();
        }
        create_tensor_external(env, tensor)
    }
}

/// JavaScript: `eye(n)` — n×n identity matrix.
pub extern "C" fn js_eye(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
            throw_error(env, "INVALID_ARGS", "eye(n) requires exactly 1 argument");
            return ptr::null_mut();
        }
        let mut n = 0i32;
        if napi_get_value_int32(env, args[0], &mut n) != NapiStatus::Ok || n <= 0 {
            throw_error(env, "INVALID_ARGS", "n must be a positive integer");
            return ptr::null_mut();
        }
        let tensor = torsh_tensor_eye(n as usize);
        if tensor.is_null() {
            throw_error(env, "CREATION_FAILED", "Failed to create identity matrix");
            return ptr::null_mut();
        }
        create_tensor_external(env, tensor)
    }
}

/// JavaScript: `linspace(start, end, steps)` — linearly-spaced 1D tensor.
pub extern "C" fn js_linspace(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
            throw_error(
                env,
                "INVALID_ARGS",
                "linspace(start, end, steps) requires 3 arguments",
            );
            return ptr::null_mut();
        }
        let mut start = 0.0f64;
        let mut end = 0.0f64;
        let mut steps_d = 0.0f64;
        if napi_get_value_double(env, args[0], &mut start) != NapiStatus::Ok
            || napi_get_value_double(env, args[1], &mut end) != NapiStatus::Ok
            || napi_get_value_double(env, args[2], &mut steps_d) != NapiStatus::Ok
        {
            throw_error(env, "INVALID_ARGS", "Arguments must be numbers");
            return ptr::null_mut();
        }
        let steps = steps_d as usize;
        if steps == 0 {
            throw_error(env, "INVALID_ARGS", "steps must be > 0");
            return ptr::null_mut();
        }
        let tensor = torsh_tensor_linspace(start, end, steps);
        if tensor.is_null() {
            throw_error(env, "CREATION_FAILED", "Failed to create linspace tensor");
            return ptr::null_mut();
        }
        create_tensor_external(env, tensor)
    }
}
