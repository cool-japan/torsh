//! NAPI handlers for tensor clone and detach.

use std::ptr;

use crate::c_api::{torsh_tensor_clone, torsh_tensor_detach};

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
}

// ─────────────────────────────────────────────────────────────────────────────

/// JavaScript: `clone(tensor)` — deep-copy a tensor.
pub extern "C" fn js_clone(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        let result = torsh_tensor_clone(tensor);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "Clone operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: `detach(tensor)` — detach tensor from the autograd graph.
pub extern "C" fn js_detach(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        let result = torsh_tensor_detach(tensor);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "Detach operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}
