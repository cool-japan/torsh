//! NAPI utility handlers: getShape, getData, manualSeed, cudaAvailable,
//! cudaDeviceCount, saveTensor, loadTensor.

use std::os::raw::c_char;
use std::ptr;
use std::slice;

use crate::c_api::types::TorshError;
use crate::c_api::{
    torsh_cuda_device_count, torsh_cuda_is_available, torsh_set_random_seed, torsh_tensor_data,
    torsh_tensor_from_data, torsh_tensor_ndim, torsh_tensor_numel, torsh_tensor_shape,
};

use super::helpers::{
    create_tensor_external, get_tensor_from_external, throw_error, vec_to_js_array,
    NapiCallbackInfo, NapiEnv, NapiStatus, NapiValue,
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
    fn napi_get_value_string_utf8(
        env: NapiEnv,
        value: NapiValue,
        buf: *mut c_char,
        bufsize: usize,
        result: *mut usize,
    ) -> NapiStatus;
    fn napi_create_int32(env: NapiEnv, value: i32, result: *mut NapiValue) -> NapiStatus;
    fn napi_create_bool(env: NapiEnv, value: i32, result: *mut NapiValue) -> NapiStatus;
    fn napi_get_boolean(env: NapiEnv, value: i32, result: *mut NapiValue) -> NapiStatus;
    fn napi_create_array_with_length(
        env: NapiEnv,
        length: usize,
        result: *mut NapiValue,
    ) -> NapiStatus;
    fn napi_set_element(
        env: NapiEnv,
        object: NapiValue,
        index: u32,
        value: NapiValue,
    ) -> NapiStatus;
    fn napi_get_undefined(env: NapiEnv, result: *mut NapiValue) -> NapiStatus;
}

// ─────────────────────────────────────────────────────────────────────────────

/// JavaScript: `getShape(tensor)` — returns a JS array of dimension sizes.
pub extern "C" fn js_get_shape(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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

        let mut ndims = torsh_tensor_ndim(tensor);
        let mut dims = vec![0usize; ndims];
        let result_code = torsh_tensor_shape(tensor, dims.as_mut_ptr(), &mut ndims);

        if result_code != TorshError::Success {
            throw_error(env, "OPERATION_FAILED", "Failed to get tensor shape");
            return ptr::null_mut();
        }

        let mut shape_array = ptr::null_mut();
        if napi_create_array_with_length(env, ndims, &mut shape_array) != NapiStatus::Ok {
            return ptr::null_mut();
        }

        for (i, &dim) in dims.iter().enumerate() {
            let mut js_dim = ptr::null_mut();
            if napi_create_int32(env, dim as i32, &mut js_dim) != NapiStatus::Ok {
                return ptr::null_mut();
            }
            if napi_set_element(env, shape_array, i as u32, js_dim) != NapiStatus::Ok {
                return ptr::null_mut();
            }
        }

        shape_array
    }
}

/// JavaScript: `getData(tensor)` — returns a flat JS array of f32 values.
pub extern "C" fn js_get_data(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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

        let numel = torsh_tensor_numel(tensor);
        let data_ptr = torsh_tensor_data(tensor);

        if data_ptr.is_null() {
            throw_error(env, "OPERATION_FAILED", "Failed to get tensor data");
            return ptr::null_mut();
        }

        let src_data = slice::from_raw_parts(data_ptr as *const f32, numel);
        let data: Vec<f32> = src_data.to_vec();

        vec_to_js_array(env, &data)
    }
}

/// JavaScript: `manualSeed(seed)` — set the global random seed.
pub extern "C" fn js_manual_seed(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
            throw_error(env, "INVALID_ARGS", "manualSeed(seed) requires 1 argument");
            return ptr::null_mut();
        }

        let mut seed_d = 0.0f64;
        if napi_get_value_double(env, args[0], &mut seed_d) != NapiStatus::Ok {
            throw_error(env, "INVALID_ARGS", "seed must be a number");
            return ptr::null_mut();
        }

        torsh_set_random_seed(seed_d as u64);

        // Return undefined (void in JS)
        let mut undefined = ptr::null_mut();
        napi_get_undefined(env, &mut undefined);
        undefined
    }
}

/// JavaScript: `cudaAvailable()` — returns a boolean.
pub extern "C" fn js_cuda_available(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut _argc = 0usize;
        let _ = napi_get_cb_info(
            env,
            info,
            &mut _argc,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );

        let available = if torsh_cuda_is_available() != 0 {
            1i32
        } else {
            0i32
        };
        let mut result = ptr::null_mut();
        // Use napi_get_boolean instead of napi_create_bool for ABI compatibility
        if napi_get_boolean(env, available, &mut result) != NapiStatus::Ok {
            return ptr::null_mut();
        }
        result
    }
}

/// JavaScript: `cudaDeviceCount()` — returns the number of CUDA devices.
pub extern "C" fn js_cuda_device_count(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut _argc = 0usize;
        let _ = napi_get_cb_info(
            env,
            info,
            &mut _argc,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        );

        let count = torsh_cuda_device_count();
        let mut result = ptr::null_mut();
        if napi_create_int32(env, count, &mut result) != NapiStatus::Ok {
            return ptr::null_mut();
        }
        result
    }
}

/// JavaScript: `saveTensor(tensor, path)` — serialize tensor data to a file.
///
/// Format: 8-byte header `[ndim: u32, reserved: u32]`, then `ndim * 8` bytes
/// for shape (each dim as `u64` little-endian), then `numel * 4` bytes of
/// `f32` data.  This is intentionally simple, dependency-free, and
/// deterministic across platforms.
pub extern "C" fn js_save_tensor(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
                "saveTensor(tensor, path) requires 2 arguments",
            );
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        // Read path string
        let mut path_len = 0usize;
        if napi_get_value_string_utf8(env, args[1], ptr::null_mut(), 0, &mut path_len)
            != NapiStatus::Ok
        {
            throw_error(env, "INVALID_ARGS", "path must be a string");
            return ptr::null_mut();
        }
        let mut path_buf = vec![0u8; path_len + 1];
        let _ = napi_get_value_string_utf8(
            env,
            args[1],
            path_buf.as_mut_ptr() as *mut c_char,
            path_buf.len(),
            &mut path_len,
        );
        let path = match std::str::from_utf8(&path_buf[..path_len]) {
            Ok(s) => s.to_owned(),
            Err(_) => {
                throw_error(env, "INVALID_ARGS", "path contains invalid UTF-8");
                return ptr::null_mut();
            }
        };

        // Gather tensor metadata
        let mut ndims = torsh_tensor_ndim(tensor);
        let mut dims = vec![0usize; ndims];
        if torsh_tensor_shape(tensor, dims.as_mut_ptr(), &mut ndims) != TorshError::Success {
            throw_error(env, "OPERATION_FAILED", "Failed to get tensor shape");
            return ptr::null_mut();
        }
        let numel = torsh_tensor_numel(tensor);
        let data_ptr = torsh_tensor_data(tensor);
        if data_ptr.is_null() {
            throw_error(env, "OPERATION_FAILED", "Failed to get tensor data");
            return ptr::null_mut();
        }
        let data_slice = slice::from_raw_parts(data_ptr as *const f32, numel);

        // Build binary content
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(&(ndims as u32).to_le_bytes()); // 4 bytes: ndim
        bytes.extend_from_slice(&0u32.to_le_bytes()); // 4 bytes: reserved
        for &d in &dims {
            bytes.extend_from_slice(&(d as u64).to_le_bytes()); // 8 bytes per dim
        }
        for &v in data_slice {
            bytes.extend_from_slice(&v.to_le_bytes()); // 4 bytes per element
        }

        if let Err(e) = std::fs::write(&path, &bytes) {
            let msg = format!("Failed to write tensor to '{}': {}", path, e);
            throw_error(env, "IO_ERROR", &msg);
            return ptr::null_mut();
        }

        let mut undefined = ptr::null_mut();
        napi_get_undefined(env, &mut undefined);
        undefined
    }
}

/// JavaScript: `loadTensor(path)` — deserialise a tensor written by `saveTensor`.
pub extern "C" fn js_load_tensor(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
            throw_error(env, "INVALID_ARGS", "loadTensor(path) requires 1 argument");
            return ptr::null_mut();
        }

        // Read path string
        let mut path_len = 0usize;
        if napi_get_value_string_utf8(env, args[0], ptr::null_mut(), 0, &mut path_len)
            != NapiStatus::Ok
        {
            throw_error(env, "INVALID_ARGS", "path must be a string");
            return ptr::null_mut();
        }
        let mut path_buf = vec![0u8; path_len + 1];
        let _ = napi_get_value_string_utf8(
            env,
            args[0],
            path_buf.as_mut_ptr() as *mut c_char,
            path_buf.len(),
            &mut path_len,
        );
        let path = match std::str::from_utf8(&path_buf[..path_len]) {
            Ok(s) => s.to_owned(),
            Err(_) => {
                throw_error(env, "INVALID_ARGS", "path contains invalid UTF-8");
                return ptr::null_mut();
            }
        };

        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                let msg = format!("Failed to read tensor from '{}': {}", path, e);
                throw_error(env, "IO_ERROR", &msg);
                return ptr::null_mut();
            }
        };

        // Validate minimum header size (4 + 4 bytes)
        if bytes.len() < 8 {
            throw_error(env, "PARSE_ERROR", "File too short to be a valid tensor");
            return ptr::null_mut();
        }

        let ndims = u32::from_le_bytes(bytes[0..4].try_into().unwrap_or([0; 4])) as usize;
        // bytes[4..8] is reserved, ignore

        let shape_start = 8;
        let shape_end = shape_start + ndims * 8;
        if bytes.len() < shape_end {
            throw_error(env, "PARSE_ERROR", "File truncated in shape section");
            return ptr::null_mut();
        }

        let mut dims = Vec::with_capacity(ndims);
        for i in 0..ndims {
            let off = shape_start + i * 8;
            let d = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap_or([0; 8])) as usize;
            dims.push(d);
        }

        let numel: usize = if dims.is_empty() {
            1
        } else {
            dims.iter().product()
        };
        let data_start = shape_end;
        let data_end = data_start + numel * 4;
        if bytes.len() < data_end {
            throw_error(env, "PARSE_ERROR", "File truncated in data section");
            return ptr::null_mut();
        }

        let mut data = Vec::with_capacity(numel);
        for i in 0..numel {
            let off = data_start + i * 4;
            let v = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap_or([0; 4]));
            data.push(v);
        }

        let tensor = torsh_tensor_from_data(data.as_ptr(), data.len(), dims.as_ptr(), dims.len());
        if tensor.is_null() {
            throw_error(
                env,
                "CREATION_FAILED",
                "Failed to reconstruct tensor from file",
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, tensor)
    }
}
