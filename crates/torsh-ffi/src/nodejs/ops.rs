//! NAPI handlers for tensor math/manipulation operations:
//! add, sub, multiply, matmul, relu, addScalar, subScalar, mulScalar, divScalar,
//! transpose, reshape.

use std::ptr;

use crate::c_api::types::TorshError;
use crate::c_api::{
    torsh_tensor_add, torsh_tensor_add_scalar, torsh_tensor_div, torsh_tensor_div_scalar,
    torsh_tensor_free, torsh_tensor_matmul, torsh_tensor_mul, torsh_tensor_mul_scalar,
    torsh_tensor_relu, torsh_tensor_reshape, torsh_tensor_shape, torsh_tensor_sub,
    torsh_tensor_sub_scalar, torsh_tensor_transpose, torsh_tensor_zeros,
};

use super::helpers::{
    create_tensor_external, get_tensor_from_external, js_array_to_shape, throw_error,
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
    fn napi_is_array(env: NapiEnv, value: NapiValue, result: *mut bool) -> NapiStatus;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers (local)
// ─────────────────────────────────────────────────────────────────────────────

/// Allocate an output tensor with the same shape as `template`, invoke `op`,
/// free the output on failure, and wrap the result in a NAPI external.
///
/// `op` receives `(a, b, out)` and returns a `TorshError`.
unsafe fn binary_op(
    env: NapiEnv,
    info: NapiCallbackInfo,
    op_name: &str,
    op: unsafe extern "C" fn(
        *const crate::c_api::types::TorshTensor,
        *const crate::c_api::types::TorshTensor,
        *mut crate::c_api::types::TorshTensor,
    ) -> TorshError,
) -> NapiValue {
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
        throw_error(env, "INVALID_ARGS", "Expected 2 arguments");
        return ptr::null_mut();
    }

    let tensor_a = get_tensor_from_external(env, args[0]);
    let tensor_b = get_tensor_from_external(env, args[1]);
    if tensor_a.is_null() || tensor_b.is_null() {
        throw_error(env, "INVALID_TENSOR", "Invalid tensor arguments");
        return ptr::null_mut();
    }

    let mut shape_dims = vec![0usize; 16];
    let mut ndim = 0usize;
    if torsh_tensor_shape(tensor_a, shape_dims.as_mut_ptr(), &mut ndim) != TorshError::Success {
        throw_error(env, "SHAPE_ERROR", "Failed to get tensor shape");
        return ptr::null_mut();
    }

    let result = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
    if result.is_null() {
        throw_error(env, "ALLOCATION_FAILED", "Failed to allocate output tensor");
        return ptr::null_mut();
    }

    if op(tensor_a, tensor_b, result) != TorshError::Success {
        torsh_tensor_free(result);
        let msg = format!("{} operation failed", op_name);
        throw_error(env, "OPERATION_FAILED", &msg);
        return ptr::null_mut();
    }

    create_tensor_external(env, result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary tensor–tensor ops
// ─────────────────────────────────────────────────────────────────────────────

/// JavaScript: `add(a, b)` — element-wise tensor addition.
pub extern "C" fn js_add(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe { binary_op(env, info, "add", torsh_tensor_add) }
}

/// JavaScript: `sub(a, b)` — element-wise tensor subtraction.
pub extern "C" fn js_sub(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe { binary_op(env, info, "sub", torsh_tensor_sub) }
}

/// JavaScript: `multiply(a, b)` — element-wise tensor multiplication.
pub extern "C" fn js_multiply(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe { binary_op(env, info, "multiply", torsh_tensor_mul) }
}

/// JavaScript: `divide(a, b)` — element-wise tensor division.
pub extern "C" fn js_divide(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe { binary_op(env, info, "divide", torsh_tensor_div) }
}

/// JavaScript: `matmul(a, b)` — matrix multiplication.
pub extern "C" fn js_matmul(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
            throw_error(env, "INVALID_ARGS", "Expected 2 arguments");
            return ptr::null_mut();
        }

        let tensor_a = get_tensor_from_external(env, args[0]);
        let tensor_b = get_tensor_from_external(env, args[1]);
        if tensor_a.is_null() || tensor_b.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor arguments");
            return ptr::null_mut();
        }

        // For matmul the output shape is [m, n], not necessarily same as input.
        // We pass a placeholder shape and rely on torsh_tensor_matmul filling it.
        let mut shape_a = vec![0usize; 16];
        let mut ndim_a = 0usize;
        if torsh_tensor_shape(tensor_a, shape_a.as_mut_ptr(), &mut ndim_a) != TorshError::Success {
            throw_error(env, "SHAPE_ERROR", "Failed to get shape of A");
            return ptr::null_mut();
        }
        let mut shape_b = vec![0usize; 16];
        let mut ndim_b = 0usize;
        if torsh_tensor_shape(tensor_b, shape_b.as_mut_ptr(), &mut ndim_b) != TorshError::Success {
            throw_error(env, "SHAPE_ERROR", "Failed to get shape of B");
            return ptr::null_mut();
        }

        // Output shape: [shape_a[0], shape_b[ndim_b-1]]
        if ndim_a < 2 || ndim_b < 2 {
            throw_error(env, "SHAPE_ERROR", "matmul requires 2-D tensors");
            return ptr::null_mut();
        }
        let out_shape = [shape_a[0], shape_b[ndim_b - 1]];
        let result = torsh_tensor_zeros(out_shape.as_ptr(), 2);
        if result.is_null() {
            throw_error(env, "ALLOCATION_FAILED", "Failed to allocate output tensor");
            return ptr::null_mut();
        }

        if torsh_tensor_matmul(tensor_a, tensor_b, result) != TorshError::Success {
            torsh_tensor_free(result);
            throw_error(env, "OPERATION_FAILED", "Matrix multiplication failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: `relu(tensor)` — ReLU activation.
pub extern "C" fn js_relu(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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

        let mut shape_dims = vec![0usize; 16];
        let mut ndim = 0usize;
        if torsh_tensor_shape(tensor, shape_dims.as_mut_ptr(), &mut ndim) != TorshError::Success {
            throw_error(env, "SHAPE_ERROR", "Failed to get tensor shape");
            return ptr::null_mut();
        }

        let result = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
        if result.is_null() {
            throw_error(env, "ALLOCATION_FAILED", "Failed to allocate output tensor");
            return ptr::null_mut();
        }

        if torsh_tensor_relu(tensor, result) != TorshError::Success {
            torsh_tensor_free(result);
            throw_error(env, "OPERATION_FAILED", "ReLU operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar ops
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `(tensor, scalar)` args from NAPI; returns `(None, None)` on failure.
unsafe fn parse_tensor_scalar(
    env: NapiEnv,
    info: NapiCallbackInfo,
    op_name: &str,
) -> Option<(*mut crate::c_api::types::TorshTensor, f32)> {
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
        return None;
    }
    if argc != 2 {
        let msg = format!("{} expects (tensor, scalar)", op_name);
        throw_error(env, "INVALID_ARGS", &msg);
        return None;
    }
    let tensor = get_tensor_from_external(env, args[0]);
    if tensor.is_null() {
        throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
        return None;
    }
    let mut scalar = 0.0f64;
    if napi_get_value_double(env, args[1], &mut scalar) != NapiStatus::Ok {
        throw_error(env, "INVALID_ARGS", "Scalar must be a number");
        return None;
    }
    Some((tensor, scalar as f32))
}

/// JavaScript: `addScalar(tensor, scalar)`.
pub extern "C" fn js_add_scalar(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let (tensor, scalar) = match parse_tensor_scalar(env, info, "addScalar") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };
        let result = torsh_tensor_add_scalar(tensor, scalar);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "addScalar failed");
            return ptr::null_mut();
        }
        create_tensor_external(env, result)
    }
}

/// JavaScript: `subScalar(tensor, scalar)`.
pub extern "C" fn js_sub_scalar(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let (tensor, scalar) = match parse_tensor_scalar(env, info, "subScalar") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };
        let result = torsh_tensor_sub_scalar(tensor, scalar);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "subScalar failed");
            return ptr::null_mut();
        }
        create_tensor_external(env, result)
    }
}

/// JavaScript: `mulScalar(tensor, scalar)`.
pub extern "C" fn js_mul_scalar(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let (tensor, scalar) = match parse_tensor_scalar(env, info, "mulScalar") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };
        let result = torsh_tensor_mul_scalar(tensor, scalar);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "mulScalar failed");
            return ptr::null_mut();
        }
        create_tensor_external(env, result)
    }
}

/// JavaScript: `divScalar(tensor, scalar)`.
pub extern "C" fn js_div_scalar(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let (tensor, scalar) = match parse_tensor_scalar(env, info, "divScalar") {
            Some(v) => v,
            None => return ptr::null_mut(),
        };
        let result = torsh_tensor_div_scalar(tensor, scalar);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "divScalar failed");
            return ptr::null_mut();
        }
        create_tensor_external(env, result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape manipulation
// ─────────────────────────────────────────────────────────────────────────────

/// JavaScript: `transpose(tensor)` — swap the last two dimensions of a 2D tensor.
///
/// The TypeScript signature is `transpose(dim0?, dim1?)` but the C API only
/// supports 2-D transpose, so dim0/dim1 are accepted but ignored for now.
pub extern "C" fn js_transpose(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 3usize; // tensor [, dim0, dim1]
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
        if argc == 0 {
            throw_error(env, "INVALID_ARGS", "Expected tensor argument");
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        let result = torsh_tensor_transpose(tensor);
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "Transpose operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: `reshape(tensor, shapeArray)` — reshape tensor.
pub extern "C" fn js_reshape(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
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
                "reshape(tensor, shape) requires 2 arguments",
            );
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            throw_error(env, "INVALID_TENSOR", "Invalid tensor argument");
            return ptr::null_mut();
        }

        let shape = js_array_to_shape(env, args[1]);
        if shape.is_empty() {
            throw_error(
                env,
                "INVALID_ARGS",
                "shape must be a non-empty array of integers",
            );
            return ptr::null_mut();
        }

        let result = torsh_tensor_reshape(tensor, shape.as_ptr(), shape.len());
        if result.is_null() {
            throw_error(env, "OPERATION_FAILED", "Reshape operation failed");
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}
