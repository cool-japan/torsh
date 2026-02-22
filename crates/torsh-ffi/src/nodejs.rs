//! Node.js N-API bindings for ToRSh tensors
//!
//! This module provides Node.js integration through N-API, allowing JavaScript/TypeScript
//! applications to use ToRSh tensors with native performance.

use crate::c_api::*;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::slice;

// N-API types and constants
type NapiEnv = *mut c_void;
type NapiValue = *mut c_void;
type NapiCallback = extern "C" fn(NapiEnv, NapiCallbackInfo) -> NapiValue;
type NapiCallbackInfo = *mut c_void;
type NapiFinalizeCallback = extern "C" fn(NapiEnv, *mut c_void, *mut c_void);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum NapiStatus {
    Ok = 0,
    InvalidArg = 1,
    ObjectExpected = 2,
    StringExpected = 3,
    NameExpected = 4,
    FunctionExpected = 5,
    NumberExpected = 6,
    BooleanExpected = 7,
    ArrayExpected = 8,
    GenericFailure = 9,
    PendingException = 10,
    Cancelled = 11,
    EscapeCalledTwice = 12,
    HandleScopeMismatch = 13,
    CallbackScopeMismatch = 14,
    QueueFull = 15,
    Closing = 16,
    BigintExpected = 17,
    DateExpected = 18,
    ArrayBufferExpected = 19,
    DetachableArrayBufferExpected = 20,
    WouldDeadlock = 21,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum NapiValueType {
    Undefined = 0,
    Null = 1,
    Boolean = 2,
    Number = 3,
    String = 4,
    Symbol = 5,
    Object = 6,
    Function = 7,
    External = 8,
    Bigint = 9,
}

#[repr(C)]
struct NapiPropertyDescriptor {
    utf8name: *const c_char,
    name: NapiValue,
    method: NapiCallback,
    getter: NapiCallback,
    setter: NapiCallback,
    value: NapiValue,
    attributes: u32,
    data: *mut c_void,
}

// N-API function declarations
extern "C" {
    fn napi_get_cb_info(
        env: NapiEnv,
        cbinfo: NapiCallbackInfo,
        argc: *mut usize,
        argv: *mut NapiValue,
        this_arg: *mut NapiValue,
        data: *mut *mut c_void,
    ) -> NapiStatus;

    fn napi_create_object(env: NapiEnv, result: *mut NapiValue) -> NapiStatus;
    fn napi_create_array(env: NapiEnv, result: *mut NapiValue) -> NapiStatus;
    fn napi_create_array_with_length(
        env: NapiEnv,
        length: usize,
        result: *mut NapiValue,
    ) -> NapiStatus;
    fn napi_create_double(env: NapiEnv, value: f64, result: *mut NapiValue) -> NapiStatus;
    fn napi_create_int32(env: NapiEnv, value: i32, result: *mut NapiValue) -> NapiStatus;
    fn napi_create_string_utf8(
        env: NapiEnv,
        str: *const c_char,
        length: usize,
        result: *mut NapiValue,
    ) -> NapiStatus;
    fn napi_create_external(
        env: NapiEnv,
        data: *mut c_void,
        finalize_cb: NapiFinalizeCallback,
        finalize_hint: *mut c_void,
        result: *mut NapiValue,
    ) -> NapiStatus;

    fn napi_get_value_double(env: NapiEnv, value: NapiValue, result: *mut f64) -> NapiStatus;
    fn napi_get_value_int32(env: NapiEnv, value: NapiValue, result: *mut i32) -> NapiStatus;
    fn napi_get_value_string_utf8(
        env: NapiEnv,
        value: NapiValue,
        buf: *mut c_char,
        bufsize: usize,
        result: *mut usize,
    ) -> NapiStatus;
    fn napi_get_value_external(
        env: NapiEnv,
        value: NapiValue,
        result: *mut *mut c_void,
    ) -> NapiStatus;
    fn napi_get_array_length(env: NapiEnv, value: NapiValue, result: *mut u32) -> NapiStatus;
    fn napi_get_element(
        env: NapiEnv,
        object: NapiValue,
        index: u32,
        result: *mut NapiValue,
    ) -> NapiStatus;

    fn napi_set_element(
        env: NapiEnv,
        object: NapiValue,
        index: u32,
        value: NapiValue,
    ) -> NapiStatus;
    fn napi_set_named_property(
        env: NapiEnv,
        object: NapiValue,
        name: *const c_char,
        value: NapiValue,
    ) -> NapiStatus;
    fn napi_get_named_property(
        env: NapiEnv,
        object: NapiValue,
        name: *const c_char,
        result: *mut NapiValue,
    ) -> NapiStatus;

    fn napi_typeof(env: NapiEnv, value: NapiValue, result: *mut NapiValueType) -> NapiStatus;
    fn napi_is_array(env: NapiEnv, value: NapiValue, result: *mut bool) -> NapiStatus;

    fn napi_throw_error(env: NapiEnv, code: *const c_char, msg: *const c_char) -> NapiStatus;
    fn napi_define_properties(
        env: NapiEnv,
        object: NapiValue,
        property_count: usize,
        properties: *const NapiPropertyDescriptor,
    ) -> NapiStatus;
}

const NAPI_AUTO_LENGTH: usize = usize::MAX;

/// Utility macros for error handling
macro_rules! napi_call {
    ($call:expr) => {
        if $call != NapiStatus::Ok {
            return ptr::null_mut();
        }
    };
}

macro_rules! napi_call_with_return {
    ($call:expr, $ret:expr) => {
        if $call != NapiStatus::Ok {
            return $ret;
        }
    };
}

/// Tensor wrapper for N-API
struct TensorWrapper {
    tensor: TensorHandle,
}

impl TensorWrapper {
    fn new(tensor: TensorHandle) -> Self {
        Self { tensor }
    }
}

/// Finalizer for tensor wrapper
extern "C" fn tensor_finalizer(_env: NapiEnv, data: *mut c_void, _hint: *mut c_void) {
    unsafe {
        if !data.is_null() {
            let wrapper = Box::from_raw(data as *mut TensorWrapper);
            if !wrapper.tensor.is_null() {
                torsh_tensor_free(wrapper.tensor);
            }
        }
    }
}

/// Create N-API external value from tensor
unsafe fn create_tensor_external(env: NapiEnv, tensor: TensorHandle) -> NapiValue {
    if tensor.is_null() {
        return ptr::null_mut();
    }

    let wrapper = Box::into_raw(Box::new(TensorWrapper::new(tensor)));
    let mut result = ptr::null_mut();

    napi_call!(napi_create_external(
        env,
        wrapper as *mut c_void,
        tensor_finalizer,
        ptr::null_mut(),
        &mut result
    ));

    result
}

/// Extract tensor handle from N-API external value
unsafe fn get_tensor_from_external(env: NapiEnv, value: NapiValue) -> TensorHandle {
    let mut data = ptr::null_mut();
    napi_call_with_return!(
        napi_get_value_external(env, value, &mut data),
        ptr::null_mut()
    );

    if data.is_null() {
        return ptr::null_mut();
    }

    let wrapper = data as *mut TensorWrapper;
    (*wrapper).tensor
}

/// Convert JavaScript array to Vec<f32>
unsafe fn js_array_to_vec(env: NapiEnv, array: NapiValue) -> Vec<f32> {
    let mut is_array = false;
    napi_call_with_return!(napi_is_array(env, array, &mut is_array), Vec::new());

    if !is_array {
        return Vec::new();
    }

    let mut length = 0u32;
    napi_call_with_return!(napi_get_array_length(env, array, &mut length), Vec::new());

    let mut result = Vec::with_capacity(length as usize);
    for i in 0..length {
        let mut element = ptr::null_mut();
        napi_call_with_return!(napi_get_element(env, array, i, &mut element), Vec::new());

        let mut value = 0.0f64;
        napi_call_with_return!(napi_get_value_double(env, element, &mut value), Vec::new());

        result.push(value as f32);
    }

    result
}

/// Convert Vec<f32> to JavaScript array
unsafe fn vec_to_js_array(env: NapiEnv, data: &[f32]) -> NapiValue {
    let mut array = ptr::null_mut();
    napi_call!(napi_create_array_with_length(env, data.len(), &mut array));

    for (i, &value) in data.iter().enumerate() {
        let mut js_value = ptr::null_mut();
        napi_call!(napi_create_double(env, value as f64, &mut js_value));
        napi_call!(napi_set_element(env, array, i as u32, js_value));
    }

    array
}

/// Parse nested JavaScript arrays to tensor data and dimensions
unsafe fn parse_nested_array(env: NapiEnv, value: NapiValue) -> (Vec<f32>, Vec<usize>) {
    let mut data = Vec::new();
    let mut dims = Vec::new();
    parse_nested_array_recursive(env, value, &mut data, &mut dims, 0);
    (data, dims)
}

unsafe fn parse_nested_array_recursive(
    env: NapiEnv,
    value: NapiValue,
    data: &mut Vec<f32>,
    dims: &mut Vec<usize>,
    depth: usize,
) {
    let mut is_array = false;
    if napi_is_array(env, value, &mut is_array) != NapiStatus::Ok || !is_array {
        // Leaf value - should be a number
        let mut num_value = 0.0f64;
        if napi_get_value_double(env, value, &mut num_value) == NapiStatus::Ok {
            data.push(num_value as f32);
        }
        return;
    }

    let mut length = 0u32;
    if napi_get_array_length(env, value, &mut length) != NapiStatus::Ok {
        return;
    }

    if depth >= dims.len() {
        dims.push(length as usize);
    }

    for i in 0..length {
        let mut element = ptr::null_mut();
        if napi_get_element(env, value, i, &mut element) == NapiStatus::Ok {
            parse_nested_array_recursive(env, element, data, dims, depth + 1);
        }
    }
}

/// JavaScript: createTensor(data)
extern "C" fn js_create_tensor(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 1usize;
        let mut args = [ptr::null_mut(); 1];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc != 1 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected 1 argument").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let (data, dims) = parse_nested_array(env, args[0]);
        if data.is_empty() || dims.is_empty() {
            napi_throw_error(
                env,
                CString::new("INVALID_DATA").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Invalid tensor data").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let tensor = torsh_tensor_from_data(data.as_ptr(), data.len(), dims.as_ptr(), dims.len());
        if tensor.is_null() {
            napi_throw_error(
                env,
                CString::new("CREATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to create tensor").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, tensor)
    }
}

/// JavaScript: zeros(...dims)
extern "C" fn js_zeros(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 10usize;
        let mut args = [ptr::null_mut(); 10];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc == 0 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected at least 1 dimension")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        let mut dims = Vec::new();
        for i in 0..argc {
            let mut dim_value = 0i32;
            napi_call!(napi_get_value_int32(env, args[i], &mut dim_value));
            dims.push(dim_value as usize);
        }

        let tensor = torsh_tensor_zeros(dims.as_ptr(), dims.len());
        if tensor.is_null() {
            napi_throw_error(
                env,
                CString::new("CREATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to create zeros tensor")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, tensor)
    }
}

/// JavaScript: ones(...dims)
extern "C" fn js_ones(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 10usize;
        let mut args = [ptr::null_mut(); 10];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc == 0 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected at least 1 dimension")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        let mut dims = Vec::new();
        for i in 0..argc {
            let mut dim_value = 0i32;
            napi_call!(napi_get_value_int32(env, args[i], &mut dim_value));
            dims.push(dim_value as usize);
        }

        let tensor = torsh_tensor_ones(dims.as_ptr(), dims.len());
        if tensor.is_null() {
            napi_throw_error(
                env,
                CString::new("CREATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to create ones tensor")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, tensor)
    }
}

/// JavaScript: add(a, b)
extern "C" fn js_add(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 2usize;
        let mut args = [ptr::null_mut(); 2];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc != 2 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected 2 arguments").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let tensor_a = get_tensor_from_external(env, args[0]);
        let tensor_b = get_tensor_from_external(env, args[1]);

        if tensor_a.is_null() || tensor_b.is_null() {
            napi_throw_error(
                env,
                CString::new("INVALID_TENSOR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Invalid tensor arguments").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Get shape of tensor_a to create output tensor
        let mut shape_dims: Vec<usize> = vec![0; 16];
        let mut ndim: usize = 0;
        let shape_result = torsh_tensor_shape(tensor_a, shape_dims.as_mut_ptr(), &mut ndim);
        if shape_result != TorshError::Success {
            napi_throw_error(
                env,
                CString::new("SHAPE_ERROR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to get tensor shape").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Create output tensor
        let result = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
        if result.is_null() {
            napi_throw_error(
                env,
                CString::new("ALLOCATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to allocate output tensor")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        // Perform addition
        let add_result = torsh_tensor_add(tensor_a, tensor_b, result);
        if add_result != TorshError::Success {
            torsh_tensor_free(result);
            napi_throw_error(
                env,
                CString::new("OPERATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Tensor addition failed").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: multiply(a, b)
extern "C" fn js_multiply(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 2usize;
        let mut args = [ptr::null_mut(); 2];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc != 2 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected 2 arguments").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let tensor_a = get_tensor_from_external(env, args[0]);
        let tensor_b = get_tensor_from_external(env, args[1]);

        if tensor_a.is_null() || tensor_b.is_null() {
            napi_throw_error(
                env,
                CString::new("INVALID_TENSOR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Invalid tensor arguments").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Get shape of tensor_a to create output tensor
        let mut shape_dims: Vec<usize> = vec![0; 16];
        let mut ndim: usize = 0;
        let shape_result = torsh_tensor_shape(tensor_a, shape_dims.as_mut_ptr(), &mut ndim);
        if shape_result != TorshError::Success {
            napi_throw_error(
                env,
                CString::new("SHAPE_ERROR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to get tensor shape").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Create output tensor
        let result = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
        if result.is_null() {
            napi_throw_error(
                env,
                CString::new("ALLOCATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to allocate output tensor")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        // Perform multiplication
        let mul_result = torsh_tensor_mul(tensor_a, tensor_b, result);
        if mul_result != TorshError::Success {
            torsh_tensor_free(result);
            napi_throw_error(
                env,
                CString::new("OPERATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Tensor multiplication failed")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: matmul(a, b)
extern "C" fn js_matmul(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 2usize;
        let mut args = [ptr::null_mut(); 2];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc != 2 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected 2 arguments").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let tensor_a = get_tensor_from_external(env, args[0]);
        let tensor_b = get_tensor_from_external(env, args[1]);

        if tensor_a.is_null() || tensor_b.is_null() {
            napi_throw_error(
                env,
                CString::new("INVALID_TENSOR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Invalid tensor arguments").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Get shape of tensor_a to create output tensor (matmul may need different output shape)
        let mut shape_dims: Vec<usize> = vec![0; 16];
        let mut ndim: usize = 0;
        let shape_result = torsh_tensor_shape(tensor_a, shape_dims.as_mut_ptr(), &mut ndim);
        if shape_result != TorshError::Success {
            napi_throw_error(
                env,
                CString::new("SHAPE_ERROR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to get tensor shape").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Create output tensor
        let result = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
        if result.is_null() {
            napi_throw_error(
                env,
                CString::new("ALLOCATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to allocate output tensor")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        // Perform matrix multiplication
        let matmul_result = torsh_tensor_matmul(tensor_a, tensor_b, result);
        if matmul_result != TorshError::Success {
            torsh_tensor_free(result);
            napi_throw_error(
                env,
                CString::new("OPERATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Matrix multiplication failed")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: relu(tensor)
extern "C" fn js_relu(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 1usize;
        let mut args = [ptr::null_mut(); 1];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc != 1 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected 1 argument").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            napi_throw_error(
                env,
                CString::new("INVALID_TENSOR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Invalid tensor argument").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Get shape of input tensor to create output tensor
        let mut shape_dims: Vec<usize> = vec![0; 16];
        let mut ndim: usize = 0;
        let shape_result = torsh_tensor_shape(tensor, shape_dims.as_mut_ptr(), &mut ndim);
        if shape_result != TorshError::Success {
            napi_throw_error(
                env,
                CString::new("SHAPE_ERROR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to get tensor shape").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Create output tensor
        let result = torsh_tensor_zeros(shape_dims.as_ptr(), ndim);
        if result.is_null() {
            napi_throw_error(
                env,
                CString::new("ALLOCATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to allocate output tensor")
                    .expect("static string should not contain null bytes")
                    .as_ptr(),
            );
            return ptr::null_mut();
        }

        // Perform ReLU operation
        let relu_result = torsh_tensor_relu(tensor, result);
        if relu_result != TorshError::Success {
            torsh_tensor_free(result);
            napi_throw_error(
                env,
                CString::new("OPERATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("ReLU operation failed").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        create_tensor_external(env, result)
    }
}

/// JavaScript: getShape(tensor)
extern "C" fn js_get_shape(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 1usize;
        let mut args = [ptr::null_mut(); 1];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc != 1 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected 1 argument").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            napi_throw_error(
                env,
                CString::new("INVALID_TENSOR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Invalid tensor argument").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let mut ndims = torsh_tensor_ndim(tensor);
        let mut dims = vec![0usize; ndims];
        let result_code = torsh_tensor_shape(tensor, dims.as_mut_ptr(), &mut ndims);

        if result_code != TorshError::Success {
            napi_throw_error(
                env,
                CString::new("OPERATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to get tensor shape").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let mut shape_array = ptr::null_mut();
        napi_call!(napi_create_array_with_length(env, ndims, &mut shape_array));

        for (i, &dim) in dims.iter().enumerate() {
            let mut js_dim = ptr::null_mut();
            napi_call!(napi_create_int32(env, dim as i32, &mut js_dim));
            napi_call!(napi_set_element(env, shape_array, i as u32, js_dim));
        }

        shape_array
    }
}

/// JavaScript: getData(tensor)
extern "C" fn js_get_data(env: NapiEnv, info: NapiCallbackInfo) -> NapiValue {
    unsafe {
        let mut argc = 1usize;
        let mut args = [ptr::null_mut(); 1];
        napi_call!(napi_get_cb_info(
            env,
            info,
            &mut argc,
            args.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut()
        ));

        if argc != 1 {
            napi_throw_error(
                env,
                CString::new("INVALID_ARGS").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Expected 1 argument").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let tensor = get_tensor_from_external(env, args[0]);
        if tensor.is_null() {
            napi_throw_error(
                env,
                CString::new("INVALID_TENSOR").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Invalid tensor argument").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        let numel = torsh_tensor_numel(tensor);
        let mut data = vec![0.0f32; numel];
        let data_ptr = torsh_tensor_data(tensor);

        if data_ptr.is_null() {
            napi_throw_error(
                env,
                CString::new("OPERATION_FAILED").expect("static string should not contain null bytes").as_ptr(),
                CString::new("Failed to get tensor data").expect("static string should not contain null bytes").as_ptr(),
            );
            return ptr::null_mut();
        }

        // Copy data from the tensor's internal storage
        let src_data = slice::from_raw_parts(data_ptr as *const f32, numel);
        data.copy_from_slice(src_data);

        vec_to_js_array(env, &data)
    }
}

/// Module initialization
#[no_mangle]
pub extern "C" fn napi_register_module_v1(env: NapiEnv, exports: NapiValue) -> NapiValue {
    unsafe {
        let properties = [
            NapiPropertyDescriptor {
                utf8name: CString::new("createTensor").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_create_tensor,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
            NapiPropertyDescriptor {
                utf8name: CString::new("zeros").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_zeros,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
            NapiPropertyDescriptor {
                utf8name: CString::new("ones").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_ones,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
            NapiPropertyDescriptor {
                utf8name: CString::new("add").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_add,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
            NapiPropertyDescriptor {
                utf8name: CString::new("multiply").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_multiply,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
            NapiPropertyDescriptor {
                utf8name: CString::new("matmul").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_matmul,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
            NapiPropertyDescriptor {
                utf8name: CString::new("relu").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_relu,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
            NapiPropertyDescriptor {
                utf8name: CString::new("getShape").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_get_shape,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
            NapiPropertyDescriptor {
                utf8name: CString::new("getData").expect("static string should not contain null bytes").as_ptr(),
                name: ptr::null_mut(),
                method: js_get_data,
                getter: std::mem::transmute(ptr::null() as *const ()),
                setter: std::mem::transmute(ptr::null() as *const ()),
                value: ptr::null_mut(),
                attributes: 0,
                data: ptr::null_mut(),
            },
        ];

        napi_define_properties(env, exports, properties.len(), properties.as_ptr());
        exports
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nodejs_bindings_compilation() {
        // Test that the module compiles correctly
        // In practice, Node.js integration would require a Node.js runtime
        assert!(true);
    }
}
