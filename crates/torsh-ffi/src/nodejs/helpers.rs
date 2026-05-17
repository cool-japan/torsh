//! Low-level N-API helper types and conversion utilities shared by all submodules.

use std::os::raw::{c_char, c_void};
use std::ptr;

use crate::c_api::types::{TorshTensor, TorshError};
use crate::c_api::{torsh_tensor_free, torsh_tensor_shape, torsh_tensor_ndim};

// ─────────────────────────────────────────────────────────────────────────────
// N-API core types
// ─────────────────────────────────────────────────────────────────────────────

pub type NapiEnv = *mut c_void;
pub type NapiValue = *mut c_void;
pub type NapiCallback = extern "C" fn(NapiEnv, NapiCallbackInfo) -> NapiValue;
pub type NapiCallbackInfo = *mut c_void;
pub type NapiFinalizeCallback = extern "C" fn(NapiEnv, *mut c_void, *mut c_void);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NapiStatus {
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
pub enum NapiValueType {
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
pub struct NapiPropertyDescriptor {
    pub utf8name: *const c_char,
    pub name: NapiValue,
    pub method: NapiCallback,
    pub getter: NapiCallback,
    pub setter: NapiCallback,
    pub value: NapiValue,
    pub attributes: u32,
    pub data: *mut c_void,
}

// ─────────────────────────────────────────────────────────────────────────────
// N-API extern declarations
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {
    pub fn napi_get_cb_info(
        env: NapiEnv,
        cbinfo: NapiCallbackInfo,
        argc: *mut usize,
        argv: *mut NapiValue,
        this_arg: *mut NapiValue,
        data: *mut *mut c_void,
    ) -> NapiStatus;

    pub fn napi_create_object(env: NapiEnv, result: *mut NapiValue) -> NapiStatus;
    pub fn napi_create_array(env: NapiEnv, result: *mut NapiValue) -> NapiStatus;
    pub fn napi_create_array_with_length(
        env: NapiEnv,
        length: usize,
        result: *mut NapiValue,
    ) -> NapiStatus;
    pub fn napi_create_double(env: NapiEnv, value: f64, result: *mut NapiValue) -> NapiStatus;
    pub fn napi_create_int32(env: NapiEnv, value: i32, result: *mut NapiValue) -> NapiStatus;
    pub fn napi_create_bool(env: NapiEnv, value: i32, result: *mut NapiValue) -> NapiStatus;
    pub fn napi_create_string_utf8(
        env: NapiEnv,
        str: *const c_char,
        length: usize,
        result: *mut NapiValue,
    ) -> NapiStatus;
    pub fn napi_create_external(
        env: NapiEnv,
        data: *mut c_void,
        finalize_cb: NapiFinalizeCallback,
        finalize_hint: *mut c_void,
        result: *mut NapiValue,
    ) -> NapiStatus;

    pub fn napi_get_value_double(env: NapiEnv, value: NapiValue, result: *mut f64) -> NapiStatus;
    pub fn napi_get_value_int32(env: NapiEnv, value: NapiValue, result: *mut i32) -> NapiStatus;
    pub fn napi_get_value_uint32(env: NapiEnv, value: NapiValue, result: *mut u32) -> NapiStatus;
    pub fn napi_get_value_int64(env: NapiEnv, value: NapiValue, result: *mut i64) -> NapiStatus;
    pub fn napi_get_value_bool(env: NapiEnv, value: NapiValue, result: *mut bool) -> NapiStatus;
    pub fn napi_get_value_string_utf8(
        env: NapiEnv,
        value: NapiValue,
        buf: *mut c_char,
        bufsize: usize,
        result: *mut usize,
    ) -> NapiStatus;
    pub fn napi_get_value_external(
        env: NapiEnv,
        value: NapiValue,
        result: *mut *mut c_void,
    ) -> NapiStatus;
    pub fn napi_get_array_length(env: NapiEnv, value: NapiValue, result: *mut u32) -> NapiStatus;
    pub fn napi_get_element(
        env: NapiEnv,
        object: NapiValue,
        index: u32,
        result: *mut NapiValue,
    ) -> NapiStatus;

    pub fn napi_set_element(
        env: NapiEnv,
        object: NapiValue,
        index: u32,
        value: NapiValue,
    ) -> NapiStatus;
    pub fn napi_set_named_property(
        env: NapiEnv,
        object: NapiValue,
        name: *const c_char,
        value: NapiValue,
    ) -> NapiStatus;
    pub fn napi_get_named_property(
        env: NapiEnv,
        object: NapiValue,
        name: *const c_char,
        result: *mut NapiValue,
    ) -> NapiStatus;

    pub fn napi_typeof(env: NapiEnv, value: NapiValue, result: *mut NapiValueType) -> NapiStatus;
    pub fn napi_is_array(env: NapiEnv, value: NapiValue, result: *mut bool) -> NapiStatus;

    pub fn napi_throw_error(env: NapiEnv, code: *const c_char, msg: *const c_char) -> NapiStatus;
    pub fn napi_get_undefined(env: NapiEnv, result: *mut NapiValue) -> NapiStatus;
}

pub const NAPI_AUTO_LENGTH: usize = usize::MAX;

// ─────────────────────────────────────────────────────────────────────────────
// Convenience macros (local to this crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Return null if the N-API call fails.
#[macro_export]
macro_rules! napi_call {
    ($call:expr) => {
        if $call != $crate::nodejs::helpers::NapiStatus::Ok {
            return ::std::ptr::null_mut();
        }
    };
}

/// Return `$ret` if the N-API call fails.
#[macro_export]
macro_rules! napi_call_with_return {
    ($call:expr, $ret:expr) => {
        if $call != $crate::nodejs::helpers::NapiStatus::Ok {
            return $ret;
        }
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor wrapper (external value finaliser)
// ─────────────────────────────────────────────────────────────────────────────

pub type TensorHandle = *mut TorshTensor;

/// Heap-allocated wrapper passed as the external data for a NAPI external value.
pub struct TensorWrapper {
    pub tensor: TensorHandle,
}

impl TensorWrapper {
    pub fn new(tensor: TensorHandle) -> Self {
        Self { tensor }
    }
}

/// Called by Node.js GC when the external value is collected.
pub extern "C" fn tensor_finalizer(_env: NapiEnv, data: *mut c_void, _hint: *mut c_void) {
    unsafe {
        if !data.is_null() {
            let wrapper = Box::from_raw(data as *mut TensorWrapper);
            if !wrapper.tensor.is_null() {
                torsh_tensor_free(wrapper.tensor);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Wrap a native `TensorHandle` in a NAPI external value.
pub unsafe fn create_tensor_external(env: NapiEnv, tensor: TensorHandle) -> NapiValue {
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

/// Extract a `TensorHandle` from a NAPI external value.
pub unsafe fn get_tensor_from_external(env: NapiEnv, value: NapiValue) -> TensorHandle {
    let mut data = ptr::null_mut();
    napi_call_with_return!(
        napi_get_value_external(env, value, &mut data),
        ptr::null_mut()
    );
    if data.is_null() {
        return ptr::null_mut();
    }
    (*(data as *mut TensorWrapper)).tensor
}

/// Convert a flat JS number array to `Vec<f32>`.
pub unsafe fn js_array_to_vec(env: NapiEnv, array: NapiValue) -> Vec<f32> {
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

/// Convert a `&[f32]` to a JS number array.
pub unsafe fn vec_to_js_array(env: NapiEnv, data: &[f32]) -> NapiValue {
    let mut array = ptr::null_mut();
    napi_call!(napi_create_array_with_length(env, data.len(), &mut array));
    for (i, &value) in data.iter().enumerate() {
        let mut js_value = ptr::null_mut();
        napi_call!(napi_create_double(env, value as f64, &mut js_value));
        napi_call!(napi_set_element(env, array, i as u32, js_value));
    }
    array
}

/// Parse a nested JS array recursively, returning flat data and dimension sizes.
pub unsafe fn parse_nested_array(env: NapiEnv, value: NapiValue) -> (Vec<f32>, Vec<usize>) {
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

/// Read a shape from a NAPI array of ints; returns an empty vec on failure.
pub unsafe fn js_array_to_shape(env: NapiEnv, array: NapiValue) -> Vec<usize> {
    let mut is_array = false;
    if napi_is_array(env, array, &mut is_array) != NapiStatus::Ok || !is_array {
        return Vec::new();
    }
    let mut length = 0u32;
    if napi_get_array_length(env, array, &mut length) != NapiStatus::Ok {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(length as usize);
    for i in 0..length {
        let mut element = ptr::null_mut();
        if napi_get_element(env, array, i, &mut element) != NapiStatus::Ok {
            return Vec::new();
        }
        let mut v = 0i32;
        if napi_get_value_int32(env, element, &mut v) != NapiStatus::Ok {
            return Vec::new();
        }
        result.push(v as usize);
    }
    result
}

/// Throw a NAPI error with the given code/message and return null.
///
/// Caller is responsible for returning `null_mut()` after this.
pub unsafe fn throw_error(env: NapiEnv, code: &str, msg: &str) {
    use std::ffi::CString;
    let c_code = CString::new(code).unwrap_or_else(|_| CString::new("ERROR").expect("static"));
    let c_msg = CString::new(msg).unwrap_or_else(|_| CString::new("unknown").expect("static"));
    let _ = napi_throw_error(env, c_code.as_ptr(), c_msg.as_ptr());
}

/// Get the shape of a tensor as a `Vec<usize>`.  Returns `None` on failure.
pub unsafe fn tensor_shape_vec(tensor: TensorHandle) -> Option<Vec<usize>> {
    if tensor.is_null() {
        return None;
    }
    let ndim = torsh_tensor_ndim(tensor);
    let mut dims = vec![0usize; ndim];
    let err = torsh_tensor_shape(tensor, dims.as_mut_ptr(), &mut { ndim });
    if err != TorshError::Success {
        return None;
    }
    Some(dims)
}
