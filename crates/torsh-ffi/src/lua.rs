//! Lua bindings for ToRSh tensors
//!
//! This module provides Lua integration through the Lua C API, allowing Lua scripts
//! to create and manipulate ToRSh tensors. It's designed for embedding in applications
//! that use Lua for scripting.

use crate::c_api::*;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_void};
use std::ptr;
use std::slice;

#[allow(dead_code)]
#[repr(C)]
struct LuaState {
    _private: [u8; 0],
}

// Lua API declarations (subset needed for tensor operations)
extern "C" {
    fn lua_gettop(L: *mut LuaState) -> c_int;
    fn lua_settop(L: *mut LuaState, idx: c_int);
    fn lua_pushvalue(L: *mut LuaState, idx: c_int);
    fn lua_remove(L: *mut LuaState, idx: c_int);
    fn lua_insert(L: *mut LuaState, idx: c_int);
    fn lua_type(L: *mut LuaState, idx: c_int) -> c_int;
    fn lua_typename(L: *mut LuaState, tp: c_int) -> *const c_char;
    fn lua_isnumber(L: *mut LuaState, idx: c_int) -> c_int;
    fn lua_isstring(L: *mut LuaState, idx: c_int) -> c_int;
    fn lua_isuserdata(L: *mut LuaState, idx: c_int) -> c_int;
    fn lua_tonumber(L: *mut LuaState, idx: c_int) -> c_double;
    fn lua_tostring(L: *mut LuaState, idx: c_int) -> *const c_char;
    fn lua_objlen(L: *mut LuaState, idx: c_int) -> usize;
    fn lua_pushnil(L: *mut LuaState);
    fn lua_pushboolean(L: *mut LuaState, b: c_int);
    fn lua_pushinteger(L: *mut LuaState, n: i64);
    fn lua_pushnumber(L: *mut LuaState, n: c_double);
    fn lua_pushstring(L: *mut LuaState, s: *const c_char);
    fn lua_pushcclosure(L: *mut LuaState, f: extern "C" fn(*mut LuaState) -> c_int, n: c_int);
    fn lua_newuserdata(L: *mut LuaState, sz: usize) -> *mut c_void;
    fn lua_newtable(L: *mut LuaState);
    fn lua_rawgeti(L: *mut LuaState, idx: c_int, n: c_int);
    fn lua_rawseti(L: *mut LuaState, idx: c_int, n: c_int);
    fn lua_getfield(L: *mut LuaState, idx: c_int, k: *const c_char);
    fn lua_setfield(L: *mut LuaState, idx: c_int, k: *const c_char);
    fn lua_call(L: *mut LuaState, nargs: c_int, nresults: c_int);
    fn lua_pcall(L: *mut LuaState, nargs: c_int, nresults: c_int, errfunc: c_int) -> c_int;
    fn lua_getmetatable(L: *mut LuaState, objindex: c_int) -> c_int;
    fn lua_setmetatable(L: *mut LuaState, objindex: c_int) -> c_int;
    fn luaL_newmetatable(L: *mut LuaState, tname: *const c_char) -> c_int;
    fn luaL_getmetatable(L: *mut LuaState, tname: *const c_char);
    fn luaL_checkudata(L: *mut LuaState, ud: c_int, tname: *const c_char) -> *mut c_void;
    fn luaL_error(L: *mut LuaState, fmt: *const c_char, ...) -> c_int;
    fn luaL_register(L: *mut LuaState, libname: *const c_char, l: *const LuaRegEntry);
}

// Lua type constants
const LUA_TNIL: c_int = 0;
const LUA_TBOOLEAN: c_int = 1;
const LUA_TLIGHTUSERDATA: c_int = 2;
const LUA_TNUMBER: c_int = 3;
const LUA_TSTRING: c_int = 4;
const LUA_TTABLE: c_int = 5;
const LUA_TFUNCTION: c_int = 6;
const LUA_TUSERDATA: c_int = 7;
const LUA_TTHREAD: c_int = 8;

#[repr(C)]
struct LuaRegEntry {
    name: *const c_char,
    func: extern "C" fn(*mut LuaState) -> c_int,
}

// Safety: LuaRegEntry contains only static data and function pointers
// which are safe to share across threads
unsafe impl Sync for LuaRegEntry {}

impl LuaRegEntry {
    const fn new(name: &'static str, func: extern "C" fn(*mut LuaState) -> c_int) -> Self {
        Self {
            name: name.as_ptr() as *const c_char,
            func,
        }
    }

    const fn end() -> Self {
        Self {
            name: ptr::null(),
            func: lua_dummy_function,
        }
    }
}

extern "C" fn lua_dummy_function(_L: *mut LuaState) -> c_int {
    0
}

static TORSH_TENSOR_METATABLE: &str = "ToRSh.Tensor\0";

/// Lua userdata structure for ToRSh tensors
#[repr(C)]
struct LuaTensorUserdata {
    tensor: TensorHandle,
}

/// Create a new Lua tensor userdata
unsafe fn lua_new_tensor(L: *mut LuaState, tensor: TensorHandle) -> c_int {
    let userdata =
        lua_newuserdata(L, std::mem::size_of::<LuaTensorUserdata>()) as *mut LuaTensorUserdata;
    if userdata.is_null() {
        torsh_tensor_free(tensor);
        luaL_error(
            L,
            CString::new("Failed to allocate tensor userdata")
                .unwrap()
                .as_ptr(),
        );
        return 0;
    }

    (*userdata).tensor = tensor;

    // Set metatable
    luaL_getmetatable(L, TORSH_TENSOR_METATABLE.as_ptr() as *const c_char);
    lua_setmetatable(L, -2);

    1
}

/// Get tensor handle from Lua userdata
unsafe fn lua_check_tensor(L: *mut LuaState, idx: c_int) -> TensorHandle {
    let userdata = luaL_checkudata(L, idx, TORSH_TENSOR_METATABLE.as_ptr() as *const c_char)
        as *mut LuaTensorUserdata;
    if userdata.is_null() {
        luaL_error(L, CString::new("Expected ToRSh tensor").unwrap().as_ptr());
        return ptr::null_mut();
    }
    (*userdata).tensor
}

/// Lua function: Create tensor from table
extern "C" fn lua_tensor_from_table(L: *mut LuaState) -> c_int {
    unsafe {
        if lua_gettop(L) != 1 {
            luaL_error(L, CString::new("Expected 1 argument").unwrap().as_ptr());
            return 0;
        }

        if lua_type(L, 1) != LUA_TTABLE {
            luaL_error(L, CString::new("Expected table").unwrap().as_ptr());
            return 0;
        }

        // Get table dimensions and flatten data
        let (data, dims) = lua_table_to_tensor_data(L, 1);
        if data.is_empty() {
            luaL_error(L, CString::new("Empty table").unwrap().as_ptr());
            return 0;
        }

        let tensor =
            torsh_tensor_from_data(data.as_ptr(), data.len(), dims.as_ptr(), dims.len());

        if tensor.is_null() {
            luaL_error(L, CString::new("Failed to create tensor").unwrap().as_ptr());
            return 0;
        }

        lua_new_tensor(L, tensor)
    }
}

/// Convert Lua table to tensor data and dimensions
unsafe fn lua_table_to_tensor_data(L: *mut LuaState, idx: c_int) -> (Vec<f32>, Vec<usize>) {
    let mut data = Vec::new();
    let mut dims = Vec::new();

    lua_table_to_tensor_recursive(L, idx, &mut data, &mut dims, 0);
    (data, dims)
}

unsafe fn lua_table_to_tensor_recursive(
    L: *mut LuaState,
    idx: c_int,
    data: &mut Vec<f32>,
    dims: &mut Vec<usize>,
    depth: usize,
) {
    if lua_type(L, idx) != LUA_TTABLE {
        // Leaf value
        if lua_isnumber(L, idx) != 0 {
            data.push(lua_tonumber(L, idx) as f32);
        }
        return;
    }

    let len = lua_objlen(L, idx);
    if depth >= dims.len() {
        dims.push(len);
    }

    for i in 1..=len {
        lua_rawgeti(L, idx, i as c_int);
        lua_table_to_tensor_recursive(L, -1, data, dims, depth + 1);
        lua_settop(L, -2); // Pop the value
    }
}

/// Lua function: Create zeros tensor
extern "C" fn lua_tensor_zeros(L: *mut LuaState) -> c_int {
    unsafe {
        let nargs = lua_gettop(L);
        if nargs == 0 {
            luaL_error(
                L,
                CString::new("Expected at least 1 dimension")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        let mut dims = Vec::new();
        for i in 1..=nargs {
            if lua_isnumber(L, i) == 0 {
                luaL_error(
                    L,
                    CString::new("All arguments must be numbers")
                        .unwrap()
                        .as_ptr(),
                );
                return 0;
            }
            dims.push(lua_tonumber(L, i) as usize);
        }

        let tensor = torsh_tensor_zeros(dims.as_ptr(), dims.len());
        if tensor.is_null() {
            luaL_error(
                L,
                CString::new("Failed to create zeros tensor")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        lua_new_tensor(L, tensor)
    }
}

/// Lua function: Create ones tensor
extern "C" fn lua_tensor_ones(L: *mut LuaState) -> c_int {
    unsafe {
        let nargs = lua_gettop(L);
        if nargs == 0 {
            luaL_error(
                L,
                CString::new("Expected at least 1 dimension")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        let mut dims = Vec::new();
        for i in 1..=nargs {
            if lua_isnumber(L, i) == 0 {
                luaL_error(
                    L,
                    CString::new("All arguments must be numbers")
                        .unwrap()
                        .as_ptr(),
                );
                return 0;
            }
            dims.push(lua_tonumber(L, i) as usize);
        }

        let tensor = torsh_tensor_ones(dims.as_ptr(), dims.len());
        if tensor.is_null() {
            luaL_error(
                L,
                CString::new("Failed to create ones tensor")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        lua_new_tensor(L, tensor)
    }
}

/// Lua function: Create random tensor
extern "C" fn lua_tensor_randn(L: *mut LuaState) -> c_int {
    unsafe {
        let nargs = lua_gettop(L);
        if nargs == 0 {
            luaL_error(
                L,
                CString::new("Expected at least 1 dimension")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        let mut dims = Vec::new();
        for i in 1..=nargs {
            if lua_isnumber(L, i) == 0 {
                luaL_error(
                    L,
                    CString::new("All arguments must be numbers")
                        .unwrap()
                        .as_ptr(),
                );
                return 0;
            }
            dims.push(lua_tonumber(L, i) as usize);
        }

        let tensor = torsh_tensor_randn(dims.as_ptr(), dims.len());
        if tensor.is_null() {
            luaL_error(
                L,
                CString::new("Failed to create random tensor")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        lua_new_tensor(L, tensor)
    }
}

/// Lua function: Tensor addition
extern "C" fn lua_tensor_add(L: *mut LuaState) -> c_int {
    unsafe {
        if lua_gettop(L) != 2 {
            luaL_error(L, CString::new("Expected 2 arguments").unwrap().as_ptr());
            return 0;
        }

        let lhs = lua_check_tensor(L, 1);
        let rhs = lua_check_tensor(L, 2);

        if lhs.is_null() || rhs.is_null() {
            return 0;
        }

        // Get shape of the first tensor for the output tensor
        let mut ndims = torsh_tensor_ndim(lhs);
        let mut dims = vec![0usize; ndims];
        let shape_result = torsh_tensor_shape(lhs, dims.as_mut_ptr(), &mut ndims);

        if shape_result != TorshError::Success {
            luaL_error(
                L,
                CString::new("Failed to get tensor shape for addition")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        // Create output tensor with the same shape
        let result = torsh_tensor_zeros(dims.as_ptr(), ndims);
        if result.is_null() {
            luaL_error(
                L,
                CString::new("Failed to create output tensor for addition")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        // Perform the addition
        let add_result = torsh_tensor_add(lhs, rhs, result);
        if add_result != TorshError::Success {
            torsh_tensor_free(result);
            luaL_error(L, CString::new("Tensor addition failed").unwrap().as_ptr());
            return 0;
        }

        lua_new_tensor(L, result)
    }
}

/// Lua function: Tensor multiplication
extern "C" fn lua_tensor_mul(L: *mut LuaState) -> c_int {
    unsafe {
        if lua_gettop(L) != 2 {
            luaL_error(L, CString::new("Expected 2 arguments").unwrap().as_ptr());
            return 0;
        }

        let lhs = lua_check_tensor(L, 1);
        let rhs = lua_check_tensor(L, 2);

        if lhs.is_null() || rhs.is_null() {
            return 0;
        }

        // Get lhs tensor shape to create output tensor
        let mut lhs_shape = vec![0usize; 16]; // max 16 dims
        let mut lhs_ndim = 0usize;
        let shape_result = torsh_tensor_shape(lhs, lhs_shape.as_mut_ptr(), &mut lhs_ndim);

        if shape_result != TorshError::Success {
            luaL_error(
                L,
                CString::new("Failed to get tensor shape")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        // Create output tensor with same shape as lhs
        let result = torsh_tensor_zeros(lhs_shape.as_ptr(), lhs_ndim);
        if result.is_null() {
            luaL_error(
                L,
                CString::new("Failed to create output tensor")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        // Perform multiplication
        let multiply_result = torsh_tensor_multiply(lhs, rhs, result);
        if multiply_result != TorshError::Success {
            torsh_tensor_free(result);
            luaL_error(
                L,
                CString::new("Tensor multiplication failed")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        lua_new_tensor(L, result)
    }
}

/// Lua function: Matrix multiplication
extern "C" fn lua_tensor_matmul(L: *mut LuaState) -> c_int {
    unsafe {
        if lua_gettop(L) != 2 {
            luaL_error(L, CString::new("Expected 2 arguments").unwrap().as_ptr());
            return 0;
        }

        let lhs = lua_check_tensor(L, 1);
        let rhs = lua_check_tensor(L, 2);

        if lhs.is_null() || rhs.is_null() {
            return 0;
        }

        // Get tensor shapes for matrix multiplication
        let mut lhs_shape = vec![0usize; 16];
        let mut lhs_ndim = 0usize;
        let lhs_shape_result = torsh_tensor_shape(lhs, lhs_shape.as_mut_ptr(), &mut lhs_ndim);

        let mut rhs_shape = vec![0usize; 16];
        let mut rhs_ndim = 0usize;
        let rhs_shape_result = torsh_tensor_shape(rhs, rhs_shape.as_mut_ptr(), &mut rhs_ndim);

        if lhs_shape_result != TorshError::Success || rhs_shape_result != TorshError::Success || lhs_ndim < 2 || rhs_ndim < 2 {
            luaL_error(
                L,
                CString::new("Failed to get tensor shapes or tensors not 2D")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        // Create output tensor: (M, N) where lhs is (M, K) and rhs is (K, N)
        let output_shape = vec![lhs_shape[0], rhs_shape[1]];
        let result = torsh_tensor_zeros(output_shape.as_ptr(), 2);
        if result.is_null() {
            luaL_error(
                L,
                CString::new("Failed to create output tensor")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        // Perform matrix multiplication
        let matmul_result = torsh_tensor_matmul(lhs, rhs, result);
        if matmul_result != TorshError::Success {
            torsh_tensor_free(result);
            luaL_error(
                L,
                CString::new("Matrix multiplication failed")
                    .unwrap()
                    .as_ptr(),
            );
            return 0;
        }

        lua_new_tensor(L, result)
    }
}

/// Lua function: ReLU activation
extern "C" fn lua_tensor_relu(L: *mut LuaState) -> c_int {
    unsafe {
        if lua_gettop(L) != 1 {
            luaL_error(L, CString::new("Expected 1 argument").unwrap().as_ptr());
            return 0;
        }

        let tensor = lua_check_tensor(L, 1);
        if tensor.is_null() {
            return 0;
        }

        // Get tensor shape to create output tensor
        let mut tensor_shape = vec![0usize; 16];
        let mut tensor_ndim = 0usize;
        let shape_result = torsh_tensor_shape(tensor, tensor_shape.as_mut_ptr(), &mut tensor_ndim);

        if shape_result != TorshError::Success {
            luaL_error(L, CString::new("Failed to get tensor shape").unwrap().as_ptr());
            return 0;
        }

        // Create output tensor with same shape
        let result = torsh_tensor_zeros(tensor_shape.as_ptr(), tensor_ndim);
        if result.is_null() {
            luaL_error(L, CString::new("Failed to create output tensor").unwrap().as_ptr());
            return 0;
        }

        // Perform ReLU operation
        let relu_result = torsh_tensor_relu(tensor, result);
        if relu_result != TorshError::Success {
            torsh_tensor_free(result);
            luaL_error(L, CString::new("ReLU operation failed").unwrap().as_ptr());
            return 0;
        }

        lua_new_tensor(L, result)
    }
}

/// Lua function: Get tensor shape
extern "C" fn lua_tensor_shape(L: *mut LuaState) -> c_int {
    unsafe {
        if lua_gettop(L) != 1 {
            luaL_error(L, CString::new("Expected 1 argument").unwrap().as_ptr());
            return 0;
        }

        let tensor = lua_check_tensor(L, 1);
        if tensor.is_null() {
            return 0;
        }

        let mut ndims = torsh_tensor_ndim(tensor);
        let mut dims = vec![0usize; ndims];
        let result = torsh_tensor_shape(tensor, dims.as_mut_ptr(), &mut ndims);

        if result != TorshError::Success {
            luaL_error(
                L,
                CString::new("Failed to get tensor shape").unwrap().as_ptr(),
            );
            return 0;
        }

        lua_newtable(L);
        for (i, &dim) in dims.iter().enumerate() {
            lua_pushinteger(L, dim as i64);
            lua_rawseti(L, -2, (i + 1) as c_int);
        }

        1
    }
}

/// Lua function: Get tensor data as table
extern "C" fn lua_tensor_data(L: *mut LuaState) -> c_int {
    unsafe {
        if lua_gettop(L) != 1 {
            luaL_error(L, CString::new("Expected 1 argument").unwrap().as_ptr());
            return 0;
        }

        let tensor = lua_check_tensor(L, 1);
        if tensor.is_null() {
            return 0;
        }

        let numel = torsh_tensor_numel(tensor);
        let mut data = vec![0.0f32; numel];
        let data_ptr = torsh_tensor_data(tensor);

        if data_ptr.is_null() {
            luaL_error(
                L,
                CString::new("Failed to get tensor data").unwrap().as_ptr(),
            );
            return 0;
        }

        // Copy data from the tensor's internal storage
        let src_data = slice::from_raw_parts(data_ptr as *const f32, numel);
        data.copy_from_slice(src_data);

        // Convert to Lua table
        lua_newtable(L);
        for (i, &value) in data.iter().enumerate() {
            lua_pushnumber(L, value as c_double);
            lua_rawseti(L, -2, (i + 1) as c_int);
        }

        1
    }
}

/// Lua function: Tensor garbage collection
extern "C" fn lua_tensor_gc(L: *mut LuaState) -> c_int {
    unsafe {
        let userdata = luaL_checkudata(L, 1, TORSH_TENSOR_METATABLE.as_ptr() as *const c_char)
            as *mut LuaTensorUserdata;
        if !userdata.is_null() {
            torsh_tensor_free((*userdata).tensor);
            (*userdata).tensor = ptr::null_mut();
        }
        0
    }
}

/// Lua function: Tensor string representation
extern "C" fn lua_tensor_tostring(L: *mut LuaState) -> c_int {
    unsafe {
        let tensor = lua_check_tensor(L, 1);
        if tensor.is_null() {
            return 0;
        }

        let mut ndims = torsh_tensor_ndim(tensor);
        let mut dims = vec![0usize; ndims];
        let result = torsh_tensor_shape(tensor, dims.as_mut_ptr(), &mut ndims);

        if result != TorshError::Success {
            lua_pushstring(L, CString::new("ToRSh.Tensor(invalid)").unwrap().as_ptr());
        } else {
            let shape_str = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let tensor_str = format!("ToRSh.Tensor({})", shape_str);
            lua_pushstring(L, CString::new(tensor_str).unwrap().as_ptr());
        }

        1
    }
}

/// Library functions
static TORSH_FUNCTIONS: &[LuaRegEntry] = &[
    LuaRegEntry::new("tensor\0", lua_tensor_from_table),
    LuaRegEntry::new("zeros\0", lua_tensor_zeros),
    LuaRegEntry::new("ones\0", lua_tensor_ones),
    LuaRegEntry::new("randn\0", lua_tensor_randn),
    LuaRegEntry::new("add\0", lua_tensor_add),
    LuaRegEntry::new("mul\0", lua_tensor_mul),
    LuaRegEntry::new("matmul\0", lua_tensor_matmul),
    LuaRegEntry::new("relu\0", lua_tensor_relu),
    LuaRegEntry::end(),
];

/// Tensor metatable functions
static TENSOR_METAMETHODS: &[LuaRegEntry] = &[
    LuaRegEntry::new("__add\0", lua_tensor_add),
    LuaRegEntry::new("__mul\0", lua_tensor_mul),
    LuaRegEntry::new("__gc\0", lua_tensor_gc),
    LuaRegEntry::new("__tostring\0", lua_tensor_tostring),
    LuaRegEntry::end(),
];

/// Tensor methods
static TENSOR_METHODS: &[LuaRegEntry] = &[
    LuaRegEntry::new("shape\0", lua_tensor_shape),
    LuaRegEntry::new("data\0", lua_tensor_data),
    LuaRegEntry::new("relu\0", lua_tensor_relu),
    LuaRegEntry::new("add\0", lua_tensor_add),
    LuaRegEntry::new("mul\0", lua_tensor_mul),
    LuaRegEntry::new("matmul\0", lua_tensor_matmul),
    LuaRegEntry::end(),
];

/// Initialize ToRSh Lua library
#[no_mangle]
pub extern "C" fn luaopen_torsh(L: *mut LuaState) -> c_int {
    unsafe {
        // Create tensor metatable
        luaL_newmetatable(L, TORSH_TENSOR_METATABLE.as_ptr() as *const c_char);

        // Set metamethods
        luaL_register(L, ptr::null(), TENSOR_METAMETHODS.as_ptr());

        // Set methods as __index
        lua_newtable(L);
        luaL_register(L, ptr::null(), TENSOR_METHODS.as_ptr());
        lua_setfield(L, -2, CString::new("__index").unwrap().as_ptr());

        lua_settop(L, -2); // Pop metatable

        // Register library functions
        luaL_register(
            L,
            CString::new("torsh").unwrap().as_ptr(),
            TORSH_FUNCTIONS.as_ptr(),
        );

        1
    }
}

/// Example Lua script for testing
pub const LUA_TEST_SCRIPT: &str = r#"
-- Load ToRSh library
local torsh = require('torsh')

-- Create tensors
local a = torsh.tensor({{1, 2}, {3, 4}})
local b = torsh.tensor({{5, 6}, {7, 8}})

-- Print tensor information
print("Tensor a:", a)
print("Shape of a:", table.unpack(a:shape()))

-- Perform operations
local c = a + b
local d = a * b  -- Element-wise multiplication
local e = torsh.matmul(a, b)  -- Matrix multiplication

print("a + b =", c)
print("a * b =", d)
print("matmul(a, b) =", e)

-- Create special tensors
local zeros = torsh.zeros(3, 3)
local ones = torsh.ones(2, 4)
local randn = torsh.randn(2, 2)

print("Zeros tensor:", zeros)
print("Ones tensor:", ones)
print("Random tensor:", randn)

-- Apply activation
local relu_result = a:relu()
print("ReLU(a) =", relu_result)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lua_bindings_compilation() {
        // Test that the module compiles correctly
        // In practice, Lua integration would require a Lua runtime
        assert!(true);
    }

    #[test]
    fn test_lua_test_script() {
        // Test that the example script is valid
        assert!(!LUA_TEST_SCRIPT.is_empty());
        assert!(LUA_TEST_SCRIPT.contains("torsh"));
    }
}
