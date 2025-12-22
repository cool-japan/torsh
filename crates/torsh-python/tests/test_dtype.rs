//! Comprehensive unit tests for the dtype module
//!
//! Tests cover:
//! - DType creation from various strings
//! - DType name aliases (float32 vs f32, etc.)
//! - DType properties (name, itemsize, is_floating_point, is_signed)
//! - DType equality and hashing
//! - DType string representation
//! - Error handling for invalid/unsupported dtypes
//! - DType constants

use pyo3::prelude::*;
use std::ffi::CString;

/// Helper to run Python code and return the result
fn run_python_code<F, T>(code: &str, extract_fn: F) -> PyResult<T>
where
    F: FnOnce(&Bound<'_, PyAny>) -> PyResult<T>,
{
    Python::attach(|py| {
        let code_str = format!(
            "import sys\nsys.path.insert(0, '{}')\nimport torsh_python as torsh\n\n{}",
            env!("CARGO_MANIFEST_DIR"),
            code
        );
        let code_cstr = CString::new(code_str).unwrap();
        let filename = CString::new("").unwrap();
        let module_name = CString::new("").unwrap();

        let module = PyModule::from_code(py, &code_cstr, &filename, &module_name)?;

        let result = module.getattr("result")?;
        extract_fn(&result)
    })
}

// =============================================================================
// DType Creation Tests
// =============================================================================

#[test]
fn test_dtype_creation_float32() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("float32")
result = dtype.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "float32");
}

#[test]
fn test_dtype_creation_float32_alias() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("f32")
result = dtype.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "float32");
}

#[test]
fn test_dtype_creation_float64() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("float64")
result = dtype.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "float64");
}

#[test]
fn test_dtype_creation_int8() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("int8")
result = dtype.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int8");
}

#[test]
fn test_dtype_creation_int32() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("int32")
result = dtype.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int32");
}

#[test]
fn test_dtype_creation_int64() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("int64")
result = dtype.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int64");
}

#[test]
fn test_dtype_creation_uint8() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("uint8")
result = dtype.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "uint8");
}

#[test]
fn test_dtype_creation_bool() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("bool")
result = dtype.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "bool");
}

// =============================================================================
// DType Properties Tests
// =============================================================================

#[test]
fn test_dtype_itemsize_float32() {
    let result: usize = run_python_code(
        r#"
dtype = torsh.PyDType("float32")
result = dtype.itemsize
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, 4);
}

#[test]
fn test_dtype_itemsize_float64() {
    let result: usize = run_python_code(
        r#"
dtype = torsh.PyDType("float64")
result = dtype.itemsize
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, 8);
}

#[test]
fn test_dtype_itemsize_int8() {
    let result: usize = run_python_code(
        r#"
dtype = torsh.PyDType("int8")
result = dtype.itemsize
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, 1);
}

#[test]
fn test_dtype_itemsize_int16() {
    let result: usize = run_python_code(
        r#"
dtype = torsh.PyDType("int16")
result = dtype.itemsize
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, 2);
}

#[test]
fn test_dtype_itemsize_int32() {
    let result: usize = run_python_code(
        r#"
dtype = torsh.PyDType("int32")
result = dtype.itemsize
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, 4);
}

#[test]
fn test_dtype_itemsize_int64() {
    let result: usize = run_python_code(
        r#"
dtype = torsh.PyDType("int64")
result = dtype.itemsize
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, 8);
}

#[test]
fn test_dtype_itemsize_bool() {
    let result: usize = run_python_code(
        r#"
dtype = torsh.PyDType("bool")
result = dtype.itemsize
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, 1);
}

#[test]
fn test_dtype_is_floating_point_float32() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("float32")
result = dtype.is_floating_point
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_is_floating_point_float64() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("float64")
result = dtype.is_floating_point
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_is_floating_point_int32() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("int32")
result = dtype.is_floating_point
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(!result);
}

#[test]
fn test_dtype_is_floating_point_bool() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("bool")
result = dtype.is_floating_point
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(!result);
}

#[test]
fn test_dtype_is_signed_int32() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("int32")
result = dtype.is_signed
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_is_signed_int64() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("int64")
result = dtype.is_signed
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_is_signed_uint8() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("uint8")
result = dtype.is_signed
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(!result);
}

#[test]
fn test_dtype_is_signed_float32() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("float32")
result = dtype.is_signed
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_is_signed_bool() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("bool")
result = dtype.is_signed
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(!result);
}

// =============================================================================
// DType String Representation Tests
// =============================================================================

#[test]
fn test_dtype_str_float32() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("float32")
result = str(dtype)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "torch.float32");
}

#[test]
fn test_dtype_str_int64() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("int64")
result = str(dtype)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "torch.int64");
}

#[test]
fn test_dtype_repr_float32() {
    let result: String = run_python_code(
        r#"
dtype = torsh.PyDType("float32")
result = repr(dtype)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "torch.float32");
}

// =============================================================================
// DType Equality and Hashing Tests
// =============================================================================

#[test]
fn test_dtype_equality_same() {
    let result: bool = run_python_code(
        r#"
dtype1 = torsh.PyDType("float32")
dtype2 = torsh.PyDType("float32")
result = dtype1 == dtype2
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_equality_alias() {
    let result: bool = run_python_code(
        r#"
dtype1 = torsh.PyDType("float32")
dtype2 = torsh.PyDType("f32")
result = dtype1 == dtype2
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_equality_different() {
    let result: bool = run_python_code(
        r#"
dtype1 = torsh.PyDType("float32")
dtype2 = torsh.PyDType("float64")
result = dtype1 == dtype2
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(!result);
}

#[test]
fn test_dtype_hash_consistency() {
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import torsh_python as torsh

dtype1 = torsh.PyDType("float32")
dtype2 = torsh.PyDType("float32")
result = hash(dtype1) == hash(dtype2)
"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let code_cstr = CString::new(code).unwrap();
        let filename = CString::new("").unwrap();
        let module_name = CString::new("").unwrap();

        let module = PyModule::from_code(py, &code_cstr, &filename, &module_name).unwrap();

        let result: bool = module.getattr("result").unwrap().extract().unwrap();
        assert!(result);
    });
}

#[test]
fn test_dtype_in_set() {
    let result: bool = run_python_code(
        r#"
dtype1 = torsh.PyDType("float32")
dtype2 = torsh.PyDType("float32")
dtype_set = {dtype1}
result = dtype2 in dtype_set
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

// =============================================================================
// DType Error Handling Tests
// =============================================================================

#[test]
fn test_dtype_invalid_name() {
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import torsh_python as torsh

try:
    dtype = torsh.PyDType("invalid_dtype")
    result = False
except ValueError:
    result = True
"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let code_cstr = CString::new(code).unwrap();
        let filename = CString::new("").unwrap();
        let module_name = CString::new("").unwrap();

        let module = PyModule::from_code(py, &code_cstr, &filename, &module_name).unwrap();

        let result: bool = module.getattr("result").unwrap().extract().unwrap();
        assert!(result, "Should raise ValueError for invalid dtype name");
    });
}

#[test]
fn test_dtype_unsupported_uint16() {
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import torsh_python as torsh

try:
    dtype = torsh.PyDType("uint16")
    result = False
except ValueError:
    result = True
"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let code_cstr = CString::new(code).unwrap();
        let filename = CString::new("").unwrap();
        let module_name = CString::new("").unwrap();

        let module = PyModule::from_code(py, &code_cstr, &filename, &module_name).unwrap();

        let result: bool = module.getattr("result").unwrap().extract().unwrap();
        assert!(
            result,
            "Should raise ValueError for unsupported uint16 dtype"
        );
    });
}

// =============================================================================
// DType Constants Tests
// =============================================================================

#[test]
fn test_dtype_constant_float32() {
    let result: String = run_python_code(
        r#"
result = torsh.float32.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "float32");
}

#[test]
fn test_dtype_constant_float64() {
    let result: String = run_python_code(
        r#"
result = torsh.float64.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "float64");
}

#[test]
fn test_dtype_constant_int8() {
    let result: String = run_python_code(
        r#"
result = torsh.int8.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int8");
}

#[test]
fn test_dtype_constant_int32() {
    let result: String = run_python_code(
        r#"
result = torsh.int32.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int32");
}

#[test]
fn test_dtype_constant_int64() {
    let result: String = run_python_code(
        r#"
result = torsh.int64.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int64");
}

#[test]
fn test_dtype_constant_uint8() {
    let result: String = run_python_code(
        r#"
result = torsh.uint8.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "uint8");
}

#[test]
fn test_dtype_constant_bool() {
    let result: String = run_python_code(
        r#"
result = torsh.bool.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "bool");
}

// =============================================================================
// DType Alias Constants Tests
// =============================================================================

#[test]
fn test_dtype_alias_float() {
    let result: String = run_python_code(
        r#"
result = torsh.float.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "float32");
}

#[test]
fn test_dtype_alias_double() {
    let result: String = run_python_code(
        r#"
result = torsh.double.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "float64");
}

#[test]
fn test_dtype_alias_long() {
    let result: String = run_python_code(
        r#"
result = torsh.long.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int64");
}

#[test]
fn test_dtype_alias_int() {
    let result: String = run_python_code(
        r#"
result = torsh.int.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int32");
}

#[test]
fn test_dtype_alias_short() {
    let result: String = run_python_code(
        r#"
result = torsh.short.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int16");
}

#[test]
fn test_dtype_alias_char() {
    let result: String = run_python_code(
        r#"
result = torsh.char.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "int8");
}

#[test]
fn test_dtype_alias_byte() {
    let result: String = run_python_code(
        r#"
result = torsh.byte.name
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "uint8");
}

// =============================================================================
// DType Constants Equality Tests
// =============================================================================

#[test]
fn test_dtype_constants_equality_float32() {
    let result: bool = run_python_code(
        r#"
dtype = torsh.PyDType("float32")
result = dtype == torsh.float32
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_constants_equality_alias() {
    let result: bool = run_python_code(
        r#"
result = torsh.float == torsh.float32
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_dtype_constants_equality_different() {
    let result: bool = run_python_code(
        r#"
result = torsh.float32 == torsh.float64
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(!result);
}

// =============================================================================
// Comprehensive All DTypes Test
// =============================================================================

#[test]
fn test_all_supported_dtypes() {
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import torsh_python as torsh

# Test all supported dtypes
dtypes = [
    ("float32", 4, True, True),
    ("f32", 4, True, True),
    ("float64", 8, True, True),
    ("f64", 8, True, True),
    ("int8", 1, False, True),
    ("i8", 1, False, True),
    ("int16", 2, False, True),
    ("i16", 2, False, True),
    ("int32", 4, False, True),
    ("i32", 4, False, True),
    ("int64", 8, False, True),
    ("i64", 8, False, True),
    ("uint8", 1, False, False),
    ("u8", 1, False, False),
    ("uint32", 4, False, False),
    ("u32", 4, False, False),
    ("uint64", 8, False, False),
    ("u64", 8, False, False),
    ("bool", 1, False, False),
]

passed = 0
failed = 0

for dtype_str, expected_itemsize, expected_is_fp, expected_is_signed in dtypes:
    try:
        dtype = torsh.PyDType(dtype_str)
        assert dtype.itemsize == expected_itemsize, f"{{dtype_str}}: itemsize mismatch"
        assert dtype.is_floating_point == expected_is_fp, f"{{dtype_str}}: is_floating_point mismatch"
        assert dtype.is_signed == expected_is_signed, f"{{dtype_str}}: is_signed mismatch"
        passed += 1
    except Exception as e:
        print(f"Failed for {{dtype_str}}: {{e}}")
        failed += 1

result = (passed, failed)
"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let code_cstr = CString::new(code).unwrap();
        let filename = CString::new("").unwrap();
        let module_name = CString::new("").unwrap();

        let module = PyModule::from_code(py, &code_cstr, &filename, &module_name).unwrap();

        let result: (usize, usize) = module.getattr("result").unwrap().extract().unwrap();
        let (passed, failed) = result;

        println!("Passed: {}, Failed: {}", passed, failed);
        assert_eq!(failed, 0, "All dtype tests should pass");
        assert!(passed > 0, "Should have passed some tests");
    });
}
