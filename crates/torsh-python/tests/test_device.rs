//! Comprehensive unit tests for the device module
//!
//! Tests cover:
//! - Device creation from various inputs
//! - Device string parsing
//! - Device properties (type, index)
//! - Device equality and hashing
//! - Device utility functions
//! - Error handling for invalid inputs

use pyo3::prelude::*;
use std::ffi::CString;

/// Initialise the Python interpreter for standalone test binaries.
/// `Python::initialize()` is idempotent — safe to call from multiple tests.
fn init_py() {
    Python::initialize();
}

/// Helper to run Python code and return the result
fn run_python_code<F, T>(code: &str, extract_fn: F) -> PyResult<T>
where
    F: FnOnce(&Bound<'_, PyAny>) -> PyResult<T>,
{
    init_py();
    Python::attach(|py| {
        let code_str = format!(
            "import sys\nsys.path.insert(0, '{}')\nimport rstorch_python as rstorch\n\n{}",
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

#[test]
fn test_device_creation_cpu() {
    let result: String = run_python_code(
        r#"
result = str(rstorch.PyDevice("cpu"))
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cpu");
}

#[test]
fn test_device_creation_cuda_default() {
    let result: String = run_python_code(
        r#"
result = str(rstorch.PyDevice("cuda"))
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cuda:0");
}

#[test]
fn test_device_creation_cuda_with_index() {
    let result: String = run_python_code(
        r#"
result = str(rstorch.PyDevice("cuda:2"))
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cuda:2");
}

#[test]
fn test_device_creation_metal_default() {
    let result: String = run_python_code(
        r#"
result = str(rstorch.PyDevice("metal"))
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "metal:0");
}

#[test]
fn test_device_creation_metal_with_index() {
    let result: String = run_python_code(
        r#"
result = str(rstorch.PyDevice("metal:1"))
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "metal:1");
}

#[test]
fn test_device_creation_from_integer() {
    let result: String = run_python_code(
        r#"
result = str(rstorch.PyDevice(3))
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cuda:3");
}

#[test]
fn test_device_type_property_cpu() {
    let result: String = run_python_code(
        r#"
device = rstorch.PyDevice("cpu")
result = device.type
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cpu");
}

#[test]
fn test_device_type_property_cuda() {
    let result: String = run_python_code(
        r#"
device = rstorch.PyDevice("cuda:5")
result = device.type
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cuda");
}

#[test]
fn test_device_index_property_cpu() {
    init_py();
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import rstorch_python as rstorch

device = rstorch.PyDevice("cpu")
result = device.index
"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let code_cstr = CString::new(code).unwrap();
        let filename = CString::new("").unwrap();
        let module_name = CString::new("").unwrap();

        let module = PyModule::from_code(py, &code_cstr, &filename, &module_name).unwrap();

        let result = module.getattr("result").unwrap();
        assert!(result.is_none());
    });
}

#[test]
fn test_device_index_property_cuda() {
    let result: u32 = run_python_code(
        r#"
device = rstorch.PyDevice("cuda:7")
result = device.index
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, 7);
}

#[test]
fn test_device_repr_cpu() {
    let result: String = run_python_code(
        r#"
device = rstorch.PyDevice("cpu")
result = repr(device)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "device(type='cpu')");
}

#[test]
fn test_device_repr_cuda() {
    let result: String = run_python_code(
        r#"
device = rstorch.PyDevice("cuda:4")
result = repr(device)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "device(type='cuda', index=4)");
}

#[test]
fn test_device_equality_same() {
    let result: bool = run_python_code(
        r#"
device1 = rstorch.PyDevice("cuda:2")
device2 = rstorch.PyDevice("cuda:2")
result = device1 == device2
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_device_equality_different_index() {
    let result: bool = run_python_code(
        r#"
device1 = rstorch.PyDevice("cuda:2")
device2 = rstorch.PyDevice("cuda:3")
result = device1 == device2
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(!result);
}

#[test]
fn test_device_equality_different_type() {
    let result: bool = run_python_code(
        r#"
device1 = rstorch.PyDevice("cpu")
device2 = rstorch.PyDevice("cuda:0")
result = device1 == device2
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(!result);
}

#[test]
fn test_device_hash_consistency() {
    init_py();
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import rstorch_python as rstorch

device1 = rstorch.PyDevice("cuda:2")
device2 = rstorch.PyDevice("cuda:2")
result = hash(device1) == hash(device2)
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
fn test_device_in_set() {
    let result: bool = run_python_code(
        r#"
device1 = rstorch.PyDevice("cuda:2")
device2 = rstorch.PyDevice("cuda:2")
device_set = {device1}
result = device2 in device_set
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_device_invalid_string() {
    init_py();
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import rstorch_python as rstorch

try:
    device = rstorch.PyDevice("invalid")
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
        assert!(result, "Should raise ValueError for invalid device string");
    });
}

#[test]
fn test_device_invalid_cuda_id() {
    init_py();
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import rstorch_python as rstorch

try:
    device = rstorch.PyDevice("cuda:abc")
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
        assert!(result, "Should raise ValueError for invalid CUDA device ID");
    });
}

#[test]
fn test_device_negative_integer() {
    init_py();
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import rstorch_python as rstorch

try:
    device = rstorch.PyDevice(-1)
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
        assert!(result, "Should raise ValueError for negative device ID");
    });
}

#[test]
fn test_device_invalid_type() {
    init_py();
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import rstorch_python as rstorch

try:
    device = rstorch.PyDevice([1, 2, 3])
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
            "Should raise ValueError for invalid device type (list)"
        );
    });
}

#[test]
fn test_device_count_function() {
    let result: u32 = run_python_code(
        r#"
result = rstorch.device_count()
"#,
        |r| r.extract(),
    )
    .unwrap();

    // Should return at least 1 (for CPU)
    assert!(result >= 1);
}

#[test]
fn test_is_available_function() {
    let result: bool = run_python_code(
        r#"
result = rstorch.is_available()
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}

#[test]
fn test_cuda_is_available_function() {
    let result: bool = run_python_code(
        r#"
result = rstorch.cuda_is_available()
"#,
        |r| r.extract(),
    )
    .unwrap();

    // Result depends on system, just check it doesn't error
    println!("CUDA available: {}", result);
}

#[test]
fn test_mps_is_available_function() {
    let result: bool = run_python_code(
        r#"
result = rstorch.mps_is_available()
"#,
        |r| r.extract(),
    )
    .unwrap();

    // Result depends on system, just check it doesn't error
    println!("MPS available: {}", result);
}

#[test]
fn test_get_device_name_function_cpu() {
    let result: String = run_python_code(
        r#"
device = rstorch.PyDevice("cpu")
result = rstorch.get_device_name(device)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cpu");
}

#[test]
fn test_get_device_name_function_cuda() {
    let result: String = run_python_code(
        r#"
device = rstorch.PyDevice("cuda:3")
result = rstorch.get_device_name(device)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cuda:3");
}

#[test]
fn test_get_device_name_function_none() {
    let result: String = run_python_code(
        r#"
result = rstorch.get_device_name(None)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cpu");
}

#[test]
fn test_cpu_constant() {
    let result: String = run_python_code(
        r#"
result = str(rstorch.cpu)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "cpu");
}

#[test]
fn test_device_constants_equality() {
    let result: bool = run_python_code(
        r#"
device = rstorch.PyDevice("cpu")
result = device == rstorch.cpu
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result);
}
