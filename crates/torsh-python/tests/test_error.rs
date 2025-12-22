//! Comprehensive unit tests for the error module
//!
//! Tests cover:
//! - TorshError creation and string representation
//! - Error type registration
//! - Error messages

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

#[test]
fn test_torsh_error_creation() {
    let result: String = run_python_code(
        r#"
error = torsh.TorshError("Test error message")
result = str(error)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "Test error message");
}

#[test]
fn test_torsh_error_repr() {
    let result: String = run_python_code(
        r#"
error = torsh.TorshError("Test error message")
result = repr(error)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "TorshError('Test error message')");
}

#[test]
fn test_torsh_error_type_registered() {
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import torsh_python as torsh

result = hasattr(torsh, 'TorshError')
"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let code_cstr = CString::new(code).unwrap();
        let filename = CString::new("").unwrap();
        let module_name = CString::new("").unwrap();

        let module = PyModule::from_code(py, &code_cstr, &filename, &module_name).unwrap();

        let result: bool = module.getattr("result").unwrap().extract().unwrap();
        assert!(result, "TorshError should be registered in the module");
    });
}

#[test]
fn test_torsh_error_is_type() {
    Python::attach(|py| {
        let code = format!(
            r#"
import sys
sys.path.insert(0, '{}')
import torsh_python as torsh

error = torsh.TorshError("Test")
result = type(error).__name__
"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let code_cstr = CString::new(code).unwrap();
        let filename = CString::new("").unwrap();
        let module_name = CString::new("").unwrap();

        let module = PyModule::from_code(py, &code_cstr, &filename, &module_name).unwrap();

        let result: String = module.getattr("result").unwrap().extract().unwrap();
        assert_eq!(result, "TorshPyError");
    });
}

#[test]
fn test_torsh_error_empty_message() {
    let result: String = run_python_code(
        r#"
error = torsh.TorshError("")
result = str(error)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "");
}

#[test]
fn test_torsh_error_multiline_message() {
    let result: String = run_python_code(
        r#"
error = torsh.TorshError("Line 1\nLine 2\nLine 3")
result = str(error)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert_eq!(result, "Line 1\nLine 2\nLine 3");
}

#[test]
fn test_torsh_error_with_special_characters() {
    let result: String = run_python_code(
        r#"
error = torsh.TorshError("Error: 'quoted' and \"double quoted\" with symbols !@#$%^&*()")
result = str(error)
"#,
        |r| r.extract(),
    )
    .unwrap();

    assert!(result.contains("Error: 'quoted'"));
    assert!(result.contains("symbols !@#$%^&*()"));
}
