// Build script for torsh-benches
//
// Following QuantRS2 approach with minimal configuration
// COOLJAPAN Pure Rust Policy: Default is Pure Rust, Python features are optional
//
// Note: Unlike torsh-ffi (Python extension module), torsh-benches is a regular
// Rust crate that optionally calls Python. Therefore, we need pyo3_build_config
// when Python features are enabled.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");

    // Configure PyO3 when Python comparison features are enabled
    // This is required for non-extension-module crates that use PyO3
    #[cfg(any(
        feature = "pytorch",
        feature = "tensorflow",
        feature = "jax",
        feature = "numpy_baseline"
    ))]
    {
        pyo3_build_config::use_pyo3_cfgs();
    }
}
