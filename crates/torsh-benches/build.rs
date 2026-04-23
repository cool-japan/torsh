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

    // Configure PyO3 when Python comparison features are enabled.
    // In build scripts, features are checked via environment variables, not #[cfg(feature)].
    let has_pytorch = std::env::var("CARGO_FEATURE_PYTORCH").is_ok();
    let has_tensorflow = std::env::var("CARGO_FEATURE_TENSORFLOW").is_ok();
    let has_jax = std::env::var("CARGO_FEATURE_JAX").is_ok();
    let has_numpy_baseline = std::env::var("CARGO_FEATURE_NUMPY_BASELINE").is_ok();

    if has_pytorch || has_tensorflow || has_jax || has_numpy_baseline {
        pyo3_build_config::use_pyo3_cfgs();
    }
}
