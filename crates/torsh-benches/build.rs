// Build script for torsh-benches
//
// Following QuantRS2 approach with minimal configuration
// COOLJAPAN Pure Rust Policy: Default is Pure Rust, Python features are optional
//
// Note: Unlike torsh-ffi (Python extension module), torsh-benches is a regular
// Rust crate that optionally calls Python. Therefore, we need pyo3_build_config
// when Python features are enabled (including during `--all-features` builds).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    // Configure PyO3 when Python comparison features are enabled.
    // In build scripts, features are checked via environment variables, not #[cfg(feature)].
    let has_pytorch = std::env::var("CARGO_FEATURE_PYTORCH").is_ok();
    let has_tensorflow = std::env::var("CARGO_FEATURE_TENSORFLOW").is_ok();
    let has_jax = std::env::var("CARGO_FEATURE_JAX").is_ok();
    let has_numpy_baseline = std::env::var("CARGO_FEATURE_NUMPY_BASELINE").is_ok();

    if has_pytorch || has_tensorflow || has_jax || has_numpy_baseline {
        // Emit PyO3 cfgs so pyo3 macros know the Python version at compile time.
        pyo3_build_config::use_pyo3_cfgs();

        // Link the Python shared library so that test binaries (and any rlib
        // consumers) can resolve `_PyDict_Type` and related symbols at load time.
        // Without these link args, `cargo nextest --all-features` fails with a
        // dyld "symbol not found in flat namespace" error on macOS.
        let config = pyo3_build_config::get();
        if let Some(lib_dir) = &config.lib_dir {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
        if let Some(lib_name) = &config.lib_name {
            println!("cargo:rustc-link-lib={lib_name}");
        }
    }
}
