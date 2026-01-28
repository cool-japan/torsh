// Build script for torsh-ffi (Python extension module)
//
// Following QuantRS2-py minimal approach:
// - Maturin handles all Python-specific configuration
// - No manual linker flag manipulation
// - Clean and simple
//
// COOLJAPAN Pure Rust Policy Note:
// This crate is dedicated to Python bindings (cdylib only)
// The main ToRSh crates remain 100% Pure Rust

fn main() {
    // Tell Cargo that if the given file changes, rerun this build script
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");

    // Maturin handles Python framework linking for us
    // No manual linker configuration needed
}
