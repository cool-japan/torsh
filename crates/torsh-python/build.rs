// Build script for torsh-python (Python extension module).
//
// Uses pyo3-build-config to emit the correct Python library link arguments
// for both .so extension modules (loaded by Python) and test binaries.
// This replaces the previous macOS-only dynamic_lookup approach, which
// broke on Linux.
fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    pyo3_build_config::use_pyo3_cfgs();
    let config = pyo3_build_config::get();
    if let Some(lib_dir) = &config.lib_dir {
        println!("cargo:rustc-link-search=native={lib_dir}");
    }
    if let Some(lib_name) = &config.lib_name {
        println!("cargo:rustc-link-lib={lib_name}");
    }
}
