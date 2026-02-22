// Build script to ensure macOS dynamic lookup for Python symbols when
// building the PyO3 extension module inside a larger Cargo workspace.
// In theory, enabling the `extension-module` feature on PyO3 should add
// these flags automatically. However, in some workspace configurations
// (or with certain Cargo invocations like `cargo build -p torsh-python`),
// the linker flags may be missing, leading to undefined _Py* symbols.
//
// This script defensively re-adds the required flags on macOS so that
// unresolved Python symbols are resolved at runtime via the Python process
// loading the module.
//
// If PyO3 already injected these flags, re-specifying them is harmless.

fn main() {
    #[cfg(target_os = "macos")]
    {
        // Allow undefined Python C-API symbols which will be resolved
        // when the dynamic library is loaded into the Python interpreter.
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");

        // Optionally set a minimum macOS deployment target if not provided.
        if std::env::var_os("MACOSX_DEPLOYMENT_TARGET").is_none() {
            // Match the minimum specified in the linker invocation seen earlier (11.0.0)
            println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=11.0");
        }
    }
}
