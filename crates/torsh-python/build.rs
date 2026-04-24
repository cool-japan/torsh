// Build script for torsh-python (Python extension module).
//
// Uses pyo3-build-config to emit the correct Python library link arguments
// for both .so extension modules (loaded by Python) and test binaries.
//
// Fallback: when pyo3_build_config returns no lib_name (common on Linux with
// abi3 features), we drive `python3-config --ldflags --embed` or
// `{executable} -c "import sysconfig; ..."` to obtain the link flags manually.
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
        // Normal path: pyo3_build_config resolved the library name directly.
        println!("cargo:rustc-link-lib={lib_name}");
    } else {
        // Fallback path: abi3 or unusual environments where lib_name is absent.
        // Ask the Python interpreter itself for the correct link library.
        emit_python_link_fallback(&config);
    }
}

/// Determine Python link library via the Python interpreter's sysconfig when
/// pyo3_build_config did not provide one.
fn emit_python_link_fallback(config: &pyo3_build_config::InterpreterConfig) {
    // Prefer the executable pyo3 already resolved; fall back to env vars or
    // plain `python3`.
    let python = config
        .executable
        .as_ref()
        .map(|p| p.clone())
        .or_else(|| std::env::var("PYO3_PYTHON").ok())
        .or_else(|| std::env::var("PYTHON_SYS_EXECUTABLE").ok())
        .unwrap_or_else(|| "python3".to_owned());

    // Query the interpreter for LDLIBRARY (e.g. "libpython3.11.so.1.0") and
    // LIBDIR (the directory containing it).
    let output = std::process::Command::new(&python)
        .args([
            "-c",
            "import sysconfig; cfg = sysconfig.get_config_vars(); \
             print(cfg.get('LDLIBRARY',''), cfg.get('LIBDIR',''), cfg.get('LIBPL',''), sep='|')",
        ])
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<&str> = stdout.trim().splitn(3, '|').collect();
            let ldlibrary = parts.first().copied().unwrap_or("").trim();
            let libdir = parts.get(1).copied().unwrap_or("").trim();
            let libpl = parts.get(2).copied().unwrap_or("").trim();

            // Emit search paths (LIBDIR first, then LIBPL as fallback).
            if !libdir.is_empty() {
                println!("cargo:rustc-link-search=native={libdir}");
            }
            if !libpl.is_empty() && libpl != libdir {
                println!("cargo:rustc-link-search=native={libpl}");
            }

            // Strip shared-library prefix/suffix to get the bare link name.
            // e.g. "libpython3.11.so.1.0" → "python3.11"
            if !ldlibrary.is_empty() {
                if let Some(bare) = extract_lib_name(ldlibrary) {
                    println!("cargo:rustc-link-lib={bare}");
                    return;
                }
            }
        }
    }

    // Last resort: try python3-config --ldflags --embed and parse its output.
    emit_via_python3_config();
}

/// Strip `lib` prefix and `.so*` / `.dylib*` / `.a` suffix from a filename.
/// Returns the bare name suitable for `-l<name>`.
fn extract_lib_name(filename: &str) -> Option<String> {
    let name = filename.strip_prefix("lib").unwrap_or(filename);
    // Remove everything from the first `.so` or `.dylib` or `.a` onward.
    let bare = if let Some(pos) = name.find(".so") {
        &name[..pos]
    } else if let Some(pos) = name.find(".dylib") {
        &name[..pos]
    } else if let Some(stripped) = name.strip_suffix(".a") {
        stripped
    } else {
        name
    };
    if bare.is_empty() {
        None
    } else {
        Some(bare.to_owned())
    }
}

/// Drive `python3-config --ldflags --embed` and forward any -L/-l flags.
fn emit_via_python3_config() {
    let python3_config = std::env::var("PYO3_PYTHON")
        .map(|p| format!("{p}-config"))
        .unwrap_or_else(|_| "python3-config".to_owned());

    let output = std::process::Command::new(&python3_config)
        .args(["--ldflags", "--embed"])
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            for token in stdout.split_whitespace() {
                if let Some(dir) = token.strip_prefix("-L") {
                    if !dir.is_empty() {
                        println!("cargo:rustc-link-search=native={dir}");
                    }
                } else if let Some(lib) = token.strip_prefix("-l") {
                    if !lib.is_empty() {
                        println!("cargo:rustc-link-lib={lib}");
                    }
                }
            }
        }
    }
}
