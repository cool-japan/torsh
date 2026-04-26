// Build script for torsh-python.
//
// Emit Python link flags FIRST, before pyo3_build_config::use_pyo3_cfgs()
// which may call std::process::exit(0) early in abi3/extension-module mode.
// This ensures all compilation units (cdylib, integration tests) link libpython.
//
// We use cargo:rustc-link-arg (which becomes -C link-arg=) in addition to
// cargo:rustc-link-lib, because pyo3's extension-module machinery can suppress
// the latter for the test binary build units.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    // Emit Python library link unconditionally. For extension modules on Linux
    // this is harmless (symbols are resolved at load time by the Python process).
    // For test binaries it is required (they embed Python as a standalone binary).
    link_python();

    // pyo3_build_config may call std::process::exit(0) in abi3/extension-module
    // mode; link flags above are already flushed to stdout by the time it exits.
    pyo3_build_config::use_pyo3_cfgs();
}

fn link_python() {
    let python = python_executable();

    // Strategy 1: query sysconfig via the Python interpreter.
    if emit_via_sysconfig(&python) {
        return;
    }

    // Strategy 2: fall back to python-config --ldflags --embed.
    emit_via_python3_config(&python);
}

fn python_executable() -> String {
    std::env::var("PYO3_PYTHON")
        .or_else(|_| std::env::var("PYTHON_SYS_EXECUTABLE"))
        .unwrap_or_else(|_| "python3".to_owned())
}

fn emit_via_sysconfig(python: &str) -> bool {
    let output = std::process::Command::new(python)
        .args([
            "-c",
            "import sysconfig; cfg = sysconfig.get_config_vars(); \
             print(cfg.get('LDLIBRARY',''), cfg.get('LIBDIR',''), cfg.get('LIBPL',''), cfg.get('PYTHONFRAMEWORKPREFIX',''), sep='|')",
        ])
        .output();

    let Ok(out) = output else {
        return false;
    };
    if !out.status.success() {
        return false;
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let parts: Vec<&str> = stdout.trim().splitn(4, '|').collect();
    let ldlibrary = parts.first().copied().unwrap_or("").trim();
    let libdir = parts.get(1).copied().unwrap_or("").trim();
    let libpl = parts.get(2).copied().unwrap_or("").trim();
    let fwprefix = parts.get(3).copied().unwrap_or("").trim();

    if !libdir.is_empty() {
        println!("cargo:rustc-link-search=native={libdir}");
        // Also emit as a link-arg so the path is used even when cargo:rustc-link-lib
        // directives are suppressed by pyo3's extension-module machinery.
        println!("cargo:rustc-link-arg=-L{libdir}");
    }
    if !libpl.is_empty() && libpl != libdir {
        println!("cargo:rustc-link-search=native={libpl}");
        println!("cargo:rustc-link-arg=-L{libpl}");
    }

    if let Some(name) = extract_framework_name(ldlibrary) {
        if !fwprefix.is_empty() {
            println!("cargo:rustc-link-search=framework={fwprefix}");
            println!("cargo:rustc-link-arg=-F{fwprefix}");
        }
        println!("cargo:rustc-link-lib=framework={name}");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg={name}");
        return true;
    }

    if let Some(bare) = extract_lib_name(ldlibrary) {
        // Emit both forms: rustc-link-lib for the standard path, and
        // rustc-link-arg for the direct linker path (bypasses pyo3 suppression).
        println!("cargo:rustc-link-lib={bare}");
        println!("cargo:rustc-link-arg=-l{bare}");
        return true;
    }

    false
}

fn extract_framework_name(filename: &str) -> Option<String> {
    let pos = filename.find(".framework")?;
    let name = &filename[..pos];
    let bare = name.rsplit('/').next().unwrap_or(name);
    if bare.is_empty() {
        None
    } else {
        Some(bare.to_owned())
    }
}

fn extract_lib_name(filename: &str) -> Option<String> {
    if filename.is_empty() {
        return None;
    }
    let name = filename.strip_prefix("lib").unwrap_or(filename);
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

fn emit_via_python3_config(python: &str) {
    let config_bin = format!("{python}-config");
    let output = std::process::Command::new(&config_bin)
        .args(["--ldflags", "--embed"])
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            for token in stdout.split_whitespace() {
                if let Some(dir) = token.strip_prefix("-L") {
                    if !dir.is_empty() {
                        println!("cargo:rustc-link-search=native={dir}");
                        println!("cargo:rustc-link-arg=-L{dir}");
                    }
                } else if let Some(lib) = token.strip_prefix("-l") {
                    if !lib.is_empty() {
                        println!("cargo:rustc-link-lib={lib}");
                        println!("cargo:rustc-link-arg=-l{lib}");
                    }
                }
            }
        }
    }
}
