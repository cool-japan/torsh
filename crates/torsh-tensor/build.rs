fn main() {
    // Link OpenMP on macOS
    #[cfg(target_os = "macos")]
    {
        // Try to find OpenMP via Homebrew
        if let Ok(output) = std::process::Command::new("brew")
            .args(["--prefix", "libomp"])
            .output()
        {
            if output.status.success() {
                let prefix = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("cargo:rustc-link-search=native={prefix}/lib");
                println!("cargo:rustc-link-lib=omp");
            }
        } else {
            // Fallback: try common paths
            for path in &[
                "/opt/homebrew/lib", // Apple Silicon Homebrew
                "/usr/local/lib",    // Intel Homebrew
            ] {
                if std::path::Path::new(&format!("{path}/libomp.dylib")).exists() {
                    println!("cargo:rustc-link-search=native={path}");
                    println!("cargo:rustc-link-lib=omp");
                    break;
                }
            }
        }
    }

    // Link OpenMP on Linux
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=gomp");
    }
}
