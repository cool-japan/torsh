fn main() {
    // Add OpenMP linking for macOS when using OpenBLAS
    #[cfg(target_os = "macos")]
    {
        // For macOS, try to link with libomp if available
        if pkg_config::probe_library("openmp").is_ok() {
            println!("cargo:rustc-link-lib=omp");
        } else {
            // Fallback to homebrew installed OpenMP
            println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
            println!("cargo:rustc-link-lib=omp");
        }
    }

    // For other Unix systems
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        println!("cargo:rustc-link-lib=gomp");
    }
}
