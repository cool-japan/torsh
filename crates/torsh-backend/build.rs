//! Build script for torsh-backend
//!
//! This script detects CUDA availability and sets appropriate cfg flags
//! to enable conditional compilation of CUDA features.

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDATOOLKIT_HOME");

    // Only check for CUDA if the cuda feature is enabled
    if cfg!(feature = "cuda") {
        detect_cuda_availability();
    }
}

fn detect_cuda_availability() {
    let cuda_available = check_cuda_installation();

    if cuda_available {
        println!("cargo:rustc-cfg=cuda_available");
        println!("cargo:rustc-cfg=cuda_runtime_available");
        println!("cargo:warning=CUDA detected - enabling CUDA backend");
    } else {
        println!("cargo:warning=CUDA not detected - CUDA backend will use fallback implementation");
    }
}

fn check_cuda_installation() -> bool {
    // Method 1: Check for CUDA environment variables
    if let Ok(cuda_path) = env::var("CUDA_ROOT") {
        if Path::new(&cuda_path).exists() {
            return true;
        }
    }

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        if Path::new(&cuda_path).exists() {
            return true;
        }
    }

    if let Ok(cuda_path) = env::var("CUDATOOLKIT_HOME") {
        if Path::new(&cuda_path).exists() {
            return true;
        }
    }

    // Method 2: Check for nvcc in PATH
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return true;
    }

    // Method 3: Check common CUDA installation paths
    let common_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
        "C:\\CUDA",
    ];

    for path in &common_paths {
        if Path::new(path).exists() {
            return true;
        }
    }

    // Method 4: Check for CUDA libraries in system paths
    #[cfg(target_os = "linux")]
    {
        if Path::new("/usr/lib/x86_64-linux-gnu/libcuda.so").exists()
            || Path::new("/usr/lib64/libcuda.so").exists()
        {
            return true;
        }
    }

    #[cfg(target_os = "windows")]
    {
        if Path::new("C:\\Windows\\System32\\nvcuda.dll").exists() {
            return true;
        }
    }

    // Method 5: Try to run nvidia-smi
    if Command::new("nvidia-smi").output().is_ok() {
        return true;
    }

    false
}
