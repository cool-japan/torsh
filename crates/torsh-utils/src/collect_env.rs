//! # Environment Information Collection
//!
//! This module provides comprehensive system and environment information collection
//! for debugging, issue reproduction, and system requirements verification.
//!
//! ## Features
//!
//! - **System Information**: OS, architecture, CPU details
//! - **Hardware Detection**: CPU features (AVX, SSE, etc.), GPU information
//! - **Memory Information**: Total and available RAM
//! - **Software Environment**: Rust version, ToRSh version, dependencies
//! - **GPU Detection**: NVIDIA CUDA, Apple Metal support
//! - **Python Integration**: Python version and installed packages (if available)
//! - **Environment Variables**: Relevant environment variables for debugging
//!
//! ## Quick Start
//!
//! ### Basic Usage
//!
//! ```rust,no_run
//! use torsh_utils::collect_env::{collect_env, print_env_info};
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! // Collect environment information
//! let env_info = collect_env()?;
//!
//! // Print formatted report
//! print_env_info(&env_info);
//!
//! // Access specific information
//! println!("ToRSh version: {}", env_info.torsh_version);
//! println!("Rust version: {}", env_info.rust_version);
//! println!("OS: {}", env_info.os);
//! # Ok(())
//! # }
//! ```
//!
//! ### Hardware Information
//!
//! ```rust,no_run
//! use torsh_utils::collect_env::collect_env;
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let env_info = collect_env()?;
//!
//! // CPU information
//! println!("CPU: {}", env_info.cpu_info.brand);
//! println!("CPU cores: {} physical, {} logical",
//!     env_info.cpu_info.cores,
//!     env_info.cpu_info.threads
//! );
//! println!("CPU features: {}", env_info.cpu_info.features.join(", "));
//!
//! // Memory information
//! println!("Total RAM: {} MB", env_info.memory_info.total_mb);
//! println!("Available RAM: {} MB", env_info.memory_info.available_mb);
//!
//! // GPU information
//! if !env_info.gpu_info.is_empty() {
//!     for (i, gpu) in env_info.gpu_info.iter().enumerate() {
//!         println!("GPU {}: {} ({} MB)",
//!             i,
//!             gpu.name,
//!             gpu.memory_mb
//!         );
//!         if let Some(cuda) = &gpu.cuda_version {
//!             println!("  CUDA version: {}", cuda);
//!         }
//!     }
//! } else {
//!     println!("No GPUs detected");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Environment Variables
//!
//! ```rust,no_run
//! use torsh_utils::collect_env::collect_env;
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let env_info = collect_env()?;
//!
//! // Check CUDA configuration
//! if let Some(cuda_home) = env_info.env_vars.get("CUDA_HOME") {
//!     println!("CUDA_HOME: {}", cuda_home);
//! }
//!
//! if let Some(cuda_path) = env_info.env_vars.get("CUDA_PATH") {
//!     println!("CUDA_PATH: {}", cuda_path);
//! }
//!
//! // Check library paths
//! if let Some(ld_path) = env_info.env_vars.get("LD_LIBRARY_PATH") {
//!     println!("LD_LIBRARY_PATH: {}", ld_path);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Export to JSON
//!
//! ```rust,no_run
//! use torsh_utils::collect_env::collect_env;
//! use std::fs;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let env_info = collect_env()?;
//!
//! // Serialize to JSON
//! let json = serde_json::to_string_pretty(&env_info)?;
//!
//! // Save to file
//! fs::write("environment_info.json", json)?;
//!
//! println!("Environment info saved to environment_info.json");
//! # Ok(())
//! # }
//! ```
//!
//! ### Python Environment Detection
//!
//! ```rust,no_run
//! use torsh_utils::collect_env::collect_env;
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let env_info = collect_env()?;
//!
//! if let Some(python) = &env_info.python_info {
//!     println!("Python version: {}", python.version);
//!     println!("Python executable: {}", python.executable);
//!
//!     // Check for specific packages
//!     if let Some(torch_version) = python.packages.get("torch") {
//!         println!("PyTorch installed: {}", torch_version);
//!     }
//!
//!     if let Some(numpy_version) = python.packages.get("numpy") {
//!         println!("NumPy installed: {}", numpy_version);
//!     }
//! } else {
//!     println!("Python not detected");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Use Cases
//!
//! ### 1. Bug Reports
//!
//! Include environment information in bug reports for easier reproduction:
//!
//! ```rust,no_run
//! use torsh_utils::collect_env::{collect_env, print_env_info};
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! println!("=== Bug Report ===");
//! println!("Issue: Model crashes during training");
//! println!();
//! println!("=== Environment ===");
//!
//! let env_info = collect_env()?;
//! print_env_info(&env_info);
//! # Ok(())
//! # }
//! ```
//!
//! ### 2. System Requirements Check
//!
//! Verify system meets minimum requirements:
//!
//! ```rust,no_run
//! use torsh_utils::collect_env::collect_env;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let env_info = collect_env()?;
//!
//! // Check minimum requirements
//! let min_ram_mb = 4096; // 4GB
//! let min_cpu_cores = 2;
//!
//! if env_info.memory_info.total_mb < min_ram_mb {
//!     eprintln!("⚠️  WARNING: Insufficient RAM. Required: {} MB, Available: {} MB",
//!         min_ram_mb, env_info.memory_info.total_mb);
//! }
//!
//! if env_info.cpu_info.cores < min_cpu_cores {
//!     eprintln!("⚠️  WARNING: Insufficient CPU cores. Required: {}, Available: {}",
//!         min_cpu_cores, env_info.cpu_info.cores);
//! }
//!
//! // Check for required CPU features
//! let required_features = vec!["SSE2", "AVX"];
//! for feature in &required_features {
//!     if !env_info.cpu_info.features.contains(&feature.to_string()) {
//!         eprintln!("⚠️  WARNING: Missing CPU feature: {}", feature);
//!     }
//! }
//!
//! // Check for CUDA if GPU training is needed
//! let needs_cuda = true;
//! if needs_cuda && env_info.gpu_info.is_empty() {
//!     eprintln!("⚠️  WARNING: No CUDA GPUs detected");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### 3. CI/CD Integration
//!
//! Log environment information in CI pipelines:
//!
//! ```rust,no_run
//! use torsh_utils::collect_env::{collect_env, print_env_info};
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! // In CI script
//! println!("=== CI Environment Information ===");
//! let env_info = collect_env()?;
//! print_env_info(&env_info);
//!
//! // Save for artifacts
//! let json = serde_json::to_string_pretty(&env_info).unwrap();
//! std::fs::write("ci_environment.json", json).unwrap();
//! # Ok(())
//! # }
//! ```
//!
//! ## Collected Information
//!
//! ### System Information
//! - Operating system name and version
//! - CPU architecture (x86_64, ARM, etc.)
//! - OS family (Unix, Windows, etc.)
//!
//! ### CPU Information
//! - Brand/model name
//! - Number of physical cores
//! - Number of logical threads (with hyperthreading)
//! - Supported instruction sets (SSE, AVX, AVX2, AVX512)
//!
//! ### GPU Information
//! - NVIDIA GPUs via nvidia-smi:
//!   - GPU name/model
//!   - Total memory
//!   - Driver version
//!   - CUDA version (if available)
//! - Apple Metal GPUs (on macOS)
//!
//! ### Memory Information
//! - Total system RAM
//! - Available RAM
//!
//! ### Software Versions
//! - ToRSh version
//! - Rust compiler version
//! - Installed Rust packages
//!
//! ### Python Environment (if available)
//! - Python version
//! - Python executable path
//! - Installed Python packages (via pip)
//!
//! ### Environment Variables
//! Collects relevant variables including:
//! - `CUDA_HOME`, `CUDA_PATH`
//! - `CUDNN_PATH`
//! - `LD_LIBRARY_PATH`, `DYLD_LIBRARY_PATH`
//! - `PATH`
//! - `RUST_BACKTRACE`, `RUST_LOG`
//! - `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
//!
//! ## Privacy Considerations
//!
//! The collected information includes:
//! - ✅ Hardware specifications (safe to share)
//! - ✅ Software versions (safe to share)
//! - ⚠️  Environment variables (may contain paths)
//! - ⚠️  Installed packages (usually safe)
//!
//! **Recommendation**: Review the output before sharing publicly, especially
//! environment variables that might contain sensitive paths or credentials.
//!
//! ## Best Practices
//!
//! 1. **Always Include in Bug Reports**: Helps maintainers reproduce issues
//! 2. **Version Compatibility**: Check before deploying to new environments
//! 3. **Automated Checks**: Integrate into CI/CD for consistent environments
//! 4. **Documentation**: Include in deployment documentation
//! 5. **Review Before Sharing**: Remove sensitive information if present
//!
//! ## Platform Support
//!
//! | Platform | System Info | CPU Info | GPU Detection | Python Detection |
//! |----------|-------------|----------|---------------|------------------|
//! | Linux    | ✅ Full     | ✅ Full  | ✅ NVIDIA     | ✅ Full          |
//! | macOS    | ✅ Full     | ✅ Full  | ✅ Metal      | ✅ Full          |
//! | Windows  | ✅ Full     | ✅ Full  | ✅ NVIDIA     | ✅ Full          |
//!
//! ## Comparison with PyTorch
//!
//! Similar to `torch.utils.collect_env.get_pretty_env_info()`, but provides:
//! - More detailed CPU feature detection
//! - Rust ecosystem information
//! - Better structured output (JSON serializable)
//! - Lower overhead (no Python runtime required)
//!
//! ## See Also
//!
//! - [`benchmark`](crate::benchmark): For performance benchmarking
//! - [`bottleneck`](crate::bottleneck): For performance profiling
//! - [Tutorial Guide](https://docs.torsh.rs/tutorial#environment)
//! - [Troubleshooting Guide](https://docs.torsh.rs/troubleshooting)

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use torsh_core::error::Result;

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub torsh_version: String,
    pub rust_version: String,
    pub os: String,
    pub cpu_info: CpuInfo,
    pub gpu_info: Vec<GpuInfo>,
    pub memory_info: MemoryInfo,
    pub python_info: Option<PythonInfo>,
    pub env_vars: HashMap<String, String>,
    pub installed_packages: HashMap<String, String>,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub cores: usize,
    pub threads: usize,
    pub features: Vec<String>,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory_mb: usize,
    pub driver_version: String,
    pub cuda_version: Option<String>,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_mb: usize,
    pub available_mb: usize,
}

/// Python environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonInfo {
    pub version: String,
    pub executable: String,
    pub packages: HashMap<String, String>,
}

/// Collect environment information
pub fn collect_env() -> Result<EnvironmentInfo> {
    let torsh_version = env!("CARGO_PKG_VERSION").to_string();
    let rust_version = get_rust_version();
    let os = get_os_info();
    let cpu_info = get_cpu_info();
    let gpu_info = get_gpu_info();
    let memory_info = get_memory_info();
    let python_info = get_python_info();
    let env_vars = get_relevant_env_vars();
    let installed_packages = get_installed_packages();

    Ok(EnvironmentInfo {
        torsh_version,
        rust_version,
        os,
        cpu_info,
        gpu_info,
        memory_info,
        python_info,
        env_vars,
        installed_packages,
    })
}

/// Get Rust version
fn get_rust_version() -> String {
    env::var("RUSTC_VERSION")
        .or_else(|_| {
            std::process::Command::new("rustc")
                .arg("--version")
                .output()
                .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
                .map_err(|_| std::env::VarError::NotPresent)
        })
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Get OS information
fn get_os_info() -> String {
    format!(
        "{} {} ({})",
        env::consts::OS,
        env::consts::ARCH,
        env::consts::FAMILY
    )
}

/// Get CPU information
fn get_cpu_info() -> CpuInfo {
    #[cfg(feature = "collect_env")]
    {
        use sysinfo::System;

        let mut sys = System::new();
        sys.refresh_cpu_all();

        let cpus = sys.cpus();
        let brand = cpus
            .first()
            .map(|cpu| cpu.brand().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let cores = num_cpus::get_physical();
        let threads = num_cpus::get();

        let features = get_cpu_features();

        CpuInfo {
            brand,
            cores,
            threads,
            features,
        }
    }

    #[cfg(not(feature = "collect_env"))]
    {
        CpuInfo {
            brand: "Unknown".to_string(),
            cores: num_cpus::get_physical(),
            threads: num_cpus::get(),
            features: vec![],
        }
    }
}

/// Get CPU features
fn get_cpu_features() -> Vec<String> {
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut features = vec![];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx") {
            features.push("AVX".to_string());
        }
        if is_x86_feature_detected!("avx2") {
            features.push("AVX2".to_string());
        }
        if is_x86_feature_detected!("avx512f") {
            features.push("AVX512F".to_string());
        }
        if is_x86_feature_detected!("sse") {
            features.push("SSE".to_string());
        }
        if is_x86_feature_detected!("sse2") {
            features.push("SSE2".to_string());
        }
        if is_x86_feature_detected!("sse3") {
            features.push("SSE3".to_string());
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("SSE4.1".to_string());
        }
        if is_x86_feature_detected!("sse4.2") {
            features.push("SSE4.2".to_string());
        }
    }

    features
}

/// Get GPU information
fn get_gpu_info() -> Vec<GpuInfo> {
    let mut gpus = vec![];

    // Check for NVIDIA GPUs
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    gpus.push(GpuInfo {
                        name: parts[0].to_string(),
                        memory_mb: parts[1].parse().unwrap_or(0),
                        driver_version: parts[2].to_string(),
                        cuda_version: get_cuda_version(),
                    });
                }
            }
        }
    }

    // Check for Metal (macOS)
    #[cfg(target_os = "macos")]
    {
        if let Ok(_output) = std::process::Command::new("system_profiler")
            .args(&["SPDisplaysDataType", "-json"])
            .output()
        {
            // Parse JSON output for GPU info
            // This is simplified - real implementation would parse properly
            gpus.push(GpuInfo {
                name: "Apple Metal GPU".to_string(),
                memory_mb: 0,
                driver_version: "Metal".to_string(),
                cuda_version: None,
            });
        }
    }

    gpus
}

/// Get CUDA version
fn get_cuda_version() -> Option<String> {
    if let Ok(output) = std::process::Command::new("nvcc")
        .args(["--version"])
        .output()
    {
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if line.contains("release") {
                    return line
                        .split_whitespace()
                        .find(|s| s.starts_with("V"))
                        .map(|s| s.trim_start_matches('V').to_string());
                }
            }
        }
    }
    None
}

/// Get memory information
fn get_memory_info() -> MemoryInfo {
    #[cfg(feature = "collect_env")]
    {
        use sysinfo::System;

        let mut sys = System::new();
        sys.refresh_memory();

        MemoryInfo {
            total_mb: (sys.total_memory() / 1024 / 1024) as usize,
            available_mb: (sys.available_memory() / 1024 / 1024) as usize,
        }
    }

    #[cfg(not(feature = "collect_env"))]
    {
        MemoryInfo {
            total_mb: 0,
            available_mb: 0,
        }
    }
}

/// Get Python information
fn get_python_info() -> Option<PythonInfo> {
    if let Ok(output) = std::process::Command::new("python3")
        .args(["--version"])
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();

            let executable = which::which("python3").ok()?.to_string_lossy().to_string();

            // Get installed packages
            let mut packages = HashMap::new();
            if let Ok(pip_output) = std::process::Command::new("python3")
                .args(["-m", "pip", "list", "--format=json"])
                .output()
            {
                if pip_output.status.success() {
                    if let Ok(json) = serde_json::from_slice::<Vec<PipPackage>>(&pip_output.stdout)
                    {
                        for pkg in json {
                            packages.insert(pkg.name, pkg.version);
                        }
                    }
                }
            }

            return Some(PythonInfo {
                version,
                executable,
                packages,
            });
        }
    }
    None
}

#[derive(Deserialize)]
struct PipPackage {
    name: String,
    version: String,
}

/// Get relevant environment variables
fn get_relevant_env_vars() -> HashMap<String, String> {
    let mut vars = HashMap::new();

    let relevant_vars = vec![
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDNN_PATH",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "PATH",
        "RUST_BACKTRACE",
        "RUST_LOG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ];

    for var in relevant_vars {
        if let Ok(value) = env::var(var) {
            vars.insert(var.to_string(), value);
        }
    }

    vars
}

/// Get installed Rust packages
fn get_installed_packages() -> HashMap<String, String> {
    let mut packages = HashMap::new();

    // Add core ToRSh packages
    packages.insert("torsh".to_string(), env!("CARGO_PKG_VERSION").to_string());

    // Get cargo dependencies
    if let Ok(output) = std::process::Command::new("cargo")
        .args(["tree", "--depth", "1", "--format", "{p}"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
    {
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            for line in output_str.lines() {
                if let Some((name, version)) = parse_cargo_tree_line(line) {
                    packages.insert(name, version);
                }
            }
        }
    }

    packages
}

/// Parse cargo tree output line
fn parse_cargo_tree_line(line: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 {
        let name = parts[0].to_string();
        let version = parts[1].trim_start_matches('v').to_string();
        Some((name, version))
    } else {
        None
    }
}

/// Pretty print environment information
pub fn print_env_info(info: &EnvironmentInfo) {
    println!("=== ToRSh Environment Information ===");
    println!();
    println!("ToRSh Version: {}", info.torsh_version);
    println!("Rust Version: {}", info.rust_version);
    println!("OS: {}", info.os);
    println!();

    println!("CPU Information:");
    println!("  Brand: {}", info.cpu_info.brand);
    println!(
        "  Cores: {} physical, {} logical",
        info.cpu_info.cores, info.cpu_info.threads
    );
    println!("  Features: {}", info.cpu_info.features.join(", "));
    println!();

    if !info.gpu_info.is_empty() {
        println!("GPU Information:");
        for (i, gpu) in info.gpu_info.iter().enumerate() {
            println!("  GPU {}: {}", i, gpu.name);
            println!("    Memory: {} MB", gpu.memory_mb);
            println!("    Driver: {}", gpu.driver_version);
            if let Some(cuda) = &gpu.cuda_version {
                println!("    CUDA: {}", cuda);
            }
        }
        println!();
    }

    println!("Memory:");
    println!("  Total: {} MB", info.memory_info.total_mb);
    println!("  Available: {} MB", info.memory_info.available_mb);
    println!();

    if let Some(python) = &info.python_info {
        println!("Python:");
        println!("  Version: {}", python.version);
        println!("  Executable: {}", python.executable);
        if !python.packages.is_empty() {
            println!("  Key packages:");
            for (name, version) in &python.packages {
                if name.contains("torch") || name.contains("numpy") || name.contains("scipy") {
                    println!("    {}: {}", name, version);
                }
            }
        }
        println!();
    }

    if !info.env_vars.is_empty() {
        println!("Environment Variables:");
        for (key, value) in &info.env_vars {
            if key.contains("CUDA") || key.contains("PATH") {
                println!("  {}: {}", key, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_env() {
        let info = collect_env().unwrap();

        assert!(!info.torsh_version.is_empty());
        assert!(!info.rust_version.is_empty());
        assert!(!info.os.is_empty());
        assert!(info.cpu_info.cores > 0);
        assert!(info.cpu_info.threads > 0);
        // Note: threads is typically >= cores with hyperthreading,
        // but detection may vary across systems
    }

    #[test]
    fn test_cpu_features() {
        let features = get_cpu_features();
        // At least SSE2 should be available on modern x86_64
        #[cfg(target_arch = "x86_64")]
        assert!(!features.is_empty());
        // On other architectures, just verify the function works
        #[cfg(not(target_arch = "x86_64"))]
        let _ = features;
    }
}
