//! Utility functions for ToRSh CLI

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use anyhow::{Context, Result};
use byte_unit::Byte;
use chrono::Local;
use colored::*;
// Console utilities available when needed
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use sysinfo::System;
use tracing::{debug, info};

/// Display the ToRSh banner
pub fn display_banner() {
    let banner = r#"
  ______         _____   _____ _
 |__   _|       |  __ \ / ____| |
    | | ___  _ _| |__) | (___ | |__
    | |/ _ \| '__|  _  / \___ \| '_ \
   _| | (_) | |  | | \ \ ____) | | | |
  |_| \___/|_|  |_|  \_\_____/|_| |_|

"#;

    println!("{}", banner.bright_cyan().bold());
    println!(
        "{}",
        "ToRSh CLI - Advanced Deep Learning Framework Tools"
            .bright_white()
            .bold()
    );
    println!(
        "{}",
        format!("Version: {} | Build: {}", env!("CARGO_PKG_VERSION"), "dev").bright_black()
    );
    println!();
}

/// Output formatting utilities
pub mod output {
    use super::*;
    use serde::Serialize;

    /// Format output based on the specified format
    pub fn format_output<T: Serialize>(data: &T, format: &str) -> Result<String> {
        match format {
            "json" => {
                serde_json::to_string_pretty(data).with_context(|| "Failed to serialize to JSON")
            }
            "yaml" => serde_yaml::to_string(data).with_context(|| "Failed to serialize to YAML"),
            "table" => {
                // For table format, we'll need to implement custom formatting
                // This is a simplified version
                format_as_table(data)
            }
            _ => {
                anyhow::bail!("Unsupported output format: {}", format)
            }
        }
    }

    /// Format data as a table (simplified implementation)
    fn format_as_table<T: Serialize>(data: &T) -> Result<String> {
        let json_value = serde_json::to_value(data)?;
        format_json_as_table(&json_value, 0)
    }

    fn format_json_as_table(value: &Value, indent: usize) -> Result<String> {
        let mut output = String::new();
        let indent_str = "  ".repeat(indent);

        match value {
            Value::Object(map) => {
                for (key, val) in map {
                    match val {
                        Value::Object(_) | Value::Array(_) => {
                            writeln!(output, "{}{}:", indent_str, key.bright_cyan())?;
                            output.push_str(&format_json_as_table(val, indent + 1)?);
                        }
                        _ => {
                            writeln!(
                                output,
                                "{}{}: {}",
                                indent_str,
                                key.bright_cyan(),
                                format_json_value(val)
                            )?;
                        }
                    }
                }
            }
            Value::Array(arr) => {
                for (i, val) in arr.iter().enumerate() {
                    writeln!(output, "{}[{}]:", indent_str, i.to_string().bright_yellow())?;
                    output.push_str(&format_json_as_table(val, indent + 1)?);
                }
            }
            _ => {
                writeln!(output, "{}{}", indent_str, format_json_value(value))?;
            }
        }

        Ok(output)
    }

    fn format_json_value(value: &Value) -> String {
        match value {
            Value::String(s) => s.green().to_string(),
            Value::Number(n) => n.to_string().yellow().to_string(),
            Value::Bool(b) => {
                if *b {
                    "true".bright_green().to_string()
                } else {
                    "false".bright_red().to_string()
                }
            }
            Value::Null => "null".bright_black().to_string(),
            _ => value.to_string(),
        }
    }

    /// Print a formatted table
    pub fn print_table<T: Serialize>(title: &str, data: &T, format: &str) -> Result<()> {
        println!("{}", title.bright_cyan().bold());
        println!("{}", "=".repeat(title.len()).bright_cyan());
        println!();

        let formatted = format_output(data, format)?;
        println!("{}", formatted);

        Ok(())
    }

    /// Print a success message
    pub fn print_success(message: &str) {
        println!("{} {}", "✓".bright_green().bold(), message);
    }

    /// Print an error message
    pub fn print_error(message: &str) {
        eprintln!("{} {}", "✗".bright_red().bold(), message);
    }

    /// Print a warning message
    pub fn print_warning(message: &str) {
        println!("{} {}", "⚠".bright_yellow().bold(), message);
    }

    /// Print an info message
    pub fn print_info(message: &str) {
        println!("{} {}", "ℹ".bright_blue().bold(), message);
    }
}

/// Progress bar utilities
pub mod progress {
    use super::*;

    /// Create a progress bar with custom style
    pub fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
        let pb = ProgressBar::new(len);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {eta}")
                .expect("Invalid progress bar template")
                .progress_chars("█▉▊▋▌▍▎▏  "),
        );
        pb.set_message(message.to_string());
        pb
    }

    /// Create a spinner for indeterminate progress
    pub fn create_spinner(message: &str) -> ProgressBar {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .expect("Invalid spinner template")
                .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ "),
        );
        pb.set_message(message.to_string());
        pb
    }
}

/// File system utilities
pub mod fs {
    use super::*;

    /// Get file size as human-readable string
    pub fn format_file_size(size: u64) -> String {
        Byte::from_u128(size as u128)
            .unwrap_or_else(|| Byte::from_u128(0).unwrap())
            .get_appropriate_unit(byte_unit::UnitType::Binary)
            .to_string()
    }

    /// Get directory size recursively
    pub fn get_directory_size(
        path: &Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64>> + Send + '_>> {
        Box::pin(async move {
            let mut total_size = 0u64;
            let mut read_dir = tokio::fs::read_dir(path).await?;

            while let Some(entry) = read_dir.next_entry().await? {
                let metadata = entry.metadata().await?;
                if metadata.is_file() {
                    total_size += metadata.len();
                } else if metadata.is_dir() {
                    total_size += get_directory_size(&entry.path()).await?;
                }
            }

            Ok(total_size)
        })
    }

    /// Find files matching a pattern
    pub fn find_files(directory: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        let walker = walkdir::WalkDir::new(directory);

        for entry in walker {
            let entry = entry?;
            if entry.file_type().is_file() {
                let path = entry.path();
                if glob::Pattern::new(pattern)?.matches_path(path) {
                    files.push(path.to_path_buf());
                }
            }
        }

        Ok(files)
    }

    /// Create a backup of a file
    pub async fn backup_file(file_path: &Path) -> Result<PathBuf> {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let backup_path = file_path.with_extension(format!(
            "{}.backup_{}",
            file_path.extension().unwrap_or_default().to_string_lossy(),
            timestamp
        ));

        tokio::fs::copy(file_path, &backup_path).await?;
        info!("Created backup: {}", backup_path.display());

        Ok(backup_path)
    }

    /// Clean up temporary files
    pub async fn cleanup_temp_files(temp_dir: &Path) -> Result<()> {
        if temp_dir.exists() {
            tokio::fs::remove_dir_all(temp_dir).await?;
            debug!("Cleaned up temporary directory: {}", temp_dir.display());
        }
        Ok(())
    }
}

/// System information utilities
pub mod system {
    use super::*;

    /// System information structure
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct SystemInfo {
        pub os: String,
        pub kernel_version: String,
        pub total_memory: String,
        pub available_memory: String,
        pub cpu_count: usize,
        pub cpu_brand: String,
        pub cpu_frequency: u64,
        pub load_average: Vec<f64>,
        pub uptime: String,
    }

    /// Get comprehensive system information
    pub fn get_system_info() -> SystemInfo {
        let mut sys = System::new_all();
        sys.refresh_all();

        SystemInfo {
            os: format!(
                "{} {}",
                System::name().unwrap_or_default(),
                System::os_version().unwrap_or_default()
            ),
            kernel_version: System::kernel_version().unwrap_or_default(),
            total_memory: format_memory(sys.total_memory()),
            available_memory: format_memory(sys.available_memory()),
            cpu_count: sys.cpus().len(),
            cpu_brand: sys
                .cpus()
                .first()
                .map(|cpu| cpu.brand())
                .unwrap_or("Unknown")
                .to_string(),
            cpu_frequency: sys.cpus().first().map(|cpu| cpu.frequency()).unwrap_or(0),
            load_average: {
                let load = System::load_average();
                vec![load.one, load.five, load.fifteen]
            },
            uptime: format_duration(Duration::from_secs(System::uptime())),
        }
    }

    /// Format memory size
    fn format_memory(memory_kb: u64) -> String {
        let memory_bytes = memory_kb * 1024;
        Byte::from_u128(memory_bytes as u128)
            .unwrap_or_else(|| Byte::from_u128(0).unwrap())
            .get_appropriate_unit(byte_unit::UnitType::Binary)
            .to_string()
    }

    /// Check if GPU is available with real hardware detection
    pub fn check_gpu_availability() -> HashMap<String, bool> {
        let mut gpu_info = HashMap::new();

        // Check for CUDA with actual detection
        // #[cfg(feature = "cuda")]
        // {
        //     gpu_info.insert("CUDA".to_string(), detect_cuda_availability());
        // }
        // #[cfg(not(feature = "cuda"))]
        {
            // Still check for CUDA runtime even if not compiled with CUDA support
            gpu_info.insert("CUDA".to_string(), detect_cuda_runtime());
        }

        // Check for ROCm with actual detection
        // #[cfg(feature = "rocm")]
        // {
        //     gpu_info.insert("ROCm".to_string(), detect_rocm_availability());
        // }
        // #[cfg(not(feature = "rocm"))]
        {
            gpu_info.insert("ROCm".to_string(), detect_rocm_runtime());
        }

        // Check for Metal (macOS) with actual detection
        #[cfg(target_os = "macos")]
        {
            gpu_info.insert("Metal".to_string(), detect_metal_availability());
        }

        // Check for Vulkan support
        gpu_info.insert("Vulkan".to_string(), detect_vulkan_availability());

        // Check for OpenCL
        gpu_info.insert("OpenCL".to_string(), detect_opencl_availability());

        gpu_info
    }

    /// Detect CUDA availability at runtime
    // #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    fn detect_cuda_availability() -> bool {
        // This would use CUDA runtime API calls
        // For now, check if CUDA libraries are present
        detect_cuda_runtime()
    }

    fn detect_cuda_runtime() -> bool {
        // Check for CUDA runtime by looking for nvidia-smi command
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Detect ROCm availability
    // #[cfg(feature = "rocm")]
    #[allow(dead_code)]
    fn detect_rocm_availability() -> bool {
        detect_rocm_runtime()
    }

    fn detect_rocm_runtime() -> bool {
        // Check for ROCm by looking for rocm-smi command
        std::process::Command::new("rocm-smi")
            .arg("--showproductname")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Detect Metal availability (macOS only)
    #[cfg(target_os = "macos")]
    fn detect_metal_availability() -> bool {
        // Check if Metal is available by running system_profiler
        std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
            .map(|output| {
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains("Metal")
            })
            .unwrap_or(true) // Assume available on macOS if detection fails
    }

    fn detect_vulkan_availability() -> bool {
        // Check for Vulkan by looking for vulkaninfo command
        std::process::Command::new("vulkaninfo")
            .arg("--summary")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn detect_opencl_availability() -> bool {
        // Check for OpenCL by looking for clinfo command
        std::process::Command::new("clinfo")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Get comprehensive device information
    pub fn get_device_info() -> HashMap<String, serde_json::Value> {
        let mut device_info = HashMap::new();

        // Get system info for CPU details
        let sys_info = get_system_info();

        // CPU Information
        device_info.insert(
            "cpu".to_string(),
            serde_json::json!({
                "available": true,
                "device_type": "cpu",
                "description": "CPU device",
                "brand": sys_info.cpu_brand,
                "cores": sys_info.cpu_count,
                "frequency_mhz": sys_info.cpu_frequency,
                "capabilities": get_cpu_capabilities(),
            }),
        );

        // GPU Information with detailed detection
        let gpu_availability = check_gpu_availability();
        for (gpu_type, available) in gpu_availability {
            let detailed_info = if available {
                match gpu_type.as_str() {
                    "CUDA" => get_cuda_device_details(),
                    "ROCm" => get_rocm_device_details(),
                    "Metal" => get_metal_device_details(),
                    "Vulkan" => get_vulkan_device_details(),
                    "OpenCL" => get_opencl_device_details(),
                    _ => serde_json::json!({}),
                }
            } else {
                serde_json::json!({
                    "reason": "Runtime or drivers not detected"
                })
            };

            device_info.insert(
                gpu_type.to_lowercase(),
                serde_json::json!({
                    "available": available,
                    "device_type": "gpu",
                    "description": format!("{} GPU device", gpu_type),
                    "details": detailed_info
                }),
            );
        }

        device_info
    }

    /// Get CPU capabilities (SIMD instructions, etc.)
    fn get_cpu_capabilities() -> Vec<String> {
        let mut capabilities = Vec::new();

        // Check for common SIMD instruction sets
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse") {
                capabilities.push("SSE".to_string());
            }
            if is_x86_feature_detected!("sse2") {
                capabilities.push("SSE2".to_string());
            }
            if is_x86_feature_detected!("sse3") {
                capabilities.push("SSE3".to_string());
            }
            if is_x86_feature_detected!("sse4.1") {
                capabilities.push("SSE4.1".to_string());
            }
            if is_x86_feature_detected!("sse4.2") {
                capabilities.push("SSE4.2".to_string());
            }
            if is_x86_feature_detected!("avx") {
                capabilities.push("AVX".to_string());
            }
            if is_x86_feature_detected!("avx2") {
                capabilities.push("AVX2".to_string());
            }
            if is_x86_feature_detected!("fma") {
                capabilities.push("FMA".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                capabilities.push("NEON".to_string());
            }
        }

        capabilities
    }

    /// Get detailed CUDA device information
    fn get_cuda_device_details() -> serde_json::Value {
        // Use nvidia-smi to get device details
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name,memory.total,driver_version,cuda_version")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                let lines: Vec<&str> = info.trim().split('\n').collect();

                return serde_json::json!({
                    "devices": lines.iter().enumerate().map(|(i, line)| {
                        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                        if parts.len() >= 4 {
                            serde_json::json!({
                                "id": i,
                                "name": parts[0],
                                "memory_mb": parts[1],
                                "driver_version": parts[2],
                                "cuda_version": parts[3]
                            })
                        } else {
                            serde_json::json!({
                                "id": i,
                                "name": "Unknown GPU",
                                "error": "Failed to parse GPU info"
                            })
                        }
                    }).collect::<Vec<_>>()
                });
            }
        }

        serde_json::json!({ "error": "Failed to query CUDA devices" })
    }

    /// Get detailed ROCm device information
    fn get_rocm_device_details() -> serde_json::Value {
        if let Ok(output) = std::process::Command::new("rocm-smi")
            .arg("--showproductname")
            .arg("--showmeminfo=vram")
            .output()
        {
            if output.status.success() {
                return serde_json::json!({
                    "detected": true,
                    "raw_output": String::from_utf8_lossy(&output.stdout)
                });
            }
        }

        serde_json::json!({ "error": "Failed to query ROCm devices" })
    }

    /// Get detailed Metal device information (macOS only)
    #[cfg(target_os = "macos")]
    fn get_metal_device_details() -> serde_json::Value {
        if let Ok(output) = std::process::Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .arg("-detailLevel")
            .arg("full")
            .output()
        {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                return serde_json::json!({
                    "detected": true,
                    "metal_support": info.contains("Metal"),
                    "summary": "Metal GPU acceleration available"
                });
            }
        }

        serde_json::json!({ "error": "Failed to query Metal devices" })
    }

    #[cfg(not(target_os = "macos"))]
    fn get_metal_device_details() -> serde_json::Value {
        serde_json::json!({ "error": "Metal is only available on macOS" })
    }

    /// Get Vulkan device information
    fn get_vulkan_device_details() -> serde_json::Value {
        if let Ok(output) = std::process::Command::new("vulkaninfo")
            .arg("--summary")
            .output()
        {
            if output.status.success() {
                return serde_json::json!({
                    "detected": true,
                    "summary": "Vulkan runtime available"
                });
            }
        }

        serde_json::json!({ "error": "Failed to query Vulkan devices" })
    }

    /// Get OpenCL device information
    fn get_opencl_device_details() -> serde_json::Value {
        if let Ok(output) = std::process::Command::new("clinfo").arg("--list").output() {
            if output.status.success() {
                let info = String::from_utf8_lossy(&output.stdout);
                return serde_json::json!({
                    "detected": true,
                    "devices_summary": info.lines().take(10).collect::<Vec<_>>()
                });
            }
        }

        serde_json::json!({ "error": "Failed to query OpenCL devices" })
    }
}

/// Time and duration utilities
pub mod time {
    use super::*;

    /// Format duration as human-readable string
    pub fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs();
        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m {}s", secs / 60, secs % 60)
        } else if secs < 86400 {
            format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
        } else {
            format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
        }
    }

    /// Get current timestamp as string
    pub fn current_timestamp() -> String {
        Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
    }

    /// Parse human-readable duration
    pub fn parse_duration(s: &str) -> Result<Duration> {
        humantime::parse_duration(s).with_context(|| format!("Failed to parse duration: {}", s))
    }

    /// Measure execution time
    pub async fn measure_time<F, T>(f: F) -> (T, Duration)
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = f.await;
        let duration = start.elapsed();
        (result, duration)
    }
}

/// Network utilities
pub mod network {
    use super::*;

    /// Download a file with progress
    pub async fn download_file_with_progress(
        url: &str,
        output_path: &Path,
        show_progress: bool,
    ) -> Result<()> {
        let client = reqwest::Client::new();
        let response = client.get(url).send().await?;

        let total_size = response.content_length().unwrap_or(0);

        let pb = if show_progress && total_size > 0 {
            Some(progress::create_progress_bar(
                total_size,
                &format!(
                    "Downloading {}",
                    output_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                ),
            ))
        } else {
            None
        };

        let mut file = tokio::fs::File::create(output_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            if let Some(pb) = &pb {
                pb.set_position(downloaded);
            }
        }

        if let Some(pb) = pb {
            pb.finish_with_message("Download completed");
        }

        Ok(())
    }

    /// Check if URL is accessible
    pub async fn check_url_accessible(url: &str) -> bool {
        let client = reqwest::Client::new();
        client.head(url).send().await.is_ok()
    }
}

/// Validation utilities
pub mod validation {
    use super::*;

    /// Validate file exists and is readable
    pub fn validate_file_exists(path: &Path) -> Result<()> {
        if !path.exists() {
            anyhow::bail!("File does not exist: {}", path.display());
        }
        if !path.is_file() {
            anyhow::bail!("Path is not a file: {}", path.display());
        }
        Ok(())
    }

    /// Validate directory exists and is accessible
    pub fn validate_directory_exists(path: &Path) -> Result<()> {
        if !path.exists() {
            anyhow::bail!("Directory does not exist: {}", path.display());
        }
        if !path.is_dir() {
            anyhow::bail!("Path is not a directory: {}", path.display());
        }
        Ok(())
    }

    /// Validate model format
    pub fn validate_model_format(format: &str) -> Result<()> {
        let supported_formats = ["torsh", "pytorch", "onnx", "tensorflow", "tflite"];
        if !supported_formats.contains(&format) {
            anyhow::bail!(
                "Unsupported model format: {}. Supported formats: {}",
                format,
                supported_formats.join(", ")
            );
        }
        Ok(())
    }

    /// Validate device string
    pub fn validate_device(device: &str) -> Result<()> {
        if device == "cpu" {
            return Ok(());
        }

        if device.starts_with("cuda") {
            let parts: Vec<&str> = device.split(':').collect();
            if parts.len() == 2 {
                if parts[1].parse::<usize>().is_err() {
                    anyhow::bail!("Invalid CUDA device ID: {}", parts[1]);
                }
                return Ok(());
            } else if parts.len() == 1 && parts[0] == "cuda" {
                return Ok(());
            }
        }

        if device == "metal" {
            return Ok(());
        }

        anyhow::bail!(
            "Invalid device format: {}. Use 'cpu', 'cuda', 'cuda:N', or 'metal'",
            device
        );
    }
}

/// Interactive utilities
pub mod interactive {
    use super::*;
    use dialoguer::{Confirm, Input, Select};

    /// Ask user for confirmation
    pub fn confirm(message: &str, default: bool) -> Result<bool> {
        Confirm::new()
            .with_prompt(message)
            .default(default)
            .interact()
            .with_context(|| "Failed to get user confirmation")
    }

    /// Get text input from user
    pub fn input<T>(message: &str, default: Option<T>) -> Result<T>
    where
        T: Clone + std::fmt::Display + std::str::FromStr,
        T::Err: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static,
    {
        let mut input = Input::new().with_prompt(message);

        if let Some(default_value) = default {
            input = input.default(default_value);
        }

        input
            .interact_text()
            .with_context(|| "Failed to get user input")
    }

    /// Select from a list of options
    pub fn select(message: &str, options: &[String]) -> Result<usize> {
        Select::new()
            .with_prompt(message)
            .items(options)
            .interact()
            .with_context(|| "Failed to get user selection")
    }
}

/// Export format_duration function at module level
pub use time::format_duration;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_format_duration() {
        assert_eq!(time::format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(time::format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(time::format_duration(Duration::from_secs(3661)), "1h 1m");
    }

    #[test]
    fn test_validation() {
        assert!(validation::validate_model_format("torsh").is_ok());
        assert!(validation::validate_model_format("invalid").is_err());

        assert!(validation::validate_device("cpu").is_ok());
        assert!(validation::validate_device("cuda:0").is_ok());
        assert!(validation::validate_device("invalid").is_err());
    }

    #[tokio::test]
    async fn test_file_operations() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.txt");

        tokio::fs::write(&test_file, "test content").await.unwrap();

        let size = fs::get_directory_size(temp_dir.path()).await.unwrap();
        assert!(size > 0);

        let backup = fs::backup_file(&test_file).await.unwrap();
        assert!(backup.exists());
    }

    #[test]
    fn test_output_formatting() {
        use serde_json::json;

        let data = json!({
            "name": "test",
            "value": 42,
            "active": true
        });

        let json_output = output::format_output(&data, "json").unwrap();
        assert!(json_output.contains("test"));

        let yaml_output = output::format_output(&data, "yaml").unwrap();
        assert!(yaml_output.contains("name: test"));
    }
}
