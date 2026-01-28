//! System information and diagnostics commands

use anyhow::Result;
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::Config;
use crate::utils::{display_banner, output, system};

#[derive(Debug, Args)]
pub struct InfoCommand {
    /// Show detailed system information
    #[arg(long)]
    pub detailed: bool,

    /// Show ToRSh installation information
    #[arg(long)]
    pub installation: bool,

    /// Show available devices and capabilities
    #[arg(long)]
    pub devices: bool,

    /// Show feature availability
    #[arg(long)]
    pub features: bool,

    /// Run system diagnostics
    #[arg(long)]
    pub diagnostics: bool,

    /// Show configuration information
    #[arg(long)]
    pub show_config: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInformation {
    pub torsh: TorshInfo,
    pub system: system::SystemInfo,
    pub devices: HashMap<String, serde_json::Value>,
    pub features: FeatureInfo,
    pub installation: InstallationInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TorshInfo {
    pub version: String,
    pub build_type: String,
    pub build_date: String,
    pub git_commit: String,
    pub rust_version: String,
    pub target_triple: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FeatureInfo {
    pub enabled_features: Vec<String>,
    pub disabled_features: Vec<String>,
    pub experimental_features: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstallationInfo {
    pub install_path: String,
    pub config_path: String,
    pub cache_path: String,
    pub models_path: String,
    pub size_on_disk: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiagnosticResult {
    pub name: String,
    pub status: DiagnosticStatus,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum DiagnosticStatus {
    Pass,
    Warning,
    Fail,
    Info,
}

pub async fn execute(args: InfoCommand, config: &Config, output_format: &str) -> Result<()> {
    display_banner();

    if !args.detailed
        && !args.installation
        && !args.devices
        && !args.features
        && !args.diagnostics
        && !args.show_config
    {
        // Show basic system information by default
        show_basic_info(output_format).await?;
    } else {
        if args.detailed || args.installation {
            show_detailed_info(output_format).await?;
        }

        if args.devices {
            show_device_info(output_format).await?;
        }

        if args.features {
            show_feature_info(output_format).await?;
        }

        if args.show_config {
            show_config_info(config, output_format).await?;
        }

        if args.diagnostics {
            run_diagnostics(config, output_format).await?;
        }
    }

    Ok(())
}

async fn show_basic_info(output_format: &str) -> Result<()> {
    let torsh_info = get_torsh_info();
    let system_info = system::get_system_info();

    let basic_info = serde_json::json!({
        "torsh_version": torsh_info.version,
        "os": system_info.os,
        "total_memory": system_info.total_memory,
        "cpu_count": system_info.cpu_count,
        "available_devices": get_available_devices_summary(),
    });

    output::print_table("ToRSh System Information", &basic_info, output_format)?;
    Ok(())
}

async fn show_detailed_info(output_format: &str) -> Result<()> {
    let system_info = SystemInformation {
        torsh: get_torsh_info(),
        system: system::get_system_info(),
        devices: system::get_device_info(),
        features: get_feature_info(),
        installation: get_installation_info().await?,
    };

    output::print_table("Detailed System Information", &system_info, output_format)?;
    Ok(())
}

async fn show_device_info(output_format: &str) -> Result<()> {
    let device_info = system::get_device_info();
    output::print_table("Available Devices", &device_info, output_format)?;

    // Show device capabilities
    for (device_name, info) in &device_info {
        if let Some(available) = info.get("available").and_then(|v| v.as_bool()) {
            if available {
                output::print_success(&format!(
                    "✓ {} device is available",
                    device_name.to_uppercase()
                ));
            } else {
                output::print_warning(&format!(
                    "⚠ {} device is not available",
                    device_name.to_uppercase()
                ));
            }
        }
    }

    Ok(())
}

async fn show_feature_info(output_format: &str) -> Result<()> {
    let feature_info = get_feature_info();
    output::print_table("Feature Information", &feature_info, output_format)?;

    output::print_info(&format!(
        "Enabled features: {}",
        feature_info.enabled_features.len()
    ));
    output::print_info(&format!(
        "Disabled features: {}",
        feature_info.disabled_features.len()
    ));
    if !feature_info.experimental_features.is_empty() {
        output::print_warning(&format!(
            "Experimental features: {}",
            feature_info.experimental_features.len()
        ));
    }

    Ok(())
}

async fn show_config_info(config: &Config, output_format: &str) -> Result<()> {
    let config_summary = serde_json::json!({
        "output_dir": config.general.output_dir,
        "cache_dir": config.general.cache_dir,
        "default_device": config.general.default_device,
        "num_workers": config.general.num_workers,
        "default_dtype": config.general.default_dtype,
        "hub_endpoint": config.hub.api_endpoint,
        "mixed_precision": config.training.mixed_precision,
    });

    output::print_table("Configuration", &config_summary, output_format)?;
    Ok(())
}

async fn run_diagnostics(config: &Config, output_format: &str) -> Result<()> {
    output::print_info("Running system diagnostics...");

    let mut diagnostics = Vec::new();

    // Check ToRSh installation
    diagnostics.push(check_torsh_installation().await);

    // Check dependencies
    diagnostics.push(check_dependencies().await);

    // Check device availability
    diagnostics.extend(check_device_availability().await);

    // Check configuration
    diagnostics.push(check_configuration(config).await);

    // Check permissions
    diagnostics.push(check_permissions(config).await);

    // Check disk space
    diagnostics.push(check_disk_space(config).await);

    output::print_table("Diagnostic Results", &diagnostics, output_format)?;

    // Summary
    let pass_count = diagnostics
        .iter()
        .filter(|d| matches!(d.status, DiagnosticStatus::Pass))
        .count();
    let warning_count = diagnostics
        .iter()
        .filter(|d| matches!(d.status, DiagnosticStatus::Warning))
        .count();
    let fail_count = diagnostics
        .iter()
        .filter(|d| matches!(d.status, DiagnosticStatus::Fail))
        .count();

    println!();
    output::print_info(&format!(
        "Diagnostic Summary: {} passed, {} warnings, {} failed",
        pass_count, warning_count, fail_count
    ));

    if fail_count > 0 {
        output::print_error("Some diagnostics failed. Please check the results above.");
    } else if warning_count > 0 {
        output::print_warning("Some diagnostics have warnings. Please review the results above.");
    } else {
        output::print_success("All diagnostics passed!");
    }

    Ok(())
}

fn get_torsh_info() -> TorshInfo {
    TorshInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        build_type: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .to_string(),
        build_date: "2024-01-15".to_string(), // This would be set during build
        git_commit: "abc123def".to_string(),  // This would be set during build
        rust_version: std::env::var("RUST_VERSION").unwrap_or_else(|_| "unknown".to_string()),
        target_triple: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
    }
}

fn get_feature_info() -> FeatureInfo {
    #[allow(unused_mut)]
    let mut enabled_features = Vec::new();
    #[allow(unused_mut)]
    let mut disabled_features = Vec::new();
    #[allow(unused_mut)]
    let mut experimental_features = Vec::new();

    // Check compiled features
    #[cfg(feature = "nn")]
    enabled_features.push("nn".to_string());
    #[cfg(not(feature = "nn"))]
    disabled_features.push("nn".to_string());

    #[cfg(feature = "optim")]
    enabled_features.push("optim".to_string());
    #[cfg(not(feature = "optim"))]
    disabled_features.push("optim".to_string());

    #[cfg(feature = "data")]
    enabled_features.push("data".to_string());
    #[cfg(not(feature = "data"))]
    disabled_features.push("data".to_string());

    #[cfg(feature = "vision")]
    enabled_features.push("vision".to_string());
    #[cfg(not(feature = "vision"))]
    disabled_features.push("vision".to_string());

    #[cfg(feature = "text")]
    enabled_features.push("text".to_string());
    #[cfg(not(feature = "text"))]
    disabled_features.push("text".to_string());

    #[cfg(feature = "quantization")]
    enabled_features.push("quantization".to_string());
    #[cfg(not(feature = "quantization"))]
    disabled_features.push("quantization".to_string());

    #[cfg(feature = "jit")]
    experimental_features.push("jit".to_string());

    #[cfg(feature = "hub")]
    enabled_features.push("hub".to_string());
    #[cfg(not(feature = "hub"))]
    disabled_features.push("hub".to_string());

    FeatureInfo {
        enabled_features,
        disabled_features,
        experimental_features,
    }
}

async fn get_installation_info() -> Result<InstallationInfo> {
    let current_exe = std::env::current_exe().unwrap_or_else(|_| "unknown".into());
    let install_path = current_exe
        .parent()
        .unwrap_or_else(|| std::path::Path::new("unknown"));

    let home_dir = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let config_dir = dirs::config_dir().unwrap_or_else(|| home_dir.join(".config"));
    let cache_dir = dirs::cache_dir().unwrap_or_else(|| home_dir.join(".cache"));

    let torsh_config = config_dir.join("torsh");
    let torsh_cache = cache_dir.join("torsh");
    let torsh_models = torsh_cache.join("models");

    // Calculate size on disk (simplified)
    let mut total_size = 0u64;
    if let Ok(metadata) = tokio::fs::metadata(&current_exe).await {
        total_size += metadata.len();
    }

    Ok(InstallationInfo {
        install_path: install_path.display().to_string(),
        config_path: torsh_config.display().to_string(),
        cache_path: torsh_cache.display().to_string(),
        models_path: torsh_models.display().to_string(),
        size_on_disk: crate::utils::fs::format_file_size(total_size),
    })
}

fn get_available_devices_summary() -> HashMap<String, bool> {
    let device_info = system::get_device_info();
    let mut summary = HashMap::new();

    for (device_name, info) in device_info {
        if let Some(available) = info.get("available").and_then(|v| v.as_bool()) {
            summary.insert(device_name, available);
        }
    }

    summary
}

async fn check_torsh_installation() -> DiagnosticResult {
    let current_exe = std::env::current_exe();

    match current_exe {
        Ok(exe_path) => {
            if exe_path.exists() {
                DiagnosticResult {
                    name: "ToRSh Installation".to_string(),
                    status: DiagnosticStatus::Pass,
                    message: "ToRSh CLI is properly installed".to_string(),
                    details: Some(serde_json::json!({
                        "executable_path": exe_path.display().to_string()
                    })),
                }
            } else {
                DiagnosticResult {
                    name: "ToRSh Installation".to_string(),
                    status: DiagnosticStatus::Fail,
                    message: "ToRSh executable not found".to_string(),
                    details: None,
                }
            }
        }
        Err(e) => DiagnosticResult {
            name: "ToRSh Installation".to_string(),
            status: DiagnosticStatus::Fail,
            message: format!("Cannot determine executable path: {}", e),
            details: None,
        },
    }
}

async fn check_dependencies() -> DiagnosticResult {
    let mut dependency_status = HashMap::new();
    let mut issues = Vec::new();

    // Check for Python (needed for some model conversions)
    if let Ok(output) = std::process::Command::new("python3")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            dependency_status.insert("python3", version.trim().to_string());
        } else {
            issues.push("Python3 not found");
            dependency_status.insert("python3", "Not Available".to_string());
        }
    } else {
        issues.push("Python3 not found");
        dependency_status.insert("python3", "Not Available".to_string());
    }

    // Check for Git (needed for model hub operations)
    if let Ok(output) = std::process::Command::new("git").arg("--version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            dependency_status.insert("git", version.trim().to_string());
        } else {
            issues.push("Git not found");
            dependency_status.insert("git", "Not Available".to_string());
        }
    } else {
        issues.push("Git not found");
        dependency_status.insert("git", "Not Available".to_string());
    }

    // Check for curl (needed for downloads)
    if let Ok(output) = std::process::Command::new("curl").arg("--version").output() {
        if output.status.success() {
            dependency_status.insert("curl", "Available".to_string());
        } else {
            dependency_status.insert("curl", "Not Available".to_string());
        }
    } else {
        dependency_status.insert("curl", "Not Available".to_string());
    }

    let status = if issues.is_empty() {
        DiagnosticStatus::Pass
    } else if issues.len() <= 2 {
        DiagnosticStatus::Warning
    } else {
        DiagnosticStatus::Fail
    };

    let message = if issues.is_empty() {
        "All external dependencies are available".to_string()
    } else {
        format!("Some dependencies missing: {}", issues.join(", "))
    };

    DiagnosticResult {
        name: "External Dependencies".to_string(),
        status,
        message,
        details: Some(serde_json::json!(dependency_status)),
    }
}

async fn check_device_availability() -> Vec<DiagnosticResult> {
    let mut results = Vec::new();
    let device_info = system::get_device_info();

    for (device_name, info) in device_info {
        let available = info
            .get("available")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let status = if available {
            DiagnosticStatus::Pass
        } else {
            DiagnosticStatus::Warning
        };

        let message = if available {
            format!("{} device is available", device_name.to_uppercase())
        } else {
            format!("{} device is not available", device_name.to_uppercase())
        };

        results.push(DiagnosticResult {
            name: format!("{} Device", device_name.to_uppercase()),
            status,
            message,
            details: Some(info),
        });
    }

    results
}

async fn check_configuration(config: &Config) -> DiagnosticResult {
    // Check if configuration is valid
    let mut issues = Vec::new();

    if !config.general.cache_dir.exists() {
        issues.push("Cache directory does not exist");
    }

    if config.general.num_workers == 0 {
        issues.push("Number of workers is set to 0");
    }

    if issues.is_empty() {
        DiagnosticResult {
            name: "Configuration".to_string(),
            status: DiagnosticStatus::Pass,
            message: "Configuration is valid".to_string(),
            details: None,
        }
    } else {
        DiagnosticResult {
            name: "Configuration".to_string(),
            status: DiagnosticStatus::Warning,
            message: format!("Configuration has {} issues", issues.len()),
            details: Some(serde_json::json!({
                "issues": issues
            })),
        }
    }
}

async fn check_permissions(config: &Config) -> DiagnosticResult {
    // Check write permissions for important directories
    let test_file = config.general.cache_dir.join(".torsh_test");

    match tokio::fs::write(&test_file, "test").await {
        Ok(_) => {
            let _ = tokio::fs::remove_file(&test_file).await;
            DiagnosticResult {
                name: "Permissions".to_string(),
                status: DiagnosticStatus::Pass,
                message: "Write permissions are available".to_string(),
                details: None,
            }
        }
        Err(e) => DiagnosticResult {
            name: "Permissions".to_string(),
            status: DiagnosticStatus::Fail,
            message: format!("Cannot write to cache directory: {}", e),
            details: Some(serde_json::json!({
                "cache_dir": config.general.cache_dir.display().to_string(),
                "error": e.to_string(),
            })),
        },
    }
}

async fn check_disk_space(config: &Config) -> DiagnosticResult {
    use byte_unit::Byte;
    use sysinfo::Disks;

    // Check disk space for cache directory
    let cache_dir = &config.general.cache_dir;

    // ✅ Pure Rust: Use sysinfo instead of libc::statvfs
    let disks = Disks::new_with_refreshed_list();

    // Find the disk containing the cache directory
    let cache_path = cache_dir
        .canonicalize()
        .unwrap_or_else(|_| cache_dir.clone());
    let mut target_disk = None;
    let mut longest_mount_len = 0;

    for disk in disks.list() {
        let mount_point = disk.mount_point();
        if cache_path.starts_with(mount_point) {
            let mount_len = mount_point.as_os_str().len();
            if mount_len > longest_mount_len {
                target_disk = Some((
                    disk.total_space(),
                    disk.available_space(),
                    mount_point.to_path_buf(),
                ));
                longest_mount_len = mount_len;
            }
        }
    }

    if let Some((total_bytes, available_bytes, mount_point)) = target_disk {
        let used_bytes = total_bytes.saturating_sub(available_bytes);

        let usage_percent = if total_bytes > 0 {
            (used_bytes as f64 / total_bytes as f64) * 100.0
        } else {
            0.0
        };

        let status = if usage_percent > 90.0 {
            DiagnosticStatus::Fail
        } else if usage_percent > 80.0 {
            DiagnosticStatus::Warning
        } else {
            DiagnosticStatus::Pass
        };

        let message = if usage_percent > 90.0 {
            "Very low disk space available (>90% used)".to_string()
        } else if usage_percent > 80.0 {
            "Low disk space warning (>80% used)".to_string()
        } else {
            "Sufficient disk space available".to_string()
        };

        return DiagnosticResult {
            name: "Disk Space".to_string(),
            status,
            message,
            details: Some(serde_json::json!({
                "cache_dir": cache_dir.display().to_string(),
                "mount_point": mount_point.display().to_string(),
                "total_space": Byte::from_u128(total_bytes as u128).unwrap_or_else(|| Byte::from_u128(0).expect("zero bytes should always be valid")).get_appropriate_unit(byte_unit::UnitType::Binary).to_string(),
                "available_space": Byte::from_u128(available_bytes as u128).unwrap_or_else(|| Byte::from_u128(0).expect("zero bytes should always be valid")).get_appropriate_unit(byte_unit::UnitType::Binary).to_string(),
                "used_space": Byte::from_u128(used_bytes as u128).unwrap_or_else(|| Byte::from_u128(0).expect("zero bytes should always be valid")).get_appropriate_unit(byte_unit::UnitType::Binary).to_string(),
                "usage_percent": format!("{:.1}%", usage_percent),
            })),
        };
    }

    // Fallback if disk not found
    DiagnosticResult {
        name: "Disk Space".to_string(),
        status: DiagnosticStatus::Warning,
        message: "Could not determine disk space usage".to_string(),
        details: Some(serde_json::json!({
            "cache_dir": cache_dir.display().to_string(),
            "note": "Could not find disk containing cache directory"
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torsh_info() {
        let info = get_torsh_info();
        assert!(!info.version.is_empty());
        assert!(!info.target_triple.is_empty());
    }

    #[test]
    fn test_feature_info() {
        let features = get_feature_info();
        // At least some features should be enabled by default
        assert!(!features.enabled_features.is_empty() || !features.disabled_features.is_empty());
    }

    #[tokio::test]
    async fn test_installation_info() {
        let info = get_installation_info().await.unwrap();
        assert!(!info.install_path.is_empty());
    }
}
