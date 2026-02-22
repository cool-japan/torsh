//! # Utility Functions for ToRSh Hub
//!
//! This module provides various utility functions to simplify common tasks
//! when working with the ToRSh Hub, including model inspection, path management,
//! format conversion, and validation helpers.
//!
//! ## Features
//!
//! - **Model Inspection**: Get model size, parameter count, and architecture info
//! - **Path Management**: Safe path handling and cache directory management
//! - **Format Conversion**: Convert between different model formats and metadata
//! - **Validation**: Common validation functions for inputs and outputs
//! - **Display Helpers**: Format model information for display
//!
//! ## Examples
//!
//! ```no_run
//! use torsh_hub::utils::*;
//! use std::path::Path;
//!
//! // Format file sizes for display
//! let size = format_size(1024 * 1024 * 100); // "100.00 MB"
//!
//! // Sanitize model names
//! let safe_name = sanitize_model_name("My Model (v1.0)"); // "my-model-v1-0"
//!
//! // Check if a path is safe
//! assert!(is_safe_path(Path::new("./models/my_model.onnx")));
//! ```

use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};

/// Format a file size in bytes to a human-readable string
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::format_size;
///
/// assert_eq!(format_size(1024), "1.00 KB");
/// assert_eq!(format_size(1024 * 1024), "1.00 MB");
/// assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
/// ```
pub fn format_size(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    const TB: f64 = GB * 1024.0;

    let bytes_f = bytes as f64;

    if bytes_f >= TB {
        format!("{:.2} TB", bytes_f / TB)
    } else if bytes_f >= GB {
        format!("{:.2} GB", bytes_f / GB)
    } else if bytes_f >= MB {
        format!("{:.2} MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.2} KB", bytes_f / KB)
    } else {
        format!("{} B", bytes)
    }
}

/// Sanitize a model name to be filesystem-safe
///
/// Converts to lowercase, replaces spaces and special characters with hyphens,
/// and removes consecutive hyphens.
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::sanitize_model_name;
///
/// assert_eq!(sanitize_model_name("My Model"), "my-model");
/// assert_eq!(sanitize_model_name("BERT (base)"), "bert-base");
/// assert_eq!(sanitize_model_name("GPT-2  v1.0"), "gpt-2-v1-0");
/// ```
pub fn sanitize_model_name(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Check if a path is safe (doesn't contain directory traversal)
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::is_safe_path;
/// use std::path::Path;
///
/// assert!(is_safe_path(Path::new("models/my_model.onnx")));
/// assert!(!is_safe_path(Path::new("../../../etc/passwd")));
/// assert!(!is_safe_path(Path::new("/absolute/path")));
/// ```
pub fn is_safe_path(path: &Path) -> bool {
    // Check for absolute paths
    if path.is_absolute() {
        return false;
    }

    // Check for directory traversal
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => return false,
            std::path::Component::RootDir => return false,
            _ => {}
        }
    }

    true
}

/// Get the cache directory for a specific model
///
/// Creates the directory if it doesn't exist.
///
/// # Examples
///
/// ```no_run
/// use torsh_hub::utils::get_model_cache_dir;
///
/// let cache_dir = get_model_cache_dir("bert-base-uncased").unwrap();
/// println!("Cache directory: {:?}", cache_dir);
/// ```
pub fn get_model_cache_dir(model_name: &str) -> Result<PathBuf> {
    let sanitized_name = sanitize_model_name(model_name);

    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| TorshError::IoError("Could not determine cache directory".into()))?
        .join("torsh")
        .join("hub")
        .join("models")
        .join(sanitized_name);

    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Get the temporary directory for hub operations
///
/// Creates the directory if it doesn't exist.
pub fn get_temp_dir() -> Result<PathBuf> {
    let temp_dir = std::env::temp_dir()
        .join("torsh-hub")
        .join(uuid::Uuid::new_v4().to_string());

    std::fs::create_dir_all(&temp_dir)?;
    Ok(temp_dir)
}

/// Parse a model repository string in the format "org/model" or "org/model:tag"
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::parse_repo_string;
///
/// let (org, model, tag) = parse_repo_string("huggingface/bert-base").unwrap();
/// assert_eq!(org, "huggingface");
/// assert_eq!(model, "bert-base");
/// assert_eq!(tag, None);
///
/// let (org, model, tag) = parse_repo_string("openai/gpt2:v1.0").unwrap();
/// assert_eq!(org, "openai");
/// assert_eq!(model, "gpt2");
/// assert_eq!(tag, Some("v1.0".to_string()));
/// ```
pub fn parse_repo_string(repo: &str) -> Result<(String, String, Option<String>)> {
    let parts: Vec<&str> = repo.split(':').collect();

    let (repo_part, tag) = match parts.len() {
        1 => (parts[0], None),
        2 => (parts[0], Some(parts[1].to_string())),
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid repository string: {}",
                repo
            )))
        }
    };

    let repo_parts: Vec<&str> = repo_part.split('/').collect();
    if repo_parts.len() != 2 {
        return Err(TorshError::InvalidArgument(format!(
            "Invalid repository format, expected 'org/model': {}",
            repo
        )));
    }

    Ok((repo_parts[0].to_string(), repo_parts[1].to_string(), tag))
}

/// Extract file extension from a path or URL
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::extract_extension;
///
/// assert_eq!(extract_extension("model.onnx"), Some("onnx"));
/// assert_eq!(extract_extension("https://example.com/model.safetensors"), Some("safetensors"));
/// assert_eq!(extract_extension("model"), None);
/// ```
pub fn extract_extension(path: &str) -> Option<&str> {
    Path::new(path).extension().and_then(|ext| ext.to_str())
}

/// Check if a file extension is a supported model format
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::is_supported_model_format;
///
/// assert!(is_supported_model_format("onnx"));
/// assert!(is_supported_model_format("safetensors"));
/// assert!(is_supported_model_format("pt"));
/// assert!(!is_supported_model_format("txt"));
/// ```
pub fn is_supported_model_format(extension: &str) -> bool {
    matches!(
        extension.to_lowercase().as_str(),
        "onnx" | "pb" | "pt" | "pth" | "safetensors" | "bin" | "h5" | "tflite" | "torsh"
    )
}

/// Estimate the number of parameters in a model from file size
///
/// This is a rough estimation assuming 4 bytes per parameter (float32).
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::estimate_parameters_from_size;
///
/// // 100 MB file ~= 25M parameters
/// assert_eq!(estimate_parameters_from_size(100 * 1024 * 1024), 26214400);
/// ```
pub fn estimate_parameters_from_size(size_bytes: u64) -> u64 {
    // Assume float32 (4 bytes per parameter)
    size_bytes / 4
}

/// Format a parameter count to human-readable form
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::format_parameter_count;
///
/// assert_eq!(format_parameter_count(1_500_000), "1.50M");
/// assert_eq!(format_parameter_count(15_000_000_000), "15.00B");
/// assert_eq!(format_parameter_count(500), "500");
/// ```
pub fn format_parameter_count(count: u64) -> String {
    const MILLION: f64 = 1_000_000.0;
    const BILLION: f64 = 1_000_000_000.0;

    let count_f = count as f64;

    if count_f >= BILLION {
        format!("{:.2}B", count_f / BILLION)
    } else if count_f >= MILLION {
        format!("{:.2}M", count_f / MILLION)
    } else {
        format!("{}", count)
    }
}

/// Validate a semantic version string
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::validate_semver;
///
/// assert!(validate_semver("1.0.0").is_ok());
/// assert!(validate_semver("2.1.3-alpha").is_ok());
/// assert!(validate_semver("invalid").is_err());
/// ```
pub fn validate_semver(version: &str) -> Result<()> {
    let parts: Vec<&str> = version
        .split('-')
        .next()
        .unwrap_or(version)
        .split('.')
        .collect();

    if parts.len() < 2 || parts.len() > 3 {
        return Err(TorshError::InvalidArgument(format!(
            "Invalid semantic version format: {}",
            version
        )));
    }

    for part in &parts {
        if part.parse::<u32>().is_err() {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid version number in: {}",
                version
            )));
        }
    }

    Ok(())
}

/// Compare two semantic versions
///
/// Returns:
/// - `std::cmp::Ordering::Less` if v1 < v2
/// - `std::cmp::Ordering::Equal` if v1 == v2
/// - `std::cmp::Ordering::Greater` if v1 > v2
///
/// # Examples
///
/// ```
/// use torsh_hub::utils::compare_versions;
/// use std::cmp::Ordering;
///
/// assert_eq!(compare_versions("1.0.0", "2.0.0").unwrap(), Ordering::Less);
/// assert_eq!(compare_versions("2.1.0", "2.0.0").unwrap(), Ordering::Greater);
/// assert_eq!(compare_versions("1.0.0", "1.0.0").unwrap(), Ordering::Equal);
/// ```
pub fn compare_versions(v1: &str, v2: &str) -> Result<std::cmp::Ordering> {
    validate_semver(v1)?;
    validate_semver(v2)?;

    let v1_parts: Vec<u32> = v1
        .split('-')
        .next()
        .unwrap_or(v1)
        .split('.')
        .filter_map(|s| s.parse().ok())
        .collect();

    let v2_parts: Vec<u32> = v2
        .split('-')
        .next()
        .unwrap_or(v2)
        .split('.')
        .filter_map(|s| s.parse().ok())
        .collect();

    for i in 0..v1_parts.len().max(v2_parts.len()) {
        let v1_part = v1_parts.get(i).copied().unwrap_or(0);
        let v2_part = v2_parts.get(i).copied().unwrap_or(0);

        match v1_part.cmp(&v2_part) {
            std::cmp::Ordering::Equal => continue,
            ordering => return Ok(ordering),
        }
    }

    Ok(std::cmp::Ordering::Equal)
}

/// Clean up old cache files based on age
///
/// Removes files older than the specified number of days.
///
/// # Examples
///
/// ```no_run
/// use torsh_hub::utils::cleanup_old_cache;
///
/// // Remove cache files older than 30 days
/// let removed_count = cleanup_old_cache(30).unwrap();
/// println!("Removed {} old cache files", removed_count);
/// ```
pub fn cleanup_old_cache(days: u64) -> Result<usize> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| TorshError::IoError("Could not determine cache directory".into()))?
        .join("torsh")
        .join("hub");

    if !cache_dir.exists() {
        return Ok(0);
    }

    let cutoff_time =
        std::time::SystemTime::now() - std::time::Duration::from_secs(days * 24 * 60 * 60);

    let mut removed_count = 0;

    fn remove_old_files(
        dir: &Path,
        cutoff: std::time::SystemTime,
        count: &mut usize,
    ) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                remove_old_files(&path, cutoff, count)?;

                // Try to remove directory if empty
                if std::fs::read_dir(&path)?.next().is_none() {
                    let _ = std::fs::remove_dir(&path);
                }
            } else if path.is_file() {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if modified < cutoff {
                            std::fs::remove_file(&path)?;
                            *count += 1;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    remove_old_files(&cache_dir, cutoff_time, &mut removed_count)?;

    Ok(removed_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_size(1024 * 1024 * 1024 * 1024), "1.00 TB");
    }

    #[test]
    fn test_sanitize_model_name() {
        assert_eq!(sanitize_model_name("simple"), "simple");
        assert_eq!(sanitize_model_name("My Model"), "my-model");
        assert_eq!(sanitize_model_name("BERT-base"), "bert-base");
        assert_eq!(sanitize_model_name("GPT (v2)"), "gpt-v2");
        assert_eq!(sanitize_model_name("model___name"), "model-name");
        assert_eq!(sanitize_model_name("test--model"), "test-model");
    }

    #[test]
    fn test_is_safe_path() {
        assert!(is_safe_path(Path::new("model.onnx")));
        assert!(is_safe_path(Path::new("models/bert/model.onnx")));
        assert!(!is_safe_path(Path::new("../etc/passwd")));
        assert!(!is_safe_path(Path::new("/absolute/path")));
        assert!(!is_safe_path(Path::new("models/../../etc/passwd")));
    }

    #[test]
    fn test_parse_repo_string() {
        let (org, model, tag) = parse_repo_string("huggingface/bert").unwrap();
        assert_eq!(org, "huggingface");
        assert_eq!(model, "bert");
        assert_eq!(tag, None);

        let (org, model, tag) = parse_repo_string("openai/gpt2:v1.0").unwrap();
        assert_eq!(org, "openai");
        assert_eq!(model, "gpt2");
        assert_eq!(tag, Some("v1.0".to_string()));

        assert!(parse_repo_string("invalid").is_err());
        assert!(parse_repo_string("too/many/parts").is_err());
    }

    #[test]
    fn test_extract_extension() {
        assert_eq!(extract_extension("model.onnx"), Some("onnx"));
        assert_eq!(extract_extension("model.pt"), Some("pt"));
        assert_eq!(
            extract_extension("/path/to/model.safetensors"),
            Some("safetensors")
        );
        assert_eq!(extract_extension("model"), None);
        assert_eq!(extract_extension(""), None);
    }

    #[test]
    fn test_is_supported_model_format() {
        assert!(is_supported_model_format("onnx"));
        assert!(is_supported_model_format("ONNX"));
        assert!(is_supported_model_format("pt"));
        assert!(is_supported_model_format("safetensors"));
        assert!(is_supported_model_format("pb"));
        assert!(!is_supported_model_format("txt"));
        assert!(!is_supported_model_format("jpg"));
    }

    #[test]
    fn test_estimate_parameters_from_size() {
        assert_eq!(estimate_parameters_from_size(4), 1);
        assert_eq!(estimate_parameters_from_size(400), 100);
        assert_eq!(estimate_parameters_from_size(1024 * 1024 * 4), 1_048_576);
    }

    #[test]
    fn test_format_parameter_count() {
        assert_eq!(format_parameter_count(100), "100");
        assert_eq!(format_parameter_count(1_500_000), "1.50M");
        assert_eq!(format_parameter_count(125_000_000), "125.00M");
        assert_eq!(format_parameter_count(1_500_000_000), "1.50B");
    }

    #[test]
    fn test_validate_semver() {
        assert!(validate_semver("1.0.0").is_ok());
        assert!(validate_semver("2.1.3").is_ok());
        assert!(validate_semver("1.0").is_ok());
        assert!(validate_semver("1.0.0-alpha").is_ok());
        assert!(validate_semver("invalid").is_err());
        assert!(validate_semver("1").is_err());
        assert!(validate_semver("1.a.0").is_err());
    }

    #[test]
    fn test_compare_versions() {
        use std::cmp::Ordering;

        assert_eq!(compare_versions("1.0.0", "2.0.0").unwrap(), Ordering::Less);
        assert_eq!(
            compare_versions("2.0.0", "1.0.0").unwrap(),
            Ordering::Greater
        );
        assert_eq!(compare_versions("1.0.0", "1.0.0").unwrap(), Ordering::Equal);
        assert_eq!(
            compare_versions("1.2.0", "1.1.9").unwrap(),
            Ordering::Greater
        );
        assert_eq!(compare_versions("1.0.1", "1.0.2").unwrap(), Ordering::Less);
        assert_eq!(compare_versions("2.0", "2.0.0").unwrap(), Ordering::Equal);
    }
}
