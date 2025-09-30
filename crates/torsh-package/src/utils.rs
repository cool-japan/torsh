//! Utility functions
//!
//! This module contains common utility functions used throughout the package system,
//! including cryptographic operations, validation, and helper functions.

use sha2::{Digest, Sha256};
use std::path::Path;
use torsh_core::error::Result;

use crate::package::Package;

/// Calculate SHA-256 hash of data
pub fn calculate_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Quick export function (temporarily disabled - requires torsh-nn)
#[cfg(feature = "with-nn")]
pub fn export_module<M: torsh_nn::Module, P: AsRef<Path>>(
    module: &M,
    path: P,
    name: &str,
    version: &str,
) -> Result<()> {
    crate::builder::PackageBuilder::new(name.to_string(), version.to_string())
        .add_module("main", module)?
        .build(path)
}

/// Quick import function
pub fn import_module<P: AsRef<Path>>(path: P, module_name: &str) -> Result<Vec<u8>> {
    let package = Package::load(path)?;
    package.get_module(module_name)
}

/// Validate package name according to naming conventions
pub fn validate_package_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 100 {
        return false;
    }

    // Must start with alphanumeric, can contain alphanumeric, hyphens, and underscores
    let first_char = name.chars().next().unwrap();
    if !first_char.is_alphanumeric() {
        return false;
    }

    name.chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
}

/// Validate semantic version string
pub fn validate_version(version: &str) -> bool {
    semver::Version::parse(version).is_ok()
}

/// Get file extension from resource name
pub fn get_file_extension(filename: &str) -> Option<&str> {
    std::path::Path::new(filename)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
}

/// Sanitize filename for safe storage
pub fn sanitize_filename(filename: &str) -> String {
    filename
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Check if file path is safe (no directory traversal)
pub fn is_safe_path(path: &str) -> bool {
    !path.contains("..") && !path.starts_with('/') && !path.starts_with('\\')
}

/// Format file size in human-readable format
pub fn format_file_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: u64 = 1024;

    if size == 0 {
        return "0 B".to_string();
    }

    let mut size = size as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD as f64 && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD as f64;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Estimate compression ratio for data
pub fn estimate_compression_ratio(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 1.0;
    }

    // Simple entropy estimation
    let mut counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    // Rough compression ratio estimation based on entropy
    // Maximum entropy is 8 bits, so we can estimate compression potential
    let max_entropy = 8.0;
    let compression_potential = (max_entropy - entropy) / max_entropy;
    1.0 - compression_potential.max(0.0).min(0.9) // Cap at 90% compression
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_hash() {
        let data = b"hello world";
        let hash = calculate_hash(data);

        // SHA256 of "hello world" is known
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_validate_package_name() {
        assert!(validate_package_name("my-package"));
        assert!(validate_package_name("package_name"));
        assert!(validate_package_name("Package123"));

        assert!(!validate_package_name(""));
        assert!(!validate_package_name("-invalid"));
        assert!(!validate_package_name("invalid@name"));
        assert!(!validate_package_name("a".repeat(101).as_str()));
    }

    #[test]
    fn test_validate_version() {
        assert!(validate_version("1.0.0"));
        assert!(validate_version("2.1.3-alpha.1"));
        assert!(validate_version("0.0.1-beta+build.123"));

        assert!(!validate_version(""));
        assert!(!validate_version("1.0"));
        assert!(!validate_version("invalid"));
    }

    #[test]
    fn test_get_file_extension() {
        assert_eq!(get_file_extension("file.txt"), Some("txt"));
        assert_eq!(get_file_extension("archive.tar.gz"), Some("gz"));
        assert_eq!(get_file_extension("README"), None);
        assert_eq!(get_file_extension(".hidden"), None);
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("normal_file.txt"), "normal_file.txt");
        assert_eq!(
            sanitize_filename("file with spaces.txt"),
            "file_with_spaces.txt"
        );
        assert_eq!(sanitize_filename("file@#$%.txt"), "file____.txt");
        assert_eq!(sanitize_filename("αβγ.txt"), "___.txt");
    }

    #[test]
    fn test_is_safe_path() {
        assert!(is_safe_path("safe/path/file.txt"));
        assert!(is_safe_path("file.txt"));
        assert!(is_safe_path("subdir/file.txt"));

        assert!(!is_safe_path("../etc/passwd"));
        assert!(!is_safe_path("/absolute/path"));
        assert!(!is_safe_path("\\windows\\path"));
        assert!(!is_safe_path("safe/../unsafe"));
    }

    #[test]
    fn test_format_file_size() {
        assert_eq!(format_file_size(0), "0 B");
        assert_eq!(format_file_size(512), "512 B");
        assert_eq!(format_file_size(1024), "1.0 KB");
        assert_eq!(format_file_size(1536), "1.5 KB");
        assert_eq!(format_file_size(1048576), "1.0 MB");
        assert_eq!(format_file_size(1073741824), "1.0 GB");
    }

    #[test]
    fn test_estimate_compression_ratio() {
        // Highly repetitive data should compress well
        let repetitive = vec![b'A'; 1000];
        let ratio = estimate_compression_ratio(&repetitive);
        assert!(ratio < 0.5); // Should compress to less than 50%

        // Random data should compress poorly
        let random: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        let ratio = estimate_compression_ratio(&random);
        assert!(ratio > 0.8); // Should compress poorly

        // Empty data
        assert_eq!(estimate_compression_ratio(&[]), 1.0);
    }

    #[test]
    fn test_import_module_nonexistent() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let nonexistent_path = temp_dir.path().join("nonexistent.torshpkg");

        let result = import_module(nonexistent_path, "test");
        assert!(result.is_err());
    }
}
