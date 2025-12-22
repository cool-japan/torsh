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

/// Validate resource path is within package bounds
pub fn validate_resource_path(path: &str) -> Result<()> {
    use torsh_core::error::TorshError;

    if path.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Resource path cannot be empty".to_string(),
        ));
    }

    if path.len() > 1024 {
        return Err(TorshError::InvalidArgument(
            "Resource path exceeds maximum length of 1024 characters".to_string(),
        ));
    }

    if !is_safe_path(path) {
        return Err(TorshError::InvalidArgument(format!(
            "Resource path contains unsafe components: {}",
            path
        )));
    }

    Ok(())
}

/// Validate package metadata integrity
pub fn validate_package_metadata(
    name: &str,
    version: &str,
    description: Option<&str>,
) -> Result<()> {
    use torsh_core::error::TorshError;

    if !validate_package_name(name) {
        return Err(TorshError::InvalidArgument(format!(
            "Invalid package name: {}",
            name
        )));
    }

    if !validate_version(version) {
        return Err(TorshError::InvalidArgument(format!(
            "Invalid semantic version: {}",
            version
        )));
    }

    if let Some(desc) = description {
        if desc.len() > 10000 {
            return Err(TorshError::InvalidArgument(
                "Package description exceeds maximum length of 10000 characters".to_string(),
            ));
        }
    }

    Ok(())
}

/// Calculate checksum for integrity verification
pub fn calculate_checksum(data: &[u8]) -> u64 {
    // Simple CRC-64 implementation
    let mut checksum = 0u64;
    for &byte in data {
        checksum = checksum.wrapping_mul(31).wrapping_add(byte as u64);
    }
    checksum
}

/// Verify data integrity using checksum
pub fn verify_checksum(data: &[u8], expected: u64) -> bool {
    calculate_checksum(data) == expected
}

/// Normalize path separators to forward slashes
pub fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

/// Get relative path between two paths
pub fn get_relative_path(from: &str, to: &str) -> String {
    let from_parts: Vec<&str> = from.split('/').filter(|s| !s.is_empty()).collect();
    let to_parts: Vec<&str> = to.split('/').filter(|s| !s.is_empty()).collect();

    let mut common = 0;
    for (a, b) in from_parts.iter().zip(to_parts.iter()) {
        if a == b {
            common += 1;
        } else {
            break;
        }
    }

    let mut result = Vec::new();
    for _ in common..from_parts.len() {
        result.push("..");
    }
    result.extend(to_parts[common..].iter());

    if result.is_empty() {
        ".".to_string()
    } else {
        result.join("/")
    }
}

/// Parse content type from file extension
pub fn parse_content_type(filename: &str) -> &'static str {
    match get_file_extension(filename) {
        Some("txt") | Some("md") => "text/plain",
        Some("json") => "application/json",
        Some("xml") => "application/xml",
        Some("html") => "text/html",
        Some("css") => "text/css",
        Some("js") => "application/javascript",
        Some("py") => "text/x-python",
        Some("rs") => "text/x-rust",
        Some("zip") => "application/zip",
        Some("tar") => "application/x-tar",
        Some("gz") => "application/gzip",
        Some("torshpkg") => "application/x-torsh-package",
        Some("onnx") => "application/onnx",
        Some("pkl") | Some("pickle") => "application/python-pickle",
        _ => "application/octet-stream",
    }
}

/// Performance timer for operation profiling
#[derive(Debug, Clone)]
pub struct PerformanceTimer {
    start: std::time::Instant,
    name: String,
}

impl PerformanceTimer {
    /// Create a new performance timer
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start: std::time::Instant::now(),
            name: name.into(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Get elapsed time in seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Print elapsed time
    pub fn print_elapsed(&self) {
        eprintln!("[{}] Elapsed: {:.3}s", self.name, self.elapsed_secs());
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.start = std::time::Instant::now();
    }
}

impl Drop for PerformanceTimer {
    fn drop(&mut self) {
        if cfg!(debug_assertions) {
            self.print_elapsed();
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes allocated
    pub allocated: u64,
    /// Peak memory usage
    pub peak: u64,
    /// Number of allocations
    pub allocations: u64,
}

impl MemoryStats {
    /// Create new memory statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record allocation
    pub fn record_allocation(&mut self, size: u64) {
        self.allocated += size;
        self.allocations += 1;
        if self.allocated > self.peak {
            self.peak = self.allocated;
        }
    }

    /// Record deallocation
    pub fn record_deallocation(&mut self, size: u64) {
        self.allocated = self.allocated.saturating_sub(size);
    }

    /// Format memory stats as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Allocated: {}, Peak: {}, Allocations: {}",
            format_file_size(self.allocated),
            format_file_size(self.peak),
            self.allocations
        )
    }
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

    #[test]
    fn test_validate_resource_path() {
        assert!(validate_resource_path("valid/path.txt").is_ok());
        assert!(validate_resource_path("another_file.rs").is_ok());

        assert!(validate_resource_path("").is_err());
        assert!(validate_resource_path("../unsafe").is_err());
        assert!(validate_resource_path("/absolute").is_err());
        assert!(validate_resource_path(&"x".repeat(1025)).is_err());
    }

    #[test]
    fn test_validate_package_metadata() {
        assert!(validate_package_metadata("my-package", "1.0.0", None).is_ok());
        assert!(validate_package_metadata("test", "2.1.3", Some("A test package")).is_ok());

        assert!(validate_package_metadata("", "1.0.0", None).is_err());
        assert!(validate_package_metadata("test", "invalid", None).is_err());
        assert!(validate_package_metadata("test", "1.0.0", Some(&"x".repeat(10001))).is_err());
    }

    #[test]
    fn test_calculate_checksum() {
        let data1 = b"hello world";
        let data2 = b"hello world";
        let data3 = b"different data";

        let checksum1 = calculate_checksum(data1);
        let checksum2 = calculate_checksum(data2);
        let checksum3 = calculate_checksum(data3);

        assert_eq!(checksum1, checksum2);
        assert_ne!(checksum1, checksum3);
    }

    #[test]
    fn test_verify_checksum() {
        let data = b"test data";
        let checksum = calculate_checksum(data);

        assert!(verify_checksum(data, checksum));
        assert!(!verify_checksum(data, checksum + 1));
    }

    #[test]
    fn test_normalize_path() {
        assert_eq!(normalize_path("path/to/file"), "path/to/file");
        assert_eq!(normalize_path("path\\to\\file"), "path/to/file");
        assert_eq!(normalize_path("mixed\\path/to\\file"), "mixed/path/to/file");
    }

    #[test]
    fn test_get_relative_path() {
        assert_eq!(get_relative_path("a/b/c", "a/b/d"), "../d");
        assert_eq!(get_relative_path("a/b", "a/b/c/d"), "c/d");
        assert_eq!(get_relative_path("a/b/c", "a/b/c"), ".");
        assert_eq!(get_relative_path("a/b/c", "x/y/z"), "../../../x/y/z");
        assert_eq!(get_relative_path("a/b", "x/y"), "../../x/y");
    }

    #[test]
    fn test_parse_content_type() {
        assert_eq!(parse_content_type("file.txt"), "text/plain");
        assert_eq!(parse_content_type("data.json"), "application/json");
        assert_eq!(parse_content_type("script.py"), "text/x-python");
        assert_eq!(parse_content_type("code.rs"), "text/x-rust");
        assert_eq!(parse_content_type("model.onnx"), "application/onnx");
        assert_eq!(
            parse_content_type("package.torshpkg"),
            "application/x-torsh-package"
        );
        assert_eq!(
            parse_content_type("unknown.xyz"),
            "application/octet-stream"
        );
    }

    #[test]
    fn test_performance_timer() {
        let timer = PerformanceTimer::new("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10);
        assert!(elapsed < 100);
    }

    #[test]
    fn test_memory_stats() {
        let mut stats = MemoryStats::new();
        assert_eq!(stats.allocated, 0);
        assert_eq!(stats.peak, 0);

        stats.record_allocation(1024);
        assert_eq!(stats.allocated, 1024);
        assert_eq!(stats.peak, 1024);
        assert_eq!(stats.allocations, 1);

        stats.record_allocation(512);
        assert_eq!(stats.allocated, 1536);
        assert_eq!(stats.peak, 1536);
        assert_eq!(stats.allocations, 2);

        stats.record_deallocation(512);
        assert_eq!(stats.allocated, 1024);
        assert_eq!(stats.peak, 1536); // Peak stays the same

        let formatted = stats.format();
        assert!(formatted.contains("KB"));
    }
}
