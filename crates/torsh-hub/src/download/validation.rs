//! Validation utilities for download operations
//!
//! This module provides comprehensive validation functions for URLs, files, hashes,
//! archives, and security checks used throughout the ToRSh Hub download system.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};

/// Supported hash algorithms for file verification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Sha256,
}

impl HashAlgorithm {
    /// Get the expected hex string length for this hash algorithm
    pub fn hex_length(&self) -> usize {
        match self {
            HashAlgorithm::Sha256 => 64,
        }
    }

    /// Get the algorithm name as a string
    pub fn name(&self) -> &'static str {
        match self {
            HashAlgorithm::Sha256 => "SHA-256",
        }
    }
}

impl Default for HashAlgorithm {
    fn default() -> Self {
        HashAlgorithm::Sha256
    }
}

/// Validate a URL before attempting to download from it
///
/// This function performs comprehensive URL validation to provide better error messages
/// to users before attempting downloads.
///
/// # Arguments
/// * `url` - The URL to validate
///
/// # Returns
/// * `Ok(())` if the URL appears valid
/// * `Err(TorshError)` with a descriptive message if invalid
///
/// # Examples
/// ```rust
/// use torsh_hub::download::validation::validate_url;
///
/// // Valid URL
/// assert!(validate_url("https://example.com/model.torsh").is_ok());
///
/// // Invalid URL
/// assert!(validate_url("not-a-url").is_err());
/// ```
pub fn validate_url(url: &str) -> Result<()> {
    // Check if URL is empty
    if url.trim().is_empty() {
        return Err(TorshError::config_error_with_context(
            "URL cannot be empty",
            "URL validation",
        ));
    }

    // Basic URL format validation
    if !url.starts_with("http://") && !url.starts_with("https://") && !url.starts_with("ftp://") {
        return Err(TorshError::config_error_with_context(
            &format!(
                "URL must start with http://, https://, or ftp://, got: {}",
                url
            ),
            "URL validation",
        ));
    }

    // Check for obviously invalid characters
    if url.contains(' ') {
        return Err(TorshError::config_error_with_context(
            &format!("URL contains spaces: {}", url),
            "URL validation",
        ));
    }

    // Check for null bytes (security issue)
    if url.contains('\0') {
        return Err(TorshError::config_error_with_context(
            "URL contains null bytes",
            "URL validation - security",
        ));
    }

    // Check if URL has a reasonable length (prevent extremely long URLs)
    if url.len() > 2048 {
        return Err(TorshError::config_error_with_context(
            &format!("URL is too long ({} characters, max 2048)", url.len()),
            "URL validation",
        ));
    }

    // Validate that URL has a domain or is localhost/IP
    let has_valid_host = url.contains('.')
        || url.contains("localhost")
        || url.chars().filter(|&c| c == '/').count() >= 2;
    if !has_valid_host {
        return Err(TorshError::config_error_with_context(
            &format!("URL appears to be malformed: {}", url),
            "URL validation",
        ));
    }

    Ok(())
}

/// Validate multiple URLs at once
///
/// This is a convenience function for validating multiple URLs and collecting
/// all validation errors.
///
/// # Arguments
/// * `urls` - A slice of URLs to validate
///
/// # Returns
/// * `Ok(())` if all URLs are valid
/// * `Err(TorshError)` with details about all invalid URLs
pub fn validate_urls(urls: &[&str]) -> Result<()> {
    let mut errors = Vec::new();

    for (i, url) in urls.iter().enumerate() {
        if let Err(e) = validate_url(url) {
            errors.push(format!("URL {}: {}", i + 1, e));
        }
    }

    if !errors.is_empty() {
        return Err(TorshError::config_error_with_context(
            &format!("URL validation failed:\n{}", errors.join("\n")),
            "Multiple URL validation",
        ));
    }

    Ok(())
}

/// Validate HTTP response status codes
///
/// # Arguments
/// * `status_code` - The HTTP status code to validate
///
/// # Returns
/// * `Ok(())` if the status indicates success
/// * `Err(TorshError)` for error status codes with appropriate messages
pub fn validate_http_status(status_code: u16) -> Result<()> {
    match status_code {
        200..=299 => Ok(()),
        300..=399 => Err(TorshError::IoError(format!(
            "HTTP redirect status {}: server returned a redirect that should be handled automatically",
            status_code
        ))),
        400 => Err(TorshError::IoError("HTTP 400: Bad request - check URL format".to_string())),
        401 => Err(TorshError::IoError("HTTP 401: Unauthorized - authentication required".to_string())),
        403 => Err(TorshError::IoError("HTTP 403: Forbidden - access denied".to_string())),
        404 => Err(TorshError::IoError("HTTP 404: Not found - resource does not exist".to_string())),
        408 => Err(TorshError::IoError("HTTP 408: Request timeout - server took too long to respond".to_string())),
        429 => Err(TorshError::IoError("HTTP 429: Too many requests - rate limited, try again later".to_string())),
        500 => Err(TorshError::IoError("HTTP 500: Internal server error - server encountered an error".to_string())),
        502 => Err(TorshError::IoError("HTTP 502: Bad gateway - upstream server error".to_string())),
        503 => Err(TorshError::IoError("HTTP 503: Service unavailable - server temporarily unavailable".to_string())),
        504 => Err(TorshError::IoError("HTTP 504: Gateway timeout - upstream server timeout".to_string())),
        _ => Err(TorshError::IoError(format!("HTTP {}: Unexpected status code", status_code))),
    }
}

/// Verify downloaded file integrity
///
/// This function performs comprehensive file verification including existence,
/// size, and hash validation.
///
/// # Arguments
/// * `file_path` - Path to the file to verify
/// * `expected_hash` - Optional expected hash value (hex string)
/// * `expected_size` - Optional expected file size in bytes
/// * `hash_algorithm` - Hash algorithm to use for verification
///
/// # Returns
/// * `Ok(true)` if all verifications pass
/// * `Ok(false)` if file exists but verification fails
/// * `Err(TorshError)` for I/O or other errors
pub fn verify_file_integrity(
    file_path: &Path,
    expected_hash: Option<&str>,
    expected_size: Option<u64>,
    hash_algorithm: HashAlgorithm,
) -> Result<bool> {
    // Check file exists
    if !file_path.exists() {
        return Ok(false);
    }

    // Check size if provided
    if let Some(size) = expected_size {
        let metadata = fs::metadata(file_path)?;
        if metadata.len() != size {
            return Ok(false);
        }
    }

    // Check hash if provided
    if let Some(hash) = expected_hash {
        let actual_hash = calculate_file_hash(file_path, hash_algorithm)?;
        return Ok(actual_hash.to_lowercase() == hash.to_lowercase());
    }

    Ok(true)
}

/// Calculate hash of a file using the specified algorithm
///
/// # Arguments
/// * `file_path` - Path to the file to hash
/// * `algorithm` - Hash algorithm to use (currently only SHA-256 is supported)
///
/// # Returns
/// * `Ok(String)` containing the hex-encoded SHA-256 hash
/// * `Err(TorshError)` for I/O or other errors
pub fn calculate_file_hash(file_path: &Path, algorithm: HashAlgorithm) -> Result<String> {
    use sha2::{Digest, Sha256};

    let mut file = std::fs::File::open(file_path)?;
    let mut buffer = [0; 8192];

    match algorithm {
        HashAlgorithm::Sha256 => {
            let mut hasher = Sha256::new();
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buffer[..n]);
            }
            Ok(hex::encode(hasher.finalize()))
        }
    }
}

/// Validate a hash string format
///
/// # Arguments
/// * `hash` - The hash string to validate
/// * `algorithm` - Expected hash algorithm
///
/// # Returns
/// * `Ok(())` if hash format is valid
/// * `Err(TorshError)` if format is invalid
pub fn validate_hash_format(hash: &str, algorithm: HashAlgorithm) -> Result<()> {
    if hash.is_empty() {
        return Err(TorshError::config_error_with_context(
            "Hash cannot be empty",
            "Hash validation",
        ));
    }

    let expected_length = algorithm.hex_length();
    if hash.len() != expected_length {
        return Err(TorshError::config_error_with_context(
            &format!(
                "Invalid {} hash length: expected {} characters, got {}",
                algorithm.name(),
                expected_length,
                hash.len()
            ),
            "Hash validation",
        ));
    }

    // Check if all characters are valid hex
    if !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(TorshError::config_error_with_context(
            &format!(
                "Invalid {} hash format: contains non-hex characters",
                algorithm.name()
            ),
            "Hash validation",
        ));
    }

    Ok(())
}

/// Validate file path for security issues
///
/// This function checks for path traversal attacks and other security issues.
///
/// # Arguments
/// * `path` - The file path to validate
/// * `base_dir` - Optional base directory to restrict paths to
///
/// # Returns
/// * `Ok(())` if path is safe
/// * `Err(TorshError)` if path has security issues
pub fn validate_file_path(path: &Path, base_dir: Option<&Path>) -> Result<()> {
    // Check for path traversal attempts
    let path_str = path.to_string_lossy();
    if path_str.contains("..") {
        return Err(TorshError::config_error_with_context(
            "Path contains '..' which could be a path traversal attack",
            "Security validation",
        ));
    }

    // Check for absolute paths when base_dir is specified
    if let Some(base) = base_dir {
        if path.is_absolute() {
            return Err(TorshError::config_error_with_context(
                "Absolute paths not allowed when base directory is specified",
                "Security validation",
            ));
        }

        // Resolve and check if path stays within base directory
        let full_path = base.join(path);
        // Try to canonicalize base directory, but allow for test environments where it may not exist
        let canonical_base = match base.canonicalize() {
            Ok(canonical) => canonical,
            Err(_) => {
                // For testing purposes, allow non-existent base directories
                // but still check the path doesn't escape using string comparison
                let base_str = base.to_string_lossy();
                let full_path_str = full_path.to_string_lossy();
                if full_path_str.contains("..") || !full_path_str.starts_with(&*base_str) {
                    return Err(TorshError::config_error_with_context(
                        "Path attempts to escape base directory",
                        "Security validation",
                    ));
                }
                return Ok(());
            }
        };

        // For validation, we'll use a simplified check with canonical paths
        let canonical_full_path = canonical_base.join(path);
        let full_path_str = canonical_full_path.to_string_lossy();
        let base_str = canonical_base.to_string_lossy();
        if !full_path_str.starts_with(&*base_str) {
            return Err(TorshError::config_error_with_context(
                "Path attempts to escape base directory",
                "Security validation",
            ));
        }
    }

    // Check for null bytes in path
    if path_str.contains('\0') {
        return Err(TorshError::config_error_with_context(
            "Path contains null bytes",
            "Security validation",
        ));
    }

    Ok(())
}

/// Validate archive file format based on extension
///
/// # Arguments
/// * `file_path` - Path to the archive file
/// * `allowed_formats` - List of allowed archive formats
///
/// # Returns
/// * `Ok(())` if archive format is allowed
/// * `Err(TorshError)` if format is not allowed
pub fn validate_archive_format(file_path: &Path, allowed_formats: &[&str]) -> Result<()> {
    if allowed_formats.is_empty() {
        return Ok(()); // Allow all formats if none specified
    }

    let file_name = file_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");

    let allowed_lower: Vec<String> = allowed_formats.iter().map(|f| f.to_lowercase()).collect();

    // Check for compound extensions like "tar.gz"
    let mut found_match = false;
    for allowed_format in &allowed_lower {
        if file_name
            .to_lowercase()
            .ends_with(&format!(".{}", allowed_format))
        {
            found_match = true;
            break;
        }
    }

    if !found_match {
        // Also check simple extension for backwards compatibility
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !allowed_lower.contains(&extension) {
            return Err(TorshError::config_error_with_context(
                &format!(
                    "Archive format not allowed. File: '{}', Allowed formats: {}",
                    file_name,
                    allowed_formats.join(", ")
                ),
                "Archive validation",
            ));
        }
    }

    Ok(())
}

/// Validate content type header
///
/// # Arguments
/// * `content_type` - The content-type header value
/// * `expected_types` - List of expected/allowed content types
///
/// # Returns
/// * `Ok(())` if content type is acceptable
/// * `Err(TorshError)` if content type is not allowed
pub fn validate_content_type(content_type: &str, expected_types: &[&str]) -> Result<()> {
    if expected_types.is_empty() {
        return Ok(()); // Allow all types if none specified
    }

    let content_type_lower = content_type.to_lowercase();

    for expected in expected_types {
        if content_type_lower.starts_with(&expected.to_lowercase()) {
            return Ok(());
        }
    }

    Err(TorshError::config_error_with_context(
        &format!(
            "Unexpected content type '{}'. Expected one of: {}",
            content_type,
            expected_types.join(", ")
        ),
        "Content type validation",
    ))
}

/// Validate file size constraints
///
/// # Arguments
/// * `size` - The file size to validate
/// * `min_size` - Optional minimum size in bytes
/// * `max_size` - Optional maximum size in bytes
///
/// # Returns
/// * `Ok(())` if size is within constraints
/// * `Err(TorshError)` if size is outside constraints
pub fn validate_file_size(size: u64, min_size: Option<u64>, max_size: Option<u64>) -> Result<()> {
    if let Some(min) = min_size {
        if size < min {
            return Err(TorshError::config_error_with_context(
                &format!("File size {} bytes is below minimum of {} bytes", size, min),
                "File size validation",
            ));
        }
    }

    if let Some(max) = max_size {
        if size > max {
            return Err(TorshError::config_error_with_context(
                &format!("File size {} bytes exceeds maximum of {} bytes", size, max),
                "File size validation",
            ));
        }
    }

    Ok(())
}

/// Validate range request parameters
///
/// # Arguments
/// * `start` - Start byte offset
/// * `end` - End byte offset
/// * `total_size` - Total file size
///
/// # Returns
/// * `Ok(())` if range is valid
/// * `Err(TorshError)` if range is invalid
pub fn validate_byte_range(start: u64, end: u64, total_size: u64) -> Result<()> {
    if start > end {
        return Err(TorshError::config_error_with_context(
            &format!("Invalid range: start {} is greater than end {}", start, end),
            "Range validation",
        ));
    }

    if end >= total_size {
        return Err(TorshError::config_error_with_context(
            &format!(
                "Invalid range: end {} exceeds file size {}",
                end, total_size
            ),
            "Range validation",
        ));
    }

    if start >= total_size {
        return Err(TorshError::config_error_with_context(
            &format!(
                "Invalid range: start {} exceeds file size {}",
                start, total_size
            ),
            "Range validation",
        ));
    }

    Ok(())
}

/// Legacy function for backward compatibility
///
/// Verify downloaded file integrity (simplified interface)
pub fn verify_download(
    file_path: &Path,
    expected_hash: Option<&str>,
    expected_size: Option<u64>,
) -> Result<bool> {
    verify_file_integrity(
        file_path,
        expected_hash,
        expected_size,
        HashAlgorithm::Sha256,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_validate_url() {
        // Valid URLs
        assert!(validate_url("https://example.com/file.torsh").is_ok());
        assert!(validate_url("http://localhost:8080/model").is_ok());
        assert!(validate_url("ftp://files.example.com/data").is_ok());

        // Invalid URLs
        assert!(validate_url("").is_err());
        assert!(validate_url("not-a-url").is_err());
        assert!(validate_url("https://example .com").is_err());
        assert!(validate_url(&"x".repeat(3000)).is_err());
        assert!(validate_url("https://example.com\0").is_err());
    }

    #[test]
    fn test_validate_urls() {
        let valid_urls = &["https://example.com/1", "https://example.com/2"];
        assert!(validate_urls(valid_urls).is_ok());

        let mixed_urls = &["https://example.com/1", "invalid-url"];
        assert!(validate_urls(mixed_urls).is_err());

        let empty_urls: &[&str] = &[];
        assert!(validate_urls(empty_urls).is_ok());
    }

    #[test]
    fn test_validate_http_status() {
        assert!(validate_http_status(200).is_ok());
        assert!(validate_http_status(201).is_ok());
        assert!(validate_http_status(404).is_err());
        assert!(validate_http_status(500).is_err());
    }

    #[test]
    fn test_hash_algorithm() {
        assert_eq!(HashAlgorithm::Sha256.hex_length(), 64);
        assert_eq!(HashAlgorithm::Sha256.name(), "SHA-256");
        assert_eq!(HashAlgorithm::default(), HashAlgorithm::Sha256);
    }

    #[test]
    fn test_validate_hash_format() {
        // Valid SHA-256 hash
        let valid_sha256 = "a".repeat(64);
        assert!(validate_hash_format(&valid_sha256, HashAlgorithm::Sha256).is_ok());

        // Invalid length
        let invalid_length = "a".repeat(32);
        assert!(validate_hash_format(&invalid_length, HashAlgorithm::Sha256).is_err());

        // Invalid characters
        let invalid_chars = "g".repeat(64);
        assert!(validate_hash_format(&invalid_chars, HashAlgorithm::Sha256).is_err());

        // Empty hash
        assert!(validate_hash_format("", HashAlgorithm::Sha256).is_err());
    }

    #[test]
    fn test_verify_file_integrity() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Create test file
        fs::write(&file_path, b"test content").unwrap();

        // Test size verification
        assert!(verify_file_integrity(&file_path, None, Some(12), HashAlgorithm::Sha256).unwrap());
        assert!(!verify_file_integrity(&file_path, None, Some(10), HashAlgorithm::Sha256).unwrap());

        // Test with non-existent file
        let missing = temp_dir.path().join("missing.txt");
        assert!(!verify_file_integrity(&missing, None, None, HashAlgorithm::Sha256).unwrap());
    }

    #[test]
    fn test_calculate_file_hash() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Create test file
        fs::write(&file_path, b"test content").unwrap();

        // Calculate hash
        let hash = calculate_file_hash(&file_path, HashAlgorithm::Sha256).unwrap();
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_validate_file_path() {
        // Safe paths
        assert!(validate_file_path(Path::new("safe/path.txt"), None).is_ok());
        assert!(validate_file_path(Path::new("file.txt"), Some(Path::new("/tmp"))).is_ok());

        // Unsafe paths
        assert!(validate_file_path(Path::new("../etc/passwd"), None).is_err());
        assert!(validate_file_path(Path::new("path\0file"), None).is_err());
        assert!(validate_file_path(Path::new("/etc/passwd"), Some(Path::new("/tmp"))).is_err());
    }

    #[test]
    fn test_validate_archive_format() {
        let tar_file = Path::new("file.tar.gz");
        let zip_file = Path::new("file.zip");
        let exe_file = Path::new("file.exe");

        let allowed = &["tar.gz", "zip"];

        assert!(validate_archive_format(tar_file, allowed).is_ok());
        assert!(validate_archive_format(zip_file, allowed).is_ok());
        assert!(validate_archive_format(exe_file, allowed).is_err());

        // Allow all formats
        assert!(validate_archive_format(exe_file, &[]).is_ok());
    }

    #[test]
    fn test_validate_content_type() {
        let allowed = &["application/json", "text/plain"];

        assert!(validate_content_type("application/json", allowed).is_ok());
        assert!(validate_content_type("application/json; charset=utf-8", allowed).is_ok());
        assert!(validate_content_type("text/plain", allowed).is_ok());
        assert!(validate_content_type("application/octet-stream", allowed).is_err());

        // Allow all types
        assert!(validate_content_type("application/octet-stream", &[]).is_ok());
    }

    #[test]
    fn test_validate_file_size() {
        assert!(validate_file_size(100, Some(50), Some(200)).is_ok());
        assert!(validate_file_size(25, Some(50), Some(200)).is_err());
        assert!(validate_file_size(250, Some(50), Some(200)).is_err());
        assert!(validate_file_size(100, None, None).is_ok());
    }

    #[test]
    fn test_validate_byte_range() {
        assert!(validate_byte_range(0, 99, 100).is_ok());
        assert!(validate_byte_range(50, 99, 100).is_ok());
        assert!(validate_byte_range(50, 25, 100).is_err()); // start > end
        assert!(validate_byte_range(0, 100, 100).is_err()); // end >= size
        assert!(validate_byte_range(100, 150, 100).is_err()); // start >= size
    }

    #[test]
    fn test_verify_download_compatibility() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        // Create test file
        fs::write(&file_path, b"test content").unwrap();

        // Test backward compatibility
        assert!(verify_download(&file_path, None, Some(12)).unwrap());
        assert!(!verify_download(&file_path, None, Some(10)).unwrap());

        // Non-existent file
        let missing = temp_dir.path().join("missing.txt");
        assert!(!verify_download(&missing, None, None).unwrap());
    }
}
