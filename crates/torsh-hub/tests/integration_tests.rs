//! Integration tests for torsh-hub
//!
//! These tests verify end-to-end workflows and integration between modules.

use torsh_hub::*;

/// Test utility functions
#[test]
fn test_utils_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Test format_size
    assert_eq!(format_size(1024), "1.00 KB");
    assert_eq!(format_size(1024 * 1024), "1.00 MB");
    assert_eq!(format_size(1024 * 1024 * 1024), "1.00 GB");

    // Test sanitize_model_name
    assert_eq!(sanitize_model_name("My Model"), "my-model");
    assert_eq!(sanitize_model_name("BERT (base)"), "bert-base");

    // Test parse_repo_string
    let (org, model, tag) = parse_repo_string("huggingface/bert-base")?;
    assert_eq!(org, "huggingface");
    assert_eq!(model, "bert-base");
    assert_eq!(tag, None);

    let (org, model, tag) = parse_repo_string("openai/gpt2:v1.0")?;
    assert_eq!(org, "openai");
    assert_eq!(model, "gpt2");
    assert_eq!(tag, Some("v1.0".to_string()));

    // Test version operations
    assert!(validate_semver("1.2.3").is_ok());
    assert!(validate_semver("invalid").is_err());

    use std::cmp::Ordering;
    assert_eq!(compare_versions("1.0.0", "2.0.0")?, Ordering::Less);
    assert_eq!(compare_versions("2.0.0", "1.0.0")?, Ordering::Greater);
    assert_eq!(compare_versions("1.0.0", "1.0.0")?, Ordering::Equal);

    // Test is_safe_path
    assert!(is_safe_path(std::path::Path::new("models/bert.onnx")));
    assert!(!is_safe_path(std::path::Path::new("../etc/passwd")));

    // Test format_parameter_count
    assert_eq!(format_parameter_count(1_500_000), "1.50M");
    assert_eq!(format_parameter_count(7_000_000_000), "7.00B");

    Ok(())
}

/// Test URL validation
#[test]
fn test_url_validation_integration() -> Result<(), Box<dyn std::error::Error>> {
    use download::validate_url;

    // Valid URLs
    assert!(validate_url("http://example.com/model.onnx").is_ok());
    assert!(validate_url("https://example.com/model.safetensors").is_ok());
    assert!(validate_url("ftp://example.com/model.pt").is_ok());

    // Invalid URLs
    assert!(validate_url("").is_err());
    assert!(validate_url("not-a-url").is_err());
    assert!(validate_url("file:///local/path").is_err());
    assert!(validate_url("https://example.com/model name.onnx").is_err()); // Spaces

    Ok(())
}

/// Test version comparison edge cases
#[test]
fn test_version_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
    use model_info::Version;

    let v1_0_0 = Version::new(1, 0, 0);
    let v1_0_1 = Version::new(1, 0, 1);
    let v1_1_0 = Version::new(1, 1, 0);
    let v2_0_0 = Version::new(2, 0, 0);

    // Test ordering
    assert!(v1_0_0 < v1_0_1);
    assert!(v1_0_1 < v1_1_0);
    assert!(v1_1_0 < v2_0_0);

    // Test equality
    assert_eq!(v1_0_0, Version::new(1, 0, 0));

    // Test string representation
    assert_eq!(v1_0_0.to_string(), "1.0.0");
    assert_eq!(v2_0_0.to_string(), "2.0.0");

    Ok(())
}

/// Test bandwidth formatting
#[test]
fn test_bandwidth_formatting() {
    use bandwidth::format_bytes;

    // Test that format_bytes returns a string with size unit
    assert!(format_bytes(1024).contains("KB"));
    assert!(format_bytes(1024 * 1024).contains("MB"));
    assert!(format_bytes(1024 * 1024 * 1024).contains("GB"));
    assert!(format_bytes(512).contains("B"));
}

/// Test concurrent operations
#[test]
fn test_concurrent_operations() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    // Spawn multiple threads that increment counter
    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                let mut num = counter_clone.lock().unwrap();
                *num += 1;
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let final_count = *counter.lock().unwrap();
    assert_eq!(final_count, 1000, "Concurrent increments should total 1000");

    Ok(())
}

/// Test model category parsing
#[test]
fn test_model_category_parsing() -> Result<(), Box<dyn std::error::Error>> {
    use std::str::FromStr;

    // Test various category names
    assert!(ModelCategory::from_str("vision").is_ok());
    assert!(ModelCategory::from_str("nlp").is_ok());
    assert!(ModelCategory::from_str("audio").is_ok());
    assert!(ModelCategory::from_str("multimodal").is_ok());

    // Case insensitive
    assert!(ModelCategory::from_str("VISION").is_ok());
    assert!(ModelCategory::from_str("Nlp").is_ok());

    Ok(())
}

/// Test string operations performance
#[test]
fn test_string_operations_integration() {
    // Test string transformations
    let model_name = "bert-base-uncased";
    let sanitized = sanitize_model_name(model_name);
    assert_eq!(sanitized, "bert-base-uncased");

    let complex_name = "BERT (Base) - Uncased v2.0";
    let sanitized = sanitize_model_name(complex_name);
    assert_eq!(sanitized, "bert-base-uncased-v2-0");

    // Test path safety
    assert!(is_safe_path(std::path::Path::new("safe/path/model.onnx")));
    assert!(!is_safe_path(std::path::Path::new("../../../etc/passwd")));
}

/// Test extension extraction
#[test]
fn test_extension_extraction() {
    assert_eq!(extract_extension("model.onnx"), Some("onnx"));
    assert_eq!(extract_extension("model.safetensors"), Some("safetensors"));
    assert_eq!(extract_extension("model.pt"), Some("pt"));
    assert_eq!(extract_extension("model"), None);

    // Test format support
    assert!(is_supported_model_format("onnx"));
    assert!(is_supported_model_format("safetensors"));
    assert!(is_supported_model_format("pt"));
    assert!(!is_supported_model_format("txt"));
}

/// Test parameter estimation
#[test]
fn test_parameter_estimation() {
    // 100 MB file = ~25M parameters (assuming float32)
    let size = 100 * 1024 * 1024;
    let params = estimate_parameters_from_size(size);
    assert_eq!(params, 26_214_400);

    // Format parameter count
    assert_eq!(format_parameter_count(params), "26.21M");
}

/// Test cache directory creation
#[test]
fn test_cache_directory() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = "test-model-12345";
    let cache_dir = get_model_cache_dir(model_name)?;

    // Verify directory exists
    assert!(cache_dir.exists());

    // Verify path contains sanitized name
    assert!(cache_dir.to_str().unwrap().contains("test-model-12345"));

    // Cleanup
    std::fs::remove_dir_all(cache_dir.parent().unwrap())?;

    Ok(())
}

/// Test version validation
#[test]
fn test_version_validation() {
    assert!(validate_semver("1.0.0").is_ok());
    assert!(validate_semver("1.2.3").is_ok());
    assert!(validate_semver("0.1.0").is_ok());
    assert!(validate_semver("2.0.0-alpha").is_ok());
    assert!(validate_semver("1.0.0-rc.1").is_ok());

    // Invalid versions
    assert!(validate_semver("1").is_err());
    assert!(validate_semver("1.0").is_ok()); // 2-part versions are valid
    assert!(validate_semver("abc").is_err());
    assert!(validate_semver("1.a.0").is_err());
}
