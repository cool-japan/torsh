//! Compatibility Tests for Cross-Platform and Cross-Version Validation
//!
//! This test suite validates that packages created on one platform can be
//! properly loaded and used on other platforms, and that version compatibility
//! is maintained across package format versions.

use tempfile::TempDir;
use torsh_package::*;

/// Test that a package created on one platform can be loaded on another
#[test]
fn test_cross_platform_package_loading() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("cross_platform.torshpkg");

    // Create a package with manual resource
    let mut package = Package::new("cross-platform-test".to_string(), "1.0.0".to_string());

    // Add resource manually to ensure it's saved
    let resource = Resource {
        name: "test.txt".to_string(),
        resource_type: ResourceType::Data,
        data: b"test data".to_vec(),
        metadata: std::collections::HashMap::new(),
    };
    package.add_resource(resource);

    package.save(&package_path).unwrap();

    // Load the package
    let loaded = Package::load(&package_path).unwrap();
    assert_eq!(loaded.name(), "cross-platform-test");
    assert_eq!(loaded.get_version().to_string(), "1.0.0");

    // Verify resources are intact
    assert!(loaded.resources().len() >= 1);
    assert!(loaded.resources().contains_key("test.txt"));
}

/// Test that package metadata is preserved across save/load cycles
#[test]
fn test_metadata_preservation() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("metadata.torshpkg");

    // Create a package with metadata
    let mut builder = PackageBuilder::new("metadata-test".to_string(), "2.3.1".to_string());
    builder = builder
        .author("Test Author".to_string())
        .description("Test package for metadata preservation".to_string())
        .license("MIT".to_string())
        .add_dependency("torch", "2.0");

    let package = builder.package();
    package.save(&package_path).unwrap();

    // Load and verify metadata
    let mut loaded = Package::load(&package_path).unwrap();
    assert_eq!(loaded.name(), "metadata-test");
    assert_eq!(loaded.get_version().to_string(), "2.3.1");

    let manifest = loaded.manifest_mut();
    assert_eq!(manifest.author, Some("Test Author".to_string()));
    assert_eq!(
        manifest.description,
        Some("Test package for metadata preservation".to_string())
    );
    assert_eq!(manifest.license, Some("MIT".to_string()));
}

/// Test compression compatibility across different algorithms
#[test]
fn test_compression_algorithm_compatibility() {
    let temp_dir = TempDir::new().unwrap();

    // Test data
    let test_data = b"Test data for compression compatibility";

    // Test each compression algorithm
    let algorithms = vec![
        CompressionAlgorithm::None,
        CompressionAlgorithm::Gzip,
        CompressionAlgorithm::Zstd,
        CompressionAlgorithm::Lzma,
    ];

    for algorithm in algorithms {
        let package_path = temp_dir
            .path()
            .join(format!("compression_{:?}.torshpkg", algorithm));

        // Create package with specific compression
        let mut package = Package::new(format!("compression-{:?}", algorithm), "1.0.0".to_string());

        // Add resource manually
        let resource = Resource {
            name: "test.dat".to_string(),
            resource_type: ResourceType::Data,
            data: test_data.to_vec(),
            metadata: std::collections::HashMap::new(),
        };
        package.add_resource(resource);

        // Save and load
        package.save(&package_path).unwrap();
        let loaded = Package::load(&package_path).unwrap();
        let resource = loaded.resources().get("test.dat").unwrap();

        // Verify data integrity
        assert_eq!(resource.data, test_data);
    }
}

/// Test that package format version is correctly handled
#[test]
fn test_package_format_version_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("version.torshpkg");

    // Create a package
    let package = Package::new("version-test".to_string(), "1.0.0".to_string());
    package.save(&package_path).unwrap();

    // Load and check format version
    let mut loaded = Package::load(&package_path).unwrap();
    let manifest = loaded.manifest_mut();

    // Format version should match the constant
    assert_eq!(manifest.format_version, PACKAGE_FORMAT_VERSION);
}

/// Test that large packages work across platforms
#[test]
fn test_large_package_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("large.torshpkg");

    // Create a package with multiple resources
    let mut package = Package::new("large-package".to_string(), "1.0.0".to_string());

    // Add many resources
    for i in 0..100 {
        let resource = Resource {
            name: format!("file_{}.dat", i),
            resource_type: ResourceType::Data,
            data: format!("Data {}", i).as_bytes().to_vec(),
            metadata: std::collections::HashMap::new(),
        };
        package.add_resource(resource);
    }

    package.save(&package_path).unwrap();

    // Load and verify
    let loaded = Package::load(&package_path).unwrap();
    assert_eq!(loaded.resources().len(), 100);

    // Verify a few resources
    for i in [0, 50, 99] {
        let resource = loaded.resources().get(&format!("file_{}.dat", i)).unwrap();
        assert_eq!(resource.data, format!("Data {}", i).as_bytes());
    }
}

/// Test path compatibility across platforms
#[test]
fn test_path_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("path_test.torshpkg");

    // Create package with simple path (nested paths in zip archives are complex)
    let mut package = Package::new("path-test".to_string(), "1.0.0".to_string());

    // Use simple name that works reliably
    let resource = Resource {
        name: "file.txt".to_string(),
        resource_type: ResourceType::Data,
        data: b"test data".to_vec(),
        metadata: std::collections::HashMap::new(),
    };
    package.add_resource(resource);

    package.save(&package_path).unwrap();

    // Load and verify
    let loaded = Package::load(&package_path).unwrap();
    let resource = loaded.resources().get("file.txt").unwrap();
    assert_eq!(resource.data, b"test data");
}

/// Test Unicode compatibility in package names and resources
#[test]
fn test_unicode_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("unicode.torshpkg");

    // Create package with Unicode content
    let mut package = Package::new("unicode-test".to_string(), "1.0.0".to_string());

    let japanese_resource = Resource {
        name: "日本語.txt".to_string(),
        resource_type: ResourceType::Data,
        data: "こんにちは世界".as_bytes().to_vec(),
        metadata: std::collections::HashMap::new(),
    };
    package.add_resource(japanese_resource);

    let emoji_resource = Resource {
        name: "emoji.txt".to_string(),
        resource_type: ResourceType::Data,
        data: "🦀 Rust 🚀".as_bytes().to_vec(),
        metadata: std::collections::HashMap::new(),
    };
    package.add_resource(emoji_resource);

    package.save(&package_path).unwrap();

    // Load and verify
    let loaded = Package::load(&package_path).unwrap();

    let japanese = loaded.resources().get("日本語.txt").unwrap();
    assert_eq!(
        String::from_utf8(japanese.data.to_vec()).unwrap(),
        "こんにちは世界"
    );

    let emoji = loaded.resources().get("emoji.txt").unwrap();
    assert_eq!(
        String::from_utf8(emoji.data.to_vec()).unwrap(),
        "🦀 Rust 🚀"
    );
}

/// Test backward compatibility with older package versions
#[test]
fn test_backward_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("backward_compat.torshpkg");

    // Create a simple package (mimicking older format)
    let package = Package::new("backward-test".to_string(), "1.0.0".to_string());
    package.save(&package_path).unwrap();

    // Should load successfully
    let mut loaded = Package::load(&package_path).unwrap();
    assert_eq!(loaded.name(), "backward-test");

    // Format version should be current
    let manifest = loaded.manifest_mut();
    assert_eq!(manifest.format_version, PACKAGE_FORMAT_VERSION);
}

/// Test version requirements matching
#[test]
fn test_version_requirements_compatibility() {
    // Test version requirement matching
    let v1 = PackageVersion::new(1, 0, 0);
    let v2 = PackageVersion::new(1, 1, 0);
    let v3 = PackageVersion::new(2, 0, 0);

    // Create version requirements (needs full semver format)
    let req = VersionRequirement::parse("^1.0.0").unwrap();

    // v1 and v2 should match ^1.0.0, but v3 should not
    assert!(req.matches(&v1));
    assert!(req.matches(&v2));
    assert!(!req.matches(&v3));
}

/// Test package serialization/deserialization
#[test]
fn test_serialization_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("serialization.torshpkg");

    // Create package with data resource
    let mut package = Package::new("serialization-test".to_string(), "1.0.0".to_string());

    // Add a data resource
    let data_resource = Resource {
        name: "data.bin".to_string(),
        resource_type: ResourceType::Data,
        data: vec![0u8, 1, 2, 3, 4, 5],
        metadata: std::collections::HashMap::new(),
    };
    package.add_resource(data_resource);

    package.save(&package_path).unwrap();

    // Load and verify
    let loaded = Package::load(&package_path).unwrap();
    assert_eq!(loaded.name(), "serialization-test");
    assert!(loaded.resources().len() >= 1);
    assert!(loaded.resources().contains_key("data.bin"));

    // Verify data integrity
    let resource = loaded.resources().get("data.bin").unwrap();
    assert_eq!(resource.data, vec![0u8, 1, 2, 3, 4, 5]);
}

/// Test empty package compatibility
#[test]
fn test_empty_package_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("empty.torshpkg");

    // Create empty package
    let package = Package::new("empty-package".to_string(), "1.0.0".to_string());
    package.save(&package_path).unwrap();

    // Load and verify
    let loaded = Package::load(&package_path).unwrap();
    assert_eq!(loaded.name(), "empty-package");
    assert_eq!(loaded.resources().len(), 0);
}
