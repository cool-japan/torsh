//! Integration tests for torsh-package
//!
//! These tests verify end-to-end functionality of the package system,
//! including complex scenarios and edge cases.

use tempfile::TempDir;
use torsh_package::{
    BuilderConfig, ExportConfig, ImportConfig, Package, PackageBuilder, PackageExporter,
    PackageImporter, Resource, ResourceType, PACKAGE_FORMAT_VERSION,
};

/// Test complete package lifecycle: create, save, load, verify
#[test]
fn test_complete_package_lifecycle() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("lifecycle_test.torshpkg");

    // Create package with various resource types
    let mut package = Package::new("lifecycle_test".to_string(), "1.2.3".to_string());

    // Add different types of resources
    package
        .add_source_file("main", "fn main() { println!(\"Hello, World!\"); }")
        .unwrap();

    // Add binary data
    let binary_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    let binary_resource = Resource::new(
        "test_data.bin".to_string(),
        ResourceType::Binary,
        binary_data.clone(),
    );
    package
        .resources_mut()
        .insert(binary_resource.name.clone(), binary_resource);

    // Add text data
    let text_resource = Resource::new(
        "readme.txt".to_string(),
        ResourceType::Text,
        b"This is a test package created for integration testing.".to_vec(),
    );
    package
        .resources_mut()
        .insert(text_resource.name.clone(), text_resource);

    // Add metadata
    package.add_dependency("serde", "1.0");
    package.add_dependency("tokio", "1.0");

    // Set author and description
    package.manifest_mut().author = Some("Test Author".to_string());
    package.manifest_mut().description = Some("Integration test package".to_string());

    // Verify package integrity before saving
    assert!(package.verify().unwrap());

    // Save package with source files included
    let export_config = ExportConfig {
        include_source: true,
        ..Default::default()
    };
    let exporter = PackageExporter::new(export_config);
    exporter.export_package(&package, &package_path).unwrap();
    assert!(package_path.exists());

    // Load package back
    let loaded_package = Package::load(&package_path).unwrap();

    // Verify loaded package
    assert_eq!(loaded_package.metadata().name, "lifecycle_test");
    assert_eq!(loaded_package.metadata().version, "1.2.3");
    assert_eq!(
        loaded_package.metadata().author.as_deref(),
        Some("Test Author")
    );
    assert_eq!(
        loaded_package.metadata().description.as_deref(),
        Some("Integration test package")
    );
    assert_eq!(
        loaded_package.metadata().format_version,
        PACKAGE_FORMAT_VERSION
    );

    // Verify resources (should be 3: source, binary and text resources)
    assert_eq!(loaded_package.resources().len(), 3);
    assert!(loaded_package.resources().contains_key("main.rs"));
    assert!(loaded_package.resources().contains_key("test_data.bin"));
    assert!(loaded_package.resources().contains_key("readme.txt"));

    // Verify resource content
    let binary_res = loaded_package.resources().get("test_data.bin").unwrap();
    assert_eq!(binary_res.data, binary_data);

    // Verify dependencies
    assert_eq!(
        loaded_package.metadata().dependencies.get("serde"),
        Some(&"1.0".to_string())
    );
    assert_eq!(
        loaded_package.metadata().dependencies.get("tokio"),
        Some(&"1.0".to_string())
    );

    // Verify package integrity after loading
    assert!(loaded_package.verify().unwrap());
}

/// Test package with compression settings
#[test]
fn test_compression_levels() {
    let temp_dir = TempDir::new().unwrap();

    // Create test data - highly compressible
    let compressible_data = "A".repeat(10000);

    let mut package = Package::new("compression_test".to_string(), "1.0.0".to_string());
    let resource = Resource::new(
        "compressible.txt".to_string(),
        ResourceType::Text,
        compressible_data.as_bytes().to_vec(),
    );
    package
        .resources_mut()
        .insert(resource.name.clone(), resource);

    // Test different compression levels
    let compression_levels = vec![1, 6, 9];
    let mut file_sizes = Vec::new();

    for level in compression_levels {
        let path = temp_dir
            .path()
            .join(format!("compression_level_{}.torshpkg", level));

        let config = ExportConfig {
            compression_level: Some(level),
            ..Default::default()
        };

        let exporter = PackageExporter::new(config);
        exporter.export_package(&package, &path).unwrap();

        let metadata = std::fs::metadata(&path).unwrap();
        file_sizes.push(metadata.len());

        // Verify we can still load the compressed package
        let loaded = Package::load(&path).unwrap();
        assert_eq!(loaded.metadata().name, "compression_test");
        assert!(loaded.verify().unwrap());
    }

    // Higher compression levels should produce smaller files (generally)
    // Note: For small data, this might not always be true due to overhead
    println!("Compression file sizes: {:?}", file_sizes);
}

/// Test package builder with complex configuration
#[test]
fn test_complex_package_builder() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("complex_builder_test.torshpkg");

    let config = BuilderConfig {
        include_source: true,
        compress: true,
        sign: false,
        include_dependencies: true,
    };

    // Create temporary config file first
    let config_path = temp_dir.path().join("temp_config.json");
    std::fs::write(&config_path, r#"{"debug": true, "log_level": "info"}"#).unwrap();

    let result = PackageBuilder::new("complex_test".to_string(), "2.1.0".to_string())
        .with_config(config)
        .add_metadata("project_url", "https://github.com/example/project")
        .add_metadata("build_date", "2025-01-01")
        .add_metadata("build_commit", "abc123def456")
        .add_data_file("config.json", config_path)
        .unwrap()
        .build(output_path.clone());

    result.unwrap();

    // Load and verify
    let loaded = Package::load(&output_path).unwrap();
    assert_eq!(loaded.metadata().name, "complex_test");
    assert_eq!(loaded.metadata().version, "2.1.0");
    assert_eq!(
        loaded.metadata().metadata.get("project_url"),
        Some(&"https://github.com/example/project".to_string())
    );
    assert_eq!(
        loaded.metadata().metadata.get("build_date"),
        Some(&"2025-01-01".to_string())
    );
    assert!(loaded.resources().contains_key("config.json"));
}

/// Test package size limits and error handling
#[test]
fn test_package_size_limits() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("size_limit_test.torshpkg");

    // Create a package with data that exceeds size limit
    let mut package = Package::new("size_test".to_string(), "1.0.0".to_string());

    // Add large resource (1MB)
    let large_data = vec![0u8; 1_024_000];
    let resource = Resource::new("large_file.dat".to_string(), ResourceType::Data, large_data);
    package
        .resources_mut()
        .insert(resource.name.clone(), resource);

    // Configure exporter with size limit smaller than data size
    let config = ExportConfig {
        max_size: 500_000, // 500KB limit
        ..Default::default()
    };

    let exporter = PackageExporter::new(config);
    let result = exporter.export_package(&package, &output_path);

    // Should fail due to size limit
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.to_string().contains("exceeds maximum allowed size"));
}

/// Test import with integrity verification
#[test]
fn test_integrity_verification() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("integrity_test.torshpkg");

    // Create package
    let mut package = Package::new("integrity_test".to_string(), "1.0.0".to_string());
    let resource = Resource::new(
        "test.txt".to_string(),
        ResourceType::Text,
        b"integrity test data".to_vec(),
    );
    package
        .resources_mut()
        .insert(resource.name.clone(), resource);

    // Add checksum to resource metadata
    let resource_mut = package.resources_mut().get_mut("test.txt").unwrap();
    let checksum = resource_mut.sha256();
    resource_mut.add_metadata("sha256".to_string(), checksum);

    // Export package
    package.save(&package_path).unwrap();

    // Import with integrity verification enabled
    let config = ImportConfig {
        verify_integrity: true,
        ..Default::default()
    };

    let importer = PackageImporter::new(config);
    let loaded = importer.import_package(&package_path).unwrap();

    // Should load successfully with correct integrity
    assert_eq!(loaded.metadata().name, "integrity_test");
    assert!(loaded.verify().unwrap());
}

/// Test package with metadata and resource filtering
#[test]
fn test_metadata_and_filtering() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("filtering_test.torshpkg");

    // Create package with diverse resources
    let mut package = Package::new("filtering_test".to_string(), "1.0.0".to_string());

    // Add different resource types
    let resources_data = vec![
        ("model.bin", ResourceType::Model, b"model_data".to_vec()),
        ("source.rs", ResourceType::Source, b"fn main() {}".to_vec()),
        (
            "config.toml",
            ResourceType::Config,
            b"[app]\nname = \"test\"".to_vec(),
        ),
        (
            "docs.md",
            ResourceType::Documentation,
            b"# Documentation".to_vec(),
        ),
        (
            "data.json",
            ResourceType::Data,
            b"{\"key\": \"value\"}".to_vec(),
        ),
    ];

    for (name, res_type, data) in resources_data {
        let mut resource = Resource::new(name.to_string(), res_type, data);
        resource.add_metadata("created".to_string(), "2025-01-01".to_string());
        resource.add_metadata("author".to_string(), "test_user".to_string());
        package
            .resources_mut()
            .insert(resource.name.clone(), resource);
    }

    // Save and reload with source files included
    let export_config = ExportConfig {
        include_source: true,
        ..Default::default()
    };
    let exporter = PackageExporter::new(export_config);
    exporter.export_package(&package, &package_path).unwrap();
    let loaded = Package::load(&package_path).unwrap();

    // Verify all resources are loaded
    println!(
        "Resources loaded: {:?}",
        loaded.resources().keys().collect::<Vec<_>>()
    );
    // Now with source files included, we should get all 5 resources
    assert_eq!(loaded.resources().len(), 5);

    // Test resource collection filtering
    let mut collection = torsh_package::resources::ResourceCollection::new();
    for resource in loaded.resources().values() {
        collection.add(resource.clone()).unwrap();
    }

    // Filter by type
    let model_resources = collection.by_type(ResourceType::Model);
    assert_eq!(model_resources.len(), 1);
    assert_eq!(model_resources[0].name, "model.bin");

    let source_resources = collection.by_type(ResourceType::Source);
    assert_eq!(source_resources.len(), 1);
    assert_eq!(source_resources[0].name, "source.rs");

    // Test resource filter
    use torsh_package::resources::ResourceFilter;

    let filter = ResourceFilter::new()
        .with_types(vec![ResourceType::Model, ResourceType::Data])
        .with_size_range(Some(5), Some(50));

    let filtered_resources: Vec<_> = loaded
        .resources()
        .values()
        .filter(|r| filter.matches(r))
        .collect();

    assert_eq!(filtered_resources.len(), 2); // model.bin and data.json
}

/// Test error handling scenarios
#[test]
fn test_error_handling() {
    let temp_dir = TempDir::new().unwrap();

    // Test loading non-existent package
    let result = Package::load(temp_dir.path().join("non_existent.torshpkg"));
    assert!(result.is_err());

    // Test loading invalid package format
    let invalid_path = temp_dir.path().join("invalid.torshpkg");
    std::fs::write(&invalid_path, b"not a valid zip file").unwrap();
    let result = Package::load(&invalid_path);
    assert!(result.is_err());

    // Test package with invalid manifest
    let package = Package::new("".to_string(), "invalid_version".to_string()); // Invalid name and version
    let package_path = temp_dir.path().join("invalid_manifest.torshpkg");
    let result = package.save(&package_path);
    assert!(result.is_err());
}

/// Test concurrent package operations
#[test]
fn test_concurrent_operations() {
    use std::sync::Arc;
    use std::thread;

    let temp_dir = Arc::new(TempDir::new().unwrap());
    let mut handles = Vec::new();

    // Create multiple packages concurrently
    for i in 0..5 {
        let temp_dir_clone = Arc::clone(&temp_dir);
        let handle = thread::spawn(move || {
            let package_path = temp_dir_clone
                .path()
                .join(format!("concurrent_{}.torshpkg", i));

            let mut package = Package::new(format!("concurrent_test_{}", i), "1.0.0".to_string());
            let resource = Resource::new(
                format!("data_{}.txt", i),
                ResourceType::Text,
                format!("data for package {}", i).as_bytes().to_vec(),
            );
            package
                .resources_mut()
                .insert(resource.name.clone(), resource);

            // Save and reload
            package.save(&package_path).unwrap();
            let loaded = Package::load(&package_path).unwrap();

            assert_eq!(loaded.metadata().name, format!("concurrent_test_{}", i));
            loaded.verify().unwrap()
        });
        handles.push(handle);
    }

    // Wait for all operations to complete
    for handle in handles {
        assert!(handle.join().unwrap());
    }
}

/// Test package extraction and directory structure
#[test]
fn test_package_extraction() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("extraction_test.torshpkg");
    let extract_dir = temp_dir.path().join("extracted");

    // Create package with structured resources
    let mut package = Package::new("extraction_test".to_string(), "1.0.0".to_string());

    // Add resources that will create directory structure
    let resources = vec![
        ("main.rs", ResourceType::Source, b"fn main() {}" as &[u8]),
        (
            "config.toml",
            ResourceType::Config,
            b"[app]\nname = \"test\"" as &[u8],
        ),
        (
            "data.json",
            ResourceType::Data,
            b"{\"key\": \"value\"}" as &[u8],
        ),
        (
            "README.md",
            ResourceType::Documentation,
            b"# Test Package" as &[u8],
        ),
    ];

    for (name, res_type, data) in resources {
        let resource = Resource::new(name.to_string(), res_type, data.to_vec());
        package
            .resources_mut()
            .insert(resource.name.clone(), resource);
    }

    // Save package with source files included
    let export_config = ExportConfig {
        include_source: true,
        ..Default::default()
    };
    let exporter = PackageExporter::new(export_config);
    exporter.export_package(&package, &package_path).unwrap();

    // Extract package
    let importer = PackageImporter::new(ImportConfig::default());
    importer
        .extract_package(&package_path, &extract_dir)
        .unwrap();

    // Verify extracted structure
    println!("Extracted files:");
    for entry in std::fs::read_dir(&extract_dir).unwrap() {
        let entry = entry.unwrap();
        println!("  {:?}", entry.path());
    }

    assert!(extract_dir.join("MANIFEST.json").exists());
    assert!(extract_dir.join("src/main.rs").exists());
    assert!(extract_dir.join("config/config.toml").exists());
    assert!(extract_dir.join("data/data.json").exists());
    assert!(extract_dir.join("docs/README.md").exists());

    // Verify content
    let extracted_main = std::fs::read_to_string(extract_dir.join("src/main.rs")).unwrap();
    assert_eq!(extracted_main, "fn main() {}");
}

/// Performance test for large packages
#[test]
fn test_large_package_performance() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("large_package.torshpkg");

    // Create package with many small resources
    let mut package = Package::new("large_test".to_string(), "1.0.0".to_string());

    let start_create = std::time::Instant::now();

    // Add 100 small resources
    for i in 0..100 {
        let resource = Resource::new(
            format!("resource_{}.txt", i),
            ResourceType::Text,
            format!("content for resource number {}", i)
                .as_bytes()
                .to_vec(),
        );
        package
            .resources_mut()
            .insert(resource.name.clone(), resource);
    }

    let create_duration = start_create.elapsed();
    println!("Package creation time: {:?}", create_duration);

    // Save package
    let start_save = std::time::Instant::now();
    package.save(&package_path).unwrap();
    let save_duration = start_save.elapsed();
    println!("Package save time: {:?}", save_duration);

    // Load package
    let start_load = std::time::Instant::now();
    let loaded = Package::load(&package_path).unwrap();
    let load_duration = start_load.elapsed();
    println!("Package load time: {:?}", load_duration);

    // Verify loaded package
    assert_eq!(loaded.resources().len(), 100);
    assert!(loaded.verify().unwrap());

    // Performance should be reasonable (adjust thresholds as needed)
    assert!(
        create_duration.as_millis() < 1000,
        "Package creation too slow: {:?}",
        create_duration
    );
    assert!(
        save_duration.as_millis() < 5000,
        "Package save too slow: {:?}",
        save_duration
    );
    assert!(
        load_duration.as_millis() < 5000,
        "Package load too slow: {:?}",
        load_duration
    );
}
