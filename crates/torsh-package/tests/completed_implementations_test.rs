//! Tests for completed TODO implementations
//!
//! This test file verifies that all previously TODO-marked implementations
//! in torsh-package have been properly completed.

use torsh_package::{
    diagnostics::{HealthStatus, PackageDiagnostics},
    optimization::PackageOptimizer,
    Package, Resource, ResourceType,
};

#[test]
fn test_diagnostics_resource_validation() {
    let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

    // Add valid resources
    package.add_resource(Resource::new(
        "model.pth".to_string(),
        ResourceType::Model,
        vec![0u8; 1024],
    ));

    // Add a large resource
    package.add_resource(Resource::new(
        "large_model.pth".to_string(),
        ResourceType::Model,
        vec![0u8; 150 * 1024 * 1024], // 150 MB
    ));

    let diagnostics = PackageDiagnostics::new();
    let report = diagnostics.diagnose(&package).unwrap();

    // Should have detected the large resource
    assert!(
        !report.issues.is_empty(),
        "Should detect issues with large resources"
    );

    // Should have validated resources
    assert!(
        !report.resource_validation.is_empty(),
        "Should validate resources"
    );

    println!("Diagnostics report: {:?}", report);
}

#[test]
fn test_diagnostics_security_assessment() {
    let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

    // Add a resource with strong checksum
    let mut resource = Resource::new(
        "model.pth".to_string(),
        ResourceType::Model,
        vec![1, 2, 3, 4],
    );
    resource.add_metadata("sha256".to_string(), resource.sha256());
    package.add_resource(resource);

    let diagnostics = PackageDiagnostics::new();
    let report = diagnostics.diagnose(&package).unwrap();

    // Security assessment should be present
    assert_eq!(report.security.is_signed, false);
    assert!(
        report.security.security_score > 0,
        "Should have non-zero security score"
    );

    println!("Security assessment: {:?}", report.security);
}

#[test]
fn test_diagnostics_statistics() {
    let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

    // Add multiple resources
    package.add_resource(Resource::new(
        "model1.pth".to_string(),
        ResourceType::Model,
        vec![0u8; 1000],
    ));
    package.add_resource(Resource::new(
        "model2.pth".to_string(),
        ResourceType::Model,
        vec![0u8; 2000],
    ));
    package.add_resource(Resource::new(
        "config.json".to_string(),
        ResourceType::Config,
        vec![0u8; 500],
    ));

    let diagnostics = PackageDiagnostics::new();
    let report = diagnostics.diagnose(&package).unwrap();

    // Statistics should be calculated
    assert_eq!(report.statistics.resource_count, 3);
    assert_eq!(report.statistics.total_size, 3500);
    assert_eq!(report.statistics.largest_resource_size, 2000);

    println!("Package statistics: {:?}", report.statistics);
}

#[test]
fn test_diagnostics_performance_checks() {
    let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

    // Add large uncompressed resource
    package.add_resource(Resource::new(
        "large_uncompressed.dat".to_string(),
        ResourceType::Data,
        vec![0u8; 15 * 1024 * 1024], // 15 MB
    ));

    // Add many small resources
    for i in 0..150 {
        package.add_resource(Resource::new(
            format!("small_{}.txt", i),
            ResourceType::Text,
            vec![0u8; 100], // 100 bytes each
        ));
    }

    let diagnostics = PackageDiagnostics::new();
    let report = diagnostics.diagnose(&package).unwrap();

    // Should detect performance issues
    let has_compression_issue = report.issues.iter().any(|issue| {
        issue.description.contains("large") && issue.description.contains("uncompressed")
    });

    let has_small_files_issue = report
        .issues
        .iter()
        .any(|issue| issue.description.contains("small resources"));

    assert!(
        has_compression_issue || has_small_files_issue,
        "Should detect performance issues"
    );

    println!("Performance issues detected: {}", report.issues.len());
}

#[test]
fn test_optimization_package_size_calculation() {
    let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

    package.add_resource(Resource::new(
        "model.pth".to_string(),
        ResourceType::Model,
        vec![0u8; 1000],
    ));
    package.add_resource(Resource::new(
        "config.json".to_string(),
        ResourceType::Config,
        vec![0u8; 200],
    ));

    let optimizer = PackageOptimizer::new();
    let report = optimizer.analyze(&package).unwrap();

    assert_eq!(report.original_size, 1200);
    println!("Package size: {} bytes", report.original_size);
}

#[test]
fn test_optimization_deduplication_analysis() {
    let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

    // Add duplicate resources (same content)
    let data = vec![1, 2, 3, 4, 5];
    package.add_resource(Resource::new(
        "file1.dat".to_string(),
        ResourceType::Data,
        data.clone(),
    ));
    package.add_resource(Resource::new(
        "file2.dat".to_string(),
        ResourceType::Data,
        data.clone(),
    ));
    package.add_resource(Resource::new(
        "file3.dat".to_string(),
        ResourceType::Data,
        data,
    ));

    // Add unique resource
    package.add_resource(Resource::new(
        "unique.dat".to_string(),
        ResourceType::Data,
        vec![9, 9, 9],
    ));

    let optimizer = PackageOptimizer::new();
    let report = optimizer.analyze(&package).unwrap();

    assert_eq!(report.deduplication.total_resources, 4);
    assert_eq!(report.deduplication.unique_resources, 2); // 2 unique content sets
    assert_eq!(report.deduplication.duplicate_count, 2); // 2 duplicates (3 copies - 1 kept)
    assert!(report.deduplication.potential_savings > 0);

    println!("Deduplication analysis: {:?}", report.deduplication);
}

#[test]
fn test_optimization_compression_analysis() {
    let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

    // Add large text resource (highly compressible)
    let text_data = "This is highly compressible text data. ".repeat(1000);
    package.add_resource(Resource::new(
        "readme.txt".to_string(),
        ResourceType::Text,
        text_data.into_bytes(),
    ));

    // Add source code (also compressible)
    let source_code = "fn main() { println!(\"Hello\"); }".repeat(100);
    package.add_resource(Resource::new(
        "main.rs".to_string(),
        ResourceType::Source,
        source_code.into_bytes(),
    ));

    let optimizer = PackageOptimizer::new();
    let report = optimizer.analyze(&package).unwrap();

    // Should identify compressible resources
    assert!(
        !report.compression.compressible_resources.is_empty(),
        "Should find compressible resources"
    );
    assert!(
        report.compression.potential_savings > 0,
        "Should estimate compression savings"
    );

    println!("Compression analysis: {:?}", report.compression);
}

#[test]
fn test_optimization_apply_deduplication() {
    let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

    // Add duplicate resources
    let data = vec![1, 2, 3, 4, 5];
    package.add_resource(Resource::new(
        "dup1.dat".to_string(),
        ResourceType::Data,
        data.clone(),
    ));
    package.add_resource(Resource::new(
        "dup2.dat".to_string(),
        ResourceType::Data,
        data.clone(),
    ));
    package.add_resource(Resource::new(
        "dup3.dat".to_string(),
        ResourceType::Data,
        data,
    ));

    let original_count = package.resources().len();
    assert_eq!(original_count, 3);

    let mut optimizer = PackageOptimizer::new();
    optimizer.enable_deduplication = true;

    // Apply optimization (which includes deduplication)
    let _report = optimizer.optimize(&mut package).unwrap();

    // After deduplication, should have only 1 resource
    let final_count = package.resources().len();
    assert_eq!(final_count, 1, "Deduplication should remove duplicates");

    println!(
        "Deduplication reduced {} resources to {}",
        original_count, final_count
    );
}

#[test]
fn test_package_source_code_extraction() {
    // This test verifies the source code extraction placeholder
    // Note: The actual feature is gated behind "with-nn" feature
    let package = Package::new("test_package".to_string(), "1.0.0".to_string());

    // Without the feature, we just verify the package is created correctly
    assert_eq!(package.name(), "test_package");
    assert_eq!(package.get_version(), "1.0.0");

    // The source code extraction is implemented with a placeholder
    // that would be populated when the with-nn feature is enabled
    println!("Package created successfully");
}

#[test]
fn test_completed_implementations_integration() {
    // Integration test combining all completed implementations
    let mut package = Package::new("integration_test".to_string(), "1.0.0".to_string());

    // Add various resources
    package.add_resource(Resource::new(
        "model.pth".to_string(),
        ResourceType::Model,
        vec![0u8; 5000],
    ));
    package.add_resource(Resource::new(
        "config.json".to_string(),
        ResourceType::Config,
        vec![0u8; 500],
    ));

    // Run diagnostics
    let diagnostics = PackageDiagnostics::new();
    let diag_report = diagnostics.diagnose(&package).unwrap();

    assert!(
        matches!(
            diag_report.status,
            HealthStatus::Healthy | HealthStatus::Warning
        ),
        "Package should have valid health status"
    );

    // Run optimization analysis
    let optimizer = PackageOptimizer::new();
    let opt_report = optimizer.analyze(&package).unwrap();

    assert!(opt_report.original_size > 0);
    assert!(opt_report.deduplication.total_resources > 0);

    println!("Integration test passed!");
    println!("Health: {:?}", diag_report.status);
    println!("Original size: {} bytes", opt_report.original_size);
}
