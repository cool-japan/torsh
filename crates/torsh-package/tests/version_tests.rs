//! Version management and compatibility tests for torsh-package

use tempfile::TempDir;
use torsh_package::{
    manifest::ManifestBuilder,
    version::{CompatibilityChecker, PackageVersion, VersionComparator, VersionRequirement},
    Package, Resource, ResourceType,
};

#[test]
fn test_semantic_versioning() {
    // Test version creation
    let v1 = PackageVersion::new(1, 2, 3);
    assert_eq!(v1.major, 1);
    assert_eq!(v1.minor, 2);
    assert_eq!(v1.patch, 3);
    assert_eq!(v1.to_string(), "1.2.3");

    // Test version parsing
    let v2 = PackageVersion::parse("2.0.0-alpha.1+build.123").unwrap();
    assert_eq!(v2.major, 2);
    assert_eq!(v2.minor, 0);
    assert_eq!(v2.patch, 0);
    assert_eq!(v2.pre_release.as_deref(), Some("alpha.1"));
    assert_eq!(v2.build_metadata.as_deref(), Some("build.123"));
    assert!(v2.is_pre_release());

    // Test version comparison
    let v3 = PackageVersion::new(1, 2, 4);
    let v4 = PackageVersion::new(1, 3, 0);
    let v5 = PackageVersion::new(2, 0, 0);

    assert!(v1 < v3);
    assert!(v3 < v4);
    assert!(v4 < v5);

    // Test version increment
    assert_eq!(v1.next_patch(), PackageVersion::new(1, 2, 4));
    assert_eq!(v1.next_minor(), PackageVersion::new(1, 3, 0));
    assert_eq!(v1.next_major(), PackageVersion::new(2, 0, 0));
}

#[test]
fn test_version_requirements() {
    let v1_2_3 = PackageVersion::new(1, 2, 3);
    let v1_2_4 = PackageVersion::new(1, 2, 4);
    let v1_3_0 = PackageVersion::new(1, 3, 0);
    let v2_0_0 = PackageVersion::new(2, 0, 0);

    // Test exact requirement
    let exact_req = VersionRequirement::parse("=1.2.3").unwrap();
    assert!(exact_req.matches(&v1_2_3));
    assert!(!exact_req.matches(&v1_2_4));

    // Test greater than
    let gt_req = VersionRequirement::parse(">1.2.3").unwrap();
    assert!(!gt_req.matches(&v1_2_3));
    assert!(gt_req.matches(&v1_2_4));
    assert!(gt_req.matches(&v1_3_0));

    // Test greater than or equal
    let gte_req = VersionRequirement::parse(">=1.2.3").unwrap();
    assert!(gte_req.matches(&v1_2_3));
    assert!(gte_req.matches(&v1_2_4));
    assert!(gte_req.matches(&v2_0_0));

    // Test less than
    let lt_req = VersionRequirement::parse("<1.3.0").unwrap();
    assert!(lt_req.matches(&v1_2_3));
    assert!(lt_req.matches(&v1_2_4));
    assert!(!lt_req.matches(&v1_3_0));

    // Test compatible (caret)
    let compatible_req = VersionRequirement::parse("^1.2.3").unwrap();
    assert!(compatible_req.matches(&v1_2_3));
    assert!(compatible_req.matches(&v1_2_4));
    assert!(compatible_req.matches(&v1_3_0));
    assert!(!compatible_req.matches(&v2_0_0));
}

#[test]
fn test_compatibility_checker() {
    let mut checker = CompatibilityChecker::new();

    // Add multiple requirements
    checker.add_requirement(VersionRequirement::parse(">=1.2.0").unwrap());
    checker.add_requirement(VersionRequirement::parse("<2.0.0").unwrap());
    checker.add_requirement(VersionRequirement::parse("^1.2.0").unwrap());

    // Test compatible versions
    assert!(checker.check(&PackageVersion::new(1, 2, 0)));
    assert!(checker.check(&PackageVersion::new(1, 2, 5)));
    assert!(checker.check(&PackageVersion::new(1, 5, 0)));

    // Test incompatible versions
    assert!(!checker.check(&PackageVersion::new(1, 1, 9))); // Too old
    assert!(!checker.check(&PackageVersion::new(2, 0, 0))); // Major version change
    assert!(!checker.check(&PackageVersion::new(0, 9, 0))); // Too old major
}

#[test]
fn test_package_version_compatibility() {
    let temp_dir = TempDir::new().unwrap();

    // Create packages with different versions
    let versions = vec!["1.0.0", "1.0.1", "1.1.0", "2.0.0"];
    let mut packages = Vec::new();

    for version in versions {
        let package_path = temp_dir
            .path()
            .join(format!("package_{}.torshpkg", version));

        let mut package = Package::new("test_package".to_string(), version.to_string());

        // Add a test resource
        let resource = Resource::new(
            "test.txt".to_string(),
            ResourceType::Text,
            format!("Version: {}", version).as_bytes().to_vec(),
        );
        package
            .resources_mut()
            .insert(resource.name.clone(), resource);

        // Save package
        package.save(&package_path).unwrap();
        packages.push((version.to_string(), package_path));
    }

    // Test loading and version verification
    for (expected_version, path) in packages {
        let loaded = Package::load(&path).unwrap();
        assert_eq!(loaded.metadata().version, expected_version);

        // Verify version can be parsed
        let parsed_version = PackageVersion::parse(&loaded.metadata().version).unwrap();
        assert_eq!(parsed_version.to_string(), expected_version);
    }
}

#[test]
fn test_manifest_version_validation() {
    // Test valid manifest
    let valid_manifest = ManifestBuilder::new("test".to_string(), "1.2.3".to_string())
        .author("Test Author".to_string())
        .description("Test package".to_string())
        .build();

    assert!(valid_manifest.validate().is_ok());

    // Test invalid version format
    let mut invalid_manifest = valid_manifest.clone();
    invalid_manifest.version = "not.a.version".to_string();

    let result = invalid_manifest.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid version format"));

    // Test empty name
    let mut invalid_manifest = valid_manifest.clone();
    invalid_manifest.name = String::new();

    let result = invalid_manifest.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Package name cannot be empty"));

    // Test empty version
    let mut invalid_manifest = valid_manifest;
    invalid_manifest.version = String::new();

    let result = invalid_manifest.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("Package version cannot be empty"));
}

#[test]
fn test_format_version_compatibility() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("format_test.torshpkg");

    // Create package with current format version
    let mut package = Package::new("format_test".to_string(), "1.0.0".to_string());
    let resource = Resource::new(
        "test.txt".to_string(),
        ResourceType::Text,
        b"test data".to_vec(),
    );
    package
        .resources_mut()
        .insert(resource.name.clone(), resource);

    // Save package
    package.save(&package_path).unwrap();

    // Load and verify format version
    let loaded = Package::load(&package_path).unwrap();
    assert_eq!(
        loaded.metadata().format_version,
        torsh_package::PACKAGE_FORMAT_VERSION
    );

    // Verify format version can be parsed
    let format_version = semver::Version::parse(&loaded.metadata().format_version).unwrap();
    assert_eq!(format_version.major, 1);

    // Test compatibility check
    let current_format = semver::Version::parse(torsh_package::PACKAGE_FORMAT_VERSION).unwrap();
    assert_eq!(format_version.major, current_format.major);
}

#[test]
fn test_pre_release_versions() {
    // Test pre-release version parsing
    let pre_versions = vec![
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-beta",
        "1.0.0-rc.1",
        "1.0.0-alpha.1+build.123",
    ];

    for version_str in pre_versions {
        let version = PackageVersion::parse(version_str).unwrap();
        assert!(version.is_pre_release());

        // Pre-release should be less than release version
        let release_version = PackageVersion::new(version.major, version.minor, version.patch);
        assert!(version < release_version);
    }

    // Test pre-release ordering
    let alpha = PackageVersion::parse("1.0.0-alpha").unwrap();
    let alpha_1 = PackageVersion::parse("1.0.0-alpha.1").unwrap();
    let beta = PackageVersion::parse("1.0.0-beta").unwrap();
    let rc = PackageVersion::parse("1.0.0-rc.1").unwrap();
    let release = PackageVersion::parse("1.0.0").unwrap();

    assert!(alpha < alpha_1);
    assert!(alpha_1 < beta);
    assert!(beta < rc);
    assert!(rc < release);
}

#[test]
fn test_build_metadata_handling() {
    // Build metadata should not affect version comparison
    let v1 = PackageVersion::parse("1.0.0+build.1").unwrap();
    let v2 = PackageVersion::parse("1.0.0+build.2").unwrap();
    let v3 = PackageVersion::parse("1.0.0").unwrap();

    // All should be considered equal for compatibility
    assert_eq!(v1.major, v2.major);
    assert_eq!(v1.minor, v2.minor);
    assert_eq!(v1.patch, v2.patch);

    // Build metadata should be preserved
    assert_eq!(v1.build_metadata.as_deref(), Some("build.1"));
    assert_eq!(v2.build_metadata.as_deref(), Some("build.2"));
    assert_eq!(v3.build_metadata, None);

    // But comparison should treat them as equal
    let req = VersionRequirement::new(VersionComparator::Exact, v3);
    assert!(req.matches(&v1));
    assert!(req.matches(&v2));
}

#[test]
fn test_version_requirement_display() {
    let version = PackageVersion::new(1, 2, 3);

    let exact = VersionRequirement::new(VersionComparator::Exact, version.clone());
    assert_eq!(exact.to_string(), "=1.2.3");

    let gt = VersionRequirement::new(VersionComparator::GreaterThan, version.clone());
    assert_eq!(gt.to_string(), ">1.2.3");

    let gte = VersionRequirement::new(VersionComparator::GreaterThanOrEqual, version.clone());
    assert_eq!(gte.to_string(), ">=1.2.3");

    let lt = VersionRequirement::new(VersionComparator::LessThan, version.clone());
    assert_eq!(lt.to_string(), "<1.2.3");

    let lte = VersionRequirement::new(VersionComparator::LessThanOrEqual, version.clone());
    assert_eq!(lte.to_string(), "<=1.2.3");

    let compatible = VersionRequirement::new(VersionComparator::Compatible, version);
    assert_eq!(compatible.to_string(), "^1.2.3");
}

#[test]
fn test_dependency_version_management() {
    let temp_dir = TempDir::new().unwrap();
    let package_path = temp_dir.path().join("deps_test.torshpkg");

    // Create package with version-specific dependencies
    let mut package = Package::new("deps_test".to_string(), "2.1.0".to_string());

    // Add dependencies with version requirements
    package.add_dependency("serde", "^1.0.0");
    package.add_dependency("tokio", ">=1.20.0");
    package.add_dependency("uuid", "=1.5.0");

    let resource = Resource::new(
        "main.rs".to_string(),
        ResourceType::Source,
        b"fn main() {}".to_vec(),
    );
    package
        .resources_mut()
        .insert(resource.name.clone(), resource);

    // Save and reload
    package.save(&package_path).unwrap();
    let loaded = Package::load(&package_path).unwrap();

    // Verify dependencies
    let deps = &loaded.metadata().dependencies;
    assert_eq!(deps.get("serde"), Some(&"^1.0.0".to_string()));
    assert_eq!(deps.get("tokio"), Some(&">=1.20.0".to_string()));
    assert_eq!(deps.get("uuid"), Some(&"=1.5.0".to_string()));

    // Test dependency requirement parsing
    for (name, version_req_str) in deps {
        let req = VersionRequirement::parse(version_req_str);
        assert!(
            req.is_ok(),
            "Failed to parse dependency {}: {}",
            name,
            version_req_str
        );

        // Test some version compatibility
        match name.as_str() {
            "serde" => {
                let req = req.unwrap();
                assert!(req.matches(&PackageVersion::parse("1.0.0").unwrap()));
                assert!(req.matches(&PackageVersion::parse("1.5.0").unwrap()));
                assert!(!req.matches(&PackageVersion::parse("2.0.0").unwrap()));
            }
            "tokio" => {
                let req = req.unwrap();
                assert!(req.matches(&PackageVersion::parse("1.20.0").unwrap()));
                assert!(req.matches(&PackageVersion::parse("1.25.0").unwrap()));
                assert!(!req.matches(&PackageVersion::parse("1.19.0").unwrap()));
            }
            "uuid" => {
                let req = req.unwrap();
                assert!(req.matches(&PackageVersion::parse("1.5.0").unwrap()));
                assert!(!req.matches(&PackageVersion::parse("1.5.1").unwrap()));
                assert!(!req.matches(&PackageVersion::parse("1.4.0").unwrap()));
            }
            _ => {}
        }
    }
}
