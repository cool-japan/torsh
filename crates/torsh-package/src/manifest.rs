//! Package manifest handling

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Package manifest containing metadata and contents information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    /// Package name
    pub name: String,

    /// Package version
    pub version: String,

    /// Package format version
    pub format_version: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Package author
    pub author: Option<String>,

    /// Package description
    pub description: Option<String>,

    /// License
    pub license: Option<String>,

    /// List of modules in the package
    pub modules: Vec<ModuleInfo>,

    /// List of resources in the package
    pub resources: Vec<ResourceInfo>,

    /// Dependencies
    pub dependencies: HashMap<String, String>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,

    /// Cryptographic signature (if signed)
    pub signature: Option<PackageSignature>,
}

/// Information about a module in the package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    /// Module name
    pub name: String,

    /// Module class name
    pub class_name: String,

    /// Module version
    pub version: String,

    /// Module dependencies
    pub dependencies: Vec<String>,

    /// Whether source code is included
    pub has_source: bool,
}

/// Information about a resource in the package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    /// Resource name
    pub name: String,

    /// Resource type
    pub resource_type: String,

    /// Resource size in bytes
    pub size: u64,

    /// SHA256 hash
    pub sha256: String,

    /// Compression method
    pub compression: Option<String>,
}

/// Package signature for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageSignature {
    /// Signature algorithm
    pub algorithm: String,

    /// Public key ID
    pub key_id: String,

    /// Signature data
    pub signature: String,

    /// Timestamp
    pub signed_at: DateTime<Utc>,
}

impl PackageManifest {
    /// Create a new package manifest
    pub fn new(name: String, version: String) -> Self {
        Self {
            name,
            version,
            format_version: crate::PACKAGE_FORMAT_VERSION.to_string(),
            created_at: Utc::now(),
            author: None,
            description: None,
            license: None,
            modules: Vec::new(),
            resources: Vec::new(),
            dependencies: HashMap::new(),
            metadata: HashMap::new(),
            signature: None,
        }
    }

    /// Validate manifest
    pub fn validate(&self) -> Result<(), String> {
        // Check required fields
        if self.name.is_empty() {
            return Err("Package name cannot be empty".to_string());
        }

        if self.version.is_empty() {
            return Err("Package version cannot be empty".to_string());
        }

        // Validate version format
        if semver::Version::parse(&self.version).is_err() {
            return Err(format!("Invalid version format: {}", self.version));
        }

        // Check format version compatibility
        let current_version = semver::Version::parse(crate::PACKAGE_FORMAT_VERSION).unwrap();
        let manifest_version = semver::Version::parse(&self.format_version)
            .map_err(|_| "Invalid format version in manifest")?;

        if manifest_version.major != current_version.major {
            return Err(format!(
                "Incompatible package format version: {} (expected {}.x.x)",
                self.format_version, current_version.major
            ));
        }

        // Validate modules
        for module in &self.modules {
            if module.name.is_empty() {
                return Err("Module name cannot be empty".to_string());
            }
        }

        Ok(())
    }

    /// Add author information
    pub fn with_author(mut self, author: String) -> Self {
        self.author = Some(author);
        self
    }

    /// Add description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Add license
    pub fn with_license(mut self, license: String) -> Self {
        self.license = Some(license);
        self
    }

    /// Get total package size
    pub fn total_size(&self) -> u64 {
        self.resources.iter().map(|r| r.size).sum()
    }

    /// Get module by name
    pub fn get_module(&self, name: &str) -> Option<&ModuleInfo> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// Get resource by name
    pub fn get_resource(&self, name: &str) -> Option<&ResourceInfo> {
        self.resources.iter().find(|r| r.name == name)
    }
}

/// Manifest builder for convenient creation
pub struct ManifestBuilder {
    manifest: PackageManifest,
}

impl ManifestBuilder {
    /// Create a new manifest builder
    pub fn new(name: String, version: String) -> Self {
        Self {
            manifest: PackageManifest::new(name, version),
        }
    }

    /// Set author
    pub fn author(mut self, author: String) -> Self {
        self.manifest.author = Some(author);
        self
    }

    /// Set description
    pub fn description(mut self, description: String) -> Self {
        self.manifest.description = Some(description);
        self
    }

    /// Set license
    pub fn license(mut self, license: String) -> Self {
        self.manifest.license = Some(license);
        self
    }

    /// Add a module
    pub fn add_module(mut self, module: ModuleInfo) -> Self {
        self.manifest.modules.push(module);
        self
    }

    /// Add a resource
    pub fn add_resource(mut self, resource: ResourceInfo) -> Self {
        self.manifest.resources.push(resource);
        self
    }

    /// Add a dependency
    pub fn add_dependency(mut self, name: String, version: String) -> Self {
        self.manifest.dependencies.insert(name, version);
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.manifest.metadata.insert(key, value);
        self
    }

    /// Build the manifest
    pub fn build(self) -> PackageManifest {
        self.manifest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_creation() {
        let manifest = PackageManifest::new("test_package".to_string(), "1.0.0".to_string());

        assert_eq!(manifest.name, "test_package");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.format_version, crate::PACKAGE_FORMAT_VERSION);
    }

    #[test]
    fn test_manifest_validation() {
        let mut manifest = PackageManifest::new("test".to_string(), "1.0.0".to_string());
        assert!(manifest.validate().is_ok());

        // Invalid version
        manifest.version = "invalid".to_string();
        assert!(manifest.validate().is_err());

        // Empty name
        manifest.version = "1.0.0".to_string();
        manifest.name = String::new();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_manifest_builder() {
        let manifest = ManifestBuilder::new("test".to_string(), "1.0.0".to_string())
            .author("Test Author".to_string())
            .description("Test package".to_string())
            .license("MIT".to_string())
            .add_dependency("torsh-core".to_string(), "0.1.0-alpha.1".to_string())
            .build();

        assert_eq!(manifest.author.as_deref(), Some("Test Author"));
        assert_eq!(manifest.description.as_deref(), Some("Test package"));
        assert_eq!(manifest.license.as_deref(), Some("MIT"));
        assert_eq!(
            manifest.dependencies.get("torsh-core"),
            Some(&"0.1.0".to_string())
        );
    }
}
