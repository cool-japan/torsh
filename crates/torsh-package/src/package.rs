//! Core Package implementation
//!
//! This module contains the main Package struct and its core functionality
//! for creating, managing, and persisting model packages.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use torsh_core::error::{Result, TorshError};

use crate::exporter::{ExportConfig, PackageExporter};
use crate::importer::PackageImporter;
use crate::manifest::{ModuleInfo, PackageManifest};
use crate::resources::{Resource, ResourceType};
use crate::utils::calculate_hash;
use crate::PACKAGE_FORMAT_VERSION;

/// Main package structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Package {
    pub(crate) manifest: PackageManifest,
    pub(crate) resources: HashMap<String, Resource>,
}

impl Package {
    /// Create a new package
    pub fn new(name: String, version: String) -> Self {
        let manifest = PackageManifest {
            name,
            version,
            format_version: PACKAGE_FORMAT_VERSION.to_string(),
            created_at: chrono::Utc::now(),
            author: None,
            description: None,
            license: None,
            dependencies: HashMap::new(),
            modules: Vec::new(),
            resources: Vec::new(),
            metadata: HashMap::new(),
            signature: None,
        };

        Self {
            manifest,
            resources: HashMap::new(),
        }
    }

    /// Get the package name
    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Get the package version
    pub fn get_version(&self) -> &str {
        &self.manifest.version
    }

    /// Load package from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let importer = PackageImporter::new(crate::importer::ImportConfig::default());
        importer.import_package(path)
    }

    /// Save package to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let exporter = PackageExporter::new(ExportConfig::default());
        exporter.export_package(self, path)
    }

    /// Add a module to the package (temporarily disabled - requires torsh-nn)
    #[cfg(feature = "with-nn")]
    pub fn add_module<M: torsh_nn::Module>(
        &mut self,
        name: &str,
        module: &M,
        include_source: bool,
    ) -> Result<()> {
        // Collect parameter metadata for demonstration purposes
        // In a real implementation, this would serialize the actual tensor data
        let parameters = module.parameters();
        let mut param_metadata = Vec::new();

        for (param_name, param) in parameters {
            // For now, just store parameter metadata instead of actual data
            // This is a simplified approach for demonstration
            let shape = param.shape().unwrap_or_default();
            let numel = param.numel().unwrap_or(0);

            let metadata = format!(
                "{}:shape={:?},numel={},requires_grad={}",
                param_name,
                shape,
                numel,
                param.requires_grad()
            );
            param_metadata.push(metadata);
        }

        // Create a simple serialized representation
        let param_data = serde_json::to_vec(&param_metadata)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        // Create resource for the module
        let resource = Resource {
            name: format!("{}.pth", name),
            resource_type: ResourceType::Model,
            data: param_data,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("type".to_string(), "module".to_string());
                meta.insert("name".to_string(), name.to_string());
                meta
            },
        };

        self.resources.insert(resource.name.clone(), resource);

        // Add module info to manifest
        let module_info = ModuleInfo {
            name: name.to_string(),
            class_name: name.to_string(), // Use module name as class name for simplicity
            version: "1.0.0".to_string(), // Default version
            dependencies: Vec::new(),
            has_source: include_source,
        };

        self.manifest.modules.push(module_info);

        // Add torsh dependency
        self.manifest.dependencies.insert(
            "torsh-nn".to_string(),
            env!("CARGO_PKG_VERSION").to_string(),
        );

        if include_source {
            // TODO: Extract and include source code
        }

        Ok(())
    }

    /// Get module data
    pub fn get_module(&self, name: &str) -> Result<Vec<u8>> {
        let module_path = format!("{}.pth", name);
        self.resources
            .get(&module_path)
            .map(|resource| resource.data.clone())
            .ok_or_else(|| {
                TorshError::General(torsh_core::error::GeneralError::InvalidArgument(format!(
                    "Module '{}' not found",
                    name
                )))
            })
    }

    /// Add a data file to the package
    pub fn add_data_file<P: AsRef<Path>>(&mut self, name: &str, path: P) -> Result<()> {
        let data = fs::read(&path)
            .map_err(|e| TorshError::IoError(format!("Failed to read file: {}", e)))?;

        let resource = Resource {
            name: name.to_string(),
            resource_type: ResourceType::Data,
            data,
            metadata: HashMap::new(),
        };

        self.resources.insert(resource.name.clone(), resource);

        Ok(())
    }

    /// Add source code to the package
    pub fn add_source_file(&mut self, name: &str, source: &str) -> Result<()> {
        let resource = Resource {
            name: format!("{}.rs", name),
            resource_type: ResourceType::Source,
            data: source.as_bytes().to_vec(),
            metadata: HashMap::new(),
        };

        self.resources.insert(resource.name.clone(), resource);

        Ok(())
    }

    /// List all modules in the package
    pub fn list_modules(&self) -> Vec<&ModuleInfo> {
        self.manifest.modules.iter().collect()
    }

    /// Get package metadata
    pub fn metadata(&self) -> &PackageManifest {
        &self.manifest
    }

    /// Get resources
    pub fn resources(&self) -> &std::collections::HashMap<String, Resource> {
        &self.resources
    }

    /// Add a resource to the package
    pub fn add_resource(&mut self, resource: Resource) {
        self.resources.insert(resource.name.clone(), resource);
    }

    /// Get mutable access to resources (for testing and advanced usage)
    pub fn resources_mut(&mut self) -> &mut std::collections::HashMap<String, Resource> {
        &mut self.resources
    }

    /// Get mutable access to manifest (for testing and advanced usage)
    pub fn manifest_mut(&mut self) -> &mut PackageManifest {
        &mut self.manifest
    }

    /// Add dependency information
    pub fn add_dependency(&mut self, name: &str, version: &str) {
        self.manifest
            .dependencies
            .insert(name.to_string(), version.to_string());
    }

    /// Verify package integrity
    pub fn verify(&self) -> Result<bool> {
        // Check manifest validity
        if self.manifest.name.is_empty() {
            return Ok(false);
        }

        if self.manifest.version.is_empty() {
            return Ok(false);
        }

        // Verify format version compatibility
        let format_version =
            semver::Version::parse(&self.manifest.format_version).map_err(|e| {
                TorshError::General(torsh_core::error::GeneralError::InvalidArgument(
                    e.to_string(),
                ))
            })?;
        let current_format = semver::Version::parse(PACKAGE_FORMAT_VERSION).map_err(|e| {
            TorshError::General(torsh_core::error::GeneralError::ConfigError(e.to_string()))
        })?;

        if format_version.major != current_format.major {
            return Ok(false);
        }

        // Verify checksums if available
        for resource in self.resources.values() {
            if let Some(expected_hash) = resource.metadata.get("sha256") {
                let actual_hash = calculate_hash(&resource.data);
                if &actual_hash != expected_hash {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_creation() {
        let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

        // Add some test data
        package.add_source_file("test", "fn main() {}").unwrap();

        assert_eq!(package.manifest.name, "test_package");
        assert_eq!(package.manifest.version, "1.0.0");
        assert_eq!(package.resources.len(), 1);
    }

    #[test]
    fn test_package_verification() {
        let package = Package::new("test".to_string(), "1.0.0".to_string());
        assert!(package.verify().unwrap());

        // Test invalid package
        let mut invalid_package = Package::new("".to_string(), "1.0.0".to_string());
        assert!(!invalid_package.verify().unwrap());

        invalid_package.manifest.name = "test".to_string();
        invalid_package.manifest.version = "".to_string();
        assert!(!invalid_package.verify().unwrap());
    }

    #[test]
    fn test_add_dependency() {
        let mut package = Package::new("test".to_string(), "1.0.0".to_string());
        package.add_dependency("serde", "1.0");

        assert_eq!(
            package.manifest.dependencies.get("serde"),
            Some(&"1.0".to_string())
        );
    }

    #[test]
    fn test_add_data_file() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, b"test data").unwrap();

        let mut package = Package::new("test".to_string(), "1.0.0".to_string());
        package.add_data_file("test.txt", &file_path).unwrap();

        let resource = package.resources.get("test.txt").unwrap();
        assert_eq!(resource.data, b"test data");
        assert_eq!(resource.resource_type, ResourceType::Data);
    }
}
