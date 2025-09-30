//! Package importer functionality

use crate::{Package, PackageManifest, Resource, ResourceType};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use tempfile::TempDir;
use torsh_core::error::{Result, TorshError};
use zip::read::ZipArchive;

/// Configuration for package import
#[derive(Debug, Clone)]
pub struct ImportConfig {
    /// Verify package integrity
    pub verify_integrity: bool,

    /// Verify signatures
    pub verify_signatures: bool,

    /// Extract to temporary directory
    pub use_temp_dir: bool,

    /// Verbose output
    pub verbose: bool,

    /// Allow loading from untrusted sources
    pub allow_untrusted: bool,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            verify_integrity: true,
            verify_signatures: false,
            use_temp_dir: true,
            verbose: false,
            allow_untrusted: false,
        }
    }
}

/// Package importer
pub struct PackageImporter {
    config: ImportConfig,
}

impl PackageImporter {
    /// Create a new importer
    pub fn new(config: ImportConfig) -> Self {
        Self { config }
    }

    /// Import a package from a file
    pub fn import_package<P: AsRef<Path>>(&self, path: P) -> Result<Package> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(TorshError::IoError(format!(
                "Package file not found: {:?}",
                path
            )));
        }

        if self.config.verbose {
            println!("Importing package from: {:?}", path);
        }

        // Open zip file
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(file)
            .map_err(|e| TorshError::IoError(format!("Failed to open package: {}", e)))?;

        // Read manifest first
        let manifest = self.read_manifest(&mut archive)?;

        // Validate manifest
        manifest
            .validate()
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid manifest: {}", e)))?;

        // Create package
        let mut package = Package::new(manifest.name.clone(), manifest.version.clone());
        package.manifest = manifest;

        // Read resources
        self.read_resources(&mut archive, &mut package)?;

        // Verify integrity if requested
        if self.config.verify_integrity {
            if self.config.verbose {
                println!("Verifying package integrity...");
            }

            if !package.verify()? {
                return Err(TorshError::InvalidArgument(
                    "Package integrity verification failed".to_string(),
                ));
            }
        }

        if self.config.verbose {
            println!("Package imported successfully");
        }

        Ok(package)
    }

    /// Read manifest from archive
    fn read_manifest(&self, archive: &mut ZipArchive<File>) -> Result<PackageManifest> {
        let mut manifest_file = archive
            .by_name("MANIFEST.json")
            .map_err(|_| TorshError::InvalidArgument("No manifest found in package".to_string()))?;

        let mut manifest_json = String::new();
        manifest_file.read_to_string(&mut manifest_json)?;

        let manifest: PackageManifest = serde_json::from_str(&manifest_json).map_err(|e| {
            TorshError::SerializationError(format!("Failed to parse manifest: {}", e))
        })?;

        Ok(manifest)
    }

    /// Read all resources from archive
    fn read_resources(&self, archive: &mut ZipArchive<File>, package: &mut Package) -> Result<()> {
        // First pass: collect resources and their metadata file names
        let mut resources_to_add = Vec::new();
        let mut metadata_files = HashMap::new();

        for i in 0..archive.len() {
            let file = archive
                .by_index(i)
                .map_err(|e| TorshError::IoError(e.to_string()))?;

            let name = file.name().to_string();

            // Skip manifest
            if name == "MANIFEST.json" {
                continue;
            }

            // Collect metadata files for later
            if name.ends_with(".metadata") {
                let resource_name = name.trim_end_matches(".metadata");
                metadata_files.insert(resource_name.to_string(), i);
                continue;
            }

            resources_to_add.push((i, name));
        }

        // Process resources with their metadata
        for (index, name) in resources_to_add {
            // Read the resource file
            let mut data = Vec::new();
            let resource_type;
            {
                let mut file = archive
                    .by_index(index)
                    .map_err(|e| TorshError::IoError(e.to_string()))?;

                if self.config.verbose {
                    println!("Reading resource: {}", name);
                }

                // Determine resource type from path
                resource_type = self.determine_resource_type(&name);

                // Read resource data
                file.read_to_end(&mut data)?;
            }

            // Extract resource name from path
            let resource_name = Path::new(&name)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&name)
                .to_string();

            // Create resource
            let mut resource = Resource::new(resource_name.clone(), resource_type, data);

            // Read metadata if exists
            if let Some(&metadata_index) = metadata_files.get(&name) {
                let mut metadata_json = String::new();
                {
                    let mut metadata_file = archive
                        .by_index(metadata_index)
                        .map_err(|e| TorshError::IoError(e.to_string()))?;
                    metadata_file.read_to_string(&mut metadata_json)?;
                }

                if let Ok(metadata) =
                    serde_json::from_str::<HashMap<String, String>>(&metadata_json)
                {
                    resource.metadata = metadata;
                }
            }

            package.resources.insert(resource_name, resource);
        }

        Ok(())
    }

    /// Determine resource type from path
    fn determine_resource_type(&self, path: &str) -> ResourceType {
        if path.starts_with("models/") {
            ResourceType::Model
        } else if path.starts_with("src/") {
            ResourceType::Source
        } else if path.starts_with("data/") {
            ResourceType::Data
        } else if path.starts_with("config/") {
            ResourceType::Config
        } else if path.starts_with("docs/") {
            ResourceType::Documentation
        } else {
            // Determine from extension
            Path::new(path)
                .extension()
                .and_then(|e| e.to_str())
                .map(ResourceType::from_extension)
                .unwrap_or(ResourceType::Data)
        }
    }

    /// Extract package to directory
    pub fn extract_package<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        package_path: P,
        output_dir: Q,
    ) -> Result<()> {
        let package_path = package_path.as_ref();
        let output_dir = output_dir.as_ref();

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Open package
        let file = File::open(package_path)?;
        let mut archive = ZipArchive::new(file)
            .map_err(|e| TorshError::IoError(format!("Failed to open package: {}", e)))?;

        // Extract all files
        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| TorshError::IoError(e.to_string()))?;

            let outpath = output_dir.join(file.name());

            if file.name().ends_with('/') {
                // Create directory
                std::fs::create_dir_all(&outpath)?;
            } else {
                // Create parent directory
                if let Some(parent) = outpath.parent() {
                    std::fs::create_dir_all(parent)?;
                }

                // Extract file
                let mut outfile = File::create(&outpath)?;
                io::copy(&mut file, &mut outfile)?;
            }

            if self.config.verbose {
                println!("Extracted: {}", file.name());
            }
        }

        Ok(())
    }
}

/// Import and load a module from a package
pub fn import_and_load_module<P: AsRef<Path>>(
    package_path: P,
    module_name: &str,
) -> Result<Vec<u8>> {
    let importer = PackageImporter::new(ImportConfig::default());
    let package = importer.import_package(package_path)?;

    package.get_module(module_name)
}

/// Import context for managing imported packages
pub struct ImportContext {
    packages: HashMap<String, Package>,
    temp_dirs: Vec<TempDir>,
}

impl ImportContext {
    /// Create a new import context
    pub fn new() -> Self {
        Self {
            packages: HashMap::new(),
            temp_dirs: Vec::new(),
        }
    }

    /// Import a package into the context
    pub fn import_package<P: AsRef<Path>>(
        &mut self,
        path: P,
        alias: Option<String>,
    ) -> Result<String> {
        let importer = PackageImporter::new(ImportConfig::default());
        let package = importer.import_package(path)?;

        let package_id = alias
            .unwrap_or_else(|| format!("{}-{}", package.manifest.name, package.manifest.version));

        self.packages.insert(package_id.clone(), package);

        Ok(package_id)
    }

    /// Get a package by ID
    pub fn get_package(&self, id: &str) -> Option<&Package> {
        self.packages.get(id)
    }

    /// List imported packages
    pub fn list_packages(&self) -> Vec<(&str, &PackageManifest)> {
        self.packages
            .iter()
            .map(|(id, pkg)| (id.as_str(), &pkg.manifest))
            .collect()
    }

    /// Get a module from a package
    pub fn get_module(&self, package_id: &str, module_name: &str) -> Result<Vec<u8>> {
        self.packages
            .get(package_id)
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!("Package '{}' not found", package_id))
            })?
            .get_module(module_name)
    }

    /// Clear all imported packages
    pub fn clear(&mut self) {
        self.packages.clear();
        self.temp_dirs.clear();
    }
}

impl Default for ImportContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporter::PackageExporter;
    use tempfile::TempDir;

    #[test]
    fn test_import_config() {
        let config = ImportConfig::default();
        assert!(config.verify_integrity);
        assert!(!config.verify_signatures);
        assert!(config.use_temp_dir);
    }

    #[test]
    fn test_package_import_export_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let package_path = temp_dir.path().join("test.torshpkg");

        // Create and export a package
        let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());
        let resource = Resource::new(
            "test.txt".to_string(),
            ResourceType::Text,
            b"Hello, World!".to_vec(),
        );
        package.resources.insert(resource.name.clone(), resource);

        let exporter = PackageExporter::new(Default::default());
        exporter.export_package(&package, &package_path).unwrap();

        // Import the package
        let importer = PackageImporter::new(ImportConfig::default());
        let imported = importer.import_package(&package_path).unwrap();

        assert_eq!(imported.manifest.name, "test_package");
        assert_eq!(imported.manifest.version, "1.0.0");
        assert_eq!(imported.resources.len(), 1);
        assert!(imported.resources.contains_key("test.txt"));
    }

    #[test]
    fn test_import_context() {
        let mut context = ImportContext::new();

        // Create a test package
        let temp_dir = TempDir::new().unwrap();
        let package_path = temp_dir.path().join("test.torshpkg");

        let mut package = Package::new("test".to_string(), "1.0.0".to_string());
        let resource = Resource::new(
            "model.bin".to_string(),
            ResourceType::Model,
            vec![1, 2, 3, 4],
        );
        package.resources.insert(resource.name.clone(), resource);

        let exporter = PackageExporter::new(Default::default());
        exporter.export_package(&package, &package_path).unwrap();

        // Import into context
        let package_id = context
            .import_package(&package_path, Some("test_pkg".to_string()))
            .unwrap();

        assert_eq!(package_id, "test_pkg");
        assert!(context.get_package("test_pkg").is_some());

        let packages = context.list_packages();
        assert_eq!(packages.len(), 1);
        assert_eq!(packages[0].0, "test_pkg");
    }
}
