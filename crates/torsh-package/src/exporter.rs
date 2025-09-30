//! Package exporter functionality

use crate::{Package, PackageManifest, Resource, ResourceType};
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};
use zip::write::{ExtendedFileOptions, FileOptions, ZipWriter};
use zip::CompressionMethod;

/// Configuration for package export
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Compression method to use
    pub compression: CompressionMethod,

    /// Compression level (0-9)
    pub compression_level: Option<i64>,

    /// Include source code
    pub include_source: bool,

    /// Include debug information
    pub include_debug_info: bool,

    /// Sign the package
    pub sign_package: bool,

    /// Verbose output
    pub verbose: bool,

    /// Maximum package size in bytes (0 = unlimited)
    pub max_size: u64,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            compression: CompressionMethod::Deflated,
            compression_level: Some(6),
            include_source: false,
            include_debug_info: false,
            sign_package: false,
            verbose: false,
            max_size: 0,
        }
    }
}

/// Package exporter
pub struct PackageExporter {
    config: ExportConfig,
}

impl PackageExporter {
    /// Create a new exporter
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export a package to a file
    pub fn export_package<P: AsRef<Path>>(&self, package: &Package, path: P) -> Result<()> {
        let path = path.as_ref();

        // Validate package
        package
            .manifest
            .validate()
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid manifest: {}", e)))?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create zip file
        let file = File::create(path)?;
        let mut zip = ZipWriter::new(file);

        // Set compression options
        let options = FileOptions::<ExtendedFileOptions>::default()
            .compression_method(self.config.compression)
            .compression_level(self.config.compression_level);

        // Write manifest
        if self.config.verbose {
            println!("Writing manifest...");
        }
        self.write_manifest(&mut zip, &package.manifest, &options)?;

        // Write resources
        let mut total_size = 0u64;
        for (name, resource) in &package.resources {
            if self.config.verbose {
                println!("Writing resource: {}", name);
            }

            // Check size limit
            if self.config.max_size > 0 {
                total_size += resource.size() as u64;
                if total_size > self.config.max_size {
                    return Err(TorshError::InvalidArgument(format!(
                        "Package size ({} bytes) exceeds maximum allowed size ({} bytes)",
                        total_size, self.config.max_size
                    )));
                }
            }

            // Skip source if not included
            if resource.resource_type == ResourceType::Source && !self.config.include_source {
                continue;
            }

            self.write_resource(&mut zip, name, resource, &options)?;
        }

        // Finalize zip
        zip.finish()
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        if self.config.verbose {
            println!("Package exported successfully to: {:?}", path);
        }

        Ok(())
    }

    /// Write manifest to zip
    fn write_manifest<W: Write + io::Seek>(
        &self,
        zip: &mut ZipWriter<W>,
        manifest: &PackageManifest,
        options: &FileOptions<ExtendedFileOptions>,
    ) -> Result<()> {
        // Serialize manifest
        let json = serde_json::to_string_pretty(manifest)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;

        // Write to zip
        zip.start_file("MANIFEST.json", options.clone())
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        zip.write_all(json.as_bytes())?;

        Ok(())
    }

    /// Write resource to zip
    fn write_resource<W: Write + io::Seek>(
        &self,
        zip: &mut ZipWriter<W>,
        name: &str,
        resource: &Resource,
        options: &FileOptions<ExtendedFileOptions>,
    ) -> Result<()> {
        // Determine path in archive
        let archive_path = match resource.resource_type {
            ResourceType::Model => format!("models/{}", name),
            ResourceType::Source => format!("src/{}", name),
            ResourceType::Data => format!("data/{}", name),
            ResourceType::Config => format!("config/{}", name),
            ResourceType::Documentation => format!("docs/{}", name),
            _ => format!("resources/{}", name),
        };

        // Write to zip
        zip.start_file(&archive_path, options.clone())
            .map_err(|e| TorshError::IoError(e.to_string()))?;

        zip.write_all(&resource.data)?;

        // Write metadata if present
        if !resource.metadata.is_empty() {
            let metadata_path = format!("{}.metadata", archive_path);
            let metadata_json = serde_json::to_string(&resource.metadata)
                .map_err(|e| TorshError::SerializationError(e.to_string()))?;

            zip.start_file(&metadata_path, options.clone())
                .map_err(|e| TorshError::IoError(e.to_string()))?;

            zip.write_all(metadata_json.as_bytes())?;
        }

        Ok(())
    }
}

/// Export builder for convenient package creation and export
pub struct ExportBuilder {
    package: Package,
    config: ExportConfig,
    output_path: Option<PathBuf>,
}

impl ExportBuilder {
    /// Create a new export builder
    pub fn new(name: String, version: String) -> Self {
        Self {
            package: Package::new(name, version),
            config: ExportConfig::default(),
            output_path: None,
        }
    }

    /// Set export configuration
    pub fn with_config(mut self, config: ExportConfig) -> Self {
        self.config = config;
        self
    }

    /// Set output path
    pub fn output_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.output_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Add author
    pub fn author(mut self, author: String) -> Self {
        self.package.manifest.author = Some(author);
        self
    }

    /// Add description
    pub fn description(mut self, description: String) -> Self {
        self.package.manifest.description = Some(description);
        self
    }

    /// Add license
    pub fn license(mut self, license: String) -> Self {
        self.package.manifest.license = Some(license);
        self
    }

    /// Add a module (temporarily disabled - requires torsh-nn)
    #[cfg(feature = "with-nn")]
    pub fn add_module<M: torsh_nn::Module>(mut self, name: &str, module: &M) -> Result<Self> {
        self.package
            .add_module(name, module, self.config.include_source)?;
        Ok(self)
    }

    /// Add a data file
    pub fn add_data_file<P: AsRef<Path>>(mut self, name: &str, path: P) -> Result<Self> {
        self.package.add_data_file(name, path)?;
        Ok(self)
    }

    /// Add a resource
    pub fn add_resource(mut self, resource: Resource) -> Self {
        self.package
            .resources
            .insert(resource.name.clone(), resource);
        self
    }

    /// Add metadata
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.package.manifest.metadata.insert(key, value);
        self
    }

    /// Build and export the package
    pub fn export(self) -> Result<PathBuf> {
        let output_path = self.output_path.unwrap_or_else(|| {
            PathBuf::from(format!(
                "{}-{}.torshpkg",
                self.package.manifest.name, self.package.manifest.version
            ))
        });

        let exporter = PackageExporter::new(self.config);
        exporter.export_package(&self.package, &output_path)?;

        Ok(output_path)
    }
}

/// Quick export function for a single module (temporarily disabled - requires torsh-nn)
#[cfg(feature = "with-nn")]
pub fn export_single_module<M: torsh_nn::Module, P: AsRef<Path>>(
    module: &M,
    name: &str,
    version: &str,
    output_path: P,
) -> Result<()> {
    ExportBuilder::new(name.to_string(), version.to_string())
        .add_module("main", module)?
        .output_path(output_path)
        .export()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_export_config() {
        let config = ExportConfig::default();
        assert_eq!(config.compression, CompressionMethod::Deflated);
        assert_eq!(config.compression_level, Some(6));
        assert!(!config.include_source);
    }

    #[test]
    fn test_package_export() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test.torshpkg");

        let mut package = Package::new("test_package".to_string(), "1.0.0".to_string());

        // Add a test resource
        let resource = Resource::new(
            "test.txt".to_string(),
            ResourceType::Text,
            b"Hello, World!".to_vec(),
        );
        package.resources.insert(resource.name.clone(), resource);

        let exporter = PackageExporter::new(ExportConfig::default());
        exporter.export_package(&package, &output_path).unwrap();

        assert!(output_path.exists());
    }

    #[test]
    fn test_export_builder() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("test.torshpkg");

        let path = ExportBuilder::new("test".to_string(), "1.0.0".to_string())
            .author("Test Author".to_string())
            .description("Test package".to_string())
            .license("MIT".to_string())
            .output_path(&output_path)
            .export()
            .unwrap();

        assert_eq!(path, output_path);
        assert!(path.exists());
    }
}
