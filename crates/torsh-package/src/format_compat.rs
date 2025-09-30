//! Format compatibility layer for different model package formats
//!
//! This module provides compatibility with external package formats including
//! PyTorch torch.package, HuggingFace Hub, ONNX models, and MLflow packages.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

use crate::package::Package;
use crate::resources::{Resource, ResourceType};

/// Supported external package formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackageFormat {
    /// PyTorch torch.package format
    PyTorch,
    /// HuggingFace Hub format
    HuggingFace,
    /// ONNX model format
    Onnx,
    /// MLflow model format
    MLflow,
    /// Native ToRSh format
    ToRSh,
}

/// Format-specific converter trait
pub trait FormatConverter {
    /// Convert from the external format to ToRSh Package
    fn import_from_format(&self, path: &std::path::Path) -> Result<Package>;

    /// Convert from ToRSh Package to the external format
    fn export_to_format(&self, package: &Package, path: &std::path::Path) -> Result<()>;

    /// Get the format this converter handles
    fn format(&self) -> PackageFormat;

    /// Validate if a path contains a valid package of this format
    fn is_valid_format(&self, path: &std::path::Path) -> bool;
}

/// PyTorch torch.package compatibility layer
pub struct PyTorchConverter {
    preserve_python_code: bool,
    extract_models: bool,
}

/// HuggingFace Hub compatibility layer
pub struct HuggingFaceConverter {
    include_tokenizer: bool,
    include_config: bool,
    model_type: Option<String>,
}

/// ONNX model compatibility layer
pub struct OnnxConverter {
    include_metadata: bool,
    optimize_for_inference: bool,
}

/// MLflow model compatibility layer
pub struct MLflowConverter {
    include_conda_env: bool,
    include_requirements: bool,
    flavor: Option<String>,
}

/// PyTorch package manifest structure (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PyTorchManifest {
    code_version: String,
    main_module: String,
    dependencies: Vec<String>,
    python_version: Option<String>,
}

/// HuggingFace model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HuggingFaceConfig {
    model_type: String,
    task: Option<String>,
    architectures: Option<Vec<String>>,
    tokenizer_class: Option<String>,
    vocab_size: Option<u64>,
}

/// ONNX model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OnnxMetadata {
    ir_version: i64,
    producer_name: String,
    producer_version: String,
    domain: String,
    model_version: i64,
    doc_string: String,
}

/// MLflow model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MLflowMetadata {
    artifact_path: String,
    flavors: HashMap<String, serde_json::Value>,
    model_uuid: String,
    run_id: String,
    utc_time_created: String,
    mlflow_version: String,
}

impl Default for PyTorchConverter {
    fn default() -> Self {
        Self {
            preserve_python_code: true,
            extract_models: true,
        }
    }
}

impl PyTorchConverter {
    /// Create a new PyTorch converter
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure whether to preserve Python code
    pub fn with_preserve_python_code(mut self, preserve: bool) -> Self {
        self.preserve_python_code = preserve;
        self
    }

    /// Configure whether to extract model weights
    pub fn with_extract_models(mut self, extract: bool) -> Self {
        self.extract_models = extract;
        self
    }

    /// Extract PyTorch package contents
    fn extract_pytorch_package(
        &self,
        path: &std::path::Path,
    ) -> Result<(PyTorchManifest, Vec<Resource>)> {
        let file = fs::File::open(path)
            .map_err(|e| TorshError::IoError(format!("Failed to open PyTorch package: {}", e)))?;

        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid ZIP archive: {}", e)))?;

        let mut manifest = None;
        let mut resources = Vec::new();

        for i in 0..archive.len() {
            let mut file = archive
                .by_index(i)
                .map_err(|e| TorshError::IoError(format!("Failed to read archive entry: {}", e)))?;

            let file_name = file.name().to_string();

            // Read file contents
            let mut contents = Vec::new();
            std::io::Read::read_to_end(&mut file, &mut contents)
                .map_err(|e| TorshError::IoError(format!("Failed to read file contents: {}", e)))?;

            if file_name == ".data/version" {
                // PyTorch package version info - convert to manifest
                let version_str = String::from_utf8(contents).map_err(|_| {
                    TorshError::InvalidArgument("Invalid UTF-8 in version file".to_string())
                })?;

                manifest = Some(PyTorchManifest {
                    code_version: version_str.trim().to_string(),
                    main_module: "main".to_string(), // Default
                    dependencies: Vec::new(),
                    python_version: None,
                });
            } else if file_name.ends_with(".py") && self.preserve_python_code {
                // Python source code
                resources.push(Resource {
                    name: file_name.clone(),
                    resource_type: ResourceType::Source,
                    data: contents,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("language".to_string(), "python".to_string());
                        meta.insert("original_format".to_string(), "pytorch".to_string());
                        meta
                    },
                });
            } else if file_name.ends_with(".pkl") && self.extract_models {
                // Pickle files (likely model weights)
                resources.push(Resource {
                    name: file_name.clone(),
                    resource_type: ResourceType::Model,
                    data: contents,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("format".to_string(), "pickle".to_string());
                        meta.insert("original_format".to_string(), "pytorch".to_string());
                        meta
                    },
                });
            } else {
                // Other data files
                resources.push(Resource {
                    name: file_name.clone(),
                    resource_type: ResourceType::Data,
                    data: contents,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("original_format".to_string(), "pytorch".to_string());
                        meta
                    },
                });
            }
        }

        let manifest = manifest.unwrap_or_else(|| PyTorchManifest {
            code_version: "1.0.0".to_string(),
            main_module: "main".to_string(),
            dependencies: Vec::new(),
            python_version: None,
        });

        Ok((manifest, resources))
    }
}

impl FormatConverter for PyTorchConverter {
    fn import_from_format(&self, path: &std::path::Path) -> Result<Package> {
        let (pytorch_manifest, resources) = self.extract_pytorch_package(path)?;

        let package_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_pytorch_model")
            .to_string();

        let mut package = Package::new(package_name, pytorch_manifest.code_version);

        // Add resources
        for resource in resources {
            package.add_resource(resource);
        }

        // Add PyTorch-specific metadata
        package
            .manifest_mut()
            .metadata
            .insert("original_format".to_string(), "pytorch".to_string());
        package
            .manifest_mut()
            .metadata
            .insert("main_module".to_string(), pytorch_manifest.main_module);

        if let Some(python_version) = pytorch_manifest.python_version {
            package
                .manifest_mut()
                .metadata
                .insert("python_version".to_string(), python_version);
        }

        // Add dependencies
        for dep in pytorch_manifest.dependencies {
            package.add_dependency(&dep, "*");
        }

        Ok(package)
    }

    fn export_to_format(&self, package: &Package, path: &std::path::Path) -> Result<()> {
        let file = fs::File::create(path)
            .map_err(|e| TorshError::IoError(format!("Failed to create output file: {}", e)))?;

        let mut zip = zip::ZipWriter::new(file);

        // Add version file
        let version_data = package.get_version().as_bytes();
        zip.start_file::<_, ()>(".data/version", zip::write::FileOptions::default())
            .map_err(|e| TorshError::IoError(format!("Failed to create version file: {}", e)))?;
        std::io::Write::write_all(&mut zip, version_data)
            .map_err(|e| TorshError::IoError(format!("Failed to write version data: {}", e)))?;

        // Add resources
        for (name, resource) in package.resources() {
            let file_path =
                if resource.resource_type == ResourceType::Source && name.ends_with(".py") {
                    format!("code/{}", name)
                } else if resource.resource_type == ResourceType::Model {
                    format!("data/{}", name)
                } else {
                    name.clone()
                };

            zip.start_file::<_, ()>(&file_path, zip::write::FileOptions::default())
                .map_err(|e| {
                    TorshError::IoError(format!("Failed to create file {}: {}", file_path, e))
                })?;
            std::io::Write::write_all(&mut zip, &resource.data).map_err(|e| {
                TorshError::IoError(format!("Failed to write resource data: {}", e))
            })?;
        }

        zip.finish()
            .map_err(|e| TorshError::IoError(format!("Failed to finalize ZIP archive: {}", e)))?;

        Ok(())
    }

    fn format(&self) -> PackageFormat {
        PackageFormat::PyTorch
    }

    fn is_valid_format(&self, path: &std::path::Path) -> bool {
        // Check if it's a ZIP file with PyTorch package structure
        if let Ok(file) = fs::File::open(path) {
            if let Ok(mut archive) = zip::ZipArchive::new(file) {
                // Look for characteristic PyTorch package files
                for i in 0..archive.len() {
                    if let Ok(file) = archive.by_index(i) {
                        let name = file.name();
                        if name == ".data/version"
                            || name.starts_with("code/")
                            || name.ends_with(".pkl")
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

impl Default for HuggingFaceConverter {
    fn default() -> Self {
        Self {
            include_tokenizer: true,
            include_config: true,
            model_type: None,
        }
    }
}

impl HuggingFaceConverter {
    /// Create a new HuggingFace converter
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure whether to include tokenizer files
    pub fn with_include_tokenizer(mut self, include: bool) -> Self {
        self.include_tokenizer = include;
        self
    }

    /// Configure whether to include model configuration
    pub fn with_include_config(mut self, include: bool) -> Self {
        self.include_config = include;
        self
    }

    /// Set expected model type
    pub fn with_model_type(mut self, model_type: String) -> Self {
        self.model_type = Some(model_type);
        self
    }

    /// Load HuggingFace model directory
    fn load_huggingface_model(
        &self,
        path: &std::path::Path,
    ) -> Result<(HuggingFaceConfig, Vec<Resource>)> {
        let model_dir = path;

        if !model_dir.is_dir() {
            return Err(TorshError::InvalidArgument(
                "HuggingFace path must be a directory".to_string(),
            ));
        }

        let mut config = None;
        let mut resources = Vec::new();

        // Read configuration file
        let config_path = model_dir.join("config.json");
        if config_path.exists() && self.include_config {
            let config_data = fs::read(&config_path)
                .map_err(|e| TorshError::IoError(format!("Failed to read config.json: {}", e)))?;

            config = Some(
                serde_json::from_slice::<HuggingFaceConfig>(&config_data).map_err(|e| {
                    TorshError::SerializationError(format!("Invalid config.json: {}", e))
                })?,
            );

            resources.push(Resource {
                name: "config.json".to_string(),
                resource_type: ResourceType::Config,
                data: config_data,
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("original_format".to_string(), "huggingface".to_string());
                    meta
                },
            });
        }

        // Read model files (pytorch_model.bin, model.safetensors, etc.)
        for entry in fs::read_dir(model_dir)
            .map_err(|e| TorshError::IoError(format!("Failed to read model directory: {}", e)))?
        {
            let entry = entry.map_err(|e| {
                TorshError::IoError(format!("Failed to read directory entry: {}", e))
            })?;
            let file_path = entry.path();
            let file_name = file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            if file_name.ends_with(".bin") || file_name.ends_with(".safetensors") {
                // Model weight files
                let data = fs::read(&file_path).map_err(|e| {
                    TorshError::IoError(format!("Failed to read {}: {}", file_name, e))
                })?;

                resources.push(Resource {
                    name: file_name.clone(),
                    resource_type: ResourceType::Model,
                    data,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("original_format".to_string(), "huggingface".to_string());
                        if file_name.ends_with(".safetensors") {
                            meta.insert("format".to_string(), "safetensors".to_string());
                        } else {
                            meta.insert("format".to_string(), "pytorch".to_string());
                        }
                        meta
                    },
                });
            } else if self.include_tokenizer
                && (file_name.starts_with("tokenizer") || file_name.ends_with(".json"))
            {
                // Tokenizer files
                let data = fs::read(&file_path).map_err(|e| {
                    TorshError::IoError(format!("Failed to read {}: {}", file_name, e))
                })?;

                resources.push(Resource {
                    name: file_name,
                    resource_type: ResourceType::Data,
                    data,
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("original_format".to_string(), "huggingface".to_string());
                        meta.insert("type".to_string(), "tokenizer".to_string());
                        meta
                    },
                });
            }
        }

        let config = config.unwrap_or_else(|| HuggingFaceConfig {
            model_type: self
                .model_type
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            task: None,
            architectures: None,
            tokenizer_class: None,
            vocab_size: None,
        });

        Ok((config, resources))
    }
}

impl FormatConverter for HuggingFaceConverter {
    fn import_from_format(&self, path: &std::path::Path) -> Result<Package> {
        let (hf_config, resources) = self.load_huggingface_model(path)?;

        let package_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_huggingface_model")
            .to_string();

        let mut package = Package::new(package_name, "1.0.0".to_string());

        // Add resources
        for resource in resources {
            package.add_resource(resource);
        }

        // Add HuggingFace-specific metadata
        package
            .manifest_mut()
            .metadata
            .insert("original_format".to_string(), "huggingface".to_string());
        package
            .manifest_mut()
            .metadata
            .insert("model_type".to_string(), hf_config.model_type);

        if let Some(task) = hf_config.task {
            package
                .manifest_mut()
                .metadata
                .insert("task".to_string(), task);
        }

        if let Some(architectures) = hf_config.architectures {
            package.manifest_mut().metadata.insert(
                "architectures".to_string(),
                serde_json::to_string(&architectures).unwrap_or_default(),
            );
        }

        Ok(package)
    }

    fn export_to_format(&self, package: &Package, path: &std::path::Path) -> Result<()> {
        let output_dir = path;

        if !output_dir.exists() {
            fs::create_dir_all(output_dir).map_err(|e| {
                TorshError::IoError(format!("Failed to create output directory: {}", e))
            })?;
        }

        // Export resources to appropriate files
        for (name, resource) in package.resources() {
            let file_path = output_dir.join(name);
            fs::write(&file_path, &resource.data)
                .map_err(|e| TorshError::IoError(format!("Failed to write {}: {}", name, e)))?;
        }

        // Create or update config.json if not present
        let config_path = output_dir.join("config.json");
        if !config_path.exists() {
            let default_config = HuggingFaceConfig {
                model_type: package
                    .metadata()
                    .metadata
                    .get("model_type")
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string()),
                task: package.metadata().metadata.get("task").cloned(),
                architectures: package
                    .metadata()
                    .metadata
                    .get("architectures")
                    .and_then(|s| serde_json::from_str(s).ok()),
                tokenizer_class: None,
                vocab_size: None,
            };

            let config_json = serde_json::to_string_pretty(&default_config).map_err(|e| {
                TorshError::SerializationError(format!("Failed to serialize config: {}", e))
            })?;

            fs::write(&config_path, config_json)
                .map_err(|e| TorshError::IoError(format!("Failed to write config.json: {}", e)))?;
        }

        Ok(())
    }

    fn format(&self) -> PackageFormat {
        PackageFormat::HuggingFace
    }

    fn is_valid_format(&self, path: &std::path::Path) -> bool {
        let model_dir = path;

        if !model_dir.is_dir() {
            return false;
        }

        // Check for characteristic HuggingFace files
        let config_path = model_dir.join("config.json");
        if config_path.exists() {
            return true;
        }

        // Check for model weight files
        if let Ok(entries) = fs::read_dir(model_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let file_name = entry.file_name();
                    let file_name_str = file_name.to_string_lossy();
                    if file_name_str.ends_with(".bin") || file_name_str.ends_with(".safetensors") {
                        return true;
                    }
                }
            }
        }

        false
    }
}

/// Format compatibility manager
pub struct FormatCompatibilityManager {
    converters: HashMap<PackageFormat, Box<dyn FormatConverter>>,
}

impl Default for FormatCompatibilityManager {
    fn default() -> Self {
        let mut manager = Self {
            converters: HashMap::new(),
        };

        // Register default converters
        manager.register_converter(Box::new(PyTorchConverter::new()));
        manager.register_converter(Box::new(HuggingFaceConverter::new()));

        manager
    }
}

impl FormatCompatibilityManager {
    /// Create a new format compatibility manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a format converter
    pub fn register_converter(&mut self, converter: Box<dyn FormatConverter>) {
        let format = converter.format();
        self.converters.insert(format, converter);
    }

    /// Auto-detect format and import package
    pub fn import_package(&self, path: &std::path::Path) -> Result<(PackageFormat, Package)> {
        for (format, converter) in &self.converters {
            if converter.is_valid_format(path) {
                let package = converter.import_from_format(path)?;
                return Ok((*format, package));
            }
        }

        Err(TorshError::InvalidArgument(
            "Unrecognized package format".to_string(),
        ))
    }

    /// Export package to specific format
    pub fn export_package(
        &self,
        package: &Package,
        format: PackageFormat,
        path: &std::path::Path,
    ) -> Result<()> {
        let converter = self.converters.get(&format).ok_or_else(|| {
            TorshError::InvalidArgument(format!("Unsupported export format: {:?}", format))
        })?;

        converter.export_to_format(package, path)
    }

    /// List supported formats
    pub fn supported_formats(&self) -> Vec<PackageFormat> {
        self.converters.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_pytorch_converter_format_detection() {
        let converter = PyTorchConverter::new();
        assert_eq!(converter.format(), PackageFormat::PyTorch);
    }

    #[test]
    fn test_huggingface_converter_creation() {
        let converter = HuggingFaceConverter::new()
            .with_include_tokenizer(false)
            .with_model_type("bert".to_string());

        assert_eq!(converter.format(), PackageFormat::HuggingFace);
        assert!(!converter.include_tokenizer);
        assert_eq!(converter.model_type, Some("bert".to_string()));
    }

    #[test]
    fn test_format_manager() {
        let manager = FormatCompatibilityManager::new();
        let formats = manager.supported_formats();

        assert!(formats.contains(&PackageFormat::PyTorch));
        assert!(formats.contains(&PackageFormat::HuggingFace));
    }

    #[test]
    fn test_huggingface_directory_validation() {
        let temp_dir = TempDir::new().unwrap();

        // Create a mock HuggingFace model directory
        let config_path = temp_dir.path().join("config.json");
        let mut config_file = fs::File::create(&config_path).unwrap();
        writeln!(
            config_file,
            r#"{{"model_type": "bert", "task": "text-classification"}}"#
        )
        .unwrap();

        let converter = HuggingFaceConverter::new();
        assert!(converter.is_valid_format(temp_dir.path()));
    }

    #[test]
    fn test_package_format_enum() {
        assert_eq!(PackageFormat::PyTorch, PackageFormat::PyTorch);
        assert_ne!(PackageFormat::PyTorch, PackageFormat::HuggingFace);
    }
}
