//! Format compatibility layer for different model package formats
//!
//! This module provides compatibility with external package formats including
//! PyTorch torch.package, HuggingFace Hub, ONNX models, and MLflow packages.

use std::collections::HashMap;
use std::fs;

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

impl Default for OnnxConverter {
    fn default() -> Self {
        Self {
            include_metadata: true,
            optimize_for_inference: false,
        }
    }
}

impl OnnxConverter {
    /// Create a new ONNX converter
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure whether to include model metadata
    pub fn with_include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Configure optimization for inference
    pub fn with_optimize_for_inference(mut self, optimize: bool) -> Self {
        self.optimize_for_inference = optimize;
        self
    }

    /// Extract ONNX model metadata
    fn extract_onnx_metadata(&self, path: &std::path::Path) -> Result<OnnxMetadata> {
        // For now, return a basic metadata structure
        // In a real implementation, you would parse the ONNX protobuf format
        Ok(OnnxMetadata {
            ir_version: 8,
            producer_name: "torsh-package".to_string(),
            producer_version: "1.0.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            doc_string: format!("ONNX model imported from {:?}", path),
        })
    }
}

impl FormatConverter for OnnxConverter {
    fn import_from_format(&self, path: &std::path::Path) -> Result<Package> {
        let model_data = fs::read(path)
            .map_err(|e| TorshError::IoError(format!("Failed to read ONNX model: {}", e)))?;

        let package_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_onnx_model")
            .to_string();

        let mut package = Package::new(package_name, "1.0.0".to_string());

        // Add ONNX model as a resource
        let model_resource = Resource {
            name: "model.onnx".to_string(),
            resource_type: ResourceType::Model,
            data: model_data,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("original_format".to_string(), "onnx".to_string());
                meta.insert("format".to_string(), "onnx".to_string());
                meta
            },
        };
        package.add_resource(model_resource);

        // Add metadata if requested
        if self.include_metadata {
            let onnx_metadata = self.extract_onnx_metadata(path)?;
            package.manifest_mut().metadata.insert(
                "onnx_ir_version".to_string(),
                onnx_metadata.ir_version.to_string(),
            );
            package
                .manifest_mut()
                .metadata
                .insert("onnx_producer".to_string(), onnx_metadata.producer_name);
            package.manifest_mut().metadata.insert(
                "onnx_producer_version".to_string(),
                onnx_metadata.producer_version,
            );
        }

        package
            .manifest_mut()
            .metadata
            .insert("original_format".to_string(), "onnx".to_string());

        Ok(package)
    }

    fn export_to_format(&self, package: &Package, path: &std::path::Path) -> Result<()> {
        // Find the ONNX model resource
        let model_resource = package
            .resources()
            .iter()
            .find(|(_, resource)| {
                resource.resource_type == ResourceType::Model
                    && (resource.name.ends_with(".onnx")
                        || resource
                            .metadata
                            .get("format")
                            .map_or(false, |f| f == "onnx"))
            })
            .map(|(_, resource)| resource)
            .ok_or_else(|| {
                TorshError::InvalidArgument("No ONNX model found in package".to_string())
            })?;

        // Write the ONNX model to file
        fs::write(path, &model_resource.data)
            .map_err(|e| TorshError::IoError(format!("Failed to write ONNX model: {}", e)))?;

        Ok(())
    }

    fn format(&self) -> PackageFormat {
        PackageFormat::Onnx
    }

    fn is_valid_format(&self, path: &std::path::Path) -> bool {
        if let Ok(file) = fs::File::open(path) {
            use std::io::Read;
            let mut buffer = [0u8; 16];
            let mut reader = std::io::BufReader::new(file);

            // Check for ONNX magic bytes (protobuf)
            if reader.read_exact(&mut buffer).is_ok() {
                // ONNX files typically start with protobuf headers
                // This is a simplified check; real implementation would parse protobuf
                return path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map_or(false, |e| e == "onnx");
            }
        }
        false
    }
}

impl Default for MLflowConverter {
    fn default() -> Self {
        Self {
            include_conda_env: true,
            include_requirements: true,
            flavor: None,
        }
    }
}

impl MLflowConverter {
    /// Create a new MLflow converter
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure whether to include conda environment
    pub fn with_include_conda_env(mut self, include: bool) -> Self {
        self.include_conda_env = include;
        self
    }

    /// Configure whether to include requirements.txt
    pub fn with_include_requirements(mut self, include: bool) -> Self {
        self.include_requirements = include;
        self
    }

    /// Set MLflow flavor
    pub fn with_flavor(mut self, flavor: String) -> Self {
        self.flavor = Some(flavor);
        self
    }

    /// Load MLflow model directory
    fn load_mlflow_model(&self, path: &std::path::Path) -> Result<(MLflowMetadata, Vec<Resource>)> {
        if !path.is_dir() {
            return Err(TorshError::InvalidArgument(
                "MLflow path must be a directory".to_string(),
            ));
        }

        let mut metadata = None;
        let mut resources = Vec::new();

        // Read MLmodel file
        let mlmodel_path = path.join("MLmodel");
        if mlmodel_path.exists() {
            let mlmodel_data = fs::read_to_string(&mlmodel_path)
                .map_err(|e| TorshError::IoError(format!("Failed to read MLmodel: {}", e)))?;

            // Parse MLmodel (YAML format)
            // For simplicity, we'll create a basic metadata structure
            metadata = Some(MLflowMetadata {
                artifact_path: path.to_string_lossy().to_string(),
                flavors: HashMap::new(),
                model_uuid: uuid::Uuid::new_v4().to_string(),
                run_id: "imported".to_string(),
                utc_time_created: chrono::Utc::now().to_rfc3339(),
                mlflow_version: "2.0.0".to_string(),
            });

            resources.push(Resource {
                name: "MLmodel".to_string(),
                resource_type: ResourceType::Config,
                data: mlmodel_data.into_bytes(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("original_format".to_string(), "mlflow".to_string());
                    meta
                },
            });
        }

        // Read model files from subdirectories
        for entry in fs::read_dir(path)
            .map_err(|e| TorshError::IoError(format!("Failed to read MLflow directory: {}", e)))?
        {
            let entry = entry.map_err(|e| {
                TorshError::IoError(format!("Failed to read directory entry: {}", e))
            })?;
            let file_path = entry.path();

            if file_path.is_file() {
                let file_name = file_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_string();

                if file_name != "MLmodel" {
                    let data = fs::read(&file_path).map_err(|e| {
                        TorshError::IoError(format!("Failed to read {}: {}", file_name, e))
                    })?;

                    let resource_type = if file_name.ends_with(".pkl")
                        || file_name.ends_with(".pt")
                        || file_name.ends_with(".h5")
                    {
                        ResourceType::Model
                    } else if file_name.ends_with(".json") || file_name.ends_with(".yaml") {
                        ResourceType::Config
                    } else if file_name == "requirements.txt" || file_name == "conda.yaml" {
                        ResourceType::Documentation
                    } else {
                        ResourceType::Data
                    };

                    resources.push(Resource {
                        name: file_name,
                        resource_type,
                        data,
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("original_format".to_string(), "mlflow".to_string());
                            meta
                        },
                    });
                }
            }
        }

        let metadata = metadata.unwrap_or_else(|| MLflowMetadata {
            artifact_path: path.to_string_lossy().to_string(),
            flavors: HashMap::new(),
            model_uuid: uuid::Uuid::new_v4().to_string(),
            run_id: "imported".to_string(),
            utc_time_created: chrono::Utc::now().to_rfc3339(),
            mlflow_version: "2.0.0".to_string(),
        });

        Ok((metadata, resources))
    }
}

impl FormatConverter for MLflowConverter {
    fn import_from_format(&self, path: &std::path::Path) -> Result<Package> {
        let (mlflow_metadata, resources) = self.load_mlflow_model(path)?;

        let package_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("imported_mlflow_model")
            .to_string();

        let mut package = Package::new(package_name, "1.0.0".to_string());

        // Add resources
        for resource in resources {
            package.add_resource(resource);
        }

        // Add MLflow-specific metadata
        package
            .manifest_mut()
            .metadata
            .insert("original_format".to_string(), "mlflow".to_string());
        package
            .manifest_mut()
            .metadata
            .insert("mlflow_version".to_string(), mlflow_metadata.mlflow_version);
        package
            .manifest_mut()
            .metadata
            .insert("model_uuid".to_string(), mlflow_metadata.model_uuid);
        package
            .manifest_mut()
            .metadata
            .insert("run_id".to_string(), mlflow_metadata.run_id);

        if let Some(flavor) = &self.flavor {
            package
                .manifest_mut()
                .metadata
                .insert("flavor".to_string(), flavor.clone());
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

        // Create MLmodel file if not present
        let mlmodel_path = output_dir.join("MLmodel");
        if !mlmodel_path.exists() {
            let mlmodel_content = format!(
                r#"artifact_path: {}
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.pyfunc.model
    python_version: 3.9
model_uuid: {}
run_id: {}
utc_time_created: '{}'
mlflow_version: 2.0.0
"#,
                output_dir.to_string_lossy(),
                package
                    .metadata()
                    .metadata
                    .get("model_uuid")
                    .cloned()
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                package
                    .metadata()
                    .metadata
                    .get("run_id")
                    .cloned()
                    .unwrap_or_else(|| "exported".to_string()),
                chrono::Utc::now().to_rfc3339()
            );

            fs::write(&mlmodel_path, mlmodel_content)
                .map_err(|e| TorshError::IoError(format!("Failed to write MLmodel: {}", e)))?;
        }

        Ok(())
    }

    fn format(&self) -> PackageFormat {
        PackageFormat::MLflow
    }

    fn is_valid_format(&self, path: &std::path::Path) -> bool {
        if !path.is_dir() {
            return false;
        }

        // Check for MLmodel file
        let mlmodel_path = path.join("MLmodel");
        mlmodel_path.exists()
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
        manager.register_converter(Box::new(OnnxConverter::new()));
        manager.register_converter(Box::new(MLflowConverter::new()));

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
