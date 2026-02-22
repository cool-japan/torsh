//! ONNX model support for ToRSh Hub
//!
//! This module provides functionality to load, convert, and run ONNX models
//! within the ToRSh ecosystem.

use ort::{
    execution_providers::{
        CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
        DirectMLExecutionProvider, OpenVINOExecutionProvider, TensorRTExecutionProvider,
    },
    session::builder::{GraphOptimizationLevel, SessionBuilder},
    session::Session,
    value::Value,
};
// SciRS2 POLICY: Use UNIFIED ndarray access (v0.1.0-RC.1+)
// Note: For ONNX interop, we use (Vec<usize>, Vec<T>) tuple format instead of ndarray types
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// ONNX model wrapper for ToRSh integration
pub struct OnnxModel {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    metadata: OnnxModelMetadata,
}

/// ONNX model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxModelMetadata {
    pub model_name: String,
    pub version: String,
    pub description: Option<String>,
    pub producer: Option<String>,
    pub domain: Option<String>,
    pub opset_version: i64,
    pub input_shapes: Vec<InputShape>,
    pub output_shapes: Vec<OutputShape>,
}

/// Input shape information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputShape {
    pub name: String,
    pub shape: Vec<Option<i64>>, // None for dynamic dimensions
    pub data_type: String,
}

/// Output shape information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputShape {
    pub name: String,
    pub shape: Vec<Option<i64>>, // None for dynamic dimensions
    pub data_type: String,
}

/// ONNX Runtime configuration
#[derive(Debug)]
pub struct OnnxConfig {
    pub execution_providers: Vec<String>,
    pub graph_optimization_level: GraphOptimizationLevel,
    pub enable_profiling: bool,
    pub enable_mem_pattern: bool,
    pub enable_cpu_mem_arena: bool,
    pub inter_op_num_threads: Option<usize>,
    pub intra_op_num_threads: Option<usize>,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            execution_providers: vec!["CPUExecutionProvider".to_string()],
            graph_optimization_level: GraphOptimizationLevel::Level3,
            enable_profiling: false,
            enable_mem_pattern: true,
            enable_cpu_mem_arena: true,
            inter_op_num_threads: None,
            intra_op_num_threads: None,
        }
    }
}

/// Configure execution providers for ONNX Runtime session builder
fn configure_execution_providers(
    mut session_builder: SessionBuilder,
    execution_providers: &[String],
) -> ort::Result<SessionBuilder> {
    for provider in execution_providers {
        session_builder = match provider.as_str() {
            "CPUExecutionProvider" => session_builder
                .with_execution_providers([CPUExecutionProvider::default().build()])?,
            "CUDAExecutionProvider" => session_builder
                .with_execution_providers([CUDAExecutionProvider::default().build()])?,
            "TensorRTExecutionProvider" => session_builder
                .with_execution_providers([TensorRTExecutionProvider::default().build()])?,
            "OpenVINOExecutionProvider" => session_builder
                .with_execution_providers([OpenVINOExecutionProvider::default().build()])?,
            "CoreMLExecutionProvider" => session_builder
                .with_execution_providers([CoreMLExecutionProvider::default().build()])?,
            "DirectMLExecutionProvider" => session_builder
                .with_execution_providers([DirectMLExecutionProvider::default().build()])?,
            _ => {
                eprintln!(
                    "Warning: Unsupported execution provider '{}', skipping",
                    provider
                );
                session_builder
            }
        };
    }
    Ok(session_builder)
}

impl OnnxModel {
    /// Load an ONNX model from file
    pub fn from_file<P: AsRef<Path>>(path: P, config: Option<OnnxConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        let path = path.as_ref();

        if !path.exists() {
            return Err(TorshError::IoError(format!(
                "ONNX model file not found: {:?}",
                path
            )));
        }

        // Create session builder (environment is handled internally in newer ONNX Runtime)
        let mut session_builder = SessionBuilder::new()
            .map_err(|e| TorshError::Other(format!("Failed to create session builder: {}", e)))?;

        // Configure execution providers with the newer API
        session_builder =
            configure_execution_providers(session_builder, &config.execution_providers).map_err(
                |e| TorshError::Other(format!("Failed to configure execution providers: {}", e)),
            )?;

        // Set optimization level
        session_builder = session_builder
            .with_optimization_level(config.graph_optimization_level)
            .map_err(|e| TorshError::Other(format!("Failed to set optimization level: {}", e)))?;

        // Set threading options
        if let Some(inter_threads) = config.inter_op_num_threads {
            session_builder = session_builder
                .with_inter_threads(inter_threads)
                .map_err(|e| TorshError::Other(format!("Failed to set inter-op threads: {}", e)))?;
        }

        if let Some(intra_threads) = config.intra_op_num_threads {
            session_builder = session_builder
                .with_intra_threads(intra_threads)
                .map_err(|e| TorshError::Other(format!("Failed to set intra-op threads: {}", e)))?;
        }

        // Load the model
        let session = session_builder
            .commit_from_file(path)
            .map_err(|e| TorshError::Other(format!("Failed to load ONNX model: {}", e)))?;

        // Extract input and output information
        let input_names = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let output_names = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        // Extract metadata
        let metadata = Self::extract_metadata(&session, path)?;

        Ok(Self {
            session,
            input_names,
            output_names,
            metadata,
        })
    }

    /// Load an ONNX model from bytes
    pub fn from_bytes(model_bytes: &[u8], config: Option<OnnxConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();

        // Create session builder (environment is handled internally in newer ONNX Runtime)
        let mut session_builder = SessionBuilder::new()
            .map_err(|e| TorshError::Other(format!("Failed to create session builder: {}", e)))?;

        // Configure execution providers with the newer API
        session_builder =
            configure_execution_providers(session_builder, &config.execution_providers).map_err(
                |e| TorshError::Other(format!("Failed to configure execution providers: {}", e)),
            )?;

        // Set optimization level
        session_builder = session_builder
            .with_optimization_level(config.graph_optimization_level)
            .map_err(|e| TorshError::Other(format!("Failed to set optimization level: {}", e)))?;

        // Load the model from bytes
        let session = session_builder
            .commit_from_memory(model_bytes)
            .map_err(|e| {
                TorshError::Other(format!("Failed to load ONNX model from memory: {}", e))
            })?;

        // Extract input and output information
        let input_names = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let output_names = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        // Create basic metadata (without file path)
        let metadata = OnnxModelMetadata {
            model_name: "onnx_model".to_string(),
            version: "1.0".to_string(),
            description: Some("ONNX model loaded from memory".to_string()),
            producer: None,
            domain: None,
            opset_version: 1,
            input_shapes: session
                .inputs
                .iter()
                .map(|input| InputShape {
                    name: input.name.clone(),
                    shape: Self::extract_shape_from_value_type(&input.input_type),
                    data_type: format!("{:?}", &input.input_type),
                })
                .collect(),
            output_shapes: session
                .outputs
                .iter()
                .map(|output| OutputShape {
                    name: output.name.clone(),
                    shape: Self::extract_shape_from_value_type(&output.output_type),
                    data_type: format!("{:?}", &output.output_type),
                })
                .collect(),
        };

        Ok(Self {
            session,
            input_names,
            output_names,
            metadata,
        })
    }

    /// Run inference on the model
    pub fn forward(
        &mut self,
        inputs: &HashMap<String, Tensor<f32>>,
    ) -> Result<HashMap<String, Tensor<f32>>> {
        // Convert ToRSh tensors to ONNX values
        let mut onnx_inputs = HashMap::new();

        for input_name in &self.input_names {
            let tensor = inputs.get(input_name).ok_or_else(|| {
                TorshError::InvalidArgument(format!("Missing input: {}", input_name))
            })?;

            let onnx_value = self.tensor_to_onnx_value(tensor)?;
            onnx_inputs.insert(input_name.clone(), onnx_value);
        }

        // Run inference
        let outputs = {
            let outputs = self
                .session
                .run(onnx_inputs)
                .map_err(|e| TorshError::Other(format!("ONNX inference failed: {}", e)))?;

            // Collect outputs while the mutable borrow is active
            outputs.into_iter().collect::<Vec<_>>()
        };

        // Convert ONNX values back to ToRSh tensors
        let mut result = HashMap::new();
        let output_names = self.output_names.clone();
        for (i, (_name, output)) in outputs.into_iter().enumerate() {
            let output_name = &output_names[i];
            let tensor = Self::convert_onnx_value_to_tensor(output)?;
            result.insert(output_name.clone(), tensor);
        }

        Ok(result)
    }

    /// Get input names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Get model metadata
    pub fn metadata(&self) -> &OnnxModelMetadata {
        &self.metadata
    }

    /// Convert ToRSh tensor to ONNX value
    fn tensor_to_onnx_value(&self, tensor: &Tensor<f32>) -> Result<Value> {
        // Get tensor shape and convert to Vec<f32>
        let binding = tensor.shape();
        let shape: Vec<usize> = binding.dims().to_vec();
        let data: Vec<f32> = tensor.to_vec()?;

        // Convert to ONNX value using (shape, data) tuple format
        // This avoids ndarray version conflicts with ort crate
        Ok(Value::from_array((shape, data))
            .map_err(|e| TorshError::Other(format!("Failed to create ONNX value: {}", e)))?
            .into())
    }

    /// Convert ONNX value to ToRSh tensor
    fn convert_onnx_value_to_tensor(value: Value) -> Result<Tensor<f32>> {
        // Extract array from ONNX value
        let array = value
            .try_extract_tensor::<f32>()
            .map_err(|e| TorshError::Other(format!("Failed to extract ONNX value: {}", e)))?;

        // Get shape and data
        let (shape, data) = array;
        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let data: Vec<f32> = data.to_vec();

        // Create ToRSh tensor using the tensor creation API
        use torsh_tensor::creation::from_vec;
        from_vec(data, &shape, torsh_core::DeviceType::Cpu)
    }

    /// Extract metadata from ONNX session
    fn extract_metadata(session: &Session, model_path: &Path) -> Result<OnnxModelMetadata> {
        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("onnx_model")
            .to_string();

        let input_shapes = session
            .inputs
            .iter()
            .map(|input| InputShape {
                name: input.name.clone(),
                shape: Self::extract_shape_from_value_type(&input.input_type),
                data_type: format!("{:?}", &input.input_type),
            })
            .collect();

        let output_shapes = session
            .outputs
            .iter()
            .map(|output| OutputShape {
                name: output.name.clone(),
                shape: Self::extract_shape_from_value_type(&output.output_type),
                data_type: format!("{:?}", &output.output_type),
            })
            .collect();

        Ok(OnnxModelMetadata {
            model_name,
            version: "1.0".to_string(),
            description: Some(format!("ONNX model loaded from {:?}", model_path)),
            producer: None,
            domain: None,
            opset_version: 1,
            input_shapes,
            output_shapes,
        })
    }

    /// Extract shape dimensions from ONNX ValueType
    fn extract_shape_from_value_type(value_type: &ort::value::ValueType) -> Vec<Option<i64>> {
        match value_type {
            ort::value::ValueType::Tensor {
                ty: _,
                shape,
                dimension_symbols: _,
            } => {
                // Extract dimensions from the tensor type shape
                shape
                    .iter()
                    .map(|&dim| {
                        if dim == -1 {
                            None // Dynamic dimension
                        } else {
                            Some(dim)
                        }
                    })
                    .collect()
            }
            _ => vec![], // Non-tensor types have no shape
        }
    }
}

/// ONNX model loader utility functions
pub struct OnnxLoader;

impl OnnxLoader {
    /// Download and load ONNX model from URL
    pub async fn from_url(url: &str, config: Option<OnnxConfig>) -> Result<OnnxModel> {
        use crate::download::download_file;

        // Create temporary file for the model
        let temp_dir = tempfile::tempdir()
            .map_err(|e| TorshError::IoError(format!("Failed to create temp directory: {}", e)))?;

        let model_path = temp_dir.path().join("model.onnx");

        // Download the model
        download_file(url, &model_path, true)?;

        // Load the model
        OnnxModel::from_file(&model_path, config)
    }

    /// Load ONNX model from ToRSh Hub
    pub fn from_hub(repo: &str, model_name: &str, config: Option<OnnxConfig>) -> Result<OnnxModel> {
        use crate::{download_repo, parse_repo_info, HubConfig};

        let hub_config = HubConfig::default();

        // Parse repository information
        let (owner, repo_name, branch) = parse_repo_info(repo)?;

        // Download repository
        let repo_dir = download_repo(&owner, &repo_name, &branch, &hub_config)?;

        // Look for ONNX model file
        let model_path = repo_dir.join(format!("{}.onnx", model_name));
        if !model_path.exists() {
            return Err(TorshError::IoError(format!(
                "ONNX model file not found: {:?}",
                model_path
            )));
        }

        // Load the model
        OnnxModel::from_file(&model_path, config)
    }

    /// List available ONNX models in a repository
    pub fn list_models_in_repo(repo_dir: &Path) -> Result<Vec<String>> {
        let mut models = Vec::new();

        if !repo_dir.exists() {
            return Ok(models);
        }

        // Look for .onnx files
        for entry in std::fs::read_dir(repo_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(extension) = path.extension() {
                if extension == "onnx" {
                    if let Some(file_name) = path.file_stem().and_then(|s| s.to_str()) {
                        models.push(file_name.to_string());
                    }
                }
            }
        }

        Ok(models)
    }

    /// Validate ONNX model file
    pub fn validate_model(path: &Path) -> Result<bool> {
        // Try to load the model to validate it
        match OnnxModel::from_file(path, None) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

/// Convert ONNX model to ToRSh Module (wrapper)
pub struct OnnxToTorshWrapper {
    onnx_model: RwLock<OnnxModel>,
}

impl OnnxToTorshWrapper {
    pub fn new(onnx_model: OnnxModel) -> Self {
        Self {
            onnx_model: RwLock::new(onnx_model),
        }
    }
}

impl torsh_nn::Module for OnnxToTorshWrapper {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        // For simplicity, assume single input/output
        let (input_name, output_name) = {
            let onnx_model = self.onnx_model.read().expect("lock should not be poisoned");
            if onnx_model.input_names.len() != 1 || onnx_model.output_names.len() != 1 {
                return Err(TorshError::InvalidArgument(
                    "OnnxToTorshWrapper only supports models with single input and output"
                        .to_string(),
                ));
            }
            (
                onnx_model.input_names[0].clone(),
                onnx_model.output_names[0].clone(),
            )
        };

        let mut inputs = HashMap::new();
        inputs.insert(input_name, input.clone());

        let outputs = self
            .onnx_model
            .write()
            .expect("lock should not be poisoned")
            .forward(&inputs)?;

        Ok(outputs
            .get(&output_name)
            .ok_or_else(|| TorshError::Other("Missing output tensor".to_string()))?
            .clone())
    }

    fn parameters(&self) -> std::collections::HashMap<String, torsh_nn::Parameter> {
        // ONNX models don't expose parameters directly
        std::collections::HashMap::new()
    }

    fn train(&mut self) {
        // ONNX models are inference-only
    }

    fn eval(&mut self) {
        // ONNX models are always in eval mode
    }

    fn training(&self) -> bool {
        false
    }

    fn load_state_dict(
        &mut self,
        _state_dict: &std::collections::HashMap<String, Tensor<f32>>,
        _strict: bool,
    ) -> Result<()> {
        Err(TorshError::Other(
            "ONNX models don't support state dict loading".to_string(),
        ))
    }

    fn state_dict(&self) -> std::collections::HashMap<String, Tensor<f32>> {
        // ONNX models don't support state dict saving
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_onnx_config_default() {
        let config = OnnxConfig::default();
        assert_eq!(config.execution_providers.len(), 1);
        assert!(matches!(
            config.graph_optimization_level,
            GraphOptimizationLevel::Level3
        ));
        assert!(!config.enable_profiling);
        assert!(config.enable_mem_pattern);
        assert!(config.enable_cpu_mem_arena);
    }

    #[test]
    fn test_onnx_loader_validate_model_nonexistent() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("onnx");

        // File doesn't exist, should return Ok(false)
        let result = OnnxLoader::validate_model(&path);
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_input_output_shapes() {
        let input_shape = InputShape {
            name: "input".to_string(),
            shape: vec![Some(1), Some(3), Some(224), Some(224)],
            data_type: "Float".to_string(),
        };

        assert_eq!(input_shape.name, "input");
        assert_eq!(input_shape.shape.len(), 4);
        assert_eq!(input_shape.data_type, "Float");
    }

    #[test]
    fn test_onnx_metadata() {
        let metadata = OnnxModelMetadata {
            model_name: "test_model".to_string(),
            version: "1.0".to_string(),
            description: Some("Test ONNX model".to_string()),
            producer: Some("ToRSh".to_string()),
            domain: None,
            opset_version: 11,
            input_shapes: vec![],
            output_shapes: vec![],
        };

        assert_eq!(metadata.model_name, "test_model");
        assert_eq!(metadata.version, "1.0");
        assert_eq!(metadata.opset_version, 11);
    }
}
