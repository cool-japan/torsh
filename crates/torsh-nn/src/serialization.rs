//! Model serialization and deserialization for ToRSh neural networks
//!
//! Note: This module requires std for file operations and is only available with the "std" feature.

#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use serde_json;
#[cfg(feature = "std")]
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    path::Path,
};
#[cfg(feature = "std")]
use torsh_core::error::{Result, TorshError};
#[cfg(feature = "std")]
use torsh_tensor::Tensor;

#[cfg(feature = "safetensors")]
use safetensors::SafeTensors;

/// Represents a serializable model state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// Model parameters (weights and biases)
    pub parameters: HashMap<String, SerializableTensor>,
    /// Model hyperparameters and configuration
    pub config: HashMap<String, SerializableValue>,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Metadata for the serialized model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model architecture name
    pub architecture: String,
    /// Model version
    pub version: String,
    /// Timestamp of serialization
    pub created_at: String,
    /// Framework version used
    pub framework_version: String,
    /// Additional tags or labels
    pub tags: Vec<String>,
}

/// Serializable representation of a tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableTensor {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Tensor data type
    pub dtype: String,
    /// Flattened tensor data
    pub data: Vec<f32>,
    /// Whether the tensor requires gradients
    pub requires_grad: bool,
}

/// Serializable value for hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SerializableValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<SerializableValue>),
}

impl From<Tensor<f32>> for SerializableTensor {
    fn from(tensor: Tensor<f32>) -> Self {
        let shape = tensor.shape().dims().to_vec();
        let data = tensor.to_vec().expect("Failed to extract tensor data");
        let requires_grad = tensor.requires_grad();

        Self {
            shape,
            dtype: "f32".to_string(),
            data,
            requires_grad,
        }
    }
}

impl TryFrom<SerializableTensor> for Tensor<f32> {
    type Error = TorshError;

    fn try_from(serializable: SerializableTensor) -> Result<Self> {
        let tensor = Tensor::from_vec(serializable.data, &serializable.shape)?;
        if serializable.requires_grad {
            Ok(tensor.requires_grad_(true))
        } else {
            Ok(tensor)
        }
    }
}

impl ModelState {
    /// Create a new model state
    pub fn new(architecture: String) -> Self {
        let metadata = ModelMetadata {
            architecture,
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            tags: Vec::new(),
        };

        Self {
            parameters: HashMap::new(),
            config: HashMap::new(),
            metadata,
        }
    }

    /// Add a parameter tensor to the model state
    pub fn add_parameter(&mut self, name: String, tensor: Tensor<f32>) {
        self.parameters
            .insert(name, SerializableTensor::from(tensor));
    }

    /// Add a configuration value to the model state
    pub fn add_config<T>(&mut self, key: String, value: T)
    where
        T: Into<SerializableValue>,
    {
        self.config.insert(key, value.into());
    }

    /// Get a parameter tensor by name
    pub fn get_parameter(&self, name: &str) -> Option<Result<Tensor<f32>>> {
        self.parameters.get(name).map(|t| t.clone().try_into())
    }

    /// Get a configuration value by name
    pub fn get_config(&self, key: &str) -> Option<&SerializableValue> {
        self.config.get(key)
    }

    /// Save the model state to a JSON file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path)?;
        let json = serde_json::to_string_pretty(self)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load the model state from a JSON file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let model_state: ModelState = serde_json::from_str(&contents)?;
        Ok(model_state)
    }

    /// Save the model state to binary format (more efficient)
    pub fn save_to_binary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path)?;
        let data = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        file.write_all(&data)?;
        Ok(())
    }

    /// Load the model state from binary format
    pub fn load_from_binary<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        let (model_state, _): (ModelState, usize) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        Ok(model_state)
    }

    /// Save parameters to SafeTensors format
    #[cfg(feature = "safetensors")]
    pub fn save_to_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::BTreeMap;

        let mut tensors = BTreeMap::new();

        for (name, serializable_tensor) in &self.parameters {
            let data_bytes: &[u8] = bytemuck::cast_slice(&serializable_tensor.data);
            let shape: Vec<usize> = serializable_tensor.shape.clone();

            let tensor_view = TensorView::new(Dtype::F32, shape, data_bytes)
                .map_err(|e| TorshError::SerializationError(format!("SafeTensors error: {}", e)))?;

            tensors.insert(name.clone(), tensor_view);
        }

        // Create metadata with config information
        let metadata = serde_json::to_string(&self.metadata)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        let mut metadata_map = HashMap::new();
        metadata_map.insert("torsh_metadata".to_string(), metadata);

        if !self.config.is_empty() {
            let config_json = serde_json::to_string(&self.config)
                .map_err(|e| TorshError::SerializationError(e.to_string()))?;
            metadata_map.insert("torsh_config".to_string(), config_json);
        }

        let safetensors_data =
            safetensors::serialize(&tensors, Some(metadata_map)).map_err(|e| {
                TorshError::SerializationError(format!("SafeTensors serialization error: {}", e))
            })?;

        std::fs::write(path, safetensors_data)?;
        Ok(())
    }

    /// Load parameters from SafeTensors format
    #[cfg(feature = "safetensors")]
    pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = std::fs::read(path)?;

        // Use SafeTensors::deserialize method
        let safetensors = SafeTensors::deserialize(&data).map_err(|e| {
            TorshError::SerializationError(format!("SafeTensors deserialization error: {}", e))
        })?;

        let mut parameters = HashMap::new();

        for (name, tensor_view) in safetensors.tensors() {
            if tensor_view.dtype() != safetensors::tensor::Dtype::F32 {
                return Err(TorshError::SerializationError(format!(
                    "Unsupported dtype: {:?}. Only F32 is currently supported.",
                    tensor_view.dtype()
                )));
            }

            let shape = tensor_view.shape().to_vec();
            let data: Vec<f32> = bytemuck::cast_slice(tensor_view.data()).to_vec();

            let serializable_tensor = SerializableTensor {
                shape,
                dtype: "f32".to_string(),
                data,
                requires_grad: false, // SafeTensors doesn't store grad info
            };

            parameters.insert(name.to_string(), serializable_tensor);
        }

        // Create default metadata since we can't access it easily from this version
        let metadata = ModelMetadata {
            architecture: "unknown".to_string(),
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
            tags: vec!["safetensors".to_string()],
        };

        // No config available in this implementation
        let config = HashMap::new();

        Ok(ModelState {
            parameters,
            config,
            metadata,
        })
    }
}

// Implement conversions for common types to SerializableValue
impl From<bool> for SerializableValue {
    fn from(value: bool) -> Self {
        SerializableValue::Bool(value)
    }
}

impl From<i32> for SerializableValue {
    fn from(value: i32) -> Self {
        SerializableValue::Int(value as i64)
    }
}

impl From<i64> for SerializableValue {
    fn from(value: i64) -> Self {
        SerializableValue::Int(value)
    }
}

impl From<f32> for SerializableValue {
    fn from(value: f32) -> Self {
        SerializableValue::Float(value as f64)
    }
}

impl From<f64> for SerializableValue {
    fn from(value: f64) -> Self {
        SerializableValue::Float(value)
    }
}

impl From<String> for SerializableValue {
    fn from(value: String) -> Self {
        SerializableValue::String(value)
    }
}

impl From<&str> for SerializableValue {
    fn from(value: &str) -> Self {
        SerializableValue::String(value.to_string())
    }
}

/// Trait for models that can be serialized
pub trait Serializable {
    /// Convert the model to a serializable state
    fn to_state(&self) -> ModelState;

    /// Load the model from a serializable state
    fn from_state(state: &ModelState) -> Result<Self>
    where
        Self: Sized;

    /// Save the model to a file
    fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let state = self.to_state();
        state.save_to_file(path)
    }

    /// Load the model from a file
    fn load<P: AsRef<Path>>(path: P) -> Result<Self>
    where
        Self: Sized,
    {
        let state = ModelState::load_from_file(path)?;
        Self::from_state(&state)
    }

    /// Save the model to binary format
    fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let state = self.to_state();
        state.save_to_binary(path)
    }

    /// Load the model from binary format
    fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self>
    where
        Self: Sized,
    {
        let state = ModelState::load_from_binary(path)?;
        Self::from_state(&state)
    }

    /// Save the model to SafeTensors format
    #[cfg(feature = "safetensors")]
    fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let state = self.to_state();
        state.save_to_safetensors(path)
    }

    /// Load the model from SafeTensors format
    #[cfg(feature = "safetensors")]
    fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<Self>
    where
        Self: Sized,
    {
        let state = ModelState::load_from_safetensors(path)?;
        Self::from_state(&state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use torsh_tensor::creation;

    #[test]
    fn test_model_state_creation() -> Result<()> {
        let mut state = ModelState::new("test_model".to_string());

        // Add some parameters
        let tensor = creation::ones(&[2, 3])?;
        state.add_parameter("weight".to_string(), tensor);

        // Add some config
        state.add_config("learning_rate".to_string(), 0.001_f32);
        state.add_config("epochs".to_string(), 100_i32);
        state.add_config("optimizer".to_string(), "adam");

        assert_eq!(state.parameters.len(), 1);
        assert_eq!(state.config.len(), 3);
        assert_eq!(state.metadata.architecture, "test_model");
        Ok(())
    }

    #[test]
    fn test_tensor_serialization() -> Result<()> {
        let original = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let serializable = SerializableTensor::from(original.clone());
        let deserialized: Tensor<f32> = serializable.try_into().unwrap();

        assert_eq!(original.shape().dims(), deserialized.shape().dims());
        assert_eq!(original.to_vec()?, deserialized.to_vec()?);
        Ok(())
    }

    #[test]
    fn test_file_serialization() -> Result<()> {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_model.json");

        let mut state = ModelState::new("test_model".to_string());
        let tensor = creation::randn(&[3, 4])?;
        state.add_parameter("weight".to_string(), tensor);
        state.add_config("learning_rate".to_string(), 0.01_f32);

        // Save to file
        state.save_to_file(&file_path).unwrap();

        // Load from file
        let loaded_state = ModelState::load_from_file(&file_path).unwrap();

        assert_eq!(
            state.metadata.architecture,
            loaded_state.metadata.architecture
        );
        assert_eq!(state.parameters.len(), loaded_state.parameters.len());
        assert_eq!(state.config.len(), loaded_state.config.len());
        Ok(())
    }

    #[test]
    fn test_binary_serialization() -> Result<()> {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_model.bin");

        let mut state = ModelState::new("binary_test".to_string());
        let tensor = creation::zeros(&[2, 2])?;
        state.add_parameter("bias".to_string(), tensor);

        // Save to binary
        state.save_to_binary(&file_path).unwrap();

        // Load from binary
        let loaded_state = ModelState::load_from_binary(&file_path).unwrap();

        assert_eq!(
            state.metadata.architecture,
            loaded_state.metadata.architecture
        );
        assert_eq!(state.parameters.len(), loaded_state.parameters.len());
        Ok(())
    }

    #[test]
    #[cfg(feature = "safetensors")]
    fn test_safetensors_serialization() -> Result<()> {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_model.safetensors");

        let mut state = ModelState::new("safetensors_test".to_string());
        let weight_tensor = creation::randn(&[3, 4])?;
        let bias_tensor = creation::zeros(&[4])?;

        state.add_parameter("weight".to_string(), weight_tensor.clone());
        state.add_parameter("bias".to_string(), bias_tensor.clone());
        state.add_config("learning_rate".to_string(), 0.001_f32);
        state.add_config("batch_size".to_string(), 32_i32);

        // Save to SafeTensors
        state.save_to_safetensors(&file_path).unwrap();

        // Load from SafeTensors
        let loaded_state = ModelState::load_from_safetensors(&file_path).unwrap();

        // Note: SafeTensors implementation doesn't preserve metadata in this version
        assert_eq!(loaded_state.metadata.architecture, "unknown");
        assert_eq!(state.parameters.len(), loaded_state.parameters.len());

        // Check parameter data
        let loaded_weight = loaded_state.get_parameter("weight").unwrap().unwrap();
        let loaded_bias = loaded_state.get_parameter("bias").unwrap().unwrap();

        assert_eq!(weight_tensor.shape().dims(), loaded_weight.shape().dims());
        assert_eq!(bias_tensor.shape().dims(), loaded_bias.shape().dims());

        // Verify data integrity (approximately, since serialization may have small differences)
        let weight_data = weight_tensor.to_vec()?;
        let loaded_weight_data = loaded_weight.to_vec()?;
        for (a, b) in weight_data.iter().zip(loaded_weight_data.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        Ok(())
    }
}
