//! Enhanced model serialization with real tensor I/O
//!
//! This module extends the basic serialization with real tensor operations,
//! using torsh-tensor for actual data serialization.

// Infrastructure module - functions designed for CLI command integration
#![allow(dead_code)]

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

// ToRSh integration
use torsh_core::device::DeviceType;

use super::tensor_integration::ModelTensor;
use super::types::{DType, Device, LayerInfo, ModelMetadata, TensorInfo, TorshModel};

/// Enhanced model format version
const ENHANCED_FORMAT_VERSION: &str = "0.1.0-enhanced";

/// Model with real tensor data
#[derive(Debug)]
pub struct EnhancedTorshModel {
    /// Layer information
    pub layers: Vec<LayerInfo>,
    /// Real tensor weights
    pub tensors: HashMap<String, ModelTensor>,
    /// Metadata
    pub metadata: ModelMetadata,
}

impl EnhancedTorshModel {
    /// Create from basic model structure
    pub fn from_model(model: &TorshModel, device: DeviceType) -> Result<Self> {
        let mut tensors = HashMap::new();

        // Create real tensors for each weight
        for (name, tensor_info) in &model.weights {
            let tensor = ModelTensor::new_random(
                name.clone(),
                tensor_info.shape.clone(),
                tensor_info.requires_grad,
                device,
            )?;
            tensors.insert(name.clone(), tensor);
        }

        Ok(Self {
            layers: model.layers.clone(),
            tensors,
            metadata: model.metadata.clone(),
        })
    }

    /// Convert to basic model structure
    pub fn to_model(&self) -> TorshModel {
        let mut weights = HashMap::new();

        for (name, tensor) in &self.tensors {
            let weight_info = TensorInfo {
                name: name.clone(),
                shape: tensor.shape(),
                dtype: DType::F32,
                requires_grad: tensor.requires_grad,
                device: Device::Cpu,
            };
            weights.insert(name.clone(), weight_info);
        }

        TorshModel {
            layers: self.layers.clone(),
            weights,
            metadata: self.metadata.clone(),
        }
    }

    /// Get total parameter count
    pub fn parameter_count(&self) -> usize {
        self.tensors.values().map(|t| t.numel()).sum()
    }

    /// Get memory footprint in bytes
    pub fn memory_footprint(&self) -> usize {
        self.tensors.values().map(|t| t.numel() * 4).sum() // f32 = 4 bytes
    }
}

/// Save enhanced model with real tensor data
pub async fn save_enhanced_model(model: &EnhancedTorshModel, path: &Path) -> Result<()> {
    info!(
        "Saving enhanced model to {} ({} tensors, {:.2} MB)",
        path.display(),
        model.tensors.len(),
        model.memory_footprint() as f64 / (1024.0 * 1024.0)
    );

    // Create model directory structure
    let model_dir = path.with_extension("torsh.d");
    tokio::fs::create_dir_all(&model_dir)
        .await
        .context("Failed to create model directory")?;

    // Save metadata
    let metadata_path = model_dir.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&model.metadata)?;
    tokio::fs::write(&metadata_path, metadata_json)
        .await
        .context("Failed to save metadata")?;

    // Save layers
    let layers_path = model_dir.join("layers.json");
    let layers_json = serde_json::to_string_pretty(&model.layers)?;
    tokio::fs::write(&layers_path, layers_json)
        .await
        .context("Failed to save layers")?;

    // Save each tensor as separate file
    let tensors_dir = model_dir.join("tensors");
    tokio::fs::create_dir_all(&tensors_dir)
        .await
        .context("Failed to create tensors directory")?;

    let mut tensor_manifest = Vec::new();

    for (name, tensor) in &model.tensors {
        debug!("Saving tensor: {} (shape {:?})", name, tensor.shape());

        // Sanitize filename
        let safe_name = name.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
        let tensor_path = tensors_dir.join(format!("{}.bin", safe_name));

        // Serialize tensor to bytes
        let tensor_bytes = tensor.to_bytes()?;

        // Save tensor data
        tokio::fs::write(&tensor_path, &tensor_bytes)
            .await
            .with_context(|| format!("Failed to save tensor: {}", name))?;

        // Add to manifest
        tensor_manifest.push(serde_json::json!({
            "name": name,
            "file": format!("{}.bin", safe_name),
            "shape": tensor.shape(),
            "dtype": "f32",
            "requires_grad": tensor.requires_grad,
            "size_bytes": tensor_bytes.len(),
        }));
    }

    // Save tensor manifest
    let manifest_path = model_dir.join("tensor_manifest.json");
    let manifest_json = serde_json::to_string_pretty(&tensor_manifest)?;
    tokio::fs::write(&manifest_path, manifest_json)
        .await
        .context("Failed to save tensor manifest")?;

    // Create version file
    let version_path = model_dir.join("version.txt");
    tokio::fs::write(&version_path, ENHANCED_FORMAT_VERSION)
        .await
        .context("Failed to save version")?;

    info!(
        "Successfully saved enhanced model: {} parameters, {:.2} MB",
        model.parameter_count(),
        model.memory_footprint() as f64 / (1024.0 * 1024.0)
    );

    Ok(())
}

/// Load enhanced model with real tensor data
pub async fn load_enhanced_model(path: &Path) -> Result<EnhancedTorshModel> {
    info!("Loading enhanced model from {}", path.display());

    let model_dir = path.with_extension("torsh.d");

    if !model_dir.exists() {
        anyhow::bail!(
            "Model directory not found: {}. Did you mean to use .torsh.d extension?",
            model_dir.display()
        );
    }

    // Load and verify version
    let version_path = model_dir.join("version.txt");
    let version = tokio::fs::read_to_string(&version_path)
        .await
        .context("Failed to read version file")?;

    debug!("Model format version: {}", version.trim());

    if version.trim() != ENHANCED_FORMAT_VERSION {
        warn!(
            "Version mismatch: expected {}, got {}",
            ENHANCED_FORMAT_VERSION,
            version.trim()
        );
    }

    // Load metadata
    let metadata_path = model_dir.join("metadata.json");
    let metadata_json = tokio::fs::read_to_string(&metadata_path)
        .await
        .context("Failed to read metadata")?;
    let metadata: ModelMetadata =
        serde_json::from_str(&metadata_json).context("Failed to parse metadata")?;

    debug!("Loaded metadata: {}", metadata.format);

    // Load layers
    let layers_path = model_dir.join("layers.json");
    let layers_json = tokio::fs::read_to_string(&layers_path)
        .await
        .context("Failed to read layers")?;
    let layers: Vec<LayerInfo> =
        serde_json::from_str(&layers_json).context("Failed to parse layers")?;

    debug!("Loaded {} layers", layers.len());

    // Load tensor manifest
    let manifest_path = model_dir.join("tensor_manifest.json");
    let manifest_json = tokio::fs::read_to_string(&manifest_path)
        .await
        .context("Failed to read tensor manifest")?;
    let tensor_manifest: Vec<serde_json::Value> =
        serde_json::from_str(&manifest_json).context("Failed to parse tensor manifest")?;

    debug!("Loading {} tensors", tensor_manifest.len());

    // Load each tensor
    let tensors_dir = model_dir.join("tensors");
    let mut tensors = HashMap::new();

    for entry in tensor_manifest {
        let name = entry["name"]
            .as_str()
            .context("Missing tensor name")?
            .to_string();
        let file = entry["file"].as_str().context("Missing tensor file")?;
        let shape: Vec<usize> =
            serde_json::from_value(entry["shape"].clone()).context("Invalid tensor shape")?;
        let requires_grad = entry["requires_grad"].as_bool().unwrap_or(false);

        debug!("Loading tensor: {} from {}", name, file);

        let tensor_path = tensors_dir.join(file);
        let tensor_bytes = tokio::fs::read(&tensor_path)
            .await
            .with_context(|| format!("Failed to read tensor file: {}", file))?;

        // Reconstruct tensor from bytes
        let tensor = ModelTensor::from_bytes(
            name.clone(),
            &tensor_bytes,
            shape,
            requires_grad,
            DeviceType::Cpu,
        )?;

        tensors.insert(name, tensor);
    }

    let model = EnhancedTorshModel {
        layers,
        tensors,
        metadata,
    };

    info!(
        "Successfully loaded model: {} parameters, {:.2} MB",
        model.parameter_count(),
        model.memory_footprint() as f64 / (1024.0 * 1024.0)
    );

    Ok(model)
}

/// Verify model integrity after loading
pub fn verify_enhanced_model(model: &EnhancedTorshModel) -> Result<()> {
    debug!("Verifying enhanced model integrity");

    // Check layers have valid shapes
    for layer in &model.layers {
        if layer.input_shape.is_empty() || layer.output_shape.is_empty() {
            anyhow::bail!("Layer {} has invalid shape", layer.name);
        }
    }

    // Check tensors match expected structure
    for layer in &model.layers {
        let weight_name = format!("{}.weight", layer.name);
        let bias_name = format!("{}.bias", layer.name);

        if !model.tensors.contains_key(&weight_name) {
            warn!("Layer {} missing weight tensor", layer.name);
        }

        if !model.tensors.contains_key(&bias_name) {
            warn!("Layer {} missing bias tensor", layer.name);
        }
    }

    // Check all tensors have valid data
    for (name, tensor) in &model.tensors {
        if tensor.numel() == 0 {
            anyhow::bail!("Tensor {} has zero elements", name);
        }
    }

    info!("Model verification passed");
    Ok(())
}

/// Get detailed model statistics
pub fn get_enhanced_model_stats(model: &EnhancedTorshModel) -> HashMap<String, serde_json::Value> {
    use serde_json::json;

    let mut stats = HashMap::new();

    let total_params = model.parameter_count();
    let memory_mb = model.memory_footprint() as f64 / (1024.0 * 1024.0);

    // Trainable vs non-trainable
    let trainable_params: usize = model
        .tensors
        .values()
        .filter(|t| t.requires_grad)
        .map(|t| t.numel())
        .sum();

    // Layer type distribution
    let mut layer_types: HashMap<String, usize> = HashMap::new();
    for layer in &model.layers {
        *layer_types.entry(layer.layer_type.clone()).or_insert(0) += 1;
    }

    // Tensor size distribution
    let mut tensor_sizes: Vec<(String, usize)> = model
        .tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.numel()))
        .collect();
    tensor_sizes.sort_by(|a, b| b.1.cmp(&a.1));

    let largest_tensors: Vec<_> = tensor_sizes
        .iter()
        .take(5)
        .map(|(name, size)| json!({"name": name, "params": size}))
        .collect();

    stats.insert("total_parameters".to_string(), json!(total_params));
    stats.insert("trainable_parameters".to_string(), json!(trainable_params));
    stats.insert(
        "non_trainable_parameters".to_string(),
        json!(total_params - trainable_params),
    );
    stats.insert("memory_footprint_mb".to_string(), json!(memory_mb));
    stats.insert("num_layers".to_string(), json!(model.layers.len()));
    stats.insert("num_tensors".to_string(), json!(model.tensors.len()));
    stats.insert("layer_types".to_string(), json!(layer_types));
    stats.insert("largest_tensors".to_string(), json!(largest_tensors));

    stats
}

#[cfg(test)]
mod tests {
    use super::super::tensor_integration::create_real_model;
    use super::*;

    #[tokio::test]
    async fn test_save_load_enhanced_model() {
        let model = create_real_model("test", 2, DeviceType::Cpu).unwrap();
        let enhanced = EnhancedTorshModel::from_model(&model, DeviceType::Cpu).unwrap();

        let temp_dir = std::env::temp_dir();
        let model_path = temp_dir.join("test_enhanced_model.torsh");

        // Save
        save_enhanced_model(&enhanced, &model_path).await.unwrap();

        // Load
        let loaded = load_enhanced_model(&model_path).await.unwrap();

        // Verify
        assert_eq!(loaded.layers.len(), enhanced.layers.len());
        assert_eq!(loaded.tensors.len(), enhanced.tensors.len());
        assert_eq!(loaded.parameter_count(), enhanced.parameter_count());

        // Cleanup
        let model_dir = model_path.with_extension("torsh.d");
        let _ = tokio::fs::remove_dir_all(model_dir).await;
    }

    #[test]
    fn test_enhanced_model_conversion() {
        let model = create_real_model("test", 2, DeviceType::Cpu).unwrap();
        let enhanced = EnhancedTorshModel::from_model(&model, DeviceType::Cpu).unwrap();
        let converted = enhanced.to_model();

        assert_eq!(converted.layers.len(), model.layers.len());
        assert_eq!(converted.weights.len(), model.weights.len());
    }

    #[test]
    fn test_model_statistics() {
        let model = create_real_model("test", 3, DeviceType::Cpu).unwrap();
        let enhanced = EnhancedTorshModel::from_model(&model, DeviceType::Cpu).unwrap();
        let stats = get_enhanced_model_stats(&enhanced);

        assert!(stats.contains_key("total_parameters"));
        assert!(stats.contains_key("memory_footprint_mb"));
        assert!(stats.contains_key("largest_tensors"));
    }

    #[test]
    fn test_model_verification() {
        let model = create_real_model("test", 2, DeviceType::Cpu).unwrap();
        let enhanced = EnhancedTorshModel::from_model(&model, DeviceType::Cpu).unwrap();

        assert!(verify_enhanced_model(&enhanced).is_ok());
    }
}
