//! Model checkpointing and serialization system for ToRSh
//!
//! This module provides comprehensive support for saving and loading neural network models,
//! including state dictionaries, optimizer states, and training metadata.

use crate::{Module, Parameter};
use serde::{Deserialize, Serialize};
use torsh_core::{device::DeviceType, error::Result, error::TorshError};
use torsh_tensor::Tensor;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{collections::HashMap, io::{Read, Write}, path::Path, sync::Arc, string::String, vec::Vec, boxed::Box};

#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, string::String, vec::Vec, boxed::Box};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;


/// Model checkpoint containing all necessary state for training resumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    /// Model state dictionary (parameter tensors)
    pub state_dict: HashMap<String, Vec<f32>>,
    
    /// Parameter shapes for reconstruction
    pub parameter_shapes: HashMap<String, Vec<usize>>,
    
    /// Model metadata
    pub metadata: CheckpointMetadata,
    
    /// Optimizer state (if available)
    pub optimizer_state: Option<OptimizerState>,
    
    /// Training statistics
    pub training_stats: Option<TrainingStats>,
}

/// Metadata associated with a model checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Model architecture name
    pub model_name: String,
    
    /// Model version/configuration
    pub model_version: String,
    
    /// Framework version
    pub torsh_version: String,
    
    /// Creation timestamp
    pub timestamp: String,
    
    /// Training epoch when checkpoint was created
    pub epoch: Option<usize>,
    
    /// Global training step
    pub global_step: Option<usize>,
    
    /// Best validation metric achieved
    pub best_metric: Option<f64>,
    
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

/// Optimizer state for training resumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Optimizer type (SGD, Adam, etc.)
    pub optimizer_type: String,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Momentum states (for applicable optimizers)
    pub momentum_states: HashMap<String, Vec<f32>>,
    
    /// Velocity states (for Adam-like optimizers)
    pub velocity_states: HashMap<String, Vec<f32>>,
    
    /// Step counts for per-parameter adaptation
    pub step_counts: HashMap<String, usize>,
    
    /// Additional optimizer-specific state
    pub custom_state: HashMap<String, Vec<f32>>,
}

/// Training statistics and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Training loss history
    pub train_losses: Vec<f64>,
    
    /// Validation loss history  
    pub val_losses: Vec<f64>,
    
    /// Training accuracy history
    pub train_accuracies: Vec<f64>,
    
    /// Validation accuracy history
    pub val_accuracies: Vec<f64>,
    
    /// Learning rate schedule history
    pub learning_rates: Vec<f64>,
    
    /// Epoch durations in seconds
    pub epoch_durations: Vec<f64>,
    
    /// Custom metrics
    pub custom_metrics: HashMap<String, Vec<f64>>,
}

/// Configuration for checkpoint saving behavior
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub save_dir: String,
    
    /// Checkpoint filename prefix
    pub filename_prefix: String,
    
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    
    /// Save frequency (in epochs)
    pub save_every_n_epochs: usize,
    
    /// Whether to save best model separately
    pub save_best: bool,
    
    /// Metric to use for determining best model
    pub best_metric_name: String,
    
    /// Whether higher metric values are better
    pub higher_is_better: bool,
    
    /// Compression level (0-9, 0 = no compression)
    pub compression_level: u32,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            save_dir: "checkpoints".to_string(),
            filename_prefix: "model".to_string(),
            max_checkpoints: 5,
            save_every_n_epochs: 1,
            save_best: true,
            best_metric_name: "val_loss".to_string(),
            higher_is_better: false,
            compression_level: 6,
        }
    }
}

/// Main checkpoint manager for handling model persistence
pub struct CheckpointManager {
    config: CheckpointConfig,
    saved_checkpoints: Vec<String>,
    best_metric_value: Option<f64>,
    best_checkpoint_path: Option<String>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager with the given configuration
    pub fn new(config: CheckpointConfig) -> Result<Self> {
        // Create checkpoint directory if it doesn't exist
        std::fs::create_dir_all(&config.save_dir).map_err(|e| {
            TorshError::IOError(format!("Failed to create checkpoint directory: {}", e))
        })?;

        Ok(Self {
            config,
            saved_checkpoints: Vec::new(),
            best_metric_value: None,
            best_checkpoint_path: None,
        })
    }

    /// Save a model checkpoint
    pub fn save_checkpoint<M: Module>(
        &mut self,
        model: &M,
        epoch: usize,
        global_step: usize,
        optimizer_state: Option<OptimizerState>,
        training_stats: Option<TrainingStats>,
        custom_metadata: Option<HashMap<String, String>>,
    ) -> Result<String> {
        // Create checkpoint
        let checkpoint = self.create_checkpoint(
            model,
            epoch,
            global_step,
            optimizer_state,
            training_stats,
            custom_metadata,
        )?;

        // Generate checkpoint filename
        let filename = format!(
            "{}_epoch_{}_step_{}.torsh",
            self.config.filename_prefix, epoch, global_step
        );
        let filepath = Path::new(&self.config.save_dir).join(&filename);

        // Save checkpoint
        self.save_checkpoint_to_file(&checkpoint, &filepath)?;

        // Update saved checkpoints list
        self.saved_checkpoints.push(filepath.to_string_lossy().to_string());

        // Check if this is the best model
        if self.config.save_best {
            if let Some(ref stats) = checkpoint.training_stats {
                if let Some(metric_values) = stats.custom_metrics.get(&self.config.best_metric_name) {
                    if let Some(&latest_metric) = metric_values.last() {
                        let is_best = self.is_best_metric(latest_metric);
                        if is_best {
                            self.save_best_model(&checkpoint)?;
                            self.best_metric_value = Some(latest_metric);
                            self.best_checkpoint_path = Some(filepath.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(filepath.to_string_lossy().to_string())
    }

    /// Load a model checkpoint from file
    pub fn load_checkpoint<M: Module>(
        &self,
        model: &mut M,
        checkpoint_path: &str,
    ) -> Result<ModelCheckpoint> {
        let checkpoint = self.load_checkpoint_from_file(checkpoint_path)?;
        self.load_state_dict(model, &checkpoint)?;
        Ok(checkpoint)
    }

    /// Load only the state dictionary without metadata
    pub fn load_state_dict<M: Module>(
        &self,
        model: &mut M,
        checkpoint: &ModelCheckpoint,
    ) -> Result<()> {
        let model_params = model.named_parameters();
        
        for (param_name, parameter) in model_params {
            if let Some(param_data) = checkpoint.state_dict.get(&param_name) {
                if let Some(param_shape) = checkpoint.parameter_shapes.get(&param_name) {
                    // Reconstruct tensor from flattened data
                    let tensor = Tensor::from_data(
                        param_data.clone(),
                        param_shape.clone(),
                        parameter.device(),
                    )?;
                    
                    // Update parameter
                    *parameter.tensor().write() = tensor;
                } else {
                    return Err(TorshError::InvalidArgument(
                        format!("Shape not found for parameter: {}", param_name)
                    ));
                }
            } else {
                return Err(TorshError::InvalidArgument(
                    format!("Parameter not found in checkpoint: {}", param_name)
                ));
            }
        }

        Ok(())
    }

    /// Get the path to the best saved model
    pub fn best_model_path(&self) -> Option<&String> {
        self.best_checkpoint_path.as_ref()
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Vec<String> {
        let mut checkpoints = Vec::new();
        
        if let Ok(entries) = std::fs::read_dir(&self.config.save_dir) {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.ends_with(".torsh") && filename.starts_with(&self.config.filename_prefix) {
                        checkpoints.push(entry.path().to_string_lossy().to_string());
                    }
                }
            }
        }
        
        checkpoints.sort();
        checkpoints
    }

    /// Create a checkpoint from model state
    fn create_checkpoint<M: Module>(
        &self,
        model: &M,
        epoch: usize,
        global_step: usize,
        optimizer_state: Option<OptimizerState>,
        training_stats: Option<TrainingStats>,
        custom_metadata: Option<HashMap<String, String>>,
    ) -> Result<ModelCheckpoint> {
        let model_params = model.named_parameters();
        let mut state_dict = HashMap::new();
        let mut parameter_shapes = HashMap::new();

        // Extract parameter data and shapes
        for (param_name, parameter) in model_params {
            let tensor = parameter.tensor().read();
            let data = tensor.data().unwrap_or_else(|_| vec![]);
            let shape = tensor.shape().dims().to_vec();
            
            state_dict.insert(param_name.clone(), data);
            parameter_shapes.insert(param_name, shape);
        }

        // Create metadata
        let mut custom = custom_metadata.unwrap_or_default();
        let metadata = CheckpointMetadata {
            model_name: model.name().unwrap_or("unknown").to_string(),
            model_version: "1.0".to_string(),
            torsh_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            epoch: Some(epoch),
            global_step: Some(global_step),
            best_metric: self.best_metric_value,
            custom,
        };

        Ok(ModelCheckpoint {
            state_dict,
            parameter_shapes,
            metadata,
            optimizer_state,
            training_stats,
        })
    }

    /// Save checkpoint to file with compression
    fn save_checkpoint_to_file(
        &self,
        checkpoint: &ModelCheckpoint,
        filepath: &Path,
    ) -> Result<()> {
        let serialized = oxicode::serde::encode_to_vec(checkpoint, oxicode::config::standard()).map_err(|e| {
            TorshError::SerializationError(format!("Failed to serialize checkpoint: {}", e))
        })?;

        if self.config.compression_level > 0 {
            // Use compression
            let mut encoder = flate2::write::GzEncoder::new(
                Vec::new(),
                flate2::Compression::new(self.config.compression_level),
            );
            encoder.write_all(&serialized).map_err(|e| {
                TorshError::IOError(format!("Failed to compress checkpoint: {}", e))
            })?;
            let compressed = encoder.finish().map_err(|e| {
                TorshError::IOError(format!("Failed to finish compression: {}", e))
            })?;
            
            std::fs::write(filepath, compressed).map_err(|e| {
                TorshError::IOError(format!("Failed to write checkpoint file: {}", e))
            })?;
        } else {
            // No compression
            std::fs::write(filepath, serialized).map_err(|e| {
                TorshError::IOError(format!("Failed to write checkpoint file: {}", e))
            })?;
        }

        Ok(())
    }

    /// Load checkpoint from file with decompression
    fn load_checkpoint_from_file(&self, filepath: &str) -> Result<ModelCheckpoint> {
        let data = std::fs::read(filepath).map_err(|e| {
            TorshError::IOError(format!("Failed to read checkpoint file: {}", e))
        })?;

        let deserialized_data = if self.config.compression_level > 0 {
            // Try decompression
            let mut decoder = flate2::read::GzDecoder::new(&data[..]);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed).map_err(|e| {
                TorshError::IOError(format!("Failed to decompress checkpoint: {}", e))
            })?;
            decompressed
        } else {
            data
        };

        let (checkpoint, _): (ModelCheckpoint, usize) = oxicode::serde::decode_from_slice(&deserialized_data, oxicode::config::standard()).map_err(|e| {
            TorshError::SerializationError(format!("Failed to deserialize checkpoint: {}", e))
        })?;

        Ok(checkpoint)
    }

    /// Save the best model to a separate file
    fn save_best_model(&self, checkpoint: &ModelCheckpoint) -> Result<()> {
        let best_filename = format!("{}_best.torsh", self.config.filename_prefix);
        let best_filepath = Path::new(&self.config.save_dir).join(best_filename);
        self.save_checkpoint_to_file(checkpoint, &best_filepath)
    }

    /// Check if a metric value is the best seen so far
    fn is_best_metric(&self, metric_value: f64) -> bool {
        match self.best_metric_value {
            None => true,
            Some(best) => {
                if self.config.higher_is_better {
                    metric_value > best
                } else {
                    metric_value < best
                }
            }
        }
    }

    /// Clean up old checkpoints based on max_checkpoints setting
    fn cleanup_old_checkpoints(&mut self) -> Result<()> {
        if self.saved_checkpoints.len() > self.config.max_checkpoints {
            let num_to_remove = self.saved_checkpoints.len() - self.config.max_checkpoints;
            
            for checkpoint_path in self.saved_checkpoints.drain(0..num_to_remove) {
                if let Err(e) = std::fs::remove_file(&checkpoint_path) {
                    eprintln!("Warning: Failed to remove old checkpoint {}: {}", checkpoint_path, e);
                }
            }
        }
        
        Ok(())
    }
}

/// Utility functions for checkpoint operations
pub mod utils {
    use super::*;

    /// Convert a model to a state dictionary
    pub fn model_to_state_dict<M: Module>(model: &M) -> HashMap<String, Vec<f32>> {
        let mut state_dict = HashMap::new();
        let params = model.named_parameters();
        
        for (name, parameter) in params {
            let tensor = parameter.tensor().read();
            state_dict.insert(name, tensor.data().unwrap_or_else(|_| vec![]));
        }
        
        state_dict
    }

    /// Get model parameter count
    pub fn count_parameters<M: Module>(model: &M) -> usize {
        model
            .parameters()
            .values()
            .map(|p| p.tensor().read().data().unwrap_or_else(|_| vec![]).len())
            .sum()
    }

    /// Get model memory usage estimate in bytes
    pub fn estimate_model_memory<M: Module>(model: &M) -> usize {
        count_parameters(model) * std::mem::size_of::<f32>()
    }

    /// Validate checkpoint compatibility with model
    pub fn validate_checkpoint_compatibility<M: Module>(
        model: &M,
        checkpoint: &ModelCheckpoint,
    ) -> Result<()> {
        let model_params = model.named_parameters();
        
        for (param_name, _) in model_params {
            if !checkpoint.state_dict.contains_key(&param_name) {
                return Err(TorshError::InvalidArgument(
                    format!("Parameter {} not found in checkpoint", param_name)
                ));
            }
            
            if !checkpoint.parameter_shapes.contains_key(&param_name) {
                return Err(TorshError::InvalidArgument(
                    format!("Shape for parameter {} not found in checkpoint", param_name)
                ));
            }
        }
        
        Ok(())
    }

    /// Create a checkpoint from state dictionary
    pub fn state_dict_to_checkpoint(
        state_dict: HashMap<String, Vec<f32>>,
        parameter_shapes: HashMap<String, Vec<usize>>,
        model_name: String,
    ) -> ModelCheckpoint {
        let metadata = CheckpointMetadata {
            model_name,
            model_version: "1.0".to_string(),
            torsh_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            epoch: None,
            global_step: None,
            best_metric: None,
            custom: HashMap::new(),
        };

        ModelCheckpoint {
            state_dict,
            parameter_shapes,
            metadata,
            optimizer_state: None,
            training_stats: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, Module};
    use torsh_core::device::DeviceType;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_creation() {
        let model = Linear::new(10, 5, true);
        let temp_dir = TempDir::new().unwrap();
        
        let config = CheckpointConfig {
            save_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        
        let mut manager = CheckpointManager::new(config).unwrap();
        
        let checkpoint_path = manager.save_checkpoint(
            &model,
            1,
            100,
            None,
            None,
            None,
        ).unwrap();
        
        assert!(Path::new(&checkpoint_path).exists());
    }

    #[test]
    fn test_checkpoint_loading() {
        let mut model1 = Linear::new(10, 5, true);
        let mut model2 = Linear::new(10, 5, true);
        let temp_dir = TempDir::new().unwrap();
        
        let config = CheckpointConfig {
            save_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        
        let mut manager = CheckpointManager::new(config).unwrap();
        
        // Save model1 state
        let checkpoint_path = manager.save_checkpoint(
            &model1,
            1,
            100,
            None,
            None,
            None,
        ).unwrap();
        
        // Load into model2
        let _checkpoint = manager.load_checkpoint(&mut model2, &checkpoint_path).unwrap();
        
        // Verify parameters match
        let params1 = model1.named_parameters();
        let params2 = model2.named_parameters();
        
        for ((name1, param1), (name2, param2)) in params1.into_iter().zip(params2.into_iter()) {
            assert_eq!(name1, name2);
            assert_eq!(
                param1.tensor().read().data().unwrap_or_else(|_| vec![]),
                param2.tensor().read().data().unwrap_or_else(|_| vec![])
            );
        }
    }

    #[test]
    fn test_parameter_counting() {
        let model = Linear::new(100, 50, true);
        let param_count = utils::count_parameters(&model);
        
        // 100 * 50 (weight) + 50 (bias) = 5050
        assert_eq!(param_count, 5050);
    }

    #[test]
    fn test_checkpoint_validation() {
        let model = Linear::new(10, 5, true);
        let temp_dir = TempDir::new().unwrap();
        
        let config = CheckpointConfig {
            save_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        
        let mut manager = CheckpointManager::new(config).unwrap();
        
        let checkpoint_path = manager.save_checkpoint(
            &model,
            1,
            100,
            None,
            None,
            None,
        ).unwrap();
        
        let checkpoint = manager.load_checkpoint_from_file(&checkpoint_path).unwrap();
        
        // Should validate successfully
        utils::validate_checkpoint_compatibility(&model, &checkpoint).unwrap();
    }
}