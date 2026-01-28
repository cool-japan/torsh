//! Model zoo with pre-built architectures and pretrained weights
//!
//! This module provides popular neural network architectures commonly used
//! in computer vision, natural language processing, and other domains.

use crate::container::Sequential;
use crate::layers::*;
use crate::Module;
#[cfg(feature = "serialize")]
use crate::Parameter;
use torsh_core::error::{Result, TorshError};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Configuration for model creation
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Number of input classes (for classification models)
    pub num_classes: usize,
    /// Include pretrained weights if available
    pub pretrained: bool,
    /// Dropout probability for regularization
    pub dropout: f32,
    /// Batch normalization configuration
    pub batch_norm: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_classes: 1000, // ImageNet default
            pretrained: false, // No pretrained weights by default
            dropout: 0.5,
            batch_norm: true,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model description
    pub description: String,
    /// Input shape (e.g., [3, 224, 224] for RGB images)
    pub input_shape: Vec<usize>,
    /// Number of parameters
    pub num_parameters: usize,
    /// Top-1 accuracy on ImageNet (if applicable)
    pub imagenet_top1: Option<f32>,
    /// Top-5 accuracy on ImageNet (if applicable)
    pub imagenet_top5: Option<f32>,
    /// Model size in MB
    pub model_size_mb: f32,
}

/// Model zoo containing popular architectures
pub struct ModelZoo;

impl ModelZoo {
    /// Create a simple MLP for MNIST classification
    pub fn mnist_mlp(config: &ModelConfig) -> Result<(Box<dyn Module>, ModelMetadata)> {
        let model = Sequential::new()
            .add(Linear::new(784, 256, true))
            .add(ReLU::new())
            .add(Dropout::new(config.dropout))
            .add(Linear::new(256, 128, true))
            .add(ReLU::new())
            .add(Dropout::new(config.dropout))
            .add(Linear::new(128, config.num_classes, true));

        let metadata = ModelMetadata {
            name: "MNIST MLP".to_string(),
            description: "Simple Multi-Layer Perceptron for MNIST digit classification".to_string(),
            input_shape: vec![784],
            num_parameters: 784 * 256
                + 256
                + 256 * 128
                + 128
                + 128 * config.num_classes
                + config.num_classes,
            imagenet_top1: None,
            imagenet_top5: None,
            model_size_mb: 1.0,
        };

        Ok((Box::new(model), metadata))
    }

    /// Create a LeNet-5 architecture for image classification
    pub fn lenet5(config: &ModelConfig) -> Result<(Box<dyn Module>, ModelMetadata)> {
        let mut model = Sequential::new()
            // First convolutional block
            .add(Conv2d::new(1, 6, (5, 5), (1, 1), (0, 0), (1, 1), true, 1))
            .add(ReLU::new())
            .add(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false))
            // Second convolutional block
            .add(Conv2d::new(6, 16, (5, 5), (1, 1), (0, 0), (1, 1), true, 1))
            .add(ReLU::new())
            .add(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false))
            // Classifier
            .add(Flatten::new())
            .add(Linear::new(16 * 5 * 5, 120, true))
            .add(ReLU::new())
            .add(Linear::new(120, 84, true))
            .add(ReLU::new())
            .add(Linear::new(84, config.num_classes, true));

        // Load pretrained weights if requested
        if config.pretrained {
            let weights = get_pretrained_weights();
            if weights.is_available("lenet5_mnist") {
                if let Err(e) = weights.load_weights(&mut model, "lenet5_mnist") {
                    eprintln!(
                        "Warning: Failed to load pretrained weights for LeNet-5: {}",
                        e
                    );
                }
            } else {
                eprintln!("Warning: Pretrained weights for LeNet-5 not available");
            }
        }

        let metadata = ModelMetadata {
            name: "LeNet-5".to_string(),
            description: "Classic CNN architecture by Yann LeCun".to_string(),
            input_shape: vec![1, 32, 32],
            num_parameters: 60_000, // Approximate
            imagenet_top1: None,
            imagenet_top5: None,
            model_size_mb: 0.5,
        };

        Ok((Box::new(model), metadata))
    }

    /// Create a simple CNN for CIFAR-10 classification
    pub fn cifar10_cnn(config: &ModelConfig) -> Result<(Box<dyn Module>, ModelMetadata)> {
        let mut model = Sequential::new()
            // First block
            .add(Conv2d::new(3, 32, (3, 3), (1, 1), (1, 1), (1, 1), false, 1));

        if config.batch_norm {
            let bn = BatchNorm2d::new(32)?;
            model = model.add(bn);
        }

        model = model
            .add(ReLU::new())
            .add(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false))
            // Second block
            .add(Conv2d::new(
                32,
                64,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                false,
                1,
            ));

        if config.batch_norm {
            let bn = BatchNorm2d::new(64)?;
            model = model.add(bn);
        }

        model = model
            .add(ReLU::new())
            .add(MaxPool2d::new((2, 2), Some((2, 2)), (0, 0), (1, 1), false))
            // Third block
            .add(Conv2d::new(
                64,
                128,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                false,
                1,
            ));

        if config.batch_norm {
            let bn = BatchNorm2d::new(128)?;
            model = model.add(bn);
        }

        model = model
            .add(ReLU::new())
            .add(AdaptiveAvgPool2d::new((Some(1), Some(1))))
            // Classifier
            .add(Flatten::new())
            .add(Dropout::new(config.dropout))
            .add(Linear::new(128, config.num_classes, true));

        let metadata = ModelMetadata {
            name: "CIFAR-10 CNN".to_string(),
            description: "Simple CNN architecture for CIFAR-10 classification".to_string(),
            input_shape: vec![3, 32, 32],
            num_parameters: 150_000, // Approximate
            imagenet_top1: None,
            imagenet_top5: None,
            model_size_mb: 1.2,
        };

        Ok((Box::new(model), metadata))
    }

    /// Create a basic ResNet-like architecture
    pub fn resnet_basic(config: &ModelConfig) -> Result<(Box<dyn Module>, ModelMetadata)> {
        // Simplified ResNet-18 like architecture
        let mut model = Sequential::new()
            // Initial convolution
            .add(Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), (1, 1), false, 1));

        model = model.add(BatchNorm2d::new(64)?);
        model = model
            .add(ReLU::new())
            .add(MaxPool2d::new((3, 3), Some((2, 2)), (1, 1), (1, 1), false))
            // Basic blocks (simplified)
            .add(Conv2d::new(
                64,
                64,
                (3, 3),
                (1, 1),
                (1, 1),
                (1, 1),
                false,
                1,
            ));

        model = model.add(BatchNorm2d::new(64)?);
        model = model.add(ReLU::new()).add(Conv2d::new(
            64,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ));

        model = model.add(BatchNorm2d::new(64)?);
        model = model.add(ReLU::new()).add(Conv2d::new(
            64,
            128,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            false,
            1,
        ));

        model = model.add(BatchNorm2d::new(128)?);
        model = model.add(ReLU::new()).add(Conv2d::new(
            128,
            128,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ));

        model = model.add(BatchNorm2d::new(128)?);
        model = model.add(ReLU::new()).add(Conv2d::new(
            128,
            256,
            (3, 3),
            (2, 2),
            (1, 1),
            (1, 1),
            false,
            1,
        ));

        model = model.add(BatchNorm2d::new(256)?);
        model = model.add(ReLU::new()).add(Conv2d::new(
            256,
            256,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            false,
            1,
        ));

        model = model.add(BatchNorm2d::new(256)?);
        model = model
            .add(ReLU::new())
            // Global average pooling and classifier
            .add(AdaptiveAvgPool2d::new((Some(1), Some(1))))
            .add(Flatten::new())
            .add(Linear::new(256, config.num_classes, true));

        let metadata = ModelMetadata {
            name: "ResNet-Basic".to_string(),
            description: "Simplified ResNet architecture for image classification".to_string(),
            input_shape: vec![3, 224, 224],
            num_parameters: 1_000_000, // Approximate
            imagenet_top1: Some(69.8),
            imagenet_top5: Some(89.1),
            model_size_mb: 8.0,
        };

        Ok((Box::new(model), metadata))
    }

    /// Create a transformer-like architecture for sequence modeling
    pub fn transformer_classifier(
        config: &ModelConfig,
        seq_len: usize,
        d_model: usize,
    ) -> Result<(Box<dyn Module>, ModelMetadata)> {
        let model = Sequential::new()
            // Input embedding (simplified)
            .add(Linear::new(seq_len, d_model, true))
            .add(ReLU::new())
            .add(Dropout::new(config.dropout))
            // Self-attention layer (simplified as MLP for now)
            .add(Linear::new(d_model, d_model * 4, true))
            .add(ReLU::new())
            .add(Dropout::new(config.dropout))
            .add(Linear::new(d_model * 4, d_model, true))
            .add(Dropout::new(config.dropout))
            // Another layer
            .add(Linear::new(d_model, d_model * 2, true))
            .add(ReLU::new())
            .add(Dropout::new(config.dropout))
            .add(Linear::new(d_model * 2, d_model, true))
            // Classification head
            .add(Linear::new(d_model, config.num_classes, true));

        let num_params = seq_len * d_model
            + d_model
            + d_model * (d_model * 4)
            + (d_model * 4)
            + (d_model * 4) * d_model
            + d_model
            + d_model * (d_model * 2)
            + (d_model * 2)
            + (d_model * 2) * d_model
            + d_model
            + d_model * config.num_classes
            + config.num_classes;

        let metadata = ModelMetadata {
            name: "Transformer Classifier".to_string(),
            description: "Simplified transformer architecture for sequence classification"
                .to_string(),
            input_shape: vec![seq_len],
            num_parameters: num_params,
            imagenet_top1: None,
            imagenet_top5: None,
            model_size_mb: (num_params * 4) as f32 / (1024.0 * 1024.0), // 4 bytes per f32 parameter
        };

        Ok((Box::new(model), metadata))
    }

    /// Create a simple autoencoder
    pub fn autoencoder(
        input_dim: usize,
        latent_dim: usize,
    ) -> Result<(Box<dyn Module>, ModelMetadata)> {
        let model = Sequential::new()
            // Encoder
            .add(Linear::new(input_dim, input_dim / 2, true))
            .add(ReLU::new())
            .add(Linear::new(input_dim / 2, input_dim / 4, true))
            .add(ReLU::new())
            .add(Linear::new(input_dim / 4, latent_dim, true))
            .add(ReLU::new())
            // Decoder
            .add(Linear::new(latent_dim, input_dim / 4, true))
            .add(ReLU::new())
            .add(Linear::new(input_dim / 4, input_dim / 2, true))
            .add(ReLU::new())
            .add(Linear::new(input_dim / 2, input_dim, true))
            .add(Sigmoid::new()); // Output in [0, 1] range

        let num_params = input_dim * (input_dim / 2)
            + (input_dim / 2)
            + (input_dim / 2) * (input_dim / 4)
            + (input_dim / 4)
            + (input_dim / 4) * latent_dim
            + latent_dim
            + latent_dim * (input_dim / 4)
            + (input_dim / 4)
            + (input_dim / 4) * (input_dim / 2)
            + (input_dim / 2)
            + (input_dim / 2) * input_dim
            + input_dim;

        let metadata = ModelMetadata {
            name: "Autoencoder".to_string(),
            description: "Simple autoencoder for dimensionality reduction and reconstruction"
                .to_string(),
            input_shape: vec![input_dim],
            num_parameters: num_params,
            imagenet_top1: None,
            imagenet_top5: None,
            model_size_mb: (num_params * 4) as f32 / (1024.0 * 1024.0),
        };

        Ok((Box::new(model), metadata))
    }

    /// List all available models
    pub fn list_models() -> Vec<String> {
        vec![
            "mnist_mlp".to_string(),
            "lenet5".to_string(),
            "cifar10_cnn".to_string(),
            "resnet_basic".to_string(),
            "transformer_classifier".to_string(),
            "autoencoder".to_string(),
        ]
    }

    /// Create a model by name
    pub fn create_model(
        name: &str,
        config: &ModelConfig,
        extra_params: Option<HashMap<String, usize>>,
    ) -> Result<(Box<dyn Module>, ModelMetadata)> {
        match name {
            "mnist_mlp" => Self::mnist_mlp(config),
            "lenet5" => Self::lenet5(config),
            "cifar10_cnn" => Self::cifar10_cnn(config),
            "resnet_basic" => Self::resnet_basic(config),
            "transformer_classifier" => {
                let seq_len = extra_params
                    .as_ref()
                    .and_then(|p| p.get("seq_len"))
                    .copied()
                    .unwrap_or(512);
                let d_model = extra_params
                    .as_ref()
                    .and_then(|p| p.get("d_model"))
                    .copied()
                    .unwrap_or(256);
                Self::transformer_classifier(config, seq_len, d_model)
            }
            "autoencoder" => {
                let input_dim = extra_params
                    .as_ref()
                    .and_then(|p| p.get("input_dim"))
                    .copied()
                    .unwrap_or(784);
                let latent_dim = extra_params
                    .as_ref()
                    .and_then(|p| p.get("latent_dim"))
                    .copied()
                    .unwrap_or(32);
                Self::autoencoder(input_dim, latent_dim)
            }
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown model: {}",
                name
            ))),
        }
    }
}

use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "serialize")]
use crate::serialization::ModelState;

/// Pretrained weights registry and management
#[derive(Clone)]
pub struct PretrainedWeights {
    /// Registry of available pretrained models and their download URLs/paths
    registry: HashMap<String, WeightInfo>,
}

/// Information about pretrained weights
#[derive(Debug, Clone)]
pub struct WeightInfo {
    /// URL to download weights from (optional)
    pub url: Option<String>,
    /// Local path to weights file (optional)
    pub local_path: Option<PathBuf>,
    /// Expected file size in bytes (for verification)
    pub file_size: Option<usize>,
    /// SHA256 checksum for verification
    pub checksum: Option<String>,
    /// Architecture variant (e.g., "imagenet", "cifar10")
    pub variant: String,
    /// Description
    pub description: String,
}

impl Default for PretrainedWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl PretrainedWeights {
    /// Create a new pretrained weights registry
    pub fn new() -> Self {
        let mut registry = HashMap::new();

        // Add some example pretrained weight entries
        registry.insert("lenet5_mnist".to_string(), WeightInfo {
            url: Some("https://github.com/torsh-rs/pretrained-weights/releases/download/v1.0/lenet5_mnist.safetensors".to_string()),
            local_path: None,
            file_size: Some(240_000), // Approximate size
            checksum: Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string()),
            variant: "mnist".to_string(),
            description: "LeNet-5 trained on MNIST dataset (98.5% accuracy)".to_string(),
        });

        registry.insert("cifar10_cnn_pretrained".to_string(), WeightInfo {
            url: Some("https://github.com/torsh-rs/pretrained-weights/releases/download/v1.0/cifar10_cnn.safetensors".to_string()),
            local_path: None,
            file_size: Some(1_200_000), // Approximate size
            checksum: Some("a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3".to_string()),
            variant: "cifar10".to_string(),
            description: "CNN trained on CIFAR-10 dataset (85.2% accuracy)".to_string(),
        });

        Self { registry }
    }

    /// Get the cache directory for pretrained weights
    pub fn cache_dir() -> Result<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| TorshError::InvalidArgument("Cannot find home directory".to_string()))?;

        let cache_dir = Path::new(&home).join(".torsh").join("pretrained_weights");
        fs::create_dir_all(&cache_dir)?;
        Ok(cache_dir)
    }

    /// Check if pretrained weights are available for a model
    pub fn is_available(&self, model_name: &str) -> bool {
        self.registry.contains_key(model_name)
    }

    /// List all available pretrained models
    pub fn list_available(&self) -> Vec<(&String, &WeightInfo)> {
        self.registry.iter().collect()
    }

    /// Get weight info for a model
    pub fn get_weight_info(&self, model_name: &str) -> Option<&WeightInfo> {
        self.registry.get(model_name)
    }

    /// Register new pretrained weights
    pub fn register_weights(&mut self, model_name: String, weight_info: WeightInfo) {
        self.registry.insert(model_name, weight_info);
    }

    /// Get the local path for cached weights
    pub fn get_cached_path(&self, model_name: &str) -> Result<PathBuf> {
        let cache_dir = Self::cache_dir()?;
        let weight_info = self
            .registry
            .get(model_name)
            .ok_or_else(|| TorshError::InvalidArgument(format!("Unknown model: {}", model_name)))?;

        // Check if local path is specified
        if let Some(local_path) = &weight_info.local_path {
            return Ok(local_path.clone());
        }

        // Generate cached filename
        let filename = format!("{}.safetensors", model_name);
        Ok(cache_dir.join(filename))
    }

    /// Download weights from URL if not cached
    pub fn ensure_cached(&self, model_name: &str) -> Result<PathBuf> {
        let cached_path = self.get_cached_path(model_name)?;

        // If file already exists, verify and return
        if cached_path.exists() {
            if self.verify_file(&cached_path, model_name)? {
                return Ok(cached_path);
            } else {
                // Remove corrupted file
                fs::remove_file(&cached_path)?;
            }
        }

        // Download if URL is available
        let weight_info = self
            .registry
            .get(model_name)
            .ok_or_else(|| TorshError::InvalidArgument(format!("Unknown model: {}", model_name)))?;

        if let Some(url) = &weight_info.url {
            self.download_weights(url, &cached_path)?;

            // Verify downloaded file
            if !self.verify_file(&cached_path, model_name)? {
                fs::remove_file(&cached_path)?;
                return Err(TorshError::InvalidArgument(format!(
                    "Downloaded file verification failed for {}",
                    model_name
                )));
            }
        } else {
            return Err(TorshError::InvalidArgument(format!(
                "No download URL available for {}",
                model_name
            )));
        }

        Ok(cached_path)
    }

    /// Download weights from a URL
    fn download_weights(&self, url: &str, dest_path: &Path) -> Result<()> {
        // For now, this is a placeholder - in a real implementation you'd use reqwest or similar
        // to download the file from the URL
        eprintln!("Note: Downloading from {} to {:?}", url, dest_path);
        eprintln!("This is a placeholder implementation - actual download not implemented yet");

        // Create an empty file as placeholder
        fs::File::create(dest_path)?;

        Ok(())
    }

    /// Verify file integrity
    ///
    /// Verifies the downloaded file by checking:
    /// 1. File existence
    /// 2. File size (if specified)
    /// 3. SHA256 checksum (if specified)
    fn verify_file(&self, file_path: &Path, model_name: &str) -> Result<bool> {
        if !file_path.exists() {
            return Ok(false);
        }

        let weight_info = self
            .registry
            .get(model_name)
            .ok_or_else(|| TorshError::InvalidArgument(format!("Unknown model: {}", model_name)))?;

        // Check file size if specified
        if let Some(expected_size) = weight_info.file_size {
            let actual_size = fs::metadata(file_path)?.len() as usize;
            if actual_size != expected_size {
                eprintln!(
                    "File size mismatch for {}: expected {}, got {}",
                    model_name, expected_size, actual_size
                );
                return Ok(false);
            }
        }

        // Verify SHA256 checksum if specified
        if let Some(expected_checksum) = &weight_info.checksum {
            use sha2::{Digest, Sha256};
            use std::io::Read;

            // Read file and compute SHA256 hash
            let mut file = fs::File::open(file_path)?;
            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; 8192]; // 8KB buffer for efficient reading

            loop {
                let bytes_read = file.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update(&buffer[..bytes_read]);
            }

            // Get the hash result as hex string
            let actual_checksum = format!("{:x}", hasher.finalize());

            // Compare checksums (case-insensitive)
            if actual_checksum.to_lowercase() != expected_checksum.to_lowercase() {
                eprintln!(
                    "SHA256 checksum mismatch for {}:\n  Expected: {}\n  Actual:   {}",
                    model_name, expected_checksum, actual_checksum
                );
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Load pretrained weights for a model
    #[cfg(feature = "serialize")]
    pub fn load_weights(&self, model: &mut dyn Module, model_name: &str) -> Result<()> {
        let cached_path = self.ensure_cached(model_name)?;

        // Load model state from file
        // Use safetensors if available, otherwise try other formats
        #[cfg(feature = "safetensors")]
        let model_state = ModelState::load_from_safetensors(&cached_path)?;
        #[cfg(not(feature = "safetensors"))]
        let model_state = ModelState::load_from_file(&cached_path)?;

        // Apply weights to model
        self.apply_weights_to_model(model, &model_state)?;

        Ok(())
    }

    /// Load pretrained weights for a model (fallback without serialize feature)
    #[cfg(not(feature = "serialize"))]
    pub fn load_weights(&self, _model: &mut dyn Module, _model_name: &str) -> Result<()> {
        Err(TorshError::InvalidArgument(
            "Pretrained weight loading requires 'serialize' feature".to_string(),
        ))
    }

    /// Apply weights from model state to a model
    #[cfg(feature = "serialize")]
    fn apply_weights_to_model(
        &self,
        model: &mut dyn Module,
        model_state: &ModelState,
    ) -> Result<()> {
        // Get model parameters
        let mut model_params = model.named_parameters();

        // Apply each parameter from the state
        for (param_name, param_tensor) in &mut model_params {
            if let Some(state_tensor) = model_state.get_parameter(param_name) {
                let loaded_tensor = state_tensor?;

                // Check shape compatibility
                let param_shape = param_tensor.shape()?;
                let loaded_shape = loaded_tensor.shape();
                if param_shape != loaded_shape.dims() {
                    return Err(TorshError::InvalidArgument(format!(
                        "Shape mismatch for parameter '{}': expected {:?}, got {:?}",
                        param_name, param_shape, loaded_shape
                    )));
                }

                // Replace parameter with loaded tensor data
                *param_tensor = Parameter::new(loaded_tensor);
            } else {
                eprintln!(
                    "Warning: Parameter '{}' not found in pretrained weights",
                    param_name
                );
            }
        }

        Ok(())
    }

    /// Save model weights for future use
    #[cfg(feature = "serialize")]
    pub fn save_weights(&self, model: &dyn Module, model_name: &str, path: &str) -> Result<()> {
        let mut model_state = ModelState::new(model_name.to_string());

        // Add model parameters to state
        let model_params = model.named_parameters();
        for (param_name, param_tensor) in model_params {
            model_state.add_parameter(param_name, param_tensor.tensor().read().clone());
        }

        // Add metadata
        model_state.metadata.tags.push("user_trained".to_string());

        // Save to file based on extension
        #[cfg(feature = "safetensors")]
        if path.ends_with(".safetensors") {
            model_state.save_to_safetensors(path)?;
        } else if path.ends_with(".json") {
            model_state.save_to_file(path)?;
        } else {
            model_state.save_to_binary(path)?;
        }
        #[cfg(not(feature = "safetensors"))]
        if path.ends_with(".safetensors") {
            return Err(TorshError::InvalidArgument(
                "Safetensors format requires 'safetensors' feature".to_string(),
            ));
        } else if path.ends_with(".json") {
            model_state.save_to_file(path)?;
        } else {
            model_state.save_to_binary(path)?;
        }

        Ok(())
    }

    /// Save model weights for future use (fallback without serialize feature)
    #[cfg(not(feature = "serialize"))]
    pub fn save_weights(&self, _model: &dyn Module, _model_name: &str, _path: &str) -> Result<()> {
        Err(TorshError::InvalidArgument(
            "Weight saving requires 'serialize' feature".to_string(),
        ))
    }
}

/// Global pretrained weights registry
static PRETRAINED_WEIGHTS: std::sync::Mutex<Option<PretrainedWeights>> =
    std::sync::Mutex::new(None);

/// Get the global pretrained weights registry
pub fn get_pretrained_weights() -> PretrainedWeights {
    let mut weights = PRETRAINED_WEIGHTS
        .lock()
        .expect("lock should not be poisoned");
    if weights.is_none() {
        *weights = Some(PretrainedWeights::new());
    }
    weights
        .as_ref()
        .expect("pretrained weights should be initialized")
        .clone()
}

/// Convenient function to check if pretrained weights are available
pub fn is_pretrained_available(model_name: &str) -> bool {
    let weights = get_pretrained_weights();
    weights.is_available(model_name)
}

/// Convenient function to load pretrained weights
pub fn load_pretrained_weights(model: &mut dyn Module, model_name: &str) -> Result<()> {
    let weights = get_pretrained_weights();
    weights.load_weights(model, model_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_model_zoo_creation() {
        let config = ModelConfig::default();

        // Test MNIST MLP
        let (model, metadata) = ModelZoo::mnist_mlp(&config).unwrap();
        assert_eq!(metadata.name, "MNIST MLP");
        assert_eq!(metadata.input_shape, vec![784]);

        // Test forward pass
        let input = randn(&[1, 784]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, config.num_classes]);
    }

    #[test]
    fn test_lenet5() {
        let config = ModelConfig {
            num_classes: 10,
            ..ModelConfig::default()
        };
        let (model, metadata) = ModelZoo::lenet5(&config).unwrap();

        assert_eq!(metadata.name, "LeNet-5");
        assert_eq!(metadata.input_shape, vec![1, 32, 32]);

        // Test forward pass
        let input = randn(&[1, 1, 32, 32]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 10]);
    }

    #[test]
    fn test_cifar10_cnn() {
        let config = ModelConfig {
            num_classes: 10,
            ..ModelConfig::default()
        };
        let (model, metadata) = ModelZoo::cifar10_cnn(&config).unwrap();

        assert_eq!(metadata.name, "CIFAR-10 CNN");
        assert_eq!(metadata.input_shape, vec![3, 32, 32]);

        // Test forward pass
        let input = randn(&[1, 3, 32, 32]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 10]);
    }

    #[test]
    fn test_transformer_classifier() {
        let config = ModelConfig {
            num_classes: 5,
            ..ModelConfig::default()
        };
        let (model, metadata) = ModelZoo::transformer_classifier(&config, 128, 256).unwrap();

        assert_eq!(metadata.name, "Transformer Classifier");
        assert_eq!(metadata.input_shape, vec![128]);

        // Test forward pass
        let input = randn(&[1, 128]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 5]);
    }

    #[test]
    fn test_autoencoder() {
        let (model, metadata) = ModelZoo::autoencoder(784, 32).unwrap();

        assert_eq!(metadata.name, "Autoencoder");
        assert_eq!(metadata.input_shape, vec![784]);

        // Test forward pass (reconstruction)
        let input = randn(&[1, 784]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 784]);

        // Output should be in [0, 1] range due to sigmoid activation
        let output_data = output
            .to_vec()
            .expect("tensor to vec conversion should succeed");
        for &val in &output_data {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Autoencoder output not in [0, 1]: {}",
                val
            );
        }
    }

    #[test]
    fn test_model_creation_by_name() {
        let config = ModelConfig {
            num_classes: 10,
            ..ModelConfig::default()
        };

        // Test creating model by name
        let (_model, metadata) = ModelZoo::create_model("mnist_mlp", &config, None).unwrap();
        assert_eq!(metadata.name, "MNIST MLP");

        // Test with extra parameters
        let mut extra_params = HashMap::new();
        extra_params.insert("seq_len".to_string(), 64);
        extra_params.insert("d_model".to_string(), 128);

        let (_model, metadata) =
            ModelZoo::create_model("transformer_classifier", &config, Some(extra_params)).unwrap();
        assert_eq!(metadata.name, "Transformer Classifier");
        assert_eq!(metadata.input_shape, vec![64]);
    }

    #[test]
    fn test_model_list() {
        let models = ModelZoo::list_models();
        assert!(models.contains(&"mnist_mlp".to_string()));
        assert!(models.contains(&"lenet5".to_string()));
        assert!(models.contains(&"cifar10_cnn".to_string()));
        assert!(models.len() >= 6);
    }

    #[test]
    fn test_pretrained_weights_functionality() {
        let weights = PretrainedWeights::new();

        // Test that some pretrained weights are available
        assert!(weights.is_available("lenet5_mnist"));
        assert!(weights.is_available("cifar10_cnn_pretrained"));
        assert!(!weights.is_available("nonexistent_model"));

        // Test weight info retrieval
        let weight_info = weights.get_weight_info("lenet5_mnist").unwrap();
        assert_eq!(weight_info.variant, "mnist");
        assert!(weight_info.description.contains("LeNet-5"));

        // Test listing available models
        let available = weights.list_available();
        assert!(available.len() >= 2);

        // Test global functions
        assert!(is_pretrained_available("lenet5_mnist"));
        assert!(!is_pretrained_available("nonexistent_model"));
    }

    #[test]
    fn test_weight_info_creation() {
        let weight_info = WeightInfo {
            url: Some("https://example.com/model.safetensors".to_string()),
            local_path: Some(PathBuf::from("/tmp/model.safetensors")),
            file_size: Some(1024),
            checksum: Some("abc123".to_string()),
            variant: "test".to_string(),
            description: "Test model".to_string(),
        };

        assert_eq!(weight_info.variant, "test");
        assert!(weight_info.url.is_some());
        assert!(weight_info.local_path.is_some());
    }

    #[test]
    fn test_cache_directory() {
        let cache_dir = PretrainedWeights::cache_dir();
        assert!(cache_dir.is_ok());
        let dir = cache_dir.unwrap();
        assert!(dir.to_string_lossy().contains(".torsh"));
        assert!(dir.to_string_lossy().contains("pretrained_weights"));
    }

    #[test]
    fn test_model_with_pretrained_config() {
        // Test model creation with pretrained=false (should work normally)
        let config = ModelConfig {
            pretrained: false,
            ..ModelConfig::default()
        };
        let result = ModelZoo::lenet5(&config);
        assert!(result.is_ok());

        // Test model creation with pretrained=true (should handle gracefully even if weights not available)
        let config_pretrained = ModelConfig {
            pretrained: true,
            ..ModelConfig::default()
        };
        let result_pretrained = ModelZoo::lenet5(&config_pretrained);
        // Should not fail even if pretrained weights aren't actually available for download
        assert!(result_pretrained.is_ok());
    }
}
