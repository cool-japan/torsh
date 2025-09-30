//! Audio classification models for various tasks
//!
//! This module provides specialized audio classifiers for:
//! - Audio scene classification
//! - Emotion recognition
//! - Urban sound classification
//! - Music genre classification

use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::{
    Conv1d, Dropout, Embedding, GroupNorm, LayerNorm, Linear, MultiheadAttention, GELU,
};
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

/// Configuration for audio classification models
#[derive(Debug, Clone)]
pub struct AudioClassifierConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub num_classes: usize,
    pub dropout_rate: f32,
    pub use_batch_norm: bool,
    pub activation: String,
    pub num_layers: usize,
}

impl Default for AudioClassifierConfig {
    fn default() -> Self {
        Self {
            input_dim: 768,
            hidden_dim: 256,
            num_classes: 10,
            dropout_rate: 0.1,
            use_batch_norm: true,
            activation: "relu".to_string(),
            num_layers: 3,
        }
    }
}

impl AudioClassifierConfig {
    /// Configuration for emotion recognition (7 emotions)
    pub fn emotion_recognition() -> Self {
        Self {
            num_classes: 7, // angry, disgust, fear, happy, neutral, sad, surprise
            hidden_dim: 512,
            num_layers: 4,
            dropout_rate: 0.3,
            ..Self::default()
        }
    }

    /// Configuration for music genre classification (10 genres)
    pub fn music_genre() -> Self {
        Self {
            num_classes: 10, // blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock
            hidden_dim: 256,
            num_layers: 3,
            dropout_rate: 0.2,
            ..Self::default()
        }
    }

    /// Configuration for urban sound classification (10 classes)
    pub fn urban_sound() -> Self {
        Self {
            num_classes: 10, // air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
            hidden_dim: 384,
            num_layers: 4,
            dropout_rate: 0.25,
            ..Self::default()
        }
    }

    /// Configuration for audio scene classification (10 scenes)
    pub fn audio_scene() -> Self {
        Self {
            num_classes: 10, // airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram
            hidden_dim: 512,
            num_layers: 5,
            dropout_rate: 0.2,
            ..Self::default()
        }
    }
}

/// Common classification head for audio models
pub struct AudioClassifierHead {
    layers: Vec<Linear>,
    dropouts: Vec<Dropout>,
    config: AudioClassifierConfig,
}

impl AudioClassifierHead {
    pub fn new(config: AudioClassifierConfig) -> Self {
        let mut layers = Vec::new();
        let mut dropouts = Vec::new();

        // Input layer
        layers.push(Linear::new(config.input_dim, config.hidden_dim, true));
        dropouts.push(Dropout::new(config.dropout_rate));

        // Hidden layers
        for _ in 1..config.num_layers - 1 {
            layers.push(Linear::new(config.hidden_dim, config.hidden_dim, true));
            dropouts.push(Dropout::new(config.dropout_rate));
        }

        // Output layer
        layers.push(Linear::new(config.hidden_dim, config.num_classes, true));

        Self {
            layers,
            dropouts,
            config,
        }
    }

    pub fn config(&self) -> &AudioClassifierConfig {
        &self.config
    }
}

impl Module for AudioClassifierHead {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = input.clone();

        // Forward through all layers except the last
        for i in 0..self.layers.len() - 1 {
            hidden = self.layers[i].forward(&hidden)?;

            // Apply activation (ReLU for now, could be configurable)
            hidden = hidden.relu()?;

            // Apply dropout
            hidden = self.dropouts[i].forward(&hidden)?;
        }

        // Final layer (no activation, no dropout)
        let output = self.layers.last().unwrap().forward(&hidden)?;
        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropouts.first().map_or(false, |d| d.training())
    }

    fn train(&mut self) {
        for dropout in &mut self.dropouts {
            dropout.train();
        }
    }

    fn eval(&mut self) {
        for dropout in &mut self.dropouts {
            dropout.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

// Forward declarations for specialized classifiers
pub struct AudioSceneClassifier;
pub struct EmotionRecognitionClassifier;
pub struct UrbanSoundClassifier;
pub struct MusicGenreClassifier;

// Note: Key types (AudioClassifierConfig, AudioClassifierHead) are already public
// Removed redundant re-export to fix duplicate definition errors
