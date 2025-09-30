//! ALIGN Model Implementation
//!
//! Main ALIGN model combining vision and text encoders for large-scale
//! vision-language learning with contrastive learning.

use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::Linear;
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

use super::config::ALIGNConfig;
use super::text::ALIGNTextEncoder;
use super::vision::ALIGNVisionEncoder;

/// ALIGN Model for Large-scale Vision-Language Learning
pub struct ALIGNModel {
    vision_encoder: ALIGNVisionEncoder,
    text_encoder: ALIGNTextEncoder,
    vision_projection: Linear,
    text_projection: Linear,
    temperature: Parameter,
    config: ALIGNConfig,
}

impl ALIGNModel {
    /// Create new ALIGN model with given configuration
    pub fn new(config: ALIGNConfig) -> Result<Self> {
        config.validate()?;

        let vision_encoder = ALIGNVisionEncoder::new(config.vision_config.clone());
        let text_encoder = ALIGNTextEncoder::new(config.text_config.clone())?;

        let vision_projection =
            Linear::new(config.vision_config.head_size, config.projection_dim, false);
        let text_projection =
            Linear::new(config.text_config.hidden_size, config.projection_dim, false);

        let temperature = if config.learnable_temperature {
            Parameter::new(creation::tensor_scalar(config.temperature.ln())?)
        } else {
            Parameter::new(creation::tensor_scalar(config.temperature)?)
        };

        Ok(Self {
            vision_encoder,
            text_encoder,
            vision_projection,
            text_projection,
            temperature,
            config,
        })
    }

    /// Create ALIGN model with default configuration
    pub fn align_large() -> Result<Self> {
        Self::new(ALIGNConfig::align_large())
    }

    /// Create smaller ALIGN model
    pub fn align_small() -> Result<Self> {
        Self::new(ALIGNConfig::align_small())
    }

    /// Get vision features from images
    pub fn get_vision_features(&self, images: &Tensor) -> Result<Tensor> {
        let vision_embeds = self.vision_encoder.forward(images)?;
        let vision_embeds = self.vision_projection.forward(&vision_embeds)?;

        // L2 normalize
        let norm = vision_embeds.norm()?;
        vision_embeds.div(&norm)
    }

    /// Get text features from input tokens
    pub fn get_text_features(&self, input_ids: &Tensor) -> Result<Tensor> {
        let text_embeds = self.text_encoder.forward(input_ids)?;
        let text_embeds = self.text_projection.forward(&text_embeds)?;

        // L2 normalize
        let norm = text_embeds.norm()?;
        text_embeds.div(&norm)
    }

    /// Compute contrastive loss for vision-text alignment
    pub fn contrastive_loss(&self, vision_embeds: &Tensor, text_embeds: &Tensor) -> Result<Tensor> {
        let temperature_tensor = self.temperature.tensor();
        let temperature_guard = temperature_tensor.read();
        let temperature = if self.config.learnable_temperature {
            temperature_guard.exp()?
        } else {
            temperature_guard.clone()
        };

        // Compute similarity matrix
        let logits = vision_embeds.matmul(&text_embeds.transpose(0, 1)?)?;
        let logits = logits.div(&temperature)?;

        // Labels for contrastive learning (diagonal is positive)
        let batch_size = vision_embeds.size(0)?;
        let labels = creation::arange(0i64, batch_size as i64, 1i64)?;

        // Cross entropy loss for both directions
        let loss_i2t = logits.mean(None, false)?; // Placeholder for cross_entropy
        let loss_t2i = logits.transpose(0, 1)?.mean(None, false)?; // Placeholder for cross_entropy

        let total_loss = loss_i2t.add(&loss_t2i)?.div_scalar(2.0)?;
        Ok(total_loss)
    }

    /// Get configuration
    pub fn config(&self) -> &ALIGNConfig {
        &self.config
    }

    /// Compute similarity between images and texts
    pub fn compute_similarity(&self, images: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
        let vision_features = self.get_vision_features(images)?;
        let text_features = self.get_text_features(input_ids)?;

        // Compute cosine similarity
        vision_features.matmul(&text_features.transpose(-2, -1)?)
    }
}

impl Module for ALIGNModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For ALIGN, we typically need both images and text
        // This is a simplified forward that just returns the vision features
        self.get_vision_features(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.vision_encoder.parameters() {
            params.insert(format!("vision_encoder.{}", name), param);
        }

        for (name, param) in self.text_encoder.parameters() {
            params.insert(format!("text_encoder.{}", name), param);
        }

        for (name, param) in self.vision_projection.parameters() {
            params.insert(format!("vision_projection.{}", name), param);
        }

        for (name, param) in self.text_projection.parameters() {
            params.insert(format!("text_projection.{}", name), param);
        }

        if self.config.learnable_temperature {
            params.insert("temperature".to_string(), self.temperature.clone());
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.vision_encoder.training() && self.text_encoder.training()
    }

    fn train(&mut self) {
        self.vision_encoder.train();
        self.text_encoder.train();
    }

    fn eval(&mut self) {
        self.vision_encoder.eval();
        self.text_encoder.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.vision_encoder.to_device(device)?;
        self.text_encoder.to_device(device)?;
        self.vision_projection.to_device(device)?;
        self.text_projection.to_device(device)?;
        Ok(())
    }
}
