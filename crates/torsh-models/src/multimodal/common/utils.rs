//! Common utilities for multimodal models

use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

/// Global Average Pooling 2D layer
pub struct GlobalAveragePooling2d;

impl GlobalAveragePooling2d {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: [batch, channels, height, width]
        // Output: [batch, channels]
        x.mean(Some(&[2, 3]), false)
    }
}

impl Module for GlobalAveragePooling2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn training(&self) -> bool {
        true
    }

    fn train(&mut self) {
        // No-op
    }

    fn eval(&mut self) {
        // No-op
    }

    fn to_device(&mut self, _device: DeviceType) -> Result<()> {
        Ok(())
    }
}

/// Squeeze-and-Excitation block
pub struct SqueezeExcitation {
    fc1: Linear,
    fc2: Linear,
    activation: ReLU,
    sigmoid: Sigmoid,
    reduction_ratio: usize,
}

impl SqueezeExcitation {
    pub fn new(channels: usize, reduction_ratio: usize) -> Self {
        let reduced_channels = channels / reduction_ratio;
        let fc1 = Linear::new(channels, reduced_channels, true);
        let fc2 = Linear::new(reduced_channels, channels, true);
        let activation = ReLU::new();
        let sigmoid = Sigmoid::new();

        Self {
            fc1,
            fc2,
            activation,
            sigmoid,
            reduction_ratio,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x shape: [batch, channels, height, width]
        let batch_size = x.size(0)?;
        let channels = x.size(1)?;

        // Global average pooling: [batch, channels, 1, 1]
        let squeezed = x.mean(Some(&[2, 3]), true)?;

        // Flatten to [batch, channels]
        let squeezed = squeezed.view(&[batch_size as i32, channels as i32])?;

        // First FC layer + activation
        let excited = self.fc1.forward(&squeezed)?;
        let excited = self.activation.forward(&excited)?;

        // Second FC layer + sigmoid
        let excited = self.fc2.forward(&excited)?;
        let excited = self.sigmoid.forward(&excited)?;

        // Reshape back to [batch, channels, 1, 1]
        let excited = excited.view(&[batch_size as i32, channels as i32, 1, 1])?;

        // Apply channel-wise multiplication
        x.mul(&excited)
    }
}

impl Module for SqueezeExcitation {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.fc1.parameters() {
            params.insert(format!("fc1.{}", name), param);
        }
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.fc1.training() && self.fc2.training()
    }

    fn train(&mut self) {
        self.fc1.train();
        self.fc2.train();
        self.activation.train();
        self.sigmoid.train();
    }

    fn eval(&mut self) {
        self.fc1.eval();
        self.fc2.eval();
        self.activation.eval();
        self.sigmoid.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        Ok(())
    }
}

/// Cross-modal projection layer for aligning vision and text representations
pub struct CrossModalProjection {
    vision_proj: Linear,
    text_proj: Linear,
    normalize: bool,
    dropout: Dropout,
}

impl CrossModalProjection {
    pub fn new(
        vision_dim: usize,
        text_dim: usize,
        projection_dim: usize,
        normalize: bool,
        dropout: f32,
    ) -> Self {
        let vision_proj = Linear::new(vision_dim, projection_dim, false);
        let text_proj = Linear::new(text_dim, projection_dim, false);
        let dropout = Dropout::new(dropout);

        Self {
            vision_proj,
            text_proj,
            normalize,
            dropout,
        }
    }

    pub fn project_vision(&self, vision_features: &Tensor) -> Result<Tensor> {
        let projected = self.vision_proj.forward(vision_features)?;
        let projected = self.dropout.forward(&projected)?;

        if self.normalize {
            let norm = projected.norm()?;
            projected.div(&norm)
        } else {
            Ok(projected)
        }
    }

    pub fn project_text(&self, text_features: &Tensor) -> Result<Tensor> {
        let projected = self.text_proj.forward(text_features)?;
        let projected = self.dropout.forward(&projected)?;

        if self.normalize {
            let norm = projected.norm()?;
            projected.div(&norm)
        } else {
            Ok(projected)
        }
    }

    pub fn forward(
        &self,
        vision_features: &Tensor,
        text_features: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let vision_proj = self.project_vision(vision_features)?;
        let text_proj = self.project_text(text_features)?;
        Ok((vision_proj, text_proj))
    }
}

impl Module for CrossModalProjection {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Default forward for single input - just use vision projection
        self.project_vision(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.vision_proj.parameters() {
            params.insert(format!("vision_proj.{}", name), param);
        }
        for (name, param) in self.text_proj.parameters() {
            params.insert(format!("text_proj.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.vision_proj.training() && self.text_proj.training() && self.dropout.training()
    }

    fn train(&mut self) {
        self.vision_proj.train();
        self.text_proj.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.vision_proj.eval();
        self.text_proj.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.vision_proj.to_device(device)?;
        self.text_proj.to_device(device)?;
        Ok(())
    }
}

/// Utility function to compute contrastive loss (InfoNCE)
pub fn contrastive_loss(
    vision_features: &Tensor,
    text_features: &Tensor,
    temperature: f32,
) -> Result<Tensor> {
    // vision_features: [batch_size, embed_dim]
    // text_features: [batch_size, embed_dim]

    let batch_size = vision_features.size(0)?;

    // Compute similarity matrix
    let logits = vision_features.matmul(&text_features.transpose(-2, -1)?)?;
    let logits = logits.div_scalar(temperature)?;

    // Create labels (diagonal matrix for positive pairs)
    let labels = creation::arange(0i64, batch_size as i64, 1i64)?;

    // TODO: Implement proper cross-entropy loss when available
    // For now, use a placeholder approximation
    let target_one_hot = torsh_tensor::creation::zeros_like(&logits)?;
    let loss_v2t = logits.mean(None, false)?; // Placeholder
    let loss_t2v = logits.transpose(-2, -1)?.mean(None, false)?; // Placeholder

    // Average the two losses
    let total_loss = loss_v2t.add(&loss_t2v)?.div_scalar(2.0)?;

    Ok(total_loss)
}

/// Positional encoding utilities
pub fn create_sinusoidal_position_embeddings(seq_len: usize, embed_dim: usize) -> Result<Tensor> {
    let mut embeddings = Vec::with_capacity(seq_len * embed_dim);

    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / embed_dim as f32);
            if i % 2 == 0 {
                embeddings.push(angle.sin());
            } else {
                embeddings.push(angle.cos());
            }
        }
    }

    torsh_tensor::creation::from_vec(
        embeddings,
        &[seq_len, embed_dim],
        torsh_core::DeviceType::Cpu,
    )
}
