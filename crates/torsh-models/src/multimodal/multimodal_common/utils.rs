//! Common utilities for multimodal models

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
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
///
/// Computes the InfoNCE contrastive loss for vision-language alignment.
/// This loss encourages matching vision-text pairs to have high similarity
/// while non-matching pairs have low similarity.
///
/// # Arguments
///
/// * `vision_features` - Vision embeddings of shape [batch_size, embed_dim]
/// * `text_features` - Text embeddings of shape [batch_size, embed_dim]
/// * `temperature` - Temperature parameter for scaling similarities
///
/// # Returns
///
/// The contrastive loss averaged over vision-to-text and text-to-vision directions
///
/// # References
///
/// Based on the CLIP paper: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
pub fn contrastive_loss(
    vision_features: &Tensor,
    text_features: &Tensor,
    temperature: f32,
) -> Result<Tensor> {
    // vision_features: [batch_size, embed_dim]
    // text_features: [batch_size, embed_dim]

    let batch_size = vision_features.size(0)? as i64;

    // Compute similarity matrix: [batch_size, batch_size]
    let logits = vision_features.matmul(&text_features.transpose(-2, -1)?)?;
    let logits = logits.div_scalar(temperature)?;

    // Create target labels: diagonal indices (0, 1, 2, ..., batch_size-1)
    // Each vision embedding should match with its corresponding text embedding
    let labels = creation::arange(0i64, batch_size, 1i64)?;

    // Compute cross-entropy loss in vision-to-text direction
    // logits: [batch_size, batch_size], labels: [batch_size]
    let loss_v2t = compute_cross_entropy(&logits, &labels)?;

    // Compute cross-entropy loss in text-to-vision direction
    // Transpose logits to get text-to-vision similarities
    let logits_t2v = logits.transpose(-2, -1)?;
    let loss_t2v = compute_cross_entropy(&logits_t2v, &labels)?;

    // Average the two losses for symmetric contrastive learning
    let total_loss = loss_v2t.add(&loss_t2v)?.div_scalar(2.0)?;

    Ok(total_loss)
}

/// Helper function to compute cross-entropy loss with proper numerical stability
///
/// # Arguments
///
/// * `logits` - Unnormalized logits of shape [batch_size, num_classes]
/// * `labels` - Target class indices of shape [batch_size]
///
/// # Returns
///
/// Mean cross-entropy loss across the batch
fn compute_cross_entropy(logits: &Tensor, labels: &Tensor<i64>) -> Result<Tensor> {
    // Use log_softmax for numerical stability (built-in method)
    let log_probs = logits.log_softmax(-1)?;

    // Gather log probabilities for the true labels
    // For each batch element, select the log probability of the correct class
    let batch_size = logits.size(0)?;
    let mut loss_values = Vec::with_capacity(batch_size);

    let labels_vec = labels.to_vec()?;
    for (i, &label_idx) in labels_vec.iter().enumerate() {
        // Get log probability for the true class
        let log_prob = log_probs.select(0, i as i64)?.select(0, label_idx)?;
        let prob_value: f32 = log_prob.item()?;
        loss_values.push(-prob_value); // Negative log likelihood
    }

    // Compute mean loss
    let mean_loss = loss_values.iter().sum::<f32>() / batch_size as f32;

    // Return as scalar tensor
    Tensor::scalar(mean_loss)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contrastive_loss_shape() -> Result<()> {
        // Test that contrastive loss returns a scalar
        let batch_size = 4;
        let embed_dim = 128;

        // Create random vision and text features
        let vision_features = torsh_tensor::creation::randn(&[batch_size, embed_dim])?;
        let text_features = torsh_tensor::creation::randn(&[batch_size, embed_dim])?;

        // Compute contrastive loss
        let loss = contrastive_loss(&vision_features, &text_features, 0.07)?;

        // Loss should be a scalar (0-dimensional or shape [1])
        assert!(
            loss.shape().dims().is_empty() || loss.shape().dims() == &[1],
            "Loss should be a scalar tensor"
        );

        Ok(())
    }

    #[test]
    fn test_contrastive_loss_perfect_match() -> Result<()> {
        // Test loss when vision and text features are identical
        let batch_size = 4;
        let embed_dim = 64;

        // Create identical features
        let features = torsh_tensor::creation::randn(&[batch_size, embed_dim])?;

        // Normalize features for better numerical stability
        let norm = features.norm()?;
        let features = features.div(&norm)?;

        // Compute loss with identical features
        let loss = contrastive_loss(&features, &features, 0.07)?;

        // Loss should be relatively small (close to 0) for perfect matches
        // but not exactly 0 due to softmax over all batch elements
        let loss_value: f32 = loss.item()?;
        assert!(
            loss_value >= 0.0,
            "Loss should be non-negative, got: {}",
            loss_value
        );

        Ok(())
    }

    #[test]
    fn test_contrastive_loss_temperature_scaling() -> Result<()> {
        // Test that temperature affects loss magnitude
        let batch_size = 4;
        let embed_dim = 64;

        let vision_features = torsh_tensor::creation::randn(&[batch_size, embed_dim])?;
        let text_features = torsh_tensor::creation::randn(&[batch_size, embed_dim])?;

        // Compute loss with different temperatures
        let loss_low_temp = contrastive_loss(&vision_features, &text_features, 0.01)?;
        let loss_high_temp = contrastive_loss(&vision_features, &text_features, 1.0)?;

        let loss_low: f32 = loss_low_temp.item()?;
        let loss_high: f32 = loss_high_temp.item()?;

        // Both losses should be non-negative
        assert!(loss_low >= 0.0, "Loss should be non-negative");
        assert!(loss_high >= 0.0, "Loss should be non-negative");

        // Temperature affects the sharpness of the distribution
        // but the exact relationship depends on the data
        println!(
            "Low temp loss: {:.4}, High temp loss: {:.4}",
            loss_low, loss_high
        );

        Ok(())
    }

    #[test]
    fn test_contrastive_loss_batch_size() -> Result<()> {
        // Test with different batch sizes
        for batch_size in [2, 4, 8, 16] {
            let embed_dim = 64;

            let vision_features = torsh_tensor::creation::randn(&[batch_size, embed_dim])?;
            let text_features = torsh_tensor::creation::randn(&[batch_size, embed_dim])?;

            let loss = contrastive_loss(&vision_features, &text_features, 0.07)?;
            let loss_value: f32 = loss.item()?;

            assert!(
                loss_value >= 0.0 && loss_value < 100.0,
                "Loss should be reasonable for batch_size={}, got: {}",
                batch_size,
                loss_value
            );
        }

        Ok(())
    }

    #[test]
    fn test_sinusoidal_position_embeddings_shape() -> Result<()> {
        // Test shape of sinusoidal position embeddings
        let seq_len = 100;
        let embed_dim = 512;

        let embeddings = create_sinusoidal_position_embeddings(seq_len, embed_dim)?;

        assert_eq!(
            embeddings.shape().dims(),
            &[seq_len, embed_dim],
            "Embeddings should have shape [seq_len, embed_dim]"
        );

        Ok(())
    }

    #[test]
    fn test_sinusoidal_position_embeddings_properties() -> Result<()> {
        // Test that embeddings have expected properties
        let seq_len = 10;
        let embed_dim = 64;

        let embeddings = create_sinusoidal_position_embeddings(seq_len, embed_dim)?;

        // Convert to vec for easier inspection
        let emb_vec = embeddings.to_vec()?;

        // Check that values are in reasonable range [-1, 1] (sin/cos range)
        for &val in &emb_vec {
            assert!(
                val >= -1.0 && val <= 1.0,
                "Sinusoidal embedding values should be in [-1, 1], got: {}",
                val
            );
        }

        // Different positions should have different embeddings
        let pos0 = &emb_vec[0..embed_dim];
        let pos1 = &emb_vec[embed_dim..2 * embed_dim];

        let diff_count = pos0
            .iter()
            .zip(pos1.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-6)
            .count();

        assert!(
            diff_count > 0,
            "Different positions should have different embeddings"
        );

        Ok(())
    }
}
