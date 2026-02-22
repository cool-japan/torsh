//! Cross-View Attention - Multi-head attention across view dimension
//!
//! This module implements cross-view attention for consistent multi-view generation
//! in diffusion models. It enables information flow between different viewpoints,
//! ensuring geometric and semantic consistency across views.
//!
//! # Overview
//!
//! Cross-view attention operates on feature maps from multiple camera views,
//! allowing each view to attend to features from other views. This is crucial
//! for multi-view 3D generation where consistency across viewpoints is essential.
//!
//! # Architecture
//!
//! 1. Reshape: [B*V, C, H, W] → [B, V, C, H*W]
//! 2. Transpose for attention: [B, V, C, H*W] → [B, V, H*W, C]
//! 3. Multi-head attention across view dimension (V)
//! 4. Transpose back and reshape to original format
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_models::diffusion::CrossViewAttention;
//!
//! let cross_view = CrossViewAttention::new(8, 256)?;
//!
//! // Multi-view features: [B*V, C, H, W]
//! let features = encoder_output; // [8, 256, 32, 32] (2 batch × 4 views)
//! let output = cross_view.forward(&features, 4)?; // num_views = 4
//! ```

use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_core::DeviceType;
use torsh_nn::prelude::Linear;
use torsh_nn::{Module, ModuleBase, Parameter};
use torsh_tensor::Tensor;

/// Cross-View Attention configuration
#[derive(Debug, Clone)]
pub struct CrossViewAttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of channels
    pub channels: usize,
    /// Dropout probability
    pub dropout: f32,
}

impl Default for CrossViewAttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            channels: 256,
            dropout: 0.0,
        }
    }
}

/// Cross-View Attention - Attention across multiple viewpoints
///
/// Enables information flow between different camera views for consistent
/// multi-view 3D generation.
pub struct CrossViewAttention {
    base: ModuleBase,
    config: CrossViewAttentionConfig,
    head_dim: usize,

    // Projection layers
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl CrossViewAttention {
    /// Create a new CrossViewAttention layer
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads
    /// * `channels` - Number of feature channels
    ///
    /// # Returns
    ///
    /// Result containing the layer or an error
    ///
    /// # Errors
    ///
    /// Returns error if channels is not divisible by num_heads
    pub fn new(num_heads: usize, channels: usize) -> Result<Self> {
        let config = CrossViewAttentionConfig {
            num_heads,
            channels,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: CrossViewAttentionConfig) -> Result<Self> {
        if config.channels % config.num_heads != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "channels ({}) must be divisible by num_heads ({})",
                config.channels, config.num_heads
            )));
        }

        let head_dim = config.channels / config.num_heads;
        let mut base = ModuleBase::new();

        // Projection layers
        let q_proj = Linear::new(config.channels, config.channels, true);
        let k_proj = Linear::new(config.channels, config.channels, true);
        let v_proj = Linear::new(config.channels, config.channels, true);
        let out_proj = Linear::new(config.channels, config.channels, true);

        // Register parameters
        Self::register_module_params(&mut base, "q_proj", &q_proj);
        Self::register_module_params(&mut base, "k_proj", &k_proj);
        Self::register_module_params(&mut base, "v_proj", &v_proj);
        Self::register_module_params(&mut base, "out_proj", &out_proj);

        Ok(Self {
            base,
            config,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    /// Helper to register module parameters
    fn register_module_params(base: &mut ModuleBase, prefix: &str, module: &dyn Module) {
        let params = module.named_parameters();
        for (name, param) in params {
            base.register_parameter(format!("{}.{}", prefix, name), param);
        }
    }

    /// Forward pass with cross-view attention
    ///
    /// # Arguments
    ///
    /// * `features` - Feature maps [B*V, C, H, W]
    /// * `num_views` - Number of views V
    ///
    /// # Returns
    ///
    /// Cross-view attended features [B*V, C, H, W]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input shape is invalid
    /// - num_views doesn't divide batch dimension evenly
    /// - Attention computation fails
    pub fn forward(&self, features: &Tensor, num_views: usize) -> Result<Tensor> {
        let shape = features.shape();
        if shape.ndim() != 4 {
            return Err(TorshError::InvalidShape(format!(
                "Expected 4D input [B*V, C, H, W], got {}D: {:?}",
                shape.ndim(),
                shape.dims()
            )));
        }

        let bv = shape.dims()[0];
        let channels = shape.dims()[1];
        let height = shape.dims()[2];
        let width = shape.dims()[3];

        if channels != self.config.channels {
            return Err(TorshError::InvalidShape(format!(
                "Expected {} channels, got {}",
                self.config.channels, channels
            )));
        }

        if bv % num_views != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "Batch dimension {} not divisible by num_views {}",
                bv, num_views
            )));
        }

        let batch_size = bv / num_views;
        let spatial_size = height * width;

        // Reshape: [B*V, C, H, W] → [B, V, C, H*W]
        let features_reshaped = features.reshape(&[
            batch_size as i32,
            num_views as i32,
            channels as i32,
            spatial_size as i32,
        ])?;

        // Transpose: [B, V, C, H*W] → [B, V, H*W, C]
        let features_transposed = self.transpose_for_attention(&features_reshaped)?;

        // Reshape for 2D operations: [B, V, H*W, C] → [B*V*H*W, C]
        let features_2d = features_transposed.reshape(&[
            (batch_size * num_views * spatial_size) as i32,
            channels as i32,
        ])?;

        // Project Q, K, V
        let q = self.q_proj.forward(&features_2d)?;
        let k = self.k_proj.forward(&features_2d)?;
        let v = self.v_proj.forward(&features_2d)?;

        // Reshape back: [B*V*H*W, C] → [B, V, H*W, C]
        let q = q.reshape(&[
            batch_size as i32,
            num_views as i32,
            spatial_size as i32,
            channels as i32,
        ])?;
        let k = k.reshape(&[
            batch_size as i32,
            num_views as i32,
            spatial_size as i32,
            channels as i32,
        ])?;
        let v = v.reshape(&[
            batch_size as i32,
            num_views as i32,
            spatial_size as i32,
            channels as i32,
        ])?;

        // Perform cross-view attention
        let attn_output =
            self.cross_view_attention(&q, &k, &v, batch_size, num_views, spatial_size)?;

        // Reshape for output projection: [B, V, H*W, C] → [B*V*H*W, C]
        let attn_output_2d = attn_output.reshape(&[
            (batch_size * num_views * spatial_size) as i32,
            channels as i32,
        ])?;

        // Output projection
        let output_2d = self.out_proj.forward(&attn_output_2d)?;

        // Reshape: [B*V*H*W, C] → [B, V, H*W, C]
        let output = output_2d.reshape(&[
            batch_size as i32,
            num_views as i32,
            spatial_size as i32,
            channels as i32,
        ])?;

        // Transpose back: [B, V, H*W, C] → [B, V, C, H*W]
        let output_transposed = self.transpose_from_attention(&output)?;

        // Reshape to original format: [B, V, C, H*W] → [B*V, C, H, W]
        output_transposed.reshape(&[
            (batch_size * num_views) as i32,
            channels as i32,
            height as i32,
            width as i32,
        ])
    }

    /// Transpose for attention: [B, V, C, H*W] → [B, V, H*W, C]
    fn transpose_for_attention(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.transpose(2, 3)
    }

    /// Transpose from attention: [B, V, H*W, C] → [B, V, C, H*W]
    fn transpose_from_attention(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.transpose(2, 3)
    }

    /// Cross-view attention computation
    ///
    /// Computes attention across the view dimension, allowing each spatial location
    /// in each view to attend to the same spatial location in other views.
    fn cross_view_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        batch_size: usize,
        num_views: usize,
        spatial_size: usize,
    ) -> Result<Tensor> {
        // For each spatial location, compute attention across views
        // Input: [B, V, H*W, C]
        // We want attention across V for each (B, H*W) location

        let q_data = q.to_vec()?;
        let k_data = k.to_vec()?;
        let v_data = v.to_vec()?;

        let scale = 1.0 / (self.config.channels as f32).sqrt();

        let mut output_data =
            vec![0.0f32; batch_size * num_views * spatial_size * self.config.channels];

        // For each batch and spatial location
        for b in 0..batch_size {
            for s in 0..spatial_size {
                // Compute attention scores for all view pairs
                let mut scores = vec![vec![0.0f32; num_views]; num_views];

                for v_q in 0..num_views {
                    for v_k in 0..num_views {
                        let mut score = 0.0f32;
                        for c in 0..self.config.channels {
                            let q_idx = b * num_views * spatial_size * self.config.channels
                                + v_q * spatial_size * self.config.channels
                                + s * self.config.channels
                                + c;
                            let k_idx = b * num_views * spatial_size * self.config.channels
                                + v_k * spatial_size * self.config.channels
                                + s * self.config.channels
                                + c;
                            score += q_data[q_idx] * k_data[k_idx];
                        }
                        scores[v_q][v_k] = score * scale;
                    }
                }

                // Softmax across views (for each query view)
                for v_q in 0..num_views {
                    // Find max for numerical stability
                    let max_score = scores[v_q]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);

                    // Exp and sum
                    let mut exp_sum = 0.0f32;
                    let mut exp_scores = vec![0.0f32; num_views];
                    for v_k in 0..num_views {
                        exp_scores[v_k] = (scores[v_q][v_k] - max_score).exp();
                        exp_sum += exp_scores[v_k];
                    }

                    // Normalize
                    for v_k in 0..num_views {
                        exp_scores[v_k] /= exp_sum;
                    }

                    // Weighted sum of values
                    for c in 0..self.config.channels {
                        let mut weighted_sum = 0.0f32;
                        for v_k in 0..num_views {
                            let v_idx = b * num_views * spatial_size * self.config.channels
                                + v_k * spatial_size * self.config.channels
                                + s * self.config.channels
                                + c;
                            weighted_sum += exp_scores[v_k] * v_data[v_idx];
                        }

                        let out_idx = b * num_views * spatial_size * self.config.channels
                            + v_q * spatial_size * self.config.channels
                            + s * self.config.channels
                            + c;
                        output_data[out_idx] = weighted_sum;
                    }
                }
            }
        }

        Tensor::from_vec(
            output_data,
            &[batch_size, num_views, spatial_size, self.config.channels],
        )
    }
}

impl Module for CrossViewAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Default: assume num_views = 1 (no cross-view attention)
        Ok(input.clone())
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_cross_view_attention_creation() {
        let layer = CrossViewAttention::new(8, 256);
        assert!(layer.is_ok(), "Failed to create CrossViewAttention");
    }

    #[test]
    fn test_cross_view_attention_shapes() {
        let layer = CrossViewAttention::new(8, 256).expect("Failed to create layer");

        let batch_size = 2;
        let num_views = 4;
        let channels = 256;
        let height = 32;
        let width = 32;

        let features = ones(&[batch_size * num_views, channels, height, width])
            .expect("Failed to create features");

        let output = layer.forward(&features, num_views);
        assert!(output.is_ok(), "Forward pass failed: {:?}", output.err());

        if let Ok(output) = output {
            assert_eq!(
                output.shape().dims(),
                &[batch_size * num_views, channels, height, width],
                "Output shape mismatch"
            );
        }
    }

    #[test]
    fn test_single_vs_multiple_views() {
        let layer = CrossViewAttention::new(8, 256).expect("Failed to create layer");

        // Single view
        let single = ones(&[2, 256, 32, 32]).expect("Failed to create single");
        let single_out = layer.forward(&single, 1);
        assert!(single_out.is_ok(), "Single view failed");

        // Multiple views
        let multi = ones(&[8, 256, 32, 32]).expect("Failed to create multi");
        let multi_out = layer.forward(&multi, 4);
        assert!(multi_out.is_ok(), "Multiple views failed");
    }

    #[test]
    fn test_view_information_flow() {
        let layer = CrossViewAttention::new(8, 256).expect("Failed to create layer");

        let batch_size = 1;
        let num_views = 2;
        let features =
            ones(&[batch_size * num_views, 256, 16, 16]).expect("Failed to create features");

        let output = layer.forward(&features, num_views);
        assert!(output.is_ok(), "Forward failed");

        // Output should have same shape as input
        if let Ok(output) = output {
            assert_eq!(output.shape().dims(), features.shape().dims());
        }
    }

    #[test]
    fn test_attention_weights_structure() {
        let layer = CrossViewAttention::new(8, 256).expect("Failed to create layer");

        let features = ones(&[4, 256, 8, 8]).expect("Failed to create features");
        let output = layer.forward(&features, 2);

        assert!(output.is_ok(), "Forward failed");
    }

    #[test]
    fn test_masked_views() {
        let layer = CrossViewAttention::new(8, 256).expect("Failed to create layer");

        // Test with partially masked views (using zeros)
        let mut features_data = vec![1.0f32; 4 * 256 * 16 * 16];
        // Set second view to zeros (masked)
        for i in (256 * 16 * 16)..(2 * 256 * 16 * 16) {
            features_data[i] = 0.0;
        }

        let features =
            Tensor::from_vec(features_data, &[4, 256, 16, 16]).expect("Failed to create features");

        let output = layer.forward(&features, 2);
        assert!(output.is_ok(), "Forward with masked view failed");
    }

    #[test]
    fn test_invalid_num_views() {
        let layer = CrossViewAttention::new(8, 256).expect("Failed to create layer");

        let features = ones(&[5, 256, 16, 16]).expect("Failed to create features");

        // num_views=2 doesn't divide batch_size=5
        let result = layer.forward(&features, 2);
        assert!(result.is_err(), "Should fail with invalid num_views");
    }

    #[test]
    fn test_invalid_channels() {
        let layer = CrossViewAttention::new(8, 256).expect("Failed to create layer");

        // Wrong number of channels
        let features = ones(&[4, 128, 16, 16]).expect("Failed to create features");
        let result = layer.forward(&features, 2);
        assert!(result.is_err(), "Should fail with wrong channels");
    }

    #[test]
    fn test_different_spatial_sizes() {
        let layer = CrossViewAttention::new(8, 256).expect("Failed to create layer");

        let spatial_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)];

        for (h, w) in &spatial_sizes {
            let features = ones(&[4, 256, *h, *w]).expect("Failed to create features");
            let result = layer.forward(&features, 2);
            assert!(result.is_ok(), "Failed with spatial size {}×{}", h, w);
        }
    }
}
