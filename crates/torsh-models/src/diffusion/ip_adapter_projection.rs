//! IP-Adapter Projection - Image feature projection for identity preservation
//!
//! This module implements the projection layer that transforms CLIP image
//! features into the diffusion model's conditioning space, enabling
//! identity-preserving image conditioning.
//!
//! # Architecture
//!
//! The projection consists of:
//! - Layer 1: Linear(1024, 2048) + GELU
//! - Layer 2: Linear(2048, cross_attn_dim * num_tokens)
//! - Reshape to [B, num_tokens, cross_attn_dim]
//!
//! # Parameters
//!
//! Approximately 22M parameters for typical configuration:
//! - num_tokens = 16
//! - cross_attn_dim = 768
//! - CLIP feature dim = 1024
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_models::diffusion::IPAdapterProjection;
//!
//! let projection = IPAdapterProjection::new(16, 768)?;
//! let clip_features = clip_model.forward(&image)?; // [B, 257, 1024]
//! let projected = projection.forward(&clip_features)?; // [B, 16, 768]
//! ```

use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_core::DeviceType;
use torsh_nn::prelude::{Linear, GELU};
use torsh_nn::{Module, ModuleBase, Parameter};
use torsh_tensor::Tensor;

/// IP-Adapter Projection configuration
#[derive(Debug, Clone)]
pub struct IPAdapterProjectionConfig {
    /// Number of output tokens for cross-attention
    pub num_tokens: usize,
    /// Cross-attention dimension in the diffusion model
    pub cross_attn_dim: usize,
    /// CLIP feature dimension (typically 1024 for ViT-L/14)
    pub clip_feature_dim: usize,
    /// Hidden dimension for MLP (default: 2048)
    pub hidden_dim: usize,
}

impl Default for IPAdapterProjectionConfig {
    fn default() -> Self {
        Self {
            num_tokens: 16,
            cross_attn_dim: 768,
            clip_feature_dim: 1024,
            hidden_dim: 2048,
        }
    }
}

impl IPAdapterProjectionConfig {
    /// Configuration for Stable Diffusion 1.5 with CLIP ViT-L/14
    pub fn sd15_vit_l() -> Self {
        Self {
            num_tokens: 16,
            cross_attn_dim: 768,
            clip_feature_dim: 1024,
            hidden_dim: 2048,
        }
    }

    /// Configuration for SDXL with CLIP ViT-bigG
    pub fn sdxl_vit_g() -> Self {
        Self {
            num_tokens: 16,
            cross_attn_dim: 2048,
            clip_feature_dim: 1280,
            hidden_dim: 3072,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.num_tokens == 0 {
            return Err(TorshError::InvalidArgument(
                "num_tokens must be greater than 0".to_string(),
            ));
        }
        if self.cross_attn_dim == 0 {
            return Err(TorshError::InvalidArgument(
                "cross_attn_dim must be greater than 0".to_string(),
            ));
        }
        if self.clip_feature_dim == 0 {
            return Err(TorshError::InvalidArgument(
                "clip_feature_dim must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// IP-Adapter Projection - Projects CLIP features to diffusion space
///
/// This module takes CLIP image features [B, N, clip_dim] and projects them
/// to [B, num_tokens, cross_attn_dim] for use in cross-attention layers.
pub struct IPAdapterProjection {
    base: ModuleBase,
    config: IPAdapterProjectionConfig,

    // MLP layers
    fc1: Linear,
    fc2: Linear,

    // Activation
    activation: GELU,
}

impl IPAdapterProjection {
    /// Create a new IPAdapterProjection with default configuration
    ///
    /// # Arguments
    ///
    /// * `num_tokens` - Number of output tokens (typically 16)
    /// * `cross_attn_dim` - Cross-attention dimension (typically 768 for SD1.5)
    ///
    /// # Returns
    ///
    /// Result containing the IPAdapterProjection or an error
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let projection = IPAdapterProjection::new(16, 768)?;
    /// ```
    pub fn new(num_tokens: usize, cross_attn_dim: usize) -> Result<Self> {
        let config = IPAdapterProjectionConfig {
            num_tokens,
            cross_attn_dim,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new IPAdapterProjection with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the projection
    ///
    /// # Returns
    ///
    /// Result containing the IPAdapterProjection or an error
    pub fn with_config(config: IPAdapterProjectionConfig) -> Result<Self> {
        config.validate()?;

        let mut base = ModuleBase::new();

        // Layer 1: clip_feature_dim → hidden_dim
        let fc1 = Linear::new(config.clip_feature_dim, config.hidden_dim, true);

        // Layer 2: hidden_dim → num_tokens * cross_attn_dim
        let output_dim = config.num_tokens * config.cross_attn_dim;
        let fc2 = Linear::new(config.hidden_dim, output_dim, true);

        let activation = GELU::new(false); // False = approximate GELU

        // Register parameters
        Self::register_module_params(&mut base, "fc1", &fc1);
        Self::register_module_params(&mut base, "fc2", &fc2);

        Ok(Self {
            base,
            config,
            fc1,
            fc2,
            activation,
        })
    }

    /// Helper to register module parameters with prefix
    fn register_module_params(base: &mut ModuleBase, prefix: &str, module: &dyn Module) {
        let params = module.named_parameters();
        for (name, param) in params {
            base.register_parameter(format!("{}.{}", prefix, name), param);
        }
    }

    /// Forward pass through the projection
    ///
    /// # Arguments
    ///
    /// * `clip_features` - CLIP image features [B, N, clip_dim]
    ///   - For CLIP ViT-L/14: [B, 257, 1024] (1 CLS + 256 patches)
    ///   - The CLS token and patch tokens are both used
    ///
    /// # Returns
    ///
    /// Projected features [B, num_tokens, cross_attn_dim]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input shape is incorrect
    /// - Feature dimension doesn't match config
    /// - Forward pass fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let clip_features = clip_model.forward(&image)?; // [2, 257, 1024]
    /// let projected = projection.forward(&clip_features)?; // [2, 16, 768]
    /// ```
    pub fn forward(&self, clip_features: &Tensor) -> Result<Tensor> {
        let input_shape = clip_features.shape();
        if input_shape.ndim() != 3 {
            return Err(TorshError::InvalidShape(format!(
                "Expected 3D input [B, N, clip_dim], got shape {:?}",
                input_shape.dims()
            )));
        }

        let batch_size = input_shape.dims()[0];
        let feature_dim = input_shape.dims()[2];

        if feature_dim != self.config.clip_feature_dim {
            return Err(TorshError::InvalidShape(format!(
                "Expected feature dimension {}, got {}",
                self.config.clip_feature_dim, feature_dim
            )));
        }

        // Global average pooling across patch dimension [B, N, D] → [B, D]
        // This reduces the 257 tokens to a single feature vector
        let pooled = self.global_average_pool(clip_features)?;

        // MLP projection: [B, clip_dim] → [B, hidden_dim] → [B, num_tokens * cross_attn_dim]
        let mut x = self.fc1.forward(&pooled)?;
        x = self.activation.forward(&x)?;
        x = self.fc2.forward(&x)?;

        // Reshape: [B, num_tokens * cross_attn_dim] → [B, num_tokens, cross_attn_dim]
        let output_shape = [
            batch_size,
            self.config.num_tokens,
            self.config.cross_attn_dim,
        ];
        let result = x.reshape(&output_shape.map(|x| x as i32))?;

        Ok(result)
    }

    /// Global average pooling across the patch dimension
    ///
    /// Reduces [B, N, D] to [B, D] by averaging across N patches.
    fn global_average_pool(&self, features: &Tensor) -> Result<Tensor> {
        let shape = features.shape();
        let batch_size = shape.dims()[0];
        let num_patches = shape.dims()[1];
        let feature_dim = shape.dims()[2];

        // Get data and perform manual averaging
        let data = features.to_vec()?;

        let mut pooled_data = vec![0.0f32; batch_size * feature_dim];

        for b in 0..batch_size {
            for d in 0..feature_dim {
                let mut sum = 0.0f32;
                for n in 0..num_patches {
                    let idx = b * num_patches * feature_dim + n * feature_dim + d;
                    sum += data[idx];
                }
                let pooled_idx = b * feature_dim + d;
                pooled_data[pooled_idx] = sum / num_patches as f32;
            }
        }

        Tensor::from_vec(pooled_data, &[batch_size, feature_dim])
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let fc1_params =
            self.config.clip_feature_dim * self.config.hidden_dim + self.config.hidden_dim;
        let fc2_params = self.config.hidden_dim
            * (self.config.num_tokens * self.config.cross_attn_dim)
            + (self.config.num_tokens * self.config.cross_attn_dim);
        fc1_params + fc2_params
    }
}

impl Module for IPAdapterProjection {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input)
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
    fn test_ip_adapter_projection_creation() {
        let projection = IPAdapterProjection::new(16, 768);
        assert!(projection.is_ok(), "Failed to create IPAdapterProjection");
    }

    #[test]
    fn test_ip_adapter_projection_shapes() {
        let projection = IPAdapterProjection::new(16, 768).expect("Failed to create projection");

        // CLIP ViT-L/14 features: [B, 257, 1024]
        let batch_size = 2;
        let clip_features = ones(&[batch_size, 257, 1024]).expect("Failed to create features");

        let output = projection.forward(&clip_features);
        assert!(output.is_ok(), "Forward pass failed");

        if let Ok(output) = output {
            let output_shape = output.shape();
            assert_eq!(
                output_shape.dims(),
                &[batch_size, 16, 768],
                "Output shape mismatch"
            );
        }
    }

    #[test]
    fn test_clip_feature_compatibility() {
        let projection = IPAdapterProjection::new(16, 768).expect("Failed to create projection");

        // Test with different CLIP configurations
        let test_cases = vec![
            (1, 257, 1024), // Single image, ViT-L/14
            (4, 257, 1024), // Batch of 4, ViT-L/14
            (2, 197, 1024), // ViT-L/14 at different resolution
        ];

        for (batch, patches, dim) in test_cases {
            let features = ones(&[batch, patches, dim]).expect("Failed to create features");
            let result = projection.forward(&features);
            assert!(
                result.is_ok(),
                "Failed for input shape [{}, {}, {}]",
                batch,
                patches,
                dim
            );

            if let Ok(output) = result {
                assert_eq!(output.shape().dims()[0], batch, "Batch size mismatch");
                assert_eq!(output.shape().dims()[1], 16, "Token count mismatch");
                assert_eq!(output.shape().dims()[2], 768, "Feature dim mismatch");
            }
        }
    }

    #[test]
    fn test_different_token_counts() {
        let token_counts = [4, 8, 16, 32];

        for num_tokens in &token_counts {
            let projection =
                IPAdapterProjection::new(*num_tokens, 768).expect("Failed to create projection");

            let features = ones(&[2, 257, 1024]).expect("Failed to create features");
            let output = projection.forward(&features).expect("Forward failed");

            assert_eq!(
                output.shape().dims()[1],
                *num_tokens,
                "Token count mismatch"
            );
        }
    }

    #[test]
    fn test_batch_processing() {
        let projection = IPAdapterProjection::new(16, 768).expect("Failed to create projection");

        let batch_sizes = [1, 2, 4, 8];

        for batch_size in &batch_sizes {
            let features = ones(&[*batch_size, 257, 1024]).expect("Failed to create features");
            let output = projection.forward(&features).expect("Forward failed");

            assert_eq!(output.shape().dims()[0], *batch_size, "Batch size mismatch");
        }
    }

    #[test]
    fn test_invalid_input_shape() {
        let projection = IPAdapterProjection::new(16, 768).expect("Failed to create projection");

        // Test with 2D input (should fail)
        let invalid = ones(&[2, 1024]).expect("Failed to create tensor");
        let result = projection.forward(&invalid);
        assert!(result.is_err(), "Should fail with 2D input");

        // Test with 4D input (should fail)
        let invalid = ones(&[2, 257, 1024, 1]).expect("Failed to create tensor");
        let result = projection.forward(&invalid);
        assert!(result.is_err(), "Should fail with 4D input");
    }

    #[test]
    fn test_invalid_feature_dimension() {
        let projection = IPAdapterProjection::new(16, 768).expect("Failed to create projection");

        // Wrong feature dimension (512 instead of 1024)
        let invalid = ones(&[2, 257, 512]).expect("Failed to create tensor");
        let result = projection.forward(&invalid);
        assert!(result.is_err(), "Should fail with wrong feature dimension");
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = IPAdapterProjectionConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: num_tokens = 0
        let invalid = IPAdapterProjectionConfig {
            num_tokens: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        // Invalid: cross_attn_dim = 0
        let invalid = IPAdapterProjectionConfig {
            cross_attn_dim: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_parameter_count() {
        let projection = IPAdapterProjection::new(16, 768).expect("Failed to create projection");

        let num_params = projection.num_parameters();

        // fc1: (1024 * 2048) + 2048 = 2,099,200
        // fc2: (2048 * 16 * 768) + (16 * 768) = 25,177,088 + 12,288 = 25,189,376
        // Total: ~27.3M parameters
        let expected = (1024 * 2048 + 2048) + (2048 * 16 * 768 + 16 * 768);
        assert_eq!(num_params, expected, "Parameter count mismatch");
    }

    #[test]
    fn test_sd15_config() {
        let config = IPAdapterProjectionConfig::sd15_vit_l();
        assert_eq!(config.num_tokens, 16);
        assert_eq!(config.cross_attn_dim, 768);
        assert_eq!(config.clip_feature_dim, 1024);
    }

    #[test]
    fn test_sdxl_config() {
        let config = IPAdapterProjectionConfig::sdxl_vit_g();
        assert_eq!(config.num_tokens, 16);
        assert_eq!(config.cross_attn_dim, 2048);
        assert_eq!(config.clip_feature_dim, 1280);
    }
}
