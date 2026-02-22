//! Camera Embedding - Encodes camera parameters to embedding vectors
//!
//! This module implements camera parameter encoding for multi-view synthesis,
//! converting camera intrinsics and extrinsics into dense embeddings that can
//! condition the diffusion model.
//!
//! # Camera Parameters
//!
//! ## Intrinsics (4 values)
//! - fx, fy: Focal lengths in x and y
//! - cx, cy: Principal point coordinates
//!
//! ## Extrinsics (12 values)
//! - R: 3×3 rotation matrix (9 values)
//! - t: 3D translation vector (3 values)
//!
//! Total: 16 values per camera
//!
//! # Architecture
//!
//! 3-layer MLP with SiLU activation:
//! - Linear(16, 256) + SiLU
//! - Linear(256, 512) + SiLU
//! - Linear(512, embed_dim)
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_models::diffusion::CameraEmbedding;
//!
//! let camera_embed = CameraEmbedding::new(768)?;
//!
//! // Camera params: [B, V, 16] (batch, num_views, params)
//! let camera_params = create_camera_params(batch_size, num_views)?;
//! let embeddings = camera_embed.forward(&camera_params)?; // [B, V, 768]
//! ```

use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_core::DeviceType;
use torsh_nn::prelude::{Linear, SiLU};
use torsh_nn::{Module, ModuleBase, Parameter};
use torsh_tensor::Tensor;

/// Camera Embedding configuration
#[derive(Debug, Clone)]
pub struct CameraEmbeddingConfig {
    /// Output embedding dimension
    pub embed_dim: usize,
    /// Hidden dimensions for MLP layers
    pub hidden_dims: Vec<usize>,
}

impl Default for CameraEmbeddingConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            hidden_dims: vec![256, 512],
        }
    }
}

/// Camera Embedding - Encodes camera parameters to embeddings
///
/// Converts camera intrinsics (fx, fy, cx, cy) and extrinsics (R, t)
/// into dense embeddings for conditioning multi-view diffusion models.
pub struct CameraEmbedding {
    base: ModuleBase,
    config: CameraEmbeddingConfig,

    // MLP layers
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,

    // Activation
    activation: SiLU,
}

impl CameraEmbedding {
    /// Create a new CameraEmbedding
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Output embedding dimension (typically 768)
    ///
    /// # Returns
    ///
    /// Result containing the CameraEmbedding or an error
    pub fn new(embed_dim: usize) -> Result<Self> {
        let config = CameraEmbeddingConfig {
            embed_dim,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: CameraEmbeddingConfig) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Input: 16 camera parameters
        let input_dim = 16;
        let hidden1 = config.hidden_dims[0];
        let hidden2 = config.hidden_dims[1];

        // 3-layer MLP
        let fc1 = Linear::new(input_dim, hidden1, true);
        let fc2 = Linear::new(hidden1, hidden2, true);
        let fc3 = Linear::new(hidden2, config.embed_dim, true);

        let activation = SiLU::new();

        // Register parameters
        Self::register_module_params(&mut base, "fc1", &fc1);
        Self::register_module_params(&mut base, "fc2", &fc2);
        Self::register_module_params(&mut base, "fc3", &fc3);

        Ok(Self {
            base,
            config,
            fc1,
            fc2,
            fc3,
            activation,
        })
    }

    /// Helper to register module parameters
    fn register_module_params(base: &mut ModuleBase, prefix: &str, module: &dyn Module) {
        let params = module.named_parameters();
        for (name, param) in params {
            base.register_parameter(format!("{}.{}", prefix, name), param);
        }
    }

    /// Forward pass through camera embedding
    ///
    /// # Arguments
    ///
    /// * `camera_params` - Camera parameters [B, V, 16] or [B, 16]
    ///   - B: batch size
    ///   - V: number of views (optional)
    ///   - 16: [fx, fy, cx, cy, R_00, R_01, R_02, R_10, R_11, R_12, R_20, R_21, R_22, t_0, t_1, t_2]
    ///
    /// # Returns
    ///
    /// Camera embeddings:
    /// - If input is [B, V, 16]: returns [B, V, embed_dim]
    /// - If input is [B, 16]: returns [B, embed_dim]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input shape is invalid (not 2D or 3D)
    /// - Last dimension is not 16
    /// - Forward pass fails
    pub fn forward(&self, camera_params: &Tensor) -> Result<Tensor> {
        let shape = camera_params.shape();
        let ndim = shape.ndim();

        if ndim != 2 && ndim != 3 {
            return Err(TorshError::InvalidShape(format!(
                "Expected 2D or 3D input, got {}D: {:?}",
                ndim,
                shape.dims()
            )));
        }

        let param_dim = shape.dims()[ndim - 1];
        if param_dim != 16 {
            return Err(TorshError::InvalidShape(format!(
                "Expected 16 camera parameters, got {}",
                param_dim
            )));
        }

        // Handle 3D input [B, V, 16]
        if ndim == 3 {
            let batch_size = shape.dims()[0];
            let num_views = shape.dims()[1];

            // Reshape to 2D: [B, V, 16] → [B*V, 16]
            let params_2d = camera_params.reshape(&[(batch_size * num_views) as i32, 16])?;

            // Forward through MLP
            let embeddings_2d = self.forward_2d(&params_2d)?;

            // Reshape back: [B*V, embed_dim] → [B, V, embed_dim]
            embeddings_2d.reshape(&[
                batch_size as i32,
                num_views as i32,
                self.config.embed_dim as i32,
            ])
        } else {
            // Handle 2D input [B, 16]
            self.forward_2d(camera_params)
        }
    }

    /// Forward pass for 2D input [B, 16]
    fn forward_2d(&self, params: &Tensor) -> Result<Tensor> {
        // Layer 1: fc1 + SiLU
        let mut x = self.fc1.forward(params)?;
        x = self.activation.forward(&x)?;

        // Layer 2: fc2 + SiLU
        x = self.fc2.forward(&x)?;
        x = self.activation.forward(&x)?;

        // Layer 3: fc3 (no activation)
        self.fc3.forward(&x)
    }
}

impl Module for CameraEmbedding {
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
    fn test_camera_embedding_creation() {
        let embed = CameraEmbedding::new(768);
        assert!(embed.is_ok(), "Failed to create CameraEmbedding");
    }

    #[test]
    fn test_camera_embedding_shapes() {
        let embed = CameraEmbedding::new(768).expect("Failed to create embedding");

        // Test 2D input [B, 16]
        let params_2d = ones(&[4, 16]).expect("Failed to create params");
        let output_2d = embed.forward(&params_2d);
        assert!(output_2d.is_ok(), "2D forward failed");
        if let Ok(output) = output_2d {
            assert_eq!(output.shape().dims(), &[4, 768], "2D output shape mismatch");
        }

        // Test 3D input [B, V, 16]
        let params_3d = ones(&[2, 4, 16]).expect("Failed to create params");
        let output_3d = embed.forward(&params_3d);
        assert!(output_3d.is_ok(), "3D forward failed");
        if let Ok(output) = output_3d {
            assert_eq!(
                output.shape().dims(),
                &[2, 4, 768],
                "3D output shape mismatch"
            );
        }
    }

    #[test]
    fn test_single_view_vs_multiview() {
        let embed = CameraEmbedding::new(768).expect("Failed to create embedding");

        // Single view [B, 16]
        let single = ones(&[2, 16]).expect("Failed to create single");
        let single_out = embed.forward(&single).expect("Single view failed");

        // Multi-view [B, 1, 16]
        let multi = ones(&[2, 1, 16]).expect("Failed to create multi");
        let multi_out = embed.forward(&multi).expect("Multi-view failed");

        assert_eq!(single_out.shape().dims()[0], 2);
        assert_eq!(single_out.shape().dims()[1], 768);
        assert_eq!(multi_out.shape().dims()[0], 2);
        assert_eq!(multi_out.shape().dims()[1], 1);
        assert_eq!(multi_out.shape().dims()[2], 768);
    }

    #[test]
    fn test_camera_parameter_encoding() {
        let embed = CameraEmbedding::new(768).expect("Failed to create embedding");

        // Create realistic camera parameters
        let fx = 500.0f32;
        let fy = 500.0f32;
        let cx = 256.0f32;
        let cy = 256.0f32;

        // Identity rotation
        let r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        // Some translation
        let t = [0.0, 0.0, -5.0];

        let mut params = vec![fx, fy, cx, cy];
        params.extend_from_slice(&r);
        params.extend_from_slice(&t);

        let params_tensor = Tensor::from_vec(params, &[1, 16]).expect("Failed to create tensor");
        let output = embed.forward(&params_tensor);

        assert!(output.is_ok(), "Forward with realistic params failed");
        if let Ok(output) = output {
            assert_eq!(output.shape().dims(), &[1, 768]);
        }
    }

    #[test]
    fn test_different_embed_dims() {
        let embed_dims = [256, 512, 768, 1024];

        for &dim in &embed_dims {
            let embed = CameraEmbedding::new(dim).expect("Failed to create embedding");
            let params = ones(&[2, 16]).expect("Failed to create params");
            let output = embed.forward(&params).expect("Forward failed");

            assert_eq!(
                output.shape().dims()[1],
                dim,
                "Output dim mismatch for embed_dim={}",
                dim
            );
        }
    }

    #[test]
    fn test_batch_processing() {
        let embed = CameraEmbedding::new(768).expect("Failed to create embedding");

        let batch_sizes = [1, 2, 4, 8];

        for &batch_size in &batch_sizes {
            let params = ones(&[batch_size, 4, 16]).expect("Failed to create params");
            let output = embed.forward(&params).expect("Forward failed");

            assert_eq!(output.shape().dims()[0], batch_size, "Batch size mismatch");
            assert_eq!(output.shape().dims()[1], 4, "Num views mismatch");
            assert_eq!(output.shape().dims()[2], 768, "Embed dim mismatch");
        }
    }

    #[test]
    fn test_invalid_input_shape() {
        let embed = CameraEmbedding::new(768).expect("Failed to create embedding");

        // 1D input (should fail)
        let invalid = ones(&[16]).expect("Failed to create tensor");
        assert!(
            embed.forward(&invalid).is_err(),
            "Should fail with 1D input"
        );

        // Wrong param count (should fail)
        let invalid = ones(&[2, 12]).expect("Failed to create tensor");
        assert!(
            embed.forward(&invalid).is_err(),
            "Should fail with wrong param count"
        );
    }

    #[test]
    fn test_multiple_views() {
        let embed = CameraEmbedding::new(768).expect("Failed to create embedding");

        let num_views = [1, 2, 4, 8];

        for &views in &num_views {
            let params = ones(&[2, views, 16]).expect("Failed to create params");
            let output = embed.forward(&params).expect("Forward failed");

            assert_eq!(output.shape().dims()[1], views, "View count mismatch");
        }
    }
}
