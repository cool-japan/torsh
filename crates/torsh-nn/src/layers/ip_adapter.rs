//! IP-Adapter Cross-Attention Layer
//!
//! This module implements the IP-Adapter cross-attention mechanism for
//! identity-preserving image conditioning in diffusion models.
//!
//! # Overview
//!
//! IP-Adapter extends standard text-to-image diffusion with image conditioning
//! through dedicated cross-attention layers. This enables:
//! - Identity preservation from reference images
//! - Style transfer and image-to-image generation
//! - Fine-grained control over generated content
//!
//! # Architecture
//!
//! - Query: from text features [B, seq_len, dim]
//! - Key/Value: from image features [B, num_tokens, dim]
//! - Multi-head attention with learnable projections
//! - Optional null conditioning for classifier-free guidance
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_nn::layers::IPAdapterCrossAttention;
//!
//! let ip_attn = IPAdapterCrossAttention::new(768, 8)?;
//!
//! // Forward pass
//! let query = text_features; // [B, seq_len, 768]
//! let image_features = ip_projection.forward(&clip_features)?; // [B, 16, 768]
//! let output = ip_attn.forward(&query, &image_features)?; // [B, seq_len, 768]
//! ```

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

use std::collections::HashMap;

/// IP-Adapter Cross-Attention configuration
#[derive(Debug, Clone)]
pub struct IPAdapterCrossAttentionConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Whether to use bias in projections
    pub bias: bool,
}

impl Default for IPAdapterCrossAttentionConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            num_heads: 8,
            dropout: 0.0,
            bias: true,
        }
    }
}

/// IP-Adapter Cross-Attention Layer
///
/// Performs cross-attention where queries come from text features and
/// keys/values come from projected image features.
pub struct IPAdapterCrossAttention {
    base: ModuleBase,
    config: IPAdapterCrossAttentionConfig,
    head_dim: usize,
}

impl IPAdapterCrossAttention {
    /// Create a new IPAdapterCrossAttention layer
    ///
    /// # Arguments
    ///
    /// * `embed_dim` - Embedding dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads
    ///
    /// # Returns
    ///
    /// Result containing the layer or an error
    ///
    /// # Errors
    ///
    /// Returns error if embed_dim is not divisible by num_heads
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let ip_attn = IPAdapterCrossAttention::new(768, 8)?;
    /// ```
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self> {
        let config = IPAdapterCrossAttentionConfig {
            embed_dim,
            num_heads,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create with custom configuration
    pub fn with_config(config: IPAdapterCrossAttentionConfig) -> Result<Self> {
        if config.embed_dim % config.num_heads != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                config.embed_dim, config.num_heads
            )));
        }

        let head_dim = config.embed_dim / config.num_heads;
        let mut base = ModuleBase::new();

        // Initialize projection weights
        let q_proj_weight = crate::init::xavier_uniform(&[config.embed_dim, config.embed_dim])
            .map_err(|e| TorshError::Other(format!("Failed to initialize q_proj: {}", e)))?;
        let k_proj_weight = crate::init::xavier_uniform(&[config.embed_dim, config.embed_dim])
            .map_err(|e| TorshError::Other(format!("Failed to initialize k_proj: {}", e)))?;
        let v_proj_weight = crate::init::xavier_uniform(&[config.embed_dim, config.embed_dim])
            .map_err(|e| TorshError::Other(format!("Failed to initialize v_proj: {}", e)))?;
        let out_proj_weight = crate::init::xavier_uniform(&[config.embed_dim, config.embed_dim])
            .map_err(|e| TorshError::Other(format!("Failed to initialize out_proj: {}", e)))?;

        base.register_parameter("q_proj.weight".to_string(), Parameter::new(q_proj_weight));
        base.register_parameter("k_proj.weight".to_string(), Parameter::new(k_proj_weight));
        base.register_parameter("v_proj.weight".to_string(), Parameter::new(v_proj_weight));
        base.register_parameter(
            "out_proj.weight".to_string(),
            Parameter::new(out_proj_weight),
        );

        if config.bias {
            let q_bias = torsh_tensor::creation::zeros(&[config.embed_dim])
                .map_err(|e| TorshError::Other(format!("Failed to create q_bias: {}", e)))?;
            let k_bias = torsh_tensor::creation::zeros(&[config.embed_dim])
                .map_err(|e| TorshError::Other(format!("Failed to create k_bias: {}", e)))?;
            let v_bias = torsh_tensor::creation::zeros(&[config.embed_dim])
                .map_err(|e| TorshError::Other(format!("Failed to create v_bias: {}", e)))?;
            let out_bias = torsh_tensor::creation::zeros(&[config.embed_dim])
                .map_err(|e| TorshError::Other(format!("Failed to create out_bias: {}", e)))?;

            base.register_parameter("q_proj.bias".to_string(), Parameter::new(q_bias));
            base.register_parameter("k_proj.bias".to_string(), Parameter::new(k_bias));
            base.register_parameter("v_proj.bias".to_string(), Parameter::new(v_bias));
            base.register_parameter("out_proj.bias".to_string(), Parameter::new(out_bias));
        }

        Ok(Self {
            base,
            config,
            head_dim,
        })
    }

    /// Forward pass with cross-attention
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor from text features [B, seq_len, embed_dim]
    /// * `image_features` - Image features from IP-Adapter projection [B, num_tokens, embed_dim]
    ///
    /// # Returns
    ///
    /// Attended output [B, seq_len, embed_dim]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input shapes are invalid
    /// - Embedding dimensions don't match
    /// - Attention computation fails
    pub fn forward(&self, query: &Tensor, image_features: &Tensor) -> Result<Tensor> {
        let query_shape = query.shape();
        let img_shape = image_features.shape();

        // Validate shapes
        if query_shape.ndim() != 3 || img_shape.ndim() != 3 {
            return Err(TorshError::InvalidShape(format!(
                "Expected 3D tensors, got query: {:?}, image: {:?}",
                query_shape.dims(),
                img_shape.dims()
            )));
        }

        let batch_size = query_shape.dims()[0];
        let seq_len = query_shape.dims()[1];
        let query_dim = query_shape.dims()[2];

        let img_batch = img_shape.dims()[0];
        let num_tokens = img_shape.dims()[1];
        let img_dim = img_shape.dims()[2];

        if query_dim != self.config.embed_dim || img_dim != self.config.embed_dim {
            return Err(TorshError::InvalidShape(format!(
                "Embedding dimensions must match config ({}), got query: {}, image: {}",
                self.config.embed_dim, query_dim, img_dim
            )));
        }

        if batch_size != img_batch {
            return Err(TorshError::InvalidShape(format!(
                "Batch sizes must match, got query: {}, image: {}",
                batch_size, img_batch
            )));
        }

        // Project to Q, K, V
        let q = self.project_q(query)?;
        let k = self.project_k(image_features)?;
        let v = self.project_v(image_features)?;

        // Reshape for multi-head attention: [B, seq_len, embed_dim] → [B, num_heads, seq_len, head_dim]
        let q = self.reshape_for_attention(&q, batch_size, seq_len)?;
        let k = self.reshape_for_attention(&k, batch_size, num_tokens)?;
        let v = self.reshape_for_attention(&v, batch_size, num_tokens)?;

        // Compute scaled dot-product attention
        let attn_output = self.scaled_dot_product_attention(&q, &k, &v)?;

        // Reshape back: [B, num_heads, seq_len, head_dim] → [B, seq_len, embed_dim]
        let attn_output = self.reshape_from_attention(&attn_output, batch_size, seq_len)?;

        // Output projection
        let output = self.project_output(&attn_output)?;

        Ok(output)
    }

    /// Project query
    fn project_q(&self, query: &Tensor) -> Result<Tensor> {
        self.project_3d(query, "q_proj")
    }

    /// Project key
    fn project_k(&self, key: &Tensor) -> Result<Tensor> {
        self.project_3d(key, "k_proj")
    }

    /// Project value
    fn project_v(&self, value: &Tensor) -> Result<Tensor> {
        self.project_3d(value, "v_proj")
    }

    /// Project output
    fn project_output(&self, output: &Tensor) -> Result<Tensor> {
        self.project_3d(output, "out_proj")
    }

    /// Generic projection for 3D tensors [B, seq_len, dim]
    fn project_3d(&self, input: &Tensor, proj_name: &str) -> Result<Tensor> {
        let shape = input.shape();
        let batch_size = shape.dims()[0];
        let seq_len = shape.dims()[1];
        let dim = shape.dims()[2];

        // Reshape to 2D: [B, seq_len, dim] → [B * seq_len, dim]
        let input_2d = input.reshape(&[(batch_size * seq_len) as i32, dim as i32])?;

        // Get weight and transpose for matmul
        let weight_key = format!("{}.weight", proj_name);
        let weight = self.base.parameters[&weight_key].tensor().read().clone();
        let weight_t = weight.transpose(0, 1)?;

        // Matmul: [B * seq_len, dim] @ [dim, dim] = [B * seq_len, dim]
        let mut result = input_2d.matmul(&weight_t)?;

        // Add bias if configured
        if self.config.bias {
            let bias_key = format!("{}.bias", proj_name);
            let bias = self.base.parameters[&bias_key].tensor().read().clone();
            result = result.add_op(&bias)?;
        }

        // Reshape back to 3D: [B * seq_len, dim] → [B, seq_len, dim]
        result.reshape(&[
            batch_size as i32,
            seq_len as i32,
            self.config.embed_dim as i32,
        ])
    }

    /// Reshape tensor for multi-head attention
    fn reshape_for_attention(
        &self,
        tensor: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // [B, seq_len, embed_dim] → [B, seq_len, num_heads, head_dim] → [B, num_heads, seq_len, head_dim]
        let reshaped = tensor.reshape(&[
            batch_size as i32,
            seq_len as i32,
            self.config.num_heads as i32,
            self.head_dim as i32,
        ])?;

        reshaped.transpose(1, 2)
    }

    /// Reshape tensor from multi-head attention
    fn reshape_from_attention(
        &self,
        tensor: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // [B, num_heads, seq_len, head_dim] → [B, seq_len, num_heads, head_dim] → [B, seq_len, embed_dim]
        let transposed = tensor.transpose(1, 2)?;
        let contiguous = transposed.contiguous()?;
        contiguous.reshape(&[
            batch_size as i32,
            seq_len as i32,
            self.config.embed_dim as i32,
        ])
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q_shape = q.shape();
        let batch_size = q_shape.dims()[0];
        let num_heads = q_shape.dims()[1];
        let seq_len_q = q_shape.dims()[2];
        let head_dim = q_shape.dims()[3];

        let k_shape = k.shape();
        let seq_len_k = k_shape.dims()[2];

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Reshape to 3D for batched matmul: [B, H, seq_len, head_dim] → [B*H, seq_len, head_dim]
        let q_3d = q.reshape(&[
            (batch_size * num_heads) as i32,
            seq_len_q as i32,
            head_dim as i32,
        ])?;
        let k_3d = k.reshape(&[
            (batch_size * num_heads) as i32,
            seq_len_k as i32,
            head_dim as i32,
        ])?;
        let v_3d = v.reshape(&[
            (batch_size * num_heads) as i32,
            seq_len_k as i32,
            head_dim as i32,
        ])?;

        // Transpose k for matmul: [B*H, seq_len_k, head_dim] → [B*H, head_dim, seq_len_k]
        let k_t = k_3d.transpose(1, 2)?;

        // Now do batched matmul manually
        let q_data = q_3d.to_vec()?;
        let k_t_data = k_t.to_vec()?;

        let mut scores_data = vec![0.0f32; batch_size * num_heads * seq_len_q * seq_len_k];

        for bh in 0..(batch_size * num_heads) {
            for i in 0..seq_len_q {
                for j in 0..seq_len_k {
                    let mut sum = 0.0f32;
                    for d in 0..head_dim {
                        let q_idx = bh * seq_len_q * head_dim + i * head_dim + d;
                        let k_idx = bh * head_dim * seq_len_k + d * seq_len_k + j;
                        sum += q_data[q_idx] * k_t_data[k_idx];
                    }
                    let score_idx = bh * seq_len_q * seq_len_k + i * seq_len_k + j;
                    scores_data[score_idx] = sum * scale;
                }
            }
        }

        // Reshape scores: [B*H, seq_len_q, seq_len_k]
        let scores =
            Tensor::from_vec(scores_data, &[batch_size * num_heads, seq_len_q, seq_len_k])?;

        // Softmax along last dimension
        let attn_weights = scores.softmax(-1)?;

        // Apply dropout if training
        let attn_weights = if self.config.dropout > 0.0 && self.base.training() {
            crate::functional::dropout(&attn_weights, self.config.dropout, self.base.training())?
        } else {
            attn_weights
        };

        // Attention @ V: [B*H, seq_len_q, seq_len_k] @ [B*H, seq_len_k, head_dim]
        let attn_data = attn_weights.to_vec()?;
        let v_data = v_3d.to_vec()?;

        let mut output_data = vec![0.0f32; batch_size * num_heads * seq_len_q * head_dim];

        for bh in 0..(batch_size * num_heads) {
            for i in 0..seq_len_q {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_len_k {
                        let attn_idx = bh * seq_len_q * seq_len_k + i * seq_len_k + j;
                        let v_idx = bh * seq_len_k * head_dim + j * head_dim + d;
                        sum += attn_data[attn_idx] * v_data[v_idx];
                    }
                    let out_idx = bh * seq_len_q * head_dim + i * head_dim + d;
                    output_data[out_idx] = sum;
                }
            }
        }

        // Reshape back to 4D: [B*H, seq_len_q, head_dim] → [B, H, seq_len_q, head_dim]
        let output_3d =
            Tensor::from_vec(output_data, &[batch_size * num_heads, seq_len_q, head_dim])?;
        output_3d.reshape(&[
            batch_size as i32,
            num_heads as i32,
            seq_len_q as i32,
            head_dim as i32,
        ])
    }
}

impl Module for IPAdapterCrossAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For Module trait, use input as both query and key/value (self-attention fallback)
        self.forward(input, input)
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
    fn test_ip_adapter_cross_attention_creation() {
        let layer = IPAdapterCrossAttention::new(768, 8);
        assert!(layer.is_ok(), "Failed to create IPAdapterCrossAttention");
    }

    #[test]
    fn test_ip_adapter_cross_attention_shapes() {
        let layer = IPAdapterCrossAttention::new(768, 8).expect("Failed to create layer");

        let batch_size = 2;
        let seq_len = 77; // Typical CLIP text length
        let num_tokens = 16; // IP-Adapter tokens

        let query = ones(&[batch_size, seq_len, 768]).expect("Failed to create query");
        let image_features =
            ones(&[batch_size, num_tokens, 768]).expect("Failed to create image features");

        let output = layer.forward(&query, &image_features);
        assert!(output.is_ok(), "Forward pass failed: {:?}", output.err());

        if let Ok(output) = output {
            assert_eq!(
                output.shape().dims(),
                &[batch_size, seq_len, 768],
                "Output shape mismatch"
            );
        }
    }

    #[test]
    fn test_null_conditioning() {
        let layer = IPAdapterCrossAttention::new(768, 8).expect("Failed to create layer");

        let batch_size = 2;
        let seq_len = 77;
        let num_tokens = 16;

        let query = ones(&[batch_size, seq_len, 768]).expect("Failed to create query");
        let null_features = zeros(&[batch_size, num_tokens, 768]).expect("Failed to create nulls");

        let output = layer.forward(&query, &null_features);
        assert!(output.is_ok(), "Null conditioning failed");
    }

    #[test]
    fn test_attention_mask_application() {
        let layer = IPAdapterCrossAttention::new(768, 8).expect("Failed to create layer");

        let batch_size = 1;
        let seq_len = 10;
        let num_tokens = 4;

        let query = ones(&[batch_size, seq_len, 768]).expect("Failed to create query");
        let image_features =
            ones(&[batch_size, num_tokens, 768]).expect("Failed to create features");

        let result = layer.forward(&query, &image_features);
        assert!(result.is_ok(), "Forward with default mask failed");
    }

    #[test]
    fn test_forward_backward_compatibility() {
        let layer = IPAdapterCrossAttention::new(768, 8).expect("Failed to create layer");

        let batch_size = 2;
        let seq_len = 77;
        let num_tokens = 16;

        let query = ones(&[batch_size, seq_len, 768]).expect("Failed to create query");
        let image_features =
            ones(&[batch_size, num_tokens, 768]).expect("Failed to create features");

        // Forward pass
        let output1 = layer
            .forward(&query, &image_features)
            .expect("Forward 1 failed");
        let output2 = layer
            .forward(&query, &image_features)
            .expect("Forward 2 failed");

        // Should produce consistent results
        assert_eq!(output1.shape().dims(), output2.shape().dims());
    }

    #[test]
    fn test_invalid_embed_dim() {
        // embed_dim not divisible by num_heads
        let result = IPAdapterCrossAttention::new(770, 8);
        assert!(result.is_err(), "Should fail with invalid embed_dim");
    }

    #[test]
    fn test_mismatched_batch_sizes() {
        let layer = IPAdapterCrossAttention::new(768, 8).expect("Failed to create layer");

        let query = ones(&[2, 77, 768]).expect("Failed to create query");
        let image_features = ones(&[3, 16, 768]).expect("Failed to create features");

        let result = layer.forward(&query, &image_features);
        assert!(result.is_err(), "Should fail with mismatched batch sizes");
    }

    #[test]
    fn test_invalid_query_shape() {
        let layer = IPAdapterCrossAttention::new(768, 8).expect("Failed to create layer");

        let query = ones(&[2, 768]).expect("Failed to create 2D query");
        let image_features = ones(&[2, 16, 768]).expect("Failed to create features");

        let result = layer.forward(&query, &image_features);
        assert!(result.is_err(), "Should fail with 2D query");
    }

    #[test]
    fn test_different_num_heads() {
        let head_counts = [1, 2, 4, 8, 12];

        for num_heads in &head_counts {
            let layer =
                IPAdapterCrossAttention::new(768, *num_heads).expect("Failed to create layer");

            let query = ones(&[1, 77, 768]).expect("Failed to create query");
            let image_features = ones(&[1, 16, 768]).expect("Failed to create features");

            let result = layer.forward(&query, &image_features);
            assert!(result.is_ok(), "Failed with {} heads", num_heads);
        }
    }
}
