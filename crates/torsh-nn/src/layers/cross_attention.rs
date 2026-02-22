//! Cross-Attention Layer for Encoder-Decoder Architectures
//!
//! This module implements cross-attention mechanisms where queries come from one source
//! (typically the decoder) and keys/values come from another source (typically the encoder).
//! This is the key component in Transformer encoder-decoder architectures.
//!
//! # Architecture
//!
//! Cross-attention computes:
//! ```text
//! CrossAttention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
//! where:
//!   Q = query (from decoder)
//!   K, V = key, value (from encoder)
//! ```
//!
//! # Examples
//!
//! ## Basic Cross-Attention
//!
//! ```rust
//! use torsh_nn::layers::CrossAttention;
//! use torsh_tensor::creation;
//! use torsh_core::error::Result;
//!
//! # fn main() -> Result<()> {
//! // Create cross-attention layer
//! let cross_attn = CrossAttention::new(512, 512, 512, 8)?;
//!
//! // Decoder query: [batch_size=2, seq_len=10, query_dim=512]
//! let query = creation::randn(&[2, 10, 512])?;
//!
//! // Encoder key/value: [batch_size=2, seq_len=20, kv_dim=512]
//! let encoder_output = creation::randn(&[2, 20, 512])?;
//!
//! // Apply cross-attention
//! let output = cross_attn.forward_cross(&query, &encoder_output, &encoder_output, None)?;
//! assert_eq!(output.shape().dims(), &[2, 10, 512]);
//! # Ok(())
//! # }
//! ```
//!
//! ## With Attention Mask
//!
//! ```rust
//! use torsh_nn::layers::CrossAttention;
//! use torsh_tensor::creation;
//! use torsh_core::error::Result;
//!
//! # fn main() -> Result<()> {
//! let cross_attn = CrossAttention::new(256, 256, 256, 4)?;
//!
//! let query = creation::randn(&[1, 5, 256])?;
//! let key = creation::randn(&[1, 10, 256])?;
//! let value = creation::randn(&[1, 10, 256])?;
//!
//! // Create attention mask: [1, 5, 10] (1 = mask out, 0 = attend)
//! let mask = creation::zeros(&[1, 5, 10])?;
//!
//! let output = cross_attn.forward_cross(&query, &key, &value, Some(&mask))?;
//! assert_eq!(output.shape().dims(), &[1, 5, 256]);
//! # Ok(())
//! # }
//! ```
//!
//! # PyTorch Compatibility
//!
//! This implementation is designed to be compatible with PyTorch's MultiheadAttention
//! when used in cross-attention mode:
//!
//! ```python
//! # PyTorch equivalent
//! cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
//! output, _ = cross_attn(query, key, value, attn_mask=mask)
//! ```
//!
//! # Use in Transformer Decoder
//!
//! Cross-attention is typically the second attention layer in a Transformer decoder block:
//!
//! ```text
//! Decoder Block:
//! 1. Self-Attention (decoder attends to itself)
//! 2. Cross-Attention (decoder attends to encoder) <- This module
//! 3. Feed-Forward Network
//! ```

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::{Result, TorshError};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use torsh_tensor::{creation::*, Tensor};

/// Cross-Attention Layer for Encoder-Decoder Architectures
///
/// Implements multi-head cross-attention where queries come from one source
/// (typically decoder) and keys/values come from another source (typically encoder).
///
/// # Architecture Details
///
/// - Separate projections for Q, K, V with different input dimensions
/// - Multi-head attention with configurable number of heads
/// - Optional dropout for regularization
/// - Optional bias terms for all projections
/// - Configurable attention masking
///
/// # Parameters
///
/// - `q_proj`: Linear projection for queries [query_dim -> embed_dim]
/// - `k_proj`: Linear projection for keys [kv_dim -> embed_dim]
/// - `v_proj`: Linear projection for values [kv_dim -> embed_dim]
/// - `out_proj`: Linear projection for output [embed_dim -> embed_dim]
///
/// # Shape Notation
///
/// - B: batch size
/// - Lq: query sequence length
/// - Lk: key/value sequence length
/// - D: embed_dim
/// - H: num_heads
/// - d: head_dim (D / H)
pub struct CrossAttention {
    base: ModuleBase,
    query_dim: usize,
    kv_dim: usize,
    embed_dim: usize,
    num_heads: usize,
    dropout: f32,
    bias: bool,
}

impl CrossAttention {
    /// Create a new cross-attention layer
    ///
    /// # Arguments
    ///
    /// - `query_dim`: Dimension of query input (from decoder)
    /// - `kv_dim`: Dimension of key/value input (from encoder)
    /// - `embed_dim`: Dimension of attention embeddings (must be divisible by num_heads)
    /// - `num_heads`: Number of parallel attention heads
    ///
    /// # Returns
    ///
    /// A new `CrossAttention` layer with Xavier-initialized projection matrices
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `embed_dim` is not divisible by `num_heads`
    /// - Weight initialization fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_nn::layers::CrossAttention;
    /// use torsh_core::error::Result;
    ///
    /// # fn main() -> Result<()> {
    /// // Standard configuration: same dimensions
    /// let attn1 = CrossAttention::new(512, 512, 512, 8)?;
    ///
    /// // Different dimensions: decoder=256, encoder=512
    /// let attn2 = CrossAttention::new(256, 512, 512, 8)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        query_dim: usize,
        kv_dim: usize,
        embed_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(TorshError::InvalidArgument(format!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                embed_dim, num_heads
            )));
        }

        let mut base = ModuleBase::new();

        // Initialize projection weights
        // Q projection: [query_dim -> embed_dim]
        let q_proj_weight = crate::init::xavier_uniform(&[query_dim, embed_dim])?;
        base.register_parameter("q_proj_weight".to_string(), Parameter::new(q_proj_weight));

        // K projection: [kv_dim -> embed_dim]
        let k_proj_weight = crate::init::xavier_uniform(&[kv_dim, embed_dim])?;
        base.register_parameter("k_proj_weight".to_string(), Parameter::new(k_proj_weight));

        // V projection: [kv_dim -> embed_dim]
        let v_proj_weight = crate::init::xavier_uniform(&[kv_dim, embed_dim])?;
        base.register_parameter("v_proj_weight".to_string(), Parameter::new(v_proj_weight));

        // Output projection: [embed_dim -> embed_dim]
        let out_proj_weight = crate::init::xavier_uniform(&[embed_dim, embed_dim])?;
        base.register_parameter(
            "out_proj_weight".to_string(),
            Parameter::new(out_proj_weight),
        );

        // Initialize biases (all zeros)
        let q_proj_bias = zeros(&[embed_dim])?;
        base.register_parameter("q_proj_bias".to_string(), Parameter::new(q_proj_bias));

        let k_proj_bias = zeros(&[embed_dim])?;
        base.register_parameter("k_proj_bias".to_string(), Parameter::new(k_proj_bias));

        let v_proj_bias = zeros(&[embed_dim])?;
        base.register_parameter("v_proj_bias".to_string(), Parameter::new(v_proj_bias));

        let out_proj_bias = zeros(&[embed_dim])?;
        base.register_parameter("out_proj_bias".to_string(), Parameter::new(out_proj_bias));

        Ok(Self {
            base,
            query_dim,
            kv_dim,
            embed_dim,
            num_heads,
            dropout: 0.0,
            bias: true,
        })
    }

    /// Create cross-attention with custom configuration
    ///
    /// # Arguments
    ///
    /// - `query_dim`: Query dimension
    /// - `kv_dim`: Key/Value dimension
    /// - `embed_dim`: Embedding dimension
    /// - `num_heads`: Number of heads
    /// - `dropout`: Dropout probability (0.0 to 1.0)
    /// - `bias`: Whether to use bias in projections
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_nn::layers::CrossAttention;
    /// use torsh_core::error::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let attn = CrossAttention::with_config(512, 512, 512, 8, 0.1, true)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_config(
        query_dim: usize,
        kv_dim: usize,
        embed_dim: usize,
        num_heads: usize,
        dropout: f32,
        bias: bool,
    ) -> Result<Self> {
        let mut layer = Self::new(query_dim, kv_dim, embed_dim, num_heads)?;
        layer.dropout = dropout;
        layer.bias = bias;
        Ok(layer)
    }

    /// Set dropout probability
    ///
    /// # Arguments
    ///
    /// - `dropout`: Dropout probability (0.0 to 1.0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_nn::layers::CrossAttention;
    /// use torsh_core::error::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let mut attn = CrossAttention::new(512, 512, 512, 8)?;
    /// attn.set_dropout(0.1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_dropout(&mut self, dropout: f32) {
        self.dropout = dropout;
    }

    /// Get the number of attention heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get the embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Forward pass with separate query, key, and value inputs
    ///
    /// # Arguments
    ///
    /// - `query`: Query tensor [B, Lq, query_dim]
    /// - `key`: Key tensor [B, Lk, kv_dim]
    /// - `value`: Value tensor [B, Lk, kv_dim]
    /// - `attn_mask`: Optional attention mask [B, Lq, Lk] or broadcastable
    ///
    /// # Returns
    ///
    /// Output tensor [B, Lq, embed_dim]
    ///
    /// # Shape Requirements
    ///
    /// - query: [batch_size, query_len, query_dim]
    /// - key: [batch_size, key_len, kv_dim]
    /// - value: [batch_size, key_len, kv_dim]
    /// - mask (optional): [batch_size, query_len, key_len] or broadcastable
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input shapes are incompatible
    /// - Tensor operations fail
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_nn::layers::CrossAttention;
    /// use torsh_tensor::creation;
    /// use torsh_core::error::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let attn = CrossAttention::new(512, 512, 512, 8)?;
    ///
    /// let query = creation::randn(&[2, 10, 512])?;
    /// let key = creation::randn(&[2, 20, 512])?;
    /// let value = creation::randn(&[2, 20, 512])?;
    ///
    /// let output = attn.forward_cross(&query, &key, &value, None)?;
    /// assert_eq!(output.shape().dims(), &[2, 10, 512]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn forward_cross(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let query_shape_ref = query.shape();
        let query_shape = query_shape_ref.dims();
        let key_shape_ref = key.shape();
        let key_shape = key_shape_ref.dims();

        // Validate input shapes
        if query_shape.len() != 3 {
            return Err(TorshError::InvalidShape(format!(
                "Query must be 3D [batch, seq_len, dim], got shape {:?}",
                query_shape
            )));
        }
        if key_shape.len() != 3 || value.shape().dims().len() != 3 {
            return Err(TorshError::InvalidShape(
                "Key and value must be 3D tensors".to_string(),
            ));
        }

        let batch_size = query_shape[0];
        let query_len = query_shape[1];
        let key_len = key_shape[1];
        let head_dim = self.embed_dim / self.num_heads;

        // Get projection weights
        let q_proj_weight = self.base.parameters["q_proj_weight"]
            .tensor()
            .read()
            .clone();
        let k_proj_weight = self.base.parameters["k_proj_weight"]
            .tensor()
            .read()
            .clone();
        let v_proj_weight = self.base.parameters["v_proj_weight"]
            .tensor()
            .read()
            .clone();
        let out_proj_weight = self.base.parameters["out_proj_weight"]
            .tensor()
            .read()
            .clone();

        // Project queries: [B, Lq, query_dim] -> [B, Lq, embed_dim]
        // Reshape to 2D for matmul: [B*Lq, query_dim] @ [query_dim, embed_dim] -> [B*Lq, embed_dim]
        let q_reshaped =
            query.reshape(&[(batch_size * query_len) as i32, query_shape[2] as i32])?;
        let mut q_projected = q_reshaped.matmul(&q_proj_weight)?;
        if self.bias {
            let q_proj_bias = self.base.parameters["q_proj_bias"].tensor().read().clone();
            q_projected = q_projected.add_op(&q_proj_bias)?;
        }
        // Reshape back to 3D: [B, Lq, embed_dim]
        let q =
            q_projected.reshape(&[batch_size as i32, query_len as i32, self.embed_dim as i32])?;

        // Project keys: [B, Lk, kv_dim] -> [B, Lk, embed_dim]
        // Reshape to 2D for matmul: [B*Lk, kv_dim] @ [kv_dim, embed_dim] -> [B*Lk, embed_dim]
        let k_reshaped = key.reshape(&[(batch_size * key_len) as i32, key_shape[2] as i32])?;
        let mut k_projected = k_reshaped.matmul(&k_proj_weight)?;
        if self.bias {
            let k_proj_bias = self.base.parameters["k_proj_bias"].tensor().read().clone();
            k_projected = k_projected.add_op(&k_proj_bias)?;
        }
        // Reshape back to 3D: [B, Lk, embed_dim]
        let k = k_projected.reshape(&[batch_size as i32, key_len as i32, self.embed_dim as i32])?;

        // Project values: [B, Lk, kv_dim] -> [B, Lk, embed_dim]
        // Reshape to 2D for matmul: [B*Lk, kv_dim] @ [kv_dim, embed_dim] -> [B*Lk, embed_dim]
        let v_reshaped = value.reshape(&[
            (batch_size * key_len) as i32,
            value.shape().dims()[2] as i32,
        ])?;
        let mut v_projected = v_reshaped.matmul(&v_proj_weight)?;
        if self.bias {
            let v_proj_bias = self.base.parameters["v_proj_bias"].tensor().read().clone();
            v_projected = v_projected.add_op(&v_proj_bias)?;
        }
        // Reshape back to 3D: [B, Lk, embed_dim]
        let v = v_projected.reshape(&[batch_size as i32, key_len as i32, self.embed_dim as i32])?;

        // Reshape for multi-head attention
        // [B, L, embed_dim] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        let q = q
            .reshape(&[
                batch_size as i32,
                query_len as i32,
                self.num_heads as i32,
                head_dim as i32,
            ])?
            .transpose(1, 2)?;

        let k = k
            .reshape(&[
                batch_size as i32,
                key_len as i32,
                self.num_heads as i32,
                head_dim as i32,
            ])?
            .transpose(1, 2)?;

        let v = v
            .reshape(&[
                batch_size as i32,
                key_len as i32,
                self.num_heads as i32,
                head_dim as i32,
            ])?
            .transpose(1, 2)?;

        // Compute scaled dot-product attention
        let attended = self.scaled_dot_product_attention(&q, &k, &v, attn_mask)?;

        // Reshape back: [B, num_heads, Lq, head_dim] -> [B, Lq, embed_dim]
        let attended = attended.transpose(1, 2)?.contiguous()?.reshape(&[
            batch_size as i32,
            query_len as i32,
            self.embed_dim as i32,
        ])?;

        // Apply output projection: [B, Lq, embed_dim] -> [B, Lq, embed_dim]
        // Reshape to 2D for matmul: [B*Lq, embed_dim] @ [embed_dim, embed_dim] -> [B*Lq, embed_dim]
        let attended_reshaped =
            attended.reshape(&[(batch_size * query_len) as i32, self.embed_dim as i32])?;
        let mut output = attended_reshaped.matmul(&out_proj_weight)?;
        if self.bias {
            let out_proj_bias = self.base.parameters["out_proj_bias"]
                .tensor()
                .read()
                .clone();
            output = output.add_op(&out_proj_bias)?;
        }
        // Reshape back to 3D: [B, Lq, embed_dim]
        let output =
            output.reshape(&[batch_size as i32, query_len as i32, self.embed_dim as i32])?;

        Ok(output)
    }

    /// Scaled dot-product attention implementation
    ///
    /// Computes: softmax(Q * K^T / sqrt(d_k)) * V
    ///
    /// # Arguments
    ///
    /// - `query`: [B, num_heads, Lq, head_dim]
    /// - `key`: [B, num_heads, Lk, head_dim]
    /// - `value`: [B, num_heads, Lk, head_dim]
    /// - `attn_mask`: Optional mask [B, Lq, Lk] or broadcastable
    ///
    /// # Returns
    ///
    /// Attention output [B, num_heads, Lq, head_dim]
    fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q_shape_ref = query.shape();
        let q_shape = q_shape_ref.dims();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let query_len = q_shape[2];
        let head_dim = q_shape[3];

        let k_shape_ref = key.shape();
        let k_shape = k_shape_ref.dims();
        let key_len = k_shape[2];

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Compute Q @ K^T / sqrt(d_k)
        // [B, H, Lq, d] @ [B, H, d, Lk] -> [B, H, Lq, Lk]
        // Reshape to 2D by merging all leading dimensions: [B*H*Lq, d] @ [d, B*H*Lk] is wrong
        // Better: loop over B*H and do [Lq, d] @ [d, Lk] -> [Lq, Lk]

        let key_transposed = key.transpose(-2, -1)?;

        // Get raw data for manual batched matmul
        let q_data = query.data()?;
        let k_t_data = key_transposed.data()?;

        // Perform batched matrix multiplication manually
        let mut scores_data = Vec::with_capacity(batch_size * num_heads * query_len * key_len);

        for b in 0..batch_size {
            for h in 0..num_heads {
                // Extract Q slice: [Lq, d]
                let q_offset = ((b * num_heads + h) * query_len * head_dim) as usize;

                // Extract K^T slice: [d, Lk]
                let k_offset = ((b * num_heads + h) * head_dim * key_len) as usize;

                // Compute Q @ K^T for this head: [Lq, d] @ [d, Lk] -> [Lq, Lk]
                for i in 0..query_len {
                    for j in 0..key_len {
                        let mut sum = 0.0f32;
                        for k in 0..head_dim {
                            let q_val = q_data[q_offset + i * head_dim + k];
                            let k_val = k_t_data[k_offset + k * key_len + j];
                            sum += q_val * k_val;
                        }
                        scores_data.push(sum);
                    }
                }
            }
        }

        // Reshape back to 4D: [B, H, Lq, Lk]
        let mut scores = Tensor::from_data(
            scores_data,
            vec![batch_size, num_heads, query_len, key_len],
            query.device(),
        )?
        .mul_scalar(scale)?;

        // Apply attention mask if provided
        if let Some(mask) = attn_mask {
            // Mask should be [B, Lq, Lk] or broadcastable
            // Add very large negative value where mask is 1
            let large_neg = mask.mul_scalar(-1e9)?;
            scores = scores.add_op(&large_neg)?;
        }

        // Apply softmax along last dimension
        let attn_weights = scores.softmax(-1)?;

        // Apply dropout if training and dropout > 0
        let attn_weights = if self.dropout > 0.0 && self.training() {
            crate::functional::dropout(&attn_weights, self.dropout, true)?
        } else {
            attn_weights
        };

        // Compute final output: [B, H, Lq, Lk] @ [B, H, Lk, d] -> [B, H, Lq, d]
        // Similar batched matmul approach
        let attn_data = attn_weights.data()?;
        let v_data = value.data()?;

        let mut output_data = Vec::with_capacity(batch_size * num_heads * query_len * head_dim);

        for b in 0..batch_size {
            for h in 0..num_heads {
                // Extract attention weights: [Lq, Lk]
                let a_offset = ((b * num_heads + h) * query_len * key_len) as usize;

                // Extract V slice: [Lk, d]
                let v_offset = ((b * num_heads + h) * key_len * head_dim) as usize;

                // Compute A @ V for this head: [Lq, Lk] @ [Lk, d] -> [Lq, d]
                for i in 0..query_len {
                    for j in 0..head_dim {
                        let mut sum = 0.0f32;
                        for k in 0..key_len {
                            let a_val = attn_data[a_offset + i * key_len + k];
                            let v_val = v_data[v_offset + k * head_dim + j];
                            sum += a_val * v_val;
                        }
                        output_data.push(sum);
                    }
                }
            }
        }

        // Reshape to 4D: [B, H, Lq, d]
        Tensor::from_data(
            output_data,
            vec![batch_size, num_heads, query_len, head_dim],
            query.device(),
        )
    }
}

impl Module for CrossAttention {
    /// Standard forward pass (expects concatenated input for Module trait compatibility)
    ///
    /// For cross-attention, use `forward_cross()` instead which provides the proper interface.
    /// This implementation expects input to be the query and uses it as key/value as well
    /// (equivalent to self-attention), primarily for trait compliance.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // For Module trait compliance, treat as self-attention
        self.forward_cross(input, input, input, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
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

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl core::fmt::Debug for CrossAttention {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CrossAttention")
            .field("query_dim", &self.query_dim)
            .field("kv_dim", &self.kv_dim)
            .field("embed_dim", &self.embed_dim)
            .field("num_heads", &self.num_heads)
            .field("dropout", &self.dropout)
            .field("bias", &self.bias)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_cross_attention_creation() -> Result<()> {
        let attn = CrossAttention::new(512, 512, 512, 8)?;
        assert_eq!(attn.num_heads(), 8);
        assert_eq!(attn.embed_dim(), 512);
        Ok(())
    }

    #[test]
    fn test_cross_attention_creation_invalid_heads() {
        let result = CrossAttention::new(512, 512, 513, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_attention_forward() -> Result<()> {
        let attn = CrossAttention::new(256, 256, 256, 4)?;

        let query = creation::randn(&[2, 10, 256])?;
        let key = creation::randn(&[2, 20, 256])?;
        let value = creation::randn(&[2, 20, 256])?;

        let output = attn.forward_cross(&query, &key, &value, None)?;

        assert_eq!(output.shape().dims(), &[2, 10, 256]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_different_dims() -> Result<()> {
        // Decoder dim = 256, Encoder dim = 512, Embed dim = 512
        let attn = CrossAttention::new(256, 512, 512, 8)?;

        let query = creation::randn(&[1, 5, 256])?;
        let key = creation::randn(&[1, 10, 512])?;
        let value = creation::randn(&[1, 10, 512])?;

        let output = attn.forward_cross(&query, &key, &value, None)?;

        assert_eq!(output.shape().dims(), &[1, 5, 512]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_with_mask() -> Result<()> {
        let attn = CrossAttention::new(128, 128, 128, 4)?;

        let query = creation::randn(&[1, 5, 128])?;
        let key = creation::randn(&[1, 10, 128])?;
        let value = creation::randn(&[1, 10, 128])?;

        // Create attention mask: [1, 5, 10]
        let mask = creation::zeros(&[1, 5, 10])?;

        let output = attn.forward_cross(&query, &key, &value, Some(&mask))?;

        assert_eq!(output.shape().dims(), &[1, 5, 128]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_batch_processing() -> Result<()> {
        let attn = CrossAttention::new(512, 512, 512, 8)?;

        let batch_size = 16;
        let query = creation::randn(&[batch_size, 15, 512])?;
        let key = creation::randn(&[batch_size, 30, 512])?;
        let value = creation::randn(&[batch_size, 30, 512])?;

        let output = attn.forward_cross(&query, &key, &value, None)?;

        assert_eq!(output.shape().dims(), &[batch_size, 15, 512]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_training_mode() -> Result<()> {
        let mut attn = CrossAttention::new(256, 256, 256, 4)?;

        assert!(attn.training());

        attn.eval();
        assert!(!attn.training());

        attn.train();
        assert!(attn.training());

        Ok(())
    }

    #[test]
    fn test_cross_attention_with_dropout() -> Result<()> {
        let mut attn = CrossAttention::with_config(256, 256, 256, 4, 0.1, true)?;
        attn.train();

        let query = creation::randn(&[2, 5, 256])?;
        let key = creation::randn(&[2, 10, 256])?;
        let value = creation::randn(&[2, 10, 256])?;

        let output = attn.forward_cross(&query, &key, &value, None)?;

        assert_eq!(output.shape().dims(), &[2, 5, 256]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_parameters() -> Result<()> {
        let attn = CrossAttention::new(512, 512, 512, 8)?;

        let params = attn.parameters();

        // Should have 8 parameters: q_proj, k_proj, v_proj, out_proj (weights + biases)
        assert_eq!(params.len(), 8);
        assert!(params.contains_key("q_proj_weight"));
        assert!(params.contains_key("k_proj_weight"));
        assert!(params.contains_key("v_proj_weight"));
        assert!(params.contains_key("out_proj_weight"));
        assert!(params.contains_key("q_proj_bias"));
        assert!(params.contains_key("k_proj_bias"));
        assert!(params.contains_key("v_proj_bias"));
        assert!(params.contains_key("out_proj_bias"));

        Ok(())
    }

    #[test]
    fn test_cross_attention_module_trait() -> Result<()> {
        let attn = CrossAttention::new(256, 256, 256, 4)?;

        // Test Module trait forward (uses self-attention fallback)
        let input = creation::randn(&[2, 10, 256])?;
        let output = attn.forward(&input)?;

        assert_eq!(output.shape().dims(), &[2, 10, 256]);
        Ok(())
    }

    #[test]
    fn test_cross_attention_encoder_decoder_use_case() -> Result<()> {
        // Simulate encoder-decoder architecture
        let embed_dim = 512;
        let num_heads = 8;

        let attn = CrossAttention::new(embed_dim, embed_dim, embed_dim, num_heads)?;

        // Encoder output: [batch=4, encoder_len=50, dim=512]
        let encoder_output = creation::randn(&[4, 50, 512])?;

        // Decoder query: [batch=4, decoder_len=20, dim=512]
        let decoder_query = creation::randn(&[4, 20, 512])?;

        // Cross-attention: decoder attends to encoder
        let output = attn.forward_cross(&decoder_query, &encoder_output, &encoder_output, None)?;

        // Output should match decoder sequence length
        assert_eq!(output.shape().dims(), &[4, 20, 512]);

        Ok(())
    }
}
