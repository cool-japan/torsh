//! Attention mechanisms for transformer models
//!
//! This module provides PyTorch-compatible attention mechanisms for transformer-based
//! neural networks, powered by scirs2-linalg's optimized implementations.
//!
//! ## Features
//!
//! - **Scaled Dot-Product Attention**: Core attention mechanism
//! - **Multi-Head Attention**: Parallel attention heads for richer representations
//! - **Causal Attention**: For autoregressive models
//! - **Flash Attention**: Memory-efficient attention for long sequences
//! - **Cross Attention**: For encoder-decoder architectures
//!
//! ## Examples
//!
//! ```ignore
//! use torsh_linalg::attention::scaled_dot_product_attention;
//! use torsh_tensor::Tensor;
//!
//! let batch_size = 2;
//! let seq_len = 10;
//! let d_model = 64;
//!
//! // Create query, key, value tensors
//! let query = Tensor::randn(&[batch_size, seq_len, d_model])?;
//! let key = Tensor::randn(&[batch_size, seq_len, d_model])?;
//! let value = Tensor::randn(&[batch_size, seq_len, d_model])?;
//!
//! // Compute attention
//! let output = scaled_dot_product_attention(&query, &key, &value, None, None)?;
//! ```

use torsh_core::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(feature = "scirs2-integration")]
use scirs2_core::ndarray::Array3;

/// Configuration for attention mechanisms
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Dropout probability (0.0 to 1.0)
    pub dropout_prob: f32,
    /// Whether to use causal masking
    pub causal: bool,
    /// Scaling factor (default: 1/sqrt(head_dim))
    pub scale: Option<f32>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout_prob: 0.0,
            causal: false,
            scale: None,
        }
    }
}

/// Attention mask types
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// Causal mask for autoregressive models
    Causal,
    /// Custom boolean mask (true = attend, false = mask)
    Custom(Tensor),
    /// Padding mask (true = valid token, false = padding)
    Padding(Tensor),
}

/// Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch, seq_len_v, d_model]
/// * `mask` - Optional attention mask
/// * `scale` - Optional scaling factor (default: 1/sqrt(d_model))
///
/// # Returns
///
/// Output tensor of shape [batch, seq_len_q, d_model]
#[cfg(feature = "scirs2-integration")]
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&AttentionMask>,
    scale: Option<f32>,
) -> Result<Tensor> {
    // Validate dimensions
    if query.shape().ndim() != 3 || key.shape().ndim() != 3 || value.shape().ndim() != 3 {
        return Err(TorshError::InvalidArgument(
            "Attention requires 3D tensors [batch, seq_len, d_model]".to_string(),
        ));
    }

    let query_shape_binding = query.shape();
    let query_shape = query_shape_binding.dims();
    let key_shape_binding = key.shape();
    let key_shape = key_shape_binding.dims();
    let value_shape_binding = value.shape();
    let value_shape = value_shape_binding.dims();

    let (batch_q, _seq_len_q, d_model) = (query_shape[0], query_shape[1], query_shape[2]);
    let (batch_k, seq_len_k, d_k) = (key_shape[0], key_shape[1], key_shape[2]);
    let (batch_v, seq_len_v, d_v) = (value_shape[0], value_shape[1], value_shape[2]);

    // Validate batch sizes match
    if batch_q != batch_k || batch_k != batch_v {
        return Err(TorshError::InvalidArgument(
            "Batch sizes must match for attention".to_string(),
        ));
    }

    // Validate key and value sequence lengths match
    if seq_len_k != seq_len_v {
        return Err(TorshError::InvalidArgument(
            "Key and value sequence lengths must match".to_string(),
        ));
    }

    // Validate dimensions match
    if d_model != d_k || d_k != d_v {
        return Err(TorshError::InvalidArgument(
            "Model dimensions must match for attention".to_string(),
        ));
    }

    // Convert tensors to ndarray format for scirs2-linalg
    let query_data = tensor_to_array3(query)?;
    let key_data = tensor_to_array3(key)?;
    let value_data = tensor_to_array3(value)?;

    // Compute scaling factor
    let scale_factor = scale.unwrap_or(1.0 / (d_model as f32).sqrt());

    // Apply scirs2-linalg's scaled dot-product attention
    let attention_mask = match mask {
        Some(AttentionMask::Causal) => Some(scirs2_linalg::attention::AttentionMask::Causal),
        Some(AttentionMask::Custom(_)) | Some(AttentionMask::Padding(_)) => {
            // For now, custom masks require more complex handling
            None
        }
        None => None,
    };

    let output = scirs2_linalg::attention::scaled_dot_product_attention(
        &query_data.view(),
        &key_data.view(),
        &value_data.view(),
        attention_mask.as_ref(),
        scale_factor,
    )
    .map_err(|e| TorshError::ComputeError(format!("Attention computation failed: {e}")))?;

    // Convert back to tensor
    array3_to_tensor(&output, query.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn scaled_dot_product_attention(
    _query: &Tensor,
    _key: &Tensor,
    _value: &Tensor,
    _mask: Option<&AttentionMask>,
    _scale: Option<f32>,
) -> Result<Tensor> {
    Err(TorshError::NotImplemented(
        "Attention requires scirs2-integration feature".to_string(),
    ))
}

/// Multi-head attention with linear projections
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch, seq_len_q, d_model]
/// * `key` - Key tensor of shape [batch, seq_len_k, d_model]
/// * `value` - Value tensor of shape [batch, seq_len_v, d_model]
/// * `wq` - Query projection weights [d_model, d_model]
/// * `wk` - Key projection weights [d_model, d_model]
/// * `wv` - Value projection weights [d_model, d_model]
/// * `wo` - Output projection weights [d_model, d_model]
/// * `mask` - Optional attention mask
/// * `config` - Attention configuration
///
/// # Returns
///
/// Output tensor of shape [batch, seq_len_q, d_model]
#[cfg(feature = "scirs2-integration")]
pub fn multi_head_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    wq: &Tensor,
    wk: &Tensor,
    wv: &Tensor,
    wo: &Tensor,
    mask: Option<&AttentionMask>,
    config: &AttentionConfig,
) -> Result<Tensor> {
    // Validate dimensions
    if query.shape().ndim() != 3 {
        return Err(TorshError::InvalidArgument(
            "Query must be 3D tensor [batch, seq_len, d_model]".to_string(),
        ));
    }

    // Convert to ndarray format
    let query_data = tensor_to_array3(query)?;
    let key_data = tensor_to_array3(key)?;
    let value_data = tensor_to_array3(value)?;
    let wq_data = tensor_to_array2(wq)?;
    let wk_data = tensor_to_array2(wk)?;
    let wv_data = tensor_to_array2(wv)?;
    let wo_data = tensor_to_array2(wo)?;

    // Create scirs2 attention config
    let scirs2_config = scirs2_linalg::attention::AttentionConfig {
        num_heads: config.num_heads,
        head_dim: config.head_dim,
        dropout_prob: config.dropout_prob,
        causal: config.causal,
        scale: config.scale,
    };

    let attention_mask = match mask {
        Some(AttentionMask::Causal) => Some(scirs2_linalg::attention::AttentionMask::Causal),
        _ => None,
    };

    // Apply multi-head attention
    let output = scirs2_linalg::attention::multi_head_attention(
        &query_data.view(),
        &key_data.view(),
        &value_data.view(),
        &wq_data.view(),
        &wk_data.view(),
        &wv_data.view(),
        &wo_data.view(),
        attention_mask.as_ref(),
        &scirs2_config,
    )
    .map_err(|e| TorshError::ComputeError(format!("Multi-head attention failed: {e}")))?;

    // Convert back to tensor
    array3_to_tensor(&output, query.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn multi_head_attention(
    _query: &Tensor,
    _key: &Tensor,
    _value: &Tensor,
    _wq: &Tensor,
    _wk: &Tensor,
    _wv: &Tensor,
    _wo: &Tensor,
    _mask: Option<&AttentionMask>,
    _config: &AttentionConfig,
) -> Result<Tensor> {
    Err(TorshError::NotImplemented(
        "Multi-head attention requires scirs2-integration feature".to_string(),
    ))
}

/// Causal attention for autoregressive models
///
/// Applies scaled dot-product attention with a causal mask, ensuring that
/// each position can only attend to previous positions.
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch, seq_len, d_model]
/// * `key` - Key tensor of shape [batch, seq_len, d_model]
/// * `value` - Value tensor of shape [batch, seq_len, d_model]
/// * `scale` - Optional scaling factor (default: 1/sqrt(d_model))
///
/// # Returns
///
/// Output tensor of shape [batch, seq_len, d_model]
pub fn causal_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: Option<f32>,
) -> Result<Tensor> {
    scaled_dot_product_attention(query, key, value, Some(&AttentionMask::Causal), scale)
}

/// Flash attention for memory-efficient attention computation
///
/// Uses the Flash Attention algorithm to reduce memory usage and improve
/// performance for long sequences.
///
/// # Arguments
///
/// * `query` - Query tensor of shape [batch, seq_len, d_model]
/// * `key` - Key tensor of shape [batch, seq_len, d_model]
/// * `value` - Value tensor of shape [batch, seq_len, d_model]
/// * `scale` - Optional scaling factor
///
/// # Returns
///
/// Output tensor of shape [batch, seq_len, d_model]
#[cfg(feature = "scirs2-integration")]
pub fn flash_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: Option<f32>,
) -> Result<Tensor> {
    // Convert to ndarray format
    let query_data = tensor_to_array3(query)?;
    let key_data = tensor_to_array3(key)?;
    let value_data = tensor_to_array3(value)?;

    let scale_factor = scale.unwrap_or_else(|| {
        let d_model = query.shape().dims()[2];
        1.0 / (d_model as f32).sqrt()
    });

    // Apply flash attention (with block size parameter)
    let block_size = 64; // Default block size for flash attention
    let output = scirs2_linalg::attention::flash_attention(
        &query_data.view(),
        &key_data.view(),
        &value_data.view(),
        None, // No mask
        scale_factor,
        block_size,
    )
    .map_err(|e| TorshError::ComputeError(format!("Flash attention failed: {e}")))?;

    array3_to_tensor(&output, query.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn flash_attention(
    _query: &Tensor,
    _key: &Tensor,
    _value: &Tensor,
    _scale: Option<f32>,
) -> Result<Tensor> {
    Err(TorshError::NotImplemented(
        "Flash attention requires scirs2-integration feature".to_string(),
    ))
}

// Helper functions for tensor <-> ndarray conversions

#[cfg(feature = "scirs2-integration")]
fn tensor_to_array3(tensor: &Tensor) -> Result<Array3<f32>> {
    if tensor.shape().ndim() != 3 {
        return Err(TorshError::InvalidArgument(
            "Expected 3D tensor".to_string(),
        ));
    }

    let shape_binding = tensor.shape();
    let shape = shape_binding.dims();
    let (d0, d1, d2) = (shape[0], shape[1], shape[2]);

    let mut data = Vec::with_capacity(d0 * d1 * d2);
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                data.push(tensor.get(&[i, j, k])?);
            }
        }
    }

    Array3::from_shape_vec((d0, d1, d2), data)
        .map_err(|e| TorshError::ComputeError(format!("Failed to create Array3: {e}")))
}

#[cfg(feature = "scirs2-integration")]
fn tensor_to_array2(tensor: &Tensor) -> Result<scirs2_core::ndarray::Array2<f32>> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Expected 2D tensor".to_string(),
        ));
    }

    let shape_binding = tensor.shape();
    let shape = shape_binding.dims();
    let (rows, cols) = (shape[0], shape[1]);

    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(tensor.get(&[i, j])?);
        }
    }

    scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| TorshError::ComputeError(format!("Failed to create Array2: {e}")))
}

#[cfg(feature = "scirs2-integration")]
fn array3_to_tensor(array: &Array3<f32>, device: torsh_core::DeviceType) -> Result<Tensor> {
    let shape = array.shape();
    let dims = vec![shape[0], shape[1], shape[2]];

    let mut data = Vec::with_capacity(shape[0] * shape[1] * shape[2]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                data.push(array[[i, j, k]]);
            }
        }
    }

    Tensor::from_data(data, dims, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_relative_eq!(config.dropout_prob, 0.0);
        assert!(!config.causal);
        assert!(config.scale.is_none());
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_scaled_dot_product_attention_basic() -> Result<()> {
        let batch_size = 2;
        let seq_len = 4;
        let d_model = 8;

        // Create simple tensors
        let query_data = vec![1.0f32; batch_size * seq_len * d_model];
        let key_data = vec![1.0f32; batch_size * seq_len * d_model];
        let value_data = vec![2.0f32; batch_size * seq_len * d_model];

        let query = Tensor::from_data(
            query_data,
            vec![batch_size, seq_len, d_model],
            torsh_core::DeviceType::Cpu,
        )?;
        let key = Tensor::from_data(
            key_data,
            vec![batch_size, seq_len, d_model],
            torsh_core::DeviceType::Cpu,
        )?;
        let value = Tensor::from_data(
            value_data,
            vec![batch_size, seq_len, d_model],
            torsh_core::DeviceType::Cpu,
        )?;

        let output = scaled_dot_product_attention(&query, &key, &value, None, None)?;

        // Check output shape
        assert_eq!(output.shape().dims(), &[batch_size, seq_len, d_model]);

        // Output should be close to value since all queries/keys are identical
        for i in 0..batch_size {
            for j in 0..seq_len {
                for k in 0..d_model {
                    let out_val = output.get(&[i, j, k])?;
                    assert!((out_val - 2.0).abs() < 0.1);
                }
            }
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_causal_attention() -> Result<()> {
        let batch_size = 1;
        let seq_len = 3;
        let d_model = 4;

        // Create simple test tensors
        let query_data = vec![1.0f32; batch_size * seq_len * d_model];
        let key_data = vec![1.0f32; batch_size * seq_len * d_model];
        let value_data = vec![1.0f32; batch_size * seq_len * d_model];

        let query = Tensor::from_data(
            query_data,
            vec![batch_size, seq_len, d_model],
            torsh_core::DeviceType::Cpu,
        )?;
        let key = Tensor::from_data(
            key_data,
            vec![batch_size, seq_len, d_model],
            torsh_core::DeviceType::Cpu,
        )?;
        let value = Tensor::from_data(
            value_data,
            vec![batch_size, seq_len, d_model],
            torsh_core::DeviceType::Cpu,
        )?;

        let output = causal_attention(&query, &key, &value, None)?;

        // Check output shape
        assert_eq!(output.shape().dims(), &[batch_size, seq_len, d_model]);

        Ok(())
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_attention_dimension_validation() {
        let batch_size = 2;
        let seq_len = 4;
        let d_model = 8;

        // Create query with wrong dimensions
        let query_data = vec![1.0f32; batch_size * seq_len];
        let query = Tensor::from_data(
            query_data,
            vec![batch_size, seq_len],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let key_data = vec![1.0f32; batch_size * seq_len * d_model];
        let key = Tensor::from_data(
            key_data,
            vec![batch_size, seq_len, d_model],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let value_data = vec![1.0f32; batch_size * seq_len * d_model];
        let value = Tensor::from_data(
            value_data,
            vec![batch_size, seq_len, d_model],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        // Should fail due to mismatched dimensions
        let result = scaled_dot_product_attention(&query, &key, &value, None, None);
        assert!(result.is_err());
    }
}
