//! Sparse attention mechanisms
//!
//! This module provides attention mechanisms optimized for sparse tensors,
//! including multi-head attention with sparse patterns for improved efficiency.

use crate::layers::linear::SparseLinear;
use crate::{CooTensor, CscTensor, CsrTensor, SparseTensor, TorshResult};
use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Sparse Multi-Head Attention mechanism
///
/// Implements efficient attention for sparse matrices, reducing computational complexity
/// by only computing attention for non-zero positions in the sparse attention mask
pub struct SparseAttention {
    /// Query projection layer
    query_proj: SparseLinear,
    /// Key projection layer
    key_proj: SparseLinear,
    /// Value projection layer
    value_proj: SparseLinear,
    /// Output projection layer
    output_proj: SparseLinear,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Model dimension
    model_dim: usize,
    /// Dropout probability (not implemented yet)
    #[allow(dead_code)]
    dropout: f32,
    /// Temperature scaling factor
    scale: f32,
}

impl SparseAttention {
    /// Create a new sparse attention layer
    pub fn new(
        model_dim: usize,
        num_heads: usize,
        sparsity: f32,
        dropout: f32,
    ) -> TorshResult<Self> {
        if model_dim % num_heads != 0 {
            return Err(TorshError::InvalidArgument(
                "Model dimension must be divisible by number of heads".to_string(),
            ));
        }

        let head_dim = model_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Create projection layers with sparse weights
        let query_proj = SparseLinear::new(model_dim, model_dim, sparsity, false)?;
        let key_proj = SparseLinear::new(model_dim, model_dim, sparsity, false)?;
        let value_proj = SparseLinear::new(model_dim, model_dim, sparsity, false)?;
        let output_proj = SparseLinear::new(model_dim, model_dim, sparsity, false)?;

        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            num_heads,
            head_dim,
            model_dim,
            dropout,
            scale,
        })
    }

    /// Forward pass with sparse attention mask
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&CsrTensor>,
    ) -> TorshResult<Tensor> {
        let batch_size = query.shape().dims()[0];
        let seq_len = query.shape().dims()[1];

        // Validate input shapes
        self.validate_inputs(query, key, value)?;

        // Reshape 3D input to 2D for linear projections (batch*seq_len, model_dim)
        let query_2d = self.reshape_3d_to_2d(query, batch_size, seq_len)?;
        let key_2d = self.reshape_3d_to_2d(key, batch_size, seq_len)?;
        let value_2d = self.reshape_3d_to_2d(value, batch_size, seq_len)?;

        // Project to Q, K, V
        let q_2d = self.query_proj.forward(&query_2d)?;
        let k_2d = self.key_proj.forward(&key_2d)?;
        let v_2d = self.value_proj.forward(&value_2d)?;

        // Reshape back to 3D
        let q = self.reshape_2d_to_3d(&q_2d, batch_size, seq_len)?;
        let k = self.reshape_2d_to_3d(&k_2d, batch_size, seq_len)?;
        let v = self.reshape_2d_to_3d(&v_2d, batch_size, seq_len)?;

        // Reshape for multi-head attention
        let q_reshaped = self.reshape_for_attention(&q, batch_size, seq_len)?;
        let k_reshaped = self.reshape_for_attention(&k, batch_size, seq_len)?;
        let v_reshaped = self.reshape_for_attention(&v, batch_size, seq_len)?;

        // Compute attention scores with sparsity
        let attention_output = self.compute_sparse_attention(
            &q_reshaped,
            &k_reshaped,
            &v_reshaped,
            batch_size,
            seq_len,
            attention_mask,
        )?;

        // Reshape back and apply output projection
        let output_reshaped =
            self.reshape_from_attention(&attention_output, batch_size, seq_len)?;
        let output_2d = self.reshape_3d_to_2d(&output_reshaped, batch_size, seq_len)?;
        let projected_2d = self.output_proj.forward(&output_2d)?;
        self.reshape_2d_to_3d(&projected_2d, batch_size, seq_len)
    }

    /// Validate input tensor shapes
    fn validate_inputs(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> TorshResult<()> {
        let q_shape = query.shape();
        let k_shape = key.shape();
        let v_shape = value.shape();

        if q_shape.ndim() != 3 || k_shape.ndim() != 3 || v_shape.ndim() != 3 {
            return Err(TorshError::InvalidArgument(
                "Input tensors must be 3D (batch, seq_len, model_dim)".to_string(),
            ));
        }

        if q_shape.dims()[2] != self.model_dim {
            return Err(TorshError::InvalidArgument(
                "Query dimension doesn't match model dimension".to_string(),
            ));
        }

        if k_shape.dims()[2] != self.model_dim || v_shape.dims()[2] != self.model_dim {
            return Err(TorshError::InvalidArgument(
                "Key/Value dimensions don't match model dimension".to_string(),
            ));
        }

        Ok(())
    }

    /// Reshape tensor for multi-head attention
    fn reshape_for_attention(
        &self,
        tensor: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> TorshResult<Tensor> {
        // Reshape from (batch, seq_len, model_dim) to (batch, num_heads, seq_len, head_dim)
        let reshaped = zeros::<f32>(&[batch_size, self.num_heads, seq_len, self.head_dim])?;

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        let model_idx = h * self.head_dim + d;
                        let val = tensor.get(&[b, s, model_idx])?;
                        reshaped.set(&[b, h, s, d], val)?;
                    }
                }
            }
        }

        Ok(reshaped)
    }

    /// Reshape tensor back from multi-head attention
    fn reshape_from_attention(
        &self,
        tensor: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> TorshResult<Tensor> {
        // Reshape from (batch, num_heads, seq_len, head_dim) to (batch, seq_len, model_dim)
        let reshaped = zeros::<f32>(&[batch_size, seq_len, self.model_dim])?;

        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        let model_idx = h * self.head_dim + d;
                        let val = tensor.get(&[b, h, s, d])?;
                        reshaped.set(&[b, s, model_idx], val)?;
                    }
                }
            }
        }

        Ok(reshaped)
    }

    /// Compute sparse attention with optional attention mask
    #[allow(clippy::too_many_arguments)]
    fn compute_sparse_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        batch_size: usize,
        seq_len: usize,
        attention_mask: Option<&CsrTensor>,
    ) -> TorshResult<Tensor> {
        let output = zeros::<f32>(&[batch_size, self.num_heads, seq_len, self.head_dim])?;

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                // Compute attention scores for this head
                let scores = zeros::<f32>(&[seq_len, seq_len])?;

                for i in 0..seq_len {
                    for j in 0..seq_len {
                        // Check if this position should be computed (based on sparse mask)
                        let should_compute = if let Some(mask) = attention_mask {
                            // Only compute if position is non-zero in mask
                            let (cols, _) = mask.get_row(i)?;
                            cols.contains(&j)
                        } else {
                            true // Compute all positions if no mask
                        };

                        if should_compute {
                            let mut score = 0.0;
                            for d in 0..self.head_dim {
                                score += query.get(&[b, h, i, d])? * key.get(&[b, h, j, d])?;
                            }
                            scores.set(&[i, j], score * self.scale)?;
                        } else {
                            scores.set(&[i, j], f32::NEG_INFINITY)?;
                        }
                    }
                }

                // Apply softmax to attention scores
                for i in 0..seq_len {
                    // Find max for numerical stability
                    let mut max_score = f32::NEG_INFINITY;
                    for j in 0..seq_len {
                        let score = scores.get(&[i, j])?;
                        if score > max_score && score != f32::NEG_INFINITY {
                            max_score = score;
                        }
                    }

                    // Compute softmax
                    let mut sum_exp = 0.0;
                    let mut exp_scores = vec![0.0; seq_len];
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..seq_len {
                        let score = scores.get(&[i, j])?;
                        if score != f32::NEG_INFINITY {
                            exp_scores[j] = (score - max_score).exp();
                            sum_exp += exp_scores[j];
                        }
                    }

                    // Normalize and apply to values
                    for d in 0..self.head_dim {
                        let mut weighted_sum = 0.0;
                        #[allow(clippy::needless_range_loop)]
                        for j in 0..seq_len {
                            if exp_scores[j] > 0.0 {
                                let attention_weight = exp_scores[j] / sum_exp;
                                weighted_sum += attention_weight * value.get(&[b, h, j, d])?;
                            }
                        }
                        output.set(&[b, h, i, d], weighted_sum)?;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.query_proj.num_parameters()
            + self.key_proj.num_parameters()
            + self.value_proj.num_parameters()
            + self.output_proj.num_parameters()
    }

    /// Get model dimension
    pub fn model_dim(&self) -> usize {
        self.model_dim
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Reshape 3D tensor to 2D for linear projections
    fn reshape_3d_to_2d(
        &self,
        tensor: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> TorshResult<Tensor> {
        let reshaped = zeros::<f32>(&[batch_size * seq_len, self.model_dim])?;

        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..self.model_dim {
                    let val = tensor.get(&[b, s, d])?;
                    reshaped.set(&[b * seq_len + s, d], val)?;
                }
            }
        }

        Ok(reshaped)
    }

    /// Reshape 2D tensor back to 3D
    fn reshape_2d_to_3d(
        &self,
        tensor: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> TorshResult<Tensor> {
        let reshaped = zeros::<f32>(&[batch_size, seq_len, self.model_dim])?;

        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..self.model_dim {
                    let val = tensor.get(&[b * seq_len + s, d])?;
                    reshaped.set(&[b, s, d], val)?;
                }
            }
        }

        Ok(reshaped)
    }
}
