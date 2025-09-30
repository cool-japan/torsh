//! Sparse attention mechanisms
//!
//! This module provides attention mechanisms optimized for sparse data and sparse
//! attention patterns. These implementations are designed for transformer-based
//! architectures and other attention-based neural networks where sparsity can
//! significantly reduce computational complexity.

use crate::layers::linear::SparseLinear;
use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};
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
/// by only computing attention for non-zero positions in the sparse attention mask.
/// This is particularly useful for long sequences or structured sparsity patterns
/// like local attention or hierarchical attention.
///
/// # Mathematical Formulation
/// For standard attention: Attention(Q,K,V) = softmax(QK^T / √d_k)V
/// For sparse attention: Only compute attention scores where mask[i,j] ≠ 0
///
/// # Benefits
/// - Reduces O(n²) complexity to O(s) where s is the number of sparse connections
/// - Maintains quality for many attention patterns (local, strided, etc.)
/// - Enables processing of much longer sequences
#[derive(Debug, Clone)]
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
    /// Dropout probability (for future implementation)
    #[allow(dead_code)]
    dropout: f32,
    /// Temperature scaling factor
    scale: f32,
}

impl SparseAttention {
    /// Create a new sparse attention layer
    ///
    /// # Arguments
    /// * `model_dim` - Model dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads
    /// * `sparsity` - Sparsity level for projection layers (0.0 = dense, 1.0 = fully sparse)
    /// * `dropout` - Dropout probability (currently unused)
    ///
    /// # Returns
    /// * `TorshResult<Self>` - New sparse attention layer or error
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::attention::SparseAttention;
    ///
    /// // Create 8-head attention with 512 model dimension and 0.9 sparsity
    /// let attention = SparseAttention::new(512, 8, 0.9, 0.1).unwrap();
    /// ```
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

        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TorshError::InvalidArgument(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&dropout) {
            return Err(TorshError::InvalidArgument(
                "Dropout must be between 0.0 and 1.0".to_string(),
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
    ///
    /// # Arguments
    /// * `query` - Query tensor (batch_size, seq_len, model_dim)
    /// * `key` - Key tensor (batch_size, seq_len, model_dim)
    /// * `value` - Value tensor (batch_size, seq_len, model_dim)
    /// * `attention_mask` - Optional sparse attention mask (seq_len, seq_len)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Attention output (batch_size, seq_len, model_dim)
    ///
    /// # Note
    /// If no attention mask is provided, standard dense attention is computed.
    /// The attention mask should be a sparse matrix where non-zero entries
    /// indicate which attention connections to compute.
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

    /// Self-attention convenience method
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, seq_len, model_dim)
    /// * `attention_mask` - Optional sparse attention mask
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Self-attention output
    pub fn self_attention(
        &self,
        input: &Tensor,
        attention_mask: Option<&CsrTensor>,
    ) -> TorshResult<Tensor> {
        self.forward(input, input, input, attention_mask)
    }

    /// Create a local attention mask
    ///
    /// Creates a sparse attention mask that only allows attention within
    /// a local window around each position.
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    /// * `window_size` - Size of the local window (total window = 2*window_size + 1)
    ///
    /// # Returns
    /// * `TorshResult<CsrTensor>` - Local attention mask
    pub fn create_local_attention_mask(
        seq_len: usize,
        window_size: usize,
    ) -> TorshResult<CsrTensor> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..seq_len {
            let start = i.saturating_sub(window_size);
            let end = std::cmp::min(i + window_size + 1, seq_len);

            for j in start..end {
                row_indices.push(i);
                col_indices.push(j);
                values.push(1.0);
            }
        }

        let shape = Shape::new(vec![seq_len, seq_len]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape)?;
        CsrTensor::from_coo(&coo)
    }

    /// Create a strided attention mask
    ///
    /// Creates a sparse attention mask that allows attention at regular
    /// intervals (stride pattern).
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    /// * `stride` - Stride between attended positions
    /// * `local_window` - Size of local window around each position
    ///
    /// # Returns
    /// * `TorshResult<CsrTensor>` - Strided attention mask
    pub fn create_strided_attention_mask(
        seq_len: usize,
        stride: usize,
        local_window: usize,
    ) -> TorshResult<CsrTensor> {
        if stride == 0 {
            return Err(TorshError::InvalidArgument(
                "Stride must be greater than 0".to_string(),
            ));
        }

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..seq_len {
            // Local window
            let local_start = i.saturating_sub(local_window);
            let local_end = std::cmp::min(i + local_window + 1, seq_len);
            for j in local_start..local_end {
                row_indices.push(i);
                col_indices.push(j);
                values.push(1.0);
            }

            // Strided connections
            let mut j = i % stride;
            while j < seq_len {
                if j < local_start || j >= local_end {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(1.0);
                }
                j += stride;
            }
        }

        let shape = Shape::new(vec![seq_len, seq_len]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape)?;
        CsrTensor::from_coo(&coo)
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

        if q_shape.dims()[0] != k_shape.dims()[0] || q_shape.dims()[0] != v_shape.dims()[0] {
            return Err(TorshError::InvalidArgument(
                "Batch sizes must match across Q, K, V".to_string(),
            ));
        }

        if k_shape.dims()[1] != v_shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Key and Value sequence lengths must match".to_string(),
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

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get scale factor
    pub fn scale(&self) -> f32 {
        self.scale
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

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_sparse_attention_creation() {
        let attention = SparseAttention::new(64, 8, 0.5, 0.1).unwrap();
        assert_eq!(attention.model_dim(), 64);
        assert_eq!(attention.num_heads(), 8);
        assert_eq!(attention.head_dim(), 8);
        assert!(attention.num_parameters() > 0);
    }

    #[test]
    fn test_invalid_model_dim() {
        // Model dim not divisible by num_heads
        assert!(SparseAttention::new(65, 8, 0.5, 0.1).is_err());
    }

    #[test]
    fn test_invalid_sparsity() {
        assert!(SparseAttention::new(64, 8, 1.5, 0.1).is_err());
        assert!(SparseAttention::new(64, 8, -0.1, 0.1).is_err());
    }

    #[test]
    fn test_sparse_attention_forward() {
        let attention = SparseAttention::new(32, 4, 0.3, 0.1).unwrap();
        let query = ones::<f32>(&[2, 5, 32]).unwrap();
        let key = ones::<f32>(&[2, 5, 32]).unwrap();
        let value = ones::<f32>(&[2, 5, 32]).unwrap();

        let output = attention.forward(&query, &key, &value, None).unwrap();
        assert_eq!(output.shape().dims(), &[2, 5, 32]);
    }

    #[test]
    fn test_self_attention() {
        let attention = SparseAttention::new(16, 2, 0.4, 0.0).unwrap();
        let input = ones::<f32>(&[1, 4, 16]).unwrap();

        let output = attention.self_attention(&input, None).unwrap();
        assert_eq!(output.shape().dims(), &[1, 4, 16]);
    }

    #[test]
    fn test_local_attention_mask() {
        let mask = SparseAttention::create_local_attention_mask(5, 1).unwrap();
        assert_eq!(mask.shape().dims(), &[5, 5]);
        assert!(mask.nnz() > 0);
        assert!(mask.nnz() <= 15); // 5 positions * 3 connections each (max)
    }

    #[test]
    fn test_strided_attention_mask() {
        let mask = SparseAttention::create_strided_attention_mask(8, 2, 1).unwrap();
        assert_eq!(mask.shape().dims(), &[8, 8]);
        assert!(mask.nnz() > 0);
    }

    #[test]
    fn test_attention_with_local_mask() {
        let attention = SparseAttention::new(16, 2, 0.2, 0.0).unwrap();
        let input = ones::<f32>(&[1, 4, 16]).unwrap();
        let mask = SparseAttention::create_local_attention_mask(4, 1).unwrap();

        let output = attention.self_attention(&input, Some(&mask)).unwrap();
        assert_eq!(output.shape().dims(), &[1, 4, 16]);
    }

    #[test]
    fn test_dimension_validation() {
        let attention = SparseAttention::new(32, 4, 0.3, 0.1).unwrap();
        let wrong_query = ones::<f32>(&[2, 5, 16]).unwrap(); // Wrong model dim
        let key = ones::<f32>(&[2, 5, 32]).unwrap();
        let value = ones::<f32>(&[2, 5, 32]).unwrap();

        assert!(attention.forward(&wrong_query, &key, &value, None).is_err());
    }

    #[test]
    fn test_batch_size_validation() {
        let attention = SparseAttention::new(16, 2, 0.3, 0.1).unwrap();
        let query = ones::<f32>(&[2, 5, 16]).unwrap();
        let key = ones::<f32>(&[3, 5, 16]).unwrap(); // Different batch size
        let value = ones::<f32>(&[2, 5, 16]).unwrap();

        assert!(attention.forward(&query, &key, &value, None).is_err());
    }

    #[test]
    fn test_sequence_length_validation() {
        let attention = SparseAttention::new(16, 2, 0.3, 0.1).unwrap();
        let query = ones::<f32>(&[2, 5, 16]).unwrap();
        let key = ones::<f32>(&[2, 4, 16]).unwrap(); // Different seq len
        let value = ones::<f32>(&[2, 4, 16]).unwrap();

        assert!(attention.forward(&query, &key, &value, None).is_err());
    }

    #[test]
    fn test_invalid_stride() {
        assert!(SparseAttention::create_strided_attention_mask(8, 0, 1).is_err());
    }
}
