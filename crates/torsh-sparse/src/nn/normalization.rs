//! Sparse normalization layers
//!
//! This module provides normalization layers optimized for sparse tensors. Normalization
//! is crucial for stable training in deep networks, and these implementations preserve
//! sparsity while providing the benefits of normalized activations.
//!
//! # Normalization Types
//! - Batch Normalization: Normalizes across the batch dimension
//! - Layer Normalization: Normalizes across feature dimensions
//!
//! # Sparse Optimizations
//! - Only compute statistics on non-zero elements
//! - Preserve sparsity patterns throughout normalization
//! - Efficient sparse tensor format conversions

use crate::{CooTensor, CsrTensor, CscTensor, SparseTensor, SparseFormat, TorshResult};
use scirs2_core::random::{Random, rng};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Helper function to unzip triplets for COO tensor creation
fn unzip_triplets(triplets: Vec<(usize, usize, f32)>) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    triplets.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut rows, mut cols, mut vals), (r, c, v)| {
            rows.push(r);
            cols.push(c);
            vals.push(v);
            (rows, cols, vals)
        },
    )
}

/// Sparse Batch Normalization layer
///
/// Applies batch normalization to sparse tensors by normalizing across the batch dimension.
/// Only non-zero elements are normalized, maintaining computational and memory efficiency.
///
/// # Mathematical Formulation
/// For each feature i:
/// - μ_i = (1/N) Σ x_i (mean of non-zero elements)
/// - σ²_i = (1/N) Σ (x_i - μ_i)² (variance of non-zero elements)
/// - y_i = γ_i * ((x_i - μ_i) / √(σ²_i + ε)) + β_i
///
/// Where γ_i and β_i are learnable parameters (if affine=true).
#[derive(Debug, Clone)]
pub struct SparseBatchNorm {
    /// Number of features
    num_features: usize,
    /// Small constant for numerical stability
    eps: f32,
    /// Momentum for running statistics
    momentum: f32,
    /// Whether to use affine transformation (scale and shift)
    affine: bool,
    /// Whether layer is in training mode
    training: bool,
    /// Running mean (maintained during training)
    running_mean: Tensor,
    /// Running variance (maintained during training)
    running_var: Tensor,
    /// Learnable scale parameter (gamma)
    weight: Option<Tensor>,
    /// Learnable shift parameter (beta)
    bias: Option<Tensor>,
}

impl SparseBatchNorm {
    /// Create a new sparse batch normalization layer
    ///
    /// # Arguments
    /// * `num_features` - Number of features to normalize
    /// * `eps` - Small constant for numerical stability (default: 1e-5)
    /// * `momentum` - Momentum for running statistics (default: 0.1)
    /// * `affine` - Whether to include learnable affine parameters
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::normalization::SparseBatchNorm;
    ///
    /// let bn = SparseBatchNorm::new(128, 1e-5, 0.1, true).unwrap();
    /// ```
    pub fn new(num_features: usize, eps: f32, momentum: f32, affine: bool) -> TorshResult<Self> {
        if num_features == 0 {
            return Err(TorshError::InvalidArgument(
                "Number of features must be greater than 0".to_string(),
            ));
        }

        if eps < 0.0 {
            return Err(TorshError::InvalidArgument(
                "Epsilon must be non-negative".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&momentum) {
            return Err(TorshError::InvalidArgument(
                "Momentum must be between 0.0 and 1.0".to_string(),
            ));
        }

        let running_mean = zeros::<f32>(&[num_features])?;
        let mut running_var = zeros::<f32>(&[num_features])?;
        // Initialize running variance to 1.0
        for i in 0..num_features {
            running_var.set(&[i], 1.0)?;
        }

        let (weight, bias) = if affine {
            let mut weight = zeros::<f32>(&[num_features])?;
            // Initialize weight to 1.0
            for i in 0..num_features {
                weight.set(&[i], 1.0)?;
            }
            let bias = zeros::<f32>(&[num_features])?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            training: true,
            running_mean,
            running_var,
            weight,
            bias,
        })
    }

    /// Set training mode
    ///
    /// # Arguments
    /// * `mode` - True for training mode, false for evaluation mode
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Check if layer is in training mode
    pub fn training(&self) -> bool {
        self.training
    }

    /// Forward pass for sparse tensors
    ///
    /// # Arguments
    /// * `input` - Input sparse tensor
    ///
    /// # Returns
    /// * Normalized sparse tensor in the same format as input
    pub fn forward(&mut self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        if self.training {
            // Calculate statistics from non-zero elements only
            self.update_statistics(&triplets)?;
        }

        // Apply normalization to non-zero elements
        let normalized_triplets = self.normalize_triplets(&triplets)?;

        // Create new COO tensor with normalized values
        let (rows, cols, values) = unzip_triplets(normalized_triplets);
        let normalized_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            SparseFormat::Coo => Ok(Box::new(normalized_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&normalized_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&normalized_coo)?)),
        }
    }

    /// Update running statistics during training
    fn update_statistics(&mut self, triplets: &[(usize, usize, f32)]) -> TorshResult<()> {
        // Calculate mean and variance for each feature (column)
        let mut feature_sums = vec![0.0f32; self.num_features];
        let mut feature_counts = vec![0usize; self.num_features];
        let mut feature_sq_sums = vec![0.0f32; self.num_features];

        // Accumulate statistics from non-zero elements
        for &(_, col, val) in triplets {
            if col < self.num_features {
                feature_sums[col] += val;
                feature_counts[col] += 1;
                feature_sq_sums[col] += val * val;
            }
        }

        // Update running statistics
        for i in 0..self.num_features {
            if feature_counts[i] > 0 {
                let mean = feature_sums[i] / feature_counts[i] as f32;
                let var = (feature_sq_sums[i] / feature_counts[i] as f32) - mean * mean;

                // Update running mean
                let old_mean = self.running_mean.get(&[i])?;
                let new_mean = (1.0 - self.momentum) * old_mean + self.momentum * mean;
                self.running_mean.set(&[i], new_mean)?;

                // Update running variance
                let old_var = self.running_var.get(&[i])?;
                let new_var = (1.0 - self.momentum) * old_var + self.momentum * var;
                self.running_var.set(&[i], new_var)?;
            }
        }

        Ok(())
    }

    /// Normalize sparse triplets
    fn normalize_triplets(
        &self,
        triplets: &[(usize, usize, f32)],
    ) -> TorshResult<Vec<(usize, usize, f32)>> {
        let mut normalized = Vec::with_capacity(triplets.len());

        for &(row, col, val) in triplets {
            if col < self.num_features {
                let mean = self.running_mean.get(&[col])?;
                let var = self.running_var.get(&[col])?;

                // Normalize: (x - mean) / sqrt(var + eps)
                let normalized_val = (val - mean) / (var + self.eps).sqrt();

                // Apply affine transformation if enabled
                let final_val = if self.affine {
                    let weight = self
                        .weight
                        .as_ref()
                        .ok_or_else(|| {
                            TorshError::InvalidState(
                                "Weight not initialized for affine transformation".to_string(),
                            )
                        })?
                        .get(&[col])?;
                    let bias = self
                        .bias
                        .as_ref()
                        .ok_or_else(|| {
                            TorshError::InvalidState(
                                "Bias not initialized for affine transformation".to_string(),
                            )
                        })?
                        .get(&[col])?;
                    normalized_val * weight + bias
                } else {
                    normalized_val
                };

                // Only keep non-zero values to maintain sparsity
                if final_val.abs() > 1e-10 {
                    normalized.push((row, col, final_val));
                }
            }
        }

        Ok(normalized)
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Get epsilon value
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Get momentum value
    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Check if affine transformation is enabled
    pub fn affine(&self) -> bool {
        self.affine
    }

    /// Get running mean (for inspection)
    pub fn running_mean(&self) -> &Tensor {
        &self.running_mean
    }

    /// Get running variance (for inspection)
    pub fn running_var(&self) -> &Tensor {
        &self.running_var
    }
}

/// Sparse Layer Normalization layer
///
/// Applies layer normalization to sparse tensors, normalizing across the last dimension
/// while preserving sparsity patterns. Layer normalization is particularly useful for
/// sequence models and transformers.
///
/// # Mathematical Formulation
/// For each sample x:
/// - μ = (1/H) Σ x_i (mean across features)
/// - σ² = (1/H) Σ (x_i - μ)² (variance across features)
/// - y_i = γ_i * ((x_i - μ) / √(σ² + ε)) + β_i
#[derive(Debug, Clone)]
pub struct SparseLayerNorm {
    /// Normalized shape (typically the last dimension)
    normalized_shape: Vec<usize>,
    /// Small constant for numerical stability
    eps: f32,
    /// Whether to use affine transformation
    elementwise_affine: bool,
    /// Learnable scale parameter
    weight: Option<Tensor>,
    /// Learnable shift parameter
    bias: Option<Tensor>,
}

impl SparseLayerNorm {
    /// Create a new sparse layer normalization layer
    ///
    /// # Arguments
    /// * `normalized_shape` - Shape of the dimensions to normalize (typically last dimension)
    /// * `eps` - Small constant for numerical stability (default: 1e-5)
    /// * `elementwise_affine` - Whether to include learnable affine parameters
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::normalization::SparseLayerNorm;
    ///
    /// // Normalize last dimension of size 512
    /// let ln = SparseLayerNorm::new(vec![512], 1e-5, true).unwrap();
    /// ```
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
    ) -> TorshResult<Self> {
        if normalized_shape.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Normalized shape cannot be empty".to_string(),
            ));
        }

        if eps < 0.0 {
            return Err(TorshError::InvalidArgument(
                "Epsilon must be non-negative".to_string(),
            ));
        }

        let total_elements: usize = normalized_shape.iter().product();

        let (weight, bias) = if elementwise_affine {
            let mut weight = zeros::<f32>(&[total_elements])?;
            // Initialize weight to 1.0
            for i in 0..total_elements {
                weight.set(&[i], 1.0)?;
            }
            let bias = zeros::<f32>(&[total_elements])?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
            bias,
        })
    }

    /// Forward pass for sparse tensors
    ///
    /// # Arguments
    /// * `input` - Input sparse tensor
    ///
    /// # Returns
    /// * Normalized sparse tensor in the same format as input
    pub fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert to COO for processing
        let coo = input.to_coo()?;
        let triplets = coo.triplets();
        let shape = input.shape().clone();

        // Apply layer normalization to each row (or the normalized dimension)
        let normalized_triplets = self.normalize_by_groups(&triplets, &shape)?;

        // Create new COO tensor with normalized values
        let (rows, cols, values) = unzip_triplets(normalized_triplets);
        let normalized_coo = CooTensor::new(rows, cols, values, shape)?;

        // Convert back to original format
        match input.format() {
            SparseFormat::Coo => Ok(Box::new(normalized_coo)),
            SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&normalized_coo)?)),
            SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&normalized_coo)?)),
        }
    }

    /// Normalize triplets by groups (typically by row for 2D matrices)
    fn normalize_by_groups(
        &self,
        triplets: &[(usize, usize, f32)],
        _shape: &Shape,
    ) -> TorshResult<Vec<(usize, usize, f32)>> {
        // Group triplets by the dimension we're normalizing over (typically rows)
        let mut groups: HashMap<usize, Vec<(usize, usize, f32)>> = HashMap::new();

        for &triplet in triplets {
            let group_key = triplet.0; // Group by row
            groups.entry(group_key).or_default().push(triplet);
        }

        let mut normalized = Vec::new();

        // Normalize each group
        for (_, group_triplets) in groups {
            if group_triplets.is_empty() {
                continue;
            }

            // Calculate mean and variance for this group
            let values: Vec<f32> = group_triplets.iter().map(|&(_, _, v)| v).collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
            let std_dev = (variance + self.eps).sqrt();

            // Normalize each element in the group
            for (row, col, val) in group_triplets {
                let normalized_val = (val - mean) / std_dev;

                // Apply affine transformation if enabled
                let final_val = if self.elementwise_affine {
                    let weight = self
                        .weight
                        .as_ref()
                        .ok_or_else(|| {
                            TorshError::InvalidState(
                                "Weight not initialized for elementwise affine transformation"
                                    .to_string(),
                            )
                        })?
                        .get(&[col % self.weight.as_ref().expect("weight should be present").shape().numel()])?;
                    let bias = self
                        .bias
                        .as_ref()
                        .ok_or_else(|| {
                            TorshError::InvalidState(
                                "Bias not initialized for elementwise affine transformation"
                                    .to_string(),
                            )
                        })?
                        .get(&[col % self.bias.as_ref().expect("bias should be present").shape().numel()])?;
                    normalized_val * weight + bias
                } else {
                    normalized_val
                };

                // Only keep non-zero values to maintain sparsity
                if final_val.abs() > 1e-10 {
                    normalized.push((row, col, final_val));
                }
            }
        }

        Ok(normalized)
    }

    /// Get normalized shape
    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    /// Get epsilon value
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Check if elementwise affine transformation is enabled
    pub fn elementwise_affine(&self) -> bool {
        self.elementwise_affine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_tensor::SparseFormat;

    #[test]
    fn test_sparse_batch_norm_creation() {
        let bn = SparseBatchNorm::new(64, 1e-5, 0.1, true).unwrap();
        assert_eq!(bn.num_features(), 64);
        assert_eq!(bn.eps(), 1e-5);
        assert_eq!(bn.momentum(), 0.1);
        assert!(bn.affine());
        assert!(bn.training());
    }

    #[test]
    fn test_sparse_layer_norm_creation() {
        let ln = SparseLayerNorm::new(vec![128], 1e-5, true).unwrap();
        assert_eq!(ln.normalized_shape(), &[128]);
        assert_eq!(ln.eps(), 1e-5);
        assert!(ln.elementwise_affine());
    }

    #[test]
    fn test_batch_norm_training_mode() {
        let mut bn = SparseBatchNorm::new(32, 1e-5, 0.1, false).unwrap();
        assert!(bn.training());
        bn.train(false);
        assert!(!bn.training());
        bn.train(true);
        assert!(bn.training());
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(SparseBatchNorm::new(0, 1e-5, 0.1, true).is_err()); // zero features
        assert!(SparseBatchNorm::new(64, -1e-5, 0.1, true).is_err()); // negative eps
        assert!(SparseBatchNorm::new(64, 1e-5, 1.5, true).is_err()); // invalid momentum

        assert!(SparseLayerNorm::new(vec![], 1e-5, true).is_err()); // empty shape
        assert!(SparseLayerNorm::new(vec![128], -1e-5, true).is_err()); // negative eps
    }

    #[test]
    fn test_unzip_triplets() {
        let triplets = vec![(0, 1, 2.0), (1, 0, 3.0), (2, 2, 4.0)];
        let (rows, cols, values) = unzip_triplets(triplets);
        assert_eq!(rows, vec![0, 1, 2]);
        assert_eq!(cols, vec![1, 0, 2]);
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_batch_norm_statistics() {
        let mut bn = SparseBatchNorm::new(3, 1e-5, 0.1, false).unwrap();

        // Test statistics update
        let triplets = vec![(0, 0, 1.0), (1, 0, 2.0), (0, 1, 3.0), (1, 1, 4.0)];
        bn.update_statistics(&triplets).unwrap();

        // Check that running statistics were updated
        let mean0 = bn.running_mean().get(&[0]).unwrap();
        let mean1 = bn.running_mean().get(&[1]).unwrap();
        assert!(mean0 > 0.0); // Should be updated from 0
        assert!(mean1 > 0.0); // Should be updated from 0
    }

    #[test]
    fn test_layer_norm_groups() {
        let ln = SparseLayerNorm::new(vec![4], 1e-5, false).unwrap();
        let triplets = vec![(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)];
        let shape = Shape::new(vec![2, 4]);

        let normalized = ln.normalize_by_groups(&triplets, &shape).unwrap();
        assert_eq!(normalized.len(), 4); // All non-zero values should be preserved
    }

    #[test]
    fn test_sparsity_preservation() {
        let bn = SparseBatchNorm::new(4, 1e-5, 0.1, false).unwrap();
        let triplets = vec![(0, 0, 1.0), (0, 2, 2.0)]; // Sparse: only columns 0 and 2 have values

        let normalized = bn.normalize_triplets(&triplets).unwrap();
        // Should maintain sparsity (only non-zero elements)
        assert!(normalized.len() <= triplets.len());
    }
}