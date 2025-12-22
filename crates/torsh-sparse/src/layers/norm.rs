//! Sparse normalization layers
//!
//! This module provides normalization operations optimized for sparse tensors,
//! including batch normalization and layer normalization while preserving sparsity patterns.

use crate::{CooTensor, CscTensor, CsrTensor, SparseTensor, TorshResult};
use torsh_core::{Shape, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// Helper function to unzip triplets
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
/// This layer applies batch normalization to sparse tensors while preserving sparsity patterns.
/// Only non-zero elements are normalized, maintaining computational and memory efficiency.
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
    pub fn new(num_features: usize, eps: f32, momentum: f32, affine: bool) -> TorshResult<Self> {
        let running_mean = zeros::<f32>(&[num_features])?;
        let running_var = zeros::<f32>(&[num_features])?;

        let (weight, bias) = if affine {
            let weight = zeros::<f32>(&[num_features])?;
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
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Forward pass for sparse tensors
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
            crate::SparseFormat::Coo => Ok(Box::new(normalized_coo)),
            crate::SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&normalized_coo)?)),
            crate::SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&normalized_coo)?)),
            _ => Ok(Box::new(normalized_coo)), // Default to COO for other formats
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

                normalized.push((row, col, final_val));
            }
        }

        Ok(normalized)
    }
}

/// Sparse Layer Normalization layer
///
/// Applies layer normalization to sparse tensors, normalizing across the last dimension
/// while preserving sparsity patterns.
pub struct SparseLayerNorm {
    /// Normalized shape (typically the last dimension)
    #[allow(dead_code)]
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
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
    ) -> TorshResult<Self> {
        let total_elements: usize = normalized_shape.iter().product();

        let (weight, bias) = if elementwise_affine {
            let weight = zeros::<f32>(&[total_elements])?;
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
            crate::SparseFormat::Coo => Ok(Box::new(normalized_coo)),
            crate::SparseFormat::Csr => Ok(Box::new(CsrTensor::from_coo(&normalized_coo)?)),
            crate::SparseFormat::Csc => Ok(Box::new(CscTensor::from_coo(&normalized_coo)?)),
            _ => Ok(Box::new(normalized_coo)), // Default to COO for other formats
        }
    }

    /// Normalize triplets by groups (typically by row for 2D matrices)
    fn normalize_by_groups(
        &self,
        triplets: &[(usize, usize, f32)],
        _shape: &Shape,
    ) -> TorshResult<Vec<(usize, usize, f32)>> {
        // Group triplets by the dimension we're normalizing over (typically rows)
        let mut groups: std::collections::HashMap<usize, Vec<(usize, usize, f32)>> =
            std::collections::HashMap::new();

        for &triplet in triplets {
            let group_key = triplet.0; // Group by row
            groups.entry(group_key).or_default().push(triplet);
        }

        let mut normalized = Vec::new();

        // Normalize each group
        for (_, group_triplets) in groups {
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
                        .get(&[col])?;
                    let bias = self
                        .bias
                        .as_ref()
                        .ok_or_else(|| {
                            TorshError::InvalidState(
                                "Bias not initialized for elementwise affine transformation"
                                    .to_string(),
                            )
                        })?
                        .get(&[col])?;
                    normalized_val * weight + bias
                } else {
                    normalized_val
                };

                normalized.push((row, col, final_val));
            }
        }

        Ok(normalized)
    }
}
