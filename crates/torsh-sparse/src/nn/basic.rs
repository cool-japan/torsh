//! Sparse basic neural network layers
//!
//! This module provides fundamental sparse neural network layers that form the building blocks
//! for larger sparse architectures. These layers are optimized for sparse tensors and can
//! significantly reduce computation when dealing with sparse data.

use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};
use scirs2_core::random::{Random, rng};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Sparse linear layer (also known as sparse fully connected layer)
///
/// This layer implements a linear transformation with sparse weight matrices,
/// which can significantly reduce memory usage and computation for sparse inputs.
/// The layer supports both structured and unstructured pruning for model compression.
#[derive(Debug, Clone)]
pub struct SparseLinear {
    /// Weight matrix in sparse format
    weight: CsrTensor,
    /// Optional bias vector
    bias: Option<Tensor>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Sparsity level (fraction of zero weights)
    sparsity: f32,
}

impl SparseLinear {
    /// Create a new sparse linear layer
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `sparsity` - Sparsity level (0.0 = dense, 1.0 = fully sparse)
    /// * `use_bias` - Whether to include bias term
    ///
    /// # Returns
    /// * `TorshResult<Self>` - New sparse linear layer or error
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::basic::SparseLinear;
    ///
    /// let layer = SparseLinear::new(784, 128, 0.9, true).unwrap();
    /// ```
    pub fn new(
        in_features: usize,
        out_features: usize,
        sparsity: f32,
        use_bias: bool,
    ) -> TorshResult<Self> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TorshError::InvalidArgument(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Generate sparse weight matrix
        let weight = Self::generate_sparse_weights(in_features, out_features, sparsity)?;

        // Generate bias if requested
        let bias = if use_bias {
            Some(randn::<f32>(&[out_features])?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            sparsity,
        })
    }

    /// Create from existing sparse weight matrix
    ///
    /// # Arguments
    /// * `weight` - Pre-trained sparse weight matrix
    /// * `bias` - Optional bias vector
    ///
    /// # Returns
    /// * `TorshResult<Self>` - New sparse linear layer or error
    pub fn from_weight(weight: CsrTensor, bias: Option<Tensor>) -> TorshResult<Self> {
        let shape = weight.shape();
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Weight matrix must be 2D".to_string(),
            ));
        }

        let out_features = shape.dims()[0];
        let in_features = shape.dims()[1];

        if let Some(ref bias_tensor) = bias {
            if bias_tensor.shape().dims() != [out_features] {
                return Err(TorshError::InvalidArgument(
                    "Bias dimension must match output features".to_string(),
                ));
            }
        }

        let total_elements = out_features * in_features;
        let nnz = weight.nnz();
        let sparsity = 1.0 - (nnz as f32 / total_elements as f32);

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            sparsity,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `input` - Input tensor (1D or 2D for batches)
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Output tensor or error
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape
        let input_shape = input.shape();
        let batch_size = if input_shape.ndim() == 1 {
            1
        } else if input_shape.ndim() == 2 {
            input_shape.dims()[0]
        } else {
            return Err(TorshError::InvalidArgument(
                "Input must be 1D or 2D tensor".to_string(),
            ));
        };

        let input_features = if input_shape.ndim() == 1 {
            input_shape.dims()[0]
        } else {
            input_shape.dims()[1]
        };

        if input_features != self.in_features {
            return Err(TorshError::InvalidArgument(format!(
                "Input features {} don't match layer input features {}",
                input_features, self.in_features
            )));
        }

        // Handle batch processing
        let output = if input_shape.ndim() == 1 {
            // Single sample
            self.weight.matvec(input)?
        } else {
            // Batch processing
            let batch_output = zeros::<f32>(&[batch_size, self.out_features])?;

            for b in 0..batch_size {
                // Extract single sample from batch
                let sample = zeros::<f32>(&[self.in_features])?;
                for i in 0..self.in_features {
                    sample.set(&[i], input.get(&[b, i])?)?;
                }

                // Process sample
                let sample_output = self.weight.matvec(&sample)?;

                // Store in batch output
                for i in 0..self.out_features {
                    batch_output.set(&[b, i], sample_output.get(&[i])?)?;
                }
            }

            batch_output
        };

        // Add bias if present
        if let Some(ref bias) = self.bias {
            if input_shape.ndim() == 1 {
                // Single sample
                for i in 0..self.out_features {
                    output.set(&[i], output.get(&[i])? + bias.get(&[i])?)?;
                }
            } else {
                // Batch processing
                for b in 0..batch_size {
                    for i in 0..self.out_features {
                        let current = output.get(&[b, i])?;
                        output.set(&[b, i], current + bias.get(&[i])?)?;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get weight sparsity
    pub fn sparsity(&self) -> f32 {
        self.sparsity
    }

    /// Get number of parameters (including bias)
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.weight.nnz();
        let bias_params = self.bias.as_ref().map_or(0, |b| b.shape().numel());
        weight_params + bias_params
    }

    /// Apply structured pruning (remove entire rows/columns)
    ///
    /// # Arguments
    /// * `ratio` - Fraction of rows/columns to remove (0.0 to 1.0)
    /// * `dimension` - 0 for row pruning, 1 for column pruning
    ///
    /// # Returns
    /// * `TorshResult<()>` - Success or error
    pub fn structured_prune(&mut self, ratio: f32, dimension: usize) -> TorshResult<()> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(TorshError::InvalidArgument(
                "Pruning ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        if dimension > 1 {
            return Err(TorshError::InvalidArgument(
                "Dimension must be 0 (rows) or 1 (columns)".to_string(),
            ));
        }

        // Convert to COO for easier manipulation
        let coo = self.weight.to_coo()?;
        let triplets = coo.triplets();

        // Calculate importance scores (L2 norm of rows/columns)
        let importance_scores = if dimension == 0 {
            // Row-wise pruning
            let mut scores = vec![0.0; self.out_features];
            for (row, _, val) in &triplets {
                scores[*row] += val * val;
            }
            scores.iter_mut().for_each(|x| *x = x.sqrt());
            scores
        } else {
            // Column-wise pruning
            let mut scores = vec![0.0; self.in_features];
            for (_, col, val) in &triplets {
                scores[*col] += val * val;
            }
            scores.iter_mut().for_each(|x| *x = x.sqrt());
            scores
        };

        // Select indices to keep
        let num_to_prune = (importance_scores.len() as f32 * ratio) as usize;
        let mut indexed_scores: Vec<(usize, f32)> =
            importance_scores.into_iter().enumerate().collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)); // Descending order

        let keep_indices: std::collections::HashSet<usize> = indexed_scores
            .into_iter()
            .skip(num_to_prune)
            .map(|(idx, _)| idx)
            .collect();

        // Filter triplets
        let filtered_triplets: Vec<_> = triplets
            .into_iter()
            .filter(|(row, col, _)| {
                if dimension == 0 {
                    keep_indices.contains(row)
                } else {
                    keep_indices.contains(col)
                }
            })
            .collect();

        // Rebuild sparse matrix
        let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) =
            filtered_triplets.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            );

        let pruned_coo = CooTensor::new(
            row_indices,
            col_indices,
            values,
            self.weight.shape().clone(),
        )?;
        self.weight = CsrTensor::from_coo(&pruned_coo)?;

        // Update sparsity
        let total_elements = self.out_features * self.in_features;
        let nnz = self.weight.nnz();
        self.sparsity = 1.0 - (nnz as f32 / total_elements as f32);

        Ok(())
    }

    /// Apply magnitude-based unstructured pruning
    ///
    /// # Arguments
    /// * `ratio` - Fraction of weights to prune (0.0 to 1.0)
    ///
    /// # Returns
    /// * `TorshResult<()>` - Success or error
    pub fn magnitude_prune(&mut self, ratio: f32) -> TorshResult<()> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(TorshError::InvalidArgument(
                "Pruning ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        let coo = self.weight.to_coo()?;
        let triplets = coo.triplets();

        // Sort by magnitude
        let mut magnitude_triplets: Vec<_> = triplets
            .into_iter()
            .map(|(r, c, v)| (r, c, v, v.abs()))
            .collect();
        magnitude_triplets
            .sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal)); // Descending order

        // Keep top (1 - ratio) fraction
        let num_to_keep = ((magnitude_triplets.len() as f32) * (1.0 - ratio)) as usize;
        let kept_triplets = &magnitude_triplets[..num_to_keep];

        // Rebuild sparse matrix
        let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) =
            kept_triplets.iter().map(|(r, c, v, _)| (*r, *c, *v)).fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut rows, mut cols, mut vals), (r, c, v)| {
                    rows.push(r);
                    cols.push(c);
                    vals.push(v);
                    (rows, cols, vals)
                },
            );

        let pruned_coo = CooTensor::new(
            row_indices,
            col_indices,
            values,
            self.weight.shape().clone(),
        )?;
        self.weight = CsrTensor::from_coo(&pruned_coo)?;

        // Update sparsity
        let total_elements = self.out_features * self.in_features;
        let nnz = self.weight.nnz();
        self.sparsity = 1.0 - (nnz as f32 / total_elements as f32);

        Ok(())
    }

    /// Generate sparse weight matrix
    fn generate_sparse_weights(
        in_features: usize,
        out_features: usize,
        sparsity: f32,
    ) -> TorshResult<CsrTensor> {
        let total_elements = in_features * out_features;
        let nnz = ((total_elements as f32) * (1.0 - sparsity)) as usize;

        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Generate random sparse pattern
        let mut positions = std::collections::HashSet::new();
        while positions.len() < nnz {
            let mut rng = scirs2_core::random::thread_rng();
            let row = rng.gen_range(0.. out_features);
            let col = rng.gen_range(0.. in_features);
            positions.insert((row, col));
        }

        // Convert to COO format with random values
        for (row, col) in positions {
            row_indices.push(row);
            col_indices.push(col);
            // Xavier/Glorot initialization
            let std_dev = (2.0 / (in_features + out_features) as f32).sqrt();
            let mut rng = scirs2_core::random::thread_rng();
            values.push(rng.random::<f32>() * 2.0 * std_dev - std_dev);
        }

        let shape = Shape::new(vec![out_features, in_features]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape)?;
        CsrTensor::from_coo(&coo)
    }
}

/// Sparse embedding layer
///
/// This layer implements sparse embeddings for natural language processing and
/// recommendation systems. The sparse representation allows for efficient storage
/// and computation when dealing with large vocabularies.
#[derive(Debug, Clone)]
pub struct SparseEmbedding {
    /// Embedding weight matrix (vocab_size x embedding_dim)
    weight: CsrTensor,
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embedding_dim: usize,
    /// Sparsity level
    sparsity: f32,
}

impl SparseEmbedding {
    /// Create a new sparse embedding layer
    ///
    /// # Arguments
    /// * `vocab_size` - Size of the vocabulary
    /// * `embedding_dim` - Dimension of embeddings
    /// * `sparsity` - Sparsity level (0.0 = dense, 1.0 = fully sparse)
    ///
    /// # Returns
    /// * `TorshResult<Self>` - New sparse embedding layer or error
    ///
    /// # Example
    /// ```rust
    /// use torsh_sparse::nn::basic::SparseEmbedding;
    ///
    /// let embedding = SparseEmbedding::new(10000, 300, 0.8).unwrap();
    /// ```
    pub fn new(vocab_size: usize, embedding_dim: usize, sparsity: f32) -> TorshResult<Self> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TorshError::InvalidArgument(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Generate sparse embedding matrix
        let weight = Self::generate_sparse_embeddings(vocab_size, embedding_dim, sparsity)?;

        Ok(Self {
            weight,
            vocab_size,
            embedding_dim,
            sparsity,
        })
    }

    /// Forward pass - lookup embeddings for given indices
    ///
    /// # Arguments
    /// * `indices` - Vocabulary indices to lookup
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Batch of embeddings or error
    pub fn forward(&self, indices: &[usize]) -> TorshResult<Tensor> {
        // Validate indices
        for &idx in indices {
            if idx >= self.vocab_size {
                return Err(TorshError::InvalidArgument(format!(
                    "Index {} out of vocabulary range [0, {})",
                    idx, self.vocab_size
                )));
            }
        }

        let batch_size = indices.len();
        let output = zeros::<f32>(&[batch_size, self.embedding_dim])?;

        // Look up each embedding
        for (batch_idx, &vocab_idx) in indices.iter().enumerate() {
            let (cols, vals) = self.weight.get_row(vocab_idx)?;

            // Set the embedding values
            for (&col, &val) in cols.iter().zip(vals.iter()) {
                output.set(&[batch_idx, col], val)?;
            }
        }

        Ok(output)
    }

    /// Get embedding for a single index
    ///
    /// # Arguments
    /// * `index` - Vocabulary index
    ///
    /// # Returns
    /// * `TorshResult<Tensor>` - Single embedding vector or error
    pub fn get_embedding(&self, index: usize) -> TorshResult<Tensor> {
        if index >= self.vocab_size {
            return Err(TorshError::InvalidArgument(format!(
                "Index {} out of vocabulary range [0, {})",
                index, self.vocab_size
            )));
        }

        let embedding = zeros::<f32>(&[self.embedding_dim])?;
        let (cols, vals) = self.weight.get_row(index)?;

        for (&col, &val) in cols.iter().zip(vals.iter()) {
            embedding.set(&[col], val)?;
        }

        Ok(embedding)
    }

    /// Get sparsity level
    pub fn sparsity(&self) -> f32 {
        self.sparsity
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weight.nnz()
    }

    /// Generate sparse embedding matrix
    fn generate_sparse_embeddings(
        vocab_size: usize,
        embedding_dim: usize,
        sparsity: f32,
    ) -> TorshResult<CsrTensor> {
        let total_elements = vocab_size * embedding_dim;
        let nnz = ((total_elements as f32) * (1.0 - sparsity)) as usize;

        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Generate random sparse pattern
        let mut positions = std::collections::HashSet::new();
        while positions.len() < nnz {
            let mut rng = scirs2_core::random::thread_rng();
            let row = rng.gen_range(0.. vocab_size);
            let col = rng.gen_range(0.. embedding_dim);
            positions.insert((row, col));
        }

        // Convert to COO format with random values
        for (row, col) in positions {
            row_indices.push(row);
            col_indices.push(col);
            // Normal initialization with std=1.0
            let mut rng = scirs2_core::random::thread_rng();
            values.push(rng.random::<f32>() * 2.0 - 1.0);
        }

        let shape = Shape::new(vec![vocab_size, embedding_dim]);
        let coo = CooTensor::new(row_indices, col_indices, values, shape)?;
        CsrTensor::from_coo(&coo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_sparse_linear_creation() {
        let layer = SparseLinear::new(10, 5, 0.5, true).unwrap();
        assert_eq!(layer.sparsity(), 0.5);
        assert!(layer.num_parameters() > 0);
    }

    #[test]
    fn test_sparse_linear_forward() {
        let layer = SparseLinear::new(4, 2, 0.3, false).unwrap();
        let input = ones::<f32>(&[4]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2]);
    }

    #[test]
    fn test_sparse_linear_batch_forward() {
        let layer = SparseLinear::new(3, 2, 0.4, true).unwrap();
        let input = ones::<f32>(&[2, 3]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_sparse_embedding_creation() {
        let embedding = SparseEmbedding::new(100, 50, 0.7).unwrap();
        assert_eq!(embedding.sparsity(), 0.7);
        assert!(embedding.num_parameters() > 0);
    }

    #[test]
    fn test_sparse_embedding_forward() {
        let embedding = SparseEmbedding::new(10, 4, 0.6).unwrap();
        let indices = vec![0, 1, 2];
        let output = embedding.forward(&indices).unwrap();
        assert_eq!(output.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_sparse_embedding_single_lookup() {
        let embedding = SparseEmbedding::new(5, 3, 0.5).unwrap();
        let output = embedding.get_embedding(2).unwrap();
        assert_eq!(output.shape().dims(), &[3]);
    }

    #[test]
    fn test_magnitude_pruning() {
        let mut layer = SparseLinear::new(4, 3, 0.2, false).unwrap();
        let original_sparsity = layer.sparsity();
        layer.magnitude_prune(0.3).unwrap();
        assert!(layer.sparsity() > original_sparsity);
    }

    #[test]
    fn test_structured_pruning() {
        let mut layer = SparseLinear::new(6, 4, 0.3, false).unwrap();
        let original_sparsity = layer.sparsity();
        layer.structured_prune(0.25, 0).unwrap(); // Prune 25% of rows
        assert!(layer.sparsity() >= original_sparsity);
    }

    #[test]
    fn test_invalid_sparsity() {
        assert!(SparseLinear::new(10, 5, 1.5, true).is_err());
        assert!(SparseEmbedding::new(100, 50, -0.1).is_err());
    }

    #[test]
    fn test_invalid_indices() {
        let embedding = SparseEmbedding::new(5, 3, 0.5).unwrap();
        assert!(embedding.forward(&[0, 1, 10]).is_err()); // Index 10 out of range
        assert!(embedding.get_embedding(5).is_err()); // Index 5 out of range
    }
}