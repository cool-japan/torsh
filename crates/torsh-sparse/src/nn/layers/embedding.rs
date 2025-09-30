//! Sparse embedding layer implementation

use super::super::common::{
    traits::{SparseConverter, SparseLayer},
    types::{SparseLayerConfig, SparseStats},
    utils::SparseWeightGenerator,
};
use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};
use scirs2_core::random::{Random, Rng};
use std::collections::HashSet;
use torsh_core::{Shape, TorshError};
use torsh_tensor::{creation::zeros, Tensor};

/// Sparse embedding layer
///
/// This layer implements an embedding lookup table using sparse matrices,
/// which can significantly reduce memory usage for large vocabularies
/// with sparse word distributions.
pub struct SparseEmbedding {
    /// Embedding weight matrix (vocab_size x embedding_dim) in sparse format
    weight: CsrTensor,
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embedding_dim: usize,
    /// Sparsity level (fraction of zero weights)
    sparsity: f32,
    /// Training mode flag
    training: bool,
    /// Layer configuration
    config: SparseLayerConfig,
}

impl SparseEmbedding {
    /// Create a new sparse embedding layer
    pub fn new(vocab_size: usize, embedding_dim: usize, sparsity: f32) -> TorshResult<Self> {
        let config = SparseLayerConfig::default();
        Self::with_config(vocab_size, embedding_dim, sparsity, config)
    }

    /// Create sparse embedding layer with configuration
    pub fn with_config(
        vocab_size: usize,
        embedding_dim: usize,
        sparsity: f32,
        config: SparseLayerConfig,
    ) -> TorshResult<Self> {
        config
            .validate()
            .map_err(|e| TorshError::InvalidArgument(e))?;

        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TorshError::InvalidArgument(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        if vocab_size == 0 || embedding_dim == 0 {
            return Err(TorshError::InvalidArgument(
                "Vocabulary size and embedding dimension must be positive".to_string(),
            ));
        }

        // Generate sparse embedding matrix
        let weight = Self::generate_sparse_embeddings(vocab_size, embedding_dim, sparsity)?;

        Ok(Self {
            weight,
            vocab_size,
            embedding_dim,
            sparsity,
            training: true,
            config,
        })
    }

    /// Create from existing sparse weight matrix
    pub fn from_weight(weight: CsrTensor) -> TorshResult<Self> {
        let shape = weight.shape();
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Weight matrix must be 2D".to_string(),
            ));
        }

        let vocab_size = shape.dims()[0];
        let embedding_dim = shape.dims()[1];

        let total_elements = vocab_size * embedding_dim;
        let nnz = weight.nnz();
        let sparsity = 1.0 - (nnz as f32 / total_elements as f32);

        let config = SparseLayerConfig::default();

        Ok(Self {
            weight,
            vocab_size,
            embedding_dim,
            sparsity,
            training: true,
            config,
        })
    }

    /// Forward pass - lookup embeddings for given indices
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
        let mut output = zeros::<f32>(&[batch_size, self.embedding_dim])?;

        // Look up each embedding
        for (batch_idx, &vocab_idx) in indices.iter().enumerate() {
            let row_data = self.weight.get_row_data(vocab_idx)?;

            // Set the embedding values for non-zero entries
            for (col_idx, value) in row_data
                .0
                .iter()
                .zip(row_data.1.iter())
                .map(|(&idx, &val)| (idx, val))
            {
                output.set_item(&[batch_idx, col_idx], value)?;
            }
        }

        Ok(output)
    }

    /// Forward pass with tensor input
    pub fn forward_tensor(&self, indices: &Tensor) -> TorshResult<Tensor> {
        let input_shape = indices.shape();

        if input_shape.ndim() == 1 {
            // 1D indices: [seq_len] -> [seq_len, embedding_dim]
            let seq_len = input_shape.dims()[0];
            let mut indices_vec = Vec::with_capacity(seq_len);

            for i in 0..seq_len {
                let idx = indices.get_item(&[i])? as usize;
                indices_vec.push(idx);
            }

            self.forward(&indices_vec)
        } else if input_shape.ndim() == 2 {
            // 2D indices: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
            let batch_size = input_shape.dims()[0];
            let seq_len = input_shape.dims()[1];
            let mut output = zeros::<f32>(&[batch_size, seq_len, self.embedding_dim])?;

            for batch_idx in 0..batch_size {
                for seq_idx in 0..seq_len {
                    let vocab_idx = indices.get_item(&[batch_idx, seq_idx])? as usize;

                    if vocab_idx >= self.vocab_size {
                        return Err(TorshError::InvalidArgument(format!(
                            "Index {} out of vocabulary range [0, {})",
                            vocab_idx, self.vocab_size
                        )));
                    }

                    let row_data = self.weight.get_row_data(vocab_idx)?;

                    // Set the embedding values
                    for (col_idx, value) in row_data
                        .0
                        .iter()
                        .zip(row_data.1.iter())
                        .map(|(&idx, &val)| (idx, val))
                    {
                        output.set_item(&[batch_idx, seq_idx, col_idx], value)?;
                    }
                }
            }

            Ok(output)
        } else {
            Err(TorshError::InvalidArgument(
                "Input indices must be 1D or 2D tensor".to_string(),
            ))
        }
    }

    /// Get embedding for a single index
    pub fn get_embedding(&self, index: usize) -> TorshResult<Tensor> {
        if index >= self.vocab_size {
            return Err(TorshError::InvalidArgument(format!(
                "Index {} out of vocabulary range [0, {})",
                index, self.vocab_size
            )));
        }

        let mut embedding = zeros::<f32>(&[self.embedding_dim])?;
        let row_data = self.weight.get_row_data(index)?;

        for (col_idx, value) in row_data
            .0
            .iter()
            .zip(row_data.1.iter())
            .map(|(&idx, &val)| (idx, val))
        {
            embedding.set_item(&[col_idx], value)?;
        }

        Ok(embedding)
    }

    /// Get weight matrix
    pub fn weight(&self) -> &CsrTensor {
        &self.weight
    }

    /// Get mutable weight matrix
    pub fn weight_mut(&mut self) -> &mut CsrTensor {
        &mut self.weight
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get current sparsity level
    pub fn sparsity(&self) -> f32 {
        self.sparsity
    }

    /// Get number of parameters (non-zero weights)
    pub fn num_parameters(&self) -> usize {
        self.weight.nnz()
    }

    /// Update sparsity level by pruning weights
    pub fn prune_to_sparsity(&mut self, target_sparsity: f32) -> TorshResult<()> {
        if !(0.0..=1.0).contains(&target_sparsity) {
            return Err(TorshError::InvalidArgument(
                "Target sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        if target_sparsity <= self.sparsity {
            return Ok(()); // Already sparse enough
        }

        // Convert to dense, prune, convert back
        let dense_weight = self.weight.to_dense()?;
        self.weight = SparseWeightGenerator::prune_by_magnitude(&dense_weight, target_sparsity)?;
        self.sparsity = target_sparsity;

        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &SparseLayerConfig {
        &self.config
    }

    /// Calculate output dimensions for given input dimensions
    pub fn output_dimensions(&self, input_dims: &[usize]) -> Vec<usize> {
        match input_dims.len() {
            1 => vec![input_dims[0], self.embedding_dim], // [seq_len] -> [seq_len, embedding_dim]
            2 => vec![input_dims[0], input_dims[1], self.embedding_dim], // [batch, seq] -> [batch, seq, embedding_dim]
            _ => vec![],                                                 // Invalid input
        }
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
        let mut positions = HashSet::new();
        let mut rng = scirs2_core::random::thread_rng();

        while positions.len() < nnz {
            let row = rng.gen_range(0..vocab_size);
            let col = rng.gen_range(0..embedding_dim);
            positions.insert((row, col));
        }

        // Convert to triplet format with Xavier initialization
        let std = (2.0 / (vocab_size + embedding_dim) as f32).sqrt();

        for (row, col) in positions {
            row_indices.push(row);
            col_indices.push(col);
            // Xavier normal initialization
            let value: f32 = rng.gen_range(-1.0..1.0) * std;
            values.push(value);
        }

        CsrTensor::from_triplets(
            row_indices,
            col_indices,
            values,
            [vocab_size, embedding_dim],
        )
    }
}

impl SparseLayer for SparseEmbedding {
    fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert sparse input to dense for embedding lookup
        let dense_input = input.to_dense()?;

        // Extract indices from dense tensor
        let shape = dense_input.shape();
        let mut indices = Vec::new();

        if shape.ndim() == 1 {
            for i in 0..shape.dims()[0] {
                indices.push(dense_input.get_item(&[i])? as usize);
            }
        } else {
            return Err(TorshError::InvalidArgument(
                "Sparse embedding input must be 1D indices".to_string(),
            ));
        }

        let output = self.forward(&indices)?;

        // Convert output back to sparse if needed
        let sparse_output = SparseWeightGenerator::dense_to_sparse(&output)?;
        Ok(Box::new(sparse_output))
    }

    fn parameters(&self) -> Vec<&CsrTensor> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut CsrTensor> {
        vec![&mut self.weight]
    }

    fn layer_type(&self) -> &'static str {
        "SparseEmbedding"
    }

    fn dimensions(&self) -> (Vec<usize>, Vec<usize>) {
        let input_dims = vec![self.vocab_size];
        let output_dims = vec![self.embedding_dim];
        (input_dims, output_dims)
    }

    fn sparsity_stats(&self) -> SparseStats {
        let mut stats = SparseStats::new();
        stats.update(&self.weight, true);
        stats
    }

    fn train(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }
}

/// Memory usage statistics for sparse embedding layer
#[derive(Debug, Clone)]
pub struct SparseEmbeddingStats {
    /// Number of parameters in equivalent dense layer
    pub dense_parameters: usize,
    /// Number of non-zero parameters in sparse layer
    pub sparse_parameters: usize,
    /// Memory usage of equivalent dense layer (bytes)
    pub dense_memory_bytes: usize,
    /// Memory usage of sparse layer (bytes)
    pub sparse_memory_bytes: usize,
    /// Memory reduction factor (0.0 to 1.0)
    pub memory_reduction: f32,
    /// Vocabulary coverage (fraction of vocab entries with non-zero weights)
    pub vocabulary_coverage: f32,
}

impl SparseEmbedding {
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> SparseEmbeddingStats {
        let dense_params = self.vocab_size * self.embedding_dim;
        let sparse_params = self.weight.nnz();

        let dense_memory = dense_params * std::mem::size_of::<f32>();
        let sparse_memory =
            sparse_params * (std::mem::size_of::<f32>() + std::mem::size_of::<usize>());

        // Calculate vocabulary coverage
        let mut covered_vocab = std::collections::HashSet::new();
        for i in 0..self.vocab_size {
            if let Ok(row_data) = self.weight.get_row_data(i) {
                if !row_data.0.is_empty() && !row_data.1.is_empty() {
                    covered_vocab.insert(i);
                }
            }
        }
        let vocabulary_coverage = covered_vocab.len() as f32 / self.vocab_size as f32;

        SparseEmbeddingStats {
            dense_parameters: dense_params,
            sparse_parameters: sparse_params,
            dense_memory_bytes: dense_memory,
            sparse_memory_bytes: sparse_memory,
            memory_reduction: 1.0 - (sparse_memory as f32 / dense_memory as f32),
            vocabulary_coverage,
        }
    }
}

impl SparseEmbeddingStats {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.dense_parameters as f32 / self.sparse_parameters as f32
    }

    /// Get efficiency score
    pub fn efficiency_score(&self) -> f32 {
        (self.memory_reduction + self.vocabulary_coverage) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_embedding_creation() {
        let layer = SparseEmbedding::new(1000, 128, 0.9);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.vocab_size(), 1000);
        assert_eq!(layer.embedding_dim(), 128);
        assert_eq!(layer.sparsity(), 0.9);
    }

    #[test]
    fn test_sparse_embedding_dimensions() {
        let layer = SparseEmbedding::new(500, 64, 0.8).unwrap();

        let output_dims = layer.output_dimensions(&[10]);
        assert_eq!(output_dims, vec![10, 64]);

        let output_dims = layer.output_dimensions(&[5, 10]);
        assert_eq!(output_dims, vec![5, 10, 64]);
    }

    #[test]
    fn test_embedding_lookup() {
        let layer = SparseEmbedding::new(100, 50, 0.5).unwrap();

        let indices = vec![0, 1, 99];
        let result = layer.forward(&indices);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape().dims(), &[3, 50]);
    }

    #[test]
    fn test_single_embedding_lookup() {
        let layer = SparseEmbedding::new(100, 50, 0.5).unwrap();

        let result = layer.get_embedding(42);
        assert!(result.is_ok());

        let embedding = result.unwrap();
        assert_eq!(embedding.shape().dims(), &[50]);
    }

    #[test]
    fn test_invalid_indices() {
        let layer = SparseEmbedding::new(100, 50, 0.5).unwrap();

        let indices = vec![0, 1, 100]; // Index 100 is out of bounds
        let result = layer.forward(&indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_stats() {
        let layer = SparseEmbedding::new(1000, 128, 0.9).unwrap();
        let stats = layer.memory_stats();

        assert_eq!(stats.dense_parameters, 128000); // 1000 * 128
        assert!(stats.sparse_parameters < stats.dense_parameters);
        assert!(stats.memory_reduction > 0.0);
        assert!(stats.vocabulary_coverage > 0.0);
    }

    #[test]
    fn test_sparsity_validation() {
        let result = SparseEmbedding::new(100, 50, 1.5);
        assert!(result.is_err());

        let result = SparseEmbedding::new(100, 50, -0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_pruning() {
        let mut layer = SparseEmbedding::new(100, 50, 0.5).unwrap();
        let initial_sparsity = layer.sparsity();

        let result = layer.prune_to_sparsity(0.8);
        assert!(result.is_ok());
        assert!(layer.sparsity() > initial_sparsity);
    }
}
