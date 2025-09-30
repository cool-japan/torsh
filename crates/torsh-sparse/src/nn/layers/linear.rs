//! Sparse linear layer implementation

use super::super::common::{
    traits::{SparseConverter, SparseLayer},
    types::{SparseLayerConfig, SparseStats},
    utils::SparseWeightGenerator,
};
use crate::{CsrTensor, SparseTensor, TorshResult};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};
use torsh_tensor::{creation::randn, Tensor};

/// Sparse linear layer (also known as sparse fully connected layer)
///
/// This layer implements a linear transformation using sparse weight matrices,
/// which can significantly reduce memory usage and computational cost for
/// networks with high sparsity levels.
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
    /// Training mode flag
    training: bool,
    /// Layer configuration
    config: SparseLayerConfig,
}

impl SparseLinear {
    /// Create a new sparse linear layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        sparsity: f32,
        use_bias: bool,
    ) -> TorshResult<Self> {
        let config = SparseLayerConfig::linear(sparsity);
        Self::with_config(in_features, out_features, config, use_bias)
    }

    /// Create sparse linear layer with configuration
    pub fn with_config(
        in_features: usize,
        out_features: usize,
        config: SparseLayerConfig,
        use_bias: bool,
    ) -> TorshResult<Self> {
        config
            .validate()
            .map_err(|e| TorshError::InvalidArgument(e))?;

        if !(0.0..=1.0).contains(&config.input_sparsity) {
            return Err(TorshError::InvalidArgument(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Generate sparse weight matrix
        let weight = SparseWeightGenerator::generate_sparse_weights(
            out_features,
            in_features,
            config.input_sparsity,
        )?;

        // Generate bias if requested
        let bias = if use_bias && config.use_bias {
            Some(randn::<f32>(&[out_features])?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            sparsity: config.input_sparsity,
            training: true,
            config,
        })
    }

    /// Create from existing sparse weight matrix
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

        let config = SparseLayerConfig::linear(sparsity);

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            sparsity,
            training: true,
            config,
        })
    }

    /// Forward pass
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

        // Perform sparse matrix-vector or matrix-matrix multiplication
        let output = if input_shape.ndim() == 1 {
            // Single sample: sparse matrix-vector multiplication
            self.weight.matvec(input)?
        } else {
            // Batch: sparse matrix-matrix multiplication
            self.weight.matmul(input)?
        };

        // Add bias if present
        if let Some(ref bias) = self.bias {
            output.add_bias(bias)
        } else {
            Ok(output)
        }
    }

    /// Get weight matrix
    pub fn weight(&self) -> &CsrTensor {
        &self.weight
    }

    /// Get mutable weight matrix
    pub fn weight_mut(&mut self) -> &mut CsrTensor {
        &mut self.weight
    }

    /// Get bias vector
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Get mutable bias vector
    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }

    /// Get input features count
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features count
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Get current sparsity level
    pub fn sparsity(&self) -> f32 {
        self.sparsity
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
            1 => vec![self.out_features],
            2 => vec![input_dims[0], self.out_features],
            _ => vec![], // Invalid input
        }
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> SparseMemoryStats {
        let dense_params = self.in_features * self.out_features;
        let sparse_params = self.weight.nnz();
        let bias_params = self.bias.as_ref().map_or(0, |b| b.numel());

        let dense_memory = dense_params * std::mem::size_of::<f32>();
        let sparse_memory = sparse_params
            * (std::mem::size_of::<f32>() + std::mem::size_of::<usize>())
            + bias_params * std::mem::size_of::<f32>();

        SparseMemoryStats {
            dense_parameters: dense_params,
            sparse_parameters: sparse_params,
            bias_parameters: bias_params,
            dense_memory_bytes: dense_memory,
            sparse_memory_bytes: sparse_memory,
            memory_reduction: 1.0 - (sparse_memory as f32 / dense_memory as f32),
        }
    }
}

impl SparseLayer for SparseLinear {
    fn forward(&self, input: &dyn SparseTensor) -> TorshResult<Box<dyn SparseTensor>> {
        // Convert sparse input to dense for linear layer
        let dense_input = input.to_dense()?;
        let output = self.forward(&dense_input)?;

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
        "SparseLinear"
    }

    fn dimensions(&self) -> (Vec<usize>, Vec<usize>) {
        let input_dims = vec![self.in_features];
        let output_dims = vec![self.out_features];
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

/// Memory usage statistics for sparse linear layer
#[derive(Debug, Clone)]
pub struct SparseMemoryStats {
    /// Number of parameters in equivalent dense layer
    pub dense_parameters: usize,
    /// Number of non-zero parameters in sparse layer
    pub sparse_parameters: usize,
    /// Number of bias parameters
    pub bias_parameters: usize,
    /// Memory usage of equivalent dense layer (bytes)
    pub dense_memory_bytes: usize,
    /// Memory usage of sparse layer (bytes)
    pub sparse_memory_bytes: usize,
    /// Memory reduction factor (0.0 to 1.0)
    pub memory_reduction: f32,
}

impl SparseMemoryStats {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.dense_parameters as f32 / self.sparse_parameters as f32
    }

    /// Get efficiency score
    pub fn efficiency_score(&self) -> f32 {
        self.memory_reduction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_linear_creation() {
        let layer = SparseLinear::new(100, 50, 0.9, true);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.in_features(), 100);
        assert_eq!(layer.out_features(), 50);
        assert_eq!(layer.sparsity(), 0.9);
        assert!(layer.bias().is_some());
    }

    #[test]
    fn test_sparse_linear_dimensions() {
        let layer = SparseLinear::new(784, 128, 0.8, false).unwrap();

        let output_dims = layer.output_dimensions(&[784]);
        assert_eq!(output_dims, vec![128]);

        let output_dims = layer.output_dimensions(&[32, 784]);
        assert_eq!(output_dims, vec![32, 128]);
    }

    #[test]
    fn test_memory_stats() {
        let layer = SparseLinear::new(100, 50, 0.9, true).unwrap();
        let stats = layer.memory_stats();

        assert_eq!(stats.dense_parameters, 5000); // 100 * 50
        assert!(stats.sparse_parameters < stats.dense_parameters);
        assert_eq!(stats.bias_parameters, 50);
        assert!(stats.memory_reduction > 0.0);
    }

    #[test]
    fn test_sparsity_validation() {
        let result = SparseLinear::new(10, 10, 1.5, true);
        assert!(result.is_err());

        let result = SparseLinear::new(10, 10, -0.1, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_pruning() {
        let mut layer = SparseLinear::new(10, 10, 0.5, false).unwrap();
        let initial_sparsity = layer.sparsity();

        let result = layer.prune_to_sparsity(0.8);
        assert!(result.is_ok());
        assert!(layer.sparsity() > initial_sparsity);
    }
}
