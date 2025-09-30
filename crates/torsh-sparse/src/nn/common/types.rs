//! Common types for sparse neural networks

use crate::{CooTensor, CscTensor, CsrTensor, SparseTensor, TorshResult};
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Sparse tensor format types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseFormat {
    /// Coordinate format (COO)
    Coo,
    /// Compressed Sparse Row (CSR)
    Csr,
    /// Compressed Sparse Column (CSC)
    Csc,
}

impl SparseFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            SparseFormat::Coo => "COO",
            SparseFormat::Csr => "CSR",
            SparseFormat::Csc => "CSC",
        }
    }
}

/// Sparse layer configuration
#[derive(Debug, Clone)]
pub struct SparseLayerConfig {
    /// Input sparsity level (fraction of zero elements)
    pub input_sparsity: f32,
    /// Target output sparsity level
    pub output_sparsity: Option<f32>,
    /// Whether to preserve input sparsity pattern
    pub preserve_sparsity: bool,
    /// Preferred sparse format for computations
    pub preferred_format: SparseFormat,
    /// Whether to use bias
    pub use_bias: bool,
}

impl Default for SparseLayerConfig {
    fn default() -> Self {
        Self {
            input_sparsity: 0.9,
            output_sparsity: None,
            preserve_sparsity: true,
            preferred_format: SparseFormat::Csr,
            use_bias: true,
        }
    }
}

impl SparseLayerConfig {
    /// Create configuration for sparse linear layer
    pub fn linear(sparsity: f32) -> Self {
        Self {
            input_sparsity: sparsity,
            output_sparsity: Some(sparsity * 0.8), // Slightly denser output
            preserve_sparsity: false,              // Linear layers can change sparsity
            preferred_format: SparseFormat::Csr,
            use_bias: true,
        }
    }

    /// Create configuration for sparse convolution
    pub fn convolution(sparsity: f32) -> Self {
        Self {
            input_sparsity: sparsity,
            output_sparsity: Some(sparsity * 0.9), // Preserve most sparsity
            preserve_sparsity: true,
            preferred_format: SparseFormat::Coo, // Better for convolutions
            use_bias: true,
        }
    }

    /// Create configuration for graph neural networks
    pub fn graph(sparsity: f32) -> Self {
        Self {
            input_sparsity: sparsity,
            output_sparsity: Some(sparsity),
            preserve_sparsity: true,
            preferred_format: SparseFormat::Csr, // Efficient for graph operations
            use_bias: false,                     // Often disabled in GNNs
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.input_sparsity) {
            return Err("Input sparsity must be between 0.0 and 1.0".to_string());
        }

        if let Some(output_sparsity) = self.output_sparsity {
            if !(0.0..=1.0).contains(&output_sparsity) {
                return Err("Output sparsity must be between 0.0 and 1.0".to_string());
            }
        }

        Ok(())
    }
}

/// Sparse initialization strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseInitStrategy {
    /// Random sparse weights with given sparsity
    Random,
    /// Structured sparsity (e.g., block-wise)
    Structured,
    /// Magnitude-based pruning of dense weights
    MagnitudePruning,
    /// Gradual sparsification during training
    Gradual,
}

/// Sparse weight initialization configuration
#[derive(Debug, Clone)]
pub struct SparseInitConfig {
    /// Initialization strategy
    pub strategy: SparseInitStrategy,
    /// Target sparsity level
    pub sparsity: f32,
    /// Standard deviation for weight initialization
    pub std: f32,
    /// Block size for structured sparsity
    pub block_size: Option<(usize, usize)>,
    /// Random seed for reproducible initialization
    pub seed: Option<u64>,
}

impl Default for SparseInitConfig {
    fn default() -> Self {
        Self {
            strategy: SparseInitStrategy::Random,
            sparsity: 0.9,
            std: 0.02,
            block_size: None,
            seed: None,
        }
    }
}

impl SparseInitConfig {
    /// Create config for random sparse initialization
    pub fn random(sparsity: f32, std: f32) -> Self {
        Self {
            strategy: SparseInitStrategy::Random,
            sparsity,
            std,
            ..Default::default()
        }
    }

    /// Create config for structured sparse initialization
    pub fn structured(sparsity: f32, block_size: (usize, usize)) -> Self {
        Self {
            strategy: SparseInitStrategy::Structured,
            sparsity,
            block_size: Some(block_size),
            ..Default::default()
        }
    }

    /// Create config for magnitude-based pruning
    pub fn magnitude_pruning(sparsity: f32) -> Self {
        Self {
            strategy: SparseInitStrategy::MagnitudePruning,
            sparsity,
            ..Default::default()
        }
    }
}

/// Common sparse tensor operations
pub struct SparseOps;

impl SparseOps {
    /// Calculate sparsity level of a tensor
    pub fn calculate_sparsity<T: SparseTensor>(tensor: &T) -> f32 {
        let total_elements = tensor.shape().numel();
        let nnz = tensor.nnz();
        1.0 - (nnz as f32 / total_elements as f32)
    }

    /// Check if sparsity level meets threshold
    pub fn is_sparse_enough<T: SparseTensor>(tensor: &T, threshold: f32) -> bool {
        Self::calculate_sparsity(tensor) >= threshold
    }

    /// Get memory footprint reduction from sparsity
    pub fn memory_reduction<T: SparseTensor>(tensor: &T) -> f32 {
        let dense_size = tensor.shape().numel() * std::mem::size_of::<f32>();
        let sparse_size =
            tensor.nnz() * (std::mem::size_of::<f32>() + std::mem::size_of::<usize>());
        1.0 - (sparse_size as f32 / dense_size as f32)
    }

    /// Estimate FLOPs reduction from sparsity
    pub fn flops_reduction<T: SparseTensor>(tensor: &T) -> f32 {
        Self::calculate_sparsity(tensor)
    }
}

/// Statistics for sparse operations
#[derive(Debug, Clone)]
pub struct SparseStats {
    /// Input sparsity
    pub input_sparsity: f32,
    /// Output sparsity
    pub output_sparsity: f32,
    /// Memory reduction factor
    pub memory_reduction: f32,
    /// Estimated FLOPs reduction
    pub flops_reduction: f32,
    /// Number of non-zero elements processed
    pub nnz_processed: usize,
}

impl SparseStats {
    pub fn new() -> Self {
        Self {
            input_sparsity: 0.0,
            output_sparsity: 0.0,
            memory_reduction: 0.0,
            flops_reduction: 0.0,
            nnz_processed: 0,
        }
    }

    /// Update statistics with new tensor
    pub fn update<T: SparseTensor>(&mut self, tensor: &T, is_input: bool) {
        let sparsity = SparseOps::calculate_sparsity(tensor);

        if is_input {
            self.input_sparsity = sparsity;
        } else {
            self.output_sparsity = sparsity;
        }

        self.memory_reduction = SparseOps::memory_reduction(tensor);
        self.flops_reduction = SparseOps::flops_reduction(tensor);
        self.nnz_processed += tensor.nnz();
    }

    /// Get efficiency ratio (higher is better)
    pub fn efficiency_ratio(&self) -> f32 {
        (self.memory_reduction + self.flops_reduction) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_layer_config() {
        let config = SparseLayerConfig::default();
        assert!(config.validate().is_ok());

        let config = SparseLayerConfig::linear(0.8);
        assert_eq!(config.input_sparsity, 0.8);
        approx::assert_relative_eq!(config.output_sparsity.unwrap(), 0.64, epsilon = 1e-6); // 0.8 * 0.8

        let config = SparseLayerConfig::graph(0.9);
        assert!(!config.use_bias);
        assert_eq!(config.preferred_format, SparseFormat::Csr);
    }

    #[test]
    fn test_sparse_init_config() {
        let config = SparseInitConfig::random(0.9, 0.01);
        assert_eq!(config.strategy, SparseInitStrategy::Random);
        assert_eq!(config.sparsity, 0.9);
        assert_eq!(config.std, 0.01);

        let config = SparseInitConfig::structured(0.8, (4, 4));
        assert_eq!(config.strategy, SparseInitStrategy::Structured);
        assert_eq!(config.block_size, Some((4, 4)));
    }

    #[test]
    fn test_sparse_format() {
        assert_eq!(SparseFormat::Csr.as_str(), "CSR");
        assert_eq!(SparseFormat::Coo.as_str(), "COO");
        assert_eq!(SparseFormat::Csc.as_str(), "CSC");
    }

    #[test]
    fn test_sparse_stats() {
        let mut stats = SparseStats::new();

        // Test efficiency ratio calculation
        stats.memory_reduction = 0.8;
        stats.flops_reduction = 0.9;
        assert_eq!(stats.efficiency_ratio(), 0.85);
    }
}
