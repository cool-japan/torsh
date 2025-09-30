//! Utility functions for sparse neural networks

use super::types::{SparseFormat, SparseInitConfig, SparseInitStrategy};
use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};
use scirs2_core::random::{Random, Rng};
use torsh_core::{Shape, TorshError};
use torsh_tensor::{creation::randn, Tensor};

/// Sparse weight generation utilities
pub struct SparseWeightGenerator;

impl SparseWeightGenerator {
    /// Generate sparse weight matrix with specified sparsity
    pub fn generate_sparse_weights(
        rows: usize,
        cols: usize,
        sparsity: f32,
    ) -> TorshResult<CsrTensor> {
        if !(0.0..=1.0).contains(&sparsity) {
            return Err(TorshError::InvalidArgument(
                "Sparsity must be between 0.0 and 1.0".to_string(),
            ));
        }

        let total_elements = rows * cols;
        let nnz = ((1.0 - sparsity) * total_elements as f32) as usize;

        let mut rng = scirs2_core::random::thread_rng();

        // Generate random indices for non-zero elements
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        // Use std for weight initialization
        let std = (2.0 / (rows + cols) as f32).sqrt();

        for _ in 0..nnz {
            let row = rng.gen_range(0..rows);
            let col = rng.gen_range(0..cols);
            let value: f32 = rng.gen_range(-1.0..1.0) * std;

            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
        }

        CsrTensor::from_triplets(row_indices, col_indices, values, [rows, cols])
    }

    /// Generate sparse weights from configuration
    pub fn from_config(
        rows: usize,
        cols: usize,
        config: &SparseInitConfig,
    ) -> TorshResult<CsrTensor> {
        match config.strategy {
            SparseInitStrategy::Random => Self::generate_random_sparse(rows, cols, config),
            SparseInitStrategy::Structured => Self::generate_structured_sparse(rows, cols, config),
            SparseInitStrategy::MagnitudePruning => {
                Self::generate_magnitude_pruned(rows, cols, config)
            }
            SparseInitStrategy::Gradual => {
                // Start dense, will be sparsified during training
                Self::generate_dense_for_gradual(rows, cols, config)
            }
        }
    }

    /// Generate random sparse weights
    fn generate_random_sparse(
        rows: usize,
        cols: usize,
        config: &SparseInitConfig,
    ) -> TorshResult<CsrTensor> {
        let total_elements = rows * cols;
        let nnz = ((1.0 - config.sparsity) * total_elements as f32) as usize;

        let mut rng = if let Some(seed) = config.seed {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for _ in 0..nnz {
            let row = rng.gen_range(0..rows);
            let col = rng.gen_range(0..cols);
            let value: f32 = rng.gen_range(-1.0..1.0) * config.std;

            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
        }

        CsrTensor::from_triplets(row_indices, col_indices, values, [rows, cols])
    }

    /// Generate structured sparse weights (block-wise sparsity)
    fn generate_structured_sparse(
        rows: usize,
        cols: usize,
        config: &SparseInitConfig,
    ) -> TorshResult<CsrTensor> {
        let (block_rows, block_cols) = config.block_size.unwrap_or((4, 4));

        if rows % block_rows != 0 || cols % block_cols != 0 {
            return Err(TorshError::InvalidArgument(
                "Matrix dimensions must be divisible by block size".to_string(),
            ));
        }

        let num_blocks_row = rows / block_rows;
        let num_blocks_col = cols / block_cols;
        let total_blocks = num_blocks_row * num_blocks_col;
        let active_blocks = ((1.0 - config.sparsity) * total_blocks as f32) as usize;

        let mut rng = if let Some(seed) = config.seed {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Randomly select which blocks to keep
        let mut active_block_indices = Vec::new();
        for _ in 0..active_blocks {
            let block_idx = rng.gen_range(0..total_blocks);
            if !active_block_indices.contains(&block_idx) {
                active_block_indices.push(block_idx);
            }
        }

        // Fill active blocks with values
        for &block_idx in &active_block_indices {
            let block_row = block_idx / num_blocks_col;
            let block_col = block_idx % num_blocks_col;

            for i in 0..block_rows {
                for j in 0..block_cols {
                    let row = block_row * block_rows + i;
                    let col = block_col * block_cols + j;
                    let value: f32 = rng.gen_range(-1.0..1.0) * config.std;

                    row_indices.push(row);
                    col_indices.push(col);
                    values.push(value);
                }
            }
        }

        CsrTensor::from_triplets(row_indices, col_indices, values, [rows, cols])
    }

    /// Generate weights using magnitude-based pruning
    fn generate_magnitude_pruned(
        rows: usize,
        cols: usize,
        config: &SparseInitConfig,
    ) -> TorshResult<CsrTensor> {
        // Generate dense weights first
        let dense_weights = randn::<f32>(&[rows, cols])?.mul_scalar(config.std)?;

        // Convert to sparse by pruning smallest magnitude weights
        Self::prune_by_magnitude(&dense_weights, config.sparsity)
    }

    /// Generate dense weights for gradual sparsification
    fn generate_dense_for_gradual(
        rows: usize,
        cols: usize,
        config: &SparseInitConfig,
    ) -> TorshResult<CsrTensor> {
        // Start with dense initialization
        let dense_weights = randn::<f32>(&[rows, cols])?.mul_scalar(config.std)?;

        // Convert to sparse format but keep all elements initially
        Self::dense_to_sparse(&dense_weights)
    }

    /// Convert dense tensor to sparse format
    pub fn dense_to_sparse(dense: &Tensor) -> TorshResult<CsrTensor> {
        let shape = dense.shape();
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Only 2D tensors supported".to_string(),
            ));
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Extract all non-zero elements
        for i in 0..rows {
            for j in 0..cols {
                let value = dense.get_item(&[i, j])?;
                if value.abs() > 1e-8 {
                    // Threshold for considering zero
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(value);
                }
            }
        }

        CsrTensor::from_triplets(row_indices, col_indices, values, [rows, cols])
    }

    /// Prune weights by magnitude to achieve target sparsity
    pub fn prune_by_magnitude(dense: &Tensor, target_sparsity: f32) -> TorshResult<CsrTensor> {
        let shape = dense.shape();
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Only 2D tensors supported".to_string(),
            ));
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let total_elements = rows * cols;
        let keep_elements = ((1.0 - target_sparsity) * total_elements as f32) as usize;

        // Get all weights with their positions
        let mut weight_positions = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                let value = dense.get_item(&[i, j])?;
                weight_positions.push((value.abs(), i, j, value));
            }
        }

        // Sort by magnitude (descending)
        weight_positions.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Keep only the largest magnitude weights
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for (_, row, col, value) in weight_positions.into_iter().take(keep_elements) {
            row_indices.push(row);
            col_indices.push(col);
            values.push(value);
        }

        CsrTensor::from_triplets(row_indices, col_indices, values, [rows, cols])
    }
}

/// Sparse tensor format conversion utilities
pub struct SparseConverter;

impl SparseConverter {
    /// Convert CSR to COO format
    pub fn csr_to_coo(csr: &CsrTensor) -> TorshResult<CooTensor> {
        // Implementation would depend on the actual tensor APIs
        // This is a placeholder showing the interface
        let shape = csr.shape();
        let nnz = csr.nnz();

        // Extract triplets and create COO tensor
        // Actual implementation would use csr.triplets() or similar
        CooTensor::new(vec![], vec![], vec![], shape.dims().into())
    }

    /// Convert COO to CSR format
    pub fn coo_to_csr(coo: &CooTensor) -> TorshResult<CsrTensor> {
        let shape = coo.shape();
        // Implementation would extract triplets and create CSR
        CsrTensor::from_triplets(vec![], vec![], vec![], [shape.dims()[0], shape.dims()[1]])
    }

    /// Get optimal format for operation type
    pub fn optimal_format_for_operation(operation: &str) -> super::types::SparseFormat {
        match operation {
            "matmul" | "linear" => super::types::SparseFormat::Csr,
            "conv" | "attention" => super::types::SparseFormat::Coo,
            "graph" => super::types::SparseFormat::Csr,
            _ => super::types::SparseFormat::Csr, // Default
        }
    }
}

/// Sparse tensor analysis utilities
pub struct SparseAnalyzer;

impl SparseAnalyzer {
    /// Analyze sparsity pattern distribution
    pub fn analyze_sparsity_pattern<T: SparseTensor>(tensor: &T) -> SparsePatternAnalysis {
        let shape = tensor.shape();
        let nnz = tensor.nnz();
        let total_elements = shape.numel();
        let sparsity = 1.0 - (nnz as f32 / total_elements as f32);

        SparsePatternAnalysis {
            sparsity,
            nnz,
            total_elements,
            density_per_row: Self::calculate_row_density(tensor),
            clustering_coefficient: Self::calculate_clustering(tensor),
        }
    }

    /// Calculate average density per row
    fn calculate_row_density<T: SparseTensor>(tensor: &T) -> f32 {
        let shape = tensor.shape();
        if shape.ndim() != 2 {
            return 0.0;
        }

        let rows = shape.dims()[0];
        let cols = shape.dims()[1];
        let nnz = tensor.nnz();

        (nnz as f32) / (rows as f32 * cols as f32)
    }

    /// Calculate clustering coefficient (measure of local density)
    fn calculate_clustering<T: SparseTensor>(_tensor: &T) -> f32 {
        // Simplified clustering measure
        // Real implementation would analyze local neighborhood density
        0.0
    }

    /// Recommend optimization strategies based on sparsity pattern
    pub fn recommend_optimizations(analysis: &SparsePatternAnalysis) -> Vec<String> {
        let mut recommendations = Vec::new();

        if analysis.sparsity > 0.95 {
            recommendations.push("Consider structured sparsity pruning".to_string());
        }

        if analysis.density_per_row < 0.1 {
            recommendations.push("Use CSR format for efficient row operations".to_string());
        }

        if analysis.clustering_coefficient > 0.5 {
            recommendations.push("Consider block-sparse representations".to_string());
        }

        recommendations
    }
}

/// Sparsity pattern analysis results
#[derive(Debug, Clone)]
pub struct SparsePatternAnalysis {
    /// Overall sparsity level
    pub sparsity: f32,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Total number of elements
    pub total_elements: usize,
    /// Average density per row
    pub density_per_row: f32,
    /// Clustering coefficient
    pub clustering_coefficient: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_weight_generation() {
        let result = SparseWeightGenerator::generate_sparse_weights(100, 50, 0.9);
        assert!(result.is_ok());

        let sparse_matrix = result.unwrap();
        let expected_nnz = ((1.0 - 0.9) * 100.0 * 50.0) as usize;
        // Allow for small variation in random generation (±1)
        let actual_nnz = sparse_matrix.nnz();
        assert!(
            actual_nnz >= expected_nnz - 1 && actual_nnz <= expected_nnz + 1,
            "Expected ~{}, got {}",
            expected_nnz,
            actual_nnz
        );
    }

    #[test]
    fn test_sparse_init_config() {
        let config = SparseInitConfig::random(0.8, 0.01);
        let result = SparseWeightGenerator::from_config(10, 10, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_recommendation() {
        assert_eq!(
            SparseConverter::optimal_format_for_operation("matmul"),
            SparseFormat::Csr
        );
        assert_eq!(
            SparseConverter::optimal_format_for_operation("conv"),
            SparseFormat::Coo
        );
    }
}
