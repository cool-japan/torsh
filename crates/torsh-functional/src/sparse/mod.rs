//! Sparse tensor operations for ToRSh functional API
//!
//! This module provides a comprehensive set of operations for sparse tensors,
//! organized into focused sub-modules for better maintainability and discoverability.
//!
//! ## Module Organization
//!
//! - [`core`] - Core SparseTensor struct and basic operations (creation, conversion, coalescing)
//! - [`arithmetic`] - Basic arithmetic operations (addition, scalar multiplication)
//! - [`linalg`] - Linear algebra operations (matrix multiplication, transpose, identity)
//! - [`mod@format`] - Format conversion utilities (COO ↔ CSR conversion)
//! - [`convolution`] - Convolution operations (1D and 2D sparse convolutions)
//! - [`reduction`] - Reduction operations (sum, mean, max, min)
//!
//! ## Usage Examples
//!
//! ```rust
//! use torsh_functional::sparse::{SparseTensor, sparse_coo_tensor, sparse_mm, sparse_sum};
//! use torsh_tensor::Tensor;
//!
//! // Create a sparse tensor
//! let values = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
//! let indices = Tensor::from_data(vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0], vec![2, 3], torsh_core::DeviceType::Cpu)?;
//! let sparse = sparse_coo_tensor(&indices, &values, &[3, 3])?;
//!
//! // Perform operations
//! let sum_result = sparse_sum(&sparse, None)?;  // Sum all elements
//! let dense = sparse.to_dense()?;              // Convert to dense
//! ```
//!
//! ## Performance Characteristics
//!
//! Sparse operations are optimized for tensors with many zero elements:
//! - **Memory efficiency**: Only stores non-zero elements
//! - **Computational efficiency**: Operations skip zero elements
//! - **Cache efficiency**: Better memory access patterns for sparse data
//!
//! ## Mathematical Background
//!
//! All operations properly handle the sparse representation while maintaining
//! mathematical equivalence with dense operations:
//! - Zero elements are implicitly handled in reductions and arithmetic
//! - Convolutions only process non-zero input elements
//! - Linear algebra operations leverage sparsity for performance

pub mod arithmetic;
pub mod convolution;
pub mod core;
pub mod format;
pub mod linalg;
pub mod reduction;

// Re-export core types and functions for convenient access
pub use core::{sparse_coo_tensor, SparseTensor};

// Re-export arithmetic operations
pub use arithmetic::{sparse_add, sparse_mul};

// Re-export linear algebra operations
pub use linalg::{sparse_eye, sparse_mm, sparse_transpose};

// Re-export format conversion utilities
pub use format::{csr_to_sparse, sparse_to_csr};

// Re-export convolution operations
pub use convolution::{sparse_conv1d, sparse_conv2d};

// Re-export reduction operations
pub use reduction::{sparse_max, sparse_mean, sparse_min, sparse_sum};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use torsh_tensor::Tensor;

    /// Test the complete sparse tensor workflow
    #[test]
    fn test_sparse_workflow_integration() -> torsh_core::Result<()> {
        // Create a sparse matrix [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
        let values = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5],
            torsh_core::DeviceType::Cpu,
        )?;
        let indices = Tensor::from_data(
            vec![
                0.0, 0.0, 1.0, 2.0, 2.0, // rows
                0.0, 2.0, 1.0, 0.0, 2.0,
            ], // cols
            vec![2, 5],
            torsh_core::DeviceType::Cpu,
        )?;
        let sparse_a = sparse_coo_tensor(&indices, &values, &[3, 3])?;

        // Test various operations

        // 1. Basic properties
        assert_eq!(sparse_a.nnz(), 5);
        assert_eq!(sparse_a.shape(), &[3, 3]);
        assert_eq!(sparse_a.ndim(), 2);

        // 2. Conversion to dense and back
        let dense = sparse_a.to_dense()?;
        let sparse_b = SparseTensor::from_dense(&dense)?;

        // Both should represent the same matrix
        // Sparse tensor round-trip conversion fixed (COO indexing bug resolved)
        let dense_a = sparse_a.to_dense()?.to_vec()?;
        let dense_b = sparse_b.to_dense()?.to_vec()?;

        // Validate round-trip conversion produces identical results
        for (a, b) in dense_a.iter().zip(dense_b.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Round-trip conversion mismatch: {} vs {}",
                a,
                b
            );
        }

        // 3. Arithmetic operations
        let scaled = sparse_mul(&sparse_a, 2.0)?;
        let scaled_values = scaled.values.to_vec()?;
        let original_values = sparse_a.values.to_vec()?;
        for (scaled, original) in scaled_values.iter().zip(original_values.iter()) {
            assert!((scaled - original * 2.0).abs() < 1e-6);
        }

        // 4. Linear algebra operations
        let identity = sparse_eye(3)?;
        let result = sparse_mm(&sparse_a, &identity.to_dense()?)?;
        let original_dense = sparse_a.to_dense()?.to_vec()?;
        let result_data = result.to_vec()?;
        for (original, result) in original_dense.iter().zip(result_data.iter()) {
            assert!((original - result).abs() < 1e-6);
        }

        // 5. Reduction operations
        let total_sum = sparse_sum(&sparse_a, None)?;
        let sum_value = total_sum.to_vec()?[0];
        let expected_sum = 1.0 + 2.0 + 3.0 + 4.0 + 5.0;
        assert!((sum_value - expected_sum).abs() < 1e-6);

        // 6. Format conversion
        let (csr_values, col_indices, row_ptrs) = sparse_to_csr(&sparse_a)?;
        let reconstructed = csr_to_sparse(&csr_values, &col_indices, &row_ptrs, &[3, 3])?;

        let original_dense = sparse_a.to_dense()?.to_vec()?;
        let reconstructed_dense = reconstructed.to_dense()?.to_vec()?;
        for (original, reconstructed) in original_dense.iter().zip(reconstructed_dense.iter()) {
            assert!((original - reconstructed).abs() < 1e-6);
        }

        Ok(())
    }

    /// Test sparse operations maintain sparsity patterns
    #[test]
    fn test_sparsity_preservation() -> torsh_core::Result<()> {
        // Create a very sparse matrix (only 2 non-zeros in a 10x10 matrix)
        let values = Tensor::from_data(vec![1.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 9.0, 0.0, 9.0], // corners of a 10x10 matrix
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let sparse = sparse_coo_tensor(&indices, &values, &[10, 10])?;

        // Verify operations maintain efficiency
        assert_eq!(sparse.nnz(), 2); // Very sparse

        // Scalar multiplication should preserve sparsity
        let scaled = sparse_mul(&sparse, 3.0)?;
        assert_eq!(scaled.nnz(), 2);

        // Transpose should preserve sparsity
        let transposed = sparse_transpose(&sparse)?;
        assert_eq!(transposed.nnz(), 2);

        // Sum should be efficient (only processes 2 elements)
        let sum = sparse_sum(&sparse, None)?;
        let sum_value = sum.to_vec()?[0];
        assert!((sum_value - 3.0).abs() < 1e-6); // 1.0 + 2.0 = 3.0

        Ok(())
    }
}
