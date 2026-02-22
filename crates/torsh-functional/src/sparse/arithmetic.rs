//! Sparse tensor arithmetic operations
//!
//! This module provides basic arithmetic operations for sparse tensors including
//! addition and scalar multiplication.

use crate::sparse::core::SparseTensor;
use torsh_core::{Result as TorshResult, TorshError};

/// Sparse tensor addition
///
/// Adds two sparse tensors element-wise. Both tensors must have the same shape.
/// The operation is performed by converting to dense, adding, and converting back
/// to sparse format to handle overlapping indices correctly.
pub fn sparse_add(a: &SparseTensor, b: &SparseTensor) -> TorshResult<SparseTensor> {
    if a.shape != b.shape {
        return Err(TorshError::invalid_argument_with_context(
            "Sparse tensors must have the same shape",
            "sparse_add",
        ));
    }

    // Convert both to dense, add, then convert back to sparse
    let a_dense = a.to_dense()?;
    let b_dense = b.to_dense()?;
    let result_dense = a_dense.add_op(&b_dense)?;

    SparseTensor::from_dense(&result_dense)
}

/// Sparse tensor element-wise scalar multiplication
///
/// Multiplies all non-zero elements of the sparse tensor by a scalar value.
/// This operation preserves the sparse structure and is very efficient.
pub fn sparse_mul(sparse: &SparseTensor, scalar: f32) -> TorshResult<SparseTensor> {
    let new_values = sparse.values.mul_scalar(scalar)?;

    Ok(SparseTensor {
        values: new_values,
        indices: sparse.indices.clone(),
        shape: sparse.shape.clone(),
        ndim: sparse.ndim,
        nnz: sparse.nnz,
        is_coalesced: sparse.is_coalesced,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::core::sparse_coo_tensor;
    use torsh_tensor::Tensor;

    #[test]
    fn test_sparse_add() -> TorshResult<()> {
        // Create first sparse tensor: [[1, 0], [0, 2]]
        let values_a = Tensor::from_data(vec![1.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let indices_a = Tensor::from_data(
            vec![0.0, 1.0, 0.0, 1.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let sparse_a = sparse_coo_tensor(&indices_a, &values_a, &[2, 2])?;

        // Create second sparse tensor: [[0, 3], [4, 0]]
        let values_b = Tensor::from_data(vec![3.0, 4.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let indices_b = Tensor::from_data(
            vec![0.0, 1.0, 1.0, 0.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let sparse_b = sparse_coo_tensor(&indices_b, &values_b, &[2, 2])?;

        // Add them using sparse_add
        let result = sparse_add(&sparse_a, &sparse_b)?;

        // Convert to dense to verify: [[1, 3], [4, 2]]
        let dense_result = result.to_dense()?;
        let result_data = dense_result.to_vec()?;
        let expected = vec![1.0, 3.0, 4.0, 2.0];

        // Verify the result matches expected
        for (i, (&actual, &expected)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                actual,
                expected
            );
        }

        Ok(())
    }

    #[test]
    fn test_sparse_mul() -> TorshResult<()> {
        let values = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;
        let sparse = sparse_coo_tensor(&indices, &values, &[3, 3])?;

        // Multiply by 2.5
        let result = sparse_mul(&sparse, 2.5)?;

        // Check that values are scaled correctly
        let result_values = result.values.to_vec()?;
        let expected_values = vec![2.5, 5.0, 7.5];

        for (actual, expected) in result_values.iter().zip(expected_values.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        // Check that structure is preserved
        assert_eq!(result.nnz, sparse.nnz);
        assert_eq!(result.shape, sparse.shape);
        assert_eq!(result.ndim, sparse.ndim);

        Ok(())
    }
}
