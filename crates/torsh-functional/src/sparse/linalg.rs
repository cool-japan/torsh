//! Sparse tensor linear algebra operations
//!
//! This module provides linear algebra operations for sparse tensors including
//! matrix multiplication, transpose, and identity matrix creation.

use crate::sparse::core::SparseTensor;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Sparse matrix multiplication (SpMM)
///
/// Performs matrix multiplication between a sparse matrix and a dense matrix.
/// The sparse matrix must be 2D and the dense matrix must be 2D with compatible dimensions.
///
/// # Mathematical Formula
/// For sparse matrix A (m×k) and dense matrix B (k×n):
/// C[i,j] = Σ(A[i,l] * B[l,j]) for all l where A[i,l] ≠ 0
///
/// # Arguments
/// * `sparse` - The sparse matrix (2D SparseTensor)
/// * `dense` - The dense matrix (2D Tensor)
///
/// # Returns
/// The result of the matrix multiplication as a dense Tensor
pub fn sparse_mm(sparse: &SparseTensor, dense: &Tensor) -> TorshResult<Tensor> {
    if sparse.ndim != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Sparse tensor must be 2D for matrix multiplication",
            "sparse_mm",
        ));
    }

    let dense_shape_binding = dense.shape();
    let dense_shape = dense_shape_binding.dims();
    if dense_shape.len() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Dense tensor must be 2D for matrix multiplication",
            "sparse_mm",
        ));
    }

    if sparse.shape[1] != dense_shape[0] {
        return Err(TorshError::invalid_argument_with_context(
            "Incompatible dimensions for matrix multiplication",
            "sparse_mm",
        ));
    }

    let result_shape = vec![sparse.shape[0], dense_shape[1]];
    let mut result_data = vec![0.0f32; result_shape.iter().product()];

    let sparse_values = sparse.values.to_vec()?;
    let sparse_indices = sparse.indices.to_vec()?;
    let dense_data = dense.to_vec()?;

    // Perform sparse matrix multiplication
    for i in 0..sparse.nnz {
        let row = sparse_indices[i] as usize;
        let col = sparse_indices[sparse.nnz + i] as usize;
        let value = sparse_values[i];

        for j in 0..dense_shape[1] {
            let dense_idx = col * dense_shape[1] + j;
            let result_idx = row * dense_shape[1] + j;
            result_data[result_idx] += value * dense_data[dense_idx];
        }
    }

    Tensor::from_data(result_data, result_shape, sparse.values.device())
}

/// Create a sparse identity matrix
///
/// Creates an identity matrix in sparse format with ones on the main diagonal.
/// This is very efficient for sparse representation since identity matrices
/// have exactly n non-zero elements.
///
/// # Arguments
/// * `size` - The size of the square identity matrix
///
/// # Returns
/// A sparse identity matrix of shape [size, size]
pub fn sparse_eye(size: usize) -> TorshResult<SparseTensor> {
    let values = vec![1.0f32; size];
    let mut indices = vec![0.0f32; 2 * size];

    for i in 0..size {
        indices[i] = i as f32; // row indices
        indices[size + i] = i as f32; // column indices
    }

    let values_tensor = Tensor::from_data(values, vec![size], torsh_core::DeviceType::Cpu)?;
    let indices_tensor = Tensor::from_data(indices, vec![2, size], torsh_core::DeviceType::Cpu)?;

    SparseTensor::new(values_tensor, indices_tensor, vec![size, size])
}

/// Sparse tensor transpose
///
/// Transposes a 2D sparse tensor by swapping row and column indices.
/// This operation is very efficient for sparse tensors as it only requires
/// swapping index arrays without touching the values.
///
/// # Arguments
/// * `sparse` - The 2D sparse tensor to transpose
///
/// # Returns
/// The transposed sparse tensor with swapped dimensions
pub fn sparse_transpose(sparse: &SparseTensor) -> TorshResult<SparseTensor> {
    if sparse.ndim != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Transpose currently only supports 2D sparse tensors",
            "sparse_transpose",
        ));
    }

    let indices_data = sparse.indices.to_vec()?;
    let mut new_indices = vec![0.0f32; 2 * sparse.nnz];

    // Swap row and column indices
    for i in 0..sparse.nnz {
        new_indices[i] = indices_data[sparse.nnz + i]; // old col -> new row
        new_indices[sparse.nnz + i] = indices_data[i]; // old row -> new col
    }

    let new_indices_tensor =
        Tensor::from_data(new_indices, vec![2, sparse.nnz], sparse.indices.device())?;
    let new_shape = vec![sparse.shape[1], sparse.shape[0]];

    Ok(SparseTensor {
        values: sparse.values.clone(),
        indices: new_indices_tensor,
        shape: new_shape,
        ndim: sparse.ndim,
        nnz: sparse.nnz,
        is_coalesced: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::core::sparse_coo_tensor;

    #[test]
    fn test_sparse_mm() -> TorshResult<()> {
        // Create sparse matrix [[1, 0], [0, 2]]
        let values = Tensor::from_data(vec![1.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 0.0, 1.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let sparse = sparse_coo_tensor(&indices, &values, &[2, 2])?;

        // Create dense matrix [[3, 4], [5, 6]]
        let dense = Tensor::from_data(
            vec![3.0, 4.0, 5.0, 6.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;

        // Multiply: [[1, 0], [0, 2]] × [[3, 4], [5, 6]] = [[3, 4], [10, 12]]
        let result = sparse_mm(&sparse, &dense)?;
        let result_data = result.to_vec()?;
        let expected = vec![3.0, 4.0, 10.0, 12.0];

        for (actual, expected) in result_data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_sparse_eye() -> TorshResult<()> {
        let sparse_eye = sparse_eye(3)?;
        assert_eq!(sparse_eye.nnz(), 3);
        assert_eq!(sparse_eye.shape(), &[3, 3]);

        let dense_eye = sparse_eye.to_dense()?;
        let dense_data = dense_eye.to_vec()?;
        let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        for (actual, expected) in dense_data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_sparse_transpose() -> TorshResult<()> {
        // Create sparse matrix [[1, 2], [0, 3]]
        let values = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0], // rows: [0, 0, 1], cols: [0, 1, 1]
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;
        let sparse = sparse_coo_tensor(&indices, &values, &[2, 2])?;

        // Transpose to get [[1, 0], [2, 3]]
        let transposed = sparse_transpose(&sparse)?;
        assert_eq!(transposed.shape(), &[2, 2]);

        let dense_transposed = transposed.to_dense()?;
        let transposed_data = dense_transposed.to_vec()?;
        let expected = vec![1.0, 0.0, 2.0, 3.0];

        for (actual, expected) in transposed_data.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        Ok(())
    }
}
