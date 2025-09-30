//! Sparse tensor format conversion utilities
//!
//! This module provides utilities for converting between different sparse tensor formats
//! including COO (coordinate) and CSR (compressed sparse row) formats.

use crate::sparse::core::SparseTensor;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Convert sparse tensor to CSR (Compressed Sparse Row) format
///
/// CSR format stores a sparse matrix using three arrays:
/// - values: non-zero values in row-major order
/// - col_indices: column indices of non-zero values
/// - row_ptrs: pointers to the start of each row in the values array
///
/// This is a simplified CSR conversion suitable for basic operations.
/// For production use, this would need proper CSR format implementation
/// with sorted indices and optimized storage.
///
/// # Arguments
/// * `sparse` - The sparse tensor in COO format to convert
///
/// # Returns
/// A tuple of (values, col_indices, row_ptrs) tensors representing CSR format
pub fn sparse_to_csr(sparse: &SparseTensor) -> TorshResult<(Tensor, Tensor, Tensor)> {
    if sparse.ndim != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "CSR format only supports 2D tensors",
            "sparse_to_csr",
        ));
    }

    // This is a simplified CSR conversion
    // In practice, this would need proper CSR format implementation
    let values = sparse.values.clone();
    let indices_data = sparse.indices.to_vec()?;

    // Extract column indices
    let mut col_indices = Vec::with_capacity(sparse.nnz);
    for i in 0..sparse.nnz {
        col_indices.push(indices_data[sparse.nnz + i]);
    }

    // Create row pointers (simplified)
    let mut row_ptrs = vec![0.0f32; sparse.shape[0] + 1];
    let mut current_row = 0usize;
    let mut ptr = 0;

    for i in 0..sparse.nnz {
        let row = indices_data[i] as usize;
        while current_row <= row {
            row_ptrs[current_row] = ptr as f32;
            current_row += 1;
        }
        ptr += 1;
    }
    while current_row <= sparse.shape[0] {
        row_ptrs[current_row] = ptr as f32;
        current_row += 1;
    }

    let col_indices_tensor =
        Tensor::from_data(col_indices, vec![sparse.nnz], sparse.values.device())?;
    let row_ptrs_tensor =
        Tensor::from_data(row_ptrs, vec![sparse.shape[0] + 1], sparse.values.device())?;

    Ok((values, col_indices_tensor, row_ptrs_tensor))
}

/// Convert CSR format back to COO format SparseTensor
///
/// This function reconstructs a COO format SparseTensor from CSR components.
/// It's the inverse operation of sparse_to_csr.
///
/// # Arguments
/// * `values` - Non-zero values tensor
/// * `col_indices` - Column indices tensor
/// * `row_ptrs` - Row pointers tensor
/// * `shape` - Shape of the original matrix
///
/// # Returns
/// A SparseTensor in COO format
pub fn csr_to_sparse(
    values: &Tensor,
    col_indices: &Tensor,
    row_ptrs: &Tensor,
    shape: &[usize],
) -> TorshResult<SparseTensor> {
    if shape.len() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "CSR to COO conversion only supports 2D tensors",
            "csr_to_sparse",
        ));
    }

    let values_data = values.to_vec()?;
    let col_indices_data = col_indices.to_vec()?;
    let row_ptrs_data = row_ptrs.to_vec()?;

    let nnz = values_data.len();
    let mut row_indices = Vec::with_capacity(nnz);

    // Reconstruct row indices from row pointers
    for row in 0..shape[0] {
        let start = row_ptrs_data[row] as usize;
        let end = row_ptrs_data[row + 1] as usize;

        for _ in start..end {
            row_indices.push(row as f32);
        }
    }

    // Create indices tensor in COO format [2, nnz]
    let mut indices_data = Vec::with_capacity(2 * nnz);
    indices_data.extend_from_slice(&row_indices);
    indices_data.extend_from_slice(&col_indices_data);

    let indices_tensor = Tensor::from_data(indices_data, vec![2, nnz], values.device())?;

    SparseTensor::new(values.clone(), indices_tensor, shape.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::core::sparse_coo_tensor;

    #[test]
    fn test_sparse_to_csr() -> TorshResult<()> {
        // Create sparse matrix [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
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
        let sparse = sparse_coo_tensor(&indices, &values, &[3, 3])?;

        // Convert to CSR
        let (csr_values, col_indices, row_ptrs) = sparse_to_csr(&sparse)?;

        // Check values (should be the same)
        let csr_values_data = csr_values.to_vec()?;
        assert_eq!(csr_values_data.len(), 5);

        // Check column indices
        let col_indices_data = col_indices.to_vec()?;
        assert_eq!(col_indices_data.len(), 5);

        // Check row pointers (should have 4 elements for 3 rows)
        let row_ptrs_data = row_ptrs.to_vec()?;
        assert_eq!(row_ptrs_data.len(), 4);
        assert_eq!(row_ptrs_data[0], 0.0); // Start of row 0
        assert_eq!(row_ptrs_data[3], 5.0); // End pointer

        Ok(())
    }

    #[test]
    fn test_csr_to_sparse_roundtrip() -> TorshResult<()> {
        // Create original sparse matrix
        let values = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0], // diagonal matrix
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;
        let original_sparse = sparse_coo_tensor(&indices, &values, &[3, 3])?;

        // Convert to CSR and back
        let (csr_values, col_indices, row_ptrs) = sparse_to_csr(&original_sparse)?;
        let reconstructed_sparse = csr_to_sparse(&csr_values, &col_indices, &row_ptrs, &[3, 3])?;

        // Check that we get back the same matrix (after coalescing if needed)
        let original_dense = original_sparse.to_dense()?;
        let reconstructed_dense = reconstructed_sparse.to_dense()?;

        let original_data = original_dense.to_vec()?;
        let reconstructed_data = reconstructed_dense.to_vec()?;

        for (original, reconstructed) in original_data.iter().zip(reconstructed_data.iter()) {
            assert!((original - reconstructed).abs() < 1e-6);
        }

        Ok(())
    }
}
