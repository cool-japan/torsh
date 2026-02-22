//! Core sparse tensor implementation
//!
//! This module provides the SparseTensor struct and basic operations for sparse tensors
//! using COO (coordinate) format.

use std::collections::HashMap;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Sparse tensor representation using COO (coordinate) format
#[derive(Debug, Clone)]
pub struct SparseTensor {
    /// Tensor values (non-zero elements)
    pub values: Tensor,
    /// Indices of non-zero elements [2, nnz] for 2D, [3, nnz] for 3D, etc.
    pub indices: Tensor,
    /// Shape of the full tensor
    pub shape: Vec<usize>,
    /// Number of dimensions
    pub ndim: usize,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Whether the tensor is coalesced (indices are ordered and unique)
    pub is_coalesced: bool,
}

impl SparseTensor {
    /// Create a new sparse tensor from values and indices
    pub fn new(values: Tensor, indices: Tensor, shape: Vec<usize>) -> TorshResult<Self> {
        let values_shape = values.shape().dims().to_vec();
        let indices_shape = indices.shape().dims().to_vec();

        if values_shape.len() != 1 {
            return Err(TorshError::invalid_argument_with_context(
                "Values must be a 1D tensor",
                "SparseTensor::new",
            ));
        }

        if indices_shape.len() != 2 {
            return Err(TorshError::invalid_argument_with_context(
                "Indices must be a 2D tensor",
                "SparseTensor::new",
            ));
        }

        let nnz = values_shape[0];
        let ndim = shape.len();

        if indices_shape[0] != ndim {
            return Err(TorshError::invalid_argument_with_context(
                "Indices first dimension must equal tensor ndim",
                "SparseTensor::new",
            ));
        }

        if indices_shape[1] != nnz {
            return Err(TorshError::invalid_argument_with_context(
                "Indices second dimension must equal number of values",
                "SparseTensor::new",
            ));
        }

        Ok(SparseTensor {
            values,
            indices,
            shape,
            ndim,
            nnz,
            is_coalesced: false,
        })
    }

    /// Create a sparse tensor from dense tensor
    pub fn from_dense(dense: &Tensor) -> TorshResult<Self> {
        let shape = dense.shape().dims().to_vec();
        let ndim = shape.len();

        // Find non-zero elements
        let dense_data = dense.to_vec()?;
        let mut values_vec = Vec::new();
        let mut coords_vec = Vec::new(); // Store all coordinates temporarily

        // Iterate through all elements
        let total_elements: usize = shape.iter().product();
        for flat_idx in 0..total_elements {
            let value = dense_data[flat_idx];
            if value.abs() > 1e-8 {
                // Consider as non-zero
                values_vec.push(value);

                // Convert flat index to multi-dimensional indices
                let mut remaining = flat_idx;
                let mut coords = Vec::with_capacity(ndim);
                for &dim_size in shape.iter().rev() {
                    coords.push(remaining % dim_size);
                    remaining /= dim_size;
                }
                coords.reverse();

                coords_vec.push(coords);
            }
        }

        let nnz = values_vec.len();

        // Build indices in dimension-by-dimension order (not element-by-element)
        // Indices should be [ndim, nnz] where row i contains all coords for dimension i
        let mut indices_vec = Vec::with_capacity(ndim * nnz);
        for dim in 0..ndim {
            for coords in &coords_vec {
                indices_vec.push(coords[dim] as f32);
            }
        }

        let values = Tensor::from_data(values_vec, vec![nnz], dense.device())?;
        let indices = Tensor::from_data(indices_vec, vec![ndim, nnz], dense.device())?;

        Ok(SparseTensor {
            values,
            indices,
            shape,
            ndim,
            nnz,
            is_coalesced: false,
        })
    }

    /// Convert sparse tensor to dense tensor
    pub fn to_dense(&self) -> TorshResult<Tensor> {
        let total_elements: usize = self.shape.iter().product();
        let mut dense_data = vec![0.0f32; total_elements];

        let values_data = self.values.to_vec()?;
        let indices_data = self.indices.to_vec()?;

        for i in 0..self.nnz {
            // Extract coordinates for this non-zero element
            let mut flat_idx = 0;
            let mut stride = 1;

            for j in (0..self.ndim).rev() {
                let coord = indices_data[j * self.nnz + i] as usize;
                flat_idx += coord * stride;
                stride *= self.shape[j];
            }

            dense_data[flat_idx] = values_data[i];
        }

        Tensor::from_data(dense_data, self.shape.clone(), self.values.device())
    }

    /// Coalesce the sparse tensor (combine duplicate indices)
    pub fn coalesce(&mut self) -> TorshResult<()> {
        if self.is_coalesced {
            return Ok(());
        }

        let values_data = self.values.to_vec()?;
        let indices_data = self.indices.to_vec()?;

        // Group by indices and sum values
        let mut index_to_value: HashMap<Vec<usize>, f32> = HashMap::new();

        for i in 0..self.nnz {
            let mut coords = Vec::with_capacity(self.ndim);
            for j in 0..self.ndim {
                coords.push(indices_data[j * self.nnz + i] as usize);
            }

            *index_to_value.entry(coords).or_insert(0.0) += values_data[i];
        }

        // Filter out zero values and create new arrays
        let mut new_values = Vec::new();
        let mut new_indices = Vec::new();

        for (coords, value) in index_to_value {
            if value.abs() > 1e-8 {
                new_values.push(value);
                for coord in coords {
                    new_indices.push(coord as f32);
                }
            }
        }

        let new_nnz = new_values.len();
        self.values = Tensor::from_data(new_values, vec![new_nnz], self.values.device())?;
        self.indices =
            Tensor::from_data(new_indices, vec![self.ndim, new_nnz], self.indices.device())?;
        self.nnz = new_nnz;
        self.is_coalesced = true;

        Ok(())
    }

    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Check if the tensor is coalesced
    pub fn is_coalesced(&self) -> bool {
        self.is_coalesced
    }
}

/// Create a sparse tensor from COO format
pub fn sparse_coo_tensor(
    indices: &Tensor,
    values: &Tensor,
    shape: &[usize],
) -> TorshResult<SparseTensor> {
    SparseTensor::new(values.clone(), indices.clone(), shape.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_tensor_creation() -> TorshResult<()> {
        let values = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;
        let shape = vec![3, 3];

        let sparse = SparseTensor::new(values, indices, shape)?;
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.shape(), &[3, 3]);
        assert_eq!(sparse.ndim(), 2);

        Ok(())
    }

    #[test]
    fn test_sparse_to_dense() -> TorshResult<()> {
        let values = Tensor::from_data(vec![1.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 0.0, 1.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let shape = vec![2, 2];

        let sparse = SparseTensor::new(values, indices, shape)?;
        let dense = sparse.to_dense()?;

        let expected_data = vec![1.0, 0.0, 0.0, 2.0];
        let dense_data = dense.to_vec()?;

        for (actual, expected) in dense_data.iter().zip(expected_data.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }

        Ok(())
    }
}
