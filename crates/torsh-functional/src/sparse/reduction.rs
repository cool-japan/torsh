//! Sparse tensor reduction operations
//!
//! This module provides efficient reduction operations for sparse tensors including
//! sum, mean, max, and min operations along specified dimensions.

use crate::sparse::core::SparseTensor;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Sparse tensor sum operation
///
/// Computes the sum of tensor elements over specified dimensions or all elements.
/// For sparse tensors, this is very efficient as it only processes non-zero elements.
///
/// # Mathematical Formula
/// - Global sum: result = Σ(all non-zero values)
/// - Dimension sum: `result[...] = Σ(values where coordinate[dim] varies)`
///
/// # Arguments
/// * `sparse` - The input sparse tensor
/// * `dim` - Optional dimension to sum over. If None, sums all elements.
///
/// # Returns
/// Dense tensor with the sum results
pub fn sparse_sum(sparse: &SparseTensor, dim: Option<usize>) -> TorshResult<Tensor> {
    match dim {
        None => {
            // Sum all elements
            let values = sparse.values.to_vec()?;
            let total_sum: f32 = values.iter().sum();
            Tensor::from_data(vec![total_sum], vec![1], sparse.values.device())
        }
        Some(d) => {
            if d >= sparse.ndim {
                return Err(TorshError::invalid_argument_with_context(
                    "Dimension out of range",
                    "sparse_sum",
                ));
            }

            let values = sparse.values.to_vec()?;
            let indices = sparse.indices.to_vec()?;
            let mut result_shape = sparse.shape.clone();
            result_shape.remove(d);

            if result_shape.is_empty() {
                result_shape.push(1);
            }

            let result_size: usize = result_shape.iter().product();
            let mut result_data = vec![0.0f32; result_size];

            for i in 0..sparse.nnz {
                let mut coords = Vec::new();
                for j in 0..sparse.ndim {
                    coords.push(indices[j * sparse.nnz + i] as usize);
                }

                // Remove the dimension we're summing over
                coords.remove(d);

                // Calculate flat index in result
                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..coords.len()).rev() {
                    flat_idx += coords[j] * stride;
                    stride *= result_shape[j];
                }

                result_data[flat_idx] += values[i];
            }

            Tensor::from_data(result_data, result_shape, sparse.values.device())
        }
    }
}

/// Sparse tensor mean operation
///
/// Computes the mean of tensor elements over specified dimensions or all elements.
/// For sparse tensors, this considers zero elements in the denominator for accurate means.
///
/// # Mathematical Formula
/// - Global mean: result = (Σ non-zero values) / (total_elements)
/// - Dimension mean: result[...] = (Σ values in dimension) / (dimension_size)
///
/// # Arguments
/// * `sparse` - The input sparse tensor
/// * `dim` - Optional dimension to compute mean over. If None, computes mean of all elements.
///
/// # Returns
/// Dense tensor with the mean results
pub fn sparse_mean(sparse: &SparseTensor, dim: Option<usize>) -> TorshResult<Tensor> {
    match dim {
        None => {
            // Mean of all elements (considering zeros)
            let values = sparse.values.to_vec()?;
            let total_sum: f32 = values.iter().sum();
            let total_elements: usize = sparse.shape.iter().product();
            let mean = total_sum / total_elements as f32;
            Tensor::from_data(vec![mean], vec![1], sparse.values.device())
        }
        Some(d) => {
            let sum_result = sparse_sum(sparse, Some(d))?;
            let dim_size = sparse.shape[d] as f32;
            sum_result.div_scalar(dim_size)
        }
    }
}

/// Sparse tensor maximum operation
///
/// Finds the maximum value over specified dimensions or all elements.
/// For sparse tensors, this considers zero as a potential maximum value.
///
/// # Mathematical Formula
/// result = max(max(non-zero values), 0.0)
///
/// # Arguments
/// * `sparse` - The input sparse tensor
/// * `dim` - Optional dimension to find max over. If None, finds global maximum.
///
/// # Returns
/// Dense tensor with the maximum values
pub fn sparse_max(sparse: &SparseTensor, dim: Option<usize>) -> TorshResult<Tensor> {
    match dim {
        None => {
            let values = sparse.values.to_vec()?;
            let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            // Consider zero as a potential maximum
            let final_max = max_val.max(0.0);
            Tensor::from_data(vec![final_max], vec![1], sparse.values.device())
        }
        Some(d) => {
            if d >= sparse.ndim {
                return Err(TorshError::invalid_argument_with_context(
                    "Dimension out of range",
                    "sparse_max",
                ));
            }

            let values = sparse.values.to_vec()?;
            let indices = sparse.indices.to_vec()?;
            let mut result_shape = sparse.shape.clone();
            result_shape.remove(d);

            if result_shape.is_empty() {
                result_shape.push(1);
            }

            let result_size: usize = result_shape.iter().product();
            let mut result_data = vec![f32::NEG_INFINITY; result_size];

            for i in 0..sparse.nnz {
                let mut coords = Vec::new();
                for j in 0..sparse.ndim {
                    coords.push(indices[j * sparse.nnz + i] as usize);
                }

                coords.remove(d);

                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..coords.len()).rev() {
                    flat_idx += coords[j] * stride;
                    stride *= result_shape[j];
                }

                result_data[flat_idx] = result_data[flat_idx].max(values[i]);
            }

            // Consider zero as potential maximum
            for val in &mut result_data {
                *val = val.max(0.0);
            }

            Tensor::from_data(result_data, result_shape, sparse.values.device())
        }
    }
}

/// Sparse tensor minimum operation
///
/// Finds the minimum value over specified dimensions or all elements.
/// For sparse tensors, this considers zero as a potential minimum value.
///
/// # Mathematical Formula
/// result = min(min(non-zero values), 0.0)
///
/// # Arguments
/// * `sparse` - The input sparse tensor
/// * `dim` - Optional dimension to find min over. If None, finds global minimum.
///
/// # Returns
/// Dense tensor with the minimum values
pub fn sparse_min(sparse: &SparseTensor, dim: Option<usize>) -> TorshResult<Tensor> {
    match dim {
        None => {
            let values = sparse.values.to_vec()?;
            let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            // Consider zero as a potential minimum
            let final_min = min_val.min(0.0);
            Tensor::from_data(vec![final_min], vec![1], sparse.values.device())
        }
        Some(d) => {
            if d >= sparse.ndim {
                return Err(TorshError::invalid_argument_with_context(
                    "Dimension out of range",
                    "sparse_min",
                ));
            }

            let values = sparse.values.to_vec()?;
            let indices = sparse.indices.to_vec()?;
            let mut result_shape = sparse.shape.clone();
            result_shape.remove(d);

            if result_shape.is_empty() {
                result_shape.push(1);
            }

            let result_size: usize = result_shape.iter().product();
            let mut result_data = vec![f32::INFINITY; result_size];

            for i in 0..sparse.nnz {
                let mut coords = Vec::new();
                for j in 0..sparse.ndim {
                    coords.push(indices[j * sparse.nnz + i] as usize);
                }

                coords.remove(d);

                let mut flat_idx = 0;
                let mut stride = 1;
                for j in (0..coords.len()).rev() {
                    flat_idx += coords[j] * stride;
                    stride *= result_shape[j];
                }

                result_data[flat_idx] = result_data[flat_idx].min(values[i]);
            }

            // Consider zero as potential minimum
            for val in &mut result_data {
                *val = val.min(0.0);
            }

            Tensor::from_data(result_data, result_shape, sparse.values.device())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::core::sparse_coo_tensor;

    #[test]
    fn test_sparse_sum() -> TorshResult<()> {
        let values = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;
        let shape = vec![3, 3];

        let sparse = sparse_coo_tensor(&indices, &values, &shape)?;

        // Test sum of all elements
        let total_sum = sparse_sum(&sparse, None)?;
        let sum_data = total_sum.to_vec()?;
        assert!((sum_data[0] - 6.0).abs() < 1e-6);

        // Test sum along dimension 0
        let sum_dim0 = sparse_sum(&sparse, Some(0))?;
        let sum_dim0_data = sum_dim0.to_vec()?;
        assert_eq!(sum_dim0_data.len(), 3);
        assert!((sum_dim0_data[0] - 1.0).abs() < 1e-6);
        assert!((sum_dim0_data[1] - 2.0).abs() < 1e-6);
        assert!((sum_dim0_data[2] - 3.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_sparse_mean() -> TorshResult<()> {
        let values = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;
        let shape = vec![3, 3];

        let sparse = sparse_coo_tensor(&indices, &values, &shape)?;

        // Test mean of all elements (6.0 / 9 = 0.6666...)
        let total_mean = sparse_mean(&sparse, None)?;
        let mean_data = total_mean.to_vec()?;
        assert!((mean_data[0] - (6.0 / 9.0)).abs() < 1e-6);

        // Test mean along dimension 0
        let mean_dim0 = sparse_mean(&sparse, Some(0))?;
        let mean_dim0_data = mean_dim0.to_vec()?;
        assert_eq!(mean_dim0_data.len(), 3);
        assert!((mean_dim0_data[0] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((mean_dim0_data[1] - (2.0 / 3.0)).abs() < 1e-6);
        assert!((mean_dim0_data[2] - (3.0 / 3.0)).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_sparse_max() -> TorshResult<()> {
        let values = Tensor::from_data(vec![1.0, -2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;
        let shape = vec![3, 3];

        let sparse = sparse_coo_tensor(&indices, &values, &shape)?;

        // Test max of all elements
        let total_max = sparse_max(&sparse, None)?;
        let max_data = total_max.to_vec()?;
        assert!((max_data[0] - 3.0).abs() < 1e-6);

        // Test max along dimension 0
        let max_dim0 = sparse_max(&sparse, Some(0))?;
        let max_dim0_data = max_dim0.to_vec()?;
        assert_eq!(max_dim0_data.len(), 3);
        assert!((max_dim0_data[0] - 1.0).abs() < 1e-6);
        assert!((max_dim0_data[1] - 0.0).abs() < 1e-6); // zero is greater than -2.0
        assert!((max_dim0_data[2] - 3.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_sparse_min() -> TorshResult<()> {
        let values = Tensor::from_data(vec![1.0, -2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;
        let shape = vec![3, 3];

        let sparse = sparse_coo_tensor(&indices, &values, &shape)?;

        // Test min of all elements
        let total_min = sparse_min(&sparse, None)?;
        let min_data = total_min.to_vec()?;
        assert!((min_data[0] - (-2.0)).abs() < 1e-6);

        // Test min along dimension 0
        let min_dim0 = sparse_min(&sparse, Some(0))?;
        let min_dim0_data = min_dim0.to_vec()?;
        assert_eq!(min_dim0_data.len(), 3);
        assert!((min_dim0_data[0] - 0.0).abs() < 1e-6); // zero is less than 1.0
        assert!((min_dim0_data[1] - (-2.0)).abs() < 1e-6);
        assert!((min_dim0_data[2] - 0.0).abs() < 1e-6); // zero is less than 3.0

        Ok(())
    }
}
