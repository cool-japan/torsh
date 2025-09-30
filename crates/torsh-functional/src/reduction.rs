//! Reduction operations

use std::collections::HashMap;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{creation::zeros, stats::StatMode, Tensor};

// ============================================================================
// Basic Reduction Operations
// ============================================================================

/// Sum all elements in a tensor
pub fn sum(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.sum()
}

/// Sum along specified dimensions
pub fn sum_dim(tensor: &Tensor, dim: &[isize], keepdim: bool) -> TorshResult<Tensor> {
    let i32_dims: Vec<i32> = dim.iter().map(|&d| d as i32).collect();
    tensor.sum_dim(&i32_dims, keepdim)
}

/// Mean of all elements in a tensor
pub fn mean(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.mean(None, false)
}

/// Mean along specified dimensions
pub fn mean_dim(tensor: &Tensor, dim: &[isize], keepdim: bool) -> TorshResult<Tensor> {
    let usize_dims: Vec<usize> = dim.iter().map(|&d| d as usize).collect();
    tensor.mean(Some(&usize_dims), keepdim)
}

/// Maximum value in a tensor
pub fn max(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.max(None, false)
}

/// Maximum along specified dimension
pub fn max_dim(tensor: &Tensor, dim: isize, keepdim: bool) -> TorshResult<(Tensor, Tensor<i64>)> {
    let values = tensor.max_dim(dim as i32, keepdim)?;
    let indices = tensor.argmax(Some(dim as i32))?;
    Ok((values, indices))
}

/// Minimum value in a tensor
pub fn min(tensor: &Tensor) -> TorshResult<Tensor> {
    tensor.min()
}

/// Minimum along specified dimension
pub fn min_dim(tensor: &Tensor, dim: isize, keepdim: bool) -> TorshResult<(Tensor, Tensor<i64>)> {
    let values = tensor.min_dim(dim as i32, keepdim)?;
    let indices = tensor.argmin(Some(dim as i32))?;
    Ok((values, indices))
}

/// Index of maximum value in a tensor
pub fn argmax(tensor: &Tensor) -> TorshResult<Tensor<i64>> {
    tensor.argmax(None)
}

/// Index of maximum value along specified dimension
pub fn argmax_dim(tensor: &Tensor, dim: isize, keepdim: bool) -> TorshResult<Tensor<i64>> {
    let indices = tensor.argmax(Some(dim as i32))?;
    if keepdim {
        // Add dimension back for keepdim=true
        let mut new_shape = indices.shape().dims().to_vec();
        new_shape.insert(dim as usize, 1);
        let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
        indices.view(&new_shape_i32)
    } else {
        Ok(indices)
    }
}

/// Index of minimum value in a tensor
pub fn argmin(tensor: &Tensor) -> TorshResult<Tensor<i64>> {
    tensor.argmin(None)
}

/// Index of minimum value along specified dimension
pub fn argmin_dim(tensor: &Tensor, dim: isize, keepdim: bool) -> TorshResult<Tensor<i64>> {
    let indices = tensor.argmin(Some(dim as i32))?;
    if keepdim {
        // Add dimension back for keepdim=true
        let mut new_shape = indices.shape().dims().to_vec();
        new_shape.insert(dim as usize, 1);
        let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
        indices.view(&new_shape_i32)
    } else {
        Ok(indices)
    }
}

/// Product of all elements in a tensor
pub fn prod(tensor: &Tensor) -> TorshResult<Tensor> {
    // Flatten tensor and compute product manually since prod() might not be implemented
    let flat = tensor.flatten()?;
    let size = flat.shape().dims()[0];
    let mut result = 1.0f32;
    for i in 0..size {
        result *= flat.get(&[i])?;
    }
    Tensor::from_vec(vec![result], &[])
}

/// Product along specified dimensions
pub fn prod_dim(_tensor: &Tensor, _dim: &[isize], _keepdim: bool) -> TorshResult<Tensor> {
    // Simple implementation: return error for now as this requires complex dimension reduction
    let _ = (_dim, _keepdim); // silence unused warnings
    Err(TorshError::Other(
        "prod_dim not yet fully implemented".to_string(),
    ))
}

/// Standard deviation of all elements
pub fn std(tensor: &Tensor, unbiased: bool) -> TorshResult<Tensor> {
    let mode = if unbiased {
        StatMode::Sample
    } else {
        StatMode::Population
    };
    tensor.std(None, false, mode)
}

/// Standard deviation along specified dimensions
pub fn std_dim(
    tensor: &Tensor,
    dim: &[isize],
    unbiased: bool,
    keepdim: bool,
) -> TorshResult<Tensor> {
    let usize_dims: Vec<usize> = dim.iter().map(|&d| d as usize).collect();
    let mode = if unbiased {
        StatMode::Sample
    } else {
        StatMode::Population
    };
    tensor.std(Some(&usize_dims), keepdim, mode)
}

/// Variance of all elements
pub fn var(tensor: &Tensor, unbiased: bool) -> TorshResult<Tensor> {
    let mode = if unbiased {
        StatMode::Sample
    } else {
        StatMode::Population
    };
    tensor.var(None, false, mode)
}

/// Variance along specified dimensions
pub fn var_dim(
    tensor: &Tensor,
    dim: &[isize],
    unbiased: bool,
    keepdim: bool,
) -> TorshResult<Tensor> {
    let usize_dims: Vec<usize> = dim.iter().map(|&d| d as usize).collect();
    let mode = if unbiased {
        StatMode::Sample
    } else {
        StatMode::Population
    };
    tensor.var(Some(&usize_dims), keepdim, mode)
}

/// L1 norm (sum of absolute values)
pub fn norm_l1(tensor: &Tensor) -> TorshResult<Tensor> {
    let abs_tensor = tensor.abs()?;
    abs_tensor.sum()
}

/// L2 norm (Euclidean norm)
pub fn norm_l2(tensor: &Tensor) -> TorshResult<Tensor> {
    let squared = tensor.square()?;
    let sum = squared.sum()?;
    sum.sqrt()
}

/// P-norm
pub fn norm_p(tensor: &Tensor, p: f32) -> TorshResult<Tensor> {
    if p == 1.0 {
        norm_l1(tensor)
    } else if p == 2.0 {
        norm_l2(tensor)
    } else {
        let abs_tensor = tensor.abs()?;
        let powered = abs_tensor.pow_scalar(p)?;
        let sum = powered.sum()?;
        sum.pow_scalar(1.0 / p)
    }
}

/// Frobenius norm (for matrices)
pub fn norm_frobenius(tensor: &Tensor) -> TorshResult<Tensor> {
    norm_l2(tensor)
}

/// Nuclear norm (sum of singular values)
pub fn norm_nuclear(_tensor: &Tensor) -> TorshResult<Tensor> {
    // This would require SVD implementation
    // For now, return a simplified version
    Err(TorshError::Other(
        "Nuclear norm not yet implemented".to_string(),
    ))
}

/// Count non-zero elements
pub fn count_nonzero(tensor: &Tensor) -> TorshResult<Tensor> {
    let zero_tensor = zeros(tensor.shape().dims())?;
    let nonzero_mask = tensor.ne(&zero_tensor)?;
    // Create a tensor of ones with the same shape and sum where mask is true
    let ones = tensor.ones_like()?;
    let zeros = tensor.zeros_like()?;
    let count_tensor = ones.where_tensor(&nonzero_mask, &zeros)?;
    count_tensor.sum()
}

/// Count non-zero elements along dimension
pub fn count_nonzero_dim(tensor: &Tensor, dim: isize) -> TorshResult<Tensor> {
    let zero_tensor = zeros(tensor.shape().dims())?;
    let nonzero_mask = tensor.ne(&zero_tensor)?;
    // Create a tensor of ones with the same shape and sum where mask is true
    let ones = tensor.ones_like()?;
    let zeros = tensor.zeros_like()?;
    let count_tensor = ones.where_tensor(&nonzero_mask, &zeros)?;
    count_tensor.sum_dim(&[dim as i32], false)
}

/// Cumulative sum
pub fn cumsum(tensor: &Tensor, dim: isize) -> TorshResult<Tensor> {
    tensor.cumsum(dim.try_into().unwrap())
}

/// Cumulative product
pub fn cumprod(tensor: &Tensor, dim: isize) -> TorshResult<Tensor> {
    tensor.cumprod(dim.try_into().unwrap())
}

/// All elements are true (non-zero)
pub fn all(tensor: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.all()
}

/// All elements are true along dimension
pub fn all_dim(
    tensor: &Tensor,
    dim: isize,
    keepdim: bool,
) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.all_dim(dim.try_into().unwrap(), keepdim)
}

/// Any element is true (non-zero)
pub fn any(tensor: &Tensor) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.any()
}

/// Any element is true along dimension
pub fn any_dim(
    tensor: &Tensor,
    dim: isize,
    keepdim: bool,
) -> TorshResult<torsh_tensor::Tensor<bool>> {
    tensor.any_dim(dim.try_into().unwrap(), keepdim)
}

// ============================================================================
// Unique Operations
// ============================================================================

/// Find unique elements in a tensor
pub fn unique(
    tensor: &Tensor,
    sorted: bool,
    return_inverse: bool,
    return_counts: bool,
    dim: Option<isize>,
) -> TorshResult<UniqueResult> {
    if let Some(_d) = dim {
        // Unique along dimension
        return Err(TorshError::Other(
            "unique along dimension not yet implemented".to_string(),
        ));
    }

    // Flatten tensor and get unique values
    let flat = tensor.flatten()?;
    let size = flat.shape().dims()[0];

    let mut unique_map: HashMap<OrderedFloat, usize> = HashMap::new();
    let mut unique_values = Vec::new();
    let mut inverse_indices = vec![0; size];

    // Find unique values
    for i in 0..size {
        let value = flat.get(&[i])?;
        let key = OrderedFloat(value);

        match unique_map.get(&key) {
            Some(&idx) => {
                if return_inverse {
                    inverse_indices[i] = idx;
                }
            }
            None => {
                let idx = unique_values.len();
                unique_values.push(value);
                unique_map.insert(key, idx);
                if return_inverse {
                    inverse_indices[i] = idx;
                }
            }
        }
    }

    // Sort if requested
    if sorted {
        let mut indices: Vec<_> = (0..unique_values.len()).collect();
        indices.sort_by(|&a, &b| unique_values[a].partial_cmp(&unique_values[b]).unwrap());

        // Reorder unique values
        let sorted_values: Vec<_> = indices.iter().map(|&i| unique_values[i]).collect();

        // Update inverse indices if needed
        if return_inverse {
            let mut index_map = vec![0; indices.len()];
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                index_map[old_idx] = new_idx;
            }

            for inv_idx in inverse_indices.iter_mut() {
                *inv_idx = index_map[*inv_idx];
            }
        }

        unique_values = sorted_values;
    }

    // Create output tensor
    let output = Tensor::from_vec(unique_values.clone(), &[unique_values.len()])?;

    // Compute counts if requested
    let counts = if return_counts {
        let mut count_vec = vec![0; unique_values.len()];
        if return_inverse {
            for &idx in inverse_indices.iter() {
                count_vec[idx] += 1;
            }
        } else {
            // Recompute if we didn't track inverse indices
            for i in 0..size {
                let value = flat.get(&[i])?;
                for (j, &unique_val) in unique_values.iter().enumerate() {
                    if (value - unique_val).abs() < f32::EPSILON {
                        count_vec[j] += 1;
                        break;
                    }
                }
            }
        }
        let count_data: Vec<f32> = count_vec.into_iter().map(|c| c as f32).collect();
        Some(Tensor::from_vec(count_data.clone(), &[count_data.len()])?)
    } else {
        None
    };

    // Create inverse tensor if requested
    let inverse = if return_inverse {
        let inverse_data: Vec<f32> = inverse_indices.into_iter().map(|i| i as f32).collect();
        Some(Tensor::from_vec(
            inverse_data.clone(),
            &[inverse_data.len()],
        )?)
    } else {
        None
    };

    Ok(UniqueResult {
        values: output,
        inverse,
        counts,
    })
}

/// Find unique consecutive elements
pub fn unique_consecutive(
    tensor: &Tensor,
    return_inverse: bool,
    return_counts: bool,
    dim: Option<isize>,
) -> TorshResult<UniqueResult> {
    if let Some(_d) = dim {
        // Unique consecutive along dimension
        return Err(TorshError::Other(
            "unique_consecutive along dimension not yet implemented".to_string(),
        ));
    }

    // Flatten tensor
    let flat = tensor.flatten()?;
    let size = flat.shape().dims()[0];

    if size == 0 {
        return Ok(UniqueResult {
            values: zeros(&[0])?,
            inverse: if return_inverse {
                Some(zeros(&[0])?)
            } else {
                None
            },
            counts: if return_counts {
                Some(zeros(&[0])?)
            } else {
                None
            },
        });
    }

    let mut unique_values = Vec::new();
    let mut inverse_indices = vec![0; size];
    let mut counts = Vec::new();

    // Process first element
    let mut current_value = flat.get(&[0])?;
    unique_values.push(current_value);
    let mut current_count = 1;

    // Process remaining elements
    for i in 1..size {
        let value = flat.get(&[i])?;

        if (value - current_value).abs() < f32::EPSILON {
            // Same as previous
            current_count += 1;
            if return_inverse {
                inverse_indices[i] = unique_values.len() - 1;
            }
        } else {
            // Different from previous
            if return_counts {
                counts.push(current_count);
            }

            current_value = value;
            unique_values.push(value);
            current_count = 1;

            if return_inverse {
                inverse_indices[i] = unique_values.len() - 1;
            }
        }
    }

    // Don't forget the last group
    if return_counts {
        counts.push(current_count);
    }

    // Create output tensors
    let output = Tensor::from_vec(unique_values.clone(), &[unique_values.len()])?;

    let counts_tensor = if return_counts {
        let count_data: Vec<f32> = counts.into_iter().map(|c| c as f32).collect();
        Some(Tensor::from_vec(count_data.clone(), &[count_data.len()])?)
    } else {
        None
    };

    let inverse_tensor = if return_inverse {
        let inverse_data: Vec<f32> = inverse_indices.into_iter().map(|i| i as f32).collect();
        Some(Tensor::from_vec(
            inverse_data.clone(),
            &[inverse_data.len()],
        )?)
    } else {
        None
    };

    Ok(UniqueResult {
        values: output,
        inverse: inverse_tensor,
        counts: counts_tensor,
    })
}

/// Result of unique operations
pub struct UniqueResult {
    pub values: Tensor,
    pub inverse: Option<Tensor>,
    pub counts: Option<Tensor>,
}

/// Wrapper for f32 that implements Eq and Hash
#[derive(Debug, Clone, Copy)]
struct OrderedFloat(f32);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        (self.0 - other.0).abs() < f32::EPSILON
    }
}

impl Eq for OrderedFloat {}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the bits of the float
        self.0.to_bits().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::tensor;

    #[test]
    fn test_unique() {
        let tensor = tensor![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0].unwrap();

        // Test basic unique
        let result = unique(&tensor, true, false, false, None).unwrap();
        assert_eq!(result.values.shape().dims()[0], 7); // Should have 7 unique values

        // Test with counts
        let result = unique(&tensor, true, false, true, None).unwrap();
        assert!(result.counts.is_some());
    }

    #[test]
    fn test_unique_consecutive() {
        let tensor = tensor![1.0f32, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0].unwrap();

        let result = unique_consecutive(&tensor, true, true, None).unwrap();

        // Should have 4 groups: [1,1], [2,2,2], [3], [1,1]
        assert_eq!(result.values.shape().dims()[0], 4);

        let expected_values = vec![1.0, 2.0, 3.0, 1.0];
        let expected_counts = vec![2.0, 3.0, 1.0, 2.0];

        // Verify values and counts
        for i in 0..4 {
            assert!((result.values.get(&[i]).unwrap() - expected_values[i]).abs() < f32::EPSILON);
            assert!(
                (result.counts.as_ref().unwrap().get(&[i]).unwrap() - expected_counts[i]).abs()
                    < f32::EPSILON
            );
        }
    }
}
