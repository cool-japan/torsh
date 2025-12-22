//! Tensor Manipulation Utilities
//!
//! This module provides comprehensive tensor manipulation utilities including
//! various squeeze/unsqueeze variants, advanced transpose operations, and
//! dimension manipulation helpers.
//!
//! # Features
//!
//! - **Smart squeezing**: Automatic removal of size-1 dimensions
//! - **Conditional unsqueezing**: Add dimensions based on patterns
//! - **Multi-transpose**: Transpose multiple dimensions at once
//! - **Dimension swapping**: Flexible dimension reordering
//! - **Shape inference**: Automatic shape calculation

use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
};

use crate::Tensor;

/// Extension trait for advanced tensor manipulation
pub trait TensorManipulationExt<T: TensorElement> {
    /// Squeeze all dimensions of size 1
    fn squeeze_all(&self) -> Result<Tensor<T>>;

    /// Squeeze specific dimensions
    fn squeeze_dims(&self, dims: &[i32]) -> Result<Tensor<T>>;

    /// Unsqueeze at multiple positions
    fn unsqueeze_dims(&self, dims: &[i32]) -> Result<Tensor<T>>;

    /// Add a batch dimension at the front
    fn add_batch_dim(&self) -> Result<Tensor<T>>;

    /// Remove the batch dimension (first dimension)
    fn remove_batch_dim(&self) -> Result<Tensor<T>>;

    /// Ensure tensor has at least N dimensions (add trailing dimensions)
    fn atleast_nd(&self, n: usize) -> Result<Tensor<T>>;

    /// Transpose to channel-last format (NCHW -> NHWC)
    fn to_channel_last(&self) -> Result<Tensor<T>>;

    /// Transpose to channel-first format (NHWC -> NCHW)
    fn to_channel_first(&self) -> Result<Tensor<T>>;

    /// Swap two dimensions
    fn swap_dims(&self, dim0: i32, dim1: i32) -> Result<Tensor<T>>;

    /// Move a dimension to a new position
    fn move_dim(&self, src: i32, dst: i32) -> Result<Tensor<T>>;

    /// Expand singleton dimensions to match target shape
    fn expand_to(&self, target_shape: &[usize]) -> Result<Tensor<T>>;

    /// Repeat tensor along new dimension
    fn repeat_along(&self, dim: i32, repeats: usize) -> Result<Tensor<T>>;
}

impl<T: TensorElement + Copy> TensorManipulationExt<T> for Tensor<T> {
    fn squeeze_all(&self) -> Result<Tensor<T>> {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        let new_shape: Vec<usize> = shape.iter().filter(|&&s| s != 1).copied().collect();

        if new_shape.is_empty() {
            // All dimensions were 1, create scalar
            self.reshape(&[1])
        } else {
            let new_shape_i32: Vec<i32> = new_shape.iter().map(|&s| s as i32).collect();
            self.reshape(&new_shape_i32)
        }
    }

    fn squeeze_dims(&self, dims: &[i32]) -> Result<Tensor<T>> {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        let ndim = shape.len() as i32;

        // Normalize dimensions
        let normalized_dims: Result<Vec<usize>> = dims
            .iter()
            .map(|&d| {
                let normalized = if d < 0 { ndim + d } else { d };
                if normalized < 0 || normalized >= ndim {
                    Err(TorshError::InvalidArgument(format!(
                        "Dimension {} out of range for tensor with {} dimensions",
                        d, ndim
                    )))
                } else {
                    Ok(normalized as usize)
                }
            })
            .collect();

        let normalized_dims = normalized_dims?;

        // Check that specified dimensions are size 1
        for &dim in &normalized_dims {
            if shape[dim] != 1 {
                return Err(TorshError::InvalidArgument(format!(
                    "Cannot squeeze dimension {} of size {}",
                    dim, shape[dim]
                )));
            }
        }

        // Build new shape without squeezed dimensions
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !normalized_dims.contains(i))
            .map(|(_, &s)| s)
            .collect();

        if new_shape.is_empty() {
            self.reshape(&[1])
        } else {
            let new_shape_i32: Vec<i32> = new_shape.iter().map(|&s| s as i32).collect();
            self.reshape(&new_shape_i32)
        }
    }

    fn unsqueeze_dims(&self, dims: &[i32]) -> Result<Tensor<T>> {
        let mut result = self.clone();

        // Sort dimensions to handle in ascending order
        let mut sorted_dims: Vec<i32> = dims.to_vec();
        sorted_dims.sort_unstable();

        // Process in order, adjusting subsequent dims for already-inserted dimensions
        for (i, &dim) in sorted_dims.iter().enumerate() {
            // Adjust for previously inserted dimensions
            let adjusted_dim = dim + i as i32;
            result = result.unsqueeze(adjusted_dim)?;
        }

        Ok(result)
    }

    fn add_batch_dim(&self) -> Result<Tensor<T>> {
        self.unsqueeze(0)
    }

    fn remove_batch_dim(&self) -> Result<Tensor<T>> {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        if shape.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot remove batch dim from scalar tensor".to_string(),
            ));
        }

        if shape[0] != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Batch dimension has size {}, expected 1",
                shape[0]
            )));
        }

        self.squeeze(0)
    }

    fn atleast_nd(&self, n: usize) -> Result<Tensor<T>> {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        let current_ndim = shape.len();

        if current_ndim >= n {
            return Ok(self.clone());
        }

        let mut new_shape = shape.to_vec();
        for _ in current_ndim..n {
            new_shape.push(1);
        }

        let new_shape_i32: Vec<i32> = new_shape.iter().map(|&s| s as i32).collect();
        self.reshape(&new_shape_i32)
    }

    fn to_channel_last(&self) -> Result<Tensor<T>> {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        match shape.len() {
            4 => {
                // NCHW -> NHWC
                self.permute(&[0, 2, 3, 1])
            }
            3 => {
                // CHW -> HWC
                self.permute(&[1, 2, 0])
            }
            _ => Err(TorshError::InvalidArgument(
                "to_channel_last expects 3D or 4D tensor".to_string(),
            )),
        }
    }

    fn to_channel_first(&self) -> Result<Tensor<T>> {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        match shape.len() {
            4 => {
                // NHWC -> NCHW
                self.permute(&[0, 3, 1, 2])
            }
            3 => {
                // HWC -> CHW
                self.permute(&[2, 0, 1])
            }
            _ => Err(TorshError::InvalidArgument(
                "to_channel_first expects 3D or 4D tensor".to_string(),
            )),
        }
    }

    fn swap_dims(&self, dim0: i32, dim1: i32) -> Result<Tensor<T>> {
        self.transpose(dim0, dim1)
    }

    fn move_dim(&self, src: i32, dst: i32) -> Result<Tensor<T>> {
        let ndim = self.shape().dims().len() as i32;

        // Normalize dimensions
        let src = if src < 0 { ndim + src } else { src };
        let dst = if dst < 0 { ndim + dst } else { dst };

        if src < 0 || src >= ndim || dst < 0 || dst >= ndim {
            return Err(TorshError::InvalidArgument(
                "Dimension out of range".to_string(),
            ));
        }

        if src == dst {
            return Ok(self.clone());
        }

        // Build permutation to move dimension
        let mut perm: Vec<i32> = (0..ndim).collect();
        let src_dim = perm.remove(src as usize);

        perm.insert(dst as usize, src_dim);

        self.permute(&perm)
    }

    fn expand_to(&self, target_shape: &[usize]) -> Result<Tensor<T>> {
        let shape_binding = self.shape();
        let current_shape = shape_binding.dims();

        if current_shape.len() > target_shape.len() {
            return Err(TorshError::InvalidArgument(
                "Cannot expand to shape with fewer dimensions".to_string(),
            ));
        }

        // Check compatibility
        for (i, &current_size) in current_shape.iter().rev().enumerate() {
            let target_idx = target_shape.len() - 1 - i;
            let target_size = target_shape[target_idx];

            if current_size != 1 && current_size != target_size {
                return Err(TorshError::InvalidArgument(format!(
                    "Cannot expand dimension {} from {} to {}",
                    target_idx, current_size, target_size
                )));
            }
        }

        self.expand(target_shape)
    }

    fn repeat_along(&self, dim: i32, repeats: usize) -> Result<Tensor<T>> {
        // First unsqueeze at dim, then repeat
        let unsqueezed = self.unsqueeze(dim)?;
        let shape_binding = unsqueezed.shape();
        let shape = shape_binding.dims();

        let mut repeat_shape = vec![1; shape.len()];
        let normalized_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        repeat_shape[normalized_dim] = repeats;

        unsqueezed.repeat(&repeat_shape)
    }
}

/// Helper functions for shape manipulation
pub mod shape_utils {
    use super::*;

    /// Calculate the number of elements in a shape
    pub fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Check if two shapes are broadcast-compatible
    pub fn are_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);

        for i in 0..max_len {
            let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };

            let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

            if dim1 != 1 && dim2 != 1 && dim1 != dim2 {
                return false;
            }
        }

        true
    }

    /// Calculate broadcast shape
    pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
        if !are_broadcastable(shape1, shape2) {
            return None;
        }

        let len1 = shape1.len();
        let len2 = shape2.len();
        let max_len = len1.max(len2);

        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let dim1 = if i < len1 { shape1[len1 - 1 - i] } else { 1 };

            let dim2 = if i < len2 { shape2[len2 - 1 - i] } else { 1 };

            result.push(dim1.max(dim2));
        }

        result.reverse();
        Some(result)
    }

    /// Infer shape with -1 (unknown dimension)
    pub fn infer_shape(shape: &[i32], total_elements: usize) -> Result<Vec<usize>> {
        let mut result = Vec::new();
        let mut unknown_idx = None;
        let mut known_product = 1usize;

        for (i, &dim) in shape.iter().enumerate() {
            if dim == -1 {
                if unknown_idx.is_some() {
                    return Err(TorshError::InvalidArgument(
                        "Only one dimension can be inferred".to_string(),
                    ));
                }
                unknown_idx = Some(i);
                result.push(0); // Placeholder
            } else if dim < 0 {
                return Err(TorshError::InvalidArgument(format!(
                    "Invalid dimension size: {}",
                    dim
                )));
            } else {
                result.push(dim as usize);
                known_product *= dim as usize;
            }
        }

        if let Some(idx) = unknown_idx {
            if known_product == 0 {
                return Err(TorshError::InvalidArgument(
                    "Cannot infer dimension with zero-sized dimensions".to_string(),
                ));
            }

            if total_elements % known_product != 0 {
                return Err(TorshError::InvalidArgument(
                    "Cannot infer dimension: size is not divisible".to_string(),
                ));
            }

            result[idx] = total_elements / known_product;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::*;

    #[test]
    fn test_squeeze_all() {
        let tensor = zeros::<f32>(&[1, 3, 1, 4, 1]).unwrap();
        let squeezed = tensor.squeeze_all().unwrap();

        assert_eq!(squeezed.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_squeeze_dims() {
        let tensor = zeros::<f32>(&[1, 3, 1, 4]).unwrap();
        let squeezed = tensor.squeeze_dims(&[0, 2]).unwrap();

        assert_eq!(squeezed.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_unsqueeze_dims() {
        let tensor = zeros::<f32>(&[3, 4]).unwrap();
        // unsqueeze at 0: [3, 4] -> [1, 3, 4]
        // unsqueeze at 2+1=3 (adjusted): [1, 3, 4] -> [1, 3, 4, 1]
        let unsqueezed = tensor.unsqueeze_dims(&[0, 2]).unwrap();

        assert_eq!(unsqueezed.shape().dims(), &[1, 3, 4, 1]);
    }

    #[test]
    fn test_add_remove_batch_dim() {
        let tensor = zeros::<f32>(&[3, 4]).unwrap();
        let with_batch = tensor.add_batch_dim().unwrap();

        assert_eq!(with_batch.shape().dims(), &[1, 3, 4]);

        let without_batch = with_batch.remove_batch_dim().unwrap();
        assert_eq!(without_batch.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_atleast_nd() {
        let tensor = zeros::<f32>(&[3, 4]).unwrap();
        let expanded = tensor.atleast_nd(4).unwrap();

        assert_eq!(expanded.shape().dims(), &[3, 4, 1, 1]);
    }

    #[test]
    fn test_channel_conversions() {
        let tensor = zeros::<f32>(&[2, 3, 4, 5]).unwrap(); // NCHW

        let channel_last = tensor.to_channel_last().unwrap();
        assert_eq!(channel_last.shape().dims(), &[2, 4, 5, 3]); // NHWC

        let channel_first = channel_last.to_channel_first().unwrap();
        assert_eq!(channel_first.shape().dims(), &[2, 3, 4, 5]); // Back to NCHW
    }

    #[test]
    fn test_move_dim() {
        let tensor = zeros::<f32>(&[2, 3, 4, 5]).unwrap();
        let moved = tensor.move_dim(1, 3).unwrap();

        // Move dimension 1 to position 3
        assert_eq!(moved.shape().dims(), &[2, 4, 5, 3]);
    }

    #[test]
    fn test_shape_utils_broadcastable() {
        use shape_utils::*;

        assert!(are_broadcastable(&[3, 1, 4], &[1, 5, 4]));
        assert!(are_broadcastable(&[3, 4], &[3, 4]));
        assert!(are_broadcastable(&[1], &[3, 4]));

        assert!(!are_broadcastable(&[3, 4], &[2, 4]));
    }

    #[test]
    fn test_shape_utils_broadcast_shape() {
        use shape_utils::*;

        let result = broadcast_shape(&[3, 1, 4], &[1, 5, 4]);
        assert_eq!(result, Some(vec![3, 5, 4]));

        let result = broadcast_shape(&[3, 4], &[2, 4]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_shape_utils_infer_shape() {
        use shape_utils::*;

        let inferred = infer_shape(&[2, -1, 3], 24).unwrap();
        assert_eq!(inferred, vec![2, 4, 3]);

        let inferred = infer_shape(&[3, 4], 12).unwrap();
        assert_eq!(inferred, vec![3, 4]);
    }

    #[test]
    fn test_squeeze_dims_invalid() {
        let tensor = zeros::<f32>(&[3, 4]).unwrap();
        let result = tensor.squeeze_dims(&[0]); // Dimension 0 has size 3, not 1

        assert!(result.is_err());
    }

    #[test]
    fn test_remove_batch_dim_invalid() {
        let tensor = zeros::<f32>(&[3, 4]).unwrap();
        let result = tensor.remove_batch_dim(); // First dim has size 3, not 1

        assert!(result.is_err());
    }
}
