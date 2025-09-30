//! Shape and view operations for tensors
//!
//! This module provides comprehensive tensor shape manipulation and view operations
//! including reshaping, transposing, slicing, squeezing, unsqueezing, and permuting.
//!
//! # Features
//!
//! - **Zero-copy views**: Efficient view operations that share underlying data
//! - **Safe reshaping**: Comprehensive validation and overflow checking
//! - **Dimension manipulation**: Squeeze, unsqueeze, transpose, and permute operations
//! - **Slicing operations**: Flexible tensor slicing with stride computation
//! - **Broadcasting support**: Expand operations for broadcasting compatibility
//! - **Contiguity checking**: Efficient memory layout validation

use std::sync::{Arc, RwLock, Weak};
use torsh_core::{
    dtype::TensorElement,
    error::{Result, TorshError},
    shape::Shape,
};

use crate::core_ops::{Operation, Tensor};

impl<T: TensorElement + Copy> Tensor<T> {
    /// Get size of a specific dimension
    pub fn size(&self, dim: i32) -> Result<usize> {
        self.shape().size(dim)
    }

    /// Reshape the tensor
    pub fn view(&self, shape: &[i32]) -> Result<Self> {
        // Validate that there's at most one -1 in the shape
        let infer_count = shape.iter().filter(|&&x| x == -1).count();
        if infer_count > 1 {
            return Err(TorshError::InvalidShape(
                "Only one dimension can be inferred (only one -1 allowed)".to_string(),
            ));
        }

        let new_shape: Result<Vec<usize>> = shape
            .iter()
            .map(|&d| {
                if d == -1 {
                    // Infer dimension - first validate all other dimensions are valid
                    let known_dims: Result<Vec<usize>> = shape
                        .iter()
                        .filter(|&&x| x != -1)
                        .map(|&x| {
                            if x < 0 {
                                Err(TorshError::InvalidShape(format!(
                                    "Invalid dimension size: {x} (negative dimensions not allowed except -1)"
                                )))
                            } else {
                                Ok(x as usize)
                            }
                        })
                        .collect();

                    let known_dims = known_dims?;

                    // Check for overflow in product calculation
                    let known_product = known_dims.iter().try_fold(1usize, |acc, &dim| {
                        acc.checked_mul(dim).ok_or_else(|| {
                            TorshError::InvalidShape(
                                "Shape dimensions too large (would overflow)".to_string()
                            )
                        })
                    })?;

                    if known_product == 0 {
                        return Err(TorshError::InvalidShape(
                            "Cannot infer dimension with zero-sized dimensions".to_string(),
                        ));
                    }

                    let total = self.numel();
                    if total % known_product != 0 {
                        return Err(TorshError::InvalidShape(
                            "Cannot infer dimension: size is not divisible".to_string(),
                        ));
                    }

                    Ok(total / known_product)
                } else if d < 0 {
                    Err(TorshError::InvalidShape(format!(
                        "Invalid dimension size: {d}"
                    )))
                } else {
                    Ok(d as usize)
                }
            })
            .collect();

        let new_shape = new_shape?;

        // Check for overflow in total elements calculation
        let new_numel = new_shape.iter().try_fold(1usize, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                TorshError::InvalidShape(
                    "Reshaped tensor would be too large (would overflow)".to_string(),
                )
            })
        })?;

        if new_numel != self.numel() {
            return Err(TorshError::InvalidShape(format!(
                "Shape {:?} is invalid for tensor of size {}",
                new_shape,
                self.numel()
            )));
        }

        // Create a new tensor with the same data but different shape
        let data = self.to_vec()?;
        Self::from_data(data, new_shape, self.device)
    }

    /// Create an efficient view with different shape (shares data, no copying)
    /// This is the zero-copy version of view() for compatible shapes
    pub fn view_as(&self, shape: &[usize]) -> Result<Self> {
        // Validate that the total number of elements is the same
        let new_numel = shape.iter().product::<usize>();
        if new_numel != self.numel() {
            return Err(TorshError::InvalidShape(format!(
                "Shape {:?} is invalid for tensor of size {}",
                shape,
                self.numel()
            )));
        }

        // Only create efficient views for contiguous tensors or existing views
        // that are still relatively simple
        if !self.is_contiguous() {
            return Err(TorshError::InvalidShape(
                "Cannot create efficient view of non-contiguous tensor".to_string(),
            ));
        }

        // Create new tensor sharing the same storage
        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(shape.to_vec()),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)), // Views don't share gradients
            operation: Operation::Leaf,        // Views reset operation tracking
            strides: None,                     // Use default contiguous strides for simple reshapes
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                // If this is already a view, keep reference to the original base
                self.base_tensor.clone()
            } else {
                // This is a base tensor, so create a weak reference to it
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Create a view of a slice along a dimension (shares data, no copying)
    pub fn slice_tensor(&self, dim: usize, start: usize, end: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let shape = self.shape.dims();
        if start >= shape[dim] || end > shape[dim] || start >= end {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid slice range [{}:{}] for dimension {} of size {}",
                start, end, dim, shape[dim]
            )));
        }

        // Calculate new shape
        let mut new_shape = shape.to_vec();
        new_shape[dim] = end - start;

        // Calculate new strides and offset
        let current_strides = self.strides();
        let offset_adjustment = start * current_strides[dim];

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(current_strides),
            storage_offset: self.storage_offset + offset_adjustment,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Create a transposed view (shares data, no copying)
    pub fn transpose_view(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimensions {} and {} out of range for tensor with {} dimensions",
                dim0,
                dim1,
                self.ndim()
            )));
        }

        if dim0 == dim1 {
            return Ok(self.clone());
        }

        // Create new shape and strides
        let mut new_shape = self.shape.dims().to_vec();
        let mut new_strides = self.strides();

        // Swap dimensions
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(new_strides),
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Squeeze a tensor along a specific dimension (removes dimension of size 1)
    pub fn squeeze_tensor(&self, dim: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let shape = self.shape.dims();
        if shape[dim] != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Cannot squeeze dimension {} of size {}",
                dim, shape[dim]
            )));
        }

        // Remove the dimension from shape and strides
        let mut new_shape = shape.to_vec();
        new_shape.remove(dim);

        let mut new_strides = self.strides();
        new_strides.remove(dim);

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(new_strides),
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Unsqueeze a tensor at a specific dimension (adds dimension of size 1)
    pub fn unsqueeze_tensor(&self, dim: usize) -> Result<Self> {
        if dim > self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for insertion in tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        // Insert new dimension into shape and strides
        let mut new_shape = self.shape.dims().to_vec();
        new_shape.insert(dim, 1);

        let mut new_strides = self.strides();
        // For the new dimension, stride should be the product of all dimensions to the right
        let new_stride = if dim == new_shape.len() - 1 {
            1 // Last dimension always has stride 1
        } else {
            new_strides[dim] // Use the stride that was at this position
        };
        new_strides.insert(dim, new_stride);

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(new_strides),
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Transpose two dimensions (with data copying)
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Self> {
        let ndim = self.ndim();
        let dim0 = if dim0 < 0 {
            (ndim as i32 + dim0) as usize
        } else {
            dim0 as usize
        };
        let dim1 = if dim1 < 0 {
            (ndim as i32 + dim1) as usize
        } else {
            dim1 as usize
        };

        if dim0 >= ndim || dim1 >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimensions {} and {} out of range for tensor with {} dimensions",
                dim0, dim1, ndim
            )));
        }

        if ndim == 2 && dim0 != dim1 {
            self.transpose_2d()
        } else {
            self.transpose_view(dim0, dim1)
        }
    }

    /// 2D transpose implementation
    fn transpose_2d(&self) -> Result<Self> {
        let shape = self.shape.dims();
        if shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "transpose_2d only works with 2D tensors".to_string(),
            ));
        }

        let (rows, cols) = (shape[0], shape[1]);
        let data = self.to_vec()?;
        let mut transposed_data = Vec::with_capacity(data.len());

        for col in 0..cols {
            for row in 0..rows {
                transposed_data.push(data[row * cols + col]);
            }
        }

        Self::from_data(transposed_data, vec![cols, rows], self.device)
    }

    /// Permute dimensions according to the given order
    pub fn permute(&self, dims: &[i32]) -> Result<Self> {
        let ndim = self.ndim();

        if dims.len() != ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Number of dimensions in permutation ({}) doesn't match tensor dimensions ({})",
                dims.len(),
                ndim
            )));
        }

        // Convert negative indices and validate
        let perm_dims: Result<Vec<usize>> = dims
            .iter()
            .map(|&d| {
                let dim = if d < 0 { ndim as i32 + d } else { d } as usize;
                if dim >= ndim {
                    Err(TorshError::InvalidArgument(format!(
                        "Dimension {} out of range for tensor with {} dimensions",
                        d, ndim
                    )))
                } else {
                    Ok(dim)
                }
            })
            .collect();

        let perm_dims = perm_dims?;

        // Check for duplicates
        let mut sorted_dims = perm_dims.clone();
        sorted_dims.sort_unstable();
        for i in 0..ndim {
            if sorted_dims[i] != i {
                return Err(TorshError::InvalidArgument(
                    "Permutation must contain each dimension exactly once".to_string(),
                ));
            }
        }

        // Create new shape and strides
        let old_shape = self.shape.dims();
        let old_strides = self.strides();

        let new_shape: Vec<usize> = perm_dims.iter().map(|&i| old_shape[i]).collect();
        let new_strides: Vec<usize> = perm_dims.iter().map(|&i| old_strides[i]).collect();

        Ok(Self {
            storage: self.storage.clone(),
            shape: Shape::new(new_shape),
            device: self.device,
            requires_grad: self.requires_grad,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: Some(new_strides),
            storage_offset: self.storage_offset,
            base_tensor: if self.is_view() {
                self.base_tensor.clone()
            } else {
                Some(Arc::downgrade(&Arc::new(self.clone())))
            },
        })
    }

    /// Squeeze dimension with size 1
    pub fn squeeze(&self, dim: i32) -> Result<Self> {
        let ndim = self.ndim();
        let dim = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };

        self.squeeze_tensor(dim)
    }

    /// Squeeze all dimensions with size 1
    pub fn squeeze_all(&self) -> Result<Self> {
        let shape = self.shape.dims();
        let new_shape: Vec<usize> = shape.iter().copied().filter(|&s| s != 1).collect();

        if new_shape.is_empty() {
            // If all dimensions were 1, result should be a scalar (0-dimensional tensor)
            let data = self.to_vec()?;
            Self::from_data(data, vec![], self.device)
        } else {
            let data = self.to_vec()?;
            Self::from_data(data, new_shape, self.device)
        }
    }

    /// Add a dimension of size 1 at the specified position
    pub fn unsqueeze(&self, dim: i32) -> Result<Self> {
        let ndim = self.ndim();
        let dim = if dim < 0 {
            (ndim as i32 + dim + 1) as usize
        } else {
            dim as usize
        };

        self.unsqueeze_tensor(dim)
    }

    /// Reshape tensor to new shape
    pub fn reshape(&self, shape: &[i32]) -> Result<Self> {
        self.view(shape)
    }

    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        // A tensor is contiguous if its strides match the default strides for its shape
        let default_strides = self.compute_default_strides();
        let current_strides = self.strides();

        current_strides == default_strides
    }

    /// Make tensor contiguous if it isn't already
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            // Need to copy data to make it contiguous
            let data = self.to_vec()?;
            Self::from_data(data, self.shape.dims().to_vec(), self.device)
        }
    }

    /// Expand tensor to a larger size
    pub fn expand(&self, shape: &[usize]) -> Result<Self> {
        let old_shape = self.shape.dims();

        // Validate that expansion is possible
        if shape.len() < old_shape.len() {
            return Err(TorshError::InvalidShape(
                "Cannot expand to smaller number of dimensions".to_string(),
            ));
        }

        // Check dimension compatibility (broadcasting rules)
        let offset = shape.len() - old_shape.len();
        for (i, &old_dim) in old_shape.iter().enumerate() {
            let new_dim = shape[offset + i];
            if old_dim != 1 && old_dim != new_dim {
                return Err(TorshError::InvalidShape(format!(
                    "Cannot expand dimension {} from {} to {}",
                    i, old_dim, new_dim
                )));
            }
        }

        // For now, implement expansion by copying data
        // TODO: Implement efficient expansion with strided views
        let source_data = self.to_vec()?;
        let target_numel = shape.iter().product();
        let mut result_data = Vec::with_capacity(target_numel);

        self.expand_data_recursive(&source_data, &mut result_data, shape, old_shape, 0, 0)?;

        Self::from_data(result_data, shape.to_vec(), self.device)
    }

    /// Helper for recursive data expansion
    fn expand_data_recursive(
        &self,
        source: &[T],
        dest: &mut Vec<T>,
        target_shape: &[usize],
        source_shape: &[usize],
        target_dim: usize,
        source_offset: usize,
    ) -> Result<()> {
        if target_dim == target_shape.len() {
            // Base case: copy single element
            dest.push(source[source_offset]);
            return Ok(());
        }

        let target_size = target_shape[target_dim];
        let source_dim_idx = target_dim + source_shape.len() - target_shape.len();

        if source_dim_idx < source_shape.len() {
            let source_size = source_shape[source_dim_idx];
            let stride = if source_dim_idx + 1 < source_shape.len() {
                source_shape[source_dim_idx + 1..].iter().product()
            } else {
                1
            };

            if source_size == 1 {
                // Broadcast this dimension
                for _ in 0..target_size {
                    self.expand_data_recursive(
                        source,
                        dest,
                        target_shape,
                        source_shape,
                        target_dim + 1,
                        source_offset,
                    )?;
                }
            } else {
                // Copy along this dimension
                for i in 0..target_size {
                    self.expand_data_recursive(
                        source,
                        dest,
                        target_shape,
                        source_shape,
                        target_dim + 1,
                        source_offset + i * stride,
                    )?;
                }
            }
        } else {
            // This is a new dimension, repeat the entire subtensor
            for _ in 0..target_size {
                self.expand_data_recursive(
                    source,
                    dest,
                    target_shape,
                    source_shape,
                    target_dim + 1,
                    source_offset,
                )?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_tensor_view() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).unwrap();

        let reshaped = tensor.view(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert_eq!(reshaped.numel(), 6);
    }

    #[test]
    fn test_tensor_view_with_inference() {
        let data = vec![1.0f32; 24];
        let tensor = Tensor::from_data(data, vec![2, 3, 4], DeviceType::Cpu).unwrap();

        let reshaped = tensor.view(&[6, -1]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[6, 4]);
    }

    #[test]
    fn test_tensor_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).unwrap();

        let slice = tensor.slice_tensor(1, 1, 3).unwrap();
        assert_eq!(slice.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_tensor_transpose() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu).unwrap();

        let transposed = tensor.transpose(0, 1).unwrap();
        assert_eq!(transposed.shape().dims(), &[2, 2]);
        assert_eq!(transposed.get(&[0, 1]).unwrap(), 3.0);
        assert_eq!(transposed.get(&[1, 0]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensor_squeeze_unsqueeze() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data(data, vec![1, 3], DeviceType::Cpu).unwrap();

        let squeezed = tensor.squeeze(0).unwrap();
        assert_eq!(squeezed.shape().dims(), &[3]);

        let unsqueezed = squeezed.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape().dims(), &[1, 3]);
    }

    #[test]
    fn test_tensor_permute() {
        let data = vec![1.0f32; 24];
        let tensor = Tensor::from_data(data, vec![2, 3, 4], DeviceType::Cpu).unwrap();

        let permuted = tensor.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape().dims(), &[4, 2, 3]);
    }

    #[test]
    fn test_is_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu).unwrap();
        assert!(tensor.is_contiguous());

        let transposed = tensor.transpose_view(0, 1).unwrap();
        assert!(!transposed.is_contiguous());

        let contiguous = transposed.contiguous().unwrap();
        assert!(contiguous.is_contiguous());
    }

    #[test]
    fn test_expand() {
        let data = vec![1.0f32, 2.0];
        let tensor = Tensor::from_data(data, vec![1, 2], DeviceType::Cpu).unwrap();

        let expanded = tensor.expand(&[3, 2]).unwrap();
        assert_eq!(expanded.shape().dims(), &[3, 2]);
        assert_eq!(expanded.numel(), 6);
    }

    #[test]
    fn test_view_error_handling() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::from_data(data, vec![3], DeviceType::Cpu).unwrap();

        // Should fail - wrong total size
        assert!(tensor.view(&[2, 2]).is_err());

        // Should fail - multiple -1
        assert!(tensor.view(&[-1, -1]).is_err());
    }
}
