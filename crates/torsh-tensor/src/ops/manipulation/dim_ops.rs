//! Dimension manipulation and broadcasting operations
//!
//! This module provides PyTorch-compatible dimension manipulation operations including:
//! - Dimension manipulation: movedim, moveaxis, swapaxes, swapdims
//! - Shape transformation: unflatten
//! - Broadcasting: broadcast_to, expand_as
//! - Advanced indexing: take_along_dim

use crate::{Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};

impl<T: TensorElement + Copy + Default> Tensor<T> {
    /// Unflattens a dimension into multiple dimensions
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.unflatten(tensor, dim, sizes)`
    ///
    /// # Arguments
    /// * `dim` - Dimension to unflatten
    /// * `sizes` - Target sizes for the unflattened dimensions
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], DeviceType::Cpu)?;
    /// let y = x.unflatten(0, &[2, 3])?; // Shape becomes [2, 3]
    /// ```
    pub fn unflatten(&self, dim: isize, sizes: &[usize]) -> Result<Self> {
        if sizes.is_empty() {
            return Err(TorshError::InvalidArgument(
                "sizes cannot be empty".to_string(),
            ));
        }

        let shape_obj = self.shape();
        let shape = shape_obj.dims();
        let ndim = shape.len();

        // Normalize dimension
        let dim = if dim < 0 {
            (ndim as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for {}-D tensor",
                dim, ndim
            )));
        }

        // Verify that the product of sizes equals the dimension size
        let sizes_product: usize = sizes.iter().product();
        if sizes_product != shape[dim] {
            return Err(TorshError::InvalidArgument(format!(
                "sizes product {} does not match dimension size {}",
                sizes_product, shape[dim]
            )));
        }

        // Build new shape
        let mut new_shape = Vec::new();
        new_shape.extend_from_slice(&shape[..dim]);
        new_shape.extend_from_slice(sizes);
        new_shape.extend_from_slice(&shape[dim + 1..]);

        // Reshape (data layout doesn't change, only shape interpretation)
        let data = self.data()?;
        Self::from_data(data, new_shape, self.device.clone())
    }

    /// Move dimensions from source positions to destination positions
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.movedim(tensor, source, destination)`
    ///
    /// # Arguments
    /// * `source` - Original positions of dimensions to move
    /// * `destination` - Target positions for the dimensions
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0; 24], vec![2, 3, 4], DeviceType::Cpu)?;
    /// let y = x.movedim(&[0, 1], &[2, 0])?; // [2,3,4] -> [3,4,2]
    /// ```
    pub fn movedim(&self, source: &[isize], destination: &[isize]) -> Result<Self> {
        if source.len() != destination.len() {
            return Err(TorshError::InvalidArgument(
                "source and destination must have the same length".to_string(),
            ));
        }

        let ndim = self.ndim();

        // Normalize source and destination dimensions
        let norm_source: Result<Vec<usize>> = source
            .iter()
            .map(|&d| {
                let dim = if d < 0 {
                    (ndim as isize + d) as usize
                } else {
                    d as usize
                };
                if dim >= ndim {
                    Err(TorshError::InvalidArgument(format!(
                        "Dimension {} out of range for {}-D tensor",
                        d, ndim
                    )))
                } else {
                    Ok(dim)
                }
            })
            .collect();
        let norm_source = norm_source?;

        let norm_dest: Result<Vec<usize>> = destination
            .iter()
            .map(|&d| {
                let dim = if d < 0 {
                    (ndim as isize + d) as usize
                } else {
                    d as usize
                };
                if dim >= ndim {
                    Err(TorshError::InvalidArgument(format!(
                        "Dimension {} out of range for {}-D tensor",
                        d, ndim
                    )))
                } else {
                    Ok(dim)
                }
            })
            .collect();
        let norm_dest = norm_dest?;

        // Check for duplicates in source
        for i in 0..norm_source.len() {
            for j in i + 1..norm_source.len() {
                if norm_source[i] == norm_source[j] {
                    return Err(TorshError::InvalidArgument(
                        "repeated dim in source".to_string(),
                    ));
                }
            }
        }

        // Check for duplicates in destination
        for i in 0..norm_dest.len() {
            for j in i + 1..norm_dest.len() {
                if norm_dest[i] == norm_dest[j] {
                    return Err(TorshError::InvalidArgument(
                        "repeated dim in destination".to_string(),
                    ));
                }
            }
        }

        // Build permutation array
        let mut perm: Vec<usize> = (0..ndim).collect();

        // Remove source dimensions from perm
        let mut removed_dims = Vec::new();
        for &src in norm_source.iter().rev() {
            removed_dims.push(perm.remove(src));
        }
        removed_dims.reverse();

        // Insert at destination positions
        for (&dst, &dim) in norm_dest.iter().zip(removed_dims.iter()) {
            perm.insert(dst, dim);
        }

        self.permute(&perm)
    }

    /// Move axis from source position to destination position (alias for movedim)
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.moveaxis(tensor, source, destination)`
    ///
    /// # Arguments
    /// * `source` - Original positions of axes to move
    /// * `destination` - Target positions for the axes
    pub fn moveaxis(&self, source: &[isize], destination: &[isize]) -> Result<Self> {
        self.movedim(source, destination)
    }

    /// Swap two dimensions
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.swapaxes(tensor, axis0, axis1)` or `torch.swapdims(tensor, dim0, dim1)`
    ///
    /// # Arguments
    /// * `axis0` - First dimension
    /// * `axis1` - Second dimension
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0; 12], vec![2, 3, 2], DeviceType::Cpu)?;
    /// let y = x.swapaxes(0, 2)?; // [2,3,2] -> [2,3,2] with dims 0 and 2 swapped
    /// ```
    pub fn swapaxes(&self, axis0: isize, axis1: isize) -> Result<Self> {
        let ndim = self.ndim();

        // Normalize dimensions
        let dim0 = if axis0 < 0 {
            (ndim as isize + axis0) as usize
        } else {
            axis0 as usize
        };
        let dim1 = if axis1 < 0 {
            (ndim as isize + axis1) as usize
        } else {
            axis1 as usize
        };

        if dim0 >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for {}-D tensor",
                axis0, ndim
            )));
        }
        if dim1 >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for {}-D tensor",
                axis1, ndim
            )));
        }

        // Build permutation: swap dim0 and dim1
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(dim0, dim1);

        self.permute(&perm)
    }

    /// Swap two dimensions (alias for swapaxes)
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.swapdims(tensor, dim0, dim1)`
    pub fn swapdims(&self, dim0: isize, dim1: isize) -> Result<Self> {
        self.swapaxes(dim0, dim1)
    }

    /// Broadcast tensor to a new shape
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.broadcast_to(tensor, shape)`
    ///
    /// # Arguments
    /// * `shape` - Target shape for broadcasting
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu)?;
    /// let y = x.broadcast_to(&[3, 2])?; // Broadcast [2] to [3, 2]
    /// ```
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        let current_shape = self.shape().dims();
        let current_ndim = current_shape.len();
        let target_ndim = shape.len();

        // Check that target has at least as many dimensions
        if target_ndim < current_ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Cannot broadcast from shape {:?} to shape {:?}: target has fewer dimensions",
                current_shape, shape
            )));
        }

        // Align shapes from the right (trailing dimensions)
        let offset = target_ndim - current_ndim;

        // Check broadcasting compatibility and build repeat counts
        let mut repeats = vec![1; target_ndim];
        for (i, &current_dim) in current_shape.iter().enumerate() {
            let target_idx = offset + i;
            let target_dim = shape[target_idx];

            if current_dim != target_dim {
                if current_dim == 1 {
                    // Can broadcast dimension of size 1 to any size
                    repeats[target_idx] = target_dim;
                } else if target_dim == 1 {
                    // Broadcasting to size 1 requires current dim to be 1
                    return Err(TorshError::InvalidArgument(format!(
                        "Cannot broadcast dimension {} from size {} to size 1",
                        i, current_dim
                    )));
                } else {
                    // Incompatible dimensions
                    return Err(TorshError::InvalidArgument(format!(
                        "Cannot broadcast from shape {:?} to shape {:?}: dimension {} has size {} but target is {}",
                        current_shape, shape, i, current_dim, target_dim
                    )));
                }
            }
        }

        // Handle leading dimensions (prepend 1s)
        for i in 0..offset {
            repeats[i] = shape[i];
        }

        // If already the right shape, return clone
        if repeats.iter().all(|&r| r == 1) && current_ndim == target_ndim {
            return Ok(self.clone());
        }

        // First reshape to add leading dimensions if needed
        let mut result = if offset > 0 {
            let mut new_shape = vec![1; offset];
            new_shape.extend_from_slice(current_shape);
            let data = self.data()?;
            Self::from_data(data, new_shape, self.device.clone())?
        } else {
            self.clone()
        };

        // Now use repeat to expand each dimension
        result.repeat(&repeats)
    }

    /// Expand tensor to match another tensor's shape
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.expand_as(tensor, other)`
    ///
    /// # Arguments
    /// * `other` - Target tensor whose shape to match
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu)?;
    /// let y = Tensor::from_data(vec![0.0; 6], vec![3, 2], DeviceType::Cpu)?;
    /// let z = x.expand_as(&y)?; // Expand x to match y's shape [3, 2]
    /// ```
    pub fn expand_as(&self, other: &Self) -> Result<Self> {
        self.broadcast_to(other.shape().dims())
    }

    /// Gather values along an axis specified by indices
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.take_along_dim(tensor, indices, dim)`
    ///
    /// # Arguments
    /// * `indices` - Tensor of indices (must be i64 type)
    /// * `dim` - Dimension along which to gather (None means flatten first)
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)?;
    /// let indices = Tensor::from_data(vec![1i64, 0], vec![2, 1], DeviceType::Cpu)?;
    /// let y = x.take_along_dim(&indices, Some(1))?; // Gather along dim 1
    /// ```
    pub fn take_along_dim(&self, indices: &Tensor<i64>, dim: Option<isize>) -> Result<Self>
    where
        T: Copy,
    {
        match dim {
            None => {
                // Flatten both tensors and use simple indexing
                let data = self.data()?;
                let idx_data = indices.data()?;

                let mut result = Vec::with_capacity(idx_data.len());

                for &idx in idx_data.iter() {
                    if idx < 0 || idx as usize >= data.len() {
                        return Err(TorshError::InvalidArgument(format!(
                            "Index {} out of range for tensor with {} elements",
                            idx,
                            data.len()
                        )));
                    }
                    result.push(data[idx as usize]);
                }

                Self::from_data(result, indices.shape().dims().to_vec(), self.device.clone())
            }
            Some(d) => {
                let ndim = self.ndim();
                let dim = if d < 0 {
                    (ndim as isize + d) as usize
                } else {
                    d as usize
                };

                if dim >= ndim {
                    return Err(TorshError::InvalidArgument(format!(
                        "Dimension {} out of range for {}-D tensor",
                        d, ndim
                    )));
                }

                let self_shape_obj = self.shape();
                let self_shape = self_shape_obj.dims();
                let indices_shape_obj = indices.shape();
                let indices_shape = indices_shape_obj.dims();

                // Verify shapes match except at the gather dimension
                if self_shape.len() != indices_shape.len() {
                    return Err(TorshError::ShapeMismatch {
                        expected: self_shape.to_vec(),
                        got: indices_shape.to_vec(),
                    });
                }

                for (i, (&s, &idx_s)) in self_shape.iter().zip(indices_shape.iter()).enumerate() {
                    if i != dim && s != idx_s {
                        return Err(TorshError::ShapeMismatch {
                            expected: self_shape.to_vec(),
                            got: indices_shape.to_vec(),
                        });
                    }
                }

                let data = self.data()?;
                let idx_data = indices.data()?;

                let dim_size = self_shape[dim];
                let outer_size: usize = self_shape[..dim].iter().product();
                let inner_size: usize = self_shape[dim + 1..].iter().product();

                let indices_dim_size = indices_shape[dim];
                let mut result = Vec::with_capacity(idx_data.len());

                for outer in 0..outer_size {
                    for d in 0..indices_dim_size {
                        for inner in 0..inner_size {
                            let idx_flat = outer * indices_dim_size * inner_size + d * inner_size + inner;
                            let gather_idx = idx_data[idx_flat];

                            if gather_idx < 0 || gather_idx as usize >= dim_size {
                                return Err(TorshError::InvalidArgument(format!(
                                    "Index {} out of range for dimension size {}",
                                    gather_idx, dim_size
                                )));
                            }

                            let src_idx = outer * dim_size * inner_size
                                + (gather_idx as usize) * inner_size
                                + inner;

                            result.push(data[src_idx]);
                        }
                    }
                }

                Self::from_data(result, indices_shape.to_vec(), self.device.clone())
            }
        }
    }
}
