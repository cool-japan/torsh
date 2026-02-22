//! Core tensor manipulation operations: cat, stack, chunk, split, flip, roll, rot90, tile, repeat
//!
//! This module provides PyTorch-compatible tensor manipulation operations including:
//! - Concatenation: cat, stack
//! - Splitting: split, chunk
//! - Flipping: flip, fliplr, flipud, rot90
//! - Repeating: tile, repeat, repeat_interleave
//! - Rolling: roll

use crate::{Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};

impl<T: TensorElement + Copy + Default> Tensor<T> {
    /// Concatenate tensors along a given dimension
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.cat(tensors, dim)`
    ///
    /// # Arguments
    /// * `tensors` - Sequence of tensors to concatenate
    /// * `dim` - Dimension along which to concatenate
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu)?;
    /// let b = Tensor::from_data(vec![3.0, 4.0], vec![2], DeviceType::Cpu)?;
    /// let result = Tensor::cat(&[a, b], 0)?; // [1.0, 2.0, 3.0, 4.0]
    /// ```
    pub fn cat(tensors: &[Self], dim: isize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument(
                "cat requires at least one tensor".to_string(),
            ));
        }

        let ndim = tensors[0].ndim();
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

        // Verify all tensors have compatible shapes
        let first_shape = tensors[0].shape().dims();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.ndim() != ndim {
                return Err(TorshError::InvalidArgument(format!(
                    "All tensors must have the same number of dimensions, tensor {} has {} dims vs {} dims",
                    i, tensor.ndim(), ndim
                )));
            }

            for (d, (&s1, &s2)) in first_shape
                .iter()
                .zip(tensor.shape().dims().iter())
                .enumerate()
            {
                if d != dim && s1 != s2 {
                    return Err(TorshError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: tensor.shape().to_vec(),
                    });
                }
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        output_shape[dim] = tensors.iter().map(|t| t.shape().dims()[dim]).sum();

        // Concatenate data
        let mut result_data = Vec::new();
        let mut outer_size = 1;
        let mut inner_size = 1;

        for i in 0..dim {
            outer_size *= first_shape[i];
        }
        for i in dim + 1..ndim {
            inner_size *= first_shape[i];
        }

        for _ in 0..outer_size {
            for tensor in tensors {
                let dim_size = tensor.shape().dims()[dim];
                let chunk_size = dim_size * inner_size;

                // This is a simplified implementation
                // A full implementation would need proper multi-dimensional indexing
                let data = tensor.data()?;
                for i in 0..chunk_size {
                    result_data.push(data[i % data.len()]);
                }
            }
        }

        let device = tensors[0].device.clone();
        Self::from_data(result_data, output_shape, device)
    }

    /// Stack tensors along a new dimension
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.stack(tensors, dim)`
    ///
    /// # Arguments
    /// * `tensors` - Sequence of tensors to stack
    /// * `dim` - Dimension along which to stack
    ///
    /// # Examples
    /// ```ignore
    /// let a = Tensor::from_data(vec![1.0, 2.0], vec![2], DeviceType::Cpu)?;
    /// let b = Tensor::from_data(vec![3.0, 4.0], vec![2], DeviceType::Cpu)?;
    /// let result = Tensor::stack(&[a, b], 0)?; // shape: [2, 2]
    /// ```
    pub fn stack(tensors: &[Self], dim: isize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument(
                "stack requires at least one tensor".to_string(),
            ));
        }

        // Verify all tensors have the same shape
        let first_shape = tensors[0].shape().dims();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape().dims() != first_shape {
                return Err(TorshError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    got: tensor.shape().dims().to_vec(),
                });
            }
        }

        let ndim = first_shape.len();
        let dim = if dim < 0 {
            ((ndim + 1) as isize + dim) as usize
        } else {
            dim as usize
        };

        if dim > ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for stacking {}-D tensors",
                dim, ndim
            )));
        }

        // First unsqueeze all tensors at dim, then cat
        let unsqueezed: Result<Vec<Self>> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim as isize))
            .collect();
        let unsqueezed = unsqueezed?;

        Self::cat(&unsqueezed, dim as isize)
    }

    /// Split tensor into chunks along a dimension
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.chunk(tensor, chunks, dim)`
    pub fn chunk(&self, chunks: usize, dim: isize) -> Result<Vec<Self>> {
        if chunks == 0 {
            return Err(TorshError::InvalidArgument(
                "chunks must be greater than 0".to_string(),
            ));
        }

        let ndim = self.ndim();
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

        let dim_size = self.shape().dims()[dim];
        let chunk_size = (dim_size + chunks - 1) / chunks; // Ceiling division

        let mut result = Vec::new();
        for i in 0..chunks {
            let start = i * chunk_size;
            if start >= dim_size {
                break;
            }
            let end = ((i + 1) * chunk_size).min(dim_size);

            // Create slice indices
            let slice_tensor = self.narrow(dim as isize, start, end - start)?;
            result.push(slice_tensor);
        }

        Ok(result)
    }

    /// Split tensor into sections of given size
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.split(tensor, split_size, dim)`
    pub fn split(&self, split_size: usize, dim: isize) -> Result<Vec<Self>> {
        if split_size == 0 {
            return Err(TorshError::InvalidArgument(
                "split_size must be greater than 0".to_string(),
            ));
        }

        let ndim = self.ndim();
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

        let dim_size = self.shape().dims()[dim];
        let num_splits = (dim_size + split_size - 1) / split_size;

        let mut result = Vec::new();
        for i in 0..num_splits {
            let start = i * split_size;
            let size = split_size.min(dim_size - start);

            let slice_tensor = self.narrow(dim as isize, start, size)?;
            result.push(slice_tensor);
        }

        Ok(result)
    }

    /// Flip tensor along given dimensions
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.flip(tensor, dims)`
    pub fn flip(&self, dims: &[isize]) -> Result<Self> {
        let ndim = self.ndim();
        let data = self.data()?;
        let shape = self.shape().dims();

        // Normalize dimensions
        let normalized_dims: Result<Vec<usize>> = dims
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
        let normalized_dims = normalized_dims?;

        let total_elements = shape.iter().product();
        let mut result_data = vec![T::default(); total_elements];

        // For each element, compute flipped index
        for i in 0..total_elements {
            let mut indices = vec![0; ndim];
            let mut remaining = i;

            // Convert flat index to multi-dimensional indices
            for d in (0..ndim).rev() {
                let mut stride = 1;
                for dim in d + 1..ndim {
                    stride *= shape[dim];
                }
                indices[d] = remaining / stride;
                remaining %= stride;
            }

            // Flip indices for specified dimensions
            for &dim in &normalized_dims {
                indices[dim] = shape[dim] - 1 - indices[dim];
            }

            // Convert multi-dimensional indices back to flat index
            let mut flipped_idx = 0;
            for d in 0..ndim {
                let mut stride = 1;
                for dim in d + 1..ndim {
                    stride *= shape[dim];
                }
                flipped_idx += indices[d] * stride;
            }

            result_data[i] = data[flipped_idx];
        }

        Self::from_data(result_data, shape.to_vec(), self.device.clone())
    }

    /// Flip tensor left-right (last dimension)
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.fliplr(tensor)`
    pub fn fliplr(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(TorshError::InvalidArgument(
                "fliplr requires at least 2-D tensor".to_string(),
            ));
        }
        self.flip(&[-1])
    }

    /// Flip tensor up-down (first dimension)
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.flipud(tensor)`
    pub fn flipud(&self) -> Result<Self> {
        if self.ndim() < 1 {
            return Err(TorshError::InvalidArgument(
                "flipud requires at least 1-D tensor".to_string(),
            ));
        }
        self.flip(&[0])
    }

    /// Roll tensor along given dimensions
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.roll(tensor, shifts, dims)`
    pub fn roll(&self, shifts: &[isize], dims: &[isize]) -> Result<Self> {
        if shifts.len() != dims.len() {
            return Err(TorshError::InvalidArgument(
                "shifts and dims must have the same length".to_string(),
            ));
        }

        let ndim = self.ndim();
        let data = self.data()?;
        let shape = self.shape().dims();

        // Normalize dimensions
        let normalized_dims: Result<Vec<usize>> = dims
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
        let normalized_dims = normalized_dims?;

        let total_elements = shape.iter().product();
        let mut result_data = vec![T::default(); total_elements];

        // For each element, compute rolled index
        for i in 0..total_elements {
            let mut indices = vec![0; ndim];
            let mut remaining = i;

            // Convert flat index to multi-dimensional indices
            for d in (0..ndim).rev() {
                let mut stride = 1;
                for dim in d + 1..ndim {
                    stride *= shape[dim];
                }
                indices[d] = remaining / stride;
                remaining %= stride;
            }

            // Apply shifts for specified dimensions
            for (shift, &dim) in shifts.iter().zip(&normalized_dims) {
                let dim_size = shape[dim] as isize;
                let shifted = (indices[dim] as isize + shift) % dim_size;
                indices[dim] = if shifted < 0 {
                    (shifted + dim_size) as usize
                } else {
                    shifted as usize
                };
            }

            // Convert multi-dimensional indices back to flat index
            let mut rolled_idx = 0;
            for d in 0..ndim {
                let mut stride = 1;
                for dim in d + 1..ndim {
                    stride *= shape[dim];
                }
                rolled_idx += indices[d] * stride;
            }

            result_data[i] = data[rolled_idx];
        }

        Self::from_data(result_data, shape.to_vec(), self.device.clone())
    }

    /// Rotate tensor 90 degrees in the plane specified by dims
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.rot90(tensor, k, dims)`
    ///
    /// # Arguments
    /// * `k` - Number of times to rotate by 90 degrees
    /// * `dims` - Plane of rotation (must be exactly 2 dimensions)
    pub fn rot90(&self, k: isize, dims: &[isize]) -> Result<Self> {
        if dims.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "dims must specify exactly 2 dimensions".to_string(),
            ));
        }

        let ndim = self.ndim();

        // Normalize dimensions
        let dim0 = if dims[0] < 0 {
            (ndim as isize + dims[0]) as usize
        } else {
            dims[0] as usize
        };
        let dim1 = if dims[1] < 0 {
            (ndim as isize + dims[1]) as usize
        } else {
            dims[1] as usize
        };

        if dim0 >= ndim || dim1 >= ndim {
            return Err(TorshError::InvalidArgument(
                "dims out of range".to_string(),
            ));
        }

        if dim0 == dim1 {
            return Err(TorshError::InvalidArgument(
                "dims must be different".to_string(),
            ));
        }

        // Normalize k to [0, 3]
        let k = ((k % 4) + 4) % 4;

        match k {
            0 => Ok(self.clone()),
            1 => {
                // Rotate 90 degrees: transpose and flip
                let mut perm: Vec<usize> = (0..ndim).collect();
                perm.swap(dim0, dim1);
                let transposed = self.permute(&perm)?;
                transposed.flip(&[dim1 as isize])
            }
            2 => {
                // Rotate 180 degrees: flip both dimensions
                self.flip(&[dim0 as isize, dim1 as isize])
            }
            3 => {
                // Rotate 270 degrees: flip and transpose
                let flipped = self.flip(&[dim0 as isize])?;
                let mut perm: Vec<usize> = (0..ndim).collect();
                perm.swap(dim0, dim1);
                flipped.permute(&perm)
            }
            _ => unreachable!(),
        }
    }

    /// Repeat tensor along dimensions
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.tile(tensor, dims)`
    pub fn tile(&self, repeats: &[usize]) -> Result<Self> {
        let shape_obj = self.shape();
        let shape = shape_obj.dims();
        let ndim = shape.len();

        if repeats.is_empty() {
            return Err(TorshError::InvalidArgument(
                "repeats cannot be empty".to_string(),
            ));
        }

        // If repeats has fewer dimensions, prepend with 1s
        let mut full_repeats = vec![1; ndim.max(repeats.len())];
        let offset = full_repeats.len() - repeats.len();
        full_repeats[offset..].copy_from_slice(repeats);

        // If tensor has fewer dimensions, prepend shape with 1s
        let mut full_shape = vec![1; full_repeats.len()];
        let shape_offset = full_shape.len() - ndim;
        full_shape[shape_offset..].copy_from_slice(shape);

        // Calculate output shape
        let output_shape: Vec<usize> = full_shape
            .iter()
            .zip(&full_repeats)
            .map(|(&s, &r)| s * r)
            .collect();

        // Get data
        let data = self.data()?;
        let total_output = output_shape.iter().product();
        let mut result_data = Vec::with_capacity(total_output);

        // Repeat data
        // Simplified implementation - expand each dimension iteratively
        let mut current_data = data.clone();
        let mut current_shape = full_shape.clone();

        for (dim, &repeat) in full_repeats.iter().enumerate() {
            if repeat == 1 {
                continue;
            }

            let mut new_data = Vec::new();
            let dim_size = current_shape[dim];
            let outer_size: usize = current_shape[..dim].iter().product();
            let inner_size: usize = current_shape[dim + 1..].iter().product();

            for outer in 0..outer_size {
                for _ in 0..repeat {
                    for d in 0..dim_size {
                        for inner in 0..inner_size {
                            let idx =
                                outer * dim_size * inner_size + d * inner_size + inner;
                            new_data.push(current_data[idx]);
                        }
                    }
                }
            }

            current_data = new_data;
            current_shape[dim] *= repeat;
        }

        result_data = current_data;

        Self::from_data(result_data, output_shape, self.device.clone())
    }

    /// Repeat elements of a tensor
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.repeat(tensor, *sizes)`
    ///
    /// # Arguments
    /// * `sizes` - Number of times to repeat along each dimension
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let y = x.repeat(&[4, 2])?; // Shape becomes [4, 6] (repeats 4 times, then each element 2 times)
    /// ```
    pub fn repeat(&self, sizes: &[usize]) -> Result<Self> {
        if sizes.is_empty() {
            return Err(TorshError::InvalidArgument(
                "sizes cannot be empty".to_string(),
            ));
        }

        let shape_obj = self.shape();
        let shape = shape_obj.dims();
        let ndim = shape.len();

        if sizes.len() < ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Number of dimensions of repeat dims ({}) can not be smaller than number of dimensions of tensor ({})",
                sizes.len(),
                ndim
            )));
        }

        // Prepend 1s to shape if needed
        let mut full_shape = shape.to_vec();
        while full_shape.len() < sizes.len() {
            full_shape.insert(0, 1);
        }

        // Calculate output shape
        let output_shape: Vec<usize> = full_shape
            .iter()
            .zip(sizes.iter())
            .map(|(&s, &r)| s * r)
            .collect();

        let data = self.data()?;
        let mut result_data = Vec::new();

        // Expand dimensions iteratively
        let mut current_data = data.clone();
        let mut current_shape = full_shape.clone();

        for (dim, &repeat) in sizes.iter().enumerate() {
            if repeat == 1 {
                continue;
            }

            let mut new_data = Vec::new();
            let dim_size = current_shape[dim];
            let outer_size: usize = current_shape[..dim].iter().product();
            let inner_size: usize = current_shape[dim + 1..].iter().product();

            for outer in 0..outer_size {
                for _ in 0..repeat {
                    for d in 0..dim_size {
                        for inner in 0..inner_size {
                            let idx = outer * dim_size * inner_size + d * inner_size + inner;
                            new_data.push(current_data[idx]);
                        }
                    }
                }
            }

            current_data = new_data;
            current_shape[dim] *= repeat;
        }

        result_data = current_data;

        Self::from_data(result_data, output_shape, self.device.clone())
    }

    /// Repeat elements of a tensor along a dimension
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.repeat_interleave(tensor, repeats, dim)`
    ///
    /// # Arguments
    /// * `repeats` - Number of times to repeat each element
    /// * `dim` - Dimension along which to repeat (None means flatten first)
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let y = x.repeat_interleave(2, Some(0))?; // [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    /// ```
    pub fn repeat_interleave(&self, repeats: usize, dim: Option<isize>) -> Result<Self> {
        if repeats == 0 {
            return Err(TorshError::InvalidArgument(
                "repeats must be positive".to_string(),
            ));
        }

        match dim {
            None => {
                // Flatten and repeat each element
                let data = self.data()?;
                let mut result_data = Vec::with_capacity(data.len() * repeats);

                for &val in data.iter() {
                    for _ in 0..repeats {
                        result_data.push(val);
                    }
                }

                Self::from_data(result_data, vec![data.len() * repeats], self.device.clone())
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

                let shape_obj = self.shape();
                let shape = shape_obj.dims();
                let data = self.data()?;

                // Calculate output shape
                let mut output_shape = shape.to_vec();
                output_shape[dim] *= repeats;

                // Repeat along the specified dimension
                let dim_size = shape[dim];
                let outer_size: usize = shape[..dim].iter().product();
                let inner_size: usize = shape[dim + 1..].iter().product();

                let mut result_data = Vec::with_capacity(data.len() * repeats);

                for outer in 0..outer_size {
                    for d in 0..dim_size {
                        for _ in 0..repeats {
                            for inner in 0..inner_size {
                                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                                result_data.push(data[idx]);
                            }
                        }
                    }
                }

                Self::from_data(result_data, output_shape, self.device.clone())
            }
        }
    }
}
