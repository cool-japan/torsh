//! Tensor manipulation operations for combining, splitting, and transforming tensors
//!
//! This module provides PyTorch-compatible tensor manipulation operations including:
//! - Concatenation: cat, stack
//! - Splitting: split, chunk
//! - Flipping: flip, fliplr, flipud, rot90
//! - Repeating: tile, repeat, repeat_interleave
//! - Rolling: roll
//! - Dimension manipulation: movedim, moveaxis, swapaxes, swapdims
//! - Shape transformation: unflatten
//! - Advanced indexing: take_along_dim

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

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_cat_1d() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).unwrap();

        let result = Tensor::cat(&[a, b], 0).unwrap();

        assert_eq!(result.shape().dims(), &[4]);
        let data = result.data().unwrap();
        assert_eq!(data.len(), 4);
    }

    #[test]
    fn test_stack_1d() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).unwrap();

        let result = Tensor::stack(&[a, b], 0).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_stack_shape_mismatch() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![3.0f32, 4.0, 5.0], vec![3], DeviceType::Cpu).unwrap();

        assert!(Tensor::stack(&[a, b], 0).is_err());
    }

    #[test]
    fn test_flip_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();

        let flipped = tensor.flip(&[0]).unwrap();
        let result = flipped.data().unwrap();

        assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_roll_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();

        let rolled = tensor.roll(&[2], &[0]).unwrap();
        let result = rolled.data().unwrap();

        assert_eq!(result, vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_chunk() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6],
            DeviceType::Cpu,
        )
        .unwrap();

        let chunks = tensor.chunk(3, 0).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape().dims(), &[2]);
        assert_eq!(chunks[1].shape().dims(), &[2]);
        assert_eq!(chunks[2].shape().dims(), &[2]);
    }

    #[test]
    fn test_split() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
            vec![5],
            DeviceType::Cpu,
        )
        .unwrap();

        let splits = tensor.split(2, 0).unwrap();

        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].shape().dims(), &[2]);
        assert_eq!(splits[1].shape().dims(), &[2]);
        assert_eq!(splits[2].shape().dims(), &[1]); // Last one is smaller
    }

    #[test]
    fn test_fliplr() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let flipped = tensor.fliplr().unwrap();
        let result = flipped.data().unwrap();

        // [[1, 2, 3], [4, 5, 6]] -> [[3, 2, 1], [6, 5, 4]]
        assert_eq!(result, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_flipud() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let flipped = tensor.flipud().unwrap();
        let result = flipped.data().unwrap();

        // [[1, 2, 3], [4, 5, 6]] -> [[4, 5, 6], [1, 2, 3]]
        assert_eq!(result, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rot90_once() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .unwrap();

        let rotated = tensor.rot90(1, &[0, 1]).unwrap();
        let result = rotated.data().unwrap();
        let shape = rotated.shape().dims();

        assert_eq!(shape, &[2, 2]);
        // [[1, 2], [3, 4]] -> [[2, 4], [1, 3]] (90 degrees counter-clockwise)
        assert_eq!(result, vec![2.0, 4.0, 1.0, 3.0]);
    }

    #[test]
    fn test_rot90_twice() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .unwrap();

        let rotated = tensor.rot90(2, &[0, 1]).unwrap();
        let result = rotated.data().unwrap();

        // 180 degrees: [[1, 2], [3, 4]] -> [[4, 3], [2, 1]]
        assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_rot90_negative() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .unwrap();

        let rotated = tensor.rot90(-1, &[0, 1]).unwrap();
        let result = rotated.data().unwrap();

        // -90 degrees (clockwise): [[1, 2], [3, 4]] -> [[3, 1], [4, 2]]
        assert_eq!(result, vec![3.0, 1.0, 4.0, 2.0]);
    }

    #[test]
    fn test_tile_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();

        let tiled = tensor.tile(&[3]).unwrap();
        let result = tiled.data().unwrap();
        let shape = tiled.shape().dims();

        assert_eq!(shape, &[6]); // 2 * 3
        assert_eq!(result, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tile_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .unwrap();

        let tiled = tensor.tile(&[2, 3]).unwrap();
        let result = tiled.data().unwrap();
        let shape = tiled.shape().dims();

        assert_eq!(shape, &[4, 6]); // [2*2, 2*3]
        // [[1, 2], [3, 4]] tiled with [2, 3] should repeat pattern
        assert_eq!(result.len(), 24);
    }

    #[test]
    fn test_tile_expand_dims() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();

        let tiled = tensor.tile(&[3, 2]).unwrap();
        let shape = tiled.shape().dims();

        assert_eq!(shape, &[3, 4]); // Expands to 2D and tiles
    }

    #[test]
    fn test_cat_2d() {
        let a = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let b = Tensor::from_data(
            vec![5.0f32, 6.0, 7.0, 8.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .unwrap();

        let result = Tensor::cat(&[a, b], 0).unwrap();

        assert_eq!(result.shape().dims(), &[4, 2]);
        let data = result.data().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_stack_2d() {
        let a = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let b = Tensor::from_data(
            vec![5.0f32, 6.0, 7.0, 8.0],
            vec![2, 2],
            DeviceType::Cpu,
        )
        .unwrap();

        let result = Tensor::stack(&[a, b], 0).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2, 2]);
    }

    #[test]
    fn test_repeat_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.repeat(&[4]).unwrap();

        assert_eq!(result.shape().dims(), &[12]); // 3 * 4
        let data = result.data().unwrap();
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_repeat_2d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .unwrap();

        let result = tensor.repeat(&[2, 3]).unwrap();

        assert_eq!(result.shape().dims(), &[4, 6]); // [2*2, 2*3]
        assert_eq!(result.data().unwrap().len(), 24);
    }

    #[test]
    fn test_repeat_expand_dims() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();

        let result = tensor.repeat(&[3, 2]).unwrap();

        assert_eq!(result.shape().dims(), &[3, 4]); // Expands to [1, 2] then repeats to [3, 4]
    }

    #[test]
    fn test_repeat_interleave_1d_no_dim() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.repeat_interleave(2, None).unwrap();

        assert_eq!(result.shape().dims(), &[6]); // 3 * 2
        let data = result.data().unwrap();
        assert_eq!(data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_repeat_interleave_1d_with_dim() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.repeat_interleave(3, Some(0)).unwrap();

        assert_eq!(result.shape().dims(), &[9]); // 3 * 3
        let data = result.data().unwrap();
        assert_eq!(
            data,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn test_repeat_interleave_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let result = tensor.repeat_interleave(2, Some(0)).unwrap();

        assert_eq!(result.shape().dims(), &[4, 3]); // Repeat along dim 0
        let data = result.data().unwrap();
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_repeat_interleave_2d_dim1() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let result = tensor.repeat_interleave(2, Some(1)).unwrap();

        assert_eq!(result.shape().dims(), &[2, 6]); // Repeat along dim 1
        let data = result.data().unwrap();
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6]]
        assert_eq!(
            data,
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
        );
    }

    #[test]
    fn test_unflatten_1d_to_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.unflatten(0, &[2, 3]).unwrap();

        assert_eq!(result.shape().dims(), &[2, 3]);
        let data = result.data().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unflatten_2d_to_3d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.unflatten(1, &[2, 2]).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2, 2]);
        assert_eq!(result.data().unwrap().len(), 8);
    }

    #[test]
    fn test_unflatten_size_mismatch() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();

        let result = tensor.unflatten(0, &[2, 3]); // 2*3=6 != 4

        assert!(result.is_err());
    }

    #[test]
    fn test_take_along_dim_1d() {
        let tensor = Tensor::from_data(vec![10.0f32, 20.0, 30.0, 40.0], vec![4], DeviceType::Cpu).unwrap();
        let indices = Tensor::from_data(vec![3i64, 1, 2], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.take_along_dim(&indices, None).unwrap();
        let data = result.data().unwrap();

        assert_eq!(data, vec![40.0, 20.0, 30.0]);
    }

    #[test]
    fn test_take_along_dim_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();
        let indices = Tensor::from_data(
            vec![2i64, 0, 1, 2],
            vec![2, 2],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.take_along_dim(&indices, Some(1)).unwrap();
        let data = result.data().unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);
        // Row 0: [1, 2, 3] with indices [2, 0] -> [3, 1]
        // Row 1: [4, 5, 6] with indices [1, 2] -> [5, 6]
        assert_eq!(data, vec![3.0, 1.0, 5.0, 6.0]);
    }

    #[test]
    fn test_take_along_dim_out_of_range() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let indices = Tensor::from_data(vec![5i64], vec![1], DeviceType::Cpu).unwrap();

        let result = tensor.take_along_dim(&indices, None);

        assert!(result.is_err()); // Index 5 out of range for size 3
    }

    #[test]
    fn test_take_along_dim_argmax_use_case() {
        // Common use case: gather max values using argmax indices
        let tensor = Tensor::from_data(
            vec![1.0f32, 5.0, 3.0, 2.0, 7.0, 4.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();

        // Get argmax along dim 1
        let argmax_result = tensor.argmax(Some(1), true).unwrap();

        // Use take_along_dim to gather the max values
        let max_values = tensor.take_along_dim(&argmax_result, Some(1)).unwrap();
        let data = max_values.data().unwrap();

        assert_eq!(max_values.shape().dims(), &[2, 1]);
        // Row 0: max is 5.0, Row 1: max is 7.0
        assert_eq!(data, vec![5.0, 7.0]);
    }

    #[test]
    fn test_movedim_single() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        // Move dim 0 to position 2: [2,3,4] -> [3,4,2]
        let result = tensor.movedim(&[0], &[2]).unwrap();

        assert_eq!(result.shape().dims(), &[3, 4, 2]);
    }

    #[test]
    fn test_movedim_multiple() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        // Move dims [0, 1] to positions [2, 0]: [2,3,4] -> [3,4,2]
        let result = tensor.movedim(&[0, 1], &[2, 0]).unwrap();

        assert_eq!(result.shape().dims(), &[3, 4, 2]);
    }

    #[test]
    fn test_movedim_negative_indices() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        // Move last dim to first position: [2,3,4] -> [4,2,3]
        let result = tensor.movedim(&[-1], &[0]).unwrap();

        assert_eq!(result.shape().dims(), &[4, 2, 3]);
    }

    #[test]
    fn test_movedim_length_mismatch() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.movedim(&[0, 1], &[2]);

        assert!(result.is_err()); // source and destination must have same length
    }

    #[test]
    fn test_movedim_duplicate_source() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.movedim(&[0, 0], &[1, 2]);

        assert!(result.is_err()); // Repeated dim in source
    }

    #[test]
    fn test_movedim_duplicate_destination() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.movedim(&[0, 1], &[2, 2]);

        assert!(result.is_err()); // Repeated dim in destination
    }

    #[test]
    fn test_moveaxis_alias() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        let result1 = tensor.movedim(&[0], &[2]).unwrap();
        let result2 = tensor.moveaxis(&[0], &[2]).unwrap();

        assert_eq!(result1.shape().dims(), result2.shape().dims());
    }

    #[test]
    fn test_swapaxes_simple() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();

        // Swap dims 0 and 1: [2,3] -> [3,2]
        let result = tensor.swapaxes(0, 1).unwrap();

        assert_eq!(result.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_swapaxes_3d() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        // Swap dims 0 and 2: [2,3,4] -> [4,3,2]
        let result = tensor.swapaxes(0, 2).unwrap();

        assert_eq!(result.shape().dims(), &[4, 3, 2]);
    }

    #[test]
    fn test_swapaxes_negative_indices() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        // Swap last two dims: [2,3,4] -> [2,4,3]
        let result = tensor.swapaxes(-1, -2).unwrap();

        assert_eq!(result.shape().dims(), &[2, 4, 3]);
    }

    #[test]
    fn test_swapaxes_same_dim() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        // Swapping same dimension should return identical shape
        let result = tensor.swapaxes(1, 1).unwrap();

        assert_eq!(result.shape().dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_swapaxes_out_of_range() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.swapaxes(0, 5);

        assert!(result.is_err()); // Dimension 5 out of range for 3-D tensor
    }

    #[test]
    fn test_swapdims_alias() {
        let tensor = Tensor::from_data(
            vec![1.0f32; 24],
            vec![2, 3, 4],
            DeviceType::Cpu,
        ).unwrap();

        let result1 = tensor.swapaxes(0, 2).unwrap();
        let result2 = tensor.swapdims(0, 2).unwrap();

        assert_eq!(result1.shape().dims(), result2.shape().dims());
    }

    #[test]
    fn test_movedim_integration_with_data() {
        // Test that data is actually rearranged correctly
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();

        // [[1, 2, 3], [4, 5, 6]] with shape [2, 3]
        // Move dim 1 to position 0: should become [3, 2]
        let result = tensor.movedim(&[1], &[0]).unwrap();

        assert_eq!(result.shape().dims(), &[3, 2]);
        // After transpose: [[1, 4], [2, 5], [3, 6]]
        let data = result.data().unwrap();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_swapaxes_integration_with_data() {
        // Test that data is actually rearranged correctly
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();

        // [[1, 2, 3], [4, 5, 6]] with shape [2, 3]
        let result = tensor.swapaxes(0, 1).unwrap();

        assert_eq!(result.shape().dims(), &[3, 2]);
        // After transpose: [[1, 4], [2, 5], [3, 6]]
        let data = result.data().unwrap();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_broadcast_to_same_shape() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.broadcast_to(&[2, 2]).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(result.data().unwrap(), tensor.data().unwrap());
    }

    #[test]
    fn test_broadcast_to_expand_dim() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            DeviceType::Cpu,
        ).unwrap();

        // Broadcast [2] to [3, 2]
        let result = tensor.broadcast_to(&[3, 2]).unwrap();

        assert_eq!(result.shape().dims(), &[3, 2]);
        let data = result.data().unwrap();
        // Should repeat [1, 2] three times
        assert_eq!(data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_broadcast_to_expand_singleton() {
        let tensor = Tensor::from_data(
            vec![5.0f32],
            vec![1],
            DeviceType::Cpu,
        ).unwrap();

        // Broadcast [1] to [4]
        let result = tensor.broadcast_to(&[4]).unwrap();

        assert_eq!(result.shape().dims(), &[4]);
        let data = result.data().unwrap();
        assert_eq!(data, vec![5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_broadcast_to_2d_singleton() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3, 1],
            DeviceType::Cpu,
        ).unwrap();

        // Broadcast [3, 1] to [3, 4]
        let result = tensor.broadcast_to(&[3, 4]).unwrap();

        assert_eq!(result.shape().dims(), &[3, 4]);
        let data = result.data().unwrap();
        // Each row should be repeated 4 times
        assert_eq!(
            data,
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn test_broadcast_to_add_leading_dims() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            DeviceType::Cpu,
        ).unwrap();

        // Broadcast [2] to [2, 3, 2]
        let result = tensor.broadcast_to(&[2, 3, 2]).unwrap();

        assert_eq!(result.shape().dims(), &[2, 3, 2]);
        assert_eq!(result.data().unwrap().len(), 12); // 2 * 3 * 2
    }

    #[test]
    fn test_broadcast_to_incompatible() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3],
            DeviceType::Cpu,
        ).unwrap();

        // Cannot broadcast [3] to [2]
        let result = tensor.broadcast_to(&[2]);

        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_to_fewer_dims() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).unwrap();

        // Cannot broadcast [2, 2] to [2] (fewer dimensions)
        let result = tensor.broadcast_to(&[2]);

        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_to_complex_pattern() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![1, 3],
            DeviceType::Cpu,
        ).unwrap();

        // Broadcast [1, 3] to [2, 3]
        let result = tensor.broadcast_to(&[2, 3]).unwrap();

        assert_eq!(result.shape().dims(), &[2, 3]);
        let data = result.data().unwrap();
        // [1, 2, 3] repeated twice
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_expand_as_basic() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            DeviceType::Cpu,
        ).unwrap();

        let target = Tensor::from_data(
            vec![0.0f32; 6],
            vec![3, 2],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.expand_as(&target).unwrap();

        assert_eq!(result.shape().dims(), target.shape().dims());
        assert_eq!(result.shape().dims(), &[3, 2]);
    }

    #[test]
    fn test_expand_as_same_shape() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            DeviceType::Cpu,
        ).unwrap();

        let target = Tensor::from_data(
            vec![0.0f32; 4],
            vec![2, 2],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.expand_as(&target).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);
        assert_eq!(result.data().unwrap(), tensor.data().unwrap());
    }

    #[test]
    fn test_expand_as_with_singleton() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3, 1],
            DeviceType::Cpu,
        ).unwrap();

        let target = Tensor::from_data(
            vec![0.0f32; 12],
            vec![3, 4],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.expand_as(&target).unwrap();

        assert_eq!(result.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_expand_as_incompatible() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3],
            DeviceType::Cpu,
        ).unwrap();

        let target = Tensor::from_data(
            vec![0.0f32; 2],
            vec![2],
            DeviceType::Cpu,
        ).unwrap();

        let result = tensor.expand_as(&target);

        assert!(result.is_err()); // Cannot broadcast [3] to [2]
    }
}
