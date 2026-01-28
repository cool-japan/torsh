//! Tensor manipulation operations
//!
//! This module provides PyTorch-compatible tensor manipulation operations including:
//! - Stacking: stack
//! - Splitting: chunk, split
//! - Flipping: flip, fliplr, flipud
//! - Rolling: roll, rot90
//! - Tiling: tile, repeat_interleave
//! - Utilities: unflatten, take_along_dim

use crate::{Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};

impl<T: TensorElement + Copy + Default> Tensor<T> {
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
        let first_shape = tensors[0].shape().to_vec();
        for tensor in tensors.iter().skip(1) {
            if tensor.shape().dims() != first_shape.as_slice() {
                return Err(TorshError::ShapeMismatch {
                    expected: first_shape.clone(),
                    got: tensor.shape().to_vec(),
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

        // Calculate output shape: insert new dimension at position `dim`
        let mut output_shape = first_shape.to_vec();
        output_shape.insert(dim, tensors.len());

        // Stack tensors by interleaving data
        let elem_count: usize = first_shape.iter().product();
        let mut result_data = Vec::with_capacity(elem_count * tensors.len());

        // Calculate strides for proper data layout
        let outer_size: usize = first_shape[..dim].iter().product();
        let inner_size: usize = first_shape[dim..].iter().product();

        for outer in 0..outer_size {
            for tensor in tensors {
                let data = tensor.to_vec()?;
                for inner in 0..inner_size {
                    let idx = outer * inner_size + inner;
                    result_data.push(data[idx]);
                }
            }
        }

        let device = tensors[0].device.clone();
        Self::from_data(result_data, output_shape, device)
    }

    /// Split tensor into chunks
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
        let mut start = 0;

        while start < dim_size {
            let end = (start + chunk_size).min(dim_size);
            let slice_tensor = self.narrow(dim as i32, start as i64, end - start)?;
            result.push(slice_tensor);
            start = end;
        }

        Ok(result)
    }

    /// Split tensor into parts of given size
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
        let mut result = Vec::new();
        let mut start = 0;

        while start < dim_size {
            let size = split_size.min(dim_size - start);
            let slice_tensor = self.narrow(dim as i32, start as i64, size)?;
            result.push(slice_tensor);
            start += split_size;
        }

        Ok(result)
    }

    /// Flip tensor along given dimensions
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.flip(tensor, dims)`
    pub fn flip(&self, dims: &[isize]) -> Result<Self> {
        if dims.is_empty() {
            return Ok(self.clone());
        }

        let ndim = self.ndim();

        // Normalize dimensions
        let mut norm_dims = Vec::new();
        for &dim in dims {
            let d = if dim < 0 {
                (ndim as isize + dim) as usize
            } else {
                dim as usize
            };

            if d >= ndim {
                return Err(TorshError::InvalidArgument(format!(
                    "Dimension {} out of range for {}-D tensor",
                    dim, ndim
                )));
            }
            norm_dims.push(d);
        }

        // Flip data
        let data = self.to_vec()?;
        let shape = self.shape().to_vec();
        let mut result_data = vec![T::default(); data.len()];

        // Calculate strides
        let mut strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Copy data with flipped indices
        for i in 0..data.len() {
            let mut indices = vec![0; ndim];
            let mut remainder = i;

            for d in 0..ndim {
                indices[d] = remainder / strides[d];
                remainder %= strides[d];
            }

            // Flip specified dimensions
            for &flip_dim in &norm_dims {
                indices[flip_dim] = shape[flip_dim] - 1 - indices[flip_dim];
            }

            // Calculate flipped index
            let mut flipped_idx = 0;
            for d in 0..ndim {
                flipped_idx += indices[d] * strides[d];
            }

            result_data[flipped_idx] = data[i];
        }

        Self::from_data(result_data, shape.to_vec(), self.device)
    }

    /// Flip tensor left-right (last dimension)
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.fliplr(tensor)`
    pub fn fliplr(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(TorshError::InvalidArgument(
                "fliplr requires at least 2 dimensions".to_string(),
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
                "flipud requires at least 1 dimension".to_string(),
            ));
        }
        self.flip(&[0])
    }

    /// Roll tensor elements along given dimensions
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.roll(tensor, shifts, dims)`
    pub fn roll(&self, shifts: &[isize], dims: &[isize]) -> Result<Self> {
        if shifts.len() != dims.len() {
            return Err(TorshError::InvalidArgument(
                "shifts and dims must have the same length".to_string(),
            ));
        }

        if dims.is_empty() {
            // Roll flattened tensor
            let data = self.to_vec()?;
            let shift = if shifts.is_empty() { 0 } else { shifts[0] };
            let n = data.len();
            let shift = ((shift % n as isize) + n as isize) as usize % n;

            let mut result_data = vec![T::default(); n];
            for (i, &val) in data.iter().enumerate() {
                result_data[(i + shift) % n] = val;
            }

            return Self::from_data(result_data, self.shape().dims().to_vec(), self.device);
        }

        let ndim = self.ndim();

        // Normalize dimensions
        let mut norm_dims = Vec::new();
        for &dim in dims {
            let d = if dim < 0 {
                (ndim as isize + dim) as usize
            } else {
                dim as usize
            };

            if d >= ndim {
                return Err(TorshError::InvalidArgument(format!(
                    "Dimension {} out of range for {}-D tensor",
                    dim, ndim
                )));
            }
            norm_dims.push(d);
        }

        let data = self.to_vec()?;
        let shape = self.shape().to_vec();
        let mut result_data = vec![T::default(); data.len()];

        // Calculate strides
        let mut strides = vec![1; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        // Copy data with rolled indices
        for i in 0..data.len() {
            let mut indices = vec![0; ndim];
            let mut remainder = i;

            for d in 0..ndim {
                indices[d] = remainder / strides[d];
                remainder %= strides[d];
            }

            // Roll specified dimensions
            for (dim_idx, &roll_dim) in norm_dims.iter().enumerate() {
                let shift = shifts[dim_idx];
                let dim_size = shape[roll_dim] as isize;
                let rolled =
                    ((indices[roll_dim] as isize + shift) % dim_size + dim_size) % dim_size;
                indices[roll_dim] = rolled as usize;
            }

            // Calculate rolled index
            let mut rolled_idx = 0;
            for d in 0..ndim {
                rolled_idx += indices[d] * strides[d];
            }

            result_data[rolled_idx] = data[i];
        }

        Self::from_data(result_data, shape.to_vec(), self.device)
    }

    /// Rotate tensor 90 degrees
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.rot90(tensor, k, dims)`
    pub fn rot90(&self, k: isize, dims: &[isize]) -> Result<Self> {
        if dims.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "dims must contain exactly 2 dimensions".to_string(),
            ));
        }

        let ndim = self.ndim();
        if ndim < 2 {
            return Err(TorshError::InvalidArgument(
                "rot90 requires at least 2 dimensions".to_string(),
            ));
        }

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
            return Err(TorshError::InvalidArgument("dims out of range".to_string()));
        }

        if dim0 == dim1 {
            return Err(TorshError::InvalidArgument(
                "dims must be different".to_string(),
            ));
        }

        // Normalize k to [0, 4)
        let k = ((k % 4) + 4) % 4;

        let mut result = self.clone();
        for _ in 0..k {
            // Transpose and flip
            result = result.transpose_view(dim0, dim1)?;
            result = result.flip(&[dim1 as isize])?;
        }

        Ok(result)
    }

    /// Tile tensor by repeating
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.tile(tensor, repeats)`
    pub fn tile(&self, repeats: &[usize]) -> Result<Self> {
        if repeats.is_empty() {
            return Ok(self.clone());
        }

        let shape = self.shape().to_vec();
        let ndim = shape.len();

        // Extend shape if needed
        let mut new_shape = shape.to_vec();
        if repeats.len() > ndim {
            let diff = repeats.len() - ndim;
            for _ in 0..diff {
                new_shape.insert(0, 1);
            }
        }

        // Calculate output shape
        let mut output_shape = new_shape.clone();
        let repeat_offset = if repeats.len() < output_shape.len() {
            output_shape.len() - repeats.len()
        } else {
            0
        };

        for (i, &rep) in repeats.iter().enumerate() {
            let idx = repeat_offset + i;
            if idx < output_shape.len() {
                output_shape[idx] *= rep;
            }
        }

        // Tile using existing repeat method from data_ops
        self.repeat(repeats)
    }

    /// Repeat elements of a tensor along a dimension
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.repeat_interleave(tensor, repeats, dim)`
    ///
    /// # Arguments
    /// * `repeats` - Number of times to repeat each element
    /// * `dim` - Dimension along which to repeat (None = flatten first)
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let y = x.repeat_interleave(2, None)?; // [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
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
                let data = self.to_vec()?;
                let mut result_data = Vec::with_capacity(data.len() * repeats);

                for &val in data.iter() {
                    for _ in 0..repeats {
                        result_data.push(val);
                    }
                }

                Self::from_data(result_data, vec![data.len() * repeats], self.device)
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

                let shape = self.shape().to_vec();
                let data = self.to_vec()?;

                // Calculate output shape
                let mut output_shape = shape.clone();
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

                Self::from_data(result_data, output_shape, self.device)
            }
        }
    }

    /// Unflatten a dimension into multiple dimensions
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

        let shape = self.shape().to_vec();
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
        let data = self.to_vec()?;
        Self::from_data(data, new_shape, self.device)
    }

    /// Gather values along a dimension using indices
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.take_along_dim(tensor, indices, dim)`
    ///
    /// # Arguments
    /// * `indices` - Indices to gather
    /// * `dim` - Dimension along which to gather (None = flatten first)
    ///
    /// # Examples
    /// ```ignore
    /// let x = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu)?;
    /// let indices = Tensor::from_data(vec![0i64, 2], vec![2], DeviceType::Cpu)?;
    /// let y = x.take_along_dim(&indices, None)?; // [1.0, 3.0]
    /// ```
    pub fn take_along_dim(&self, indices: &Tensor<i64>, dim: Option<isize>) -> Result<Self> {
        match dim {
            None => {
                // Flatten both tensors and use simple indexing
                let data = self.to_vec()?;
                let idx_data = indices.to_vec()?;

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

                Self::from_data(result, indices.shape().to_vec(), self.device)
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

                let self_shape = self.shape().to_vec();
                let indices_shape = indices.shape().to_vec();

                // Verify shapes match except at the gather dimension
                if self_shape.len() != indices_shape.len() {
                    return Err(TorshError::ShapeMismatch {
                        expected: self_shape.clone(),
                        got: indices_shape.clone(),
                    });
                }

                for (i, (&s, &idx_s)) in self_shape.iter().zip(indices_shape.iter()).enumerate() {
                    if i != dim && s != idx_s {
                        return Err(TorshError::ShapeMismatch {
                            expected: self_shape.clone(),
                            got: indices_shape.clone(),
                        });
                    }
                }

                let data = self.to_vec()?;
                let idx_data = indices.to_vec()?;

                let dim_size = self_shape[dim];
                let outer_size: usize = self_shape[..dim].iter().product();
                let inner_size: usize = self_shape[dim + 1..].iter().product();

                let indices_dim_size = indices_shape[dim];
                let mut result = Vec::with_capacity(idx_data.len());

                for outer in 0..outer_size {
                    for d in 0..indices_dim_size {
                        for inner in 0..inner_size {
                            let idx_flat =
                                outer * indices_dim_size * inner_size + d * inner_size + inner;
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

                Self::from_data(result, indices_shape, self.device)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    // Stack tests
    #[test]
    fn test_stack_1d() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).unwrap();

        let result = Tensor::stack(&[a, b], 0).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);
        let data = result.data().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_stack_negative_dim() {
        let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).unwrap();

        let result = Tensor::stack(&[a, b], -1).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    // Chunk tests
    #[test]
    fn test_chunk_even() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6],
            DeviceType::Cpu,
        )
        .unwrap();

        let chunks = tensor.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape().dims(), &[2]);
        assert_eq!(chunks[0].data().unwrap(), vec![1.0, 2.0]);
        assert_eq!(chunks[1].data().unwrap(), vec![3.0, 4.0]);
        assert_eq!(chunks[2].data().unwrap(), vec![5.0, 6.0]);
    }

    #[test]
    fn test_chunk_uneven() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();

        let chunks = tensor.chunk(2, 0).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape().dims(), &[3]);
        assert_eq!(chunks[1].shape().dims(), &[2]);
    }

    // Split tests
    #[test]
    fn test_split_even() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6],
            DeviceType::Cpu,
        )
        .unwrap();

        let splits = tensor.split(2, 0).unwrap();
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].data().unwrap(), vec![1.0, 2.0]);
        assert_eq!(splits[1].data().unwrap(), vec![3.0, 4.0]);
    }

    #[test]
    fn test_split_uneven() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu).unwrap();

        let splits = tensor.split(2, 0).unwrap();
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].shape().dims(), &[2]);
        assert_eq!(splits[1].shape().dims(), &[2]);
        assert_eq!(splits[2].shape().dims(), &[1]); // Last one smaller
    }

    // Flip tests
    #[test]
    fn test_flip_1d() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();

        let result = tensor.flip(&[0]).unwrap();
        assert_eq!(result.data().unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_2d() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let result = tensor.flip(&[0]).unwrap();
        assert_eq!(result.data().unwrap(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_fliplr() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let result = tensor.fliplr().unwrap();
        assert_eq!(result.data().unwrap(), vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_flipud() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let result = tensor.flipud().unwrap();
        assert_eq!(result.data().unwrap(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    // Roll tests
    #[test]
    fn test_roll_1d() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();

        let result = tensor.roll(&[1], &[0]).unwrap();
        assert_eq!(result.data().unwrap(), vec![4.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roll_negative() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();

        let result = tensor.roll(&[-1], &[0]).unwrap();
        assert_eq!(result.data().unwrap(), vec![2.0, 3.0, 4.0, 1.0]);
    }

    // Rot90 tests
    #[test]
    fn test_rot90_once() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let result = tensor.rot90(1, &[0, 1]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        // After 90° rotation: [[1,2],[3,4]] -> [[2,4],[1,3]]
    }

    #[test]
    fn test_rot90_twice() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let result = tensor.rot90(2, &[0, 1]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        // After 180° rotation: [[1,2],[3,4]] -> [[4,3],[2,1]]
        assert_eq!(result.data().unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    // Tile tests
    #[test]
    fn test_tile_1d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();

        let result = tensor.tile(&[2]).unwrap();
        assert_eq!(result.shape().dims(), &[4]);
        assert_eq!(result.data().unwrap(), vec![1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_tile_2d() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![1, 2], DeviceType::Cpu).unwrap();

        let result = tensor.tile(&[2, 1]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    // Repeat interleave tests
    #[test]
    fn test_repeat_interleave_flatten() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.repeat_interleave(2, None).unwrap();
        assert_eq!(result.shape().dims(), &[6]);
        assert_eq!(result.data().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_repeat_interleave_dim() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let result = tensor.repeat_interleave(2, Some(0)).unwrap();
        assert_eq!(result.shape().dims(), &[4, 2]);
    }

    // Unflatten tests
    #[test]
    fn test_unflatten_basic() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![6],
            DeviceType::Cpu,
        )
        .unwrap();

        let result = tensor.unflatten(0, &[2, 3]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
        assert_eq!(result.data().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_unflatten_multiple_dims() {
        let tensor = Tensor::from_data(vec![1.0f32; 24], vec![24], DeviceType::Cpu).unwrap();

        let result = tensor.unflatten(0, &[2, 3, 4]).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3, 4]);
    }

    // Take along dim tests
    #[test]
    fn test_take_along_dim_flatten() {
        let tensor =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();

        let indices = Tensor::from_data(vec![0i64, 2, 1], vec![3], DeviceType::Cpu).unwrap();

        let result = tensor.take_along_dim(&indices, None).unwrap();
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(result.data().unwrap(), vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_take_along_dim_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let indices =
            Tensor::from_data(vec![0i64, 2, 1, 1, 0, 2], vec![2, 3], DeviceType::Cpu).unwrap();

        let result = tensor.take_along_dim(&indices, Some(1)).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
        // Row 0: [1.0, 2.0, 3.0] indexed by [0, 2, 1] = [1.0, 3.0, 2.0]
        // Row 1: [4.0, 5.0, 6.0] indexed by [1, 0, 2] = [5.0, 4.0, 6.0]
        assert_eq!(result.data().unwrap(), vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);
    }
}
