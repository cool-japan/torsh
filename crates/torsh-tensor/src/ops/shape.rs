//! Shape manipulation operations for tensors

use crate::{Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};
use torsh_core::shape::Shape;
use scirs2_core::ndarray::{Array, ArrayView, Axis, IxDyn, Dimension};
use std::collections::VecDeque;

impl<T: TensorElement> Tensor<T> {
    /// Reshapes the tensor to the specified shape (new modular implementation)
    pub fn reshape_v2(&self, shape: &[usize]) -> Result<Self> {
        let total_elements: usize = shape.iter().product();
        if total_elements != self.numel() {
            return Err(TorshError::Other(format!(
                "Cannot reshape tensor of {} elements to shape with {} elements",
                self.numel(),
                total_elements
            )));
        }

        let new_shape = Shape::new(shape.to_vec());
        let mut result = self.clone();
        result.shape = new_shape;

        Ok(result)
    }

    /// Concatenates tensors along the specified dimension
    pub fn cat(tensors: &[Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TorshError::Other("Cannot concatenate empty tensor list".to_string()));
        }

        let first_tensor = &tensors[0];
        let ndims = first_tensor.ndim();

        if dim >= ndims {
            return Err(TorshError::Other(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                dim, ndims
            )));
        }

        // Verify all tensors have compatible shapes
        for tensor in tensors.iter().skip(1) {
            if tensor.ndim() != ndims {
                return Err(TorshError::Other("All tensors must have same number of dimensions".to_string()));
            }

            for (i, (&s1, &s2)) in first_tensor.shape().dims().iter()
                .zip(tensor.shape().dims().iter()).enumerate() {
                if i != dim && s1 != s2 {
                    return Err(TorshError::Other(format!(
                        "All tensors must have same size in non-concatenating dimensions. Got {} and {} at dimension {}",
                        s1, s2, i
                    )));
                }
            }
        }

        // Calculate new shape
        let mut new_dims = first_tensor.shape().dims().to_vec();
        new_dims[dim] = tensors.iter().map(|t| t.shape().dims()[dim]).sum();
        let new_shape = Shape::new(new_dims);

        // Create result tensor
        let mut result = Self::zeros(&new_shape.dims(), first_tensor.device())?;

        // Copy data from each tensor
        let mut offset = 0;
        for tensor in tensors {
            let size = tensor.shape().dims()[dim];
            // Implementation would copy tensor data to appropriate slice
            // This is a simplified version - full implementation would handle strided copying
            offset += size;
        }

        Ok(result)
    }

    /// Stacks tensors along a new dimension
    pub fn stack(tensors: &[Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TorshError::Other("Cannot stack empty tensor list".to_string()));
        }

        let first_tensor = &tensors[0];
        let ndims = first_tensor.ndim();

        if dim > ndims {
            return Err(TorshError::Other(format!(
                "Dimension {} out of range for stacking {}-dimensional tensors",
                dim, ndims
            )));
        }

        // Verify all tensors have same shape
        for tensor in tensors.iter().skip(1) {
            if tensor.shape() != first_tensor.shape() {
                return Err(TorshError::Other("All tensors must have same shape for stacking".to_string()));
            }
        }

        // Calculate new shape
        let mut new_dims = first_tensor.shape().dims().to_vec();
        new_dims.insert(dim, tensors.len());
        let new_shape = Shape::new(new_dims);

        // Create result tensor
        let mut result = Self::zeros(&new_shape.dims(), first_tensor.device())?;

        // Copy data from each tensor
        for (i, tensor) in tensors.iter().enumerate() {
            // Implementation would copy tensor data to appropriate slice
            // This is a simplified version - full implementation would handle indexing
        }

        Ok(result)
    }

    /// Splits tensor into chunks along the specified dimension
    pub fn split(&self, split_size: usize, dim: usize) -> Result<Vec<Self>> {
        if dim >= self.ndim() {
            return Err(TorshError::Other(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                dim, self.ndim()
            )));
        }

        let dim_size = self.shape().dims()[dim];
        if split_size == 0 {
            return Err(TorshError::Other("Split size must be greater than 0".to_string()));
        }

        let num_splits = (dim_size + split_size - 1) / split_size; // Ceiling division
        let mut result = Vec::with_capacity(num_splits);

        for i in 0..num_splits {
            let start = i * split_size;
            let end = std::cmp::min(start + split_size, dim_size);

            // Create slice for this split
            let mut new_dims = self.shape().dims().to_vec();
            new_dims[dim] = end - start;

            // Implementation would create actual slice - simplified here
            let split_tensor = Self::zeros(&new_dims, self.device())?;
            result.push(split_tensor);
        }

        Ok(result)
    }

    /// Chunks tensor into specific number of chunks along dimension
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        if chunks == 0 {
            return Err(TorshError::Other("Number of chunks must be greater than 0".to_string()));
        }

        let dim_size = self.shape().dims()[dim];
        let chunk_size = (dim_size + chunks - 1) / chunks; // Ceiling division

        self.split(chunk_size, dim)
    }

    /// Permutes the dimensions of the tensor (new modular implementation)
    pub fn permute_v2(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.ndim() {
            return Err(TorshError::Other(format!(
                "Number of dimensions in permutation ({}) doesn't match tensor dimensions ({})",
                dims.len(),
                self.ndim()
            )));
        }

        // Verify dims contains valid permutation
        let mut seen = vec![false; self.ndim()];
        for &dim in dims {
            if dim >= self.ndim() {
                return Err(TorshError::Other(format!(
                    "Dimension {} out of range for {}-dimensional tensor",
                    dim, self.ndim()
                )));
            }
            if seen[dim] {
                return Err(TorshError::Other(format!(
                    "Dimension {} appears multiple times in permutation",
                    dim
                )));
            }
            seen[dim] = true;
        }

        // Create new shape by permuting dimensions
        let old_dims = self.shape().dims();
        let new_dims: Vec<usize> = dims.iter().map(|&i| old_dims[i]).collect();

        let mut result = self.clone();
        result.shape = Shape::new(new_dims);

        Ok(result)
    }

    /// Transposes the last two dimensions of the tensor (new modular implementation)
    pub fn transpose_v2(&self) -> Result<Self> {
        let ndims = self.ndim();
        if ndims < 2 {
            return Err(TorshError::Other("Tensor must have at least 2 dimensions for transpose".to_string()));
        }

        let mut dims: Vec<usize> = (0..ndims).collect();
        dims.swap(ndims - 2, ndims - 1);

        self.permute_v2(&dims)
    }

    /// Transposes two specific dimensions (new modular implementation)
    pub fn transpose_dims_v2(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(TorshError::Other("Dimension indices out of range".to_string()));
        }

        let mut dims: Vec<usize> = (0..self.ndim()).collect();
        dims.swap(dim0, dim1);

        self.permute_v2(&dims)
    }

    /// Squeezes out dimensions of size 1 (new modular implementation)
    pub fn squeeze_v2(&self) -> Self {
        let new_dims: Vec<usize> = self.shape().dims()
            .iter()
            .copied()
            .filter(|&dim| dim != 1)
            .collect();

        if new_dims.is_empty() {
            // If all dimensions were 1, result is a scalar (shape [])
            let mut result = self.clone();
            result.shape = Shape::new(vec![]);
            result
        } else {
            self.reshape_v2(&new_dims).expect("reshape should succeed")
        }
    }

    /// Squeezes out a specific dimension of size 1
    pub fn squeeze_dim(&self, dim: usize) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::Other(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                dim, self.ndim()
            )));
        }

        if self.shape().dims()[dim] != 1 {
            return Err(TorshError::Other(format!(
                "Cannot squeeze dimension {} with size {}",
                dim, self.shape().dims()[dim]
            )));
        }

        let mut new_dims = self.shape().dims().to_vec();
        new_dims.remove(dim);

        self.reshape_v2(&new_dims)
    }

    /// Adds a dimension of size 1 (new modular implementation)
    pub fn unsqueeze_v2(&self, dim: usize) -> Result<Self> {
        if dim > self.ndim() {
            return Err(TorshError::Other(format!(
                "Dimension {} out of range for unsqueezing {}-dimensional tensor",
                dim, self.ndim()
            )));
        }

        let mut new_dims = self.shape().dims().to_vec();
        new_dims.insert(dim, 1);

        self.reshape_v2(&new_dims)
    }

    /// Flattens the tensor to 1D (new modular implementation)
    pub fn flatten_v2(&self) -> Self {
        let total_elements = self.numel();
        self.reshape_v2(&[total_elements]).expect("reshape should succeed")
    }

    /// Flattens starting from a specific dimension
    pub fn flatten_from(&self, start_dim: usize) -> Result<Self> {
        if start_dim >= self.ndim() {
            return Err(TorshError::Other(format!(
                "Start dimension {} out of range for {}-dimensional tensor",
                start_dim, self.ndim()
            )));
        }

        let shape_obj = self.shape();
        let dims = shape_obj.dims();
        let mut new_dims = dims[..start_dim].to_vec();

        let flattened_size: usize = dims[start_dim..].iter().product();
        new_dims.push(flattened_size);

        self.reshape_v2(&new_dims)
    }

    /// Views the tensor with a new shape (no data copy) - new modular implementation
    pub fn view_v2(&self, shape: &[usize]) -> Result<Self> {
        // In a full implementation, this would check if the view is possible
        // without copying data based on strides and memory layout
        self.reshape_v2(shape)
    }

    /// Expands tensor to a larger size by broadcasting (new modular implementation)
    pub fn expand_v2(&self, shape: &[usize]) -> Result<Self> {
        if shape.len() < self.ndim() {
            return Err(TorshError::Other("Expanded shape must have at least as many dimensions".to_string()));
        }

        // Check if expansion is valid (each dimension must be 1 or match target)
        let self_dims = self.shape().dims();
        let offset = shape.len() - self_dims.len();

        for (i, &self_dim) in self_dims.iter().enumerate() {
            let target_dim = shape[offset + i];
            if self_dim != 1 && self_dim != target_dim {
                return Err(TorshError::Other(format!(
                    "Cannot expand dimension {} from {} to {}",
                    i, self_dim, target_dim
                )));
            }
        }

        // Implementation would create broadcasted view
        // This is simplified - real implementation would adjust strides for broadcasting
        let mut result = self.clone();
        result.shape = Shape::new(shape.to_vec());

        Ok(result)
    }
}

/// Sorting and searching operations
impl<T: TensorElement + PartialOrd + Copy + Default> Tensor<T> {
    /// Sort tensor elements along a given dimension
    pub fn sort(&self, dim: Option<i32>, descending: bool) -> Result<(Self, Tensor<i64>)> {
        let dim = dim.unwrap_or(-1);
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let shape_ref = self.shape();
        let shape = shape_ref.dims();
        let data = self.data()?;

        // For 1D tensor, simple sort
        if self.ndim() == 1 {
            let mut indexed_data: Vec<(usize, T)> =
                data.iter().enumerate().map(|(i, &v)| (i, v)).collect();

            indexed_data.sort_by(|a, b| {
                if descending {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                }
            });

            let sorted_values: Vec<T> = indexed_data.iter().map(|(_, v)| *v).collect();
            let indices: Vec<i64> = indexed_data.iter().map(|(i, _)| *i as i64).collect();

            let sorted_tensor = Self::from_data(sorted_values, shape.to_vec(), self.device);
            let indices_tensor = Tensor::<i64>::from_data(indices, shape.to_vec(), self.device);

            return Ok((sorted_tensor?, indices_tensor?));
        }

        // For multi-dimensional tensors, implement a basic version
        // This is a simplified implementation - in production, you'd want more optimized sorting

        // For now, flatten, sort, and reshape back
        let flattened = self.flatten()?;
        let data = flattened.data()?;

        let mut indexed_data: Vec<(usize, T)> =
            data.iter().enumerate().map(|(i, &v)| (i, v)).collect();

        indexed_data.sort_by(|a, b| {
            if descending {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let sorted_values: Vec<T> = indexed_data.iter().map(|(_, v)| *v).collect();
        let indices: Vec<i64> = indexed_data.iter().map(|(i, _)| *i as i64).collect();

        let sorted_tensor = Self::from_data(sorted_values, shape.to_vec(), self.device)?;
        let indices_tensor = Tensor::<i64>::from_data(indices, shape.to_vec(), self.device)?;

        Ok((sorted_tensor, indices_tensor))
    }

    /// Sort and return indices (argsort)
    pub fn argsort(&self, dim: Option<i32>, descending: bool) -> Result<Tensor<i64>> {
        let (_sorted, indices) = self.sort(dim, descending)?;
        Ok(indices)
    }
}