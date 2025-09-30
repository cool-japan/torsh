//! Data manipulation and access operations for tensors
//!
//! This module provides comprehensive data manipulation operations including
//! element access, filling, copying, gathering, scattering, and repetition operations.
//!
//! # Features
//!
//! - **Element access**: Multi-dimensional and flat indexing operations
//! - **In-place operations**: Fill, copy, zero, and ones operations
//! - **Gather/Scatter**: Advanced indexing operations for data rearrangement
//! - **Repetition**: Repeat and expand operations for broadcasting
//! - **Index utilities**: Helper functions for coordinate conversion

use torsh_core::{
    device::DeviceType,
    dtype::TensorElement,
    error::{Result, TorshError},
};

use crate::core_ops::Tensor;

impl<T: TensorElement + Copy> Tensor<T> {
    /// Create tensor from a scalar value repeated to fill the shape
    pub fn from_scalar(value: T, shape: &[usize], device: DeviceType) -> Result<Self>
    where
        T: Copy,
    {
        let numel = shape.iter().product::<usize>();
        let data = vec![value; numel];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// Fill tensor with a single value (in-place)
    pub fn fill_(&mut self, value: T) -> Result<()>
    where
        T: Copy,
    {
        // For storage-based approach, we need to update all elements
        for i in 0..self.numel() {
            self.storage.set(i, value)?;
        }

        Ok(())
    }

    /// Zero out the tensor (in-place)
    pub fn zero_(&mut self) -> Result<()>
    where
        T: Copy,
    {
        self.fill_(T::zero())
    }

    /// Fill with ones (in-place)
    pub fn ones_(&mut self) -> Result<()>
    where
        T: Copy,
    {
        self.fill_(T::one())
    }

    /// Copy data from another tensor (in-place)
    pub fn copy_(&mut self, other: &Self) -> Result<()>
    where
        T: Copy,
    {
        if self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        let other_data = other.to_vec()?;
        for (i, &value) in other_data.iter().enumerate() {
            self.storage.set(i, value)?;
        }
        Ok(())
    }

    /// Get an element by multi-dimensional index
    pub fn get_item(&self, indices: &[usize]) -> Result<T>
    where
        T: Copy,
    {
        if indices.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        let binding = self.shape();
        let shape = binding.dims();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(TorshError::IndexOutOfBounds {
                    index: idx,
                    size: shape[i],
                });
            }
        }

        let flat_index = self.multi_to_flat_index(indices)?;
        self.get_item_flat(flat_index)
    }

    /// Set an element by multi-dimensional index
    pub fn set_item(&mut self, indices: &[usize], value: T) -> Result<()>
    where
        T: Copy,
    {
        if indices.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        let binding = self.shape();
        let shape = binding.dims();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(TorshError::IndexOutOfBounds {
                    index: idx,
                    size: shape[i],
                });
            }
        }

        let flat_index = self.multi_to_flat_index(indices)?;
        self.set_item_flat(flat_index, value)
    }

    /// Get element by flat index
    pub fn get_item_flat(&self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        if index >= self.numel() {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.numel(),
            });
        }

        self.storage.get(index)
    }

    /// Set element by flat index
    pub fn set_item_flat(&mut self, index: usize, value: T) -> Result<()>
    where
        T: Copy,
    {
        if index >= self.numel() {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.numel(),
            });
        }

        self.storage.set(index, value)
    }

    /// Convert multi-dimensional indices to flat index
    pub fn multi_to_flat_index(&self, indices: &[usize]) -> Result<usize> {
        let binding = self.shape();
        let shape = binding.dims();
        if indices.len() != shape.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Expected {} indices, got {}",
                shape.len(),
                indices.len()
            )));
        }

        let mut flat_index = 0;
        let mut stride = 1;

        for i in (0..indices.len()).rev() {
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }

        Ok(flat_index)
    }

    /// Gather values along an axis using indices
    pub fn gather(&self, dim: usize, indices: &Tensor<i64>) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let self_data = self.to_vec()?;
        let indices_data = indices.to_vec()?;

        let mut result_data = Vec::new();
        let result_shape = indices.shape().dims().to_vec();

        if self.ndim() == 1 {
            // 1D case
            for &index in &indices_data {
                let idx = if index < 0 {
                    (self.shape().dims()[0] as i64 + index) as usize
                } else {
                    index as usize
                };

                if idx >= self.shape().dims()[0] {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {} out of range for tensor with size {}",
                        index,
                        self.shape().dims()[0]
                    )));
                }

                result_data.push(self_data[idx]);
            }
        } else {
            // Multi-dimensional case
            let self_shape_ref = self.shape();
            let self_shape = self_shape_ref.dims();
            let indices_shape_ref = indices.shape();
            let indices_shape = indices_shape_ref.dims();
            let dim_size = self_shape[dim];

            // Calculate strides for both tensors
            let mut self_strides = vec![1; self_shape.len()];
            let mut indices_strides = vec![1; indices_shape.len()];

            for i in (0..self_shape.len() - 1).rev() {
                self_strides[i] = self_strides[i + 1] * self_shape[i + 1];
            }

            for i in (0..indices_shape.len() - 1).rev() {
                indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
            }

            let total_elements = indices_data.len();

            for (i, &index_value) in indices_data.iter().enumerate().take(total_elements) {
                // Convert flat index to multi-dimensional coordinates for indices tensor
                let mut indices_coords = vec![0; indices_shape.len()];
                let mut temp_i = i;
                for j in 0..indices_shape.len() {
                    indices_coords[j] = temp_i / indices_strides[j];
                    temp_i %= indices_strides[j];
                }

                // Get the index value
                let idx = if index_value < 0 {
                    (dim_size as i64 + index_value) as usize
                } else {
                    index_value as usize
                };

                if idx >= dim_size {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {index_value} out of range for dimension {dim} with size {dim_size}"
                    )));
                }

                // Build coordinates for the source tensor
                let mut self_coords = indices_coords.clone();
                if dim < self_coords.len() {
                    self_coords[dim] = idx;
                }

                // Convert coordinates to flat index for source tensor
                let mut flat_idx = 0;
                for j in 0..self_coords.len() {
                    flat_idx += self_coords[j] * self_strides[j];
                }

                result_data.push(self_data[flat_idx]);
            }
        }

        Self::from_data(result_data, result_shape, self.device)
    }

    /// Scatter values along an axis using indices
    pub fn scatter(&self, dim: usize, indices: &Tensor<i64>, src: &Tensor<T>) -> Result<Self> {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let mut result_data = self.to_vec()?;
        let indices_data = indices.to_vec()?;
        let src_data = src.to_vec()?;

        if indices_data.len() != src_data.len() {
            return Err(TorshError::InvalidArgument(
                "Indices and source tensor must have the same number of elements".to_string(),
            ));
        }

        if self.ndim() == 1 {
            // 1D case
            for (i, &index) in indices_data.iter().enumerate() {
                let idx = if index < 0 {
                    (self.shape().dims()[0] as i64 + index) as usize
                } else {
                    index as usize
                };

                if idx >= self.shape().dims()[0] {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {} out of range for tensor with size {}",
                        index,
                        self.shape().dims()[0]
                    )));
                }

                result_data[idx] = src_data[i];
            }
        } else {
            // Multi-dimensional case
            let self_shape_ref = self.shape();
            let self_shape = self_shape_ref.dims();
            let indices_shape_ref = indices.shape();
            let indices_shape = indices_shape_ref.dims();
            let dim_size = self_shape[dim];

            // Calculate strides for both tensors
            let mut self_strides = vec![1; self_shape.len()];
            let mut indices_strides = vec![1; indices_shape.len()];

            for i in (0..self_shape.len() - 1).rev() {
                self_strides[i] = self_strides[i + 1] * self_shape[i + 1];
            }

            for i in (0..indices_shape.len() - 1).rev() {
                indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
            }

            let total_elements = indices_data.len();

            for (i, &index_value) in indices_data.iter().enumerate().take(total_elements) {
                // Convert flat index to multi-dimensional coordinates for indices tensor
                let mut indices_coords = vec![0; indices_shape.len()];
                let mut temp_i = i;
                for j in 0..indices_shape.len() {
                    indices_coords[j] = temp_i / indices_strides[j];
                    temp_i %= indices_strides[j];
                }

                // Get the index value
                let idx = if index_value < 0 {
                    (dim_size as i64 + index_value) as usize
                } else {
                    index_value as usize
                };

                if idx >= dim_size {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {index_value} out of range for dimension {dim} with size {dim_size}"
                    )));
                }

                // Build coordinates for the destination tensor
                let mut self_coords = indices_coords.clone();
                if dim < self_coords.len() {
                    self_coords[dim] = idx;
                }

                // Convert coordinates to flat index for destination tensor
                let mut flat_idx = 0;
                for j in 0..self_coords.len() {
                    flat_idx += self_coords[j] * self_strides[j];
                }

                result_data[flat_idx] = src_data[i];
            }
        }

        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }

    /// Repeat tensor along specified dimensions
    pub fn repeat(&self, repeats: &[usize]) -> Result<Self> {
        if repeats.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Number of repeats {} must match tensor dimensions {}",
                repeats.len(),
                self.ndim()
            )));
        }

        let self_data = self.to_vec()?;
        let shape_binding = self.shape();
        let self_shape = shape_binding.dims();

        // Calculate new shape
        let new_shape: Vec<usize> = self_shape
            .iter()
            .zip(repeats.iter())
            .map(|(&dim, &repeat)| dim * repeat)
            .collect();

        let new_numel = new_shape.iter().product();
        let mut result_data = Vec::with_capacity(new_numel);

        // For each element in the result tensor
        for result_idx in 0..new_numel {
            // Convert flat index to multi-dimensional coordinates in result tensor
            let mut result_coords = vec![0; new_shape.len()];
            let mut temp_idx = result_idx;

            for i in (0..new_shape.len()).rev() {
                result_coords[i] = temp_idx % new_shape[i];
                temp_idx /= new_shape[i];
            }

            // Map to source coordinates
            let source_coords: Vec<usize> = result_coords
                .iter()
                .zip(self_shape.iter())
                .map(|(&result_coord, &dim_size)| result_coord % dim_size)
                .collect();

            // Convert to flat index in source
            let mut source_idx = 0;
            let mut stride = 1;
            for i in (0..self_shape.len()).rev() {
                source_idx += source_coords[i] * stride;
                stride *= self_shape[i];
            }

            result_data.push(self_data[source_idx]);
        }

        Self::from_data(result_data, new_shape, self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_fill_operations() {
        let mut tensor = Tensor::<f32>::zeros(&[2, 3], DeviceType::Cpu).unwrap();

        // Test fill_
        tensor.fill_(5.0).unwrap();
        assert_eq!(tensor.get_item(&[0, 0]).unwrap(), 5.0);
        assert_eq!(tensor.get_item(&[1, 2]).unwrap(), 5.0);

        // Test zero_
        tensor.zero_().unwrap();
        assert_eq!(tensor.get_item(&[0, 0]).unwrap(), 0.0);

        // Test ones_
        tensor.ones_().unwrap();
        assert_eq!(tensor.get_item(&[1, 1]).unwrap(), 1.0);
    }

    #[test]
    fn test_item_access() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).unwrap();

        // Test get_item
        assert_eq!(tensor.get_item(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get_item(&[1, 2]).unwrap(), 6.0);

        // Test set_item
        tensor.set_item(&[0, 1], 10.0).unwrap();
        assert_eq!(tensor.get_item(&[0, 1]).unwrap(), 10.0);

        // Test flat access
        assert_eq!(tensor.get_item_flat(0).unwrap(), 1.0);
        tensor.set_item_flat(0, 15.0).unwrap();
        assert_eq!(tensor.get_item_flat(0).unwrap(), 15.0);
    }

    #[test]
    fn test_gather_1d() {
        let data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let tensor = Tensor::from_data(data, vec![5], DeviceType::Cpu).unwrap();

        let indices = Tensor::from_data(vec![0i64, 2, 4], vec![3], DeviceType::Cpu).unwrap();
        let gathered = tensor.gather(0, &indices).unwrap();

        assert_eq!(gathered.to_vec().unwrap(), vec![10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_scatter_1d() {
        let mut tensor = Tensor::<f32>::zeros(&[5], DeviceType::Cpu).unwrap();
        let indices = Tensor::from_data(vec![1i64, 3], vec![2], DeviceType::Cpu).unwrap();
        let src = Tensor::from_data(vec![100.0f32, 200.0], vec![2], DeviceType::Cpu).unwrap();

        let scattered = tensor.scatter(0, &indices, &src).unwrap();
        let result = scattered.to_vec().unwrap();

        assert_eq!(result[1], 100.0);
        assert_eq!(result[3], 200.0);
        assert_eq!(result[0], 0.0); // unchanged
    }

    #[test]
    fn test_repeat() {
        let data = vec![1.0f32, 2.0];
        let tensor = Tensor::from_data(data, vec![2], DeviceType::Cpu).unwrap();

        let repeated = tensor.repeat(&[3]).unwrap();
        assert_eq!(repeated.shape().dims(), &[6]);
        assert_eq!(
            repeated.to_vec().unwrap(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    #[test]
    fn test_copy_() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let mut tensor1 = Tensor::from_data(data1, vec![3], DeviceType::Cpu).unwrap();

        let data2 = vec![4.0f32, 5.0, 6.0];
        let tensor2 = Tensor::from_data(data2, vec![3], DeviceType::Cpu).unwrap();

        tensor1.copy_(&tensor2).unwrap();
        assert_eq!(tensor1.to_vec().unwrap(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_from_scalar() {
        let tensor = Tensor::<f32>::from_scalar(42.0, &[2, 3], DeviceType::Cpu).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(tensor.get_item(&[i, j]).unwrap(), 42.0);
            }
        }
    }

    #[test]
    fn test_multi_to_flat_index() {
        let tensor = Tensor::<f32>::zeros(&[2, 3, 4], DeviceType::Cpu).unwrap();

        assert_eq!(tensor.multi_to_flat_index(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(tensor.multi_to_flat_index(&[1, 2, 3]).unwrap(), 23);
        assert_eq!(tensor.multi_to_flat_index(&[1, 0, 0]).unwrap(), 12);
    }

    #[test]
    fn test_error_handling() {
        let tensor = Tensor::<f32>::zeros(&[2, 3], DeviceType::Cpu).unwrap();

        // Index out of bounds
        assert!(tensor.get_item(&[2, 0]).is_err());
        assert!(tensor.get_item(&[0, 3]).is_err());

        // Wrong number of indices
        assert!(tensor.get_item(&[0]).is_err());
        assert!(tensor.get_item(&[0, 1, 2]).is_err());
    }
}
