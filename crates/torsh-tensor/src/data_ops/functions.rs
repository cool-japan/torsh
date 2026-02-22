//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::core_ops::Tensor;
use torsh_core::{
    device::DeviceType,
    dtype::TensorElement,
    error::{Result, TorshError},
};

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
            let self_shape_ref = self.shape();
            let self_shape = self_shape_ref.dims();
            let indices_shape_ref = indices.shape();
            let indices_shape = indices_shape_ref.dims();
            let dim_size = self_shape[dim];
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
                let mut indices_coords = vec![0; indices_shape.len()];
                let mut temp_i = i;
                for j in 0..indices_shape.len() {
                    indices_coords[j] = temp_i / indices_strides[j];
                    temp_i %= indices_strides[j];
                }
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
                let mut self_coords = indices_coords.clone();
                if dim < self_coords.len() {
                    self_coords[dim] = idx;
                }
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
            let self_shape_ref = self.shape();
            let self_shape = self_shape_ref.dims();
            let indices_shape_ref = indices.shape();
            let indices_shape = indices_shape_ref.dims();
            let dim_size = self_shape[dim];
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
                let mut indices_coords = vec![0; indices_shape.len()];
                let mut temp_i = i;
                for j in 0..indices_shape.len() {
                    indices_coords[j] = temp_i / indices_strides[j];
                    temp_i %= indices_strides[j];
                }
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
                let mut self_coords = indices_coords.clone();
                if dim < self_coords.len() {
                    self_coords[dim] = idx;
                }
                let mut flat_idx = 0;
                for j in 0..self_coords.len() {
                    flat_idx += self_coords[j] * self_strides[j];
                }
                result_data[flat_idx] = src_data[i];
            }
        }
        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }
    /// Scatter values along an axis using indices and add to existing values
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.scatter_add(tensor, dim, index, src)`
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to index
    /// * `indices` - Index tensor (same shape as src)
    /// * `src` - Source tensor containing values to add
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[5], DeviceType::Cpu)?;
    /// let indices = Tensor::from_data(vec![0i64, 1, 2, 0, 1], vec![5], DeviceType::Cpu)?;
    /// let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)?;
    /// let result = tensor.scatter_add(0, &indices, &src)?;
    /// // result[0] += 1.0 + 4.0 = 5.0
    /// // result[1] += 2.0 + 5.0 = 7.0
    /// // result[2] += 3.0 = 3.0
    /// ```
    pub fn scatter_add(&self, dim: usize, indices: &Tensor<i64>, src: &Tensor<T>) -> Result<Self>
    where
        T: std::ops::Add<Output = T>,
    {
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
                result_data[idx] = result_data[idx] + src_data[i];
            }
        } else {
            let self_shape_ref = self.shape();
            let self_shape = self_shape_ref.dims();
            let indices_shape_ref = indices.shape();
            let indices_shape = indices_shape_ref.dims();
            let dim_size = self_shape[dim];
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
                let mut indices_coords = vec![0; indices_shape.len()];
                let mut temp_i = i;
                for j in 0..indices_shape.len() {
                    indices_coords[j] = temp_i / indices_strides[j];
                    temp_i %= indices_strides[j];
                }
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
                let mut self_coords = indices_coords.clone();
                if dim < self_coords.len() {
                    self_coords[dim] = idx;
                }
                let mut flat_idx = 0;
                for j in 0..self_coords.len() {
                    flat_idx += self_coords[j] * self_strides[j];
                }
                result_data[flat_idx] = result_data[flat_idx] + src_data[i];
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
        let new_shape: Vec<usize> = self_shape
            .iter()
            .zip(repeats.iter())
            .map(|(&dim, &repeat)| dim * repeat)
            .collect();
        let new_numel = new_shape.iter().product();
        let mut result_data = Vec::with_capacity(new_numel);
        for result_idx in 0..new_numel {
            let mut result_coords = vec![0; new_shape.len()];
            let mut temp_idx = result_idx;
            for i in (0..new_shape.len()).rev() {
                result_coords[i] = temp_idx % new_shape[i];
                temp_idx /= new_shape[i];
            }
            let source_coords: Vec<usize> = result_coords
                .iter()
                .zip(self_shape.iter())
                .map(|(&result_coord, &dim_size)| result_coord % dim_size)
                .collect();
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
    /// Add values to tensor at specified indices along a dimension
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.index_add(tensor, dim, index, source, alpha=1.0)`
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to index
    /// * `index` - 1D tensor containing indices
    /// * `source` - Source tensor to add
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[3, 5], DeviceType::Cpu)?;
    /// let index = Tensor::from_data(vec![0i64, 2], vec![2], DeviceType::Cpu)?;
    /// let source = Tensor::ones(&[2, 5], DeviceType::Cpu)?;
    /// let result = tensor.index_add(0, &index, &source)?;
    /// ```
    pub fn index_add(&self, dim: isize, index: &Tensor<i64>, source: &Self) -> Result<Self>
    where
        T: std::ops::Add<Output = T>,
    {
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
        if index.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "index must be 1D tensor".to_string(),
            ));
        }
        let index_size = index.shape().dims()[0];
        let self_shape = self.shape().to_vec();
        let source_shape = source.shape().to_vec();
        if source_shape.len() != self_shape.len() {
            return Err(TorshError::ShapeMismatch {
                expected: self_shape.clone(),
                got: source_shape.clone(),
            });
        }
        for (i, (&s, &src_s)) in self_shape.iter().zip(source_shape.iter()).enumerate() {
            if i == dim {
                if src_s != index_size {
                    return Err(TorshError::InvalidArgument(format!(
                        "source dimension {} size {} must match index size {}",
                        i, src_s, index_size
                    )));
                }
            } else if s != src_s {
                return Err(TorshError::ShapeMismatch {
                    expected: self_shape.clone(),
                    got: source_shape.clone(),
                });
            }
        }
        let mut result_data = self.to_vec()?;
        let source_data = source.to_vec()?;
        let index_data = index.to_vec()?;
        let dim_size = self_shape[dim];
        let outer_size: usize = self_shape[..dim].iter().product();
        let inner_size: usize = self_shape[dim + 1..].iter().product();
        for (src_idx_in_dim, &target_idx) in index_data.iter().enumerate() {
            if target_idx < 0 || target_idx as usize >= dim_size {
                return Err(TorshError::InvalidArgument(format!(
                    "Index {} out of range for dimension size {}",
                    target_idx, dim_size
                )));
            }
            let target_idx = target_idx as usize;
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let result_idx =
                        outer * dim_size * inner_size + target_idx * inner_size + inner;
                    let source_idx =
                        outer * index_size * inner_size + src_idx_in_dim * inner_size + inner;
                    result_data[result_idx] = result_data[result_idx] + source_data[source_idx];
                }
            }
        }
        Self::from_data(result_data, self_shape, self.device)
    }
    /// Copy values from source to tensor at specified indices along a dimension
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.index_copy(tensor, dim, index, source)`
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to index
    /// * `index` - 1D tensor containing indices
    /// * `source` - Source tensor to copy from
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[3, 5], DeviceType::Cpu)?;
    /// let index = Tensor::from_data(vec![0i64, 2], vec![2], DeviceType::Cpu)?;
    /// let source = Tensor::ones(&[2, 5], DeviceType::Cpu)?;
    /// let result = tensor.index_copy(0, &index, &source)?;
    /// ```
    pub fn index_copy(&self, dim: isize, index: &Tensor<i64>, source: &Self) -> Result<Self> {
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
        if index.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "index must be 1D tensor".to_string(),
            ));
        }
        let index_size = index.shape().dims()[0];
        let self_shape = self.shape().to_vec();
        let source_shape = source.shape().to_vec();
        if source_shape.len() != self_shape.len() {
            return Err(TorshError::ShapeMismatch {
                expected: self_shape.clone(),
                got: source_shape.clone(),
            });
        }
        for (i, (&s, &src_s)) in self_shape.iter().zip(source_shape.iter()).enumerate() {
            if i == dim {
                if src_s != index_size {
                    return Err(TorshError::InvalidArgument(format!(
                        "source dimension {} size {} must match index size {}",
                        i, src_s, index_size
                    )));
                }
            } else if s != src_s {
                return Err(TorshError::ShapeMismatch {
                    expected: self_shape.clone(),
                    got: source_shape.clone(),
                });
            }
        }
        let mut result_data = self.to_vec()?;
        let source_data = source.to_vec()?;
        let index_data = index.to_vec()?;
        let dim_size = self_shape[dim];
        let outer_size: usize = self_shape[..dim].iter().product();
        let inner_size: usize = self_shape[dim + 1..].iter().product();
        for (src_idx_in_dim, &target_idx) in index_data.iter().enumerate() {
            if target_idx < 0 || target_idx as usize >= dim_size {
                return Err(TorshError::InvalidArgument(format!(
                    "Index {} out of range for dimension size {}",
                    target_idx, dim_size
                )));
            }
            let target_idx = target_idx as usize;
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let result_idx =
                        outer * dim_size * inner_size + target_idx * inner_size + inner;
                    let source_idx =
                        outer * index_size * inner_size + src_idx_in_dim * inner_size + inner;
                    result_data[result_idx] = source_data[source_idx];
                }
            }
        }
        Self::from_data(result_data, self_shape, self.device)
    }
    /// Fill values in tensor at specified indices along a dimension
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.index_fill(tensor, dim, index, value)`
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to index
    /// * `index` - 1D tensor containing indices
    /// * `value` - Scalar value to fill
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[3, 5], DeviceType::Cpu)?;
    /// let index = Tensor::from_data(vec![0i64, 2], vec![2], DeviceType::Cpu)?;
    /// let result = tensor.index_fill(0, &index, 3.14)?;
    /// ```
    pub fn index_fill(&self, dim: isize, index: &Tensor<i64>, value: T) -> Result<Self> {
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
        if index.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "index must be 1D tensor".to_string(),
            ));
        }
        let mut result_data = self.to_vec()?;
        let index_data = index.to_vec()?;
        let self_shape = self.shape().to_vec();
        let dim_size = self_shape[dim];
        let outer_size: usize = self_shape[..dim].iter().product();
        let inner_size: usize = self_shape[dim + 1..].iter().product();
        for &target_idx in index_data.iter() {
            if target_idx < 0 || target_idx as usize >= dim_size {
                return Err(TorshError::InvalidArgument(format!(
                    "Index {} out of range for dimension size {}",
                    target_idx, dim_size
                )));
            }
            let target_idx = target_idx as usize;
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let result_idx =
                        outer * dim_size * inner_size + target_idx * inner_size + inner;
                    result_data[result_idx] = value;
                }
            }
        }
        Self::from_data(result_data, self_shape, self.device)
    }
    /// Place values at specified flat indices (in-place-like operation, returns new tensor)
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.put_(tensor, indices, values)` but returns new tensor
    ///
    /// # Arguments
    /// * `indices` - 1D tensor of flat indices
    /// * `values` - 1D tensor of values (must match indices length or be broadcastable)
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[3, 3], DeviceType::Cpu)?;  // [[0,0,0],[0,0,0],[0,0,0]]
    /// let indices = Tensor::from_data(vec![0i64, 4, 8], vec![3], DeviceType::Cpu)?;
    /// let values = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let result = tensor.put_(&indices, &values)?;  // [[1,0,0],[0,2,0],[0,0,3]]
    /// ```
    pub fn put_(&self, indices: &Tensor<i64>, values: &Tensor<T>) -> Result<Self> {
        if indices.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "indices must be 1D tensor".to_string(),
            ));
        }
        if values.ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "values must be 1D tensor".to_string(),
            ));
        }
        let indices_data = indices.to_vec()?;
        let values_data = values.to_vec()?;
        if indices_data.len() != values_data.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Number of values {} must match number of indices {}",
                values_data.len(),
                indices_data.len()
            )));
        }
        let mut result_data = self.to_vec()?;
        let numel = self.numel();
        for (i, &index) in indices_data.iter().enumerate() {
            let idx = if index < 0 {
                ((numel as i64) + index) as usize
            } else {
                index as usize
            };
            if idx >= numel {
                return Err(TorshError::InvalidArgument(format!(
                    "Index {} out of range for tensor with {} elements",
                    index, numel
                )));
            }
            result_data[idx] = values_data[i];
        }
        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }
    /// Scatter values from source tensor where mask is true (PyTorch-compatible)
    ///
    /// Copies values from the source tensor to positions where the mask is true.
    /// The mask must have the same shape as self. Source values are taken sequentially
    /// and placed at positions where mask is true.
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.masked_scatter(tensor, mask, source)`
    ///
    /// # Arguments
    /// * `mask` - Boolean tensor with same shape as self
    /// * `source` - Tensor containing values to scatter (must have at least as many elements as true values in mask)
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[3, 3], DeviceType::Cpu)?;
    /// let mask = Tensor::from_data(
    ///     vec![true, false, false, false, true, false, false, false, true],
    ///     vec![3, 3],
    ///     DeviceType::Cpu
    /// )?;
    /// let source = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let result = tensor.masked_scatter(&mask, &source)?;  // [[1,0,0],[0,2,0],[0,0,3]]
    /// ```
    pub fn masked_scatter(&self, mask: &Tensor<bool>, source: &Tensor<T>) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: mask.shape().dims().to_vec(),
            });
        }
        let mask_data = mask.to_vec()?;
        let true_count = mask_data.iter().filter(|&&x| x).count();
        if source.numel() < true_count {
            return Err(TorshError::InvalidArgument(format!(
                "Source tensor has {} elements but need {} for scatter (mask has {} true values)",
                source.numel(),
                true_count,
                true_count
            )));
        }
        let self_data = self.to_vec()?;
        let source_data = source.to_vec()?;
        let mut result_data = Vec::with_capacity(self_data.len());
        let mut source_idx = 0;
        for (i, &self_val) in self_data.iter().enumerate() {
            if i < mask_data.len() && mask_data[i] {
                result_data.push(source_data[source_idx]);
                source_idx += 1;
            } else {
                result_data.push(self_val);
            }
        }
        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }
    /// Multi-dimensional indexed put operation (PyTorch-compatible)
    ///
    /// Places values from source tensor at positions specified by index tensors.
    /// Each index tensor specifies indices along one dimension. Index tensors must
    /// be broadcastable to the same shape.
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.index_put(tensor, indices, values)` where indices is a tuple of index tensors
    ///
    /// # Arguments
    /// * `indices` - Slice of index tensors, one per dimension to index
    /// * `values` - Tensor of values to place (must broadcast to indexed positions)
    ///
    /// # Examples
    /// ```ignore
    /// // 2D example: index_put a 3x3 matrix with row=[0,1] col=[1,2]
    /// let tensor = Tensor::zeros(&[3, 3], DeviceType::Cpu)?;
    /// let row_idx = Tensor::from_data(vec![0i64, 1], vec![2], DeviceType::Cpu)?;
    /// let col_idx = Tensor::from_data(vec![1i64, 2], vec![2], DeviceType::Cpu)?;
    /// let values = Tensor::from_data(vec![10.0f32, 20.0], vec![2], DeviceType::Cpu)?;
    /// let result = tensor.index_put(&[row_idx, col_idx], &values)?;
    /// // result[0,1] = 10.0, result[1,2] = 20.0
    /// ```
    pub fn index_put(&self, indices: &[Tensor<i64>], values: &Tensor<T>) -> Result<Self> {
        if indices.is_empty() {
            return Err(TorshError::InvalidArgument(
                "indices cannot be empty".to_string(),
            ));
        }
        if indices.len() > self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Too many indices ({}) for tensor with {} dimensions",
                indices.len(),
                self.ndim()
            )));
        }
        let index_shape_ref = indices[0].shape();
        let index_shape = index_shape_ref.dims();
        let num_indices = indices[0].numel();
        for idx_tensor in indices.iter() {
            if idx_tensor.shape().dims() != index_shape {
                return Err(TorshError::ShapeMismatch {
                    expected: index_shape.to_vec(),
                    got: idx_tensor.shape().dims().to_vec(),
                });
            }
        }
        if values.numel() != num_indices && values.numel() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Values tensor has {} elements but need {} (or 1 for broadcasting)",
                values.numel(),
                num_indices
            )));
        }
        let mut result_data = self.to_vec()?;
        let self_shape_ref = self.shape();
        let self_shape = self_shape_ref.dims();
        let values_data = values.to_vec()?;
        let index_data: Result<Vec<Vec<i64>>> = indices.iter().map(|idx| idx.to_vec()).collect();
        let index_data = index_data?;
        let mut strides = vec![1; self_shape.len()];
        for i in (0..self_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self_shape[i + 1];
        }
        for i in 0..num_indices {
            let value = if values_data.len() == 1 {
                values_data[0]
            } else {
                values_data[i]
            };
            let mut flat_idx = 0;
            for (dim, idx_vec) in index_data.iter().enumerate() {
                let mut idx = idx_vec[i];
                if idx < 0 {
                    idx += self_shape[dim] as i64;
                }
                if idx < 0 || idx >= self_shape[dim] as i64 {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        idx_vec[i], dim, self_shape[dim]
                    )));
                }
                flat_idx += (idx as usize) * strides[dim];
            }
            result_data[flat_idx] = value;
        }
        Self::from_data(result_data, self_shape.to_vec(), self.device)
    }
    /// Scatter with reduction operation (PyTorch-compatible)
    ///
    /// Generalized scatter operation that applies a reduction operation (sum, prod, mean, etc.)
    /// when scattering values to the same index position.
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.scatter_reduce(tensor, dim, index, src, reduce)`
    ///
    /// # Arguments
    /// * `dim` - Dimension along which to scatter
    /// * `indices` - Index tensor specifying where to scatter values
    /// * `src` - Source tensor containing values to scatter
    /// * `reduce` - Reduction operation ("sum", "prod", "mean", "amax", "amin")
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[5], DeviceType::Cpu)?;
    /// let indices = Tensor::from_data(vec![0i64, 1, 2, 0, 1], vec![5], DeviceType::Cpu)?;
    /// let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], vec![5], DeviceType::Cpu)?;
    /// let result = tensor.scatter_reduce(0, &indices, &src, "sum")?;
    /// // result[0] = 1.0 + 4.0 = 5.0 (sum reduction)
    /// // result[1] = 2.0 + 5.0 = 7.0
    /// ```
    pub fn scatter_reduce(
        &self,
        dim: usize,
        indices: &Tensor<i64>,
        src: &Tensor<T>,
        reduce: &str,
    ) -> Result<Self>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd
            + num_traits::FromPrimitive,
    {
        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                dim,
                self.ndim()
            )));
        }
        if indices.shape() != src.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: indices.shape().dims().to_vec(),
                got: src.shape().dims().to_vec(),
            });
        }
        let indices_data = indices.to_vec()?;
        let src_data = src.to_vec()?;
        let mut result_data = self.to_vec()?;
        let self_shape_ref = self.shape();
        let self_shape = self_shape_ref.dims();
        let mut counts = if reduce == "mean" {
            vec![0usize; result_data.len()]
        } else {
            vec![]
        };
        if self.ndim() == 1 {
            for (i, &index) in indices_data.iter().enumerate() {
                let idx = if index < 0 {
                    (self_shape[0] as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= self_shape[0] {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {} out of bounds for dimension size {}",
                        index, self_shape[0]
                    )));
                }
                result_data[idx] = match reduce {
                    "sum" => result_data[idx] + src_data[i],
                    "prod" => result_data[idx] * src_data[i],
                    "mean" => {
                        counts[idx] += 1;
                        result_data[idx] + src_data[i]
                    }
                    "amax" => {
                        if src_data[i] > result_data[idx] {
                            src_data[i]
                        } else {
                            result_data[idx]
                        }
                    }
                    "amin" => {
                        if src_data[i] < result_data[idx] {
                            src_data[i]
                        } else {
                            result_data[idx]
                        }
                    }
                    _ => {
                        return Err(TorshError::InvalidArgument(format!(
                            "Unknown reduce operation: {}. Supported: sum, prod, mean, amax, amin",
                            reduce
                        )));
                    }
                };
            }
            if reduce == "mean" {
                for (i, count) in counts.iter().enumerate() {
                    if *count > 0 {
                        result_data[i] = T::from_usize(*count)
                            .and_then(|c| Some(result_data[i] / c))
                            .unwrap_or(result_data[i]);
                    }
                }
            }
        } else {
            let dim_size = self_shape[dim];
            let _outer_size: usize = self_shape[..dim].iter().product();
            let _inner_size: usize = self_shape[dim + 1..].iter().product();
            let mut self_strides = vec![1; self_shape.len()];
            for i in (0..self_shape.len() - 1).rev() {
                self_strides[i] = self_strides[i + 1] * self_shape[i + 1];
            }
            let src_shape_ref = src.shape();
            let src_shape = src_shape_ref.dims();
            let mut src_strides = vec![1; src_shape.len()];
            for i in (0..src_shape.len() - 1).rev() {
                src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
            }
            for i in 0..indices_data.len() {
                let index = indices_data[i];
                let idx = if index < 0 {
                    (dim_size as i64 + index) as usize
                } else {
                    index as usize
                };
                if idx >= dim_size {
                    return Err(TorshError::InvalidArgument(format!(
                        "Index {} out of bounds for dimension {} size {}",
                        index, dim, dim_size
                    )));
                }
                let mut coords = vec![0; self_shape.len()];
                let mut remainder = i;
                for (d, &stride) in src_strides.iter().enumerate() {
                    coords[d] = remainder / stride;
                    remainder %= stride;
                }
                coords[dim] = idx;
                let flat_idx = coords
                    .iter()
                    .zip(self_strides.iter())
                    .map(|(c, s)| c * s)
                    .sum::<usize>();
                result_data[flat_idx] = match reduce {
                    "sum" => result_data[flat_idx] + src_data[i],
                    "prod" => result_data[flat_idx] * src_data[i],
                    "mean" => {
                        counts[flat_idx] += 1;
                        result_data[flat_idx] + src_data[i]
                    }
                    "amax" => {
                        if src_data[i] > result_data[flat_idx] {
                            src_data[i]
                        } else {
                            result_data[flat_idx]
                        }
                    }
                    "amin" => {
                        if src_data[i] < result_data[flat_idx] {
                            src_data[i]
                        } else {
                            result_data[flat_idx]
                        }
                    }
                    _ => {
                        return Err(TorshError::InvalidArgument(format!(
                            "Unknown reduce operation: {}",
                            reduce
                        )));
                    }
                };
            }
            if reduce == "mean" {
                for (i, count) in counts.iter().enumerate() {
                    if *count > 0 {
                        result_data[i] = T::from_usize(*count)
                            .and_then(|c| Some(result_data[i] / c))
                            .unwrap_or(result_data[i]);
                    }
                }
            }
        }
        Self::from_data(result_data, self_shape.to_vec(), self.device)
    }
    /// Scatter values to the diagonal (PyTorch-compatible)
    ///
    /// Embeds the values of src tensor into self along the diagonal elements,
    /// with respect to dim1 and dim2. The offset determines which diagonal to use.
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.diagonal_scatter(tensor, src, offset, dim1, dim2)`
    ///
    /// # Arguments
    /// * `src` - Source tensor containing values for the diagonal
    /// * `offset` - Diagonal offset (0=main diagonal, >0=above, <0=below)
    /// * `dim1` - First dimension (default: 0)
    /// * `dim2` - Second dimension (default: 1)
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[3, 3], DeviceType::Cpu)?;
    /// let src = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let result = tensor.diagonal_scatter(&src, 0, 0, 1)?;
    /// // result = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    /// ```
    pub fn diagonal_scatter(
        &self,
        src: &Tensor<T>,
        offset: isize,
        dim1: usize,
        dim2: usize,
    ) -> Result<Self> {
        if dim1 >= self.ndim() || dim2 >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimensions ({}, {}) out of range for {}-dimensional tensor",
                dim1,
                dim2,
                self.ndim()
            )));
        }
        if dim1 == dim2 {
            return Err(TorshError::InvalidArgument(
                "dim1 and dim2 must be different".to_string(),
            ));
        }
        let self_shape_ref = self.shape();
        let self_shape = self_shape_ref.dims();
        let dim1_size = self_shape[dim1];
        let dim2_size = self_shape[dim2];
        let diag_len = if offset >= 0 {
            let offset_u = offset as usize;
            if offset_u >= dim2_size {
                0
            } else {
                std::cmp::min(dim1_size, dim2_size - offset_u)
            }
        } else {
            let offset_u = (-offset) as usize;
            if offset_u >= dim1_size {
                0
            } else {
                std::cmp::min(dim1_size - offset_u, dim2_size)
            }
        };
        if src.numel() != diag_len {
            return Err(TorshError::ShapeMismatch {
                expected: vec![diag_len],
                got: vec![src.numel()],
            });
        }
        let mut result_data = self.to_vec()?;
        let src_data = src.to_vec()?;
        let mut strides = vec![1; self_shape.len()];
        for i in (0..self_shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * self_shape[i + 1];
        }
        for i in 0..diag_len {
            let mut indices = vec![0; self_shape.len()];
            if offset >= 0 {
                indices[dim1] = i;
                indices[dim2] = i + offset as usize;
            } else {
                indices[dim1] = i + (-offset) as usize;
                indices[dim2] = i;
            }
            let mut flat_idx = 0;
            for (d, &idx) in indices.iter().enumerate() {
                flat_idx += idx * strides[d];
            }
            result_data[flat_idx] = src_data[i];
        }
        Self::from_data(result_data, self_shape.to_vec(), self.device)
    }
    /// Scatter values to a selected slice along dimension (PyTorch-compatible)
    ///
    /// Embeds the values of src tensor into self at the given index along dimension dim.
    /// This is the inverse of `select()` operation.
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.select_scatter(tensor, src, dim, index)`
    ///
    /// # Arguments
    /// * `src` - Source tensor to scatter (shape should match self with dim removed)
    /// * `dim` - Dimension along which to select
    /// * `index` - Index position to scatter to
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[3, 4, 5], DeviceType::Cpu)?;
    /// let src = Tensor::ones(&[3, 5], DeviceType::Cpu)?; // dim=1 removed
    /// let result = tensor.select_scatter(&src, 1, 2)?;
    /// // result[:, 2, :] = src
    /// ```
    pub fn select_scatter(&self, src: &Tensor<T>, dim: isize, index: isize) -> Result<Self> {
        let ndim = self.ndim() as isize;
        let dim_normalized = if dim < 0 { ndim + dim } else { dim };
        if dim_normalized < 0 || dim_normalized >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                dim,
                self.ndim()
            )));
        }
        let dim_u = dim_normalized as usize;
        let self_shape_ref = self.shape();
        let self_shape = self_shape_ref.dims();
        let index_normalized = if index < 0 {
            (self_shape[dim_u] as isize) + index
        } else {
            index
        };
        if index_normalized < 0 || index_normalized >= self_shape[dim_u] as isize {
            return Err(TorshError::InvalidArgument(format!(
                "Index {} out of bounds for dimension {} with size {}",
                index, dim_u, self_shape[dim_u]
            )));
        }
        let index_u = index_normalized as usize;
        let expected_src_shape: Vec<usize> = self_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dim_u)
            .map(|(_, &s)| s)
            .collect();
        let src_shape_ref = src.shape();
        let src_shape = src_shape_ref.dims();
        if src_shape != expected_src_shape.as_slice() {
            return Err(TorshError::ShapeMismatch {
                expected: expected_src_shape,
                got: src_shape.to_vec(),
            });
        }
        let mut result_data = self.to_vec()?;
        let src_data = src.to_vec()?;
        let mut self_strides = vec![1; self_shape.len()];
        for i in (0..self_shape.len() - 1).rev() {
            self_strides[i] = self_strides[i + 1] * self_shape[i + 1];
        }
        let outer_size: usize = self_shape[..dim_u].iter().product();
        let inner_size: usize = self_shape[dim_u + 1..].iter().product();
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let self_idx =
                    outer * (self_shape[dim_u] * inner_size) + index_u * inner_size + inner;
                let src_idx = outer * inner_size + inner;
                result_data[self_idx] = src_data[src_idx];
            }
        }
        Self::from_data(result_data, self_shape.to_vec(), self.device)
    }
    /// Scatter values to a slice along dimension (PyTorch-compatible)
    ///
    /// Embeds the values of src tensor into self along dimension dim, starting at
    /// start index, ending at end index, with the given step.
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.slice_scatter(tensor, src, dim, start, end, step)`
    ///
    /// # Arguments
    /// * `src` - Source tensor to scatter
    /// * `dim` - Dimension along which to slice
    /// * `start` - Starting index (None means 0)
    /// * `end` - Ending index (None means size of dim)
    /// * `step` - Step size (default: 1)
    ///
    /// # Examples
    /// ```ignore
    /// let tensor = Tensor::zeros(&[5, 5], DeviceType::Cpu)?;
    /// let src = Tensor::ones(&[2, 5], DeviceType::Cpu)?;
    /// let result = tensor.slice_scatter(&src, 0, Some(1), Some(3), 1)?;
    /// // result[1:3, :] = src
    /// ```
    pub fn slice_scatter(
        &self,
        src: &Tensor<T>,
        dim: isize,
        start: Option<isize>,
        end: Option<isize>,
        step: usize,
    ) -> Result<Self> {
        if step == 0 {
            return Err(TorshError::InvalidArgument(
                "Step must be greater than 0".to_string(),
            ));
        }
        let ndim = self.ndim() as isize;
        let dim_normalized = if dim < 0 { ndim + dim } else { dim };
        if dim_normalized < 0 || dim_normalized >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                dim,
                self.ndim()
            )));
        }
        let dim_u = dim_normalized as usize;
        let self_shape_ref = self.shape();
        let self_shape = self_shape_ref.dims();
        let dim_size = self_shape[dim_u] as isize;
        let start_normalized = start.unwrap_or(0);
        let start_normalized = if start_normalized < 0 {
            dim_size + start_normalized
        } else {
            start_normalized
        };
        let start_normalized = std::cmp::max(0, std::cmp::min(start_normalized, dim_size)) as usize;
        let end_normalized = end.unwrap_or(dim_size);
        let end_normalized = if end_normalized < 0 {
            dim_size + end_normalized
        } else {
            end_normalized
        };
        let end_normalized = std::cmp::max(0, std::cmp::min(end_normalized, dim_size)) as usize;
        let slice_len = if end_normalized > start_normalized {
            (end_normalized - start_normalized + step - 1) / step
        } else {
            0
        };
        let mut expected_src_shape = self_shape.to_vec();
        expected_src_shape[dim_u] = slice_len;
        let src_shape_ref = src.shape();
        let src_shape = src_shape_ref.dims();
        if src_shape != expected_src_shape.as_slice() {
            return Err(TorshError::ShapeMismatch {
                expected: expected_src_shape,
                got: src_shape.to_vec(),
            });
        }
        let mut result_data = self.to_vec()?;
        let src_data = src.to_vec()?;
        let mut self_strides = vec![1; self_shape.len()];
        for i in (0..self_shape.len() - 1).rev() {
            self_strides[i] = self_strides[i + 1] * self_shape[i + 1];
        }
        let outer_size: usize = self_shape[..dim_u].iter().product();
        let inner_size: usize = self_shape[dim_u + 1..].iter().product();
        for outer in 0..outer_size {
            for slice_idx in 0..slice_len {
                let self_dim_idx = start_normalized + slice_idx * step;
                for inner in 0..inner_size {
                    let self_idx = outer * (self_shape[dim_u] * inner_size)
                        + self_dim_idx * inner_size
                        + inner;
                    let src_idx = outer * (slice_len * inner_size) + slice_idx * inner_size + inner;
                    result_data[self_idx] = src_data[src_idx];
                }
            }
        }
        Self::from_data(result_data, self_shape.to_vec(), self.device)
    }
}
