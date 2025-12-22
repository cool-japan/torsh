//! Advanced tensor operations including reductions, linear algebra, and backend integration
//!
//! This module provides high-level tensor operations including reductions, linear algebra
//! operations, SciRS2 backend integration, and advanced data manipulation functions.
//!
//! # Features
//!
//! - **Reductions**: max, norm, sum, mean operations
//! - **Linear algebra**: Matrix multiplication and vector operations
//! - **SciRS2 integration**: Optimized backend operations for performance
//! - **Activation functions**: ReLU, sigmoid, tanh through SciRS2 backend
//! - **Functional programming**: Apply operations and data transformations
//! - **Memory management**: Copy-on-write semantics and unique data operations

use std::sync::Arc;
use torsh_core::{
    device::DeviceType,
    dtype::{FloatElement, TensorElement},
    error::{Result, TorshError},
};

use crate::{core_ops::Tensor, storage::TensorStorage};

// Float-specific operations
impl<T: FloatElement + Copy> Tensor<T> {
    /// Create a 0-dimensional tensor (scalar) from a single value
    pub fn scalar(value: T) -> Result<Self> {
        Self::from_data(vec![value], vec![], DeviceType::Cpu)
    }

    /// Convert tensor to ndarray (temporary placeholder implementation)
    ///
    /// TODO: Implement proper ndarray conversion following SciRS2 POLICY
    /// This should use scirs2_core::ndarray for array operations
    pub fn as_ndarray(&self) -> Result<scirs2_core::ndarray::ArrayD<T>> {
        use scirs2_core::ndarray::ArrayD;
        let data = self.data()?;
        let shape_obj = self.shape().clone();
        let shape = shape_obj.dims();
        ArrayD::from_shape_vec(shape, data.to_vec())
            .map_err(|e| TorshError::InvalidShape(format!("ndarray conversion failed: {}", e)))
    }

    /// Create tensor from ndarray (temporary placeholder implementation)
    ///
    /// TODO: Implement proper ndarray conversion following SciRS2 POLICY
    /// This should use scirs2_core::ndarray for array operations
    pub fn from_ndarray(
        array: scirs2_core::ndarray::ArrayD<T>,
        device: DeviceType,
    ) -> Result<Self> {
        let shape = array.shape().to_vec();
        let (data, _offset) = array.into_raw_vec_and_offset();
        Self::from_data(data, shape, device)
    }

    /// Maximum element in tensor
    pub fn max(&self, dim: Option<usize>, keepdim: bool) -> Result<Self> {
        match dim {
            None => {
                // Global maximum
                let data = self.to_vec()?;
                let max_val =
                    data.into_iter()
                        .fold(<T as FloatElement>::neg_infinity(), |acc, x| {
                            if x > acc {
                                x
                            } else {
                                acc
                            }
                        });
                if keepdim {
                    let shape = vec![1; self.shape().dims().len()];
                    Self::from_data(vec![max_val], shape, self.device)
                } else {
                    Self::scalar(max_val)
                }
            }
            Some(axis) => {
                // Maximum along specific dimension
                let shape_binding = self.shape();
                let input_shape = shape_binding.dims();

                if axis >= input_shape.len() {
                    return Err(TorshError::InvalidOperation(format!(
                        "Axis {} out of bounds for {}-dimensional tensor",
                        axis,
                        input_shape.len()
                    )));
                }

                // Calculate output shape
                let mut output_shape = input_shape.to_vec();
                if keepdim {
                    output_shape[axis] = 1;
                } else {
                    output_shape.remove(axis);
                }

                let data = self.data()?;
                let outer_size: usize = input_shape[..axis].iter().product();
                let axis_size = input_shape[axis];
                let inner_size: usize = input_shape[axis + 1..].iter().product();

                let output_size = outer_size * inner_size;
                let mut result_data = vec![<T as FloatElement>::neg_infinity(); output_size];

                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        let mut max_val = <T as FloatElement>::neg_infinity();
                        for a in 0..axis_size {
                            let input_idx = outer * axis_size * inner_size + a * inner_size + inner;
                            let val = data[input_idx];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                        let output_idx = outer * inner_size + inner;
                        result_data[output_idx] = max_val;
                    }
                }

                Self::from_data(result_data, output_shape, self.device)
            }
        }
    }

    /// Maximum along specified dimension
    pub fn max_dim(&self, dim: i32, keepdim: bool) -> Result<Self> {
        let shape_binding = self.shape();
        let input_shape = shape_binding.dims();

        let actual_dim = if dim < 0 {
            (input_shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= input_shape.len() {
            return Err(TorshError::InvalidOperation(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                actual_dim,
                input_shape.len()
            )));
        }

        // Calculate output shape
        let mut output_shape = input_shape.to_vec();
        if keepdim {
            output_shape[actual_dim] = 1;
        } else {
            output_shape.remove(actual_dim);
        }

        let data = self.data()?;
        let outer_size: usize = input_shape[..actual_dim].iter().product();
        let dim_size = input_shape[actual_dim];
        let inner_size: usize = input_shape[actual_dim + 1..].iter().product();

        let output_size = outer_size * inner_size;
        let mut result_data = vec![<T as FloatElement>::neg_infinity(); output_size];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut max_val = <T as FloatElement>::neg_infinity();
                for d in 0..dim_size {
                    let input_idx = outer * dim_size * inner_size + d * inner_size + inner;
                    let val = data[input_idx];
                    if val > max_val {
                        max_val = val;
                    }
                }
                let output_idx = outer * inner_size + inner;
                result_data[output_idx] = max_val;
            }
        }

        Self::from_data(result_data, output_shape, self.device)
    }

    /// Minimum along specified dimension
    pub fn min_dim(&self, dim: i32, keepdim: bool) -> Result<Self> {
        use scirs2_core::ndarray::Axis;

        let normalized_dim = if dim < 0 {
            (self.shape().len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= self.shape().len() {
            return Err(torsh_core::error::TorshError::InvalidDimension {
                dim: normalized_dim,
                ndim: self.shape().len(),
            });
        }

        let array = self.as_ndarray()?;
        let result = array.map_axis(Axis(normalized_dim), |view| {
            view.iter()
                .copied()
                .fold(<T as FloatElement>::infinity(), |acc, x| {
                    if x < acc {
                        x
                    } else {
                        acc
                    }
                })
        });

        let result_shape = if keepdim {
            let mut shape = self.shape().to_vec();
            shape[normalized_dim] = 1;
            shape
        } else {
            result.shape().to_vec()
        };

        Self::from_ndarray(
            result
                .to_shape(result_shape)
                .map_err(|e| TorshError::InvalidShape(format!("Shape conversion failed: {}", e)))?
                .to_owned(),
            self.device(),
        )
    }
}

/// Boolean reduction operations for tensors
impl<T: TensorElement + Copy> Tensor<T>
where
    T: PartialEq + num_traits::Zero,
{
    /// Check if all elements are non-zero (true)
    pub fn all(&self) -> Result<Tensor<bool>> {
        let data = self.to_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let all_true = data.iter().all(|&x| x != zero);
        Tensor::from_data(vec![all_true], vec![], self.device())
    }

    /// Check if any element is non-zero (true)
    pub fn any(&self) -> Result<Tensor<bool>> {
        let data = self.to_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let any_true = data.iter().any(|&x| x != zero);
        Tensor::from_data(vec![any_true], vec![], self.device())
    }

    /// Check if all elements along dimension are non-zero (true)
    pub fn all_dim(&self, dim: i32, _keepdim: bool) -> Result<Tensor<bool>> {
        let normalized_dim = if dim < 0 {
            (self.shape().len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= self.shape().len() {
            return Err(torsh_core::error::TorshError::InvalidDimension {
                dim: normalized_dim,
                ndim: self.shape().len(),
            });
        }

        // TODO: Implement proper all() reduction without ndarray dependency
        // For now, return a simple placeholder
        Err(TorshError::NotImplemented(
            "Boolean all() reduction along dimension not yet implemented".to_string(),
        ))
    }

    /// Check if any element along dimension is non-zero (true)
    pub fn any_dim(&self, dim: i32, _keepdim: bool) -> Result<Tensor<bool>> {
        let normalized_dim = if dim < 0 {
            (self.shape().len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= self.shape().len() {
            return Err(torsh_core::error::TorshError::InvalidDimension {
                dim: normalized_dim,
                ndim: self.shape().len(),
            });
        }

        // TODO: Implement proper any() reduction without ndarray dependency
        // For now, return a simple placeholder
        Err(TorshError::NotImplemented(
            "Boolean any() reduction along dimension not yet implemented".to_string(),
        ))
    }
}

// General tensor operations
impl<T: TensorElement + Copy> Tensor<T> {
    /// Compute sum of all elements
    pub fn sum(&self) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero,
    {
        let data = self.data()?;
        let sum_value = data
            .iter()
            .fold(<T as num_traits::Zero>::zero(), |acc, &x| acc + x);
        let mut result = Tensor::from_data(vec![sum_value], vec![], self.device())?;

        // Preserve gradient tracking
        if self.requires_grad {
            result.requires_grad = true;
            // TODO: Add proper Sum operation for autograd backward pass
            // For now, this will work for simple cases
        }

        Ok(result)
    }

    /// Compute sum along specified dimensions
    pub fn sum_dim(&self, dims: &[i32], keepdim: bool) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero,
    {
        if dims.is_empty() {
            return self.sum();
        }

        let shape_binding = self.shape();
        let input_shape = shape_binding.dims();

        // Handle single dimension case (most common)
        if dims.len() == 1 {
            let dim = dims[0];
            let actual_dim = if dim < 0 {
                (input_shape.len() as i32 + dim) as usize
            } else {
                dim as usize
            };

            if actual_dim >= input_shape.len() {
                return Err(TorshError::InvalidOperation(format!(
                    "Dimension {} out of range for {}-dimensional tensor",
                    actual_dim,
                    input_shape.len()
                )));
            }

            // Calculate output shape
            let mut output_shape = input_shape.to_vec();
            if keepdim {
                output_shape[actual_dim] = 1;
            } else {
                output_shape.remove(actual_dim);
            }

            let data = self.data()?;
            let outer_size: usize = input_shape[..actual_dim].iter().product();
            let dim_size = input_shape[actual_dim];
            let inner_size: usize = input_shape[actual_dim + 1..].iter().product();

            let output_size = outer_size * inner_size;
            let mut result_data = vec![num_traits::Zero::zero(); output_size];

            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let mut sum = num_traits::Zero::zero();
                    for d in 0..dim_size {
                        let input_idx = outer * dim_size * inner_size + d * inner_size + inner;
                        sum = sum + data[input_idx];
                    }
                    let output_idx = outer * inner_size + inner;
                    result_data[output_idx] = sum;
                }
            }

            Self::from_data(result_data, output_shape, self.device)
        } else {
            // For multiple dimensions, fall back to full sum for now
            self.sum()
        }
    }

    /// Compute mean along specified dimensions
    pub fn mean(&self, dims: Option<&[usize]>, keepdim: bool) -> Result<Self>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + num_traits::Zero
            + num_traits::One
            + num_traits::FromPrimitive,
    {
        let sum = if let Some(dims) = dims {
            self.sum_dim(&dims.iter().map(|&d| d as i32).collect::<Vec<_>>(), keepdim)?
        } else {
            let scalar_sum = self.sum()?;
            if keepdim {
                // Reshape scalar to tensor with same ndim as original, all dims = 1
                let keepdim_shape = vec![1; self.shape().ndim()];
                scalar_sum.view(&keepdim_shape)?
            } else {
                scalar_sum
            }
        };

        let count = if let Some(dims) = dims {
            dims.iter()
                .map(|&d| self.shape().dims()[d])
                .product::<usize>() as f64
        } else {
            self.numel() as f64
        };

        sum.div_scalar(
            <T as num_traits::FromPrimitive>::from_f64(count)
                .unwrap_or_else(|| <T as num_traits::One>::one()),
        )
    }

    /// Compute cumulative product along specified dimension
    pub fn cumprod(&self, dim: i32) -> Result<Self>
    where
        T: std::ops::Mul<Output = T> + num_traits::One + Copy,
    {
        let normalized_dim = if dim < 0 {
            (self.shape().len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= self.shape().len() {
            return Err(torsh_core::error::TorshError::InvalidDimension {
                dim: normalized_dim,
                ndim: self.shape().len(),
            });
        }

        let shape = self.shape().clone();
        let input_shape = shape.dims();
        let data = self.data()?;
        let mut result_data = data.to_vec();

        let outer_size: usize = input_shape[..normalized_dim].iter().product();
        let dim_size = input_shape[normalized_dim];
        let inner_size: usize = input_shape[normalized_dim + 1..].iter().product();

        for outer_idx in 0..outer_size {
            for inner_idx in 0..inner_size {
                let mut running_product = <T as num_traits::One>::one();
                for dim_idx in 0..dim_size {
                    let index =
                        outer_idx * (dim_size * inner_size) + dim_idx * inner_size + inner_idx;
                    running_product = running_product * result_data[index];
                    result_data[index] = running_product;
                }
            }
        }

        Self::from_data(result_data, input_shape.to_vec(), self.device())
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self>
    where
        T: num_traits::Float + std::iter::Sum,
    {
        self.basic_matmul(other)
    }

    /// Sort tensor along specified dimension
    pub fn sort(&self, _dim: Option<i32>, _descending: bool) -> Result<(Self, Self)>
    where
        T: PartialOrd + num_traits::Zero + num_traits::FromPrimitive,
    {
        // Simple implementation - sort entire tensor as 1D
        let data = self.to_vec()?;
        let mut indexed_data: Vec<(usize, T)> =
            data.iter().enumerate().map(|(i, &val)| (i, val)).collect();

        // Sort by value
        indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Extract sorted data and indices
        let sorted_data: Vec<T> = indexed_data.iter().map(|(_, val)| *val).collect();
        let indices: Vec<T> = indexed_data
            .iter()
            .map(|(i, _)| {
                <T as num_traits::FromPrimitive>::from_usize(*i)
                    .unwrap_or_else(|| <T as num_traits::Zero>::zero())
            })
            .collect();

        let sorted_tensor =
            Self::from_data(sorted_data, self.shape().dims().to_vec(), self.device())?;
        let indices_tensor = Self::from_data(indices, self.shape().dims().to_vec(), self.device())?;

        Ok((sorted_tensor, indices_tensor))
    }

    /// Min reduction method without trait bounds (for Iterator compatibility)
    pub fn min(&self) -> Result<Self>
    where
        T: std::cmp::PartialOrd + Copy,
    {
        let data = self.data()?;
        if data.is_empty() {
            return Err(TorshError::InvalidOperation(
                "Cannot compute min of empty tensor".to_string(),
            ));
        }

        let min_val = data
            .iter()
            .fold(data[0], |acc, &x| if x < acc { x } else { acc });
        Self::from_data(vec![min_val], vec![], self.device)
    }

    /// Transpose operation (2D tensor)
    pub fn t(&self) -> Result<Self>
    where
        T: Copy + num_traits::Zero,
    {
        let shape = self.shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return Err(TorshError::InvalidOperation(
                "Transpose operation only supported for 2D tensors".to_string(),
            ));
        }

        let (rows, cols) = (dims[0], dims[1]);
        let data = self.data()?;
        let mut transposed_data = vec![num_traits::Zero::zero(); data.len()];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = data[i * cols + j];
            }
        }

        Self::from_data(transposed_data, vec![cols, rows], self.device)
    }

    /// Check if two tensors share the same underlying storage
    pub fn shares_storage(&self, other: &Self) -> bool {
        // For storage abstraction, we need to check the underlying storage
        match (&self.storage, &other.storage) {
            (TensorStorage::InMemory(a), TensorStorage::InMemory(b)) => Arc::ptr_eq(a, b),
            (TensorStorage::MemoryMapped(a), TensorStorage::MemoryMapped(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Get data as a vector (backward compatibility method)
    pub fn data(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.to_vec()
    }

    /// Apply a function to all elements in-place using direct storage access
    pub fn data_mut_apply<F>(&mut self, mut func: F) -> Result<()>
    where
        F: FnMut(&mut T),
        T: Copy,
    {
        self.ensure_exclusive_data()?;

        match &mut self.storage {
            TensorStorage::InMemory(data) => {
                let mut data_guard = data.write().unwrap();
                for item in data_guard.iter_mut() {
                    func(item);
                }
                Ok(())
            }
            TensorStorage::MemoryMapped(_) => {
                // For memory-mapped storage, we need to read-modify-write
                let data = self.to_vec()?;
                let mut new_data = data;
                for item in new_data.iter_mut() {
                    func(item);
                }
                // Write back the data
                self.storage = TensorStorage::create_optimal(new_data)?;
                Ok(())
            }
            #[cfg(feature = "simd")]
            TensorStorage::Aligned(data) => {
                let mut data_guard = data.write().unwrap();
                for item in data_guard.as_mut_slice().iter_mut() {
                    func(item);
                }
                Ok(())
            }
        }
    }

    /// Clone the tensor with independent data (deep copy)
    pub fn clone_data(&self) -> Self
    where
        T: Copy,
    {
        let data = self.to_vec().unwrap();
        Self::from_data(data, self.shape().dims().to_vec(), self.device).unwrap()
    }

    /// Ensure tensor has unique data (copy-on-write semantics)
    pub fn make_unique(&mut self) -> Result<()> {
        // For storage-based approach, create new storage if shared
        match &self.storage {
            TensorStorage::InMemory(data) => {
                if Arc::strong_count(data) > 1 {
                    let data_vec = self.to_vec()?;
                    self.storage = TensorStorage::create_optimal(data_vec)?;
                }
            }
            TensorStorage::MemoryMapped(storage) => {
                if Arc::strong_count(storage) > 1 {
                    let data_vec = self.to_vec()?;
                    self.storage = TensorStorage::create_optimal(data_vec)?;
                }
            }
            #[cfg(feature = "simd")]
            TensorStorage::Aligned(data) => {
                if Arc::strong_count(data) > 1 {
                    let data_vec = self.to_vec()?;
                    self.storage = TensorStorage::create_optimal(data_vec)?;
                }
            }
        }
        Ok(())
    }

    /// Apply function in-place
    pub fn apply_<F>(&mut self, func: F) -> Result<()>
    where
        F: Fn(T) -> T,
        T: Copy,
    {
        let data = self.to_vec()?;
        let new_data: Vec<T> = data.into_iter().map(func).collect();

        // Update storage with new data
        self.storage = TensorStorage::create_optimal(new_data)?;
        Ok(())
    }

    /// Apply function element-wise to create new tensor
    pub fn map<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(T) -> T,
        T: Copy,
    {
        let data = self.to_vec()?;
        let new_data: Vec<T> = data.into_iter().map(func).collect();
        let mut result = Self::from_data(new_data, self.shape().dims().to_vec(), self.device)?;

        // Preserve gradient tracking flag from original tensor
        result.requires_grad = self.requires_grad;

        Ok(result)
    }

    /// Extract a scalar value from a single-element tensor
    pub fn item(&self) -> Result<T>
    where
        T: Copy,
    {
        let data = self.data()?;
        if data.len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "item() can only be called on single-element tensors, got {} elements",
                data.len()
            )));
        }
        Ok(data[0])
    }

    /// Concatenate tensors along a dimension
    pub fn cat(tensors: &[&Self], dim: i32) -> Result<Self>
    where
        T: Copy,
    {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        // For now, implement simple concatenation for 1D tensors along dim 0
        // TODO: Implement proper multi-dimensional concatenation
        let mut all_data = Vec::new();
        let mut total_len = 0;

        for tensor in tensors {
            let data = tensor.data()?;
            all_data.extend_from_slice(&data);
            total_len += data.len();
        }

        // Use the shape of the first tensor as base, but extend the concatenation dimension
        let first_tensor_shape = tensors[0].shape();
        let first_shape = first_tensor_shape.dims();
        let mut result_shape = first_shape.to_vec();

        if dim == 0 && result_shape.len() == 1 {
            result_shape[0] = total_len;
        } else if result_shape.is_empty() {
            result_shape = vec![total_len];
        }

        Self::from_data(all_data, result_shape, tensors[0].device)
    }

    /// Ensure exclusive ownership of data using copy-on-write semantics
    /// If the data is shared (Arc has multiple strong references), clone it
    fn ensure_exclusive_data(&mut self) -> Result<()> {
        match &self.storage {
            TensorStorage::InMemory(data) => {
                if Arc::strong_count(data) > 1 {
                    // Data is shared, need to clone it to get exclusive access
                    let cloned_data = {
                        let data_guard = data.read().unwrap();
                        data_guard.clone()
                    };
                    self.storage = TensorStorage::in_memory(cloned_data);
                }
            }
            TensorStorage::MemoryMapped(storage) => {
                if Arc::strong_count(storage) > 1 {
                    // Clone memory-mapped storage by converting to vec and back
                    let data_vec = self.storage.to_vec()?;
                    self.storage = TensorStorage::create_optimal(data_vec)?;
                }
            }
            #[cfg(feature = "simd")]
            TensorStorage::Aligned(data) => {
                if Arc::strong_count(data) > 1 {
                    // Data is shared, need to clone it to get exclusive access
                    let vec_data = {
                        let data_guard = data.read().unwrap();
                        data_guard.as_slice().to_vec()
                    };
                    self.storage = TensorStorage::aligned(vec_data)?;
                }
            }
        }
        Ok(())
    }
}

// Numeric operations
impl<T: TensorElement + Copy> Tensor<T>
where
    T: num_traits::Float,
{
    /// Compute the L2 norm of the tensor
    pub fn norm(&self) -> Result<Self> {
        let data = self.data()?;
        let sum_squares: T = data
            .iter()
            .map(|&x| x * x)
            .fold(num_traits::Zero::zero(), |acc, x| acc + x);
        let norm_value = sum_squares.sqrt();

        // Return scalar tensor (1-element tensor with shape [])
        Tensor::from_data(vec![norm_value], vec![], self.device())
    }
}

// SciRS2 backend integration (placeholder implementations)
impl<T: TensorElement + Copy> Tensor<T> {
    /// Use SciRS2 backend for optimized matrix multiplication
    pub fn matmul_scirs2(&self, other: &Self) -> Result<Self>
    where
        T: num_traits::Float + num_traits::Zero + num_traits::One + std::iter::Sum,
    {
        // TODO: Integrate with actual SciRS2 backend
        // For now, implement basic matrix multiplication
        self.basic_matmul(other)
    }

    /// Use SciRS2 backend for optimized sum reduction
    pub fn sum_scirs2(&self) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero,
    {
        // TODO: Integrate with actual SciRS2 backend
        let data = self.data()?;
        let sum_value = data
            .iter()
            .fold(<T as num_traits::Zero>::zero(), |acc, &x| acc + x);
        Tensor::from_data(vec![sum_value], vec![], self.device())
    }

    /// Use SciRS2 backend for optimized mean reduction
    pub fn mean_scirs2(&self) -> Result<Self>
    where
        T: std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + num_traits::Zero
            + From<usize>
            + num_traits::FromPrimitive,
    {
        // TODO: Integrate with actual SciRS2 backend
        let data = self.data()?;
        if data.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot compute mean of empty tensor".to_string(),
            ));
        }
        let sum_value = data
            .iter()
            .fold(<T as num_traits::Zero>::zero(), |acc, &x| acc + x);
        let mean_value = sum_value / T::from(data.len());
        Tensor::from_data(vec![mean_value], vec![], self.device())
    }

    /// Use SciRS2 backend for optimized ReLU activation
    pub fn relu_scirs2(&self) -> Result<Self>
    where
        T: PartialOrd + num_traits::Zero,
    {
        // TODO: Integrate with actual SciRS2 backend
        let zero = <T as num_traits::Zero>::zero();
        self.map(|x| if x > zero { x } else { zero })
    }

    /// Use SciRS2 backend for optimized sigmoid activation
    pub fn sigmoid_scirs2(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // TODO: Integrate with actual SciRS2 backend
        self.map(|x| {
            let one = <T as num_traits::One>::one();
            one / (one + (-x).exp())
        })
    }

    /// Use SciRS2 backend for optimized tanh activation
    pub fn tanh_scirs2(&self) -> Result<Self>
    where
        T: num_traits::Float,
    {
        // TODO: Integrate with actual SciRS2 backend
        self.map(|x| x.tanh())
    }

    /// Basic matrix multiplication implementation
    fn basic_matmul(&self, other: &Self) -> Result<Self>
    where
        T: num_traits::Float + std::iter::Sum,
    {
        let self_binding = self.shape();
        let self_shape = self_binding.dims();
        let other_binding = other.shape();
        let other_shape = other_binding.dims();

        // Check dimensions for matrix multiplication
        if self_shape.len() != 2 || other_shape.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        if self_shape[1] != other_shape[0] {
            return Err(TorshError::ShapeMismatch {
                expected: vec![self_shape[0], other_shape[1]],
                got: vec![self_shape[1], other_shape[0]],
            });
        }

        let (m, k) = (self_shape[0], self_shape[1]);
        let n = other_shape[1];

        let self_data = self.data()?;
        let other_data = other.data()?;
        let mut result_data = vec![num_traits::Zero::zero(); m * n];

        // Basic matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = num_traits::Zero::zero();
                for k_idx in 0..k {
                    sum = sum + self_data[i * k + k_idx] * other_data[k_idx * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }

        Self::from_data(result_data, vec![m, n], self.device)
    }
    /// Softmax activation along specified dimension
    /// Computes softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    pub fn softmax(&self, dim: i32) -> Result<Self>
    where
        T: torsh_core::dtype::FloatElement
            + Copy
            + std::ops::Sub<Output = T>
            + std::ops::Div<Output = T>,
    {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        // Validate tensor has data
        if data.is_empty() || shape.is_empty() {
            return Err(TorshError::InvalidOperation(
                "Cannot compute softmax on empty tensor".to_string(),
            ));
        }

        // Handle negative dimension
        let actual_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= shape.len() {
            return Err(TorshError::InvalidOperation(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                actual_dim,
                shape.len()
            )));
        }

        // For numerical stability, subtract max before exp
        let max_tensor = self.max(Some(actual_dim), true)?;

        // Expand max_tensor to match input shape for broadcasting
        let expanded_max = max_tensor.expand(shape)?;
        let shifted = self.sub(&expanded_max)?;
        let exp_tensor = shifted.exp()?;
        let sum_tensor = exp_tensor.sum_dim(&[actual_dim as i32], true)?;

        // Expand sum_tensor to match exp_tensor shape for broadcasting
        let expanded_sum = sum_tensor.expand(shape)?;
        exp_tensor.div(&expanded_sum)
    }

    /// Log softmax activation along specified dimension
    /// Computes log_softmax(x_i) = log(softmax(x_i))
    pub fn log_softmax(&self, dim: i32) -> Result<Self>
    where
        T: torsh_core::dtype::FloatElement + Copy + std::ops::Sub<Output = T>,
    {
        let softmax_result = self.softmax(dim)?;
        softmax_result.log()
    }

    /// Computes cumulative sum along a dimension
    pub fn cumsum(&self, dim: i32) -> Result<Self>
    where
        T: std::ops::Add<Output = T> + num_traits::Zero + Copy,
    {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        // Handle negative dimension
        let actual_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= shape.len() {
            return Err(TorshError::InvalidOperation(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                actual_dim,
                shape.len()
            )));
        }

        let data = self.data()?;
        let mut result_data = data.clone();

        // Simplified cumsum implementation for now
        // This is a basic implementation that works along the flattened array
        if actual_dim == shape.len() - 1 || shape.len() == 1 {
            let mut cumulative = <T as num_traits::Zero>::zero();
            for i in 0..result_data.len() {
                cumulative = cumulative + result_data[i];
                result_data[i] = cumulative;
            }
        }

        Self::from_data(result_data, shape.to_vec(), self.device)
    }

    /// Find the indices of minimum values along a dimension
    pub fn argmin(&self, dim: Option<i32>) -> Result<Tensor<i64>>
    where
        T: std::cmp::PartialOrd + Copy,
    {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        if shape.is_empty() {
            return Err(TorshError::InvalidOperation(
                "Cannot compute argmin on empty tensor".to_string(),
            ));
        }

        match dim {
            Some(d) => {
                // Handle negative dimension
                let actual_dim = if d < 0 {
                    (shape.len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if actual_dim >= shape.len() {
                    return Err(TorshError::InvalidOperation(format!(
                        "Dimension {} out of range for {}-dimensional tensor",
                        actual_dim,
                        shape.len()
                    )));
                }

                // For simplicity, return the first minimum index found
                // This is a basic implementation - real argmin would handle the specified dimension properly
                let min_val = data
                    .iter()
                    .fold(data[0], |acc, &x| if x < acc { x } else { acc });
                let min_idx = data.iter().position(|&x| x == min_val).unwrap_or(0);

                let result_data = vec![min_idx as i64];
                Tensor::<i64>::from_data(result_data, vec![1], self.device)
            }
            None => {
                // Find argmin over the entire flattened tensor
                let min_val = data
                    .iter()
                    .fold(data[0], |acc, &x| if x < acc { x } else { acc });
                let min_idx = data.iter().position(|&x| x == min_val).unwrap_or(0);

                let result_data = vec![min_idx as i64];
                Tensor::<i64>::from_data(result_data, vec![], self.device)
            }
        }
    }

    /// Find the indices of maximum values along a dimension
    pub fn argmax(&self, dim: Option<i32>) -> Result<Tensor<i64>>
    where
        T: std::cmp::PartialOrd + Copy,
    {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        if shape.is_empty() {
            return Err(TorshError::InvalidOperation(
                "Cannot compute argmax on empty tensor".to_string(),
            ));
        }

        match dim {
            Some(d) => {
                // Handle negative dimension
                let actual_dim = if d < 0 {
                    (shape.len() as i32 + d) as usize
                } else {
                    d as usize
                };

                if actual_dim >= shape.len() {
                    return Err(TorshError::InvalidOperation(format!(
                        "Dimension {} out of range for {}-dimensional tensor",
                        actual_dim,
                        shape.len()
                    )));
                }

                // For simplicity, return the first maximum index found
                // This is a basic implementation - real argmax would handle the specified dimension properly
                let max_val = data
                    .iter()
                    .fold(data[0], |acc, &x| if x > acc { x } else { acc });
                let max_idx = data.iter().position(|&x| x == max_val).unwrap_or(0);

                let result_data = vec![max_idx as i64];
                Tensor::<i64>::from_data(result_data, vec![1], self.device)
            }
            None => {
                // Find argmax over the entire flattened tensor
                let max_val = data
                    .iter()
                    .fold(data[0], |acc, &x| if x > acc { x } else { acc });
                let max_idx = data.iter().position(|&x| x == max_val).unwrap_or(0);

                let result_data = vec![max_idx as i64];
                Tensor::<i64>::from_data(result_data, vec![], self.device)
            }
        }
    }

    /// Returns the k largest elements along a dimension
    pub fn topk(
        &self,
        k: usize,
        dim: Option<i32>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self, Tensor<i64>)>
    where
        T: std::cmp::PartialOrd + Copy + num_traits::Zero,
    {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        if shape.is_empty() {
            return Err(TorshError::InvalidOperation(
                "Cannot compute topk on empty tensor".to_string(),
            ));
        }

        if k == 0 {
            return Err(TorshError::InvalidArgument(
                "k must be greater than 0".to_string(),
            ));
        }

        // Log dimension and sorting info
        if let Some(_d) = dim {
        } else {
        }

        // For simplicity, implement topk on flattened tensor
        // TODO: Implement proper per-dimension topk when dim is specified
        let mut indexed_data: Vec<(usize, T)> =
            data.iter().enumerate().map(|(i, &val)| (i, val)).collect();

        // Sort by value (largest first if largest=true, smallest first if largest=false)
        if largest {
            indexed_data.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Take top k elements
        let top_k = indexed_data
            .into_iter()
            .take(k.min(data.len()))
            .collect::<Vec<_>>();

        // If sorted=false, shuffle the results to remove order
        // (in practice, keeping sorted is usually preferred for performance)
        if !sorted {
            // TODO: Implement shuffling when needed
        }

        // Extract values and indices
        let values: Vec<T> = top_k.iter().map(|(_, val)| *val).collect();
        let indices: Vec<i64> = top_k.iter().map(|(idx, _)| *idx as i64).collect();

        // Create result tensors
        let values_tensor = Self::from_data(values, vec![k.min(data.len())], self.device)?;
        let indices_tensor =
            Tensor::<i64>::from_data(indices, vec![k.min(data.len())], self.device)?;

        Ok((values_tensor, indices_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_scalar_creation() {
        let scalar = Tensor::<f32>::scalar(42.0).unwrap();
        assert_eq!(scalar.shape().dims(), &[] as &[usize]);
        assert_eq!(scalar.item().unwrap(), 42.0);
    }

    #[test]
    fn test_max_reduction() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0];
        let tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu).unwrap();
        let max_val = tensor.max(None, false).unwrap();
        assert_eq!(max_val.item().unwrap(), 5.0);
    }

    #[test]
    fn test_norm_computation() {
        let data = vec![3.0f32, 4.0]; // 3-4-5 triangle
        let tensor = Tensor::from_data(data, vec![2], DeviceType::Cpu).unwrap();
        let norm = tensor.norm().unwrap();
        assert!((norm.item().unwrap() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_operations() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu).unwrap();

        // Test apply_
        tensor.apply_(|x| x * 2.0).unwrap();
        assert_eq!(tensor.data().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);

        // Test map
        let original = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let mapped = original.map(|x| x + 1.0).unwrap();
        assert_eq!(mapped.data().unwrap(), vec![2.0, 3.0, 4.0]);
        assert_eq!(original.data().unwrap(), vec![1.0, 2.0, 3.0]); // Original unchanged
    }

    #[test]
    fn test_activation_functions() {
        let data = vec![-1.0f32, 0.0, 1.0, 2.0];
        let tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu).unwrap();

        // Test ReLU
        let relu_result = tensor.relu().unwrap();
        assert_eq!(relu_result.data().unwrap(), vec![0.0, 0.0, 1.0, 2.0]);

        // Test abs
        let abs_result = tensor.abs().unwrap();
        assert_eq!(abs_result.data().unwrap(), vec![1.0, 0.0, 1.0, 2.0]);

        // Test clamp
        let clamped = tensor.clamp(-0.5, 1.5).unwrap();
        assert_eq!(clamped.data().unwrap(), vec![-0.5, 0.0, 1.0, 1.5]);
    }

    #[test]
    fn test_storage_sharing() {
        let tensor1 = Tensor::<f32>::zeros(&[2, 2], DeviceType::Cpu).unwrap();
        let tensor2 = tensor1.clone();
        let tensor3 = tensor1.clone_data();

        assert!(tensor1.shares_storage(&tensor2));
        assert!(!tensor1.shares_storage(&tensor3));
    }

    #[test]
    fn test_basic_matmul() {
        let a =
            Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let b =
            Tensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], DeviceType::Cpu).unwrap();

        let result = a.basic_matmul(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        assert_eq!(result.data().unwrap(), expected);
    }

    #[test]
    fn test_reductions() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu).unwrap();

        let sum = tensor.sum().unwrap();
        assert_eq!(sum.item().unwrap(), 10.0);

        let mean = tensor.mean(None, false).unwrap();
        assert_eq!(mean.item().unwrap(), 2.5);
    }

    #[test]
    fn test_copy_on_write() {
        let mut tensor1 = Tensor::<f32>::ones(&[2], DeviceType::Cpu).unwrap();
        let tensor2 = tensor1.clone();

        // Both should share storage initially
        assert!(tensor1.shares_storage(&tensor2));

        // After make_unique, they should not share storage
        tensor1.make_unique().unwrap();
        assert!(!tensor1.shares_storage(&tensor2));
    }

    #[test]
    fn test_item_extraction() {
        let scalar = Tensor::from_data(vec![42.0f32], vec![], DeviceType::Cpu).unwrap();
        assert_eq!(scalar.item().unwrap(), 42.0);

        let vector = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        assert!(vector.item().is_err()); // Should fail for multi-element tensor
    }
}
