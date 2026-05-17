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
    pub fn all_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor<bool>> {
        let shape_binding = self.shape();
        let input_shape = shape_binding.dims();

        let normalized_dim = if dim < 0 {
            (input_shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= input_shape.len() {
            return Err(torsh_core::error::TorshError::InvalidDimension {
                dim: normalized_dim,
                ndim: input_shape.len(),
            });
        }

        let data = self.data()?;
        let zero = <T as num_traits::Zero>::zero();

        let outer_size: usize = input_shape[..normalized_dim].iter().product();
        let dim_size = input_shape[normalized_dim];
        let inner_size: usize = input_shape[normalized_dim + 1..].iter().product();

        let output_size = outer_size * inner_size;
        let mut result_data = vec![true; output_size];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let all_nonzero = (0..dim_size).all(|d| {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    data[idx] != zero
                });
                let out_idx = outer * inner_size + inner;
                result_data[out_idx] = all_nonzero;
            }
        }

        let mut output_shape = input_shape.to_vec();
        if keepdim {
            output_shape[normalized_dim] = 1;
        } else {
            output_shape.remove(normalized_dim);
        }

        Tensor::<bool>::from_data(result_data, output_shape, self.device())
    }

    /// Check if any element along dimension is non-zero (true)
    pub fn any_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor<bool>> {
        let shape_binding = self.shape();
        let input_shape = shape_binding.dims();

        let normalized_dim = if dim < 0 {
            (input_shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= input_shape.len() {
            return Err(torsh_core::error::TorshError::InvalidDimension {
                dim: normalized_dim,
                ndim: input_shape.len(),
            });
        }

        let data = self.data()?;
        let zero = <T as num_traits::Zero>::zero();

        let outer_size: usize = input_shape[..normalized_dim].iter().product();
        let dim_size = input_shape[normalized_dim];
        let inner_size: usize = input_shape[normalized_dim + 1..].iter().product();

        let output_size = outer_size * inner_size;
        let mut result_data = vec![false; output_size];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let any_nonzero = (0..dim_size).any(|d| {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    data[idx] != zero
                });
                let out_idx = outer * inner_size + inner;
                result_data[out_idx] = any_nonzero;
            }
        }

        let mut output_shape = input_shape.to_vec();
        if keepdim {
            output_shape[normalized_dim] = 1;
        } else {
            output_shape.remove(normalized_dim);
        }

        Tensor::<bool>::from_data(result_data, output_shape, self.device())
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

        let mut result = sum.div_scalar(
            <T as num_traits::FromPrimitive>::from_f64(count)
                .unwrap_or_else(|| <T as num_traits::One>::one()),
        )?;

        // Propagate requires_grad and record operation for autograd
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::core_ops::Operation::Mean {
                input: Arc::new(self.clone()),
                count,
            };
        }

        Ok(result)
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
                let mut data_guard = data.write().expect("lock should not be poisoned");
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
                let mut data_guard = data.write().expect("lock should not be poisoned");
                for item in data_guard.as_mut_slice().iter_mut() {
                    func(item);
                }
                Ok(())
            }
            #[cfg(feature = "simd")]
            TensorStorage::SimdOptimized(_) => {
                // SimdOptimized should have been converted by ensure_exclusive_data()
                // If we reach here, something went wrong - convert to optimal storage and retry
                let data = self.to_vec()?;
                let mut new_data = data;
                for item in new_data.iter_mut() {
                    func(item);
                }
                self.storage = TensorStorage::create_optimal(new_data)?;
                Ok(())
            }
        }
    }

    /// Clone the tensor with independent data (deep copy)
    pub fn clone_data(&self) -> Self
    where
        T: Copy,
    {
        let data = self
            .to_vec()
            .expect("tensor to vec conversion should succeed");
        Self::from_data(data, self.shape().dims().to_vec(), self.device)
            .expect("tensor creation should succeed")
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
            #[cfg(feature = "simd")]
            TensorStorage::SimdOptimized(_storage) => {
                // SimdOptimized storage is immutable by design (optimized for read-heavy workloads)
                // Always convert to Aligned storage which supports both SIMD and mutation
                let data_vec = self.to_vec()?;
                self.storage = TensorStorage::aligned(data_vec)?;
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

        let first_shape_binding = tensors[0].shape();
        let first_shape = first_shape_binding.dims();
        let ndim = first_shape.len();

        // Normalize dim (allow negative indexing)
        let actual_dim = if dim < 0 {
            (ndim as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for {}-dimensional tensor",
                dim, ndim
            )));
        }

        // Validate all tensors have compatible shapes (same on all dims except actual_dim)
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            let shape_binding = tensor.shape();
            let shape = shape_binding.dims();
            if shape.len() != ndim {
                return Err(TorshError::InvalidArgument(format!(
                    "Tensor {} has {} dimensions but first tensor has {}",
                    i,
                    shape.len(),
                    ndim
                )));
            }
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d != actual_dim && s1 != s2 {
                    return Err(TorshError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: shape.to_vec(),
                    });
                }
            }
        }

        // Compute output shape: same as input except actual_dim is sum of all cat dims
        let cat_dim_total: usize = tensors.iter().map(|t| t.shape().dims()[actual_dim]).sum();
        let mut result_shape = first_shape.to_vec();
        result_shape[actual_dim] = cat_dim_total;

        // Gather all data in order, interleaving elements for proper layout
        // Outer = product of dims before actual_dim
        // Cat stride = product of dims after actual_dim (inner)
        let outer_size: usize = first_shape[..actual_dim].iter().product();
        let inner_size: usize = first_shape[actual_dim + 1..].iter().product();

        let total_numel: usize = result_shape.iter().product();
        let mut result_data = Vec::with_capacity(total_numel);

        for outer in 0..outer_size {
            for tensor in tensors {
                let tensor_shape_binding = tensor.shape();
                let tensor_shape = tensor_shape_binding.dims();
                let cat_size = tensor_shape[actual_dim];
                let tensor_data = tensor.data()?;

                for cat_idx in 0..cat_size {
                    for inner in 0..inner_size {
                        let src_idx = outer * cat_size * inner_size + cat_idx * inner_size + inner;
                        result_data.push(tensor_data[src_idx]);
                    }
                }
            }
        }

        Self::from_data(result_data, result_shape, tensors[0].device)
    }

    /// Ensure exclusive ownership of data using copy-on-write semantics
    /// If the data is shared (Arc has multiple strong references), clone it
    fn ensure_exclusive_data(&mut self) -> Result<()> {
        match &self.storage {
            TensorStorage::InMemory(data) => {
                if Arc::strong_count(data) > 1 {
                    // Data is shared, need to clone it to get exclusive access
                    let cloned_data = {
                        let data_guard = data.read().expect("lock should not be poisoned");
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
                        let data_guard = data.read().expect("lock should not be poisoned");
                        data_guard.as_slice().to_vec()
                    };
                    self.storage = TensorStorage::aligned(vec_data)?;
                }
            }
            #[cfg(feature = "simd")]
            TensorStorage::SimdOptimized(storage) => {
                if Arc::strong_count(storage) > 1 || storage.is_shared() {
                    // SimdOptimized uses COW - copy the data to get exclusive access
                    let vec_data = storage.to_vec();
                    self.storage = TensorStorage::simd_optimized(vec_data)?;
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

        // Determine actual dimension to operate on (default: last dim)
        let actual_dim = match dim {
            Some(d) => {
                let norm = if d < 0 {
                    (shape.len() as i32 + d) as usize
                } else {
                    d as usize
                };
                if norm >= shape.len() {
                    return Err(TorshError::InvalidArgument(format!(
                        "Dimension {} out of range for {}-dimensional tensor",
                        d,
                        shape.len()
                    )));
                }
                norm
            }
            None => shape.len() - 1,
        };

        let dim_size = shape[actual_dim];
        let effective_k = k.min(dim_size);

        let outer_size: usize = shape[..actual_dim].iter().product();
        let inner_size: usize = shape[actual_dim + 1..].iter().product();

        // Output shape: same as input but actual_dim replaced with k
        let mut result_shape = shape.to_vec();
        result_shape[actual_dim] = effective_k;

        let mut values_data = Vec::with_capacity(outer_size * effective_k * inner_size);
        let mut indices_data = Vec::with_capacity(outer_size * effective_k * inner_size);

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                // Gather (local_index, value) pairs along actual_dim for this (outer, inner) slice
                let mut slice: Vec<(usize, T)> = (0..dim_size)
                    .map(|d| {
                        let src = outer * dim_size * inner_size + d * inner_size + inner;
                        (d, data[src])
                    })
                    .collect();

                // Sort by value to find top-k candidates
                if largest {
                    slice.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                } else {
                    slice.sort_by(|a, b| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                let mut top_k: Vec<(usize, T)> = slice.into_iter().take(effective_k).collect();

                // When sorted=false, restore original (position) order
                if !sorted {
                    top_k.sort_by_key(|(idx, _)| *idx);
                }

                for (local_idx, val) in &top_k {
                    values_data.push(*val);
                    indices_data.push(*local_idx as i64);
                }
            }
        }

        // Re-arrange from (outer, k, inner) to match result_shape layout
        // Currently we have data as outer * inner * k interleaved; need outer * k * inner
        // Transpose inner and k dimensions
        let transposed_len = outer_size * effective_k * inner_size;
        let mut values_transposed = Vec::with_capacity(transposed_len);
        let mut indices_transposed = Vec::with_capacity(transposed_len);

        for outer in 0..outer_size {
            for k_idx in 0..effective_k {
                for inner in 0..inner_size {
                    let src = outer * inner_size * effective_k + inner * effective_k + k_idx;
                    values_transposed.push(values_data[src]);
                    indices_transposed.push(indices_data[src]);
                }
            }
        }

        let values_tensor = Self::from_data(values_transposed, result_shape.clone(), self.device)?;
        let indices_tensor =
            Tensor::<i64>::from_data(indices_transposed, result_shape, self.device)?;

        Ok((values_tensor, indices_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_scalar_creation() {
        let scalar = Tensor::<f32>::scalar(42.0).expect("operation should succeed");
        assert_eq!(scalar.shape().dims(), &[] as &[usize]);
        assert_eq!(scalar.item().expect("item extraction should succeed"), 42.0);
    }

    #[test]
    fn test_max_reduction() {
        let data = vec![1.0f32, 5.0, 3.0, 2.0];
        let tensor =
            Tensor::from_data(data, vec![4], DeviceType::Cpu).expect("operation should succeed");
        let max_val = tensor.max(None, false).expect("operation should succeed");
        assert_eq!(max_val.item().expect("item extraction should succeed"), 5.0);
    }

    #[test]
    fn test_norm_computation() {
        let data = vec![3.0f32, 4.0]; // 3-4-5 triangle
        let tensor =
            Tensor::from_data(data, vec![2], DeviceType::Cpu).expect("operation should succeed");
        let norm = tensor.norm().expect("norm computation should succeed");
        assert!((norm.item().expect("item extraction should succeed") - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_operations() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut tensor =
            Tensor::from_data(data, vec![4], DeviceType::Cpu).expect("operation should succeed");

        // Test apply_
        tensor
            .apply_(|x| x * 2.0)
            .expect("operation should succeed");
        assert_eq!(
            tensor.data().expect("data retrieval should succeed"),
            vec![2.0, 4.0, 6.0, 8.0]
        );

        // Test map
        let original = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("operation should succeed");
        let mapped = original.map(|x| x + 1.0).expect("operation should succeed");
        assert_eq!(
            mapped.data().expect("data retrieval should succeed"),
            vec![2.0, 3.0, 4.0]
        );
        assert_eq!(
            original.data().expect("data retrieval should succeed"),
            vec![1.0, 2.0, 3.0]
        ); // Original unchanged
    }

    #[test]
    fn test_activation_functions() {
        let data = vec![-1.0f32, 0.0, 1.0, 2.0];
        let tensor =
            Tensor::from_data(data, vec![4], DeviceType::Cpu).expect("operation should succeed");

        // Test ReLU
        let relu_result = tensor.relu().expect("relu should succeed");
        assert_eq!(
            relu_result.data().expect("data retrieval should succeed"),
            vec![0.0, 0.0, 1.0, 2.0]
        );

        // Test abs
        let abs_result = tensor.abs().expect("abs computation should succeed");
        assert_eq!(
            abs_result.data().expect("data retrieval should succeed"),
            vec![1.0, 0.0, 1.0, 2.0]
        );

        // Test clamp
        let clamped = tensor.clamp(-0.5, 1.5).expect("operation should succeed");
        assert_eq!(
            clamped.data().expect("data retrieval should succeed"),
            vec![-0.5, 0.0, 1.0, 1.5]
        );
    }

    #[test]
    fn test_storage_sharing() {
        let tensor1 =
            Tensor::<f32>::zeros(&[2, 2], DeviceType::Cpu).expect("operation should succeed");
        let tensor2 = tensor1.clone();
        let tensor3 = tensor1.clone_data();

        assert!(tensor1.shares_storage(&tensor2));
        assert!(!tensor1.shares_storage(&tensor3));
    }

    #[test]
    fn test_basic_matmul() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("operation should succeed");
        let b = Tensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], DeviceType::Cpu)
            .expect("operation should succeed");

        let result = a.basic_matmul(&b).expect("operation should succeed");
        assert_eq!(result.shape().dims(), &[2, 2]);

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        let expected = vec![19.0, 22.0, 43.0, 50.0];
        assert_eq!(
            result.data().expect("data retrieval should succeed"),
            expected
        );
    }

    #[test]
    fn test_reductions() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor =
            Tensor::from_data(data, vec![4], DeviceType::Cpu).expect("operation should succeed");

        let sum = tensor.sum().expect("sum should succeed");
        assert_eq!(sum.item().expect("item extraction should succeed"), 10.0);

        let mean = tensor.mean(None, false).expect("operation should succeed");
        assert_eq!(mean.item().expect("item extraction should succeed"), 2.5);
    }

    #[test]
    fn test_copy_on_write() {
        let mut tensor1 =
            Tensor::<f32>::ones(&[2], DeviceType::Cpu).expect("operation should succeed");
        let tensor2 = tensor1.clone();

        // Both should share storage initially
        assert!(tensor1.shares_storage(&tensor2));

        // After make_unique, they should not share storage
        tensor1.make_unique().expect("make_unique should succeed");
        assert!(!tensor1.shares_storage(&tensor2));
    }

    #[test]
    fn test_item_extraction() {
        let scalar = Tensor::from_data(vec![42.0f32], vec![], DeviceType::Cpu)
            .expect("operation should succeed");
        assert_eq!(scalar.item().expect("item extraction should succeed"), 42.0);

        let vector = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu)
            .expect("operation should succeed");
        assert!(vector.item().is_err()); // Should fail for multi-element tensor
    }

    #[test]
    fn test_all_dim() {
        // Shape [2, 3]: [[1, 0, 1], [1, 1, 1]]
        let data = vec![1i32, 0, 1, 1, 1, 1];
        let tensor =
            Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).expect("tensor creation should succeed");

        // all along dim 0 (rows): per column check
        // col0: 1&&1=true, col1: 0&&1=false, col2: 1&&1=true
        let result = tensor.all_dim(0, false).expect("all_dim should succeed");
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(
            result.to_vec().expect("to_vec should succeed"),
            vec![true, false, true]
        );

        // all along dim 1 (cols): per row check
        // row0: 1&&0&&1=false, row1: 1&&1&&1=true
        let result_row = tensor.all_dim(1, false).expect("all_dim should succeed");
        assert_eq!(result_row.shape().dims(), &[2]);
        assert_eq!(
            result_row.to_vec().expect("to_vec should succeed"),
            vec![false, true]
        );

        // keepdim=true preserves dimension
        let result_kd = tensor.all_dim(1, true).expect("all_dim should succeed");
        assert_eq!(result_kd.shape().dims(), &[2, 1]);
    }

    #[test]
    fn test_any_dim() {
        // Shape [2, 3]: [[0, 0, 0], [0, 1, 0]]
        let data = vec![0i32, 0, 0, 0, 1, 0];
        let tensor =
            Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).expect("tensor creation should succeed");

        // any along dim 0: col0: false, col1: true, col2: false
        let result = tensor.any_dim(0, false).expect("any_dim should succeed");
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(
            result.to_vec().expect("to_vec should succeed"),
            vec![false, true, false]
        );

        // any along dim 1: row0: false, row1: true
        let result_row = tensor.any_dim(1, false).expect("any_dim should succeed");
        assert_eq!(result_row.shape().dims(), &[2]);
        assert_eq!(
            result_row.to_vec().expect("to_vec should succeed"),
            vec![false, true]
        );
    }

    #[test]
    fn test_cat_multidim() {
        // Test concatenation along dim 0
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        let b = Tensor::from_data(vec![5.0f32, 6.0], vec![1, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");

        let cat0 = Tensor::<f32>::cat(&[&a, &b], 0).expect("cat should succeed");
        assert_eq!(cat0.shape().dims(), &[3, 2]);
        assert_eq!(
            cat0.to_vec().expect("to_vec should succeed"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // Test concatenation along dim 1
        let c = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");
        let d = Tensor::from_data(vec![5.0f32, 6.0, 7.0, 8.0], vec![2, 2], DeviceType::Cpu)
            .expect("tensor creation should succeed");

        let cat1 = Tensor::<f32>::cat(&[&c, &d], 1).expect("cat should succeed");
        assert_eq!(cat1.shape().dims(), &[2, 4]);
        assert_eq!(
            cat1.to_vec().expect("to_vec should succeed"),
            vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_topk_along_dim() {
        // 2x4 tensor, topk along dim 1
        let data = vec![3.0f32, 1.0, 4.0, 2.0, 5.0, 9.0, 2.0, 6.0];
        let tensor =
            Tensor::from_data(data, vec![2, 4], DeviceType::Cpu).expect("tensor creation should succeed");

        let (vals, idxs) = tensor.topk(2, Some(1), true, true).expect("topk should succeed");
        assert_eq!(vals.shape().dims(), &[2, 2]);
        assert_eq!(idxs.shape().dims(), &[2, 2]);

        // Row 0: [3, 1, 4, 2] -> top2 = [4, 3] at positions [2, 0]
        // Row 1: [5, 9, 2, 6] -> top2 = [9, 6] at positions [1, 3]
        let vals_data = vals.to_vec().expect("to_vec should succeed");
        let idxs_data = idxs.to_vec().expect("to_vec should succeed");
        assert_eq!(vals_data[0], 4.0);
        assert_eq!(vals_data[1], 3.0);
        assert_eq!(vals_data[2], 9.0);
        assert_eq!(vals_data[3], 6.0);
        assert_eq!(idxs_data[0], 2);
        assert_eq!(idxs_data[1], 0);
        assert_eq!(idxs_data[2], 1);
        assert_eq!(idxs_data[3], 3);
    }

    // --- Regression tests for issue #43: mean must propagate requires_grad ---

    #[test]
    fn test_issue_43_mean_propagates_requires_grad() {
        // A tensor with requires_grad=true; mean result must also require grad.
        let input = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);

        let result = input.mean(None, false).expect("mean should succeed");
        assert!(
            result.requires_grad(),
            "mean result must have requires_grad=true when input does"
        );
    }

    #[test]
    fn test_issue_43_mean_no_requires_grad_when_input_has_none() {
        // When input does not require grad, result should not either.
        let input = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("tensor creation failed");

        let result = input.mean(None, false).expect("mean should succeed");
        assert!(
            !result.requires_grad(),
            "mean result must not require grad when input does not"
        );
    }

    #[test]
    fn test_issue_43_mean_backward() {
        // For mean of n elements, backward with upstream grad=1 distributes 1/n to each element.
        // mean() already reduces to a scalar, so backward() can be called directly.
        let n = 4usize;
        let input = Tensor::from_data(vec![2.0f32, 4.0, 6.0, 8.0], vec![n], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);

        let result = input.mean(None, false).expect("mean should succeed");
        assert!(result.requires_grad(), "mean result must track gradients");
        // mean(None) with keepdim=false produces a scalar (numel=1), so backward is valid
        result.backward().expect("backward should succeed");

        let grad = input.grad().expect("input must have gradient after backward");
        let grad_data = grad.data().expect("gradient data");

        // Each element should receive 1.0 / n = 0.25
        let expected = 1.0f32 / n as f32;
        for &g in &grad_data {
            assert!(
                (g - expected).abs() < 1e-6,
                "each element grad should be 1/n={expected}, got {g}"
            );
        }
    }
}
