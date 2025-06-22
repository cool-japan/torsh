//! Tensor operations

use crate::{FloatElement, Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};

/// Element-wise operations
impl<
        T: TensorElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    > Tensor<T>
{
    /// Element-wise addition with broadcasting
    pub fn add(&self, other: &Self) -> Result<Self> {
        let mut result = self.broadcast_binary_op(other, |a, b| a + b)?;

        // Track the operation for gradient computation
        if self.requires_grad || other.requires_grad {
            use std::sync::Arc;
            result.requires_grad = true;
            result.operation = crate::Operation::Add {
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(other.clone()),
            };
        }

        Ok(result)
    }

    /// Element-wise subtraction with broadcasting
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.broadcast_binary_op(other, |a, b| a - b)
    }

    /// Element-wise multiplication with broadcasting
    pub fn mul(&self, other: &Self) -> Result<Self> {
        let mut result = self.broadcast_binary_op(other, |a, b| a * b)?;

        // Track the operation for gradient computation
        if self.requires_grad || other.requires_grad {
            use std::sync::Arc;
            result.requires_grad = true;
            result.operation = crate::Operation::Mul {
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(other.clone()),
            };
        }

        Ok(result)
    }

    /// Element-wise division with broadcasting
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.broadcast_binary_op(other, |a, b| a / b)
    }

    /// Generic broadcasting binary operation
    fn broadcast_binary_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(T, T) -> T,
        T: Copy + Default,
    {
        // Check if tensors are broadcast compatible
        if !self.shape().broadcast_compatible(&other.shape()) {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        // If shapes are identical, use optimized path
        if self.shape() == other.shape() {
            return self.element_wise_op(other, op);
        }

        // Compute broadcasted shape
        let broadcast_shape = self.shape().broadcast_shape(&other.shape())?;
        let broadcast_dims = broadcast_shape.dims();
        let broadcast_size = broadcast_shape.numel();

        let self_data = self.data.lock().unwrap();
        let other_data = other.data.lock().unwrap();

        let mut result_data = Vec::with_capacity(broadcast_size);

        // Compute broadcasting for each element
        for flat_idx in 0..broadcast_size {
            let broadcast_indices = self.flat_to_multi_index(flat_idx, broadcast_dims);

            let self_idx = self.broadcast_index(&broadcast_indices, broadcast_dims);
            let other_idx = other.broadcast_index(&broadcast_indices, broadcast_dims);

            let self_val = self_data[self_idx];
            let other_val = other_data[other_idx];

            result_data.push(op(self_val, other_val));
        }

        Ok(Self::from_data(
            result_data,
            broadcast_dims.to_vec(),
            self.device,
        ))
    }

    /// Element-wise operation for tensors with identical shapes
    fn element_wise_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        let self_data = self.data.lock().unwrap();
        let other_data = other.data.lock().unwrap();

        let result_data: Vec<T> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Convert flat index to multi-dimensional indices
    fn flat_to_multi_index(&self, flat_idx: usize, dims: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; dims.len()];
        let mut remaining = flat_idx;

        for i in (0..dims.len()).rev() {
            let stride = if i == dims.len() - 1 {
                1
            } else {
                dims[i + 1..].iter().product::<usize>()
            };
            indices[i] = remaining / stride;
            remaining %= stride;
        }

        indices
    }

    /// Get the actual index in the tensor data for broadcasting
    fn broadcast_index(&self, broadcast_indices: &[usize], broadcast_dims: &[usize]) -> usize {
        let self_shape = self.shape();
        let self_dims = self_shape.dims();
        let self_ndim = self_dims.len();
        let broadcast_ndim = broadcast_dims.len();

        let mut actual_indices = vec![0; self_ndim];

        // Map from broadcast indices to actual indices (right-aligned)
        for i in 0..self_ndim {
            let broadcast_dim_idx = broadcast_ndim.saturating_sub(self_ndim) + i;
            if broadcast_dim_idx < broadcast_ndim {
                let broadcast_idx = broadcast_indices[broadcast_dim_idx];
                // If dimension is 1, index is always 0 (broadcasting)
                actual_indices[i] = if self_dims[i] == 1 {
                    0
                } else {
                    std::cmp::min(broadcast_idx, self_dims[i] - 1)
                };
            }
        }

        // Convert multi-dimensional index to flat index
        let mut flat_idx = 0;
        let mut stride = 1;
        for i in (0..self_ndim).rev() {
            flat_idx += actual_indices[i] * stride;
            stride *= self_dims[i];
        }

        flat_idx
    }

    /// Element-wise power
    pub fn pow(&self, exponent: f32) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| x.powf(T::from_f64(exponent as f64).unwrap()))
            .collect();

        let mut result = Self::from_data(result_data, self.shape().dims().to_vec(), self.device);
        result.requires_grad = self.requires_grad;

        // Track the operation for gradient computation
        if self.requires_grad {
            use std::sync::Arc;
            result.operation = crate::Operation::Power {
                input: Arc::new(self.clone()),
                exponent,
            };
        }

        Ok(result)
    }

    /// Absolute value (for float types)
    pub fn abs(&self) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data.iter().map(|&x| x.abs()).collect();

        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Negation (for float types)
    pub fn neg(&self) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data.iter().map(|&x| -x).collect();

        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Add scalar
    pub fn add_scalar(&self, scalar: T) -> Result<Self> {
        let self_data = self.data.lock().unwrap();

        let result_data: Vec<T> = self_data.iter().map(|&a| a + scalar).collect();

        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Multiply by scalar
    pub fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        let self_data = self.data.lock().unwrap();

        let result_data: Vec<T> = self_data
            .iter()
            .map(|&a| {
                let scalar_t = T::from_f64(scalar as f64)
                    .unwrap_or_else(|| panic!("Cannot convert f32 to type"));
                a * scalar_t
            })
            .collect();

        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Divide by scalar
    pub fn div_scalar(&self, scalar: f32) -> Result<Self> {
        let self_data = self.data.lock().unwrap();

        let result_data: Vec<T> = self_data
            .iter()
            .map(|&a| {
                let scalar_t = T::from_f64(scalar as f64)
                    .unwrap_or_else(|| panic!("Cannot convert f32 to type"));
                a / scalar_t
            })
            .collect();

        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Power by scalar
    pub fn pow_scalar(&self, exponent: f32) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| x.powf(T::from_f64(exponent as f64).unwrap()))
            .collect();

        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Clamp values
    pub fn clamp(&self, _min: f32, _max: f32) -> Result<Self> {
        // TODO: Implement actual clamp
        Ok(self.clone())
    }

    /// Maximum with another tensor
    pub fn maximum(&self, other: &Self) -> Result<Self>
    where
        T: PartialOrd,
    {
        self.broadcast_binary_op(other, |a, b| if a > b { a } else { b })
    }

    /// Minimum with another tensor
    pub fn minimum(&self, other: &Self) -> Result<Self>
    where
        T: PartialOrd,
    {
        self.broadcast_binary_op(other, |a, b| if a < b { a } else { b })
    }

    /// Broadcast tensor to a given shape
    pub fn broadcast_to(&self, shape: &torsh_core::shape::Shape) -> Result<Self> {
        let target_dims = shape.dims();

        // If shapes are already the same, return clone
        if self.shape().dims() == target_dims {
            return Ok(self.clone());
        }

        // Check if shapes are broadcast compatible
        if !self.shape().broadcast_compatible(shape) {
            return Err(TorshError::ShapeMismatch {
                expected: target_dims.to_vec(),
                got: self.shape().dims().to_vec(),
            });
        }

        let self_data = self.data.lock().unwrap();
        let target_size = shape.numel();
        let mut result_data = Vec::with_capacity(target_size);

        // Generate data according to broadcasting rules
        for flat_idx in 0..target_size {
            let target_indices = self.flat_to_multi_index(flat_idx, target_dims);
            let source_idx = self.broadcast_index(&target_indices, target_dims);
            result_data.push(self_data[source_idx]);
        }

        Ok(Self::from_data(
            result_data,
            target_dims.to_vec(),
            self.device,
        ))
    }

    /// In-place add
    pub fn add_(&mut self, other: &Self) -> Result<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TorshError::Other(
                "Tensors must have same shape for in-place add".to_string(),
            ));
        }

        let mut self_data = self.data.lock().unwrap();
        let other_data = other.data.lock().unwrap();

        for (a, &b) in self_data.iter_mut().zip(other_data.iter()) {
            *a = *a + b;
        }

        Ok(())
    }

    /// In-place subtract
    pub fn sub_(&mut self, other: &Self) -> Result<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TorshError::Other(
                "Tensors must have same shape for in-place sub".to_string(),
            ));
        }

        let mut self_data = self.data.lock().unwrap();
        let other_data = other.data.lock().unwrap();

        for (a, &b) in self_data.iter_mut().zip(other_data.iter()) {
            *a = *a - b;
        }

        Ok(())
    }

    /// In-place multiply
    pub fn mul_(&mut self, other: &Self) -> Result<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TorshError::Other(
                "Tensors must have same shape for in-place mul".to_string(),
            ));
        }

        let mut self_data = self.data.lock().unwrap();
        let other_data = other.data.lock().unwrap();

        for (a, &b) in self_data.iter_mut().zip(other_data.iter()) {
            *a = *a * b;
        }

        Ok(())
    }

    /// In-place multiply by scalar
    pub fn mul_scalar_(&mut self, scalar: f32) -> Result<()> {
        let mut self_data = self.data.lock().unwrap();
        let scalar_t =
            T::from_f64(scalar as f64).unwrap_or_else(|| panic!("Cannot convert f32 to type"));

        for a in self_data.iter_mut() {
            *a = *a * scalar_t;
        }

        Ok(())
    }

    /// In-place add scalar
    pub fn add_scalar_(&mut self, scalar: f32) -> Result<()> {
        let mut self_data = self.data.lock().unwrap();
        let scalar_t =
            T::from_f64(scalar as f64).unwrap_or_else(|| panic!("Cannot convert f32 to type"));

        for a in self_data.iter_mut() {
            *a = *a + scalar_t;
        }

        Ok(())
    }
}

/// Reduction operations
impl<
        T: TensorElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::AddAssign
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd,
    > Tensor<T>
{
    /// Sum all elements
    pub fn sum(&self) -> Result<Self> {
        let data = self.data.lock().unwrap();
        let sum_value = data
            .iter()
            .fold(<T as TensorElement>::zero(), |acc, &x| acc + x);
        Ok(crate::creation::tensor_scalar(sum_value))
    }

    /// Sum along specified dimensions
    pub fn sum_dim(&self, dims: &[i32], keepdim: bool) -> Result<Self> {
        // For now, implement a simplified version for the most common case
        // This should be enough to fix the softmax issue
        if dims.len() == 1 && dims[0] == 1 {
            // Sum along axis 1 (common case for softmax)
            let input_shape = self.shape();
            let input_dims = input_shape.dims();

            if input_dims.len() != 2 {
                // Fallback to full sum for non-2D tensors
                return self.sum();
            }

            let (batch_size, features) = (input_dims[0], input_dims[1]);
            let data = self.data.lock().unwrap();

            let mut result_data = vec![T::default(); batch_size];

            // Sum along axis 1
            for (batch_idx, result_item) in result_data.iter_mut().enumerate().take(batch_size) {
                let mut sum_val = T::default();
                for feat_idx in 0..features {
                    let flat_idx = batch_idx * features + feat_idx;
                    sum_val += data[flat_idx];
                }
                *result_item = sum_val;
            }

            let output_shape = if keepdim {
                crate::Shape::new(vec![batch_size, 1])
            } else {
                crate::Shape::new(vec![batch_size])
            };

            Ok(Self {
                data: std::sync::Arc::new(std::sync::Mutex::new(result_data)),
                shape: output_shape,
                requires_grad: self.requires_grad,
                operation: crate::Operation::Add {
                    lhs: std::sync::Arc::new(self.clone()),
                    rhs: std::sync::Arc::new(self.clone()),
                },
                device: self.device,
                grad: std::sync::Arc::new(std::sync::Mutex::new(None)),
            })
        } else {
            // For all other cases, just return the original tensor (placeholder)
            Ok(self.clone())
        }
    }

    /// Mean of all elements
    pub fn mean(&self) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data.lock().unwrap();
        if data.is_empty() {
            return Err(TorshError::Other(
                "Cannot compute mean of empty tensor".to_string(),
            ));
        }
        let sum_value = data
            .iter()
            .fold(<T as TensorElement>::zero(), |acc, &x| acc + x);
        let count = T::from_f64(data.len() as f64).unwrap();
        let mean_value = sum_value / count;
        Ok(crate::creation::tensor_scalar(mean_value))
    }

    /// Mean along specified dimensions
    pub fn mean_dim(&self, _dims: &[i32], _keepdim: bool) -> Result<Self>
    where
        T: FloatElement,
    {
        // TODO: Implement actual mean_dim
        Ok(self.clone())
    }

    /// Maximum value
    pub fn max(&self) -> Result<Self>
    where
        T: PartialOrd,
    {
        let data = self.data.lock().unwrap();
        if data.is_empty() {
            return Err(TorshError::Other(
                "Cannot compute max of empty tensor".to_string(),
            ));
        }
        let max_value = *data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        Ok(crate::creation::tensor_scalar(max_value))
    }

    /// Minimum value
    pub fn min(&self) -> Result<Self>
    where
        T: PartialOrd,
    {
        let data = self.data.lock().unwrap();
        if data.is_empty() {
            return Err(TorshError::Other(
                "Cannot compute min of empty tensor".to_string(),
            ));
        }
        let min_value = *data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        Ok(crate::creation::tensor_scalar(min_value))
    }

    /// Argmax
    pub fn argmax(&self, _dim: Option<i32>) -> Result<Tensor<i64>> {
        // TODO: Implement using scirs2
        Ok(crate::creation::zeros(&[1]))
    }

    /// Argmin
    pub fn argmin(&self, _dim: Option<i32>) -> Result<Tensor<i64>> {
        // TODO: Implement using scirs2
        Ok(crate::creation::zeros(&[1]))
    }
}

/// Matrix operations
impl<T: TensorElement + Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>>
    Tensor<T>
{
    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        // Check shapes for matrix multiplication
        let self_shape = self.shape();
        let other_shape = other.shape();

        if self_shape.ndim() < 2 || other_shape.ndim() < 2 {
            return Err(TorshError::InvalidShape(
                "Both tensors must have at least 2 dimensions for matmul".to_string(),
            ));
        }

        let self_dims = self_shape.dims();
        let other_dims = other_shape.dims();

        if self_dims[self_dims.len() - 1] != other_dims[other_dims.len() - 2] {
            return Err(TorshError::ShapeMismatch {
                expected: vec![self_dims[self_dims.len() - 1]],
                got: vec![other_dims[other_dims.len() - 2]],
            });
        }

        // For 2D matrix multiplication
        if self_shape.ndim() == 2 && other_shape.ndim() == 2 {
            return self.matmul_2d(other);
        }

        // For batched matrix multiplication
        self.matmul_batched(other)
    }

    /// 2D matrix multiplication implementation
    fn matmul_2d(&self, other: &Self) -> Result<Self>
    where
        T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        let self_shape = self.shape();
        let other_shape = other.shape();

        let m = self_shape.dims()[0]; // rows of self
        let k = self_shape.dims()[1]; // cols of self / rows of other
        let n = other_shape.dims()[1]; // cols of other

        let self_data = self.data.lock().unwrap();
        let other_data = other.data.lock().unwrap();

        let mut result_data = vec![T::default(); m * n];

        // Implement standard matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for k_idx in 0..k {
                    let a_val = self_data[i * k + k_idx];
                    let b_val = other_data[k_idx * n + j];
                    sum = sum + (a_val * b_val);
                }
                result_data[i * n + j] = sum;
            }
        }

        Ok(Self::from_data(result_data, vec![m, n], self.device))
    }

    /// Batched matrix multiplication
    fn matmul_batched(&self, other: &Self) -> Result<Self>
    where
        T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
    {
        let self_shape = self.shape();
        let other_shape = other.shape();

        // For now, handle the simple batched case
        // TODO: Implement full broadcasting semantics
        if self_shape.ndim() == 3 && other_shape.ndim() == 3 {
            let batch_size = self_shape.dims()[0];
            let m = self_shape.dims()[1];
            let k = self_shape.dims()[2];
            let n = other_shape.dims()[2];

            if batch_size != other_shape.dims()[0] {
                return Err(TorshError::ShapeMismatch {
                    expected: vec![batch_size],
                    got: vec![other_shape.dims()[0]],
                });
            }

            let self_data = self.data.lock().unwrap();
            let other_data = other.data.lock().unwrap();

            let mut result_data = vec![T::default(); batch_size * m * n];

            for b in 0..batch_size {
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = T::default();
                        for k_idx in 0..k {
                            let a_idx = b * (m * k) + i * k + k_idx;
                            let b_idx = b * (k * n) + k_idx * n + j;
                            let a_val = self_data[a_idx];
                            let b_val = other_data[b_idx];
                            sum = sum + (a_val * b_val);
                        }
                        let result_idx = b * (m * n) + i * n + j;
                        result_data[result_idx] = sum;
                    }
                }
            }

            return Ok(Self::from_data(
                result_data,
                vec![batch_size, m, n],
                self.device,
            ));
        }

        // For other cases, fallback to clone for now
        // TODO: Implement full broadcasting and higher-dimensional batching
        Ok(self.clone())
    }

    /// Transpose (swap last two dimensions)
    pub fn t(&self) -> Result<Self> {
        if self.ndim() < 2 {
            return Err(TorshError::InvalidShape(
                "Tensor must have at least 2 dimensions for transpose".to_string(),
            ));
        }

        let ndim = self.ndim() as i32;
        self.transpose(ndim - 2, ndim - 1)
    }
}

/// Mathematical functions
impl<T: FloatElement> Tensor<T> {
    /// Square root
    pub fn sqrt(&self) -> Result<Self> {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data.iter().map(|&x| x.sqrt()).collect();
        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Exponential function
    pub fn exp(&self) -> Result<Self> {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data.iter().map(|&x| x.exp()).collect();
        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Error function (approximation for f32/f64)
    pub fn erf(&self) -> Result<Self> {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                // Using approximation for erf function
                let x_f64 = TensorElement::to_f64(&x).unwrap_or(0.0);
                let erf_result = erf_approx(x_f64);
                T::from_f64(erf_result).unwrap_or_else(|| <T as TensorElement>::zero())
            })
            .collect();
        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }
}

/// Approximation of the error function
fn erf_approx(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Activation functions
impl<T: FloatElement> Tensor<T> {
    /// ReLU activation
    pub fn relu(&self) -> Result<Self> {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let zero = <T as TensorElement>::zero();
                if x > zero {
                    x
                } else {
                    zero
                }
            })
            .collect();
        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Result<Self> {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let one = <T as TensorElement>::one();
                one / (one + (-x).exp())
            })
            .collect();
        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Tanh activation
    pub fn tanh(&self) -> Result<Self> {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data.iter().map(|&x| x.tanh()).collect();
        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Natural logarithm
    pub fn log(&self) -> Result<Self> {
        let data = self.data.lock().unwrap();
        let result_data: Vec<T> = data.iter().map(|&x| x.ln()).collect();
        Ok(Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        ))
    }

    /// Softmax along dimension
    pub fn softmax(&self, _dim: i32) -> Result<Self> {
        // TODO: Implement actual softmax
        Ok(self.clone())
    }

    /// Log softmax along dimension
    pub fn log_softmax(&self, _dim: i32) -> Result<Self> {
        // TODO: Implement actual log_softmax
        Ok(self.clone())
    }
}

/// Loss functions
impl<T: FloatElement> Tensor<T> {
    /// Mean squared error loss
    pub fn mse_loss(&self, target: &Self) -> Result<Self> {
        if self.shape() != target.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: target.shape().dims().to_vec(),
            });
        }

        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Cross entropy loss
    pub fn cross_entropy(&self, _target: &Tensor<i64>) -> Result<Self> {
        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Binary cross entropy loss
    pub fn bce_loss(&self, target: &Self) -> Result<Self> {
        if self.shape() != target.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: target.shape().dims().to_vec(),
            });
        }

        // TODO: Implement using scirs2
        Ok(self.clone())
    }
}

/// Comparison operations
impl<T: TensorElement> Tensor<T> {
    /// Element-wise equality
    pub fn eq(&self, _other: &Self) -> Result<Tensor<bool>> {
        // TODO: Implement using scirs2
        Ok(crate::creation::zeros(&[1]))
    }

    /// Element-wise inequality
    pub fn ne(&self, _other: &Self) -> Result<Tensor<bool>> {
        // TODO: Implement using scirs2
        Ok(crate::creation::zeros(&[1]))
    }

    /// Element-wise greater than
    pub fn gt(&self, _other: &Self) -> Result<Tensor<bool>> {
        // TODO: Implement using scirs2
        Ok(crate::creation::zeros(&[1]))
    }

    /// Element-wise less than
    pub fn lt(&self, _other: &Self) -> Result<Tensor<bool>> {
        // TODO: Implement using scirs2
        Ok(crate::creation::zeros(&[1]))
    }

    /// Element-wise greater than or equal
    pub fn ge(&self, _other: &Self) -> Result<Tensor<bool>> {
        // TODO: Implement using scirs2
        Ok(crate::creation::zeros(&[1]))
    }

    /// Element-wise less than or equal
    pub fn le(&self, _other: &Self) -> Result<Tensor<bool>> {
        // TODO: Implement using scirs2
        Ok(crate::creation::zeros(&[1]))
    }
}

/// Operator overloading
impl<
        T: TensorElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    > std::ops::Add for &Tensor<T>
{
    type Output = Result<Tensor<T>>;

    fn add(self, other: Self) -> Self::Output {
        Tensor::add(self, other)
    }
}

impl<
        T: TensorElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    > std::ops::Sub for &Tensor<T>
{
    type Output = Result<Tensor<T>>;

    fn sub(self, other: Self) -> Self::Output {
        Tensor::sub(self, other)
    }
}

impl<
        T: TensorElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    > std::ops::Mul for &Tensor<T>
{
    type Output = Result<Tensor<T>>;

    fn mul(self, other: Self) -> Self::Output {
        Tensor::mul(self, other)
    }
}

impl<
        T: TensorElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    > std::ops::Div for &Tensor<T>
{
    type Output = Result<Tensor<T>>;

    fn div(self, other: Self) -> Self::Output {
        Tensor::div(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_matrix_multiplication_2d() {
        // Test 2x3 * 3x2 = 2x2
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        );
        let b = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
            DeviceType::Cpu,
        );

        let result = a.matmul(&b).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);

        let data = result.data.lock().unwrap();
        // Expected: [[22, 28], [49, 64]]
        // [1,2,3] * [1,2; 3,4; 5,6] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // [4,5,6] * [1,2; 3,4; 5,6] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert_eq!(data[0], 22.0);
        assert_eq!(data[1], 28.0);
        assert_eq!(data[2], 49.0);
        assert_eq!(data[3], 64.0);
    }

    #[test]
    fn test_element_wise_operations() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu);
        let b = Tensor::from_data(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2], DeviceType::Cpu);

        // Test addition
        let add_result = a.add(&b).unwrap();
        let add_data = add_result.data.lock().unwrap();
        assert_eq!(add_data[0], 3.0);
        assert_eq!(add_data[1], 4.0);
        assert_eq!(add_data[2], 5.0);
        assert_eq!(add_data[3], 6.0);

        // Test multiplication
        let mul_result = a.mul(&b).unwrap();
        let mul_data = mul_result.data.lock().unwrap();
        assert_eq!(mul_data[0], 2.0);
        assert_eq!(mul_data[1], 4.0);
        assert_eq!(mul_data[2], 6.0);
        assert_eq!(mul_data[3], 8.0);
    }
}
