//! Tensor operations
//!
//! This module provides a comprehensive set of tensor operations. The operations
//! are now organized into specialized modules for better maintainability:
//!
//! - `ops::arithmetic` - Element-wise arithmetic operations (add, sub, mul, div, pow, etc.)
//! - `ops::reduction` - Reduction operations (sum, mean, min, max, etc.)
//! - `ops::matrix` - Matrix operations (matmul, transpose, inverse, etc.)
//! - `ops::math` - Mathematical functions (sin, cos, exp, log, sqrt, etc.)
//! - `ops::activation` - Activation functions (relu, sigmoid, tanh, softmax, etc.)
//! - `ops::loss` - Loss functions (mse_loss, cross_entropy, etc.)
//! - `ops::comparison` - Comparison operations (eq, ne, gt, lt, etc.)
//! - `ops::shape` - Shape manipulation (cat, stack, split, reshape, etc.)
//! - `ops::quantization` - Quantization operations (quantize, dequantize, etc.)
//! - `ops::signal` - Signal processing (FFT, convolution, etc.)
//! - `ops::conversion` - Type conversion and promotion
//! - `ops::simd` - SIMD-optimized operations

// ========================================
// NEW MODULAR ORGANIZATION
// ========================================

/// Modular tensor operations organized by category
pub mod ops;

// Re-export all operations for backward compatibility
pub use ops::*;

// ========================================
// EXISTING FUNCTIONALITY (maintained for compatibility)
// ========================================

use crate::{FloatElement, Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};
use torsh_core::dtype::DType;
use scirs2_core::numeric::ToPrimitive;

// Import SIMD operations for performance optimization
#[cfg(feature = "simd")]
use torsh_backend::cpu::simd::{
    should_use_simd, simd_add_f32, simd_div_f32, simd_mul_f32, simd_sub_f32,
};

// Fallback functions when SIMD is not available
#[cfg(not(feature = "simd"))]
#[allow(dead_code)]
fn should_use_simd(_size: usize) -> bool {
    false
}

/// SIMD operation types for optimized tensor operations
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum SimdOpType {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

/// Element-wise operations (basic arithmetic)
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
    /// Element-wise addition with broadcasting (ops module implementation)
    pub fn add_op(&self, other: &Self) -> Result<Self> {
        // Try SIMD optimization for f32 tensors with identical shapes
        let mut result = if self.shape() == other.shape()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            #[cfg(feature = "simd")]
            {
                if should_use_simd(self.numel()) {
                    self.element_wise_op_simd_f32(other, SimdOpType::Add)
                        .unwrap_or_else(|_| self.element_wise_op(other, |a, b| a + b).unwrap())
                } else {
                    self.element_wise_op(other, |a, b| a + b)?
                }
            }
            #[cfg(not(feature = "simd"))]
            {
                self.element_wise_op(other, |a, b| a + b)?
            }
        } else {
            self.broadcast_binary_op(other, |a, b| a + b)?
        };

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

    /// Element-wise addition with broadcasting (standard API)
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.add_op(other)
    }

    /// Element-wise subtraction with broadcasting
    pub fn sub(&self, other: &Self) -> Result<Self> {
        // Try SIMD optimization for f32 tensors with identical shapes
        if self.shape() == other.shape()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            #[cfg(feature = "simd")]
            {
                if should_use_simd(self.numel()) {
                    return self
                        .element_wise_op_simd_f32(other, SimdOpType::Sub)
                        .or_else(|_| self.element_wise_op(other, |a, b| a - b));
                }
            }
            self.element_wise_op(other, |a, b| a - b)
        } else {
            self.broadcast_binary_op(other, |a, b| a - b)
        }
    }

    /// Element-wise multiplication with broadcasting (ops module implementation)
    pub fn mul_op(&self, other: &Self) -> Result<Self> {
        // Try SIMD optimization for f32 tensors with identical shapes
        let mut result = if self.shape() == other.shape()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            #[cfg(feature = "simd")]
            {
                if should_use_simd(self.numel()) {
                    self.element_wise_op_simd_f32(other, SimdOpType::Mul)
                        .unwrap_or_else(|_| self.element_wise_op(other, |a, b| a * b).unwrap())
                } else {
                    self.element_wise_op(other, |a, b| a * b)?
                }
            }
            #[cfg(not(feature = "simd"))]
            {
                self.element_wise_op(other, |a, b| a * b)?
            }
        } else {
            self.broadcast_binary_op(other, |a, b| a * b)?
        };

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

    /// Element-wise multiplication with broadcasting (standard API)
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.mul_op(other)
    }

    /// Element-wise division with broadcasting
    pub fn div(&self, other: &Self) -> Result<Self> {
        // Try SIMD optimization for f32 tensors with identical shapes
        if self.shape() == other.shape()
            && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
        {
            #[cfg(feature = "simd")]
            {
                if should_use_simd(self.numel()) {
                    return self
                        .element_wise_op_simd_f32(other, SimdOpType::Div)
                        .or_else(|_| self.element_wise_op(other, |a, b| a / b));
                }
            }
            self.element_wise_op(other, |a, b| a / b)
        } else {
            self.broadcast_binary_op(other, |a, b| a / b)
        }
    }

    /// Generic broadcasting binary operation with comprehensive error handling
    fn broadcast_binary_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(T, T) -> T,
        T: Copy + Default,
    {
        use crate::broadcast::{BroadcastOps, BroadcastShape};

        // Validate the broadcast operation
        BroadcastOps::validate_broadcast_operation(
            self.shape().dims(),
            other.shape().dims(),
            "binary operation",
        )?;

        // If shapes are identical, use optimized path
        if self.shape() == other.shape() {
            return self.element_wise_op(other, op);
        }

        // Check if broadcasting is memory efficient
        if !self.shape().is_broadcast_efficient(&other.shape()) {
            eprintln!(
                "Warning: Broadcasting shapes {:?} and {:?} may use significant memory",
                self.shape().dims(),
                other.shape().dims()
            );
        }

        // Compute broadcasted shape using the new implementation
        let broadcast_shape = self.shape().broadcast_shape(&other.shape())?;
        let broadcast_dims = broadcast_shape.dims();
        let broadcast_size = broadcast_shape.numel();

        // Estimate memory requirements
        let element_size = std::mem::size_of::<T>();
        let _memory_required = BroadcastOps::estimate_broadcast_memory(
            self.shape().dims(),
            other.shape().dims(),
            element_size,
        )?;

        let self_data = self.data()?;
        let other_data = other.data()?;

        let mut result_data = Vec::with_capacity(broadcast_size);

        // Compute broadcasting for each element using optimized indexing
        for flat_idx in 0..broadcast_size {
            let broadcast_indices = BroadcastOps::flat_to_multi_index(flat_idx, broadcast_dims);

            let self_idx = BroadcastOps::compute_broadcast_index(
                &broadcast_indices,
                self.shape().dims(),
                broadcast_dims,
            )?;
            let other_idx = BroadcastOps::compute_broadcast_index(
                &broadcast_indices,
                other.shape().dims(),
                broadcast_dims,
            )?;

            let self_val = self_data[self_idx];
            let other_val = other_data[other_idx];

            result_data.push(op(self_val, other_val));
        }

        Self::from_data(
            result_data,
            broadcast_dims.to_vec(),
            self.device,
        )
    }

    /// Element-wise operation for tensors with identical shapes
    fn element_wise_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        let self_data = self.data()?;
        let other_data = other.data()?;

        let result_data: Vec<T> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// SIMD-optimized element-wise operation for f32 tensors with identical shapes (basic ops)
    #[cfg(feature = "simd")]
    fn element_wise_op_simd_f32(&self, other: &Self, op_type: SimdOpType) -> Result<Self>
    where
        T: Copy + 'static,
    {
        // Only proceed if T is f32
        if std::any::TypeId::of::<T>() != std::any::TypeId::of::<f32>() {
            return Err(TorshError::InvalidArgument(
                "SIMD optimization only available for f32 tensors".to_string(),
            ));
        }

        let self_data = self.data()?;
        let other_data = other.data()?;

        // Check if SIMD is beneficial
        let size = self_data.len();
        if !should_use_simd(size) {
            // Fall back to regular element-wise operation
            drop(self_data);
            drop(other_data);
            return match op_type {
                SimdOpType::Add => self.element_wise_op(other, |a, b| a + b),
                SimdOpType::Sub => self.element_wise_op(other, |a, b| a - b),
                SimdOpType::Mul => self.element_wise_op(other, |a, b| a * b),
                SimdOpType::Div => self.element_wise_op(other, |a, b| a / b),
                _ => Err(TorshError::InvalidArgument(
                    "Min/Max operations require PartialOrd trait".to_string(),
                )),
            };
        }

        // Cast to f32 slices for SIMD operations
        let self_f32 =
            unsafe { std::slice::from_raw_parts(self_data.as_ptr() as *const f32, size) };
        let other_f32 =
            unsafe { std::slice::from_raw_parts(other_data.as_ptr() as *const f32, size) };

        let mut result_f32 = vec![0.0f32; size];

        // Apply SIMD operation
        match op_type {
            SimdOpType::Add => simd_add_f32(self_f32, other_f32, &mut result_f32),
            SimdOpType::Sub => simd_sub_f32(self_f32, other_f32, &mut result_f32),
            SimdOpType::Mul => simd_mul_f32(self_f32, other_f32, &mut result_f32),
            SimdOpType::Div => simd_div_f32(self_f32, other_f32, &mut result_f32),
            _ => {
                return Err(TorshError::InvalidArgument(
                    "Min/Max operations require PartialOrd trait".to_string(),
                ));
            }
        }

        // Cast result back to T
        let result_data =
            unsafe { std::slice::from_raw_parts(result_f32.as_ptr() as *const T, size).to_vec() };

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Fallback for SIMD operations when SIMD feature is not enabled
    #[cfg(not(feature = "simd"))]
    #[allow(dead_code)]
    fn element_wise_op_simd_f32(&self, other: &Self, op_type: SimdOpType) -> Result<Self>
    where
        T: Copy + 'static,
    {
        match op_type {
            SimdOpType::Add => self.element_wise_op(other, |a, b| a + b),
            SimdOpType::Sub => self.element_wise_op(other, |a, b| a - b),
            SimdOpType::Mul => self.element_wise_op(other, |a, b| a * b),
            SimdOpType::Div => self.element_wise_op(other, |a, b| a / b),
            _ => Err(TorshError::InvalidArgument(
                "Min/Max operations require PartialOrd trait".to_string(),
            )),
        }
    }

    /// In-place element-wise operation for tensors with identical shapes
    #[allow(dead_code)]
    fn element_wise_op_inplace<F>(&mut self, other: &Self, op: F) -> Result<()>
    where
        F: Fn(T, T) -> T,
        T: Copy,
    {
        let other_data = other.data()?;

        let mut self_data = self.data()?;

        // Modify self data in place
        for (self_val, &other_val) in self_data.iter_mut().zip(other_data.iter()) {
            *self_val = op(*self_val, other_val);
        }

        Ok(())
    }

    /// In-place broadcasting binary operation
    #[allow(dead_code)]
    fn broadcast_binary_op_inplace<F>(&mut self, other: &Self, op: F) -> Result<()>
    where
        F: Fn(T, T) -> T,
        T: Copy + Default,
    {
        let other_data = other.data()?;

        let mut self_data = self.data()?;

        let self_shape = self.shape();
        let self_dims = self_shape.dims();
        let _other_dims = other.shape().dims();
        let self_size = self_shape.numel();

        // Modify self data in place with broadcasting
        #[allow(clippy::needless_range_loop)]
        for flat_idx in 0..self_size {
            let self_indices = self.flat_to_multi_index_ops(flat_idx, self_dims);
            let other_idx = other.broadcast_index(&self_indices, self_dims);

            if other_idx < other_data.len() {
                self_data[flat_idx] = op(self_data[flat_idx], other_data[other_idx]);
            }
        }

        Ok(())
    }

    /// Element-wise power
    pub fn pow(&self, exponent: f32) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let exp_val = T::from_f64(exponent as f64).ok_or_else(|| {
                    TorshError::ConversionError(format!(
                        "Cannot convert {exponent} to element type"
                    ))
                })?;
                Ok(x.powf(exp_val))
            })
            .collect::<Result<Vec<T>>>()?;

        let mut result = Self::from_data(result_data, self.shape().dims().to_vec(), self.device)?;
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

    /// Element-wise power with tensor exponent
    pub fn pow_tensor(&self, exponent: &Self) -> Result<Self>
    where
        T: FloatElement,
    {
        // Check shapes are compatible for broadcasting
        if self.shape() != exponent.shape() {
            return Err(TorshError::BroadcastError {
                shape1: self.shape().dims().to_vec(),
                shape2: exponent.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let exp_data = exponent.data()?;

        let result_data: Vec<T> = self_data
            .iter()
            .zip(exp_data.iter())
            .map(|(&base, &exp)| base.powf(exp))
            .collect();

        let mut result = Self::from_data(result_data, self.shape().dims().to_vec(), self.device)?;
        result.requires_grad = self.requires_grad || exponent.requires_grad;

        // NOTE: Gradient tracking for power operation requires autograd integration
        // This would involve registering the operation with the computation graph
        // and implementing backward pass: d/dx(x^y) = y * x^(y-1), d/dy(x^y) = x^y * ln(x)

        Ok(result)
    }


    /// Negation (for float types)
    pub fn neg(&self) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| -x).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Add scalar
    pub fn add_scalar(&self, scalar: T) -> Result<Self> {
        let self_data = self.data()?;

        let result_data: Vec<T> = self_data.iter().map(|&a| a + scalar).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }


    /// Divide by scalar
    pub fn div_scalar(&self, scalar: f32) -> Result<Self> {
        let self_data = self.data()?;

        let result_data: Vec<T> = self_data
            .iter()
            .map(|&a| {
                let scalar_t = T::from_f64(scalar as f64)
                    .unwrap_or_else(|| panic!("Cannot convert f32 to type"));
                a / scalar_t
            })
            .collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Power by scalar
    pub fn pow_scalar(&self, exponent: f32) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| x.powf(T::from_f64(exponent as f64).unwrap()))
            .collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Clamp values between min and max
    pub fn clamp(&self, min: f32, max: f32) -> Result<Self>
    where
        T: PartialOrd,
    {
        let min_t = T::from_f64(min as f64).unwrap_or_else(|| panic!("Cannot convert min to type"));
        let max_t = T::from_f64(max as f64).unwrap_or_else(|| panic!("Cannot convert max to type"));
        
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&item| {
            if item < min_t {
                min_t
            } else if item > max_t {
                max_t
            } else {
                item
            }
        }).collect();
        
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Concatenate tensors along an existing dimension
    pub fn cat(tensors: &[&Self], dim: i32) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot concatenate empty tensor list".to_string()));
        }

        let first = tensors[0];
        let first_shape_binding = first.shape();
        let first_shape = first_shape_binding.dims();
        let ndim = first_shape.len() as i32;
        
        // Normalize dimension
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(TorshError::InvalidArgument(format!("Dimension {dim} out of range for {ndim}-d tensor")));
        }
        let dim = dim as usize;

        // Validate all tensors have compatible shapes
        let mut total_size_on_dim = 0;
        for tensor in tensors {
            let shape_binding = tensor.shape();
            let shape = shape_binding.dims();
            if shape.len() != first_shape.len() {
                return Err(TorshError::InvalidArgument("All tensors must have the same number of dimensions".to_string()));
            }
            for (i, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i != dim && s1 != s2 {
                    return Err(TorshError::InvalidArgument(format!("Sizes must match except in concatenation dimension {dim}")));
                }
            }
            total_size_on_dim += shape[dim];
        }

        // Create output shape
        let mut output_shape = first_shape.to_vec();
        output_shape[dim] = total_size_on_dim;

        // Concatenate data efficiently
        let mut output_data = Vec::with_capacity(output_shape.iter().product());
        
        // Calculate elements before and after the concatenation dimension
        let elements_before: usize = first_shape[..dim].iter().product();
        let elements_after: usize = first_shape[dim + 1..].iter().product();
        
        for before_idx in 0..elements_before {
            for tensor in tensors {
                let tensor_data = tensor.data()?;
                let tensor_shape_binding = tensor.shape();
                let tensor_shape = tensor_shape_binding.dims();
                let tensor_size_on_dim = tensor_shape[dim];
                
                for dim_idx in 0..tensor_size_on_dim {
                    for after_idx in 0..elements_after {
                        let source_idx = before_idx * tensor_shape[dim] * elements_after + 
                                       dim_idx * elements_after + after_idx;
                        if source_idx < tensor_data.len() {
                            output_data.push(tensor_data[source_idx]);
                        } else {
                            output_data.push(T::default());
                        }
                    }
                }
            }
        }

        Self::from_data(output_data, output_shape, first.device)
    }

    /// Stack tensors along a new dimension
    pub fn stack(tensors: &[&Self], dim: i32) -> Result<Self> {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot stack empty tensor list".to_string()));
        }

        let first = tensors[0];
        let first_shape_binding = first.shape();
        let first_shape = first_shape_binding.dims();
        let ndim = first_shape.len() as i32;
        
        // Normalize dimension (can be from -ndim-1 to ndim for stacking)
        let dim = if dim < 0 { ndim + 1 + dim } else { dim };
        if dim < 0 || dim > ndim {
            return Err(TorshError::InvalidArgument(format!("Dimension {dim} out of range for stacking")));
        }
        let dim = dim as usize;

        // Validate all tensors have the same shape
        for tensor in tensors {
            let shape_binding = tensor.shape();
            let shape = shape_binding.dims();
            if shape != first_shape {
                return Err(TorshError::InvalidArgument("All tensors must have the same shape for stacking".to_string()));
            }
        }

        // Create output shape (insert new dimension of size = number of tensors)
        let mut output_shape = first_shape.to_vec();
        output_shape.insert(dim, tensors.len());

        // Stack data efficiently
        let mut output_data = Vec::with_capacity(output_shape.iter().product::<usize>());
        
        // Calculate chunk sizes
        let elements_before: usize = first_shape[..dim].iter().product();
        let elements_from_dim: usize = first_shape[dim..].iter().product();

        for before_idx in 0..elements_before {
            for tensor in tensors {
                let tensor_data = tensor.data()?;
                let start_idx = before_idx * elements_from_dim;
                let end_idx = (start_idx + elements_from_dim).min(tensor_data.len());
                output_data.extend_from_slice(&tensor_data[start_idx..end_idx]);
            }
        }

        Self::from_data(output_data, output_shape, first.device)
    }

    /// Split tensor into chunks along a dimension
    pub fn split(&self, split_size: usize, dim: i32) -> Result<Vec<Self>> {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        let ndim = shape.len() as i32;
        
        // Normalize dimension
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(TorshError::InvalidArgument(format!("Dimension {dim} out of range for {ndim}-d tensor")));
        }
        let dim = dim as usize;

        let dim_size = shape[dim];
        if split_size == 0 {
            return Err(TorshError::InvalidArgument("Split size cannot be zero".to_string()));
        }

        let num_splits = dim_size.div_ceil(split_size);
        let mut result = Vec::with_capacity(num_splits);

        for i in 0..num_splits {
            let start = i * split_size;
            let end = ((i + 1) * split_size).min(dim_size);
            let chunk_size = end - start;

            // Create shape for this chunk
            let mut chunk_shape = shape.to_vec();
            chunk_shape[dim] = chunk_size;

            // Extract data for this chunk
            let data = self.data()?;
            let mut chunk_data = Vec::with_capacity(chunk_shape.iter().product());

            let elements_before: usize = shape[..dim].iter().product();
            let elements_after: usize = shape[dim + 1..].iter().product();

            for before_idx in 0..elements_before {
                for dim_idx in start..end {
                    for after_idx in 0..elements_after {
                        let source_idx = before_idx * dim_size * elements_after + 
                                       dim_idx * elements_after + after_idx;
                        if source_idx < data.len() {
                            chunk_data.push(data[source_idx]);
                        } else {
                            chunk_data.push(T::default());
                        }
                    }
                }
            }

            result.push(Self::from_data(chunk_data, chunk_shape, self.device)?);
        }

        Ok(result)
    }

    /// Split tensor into variable-sized sections along a dimension
    pub fn split_sections(&self, split_sizes: &[usize], dim: i32) -> Result<Vec<Self>> {
        if split_sizes.is_empty() {
            return Err(TorshError::InvalidArgument("Split sizes cannot be empty".to_string()));
        }

        let shape = self.shape();
        let dims = shape.dims();
        let ndim = dims.len() as i32;
        
        // Normalize dimension
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        if dim >= dims.len() {
            return Err(TorshError::InvalidArgument(
                format!("Dimension {dim} out of range for tensor with {ndim} dimensions"),
            ));
        }

        let dim_size = dims[dim];
        let total_split_size: usize = split_sizes.iter().sum();
        
        if total_split_size != dim_size {
            return Err(TorshError::InvalidArgument(
                format!("Split sizes sum to {total_split_size} but dimension has size {dim_size}")
            ));
        }

        // Check for zero sizes
        if split_sizes.contains(&0) {
            return Err(TorshError::InvalidArgument("Split sizes cannot contain zero".to_string()));
        }

        let mut result = Vec::with_capacity(split_sizes.len());
        let mut current_start = 0;

        for &split_size in split_sizes {
            let end = current_start + split_size;

            // Create shape for this split
            let mut split_shape = dims.to_vec();
            split_shape[dim] = split_size;

            // Extract data for this split
            let data = self.data()?;
            let mut split_data = Vec::with_capacity(split_shape.iter().product());

            let elements_before: usize = dims[..dim].iter().product();
            let elements_after: usize = dims[dim + 1..].iter().product();

            for before_idx in 0..elements_before {
                for dim_idx in current_start..end {
                    for after_idx in 0..elements_after {
                        let source_idx = before_idx * dim_size * elements_after + 
                                       dim_idx * elements_after + after_idx;
                        if source_idx < data.len() {
                            split_data.push(data[source_idx]);
                        } else {
                            split_data.push(T::default());
                        }
                    }
                }
            }

            result.push(Self::from_data(split_data, split_shape, self.device)?);
            current_start = end;
        }

        Ok(result)
    }

    /// Remove a tensor dimension of size 1 and return separate tensors
    pub fn unbind(&self, dim: i32) -> Result<Vec<Self>> {
        let shape = self.shape();
        let dims = shape.dims();
        let ndim = dims.len() as i32;
        
        // Normalize dimension
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        if dim >= dims.len() {
            return Err(TorshError::InvalidArgument(
                format!("Dimension {dim} out of range for tensor with {ndim} dimensions"),
            ));
        }

        let dim_size = dims[dim];
        let mut result = Vec::with_capacity(dim_size);

        // Create new shape without the unbound dimension
        let mut new_dims = Vec::with_capacity(dims.len() - 1);
        new_dims.extend_from_slice(&dims[..dim]);
        new_dims.extend_from_slice(&dims[dim + 1..]);

        // Handle scalar result case
        if new_dims.is_empty() {
            new_dims = vec![]; // Keep empty for scalars
        }

        let data = self.data()?;
        let elements_before: usize = dims[..dim].iter().product();
        let elements_after: usize = dims[dim + 1..].iter().product();
        let elements_per_slice = elements_after;

        for slice_idx in 0..dim_size {
            let mut slice_data = Vec::with_capacity(elements_per_slice * elements_before);

            for before_idx in 0..elements_before {
                let start_idx = before_idx * dim_size * elements_after + slice_idx * elements_after;
                let end_idx = start_idx + elements_after;
                
                for idx in start_idx..end_idx {
                    if idx < data.len() {
                        slice_data.push(data[idx]);
                    } else {
                        slice_data.push(T::default());
                    }
                }
            }

            result.push(Self::from_data(slice_data, new_dims.clone(), self.device)?);
        }

        Ok(result)
    }

    /// Split tensor into a specific number of chunks along a dimension
    pub fn chunk(&self, chunks: usize, dim: i32) -> Result<Vec<Self>> {
        if chunks == 0 {
            return Err(TorshError::InvalidArgument("Number of chunks cannot be zero".to_string()));
        }

        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        let ndim = shape.len() as i32;
        
        // Normalize dimension
        let dim = if dim < 0 { ndim + dim } else { dim };
        if dim < 0 || dim >= ndim {
            return Err(TorshError::InvalidArgument(format!("Dimension {dim} out of range for {ndim}-d tensor")));
        }
        let dim = dim as usize;

        let dim_size = shape[dim];
        
        // Check if we have too many chunks
        if chunks > dim_size {
            return Err(TorshError::InvalidArgument(
                format!("Cannot create {chunks} chunks from dimension of size {dim_size}")
            ));
        }

        // Calculate chunk sizes - distribute elements as evenly as possible
        let base_chunk_size = dim_size / chunks;
        let remainder = dim_size % chunks;
        
        let mut chunk_sizes = Vec::with_capacity(chunks);
        for i in 0..chunks {
            // First 'remainder' chunks get an extra element
            let size = if i < remainder { 
                base_chunk_size + 1 
            } else { 
                base_chunk_size 
            };
            chunk_sizes.push(size);
        }
        
        self.split_sections(&chunk_sizes, dim as i32)
    }


    /// Expand tensor to a larger size (using i32 dimensions, converts to usize internally)
    pub fn expand_i32(&self, sizes: &[i32]) -> Result<Self> {
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        let mut output_shape = Vec::with_capacity(sizes.len());

        // Handle negative sizes and validate expansion
        for (i, &size) in sizes.iter().enumerate() {
            if size == -1 {
                // -1 means keep the original size
                if i < shape.len() {
                    output_shape.push(shape[i]);
                } else {
                    return Err(TorshError::InvalidArgument(
                        "Cannot infer size for dimension that doesn't exist in original tensor".to_string()
                    ));
                }
            } else if size >= 0 {
                let size = size as usize;
                if i < shape.len() {
                    if shape[i] == 1 || shape[i] == size {
                        output_shape.push(size);
                    } else {
                        return Err(TorshError::InvalidArgument(format!(
                            "Cannot expand dimension {} from size {} to size {}",
                            i, shape[i], size
                        )));
                    }
                } else {
                    // New dimension (must be expanded from implicit size 1)
                    output_shape.push(size);
                }
            } else {
                return Err(TorshError::InvalidArgument("Size cannot be negative except -1".to_string()));
            }
        }

        // Broadcast the tensor to the output shape
        let output_shape_obj = torsh_core::shape::Shape::new(output_shape.clone());
        self.broadcast_to(&output_shape_obj)
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

        let self_data = self.data()?;
        let target_size = shape.numel();
        let mut result_data = Vec::with_capacity(target_size);

        // Generate data according to broadcasting rules
        for flat_idx in 0..target_size {
            let target_indices = self.flat_to_multi_index_ops(flat_idx, target_dims);
            let source_idx = self.broadcast_index(&target_indices, target_dims);
            result_data.push(self_data[source_idx]);
        }

        Self::from_data(
            result_data,
            target_dims.to_vec(),
            self.device,
        )
    }

    /// In-place add
    pub fn add_(&mut self, other: &Self) -> Result<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TorshError::Other(
                "Tensors must have same shape for in-place add".to_string(),
            ));
        }

        let other_data = other.data()?;
        let mut other_iter = other_data.iter();

        self.data_mut_apply(|item| {
            if let Some(&other_val) = other_iter.next() {
                *item = *item + other_val;
            }
        })?;

        Ok(())
    }

    /// In-place subtract
    pub fn sub_(&mut self, other: &Self) -> Result<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TorshError::Other(
                "Tensors must have same shape for in-place sub".to_string(),
            ));
        }

        let other_data = other.data()?;
        let mut other_iter = other_data.iter();

        self.data_mut_apply(|item| {
            if let Some(&other_val) = other_iter.next() {
                *item = *item - other_val;
            }
        })?;

        Ok(())
    }

    /// In-place multiply
    pub fn mul_(&mut self, other: &Self) -> Result<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TorshError::Other(
                "Tensors must have same shape for in-place mul".to_string(),
            ));
        }

        let other_data = other.data()?;
        let mut other_iter = other_data.iter();

        self.data_mut_apply(|item| {
            if let Some(&other_val) = other_iter.next() {
                *item = *item * other_val;
            }
        })?;

        Ok(())
    }

    // mul_scalar_ and add_scalar_ are already implemented in lib.rs with generic type bounds

    /// In-place divide
    pub fn div_(&mut self, other: &Self) -> Result<()> {
        if self.shape().dims() != other.shape().dims() {
            return Err(TorshError::Other(
                "Tensors must have same shape for in-place div".to_string(),
            ));
        }

        let other_data = other.data()?;
        let mut other_iter = other_data.iter();

        self.data_mut_apply(|item| {
            if let Some(&other_val) = other_iter.next() {
                *item = *item / other_val;
            }
        })?;

        Ok(())
    }

    /// In-place subtract scalar
    pub fn sub_scalar_(&mut self, scalar: f32) -> Result<()> {
        let scalar_t =
            T::from_f64(scalar as f64).unwrap_or_else(|| panic!("Cannot convert f32 to type"));

        self.data_mut_apply(|item| {
            *item = *item - scalar_t;
        })?;

        Ok(())
    }

    /// In-place divide by scalar
    pub fn div_scalar_(&mut self, scalar: f32) -> Result<()> {
        let scalar_t =
            T::from_f64(scalar as f64).unwrap_or_else(|| panic!("Cannot convert f32 to type"));

        self.data_mut_apply(|item| {
            *item = *item / scalar_t;
        })?;

        Ok(())
    }

    /// In-place power
    pub fn pow_(&mut self, exponent: f32) -> Result<()>
    where
        T: FloatElement,
    {
        let exp_f64 = exponent as f64;

        self.data_mut_apply(|item| {
            let val_f64 = TensorElement::to_f64(item).unwrap_or(0.0);
            *item = T::from_f64(val_f64.powf(exp_f64)).unwrap_or_else(|| <T as TensorElement>::zero());
        })?;

        Ok(())
    }

    /// In-place clamp
    pub fn clamp_(&mut self, min: f32, max: f32) -> Result<()>
    where
        T: PartialOrd,
    {
        let min_t = T::from_f64(min as f64).unwrap_or_else(|| panic!("Cannot convert min to type"));
        let max_t = T::from_f64(max as f64).unwrap_or_else(|| panic!("Cannot convert max to type"));

        self.data_mut_apply(|item| {
            if *item < min_t {
                *item = min_t;
            } else if *item > max_t {
                *item = max_t;
            }
        })?;

        Ok(())
    }

    /// In-place absolute value
    pub fn abs_(&mut self) -> Result<()>
    where
        T: std::ops::Neg<Output = T> + PartialOrd,
    {
        let zero = T::zero();

        self.data_mut_apply(|item| {
            if *item < zero {
                *item = -*item;
            }
        })?;

        Ok(())
    }

    /// Absolute value (returns new tensor)
    pub fn abs(&self) -> Result<Self>
    where
        T: std::ops::Neg<Output = T> + PartialOrd + Copy,
    {
        let data = self.data()?;
        let zero = T::zero();
        let abs_data: Vec<T> = data.iter().map(|&x| if x < zero { -x } else { x }).collect();
        
        Self::from_data(abs_data, self.shape().dims().to_vec(), self.device())
    }

    /// In-place negation
    pub fn neg_(&mut self) -> Result<()>
    where
        T: std::ops::Neg<Output = T>,
    {
        self.data_mut_apply(|item| {
            *item = -*item;
        })?;

        Ok(())
    }

    /// In-place reciprocal
    pub fn reciprocal_(&mut self) -> Result<()>
    where
        T: FloatElement,
    {
        let one = <T as TensorElement>::one();

        self.data_mut_apply(|item| {
            *item = one / *item;
        })?;

        Ok(())
    }

    /// In-place square root
    pub fn sqrt_(&mut self) -> Result<()>
    where
        T: FloatElement,
    {
        self.data_mut_apply(|item| {
            *item = item.sqrt();
        })?;

        Ok(())
    }

    /// In-place exponential
    pub fn exp_(&mut self) -> Result<()>
    where
        T: FloatElement,
    {
        self.data_mut_apply(|item| {
            *item = item.exp();
        })?;

        Ok(())
    }

    /// In-place logarithm
    pub fn log_(&mut self) -> Result<()>
    where
        T: FloatElement,
    {
        self.data_mut_apply(|item| {
            *item = item.ln();
        })?;

        Ok(())
    }

    /// In-place sine
    pub fn sin_(&mut self) -> Result<()>
    where
        T: FloatElement,
    {
        self.data_mut_apply(|item| {
            *item = item.sin();
        })?;

        Ok(())
    }

    /// In-place cosine
    pub fn cos_(&mut self) -> Result<()>
    where
        T: FloatElement,
    {
        self.data_mut_apply(|item| {
            *item = item.cos();
        })?;

        Ok(())
    }

    /// In-place ReLU
    pub fn relu_(&mut self) -> Result<()>
    where
        T: PartialOrd,
    {
        let zero = T::zero();

        self.data_mut_apply(|item| {
            if *item < zero {
                *item = zero;
            }
        })?;

        Ok(())
    }

    /// In-place sigmoid
    pub fn sigmoid_(&mut self) -> Result<()>
    where
        T: FloatElement,
    {
        let one = <T as TensorElement>::one();

        self.data_mut_apply(|item| {
            *item = one / (one + (-*item).exp());
        })?;

        Ok(())
    }

    /// In-place tanh
    pub fn tanh_(&mut self) -> Result<()>
    where
        T: FloatElement,
    {
        self.data_mut_apply(|item| {
            *item = item.tanh();
        })?;

        Ok(())
    }
}

/// Reduction operations
impl<
        T: TensorElement
            + FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::AddAssign
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + PartialOrd
            + std::ops::MulAssign,
    > Tensor<T>
{
    /// Sum all elements
    pub fn sum(&self) -> Result<Self> {
        let data = self.data()?;
        let sum_value = data
            .iter()
            .fold(<T as TensorElement>::zero(), |acc, &x| acc + x);
        crate::creation::tensor_scalar(sum_value)
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
            let data = self.data()?;

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

            Tensor::from_data(result_data, output_shape.dims().to_vec(), self.device)
        } else {
            // For all other cases, just return the original tensor (placeholder)
            Ok(self.clone())
        }
    }

    /// Mean along specified dimensions
    pub fn mean_dim(&self, dims: &[i32], keepdim: bool) -> Result<Self>
    where
        T: FloatElement,
    {
        // First compute sum along the dimensions
        let sum = self.sum_dim(dims, keepdim)?;

        // Calculate the number of elements along the reduced dimensions
        let mut count = 1;
        for &dim in dims {
            let dim = if dim < 0 {
                (self.ndim() as i32 + dim) as usize
            } else {
                dim as usize
            };
            count *= self.shape().dims()[dim];
        }

        // Divide by count to get mean
        sum.div_scalar(count as f32)
    }

    /// Variance along specified dimensions
    pub fn var_dim(&self, dims: &[i32], keepdim: bool, unbiased: bool) -> Result<Self>
    where
        T: FloatElement,
    {
        // Compute mean along dimensions
        let mean = self.mean_dim(dims, true)?;

        // Compute (x - mean)^2
        let diff = self.sub(&mean)?;
        let sq_diff = diff.pow_scalar(2.0)?;

        // Sum the squared differences
        let sum_sq_diff = sq_diff.sum_dim(dims, keepdim)?;

        // Calculate the number of elements along the reduced dimensions
        let mut count = 1;
        for &dim in dims {
            let dim = if dim < 0 {
                (self.ndim() as i32 + dim) as usize
            } else {
                dim as usize
            };
            count *= self.shape().dims()[dim];
        }

        let divisor = if unbiased {
            (count - 1) as f32
        } else {
            count as f32
        };

        if divisor == 0.0 {
            return Err(TorshError::InvalidArgument(
                "Cannot compute variance with divisor 0".to_string(),
            ));
        }

        // Divide by count (or count-1 for unbiased)
        sum_sq_diff.div_scalar(divisor)
    }

    /// Standard deviation along specified dimensions
    pub fn std_dim(&self, dims: &[i32], keepdim: bool, unbiased: bool) -> Result<Self>
    where
        T: FloatElement,
    {
        self.var_dim(dims, keepdim, unbiased)?.sqrt()
    }


    /// Minimum value
    pub fn min(&self) -> Result<Self>
    where
        T: PartialOrd,
    {
        let data = self.data()?;
        if data.is_empty() {
            return Err(TorshError::Other(
                "Cannot compute min of empty tensor".to_string(),
            ));
        }
        let min_value = *data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        crate::creation::tensor_scalar(min_value)
    }

    /// Cumulative sum along dimension
    pub fn cumsum(&self, dim: i32) -> Result<Self> {
        // For now, implement simple cumsum along last dimension if dim matches
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        if dim as usize >= shape.len() {
            return Err(TorshError::DimensionError(format!(
                "Dimension {dim} out of range for tensor with {dims} dimensions", 
                dims = shape.len()
            )));
        }

        let data = self.data()?;
        let mut result_data = data.clone();
        
        // Simple implementation for 1D case
        if shape.len() == 1 {
            let mut cumsum = T::default();
            for item in &mut result_data {
                cumsum += *item;
                *item = cumsum;
            }
        } else {
            // Multi-dimensional cumsum implementation
            let target_dim = if dim < 0 {
                (shape.len() as i32 + dim) as usize
            } else {
                dim as usize
            };
            
            let dim_size = shape[target_dim];
            let outer_size: usize = shape[..target_dim].iter().product();
            let inner_size: usize = shape[target_dim + 1..].iter().product();
            
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let base_idx = outer * dim_size * inner_size + inner;
                    let mut cumsum = T::default();
                    
                    for d in 0..dim_size {
                        let idx = base_idx + d * inner_size;
                        cumsum += result_data[idx];
                        result_data[idx] = cumsum;
                    }
                }
            }
        }

        Self::from_data(
            result_data,
            shape.to_vec(),
            self.device,
        )
    }

    /// Cumulative product along dimension
    pub fn cumprod(&self, dim: i32) -> Result<Self> {
        // For now, implement simple cumprod along last dimension if dim matches
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        if dim as usize >= shape.len() {
            return Err(TorshError::DimensionError(format!(
                "Dimension {dim} out of range for tensor with {dims} dimensions", 
                dims = shape.len()
            )));
        }

        let data = self.data()?;
        let mut result_data = data.clone();
        
        // Simple implementation for 1D case
        if shape.len() == 1 {
            let mut cumprod = <T as TensorElement>::one();
            for item in &mut result_data {
                cumprod *= *item;
                *item = cumprod;
            }
        } else {
            // Multi-dimensional cumprod implementation
            let target_dim = if dim < 0 {
                (shape.len() as i32 + dim) as usize
            } else {
                dim as usize
            };
            
            let dim_size = shape[target_dim];
            let outer_size: usize = shape[..target_dim].iter().product();
            let inner_size: usize = shape[target_dim + 1..].iter().product();
            
            for outer in 0..outer_size {
                for inner in 0..inner_size {
                    let base_idx = outer * dim_size * inner_size + inner;
                    let mut cumprod = <T as TensorElement>::one();
                    
                    for d in 0..dim_size {
                        let idx = base_idx + d * inner_size;
                        cumprod *= result_data[idx];
                        result_data[idx] = cumprod;
                    }
                }
            }
        }

        Self::from_data(
            result_data,
            shape.to_vec(),
            self.device,
        )
    }

    /// Check if all elements are true (for boolean tensors)
    pub fn all(&self) -> Result<Tensor<bool>> {
        // Convert to boolean and check all values
        let data = self.data()?;
        let all_true = data.iter().all(|&x| {
            // Convert to f64 for comparison
            let val = TensorElement::to_f64(&x).unwrap_or(0.0);
            val != 0.0
        });
        crate::creation::tensor_scalar(all_true)
    }

    /// Check if all elements are true along dimension
    pub fn all_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor<bool>> {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        
        if shape.is_empty() {
            return Err(TorshError::InvalidOperation("Cannot reduce empty tensor".to_string()));
        }
        
        let target_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };
        
        if target_dim >= shape.len() {
            return Err(TorshError::DimensionError(format!(
                "Dimension {dim} out of range for tensor with {dims} dimensions", 
                dims = shape.len()
            )));
        }
        
        let dim_size = shape[target_dim];
        let outer_size: usize = shape[..target_dim].iter().product();
        let inner_size: usize = shape[target_dim + 1..].iter().product();
        
        let mut result_data = Vec::new();
        
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * dim_size * inner_size + inner;
                let mut all_true = true;
                
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    let val = TensorElement::to_f64(&data[idx]).unwrap_or(0.0);
                    if val == 0.0 {
                        all_true = false;
                        break;
                    }
                }
                
                result_data.push(all_true);
            }
        }
        
        let mut result_shape = shape.to_vec();
        if keepdim {
            result_shape[target_dim] = 1;
        } else {
            result_shape.remove(target_dim);
        }
        
        Tensor::from_data(
            result_data,
            result_shape,
            self.device
        )
    }

    /// Check if any element is true (for boolean tensors)
    pub fn any(&self) -> Result<Tensor<bool>> {
        // Convert to boolean and check any values
        let data = self.data()?;
        let any_true = data.iter().any(|&x| {
            // Convert to f64 for comparison
            let val = TensorElement::to_f64(&x).unwrap_or(0.0);
            val != 0.0
        });
        crate::creation::tensor_scalar(any_true)
    }

    /// Check if any element is true along dimension
    pub fn any_dim(&self, dim: i32, keepdim: bool) -> Result<Tensor<bool>> {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        
        if shape.is_empty() {
            return Err(TorshError::InvalidOperation("Cannot reduce empty tensor".to_string()));
        }
        
        let target_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };
        
        if target_dim >= shape.len() {
            return Err(TorshError::DimensionError(format!(
                "Dimension {dim} out of range for tensor with {dims} dimensions", 
                dims = shape.len()
            )));
        }
        
        let dim_size = shape[target_dim];
        let outer_size: usize = shape[..target_dim].iter().product();
        let inner_size: usize = shape[target_dim + 1..].iter().product();
        
        let mut result_data = Vec::new();
        
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * dim_size * inner_size + inner;
                let mut any_true = false;
                
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    let val = TensorElement::to_f64(&data[idx]).unwrap_or(0.0);
                    if val != 0.0 {
                        any_true = true;
                        break;
                    }
                }
                
                result_data.push(any_true);
            }
        }
        
        let mut result_shape = shape.to_vec();
        if keepdim {
            result_shape[target_dim] = 1;
        } else {
            result_shape.remove(target_dim);
        }
        
        Tensor::from_data(
            result_data,
            result_shape,
            self.device
        )
    }

    /// Argmax
    pub fn argmax(&self, _dim: Option<i32>) -> Result<Tensor<i64>> {
        // TODO: Implement using scirs2
        crate::creation::zeros(&[1])
    }

    /// Argmin
    pub fn argmin(&self, _dim: Option<i32>) -> Result<Tensor<i64>> {
        // TODO: Implement using scirs2
        crate::creation::zeros(&[1])
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

        // For multi-dimensional tensors, sort along the specified dimension
        let strides = self.compute_strides();
        let dim_size = shape[dim];
        let total_size = shape.iter().product::<usize>();

        let mut sorted_data = vec![T::default(); total_size];
        let mut indices_data = vec![0i64; total_size];

        // Calculate number of slices to sort
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        for outer_idx in 0..outer_size {
            for inner_idx in 0..inner_size {
                // Collect values along the dimension
                let mut dim_values: Vec<(usize, T)> = Vec::with_capacity(dim_size);

                for dim_idx in 0..dim_size {
                    let mut indices = vec![0; self.ndim()];
                    let mut temp = outer_idx;

                    // Compute indices for dimensions before dim
                    for i in (0..dim).rev() {
                        indices[i] = temp % shape[i];
                        temp /= shape[i];
                    }

                    indices[dim] = dim_idx;

                    // Compute indices for dimensions after dim
                    temp = inner_idx;
                    for i in (dim + 1..self.ndim()).rev() {
                        indices[i] = temp % shape[i];
                        temp /= shape[i];
                    }

                    let flat_idx = indices
                        .iter()
                        .zip(&strides)
                        .map(|(idx, stride)| idx * stride)
                        .sum::<usize>();

                    dim_values.push((dim_idx, data[flat_idx]));
                }

                // Sort the values
                dim_values.sort_by(|a, b| {
                    if descending {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    }
                });

                // Place sorted values back
                for (new_dim_idx, (orig_dim_idx, value)) in dim_values.iter().enumerate() {
                    let mut indices = vec![0; self.ndim()];
                    let mut temp = outer_idx;

                    for i in (0..dim).rev() {
                        indices[i] = temp % shape[i];
                        temp /= shape[i];
                    }

                    indices[dim] = new_dim_idx;

                    temp = inner_idx;
                    for i in (dim + 1..self.ndim()).rev() {
                        indices[i] = temp % shape[i];
                        temp /= shape[i];
                    }

                    let flat_idx = indices
                        .iter()
                        .zip(&strides)
                        .map(|(idx, stride)| idx * stride)
                        .sum::<usize>();

                    sorted_data[flat_idx] = *value;
                    indices_data[flat_idx] = *orig_dim_idx as i64;
                }
            }
        }

        let sorted_tensor = Self::from_data(sorted_data, shape.to_vec(), self.device);
        let indices_tensor = Tensor::<i64>::from_data(indices_data, shape.to_vec(), self.device);

        Ok((sorted_tensor?, indices_tensor?))
    }

    /// Returns the indices that would sort the tensor along a given dimension
    pub fn argsort(&self, dim: Option<i32>, descending: bool) -> Result<Tensor<i64>> {
        let (_, indices) = self.sort(dim, descending)?;
        Ok(indices)
    }

    /// Returns the k largest/smallest elements along a given dimension
    pub fn topk(
        &self,
        k: usize,
        dim: Option<i32>,
        largest: bool,
        sorted: bool,
    ) -> Result<(Self, Tensor<i64>)> {
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
        let dim_size = shape[dim];

        if k > dim_size {
            return Err(TorshError::InvalidArgument(format!(
                "k ({k}) is greater than dimension size ({dim_size})"
            )));
        }

        if k == 0 {
            return Err(TorshError::InvalidArgument(
                "k must be greater than 0".to_string(),
            ));
        }

        // First sort the entire tensor
        // If we want largest, sort descending; if we want smallest, sort ascending
        let (sorted_tensor, sorted_indices) = self.sort(Some(dim as i32), largest)?;

        // Then slice to get top k
        let mut result_shape = shape.to_vec();
        result_shape[dim] = k;

        let sorted_data = sorted_tensor.data()?;
        let indices_data = sorted_indices.data()?;

        let strides = self.compute_strides();
        let total_size: usize = result_shape.iter().product();

        let mut topk_data = vec![T::default(); total_size];
        let mut topk_indices = vec![0i64; total_size];

        // Copy the top k elements
        let outer_size: usize = shape[..dim].iter().product();
        let inner_size: usize = shape[dim + 1..].iter().product();

        for outer_idx in 0..outer_size {
            for k_idx in 0..k {
                for inner_idx in 0..inner_size {
                    // Source indices
                    let mut src_indices = vec![0; self.ndim()];
                    let mut temp = outer_idx;

                    for i in (0..dim).rev() {
                        src_indices[i] = temp % shape[i];
                        temp /= shape[i];
                    }

                    src_indices[dim] = k_idx;

                    temp = inner_idx;
                    for i in (dim + 1..self.ndim()).rev() {
                        src_indices[i] = temp % shape[i];
                        temp /= shape[i];
                    }

                    let src_flat_idx = src_indices
                        .iter()
                        .zip(&strides)
                        .map(|(idx, stride)| idx * stride)
                        .sum::<usize>();

                    // Destination indices
                    let result_strides = Tensor::<T>::compute_strides_for_shape(&result_shape);
                    let dst_flat_idx = src_indices
                        .iter()
                        .zip(&result_strides)
                        .map(|(idx, stride)| idx * stride)
                        .sum::<usize>();

                    topk_data[dst_flat_idx] = sorted_data[src_flat_idx];
                    topk_indices[dst_flat_idx] = indices_data[src_flat_idx];
                }
            }
        }

        let topk_tensor = Self::from_data(topk_data, result_shape.clone(), self.device)?;
        let indices_tensor = Tensor::<i64>::from_data(topk_indices, result_shape, self.device)?;

        // If not sorted, reverse to get original order
        if !sorted && !largest {
            // TODO: Implement reverse along dimension
        }

        Ok((topk_tensor, indices_tensor))
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

        let self_data = self.data()?;
        let other_data = other.data()?;

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

        Self::from_data(result_data, vec![m, n], self.device)
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

            let self_data = self.data()?;
            let other_data = other.data()?;

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

            return Self::from_data(
                result_data,
                vec![batch_size, m, n],
                self.device,
            );
        }

        // Handle 4D batched matrix multiplication (for attention mechanisms)
        if self_shape.ndim() == 4 && other_shape.ndim() == 4 {
            let batch_size = self_shape.dims()[0];
            let num_heads = self_shape.dims()[1];
            let seq_len = self_shape.dims()[2];
            let head_dim = self_shape.dims()[3];
            let other_seq_len = other_shape.dims()[3];

            // Validate batch dimensions
            if batch_size != other_shape.dims()[0] || num_heads != other_shape.dims()[1] {
                return Err(TorshError::ShapeMismatch {
                    expected: vec![batch_size, num_heads],
                    got: vec![other_shape.dims()[0], other_shape.dims()[1]],
                });
            }

            let self_data = self.data()?;
            let other_data = other.data()?;

            let mut result_data = vec![T::default(); batch_size * num_heads * seq_len * other_seq_len];

            for b in 0..batch_size {
                for h in 0..num_heads {
                    for i in 0..seq_len {
                        for j in 0..other_seq_len {
                            let mut sum = T::default();
                            for k in 0..head_dim {
                                let self_idx = b * (num_heads * seq_len * head_dim) + 
                                               h * (seq_len * head_dim) + 
                                               i * head_dim + k;
                                let other_idx = b * (num_heads * head_dim * other_seq_len) + 
                                                h * (head_dim * other_seq_len) + 
                                                k * other_seq_len + j;
                                let a_val = self_data[self_idx];
                                let b_val = other_data[other_idx];
                                sum = sum + (a_val * b_val);
                            }
                            let result_idx = b * (num_heads * seq_len * other_seq_len) + 
                                           h * (seq_len * other_seq_len) + 
                                           i * other_seq_len + j;
                            result_data[result_idx] = sum;
                        }
                    }
                }
            }

            return Self::from_data(
                result_data,
                vec![batch_size, num_heads, seq_len, other_seq_len],
                self.device,
            );
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
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.sqrt()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Exponential function
    pub fn exp(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.exp()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Error function (approximation for f32/f64)
    pub fn erf(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                // Using approximation for erf function
                let x_f64 = TensorElement::to_f64(&x).unwrap_or(0.0);
                let erf_result = erf_approx(x_f64);
                T::from_f64(erf_result).unwrap_or_else(|| <T as TensorElement>::zero())
            })
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Sine function
    pub fn sin(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.sin()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Cosine function
    pub fn cos(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.cos()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Tangent function
    pub fn tan(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.tan()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Arc sine function
    pub fn asin(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.asin()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Arc cosine function
    pub fn acos(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.acos()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Arc tangent function
    pub fn atan(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.atan()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Two-argument arc tangent function
    pub fn atan2(&self, other: &Self) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let other_data = other.data()?;
        let result_data: Vec<T> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&y, &x)| y.atan2(x))
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Hyperbolic sine function
    pub fn sinh(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.sinh()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Hyperbolic cosine function
    pub fn cosh(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.cosh()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
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
        let data = self.data()?;
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
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let one = <T as TensorElement>::one();
                one / (one + (-x).exp())
            })
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Tanh activation
    pub fn tanh(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.tanh()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Leaky ReLU activation
    pub fn leaky_relu(&self, negative_slope: f64) -> Result<Self> {
        let data = self.data()?;
        let slope = T::from_f64(negative_slope).unwrap_or_else(|| <T as TensorElement>::zero());
        let zero = <T as TensorElement>::zero();
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| if x > zero { x } else { x * slope })
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// GELU (Gaussian Error Linear Unit) activation
    pub fn gelu(&self) -> Result<Self> {
        let data = self.data()?;
        // GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
        // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let half = T::from_f64(0.5).unwrap_or_else(|| <T as TensorElement>::zero());
        let one = <T as TensorElement>::one();
        let c1 = T::from_f64(0.7978845608).unwrap_or_else(|| <T as TensorElement>::zero()); // sqrt(2/π)
        let c2 = T::from_f64(0.044715).unwrap_or_else(|| <T as TensorElement>::zero());

        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let x3 = x * x * x;
                let inner = c1 * (x + c2 * x3);
                half * x * (one + inner.tanh())
            })
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Natural logarithm
    pub fn log(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.ln()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Natural logarithm (alias for log for PyTorch compatibility)
    pub fn ln(&self) -> Result<Self> {
        self.log()
    }

    /// Base 2 logarithm
    pub fn log2(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.log2()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Base 10 logarithm
    pub fn log10(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.log10()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Floor function
    pub fn floor(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.floor()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Ceiling function
    pub fn ceil(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.ceil()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Round function
    pub fn round(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.round()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Truncate function
    pub fn trunc(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.trunc()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Fractional part
    pub fn frac(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.fract()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Square function
    pub fn square(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x * x).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Reciprocal square root
    pub fn rsqrt(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| {
            let one = T::from_f64(1.0).unwrap();
            one / x.sqrt()
        }).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Reciprocal function
    pub fn reciprocal(&self) -> Result<Self> {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| {
            let one = T::from_f64(1.0).unwrap();
            one / x
        }).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Softmax along dimension (numerically stable)
    #[allow(clippy::needless_range_loop)]
    pub fn softmax(&self, dim: i32) -> Result<Self> {
        // For now, implement along the last dimension if dim == -1, otherwise use specified dim
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        
        if shape.is_empty() {
            return Err(TorshError::InvalidOperation("Cannot compute softmax on empty tensor".to_string()));
        }
        
        // Handle negative dimension
        let actual_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };
        
        if actual_dim >= shape.len() {
            return Err(TorshError::InvalidArgument(format!("Dimension {} out of range for tensor with {} dimensions", dim, shape.len())));
        }
        
        let dim_size = shape[actual_dim];
        let outer_size: usize = shape[..actual_dim].iter().product();
        let inner_size: usize = shape[actual_dim + 1..].iter().product();
        
        let mut result_data = vec![T::from_f64(0.0).unwrap(); data.len()];
        
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * dim_size * inner_size + inner;
                
                // Find max for numerical stability
                let mut max_val = data[base_idx];
                for d in 1..dim_size {
                    let idx = base_idx + d * inner_size;
                    if data[idx] > max_val {
                        max_val = data[idx];
                    }
                }
                
                // Compute exp(x - max) and sum
                let mut exp_sum = T::from_f64(0.0).unwrap();
                let mut exp_values = vec![T::from_f64(0.0).unwrap(); dim_size];
                
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    let exp_val = (data[idx] - max_val).exp();
                    exp_values[d] = exp_val;
                    exp_sum = exp_sum + exp_val;
                }
                
                // Compute softmax values
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    result_data[idx] = exp_values[d] / exp_sum;
                }
            }
        }
        
        Self::from_data(
            result_data,
            shape.to_vec(),
            self.device,
        )
    }

    /// Log softmax along dimension (numerically stable)
    pub fn log_softmax(&self, dim: i32) -> Result<Self> {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();
        
        if shape.is_empty() {
            return Err(TorshError::InvalidOperation("Cannot compute log_softmax on empty tensor".to_string()));
        }
        
        // Handle negative dimension
        let actual_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };
        
        if actual_dim >= shape.len() {
            return Err(TorshError::InvalidArgument(format!("Dimension {} out of range for tensor with {} dimensions", dim, shape.len())));
        }
        
        let dim_size = shape[actual_dim];
        let outer_size: usize = shape[..actual_dim].iter().product();
        let inner_size: usize = shape[actual_dim + 1..].iter().product();
        
        let mut result_data = vec![T::from_f64(0.0).unwrap(); data.len()];
        
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * dim_size * inner_size + inner;
                
                // Find max for numerical stability
                let mut max_val = data[base_idx];
                for d in 1..dim_size {
                    let idx = base_idx + d * inner_size;
                    if data[idx] > max_val {
                        max_val = data[idx];
                    }
                }
                
                // Compute log(sum(exp(x - max)))
                let mut exp_sum = T::from_f64(0.0).unwrap();
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    exp_sum = exp_sum + (data[idx] - max_val).exp();
                }
                let log_sum_exp = exp_sum.ln();
                
                // Compute log_softmax values: x - max - log(sum_exp)
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    result_data[idx] = data[idx] - max_val - log_sum_exp;
                }
            }
        }
        
        Self::from_data(
            result_data,
            shape.to_vec(),
            self.device,
        )
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

/// Helper methods for all tensor types
impl<T: TensorElement> Tensor<T> {
    /// Convert flat index to multi-dimensional indices (public version)
    pub(crate) fn flat_to_multi_index_ops(&self, flat_idx: usize, dims: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; dims.len()];
        let mut remaining = flat_idx;

        // Calculate indices in row-major order
        for i in 0..dims.len() {
            let stride = dims[i + 1..].iter().product::<usize>().max(1);
            indices[i] = remaining / stride;
            remaining %= stride;
        }

        indices
    }

    /// Get the actual index in the tensor data for broadcasting
    pub(crate) fn broadcast_index(
        &self,
        broadcast_indices: &[usize],
        broadcast_dims: &[usize],
    ) -> usize {
        let self_shape = self.shape();
        let self_dims = self_shape.dims();
        let self_ndim = self_dims.len();
        let broadcast_ndim = broadcast_dims.len();

        let mut actual_indices = vec![0; self_ndim];

        // Map from broadcast indices to actual indices (right-aligned)
        // Broadcasting aligns dimensions from the right
        let offset = broadcast_ndim.saturating_sub(self_ndim);

        for i in 0..self_ndim {
            let broadcast_dim_idx = offset + i;
            if broadcast_dim_idx < broadcast_ndim {
                let broadcast_idx = broadcast_indices[broadcast_dim_idx];

                // If dimension is 1, index is always 0 (broadcasting)
                // Otherwise, use the broadcast index
                actual_indices[i] = if self_dims[i] == 1 { 0 } else { broadcast_idx };
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
}

/// Comparison operations
impl<T: TensorElement + PartialOrd> Tensor<T> {
    /// Element-wise equality
    pub fn eq(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a == b)
    }

    /// Element-wise inequality
    pub fn ne(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a != b)
    }

    /// Element-wise greater than
    pub fn gt(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a > b)
    }

    /// Element-wise less than
    pub fn lt(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a < b)
    }

    /// Element-wise greater than or equal
    pub fn ge(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a >= b)
    }

    /// Element-wise less than or equal
    pub fn le(&self, other: &Self) -> Result<Tensor<bool>> {
        self.comparison_op(other, |a, b| a <= b)
    }

    /// Generic comparison operation with broadcasting
    fn comparison_op<F>(&self, other: &Self, op: F) -> Result<Tensor<bool>>
    where
        F: Fn(&T, &T) -> bool,
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
            let self_data = self.data()?;
            let other_data = other.data()?;

            let result_data: Vec<bool> = self_data
                .iter()
                .zip(other_data.iter())
                .map(|(a, b)| op(a, b))
                .collect();

            return Tensor::from_data(
                result_data,
                self.shape().dims().to_vec(),
                self.device,
            );
        }

        // Compute broadcasted shape
        let broadcast_shape = self.shape().broadcast_shape(&other.shape())?;
        let broadcast_dims = broadcast_shape.dims();
        let broadcast_size = broadcast_shape.numel();

        let self_data = self.data()?;
        let other_data = other.data()?;

        let mut result_data = Vec::with_capacity(broadcast_size);

        // Compute broadcasting for each element
        for flat_idx in 0..broadcast_size {
            let broadcast_indices = self.flat_to_multi_index_ops(flat_idx, broadcast_dims);

            let self_idx = self.broadcast_index(&broadcast_indices, broadcast_dims);
            let other_idx = other.broadcast_index(&broadcast_indices, broadcast_dims);

            let self_val = &self_data[self_idx];
            let other_val = &other_data[other_idx];

            result_data.push(op(self_val, other_val));
        }

        Tensor::from_data(
            result_data,
            broadcast_dims.to_vec(),
            self.device,
        )
    }

    /// Element-wise equality with scalar
    pub fn eq_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a == &b)
    }

    /// Element-wise inequality with scalar
    pub fn ne_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a != &b)
    }

    /// Element-wise greater than scalar
    pub fn gt_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a > &b)
    }

    /// Element-wise less than scalar
    pub fn lt_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a < &b)
    }

    /// Element-wise greater than or equal to scalar
    pub fn ge_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a >= &b)
    }

    /// Element-wise less than or equal to scalar
    pub fn le_scalar(&self, scalar: T) -> Result<Tensor<bool>> {
        self.comparison_scalar_op(scalar, |a, b| a <= &b)
    }

    /// Generic comparison operation with scalar
    fn comparison_scalar_op<F>(&self, scalar: T, op: F) -> Result<Tensor<bool>>
    where
        F: Fn(&T, T) -> bool,
    {
        let self_data = self.data()?;

        let result_data: Vec<bool> = self_data.iter().map(|a| op(a, scalar)).collect();

        Tensor::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }
}

/// Logical operations
impl Tensor<bool> {
    /// Element-wise logical AND
    pub fn logical_and(&self, other: &Self) -> Result<Self> {
        self.logical_op(other, |a, b| a && b)
    }

    /// Element-wise logical OR
    pub fn logical_or(&self, other: &Self) -> Result<Self> {
        self.logical_op(other, |a, b| a || b)
    }

    /// Element-wise logical XOR
    pub fn logical_xor(&self, other: &Self) -> Result<Self> {
        self.logical_op(other, |a, b| a ^ b)
    }

    /// Element-wise logical NOT
    pub fn logical_not(&self) -> Result<Self> {
        let data = self.data()?;

        let result_data: Vec<bool> = data.iter().map(|&x| !x).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Element-wise logical AND with scalar
    pub fn logical_and_scalar(&self, scalar: bool) -> Result<Self> {
        if !scalar {
            // AND with false always gives false
            Self::from_data(
                vec![false; self.shape().numel()],
                self.shape().dims().to_vec(),
                self.device,
            )
        } else {
            // AND with true preserves the original values
            Ok(self.clone())
        }
    }

    /// Element-wise logical OR with scalar
    pub fn logical_or_scalar(&self, scalar: bool) -> Result<Self> {
        if scalar {
            // OR with true always gives true
            Self::from_data(
                vec![true; self.shape().numel()],
                self.shape().dims().to_vec(),
                self.device,
            )
        } else {
            // OR with false preserves the original values
            Ok(self.clone())
        }
    }

    /// Generic logical operation with broadcasting
    fn logical_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(bool, bool) -> bool,
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
            let self_data = self.data()?;
            let other_data = other.data()?;

            let result_data: Vec<bool> = self_data
                .iter()
                .zip(other_data.iter())
                .map(|(&a, &b)| op(a, b))
                .collect();

            return Self::from_data(
                result_data,
                self.shape().dims().to_vec(),
                self.device,
            );
        }

        // Compute broadcasted shape
        let broadcast_shape = self.shape().broadcast_shape(&other.shape())?;
        let broadcast_dims = broadcast_shape.dims();
        let broadcast_size = broadcast_shape.numel();

        let self_data = self.data()?;
        let other_data = other.data()?;

        let mut result_data = Vec::with_capacity(broadcast_size);

        // Compute broadcasting for each element
        for flat_idx in 0..broadcast_size {
            let broadcast_indices = self.flat_to_multi_index_ops(flat_idx, broadcast_dims);

            let self_idx = self.broadcast_index(&broadcast_indices, broadcast_dims);
            let other_idx = other.broadcast_index(&broadcast_indices, broadcast_dims);

            let self_val = self_data[self_idx];
            let other_val = other_data[other_idx];

            result_data.push(op(self_val, other_val));
        }

        Self::from_data(
            result_data,
            broadcast_dims.to_vec(),
            self.device,
        )
    }
}

/// Bitwise operations for integer types
impl<T> Tensor<T> 
where 
    T: TensorElement 
        + Copy 
        + std::ops::BitAnd<Output = T> 
        + std::ops::BitOr<Output = T> 
        + std::ops::BitXor<Output = T> 
        + std::ops::Not<Output = T>,
{
    /// Element-wise bitwise AND with broadcasting
    pub fn bitwise_and(&self, other: &Self) -> Result<Self> {
        self.bitwise_op(other, |a, b| a & b)
    }

    /// Element-wise bitwise OR with broadcasting
    pub fn bitwise_or(&self, other: &Self) -> Result<Self> {
        self.bitwise_op(other, |a, b| a | b)
    }

    /// Element-wise bitwise XOR with broadcasting
    pub fn bitwise_xor(&self, other: &Self) -> Result<Self> {
        self.bitwise_op(other, |a, b| a ^ b)
    }

    /// Element-wise bitwise NOT
    pub fn bitwise_not(&self) -> Result<Self> {
        let data = self.data()?;

        let result_data: Vec<T> = data.iter().map(|&x| !x).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Element-wise bitwise left shift
    pub fn bitwise_left_shift(&self, shift: u32) -> Result<Self> 
    where 
        T: std::ops::Shl<u32, Output = T>
    {
        let data = self.data()?;

        let result_data: Vec<T> = data.iter().map(|&x| x << shift).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Element-wise bitwise right shift
    pub fn bitwise_right_shift(&self, shift: u32) -> Result<Self> 
    where 
        T: std::ops::Shr<u32, Output = T>
    {
        let data = self.data()?;

        let result_data: Vec<T> = data.iter().map(|&x| x >> shift).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Generic bitwise operation with broadcasting
    fn bitwise_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(T, T) -> T,
    {
        use crate::broadcast::BroadcastOps;

        // Check if tensors are broadcast compatible
        if !self.shape().broadcast_compatible(&other.shape()) {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        // If shapes are identical, use optimized path
        if self.shape() == other.shape() {
            let self_data = self.data()?;
            let other_data = other.data()?;

            let result_data: Vec<T> = self_data
                .iter()
                .zip(other_data.iter())
                .map(|(&a, &b)| op(a, b))
                .collect();

            return Self::from_data(
                result_data,
                self.shape().dims().to_vec(),
                self.device,
            );
        }

        // Use full broadcasting
        let broadcast_shape = self.shape().broadcast_shape(&other.shape())?;
        let broadcast_dims = broadcast_shape.dims();
        let broadcast_size = broadcast_shape.numel();

        let self_data = self.data()?;
        let other_data = other.data()?;

        let mut result_data = Vec::with_capacity(broadcast_size);

        // Compute broadcasting for each element
        for flat_idx in 0..broadcast_size {
            let broadcast_indices = BroadcastOps::flat_to_multi_index(flat_idx, broadcast_dims);

            let self_idx = BroadcastOps::compute_broadcast_index(
                &broadcast_indices,
                self.shape().dims(),
                broadcast_dims,
            )?;
            let other_idx = BroadcastOps::compute_broadcast_index(
                &broadcast_indices,
                other.shape().dims(),
                broadcast_dims,
            )?;

            let self_val = self_data[self_idx];
            let other_val = other_data[other_idx];

            result_data.push(op(self_val, other_val));
        }

        Self::from_data(
            result_data,
            broadcast_dims.to_vec(),
            self.device,
        )
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
        self.add_op(other)
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
        self.mul_op(other)
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

/// Additional tensor operations
impl<T: TensorElement + Copy + Default> Tensor<T> {
    /// Unique elements and their counts
    pub fn unique(&self, return_counts: bool) -> Result<(Self, Option<Tensor<i64>>)> {
        let data = self.data()?;
        let mut unique_values = Vec::new();
        let mut counts = Vec::new();

        // Simple O(n²) approach for now
        for &value in data.iter() {
            if let Some(pos) = unique_values.iter().position(|&x| {
                // Simple equality check using bit-level comparison
                std::ptr::eq(&x as *const T, &value as *const T)
                    || (unsafe {
                        let x_bytes = std::slice::from_raw_parts(
                            &x as *const T as *const u8,
                            std::mem::size_of::<T>(),
                        );
                        let value_bytes = std::slice::from_raw_parts(
                            &value as *const T as *const u8,
                            std::mem::size_of::<T>(),
                        );
                        x_bytes == value_bytes
                    })
            }) {
                if return_counts {
                    counts[pos] += 1;
                }
            } else {
                unique_values.push(value);
                if return_counts {
                    counts.push(1);
                }
            }
        }

        let unique_len = unique_values.len();
        let count_len = counts.len();
        let unique_tensor = Self::from_data(unique_values, vec![unique_len], self.device)?;
        let count_tensor = if return_counts {
            Some(Tensor::from_data(counts, vec![count_len], self.device)?)
        } else {
            None
        };

        Ok((unique_tensor, count_tensor))
    }

    /// Bincount operation
    pub fn bincount(&self, minlength: Option<usize>) -> Result<Tensor<i64>>
    where
        T: Into<i64> + Copy,
    {
        let data = self.data()?;
        let mut max_val = 0i64;
        let values: Vec<i64> = data
            .iter()
            .map(|&x| {
                let val = x.into();
                max_val = max_val.max(val);
                val
            })
            .collect();

        let length = minlength.unwrap_or(0).max((max_val + 1) as usize);
        let mut counts = vec![0i64; length];

        for val in values {
            if val >= 0 && (val as usize) < counts.len() {
                counts[val as usize] += 1;
            }
        }

        Tensor::from_data(counts, vec![length], self.device)
    }
}

/// Random sampling operations
impl Tensor<f32> {
    /// Fill tensor with random values from uniform distribution [0, 1)
    pub fn uniform_(&mut self, low: f32, high: f32) -> Result<()> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random};
        let mut data = self.data()?;
        let mut rng = Random::new();

        for value in data.iter_mut() {
            *value = rng.gen_range(low..high);
        }

        Ok(())
    }

    /// Fill tensor with random values from normal distribution
    pub fn normal_(&mut self, mean: f32, std: f32) -> Result<()> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random with Box-Muller transform
        use scirs2_core::random::{Random};

        let mut rng = Random::new();
        let normal = Normal::new(mean, std).map_err(|_| {
            TorshError::InvalidArgument(format!("std must be positive, got {std}"))
        })?;

        self.data_mut_apply(|item| {
            *item = normal.sample(&mut rng);
        })?;

        Ok(())
    }

    /// Sample from multinomial distribution
    pub fn multinomial(
        weights: &Self,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<i64>> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random};

        let weights_data = weights.data()?;
        let total: f32 = weights_data.iter().sum();

        if total <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "weights sum must be positive".to_string(),
            ));
        }

        // Check if sampling without replacement is possible
        if !replacement {
            let num_non_zero = weights_data.iter().filter(|&&w| w > 0.0).count();
            if num_samples > num_non_zero {
                return Err(TorshError::InvalidArgument(
                    format!("Cannot sample {num_samples} items without replacement from {num_non_zero} non-zero weights")
                ));
            }
        }

        let mut rng = Random::new();
        let mut samples = Vec::with_capacity(num_samples);
        let mut available_weights = weights_data.clone();

        for _ in 0..num_samples {
            let mut cumsum = 0.0;
            let current_total: f32 = available_weights.iter().sum();
            
            // Check if we can continue sampling
            if current_total <= 0.0 {
                return Err(TorshError::InvalidArgument(
                    "No more weights available for sampling".to_string(),
                ));
            }
            
            let random_val: f32 = rng.gen_range(0.0..current_total);

            for (i, &weight) in available_weights.iter().enumerate() {
                if weight > 0.0 {
                    cumsum += weight;
                    if random_val <= cumsum {
                        samples.push(i as i64);
                        if !replacement {
                            available_weights[i] = 0.0; // Remove from consideration
                        }
                        break;
                    }
                }
            }
        }

        Tensor::from_data(
            samples,
            vec![num_samples],
            weights.device,
        )
    }
}

// Complex number operations
impl<T: torsh_core::dtype::ComplexElement> Tensor<T> {
    /// Get the real part of complex tensor
    pub fn real(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: Copy
            + Default
            + std::ops::Add<Output = T::Real>
            + std::ops::Sub<Output = T::Real>
            + std::ops::Mul<Output = T::Real>
            + std::ops::Div<Output = T::Real>,
    {
        let data = self.data()?;
        let real_data: Vec<T::Real> = data.iter().map(|x| x.real()).collect();

        let result = Tensor::from_data(real_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        // Note: real() returns a real tensor, so we can't track complex gradients here
        // This would need special handling in the autograd system for mixed real/complex gradients
        if self.requires_grad {
            // For now, we don't set requires_grad since we'd need a different gradient type
            // This is a limitation that would require extending the type system
        }

        Ok(result)
    }

    /// Get the imaginary part of complex tensor
    pub fn imag(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: Copy
            + Default
            + std::ops::Add<Output = T::Real>
            + std::ops::Sub<Output = T::Real>
            + std::ops::Mul<Output = T::Real>
            + std::ops::Div<Output = T::Real>,
    {
        let data = self.data()?;
        let imag_data: Vec<T::Real> = data.iter().map(|x| x.imag()).collect();

        Tensor::from_data(
            imag_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Get the magnitude (absolute value) of complex tensor
    pub fn magnitude(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: Copy
            + Default
            + std::ops::Add<Output = T::Real>
            + std::ops::Sub<Output = T::Real>
            + std::ops::Mul<Output = T::Real>
            + std::ops::Div<Output = T::Real>,
    {
        let data = self.data()?;
        let mag_data: Vec<T::Real> = data.iter().map(|x| x.abs()).collect();

        Tensor::from_data(
            mag_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Get the phase (argument) of complex tensor
    pub fn angle(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: Copy
            + Default
            + std::ops::Add<Output = T::Real>
            + std::ops::Sub<Output = T::Real>
            + std::ops::Mul<Output = T::Real>
            + std::ops::Div<Output = T::Real>,
    {
        let data = self.data()?;
        let angle_data: Vec<T::Real> = data.iter().map(|x| x.arg()).collect();

        Tensor::from_data(
            angle_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    // conj function is already implemented in lib.rs with generic type bounds
}

// Complex mathematical functions for Complex32
impl Tensor<torsh_core::dtype::Complex32> {
    /// Complex exponential function
    pub fn exp_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let exp_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.exp()).collect();

        let mut result = Tensor::from_data(exp_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_exp".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex natural logarithm
    pub fn log_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let log_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.ln()).collect();

        let mut result = Tensor::from_data(log_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_log".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex square root
    pub fn sqrt_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let sqrt_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.sqrt()).collect();

        let mut result = Tensor::from_data(sqrt_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_sqrt".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex power function (self^exponent)
    pub fn pow_complex(&self, exponent: &Self) -> Result<Self> {
        if self.shape() != exponent.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: exponent.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let exp_data = exponent.data()?;

        let pow_data: Vec<torsh_core::dtype::Complex32> = self_data
            .iter()
            .zip(exp_data.iter())
            .map(|(&base, &exp)| base.powc(exp))
            .collect();

        let mut result = Tensor::from_data(pow_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if either input requires grad
        if self.requires_grad || exponent.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_pow".to_string(),
                vec![
                    std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone())),
                    std::sync::Arc::downgrade(&std::sync::Arc::new(exponent.clone())),
                ],
            );
        }

        Ok(result)
    }

    /// Complex power function with scalar exponent
    pub fn pow_complex_scalar(&self, exponent: torsh_core::dtype::Complex32) -> Result<Self> {
        let data = self.data()?;
        let pow_data: Vec<torsh_core::dtype::Complex32> =
            data.iter().map(|&x| x.powc(exponent)).collect();

        let mut result = Tensor::from_data(pow_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_pow_scalar".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex sine function
    pub fn sin_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let sin_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.sin()).collect();

        let mut result = Tensor::from_data(sin_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_sin".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex cosine function
    pub fn cos_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let cos_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.cos()).collect();

        let mut result = Tensor::from_data(cos_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_cos".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex hyperbolic sine function
    pub fn sinh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let sinh_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.sinh()).collect();

        let mut result = Tensor::from_data(sinh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_sinh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex hyperbolic cosine function
    pub fn cosh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let cosh_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.cosh()).collect();

        let mut result = Tensor::from_data(cosh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_cosh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex hyperbolic tangent function
    pub fn tanh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let tanh_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.tanh()).collect();

        let mut result = Tensor::from_data(tanh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_tanh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex arcsine function
    pub fn asin_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let asin_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.asin()).collect();

        let mut result = Tensor::from_data(asin_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_asin".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex arccosine function
    pub fn acos_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let acos_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.acos()).collect();

        let mut result = Tensor::from_data(acos_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_acos".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex arctangent function
    pub fn atan_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let atan_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.atan()).collect();

        let mut result = Tensor::from_data(atan_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_atan".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex inverse hyperbolic sine function
    pub fn asinh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let asinh_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.asinh()).collect();

        let mut result = Tensor::from_data(asinh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_asinh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex inverse hyperbolic cosine function
    pub fn acosh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let acosh_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.acosh()).collect();

        let mut result = Tensor::from_data(acosh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_acosh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex inverse hyperbolic tangent function
    pub fn atanh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let atanh_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.atanh()).collect();

        let mut result = Tensor::from_data(atanh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_atanh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex logarithm base 10 function
    pub fn log10_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let log10_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.log10()).collect();

        let mut result = Tensor::from_data(log10_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_log10".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex logarithm base 2 function
    pub fn log2_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let log2_data: Vec<torsh_core::dtype::Complex32> = data.iter().map(|x| x.log2()).collect();

        let mut result = Tensor::from_data(log2_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_log2".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }
    /// Extract real part of complex tensor
    pub fn real_part(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let real_data: Vec<f32> = data.iter().map(|x| x.re).collect();
        
        let mut result = Tensor::from_data(real_data, self.shape().dims().to_vec(), self.device)?;
        
        if self.requires_grad {
            result.requires_grad = true;
            // Note: Cannot track gradient dependency on complex tensor in real tensor's operation
            // This is a limitation of the current gradient system
            result.operation = crate::Operation::Leaf;
        }
        
        Ok(result)
    }

    /// Extract imaginary part of complex tensor
    pub fn imag_part(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let imag_data: Vec<f32> = data.iter().map(|x| x.im).collect();
        
        let mut result = Tensor::from_data(imag_data, self.shape().dims().to_vec(), self.device)?;
        
        if self.requires_grad {
            result.requires_grad = true;
            // Note: Cannot track gradient dependency on complex tensor in real tensor's operation
            // This is a limitation of the current gradient system
            result.operation = crate::Operation::Leaf;
        }
        
        Ok(result)
    }

    /// Create complex tensor from real and imaginary parts
    pub fn from_real_imag(real: &Tensor<f32>, imag: &Tensor<f32>) -> Result<Self> {
        if real.shape() != imag.shape() {
            return Err(TorshError::InvalidArgument(
                "Real and imaginary parts must have the same shape".to_string()
            ));
        }

        let real_data = real.data()?;
        let imag_data = imag.data()?;
        
        let complex_data: Vec<torsh_core::dtype::Complex32> = real_data.iter()
            .zip(imag_data.iter())
            .map(|(&r, &i)| torsh_core::dtype::Complex32::new(r, i))
            .collect();

        let mut result = Tensor::from_data(complex_data, real.shape().dims().to_vec(), real.device)?;
        
        if real.requires_grad || imag.requires_grad {
            result.requires_grad = true;
            // Note: Cannot track gradient dependency on real tensors in complex tensor's operation
            // This is a limitation of the current gradient system
            result.operation = crate::Operation::Leaf;
        }
        
        Ok(result)
    }

    /// Convert to polar form (magnitude, phase)
    pub fn to_polar(&self) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let data = self.data()?;
        let mut magnitudes = Vec::with_capacity(data.len());
        let mut phases = Vec::with_capacity(data.len());

        for &complex_val in &data {
            magnitudes.push(complex_val.norm());
            phases.push(complex_val.arg());
        }

        let mag_tensor = Tensor::from_data(magnitudes, self.shape().dims().to_vec(), self.device)?;
        let phase_tensor = Tensor::from_data(phases, self.shape().dims().to_vec(), self.device)?;

        Ok((mag_tensor, phase_tensor))
    }

    /// Create complex tensor from polar form (magnitude, phase)
    pub fn from_polar(magnitude: &Tensor<f32>, phase: &Tensor<f32>) -> Result<Self> {
        if magnitude.shape() != phase.shape() {
            return Err(TorshError::InvalidArgument(
                "Magnitude and phase tensors must have the same shape".to_string()
            ));
        }

        let mag_data = magnitude.data()?;
        let phase_data = phase.data()?;
        
        let complex_data: Vec<torsh_core::dtype::Complex32> = mag_data.iter()
            .zip(phase_data.iter())
            .map(|(&mag, &ph)| {
                let (sin_ph, cos_ph) = ph.sin_cos();
                torsh_core::dtype::Complex32::new(mag * cos_ph, mag * sin_ph)
            })
            .collect();

        let mut result = Tensor::from_data(complex_data, magnitude.shape().dims().to_vec(), magnitude.device)?;
        
        if magnitude.requires_grad || phase.requires_grad {
            result.requires_grad = true;
            // Note: Cannot track gradient dependency on real tensors in complex tensor's operation
            // This is a limitation of the current gradient system
            result.operation = crate::Operation::Leaf;
        }
        
        Ok(result)
    }
}

// Complex mathematical functions for Complex64
impl Tensor<torsh_core::dtype::Complex64> {
    /// Complex exponential function
    pub fn exp_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let exp_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.exp()).collect();

        let mut result = Tensor::from_data(exp_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_exp".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex natural logarithm
    pub fn log_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let log_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.ln()).collect();

        let mut result = Tensor::from_data(log_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_log".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex square root
    pub fn sqrt_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let sqrt_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.sqrt()).collect();

        let mut result = Tensor::from_data(sqrt_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_sqrt".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex power function (self^exponent)
    pub fn pow_complex(&self, exponent: &Self) -> Result<Self> {
        if self.shape() != exponent.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: exponent.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let exp_data = exponent.data()?;

        let pow_data: Vec<torsh_core::dtype::Complex64> = self_data
            .iter()
            .zip(exp_data.iter())
            .map(|(&base, &exp)| base.powc(exp))
            .collect();

        let mut result = Tensor::from_data(pow_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if either input requires grad
        if self.requires_grad || exponent.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_pow".to_string(),
                vec![
                    std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone())),
                    std::sync::Arc::downgrade(&std::sync::Arc::new(exponent.clone())),
                ],
            );
        }

        Ok(result)
    }

    /// Complex power function with scalar exponent
    pub fn pow_complex_scalar(&self, exponent: torsh_core::dtype::Complex64) -> Result<Self> {
        let data = self.data()?;
        let pow_data: Vec<torsh_core::dtype::Complex64> =
            data.iter().map(|&x| x.powc(exponent)).collect();

        let mut result = Tensor::from_data(pow_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_pow_scalar".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex sine function
    pub fn sin_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let sin_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.sin()).collect();

        let mut result = Tensor::from_data(sin_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_sin".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex cosine function
    pub fn cos_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let cos_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.cos()).collect();

        let mut result = Tensor::from_data(cos_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_cos".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex hyperbolic sine function
    pub fn sinh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let sinh_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.sinh()).collect();

        let mut result = Tensor::from_data(sinh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_sinh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex hyperbolic cosine function
    pub fn cosh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let cosh_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.cosh()).collect();

        let mut result = Tensor::from_data(cosh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_cosh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex hyperbolic tangent function
    pub fn tanh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let tanh_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.tanh()).collect();

        let mut result = Tensor::from_data(tanh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_tanh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex arcsine function
    pub fn asin_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let asin_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.asin()).collect();

        let mut result = Tensor::from_data(asin_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_asin".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex arccosine function
    pub fn acos_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let acos_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.acos()).collect();

        let mut result = Tensor::from_data(acos_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_acos".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex arctangent function
    pub fn atan_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let atan_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.atan()).collect();

        let mut result = Tensor::from_data(atan_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_atan".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex inverse hyperbolic sine function
    pub fn asinh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let asinh_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.asinh()).collect();

        let mut result = Tensor::from_data(asinh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_asinh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex inverse hyperbolic cosine function
    pub fn acosh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let acosh_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.acosh()).collect();

        let mut result = Tensor::from_data(acosh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_acosh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex inverse hyperbolic tangent function
    pub fn atanh_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let atanh_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.atanh()).collect();

        let mut result = Tensor::from_data(atanh_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_atanh".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex logarithm base 10 function
    pub fn log10_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let log10_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.log10()).collect();

        let mut result = Tensor::from_data(log10_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_log10".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Complex logarithm base 2 function
    pub fn log2_complex(&self) -> Result<Self> {
        let data = self.data()?;
        let log2_data: Vec<torsh_core::dtype::Complex64> = data.iter().map(|x| x.log2()).collect();

        let mut result = Tensor::from_data(log2_data, self.shape().dims().to_vec(), self.device)?;

        // Track gradient if input requires grad
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = crate::Operation::Custom(
                "complex_log2".to_string(),
                vec![std::sync::Arc::downgrade(&std::sync::Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }
}

// Helper functions for creating complex tensors
impl<T: torsh_core::dtype::ComplexElement> Tensor<T> {
    /// Create a complex tensor from real and imaginary parts
    pub fn complex(real: &Tensor<T::Real>, imag: &Tensor<T::Real>) -> Result<Self>
    where
        T::Real: Copy
            + Default
            + std::ops::Add<Output = T::Real>
            + std::ops::Sub<Output = T::Real>
            + std::ops::Mul<Output = T::Real>
            + std::ops::Div<Output = T::Real>,
    {
        if real.shape() != imag.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: real.shape().dims().to_vec(),
                got: imag.shape().dims().to_vec(),
            });
        }

        let real_data = real.data()?;
        let imag_data = imag.data()?;

        let complex_data: Vec<T> = real_data
            .iter()
            .zip(imag_data.iter())
            .map(|(&r, &i)| T::new(r, i))
            .collect();

        Tensor::from_data(
            complex_data,
            real.shape().dims().to_vec(),
            real.device,
        )
    }
}

// View transformations between real and complex tensors
impl<T: torsh_core::dtype::FloatElement> Tensor<T> {
    /// View a real tensor as complex
    ///
    /// The tensor's last dimension must be of size 2, representing real and imaginary parts.
    /// This creates a view where pairs of consecutive elements are interpreted as complex numbers.
    ///
    /// # Arguments
    /// * Returns a complex tensor with one fewer dimension (last dim of size 2 becomes complex numbers)
    pub fn view_as_complex<C>(&self) -> Result<Tensor<C>>
    where
        C: torsh_core::dtype::ComplexElement<Real = T> + torsh_core::dtype::TensorElement,
        T: Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        let shape = self.shape();
        let dims = shape.dims();

        // Check that last dimension is of size 2
        if dims.is_empty() || dims[dims.len() - 1] != 2 {
            return Err(TorshError::InvalidArgument(
                "Last dimension must be of size 2 for view_as_complex".to_string(),
            ));
        }

        // New shape: remove the last dimension of size 2
        let new_dims = &dims[..dims.len() - 1];
        let new_numel = new_dims.iter().product::<usize>();

        // Convert real pairs to complex numbers
        let data = self.data()?;
        let mut complex_data = Vec::with_capacity(new_numel);

        for i in 0..new_numel {
            let real_idx = i * 2;
            let imag_idx = i * 2 + 1;
            let real_part = data[real_idx];
            let imag_part = data[imag_idx];
            complex_data.push(C::new(real_part, imag_part));
        }

        let mut result = Tensor::from_data(complex_data, new_dims.to_vec(), self.device)?;

        // Preserve gradient tracking
        result.requires_grad = self.requires_grad;

        Ok(result)
    }
}

impl<T: torsh_core::dtype::ComplexElement> Tensor<T> {
    /// View a complex tensor as real
    ///
    /// This adds a new last dimension of size 2, where the real and imaginary parts
    /// are separated into consecutive elements.
    ///
    /// # Arguments  
    /// * Returns a real tensor with one additional dimension (size 2 for real/imag parts)
    pub fn view_as_real(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: Copy
            + Default
            + std::ops::Add<Output = T::Real>
            + std::ops::Sub<Output = T::Real>
            + std::ops::Mul<Output = T::Real>
            + std::ops::Div<Output = T::Real>,
    {
        let shape = self.shape();
        let dims = shape.dims();

        // New shape: add dimension of size 2 at the end
        let mut new_dims = dims.to_vec();
        new_dims.push(2);
        let new_numel = new_dims.iter().product::<usize>();

        // Convert complex numbers to real pairs
        let data = self.data()?;
        let mut real_data = Vec::with_capacity(new_numel);

        for complex_val in data.iter() {
            real_data.push(complex_val.real());
            real_data.push(complex_val.imag());
        }

        let mut result = Tensor::from_data(real_data, new_dims, self.device)?;

        // Preserve gradient tracking
        result.requires_grad = self.requires_grad;

        Ok(result)
    }
}

// Additional tensor operations - only implement missing methods
impl<T: TensorElement + Copy + Default> Tensor<T> {
    /// Masked fill operation - fill values where mask is true
    pub fn masked_fill(&self, mask: &Tensor<bool>, value: f32) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: mask.shape().dims().to_vec(),
            });
        }

        let result = self.clone();
        let mask_data = mask.data()?;
        let fill_value = T::from_f64(value as f64).unwrap_or_else(|| T::default());

        // Use iterative approach with index tracking
        let result_data_vec = {
            let result_data = result.data()?;
            result_data.as_slice().to_vec()
        };

        let mut new_data = Vec::with_capacity(result_data_vec.len());
        for (i, &val) in result_data_vec.iter().enumerate() {
            if i < mask_data.len() && mask_data[i] {
                new_data.push(fill_value);
            } else {
                new_data.push(val);
            }
        }

        Self::from_data(
            new_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Masked where operation - replace values based on condition
    pub fn masked_where(&self, mask: &Tensor<bool>, other: &Self) -> Result<Self> {
        if self.shape() != mask.shape() || self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: if self.shape() != mask.shape() {
                    mask.shape().dims().to_vec()
                } else {
                    other.shape().dims().to_vec()
                },
            });
        }

        let self_data = self.data()?;
        let mask_data = mask.data()?;
        let other_data = other.data()?;

        let new_data: Vec<T> = self_data
            .iter()
            .zip(mask_data.iter())
            .zip(other_data.iter())
            .map(|((&self_val, &mask_val), &other_val)| {
                if mask_val {
                    other_val
                } else {
                    self_val
                }
            })
            .collect();

        Self::from_data(
            new_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Masked scatter operation - scatter values where mask is true
    pub fn masked_scatter(&self, mask: &Tensor<bool>, source: &Self) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: mask.shape().dims().to_vec(),
            });
        }

        let mask_data = mask.data()?;
        let true_count = mask_data.iter().filter(|&&x| x).count();
        
        if source.numel() < true_count {
            return Err(TorshError::InvalidArgument(
                format!("Source tensor has {} elements but need {} for scatter", 
                        source.numel(), true_count)
            ));
        }

        let self_data = self.data()?;
        let source_data = source.data()?;
        
        let mut new_data = Vec::with_capacity(self_data.len());
        let mut source_idx = 0;
        
        for (i, &self_val) in self_data.iter().enumerate() {
            if i < mask_data.len() && mask_data[i] {
                new_data.push(source_data[source_idx]);
                source_idx += 1;
            } else {
                new_data.push(self_val);
            }
        }

        Self::from_data(
            new_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Masked fill with scalar operation (in-place)
    pub fn masked_fill_(&mut self, mask: &Tensor<bool>, value: f32) -> Result<()> {
        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: mask.shape().dims().to_vec(),
            });
        }

        let mask_data = mask.data()?;
        let fill_value = T::from_f64(value as f64).unwrap_or_else(|| T::default());

        let self_data = self.data()?;
        let self_data_vec = self_data.as_slice().to_vec();
        drop(self_data);

        let mut new_data = Vec::with_capacity(self_data_vec.len());
        for (i, &val) in self_data_vec.iter().enumerate() {
            if i < mask_data.len() && mask_data[i] {
                new_data.push(fill_value);
            } else {
                new_data.push(val);
            }
        }

        // Replace data in the tensor
        *self = Self::from_data(
            new_data,
            self.shape().dims().to_vec(),
            self.device,
        )?;

        Ok(())
    }

    /// Dropout operation
    pub fn dropout(&self, _p: f32) -> Result<Self> {
        // For now, just return a clone - would need proper implementation
        Ok(self.clone())
    }

    /// Create tensor filled with a specific value and same shape
    pub fn full_like(&self, value: T) -> Self {
        Tensor::from_data(
            vec![value; self.numel()],
            self.shape().dims().to_vec(),
            self.device,
        ).unwrap()
    }

    /// Element-wise conditional selection
    pub fn where_tensor(&self, condition: &Tensor<bool>, other: &Self) -> Result<Self> {
        // Check shapes match
        if self.shape() != condition.shape() || self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: condition.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;

        let condition_data = condition.data()?;

        let other_data = other.data()?;

        let mut result_data = Vec::with_capacity(self_data.len());

        for i in 0..self_data.len() {
            result_data.push(if condition_data[i] {
                self_data[i]
            } else {
                other_data[i]
            });
        }

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Get the sign of each element
    pub fn sign(&self) -> Result<Self> {
        // For now, just return a clone - would need proper implementation
        Ok(self.clone())
    }

    /// Compute dot product with another tensor
    pub fn dot(&self, _other: &Self) -> Result<Self> {
        // For now, just return a clone - would need proper implementation
        Ok(self.clone())
    }

    /// Compute maximum along a dimension with keepdim support
    pub fn max_dim(&self, _dim: i32, _keepdim: bool) -> Result<Self> {
        // For now, just return a clone - would need proper implementation
        Ok(self.clone())
    }

    /// Compute minimum along a dimension with keepdim support  
    pub fn min_dim(&self, _dim: i32, _keepdim: bool) -> Result<Self> {
        // For now, just return a clone - would need proper implementation
        Ok(self.clone())
    }

    /// Convert to contiguous tensor (no-op for now)
    pub fn contiguous(&self) -> Result<Self> {
        Ok(self.clone())
    }

    /// Convert boolean tensor to float tensor
    pub fn to_tensor<U: TensorElement>(&self) -> Result<Tensor<U>> {
        Err(TorshError::Other(
            "Type conversion not implemented".to_string(),
        ))
    }
}

// Specialized operations for floating point tensors
impl Tensor<f32> {

    /// Singular Value Decomposition (SVD)
    /// Returns (U, S, V) where A = U * diag(S) * V^T
    pub fn svd(&self) -> Result<(Self, Self, Self)> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 {
            return Err(TorshError::Other(
                "SVD requires 2D matrix".to_string(),
            ));
        }
        
        let (m, n) = (dims[0], dims[1]);
        let _data = self.data()?;
        
        // For now, implement basic SVD using simple decomposition
        // In a real implementation, this would use LAPACK or similar
        let min_dim = m.min(n);
        
        // Create identity matrices as placeholder
        let u_data = vec![0.0; m * m];
        let s_data = vec![1.0; min_dim]; // Singular values
        let v_data = vec![0.0; n * n];
        
        // Fill U and V with identity matrices (placeholder)
        let mut u_identity = u_data;
        let mut v_identity = v_data;
        
        for i in 0..m {
            if i < m {
                u_identity[i * m + i] = 1.0;
            }
        }
        
        for i in 0..n {
            if i < n {
                v_identity[i * n + i] = 1.0;
            }
        }
        
        let u = Tensor::from_data(u_identity, vec![m, m], self.device)?;
        let s = Tensor::from_data(s_data, vec![min_dim], self.device)?;
        let v = Tensor::from_data(v_identity, vec![n, n], self.device)?;
        
        Ok((u, s, v))
    }
    
    /// Eigenvalue decomposition (for symmetric matrices)
    /// Returns (eigenvalues, eigenvectors)
    pub fn eig(&self) -> Result<(Self, Self)> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Eigenvalue decomposition requires square matrix".to_string(),
            ));
        }
        
        let n = dims[0];
        
        // Placeholder implementation - would use LAPACK in real implementation
        let eigenvalues = vec![1.0; n];
        let eigenvectors = vec![0.0; n * n];
        
        // Create identity matrix as placeholder eigenvectors
        let mut identity = eigenvectors;
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        
        let vals = Tensor::from_data(eigenvalues, vec![n], self.device)?;
        let vecs = Tensor::from_data(identity, vec![n, n], self.device)?;
        
        Ok((vals, vecs))
    }
    
    /// QR decomposition
    /// Returns (Q, R) where A = Q * R
    pub fn qr(&self) -> Result<(Self, Self)> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 {
            return Err(TorshError::Other(
                "QR decomposition requires 2D matrix".to_string(),
            ));
        }
        
        let (m, n) = (dims[0], dims[1]);
        
        // Placeholder implementation - would use LAPACK in real implementation
        let q_data = vec![0.0; m * m];
        let r_data = vec![0.0; m * n];
        
        // Create identity matrix as placeholder Q
        let mut q_identity = q_data;
        for i in 0..m {
            q_identity[i * m + i] = 1.0;
        }
        
        // Create upper triangular matrix as placeholder R
        let mut r_upper = r_data;
        for i in 0..m.min(n) {
            r_upper[i * n + i] = 1.0;
        }
        
        let q = Tensor::from_data(q_identity, vec![m, m], self.device)?;
        let r = Tensor::from_data(r_upper, vec![m, n], self.device)?;
        
        Ok((q, r))
    }
    
    /// Cholesky decomposition
    /// Returns L where A = L * L^T (A must be positive definite)
    pub fn cholesky(&self) -> Result<Self> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Cholesky decomposition requires square matrix".to_string(),
            ));
        }
        
        let n = dims[0];
        let _data = self.data()?;
        
        // Placeholder implementation - would use LAPACK in real implementation
        let mut l_data = vec![0.0; n * n];
        
        // Create lower triangular identity matrix as placeholder
        for i in 0..n {
            l_data[i * n + i] = 1.0;
        }
        
        let l = Tensor::from_data(l_data, vec![n, n], self.device)?;
        
        Ok(l)
    }
    
    /// Matrix inverse (for square matrices)
    pub fn inverse(&self) -> Result<Self> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Matrix inverse requires square matrix".to_string(),
            ));
        }
        
        let n = dims[0];
        
        // Placeholder implementation - would use LAPACK in real implementation
        let mut inv_data = vec![0.0; n * n];
        
        // Create identity matrix as placeholder
        for i in 0..n {
            inv_data[i * n + i] = 1.0;
        }
        
        let inv = Tensor::from_data(inv_data, vec![n, n], self.device)?;
        
        Ok(inv)
    }
    
    /// Matrix determinant
    pub fn det(&self) -> Result<f32> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Determinant requires square matrix".to_string(),
            ));
        }
        
        // Placeholder implementation - would use LAPACK in real implementation
        Ok(1.0)
    }
    
    /// Matrix rank
    pub fn matrix_rank(&self) -> Result<usize> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 {
            return Err(TorshError::Other(
                "Rank calculation requires 2D matrix".to_string(),
            ));
        }
        
        // Placeholder implementation - would use SVD to compute rank
        Ok(dims[0].min(dims[1]))
    }
    
    /// Matrix trace (sum of diagonal elements)
    pub fn trace(&self) -> Result<f32> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Trace requires square matrix".to_string(),
            ));
        }
        
        let n = dims[0];
        let data = self.data()?;
        
        let mut trace = 0.0;
        for i in 0..n {
            trace += data[i * n + i];
        }
        
        Ok(trace)
    }
}

// Specialized operations for f64 floating point tensors
impl Tensor<f64> {

    /// Singular Value Decomposition (SVD)
    /// Returns (U, S, V) where A = U * diag(S) * V^T
    pub fn svd(&self) -> Result<(Self, Self, Self)> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 {
            return Err(TorshError::Other(
                "SVD requires 2D matrix".to_string(),
            ));
        }
        
        let (m, n) = (dims[0], dims[1]);
        let _data = self.data()?;
        
        // For now, implement basic SVD using simple decomposition
        // In a real implementation, this would use LAPACK or similar
        let min_dim = m.min(n);
        
        // Create identity matrices as placeholder
        let u_data = vec![0.0; m * m];
        let s_data = vec![1.0; min_dim]; // Singular values
        let v_data = vec![0.0; n * n];
        
        // Fill U and V with identity matrices (placeholder)
        let mut u_identity = u_data;
        let mut v_identity = v_data;
        
        for i in 0..m {
            if i < m {
                u_identity[i * m + i] = 1.0;
            }
        }
        
        for i in 0..n {
            if i < n {
                v_identity[i * n + i] = 1.0;
            }
        }
        
        let u = Tensor::from_data(u_identity, vec![m, m], self.device)?;
        let s = Tensor::from_data(s_data, vec![min_dim], self.device)?;
        let v = Tensor::from_data(v_identity, vec![n, n], self.device)?;
        
        Ok((u, s, v))
    }
    
    /// Eigenvalue decomposition (for symmetric matrices)
    /// Returns (eigenvalues, eigenvectors)
    pub fn eig(&self) -> Result<(Self, Self)> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Eigenvalue decomposition requires square matrix".to_string(),
            ));
        }
        
        let n = dims[0];
        
        // Placeholder implementation - would use LAPACK in real implementation
        let eigenvalues = vec![1.0; n];
        let eigenvectors = vec![0.0; n * n];
        
        // Create identity matrix as placeholder eigenvectors
        let mut identity = eigenvectors;
        for i in 0..n {
            identity[i * n + i] = 1.0;
        }
        
        let vals = Tensor::from_data(eigenvalues, vec![n], self.device)?;
        let vecs = Tensor::from_data(identity, vec![n, n], self.device)?;
        
        Ok((vals, vecs))
    }
    
    /// QR decomposition
    /// Returns (Q, R) where A = Q * R
    pub fn qr(&self) -> Result<(Self, Self)> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 {
            return Err(TorshError::Other(
                "QR decomposition requires 2D matrix".to_string(),
            ));
        }
        
        let (m, n) = (dims[0], dims[1]);
        
        // Placeholder implementation - would use LAPACK in real implementation
        let q_data = vec![0.0; m * m];
        let r_data = vec![0.0; m * n];
        
        // Create identity matrix as placeholder Q
        let mut q_identity = q_data;
        for i in 0..m {
            q_identity[i * m + i] = 1.0;
        }
        
        // Create upper triangular matrix as placeholder R
        let mut r_upper = r_data;
        for i in 0..m.min(n) {
            r_upper[i * n + i] = 1.0;
        }
        
        let q = Tensor::from_data(q_identity, vec![m, m], self.device)?;
        let r = Tensor::from_data(r_upper, vec![m, n], self.device)?;
        
        Ok((q, r))
    }
    
    /// Cholesky decomposition
    /// Returns L where A = L * L^T (A must be positive definite)
    pub fn cholesky(&self) -> Result<Self> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Cholesky decomposition requires square matrix".to_string(),
            ));
        }
        
        let n = dims[0];
        let _data = self.data()?;
        
        // Placeholder implementation - would use LAPACK in real implementation
        let mut l_data = vec![0.0; n * n];
        
        // Create lower triangular identity matrix as placeholder
        for i in 0..n {
            l_data[i * n + i] = 1.0;
        }
        
        let l = Tensor::from_data(l_data, vec![n, n], self.device)?;
        
        Ok(l)
    }
    
    /// Matrix inverse (for square matrices)
    pub fn inverse(&self) -> Result<Self> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Matrix inverse requires square matrix".to_string(),
            ));
        }
        
        let n = dims[0];
        
        // Placeholder implementation - would use LAPACK in real implementation
        let mut inv_data = vec![0.0; n * n];
        
        // Create identity matrix as placeholder
        for i in 0..n {
            inv_data[i * n + i] = 1.0;
        }
        
        let inv = Tensor::from_data(inv_data, vec![n, n], self.device)?;
        
        Ok(inv)
    }
    
    /// Matrix determinant
    pub fn det(&self) -> Result<f64> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Determinant requires square matrix".to_string(),
            ));
        }
        
        // Placeholder implementation - would use LAPACK in real implementation
        Ok(1.0)
    }
    
    /// Matrix trace (sum of diagonal elements)
    pub fn trace(&self) -> Result<f64> {
        let binding = self.shape();
        let dims = binding.dims();
        if dims.len() != 2 || dims[0] != dims[1] {
            return Err(TorshError::Other(
                "Trace requires square matrix".to_string(),
            ));
        }
        
        let n = dims[0];
        let data = self.data()?;
        
        let mut trace = 0.0;
        for i in 0..n {
            trace += data[i * n + i];
        }
        
        Ok(trace)
    }
}

/// Signal processing operations
impl<T: TensorElement + FloatElement + Copy + Default + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + PartialOrd> Tensor<T> {
    // Note: conv1d implementation is in conv.rs to avoid duplication
    
    // Note: conv2d implementation is in conv.rs to avoid duplication
    
    /// Cross-correlation operation
    /// 
    /// Computes the cross-correlation between two 1D tensors.
    /// 
    /// # Arguments
    /// * `other` - The other tensor to correlate with
    /// * `mode` - The correlation mode ("full", "valid", "same")
    /// 
    /// # Returns
    /// * `Result<Self>` - The cross-correlation result
    #[allow(clippy::needless_range_loop)]
    pub fn correlate1d(&self, other: &Self, mode: &str) -> Result<Self> {
        let input_shape = self.shape();
        let other_shape = other.shape();
        
        // Validate dimensions
        if input_shape.dims().len() != 1 || other_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensors, got {}D and {}D", 
                input_shape.dims().len(), 
                other_shape.dims().len()
            )));
        }
        
        let input_data = self.data()?;
        let other_data = other.data()?;
        
        let input_len = input_shape.dims()[0];
        let other_len = other_shape.dims()[0];
        
        let output_len = match mode {
            "full" => input_len + other_len - 1,
            "valid" => if input_len >= other_len { input_len - other_len + 1 } else { 0 },
            "same" => input_len,
            _ => return Err(TorshError::InvalidArgument(
                "Mode must be 'full', 'valid', or 'same'".to_string()
            )),
        };
        
        if output_len == 0 {
            return Self::from_data(vec![], vec![0], self.device());
        }
        
        let mut output_data = vec![T::default(); output_len];
        
        // Calculate the starting position for each mode
        let start_pos = match mode {
            "full" => 0,
            "valid" => other_len - 1,
            "same" => (other_len - 1) / 2,
            _ => 0,
        };
        
        // Perform cross-correlation
        for i in 0..output_len {
            let mut sum = T::default();
            let actual_i = i + start_pos;
            
            for j in 0..other_len {
                let input_idx = actual_i as isize - j as isize;
                if input_idx >= 0 && (input_idx as usize) < input_len {
                    sum = sum + input_data[input_idx as usize] * other_data[j];
                }
            }
            output_data[i] = sum;
        }
        
        Self::from_data(output_data, vec![output_len], self.device())
    }
    
    /// Simple moving average filter
    /// 
    /// Applies a simple moving average filter to smooth the signal.
    /// 
    /// # Arguments
    /// * `window_size` - The size of the moving average window
    /// 
    /// # Returns
    /// * `Result<Self>` - The filtered tensor
    pub fn moving_average(&self, window_size: usize) -> Result<Self> {
        let input_shape = self.shape();
        
        // Validate dimensions
        if input_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensor, got {}D", 
                input_shape.dims().len()
            )));
        }
        
        if window_size == 0 {
            return Err(TorshError::InvalidArgument(
                "Window size must be greater than 0".to_string()
            ));
        }
        
        let input_data = self.data()?;
        let input_len = input_shape.dims()[0];
        
        if window_size > input_len {
            return Err(TorshError::InvalidArgument(
                format!("Window size ({window_size}) cannot be larger than input length ({input_len})")
            ));
        }
        
        let output_len = input_len - window_size + 1;
        let mut output_data = vec![T::default(); output_len];
        
        // Calculate moving average
        for i in 0..output_len {
            let mut sum = T::default();
            for j in 0..window_size {
                sum = sum + input_data[i + j];
            }
            // Convert window_size to T for division
            let window_size_t = T::from_f64(window_size as f64).unwrap_or(T::default());
            output_data[i] = sum / window_size_t;
        }
        
        Self::from_data(output_data, vec![output_len], self.device())
    }
    
    /// Gaussian filter
    /// 
    /// Applies a Gaussian filter to smooth the signal.
    /// 
    /// # Arguments
    /// * `sigma` - The standard deviation of the Gaussian kernel
    /// * `kernel_size` - The size of the Gaussian kernel (should be odd)
    /// 
    /// # Returns
    /// * `Result<Self>` - The filtered tensor
    pub fn gaussian_filter(&self, sigma: f64, kernel_size: usize) -> Result<Self> {
        let input_shape = self.shape();
        
        // Validate dimensions
        if input_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensor, got {}D", 
                input_shape.dims().len()
            )));
        }
        
        if kernel_size == 0 || kernel_size % 2 == 0 {
            return Err(TorshError::InvalidArgument(
                "Kernel size must be odd and greater than 0".to_string()
            ));
        }
        
        if sigma <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Sigma must be greater than 0".to_string()
            ));
        }
        
        // Create Gaussian kernel
        let half_size = kernel_size / 2;
        let mut kernel_data = vec![T::default(); kernel_size];
        let mut kernel_sum = T::default();
        
        for (i, kernel_val) in kernel_data.iter_mut().enumerate() {
            let x = i as f64 - half_size as f64;
            let gauss_val = (-0.5 * x * x / (sigma * sigma)).exp();
            let kernel_val_computed = T::from_f64(gauss_val).unwrap_or(T::default());
            *kernel_val = kernel_val_computed;
            kernel_sum = kernel_sum + kernel_val_computed;
        }
        
        // Normalize kernel
        for kernel_val in kernel_data.iter_mut() {
            *kernel_val = *kernel_val / kernel_sum;
        }
        
        // Create kernel tensor
        let kernel = Self::from_data(kernel_data, vec![kernel_size], self.device())?;
        
        // Apply convolution with padding
        let padding = half_size;
        self.conv1d(&kernel, None, 1, padding, 1, 1)
    }
    
    /// Median filter
    /// 
    /// Applies a median filter to remove noise from the signal.
    /// 
    /// # Arguments
    /// * `window_size` - The size of the median filter window (should be odd)
    /// 
    /// # Returns
    /// * `Result<Self>` - The filtered tensor
    pub fn median_filter(&self, window_size: usize) -> Result<Self> {
        let input_shape = self.shape();
        
        // Validate dimensions
        if input_shape.dims().len() != 1 {
            return Err(TorshError::InvalidArgument(format!(
                "Expected 1D tensor, got {}D", 
                input_shape.dims().len()
            )));
        }
        
        if window_size == 0 || window_size % 2 == 0 {
            return Err(TorshError::InvalidArgument(
                "Window size must be odd and greater than 0".to_string()
            ));
        }
        
        let input_data = self.data()?;
        let input_len = input_shape.dims()[0];
        
        if window_size > input_len {
            return Err(TorshError::InvalidArgument(
                format!("Window size ({window_size}) cannot be larger than input length ({input_len})")
            ));
        }
        
        let half_window = window_size / 2;
        let mut output_data = vec![T::default(); input_len];
        
        // Apply median filter
        for (i, output_val) in output_data.iter_mut().enumerate() {
            let start = i.saturating_sub(half_window);
            let end = if i + half_window < input_len { i + half_window + 1 } else { input_len };
            
            // Extract window values
            let mut window_values: Vec<T> = input_data[start..end].to_vec();
            
            // Sort to find median
            window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // Get median value
            let median_idx = window_values.len() / 2;
            *output_val = window_values[median_idx];
        }
        
        Self::from_data(output_data, vec![input_len], self.device())
    }
}

/// Quantized 8-bit signed integer with scale and zero-point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QInt8 {
    pub value: i8,
    pub scale: f32,
    pub zero_point: i8,
}

impl QInt8 {
    pub fn new(value: i8, scale: f32, zero_point: i8) -> Self {
        Self { value, scale, zero_point }
    }
    
    pub fn quantize(float_val: f32, scale: f32, zero_point: i8) -> Self {
        let quantized = ((float_val / scale) + zero_point as f32).round() as i8;
        Self::new(quantized.clamp(i8::MIN, i8::MAX), scale, zero_point)
    }
    
    pub fn dequantize(&self) -> f32 {
        let result = (self.value - self.zero_point) as f32 * self.scale;
        // Clamp to finite values to avoid NaN/Infinity
        if result.is_finite() {
            result
        } else if result.is_nan() {
            0.0f32
        } else if result.is_infinite() && result > 0.0 {
            f32::MAX
        } else {
            f32::MIN
        }
    }
}

impl TensorElement for QInt8 {
    fn dtype() -> torsh_core::dtype::DType {
        torsh_core::dtype::DType::QInt8
    }
    
    fn from_f64(v: f64) -> Option<Self> {
        // Default scale and zero_point for conversion
        Some(Self::quantize(v as f32, 1.0, 0))
    }
    
    fn to_f64(&self) -> Option<f64> {
        Some(self.dequantize() as f64)
    }
    
    fn zero() -> Self {
        Self::new(0, 1.0, 0)
    }
    
    fn one() -> Self {
        Self::new(1, 1.0, 0)
    }
}

/// Quantized 8-bit unsigned integer with scale and zero-point  
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QUInt8 {
    pub value: u8,
    pub scale: f32,
    pub zero_point: u8,
}

impl QUInt8 {
    pub fn new(value: u8, scale: f32, zero_point: u8) -> Self {
        Self { value, scale, zero_point }
    }
    
    pub fn quantize(float_val: f32, scale: f32, zero_point: u8) -> Self {
        let quantized = ((float_val / scale) + zero_point as f32).round() as u8;
        Self::new(quantized.clamp(u8::MIN, u8::MAX), scale, zero_point)
    }
    
    pub fn dequantize(&self) -> f32 {
        (self.value as i32 - self.zero_point as i32) as f32 * self.scale
    }
}

impl TensorElement for QUInt8 {
    fn dtype() -> torsh_core::dtype::DType {
        torsh_core::dtype::DType::QUInt8
    }
    
    fn from_f64(v: f64) -> Option<Self> {
        // Default scale and zero_point for conversion
        Some(Self::quantize(v as f32, 1.0, 128))
    }
    
    fn to_f64(&self) -> Option<f64> {
        Some(self.dequantize() as f64)
    }
    
    fn zero() -> Self {
        Self::new(128, 1.0, 128) // 128 is typical zero_point for uint8
    }
    
    fn one() -> Self {
        Self::new(129, 1.0, 128)
    }
}

/// Quantization operations for floating-point tensors
impl<T: FloatElement> Tensor<T> {
    /// Quantize tensor to QInt8 format
    pub fn quantize_qint8(&self, scale: f32, zero_point: i8) -> Result<Tensor<QInt8>> {
        let data = self.data()?;
        let quantized_data: Vec<QInt8> = data.iter()
            .map(|&val| {
                let float_val = ToPrimitive::to_f64(&val).unwrap_or(0.0) as f32;
                QInt8::quantize(float_val, scale, zero_point)
            })
            .collect();
            
        Tensor::from_data(quantized_data, self.shape().dims().to_vec(), self.device())
    }
    
    /// Quantize tensor to QUInt8 format
    pub fn quantize_quint8(&self, scale: f32, zero_point: u8) -> Result<Tensor<QUInt8>> {
        let data = self.data()?;
        let quantized_data: Vec<QUInt8> = data.iter()
            .map(|&val| {
                let float_val = ToPrimitive::to_f64(&val).unwrap_or(0.0) as f32;
                QUInt8::quantize(float_val, scale, zero_point)
            })
            .collect();
            
        Tensor::from_data(quantized_data, self.shape().dims().to_vec(), self.device())
    }
    
    /// Auto-quantize tensor by computing optimal scale and zero_point from data statistics
    pub fn auto_quantize_qint8(&self) -> Result<(Tensor<QInt8>, f32, i8)> {
        let data = self.data()?;
        let float_data: Vec<f32> = data.iter()
            .map(|&val| ToPrimitive::to_f64(&val).unwrap_or(0.0) as f32)
            .collect();
            
        if float_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot quantize empty tensor".to_string()));
        }
        
        let min_val = float_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = float_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Handle extreme values that could cause numerical instability
        let (effective_min, effective_max) = if min_val == f32::NEG_INFINITY || max_val == f32::INFINITY ||
            min_val == f32::MIN || max_val == f32::MAX {
            // For extreme values, use a reasonable range to avoid numerical issues
            let abs_max = float_data.iter()
                .filter(|&&x| x.is_finite())
                .map(|&x| x.abs())
                .fold(0.0f32, |a, b| a.max(b));
            if abs_max == 0.0 {
                (-1.0f32, 1.0f32)
            } else {
                (-abs_max, abs_max)
            }
        } else {
            (min_val, max_val)
        };
        
        // Compute scale and zero_point for symmetric quantization around zero
        let scale = (effective_max - effective_min) / 255.0; // Use full range of i8
        let zero_point = (-128.0f32 - effective_min / scale).round() as i8;
        
        let quantized_tensor = self.quantize_qint8(scale, zero_point)?;
        Ok((quantized_tensor, scale, zero_point))
    }
    
    /// Auto-quantize tensor to QUInt8 format
    pub fn auto_quantize_quint8(&self) -> Result<(Tensor<QUInt8>, f32, u8)> {
        let data = self.data()?;
        let float_data: Vec<f32> = data.iter()
            .map(|&val| ToPrimitive::to_f64(&val).unwrap_or(0.0) as f32)
            .collect();
            
        if float_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot quantize empty tensor".to_string()));
        }
        
        let min_val = float_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = float_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute scale and zero_point for asymmetric quantization
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (0.0f32 - min_val / scale).round() as u8;
        
        let quantized_tensor = self.quantize_quint8(scale, zero_point)?;
        Ok((quantized_tensor, scale, zero_point))
    }
}

/// Dequantization operations for quantized tensors
impl Tensor<QInt8> {
    /// Dequantize QInt8 tensor back to f32
    pub fn dequantize_f32(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let dequantized_data: Vec<f32> = data.iter()
            .map(|qval| qval.dequantize())
            .collect();
            
        Tensor::from_data(dequantized_data, self.shape().dims().to_vec(), self.device())
    }
    
    /// Element-wise addition for quantized tensors (requires same scale and zero_point)
    pub fn add_quantized(&self, other: &Self) -> Result<Self> {
        let self_data = self.data()?;
        let other_data = other.data()?;
        
        if self_data.is_empty() || other_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot add empty tensors".to_string()));
        }
        
        // Check if scales and zero_points are compatible
        let self_scale = self_data[0].scale;
        let self_zero_point = self_data[0].zero_point;
        let other_scale = other_data[0].scale;
        let other_zero_point = other_data[0].zero_point;
        
        if (self_scale - other_scale).abs() > 1e-6 || self_zero_point != other_zero_point {
            return Err(TorshError::InvalidArgument(
                "Quantized tensors must have matching scale and zero_point for addition".to_string()
            ));
        }
        
        // Perform quantized addition
        let result_data: Vec<QInt8> = self_data.iter().zip(other_data.iter())
            .map(|(a, b)| {
                let result_val = (a.value as i32 + b.value as i32 - self_zero_point as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8;
                QInt8::new(result_val, self_scale, self_zero_point)
            })
            .collect();
            
        Self::from_data(result_data, self.shape().dims().to_vec(), self.device())
    }
}

impl Tensor<QUInt8> {
    /// Dequantize QUInt8 tensor back to f32
    pub fn dequantize_f32(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let dequantized_data: Vec<f32> = data.iter()
            .map(|qval| qval.dequantize())
            .collect();
            
        Tensor::from_data(dequantized_data, self.shape().dims().to_vec(), self.device())
    }
    
    /// Element-wise addition for quantized tensors (requires same scale and zero_point)
    pub fn add_quantized(&self, other: &Self) -> Result<Self> {
        let self_data = self.data()?;
        let other_data = other.data()?;
        
        if self_data.is_empty() || other_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot add empty tensors".to_string()));
        }
        
        // Check if scales and zero_points are compatible
        let self_scale = self_data[0].scale;
        let self_zero_point = self_data[0].zero_point;
        let other_scale = other_data[0].scale;
        let other_zero_point = other_data[0].zero_point;
        
        if (self_scale - other_scale).abs() > 1e-6 || self_zero_point != other_zero_point {
            return Err(TorshError::InvalidArgument(
                "Quantized tensors must have matching scale and zero_point for addition".to_string()
            ));
        }
        
        // Perform quantized addition
        let result_data: Vec<QUInt8> = self_data.iter().zip(other_data.iter())
            .map(|(a, b)| {
                let result_val = (a.value as i32 + b.value as i32 - self_zero_point as i32).clamp(u8::MIN as i32, u8::MAX as i32) as u8;
                QUInt8::new(result_val, self_scale, self_zero_point)
            })
            .collect();
            
        Self::from_data(result_data, self.shape().dims().to_vec(), self.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::tensor_2d;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_matrix_multiplication_2d() {
        // Test 2x3 * 3x2 = 2x2
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();
        let b = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
            DeviceType::Cpu,
        ).unwrap();

        let result = a.matmul(&b).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);

        let data = result.data().unwrap();
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
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2], DeviceType::Cpu).unwrap();

        // Test addition
        let add_result = a.add_op(&b).unwrap();
        let add_data = add_result.data().unwrap();
        assert_eq!(add_data[0], 3.0);
        assert_eq!(add_data[1], 4.0);
        assert_eq!(add_data[2], 5.0);
        assert_eq!(add_data[3], 6.0);

        // Test multiplication
        let mul_result = a.mul_op(&b).unwrap();
        let mul_data = mul_result.data().unwrap();
        assert_eq!(mul_data[0], 2.0);
        assert_eq!(mul_data[1], 4.0);
        assert_eq!(mul_data[2], 6.0);
        assert_eq!(mul_data[3], 8.0);
    }

    #[test]
    fn test_comparison_operations() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2], DeviceType::Cpu).unwrap();

        // Test equality
        let eq_result = a.eq(&b).unwrap();
        let eq_data = eq_result.data().unwrap();
        assert_eq!(eq_data.as_slice(), &[false, true, false, false]);

        // Test greater than
        let gt_result = a.gt(&b).unwrap();
        let gt_data = gt_result.data().unwrap();
        assert_eq!(gt_data.as_slice(), &[false, false, true, true]);

        // Test less than
        let lt_result = a.lt(&b).unwrap();
        let lt_data = lt_result.data().unwrap();
        assert_eq!(lt_data.as_slice(), &[true, false, false, false]);

        // Test scalar comparisons
        let eq_scalar_result = a.eq_scalar(2.0).unwrap();
        let eq_scalar_data = eq_scalar_result.data().unwrap();
        assert_eq!(eq_scalar_data.as_slice(), &[false, true, false, false]);

        let gt_scalar_result = a.gt_scalar(2.0).unwrap();
        let gt_scalar_data = gt_scalar_result.data().unwrap();
        assert_eq!(gt_scalar_data.as_slice(), &[false, false, true, true]);
    }

    #[test]
    fn test_broadcast_index() {
        // Test the broadcast_index method
        let a = Tensor::from_data(vec![0.0], vec![1, 3], DeviceType::Cpu).unwrap();

        // For shape [1, 3] broadcasting to [2, 3]
        // Index (0, 0) in broadcast should map to (0, 0) in original
        // Index (0, 1) in broadcast should map to (0, 1) in original
        // Index (0, 2) in broadcast should map to (0, 2) in original
        // Index (1, 0) in broadcast should map to (0, 0) in original (because first dim is 1)
        // Index (1, 1) in broadcast should map to (0, 1) in original
        // Index (1, 2) in broadcast should map to (0, 2) in original

        let broadcast_dims = &[2, 3];
        assert_eq!(a.broadcast_index(&[0, 0], broadcast_dims), 0);
        assert_eq!(a.broadcast_index(&[0, 1], broadcast_dims), 1);
        assert_eq!(a.broadcast_index(&[0, 2], broadcast_dims), 2);
        assert_eq!(a.broadcast_index(&[1, 0], broadcast_dims), 0); // First dim broadcasts
        assert_eq!(a.broadcast_index(&[1, 1], broadcast_dims), 1);
        assert_eq!(a.broadcast_index(&[1, 2], broadcast_dims), 2);
    }

    #[test]
    fn test_broadcast_to() {
        // Test simple broadcasting
        let a = Tensor::from_data(vec![1.0, 2.0], vec![1, 2], DeviceType::Cpu).unwrap();
        let result = a
            .broadcast_to(&torsh_core::shape::Shape::new(vec![2, 2]))
            .unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        let data = result.data().unwrap();
        // Should broadcast to [[1, 2], [1, 2]]
        assert_eq!(data.as_slice(), &[1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_logical_operations() {
        // Create boolean tensors
        let a = Tensor::from_data(vec![true, true, false, false], vec![2, 2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![true, false, true, false], vec![2, 2], DeviceType::Cpu).unwrap();

        // Test logical AND
        let and_result = a.logical_and(&b).unwrap();
        let and_data = and_result.data().unwrap();
        assert_eq!(and_data.as_slice(), &[true, false, false, false]);

        // Test logical OR
        let or_result = a.logical_or(&b).unwrap();
        let or_data = or_result.data().unwrap();
        assert_eq!(or_data.as_slice(), &[true, true, true, false]);

        // Test logical XOR
        let xor_result = a.logical_xor(&b).unwrap();
        let xor_data = xor_result.data().unwrap();
        assert_eq!(xor_data.as_slice(), &[false, true, true, false]);

        // Test logical NOT
        let not_result = a.logical_not().unwrap();
        let not_data = not_result.data().unwrap();
        assert_eq!(not_data.as_slice(), &[false, false, true, true]);
    }

    #[test]
    fn test_logical_broadcasting() {
        // Test broadcasting with logical operations
        let a = Tensor::from_data(vec![true, false, true, false], vec![2, 2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![true, false], vec![1, 2], DeviceType::Cpu).unwrap();

        // b broadcasts to [[true, false], [true, false]]
        let and_result = a.logical_and(&b).unwrap();
        assert_eq!(and_result.shape().dims(), &[2, 2]);
        let and_data = and_result.data().unwrap();
        // [true, false] AND [true, false] = [true, false]
        // [true, false] AND [true, false] = [true, false]
        assert_eq!(and_data.as_slice(), &[true, false, true, false]);
    }

    #[test]
    fn test_comparison_broadcasting() {
        // First test simple broadcasting
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![2.0, 3.0], vec![1, 2], DeviceType::Cpu).unwrap();

        // b should broadcast to [[2.0, 3.0], [2.0, 3.0]]
        // So comparisons should be:
        // a[0,0]=1.0 == b[0,0]=2.0 => false
        // a[0,1]=2.0 == b[0,1]=3.0 => false
        // a[1,0]=3.0 == b[1,0]=2.0 => false
        // a[1,1]=4.0 == b[1,1]=3.0 => false

        let eq_result = a.eq(&b).unwrap();
        assert_eq!(eq_result.shape().dims(), &[2, 2]);
        let eq_data = eq_result.data().unwrap();
        // println!("a data: {:?}", vec![1.0, 2.0, 3.0, 4.0]);
        // println!("b data: {:?}", vec![2.0, 3.0]);
        // println!("eq_data: {:?}", eq_data.as_slice());
        assert_eq!(eq_data.as_slice(), &[false, false, false, false]);

        // Test more complex broadcasting: compare [2, 3] with [1, 3]
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        ).unwrap();
        let b = Tensor::from_data(vec![1.0, 5.0, 3.0], vec![1, 3], DeviceType::Cpu).unwrap();

        // Let's manually compute what we expect
        // a is [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        // b is [[1.0, 5.0, 3.0]] which broadcasts to [[1.0, 5.0, 3.0], [1.0, 5.0, 3.0]]

        // First, let's verify our broadcast_to method works correctly
        let b_broadcast = b
            .broadcast_to(&torsh_core::shape::Shape::new(vec![2, 3]))
            .unwrap();
        let b_broadcast_data = b_broadcast.data().unwrap();
        assert_eq!(b_broadcast_data.as_slice(), &[1.0, 5.0, 3.0, 1.0, 5.0, 3.0]);

        let eq_result = a.eq(&b).unwrap();
        assert_eq!(eq_result.shape().dims(), &[2, 3]);
        let eq_data = eq_result.data().unwrap();
        // First row: [1, 2, 3] == [1, 5, 3] = [true, false, true]
        // Second row: [4, 5, 6] == [1, 5, 3] = [false, true, false]
        assert_eq!(eq_data.as_slice(), &[true, false, true, false, true, false]);
    }

    #[test]
    fn test_trigonometric_functions() {
        use std::f32::consts::PI;

        // Create test tensors with known values
        let angles = Tensor::from_data(
            vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0],
            vec![5],
            DeviceType::Cpu,
        ).unwrap();

        // Test sin
        let sin_result = angles.sin().unwrap();
        let sin_data = sin_result.data().unwrap();
        let expected_sin = [
            0.0,
            0.5,
            (2.0_f32).sqrt() / 2.0,
            (3.0_f32).sqrt() / 2.0,
            1.0,
        ];
        for (i, (&result, &expected)) in sin_data.iter().zip(expected_sin.iter()).enumerate() {
            assert!(
                (result - expected).abs() < 1e-6,
                "sin mismatch at index {}: {} vs {}",
                i,
                result,
                expected
            );
        }

        // Test cos
        let cos_result = angles.cos().unwrap();
        let cos_data = cos_result.data().unwrap();
        let expected_cos = [
            1.0,
            (3.0_f32).sqrt() / 2.0,
            (2.0_f32).sqrt() / 2.0,
            0.5,
            0.0,
        ];
        for (i, (&result, &expected)) in cos_data.iter().zip(expected_cos.iter()).enumerate() {
            assert!(
                (result - expected).abs() < 1e-6,
                "cos mismatch at index {}: {} vs {}",
                i,
                result,
                expected
            );
        }

        // Test tan
        let tan_angles = Tensor::from_data(
            vec![0.0, PI / 6.0, PI / 4.0, PI / 3.0],
            vec![4],
            DeviceType::Cpu,
        ).unwrap();
        let tan_result = tan_angles.tan().unwrap();
        let tan_data = tan_result.data().unwrap();
        let expected_tan = [0.0, 1.0 / (3.0_f32).sqrt(), 1.0, (3.0_f32).sqrt()];
        for (i, (&result, &expected)) in tan_data.iter().zip(expected_tan.iter()).enumerate() {
            assert!(
                (result - expected).abs() < 1e-6,
                "tan mismatch at index {}: {} vs {}",
                i,
                result,
                expected
            );
        }

        // Test inverse trig functions
        let values = Tensor::from_data(vec![-1.0, -0.5, 0.0, 0.5, 1.0], vec![5], DeviceType::Cpu).unwrap();

        // Test asin
        let asin_result = values.asin().unwrap();
        let asin_data = asin_result.data().unwrap();
        assert!((asin_data[0] - (-PI / 2.0)).abs() < 1e-6);
        assert!((asin_data[2] - 0.0_f32).abs() < 1e-6);
        assert!((asin_data[4] - (PI / 2.0)).abs() < 1e-6);

        // Test acos
        let acos_result = values.acos().unwrap();
        let acos_data = acos_result.data().unwrap();
        assert!((acos_data[0] - PI).abs() < 1e-6);
        assert!((acos_data[2] - (PI / 2.0)).abs() < 1e-6);
        assert!((acos_data[4] - 0.0_f32).abs() < 1e-6);

        // Test atan
        let atan_result = values.atan().unwrap();
        let atan_data = atan_result.data().unwrap();
        assert!((atan_data[2] - 0.0_f32).abs() < 1e-6);

        // Test hyperbolic functions
        let x_values = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3], DeviceType::Cpu).unwrap();

        // Test sinh
        let sinh_result = x_values.sinh().unwrap();
        let sinh_data = sinh_result.data().unwrap();
        assert!((sinh_data[0] - 0.0_f32).abs() < 1e-6);
        assert!((sinh_data[1] - ((1.0_f32.exp() - (-1.0_f32).exp()) / 2.0)).abs() < 1e-6);
        assert!((sinh_data[2] - (((-1.0_f32).exp() - 1.0_f32.exp()) / 2.0)).abs() < 1e-6);

        // Test cosh
        let cosh_result = x_values.cosh().unwrap();
        let cosh_data = cosh_result.data().unwrap();
        assert!((cosh_data[0] - 1.0_f32).abs() < 1e-6);
        assert!((cosh_data[1] - ((1.0_f32.exp() + (-1.0_f32).exp()) / 2.0)).abs() < 1e-6);
        assert!((cosh_data[2] - (((-1.0_f32).exp() + 1.0_f32.exp()) / 2.0)).abs() < 1e-6);

        // Test atan2
        let y = Tensor::from_data(vec![1.0, 1.0, -1.0, -1.0], vec![4], DeviceType::Cpu).unwrap();
        let x = Tensor::from_data(vec![1.0, -1.0, -1.0, 1.0], vec![4], DeviceType::Cpu).unwrap();
        let atan2_result = y.atan2(&x).unwrap();
        let atan2_data = atan2_result.data().unwrap();
        assert!((atan2_data[0] - (PI / 4.0)).abs() < 1e-6);
        assert!((atan2_data[1] - (3.0 * PI / 4.0)).abs() < 1e-6);
        assert!((atan2_data[2] - (-3.0 * PI / 4.0)).abs() < 1e-6);
        assert!((atan2_data[3] - (-PI / 4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sort() {
        // Test 1D sort
        let tensor =
            Tensor::from_data(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6], DeviceType::Cpu).unwrap();

        // Test ascending sort
        let (sorted, indices) = tensor.sort(None, false).unwrap();
        let sorted_data = sorted.data().unwrap();
        let indices_data = indices.data().unwrap();

        assert_eq!(sorted_data.as_slice(), &[1.0, 1.0, 3.0, 4.0, 5.0, 9.0]);
        assert_eq!(indices_data.as_slice(), &[1, 3, 0, 2, 4, 5]);

        // Test descending sort
        let (sorted, indices) = tensor.sort(None, true).unwrap();
        let sorted_data = sorted.data().unwrap();
        let indices_data = indices.data().unwrap();

        assert_eq!(sorted_data.as_slice(), &[9.0, 5.0, 4.0, 3.0, 1.0, 1.0]);
        assert_eq!(indices_data.as_slice(), &[5, 4, 2, 0, 1, 3]);

        // Test 2D sort along different dimensions
        let tensor_2d = tensor_2d(&[&[3.0, 1.0, 4.0], &[1.0, 5.0, 9.0]]).unwrap();

        // Sort along dim=0 (column-wise)
        let (sorted, indices) = tensor_2d.sort(Some(0), false).unwrap();
        let sorted_data = sorted.data().unwrap();
        let indices_data = indices.data().unwrap();

        // Expected: [[1.0, 1.0, 4.0], [3.0, 5.0, 9.0]]
        assert_eq!(sorted_data.as_slice(), &[1.0, 1.0, 4.0, 3.0, 5.0, 9.0]);
        assert_eq!(indices_data.as_slice(), &[1, 0, 0, 0, 1, 1]);

        // Sort along dim=1 (row-wise)
        let (sorted, indices) = tensor_2d.sort(Some(1), false).unwrap();
        let sorted_data = sorted.data().unwrap();
        let indices_data = indices.data().unwrap();

        // Expected: [[1.0, 3.0, 4.0], [1.0, 5.0, 9.0]]
        assert_eq!(sorted_data.as_slice(), &[1.0, 3.0, 4.0, 1.0, 5.0, 9.0]);
        assert_eq!(indices_data.as_slice(), &[1, 0, 2, 0, 1, 2]);
    }

    #[test]
    fn test_argsort() {
        let tensor =
            Tensor::from_data(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6], DeviceType::Cpu).unwrap();

        let indices = tensor.argsort(None, false).unwrap();
        let indices_data = indices.data().unwrap();

        assert_eq!(indices_data.as_slice(), &[1, 3, 0, 2, 4, 5]);

        // Test 2D argsort
        let tensor_2d = tensor_2d(&[&[3.0, 1.0, 4.0], &[1.0, 5.0, 9.0]]).unwrap();

        let indices = tensor_2d.argsort(Some(1), true).unwrap();
        let indices_data = indices.data().unwrap();

        // Expected indices for descending sort along dim=1: [[2, 0, 1], [2, 1, 0]]
        assert_eq!(indices_data.as_slice(), &[2, 0, 1, 2, 1, 0]);
    }

    #[test]
    fn test_topk() {
        let tensor =
            Tensor::from_data(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6], DeviceType::Cpu).unwrap();

        // Test top 3 largest
        let (values, indices) = tensor.topk(3, None, true, true).unwrap();
        let values_data = values.data().unwrap();
        let indices_data = indices.data().unwrap();

        assert_eq!(values_data.as_slice(), &[9.0, 5.0, 4.0]);
        assert_eq!(indices_data.as_slice(), &[5, 4, 2]);

        // Test top 3 smallest
        let (values, indices) = tensor.topk(3, None, false, true).unwrap();
        let values_data = values.data().unwrap();
        let indices_data = indices.data().unwrap();

        assert_eq!(values_data.as_slice(), &[1.0, 1.0, 3.0]);
        assert_eq!(indices_data.as_slice(), &[1, 3, 0]);

        // Test 2D topk
        let tensor_2d = tensor_2d(&[&[3.0, 1.0, 4.0], &[1.0, 5.0, 9.0]]).unwrap();

        // Top 2 along dim=1
        let (values, indices) = tensor_2d.topk(2, Some(1), true, true).unwrap();
        assert_eq!(values.shape().dims(), &[2, 2]);

        let values_data = values.data().unwrap();
        let indices_data = indices.data().unwrap();

        // Expected: [[4.0, 3.0], [9.0, 5.0]]
        assert_eq!(values_data.as_slice(), &[4.0, 3.0, 9.0, 5.0]);
        assert_eq!(indices_data.as_slice(), &[2, 0, 2, 1]);

        // Test edge cases
        assert!(tensor.topk(0, None, true, true).is_err());
        assert!(tensor.topk(10, None, true, true).is_err());
    }

    #[test]
    fn test_inplace_operations() {
        // Test basic arithmetic in-place operations
        let mut a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], DeviceType::Cpu).unwrap();

        // Test add_
        let _a_clone = a.clone();
        a.add_(&b).unwrap();
        let a_data = a.data().unwrap();
        assert_eq!(a_data.as_slice(), &[3.0, 5.0, 7.0, 9.0]);
        drop(a_data);

        // Test sub_
        a.sub_(&b).unwrap();
        let a_data = a.data().unwrap();
        assert_eq!(a_data.as_slice(), &[1.0, 2.0, 3.0, 4.0]); // Back to original
        drop(a_data);

        // Test mul_
        a.mul_(&b).unwrap();
        let a_data = a.data().unwrap();
        assert_eq!(a_data.as_slice(), &[2.0, 6.0, 12.0, 20.0]);
        drop(a_data);

        // Test div_
        a.div_(&b).unwrap();
        let a_data = a.data().unwrap();
        assert_eq!(a_data.as_slice(), &[1.0, 2.0, 3.0, 4.0]); // Back to original
        drop(a_data);

        // Test scalar operations
        a.add_scalar_(10.0).unwrap();
        let a_data = a.data().unwrap();
        assert_eq!(a_data.as_slice(), &[11.0, 12.0, 13.0, 14.0]);
        drop(a_data);

        a.sub_scalar_(10.0).unwrap();
        let a_data = a.data().unwrap();
        assert_eq!(a_data.as_slice(), &[1.0, 2.0, 3.0, 4.0]); // Back to original
        drop(a_data);

        a.mul_scalar_(2.0).unwrap();
        let a_data = a.data().unwrap();
        assert_eq!(a_data.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
        drop(a_data);

        a.div_scalar_(2.0).unwrap();
        let a_data = a.data().unwrap();
        assert_eq!(a_data.as_slice(), &[1.0, 2.0, 3.0, 4.0]); // Back to original
        drop(a_data);

        // Test mathematical operations
        let mut c = Tensor::from_data(vec![1.0, 4.0, 9.0, 16.0], vec![4], DeviceType::Cpu).unwrap();

        c.sqrt_().unwrap();
        let c_data = c.data().unwrap();
        assert_eq!(c_data.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        drop(c_data);

        c.pow_(2.0).unwrap();
        let c_data = c.data().unwrap();
        assert_eq!(c_data.as_slice(), &[1.0, 4.0, 9.0, 16.0]);
        drop(c_data);

        // Test clamp_
        let mut d = Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], DeviceType::Cpu).unwrap();
        d.clamp_(-1.0, 1.0).unwrap();
        let d_data = d.data().unwrap();
        assert_eq!(d_data.as_slice(), &[-1.0, -1.0, 0.0, 1.0, 1.0]);
        drop(d_data);

        // Test abs_
        let mut e = Tensor::from_data(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5], DeviceType::Cpu).unwrap();
        e.abs_().unwrap();
        let e_data = e.data().unwrap();
        assert_eq!(e_data.as_slice(), &[3.0, 1.0, 0.0, 1.0, 3.0]);
        drop(e_data);

        // Test neg_
        e.neg_().unwrap();
        let e_data = e.data().unwrap();
        assert_eq!(e_data.as_slice(), &[-3.0, -1.0, 0.0, -1.0, -3.0]);
        drop(e_data);

        // Test relu_
        let mut f = Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], DeviceType::Cpu).unwrap();
        f.relu_().unwrap();
        let f_data = f.data().unwrap();
        assert_eq!(f_data.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
        drop(f_data);

        // Test trigonometric operations
        let mut g = Tensor::from_data(
            vec![0.0, std::f32::consts::PI / 2.0],
            vec![2],
            DeviceType::Cpu,
        ).unwrap();
        g.sin_().unwrap();
        let g_data = g.data().unwrap();
        assert!((g_data[0] - 0.0).abs() < 1e-6);
        assert!((g_data[1] - 1.0).abs() < 1e-6);
        drop(g_data);

        // Test exp_ and log_
        let mut h = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        h.log_().unwrap();
        h.exp_().unwrap(); // Should be back to original
        let h_data = h.data().unwrap();
        assert!((h_data[0] - 1.0_f32).abs() < 1e-6);
        assert!((h_data[1] - 2.0_f32).abs() < 1e-6);
        assert!((h_data[2] - 3.0_f32).abs() < 1e-6);
    }
}

#[cfg(test)]
mod quantization_tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_qint8_quantization_dequantization() {
        // Test basic quantization and dequantization
        let float_data = vec![0.0f32, 1.0, 2.0, -1.0, -2.0];
        let tensor = Tensor::from_data(float_data.clone(), vec![5], DeviceType::Cpu).unwrap();
        
        // Test manual quantization
        let scale = 0.1f32;
        let zero_point = 0i8;
        let quantized = tensor.quantize_qint8(scale, zero_point).unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        let dequantized_data = dequantized.to_vec().unwrap();
        
        // Check that values are approximately preserved
        for (original, recovered) in float_data.iter().zip(dequantized_data.iter()) {
            assert!((original - recovered).abs() < 0.2, 
                "Original: {}, Recovered: {}", original, recovered);
        }
    }

    #[test]
    fn test_quint8_quantization_dequantization() {
        // Test basic quantization and dequantization for unsigned
        let float_data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(float_data.clone(), vec![5], DeviceType::Cpu).unwrap();
        
        // Test manual quantization
        let scale = 0.1f32;
        let zero_point = 128u8;
        let quantized = tensor.quantize_quint8(scale, zero_point).unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        let dequantized_data = dequantized.to_vec().unwrap();
        
        // Check that values are approximately preserved
        for (original, recovered) in float_data.iter().zip(dequantized_data.iter()) {
            assert!((original - recovered).abs() < 0.2, 
                "Original: {}, Recovered: {}", original, recovered);
        }
    }

    #[test]
    fn test_auto_quantization_qint8() {
        // Test automatic quantization parameter computation
        let float_data = vec![-5.0f32, -2.5, 0.0, 2.5, 5.0];
        let tensor = Tensor::from_data(float_data.clone(), vec![5], DeviceType::Cpu).unwrap();
        
        let (quantized, scale, _zero_point) = tensor.auto_quantize_qint8().unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        let dequantized_data = dequantized.to_vec().unwrap();
        
        // Scale should be (max - min) / 255
        let expected_scale = (5.0 - (-5.0)) / 255.0;
        assert!((scale - expected_scale).abs() < 1e-6, 
            "Expected scale: {}, Got: {}", expected_scale, scale);
        
        // Check that values are approximately preserved
        for (original, recovered) in float_data.iter().zip(dequantized_data.iter()) {
            assert!((original - recovered).abs() < 0.1, 
                "Original: {}, Recovered: {}", original, recovered);
        }
    }

    #[test]
    fn test_auto_quantization_quint8() {
        // Test automatic quantization for unsigned range
        let float_data = vec![0.0f32, 2.0, 4.0, 6.0, 8.0];
        let tensor = Tensor::from_data(float_data.clone(), vec![5], DeviceType::Cpu).unwrap();
        
        let (quantized, scale, _zero_point) = tensor.auto_quantize_quint8().unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        let dequantized_data = dequantized.to_vec().unwrap();
        
        // Scale should be (max - min) / 255
        let expected_scale = (8.0 - 0.0) / 255.0;
        assert!((scale - expected_scale).abs() < 1e-6, 
            "Expected scale: {}, Got: {}", expected_scale, scale);
        
        // Check that values are approximately preserved
        for (original, recovered) in float_data.iter().zip(dequantized_data.iter()) {
            assert!((original - recovered).abs() < 0.1, 
                "Original: {}, Recovered: {}", original, recovered);
        }
    }

    #[test]
    fn test_quantized_addition() {
        // Test addition of quantized tensors
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![0.5f32, 1.0, 1.5];
        
        let tensor1 = Tensor::from_data(data1, vec![3], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(data2, vec![3], DeviceType::Cpu).unwrap();
        
        let scale = 0.1f32;
        let zero_point = 0i8;
        
        let q1 = tensor1.quantize_qint8(scale, zero_point).unwrap();
        let q2 = tensor2.quantize_qint8(scale, zero_point).unwrap();
        
        let q_result = q1.add_quantized(&q2).unwrap();
        let dequantized_result = q_result.dequantize_f32().unwrap();
        let result_data = dequantized_result.to_vec().unwrap();
        
        // Check that quantized addition gives approximately correct results
        let expected = vec![1.5f32, 3.0, 4.5];
        for (expected_val, actual_val) in expected.iter().zip(result_data.iter()) {
            assert!((expected_val - actual_val).abs() < 0.3, 
                "Expected: {}, Got: {}", expected_val, actual_val);
        }
    }

    #[test]
    fn test_quantized_types_tensor_element() {
        // Test that quantized types implement TensorElement correctly
        assert_eq!(QInt8::dtype(), torsh_core::dtype::DType::QInt8);
        assert_eq!(QUInt8::dtype(), torsh_core::dtype::DType::QUInt8);
        
        // Test zero and one values
        let zero_qint8 = QInt8::zero();
        assert_eq!(zero_qint8.value, 0);
        
        let one_qint8 = QInt8::one();
        assert_eq!(one_qint8.value, 1);
        
        let zero_quint8 = QUInt8::zero();
        assert_eq!(zero_quint8.value, 128); // Typical zero_point for uint8
        
        let one_quint8 = QUInt8::one();
        assert_eq!(one_quint8.value, 129);
        
        // Test conversion functions
        let qint8_from_f64 = QInt8::from_f64(2.5).unwrap();
        assert_eq!(qint8_from_f64.dequantize(), 3.0); // Should be rounded to nearest integer (2.5 -> 3)
        
        let quint8_from_f64 = QUInt8::from_f64(3.7).unwrap();
        assert!((quint8_from_f64.dequantize() - 4.0).abs() < 0.1); // Should be close to 4.0
    }

    #[test]
    fn test_quantization_error_handling() {
        // Test error handling for empty tensors
        let empty_tensor: Tensor<f32> = Tensor::from_data(vec![], vec![0], DeviceType::Cpu).unwrap();
        assert!(empty_tensor.auto_quantize_qint8().is_err());
        assert!(empty_tensor.auto_quantize_quint8().is_err());
        
        // Test error handling for mismatched scale/zero_point in addition
        let data1 = vec![1.0f32, 2.0];
        let data2 = vec![1.0f32, 2.0];
        
        let tensor1 = Tensor::from_data(data1, vec![2], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(data2, vec![2], DeviceType::Cpu).unwrap();
        
        let q1 = tensor1.quantize_qint8(0.1, 0).unwrap();
        let q2 = tensor2.quantize_qint8(0.2, 0).unwrap(); // Different scale
        
        assert!(q1.add_quantized(&q2).is_err()); // Should fail due to different scales
    }

    #[test]
    fn test_quantization_precision_boundary() {
        // Test quantization at the boundaries of the data type ranges
        let float_data = vec![f32::MIN, f32::MAX, 0.0f32];
        let tensor = Tensor::from_data(float_data, vec![3], DeviceType::Cpu).unwrap();
        
        let (quantized, _scale, _zero_point) = tensor.auto_quantize_qint8().unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        
        // Should not panic and should produce valid results
        assert!(!dequantized.to_vec().unwrap().iter().any(|&x| x.is_nan()));
    }
}

/// Type promotion rules matching PyTorch's behavior
pub mod type_promotion {
    use torsh_core::dtype::DType;
    
    /// Type promotion hierarchy for numeric types (higher value = higher rank)
    fn type_rank(dtype: DType) -> u8 {
        match dtype {
            DType::Bool => 0,
            DType::U8 => 1,
            DType::I8 => 2,
            DType::I16 => 3,
            DType::I32 => 4,
            DType::U32 => 5,
            DType::I64 => 6,
            DType::U64 => 7,
            DType::F16 => 8,
            DType::BF16 => 9,
            DType::F32 => 10,
            DType::F64 => 11,
            DType::C64 => 12,
            DType::C128 => 13,
            DType::QInt8 => 14,   // Quantized types have special handling
            DType::QUInt8 => 15,
        }
    }
    
    /// Determine the result type for binary operations following PyTorch rules
    pub fn promote_types(lhs: DType, rhs: DType) -> DType {
        // Handle identical types
        if lhs == rhs {
            return lhs;
        }
        
        // Special handling for boolean
        if lhs == DType::Bool {
            return rhs;
        }
        if rhs == DType::Bool {
            return lhs;
        }
        
        // Special handling for quantized types
        if lhs.is_quantized() || rhs.is_quantized() {
            // Quantized operations require explicit handling
            // For now, promote to f32 for safety
            return DType::F32;
        }
        
        // Handle complex types - complex always wins
        if lhs.is_complex() && rhs.is_complex() {
            return if type_rank(lhs) > type_rank(rhs) { lhs } else { rhs };
        }
        if lhs.is_complex() {
            return lhs;
        }
        if rhs.is_complex() {
            return rhs;
        }
        
        // Handle float types - float always wins over integer
        if lhs.is_float() && rhs.is_float() {
            return if type_rank(lhs) > type_rank(rhs) { lhs } else { rhs };
        }
        if lhs.is_float() {
            return lhs;
        }
        if rhs.is_float() {
            return rhs;
        }
        
        // Handle integer types - higher precision wins
        if lhs.is_int() && rhs.is_int() {
            return if type_rank(lhs) > type_rank(rhs) { lhs } else { rhs };
        }
        
        // Fallback to higher rank
        if type_rank(lhs) > type_rank(rhs) { lhs } else { rhs }
    }
    
    /// Check if two types can be promoted to a common type
    pub fn can_promote(lhs: DType, rhs: DType) -> bool {
        // All numeric types can be promoted to some common type
        !lhs.is_quantized() || !rhs.is_quantized() || lhs == rhs
    }
    
    /// Get the common type for a slice of types
    pub fn promote_types_slice(types: &[DType]) -> Option<DType> {
        if types.is_empty() {
            return None;
        }
        
        let mut result = types[0];
        for &dtype in &types[1..] {
            if !can_promote(result, dtype) {
                return None;
            }
            result = promote_types(result, dtype);
        }
        Some(result)
    }
}

/// Type-promoted tensor operations
impl<T: TensorElement> Tensor<T> {
    /// Add tensors with automatic type promotion
    pub fn add_promoted<U: TensorElement>(&self, other: &Tensor<U>) -> Result<Box<dyn std::any::Any>> {
        let self_dtype = self.dtype();
        let other_dtype = other.dtype();
        let result_dtype = type_promotion::promote_types(self_dtype, other_dtype);
        
        // This is a simplified implementation - in practice you would need
        // to implement type conversion and create tensors of the promoted type
        match result_dtype {
            DType::F32 => {
                let self_f32 = self.to_f32()?;
                let other_f32 = other.to_f32()?;
                Ok(Box::new(self_f32.add_op(&other_f32)?))
            }
            DType::F64 => {
                let self_f64 = self.to_f64()?;
                let other_f64 = other.to_f64()?;
                Ok(Box::new(self_f64.add_op(&other_f64)?))
            }
            DType::I32 => {
                let self_i32 = self.to_i32()?;
                let other_i32 = other.to_i32()?;
                Ok(Box::new(self_i32.add_op(&other_i32)?))
            }
            DType::I64 => {
                let self_i64 = self.to_i64()?;
                let other_i64 = other.to_i64()?;
                Ok(Box::new(self_i64.add_op(&other_i64)?))
            }
            _ => Err(TorshError::InvalidArgument(format!(
                "Type promotion to {result_dtype:?} not implemented yet"
            )))
        }
    }
    
    /// Convert tensor to f32 type
    pub fn to_f32(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let converted_data: Vec<f32> = data.iter()
            .map(|val| val.to_f64().unwrap_or(0.0) as f32)
            .collect();
        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device())
    }
    
    /// Convert tensor to f64 type
    pub fn to_f64(&self) -> Result<Tensor<f64>> {
        let data = self.data()?;
        let converted_data: Vec<f64> = data.iter()
            .map(|val| val.to_f64().unwrap_or(0.0))
            .collect();
        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device())
    }
    
    /// Convert tensor to i32 type
    pub fn to_i32(&self) -> Result<Tensor<i32>> {
        let data = self.data()?;
        let converted_data: Vec<i32> = data.iter()
            .map(|val| val.to_f64().unwrap_or(0.0) as i32)
            .collect();
        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device())
    }
    
    /// Convert tensor to i64 type
    pub fn to_i64(&self) -> Result<Tensor<i64>> {
        let data = self.data()?;
        let converted_data: Vec<i64> = data.iter()
            .map(|val| val.to_f64().unwrap_or(0.0) as i64)
            .collect();
        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device())
    }
    
    /// Convert tensor to bool type
    pub fn to_bool(&self) -> Result<Tensor<bool>> {
        let data = self.data()?;
        let converted_data: Vec<bool> = data.iter()
            .map(|val| val.to_f64().unwrap_or(0.0) != 0.0)
            .collect();
        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device())
    }
    
}

#[cfg(test)]
mod type_promotion_tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_core::dtype::DType;

    #[test]
    fn test_type_promotion_rules() {
        use crate::ops::type_promotion::promote_types;
        
        // Boolean promotes to other type
        assert_eq!(promote_types(DType::Bool, DType::F32), DType::F32);
        assert_eq!(promote_types(DType::I32, DType::Bool), DType::I32);
        
        // Float types promote to higher precision
        assert_eq!(promote_types(DType::F32, DType::F64), DType::F64);
        assert_eq!(promote_types(DType::F16, DType::F32), DType::F32);
        
        // Float beats integer
        assert_eq!(promote_types(DType::I32, DType::F32), DType::F32);
        assert_eq!(promote_types(DType::F64, DType::I64), DType::F64);
        
        // Integer types promote to higher precision
        assert_eq!(promote_types(DType::I32, DType::I64), DType::I64);
        assert_eq!(promote_types(DType::U8, DType::I32), DType::I32);
        
        // Complex types win over everything
        assert_eq!(promote_types(DType::F32, DType::C64), DType::C64);
        assert_eq!(promote_types(DType::C64, DType::C128), DType::C128);
        
        // Same type returns same type
        assert_eq!(promote_types(DType::F32, DType::F32), DType::F32);
    }

    #[test]
    fn test_type_promotion_slice() {
        use crate::ops::type_promotion::promote_types_slice;
        
        // Test promotion of multiple types
        let types = vec![DType::I32, DType::F32, DType::I64];
        assert_eq!(promote_types_slice(&types), Some(DType::F32));
        
        let types = vec![DType::U8, DType::I16, DType::I32];
        assert_eq!(promote_types_slice(&types), Some(DType::I32));
        
        let types = vec![DType::Bool, DType::F64];
        assert_eq!(promote_types_slice(&types), Some(DType::F64));
        
        // Empty slice
        assert_eq!(promote_types_slice(&[]), None);
        
        // Single type
        assert_eq!(promote_types_slice(&[DType::F32]), Some(DType::F32));
    }

    #[test]
    fn test_tensor_type_conversion() {
        // Test f32 to f64 conversion
        let f32_data = vec![1.0f32, 2.5, 3.7];
        let f32_tensor = Tensor::from_data(f32_data.clone(), vec![3], DeviceType::Cpu).unwrap();
        
        let f64_tensor = f32_tensor.to_f64().unwrap();
        let f64_data = f64_tensor.to_vec().unwrap();
        
        for (orig, conv) in f32_data.iter().zip(f64_data.iter()) {
            assert!(((*orig as f64) - conv).abs() < 1e-6);
        }
        
        // Test i32 conversion
        let i32_tensor = f32_tensor.to_i32().unwrap();
        let i32_data = i32_tensor.to_vec().unwrap();
        assert_eq!(i32_data, vec![1i32, 2, 3]); // Truncated values
        
        // Test bool conversion
        let bool_values = vec![0.0f32, 1.0, -1.0, 0.0];
        let bool_tensor_src = Tensor::from_data(bool_values, vec![4], DeviceType::Cpu).unwrap();
        let bool_tensor = bool_tensor_src.to_bool().unwrap();
        let bool_data = bool_tensor.to_vec().unwrap();
        assert_eq!(bool_data, vec![false, true, true, false]);
    }

    #[test]
    fn test_dtype_introspection() {
        let f32_tensor = Tensor::from_data(vec![1.0f32], vec![1], DeviceType::Cpu).unwrap();
        assert_eq!(f32_tensor.dtype(), DType::F32);
        
        let i64_tensor = Tensor::from_data(vec![1i64], vec![1], DeviceType::Cpu).unwrap();
        assert_eq!(i64_tensor.dtype(), DType::I64);
    }

    #[test]
    fn test_type_promotion_compatibility() {
        use crate::ops::type_promotion::can_promote;
        
        // Compatible types
        assert!(can_promote(DType::F32, DType::I32));
        assert!(can_promote(DType::I32, DType::I64));
        assert!(can_promote(DType::Bool, DType::F64));
        
        // Quantized types have special handling
        assert!(can_promote(DType::QInt8, DType::QInt8)); // Same quantized type
    }
}

// Additional tensor operations that were missing from the PyTorch API

impl Tensor<f32> {
    /// Fill tensor with random values from exponential distribution (in-place)
    pub fn exponential_(&mut self, lambd: f32) -> Result<()> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random};

        let exp = Exp::new(lambd).map_err(|e| {
            TorshError::ComputeError(format!("Invalid exponential distribution parameter: {e}"))
        })?;

        let mut rng = Random::new();
        
        self.data_mut_apply(|item| {
            *item = exp.sample(&mut rng);
        })?;

        Ok(())
    }

    /// Fill tensor with random values from geometric distribution (in-place)
    pub fn geometric_(&mut self, p: f32) -> Result<()> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random};

        if p <= 0.0 || p > 1.0 {
            return Err(TorshError::ComputeError(
                "Geometric distribution parameter p must be in (0, 1]".into(),
            ));
        }

        let mut rng = Random::new();
        let mut data = self.data()?;

        for item in data.iter_mut() {
            // Generate geometric random variable using inverse transform
            let u: f32 = rng.random();
            *item = ((1.0 - u).ln() / (1.0 - p).ln()).floor();
        }

        Ok(())
    }

    /// Concatenate tensors along a given dimension (alternative implementation)
    pub fn cat_alt(tensors: &[&Self], dim: i32) -> Result<Self>
    {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot concatenate empty list of tensors".to_string(),
            ));
        }

        let first_tensor = tensors[0];
        let ndim = first_tensor.ndim() as i32;
        
        // Normalize negative dimension
        let normalized_dim = if dim < 0 {
            (ndim + dim) as usize
        } else {
            dim as usize
        };

        if normalized_dim >= first_tensor.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim, first_tensor.ndim()
            )));
        }

        // Validate all tensors have the same shape except in the concatenation dimension
        let first_shape_binding = first_tensor.shape();
        let first_shape = first_shape_binding.dims();
        let mut total_cat_dim_size = 0;

        for tensor in tensors.iter() {
            if tensor.ndim() != first_tensor.ndim() {
                return Err(TorshError::ShapeMismatch {
                    expected: vec![first_tensor.ndim()],
                    got: vec![tensor.ndim()],
                });
            }

            let tensor_shape_binding = tensor.shape();
            let tensor_shape = tensor_shape_binding.dims();
            total_cat_dim_size += tensor_shape[normalized_dim];

            // Check all other dimensions match
            for (j, (&expected, &actual)) in first_shape.iter().zip(tensor_shape).enumerate() {
                if j != normalized_dim && expected != actual {
                    return Err(TorshError::ShapeMismatch {
                        expected: first_shape.to_vec(),
                        got: tensor_shape.to_vec(),
                    });
                }
            }

            // Check device compatibility
            if tensor.device != first_tensor.device {
                return Err(TorshError::DeviceMismatch);
            }
        }

        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        output_shape[normalized_dim] = total_cat_dim_size;

        // Concatenate data
        let mut output_data = Vec::with_capacity(output_shape.iter().product());

        // Calculate strides for efficient copying - TODO: implement stride-based copying
        
        // Copy data from each tensor
        let mut current_offset = 0;
        for tensor in tensors {
            let tensor_data = tensor.data()?;

            let tensor_shape_binding = tensor.shape();
            let tensor_shape = tensor_shape_binding.dims();
            let tensor_size = tensor_shape[normalized_dim];
            
            // For each position in other dimensions, copy the slice from this tensor
            let num_slices = tensor_shape.iter().enumerate()
                .filter(|(i, _)| *i != normalized_dim)
                .map(|(_, &size)| size)
                .product::<usize>();

            if num_slices == 0 {
                // Handle empty tensor case
                continue;
            }

            // Simple concatenation along the specified dimension
            if normalized_dim == tensor.ndim() - 1 {
                // Last dimension - can copy in chunks
                let chunk_size = tensor_shape[normalized_dim];
                let num_chunks = tensor_data.len() / chunk_size;
                
                for chunk_idx in 0..num_chunks {
                    let start = chunk_idx * chunk_size;
                    let end = start + chunk_size;
                    output_data.extend_from_slice(&tensor_data[start..end]);
                }
            } else {
                // More complex case - need to interleave data properly
                Self::copy_tensor_slice(
                    &tensor_data,
                    &mut output_data,
                    tensor_shape,
                    &output_shape,
                    normalized_dim,
                    current_offset,
                )?;
            }
            
            current_offset += tensor_size;
        }

        // If output_data is empty, fill it appropriately
        if output_data.is_empty() {
            output_data.resize(output_shape.iter().product(), 0.0f32);
        }

        Self::from_data(
            output_data,
            output_shape,
            first_tensor.device,
        )
    }

    /// Stack tensors along a new dimension (alternative implementation)
    pub fn stack_alt(tensors: &[&Self], dim: i32) -> Result<Self>
    {
        if tensors.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot stack empty list of tensors".to_string(),
            ));
        }

        let first_tensor = tensors[0];
        let ndim = first_tensor.ndim() as i32;
        
        // Normalize negative dimension (for stacking, we can insert at ndim)
        let normalized_dim = if dim < 0 {
            (ndim + dim + 1) as usize
        } else {
            dim as usize
        };

        if normalized_dim > first_tensor.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for stacking tensor with {} dimensions",
                dim, first_tensor.ndim()
            )));
        }

        // Validate all tensors have the same shape
        let first_shape_binding = first_tensor.shape();
        let first_shape = first_shape_binding.dims();
        for tensor in tensors.iter() {
            if tensor.shape().dims() != first_shape {
                return Err(TorshError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    got: tensor.shape().dims().to_vec(),
                });
            }

            // Check device compatibility
            if tensor.device != first_tensor.device {
                return Err(TorshError::DeviceMismatch);
            }
        }

        // Calculate output shape by inserting the stack dimension
        let mut output_shape = Vec::with_capacity(first_shape.len() + 1);
        output_shape.extend_from_slice(&first_shape[..normalized_dim]);
        output_shape.push(tensors.len());
        output_shape.extend_from_slice(&first_shape[normalized_dim..]);

        // Stack data
        let tensor_size = first_tensor.numel();
        let mut output_data = Vec::with_capacity(tensor_size * tensors.len());

        // Copy data from each tensor in order
        for tensor in tensors {
            let tensor_data = tensor.data()?;
            output_data.extend_from_slice(&tensor_data);
        }

        Self::from_data(
            output_data,
            output_shape,
            first_tensor.device,
        )
    }


    /// Helper function to copy tensor slice for concatenation
    fn copy_tensor_slice(
        src_data: &[f32],
        dst_data: &mut Vec<f32>,
        _src_shape: &[usize],
        _dst_shape: &[usize],
        _cat_dim: usize,
        _offset: usize,
    ) -> Result<()>
    {
        // This is a simplified implementation
        // For now, just append the source data (works for last dimension concatenation)
        dst_data.extend_from_slice(src_data);
        Ok(())
    }

}

#[cfg(test)]
mod additional_ops_tests {
    use super::*;
    use crate::creation::*;

    #[test]
    fn test_exponential_fill() {
        let mut tensor = zeros(&[1000]).unwrap();
        tensor.exponential_(1.0).unwrap();

        let data = tensor.data().unwrap();

        // All values should be positive for exponential distribution
        for &val in data.iter() {
            assert!(val >= 0.0);
        }

        // Should have reasonable mean (approximately 1/lambda = 1)
        // With larger sample size and wider range for robustness
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean > 0.3 && mean < 3.0);
    }

    #[test]
    fn test_geometric_fill() {
        let mut tensor = zeros(&[100]).unwrap();
        tensor.geometric_(0.5).unwrap();

        let data = tensor.data().unwrap();

        // All values should be non-negative for geometric distribution
        for &val in data.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_concatenation() {
        use crate::creation::tensor_1d;

        // Test cat along dimension 0
        let tensor1 = tensor_1d(&[1.0, 2.0]).unwrap();
        let tensor2 = tensor_1d(&[3.0, 4.0]).unwrap();
        let tensor3 = tensor_1d(&[5.0, 6.0]).unwrap();

        let tensors = vec![&tensor1, &tensor2, &tensor3];
        let result = Tensor::cat(&tensors, 0).unwrap();

        assert_eq!(result.shape().dims(), &[6]);
        assert_eq!(result.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test 2D concatenation
        let t1 = crate::creation::tensor_2d(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let t2 = crate::creation::tensor_2d(&[&[5.0, 6.0], &[7.0, 8.0]]).unwrap();

        let tensors_2d = vec![&t1, &t2];
        let result_2d = Tensor::cat(&tensors_2d, 0).unwrap();
        assert_eq!(result_2d.shape().dims(), &[4, 2]);

        // Test concatenation along dimension 1
        let result_dim1 = Tensor::cat(&tensors_2d, 1).unwrap();
        assert_eq!(result_dim1.shape().dims(), &[2, 4]);
    }

    #[test]
    fn test_stacking() {
        use crate::creation::tensor_1d;

        // Test stack along dimension 0
        let tensor1 = tensor_1d(&[1.0, 2.0]).unwrap();
        let tensor2 = tensor_1d(&[3.0, 4.0]).unwrap();
        let tensor3 = tensor_1d(&[5.0, 6.0]).unwrap();

        let tensors = vec![&tensor1, &tensor2, &tensor3];
        let result = Tensor::stack(&tensors, 0).unwrap();

        assert_eq!(result.shape().dims(), &[3, 2]);
        assert_eq!(result.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test stack along dimension 1
        let result_dim1 = Tensor::stack(&tensors, 1).unwrap();
        assert_eq!(result_dim1.shape().dims(), &[2, 3]);

        // Test negative dimension
        let result_neg = Tensor::stack(&tensors, -1).unwrap();
        assert_eq!(result_neg.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_concatenation_errors() {
        use crate::creation::tensor_1d;

        // Test empty tensor list
        let empty_tensors: Vec<&Tensor<f32>> = vec![];
        assert!(Tensor::cat(&empty_tensors, 0).is_err());
        assert!(Tensor::stack(&empty_tensors, 0).is_err());

        // Test shape mismatch for cat
        let tensor1 = tensor_1d(&[1.0, 2.0]).unwrap();
        let tensor2 = tensor_1d(&[3.0, 4.0, 5.0]).unwrap();
        let tensors = vec![&tensor1, &tensor2];
        // This should work for cat (different size in concat dimension)
        assert!(Tensor::cat(&tensors, 0).is_ok());

        // Test shape mismatch for stack
        assert!(Tensor::stack(&tensors, 0).is_err());

        // Test dimension out of range
        let same_tensors = vec![&tensor1, &tensor1];
        assert!(Tensor::cat(&same_tensors, 2).is_err()); // 1D tensor, dim 2 is invalid
        assert!(Tensor::stack(&same_tensors, 3).is_err()); // Can't stack beyond ndim+1
    }

    #[test]
    fn test_split_operations() {
        use crate::creation::tensor_1d;

        // Test split with specific sizes
        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let splits = tensor.split_sections(&[2, 2, 2], 0).unwrap();

        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].shape().dims(), &[2]);
        assert_eq!(splits[0].to_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(splits[1].to_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(splits[2].to_vec().unwrap(), vec![5.0, 6.0]);

        // Test uneven splits
        let uneven_splits = tensor.split_sections(&[1, 2, 3], 0).unwrap();
        assert_eq!(uneven_splits.len(), 3);
        assert_eq!(uneven_splits[0].shape().dims(), &[1]);
        assert_eq!(uneven_splits[1].shape().dims(), &[2]);
        assert_eq!(uneven_splits[2].shape().dims(), &[3]);

        // Test error cases
        assert!(tensor.split_sections(&[2, 2, 3], 0).is_err()); // Sizes don't sum to dimension size
        assert!(tensor.split_sections(&[0, 6], 0).is_err()); // Zero size not allowed
        assert!(tensor.split_sections(&[6], 2).is_err()); // Dimension out of range
    }

    #[test]
    fn test_chunk_operations() {
        use crate::creation::tensor_1d;

        // Test even chunks
        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let chunks = tensor.chunk(3, 0).unwrap();

        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.shape().dims(), &[2]);
        }
        assert_eq!(chunks[0].to_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(chunks[1].to_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(chunks[2].to_vec().unwrap(), vec![5.0, 6.0]);

        // Test uneven chunks
        let tensor7 = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
        let uneven_chunks = tensor7.chunk(3, 0).unwrap();

        assert_eq!(uneven_chunks.len(), 3);
        assert_eq!(uneven_chunks[0].shape().dims(), &[3]); // Gets extra element
        assert_eq!(uneven_chunks[1].shape().dims(), &[2]);
        assert_eq!(uneven_chunks[2].shape().dims(), &[2]);

        // Test error cases
        assert!(tensor.chunk(0, 0).is_err()); // Zero chunks not allowed
        assert!(tensor.chunk(3, 2).is_err()); // Dimension out of range
        assert!(tensor_1d(&[1.0]).unwrap().chunk(5, 0).is_err()); // Too many chunks for size
    }

    #[test]
    fn test_unbind_operations() {
        use crate::creation::tensor_2d;

        // Test unbinding 2D tensor along dimension 0
        let tensor = tensor_2d(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]).unwrap();
        let unbound = tensor.unbind(0).unwrap();

        assert_eq!(unbound.len(), 3);
        for slice in &unbound {
            assert_eq!(slice.shape().dims(), &[2]); // One dimension removed
        }
        assert_eq!(unbound[0].to_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(unbound[1].to_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(unbound[2].to_vec().unwrap(), vec![5.0, 6.0]);

        // Test unbinding along dimension 1
        let unbound_dim1 = tensor.unbind(1).unwrap();
        assert_eq!(unbound_dim1.len(), 2);
        for slice in &unbound_dim1 {
            assert_eq!(slice.shape().dims(), &[3]); // One dimension removed
        }

        // Test negative dimension
        let unbound_neg = tensor.unbind(-1).unwrap();
        assert_eq!(unbound_neg.len(), 2);

        // Test error case
        assert!(tensor.unbind(3).is_err()); // Dimension out of range
    }

    #[test]
    fn test_view_as_complex() {
        use crate::creation::tensor_1d;
        use torsh_core::dtype::Complex32;

        // Test 1D tensor with 2 elements -> scalar complex
        let real_tensor = tensor_1d(&[1.0f32, 2.0f32]).unwrap();
        let complex_tensor: Tensor<Complex32> = real_tensor.view_as_complex().unwrap();

        assert_eq!(complex_tensor.shape().dims(), &[0; 0]);
        assert_eq!(complex_tensor.numel(), 1);

        let data = complex_tensor.to_vec().unwrap();
        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[0].im, 2.0);

        // Test 2D tensor [2, 2] -> [2] complex
        let real_tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            torsh_core::device::DeviceType::Cpu,
        ).unwrap();
        let complex_tensor: Tensor<Complex32> = real_tensor.view_as_complex().unwrap();

        assert_eq!(complex_tensor.shape().dims(), &[2]);
        assert_eq!(complex_tensor.numel(), 2);

        let data = complex_tensor.to_vec().unwrap();
        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[0].im, 2.0);
        assert_eq!(data[1].re, 3.0);
        assert_eq!(data[1].im, 4.0);

        // Test 3D tensor [2, 3, 2] -> [2, 3] complex
        let real_tensor = Tensor::from_data(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
            torsh_core::device::DeviceType::Cpu,
        ).unwrap();
        let complex_tensor: Tensor<Complex32> = real_tensor.view_as_complex().unwrap();

        assert_eq!(complex_tensor.shape().dims(), &[2, 3]);
        assert_eq!(complex_tensor.numel(), 6);

        let data = complex_tensor.to_vec().unwrap();
        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[0].im, 2.0);
        assert_eq!(data[1].re, 3.0);
        assert_eq!(data[1].im, 4.0);
        assert_eq!(data[5].re, 11.0);
        assert_eq!(data[5].im, 12.0);
    }

    #[test]
    fn test_view_as_complex_errors() {
        // Test error when last dimension is not 2
        let bad_tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3],
            torsh_core::device::DeviceType::Cpu,
        ).unwrap();
        let result: Result<Tensor<torsh_core::dtype::Complex32>> = bad_tensor.view_as_complex();
        assert!(result.is_err());

        // Test error when last dimension is not 2 (for 2D case)
        let bad_tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            torsh_core::device::DeviceType::Cpu,
        ).unwrap();
        let result: Result<Tensor<torsh_core::dtype::Complex32>> = bad_tensor.view_as_complex();
        assert!(result.is_err());

        // Test error with empty tensor
        let empty_tensor = Tensor::from_data(vec![], vec![], torsh_core::device::DeviceType::Cpu).unwrap();
        let result: Result<Tensor<torsh_core::dtype::Complex32>> = empty_tensor.view_as_complex();
        assert!(result.is_err());
    }

    #[test]
    fn test_view_as_real() {
        use torsh_core::dtype::Complex32;

        // Test scalar complex -> [2] real
        let complex_data = vec![Complex32::new(1.0, 2.0)];
        let complex_tensor =
            Tensor::from_data(complex_data, vec![], torsh_core::device::DeviceType::Cpu).unwrap();
        let real_tensor = complex_tensor.view_as_real().unwrap();

        assert_eq!(real_tensor.shape().dims(), &[2]);
        assert_eq!(real_tensor.numel(), 2);

        let data = real_tensor.to_vec().unwrap();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);

        // Test 1D complex [2] -> [2, 2] real
        let complex_data = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let complex_tensor =
            Tensor::from_data(complex_data, vec![2], torsh_core::device::DeviceType::Cpu).unwrap();
        let real_tensor = complex_tensor.view_as_real().unwrap();

        assert_eq!(real_tensor.shape().dims(), &[2, 2]);
        assert_eq!(real_tensor.numel(), 4);

        let data = real_tensor.to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);

        // Test 2D complex [2, 3] -> [2, 3, 2] real
        let complex_data = vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
            Complex32::new(5.0, 6.0),
            Complex32::new(7.0, 8.0),
            Complex32::new(9.0, 10.0),
            Complex32::new(11.0, 12.0),
        ];
        let complex_tensor = Tensor::from_data(
            complex_data,
            vec![2, 3],
            torsh_core::device::DeviceType::Cpu,
        ).unwrap();
        let real_tensor = complex_tensor.view_as_real().unwrap();

        assert_eq!(real_tensor.shape().dims(), &[2, 3, 2]);
        assert_eq!(real_tensor.numel(), 12);

        let data = real_tensor.to_vec().unwrap();
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
    }

    #[test]
    fn test_view_transformations_roundtrip() {
        use torsh_core::dtype::Complex32;

        // Test real -> complex -> real roundtrip
        let original = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
            torsh_core::device::DeviceType::Cpu,
        ).unwrap();

        let complex_view: Tensor<Complex32> = original.view_as_complex().unwrap();
        let real_view = complex_view.view_as_real().unwrap();

        assert_eq!(original.shape().dims(), real_view.shape().dims());
        assert_eq!(original.to_vec().unwrap(), real_view.to_vec().unwrap());

        // Test complex -> real -> complex roundtrip
        let complex_data = vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
            Complex32::new(5.0, 6.0),
        ];
        let original_complex =
            Tensor::from_data(complex_data, vec![3], torsh_core::device::DeviceType::Cpu).unwrap();

        let real_view = original_complex.view_as_real().unwrap();
        let complex_view: Tensor<Complex32> = real_view.view_as_complex().unwrap();

        assert_eq!(original_complex.shape().dims(), complex_view.shape().dims());

        let orig_data = original_complex.to_vec().unwrap();
        let roundtrip_data = complex_view.to_vec().unwrap();
        for (orig, roundtrip) in orig_data.iter().zip(roundtrip_data.iter()) {
            assert_eq!(orig.re, roundtrip.re);
            assert_eq!(orig.im, roundtrip.im);
        }
    }

    #[test]
    fn test_view_transformations_gradient_preservation() {
        use torsh_core::dtype::Complex32;

        // Test that requires_grad is preserved in view_as_complex
        let mut real_tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            torsh_core::device::DeviceType::Cpu,
        ).unwrap();
        real_tensor.requires_grad = true;

        let complex_tensor: Tensor<Complex32> = real_tensor.view_as_complex().unwrap();
        assert!(complex_tensor.requires_grad);

        // Test that requires_grad is preserved in view_as_real
        let complex_data = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let mut complex_tensor =
            Tensor::from_data(complex_data, vec![2], torsh_core::device::DeviceType::Cpu).unwrap();
        complex_tensor.requires_grad = true;

        let real_tensor = complex_tensor.view_as_real().unwrap();
        assert!(real_tensor.requires_grad);
    }

    #[test]
    fn test_complex_mathematical_functions() {
        use torsh_core::dtype::Complex32;

        // Test complex exponential
        let complex_data = vec![Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)];
        let tensor = Tensor::from_data(complex_data, vec![2], torsh_core::device::DeviceType::Cpu).unwrap();

        let exp_result = tensor.exp_complex().unwrap();
        let exp_data = exp_result.to_vec().unwrap();

        // exp(0) = 1, exp(1) ≈ 2.718
        assert!((exp_data[0].re - 1.0).abs() < 1e-6);
        assert!(exp_data[0].im.abs() < 1e-6);
        assert!((exp_data[1].re - std::f32::consts::E).abs() < 1e-6);
        assert!(exp_data[1].im.abs() < 1e-6);

        // Test complex logarithm
        let complex_data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(std::f32::consts::E, 0.0),
        ];
        let tensor = Tensor::from_data(complex_data, vec![2], torsh_core::device::DeviceType::Cpu).unwrap();

        let log_result = tensor.log_complex().unwrap();
        let log_data = log_result.to_vec().unwrap();

        // ln(1) = 0, ln(e) = 1
        assert!(log_data[0].re.abs() < 1e-6);
        assert!(log_data[0].im.abs() < 1e-6);
        assert!((log_data[1].re - 1.0).abs() < 1e-6);
        assert!(log_data[1].im.abs() < 1e-6);

        // Test complex square root
        let complex_data = vec![Complex32::new(4.0, 0.0), Complex32::new(0.0, 4.0)];
        let tensor = Tensor::from_data(complex_data, vec![2], torsh_core::device::DeviceType::Cpu).unwrap();

        let sqrt_result = tensor.sqrt_complex().unwrap();
        let sqrt_data = sqrt_result.to_vec().unwrap();

        // sqrt(4) = 2, sqrt(4i) ≈ sqrt(2) + sqrt(2)i
        assert!((sqrt_data[0].re - 2.0).abs() < 1e-6);
        assert!(sqrt_data[0].im.abs() < 1e-6);
        let expected = (2.0_f32).sqrt();
        assert!((sqrt_data[1].re - expected).abs() < 1e-6);
        assert!((sqrt_data[1].im - expected).abs() < 1e-6);
    }

    #[test]
    fn test_complex_power_functions() {
        use torsh_core::dtype::Complex32;

        // Test complex power with scalar exponent
        let complex_data = vec![Complex32::new(2.0, 0.0), Complex32::new(0.0, 2.0)];
        let tensor = Tensor::from_data(complex_data, vec![2], torsh_core::device::DeviceType::Cpu).unwrap();

        let pow_result = tensor.pow_complex_scalar(Complex32::new(2.0, 0.0)).unwrap();
        let pow_data = pow_result.to_vec().unwrap();

        // 2^2 = 4, (2i)^2 = -4
        assert!((pow_data[0].re - 4.0).abs() < 1e-6);
        assert!(pow_data[0].im.abs() < 1e-6);
        assert!((pow_data[1].re - (-4.0)).abs() < 1e-6);
        assert!(pow_data[1].im.abs() < 1e-6);

        // Test complex power with tensor exponent
        let base = vec![Complex32::new(2.0, 0.0)];
        let exponent = vec![Complex32::new(3.0, 0.0)];

        let base_tensor = Tensor::from_data(base, vec![1], torsh_core::device::DeviceType::Cpu).unwrap();
        let exp_tensor = Tensor::from_data(exponent, vec![1], torsh_core::device::DeviceType::Cpu).unwrap();

        let pow_result = base_tensor.pow_complex(&exp_tensor).unwrap();
        let pow_data = pow_result.to_vec().unwrap();

        // 2^3 = 8
        assert!((pow_data[0].re - 8.0).abs() < 1e-6);
        assert!(pow_data[0].im.abs() < 1e-6);
    }

    #[test]
    fn test_complex_trigonometric_functions() {
        use torsh_core::dtype::Complex32;

        // Test complex sine and cosine
        let complex_data = vec![
            Complex32::new(0.0, 0.0),
            Complex32::new(std::f32::consts::PI / 2.0, 0.0),
        ];
        let tensor = Tensor::from_data(complex_data, vec![2], torsh_core::device::DeviceType::Cpu).unwrap();

        let sin_result = tensor.sin_complex().unwrap();
        let sin_data = sin_result.to_vec().unwrap();

        // sin(0) = 0, sin(π/2) = 1
        assert!(sin_data[0].re.abs() < 1e-6);
        assert!(sin_data[0].im.abs() < 1e-6);
        assert!((sin_data[1].re - 1.0).abs() < 1e-6);
        assert!(sin_data[1].im.abs() < 1e-6);

        let cos_result = tensor.cos_complex().unwrap();
        let cos_data = cos_result.to_vec().unwrap();

        // cos(0) = 1, cos(π/2) = 0
        assert!((cos_data[0].re - 1.0).abs() < 1e-6);
        assert!(cos_data[0].im.abs() < 1e-6);
        assert!(cos_data[1].re.abs() < 1e-6);
        assert!(cos_data[1].im.abs() < 1e-6);
    }

    #[test]
    fn test_complex_hyperbolic_functions() {
        use torsh_core::dtype::Complex32;

        // Test complex hyperbolic functions
        let complex_data = vec![Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.0)];
        let tensor = Tensor::from_data(complex_data, vec![2], torsh_core::device::DeviceType::Cpu).unwrap();

        let sinh_result = tensor.sinh_complex().unwrap();
        let sinh_data = sinh_result.to_vec().unwrap();

        // sinh(0) = 0, sinh(1) ≈ 1.175
        assert!(sinh_data[0].re.abs() < 1e-6);
        assert!(sinh_data[0].im.abs() < 1e-6);
        assert!((sinh_data[1].re - 1.0_f32.sinh()).abs() < 1e-6);
        assert!(sinh_data[1].im.abs() < 1e-6);

        let cosh_result = tensor.cosh_complex().unwrap();
        let cosh_data = cosh_result.to_vec().unwrap();

        // cosh(0) = 1, cosh(1) ≈ 1.543
        assert!((cosh_data[0].re - 1.0).abs() < 1e-6);
        assert!(cosh_data[0].im.abs() < 1e-6);
        assert!((cosh_data[1].re - 1.0_f32.cosh()).abs() < 1e-6);
        assert!(cosh_data[1].im.abs() < 1e-6);

        let tanh_result = tensor.tanh_complex().unwrap();
        let tanh_data = tanh_result.to_vec().unwrap();

        // tanh(0) = 0, tanh(1) ≈ 0.762
        assert!(tanh_data[0].re.abs() < 1e-6);
        assert!(tanh_data[0].im.abs() < 1e-6);
        assert!((tanh_data[1].re - 1.0_f32.tanh()).abs() < 1e-6);
        assert!(tanh_data[1].im.abs() < 1e-6);
    }

    #[test]
    fn test_complex_functions_gradient_tracking() {
        use torsh_core::dtype::Complex32;

        // Test that gradient tracking is preserved
        let complex_data = vec![Complex32::new(1.0, 2.0)];
        let mut tensor =
            Tensor::from_data(complex_data, vec![1], torsh_core::device::DeviceType::Cpu).unwrap();
        tensor.requires_grad = true;

        // Test exp_complex gradient tracking
        let exp_result = tensor.exp_complex().unwrap();
        assert!(exp_result.requires_grad);

        // Test log_complex gradient tracking
        let log_result = tensor.log_complex().unwrap();
        assert!(log_result.requires_grad);

        // Test sqrt_complex gradient tracking
        let sqrt_result = tensor.sqrt_complex().unwrap();
        assert!(sqrt_result.requires_grad);

        // Test pow_complex_scalar gradient tracking
        let pow_result = tensor.pow_complex_scalar(Complex32::new(2.0, 0.0)).unwrap();
        assert!(pow_result.requires_grad);

        // Test trigonometric functions gradient tracking
        let sin_result = tensor.sin_complex().unwrap();
        assert!(sin_result.requires_grad);

        let cos_result = tensor.cos_complex().unwrap();
        assert!(cos_result.requires_grad);

        // Test hyperbolic functions gradient tracking
        let sinh_result = tensor.sinh_complex().unwrap();
        assert!(sinh_result.requires_grad);

        let cosh_result = tensor.cosh_complex().unwrap();
        assert!(cosh_result.requires_grad);

        let tanh_result = tensor.tanh_complex().unwrap();
        assert!(tanh_result.requires_grad);
    }

    #[test]
    fn test_comprehensive_complex_operations() {
        use torsh_core::dtype::{Complex32, Complex64};
        use torsh_core::device::DeviceType;

        // Test Complex32 operations
        let complex32_data = vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
            Complex32::new(0.5, -1.0),
        ];
        let tensor32 = Tensor::from_data(complex32_data.clone(), vec![3], DeviceType::Cpu).unwrap();

        // Test complex conjugate
        let conj_result = tensor32.conj().unwrap();
        let conj_data = conj_result.to_vec().unwrap();
        assert_eq!(conj_data[0], Complex32::new(1.0, -2.0));
        assert_eq!(conj_data[1], Complex32::new(3.0, -4.0));
        assert_eq!(conj_data[2], Complex32::new(0.5, 1.0));

        // Test inverse trigonometric functions
        let asin_result = tensor32.asin_complex().unwrap();
        assert_eq!(asin_result.shape().dims(), &[3]);

        let acos_result = tensor32.acos_complex().unwrap();
        assert_eq!(acos_result.shape().dims(), &[3]);

        let atan_result = tensor32.atan_complex().unwrap();
        assert_eq!(atan_result.shape().dims(), &[3]);

        // Test inverse hyperbolic functions
        let asinh_result = tensor32.asinh_complex().unwrap();
        assert_eq!(asinh_result.shape().dims(), &[3]);

        let acosh_result = tensor32.acosh_complex().unwrap();
        assert_eq!(acosh_result.shape().dims(), &[3]);

        let atanh_result = tensor32.atanh_complex().unwrap();
        assert_eq!(atanh_result.shape().dims(), &[3]);

        // Test logarithm functions
        let log10_result = tensor32.log10_complex().unwrap();
        assert_eq!(log10_result.shape().dims(), &[3]);

        let log2_result = tensor32.log2_complex().unwrap();
        assert_eq!(log2_result.shape().dims(), &[3]);

        // Test Complex64 operations
        let complex64_data = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(0.5, -1.0),
        ];
        let tensor64 = Tensor::from_data(complex64_data.clone(), vec![3], DeviceType::Cpu).unwrap();

        // Test complex conjugate for Complex64
        let conj64_result = tensor64.conj().unwrap();
        let conj64_data = conj64_result.to_vec().unwrap();
        assert_eq!(conj64_data[0], Complex64::new(1.0, -2.0));
        assert_eq!(conj64_data[1], Complex64::new(3.0, -4.0));
        assert_eq!(conj64_data[2], Complex64::new(0.5, 1.0));

        // Test that all operations maintain correct tensor shapes
        let asin64_result = tensor64.asin_complex().unwrap();
        assert_eq!(asin64_result.shape().dims(), &[3]);

        let acos64_result = tensor64.acos_complex().unwrap();
        assert_eq!(acos64_result.shape().dims(), &[3]);

        let atan64_result = tensor64.atan_complex().unwrap();
        assert_eq!(atan64_result.shape().dims(), &[3]);

        let asinh64_result = tensor64.asinh_complex().unwrap();
        assert_eq!(asinh64_result.shape().dims(), &[3]);

        let acosh64_result = tensor64.acosh_complex().unwrap();
        assert_eq!(acosh64_result.shape().dims(), &[3]);

        let atanh64_result = tensor64.atanh_complex().unwrap();
        assert_eq!(atanh64_result.shape().dims(), &[3]);

        let log10_64_result = tensor64.log10_complex().unwrap();
        assert_eq!(log10_64_result.shape().dims(), &[3]);

        let log2_64_result = tensor64.log2_complex().unwrap();
        assert_eq!(log2_64_result.shape().dims(), &[3]);
    }

    #[test]
    fn test_complex_operations_gradient_tracking() {
        use torsh_core::dtype::Complex32;
        use torsh_core::device::DeviceType;

        // Create tensor with gradient tracking enabled
        let complex_data = vec![Complex32::new(1.0, 2.0)];
        let mut tensor = Tensor::from_data(complex_data, vec![1], DeviceType::Cpu).unwrap();
        tensor.requires_grad = true;

        // Test that all new complex operations preserve gradient tracking
        let conj_result = tensor.conj().unwrap();
        assert!(conj_result.requires_grad);

        let asin_result = tensor.asin_complex().unwrap();
        assert!(asin_result.requires_grad);

        let acos_result = tensor.acos_complex().unwrap();
        assert!(acos_result.requires_grad);

        let atan_result = tensor.atan_complex().unwrap();
        assert!(atan_result.requires_grad);

        let asinh_result = tensor.asinh_complex().unwrap();
        assert!(asinh_result.requires_grad);

        let acosh_result = tensor.acosh_complex().unwrap();
        assert!(acosh_result.requires_grad);

        let atanh_result = tensor.atanh_complex().unwrap();
        assert!(atanh_result.requires_grad);

        let log10_result = tensor.log10_complex().unwrap();
        assert!(log10_result.requires_grad);

        let log2_result = tensor.log2_complex().unwrap();
        assert!(log2_result.requires_grad);
    }

    #[test]
    fn test_masked_operations() {
        use torsh_core::device::DeviceType;
        
        // Test masked_fill
        let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let mask = Tensor::from_data(vec![true, false, true, false], vec![4], DeviceType::Cpu).unwrap();
        
        let result = tensor.masked_fill(&mask, 99.0).unwrap();
        let result_data = result.data().unwrap();
        assert_eq!(result_data.as_slice(), &[99.0, 2.0, 99.0, 4.0]);

        // Test masked_where
        let other = Tensor::from_data(vec![10.0, 20.0, 30.0, 40.0], vec![4], DeviceType::Cpu).unwrap();
        let result2 = tensor.masked_where(&mask, &other).unwrap();
        let result2_data = result2.data().unwrap();
        assert_eq!(result2_data.as_slice(), &[10.0, 2.0, 30.0, 4.0]);

        // Test masked_scatter
        let source = Tensor::from_data(vec![100.0, 200.0], vec![2], DeviceType::Cpu).unwrap();
        let result3 = tensor.masked_scatter(&mask, &source).unwrap();
        let result3_data = result3.data().unwrap();
        assert_eq!(result3_data.as_slice(), &[100.0, 2.0, 200.0, 4.0]);

        // Test masked_fill_ (in-place)
        let mut tensor_mut = tensor.clone();
        tensor_mut.masked_fill_(&mask, 999.0).unwrap();
        let result4_data = tensor_mut.data().unwrap();
        assert_eq!(result4_data.as_slice(), &[999.0, 2.0, 999.0, 4.0]);
    }

    #[test]
    fn test_masked_operations_error_handling() {
        use torsh_core::device::DeviceType;
        
        let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let wrong_mask = Tensor::from_data(vec![true, false], vec![2], DeviceType::Cpu).unwrap();
        
        // Test shape mismatch error
        assert!(tensor.masked_fill(&wrong_mask, 1.0).is_err());
        
        // Test insufficient source data for scatter
        let mask = Tensor::from_data(vec![true, true, false], vec![3], DeviceType::Cpu).unwrap();
        let small_source = Tensor::from_data(vec![99.0], vec![1], DeviceType::Cpu).unwrap();
        assert!(tensor.masked_scatter(&mask, &small_source).is_err());
    }

    #[test]
    fn test_linear_algebra_operations_f32() {
        use torsh_core::device::DeviceType;
        
        // Test SVD
        let matrix_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let matrix = Tensor::from_data(matrix_data, vec![2, 2], DeviceType::Cpu).unwrap();
        
        let (u, s, v) = matrix.svd().unwrap();
        assert_eq!(u.shape().dims(), &[2, 2]);
        assert_eq!(s.shape().dims(), &[2]);
        assert_eq!(v.shape().dims(), &[2, 2]);
        
        // Test eigenvalue decomposition
        let (eigenvals, eigenvecs) = matrix.eig().unwrap();
        assert_eq!(eigenvals.shape().dims(), &[2]);
        assert_eq!(eigenvecs.shape().dims(), &[2, 2]);
        
        // Test QR decomposition
        let (q, r) = matrix.qr().unwrap();
        assert_eq!(q.shape().dims(), &[2, 2]);
        assert_eq!(r.shape().dims(), &[2, 2]);
        
        // Test Cholesky decomposition
        let l = matrix.cholesky().unwrap();
        assert_eq!(l.shape().dims(), &[2, 2]);
        
        // Test matrix inverse
        let inv = matrix.inverse().unwrap();
        assert_eq!(inv.shape().dims(), &[2, 2]);
        
        // Test determinant
        let det = matrix.det().unwrap();
        assert!(det.is_finite());
        
        // Test rank
        let rank = matrix.matrix_rank().unwrap();
        assert_eq!(rank, 2);
        
        // Test trace
        let trace = matrix.trace().unwrap();
        assert_eq!(trace, 5.0); // 1.0 + 4.0 = 5.0
    }


    #[test]
    fn test_linear_algebra_error_handling() {
        use torsh_core::device::DeviceType;
        
        // Test with non-square matrix for operations that require square matrices
        let rect_matrix = Tensor::from_data(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], DeviceType::Cpu).unwrap();
        
        // These should fail for non-square matrices
        assert!(rect_matrix.eig().is_err());
        assert!(rect_matrix.cholesky().is_err());
        assert!(rect_matrix.inverse().is_err());
        assert!(rect_matrix.det().is_err());
        assert!(rect_matrix.trace().is_err());
        
        // SVD and QR should work for rectangular matrices
        assert!(rect_matrix.svd().is_ok());
        assert!(rect_matrix.qr().is_ok());
        assert!(rect_matrix.matrix_rank().is_ok());
        
        // Test with 1D tensor (should fail for all)
        let vector = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        
        assert!(vector.svd().is_err());
        assert!(vector.eig().is_err());
        assert!(vector.qr().is_err());
        assert!(vector.cholesky().is_err());
        assert!(vector.inverse().is_err());
        assert!(vector.det().is_err());
        assert!(vector.trace().is_err());
        assert!(vector.matrix_rank().is_err());
    }

    #[test]
    fn test_trace_computation() {
        use torsh_core::device::DeviceType;
        
        // Test trace calculation with known values
        let matrix_data = vec![
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ];
        let matrix = Tensor::from_data(matrix_data, vec![3, 3], DeviceType::Cpu).unwrap();
        
        let trace = matrix.trace().unwrap();
        assert_eq!(trace, 15.0); // 1.0 + 5.0 + 9.0 = 15.0
    }
}
