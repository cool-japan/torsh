//! Tensor implementation for ToRSh with PyTorch-compatible API
//!
//! This crate provides a high-level tensor API that wraps scirs2's autograd
//! functionality with a familiar PyTorch-like interface.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod creation;
pub mod indexing;
pub mod ops;

use torsh_core::{
    device::DeviceType,
    dtype::{DType, FloatElement, TensorElement},
    error::{Result, TorshError},
    shape::Shape,
};

use std::sync::{Arc, Mutex};

/// Operation type for gradient computation
#[derive(Debug, Clone)]
enum Operation<T: TensorElement> {
    Leaf,
    Power {
        input: Arc<Tensor<T>>,
        exponent: f32,
    },
    Add {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    Mul {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
}

/// The main Tensor type for ToRSh
///
/// A simplified tensor implementation for now
#[derive(Clone)]
pub struct Tensor<T = f32>
where
    T: TensorElement,
{
    /// The data storage
    data: Arc<Mutex<Vec<T>>>,
    /// Shape of the tensor
    shape: Shape,
    /// Device information
    device: DeviceType,
    /// Whether gradients are required
    requires_grad: bool,
    /// Gradient tensor if computed
    grad: Arc<Mutex<Option<Tensor<T>>>>,
    /// Operation that created this tensor
    operation: Operation<T>,
}

impl<T: TensorElement> Tensor<T> {
    /// Create from raw data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>, device: DeviceType) -> Self {
        Self {
            data: Arc::new(Mutex::new(data)),
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(Mutex::new(None)),
            operation: Operation::Leaf,
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    /// Get the device
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Set whether this tensor requires gradients
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Get whether this tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set gradient tensor
    #[allow(dead_code)]
    pub(crate) fn set_grad(&self, grad: Tensor<T>) {
        let mut grad_lock = self.grad.lock().unwrap();
        *grad_lock = Some(grad);
    }

    /// Get mutable access to gradient
    pub fn grad_mut(&mut self) -> Option<&mut Self> {
        // For now, return None - would need to implement proper gradient access
        None
    }

    /// Convert to a different device
    pub fn to<D: Into<DeviceType>>(self, device: D) -> Result<Self> {
        let device = device.into();
        if device == self.device {
            return Ok(self);
        }

        // TODO: Implement actual device transfer when backends are ready
        Err(TorshError::UnsupportedOperation {
            op: "device transfer".to_string(),
            dtype: self.dtype().to_string(),
        })
    }

    /// Detach from the computation graph
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.requires_grad = false;
        detached
    }

    /// Get the gradient of this tensor (if it exists)
    pub fn grad(&self) -> Option<Self> {
        let grad_lock = self.grad.lock().unwrap();
        grad_lock.as_ref().cloned()
    }

    /// Check if this tensor has a gradient
    pub fn has_grad(&self) -> bool {
        let grad_lock = self.grad.lock().unwrap();
        grad_lock.is_some()
    }

    /// Zero the gradient
    pub fn zero_grad(&mut self) {
        let mut grad_lock = self.grad.lock().unwrap();
        *grad_lock = None;
    }

    /// Backward pass (compute gradients)
    pub fn backward(&self) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        if !self.requires_grad {
            return Err(TorshError::AutogradError(
                "Called backward on tensor that doesn't require grad".to_string(),
            ));
        }

        if self.shape().numel() != 1 {
            return Err(TorshError::AutogradError(
                "Gradient can only be computed for scalar outputs".to_string(),
            ));
        }

        // Create initial gradient of 1.0 for the output (chain rule starts here)
        let output_grad_data = vec![<T as TensorElement>::one()];
        let output_grad = Self::from_data(output_grad_data, vec![], self.device);

        // Start backpropagation
        self.backward_impl(&output_grad)?;

        Ok(())
    }

    /// Internal backward implementation
    fn backward_impl(&self, grad_output: &Self) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        match &self.operation {
            Operation::Leaf => {
                // Accumulate gradient for leaf nodes
                let mut grad_lock = self.grad.lock().unwrap();
                if let Some(existing_grad) = grad_lock.as_ref() {
                    // Add gradients if they exist
                    let new_grad = existing_grad.add(grad_output)?;
                    *grad_lock = Some(new_grad);
                } else {
                    // Set gradient if it doesn't exist
                    *grad_lock = Some(grad_output.clone());
                }
            }
            Operation::Power { input, exponent } => {
                if input.requires_grad {
                    // Compute gradient: d/dx(x^n) = n * x^(n-1)
                    let input_data = input.data.lock().unwrap();
                    let grad_data: Vec<T> = input_data
                        .iter()
                        .map(|&x| {
                            let exp_minus_one = *exponent - 1.0;
                            let exp_t = T::from_f64(*exponent as f64).unwrap();
                            let exp_minus_one_t = T::from_f64(exp_minus_one as f64).unwrap();
                            exp_t * x.powf(exp_minus_one_t)
                        })
                        .collect();

                    let input_grad =
                        Self::from_data(grad_data, input.shape().dims().to_vec(), input.device);
                    let final_grad = input_grad.mul(grad_output)?;

                    // Recursively compute gradients
                    input.backward_impl(&final_grad)?;
                }
            }
            Operation::Add { lhs, rhs } => {
                // Gradient flows through both operands unchanged
                if lhs.requires_grad {
                    lhs.backward_impl(grad_output)?;
                }
                if rhs.requires_grad {
                    rhs.backward_impl(grad_output)?;
                }
            }
            Operation::Mul { lhs, rhs } => {
                // Product rule: d/dx(f*g) = f'*g + f*g'
                if lhs.requires_grad {
                    let lhs_grad = (**rhs).mul(grad_output)?;
                    lhs.backward_impl(&lhs_grad)?;
                }
                if rhs.requires_grad {
                    let rhs_grad = (**lhs).mul(grad_output)?;
                    rhs.backward_impl(&rhs_grad)?;
                }
            }
        }

        Ok(())
    }

    /// Get size of a specific dimension
    pub fn size(&self, dim: i32) -> Result<usize> {
        self.shape().size(dim)
    }

    /// Reshape the tensor
    pub fn view(&self, shape: &[i32]) -> Result<Self> {
        let new_shape: Result<Vec<usize>> = shape
            .iter()
            .map(|&d| {
                if d == -1 {
                    // Infer dimension
                    let known_product: usize = shape
                        .iter()
                        .filter(|&&x| x != -1)
                        .map(|&x| x as usize)
                        .product();

                    if known_product == 0 {
                        return Err(TorshError::InvalidShape(
                            "Cannot infer dimension with zero-sized dimensions".to_string(),
                        ));
                    }

                    let total = self.numel();
                    if total % known_product != 0 {
                        return Err(TorshError::InvalidShape(
                            "Cannot infer dimension: size is not divisible".to_string(),
                        ));
                    }

                    Ok(total / known_product)
                } else if d < 0 {
                    Err(TorshError::InvalidShape(format!(
                        "Invalid dimension size: {}",
                        d
                    )))
                } else {
                    Ok(d as usize)
                }
            })
            .collect();

        let new_shape = new_shape?;
        let new_numel: usize = new_shape.iter().product();

        if new_numel != self.numel() {
            return Err(TorshError::InvalidShape(format!(
                "Shape {:?} is invalid for tensor of size {}",
                new_shape,
                self.numel()
            )));
        }

        // Create a new tensor with the same data but different shape
        let data = self.data.lock().unwrap();
        Ok(Self::from_data(data.clone(), new_shape, self.device))
    }

    /// Transpose dimensions
    pub fn transpose(&self, dim0: i32, dim1: i32) -> Result<Self> {
        let ndim = self.ndim() as i32;

        // Normalize negative dimensions
        let d0 = if dim0 < 0 { ndim + dim0 } else { dim0 } as usize;
        let d1 = if dim1 < 0 { ndim + dim1 } else { dim1 } as usize;

        if d0 >= self.ndim() || d1 >= self.ndim() {
            return Err(TorshError::InvalidShape(
                "Dimension out of range".to_string(),
            ));
        }

        if d0 == d1 {
            return Ok(self.clone());
        }

        // For now, implement simple 2D transpose
        if self.ndim() == 2 && d0 == 0 && d1 == 1 {
            self.transpose_2d()
        } else {
            // For higher dimensions, we'd need a more general implementation
            Err(TorshError::Other(
                "Transpose for >2D tensors not yet implemented".to_string(),
            ))
        }
    }

    fn transpose_2d(&self) -> Result<Self> {
        let shape_dims = self.shape().dims().to_vec();
        if shape_dims.len() != 2 {
            return Err(TorshError::InvalidShape(
                "transpose_2d requires 2D tensor".to_string(),
            ));
        }

        let rows = shape_dims[0];
        let cols = shape_dims[1];
        let data = self.data.lock().unwrap();

        let mut new_data = Vec::with_capacity(data.len());
        for j in 0..cols {
            for i in 0..rows {
                new_data.push(data[i * cols + j].clone());
            }
        }

        Ok(Self::from_data(new_data, vec![cols, rows], self.device))
    }

    /// Permute dimensions
    pub fn permute(&self, _dims: &[i32]) -> Result<Self> {
        // TODO: Implement permute using scirs2
        Ok(self.clone())
    }

    /// Squeeze dimensions of size 1
    pub fn squeeze(&self) -> Self {
        // TODO: Implement squeeze
        self.clone()
    }

    /// Unsqueeze (add dimension of size 1)
    pub fn unsqueeze(&self, dim: i32) -> Result<Self> {
        let ndim = self.ndim() as i32;

        // Normalize negative dimension
        let normalized_dim = if dim < 0 {
            // For negative dim, it's the position before inserting
            // e.g., -1 means insert before the last position
            (ndim + dim + 1) as usize
        } else {
            dim as usize
        };

        // Check if dimension is valid (can be 0 to ndim inclusive)
        if normalized_dim > self.ndim() {
            return Err(TorshError::InvalidShape(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        // Create new shape with dimension of size 1 inserted
        let shape = self.shape();
        let old_dims = shape.dims();
        let mut new_dims = Vec::with_capacity(old_dims.len() + 1);

        // Copy dimensions before the insertion point
        new_dims.extend_from_slice(&old_dims[..normalized_dim]);
        // Insert dimension of size 1
        new_dims.push(1);
        // Copy dimensions after the insertion point
        new_dims.extend_from_slice(&old_dims[normalized_dim..]);

        // Data remains the same, just the shape interpretation changes
        let data = self.data.lock().unwrap();
        let mut result = Self::from_data(data.clone(), new_dims, self.device);
        result.requires_grad = self.requires_grad;

        Ok(result)
    }
}

impl<T: FloatElement> Tensor<T> {
    /// Get a single item (for scalar tensors)
    pub fn item(&self) -> T {
        if self.numel() != 1 {
            panic!("Can only call item() on tensors with one element");
        }
        let data = self.data.lock().unwrap();
        data[0]
    }

    /// Create from a single scalar value
    pub fn from_scalar(value: T) -> Self {
        Self::from_data(vec![value], vec![], DeviceType::Cpu)
    }
}

impl<T: TensorElement> Tensor<T> {
    /// Convert to a Vec
    pub fn to_vec(&self) -> Vec<T> {
        let data = self.data.lock().unwrap();
        data.clone()
    }

    /// Create from vec with shape
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Self {
        Self::from_data(data, shape.to_vec(), DeviceType::Cpu)
    }
}

// Display implementation
impl<T: TensorElement> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, device={})",
            self.shape().dims(),
            self.dtype(),
            self.device
        )
    }
}

/// Tensor creation macro similar to PyTorch
#[macro_export]
macro_rules! tensor {
    // 1D array from bracketed values
    ([$($val:expr),+ $(,)?]) => {
        $crate::creation::tensor_1d(&[$($val),+])
    };

    // Multiple values without brackets (at least 2 values to avoid scalar conflict)
    ($val1:expr, $val2:expr $(, $val:expr)* $(,)?) => {
        $crate::creation::tensor_1d(&[$val1, $val2 $(, $val)*])
    };

    // Scalar (single value)
    ($val:expr) => {
        $crate::creation::tensor_scalar($val)
    };
}

#[macro_export]
macro_rules! tensor_2d {
    // 2D array
    ($([$($val:expr),+ $(,)?]),+ $(,)?) => {
        {
            let data = [$([$($val),+]),+];
            $crate::creation::tensor_2d_arrays(&data)
        }
    };
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::creation::*;
    pub use crate::{tensor, tensor_2d, Tensor};
    pub use torsh_core::prelude::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = tensor![1.0f32, 2.0f32, 3.0f32];
        assert_eq!(t.shape().dims(), &[3]);
        assert_eq!(t.dtype(), DType::F32);
    }

    #[test]
    fn test_2d_macro_expansion() {
        // Test 2D creation directly
        let data = [[1.0f32, 2.0], [3.0, 4.0]];
        let _t = crate::creation::tensor_2d_arrays(&data);

        // Test if the 2D macro pattern works
        let _t2 = tensor_2d![[1.0f32, 2.0], [3.0, 4.0]];
    }

    #[test]
    fn test_unsqueeze() {
        // Test 1D tensor
        let t1 = tensor![1.0f32, 2.0, 3.0];
        assert_eq!(t1.shape().dims(), &[3]);

        // Unsqueeze at dimension 0
        let t2 = t1.unsqueeze(0).unwrap();
        assert_eq!(t2.shape().dims(), &[1, 3]);

        // Unsqueeze at dimension 1
        let t3 = t1.unsqueeze(1).unwrap();
        assert_eq!(t3.shape().dims(), &[3, 1]);

        // Unsqueeze with negative dimension
        let t4 = t1.unsqueeze(-1).unwrap();
        assert_eq!(t4.shape().dims(), &[3, 1]);

        // Test 2D tensor
        let t5 = tensor_2d![[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(t5.shape().dims(), &[2, 2]);

        // Unsqueeze at dimension 0
        let t6 = t5.unsqueeze(0).unwrap();
        assert_eq!(t6.shape().dims(), &[1, 2, 2]);

        // Unsqueeze at dimension 2
        let t7 = t5.unsqueeze(2).unwrap();
        assert_eq!(t7.shape().dims(), &[2, 2, 1]);

        // Chain multiple unsqueezes (like in Conv2d bias)
        let bias = tensor![1.0f32, 2.0, 3.0, 4.0];
        let reshaped = bias
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(2)
            .unwrap()
            .unsqueeze(3)
            .unwrap();
        assert_eq!(reshaped.shape().dims(), &[1, 4, 1, 1]);
    }
}
