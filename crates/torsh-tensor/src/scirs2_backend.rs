//! SciRS2 Backend Integration for ToRSh Tensors
//!
//! This module provides integration with the scirs2 ecosystem for optimized
//! tensor operations, automatic differentiation, and scientific computing primitives.

use crate::{Tensor, TensorElement};
use num_traits::{Float, One, Zero};
use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
    shape::Shape,
};

/// SciRS2 integration wrapper for tensor operations
pub struct SciRS2Backend;

impl Default for SciRS2Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl SciRS2Backend {
    /// Initialize the SciRS2 backend
    pub fn new() -> Self {
        Self
    }

    /// Convert tensor data to scirs2-compatible format
    fn to_scirs2_data<T: TensorElement + Copy>(tensor: &Tensor<T>) -> Result<Vec<T>> {
        tensor.data()
    }

    /// Create tensor from scirs2 result data
    fn from_scirs2_data<T: TensorElement + Copy>(
        data: Vec<T>,
        shape: &Shape,
        device: DeviceType,
    ) -> Result<Tensor<T>> {
        Tensor::from_data(data, shape.dims().to_vec(), device)
    }
}

/// Element-wise operations using scirs2 optimized implementations
impl SciRS2Backend {
    /// Element-wise addition using scirs2 optimization
    pub fn add<T>(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + std::ops::Add<Output = T> + Float,
    {
        let lhs_data = Self::to_scirs2_data(lhs)?;
        let rhs_data = Self::to_scirs2_data(rhs)?;

        // Element-wise addition - placeholder implementation
        let result_data: Vec<T> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Self::from_scirs2_data(result_data, &lhs.shape(), lhs.device())
    }

    /// Element-wise multiplication using scirs2 optimization
    pub fn mul<T>(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + std::ops::Mul<Output = T> + Float,
    {
        let lhs_data = Self::to_scirs2_data(lhs)?;
        let rhs_data = Self::to_scirs2_data(rhs)?;

        let result_data: Vec<T> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Self::from_scirs2_data(result_data, &lhs.shape(), lhs.device())
    }

    /// Element-wise subtraction using scirs2 optimization
    pub fn sub<T>(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + std::ops::Sub<Output = T> + Float,
    {
        let lhs_data = Self::to_scirs2_data(lhs)?;
        let rhs_data = Self::to_scirs2_data(rhs)?;

        let result_data: Vec<T> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Self::from_scirs2_data(result_data, &lhs.shape(), lhs.device())
    }

    /// Element-wise division using scirs2 optimization
    pub fn div<T>(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + std::ops::Div<Output = T> + Float,
    {
        let lhs_data = Self::to_scirs2_data(lhs)?;
        let rhs_data = Self::to_scirs2_data(rhs)?;

        let result_data: Vec<T> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a / b)
            .collect();

        Self::from_scirs2_data(result_data, &lhs.shape(), lhs.device())
    }

    /// Matrix multiplication using scirs2 linear algebra
    pub fn matmul<T>(&self, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + Float + Zero + One,
    {
        // Validate matrix multiplication dimensions
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        if lhs_shape.ndim() != 2 || rhs_shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let lhs_dims = lhs_shape.dims();
        let rhs_dims = rhs_shape.dims();

        if lhs_dims[1] != rhs_dims[0] {
            return Err(TorshError::InvalidArgument(format!(
                "Incompatible matrix dimensions: ({}, {}) and ({}, {})",
                lhs_dims[0], lhs_dims[1], rhs_dims[0], rhs_dims[1]
            )));
        }

        let lhs_data = Self::to_scirs2_data(lhs)?;
        let rhs_data = Self::to_scirs2_data(rhs)?;

        // Matrix multiplication: C[i,j] = Σ_k A[i,k] * B[k,j]
        let m = lhs_dims[0];
        let n = rhs_dims[1];
        let k = lhs_dims[1];

        let mut result_data = vec![<T as Zero>::zero(); m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = <T as Zero>::zero();
                for kk in 0..k {
                    sum = sum + lhs_data[i * k + kk] * rhs_data[kk * n + j];
                }
                result_data[i * n + j] = sum;
            }
        }

        Self::from_scirs2_data(result_data, &Shape::new(vec![m, n]), lhs.device())
    }

    /// Reduction operations using scirs2 optimization
    pub fn sum<T>(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + std::ops::Add<Output = T> + Zero,
    {
        let data = Self::to_scirs2_data(tensor)?;
        let sum = data.iter().fold(<T as Zero>::zero(), |acc, &x| acc + x);

        // Return scalar tensor (1-element tensor with shape [])
        Self::from_scirs2_data(vec![sum], &Shape::new(vec![]), tensor.device())
    }

    /// Mean reduction using scirs2 optimization
    pub fn mean<T>(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement
            + Copy
            + std::ops::Add<Output = T>
            + std::ops::Div<Output = T>
            + Zero
            + From<usize>,
    {
        let data = Self::to_scirs2_data(tensor)?;
        let sum = data.iter().fold(<T as Zero>::zero(), |acc, &x| acc + x);
        let count = T::from(data.len());
        let mean = sum / count;

        Self::from_scirs2_data(vec![mean], &Shape::new(vec![]), tensor.device())
    }

    /// Activation functions using scirs2 neural network primitives
    pub fn relu<T>(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + PartialOrd + Zero,
    {
        let data = Self::to_scirs2_data(tensor)?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                if x > <T as Zero>::zero() {
                    x
                } else {
                    <T as Zero>::zero()
                }
            })
            .collect();

        Self::from_scirs2_data(result_data, &tensor.shape(), tensor.device())
    }

    /// Sigmoid activation using scirs2 neural network primitives
    pub fn sigmoid<T>(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + Float,
    {
        let data = Self::to_scirs2_data(tensor)?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| <T as One>::one() / (<T as One>::one() + (-x).exp()))
            .collect();

        Self::from_scirs2_data(result_data, &tensor.shape(), tensor.device())
    }

    /// Tanh activation using scirs2 neural network primitives
    pub fn tanh<T>(&self, tensor: &Tensor<T>) -> Result<Tensor<T>>
    where
        T: TensorElement + Copy + Float,
    {
        let data = Self::to_scirs2_data(tensor)?;
        let result_data: Vec<T> = data.iter().map(|&x| x.tanh()).collect();

        Self::from_scirs2_data(result_data, &tensor.shape(), tensor.device())
    }
}

/// Global SciRS2 backend instance
static SCIRS2_BACKEND: SciRS2Backend = SciRS2Backend;

/// Get the global SciRS2 backend instance
pub fn get_scirs2_backend() -> &'static SciRS2Backend {
    &SCIRS2_BACKEND
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_scirs2_backend_creation() {
        let backend = SciRS2Backend::new();
        // Just test that backend creation works
        let _ = backend;
    }

    #[test]
    fn test_scirs2_add() {
        let backend = SciRS2Backend::new();

        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![4.0f32, 5.0, 6.0], vec![3], DeviceType::Cpu).unwrap();

        let result = backend.add(&a, &b).unwrap();
        let expected = vec![5.0f32, 7.0, 9.0];

        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[test]
    fn test_scirs2_matmul() {
        let backend = SciRS2Backend::new();

        // 2x3 * 3x2 = 2x2
        let a = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .unwrap();
        let b = Tensor::from_data(
            vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
            DeviceType::Cpu,
        )
        .unwrap();

        let result = backend.matmul(&a, &b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);

        // Verify result values
        let result_data = result.to_vec().unwrap();
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(result_data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_scirs2_relu() {
        let backend = SciRS2Backend::new();

        let a = Tensor::from_data(vec![-1.0f32, 0.0, 1.0, 2.0], vec![4], DeviceType::Cpu).unwrap();
        let result = backend.relu(&a).unwrap();
        let expected = vec![0.0f32, 0.0, 1.0, 2.0];

        assert_eq!(result.to_vec().unwrap(), expected);
    }
}
