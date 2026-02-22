//! Core traits for automatic differentiation tensors
//!
//! This module defines the fundamental traits that tensors must implement to participate
//! in automatic differentiation, providing a unified interface for gradient computation
//! across different tensor implementations.
//!
//! # Features
//!
//! - **AutogradTensor trait**: Core interface for differentiable tensors
//! - **Device abstraction**: Platform-independent tensor operations
//! - **Shape management**: Tensor dimensionality and layout handling
//! - **Gradient initialization**: Support for ones_like and zeros_like operations

use torsh_core::shape::Shape;

/// Trait for tensors that can participate in automatic differentiation
pub trait AutogradTensor<T: torsh_core::dtype::TensorElement>: Send + Sync {
    /// Get the shape of the tensor
    fn shape(&self) -> Shape;

    /// Check if this tensor requires gradients
    fn requires_grad(&self) -> bool;

    /// Get read access to the tensor data
    fn data(&self) -> Box<dyn std::ops::Deref<Target = [T]> + '_>;

    /// Clone the tensor
    fn clone_tensor(&self) -> Box<dyn AutogradTensor<T>>;

    /// Convert tensor data to Vec for gradient computation
    fn to_vec(&self) -> Vec<T>;

    /// Get the device this tensor is stored on
    fn device(&self) -> &dyn torsh_core::Device;

    /// Create a ones-like tensor for gradient initialization
    fn ones_like(&self) -> Box<dyn AutogradTensor<T>>;

    /// Create a zeros-like tensor for gradient initialization
    fn zeros_like(&self) -> Box<dyn AutogradTensor<T>>;
}

/// Additional trait for tensors that support gradient accumulation
pub trait GradientAccumulation<T: torsh_core::dtype::TensorElement>: AutogradTensor<T> {
    /// Accumulate gradient into this tensor
    fn accumulate_grad(
        &mut self,
        gradient: &dyn AutogradTensor<T>,
    ) -> torsh_core::error::Result<()>;

    /// Set the gradient for this tensor
    fn set_grad(&mut self, gradient: Box<dyn AutogradTensor<T>>) -> torsh_core::error::Result<()>;

    /// Get the current gradient for this tensor
    fn grad(&self) -> Option<&dyn AutogradTensor<T>>;

    /// Clear the gradient for this tensor
    fn zero_grad(&mut self);
}

/// Trait for tensors that can be used in backward pass computation
pub trait BackwardTensor<T: torsh_core::dtype::TensorElement>: AutogradTensor<T> {
    /// Compute gradients with respect to this tensor
    fn backward(&self, gradient: Option<&dyn AutogradTensor<T>>) -> torsh_core::error::Result<()>;

    /// Check if this tensor is a leaf node (no computation history)
    fn is_leaf(&self) -> bool;

    /// Get the operation that created this tensor (for non-leaf tensors)
    fn grad_fn(&self) -> Option<&str>;
}

/// Helper trait for creating autograd tensors from data
pub trait AutogradTensorFactory<T: torsh_core::dtype::TensorElement> {
    /// Create a new autograd tensor from raw data
    fn from_data(
        data: Vec<T>,
        shape: Vec<usize>,
        requires_grad: bool,
        device: torsh_core::device::DeviceType,
    ) -> torsh_core::error::Result<Box<dyn AutogradTensor<T>>>;

    /// Create a zeros tensor with specified shape
    fn zeros(
        shape: &[usize],
        requires_grad: bool,
        device: torsh_core::device::DeviceType,
    ) -> torsh_core::error::Result<Box<dyn AutogradTensor<T>>>;

    /// Create a ones tensor with specified shape
    fn ones(
        shape: &[usize],
        requires_grad: bool,
        device: torsh_core::device::DeviceType,
    ) -> torsh_core::error::Result<Box<dyn AutogradTensor<T>>>;

    /// Create a random tensor with specified shape
    fn randn(
        shape: &[usize],
        requires_grad: bool,
        device: torsh_core::device::DeviceType,
    ) -> torsh_core::error::Result<Box<dyn AutogradTensor<T>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    use torsh_core::dtype::TensorElement;

    // Mock implementation for testing
    struct MockTensor<T> {
        data: Vec<T>,
        shape: Shape,
        requires_grad: bool,
    }

    impl<T: TensorElement + Clone> AutogradTensor<T> for MockTensor<T> {
        fn shape(&self) -> Shape {
            self.shape.clone()
        }

        fn requires_grad(&self) -> bool {
            self.requires_grad
        }

        fn data(&self) -> Box<dyn std::ops::Deref<Target = [T]> + '_> {
            Box::new(self.data.as_slice())
        }

        fn clone_tensor(&self) -> Box<dyn AutogradTensor<T>> {
            Box::new(MockTensor {
                data: self.data.clone(),
                shape: self.shape.clone(),
                requires_grad: self.requires_grad,
            })
        }

        fn to_vec(&self) -> Vec<T> {
            self.data.clone()
        }

        fn device(&self) -> &dyn torsh_core::Device {
            use std::sync::LazyLock;
            static CPU_DEVICE: LazyLock<torsh_core::device::CpuDevice> =
                LazyLock::new(|| torsh_core::device::CpuDevice::new());
            &*CPU_DEVICE
        }

        fn ones_like(&self) -> Box<dyn AutogradTensor<T>> {
            let ones_data = vec![T::one(); self.data.len()];
            Box::new(MockTensor {
                data: ones_data,
                shape: self.shape.clone(),
                requires_grad: false,
            })
        }

        fn zeros_like(&self) -> Box<dyn AutogradTensor<T>> {
            let zeros_data = vec![T::zero(); self.data.len()];
            Box::new(MockTensor {
                data: zeros_data,
                shape: self.shape.clone(),
                requires_grad: false,
            })
        }
    }

    #[test]
    fn test_autograd_tensor_basic_operations() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::new(vec![2, 2]);
        let tensor = MockTensor {
            data: data.clone(),
            shape: shape.clone(),
            requires_grad: true,
        };

        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert!(tensor.requires_grad());
        assert_eq!(tensor.to_vec(), data);

        let ones = tensor.ones_like();
        assert_eq!(ones.to_vec(), vec![1.0, 1.0, 1.0, 1.0]);

        let zeros = tensor.zeros_like();
        assert_eq!(zeros.to_vec(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_autograd_tensor_cloning() {
        let data = vec![1.0f32, 2.0];
        let shape = Shape::new(vec![2]);
        let tensor = MockTensor {
            data: data.clone(),
            shape: shape.clone(),
            requires_grad: false,
        };

        let cloned = tensor.clone_tensor();
        assert_eq!(cloned.to_vec(), data);
        assert_eq!(cloned.shape().dims(), &[2]);
        assert!(!cloned.requires_grad());
    }
}
