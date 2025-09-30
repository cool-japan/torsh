//! Tensor implementation for ToRSh with PyTorch-compatible API
//!
//! This crate provides a high-level tensor API that wraps scirs2's autograd
//! functionality with a familiar PyTorch-like interface.
//!
//! # Architecture
//!
//! The tensor implementation is organized into specialized modules:
//!
//! - [`storage`] - Storage management with automatic memory mapping optimization
//! - [`core_ops`] - Core tensor operations, creation, and gradient management
//! - [`shape_ops`] - Shape manipulation, views, and dimension operations
//! - [`data_ops`] - Data access, indexing, and manipulation operations
//! - [`advanced_ops`] - Advanced operations, reductions, and backend integration
//! - [`math_ops`] - Mathematical operations and functions
//! - [`complex_ops`] - Complex number operations and specialized autograd
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_tensor::Tensor;
//! use torsh_core::device::DeviceType;
//!
//! // Create a tensor
//! let data = vec![1.0f32, 2.0, 3.0, 4.0];
//! let tensor = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu)?;
//!
//! // Basic operations
//! let reshaped = tensor.view(&[4, 1])?;
//! let sum = tensor.sum()?;
//! let normalized = tensor / tensor.norm()?;
//!
//! // Enable gradients for autograd
//! let x = tensor.requires_grad_(true);
//! let y = x.pow(2.0)?;
//! y.backward()?;
//! # Ok::<(), torsh_core::error::TorshError>(())
//! ```
//!
//! # Features
//!
//! - **Automatic memory management**: Optimized storage with memory mapping for large tensors
//! - **Zero-copy views**: Efficient tensor views with shared underlying data
//! - **PyTorch compatibility**: Familiar API for easy migration from PyTorch
//! - **Automatic differentiation**: Full gradient computation support
//! - **Device abstraction**: CPU and GPU device support
//! - **Complex numbers**: Native complex tensor operations
//! - **SciRS2 integration**: Optimized backend operations for performance

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Core modules providing the tensor implementation
pub mod storage;
pub mod core_ops;
pub mod shape_ops;
pub mod data_ops;
pub mod advanced_ops;
pub mod math_ops;
pub mod complex_ops;

// Utility and integration modules
#[cfg(feature = "async")]
pub mod async_ops;
pub mod backend_integration;
pub mod bfloat16_ops;
pub mod broadcast;
pub mod cache_optimization;
pub mod convenience;
pub mod conv;
pub mod creation;
pub mod indexing;
pub mod memory_pool;
pub mod ops;
pub mod scirs2_backend;
pub mod scirs2_stats_integration;
pub mod stats;
pub mod tensor_views;
pub mod fft;
pub mod type_conversions;

#[cfg(feature = "custom-types")]
pub mod custom_data_types;

#[cfg(feature = "serialize")]
pub mod serialize;

// Re-export core types and traits
use torsh_core::{
    device::DeviceType,
    dtype::{DType, FloatElement, TensorElement},
    error::{Result, TorshError},
    shape::Shape,
};

// Re-export the main tensor type
pub use core_ops::{Tensor, Operation};

// Re-export storage types for advanced usage
pub use storage::{TensorStorage, MemoryMappedStorage};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

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

    // Single value (scalar)
    ($val:expr) => {
        $crate::creation::tensor_scalar($val)
    };
}

// Display implementation for Tensor
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

// Additional utility implementations
impl<T: TensorElement> Tensor<T> {
    /// Get the reference count of the underlying storage Arc (for testing CoW behavior)
    #[cfg(test)]
    pub fn data_ref_count(&self) -> usize {
        use std::sync::Arc;
        match &self.storage {
            TensorStorage::InMemory(data) => Arc::strong_count(data),
            TensorStorage::MemoryMapped(storage) => Arc::strong_count(storage),
        }
    }

    /// Create from vec with shape (convenience method)
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self>
    where
        T: Copy,
    {
        Self::from_data(data, shape.to_vec(), DeviceType::Cpu)
    }
}

// Implement AutogradTensor trait for integration with torsh-autograd
impl<T: TensorElement> torsh_autograd::AutogradTensor<T> for Tensor<T> {
    fn shape(&self) -> Shape {
        self.shape()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad()
    }

    fn data(&self) -> Box<dyn std::ops::Deref<Target = [T]> + '_> {
        // Return a boxed vector that can be dereferenced as a slice
        Box::new(self.to_vec().unwrap_or_default())
    }

    fn clone_tensor(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>> {
        Box::new(self.clone())
    }

    fn to_vec(&self) -> Vec<T>
    where
        T: Copy,
    {
        self.to_vec().unwrap_or_default()
    }

    fn device(&self) -> &dyn torsh_core::Device {
        match &self.device {
            DeviceType::Cpu => &torsh_core::device::CpuDevice,
            DeviceType::Cuda(_) => &torsh_core::device::CpuDevice, // TODO: Return proper CUDA device
            _ => &torsh_core::device::CpuDevice,
        }
    }

    fn ones_like(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>>
    where
        T: Copy,
    {
        Box::new(self.ones_like().unwrap_or_else(|_| self.clone()))
    }

    fn zeros_like(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>>
    where
        T: Copy,
    {
        Box::new(self.zeros_like().unwrap_or_else(|_| self.clone()))
    }
}

// Re-export commonly used functions and types for convenience
pub mod prelude {
    pub use crate::{Tensor, TensorStorage};
    pub use crate::core_ops::Operation;
    pub use torsh_core::{
        device::DeviceType,
        dtype::{DType, TensorElement, FloatElement},
        error::{Result, TorshError},
        shape::Shape,
    };
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_tensor_creation_and_basic_ops() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu).unwrap();

        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_tensor_reshape_and_view() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).unwrap();

        let reshaped = tensor.view(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 2]);

        let slice = tensor.slice_tensor(0, 0, 1).unwrap();
        assert_eq!(slice.shape().dims(), &[1, 3]);
    }

    #[test]
    fn test_tensor_math_operations() {
        let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(vec![4.0f32, 5.0, 6.0], vec![3], DeviceType::Cpu).unwrap();

        let sum = a.add(&b).unwrap();
        assert_eq!(sum.data().unwrap(), vec![5.0, 7.0, 9.0]);

        let product = a.mul(&b).unwrap();
        assert_eq!(product.data().unwrap(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_tensor_advanced_operations() {
        let data = vec![1.0f32, 4.0, 9.0, 16.0];
        let tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu).unwrap();

        let sqrt_result = tensor.sqrt().unwrap();
        assert_eq!(sqrt_result.data().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let norm = tensor.norm().unwrap();
        assert!(norm.item().unwrap() > 0.0);
    }

    #[test]
    fn test_tensor_data_operations() {
        let mut tensor = Tensor::<f32>::zeros(&[2, 3], DeviceType::Cpu).unwrap();

        tensor.fill_(5.0).unwrap();
        assert_eq!(tensor.get_item(&[0, 0]).unwrap(), 5.0);

        let indices = Tensor::from_data(vec![0i64, 2], vec![2], DeviceType::Cpu).unwrap();
        let src = Tensor::from_data(vec![10.0f32, 20.0], vec![2], DeviceType::Cpu).unwrap();

        let data_1d = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor_1d = Tensor::from_data(data_1d, vec![5], DeviceType::Cpu).unwrap();
        let gathered = tensor_1d.gather(0, &indices).unwrap();
        assert_eq!(gathered.data().unwrap(), vec![1.0, 3.0]);
    }

    #[test]
    fn test_tensor_storage_optimization() {
        // Small tensor should use in-memory storage
        let small = Tensor::<f32>::zeros(&[10], DeviceType::Cpu).unwrap();
        assert_eq!(small.storage_type(), "in_memory");

        // Test copy-on-write behavior
        let tensor1 = Tensor::<f32>::ones(&[5], DeviceType::Cpu).unwrap();
        let tensor2 = tensor1.clone();
        assert!(tensor1.shares_storage(&tensor2));
    }

    #[test]
    fn test_gradient_operations() {
        let tensor = Tensor::<f32>::ones(&[2, 2], DeviceType::Cpu)
            .unwrap()
            .requires_grad_(true);

        assert!(tensor.requires_grad());
        assert!(!tensor.has_grad());

        let detached = tensor.detach();
        assert!(!detached.requires_grad());
    }
}