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
//! let norm_val = tensor.norm()?.item()?;
//! let normalized = tensor.div_scalar(norm_val)?;
//!
//! // Enable gradients for autograd
//! let x = tensor.requires_grad_(true);
//! let y = x.pow(2.0)?;
//! let loss = y.sum()?;  // Create scalar for backward pass
//! loss.backward()?;
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
pub mod adaptive_auto_tuner;
pub mod advanced_ops;
pub mod advanced_simd_ops;
pub mod algorithmic_optimizations;
pub mod complex_ops;
pub mod comprehensive_integration_tests;
pub mod computation_graph;
pub mod core_ops;
pub mod cross_platform_validator;
pub mod data_ops;
pub mod expression_optimizer;
pub mod expression_templates;
pub mod hardware_accelerators;
pub mod manipulation;
pub mod math_ops;
pub mod memory_optimization;
pub mod optimization_cli;
pub mod shape_ops;
pub mod storage;
pub mod ultimate_integration_optimizer;
pub mod ultra_performance_profiler;

// Utility and integration modules
#[cfg(feature = "async")]
pub mod async_ops;
pub mod auto_batching;
pub mod backend_integration;
pub mod bfloat16_ops;
pub mod broadcast;
pub mod cache_optimization;
pub mod conv;
pub mod convenience;
pub mod creation;
pub mod custom_dtype;
pub mod custom_ops;
pub mod indexing;
pub mod lazy_loading;
// pub mod lazy_ops; // Temporarily disabled due to complex trait bounds - using fluent API instead
pub mod lockfree_cache;
pub mod memory_pool;
#[cfg(feature = "memory-profiling")]
pub mod memory_profiler;
pub mod nan_inf_detection;
#[cfg(feature = "operation-logging")]
pub mod operation_logging;
// pub mod ops; // Disabled due to duplicate definitions with core modules (all, any, sum, mean, matmul, cat, etc.)
pub mod fft;
pub mod scirs2_backend;
pub mod scirs2_stats_integration;
pub mod shape_inference_debugger;
pub mod sparse;
pub mod stats;
pub mod tensor_comprehension;
pub mod tensor_tracker;
pub mod tensor_utils;
pub mod tensor_view; // Zero-copy tensor views (CRITICAL #1)
pub mod tensor_views;
pub mod type_conversions;

// TODO: Implement custom data types module
// #[cfg(feature = "custom-types")]
// pub mod custom_data_types;

#[cfg(feature = "serialize")]
pub mod serialize;

// Re-export core types and traits
use torsh_core::{
    device::DeviceType,
    dtype::{FloatElement, TensorElement},
    error::Result,
};

// Re-export the main tensor type
pub use core_ops::{Operation, Tensor};

// Re-export convenience methods
pub use convenience::{FluentTensor, TensorConvenience, TensorFluentExt};

// Re-export lazy evaluation functionality (temporarily disabled)
// pub use lazy_ops::{LazyTensor, TensorLazyExt};

// Re-export sparse tensor functionality (COO, CSR, CSC formats)
pub use sparse::{SparseCSC, SparseCSR, SparseTensor};

// Re-export custom operation functionality
pub use custom_ops::{
    global_registry, CustomOperation, CustomOperationRegistry, OperationMetadata, OperationParams,
    TensorCustomOps,
};

// Re-export storage types for advanced usage
pub use storage::{MemoryMappedStorage, TensorStorage};

// Re-export zero-copy view types (CRITICAL #1)
pub use tensor_view::{TensorView, TensorViewMut};

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

/// 2D tensor creation macro
#[macro_export]
macro_rules! tensor_2d {
    ([$($row:expr),+ $(,)?]) => {{
        let rows: Vec<Vec<_>> = vec![$($row.to_vec()),+];
        let row_refs: Vec<&[_]> = rows.iter().map(|row| row.as_slice()).collect();
        $crate::creation::tensor_2d(&row_refs)
    }};
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
            #[cfg(feature = "simd")]
            TensorStorage::Aligned(data) => Arc::strong_count(data),
            #[cfg(feature = "simd")]
            TensorStorage::SimdOptimized(storage) => Arc::strong_count(storage),
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

// TODO: Conditional AutogradTensor trait implementation - torsh-autograd not yet available
// #[cfg(feature = "autograd")]
// impl<T: TensorElement> torsh_autograd::AutogradTensor<T> for Tensor<T> {
//     fn shape(&self) -> Shape {
//         self.shape()
//     }
//
//     fn requires_grad(&self) -> bool {
//         self.requires_grad()
//     }
//
//     fn data(&self) -> Box<dyn std::ops::Deref<Target = [T]> + '_> {
//         // Return a boxed vector that can be dereferenced as a slice
//         Box::new(self.to_vec().unwrap_or_default())
//     }
//
//     fn clone_tensor(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>> {
//         Box::new(self.clone())
//     }
//
//     fn to_vec(&self) -> Vec<T>
//     where
//         T: Copy,
//     {
//         self.to_vec().unwrap_or_default()
//     }
//
//     fn device(&self) -> &dyn torsh_core::Device {
//         match &self.device {
//             DeviceType::Cpu => {
//                 static CPU_DEVICE: torsh_core::device::CpuDevice =
//                     torsh_core::device::CpuDevice::new();
//                 &CPU_DEVICE
//             }
//             DeviceType::Cuda(_) => {
//                 static CPU_DEVICE: torsh_core::device::CpuDevice =
//                     torsh_core::device::CpuDevice::new();
//                 &CPU_DEVICE // TODO: Return proper CUDA device
//             }
//             _ => {
//                 static CPU_DEVICE: torsh_core::device::CpuDevice =
//                     torsh_core::device::CpuDevice::new();
//                 &CPU_DEVICE
//             }
//         }
//     }
//
//     fn ones_like(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>>
//     where
//         T: Copy,
//     {
//         Box::new(self.ones_like().unwrap_or_else(|_| self.clone()))
//     }
//
//     fn zeros_like(&self) -> Box<dyn torsh_autograd::AutogradTensor<T>>
//     where
//         T: Copy,
//     {
//         Box::new(self.zeros_like().unwrap_or_else(|_| self.clone()))
//     }
// }

// Re-export commonly used functions and types for convenience
pub mod prelude {
    pub use crate::advanced_simd_ops::{
        AdvancedSimdOps, ReductionType, SimdConfig, SimdPerformanceInfo,
    };
    pub use crate::algorithmic_optimizations::{
        AlgorithmConfig, AlgorithmPerformanceStats, AlgorithmicOptimizer, SchedulingStrategy,
    };
    pub use crate::comprehensive_integration_tests::{
        run_comprehensive_integration_tests, ComprehensiveIntegrationTestSuite,
        ComprehensiveTestReport, IntegrationAnalysis, IntegrationTestConfig, PerformanceAnalysis,
        StabilityAnalysis, TestCategory,
    };
    pub use crate::core_ops::Operation;
    pub use crate::creation::{eye, ones, rand, randn, zeros};
    pub use crate::cross_platform_validator::{
        CpuArchitecture, CrossPlatformReport, CrossPlatformValidator, GpuVendor,
        HardwareDetectionReport, HardwareDetector, OptimizationConfig, OptimizationReport,
        Platform, PlatformOptimizer, ValidationConfig, ValidationFramework, ValidationReport,
    };
    pub use crate::expression_optimizer::{
        ExpressionGraph, ExpressionNode, ExpressionOptimizer, NodeId, OperationType,
        OptimizationStats, OptimizationStrategy, OptimizerConfig, TensorExpressionOps,
    };
    pub use crate::hardware_accelerators::{
        AccelerationWorkload, ComplexityLevel, CpuAccelerationMetrics, CpuAcceleratorEngine,
        GpuAccelerationMetrics, GpuAcceleratorEngine, HardwareAcceleratorReport,
        HardwareAcceleratorSystem, MemoryAccelerationMetrics, MemoryAcceleratorEngine,
        NetworkAccelerationMetrics, OptimizationCoordinator, SpecializedAcceleratorEngine,
        WorkloadType,
    };
    pub use crate::memory_optimization::{
        AdvancedMemoryPool, AggregateMemoryStats, DefragmentationReport, GlobalMemoryOptimizer,
        MemoryConfig, MemoryStats,
    };
    pub use crate::optimization_cli::{
        run_cli_command, run_optimization_cli, CLICommand, CLIConfig, OptimizationCLI,
        OptimizationLevel, OptimizationType,
    };
    pub use crate::ultimate_integration_optimizer::{
        CrossLayerSynergyGains, EfficiencyImprovements, EnergyEfficiencyImprovements,
        GlobalPerformanceCache, IntelligentLearningSystem, LayerSpecificImprovements,
        OptimizationComplexity, OptimizationStatus, ScalabilityImprovements,
        SystemOptimizationCoordinator, UltimateIntegrationOptimizer, UltimateOptimizationResult,
    };
    pub use crate::{Tensor, TensorConvenience, TensorStorage};
    pub use torsh_core::{
        device::DeviceType,
        dtype::{DType, FloatElement, TensorElement},
        error::{Result, TorshError},
        shape::Shape,
    };
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_core::dtype::DType;

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
        let _src = Tensor::from_data(vec![10.0f32, 20.0], vec![2], DeviceType::Cpu).unwrap();

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
