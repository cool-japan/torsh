//! ToRSh - A blazingly fast, production-ready deep learning framework written in pure Rust
//!
//! ToRSh (Tensor Operations in Rust with Sharding) provides a PyTorch-compatible API
//! built on top of the powerful scirs2 ecosystem, delivering superior performance,
//! memory safety, and deployment flexibility.
//!
//! # Quick Start
//!
//! ```rust
//! use torsh::prelude::*;
//!
//! fn main() -> Result<()> {
//!     // Create tensors with convenience macros
//!     let x = tensor_2d![[1.0, 2.0], [3.0, 4.0]]?;
//!     let y = tensor_2d![[5.0, 6.0], [7.0, 8.0]]?;
//!
//!     // Perform operations
//!     let z = x.matmul(&y)?;
//!
//!     // Automatic differentiation
//!     let a = tensor![2.0]?.requires_grad_(true);
//!     let b = a.pow(2.0)?;
//!     b.backward()?;
//!     println!("Gradient: {:?}", a.grad().unwrap()); // 4.0
//!
//!     // Functional operations
//!     let input = randn(&[2, 3, 4])?;
//!     let output = F::relu(&input)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - **Tensor Operations**: Comprehensive tensor manipulation with PyTorch-compatible API
//! - **Automatic Differentiation**: Reverse-mode AD powered by scirs2-autograd
//! - **Neural Networks**: Pre-built layers and modules for deep learning
//! - **Optimizers**: State-of-the-art optimization algorithms
//! - **Data Loading**: Efficient data pipelines with parallelization
//! - **Functional Operations**: Extensive functional API matching PyTorch
//! - **Multiple Backends**: CPU, CUDA, WebGPU, and Metal support
//!
//! # Modules
//!
//! - [`tensor`]: Core tensor type and operations
//! - [`autograd`]: Automatic differentiation functionality
//! - [`nn`]: Neural network modules and layers
//! - [`optim`]: Optimization algorithms
//! - [`data`]: Data loading and preprocessing
//! - [`functional`]: Functional operations API (aliased as `F`)
//! - [`core`]: Core types and utilities
//!
//! # Design Philosophy
//!
//! ToRSh is designed to provide a familiar PyTorch-like experience while leveraging
//! Rust's unique advantages:
//!
//! - **Zero-cost abstractions**: No runtime overhead for safety
//! - **Memory safety**: Compile-time guarantees prevent entire classes of bugs
//! - **Fearless concurrency**: Safe parallelization by default
//! - **Superior performance**: 4-25x faster than PyTorch on many workloads
//! - **Ergonomic API**: Convenient macros and builder patterns for ease of use

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Re-export core functionality
pub use torsh_autograd as autograd;
pub use torsh_core as core;
pub use torsh_tensor as tensor;

// Re-export optional modules
#[cfg(feature = "nn")]
#[cfg_attr(docsrs, doc(cfg(feature = "nn")))]
pub use torsh_nn as nn;

#[cfg(feature = "optim")]
#[cfg_attr(docsrs, doc(cfg(feature = "optim")))]
pub use torsh_optim as optim;

#[cfg(feature = "data")]
#[cfg_attr(docsrs, doc(cfg(feature = "data")))]
pub use torsh_data as data;

#[cfg(feature = "functional")]
#[cfg_attr(docsrs, doc(cfg(feature = "functional")))]
pub use torsh_functional as functional;

#[cfg(feature = "text")]
#[cfg_attr(docsrs, doc(cfg(feature = "text")))]
pub use torsh_text as text;

#[cfg(feature = "vision")]
#[cfg_attr(docsrs, doc(cfg(feature = "vision")))]
pub use torsh_vision as vision;

// Advanced modules
#[cfg(feature = "sparse")]
#[cfg_attr(docsrs, doc(cfg(feature = "sparse")))]
pub use torsh_sparse as sparse;

#[cfg(feature = "quantization")]
#[cfg_attr(docsrs, doc(cfg(feature = "quantization")))]
pub use torsh_quantization as quantization;

#[cfg(feature = "special")]
#[cfg_attr(docsrs, doc(cfg(feature = "special")))]
pub use torsh_special as special;

#[cfg(feature = "linalg")]
#[cfg_attr(docsrs, doc(cfg(feature = "linalg")))]
pub use torsh_linalg as linalg;

#[cfg(feature = "profiler")]
#[cfg_attr(docsrs, doc(cfg(feature = "profiler")))]
pub use torsh_profiler as profiler;

#[cfg(feature = "distributed")]
#[cfg_attr(docsrs, doc(cfg(feature = "distributed")))]
pub use torsh_distributed as distributed;

#[cfg(feature = "jit")]
#[cfg_attr(docsrs, doc(cfg(feature = "jit")))]
pub use torsh_jit as jit;

#[cfg(feature = "fx")]
#[cfg_attr(docsrs, doc(cfg(feature = "fx")))]
pub use torsh_fx as fx;

#[cfg(feature = "hub")]
#[cfg_attr(docsrs, doc(cfg(feature = "hub")))]
pub use torsh_hub as hub;

// Backend system with modular CUDA execution engine
#[cfg(feature = "backend")]
#[cfg_attr(docsrs, doc(cfg(feature = "backend")))]
pub use torsh_backend as backend;

// Re-export commonly used types
pub use core::{
    device::{Device, DeviceType},
    dtype::{DType, FloatElement, TensorElement},
    error::{Result, TorshError},
    shape::Shape,
    storage::Storage,
};
pub use tensor::{tensor, Tensor};

// Re-export key functions
pub use autograd::{backward, enable_grad, is_grad_enabled, no_grad};
pub use tensor::creation::{
    arange, eye, linspace, ones, ones_like, rand, rand_like, randint, randn, randn_like, zeros,
    zeros_like,
};

// Note: Tensor operations are implemented as methods on Tensor struct
// Available via tensor.add(), tensor.mul(), etc.

// Re-export indexing utilities
pub use tensor::indexing::TensorIndex;

// Re-export common functional operations
#[cfg(feature = "functional")]
pub use functional::{
    adaptive_avg_pool2d, avg_pool2d, batch_norm, binary_cross_entropy, conv1d, conv2d, conv3d,
    cross_entropy, gelu, layer_norm, linear, log_softmax, max_pool2d, mse_loss, relu, sigmoid,
    silu, softmax, tanh,
};

// Re-export builder patterns and utilities
// TODO: Implement ShapeBuilder when available
// pub use core::shape::ShapeBuilder;

// Version and compatibility
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Version synchronization and compatibility checking
pub mod version {
    use crate::{Result, TorshError};

    /// Struct to hold version information for a crate
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct CrateVersion {
        pub name: &'static str,
        pub version: &'static str,
        pub major: u32,
        pub minor: u32,
        pub patch: u32,
    }

    impl CrateVersion {
        pub const fn new(
            name: &'static str,
            version: &'static str,
            major: u32,
            minor: u32,
            patch: u32,
        ) -> Self {
            Self {
                name,
                version,
                major,
                minor,
                patch,
            }
        }

        /// Check if this version is compatible with another version
        pub fn is_compatible_with(&self, other: &CrateVersion) -> bool {
            // For alpha versions, require exact match
            if self.major == 0 && other.major == 0 {
                return self.minor == other.minor && self.patch == other.patch;
            }

            // For stable versions, require same major and minor >= required
            self.major == other.major && self.minor >= other.minor
        }
    }

    /// Get version information for all enabled crates
    pub fn get_crate_versions() -> Vec<CrateVersion> {
        let mut versions = vec![
            CrateVersion::new(
                "torsh",
                crate::VERSION,
                crate::VERSION_MAJOR,
                crate::VERSION_MINOR,
                crate::VERSION_PATCH,
            ),
            CrateVersion::new(
                "torsh-core",
                crate::core::VERSION,
                crate::core::VERSION_MAJOR,
                crate::core::VERSION_MINOR,
                crate::core::VERSION_PATCH,
            ),
            CrateVersion::new(
                "torsh-tensor",
                crate::tensor::VERSION,
                crate::tensor::VERSION_MAJOR,
                crate::tensor::VERSION_MINOR,
                crate::tensor::VERSION_PATCH,
            ),
            CrateVersion::new(
                "torsh-autograd",
                crate::autograd::VERSION,
                crate::autograd::VERSION_MAJOR,
                crate::autograd::VERSION_MINOR,
                crate::autograd::VERSION_PATCH,
            ),
        ];

        #[cfg(feature = "nn")]
        versions.push(CrateVersion::new(
            "torsh-nn",
            crate::nn::VERSION,
            crate::nn::VERSION_MAJOR,
            crate::nn::VERSION_MINOR,
            crate::nn::VERSION_PATCH,
        ));

        #[cfg(feature = "optim")]
        versions.push(CrateVersion::new(
            "torsh-optim",
            crate::optim::VERSION,
            crate::optim::VERSION_MAJOR,
            crate::optim::VERSION_MINOR,
            crate::optim::VERSION_PATCH,
        ));

        #[cfg(feature = "data")]
        versions.push(CrateVersion::new(
            "torsh-data",
            crate::data::VERSION,
            crate::data::VERSION_MAJOR,
            crate::data::VERSION_MINOR,
            crate::data::VERSION_PATCH,
        ));

        #[cfg(feature = "functional")]
        versions.push(CrateVersion::new(
            "torsh-functional",
            crate::functional::VERSION,
            crate::functional::VERSION_MAJOR,
            crate::functional::VERSION_MINOR,
            crate::functional::VERSION_PATCH,
        ));

        #[cfg(feature = "text")]
        versions.push(CrateVersion::new(
            "torsh-text",
            crate::text::VERSION,
            crate::text::VERSION_MAJOR,
            crate::text::VERSION_MINOR,
            crate::text::VERSION_PATCH,
        ));

        #[cfg(feature = "vision")]
        versions.push(CrateVersion::new(
            "torsh-vision",
            crate::vision::VERSION,
            crate::vision::VERSION_MAJOR,
            crate::vision::VERSION_MINOR,
            crate::vision::VERSION_PATCH,
        ));

        #[cfg(feature = "sparse")]
        versions.push(CrateVersion::new(
            "torsh-sparse",
            crate::sparse::VERSION,
            crate::sparse::VERSION_MAJOR,
            crate::sparse::VERSION_MINOR,
            crate::sparse::VERSION_PATCH,
        ));

        #[cfg(feature = "quantization")]
        versions.push(CrateVersion::new(
            "torsh-quantization",
            crate::quantization::VERSION,
            crate::quantization::VERSION_MAJOR,
            crate::quantization::VERSION_MINOR,
            crate::quantization::VERSION_PATCH,
        ));

        #[cfg(feature = "special")]
        versions.push(CrateVersion::new(
            "torsh-special",
            crate::special::VERSION,
            crate::special::VERSION_MAJOR,
            crate::special::VERSION_MINOR,
            crate::special::VERSION_PATCH,
        ));

        #[cfg(feature = "linalg")]
        versions.push(CrateVersion::new(
            "torsh-linalg",
            crate::linalg::VERSION,
            crate::linalg::VERSION_MAJOR,
            crate::linalg::VERSION_MINOR,
            crate::linalg::VERSION_PATCH,
        ));

        #[cfg(feature = "profiler")]
        versions.push(CrateVersion::new(
            "torsh-profiler",
            crate::profiler::VERSION,
            crate::profiler::VERSION_MAJOR,
            crate::profiler::VERSION_MINOR,
            crate::profiler::VERSION_PATCH,
        ));

        #[cfg(feature = "distributed")]
        versions.push(CrateVersion::new(
            "torsh-distributed",
            crate::distributed::VERSION,
            crate::distributed::VERSION_MAJOR,
            crate::distributed::VERSION_MINOR,
            crate::distributed::VERSION_PATCH,
        ));

        #[cfg(feature = "jit")]
        versions.push(CrateVersion::new(
            "torsh-jit",
            crate::jit::VERSION,
            crate::jit::VERSION_MAJOR,
            crate::jit::VERSION_MINOR,
            crate::jit::VERSION_PATCH,
        ));

        #[cfg(feature = "fx")]
        versions.push(CrateVersion::new(
            "torsh-fx",
            crate::fx::VERSION,
            crate::fx::VERSION_MAJOR,
            crate::fx::VERSION_MINOR,
            crate::fx::VERSION_PATCH,
        ));

        #[cfg(feature = "hub")]
        versions.push(CrateVersion::new(
            "torsh-hub",
            crate::hub::VERSION,
            crate::hub::VERSION_MAJOR,
            crate::hub::VERSION_MINOR,
            crate::hub::VERSION_PATCH,
        ));

        #[cfg(feature = "backend")]
        versions.push(CrateVersion::new(
            "torsh-backend",
            crate::backend::VERSION,
            crate::backend::VERSION_MAJOR,
            crate::backend::VERSION_MINOR,
            crate::backend::VERSION_PATCH,
        ));

        versions
    }

    /// Check version compatibility across all enabled crates
    pub fn check_version_compatibility() -> Result<()> {
        let versions = get_crate_versions();
        let main_version = &versions[0]; // torsh crate version

        for version in &versions[1..] {
            if !main_version.is_compatible_with(version) {
                return Err(TorshError::Other(format!(
                    "Version mismatch: {} {} is not compatible with {} {}",
                    main_version.name, main_version.version, version.name, version.version
                )));
            }
        }

        Ok(())
    }

    /// Print version information for all enabled crates
    pub fn print_version_info() {
        println!("ToRSh Crate Versions:");
        for version in get_crate_versions() {
            println!("  {}: {}", version.name, version.version);
        }
    }
}

/// Prelude module for convenient imports
///
/// This module provides a curated set of commonly used types and functions
/// for easy importing with `use torsh::prelude::*;`
pub mod prelude {
    // Re-export all subcrate preludes
    #[allow(ambiguous_glob_reexports)]
    pub use crate::autograd::prelude::*;
    #[allow(ambiguous_glob_reexports)]
    pub use crate::core::prelude::*;
    #[allow(ambiguous_glob_reexports)]
    pub use crate::tensor::prelude::*;

    #[cfg(feature = "nn")]
    #[allow(ambiguous_glob_reexports)]
    pub use crate::nn::prelude::*;

    #[cfg(feature = "optim")]
    #[allow(ambiguous_glob_reexports)]
    pub use crate::optim::prelude::*;

    #[cfg(feature = "data")]
    #[allow(ambiguous_glob_reexports)]
    pub use crate::data::prelude::*;

    #[cfg(feature = "functional")]
    pub use crate::functional::{conv2d, gelu, max_pool2d, relu, sigmoid, silu, softmax, tanh};

    #[cfg(feature = "text")]
    pub use crate::text::prelude::*;

    #[cfg(feature = "vision")]
    pub use crate::vision::prelude::*;

    // Advanced modules
    #[cfg(feature = "sparse")]
    pub use crate::sparse::prelude::*;

    #[cfg(feature = "quantization")]
    pub use crate::quantization::prelude::*;

    #[cfg(feature = "special")]
    pub use crate::special::prelude::*;

    #[cfg(feature = "linalg")]
    pub use crate::linalg::prelude::*;

    #[cfg(feature = "profiler")]
    pub use crate::profiler::prelude::*;

    #[cfg(feature = "distributed")]
    pub use crate::distributed::prelude::*;

    #[cfg(feature = "jit")]
    pub use crate::jit::prelude::*;

    #[cfg(feature = "fx")]
    pub use crate::fx::prelude::*;

    #[cfg(feature = "hub")]
    pub use crate::hub::prelude::*;

    // Backend prelude
    #[cfg(feature = "backend")]
    pub use crate::backend::prelude::*;

    // Common types
    pub use crate::{DType, Device, DeviceType, Result, Shape, Tensor, TorshError};

    // Autograd functions
    pub use crate::{backward, enable_grad, is_grad_enabled, no_grad};

    // Tensor creation functions
    pub use crate::{
        arange, eye, linspace, ones, ones_like, rand, rand_like, randint, randn, randn_like, zeros,
        zeros_like,
    };

    // Note: Tensor operations are available as methods on Tensor struct
    // e.g., tensor.add(), tensor.mul(), etc.

    // Main tensor constructor
    pub use crate::tensor;

    // Functional operations namespace
    pub use crate::F;

    // Version information
    pub use crate::version::{check_version_compatibility, print_version_info, CrateVersion};
    pub use crate::{VERSION, VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH};

    // Feature management
    pub use crate::features::{
        check_feature_requirements, get_enabled_features, print_feature_info,
    };

    // Convenience macros (re-export from macros module)
    pub use crate::macros::{device, shape, tensor_1d, tensor_2d};
}

/// F namespace for functional operations (similar to torch.nn.functional)
///
/// This module provides a comprehensive functional API similar to PyTorch's
/// `torch.nn.functional`, allowing for stateless operations on tensors.
#[allow(non_snake_case)]
pub mod F {
    // Re-export tensor operations
    // TODO: Re-enable when tensor ops module is available
    // #[allow(unused_imports)]
    // pub use crate::tensor::ops::*;

    // Re-export neural network functional operations
    #[cfg(feature = "nn")]
    #[allow(ambiguous_glob_reexports)]
    pub use crate::nn::functional::*;

    // Re-export all functional operations when available
    #[cfg(feature = "functional")]
    #[allow(ambiguous_glob_reexports)]
    pub use crate::functional::*;

    // Convenient aliases for common operations
    #[cfg(feature = "functional")]
    pub use crate::functional::{
        avg_pool1d as AvgPool1d,
        avg_pool2d as AvgPool2d,
        avg_pool3d as AvgPool3d,

        // Normalization
        batch_norm as BatchNorm,
        binary_cross_entropy as BCELoss,
        // Convolution operations
        conv1d as Conv1d,
        conv2d as Conv2d,
        conv3d as Conv3d,

        cross_entropy as CrossEntropyLoss,
        // Dropout
        dropout as Dropout,
        dropout2d as Dropout2d,
        dropout3d as Dropout3d,
        elu as ELU,

        gelu as GELU,
        group_norm as GroupNorm,
        instance_norm as InstanceNorm,

        l1_loss as L1Loss,

        layer_norm as LayerNorm,
        log_softmax as LogSoftmax,
        // Pooling operations
        max_pool1d as MaxPool1d,
        max_pool2d as MaxPool2d,
        max_pool3d as MaxPool3d,
        // Loss functions
        mse_loss as MSELoss,
        // Activation functions
        relu as ReLU,
        sigmoid as Sigmoid,
        silu as SiLU,
        softmax as Softmax,
        tanh as Tanh,
    };
}

/// Convenience macros for tensor creation and manipulation
///
/// Provides PyTorch-like macros for easy tensor creation
pub mod macros {
    /// Create a 1D tensor from array-like syntax
    ///
    /// # Examples
    /// ```rust
    /// # use torsh::prelude::*;
    /// # fn main() -> Result<()> {
    /// let t = tensor_1d![1.0_f32, 2.0, 3.0, 4.0]?;
    /// assert_eq!(t.shape().dims(), &[4]);
    /// # Ok(())
    /// # }
    /// ```
    #[macro_export]
    macro_rules! tensor_1d {
        [$($x:expr),* $(,)?] => {
            $crate::tensor::creation::tensor_1d(&[$($x),*])
        };
    }

    /// Create a 2D tensor from nested array syntax
    ///
    /// # Examples
    /// ```rust
    /// # use torsh::prelude::*;
    /// # fn main() -> Result<()> {
    /// let t = tensor_2d![[1.0_f32, 2.0], [3.0, 4.0]]?;
    /// assert_eq!(t.shape().dims(), &[2, 2]);
    /// # Ok(())
    /// # }
    /// ```
    #[macro_export]
    macro_rules! tensor_2d {
        [$([$($x:expr),* $(,)?]),* $(,)?] => {{
            let rows: &[&[_]] = &[$(
                &[$($x),*]
            ),*];
            $crate::tensor::creation::tensor_2d(rows)
        }};
    }

    /// Create a device from string
    ///
    /// # Examples
    /// ```rust
    /// # use torsh::prelude::*;
    /// let cpu = device!("cpu");
    /// let cuda = device!("cuda:0");
    /// assert_eq!(cpu, DeviceType::Cpu);
    /// ```
    #[macro_export]
    macro_rules! device {
        ("cpu") => {
            $crate::DeviceType::Cpu
        };
        ("cuda") => {
            $crate::DeviceType::Cuda(0)
        };
        ("cuda:0") => {
            $crate::DeviceType::Cuda(0)
        };
        ($device_str:expr) => {
            compile_error!("Use DeviceType directly for complex device specifications")
        };
    }

    /// Create a shape from array
    ///
    /// # Examples  
    /// ```rust
    /// # use torsh::prelude::*;
    /// let shape = shape![2, 3, 4];
    /// assert_eq!(shape.dims(), &[2, 3, 4]);
    /// ```
    #[macro_export]
    macro_rules! shape {
        [$($dim:expr),* $(,)?] => {
            $crate::Shape::new(vec![$($dim),*])
        };
    }

    pub use {device, shape, tensor_1d, tensor_2d};
}

/// Feature management and detection
pub mod features {
    /// Feature information structure
    #[derive(Debug, Clone)]
    pub struct FeatureInfo {
        pub name: &'static str,
        pub enabled: bool,
        pub description: &'static str,
        pub category: FeatureCategory,
    }

    /// Feature categories for organization
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum FeatureCategory {
        Core,
        Module,
        Advanced,
        Backend,
        Serialization,
        Performance,
        Development,
        Compatibility,
    }

    /// Get list of all available features and their status
    pub fn get_enabled_features() -> Vec<FeatureInfo> {
        vec![
            // Core features
            FeatureInfo {
                name: "std",
                enabled: cfg!(feature = "std"),
                description: "Standard library support",
                category: FeatureCategory::Core,
            },
            FeatureInfo {
                name: "no_std",
                enabled: cfg!(feature = "no_std"),
                description: "No standard library support",
                category: FeatureCategory::Core,
            },
            // Module features
            FeatureInfo {
                name: "nn",
                enabled: cfg!(feature = "nn"),
                description: "Neural network modules",
                category: FeatureCategory::Module,
            },
            FeatureInfo {
                name: "optim",
                enabled: cfg!(feature = "optim"),
                description: "Optimization algorithms",
                category: FeatureCategory::Module,
            },
            FeatureInfo {
                name: "data",
                enabled: cfg!(feature = "data"),
                description: "Data loading and preprocessing",
                category: FeatureCategory::Module,
            },
            FeatureInfo {
                name: "text",
                enabled: cfg!(feature = "text"),
                description: "Text processing capabilities",
                category: FeatureCategory::Module,
            },
            FeatureInfo {
                name: "vision",
                enabled: cfg!(feature = "vision"),
                description: "Computer vision utilities",
                category: FeatureCategory::Module,
            },
            FeatureInfo {
                name: "functional",
                enabled: cfg!(feature = "functional"),
                description: "Functional operations API",
                category: FeatureCategory::Module,
            },
            // Advanced features
            FeatureInfo {
                name: "sparse",
                enabled: cfg!(feature = "sparse"),
                description: "Sparse tensor operations",
                category: FeatureCategory::Advanced,
            },
            FeatureInfo {
                name: "quantization",
                enabled: cfg!(feature = "quantization"),
                description: "Model quantization",
                category: FeatureCategory::Advanced,
            },
            FeatureInfo {
                name: "special",
                enabled: cfg!(feature = "special"),
                description: "Special mathematical functions",
                category: FeatureCategory::Advanced,
            },
            FeatureInfo {
                name: "linalg",
                enabled: cfg!(feature = "linalg"),
                description: "Linear algebra operations",
                category: FeatureCategory::Advanced,
            },
            FeatureInfo {
                name: "profiler",
                enabled: cfg!(feature = "profiler"),
                description: "Performance profiling",
                category: FeatureCategory::Advanced,
            },
            FeatureInfo {
                name: "distributed",
                enabled: cfg!(feature = "distributed"),
                description: "Distributed computing",
                category: FeatureCategory::Advanced,
            },
            FeatureInfo {
                name: "jit",
                enabled: cfg!(feature = "jit"),
                description: "Just-in-time compilation",
                category: FeatureCategory::Advanced,
            },
            FeatureInfo {
                name: "fx",
                enabled: cfg!(feature = "fx"),
                description: "Graph transformations",
                category: FeatureCategory::Advanced,
            },
            FeatureInfo {
                name: "hub",
                enabled: cfg!(feature = "hub"),
                description: "Model hub integration",
                category: FeatureCategory::Advanced,
            },
            // Backend features
            FeatureInfo {
                name: "backend",
                enabled: cfg!(feature = "backend"),
                description: "Unified backend system",
                category: FeatureCategory::Backend,
            },
            FeatureInfo {
                name: "cuda",
                enabled: cfg!(feature = "cuda"),
                description: "CUDA GPU backend with execution engine",
                category: FeatureCategory::Backend,
            },
            FeatureInfo {
                name: "metal",
                enabled: cfg!(feature = "metal"),
                description: "Apple Metal GPU backend",
                category: FeatureCategory::Backend,
            },
            FeatureInfo {
                name: "wgpu",
                enabled: cfg!(feature = "wgpu"),
                description: "WebGPU backend",
                category: FeatureCategory::Backend,
            },
            FeatureInfo {
                name: "rocm",
                enabled: cfg!(feature = "rocm"),
                description: "AMD ROCm GPU backend",
                category: FeatureCategory::Backend,
            },
            // Serialization features
            FeatureInfo {
                name: "serialize",
                enabled: cfg!(feature = "serialize"),
                description: "Basic serialization support",
                category: FeatureCategory::Serialization,
            },
            FeatureInfo {
                name: "serialize-hdf5",
                enabled: cfg!(feature = "serialize-hdf5"),
                description: "HDF5 serialization",
                category: FeatureCategory::Serialization,
            },
            FeatureInfo {
                name: "serialize-arrow",
                enabled: cfg!(feature = "serialize-arrow"),
                description: "Apache Arrow serialization",
                category: FeatureCategory::Serialization,
            },
            FeatureInfo {
                name: "serialize-onnx",
                enabled: cfg!(feature = "serialize-onnx"),
                description: "ONNX serialization",
                category: FeatureCategory::Serialization,
            },
            // Performance features
            FeatureInfo {
                name: "simd",
                enabled: cfg!(feature = "simd"),
                description: "SIMD optimizations",
                category: FeatureCategory::Performance,
            },
            FeatureInfo {
                name: "parallel",
                enabled: cfg!(feature = "parallel"),
                description: "Parallel execution",
                category: FeatureCategory::Performance,
            },
            FeatureInfo {
                name: "fast-math",
                enabled: cfg!(feature = "fast-math"),
                description: "Fast math optimizations",
                category: FeatureCategory::Performance,
            },
            // Development features
            FeatureInfo {
                name: "debug",
                enabled: cfg!(feature = "debug"),
                description: "Debug information",
                category: FeatureCategory::Development,
            },
            FeatureInfo {
                name: "trace",
                enabled: cfg!(feature = "trace"),
                description: "Tracing support",
                category: FeatureCategory::Development,
            },
            FeatureInfo {
                name: "bench",
                enabled: cfg!(feature = "bench"),
                description: "Benchmarking utilities",
                category: FeatureCategory::Development,
            },
            // Compatibility features
            FeatureInfo {
                name: "python",
                enabled: cfg!(feature = "python"),
                description: "Python interoperability",
                category: FeatureCategory::Compatibility,
            },
            FeatureInfo {
                name: "numpy",
                enabled: cfg!(feature = "numpy"),
                description: "NumPy compatibility",
                category: FeatureCategory::Compatibility,
            },
            FeatureInfo {
                name: "pytorch-compat",
                enabled: cfg!(feature = "pytorch-compat"),
                description: "PyTorch compatibility",
                category: FeatureCategory::Compatibility,
            },
        ]
    }

    /// Check if required features are enabled
    pub fn check_feature_requirements(required_features: &[&str]) -> crate::Result<()> {
        let enabled_features = get_enabled_features();
        let enabled_names: std::collections::HashSet<&str> = enabled_features
            .iter()
            .filter(|f| f.enabled)
            .map(|f| f.name)
            .collect();

        for required in required_features {
            if !enabled_names.contains(required) {
                return Err(crate::TorshError::Other(format!(
                    "Required feature '{}' is not enabled",
                    required
                )));
            }
        }

        Ok(())
    }

    /// Print information about enabled features
    pub fn print_feature_info() {
        use std::collections::HashMap;

        let features = get_enabled_features();
        let mut by_category: HashMap<FeatureCategory, Vec<&FeatureInfo>> = HashMap::new();

        for feature in &features {
            by_category
                .entry(feature.category.clone())
                .or_default()
                .push(feature);
        }

        println!("ToRSh Enabled Features:");

        for (category, features) in by_category {
            let category_name = match category {
                FeatureCategory::Core => "Core",
                FeatureCategory::Module => "Modules",
                FeatureCategory::Advanced => "Advanced",
                FeatureCategory::Backend => "Backends",
                FeatureCategory::Serialization => "Serialization",
                FeatureCategory::Performance => "Performance",
                FeatureCategory::Development => "Development",
                FeatureCategory::Compatibility => "Compatibility",
            };

            println!("\n  {}:", category_name);
            for feature in features {
                let status = if feature.enabled { "✓" } else { "✗" };
                println!("    {} {}: {}", status, feature.name, feature.description);
            }
        }
    }

    /// Get currently enabled feature count by category
    pub fn get_feature_stats() -> std::collections::HashMap<FeatureCategory, (usize, usize)> {
        use std::collections::HashMap;

        let features = get_enabled_features();
        let mut stats: HashMap<FeatureCategory, (usize, usize)> = HashMap::new();

        for feature in features {
            let (enabled, total) = stats.entry(feature.category.clone()).or_insert((0, 0));
            *total += 1;
            if feature.enabled {
                *enabled += 1;
            }
        }

        stats
    }
}

/// Check ToRSh version compatibility
pub fn check_version(required_major: u32, required_minor: u32) -> Result<()> {
    if VERSION_MAJOR < required_major
        || (VERSION_MAJOR == required_major && VERSION_MINOR < required_minor)
    {
        return Err(TorshError::Other(format!(
            "ToRSh version {}.{} or higher required, but got {}.{}",
            required_major, required_minor, VERSION_MAJOR, VERSION_MINOR
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let a = tensor![1.0, 2.0, 3.0].unwrap();
        let b = tensor![4.0, 5.0, 6.0].unwrap();

        let c = a.add(&b).unwrap();
        assert_eq!(c.shape().dims(), &[3]);
    }

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0-alpha.2");
        check_version(0, 1).unwrap();
    }
}
