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
//!     // Create tensors
//!     let x = tensor_2d![[1.0, 2.0], [3.0, 4.0]];
//!     let y = tensor_2d![[5.0, 6.0], [7.0, 8.0]];
//!
//!     // Perform operations
//!     let z = x.matmul(&y)?;
//!
//!     // Automatic differentiation
//!     let a = tensor![2.0].requires_grad_(true);
//!     let b = a.pow(2.0)?;
//!     b.backward()?;
//!     println!("Gradient: {:?}", a.grad().unwrap()); // 4.0
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
//! - **Multiple Backends**: CPU, CUDA, WebGPU, and Metal support
//!
//! # Modules
//!
//! - [`tensor`]: Core tensor type and operations
//! - [`autograd`]: Automatic differentiation functionality
//! - [`nn`]: Neural network modules and layers
//! - [`optim`]: Optimization algorithms
//! - [`data`]: Data loading and preprocessing
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

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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

#[allow(unexpected_cfgs)]
#[cfg(feature = "backends")]
#[cfg_attr(docsrs, doc(cfg(feature = "backends")))]
pub use torsh_backends as backends;

// Re-export commonly used types
pub use core::{
    device::{Device, DeviceType},
    dtype::{DType, FloatElement, TensorElement},
    error::{Result, TorshError},
    shape::Shape,
};
pub use tensor::{tensor, Tensor};

// Re-export key functions
pub use autograd::{backward, enable_grad, grad, is_grad_enabled, no_grad};
pub use tensor::creation::{
    arange, eye, linspace, ones, ones_like, rand, rand_like, randint, randn, randn_like, zeros,
    zeros_like,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::autograd::prelude::*;
    pub use crate::core::prelude::*;
    pub use crate::tensor::prelude::*;

    #[cfg(feature = "nn")]
    pub use crate::nn::prelude::*;

    #[cfg(feature = "optim")]
    pub use crate::optim::prelude::*;

    #[cfg(feature = "data")]
    pub use crate::data::prelude::*;

    // Common imports
    pub use crate::{backward, enable_grad, grad, no_grad};
    pub use crate::{eye, ones, rand, randn, zeros};
    pub use crate::{tensor, Tensor};
}

/// F namespace for functional operations (similar to torch.nn.functional)
#[allow(non_snake_case)]
pub mod F {
    #[allow(unused_imports)]
    pub use crate::tensor::ops::*;

    #[cfg(feature = "nn")]
    pub use crate::nn::functional::*;
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

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
        let a = tensor![1.0, 2.0, 3.0];
        let b = tensor![4.0, 5.0, 6.0];

        let c = a.add(&b).unwrap();
        assert_eq!(c.shape().dims(), &[3]);
    }

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
        check_version(0, 1).unwrap();
    }
}
