//! Python bindings for ToRSh - PyTorch-compatible deep learning in Rust
//!
//! This crate provides Python bindings for the ToRSh deep learning framework,
//! enabling PyTorch-compatible APIs to be used from Python.
//!
//! # Modular Structure
//!
//! The crate is organized into focused modules:
//! - `tensor` - Tensor operations and creation functions
//! - `nn` - Neural network layers and containers
//! - `optim` - Optimization algorithms
//! - `device` - Device management and utilities
//! - `dtype` - Data type definitions and conversions
//! - `error` - Error handling and conversions
//! - `utils` - Common utilities and helpers

use pyo3::prelude::*;

// Core modules - modular structure
pub mod device;
pub mod dtype;
pub mod error;
// pub mod nn;  // Temporarily disabled due to scirs2-autograd conflicts
// pub mod optim;  // Temporarily disabled due to scirs2-autograd conflicts
// pub mod tensor;  // Temporarily disabled due to scirs2-autograd conflicts
pub mod utils;

// Legacy modules (temporarily kept for compatibility)
// pub mod autograd;  // Temporarily disabled due to scirs2 API incompatibilities
// pub mod distributed;  // Temporarily disabled for compilation
// pub mod functional;  // Fixed for PyO3 0.25 but disabled until tensor ops are implemented

// Re-export main types
pub use device::PyDevice;
pub use dtype::PyDType;
pub use error::TorshPyError;
// pub use tensor::PyTensor;  // Temporarily disabled due to scirs2-autograd conflicts

/// ToRSh Python module
#[pymodule]
fn torsh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register main classes
    // m.add_class::<PyTensor>()?;  // Temporarily disabled due to scirs2-autograd conflicts
    m.add_class::<PyDevice>()?;
    m.add_class::<PyDType>()?;

    // Add submodules with new modular structure
    // let nn_module = PyModule::new(m.py(), "nn")?;
    // nn::register_nn_module(m.py(), &nn_module)?;
    // m.add_submodule(&nn_module)?;

    // let optim_module = PyModule::new(m.py(), "optim")?;
    // optim::register_optim_module(m.py(), &optim_module)?;
    // m.add_submodule(&optim_module)?;

    // let autograd_module = PyModule::new(m.py(), "autograd")?;
    // autograd::register_autograd_module(m.py(), &autograd_module)?;
    // m.add_submodule(&autograd_module)?;

    // let distributed_module = PyModule::new(m.py(), "distributed")?;
    // distributed::register_distributed_module(m.py(), &distributed_module)?;
    // m.add_submodule(&distributed_module)?;

    // let functional_module = PyModule::new(m.py(), "F")?;
    // functional::register_functional_module(m.py(), &functional_module)?;
    // m.add_submodule(&functional_module)?;

    // Add tensor creation functions
    // tensor::register_creation_functions(m)?; // Disabled: tensor module commented out

    // Add device and dtype constants
    device::register_device_constants(m)?;
    dtype::register_dtype_constants(m)?;

    // Register error types
    error::register_error_types(m)?;

    // Set version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn torsh_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    torsh(m)
}
