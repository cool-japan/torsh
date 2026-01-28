//! # ToRSh Python Bindings - PyTorch-Compatible Deep Learning in Rust
//!
//! This crate provides Python bindings for the ToRSh deep learning framework using PyO3.
//! ToRSh offers a PyTorch-compatible API with the performance and safety of Rust.
//!
//! ## Architecture
//!
//! Following the QuantRS2-py design pattern, this crate is dedicated to Python bindings only.
//! The architecture emphasizes:
//! - Direct Python module definition (no feature gates for Python)
//! - Clean, focused API surface
//! - Seamless NumPy integration via scirs2-numpy
//! - Production-ready error handling
//!
//! ## Key Features
//!
//! - **PyTorch-Compatible API**: Familiar interface for PyTorch users
//! - **NumPy Integration**: Seamless tensor conversion via scirs2-numpy
//! - **Zero-Copy Operations**: Efficient memory management
//! - **Automatic Differentiation**: Full autograd support
//! - **Neural Network Modules**: Linear layers, activations, loss functions
//! - **Optimizers**: SGD, Adam, AdamW with PyTorch-compatible API
//! - **Data Loading**: Parallel data loaders with batching
//!
//! ## Usage
//!
//! ```python
//! import torsh
//!
//! # Create tensors
//! x = torsh.randn([2, 3])
//! y = torsh.ones([3, 4])
//!
//! # Neural network
//! model = torsh.Linear(3, 4)
//! optimizer = torsh.Adam(model.parameters(), lr=0.001)
//!
//! # Training loop
//! output = model(x)
//! loss = torsh.mse_loss(output, target)
//! loss.backward()
//! optimizer.step()
//! ```
//!
//! ## SciRS2 Policy Compliance
//!
//! This crate strictly follows the SciRS2 POLICY:
//! - Uses `scirs2_core::ndarray` for array operations (not direct ndarray)
//! - Uses `scirs2_core::random` for RNG (not direct rand/rand_distr)
//! - Uses `scirs2-numpy` for NumPy integration (SciRS2-compatible)
//! - All numerical operations through scirs2 abstractions
//!
//! ## Building
//!
//! This crate is designed to be built with Maturin:
//!
//! ```bash
//! # Development build
//! maturin develop
//!
//! # Production build
//! maturin build --release
//!
//! # With GPU support
//! maturin develop --features gpu
//! ```
//!
//! ## COOLJAPAN Pure Rust Policy
//!
//! While this crate produces a Python extension module (cdylib), the core ToRSh
//! framework remains 100% Pure Rust. This crate is the Python bridge only.

#![allow(dead_code)] // Framework infrastructure for future use

use pyo3::prelude::*;

// ============================================================================
// Python Module Components
// ============================================================================

// Core tensor implementation
mod tensor;
use tensor::PyTensor;

// Functional operations (activations, loss functions)
mod functional;

// Neural network modules
mod module;
use module::PyLinear;

// Optimizers
mod optimizer;
use optimizer::{PyAdam, PySGD};

// Data loading
mod dataloader;
use dataloader::{PyDataLoader, PyDataLoaderBuilder, PyRandomDataLoader};

// Utility functions
mod utils;

// Error handling
pub mod error;
pub use error::FfiError;

// Integration modules for advanced features
mod numpy_compatibility;
mod pandas_support;
mod scipy_integration;

// NumPy compatibility is provided through utility functions, not a Python class
use pandas_support::{DataAnalysisResult, PandasSupport, TorshDataFrame, TorshSeries};
use scipy_integration::{LinalgResult, OptimizationResult, SciPyIntegration, SignalResult};

// ============================================================================
// Python Module Definition (following QuantRS2-py pattern)
// ============================================================================

/// ToRSh - PyTorch-compatible deep learning framework in Rust
///
/// This is the main Python module entry point, following the QuantRS2-py architecture.
/// Python bindings are ALWAYS enabled (no feature gates) as this is a Python extension module.
#[pymodule]
fn torsh(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "__doc__",
        "ToRSh: PyTorch-compatible deep learning framework in Pure Rust",
    )?;

    // ========================================================================
    // Core Classes
    // ========================================================================

    // Add tensor class
    m.add_class::<PyTensor>()?;

    // Add neural network modules
    m.add_class::<PyLinear>()?;

    // Add optimizers
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;

    // Add data loaders
    m.add_class::<PyDataLoader>()?;
    m.add_class::<PyRandomDataLoader>()?;
    m.add_class::<PyDataLoaderBuilder>()?;

    // ========================================================================
    // Functional Operations
    // ========================================================================

    // Activation functions
    m.add_function(wrap_pyfunction!(functional::relu, m)?)?;
    m.add_function(wrap_pyfunction!(functional::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(functional::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(functional::softmax, m)?)?;
    m.add_function(wrap_pyfunction!(functional::gelu, m)?)?;
    m.add_function(wrap_pyfunction!(functional::log_softmax, m)?)?;

    // Loss functions
    m.add_function(wrap_pyfunction!(functional::cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(functional::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(functional::binary_cross_entropy, m)?)?;

    // ========================================================================
    // Tensor Creation & Manipulation
    // ========================================================================

    m.add_function(wrap_pyfunction!(utils::tensor, m)?)?;
    m.add_function(wrap_pyfunction!(utils::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(utils::ones, m)?)?;
    m.add_function(wrap_pyfunction!(utils::randn, m)?)?;
    m.add_function(wrap_pyfunction!(utils::rand, m)?)?;
    m.add_function(wrap_pyfunction!(utils::eye, m)?)?;
    m.add_function(wrap_pyfunction!(utils::full, m)?)?;
    m.add_function(wrap_pyfunction!(utils::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(utils::arange, m)?)?;
    m.add_function(wrap_pyfunction!(utils::stack, m)?)?;
    m.add_function(wrap_pyfunction!(utils::cat, m)?)?;

    // NumPy interop
    m.add_function(wrap_pyfunction!(utils::from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(utils::to_numpy, m)?)?;

    // Random seed
    m.add_function(wrap_pyfunction!(utils::manual_seed, m)?)?;

    // ========================================================================
    // Data Loading Functions
    // ========================================================================

    m.add_function(wrap_pyfunction!(dataloader::create_dataloader, m)?)?;
    m.add_function(wrap_pyfunction!(dataloader::create_dataset_from_array, m)?)?;
    m.add_function(wrap_pyfunction!(dataloader::get_dataloader_info, m)?)?;
    m.add_function(wrap_pyfunction!(dataloader::benchmark_dataloader, m)?)?;

    // ========================================================================
    // Device Management
    // ========================================================================

    m.add_function(wrap_pyfunction!(utils::cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(utils::cuda_device_count, m)?)?;

    // ========================================================================
    // Integration Utilities (SciPy, Pandas)
    // ========================================================================

    // SciPy integration
    m.add_class::<SciPyIntegration>()?;
    m.add_class::<OptimizationResult>()?;
    m.add_class::<LinalgResult>()?;
    m.add_class::<SignalResult>()?;

    // Pandas integration
    m.add_class::<PandasSupport>()?;
    m.add_class::<TorshDataFrame>()?;
    m.add_class::<TorshSeries>()?;
    m.add_class::<DataAnalysisResult>()?;

    // NumPy compatibility is provided through utility functions

    // ========================================================================
    // Submodules for Advanced Features
    // ========================================================================

    // Create SciPy utilities submodule
    let scipy_utils = scipy_integration::create_scipy_utilities(m.py())?;
    m.add("scipy", scipy_utils)?;

    // Create Pandas utilities submodule
    let pandas_utils = pandas_support::create_pandas_utilities(m.py())?;
    m.add("pandas", pandas_utils)?;

    // ========================================================================
    // Error Handling
    // ========================================================================

    // Register custom exception types
    error::python_exceptions::register_exceptions(m)?;

    // ========================================================================
    // Module Metadata
    // ========================================================================

    // Add framework information
    m.add("__author__", "COOLJAPAN OU (Team Kitasan)")?;
    m.add("__license__", "MIT OR Apache-2.0")?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "test_torsh").unwrap();
            let result = torsh(&module);
            assert!(result.is_ok(), "Module creation failed: {:?}", result.err());
        });
    }

    #[test]
    fn test_module_has_version() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "test_torsh").unwrap();
            torsh(&module).unwrap();
            let version = module.getattr("__version__").unwrap();
            assert!(!version.to_string().is_empty());
        });
    }
}
