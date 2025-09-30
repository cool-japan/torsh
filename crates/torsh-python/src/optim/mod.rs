//! Optimization algorithms module - PyTorch-compatible optimizers
//!
//! This module provides a modular structure for optimization algorithms:
//! - `base` - Base PyOptimizer class and common functionality
//! - `sgd` - Stochastic Gradient Descent optimizer
//! - `adam` - Adam and AdamW optimizers
//! - `adagrad` - Adagrad optimizer
//! - `rmsprop` - RMSprop optimizer

pub mod adagrad;
pub mod adam;
pub mod base;
pub mod rmsprop;
pub mod sgd;

// Re-export the main types
pub use adagrad::PyAdaGrad;
pub use adam::{PyAdam, PyAdamW};
pub use base::PyOptimizer;
pub use rmsprop::PyRMSprop;
pub use sgd::PySGD;

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};

/// Register the optim module with Python
pub fn register_optim_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register base optimizer
    m.add_class::<PyOptimizer>()?;

    // Register specific optimizers
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyAdaGrad>()?;
    m.add_class::<PyRMSprop>()?;

    Ok(())
}
