//! Neural Network module - PyTorch-compatible neural network layers and containers
//!
//! This module provides a modular structure for neural network components:
//! - `module` - Base PyModule class and core functionality
//! - `linear` - Linear/Dense layers
//! - `container` - Sequential, ModuleList, and other containers
//! - `activation` - Activation functions
//! - `loss` - Loss functions
//! - `conv` - Convolutional layers (Conv1d, Conv2d)
//! - `normalization` - Normalization layers (BatchNorm, LayerNorm)
//! - `dropout` - Dropout and regularization layers
//! - `pooling` - Pooling layers (MaxPool, AvgPool, AdaptivePool)

pub mod activation;
pub mod container;
pub mod conv;
pub mod dropout;
pub mod linear;
pub mod loss;
pub mod module;
pub mod normalization;
pub mod pooling;

// Re-export the main types
pub use container::{PyModuleList, PySequential};
pub use conv::{PyConv1d, PyConv2d};
pub use dropout::{PyAlphaDropout, PyDropout, PyDropout2d, PyDropout3d};
pub use linear::PyLinear;
pub use module::PyModule as PyNNModule;
pub use normalization::{PyBatchNorm1d, PyBatchNorm2d, PyLayerNorm};
pub use pooling::{PyAdaptiveAvgPool2d, PyAdaptiveMaxPool2d, PyAvgPool2d, PyMaxPool2d};

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};

/// Register the nn module with Python
pub fn register_nn_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register base module
    m.add_class::<PyNNModule>()?;

    // Register linear layers
    m.add_class::<PyLinear>()?;

    // Register convolutional layers
    m.add_class::<PyConv1d>()?;
    m.add_class::<PyConv2d>()?;

    // Register normalization layers
    m.add_class::<PyBatchNorm1d>()?;
    m.add_class::<PyBatchNorm2d>()?;
    m.add_class::<PyLayerNorm>()?;

    // Register dropout layers
    m.add_class::<PyDropout>()?;
    m.add_class::<PyDropout2d>()?;
    m.add_class::<PyDropout3d>()?;
    m.add_class::<PyAlphaDropout>()?;

    // Register pooling layers
    m.add_class::<PyMaxPool2d>()?;
    m.add_class::<PyAvgPool2d>()?;
    m.add_class::<PyAdaptiveAvgPool2d>()?;
    m.add_class::<PyAdaptiveMaxPool2d>()?;

    // Register containers
    m.add_class::<PySequential>()?;
    m.add_class::<PyModuleList>()?;

    Ok(())
}
