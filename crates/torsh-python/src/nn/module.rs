//! Base neural network module - Foundation for all PyTorch-compatible layers

use crate::{device::PyDevice, error::PyResult, tensor::PyTensor};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Base class for all neural network modules
#[pyclass(name = "Module", subclass)]
pub struct PyModule {
    // This will be overridden by subclasses
}

#[pymethods]
impl PyModule {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    /// Get all parameters of the module
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        // Default implementation - subclasses should override
        Ok(Vec::new())
    }

    /// Get all named parameters of the module
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        // Default implementation - subclasses should override
        Ok(HashMap::new())
    }

    /// Set the module in training mode
    fn train(&mut self, mode: Option<bool>) {
        // Default implementation - subclasses should override
        let _mode = mode.unwrap_or(true);
        // Subclasses should implement actual training mode logic
    }

    /// Set the module in evaluation mode
    fn eval(&mut self) {
        // Default implementation - subclasses should override
        // Subclasses should implement actual evaluation mode logic
    }

    /// Move module to specified device
    fn to(&mut self, device: PyDevice) -> PyResult<()> {
        // Default implementation - subclasses should override
        let _device = device;
        Ok(())
    }

    /// Zero out gradients of all parameters
    fn zero_grad(&mut self) {
        // Default implementation - subclasses should override
        // Subclasses should implement actual gradient zeroing
    }

    /// Make module callable (forward pass)
    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    /// Forward pass - must be implemented by subclasses
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subclasses must implement forward method",
        ))
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Module()".to_string()
    }

    /// Apply a function to all submodules
    fn apply(&mut self, _func: PyObject) -> PyResult<()> {
        // Default implementation - subclasses should override
        Ok(())
    }

    /// Get the state dict (parameters and buffers)
    fn state_dict(&self) -> PyResult<HashMap<String, PyTensor>> {
        // Default implementation returns named parameters
        self.named_parameters()
    }

    /// Load state dict (parameters and buffers)
    fn load_state_dict(&mut self, _state_dict: HashMap<String, PyTensor>) -> PyResult<()> {
        // Default implementation - subclasses should override
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subclasses must implement load_state_dict method",
        ))
    }

    /// Get number of parameters
    fn num_parameters(&self) -> PyResult<usize> {
        let params = self.parameters()?;
        Ok(params.iter().map(|p| p.numel()).sum())
    }

    /// Check if module is in training mode
    fn training(&self) -> bool {
        // Default implementation - subclasses should track this
        true
    }
}
