//! Loss functions for neural networks

use super::module::PyModule;
use crate::{error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;

/// Mean Squared Error loss
#[pyclass(name = "MSELoss", extends = PyModule)]
pub struct PyMSELoss {
    reduction: String,
    training: bool,
}

#[pymethods]
impl PyMSELoss {
    #[new]
    fn new(reduction: Option<String>) -> (Self, PyModule) {
        let reduction = reduction.unwrap_or_else(|| "mean".to_string());
        (
            Self {
                reduction,
                training: true,
            },
            PyModule::new(),
        )
    }

    /// Forward pass through MSE Loss
    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        // MSE = mean((input - target)^2)
        let diff = py_result!(input.tensor.sub(&target.tensor))?;
        let squared = py_result!(diff.pow(2.0))?;
        let result = py_result!(squared.mean(None, false))?;
        Ok(PyTensor { tensor: result })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("MSELoss(reduction='{}')", self.reduction)
    }

    /// Set training mode
    fn train(&mut self, mode: Option<bool>) {
        self.training = mode.unwrap_or(true);
    }

    /// Set evaluation mode
    fn eval(&mut self) {
        self.training = false;
    }

    /// Check if module is in training mode
    fn training(&self) -> bool {
        self.training
    }
}

/// Cross Entropy loss
#[pyclass(name = "CrossEntropyLoss", extends = PyModule)]
pub struct PyCrossEntropyLoss {
    reduction: String,
    training: bool,
}

#[pymethods]
impl PyCrossEntropyLoss {
    #[new]
    fn new(reduction: Option<String>) -> (Self, PyModule) {
        let reduction = reduction.unwrap_or_else(|| "mean".to_string());
        (
            Self {
                reduction,
                training: true,
            },
            PyModule::new(),
        )
    }

    /// Forward pass through Cross Entropy Loss
    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        // For now, use a placeholder - proper cross entropy needs more complex implementation
        let result = py_result!(input.tensor.sub(&target.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("CrossEntropyLoss(reduction='{}')", self.reduction)
    }

    /// Set training mode
    fn train(&mut self, mode: Option<bool>) {
        self.training = mode.unwrap_or(true);
    }

    /// Set evaluation mode
    fn eval(&mut self) {
        self.training = false;
    }

    /// Check if module is in training mode
    fn training(&self) -> bool {
        self.training
    }
}

/// Binary Cross Entropy loss
#[pyclass(name = "BCELoss", extends = PyModule)]
pub struct PyBCELoss {
    reduction: String,
    training: bool,
}

#[pymethods]
impl PyBCELoss {
    #[new]
    fn new(reduction: Option<String>) -> (Self, PyModule) {
        let reduction = reduction.unwrap_or_else(|| "mean".to_string());
        (
            Self {
                reduction,
                training: true,
            },
            PyModule::new(),
        )
    }

    /// Forward pass through BCE Loss
    fn forward(&self, input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
        // For now, use a placeholder - proper BCE needs more complex implementation
        let result = py_result!(input.tensor.sub(&target.tensor))?;
        Ok(PyTensor { tensor: result })
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("BCELoss(reduction='{}')", self.reduction)
    }

    /// Set training mode
    fn train(&mut self, mode: Option<bool>) {
        self.training = mode.unwrap_or(true);
    }

    /// Set evaluation mode
    fn eval(&mut self) {
        self.training = false;
    }

    /// Check if module is in training mode
    fn training(&self) -> bool {
        self.training
    }
}
