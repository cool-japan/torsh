//! Activation functions for neural networks

use super::module::PyModule;
use crate::{error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;

/// ReLU activation function
#[pyclass(name = "ReLU", extends = PyModule)]
pub struct PyReLU {
    inplace: bool,
    training: bool,
}

#[pymethods]
impl PyReLU {
    #[new]
    fn new(inplace: Option<bool>) -> (Self, PyModule) {
        (
            Self {
                inplace: inplace.unwrap_or(false),
                training: true,
            },
            PyModule::new(),
        )
    }

    /// Forward pass through ReLU
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(input.tensor.relu())?;
        Ok(PyTensor { tensor: result })
    }

    /// String representation
    fn __repr__(&self) -> String {
        if self.inplace {
            "ReLU(inplace=True)".to_string()
        } else {
            "ReLU()".to_string()
        }
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

/// Sigmoid activation function
#[pyclass(name = "Sigmoid", extends = PyModule)]
pub struct PySigmoid {
    training: bool,
}

#[pymethods]
impl PySigmoid {
    #[new]
    fn new() -> (Self, PyModule) {
        (Self { training: true }, PyModule::new())
    }

    /// Forward pass through Sigmoid
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(input.tensor.sigmoid())?;
        Ok(PyTensor { tensor: result })
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Sigmoid()".to_string()
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

/// Tanh activation function
#[pyclass(name = "Tanh", extends = PyModule)]
pub struct PyTanh {
    training: bool,
}

#[pymethods]
impl PyTanh {
    #[new]
    fn new() -> (Self, PyModule) {
        (Self { training: true }, PyModule::new())
    }

    /// Forward pass through Tanh
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let result = py_result!(input.tensor.tanh())?;
        Ok(PyTensor { tensor: result })
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Tanh()".to_string()
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
