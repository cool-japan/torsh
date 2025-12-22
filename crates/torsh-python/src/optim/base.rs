//! Base optimizer implementation - Foundation for all PyTorch-compatible optimizers

use crate::{error::PyResult, tensor::PyTensor};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;

/// Base optimizer class - foundation for all optimizers
#[pyclass(name = "Optimizer", subclass)]
pub struct PyOptimizer {
    // This will be overridden by subclasses
}

#[pymethods]
impl PyOptimizer {
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// Perform a single optimization step - must be implemented by subclasses
    fn step(&mut self) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subclasses must implement step method",
        ))
    }

    /// Zero out gradients of all parameters
    fn zero_grad(&mut self, set_to_none: Option<bool>) {
        // Default implementation - subclasses should override
        let _set_to_none = set_to_none.unwrap_or(false);
        // Subclasses should implement actual gradient zeroing
    }

    /// Get the state dictionary (optimizer state and hyperparameters)
    fn state_dict(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        // Default implementation - subclasses should override
        Ok(HashMap::new())
    }

    /// Load state dictionary
    fn load_state_dict(&mut self, state_dict: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        // Default implementation - subclasses should override
        let _state_dict = state_dict;
        Ok(())
    }

    /// Get parameter groups
    fn param_groups(&self) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
        // Default implementation - subclasses should override
        Ok(Vec::new())
    }

    /// Get current state
    fn state(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        // Default implementation - subclasses should override
        Ok(HashMap::new())
    }

    /// Add a new parameter group
    fn add_param_group(&mut self, param_group: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        // Default implementation - subclasses should override
        let _param_group = param_group;
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Subclasses must implement add_param_group method",
        ))
    }

    /// String representation
    fn __repr__(&self) -> String {
        "Optimizer()".to_string()
    }

    /// Get defaults (default hyperparameters)
    fn defaults(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        // Default implementation - subclasses should override
        Ok(HashMap::new())
    }
}

/// Helper function to extract parameters from Python objects
pub fn extract_parameters(params: Vec<PyTensor>) -> PyResult<Vec<torsh_tensor::Tensor<f32>>> {
    params.into_iter().map(|p| Ok(p.tensor)).collect()
}

/// Helper function to create parameter group
pub fn create_param_group(
    params: Vec<PyTensor>,
    lr: f32,
    extra_params: HashMap<String, Py<PyAny>>,
) -> PyResult<HashMap<String, Py<PyAny>>> {
    let mut param_group = HashMap::new();

    Python::attach(|py| {
        // Add parameters
        let py_params: Vec<Py<PyAny>> = params
            .into_iter()
            .map(|p| p.into_pyobject(py).unwrap().into())
            .collect();
        param_group.insert(
            "params".to_string(),
            py_params.into_pyobject(py).unwrap().into(),
        );

        // Add learning rate
        param_group.insert("lr".to_string(), lr.into_pyobject(py).unwrap().into());

        // Add extra parameters
        for (key, value) in extra_params {
            param_group.insert(key, value);
        }

        Ok(param_group)
    })
}
