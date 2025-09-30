//! Neural network containers - Sequential, ModuleList, etc.

use super::module::PyModule;
use crate::{device::PyDevice, error::PyResult, tensor::PyTensor};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Sequential container - applies modules in sequence
#[pyclass(name = "Sequential", extends = PyModule)]
pub struct PySequential {
    modules: Vec<PyObject>,
    training: bool,
}

#[pymethods]
impl PySequential {
    #[new]
    fn new(modules: Option<Vec<PyObject>>) -> (Self, PyModule) {
        let modules = modules.unwrap_or_default();
        (
            Self {
                modules,
                training: true,
            },
            PyModule::new(),
        )
    }

    /// Add a module to the sequential container
    fn add_module(&mut self, name: &str, module: PyObject) {
        // For now, just add to the list (ignoring name)
        self.modules.push(module);
    }

    /// Forward pass through all modules in sequence
    fn forward(&self, mut input: PyTensor) -> PyResult<PyTensor> {
        Python::with_gil(|py| {
            for module in &self.modules {
                // Call the forward method on each module
                if let Ok(forward_method) = module.getattr(py, "forward") {
                    let result = forward_method.call1(py, (input.clone(),))?;
                    input = result.extract::<PyTensor>(py)?;
                } else {
                    // Try calling the module directly (__call__)
                    let result = module.call1(py, (input.clone(),))?;
                    input = result.extract::<PyTensor>(py)?;
                }
            }
            Ok(input)
        })
    }

    /// Get all parameters from all modules
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut all_params = Vec::new();
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(params_method) = module.getattr(py, "parameters") {
                    let params_result = params_method.call0(py)?;
                    if let Ok(params) = params_result.extract::<Vec<PyTensor>>(py) {
                        all_params.extend(params);
                    }
                }
            }
            Ok(all_params)
        })
    }

    /// Get all named parameters from all modules
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        let mut all_named_params = HashMap::new();
        Python::with_gil(|py| {
            for (i, module) in self.modules.iter().enumerate() {
                if let Ok(named_params_method) = module.getattr(py, "named_parameters") {
                    let named_params_result = named_params_method.call0(py)?;
                    if let Ok(named_params) =
                        named_params_result.extract::<HashMap<String, PyTensor>>(py)
                    {
                        for (name, param) in named_params {
                            all_named_params.insert(format!("{}.{}", i, name), param);
                        }
                    }
                }
            }
            Ok(all_named_params)
        })
    }

    /// Set training mode for all modules
    fn train(&mut self, mode: Option<bool>) {
        let mode = mode.unwrap_or(true);
        self.training = mode;
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(train_method) = module.getattr(py, "train") {
                    let _ = train_method.call1(py, (mode,));
                }
            }
        });
    }

    /// Set evaluation mode for all modules
    fn eval(&mut self) {
        self.training = false;
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(eval_method) = module.getattr(py, "eval") {
                    let _ = eval_method.call0(py);
                }
            }
        });
    }

    /// Move all modules to specified device
    fn to(&mut self, device: PyDevice) -> PyResult<()> {
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(to_method) = module.getattr(py, "to") {
                    to_method.call1(py, (device.clone(),))?;
                }
            }
            Ok(())
        })
    }

    /// Zero gradients for all modules
    fn zero_grad(&mut self) {
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(zero_grad_method) = module.getattr(py, "zero_grad") {
                    let _ = zero_grad_method.call0(py);
                }
            }
        });
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Sequential({} modules)", self.modules.len())
    }

    /// Get length (number of modules)
    fn __len__(&self) -> usize {
        self.modules.len()
    }

    /// Get module by index
    fn __getitem__(&self, index: usize) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            self.modules
                .get(index)
                .map(|obj| obj.clone_ref(py))
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of range")
                })
        })
    }

    /// Check if module is in training mode
    fn training(&self) -> bool {
        self.training
    }
}

/// ModuleList container - holds modules in a list
#[pyclass(name = "ModuleList", extends = PyModule)]
pub struct PyModuleList {
    modules: Vec<PyObject>,
    training: bool,
}

#[pymethods]
impl PyModuleList {
    #[new]
    fn new(modules: Option<Vec<PyObject>>) -> (Self, PyModule) {
        let modules = modules.unwrap_or_default();
        (
            Self {
                modules,
                training: true,
            },
            PyModule::new(),
        )
    }

    /// Append a module to the list
    fn append(&mut self, module: PyObject) {
        self.modules.push(module);
    }

    /// Extend the list with modules from another iterable
    fn extend(&mut self, modules: Vec<PyObject>) {
        self.modules.extend(modules);
    }

    /// Insert a module at the specified index
    fn insert(&mut self, index: usize, module: PyObject) {
        if index <= self.modules.len() {
            self.modules.insert(index, module);
        }
    }

    /// Get all parameters from all modules
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut all_params = Vec::new();
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(params_method) = module.getattr(py, "parameters") {
                    let params_result = params_method.call0(py)?;
                    if let Ok(params) = params_result.extract::<Vec<PyTensor>>(py) {
                        all_params.extend(params);
                    }
                }
            }
            Ok(all_params)
        })
    }

    /// Get all named parameters from all modules
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        let mut all_named_params = HashMap::new();
        Python::with_gil(|py| {
            for (i, module) in self.modules.iter().enumerate() {
                if let Ok(named_params_method) = module.getattr(py, "named_parameters") {
                    let named_params_result = named_params_method.call0(py)?;
                    if let Ok(named_params) =
                        named_params_result.extract::<HashMap<String, PyTensor>>(py)
                    {
                        for (name, param) in named_params {
                            all_named_params.insert(format!("{}.{}", i, name), param);
                        }
                    }
                }
            }
            Ok(all_named_params)
        })
    }

    /// Set training mode for all modules
    fn train(&mut self, mode: Option<bool>) {
        let mode = mode.unwrap_or(true);
        self.training = mode;
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(train_method) = module.getattr(py, "train") {
                    let _ = train_method.call1(py, (mode,));
                }
            }
        });
    }

    /// Set evaluation mode for all modules
    fn eval(&mut self) {
        self.training = false;
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(eval_method) = module.getattr(py, "eval") {
                    let _ = eval_method.call0(py);
                }
            }
        });
    }

    /// Move all modules to specified device
    fn to(&mut self, device: PyDevice) -> PyResult<()> {
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(to_method) = module.getattr(py, "to") {
                    to_method.call1(py, (device.clone(),))?;
                }
            }
            Ok(())
        })
    }

    /// Zero gradients for all modules
    fn zero_grad(&mut self) {
        Python::with_gil(|py| {
            for module in &self.modules {
                if let Ok(zero_grad_method) = module.getattr(py, "zero_grad") {
                    let _ = zero_grad_method.call0(py);
                }
            }
        });
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("ModuleList({} modules)", self.modules.len())
    }

    /// Get length (number of modules)
    fn __len__(&self) -> usize {
        self.modules.len()
    }

    /// Get module by index
    fn __getitem__(&self, index: usize) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            self.modules
                .get(index)
                .map(|obj| obj.clone_ref(py))
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of range")
                })
        })
    }

    /// Set module at index
    fn __setitem__(&mut self, index: usize, module: PyObject) -> PyResult<()> {
        if index < self.modules.len() {
            self.modules[index] = module;
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Index out of range",
            ))
        }
    }

    /// Check if module is in training mode
    fn training(&self) -> bool {
        self.training
    }
}
