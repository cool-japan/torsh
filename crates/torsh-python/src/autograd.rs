//! Autograd module bindings with enhanced gradient computation support

use crate::{error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::PyRefMut;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// Global autograd state manager
pub struct AutogradState {
    enabled: bool,
    anomaly_detection: bool,
}

impl AutogradState {
    fn new() -> Self {
        Self {
            enabled: true,
            anomaly_detection: false,
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_anomaly_detection(&mut self, enabled: bool) {
        self.anomaly_detection = enabled;
    }

    fn is_anomaly_detection_enabled(&self) -> bool {
        self.anomaly_detection
    }
}

// Global state instance (in a real implementation, this would be more sophisticated)
thread_local! {
    static AUTOGRAD_STATE: RefCell<AutogradState> = RefCell::new(AutogradState::new());
}

/// Gradient computation context manager - disables gradient computation
#[pyclass(name = "no_grad")]
pub struct PyNoGrad {
    prev_state: bool,
}

#[pymethods]
impl PyNoGrad {
    #[new]
    fn new() -> Self {
        let prev_state = AUTOGRAD_STATE.with(|state| state.borrow().is_enabled());
        Self { prev_state }
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        // Disable gradients
        AUTOGRAD_STATE.with(|state| {
            slf.prev_state = state.borrow().is_enabled();
            state.borrow_mut().set_enabled(false);
        });
        slf
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        exc_type: Option<PyObject>,
        exc_val: Option<PyObject>,
        exc_tb: Option<PyObject>,
    ) -> PyResult<bool> {
        // Restore previous gradient state
        AUTOGRAD_STATE.with(|state| {
            state.borrow_mut().set_enabled(slf.prev_state);
        });
        Ok(false)
    }

    /// Check if we're currently in no_grad context
    #[staticmethod]
    fn is_enabled() -> bool {
        !AUTOGRAD_STATE.with(|state| state.borrow().is_enabled())
    }
}

/// Enable gradient computation context manager - forces gradient computation on
#[pyclass(name = "enable_grad")]
pub struct PyEnableGrad {
    prev_state: bool,
}

#[pymethods]
impl PyEnableGrad {
    #[new]
    fn new() -> Self {
        let prev_state = AUTOGRAD_STATE.with(|state| state.borrow().is_enabled());
        Self { prev_state }
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        // Enable gradients
        AUTOGRAD_STATE.with(|state| {
            slf.prev_state = state.borrow().is_enabled();
            state.borrow_mut().set_enabled(true);
        });
        slf
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        exc_type: Option<PyObject>,
        exc_val: Option<PyObject>,
        exc_tb: Option<PyObject>,
    ) -> PyResult<bool> {
        // Restore previous gradient state
        AUTOGRAD_STATE.with(|state| {
            state.borrow_mut().set_enabled(slf.prev_state);
        });
        Ok(false)
    }

    /// Check if we're currently in enable_grad context
    #[staticmethod]
    fn is_enabled() -> bool {
        AUTOGRAD_STATE.with(|state| state.borrow().is_enabled())
    }
}

/// Set gradient computation mode context manager - sets specific gradient mode
#[pyclass(name = "set_grad_enabled")]
pub struct PySetGradEnabled {
    mode: bool,
    prev_state: bool,
}

#[pymethods]
impl PySetGradEnabled {
    #[new]
    fn new(mode: bool) -> Self {
        let prev_state = AUTOGRAD_STATE.with(|state| state.borrow().is_enabled());
        Self { mode, prev_state }
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        // Set gradient mode
        AUTOGRAD_STATE.with(|state| {
            slf.prev_state = state.borrow().is_enabled();
            state.borrow_mut().set_enabled(slf.mode);
        });
        slf
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        exc_type: Option<PyObject>,
        exc_val: Option<PyObject>,
        exc_tb: Option<PyObject>,
    ) -> PyResult<bool> {
        // Restore previous gradient state
        AUTOGRAD_STATE.with(|state| {
            state.borrow_mut().set_enabled(slf.prev_state);
        });
        Ok(false)
    }
}

/// Anomaly detection context manager
#[pyclass(name = "detect_anomaly")]
pub struct PyDetectAnomaly {
    mode: bool,
    prev_state: bool,
}

#[pymethods]
impl PyDetectAnomaly {
    #[new]
    fn new(mode: bool) -> Self {
        let prev_state = AUTOGRAD_STATE.with(|state| state.borrow().is_anomaly_detection_enabled());
        Self { mode, prev_state }
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        // Set anomaly detection mode
        AUTOGRAD_STATE.with(|state| {
            slf.prev_state = state.borrow().is_anomaly_detection_enabled();
            state.borrow_mut().set_anomaly_detection(slf.mode);
        });
        slf
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        exc_type: Option<PyObject>,
        exc_val: Option<PyObject>,
        exc_tb: Option<PyObject>,
    ) -> PyResult<bool> {
        // Restore previous anomaly detection state
        AUTOGRAD_STATE.with(|state| {
            state.borrow_mut().set_anomaly_detection(slf.prev_state);
        });
        Ok(false)
    }

    /// Check if we're currently in anomaly detection mode
    #[staticmethod]
    fn is_enabled() -> bool {
        AUTOGRAD_STATE.with(|state| state.borrow().is_anomaly_detection_enabled())
    }
}

/// Function class for custom autograd functions
#[pyclass(name = "Function")]
pub struct PyFunction;

#[pymethods]
impl PyFunction {
    #[staticmethod]
    fn apply(inputs: Vec<PyTensor>) -> PyResult<PyTensor> {
        // Basic passthrough for now
        if inputs.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Function.apply requires at least one input",
            ));
        }
        Ok(inputs[0].clone())
    }

    // Note: forward and backward need more complex implementation for PyO3 0.25
    // fn forward(ctx: PyObject, inputs: Vec<PyTensor>) -> PyResult<PyTensor> {
    //     Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
    //         "Subclasses must implement forward method"
    //     ))
    // }
    //
    // fn backward(ctx: PyObject, grad_output: &PyTensor) -> PyResult<Vec<Option<PyTensor>>> {
    //     Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
    //         "Subclasses must implement backward method"
    //     ))
    // }
}

/// Gradient computation utilities
pub struct AutogradUtils;

impl AutogradUtils {
    /// Compute gradients with respect to inputs
    pub fn grad(
        outputs: Vec<PyTensor>,
        inputs: Vec<PyTensor>,
        grad_outputs: Option<Vec<Option<PyTensor>>>,
        retain_graph: Option<bool>,
        create_graph: Option<bool>,
        only_inputs: Option<bool>,
        allow_unused: Option<bool>,
    ) -> PyResult<Vec<Option<PyTensor>>> {
        // For now, implement basic backward pass
        if outputs.len() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Multiple outputs not yet supported",
            ));
        }

        let output = &outputs[0];
        py_result!(output.tensor.backward())?;

        // Return gradients for inputs
        let mut grads = Vec::new();
        for input in inputs {
            grads.push(input.tensor.grad().map(|g| PyTensor { tensor: g }));
        }

        Ok(grads)
    }
}

use pyo3::types::{PyModule, PyModuleMethods};

/// Register autograd module
pub fn register_autograd_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add context managers
    m.add_class::<PyNoGrad>()?;
    m.add_class::<PyEnableGrad>()?;
    m.add_class::<PySetGradEnabled>()?;
    m.add_class::<PyDetectAnomaly>()?;

    // Add Function class
    m.add_class::<PyFunction>()?;

    // Add utility functions
    #[pyfunction]
    fn grad(
        outputs: Vec<PyTensor>,
        inputs: Vec<PyTensor>,
        grad_outputs: Option<Vec<Option<PyTensor>>>,
        retain_graph: Option<bool>,
        create_graph: Option<bool>,
        only_inputs: Option<bool>,
        allow_unused: Option<bool>,
    ) -> PyResult<Vec<Option<PyTensor>>> {
        AutogradUtils::grad(
            outputs,
            inputs,
            grad_outputs,
            retain_graph,
            create_graph,
            only_inputs,
            allow_unused,
        )
    }

    m.add_function(wrap_pyfunction!(grad, m)?)?;

    #[pyfunction]
    fn backward(
        tensors: Vec<PyTensor>,
        grad_tensors: Option<Vec<Option<PyTensor>>>,
        retain_graph: Option<bool>,
        create_graph: Option<bool>,
        inputs: Option<Vec<PyTensor>>,
    ) -> PyResult<()> {
        // Implement backward pass
        for tensor in tensors {
            py_result!(tensor.tensor.backward())?;
        }
        Ok(())
    }

    m.add_function(wrap_pyfunction!(backward, m)?)?;

    #[pyfunction]
    fn is_grad_enabled() -> bool {
        AUTOGRAD_STATE.with(|state| state.borrow().is_enabled())
    }

    #[pyfunction]
    fn set_grad_enabled(mode: bool) {
        AUTOGRAD_STATE.with(|state| {
            state.borrow_mut().set_enabled(mode);
        });
    }

    #[pyfunction]
    fn detect_anomaly(mode: Option<bool>) -> PyResult<PyDetectAnomaly> {
        let mode = mode.unwrap_or(true);
        Ok(PyDetectAnomaly::new(mode))
    }

    #[pyfunction]
    fn is_anomaly_detection_enabled() -> bool {
        AUTOGRAD_STATE.with(|state| state.borrow().is_anomaly_detection_enabled())
    }

    #[pyfunction]
    fn set_anomaly_detection(mode: bool) {
        AUTOGRAD_STATE.with(|state| {
            state.borrow_mut().set_anomaly_detection(mode);
        });
    }

    m.add_function(wrap_pyfunction!(is_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(detect_anomaly, m)?)?;
    m.add_function(wrap_pyfunction!(is_anomaly_detection_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_anomaly_detection, m)?)?;

    Ok(())
}
