//! RMSprop optimizer

use super::base::{create_param_group, extract_parameters, PyOptimizer};
use crate::{error::PyResult, py_optimizer_result, py_result, tensor::PyTensor};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyBool;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_optim::{rmsprop::RMSprop, Optimizer};

/// RMSprop optimizer - Root Mean Square Propagation
#[pyclass(name = "RMSprop", extends = PyOptimizer)]
pub struct PyRMSprop {
    rmsprop: RMSprop,
    param_groups: Vec<HashMap<String, PyObject>>,
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
}

#[pymethods]
impl PyRMSprop {
    #[new]
    fn new(
        params: Vec<PyTensor>,
        lr: Option<f32>,
        alpha: Option<f32>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        momentum: Option<f32>,
        centered: Option<bool>,
    ) -> (Self, PyOptimizer) {
        let lr = lr.unwrap_or(0.01);
        let alpha = alpha.unwrap_or(0.99);
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let momentum = momentum.unwrap_or(0.0);
        let centered = centered.unwrap_or(false);

        // Extract tensor parameters and wrap in Arc<RwLock>
        let tensor_params = extract_parameters(params.clone()).unwrap();
        let wrapped_params: Vec<Arc<RwLock<_>>> = tensor_params
            .into_iter()
            .map(|tensor| Arc::new(RwLock::new(tensor)))
            .collect();
        let rmsprop = RMSprop::new(
            wrapped_params,
            Some(lr),
            Some(alpha),
            Some(eps),
            Some(weight_decay),
            Some(momentum),
            centered,
        );

        // Create parameter groups
        let mut param_group_data = HashMap::new();
        Python::with_gil(|py| {
            param_group_data.insert(
                "alpha".to_string(),
                alpha.into_pyobject(py).unwrap().into_any().unbind(),
            );
            param_group_data.insert(
                "eps".to_string(),
                eps.into_pyobject(py).unwrap().into_any().unbind(),
            );
            param_group_data.insert(
                "weight_decay".to_string(),
                weight_decay.into_pyobject(py).unwrap().into_any().unbind(),
            );
            param_group_data.insert(
                "momentum".to_string(),
                momentum.into_pyobject(py).unwrap().into_any().unbind(),
            );
            param_group_data.insert(
                "centered".to_string(),
                PyBool::new(py, centered).to_owned().into(),
            );
        });

        let param_groups = vec![create_param_group(params, lr, param_group_data).unwrap()];

        (
            Self {
                rmsprop,
                param_groups,
                lr,
                alpha,
                eps,
                weight_decay,
                momentum,
                centered,
            },
            PyOptimizer {},
        )
    }

    /// Perform a single optimization step
    fn step(&mut self) -> PyResult<()> {
        py_optimizer_result!(self.rmsprop.step())?;
        Ok(())
    }

    /// Zero out gradients of all parameters
    fn zero_grad(&mut self, set_to_none: Option<bool>) {
        let _set_to_none = set_to_none.unwrap_or(false);
        self.rmsprop.zero_grad();
    }

    /// Get parameter groups
    fn param_groups(&self) -> PyResult<Vec<HashMap<String, PyObject>>> {
        // Manual clone since Py<PyAny> doesn't implement Clone
        Python::with_gil(|py| {
            let cloned_groups = self
                .param_groups
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                        .collect()
                })
                .collect();
            Ok(cloned_groups)
        })
    }

    /// Get current state
    fn state(&self) -> PyResult<HashMap<String, PyObject>> {
        let mut state = HashMap::new();
        Python::with_gil(|py| {
            state.insert(
                "step".to_string(),
                0i64.into_pyobject(py).unwrap().into_any().unbind(),
            );
            state.insert(
                "square_avg".to_string(),
                "{}".into_pyobject(py).unwrap().into_any().unbind(),
            );
            if self.momentum > 0.0 {
                state.insert(
                    "momentum_buffer".to_string(),
                    "{}".into_pyobject(py).unwrap().into_any().unbind(),
                );
            }
            if self.centered {
                state.insert(
                    "grad_avg".to_string(),
                    "{}".into_pyobject(py).unwrap().into_any().unbind(),
                );
            }
        });
        Ok(state)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "RMSprop(lr={}, alpha={}, eps={}, weight_decay={}, momentum={}, centered={})",
            self.lr, self.alpha, self.eps, self.weight_decay, self.momentum, self.centered
        )
    }

    /// Get defaults
    fn defaults(&self) -> PyResult<HashMap<String, PyObject>> {
        let mut defaults = HashMap::new();
        Python::with_gil(|py| {
            defaults.insert(
                "lr".to_string(),
                self.lr.into_pyobject(py).unwrap().into_any().unbind(),
            );
            defaults.insert(
                "alpha".to_string(),
                self.alpha.into_pyobject(py).unwrap().into_any().unbind(),
            );
            defaults.insert(
                "eps".to_string(),
                self.eps.into_pyobject(py).unwrap().into_any().unbind(),
            );
            defaults.insert(
                "weight_decay".to_string(),
                self.weight_decay
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "momentum".to_string(),
                self.momentum.into_pyobject(py).unwrap().into_any().unbind(),
            );
            defaults.insert(
                "centered".to_string(),
                PyBool::new(py, self.centered).to_owned().into(),
            );
        });
        Ok(defaults)
    }

    /// Get learning rate
    #[getter]
    fn lr(&self) -> f32 {
        self.lr
    }

    /// Set learning rate
    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        Python::with_gil(|py| {
            for param_group in &mut self.param_groups {
                param_group.insert(
                    "lr".to_string(),
                    lr.into_pyobject(py).unwrap().into_any().unbind(),
                );
            }
        });
    }

    /// Get alpha (smoothing constant)
    #[getter]
    fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Get epsilon
    #[getter]
    fn eps(&self) -> f32 {
        self.eps
    }

    /// Get weight decay
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get momentum
    #[getter]
    fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Get centered flag
    #[getter]
    fn centered(&self) -> bool {
        self.centered
    }
}
