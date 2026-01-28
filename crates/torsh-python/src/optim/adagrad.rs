//! AdaGrad optimizer

use super::base::{create_param_group, extract_parameters, PyOptimizer};
use crate::{error::PyResult, tensor::PyTensor};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::collections::HashMap;
use std::sync::Arc;
use torsh_optim::{adagrad::AdaGrad, Optimizer};

/// AdaGrad optimizer - Adaptive Gradient Algorithm
#[pyclass(name = "Adagrad", extends = PyOptimizer)]
pub struct PyAdaGrad {
    adagrad: AdaGrad,
    param_groups: Vec<HashMap<String, Py<PyAny>>>,
    lr: f32,
    lr_decay: f32,
    weight_decay: f32,
    eps: f32,
}

#[pymethods]
impl PyAdaGrad {
    #[new]
    fn new(
        params: Vec<PyTensor>,
        lr: Option<f32>,
        lr_decay: Option<f32>,
        weight_decay: Option<f32>,
        eps: Option<f32>,
    ) -> (Self, PyOptimizer) {
        let lr = lr.unwrap_or(0.01);
        let lr_decay = lr_decay.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let eps = eps.unwrap_or(1e-10);

        // Extract tensor parameters and wrap in Arc<RwLock>
        let tensor_params =
            extract_parameters(params.clone()).expect("parameter extraction should succeed");
        let wrapped_params: Vec<Arc<RwLock<_>>> = tensor_params
            .into_iter()
            .map(|tensor| Arc::new(RwLock::new(tensor)))
            .collect();
        let adagrad = AdaGrad::new(
            wrapped_params,
            Some(lr),
            Some(lr_decay),
            Some(weight_decay),
            Some(0.0),
            Some(eps),
        );

        // Create parameter groups
        let mut param_group_data = HashMap::new();
        Python::attach(|py| {
            param_group_data.insert(
                "lr_decay".to_string(),
                lr_decay
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            param_group_data.insert(
                "weight_decay".to_string(),
                weight_decay
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            param_group_data.insert(
                "eps".to_string(),
                eps.into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
        });

        let param_groups = vec![create_param_group(params, lr, param_group_data)
            .expect("param group creation should succeed")];

        (
            Self {
                adagrad,
                param_groups,
                lr,
                lr_decay,
                weight_decay,
                eps,
            },
            PyOptimizer {},
        )
    }

    /// Perform a single optimization step
    fn step(&mut self) -> PyResult<()> {
        self.adagrad.step().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Optimizer step failed: {}",
                e
            ))
        })?;
        Ok(())
    }

    /// Zero out gradients of all parameters
    fn zero_grad(&mut self, set_to_none: Option<bool>) {
        let _set_to_none = set_to_none.unwrap_or(false);
        self.adagrad.zero_grad();
    }

    /// Get parameter groups
    fn param_groups(&self) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
        // Manual clone since Py<PyAny> doesn't implement Clone
        Python::attach(|py| {
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
    fn state(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut state = HashMap::new();
        Python::attach(|py| {
            state.insert(
                "step".to_string(),
                0i64.into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            state.insert(
                "sum".to_string(),
                "{}".into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
        });
        Ok(state)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Adagrad(lr={}, lr_decay={}, eps={}, weight_decay={})",
            self.lr, self.lr_decay, self.eps, self.weight_decay
        )
    }

    /// Get defaults
    fn defaults(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut defaults = HashMap::new();
        Python::attach(|py| {
            defaults.insert(
                "lr".to_string(),
                self.lr
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "lr_decay".to_string(),
                self.lr_decay
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "weight_decay".to_string(),
                self.weight_decay
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
            );
            defaults.insert(
                "eps".to_string(),
                self.eps
                    .into_pyobject(py)
                    .expect("Python object conversion should succeed")
                    .into_any()
                    .unbind(),
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
        Python::attach(|py| {
            for param_group in &mut self.param_groups {
                param_group.insert(
                    "lr".to_string(),
                    lr.into_pyobject(py)
                        .expect("Python object conversion should succeed")
                        .into_any()
                        .unbind(),
                );
            }
        });
    }

    /// Get learning rate decay
    #[getter]
    fn lr_decay(&self) -> f32 {
        self.lr_decay
    }

    /// Get weight decay
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get epsilon
    #[getter]
    fn eps(&self) -> f32 {
        self.eps
    }
}
