//! Adam and AdamW optimizers

use super::base::{create_param_group, extract_parameters, PyOptimizer};
use crate::{error::PyResult, tensor::PyTensor};
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_optim::{Adam, AdamW, Optimizer};

/// Adam optimizer - Adaptive Moment Estimation
#[pyclass(name = "Adam", extends = PyOptimizer)]
pub struct PyAdam {
    adam: Adam,
    param_groups: Vec<HashMap<String, Py<PyAny>>>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

#[pymethods]
impl PyAdam {
    #[new]
    fn new(
        params: Vec<PyTensor>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: Option<bool>,
    ) -> (Self, PyOptimizer) {
        let lr = lr.unwrap_or(0.001);
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let amsgrad = amsgrad.unwrap_or(false);

        // Extract tensor parameters and wrap in Arc<RwLock>
        let tensor_params = extract_parameters(params.clone()).unwrap();
        let wrapped_params: Vec<Arc<RwLock<_>>> = tensor_params
            .into_iter()
            .map(|tensor| Arc::new(RwLock::new(tensor)))
            .collect();
        let adam = Adam::new(
            wrapped_params,
            Some(lr),
            Some(betas),
            Some(eps),
            Some(weight_decay),
            amsgrad,
        );

        // Create parameter groups
        let mut param_group_data = HashMap::new();
        Python::attach(|py| {
            param_group_data.insert(
                "betas".to_string(),
                betas.into_pyobject(py).unwrap().into_any().unbind(),
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
                "amsgrad".to_string(),
                PyBool::new(py, amsgrad).to_owned().into(),
            );
        });

        let param_groups = vec![create_param_group(params, lr, param_group_data).unwrap()];

        (
            Self {
                adam,
                param_groups,
                lr,
                betas,
                eps,
                weight_decay,
                amsgrad,
            },
            PyOptimizer {},
        )
    }

    /// Perform a single optimization step
    fn step(&mut self) -> PyResult<()> {
        self.adam.step().map_err(|e| {
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
        self.adam.zero_grad();
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
                0i64.into_pyobject(py).unwrap().into_any().unbind(),
            );
            state.insert(
                "exp_avg".to_string(),
                "{}".into_pyobject(py).unwrap().into_any().unbind(),
            );
            state.insert(
                "exp_avg_sq".to_string(),
                "{}".into_pyobject(py).unwrap().into_any().unbind(),
            );
            if self.amsgrad {
                state.insert(
                    "max_exp_avg_sq".to_string(),
                    "{}".into_pyobject(py).unwrap().into_any().unbind(),
                );
            }
        });
        Ok(state)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Adam(lr={}, betas={:?}, eps={}, weight_decay={}, amsgrad={})",
            self.lr, self.betas, self.eps, self.weight_decay, self.amsgrad
        )
    }

    /// Get defaults
    fn defaults(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut defaults = HashMap::new();
        Python::attach(|py| {
            defaults.insert(
                "lr".to_string(),
                self.lr.into_pyobject(py).unwrap().into_any().unbind(),
            );
            defaults.insert(
                "betas".to_string(),
                self.betas.into_pyobject(py).unwrap().into_any().unbind(),
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
                "amsgrad".to_string(),
                PyBool::new(py, self.amsgrad).to_owned().into(),
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
                    lr.into_pyobject(py).unwrap().into_any().unbind(),
                );
            }
        });
    }

    /// Get betas
    #[getter]
    fn betas(&self) -> (f32, f32) {
        self.betas
    }

    /// Get eps
    #[getter]
    fn eps(&self) -> f32 {
        self.eps
    }

    /// Get weight decay
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get amsgrad flag
    #[getter]
    fn amsgrad(&self) -> bool {
        self.amsgrad
    }
}

/// AdamW optimizer - Adam with decoupled weight decay
#[pyclass(name = "AdamW", extends = PyOptimizer)]
pub struct PyAdamW {
    adamw: AdamW,
    param_groups: Vec<HashMap<String, Py<PyAny>>>,
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

#[pymethods]
impl PyAdamW {
    #[new]
    fn new(
        params: Vec<PyTensor>,
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: Option<bool>,
    ) -> (Self, PyOptimizer) {
        let lr = lr.unwrap_or(0.001);
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.01);
        let amsgrad = amsgrad.unwrap_or(false);

        // Extract tensor parameters and wrap in Arc<RwLock>
        let tensor_params = extract_parameters(params.clone()).unwrap();
        let wrapped_params: Vec<Arc<RwLock<_>>> = tensor_params
            .into_iter()
            .map(|tensor| Arc::new(RwLock::new(tensor)))
            .collect();
        let adamw = AdamW::new(
            wrapped_params,
            Some(lr),
            Some(betas),
            Some(eps),
            Some(weight_decay),
            amsgrad,
        );

        // Create parameter groups
        let mut param_group_data = HashMap::new();
        Python::attach(|py| {
            param_group_data.insert(
                "betas".to_string(),
                betas.into_pyobject(py).unwrap().into_any().unbind(),
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
                "amsgrad".to_string(),
                PyBool::new(py, amsgrad).to_owned().into(),
            );
        });

        let param_groups = vec![create_param_group(params, lr, param_group_data).unwrap()];

        (
            Self {
                adamw,
                param_groups,
                lr,
                betas,
                eps,
                weight_decay,
                amsgrad,
            },
            PyOptimizer {},
        )
    }

    /// Perform a single optimization step
    fn step(&mut self) -> PyResult<()> {
        self.adamw.step().map_err(|e| {
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
        self.adamw.zero_grad();
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
                0i64.into_pyobject(py).unwrap().into_any().unbind(),
            );
            state.insert(
                "exp_avg".to_string(),
                "{}".into_pyobject(py).unwrap().into_any().unbind(),
            );
            state.insert(
                "exp_avg_sq".to_string(),
                "{}".into_pyobject(py).unwrap().into_any().unbind(),
            );
            if self.amsgrad {
                state.insert(
                    "max_exp_avg_sq".to_string(),
                    "{}".into_pyobject(py).unwrap().into_any().unbind(),
                );
            }
        });
        Ok(state)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "AdamW(lr={}, betas={:?}, eps={}, weight_decay={}, amsgrad={})",
            self.lr, self.betas, self.eps, self.weight_decay, self.amsgrad
        )
    }

    /// Get defaults
    fn defaults(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut defaults = HashMap::new();
        Python::attach(|py| {
            defaults.insert(
                "lr".to_string(),
                self.lr.into_pyobject(py).unwrap().into_any().unbind(),
            );
            defaults.insert(
                "betas".to_string(),
                self.betas.into_pyobject(py).unwrap().into_any().unbind(),
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
                "amsgrad".to_string(),
                PyBool::new(py, self.amsgrad).to_owned().into(),
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
                    lr.into_pyobject(py).unwrap().into_any().unbind(),
                );
            }
        });
    }

    /// Get betas
    #[getter]
    fn betas(&self) -> (f32, f32) {
        self.betas
    }

    /// Get eps
    #[getter]
    fn eps(&self) -> f32 {
        self.eps
    }

    /// Get weight decay
    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get amsgrad flag
    #[getter]
    fn amsgrad(&self) -> bool {
        self.amsgrad
    }
}
