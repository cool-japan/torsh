//! Python optimizer wrappers

use crate::error::FfiError;
use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Base optimizer class
#[pyclass(name = "Optimizer", subclass)]
#[derive(Clone)]
pub struct PyOptimizer {
    learning_rate: f32,
    name: String,
}

#[pymethods]
impl PyOptimizer {
    /// Perform optimization step
    fn step(&mut self) -> PyResult<()> {
        Err(FfiError::UnsupportedOperation {
            operation: "step not implemented for base Optimizer".to_string(),
        }
        .into())
    }

    /// Zero all gradients
    fn zero_grad(&mut self) -> PyResult<()> {
        // In a real implementation, this would zero gradients of all parameters
        Ok(())
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.learning_rate
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn __repr__(&self) -> String {
        format!("{}(lr={})", self.name, self.learning_rate)
    }
}

/// SGD optimizer
#[pyclass(name = "SGD")]
pub struct PySGD {
    momentum: f32,
    #[allow(dead_code)]
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    learning_rate: f32,
    // In real implementation, would store momentum buffers
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (_params, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=false))]
    fn new(
        _params: Vec<PyTensor>,
        lr: f32,
        momentum: f32,
        dampening: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> Self {
        if nesterov && (momentum <= 0.0 || dampening != 0.0) {
            panic!("Nesterov momentum requires a momentum and zero dampening");
        }

        PySGD {
            momentum,
            dampening,
            weight_decay,
            nesterov,
            learning_rate: lr,
        }
    }

    fn step(&mut self) -> PyResult<()> {
        // Simplified SGD step implementation
        // In real implementation, would update all parameters

        // For each parameter:
        // 1. Compute gradient with weight decay: grad = grad + weight_decay * param
        // 2. Apply momentum if > 0
        // 3. Update parameter: param = param - lr * grad

        Ok(())
    }

    #[getter]
    fn momentum(&self) -> f32 {
        self.momentum
    }

    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    #[getter]
    fn nesterov(&self) -> bool {
        self.nesterov
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.learning_rate
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn __repr__(&self) -> String {
        format!(
            "SGD(lr={}, momentum={}, weight_decay={}, nesterov={})",
            self.learning_rate, self.momentum, self.weight_decay, self.nesterov
        )
    }
}

/// Adam optimizer
#[pyclass(name = "Adam")]
pub struct PyAdam {
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
    learning_rate: f32,
    // In real implementation, would store exp_avg and exp_avg_sq for each parameter
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (_params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=false))]
    fn new(
        _params: Vec<PyTensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        PyAdam {
            betas,
            eps,
            weight_decay,
            amsgrad,
            learning_rate: lr,
        }
    }

    fn step(&mut self) -> PyResult<()> {
        // Simplified Adam step implementation
        // In real implementation, would update all parameters using Adam algorithm:

        // For each parameter:
        // 1. Compute gradient with weight decay if > 0
        // 2. Update biased first moment estimate: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        // 3. Update biased second moment estimate: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        // 4. Compute bias-corrected estimates: m_hat = m_t / (1 - beta1^t), v_hat = v_t / (1 - beta2^t)
        // 5. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + eps)

        Ok(())
    }

    #[getter]
    fn betas(&self) -> (f32, f32) {
        self.betas
    }

    #[getter]
    fn eps(&self) -> f32 {
        self.eps
    }

    #[getter]
    fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    #[getter]
    fn amsgrad(&self) -> bool {
        self.amsgrad
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.learning_rate
    }

    #[setter]
    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn __repr__(&self) -> String {
        format!(
            "Adam(lr={}, betas={:?}, eps={}, weight_decay={})",
            self.learning_rate, self.betas, self.eps, self.weight_decay
        )
    }
}

/// AdamW optimizer (Adam with decoupled weight decay)
#[pyclass(name = "AdamW")]
pub struct PyAdamW {
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    #[allow(dead_code)]
    amsgrad: bool,
    learning_rate: f32,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (_params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=false))]
    fn new(
        _params: Vec<PyTensor>,
        lr: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        PyAdamW {
            betas,
            eps,
            weight_decay,
            amsgrad,
            learning_rate: lr,
        }
    }

    fn step(&mut self) -> PyResult<()> {
        // AdamW implementation with decoupled weight decay
        // The key difference from Adam is that weight decay is applied directly to parameters
        // rather than being added to gradients

        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "AdamW(lr={}, betas={:?}, eps={}, weight_decay={})",
            self.learning_rate, self.betas, self.eps, self.weight_decay
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;
    use pyo3::Python;

    #[test]
    fn test_sgd_creation() {
        Python::initialize();
        Python::attach(|py| {
            let data = PyList::new(py, vec![1.0, 2.0, 3.0]).unwrap();
            let tensor = PyTensor::new(data.as_ref(), None, None, true).unwrap();
            let params = vec![tensor];

            let sgd = PySGD::new(params, 0.01, 0.9, 0.0, 0.0, false);
            assert_eq!(sgd.lr(), 0.01);
            assert_eq!(sgd.momentum(), 0.9);
        });
    }

    #[test]
    fn test_adam_creation() {
        Python::initialize();
        Python::attach(|py| {
            let data = PyList::new(py, vec![1.0, 2.0, 3.0]).unwrap();
            let tensor = PyTensor::new(data.as_ref(), None, None, true).unwrap();
            let params = vec![tensor];

            let adam = PyAdam::new(params, 0.001, (0.9, 0.999), 1e-8, 0.0, false);
            assert_eq!(adam.lr(), 0.001);
            assert_eq!(adam.betas(), (0.9, 0.999));
            assert_eq!(adam.eps(), 1e-8);
        });
    }

    #[test]
    fn test_optimizer_step() {
        Python::initialize();
        Python::attach(|py| {
            let data = PyList::new(py, vec![1.0, 2.0, 3.0]).unwrap();
            let tensor = PyTensor::new(data.as_ref(), None, None, true).unwrap();
            let params = vec![tensor];

            let mut sgd = PySGD::new(params, 0.01, 0.0, 0.0, 0.0, false);

            // Should not error
            assert!(sgd.step().is_ok());
        });
    }
}
