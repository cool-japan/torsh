//! Dropout and regularization layers

use super::module::PyModule;
use crate::{error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Dropout layer
#[pyclass(name = "Dropout", extends = PyModule)]
pub struct PyDropout {
    p: f32,
    inplace: bool,
    training: bool,
}

#[pymethods]
impl PyDropout {
    #[new]
    fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<(Self, PyModule)> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        if !(0.0..=1.0).contains(&p) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout probability has to be between 0 and 1, but got {p}",
            ));
        }

        Ok((
            Self {
                p,
                inplace,
                training: true,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through dropout
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // ✅ Proper dropout implementation with random mask
        if !self.training || self.p == 0.0 {
            // In eval mode or p=0, return input unchanged
            return Ok(PyTensor {
                tensor: input.tensor.clone(),
            });
        }

        if self.p == 1.0 {
            // All values dropped
            let zeros = py_result!(torsh_tensor::creation::zeros_like(&input.tensor))?;
            return Ok(PyTensor { tensor: zeros });
        }

        // ✅ SciRS2 POLICY: Use scirs2_core::random for RNG
        use scirs2_core::random::Distribution;
        use scirs2_core::random::{thread_rng, Uniform};

        let mut rng = thread_rng();
        let dist = Uniform::new(0.0_f32, 1.0_f32).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create uniform distribution: {}",
                e
            ))
        })?;

        let mut data = py_result!(input.tensor.data())?;
        let scale = 1.0 / (1.0 - self.p);

        for val in data.iter_mut() {
            if dist.sample(&mut rng) < self.p {
                *val = 0.0;
            } else {
                *val *= scale; // Scale to maintain expected value
            }
        }

        let shape = input.tensor.shape().dims().to_vec();
        let result = py_result!(torsh_tensor::Tensor::from_data(
            data,
            shape,
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
    }

    /// Get layer parameters (Dropout has no parameters)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    /// Get named parameters (Dropout has no parameters)
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        Ok(HashMap::new())
    }

    /// Set training mode
    fn train(&mut self, mode: Option<bool>) -> PyResult<()> {
        self.training = mode.unwrap_or(true);
        Ok(())
    }

    /// Set evaluation mode
    fn eval(&mut self) -> PyResult<()> {
        self.training = false;
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Dropout(p={}, inplace={})", self.p, self.inplace)
    }
}

/// 2D Dropout layer
#[pyclass(name = "Dropout2d", extends = PyModule)]
pub struct PyDropout2d {
    p: f32,
    inplace: bool,
    training: bool,
}

#[pymethods]
impl PyDropout2d {
    #[new]
    fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<(Self, PyModule)> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        if !(0.0..=1.0).contains(&p) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout probability has to be between 0 and 1, but got {p}",
            ));
        }

        Ok((
            Self {
                p,
                inplace,
                training: true,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through 2D dropout
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // ✅ Proper 2D dropout implementation - drops entire channels
        if !self.training || self.p == 0.0 {
            return Ok(PyTensor {
                tensor: input.tensor.clone(),
            });
        }

        if self.p == 1.0 {
            let zeros = py_result!(torsh_tensor::creation::zeros_like(&input.tensor))?;
            return Ok(PyTensor { tensor: zeros });
        }

        // ✅ SciRS2 POLICY: Use scirs2_core::random for RNG
        use scirs2_core::random::Distribution;
        use scirs2_core::random::{thread_rng, Uniform};

        let shape = input.tensor.shape().dims().to_vec();
        if shape.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dropout2d expects at least 2D input",
            ));
        }

        let mut rng = thread_rng();
        let dist = Uniform::new(0.0_f32, 1.0_f32).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create uniform distribution: {}",
                e
            ))
        })?;

        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        let mut data = py_result!(input.tensor.data())?;
        let scale = 1.0 / (1.0 - self.p);

        // Drop entire channels
        for b in 0..batch_size {
            for c in 0..channels {
                if dist.sample(&mut rng) < self.p {
                    // Drop entire channel
                    let start = (b * channels + c) * spatial_size;
                    let end = start + spatial_size;
                    for val in &mut data[start..end] {
                        *val = 0.0;
                    }
                } else {
                    // Scale channel
                    let start = (b * channels + c) * spatial_size;
                    let end = start + spatial_size;
                    for val in &mut data[start..end] {
                        *val *= scale;
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            data,
            shape.to_vec(),
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
    }

    /// Get layer parameters (Dropout2d has no parameters)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    /// Get named parameters (Dropout2d has no parameters)
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        Ok(HashMap::new())
    }

    /// Set training mode
    fn train(&mut self, mode: Option<bool>) -> PyResult<()> {
        self.training = mode.unwrap_or(true);
        Ok(())
    }

    /// Set evaluation mode
    fn eval(&mut self) -> PyResult<()> {
        self.training = false;
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Dropout2d(p={}, inplace={})", self.p, self.inplace)
    }
}

/// 3D Dropout layer
#[pyclass(name = "Dropout3d", extends = PyModule)]
pub struct PyDropout3d {
    p: f32,
    inplace: bool,
    training: bool,
}

#[pymethods]
impl PyDropout3d {
    #[new]
    fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<(Self, PyModule)> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        if !(0.0..=1.0).contains(&p) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout probability has to be between 0 and 1, but got {p}",
            ));
        }

        Ok((
            Self {
                p,
                inplace,
                training: true,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through 3D dropout
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // ✅ Proper 3D dropout implementation - drops entire channels (same as 2D)
        if !self.training || self.p == 0.0 {
            return Ok(PyTensor {
                tensor: input.tensor.clone(),
            });
        }

        if self.p == 1.0 {
            let zeros = py_result!(torsh_tensor::creation::zeros_like(&input.tensor))?;
            return Ok(PyTensor { tensor: zeros });
        }

        // ✅ SciRS2 POLICY: Use scirs2_core::random for RNG
        use scirs2_core::random::Distribution;
        use scirs2_core::random::{thread_rng, Uniform};

        let shape = input.tensor.shape().dims().to_vec();
        if shape.len() < 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dropout3d expects at least 3D input",
            ));
        }

        let mut rng = thread_rng();
        let dist = Uniform::new(0.0_f32, 1.0_f32).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create uniform distribution: {}",
                e
            ))
        })?;

        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_size: usize = shape[2..].iter().product();

        let mut data = py_result!(input.tensor.data())?;
        let scale = 1.0 / (1.0 - self.p);

        // Drop entire channels
        for b in 0..batch_size {
            for c in 0..channels {
                if dist.sample(&mut rng) < self.p {
                    // Drop entire channel
                    let start = (b * channels + c) * spatial_size;
                    let end = start + spatial_size;
                    for val in &mut data[start..end] {
                        *val = 0.0;
                    }
                } else {
                    // Scale channel
                    let start = (b * channels + c) * spatial_size;
                    let end = start + spatial_size;
                    for val in &mut data[start..end] {
                        *val *= scale;
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            data,
            shape.to_vec(),
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
    }

    /// Get layer parameters (Dropout3d has no parameters)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    /// Get named parameters (Dropout3d has no parameters)
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        Ok(HashMap::new())
    }

    /// Set training mode
    fn train(&mut self, mode: Option<bool>) -> PyResult<()> {
        self.training = mode.unwrap_or(true);
        Ok(())
    }

    /// Set evaluation mode
    fn eval(&mut self) -> PyResult<()> {
        self.training = false;
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Dropout3d(p={}, inplace={})", self.p, self.inplace)
    }
}

/// Alpha Dropout layer (for SELU activation)
#[pyclass(name = "AlphaDropout", extends = PyModule)]
pub struct PyAlphaDropout {
    p: f32,
    inplace: bool,
    training: bool,
}

#[pymethods]
impl PyAlphaDropout {
    #[new]
    fn new(p: Option<f32>, inplace: Option<bool>) -> PyResult<(Self, PyModule)> {
        let p = p.unwrap_or(0.5);
        let inplace = inplace.unwrap_or(false);

        if !(0.0..=1.0).contains(&p) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dropout probability has to be between 0 and 1, but got {p}",
            ));
        }

        Ok((
            Self {
                p,
                inplace,
                training: true,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through alpha dropout
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // ✅ Proper AlphaDropout implementation for SELU networks
        if !self.training || self.p == 0.0 {
            return Ok(PyTensor {
                tensor: input.tensor.clone(),
            });
        }

        if self.p == 1.0 {
            // When p=1, all values are set to alpha'
            let alpha_prime = -1.7580993408473766_f32;
            let mut data = py_result!(input.tensor.data())?;
            for val in data.iter_mut() {
                *val = alpha_prime;
            }
            let shape = input.tensor.shape().dims().to_vec();
            let result = py_result!(torsh_tensor::Tensor::from_data(
                data,
                shape,
                input.tensor.device()
            ))?;
            return Ok(PyTensor { tensor: result });
        }

        // ✅ SciRS2 POLICY: Use scirs2_core::random for RNG
        use scirs2_core::random::Distribution;
        use scirs2_core::random::{thread_rng, Uniform};

        // SELU constants
        let _alpha = 1.6732632423543772_f32;
        let alpha_prime = -1.7580993408473766_f32; // -alpha * lambda where lambda = 1.0507

        // Calculate affine transformation parameters to maintain self-normalization
        let a = ((1.0 - self.p) * (1.0 + self.p * alpha_prime * alpha_prime)).sqrt();
        let b = -a * alpha_prime * self.p;

        let mut rng = thread_rng();
        let dist = Uniform::new(0.0_f32, 1.0_f32).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to create uniform distribution: {}",
                e
            ))
        })?;

        let mut data = py_result!(input.tensor.data())?;

        for val in data.iter_mut() {
            if dist.sample(&mut rng) < self.p {
                // Set to alpha' and apply affine transformation
                *val = (*val * 0.0 + alpha_prime) * a + b;
            } else {
                // Keep value and apply affine transformation
                *val = *val * a + b;
            }
        }

        let shape = input.tensor.shape().dims().to_vec();
        let result = py_result!(torsh_tensor::Tensor::from_data(
            data,
            shape,
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
    }

    /// Get layer parameters (AlphaDropout has no parameters)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    /// Get named parameters (AlphaDropout has no parameters)
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        Ok(HashMap::new())
    }

    /// Set training mode
    fn train(&mut self, mode: Option<bool>) -> PyResult<()> {
        self.training = mode.unwrap_or(true);
        Ok(())
    }

    /// Set evaluation mode
    fn eval(&mut self) -> PyResult<()> {
        self.training = false;
        Ok(())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("AlphaDropout(p={}, inplace={})", self.p, self.inplace)
    }
}
