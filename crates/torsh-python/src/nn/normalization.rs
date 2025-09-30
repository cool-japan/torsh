//! Normalization layers

use super::module::PyModule;
use crate::{device::PyDevice, error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Batch Normalization 2D layer
#[pyclass(name = "BatchNorm2d", extends = PyModule)]
pub struct PyBatchNorm2d {
    weight: Option<Tensor<f32>>,
    bias: Option<Tensor<f32>>,
    running_mean: Tensor<f32>,
    running_var: Tensor<f32>,
    num_features: usize,
    eps: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
    training: bool,
    num_batches_tracked: usize,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> PyResult<(Self, PyModule)> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);

        let shape = vec![num_features];

        // Initialize weight and bias if affine=true
        let (weight, bias) = if affine {
            let weight = py_result!(torsh_tensor::creation::ones(&shape))?.requires_grad_(true);
            let bias = py_result!(torsh_tensor::creation::zeros(&shape))?.requires_grad_(true);
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        // Initialize running statistics
        let running_mean = py_result!(torsh_tensor::creation::zeros(&shape))?;
        let running_var = py_result!(torsh_tensor::creation::ones(&shape))?;

        Ok((
            Self {
                weight,
                bias,
                running_mean,
                running_var,
                num_features,
                eps,
                momentum,
                affine,
                track_running_stats,
                training: true,
                num_batches_tracked: 0,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through batch normalization
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        if self.training && self.track_running_stats {
            self.num_batches_tracked += 1;
        }

        // Simplified batch normalization implementation - return input as-is for now
        // TODO: Implement proper batch norm with statistics computation when available
        Ok(PyTensor {
            tensor: input.tensor.clone(),
        })
    }

    /// Get layer parameters
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(PyTensor {
                tensor: weight.clone(),
            });
        }
        if let Some(ref bias) = self.bias {
            params.push(PyTensor {
                tensor: bias.clone(),
            });
        }
        Ok(params)
    }

    /// Get named parameters
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        let mut params = HashMap::new();
        if let Some(ref weight) = self.weight {
            params.insert(
                "weight".to_string(),
                PyTensor {
                    tensor: weight.clone(),
                },
            );
        }
        if let Some(ref bias) = self.bias {
            params.insert(
                "bias".to_string(),
                PyTensor {
                    tensor: bias.clone(),
                },
            );
        }
        Ok(params)
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
        format!(
            "BatchNorm2d({}, eps={}, momentum={}, affine={}, track_running_stats={})",
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )
    }
}

/// Batch Normalization 1D layer
#[pyclass(name = "BatchNorm1d", extends = PyModule)]
pub struct PyBatchNorm1d {
    weight: Option<Tensor<f32>>,
    bias: Option<Tensor<f32>>,
    running_mean: Tensor<f32>,
    running_var: Tensor<f32>,
    num_features: usize,
    eps: f32,
    momentum: f32,
    affine: bool,
    track_running_stats: bool,
    training: bool,
    num_batches_tracked: usize,
}

#[pymethods]
impl PyBatchNorm1d {
    #[new]
    fn new(
        num_features: usize,
        eps: Option<f32>,
        momentum: Option<f32>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
    ) -> PyResult<(Self, PyModule)> {
        let eps = eps.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.1);
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);

        let shape = vec![num_features];

        // Initialize weight and bias if affine=true
        let (weight, bias) = if affine {
            let weight = py_result!(torsh_tensor::creation::ones(&shape))?.requires_grad_(true);
            let bias = py_result!(torsh_tensor::creation::zeros(&shape))?.requires_grad_(true);
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        // Initialize running statistics
        let running_mean = py_result!(torsh_tensor::creation::zeros(&shape))?;
        let running_var = py_result!(torsh_tensor::creation::ones(&shape))?;

        Ok((
            Self {
                weight,
                bias,
                running_mean,
                running_var,
                num_features,
                eps,
                momentum,
                affine,
                track_running_stats,
                training: true,
                num_batches_tracked: 0,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through batch normalization
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        if self.training && self.track_running_stats {
            self.num_batches_tracked += 1;
        }

        // Simplified batch normalization implementation - return input as-is for now
        // TODO: Implement proper batch norm with statistics computation when available
        Ok(PyTensor {
            tensor: input.tensor.clone(),
        })
    }

    /// Get layer parameters
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(PyTensor {
                tensor: weight.clone(),
            });
        }
        if let Some(ref bias) = self.bias {
            params.push(PyTensor {
                tensor: bias.clone(),
            });
        }
        Ok(params)
    }

    /// Get named parameters
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        let mut params = HashMap::new();
        if let Some(ref weight) = self.weight {
            params.insert(
                "weight".to_string(),
                PyTensor {
                    tensor: weight.clone(),
                },
            );
        }
        if let Some(ref bias) = self.bias {
            params.insert(
                "bias".to_string(),
                PyTensor {
                    tensor: bias.clone(),
                },
            );
        }
        Ok(params)
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
        format!(
            "BatchNorm1d({}, eps={}, momentum={}, affine={}, track_running_stats={})",
            self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats
        )
    }
}

/// Layer Normalization layer
#[pyclass(name = "LayerNorm", extends = PyModule)]
pub struct PyLayerNorm {
    weight: Option<Tensor<f32>>,
    bias: Option<Tensor<f32>>,
    normalized_shape: Vec<usize>,
    eps: f32,
    elementwise_affine: bool,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    fn new(
        normalized_shape: Vec<usize>,
        eps: Option<f32>,
        elementwise_affine: Option<bool>,
    ) -> PyResult<(Self, PyModule)> {
        let eps = eps.unwrap_or(1e-5);
        let elementwise_affine = elementwise_affine.unwrap_or(true);

        // Initialize weight and bias if elementwise_affine=true
        let (weight, bias) = if elementwise_affine {
            let weight =
                py_result!(torsh_tensor::creation::ones(&normalized_shape))?.requires_grad_(true);
            let bias =
                py_result!(torsh_tensor::creation::zeros(&normalized_shape))?.requires_grad_(true);
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        Ok((
            Self {
                weight,
                bias,
                normalized_shape,
                eps,
                elementwise_affine,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through layer normalization
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // Simplified layer normalization implementation - return input as-is for now
        // TODO: Implement proper layer norm with mean/variance computation when available
        Ok(PyTensor {
            tensor: input.tensor.clone(),
        })
    }

    /// Get layer parameters
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(PyTensor {
                tensor: weight.clone(),
            });
        }
        if let Some(ref bias) = self.bias {
            params.push(PyTensor {
                tensor: bias.clone(),
            });
        }
        Ok(params)
    }

    /// Get named parameters
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        let mut params = HashMap::new();
        if let Some(ref weight) = self.weight {
            params.insert(
                "weight".to_string(),
                PyTensor {
                    tensor: weight.clone(),
                },
            );
        }
        if let Some(ref bias) = self.bias {
            params.insert(
                "bias".to_string(),
                PyTensor {
                    tensor: bias.clone(),
                },
            );
        }
        Ok(params)
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "LayerNorm({:?}, eps={}, elementwise_affine={})",
            self.normalized_shape, self.eps, self.elementwise_affine
        )
    }
}
