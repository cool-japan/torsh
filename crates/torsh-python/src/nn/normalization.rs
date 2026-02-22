//! Normalization layers

use super::module::PyModule;
use crate::{error::PyResult, py_result, tensor::PyTensor};
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
        // ✅ Proper 2D batch normalization implementation for 4D tensors (NCHW)
        let shape = input.tensor.shape().dims().to_vec();

        // Expect 4D input: (batch, channels, height, width)
        if shape.len() != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 4D input (NCHW), got {}D",
                shape.len()
            )));
        }

        let batch_size = shape[0];
        let num_channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial_size = height * width;

        if num_channels != self.num_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} channels, got {}",
                self.num_features, num_channels
            )));
        }

        let input_data = py_result!(input.tensor.data())?;
        let mut output_data = input_data.clone();

        if self.training {
            // Training mode: compute batch statistics across spatial dimensions
            if self.track_running_stats {
                self.num_batches_tracked += 1;
            }

            // Compute mean and variance for each channel across batch and spatial dims
            for c in 0..num_channels {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let mut count = 0;

                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx =
                                b * num_channels * spatial_size + c * spatial_size + h * width + w;
                            let val = input_data[idx];
                            sum += val;
                            sum_sq += val * val;
                            count += 1;
                        }
                    }
                }

                let mean = sum / count as f32;
                let var = (sum_sq / count as f32) - (mean * mean);

                // Update running statistics
                if self.track_running_stats {
                    let mut running_mean_data = py_result!(self.running_mean.data())?;
                    let mut running_var_data = py_result!(self.running_var.data())?;

                    running_mean_data[c] =
                        (1.0 - self.momentum) * running_mean_data[c] + self.momentum * mean;
                    running_var_data[c] =
                        (1.0 - self.momentum) * running_var_data[c] + self.momentum * var;

                    self.running_mean = py_result!(torsh_tensor::Tensor::from_data(
                        running_mean_data,
                        vec![num_channels],
                        self.running_mean.device()
                    ))?;
                    self.running_var = py_result!(torsh_tensor::Tensor::from_data(
                        running_var_data,
                        vec![num_channels],
                        self.running_var.device()
                    ))?;
                }

                // Normalize
                let std = (var + self.eps).sqrt();
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx =
                                b * num_channels * spatial_size + c * spatial_size + h * width + w;
                            output_data[idx] = (output_data[idx] - mean) / std;
                        }
                    }
                }

                // Apply affine transformation
                if self.affine {
                    if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                        let weight_data = py_result!(weight.data())?;
                        let bias_data = py_result!(bias.data())?;

                        for b in 0..batch_size {
                            for h in 0..height {
                                for w in 0..width {
                                    let idx = b * num_channels * spatial_size
                                        + c * spatial_size
                                        + h * width
                                        + w;
                                    output_data[idx] =
                                        output_data[idx] * weight_data[c] + bias_data[c];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Evaluation mode: use running statistics
            let running_mean_data = py_result!(self.running_mean.data())?;
            let running_var_data = py_result!(self.running_var.data())?;

            for c in 0..num_channels {
                let mean = running_mean_data[c];
                let var = running_var_data[c];
                let std = (var + self.eps).sqrt();

                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx =
                                b * num_channels * spatial_size + c * spatial_size + h * width + w;
                            output_data[idx] = (output_data[idx] - mean) / std;
                        }
                    }
                }

                // Apply affine transformation
                if self.affine {
                    if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                        let weight_data = py_result!(weight.data())?;
                        let bias_data = py_result!(bias.data())?;

                        for b in 0..batch_size {
                            for h in 0..height {
                                for w in 0..width {
                                    let idx = b * num_channels * spatial_size
                                        + c * spatial_size
                                        + h * width
                                        + w;
                                    output_data[idx] =
                                        output_data[idx] * weight_data[c] + bias_data[c];
                                }
                            }
                        }
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            output_data,
            shape.to_vec(),
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
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
        // ✅ Proper batch normalization implementation with statistics
        let shape = input.tensor.shape().dims().to_vec();

        // Expect input: (batch, channels) for 1D
        if shape.len() < 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected at least 2D input, got {}D",
                shape.len()
            )));
        }

        let batch_size = shape[0];
        let num_features = shape[1];

        if num_features != self.num_features {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected {} features, got {}",
                self.num_features, num_features
            )));
        }

        let input_data = py_result!(input.tensor.data())?;
        let mut output_data = input_data.clone();

        if self.training {
            // Training mode: compute batch statistics
            if self.track_running_stats {
                self.num_batches_tracked += 1;
            }

            // Compute mean and variance for each feature
            for c in 0..num_features {
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let mut count = 0;

                for b in 0..batch_size {
                    let idx = b * num_features + c;
                    let val = input_data[idx];
                    sum += val;
                    sum_sq += val * val;
                    count += 1;
                }

                let mean = sum / count as f32;
                let var = (sum_sq / count as f32) - (mean * mean);

                // Update running statistics
                if self.track_running_stats {
                    let mut running_mean_data = py_result!(self.running_mean.data())?;
                    let mut running_var_data = py_result!(self.running_var.data())?;

                    running_mean_data[c] =
                        (1.0 - self.momentum) * running_mean_data[c] + self.momentum * mean;
                    running_var_data[c] =
                        (1.0 - self.momentum) * running_var_data[c] + self.momentum * var;

                    self.running_mean = py_result!(torsh_tensor::Tensor::from_data(
                        running_mean_data,
                        vec![num_features],
                        self.running_mean.device()
                    ))?;
                    self.running_var = py_result!(torsh_tensor::Tensor::from_data(
                        running_var_data,
                        vec![num_features],
                        self.running_var.device()
                    ))?;
                }

                // Normalize
                let std = (var + self.eps).sqrt();
                for b in 0..batch_size {
                    let idx = b * num_features + c;
                    output_data[idx] = (output_data[idx] - mean) / std;
                }

                // Apply affine transformation
                if self.affine {
                    if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                        let weight_data = py_result!(weight.data())?;
                        let bias_data = py_result!(bias.data())?;

                        for b in 0..batch_size {
                            let idx = b * num_features + c;
                            output_data[idx] = output_data[idx] * weight_data[c] + bias_data[c];
                        }
                    }
                }
            }
        } else {
            // Evaluation mode: use running statistics
            let running_mean_data = py_result!(self.running_mean.data())?;
            let running_var_data = py_result!(self.running_var.data())?;

            for c in 0..num_features {
                let mean = running_mean_data[c];
                let var = running_var_data[c];
                let std = (var + self.eps).sqrt();

                for b in 0..batch_size {
                    let idx = b * num_features + c;
                    output_data[idx] = (output_data[idx] - mean) / std;
                }

                // Apply affine transformation
                if self.affine {
                    if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                        let weight_data = py_result!(weight.data())?;
                        let bias_data = py_result!(bias.data())?;

                        for b in 0..batch_size {
                            let idx = b * num_features + c;
                            output_data[idx] = output_data[idx] * weight_data[c] + bias_data[c];
                        }
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            output_data,
            shape.to_vec(),
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
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
        // ✅ Proper layer normalization implementation
        let shape = input.tensor.shape().dims().to_vec();
        let ndim = shape.len();
        let norm_ndim = self.normalized_shape.len();

        // Verify that normalized_shape matches the last dimensions of input
        if norm_ndim > ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "normalized_shape has {} dimensions but input has only {}",
                norm_ndim, ndim
            )));
        }

        // Check that the normalized dimensions match
        for i in 0..norm_ndim {
            if shape[ndim - norm_ndim + i] != self.normalized_shape[i] {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Input shape {:?} doesn't match normalized_shape {:?}",
                    shape, self.normalized_shape
                )));
            }
        }

        // Calculate the number of elements to normalize over
        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size: usize = shape[..ndim - norm_ndim].iter().product();

        let input_data = py_result!(input.tensor.data())?;
        let mut output_data = input_data.clone();

        // Normalize each batch independently
        for batch_idx in 0..batch_size {
            let start = batch_idx * norm_size;
            let end = start + norm_size;

            // Compute mean
            let mut sum = 0.0;
            for i in start..end {
                sum += input_data[i];
            }
            let mean = sum / norm_size as f32;

            // Compute variance
            let mut sum_sq_diff = 0.0;
            for i in start..end {
                let diff = input_data[i] - mean;
                sum_sq_diff += diff * diff;
            }
            let variance = sum_sq_diff / norm_size as f32;

            // Normalize
            let std = (variance + self.eps).sqrt();
            for i in start..end {
                output_data[i] = (output_data[i] - mean) / std;
            }

            // Apply affine transformation if enabled
            if self.elementwise_affine {
                if let (Some(ref weight), Some(ref bias)) = (&self.weight, &self.bias) {
                    let weight_data = py_result!(weight.data())?;
                    let bias_data = py_result!(bias.data())?;

                    for i in 0..norm_size {
                        let idx = start + i;
                        output_data[idx] = output_data[idx] * weight_data[i] + bias_data[i];
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            output_data,
            shape.to_vec(),
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
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
