//! Convolutional neural network layers

use super::module::PyModule;
use crate::{device::PyDevice, error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// 2D Convolutional layer
#[pyclass(name = "Conv2d", extends = PyModule)]
pub struct PyConv2d {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    has_bias: bool,
    training: bool,
}

#[pymethods]
impl PyConv2d {
    #[new]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: PyObject,
        stride: Option<PyObject>,
        padding: Option<PyObject>,
        dilation: Option<PyObject>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<(Self, PyModule)> {
        let has_bias = bias.unwrap_or(true);
        let groups = groups.unwrap_or(1);

        // Parse kernel size
        let kernel_size = Python::with_gil(|py| -> PyResult<(usize, usize)> {
            if let Ok(size) = kernel_size.extract::<usize>(py) {
                Ok((size, size))
            } else if let Ok(tuple) = kernel_size.extract::<(usize, usize)>(py) {
                Ok(tuple)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "kernel_size must be an integer or tuple of integers",
                ))
            }
        })?;

        // Parse stride (default to kernel_size)
        let stride = if let Some(stride_obj) = stride {
            Python::with_gil(|py| -> PyResult<(usize, usize)> {
                if let Ok(stride) = stride_obj.extract::<usize>(py) {
                    Ok((stride, stride))
                } else if let Ok(tuple) = stride_obj.extract::<(usize, usize)>(py) {
                    Ok(tuple)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "stride must be an integer or tuple of integers",
                    ))
                }
            })?
        } else {
            (1, 1)
        };

        // Parse padding (default to 0)
        let padding = if let Some(padding_obj) = padding {
            Python::with_gil(|py| -> PyResult<(usize, usize)> {
                if let Ok(padding) = padding_obj.extract::<usize>(py) {
                    Ok((padding, padding))
                } else if let Ok(tuple) = padding_obj.extract::<(usize, usize)>(py) {
                    Ok(tuple)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "padding must be an integer or tuple of integers",
                    ))
                }
            })?
        } else {
            (0, 0)
        };

        // Parse dilation (default to 1)
        let dilation = if let Some(dilation_obj) = dilation {
            Python::with_gil(|py| -> PyResult<(usize, usize)> {
                if let Ok(dilation) = dilation_obj.extract::<usize>(py) {
                    Ok((dilation, dilation))
                } else if let Ok(tuple) = dilation_obj.extract::<(usize, usize)>(py) {
                    Ok(tuple)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "dilation must be an integer or tuple of integers",
                    ))
                }
            })?
        } else {
            (1, 1)
        };

        // Initialize weight with Kaiming uniform initialization
        let weight_shape = vec![
            out_channels,
            in_channels / groups,
            kernel_size.0,
            kernel_size.1,
        ];
        let weight = py_result!(torsh_tensor::creation::randn(&weight_shape))?.requires_grad_(true);

        // Initialize bias if needed
        let bias = if has_bias {
            let bias_shape = vec![out_channels];
            Some(py_result!(torsh_tensor::creation::zeros(&bias_shape))?.requires_grad_(true))
        } else {
            None
        };

        Ok((
            Self {
                weight,
                bias,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                has_bias,
                training: true,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through the convolutional layer
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // Perform 2D convolution
        let result = py_result!(input.tensor.conv2d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        ))?;

        Ok(PyTensor { tensor: result })
    }

    /// Get layer parameters
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut params = vec![PyTensor {
            tensor: self.weight.clone(),
        }];
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
        params.insert(
            "weight".to_string(),
            PyTensor {
                tensor: self.weight.clone(),
            },
        );
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

    /// Get layer info string
    fn extra_repr(&self) -> String {
        let bias_str = if self.has_bias {
            "bias=True"
        } else {
            "bias=False"
        };
        format!(
            "{}, {}, kernel_size={:?}, stride={:?}, padding={:?}, dilation={:?}, groups={}, {}",
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            bias_str
        )
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("Conv2d({})", self.extra_repr())
    }
}

/// 1D Convolutional layer
#[pyclass(name = "Conv1d", extends = PyModule)]
pub struct PyConv1d {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    has_bias: bool,
    training: bool,
}

#[pymethods]
impl PyConv1d {
    #[new]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<(Self, PyModule)> {
        let has_bias = bias.unwrap_or(true);
        let stride = stride.unwrap_or(1);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);
        let groups = groups.unwrap_or(1);

        // Initialize weight with Kaiming uniform initialization
        let weight_shape = vec![out_channels, in_channels / groups, kernel_size];
        let weight = py_result!(torsh_tensor::creation::randn(&weight_shape))?.requires_grad_(true);

        // Initialize bias if needed
        let bias = if has_bias {
            let bias_shape = vec![out_channels];
            Some(py_result!(torsh_tensor::creation::zeros(&bias_shape))?.requires_grad_(true))
        } else {
            None
        };

        Ok((
            Self {
                weight,
                bias,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                has_bias,
                training: true,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through the 1D convolutional layer
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // Perform 1D convolution
        let result = py_result!(input.tensor.conv1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        ))?;

        Ok(PyTensor { tensor: result })
    }

    /// Get layer parameters
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let mut params = vec![PyTensor {
            tensor: self.weight.clone(),
        }];
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
        params.insert(
            "weight".to_string(),
            PyTensor {
                tensor: self.weight.clone(),
            },
        );
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
        let bias_str = if self.has_bias {
            "bias=True"
        } else {
            "bias=False"
        };
        format!(
            "Conv1d({}, {}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, {})",
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            bias_str
        )
    }
}
