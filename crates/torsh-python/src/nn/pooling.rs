//! Pooling layers

use super::module::PyModule;
use crate::{error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use std::collections::HashMap;

/// 2D Max Pooling layer
#[pyclass(name = "MaxPool2d", extends = PyModule)]
pub struct PyMaxPool2d {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
    return_indices: bool,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    fn new(
        kernel_size: PyObject,
        stride: Option<PyObject>,
        padding: Option<PyObject>,
        dilation: Option<PyObject>,
        ceil_mode: Option<bool>,
        return_indices: Option<bool>,
    ) -> PyResult<(Self, PyModule)> {
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

        // Parse stride (defaults to kernel_size if None)
        let stride = if let Some(stride_obj) = stride {
            Some(Python::with_gil(|py| -> PyResult<(usize, usize)> {
                if let Ok(stride) = stride_obj.extract::<usize>(py) {
                    Ok((stride, stride))
                } else if let Ok(tuple) = stride_obj.extract::<(usize, usize)>(py) {
                    Ok(tuple)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "stride must be an integer or tuple of integers",
                    ))
                }
            })?)
        } else {
            None
        };

        // Parse padding
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

        // Parse dilation
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

        Ok((
            Self {
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode: ceil_mode.unwrap_or(false),
                return_indices: return_indices.unwrap_or(false),
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through max pool 2d
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // Simplified max pooling implementation - return input as-is for now
        // TODO: Implement proper max pooling when pooling ops are available
        Ok(PyTensor {
            tensor: input.tensor.clone(),
        })
    }

    /// Get layer parameters (MaxPool2d has no parameters)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    /// Get named parameters (MaxPool2d has no parameters)
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        Ok(HashMap::new())
    }

    /// String representation
    fn __repr__(&self) -> String {
        let stride_str = if let Some(stride) = self.stride {
            format!("stride={:?}", stride)
        } else {
            "stride=None".to_string()
        };
        format!(
            "MaxPool2d(kernel_size={:?}, {}, padding={:?}, dilation={:?}, ceil_mode={}, return_indices={})",
            self.kernel_size, stride_str, self.padding, self.dilation, self.ceil_mode, self.return_indices
        )
    }
}

/// 2D Average Pooling layer
#[pyclass(name = "AvgPool2d", extends = PyModule)]
pub struct PyAvgPool2d {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Option<usize>,
}

#[pymethods]
impl PyAvgPool2d {
    #[new]
    fn new(
        kernel_size: PyObject,
        stride: Option<PyObject>,
        padding: Option<PyObject>,
        ceil_mode: Option<bool>,
        count_include_pad: Option<bool>,
        divisor_override: Option<usize>,
    ) -> PyResult<(Self, PyModule)> {
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

        // Parse stride
        let stride = if let Some(stride_obj) = stride {
            Some(Python::with_gil(|py| -> PyResult<(usize, usize)> {
                if let Ok(stride) = stride_obj.extract::<usize>(py) {
                    Ok((stride, stride))
                } else if let Ok(tuple) = stride_obj.extract::<(usize, usize)>(py) {
                    Ok(tuple)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "stride must be an integer or tuple of integers",
                    ))
                }
            })?)
        } else {
            None
        };

        // Parse padding
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

        Ok((
            Self {
                kernel_size,
                stride,
                padding,
                ceil_mode: ceil_mode.unwrap_or(false),
                count_include_pad: count_include_pad.unwrap_or(true),
                divisor_override,
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through average pool 2d
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // Simplified average pooling implementation - return input as-is for now
        // TODO: Implement proper average pooling when pooling ops are available
        Ok(PyTensor {
            tensor: input.tensor.clone(),
        })
    }

    /// Get layer parameters (AvgPool2d has no parameters)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    /// Get named parameters (AvgPool2d has no parameters)
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        Ok(HashMap::new())
    }

    /// String representation
    fn __repr__(&self) -> String {
        let stride_str = if let Some(stride) = self.stride {
            format!("stride={:?}", stride)
        } else {
            "stride=None".to_string()
        };
        let divisor_str = if let Some(divisor) = self.divisor_override {
            format!("divisor_override={}", divisor)
        } else {
            "divisor_override=None".to_string()
        };
        format!(
            "AvgPool2d(kernel_size={:?}, {}, padding={:?}, ceil_mode={}, count_include_pad={}, {})",
            self.kernel_size,
            stride_str,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            divisor_str
        )
    }
}

/// Adaptive Average Pooling 2D layer
#[pyclass(name = "AdaptiveAvgPool2d", extends = PyModule)]
pub struct PyAdaptiveAvgPool2d {
    output_size: (usize, usize),
}

#[pymethods]
impl PyAdaptiveAvgPool2d {
    #[new]
    fn new(output_size: PyObject) -> PyResult<(Self, PyModule)> {
        // Parse output size
        let output_size = Python::with_gil(|py| -> PyResult<(usize, usize)> {
            if let Ok(size) = output_size.extract::<usize>(py) {
                Ok((size, size))
            } else if let Ok(tuple) = output_size.extract::<(usize, usize)>(py) {
                Ok(tuple)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "output_size must be an integer or tuple of integers",
                ))
            }
        })?;

        Ok((Self { output_size }, PyModule::new()))
    }

    /// Forward pass through adaptive average pool 2d
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // Simplified adaptive average pooling implementation - return input as-is for now
        // TODO: Implement proper adaptive average pooling when pooling ops are available
        Ok(PyTensor {
            tensor: input.tensor.clone(),
        })
    }

    /// Get layer parameters (AdaptiveAvgPool2d has no parameters)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    /// Get named parameters (AdaptiveAvgPool2d has no parameters)
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        Ok(HashMap::new())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("AdaptiveAvgPool2d(output_size={:?})", self.output_size)
    }
}

/// Adaptive Max Pooling 2D layer
#[pyclass(name = "AdaptiveMaxPool2d", extends = PyModule)]
pub struct PyAdaptiveMaxPool2d {
    output_size: (usize, usize),
    return_indices: bool,
}

#[pymethods]
impl PyAdaptiveMaxPool2d {
    #[new]
    fn new(output_size: PyObject, return_indices: Option<bool>) -> PyResult<(Self, PyModule)> {
        // Parse output size
        let output_size = Python::with_gil(|py| -> PyResult<(usize, usize)> {
            if let Ok(size) = output_size.extract::<usize>(py) {
                Ok((size, size))
            } else if let Ok(tuple) = output_size.extract::<(usize, usize)>(py) {
                Ok(tuple)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "output_size must be an integer or tuple of integers",
                ))
            }
        })?;

        Ok((
            Self {
                output_size,
                return_indices: return_indices.unwrap_or(false),
            },
            PyModule::new(),
        ))
    }

    /// Forward pass through adaptive max pool 2d
    fn forward(&mut self, input: &PyTensor) -> PyResult<PyTensor> {
        // Simplified adaptive max pooling implementation - return input as-is for now
        // TODO: Implement proper adaptive max pooling when pooling ops are available
        Ok(PyTensor {
            tensor: input.tensor.clone(),
        })
    }

    /// Get layer parameters (AdaptiveMaxPool2d has no parameters)
    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    /// Get named parameters (AdaptiveMaxPool2d has no parameters)
    fn named_parameters(&self) -> PyResult<HashMap<String, PyTensor>> {
        Ok(HashMap::new())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "AdaptiveMaxPool2d(output_size={:?}, return_indices={})",
            self.output_size, self.return_indices
        )
    }
}
