//! Pooling layers

use super::module::PyModule;
use crate::{error::PyResult, py_result, tensor::PyTensor};
use pyo3::prelude::*;
use pyo3::types::PyAny;
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
        kernel_size: Py<PyAny>,
        stride: Option<Py<PyAny>>,
        padding: Option<Py<PyAny>>,
        dilation: Option<Py<PyAny>>,
        ceil_mode: Option<bool>,
        return_indices: Option<bool>,
    ) -> PyResult<(Self, PyModule)> {
        // Parse kernel size
        let kernel_size = Python::attach(|py| -> PyResult<(usize, usize)> {
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
            Some(Python::attach(|py| -> PyResult<(usize, usize)> {
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
            Python::attach(|py| -> PyResult<(usize, usize)> {
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
            Python::attach(|py| -> PyResult<(usize, usize)> {
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
        // ✅ Proper max pooling implementation
        let shape = input.tensor.shape().dims().to_vec();

        // Expect 4D input: (batch, channels, height, width)
        if shape.len() != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 4D input (NCHW), got {}D",
                shape.len()
            )));
        }

        let (batch_size, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.kernel_size;
        let (stride_h, stride_w) = self.stride.unwrap_or(self.kernel_size);
        let (pad_h, pad_w) = self.padding;

        // Calculate output dimensions
        let out_h = if self.ceil_mode {
            ((in_h + 2 * pad_h - kh) as f32 / stride_h as f32).ceil() as usize + 1
        } else {
            (in_h + 2 * pad_h - kh) / stride_h + 1
        };
        let out_w = if self.ceil_mode {
            ((in_w + 2 * pad_w - kw) as f32 / stride_w as f32).ceil() as usize + 1
        } else {
            (in_w + 2 * pad_w - kw) / stride_w + 1
        };

        let input_data = py_result!(input.tensor.data())?;
        let mut output_data = vec![f32::NEG_INFINITY; batch_size * channels * out_h * out_w];

        // Perform max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let ih = (oh * stride_h + kh_idx) as i32 - pad_h as i32;
                                let iw = (ow * stride_w + kw_idx) as i32 - pad_w as i32;

                                if ih >= 0 && ih < in_h as i32 && iw >= 0 && iw < in_w as i32 {
                                    let input_idx = b * channels * in_h * in_w
                                        + c * in_h * in_w
                                        + ih as usize * in_w
                                        + iw as usize;
                                    max_val = max_val.max(input_data[input_idx]);
                                }
                            }
                        }

                        let output_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output_data[output_idx] = max_val;
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            output_data,
            vec![batch_size, channels, out_h, out_w],
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
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
        kernel_size: Py<PyAny>,
        stride: Option<Py<PyAny>>,
        padding: Option<Py<PyAny>>,
        ceil_mode: Option<bool>,
        count_include_pad: Option<bool>,
        divisor_override: Option<usize>,
    ) -> PyResult<(Self, PyModule)> {
        // Parse kernel size
        let kernel_size = Python::attach(|py| -> PyResult<(usize, usize)> {
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
            Some(Python::attach(|py| -> PyResult<(usize, usize)> {
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
            Python::attach(|py| -> PyResult<(usize, usize)> {
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
        // ✅ Proper average pooling implementation
        let shape = input.tensor.shape().dims().to_vec();

        // Expect 4D input: (batch, channels, height, width)
        if shape.len() != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 4D input (NCHW), got {}D",
                shape.len()
            )));
        }

        let (batch_size, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = self.kernel_size;
        let (stride_h, stride_w) = self.stride.unwrap_or(self.kernel_size);
        let (pad_h, pad_w) = self.padding;

        // Calculate output dimensions
        let out_h = if self.ceil_mode {
            ((in_h + 2 * pad_h - kh) as f32 / stride_h as f32).ceil() as usize + 1
        } else {
            (in_h + 2 * pad_h - kh) / stride_h + 1
        };
        let out_w = if self.ceil_mode {
            ((in_w + 2 * pad_w - kw) as f32 / stride_w as f32).ceil() as usize + 1
        } else {
            (in_w + 2 * pad_w - kw) / stride_w + 1
        };

        let input_data = py_result!(input.tensor.data())?;
        let mut output_data = vec![0.0; batch_size * channels * out_h * out_w];

        // Perform average pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;
                        let mut count = 0;

                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let ih = (oh * stride_h + kh_idx) as i32 - pad_h as i32;
                                let iw = (ow * stride_w + kw_idx) as i32 - pad_w as i32;

                                if ih >= 0 && ih < in_h as i32 && iw >= 0 && iw < in_w as i32 {
                                    let input_idx = b * channels * in_h * in_w
                                        + c * in_h * in_w
                                        + ih as usize * in_w
                                        + iw as usize;
                                    sum += input_data[input_idx];
                                    count += 1;
                                } else if self.count_include_pad {
                                    count += 1;
                                }
                            }
                        }

                        let divisor = if let Some(div) = self.divisor_override {
                            div as f32
                        } else {
                            count as f32
                        };

                        let output_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output_data[output_idx] = if divisor > 0.0 { sum / divisor } else { 0.0 };
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            output_data,
            vec![batch_size, channels, out_h, out_w],
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
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
    fn new(output_size: Py<PyAny>) -> PyResult<(Self, PyModule)> {
        // Parse output size
        let output_size = Python::attach(|py| -> PyResult<(usize, usize)> {
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
        // ✅ Proper adaptive average pooling implementation
        let shape = input.tensor.shape().dims().to_vec();

        // Expect 4D input: (batch, channels, height, width)
        if shape.len() != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 4D input (NCHW), got {}D",
                shape.len()
            )));
        }

        let (batch_size, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = self.output_size;

        let input_data = py_result!(input.tensor.data())?;
        let mut output_data = vec![0.0; batch_size * channels * out_h * out_w];

        // Perform adaptive average pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Calculate adaptive pooling window
                        let start_h = (oh * in_h) / out_h;
                        let end_h = ((oh + 1) * in_h) / out_h;
                        let start_w = (ow * in_w) / out_w;
                        let end_w = ((ow + 1) * in_w) / out_w;

                        let mut sum = 0.0;
                        let mut count = 0;

                        for ih in start_h..end_h {
                            for iw in start_w..end_w {
                                let input_idx =
                                    b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                                sum += input_data[input_idx];
                                count += 1;
                            }
                        }

                        let output_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output_data[output_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            output_data,
            vec![batch_size, channels, out_h, out_w],
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
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
    fn new(output_size: Py<PyAny>, return_indices: Option<bool>) -> PyResult<(Self, PyModule)> {
        // Parse output size
        let output_size = Python::attach(|py| -> PyResult<(usize, usize)> {
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
        // ✅ Proper adaptive max pooling implementation
        let shape = input.tensor.shape().dims().to_vec();

        // Expect 4D input: (batch, channels, height, width)
        if shape.len() != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 4D input (NCHW), got {}D",
                shape.len()
            )));
        }

        let (batch_size, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = self.output_size;

        let input_data = py_result!(input.tensor.data())?;
        let mut output_data = vec![f32::NEG_INFINITY; batch_size * channels * out_h * out_w];

        // Perform adaptive max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Calculate adaptive pooling window
                        let start_h = (oh * in_h) / out_h;
                        let end_h = ((oh + 1) * in_h) / out_h;
                        let start_w = (ow * in_w) / out_w;
                        let end_w = ((ow + 1) * in_w) / out_w;

                        let mut max_val = f32::NEG_INFINITY;

                        for ih in start_h..end_h {
                            for iw in start_w..end_w {
                                let input_idx =
                                    b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                                max_val = max_val.max(input_data[input_idx]);
                            }
                        }

                        let output_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output_data[output_idx] = max_val;
                    }
                }
            }
        }

        let result = py_result!(torsh_tensor::Tensor::from_data(
            output_data,
            vec![batch_size, channels, out_h, out_w],
            input.tensor.device()
        ))?;

        Ok(PyTensor { tensor: result })
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
