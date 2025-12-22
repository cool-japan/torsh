//! Tensor creation functions - zeros, ones, randn, etc.

use super::core::PyTensor;
use crate::{device::PyDevice, dtype::PyDType, error::PyResult, py_result};
use pyo3::prelude::*;

/// Register simplified tensor creation functions
pub fn register_creation_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use pyo3::wrap_pyfunction;

    #[pyfunction]
    fn tensor(
        data: &Bound<'_, PyAny>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        PyTensor::new(data, dtype, device, requires_grad)
    }

    #[pyfunction]
    fn zeros(
        size: Vec<usize>,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let tensor_result = py_result!(torsh_tensor::creation::zeros(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn ones(
        size: Vec<usize>,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let tensor_result = py_result!(torsh_tensor::creation::ones(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn randn(
        size: Vec<usize>,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let tensor_result = py_result!(torsh_tensor::creation::randn(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn rand(
        size: Vec<usize>,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let tensor_result = py_result!(torsh_tensor::creation::rand(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn empty(
        size: Vec<usize>,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        // Use zeros as a fallback since empty is not available
        let tensor_result = py_result!(torsh_tensor::creation::zeros(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn full(
        size: Vec<usize>,
        fill_value: f32,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let tensor_result = py_result!(torsh_tensor::creation::full(&size, fill_value))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn eye(
        n: usize,
        _m: Option<usize>,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        // torsh_tensor::creation::eye only takes one parameter
        let tensor_result = py_result!(torsh_tensor::creation::eye(n))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn arange(
        start: f32,
        end: Option<f32>,
        step: Option<f32>,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let (start, end) = if let Some(end) = end {
            (start, end)
        } else {
            (0.0, start)
        };
        let step = step.unwrap_or(1.0);
        let tensor_result = py_result!(torsh_tensor::creation::arange(start, end, step))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn linspace(
        start: f32,
        end: f32,
        steps: usize,
        _dtype: Option<PyDType>,
        _device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let tensor_result = py_result!(torsh_tensor::creation::linspace(start, end, steps))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    // "_like" functions - create tensors with same shape as input
    #[pyfunction]
    fn zeros_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::zeros_like(&input.tensor))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn ones_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let tensor_result = py_result!(torsh_tensor::creation::ones_like(&input.tensor))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn full_like(
        input: &PyTensor,
        fill_value: f32,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let shape = input.tensor.shape().dims().to_vec();
        let tensor_result = py_result!(torsh_tensor::creation::full(&shape, fill_value))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn empty_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        // Use zeros_like as fallback since empty is not critical
        let tensor_result = py_result!(torsh_tensor::creation::zeros_like(&input.tensor))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn randn_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let shape = input.tensor.shape().dims().to_vec();
        let tensor_result = py_result!(torsh_tensor::creation::randn(&shape))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn rand_like(
        input: &PyTensor,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let _dtype = dtype;
        let _device = device;
        let shape = input.tensor.shape().dims().to_vec();
        let tensor_result = py_result!(torsh_tensor::creation::rand(&shape))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    // Register all functions
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(rand, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(eye, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(linspace, m)?)?;

    // Register "_like" functions
    m.add_function(wrap_pyfunction!(zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(full_like, m)?)?;
    m.add_function(wrap_pyfunction!(empty_like, m)?)?;
    m.add_function(wrap_pyfunction!(randn_like, m)?)?;
    m.add_function(wrap_pyfunction!(rand_like, m)?)?;

    Ok(())
}
