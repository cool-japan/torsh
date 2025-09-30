//! Tensor creation functions - zeros, ones, randn, etc.

use super::core::PyTensor;
use crate::{device::PyDevice, dtype::PyDType, error::PyResult, py_result};
use pyo3::prelude::*;
use torsh_core::device::DeviceType;

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
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
        let tensor_result = py_result!(torsh_tensor::creation::zeros(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn ones(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
        let tensor_result = py_result!(torsh_tensor::creation::ones(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn randn(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
        let tensor_result = py_result!(torsh_tensor::creation::randn(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn rand(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
        let tensor_result = py_result!(torsh_tensor::creation::rand(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn empty(
        size: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
        // Use zeros as a fallback since empty is not available
        let tensor_result = py_result!(torsh_tensor::creation::zeros(&size))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn full(
        size: Vec<usize>,
        fill_value: f32,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
        let tensor_result = py_result!(torsh_tensor::creation::full(&size, fill_value))?;
        let tensor = tensor_result.requires_grad_(requires_grad.unwrap_or(false));
        Ok(PyTensor { tensor })
    }

    #[pyfunction]
    fn eye(
        n: usize,
        m: Option<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
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
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
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
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: Option<bool>,
    ) -> PyResult<PyTensor> {
        let device = device.map(|d| d.device).unwrap_or(DeviceType::Cpu);
        let tensor_result = py_result!(torsh_tensor::creation::linspace(start, end, steps))?;
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

    Ok(())
}
