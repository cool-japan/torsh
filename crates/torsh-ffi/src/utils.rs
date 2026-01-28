//! Utility functions for Python bindings

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error::FfiError;
use crate::tensor::PyTensor;
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use pyo3::Py;

/// Create tensor from data
#[pyfunction]
#[pyo3(signature = (data, dtype=None, requires_grad=false))]
pub fn tensor(
    data: &Bound<'_, PyAny>,
    dtype: Option<&str>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    PyTensor::new(data, None, dtype, requires_grad)
}

/// Create tensor of zeros
#[pyfunction]
#[pyo3(signature = (shape, dtype=None, requires_grad=false))]
pub fn zeros(shape: Vec<usize>, dtype: Option<&str>, requires_grad: bool) -> PyResult<PyTensor> {
    let total_elements: usize = shape.iter().product();
    let data = vec![0.0; total_elements];

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(py_data.as_ref(), Some(shape), dtype, requires_grad)
    })
}

/// Create tensor of ones
#[pyfunction]
#[pyo3(signature = (shape, dtype=None, requires_grad=false))]
pub fn ones(shape: Vec<usize>, dtype: Option<&str>, requires_grad: bool) -> PyResult<PyTensor> {
    let total_elements: usize = shape.iter().product();
    let data = vec![1.0; total_elements];

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(py_data.as_ref(), Some(shape), dtype, requires_grad)
    })
}

/// Create tensor with random normal distribution
#[pyfunction]
#[pyo3(signature = (shape, mean=0.0, std=1.0, dtype=None, requires_grad=false))]
pub fn randn(
    shape: Vec<usize>,
    mean: f32,
    std: f32,
    dtype: Option<&str>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let total_elements: usize = shape.iter().product();

    // Simple pseudo-random normal distribution using Box-Muller transform
    let mut data = Vec::with_capacity(total_elements);
    let mut rng_state = 12345u64; // Simple LCG seed

    for i in 0..total_elements {
        if i % 2 == 0 {
            // Generate two random numbers using Box-Muller
            let u1 = lcg_random(&mut rng_state);
            let u2 = lcg_random(&mut rng_state);

            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).sin();

            data.push(mean + std * z0);
            if i + 1 < total_elements {
                data.push(mean + std * z1);
            }
        }
    }

    // Ensure we have the exact number of elements
    data.truncate(total_elements);

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(py_data.as_ref(), Some(shape), dtype, requires_grad)
    })
}

/// Create tensor with random uniform distribution
#[pyfunction]
#[pyo3(signature = (shape, low=0.0, high=1.0, dtype=None, requires_grad=false))]
pub fn rand(
    shape: Vec<usize>,
    low: f32,
    high: f32,
    dtype: Option<&str>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let total_elements: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total_elements);
    let mut rng_state = 54321u64; // Different seed for uniform distribution

    for _ in 0..total_elements {
        let random_val = lcg_random(&mut rng_state);
        data.push(low + (high - low) * random_val);
    }

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(py_data.as_ref(), Some(shape), dtype, requires_grad)
    })
}

/// Create identity matrix
#[pyfunction]
#[pyo3(signature = (n, dtype=None, requires_grad=false))]
pub fn eye(n: usize, dtype: Option<&str>, requires_grad: bool) -> PyResult<PyTensor> {
    let mut data = vec![0.0; n * n];

    for i in 0..n {
        data[i * n + i] = 1.0;
    }

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(py_data.as_ref(), Some(vec![n, n]), dtype, requires_grad)
    })
}

/// Create tensor filled with a scalar value
#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype=None, requires_grad=false))]
pub fn full(
    shape: Vec<usize>,
    fill_value: f32,
    dtype: Option<&str>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    let total_elements: usize = shape.iter().product();
    let data = vec![fill_value; total_elements];

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(py_data.as_ref(), Some(shape), dtype, requires_grad)
    })
}

/// Create tensor from NumPy array
#[pyfunction]
pub fn from_numpy(array: PyReadonlyArrayDyn<f32>) -> PyResult<PyTensor> {
    let data: Vec<f32> = array.as_array().iter().cloned().collect();
    let shape: Vec<usize> = array.shape().to_vec();

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(py_data.as_ref(), Some(shape), Some("f32"), false)
    })
}

/// Convert tensor to NumPy array
#[pyfunction]
pub fn to_numpy(tensor: &PyTensor, py: Python) -> PyResult<Py<PyAny>> {
    tensor.to_numpy_internal(py)
}

/// Create tensor with linearly spaced values
#[pyfunction]
#[pyo3(signature = (start, end, steps, dtype=None, requires_grad=false))]
pub fn linspace(
    start: f32,
    end: f32,
    steps: usize,
    dtype: Option<&str>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    if steps == 0 {
        return Err(FfiError::InvalidParameter {
            parameter: "steps".to_string(),
            value: "0".to_string(),
        }
        .into());
    }

    let mut data = Vec::with_capacity(steps);

    if steps == 1 {
        data.push(start);
    } else {
        let step_size = (end - start) / (steps - 1) as f32;
        for i in 0..steps {
            data.push(start + i as f32 * step_size);
        }
    }

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(py_data.as_ref(), Some(vec![steps]), dtype, requires_grad)
    })
}

/// Create tensor with values in a range
#[pyfunction]
#[pyo3(signature = (start, end, step=1.0, dtype=None, requires_grad=false))]
pub fn arange(
    start: f32,
    end: f32,
    step: f32,
    dtype: Option<&str>,
    requires_grad: bool,
) -> PyResult<PyTensor> {
    if step == 0.0 {
        return Err(FfiError::InvalidParameter {
            parameter: "step".to_string(),
            value: "0.0".to_string(),
        }
        .into());
    }

    let mut data = Vec::new();
    let mut current = start;

    if step > 0.0 {
        while current < end {
            data.push(current);
            current += step;
        }
    } else {
        while current > end {
            data.push(current);
            current += step;
        }
    }

    Python::attach(|py| {
        let py_data = PyList::new(py, &data)?;
        PyTensor::new(
            py_data.as_ref(),
            Some(vec![data.len()]),
            dtype,
            requires_grad,
        )
    })
}

/// Stack tensors along a new dimension
#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn stack(tensors: Vec<PyTensor>, dim: i32) -> PyResult<PyTensor> {
    if tensors.is_empty() {
        return Err(FfiError::InvalidParameter {
            parameter: "tensors".to_string(),
            value: "empty list".to_string(),
        }
        .into());
    }

    // Check all tensors have the same shape
    let first_shape = &tensors[0].shape();
    for tensor in &tensors[1..] {
        if tensor.shape() != *first_shape {
            return Err(FfiError::ShapeMismatch {
                expected: first_shape.clone(),
                actual: tensor.shape(),
            }
            .into());
        }
    }

    // For now, simple implementation for dim=0
    if dim != 0 {
        return Err(FfiError::UnsupportedOperation {
            operation: format!("stack with dim={} not yet implemented", dim),
        }
        .into());
    }

    let mut stacked_data = Vec::new();
    for tensor in &tensors {
        stacked_data.extend_from_slice(&tensor.data);
    }

    let mut new_shape = vec![tensors.len()];
    new_shape.extend_from_slice(first_shape);

    Python::attach(|py| {
        let py_data = PyList::new(py, &stacked_data)?;
        PyTensor::new(py_data.as_ref(), Some(new_shape), Some("f32"), false)
    })
}

/// Concatenate tensors along existing dimension
#[pyfunction]
#[pyo3(signature = (tensors, dim=0))]
pub fn cat(tensors: Vec<PyTensor>, dim: i32) -> PyResult<PyTensor> {
    if tensors.is_empty() {
        return Err(FfiError::InvalidParameter {
            parameter: "tensors".to_string(),
            value: "empty list".to_string(),
        }
        .into());
    }

    // For now, simple implementation for dim=0
    if dim != 0 {
        return Err(FfiError::UnsupportedOperation {
            operation: format!("cat with dim={} not yet implemented", dim),
        }
        .into());
    }

    let mut concatenated_data = Vec::new();
    let mut total_size = 0;

    for tensor in &tensors {
        concatenated_data.extend_from_slice(&tensor.data);
        total_size += tensor.data.len();
    }

    Python::attach(|py| {
        let py_data = PyList::new(py, &concatenated_data)?;
        PyTensor::new(py_data.as_ref(), Some(vec![total_size]), Some("f32"), false)
    })
}

/// Check if CUDA is available
#[pyfunction]
pub fn cuda_is_available() -> bool {
    // In a real implementation, this would check for CUDA availability
    cfg!(feature = "gpu")
}

/// Get number of CUDA devices
#[pyfunction]
pub fn cuda_device_count() -> usize {
    // In a real implementation, this would return actual CUDA device count
    if cuda_is_available() {
        1
    } else {
        0
    }
}

/// Set manual seed for reproducibility
#[pyfunction]
pub fn manual_seed(_seed: u64) {
    // In a real implementation, this would set the global random seed
    // For now, this is a placeholder
}

/// Simple Linear Congruential Generator for reproducible randomness
fn lcg_random(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    (*state as f32) / (u64::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_zeros() {
        Python::initialize();
        let result = zeros(vec![2, 3], None, false).unwrap();
        assert_eq!(result.shape(), vec![2, 3]);
        assert_eq!(result.data, vec![0.0; 6]);
    }

    #[test]
    fn test_ones() {
        Python::initialize();
        let result = ones(vec![2, 2], None, false).unwrap();
        assert_eq!(result.shape(), vec![2, 2]);
        assert_eq!(result.data, vec![1.0; 4]);
    }

    #[test]
    fn test_eye() {
        Python::initialize();
        let result = eye(3, None, false).unwrap();
        assert_eq!(result.shape(), vec![3, 3]);

        let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(result.data, expected);
    }

    #[test]
    fn test_linspace() {
        Python::initialize();
        let result = linspace(0.0, 1.0, 5, None, false).unwrap();
        assert_eq!(result.shape(), vec![5]);

        let expected = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        for (a, b) in result.data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_arange() {
        Python::initialize();
        let result = arange(0.0, 5.0, 1.0, None, false).unwrap();
        assert_eq!(result.shape(), vec![5]);
        assert_eq!(result.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_randn() {
        Python::initialize();
        let result = randn(vec![100], 0.0, 1.0, None, false).unwrap();
        assert_eq!(result.shape(), vec![100]);

        // Check that the mean is approximately 0 and std is approximately 1
        let mean = result.data.iter().sum::<f32>() / result.data.len() as f32;
        assert!(mean.abs() < 0.5); // Should be close to 0 for large sample

        let variance: f32 =
            result.data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / result.data.len() as f32;
        let std_dev = variance.sqrt();
        assert!((std_dev - 1.0).abs() < 0.5); // Should be close to 1
    }
}
