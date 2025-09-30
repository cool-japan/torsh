//! Input validation utilities for Python bindings

use crate::error::PyResult;
use pyo3::prelude::*;

/// Validate that a shape is valid (all dimensions > 0)
pub fn validate_shape(shape: &[usize]) -> PyResult<()> {
    for (i, &dim) in shape.iter().enumerate() {
        if dim == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid shape: dimension {} cannot be zero",
                i
            )));
        }
    }
    Ok(())
}

/// Validate that an index is within bounds for a given dimension
pub fn validate_index(index: i64, dim_size: usize) -> PyResult<usize> {
    let positive_index = if index < 0 {
        let abs_index = (-index) as usize;
        if abs_index > dim_size {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} is out of bounds for dimension with size {}",
                index, dim_size
            )));
        }
        dim_size - abs_index
    } else {
        let pos_index = index as usize;
        if pos_index >= dim_size {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Index {} is out of bounds for dimension with size {}",
                index, dim_size
            )));
        }
        pos_index
    };
    Ok(positive_index)
}

/// Validate that dimensions are compatible for broadcasting
pub fn validate_broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> PyResult<Vec<usize>> {
    let mut result_shape = Vec::new();
    let max_dims = shape1.len().max(shape2.len());

    for i in 0..max_dims {
        let dim1 = if i < shape1.len() {
            shape1[shape1.len() - 1 - i]
        } else {
            1
        };
        let dim2 = if i < shape2.len() {
            shape2[shape2.len() - 1 - i]
        } else {
            1
        };

        if dim1 == dim2 || dim1 == 1 || dim2 == 1 {
            result_shape.push(dim1.max(dim2));
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                shape1, shape2
            )));
        }
    }

    result_shape.reverse();
    Ok(result_shape)
}

/// Validate that a learning rate is positive
pub fn validate_learning_rate(lr: f32) -> PyResult<()> {
    if lr <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Learning rate must be positive",
        ));
    }
    Ok(())
}

/// Validate that momentum is in valid range [0, 1]
pub fn validate_momentum(momentum: f32) -> PyResult<()> {
    if !(0.0..=1.0).contains(&momentum) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Momentum must be in range [0, 1]",
        ));
    }
    Ok(())
}

/// Validate that weight decay is non-negative
pub fn validate_weight_decay(weight_decay: f32) -> PyResult<()> {
    if weight_decay < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Weight decay must be non-negative",
        ));
    }
    Ok(())
}

/// Validate that epsilon is positive
pub fn validate_epsilon(eps: f32) -> PyResult<()> {
    if eps <= 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Epsilon must be positive",
        ));
    }
    Ok(())
}

/// Validate beta parameters for Adam-like optimizers
pub fn validate_betas(betas: (f32, f32)) -> PyResult<()> {
    let (beta1, beta2) = betas;
    if !(0.0..1.0).contains(&beta1) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Beta1 must be in range [0, 1)",
        ));
    }
    if !(0.0..1.0).contains(&beta2) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Beta2 must be in range [0, 1)",
        ));
    }
    Ok(())
}

/// Validate that tensor dimensions match for operations
pub fn validate_tensor_shapes_match(shape1: &[usize], shape2: &[usize]) -> PyResult<()> {
    if shape1 != shape2 {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Tensor shapes do not match: {:?} vs {:?}",
            shape1, shape2
        )));
    }
    Ok(())
}

/// Validate that a dimension index is valid for a tensor
pub fn validate_dimension(dim: i32, ndim: usize) -> PyResult<usize> {
    let positive_dim = if dim < 0 {
        let abs_dim = (-dim) as usize;
        if abs_dim > ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim, ndim
            )));
        }
        ndim - abs_dim
    } else {
        let pos_dim = dim as usize;
        if pos_dim >= ndim {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Dimension {} is out of bounds for tensor with {} dimensions",
                dim, ndim
            )));
        }
        pos_dim
    };
    Ok(positive_dim)
}

/// Validate that parameters list is not empty
pub fn validate_parameters_not_empty<T>(params: &[T]) -> PyResult<()> {
    if params.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Parameters list cannot be empty",
        ));
    }
    Ok(())
}
