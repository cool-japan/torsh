//! Functional operations for PyTorch-style API

use pyo3::prelude::*;
use crate::python::tensor::PyTensor;
use crate::error::{FfiError, FfiResult};

/// ReLU activation function
#[pyfunction]
#[pyo3(signature = (input, inplace=false))]
pub fn relu(input: &PyTensor, inplace: bool) -> PyResult<PyTensor> {
    let result_data: Vec<f32> = input.data.iter()
        .map(|&x| x.max(0.0))
        .collect();
    
    if inplace {
        // In a real implementation, this would modify the input tensor in-place
        // For now, we'll return a new tensor
    }
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result_data);
        PyTensor::new(
            data.as_ref(),
            Some(input.shape()),
            Some("f32"),
            input.requires_grad(),
        )
    })
}

/// Sigmoid activation function
#[pyfunction]
pub fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    let result_data: Vec<f32> = input.data.iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result_data);
        PyTensor::new(
            data.as_ref(),
            Some(input.shape()),
            Some("f32"),
            input.requires_grad(),
        )
    })
}

/// Tanh activation function
#[pyfunction]
pub fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    let result_data: Vec<f32> = input.data.iter()
        .map(|&x| x.tanh())
        .collect();
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result_data);
        PyTensor::new(
            data.as_ref(),
            Some(input.shape()),
            Some("f32"),
            input.requires_grad(),
        )
    })
}

/// GELU activation function (Gaussian Error Linear Unit)
#[pyfunction]
pub fn gelu(input: &PyTensor) -> PyResult<PyTensor> {
    let result_data: Vec<f32> = input.data.iter()
        .map(|&x| {
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect();
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result_data);
        PyTensor::new(
            data.as_ref(),
            Some(input.shape()),
            Some("f32"),
            input.requires_grad(),
        )
    })
}

/// Softmax function
#[pyfunction]
#[pyo3(signature = (input, dim=-1))]
pub fn softmax(input: &PyTensor, dim: i32) -> PyResult<PyTensor> {
    if input.shape().len() != 2 {
        return Err(FfiError::UnsupportedOperation {
            operation: "Softmax currently only supports 2D tensors".to_string(),
        }.into());
    }
    
    let batch_size = input.shape()[0];
    let features = input.shape()[1];
    let mut result_data = vec![0.0; input.data.len()];
    
    // Apply softmax along the last dimension (dim=-1)
    for batch_idx in 0..batch_size {
        let start_idx = batch_idx * features;
        let end_idx = start_idx + features;
        let batch_slice = &input.data[start_idx..end_idx];
        
        // Find max for numerical stability
        let max_val = batch_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials and sum
        let mut sum = 0.0;
        for i in 0..features {
            let exp_val = (batch_slice[i] - max_val).exp();
            result_data[start_idx + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for i in 0..features {
            result_data[start_idx + i] /= sum;
        }
    }
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result_data);
        PyTensor::new(
            data.as_ref(),
            Some(input.shape()),
            Some("f32"),
            input.requires_grad(),
        )
    })
}

/// Log softmax function
#[pyfunction]
#[pyo3(signature = (input, dim=-1))]
pub fn log_softmax(input: &PyTensor, dim: i32) -> PyResult<PyTensor> {
    let softmax_result = softmax(input, dim)?;
    
    let result_data: Vec<f32> = softmax_result.data.iter()
        .map(|&x| x.ln())
        .collect();
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result_data);
        PyTensor::new(
            data.as_ref(),
            Some(input.shape()),
            Some("f32"),
            input.requires_grad(),
        )
    })
}

/// Cross entropy loss
#[pyfunction]
#[pyo3(signature = (input, target, reduction="mean"))]
pub fn cross_entropy(input: &PyTensor, target: &PyTensor, reduction: &str) -> PyResult<PyTensor> {
    if input.shape().len() != 2 || target.shape().len() != 1 {
        return Err(FfiError::ShapeMismatch {
            expected: vec![0, 0], // Placeholder
            actual: vec![input.shape().len(), target.shape().len()],
        }.into());
    }
    
    let batch_size = input.shape()[0];
    let num_classes = input.shape()[1];
    
    if target.shape()[0] != batch_size {
        return Err(FfiError::ShapeMismatch {
            expected: vec![batch_size],
            actual: target.shape(),
        }.into());
    }
    
    // Apply log_softmax to input
    let log_probs = log_softmax(input, -1)?;
    
    let mut losses = Vec::new();
    
    // Compute negative log likelihood for each sample
    for batch_idx in 0..batch_size {
        let target_class = target.data[batch_idx] as usize;
        if target_class >= num_classes {
            return Err(FfiError::InvalidParameter {
                parameter: "target".to_string(),
                value: format!("class {} >= num_classes {}", target_class, num_classes),
            }.into());
        }
        
        let log_prob = log_probs.data[batch_idx * num_classes + target_class];
        losses.push(-log_prob);
    }
    
    let result = match reduction {
        "mean" => {
            let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;
            vec![mean_loss]
        },
        "sum" => {
            let sum_loss = losses.iter().sum::<f32>();
            vec![sum_loss]
        },
        "none" => losses,
        _ => return Err(FfiError::InvalidParameter {
            parameter: "reduction".to_string(),
            value: reduction.to_string(),
        }.into()),
    };
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result);
        let shape = if reduction == "none" { 
            vec![batch_size] 
        } else { 
            vec![] // Scalar
        };
        PyTensor::new(
            data.as_ref(),
            Some(shape),
            Some("f32"),
            input.requires_grad() || target.requires_grad(),
        )
    })
}

/// Mean squared error loss
#[pyfunction]
#[pyo3(signature = (input, target, reduction="mean"))]
pub fn mse_loss(input: &PyTensor, target: &PyTensor, reduction: &str) -> PyResult<PyTensor> {
    if input.shape() != target.shape() {
        return Err(FfiError::ShapeMismatch {
            expected: input.shape(),
            actual: target.shape(),
        }.into());
    }
    
    let squared_errors: Vec<f32> = input.data.iter()
        .zip(target.data.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .collect();
    
    let result = match reduction {
        "mean" => {
            let mean_loss = squared_errors.iter().sum::<f32>() / squared_errors.len() as f32;
            vec![mean_loss]
        },
        "sum" => {
            let sum_loss = squared_errors.iter().sum::<f32>();
            vec![sum_loss]
        },
        "none" => squared_errors,
        _ => return Err(FfiError::InvalidParameter {
            parameter: "reduction".to_string(),
            value: reduction.to_string(),
        }.into()),
    };
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result);
        let shape = if reduction == "none" { 
            input.shape() 
        } else { 
            vec![] // Scalar
        };
        PyTensor::new(
            data.as_ref(),
            Some(shape),
            Some("f32"),
            input.requires_grad() || target.requires_grad(),
        )
    })
}

/// Binary cross entropy loss
#[pyfunction]
#[pyo3(signature = (input, target, weight=None, reduction="mean"))]
pub fn binary_cross_entropy(
    input: &PyTensor, 
    target: &PyTensor, 
    weight: Option<&PyTensor>,
    reduction: &str
) -> PyResult<PyTensor> {
    if input.shape() != target.shape() {
        return Err(FfiError::ShapeMismatch {
            expected: input.shape(),
            actual: target.shape(),
        }.into());
    }
    
    let losses: Vec<f32> = input.data.iter()
        .zip(target.data.iter())
        .map(|(&pred, &target)| {
            // BCE loss: -[target * log(pred) + (1 - target) * log(1 - pred)]
            let pred_clamped = pred.clamp(1e-7, 1.0 - 1e-7); // Numerical stability
            -(target * pred_clamped.ln() + (1.0 - target) * (1.0 - pred_clamped).ln())
        })
        .collect();
    
    let result = match reduction {
        "mean" => {
            let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;
            vec![mean_loss]
        },
        "sum" => {
            let sum_loss = losses.iter().sum::<f32>();
            vec![sum_loss]
        },
        "none" => losses,
        _ => return Err(FfiError::InvalidParameter {
            parameter: "reduction".to_string(),
            value: reduction.to_string(),
        }.into()),
    };
    
    Python::with_gil(|py| {
        let data = pyo3::types::PyList::new_bound(py, &result);
        let shape = if reduction == "none" { 
            input.shape() 
        } else { 
            vec![] // Scalar
        };
        PyTensor::new(
            data.as_ref(),
            Some(shape),
            Some("f32"),
            input.requires_grad() || target.requires_grad(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyList;
    
    #[test]
    fn test_relu() {
        Python::with_gil(|py| {
            let data = PyList::new_bound(py, vec![-1.0, 0.0, 1.0, 2.0]);
            let input = PyTensor::new(data.as_ref(), None, None, false).unwrap();
            
            let output = relu(&input, false).unwrap();
            assert_eq!(output.data, vec![0.0, 0.0, 1.0, 2.0]);
        });
    }
    
    #[test]
    fn test_sigmoid() {
        Python::with_gil(|py| {
            let data = PyList::new_bound(py, vec![0.0]);
            let input = PyTensor::new(data.as_ref(), None, None, false).unwrap();
            
            let output = sigmoid(&input).unwrap();
            assert!((output.data[0] - 0.5).abs() < 1e-6);
        });
    }
    
    #[test]
    fn test_softmax() {
        Python::with_gil(|py| {
            let data = PyList::new_bound(py, vec![1.0, 2.0, 3.0]);
            let input = PyTensor::new(
                data.as_ref(), 
                Some(vec![1, 3]), 
                None, 
                false
            ).unwrap();
            
            let output = softmax(&input, -1).unwrap();
            let sum: f32 = output.data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        });
    }
    
    #[test]
    fn test_mse_loss() {
        Python::with_gil(|py| {
            let input_data = PyList::new_bound(py, vec![1.0, 2.0, 3.0]);
            let target_data = PyList::new_bound(py, vec![1.5, 2.5, 3.5]);
            
            let input = PyTensor::new(input_data.as_ref(), None, None, false).unwrap();
            let target = PyTensor::new(target_data.as_ref(), None, None, false).unwrap();
            
            let loss = mse_loss(&input, &target, "mean").unwrap();
            // Expected: mean of [(1-1.5)^2, (2-2.5)^2, (3-3.5)^2] = mean of [0.25, 0.25, 0.25] = 0.25
            assert!((loss.data[0] - 0.25).abs() < 1e-6);
        });
    }
}