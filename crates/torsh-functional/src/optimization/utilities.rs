//! Utility functions for optimization algorithms
//!
//! This module provides basic tensor operations used by optimization algorithms.

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Compute dot product of two tensors (flattened)
pub fn dot_product(a: &Tensor, b: &Tensor) -> TorshResult<f32> {
    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;

    if a_data.len() != b_data.len() {
        return Err(TorshError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }

    let mut sum = 0.0;
    for (ai, bi) in a_data.iter().zip(b_data.iter()) {
        sum += ai * bi;
    }

    Ok(sum)
}

/// Tensor addition
pub fn tensor_add(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    // Simple implementation - in practice would use proper tensor addition
    let a_data = a.data()?;
    let b_data = b.data()?;

    if a_data.len() != b_data.len() {
        return Err(TorshError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }

    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| ai + bi)
        .collect();
    Tensor::from_data(result, a.shape().dims().to_vec(), a.device())
}

/// Tensor subtraction
pub fn tensor_sub(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    let a_data = a.data()?;
    let b_data = b.data()?;

    if a_data.len() != b_data.len() {
        return Err(TorshError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }

    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| ai - bi)
        .collect();
    Tensor::from_data(result, a.shape().dims().to_vec(), a.device())
}

/// Scalar multiplication
pub fn tensor_scalar_mul(tensor: &Tensor, scalar: f32) -> TorshResult<Tensor> {
    let data = tensor.data()?;
    let result: Vec<f32> = data.iter().map(|&x| x * scalar).collect();
    Tensor::from_data(result, tensor.shape().dims().to_vec(), tensor.device())
}

/// Element-wise multiplication
pub fn tensor_elementwise_mul(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    let a_data = a.data()?;
    let b_data = b.data()?;

    if a_data.len() != b_data.len() {
        return Err(TorshError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }

    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| ai * bi)
        .collect();
    Tensor::from_data(result, a.shape().dims().to_vec(), a.device())
}

/// Element-wise division
pub fn tensor_elementwise_div(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    let a_data = a.data()?;
    let b_data = b.data()?;

    if a_data.len() != b_data.len() {
        return Err(TorshError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }

    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| ai / bi)
        .collect();
    Tensor::from_data(result, a.shape().dims().to_vec(), a.device())
}

/// Tensor norm (Euclidean)
pub fn tensor_norm(tensor: &Tensor) -> TorshResult<f32> {
    let data = tensor.data()?;
    let sum_squares: f32 = data.iter().map(|&x| x * x).sum();
    Ok(sum_squares.sqrt())
}

/// Create zeros tensor with same shape
pub fn tensor_zeros_like(tensor: &Tensor) -> TorshResult<Tensor> {
    let shape = tensor.shape().dims().to_vec();
    let zeros = vec![0.0; tensor.numel()];
    Tensor::from_data(zeros, shape, tensor.device())
}

/// Create tensor filled with constant value
pub fn tensor_full_like(tensor: &Tensor, value: f32) -> TorshResult<Tensor> {
    let shape = tensor.shape().dims().to_vec();
    let filled = vec![value; tensor.numel()];
    Tensor::from_data(filled, shape, tensor.device())
}

/// Element-wise square root
pub fn tensor_sqrt(tensor: &Tensor) -> TorshResult<Tensor> {
    let data = tensor.data()?;
    let result: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();
    Tensor::from_data(result, tensor.shape().dims().to_vec(), tensor.device())
}
