//! Core types and utilities for linear algebra operations
//!
//! This module provides fundamental types and conversion utilities used throughout
//! the linear algebra operations, including conversions between Tensor and ndarray
//! formats for integration with scirs2.

use torsh_core::device::DeviceType;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{creation::from_vec, Tensor};

// Import scirs2 for enhanced linear algebra operations
use scirs2_core::ndarray::{Array1, Array2};

/// Norm order specification for matrix norms
///
/// Mathematical reference: For an m×n matrix A:
/// - Frobenius norm: ||A||_F = √(∑ᵢⱼ |aᵢⱼ|²)
/// - Nuclear norm: ||A||_* = ∑ᵢ σᵢ (sum of singular values)
/// - Infinity norm: ||A||_∞ = max_i (∑ⱼ |aᵢⱼ|)
/// - p-norm: ||A||_p = (∑ᵢⱼ |aᵢⱼ|^p)^(1/p)
#[derive(Debug, Clone, Copy)]
pub enum NormOrd {
    /// Frobenius norm (matrix generalization of Euclidean norm)
    Fro,
    /// Nuclear norm (sum of singular values)
    Nuclear,
    /// Infinity norm (maximum row sum)
    Inf,
    /// Negative infinity norm (minimum row sum)
    NegInf,
    /// p-norm for arbitrary p > 0
    P(f32),
}

/// Convert a 2D Tensor to ndarray Array2 for scirs2 operations
///
/// This function enables integration with scirs2's linear algebra operations
/// by converting tensors to the ndarray format required by scirs2.
///
/// # Arguments
/// * `tensor` - A 2D tensor to convert
///
/// # Returns
/// * `Array2<f32>` - The tensor data as an ndarray Array2
///
/// # Errors
/// * Returns error if tensor is not 2D
/// * Returns error if tensor data cannot be reshaped
pub fn tensor_to_array2(tensor: &Tensor) -> TorshResult<Array2<f32>> {
    let shape = tensor.shape();
    if shape.ndim() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Expected 2D tensor",
            "tensor_to_array2",
        ));
    }

    let data = tensor.data()?;
    let dims = shape.dims();

    Array2::from_shape_vec((dims[0], dims[1]), data.clone()).map_err(|e| {
        TorshError::invalid_argument_with_context(
            &format!("Failed to create Array2: {}", e),
            "tensor_to_array2",
        )
    })
}

/// Convert ndarray Array2 back to Tensor
///
/// This function converts the result of scirs2 operations back to torsh tensors.
///
/// # Arguments
/// * `array` - An ndarray Array2 to convert
///
/// # Returns
/// * `Tensor` - The array data as a torsh tensor on CPU device
#[allow(dead_code)]
pub fn array2_to_tensor(array: Array2<f32>) -> TorshResult<Tensor> {
    let shape = array.shape().to_vec();
    let data = array.into_raw_vec();
    Ok(from_vec(data, &[shape[0], shape[1]], DeviceType::Cpu)?)
}

/// Convert 1D Tensor to ndarray Array1
///
/// This function enables integration with scirs2's operations that work on vectors.
///
/// # Arguments
/// * `tensor` - A 1D tensor to convert
///
/// # Returns
/// * `Array1<f32>` - The tensor data as an ndarray Array1
///
/// # Errors
/// * Returns error if tensor is not 1D
#[allow(dead_code)]
pub fn tensor_to_array1(tensor: &Tensor) -> TorshResult<Array1<f32>> {
    let shape = tensor.shape();
    if shape.ndim() != 1 {
        return Err(TorshError::invalid_argument_with_context(
            "Expected 1D tensor",
            "tensor_to_array1",
        ));
    }

    let data = tensor.data()?;
    Ok(Array1::from_vec(data.clone()))
}

/// Convert Array1 back to Tensor
///
/// This function converts vector results from scirs2 operations back to torsh tensors.
///
/// # Arguments
/// * `array` - An ndarray Array1 to convert
///
/// # Returns
/// * `Tensor` - The array data as a torsh tensor on CPU device
#[allow(dead_code)]
pub fn array1_to_tensor(array: Array1<f32>) -> TorshResult<Tensor> {
    let length = array.len();
    let data = array.into_raw_vec();
    Ok(from_vec(data, &[length], DeviceType::Cpu)?)
}
