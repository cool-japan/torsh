//! Matrix comparison and similarity operations
//!
//! This module provides functions for comparing matrices and tensors,
//! including element-wise comparisons, equality checks, and distance metrics.

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

use crate::matrix_functions::matrix_norm;

/// Check if all elements of two matrices are close within tolerance
/// Similar to numpy.allclose
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `rtol` - Relative tolerance (default: 1e-5)
/// * `atol` - Absolute tolerance (default: 1e-8)
///
/// # Returns
///
/// True if all elements are within tolerance, false otherwise
///
/// The comparison uses: |a - b| <= atol + rtol * |b|
pub fn allclose(a: &Tensor, b: &Tensor, rtol: Option<f32>, atol: Option<f32>) -> TorshResult<bool> {
    if a.shape() != b.shape() {
        return Ok(false);
    }

    let rtol = rtol.unwrap_or(1e-5);
    let atol = atol.unwrap_or(1e-8);

    let shape = a.shape();
    let dims = shape.dims();

    // Handle different tensor dimensions
    match dims.len() {
        1 => {
            for i in 0..dims[0] {
                let a_val = a.get(&[i])?;
                let b_val = b.get(&[i])?;
                let tolerance = atol + rtol * b_val.abs();
                if (a_val - b_val).abs() > tolerance {
                    return Ok(false);
                }
            }
        }
        2 => {
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    let a_val = a.get(&[i, j])?;
                    let b_val = b.get(&[i, j])?;
                    let tolerance = atol + rtol * b_val.abs();
                    if (a_val - b_val).abs() > tolerance {
                        return Ok(false);
                    }
                }
            }
        }
        3 => {
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    for k in 0..dims[2] {
                        let a_val = a.get(&[i, j, k])?;
                        let b_val = b.get(&[i, j, k])?;
                        let tolerance = atol + rtol * b_val.abs();
                        if (a_val - b_val).abs() > tolerance {
                            return Ok(false);
                        }
                    }
                }
            }
        }
        _ => {
            return Err(torsh_core::TorshError::InvalidArgument(
                "allclose supports tensors up to 3 dimensions".to_string(),
            ));
        }
    }

    Ok(true)
}

/// Element-wise comparison returning a boolean tensor indicating which elements are close
/// Similar to numpy.isclose
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
/// * `rtol` - Relative tolerance (default: 1e-5)
/// * `atol` - Absolute tolerance (default: 1e-8)
///
/// # Returns
///
/// Boolean tensor (as f32: 1.0 for close, 0.0 for not close) with same shape as inputs
///
/// The comparison uses: |a - b| <= atol + rtol * |b|
pub fn isclose(
    a: &Tensor,
    b: &Tensor,
    rtol: Option<f32>,
    atol: Option<f32>,
) -> TorshResult<Tensor> {
    if a.shape() != b.shape() {
        return Err(torsh_core::TorshError::InvalidArgument(
            "Input tensors must have the same shape".to_string(),
        ));
    }

    let rtol = rtol.unwrap_or(1e-5);
    let atol = atol.unwrap_or(1e-8);

    let shape = a.shape();
    let dims = shape.dims();
    let total_elements = dims.iter().product::<usize>();
    let mut result_data = vec![0.0f32; total_elements];

    // Handle different tensor dimensions
    #[allow(clippy::needless_range_loop)]
    match dims.len() {
        1 => {
            for i in 0..dims[0] {
                let a_val = a.get(&[i])?;
                let b_val = b.get(&[i])?;
                let tolerance = atol + rtol * b_val.abs();
                result_data[i] = if (a_val - b_val).abs() <= tolerance {
                    1.0
                } else {
                    0.0
                };
            }
        }
        2 => {
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    let a_val = a.get(&[i, j])?;
                    let b_val = b.get(&[i, j])?;
                    let tolerance = atol + rtol * b_val.abs();
                    let flat_idx = i * dims[1] + j;
                    result_data[flat_idx] = if (a_val - b_val).abs() <= tolerance {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
        }
        3 => {
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    for k in 0..dims[2] {
                        let a_val = a.get(&[i, j, k])?;
                        let b_val = b.get(&[i, j, k])?;
                        let tolerance = atol + rtol * b_val.abs();
                        let flat_idx = i * dims[1] * dims[2] + j * dims[2] + k;
                        result_data[flat_idx] = if (a_val - b_val).abs() <= tolerance {
                            1.0
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        _ => {
            return Err(torsh_core::TorshError::InvalidArgument(
                "isclose supports tensors up to 3 dimensions".to_string(),
            ));
        }
    }

    Tensor::from_data(result_data, dims.to_vec(), a.device())
}

/// Check if two matrices are exactly equal
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// True if all elements are exactly equal, false otherwise
pub fn matrix_equals(a: &Tensor, b: &Tensor) -> TorshResult<bool> {
    if a.shape() != b.shape() {
        return Ok(false);
    }

    let shape = a.shape();
    let dims = shape.dims();

    // Handle different tensor dimensions
    match dims.len() {
        1 => {
            for i in 0..dims[0] {
                let a_val = a.get(&[i])?;
                let b_val = b.get(&[i])?;
                if a_val != b_val {
                    return Ok(false);
                }
            }
        }
        2 => {
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    let a_val = a.get(&[i, j])?;
                    let b_val = b.get(&[i, j])?;
                    if a_val != b_val {
                        return Ok(false);
                    }
                }
            }
        }
        3 => {
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    for k in 0..dims[2] {
                        let a_val = a.get(&[i, j, k])?;
                        let b_val = b.get(&[i, j, k])?;
                        if a_val != b_val {
                            return Ok(false);
                        }
                    }
                }
            }
        }
        _ => {
            return Err(torsh_core::TorshError::InvalidArgument(
                "matrix_equals supports tensors up to 3 dimensions".to_string(),
            ));
        }
    }

    Ok(true)
}

/// Compute the Frobenius distance between two matrices
/// ||A - B||_F where ||·||_F is the Frobenius norm
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// The Frobenius distance between the matrices
pub fn frobenius_distance(a: &Tensor, b: &Tensor) -> TorshResult<f32> {
    if a.shape() != b.shape() {
        return Err(torsh_core::TorshError::InvalidArgument(
            "Input tensors must have the same shape".to_string(),
        ));
    }

    let diff = a.sub(b)?;
    matrix_norm(&diff, Some("fro"))
}

/// Check if a matrix is approximately symmetric within tolerance
///
/// # Arguments
///
/// * `tensor` - Matrix to check
/// * `tol` - Tolerance for symmetry check (default: 1e-8)
///
/// # Returns
///
/// True if matrix is symmetric within tolerance, false otherwise
pub fn is_symmetric(tensor: &Tensor, tol: Option<f32>) -> TorshResult<bool> {
    if tensor.shape().ndim() != 2 {
        return Ok(false);
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);
    if m != n {
        return Ok(false);
    }

    let tol = tol.unwrap_or(1e-8);
    let transpose = tensor.transpose_view(0, 1)?;

    allclose(tensor, &transpose, Some(tol), Some(tol))
}
