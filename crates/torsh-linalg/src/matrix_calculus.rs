//! Matrix calculus operations for optimization and analysis
//!
//! This module provides PyTorch-compatible matrix calculus operations, powered by
//! scirs2-linalg's numerical differentiation implementations.
//!
//! ## Features
//!
//! - **Gradients**: Compute gradients of scalar-valued functions
//! - **Jacobians**: Compute Jacobians of vector-valued functions
//! - **Hessians**: Compute Hessians for second-order optimization
//! - **Directional Derivatives**: Compute derivatives along specified directions
//! - **Hessian-Vector Products**: Efficient computation for large-scale optimization
//!
//! ## Examples
//!
//! ```ignore
//! use torsh_linalg::matrix_calculus::gradient;
//! use torsh_tensor::Tensor;
//!
//! // Define a quadratic function f(x) = x^T x
//! let f = |x: &Tensor| -> Result<f32> {
//!     let x_sq = x.pow(2.0)?;
//!     x_sq.sum()
//! };
//!
//! // Compute gradient at x = [1.0, 2.0, 3.0]
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
//! let grad = gradient(f, &x, None)?;
//! // grad should be approximately [2.0, 4.0, 6.0]
//! ```

use torsh_core::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(feature = "scirs2-integration")]
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Compute the gradient of a scalar-valued function
///
/// For a function f: R^n -> R, the gradient is an n-dimensional vector where
/// grad\[i\] = df/dx_i.
///
/// # Arguments
///
/// * `f` - Function mapping a tensor to a scalar
/// * `x` - Point at which to evaluate the gradient
/// * `epsilon` - Step size for finite difference approximation (default: sqrt(machine_epsilon))
///
/// # Returns
///
/// Gradient tensor of the same shape as `x`
#[cfg(feature = "scirs2-integration")]
pub fn gradient<F>(f: F, x: &Tensor, epsilon: Option<f32>) -> Result<Tensor>
where
    F: Fn(&ArrayView1<f32>) -> scirs2_linalg::error::LinalgResult<f32>,
{
    // Validate input is 1D
    if x.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Gradient requires 1D tensor".to_string(),
        ));
    }

    // Convert tensor to ndarray
    let x_array = tensor_to_array1(x)?;

    // Compute gradient using scirs2-linalg
    let grad_array = scirs2_linalg::matrix_calculus::gradient(f, &x_array.view(), epsilon)
        .map_err(|e| TorshError::ComputeError(format!("Gradient computation failed: {e}")))?;

    // Convert back to tensor
    array1_to_tensor(&grad_array, x.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn gradient<F>(_f: F, _x: &Tensor, _epsilon: Option<f32>) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<f32>,
{
    Err(TorshError::NotImplemented(
        "Gradient requires scirs2-integration feature".to_string(),
    ))
}

/// Compute the Jacobian matrix of a vector-valued function
///
/// For a function f: R^n -> R^m, the Jacobian is an m×n matrix where
/// J\[i,j\] = df_i/dx_j.
///
/// # Arguments
///
/// * `f` - Function mapping an n-dimensional tensor to an m-dimensional tensor
/// * `x` - Point at which to evaluate the Jacobian
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// Jacobian matrix of size [m, n]
#[cfg(feature = "scirs2-integration")]
pub fn jacobian<F>(f: F, x: &Tensor, epsilon: Option<f32>) -> Result<Tensor>
where
    F: Fn(&ArrayView1<f32>) -> scirs2_linalg::error::LinalgResult<Array1<f32>>,
{
    // Validate input is 1D
    if x.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Jacobian requires 1D tensor".to_string(),
        ));
    }

    // Convert tensor to ndarray
    let x_array = tensor_to_array1(x)?;

    // Compute Jacobian using scirs2-linalg
    let jac_array = scirs2_linalg::matrix_calculus::jacobian(f, &x_array.view(), epsilon)
        .map_err(|e| TorshError::ComputeError(format!("Jacobian computation failed: {e}")))?;

    // Convert back to tensor
    array2_to_tensor(&jac_array, x.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn jacobian<F>(_f: F, _x: &Tensor, _epsilon: Option<f32>) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    Err(TorshError::NotImplemented(
        "Jacobian requires scirs2-integration feature".to_string(),
    ))
}

/// Compute the Hessian matrix of a scalar-valued function
///
/// For a function f: R^n -> R, the Hessian is an n×n symmetric matrix where
/// H\[i,j\] = d²f/(dx_i dx_j).
///
/// # Arguments
///
/// * `f` - Function mapping an n-dimensional tensor to a scalar
/// * `x` - Point at which to evaluate the Hessian
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// Hessian matrix of size [n, n]
#[cfg(feature = "scirs2-integration")]
pub fn hessian<F>(f: F, x: &Tensor, epsilon: Option<f32>) -> Result<Tensor>
where
    F: Fn(&ArrayView1<f32>) -> scirs2_linalg::error::LinalgResult<f32>,
{
    // Validate input is 1D
    if x.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Hessian requires 1D tensor".to_string(),
        ));
    }

    // Convert tensor to ndarray
    let x_array = tensor_to_array1(x)?;

    // Compute Hessian using scirs2-linalg
    let hess_array = scirs2_linalg::matrix_calculus::hessian(f, &x_array.view(), epsilon)
        .map_err(|e| TorshError::ComputeError(format!("Hessian computation failed: {e}")))?;

    // Convert back to tensor
    array2_to_tensor(&hess_array, x.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn hessian<F>(_f: F, _x: &Tensor, _epsilon: Option<f32>) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<f32>,
{
    Err(TorshError::NotImplemented(
        "Hessian requires scirs2-integration feature".to_string(),
    ))
}

/// Compute the directional derivative of a scalar-valued function
///
/// Computes the derivative of f along direction d at point x:
/// D_d f(x) = lim_{h->0} [f(x + h*d) - f(x)] / h
///
/// # Arguments
///
/// * `f` - Function mapping an n-dimensional tensor to a scalar
/// * `x` - Point at which to evaluate the directional derivative
/// * `direction` - Direction vector (will be normalized)
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// Directional derivative (scalar)
#[cfg(feature = "scirs2-integration")]
pub fn directional_derivative<F>(
    f: F,
    x: &Tensor,
    direction: &Tensor,
    epsilon: Option<f32>,
) -> Result<f32>
where
    F: Fn(&ArrayView1<f32>) -> scirs2_linalg::error::LinalgResult<f32>,
{
    // Validate inputs
    if x.shape().ndim() != 1 || direction.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Directional derivative requires 1D tensors".to_string(),
        ));
    }

    if x.shape().dims()[0] != direction.shape().dims()[0] {
        return Err(TorshError::InvalidArgument(
            "x and direction must have the same length".to_string(),
        ));
    }

    // Convert tensors to ndarray
    let x_array = tensor_to_array1(x)?;
    let direction_array = tensor_to_array1(direction)?;

    // Compute directional derivative using scirs2-linalg
    scirs2_linalg::matrix_calculus::directional_derivative(
        f,
        &x_array.view(),
        &direction_array.view(),
        epsilon,
    )
    .map_err(|e| {
        TorshError::ComputeError(format!("Directional derivative computation failed: {e}"))
    })
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn directional_derivative<F>(
    _f: F,
    _x: &Tensor,
    _direction: &Tensor,
    _epsilon: Option<f32>,
) -> Result<f32>
where
    F: Fn(&Tensor) -> Result<f32>,
{
    Err(TorshError::NotImplemented(
        "Directional derivative requires scirs2-integration feature".to_string(),
    ))
}

/// Compute Hessian-vector product efficiently
///
/// Computes H*v where H is the Hessian of f at x, without explicitly
/// forming the full Hessian matrix. This is more efficient for large-scale
/// optimization problems.
///
/// # Arguments
///
/// * `f` - Function mapping an n-dimensional tensor to a scalar
/// * `x` - Point at which to evaluate the Hessian
/// * `v` - Vector to multiply with the Hessian
/// * `epsilon` - Step size for finite difference approximation
///
/// # Returns
///
/// Hessian-vector product of shape \[n\]
#[cfg(feature = "scirs2-integration")]
pub fn hessian_vector_product<F>(
    f: F,
    x: &Tensor,
    v: &Tensor,
    epsilon: Option<f32>,
) -> Result<Tensor>
where
    F: Fn(&ArrayView1<f32>) -> scirs2_linalg::error::LinalgResult<f32> + Copy,
{
    // Validate inputs
    if x.shape().ndim() != 1 || v.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Hessian-vector product requires 1D tensors".to_string(),
        ));
    }

    if x.shape().dims()[0] != v.shape().dims()[0] {
        return Err(TorshError::InvalidArgument(
            "x and v must have the same length".to_string(),
        ));
    }

    // Convert tensors to ndarray
    let x_array = tensor_to_array1(x)?;
    let v_array = tensor_to_array1(v)?;

    // Compute Hessian-vector product using scirs2-linalg
    let hv_array = scirs2_linalg::matrix_calculus::enhanced::hessian_vector_product(
        f,
        &x_array.view(),
        &v_array.view(),
        epsilon,
    )
    .map_err(|e| {
        TorshError::ComputeError(format!("Hessian-vector product computation failed: {e}"))
    })?;

    // Convert back to tensor
    array1_to_tensor(&hv_array, x.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn hessian_vector_product<F>(
    _f: F,
    _x: &Tensor,
    _v: &Tensor,
    _epsilon: Option<f32>,
) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<f32>,
{
    Err(TorshError::NotImplemented(
        "Hessian-vector product requires scirs2-integration feature".to_string(),
    ))
}

// Helper functions for tensor <-> ndarray conversions

#[cfg(feature = "scirs2-integration")]
fn tensor_to_array1(tensor: &Tensor) -> Result<Array1<f32>> {
    if tensor.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Expected 1D tensor".to_string(),
        ));
    }

    let len = tensor.shape().dims()[0];
    let mut data = Vec::with_capacity(len);
    for i in 0..len {
        data.push(tensor.get(&[i])?);
    }

    Ok(Array1::from_vec(data))
}

#[cfg(feature = "scirs2-integration")]
fn array1_to_tensor(array: &Array1<f32>, device: torsh_core::DeviceType) -> Result<Tensor> {
    let len = array.len();
    let data = array.to_vec();
    Tensor::from_data(data, vec![len], device)
}

#[cfg(feature = "scirs2-integration")]
fn array2_to_tensor(array: &Array2<f32>, device: torsh_core::DeviceType) -> Result<Tensor> {
    let shape = array.shape();
    let (rows, cols) = (shape[0], shape[1]);

    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(array[[i, j]]);
        }
    }

    Tensor::from_data(data, vec![rows, cols], device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_gradient_quadratic() -> Result<()> {
        // f(x) = x[0]^2 + 2*x[1]^2
        let f = |x: &ArrayView1<f32>| -> scirs2_linalg::error::LinalgResult<f32> {
            Ok(x[0] * x[0] + 2.0 * x[1] * x[1])
        };

        // At x = [1.0, 1.0], gradient should be [2.0, 4.0]
        let x = Tensor::from_data(vec![1.0f32, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let grad = gradient(f, &x, None)?;

        assert_eq!(grad.shape().dims(), &[2]);
        assert_relative_eq!(grad.get(&[0])?, 2.0, epsilon = 1e-3);
        assert_relative_eq!(grad.get(&[1])?, 4.0, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_jacobian_linear() -> Result<()> {
        // f(x) = [2*x[0], 3*x[1]]
        let f = |x: &ArrayView1<f32>| -> scirs2_linalg::error::LinalgResult<Array1<f32>> {
            Ok(Array1::from_vec(vec![2.0 * x[0], 3.0 * x[1]]))
        };

        // Jacobian should be [[2, 0], [0, 3]]
        let x = Tensor::from_data(vec![1.0f32, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let jac = jacobian(f, &x, None)?;

        assert_eq!(jac.shape().dims(), &[2, 2]);
        assert_relative_eq!(jac.get(&[0, 0])?, 2.0, epsilon = 1e-3);
        assert_relative_eq!(jac.get(&[0, 1])?, 0.0, epsilon = 1e-3);
        assert_relative_eq!(jac.get(&[1, 0])?, 0.0, epsilon = 1e-3);
        assert_relative_eq!(jac.get(&[1, 1])?, 3.0, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_hessian_quadratic() -> Result<()> {
        // f(x) = x[0]^2 + x[1]^2 + x[0]*x[1]
        let f = |x: &ArrayView1<f32>| -> scirs2_linalg::error::LinalgResult<f32> {
            Ok(x[0] * x[0] + x[1] * x[1] + x[0] * x[1])
        };

        // Hessian should be [[2, 1], [1, 2]]
        let x = Tensor::from_data(vec![0.0f32, 0.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let hess = hessian(f, &x, None)?;

        assert_eq!(hess.shape().dims(), &[2, 2]);
        assert_relative_eq!(hess.get(&[0, 0])?, 2.0, epsilon = 1e-3);
        assert_relative_eq!(hess.get(&[0, 1])?, 1.0, epsilon = 1e-3);
        assert_relative_eq!(hess.get(&[1, 0])?, 1.0, epsilon = 1e-3);
        assert_relative_eq!(hess.get(&[1, 1])?, 2.0, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_directional_derivative() -> Result<()> {
        // f(x) = x[0]^2 + x[1]^2
        let f = |x: &ArrayView1<f32>| -> scirs2_linalg::error::LinalgResult<f32> {
            Ok(x[0] * x[0] + x[1] * x[1])
        };

        // At x = [1, 1], gradient is [2, 2]
        // Directional derivative in direction [1, 0] should be 2.0
        let x = Tensor::from_data(vec![1.0f32, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let direction = Tensor::from_data(vec![1.0f32, 0.0], vec![2], torsh_core::DeviceType::Cpu)?;

        let dd = directional_derivative(f, &x, &direction, None)?;
        assert_relative_eq!(dd, 2.0, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_dimension_validation() {
        // Test with wrong dimensions
        let x = Tensor::from_data(
            vec![1.0f32, 1.0, 1.0, 1.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let f = |_: &ArrayView1<f32>| -> scirs2_linalg::error::LinalgResult<f32> { Ok(0.0) };

        // Should fail because x is 2D
        let result = gradient(f, &x, None);
        assert!(result.is_err());
    }
}
