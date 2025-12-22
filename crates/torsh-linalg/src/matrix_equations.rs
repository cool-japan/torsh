//! Advanced matrix equation solvers
//!
//! This module provides PyTorch-compatible solvers for advanced matrix equations beyond
//! simple Ax = b, including Sylvester, Lyapunov, and Riccati equations.
//!
//! ## Features
//!
//! - **Sylvester Equation**: AX + XB = C (fundamental in control theory)
//! - **Lyapunov Equation**: AX + XA^T = C (stability analysis)
//! - **Riccati Equation**: Continuous and discrete-time algebraic Riccati equations
//! - **Stein Equation**: AXA^T - X + Q = 0 (discrete-time Lyapunov)
//!
//! ## Examples
//!
//! ```ignore
//! use torsh_linalg::matrix_equations::solve_sylvester;
//! use torsh_tensor::Tensor;
//!
//! // Solve AX + XB = C
//! let a = Tensor::from_slice(&[1.0, 0.0, 0.0, 2.0], &[2, 2])?;
//! let b = Tensor::from_slice(&[3.0, 0.0, 0.0, 4.0], &[2, 2])?;
//! let c = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
//!
//! let x = solve_sylvester(&a, &b, &c)?;
//! ```

use torsh_core::{Result, TorshError};
use torsh_tensor::Tensor;

#[cfg(feature = "scirs2-integration")]
use scirs2_core::ndarray::{Array2, ArrayView2};

/// Solve the Sylvester equation: AX + XB = C
///
/// The Sylvester equation is a fundamental matrix equation that appears in many
/// applications including control theory, signal processing, and dynamical systems.
///
/// # Arguments
///
/// * `a` - Matrix A of shape [m, m]
/// * `b` - Matrix B of shape [n, n]
/// * `c` - Matrix C of shape [m, n]
///
/// # Returns
///
/// Solution matrix X of shape [m, n]
///
/// # Notes
///
/// The equation has a unique solution if and only if A and -B have no common eigenvalues.
#[cfg(feature = "scirs2-integration")]
pub fn solve_sylvester(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    // Validate inputs
    if a.shape().ndim() != 2 || b.shape().ndim() != 2 || c.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Sylvester equation requires 2D tensors".to_string(),
        ));
    }

    let a_shape_binding = a.shape();
    let a_shape = a_shape_binding.dims();
    let b_shape_binding = b.shape();
    let b_shape = b_shape_binding.dims();
    let c_shape_binding = c.shape();
    let c_shape = c_shape_binding.dims();

    // Validate A is square
    if a_shape[0] != a_shape[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be square".to_string(),
        ));
    }

    // Validate B is square
    if b_shape[0] != b_shape[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix B must be square".to_string(),
        ));
    }

    // Validate C dimensions match
    if c_shape[0] != a_shape[0] || c_shape[1] != b_shape[0] {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix C must have shape [{}, {}], got [{}, {}]",
            a_shape[0], b_shape[0], c_shape[0], c_shape[1]
        )));
    }

    // Convert tensors to ndarray
    let a_array = tensor_to_array2(a)?;
    let b_array = tensor_to_array2(b)?;
    let c_array = tensor_to_array2(c)?;

    // Solve using scirs2-linalg
    let x_array = scirs2_linalg::matrix_equations::solve_sylvester(
        &a_array.view(),
        &b_array.view(),
        &c_array.view(),
    )
    .map_err(|e| TorshError::ComputeError(format!("Sylvester solver failed: {e}")))?;

    // Convert back to tensor
    array2_to_tensor(&x_array.view(), a.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn solve_sylvester(_a: &Tensor, _b: &Tensor, _c: &Tensor) -> Result<Tensor> {
    Err(TorshError::NotImplemented(
        "Sylvester solver requires scirs2-integration feature".to_string(),
    ))
}

/// Solve the Lyapunov equation: AX + XA^T = C
///
/// The Lyapunov equation is a special case of the Sylvester equation where B = A^T.
/// It's fundamental in stability analysis of dynamical systems.
///
/// # Arguments
///
/// * `a` - Square matrix A of shape [n, n]
/// * `c` - Symmetric matrix C of shape [n, n]
///
/// # Returns
///
/// Solution matrix X of shape [n, n]
///
/// # Notes
///
/// For a stable system (all eigenvalues of A have negative real parts),
/// if C is positive semi-definite, then X will also be positive semi-definite.
#[cfg(feature = "scirs2-integration")]
pub fn solve_lyapunov(a: &Tensor, c: &Tensor) -> Result<Tensor> {
    // Validate inputs
    if a.shape().ndim() != 2 || c.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Lyapunov equation requires 2D tensors".to_string(),
        ));
    }

    let a_shape_binding = a.shape();
    let a_shape = a_shape_binding.dims();
    let c_shape_binding = c.shape();
    let c_shape = c_shape_binding.dims();

    // Validate A is square
    if a_shape[0] != a_shape[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be square".to_string(),
        ));
    }

    // Validate C is square and matches A
    if c_shape[0] != c_shape[1] || c_shape[0] != a_shape[0] {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix C must be square with same size as A, got [{}, {}]",
            c_shape[0], c_shape[1]
        )));
    }

    // Lyapunov equation AX + XA^T = C is equivalent to Sylvester AX + XA^T = C
    // So we solve with B = A^T
    let a_array = tensor_to_array2(a)?;
    let c_array = tensor_to_array2(c)?;

    // Transpose A for B
    let b_array = a_array.t().to_owned();

    // Solve Sylvester equation
    let x_array = scirs2_linalg::matrix_equations::solve_sylvester(
        &a_array.view(),
        &b_array.view(),
        &c_array.view(),
    )
    .map_err(|e| TorshError::ComputeError(format!("Lyapunov solver failed: {e}")))?;

    array2_to_tensor(&x_array.view(), a.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn solve_lyapunov(_a: &Tensor, _c: &Tensor) -> Result<Tensor> {
    Err(TorshError::NotImplemented(
        "Lyapunov solver requires scirs2-integration feature".to_string(),
    ))
}

/// Solve the continuous-time algebraic Riccati equation (CARE)
///
/// Solves: A^T X + X A - X B R^{-1} B^T X + Q = 0
///
/// This equation appears in optimal control problems, particularly in Linear-Quadratic
/// Regulator (LQR) design.
///
/// # Arguments
///
/// * `a` - State matrix of shape [n, n]
/// * `b` - Input matrix of shape [n, m]
/// * `q` - State cost matrix of shape [n, n] (must be symmetric positive semi-definite)
/// * `r` - Input cost matrix of shape [m, m] (must be symmetric positive definite)
///
/// # Returns
///
/// Solution matrix X of shape [n, n] (symmetric positive semi-definite)
///
/// # Notes
///
/// The optimal feedback gain for LQR is K = R^{-1} B^T X
#[cfg(feature = "scirs2-integration")]
pub fn solve_continuous_riccati(a: &Tensor, b: &Tensor, q: &Tensor, r: &Tensor) -> Result<Tensor> {
    // Validate inputs
    if a.shape().ndim() != 2
        || b.shape().ndim() != 2
        || q.shape().ndim() != 2
        || r.shape().ndim() != 2
    {
        return Err(TorshError::InvalidArgument(
            "Riccati equation requires 2D tensors".to_string(),
        ));
    }

    let a_shape_binding = a.shape();
    let a_shape = a_shape_binding.dims();
    let b_shape_binding = b.shape();
    let b_shape = b_shape_binding.dims();
    let q_shape_binding = q.shape();
    let q_shape = q_shape_binding.dims();
    let r_shape_binding = r.shape();
    let r_shape = r_shape_binding.dims();

    let n = a_shape[0];
    let m = b_shape[1];

    // Validate dimensions
    if a_shape[0] != a_shape[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be square".to_string(),
        ));
    }

    if b_shape[0] != n {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix B must have {} rows, got {}",
            n, b_shape[0]
        )));
    }

    if q_shape[0] != n || q_shape[1] != n {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix Q must be {}x{}, got {}x{}",
            n, n, q_shape[0], q_shape[1]
        )));
    }

    if r_shape[0] != m || r_shape[1] != m {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix R must be {}x{}, got {}x{}",
            m, m, r_shape[0], r_shape[1]
        )));
    }

    // Convert tensors to ndarray
    let a_array = tensor_to_array2(a)?;
    let b_array = tensor_to_array2(b)?;
    let q_array = tensor_to_array2(q)?;
    let r_array = tensor_to_array2(r)?;

    // Solve using scirs2-linalg
    let x_array = scirs2_linalg::matrix_equations::solve_continuous_riccati(
        &a_array.view(),
        &b_array.view(),
        &q_array.view(),
        &r_array.view(),
    )
    .map_err(|e| TorshError::ComputeError(format!("Continuous Riccati solver failed: {e}")))?;

    array2_to_tensor(&x_array.view(), a.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn solve_continuous_riccati(
    _a: &Tensor,
    _b: &Tensor,
    _q: &Tensor,
    _r: &Tensor,
) -> Result<Tensor> {
    Err(TorshError::NotImplemented(
        "Continuous Riccati solver requires scirs2-integration feature".to_string(),
    ))
}

/// Solve the discrete-time algebraic Riccati equation (DARE)
///
/// Solves: A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q = 0
///
/// This equation appears in discrete-time optimal control problems.
///
/// # Arguments
///
/// * `a` - State matrix of shape [n, n]
/// * `b` - Input matrix of shape [n, m]
/// * `q` - State cost matrix of shape [n, n] (must be symmetric positive semi-definite)
/// * `r` - Input cost matrix of shape [m, m] (must be symmetric positive definite)
///
/// # Returns
///
/// Solution matrix X of shape [n, n] (symmetric positive semi-definite)
///
/// # Notes
///
/// The optimal feedback gain for discrete LQR is K = (R + B^T X B)^{-1} B^T X A
#[cfg(feature = "scirs2-integration")]
pub fn solve_discrete_riccati(a: &Tensor, b: &Tensor, q: &Tensor, r: &Tensor) -> Result<Tensor> {
    // Validate inputs (same validation as continuous case)
    if a.shape().ndim() != 2
        || b.shape().ndim() != 2
        || q.shape().ndim() != 2
        || r.shape().ndim() != 2
    {
        return Err(TorshError::InvalidArgument(
            "Riccati equation requires 2D tensors".to_string(),
        ));
    }

    let a_shape_binding = a.shape();
    let a_shape = a_shape_binding.dims();
    let b_shape_binding = b.shape();
    let b_shape = b_shape_binding.dims();
    let q_shape_binding = q.shape();
    let q_shape = q_shape_binding.dims();
    let r_shape_binding = r.shape();
    let r_shape = r_shape_binding.dims();

    let n = a_shape[0];
    let m = b_shape[1];

    // Validate dimensions
    if a_shape[0] != a_shape[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be square".to_string(),
        ));
    }

    if b_shape[0] != n {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix B must have {} rows, got {}",
            n, b_shape[0]
        )));
    }

    if q_shape[0] != n || q_shape[1] != n {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix Q must be {}x{}, got {}x{}",
            n, n, q_shape[0], q_shape[1]
        )));
    }

    if r_shape[0] != m || r_shape[1] != m {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix R must be {}x{}, got {}x{}",
            m, m, r_shape[0], r_shape[1]
        )));
    }

    // Convert tensors to ndarray
    let a_array = tensor_to_array2(a)?;
    let b_array = tensor_to_array2(b)?;
    let q_array = tensor_to_array2(q)?;
    let r_array = tensor_to_array2(r)?;

    // Solve using scirs2-linalg
    let x_array = scirs2_linalg::matrix_equations::solve_discrete_riccati(
        &a_array.view(),
        &b_array.view(),
        &q_array.view(),
        &r_array.view(),
    )
    .map_err(|e| TorshError::ComputeError(format!("Discrete Riccati solver failed: {e}")))?;

    array2_to_tensor(&x_array.view(), a.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn solve_discrete_riccati(
    _a: &Tensor,
    _b: &Tensor,
    _q: &Tensor,
    _r: &Tensor,
) -> Result<Tensor> {
    Err(TorshError::NotImplemented(
        "Discrete Riccati solver requires scirs2-integration feature".to_string(),
    ))
}

/// Solve the Stein equation: AXA^T - X + Q = 0
///
/// The Stein equation is the discrete-time analog of the Lyapunov equation.
/// It's used in stability analysis of discrete-time systems.
///
/// # Arguments
///
/// * `a` - Square matrix A of shape [n, n]
/// * `q` - Symmetric matrix Q of shape [n, n]
///
/// # Returns
///
/// Solution matrix X of shape [n, n]
///
/// # Notes
///
/// For a stable discrete-time system (all eigenvalues of A inside the unit circle),
/// if Q is positive semi-definite, then X will also be positive semi-definite.
#[cfg(feature = "scirs2-integration")]
pub fn solve_stein(a: &Tensor, q: &Tensor) -> Result<Tensor> {
    // Validate inputs
    if a.shape().ndim() != 2 || q.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Stein equation requires 2D tensors".to_string(),
        ));
    }

    let a_shape_binding = a.shape();
    let a_shape = a_shape_binding.dims();
    let q_shape_binding = q.shape();
    let q_shape = q_shape_binding.dims();

    // Validate A is square
    if a_shape[0] != a_shape[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be square".to_string(),
        ));
    }

    // Validate Q is square and matches A
    if q_shape[0] != q_shape[1] || q_shape[0] != a_shape[0] {
        return Err(TorshError::InvalidArgument(format!(
            "Matrix Q must be square with same size as A, got [{}, {}]",
            q_shape[0], q_shape[1]
        )));
    }

    // Convert tensors to ndarray
    let a_array = tensor_to_array2(a)?;
    let q_array = tensor_to_array2(q)?;

    // Solve using scirs2-linalg
    let x_array = scirs2_linalg::matrix_equations::solve_stein(&a_array.view(), &q_array.view())
        .map_err(|e| TorshError::ComputeError(format!("Stein solver failed: {e}")))?;

    array2_to_tensor(&x_array.view(), a.device())
}

#[cfg(not(feature = "scirs2-integration"))]
pub fn solve_stein(_a: &Tensor, _q: &Tensor) -> Result<Tensor> {
    Err(TorshError::NotImplemented(
        "Stein solver requires scirs2-integration feature".to_string(),
    ))
}

// Helper functions for tensor <-> ndarray conversions

#[cfg(feature = "scirs2-integration")]
fn tensor_to_array2(tensor: &Tensor) -> Result<Array2<f32>> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Expected 2D tensor".to_string(),
        ));
    }

    let shape_binding = tensor.shape();
    let shape = shape_binding.dims();
    let (rows, cols) = (shape[0], shape[1]);

    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            data.push(tensor.get(&[i, j])?);
        }
    }

    Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| TorshError::ComputeError(format!("Failed to create Array2: {e}")))
}

#[cfg(feature = "scirs2-integration")]
fn array2_to_tensor(array: &ArrayView2<f32>, device: torsh_core::DeviceType) -> Result<Tensor> {
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
    fn test_solve_sylvester_diagonal() -> Result<()> {
        // For diagonal A and B, the solution is simple: X[i,j] = C[i,j] / (A[i,i] + B[j,j])
        let a = Tensor::from_data(
            vec![1.0f32, 0.0, 0.0, 2.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let b = Tensor::from_data(
            vec![3.0f32, 0.0, 0.0, 4.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let c = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_sylvester(&a, &b, &c)?;

        // Check shape
        assert_eq!(x.shape().dims(), &[2, 2]);

        // For diagonal matrices: X[i,j] = C[i,j] / (A[i,i] + B[j,j])
        // X[0,0] = 1.0 / (1.0 + 3.0) = 0.25
        // X[0,1] = 2.0 / (1.0 + 4.0) = 0.4
        // X[1,0] = 3.0 / (2.0 + 3.0) = 0.6
        // X[1,1] = 4.0 / (2.0 + 4.0) = 0.666...
        assert_relative_eq!(x.get(&[0, 0])?, 0.25, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[0, 1])?, 0.4, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[1, 0])?, 0.6, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[1, 1])?, 0.666666, epsilon = 1e-4);

        Ok(())
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_solve_lyapunov_identity() -> Result<()> {
        // For A = -I, the equation becomes: -IX + X(-I) = C, or -2X = C, so X = -C/2
        let a = Tensor::from_data(
            vec![-1.0f32, 0.0, 0.0, -1.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let c = Tensor::from_data(
            vec![2.0f32, 0.0, 0.0, 4.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_lyapunov(&a, &c)?;

        // Check shape
        assert_eq!(x.shape().dims(), &[2, 2]);

        // X should be -C/2
        assert_relative_eq!(x.get(&[0, 0])?, -1.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[0, 1])?, 0.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[1, 0])?, 0.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[1, 1])?, -2.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    #[cfg(feature = "scirs2-integration")]
    fn test_solve_stein_identity() -> Result<()> {
        // For A = 0.5*I, the equation becomes: 0.25*I*X - X + Q = 0, or -0.75X = -Q, so X = Q/0.75
        let a = Tensor::from_data(
            vec![0.5f32, 0.0, 0.0, 0.5],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let q = Tensor::from_data(
            vec![1.0f32, 0.0, 0.0, 1.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_stein(&a, &q)?;

        // Check shape
        assert_eq!(x.shape().dims(), &[2, 2]);

        // X should be approximately Q/0.75 = Q * 4/3
        assert_relative_eq!(x.get(&[0, 0])?, 4.0 / 3.0, epsilon = 1e-4);
        assert_relative_eq!(x.get(&[1, 1])?, 4.0 / 3.0, epsilon = 1e-4);

        Ok(())
    }

    #[test]
    fn test_dimension_validation_sylvester() {
        // Test with non-square A
        let a =
            Tensor::from_data(vec![1.0f32; 6], vec![2, 3], torsh_core::DeviceType::Cpu).unwrap();
        let b =
            Tensor::from_data(vec![1.0f32; 4], vec![2, 2], torsh_core::DeviceType::Cpu).unwrap();
        let c =
            Tensor::from_data(vec![1.0f32; 4], vec![2, 2], torsh_core::DeviceType::Cpu).unwrap();

        let result = solve_sylvester(&a, &b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_validation_lyapunov() {
        // Test with non-square A
        let a =
            Tensor::from_data(vec![1.0f32; 6], vec![2, 3], torsh_core::DeviceType::Cpu).unwrap();
        let c =
            Tensor::from_data(vec![1.0f32; 4], vec![2, 2], torsh_core::DeviceType::Cpu).unwrap();

        let result = solve_lyapunov(&a, &c);
        assert!(result.is_err());
    }
}
