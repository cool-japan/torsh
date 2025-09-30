//! Core linear algebra solver functions
//!
//! This module contains the fundamental linear algebra solver functions extracted from the
//! main solve.rs module. These functions provide the core building blocks for solving
//! linear systems, matrix inversion, and least squares problems.
//!
//! # Functions
//!
//! - [`solve`] - Solve linear system Ax = b using LU decomposition
//! - [`solve_triangular`] - Solve triangular system Ax = b
//! - [`inv`] - Compute matrix inverse A^(-1)
//! - [`pinv`] - Compute Moore-Penrose pseudoinverse A^+
//! - [`lstsq`] - Solve least squares problem min ||Ax - b||_2
//!
//! # Example
//!
//! ```rust
//! use torsh_linalg::solvers::core::solve;
//! use torsh_tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let a = Tensor::from_data(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
//! let b = Tensor::from_data(vec![3.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
//! let x = solve(&a, &b)?;
//! # Ok(())
//! # }
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::manual_div_ceil)]

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Solve linear system Ax = b
///
/// Uses LU decomposition with partial pivoting to solve the system Ax = b.
/// Matrix A must be square and non-singular.
///
/// # Arguments
///
/// * `a` - Square coefficient matrix A (m×m)
/// * `b` - Right-hand side vector or matrix b (m×k)
///
/// # Returns
///
/// Solution vector or matrix x such that Ax = b
///
/// # Errors
///
/// Returns an error if:
/// - Matrix A is not 2D
/// - Matrix A is not square
/// - Dimensions of A and b don't match
/// - Matrix A is singular (determinant is zero)
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::core::solve;
/// use torsh_tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = Tensor::from_data(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
/// let b = Tensor::from_data(vec![3.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
/// let x = solve(&a, &b)?;
/// // x should be approximately [1.0, 1.0]
/// # Ok(())
/// # }
/// ```
pub fn solve(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    use crate::decomposition::lu;

    if a.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be 2D".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);

    if m != n {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be square".to_string(),
        ));
    }

    // Check dimensions match
    if b.shape().dims()[0] != m {
        return Err(TorshError::InvalidArgument(
            "Dimension mismatch between A and b".to_string(),
        ));
    }

    // Perform LU decomposition: PA = LU
    let (p, l, u) = lu(a)?;

    // Apply permutation to b: Pb
    let pb = if b.shape().ndim() == 1 {
        p.matmul(&b.unsqueeze(1)?)?.squeeze(1)?
    } else {
        p.matmul(b)?
    };

    // Solve Ly = Pb (forward substitution)
    let y = solve_triangular(&l, &pb, false)?;

    // Solve Ux = y (back substitution)
    let x = solve_triangular(&u, &y, true)?;

    Ok(x)
}

/// Solve triangular system Ax = b
///
/// Solves a triangular linear system using either forward substitution (lower triangular)
/// or back substitution (upper triangular).
///
/// # Arguments
///
/// * `a` - Square triangular matrix A (n×n)
/// * `b` - Right-hand side vector or matrix b (n×k)
/// * `upper` - If true, A is upper triangular; if false, A is lower triangular
///
/// # Returns
///
/// Solution vector or matrix x such that Ax = b
///
/// # Errors
///
/// Returns an error if:
/// - Matrix A is not 2D
/// - Matrix A is not square
/// - Dimensions of A and b don't match
/// - Matrix A is singular (has zero diagonal elements)
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::core::solve_triangular;
/// use torsh_tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Upper triangular matrix
/// let a = Tensor::from_data(vec![2.0, 1.0, 0.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
/// let b = Tensor::from_data(vec![3.0, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
/// let x = solve_triangular(&a, &b, true)?;
/// // x should be approximately [1.0, 1.0]
/// # Ok(())
/// # }
/// ```
pub fn solve_triangular(a: &Tensor, b: &Tensor, upper: bool) -> TorshResult<Tensor> {
    if a.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be 2D".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);

    if m != n {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be square".to_string(),
        ));
    }

    // Check dimensions match
    if b.shape().dims()[0] != m {
        return Err(TorshError::InvalidArgument(
            "Dimension mismatch between A and b".to_string(),
        ));
    }

    let b_cols = if b.shape().ndim() == 1 {
        1
    } else {
        b.shape().dims()[1]
    };

    // Initialize solution
    let x_shape = if b.shape().ndim() == 1 {
        vec![n]
    } else {
        vec![n, b_cols]
    };

    let x = torsh_tensor::creation::zeros::<f32>(&x_shape)?;

    if upper {
        // Back substitution for upper triangular system
        for col in 0..b_cols {
            for i in (0..n).rev() {
                let mut sum = if b.shape().ndim() == 1 {
                    b.get(&[i])?
                } else {
                    b.get(&[i, col])?
                };

                // Subtract contributions from already solved variables
                for j in (i + 1)..n {
                    let x_val = if b_cols == 1 {
                        x.get(&[j])?
                    } else {
                        x.get(&[j, col])?
                    };
                    sum -= a.get(&[i, j])? * x_val;
                }

                let diag = a.get(&[i, i])?;
                // Use relative tolerance for singularity detection
                let tolerance = (diag.abs() * 1e-12).max(1e-15);
                if diag.abs() < tolerance {
                    return Err(TorshError::InvalidArgument(format!(
                        "Matrix is singular: diagonal element {} has magnitude {}",
                        i,
                        diag.abs()
                    )));
                }

                let x_val = sum / diag;
                if b_cols == 1 {
                    x.set(&[i], x_val)?;
                } else {
                    x.set(&[i, col], x_val)?;
                }
            }
        }
    } else {
        // Forward substitution for lower triangular system
        for col in 0..b_cols {
            for i in 0..n {
                let mut sum = if b.shape().ndim() == 1 {
                    b.get(&[i])?
                } else {
                    b.get(&[i, col])?
                };

                // Subtract contributions from already solved variables
                for j in 0..i {
                    let x_val = if b_cols == 1 {
                        x.get(&[j])?
                    } else {
                        x.get(&[j, col])?
                    };
                    sum -= a.get(&[i, j])? * x_val;
                }

                let diag = a.get(&[i, i])?;
                // Use relative tolerance for singularity detection
                let tolerance = (diag.abs() * 1e-12).max(1e-15);
                if diag.abs() < tolerance {
                    return Err(TorshError::InvalidArgument(format!(
                        "Matrix is singular: diagonal element {} has magnitude {}",
                        i,
                        diag.abs()
                    )));
                }

                let x_val = sum / diag;
                if b_cols == 1 {
                    x.set(&[i], x_val)?;
                } else {
                    x.set(&[i, col], x_val)?;
                }
            }
        }
    }

    Ok(x)
}

/// Compute matrix inverse A^(-1)
///
/// Finds the matrix inverse A^(-1) such that A * A^(-1) = I, where I is the identity matrix.
/// Uses the general linear solver internally.
///
/// # Arguments
///
/// * `tensor` - Square matrix A to invert (n×n)
///
/// # Returns
///
/// Inverse matrix A^(-1)
///
/// # Errors
///
/// Returns an error if:
/// - Matrix is not 2D
/// - Matrix is not square
/// - Matrix is singular (not invertible)
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::core::inv;
/// use torsh_tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = Tensor::from_data(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
/// let a_inv = inv(&a)?;
/// // a_inv should be approximately [[1.0, -1.0], [-1.0, 2.0]]
/// # Ok(())
/// # }
/// ```
pub fn inv(tensor: &Tensor) -> TorshResult<Tensor> {
    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument("Matrix must be 2D".to_string()));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);

    if m != n {
        return Err(TorshError::InvalidArgument(
            "Matrix must be square".to_string(),
        ));
    }

    // Create identity matrix
    let eye = torsh_tensor::creation::eye::<f32>(n)?;

    // Solve AX = I to get X = A^(-1)
    // Each column of X is the solution to Ax_i = e_i
    solve(tensor, &eye)
}

/// Compute Moore-Penrose pseudoinverse A^+
///
/// Computes the Moore-Penrose pseudoinverse using SVD decomposition.
/// For a matrix A, the pseudoinverse A^+ satisfies:
/// - A * A^+ * A = A
/// - A^+ * A * A^+ = A^+
/// - (A * A^+)^T = A * A^+
/// - (A^+ * A)^T = A^+ * A
///
/// # Arguments
///
/// * `tensor` - Matrix A to compute pseudoinverse for (m×n)
/// * `rcond` - Relative condition number cutoff for small singular values
///
/// # Returns
///
/// Pseudoinverse matrix A^+ (n×m)
///
/// # Errors
///
/// Returns an error if:
/// - Matrix is not 2D
/// - SVD decomposition fails
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::core::pinv;
/// use torsh_tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = Tensor::from_data(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
/// let a_pinv = pinv(&a, Some(1e-15))?;
/// // a_pinv is the pseudoinverse of a
/// # Ok(())
/// # }
/// ```
pub fn pinv(tensor: &Tensor, rcond: Option<f32>) -> TorshResult<Tensor> {
    use crate::decomposition::svd;

    if tensor.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Pseudoinverse requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (tensor.shape().dims()[0], tensor.shape().dims()[1]);

    // Compute SVD: A = U * S * V^T
    let (u, s, vt) = svd(tensor, true)?;

    // Set tolerance for small singular values
    let rcond = rcond.unwrap_or(1e-15);
    let max_sv = {
        let mut max_val = 0.0f32;
        let min_dim = m.min(n);
        for i in 0..min_dim {
            let sv = s.get(&[i])?;
            if sv > max_val {
                max_val = sv;
            }
        }
        max_val
    };
    let tol = rcond * max_sv;

    // Create S^+ by taking reciprocals of non-zero singular values
    // Optimize memory allocation for diagonal matrix
    let mut s_pinv_data = vec![0.0f32; n * m];
    let min_dim = m.min(n);

    for i in 0..min_dim {
        let sv = s.get(&[i])?;
        if sv.abs() > tol {
            s_pinv_data[i * m + i] = 1.0 / sv;
        }
    }

    let s_pinv = torsh_tensor::Tensor::from_data(s_pinv_data, vec![n, m], tensor.device())?;

    // Compute A^+ = V * S^+ * U^T
    // First compute V from V^T
    let v = vt.transpose(-2, -1)?;

    // Compute V * S^+
    let vs_pinv = v.matmul(&s_pinv)?;

    // Compute (V * S^+) * U^T
    let ut = u.transpose(-2, -1)?;
    let result = vs_pinv.matmul(&ut)?;

    Ok(result)
}

/// Solve least squares problem min ||Ax - b||_2
///
/// Solves the least squares problem using SVD-based pseudoinverse approach.
/// Finds the solution x that minimizes the 2-norm of the residual ||Ax - b||_2.
///
/// # Arguments
///
/// * `a` - Coefficient matrix A (m×n)
/// * `b` - Right-hand side vector or matrix b (m×k)
/// * `rcond` - Relative condition number cutoff for rank determination
///
/// # Returns
///
/// A tuple containing:
/// - `x`: The least squares solution (n×k)
/// - `residuals`: Sum of squared residuals for each column (k×1, empty for underdetermined)
/// - `rank`: The effective rank of matrix A (1×1)
/// - `s`: The singular values of A
///
/// # Errors
///
/// Returns an error if:
/// - Matrix A is not 2D
/// - Dimensions of A and b don't match
/// - SVD decomposition fails
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::core::lstsq;
/// use torsh_tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Simple system for demonstration
/// let a = Tensor::from_data(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
/// let b = Tensor::from_data(vec![3.0, 2.0], vec![2, 1], torsh_core::DeviceType::Cpu)?;
/// let (x, residuals, rank, s) = lstsq(&a, &b, None)?;
/// // x contains the least squares solution
/// # Ok(())
/// # }
/// ```
pub fn lstsq(
    a: &Tensor,
    b: &Tensor,
    rcond: Option<f32>,
) -> TorshResult<(Tensor, Tensor, Tensor, Tensor)> {
    use crate::decomposition::svd;
    use crate::matrix_rank;

    if a.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Matrix A must be 2D".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);

    // Check dimensions match
    if b.shape().dims()[0] != m {
        return Err(TorshError::InvalidArgument(
            "Dimension mismatch between A and b".to_string(),
        ));
    }

    // Compute SVD of A
    let (_u, s, _vt) = svd(a, true)?;

    // s is already a 1D tensor from SVD, just return it
    let s_tensor = s;

    // Compute rank
    let rank = matrix_rank(a, rcond)?;
    let rank_tensor = torsh_tensor::Tensor::from_data(vec![rank as f32], vec![1], a.device())?;

    // Compute the least squares solution using pseudoinverse
    let x = {
        let a_pinv = pinv(a, rcond)?;
        a_pinv.matmul(b)?
    };

    // Compute residuals if overdetermined (m > n)
    let residuals = if m > n && rank == n {
        // Compute Ax
        let ax = a.matmul(&x)?;

        // Compute residual = b - Ax
        let residual = b.sub(&ax)?;

        // Compute sum of squared residuals for each column of b
        let b_cols = if b.shape().ndim() == 1 {
            1
        } else {
            b.shape().dims()[1]
        };
        let mut residual_sums = vec![0.0f32; b_cols];

        for col in 0..b_cols {
            let mut sum = 0.0f32;
            for row in 0..m {
                let val = if b.shape().ndim() == 1 {
                    residual.get(&[row])?
                } else {
                    residual.get(&[row, col])?
                };
                sum += val * val;
            }
            residual_sums[col] = sum;
        }

        if b_cols == 1 && b.shape().ndim() == 1 {
            torsh_tensor::Tensor::from_data(vec![residual_sums[0]], vec![1], a.device())?
        } else {
            torsh_tensor::Tensor::from_data(residual_sums, vec![b_cols], a.device())?
        }
    } else {
        // Return empty tensor for underdetermined systems
        torsh_tensor::creation::zeros::<f32>(&[0])?
    };

    Ok((x, residuals, rank_tensor, s_tensor))
}
