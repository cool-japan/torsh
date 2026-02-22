//! Sparse linear algebra operations and preconditioners
//!
//! This module provides sparse matrix solvers and preconditioners for iterative methods.
//! Note: This is a foundational implementation with stubs for future development.

#![allow(clippy::needless_range_loop)]

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Compute vector 2-norm efficiently
fn vector_norm_sparse(tensor: &Tensor) -> TorshResult<f32> {
    let n = tensor.shape().dims()[0];
    let mut sum = 0.0f32;

    for i in 0..n {
        let val = tensor.get(&[i])?;
        sum += val * val;
    }

    Ok(sum.sqrt())
}

/// Compute inner product efficiently
fn vector_dot_sparse(a: &Tensor, b: &Tensor) -> TorshResult<f32> {
    let n = a.shape().dims()[0];
    let mut sum = 0.0f32;

    for i in 0..n {
        sum += a.get(&[i])? * b.get(&[i])?;
    }

    Ok(sum)
}

/// Element-wise multiplication of two vectors
fn vector_hadamard(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    let n = a.shape().dims()[0];
    let mut result_data = vec![0.0f32; n];

    for i in 0..n {
        result_data[i] = a.get(&[i])? * b.get(&[i])?;
    }

    Tensor::from_data(result_data, vec![n], a.device())
}

/// Trait for preconditioners
pub trait Preconditioner {
    /// Apply the preconditioner: solve P * x = b approximately
    fn apply(&self, b: &Tensor) -> TorshResult<Tensor>;

    /// Setup the preconditioner from matrix A
    fn setup(&mut self, a: &Tensor) -> TorshResult<()>;
}

/// Diagonal preconditioner (Jacobi)
///
/// Uses the diagonal elements of A as a simple preconditioner: P = diag(A)
pub struct DiagonalPreconditioner {
    diagonal_inv: Option<Tensor>,
}

impl DiagonalPreconditioner {
    pub fn new() -> Self {
        Self { diagonal_inv: None }
    }
}

impl Default for DiagonalPreconditioner {
    fn default() -> Self {
        Self::new()
    }
}

impl Preconditioner for DiagonalPreconditioner {
    fn apply(&self, b: &Tensor) -> TorshResult<Tensor> {
        match &self.diagonal_inv {
            Some(diag_inv) => {
                if b.shape().ndim() == 1 {
                    // Use optimized element-wise multiplication
                    Ok(vector_hadamard(b, diag_inv)?)
                } else {
                    Err(TorshError::InvalidArgument(
                        "Diagonal preconditioner currently supports only 1D tensors".to_string(),
                    ))
                }
            }
            None => Err(TorshError::InvalidArgument(
                "Preconditioner not initialized. Call setup() first".to_string(),
            )),
        }
    }

    fn setup(&mut self, a: &Tensor) -> TorshResult<()> {
        if a.shape().ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Diagonal preconditioner requires 2D matrix".to_string(),
            ));
        }

        let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
        if m != n {
            return Err(TorshError::InvalidArgument(
                "Diagonal preconditioner requires square matrix".to_string(),
            ));
        }

        // Extract diagonal and compute inverse
        let mut diag_inv_data = vec![0.0f32; n];
        for i in 0..n {
            let diag_val = a.get(&[i, i])?;
            if diag_val.abs() < 1e-12 {
                return Err(TorshError::InvalidArgument(format!(
                    "Matrix has zero diagonal element at position {i}"
                )));
            }
            diag_inv_data[i] = 1.0 / diag_val;
        }

        self.diagonal_inv = Some(Tensor::from_data(diag_inv_data, vec![n], a.device())?);
        Ok(())
    }
}

/// Conjugate Gradient solver for symmetric positive definite systems
///
/// Solves Ax = b where A is symmetric positive definite using the CG method.
/// This is a basic implementation suitable for dense matrices.
pub fn conjugate_gradient(
    a: &Tensor,
    b: &Tensor,
    x0: Option<&Tensor>,
    tol: Option<f32>,
    max_iter: Option<usize>,
    preconditioner: Option<&dyn Preconditioner>,
) -> TorshResult<(Tensor, usize, f32)> {
    if a.shape().ndim() != 2 || b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(format!(
            "CG solver requires 2D matrix A and 1D vector b, got {}D matrix and {}D vector",
            a.shape().ndim(),
            b.shape().ndim()
        )));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
    if m != n {
        return Err(TorshError::InvalidArgument(format!(
            "CG solver requires square matrix, got {m}x{n} matrix"
        )));
    }

    if b.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(format!(
            "Dimension mismatch: matrix A is {}x{} but vector b has length {}",
            a.shape().dims()[0],
            a.shape().dims()[1],
            b.shape().dims()[0]
        )));
    }

    let tol = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(n);

    // Initialize solution vector
    let mut x = match x0 {
        Some(x_init) => x_init.clone(),
        None => torsh_tensor::creation::zeros::<f32>(&[n])?,
    };

    // Compute initial residual: r = b - Ax
    let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
    let mut r = b.sub(&ax)?;

    // Apply preconditioner if provided
    let mut z = match preconditioner {
        Some(prec) => prec.apply(&r)?,
        None => r.clone(),
    };

    let mut p = z.clone();

    // Initial residual norm
    let mut rz_dot = inner_product(&r, &z)?;
    let initial_rz_dot = rz_dot;

    for iter in 0..max_iter {
        // Check convergence
        if rz_dot.sqrt() < tol * initial_rz_dot.sqrt() {
            return Ok((x, iter, rz_dot.sqrt()));
        }

        // Compute Ap
        let ap = a.matmul(&p.unsqueeze(1)?)?.squeeze(1)?;

        // Compute step size: alpha = (r^T z) / (p^T Ap)
        let pap = inner_product(&p, &ap)?;
        if pap.abs() < 1e-12 {
            return Err(TorshError::InvalidArgument(
                "CG breakdown: p^T A p = 0".to_string(),
            ));
        }
        let alpha = rz_dot / pap;

        // Update solution: x = x + alpha * p
        x = x.add(&p.mul_scalar(alpha)?)?;

        // Update residual: r = r - alpha * Ap
        r = r.sub(&ap.mul_scalar(alpha)?)?;

        // Apply preconditioner
        z = match preconditioner {
            Some(prec) => prec.apply(&r)?,
            None => r.clone(),
        };

        // Compute new rz_dot
        let new_rz_dot = inner_product(&r, &z)?;

        // Compute beta for next iteration
        let beta = new_rz_dot / rz_dot;

        // Update search direction: p = z + beta * p
        p = z.add(&p.mul_scalar(beta)?)?;

        rz_dot = new_rz_dot;
    }

    Ok((x, max_iter, rz_dot.sqrt()))
}

/// GMRES solver for general linear systems
///
/// Implements the Generalized Minimal Residual method for solving Ax = b.
/// This solver works for general (non-symmetric) linear systems.
///
/// # Arguments
/// * `a` - The system matrix A (n×n)
/// * `b` - The right-hand side vector b (n×1)
/// * `x0` - Initial guess (optional, defaults to zero vector)
/// * `tol` - Convergence tolerance (optional, defaults to 1e-6)
/// * `max_iter` - Maximum number of iterations (optional, defaults to n)
/// * `restart` - Restart parameter for GMRES(m) (optional, defaults to n)
///
/// # Returns
/// * `(x, iterations, residual_norm)` - Solution vector, number of iterations, and final residual norm
pub fn gmres(
    a: &Tensor,
    b: &Tensor,
    x0: Option<&Tensor>,
    tol: Option<f32>,
    max_iter: Option<usize>,
    restart: Option<usize>,
) -> TorshResult<(Tensor, usize, f32)> {
    if a.shape().ndim() != 2 || b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "GMRES solver requires 2D matrix A and 1D vector b".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
    if m != n {
        return Err(TorshError::InvalidArgument(
            "GMRES solver requires square matrix".to_string(),
        ));
    }

    if b.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(format!(
            "Dimension mismatch: matrix A is {}x{} but vector b has length {}",
            a.shape().dims()[0],
            a.shape().dims()[1],
            b.shape().dims()[0]
        )));
    }

    let tol = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(n);
    let restart = restart.unwrap_or(n.min(30)); // Limit restart to avoid excessive memory usage

    // Initialize solution vector
    let mut x = match x0 {
        Some(x_init) => x_init.clone(),
        None => torsh_tensor::creation::zeros::<f32>(&[n])?,
    };

    let mut total_iterations = 0;
    let mut final_residual_norm = 0.0f32;

    // Main GMRES iteration loop (with restarts)
    for _restart_iter in 0..max_iter.div_ceil(restart) {
        if total_iterations >= max_iter {
            break;
        }

        // Compute initial residual: r = b - Ax
        let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
        let r = b.sub(&ax)?;

        // Compute initial residual norm
        let beta = vector_norm(&r)?;

        // Check for convergence
        if beta < tol {
            final_residual_norm = beta;
            break;
        }

        // Initialize Krylov subspace
        let mut v = vec![Tensor::zeros(&[n], b.device())?; restart + 1];

        // v[0] = r / ||r||
        v[0] = r.div_scalar(beta)?;

        // Hessenberg matrix H (upper Hessenberg)
        let mut h = vec![vec![0.0f32; restart]; restart + 1];

        let mut j = 0;
        while j < restart && total_iterations < max_iter {
            // Arnoldi iteration
            let mut w = a.matmul(&v[j].unsqueeze(1)?)?.squeeze(1)?;

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = inner_product(&v[i], &w)?;
                let temp = v[i].mul_scalar(h[i][j])?;
                w = w.sub(&temp)?;
            }

            h[j + 1][j] = vector_norm(&w)?;

            if h[j + 1][j] < 1e-12 {
                // Lucky breakdown - solution is in the current Krylov subspace
                break;
            }

            if j + 1 < v.len() {
                v[j + 1] = w.div_scalar(h[j + 1][j])?;
            }

            j += 1;
            total_iterations += 1;
        }

        // Solve least squares problem: minimize ||beta * e1 - Hy||
        let mut rhs = vec![0.0f32; j + 1];
        rhs[0] = beta;

        let y = solve_upper_hessenberg(&h, &rhs, j)?;

        // Update solution: x = x + V * y
        for i in 0..j {
            let contribution = v[i].mul_scalar(y[i])?;
            x = x.add(&contribution)?;
        }

        // Check convergence
        let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
        let residual = b.sub(&ax)?;
        final_residual_norm = vector_norm(&residual)?;

        if final_residual_norm < tol {
            break;
        }
    }

    Ok((x, total_iterations, final_residual_norm))
}

/// BiCGSTAB solver for general linear systems
///
/// Implements the Bi-Conjugate Gradient Stabilized method for solving Ax = b.
/// This solver works for general (non-symmetric) linear systems and is often
/// more stable than CG for non-symmetric matrices.
///
/// # Arguments
/// * `a` - The system matrix A (n×n)
/// * `b` - The right-hand side vector b (n×1)
/// * `x0` - Initial guess (optional, defaults to zero vector)
/// * `tol` - Convergence tolerance (optional, defaults to 1e-6)
/// * `max_iter` - Maximum number of iterations (optional, defaults to n)
///
/// # Returns
/// * `(x, iterations, residual_norm)` - Solution vector, number of iterations, and final residual norm
pub fn bicgstab(
    a: &Tensor,
    b: &Tensor,
    x0: Option<&Tensor>,
    tol: Option<f32>,
    max_iter: Option<usize>,
) -> TorshResult<(Tensor, usize, f32)> {
    if a.shape().ndim() != 2 || b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "BiCGSTAB solver requires 2D matrix A and 1D vector b".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
    if m != n {
        return Err(TorshError::InvalidArgument(
            "BiCGSTAB solver requires square matrix".to_string(),
        ));
    }

    if b.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(format!(
            "Dimension mismatch: matrix A is {}x{} but vector b has length {}",
            a.shape().dims()[0],
            a.shape().dims()[1],
            b.shape().dims()[0]
        )));
    }

    let tol = tol.unwrap_or(1e-6);
    let max_iter = max_iter.unwrap_or(n);

    // Initialize solution vector
    let mut x = match x0 {
        Some(x_init) => x_init.clone(),
        None => torsh_tensor::creation::zeros::<f32>(&[n])?,
    };

    // Compute initial residual: r = b - Ax
    let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
    let mut r = b.sub(&ax)?;

    // Choose r0 (arbitrary, often r0 = r)
    let r0 = r.clone();

    // Initialize vectors
    let mut p = r.clone();
    let mut v = torsh_tensor::creation::zeros::<f32>(&[n])?;

    // Initial values
    let mut rho = inner_product(&r0, &r)?;
    let mut alpha = 1.0f32;
    let mut omega = 1.0f32;

    let initial_residual_norm = vector_norm(&r)?;
    let mut residual_norm = initial_residual_norm;

    for iter in 0..max_iter {
        // Check convergence
        if residual_norm < tol * initial_residual_norm {
            return Ok((x, iter, residual_norm));
        }

        let rho_new = inner_product(&r0, &r)?;

        if rho_new.abs() < 1e-12 {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: rho = 0".to_string(),
            ));
        }

        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        let pv = p.sub(&v.mul_scalar(omega)?)?;
        p = r.add(&pv.mul_scalar(beta)?)?;

        // v = A * p
        v = a.matmul(&p.unsqueeze(1)?)?.squeeze(1)?;

        // alpha = rho / (r0^T * v)
        let r0v = inner_product(&r0, &v)?;
        if r0v.abs() < 1e-12 {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: r0^T * v = 0".to_string(),
            ));
        }
        alpha = rho / r0v;

        // s = r - alpha * v
        let s = r.sub(&v.mul_scalar(alpha)?)?;

        // Check for early convergence
        let s_norm = vector_norm(&s)?;
        if s_norm < tol * initial_residual_norm {
            // x = x + alpha * p
            x = x.add(&p.mul_scalar(alpha)?)?;
            residual_norm = s_norm;
            return Ok((x, iter + 1, residual_norm));
        }

        // t = A * s
        let t = a.matmul(&s.unsqueeze(1)?)?.squeeze(1)?;

        // omega = (t^T * s) / (t^T * t)
        let ts = inner_product(&t, &s)?;
        let tt = inner_product(&t, &t)?;

        if tt.abs() < 1e-12 {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: t^T * t = 0".to_string(),
            ));
        }
        omega = ts / tt;

        // Update solution: x = x + alpha * p + omega * s
        let alpha_p = p.mul_scalar(alpha)?;
        let omega_s = s.mul_scalar(omega)?;
        x = x.add(&alpha_p)?.add(&omega_s)?;

        // Update residual: r = s - omega * t
        r = s.sub(&t.mul_scalar(omega)?)?;

        residual_norm = vector_norm(&r)?;

        // Check for omega breakdown
        if omega.abs() < 1e-12 {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: omega = 0".to_string(),
            ));
        }
    }

    Ok((x, max_iter, residual_norm))
}

/// Helper function to compute inner product of two vectors (optimized)
fn inner_product(a: &Tensor, b: &Tensor) -> TorshResult<f32> {
    if a.shape().ndim() != 1 || b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(format!(
            "Inner product requires 1D tensors, got {}D and {}D tensors",
            a.shape().ndim(),
            b.shape().ndim()
        )));
    }

    let len_a = a.shape().dims()[0];
    let len_b = b.shape().dims()[0];

    if len_a != len_b {
        return Err(TorshError::InvalidArgument(format!(
            "Inner product requires vectors of same length, got {len_a} and {len_b}"
        )));
    }

    // Use optimized vector dot product
    vector_dot_sparse(a, b)
}

/// Helper function to compute vector 2-norm (optimized)
fn vector_norm(v: &Tensor) -> TorshResult<f32> {
    if v.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(format!(
            "Vector norm requires 1D tensor, got {}D tensor",
            v.shape().ndim()
        )));
    }

    // Use optimized vector norm computation
    vector_norm_sparse(v)
}

/// Helper function to solve upper Hessenberg least squares problem
fn solve_upper_hessenberg(h: &[Vec<f32>], rhs: &[f32], n: usize) -> TorshResult<Vec<f32>> {
    if n == 0 {
        return Ok(vec![]);
    }

    let mut y = vec![0.0f32; n];
    let mut b = rhs.to_vec();

    // Apply Givens rotations to reduce to upper triangular form
    let mut c = vec![0.0f32; n];
    let mut s = vec![0.0f32; n];

    for k in 0..n {
        if k < h.len() - 1 && k < h[k + 1].len() {
            let h_k = h[k][k];
            let h_k1 = h[k + 1][k];

            let denom = (h_k * h_k + h_k1 * h_k1).sqrt();
            if denom > 1e-12 {
                c[k] = h_k / denom;
                s[k] = h_k1 / denom;

                // Apply rotation to RHS
                if k + 1 < b.len() {
                    let temp = c[k] * b[k] + s[k] * b[k + 1];
                    b[k + 1] = -s[k] * b[k] + c[k] * b[k + 1];
                    b[k] = temp;
                }
            }
        }
    }

    // Back substitution
    for i in (0..n).rev() {
        if i < h.len() && i < h[i].len() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                if j < h[i].len() {
                    sum -= h[i][j] * y[j];
                }
            }

            if h[i][i].abs() > 1e-12 {
                y[i] = sum / h[i][i];
            } else {
                y[i] = 0.0; // Rank deficient case
            }
        }
    }

    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::eye;

    #[test]
    fn test_diagonal_preconditioner() -> TorshResult<()> {
        // Create a simple diagonal matrix
        let a = eye::<f32>(3)?;
        a.set(&[0, 0], 2.0)?;
        a.set(&[1, 1], 4.0)?;
        a.set(&[2, 2], 0.5)?;

        let mut prec = DiagonalPreconditioner::new();
        prec.setup(&a)?;

        let b = Tensor::from_data(vec![2.0, 4.0, 1.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let x = prec.apply(&b)?;

        // Should be [1.0, 1.0, 2.0] = [2/2, 4/4, 1/0.5]
        assert_relative_eq!(x.get(&[0])?, 1.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[1])?, 1.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[2])?, 2.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_conjugate_gradient() -> TorshResult<()> {
        // Create a simple SPD system: A = I, b = [1, 2, 3]
        let a = eye::<f32>(3)?;
        let b = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;

        let (x, iterations, residual) = conjugate_gradient(&a, &b, None, Some(1e-8), None, None)?;

        // Solution should be exactly b since A = I
        assert_relative_eq!(x.get(&[0])?, 1.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[1])?, 2.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[2])?, 3.0, epsilon = 1e-5);

        // Should converge in at most n iterations for exact arithmetic
        assert!(iterations <= 3);
        assert!(residual < 1e-6);

        Ok(())
    }

    #[test]
    fn test_gmres() -> TorshResult<()> {
        // Create a simple system: A = I, b = [1, 2, 3]
        let a = eye::<f32>(3)?;
        let b = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;

        let (x, iterations, residual) = gmres(&a, &b, None, Some(1e-8), None, None)?;

        // Solution should be exactly b since A = I
        assert_relative_eq!(x.get(&[0])?, 1.0, epsilon = 1e-4);
        assert_relative_eq!(x.get(&[1])?, 2.0, epsilon = 1e-4);
        assert_relative_eq!(x.get(&[2])?, 3.0, epsilon = 1e-4);

        // Should converge quickly for such a simple system
        assert!(iterations <= 10);
        assert!(residual < 1e-6);

        Ok(())
    }

    #[test]
    fn test_gmres_non_symmetric() -> TorshResult<()> {
        // Create a non-symmetric system
        let a = eye::<f32>(2)?;
        a.set(&[0, 1], 0.5)?; // Make it non-symmetric
        let b = Tensor::from_data(vec![1.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;

        let (x, _iterations, residual) = gmres(&a, &b, None, Some(1e-6), Some(10), Some(2))?;

        // Verify that Ax ≈ b
        let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i])?, b.get(&[i])?, epsilon = 1e-4);
        }

        assert!(residual < 1e-5);

        Ok(())
    }

    #[test]
    fn test_bicgstab() -> TorshResult<()> {
        // Create a simple system: A = I, b = [1, 2, 3]
        let a = eye::<f32>(3)?;
        let b = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;

        let (x, iterations, residual) = bicgstab(&a, &b, None, Some(1e-8), None)?;

        // Solution should be exactly b since A = I
        assert_relative_eq!(x.get(&[0])?, 1.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[1])?, 2.0, epsilon = 1e-5);
        assert_relative_eq!(x.get(&[2])?, 3.0, epsilon = 1e-5);

        // Should converge quickly for such a simple system
        assert!(iterations <= 10);
        assert!(residual < 1e-6);

        Ok(())
    }

    #[test]
    fn test_bicgstab_non_symmetric() -> TorshResult<()> {
        // Create a non-symmetric system
        let a = eye::<f32>(2)?;
        a.set(&[0, 1], 0.7)?; // Make it non-symmetric
        a.set(&[1, 0], 0.3)?;
        let b = Tensor::from_data(vec![2.0, 1.5], vec![2], torsh_core::DeviceType::Cpu)?;

        let (x, _iterations, residual) = bicgstab(&a, &b, None, Some(1e-6), None)?;

        // Verify that Ax ≈ b
        let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
        for i in 0..2 {
            assert_relative_eq!(ax.get(&[i])?, b.get(&[i])?, epsilon = 1e-4);
        }

        assert!(residual < 1e-5);

        Ok(())
    }

    #[test]
    fn test_vector_norm() -> TorshResult<()> {
        let v = Tensor::from_data(vec![3.0, 4.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let norm = vector_norm(&v)?;

        // ||[3, 4]|| = 5
        assert_relative_eq!(norm, 5.0, epsilon = 1e-6);

        Ok(())
    }
}
