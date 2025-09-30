//! Sparse linear algebra operations
//!
//! This module provides advanced sparse linear algebra functionality including
//! decompositions, solvers, and iterative methods.

use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};
use torsh_core::{Shape, TorshError};
use torsh_tensor::{
    creation::{ones, zeros},
    Tensor,
};

/// Incomplete LU decomposition for sparse matrices
pub struct IncompleteLU {
    /// Lower triangular matrix (CSR format)
    l_matrix: CsrTensor,
    /// Upper triangular matrix (CSR format)  
    u_matrix: CsrTensor,
    /// Original matrix shape
    shape: Shape,
}

impl IncompleteLU {
    /// Compute incomplete LU decomposition
    pub fn new(matrix: &dyn SparseTensor, fill_factor: f32) -> TorshResult<Self> {
        if matrix.shape().ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Matrix must be 2D for LU decomposition".to_string(),
            ));
        }

        let shape = matrix.shape().clone();
        let n = shape.dims()[0];

        if shape.dims()[0] != shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Matrix must be square for LU decomposition".to_string(),
            ));
        }

        // Convert to CSR for efficient row operations
        let csr = matrix.to_csr()?;

        // Initialize L and U matrices
        let mut l_rows = Vec::new();
        let mut l_cols = Vec::new();
        let mut l_vals = Vec::new();

        let mut u_rows = Vec::new();
        let mut u_cols = Vec::new();
        let mut u_vals = Vec::new();

        // Perform incomplete LU decomposition
        // This is a simplified version - full implementation would include
        // sophisticated fill-in control and pivoting

        for i in 0..n {
            // Add diagonal element to L (always 1)
            l_rows.push(i);
            l_cols.push(i);
            l_vals.push(1.0);

            let (cols, vals) = csr.get_row(i)?;

            // Split row into L and U parts
            for (j, &col) in cols.iter().enumerate() {
                let val = vals[j];

                if col < i {
                    // Lower triangular part
                    if val.abs() > fill_factor {
                        l_rows.push(i);
                        l_cols.push(col);
                        l_vals.push(val);
                    }
                } else {
                    // Upper triangular part (including diagonal)
                    if val.abs() > fill_factor {
                        u_rows.push(i);
                        u_cols.push(col);
                        u_vals.push(val);
                    }
                }
            }
        }

        // Create L and U matrices in CSR format
        let l_coo = CooTensor::new(l_rows, l_cols, l_vals, shape.clone())?;
        let u_coo = CooTensor::new(u_rows, u_cols, u_vals, shape.clone())?;

        let l_matrix = CsrTensor::from_coo(&l_coo)?;
        let u_matrix = CsrTensor::from_coo(&u_coo)?;

        Ok(IncompleteLU {
            l_matrix,
            u_matrix,
            shape,
        })
    }

    /// Solve Ly = b (forward substitution)
    pub fn solve_lower(&self, b: &Tensor) -> TorshResult<Tensor> {
        if b.shape().ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Right-hand side must be a vector".to_string(),
            ));
        }

        let n = self.shape.dims()[0];
        if b.shape().dims()[0] != n {
            return Err(TorshError::InvalidArgument(
                "Vector size doesn't match matrix dimensions".to_string(),
            ));
        }

        let y = zeros::<f32>(&[n])?;

        // Forward substitution: Ly = b
        for i in 0..n {
            let mut sum = 0.0;
            let (cols, vals) = self.l_matrix.get_row(i)?;

            for (k, &col) in cols.iter().enumerate() {
                if col < i {
                    sum += vals[k] * y.get(&[col])?;
                } else if col == i {
                    // Diagonal element (always 1 for L)
                    break;
                }
            }

            y.set(&[i], b.get(&[i])? - sum)?;
        }

        Ok(y)
    }

    /// Solve Ux = y (backward substitution)
    pub fn solve_upper(&self, y: &Tensor) -> TorshResult<Tensor> {
        if y.shape().ndim() != 1 {
            return Err(TorshError::InvalidArgument(
                "Right-hand side must be a vector".to_string(),
            ));
        }

        let n = self.shape.dims()[0];
        if y.shape().dims()[0] != n {
            return Err(TorshError::InvalidArgument(
                "Vector size doesn't match matrix dimensions".to_string(),
            ));
        }

        let x = zeros::<f32>(&[n])?;

        // Backward substitution: Ux = y
        for i in (0..n).rev() {
            let mut sum = 0.0;
            let mut diag_val = 1.0;
            let (cols, vals) = self.u_matrix.get_row(i)?;

            for (k, &col) in cols.iter().enumerate() {
                if col > i {
                    sum += vals[k] * x.get(&[col])?;
                } else if col == i {
                    diag_val = vals[k];
                }
            }

            if diag_val.abs() < f32::EPSILON {
                return Err(TorshError::InvalidArgument(
                    "Matrix is singular (zero diagonal element)".to_string(),
                ));
            }

            x.set(&[i], (y.get(&[i])? - sum) / diag_val)?;
        }

        Ok(x)
    }

    /// Solve the system LUx = b
    pub fn solve(&self, b: &Tensor) -> TorshResult<Tensor> {
        let y = self.solve_lower(b)?;
        self.solve_upper(&y)
    }
}

/// Conjugate Gradient method for solving sparse linear systems Ax = b
/// where A is symmetric positive definite
pub fn conjugate_gradient(
    matrix: &dyn SparseTensor,
    b: &Tensor,
    x0: Option<&Tensor>,
    tol: f32,
    max_iter: usize,
) -> TorshResult<(Tensor, usize, f32)> {
    if matrix.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument("Matrix must be 2D".to_string()));
    }

    let n = matrix.shape().dims()[0];
    if matrix.shape().dims()[0] != matrix.shape().dims()[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix must be square".to_string(),
        ));
    }

    if b.shape().ndim() != 1 || b.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(
            "Vector b must match matrix dimensions".to_string(),
        ));
    }

    // Convert to CSR for efficient matrix-vector operations
    let csr_matrix = matrix.to_csr()?;

    // Initialize solution vector
    let x = match x0 {
        Some(x0_val) => x0_val.clone(),
        None => zeros::<f32>(&[n])?,
    };

    // Compute initial residual: r = b - Ax
    let ax = csr_matrix.matvec(&x)?;
    let r = b.clone();
    for i in 0..n {
        r.set(&[i], b.get(&[i])? - ax.get(&[i])?)?;
    }

    let p = r.clone();
    let mut rsold = dot_product(&r, &r)?;

    for iter in 0..max_iter {
        // Check convergence
        if rsold.sqrt() < tol {
            return Ok((x, iter, rsold.sqrt()));
        }

        // Compute Ap
        let ap = csr_matrix.matvec(&p)?;

        // Compute alpha = r^T r / p^T A p
        let pap = dot_product(&p, &ap)?;
        if pap <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Matrix is not positive definite".to_string(),
            ));
        }
        let alpha = rsold / pap;

        // Update solution: x = x + alpha * p
        for i in 0..n {
            x.set(&[i], x.get(&[i])? + alpha * p.get(&[i])?)?;
        }

        // Update residual: r = r - alpha * Ap
        for i in 0..n {
            r.set(&[i], r.get(&[i])? - alpha * ap.get(&[i])?)?;
        }

        let rsnew = dot_product(&r, &r)?;

        // Check convergence
        if rsnew.sqrt() < tol {
            return Ok((x, iter + 1, rsnew.sqrt()));
        }

        // Compute beta = r_new^T r_new / r_old^T r_old
        let beta = rsnew / rsold;

        // Update search direction: p = r + beta * p
        for i in 0..n {
            p.set(&[i], r.get(&[i])? + beta * p.get(&[i])?)?;
        }

        rsold = rsnew;
    }

    Ok((x, max_iter, rsold.sqrt()))
}

/// BiConjugate Gradient Stabilized (BiCGSTAB) method for non-symmetric matrices
pub fn bicgstab(
    matrix: &dyn SparseTensor,
    b: &Tensor,
    x0: Option<&Tensor>,
    tol: f32,
    max_iter: usize,
) -> TorshResult<(Tensor, usize, f32)> {
    if matrix.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument("Matrix must be 2D".to_string()));
    }

    let n = matrix.shape().dims()[0];
    if matrix.shape().dims()[0] != matrix.shape().dims()[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix must be square".to_string(),
        ));
    }

    if b.shape().ndim() != 1 || b.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(
            "Vector b must match matrix dimensions".to_string(),
        ));
    }

    let csr_matrix = matrix.to_csr()?;

    // Initialize solution vector
    let x = match x0 {
        Some(x0_val) => x0_val.clone(),
        None => zeros::<f32>(&[n])?,
    };

    // Compute initial residual: r = b - Ax
    let ax = csr_matrix.matvec(&x)?;
    let r = b.clone();
    for i in 0..n {
        r.set(&[i], b.get(&[i])? - ax.get(&[i])?)?;
    }

    let r_hat = r.clone();
    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut v = zeros::<f32>(&[n])?;
    let p = zeros::<f32>(&[n])?;

    for iter in 0..max_iter {
        let rho_new = dot_product(&r_hat, &r)?;

        if rho_new.abs() < f32::EPSILON {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: rho = 0".to_string(),
            ));
        }

        let beta = (rho_new / rho) * (alpha / omega);

        // Update p: p = r + beta * (p - omega * v)
        for i in 0..n {
            p.set(
                &[i],
                r.get(&[i])? + beta * (p.get(&[i])? - omega * v.get(&[i])?),
            )?;
        }

        // v = A * p
        v = csr_matrix.matvec(&p)?;

        let v_dot_rhat = dot_product(&r_hat, &v)?;
        if v_dot_rhat.abs() < f32::EPSILON {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: <r_hat, v> = 0".to_string(),
            ));
        }

        alpha = rho_new / v_dot_rhat;

        // s = r - alpha * v
        let s = zeros::<f32>(&[n])?;
        for i in 0..n {
            s.set(&[i], r.get(&[i])? - alpha * v.get(&[i])?)?;
        }

        // Check convergence
        let s_norm = vector_norm(&s)?;
        if s_norm < tol {
            // Update x and return
            for i in 0..n {
                x.set(&[i], x.get(&[i])? + alpha * p.get(&[i])?)?;
            }
            return Ok((x, iter + 1, s_norm));
        }

        // t = A * s
        let t = csr_matrix.matvec(&s)?;

        let t_dot_t = dot_product(&t, &t)?;
        if t_dot_t.abs() < f32::EPSILON {
            omega = 0.0;
        } else {
            omega = dot_product(&t, &s)? / t_dot_t;
        }

        // Update x: x = x + alpha * p + omega * s
        for i in 0..n {
            x.set(
                &[i],
                x.get(&[i])? + alpha * p.get(&[i])? + omega * s.get(&[i])?,
            )?;
        }

        // Update r: r = s - omega * t
        for i in 0..n {
            r.set(&[i], s.get(&[i])? - omega * t.get(&[i])?)?;
        }

        // Check convergence
        let r_norm = vector_norm(&r)?;
        if r_norm < tol {
            return Ok((x, iter + 1, r_norm));
        }

        if omega.abs() < f32::EPSILON {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: omega = 0".to_string(),
            ));
        }

        rho = rho_new;
    }

    Ok((x, max_iter, vector_norm(&r)?))
}

/// Compute the largest eigenvalue and corresponding eigenvector using power iteration
pub fn power_iteration(
    matrix: &dyn SparseTensor,
    max_iter: usize,
    tol: f32,
) -> TorshResult<(f32, Tensor)> {
    if matrix.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument("Matrix must be 2D".to_string()));
    }

    let n = matrix.shape().dims()[0];
    if matrix.shape().dims()[0] != matrix.shape().dims()[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix must be square".to_string(),
        ));
    }

    let csr_matrix = matrix.to_csr()?;

    // Initialize with random vector
    let v = ones::<f32>(&[n])?;

    // Normalize
    let norm = vector_norm(&v)?;
    for i in 0..n {
        v.set(&[i], v.get(&[i])? / norm)?;
    }

    let mut eigenvalue = 0.0;

    for _iter in 0..max_iter {
        // v_new = A * v
        let v_new = csr_matrix.matvec(&v)?;

        // Compute eigenvalue approximation
        let eigenvalue_new = dot_product(&v, &v_new)?;

        // Normalize v_new
        let norm = vector_norm(&v_new)?;
        if norm < f32::EPSILON {
            return Err(TorshError::InvalidArgument(
                "Power iteration failed: norm became zero".to_string(),
            ));
        }

        for i in 0..n {
            v.set(&[i], v_new.get(&[i])? / norm)?;
        }

        // Check convergence
        if (eigenvalue_new - eigenvalue).abs() < tol {
            return Ok((eigenvalue_new, v));
        }

        eigenvalue = eigenvalue_new;
    }

    Ok((eigenvalue, v))
}

/// Helper function to compute dot product of two vectors
fn dot_product(a: &Tensor, b: &Tensor) -> TorshResult<f32> {
    if a.shape() != b.shape() {
        return Err(TorshError::InvalidArgument(
            "Vectors must have the same shape".to_string(),
        ));
    }

    let n = a.shape().dims()[0];
    let mut result = 0.0;

    for i in 0..n {
        result += a.get(&[i])? * b.get(&[i])?;
    }

    Ok(result)
}

/// Helper function to compute vector norm
fn vector_norm(v: &Tensor) -> TorshResult<f32> {
    dot_product(v, v).map(|x| x.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_conjugate_gradient() {
        // Create a well-conditioned symmetric positive definite matrix
        // [4, 1, 0]
        // [1, 4, 1]
        // [0, 1, 4]
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let vals = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let matrix = CooTensor::new(rows, cols, vals, Shape::new(vec![3, 3])).unwrap();

        // Right-hand side: b = [1, 1, 1]
        let b = zeros::<f32>(&[3]).unwrap();
        b.set(&[0], 1.0).unwrap();
        b.set(&[1], 1.0).unwrap();
        b.set(&[2], 1.0).unwrap();

        let (solution, iterations, residual) = conjugate_gradient(
            &matrix as &dyn SparseTensor,
            &b,
            None,
            1e-4, // Relaxed tolerance
            50,   // Reduced max iterations
        )
        .unwrap();

        // Check that the solution is reasonable
        assert!(iterations < 50);
        assert!(residual < 1e-4);
        assert_eq!(solution.shape().dims(), &[3]);
    }

    #[test]
    fn test_incomplete_lu() {
        // Create a simple matrix for LU decomposition
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 1, 2, 0, 2];
        let vals = vec![4.0, 1.0, 3.0, 2.0, 1.0, 5.0];
        let matrix = CooTensor::new(rows, cols, vals, Shape::new(vec![3, 3])).unwrap();

        let ilu = IncompleteLU::new(&matrix as &dyn SparseTensor, 0.01).unwrap();

        // Test solving a system
        let b = ones::<f32>(&[3]).unwrap();
        let solution = ilu.solve(&b).unwrap();

        assert_eq!(solution.shape().dims(), &[3]);
    }

    #[test]
    fn test_power_iteration() {
        // Create a simple symmetric matrix
        let rows = vec![0, 0, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 1, 2];
        let vals = vec![3.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let matrix = CooTensor::new(rows, cols, vals, Shape::new(vec![3, 3])).unwrap();

        let (eigenvalue, eigenvector) =
            power_iteration(&matrix as &dyn SparseTensor, 100, 1e-6).unwrap();

        // Check that we got reasonable results
        assert!(eigenvalue > 0.0);
        assert_eq!(eigenvector.shape().dims(), &[3]);

        // Check that eigenvector is normalized
        let norm = vector_norm(&eigenvector).unwrap();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }
}
