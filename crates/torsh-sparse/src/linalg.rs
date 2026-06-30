//! Sparse linear algebra operations
//!
//! This module provides advanced sparse linear algebra functionality including
//! decompositions, solvers, and iterative methods.

use crate::{CooTensor, CsrTensor, SparseTensor, TorshResult};
use std::collections::HashMap;
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

/// Conjugate Gradient method for solving sparse linear systems `Ax = b`
/// where `A` is symmetric positive definite.
///
/// All inner arithmetic is carried out in `f64` over dense owned buffers, and the sparse
/// matrix-vector products are evaluated directly over the CSR arrays. The caller's `b` and
/// `A` tensors are therefore never mutated (note that `Tensor::clone` shares storage, so
/// writing through a clone would silently corrupt the original).
///
/// # Arguments
/// * `matrix` - Symmetric positive-definite sparse coefficient matrix
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (if `None`, the zero vector is used)
/// * `tol` - Convergence tolerance on the residual norm `||b - Ax||`
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
/// Tuple of `(solution, iterations performed, final residual norm)`.
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

    // CSR view for direct f64 sparse matrix-vector products. Accumulating in f64 (rather
    // than the f32 matrix dtype) keeps the residual recurrence accurate.
    let csr_matrix = matrix.to_csr()?;
    let row_ptr = csr_matrix.row_ptr();
    let col_indices = csr_matrix.col_indices();
    let values = csr_matrix.values();

    // y = A * x with f64 accumulation; never touches the caller's tensors.
    let mat_vec = |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for (row, y_row) in y.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for idx in row_ptr[row]..row_ptr[row + 1] {
                sum += f64::from(values[idx]) * x[col_indices[idx]];
            }
            *y_row = sum;
        }
        y
    };

    let tol = f64::from(tol);

    // Dense f64 right-hand side and initial iterate (owned copies, never aliasing the
    // caller's tensors).
    let mut b_vec = vec![0.0f64; n];
    for (i, b_val) in b_vec.iter_mut().enumerate() {
        *b_val = f64::from(b.get(&[i])?);
    }

    let mut x_vec = vec![0.0f64; n];
    if let Some(x_init) = x0 {
        for (i, x_val) in x_vec.iter_mut().enumerate() {
            *x_val = f64::from(x_init.get(&[i])?);
        }
    }

    // Initial residual r = b - A x and search direction p = r (an owned deep copy).
    let ax = mat_vec(&x_vec);
    let mut r = vec![0.0f64; n];
    for (r_val, (&b_val, &ax_val)) in r.iter_mut().zip(b_vec.iter().zip(ax.iter())) {
        *r_val = b_val - ax_val;
    }
    let mut p = r.clone();
    let mut rsold = dot_f64(&r, &r);

    for iter in 0..max_iter {
        // Converged before taking another step.
        if rsold.sqrt() < tol {
            return Ok((f64_slice_to_tensor(&x_vec)?, iter, rsold.sqrt() as f32));
        }

        // ap = A p and the curvature p^T A p.
        let ap = mat_vec(&p);
        let pap = dot_f64(&p, &ap);
        if pap <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "Matrix is not positive definite".to_string(),
            ));
        }
        let alpha = rsold / pap;

        // x <- x + alpha p ; r <- r - alpha A p.
        for (x_val, &p_val) in x_vec.iter_mut().zip(p.iter()) {
            *x_val += alpha * p_val;
        }
        for (r_val, &ap_val) in r.iter_mut().zip(ap.iter()) {
            *r_val -= alpha * ap_val;
        }

        let rsnew = dot_f64(&r, &r);
        if rsnew.sqrt() < tol {
            return Ok((f64_slice_to_tensor(&x_vec)?, iter + 1, rsnew.sqrt() as f32));
        }

        // p <- r + beta p with beta = rsnew / rsold.
        let beta = rsnew / rsold;
        for (p_val, &r_val) in p.iter_mut().zip(r.iter()) {
            *p_val = r_val + beta * *p_val;
        }

        rsold = rsnew;
    }

    Ok((f64_slice_to_tensor(&x_vec)?, max_iter, rsold.sqrt() as f32))
}

/// BiConjugate Gradient Stabilized (BiCGSTAB) method for non-symmetric systems.
///
/// Like [`conjugate_gradient`], all inner arithmetic is performed in `f64` over dense owned
/// buffers with the sparse matrix-vector products evaluated directly over the CSR arrays, so
/// the caller's `b` and `A` tensors are never mutated (note that `Tensor::clone` shares
/// storage, so writing through a clone would silently corrupt the original — and the shadow
/// residual `r_hat` must be a genuine copy, not an alias of `r`).
///
/// # Arguments
/// * `matrix` - Sparse coefficient matrix (can be non-symmetric)
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (if `None`, the zero vector is used)
/// * `tol` - Convergence tolerance on the residual norm `||b - Ax||`
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
/// Tuple of `(solution, iterations performed, final residual norm)`.
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

    // Below this magnitude a BiCGSTAB scalar is treated as a breakdown (division by an
    // effectively zero quantity); `f64::EPSILON` is the natural machine-zero threshold for
    // the f64 working precision.
    const BREAKDOWN_EPS: f64 = f64::EPSILON;

    // CSR view for direct f64 sparse matrix-vector products.
    let csr_matrix = matrix.to_csr()?;
    let row_ptr = csr_matrix.row_ptr();
    let col_indices = csr_matrix.col_indices();
    let values = csr_matrix.values();

    // y = A * x with f64 accumulation; never touches the caller's tensors.
    let mat_vec = |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for (row, y_row) in y.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for idx in row_ptr[row]..row_ptr[row + 1] {
                sum += f64::from(values[idx]) * x[col_indices[idx]];
            }
            *y_row = sum;
        }
        y
    };

    let tol = f64::from(tol);

    // Dense f64 right-hand side and initial iterate (owned copies, never aliasing the
    // caller's tensors).
    let mut b_vec = vec![0.0f64; n];
    for (i, b_val) in b_vec.iter_mut().enumerate() {
        *b_val = f64::from(b.get(&[i])?);
    }

    let mut x_vec = vec![0.0f64; n];
    if let Some(x_init) = x0 {
        for (i, x_val) in x_vec.iter_mut().enumerate() {
            *x_val = f64::from(x_init.get(&[i])?);
        }
    }

    // Initial residual r = b - A x; `r_hat` is the fixed shadow residual and must be a
    // genuine deep copy (aliasing it to `r` is the classic BiCGSTAB-breaks bug).
    let ax = mat_vec(&x_vec);
    let mut r = vec![0.0f64; n];
    for (r_val, (&b_val, &ax_val)) in r.iter_mut().zip(b_vec.iter().zip(ax.iter())) {
        *r_val = b_val - ax_val;
    }
    let r_hat = r.clone();

    let mut rho = 1.0f64;
    let mut alpha = 1.0f64;
    let mut omega = 1.0f64;
    let mut v = vec![0.0f64; n];
    let mut p = vec![0.0f64; n];

    for iter in 0..max_iter {
        let rho_new = dot_f64(&r_hat, &r);
        if rho_new.abs() < BREAKDOWN_EPS {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: rho = 0".to_string(),
            ));
        }

        let beta = (rho_new / rho) * (alpha / omega);

        // p <- r + beta (p - omega v).
        for ((p_val, &r_val), &v_val) in p.iter_mut().zip(r.iter()).zip(v.iter()) {
            *p_val = r_val + beta * (*p_val - omega * v_val);
        }

        // v <- A p.
        v = mat_vec(&p);

        let v_dot_rhat = dot_f64(&r_hat, &v);
        if v_dot_rhat.abs() < BREAKDOWN_EPS {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: <r_hat, v> = 0".to_string(),
            ));
        }
        alpha = rho_new / v_dot_rhat;

        // s <- r - alpha v.
        let mut s = vec![0.0f64; n];
        for ((s_val, &r_val), &v_val) in s.iter_mut().zip(r.iter()).zip(v.iter()) {
            *s_val = r_val - alpha * v_val;
        }

        // Early convergence on the half-step residual s.
        let s_norm = norm_f64(&s);
        if s_norm < tol {
            for (x_val, &p_val) in x_vec.iter_mut().zip(p.iter()) {
                *x_val += alpha * p_val;
            }
            return Ok((f64_slice_to_tensor(&x_vec)?, iter + 1, s_norm as f32));
        }

        // t <- A s, then omega = <t, s> / <t, t>.
        let t = mat_vec(&s);
        let t_dot_t = dot_f64(&t, &t);
        omega = if t_dot_t.abs() < BREAKDOWN_EPS {
            0.0
        } else {
            dot_f64(&t, &s) / t_dot_t
        };

        // x <- x + alpha p + omega s ; r <- s - omega t.
        for ((x_val, &p_val), &s_val) in x_vec.iter_mut().zip(p.iter()).zip(s.iter()) {
            *x_val += alpha * p_val + omega * s_val;
        }
        for ((r_val, &s_val), &t_val) in r.iter_mut().zip(s.iter()).zip(t.iter()) {
            *r_val = s_val - omega * t_val;
        }

        let r_norm = norm_f64(&r);
        if r_norm < tol {
            return Ok((f64_slice_to_tensor(&x_vec)?, iter + 1, r_norm as f32));
        }

        if omega.abs() < BREAKDOWN_EPS {
            return Err(TorshError::InvalidArgument(
                "BiCGSTAB breakdown: omega = 0".to_string(),
            ));
        }

        rho = rho_new;
    }

    Ok((f64_slice_to_tensor(&x_vec)?, max_iter, norm_f64(&r) as f32))
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

/// Sparse Cholesky decomposition for symmetric positive definite matrices
/// Computes A = L * L^T where L is lower triangular
pub struct SparseCholesky {
    /// Lower triangular factor (CSR format)
    l_matrix: CsrTensor,
    /// Original matrix shape
    shape: Shape,
}

impl SparseCholesky {
    /// Compute incomplete Cholesky decomposition with fill-in control
    /// fill_factor controls how many elements to keep (smaller = more sparse)
    pub fn new(matrix: &dyn SparseTensor, fill_factor: f32) -> TorshResult<Self> {
        if matrix.shape().ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Matrix must be 2D for Cholesky decomposition".to_string(),
            ));
        }

        let shape = matrix.shape().clone();
        let n = shape.dims()[0];

        if shape.dims()[0] != shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Matrix must be square for Cholesky decomposition".to_string(),
            ));
        }

        // Convert to CSR for efficient row operations
        let csr = matrix.to_csr()?;

        // Build a HashMap for efficient element access
        let mut elements: HashMap<(usize, usize), f32> = HashMap::new();
        for i in 0..n {
            let (cols, vals) = csr.get_row(i)?;
            for (idx, &col) in cols.iter().enumerate() {
                if col <= i {
                    // Only store lower triangular part
                    elements.insert((i, col), vals[idx]);
                } else {
                    // For symmetric matrices, also add upper triangular contribution
                    elements.insert((col, i), vals[idx]);
                }
            }
        }

        // Perform incomplete Cholesky decomposition
        let mut l_elements: HashMap<(usize, usize), f32> = HashMap::new();

        for i in 0..n {
            // Compute L[i,i]
            let mut sum = elements.get(&(i, i)).copied().unwrap_or(0.0);

            for k in 0..i {
                if let Some(&l_ik) = l_elements.get(&(i, k)) {
                    sum -= l_ik * l_ik;
                }
            }

            if sum <= 0.0 {
                return Err(TorshError::InvalidArgument(
                    "Matrix is not positive definite".to_string(),
                ));
            }

            let l_ii = sum.sqrt();
            l_elements.insert((i, i), l_ii);

            // Compute L[j,i] for j > i
            for j in (i + 1)..n {
                let mut sum = elements.get(&(j, i)).copied().unwrap_or(0.0);

                for k in 0..i {
                    let l_ik = l_elements.get(&(i, k)).copied().unwrap_or(0.0);
                    let l_jk = l_elements.get(&(j, k)).copied().unwrap_or(0.0);
                    sum -= l_ik * l_jk;
                }

                let l_ji = sum / l_ii;

                // Apply fill-in control
                if l_ji.abs() > fill_factor {
                    l_elements.insert((j, i), l_ji);
                }
            }
        }

        // Convert HashMap to CSR format
        let mut triplets = Vec::new();
        for ((row, col), val) in l_elements {
            triplets.push((row, col, val));
        }

        let l_matrix = CooTensor::from_triplets(triplets, (n, n))?.to_csr()?;

        Ok(Self { l_matrix, shape })
    }

    /// Solve Lx = b (forward substitution)
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

        let x = zeros::<f32>(&[n])?;

        // Forward substitution: Lx = b
        for i in 0..n {
            let mut sum = 0.0;
            let (cols, vals) = self.l_matrix.get_row(i)?;

            for (k, &col) in cols.iter().enumerate() {
                if col < i {
                    sum += vals[k] * x.get(&[col])?;
                } else if col == i {
                    // Diagonal element
                    x.set(&[i], (b.get(&[i])? - sum) / vals[k])?;
                    break;
                }
            }
        }

        Ok(x)
    }

    /// Solve L^T x = y (backward substitution)
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

        // Backward substitution: L^T x = y
        for i in (0..n).rev() {
            let mut sum = 0.0;
            let mut diag_val = 1.0;
            let (cols, vals) = self.l_matrix.get_row(i)?;

            // First find diagonal value
            for (k, &col) in cols.iter().enumerate() {
                if col == i {
                    diag_val = vals[k];
                    break;
                }
            }

            // Compute sum of L^T[i,j] * x[j] for j > i
            // L^T[i,j] = L[j,i], so we need to search other rows
            for j in (i + 1)..n {
                let (j_cols, j_vals) = self.l_matrix.get_row(j)?;
                for (k, &col) in j_cols.iter().enumerate() {
                    if col == i {
                        sum += j_vals[k] * x.get(&[j])?;
                        break;
                    }
                }
            }

            x.set(&[i], (y.get(&[i])? - sum) / diag_val)?;
        }

        Ok(x)
    }

    /// Solve the system A x = b where A = L * L^T
    pub fn solve(&self, b: &Tensor) -> TorshResult<Tensor> {
        let y = self.solve_lower(b)?;
        self.solve_upper(&y)
    }

    /// Get the lower triangular factor
    pub fn get_l(&self) -> &CsrTensor {
        &self.l_matrix
    }
}

/// GMRES(m) — Generalized Minimal Residual solver for sparse linear systems.
///
/// Solves the (possibly non-symmetric) system `Ax = b` by repeatedly building an
/// orthonormal Krylov basis with the Arnoldi process and minimizing the residual norm
/// over that subspace. Each restart cycle:
///
/// 1. forms `v_0 = r / ||r||` from the current true residual `r = b - Ax`,
/// 2. extends the basis with modified Gram-Schmidt plus a Daniel-Gragg-Kaufman-Stewart
///    re-orthogonalization pass for robustness against cancellation,
/// 3. keeps the projected least-squares problem in upper-triangular form via incremental
///    Givens rotations (which also yields a running residual estimate), and
/// 4. updates `x` with the minimizer over the subspace.
///
/// All inner arithmetic is performed in `f64`; only the matrix entries and the returned
/// solution use `f32`. The right-hand side and solution are kept as dense `f64` vectors so
/// the routine never mutates the caller's tensors (note that `Tensor::clone` shares
/// storage, so writing through a clone would corrupt the original).
///
/// # Arguments
/// * `a` - Sparse coefficient matrix (can be non-symmetric)
/// * `b` - Right-hand side vector
/// * `x0` - Initial guess (if `None`, the zero vector is used)
/// * `restart` - Krylov subspace dimension before restarting (the `m` in GMRES(m))
/// * `max_iter` - Maximum number of restart cycles
/// * `tol` - Convergence tolerance on the relative residual `||b - Ax|| / ||b||`
///
/// # Returns
/// Tuple of `(solution, total Arnoldi steps performed, final relative residual)`.
pub fn gmres(
    a: &dyn SparseTensor,
    b: &Tensor,
    x0: Option<&Tensor>,
    restart: usize,
    max_iter: usize,
    tol: f64,
) -> TorshResult<(Tensor, usize, f64)> {
    if a.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument("Matrix must be 2D".to_string()));
    }

    let n = a.shape().dims()[0];
    if a.shape().dims()[0] != a.shape().dims()[1] {
        return Err(TorshError::InvalidArgument(
            "Matrix must be square".to_string(),
        ));
    }

    if b.shape().ndim() != 1 || b.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(
            "Vector size doesn't match matrix dimensions".to_string(),
        ));
    }

    if restart == 0 {
        return Err(TorshError::InvalidArgument(
            "GMRES restart dimension must be at least 1".to_string(),
        ));
    }

    // Re-orthogonalization (DGKS) threshold: 1/sqrt(2). A second Gram-Schmidt pass is
    // triggered when the first pass leaves less than this fraction of the input norm.
    const REORTH_FACTOR: f64 = std::f64::consts::FRAC_1_SQRT_2;
    // Happy-breakdown threshold relative to ||A v_j||: a sub-diagonal entry this small
    // means the Krylov subspace is (numerically) A-invariant and the solve is exact.
    const HAPPY_BREAKDOWN_FACTOR: f64 = 1e-13;

    // CSR view for fast f64 sparse matrix-vector products (the accumulation happens in
    // f64 rather than f32, and there is no per-iteration tensor allocation).
    let csr = a.to_csr()?;
    let row_ptr = csr.row_ptr();
    let col_indices = csr.col_indices();
    let values = csr.values();

    // y = A * x with f64 accumulation.
    let mat_vec = |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for (row, y_row) in y.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for idx in row_ptr[row]..row_ptr[row + 1] {
                sum += f64::from(values[idx]) * x[col_indices[idx]];
            }
            *y_row = sum;
        }
        y
    };

    // Dense f64 copies of b and the initial guess. These deliberately avoid aliasing the
    // caller's tensors.
    let mut b_vec = vec![0.0f64; n];
    for (i, b_val) in b_vec.iter_mut().enumerate() {
        *b_val = f64::from(b.get(&[i])?);
    }

    let mut x_vec = vec![0.0f64; n];
    if let Some(x_init) = x0 {
        for (i, x_val) in x_vec.iter_mut().enumerate() {
            *x_val = f64::from(x_init.get(&[i])?);
        }
    }

    let b_norm = norm_f64(&b_vec);

    // Residual workspace reused across cycles.
    let mut r = vec![0.0f64; n];

    // Degenerate right-hand side: the relative-residual formulation is undefined, so
    // return the current iterate together with its absolute residual norm.
    if b_norm <= f64::EPSILON {
        let ax = mat_vec(&x_vec);
        for (r_val, (&b_val, &ax_val)) in r.iter_mut().zip(b_vec.iter().zip(ax.iter())) {
            *r_val = b_val - ax_val;
        }
        return Ok((f64_slice_to_tensor(&x_vec)?, 0, norm_f64(&r)));
    }

    // A Krylov subspace of dimension n already spans R^n, so never build more than n
    // Arnoldi vectors (doing so would divide by a round-off-level sub-diagonal entry).
    let m = restart.min(n);

    let mut total_iterations = 0usize;

    for _restart_cycle in 0..max_iter {
        // True residual at the start of the cycle: r = b - A x.
        let ax = mat_vec(&x_vec);
        for (r_val, (&b_val, &ax_val)) in r.iter_mut().zip(b_vec.iter().zip(ax.iter())) {
            *r_val = b_val - ax_val;
        }
        let beta = norm_f64(&r);
        if beta / b_norm < tol {
            return Ok((
                f64_slice_to_tensor(&x_vec)?,
                total_iterations,
                beta / b_norm,
            ));
        }

        // Orthonormal Krylov basis, v[0] = r / ||r||.
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        let inv_beta = 1.0 / beta;
        v.push(r.iter().map(|&r_val| r_val * inv_beta).collect());

        // (m+1) x m upper-Hessenberg matrix, Givens coefficients, and the rotated
        // right-hand side g (||r|| e1 progressively rotated).
        let mut h = vec![vec![0.0f64; m]; m + 1];
        let mut cs = vec![0.0f64; m];
        let mut sn = vec![0.0f64; m];
        let mut g = vec![0.0f64; m + 1];
        g[0] = beta;

        // Number of fully processed Arnoldi columns this cycle.
        let mut k = 0usize;

        for j in 0..m {
            // w = A v[j].
            let mut w = mat_vec(&v[j]);
            let input_norm = norm_f64(&w);

            // Modified Gram-Schmidt against the existing basis.
            for i in 0..=j {
                let h_ij = dot_f64(&w, &v[i]);
                h[i][j] = h_ij;
                for (w_val, &v_val) in w.iter_mut().zip(v[i].iter()) {
                    *w_val -= h_ij * v_val;
                }
            }
            let mut h_next = norm_f64(&w);

            // DGKS re-orthogonalization when the first pass lost too much to cancellation.
            if h_next < REORTH_FACTOR * input_norm {
                for i in 0..=j {
                    let correction = dot_f64(&w, &v[i]);
                    h[i][j] += correction;
                    for (w_val, &v_val) in w.iter_mut().zip(v[i].iter()) {
                        *w_val -= correction * v_val;
                    }
                }
                h_next = norm_f64(&w);
            }
            h[j + 1][j] = h_next;

            // Apply the previously computed Givens rotations to the new column.
            for i in 0..j {
                let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = cs[i] * h[i + 1][j] - sn[i] * h[i][j];
                h[i][j] = temp;
            }

            // New Givens rotation eliminating the sub-diagonal h[j+1][j].
            let denom = h[j][j].hypot(h[j + 1][j]);
            if denom <= f64::MIN_POSITIVE {
                break;
            }
            cs[j] = h[j][j] / denom;
            sn[j] = h[j + 1][j] / denom;
            h[j][j] = denom;
            h[j + 1][j] = 0.0;

            // Rotate the right-hand side; |g[j+1]| is the new residual estimate.
            let g_j = g[j];
            g[j] = cs[j] * g_j;
            g[j + 1] = -sn[j] * g_j;

            k = j + 1;
            total_iterations += 1;

            if g[j + 1].abs() / b_norm < tol {
                break;
            }

            // Lucky breakdown: the subspace is A-invariant, the projection is exact.
            if h_next <= HAPPY_BREAKDOWN_FACTOR * input_norm {
                break;
            }

            // Extend the basis: v[j+1] = w / ||w|| (skip after the final column).
            if j + 1 < m {
                let inv = 1.0 / h_next;
                v.push(w.iter().map(|&w_val| w_val * inv).collect());
            }
        }

        // Solve the k x k upper-triangular system R y = g by back-substitution.
        let mut y = vec![0.0f64; k];
        for i in (0..k).rev() {
            let mut sum = g[i];
            for l in (i + 1)..k {
                sum -= h[i][l] * y[l];
            }
            if h[i][i].abs() <= f64::MIN_POSITIVE {
                return Err(TorshError::ComputeError(
                    "Singular Hessenberg factor in GMRES least-squares solve".to_string(),
                ));
            }
            y[i] = sum / h[i][i];
        }

        // x = x + V_k y.
        for i in 0..k {
            let y_i = y[i];
            for (x_val, &v_val) in x_vec.iter_mut().zip(v[i].iter()) {
                *x_val += y_i * v_val;
            }
        }

        // No progress was possible this cycle (defensive): stop restarting.
        if k == 0 {
            break;
        }
    }

    // Final true residual after exhausting the restart budget.
    let ax = mat_vec(&x_vec);
    for (r_val, (&b_val, &ax_val)) in r.iter_mut().zip(b_vec.iter().zip(ax.iter())) {
        *r_val = b_val - ax_val;
    }
    let final_rel_residual = norm_f64(&r) / b_norm;

    Ok((
        f64_slice_to_tensor(&x_vec)?,
        total_iterations,
        final_rel_residual,
    ))
}

/// Dense dot product of two `f64` vectors of equal length.
fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Euclidean (L2) norm of an `f64` vector.
fn norm_f64(a: &[f64]) -> f64 {
    dot_f64(a, a).sqrt()
}

/// Materialize an `f64` working vector into a 1-D `f32` tensor (the public result type).
fn f64_slice_to_tensor(data: &[f64]) -> TorshResult<Tensor> {
    let tensor = zeros::<f32>(&[data.len()])?;
    for (i, &value) in data.iter().enumerate() {
        tensor.set(&[i], value as f32)?;
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_conjugate_gradient() {
        // Symmetric positive-definite tridiagonal system:
        //   [4  1  0]      [1]
        //   [1  4  1] x =  [1]
        //   [0  1  4]      [1]
        // Exact solution: x = [3/14, 1/7, 3/14].
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let vals = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let matrix = CooTensor::new(rows, cols, vals, Shape::new(vec![3, 3])).unwrap();

        // Right-hand side: b = [1, 1, 1].
        let b = zeros::<f32>(&[3]).unwrap();
        b.set(&[0], 1.0).unwrap();
        b.set(&[1], 1.0).unwrap();
        b.set(&[2], 1.0).unwrap();

        let (solution, iterations, residual) =
            conjugate_gradient(&matrix as &dyn SparseTensor, &b, None, 1e-8, 50).unwrap();

        // Regression guard (the bug): the right-hand side must be left byte-for-byte
        // unchanged. `Tensor::clone` shares storage, so a careless residual update would
        // silently corrupt the caller's `b`. Checked first so it acts as the canary.
        for (i, &exp_b) in [1.0_f32, 1.0, 1.0].iter().enumerate() {
            assert_eq!(
                b.get(&[i]).unwrap(),
                exp_b,
                "conjugate_gradient mutated the caller's b at index {i}"
            );
        }

        // Converged to a tight residual within the iteration budget.
        assert!(iterations <= 50);
        assert!(
            residual < 1e-6,
            "CG failed to converge: residual {residual}"
        );
        assert_eq!(solution.shape().dims(), &[3]);

        // The computed solution must match the exact solution.
        let expected: [f32; 3] = [3.0 / 14.0, 1.0 / 7.0, 3.0 / 14.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert_relative_eq!(solution.get(&[i]).unwrap(), exp, epsilon = 1e-4);
        }

        // A*x must reproduce b.
        let solution_2d = zeros::<f32>(&[3, 1]).unwrap();
        for i in 0..3 {
            solution_2d
                .set(&[i, 0], solution.get(&[i]).unwrap())
                .unwrap();
        }
        let ax = crate::ops::spmm(&matrix as &dyn SparseTensor, &solution_2d).unwrap();
        for i in 0..3 {
            let component_error = (ax.get(&[i, 0]).unwrap() - b.get(&[i]).unwrap()).abs();
            assert!(
                component_error < 1e-4,
                "A*x - b component {i} = {component_error}"
            );
        }
    }

    #[test]
    fn test_bicgstab() {
        // Well-conditioned, non-symmetric 3x3 system (strictly diagonally dominant):
        //   [4  1  2]      [1]
        //   [1  3  1] x =  [2]
        //   [2  0  5]      [3]
        // Exact solution: x = [-2/9, 23/45, 31/45].
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 2];
        let vals = vec![4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 5.0];
        let matrix = CooTensor::new(rows, cols, vals, Shape::new(vec![3, 3])).unwrap();

        // Right-hand side: b = [1, 2, 3].
        let b = zeros::<f32>(&[3]).unwrap();
        b.set(&[0], 1.0).unwrap();
        b.set(&[1], 2.0).unwrap();
        b.set(&[2], 3.0).unwrap();

        let (solution, iterations, residual) =
            bicgstab(&matrix as &dyn SparseTensor, &b, None, 1e-8, 100).unwrap();

        // Regression guard (the bug): the right-hand side must be left byte-for-byte
        // unchanged. `Tensor::clone` shares storage, so a careless residual update would
        // silently corrupt the caller's `b`. Checked first so it acts as the canary.
        for (i, &exp_b) in [1.0_f32, 2.0, 3.0].iter().enumerate() {
            assert_eq!(
                b.get(&[i]).unwrap(),
                exp_b,
                "bicgstab mutated the caller's b at index {i}"
            );
        }

        // Converged to a tight residual within the iteration budget.
        assert!(iterations <= 100);
        assert!(
            residual < 1e-6,
            "BiCGSTAB failed to converge: residual {residual}"
        );
        assert_eq!(solution.shape().dims(), &[3]);

        // The computed solution must match the exact solution.
        let expected: [f32; 3] = [-2.0 / 9.0, 23.0 / 45.0, 31.0 / 45.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert_relative_eq!(solution.get(&[i]).unwrap(), exp, epsilon = 1e-4);
        }

        // A*x must reproduce b.
        let solution_2d = zeros::<f32>(&[3, 1]).unwrap();
        for i in 0..3 {
            solution_2d
                .set(&[i, 0], solution.get(&[i]).unwrap())
                .unwrap();
        }
        let ax = crate::ops::spmm(&matrix as &dyn SparseTensor, &solution_2d).unwrap();
        for i in 0..3 {
            let component_error = (ax.get(&[i, 0]).unwrap() - b.get(&[i]).unwrap()).abs();
            assert!(
                component_error < 1e-4,
                "A*x - b component {i} = {component_error}"
            );
        }
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

    #[test]
    fn test_gmres() {
        // Non-symmetric, well-conditioned 3x3 system:
        //   [4  1  2]      [1]
        //   [1  3  1] x =  [2]
        //   [2  0  5]      [3]
        // Exact solution: x = [-2/9, 23/45, 31/45].
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 2];
        let vals = vec![4.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 5.0];
        let matrix = CooTensor::new(rows, cols, vals, Shape::new(vec![3, 3])).unwrap();

        // Right-hand side: b = [1, 2, 3].
        let b = zeros::<f32>(&[3]).unwrap();
        b.set(&[0], 1.0).unwrap();
        b.set(&[1], 2.0).unwrap();
        b.set(&[2], 3.0).unwrap();

        let (solution, iterations, residual) =
            gmres(&matrix as &dyn SparseTensor, &b, None, 5, 20, 1e-10).unwrap();

        // GMRES must reach a tight residual within at most n steps per cycle.
        assert_eq!(solution.shape().dims(), &[3]);
        assert!(iterations <= 100);
        assert!(
            residual < 1e-8,
            "GMRES failed to converge: relative residual {residual}"
        );

        // The computed solution must match the exact solution.
        let expected: [f32; 3] = [-2.0 / 9.0, 23.0 / 45.0, 31.0 / 45.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert_relative_eq!(solution.get(&[i]).unwrap(), exp, epsilon = 1e-4);
        }

        // Regression guard: the right-hand side must be left untouched. `Tensor::clone`
        // shares storage, so a careless residual update would corrupt the caller's `b`.
        for (i, &exp_b) in [1.0_f32, 2.0, 3.0].iter().enumerate() {
            assert_relative_eq!(b.get(&[i]).unwrap(), exp_b, epsilon = 1e-6);
        }

        // A*x must reproduce b.
        let solution_2d = zeros::<f32>(&[3, 1]).unwrap();
        for i in 0..3 {
            solution_2d
                .set(&[i, 0], solution.get(&[i]).unwrap())
                .unwrap();
        }
        let ax = crate::ops::spmm(&matrix as &dyn SparseTensor, &solution_2d).unwrap();
        for i in 0..3 {
            let component_error = (ax.get(&[i, 0]).unwrap() - b.get(&[i]).unwrap()).abs();
            assert!(
                component_error < 1e-4,
                "A*x - b component {i} = {component_error}"
            );
        }
    }

    #[test]
    fn test_gmres_spd() {
        // Symmetric positive definite tridiagonal system:
        //   [4  1  0]      [1]
        //   [1  4  1] x =  [1]
        //   [0  1  4]      [1]
        // Exact solution: x = [3/14, 1/7, 3/14].
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let vals = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let matrix = CooTensor::new(rows, cols, vals, Shape::new(vec![3, 3])).unwrap();

        let b = ones::<f32>(&[3]).unwrap();

        let (solution, _iterations, residual) =
            gmres(&matrix as &dyn SparseTensor, &b, None, 3, 10, 1e-12).unwrap();

        assert!(
            residual < 1e-10,
            "GMRES (SPD) failed to converge: relative residual {residual}"
        );

        let expected: [f32; 3] = [3.0 / 14.0, 1.0 / 7.0, 3.0 / 14.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert_relative_eq!(solution.get(&[i]).unwrap(), exp, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_gmres_restart_nonsymmetric() {
        // 8x8 strictly diagonally dominant, non-symmetric matrix:
        //   diagonal = 8, sub-diagonal = -1, super-diagonal = -2.
        let n = 8;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(8.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-1.0);
            }
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                vals.push(-2.0);
            }
        }
        let matrix = CooTensor::new(rows, cols, vals, Shape::new(vec![n, n])).unwrap();

        let b = ones::<f32>(&[n]).unwrap();

        // restart = 4 < n exercises the GMRES(m) restart loop.
        let (solution, _iterations, residual) =
            gmres(&matrix as &dyn SparseTensor, &b, None, 4, 100, 1e-10).unwrap();

        assert_eq!(solution.shape().dims(), &[n]);
        assert!(
            residual < 1e-8,
            "restarted GMRES failed to converge: relative residual {residual}"
        );

        // Verify A*x = b directly.
        let solution_2d = zeros::<f32>(&[n, 1]).unwrap();
        for i in 0..n {
            solution_2d
                .set(&[i, 0], solution.get(&[i]).unwrap())
                .unwrap();
        }
        let ax = crate::ops::spmm(&matrix as &dyn SparseTensor, &solution_2d).unwrap();
        for i in 0..n {
            let component_error = (ax.get(&[i, 0]).unwrap() - 1.0).abs();
            assert!(
                component_error < 1e-4,
                "A*x - b component {i} = {component_error}"
            );
        }
    }
}
