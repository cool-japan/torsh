//! Error analysis and iterative refinement for linear system solutions
//!
//! This module provides error analysis and iterative refinement capabilities for improving
//! the accuracy of linear system solutions. These methods are particularly valuable for
//! ill-conditioned systems where standard direct methods may not provide sufficient accuracy.
//!
//! ## Error Analysis
//!
//! Error analysis helps assess the quality of a computed solution by estimating various
//! error bounds:
//!
//! - **Backward error**: ||b - Ax|| - measures how much the computed solution violates
//!   the original equation
//! - **Condition number**: cond(A) - measures the sensitivity of the solution to
//!   perturbations in the input
//! - **Forward error bound**: An estimate of ||x_true - x_computed|| based on the
//!   condition number and backward error
//!
//! ## Iterative Refinement
//!
//! Iterative refinement is a technique for improving solution accuracy by:
//! 1. Computing the residual r = b - Ax for the current solution x
//! 2. Solving A * correction = r for the correction
//! 3. Updating the solution: x := x + correction
//! 4. Repeating until convergence
//!
//! This process can significantly improve accuracy, especially for ill-conditioned systems.
//!
//! ## Mixed Precision Refinement
//!
//! Mixed precision refinement performs residual computations in higher precision (f64)
//! while maintaining the solution in standard precision (f32). This provides better
//! numerical stability for challenging problems without the full computational cost
//! of double precision throughout.
//!
//! ## Examples
//!
//! ```rust
//! use torsh_linalg::solvers::refinement::{estimate_error_bounds, solve_with_refinement};
//! use torsh_tensor::{creation::{eye, ones}, tensor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a linear system
//! let a = eye::<f32>(3)?;
//! let b = ones::<f32>(&[3])?;
//!
//! // Solve with automatic refinement
//! let (solution, was_refined, iterations, residual_norm) =
//!     solve_with_refinement(&a, &b, Some(1e-12))?;
//!
//! // Analyze solution quality
//! let (backward_error, condition_number, forward_error_bound) =
//!     estimate_error_bounds(&a, &solution, &b)?;
//! # Ok(())
//! # }
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_question_mark)]

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Estimate error bounds for a linear system solution
///
/// Computes various error metrics for the solution x to the linear system Ax = b:
/// - Forward error bound: ||x_true - x_computed||
/// - Backward error: ||b - Ax_computed||
/// - Condition-based error bound: cond(A) * ||r|| / ||A||
///
/// Returns (backward_error, condition_number, estimated_forward_error)
pub fn estimate_error_bounds(a: &Tensor, x: &Tensor, b: &Tensor) -> TorshResult<(f32, f32, f32)> {
    if a.shape().ndim() != 2 || x.shape().ndim() != 1 || b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Error bounds estimation requires 2D matrix A and 1D vectors x, b".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
    let x_len = x.shape().dims()[0];
    let b_len = b.shape().dims()[0];

    if m != n || n != x_len || m != b_len {
        return Err(TorshError::InvalidArgument(
            "Dimension mismatch in error bounds estimation".to_string(),
        ));
    }

    // Compute residual: r = b - Ax
    let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
    let residual = b.sub(&ax)?;

    // Compute backward error: ||r||_2
    let mut backward_error = 0.0f32;
    for i in 0..n {
        let r_i = residual.get(&[i])?;
        backward_error += r_i * r_i;
    }
    backward_error = backward_error.sqrt();

    // Compute condition number
    let condition_number = crate::cond(a, Some("2"))?;

    // Compute matrix norm ||A||_2 (approximately using Frobenius norm)
    let mut a_norm = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let a_ij = a.get(&[i, j])?;
            a_norm += a_ij * a_ij;
        }
    }
    a_norm = a_norm.sqrt();

    // Estimate forward error bound: cond(A) * ||r|| / ||A||
    let estimated_forward_error = if a_norm > 1e-12 {
        condition_number * backward_error / a_norm
    } else {
        f32::INFINITY
    };

    Ok((backward_error, condition_number, estimated_forward_error))
}

/// Iterative refinement for improving solution accuracy
///
/// Performs iterative refinement to improve the accuracy of the solution x to Ax = b.
/// This is particularly useful for ill-conditioned systems.
///
/// Algorithm:
/// 1. Compute residual r = b - Ax
/// 2. Solve A * correction = r
/// 3. Update solution: x = x + correction
/// 4. Repeat until convergence or max iterations
///
/// Returns (refined_solution, iterations, final_residual_norm)
pub fn iterative_refinement(
    a: &Tensor,
    b: &Tensor,
    x_initial: &Tensor,
    max_iter: Option<usize>,
    tol: Option<f32>,
) -> TorshResult<(Tensor, usize, f32)> {
    if a.shape().ndim() != 2 || b.shape().ndim() != 1 || x_initial.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Iterative refinement requires 2D matrix A and 1D vectors b, x_initial".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
    let b_len = b.shape().dims()[0];
    let x_len = x_initial.shape().dims()[0];

    if m != n || n != x_len || m != b_len {
        return Err(TorshError::InvalidArgument(
            "Dimension mismatch in iterative refinement".to_string(),
        ));
    }

    let max_iter = max_iter.unwrap_or(10);
    let tol = tol.unwrap_or(1e-12);

    let mut x = x_initial.clone();
    let mut iteration = 0;

    for iter in 0..max_iter {
        iteration = iter;

        // Compute residual: r = b - Ax
        let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
        let residual = b.sub(&ax)?;

        // Compute residual norm
        let mut residual_norm = 0.0f32;
        for i in 0..n {
            let r_i = residual.get(&[i])?;
            residual_norm += r_i * r_i;
        }
        residual_norm = residual_norm.sqrt();

        // Check convergence
        if residual_norm < tol {
            return Ok((x, iteration, residual_norm));
        }

        // Solve for correction: A * correction = residual
        let correction = crate::solve(a, &residual)?;

        // Update solution: x = x + correction
        x = x.add(&correction)?;
    }

    // Compute final residual norm
    let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
    let final_residual = b.sub(&ax)?;
    let mut final_residual_norm = 0.0f32;
    for i in 0..n {
        let r_i = final_residual.get(&[i])?;
        final_residual_norm += r_i * r_i;
    }
    final_residual_norm = final_residual_norm.sqrt();

    Ok((x, iteration, final_residual_norm))
}

/// Mixed precision iterative refinement
///
/// Performs iterative refinement using higher precision arithmetic for residual computation.
/// This can significantly improve accuracy for ill-conditioned problems.
///
/// Note: This is a simplified version that doesn't actually use mixed precision,
/// but implements the algorithmic structure for future enhancement.
pub fn mixed_precision_refinement(
    a: &Tensor,
    b: &Tensor,
    x_initial: &Tensor,
    max_iter: Option<usize>,
    tol: Option<f32>,
) -> TorshResult<(Tensor, usize, f32)> {
    // Implement actual mixed precision arithmetic using f64 for residual computation
    mixed_precision_refinement_f64(a, b, x_initial, max_iter, tol)
}

/// Mixed precision iterative refinement using f64 for residual computation
///
/// This implementation uses f64 precision for residual computation and correction
/// while maintaining f32 for the main solution. This provides better numerical
/// stability for ill-conditioned problems.
#[allow(dead_code)]
fn mixed_precision_refinement_f64(
    a: &Tensor,
    b: &Tensor,
    x_initial: &Tensor,
    max_iter: Option<usize>,
    tol: Option<f32>,
) -> TorshResult<(Tensor, usize, f32)> {
    let n = a.shape().dims()[0];
    let max_iterations = max_iter.unwrap_or(10);
    let tolerance = tol.unwrap_or(1e-12);

    let mut x = x_initial.clone();

    // Convert matrices to f64 for higher precision arithmetic
    let mut a_f64 = vec![vec![0.0f64; n]; n];
    let mut b_f64 = vec![0.0f64; n];

    for i in 0..n {
        b_f64[i] = b.get(&[i])? as f64;
        for j in 0..n {
            a_f64[i][j] = a.get(&[i, j])? as f64;
        }
    }

    for iteration in 0..max_iterations {
        // Convert current solution to f64
        let mut x_f64 = vec![0.0f64; n];
        for i in 0..n {
            x_f64[i] = x.get(&[i])? as f64;
        }

        // Compute residual in f64: r = b - A*x
        let mut residual_f64 = vec![0.0f64; n];
        for i in 0..n {
            let mut ax_i = 0.0f64;
            for j in 0..n {
                ax_i += a_f64[i][j] * x_f64[j];
            }
            residual_f64[i] = b_f64[i] - ax_i;
        }

        // Compute residual norm in f64
        let mut residual_norm_f64 = 0.0f64;
        for &r in &residual_f64 {
            residual_norm_f64 += r * r;
        }
        residual_norm_f64 = residual_norm_f64.sqrt();
        let residual_norm = residual_norm_f64 as f32;

        // Check convergence
        if residual_norm < tolerance {
            return Ok((x, iteration, residual_norm));
        }

        // Convert residual back to f32 tensor for correction computation
        let residual_data: Vec<f32> = residual_f64.iter().map(|&r| r as f32).collect();
        let residual_tensor = torsh_tensor::Tensor::from_data(residual_data, vec![n], a.device())?;

        // Solve for correction in f32: A * correction = residual
        let correction = crate::solve(a, &residual_tensor)?;

        // Update solution: x = x + correction
        x = x.add(&correction)?;
    }

    // Compute final residual norm using f64 precision
    let mut x_f64 = vec![0.0f64; n];
    for i in 0..n {
        x_f64[i] = x.get(&[i])? as f64;
    }

    let mut final_residual_norm_f64 = 0.0f64;
    for i in 0..n {
        let mut ax_i = 0.0f64;
        for j in 0..n {
            ax_i += a_f64[i][j] * x_f64[j];
        }
        let residual_i = b_f64[i] - ax_i;
        final_residual_norm_f64 += residual_i * residual_i;
    }
    final_residual_norm_f64 = final_residual_norm_f64.sqrt();

    Ok((x, max_iterations, final_residual_norm_f64 as f32))
}

/// Compute solution with automatic iterative refinement
///
/// Solves Ax = b and automatically applies iterative refinement if the initial solution
/// has poor accuracy (high residual or condition number).
///
/// Returns (solution, was_refined, iterations, final_residual_norm)
pub fn solve_with_refinement(
    a: &Tensor,
    b: &Tensor,
    refinement_threshold: Option<f32>,
) -> TorshResult<(Tensor, bool, usize, f32)> {
    // Solve initial system
    let x_initial = crate::solve(a, b)?;

    // Estimate error bounds
    let (backward_error, condition_number, _estimated_forward_error) =
        estimate_error_bounds(a, &x_initial, b)?;

    // Determine if refinement is needed
    let refinement_threshold = refinement_threshold.unwrap_or(1e-10);
    let needs_refinement = backward_error > refinement_threshold || condition_number > 1e12;

    if !needs_refinement {
        return Ok((x_initial, false, 0, backward_error));
    }

    // Apply iterative refinement
    let (x_refined, iterations, final_residual_norm) =
        iterative_refinement(a, b, &x_initial, Some(5), Some(refinement_threshold))?;

    Ok((x_refined, true, iterations, final_residual_norm))
}
