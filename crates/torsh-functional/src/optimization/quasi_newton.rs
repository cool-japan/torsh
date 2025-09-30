//! Quasi-Newton optimization methods
//!
//! This module provides quasi-Newton methods including Limited-memory BFGS (L-BFGS)
//! for large-scale optimization problems.

use super::line_search::{backtracking_line_search, wolfe_line_search, LineSearchMethod};
use super::utilities::{dot_product, tensor_add, tensor_norm, tensor_scalar_mul, tensor_sub};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// BFGS optimizer parameters
#[derive(Debug, Clone)]
pub struct BFGSParams {
    /// Initial inverse Hessian approximation scale
    pub initial_hessian_scale: f32,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Line search parameters
    pub line_search: LineSearchMethod,
}

impl Default for BFGSParams {
    fn default() -> Self {
        Self {
            initial_hessian_scale: 1.0,
            max_iter: 1000,
            tolerance: 1e-6,
            line_search: LineSearchMethod::Wolfe,
        }
    }
}

/// Limited-memory BFGS (L-BFGS) optimizer
///
/// Memory-efficient quasi-Newton method for large-scale optimization.
///
/// # Arguments
/// * `objective` - Objective function to minimize
/// * `gradient` - Gradient function
/// * `x0` - Initial point
/// * `m` - Number of correction pairs to store (typically 5-20)
/// * `params` - Optimization parameters
pub fn lbfgs_optimizer<F, G>(
    objective: F,
    gradient: G,
    x0: &Tensor,
    m: Option<usize>,
    params: Option<BFGSParams>,
) -> TorshResult<(Tensor, Vec<f32>)>
where
    F: Fn(&Tensor) -> TorshResult<f32>,
    G: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let params = params.unwrap_or_default();
    let m = m.unwrap_or(10);

    let mut x = x0.clone();
    let mut objective_values = Vec::new();

    // Storage for L-BFGS correction pairs
    let mut s_history: Vec<Tensor> = Vec::with_capacity(m);
    let mut y_history: Vec<Tensor> = Vec::with_capacity(m);
    let mut rho_history: Vec<f32> = Vec::with_capacity(m);

    let mut _grad_prev = gradient(&x)?;

    for iter in 0..params.max_iter {
        let f_val = objective(&x)?;
        objective_values.push(f_val);

        let grad = gradient(&x)?;
        let grad_norm = tensor_norm(&grad)?;

        if grad_norm < params.tolerance {
            break;
        }

        // Two-loop recursion to compute search direction
        let p = if iter == 0 {
            // Initial iteration: use steepest descent
            tensor_scalar_mul(&grad, -1.0)?
        } else {
            lbfgs_two_loop_recursion(&grad, &s_history, &y_history, &rho_history)?
        };

        // Line search
        let alpha = match params.line_search {
            LineSearchMethod::Wolfe => wolfe_line_search(&objective, &gradient, &x, &p, None)?,
            LineSearchMethod::Backtracking => {
                backtracking_line_search(&objective, &gradient, &x, &p, None)?
            }
            _ => {
                return Err(TorshError::InvalidArgument(
                    "Unsupported line search method for L-BFGS".to_string(),
                ))
            }
        };

        // Update
        let x_new = tensor_add(&x, &tensor_scalar_mul(&p, alpha)?)?;
        let grad_new = gradient(&x_new)?;

        // Store correction pairs
        let s = tensor_scalar_mul(&p, alpha)?;
        let y = tensor_sub(&grad_new, &grad)?;
        let rho = 1.0 / dot_product(&y, &s)?;

        if s_history.len() == m {
            s_history.remove(0);
            y_history.remove(0);
            rho_history.remove(0);
        }

        s_history.push(s);
        y_history.push(y);
        rho_history.push(rho);

        x = x_new;
        _grad_prev = grad;

        if iter % 100 == 0 {
            println!(
                "Iteration {}: f = {:.6e}, |∇f| = {:.6e}, α = {:.6e}",
                iter, f_val, grad_norm, alpha
            );
        }
    }

    Ok((x, objective_values))
}

/// L-BFGS two-loop recursion for computing search direction
fn lbfgs_two_loop_recursion(
    grad: &Tensor,
    s_history: &[Tensor],
    y_history: &[Tensor],
    rho_history: &[f32],
) -> TorshResult<Tensor> {
    let m = s_history.len();
    let mut q = tensor_scalar_mul(grad, -1.0)?;
    let mut alpha = vec![0.0; m];

    // First loop (backward)
    for i in (0..m).rev() {
        alpha[i] = rho_history[i] * dot_product(&s_history[i], &q)?;
        q = tensor_sub(&q, &tensor_scalar_mul(&y_history[i], alpha[i])?)?;
    }

    // Apply initial Hessian approximation (identity scaled)
    let mut r = q;
    if !s_history.is_empty() {
        let last_idx = m - 1;
        let sy = dot_product(&s_history[last_idx], &y_history[last_idx])?;
        let yy = dot_product(&y_history[last_idx], &y_history[last_idx])?;
        let gamma = sy / yy;
        r = tensor_scalar_mul(&r, gamma)?;
    }

    // Second loop (forward)
    for i in 0..m {
        let beta = rho_history[i] * dot_product(&y_history[i], &r)?;
        r = tensor_add(&r, &tensor_scalar_mul(&s_history[i], alpha[i] - beta)?)?;
    }

    Ok(r)
}
