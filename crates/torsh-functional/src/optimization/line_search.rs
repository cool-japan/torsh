//! Line search methods for optimization algorithms
//!
//! This module provides various line search methods including backtracking line search
//! and Wolfe line search, which are commonly used in optimization algorithms to find
//! appropriate step sizes.

use super::utilities::{dot_product, tensor_add, tensor_scalar_mul};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Line search methods for optimization
#[derive(Debug, Clone, Copy)]
pub enum LineSearchMethod {
    /// Backtracking line search with Armijo condition
    Backtracking,
    /// Exact line search (minimize along search direction)
    Exact,
    /// Wolfe conditions line search
    Wolfe,
    /// Strong Wolfe conditions
    StrongWolfe,
}

/// Backtracking line search parameters
#[derive(Debug, Clone)]
pub struct BacktrackingParams {
    /// Initial step size
    pub alpha0: f32,
    /// Armijo parameter (typically 1e-4)
    pub c1: f32,
    /// Backtracking factor (typically 0.5)
    pub rho: f32,
    /// Maximum number of backtracking steps
    pub max_iter: usize,
}

impl Default for BacktrackingParams {
    fn default() -> Self {
        Self {
            alpha0: 1.0,
            c1: 1e-4,
            rho: 0.5,
            max_iter: 50,
        }
    }
}

/// Perform backtracking line search
///
/// Finds step size that satisfies Armijo condition:
/// f(x + α*p) ≤ f(x) + c1*α*∇f(x)ᵀp
///
/// # Arguments
/// * `objective` - Objective function f(x)
/// * `gradient` - Gradient function ∇f(x)
/// * `x` - Current point
/// * `p` - Search direction
/// * `params` - Line search parameters
pub fn backtracking_line_search<F, G>(
    objective: F,
    gradient: G,
    x: &Tensor,
    p: &Tensor,
    params: Option<BacktrackingParams>,
) -> TorshResult<f32>
where
    F: Fn(&Tensor) -> TorshResult<f32>,
    G: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let params = params.unwrap_or_default();

    let f0 = objective(x)?;
    let grad0 = gradient(x)?;

    // Compute directional derivative: ∇f(x)ᵀp
    let directional_deriv = dot_product(&grad0, p)?;

    if directional_deriv >= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Search direction is not a descent direction".to_string(),
        ));
    }

    let mut alpha = params.alpha0;

    for _ in 0..params.max_iter {
        // Compute x + α*p
        let x_new = tensor_add(x, &tensor_scalar_mul(p, alpha)?)?;
        let f_new = objective(&x_new)?;

        // Check Armijo condition
        if f_new <= f0 + params.c1 * alpha * directional_deriv {
            return Ok(alpha);
        }

        alpha *= params.rho;
    }

    // Return the last step size if no convergence
    Ok(alpha)
}

/// Wolfe line search parameters
#[derive(Debug, Clone)]
pub struct WolfeParams {
    /// Armijo parameter (typically 1e-4)
    pub c1: f32,
    /// Curvature parameter (typically 0.9)
    pub c2: f32,
    /// Initial step size
    pub alpha0: f32,
    /// Maximum step size
    pub alpha_max: f32,
    /// Maximum number of iterations
    pub max_iter: usize,
}

impl Default for WolfeParams {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            alpha0: 1.0,
            alpha_max: 100.0,
            max_iter: 20,
        }
    }
}

/// Perform Wolfe line search
///
/// Finds step size that satisfies both Armijo and curvature conditions.
///
/// # Arguments
/// * `objective` - Objective function f(x)
/// * `gradient` - Gradient function ∇f(x)
/// * `x` - Current point
/// * `p` - Search direction
/// * `params` - Wolfe line search parameters
pub fn wolfe_line_search<F, G>(
    objective: F,
    gradient: G,
    x: &Tensor,
    p: &Tensor,
    params: Option<WolfeParams>,
) -> TorshResult<f32>
where
    F: Fn(&Tensor) -> TorshResult<f32>,
    G: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let params = params.unwrap_or_default();

    let f0 = objective(x)?;
    let grad0 = gradient(x)?;
    let directional_deriv0 = dot_product(&grad0, p)?;

    if directional_deriv0 >= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Search direction is not a descent direction".to_string(),
        ));
    }

    let mut alpha_lo = 0.0;
    let mut alpha_hi = params.alpha_max;
    let mut alpha = params.alpha0;

    for _ in 0..params.max_iter {
        let x_new = tensor_add(x, &tensor_scalar_mul(p, alpha)?)?;
        let f_new = objective(&x_new)?;

        // Check Armijo condition
        if f_new > f0 + params.c1 * alpha * directional_deriv0 {
            alpha_hi = alpha;
            alpha = (alpha_lo + alpha_hi) / 2.0;
            continue;
        }

        let grad_new = gradient(&x_new)?;
        let directional_deriv_new = dot_product(&grad_new, p)?;

        // Check curvature condition
        if directional_deriv_new.abs() <= -params.c2 * directional_deriv0 {
            return Ok(alpha);
        }

        if directional_deriv_new >= 0.0 {
            alpha_hi = alpha_lo;
        }

        alpha_lo = alpha;
        alpha = if alpha_hi == params.alpha_max {
            2.0 * alpha
        } else {
            (alpha_lo + alpha_hi) / 2.0
        };
    }

    Ok(alpha)
}
