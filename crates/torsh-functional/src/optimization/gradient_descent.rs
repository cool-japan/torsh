//! Gradient descent variants for optimization
//!
//! This module provides various gradient descent optimization algorithms including
//! basic gradient descent, momentum gradient descent, and Adam optimizer.

use super::line_search::{backtracking_line_search, wolfe_line_search, LineSearchMethod};
use super::utilities::{
    tensor_add, tensor_elementwise_div, tensor_elementwise_mul, tensor_full_like, tensor_norm,
    tensor_scalar_mul, tensor_sqrt, tensor_sub, tensor_zeros_like,
};
use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Gradient descent optimizer parameters
#[derive(Debug, Clone)]
pub struct GradientDescentParams {
    /// Learning rate
    pub learning_rate: f32,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Line search method
    pub line_search: Option<LineSearchMethod>,
}

impl Default for GradientDescentParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iter: 1000,
            tolerance: 1e-6,
            line_search: Some(LineSearchMethod::Backtracking),
        }
    }
}

/// Basic gradient descent optimization
///
/// # Arguments
/// * `objective` - Objective function to minimize
/// * `gradient` - Gradient function
/// * `x0` - Initial point
/// * `params` - Optimization parameters
pub fn gradient_descent<F, G>(
    objective: F,
    gradient: G,
    x0: &Tensor,
    params: Option<GradientDescentParams>,
) -> TorshResult<(Tensor, Vec<f32>)>
where
    F: Fn(&Tensor) -> TorshResult<f32>,
    G: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let params = params.unwrap_or_default();
    let mut x = x0.clone();
    let mut objective_values = Vec::new();

    for iter in 0..params.max_iter {
        let f_val = objective(&x)?;
        objective_values.push(f_val);

        let grad = gradient(&x)?;
        let grad_norm = tensor_norm(&grad)?;

        if grad_norm < params.tolerance {
            break;
        }

        // Search direction is negative gradient
        let p = tensor_scalar_mul(&grad, -1.0)?;

        // Determine step size
        let alpha = match params.line_search {
            Some(LineSearchMethod::Backtracking) => {
                backtracking_line_search(&objective, &gradient, &x, &p, None)?
            }
            Some(LineSearchMethod::Wolfe) => {
                wolfe_line_search(&objective, &gradient, &x, &p, None)?
            }
            _ => params.learning_rate,
        };

        // Update: x = x + α*p
        x = tensor_add(&x, &tensor_scalar_mul(&p, alpha)?)?;

        if iter % 100 == 0 {
            println!(
                "Iteration {}: f = {:.6e}, |∇f| = {:.6e}, α = {:.6e}",
                iter, f_val, grad_norm, alpha
            );
        }
    }

    Ok((x, objective_values))
}

/// Momentum gradient descent parameters
#[derive(Debug, Clone)]
pub struct MomentumParams {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum parameter (typically 0.9)
    pub momentum: f32,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f32,
}

impl Default for MomentumParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            max_iter: 1000,
            tolerance: 1e-6,
        }
    }
}

/// Momentum gradient descent
///
/// Updates: v = β*v + ∇f(x), x = x - α*v
///
/// # Arguments
/// * `objective` - Objective function to minimize
/// * `gradient` - Gradient function
/// * `x0` - Initial point
/// * `params` - Optimization parameters
pub fn momentum_gradient_descent<F, G>(
    objective: F,
    gradient: G,
    x0: &Tensor,
    params: Option<MomentumParams>,
) -> TorshResult<(Tensor, Vec<f32>)>
where
    F: Fn(&Tensor) -> TorshResult<f32>,
    G: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let params = params.unwrap_or_default();
    let mut x = x0.clone();
    let mut v = tensor_zeros_like(&x)?; // Initialize momentum to zero
    let mut objective_values = Vec::new();

    for iter in 0..params.max_iter {
        let f_val = objective(&x)?;
        objective_values.push(f_val);

        let grad = gradient(&x)?;
        let grad_norm = tensor_norm(&grad)?;

        if grad_norm < params.tolerance {
            break;
        }

        // Update momentum: v = β*v + ∇f(x)
        v = tensor_add(&tensor_scalar_mul(&v, params.momentum)?, &grad)?;

        // Update parameters: x = x - α*v
        x = tensor_sub(&x, &tensor_scalar_mul(&v, params.learning_rate)?)?;

        if iter % 100 == 0 {
            println!(
                "Iteration {}: f = {:.6e}, |∇f| = {:.6e}",
                iter, f_val, grad_norm
            );
        }
    }

    Ok((x, objective_values))
}

/// Adam optimizer parameters
#[derive(Debug, Clone)]
pub struct AdamParams {
    /// Learning rate
    pub learning_rate: f32,
    /// First moment decay rate (typically 0.9)
    pub beta1: f32,
    /// Second moment decay rate (typically 0.999)
    pub beta2: f32,
    /// Small constant for numerical stability (typically 1e-8)
    pub epsilon: f32,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f32,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            max_iter: 1000,
            tolerance: 1e-6,
        }
    }
}

/// Adam optimizer
///
/// Adaptive moment estimation optimizer that combines momentum and RMSprop.
///
/// # Arguments
/// * `objective` - Objective function to minimize
/// * `gradient` - Gradient function
/// * `x0` - Initial point
/// * `params` - Optimization parameters
pub fn adam_optimizer<F, G>(
    objective: F,
    gradient: G,
    x0: &Tensor,
    params: Option<AdamParams>,
) -> TorshResult<(Tensor, Vec<f32>)>
where
    F: Fn(&Tensor) -> TorshResult<f32>,
    G: Fn(&Tensor) -> TorshResult<Tensor>,
{
    let params = params.unwrap_or_default();
    let mut x = x0.clone();
    let mut m = tensor_zeros_like(&x)?; // First moment
    let mut v = tensor_zeros_like(&x)?; // Second moment
    let mut objective_values = Vec::new();

    for iter in 0..params.max_iter {
        let t = (iter + 1) as f32; // Time step

        let f_val = objective(&x)?;
        objective_values.push(f_val);

        let grad = gradient(&x)?;
        let grad_norm = tensor_norm(&grad)?;

        if grad_norm < params.tolerance {
            break;
        }

        // Update first moment: m = β1*m + (1-β1)*g
        m = tensor_add(
            &tensor_scalar_mul(&m, params.beta1)?,
            &tensor_scalar_mul(&grad, 1.0 - params.beta1)?,
        )?;

        // Update second moment: v = β2*v + (1-β2)*g²
        let grad_squared = tensor_elementwise_mul(&grad, &grad)?;
        v = tensor_add(
            &tensor_scalar_mul(&v, params.beta2)?,
            &tensor_scalar_mul(&grad_squared, 1.0 - params.beta2)?,
        )?;

        // Bias correction
        let m_hat = tensor_scalar_mul(&m, 1.0 / (1.0 - params.beta1.powf(t)))?;
        let v_hat = tensor_scalar_mul(&v, 1.0 / (1.0 - params.beta2.powf(t)))?;

        // Update parameters: x = x - α * m̂ / (√v̂ + ε)
        let denominator = tensor_add(
            &tensor_sqrt(&v_hat)?,
            &tensor_full_like(&x, params.epsilon)?,
        )?;
        let update = tensor_elementwise_div(&m_hat, &denominator)?;
        x = tensor_sub(&x, &tensor_scalar_mul(&update, params.learning_rate)?)?;

        if iter % 100 == 0 {
            println!(
                "Iteration {}: f = {:.6e}, |∇f| = {:.6e}",
                iter, f_val, grad_norm
            );
        }
    }

    Ok((x, objective_values))
}
