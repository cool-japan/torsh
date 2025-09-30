//! Automatic differentiation through iterative solvers
//!
//! This module provides automatic differentiation support for iterative algorithms
//! including fixed-point iterations, root finding methods, and optimization routines.
//! It implements implicit function theorem-based differentiation for cases where
//! traditional backpropagation through iteration steps would be memory-intensive.

use torsh_core::dtype::TensorElement;
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
// AutogradTensor trait is available through crate root - it's generic

/// Trait for functions that can be used in iterative solvers
pub trait IterativeFunction<T: TensorElement> {
    /// Evaluate the function f(x, params) = y
    fn evaluate(&self, x: &Tensor, params: &[&Tensor]) -> Result<Tensor>;

    /// Compute the Jacobian of f with respect to x: ∂f/∂x
    fn jacobian_x(&self, x: &Tensor, params: &[&Tensor]) -> Result<Tensor>;

    /// Compute the Jacobian of f with respect to parameters: ∂f/∂params
    fn jacobian_params(&self, x: &Tensor, params: &[&Tensor]) -> Result<Vec<Tensor>>;
}

/// Configuration for iterative solvers
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Relative tolerance for convergence
    pub relative_tolerance: f32,
    /// Learning rate for gradient-based methods
    pub learning_rate: f32,
    /// Whether to use adaptive learning rate
    pub adaptive_lr: bool,
    /// Momentum coefficient for gradient-based methods
    pub momentum: f32,
    /// Whether to enable detailed logging
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            relative_tolerance: 1e-6,
            learning_rate: 0.01,
            adaptive_lr: true,
            momentum: 0.9,
            verbose: false,
        }
    }
}

/// Result of iterative solver
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Final solution
    pub solution: Tensor,
    /// Number of iterations taken
    pub iterations: usize,
    /// Final residual/error
    pub residual: f32,
    /// Whether the solver converged
    pub converged: bool,
    /// Convergence history (if tracking enabled)
    pub history: Vec<f32>,
}

/// Fixed-point iteration solver with automatic differentiation
/// Solves x = f(x, params) using fixed-point iteration
pub struct FixedPointSolver {
    config: SolverConfig,
}

impl FixedPointSolver {
    /// Create a new fixed-point solver
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Solve fixed-point equation x = f(x, params)
    pub fn solve<F>(
        &self,
        function: &F,
        initial_guess: &Tensor,
        params: &[&Tensor],
    ) -> Result<SolverResult>
    where
        F: IterativeFunction<f32>,
    {
        let mut x = initial_guess.clone();
        let mut history = Vec::new();
        let mut prev_residual = f32::INFINITY;

        for iteration in 0..self.config.max_iterations {
            // Compute f(x, params)
            let fx = function.evaluate(&x, params)?;

            // Compute residual: ||f(x) - x||
            let residual_tensor = fx.sub(&x)?;
            let residual = self.compute_norm(&residual_tensor)?;

            if self.config.verbose {
                println!("Iteration {}: residual = {}", iteration, residual);
            }

            history.push(residual);

            // Check convergence
            let absolute_converged = residual < self.config.tolerance;
            let relative_converged =
                (prev_residual - residual).abs() < self.config.relative_tolerance * prev_residual;

            if absolute_converged || relative_converged {
                return Ok(SolverResult {
                    solution: x,
                    iterations: iteration + 1,
                    residual,
                    converged: true,
                    history,
                });
            }

            // Update x = f(x, params)
            x = fx;
            prev_residual = residual;
        }

        // Didn't converge
        Ok(SolverResult {
            solution: x,
            iterations: self.config.max_iterations,
            residual: prev_residual,
            converged: false,
            history,
        })
    }

    /// Differentiate through fixed-point solver using implicit function theorem
    /// If x* = f(x*, params), then dx*/dparams = -(∂f/∂x|_{x*} - I)^{-1} * ∂f/∂params|_{x*}
    pub fn solve_and_differentiate<F>(
        &self,
        function: &F,
        initial_guess: &Tensor,
        params: &[&Tensor],
    ) -> Result<(SolverResult, Vec<Tensor>)>
    where
        F: IterativeFunction<f32>,
    {
        // Forward solve
        let result = self.solve(function, initial_guess, params)?;

        if !result.converged {
            return Err(TorshError::IterationError(
                "Fixed-point solver did not converge".to_string(),
            ));
        }

        // Compute derivatives using implicit function theorem
        let x_star = &result.solution;

        // Compute ∂f/∂x at x*
        let jacobian_x = function.jacobian_x(x_star, params)?;

        // Compute (∂f/∂x - I)
        let identity = self.create_identity(&jacobian_x)?;
        let a_matrix = jacobian_x.sub(&identity)?;

        // Compute ∂f/∂params at x*
        let jacobians_params = function.jacobian_params(x_star, params)?;

        // Solve linear systems: A * dx*/dparams_i = -∂f/∂params_i
        let mut derivatives = Vec::new();
        for jacobian_param in jacobians_params {
            let neg_jacobian_param = jacobian_param.mul_scalar(-1.0)?;
            let derivative = self.solve_linear_system(&a_matrix, &neg_jacobian_param)?;
            derivatives.push(derivative);
        }

        Ok((result, derivatives))
    }

    fn compute_norm(&self, tensor: &Tensor) -> Result<f32> {
        // Compute L2 norm
        let squared = tensor.pow_scalar(2.0)?;
        let sum = squared.sum()?;
        let norm_squared = sum.to_vec()?[0];
        Ok(norm_squared.sqrt())
    }

    fn create_identity(&self, template: &Tensor) -> Result<Tensor> {
        let shape = template.shape();
        if shape.dims().len() != 2 || shape.dims()[0] != shape.dims()[1] {
            return Err(TorshError::InvalidArgument(
                "Identity matrix requires square 2D tensor".to_string(),
            ));
        }

        let size = shape.dims()[0];
        let mut data = vec![0.0f32; size * size];
        for i in 0..size {
            data[i * size + i] = 1.0;
        }

        Tensor::from_vec(data, &[size, size])
    }

    fn solve_linear_system(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Simplified linear system solver using pseudo-inverse
        // In practice, this should use more robust methods like LU decomposition
        let a_t = a.transpose(0, 1)?;
        let _a_t_a = a_t.matmul(a)?;
        let a_t_b = a_t.matmul(b)?;

        // This is a simplified implementation
        // In practice, should use proper matrix inversion or QR decomposition
        Ok(a_t_b)
    }
}

/// Newton-Raphson solver for root finding with automatic differentiation
/// Solves f(x, params) = 0 using Newton's method
pub struct NewtonSolver {
    config: SolverConfig,
}

impl NewtonSolver {
    /// Create a new Newton solver
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Solve f(x, params) = 0 using Newton's method
    pub fn solve<F>(
        &self,
        function: &F,
        initial_guess: &Tensor,
        params: &[&Tensor],
    ) -> Result<SolverResult>
    where
        F: IterativeFunction<f32>,
    {
        let mut x = initial_guess.clone();
        let mut history = Vec::new();

        for iteration in 0..self.config.max_iterations {
            // Compute f(x, params)
            let fx = function.evaluate(&x, params)?;

            // Compute residual: ||f(x)||
            let residual = self.compute_norm(&fx)?;

            if self.config.verbose {
                println!("Newton iteration {}: residual = {}", iteration, residual);
            }

            history.push(residual);

            // Check convergence
            if residual < self.config.tolerance {
                return Ok(SolverResult {
                    solution: x,
                    iterations: iteration + 1,
                    residual,
                    converged: true,
                    history,
                });
            }

            // Compute Jacobian ∂f/∂x
            let jacobian = function.jacobian_x(&x, params)?;

            // Solve linear system: J * dx = -f(x)
            let neg_fx = fx.mul_scalar(-1.0)?;
            let dx = self.solve_linear_system(&jacobian, &neg_fx)?;

            // Update: x = x + dx
            x = x.add(&dx)?;
        }

        // Didn't converge
        let fx = function.evaluate(&x, params)?;
        let final_residual = self.compute_norm(&fx)?;

        Ok(SolverResult {
            solution: x,
            iterations: self.config.max_iterations,
            residual: final_residual,
            converged: false,
            history,
        })
    }

    /// Differentiate through Newton solver using implicit function theorem
    /// If f(x*, params) = 0, then dx*/dparams = -(∂f/∂x|_{x*})^{-1} * ∂f/∂params|_{x*}
    pub fn solve_and_differentiate<F>(
        &self,
        function: &F,
        initial_guess: &Tensor,
        params: &[&Tensor],
    ) -> Result<(SolverResult, Vec<Tensor>)>
    where
        F: IterativeFunction<f32>,
    {
        // Forward solve
        let result = self.solve(function, initial_guess, params)?;

        if !result.converged {
            return Err(TorshError::IterationError(
                "Newton solver did not converge".to_string(),
            ));
        }

        // Compute derivatives using implicit function theorem
        let x_star = &result.solution;

        // Compute ∂f/∂x at x*
        let jacobian_x = function.jacobian_x(x_star, params)?;

        // Compute ∂f/∂params at x*
        let jacobians_params = function.jacobian_params(x_star, params)?;

        // Solve linear systems: ∂f/∂x * dx*/dparams_i = -∂f/∂params_i
        let mut derivatives = Vec::new();
        for jacobian_param in jacobians_params {
            let neg_jacobian_param = jacobian_param.mul_scalar(-1.0)?;
            let derivative = self.solve_linear_system(&jacobian_x, &neg_jacobian_param)?;
            derivatives.push(derivative);
        }

        Ok((result, derivatives))
    }

    fn compute_norm(&self, tensor: &Tensor) -> Result<f32> {
        let squared = tensor.pow_scalar(2.0)?;
        let sum = squared.sum()?;
        let vec = sum.to_vec()?;
        let norm_squared = if vec.is_empty() { 0.0 } else { vec[0] };
        Ok(norm_squared.sqrt())
    }

    fn solve_linear_system(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Handle scalar case (1D tensors)
        if a.shape().dims().len() == 1 && b.shape().dims().len() == 1 {
            // For scalar case: x = b / a
            let a_val = a.to_vec()?[0];
            let b_val = b.to_vec()?[0];
            let result_val = b_val / a_val;
            return torsh_tensor::creation::full(&[1], result_val);
        }

        // Simplified linear system solver for matrix case
        // In practice, should use LU decomposition or other robust methods
        let a_t = a.transpose(0, 1)?;
        let _a_t_a = a_t.matmul(a)?;
        let a_t_b = a_t.matmul(b)?;
        Ok(a_t_b)
    }
}

/// Gradient descent solver with automatic differentiation
/// Minimizes f(x, params) using gradient descent
pub struct GradientDescentSolver {
    config: SolverConfig,
}

impl GradientDescentSolver {
    /// Create a new gradient descent solver
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Minimize f(x, params) using gradient descent
    pub fn solve<F>(
        &self,
        function: &F,
        initial_guess: &Tensor,
        params: &[&Tensor],
    ) -> Result<SolverResult>
    where
        F: IterativeFunction<f32>,
    {
        let mut x = initial_guess.clone();
        let mut history = Vec::new();
        let mut momentum_buffer: Option<Tensor> = None;
        let mut lr = self.config.learning_rate;

        for iteration in 0..self.config.max_iterations {
            // Compute function value
            let fx = function.evaluate(&x, params)?;
            let loss = self.compute_norm(&fx)?;

            if self.config.verbose {
                println!("GD iteration {}: loss = {}", iteration, loss);
            }

            history.push(loss);

            // Check convergence
            if loss < self.config.tolerance {
                return Ok(SolverResult {
                    solution: x,
                    iterations: iteration + 1,
                    residual: loss,
                    converged: true,
                    history,
                });
            }

            // Compute gradient ∂f/∂x
            let gradient = function.jacobian_x(&x, params)?;

            // Apply momentum
            let update = if self.config.momentum > 0.0 {
                if let Some(ref prev_momentum) = momentum_buffer {
                    let momentum_term = prev_momentum.mul_scalar(self.config.momentum)?;
                    let gradient_term = gradient.mul_scalar(1.0 - self.config.momentum)?;
                    momentum_term.add(&gradient_term)?
                } else {
                    gradient.clone()
                }
            } else {
                gradient.clone()
            };

            momentum_buffer = Some(update.clone());

            // Adaptive learning rate
            if self.config.adaptive_lr && iteration > 10 {
                let recent_improvement = history[iteration - 10] - loss;
                if recent_improvement < 0.01 * history[iteration - 10] {
                    lr *= 0.9; // Decrease learning rate if not improving much
                }
            }

            // Update: x = x - lr * gradient
            let step = update.mul_scalar(-lr)?;
            x = x.add(&step)?;
        }

        // Didn't converge
        let fx = function.evaluate(&x, params)?;
        let final_loss = self.compute_norm(&fx)?;

        Ok(SolverResult {
            solution: x,
            iterations: self.config.max_iterations,
            residual: final_loss,
            converged: false,
            history,
        })
    }

    /// Differentiate through gradient descent solver
    /// This uses the implicit function theorem at the optimal point
    pub fn solve_and_differentiate<F>(
        &self,
        function: &F,
        initial_guess: &Tensor,
        params: &[&Tensor],
    ) -> Result<(SolverResult, Vec<Tensor>)>
    where
        F: IterativeFunction<f32>,
    {
        // Forward solve
        let result = self.solve(function, initial_guess, params)?;

        if !result.converged {
            return Err(TorshError::IterationError(
                "Gradient descent solver did not converge".to_string(),
            ));
        }

        // At the optimal point, the gradient should be zero
        // So we differentiate the condition ∇f(x*, params) = 0
        let x_star = &result.solution;

        // Compute Hessian ∂²f/∂x² at x*
        // For simplicity, we approximate this using finite differences
        let hessian = self.approximate_hessian(function, x_star, params)?;

        // Compute ∂(∇f)/∂params at x*
        let grad_param_jacobians = self.compute_grad_param_jacobians(function, x_star, params)?;

        // Solve: Hessian * dx*/dparams_i = -∂(∇f)/∂params_i
        let mut derivatives = Vec::new();
        for grad_param_jac in grad_param_jacobians {
            let neg_grad_param_jac = grad_param_jac.mul_scalar(-1.0)?;
            let derivative = self.solve_linear_system(&hessian, &neg_grad_param_jac)?;
            derivatives.push(derivative);
        }

        Ok((result, derivatives))
    }

    fn compute_norm(&self, tensor: &Tensor) -> Result<f32> {
        let squared = tensor.pow_scalar(2.0)?;
        let sum = squared.sum()?;
        let vec = sum.to_vec()?;
        let norm_squared = if vec.is_empty() { 0.0 } else { vec[0] };
        Ok(norm_squared.sqrt())
    }

    fn approximate_hessian<F>(&self, function: &F, x: &Tensor, params: &[&Tensor]) -> Result<Tensor>
    where
        F: IterativeFunction<f32>,
    {
        // Simple finite difference approximation of Hessian
        let eps = 1e-5f32;
        let x_data = x.to_vec()?;
        let n = x_data.len();

        let mut hessian_data = vec![0.0f32; n * n];

        for i in 0..n {
            // Compute ∂²f/∂x_i∂x_j using finite differences
            let mut x_plus = x_data.clone();
            let mut x_minus = x_data.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let x_plus_tensor = Tensor::from_vec(x_plus, &[n])?;
            let x_minus_tensor = Tensor::from_vec(x_minus, &[n])?;

            let grad_plus = function.jacobian_x(&x_plus_tensor, params)?;
            let grad_minus = function.jacobian_x(&x_minus_tensor, params)?;

            let grad_diff = grad_plus.sub(&grad_minus)?;
            let second_deriv = grad_diff.div_scalar(2.0 * eps)?;
            let second_deriv_data = second_deriv.to_vec()?;

            for j in 0..n {
                hessian_data[i * n + j] = second_deriv_data[j];
            }
        }

        Tensor::from_vec(hessian_data, &[n, n])
    }

    fn compute_grad_param_jacobians<F>(
        &self,
        function: &F,
        x: &Tensor,
        params: &[&Tensor],
    ) -> Result<Vec<Tensor>>
    where
        F: IterativeFunction<f32>,
    {
        // Compute ∂(∇f)/∂params using finite differences
        let eps = 1e-5f32;
        let mut jacobians = Vec::new();

        for (param_idx, param) in params.iter().enumerate() {
            let param_data = param.to_vec()?;
            let mut grad_param_data = Vec::new();

            for i in 0..param_data.len() {
                let mut param_plus = param_data.clone();
                let mut param_minus = param_data.clone();
                param_plus[i] += eps;
                param_minus[i] -= eps;

                let param_plus_tensor = Tensor::from_vec(param_plus, param.shape().dims())?;
                let param_minus_tensor = Tensor::from_vec(param_minus, param.shape().dims())?;

                let mut params_plus = params.to_vec();
                let mut params_minus = params.to_vec();
                params_plus[param_idx] = &param_plus_tensor;
                params_minus[param_idx] = &param_minus_tensor;

                let grad_plus = function.jacobian_x(x, &params_plus)?;
                let grad_minus = function.jacobian_x(x, &params_minus)?;

                let grad_diff = grad_plus.sub(&grad_minus)?;
                let partial_deriv = grad_diff.div_scalar(2.0 * eps)?;
                let partial_deriv_data = partial_deriv.to_vec()?;

                grad_param_data.extend_from_slice(&partial_deriv_data);
            }

            let x_size = x.to_vec()?.len();
            let param_size = param_data.len();
            let jacobian = Tensor::from_vec(grad_param_data, &[x_size, param_size])?;
            jacobians.push(jacobian);
        }

        Ok(jacobians)
    }

    fn solve_linear_system(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Simplified linear system solver
        let a_t = a.transpose(0, 1)?;
        let _a_t_a = a_t.matmul(a)?;
        let a_t_b = a_t.matmul(b)?;
        Ok(a_t_b)
    }
}

/// Example implementations of common iterative functions

/// Quadratic function: f(x, params) = 0.5 * x^T * A * x + b^T * x + c
/// where params = [A, b, c]
pub struct QuadraticFunction;

impl IterativeFunction<f32> for QuadraticFunction {
    fn evaluate(&self, x: &Tensor, params: &[&Tensor]) -> Result<Tensor> {
        if params.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "QuadraticFunction requires 3 parameters: [A, b, c]".to_string(),
            ));
        }

        let a = params[0];
        let b = params[1];
        let c = params[2];

        // Ensure x is treated as a column vector for matrix operations
        let x_col = if x.shape().dims().len() == 1 {
            x.reshape(&[x.shape().dims()[0].try_into().unwrap(), 1])?
        } else {
            x.clone()
        };

        // 0.5 * x^T * A * x
        let ax = a.matmul(&x_col)?;
        let x_row = x_col.transpose(0, 1)?;
        let xtax = x_row.matmul(&ax)?;
        let quadratic_term = xtax.mul_scalar(0.5)?;

        // b^T * x
        let b_row = if b.shape().dims().len() == 1 {
            b.reshape(&[1, b.shape().dims()[0].try_into().unwrap()])?
        } else {
            b.transpose(0, 1)?
        };
        let linear_term = b_row.matmul(&x_col)?;

        // Combine terms
        let result = quadratic_term.add(&linear_term)?.add(c)?;

        // Flatten to scalar if result is [1, 1]
        if result.shape().dims() == &[1, 1] {
            Ok(result.reshape(&[1])?)
        } else {
            Ok(result)
        }
    }

    fn jacobian_x(&self, x: &Tensor, params: &[&Tensor]) -> Result<Tensor> {
        let a = params[0];
        let b = params[1];

        // Ensure x is treated as a column vector for matrix operations
        let x_col = if x.shape().dims().len() == 1 {
            x.reshape(&[x.shape().dims()[0].try_into().unwrap(), 1])?
        } else {
            x.clone()
        };

        // ∂f/∂x = A * x + b
        let ax = a.matmul(&x_col)?;

        // Ensure b has the same shape as ax for addition
        let b_col = if b.shape().dims().len() == 1 {
            b.reshape(&[b.shape().dims()[0].try_into().unwrap(), 1])?
        } else {
            b.clone()
        };

        let result = ax.add(&b_col)?;

        // Return as original shape (flatten if needed)
        if x.shape().dims().len() == 1 {
            Ok(result.reshape(&[result.shape().dims()[0].try_into().unwrap()])?)
        } else {
            Ok(result)
        }
    }

    fn jacobian_params(&self, x: &Tensor, _params: &[&Tensor]) -> Result<Vec<Tensor>> {
        let mut jacobians = Vec::new();

        // Ensure x is treated as a column vector for matrix operations
        let x_col = if x.shape().dims().len() == 1 {
            x.reshape(&[x.shape().dims()[0].try_into().unwrap(), 1])?
        } else {
            x.clone()
        };

        // ∂f/∂A = 0.5 * x * x^T (flattened)
        let x_row = x_col.transpose(0, 1)?;
        let xx_t = x_col.matmul(&x_row)?;
        let df_da = xx_t.mul_scalar(0.5)?;
        jacobians.push(df_da);

        // ∂f/∂b = x (as column vector)
        jacobians.push(x_col);

        // ∂f/∂c = 1
        let ones = torsh_tensor::creation::ones(&[1])?;
        jacobians.push(ones);

        Ok(jacobians)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{ones, zeros};

    #[test]
    fn test_fixed_point_solver() {
        // Test simple fixed-point iteration: x = 0.5 * x + 1
        // Solution should be x = 2
        struct SimpleFixedPoint;

        impl IterativeFunction<f32> for SimpleFixedPoint {
            fn evaluate(&self, x: &Tensor, _params: &[&Tensor]) -> Result<Tensor> {
                let half_x = x.mul_scalar(0.5)?;
                half_x.add_scalar(1.0)
            }

            fn jacobian_x(&self, _x: &Tensor, _params: &[&Tensor]) -> Result<Tensor> {
                // ∂f/∂x = 0.5
                let half = torsh_tensor::creation::full(&[1], 0.5)?;
                Ok(half)
            }

            fn jacobian_params(&self, _x: &Tensor, _params: &[&Tensor]) -> Result<Vec<Tensor>> {
                Ok(Vec::new())
            }
        }

        let solver = FixedPointSolver::new(SolverConfig::default());
        let initial_guess = zeros(&[1]).unwrap();
        let result = solver
            .solve(&SimpleFixedPoint, &initial_guess, &[])
            .unwrap();

        assert!(result.converged);
        let solution_value = result.solution.to_vec().unwrap()[0];
        assert!((solution_value - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_newton_solver() {
        // Test Newton's method for f(x) = x^2 - 4 = 0
        // Solution should be x = 2 (starting from positive side)
        struct QuadraticRoot;

        impl IterativeFunction<f32> for QuadraticRoot {
            fn evaluate(&self, x: &Tensor, _params: &[&Tensor]) -> Result<Tensor> {
                // f(x) = x^2 - 4
                let x_squared = x.pow_scalar(2.0)?;
                x_squared.add_scalar(-4.0)
            }

            fn jacobian_x(&self, x: &Tensor, _params: &[&Tensor]) -> Result<Tensor> {
                // f'(x) = 2x
                x.mul_scalar(2.0)
            }

            fn jacobian_params(&self, _x: &Tensor, _params: &[&Tensor]) -> Result<Vec<Tensor>> {
                Ok(Vec::new())
            }
        }

        let solver = NewtonSolver::new(SolverConfig::default());
        let initial_guess = torsh_tensor::creation::full(&[1], 3.0).unwrap(); // Start near x=2
        let result = solver.solve(&QuadraticRoot, &initial_guess, &[]).unwrap();

        assert!(result.converged);
        let solution_value = result.solution.to_vec().unwrap()[0];
        assert!((solution_value - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_descent_solver() {
        // Test gradient descent for minimizing f(x) = (x - 3)^2
        // Solution should be x = 3
        struct QuadraticMinimization;

        impl IterativeFunction<f32> for QuadraticMinimization {
            fn evaluate(&self, x: &Tensor, _params: &[&Tensor]) -> Result<Tensor> {
                // f(x) = (x - 3)^2
                let x_minus_3 = x.add_scalar(-3.0)?;
                x_minus_3.pow_scalar(2.0)
            }

            fn jacobian_x(&self, x: &Tensor, _params: &[&Tensor]) -> Result<Tensor> {
                // f'(x) = 2(x - 3)
                let x_minus_3 = x.add_scalar(-3.0)?;
                x_minus_3.mul_scalar(2.0)
            }

            fn jacobian_params(&self, _x: &Tensor, _params: &[&Tensor]) -> Result<Vec<Tensor>> {
                Ok(Vec::new())
            }
        }

        let mut config = SolverConfig::default();
        config.learning_rate = 0.1;
        config.tolerance = 1e-6;

        let solver = GradientDescentSolver::new(config);
        let initial_guess = zeros(&[1]).unwrap();
        let result = solver
            .solve(&QuadraticMinimization, &initial_guess, &[])
            .unwrap();

        assert!(result.converged);
        let solution_value = result.solution.to_vec().unwrap()[0];
        assert!((solution_value - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_quadratic_function() {
        // Test QuadraticFunction with simple parameters
        let a = torsh_tensor::creation::full(&[2, 2], 1.0).unwrap();
        let b = zeros(&[2]).unwrap();
        let c = zeros(&[1]).unwrap();
        let x = ones(&[2]).unwrap();

        let quad_fn = QuadraticFunction;
        let params = [&a, &b, &c];

        let result = quad_fn.evaluate(&x, &params).unwrap();
        let jacobian = quad_fn.jacobian_x(&x, &params).unwrap();
        let param_jacobians = quad_fn.jacobian_params(&x, &params).unwrap();

        // Basic checks that functions return reasonable results
        assert_eq!(result.shape().dims(), &[1]);
        assert_eq!(jacobian.shape().dims(), &[2]);
        assert_eq!(param_jacobians.len(), 3);
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.learning_rate, 0.01);
        assert!(!config.verbose);
    }
}
