//! Optimization utilities for tensor operations
//!
//! This module provides optimization algorithms including line search methods,
//! gradient descent variants, quasi-Newton methods, and adaptive algorithm selection
//! commonly used in machine learning and numerical optimization.
//!
//! # Organization
//!
//! The optimization module is organized into focused sub-modules:
//!
//! - [`utilities`]: Basic tensor operations used by optimization algorithms
//! - [`line_search`]: Line search methods (backtracking, Wolfe conditions)
//! - [`mod@gradient_descent`]: Gradient descent variants (basic, momentum, Adam)
//! - [`quasi_newton`]: Quasi-Newton methods (L-BFGS)
//! - [`adaptive`]: Adaptive algorithm selection based on problem characteristics
//!
//! # Examples
//!
//! ```rust
//! use torsh_functional::optimization::*;
//! use torsh_tensor::Tensor;
//! use torsh_core::Result as TorshResult;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Basic gradient descent
//!     let objective = |x: &Tensor| -> TorshResult<f32> {
//!         let data = x.data()?;
//!         Ok(data[0].powi(2)) // f(x) = x²
//!     };
//!
//!     let gradient = |x: &Tensor| -> TorshResult<Tensor> {
//!         let data = x.data()?;
//!         Ok(Tensor::from_vec(vec![2.0 * data[0]], &[1])?) // f'(x) = 2x
//!     };
//!
//!     let x0 = Tensor::from_vec(vec![1.0], &[1])?;
//!     let (x_opt, history) = gradient_descent(objective, gradient, &x0, None)?;
//!     Ok(())
//! }
//! ```

pub mod adaptive;
pub mod gradient_descent;
pub mod line_search;
pub mod quasi_newton;
pub mod utilities;

// Re-export commonly used types and functions
pub use adaptive::{
    analyze_optimization_problem, auto_configure_optimization, AdaptiveAlgorithmSelector,
    OptimizationAlgorithm, TensorCharacteristics,
};
pub use gradient_descent::{
    adam_optimizer, gradient_descent, momentum_gradient_descent, AdamParams, GradientDescentParams,
    MomentumParams,
};
pub use line_search::{
    backtracking_line_search, wolfe_line_search, BacktrackingParams, LineSearchMethod, WolfeParams,
};
pub use quasi_newton::{lbfgs_optimizer, BFGSParams};
pub use utilities::*;
