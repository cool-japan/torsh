//! Linear algebra solver modules and unified interface
//!
//! This module provides a comprehensive collection of linear algebra solvers organized into
//! specialized modules by functionality. The module provides both modular access to specific
//! solver categories and a unified interface through comprehensive re-exports for backward
//! compatibility.
//!
//! # Module Organization
//!
//! - [`core`] - Fundamental solvers (solve, solve_triangular, inv, pinv, lstsq)
//! - [`structured`] - Specialized structured matrix solvers (banded, tridiagonal, etc.)
//! - [`refinement`] - Error analysis and iterative refinement methods
//! - [`regularization`] - Regularized solvers (Tikhonov, SVD, damped least squares)
//! - [`advanced`] - Advanced methods (multigrid solvers with configurations)
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_linalg::solvers::{solve, solve_triangular, inv};
//! use torsh_tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Solve general linear system
//! let a = Tensor::from_data(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
//! let b = Tensor::from_data(vec![3.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
//! let x = solve(&a, &b)?;
//!
//! // Compute matrix inverse
//! let a_inv = inv(&a)?;
//! # Ok(())
//! # }
//! ```

// Declare specialized modules
pub mod advanced;
pub mod core;
pub mod refinement;
pub mod regularization;
pub mod structured;

// ================================================================================================
// COMPREHENSIVE RE-EXPORTS FOR BACKWARD COMPATIBILITY
// ================================================================================================

// --------------------------------------------------------------------------------
// Core Linear Algebra Functions
// --------------------------------------------------------------------------------

/// Solve linear system Ax = b using LU decomposition with partial pivoting
pub use self::core::solve;

/// Solve triangular system Ax = b where A is upper or lower triangular
pub use self::core::solve_triangular;

/// Compute matrix inverse A^(-1) such that A * A^(-1) = I
pub use self::core::inv;

/// Compute Moore-Penrose pseudoinverse A^+ for general matrices
pub use self::core::pinv;

/// Solve least squares problem min ||Ax - b||_2 with residual and rank analysis
pub use self::core::lstsq;

// --------------------------------------------------------------------------------
// Structured Matrix Solvers
// --------------------------------------------------------------------------------

/// Solve band linear system using specialized band storage format
pub use self::structured::solve_banded;

/// Solve tridiagonal linear system using Thomas algorithm (O(n) complexity)
pub use self::structured::solve_tridiagonal;

/// Solve pentadiagonal linear system using specialized algorithm
pub use self::structured::solve_pentadiagonal;

/// Solve Toeplitz linear system using Levinson algorithm (O(n²) complexity)
pub use self::structured::solve_toeplitz;

/// Solve Hankel linear system for anti-diagonal constant matrices
pub use self::structured::solve_hankel;

/// Solve circulant linear system using eigenvalue decomposition
pub use self::structured::solve_circulant;

/// Solve Vandermonde linear system using Björck-Pereyra algorithm (O(n²) complexity)
pub use self::structured::solve_vandermonde;

// --------------------------------------------------------------------------------
// Error Analysis and Iterative Refinement
// --------------------------------------------------------------------------------

/// Estimate error bounds for linear system solution (backward/forward/condition-based)
pub use self::refinement::estimate_error_bounds;

/// Iterative refinement for improving solution accuracy of ill-conditioned systems
pub use self::refinement::iterative_refinement;

/// Mixed precision iterative refinement using higher precision for residual computation
pub use self::refinement::mixed_precision_refinement;

/// Solve with automatic iterative refinement based on error analysis
pub use self::refinement::solve_with_refinement;

// --------------------------------------------------------------------------------
// Regularization Methods
// --------------------------------------------------------------------------------

/// Solve regularized linear system using Tikhonov regularization
pub use self::regularization::solve_tikhonov;

/// Solve using truncated SVD for rank-deficient problems
pub use self::regularization::solve_truncated_svd;

/// Solve using damped least squares with adaptive regularization
pub use self::regularization::solve_damped_least_squares;

// --------------------------------------------------------------------------------
// Advanced Multigrid Methods
// --------------------------------------------------------------------------------

/// Solve linear system using multigrid method with default configuration
pub use self::advanced::solve_multigrid;

/// Solve linear system using multigrid method with custom configuration
pub use self::advanced::solve_multigrid_with_config;

/// Configuration parameters for multigrid solvers
pub use self::advanced::MultigridConfig;

/// Multigrid cycle types (V-cycle, W-cycle, F-cycle)
pub use self::advanced::MultigridCycle;

/// Smoothing methods for multigrid (Jacobi, Gauss-Seidel, SOR)
pub use self::advanced::SmoothingMethod;

/// Coarsening strategies for multigrid hierarchy
pub use self::advanced::CoarseningStrategy;

/// Complete multigrid solver with state management
pub use self::advanced::MultigridSolver;
