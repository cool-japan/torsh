//! Linear system solvers
//!
//! This module provides a comprehensive set of linear algebra solvers organized into
//! specialized modules for different types of problems. The modular structure allows
//! for efficient compilation and easy maintenance while preserving backward compatibility.
//!
//! # Modular Organization
//!
//! - **Core Solvers**: General-purpose linear system solvers (LU, triangular, least squares)
//! - **Structured Matrix Solvers**: Specialized algorithms for structured matrices (banded, tridiagonal, etc.)
//! - **Error Analysis & Refinement**: Iterative refinement and error bound estimation
//! - **Regularization Methods**: Solvers for ill-conditioned systems (Tikhonov, SVD, damped)
//! - **Advanced Methods**: Multigrid and other advanced iterative techniques
//!
//! # Usage Examples
//!
//! ```rust
//! use torsh_linalg::solve::{solve, solve_triangular, inv};
//! use torsh_tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create example matrices
//! let matrix_a = Tensor::from_data(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
//! let vector_b = Tensor::from_data(vec![3.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
//! let upper_triangular = Tensor::from_data(vec![2.0, 1.0, 0.0, 1.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
//! let b = Tensor::from_data(vec![3.0, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
//!
//! // Solve general linear system Ax = b
//! let solution = solve(&matrix_a, &vector_b)?;
//!
//! // Solve triangular system
//! let x = solve_triangular(&upper_triangular, &b, true)?;
//!
//! // Matrix inversion
//! let a_inv = inv(&matrix_a)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! - Choose specialized solvers for structured matrices when possible (O(n) vs O(n³))
//! - Use iterative refinement for improved accuracy with ill-conditioned systems
//! - Consider regularization methods for underdetermined or ill-posed problems
//! - Multigrid methods are highly efficient for PDE-type problems

#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::manual_div_ceil)]

// Re-export all solver functions and types from the modular system
pub use crate::solvers::*;

// Provide direct access to specialized modules for advanced users
pub mod core {
    //! Core linear algebra solvers
    pub use crate::solvers::core::*;
}

pub mod structured {
    //! Structured matrix solvers
    pub use crate::solvers::structured::*;
}

pub mod refinement {
    //! Error analysis and iterative refinement
    pub use crate::solvers::refinement::*;
}

pub mod regularization {
    //! Regularization methods for ill-conditioned systems
    pub use crate::solvers::regularization::*;
}

pub mod advanced {
    //! Advanced iterative methods
    pub use crate::solvers::advanced::*;
}
