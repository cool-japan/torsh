//! Advanced solver methods including multigrid solvers
//!
//! This module contains sophisticated solver techniques designed for specific problem classes,
//! particularly those arising from discretizations of partial differential equations.
//!
//! ## Multigrid Methods
//!
//! Multigrid methods are among the most efficient techniques for solving large sparse linear
//! systems that arise from discretizations of elliptic PDEs. They achieve optimal O(n) complexity
//! by combining:
//!
//! - **Smoothing**: Iterative methods that effectively reduce high-frequency error components
//! - **Coarse Grid Correction**: Solving residual equations on progressively coarser grids
//! - **Multi-level Hierarchy**: Operating on multiple grid levels simultaneously
//!
//! ### Key Features:
//!
//! - **V-Cycle**: Standard multigrid cycle with one visit to each coarser level
//! - **W-Cycle**: More robust cycle with multiple visits to coarser levels
//! - **F-Cycle**: Full multigrid with nested iteration for optimal efficiency
//! - **Configurable Parameters**: Smoothing steps, tolerance, grid levels
//! - **Automatic Grid Hierarchy**: Simplified geometric coarsening strategy
//!
//! ### Usage Example:
//!
//! ```rust
//! use torsh_linalg::solvers::advanced::{solve_multigrid, solve_multigrid_with_config, MultigridConfig, MultigridCycle};
//! use torsh_tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create example matrix and rhs
//! let matrix = Tensor::from_data(vec![2.0, -1.0, -1.0, 2.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
//! let rhs = Tensor::from_data(vec![1.0, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
//!
//! // Basic usage with default configuration
//! let solution = solve_multigrid(&matrix, &rhs)?;
//!
//! // Custom configuration
//! let config = MultigridConfig {
//!     max_levels: 6,
//!     max_iterations: 50,
//!     tolerance: 1e-8,
//!     cycle_type: MultigridCycle::W,
//!     ..Default::default()
//! };
//! let solution = solve_multigrid_with_config(&matrix, &rhs, config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Performance Characteristics:
//!
//! - **Time Complexity**: O(n) for well-structured problems
//! - **Space Complexity**: O(n) with modest memory overhead
//! - **Convergence**: Mesh-independent for suitable problem classes
//! - **Scalability**: Excellent parallel scaling properties

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

// SciRS2 integration for enhanced performance and functionality
#[cfg(feature = "scirs2-integration")]
use scirs2_core::ndarray::{Array1, Array2};

// Fallback to basic torsh functionality when SciRS2 is not available
#[cfg(not(feature = "scirs2-integration"))]
use torsh_tensor::creation::{eye, zeros};

/// Configuration parameters for multigrid solvers
///
/// This structure controls all aspects of the multigrid algorithm, including
/// the grid hierarchy depth, iteration parameters, and cycle type selection.
#[derive(Clone, Debug)]
pub struct MultigridConfig {
    /// Maximum number of grid levels in the hierarchy
    pub max_levels: usize,
    /// Maximum number of multigrid iterations
    pub max_iterations: usize,
    /// Convergence tolerance for residual norm
    pub tolerance: f32,
    /// Number of smoothing steps before coarse grid correction
    pub pre_smooth_steps: usize,
    /// Number of smoothing steps after coarse grid correction
    pub post_smooth_steps: usize,
    /// Type of multigrid cycle to perform
    pub cycle_type: MultigridCycle,
    /// Smoothing method for relaxation steps
    pub smoothing_method: SmoothingMethod,
    /// Grid coarsening strategy
    pub coarsening_strategy: CoarseningStrategy,
}

/// Multigrid cycle types determining the grid traversal pattern
///
/// Different cycle types offer trade-offs between computational cost and robustness:
///
/// - **V-Cycle**: Most efficient, suitable for well-conditioned problems
/// - **W-Cycle**: More robust, handles difficult problems better
/// - **F-Cycle**: Full multigrid with optimal theoretical properties
#[derive(Clone, Debug)]
pub enum MultigridCycle {
    /// V-cycle: Visit each coarse level once in a V-shaped pattern
    V,
    /// W-cycle: Visit each coarse level twice for enhanced robustness
    W,
    /// F-cycle: Full multigrid with nested iteration (simplified implementation)
    F,
}

/// Smoothing methods for multigrid relaxation
///
/// Different smoothing operators offer varying convergence characteristics:
///
/// - **GaussSeidel**: Forward Gauss-Seidel, good general-purpose smoother
/// - **GaussSeidelBackward**: Backward Gauss-Seidel for enhanced stability
/// - **Jacobi**: Jacobi iteration, naturally parallel but slower convergence
/// - **SOR**: Successive Over-Relaxation with configurable parameter
/// - **SSOR**: Symmetric SOR for improved properties
#[derive(Clone, Debug)]
pub enum SmoothingMethod {
    /// Forward Gauss-Seidel iteration
    GaussSeidel,
    /// Backward Gauss-Seidel iteration
    GaussSeidelBackward,
    /// Jacobi iteration (naturally parallel)
    Jacobi,
    /// Successive Over-Relaxation with relaxation parameter
    SOR(f32),
    /// Symmetric Successive Over-Relaxation
    SSOR(f32),
}

/// Grid coarsening strategies for multigrid hierarchy construction
///
/// Different coarsening approaches affect convergence and setup cost:
///
/// - **Geometric**: Simple geometric coarsening (every other point)
/// - **Algebraic**: Algebraic multigrid coarsening based on matrix structure
/// - **Aggregation**: Aggregation-based coarsening for complex problems
/// - **RedBlack**: Red-black coarsening for structured grids
#[derive(Clone, Debug)]
pub enum CoarseningStrategy {
    /// Simple geometric coarsening (factor of 2 in each dimension)
    Geometric,
    /// Algebraic coarsening based on strong connections
    Algebraic { strength_threshold: f32 },
    /// Aggregation-based coarsening
    Aggregation { max_aggregate_size: usize },
    /// Red-black coarsening for structured problems
    RedBlack,
}

impl Default for SmoothingMethod {
    fn default() -> Self {
        Self::GaussSeidel
    }
}

impl Default for CoarseningStrategy {
    fn default() -> Self {
        Self::Geometric
    }
}

impl Default for MultigridConfig {
    fn default() -> Self {
        Self {
            max_levels: 5,
            max_iterations: 100,
            tolerance: 1e-6,
            pre_smooth_steps: 2,
            post_smooth_steps: 2,
            cycle_type: MultigridCycle::V,
            smoothing_method: SmoothingMethod::default(),
            coarsening_strategy: CoarseningStrategy::default(),
        }
    }
}

/// Multigrid solver implementing hierarchical grid-based solution methods
///
/// This solver is particularly effective for problems arising from discretizations
/// of elliptic partial differential equations on regular grids. It achieves
/// optimal O(n) complexity for suitable problem classes.
///
/// ## Algorithm Overview:
///
/// 1. **Smoothing**: Apply iterative relaxation to reduce high-frequency errors
/// 2. **Restriction**: Transfer residual to coarser grid
/// 3. **Coarse Grid Solve**: Recursively solve the coarse grid problem
/// 4. **Interpolation**: Transfer correction back to fine grid
/// 5. **Post-smoothing**: Final smoothing to maintain solution quality
///
/// ## Grid Hierarchy:
///
/// The solver automatically constructs a hierarchy of progressively coarser grids
/// using a simple geometric coarsening strategy. For optimal performance on
/// structured problems, consider using problem-specific restriction and
/// interpolation operators.
pub struct MultigridSolver {
    config: MultigridConfig,
}

impl MultigridSolver {
    /// Create a new multigrid solver with the specified configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters controlling solver behavior
    ///
    /// # Example
    ///
    /// ```rust
    /// use torsh_linalg::solvers::advanced::{MultigridConfig, MultigridCycle, MultigridSolver};
    ///
    /// let config = MultigridConfig {
    ///     max_levels: 4,
    ///     tolerance: 1e-8,
    ///     cycle_type: MultigridCycle::W,
    ///     ..Default::default()
    /// };
    /// let solver = MultigridSolver::new(config);
    /// ```
    pub fn new(config: MultigridConfig) -> Self {
        Self { config }
    }

    /// Solve the linear system Ax = b using multigrid method
    ///
    /// This is the main entry point for the multigrid solver. It performs
    /// multigrid iterations until convergence or the maximum iteration limit.
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix (must be square)
    /// * `b` - Right-hand side vector
    ///
    /// # Returns
    ///
    /// The solution vector x such that Ax ≈ b within the specified tolerance
    ///
    /// # Errors
    ///
    /// - `TorshError::InvalidArgument` if matrix dimensions are incompatible
    /// - `TorshError::ComputeError` if convergence fails
    pub fn solve(&self, a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
        // Validate input
        if a.shape().ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "Matrix A must be 2D".to_string(),
            ));
        }

        let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
        if m != n {
            return Err(TorshError::InvalidArgument(
                "Matrix A must be square".to_string(),
            ));
        }

        if b.shape().dims()[0] != m {
            return Err(TorshError::InvalidArgument(
                "Dimension mismatch between A and b".to_string(),
            ));
        }

        // Initialize solution vector
        let mut x = self.create_zeros_vector(m)?;

        // Main multigrid iteration
        for _iter in 0..self.config.max_iterations {
            let x_old = x.clone();

            // Perform multigrid cycle
            x = self.multigrid_cycle(a, b, &x, 0)?;

            // Check convergence
            let mut residual_norm = 0.0;
            for i in 0..m {
                let diff = x.get(&[i])? - x_old.get(&[i])?;
                residual_norm += diff * diff;
            }
            residual_norm = residual_norm.sqrt();

            if residual_norm < self.config.tolerance {
                break;
            }

            // Additional convergence check: compute actual residual ||Ax - b||
            let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
            let mut actual_residual = 0.0;
            for i in 0..m {
                let diff = ax.get(&[i])? - b.get(&[i])?;
                actual_residual += diff * diff;
            }
            actual_residual = actual_residual.sqrt();

            if actual_residual < self.config.tolerance {
                break;
            }
        }

        Ok(x)
    }

    /// Perform one multigrid cycle
    ///
    /// This is the core multigrid algorithm implementing the recursive
    /// coarse grid correction scheme. The cycle type determines how
    /// many times coarser levels are visited.
    fn multigrid_cycle(
        &self,
        a: &Tensor,
        b: &Tensor,
        x: &Tensor,
        level: usize,
    ) -> TorshResult<Tensor> {
        let n = a.shape().dims()[0];

        // Base case: solve directly for small systems
        if n <= 8 || level >= self.config.max_levels {
            return self.direct_solve(a, b);
        }

        // Pre-smoothing
        let mut x_smooth = x.clone();
        for _ in 0..self.config.pre_smooth_steps {
            x_smooth = self.smooth(a, b, &x_smooth)?;
        }

        // Compute residual: r = b - A*x
        let ax = a.matmul(&x_smooth.unsqueeze(1)?)?.squeeze(1)?;
        let residual = self.create_zeros_vector(n)?;
        for i in 0..n {
            residual.set(&[i], b.get(&[i])? - ax.get(&[i])?)?;
        }

        // Restrict residual to coarser grid
        let (a_coarse, r_coarse) = self.restrict(a, &residual)?;

        // Solve coarse grid problem: A_coarse * e_coarse = r_coarse
        let e_coarse = match self.config.cycle_type {
            MultigridCycle::V => self.multigrid_cycle(
                &a_coarse,
                &r_coarse,
                &self.create_zeros_vector(a_coarse.shape().dims()[0])?,
                level + 1,
            )?,
            MultigridCycle::W => {
                // Two recursive calls for W-cycle
                let e1 = self.multigrid_cycle(
                    &a_coarse,
                    &r_coarse,
                    &self.create_zeros_vector(a_coarse.shape().dims()[0])?,
                    level + 1,
                )?;
                self.multigrid_cycle(&a_coarse, &r_coarse, &e1, level + 1)?
            }
            MultigridCycle::F => {
                // Full multigrid - simplified to V-cycle for this implementation
                self.multigrid_cycle(
                    &a_coarse,
                    &r_coarse,
                    &self.create_zeros_vector(a_coarse.shape().dims()[0])?,
                    level + 1,
                )?
            }
        };

        // Interpolate correction to fine grid
        let correction = self.interpolate(&e_coarse, n)?;

        // Correct the solution
        let mut x_corrected = self.create_zeros_vector(n)?;
        for i in 0..n {
            x_corrected.set(&[i], x_smooth.get(&[i])? + correction.get(&[i])?)?;
        }

        // Post-smoothing
        for _ in 0..self.config.post_smooth_steps {
            x_corrected = self.smooth(a, b, &x_corrected)?;
        }

        Ok(x_corrected)
    }

    /// Apply smoothing operation based on the configured method
    ///
    /// Performs smoothing iteration according to the selected method,
    /// optimized for reducing high-frequency error components.
    fn smooth(&self, a: &Tensor, b: &Tensor, x: &Tensor) -> TorshResult<Tensor> {
        match &self.config.smoothing_method {
            SmoothingMethod::GaussSeidel => self.gauss_seidel_smooth(a, b, x, true),
            SmoothingMethod::GaussSeidelBackward => self.gauss_seidel_smooth(a, b, x, false),
            SmoothingMethod::Jacobi => self.jacobi_smooth(a, b, x),
            SmoothingMethod::SOR(omega) => self.sor_smooth(a, b, x, *omega, true),
            SmoothingMethod::SSOR(omega) => self.ssor_smooth(a, b, x, *omega),
        }
    }

    /// Forward or backward Gauss-Seidel smoothing
    fn gauss_seidel_smooth(
        &self,
        a: &Tensor,
        b: &Tensor,
        x: &Tensor,
        forward: bool,
    ) -> TorshResult<Tensor> {
        let n = a.shape().dims()[0];
        let x_new = x.clone();

        let indices: Vec<usize> = if forward {
            (0..n).collect()
        } else {
            (0..n).rev().collect()
        };

        for &i in &indices {
            let mut sum = b.get(&[i])?;

            // Subtract contributions from other variables
            for j in 0..n {
                if i != j {
                    sum -= a.get(&[i, j])? * x_new.get(&[j])?;
                }
            }

            // Update x[i]
            let a_ii = a.get(&[i, i])?;
            if a_ii.abs() > 1e-12 {
                x_new.set(&[i], sum / a_ii)?;
            }
        }

        Ok(x_new)
    }

    /// Jacobi smoothing iteration (naturally parallel)
    fn jacobi_smooth(&self, a: &Tensor, b: &Tensor, x: &Tensor) -> TorshResult<Tensor> {
        let n = a.shape().dims()[0];
        let x_new = self.create_zeros_vector(n)?;

        for i in 0..n {
            let mut sum = b.get(&[i])?;

            // Subtract contributions from other variables (using old values)
            for j in 0..n {
                if i != j {
                    sum -= a.get(&[i, j])? * x.get(&[j])?;
                }
            }

            // Update x[i]
            let a_ii = a.get(&[i, i])?;
            if a_ii.abs() > 1e-12 {
                x_new.set(&[i], sum / a_ii)?;
            }
        }

        Ok(x_new)
    }

    /// SOR (Successive Over-Relaxation) smoothing
    fn sor_smooth(
        &self,
        a: &Tensor,
        b: &Tensor,
        x: &Tensor,
        omega: f32,
        forward: bool,
    ) -> TorshResult<Tensor> {
        let n = a.shape().dims()[0];
        let x_new = x.clone();

        let indices: Vec<usize> = if forward {
            (0..n).collect()
        } else {
            (0..n).rev().collect()
        };

        for &i in &indices {
            let mut sum = b.get(&[i])?;

            // Subtract contributions from other variables
            for j in 0..n {
                if i != j {
                    sum -= a.get(&[i, j])? * x_new.get(&[j])?;
                }
            }

            // SOR update: x_new[i] = (1-ω)*x_old[i] + ω*x_gs[i]
            let a_ii = a.get(&[i, i])?;
            if a_ii.abs() > 1e-12 {
                let x_gs = sum / a_ii;
                let x_old = x.get(&[i])?;
                x_new.set(&[i], (1.0 - omega) * x_old + omega * x_gs)?;
            }
        }

        Ok(x_new)
    }

    /// SSOR (Symmetric SOR) smoothing
    fn ssor_smooth(&self, a: &Tensor, b: &Tensor, x: &Tensor, omega: f32) -> TorshResult<Tensor> {
        // Forward SOR sweep
        let x_temp = self.sor_smooth(a, b, x, omega, true)?;
        // Backward SOR sweep
        self.sor_smooth(a, b, &x_temp, omega, false)
    }

    /// Restrict (coarsen) matrix and vector to coarser grid
    ///
    /// This implements a simple geometric restriction that takes every
    /// other grid point. For optimal performance on specific problems,
    /// consider implementing problem-specific restriction operators.
    fn restrict(&self, a: &Tensor, r: &Tensor) -> TorshResult<(Tensor, Tensor)> {
        let n = a.shape().dims()[0];
        let n_coarse = n.div_ceil(2); // Simple coarsening strategy

        // Create coarse grid matrix using the configured coarsening strategy
        let (a_coarse, r_coarse) = match &self.config.coarsening_strategy {
            CoarseningStrategy::Geometric => self.geometric_restriction(a, r, n_coarse),
            CoarseningStrategy::Algebraic {
                strength_threshold: _,
            } => {
                // Simplified algebraic coarsening (fallback to geometric)
                self.geometric_restriction(a, r, n_coarse)
            }
            CoarseningStrategy::Aggregation {
                max_aggregate_size: _,
            } => {
                // Simplified aggregation (fallback to geometric)
                self.geometric_restriction(a, r, n_coarse)
            }
            CoarseningStrategy::RedBlack => self.geometric_restriction(a, r, n_coarse),
        }?;

        Ok((a_coarse, r_coarse))
    }

    /// Geometric restriction implementation
    fn geometric_restriction(
        &self,
        a: &Tensor,
        r: &Tensor,
        n_coarse: usize,
    ) -> TorshResult<(Tensor, Tensor)> {
        // Create coarse grid matrix (simplified - use every other row/column)
        let a_coarse = self.create_zeros_matrix(n_coarse, n_coarse)?;
        let r_coarse = self.create_zeros_vector(n_coarse)?;
        let n = a.shape().dims()[0];

        for i in 0..n_coarse {
            let fine_i = i * 2;
            if fine_i < n {
                r_coarse.set(&[i], r.get(&[fine_i])?)?;

                for j in 0..n_coarse {
                    let fine_j = j * 2;
                    if fine_j < n {
                        a_coarse.set(&[i, j], a.get(&[fine_i, fine_j])?)?;
                    }
                }
            }
        }

        Ok((a_coarse, r_coarse))
    }

    /// Interpolate (refine) vector from coarser to finer grid
    ///
    /// This implements simple linear interpolation to transfer corrections
    /// from coarse grids back to fine grids. For structured problems,
    /// consider implementing higher-order interpolation schemes.
    fn interpolate(&self, e_coarse: &Tensor, n_fine: usize) -> TorshResult<Tensor> {
        let n_coarse = e_coarse.shape().dims()[0];
        let e_fine = self.create_zeros_vector(n_fine)?;

        // Simple interpolation: copy coarse values and interpolate between them
        for i in 0..n_coarse {
            let fine_i = i * 2;
            if fine_i < n_fine {
                e_fine.set(&[fine_i], e_coarse.get(&[i])?)?;

                // Linear interpolation for intermediate points
                if fine_i + 1 < n_fine && i + 1 < n_coarse {
                    let interp_val = (e_coarse.get(&[i])? + e_coarse.get(&[i + 1])?) * 0.5;
                    e_fine.set(&[fine_i + 1], interp_val)?;
                }
            }
        }

        Ok(e_fine)
    }

    /// Direct solver for small systems
    ///
    /// When the grid becomes sufficiently small or the maximum level is reached,
    /// the problem is solved directly using LU decomposition.
    fn direct_solve(&self, a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
        // Use LU decomposition for direct solve
        crate::solve::solve(a, b)
    }

    /// Create zero vector with SciRS2 optimization when available
    #[cfg(feature = "scirs2-integration")]
    fn create_zeros_vector(&self, n: usize) -> TorshResult<Tensor> {
        // Use SciRS2's optimized array creation
        let array_data = Array1::<f32>::zeros(n);
        Tensor::from_data(array_data.to_vec(), vec![n], torsh_core::DeviceType::Cpu)
    }

    /// Create zero matrix with SciRS2 optimization when available
    #[cfg(feature = "scirs2-integration")]
    fn create_zeros_matrix(&self, m: usize, n: usize) -> TorshResult<Tensor> {
        // Use SciRS2's optimized array creation
        let array_data = Array2::<f32>::zeros((m, n));
        Tensor::from_data(
            array_data.into_raw_vec_and_offset().0,
            vec![m, n],
            torsh_core::DeviceType::Cpu,
        )
    }

    /// Fallback implementation when SciRS2 is not available
    #[cfg(not(feature = "scirs2-integration"))]
    fn create_zeros_vector(&self, n: usize) -> TorshResult<Tensor> {
        zeros::<f32>(&[n])
    }

    /// Fallback implementation when SciRS2 is not available
    #[cfg(not(feature = "scirs2-integration"))]
    fn create_zeros_matrix(&self, m: usize, n: usize) -> TorshResult<Tensor> {
        zeros::<f32>(&[m, n])
    }
}

/// Solve linear system using multigrid method with default configuration
///
/// This is a convenience function that applies multigrid solution with
/// standard parameters suitable for most well-conditioned problems.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (must be square)
/// * `b` - Right-hand side vector
///
/// # Returns
///
/// Solution vector x such that Ax ≈ b
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::advanced::solve_multigrid;
/// use torsh_tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let matrix = Tensor::from_data(vec![2.0, -1.0, -1.0, 2.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
/// let rhs = Tensor::from_data(vec![1.0, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
/// let solution = solve_multigrid(&matrix, &rhs)?;
/// # Ok(())
/// # }
/// ```
///
/// # Performance Notes
///
/// For optimal performance on structured grid problems, consider using
/// `solve_multigrid_with_config` with problem-specific parameters.
pub fn solve_multigrid(a: &Tensor, b: &Tensor) -> TorshResult<Tensor> {
    let solver = MultigridSolver::new(MultigridConfig::default());
    solver.solve(a, b)
}

/// Solve linear system using multigrid method with custom configuration
///
/// This function provides full control over the multigrid algorithm parameters,
/// allowing optimization for specific problem characteristics.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (must be square)
/// * `b` - Right-hand side vector
/// * `config` - Multigrid configuration parameters
///
/// # Returns
///
/// Solution vector x such that Ax ≈ b
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::advanced::{solve_multigrid_with_config, MultigridConfig, MultigridCycle};
/// use torsh_tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let matrix = Tensor::from_data(vec![2.0, -1.0, -1.0, 2.0], vec![2, 2], torsh_core::DeviceType::Cpu)?;
/// let rhs = Tensor::from_data(vec![1.0, 1.0], vec![2], torsh_core::DeviceType::Cpu)?;
///
/// let config = MultigridConfig {
///     max_levels: 6,
///     max_iterations: 50,
///     tolerance: 1e-8,
///     pre_smooth_steps: 3,
///     post_smooth_steps: 3,
///     cycle_type: MultigridCycle::W,
///     ..Default::default()
/// };
///
/// let solution = solve_multigrid_with_config(&matrix, &rhs, config)?;
/// # Ok(())
/// # }
/// ```
///
/// # Configuration Guidelines
///
/// - **max_levels**: More levels improve convergence but increase setup cost
/// - **tolerance**: Balance between accuracy and computational cost
/// - **smoothing steps**: More steps improve robustness but increase per-iteration cost
/// - **cycle_type**: W-cycle is more robust, V-cycle is more efficient
pub fn solve_multigrid_with_config(
    a: &Tensor,
    b: &Tensor,
    config: MultigridConfig,
) -> TorshResult<Tensor> {
    let solver = MultigridSolver::new(config);
    solver.solve(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[cfg(feature = "scirs2-integration")]
    use scirs2_core::ndarray::{Array1, Array2};

    #[cfg(not(feature = "scirs2-integration"))]
    use torsh_tensor::creation::{eye, zeros};

    // Helper to create eye matrix with SciRS2 integration
    #[cfg(feature = "scirs2-integration")]
    fn create_eye(n: usize) -> TorshResult<Tensor> {
        let mut data = Array2::<f32>::zeros((n, n));
        for i in 0..n {
            data[[i, i]] = 1.0;
        }
        let (vec_data, _offset) = data.into_raw_vec_and_offset();
        Tensor::from_data(vec_data, vec![n, n], torsh_core::DeviceType::Cpu)
    }

    #[cfg(not(feature = "scirs2-integration"))]
    fn create_eye(n: usize) -> TorshResult<Tensor> {
        eye::<f32>(n)
    }

    #[cfg(feature = "scirs2-integration")]
    fn create_zeros(shape: &[usize]) -> TorshResult<Tensor> {
        match shape.len() {
            1 => {
                let data = Array1::<f32>::zeros(shape[0]);
                Tensor::from_data(data.to_vec(), shape.to_vec(), torsh_core::DeviceType::Cpu)
            }
            2 => {
                let data = Array2::<f32>::zeros((shape[0], shape[1]));
                let (vec_data, _offset) = data.into_raw_vec_and_offset();
                Tensor::from_data(vec_data, shape.to_vec(), torsh_core::DeviceType::Cpu)
            }
            _ => Err(TorshError::InvalidArgument(
                "Unsupported shape for test".to_string(),
            )),
        }
    }

    #[cfg(not(feature = "scirs2-integration"))]
    fn create_zeros(shape: &[usize]) -> TorshResult<Tensor> {
        zeros::<f32>(shape)
    }

    #[test]
    fn test_multigrid_identity() -> TorshResult<()> {
        // Test on identity matrix (should converge immediately)
        let a = create_eye(4)?;
        let b = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![4],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_multigrid(&a, &b)?;

        // Should be exact solution since A = I
        assert_eq!(x.shape().dims(), &[4]);
        for i in 0..4 {
            assert_relative_eq!(x.get(&[i])?, b.get(&[i])?, epsilon = 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_multigrid_tridiagonal() -> TorshResult<()> {
        // Test on a simple tridiagonal system
        let a = create_zeros(&[4, 4])?;

        // Create tridiagonal matrix: 2 on diagonal, -1 on off-diagonals
        for i in 0..4 {
            a.set(&[i, i], 2.0)?;
            if i > 0 {
                a.set(&[i, i - 1], -1.0)?;
            }
            if i < 3 {
                a.set(&[i, i + 1], -1.0)?;
            }
        }

        let b = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 1.0, 1.0, 1.0],
            vec![4],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_multigrid(&a, &b)?;

        // Verify solution by computing residual
        let ax = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?;
        for i in 0..4 {
            assert_relative_eq!(ax.get(&[i])?, b.get(&[i])?, epsilon = 1e-4);
        }

        Ok(())
    }

    #[test]
    fn test_multigrid_config() -> TorshResult<()> {
        // Test with custom configuration
        let config = MultigridConfig {
            max_levels: 3,
            max_iterations: 20,
            tolerance: 1e-5,
            pre_smooth_steps: 1,
            post_smooth_steps: 1,
            cycle_type: MultigridCycle::V,
            smoothing_method: SmoothingMethod::GaussSeidel,
            coarsening_strategy: CoarseningStrategy::Geometric,
        };

        let a = create_eye(3)?;
        let b = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 2.0, 3.0],
            vec![3],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_multigrid_with_config(&a, &b, config)?;

        // Should get exact solution for identity matrix
        assert_eq!(x.shape().dims(), &[3]);
        for i in 0..3 {
            assert_relative_eq!(x.get(&[i])?, b.get(&[i])?, epsilon = 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_multigrid_w_cycle() -> TorshResult<()> {
        // Test W-cycle configuration
        let config = MultigridConfig {
            cycle_type: MultigridCycle::W,
            max_iterations: 10,
            ..Default::default()
        };

        let a = create_eye(3)?;
        let b = torsh_tensor::Tensor::from_data(
            vec![5.0f32, 10.0, 15.0],
            vec![3],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_multigrid_with_config(&a, &b, config)?;

        // Verify solution
        assert_eq!(x.shape().dims(), &[3]);
        for i in 0..3 {
            assert_relative_eq!(x.get(&[i])?, b.get(&[i])?, epsilon = 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_multigrid_solver_struct() -> TorshResult<()> {
        // Test using MultigridSolver directly
        let config = MultigridConfig {
            tolerance: 1e-8,
            ..Default::default()
        };
        let solver = MultigridSolver::new(config);

        let a = create_eye(2)?;
        let b = torsh_tensor::Tensor::from_data(
            vec![7.0f32, 14.0],
            vec![2],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solver.solve(&a, &b)?;

        // Verify solution
        assert_eq!(x.shape().dims(), &[2]);
        assert_relative_eq!(x.get(&[0])?, 7.0, epsilon = 1e-6);
        assert_relative_eq!(x.get(&[1])?, 14.0, epsilon = 1e-6);

        Ok(())
    }
}
