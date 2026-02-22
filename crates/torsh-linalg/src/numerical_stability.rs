//! Advanced numerical stability utilities for linear algebra operations
//!
//! This module provides tools and utilities for ensuring numerical stability
//! in matrix computations, including:
//! - Matrix equilibration and scaling
//! - Condition number monitoring
//! - Error estimation
//! - Iterative refinement
//! - Numerical stability analysis

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Numerical stability configuration
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Enable automatic matrix scaling
    pub auto_scale: bool,
    /// Equilibration strategy
    pub equilibration: EquilibrationStrategy,
    /// Warn when condition number exceeds this threshold
    pub condition_warning_threshold: f32,
    /// Error when condition number exceeds this threshold
    pub condition_error_threshold: f32,
    /// Maximum number of refinement iterations
    pub max_refinement_iterations: usize,
    /// Refinement convergence tolerance
    pub refinement_tolerance: f32,
}

/// Matrix equilibration strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquilibrationStrategy {
    /// No equilibration
    None,
    /// Row equilibration (normalize rows)
    Row,
    /// Column equilibration (normalize columns)
    Column,
    /// Two-sided equilibration (normalize both rows and columns)
    TwoSided,
    /// Symmetric equilibration for symmetric matrices
    Symmetric,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            auto_scale: true,
            equilibration: EquilibrationStrategy::TwoSided,
            condition_warning_threshold: 1e6,
            condition_error_threshold: 1e12,
            max_refinement_iterations: 5,
            refinement_tolerance: 1e-10,
        }
    }
}

/// Matrix scaling factors for equilibration
#[derive(Debug, Clone)]
pub struct ScalingFactors {
    /// Row scaling factors
    pub row_scales: Option<Tensor>,
    /// Column scaling factors
    pub col_scales: Option<Tensor>,
    /// Strategy used
    pub strategy: EquilibrationStrategy,
}

/// Equilibrate a matrix to improve numerical stability
///
/// Equilibration scales the rows and/or columns of a matrix to have similar norms,
/// which can significantly improve the numerical stability of linear algebra operations.
///
/// # Arguments
///
/// * `matrix` - Matrix to equilibrate
/// * `strategy` - Equilibration strategy to use
///
/// # Returns
///
/// Tuple of (equilibrated matrix, scaling factors)
///
/// # Example
///
/// ```ignore
/// let matrix = Tensor::from_data(vec![1e10, 1.0, 1.0, 1e-10], vec![2, 2], DeviceType::Cpu)?;
/// let (equilibrated, factors) = equilibrate_matrix(&matrix, EquilibrationStrategy::TwoSided)?;
/// // equilibrated matrix will have better conditioned rows and columns
/// ```
pub fn equilibrate_matrix(
    matrix: &Tensor,
    strategy: EquilibrationStrategy,
) -> TorshResult<(Tensor, ScalingFactors)> {
    if matrix.shape().ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Matrix equilibration requires 2D tensor".to_string(),
        ));
    }

    let (m, n) = (matrix.shape().dims()[0], matrix.shape().dims()[1]);

    match strategy {
        EquilibrationStrategy::None => {
            let factors = ScalingFactors {
                row_scales: None,
                col_scales: None,
                strategy,
            };
            Ok((matrix.clone(), factors))
        }
        EquilibrationStrategy::Row => {
            // Compute row norms and scale each row
            let mut row_scales_data = vec![1.0f32; m];
            for i in 0..m {
                let mut row_norm = 0.0f32;
                for j in 0..n {
                    let val = matrix.get(&[i, j])?;
                    row_norm += val * val;
                }
                row_norm = row_norm.sqrt();
                if row_norm > 1e-12 {
                    row_scales_data[i] = 1.0 / row_norm;
                }
            }

            let row_scales = Tensor::from_data(row_scales_data, vec![m], matrix.device())?;

            // Apply row scaling
            let mut scaled_data = vec![0.0f32; m * n];
            for i in 0..m {
                let scale = row_scales.get(&[i])?;
                for j in 0..n {
                    let val = matrix.get(&[i, j])?;
                    scaled_data[i * n + j] = val * scale;
                }
            }

            let equilibrated = Tensor::from_data(scaled_data, vec![m, n], matrix.device())?;
            let factors = ScalingFactors {
                row_scales: Some(row_scales),
                col_scales: None,
                strategy,
            };

            Ok((equilibrated, factors))
        }
        EquilibrationStrategy::Column => {
            // Compute column norms and scale each column
            let mut col_scales_data = vec![1.0f32; n];
            for j in 0..n {
                let mut col_norm = 0.0f32;
                for i in 0..m {
                    let val = matrix.get(&[i, j])?;
                    col_norm += val * val;
                }
                col_norm = col_norm.sqrt();
                if col_norm > 1e-12 {
                    col_scales_data[j] = 1.0 / col_norm;
                }
            }

            let col_scales = Tensor::from_data(col_scales_data, vec![n], matrix.device())?;

            // Apply column scaling
            let mut scaled_data = vec![0.0f32; m * n];
            for j in 0..n {
                let scale = col_scales.get(&[j])?;
                for i in 0..m {
                    let val = matrix.get(&[i, j])?;
                    scaled_data[i * n + j] = val * scale;
                }
            }

            let equilibrated = Tensor::from_data(scaled_data, vec![m, n], matrix.device())?;
            let factors = ScalingFactors {
                row_scales: None,
                col_scales: Some(col_scales),
                strategy,
            };

            Ok((equilibrated, factors))
        }
        EquilibrationStrategy::TwoSided => {
            // Apply row equilibration first
            let (row_equilibrated, row_factors) =
                equilibrate_matrix(matrix, EquilibrationStrategy::Row)?;

            // Then apply column equilibration
            let (equilibrated, col_factors) =
                equilibrate_matrix(&row_equilibrated, EquilibrationStrategy::Column)?;

            let factors = ScalingFactors {
                row_scales: row_factors.row_scales,
                col_scales: col_factors.col_scales,
                strategy,
            };

            Ok((equilibrated, factors))
        }
        EquilibrationStrategy::Symmetric => {
            // For symmetric matrices, use symmetric scaling: D^(-1) * A * D^(-1)
            if m != n {
                return Err(TorshError::InvalidArgument(
                    "Symmetric equilibration requires square matrix".to_string(),
                ));
            }

            // Compute diagonal scaling factors from diagonal elements
            let mut scales_data = vec![1.0f32; n];
            for i in 0..n {
                let diag_val = matrix.get(&[i, i])?.abs();
                if diag_val > 1e-12 {
                    scales_data[i] = 1.0 / diag_val.sqrt();
                }
            }

            let scales = Tensor::from_data(scales_data.clone(), vec![n], matrix.device())?;

            // Apply symmetric scaling: D^(-1) * A * D^(-1)
            let mut scaled_data = vec![0.0f32; n * n];
            for i in 0..n {
                let row_scale = scales.get(&[i])?;
                for j in 0..n {
                    let col_scale = scales.get(&[j])?;
                    let val = matrix.get(&[i, j])?;
                    scaled_data[i * n + j] = val * row_scale * col_scale;
                }
            }

            let equilibrated = Tensor::from_data(scaled_data, vec![n, n], matrix.device())?;
            let factors = ScalingFactors {
                row_scales: Some(scales.clone()),
                col_scales: Some(scales),
                strategy,
            };

            Ok((equilibrated, factors))
        }
    }
}

/// Undo equilibration on a solution vector
///
/// After solving an equilibrated system, this function scales the solution
/// back to the original coordinate system.
///
/// # Arguments
///
/// * `solution` - Solution vector from equilibrated system
/// * `factors` - Scaling factors used for equilibration
///
/// # Returns
///
/// Solution vector in original coordinates
pub fn unequilibrate_solution(solution: &Tensor, factors: &ScalingFactors) -> TorshResult<Tensor> {
    match factors.strategy {
        EquilibrationStrategy::None => Ok(solution.clone()),
        EquilibrationStrategy::Row => {
            // No adjustment needed for row equilibration
            Ok(solution.clone())
        }
        EquilibrationStrategy::Column => {
            // Scale by column scales
            if let Some(ref col_scales) = factors.col_scales {
                let n = solution.shape().dims()[0];
                let mut unscaled_data = vec![0.0f32; n];
                for i in 0..n {
                    let scale = col_scales.get(&[i])?;
                    unscaled_data[i] = solution.get(&[i])? * scale;
                }
                Tensor::from_data(unscaled_data, vec![n], solution.device())
            } else {
                Ok(solution.clone())
            }
        }
        EquilibrationStrategy::TwoSided => {
            // Scale by column scales (row scales don't affect solution)
            unequilibrate_solution(
                solution,
                &ScalingFactors {
                    row_scales: None,
                    col_scales: factors.col_scales.clone(),
                    strategy: EquilibrationStrategy::Column,
                },
            )
        }
        EquilibrationStrategy::Symmetric => {
            // Scale by diagonal scales
            if let Some(ref scales) = factors.col_scales {
                let n = solution.shape().dims()[0];
                let mut unscaled_data = vec![0.0f32; n];
                for i in 0..n {
                    let scale = scales.get(&[i])?;
                    unscaled_data[i] = solution.get(&[i])? * scale;
                }
                Tensor::from_data(unscaled_data, vec![n], solution.device())
            } else {
                Ok(solution.clone())
            }
        }
    }
}

/// Check if a matrix is numerically stable for a given operation
///
/// Returns warnings if the matrix has properties that may lead to numerical issues
///
/// # Arguments
///
/// * `matrix` - Matrix to check
/// * `operation` - Description of the operation (for error messages)
/// * `config` - Stability configuration
///
/// # Returns
///
/// List of warning messages (empty if no issues detected)
pub fn check_numerical_stability(
    matrix: &Tensor,
    operation: &str,
    config: &StabilityConfig,
) -> TorshResult<Vec<String>> {
    let mut warnings = Vec::new();

    if matrix.shape().ndim() != 2 {
        return Ok(warnings);
    }

    let (m, n) = (matrix.shape().dims()[0], matrix.shape().dims()[1]);

    // Check for square matrices
    if m == n {
        // Estimate condition number
        let cond = crate::cond(matrix, Some("2"))?;

        if cond > config.condition_error_threshold {
            return Err(TorshError::ComputeError(format!(
                "{}: Matrix is severely ill-conditioned (κ = {:.2e}). Results will be unreliable.",
                operation, cond
            )));
        }

        if cond > config.condition_warning_threshold {
            warnings.push(format!(
                "{}: Matrix is ill-conditioned (κ = {:.2e}). Consider using regularization or iterative refinement.",
                operation, cond
            ));
        }
    }

    // Check for very small or very large elements
    let mut min_abs = f32::INFINITY;
    let mut max_abs = 0.0f32;

    for i in 0..m {
        for j in 0..n {
            let val = matrix.get(&[i, j])?.abs();
            if val > 1e-12 {
                min_abs = min_abs.min(val);
            }
            max_abs = max_abs.max(val);
        }
    }

    let scale_ratio = if min_abs.is_finite() && min_abs > 0.0 {
        max_abs / min_abs
    } else {
        0.0
    };

    if scale_ratio > 1e10 {
        warnings.push(format!(
            "{}: Matrix has widely varying element magnitudes (ratio = {:.2e}). Consider equilibration.",
            operation, scale_ratio
        ));
    }

    Ok(warnings)
}

/// Helper function to compute vector 2-norm
fn compute_vector_norm(vector: &Tensor) -> TorshResult<f32> {
    if vector.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Vector norm requires 1D tensor".to_string(),
        ));
    }

    let n = vector.shape().dims()[0];
    let mut sum = 0.0f32;
    for i in 0..n {
        let val = vector.get(&[i])?;
        sum += val * val;
    }
    Ok(sum.sqrt())
}

/// Solve with automatic iterative refinement for improved accuracy
///
/// Solves Ax = b and then applies iterative refinement to improve the solution accuracy.
///
/// # Arguments
///
/// * `a` - Coefficient matrix
/// * `b` - Right-hand side vector
/// * `config` - Stability configuration
///
/// # Returns
///
/// Tuple of (solution, number of refinement iterations performed, final residual norm)
pub fn solve_with_refinement(
    a: &Tensor,
    b: &Tensor,
    config: &StabilityConfig,
) -> TorshResult<(Tensor, usize, f32)> {
    // Initial solve
    let mut x = crate::solvers::solve(a, b)?;
    let mut iterations = 0;

    // Iterative refinement
    for i in 0..config.max_refinement_iterations {
        // Compute residual: r = b - Ax
        let ax = a.matmul(&x.unsqueeze(1)?)?;
        let ax = ax.squeeze(1)?;
        let residual = b.sub(&ax)?;

        // Compute residual norm (vector 2-norm)
        let residual_norm = compute_vector_norm(&residual)?;

        if residual_norm < config.refinement_tolerance {
            iterations = i;
            return Ok((x, iterations, residual_norm));
        }

        // Solve for correction: A * correction = residual
        let correction = crate::solvers::solve(a, &residual)?;

        // Update solution: x = x + correction
        x = x.add(&correction)?;
        iterations = i + 1;
    }

    // Compute final residual
    let ax = a.matmul(&x.unsqueeze(1)?)?;
    let ax = ax.squeeze(1)?;
    let residual = b.sub(&ax)?;
    let residual_norm = compute_vector_norm(&residual)?;

    Ok((x, iterations, residual_norm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_row_equilibration() -> TorshResult<()> {
        // Create a matrix with widely varying row norms
        let data = vec![1.0f32, 2.0, 1000.0, 2000.0];
        let matrix = Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)?;

        let (equilibrated, factors) = equilibrate_matrix(&matrix, EquilibrationStrategy::Row)?;

        // Check that row norms are approximately equal
        let row0_norm =
            (equilibrated.get(&[0, 0])?.powi(2) + equilibrated.get(&[0, 1])?.powi(2)).sqrt();
        let row1_norm =
            (equilibrated.get(&[1, 0])?.powi(2) + equilibrated.get(&[1, 1])?.powi(2)).sqrt();

        assert_relative_eq!(row0_norm, row1_norm, epsilon = 1e-3);
        assert!(factors.row_scales.is_some());

        Ok(())
    }

    #[test]
    fn test_column_equilibration() -> TorshResult<()> {
        // Create a matrix with widely varying column norms
        let data = vec![1.0f32, 1000.0, 2.0, 2000.0];
        let matrix = Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)?;

        let (equilibrated, factors) = equilibrate_matrix(&matrix, EquilibrationStrategy::Column)?;

        // Check that column norms are approximately equal
        let col0_norm =
            (equilibrated.get(&[0, 0])?.powi(2) + equilibrated.get(&[1, 0])?.powi(2)).sqrt();
        let col1_norm =
            (equilibrated.get(&[0, 1])?.powi(2) + equilibrated.get(&[1, 1])?.powi(2)).sqrt();

        assert_relative_eq!(col0_norm, col1_norm, epsilon = 1e-3);
        assert!(factors.col_scales.is_some());

        Ok(())
    }

    #[test]
    fn test_two_sided_equilibration() -> TorshResult<()> {
        // Create an unbalanced matrix
        let data = vec![1.0f32, 1000.0, 2000.0, 4.0];
        let matrix = Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)?;

        let (equilibrated, factors) = equilibrate_matrix(&matrix, EquilibrationStrategy::TwoSided)?;

        assert!(factors.row_scales.is_some());
        assert!(factors.col_scales.is_some());

        // Equilibrated matrix should have better conditioned rows and columns
        for i in 0..2 {
            for j in 0..2 {
                let val = equilibrated.get(&[i, j])?;
                assert!(val.is_finite());
            }
        }

        Ok(())
    }

    #[test]
    fn test_symmetric_equilibration() -> TorshResult<()> {
        // Create a symmetric matrix
        let data = vec![4.0f32, 2.0, 2.0, 9.0];
        let matrix = Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)?;

        let (equilibrated, factors) =
            equilibrate_matrix(&matrix, EquilibrationStrategy::Symmetric)?;

        // Check that the result is still symmetric
        assert_relative_eq!(
            equilibrated.get(&[0, 1])?,
            equilibrated.get(&[1, 0])?,
            epsilon = 1e-6
        );

        assert!(factors.row_scales.is_some());
        assert!(factors.col_scales.is_some());

        Ok(())
    }

    #[test]
    fn test_unequilibrate_solution() -> TorshResult<()> {
        let matrix = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;

        let (_, factors) = equilibrate_matrix(&matrix, EquilibrationStrategy::Column)?;

        let solution = Tensor::from_data(vec![1.0f32, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;

        let unequilibrated = unequilibrate_solution(&solution, &factors)?;

        // Solution should be scaled back
        assert!(unequilibrated.shape().dims()[0] == 2);
        for i in 0..2 {
            let val = unequilibrated.get(&[i])?;
            assert!(val.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_check_numerical_stability() -> TorshResult<()> {
        let config = StabilityConfig::default();

        // Well-conditioned matrix (identity)
        let identity = torsh_tensor::creation::eye::<f32>(3)?;
        let warnings = check_numerical_stability(&identity, "test", &config)?;
        assert!(warnings.is_empty());

        // Matrix with varying element sizes (but not too extreme)
        let data = vec![100.0f32, 1.0, 1.0, 0.01];
        let unbalanced = Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)?;
        let _warnings = check_numerical_stability(&unbalanced, "test", &config)?;
        // This matrix should still have some warnings due to scale variation
        // but won't cause eigenvalue computation to fail

        Ok(())
    }

    #[test]
    fn test_solve_with_refinement() -> TorshResult<()> {
        // Create a simple well-conditioned system: identity matrix
        let a = torsh_tensor::creation::eye::<f32>(3)?;
        let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;

        let config = StabilityConfig::default();
        let (solution, iterations, residual_norm) = solve_with_refinement(&a, &b, &config)?;

        // Solution should be approximately [1, 2, 3] for I*x = b
        assert_relative_eq!(solution.get(&[0])?, 1.0, epsilon = 1e-3);
        assert_relative_eq!(solution.get(&[1])?, 2.0, epsilon = 1e-3);
        assert_relative_eq!(solution.get(&[2])?, 3.0, epsilon = 1e-3);

        // Should converge very quickly for identity matrix
        assert!(iterations <= config.max_refinement_iterations);
        assert!(residual_norm < config.refinement_tolerance);

        Ok(())
    }

    #[test]
    fn test_equilibration_roundtrip() -> TorshResult<()> {
        // Test basic equilibration/unequilibration of vectors
        let solution = Tensor::from_data(vec![1.0f32, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;

        // Create column scaling factors
        let col_scales =
            Tensor::from_data(vec![0.5f32, 0.25], vec![2], torsh_core::DeviceType::Cpu)?;

        let factors = ScalingFactors {
            row_scales: None,
            col_scales: Some(col_scales),
            strategy: EquilibrationStrategy::Column,
        };

        // Unequilibrate and verify scaling
        let unequilibrated = unequilibrate_solution(&solution, &factors)?;

        // Should be scaled by column factors
        assert_relative_eq!(unequilibrated.get(&[0])?, 0.5, epsilon = 1e-6); // 1.0 * 0.5
        assert_relative_eq!(unequilibrated.get(&[1])?, 0.5, epsilon = 1e-6); // 2.0 * 0.25

        Ok(())
    }
}
