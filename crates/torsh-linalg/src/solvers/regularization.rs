//! Regularization techniques for ill-conditioned linear systems
//!
//! This module provides specialized solvers for linear systems that are ill-conditioned
//! or singular. Regularization techniques add constraints or modify the problem to
//! obtain stable, meaningful solutions even when the coefficient matrix is nearly singular.
//!
//! ## Available Methods
//!
//! ### Tikhonov Regularization (Ridge Regression)
//! - **Function**: [`solve_tikhonov`]
//! - **Problem**: Solves `min ||Ax - b||² + λ||x||²`
//! - **Use case**: When you want to minimize the norm of the solution vector
//! - **Advantages**: Simple, computationally efficient, always produces a solution
//! - **Disadvantages**: May over-smooth the solution
//!
//! ### Truncated SVD Regularization
//! - **Function**: [`solve_truncated_svd`]
//! - **Problem**: Solves `Ax = b` using `A ≈ U_k Σ_k V_k^T` where `k < rank(A)`
//! - **Use case**: When you want to filter out components corresponding to small singular values
//! - **Advantages**: Preserves the most important components of the solution
//! - **Disadvantages**: More computationally expensive (requires SVD)
//!
//! ### Damped Least Squares (Levenberg-Marquardt style)
//! - **Function**: [`solve_damped_least_squares`]
//! - **Problem**: Solves `min ||Ax - b||² + λ||D(x - x₀)||²`
//! - **Use case**: When you have prior information about the solution or want variable-specific damping
//! - **Advantages**: Flexible damping per variable, incorporates prior estimates
//! - **Disadvantages**: More parameters to tune
//!
//! ## When to Use Regularization
//!
//! Regularization is essential when dealing with:
//! - Ill-conditioned matrices (high condition number)
//! - Singular or near-singular matrices
//! - Overdetermined systems with noise in the data
//! - Underdetermined systems (more unknowns than equations)
//! - Inverse problems where the solution needs to be smooth or satisfy certain constraints
//!
//! ## Choosing the Right Method
//!
//! - **Tikhonov**: Start here for most problems. Simple and effective.
//! - **Truncated SVD**: Use when you want to understand which components contribute most to the solution.
//! - **Damped Least Squares**: Use when you have prior knowledge about the solution or need variable-specific regularization.
//!
//! ## Examples
//!
//! ```rust
//! use torsh_linalg::solvers::regularization::*;
//! use torsh_tensor::Tensor;
//!
//! # fn example() -> torsh_linalg::TorshResult<()> {
//! // Create an ill-conditioned system
//! let a = Tensor::from_data(vec![1.0, 1.0, 1.0, 1.001], vec![2, 2],
//!                          torsh_core::DeviceType::Cpu)?;
//! let b = Tensor::from_data(vec![2.0, 2.001], vec![2],
//!                          torsh_core::DeviceType::Cpu)?;
//!
//! // Solve with Tikhonov regularization
//! let x1 = solve_tikhonov(&a, &b, 0.01)?;
//!
//! // Solve with truncated SVD
//! let x2 = solve_truncated_svd(&a, &b, Some(1e-6))?;
//!
//! // Solve with damped least squares
//! let x3 = solve_damped_least_squares(&a, &b, 0.01, None, None)?;
//! # Ok(())
//! # }
//! ```

use crate::solve::solve; // Import the basic solve function
use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Tikhonov regularization (Ridge regression) for ill-conditioned systems
///
/// Solves the regularized least squares problem:
/// ```text
/// min ||Ax - b||² + λ||x||²
/// ```
///
/// This is equivalent to solving the augmented normal equations:
/// ```text
/// (A^T A + λI) x = A^T b
/// ```
///
/// # Arguments
///
/// * `a` - Coefficient matrix (m × n)
/// * `b` - Right-hand side vector (length m)
/// * `lambda` - Regularization parameter λ > 0 (controls the trade-off between fitting the data and regularization)
///
/// # Returns
///
/// The regularized solution x that minimizes the penalized residual.
///
/// # Mathematical Background
///
/// Tikhonov regularization adds a penalty term λ||x||² to the least squares objective.
/// This has the effect of:
/// - Stabilizing the solution for ill-conditioned matrices
/// - Producing solutions with smaller norms
/// - Trading off between data fitting and solution smoothness
///
/// The regularization parameter λ controls this trade-off:
/// - Large λ: More regularization, smoother solutions, may underfit
/// - Small λ: Less regularization, may overfit or be unstable
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::regularization::solve_tikhonov;
/// use torsh_tensor::Tensor;
///
/// # fn example() -> torsh_linalg::TorshResult<()> {
/// let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2],
///                          torsh_core::DeviceType::Cpu)?;
/// let b = Tensor::from_data(vec![5.0, 6.0], vec![2],
///                          torsh_core::DeviceType::Cpu)?;
/// let x = solve_tikhonov(&a, &b, 0.1)?;
/// # Ok(())
/// # }
/// ```
pub fn solve_tikhonov(a: &Tensor, b: &Tensor, lambda: f32) -> TorshResult<Tensor> {
    if a.shape().ndim() != 2 || b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Tikhonov regularization requires 2D matrix A and 1D vector b".to_string(),
        ));
    }

    if lambda <= 0.0 {
        return Err(TorshError::InvalidArgument(
            "Regularization parameter λ must be positive".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
    let b_len = b.shape().dims()[0];

    if m != b_len {
        return Err(TorshError::InvalidArgument(
            "Dimension mismatch between A and b".to_string(),
        ));
    }

    // Compute A^T
    let at = a.transpose(-2, -1)?;

    // Compute A^T * A
    let ata = at.matmul(a)?;

    // Add regularization: A^T A + λI
    let regularized = ata;
    for i in 0..n {
        let old_val = regularized.get(&[i, i])?;
        regularized.set(&[i, i], old_val + lambda)?;
    }

    // Compute A^T * b
    let atb = at.matmul(&b.unsqueeze(1)?)?.squeeze(1)?;

    // Solve regularized system
    solve(&regularized, &atb)
}

/// Truncated SVD regularization for ill-conditioned systems
///
/// Solves Ax = b using truncated SVD: A ≈ U_k Σ_k V_k^T where k < rank(A).
/// This effectively filters out small singular values that contribute to instability.
///
/// # Arguments
///
/// * `a` - Coefficient matrix (m × n)
/// * `b` - Right-hand side vector (length m)
/// * `rank_threshold` - Threshold for truncation (singular values below this are ignored)
///   If None, defaults to 1e-12
///
/// # Returns
///
/// The truncated SVD solution that uses only the most significant singular values.
///
/// # Mathematical Background
///
/// The SVD decomposition gives us A = UΣV^T where:
/// - U contains the left singular vectors (m × m)
/// - Σ contains the singular values in descending order (min(m,n) × min(m,n))
/// - V^T contains the right singular vectors (n × n)
///
/// The solution is computed as:
/// ```text
/// x = V_k Σ_k^(-1) U_k^T b
/// ```
/// where the subscript k indicates using only the first k components
/// corresponding to singular values above the threshold.
///
/// # Advantages
///
/// - Automatically determines the effective rank of the matrix
/// - Preserves the most important components of the solution
/// - Provides insight into which directions are well-determined
///
/// # Disadvantages
///
/// - More computationally expensive than Tikhonov regularization
/// - Choice of threshold can be problem-dependent
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::regularization::solve_truncated_svd;
/// use torsh_tensor::Tensor;
///
/// # fn example() -> torsh_linalg::TorshResult<()> {
/// let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2],
///                          torsh_core::DeviceType::Cpu)?;
/// let b = Tensor::from_data(vec![5.0, 6.0], vec![2],
///                          torsh_core::DeviceType::Cpu)?;
/// let x = solve_truncated_svd(&a, &b, Some(1e-10))?;
/// # Ok(())
/// # }
/// ```
pub fn solve_truncated_svd(
    a: &Tensor,
    b: &Tensor,
    rank_threshold: Option<f32>,
) -> TorshResult<Tensor> {
    use crate::decomposition::svd;

    if a.shape().ndim() != 2 || b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Truncated SVD requires 2D matrix A and 1D vector b".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
    let b_len = b.shape().dims()[0];

    if m != b_len {
        return Err(TorshError::InvalidArgument(
            "Dimension mismatch between A and b".to_string(),
        ));
    }

    // Compute SVD: A = U Σ V^T
    let (u, s, vt) = svd(a, false)?;

    let rank_threshold = rank_threshold.unwrap_or(1e-12);
    let min_dim = m.min(n);

    // Find truncation rank
    let mut truncated_rank = 0;
    for i in 0..min_dim {
        let sv = s.get(&[i])?;
        if sv >= rank_threshold {
            truncated_rank += 1;
        } else {
            break;
        }
    }

    if truncated_rank == 0 {
        return Err(TorshError::InvalidArgument(
            "All singular values below threshold".to_string(),
        ));
    }

    // Compute U^T * b
    let utb = u.transpose(-2, -1)?.matmul(&b.unsqueeze(1)?)?.squeeze(1)?;

    // Solve with truncated system: Σ_k y = U_k^T b, then x = V_k y
    let y = torsh_tensor::creation::zeros::<f32>(&[truncated_rank])?;
    for i in 0..truncated_rank {
        let sigma_i = s.get(&[i])?;
        let utb_i = utb.get(&[i])?;
        y.set(&[i], utb_i / sigma_i)?;
    }

    // Compute x = V_k * y (V_k is first truncated_rank rows of V^T transposed)
    let x = torsh_tensor::creation::zeros::<f32>(&[n])?;
    for j in 0..n {
        let mut sum = 0.0f32;
        for i in 0..truncated_rank {
            let v_ji = vt.get(&[i, j])?; // V^T[i,j] = V[j,i]
            let y_i = y.get(&[i])?;
            sum += v_ji * y_i;
        }
        x.set(&[j], sum)?;
    }

    Ok(x)
}

/// Damped least squares (Levenberg-Marquardt style regularization)
///
/// Solves the generalized regularized least squares problem:
/// ```text
/// min ||Ax - b||² + λ||D(x - x₀)||²
/// ```
/// where D is a diagonal damping matrix and x₀ is a prior estimate.
///
/// This is equivalent to solving:
/// ```text
/// (A^T A + λ D^T D) x = A^T b + λ D^T D x₀
/// ```
///
/// # Arguments
///
/// * `a` - Coefficient matrix (m × n)
/// * `b` - Right-hand side vector (length m)
/// * `lambda` - Damping parameter λ ≥ 0 (controls regularization strength)
/// * `damping_factors` - Diagonal damping matrix D (optional, defaults to identity)
/// * `x_prior` - Prior estimate x₀ (optional, defaults to zero)
///
/// # Returns
///
/// The damped least squares solution that minimizes the penalized residual.
///
/// # Mathematical Background
///
/// Damped least squares extends Tikhonov regularization by allowing:
/// 1. **Variable-specific damping**: Different regularization for each variable via D
/// 2. **Prior information**: Regularizing towards a prior estimate x₀ instead of zero
///
/// This is particularly useful in:
/// - Nonlinear optimization (Levenberg-Marquardt algorithm)
/// - Problems where some variables should be regularized more than others
/// - When you have prior knowledge about the expected solution
///
/// # Parameters
///
/// - **λ = 0**: Reduces to standard least squares
/// - **D = I, x₀ = 0**: Reduces to Tikhonov regularization
/// - **Large D_ii**: Variable x_i is heavily damped towards x₀_i
/// - **Small D_ii**: Variable x_i is lightly regularized
///
/// # Example
///
/// ```rust
/// use torsh_linalg::solvers::regularization::solve_damped_least_squares;
/// use torsh_tensor::Tensor;
///
/// # fn example() -> torsh_linalg::TorshResult<()> {
/// let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2],
///                          torsh_core::DeviceType::Cpu)?;
/// let b = Tensor::from_data(vec![5.0, 6.0], vec![2],
///                          torsh_core::DeviceType::Cpu)?;
///
/// // Basic usage (equivalent to Tikhonov)
/// let x1 = solve_damped_least_squares(&a, &b, 0.1, None, None)?;
///
/// // With custom damping factors
/// let damping = Tensor::from_data(vec![1.0, 2.0], vec![2],
///                                torsh_core::DeviceType::Cpu)?;
/// let x2 = solve_damped_least_squares(&a, &b, 0.1, Some(&damping), None)?;
///
/// // With prior estimate
/// let prior = Tensor::from_data(vec![1.0, 0.5], vec![2],
///                              torsh_core::DeviceType::Cpu)?;
/// let x3 = solve_damped_least_squares(&a, &b, 0.1, None, Some(&prior))?;
/// # Ok(())
/// # }
/// ```
pub fn solve_damped_least_squares(
    a: &Tensor,
    b: &Tensor,
    lambda: f32,
    damping_factors: Option<&Tensor>,
    x_prior: Option<&Tensor>,
) -> TorshResult<Tensor> {
    if a.shape().ndim() != 2 || b.shape().ndim() != 1 {
        return Err(TorshError::InvalidArgument(
            "Damped least squares requires 2D matrix A and 1D vector b".to_string(),
        ));
    }

    if lambda < 0.0 {
        return Err(TorshError::InvalidArgument(
            "Damping parameter λ must be non-negative".to_string(),
        ));
    }

    let (m, n) = (a.shape().dims()[0], a.shape().dims()[1]);
    let b_len = b.shape().dims()[0];

    if m != b_len {
        return Err(TorshError::InvalidArgument(
            "Dimension mismatch between A and b".to_string(),
        ));
    }

    // Default damping factors (identity)
    let default_damping = torsh_tensor::creation::ones::<f32>(&[n])?;
    let damping = damping_factors.unwrap_or(&default_damping);

    if damping.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(
            "Damping factors must have same length as number of variables".to_string(),
        ));
    }

    // Default prior (zero)
    let default_prior = torsh_tensor::creation::zeros::<f32>(&[n])?;
    let x_prior = x_prior.unwrap_or(&default_prior);

    if x_prior.shape().dims()[0] != n {
        return Err(TorshError::InvalidArgument(
            "Prior estimate must have same length as number of variables".to_string(),
        ));
    }

    // Compute A^T
    let at = a.transpose(-2, -1)?;

    // Compute A^T * A
    let ata = at.matmul(a)?;

    // Add damping: A^T A + λ D^T D
    let regularized = ata;
    for i in 0..n {
        let damping_i = damping.get(&[i])?;
        let old_val = regularized.get(&[i, i])?;
        regularized.set(&[i, i], old_val + lambda * damping_i * damping_i)?;
    }

    // Compute modified RHS: A^T b + λ D^T D x_prior
    let rhs = at.matmul(&b.unsqueeze(1)?)?.squeeze(1)?;
    for i in 0..n {
        let damping_i = damping.get(&[i])?;
        let prior_i = x_prior.get(&[i])?;
        let old_val = rhs.get(&[i])?;
        rhs.set(&[i], old_val + lambda * damping_i * damping_i * prior_i)?;
    }

    // Solve regularized system
    solve(&regularized, &rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::eye;

    #[test]
    fn test_solve_tikhonov() -> TorshResult<()> {
        // Create a simple 2x2 system that's moderately ill-conditioned
        let a = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 1.0, 1.0, 1.001],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let b = torsh_tensor::Tensor::from_data(
            vec![2.0f32, 2.001],
            vec![2],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_tikhonov(&a, &b, 0.1)?;

        // The regularized solution should be finite and reasonable
        assert!(x.get(&[0])?.is_finite());
        assert!(x.get(&[1])?.is_finite());

        // Check the solution satisfies the regularized system approximately
        let residual = a.matmul(&x.unsqueeze(1)?)?.squeeze(1)?.sub(&b)?;
        let residual_norm = residual.norm()?.item()?;
        assert!(residual_norm < 10.0); // Should be reasonably small

        Ok(())
    }

    #[test]
    fn test_solve_truncated_svd() -> TorshResult<()> {
        // Create a rank-deficient 3x2 system
        let a = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 2.0, 2.0, 4.0, 1.0, 2.0],
            vec![3, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let b = torsh_tensor::Tensor::from_data(
            vec![3.0f32, 6.0, 3.0],
            vec![3],
            torsh_core::DeviceType::Cpu,
        )?;

        let x = solve_truncated_svd(&a, &b, Some(0.1))?; // Lower threshold

        // The solution should be finite
        assert!(x.get(&[0])?.is_finite());
        assert!(x.get(&[1])?.is_finite());

        // Check dimensions
        assert_eq!(x.shape().dims()[0], 2);

        Ok(())
    }

    #[test]
    fn test_solve_damped_least_squares() -> TorshResult<()> {
        // Simple 2x2 system
        let a = eye::<f32>(2)?;
        let b = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            torsh_core::DeviceType::Cpu,
        )?;

        // Test with default parameters (should be close to [1, 2])
        let x1 = solve_damped_least_squares(&a, &b, 0.1, None, None)?;

        // Test with custom damping factors
        let damping = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            torsh_core::DeviceType::Cpu,
        )?;
        let x2 = solve_damped_least_squares(&a, &b, 0.1, Some(&damping), None)?;

        // Test with prior estimate
        let prior = torsh_tensor::Tensor::from_data(
            vec![0.5f32, 0.5],
            vec![2],
            torsh_core::DeviceType::Cpu,
        )?;
        let x3 = solve_damped_least_squares(&a, &b, 0.1, None, Some(&prior))?;

        // All solutions should be finite
        assert!(x1.get(&[0])?.is_finite() && x1.get(&[1])?.is_finite());
        assert!(x2.get(&[0])?.is_finite() && x2.get(&[1])?.is_finite());
        assert!(x3.get(&[0])?.is_finite() && x3.get(&[1])?.is_finite());

        Ok(())
    }

    #[test]
    fn test_tikhonov_parameter_validation() {
        let a = eye::<f32>(2).unwrap();
        let b = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        // Test invalid lambda
        assert!(solve_tikhonov(&a, &b, -0.1).is_err());
        assert!(solve_tikhonov(&a, &b, 0.0).is_err());
    }

    #[test]
    fn test_damped_least_squares_parameter_validation() {
        let a = eye::<f32>(2).unwrap();
        let b = torsh_tensor::Tensor::from_data(
            vec![1.0f32, 2.0],
            vec![2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        // Test invalid lambda
        assert!(solve_damped_least_squares(&a, &b, -0.1, None, None).is_err());

        // Test mismatched damping factors
        let wrong_damping =
            torsh_tensor::Tensor::from_data(vec![1.0f32], vec![1], torsh_core::DeviceType::Cpu)
                .unwrap();
        assert!(solve_damped_least_squares(&a, &b, 0.1, Some(&wrong_damping), None).is_err());

        // Test mismatched prior
        let wrong_prior =
            torsh_tensor::Tensor::from_data(vec![1.0f32], vec![1], torsh_core::DeviceType::Cpu)
                .unwrap();
        assert!(solve_damped_least_squares(&a, &b, 0.1, None, Some(&wrong_prior)).is_err());
    }
}
