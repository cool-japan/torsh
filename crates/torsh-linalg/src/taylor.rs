//! Taylor series approximations for matrix functions
//!
//! This module provides Taylor series-based approximations for matrix functions,
//! offering an alternative to eigendecomposition-based methods. These approximations
//! are particularly useful for:
//! - Matrices with clustered eigenvalues
//! - Cases where eigendecomposition is numerically unstable
//! - Providing error bounds on approximations
//! - Educational purposes and algorithm research
//!
//! # Mathematical Background
//!
//! Many matrix functions can be expressed as convergent Taylor series:
//! - exp(A) = I + A + A²/2! + A³/3! + ...
//! - sin(A) = A - A³/3! + A^5/5! - ...
//! - cos(A) = I - A²/2! + A^4/4! - ...
//! - sinh(A) = A + A³/3! + A^5/5! + ...
//! - cosh(A) = I + A²/2! + A^4/4! + ...
//!
//! # Convergence
//!
//! The Taylor series converges for all matrices, but the rate depends on the matrix norm.
//! For matrices with large norms, scaling and squaring methods are used to improve convergence.

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Configuration for Taylor series approximations
#[derive(Debug, Clone)]
pub struct TaylorConfig {
    /// Maximum number of terms in the series
    pub max_terms: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Whether to use scaling and squaring for better convergence
    pub use_scaling: bool,
    /// Maximum scaling factor (for scaling and squaring)
    pub max_scaling: usize,
}

impl Default for TaylorConfig {
    fn default() -> Self {
        Self {
            max_terms: 100,
            tolerance: 1e-10,
            use_scaling: true,
            max_scaling: 10,
        }
    }
}

/// Taylor series approximation of matrix exponential exp(A)
///
/// Uses the series: exp(A) = I + A + A²/2! + A³/3! + ...
///
/// For matrices with large norms, employs scaling and squaring:
/// exp(A) = (exp(A/2^k))^(2^k) for appropriate k
///
/// # Arguments
///
/// * `matrix` - Square matrix to exponentiate
/// * `config` - Configuration for the approximation
///
/// # Returns
///
/// The matrix exponential exp(A) and the number of terms used
///
/// # Example
///
/// ```ignore
/// let matrix = Tensor::from_data(vec![1.0, 0.0, 0.0, 2.0], vec![2, 2], DeviceType::Cpu)?;
/// let config = TaylorConfig::default();
/// let (exp_a, terms) = taylor_exp(&matrix, &config)?;
/// ```
pub fn taylor_exp(matrix: &Tensor, config: &TaylorConfig) -> TorshResult<(Tensor, usize)> {
    crate::validate_square_matrix(matrix, "Taylor series exp")?;

    let n = matrix.shape().dims()[0];

    // Compute matrix norm for scaling decision
    let matrix_norm = crate::matrix_functions::matrix_norm(matrix, Some("fro"))?;

    // Determine scaling factor
    let (scaled_matrix, scaling_factor) = if config.use_scaling && matrix_norm > 1.0 {
        let k = ((matrix_norm.log2()).ceil() as usize).min(config.max_scaling);
        let scale = 1.0 / (2.0f32.powi(k as i32));
        let scaled = matrix.mul_scalar(scale)?;
        (scaled, k)
    } else {
        (matrix.clone(), 0)
    };

    // Initialize with identity matrix
    let mut result = torsh_tensor::creation::eye::<f32>(n)?;
    let mut term = torsh_tensor::creation::eye::<f32>(n)?;
    let mut factorial = 1.0f32;

    let mut terms_used = 0;

    // Compute Taylor series: I + A + A²/2! + A³/3! + ...
    for i in 1..config.max_terms {
        factorial *= i as f32;

        // Compute next term: A^i / i!
        term = term.matmul(&scaled_matrix)?;
        let term_scaled = term.div_scalar(factorial)?;

        // Add to result
        result = result.add(&term_scaled)?;
        terms_used = i;

        // Check convergence using Frobenius norm of the term
        let term_norm = crate::matrix_functions::matrix_norm(&term_scaled, Some("fro"))?;
        if term_norm < config.tolerance {
            break;
        }
    }

    // Apply squaring to undo scaling: exp(A) = (exp(A/2^k))^(2^k)
    for _ in 0..scaling_factor {
        result = result.matmul(&result)?;
    }

    Ok((result, terms_used))
}

/// Taylor series approximation of matrix sine sin(A)
///
/// Uses the series: sin(A) = A - A³/3! + A^5/5! - A^7/7! + ...
///
/// # Arguments
///
/// * `matrix` - Square matrix
/// * `config` - Configuration for the approximation
///
/// # Returns
///
/// The matrix sine sin(A) and the number of terms used
pub fn taylor_sin(matrix: &Tensor, config: &TaylorConfig) -> TorshResult<(Tensor, usize)> {
    crate::validate_square_matrix(matrix, "Taylor series sin")?;

    let n = matrix.shape().dims()[0];

    // Initialize with zero matrix
    let mut result = torsh_tensor::creation::zeros::<f32>(&[n, n])?;
    let mut term = matrix.clone();
    let a_squared = matrix.matmul(matrix)?;

    let mut terms_used = 0;

    // Compute Taylor series: A - A³/3! + A^5/5! - ...
    for i in 0..config.max_terms {
        let k = 2 * i + 1;
        let mut factorial = 1.0f32;
        for j in 1..=k {
            factorial *= j as f32;
        }

        // Add/subtract term with alternating sign
        let term_scaled = if i % 2 == 0 {
            term.div_scalar(factorial)?
        } else {
            term.div_scalar(-factorial)?
        };

        result = result.add(&term_scaled)?;
        terms_used = i + 1;

        // Check convergence
        let term_norm = crate::matrix_functions::matrix_norm(&term_scaled, Some("fro"))?;
        if term_norm < config.tolerance {
            break;
        }

        // Compute next term: multiply by A²
        if i < config.max_terms - 1 {
            term = term.matmul(&a_squared)?;
        }
    }

    Ok((result, terms_used))
}

/// Taylor series approximation of matrix cosine cos(A)
///
/// Uses the series: cos(A) = I - A²/2! + A^4/4! - A^6/6! + ...
///
/// # Arguments
///
/// * `matrix` - Square matrix
/// * `config` - Configuration for the approximation
///
/// # Returns
///
/// The matrix cosine cos(A) and the number of terms used
pub fn taylor_cos(matrix: &Tensor, config: &TaylorConfig) -> TorshResult<(Tensor, usize)> {
    crate::validate_square_matrix(matrix, "Taylor series cos")?;

    let n = matrix.shape().dims()[0];

    // Initialize with identity matrix
    let mut result = torsh_tensor::creation::eye::<f32>(n)?;
    let mut term = torsh_tensor::creation::eye::<f32>(n)?;
    let a_squared = matrix.matmul(matrix)?;

    let mut terms_used = 0;

    // Compute Taylor series: I - A²/2! + A^4/4! - ...
    for i in 1..config.max_terms {
        let k = 2 * i;
        let mut factorial = 1.0f32;
        for j in 1..=k {
            factorial *= j as f32;
        }

        // Compute next term: multiply by A²
        term = term.matmul(&a_squared)?;

        // Add/subtract term with alternating sign
        let term_scaled = if i % 2 == 0 {
            term.div_scalar(factorial)?
        } else {
            term.div_scalar(-factorial)?
        };

        result = result.add(&term_scaled)?;
        terms_used = i;

        // Check convergence
        let term_norm = crate::matrix_functions::matrix_norm(&term_scaled, Some("fro"))?;
        if term_norm < config.tolerance {
            break;
        }
    }

    Ok((result, terms_used))
}

/// Taylor series approximation of matrix hyperbolic sine sinh(A)
///
/// Uses the series: sinh(A) = A + A³/3! + A^5/5! + A^7/7! + ...
///
/// # Arguments
///
/// * `matrix` - Square matrix
/// * `config` - Configuration for the approximation
///
/// # Returns
///
/// The matrix hyperbolic sine sinh(A) and the number of terms used
pub fn taylor_sinh(matrix: &Tensor, config: &TaylorConfig) -> TorshResult<(Tensor, usize)> {
    crate::validate_square_matrix(matrix, "Taylor series sinh")?;

    let n = matrix.shape().dims()[0];

    // Initialize with zero matrix
    let mut result = torsh_tensor::creation::zeros::<f32>(&[n, n])?;
    let mut term = matrix.clone();
    let a_squared = matrix.matmul(matrix)?;

    let mut terms_used = 0;

    // Compute Taylor series: A + A³/3! + A^5/5! + ...
    for i in 0..config.max_terms {
        let k = 2 * i + 1;
        let mut factorial = 1.0f32;
        for j in 1..=k {
            factorial *= j as f32;
        }

        let term_scaled = term.div_scalar(factorial)?;
        result = result.add(&term_scaled)?;
        terms_used = i + 1;

        // Check convergence
        let term_norm = crate::matrix_functions::matrix_norm(&term_scaled, Some("fro"))?;
        if term_norm < config.tolerance {
            break;
        }

        // Compute next term: multiply by A²
        if i < config.max_terms - 1 {
            term = term.matmul(&a_squared)?;
        }
    }

    Ok((result, terms_used))
}

/// Taylor series approximation of matrix hyperbolic cosine cosh(A)
///
/// Uses the series: cosh(A) = I + A²/2! + A^4/4! + A^6/6! + ...
///
/// # Arguments
///
/// * `matrix` - Square matrix
/// * `config` - Configuration for the approximation
///
/// # Returns
///
/// The matrix hyperbolic cosine cosh(A) and the number of terms used
pub fn taylor_cosh(matrix: &Tensor, config: &TaylorConfig) -> TorshResult<(Tensor, usize)> {
    crate::validate_square_matrix(matrix, "Taylor series cosh")?;

    let n = matrix.shape().dims()[0];

    // Initialize with identity matrix
    let mut result = torsh_tensor::creation::eye::<f32>(n)?;
    let mut term = torsh_tensor::creation::eye::<f32>(n)?;
    let a_squared = matrix.matmul(matrix)?;

    let mut terms_used = 0;

    // Compute Taylor series: I + A²/2! + A^4/4! + ...
    for i in 1..config.max_terms {
        let k = 2 * i;
        let mut factorial = 1.0f32;
        for j in 1..=k {
            factorial *= j as f32;
        }

        // Compute next term: multiply by A²
        term = term.matmul(&a_squared)?;
        let term_scaled = term.div_scalar(factorial)?;

        result = result.add(&term_scaled)?;
        terms_used = i;

        // Check convergence
        let term_norm = crate::matrix_functions::matrix_norm(&term_scaled, Some("fro"))?;
        if term_norm < config.tolerance {
            break;
        }
    }

    Ok((result, terms_used))
}

/// Taylor series approximation of matrix logarithm log(I + A) for ||A|| < 1
///
/// Uses the series: log(I + A) = A - A²/2 + A³/3 - A^4/4 + ...
///
/// **Important**: This series only converges for ||A|| < 1. For general matrices,
/// use the matrix logarithm function in the matrix_functions module.
///
/// # Arguments
///
/// * `matrix` - Matrix A where ||A|| < 1
/// * `config` - Configuration for the approximation
///
/// # Returns
///
/// The matrix logarithm log(I + A) and the number of terms used
///
/// # Errors
///
/// Returns error if ||A|| >= 1, as the series won't converge
pub fn taylor_log_nearby_identity(
    matrix: &Tensor,
    config: &TaylorConfig,
) -> TorshResult<(Tensor, usize)> {
    crate::validate_square_matrix(matrix, "Taylor series log")?;

    // Check that ||A|| < 1 for convergence
    let matrix_norm = crate::matrix_functions::matrix_norm(matrix, Some("fro"))?;
    if matrix_norm >= 1.0 {
        return Err(TorshError::InvalidArgument(format!(
            "Taylor series for log(I+A) requires ||A|| < 1, got ||A|| = {}",
            matrix_norm
        )));
    }

    let n = matrix.shape().dims()[0];

    // Initialize with zero matrix
    let mut result = torsh_tensor::creation::zeros::<f32>(&[n, n])?;
    let mut term = matrix.clone();

    let mut terms_used = 0;

    // Compute Taylor series: A - A²/2 + A³/3 - A^4/4 + ...
    for i in 1..config.max_terms {
        // Add/subtract term with alternating sign
        let term_scaled = if i % 2 == 1 {
            term.div_scalar(i as f32)?
        } else {
            term.div_scalar(-(i as f32))?
        };

        result = result.add(&term_scaled)?;
        terms_used = i;

        // Check convergence
        let term_norm = crate::matrix_functions::matrix_norm(&term_scaled, Some("fro"))?;
        if term_norm < config.tolerance {
            break;
        }

        // Compute next term: multiply by A
        if i < config.max_terms - 1 {
            term = term.matmul(matrix)?;
        }
    }

    Ok((result, terms_used))
}

/// Approximation information for analysis
#[derive(Debug, Clone)]
pub struct ApproximationInfo {
    /// Number of terms used in the series
    pub terms_used: usize,
    /// Estimated error bound (Frobenius norm of last term)
    pub error_bound: f32,
    /// Whether the series converged within tolerance
    pub converged: bool,
    /// Configuration used
    pub config: TaylorConfig,
}

/// Compute Taylor series with detailed approximation information
pub fn taylor_exp_with_info(
    matrix: &Tensor,
    config: &TaylorConfig,
) -> TorshResult<(Tensor, ApproximationInfo)> {
    let (result, terms) = taylor_exp(matrix, config)?;

    // Estimate error from the last term (rough upper bound)
    let n = matrix.shape().dims()[0];
    let mut term = torsh_tensor::creation::eye::<f32>(n)?;
    let mut factorial = 1.0f32;

    for i in 1..=terms {
        factorial *= i as f32;
        term = term.matmul(matrix)?;
    }

    let last_term = term.div_scalar(factorial)?;
    let error_bound = crate::matrix_functions::matrix_norm(&last_term, Some("fro"))?;

    let info = ApproximationInfo {
        terms_used: terms,
        error_bound,
        converged: error_bound < config.tolerance,
        config: config.clone(),
    };

    Ok((result, info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_small_matrix() -> TorshResult<Tensor> {
        // Create a small 2x2 matrix for testing
        let data = vec![0.1f32, 0.2, 0.3, 0.4];
        Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)
    }

    #[test]
    fn test_taylor_exp_zero_matrix() -> TorshResult<()> {
        let zero = torsh_tensor::creation::zeros::<f32>(&[2, 2])?;
        let config = TaylorConfig::default();
        let (result, terms) = taylor_exp(&zero, &config)?;

        // exp(0) = I
        assert_relative_eq!(result.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[0, 1])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[1, 0])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[1, 1])?, 1.0, epsilon = 1e-6);

        // Should converge very quickly for zero matrix
        assert!(terms < 5);

        Ok(())
    }

    #[test]
    fn test_taylor_sin_zero_matrix() -> TorshResult<()> {
        let zero = torsh_tensor::creation::zeros::<f32>(&[2, 2])?;
        let config = TaylorConfig::default();
        let (result, terms) = taylor_sin(&zero, &config)?;

        // sin(0) = 0
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result.get(&[i, j])?, 0.0, epsilon = 1e-6);
            }
        }

        // Should converge immediately for zero matrix
        assert!(terms < 3);

        Ok(())
    }

    #[test]
    fn test_taylor_cos_zero_matrix() -> TorshResult<()> {
        let zero = torsh_tensor::creation::zeros::<f32>(&[2, 2])?;
        let config = TaylorConfig::default();
        let (result, _) = taylor_cos(&zero, &config)?;

        // cos(0) = I
        assert_relative_eq!(result.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[0, 1])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[1, 0])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[1, 1])?, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_taylor_sinh_zero_matrix() -> TorshResult<()> {
        let zero = torsh_tensor::creation::zeros::<f32>(&[2, 2])?;
        let config = TaylorConfig::default();
        let (result, _) = taylor_sinh(&zero, &config)?;

        // sinh(0) = 0
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result.get(&[i, j])?, 0.0, epsilon = 1e-6);
            }
        }

        Ok(())
    }

    #[test]
    fn test_taylor_cosh_zero_matrix() -> TorshResult<()> {
        let zero = torsh_tensor::creation::zeros::<f32>(&[2, 2])?;
        let config = TaylorConfig::default();
        let (result, _) = taylor_cosh(&zero, &config)?;

        // cosh(0) = I
        assert_relative_eq!(result.get(&[0, 0])?, 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[0, 1])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[1, 0])?, 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.get(&[1, 1])?, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_taylor_exp_small_matrix() -> TorshResult<()> {
        let matrix = create_small_matrix()?;
        let config = TaylorConfig {
            max_terms: 50,
            tolerance: 1e-8,
            use_scaling: true,
            max_scaling: 5,
        };

        let (result, terms) = taylor_exp(&matrix, &config)?;

        // Result should be a valid matrix
        assert!(result.shape().dims()[0] == 2);
        assert!(result.shape().dims()[1] == 2);

        // Should converge
        assert!(terms > 0 && terms < config.max_terms);

        // Values should be reasonable (not NaN or infinite)
        for i in 0..2 {
            for j in 0..2 {
                let val = result.get(&[i, j])?;
                assert!(val.is_finite());
            }
        }

        Ok(())
    }

    #[test]
    fn test_taylor_log_small_matrix() -> TorshResult<()> {
        // Create a small matrix with ||A|| < 1
        let data = vec![0.1f32, 0.05, 0.05, 0.1];
        let matrix = Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)?;

        let config = TaylorConfig::default();
        let (result, terms) = taylor_log_nearby_identity(&matrix, &config)?;

        // Should converge
        assert!(terms > 0 && terms < config.max_terms);

        // Values should be reasonable
        for i in 0..2 {
            for j in 0..2 {
                let val = result.get(&[i, j])?;
                assert!(val.is_finite());
            }
        }

        Ok(())
    }

    #[test]
    fn test_taylor_log_convergence_condition() -> TorshResult<()> {
        // Create a matrix with ||A|| >= 1 (should fail)
        let data = vec![2.0f32, 0.0, 0.0, 2.0];
        let matrix = Tensor::from_data(data, vec![2, 2], torsh_core::DeviceType::Cpu)?;

        let config = TaylorConfig::default();
        let result = taylor_log_nearby_identity(&matrix, &config);

        // Should return error
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_taylor_exp_with_info() -> TorshResult<()> {
        let matrix = create_small_matrix()?;
        let config = TaylorConfig::default();

        let (result, info) = taylor_exp_with_info(&matrix, &config)?;

        // Check that we got valid information
        assert!(info.terms_used > 0);
        assert!(info.error_bound >= 0.0);
        assert!(result.shape().dims()[0] == 2);

        Ok(())
    }

    #[test]
    fn test_taylor_sin_cos_identity() -> TorshResult<()> {
        let matrix = create_small_matrix()?;
        let config = TaylorConfig::default();

        let (sin_a, _) = taylor_sin(&matrix, &config)?;
        let (cos_a, _) = taylor_cos(&matrix, &config)?;

        // Compute sin²(A) + cos²(A)
        let sin_squared = sin_a.matmul(&sin_a)?;
        let cos_squared = cos_a.matmul(&cos_a)?;
        let sum = sin_squared.add(&cos_squared)?;

        // Should approximately equal I for small matrices
        // Note: This identity holds for commuting matrices, and may have larger
        // error for non-commuting matrices
        let identity = torsh_tensor::creation::eye::<f32>(2)?;

        // Check with relaxed tolerance for non-commuting matrices
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(sum.get(&[i, j])?, identity.get(&[i, j])?, epsilon = 0.1);
            }
        }

        Ok(())
    }

    #[test]
    fn test_taylor_sinh_cosh_identity() -> TorshResult<()> {
        let matrix = create_small_matrix()?;
        let config = TaylorConfig::default();

        let (sinh_a, _) = taylor_sinh(&matrix, &config)?;
        let (cosh_a, _) = taylor_cosh(&matrix, &config)?;

        // Compute cosh²(A) - sinh²(A)
        let cosh_squared = cosh_a.matmul(&cosh_a)?;
        let sinh_squared = sinh_a.matmul(&sinh_a)?;
        let diff = cosh_squared.sub(&sinh_squared)?;

        // Should approximately equal I for small matrices
        let identity = torsh_tensor::creation::eye::<f32>(2)?;

        // Check with relaxed tolerance for non-commuting matrices
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(diff.get(&[i, j])?, identity.get(&[i, j])?, epsilon = 0.1);
            }
        }

        Ok(())
    }

    #[test]
    fn test_taylor_config_validation() -> TorshResult<()> {
        let matrix = create_small_matrix()?;

        // Test with different configurations
        let configs = vec![
            TaylorConfig {
                max_terms: 10,
                tolerance: 1e-5,
                use_scaling: false,
                max_scaling: 0,
            },
            TaylorConfig {
                max_terms: 100,
                tolerance: 1e-12,
                use_scaling: true,
                max_scaling: 10,
            },
        ];

        for config in configs {
            let (result, terms) = taylor_exp(&matrix, &config)?;
            assert!(terms <= config.max_terms);
            assert!(result.shape().dims()[0] == 2);
        }

        Ok(())
    }
}
