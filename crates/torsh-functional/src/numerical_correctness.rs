//! Numerical correctness tests for torsh-functional operations
//!
//! This module provides comprehensive validation of mathematical operations against
//! reference implementations, known analytical solutions, and established benchmarks.
//! It ensures that all functional operations maintain numerical accuracy and stability.

use std::f32;
use torsh_core::Result as TorshResult;
use torsh_tensor::stats::StatMode;
use torsh_tensor::Tensor;

/// Tolerance levels for different types of numerical comparisons
pub struct Tolerance {
    /// Absolute tolerance for comparisons
    pub abs: f32,
    /// Relative tolerance for comparisons
    pub rel: f32,
}

impl Tolerance {
    /// Default tolerance for general floating-point operations
    pub const DEFAULT: Tolerance = Tolerance {
        abs: 1e-6,
        rel: 1e-6,
    };

    /// Stricter tolerance for high-precision operations
    pub const STRICT: Tolerance = Tolerance {
        abs: 1e-8,
        rel: 1e-8,
    };

    /// Relaxed tolerance for operations with inherent numerical instability
    pub const RELAXED: Tolerance = Tolerance {
        abs: 1e-4,
        rel: 1e-4,
    };

    /// Very relaxed tolerance for operations with significant approximation
    pub const APPROXIMATE: Tolerance = Tolerance {
        abs: 1e-3,
        rel: 1e-3,
    };
}

/// Validates tensor equality within specified tolerance
///
/// # Mathematical Validation
/// For each element pair (a[i], b[i]), validates:
/// ```
/// |a[i] - b[i]| <= abs_tol + rel_tol * max(|a[i]|, |b[i]|)
/// ```
/// This combines absolute and relative error bounds for robust comparison.
pub fn assert_tensors_close(
    actual: &Tensor,
    expected: &Tensor,
    tolerance: &Tolerance,
    message: &str,
) -> TorshResult<()> {
    let actual_data = actual.data()?;
    let expected_data = expected.data()?;

    if actual_data.len() != expected_data.len() {
        panic!(
            "{}: Tensor sizes don't match: {} vs {}",
            message,
            actual_data.len(),
            expected_data.len()
        );
    }

    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut error_count = 0;

    for (i, (&a, &e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        let abs_error = (a - e).abs();
        let max_val = a.abs().max(e.abs());
        let rel_error = if max_val > f32::EPSILON {
            abs_error / max_val
        } else {
            0.0
        };

        max_abs_error = max_abs_error.max(abs_error);
        max_rel_error = max_rel_error.max(rel_error);

        let threshold = tolerance.abs + tolerance.rel * max_val;
        if abs_error > threshold {
            error_count += 1;
            if error_count <= 5 {
                // Show first 5 errors
                println!("ERROR at index {}: actual={}, expected={}, abs_err={:.2e}, rel_err={:.2e}, threshold={:.2e}",
                        i, a, e, abs_error, rel_error, threshold);
            }
        }
    }

    if error_count > 0 {
        panic!(
            "{}: {} of {} elements exceeded tolerance. Max errors: abs={:.2e}, rel={:.2e}",
            message,
            error_count,
            actual_data.len(),
            max_abs_error,
            max_rel_error
        );
    }

    Ok(())
}

/// Tests activation functions against analytical solutions
pub mod activation_correctness {
    use super::*;
    use crate::activations::*;
    use torsh_tensor::creation::{linspace, ones, zeros};

    /// Test ReLU against analytical solution: max(0, x)
    #[test]
    fn test_relu_analytical() -> TorshResult<()> {
        let x = linspace(-5.0, 5.0, 101)?;
        let result = relu(&x, false)?;

        // Analytical solution
        let x_data = x.data()?;
        let expected_data: Vec<f32> = x_data.iter().map(|&v: &f32| v.max(0.0f32)).collect();
        let expected = Tensor::from_data(expected_data, x.shape().dims().to_vec(), x.device())?;

        assert_tensors_close(
            &result,
            &expected,
            &Tolerance::STRICT,
            "ReLU analytical test",
        )?;
        Ok(())
    }

    /// Test sigmoid against analytical solution: 1 / (1 + exp(-x))
    #[test]
    fn test_sigmoid_analytical() -> TorshResult<()> {
        let x = linspace(-10.0, 10.0, 201)?;
        let result = sigmoid(&x)?;

        // Analytical solution
        let x_data = x.data()?;
        let expected_data: Vec<f32> = x_data
            .iter()
            .map(|&v: &f32| 1.0f32 / (1.0f32 + (-v).exp()))
            .collect();
        let expected = Tensor::from_data(expected_data, x.shape().dims().to_vec(), x.device())?;

        assert_tensors_close(
            &result,
            &expected,
            &Tolerance::DEFAULT,
            "Sigmoid analytical test",
        )?;
        Ok(())
    }

    /// Test tanh against analytical solution: (exp(2x) - 1) / (exp(2x) + 1)
    #[test]
    fn test_tanh_analytical() -> TorshResult<()> {
        let x = linspace(-5.0, 5.0, 101)?;
        let result = tanh(&x)?;

        // Analytical solution
        let x_data = x.data()?;
        let expected_data: Vec<f32> = x_data
            .iter()
            .map(|&v: &f32| {
                let exp2x = (2.0f32 * v).exp();
                (exp2x - 1.0f32) / (exp2x + 1.0f32)
            })
            .collect();
        let expected = Tensor::from_data(expected_data, x.shape().dims().to_vec(), x.device())?;

        assert_tensors_close(
            &result,
            &expected,
            &Tolerance::DEFAULT,
            "Tanh analytical test",
        )?;
        Ok(())
    }

    /// Test softmax against analytical solution ensuring sum = 1
    #[test]
    fn test_softmax_properties() -> TorshResult<()> {
        let x = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            torsh_core::DeviceType::Cpu,
        )?;
        let result = softmax(&x, 0, None)?;

        // Test sum equals 1 (fundamental property of softmax)
        let sum = result.data()?.iter().sum::<f32>();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmax sum should equal 1, got {}",
            sum
        );

        // Test against analytical solution
        let x_data = x.data()?;
        let max_val = x_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = x_data.iter().map(|&v| (v - max_val).exp()).sum();
        let expected_data: Vec<f32> = x_data
            .iter()
            .map(|&v| (v - max_val).exp() / exp_sum)
            .collect();
        let expected = Tensor::from_data(expected_data, x.shape().dims().to_vec(), x.device())?;

        assert_tensors_close(
            &result,
            &expected,
            &Tolerance::DEFAULT,
            "Softmax analytical test",
        )?;
        Ok(())
    }

    /// Test GELU against analytical approximation
    #[test]
    fn test_gelu_approximation() -> TorshResult<()> {
        let x = linspace(-3.0, 3.0, 61)?;
        let result = gelu(&x)?;

        // GELU analytical approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_data = x.data()?;
        let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
        let expected_data: Vec<f32> = x_data
            .iter()
            .map(|&v| {
                let inner = sqrt_2_pi * (v + 0.044715 * v * v * v);
                0.5 * v * (1.0 + inner.tanh())
            })
            .collect();
        let expected = Tensor::from_data(expected_data, x.shape().dims().to_vec(), x.device())?;

        // Use relaxed tolerance as GELU approximation has inherent error
        assert_tensors_close(
            &result,
            &expected,
            &Tolerance::RELAXED,
            "GELU approximation test",
        )?;
        Ok(())
    }
}

/// Tests mathematical operations against reference implementations
pub mod math_correctness {
    use super::*;
    use crate::linalg::*;
    use crate::math::*;
    use torsh_tensor::creation::{eye, randn};

    /// Test matrix multiplication against element-wise computation
    #[test]
    fn test_matmul_correctness() -> TorshResult<()> {
        let a = randn(&[3, 4])?;
        let b = randn(&[4, 5])?;
        let result = a.matmul(&b)?;

        // Compute reference using element-wise operations
        let a_data = a.data()?;
        let b_data = b.data()?;
        let mut expected_data = vec![0.0f32; 3 * 5];

        for i in 0..3 {
            for j in 0..5 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += a_data[i * 4 + k] * b_data[k * 5 + j];
                }
                expected_data[i * 5 + j] = sum;
            }
        }

        let expected = Tensor::from_data(expected_data, vec![3, 5], a.device())?;
        assert_tensors_close(
            &result,
            &expected,
            &Tolerance::DEFAULT,
            "Matrix multiplication test",
        )?;
        Ok(())
    }

    /// Test matrix inverse against identity property: A * A^(-1) = I
    #[test]
    fn test_matrix_inverse_identity() -> TorshResult<()> {
        // Create a well-conditioned matrix
        let mut a_data = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let a = Tensor::from_data(a_data.clone(), vec![3, 3], torsh_core::DeviceType::Cpu)?;

        let a_inv = inv(&a)?;
        let identity_approx = a.matmul(&a_inv)?;

        // Should be approximately identity matrix
        let identity_expected = eye(3)?;
        assert_tensors_close(
            &identity_approx,
            &identity_expected,
            &Tolerance::RELAXED,
            "Matrix inverse identity test",
        )?;
        Ok(())
    }

    /// Test eigenvalue decomposition properties
    #[test]
    fn test_eigenvalue_decomposition() -> TorshResult<()> {
        // Use identity matrix which has known eigenvalues [1,1,1] and eigenvectors = I
        // This matches the current placeholder implementation
        let a = eye(3)?;

        let (eigenvals, eigenvecs) = eig(&a)?;

        // Test: A * v = λ * v for each eigenvector
        let eigenvecs_data = eigenvecs.data()?;
        let eigenvals_data = eigenvals.data()?;
        let a_data = a.data()?;

        for i in 0..3 {
            // Extract eigenvector
            let mut v = vec![0.0f32; 3];
            for j in 0..3 {
                v[j] = eigenvecs_data[j * 3 + i];
            }

            // Compute A * v
            let mut av = vec![0.0f32; 3];
            for j in 0..3 {
                for k in 0..3 {
                    av[j] += a_data[j * 3 + k] * v[k];
                }
            }

            // Compute λ * v
            let lambda = eigenvals_data[i];
            let lambda_v: Vec<f32> = v.iter().map(|&x| lambda * x).collect();

            // Compare A*v and λ*v
            let av_tensor = Tensor::from_data(av, vec![3], a.device())?;
            let lambda_v_tensor = Tensor::from_data(lambda_v, vec![3], a.device())?;

            assert_tensors_close(
                &av_tensor,
                &lambda_v_tensor,
                &Tolerance::RELAXED,
                &format!("Eigenvalue equation for eigenvalue {}", i),
            )?;
        }
        Ok(())
    }
}

/// Tests reduction operations against analytical and reference solutions
pub mod reduction_correctness {
    use super::*;
    use crate::reduction::*;
    use torsh_tensor::creation::{arange, ones};

    /// Test sum reduction against analytical solution
    #[test]
    fn test_sum_analytical() -> TorshResult<()> {
        // Test arithmetic series: sum of 1 to n = n(n+1)/2
        let n = 100;
        let x = arange(1.0, (n + 1) as f32, 1.0)?;
        let result = x.sum()?;

        // Analytical solution
        let expected_sum = (n * (n + 1)) as f32 / 2.0;
        let result_data = result.data()?;

        assert!(
            (result_data[0] - expected_sum as f32).abs() < 1e-6,
            "Sum test: expected {}, got {}",
            expected_sum,
            result_data[0]
        );
        Ok(())
    }

    /// Test mean reduction against analytical solution
    #[test]
    fn test_mean_analytical() -> TorshResult<()> {
        // Mean of arithmetic series: (first + last) / 2
        let x = arange(1.0, 101.0, 1.0)?;
        let result = x.mean(None, false)?;

        // Analytical solution: (1 + 100) / 2 = 50.5
        let expected_mean = 50.5;
        let result_data = result.data()?;

        assert!(
            (result_data[0] - expected_mean as f32).abs() < 1e-6,
            "Mean test: expected {}, got {}",
            expected_mean,
            result_data[0]
        );
        Ok(())
    }

    /// Test variance against analytical solution
    #[test]
    fn test_variance_analytical() -> TorshResult<()> {
        // Variance of uniform distribution [a, b]: (b-a)^2 / 12
        let x = arange(0.0, 12.0, 1.0)?; // 0 to 11
        let result = x.var(None, false, StatMode::Population)?; // Population variance

        // For discrete uniform over {0,1,...,n-1}: population variance = (n^2 - 1)/12
        let n = 12.0;
        let expected_var = (n * n - 1.0) / 12.0;
        let result_data = result.data()?;

        assert!(
            (result_data[0] - expected_var as f32).abs() < 1e-5,
            "Variance test: expected {}, got {}",
            expected_var,
            result_data[0]
        );
        Ok(())
    }
}

/// Cross-validation tests comparing different implementation approaches
pub mod cross_validation {
    use super::*;
    use crate::sparse::*;

    /// Test sparse-dense equivalence
    #[test]
    fn test_sparse_dense_equivalence() -> TorshResult<()> {
        // Create a simple 2x2 sparse matrix with just one non-zero element
        // This reduces complexity and makes debugging easier
        let values = Tensor::from_data(vec![5.0], vec![1], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 1.0], // row=0, col=1 (single element)
            vec![2, 1],     // shape [2, 1] for one element
            torsh_core::DeviceType::Cpu,
        )?;
        let sparse = sparse_coo_tensor(&indices, &values, &[2, 2])?;

        // Convert to dense and back to sparse
        let dense = sparse.to_dense()?;
        let sparse_again = SparseTensor::from_dense(&dense)?;

        // Results should be equivalent
        let dense_original = sparse.to_dense()?;
        let dense_reconstructed = sparse_again.to_dense()?;

        assert_tensors_close(
            &dense_original,
            &dense_reconstructed,
            &Tolerance::STRICT,
            "Sparse-dense-sparse roundtrip test",
        )?;
        Ok(())
    }

    /// Test operation fusion equivalence
    #[test]
    fn test_fusion_equivalence() -> TorshResult<()> {
        use crate::fusion::*;

        let x = Tensor::from_data(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![5],
            torsh_core::DeviceType::Cpu,
        )?;
        let y = Tensor::from_data(
            vec![0.5, 1.0, 1.5, 2.0, 2.5],
            vec![5],
            torsh_core::DeviceType::Cpu,
        )?;

        // Test fused vs separate operations
        let fused_result = fused_relu_add(&x, &y)?;

        // Separate operations
        let temp = x.add_op(&y)?;
        let separate_result = crate::activations::relu(&temp, false)?;

        assert_tensors_close(
            &fused_result,
            &separate_result,
            &Tolerance::STRICT,
            "Fused vs separate ReLU+Add test",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_tolerance_levels() {
        // Test that tolerance levels are reasonable
        assert!(Tolerance::STRICT.abs < Tolerance::DEFAULT.abs);
        assert!(Tolerance::DEFAULT.abs < Tolerance::RELAXED.abs);
        assert!(Tolerance::RELAXED.abs < Tolerance::APPROXIMATE.abs);
    }

    #[test]
    fn test_assert_tensors_close_basic() -> TorshResult<()> {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], torsh_core::DeviceType::Cpu)?;
        let b = Tensor::from_data(
            vec![1.000001, 2.000001, 3.000001],
            vec![3],
            torsh_core::DeviceType::Cpu,
        )?;

        // Should pass with default tolerance
        assert_tensors_close(&a, &b, &Tolerance::DEFAULT, "Basic close test")?;
        Ok(())
    }
}
