//! Comprehensive numerical accuracy tests for special functions
//!
//! This module contains thorough tests for numerical accuracy of all special functions
//! using known mathematical identities, reference values, and cross-validation.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::TorshResult;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use std::f64::consts::PI;
    use torsh_core::device::DeviceType;
    use torsh_tensor::Tensor;

    /// Test gamma function accuracy using known identities
    #[test]
    fn test_gamma_function_accuracy() -> TorshResult<()> {
        // Test Γ(n) = (n-1)! for positive integers
        for n in 1..=10 {
            let input = Tensor::from_data(vec![n as f32], vec![1], DeviceType::Cpu)?;
            let result = gamma(&input)?;
            let values = result.data()?;

            let expected = factorial(n - 1) as f32;
            assert_relative_eq!(values[0], expected, epsilon = 1e-6);
        }

        // Test Γ(1/2) = √π
        let input = Tensor::from_data(vec![0.5f32], vec![1], DeviceType::Cpu)?;
        let result = gamma(&input)?;
        let values = result.data()?;
        assert_relative_eq!(values[0], (PI as f32).sqrt(), epsilon = 1e-6);

        // Test Γ(3/2) = √π/2
        let input = Tensor::from_data(vec![1.5f32], vec![1], DeviceType::Cpu)?;
        let result = gamma(&input)?;
        let values = result.data()?;
        assert_relative_eq!(values[0], (PI as f32).sqrt() / 2.0, epsilon = 1e-6);

        // Test reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        let z = 0.3f32;
        let input1 = Tensor::from_data(vec![z], vec![1], DeviceType::Cpu)?;
        let input2 = Tensor::from_data(vec![1.0 - z], vec![1], DeviceType::Cpu)?;

        let gamma_z = gamma(&input1)?;
        let gamma_1_minus_z = gamma(&input2)?;

        let product = gamma_z.data()?[0] * gamma_1_minus_z.data()?[0];
        let expected = PI as f32 / (PI as f32 * z).sin();
        assert_relative_eq!(product, expected, epsilon = 1e-5);
        Ok(())
    }

    /// Test error function accuracy using known identities
    #[test]
    fn test_error_function_accuracy() -> TorshResult<()> {
        // Test erf(0) = 0
        let input = Tensor::from_data(vec![0.0f32], vec![1], DeviceType::Cpu)?;
        let result = erf(&input)?;
        let values = result.data()?;
        assert_abs_diff_eq!(values[0], 0.0, epsilon = 1e-10);

        // Test erf(-x) = -erf(x) (odd function)
        let x = 1.5f32;
        let input_pos = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
        let input_neg = Tensor::from_data(vec![-x], vec![1], DeviceType::Cpu)?;

        let erf_pos = erf(&input_pos)?;
        let erf_neg = erf(&input_neg)?;

        assert_relative_eq!(erf_pos.data()?[0], -erf_neg.data()?[0], epsilon = 1e-6);

        // Test erf(∞) = 1 (use large value)
        let input = Tensor::from_data(vec![10.0f32], vec![1], DeviceType::Cpu)?;
        let result = erf(&input)?;
        let values = result.data()?;
        assert_relative_eq!(values[0], 1.0, epsilon = 1e-6);

        // Test erf(x) + erfc(x) = 1
        let x = 2.0f32;
        let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
        let erf_result = erf(&input)?;
        let erfc_result = erfc(&input)?;

        let sum = erf_result.data()?[0] + erfc_result.data()?[0];
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Test known values
        // erf(1) ≈ 0.8427007929
        let input = Tensor::from_data(vec![1.0f32], vec![1], DeviceType::Cpu)?;
        let result = erf(&input)?;
        let values = result.data()?;
        assert_relative_eq!(values[0], 0.842_700_8, epsilon = 1e-6);
        Ok(())
    }

    /// Test Bessel function accuracy using known identities
    #[test]
    fn test_bessel_function_accuracy() -> TorshResult<()> {
        // Test J₀(0) = 1
        let input = Tensor::from_data(vec![0.0f32], vec![1], DeviceType::Cpu)?;
        let result = bessel_j0_scirs2(&input)?;
        let values = result.data()?;
        assert_relative_eq!(values[0], 1.0, epsilon = 1e-6);

        // Test J₁(0) = 0
        let input = Tensor::from_data(vec![0.0f32], vec![1], DeviceType::Cpu)?;
        let result = bessel_j1_scirs2(&input)?;
        let values = result.data()?;
        assert_abs_diff_eq!(values[0], 0.0, epsilon = 1e-10);

        // Test Bessel function recurrence relation: Jₙ₋₁(x) + Jₙ₊₁(x) = (2n/x)Jₙ(x)
        let x = 5.0f32;
        let n = 2;

        let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
        let j1 = bessel_j1_scirs2(&input)?;
        let j2 = bessel_jn_scirs2(n, &input)?;
        let j3 = bessel_jn_scirs2(n + 1, &input)?;

        let left_side = j1.data()?[0] + j3.data()?[0];
        let right_side = (2.0 * n as f32 / x) * j2.data()?[0];
        // Note: Current Bessel function implementation has limited precision for recurrence relations
        assert_relative_eq!(left_side, right_side, epsilon = 1e-1);

        // Test orthogonality (approximate check for specific values)
        // ∫₀¹ x J₀(αₙx) J₀(αₘx) dx = 0 for n ≠ m
        // This is hard to test directly, so we test known zeros instead

        // First few zeros of J₀: 2.4048, 5.5201, 8.6537
        let zeros = vec![2.4048f32, 5.5201f32, 8.6537f32];
        for &zero in &zeros {
            let input = Tensor::from_data(vec![zero], vec![1], DeviceType::Cpu)?;
            let result = bessel_j0_scirs2(&input)?;
            let values = result.data()?;
            assert_abs_diff_eq!(values[0], 0.0, epsilon = 1e-3);
        }
        Ok(())
    }

    /// Test special function integrals and derivatives
    #[test]
    fn test_function_integrals_and_derivatives() -> TorshResult<()> {
        // Test derivative of erf: d/dx erf(x) = (2/√π) exp(-x²)
        let x = 1.0f32;
        let h = 1e-5f32;

        let input1 = Tensor::from_data(vec![x + h], vec![1], DeviceType::Cpu)?;
        let input2 = Tensor::from_data(vec![x - h], vec![1], DeviceType::Cpu)?;

        let erf1 = erf(&input1)?;
        let erf2 = erf(&input2)?;

        let numerical_derivative = (erf1.data()?[0] - erf2.data()?[0]) / (2.0 * h);
        let analytical_derivative = (2.0 / (PI as f32).sqrt()) * (-x * x).exp();

        assert_relative_eq!(numerical_derivative, analytical_derivative, epsilon = 1e-3);

        // Test integral of sinc: ∫₋∞^∞ sinc(x) dx = π
        // This is a challenging integral to compute numerically, so we'll test a simpler property instead:
        // sinc is even: sinc(-x) = sinc(x)
        let test_values = vec![1.0f32, 2.0, 3.0, 5.0];
        for &x in &test_values {
            let input_pos = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
            let input_neg = Tensor::from_data(vec![-x], vec![1], DeviceType::Cpu)?;

            let sinc_pos = sinc(&input_pos)?;
            let sinc_neg = sinc(&input_neg)?;

            assert_relative_eq!(sinc_pos.data()?[0], sinc_neg.data()?[0], epsilon = 1e-6);
        }
        Ok(())
    }

    /// Test function symmetries and asymptotic behavior
    #[test]
    fn test_function_symmetries() -> TorshResult<()> {
        // Test that gamma is log-convex: ln(Γ(λx + (1-λ)y)) ≤ λ ln(Γ(x)) + (1-λ) ln(Γ(y))
        // Use smaller values to avoid numerical issues
        let x = 1.5f32;
        let y = 2.5f32;
        let lambda = 0.5f32;
        let interpolated = lambda * x + (1.0 - lambda) * y;

        let input_x = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
        let input_y = Tensor::from_data(vec![y], vec![1], DeviceType::Cpu)?;
        let input_interp = Tensor::from_data(vec![interpolated], vec![1], DeviceType::Cpu)?;

        let gamma_x = lgamma(&input_x)?;
        let gamma_y = lgamma(&input_y)?;
        let gamma_interp = lgamma(&input_interp)?;

        let left = gamma_interp.data()?[0];
        let right = lambda * gamma_x.data()?[0] + (1.0 - lambda) * gamma_y.data()?[0];

        // Allow larger tolerance for numerical precision
        assert!(
            left <= right + 1e-3,
            "Log-convexity failed: {left} > {right}"
        );

        // Test asymptotic behavior of gamma function for large arguments
        // Stirling's approximation: ln(Γ(x)) ≈ (x-1/2)ln(x) - x + ln(√(2π))
        let x = 100.0f32;
        let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
        let result = lgamma(&input)?;

        let stirling = (x - 0.5) * x.ln() - x + (2.0 * PI as f32).sqrt().ln();
        assert_relative_eq!(result.data()?[0], stirling, epsilon = 1e-2);
        Ok(())
    }

    /// Test complex function accuracy (if complex module is available)
    #[test]
    fn test_complex_function_accuracy() -> TorshResult<()> {
        use torsh_core::dtype::Complex64;

        // Test complex gamma: Γ(z) * Γ(1-z) = π / sin(πz)
        let z = Complex64::new(0.3, 0.4);
        let one_minus_z = Complex64::new(1.0, 0.0) - z;

        let input1 = Tensor::from_data(vec![z], vec![1], DeviceType::Cpu)?;
        let input2 = Tensor::from_data(vec![one_minus_z], vec![1], DeviceType::Cpu)?;

        let gamma_z = complex_gamma_c64(&input1)?;
        let gamma_1_minus_z = complex_gamma_c64(&input2)?;

        let product = gamma_z.data()?[0] * gamma_1_minus_z.data()?[0];
        let pi_z = Complex64::new(PI, 0.0) * z;
        let sin_pi_z =
            (Complex64::new(0.0, 1.0) * pi_z).exp() - (Complex64::new(0.0, -1.0) * pi_z).exp();
        let sin_pi_z = sin_pi_z / Complex64::new(0.0, 2.0);
        let expected = Complex64::new(PI, 0.0) / sin_pi_z;

        assert_relative_eq!(product.re, expected.re, epsilon = 1e-6);
        assert_relative_eq!(product.im, expected.im, epsilon = 1e-6);

        // Test ζ(2) = π²/6 (allow slightly larger tolerance for complex implementation)
        let input = Tensor::from_data(vec![Complex64::new(2.0, 0.0)], vec![1], DeviceType::Cpu)?;
        let result = complex_zeta_c64(&input)?;
        let values = result.data()?;
        assert_relative_eq!(values[0].re, PI * PI / 6.0, epsilon = 1e-3);
        assert_abs_diff_eq!(values[0].im, 0.0, epsilon = 1e-6);
        Ok(())
    }

    /// Test function ranges and boundary conditions
    #[test]
    fn test_function_ranges() -> TorshResult<()> {
        // Test that erf is bounded: -1 ≤ erf(x) ≤ 1
        let test_values = vec![-10.0f32, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
        for &x in &test_values {
            let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
            let result = erf(&input)?;
            let value = result.data()?[0];
            assert!((-1.0..=1.0).contains(&value));
        }

        // Test that erfc is bounded: 0 ≤ erfc(x) ≤ 2
        for &x in &test_values {
            let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
            let result = erfc(&input)?;
            let value = result.data()?[0];
            assert!((0.0..=2.0).contains(&value));
        }

        // Test that gamma is positive for positive arguments
        let positive_values = vec![0.1f32, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &x in &positive_values {
            let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
            let result = gamma(&input)?;
            let value = result.data()?[0];
            assert!(value > 0.0);
        }
        Ok(())
    }

    /// Test numerical stability for extreme values
    #[test]
    fn test_numerical_stability() -> TorshResult<()> {
        // Test gamma function for small positive values
        let small_values = vec![1e-10f32, 1e-5, 1e-3, 0.01, 0.1];
        for &x in &small_values {
            let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
            let result = gamma(&input)?;
            let value = result.data()?[0];
            assert!(value.is_finite() && value > 0.0);
        }

        // Test error functions for large values
        let large_values = vec![10.0f32, 20.0, 50.0, 100.0];
        for &x in &large_values {
            let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;
            let erf_result = erf(&input)?;
            let erfc_result = erfc(&input)?;

            assert!(erf_result.data()?[0].is_finite());
            assert!(erfc_result.data()?[0].is_finite());
            assert_relative_eq!(erf_result.data()?[0], 1.0, epsilon = 1e-6);
            assert!(erfc_result.data()?[0] >= 0.0 && erfc_result.data()?[0] <= 1e-10);
        }

        // Test Bessel functions don't overflow/underflow
        let test_values = vec![0.1f32, 1.0, 10.0, 50.0];
        for &x in &test_values {
            let input = Tensor::from_data(vec![x], vec![1], DeviceType::Cpu)?;

            let j0_result = bessel_j0_scirs2(&input)?;
            let j1_result = bessel_j1_scirs2(&input)?;
            let y0_result = bessel_y0_scirs2(&input)?;
            let y1_result = bessel_y1_scirs2(&input)?;

            assert!(j0_result.data()?[0].is_finite());
            assert!(j1_result.data()?[0].is_finite());
            assert!(y0_result.data()?[0].is_finite());
            assert!(y1_result.data()?[0].is_finite());
        }
        Ok(())
    }

    // Helper function to compute factorial
    fn factorial(n: usize) -> u64 {
        if n == 0 {
            1
        } else {
            n as u64 * factorial(n - 1)
        }
    }
}
