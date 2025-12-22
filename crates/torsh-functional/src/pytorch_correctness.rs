//! PyTorch numerical correctness tests
//!
//! This module provides comprehensive tests to ensure torsh-functional operations
//! produce numerically equivalent results to their PyTorch counterparts.
//!
//! The tests are based on known PyTorch behavior and expected numerical outputs
//! for various mathematical operations, activation functions, and loss functions.

#[cfg(test)]
use crate::*;
#[cfg(test)]
use torsh_core::{device::DeviceType, Result as TorshResult};
#[cfg(test)]
use torsh_tensor::{creation::*, Tensor};

/// Tolerances for different types of numerical comparisons
pub const STRICT_TOLERANCE: f32 = 1e-7;
pub const DEFAULT_TOLERANCE: f32 = 1e-6;
pub const RELAXED_TOLERANCE: f32 = 1e-4;
pub const LOOSE_TOLERANCE: f32 = 1e-3;

/// Helper to create deterministic test tensors (equivalent to torch.manual_seed + tensor creation)
#[cfg(test)]
fn create_test_tensor_1d() -> TorshResult<Tensor> {
    // Equivalent to torch.tensor([1.0, -2.0, 3.0, -0.5, 0.0])
    from_vec(vec![1.0, -2.0, 3.0, -0.5, 0.0], &[5], DeviceType::Cpu)
}

#[cfg(test)]
fn create_test_tensor_2d() -> TorshResult<Tensor> {
    // Equivalent to torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    from_vec(vec![1.0, 2.0, -1.0, 0.5], &[2, 2], DeviceType::Cpu)
}

#[cfg(test)]
fn create_regression_test_data() -> TorshResult<(Tensor, Tensor)> {
    // PyTorch equivalent:
    // input = torch.tensor([2.5, 1.0, -1.5, 3.0])
    // target = torch.tensor([2.0, 1.2, -1.0, 2.8])
    let input = from_vec(vec![2.5, 1.0, -1.5, 3.0], &[4], DeviceType::Cpu)?;
    let target = from_vec(vec![2.0, 1.2, -1.0, 2.8], &[4], DeviceType::Cpu)?;
    Ok((input, target))
}

/// Numerical correctness tests for activation functions
#[cfg(test)]
pub mod activation_correctness {
    use super::*;

    #[test]
    fn test_relu_pytorch_equivalence() -> TorshResult<()> {
        let input = create_test_tensor_1d()?;
        let output = relu(&input, false)?;

        // Expected PyTorch output: torch.relu(torch.tensor([1.0, -2.0, 3.0, -0.5, 0.0]))
        // = [1.0, 0.0, 3.0, 0.0, 0.0]
        let expected = from_vec(vec![1.0, 0.0, 3.0, 0.0, 0.0], &[5], DeviceType::Cpu)?;

        assert_tensors_close(&output, &expected, STRICT_TOLERANCE)?;
        Ok(())
    }

    #[test]
    fn test_sigmoid_pytorch_equivalence() -> TorshResult<()> {
        let input = from_vec(vec![0.0, 1.0, -1.0, 2.0, -2.0], &[5], DeviceType::Cpu)?;
        let output = sigmoid(&input)?;

        // Expected PyTorch output: torch.sigmoid(torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0]))
        // ≈ [0.5, 0.7311, 0.2689, 0.8808, 0.1192]
        let expected = from_vec(
            vec![0.5, 0.7310586, 0.2689414, 0.8807971, 0.1192029],
            &[5],
            DeviceType::Cpu,
        )?;

        assert_tensors_close(&output, &expected, DEFAULT_TOLERANCE)?;
        Ok(())
    }

    #[test]
    fn test_tanh_pytorch_equivalence() -> TorshResult<()> {
        let input = from_vec(vec![0.0, 1.0, -1.0, 0.5], &[4], DeviceType::Cpu)?;
        let output = tanh(&input)?;

        // Expected PyTorch output: torch.tanh(torch.tensor([0.0, 1.0, -1.0, 0.5]))
        // ≈ [0.0, 0.7616, -0.7616, 0.4621]
        let expected = from_vec(
            vec![0.0, 0.7615942, -0.7615942, 0.4621172],
            &[4],
            DeviceType::Cpu,
        )?;

        assert_tensors_close(&output, &expected, DEFAULT_TOLERANCE)?;
        Ok(())
    }

    #[test]
    fn test_softmax_pytorch_equivalence() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let output = softmax(&input, -1, None)?;

        // Expected PyTorch output: torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=-1)
        // ≈ [0.0900, 0.2447, 0.6652]
        let expected = from_vec(
            vec![0.09003058, 0.24472848, 0.66524094],
            &[3],
            DeviceType::Cpu,
        )?;

        assert_tensors_close(&output, &expected, RELAXED_TOLERANCE)?;
        Ok(())
    }

    #[test]
    fn test_gelu_pytorch_equivalence() -> TorshResult<()> {
        let input = from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4], DeviceType::Cpu)?;
        let output = gelu(&input)?;

        // Expected PyTorch output: torch.nn.functional.gelu(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        // ≈ [-0.1588, 0.0, 0.8412, 1.9545]
        let expected = from_vec(
            vec![-0.15880803, 0.0, 0.8411919, 1.9545977],
            &[4],
            DeviceType::Cpu,
        )?;

        assert_tensors_close(&output, &expected, RELAXED_TOLERANCE)?;
        Ok(())
    }
}

/// Numerical correctness tests for loss functions
#[cfg(test)]
pub mod loss_correctness {
    use super::*;
    use crate::loss::{l1_loss, mse_loss, smooth_l1_loss, ReductionType};

    #[test]
    fn test_mse_loss_pytorch_equivalence() -> TorshResult<()> {
        let (input, target) = create_regression_test_data()?;
        let output = mse_loss(&input, &target, ReductionType::Mean)?;

        // Expected PyTorch output:
        // input = torch.tensor([2.5, 1.0, -1.5, 3.0])
        // target = torch.tensor([2.0, 1.2, -1.0, 2.8])
        // torch.nn.functional.mse_loss(input, target, reduction='mean')
        // = 0.145
        let expected_value = 0.145;
        let output_value = output.item()?;

        assert!(
            (output_value - expected_value).abs() < DEFAULT_TOLERANCE,
            "MSE loss mismatch: got {}, expected {}",
            output_value,
            expected_value
        );
        Ok(())
    }

    #[test]
    fn test_l1_loss_pytorch_equivalence() -> TorshResult<()> {
        let (input, target) = create_regression_test_data()?;
        let output = l1_loss(&input, &target, ReductionType::Mean)?;

        // Expected PyTorch output:
        // torch.nn.functional.l1_loss(input, target, reduction='mean')
        // = 0.35
        let expected_value = 0.35;
        let output_value = output.item()?;

        assert!(
            (output_value - expected_value).abs() < DEFAULT_TOLERANCE,
            "L1 loss mismatch: got {}, expected {}",
            output_value,
            expected_value
        );
        Ok(())
    }

    #[test]
    fn test_smooth_l1_loss_pytorch_equivalence() -> TorshResult<()> {
        let (input, target) = create_regression_test_data()?;
        let output = smooth_l1_loss(&input, &target, ReductionType::Mean, 1.0)?;

        // Expected PyTorch output:
        // torch.nn.functional.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
        // ≈ 0.0725
        let expected_value = 0.0725;
        let output_value = output.item()?;

        assert!(
            (output_value - expected_value).abs() < RELAXED_TOLERANCE,
            "Smooth L1 loss mismatch: got {}, expected {}",
            output_value,
            expected_value
        );
        Ok(())
    }

    #[test]
    fn test_cross_entropy_pytorch_equivalence() -> TorshResult<()> {
        // PyTorch equivalent:
        // logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])
        // targets = torch.tensor([0, 1])
        // torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
        let logits = from_vec(vec![2.0, 1.0, 0.1, 0.5, 2.5, 1.0], &[2, 3], DeviceType::Cpu)?;
        let targets = from_vec(vec![0.0, 1.0], &[2], DeviceType::Cpu)?; // class indices as floats

        let output = cross_entropy(&logits, &targets, None, "mean", None, 0.0)?;

        // Expected PyTorch output: ≈ 0.3617
        let expected_value = 0.3617;
        let output_value = output.item()?;

        assert!(
            (output_value - expected_value).abs() < RELAXED_TOLERANCE,
            "Cross entropy loss mismatch: got {}, expected {}",
            output_value,
            expected_value
        );
        Ok(())
    }
}

/// Numerical correctness tests for mathematical operations
#[cfg(test)]
pub mod math_correctness {
    use super::*;
    use crate::math::{cos, exp, sin, sqrt};
    use crate::reduction::{mean, sum};

    #[test]
    fn test_reduction_ops_pytorch_equivalence() -> TorshResult<()> {
        let input = create_test_tensor_2d()?; // [[1.0, 2.0], [-1.0, 0.5]]

        // Test sum
        let sum_result = sum(&input)?;
        let expected_sum = 2.5; // 1.0 + 2.0 + (-1.0) + 0.5
        assert!((sum_result.item()? - expected_sum).abs() < DEFAULT_TOLERANCE);

        // Test mean
        let mean_result = mean(&input)?;
        let expected_mean = 0.625; // 2.5 / 4
        assert!((mean_result.item()? - expected_mean).abs() < DEFAULT_TOLERANCE);

        Ok(())
    }

    #[test]
    fn test_elementwise_math_pytorch_equivalence() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 4.0, 9.0, 16.0], &[4], DeviceType::Cpu)?;

        // Test sqrt
        let sqrt_result = sqrt(&input)?;
        let expected_sqrt = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu)?;
        assert_tensors_close(&sqrt_result, &expected_sqrt, DEFAULT_TOLERANCE)?;

        // Test exp
        let exp_input = from_vec(vec![0.0, 1.0, 2.0], &[3], DeviceType::Cpu)?;
        let exp_result = exp(&exp_input)?;
        let expected_exp = from_vec(
            vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2)],
            &[3],
            DeviceType::Cpu,
        )?;
        assert_tensors_close(&exp_result, &expected_exp, RELAXED_TOLERANCE)?;

        Ok(())
    }

    #[test]
    fn test_trigonometric_pytorch_equivalence() -> TorshResult<()> {
        let angles = from_vec(
            vec![
                0.0,
                std::f32::consts::PI / 6.0,
                std::f32::consts::PI / 4.0,
                std::f32::consts::PI / 2.0,
            ],
            &[4],
            DeviceType::Cpu,
        )?;

        // Test sin
        let sin_result = sin(&angles)?;
        let expected_sin = from_vec(
            vec![0.0, 0.5, std::f32::consts::FRAC_1_SQRT_2, 1.0],
            &[4],
            DeviceType::Cpu,
        )?;
        assert_tensors_close(&sin_result, &expected_sin, RELAXED_TOLERANCE)?;

        // Test cos
        let cos_result = cos(&angles)?;
        let expected_cos = from_vec(
            vec![1.0, 0.866025, std::f32::consts::FRAC_1_SQRT_2, 0.0],
            &[4],
            DeviceType::Cpu,
        )?;
        assert_tensors_close(&cos_result, &expected_cos, RELAXED_TOLERANCE)?;

        Ok(())
    }
}

/// Numerical correctness tests for convolution operations
#[cfg(test)]
pub mod conv_correctness {
    use super::*;
    use crate::conv::{conv1d, conv2d};

    #[test]
    fn test_conv1d_pytorch_equivalence() -> TorshResult<()> {
        // Simple 1D convolution test
        // PyTorch equivalent:
        // input = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (N=1, C=1, L=4)
        // weight = torch.tensor([[[1.0, -1.0]]])          # (out_c=1, in_c=1, k=2)
        // torch.nn.functional.conv1d(input, weight)
        // Expected output: [[[-1.0, -1.0, -1.0]]]

        let input = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4], DeviceType::Cpu)?;
        let weight = from_vec(vec![1.0, -1.0], &[1, 1, 2], DeviceType::Cpu)?;

        let output = conv1d(&input, &weight, None, 1, 0, 1, 1)?;
        let expected = from_vec(vec![-1.0, -1.0, -1.0], &[1, 1, 3], DeviceType::Cpu)?;

        assert_tensors_close(&output, &expected, DEFAULT_TOLERANCE)?;
        Ok(())
    }

    #[test]
    fn test_conv2d_simple_pytorch_equivalence() -> TorshResult<()> {
        // Simple 2D convolution test
        // PyTorch equivalent:
        // input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])    # (N=1, C=1, H=2, W=2)
        // weight = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])   # (out_c=1, in_c=1, kH=2, kW=2)
        // torch.nn.functional.conv2d(input, weight)
        // Expected output: [[[5.0]]]  # 1*1 + 2*0 + 3*0 + 4*1 = 5

        let input = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], DeviceType::Cpu)?;
        let weight = from_vec(vec![1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2], DeviceType::Cpu)?;

        let output = conv2d(&input, &weight, None, (1, 1), (0, 0), (1, 1), 1)?;
        let expected = from_vec(vec![5.0], &[1, 1, 1, 1], DeviceType::Cpu)?;

        assert_tensors_close(&output, &expected, DEFAULT_TOLERANCE)?;
        Ok(())
    }
}

/// Batch testing with multiple configurations
#[cfg(test)]
pub mod batch_correctness_tests {
    use super::*;

    /// Test multiple activation functions with the same input for consistency
    #[test]
    fn test_activation_batch_consistency() -> TorshResult<()> {
        let test_cases = vec![
            vec![0.0, 1.0, -1.0, 0.5, -0.5],
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![10.0, -10.0, 0.1, -0.1],
        ];

        for case in test_cases {
            let input = from_vec(case.clone(), &[case.len()], DeviceType::Cpu)?;

            // All activation functions should produce valid outputs
            let relu_out = relu(&input, false)?;
            let sigmoid_out = sigmoid(&input)?;
            let tanh_out = tanh(&input)?;

            // Basic sanity checks
            assert!(
                relu_out.data()?.iter().all(|&x| x >= 0.0),
                "ReLU should be non-negative"
            );
            assert!(
                sigmoid_out.data()?.iter().all(|&x| x >= 0.0 && x <= 1.0),
                "Sigmoid should be in [0,1]"
            );
            assert!(
                tanh_out.data()?.iter().all(|&x| x >= -1.0 && x <= 1.0),
                "Tanh should be in [-1,1]"
            );
        }

        Ok(())
    }

    /// Test loss functions with various reduction types for consistency
    #[test]
    fn test_loss_reduction_consistency() -> TorshResult<()> {
        use crate::loss::{mse_loss, ReductionType};

        let input = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu)?;
        let target = from_vec(vec![1.1, 2.1, 2.9, 3.9], &[4], DeviceType::Cpu)?;

        let none_loss = mse_loss(&input, &target, ReductionType::None)?;
        let sum_loss = mse_loss(&input, &target, ReductionType::Sum)?;
        let mean_loss = mse_loss(&input, &target, ReductionType::Mean)?;

        // Verify relationships between reductions
        let manual_sum = none_loss.sum()?;
        let manual_mean = manual_sum.div_scalar(4.0)?; // 4 elements

        assert!((sum_loss.item()? - manual_sum.item()?).abs() < DEFAULT_TOLERANCE);
        assert!((mean_loss.item()? - manual_mean.item()?).abs() < DEFAULT_TOLERANCE);

        Ok(())
    }
}

/// Edge case and boundary condition tests
#[cfg(test)]
pub mod edge_case_correctness {
    use super::*;

    #[test]
    fn test_extreme_values() -> TorshResult<()> {
        // Test with very large values
        let large_input = from_vec(vec![100.0, -100.0, 1000.0], &[3], DeviceType::Cpu)?;
        let sigmoid_out = sigmoid(&large_input)?;

        // Sigmoid should still work and produce valid outputs
        let data = sigmoid_out.data()?;
        assert!(data[0] > 0.99); // sigmoid(100) ≈ 1
        assert!(data[1] < 0.01); // sigmoid(-100) ≈ 0
        assert!(data[2] > 0.99); // sigmoid(1000) ≈ 1

        // Test with very small values
        let small_input = from_vec(vec![1e-10, -1e-10, 0.0], &[3], DeviceType::Cpu)?;
        let tanh_out = tanh(&small_input)?;

        // Tanh should handle small values correctly
        let small_data = tanh_out.data()?;
        assert!((small_data[0] - 1e-10f32).abs() < 1e-9f32); // tanh(x) ≈ x for small x
        assert!((small_data[1] + 1e-10f32).abs() < 1e-9f32);
        assert!(small_data[2].abs() < 1e-10f32);

        Ok(())
    }

    #[test]
    fn test_special_float_values() -> TorshResult<()> {
        // Test with zeros
        let zeros_input = from_vec(vec![0.0, 0.0, 0.0], &[3], DeviceType::Cpu)?;
        let relu_out = relu(&zeros_input, false)?;
        let sigmoid_out = sigmoid(&zeros_input)?;

        assert!(relu_out.data()?.iter().all(|&x| x == 0.0));
        assert!(sigmoid_out
            .data()?
            .iter()
            .all(|&x| (x - 0.5f32).abs() < DEFAULT_TOLERANCE));

        Ok(())
    }
}

// Helper function to compare tensors with tolerance (re-used from testing.rs)
#[cfg(test)]
pub fn assert_tensors_close(a: &Tensor, b: &Tensor, tolerance: f32) -> TorshResult<()> {
    let a_data = a.data()?;
    let b_data = b.data()?;

    assert_eq!(a_data.len(), b_data.len(), "Tensors have different sizes");

    for (i, (&a_val, &b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
        assert!(
            (a_val - b_val).abs() <= tolerance,
            "Mismatch at index {}: {} vs {} (tolerance: {})",
            i,
            a_val,
            b_val,
            tolerance
        );
    }

    Ok(())
}

/// PyTorch reference values for complex test cases
#[cfg(test)]
pub mod pytorch_reference_values {
    //! This module contains pre-computed reference values from PyTorch
    //! to ensure numerical equivalence without requiring PyTorch as a dependency.

    /// Reference values for sigmoid function
    pub const SIGMOID_REFERENCE: &[(f32, f32)] = &[
        (-5.0, 0.0067153),
        (-2.0, 0.1192029),
        (-1.0, 0.2689414),
        (0.0, 0.5),
        (1.0, 0.7310586),
        (2.0, 0.8807971),
        (5.0, 0.9932847),
    ];

    /// Reference values for tanh function
    pub const TANH_REFERENCE: &[(f32, f32)] = &[
        (-2.0, -0.9640276),
        (-1.0, -0.7615942),
        (0.0, 0.0),
        (1.0, 0.7615942),
        (2.0, 0.9640276),
    ];

    /// Reference values for GELU function
    pub const GELU_REFERENCE: &[(f32, f32)] = &[
        (-2.0, -0.04540229),
        (-1.0, -0.15880803),
        (0.0, 0.0),
        (1.0, 0.8411919),
        (2.0, 1.9545977),
    ];
}

/// Comprehensive test suite that validates against all reference values
#[cfg(test)]
mod reference_validation_tests {
    use super::*;
    use pytorch_reference_values::*;

    #[test]
    fn test_sigmoid_against_pytorch_references() -> TorshResult<()> {
        for &(input_val, expected) in SIGMOID_REFERENCE {
            let input = from_vec(vec![input_val], &[1], DeviceType::Cpu)?;
            let output = sigmoid(&input)?;
            let actual = output.item()?;

            assert!(
                (actual - expected).abs() < RELAXED_TOLERANCE,
                "Sigmoid({}) = {}, expected {} (diff: {})",
                input_val,
                actual,
                expected,
                (actual - expected).abs()
            );
        }
        Ok(())
    }

    #[test]
    fn test_tanh_against_pytorch_references() -> TorshResult<()> {
        for &(input_val, expected) in TANH_REFERENCE {
            let input = from_vec(vec![input_val], &[1], DeviceType::Cpu)?;
            let output = tanh(&input)?;
            let actual = output.item()?;

            assert!(
                (actual - expected).abs() < RELAXED_TOLERANCE,
                "Tanh({}) = {}, expected {} (diff: {})",
                input_val,
                actual,
                expected,
                (actual - expected).abs()
            );
        }
        Ok(())
    }

    #[test]
    fn test_gelu_against_pytorch_references() -> TorshResult<()> {
        for &(input_val, expected) in GELU_REFERENCE {
            let input = from_vec(vec![input_val], &[1], DeviceType::Cpu)?;
            let output = gelu(&input)?;
            let actual = output.item()?;

            assert!(
                (actual - expected).abs() < RELAXED_TOLERANCE,
                "GELU({}) = {}, expected {} (diff: {})",
                input_val,
                actual,
                expected,
                (actual - expected).abs()
            );
        }
        Ok(())
    }
}
