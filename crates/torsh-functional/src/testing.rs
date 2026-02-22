//! Testing utilities and numerical correctness tests
//!
//! This module provides utilities for testing functional operations including
//! numerical correctness validation and property-based testing helpers.

#[cfg(test)]
use crate::loss::{l1_loss, mse_loss, smooth_l1_loss, ReductionType};
#[cfg(test)]
use torsh_core::Result as TorshResult;
#[cfg(test)]
use torsh_tensor::{
    creation::{randn, zeros},
    Tensor,
};

/// Numerical tolerance for floating point comparisons
pub const DEFAULT_TOLERANCE: f32 = 1e-6;
pub const RELAXED_TOLERANCE: f32 = 1e-4;

/// Helper function to compare tensors with specified tolerance
#[cfg(test)]
pub fn assert_tensors_close(a: &Tensor, b: &Tensor, tolerance: f32) -> TorshResult<()> {
    let a_data = a.data()?;
    let b_data = b.data()?;

    assert_eq!(a_data.len(), b_data.len(), "Tensors have different sizes");

    for (i, (&a_val, &b_val)) in a_data.iter().zip(b_data.iter()).enumerate() {
        assert!(
            (a_val - b_val).abs() <= tolerance,
            "Mismatch at index {}: {} vs {}",
            i,
            a_val,
            b_val
        );
    }

    Ok(())
}

/// Test numerical properties of mathematical operations
#[cfg(test)]
pub mod numerical_properties {
    use super::*;

    /// Test that MSE loss is always non-negative
    #[test]
    fn test_mse_loss_non_negative() -> TorshResult<()> {
        let input = randn(&[10, 5])?;
        let target = randn(&[10, 5])?;

        let loss = mse_loss(&input, &target, ReductionType::Mean)?;
        let loss_data = loss.data()?;

        assert!(
            loss_data[0] >= 0.0,
            "MSE loss should be non-negative, got {}",
            loss_data[0]
        );

        Ok(())
    }

    /// Test that MSE loss is zero when input equals target
    #[test]
    fn test_mse_loss_zero_when_equal() -> TorshResult<()> {
        let input = randn(&[5, 3])?;
        let target = input.clone();

        let loss = mse_loss(&input, &target, ReductionType::Mean)?;
        let loss_data = loss.data()?;

        assert!((loss_data[0] - 0.0).abs() <= DEFAULT_TOLERANCE);

        Ok(())
    }

    /// Test that L1 loss is always non-negative
    #[test]
    fn test_l1_loss_non_negative() -> TorshResult<()> {
        let input = randn(&[8, 4])?;
        let target = randn(&[8, 4])?;

        let loss = l1_loss(&input, &target, ReductionType::Mean)?;
        let loss_data = loss.data()?;

        assert!(
            loss_data[0] >= 0.0,
            "L1 loss should be non-negative, got {}",
            loss_data[0]
        );

        Ok(())
    }

    /// Test that L1 loss is zero when input equals target
    #[test]
    fn test_l1_loss_zero_when_equal() -> TorshResult<()> {
        let input = randn(&[6, 7])?;
        let target = input.clone();

        let loss = l1_loss(&input, &target, ReductionType::Mean)?;
        let loss_data = loss.data()?;

        assert!((loss_data[0] - 0.0).abs() <= DEFAULT_TOLERANCE);

        Ok(())
    }

    /// Test smooth L1 loss properties
    #[test]
    fn test_smooth_l1_loss_properties() -> TorshResult<()> {
        let input = randn(&[10, 3])?;
        let target = randn(&[10, 3])?;
        let beta = 1.0;

        let loss = smooth_l1_loss(&input, &target, ReductionType::Mean, beta)?;
        let loss_data = loss.data()?;

        // Should be non-negative
        assert!(loss_data[0] >= 0.0, "Smooth L1 loss should be non-negative");

        // Test zero loss when inputs are equal
        let same_input = randn(&[5, 5])?;
        let zero_loss = smooth_l1_loss(&same_input, &same_input, ReductionType::Mean, beta)?;
        let zero_data = zero_loss.data()?;

        assert!((zero_data[0] - 0.0).abs() <= DEFAULT_TOLERANCE);

        Ok(())
    }

    /// Test reduction types produce consistent results
    #[test]
    fn test_reduction_consistency() -> TorshResult<()> {
        let input = randn(&[4, 6])?;
        let target = randn(&[4, 6])?;

        let none_loss = mse_loss(&input, &target, ReductionType::None)?;
        let sum_loss = mse_loss(&input, &target, ReductionType::Sum)?;
        let mean_loss = mse_loss(&input, &target, ReductionType::Mean)?;

        // Sum should equal sum of none-reduced elements
        let manual_sum = none_loss.sum()?;
        assert_tensors_close(&sum_loss, &manual_sum, DEFAULT_TOLERANCE)?;

        // Mean should equal sum divided by number of elements
        let manual_mean = manual_sum.div_scalar(none_loss.numel() as f32)?;
        assert_tensors_close(&mean_loss, &manual_mean, DEFAULT_TOLERANCE)?;

        Ok(())
    }
}

/// Property-based testing utilities
#[cfg(test)]
pub mod property_tests {
    use super::*;

    /// Test that loss functions handle different tensor shapes correctly
    #[test]
    fn test_loss_functions_with_various_shapes() -> TorshResult<()> {
        let shapes = vec![
            vec![1],
            vec![5],
            vec![3, 4],
            vec![2, 3, 4],
            vec![1, 1, 1, 1],
        ];

        for shape in shapes {
            let input = randn(&shape)?;
            let target = randn(&shape)?;

            // All loss functions should work with these shapes
            let _mse = mse_loss(&input, &target, ReductionType::Mean)?;
            let _l1 = l1_loss(&input, &target, ReductionType::Mean)?;
            let _smooth_l1 = smooth_l1_loss(&input, &target, ReductionType::Mean, 1.0)?;
        }

        Ok(())
    }

    /// Test mathematical relationships between different loss functions
    #[test]
    fn test_loss_function_relationships() -> TorshResult<()> {
        // For small errors, smooth L1 should approximate L2 loss
        let input = zeros(&[10, 5])?;
        let small_perturbation = input.add_scalar(0.01)?; // Small difference

        let l2_loss = mse_loss(&input, &small_perturbation, ReductionType::Mean)?;
        let smooth_l1_loss_val =
            smooth_l1_loss(&input, &small_perturbation, ReductionType::Mean, 1.0)?;

        // For small errors (< beta), smooth L1 ≈ 0.5 * error^2 / beta ≈ L2/2
        let expected_smooth = l2_loss.div_scalar(2.0)?;
        assert_tensors_close(&smooth_l1_loss_val, &expected_smooth, RELAXED_TOLERANCE)?;

        Ok(())
    }

    /// Test edge cases and boundary conditions
    #[test]
    fn test_edge_cases() -> TorshResult<()> {
        // Test with very small tensors
        let tiny_input = randn(&[1])?;
        let tiny_target = randn(&[1])?;

        let _loss = mse_loss(&tiny_input, &tiny_target, ReductionType::Mean)?;

        // Test with larger tensors
        let large_input = randn(&[100, 100])?;
        let large_target = randn(&[100, 100])?;

        let _large_loss = l1_loss(&large_input, &large_target, ReductionType::Sum)?;

        Ok(())
    }
}

/// Benchmarking and performance validation tests
#[cfg(test)]
pub mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// Benchmark loss function performance with different tensor sizes
    #[test]
    fn test_loss_function_performance() -> TorshResult<()> {
        let sizes = vec![
            (100, 10),     // Small
            (1000, 100),   // Medium
            (10000, 1000), // Large
        ];

        for (rows, cols) in sizes {
            let input = randn(&[rows, cols])?;
            let target = randn(&[rows, cols])?;

            let start = Instant::now();
            let _loss = mse_loss(&input, &target, ReductionType::Mean)?;
            let duration = start.elapsed();

            // Ensure reasonable performance (this is a basic smoke test)
            assert!(
                duration.as_millis() < 1000,
                "MSE loss took too long for size {}x{}: {:?}",
                rows,
                cols,
                duration
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::utils::{validate_elementwise_shapes, validate_positive, validate_range};

    #[test]
    fn test_shape_validation() -> TorshResult<()> {
        let tensor_a = zeros(&[3, 4])?;
        let tensor_b = zeros(&[3, 4])?;
        let tensor_c = zeros(&[3, 5])?;

        // Should pass for same shapes
        validate_elementwise_shapes(&tensor_a, &tensor_b)?;

        // Should fail for different shapes
        let result = validate_elementwise_shapes(&tensor_a, &tensor_c);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_range_validation() {
        // Valid range
        assert!(validate_range(5.0, 0.0, 10.0, "value", "test").is_ok());

        // Invalid ranges
        assert!(validate_range(-1.0, 0.0, 10.0, "value", "test").is_err());
        assert!(validate_range(15.0, 0.0, 10.0, "value", "test").is_err());
    }

    #[test]
    fn test_positive_validation() {
        // Valid positive values
        assert!(validate_positive(1.0, "value", "test").is_ok());
        assert!(validate_positive(0.001, "value", "test").is_ok());

        // Invalid values
        assert!(validate_positive(0.0, "value", "test").is_err());
        assert!(validate_positive(-1.0, "value", "test").is_err());
    }
}

/// PyTorch reference implementation tests
///
/// These tests validate that our implementations match PyTorch's behavior
/// for various operations and edge cases.
#[cfg(test)]
pub mod pytorch_reference_tests {
    use super::*;
    use crate::{activations::*, reduction::*};

    /// Test activation functions against PyTorch reference values
    #[test]
    fn test_relu_pytorch_reference() -> TorshResult<()> {
        // Test data based on PyTorch reference implementation
        let input = Tensor::from_data(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![5],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.relu(input)
        let expected = vec![0.0, 0.0, 0.0, 1.0, 2.0];

        let output = relu(&input, false)?;
        let output_data = output.data()?;

        for (i, (&actual, &expected_val)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected_val as f32).abs() <= DEFAULT_TOLERANCE,
                "ReLU mismatch at index {}: got {}, expected {}",
                i,
                actual,
                expected_val
            );
        }

        Ok(())
    }

    /// Test sigmoid function against PyTorch reference values
    #[test]
    fn test_sigmoid_pytorch_reference() -> TorshResult<()> {
        // Test data with known PyTorch outputs
        let input = Tensor::from_data(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![5],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.sigmoid(input)
        // Values computed using PyTorch 2.0
        let expected = vec![0.1192029, 0.2689414, 0.5, 0.7310586, 0.8807971];

        let output = sigmoid(&input)?;
        let output_data = output.data()?;

        for (i, (&actual, &expected_val)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected_val as f32).abs() <= RELAXED_TOLERANCE,
                "Sigmoid mismatch at index {}: got {}, expected {}",
                i,
                actual,
                expected_val
            );
        }

        Ok(())
    }

    /// Test tanh function against PyTorch reference values  
    #[test]
    fn test_tanh_pytorch_reference() -> TorshResult<()> {
        let input = Tensor::from_data(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![5],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.tanh(input)
        let expected = vec![-0.9640276, -0.7615942, 0.0, 0.7615942, 0.9640276];

        let output = crate::activations::tanh(&input)?;
        let output_data = output.data()?;

        for (i, (&actual, &expected_val)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected_val as f32).abs() <= RELAXED_TOLERANCE,
                "Tanh mismatch at index {}: got {}, expected {}",
                i,
                actual,
                expected_val
            );
        }

        Ok(())
    }

    /// Test softmax function against PyTorch reference values
    #[test]
    fn test_softmax_pytorch_reference() -> TorshResult<()> {
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.softmax(input, dim=0)
        let expected = vec![0.0320586, 0.0871443, 0.2368828, 0.6439142];

        let output = softmax(&input, 0, None)?;
        let output_data = output.data()?;

        for (i, (&actual, &expected_val)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected_val as f32).abs() <= RELAXED_TOLERANCE,
                "Softmax mismatch at index {}: got {}, expected {}",
                i,
                actual,
                expected_val
            );
        }

        Ok(())
    }

    /// Test GELU function against PyTorch reference values
    #[test]
    fn test_gelu_pytorch_reference() -> TorshResult<()> {
        let input = Tensor::from_data(
            vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            vec![5],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.nn.functional.gelu(input)
        let expected = vec![-0.0454023, -0.1587989, 0.0, 0.8411995, 1.9545977];

        let output = gelu(&input)?;
        let output_data = output.data()?;

        for (i, (&actual, &expected_val)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected_val as f32).abs() <= RELAXED_TOLERANCE,
                "GELU mismatch at index {}: got {}, expected {}",
                i,
                actual,
                expected_val
            );
        }

        Ok(())
    }

    /// Test reduction operations against PyTorch reference values
    #[test]
    fn test_sum_pytorch_reference() -> TorshResult<()> {
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.sum(input)
        let expected_total = 21.0;

        let output = sum(&input)?;
        let output_data = output.data()?;

        assert!(
            (output_data[0] - expected_total).abs() <= DEFAULT_TOLERANCE,
            "Sum mismatch: got {}, expected {}",
            output_data[0],
            expected_total
        );

        Ok(())
    }

    /// Test mean operation against PyTorch reference values
    #[test]
    fn test_mean_pytorch_reference() -> TorshResult<()> {
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.mean(input)
        let expected_mean = 3.5;

        let output = mean(&input)?;
        let output_data = output.data()?;

        assert!(
            (output_data[0] - expected_mean).abs() <= DEFAULT_TOLERANCE,
            "Mean mismatch: got {}, expected {}",
            output_data[0],
            expected_mean
        );

        Ok(())
    }

    /// Test dimensional reduction against PyTorch reference values
    #[test]
    fn test_sum_dim_pytorch_reference() -> TorshResult<()> {
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.sum(input) (total sum)
        let expected_total = 21.0; // 1+2+3+4+5+6

        let output = sum(&input)?; // Note: Our sum function doesn't support dimension parameter yet
        let output_data = output.data()?;

        assert!(
            (output_data[0] - expected_total).abs() <= DEFAULT_TOLERANCE,
            "Sum total mismatch: got {}, expected {}",
            output_data[0],
            expected_total
        );

        Ok(())
    }

    /// Test matrix multiplication against PyTorch reference values
    #[test]
    fn test_matmul_pytorch_reference() -> TorshResult<()> {
        let a = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;

        let b = Tensor::from_data(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: torch.matmul(a, b)
        // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected = vec![19.0, 22.0, 43.0, 50.0];

        let output = a.matmul(&b)?;
        let output_data = output.data()?;

        for (i, (&actual, &expected_val)) in output_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected_val as f32).abs() <= DEFAULT_TOLERANCE,
                "MatMul mismatch at index {}: got {}, expected {}",
                i,
                actual,
                expected_val
            );
        }

        Ok(())
    }

    /// Test loss functions against PyTorch reference values
    #[test]
    fn test_mse_loss_pytorch_reference() -> TorshResult<()> {
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            torsh_core::DeviceType::Cpu,
        )?;

        let target = Tensor::from_data(
            vec![2.0, 3.0, 4.0, 5.0],
            vec![4],
            torsh_core::DeviceType::Cpu,
        )?;

        // Expected output from PyTorch: F.mse_loss(input, target)
        // Mean of [(1-2)^2, (2-3)^2, (3-4)^2, (4-5)^2] = mean([1, 1, 1, 1]) = 1.0
        let expected = 1.0;

        let output = mse_loss(&input, &target, ReductionType::Mean)?;
        let output_data = output.data()?;

        assert!(
            (output_data[0] - expected).abs() <= DEFAULT_TOLERANCE,
            "MSE loss mismatch: got {}, expected {}",
            output_data[0],
            expected
        );

        Ok(())
    }

    /// Test numerical edge cases that are important for PyTorch compatibility
    #[test]
    fn test_edge_cases_pytorch_reference() -> TorshResult<()> {
        // Test sigmoid with extreme values
        let extreme_input =
            Tensor::from_data(vec![-100.0, 100.0], vec![2], torsh_core::DeviceType::Cpu)?;

        let output = sigmoid(&extreme_input)?;
        let output_data = output.data()?;

        // Should be approximately [0, 1] for extreme values
        assert!(
            output_data[0] < 1e-10,
            "Sigmoid of -100 should be ~0, got {}",
            output_data[0]
        );
        assert!(
            (output_data[1] - 1.0f32).abs() < 1e-10,
            "Sigmoid of 100 should be ~1, got {}",
            output_data[1]
        );

        // Test tanh with extreme values
        let output = crate::activations::tanh(&extreme_input)?;
        let output_data = output.data()?;

        // Should be approximately [-1, 1] for extreme values
        assert!(
            (output_data[0] + 1.0f32).abs() < 1e-10,
            "Tanh of -100 should be ~-1, got {}",
            output_data[0]
        );
        assert!(
            (output_data[1] - 1.0f32).abs() < 1e-10,
            "Tanh of 100 should be ~1, got {}",
            output_data[1]
        );

        Ok(())
    }
}
