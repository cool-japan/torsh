//! Comprehensive Edge Case Tests
//!
//! This module provides extensive testing of edge cases, boundary conditions,
//! and error handling for all functional operations.

#[cfg(test)]
mod tests {
    use crate::*;
    use torsh_core::{device::DeviceType, Result as TorshResult};
    use torsh_tensor::creation::*;

    // ============================================================================
    // Numerical Edge Cases
    // ============================================================================

    #[test]
    fn test_zero_input_relu() -> TorshResult<()> {
        let zeros = zeros::<f32>(&[5])?;
        let relu_result = relu(&zeros, false)?;
        assert!(relu_result.data()?.iter().all(|&x| x == 0.0));
        Ok(())
    }

    #[test]
    fn test_negative_values_abs() -> TorshResult<()> {
        let negative = from_vec(vec![-1.0f32, -2.0, -3.0], &[3], DeviceType::Cpu)?;
        let abs_result = math::abs(&negative)?;
        let expected = vec![1.0f32, 2.0, 3.0];
        for (actual, expected) in abs_result.data()?.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_large_values_sigmoid_saturation() -> TorshResult<()> {
        let large = from_vec(vec![100.0f32, 500.0, 1000.0], &[3], DeviceType::Cpu)?;
        let sigmoid_result = sigmoid(&large)?;
        for &val in sigmoid_result.data()?.iter() {
            assert!(val > 0.99 && val <= 1.0, "Sigmoid should saturate");
        }
        Ok(())
    }

    #[test]
    fn test_softmax_numerical_stability() -> TorshResult<()> {
        let large_logits = from_vec(vec![1000.0f32, 1001.0, 1002.0], &[3], DeviceType::Cpu)?;
        let softmax_result = softmax(&large_logits, -1, None)?;

        let sum: f32 = softmax_result.data()?.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "Softmax should sum to 1");

        for &val in softmax_result.data()?.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_uniform_distribution() -> TorshResult<()> {
        let uniform = ones::<f32>(&[5])?;
        let softmax_result = softmax(&uniform, -1, None)?;

        for &val in softmax_result.data()?.iter() {
            assert!(
                (val - 0.2).abs() < 1e-5,
                "Uniform input should give uniform output"
            );
        }
        Ok(())
    }

    #[test]
    fn test_gelu_at_zero() -> TorshResult<()> {
        let zero_tensor = from_vec(vec![0.0f32], &[1], DeviceType::Cpu)?;
        let gelu_result = gelu(&zero_tensor)?;
        assert!(gelu_result.data()?[0].abs() < 1e-5, "GELU(0) should be 0");
        Ok(())
    }

    // ============================================================================
    // Loss Function Edge Cases
    // ============================================================================

    #[test]
    fn test_mse_identical_inputs() -> TorshResult<()> {
        let input = from_vec(vec![1.0f32, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let target = input.clone();

        let mse = loss::mse_loss(&input, &target, loss::ReductionType::Mean)?;
        assert!(
            mse.data()?[0].abs() < 1e-6,
            "MSE of identical tensors should be 0"
        );
        Ok(())
    }

    #[test]
    fn test_l1_identical_inputs() -> TorshResult<()> {
        let input = from_vec(vec![1.0f32, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let target = input.clone();

        let l1 = loss::l1_loss(&input, &target, loss::ReductionType::Mean)?;
        assert!(
            l1.data()?[0].abs() < 1e-6,
            "L1 of identical tensors should be 0"
        );
        Ok(())
    }

    #[test]
    fn test_loss_with_extreme_values() -> TorshResult<()> {
        let input = from_vec(vec![0.0f32, 1.0, 100.0], &[3], DeviceType::Cpu)?;
        let target = from_vec(vec![100.0f32, 1.0, 0.0], &[3], DeviceType::Cpu)?;

        let mse = loss::mse_loss(&input, &target, loss::ReductionType::Mean)?;
        assert!(mse.data()?[0].is_finite(), "MSE should be finite");
        assert!(mse.data()?[0] > 0.0, "MSE should be positive");
        Ok(())
    }

    // ============================================================================
    // Shape and Dimension Edge Cases
    // ============================================================================

    #[test]
    fn test_scalar_tensor_relu() -> TorshResult<()> {
        let scalar = from_vec(vec![42.0f32], &[], DeviceType::Cpu)?;
        let relu_result = relu(&scalar, false)?;
        assert_eq!(relu_result.data()?[0], 42.0);
        Ok(())
    }

    #[test]
    fn test_single_element_mean() -> TorshResult<()> {
        let single = from_vec(vec![1.0f32], &[1], DeviceType::Cpu)?;
        let mean_result = reduction::mean(&single)?;
        assert!((mean_result.data()?[0] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_reduction_sum() -> TorshResult<()> {
        let tensor = ones::<f32>(&[2, 3, 4])?;
        let sum_result = reduction::sum(&tensor)?;
        // Sum of 2*3*4 = 24 ones should be 24
        assert!((sum_result.data()?[0] - 24.0).abs() < 1e-5);
        Ok(())
    }

    // ============================================================================
    // Convolution Edge Cases
    // ============================================================================

    #[test]
    fn test_conv_with_padding_output_size() -> TorshResult<()> {
        let input = ones::<f32>(&[1, 1, 5, 5])?;
        let kernel = ones::<f32>(&[1, 1, 3, 3])?;

        let output = conv::conv2d(&input, &kernel, None, (1, 1), (1, 1), (1, 1), 1)?;
        assert_eq!(output.shape().dims()[2], 5);
        assert_eq!(output.shape().dims()[3], 5);
        Ok(())
    }

    #[test]
    fn test_conv_with_stride_output_size() -> TorshResult<()> {
        let input = ones::<f32>(&[1, 1, 10, 10])?;
        let kernel = ones::<f32>(&[1, 1, 3, 3])?;

        let output = conv::conv2d(&input, &kernel, None, (2, 2), (0, 0), (1, 1), 1)?;
        assert_eq!(output.shape().dims()[2], 4);
        assert_eq!(output.shape().dims()[3], 4);
        Ok(())
    }

    // ============================================================================
    // Pooling Edge Cases
    // ============================================================================

    #[test]
    fn test_adaptive_pooling_exact_output_size() -> TorshResult<()> {
        let input = ones::<f32>(&[1, 1, 7, 7])?;
        let output = pooling::adaptive_avg_pool2d(&input, (3, 3))?;

        assert_eq!(output.shape().dims()[2], 3);
        assert_eq!(output.shape().dims()[3], 3);
        Ok(())
    }

    // ============================================================================
    // Matrix Operations Edge Cases
    // ============================================================================

    #[test]
    fn test_transpose_involutive() -> TorshResult<()> {
        let tensor = ones::<f32>(&[2, 3, 4])?;
        let transposed = tensor.transpose(-2, -1)?;
        let double_transpose = transposed.transpose(-2, -1)?;

        assert_eq!(double_transpose.shape().dims(), tensor.shape().dims());
        Ok(())
    }

    #[test]
    fn test_broadcasting_scalar_multiplication() -> TorshResult<()> {
        let tensor = ones::<f32>(&[3, 4])?;
        let scalar = from_vec(vec![2.0f32], &[], DeviceType::Cpu)?;

        let result = tensor.mul(&scalar)?;
        assert_eq!(result.shape().dims(), tensor.shape().dims());

        for &val in result.data()?.iter() {
            assert!((val - 2.0f32).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_broadcasting_dimension_expansion() -> TorshResult<()> {
        let a = ones::<f32>(&[3, 1, 4])?;
        let b = ones::<f32>(&[1, 5, 4])?;

        let result = a.add(&b)?;
        assert_eq!(result.shape().dims(), &[3, 5, 4]);
        Ok(())
    }

    // ============================================================================
    // Error Handling Quality Tests
    // ============================================================================

    #[test]
    fn test_conv_wrong_dimensions_has_context() {
        let wrong_dim = ones::<f32>(&[5]).unwrap();
        let kernel = ones::<f32>(&[1, 1, 3, 3]).unwrap();

        let result = conv::conv2d(&wrong_dim, &kernel, None, (1, 1), (0, 0), (1, 1), 1);
        assert!(result.is_err());

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("conv") || err_msg.contains("dimension") || err_msg.contains("tensor"),
            "Error message should provide context"
        );
    }

    #[test]
    fn test_matmul_incompatible_dimensions() {
        let a = ones::<f32>(&[3, 4]).unwrap();
        let b = ones::<f32>(&[6, 7]).unwrap();

        let result = a.matmul(&b);
        assert!(result.is_err(), "Incompatible matmul should fail");
    }

    // ============================================================================
    // Numerical Stability Tests
    // ============================================================================

    #[test]
    fn test_log_of_small_values() {
        let small = from_vec(vec![1e-10f32, 1e-15, 1e-20], &[3], DeviceType::Cpu).unwrap();
        let log_result = math::log(&small);
        assert!(
            log_result.is_ok(),
            "Log of small positive values should work"
        );
    }

    #[test]
    fn test_infinity_handling_relu() -> TorshResult<()> {
        let with_inf = from_vec(
            vec![f32::INFINITY, f32::NEG_INFINITY, 0.0],
            &[3],
            DeviceType::Cpu,
        )?;

        let relu_result = relu(&with_inf, false)?;
        let data = relu_result.data()?;

        assert!(data[0].is_infinite() && data[0] > 0.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
        Ok(())
    }

    // ============================================================================
    // Reduction Operations Edge Cases
    // ============================================================================

    #[test]
    fn test_sum_all_dimensions() -> TorshResult<()> {
        let tensor = ones::<f32>(&[2, 3, 4])?;
        let sum_all = reduction::sum(&tensor)?;

        assert_eq!(sum_all.numel(), 1);
        assert!((sum_all.data()?[0] - 24.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_mean_vs_sum_relationship() -> TorshResult<()> {
        let tensor = from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], DeviceType::Cpu)?;

        let sum = reduction::sum(&tensor)?;
        let mean = reduction::mean(&tensor)?;

        // mean should be sum / count
        let expected_mean = sum.data()?[0] / 5.0;
        assert!((mean.data()?[0] - expected_mean).abs() < 1e-5);
        Ok(())
    }
}
