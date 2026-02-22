//! Comprehensive tests for the functional interface
//!
//! Tests all activation functions, loss functions, and other functional operations
//! implemented in the torsh-nn functional module.

use approx::assert_relative_eq;
use torsh_nn::functional::*;
use torsh_tensor::creation::{ones, tensor_2d, zeros};

/// Test basic activation functions
#[test]
fn test_activation_functions() {
    // Create test input
    let input = tensor_2d(&[&[-2.0, -1.0, 0.0, 1.0, 2.0]]).unwrap();

    // Test ReLU
    let relu_output = relu(&input).unwrap();
    let relu_data = relu_output.to_vec().unwrap();
    assert_eq!(relu_data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);

    // Test Sigmoid (should be between 0 and 1)
    let sigmoid_output = sigmoid(&input).unwrap();
    let sigmoid_data = sigmoid_output.to_vec().unwrap();
    for &val in &sigmoid_data {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Sigmoid output {} not in [0, 1]",
            val
        );
    }

    // Test Tanh (should be between -1 and 1)
    let tanh_output = tanh(&input).unwrap();
    let tanh_data = tanh_output.to_vec().unwrap();
    for &val in &tanh_data {
        assert!(
            val >= -1.0 && val <= 1.0,
            "Tanh output {} not in [-1, 1]",
            val
        );
    }

    // Test that sigmoid(0) ≈ 0.5
    let zero_input = zeros(&[1]).unwrap();
    let sigmoid_zero = sigmoid(&zero_input).unwrap();
    let sigmoid_zero_data = sigmoid_zero.to_vec().unwrap();
    assert_relative_eq!(sigmoid_zero_data[0], 0.5, epsilon = 1e-5);

    // Test that tanh(0) ≈ 0
    let tanh_zero = tanh(&zero_input).unwrap();
    let tanh_zero_data = tanh_zero.to_vec().unwrap();
    assert_relative_eq!(tanh_zero_data[0], 0.0, epsilon = 1e-5);
}

/// Test advanced activation functions
#[test]
fn test_advanced_activations() {
    let input = tensor_2d(&[&[0.0, 1.0, 2.0, -1.0, -2.0]]).unwrap();

    // Test Swish/SiLU
    let swish_output = swish(&input).unwrap();
    let swish_data = swish_output.to_vec().unwrap();

    // Swish(0) should be approximately 0
    assert_relative_eq!(swish_data[0], 0.0, epsilon = 1e-5);

    // Test ELU with alpha = 1.0
    let elu_output = elu(&input, 1.0).unwrap();
    let elu_data = elu_output.to_vec().unwrap();

    // ELU should be identity for positive values
    assert_relative_eq!(elu_data[1], 1.0, epsilon = 1e-5);
    assert_relative_eq!(elu_data[2], 2.0, epsilon = 1e-5);

    // ELU should be negative for negative inputs but > -alpha
    assert!(elu_data[3] < 0.0 && elu_data[3] > -1.0);
    assert!(elu_data[4] < 0.0 && elu_data[4] > -1.0);

    // Test Leaky ReLU with slope 0.1
    let leaky_relu_output = leaky_relu(&input, 0.1).unwrap();
    let leaky_relu_data = leaky_relu_output.to_vec().unwrap();

    // Positive values should be unchanged
    assert_relative_eq!(leaky_relu_data[1], 1.0, epsilon = 1e-5);
    assert_relative_eq!(leaky_relu_data[2], 2.0, epsilon = 1e-5);

    // Negative values should be scaled by slope
    assert_relative_eq!(leaky_relu_data[3], -0.1, epsilon = 1e-5);
    assert_relative_eq!(leaky_relu_data[4], -0.2, epsilon = 1e-5);
}

/// Test softmax properties
#[test]
fn test_softmax_properties() {
    // Test 2D softmax
    let input = tensor_2d(&[&[1.0, 2.0, 3.0], &[2.0, 1.0, 3.0]]).unwrap();
    let softmax_output = softmax(&input, Some(1)).unwrap(); // Along dim 1
    let softmax_data = softmax_output.to_vec().unwrap();

    // Check that each row sums to approximately 1
    let batch_size = 2;
    let num_classes = 3;
    for batch in 0..batch_size {
        let mut row_sum = 0.0;
        for class in 0..num_classes {
            row_sum += softmax_data[batch * num_classes + class];
        }
        assert_relative_eq!(row_sum, 1.0, epsilon = 1e-5);
    }

    // Test that largest input gives largest probability
    // For first row, input[2] = 3.0 is largest
    assert!(softmax_data[2] > softmax_data[0] && softmax_data[2] > softmax_data[1]);

    // For second row, input[5] = 3.0 is largest (index 2 in second row)
    assert!(softmax_data[5] > softmax_data[3] && softmax_data[5] > softmax_data[4]);
}

/// Test log softmax properties
#[test]
fn test_log_softmax_properties() {
    let input = tensor_2d(&[&[1.0, 2.0, 3.0], &[2.0, 1.0, 3.0]]).unwrap();
    let log_softmax_output = log_softmax(&input, Some(1)).unwrap();
    let softmax_output = softmax(&input, Some(1)).unwrap();

    // Check that exp(log_softmax) ≈ softmax
    let exp_log_softmax = log_softmax_output.exp().unwrap();
    let log_softmax_data = exp_log_softmax.to_vec().unwrap();
    let softmax_data = softmax_output.to_vec().unwrap();

    for (log_val, soft_val) in log_softmax_data.iter().zip(softmax_data.iter()) {
        assert_relative_eq!(log_val, soft_val, epsilon = 1e-5);
    }
}

/// Test loss functions
#[test]
fn test_loss_functions() {
    // Test MSE loss
    let pred = tensor_2d(&[&[1.0, 2.0, 3.0]]).unwrap();
    let target = tensor_2d(&[&[1.5, 2.5, 2.5]]).unwrap();

    let mse_loss_result = mse_loss(&pred, &target, "mean").unwrap();
    let mse_data = mse_loss_result.to_vec().unwrap();

    // Expected MSE: ((1-1.5)^2 + (2-2.5)^2 + (3-2.5)^2) / 3 = (0.25 + 0.25 + 0.25) / 3 = 0.25
    assert_relative_eq!(mse_data[0], 0.25, epsilon = 1e-5);

    // Test MSE with sum reduction
    let mse_sum = mse_loss(&pred, &target, "sum").unwrap();
    let mse_sum_data = mse_sum.to_vec().unwrap();
    assert_relative_eq!(mse_sum_data[0], 0.75, epsilon = 1e-5); // 0.25 * 3

    // Test Binary Cross Entropy
    let pred_bce = tensor_2d(&[&[0.8, 0.2, 0.9]]).unwrap();
    let target_bce = tensor_2d(&[&[1.0, 0.0, 1.0]]).unwrap();

    let bce_result = binary_cross_entropy(&pred_bce, &target_bce, None, "mean").unwrap();
    let bce_data = bce_result.to_vec().unwrap();

    // BCE should be positive for non-perfect predictions
    assert!(bce_data[0] > 0.0);

    // Test perfect predictions should give near-zero loss
    let perfect_pred = tensor_2d(&[&[1.0, 0.0, 1.0]]).unwrap();
    let perfect_target = tensor_2d(&[&[1.0, 0.0, 1.0]]).unwrap();
    let perfect_bce = binary_cross_entropy(&perfect_pred, &perfect_target, None, "mean").unwrap();
    let perfect_bce_data = perfect_bce.to_vec().unwrap();
    assert!(perfect_bce_data[0] < 1e-5); // Should be very close to 0
}

/// Test dropout behavior
#[test]
fn test_dropout() {
    let input = ones(&[10, 10]).unwrap();

    // Test dropout in eval mode (should return input unchanged)
    let eval_output = dropout(&input, 0.5, false).unwrap();
    let eval_data = eval_output.to_vec().unwrap();
    let input_data = input.to_vec().unwrap();
    assert_eq!(eval_data, input_data);

    // Test dropout with p=0 (should return input unchanged)
    let no_dropout = dropout(&input, 0.0, true).unwrap();
    let no_dropout_data = no_dropout.to_vec().unwrap();
    assert_eq!(no_dropout_data, input_data);

    // Test that training mode with p>0 modifies the tensor
    let train_output = dropout(&input, 0.5, true).unwrap();
    let train_data = train_output.to_vec().unwrap();

    // With p=0.5, roughly half should be zeros and half should be scaled up
    let num_zeros = train_data.iter().filter(|&&x| x == 0.0).count();
    let num_nonzeros = train_data.iter().filter(|&&x| x != 0.0).count();

    // Allow some variance in the dropout pattern
    assert!(num_zeros > 0, "Dropout should zero out some elements");
    assert!(num_nonzeros > 0, "Dropout should preserve some elements");
}

/// Test activation function edge cases and numerical stability
#[test]
fn test_activation_stability() {
    // Test with large positive values
    let large_positive = tensor_2d(&[&[10.0, 20.0, 50.0]]).unwrap();

    let sigmoid_large = sigmoid(&large_positive).unwrap();
    let sigmoid_large_data = sigmoid_large.to_vec().unwrap();

    // Should approach 1.0 for large positive values
    for &val in &sigmoid_large_data {
        assert!(
            val > 0.99 && val <= 1.0,
            "Sigmoid of large positive value: {}",
            val
        );
    }

    // Test with large negative values
    let large_negative = tensor_2d(&[&[-10.0, -20.0, -50.0]]).unwrap();

    let sigmoid_negative = sigmoid(&large_negative).unwrap();
    let sigmoid_negative_data = sigmoid_negative.to_vec().unwrap();

    // Should approach 0.0 for large negative values
    for &val in &sigmoid_negative_data {
        assert!(
            val >= 0.0 && val < 0.01,
            "Sigmoid of large negative value: {}",
            val
        );
    }

    // Test tanh stability
    let tanh_large = tanh(&large_positive).unwrap();
    let tanh_large_data = tanh_large.to_vec().unwrap();

    for &val in &tanh_large_data {
        assert!(
            val > 0.99 && val <= 1.0,
            "Tanh of large positive value: {}",
            val
        );
    }
}

/// Test batch processing consistency
#[test]
fn test_batch_consistency() {
    // Test that batch processing gives same results as individual processing
    let single_input = tensor_2d(&[&[1.0, 2.0, 3.0]]).unwrap();
    let batch_input = tensor_2d(&[&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]]).unwrap();

    // Test ReLU
    let single_relu = relu(&single_input).unwrap();
    let batch_relu = relu(&batch_input).unwrap();

    let single_data = single_relu.to_vec().unwrap();
    let batch_data = batch_relu.to_vec().unwrap();

    // First row of batch should match single input
    for i in 0..3 {
        assert_relative_eq!(single_data[i], batch_data[i], epsilon = 1e-5);
        assert_relative_eq!(single_data[i], batch_data[i + 3], epsilon = 1e-5); // Second row too
    }

    // Test Sigmoid
    let single_sigmoid = sigmoid(&single_input).unwrap();
    let batch_sigmoid = sigmoid(&batch_input).unwrap();

    let single_sigmoid_data = single_sigmoid.to_vec().unwrap();
    let batch_sigmoid_data = batch_sigmoid.to_vec().unwrap();

    for i in 0..3 {
        assert_relative_eq!(
            single_sigmoid_data[i],
            batch_sigmoid_data[i],
            epsilon = 1e-5
        );
        assert_relative_eq!(
            single_sigmoid_data[i],
            batch_sigmoid_data[i + 3],
            epsilon = 1e-5
        );
    }
}
