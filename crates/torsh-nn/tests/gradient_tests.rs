//! Gradient checking and validation tests
//!
//! Tests that verify gradients are computed correctly for various operations

use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::functional::*;
use torsh_nn::gradcheck::{
    fast_gradcheck, fast_gradcheck_function, gradcheck, gradcheck_function, GradCheckConfig,
};
use torsh_nn::layers::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::creation::ones;
use torsh_tensor::creation::{tensor_1d, tensor_2d};
use torsh_tensor::Tensor;

/// Test gradient computation for simple operations
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_basic_gradient_check() {
    // Test MSE loss gradient
    let input = tensor_2d(&[&[1.0, 2.0]]).unwrap();
    let target = tensor_2d(&[&[1.5, 1.5]]).unwrap();

    // Create a simple function: MSE loss
    let loss_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        mse_loss(x, &target, "mean")
    };

    // Check gradient with default config
    let config = GradCheckConfig::default();
    let result = gradcheck_function(loss_fn, &input, &config);
    assert!(result.is_ok(), "Gradient check failed for MSE loss");

    let grad_result = result.unwrap();
    assert!(grad_result.passed, "MSE gradient check should pass");
}

/// Test gradient computation for activation functions
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_activation_gradients() {
    let input = tensor_2d(&[&[0.5, -0.5, 1.0, -1.0]]).unwrap();

    // Test sigmoid gradient
    let sigmoid_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        sigmoid(x)
    };

    let config = GradCheckConfig {
        eps: 1e-6,
        atol: 1e-5,
        rtol: 1e-3,
        double_precision: false,
        max_elements: None,
        seed: Some(42),
    };

    let result = gradcheck_function(sigmoid_fn, &input, &config);
    assert!(result.is_ok(), "Gradient check failed for sigmoid");

    // Test tanh gradient
    let tanh_fn =
        |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> { tanh(x) };

    let result = gradcheck_function(tanh_fn, &input, &config);
    assert!(result.is_ok(), "Gradient check failed for tanh");

    // Test ReLU gradient (note: ReLU has discontinuous gradient at 0)
    let relu_input = tensor_2d(&[&[0.5, 1.0, 2.0]]).unwrap(); // Avoid 0 for stability
    let relu_fn =
        |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> { relu(x) };

    let result = gradcheck_function(relu_fn, &relu_input, &config);
    assert!(result.is_ok(), "Gradient check failed for ReLU");
}

/// Test linear layer gradients
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_linear_layer_gradients() {
    let layer = Linear::new(3, 2, true);
    let input = tensor_2d(&[&[1.0, 2.0, 3.0]]).unwrap();

    // Test forward pass gradient
    let forward_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        layer.forward(x)
    };

    let config = GradCheckConfig {
        eps: 1e-6,
        atol: 1e-4,
        rtol: 1e-3,
        double_precision: false,
        max_elements: Some(10), // Limit for efficiency
        seed: Some(42),
    };

    let result = gradcheck_function(forward_fn, &input, &config);
    assert!(result.is_ok(), "Gradient check failed for Linear layer");
}

/// Test chain rule with composed functions
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_chain_rule_gradients() {
    let input = tensor_2d(&[&[0.5, -0.2, 1.0]]).unwrap();
    let target = tensor_2d(&[&[0.8, 0.1, 0.9]]).unwrap();

    // Test sigmoid + MSE loss chain
    let composed_fn =
        |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
            let sigmoid_out = sigmoid(x)?;
            mse_loss(&sigmoid_out, &target, "mean")
        };

    let config = GradCheckConfig {
        eps: 1e-6,
        atol: 1e-4,
        rtol: 1e-3,
        double_precision: false,
        max_elements: None,
        seed: Some(42),
    };

    let result = gradcheck_function(composed_fn, &input, &config);
    assert!(
        result.is_ok(),
        "Gradient check failed for sigmoid + MSE chain"
    );

    // Test ReLU + sigmoid chain
    let relu_sigmoid_fn =
        |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
            let relu_out = relu(x)?;
            sigmoid(&relu_out)
        };

    let result = gradcheck_function(relu_sigmoid_fn, &input, &config);
    assert!(
        result.is_ok(),
        "Gradient check failed for ReLU + sigmoid chain"
    );
}

/// Test gradients with different tensor shapes
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_gradient_shapes() {
    // Test 1D tensor
    let input_1d = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();
    let sigmoid_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        sigmoid(x)
    };

    let config = GradCheckConfig::default();
    let result = gradcheck_function(sigmoid_fn, &input_1d, &config);
    assert!(result.is_ok(), "Gradient check failed for 1D tensor");

    // Test 2D tensor (batch)
    let input_2d = tensor_2d(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    let result = gradcheck_function(sigmoid_fn, &input_2d, &config);
    assert!(result.is_ok(), "Gradient check failed for 2D tensor");

    // Test with mean reduction maintaining shape consistency
    let mean_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        x.mean(None, false)
    };

    let result = gradcheck_function(mean_fn, &input_2d, &config);
    assert!(result.is_ok(), "Gradient check failed for mean reduction");
}

/// Test numerical stability of gradient computation
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_gradient_numerical_stability() {
    // Test with small values
    let small_input = tensor_2d(&[&[1e-6, 1e-5, 1e-4]]).unwrap();

    let stable_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        // Use log(1 + x) which is more stable for small x
        let ones = ones(x.shape().dims())?;
        let one_plus_x = x.add(&ones)?;
        one_plus_x.log()
    };

    let config = GradCheckConfig {
        eps: 1e-8,
        atol: 1e-6,
        rtol: 1e-4,
        double_precision: false,
        max_elements: None,
        seed: Some(42),
    };

    let result = gradcheck_function(stable_fn, &small_input, &config);
    assert!(
        result.is_ok(),
        "Gradient check failed for numerically stable function"
    );

    // Test with values near zero (but not exactly zero to avoid ReLU discontinuity)
    let near_zero = tensor_2d(&[&[1e-3, -1e-3, 2e-3]]).unwrap();

    let elu_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        elu(x, 1.0)
    };

    let result = gradcheck_function(elu_fn, &near_zero, &config);
    assert!(result.is_ok(), "Gradient check failed for ELU near zero");
}

/// Test gradient computation for loss functions
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_loss_gradients() {
    let predictions = tensor_2d(&[&[0.7, 0.2, 0.1], &[0.1, 0.8, 0.1]]).unwrap();
    let targets = tensor_2d(&[&[0.8, 0.1, 0.1], &[0.2, 0.7, 0.1]]).unwrap();

    // Create a simple identity module for testing
    struct IdentityModule;
    impl Module for IdentityModule {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            Ok(input.clone())
        }
        fn parameters(&self) -> HashMap<String, Parameter> {
            HashMap::new()
        }
    }

    let module = IdentityModule;

    // Test MSE loss gradients
    let mse_fn = |pred: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        mse_loss(pred, &targets, "mean")
    };

    let result = gradcheck(&module, &predictions, mse_fn);
    assert!(result.is_ok(), "Gradient check failed for MSE loss");

    // Test Binary Cross Entropy gradients (ensure predictions are in valid range)
    let sigmoid_pred = sigmoid(&predictions).unwrap();
    let bce_fn = |pred: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        binary_cross_entropy(pred, &targets, None, "mean")
    };

    let result = gradcheck(&module, &sigmoid_pred, bce_fn);
    assert!(result.is_ok(), "Gradient check failed for BCE loss");
}

/// Test gradient computation with different reduction modes
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_reduction_gradients() {
    let input = tensor_2d(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    let target = tensor_2d(&[&[1.5, 1.5], &[2.5, 3.5]]).unwrap();

    // Create a simple identity module for testing
    struct IdentityModule;
    impl Module for IdentityModule {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            Ok(input.clone())
        }
        fn parameters(&self) -> HashMap<String, Parameter> {
            HashMap::new()
        }
    }

    let module = IdentityModule;

    // Test mean reduction
    let mean_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        mse_loss(x, &target, "mean")
    };
    let result = gradcheck(&module, &input, mean_fn);
    assert!(result.is_ok(), "Gradient check failed for mean reduction");

    // Test sum reduction
    let sum_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        mse_loss(x, &target, "sum")
    };
    let result = gradcheck(&module, &input, sum_fn);
    assert!(result.is_ok(), "Gradient check failed for sum reduction");

    // Test no reduction (elementwise)
    let none_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        let unreduced = mse_loss(x, &target, "none")?;
        unreduced.mean(None, false) // Reduce to scalar for gradient checking
    };
    let result = gradcheck(&module, &input, none_fn);
    assert!(result.is_ok(), "Gradient check failed for no reduction");
}

/// Test fast gradient checking for efficiency
#[test]
#[ignore = "Gradient checking requires full autograd integration - currently in development"]
fn test_fast_gradcheck() {
    let input = tensor_2d(&[&[1.0, 2.0, 3.0, 4.0, 5.0]]).unwrap();

    let sigmoid_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        sigmoid(x)
    };

    // Fast gradient check with fewer elements
    let fast_config = GradCheckConfig {
        eps: 1e-5,
        atol: 1e-3,
        rtol: 1e-2,
        double_precision: false,
        max_elements: Some(3), // Only check 3 elements
        seed: Some(42),
    };

    // Create a simple linear layer to test gradcheck with
    let linear = Linear::new(5, 1, true);
    let result = fast_gradcheck(&linear, &input, sigmoid_fn);
    assert!(result.is_ok(), "Fast gradient check failed");

    let grad_result = result.unwrap();
    assert!(grad_result.passed, "Fast gradient check should pass");
    // Check that all parameters passed
    assert!(
        grad_result.parameter_results.iter().all(|r| r.passed),
        "Some parameters failed gradient check"
    );
}
