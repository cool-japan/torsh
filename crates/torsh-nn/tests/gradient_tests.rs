//! Gradient checking and validation tests
//!
//! Tests that verify gradients are computed correctly for various operations.
//!
//! # Why some of these are still `#[ignore]`d
//!
//! The whole file used to be ignored behind "requires full autograd integration
//! - currently in development". That blanket reason is no longer true: the
//! activation family is differentiable end to end (see
//! `tests/hardening_nn_activations.rs`), and the cases below that exercise it
//! now run. Four concrete, *measured* limitations keep the rest ignored, and
//! each surviving `#[ignore]` names the one that blocks it:
//!
//! * **`gradcheck_function` only supports rank-1 inputs.**
//!   `GradChecker::compute_numerical_gradient_function` perturbs element `idx`
//!   with `input.get_item(&[idx])`, i.e. a single *flat* index, while
//!   `get_item` requires one index per dimension. Any 2-D input therefore fails
//!   with `InvalidArgument("Expected 2 indices, got 1")` before autograd is
//!   even consulted. Every test kept below therefore uses `tensor_1d`.
//! * **`gradcheck_function` requires a scalar-valued function.** It calls
//!   `output.backward()`, which rejects non-scalar outputs with
//!   `AutogradError("Gradient can only be computed for scalar outputs")`.
//!   Tensor-valued activations must be composed with a reduction first; the
//!   running tests do that with `.sum()`.
//! * **Module-based `gradcheck`/`fast_gradcheck` are deliberately
//!   unimplemented.** `GradChecker::compute_analytical_gradient` returns
//!   `TorshError::NotImplemented` on purpose (a checker that always passes is
//!   worse than none), because the `Module` trait exposes no autograd-tracked
//!   parameter access. Those cases cannot pass until that lands.
//! * **Module gradcheck over a parameter-less module checks nothing at all.**
//!   `GradChecker::check_module` initialises `all_passed = true` and then loops
//!   over `module.named_parameters()`. Hand it a module with no parameters and
//!   the loop body never runs: it returns `Ok(GradCheckResult { passed: true,
//!   parameter_results: [], .. })` without ever calling
//!   `compute_analytical_gradient`. That is why the two cases below stay
//!   ignored — not because they fail, but because they *pass* while measuring
//!   nothing. (Measured: `cargo test --test gradient_tests -- --ignored
//!   test_loss_gradients test_reduction_gradients` reports `2 passed`.)
//!
//! # Finite-difference policy
//!
//! The original configs used `eps: 1e-6`. In `f32` that is *below* the noise
//! floor of a central difference — the subtraction of two nearly-equal values
//! loses ~7 significant digits — so those checks measured rounding, not
//! gradients. The running tests use the campaign policy instead: step `1e-2`,
//! tolerance `2e-2` (relative, with an absolute floor), and probe points kept
//! clear of any kink so the difference straddles a single branch.

use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::functional::*;
use torsh_nn::gradcheck::{fast_gradcheck, gradcheck, gradcheck_function, GradCheckConfig};
use torsh_nn::layers::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::creation::ones;
use torsh_tensor::creation::{tensor_1d, tensor_2d};
use torsh_tensor::Tensor;

/// Central-difference settings mandated for this campaign: step `1e-2`, `2e-2`
/// relative tolerance, every element checked.
fn fd_config() -> GradCheckConfig {
    GradCheckConfig {
        eps: 1e-2,
        atol: 2e-2,
        rtol: 2e-2,
        double_precision: false,
        max_elements: None,
        seed: Some(42),
    }
}

/// Gradient-check a tensor-valued activation by contracting it to a scalar.
///
/// Summing is the right objective for an element-wise activation:
/// `d(sum f(x))/dx_i` is exactly `f'(x_i)`, so the check measures the
/// activation's own derivative.
///
/// The input is rebuilt from `data` on every call rather than shared between
/// checks. `GradChecker::check_function` runs `backward()` on
/// `input.clone().requires_grad_(true)`, and a cloned tensor shares its
/// gradient accumulator with the original — so re-using one tensor for two
/// checks silently adds the first activation's derivative to the second's.
/// Measured: checking `sigmoid` and then `tanh` on one shared tensor reported a
/// 0.23501348 discrepancy for `tanh`, which is exactly `sigmoid'(0.5)`.
fn check_activation<F>(label: &str, data: &[f32], f: F)
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let input = tensor_1d(data).unwrap();
    let result = gradcheck_function(|x| f(x)?.sum(), &input, &fd_config())
        .unwrap_or_else(|e| panic!("{label}: gradient check could not run: {e:?}"));
    assert!(
        result.passed,
        "{label}: gradient check failed - {} (max abs diff {}, max rel diff {})",
        result.summary,
        result
            .worst_parameter()
            .map(|p| p.max_abs_diff)
            .unwrap_or(f64::NAN),
        result
            .worst_parameter()
            .map(|p| p.max_rel_diff)
            .unwrap_or(f64::NAN),
    );
}

/// Test gradient computation for simple operations
#[test]
#[ignore = "gradcheck_function rejects the 2-D input (`Expected 2 indices, got 1`, \
            see the module docs), and mse_loss lives in functional/loss.rs which \
            another agent is editing in this same wave"]
fn test_basic_gradient_check() {
    // Test MSE loss gradient
    let input = tensor_2d(&[&[1.0, 2.0]]).unwrap();
    let target = tensor_2d(&[&[1.5, 1.5]]).unwrap();

    // Create a simple function: MSE loss
    let loss_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        mse_loss(x, &target, Reduction::Mean)
    };

    // Check gradient with default config
    let config = GradCheckConfig::default();
    let result = gradcheck_function(loss_fn, &input, &config);
    assert!(result.is_ok(), "Gradient check failed for MSE loss");

    let grad_result = result.unwrap();
    assert!(grad_result.passed, "MSE gradient check should pass");
}

/// Test gradient computation for activation functions.
///
/// Un-ignored: every one of these was previously either detached
/// (`sigmoid`, `tanh`, `gelu`, `elu`, `selu`) or attached-but-wrong
/// (`swish` 21.9%, `mish` sign-flipped) — see
/// `tests/hardening_nn_activations.rs` for the measurements. The rank-1 input
/// and the `.sum()` contraction work around the two `gradcheck_function`
/// limitations documented at the top of this file.
#[test]
fn test_activation_gradients() {
    // Smooth activations: any probe points are fine.
    const SMOOTH: [f32; 4] = [0.5, -0.5, 1.0, -1.0];
    check_activation("sigmoid", &SMOOTH, sigmoid);
    check_activation("tanh", &SMOOTH, tanh);
    check_activation("gelu", &SMOOTH, gelu);
    check_activation("swish", &SMOOTH, swish);
    check_activation("mish", &SMOOTH, mish);

    // Kinked activations: the central difference must not straddle x = 0, so
    // every probe point sits several finite-difference steps away from it.
    const KINKED: [f32; 5] = [0.5, 1.0, 2.0, -0.5, -1.5];
    check_activation("relu", &KINKED, relu);
    check_activation("leaky_relu", &KINKED, |x| leaky_relu(x, 0.01));
    check_activation("elu", &KINKED, |x| elu(x, 1.0));
    check_activation("selu", &KINKED, selu);
}

/// Test linear layer gradients
#[test]
#[ignore = "Linear::forward is tensor-valued, so gradcheck_function rejects it with \
            `Gradient can only be computed for scalar outputs`, and the 2-D input hits \
            the flat-index defect (see the module docs). Needs gradcheck.rs, which is \
            outside this wave's ownership"]
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
///
/// The activation half of this (a `relu -> sigmoid` chain) is covered by
/// `hardening_nn_activations::w4_relu_keeps_a_two_layer_chain_trainable`, which
/// finite-differences it directly instead of going through `gradcheck_function`.
#[test]
#[ignore = "2-D input hits gradcheck_function's flat-index defect (see the module docs), \
            and the sigmoid+MSE half depends on functional/loss.rs, which another agent \
            is editing in this same wave"]
fn test_chain_rule_gradients() {
    let input = tensor_2d(&[&[0.5, -0.2, 1.0]]).unwrap();
    let target = tensor_2d(&[&[0.8, 0.1, 0.9]]).unwrap();

    // Test sigmoid + MSE loss chain
    let composed_fn =
        |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
            let sigmoid_out = sigmoid(x)?;
            mse_loss(&sigmoid_out, &target, Reduction::Mean)
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

/// Test gradients across tensor lengths and through a reduction.
///
/// The 2-D case the original test carried cannot run — `gradcheck_function`
/// perturbs with a flat index (see the module docs) — so the shape coverage is
/// expressed as a length sweep that also crosses `Tensor`'s `numel > 100`
/// parallel dispatch band, which is the boundary that actually changes which
/// kernel evaluates the activation.
#[test]
fn test_gradient_shapes() {
    for len in [1usize, 3, 8, 128] {
        let data: Vec<f32> = (0..len).map(|i| 0.25 + 0.05 * (i as f32)).collect();
        check_activation(&format!("sigmoid @ len {len}"), &data, sigmoid);
    }

    // A reduction is already scalar-valued, so it needs no `.sum()` wrapper.
    let input = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = gradcheck_function(|x| x.mean(None, false), &input, &fd_config())
        .expect("mean reduction gradient check should run");
    assert!(
        result.passed,
        "Gradient check failed for mean reduction - {}",
        result.summary
    );
}

/// Test numerical stability of gradient computation.
///
/// The original probe points (`1e-6`..`1e-4`, and `+/-1e-3` around ELU's kink)
/// were an order of magnitude *inside* the `1e-2` finite-difference step, so a
/// central difference there straddles both branches of ELU and measures neither
/// derivative. The stability property being checked — `log(1 + x)` for small
/// `x`, and ELU on each side of the origin — is preserved by moving the points
/// out to where the difference is meaningful.
#[test]
fn test_gradient_numerical_stability() {
    let small_input = tensor_1d(&[0.05, 0.1, 0.25]).unwrap();
    let result = gradcheck_function(
        |x| {
            // log(1 + x), which stays well-conditioned for small x.
            let ones = ones(x.shape().dims())?;
            x.add(&ones)?.log()?.sum()
        },
        &small_input,
        &fd_config(),
    )
    .expect("log(1 + x) gradient check should run");
    assert!(
        result.passed,
        "Gradient check failed for numerically stable function - {}",
        result.summary
    );

    // ELU on each side of its kink, clear of the finite-difference step.
    check_activation("elu near zero", &[0.2, -0.2, 0.6, -0.6], |x| elu(x, 1.0));
}

/// Test gradient computation for loss functions
///
/// NOTE: this asserts only `result.is_ok()`, and the checker it calls never
/// looks at a gradient: `IdentityModule` declares no parameters, so
/// `GradChecker::check_module` loops over an empty `named_parameters()` map,
/// leaves `all_passed` at its `true` initial value, and returns
/// `Ok(GradCheckResult { passed: true, parameter_results: [], summary: "All 0
/// parameters passed gradient check" })`. Un-ignoring it would therefore be a
/// *vacuous* green — and tightening the assertion to `result.passed` would not
/// help, because that is `true` too. The test only becomes meaningful once the
/// module under check owns parameters, at which point it runs into the
/// third module-doc limitation (`compute_analytical_gradient` is deliberately
/// `NotImplemented`).
#[test]
#[ignore = "vacuous: IdentityModule declares no parameters, so \
            GradChecker::check_module iterates an empty named_parameters() map and returns \
            passed: true with an empty parameter_results — compute_analytical_gradient is \
            never reached, so nothing about mse_loss/binary_cross_entropy is actually checked"]
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
        mse_loss(pred, &targets, Reduction::Mean)
    };

    let result = gradcheck(&module, &predictions, mse_fn);
    assert!(result.is_ok(), "Gradient check failed for MSE loss");

    // Test Binary Cross Entropy gradients (ensure predictions are in valid range)
    let sigmoid_pred = sigmoid(&predictions).unwrap();
    let bce_fn = |pred: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        binary_cross_entropy(pred, &targets, None, Reduction::Mean)
    };

    let result = gradcheck(&module, &sigmoid_pred, bce_fn);
    assert!(result.is_ok(), "Gradient check failed for BCE loss");
}

/// Test gradient computation with different reduction modes
///
/// Same vacuous-green hazard as [`test_loss_gradients`], and for the same
/// reason: the assertions are `is_ok()` against a checker that was handed a
/// parameter-less `IdentityModule`, so it checks zero parameters and reports
/// success without ever computing a gradient.
#[test]
#[ignore = "vacuous: IdentityModule declares no parameters, so \
            GradChecker::check_module iterates an empty named_parameters() map and returns \
            passed: true with an empty parameter_results — compute_analytical_gradient is \
            never reached, so none of the three mse_loss reductions is actually checked"]
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
        mse_loss(x, &target, Reduction::Mean)
    };
    let result = gradcheck(&module, &input, mean_fn);
    assert!(result.is_ok(), "Gradient check failed for mean reduction");

    // Test sum reduction
    let sum_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        mse_loss(x, &target, Reduction::Mean)
    };
    let result = gradcheck(&module, &input, sum_fn);
    assert!(result.is_ok(), "Gradient check failed for sum reduction");

    // Test no reduction (elementwise)
    let none_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        let unreduced = mse_loss(x, &target, Reduction::None)?;
        unreduced.mean(None, false) // Reduce to scalar for gradient checking
    };
    let result = gradcheck(&module, &input, none_fn);
    assert!(result.is_ok(), "Gradient check failed for no reduction");
}

/// Test fast gradient checking for efficiency
#[test]
#[ignore = "module gradcheck is deliberately unimplemented (GradChecker::\
            compute_analytical_gradient returns NotImplemented because the Module trait \
            exposes no autograd-tracked parameter access), so `grad_result.passed` is \
            false by construction for every parameter of the Linear layer"]
fn test_fast_gradcheck() {
    let input = tensor_2d(&[&[1.0, 2.0, 3.0, 4.0, 5.0]]).unwrap();

    let sigmoid_fn = |x: &torsh_tensor::Tensor| -> torsh_core::error::Result<torsh_tensor::Tensor> {
        sigmoid(x)
    };

    // Fast gradient check with fewer elements
    let _fast_config = GradCheckConfig {
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
