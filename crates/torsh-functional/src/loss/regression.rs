//! Regression loss functions
//!
//! This module provides loss functions commonly used for regression tasks,
//! including mean squared error, L1 loss, and other regression-specific losses.

use crate::loss::common::ReductionType;
use crate::utils::{
    function_context, validate_elementwise_shapes, validate_non_empty, validate_positive,
};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Mean Squared Error loss
///
/// Measures the mean squared error between input and target tensors.
/// This is commonly used for regression tasks.
///
/// Formula: MSE(x, y) = (x - y)²
///
/// # Parameters
/// * `input` - Input tensor of any shape
/// * `target` - Target tensor with same shape as input
/// * `reduction` - Reduction type: None, Mean, or Sum
///
/// # Returns
/// Loss tensor with specified reduction applied
///
/// # Errors
/// Returns error if:
/// - Input or target tensors are empty
/// - Input and target shapes don't match
/// - Tensor operations fail
///
/// # Example
/// ```rust
/// use torsh_functional::loss::{mse_loss, ReductionType};
/// let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// let target = Tensor::from_vec(vec![1.5, 2.5, 2.5], &[3]).unwrap();
/// let loss = mse_loss(&input, &target, ReductionType::Mean).unwrap();
/// ```
pub fn mse_loss(input: &Tensor, target: &Tensor, reduction: ReductionType) -> TorshResult<Tensor> {
    let context = function_context("mse_loss");

    // Comprehensive input validation
    validate_non_empty(input, &context)?;
    validate_non_empty(target, &context)?;
    validate_elementwise_shapes(input, target)?;

    // Compute MSE
    let diff = input.sub(target).map_err(|e| {
        TorshError::config_error_with_context(
            &format!("Failed to compute input - target: {}", e),
            &context,
        )
    })?;

    let squared = diff.pow_scalar(2.0).map_err(|e| {
        TorshError::config_error_with_context(
            &format!("Failed to square differences: {}", e),
            &context,
        )
    })?;

    reduction.apply(squared).map_err(|e| {
        TorshError::config_error_with_context(
            &format!("Failed to apply reduction: {}", e),
            &context,
        )
    })
}

/// L1 Loss (Mean Absolute Error)
///
/// Creates a criterion that measures the mean absolute error between
/// input and target.
///
/// # Arguments
/// * `input` - Input tensor
/// * `target` - Target tensor with same shape as input
/// * `reduction` - Specifies the reduction to apply to the output
///
/// # Returns
/// Loss tensor with reduction applied
pub fn l1_loss(input: &Tensor, target: &Tensor, reduction: ReductionType) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input, target)?;
    let diff = input.sub(target)?;
    let abs_diff = diff.abs()?;
    reduction.apply(abs_diff)
}

/// Smooth L1 Loss (Huber Loss)
///
/// Creates a criterion that uses a squared term if the absolute element-wise error falls below beta
/// and an L1 term otherwise.
///
/// # Arguments
/// * `input` - Input tensor
/// * `target` - Target tensor with same shape as input
/// * `reduction` - Specifies the reduction to apply to the output
/// * `beta` - Threshold at which to change between L1 and L2 loss
///
/// # Returns
/// Loss tensor with reduction applied
pub fn smooth_l1_loss(
    input: &Tensor,
    target: &Tensor,
    reduction: ReductionType,
    beta: f32,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input, target)?;
    validate_positive(beta, "beta", "smooth_l1_loss")?;

    let diff = input.sub(target)?;
    let abs_diff = diff.abs()?;

    // Create mask for |diff| < beta
    let mask = abs_diff.lt_scalar(beta)?;

    // Smooth L1: 0.5 * diff^2 / beta for |diff| < beta, |diff| - 0.5 * beta otherwise
    let l2_component = diff.pow_scalar(2.0)?.div_scalar(2.0 * beta)?;
    let l1_component = abs_diff.sub_scalar(0.5 * beta)?;

    let smooth_l1 = l2_component.where_tensor(&mask, &l1_component)?;
    reduction.apply(smooth_l1)
}

/// Poisson Negative Log Likelihood Loss
///
/// Computes the Poisson negative log likelihood loss between input and target.
///
/// # Arguments
/// * `log_input` - Tensor containing log of expected value
/// * `target` - Tensor containing expected value (Poisson distribution parameter)
/// * `log_input_is_log` - If true, log_input is already log. If false, take log first.
/// * `full` - Whether to compute full loss (including constant term)
/// * `size_average` - Deprecated, use reduction instead
/// * `eps` - Small value to avoid log(0)
/// * `reduction` - Reduction type
///
/// # Returns
/// Computed loss tensor
pub fn poisson_nll_loss(
    log_input: &Tensor,
    target: &Tensor,
    log_input_is_log: bool,
    full: bool,
    _size_average: Option<bool>,
    eps: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(log_input, target)?;

    let input = if log_input_is_log {
        log_input.clone()
    } else {
        log_input.log()?
    };

    // Poisson NLL: exp(log_input) - target * log_input
    let exp_input = input.exp()?;
    let target_log_input = target.mul(&input)?;
    let mut loss = exp_input.sub(&target_log_input)?;

    if full {
        // Add stirling approximation: target * log(target) - target + 0.5 * log(2π * target)
        let target_safe = target.add_scalar(eps)?;
        let target_log_target = target.mul(&target_safe.log()?)?;
        let stirling = target_log_target
            .sub(target)?
            .add_scalar(0.5 * (2.0 * std::f32::consts::PI).ln())?
            .add(&target_safe.log()?.mul_scalar(0.5)?)?;
        loss = loss.add(&stirling)?;
    }

    reduction.apply(loss)
}

/// Gaussian Negative Log Likelihood Loss
///
/// Computes the negative log likelihood of a Gaussian distribution.
///
/// # Arguments
/// * `input` - Tensor containing predicted mean
/// * `target` - Target tensor
/// * `var` - Tensor containing variance (must be positive)
/// * `full` - Whether to include constant term
/// * `eps` - Small value for numerical stability
/// * `reduction` - Reduction type
///
/// # Returns
/// Computed loss tensor
pub fn gaussian_nll_loss(
    input: &Tensor,
    target: &Tensor,
    var: &Tensor,
    full: bool,
    eps: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input, target)?;
    validate_elementwise_shapes(input, var)?;

    // Ensure variance is positive
    let var_safe = var.add_scalar(eps)?;

    // NLL = 0.5 * ((input - target)² / var + log(var))
    let diff = input.sub(target)?;
    let diff_squared = diff.pow_scalar(2.0)?;
    let normalized_diff = diff_squared.div(&var_safe)?;
    let log_var = var_safe.log()?;
    let mut loss = normalized_diff.add(&log_var)?.mul_scalar(0.5)?;

    if full {
        // Add constant term: 0.5 * log(2π)
        loss = loss.add_scalar(0.5 * (2.0 * std::f32::consts::PI).ln())?;
    }

    reduction.apply(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_mse_loss_basic() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let target = from_vec(vec![1.5, 2.5, 2.5], &[3], DeviceType::Cpu)?;

        let loss = mse_loss(&input, &target, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Expected: ((1.0-1.5)² + (2.0-2.5)² + (3.0-2.5)²) / 3 = (0.25 + 0.25 + 0.25) / 3 = 0.25
        assert!((loss_value - 0.25).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_l1_loss_basic() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let target = from_vec(vec![1.5, 2.5, 2.5], &[3], DeviceType::Cpu)?;

        let loss = l1_loss(&input, &target, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Expected: (|1.0-1.5| + |2.0-2.5| + |3.0-2.5|) / 3 = (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert!((loss_value - 0.5).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_smooth_l1_loss_basic() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let target = from_vec(vec![1.1, 2.1, 4.0], &[3], DeviceType::Cpu)?; // diff = [0.1, 0.1, 1.0]

        let loss = smooth_l1_loss(&input, &target, ReductionType::Mean, 1.0)?;
        let loss_value = loss.item()?;

        // For beta=1.0:
        // |0.1| < 1.0 -> 0.5 * 0.1² / 1.0 = 0.005
        // |0.1| < 1.0 -> 0.5 * 0.1² / 1.0 = 0.005
        // |1.0| = 1.0 -> 1.0 - 0.5 * 1.0 = 0.5
        // Mean = (0.005 + 0.005 + 0.5) / 3 ≈ 0.17
        assert!((loss_value - 0.17).abs() < 1e-2);
        Ok(())
    }

    #[test]
    fn test_mse_loss_zero_when_equal() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let target = input.clone();

        let loss = mse_loss(&input, &target, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        assert!(loss_value.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_l1_loss_zero_when_equal() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu)?;
        let target = input.clone();

        let loss = l1_loss(&input, &target, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        assert!(loss_value.abs() < 1e-6);
        Ok(())
    }
}
