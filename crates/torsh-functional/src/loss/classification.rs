//! Classification loss functions
//!
//! This module provides loss functions commonly used for classification tasks,
//! including cross entropy, negative log likelihood, binary cross entropy, and focal loss.

use crate::loss::common::ReductionType;
use crate::utils::{function_context, validate_elementwise_shapes, validate_range};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Cross Entropy Loss
///
/// This criterion computes the cross entropy loss between input logits and target.
/// This criterion combines log_softmax and nll_loss in a single function.
pub fn cross_entropy(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: &str,
    ignore_index: Option<i64>,
    label_smoothing: f64,
) -> TorshResult<Tensor> {
    // Apply label smoothing if needed
    if label_smoothing > 0.0 {
        return cross_entropy_with_label_smoothing(
            input,
            target,
            label_smoothing,
            weight,
            reduction,
            ignore_index,
        );
    }

    // Apply log_softmax to input along the last dimension
    let dim = (input.shape().ndim() - 1) as i32;
    let log_probs = input.log_softmax(dim)?;

    // Call nll_loss
    nll_loss(&log_probs, target, weight, reduction, ignore_index)
}

/// Negative Log Likelihood Loss
///
/// The negative log likelihood loss for classification.
pub fn nll_loss(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: &str,
    ignore_index: Option<i64>,
) -> TorshResult<Tensor> {
    // input shape: (N, C) or (N, C, d1, d2, ...)
    // target shape: (N) or (N, d1, d2, ...)

    // For now, only handle the simple 2D case without weight or ignore_index
    if input.ndim() != 2 || target.ndim() != 1 {
        return Err(TorshError::UnsupportedOperation {
            op: "nll_loss with >2D input".to_string(),
            dtype: "tensor".to_string(),
        });
    }

    if weight.is_some() || ignore_index.is_some() {
        return Err(TorshError::UnsupportedOperation {
            op: "nll_loss with weight or ignore_index".to_string(),
            dtype: "tensor".to_string(),
        });
    }

    // Gather operation: for each sample, select the log probability for the target class
    let batch_size = input.shape().dims()[0];
    let mut losses = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let target_class = target.get(&[i])? as usize;
        let loss_val = -input.get(&[i, target_class])?;
        losses.push(loss_val);
    }

    let loss_tensor = Tensor::from_vec(losses, &[batch_size])?;

    // Apply reduction
    match reduction {
        "none" => Ok(loss_tensor),
        "mean" => loss_tensor.mean(None, false),
        "sum" => loss_tensor.sum(),
        _ => Err(TorshError::InvalidArgument(format!(
            "Unknown reduction: {}",
            reduction
        ))),
    }
}

/// Binary Cross Entropy Loss
///
/// Creates a criterion that measures the binary cross entropy loss between input and target.
pub fn binary_cross_entropy(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input, target)?;

    // Clamp input to prevent log(0)
    let eps = 1e-8_f32;
    let input_clamped = input.clamp(eps, 1.0 - eps)?;

    // BCE = -[target * log(input) + (1 - target) * log(1 - input)]
    let log_input = input_clamped.log()?;
    let log_one_minus_input = input_clamped.neg()?.add_scalar(1.0)?.log()?;

    let positive_loss = target.mul(&log_input)?;
    let one_minus_target = target.neg()?.add_scalar(1.0)?;
    let negative_loss = one_minus_target.mul(&log_one_minus_input)?;

    let mut loss = positive_loss.add(&negative_loss)?.neg()?;

    // Apply weight if provided
    if let Some(w) = weight {
        validate_elementwise_shapes(&loss, w)?;
        loss = loss.mul(w)?;
    }

    reduction.apply(loss)
}

/// Binary Cross Entropy with Logits Loss
///
/// This loss combines a Sigmoid layer and the Binary Cross Entropy Loss in one single class.
/// It is more numerically stable than using a plain Sigmoid followed by a BCE loss.
pub fn binary_cross_entropy_with_logits(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: ReductionType,
    pos_weight: Option<&Tensor>,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input, target)?;

    // Use the numerically stable formula:
    // loss = max(input, 0) - input * target + log(1 + exp(-abs(input)))
    let zero = Tensor::zeros_like(input)?;
    let max_input = input.maximum(&zero)?;
    let input_target = input.mul(target)?;
    let abs_input = input.abs()?;
    let log_term = abs_input.neg()?.exp()?.add_scalar(1.0)?.log()?;

    let mut loss = max_input.sub(&input_target)?.add(&log_term)?;

    // Apply positive weight if provided
    if let Some(pos_w) = pos_weight {
        let pos_weight_term = target.mul(pos_w)?.add_scalar(1.0)?.sub(target)?;
        loss = loss.mul(&pos_weight_term)?;
    }

    // Apply weight if provided
    if let Some(w) = weight {
        validate_elementwise_shapes(&loss, w)?;
        loss = loss.mul(w)?;
    }

    reduction.apply(loss)
}

/// Multi-class margin loss
///
/// Creates a criterion that optimizes multi-class classification margin loss.
pub fn multi_margin_loss(
    input: &Tensor,
    target: &Tensor,
    p: i64,
    margin: f32,
    weight: Option<&Tensor>,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    let context = function_context("multi_margin_loss");

    if input.ndim() != 2 || target.ndim() != 1 {
        return Err(TorshError::config_error_with_context(
            "multi_margin_loss expects 2D input and 1D target",
            &context,
        ));
    }

    if p != 1 && p != 2 {
        return Err(TorshError::config_error_with_context(
            "multi_margin_loss only supports p=1 or p=2",
            &context,
        ));
    }

    let batch_size = input.shape().dims()[0];
    let num_classes = input.shape().dims()[1];
    let mut losses = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let target_class = target.get(&[i])? as usize;
        let target_score = input.get(&[i, target_class])?;

        let mut sample_loss = 0.0;
        for j in 0..num_classes {
            if j != target_class {
                let score_j = input.get(&[i, j])?;
                let margin_violation = margin - target_score + score_j;
                if margin_violation > 0.0 {
                    sample_loss += if p == 1 {
                        margin_violation
                    } else {
                        margin_violation.powi(2)
                    };
                }
            }
        }

        // Apply class weight if provided
        if let Some(w) = weight {
            let class_weight = w.get(&[target_class])?;
            sample_loss *= class_weight;
        }

        losses.push(sample_loss / (num_classes - 1) as f32);
    }

    let loss_tensor = Tensor::from_vec(losses, &[batch_size])?;
    reduction.apply(loss_tensor)
}

/// Focal Loss
///
/// Addresses class imbalance by down-weighting easy examples and focusing on hard examples.
///
/// Formula: Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
pub fn focal_loss(
    input: &Tensor,
    target: &Tensor,
    alpha: f32,
    gamma: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_range(alpha, 0.0, 1.0, "alpha", "focal_loss")?;
    validate_range(gamma, 0.0, 5.0, "gamma", "focal_loss")?;

    // Apply softmax to get probabilities
    let dim = (input.shape().ndim() - 1) as i32;
    let probs = input.softmax(dim)?;

    // Get log probabilities for numerical stability
    let log_probs = probs.log()?;

    // For each sample, get the probability of the target class
    let batch_size = target.shape().dims()[0];
    let mut focal_losses = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let target_class = target.get(&[i])? as usize;

        let p_t = probs.get(&[i, target_class])?;
        let log_p_t = log_probs.get(&[i, target_class])?;

        // Focal loss: -alpha * (1-p_t)^gamma * log(p_t)
        let focal_weight = alpha * (1.0 - p_t).powf(gamma);
        let focal_loss = -focal_weight * log_p_t;

        focal_losses.push(focal_loss);
    }

    let loss_tensor = Tensor::from_vec(focal_losses, &[batch_size])?;
    reduction.apply(loss_tensor)
}

/// Cross entropy loss with label smoothing
///
/// Applies label smoothing to the target before computing cross entropy loss.
///
/// Smoothed labels: y_smooth = (1 - smoothing) * y_true + smoothing / num_classes
pub fn cross_entropy_with_label_smoothing(
    input: &Tensor,
    target: &Tensor,
    label_smoothing: f64,
    weight: Option<&Tensor>,
    reduction: &str,
    ignore_index: Option<i64>,
) -> TorshResult<Tensor> {
    if label_smoothing < 0.0 || label_smoothing >= 1.0 {
        return Err(TorshError::InvalidArgument(
            "label_smoothing must be in [0.0, 1.0)".to_string(),
        ));
    }

    let num_classes = input.shape().dims()[input.shape().ndim() - 1];
    let smoothing_value = label_smoothing as f32 / num_classes as f32;
    let confidence = 1.0 - label_smoothing as f32;

    // Apply log softmax
    let dim = (input.shape().ndim() - 1) as i32;
    let log_probs = input.log_softmax(dim)?;

    // Create smoothed target distribution
    let batch_size = target.shape().dims()[0];
    let mut smooth_targets = vec![smoothing_value; batch_size * num_classes];

    // Set confidence for true classes
    for i in 0..batch_size {
        let target_class = target.get(&[i])? as usize;
        smooth_targets[i * num_classes + target_class] = confidence + smoothing_value;
    }

    let smooth_target_tensor = Tensor::from_vec(smooth_targets, &[batch_size, num_classes])?;

    // Compute negative log likelihood with smooth targets
    let loss = log_probs
        .mul(&smooth_target_tensor)?
        .neg()?
        .sum_dim(&[1], false)?;

    // Apply weight if provided
    let loss = if let Some(w) = weight {
        // Apply class weights based on original target
        let mut weighted_losses = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let target_class = target.get(&[i])? as usize;
            let class_weight = w.get(&[target_class])?;
            let sample_loss = loss.get(&[i])?;
            weighted_losses.push(sample_loss * class_weight);
        }
        Tensor::from_vec(weighted_losses, &[batch_size])?
    } else {
        loss.squeeze(1)?
    };

    // Apply reduction
    match reduction {
        "none" => Ok(loss),
        "mean" => loss.mean(None, false),
        "sum" => loss.sum(),
        _ => Err(TorshError::InvalidArgument(format!(
            "Unknown reduction: {}",
            reduction
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_binary_cross_entropy_basic() -> TorshResult<()> {
        let input = from_vec(vec![0.8, 0.2, 0.9], &[3], DeviceType::Cpu)?;
        let target = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu)?;

        let loss = binary_cross_entropy(&input, &target, None, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // BCE should be a positive value
        assert!(loss_value > 0.0);
        Ok(())
    }

    #[test]
    fn test_focal_loss_basic() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 0.5, 3.0, 1.5, 0.8], &[2, 3], DeviceType::Cpu)?;
        let target = from_vec(vec![1.0, 2.0], &[2], DeviceType::Cpu)?; // Class indices

        let loss = focal_loss(&input, &target, 0.25, 2.0, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Focal loss should be a positive value
        assert!(loss_value > 0.0);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_simple() -> TorshResult<()> {
        let input = from_vec(vec![1.0, 2.0, 0.5], &[1, 3], DeviceType::Cpu)?;
        let target = from_vec(vec![1.0], &[1], DeviceType::Cpu)?; // Class 1

        let loss = cross_entropy(&input, &target, None, "mean", None, 0.0)?;
        let loss_value = loss.item()?;

        // Cross entropy should be positive
        assert!(loss_value > 0.0);
        Ok(())
    }
}
