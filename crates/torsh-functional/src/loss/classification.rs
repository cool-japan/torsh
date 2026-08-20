//! Classification loss functions
//!
//! This module provides loss functions commonly used for classification tasks,
//! including cross entropy, negative log likelihood, binary cross entropy, and focal loss.

use crate::loss::common::ReductionType;
use crate::loss::common::ReductionExt;
use crate::utils::{function_context, safe_log_prob, validate_elementwise_shapes, validate_range};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Sum a `[N, C]` tensor along its class axis while keeping the autograd graph.
///
/// `Tensor::sum_dim` records `Operation::SumDim`, so the reduction stays
/// differentiable. (This helper used to route through a matmul against a
/// constant column of ones because `sum_dim` once rebuilt its result from raw
/// data and severed the graph; that workaround is no longer needed.)
fn row_sum(input: &Tensor) -> TorshResult<Tensor> {
    let dims_binding = input.shape();
    let dims = dims_binding.dims();
    if dims.len() != 2 {
        return Err(TorshError::InvalidArgument(format!(
            "row_sum expects a 2-D tensor, got {}-D",
            dims.len()
        )));
    }
    input.sum_dim(&[1], false)
}

/// Read the target class index of sample `index`, validating it against `num_classes`.
fn target_class(
    target: &Tensor,
    index: usize,
    num_classes: usize,
    context: &str,
) -> TorshResult<i64> {
    let raw = target.get(&[index])?;
    if !raw.is_finite() || raw.fract() != 0.0 {
        return Err(TorshError::InvalidArgument(format!(
            "{context}: target[{index}] = {raw} is not an integral class index"
        )));
    }
    let class = raw as i64;
    if class < 0 || class as usize >= num_classes {
        return Err(TorshError::InvalidArgument(format!(
            "{context}: target[{index}] = {class} is out of range for {num_classes} classes"
        )));
    }
    Ok(class)
}

/// Look up the loss weight of a class, defaulting to 1.0 when no weights are given.
fn class_weight(weight: Option<&Tensor>, class: usize, context: &str) -> TorshResult<f32> {
    match weight {
        Some(w) => {
            if w.numel() <= class {
                return Err(TorshError::InvalidArgument(format!(
                    "{context}: weight tensor has {} entries, class {class} is out of range",
                    w.numel()
                )));
            }
            w.get(&[class])
        }
        None => Ok(1.0),
    }
}

/// Validate that `input` is `[N, C]`, `target` is `[N]`, and return `(N, C)`.
fn validate_classification_shapes(
    input: &Tensor,
    target: &Tensor,
    context: &str,
) -> TorshResult<(usize, usize)> {
    if input.ndim() != 2 || target.ndim() != 1 {
        return Err(TorshError::InvalidArgument(format!(
            "{context} expects a 2-D input [N, C] and a 1-D target [N], got {}-D and {}-D",
            input.ndim(),
            target.ndim()
        )));
    }
    let input_dims_binding = input.shape();
    let input_dims = input_dims_binding.dims();
    let batch_size = input_dims[0];
    if target.shape().dims()[0] != batch_size {
        return Err(TorshError::ShapeMismatch {
            expected: vec![batch_size],
            got: target.shape().dims().to_vec(),
        });
    }
    Ok((batch_size, input_dims[1]))
}

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
/// The negative log likelihood loss for classification. `input` holds
/// log-probabilities of shape `[N, C]` (typically the output of
/// [`Tensor::log_softmax`]) and `target` holds the class index of every sample.
///
/// The target log-probability is gathered with a constant one-hot selector and a
/// matrix product, so the returned loss stays attached to `input` in the autograd
/// graph and `backward()` reaches it.
///
/// # Arguments
/// * `input` - Log-probabilities, shape `[N, C]`
/// * `target` - Class indices, shape `[N]`
/// * `weight` - Optional per-class rescaling weight, shape `[C]`
/// * `reduction` - `"none"`, `"mean"` (weighted average) or `"sum"`
/// * `ignore_index` - Optional class index that contributes no loss and no weight
pub fn nll_loss(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: &str,
    ignore_index: Option<i64>,
) -> TorshResult<Tensor> {
    let context = "nll_loss";
    if !matches!(reduction, "none" | "mean" | "sum") {
        return Err(TorshError::InvalidArgument(format!(
            "Unknown reduction: {}",
            reduction
        )));
    }
    let (batch_size, num_classes) = validate_classification_shapes(input, target, context)?;

    // Build the negated (and weighted) one-hot selector as a constant tensor:
    // loss_i = sum_c selector[i, c] * input[i, c] = -w_{t_i} * input[i, t_i].
    let mut selector = vec![0.0f32; batch_size * num_classes];
    let mut total_weight = 0.0f32;
    for i in 0..batch_size {
        let class = target_class(target, i, num_classes, context)?;
        if Some(class) == ignore_index {
            continue;
        }
        let w = class_weight(weight, class as usize, context)?;
        selector[i * num_classes + class as usize] = -w;
        total_weight += w;
    }

    // "mean" is the weight-normalised average, matching PyTorch. Folding the
    // normalisation into the constant selector keeps the graph intact, because
    // scalar division is not a differentiable tensor operation here.
    if reduction == "mean" {
        let scale = if total_weight == 0.0 {
            0.0
        } else {
            1.0 / total_weight
        };
        for value in selector.iter_mut() {
            *value *= scale;
        }
    }

    let selector_tensor =
        Tensor::from_data(selector, vec![batch_size, num_classes], input.device())?;
    let per_sample = row_sum(&input.mul(&selector_tensor)?)?;

    match reduction {
        "none" => Ok(per_sample),
        // The per-sample terms already carry the 1/sum(w) factor for "mean".
        "mean" | "sum" => per_sample.sum(),
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

    // BCE = -[target * log(input) + (1 - target) * log(1 - input)]
    // Use safe_log_prob to prevent log(0) with proper clamping
    let log_input = safe_log_prob(input, None)?;
    let one_minus_input = input.neg()?.add_scalar(1.0)?;
    let log_one_minus_input = safe_log_prob(&one_minus_input, None)?;

    let positive_loss = target.mul(&log_input)?;
    let one_minus_target = target.neg()?.add_scalar(1.0)?;
    let negative_loss = one_minus_target.mul(&log_one_minus_input)?;

    let mut loss = positive_loss.add(&negative_loss)?.neg()?;

    // Apply weight if provided
    if let Some(w) = weight {
        validate_elementwise_shapes(&loss, w)?;
        loss = loss.mul(w)?;
    }

    reduction.apply(&loss, None)
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

    // Numerically stable decomposition, matching
    // `torch.nn.functional.binary_cross_entropy_with_logits`:
    //
    //   loss = (1 - t) * x + log_weight * (log(1 + exp(-|x|)) + max(-x, 0))
    //
    // With `log_weight = 1 + (pos_weight - 1) * t` only the log-sigmoid part is
    // rescaled. Scaling the whole expression (which is what multiplying
    // `max(x,0) - x*t + log(1+exp(-|x|))` by `log_weight` does) also rescales the
    // `(1 - t) * x` term, which diverges from PyTorch for fractional targets.
    let zero = Tensor::zeros_like(input)?;
    let one_minus_target = target.neg()?.add_scalar(1.0)?;
    let linear_term = one_minus_target.mul(input)?; // (1 - t) * x
    let abs_input = input.abs()?;
    let softplus_term = abs_input.neg()?.exp()?.add_scalar(1.0)?.log()?; // log(1 + exp(-|x|))
    let max_neg_input = input.neg()?.maximum(&zero)?; // max(-x, 0)
    let log_sigmoid_term = softplus_term.add(&max_neg_input)?;

    let mut loss = match pos_weight {
        Some(pos_w) => {
            // log_weight = t * pos_weight + 1 - t
            let log_weight = target.mul(pos_w)?.add_scalar(1.0)?.sub(target)?;
            linear_term.add(&log_weight.mul(&log_sigmoid_term)?)?
        }
        None => linear_term.add(&log_sigmoid_term)?,
    };

    // Apply weight if provided
    if let Some(w) = weight {
        validate_elementwise_shapes(&loss, w)?;
        loss = loss.mul(w)?;
    }

    reduction.apply(&loss, None)
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

    let (batch_size, num_classes) = validate_classification_shapes(input, target, &context)?;
    if num_classes < 2 {
        return Err(TorshError::config_error_with_context(
            "multi_margin_loss requires at least two classes",
            &context,
        ));
    }

    // margins[i, j] = margin - x[i, t_i] + x[i, j] is built from tensor operations
    // so the loss stays differentiable: the target score is gathered with a
    // constant one-hot selector and broadcast back over the class axis.
    let mut one_hot = vec![0.0f32; batch_size * num_classes];
    let mut sample_scale = Vec::with_capacity(batch_size);
    let mut target_classes = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let class = target_class(target, i, num_classes, &context)? as usize;
        one_hot[i * num_classes + class] = 1.0;
        target_classes.push(class);
        let w = class_weight(weight, class, &context)?;
        sample_scale.push(w / (num_classes - 1) as f32);
    }
    let one_hot_tensor = Tensor::from_data(one_hot, vec![batch_size, num_classes], input.device())?;

    let target_scores = row_sum(&input.mul(&one_hot_tensor)?)?.view(&[batch_size as i32, 1])?;
    let margin_tensor = Tensor::from_data(
        vec![margin; batch_size * num_classes],
        vec![batch_size, num_classes],
        input.device(),
    )?;
    let violations = input.sub(&target_scores)?.add(&margin_tensor)?;

    // Zero out the target column and every non-violating entry. The mask is a
    // constant, which is exactly the (sub)gradient support of `max(0, .)`.
    let violation_data = violations.data()?;
    let mut mask = vec![0.0f32; batch_size * num_classes];
    for i in 0..batch_size {
        for j in 0..num_classes {
            let index = i * num_classes + j;
            if j != target_classes[i] && violation_data[index] > 0.0 {
                mask[index] = 1.0;
            }
        }
    }
    let mask_tensor = Tensor::from_data(mask, vec![batch_size, num_classes], input.device())?;
    let hinged = violations.mul(&mask_tensor)?;
    let hinged = if p == 1 {
        hinged
    } else {
        hinged.pow_scalar(2.0)?
    };

    let scale_tensor = Tensor::from_data(sample_scale, vec![batch_size], input.device())?;
    let per_sample = row_sum(&hinged)?.mul(&scale_tensor)?;
    reduction.apply(&per_sample, None)
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
    let context = "focal_loss";
    let (batch_size, num_classes) = validate_classification_shapes(input, target, context)?;

    // log-probabilities first (numerically stable), probabilities derived from them
    let dim = (input.shape().ndim() - 1) as i32;
    let log_probs = input.log_softmax(dim)?;
    let probs = log_probs.exp()?;

    // Gather the target entries with a constant one-hot selector so that the
    // modulating factor and the log term both remain tensor expressions.
    let mut one_hot = vec![0.0f32; batch_size * num_classes];
    for i in 0..batch_size {
        let class = target_class(target, i, num_classes, context)? as usize;
        one_hot[i * num_classes + class] = 1.0;
    }
    let one_hot_tensor = Tensor::from_data(one_hot, vec![batch_size, num_classes], input.device())?;

    let p_t = row_sum(&probs.mul(&one_hot_tensor)?)?;
    let log_p_t = row_sum(&log_probs.mul(&one_hot_tensor)?)?;

    // Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
    let ones = Tensor::from_data(vec![1.0f32; batch_size], vec![batch_size], input.device())?;
    let modulating = ones.sub(&p_t)?.pow_scalar(gamma)?;
    let neg_alpha = Tensor::from_data(vec![-alpha; batch_size], vec![batch_size], input.device())?;
    let loss_tensor = modulating.mul(&log_p_t)?.mul(&neg_alpha)?;

    reduction.apply(&loss_tensor, None)
}

/// Cross entropy loss with label smoothing
///
/// Applies label smoothing to the target before computing cross entropy loss.
///
/// Smoothed labels: y_smooth = (1 - smoothing) * y_true + smoothing / num_classes
///
/// `weight` rescales each sample by the weight of its target class and `mean`
/// divides by the sum of those weights, matching PyTorch. `ignore_index` drops the
/// matching samples from both the loss and the normalisation. The result is a
/// single differentiable expression in `input`.
pub fn cross_entropy_with_label_smoothing(
    input: &Tensor,
    target: &Tensor,
    label_smoothing: f64,
    weight: Option<&Tensor>,
    reduction: &str,
    ignore_index: Option<i64>,
) -> TorshResult<Tensor> {
    let context = "cross_entropy_with_label_smoothing";
    if label_smoothing < 0.0 || label_smoothing >= 1.0 {
        return Err(TorshError::InvalidArgument(
            "label_smoothing must be in [0.0, 1.0)".to_string(),
        ));
    }
    if !matches!(reduction, "none" | "mean" | "sum") {
        return Err(TorshError::InvalidArgument(format!(
            "Unknown reduction: {}",
            reduction
        )));
    }
    let (batch_size, num_classes) = validate_classification_shapes(input, target, context)?;

    let smoothing_value = label_smoothing as f32 / num_classes as f32;
    let confidence = 1.0 - label_smoothing as f32;

    // Apply log softmax
    let dim = (input.shape().ndim() - 1) as i32;
    let log_probs = input.log_softmax(dim)?;

    // Negated smoothed target distribution, scaled by the class weight of the
    // sample. Keeping every per-sample factor inside this constant tensor means
    // the loss is a single differentiable expression in `log_probs`.
    let mut selector = vec![0.0f32; batch_size * num_classes];
    let mut total_weight = 0.0f32;
    for i in 0..batch_size {
        let class = target_class(target, i, num_classes, context)?;
        if Some(class) == ignore_index {
            continue;
        }
        let w = class_weight(weight, class as usize, context)?;
        for c in 0..num_classes {
            selector[i * num_classes + c] = -w * smoothing_value;
        }
        selector[i * num_classes + class as usize] = -w * (confidence + smoothing_value);
        total_weight += w;
    }

    if reduction == "mean" {
        let scale = if total_weight == 0.0 {
            0.0
        } else {
            1.0 / total_weight
        };
        for value in selector.iter_mut() {
            *value *= scale;
        }
    }

    let selector_tensor =
        Tensor::from_data(selector, vec![batch_size, num_classes], input.device())?;
    // `loss` already has shape [batch_size]; there is no class axis left to squeeze.
    let loss = row_sum(&log_probs.mul(&selector_tensor)?)?;

    match reduction {
        "none" => Ok(loss),
        // The per-sample terms already carry the 1/sum(w) factor for "mean".
        "mean" | "sum" => loss.sum(),
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
