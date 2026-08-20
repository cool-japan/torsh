//! Basic loss functions for neural networks
//!
//! This module provides fundamental loss functions including cross entropy,
//! MSE, L1, binary cross entropy, KL divergence, and specialized losses.

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;
use torsh_core::reduction::Reduction;

use torsh_functional::loss::common::ReductionExt;

// =============================================================================
// CLASSIFICATION LOSSES
// =============================================================================

/// Cross entropy loss function
/// Enhanced with SciRS2-inspired numerical stability and efficiency
///
/// # Device
///
/// The one-hot selector and the per-sample class-weight vector are constants
/// derived from `target`, but they are still multiplied against `input`'s
/// activations — so they are built on [`input.device()`](Tensor::device).
/// Hardcoding `DeviceType::Cpu` here (as this function used to) silently pins
/// the whole loss to the host as soon as `input` lives anywhere else.
pub fn cross_entropy(
    input: &Tensor,
    target: &Tensor<i64>,
    weight: Option<&Tensor>,
    reduction: Reduction,
    ignore_index: Option<i64>,
) -> Result<Tensor> {

    // Enhanced cross entropy implementation using numerically stable log_softmax

    // Apply log_softmax for numerical stability
    let log_probs = crate::functional::activation::log_softmax(input, Some(-1))?;

    let input_shape_binding = input.shape();
    let input_dims = input_shape_binding.dims();
    let (batch_size, num_classes) = match input_dims {
        [batch, classes] => (*batch, *classes),
        _ => {
            return Err(TorshError::InvalidShape(format!(
                "cross_entropy expects a 2-D input [N, C], got shape {input_dims:?}"
            )))
        }
    };

    let target_vec = target.to_vec()?;
    if target_vec.len() != batch_size {
        return Err(TorshError::InvalidShape(format!(
            "cross_entropy expects one target per sample: input has {batch_size} rows \
             but target has {} entries",
            target_vec.len()
        )));
    }

    // Create one-hot encoding for targets
    let mut one_hot_data = vec![0.0f32; batch_size * num_classes];
    for (i, &target_idx) in target_vec.iter().enumerate() {
        if let Some(ignore_idx) = ignore_index {
            if target_idx == ignore_idx {
                continue; // Skip ignored indices
            }
        }
        if target_idx >= 0 && (target_idx as usize) < num_classes {
            if let Some(slot) = one_hot_data.get_mut(i * num_classes + target_idx as usize) {
                *slot = 1.0;
            }
        }
    }

    let one_hot = Tensor::from_data(one_hot_data, vec![batch_size, num_classes], input.device())?;

    // Compute negative log likelihood: -sum(one_hot * log_probs)
    let neg_log_likelihood = log_probs.mul_op(&one_hot)?.neg()?;
    let loss_per_sample = neg_log_likelihood.sum_dim(&[-1], false)?;

    // Apply class weights if provided
    let weighted_loss = if let Some(weights) = weight {
        // Apply weights based on target classes. The lookup table is read once
        // instead of once per sample, which is what the loop used to do.
        let weight_vec = weights.to_vec()?;
        let mut weight_data = vec![1.0f32; batch_size];
        for (slot, &target_idx) in weight_data.iter_mut().zip(target_vec.iter()) {
            if target_idx >= 0 {
                if let Some(&class_weight) = weight_vec.get(target_idx as usize) {
                    *slot = class_weight;
                }
            }
        }
        let weight_tensor = Tensor::from_data(weight_data, vec![batch_size], input.device())?;
        loss_per_sample.mul_op(&weight_tensor)?
    } else {
        loss_per_sample
    };

    // Apply reduction
    apply_reduction(&weighted_loss, reduction, ignore_index, &target_vec)
}

/// Binary cross entropy loss function
pub fn binary_cross_entropy(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> Result<Tensor> {
    // BCE: -(target * log(input) + (1 - target) * log(1 - input))
    let eps = 1e-7; // Small epsilon for numerical stability
    let eps_tensor = torsh_tensor::creation::full_like(input, eps)?;
    let ones = torsh_tensor::creation::ones_like(input)?;

    // Clamp input to avoid log(0)
    let clamped_input = input.maximum(&eps_tensor)?;
    let clamped_input = clamped_input.minimum(&ones.sub(&eps_tensor)?)?;

    let log_input = clamped_input.log()?;
    let one_minus_input = ones.sub(&clamped_input)?;
    let log_one_minus_input = one_minus_input.log()?;

    let term1 = target.mul_op(&log_input)?;
    let one_minus_target = ones.sub(target)?;
    let term2 = one_minus_target.mul_op(&log_one_minus_input)?;

    let mut loss = term1.add(&term2)?.neg()?;

    // Apply weights if provided
    if let Some(w) = weight {
        loss = loss.mul_op(w)?;
    }

    apply_reduction(&loss, reduction, None, &[])
}

/// Binary cross entropy with logits loss function
pub fn binary_cross_entropy_with_logits(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    reduction: Reduction,
    pos_weight: Option<&Tensor>,
) -> Result<Tensor> {
    // More numerically stable version that combines sigmoid and BCE
    // BCE with logits: max(x, 0) - x * target + log(1 + exp(-abs(x)))

    let zeros = torsh_tensor::creation::zeros_like(input)?;
    let ones = torsh_tensor::creation::ones_like(input)?;

    // max(x, 0)
    let max_term = input.maximum(&zeros)?;

    // x * target
    let mult_term = input.mul_op(target)?;

    // log(1 + exp(-abs(x)))
    let abs_input = input.abs()?;
    let neg_abs_input = abs_input.neg()?;
    let exp_term = neg_abs_input.exp()?;
    let log_term = ones.add(&exp_term)?.log()?;

    // Combine terms
    let mut loss = max_term.sub(&mult_term)?.add(&log_term)?;

    // Apply positive weight if provided
    if let Some(pos_w) = pos_weight {
        let weighted_target = target.mul_op(pos_w)?;
        let pos_term = input.mul_op(&weighted_target)?;
        loss = loss.add(&pos_term)?;
    }

    // Apply weights if provided
    if let Some(w) = weight {
        loss = loss.mul_op(w)?;
    }

    apply_reduction(&loss, reduction, None, &[])
}

/// Multi-margin loss function
///
/// ```text
/// loss[b] = w[y_b] / (C - 1) * sum_{c != y_b} max(0, margin - x[b, y_b] + x[b, c])^p
/// ```
///
/// # Autograd
///
/// Every quantity that depends on `input` stays a tensor op. The class index
/// `y_b` is an `i64` label, so the two things derived from it — the one-hot
/// selector that picks `x[b, y_b]` and the `c != y_b` mask — are *constants*,
/// built once on [`input.device()`](Tensor::device) and multiplied in. The
/// predecessor instead read `input.to_vec()`, ran the double loop in `f32` and
/// handed the answer to `Tensor::from_vec`, which returns a detached leaf: the
/// forward values were right and `backward()` could not reach the scores.
///
/// The `c != y_b` mask is applied *after* the hinge rather than by skipping the
/// column, because the true class's own term is `max(0, margin) = margin`, not
/// zero; multiplying by the mask removes both the value and its gradient.
///
/// # Deviations from the predecessor
///
/// * A non-2-D `input` used to panic on `dims()[1]`; it is now an
///   `InvalidShape` error, as is a `target` whose length differs from the batch.
/// * `C == 1` used to divide `0.0` by `0.0` and yield `NaN` for every sample.
///   The divisor is now `max(C - 1, 1)`, so a single-class problem scores a
///   flat `0` — which is what the sum over `c != y_b` (an empty sum) actually
///   says.
/// * `p < 1` is rejected. The predecessor summed `margin_diff.powi(p)` only
///   over the terms it had already found positive, which the clamped form
///   cannot reproduce below `p == 1`: a clamped-to-zero entry raised to `0`
///   is `1`, and to a negative exponent is `inf`. PyTorch's `MultiMarginLoss`
///   documents `p ∈ {1, 2}` anyway; every `p >= 1` stays exact and available.
/// * A `target` outside `[0, C)` still scores `0` for that sample, unchanged;
///   the row's selector and mask are simply left at zero.
pub fn multi_margin_loss(
    input: &Tensor,
    target: &Tensor<i64>,
    p: i32,
    margin: f32,
    weight: Option<&Tensor>,
    reduction: Reduction,
) -> Result<Tensor> {
    let shape_binding = input.shape();
    let input_dims = shape_binding.dims();
    let (batch_size, num_classes) = match input_dims {
        [batch, classes] => (*batch, *classes),
        _ => {
            return Err(TorshError::InvalidShape(format!(
                "multi_margin_loss expects a 2-D input [N, C], got shape {input_dims:?}"
            )))
        }
    };

    if p < 1 {
        return Err(TorshError::InvalidArgument(format!(
            "multi_margin_loss expects p >= 1 (PyTorch documents p in {{1, 2}}), got {p}"
        )));
    }

    let target_data = target.to_vec()?;
    if target_data.len() != batch_size {
        return Err(TorshError::InvalidShape(format!(
            "multi_margin_loss expects one target per sample: input has {batch_size} rows \
             but target has {} entries",
            target_data.len()
        )));
    }

    let weight_data = match weight {
        Some(weights) => Some(weights.to_vec()?),
        None => None,
    };

    // An empty sum over `c != y_b` has no terms to average, so the divisor is
    // clamped rather than allowed to be zero (see the deviations above).
    let divisor = num_classes.saturating_sub(1).max(1) as f32;

    let mut selector = vec![0.0f32; batch_size * num_classes];
    let mut off_class = vec![0.0f32; batch_size * num_classes];
    let mut sample_scale = vec![0.0f32; batch_size];
    for (b, &target_index) in target_data.iter().enumerate() {
        // A negative label wraps to a huge `usize` and is rejected by the same
        // bound check as any other out-of-range class, exactly as before.
        let true_class = target_index as usize;
        if true_class >= num_classes {
            // Row stays all-zero: selector, mask and scale, so the sample
            // contributes `0` with no gradient, as the predecessor did.
            continue;
        }
        let row = b * num_classes;
        if let Some(slot) = selector.get_mut(row + true_class) {
            *slot = 1.0;
        }
        for c in 0..num_classes {
            if c != true_class {
                if let Some(slot) = off_class.get_mut(row + c) {
                    *slot = 1.0;
                }
            }
        }
        let class_weight = weight_data
            .as_ref()
            .and_then(|weights| weights.get(true_class).copied())
            .unwrap_or(1.0);
        if let Some(slot) = sample_scale.get_mut(b) {
            *slot = class_weight / divisor;
        }
    }

    let selector_tensor =
        Tensor::from_data(selector, vec![batch_size, num_classes], input.device())?;
    let off_class_tensor =
        Tensor::from_data(off_class, vec![batch_size, num_classes], input.device())?;
    let scale_tensor = Tensor::from_data(sample_scale, vec![batch_size], input.device())?;

    // `x[b, y_b]`, kept as `[N, 1]` so it broadcasts against `[N, C]`.
    let true_score = input.mul_op(&selector_tensor)?.sum_dim(&[-1], true)?;
    let hinge = input.sub(&true_score)?.add_scalar(margin)?.clamp_min(0.0)?;
    let powered = if p == 1 { hinge } else { hinge.pow(p as f32)? };

    let per_sample = powered
        .mul_op(&off_class_tensor)?
        .sum_dim(&[-1], false)?
        .mul_op(&scale_tensor)?;

    apply_reduction(&per_sample, reduction, None, &[])
}

/// Multilabel margin loss function
pub fn multilabel_margin_loss(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    // Simplified multilabel margin loss
    let ones = torsh_tensor::creation::ones_like(input)?;
    let margin_tensor = torsh_tensor::creation::full_like(input, 1.0)?;

    // For each sample, compute margin loss
    let target_scores = input.mul_op(target)?;
    let non_target_scores = input.mul_op(&ones.sub(target)?)?;

    // Margin: 1 - target_score + non_target_score
    let margin = margin_tensor.sub(&target_scores)?.add(&non_target_scores)?;
    let zeros = torsh_tensor::creation::zeros_like(input)?;
    let loss = margin.maximum(&zeros)?;

    apply_reduction(&loss, reduction, None, &[])
}

// =============================================================================
// REGRESSION LOSSES
// =============================================================================

/// Mean squared error loss function
pub fn mse_loss(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    // MSE: (input - target)^2
    let diff = input.sub(target)?;
    let squared_diff = diff.mul_op(&diff)?;

    apply_reduction(&squared_diff, reduction, None, &[])
}

/// L1 (Mean Absolute Error) loss function
pub fn l1_loss(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    // L1: |input - target|
    let diff = input.sub(target)?;
    let abs_diff = diff.abs()?;

    apply_reduction(&abs_diff, reduction, None, &[])
}

/// Split `|input - target|` into the part that lies inside a `threshold`-wide
/// band around zero and the part that sticks out of it, as a *differentiable*
/// pair: `inside = min(|d|, threshold)` and `outside = |d| - inside`.
///
/// Every piecewise loss below is `f(inside) + g(outside)` for a pair of smooth
/// `f`, `g`, so none of them needs a branch — which matters because the two ops
/// a branch would need, [`Tensor::le`] and [`Tensor::where_tensor`], do not
/// record, and routing through them is exactly how `huber_loss` used to fall off
/// the autograd graph. [`Tensor::clamp_max`] does record
/// (`Operation::ClampBounds`): the gradient passes wherever the element was left
/// alone and is zero wherever the clamp bit, which is precisely the piecewise
/// derivative the branch would have produced.
///
/// `|d|` feeds two consumers here, so the correctness of every caller rests on
/// the engine *accumulating* into a fan-out node rather than overwriting it;
/// `hardening_nn_losses.rs` pins that with finite differences.
fn banded_abs_error(input: &Tensor, target: &Tensor, threshold: f32) -> Result<(Tensor, Tensor)> {
    let abs_diff = input.sub(target)?.abs()?;
    let inside = abs_diff.clamp_max(threshold)?;
    let outside = abs_diff.sub(&inside)?;
    Ok((inside, outside))
}

/// Smooth L1 loss function (Huber loss)
///
/// ```text
/// loss = 0.5 * (input - target)^2 / beta   if |input - target| < beta
/// loss = |input - target| - 0.5 * beta     otherwise
/// ```
///
/// # Autograd
///
/// Expressed as `0.5 * inside^2 / beta + outside` over the band split of
/// `banded_abs_error`, which reproduces both arms exactly — inside the band
/// `outside == 0`, outside it `inside == beta` and
/// `0.5 * beta^2 / beta + (|d| - beta) == |d| - 0.5 * beta`. The predecessor
/// read `|d|` out with `to_vec()` and rebuilt the answer through
/// `Tensor::from_vec`, which returns a detached leaf and made the loss
/// non-differentiable.
///
/// `beta == 0` is the one value the band form cannot express (`0.5 * 0 / 0` is
/// `NaN`); it degenerates to plain L1, which is both what the element-wise
/// predecessor computed (`|d| < 0` is never true, so it always took the
/// `|d| - 0` arm) and what PyTorch special-cases. Negative `beta` needs no
/// special case: `min(|d|, beta) == beta` for every element, and the band form
/// collapses to `|d| - 0.5 * beta` on its own.
pub fn smooth_l1_loss(
    input: &Tensor,
    target: &Tensor,
    beta: f32,
    reduction: Reduction,
) -> Result<Tensor> {
    if beta == 0.0 {
        let abs_diff = input.sub(target)?.abs()?;
        return apply_reduction(&abs_diff, reduction, None, &[]);
    }

    let (inside, outside) = banded_abs_error(input, target, beta)?;
    let quadratic = inside.square()?.mul_scalar(0.5)?.div_scalar(beta)?;
    let loss = quadratic.add(&outside)?;

    apply_reduction(&loss, reduction, None, &[])
}

/// Huber loss function
///
/// ```text
/// loss = 0.5 * (input - target)^2                    if |input - target| <= delta
/// loss = delta * (|input - target| - 0.5 * delta)    otherwise
/// ```
///
/// # Autograd
///
/// Expressed as `0.5 * inside^2 + delta * outside` over the band split of
/// `banded_abs_error`. Inside the band that is `0.5 * d^2`; outside it,
/// `0.5 * delta^2 + delta * (|d| - delta) == delta * (|d| - 0.5 * delta)`. The
/// two forms are algebraically identical but not bit-identical — the rewrite can
/// move the last ulp — which is why the regression test pins the forward values
/// with a relative tolerance rather than exact equality.
///
/// The predecessor selected the arm with `abs_diff.le(&delta)` and
/// `where_tensor`, neither of which records, so the returned loss had
/// `requires_grad == false` no matter what `input` was.
///
/// `delta == 0` needs no special case: `inside` is then identically zero and the
/// loss is identically zero, which is what the `le`/`where_tensor` predecessor
/// also produced (`0.5 * d^2` was only selected where `|d| <= 0`, and the other
/// arm carries the factor `delta == 0`).
pub fn huber_loss(input: &Tensor, target: &Tensor, delta: f32, reduction: Reduction) -> Result<Tensor> {
    let (inside, outside) = banded_abs_error(input, target, delta)?;
    let quadratic = inside.square()?.mul_scalar(0.5)?;
    let linear = outside.mul_scalar(delta)?;
    let loss = quadratic.add(&linear)?;

    apply_reduction(&loss, reduction, None, &[])
}

// =============================================================================
// PROBABILISTIC LOSSES
// =============================================================================

/// KL divergence loss function
/// Enhanced with SciRS2-inspired numerical stability and efficiency
///
/// Computes KL(target || input) = sum(target * (log(target) - input))
/// where input is expected to be log-probabilities
pub fn kl_div(
    input: &Tensor,
    target: &Tensor,
    reduction: Reduction,
    log_target: bool,
) -> Result<Tensor> {
    // Enhanced KL divergence implementation with numerical stability

    // Add small epsilon for numerical stability
    let eps = 1e-8f32;
    let eps_tensor = torsh_tensor::creation::full_like(target, eps)?;

    // Handle target tensor based on log_target flag
    let (target_probs, log_target_probs) = if log_target {
        // Target is already in log space
        let target_probs = target.exp()?;
        (target_probs, target.clone())
    } else {
        // Target is in probability space, need to compute log
        // Add epsilon to prevent log(0)
        let stable_target = target.add(&eps_tensor)?;
        let log_target_probs = stable_target.log()?;
        (target.clone(), log_target_probs)
    };

    // Ensure input is in log space (it should be log-probabilities)
    // KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
    // where P is target and Q is input (in log space)

    // Compute log(target) - input
    let log_ratio = log_target_probs.sub(input)?;

    // Multiply by target probabilities: target * (log(target) - input)
    let kl_elements = target_probs.mul_op(&log_ratio)?;

    // Handle reduction
    match reduction {
        Reduction::Mean => {
            // Mean over all elements
            kl_elements.mean(None, false)
        }
        Reduction::Sum => {
            // Sum over all elements
            kl_elements.sum()
        }
        Reduction::BatchMean => {
            // Sum over all dimensions except batch, then mean over batch
            let batch_size = input.shape().dims()[0] as f32;
            let total_sum = kl_elements.sum()?;
            let batch_size_tensor = torsh_tensor::creation::full(&[1], batch_size)?;
            total_sum.div(&batch_size_tensor)
        }
        Reduction::None => {
            // No reduction, return element-wise losses
            Ok(kl_elements)
        }
    }
}

/// Negative log likelihood loss function
///
/// `input` holds log-probabilities of shape `[N, C]` — typically the output of
/// [`log_softmax`](crate::functional::log_softmax) — and `target` holds one
/// class index per sample.
///
/// # Autograd
///
/// The target log-probability is gathered with a constant one-hot selector and a
/// multiply, so the loss stays attached to `input` and `backward()` reaches it.
/// Reading the values out with `to_vec` and rebuilding the per-sample losses
/// through `Tensor::from_vec` — as this function used to do — returns a detached
/// leaf and makes the loss non-differentiable.
///
/// The class weight is folded into the selector entry instead of being applied
/// as a second multiply, which reproduces the previous `-log_prob * weight`
/// product exactly. Ignored samples and out-of-range class indices leave an
/// all-zero selector row and contribute `0.0`.
///
/// # Non-finite inputs
///
/// The gather is a masked sum over the whole row rather than a single-element
/// read, so a non-finite entry *anywhere* in row `b` — a `-inf` produced by
/// `probs.log()` on a zero probability, say — makes that row's loss `NaN`,
/// because `0.0 * -inf` is `NaN`. This holds even when the selected class itself
/// is finite. The element-wise predecessor read only the selected entry and was
/// immune; that immunity is not recoverable while staying differentiable, and
/// every masked-gather formulation shares the property (including
/// `torsh_functional::nll_loss`, which expresses the same sum as a matmul).
/// Feed log-probabilities from
/// [`log_softmax`](crate::functional::log_softmax), which is finite by
/// construction.
pub fn nll_loss(
    input: &Tensor,
    target: &Tensor<i64>,
    weight: Option<&Tensor>,
    ignore_index: Option<i64>,
    reduction: Reduction,
) -> Result<Tensor> {
    // NLL loss assumes log-probabilities as input
    let input_shape_binding = input.shape();
    let input_dims = input_shape_binding.dims();
    let (batch_size, num_classes) = match input_dims {
        [batch, classes] => (*batch, *classes),
        _ => {
            return Err(TorshError::InvalidShape(format!(
                "nll_loss expects a 2-D input [N, C], got shape {input_dims:?}"
            )))
        }
    };

    let target_data = target.to_vec()?;
    if target_data.len() < batch_size {
        return Err(TorshError::InvalidShape(format!(
            "nll_loss expects one target per sample: input has {batch_size} rows \
             but target has {} entries",
            target_data.len()
        )));
    }

    let weight_data = match weight {
        Some(weights) => Some(weights.to_vec()?),
        None => None,
    };

    // Constant selector: selector[b, t_b] = weight(t_b) and zero everywhere else,
    // so that -sum_c selector[b, c] * input[b, c] = -weight(t_b) * input[b, t_b].
    let mut selector = vec![0.0f32; batch_size * num_classes];
    for (b, &target_index) in target_data.iter().take(batch_size).enumerate() {
        if Some(target_index) == ignore_index {
            continue;
        }
        // A negative index wraps to a huge `usize` and is rejected by the same
        // bound check as any other out-of-range class.
        let target_class = target_index as usize;
        if target_class >= num_classes {
            continue;
        }
        let class_weight = match weight_data.as_ref() {
            Some(values) => values.get(target_class).copied().unwrap_or(1.0),
            None => 1.0,
        };
        if let Some(slot) = selector.get_mut(b * num_classes + target_class) {
            *slot = class_weight;
        }
    }

    let selector_tensor =
        Tensor::from_data(selector, vec![batch_size, num_classes], input.device())?;
    let loss_tensor = input
        .mul_op(&selector_tensor)?
        .neg()?
        .sum_dim(&[-1], false)?;

    apply_reduction(&loss_tensor, reduction, ignore_index, &target_data)
}

// =============================================================================
// RANKING AND SIMILARITY LOSSES
// =============================================================================

/// Focal loss function for addressing class imbalance
/// FL(pt) = -α * (1 - pt)^γ * log(pt)
///
/// This loss down-weights easy examples and focuses learning on hard examples.
/// Particularly useful for object detection and highly imbalanced classification.
///
/// # Autograd
///
/// `log(pt)` is selected from `log_softmax(input)` with a constant one-hot
/// selector and a row sum — the same masked-gather idiom
/// [`nll_loss`] uses — and `pt` is recovered as `exp(log(pt))`, so the whole
/// expression stays a composition of recording ops and `backward()` reaches
/// `input`. The predecessor called the recording `log_softmax` and then threw
/// the graph away by reading both tensors out with `to_vec()`, which made a
/// carefully stabilised forward pass useless for training.
///
/// # Non-finite inputs
///
/// Because the selection is a masked sum over the whole row rather than a
/// single-element read, a non-finite entry *anywhere* in row `b` makes that
/// row's loss `NaN` (`0.0 * -inf` is `NaN`), even when the target class itself
/// is finite. `log_softmax` produces `-inf` for a class whose logit is `-inf`,
/// so masking a class out with `f32::NEG_INFINITY` — a common way to express
/// "this class is impossible" — poisons the row. The element-wise predecessor
/// read only the selected entry and was immune; that immunity is not recoverable
/// while staying differentiable, and [`nll_loss`] documents the same property.
pub fn focal_loss(
    input: &Tensor,
    target: &Tensor<i64>,
    alpha: Option<f32>,
    gamma: f32,
    reduction: Reduction,
) -> Result<Tensor> {
    // Focal Loss: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    // where pt is the probability of the true class

    let input_shape = input.shape();
    let input_dims = input_shape.dims();

    let (batch_size, num_classes) = match input_dims {
        [batch, classes] => (*batch, *classes),
        _ => {
            return Err(torsh_core::error::TorshError::InvalidShape(format!(
                "Input must be 2D [batch_size, num_classes], got shape {:?}",
                input_dims
            )))
        }
    };

    let target_data = target.to_vec()?;
    if target_data.len() < batch_size {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "focal_loss expects one target per sample: input has {batch_size} rows \
             but target has {} entries",
            target_data.len()
        )));
    }

    // Constant one-hot selector: selector[b, t_b] = 1 and zero everywhere else,
    // so that sum_c selector[b, c] * log_softmax(input)[b, c] == log(pt_b).
    let mut selector = vec![0.0f32; batch_size * num_classes];
    for (b, &target_index) in target_data.iter().take(batch_size).enumerate() {
        // A negative index wraps to a huge `usize` and is rejected by the same
        // bound check as any other out-of-range class, exactly as before.
        let target_class = target_index as usize;
        if target_class >= num_classes {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Target class {} out of range for {} classes",
                target_class, num_classes
            )));
        }
        if let Some(slot) = selector.get_mut(b * num_classes + target_class) {
            *slot = 1.0;
        }
    }
    let selector_tensor =
        Tensor::from_data(selector, vec![batch_size, num_classes], input.device())?;

    // Apply log_softmax to get log probabilities, then pick the true class.
    let log_probs = input.log_softmax(-1)?;
    let log_pt = log_probs.mul_op(&selector_tensor)?.sum_dim(&[-1], false)?;

    // pt = exp(log(pt)); the exponential of the *selected* log-probability is
    // the same number the predecessor read out of `log_probs.exp()`.
    let pt = log_pt.exp()?;

    // Focal loss: -alpha * (1 - pt)^gamma * log(pt)
    let alpha_weight = alpha.unwrap_or(1.0);
    let focal_weight = pt
        .neg()?
        .add_scalar(1.0)?
        .pow(gamma)?
        .mul_scalar(alpha_weight)?;
    let per_sample = focal_weight.mul_op(&log_pt)?.neg()?;

    // Apply reduction. The reduced forms keep their historical `[1]` shape.
    match reduction {
        Reduction::Mean => per_sample.mean(None, false)?.view(&[1]),
        Reduction::Sum => per_sample.sum()?.view(&[1]),
        Reduction::None => Ok(per_sample),
        Reduction::BatchMean => Err(TorshError::Unimplemented("BatchMean isn't implemented yet".to_string()))
    }
}

/// Triplet margin loss function
///
/// Computes the triplet loss between anchor, positive, and negative samples.
/// The loss encourages the distance between anchor and positive to be smaller
/// than the distance between anchor and negative by at least margin.
///
/// Formula: L = max(d(a,p) - d(a,n) + margin, 0)
///
/// # Arguments
/// * `anchor` - Anchor samples tensor
/// * `positive` - Positive samples tensor (similar to anchor)
/// * `negative` - Negative samples tensor (dissimilar to anchor)
/// * `margin` - Minimum distance difference between positive and negative pairs
/// * `p` - Norm degree for pairwise distance (e.g., 2.0 for L2 distance)
/// * `reduction` - Reduction method: "mean", "sum", or "none"
///
/// # Autograd
///
/// Both distances go through `p_norm_distance`, so the whole loss is a
/// composition of recording ops and `backward()` reaches all three operands.
/// The predecessor read every operand out with `to_vec()`, ran the distance
/// loops in `f32` and rebuilt the answer with `Tensor::from_vec` — a detached
/// leaf, which made the loss undifferentiable while leaving its forward values
/// correct.
///
/// # Shape conventions (unchanged)
///
/// Dimension `0` is the batch and everything after it is one flat feature
/// vector, so `[N, D]`, `[N, C, H, W]` and `[N]` (a batch of scalars) all work
/// and produce `N` losses. A 0-D operand has no batch axis and is now an
/// `InvalidShape` error instead of an index panic.
pub fn triplet_margin_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f32,
    p: f32,
    reduction: Reduction,
) -> Result<Tensor> {
    let anchor_shape_obj = anchor.shape();
    let anchor_shape = anchor_shape_obj.dims();
    if anchor_shape.is_empty() {
        return Err(TorshError::InvalidShape(
            "triplet_margin_loss expects operands with a leading batch axis, got a 0-D tensor"
                .to_string(),
        ));
    }
    let feature_axes = feature_axes_of(anchor_shape);

    let dist_ap = p_norm_distance(anchor, positive, p, &feature_axes)?;
    let dist_an = p_norm_distance(anchor, negative, p, &feature_axes)?;

    // max(d(a,p) - d(a,n) + margin, 0)
    let per_sample = dist_ap.sub(&dist_an)?.add_scalar(margin)?.clamp_min(0.0)?;

    // The reduced forms keep their historical `[1]` shape.
    match reduction {
        Reduction::Mean => per_sample.mean(None, false)?.view(&[1]),
        Reduction::Sum => per_sample.sum()?.view(&[1]),
        Reduction::None => Ok(per_sample),
        Reduction::BatchMean => Err(TorshError::Unimplemented("BatchMean isn't implemented yet".to_string()))
    }
}

/// Contrastive loss function
///
/// Used for learning embeddings where similar pairs should have small distances
/// and dissimilar pairs should have large distances (at least margin apart).
///
/// Formula:
/// - For similar pairs (target=1): L = d^2
/// - For dissimilar pairs (target=0): L = max(margin - d, 0)^2
///
/// where d is the Euclidean distance between embeddings.
///
/// # Arguments
/// * `output1` - First embedding tensor [batch_size, feature_dim]
/// * `output2` - Second embedding tensor [batch_size, feature_dim]
/// * `target` - Binary labels (1 for similar, 0 for dissimilar)
/// * `margin` - Minimum distance for dissimilar pairs
/// * `reduction` - Reduction method: "mean", "sum", or "none"
///
/// # Autograd
///
/// The `target` labels are constants, so the branch they select is expressed as
/// a *mask* rather than a control-flow `if`:
///
/// ```text
/// loss = similar * d² + (1 - similar) * max(0, margin - d)²
/// ```
///
/// with `similar[i] ∈ {0, 1}` built once on
/// [`output1.device()`](Tensor::device). Both arms are evaluated for every
/// sample and the unwanted one is multiplied by zero, which is what keeps the
/// expression differentiable — the predecessor's `to_vec()` loop plus
/// `Tensor::from_vec` returned a detached leaf.
///
/// `d` is `sqrt(d²)`, whose derivative blows up at `d = 0` (two identical
/// embeddings). `d²` is floored at `1e-12` before the square root, so a
/// coincident pair gets a subgradient of `0` there instead of `inf`/`NaN`; the
/// floor shifts `d` by at most `1e-6` and is invisible at this loss's
/// tolerances. Same convention as `p_norm_distance`.
///
/// # Shape conventions (unchanged)
///
/// Dimension `0` is the batch, everything after it is one flat feature vector.
/// A 0-D operand is now an `InvalidShape` error instead of an index panic, and
/// a `target` whose length differs from the batch is rejected rather than read
/// out of bounds.
pub fn contrastive_loss(
    output1: &Tensor,
    output2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: Reduction,
) -> Result<Tensor> {
    let output1_shape_obj = output1.shape();
    let output1_shape = output1_shape_obj.dims();
    let Some(&batch_size) = output1_shape.first() else {
        return Err(TorshError::InvalidShape(
            "contrastive_loss expects embeddings with a leading batch axis, got a 0-D tensor"
                .to_string(),
        ));
    };
    let feature_axes = feature_axes_of(output1_shape);

    let squared = output1.sub(output2)?.square()?;
    let dist_squared = reduce_features(squared, &feature_axes)?;
    let dist = dist_squared.clamp_min(DISTANCE_FLOOR)?.sqrt()?;

    let target_data = target.to_vec()?;
    if target_data.len() != batch_size {
        return Err(TorshError::InvalidShape(format!(
            "contrastive_loss expects one label per sample: the embeddings have {batch_size} \
             rows but target has {} entries",
            target_data.len()
        )));
    }
    // `label > 0.5` is the predecessor's own similar/dissimilar split.
    let similar: Vec<f32> = target_data
        .iter()
        .map(|&label| if label > 0.5 { 1.0 } else { 0.0 })
        .collect();
    let dissimilar: Vec<f32> = similar.iter().map(|&value| 1.0 - value).collect();
    let similar_mask = Tensor::from_data(similar, vec![batch_size], output1.device())?;
    let dissimilar_mask = Tensor::from_data(dissimilar, vec![batch_size], output1.device())?;

    let repulsion = dist.neg()?.add_scalar(margin)?.clamp_min(0.0)?.square()?;
    let per_sample = dist_squared
        .mul_op(&similar_mask)?
        .add(&repulsion.mul_op(&dissimilar_mask)?)?;

    // The reduced forms keep their historical `[1]` shape.
    match reduction {
        Reduction::Mean => per_sample.mean(None, false)?.view(&[1]),
        Reduction::Sum => per_sample.sum()?.view(&[1]),
        Reduction::None => Ok(per_sample),
        Reduction::BatchMean => Err(TorshError::Unimplemented("BatchMean isn't implemented yet".to_string()))
    }
}

/// Cosine Embedding Loss for similarity learning
///
/// Encourages embeddings to have high cosine similarity for similar items
/// (`target == 1`) and low cosine similarity for dissimilar items
/// (`target == -1`):
///
/// ```text
/// loss[i] = 1 - cos(x1[i], x2[i])              if y[i] ==  1
/// loss[i] = max(0, cos(x1[i], x2[i]) - margin) if y[i] == -1
/// ```
///
/// # Shapes
///
/// Follows `torch.nn.functional.cosine_embedding_loss` and the sibling
/// implementation in `torsh-functional`
/// (`crates/torsh-functional/src/loss/similarity.rs`): the cosine is taken
/// along the **last** axis, so `[N, D]` inputs give `N` losses and a bare `[D]`
/// pair gives a single 0-D loss. `target` carries one label per sample and may
/// be `[N]`, `[1]` or 0-D as long as its element count matches.
///
/// # What this used to do
///
/// The predecessor contracted the *whole* tensor — `input1.mul(input2)?.sum()`
/// against [`Tensor::norm`], which is the Frobenius norm of every element — so
/// a batch produced one global cosine instead of `N` per-sample ones. It then
/// compared `target` against a 0-D scalar with `eq`, which meant every batched
/// call failed outright with `ShapeMismatch { expected: [N], got: [] }`; the
/// only shape combination that ever worked was a 1-D pair with a 0-D target,
/// for which the global cosine happens to *be* the per-sample cosine. That case
/// is value-for-value preserved and pinned in `tests/hardening_nn_losses.rs`.
///
/// # Autograd
///
/// The branch is selected by a constant mask instead of `eq` + `where_tensor`
/// (neither of which records), so the result stays on the graph:
/// `loss = pos * (1 - cos) + (1 - pos) * max(0, cos - margin)`. The positive
/// arm is chosen by `y > 0`, matching `torsh-functional`'s `gt_scalar(0.0)`;
/// for the documented `y ∈ {1, -1}` domain that is the same predicate the
/// predecessor's `eq(1.0)` applied. The denominator `|x1| · |x2|` is floored
/// before the square root so a zero embedding yields a finite gradient rather
/// than `NaN`.
pub fn cosine_embedding_loss(
    input1: &Tensor,
    input2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: Reduction,
) -> Result<Tensor> {
    let input1_shape_obj = input1.shape();
    let input1_shape = input1_shape_obj.dims();
    if input1_shape.is_empty() {
        return Err(TorshError::InvalidShape(
            "cosine_embedding_loss expects embeddings with a feature axis, got a 0-D tensor"
                .to_string(),
        ));
    }
    let feature_axis = [input1_shape.len() as i32 - 1];

    // cos = <x1, x2> / (|x1| * |x2|), all reduced along the feature axis.
    let dot_product = input1.mul_op(input2)?.sum_dim(&feature_axis, false)?;
    let norm1_squared = input1.square()?.sum_dim(&feature_axis, false)?;
    let norm2_squared = input2.square()?.sum_dim(&feature_axis, false)?;
    let denominator = norm1_squared
        .mul_op(&norm2_squared)?
        .clamp_min(DISTANCE_FLOOR)?
        .sqrt()?;
    let cosine_sim = dot_product.div(&denominator)?;

    let cosine_shape_obj = cosine_sim.shape();
    let cosine_dims = cosine_shape_obj.dims().to_vec();
    let sample_count: usize = cosine_dims.iter().product();

    let target_data = target.to_vec()?;
    if target_data.len() != sample_count {
        return Err(TorshError::InvalidShape(format!(
            "cosine_embedding_loss expects one label per sample: the embeddings yield \
             {sample_count} cosine values but target has {} entries",
            target_data.len()
        )));
    }
    let positive: Vec<f32> = target_data
        .iter()
        .map(|&label| if label > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let negative: Vec<f32> = positive.iter().map(|&value| 1.0 - value).collect();
    let positive_mask = Tensor::from_data(positive, cosine_dims.clone(), input1.device())?;
    let negative_mask = Tensor::from_data(negative, cosine_dims, input1.device())?;

    let attraction = cosine_sim.neg()?.add_scalar(1.0)?;
    let repulsion = cosine_sim.sub_scalar(margin)?.clamp_min(0.0)?;
    let loss = attraction
        .mul_op(&positive_mask)?
        .add(&repulsion.mul_op(&negative_mask)?)?;

    apply_reduction(&loss, reduction, None, &[])
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Floor applied to a sum of squares (or of `|d|^p`) before the root that turns
/// it into a distance.
///
/// `sqrt` and `x^(1/p)` have an infinite derivative at zero, so a pair of
/// *identical* embeddings — a perfectly ordinary thing to hand a metric-learning
/// loss — would otherwise back-propagate `inf`, and `0 * inf = NaN` would poison
/// the whole batch through the mask multiplications. Clamping the radicand
/// instead gives a subgradient of `0` at the kink (`clamp_min` records
/// `Operation::ClampBounds`, whose backward is zero wherever the clamp bit),
/// which is a valid choice from the subdifferential there.
///
/// `1e-12` moves a distance by at most `1e-6`, four orders of magnitude below
/// the `1e-5` forward tolerance the loss regression tests use, so no pinned
/// value shifts.
const DISTANCE_FLOOR: f32 = 1e-12;

/// The axes that make up one sample's feature vector: everything after the
/// leading batch axis.
///
/// Returns an empty slice for a rank-1 operand, which is not "reduce
/// everything" but "reduce nothing" — a `[N]` tensor is a batch of `N` scalar
/// features, and each element is already its own sample's distance. See
/// [`reduce_features`], which is where that distinction is enforced.
fn feature_axes_of(dims: &[usize]) -> Vec<i32> {
    (1..dims.len() as i32).collect()
}

/// Sums `values` over the feature axes produced by [`feature_axes_of`].
///
/// The empty-axes case must *not* be forwarded to [`Tensor::sum_dim`]: that
/// method treats an empty `dims` slice as a whole-tensor reduction, which would
/// collapse a `[N]` batch into a single scalar and silently change every rank-1
/// caller's answer.
fn reduce_features(values: Tensor, feature_axes: &[i32]) -> Result<Tensor> {
    if feature_axes.is_empty() {
        Ok(values)
    } else {
        values.sum_dim(feature_axes, false)
    }
}

/// Per-sample `p`-norm distance `||x1 - x2||_p`, reduced over `feature_axes`.
///
/// Written as `(sum |x1 - x2|^p)^(1/p)` out of recording ops
/// ([`Tensor::abs`], [`Tensor::pow`], [`Tensor::sum_dim`]) so the distance —
/// and therefore any loss built on it — stays on the autograd graph. The
/// radicand is floored by [`DISTANCE_FLOOR`]; see that constant for why.
fn p_norm_distance(x1: &Tensor, x2: &Tensor, p: f32, feature_axes: &[i32]) -> Result<Tensor> {
    let powered = x1.sub(x2)?.abs()?.pow(p)?;
    let summed = reduce_features(powered, feature_axes)?;
    summed.clamp_min(DISTANCE_FLOOR)?.pow(1.0 / p)
}

/// Helper function to apply reduction to loss tensors
fn apply_reduction(
    loss: &Tensor,
    reduction: Reduction, // was &str
    ignore_index: Option<i64>,
    target_data: &[i64],
) -> Result<Tensor> {
    // Handle ignore_index special case first
    if let Some(ignore_idx) = ignore_index {
        let valid_count = target_data.iter().filter(|&&idx| idx != ignore_idx).count() as f32;
        if valid_count > 0.0 {
            match reduction {
                Reduction::Mean => {
                    let sum = loss.sum()?;
                    sum.div_scalar(valid_count)?.view(&[1])
                }
                Reduction::Sum => loss.sum()?.view(&[1]),
                Reduction::None => Ok(loss.clone()),
                Reduction::BatchMean => {
                    let sum = loss.sum()?;
                    sum.div_scalar(valid_count)?.view(&[1])
                }
            }
        } else {
            loss.mean(None, false)
        }
    } else {
        // No ignore_index — delegate to ReductionExt
        // For Mean/Sum we need batch_size for BatchMean distinction
        let batch_size = loss.shape().dims().first().copied().unwrap_or(1);
        reduction.apply(loss, Some(batch_size))
    }
}

/// Helper function to gather target probabilities
#[allow(dead_code)]
fn gather_target_probs(probs: &Tensor, target: &Tensor) -> Result<Tensor> {
    // Simple implementation - in a real scenario you'd want proper gathering
    // This is a placeholder that assumes target contains class indices
    let target_shape = target.shape();
    let probs_shape = probs.shape();

    if target_shape.dims().len() + 1 != probs_shape.dims().len() {
        return Err(TorshError::InvalidArgument(
            "Target and input tensor shapes are incompatible for gathering".to_string(),
        ));
    }

    // For now, return a simplified version - proper gathering would be more complex
    let batch_size = target_shape.dims()[0];
    let flat_size = target_shape.numel();

    // Create a result tensor with the same shape as target
    let mut result_data = Vec::with_capacity(flat_size);
    let target_data = target.to_vec()?;
    let probs_data = probs.to_vec()?;
    let num_classes = probs_shape.dims()[probs_shape.dims().len() - 1];

    for (i, &target_class) in target_data.iter().enumerate() {
        let target_idx = target_class as usize;
        if target_idx < num_classes {
            let prob_idx = (i / batch_size) * num_classes + target_idx;
            result_data.push(probs_data[prob_idx]);
        } else {
            result_data.push(0.0);
        }
    }

    Tensor::from_vec(result_data, target_shape.dims())
}

/// Helper function to compute pairwise distance with p-norm
#[allow(dead_code)]
fn pairwise_distance(x1: &Tensor, x2: &Tensor, p: f32) -> Result<Tensor> {
    let diff = x1.sub(x2)?;
    let abs_diff = diff.abs()?;

    if p == 2.0 {
        // L2 distance: sqrt(sum((x1 - x2)^2))
        let squared = abs_diff.pow(2.0)?;
        let sum_squared = squared.sum()?;
        sum_squared.sqrt()
    } else if p == 1.0 {
        // L1 distance: sum(|x1 - x2|)
        abs_diff.sum()
    } else {
        // General p-norm: (sum(|x1 - x2|^p))^(1/p)
        let powered = abs_diff.pow(p)?;
        let sum_powered = powered.sum()?;
        let inv_p = 1.0 / p;
        sum_powered.pow(inv_p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triplet_margin_loss_basic() -> Result<()> {
        // Create simple embeddings where positive is closer to anchor than negative
        // Anchor: [1, 1]
        // Positive: [1.1, 1.1] (close to anchor)
        // Negative: [5, 5] (far from anchor)
        let anchor = Tensor::from_vec(vec![1.0, 1.0], &[1, 2])?;
        let positive = Tensor::from_vec(vec![1.1, 1.1], &[1, 2])?;
        let negative = Tensor::from_vec(vec![5.0, 5.0], &[1, 2])?;

        let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, Reduction::Mean)?;
        let loss_data = loss.to_vec()?;

        // Distance anchor-positive: sqrt((0.1)^2 + (0.1)^2) ≈ 0.141
        // Distance anchor-negative: sqrt(16 + 16) ≈ 5.657
        // Loss should be 0 since dist_an - dist_ap > margin
        assert!(
            loss_data[0] < 0.1,
            "Loss should be near zero when constraint is satisfied"
        );

        Ok(())
    }

    #[test]
    fn test_triplet_margin_loss_violation() -> Result<()> {
        // Create case where margin is violated
        // Anchor: [0, 0]
        // Positive: [2, 0] (distance = 2)
        // Negative: [1, 0] (distance = 1, closer than positive!)
        let anchor = Tensor::from_vec(vec![0.0, 0.0], &[1, 2])?;
        let positive = Tensor::from_vec(vec![2.0, 0.0], &[1, 2])?;
        let negative = Tensor::from_vec(vec![1.0, 0.0], &[1, 2])?;

        let loss = triplet_margin_loss(&anchor, &positive, &negative, 0.5, 2.0, Reduction::Mean)?;
        let loss_data = loss.to_vec()?;

        // dist_ap = 2.0, dist_an = 1.0, margin = 0.5
        // Loss = max(2.0 - 1.0 + 0.5, 0) = 1.5
        assert!((loss_data[0] - 1.5).abs() < 1e-5, "Loss should be 1.5");

        Ok(())
    }

    #[test]
    fn test_triplet_margin_loss_batch() -> Result<()> {
        // Test with batch size 2
        let anchor = Tensor::from_vec(
            vec![
                0.0, 0.0, // Sample 1
                1.0, 1.0, // Sample 2
            ],
            &[2, 2],
        )?;
        let positive = Tensor::from_vec(
            vec![
                0.1, 0.1, // Sample 1
                1.1, 1.1, // Sample 2
            ],
            &[2, 2],
        )?;
        let negative = Tensor::from_vec(
            vec![
                5.0, 5.0, // Sample 1
                6.0, 6.0, // Sample 2
            ],
            &[2, 2],
        )?;

        let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, Reduction::None)?;
        assert_eq!(loss.shape().dims(), &[2]);

        let loss_mean = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, Reduction::Mean)?;
        assert_eq!(loss_mean.shape().dims(), &[1]);

        Ok(())
    }

    #[test]
    fn test_contrastive_loss_similar_pairs() -> Result<()> {
        // Test with similar pairs (target=1)
        // Similar pairs should have small distance
        let output1 = Tensor::from_vec(vec![1.0, 2.0], &[1, 2])?;
        let output2 = Tensor::from_vec(vec![1.1, 2.1], &[1, 2])?;
        let target = Tensor::from_vec(vec![1.0], &[1])?;

        let loss = contrastive_loss(&output1, &output2, &target, 2.0, Reduction::Mean)?;
        let loss_data = loss.to_vec()?;

        // Distance^2 = (0.1)^2 + (0.1)^2 = 0.02
        // For similar pairs: loss = d^2 = 0.02
        assert!(
            (loss_data[0] - 0.02).abs() < 1e-5,
            "Loss for similar pair should be distance squared"
        );

        Ok(())
    }

    #[test]
    fn test_contrastive_loss_dissimilar_pairs() -> Result<()> {
        // Test with dissimilar pairs (target=0)
        // Dissimilar pairs should be pushed apart
        let output1 = Tensor::from_vec(vec![0.0, 0.0], &[1, 2])?;
        let output2 = Tensor::from_vec(vec![0.5, 0.0], &[1, 2])?;
        let target = Tensor::from_vec(vec![0.0], &[1])?;

        let loss = contrastive_loss(&output1, &output2, &target, 2.0, Reduction::Mean)?;
        let loss_data = loss.to_vec()?;

        // Distance = 0.5, margin = 2.0
        // For dissimilar pairs: loss = max(margin - d, 0)^2 = (2.0 - 0.5)^2 = 2.25
        assert!(
            (loss_data[0] - 2.25).abs() < 1e-5,
            "Loss for dissimilar pair should be (margin - dist)^2"
        );

        Ok(())
    }

    #[test]
    fn test_contrastive_loss_dissimilar_beyond_margin() -> Result<()> {
        // Test dissimilar pairs that are already beyond margin
        let output1 = Tensor::from_vec(vec![0.0, 0.0], &[1, 2])?;
        let output2 = Tensor::from_vec(vec![5.0, 0.0], &[1, 2])?;
        let target = Tensor::from_vec(vec![0.0], &[1])?;

        let loss = contrastive_loss(&output1, &output2, &target, 2.0, Reduction::Mean)?;
        let loss_data = loss.to_vec()?;

        // Distance = 5.0 > margin = 2.0
        // Loss should be 0 (already separated enough)
        assert!(
            loss_data[0] < 1e-5,
            "Loss should be zero when dissimilar pairs are beyond margin"
        );

        Ok(())
    }

    #[test]
    fn test_contrastive_loss_batch() -> Result<()> {
        // Test with multiple samples
        let output1 = Tensor::from_vec(
            vec![
                0.0, 0.0, // Sample 1
                1.0, 1.0, // Sample 2
            ],
            &[2, 2],
        )?;
        let output2 = Tensor::from_vec(
            vec![
                0.1, 0.0, // Sample 1
                1.0, 2.0, // Sample 2
            ],
            &[2, 2],
        )?;
        let target = Tensor::from_vec(
            vec![
                1.0, // Similar
                0.0, // Dissimilar
            ],
            &[2],
        )?;

        let loss = contrastive_loss(&output1, &output2, &target, 2.0, Reduction::None)?;
        assert_eq!(loss.shape().dims(), &[2]);

        let loss_mean = contrastive_loss(&output1, &output2, &target, 2.0, Reduction::Mean)?;
        assert_eq!(loss_mean.shape().dims(), &[1]);

        Ok(())
    }

    #[test]
    fn test_reduction_modes() -> Result<()> {
        let anchor = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[2, 2])?;
        let positive = Tensor::from_vec(vec![0.1, 0.1, 1.1, 1.1], &[2, 2])?;
        let negative = Tensor::from_vec(vec![5.0, 5.0, 6.0, 6.0], &[2, 2])?;

        // Test "none" reduction
        let loss_none = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, Reduction::None)?;
        assert_eq!(
            loss_none.shape().dims(),
            &[2],
            "none reduction should return batch_size losses"
        );

        // Test "mean" reduction
        let loss_mean = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, Reduction::Mean)?;
        assert_eq!(
            loss_mean.shape().dims(),
            &[1],
            "mean reduction should return scalar"
        );

        // Test "sum" reduction
        let loss_sum = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, Reduction::Sum)?;
        assert_eq!(
            loss_sum.shape().dims(),
            &[1],
            "sum reduction should return scalar"
        );

        Ok(())
    }
}

// =============================================================================
// MODERN LOSS FUNCTIONS
// =============================================================================

/// Dice Loss for segmentation tasks
///
/// Dice loss is particularly effective for imbalanced segmentation problems
/// where the foreground class is much smaller than the background.
///
/// # Arguments
/// * `input` - Predicted probabilities (after sigmoid/softmax) with shape (batch_size, num_classes, ...)
/// * `target` - Ground truth labels with same shape as input
/// * `smooth` - Smoothing factor to avoid division by zero (typically 1.0)
/// * `reduction` - Reduction mode: "mean", "sum", or "none"
///
/// # Formula
/// Dice = 1 - (2 * intersection + smooth) / (sum_pred + sum_target + smooth)
///
/// # Autograd
///
/// The three sums, the ratio and the `1 - x` are all recording tensor ops, so
/// the coefficient stays attached to `input`. The predecessor computed exactly
/// the same three sums *as tensors* and then read them out with `to_vec()` to do
/// the arithmetic in `f32`, which threw the graph away one step before the
/// finish line.
///
/// # Reference
/// Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 3DV 2016
pub fn dice_loss(input: &Tensor, target: &Tensor, smooth: f32, reduction: Reduction) -> Result<Tensor> {
    // Validate inputs
    if input.shape().dims() != target.shape().dims() {
        return Err(TorshError::ShapeMismatch {
            expected: target.shape().dims().to_vec(),
            got: input.shape().dims().to_vec(),
        });
    }

    // Compute intersection and the two marginals.
    let intersection_sum = input.mul_op(target)?.sum()?;
    let input_sum = input.sum()?;
    let target_sum = target.sum()?;

    // Dice = (2 * intersection + smooth) / (sum_pred + sum_target + smooth)
    let numerator = intersection_sum.mul_scalar(2.0)?.add_scalar(smooth)?;
    let denominator = input_sum.add(&target_sum)?.add_scalar(smooth)?;
    let dice_coeff = numerator.div(&denominator)?;

    // `sum()` returns a rank-0 tensor; the historical output of this function is
    // shape `[1]`, and `"none"` hands that straight back.
    let loss = dice_coeff.neg()?.add_scalar(1.0)?.view(&[1])?;
    apply_reduction(&loss, reduction, None, &[])
}

/// Tversky Loss for imbalanced segmentation
///
/// Tversky loss is a generalization of Dice loss that allows controlling the
/// balance between false positives and false negatives through alpha and beta parameters.
///
/// # Arguments
/// * `input` - Predicted probabilities (after sigmoid/softmax) with shape (batch_size, ...)
/// * `target` - Ground truth labels with same shape as input
/// * `alpha` - Weight for false positives (typically 0.3-0.7)
/// * `beta` - Weight for false negatives (typically 0.3-0.7, alpha + beta should be ≤ 1)
/// * `smooth` - Smoothing factor to avoid division by zero (typically 1.0)
/// * `reduction` - Reduction mode: "mean", "sum", or "none"
///
/// # Formula
/// Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
/// Loss = 1 - Tversky
///
/// # Autograd
///
/// Same shape of fix as [`dice_loss`]: the TP/FP/FN sums were already tensors,
/// so the index and the `1 - index` are computed with recording ops instead of
/// being read out with `to_vec()` and rebuilt as a detached leaf.
///
/// # Reference
/// Salehi et al., "Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks", MICCAI 2017
pub fn tversky_loss(
    input: &Tensor,
    target: &Tensor,
    alpha: f32,
    beta: f32,
    smooth: f32,
    reduction: Reduction,
) -> Result<Tensor> {
    // Validate inputs
    if input.shape().dims() != target.shape().dims() {
        return Err(TorshError::ShapeMismatch {
            expected: target.shape().dims().to_vec(),
            got: input.shape().dims().to_vec(),
        });
    }

    if alpha + beta > 1.0 {
        return Err(TorshError::InvalidArgument(
            "alpha + beta should be <= 1.0 for Tversky loss".to_string(),
        ));
    }

    // Compute true positives (TP), false positives (FP), false negatives (FN)
    let tp = input.mul_op(target)?.sum()?;

    let ones = torsh_tensor::creation::ones_like(target)?;
    let fp = input.mul_op(&ones.sub(target)?)?.sum()?;
    let fn_tensor = ones.sub(input)?.mul_op(target)?.sum()?;

    // Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    let numerator = tp.add_scalar(smooth)?;
    let denominator = tp
        .add(&fp.mul_scalar(alpha)?)?
        .add(&fn_tensor.mul_scalar(beta)?)?
        .add_scalar(smooth)?;
    let tversky_index = numerator.div(&denominator)?;

    let loss = tversky_index.neg()?.add_scalar(1.0)?.view(&[1])?;
    apply_reduction(&loss, reduction, None, &[])
}

/// Wing Loss for robust regression
///
/// Wing loss is designed to be more robust to outliers than MSE and L1 loss,
/// particularly effective for facial landmark detection and other regression tasks
/// with varying difficulty across samples.
///
/// # Arguments
/// * `input` - Predicted values with shape (batch_size, ...)
/// * `target` - Ground truth values with same shape as input
/// * `width` - Width parameter controlling the transition between L1 and log (typically 5.0-10.0)
/// * `curvature` - Curvature parameter epsilon (typically 0.5-2.0)
/// * `reduction` - Reduction mode: "mean", "sum", or "none"
///
/// # Formula
/// For |x| < width:
///   loss = width * ln(1 + |x|/epsilon)
/// For |x| >= width:
///   loss = |x| - C
/// where C is a constant ensuring continuity
///
/// # Autograd
///
/// Expressed as `width * ln(1 + inside/curvature) + outside` over the band split
/// of `banded_abs_error` with the band set to `width`. Inside the band that is
/// the logarithmic arm verbatim; outside it, `inside == width` and the result is
/// `width * ln(1 + width/curvature) + (|d| - width) == |d| - C`, which is the
/// linear arm with exactly the continuity constant `C` the predecessor computed
/// in `f32`. The predecessor mapped over `to_vec()` output and rebuilt the
/// tensor with `Tensor::from_data`, so the loss was a detached leaf.
///
/// `width == 0` needs no special case: the band is empty, `inside` is zero,
/// `ln(1) == 0` and the loss degenerates to `|d|` — which is what the
/// element-wise form produced too (`C == 0` there).
///
/// # Reference
/// Feng et al., "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks", CVPR 2018
pub fn wing_loss(
    input: &Tensor,
    target: &Tensor,
    width: f32,
    curvature: f32,
    reduction: Reduction,
) -> Result<Tensor> {
    // Validate inputs
    if input.shape().dims() != target.shape().dims() {
        return Err(TorshError::ShapeMismatch {
            expected: target.shape().dims().to_vec(),
            got: input.shape().dims().to_vec(),
        });
    }

    let (inside, outside) = banded_abs_error(input, target, width)?;
    let logarithmic = inside
        .div_scalar(curvature)?
        .add_scalar(1.0)?
        .log()?
        .mul_scalar(width)?;
    let loss = logarithmic.add(&outside)?;

    apply_reduction(&loss, reduction, None, &[])
}

/// Center Loss for metric learning
///
/// Center loss learns a center for each class and penalizes the distance of samples
/// from their corresponding class centers. Often used together with softmax loss
/// for improved feature discrimination in face recognition and person re-identification.
///
/// # Arguments
/// * `features` - Feature embeddings with shape (batch_size, feature_dim)
/// * `labels` - Class labels with shape (batch_size,)
/// * `centers` - Class centers with shape (num_classes, feature_dim)
/// * `reduction` - Reduction mode: "mean", "sum", or "none"
///
/// # Formula
/// loss = 0.5 * sum((features - `centers[labels]`)^2)
///
/// # Autograd
///
/// The per-sample centre is selected by multiplying `centers` on the left by a
/// constant one-hot assignment matrix `[batch_size, num_classes]`, so the loss
/// stays attached to *both* `features` and `centers` — which is what makes the
/// usual "SGD on the centres alongside the softmax loss" training loop possible
/// at all. The predecessor read every operand out with `to_vec()` and rebuilt
/// the per-sample losses through `Tensor::from_data`, i.e. a detached leaf.
///
/// [`Tensor::index_select`] and [`Tensor::gather`] would express the selection
/// more directly, but neither records (measured), so the matmul is the only
/// graph-preserving spelling available.
///
/// # Non-finite inputs
///
/// The one-hot matmul is a masked sum over *all* the centres, not a row read, so
/// a non-finite value anywhere in `centers` makes every selected centre `NaN`
/// (`0.0 * inf` is `NaN`) even for classes that value has nothing to do with.
/// The element-wise predecessor touched only the selected row and was immune;
/// that immunity is not recoverable while staying differentiable. Same property,
/// and same reasoning, as [`nll_loss`]'s masked gather.
///
/// # Reference
/// Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition", ECCV 2016
pub fn center_loss(
    features: &Tensor,
    labels: &Tensor<i64>,
    centers: &Tensor,
    reduction: Reduction,
) -> Result<Tensor> {
    let features_shape_binding = features.shape();
    let features_shape = features_shape_binding.dims();
    let (batch_size, feature_dim) = match features_shape {
        [batch, dim] => (*batch, *dim),
        _ => {
            return Err(TorshError::InvalidShape(format!(
                "center_loss expects 2-D features [N, D], got shape {features_shape:?}"
            )))
        }
    };

    let centers_shape_binding = centers.shape();
    let centers_shape = centers_shape_binding.dims();
    let num_classes = centers_shape[0];

    // Validate dimensions
    if centers_shape.len() != 2 || centers_shape[1] != feature_dim {
        return Err(TorshError::ShapeMismatch {
            expected: vec![num_classes, feature_dim],
            got: centers_shape.to_vec(),
        });
    }

    let labels_vec: Vec<i64> = labels.to_vec()?;
    if labels_vec.len() != batch_size {
        return Err(TorshError::InvalidShape(format!(
            "center_loss expects one label per sample: features have {batch_size} rows \
             but labels have {} entries",
            labels_vec.len()
        )));
    }

    // Constant assignment matrix: assignment[b, labels[b]] = 1, zero elsewhere,
    // so `assignment @ centers` is the per-sample class centre.
    let mut assignment = vec![0.0f32; batch_size * num_classes];
    for (b, &label) in labels_vec.iter().enumerate() {
        // A negative label wraps to a huge `usize` and is caught by the same
        // bound check, exactly as before.
        let label_idx = label as usize;
        if label_idx >= num_classes {
            return Err(TorshError::InvalidArgument(format!(
                "Label {} out of range for {} classes",
                label, num_classes
            )));
        }
        if let Some(slot) = assignment.get_mut(b * num_classes + label_idx) {
            *slot = 1.0;
        }
    }
    let assignment_tensor =
        Tensor::from_data(assignment, vec![batch_size, num_classes], features.device())?;

    // loss_b = 0.5 * sum_j (features[b, j] - centers[label_b, j])^2
    let selected = assignment_tensor.matmul(centers)?;
    let diff = features.sub(&selected)?;
    let loss = diff.square()?.sum_dim(&[-1], false)?.mul_scalar(0.5)?;

    apply_reduction(&loss, reduction, None, &[])
}

/// InfoNCE Loss for contrastive learning
///
/// InfoNCE (Information Noise-Contrastive Estimation) loss is widely used in
/// self-supervised learning frameworks like SimCLR and MoCo. It maximizes
/// agreement between differently augmented views of the same data.
///
/// # Arguments
/// * `anchor` - Anchor embeddings with shape (batch_size, embedding_dim)
/// * `positive` - Positive (similar) embeddings with shape (batch_size, embedding_dim)
/// * `negatives` - Negative (dissimilar) embeddings with shape (num_negatives, embedding_dim)
/// * `temperature` - Temperature parameter for softmax (typically 0.1-0.5)
/// * `reduction` - Reduction mode: "mean", "sum", or "none"
///
/// # Formula
/// loss = -log(exp(sim(anchor, positive)/τ) / (exp(sim(anchor, positive)/τ) + sum(exp(sim(anchor, negative_i)/τ))))
/// where sim is cosine similarity
///
/// # Autograd
///
/// The similarity row `[sim(a, p), sim(a, n_1), ..., sim(a, n_k)] / temperature`
/// is assembled with recording ops and handed to
/// [`log_softmax`](Tensor::log_softmax); the loss is minus the first column,
/// selected with a constant one-hot and a row sum. That is the same
/// max-subtracted log-sum-exp the predecessor computed by hand, so the numbers
/// are unchanged, but `backward()` now reaches `anchor`, `positive` *and*
/// `negatives`. The predecessor read all three operands out with `to_vec()` and
/// rebuilt the per-sample losses through `Tensor::from_data`.
///
/// # Non-finite inputs
///
/// The predecessor's `cosine_similarity` helper special-cased a zero-norm
/// embedding to a similarity of exactly `0.0`. That forward value is preserved:
/// a constant unit fallback is added to the norm wherever it is exactly zero, so
/// the division is `0 / 1` and the similarity is still `0.0`. The *gradient* at
/// such a row is not finite — `d/dx sqrt(x)` diverges at `x == 0` — which is a
/// property of the cosine similarity itself at the origin, not of this
/// formulation.
///
/// # Reference
/// Oord et al., "Representation Learning with Contrastive Predictive Coding", arXiv 2018
/// Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
pub fn infonce_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negatives: &Tensor,
    temperature: f32,
    reduction: Reduction,
) -> Result<Tensor> {
    let anchor_shape_binding = anchor.shape();
    let anchor_shape = anchor_shape_binding.dims();
    let (batch_size, embedding_dim) = match anchor_shape {
        [batch, dim] => (*batch, *dim),
        _ => {
            return Err(TorshError::InvalidShape(format!(
                "infonce_loss expects a 2-D anchor [N, D], got shape {anchor_shape:?}"
            )))
        }
    };

    // Validate shapes
    let positive_shape = positive.shape();
    if positive_shape.dims() != anchor_shape {
        return Err(TorshError::ShapeMismatch {
            expected: anchor_shape.to_vec(),
            got: positive_shape.dims().to_vec(),
        });
    }

    let negatives_shape_binding = negatives.shape();
    let negatives_shape = negatives_shape_binding.dims();
    if negatives_shape.len() != 2 || negatives_shape[1] != embedding_dim {
        return Err(TorshError::ShapeMismatch {
            expected: vec![negatives_shape[0], embedding_dim],
            got: negatives_shape.to_vec(),
        });
    }

    let num_negatives = negatives_shape[0];

    let anchor_norm = row_norms(anchor)?;
    let positive_norm = row_norms(positive)?;

    // sim(anchor_b, positive_b) / temperature, shaped [batch_size, 1]
    let positive_similarity = anchor
        .mul_op(positive)?
        .sum_dim(&[-1], true)?
        .div(&anchor_norm.mul_op(&positive_norm)?)?
        .div_scalar(temperature)?;

    // The positive is column 0 of the logit row; the negatives follow it.
    let logits = if num_negatives == 0 {
        positive_similarity
    } else {
        let negative_norm = row_norms(negatives)?;
        let negative_similarity = anchor
            .matmul(&negatives.transpose(0, 1)?)?
            .div(&anchor_norm.matmul(&negative_norm.transpose(0, 1)?)?)?
            .div_scalar(temperature)?;
        Tensor::cat(&[&positive_similarity, &negative_similarity], 1)?
    };

    // loss_b = -log_softmax(logits)[b, 0]
    let log_probs = logits.log_softmax(-1)?;
    let row_width = num_negatives + 1;
    let mut selector = vec![0.0f32; batch_size * row_width];
    for b in 0..batch_size {
        if let Some(slot) = selector.get_mut(b * row_width) {
            *slot = 1.0;
        }
    }
    let selector_tensor =
        Tensor::from_data(selector, vec![batch_size, row_width], anchor.device())?;
    let loss = log_probs
        .mul_op(&selector_tensor)?
        .sum_dim(&[-1], false)?
        .neg()?;

    apply_reduction(&loss, reduction, None, &[])
}

/// Row-wise Euclidean norms of a `[rows, dim]` tensor, shaped `[rows, 1]`, with
/// a constant unit fallback wherever the norm is exactly zero.
///
/// The fallback keeps [`infonce_loss`]'s forward pass bit-identical to its
/// element-wise predecessor, which returned a cosine similarity of `0.0` rather
/// than `0/0` for a zero embedding: a zero row divided by the substituted `1`
/// is still zero, so the similarity is still `0.0`. The substitution is an
/// *additive constant*, so it changes nothing about how the gradient flows
/// through `norm` for the rows that are not degenerate.
fn row_norms(rows: &Tensor) -> Result<Tensor> {
    let norm = rows.square()?.sum_dim(&[-1], true)?.sqrt()?;

    let norm_values = norm.to_vec()?;
    if !norm_values.iter().any(|value| *value == 0.0) {
        return Ok(norm);
    }

    let fallback: Vec<f32> = norm_values
        .iter()
        .map(|value| if *value == 0.0 { 1.0 } else { 0.0 })
        .collect();
    let fallback_tensor = Tensor::from_data(fallback, norm.shape().dims().to_vec(), rows.device())?;
    norm.add(&fallback_tensor)
}
