//! Basic loss functions for neural networks
//!
//! This module provides fundamental loss functions including cross entropy,
//! MSE, L1, binary cross entropy, KL divergence, and specialized losses.

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// =============================================================================
// CLASSIFICATION LOSSES
// =============================================================================

/// Cross entropy loss function
/// Enhanced with SciRS2-inspired numerical stability and efficiency
pub fn cross_entropy(
    input: &Tensor,
    target: &Tensor<i64>,
    weight: Option<&Tensor>,
    reduction: &str,
    ignore_index: Option<i64>,
) -> Result<Tensor> {
    // Enhanced cross entropy implementation using numerically stable log_softmax

    // Apply log_softmax for numerical stability
    let log_probs = crate::functional::activation::log_softmax(input, Some(-1))?;

    // Convert targets to appropriate format and compute negative log likelihood
    let target_shape_binding = target.shape();
    let target_shape = target_shape_binding.dims();
    let batch_size = target_shape[0];
    let num_classes = input.shape().dims()[1];

    // Create one-hot encoding for targets
    let mut one_hot_data = vec![0.0f32; batch_size * num_classes];
    let target_vec = target.to_vec()?;

    for (i, &target_idx) in target_vec.iter().enumerate() {
        if let Some(ignore_idx) = ignore_index {
            if target_idx == ignore_idx {
                continue; // Skip ignored indices
            }
        }
        if target_idx >= 0 && (target_idx as usize) < num_classes {
            one_hot_data[i * num_classes + target_idx as usize] = 1.0;
        }
    }

    let one_hot = Tensor::from_data(
        one_hot_data,
        vec![batch_size, num_classes],
        torsh_core::device::DeviceType::Cpu,
    )?;

    // Compute negative log likelihood: -sum(one_hot * log_probs)
    let neg_log_likelihood = log_probs.mul_op(&one_hot)?.neg()?;
    let loss_per_sample = neg_log_likelihood.sum_dim(&[-1], false)?;

    // Apply class weights if provided
    let weighted_loss = if let Some(weights) = weight {
        // Apply weights based on target classes
        let mut weight_data = vec![1.0f32; batch_size];
        for (i, &target_idx) in target_vec.iter().enumerate() {
            if target_idx >= 0 && (target_idx as usize) < weights.shape().dims()[0] {
                let weight_vec = weights.to_vec()?;
                weight_data[i] = weight_vec[target_idx as usize];
            }
        }
        let weight_tensor = Tensor::from_data(
            weight_data,
            vec![batch_size],
            torsh_core::device::DeviceType::Cpu,
        )?;
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
    reduction: &str,
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
    reduction: &str,
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
pub fn multi_margin_loss(
    input: &Tensor,
    target: &Tensor<i64>,
    p: i32,
    margin: f32,
    weight: Option<&Tensor>,
    reduction: &str,
) -> Result<Tensor> {
    // Multi-class margin loss
    let batch_size = input.shape().dims()[0];
    let num_classes = input.shape().dims()[1];
    let target_data = target.to_vec()?;
    let input_data = input.to_vec()?;

    let mut loss_data = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let true_class = target_data[b] as usize;
        if true_class >= num_classes {
            loss_data.push(0.0);
            continue;
        }

        let true_score = input_data[b * num_classes + true_class];
        let mut sample_loss = 0.0;

        for c in 0..num_classes {
            if c == true_class {
                continue;
            }
            let wrong_score = input_data[b * num_classes + c];
            let margin_diff = margin - true_score + wrong_score;
            if margin_diff > 0.0 {
                if p == 1 {
                    sample_loss += margin_diff;
                } else {
                    sample_loss += margin_diff.powi(p);
                }
            }
        }

        // Apply class weight if provided
        if let Some(weights) = weight {
            let weight_data = weights.to_vec()?;
            if true_class < weight_data.len() {
                sample_loss *= weight_data[true_class];
            }
        }

        loss_data.push(sample_loss / (num_classes - 1) as f32);
    }

    let loss_tensor = Tensor::from_vec(loss_data, &[batch_size])?;
    apply_reduction(&loss_tensor, reduction, None, &[])
}

/// Multilabel margin loss function
pub fn multilabel_margin_loss(input: &Tensor, target: &Tensor, reduction: &str) -> Result<Tensor> {
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
pub fn mse_loss(input: &Tensor, target: &Tensor, reduction: &str) -> Result<Tensor> {
    // MSE: (input - target)^2
    let diff = input.sub(target)?;
    let squared_diff = diff.mul_op(&diff)?;

    apply_reduction(&squared_diff, reduction, None, &[])
}

/// L1 (Mean Absolute Error) loss function
pub fn l1_loss(input: &Tensor, target: &Tensor, reduction: &str) -> Result<Tensor> {
    // L1: |input - target|
    let diff = input.sub(target)?;
    let abs_diff = diff.abs()?;

    apply_reduction(&abs_diff, reduction, None, &[])
}

/// Smooth L1 loss function (Huber loss)
pub fn smooth_l1_loss(
    input: &Tensor,
    target: &Tensor,
    beta: f32,
    reduction: &str,
) -> Result<Tensor> {
    // Smooth L1 loss:
    // loss = 0.5 * (pred - target)^2 / beta  if |pred - target| < beta
    // loss = |pred - target| - 0.5 * beta    otherwise

    let diff = input.sub(target)?;
    let abs_diff = diff.abs()?;
    let abs_diff_data = abs_diff.to_vec()?;
    let diff_data = diff.to_vec()?;

    let mut loss_data = Vec::new();

    for i in 0..abs_diff_data.len() {
        let abs_val = abs_diff_data[i];
        let diff_val = diff_data[i];

        let loss_val = if abs_val < beta {
            0.5 * diff_val * diff_val / beta
        } else {
            abs_val - 0.5 * beta
        };

        loss_data.push(loss_val);
    }

    let loss_tensor = Tensor::from_vec(loss_data, input.shape().dims())?;
    apply_reduction(&loss_tensor, reduction, None, &[])
}

/// Huber loss function
pub fn huber_loss(input: &Tensor, target: &Tensor, delta: f32, reduction: &str) -> Result<Tensor> {
    // Huber loss: 0.5 * (x - y)^2 if |x - y| <= delta, delta * (|x - y| - 0.5 * delta) otherwise
    let diff = input.sub(target)?;
    let abs_diff = diff.abs()?;

    // Create delta tensor
    let delta_tensor = torsh_tensor::creation::full(input.shape().dims(), delta)?;

    // Mask for small errors (|diff| <= delta)
    let small_errors = abs_diff.le(&delta_tensor)?;

    // L2 loss for small errors: 0.5 * diff^2
    let diff_squared = diff.pow(2.0)?;
    let l2_loss = diff_squared.mul(&torsh_tensor::creation::tensor_scalar(0.5)?)?;

    // L1 loss for large errors: delta * (|diff| - 0.5 * delta)
    let delta_half = torsh_tensor::creation::tensor_scalar(0.5 * delta)?;
    let abs_diff_minus_half_delta = abs_diff.sub(&delta_half)?;
    let l1_loss = delta_tensor.mul(&abs_diff_minus_half_delta)?;

    // Select based on error size
    let loss = l2_loss.where_tensor(&small_errors, &l1_loss)?;

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
    reduction: &str,
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
        "mean" => {
            // Mean over all elements
            kl_elements.mean(None, false)
        }
        "sum" => {
            // Sum over all elements
            kl_elements.sum()
        }
        "batchmean" => {
            // Sum over all dimensions except batch, then mean over batch
            let batch_size = input.shape().dims()[0] as f32;
            let total_sum = kl_elements.sum()?;
            let batch_size_tensor = torsh_tensor::creation::full(&[1], batch_size)?;
            total_sum.div(&batch_size_tensor)
        }
        "none" => {
            // No reduction, return element-wise losses
            Ok(kl_elements)
        }
        _ => Err(TorshError::InvalidArgument(format!(
            "Unknown reduction: {}. Expected 'mean', 'sum', 'batchmean', or 'none'",
            reduction
        ))),
    }
}

/// Negative log likelihood loss function
pub fn nll_loss(
    input: &Tensor,
    target: &Tensor<i64>,
    weight: Option<&Tensor>,
    ignore_index: Option<i64>,
    reduction: &str,
) -> Result<Tensor> {
    // NLL loss assumes log-probabilities as input
    let batch_size = input.shape().dims()[0];
    let num_classes = input.shape().dims()[1];
    let target_data = target.to_vec()?;
    let log_prob_data = input.to_vec()?;

    let mut loss_data = Vec::new();
    for b in 0..batch_size {
        let target_class = target_data[b] as usize;
        if let Some(ignore_idx) = ignore_index {
            if target_data[b] as i64 == ignore_idx {
                loss_data.push(0.0);
                continue;
            }
        }

        if target_class < num_classes {
            let mut nll = -log_prob_data[b * num_classes + target_class];

            // Apply class weight if provided
            if let Some(weights) = weight {
                let weight_data = weights.to_vec()?;
                if target_class < weight_data.len() {
                    nll *= weight_data[target_class];
                }
            }

            loss_data.push(nll);
        } else {
            loss_data.push(0.0);
        }
    }

    let loss_tensor = Tensor::from_vec(loss_data, &[batch_size])?;
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
pub fn focal_loss(
    input: &Tensor,
    target: &Tensor<i64>,
    alpha: Option<f32>,
    gamma: f32,
    reduction: &str,
) -> Result<Tensor> {
    // Apply softmax to get probabilities
    let log_probs = input.log_softmax(-1)?;

    // Gather the log probabilities for the true classes
    // This is a simplified implementation - a full implementation would need proper gather operation
    let batch_size = input.shape().dims()[0];
    let num_classes = input.shape().dims()[1];

    // For now, implement a simplified version that returns a scalar loss
    // TODO: Implement proper focal loss with scirs2 operations when available

    // Convert log probabilities to probabilities
    let probs = log_probs.exp()?;

    // Create alpha weights if provided
    let alpha_weight = alpha.unwrap_or(1.0);

    // Simplified focal loss computation
    // In a full implementation, we would:
    // 1. Gather log_probs[i, target[i]] for each sample i
    // 2. Compute pt = exp(log_pt) for true class probabilities
    // 3. Apply focal term: alpha * (1 - pt)^gamma * log_pt
    // 4. Apply reduction (mean, sum, none)

    // For now, return a placeholder loss tensor
    let mut loss_val = 0.0f32;

    // Simple approximation of focal loss behavior
    let target_data = target.to_vec()?;
    for i in 0..batch_size {
        for j in 0..num_classes {
            let prob = probs.to_vec()?[i * num_classes + j];
            let log_prob = log_probs.to_vec()?[i * num_classes + j];

            // Check if this is the target class
            if j == target_data[i] as usize {
                // Simplified focal loss term
                let focal_weight = alpha_weight * (1.0 - prob).powf(gamma);
                loss_val += focal_weight * (-log_prob);
            }
        }
    }

    match reduction {
        "mean" => loss_val /= batch_size as f32,
        "sum" => {} // Keep as is
        "none" => {
            // Should return per-sample losses, but simplified to scalar for now
        }
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid reduction mode: {}",
                reduction
            )))
        }
    }

    // Return scalar loss
    Ok(Tensor::from_data(vec![loss_val], vec![1], input.device())?)
}

/// Triplet margin loss function
///
/// Computes the triplet loss between anchor, positive, and negative samples.
/// L = max(d(a,p) - d(a,n) + margin, 0)
pub fn triplet_margin_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f32,
    p: f32,
    reduction: &str,
) -> Result<Tensor> {
    // TODO: Implement proper triplet margin loss with scirs2
    // For now, return a placeholder that computes a simple distance-based loss

    let _ = (positive, negative, p); // Suppress warnings

    // Simplified implementation - return margin as loss for now
    let loss_val = margin;

    let final_loss = match reduction {
        "mean" | "sum" => loss_val,
        "none" => loss_val,
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid reduction mode: {}",
                reduction
            )))
        }
    };

    Ok(Tensor::from_data(
        vec![final_loss],
        vec![1],
        anchor.device(),
    )?)
}

/// Contrastive loss function
///
/// Used for learning embeddings where similar pairs should have small distances
/// and dissimilar pairs should have large distances.
pub fn contrastive_loss(
    output1: &Tensor,
    output2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: &str,
) -> Result<Tensor> {
    // TODO: Implement proper contrastive loss with scirs2
    // For now, return a placeholder

    let _ = (output2, target); // Suppress warnings

    let loss_val = margin;

    let final_loss = match reduction {
        "mean" | "sum" => loss_val,
        "none" => loss_val,
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid reduction mode: {}",
                reduction
            )))
        }
    };

    Ok(Tensor::from_data(
        vec![final_loss],
        vec![1],
        output1.device(),
    )?)
}

/// Cosine Embedding Loss for similarity learning
/// Encourages embeddings to have high cosine similarity for similar items (target=1)
/// and low cosine similarity for dissimilar items (target=-1)
pub fn cosine_embedding_loss(
    input1: &Tensor,
    input2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: &str,
) -> Result<Tensor> {
    // Compute cosine similarity
    let dot_product = (input1.mul(input2)?).sum()?;
    let norm1 = input1.norm()?;
    let norm2 = input2.norm()?;
    let cosine_sim = dot_product.div(&norm1.mul(&norm2)?)?;

    // Create tensors for comparison
    let one = torsh_tensor::creation::tensor_scalar(1.0)?;
    let _neg_one = torsh_tensor::creation::tensor_scalar(-1.0)?;
    let margin_tensor = torsh_tensor::creation::tensor_scalar(margin)?;
    let zero = torsh_tensor::creation::tensor_scalar(0.0)?;

    // target == 1: loss = 1 - cosine_sim
    // target == -1: loss = max(0, cosine_sim - margin)
    let positive_mask = target.eq(&one)?;

    let positive_loss = one.sub(&cosine_sim)?;
    let negative_loss_raw = cosine_sim.sub(&margin_tensor)?;
    let negative_loss = negative_loss_raw.maximum(&zero)?;

    let loss = positive_loss.where_tensor(&positive_mask, &negative_loss)?;

    apply_reduction(&loss, reduction, None, &[])
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Helper function to apply reduction to loss tensors
fn apply_reduction(
    loss: &Tensor,
    reduction: &str,
    ignore_index: Option<i64>,
    target_data: &[i64],
) -> Result<Tensor> {
    match reduction {
        "mean" => {
            if let Some(ignore_idx) = ignore_index {
                // Count non-ignored samples
                let valid_count =
                    target_data.iter().filter(|&&idx| idx != ignore_idx).count() as f32;
                if valid_count > 0.0 {
                    let sum = loss.sum()?;
                    let count_tensor = torsh_tensor::creation::full(&[1], valid_count)?;
                    sum.div(&count_tensor)
                } else {
                    loss.mean(None, false)
                }
            } else {
                loss.mean(None, false)
            }
        }
        "sum" => loss.sum(),
        "none" => Ok(loss.clone()),
        _ => Err(TorshError::ComputeError(format!(
            "Unknown reduction: {}",
            reduction
        ))),
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
