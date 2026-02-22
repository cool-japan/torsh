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
    // Focal Loss: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    // where pt is the probability of the true class

    let input_shape = input.shape();
    let input_dims = input_shape.dims();

    if input_dims.len() != 2 {
        return Err(torsh_core::error::TorshError::InvalidShape(format!(
            "Input must be 2D [batch_size, num_classes], got shape {:?}",
            input_dims
        )));
    }

    let batch_size = input_dims[0];
    let num_classes = input_dims[1];

    // Apply log_softmax to get log probabilities
    let log_probs = input.log_softmax(-1)?;

    // Convert to probabilities
    let probs = log_probs.exp()?;

    // Get data for indexing
    let log_probs_data = log_probs.to_vec()?;
    let probs_data = probs.to_vec()?;
    let target_data = target.to_vec()?;

    // Compute focal loss for each sample
    let alpha_weight = alpha.unwrap_or(1.0);
    let mut losses = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let target_class = target_data[i] as usize;

        if target_class >= num_classes {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Target class {} out of range for {} classes",
                target_class, num_classes
            )));
        }

        // Get probability and log probability for the true class
        let idx = i * num_classes + target_class;
        let pt = probs_data[idx];
        let log_pt = log_probs_data[idx];

        // Focal loss: -alpha * (1 - pt)^gamma * log(pt)
        let focal_weight = alpha_weight * (1.0 - pt).powf(gamma);
        let sample_loss = -focal_weight * log_pt;

        losses.push(sample_loss);
    }

    // Apply reduction
    let result = match reduction {
        "mean" => {
            let mean_loss: f32 = losses.iter().sum::<f32>() / batch_size as f32;
            Tensor::from_data(vec![mean_loss], vec![1], input.device())?
        }
        "sum" => {
            let sum_loss: f32 = losses.iter().sum();
            Tensor::from_data(vec![sum_loss], vec![1], input.device())?
        }
        "none" => {
            // Return per-sample losses
            Tensor::from_data(losses, vec![batch_size], input.device())?
        }
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid reduction mode: '{}'. Expected 'mean', 'sum', or 'none'",
                reduction
            )))
        }
    };

    Ok(result)
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
pub fn triplet_margin_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f32,
    p: f32,
    reduction: &str,
) -> Result<Tensor> {
    let anchor_shape_obj = anchor.shape();
    let anchor_shape = anchor_shape_obj.dims();
    let batch_size = anchor_shape[0];
    let feature_dim: usize = anchor_shape[1..].iter().product();

    let anchor_data = anchor.to_vec()?;
    let positive_data = positive.to_vec()?;
    let negative_data = negative.to_vec()?;

    let mut losses = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let start_idx = i * feature_dim;
        let end_idx = start_idx + feature_dim;

        // Compute distance between anchor and positive: ||a - p||_p
        let mut dist_ap = 0.0f32;
        for j in start_idx..end_idx {
            let diff = anchor_data[j] - positive_data[j];
            dist_ap += diff.abs().powf(p);
        }
        dist_ap = dist_ap.powf(1.0 / p);

        // Compute distance between anchor and negative: ||a - n||_p
        let mut dist_an = 0.0f32;
        for j in start_idx..end_idx {
            let diff = anchor_data[j] - negative_data[j];
            dist_an += diff.abs().powf(p);
        }
        dist_an = dist_an.powf(1.0 / p);

        // Compute triplet loss: max(dist_ap - dist_an + margin, 0)
        let loss = (dist_ap - dist_an + margin).max(0.0);
        losses.push(loss);
    }

    // Apply reduction
    let final_loss = match reduction {
        "mean" => {
            let sum: f32 = losses.iter().sum();
            vec![sum / batch_size as f32]
        }
        "sum" => {
            let sum: f32 = losses.iter().sum();
            vec![sum]
        }
        "none" => losses,
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid reduction mode: {}. Expected 'mean', 'sum', or 'none'",
                reduction
            )))
        }
    };

    let output_shape = if reduction == "none" {
        vec![batch_size]
    } else {
        vec![1]
    };

    Tensor::from_vec(final_loss, &output_shape)
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
pub fn contrastive_loss(
    output1: &Tensor,
    output2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: &str,
) -> Result<Tensor> {
    let output1_shape_obj = output1.shape();
    let output1_shape = output1_shape_obj.dims();
    let batch_size = output1_shape[0];
    let feature_dim: usize = output1_shape[1..].iter().product();

    let output1_data = output1.to_vec()?;
    let output2_data = output2.to_vec()?;
    let target_data = target.to_vec()?;

    let mut losses = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let start_idx = i * feature_dim;
        let end_idx = start_idx + feature_dim;

        // Compute Euclidean distance: ||output1 - output2||_2
        let mut dist_squared = 0.0f32;
        for j in start_idx..end_idx {
            let diff = output1_data[j] - output2_data[j];
            dist_squared += diff * diff;
        }
        let dist = dist_squared.sqrt();

        // Get target label (1 for similar, 0 for dissimilar)
        let label = target_data[i];

        // Compute contrastive loss
        let loss = if label > 0.5 {
            // Similar pair: minimize distance
            dist_squared
        } else {
            // Dissimilar pair: push distance to at least margin
            let margin_diff = (margin - dist).max(0.0);
            margin_diff * margin_diff
        };

        losses.push(loss);
    }

    // Apply reduction
    let final_loss = match reduction {
        "mean" => {
            let sum: f32 = losses.iter().sum();
            vec![sum / batch_size as f32]
        }
        "sum" => {
            let sum: f32 = losses.iter().sum();
            vec![sum]
        }
        "none" => losses,
        _ => {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid reduction mode: {}. Expected 'mean', 'sum', or 'none'",
                reduction
            )))
        }
    };

    let output_shape = if reduction == "none" {
        vec![batch_size]
    } else {
        vec![1]
    };

    Tensor::from_vec(final_loss, &output_shape)
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

        let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, "mean")?;
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

        let loss = triplet_margin_loss(&anchor, &positive, &negative, 0.5, 2.0, "mean")?;
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

        let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, "none")?;
        assert_eq!(loss.shape().dims(), &[2]);

        let loss_mean = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, "mean")?;
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

        let loss = contrastive_loss(&output1, &output2, &target, 2.0, "mean")?;
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

        let loss = contrastive_loss(&output1, &output2, &target, 2.0, "mean")?;
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

        let loss = contrastive_loss(&output1, &output2, &target, 2.0, "mean")?;
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

        let loss = contrastive_loss(&output1, &output2, &target, 2.0, "none")?;
        assert_eq!(loss.shape().dims(), &[2]);

        let loss_mean = contrastive_loss(&output1, &output2, &target, 2.0, "mean")?;
        assert_eq!(loss_mean.shape().dims(), &[1]);

        Ok(())
    }

    #[test]
    fn test_reduction_modes() -> Result<()> {
        let anchor = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[2, 2])?;
        let positive = Tensor::from_vec(vec![0.1, 0.1, 1.1, 1.1], &[2, 2])?;
        let negative = Tensor::from_vec(vec![5.0, 5.0, 6.0, 6.0], &[2, 2])?;

        // Test "none" reduction
        let loss_none = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, "none")?;
        assert_eq!(
            loss_none.shape().dims(),
            &[2],
            "none reduction should return batch_size losses"
        );

        // Test "mean" reduction
        let loss_mean = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, "mean")?;
        assert_eq!(
            loss_mean.shape().dims(),
            &[1],
            "mean reduction should return scalar"
        );

        // Test "sum" reduction
        let loss_sum = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, "sum")?;
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
/// # Reference
/// Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation", 3DV 2016
pub fn dice_loss(input: &Tensor, target: &Tensor, smooth: f32, reduction: &str) -> Result<Tensor> {
    // Validate inputs
    if input.shape().dims() != target.shape().dims() {
        return Err(TorshError::ShapeMismatch {
            expected: target.shape().dims().to_vec(),
            got: input.shape().dims().to_vec(),
        });
    }

    // Flatten spatial dimensions while keeping batch and channel dimensions
    let input_shape_binding = input.shape();
    let input_shape = input_shape_binding.dims();
    let _batch_size = input_shape[0];
    let _num_elements: usize = input_shape.iter().product();

    // Compute intersection and union
    let intersection = input.mul_op(target)?;
    let intersection_sum = intersection.sum()?;
    let intersection_val = intersection_sum.to_vec()?[0];

    let input_sum = input.sum()?;
    let input_val = input_sum.to_vec()?[0];

    let target_sum = target.sum()?;
    let target_val = target_sum.to_vec()?[0];

    // Compute Dice coefficient
    let dice_coeff = (2.0 * intersection_val + smooth) / (input_val + target_val + smooth);
    let dice_loss_val = 1.0 - dice_coeff;

    let loss = Tensor::from_data(vec![dice_loss_val], vec![1], input.device())?;
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
/// # Reference
/// Salehi et al., "Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks", MICCAI 2017
pub fn tversky_loss(
    input: &Tensor,
    target: &Tensor,
    alpha: f32,
    beta: f32,
    smooth: f32,
    reduction: &str,
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
    let tp_val = tp.to_vec()?[0];

    let ones = torsh_tensor::creation::ones_like(target)?;
    let fp = input.mul_op(&ones.sub(target)?)?.sum()?;
    let fp_val = fp.to_vec()?[0];

    let fn_tensor = ones.sub(input)?.mul_op(target)?.sum()?;
    let fn_val = fn_tensor.to_vec()?[0];

    // Compute Tversky index
    let tversky_index = (tp_val + smooth) / (tp_val + alpha * fp_val + beta * fn_val + smooth);
    let tversky_loss_val = 1.0 - tversky_index;

    let loss = Tensor::from_data(vec![tversky_loss_val], vec![1], input.device())?;
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
/// # Reference
/// Feng et al., "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks", CVPR 2018
pub fn wing_loss(
    input: &Tensor,
    target: &Tensor,
    width: f32,
    curvature: f32,
    reduction: &str,
) -> Result<Tensor> {
    // Validate inputs
    if input.shape().dims() != target.shape().dims() {
        return Err(TorshError::ShapeMismatch {
            expected: target.shape().dims().to_vec(),
            got: input.shape().dims().to_vec(),
        });
    }

    // Compute C constant for continuity
    let c = width - width * (1.0 + width / curvature).ln();

    // Compute element-wise Wing loss
    let input_vec: Vec<f32> = input.to_vec()?;
    let target_vec: Vec<f32> = target.to_vec()?;

    let loss_vec: Vec<f32> = input_vec
        .iter()
        .zip(target_vec.iter())
        .map(|(&pred, &tgt)| {
            let abs_diff = (pred - tgt).abs();
            if abs_diff < width {
                width * (1.0 + abs_diff / curvature).ln()
            } else {
                abs_diff - c
            }
        })
        .collect();

    let loss = Tensor::from_data(loss_vec, input.shape().dims().to_vec(), input.device())?;

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
/// # Reference
/// Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition", ECCV 2016
pub fn center_loss(
    features: &Tensor,
    labels: &Tensor<i64>,
    centers: &Tensor,
    reduction: &str,
) -> Result<Tensor> {
    let features_shape_binding = features.shape();
    let features_shape = features_shape_binding.dims();
    let batch_size = features_shape[0];
    let feature_dim = features_shape[1];

    let centers_shape_binding = centers.shape();
    let centers_shape = centers_shape_binding.dims();
    let num_classes = centers_shape[0];

    // Validate dimensions
    if centers_shape[1] != feature_dim {
        return Err(TorshError::ShapeMismatch {
            expected: vec![num_classes, feature_dim],
            got: centers_shape.to_vec(),
        });
    }

    // Get feature and center vectors
    let features_vec: Vec<f32> = features.to_vec()?;
    let centers_vec: Vec<f32> = centers.to_vec()?;
    let labels_vec: Vec<i64> = labels.to_vec()?;

    // Compute squared distance from each sample to its class center
    let mut loss_vec = Vec::with_capacity(batch_size);

    for (i, &label) in labels_vec.iter().enumerate() {
        let label_idx = label as usize;
        if label_idx >= num_classes {
            return Err(TorshError::InvalidArgument(format!(
                "Label {} out of range for {} classes",
                label, num_classes
            )));
        }

        // Compute squared distance
        let mut squared_dist = 0.0_f32;
        for j in 0..feature_dim {
            let feature_val = features_vec[i * feature_dim + j];
            let center_val = centers_vec[label_idx * feature_dim + j];
            let diff = feature_val - center_val;
            squared_dist += diff * diff;
        }

        loss_vec.push(0.5 * squared_dist);
    }

    let loss = Tensor::from_data(loss_vec, vec![batch_size], features.device())?;
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
/// # Reference
/// Oord et al., "Representation Learning with Contrastive Predictive Coding", arXiv 2018
/// Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
pub fn infonce_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negatives: &Tensor,
    temperature: f32,
    reduction: &str,
) -> Result<Tensor> {
    let anchor_shape_binding = anchor.shape();
    let anchor_shape = anchor_shape_binding.dims();
    let batch_size = anchor_shape[0];
    let embedding_dim = anchor_shape[1];

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

    // Get vectors
    let anchor_vec: Vec<f32> = anchor.to_vec()?;
    let positive_vec: Vec<f32> = positive.to_vec()?;
    let negatives_vec: Vec<f32> = negatives.to_vec()?;

    // Compute InfoNCE loss for each sample in batch
    let mut loss_vec = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        // Get anchor and positive for this sample
        let anchor_start = i * embedding_dim;
        let anchor_end = anchor_start + embedding_dim;
        let anchor_emb = &anchor_vec[anchor_start..anchor_end];
        let positive_emb = &positive_vec[anchor_start..anchor_end];

        // Compute cosine similarity with positive
        let pos_sim = cosine_similarity(anchor_emb, positive_emb) / temperature;

        // Compute cosine similarities with all negatives
        let mut neg_sims = Vec::with_capacity(num_negatives);
        for j in 0..num_negatives {
            let neg_start = j * embedding_dim;
            let neg_end = neg_start + embedding_dim;
            let neg_emb = &negatives_vec[neg_start..neg_end];
            let neg_sim = cosine_similarity(anchor_emb, neg_emb) / temperature;
            neg_sims.push(neg_sim);
        }

        // Compute log-sum-exp for numerical stability
        let max_sim = pos_sim.max(neg_sims.iter().copied().fold(f32::NEG_INFINITY, f32::max));

        let exp_pos = (pos_sim - max_sim).exp();
        let sum_exp_neg: f32 = neg_sims.iter().map(|&s| (s - max_sim).exp()).sum();

        let loss = -(pos_sim - max_sim - (exp_pos + sum_exp_neg).ln());
        loss_vec.push(loss);
    }

    let loss = Tensor::from_data(loss_vec, vec![batch_size], anchor.device())?;
    apply_reduction(&loss, reduction, None, &[])
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
