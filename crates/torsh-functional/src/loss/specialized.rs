//! Specialized loss functions
//!
//! This module provides specialized loss functions for specific domains
//! such as sequence-to-sequence learning, time series, and other advanced applications.

use crate::loss::common::ReductionType;
use crate::utils::{function_context, validate_positive};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Connectionist Temporal Classification (CTC) Loss
///
/// Computes the CTC loss between a sequence of predictions and a target sequence.
/// This is commonly used in speech recognition and handwriting recognition where
/// the alignment between input and target sequences is unknown.
///
/// # Arguments
/// * `log_probs` - Tensor of log probabilities with shape (T, N, C) where:
///   - T is the input sequence length
///   - N is the batch size
///   - C is the number of classes (including blank)
/// * `targets` - Target sequences with shape (N, S) where S is target sequence length
/// * `input_lengths` - Lengths of input sequences for each batch element
/// * `target_lengths` - Lengths of target sequences for each batch element
/// * `blank` - Index of the blank label (typically 0)
/// * `reduction` - Reduction type
/// * `zero_infinity` - Whether to zero out infinite losses
///
/// # Returns
/// CTC loss tensor
pub fn ctc_loss(
    log_probs: &Tensor,
    targets: &Tensor,
    input_lengths: &Tensor,
    target_lengths: &Tensor,
    _blank: i64,
    reduction: ReductionType,
    zero_infinity: bool,
) -> TorshResult<Tensor> {
    let context = function_context("ctc_loss");

    // Validate input shapes
    if log_probs.ndim() != 3 {
        return Err(TorshError::config_error_with_context(
            "log_probs must be 3D tensor with shape (T, N, C)",
            &context,
        ));
    }

    if targets.ndim() != 2 {
        return Err(TorshError::config_error_with_context(
            "targets must be 2D tensor with shape (N, S)",
            &context,
        ));
    }

    let shape = log_probs.shape();
    let dims = shape.dims();
    let seq_len = dims[0];
    let batch_size = dims[1];
    let num_classes = dims[2];

    // For now, implement a simplified version that assumes:
    // - All sequences use full length
    // - No blank symbol handling (placeholder implementation)
    // A full CTC implementation would require dynamic programming with forward-backward algorithm

    // This is a placeholder implementation - real CTC loss requires:
    // 1. Forward-backward algorithm for probability computation
    // 2. Handling of blank symbols and repetitions
    // 3. Dynamic programming to find all valid alignments
    let mut total_loss = 0.0;

    for batch_idx in 0..batch_size {
        // Get input and target lengths for this batch
        let input_len = input_lengths.get(&[batch_idx])? as usize;
        let target_len = target_lengths.get(&[batch_idx])? as usize;

        if input_len == 0 || target_len == 0 {
            continue;
        }

        // Simple approximation: sum negative log probabilities of target sequence
        // This is NOT the actual CTC loss, just a placeholder
        let mut batch_loss = 0.0;
        for t in 0..input_len.min(seq_len).min(target_len) {
            let target_class = targets.get(&[batch_idx, t])? as usize;
            if target_class < num_classes {
                let log_prob = log_probs.get(&[t, batch_idx, target_class])?;
                batch_loss -= log_prob;
            }
        }

        if zero_infinity && batch_loss.is_infinite() {
            batch_loss = 0.0;
        }

        total_loss += batch_loss;
    }

    let loss_tensor = Tensor::from_vec(vec![total_loss], &[1])?;

    match reduction {
        ReductionType::None => Ok(loss_tensor),
        ReductionType::Mean => Ok(Tensor::from_vec(
            vec![total_loss / batch_size as f32],
            &[1],
        )?),
        ReductionType::Sum => Ok(loss_tensor),
    }
}

/// Sequence-to-Sequence Loss with attention
///
/// Computes loss for sequence-to-sequence models with attention mechanisms.
/// This combines cross-entropy loss with attention regularization.
///
/// # Arguments
/// * `predictions` - Predicted logits with shape (N, T_out, V) where V is vocab size
/// * `targets` - Target sequence with shape (N, T_out)
/// * `attention_weights` - Attention weights with shape (N, T_out, T_in)
/// * `attention_reg` - Regularization weight for attention
/// * `reduction` - Reduction type
pub fn seq2seq_loss_with_attention(
    predictions: &Tensor,
    targets: &Tensor,
    attention_weights: Option<&Tensor>,
    attention_reg: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    // Cross-entropy loss on predictions
    let ce_loss = compute_sequence_cross_entropy(predictions, targets)?;

    let total_loss = if let Some(attn_weights) = attention_weights {
        // Add attention regularization
        let attn_reg_loss = attention_regularization_loss(attn_weights)?;
        ce_loss.add(&attn_reg_loss.mul_scalar(attention_reg)?)?
    } else {
        ce_loss
    };

    reduction.apply(total_loss)
}

/// Temporal Consistency Loss
///
/// Encourages temporal smoothness in sequence predictions.
/// Useful for video analysis, time series prediction, etc.
///
/// # Arguments
/// * `predictions` - Sequence predictions with shape (N, T, *)
/// * `smoothness_weight` - Weight for smoothness regularization
/// * `reduction` - Reduction type
pub fn temporal_consistency_loss(
    predictions: &Tensor,
    smoothness_weight: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    if predictions.ndim() < 3 {
        return Err(TorshError::InvalidArgument(
            "predictions must be at least 3D (N, T, ...)".to_string(),
        ));
    }

    let shape = predictions.shape();
    let dims = shape.dims();
    let seq_len = dims[1];

    if seq_len < 2 {
        return Err(TorshError::InvalidArgument(
            "Sequence length must be at least 2".to_string(),
        ));
    }

    // Compute differences between consecutive time steps
    let pred_t = predictions.slice(1, 0, seq_len - 1)?; // t=0 to T-2
    let pred_t_plus_1 = predictions.slice(1, 1, seq_len)?; // t=1 to T-1

    // L2 difference between consecutive predictions
    let pred_t_tensor = pred_t.to_tensor()?;
    let pred_t_plus_1_tensor = pred_t_plus_1.to_tensor()?;
    let diff = pred_t_tensor.sub(&pred_t_plus_1_tensor)?;
    let smoothness_loss = diff.pow_scalar(2.0)?.mean(None, false)?;

    let total_loss = smoothness_loss.mul_scalar(smoothness_weight)?;
    reduction.apply(total_loss)
}

/// Wasserstein Loss
///
/// Approximates the Wasserstein distance between distributions.
/// Used in Wasserstein GANs and optimal transport problems.
///
/// # Arguments
/// * `real_scores` - Discriminator scores for real samples
/// * `fake_scores` - Discriminator scores for fake samples
/// * `reduction` - Reduction type
pub fn wasserstein_loss(
    real_scores: &Tensor,
    fake_scores: &Tensor,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    // Wasserstein loss: E[D(real)] - E[D(fake)]
    // We want to maximize this for the discriminator, minimize for generator
    let real_mean = real_scores.mean(None, false)?;
    let fake_mean = fake_scores.mean(None, false)?;
    let wasserstein_distance = real_mean.sub(&fake_mean)?;

    // Return negative for minimization
    let loss = wasserstein_distance.neg()?;
    reduction.apply(loss)
}

/// Gradient Penalty Loss
///
/// Implements gradient penalty for improved training of GANs (WGAN-GP).
///
/// # Arguments
/// * `gradients` - Gradients of discriminator w.r.t. interpolated samples
/// * `penalty_weight` - Weight for gradient penalty
/// * `reduction` - Reduction type
pub fn gradient_penalty_loss(
    gradients: &Tensor,
    penalty_weight: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_positive(penalty_weight, "penalty_weight", "gradient_penalty_loss")?;

    // Compute L2 norm of gradients
    let grad_norm = gradients.norm()?;

    // Gradient penalty: (||grad|| - 1)^2
    let penalty = grad_norm.sub_scalar(1.0)?.pow_scalar(2.0)?;
    let weighted_penalty = penalty.mul_scalar(penalty_weight)?;

    reduction.apply(weighted_penalty)
}

// Helper functions

fn compute_sequence_cross_entropy(predictions: &Tensor, targets: &Tensor) -> TorshResult<Tensor> {
    // Apply log_softmax to predictions
    let dim = (predictions.shape().ndim() - 1) as i32;
    let log_probs = predictions.log_softmax(dim)?;

    // Compute negative log likelihood
    let shape = targets.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let seq_len = dims[1];

    let mut total_loss = 0.0;
    for i in 0..batch_size {
        for j in 0..seq_len {
            let target_class = targets.get(&[i, j])? as usize;
            let log_prob = log_probs.get(&[i, j, target_class])?;
            total_loss -= log_prob;
        }
    }

    let loss_value = total_loss / (batch_size * seq_len) as f32;
    Tensor::from_vec(vec![loss_value], &[1])
}

fn attention_regularization_loss(attention_weights: &Tensor) -> TorshResult<Tensor> {
    // Encourage attention to be sparse and focused
    // L2 regularization on attention weights
    attention_weights.pow_scalar(2.0)?.mean(None, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::{from_vec, zeros};

    #[test]
    fn test_ctc_loss_basic() -> TorshResult<()> {
        // Simple CTC loss test with placeholder implementation
        let log_probs = zeros(&[10, 2, 5])?; // T=10, N=2, C=5
        let targets = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], DeviceType::Cpu)?;
        let input_lengths = from_vec(vec![10.0, 10.0], &[2], DeviceType::Cpu)?;
        let target_lengths = from_vec(vec![2.0, 2.0], &[2], DeviceType::Cpu)?;

        let loss = ctc_loss(
            &log_probs,
            &targets,
            &input_lengths,
            &target_lengths,
            0,
            ReductionType::Mean,
            false,
        )?;
        let loss_value = loss.item()?;

        // Loss should be non-negative
        assert!(loss_value >= 0.0);
        Ok(())
    }

    #[test]
    fn test_temporal_consistency_loss_basic() -> TorshResult<()> {
        // Create predictions with some temporal variation
        let predictions = from_vec(
            vec![1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2], // 3 time steps, 3 features
            &[1, 3, 3],
            DeviceType::Cpu,
        )?;

        let loss = temporal_consistency_loss(&predictions, 1.0, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Should be small for smooth predictions
        assert!(loss_value >= 0.0 && loss_value < 1.0);
        Ok(())
    }

    #[test]
    fn test_wasserstein_loss_basic() -> TorshResult<()> {
        let real_scores = from_vec(vec![1.0, 2.0, 1.5], &[3], DeviceType::Cpu)?;
        let fake_scores = from_vec(vec![0.5, 0.8, 0.6], &[3], DeviceType::Cpu)?;

        let loss = wasserstein_loss(&real_scores, &fake_scores, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Wasserstein loss can be positive or negative
        assert!(loss_value.is_finite());
        Ok(())
    }

    #[test]
    fn test_gradient_penalty_loss_basic() -> TorshResult<()> {
        let gradients = from_vec(vec![1.5, 0.8, 1.2], &[3], DeviceType::Cpu)?;

        let loss = gradient_penalty_loss(&gradients, 10.0, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Gradient penalty should be non-negative
        assert!(loss_value >= 0.0);
        Ok(())
    }

    #[test]
    fn test_seq2seq_loss_basic() -> TorshResult<()> {
        let predictions = zeros(&[2, 5, 10])?; // N=2, T=5, V=10
        let targets = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            &[2, 5],
            DeviceType::Cpu,
        )?;

        let loss =
            seq2seq_loss_with_attention(&predictions, &targets, None, 0.0, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Loss should be positive for non-perfect predictions
        assert!(loss_value > 0.0);
        Ok(())
    }
}
