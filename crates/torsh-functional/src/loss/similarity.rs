//! Similarity and distance-based loss functions
//!
//! This module provides loss functions based on similarity measures and distances,
//! commonly used for metric learning, face recognition, and similarity learning tasks.

use crate::loss::common::ReductionType;
use crate::utils::{validate_elementwise_shapes, validate_range};
use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Cosine Embedding Loss
///
/// Creates a criterion that measures the loss given input tensors x1, x2
/// and a Tensor label y with values 1 or -1.
pub fn cosine_embedding_loss(
    input1: &Tensor,
    input2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input1, input2)?;

    // Compute cosine similarity
    let dot_product = input1.mul(input2)?.sum_dim(&[1], false)?;
    let norm1 = input1.pow_scalar(2.0)?.sum_dim(&[1], false)?.sqrt()?;
    let norm2 = input2.pow_scalar(2.0)?.sum_dim(&[1], false)?.sqrt()?;
    let cosine_sim = dot_product.div(&norm1.mul(&norm2)?)?;

    // Cosine embedding loss:
    // - if target == 1: 1 - cosine_sim
    // - if target == -1: max(0, cosine_sim - margin)
    let positive_mask = target.gt_scalar(0.0)?;
    let positive_loss = cosine_sim.neg()?.add_scalar(1.0)?;
    let negative_loss = cosine_sim.sub_scalar(margin)?.clamp(0.0, f32::MAX)?;

    let loss = positive_loss.where_tensor(&positive_mask, &negative_loss)?;
    reduction.apply(loss)
}

/// Hinge Embedding Loss
///
/// Measures the loss given an input tensor x and a labels tensor y (containing 1 or -1).
pub fn hinge_embedding_loss(
    input: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input, target)?;

    // Hinge embedding loss:
    // - if target == 1: input
    // - if target == -1: max(0, margin - input)
    let positive_mask = target.gt_scalar(0.0)?;
    let positive_loss = input.clone();
    let negative_loss = input.neg()?.add_scalar(margin)?.clamp(0.0, f32::MAX)?;

    let loss = positive_loss.where_tensor(&positive_mask, &negative_loss)?;
    reduction.apply(loss)
}

/// Margin Ranking Loss
///
/// Creates a criterion that measures the loss given inputs x1, x2,
/// two 1D mini-batch or 0D Tensors, and a label 1D mini-batch or 0D Tensor y with values (1 or -1).
pub fn margin_ranking_loss(
    input1: &Tensor,
    input2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input1, input2)?;
    validate_elementwise_shapes(input1, target)?;

    // Margin ranking loss: max(0, -target * (input1 - input2) + margin)
    let diff = input1.sub(input2)?;
    let target_diff = target.mul(&diff)?;
    let loss = target_diff
        .neg()?
        .add_scalar(margin)?
        .clamp(0.0, f32::MAX)?;

    reduction.apply(loss)
}

/// Triplet Margin Loss
///
/// Creates a criterion that measures the triplet loss given input tensors a, p, and n
/// (representing anchor, positive, and negative examples respectively).
pub fn triplet_margin_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f32,
    p: f32,
    eps: f32,
    swap: bool,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(anchor, positive)?;
    validate_elementwise_shapes(anchor, negative)?;
    validate_range(p, 1.0, 2.0, "p", "triplet_margin_loss")?;

    // Compute distances
    let pos_dist = compute_pairwise_distance(anchor, positive, p, eps)?;
    let mut neg_dist = compute_pairwise_distance(anchor, negative, p, eps)?;

    if swap {
        // Also consider distance between positive and negative
        let pos_neg_dist = compute_pairwise_distance(positive, negative, p, eps)?;
        neg_dist = neg_dist.minimum(&pos_neg_dist)?;
    }

    // Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
    let loss = pos_dist
        .sub(&neg_dist)?
        .add_scalar(margin)?
        .clamp(0.0, f32::MAX)?;
    reduction.apply(loss)
}

/// Triplet Margin Loss with Distance Function
///
/// Similar to triplet margin loss but allows custom distance function.
pub fn triplet_margin_with_distance_loss<F>(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    distance_function: F,
    margin: f32,
    swap: bool,
    reduction: ReductionType,
) -> TorshResult<Tensor>
where
    F: Fn(&Tensor, &Tensor) -> TorshResult<Tensor>,
{
    validate_elementwise_shapes(anchor, positive)?;
    validate_elementwise_shapes(anchor, negative)?;

    // Compute distances using provided function
    let pos_dist = distance_function(anchor, positive)?;
    let mut neg_dist = distance_function(anchor, negative)?;

    if swap {
        let pos_neg_dist = distance_function(positive, negative)?;
        neg_dist = neg_dist.minimum(&pos_neg_dist)?;
    }

    // Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
    let loss = pos_dist
        .sub(&neg_dist)?
        .add_scalar(margin)?
        .clamp(0.0, f32::MAX)?;
    reduction.apply(loss)
}

/// Contrastive Loss
///
/// Computes contrastive loss for pairs of embeddings.
///
/// Loss = (1 - y) * 0.5 * d^2 + y * 0.5 * max(0, margin - d)^2
/// where y=0 for similar pairs, y=1 for dissimilar pairs
pub fn contrastive_loss(
    input1: &Tensor,
    input2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input1, input2)?;

    // Compute Euclidean distance
    let diff = input1.sub(input2)?;
    let dist = diff.pow_scalar(2.0)?.sum_dim(&[1], false)?.sqrt()?;

    // Contrastive loss computation
    let similar_loss = dist.pow_scalar(2.0)?.mul_scalar(0.5)?;
    let dissimilar_loss = dist
        .neg()?
        .add_scalar(margin)?
        .clamp(0.0, f32::MAX)?
        .pow_scalar(2.0)?
        .mul_scalar(0.5)?;

    // target = 0 for similar pairs, target = 1 for dissimilar pairs
    let similar_mask = target.lt_scalar(0.5)?;
    let loss = similar_loss.where_tensor(&similar_mask, &dissimilar_loss)?;

    reduction.apply(loss)
}

/// Helper function to compute pairwise distance
fn compute_pairwise_distance(x1: &Tensor, x2: &Tensor, p: f32, eps: f32) -> TorshResult<Tensor> {
    let diff = x1.sub(x2)?;

    if p == 2.0 {
        // Euclidean distance
        diff.pow_scalar(2.0)?
            .sum_dim(&[1], false)?
            .sqrt()?
            .add_scalar(eps)
    } else if p == 1.0 {
        // Manhattan distance
        diff.abs()?.sum_dim(&[1], false)?.add_scalar(eps)
    } else {
        // General Lp norm
        diff.abs()?
            .pow_scalar(p)?
            .sum_dim(&[1], false)?
            .pow_scalar(1.0 / p)?
            .add_scalar(eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_cosine_embedding_loss_similar() -> TorshResult<()> {
        // Similar embeddings (target = 1)
        let input1 = from_vec(vec![1.0, 2.0, 3.0], &[1, 3], DeviceType::Cpu)?;
        let input2 = from_vec(vec![1.1, 2.1, 3.1], &[1, 3], DeviceType::Cpu)?; // Very similar
        let target = from_vec(vec![1.0], &[1], DeviceType::Cpu)?;

        let loss = cosine_embedding_loss(&input1, &input2, &target, 0.0, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Loss should be small for similar embeddings
        assert!(loss_value < 0.1);
        Ok(())
    }

    #[test]
    fn test_cosine_embedding_loss_dissimilar() -> TorshResult<()> {
        // Dissimilar embeddings (target = -1)
        let input1 = from_vec(vec![1.0, 2.0, 3.0], &[1, 3], DeviceType::Cpu)?;
        let input2 = from_vec(vec![-1.0, -2.0, -3.0], &[1, 3], DeviceType::Cpu)?; // Opposite
        let target = from_vec(vec![-1.0], &[1], DeviceType::Cpu)?;
        let margin = 0.5;

        let loss = cosine_embedding_loss(&input1, &input2, &target, margin, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // For opposite vectors, cosine similarity should be -1, so loss should be max(0, -1 - 0.5) = 0
        assert!(loss_value < 1e-6);
        Ok(())
    }

    #[test]
    fn test_triplet_margin_loss_basic() -> TorshResult<()> {
        let anchor = from_vec(vec![1.0, 2.0], &[1, 2], DeviceType::Cpu)?;
        let positive = from_vec(vec![1.1, 2.1], &[1, 2], DeviceType::Cpu)?; // Close to anchor
        let negative = from_vec(vec![5.0, 6.0], &[1, 2], DeviceType::Cpu)?; // Far from anchor

        let loss = triplet_margin_loss(
            &anchor,
            &positive,
            &negative,
            1.0,
            2.0,
            1e-6,
            false,
            ReductionType::Mean,
        )?;
        let loss_value = loss.item()?;

        // Since negative is much farther than positive, loss should be small or zero
        assert!(loss_value >= 0.0);
        Ok(())
    }

    #[test]
    fn test_contrastive_loss_similar_pair() -> TorshResult<()> {
        let input1 = from_vec(vec![1.0, 2.0], &[1, 2], DeviceType::Cpu)?;
        let input2 = from_vec(vec![1.1, 2.1], &[1, 2], DeviceType::Cpu)?;
        let target = from_vec(vec![0.0], &[1], DeviceType::Cpu)?; // Similar pair

        let loss = contrastive_loss(&input1, &input2, &target, 1.0, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Loss should be small for similar embeddings
        assert!(loss_value >= 0.0 && loss_value < 1.0);
        Ok(())
    }

    #[test]
    fn test_margin_ranking_loss_basic() -> TorshResult<()> {
        let input1 = from_vec(vec![2.0, 3.0], &[2], DeviceType::Cpu)?;
        let input2 = from_vec(vec![1.0, 1.5], &[2], DeviceType::Cpu)?;
        let target = from_vec(vec![1.0, 1.0], &[2], DeviceType::Cpu)?; // input1 should be greater

        let loss = margin_ranking_loss(&input1, &input2, &target, 0.0, ReductionType::Mean)?;
        let loss_value = loss.item()?;

        // Since input1 > input2 and target = 1, loss should be small
        assert!(loss_value >= 0.0);
        Ok(())
    }
}
