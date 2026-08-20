//! Similarity and distance-based loss functions
//!
//! This module provides loss functions based on similarity measures and distances,
//! commonly used for metric learning, face recognition, and similarity learning tasks.

use crate::loss::common::{blend, branch_masks, ReductionType, ReductionExt, DISTANCE_FLOOR};
use crate::utils::{validate_elementwise_shapes, validate_range};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Check that `target` carries exactly one label per reduced sample.
///
/// Until wave 6 this check was performed — accidentally — by
/// [`torsh_tensor::Tensor::where_tensor`], which compares the shapes of all
/// three of its operands before selecting. Replacing `where_tensor` with the
/// recording [`blend`] composition removes that comparison, so the losses that
/// relied on it have to make it explicitly. The [`TorshError::ShapeMismatch`]
/// payload is byte-for-byte the one `where_tensor` used to produce, so callers
/// matching on the old diagnostic keep working.
fn validate_label_shape(per_sample: &Tensor, target: &Tensor) -> TorshResult<()> {
    let per_sample_binding = per_sample.shape();
    let expected = per_sample_binding.dims();
    let target_binding = target.shape();
    let got = target_binding.dims();

    if expected != got {
        return Err(TorshError::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        });
    }
    Ok(())
}

/// Cosine Embedding Loss
///
/// Creates a criterion that measures the loss given input tensors x1, x2
/// and a Tensor label y with values 1 or -1.
///
/// ```text
/// loss_i = 1 - cos(x1_i, x2_i)              if y_i > 0
///          max(0, cos(x1_i, x2_i) - margin) otherwise
/// ```
///
/// The branch is selected by a constant 0/1 mask (`branch_masks` / `blend` in
/// `loss::common`) rather than by `Tensor::where_tensor`, which records nothing
/// and used to leave this loss detached from `x1` and `x2` entirely.
pub fn cosine_embedding_loss(
    input1: &Tensor,
    input2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input1, input2)?;

    // Compute cosine similarity. Each sum of squares is floored before its
    // `sqrt` so that a zero embedding yields a finite (rather than infinite)
    // derivative; see `DISTANCE_FLOOR`. For any non-degenerate row the floor is
    // far below the sum and the value is untouched.
    let dot_product = input1.mul(input2)?.sum_dim(&[1], false)?;
    let norm1 = input1
        .pow_scalar(2.0)?
        .sum_dim(&[1], false)?
        .clamp_min(DISTANCE_FLOOR)?
        .sqrt()?;
    let norm2 = input2
        .pow_scalar(2.0)?
        .sum_dim(&[1], false)?
        .clamp_min(DISTANCE_FLOOR)?
        .sqrt()?;
    let cosine_sim = dot_product.div(&norm1.mul(&norm2)?)?;

    validate_label_shape(&cosine_sim, target)?;

    // `y > 0` is the predecessor's own predicate (`target.gt_scalar(0.0)`), and
    // is identical to `y == 1` over the documented `y ∈ {1, -1}` domain.
    let (positive_mask, negative_mask) = branch_masks(target, |label| label > 0.0)?;
    let positive_loss = cosine_sim.neg()?.add_scalar(1.0)?;
    let negative_loss = cosine_sim.sub_scalar(margin)?.clamp_min(0.0)?;

    let loss = blend(
        &positive_loss,
        &negative_loss,
        &positive_mask,
        &negative_mask,
    )?;
    reduction.apply(&loss, None)
}

/// Hinge Embedding Loss
///
/// Measures the loss given an input tensor x and a labels tensor y (containing 1 or -1).
///
/// ```text
/// loss_i = x_i                     if y_i > 0
///          max(0, margin - x_i)    otherwise
/// ```
///
/// Note that the `y > 0` branch is the **unclamped** input, so a negative entry
/// (and therefore a negative reduced loss) is correct and matches
/// `torch.nn.functional.hinge_embedding_loss`.
pub fn hinge_embedding_loss(
    input: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    // Element-wise: `target` already has to match `input` exactly, which is the
    // check `where_tensor` used to duplicate.
    validate_elementwise_shapes(input, target)?;

    let (positive_mask, negative_mask) = branch_masks(target, |label| label > 0.0)?;
    let positive_loss = input.clone();
    let negative_loss = input.neg()?.add_scalar(margin)?.clamp_min(0.0)?;

    let loss = blend(
        &positive_loss,
        &negative_loss,
        &positive_mask,
        &negative_mask,
    )?;
    reduction.apply(&loss, None)
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
    //
    // Every operation here already records (`clamp` gained `Operation::ClampBounds`
    // in wave 4), so this loss was measured to be attached before wave 6 and is
    // left structurally alone. `clamp(0.0, f32::MAX)` is spelled `clamp_min(0.0)`
    // only because that is the honest name for `max(0, ·)`.
    let diff = input1.sub(input2)?;
    let target_diff = target.mul(&diff)?;
    let loss = target_diff.neg()?.add_scalar(margin)?.clamp_min(0.0)?;

    reduction.apply(&loss, None)
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
        .clamp_min(0.0)?;
    reduction.apply(&loss, None)
}

/// Triplet Margin Loss with Distance Function
///
/// Similar to triplet margin loss but allows custom distance function.
///
/// # Gradient at zero distance
///
/// [`triplet_margin_loss`] floors its radicand with
/// [`DISTANCE_FLOOR`](crate::loss::common) so that a coincident anchor/positive
/// pair does not back-propagate `NaN`. That guard cannot reach inside
/// `distance_function`: if the supplied closure ends in a bare `sqrt` (or any
/// other root) it will still produce an infinite derivative when the two
/// operands coincide. Callers that need the guard should apply
/// `Tensor::clamp_min` to their own radicand.
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
        .clamp_min(0.0)?;
    reduction.apply(&loss, None)
}

/// Contrastive Loss
///
/// Computes contrastive loss for pairs of embeddings.
///
/// Loss = (1 - y) * 0.5 * d^2 + y * 0.5 * max(0, margin - d)^2
/// where y=0 for similar pairs, y=1 for dissimilar pairs
///
/// The two branches are combined with a constant 0/1 mask (`branch_masks` /
/// `blend` in `loss::common`) instead of `Tensor::where_tensor`, which records
/// nothing and used to leave this loss detached from both embeddings.
pub fn contrastive_loss(
    input1: &Tensor,
    input2: &Tensor,
    target: &Tensor,
    margin: f32,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input1, input2)?;

    // Squared Euclidean distance per pair. The similar branch is written
    // directly on `dist_squared` rather than on `sqrt(dist_squared)^2`: the two
    // are mathematically identical, but the former never routes the attraction
    // term through a square root and so keeps full precision near d = 0.
    let diff = input1.sub(input2)?;
    let dist_squared = diff.pow_scalar(2.0)?.sum_dim(&[1], false)?;

    validate_label_shape(&dist_squared, target)?;

    // The repulsion term genuinely needs the distance, so its radicand is
    // floored; without that, a pair of identical embeddings gives `sqrt'(0) =
    // inf` and — because `blend` evaluates *both* branches — `0 * inf = NaN`
    // would poison every other sample in the batch too.
    let dist = dist_squared.clamp_min(DISTANCE_FLOOR)?.sqrt()?;

    let similar_loss = dist_squared.mul_scalar(0.5)?;
    let dissimilar_loss = dist
        .neg()?
        .add_scalar(margin)?
        .clamp_min(0.0)?
        .pow_scalar(2.0)?
        .mul_scalar(0.5)?;

    // target = 0 for similar pairs, target = 1 for dissimilar pairs. `y < 0.5`
    // is the predecessor's own predicate (`target.lt_scalar(0.5)`).
    let (similar_mask, dissimilar_mask) = branch_masks(target, |label| label < 0.5)?;
    let loss = blend(
        &similar_loss,
        &dissimilar_loss,
        &similar_mask,
        &dissimilar_mask,
    )?;

    reduction.apply(&loss, None)
}

/// Per-sample `p`-norm distance `||x1 - x2||_p`, reduced over the feature axis.
///
/// Every step records, so the distance — and therefore any loss built on it —
/// stays on the autograd graph. For the two branches that finish with a root
/// (`p == 2` and the general `L_p`) the radicand is floored by
/// [`DISTANCE_FLOOR`](crate::loss::common) first, because `sqrt` and
/// `x^(1/p)` have an infinite derivative at zero and coincident operands are an
/// ordinary input here. The `p == 1` branch needs no floor: it ends in
/// `Tensor::abs`, whose recorded subgradient at zero is `0`.
///
/// `eps` is added *after* the root, which is where the predecessor put it; note
/// that `torch.nn.functional.pairwise_distance` instead adds its `eps` to the
/// radicand.
fn compute_pairwise_distance(x1: &Tensor, x2: &Tensor, p: f32, eps: f32) -> TorshResult<Tensor> {
    let diff = x1.sub(x2)?;

    if p == 2.0 {
        // Euclidean distance
        diff.pow_scalar(2.0)?
            .sum_dim(&[1], false)?
            .clamp_min(DISTANCE_FLOOR)?
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
            .clamp_min(DISTANCE_FLOOR)?
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
