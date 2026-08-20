//! Common types and utilities for loss functions
//!
//! This module provides shared types and helper functions used across
//! different loss function categories.

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

pub use torsh_core::reduction::Reduction as ReductionType;

/// Extension trait for applying reduction to loss tensors.
/// Cannot be an inherent impl because `Reduction` is defined in `torsh-core`.
pub trait ReductionExt {
    /// Reduce a per-element loss tensor using the configured mode.
    fn apply(&self, loss: &Tensor, batch_size: Option<usize>) -> TorshResult<Tensor>;
}

/// Extension methods for applying reduction to loss tensors.
/// Lives here (not in torsh-core) because it depends on Tensor.
impl ReductionExt for ReductionType {

    /// Reduce a per-element loss to the configured shape.
    ///
    /// - `None`: returns the loss unchanged.
    /// - `Mean`: divides by `numel` (total element count), matching PyTorch `"mean"`.
    /// - `Sum`: sums all elements.
    /// - `BatchMean`: divides by `batch_size` (number of samples).
    ///   Equivalent to PyTorch `"batchmean"`.
    ///
    /// Both reduced arms use recording ops (`sum()`, `div_scalar`) so the result
    /// stays on the autograd graph. `view(&[1])` restores the historical `[1]`
    /// output shape from the rank-0 tensor `sum()` returns.
    fn apply(&self, loss: &Tensor, batch_size: Option<usize>) -> TorshResult<Tensor> {
        match self {
            Self::None => Ok(loss.clone()),
            Self::Mean => loss.mean(None, false),
            Self::Sum => loss.sum(),
            Self::BatchMean => {
                let bs = batch_size.unwrap_or_else(|| {
                    loss.shape().dims().first().copied().unwrap_or(1)
                });
                loss.sum()?.div_scalar(bs as f32)
            }
        }
    }
}


/// Floor applied to a sum of squares (or of `|d|^p`) before the root that turns
/// it into a distance.
///
/// `sqrt` and `x^(1/p)` have an infinite derivative at zero, so a pair of
/// identical embeddings — an ordinary, indeed *desirable*, input for a
/// metric-learning loss — would back-propagate `inf`. With the constant-mask
/// branch composition of [`blend`] that is worse than it sounds: the branch the
/// mask discards is still *evaluated*, so `0 * inf = NaN` then poisons every
/// sample in the batch, not just the degenerate one.
///
/// Flooring the radicand at `1e-12` moves the distance by at most `1e-6` (four
/// orders below the `1e-5` tolerance the forward-value pins use) and hands the
/// kink a finite subgradient instead. It is applied through
/// [`torsh_tensor::Tensor::clamp_min`], which records `Operation::ClampBounds`
/// and whose backward is zero wherever the clamp bit.
pub(crate) const DISTANCE_FLOOR: f32 = 1e-12;

/// Build the detached 0/1 indicator of `predicate` over `source`'s elements,
/// together with its complement.
///
/// This is the branch-selection primitive that replaces
/// [`torsh_tensor::Tensor::where_tensor`]. `where_tensor` rebuilds its result
/// from raw data and records nothing, so every loss that selected a branch with
/// it returned an honestly detached leaf and `backward()` never reached the
/// inputs. Both returned tensors are plain constants — they are built with
/// [`torsh_tensor::Tensor::from_data`] and never carry `requires_grad` — which
/// is exactly right: the branch indicator is piecewise constant, so it has zero
/// derivative wherever it is defined.
///
/// See [`blend`] for how the two masks are recombined.
pub(crate) fn branch_masks<F>(source: &Tensor, predicate: F) -> TorshResult<(Tensor, Tensor)>
where
    F: Fn(f32) -> bool,
{
    let values = source.data()?;
    let selected: Vec<f32> = values
        .iter()
        .map(|&value| if predicate(value) { 1.0 } else { 0.0 })
        .collect();
    let complement: Vec<f32> = selected.iter().map(|&value| 1.0 - value).collect();

    let shape_binding = source.shape();
    let dims = shape_binding.dims().to_vec();
    let on_true = Tensor::from_data(selected, dims.clone(), source.device())?;
    let on_false = Tensor::from_data(complement, dims, source.device())?;
    Ok((on_true, on_false))
}

/// `mask_true * on_true + mask_false * on_false` — the recording replacement for
/// `on_true.where_tensor(&mask, &on_false)`.
///
/// Both branches stay on the autograd graph, and because the masks are
/// complementary constants produced by [`branch_masks`], backward routes each
/// element's gradient to exactly the branch that produced its value and gives
/// the other branch zero. The forward result is element-for-element identical to
/// `where_tensor`'s, provided neither branch holds a non-finite value at a
/// masked-out position — see [`DISTANCE_FLOOR`] for the one place in this crate
/// where that had to be arranged deliberately.
pub(crate) fn blend(
    on_true: &Tensor,
    on_false: &Tensor,
    mask_true: &Tensor,
    mask_false: &Tensor,
) -> TorshResult<Tensor> {
    on_true.mul(mask_true)?.add(&on_false.mul(mask_false)?)
}
