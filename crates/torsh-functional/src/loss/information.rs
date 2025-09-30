//! Information theory loss functions
//!
//! This module provides loss functions based on information theory concepts,
//! such as KL divergence, mutual information, and entropy-based losses.

use crate::loss::common::ReductionType;
use crate::utils::{function_context, validate_elementwise_shapes};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Kullback-Leibler divergence loss
///
/// Computes the KL divergence loss between input and target distributions.
///
/// # Arguments
/// * `input` - Tensor containing log-probabilities (log-space)
/// * `target` - Tensor containing probabilities (linear space)
/// * `reduction` - Specifies the reduction to apply to the output
/// * `log_target` - If true, target is in log space
///
/// # Returns
/// KL divergence loss tensor
///
/// # Mathematical Definition
/// For discrete distributions P and Q:
/// ```text
/// KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
/// ```
/// If input is in log space and target is in linear space:
/// ```text
/// KL(target||input) = Σ target * (log(target) - input)
/// ```
pub fn kl_div(
    input: &Tensor,
    target: &Tensor,
    reduction: ReductionType,
    log_target: bool,
) -> TorshResult<Tensor> {
    let context = function_context("kl_div");
    validate_elementwise_shapes(input, target)?;

    // Ensure input is in log space (input should be log probabilities)
    // KL divergence: target * (log(target) - input)
    let kl = if log_target {
        // Both are in log space: target * (target - input)
        target.mul(&target.sub(input)?)?
    } else {
        // target is in linear space, input is in log space
        // Need to handle target = 0 case to avoid log(0)
        let eps = 1e-8_f32;
        let target_safe = target.clamp(eps, f32::MAX)?;
        let log_target = target_safe.log()?;
        let log_ratio = log_target.sub(input)?;
        target.mul(&log_ratio)?
    };

    reduction.apply(kl)
}

/// Jensen-Shannon divergence loss
///
/// Computes the symmetric Jensen-Shannon divergence between two probability distributions.
/// JS divergence is always non-negative and bounded between 0 and log(2).
///
/// # Arguments
/// * `input` - First probability distribution (in linear space)
/// * `target` - Second probability distribution (in linear space)
/// * `reduction` - Specifies the reduction to apply to the output
///
/// # Mathematical Definition
/// ```text
/// JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
/// where M = 0.5 * (P + Q)
/// ```
pub fn js_divergence(
    input: &Tensor,
    target: &Tensor,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input, target)?;

    // Compute mixture distribution M = 0.5 * (P + Q)
    let mixture = input.add(target)?.mul_scalar(0.5)?;

    // Avoid log(0) by adding small epsilon
    let eps = 1e-8_f32;
    let input_safe = input.clamp(eps, f32::MAX)?;
    let target_safe = target.clamp(eps, f32::MAX)?;
    let mixture_safe = mixture.clamp(eps, f32::MAX)?;

    // Compute log probabilities
    let log_input = input_safe.log()?;
    let log_target = target_safe.log()?;
    let log_mixture = mixture_safe.log()?;

    // KL(P||M) = Σ P * log(P/M) = Σ P * (log(P) - log(M))
    let kl_input_mixture = input.mul(&log_input.sub(&log_mixture)?)?;

    // KL(Q||M) = Σ Q * log(Q/M) = Σ Q * (log(Q) - log(M))
    let kl_target_mixture = target.mul(&log_target.sub(&log_mixture)?)?;

    // JS = 0.5 * (KL(P||M) + KL(Q||M))
    let js = kl_input_mixture.add(&kl_target_mixture)?.mul_scalar(0.5)?;

    reduction.apply(js)
}

/// Cross entropy loss for probability distributions
///
/// Computes the cross entropy between two probability distributions.
/// This is different from classification cross entropy as it works with continuous distributions.
///
/// # Arguments
/// * `input` - Predicted probability distribution (in linear space)
/// * `target` - True probability distribution (in linear space)
/// * `reduction` - Specifies the reduction to apply to the output
///
/// # Mathematical Definition
/// ```text
/// H(P, Q) = -Σ P(x) * log(Q(x))
/// ```
pub fn cross_entropy_continuous(
    input: &Tensor,
    target: &Tensor,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(input, target)?;

    // Avoid log(0) by clamping
    let eps = 1e-8_f32;
    let input_safe = input.clamp(eps, f32::MAX)?;

    // Cross entropy: -Σ target * log(input)
    let log_input = input_safe.log()?;
    let cross_entropy = target.mul(&log_input)?.neg()?;

    reduction.apply(cross_entropy)
}

/// Mutual information estimation loss
///
/// Estimates mutual information between two random variables using the MINE (Mutual Information Neural Estimation) approach.
/// This is an advanced loss function used in representation learning and generative models.
///
/// # Arguments
/// * `joint_samples` - Samples from the joint distribution P(X,Y)
/// * `marginal_samples` - Samples from the product of marginals P(X)P(Y)
/// * `reduction` - Specifies the reduction to apply to the output
///
/// # Mathematical Definition
/// Uses the Donsker-Varadhan representation:
/// ```text
/// MI(X;Y) = sup_θ E_{P(x,y)}[T_θ(x,y)] - log(E_{P(x)P(y)}[e^{T_θ(x,y)}])
/// ```
pub fn mutual_information_loss(
    joint_samples: &Tensor,
    marginal_samples: &Tensor,
    reduction: ReductionType,
) -> TorshResult<Tensor> {
    validate_elementwise_shapes(joint_samples, marginal_samples)?;

    // First term: mean of joint samples (maximize this)
    let joint_mean = joint_samples.mean(None, false)?;

    // Second term: log of mean of exp(marginal samples) (minimize this)
    let exp_marginal = marginal_samples.exp()?;
    let marginal_mean_exp = exp_marginal.mean(None, false)?;
    let log_marginal_mean_exp = marginal_mean_exp.log()?;

    // MINE lower bound: E[T(x,y)] - log(E[e^T(x',y')])
    // We want to maximize this, so we minimize the negative
    let mi_estimate = joint_mean.sub(&log_marginal_mean_exp)?;
    let loss = mi_estimate.neg()?; // Negative because we want to maximize MI

    reduction.apply(loss)
}

/// Entropy loss
///
/// Computes the entropy of a probability distribution.
/// Can be used as a regularization term to encourage diversity.
///
/// # Arguments
/// * `input` - Probability distribution (in linear space)
/// * `reduction` - Specifies the reduction to apply to the output
///
/// # Mathematical Definition
/// ```text
/// H(P) = -Σ P(x) * log(P(x))
/// ```
pub fn entropy_loss(input: &Tensor, reduction: ReductionType) -> TorshResult<Tensor> {
    // Avoid log(0) by clamping
    let eps = 1e-8_f32;
    let input_safe = input.clamp(eps, f32::MAX)?;

    // Entropy: -Σ P * log(P)
    let log_input = input_safe.log()?;
    let entropy = input.mul(&log_input)?.neg()?;

    reduction.apply(entropy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_kl_div_identical_distributions() -> TorshResult<()> {
        // KL divergence between identical distributions should be 0
        // First create a probability distribution
        let probs = from_vec(vec![0.5, 0.3, 0.2], &[3], DeviceType::Cpu)?;

        // Then compute the log probabilities
        let log_probs = probs.log()?;

        // For identical distributions, KL(P||P) should be 0
        let kl = kl_div(&log_probs, &probs, ReductionType::Sum, false)?;
        let kl_value = kl.item()?;

        // KL divergence should be close to 0 for identical distributions
        assert!(kl_value.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_kl_div_different_distributions() -> TorshResult<()> {
        let log_input = from_vec(vec![-1.0, -2.0, -1.5], &[3], DeviceType::Cpu)?; // log probabilities
        let target = from_vec(vec![0.8, 0.1, 0.1], &[3], DeviceType::Cpu)?; // very different distribution

        let kl = kl_div(&log_input, &target, ReductionType::Sum, false)?;
        let kl_value = kl.item()?;

        // KL divergence should be positive for different distributions
        assert!(kl_value > 0.0);
        Ok(())
    }

    #[test]
    fn test_js_divergence_identical_distributions() -> TorshResult<()> {
        let p = from_vec(vec![0.5, 0.3, 0.2], &[3], DeviceType::Cpu)?;
        let q = p.clone();

        let js = js_divergence(&p, &q, ReductionType::Sum)?;
        let js_value = js.item()?;

        // JS divergence should be 0 for identical distributions
        assert!(js_value.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_js_divergence_properties() -> TorshResult<()> {
        let p = from_vec(vec![0.7, 0.2, 0.1], &[3], DeviceType::Cpu)?;
        let q = from_vec(vec![0.1, 0.2, 0.7], &[3], DeviceType::Cpu)?;

        let js = js_divergence(&p, &q, ReductionType::Sum)?;
        let js_value = js.item()?;

        // JS divergence should be non-negative and bounded by log(2) ≈ 0.693
        assert!(js_value >= 0.0);
        assert!(js_value <= 0.7); // log(2) with some tolerance
        Ok(())
    }

    #[test]
    fn test_entropy_loss_uniform_distribution() -> TorshResult<()> {
        // Uniform distribution over 4 outcomes
        let uniform = from_vec(vec![0.25, 0.25, 0.25, 0.25], &[4], DeviceType::Cpu)?;

        let entropy = entropy_loss(&uniform, ReductionType::Sum)?;
        let entropy_value = entropy.item()?;

        // Entropy of uniform distribution over n outcomes is log(n)
        // For n=4, entropy should be log(4) ≈ 1.386
        let expected_entropy = 4.0f32.ln();
        assert!((entropy_value - expected_entropy).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn test_cross_entropy_continuous_basic() -> TorshResult<()> {
        let p = from_vec(vec![0.5, 0.3, 0.2], &[3], DeviceType::Cpu)?; // true distribution
        let q = from_vec(vec![0.4, 0.4, 0.2], &[3], DeviceType::Cpu)?; // predicted distribution

        let cross_entropy = cross_entropy_continuous(&q, &p, ReductionType::Sum)?;
        let ce_value = cross_entropy.item()?;

        // Cross entropy should be positive
        assert!(ce_value > 0.0);
        Ok(())
    }
}
