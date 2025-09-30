//! Loss functions for neural networks
//!
//! This module provides PyTorch-compatible loss functions with standardized APIs.
//! All loss functions follow consistent parameter ordering and validation patterns.
//!
//! # Organization
//!
//! The loss module is organized into focused sub-modules:
//!
//! - [`common`]: Common types and utilities shared across all loss functions
//! - [`regression`]: Loss functions for regression tasks (MSE, L1, Smooth L1, etc.)
//! - [`classification`]: Loss functions for classification tasks (Cross Entropy, NLL, Binary CE, etc.)
//! - [`similarity`]: Similarity and distance-based losses (Cosine, Triplet, Contrastive, etc.)
//! - [`information`]: Information theory losses (KL Divergence, Entropy, etc.)
//! - [`specialized`]: Specialized losses (CTC, Wasserstein, Gradient Penalty, etc.)
//!
//! # Examples
//!
//! ```rust
//! use torsh_functional::loss::{mse_loss, ReductionType};
//! use torsh_tensor::creation::from_vec;
//! use torsh_core::device::DeviceType;
//!
//! let input = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu).unwrap();
//! let target = from_vec(vec![1.5, 2.5, 2.5], &[3], DeviceType::Cpu).unwrap();
//! let loss = mse_loss(&input, &target, ReductionType::Mean).unwrap();
//! ```

// Sub-modules
pub mod classification;
pub mod common;
pub mod information;
pub mod regression;
pub mod similarity;
pub mod specialized;

// Re-export common types
pub use common::ReductionType;

// Re-export regression losses
pub use regression::{gaussian_nll_loss, l1_loss, mse_loss, poisson_nll_loss, smooth_l1_loss};

// Re-export classification losses
pub use classification::{
    binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy,
    cross_entropy_with_label_smoothing, focal_loss, multi_margin_loss, nll_loss,
};

// Re-export similarity losses
pub use similarity::{
    contrastive_loss, cosine_embedding_loss, hinge_embedding_loss, margin_ranking_loss,
    triplet_margin_loss, triplet_margin_with_distance_loss,
};

// Re-export information theory losses
pub use information::{
    cross_entropy_continuous, entropy_loss, js_divergence, kl_div, mutual_information_loss,
};

// Re-export specialized losses
pub use specialized::{
    ctc_loss, gradient_penalty_loss, seq2seq_loss_with_attention, temporal_consistency_loss,
    wasserstein_loss,
};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    /// Integration test to ensure all loss functions work together
    #[test]
    fn test_loss_functions_integration() -> torsh_core::Result<()> {
        let device = DeviceType::Cpu;

        // Test regression losses
        let input = from_vec(vec![1.0, 2.0, 3.0], &[3], device)?;
        let target = from_vec(vec![1.5, 2.5, 2.5], &[3], device)?;

        let _mse = mse_loss(&input, &target, ReductionType::Mean)?;
        let _l1 = l1_loss(&input, &target, ReductionType::Mean)?;
        let _smooth_l1 = smooth_l1_loss(&input, &target, ReductionType::Mean, 1.0)?;

        // Test classification losses
        let logits = from_vec(vec![1.0, 2.0, 0.5], &[1, 3], device)?;
        let class_target = from_vec(vec![0.0], &[1], device)?;

        let _ce = cross_entropy(&logits, &class_target, None, "mean", None, 0.0)?;
        let _focal = focal_loss(&logits, &class_target, 0.25, 2.0, ReductionType::Mean)?;

        // Test similarity losses
        let emb1 = from_vec(vec![1.0, 2.0], &[1, 2], device)?;
        let emb2 = from_vec(vec![1.1, 2.1], &[1, 2], device)?;
        let sim_target = from_vec(vec![1.0], &[1], device)?;

        let _cosine = cosine_embedding_loss(&emb1, &emb2, &sim_target, 0.0, ReductionType::Mean)?;

        // Test information theory losses
        let p = from_vec(vec![0.5, 0.3, 0.2], &[3], device)?;
        let q = from_vec(vec![0.4, 0.4, 0.2], &[3], device)?;

        let _entropy = entropy_loss(&p, ReductionType::Sum)?;
        let _js = js_divergence(&p, &q, ReductionType::Sum)?;

        Ok(())
    }

    /// Test that all loss functions produce finite, non-NaN values
    #[test]
    fn test_loss_functions_numerical_stability() -> torsh_core::Result<()> {
        let device = DeviceType::Cpu;

        // Test with edge case: very small values
        let small_input = from_vec(vec![1e-6, 1e-7, 1e-8], &[3], device)?;
        let small_target = from_vec(vec![1e-6, 1e-7, 1e-8], &[3], device)?;

        let mse = mse_loss(&small_input, &small_target, ReductionType::Mean)?;
        let mse_val = mse.item()?;
        assert!(mse_val.is_finite() && !mse_val.is_nan());

        // Test with edge case: large values
        let large_input = from_vec(vec![1e3, 1e4, 1e5], &[3], device)?;
        let large_target = from_vec(vec![1e3, 1e4, 1e5], &[3], device)?;

        let l1 = l1_loss(&large_input, &large_target, ReductionType::Mean)?;
        let l1_val = l1.item()?;
        assert!(l1_val.is_finite() && !l1_val.is_nan());

        Ok(())
    }

    /// Test reduction types work consistently across all loss functions
    #[test]
    fn test_reduction_consistency_across_losses() -> torsh_core::Result<()> {
        let device = DeviceType::Cpu;
        let input = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], device)?;
        let target = from_vec(vec![1.1, 2.1, 3.1, 4.1], &[4], device)?;

        // Test MSE with different reductions
        let mse_none = mse_loss(&input, &target, ReductionType::None)?;
        let mse_sum = mse_loss(&input, &target, ReductionType::Sum)?;
        let mse_mean = mse_loss(&input, &target, ReductionType::Mean)?;

        // Check shapes
        assert_eq!(mse_none.shape().dims(), &[4]);
        assert_eq!(mse_sum.shape().dims(), &[] as &[usize]);
        assert_eq!(mse_mean.shape().dims(), &[] as &[usize]);

        // Check relationships between reductions
        let manual_sum = mse_none.sum()?;
        let manual_mean = manual_sum.div_scalar(4.0)?;

        let sum_val = mse_sum.item()?;
        let mean_val = mse_mean.item()?;
        let manual_sum_val = manual_sum.item()?;
        let manual_mean_val = manual_mean.item()?;

        assert!((sum_val - manual_sum_val).abs() < 1e-6);
        assert!((mean_val - manual_mean_val).abs() < 1e-6);

        Ok(())
    }
}
