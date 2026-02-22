//! Classifier-Free Guidance utilities
//!
//! This module implements Classifier-Free Guidance (CFG) for quality improvement
//! in diffusion models. CFG combines conditional and unconditional predictions to
//! enhance sample quality and adherence to conditioning signals.
//!
//! # Overview
//!
//! Classifier-Free Guidance works by:
//! 1. Computing both conditional and unconditional noise predictions
//! 2. Combining them with a guidance scale: `pred = uncond + scale * (cond - uncond)`
//! 3. Higher scales increase conditioning strength but may reduce diversity
//!
//! # Typical Guidance Scales
//!
//! - **1.0**: No guidance (pure conditional)
//! - **7.5**: Default for Stable Diffusion (good balance)
//! - **15.0**: Strong guidance (more adherence, less diversity)
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_models::diffusion::guidance::{apply_classifier_free_guidance, prepare_cfg_batch};
//!
//! // Prepare batched input
//! let cfg_batch = prepare_cfg_batch(&cond_latents, &uncond_latents)?;
//!
//! // Run model on batched input
//! let batch_pred = model.forward(&cfg_batch)?;
//!
//! // Split predictions
//! let (cond_pred, uncond_pred) = split_cfg_batch(&batch_pred, batch_size)?;
//!
//! // Apply CFG
//! let guided_pred = apply_classifier_free_guidance(&cond_pred, &uncond_pred, 7.5)?;
//! ```

use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Apply Classifier-Free Guidance to noise predictions
///
/// Combines conditional and unconditional predictions using the formula:
/// ```text
/// noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
/// ```
///
/// # Arguments
///
/// * `cond_pred` - Conditional noise prediction tensor
/// * `uncond_pred` - Unconditional noise prediction tensor
/// * `guidance_scale` - Guidance strength (typically 7.5 for Stable Diffusion)
///
/// # Returns
///
/// Guided noise prediction tensor
///
/// # Errors
///
/// Returns error if:
/// - Tensor shapes don't match
/// - Tensor operations fail
///
/// # Examples
///
/// ```rust,ignore
/// let guided = apply_classifier_free_guidance(&cond_pred, &uncond_pred, 7.5)?;
/// ```
pub fn apply_classifier_free_guidance(
    cond_pred: &Tensor,
    uncond_pred: &Tensor,
    guidance_scale: f32,
) -> Result<Tensor> {
    // Validate shapes match
    if cond_pred.shape().dims() != uncond_pred.shape().dims() {
        return Err(TorshError::InvalidShape(format!(
            "Conditional and unconditional predictions must have same shape: {:?} vs {:?}",
            cond_pred.shape().dims(),
            uncond_pred.shape().dims()
        )));
    }

    // Special case: guidance_scale = 0.0 returns uncond_pred
    if guidance_scale == 0.0 {
        return Ok(uncond_pred.clone());
    }

    // Special case: guidance_scale = 1.0 returns cond_pred
    if guidance_scale == 1.0 {
        return Ok(cond_pred.clone());
    }

    // Compute: uncond + scale * (cond - uncond)
    // = uncond + scale * cond - scale * uncond
    // = (1 - scale) * uncond + scale * cond
    let diff = cond_pred.sub(uncond_pred)?;
    let scaled_diff = diff.mul_scalar(guidance_scale)?;
    let result = uncond_pred.add_op(&scaled_diff)?;

    Ok(result)
}

/// Prepare batched input for CFG by concatenating conditional and unconditional latents
///
/// Concatenates along the batch dimension for efficient batched processing.
/// The resulting tensor can be processed in a single forward pass.
///
/// # Arguments
///
/// * `cond_latents` - Conditional latent tensor [B, C, H, W]
/// * `uncond_latents` - Unconditional latent tensor [B, C, H, W]
///
/// # Returns
///
/// Concatenated tensor [2*B, C, H, W] where first B samples are conditional
///
/// # Errors
///
/// Returns error if:
/// - Tensor shapes don't match (except batch dimension)
/// - Concatenation fails
///
/// # Examples
///
/// ```rust,ignore
/// let cfg_batch = prepare_cfg_batch(&cond_latents, &uncond_latents)?;
/// assert_eq!(cfg_batch.shape().dims()[0], 2 * batch_size);
/// ```
pub fn prepare_cfg_batch(cond_latents: &Tensor, uncond_latents: &Tensor) -> Result<Tensor> {
    let cond_shape = cond_latents.shape();
    let uncond_shape = uncond_latents.shape();

    // Validate shapes match except for batch dimension
    if cond_shape.ndim() != uncond_shape.ndim() {
        return Err(TorshError::InvalidShape(format!(
            "Tensors must have same number of dimensions: {} vs {}",
            cond_shape.ndim(),
            uncond_shape.ndim()
        )));
    }

    for i in 1..cond_shape.ndim() {
        if cond_shape.dims()[i] != uncond_shape.dims()[i] {
            return Err(TorshError::InvalidShape(format!(
                "Tensor dimensions must match except batch dim: {:?} vs {:?}",
                cond_shape.dims(),
                uncond_shape.dims()
            )));
        }
    }

    // Concatenate along batch dimension (dim 0)
    // Note: Tensor::cat in torsh-tensor is incomplete, so we use manual concatenation
    concatenate_batch_dim(cond_latents, uncond_latents)
}

/// Helper function to concatenate two tensors along batch dimension (dim 0)
///
/// This is a workaround for incomplete Tensor::cat implementation in torsh-tensor.
/// Specifically handles 4D tensors [B, C, H, W] concatenation along batch dimension.
fn concatenate_batch_dim(tensor1: &Tensor, tensor2: &Tensor) -> Result<Tensor> {
    let shape1 = tensor1.shape();
    let shape2 = tensor2.shape();
    let dims1 = shape1.dims();
    let dims2 = shape2.dims();

    // Validate shapes match except for batch dimension
    if dims1.len() != dims2.len() {
        return Err(TorshError::InvalidShape(format!(
            "Tensors must have same number of dimensions: {} vs {}",
            dims1.len(),
            dims2.len()
        )));
    }

    for i in 1..dims1.len() {
        if dims1[i] != dims2[i] {
            return Err(TorshError::InvalidShape(format!(
                "Dimension {} must match: {} vs {}",
                i, dims1[i], dims2[i]
            )));
        }
    }

    // Get data from both tensors
    let data1 = tensor1.to_vec()?;
    let data2 = tensor2.to_vec()?;

    // Concatenate data
    let mut combined_data = Vec::with_capacity(data1.len() + data2.len());
    combined_data.extend_from_slice(&data1);
    combined_data.extend_from_slice(&data2);

    // Calculate output shape
    let mut output_shape = dims1.to_vec();
    output_shape[0] = dims1[0] + dims2[0]; // Concatenate batch dimension

    // Create result tensor
    Tensor::from_vec(combined_data, &output_shape)
}

/// Split batched CFG predictions back into conditional and unconditional parts
///
/// Splits a batched prediction tensor from CFG processing back into separate
/// conditional and unconditional predictions.
///
/// # Arguments
///
/// * `batch_pred` - Batched prediction tensor [2*B, C, H, W]
/// * `batch_size` - Original batch size B (not 2*B)
///
/// # Returns
///
/// Tuple of (conditional_pred, unconditional_pred), each of shape [B, C, H, W]
///
/// # Errors
///
/// Returns error if:
/// - batch_pred has wrong batch size (not 2*B)
/// - Splitting fails
///
/// # Examples
///
/// ```rust,ignore
/// let (cond_pred, uncond_pred) = split_cfg_batch(&batch_pred, batch_size)?;
/// assert_eq!(cond_pred.shape().dims()[0], batch_size);
/// assert_eq!(uncond_pred.shape().dims()[0], batch_size);
/// ```
pub fn split_cfg_batch(batch_pred: &Tensor, batch_size: usize) -> Result<(Tensor, Tensor)> {
    let pred_shape = batch_pred.shape();
    let pred_batch_size = pred_shape.dims()[0];

    // Validate batch size
    if pred_batch_size != 2 * batch_size {
        return Err(TorshError::InvalidShape(format!(
            "Expected batch size {}, got {}. Batch prediction should be 2x original batch size.",
            2 * batch_size,
            pred_batch_size
        )));
    }

    // Split into two halves along batch dimension
    let cond_pred = batch_pred.narrow(0, 0, batch_size)?;
    let uncond_pred = batch_pred.narrow(0, batch_size as i64, batch_size)?;

    Ok((cond_pred, uncond_pred))
}

/// Compute effective guidance scale with warmup
///
/// Gradually increases guidance scale during initial denoising steps to avoid
/// instability. Useful for very high guidance scales (>10).
///
/// # Arguments
///
/// * `timestep` - Current timestep (0 to num_steps)
/// * `num_steps` - Total number of denoising steps
/// * `target_scale` - Target guidance scale
/// * `warmup_steps` - Number of steps for warmup (default: num_steps / 10)
///
/// # Returns
///
/// Effective guidance scale for current timestep
///
/// # Examples
///
/// ```rust,ignore
/// let scale = guidance_scale_with_warmup(5, 50, 15.0, Some(10));
/// // Returns gradually increasing scale from 1.0 to 15.0 over first 10 steps
/// ```
pub fn guidance_scale_with_warmup(
    timestep: usize,
    num_steps: usize,
    target_scale: f32,
    warmup_steps: Option<usize>,
) -> f32 {
    let warmup = warmup_steps.unwrap_or(num_steps / 10);

    if timestep >= warmup {
        target_scale
    } else {
        // Linear warmup from 1.0 to target_scale
        let progress = timestep as f32 / warmup as f32;
        1.0 + progress * (target_scale - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{ones, zeros};

    #[test]
    fn test_cfg_basic_formula() {
        // Create simple test tensors
        let cond = ones(&[2, 4, 8, 8]).expect("Failed to create cond tensor");
        let uncond = zeros(&[2, 4, 8, 8]).expect("Failed to create uncond tensor");

        // Apply CFG with scale 2.0
        let result = apply_classifier_free_guidance(&cond, &uncond, 2.0);
        assert!(result.is_ok(), "CFG application failed");

        // Expected: uncond + 2.0 * (cond - uncond) = 0 + 2.0 * (1 - 0) = 2.0
        if let Ok(result) = result {
            let result_data = result.to_vec().expect("Failed to convert result");
            assert!(
                result_data.iter().all(|&x| (x - 2.0).abs() < 1e-5),
                "CFG formula incorrect"
            );
        }
    }

    #[test]
    fn test_cfg_batch_preparation() {
        let batch_size = 3;
        let cond = ones(&[batch_size, 4, 16, 16]).expect("Failed to create cond");
        let uncond = zeros(&[batch_size, 4, 16, 16]).expect("Failed to create uncond");

        let batch = prepare_cfg_batch(&cond, &uncond);
        assert!(batch.is_ok(), "CFG batch preparation failed");

        if let Ok(batch) = batch {
            let batch_shape = batch.shape();
            assert_eq!(
                batch_shape.dims()[0],
                2 * batch_size,
                "Batch size should be 2x original"
            );
            assert_eq!(batch_shape.dims()[1], 4, "Channel dimension mismatch");
        }
    }

    #[test]
    fn test_cfg_split_reconstruction() {
        let batch_size = 2;
        let cond = ones(&[batch_size, 4, 8, 8]).expect("Failed to create cond");
        let uncond = zeros(&[batch_size, 4, 8, 8]).expect("Failed to create uncond");

        let batch = prepare_cfg_batch(&cond, &uncond).expect("Batch prep failed");
        let (cond_split, uncond_split) = split_cfg_batch(&batch, batch_size).expect("Split failed");

        // Check shapes
        assert_eq!(cond_split.shape().dims()[0], batch_size);
        assert_eq!(uncond_split.shape().dims()[0], batch_size);

        // Check values (conditional should be ones, unconditional should be zeros)
        let cond_data = cond_split.to_vec().expect("Failed to get cond data");
        let uncond_data = uncond_split.to_vec().expect("Failed to get uncond data");

        assert!(
            cond_data.iter().all(|&x| (x - 1.0).abs() < 1e-5),
            "Conditional data mismatch"
        );
        assert!(
            uncond_data.iter().all(|&x| x.abs() < 1e-5),
            "Unconditional data mismatch"
        );
    }

    #[test]
    fn test_guidance_scale_effects() {
        let cond = ones(&[1, 4, 4, 4]).expect("Failed to create cond");
        let uncond = zeros(&[1, 4, 4, 4]).expect("Failed to create uncond");

        // Test different guidance scales
        let scales = [0.0, 1.0, 2.0, 7.5, 15.0];
        let expected = [0.0, 1.0, 2.0, 7.5, 15.0]; // Since cond=1, uncond=0

        for (scale, expected_val) in scales.iter().zip(expected.iter()) {
            let result =
                apply_classifier_free_guidance(&cond, &uncond, *scale).expect("CFG failed");
            let result_data = result.to_vec().expect("Failed to get data");

            assert!(
                result_data.iter().all(|&x| (x - expected_val).abs() < 1e-5),
                "Scale {} produced wrong result",
                scale
            );
        }
    }

    #[test]
    fn test_zero_guidance_scale() {
        let cond = ones(&[1, 4, 4, 4]).expect("Failed to create cond");
        let uncond = zeros(&[1, 4, 4, 4]).expect("Failed to create uncond");

        let result = apply_classifier_free_guidance(&cond, &uncond, 0.0).expect("CFG failed");
        let result_data = result.to_vec().expect("Failed to get data");

        // Should return uncond (all zeros)
        assert!(
            result_data.iter().all(|&x| x.abs() < 1e-5),
            "Zero guidance should return uncond"
        );
    }

    #[test]
    fn test_one_guidance_scale() {
        let cond = ones(&[1, 4, 4, 4]).expect("Failed to create cond");
        let uncond = zeros(&[1, 4, 4, 4]).expect("Failed to create uncond");

        let result = apply_classifier_free_guidance(&cond, &uncond, 1.0).expect("CFG failed");
        let result_data = result.to_vec().expect("Failed to get data");

        // Should return cond (all ones)
        assert!(
            result_data.iter().all(|&x| (x - 1.0).abs() < 1e-5),
            "Unit guidance should return cond"
        );
    }

    #[test]
    fn test_high_guidance_scale_amplification() {
        let cond = ones(&[1, 4, 4, 4]).expect("Failed to create cond");
        let uncond = zeros(&[1, 4, 4, 4]).expect("Failed to create uncond");

        let result = apply_classifier_free_guidance(&cond, &uncond, 20.0).expect("CFG failed");
        let result_data = result.to_vec().expect("Failed to get data");

        // Should amplify to 20.0
        assert!(
            result_data.iter().all(|&x| (x - 20.0).abs() < 1e-5),
            "High guidance should amplify correctly"
        );
    }

    #[test]
    fn test_mismatched_shapes_error() {
        let cond = ones(&[2, 4, 8, 8]).expect("Failed to create cond");
        let uncond = zeros(&[2, 4, 16, 16]).expect("Failed to create uncond");

        let result = apply_classifier_free_guidance(&cond, &uncond, 7.5);
        assert!(result.is_err(), "Should fail with mismatched shapes");
    }

    #[test]
    fn test_invalid_batch_size_error() {
        let batch = ones(&[5, 4, 8, 8]).expect("Failed to create batch");

        // Try to split with wrong batch size
        let result = split_cfg_batch(&batch, 3);
        assert!(result.is_err(), "Should fail with wrong batch size");
    }

    #[test]
    fn test_guidance_warmup() {
        let num_steps = 50;
        let target_scale = 15.0;
        let warmup_steps = 10;

        // Test warmup progression
        for step in 0..warmup_steps {
            let scale =
                guidance_scale_with_warmup(step, num_steps, target_scale, Some(warmup_steps));
            assert!(
                scale >= 1.0 && scale <= target_scale,
                "Warmup scale out of range"
            );
            assert!(
                scale < target_scale,
                "Should not reach target during warmup"
            );
        }

        // Test post-warmup
        let scale =
            guidance_scale_with_warmup(warmup_steps, num_steps, target_scale, Some(warmup_steps));
        assert_eq!(scale, target_scale, "Should reach target after warmup");

        let scale =
            guidance_scale_with_warmup(num_steps - 1, num_steps, target_scale, Some(warmup_steps));
        assert_eq!(scale, target_scale, "Should maintain target scale");
    }

    #[test]
    fn test_guidance_warmup_default() {
        let num_steps = 50;
        let target_scale = 10.0;

        // Default warmup should be num_steps / 10 = 5
        let scale_early = guidance_scale_with_warmup(0, num_steps, target_scale, None);
        assert_eq!(scale_early, 1.0, "Should start at 1.0");

        let scale_late = guidance_scale_with_warmup(10, num_steps, target_scale, None);
        assert_eq!(
            scale_late, target_scale,
            "Should reach target after default warmup"
        );
    }
}
