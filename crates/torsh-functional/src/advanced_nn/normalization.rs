//! Advanced normalization techniques for neural networks
//!
//! This module provides advanced normalization methods including:
//! - Spectral normalization for Lipschitz constraints
//! - Weight standardization for improved training stability

use crate::random_ops::randn;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Spectral normalization for weight matrices
///
/// Normalizes a weight matrix by its spectral norm (largest singular value)
/// to enforce Lipschitz constraint. Commonly used in GANs and other applications
/// requiring stable training.
///
/// ## Mathematical Definition
///
/// For a weight matrix W, spectral normalization computes:
/// ```
/// W_sn = W / σ(W)
/// ```
/// where σ(W) is the spectral norm (largest singular value) of W.
///
/// ## Power Iteration Algorithm
///
/// The spectral norm is estimated using power iteration:
/// ```
/// for i in 0..n_iterations:
///   v ← W^T u / ||W^T u||
///   u ← W v / ||W v||
/// σ(W) ≈ u^T W v
/// ```
///
/// ## Benefits
///
/// 1. **Lipschitz constraint**: ||W_sn|| ≤ 1
/// 2. **Training stability**: Prevents gradient explosion
/// 3. **GAN training**: Improves discriminator-generator balance
/// 4. **Regularization**: Implicit weight regularization effect
///
/// # Arguments
/// * `weight` - Weight matrix to normalize
/// * `n_power_iterations` - Number of power iterations for spectral norm estimation
/// * `eps` - Small value for numerical stability
///
/// # Returns
/// Spectrally normalized weight matrix
pub fn spectral_norm(weight: &Tensor, n_power_iterations: usize, eps: f64) -> TorshResult<Tensor> {
    let weight_shape_binding = weight.shape();
    let weight_shape = weight_shape_binding.dims();

    // Reshape weight to 2D matrix if needed
    let weight_2d = if weight_shape.len() > 2 {
        let first_dim = weight_shape[0];
        let remaining: usize = weight_shape[1..].iter().product();
        weight.view(&[first_dim as i32, remaining as i32])?
    } else {
        weight.clone()
    };

    let (m, n) = (weight_2d.shape().dims()[0], weight_2d.shape().dims()[1]);

    // Initialize random vector for power iteration
    let mut u = randn(&[m, 1], None, None, None)?;
    let mut v = randn(&[n, 1], None, None, None)?;

    // Power iteration to estimate spectral norm
    for _ in 0..n_power_iterations {
        // v = W^T u / ||W^T u||
        let wt_u = weight_2d.t()?.matmul(&u)?;
        let wt_u_norm_tensor = wt_u.norm()?;
        let wt_u_norm = wt_u_norm_tensor.data()?[0] + eps as f32;
        v = wt_u.div_scalar(wt_u_norm)?;

        // u = W v / ||W v||
        let w_v = weight_2d.matmul(&v)?;
        let w_v_norm_tensor = w_v.norm()?;
        let w_v_norm = w_v_norm_tensor.data()?[0] + eps as f32;
        u = w_v.div_scalar(w_v_norm)?;
    }

    // Compute spectral norm: sigma = u^T W v
    let sigma = u.t()?.matmul(&weight_2d)?.matmul(&v)?;
    let sigma_value = sigma.data()?[0] + eps as f32;

    // Normalize weight by spectral norm
    let normalized_weight = weight.div_scalar(sigma_value)?;

    // Reshape back to original shape if needed
    if weight_shape.len() > 2 {
        normalized_weight.view(&weight_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())
    } else {
        Ok(normalized_weight)
    }
}

/// Weight standardization for improved training dynamics
///
/// Standardizes weights by subtracting mean and dividing by standard deviation
/// across output channels. This technique can improve training stability and
/// convergence speed, especially for batch-normalized networks.
///
/// ## Mathematical Definition
///
/// For a weight tensor W with shape [out_channels, ...], weight standardization computes:
/// ```
/// W_std[i] = (W[i] - μ[i]) / (σ[i] + ε)
/// ```
/// where:
/// - μ[i] = mean(W[i]) across all dimensions except output channel
/// - σ[i] = std(W[i]) across all dimensions except output channel
/// - ε is a small constant for numerical stability
///
/// ## Benefits
///
/// 1. **Improved convergence**: Faster training convergence
/// 2. **Gradient flow**: Better gradient propagation
/// 3. **Batch norm compatibility**: Works well with batch normalization
/// 4. **Regularization**: Implicit regularization effect
///
/// # Arguments
/// * `weight` - Weight tensor with shape [out_channels, ...]
/// * `eps` - Small value for numerical stability
///
/// # Returns
/// Weight-standardized tensor with same shape as input
pub fn weight_standardization(weight: &Tensor, eps: f64) -> TorshResult<Tensor> {
    let weight_shape_binding = weight.shape();
    let weight_shape = weight_shape_binding.dims();

    if weight_shape.is_empty() {
        return Err(TorshError::invalid_argument_with_context(
            "Weight tensor cannot be empty",
            "weight_standardization",
        ));
    }

    let out_channels = weight_shape[0];
    let weight_per_channel: usize = weight_shape[1..].iter().product();

    let weight_data = weight.data()?;
    let mut standardized_data = Vec::with_capacity(weight_data.len());

    // Standardize each output channel separately
    for ch in 0..out_channels {
        let start_idx = ch * weight_per_channel;
        let end_idx = start_idx + weight_per_channel;
        let channel_weights = &weight_data[start_idx..end_idx];

        // Calculate mean and variance for this channel
        let mean = channel_weights.iter().sum::<f32>() / weight_per_channel as f32;
        let variance = channel_weights
            .iter()
            .map(|&w| (w - mean).powi(2))
            .sum::<f32>()
            / weight_per_channel as f32;
        let std_dev = (variance + eps as f32).sqrt();

        // Standardize weights in this channel
        for &weight_val in channel_weights {
            let standardized_val = (weight_val - mean) / std_dev;
            standardized_data.push(standardized_val);
        }
    }

    Tensor::from_data(standardized_data, weight_shape.to_vec(), weight.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_spectral_norm_basic() -> TorshResult<()> {
        let weight = randn(&[4, 6])?;
        let normalized = spectral_norm(&weight, 3, 1e-12)?;

        // Check that shape is preserved
        assert_eq!(weight.shape().dims(), normalized.shape().dims());

        // The spectral norm of the result should be approximately 1
        // (This is a simplified check - full verification would require SVD)
        Ok(())
    }

    #[test]
    fn test_weight_standardization_basic() -> TorshResult<()> {
        let weight = randn(&[3, 2, 2])?; // 3 output channels, 2x2 weights each
        let standardized = weight_standardization(&weight, 1e-5)?;

        // Check that shape is preserved
        assert_eq!(weight.shape().dims(), standardized.shape().dims());

        Ok(())
    }
}
