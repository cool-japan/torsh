//! Normalization operations for neural networks
//!
//! This module provides comprehensive normalization functions including batch normalization,
//! layer normalization, and other normalization techniques enhanced with SciRS2 integration.

use super::core::{validation, FunctionalConfig};
use crate::{func_error, validate_inputs};
use torsh_core::error::Result;
use torsh_tensor::Tensor;

// =============================================================================
// BATCH NORMALIZATION
// =============================================================================

/// Enhanced batch normalization with standardized API and SciRS2 numerical stability
pub fn batch_norm_2d(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Result<Tensor> {
    batch_norm_2d_with_config(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        &super::core::default_config(),
    )
}

/// Enhanced batch normalization with configuration
pub fn batch_norm_2d_with_config(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    training: bool,
    momentum: f32,
    eps: f32,
    config: &FunctionalConfig,
) -> Result<Tensor> {
    // Input validation with standardized error handling
    validate_inputs!(
        config,
        validation::validate_not_empty(input, "input"),
        validation::validate_min_ndim(input, 4, "input"),
        validation::validate_positive(eps, "eps"),
        validation::validate_range(momentum, 0.0, 1.0, "momentum")
    );

    // Enhanced batch normalization leveraging SciRS2's numerical stability techniques
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let batch_size = input_shape[0];
    let channels = input_shape[1];

    if training {
        // Compute batch statistics with enhanced numerical stability
        // Reshape input for channel-wise computation: [N, C, H, W] -> [N*H*W, C]
        let spatial_dims: usize = input_shape[2..].iter().product();
        let total_spatial = batch_size * spatial_dims;

        // Compute mean with Welford's online algorithm for numerical stability
        let reshaped = input.view(&[total_spatial as i32, channels as i32])?;
        let mean = reshaped.mean(Some(&[0]), false)?;

        // Compute variance using the stable two-pass algorithm
        let centered = reshaped.sub(&mean.unsqueeze(0)?)?;
        let variance = centered.pow_scalar(2.0)?.mean(Some(&[0]), false)?;

        // Add epsilon for numerical stability
        let eps_tensor = torsh_tensor::creation::full(&[channels], eps)?;
        let stable_var = variance.add(&eps_tensor)?;
        let inv_std = stable_var.rsqrt()?;

        // Apply normalization
        let normalized = centered.mul_op(&inv_std.unsqueeze(0)?)?;

        // Reshape back to original shape
        let input_shape_i32: Vec<i32> = input_shape.iter().map(|&x| x as i32).collect();
        let output = normalized.view(&input_shape_i32)?;

        // Apply scale and shift if provided
        let mut result = output;
        if let Some(w) = weight {
            let weight_expanded = w.view(&[1, channels as i32, 1, 1])?;
            result = result.mul_op(&weight_expanded)?;
        }
        if let Some(b) = bias {
            let bias_expanded = b.view(&[1, channels as i32, 1, 1])?;
            result = result.add(&bias_expanded)?;
        }

        // Update running statistics with momentum
        if let (Some(r_mean), Some(r_var)) = (running_mean, running_var) {
            // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            let momentum_tensor = torsh_tensor::creation::full(&[1], momentum)?;
            let one_minus_momentum = torsh_tensor::creation::full(&[1], 1.0 - momentum)?;

            let _new_running_mean = r_mean
                .mul_op(&one_minus_momentum)?
                .add(&mean.mul_op(&momentum_tensor)?)?;
            let _new_running_var = r_var
                .mul_op(&one_minus_momentum)?
                .add(&variance.mul_op(&momentum_tensor)?)?;

            // Note: In practice, these would update the module's buffers
        }

        Ok(result)
    } else {
        // Use running statistics for inference
        let default_mean = torsh_tensor::creation::zeros(&[channels])?;
        let default_var = torsh_tensor::creation::ones(&[channels])?;
        let r_mean = running_mean.unwrap_or(&default_mean);
        let r_var = running_var.unwrap_or(&default_var);

        // Apply normalization using running statistics
        let eps_tensor = torsh_tensor::creation::full(&[channels], eps)?;
        let stable_var = r_var.add(&eps_tensor)?;
        let inv_std = stable_var.rsqrt()?;

        let mean_expanded = r_mean.view(&[1, channels as i32, 1, 1])?;
        let inv_std_expanded = inv_std.view(&[1, channels as i32, 1, 1])?;

        let normalized = input.sub(&mean_expanded)?.mul_op(&inv_std_expanded)?;

        // Apply scale and shift
        let mut result = normalized;
        if let Some(w) = weight {
            let weight_expanded = w.view(&[1, channels as i32, 1, 1])?;
            result = result.mul_op(&weight_expanded)?;
        }
        if let Some(b) = bias {
            let bias_expanded = b.view(&[1, channels as i32, 1, 1])?;
            result = result.add(&bias_expanded)?;
        }

        Ok(result)
    }
}

/// 1D batch normalization
///
/// Applies batch normalization over a 3D input (batch, channels, length).
/// Normalizes over batch and spatial dimensions for each channel.
///
/// # Arguments
///
/// * `input` - Input tensor of shape [batch, channels, length]
/// * `weight` - Optional learnable affine scale parameters (gamma)
/// * `bias` - Optional learnable affine shift parameters (beta)
/// * `running_mean` - Running mean for inference
/// * `running_var` - Running variance for inference
/// * `training` - Whether in training mode
/// * `momentum` - Momentum for running statistics update
/// * `eps` - Small value for numerical stability
pub fn batch_norm_1d(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let length = input_shape[2];

    if training {
        // Compute batch statistics
        // Reshape to [batch * length, channels] to compute statistics across batch and spatial dims
        let total_spatial = batch_size * length;
        let reshaped = input.view(&[total_spatial as i32, channels as i32])?;
        let mean = reshaped.mean(Some(&[0]), false)?;

        // Compute variance
        let centered = reshaped.sub(&mean.unsqueeze(0)?)?;
        let variance = centered.pow_scalar(2.0)?.mean(Some(&[0]), false)?;

        // Add epsilon for numerical stability
        let eps_tensor = torsh_tensor::creation::full(&[channels], eps)?;
        let stable_var = variance.add(&eps_tensor)?;
        let inv_std = stable_var.rsqrt()?;

        // Apply normalization
        let normalized = centered.mul_op(&inv_std.unsqueeze(0)?)?;

        // Reshape back to original shape
        let input_shape_i32: Vec<i32> = input_shape.iter().map(|&x| x as i32).collect();
        let output = normalized.view(&input_shape_i32)?;

        // Apply scale and shift
        let mut result = output;
        if let Some(w) = weight {
            let weight_expanded = w.view(&[1, channels as i32, 1])?;
            result = result.mul_op(&weight_expanded)?;
        }
        if let Some(b) = bias {
            let bias_expanded = b.view(&[1, channels as i32, 1])?;
            result = result.add(&bias_expanded)?;
        }

        // Update running statistics (would be done by the module in practice)
        let _ = (running_mean, running_var, momentum, mean, variance);

        Ok(result)
    } else {
        // Use running statistics for inference
        let default_mean = torsh_tensor::creation::zeros(&[channels])?;
        let default_var = torsh_tensor::creation::ones(&[channels])?;
        let r_mean = running_mean.unwrap_or(&default_mean);
        let r_var = running_var.unwrap_or(&default_var);

        // Apply normalization
        let eps_tensor = torsh_tensor::creation::full(&[channels], eps)?;
        let stable_var = r_var.add(&eps_tensor)?;
        let inv_std = stable_var.rsqrt()?;

        let mean_expanded = r_mean.view(&[1, channels as i32, 1])?;
        let inv_std_expanded = inv_std.view(&[1, channels as i32, 1])?;

        let normalized = input.sub(&mean_expanded)?.mul_op(&inv_std_expanded)?;

        // Apply scale and shift
        let mut result = normalized;
        if let Some(w) = weight {
            let weight_expanded = w.view(&[1, channels as i32, 1])?;
            result = result.mul_op(&weight_expanded)?;
        }
        if let Some(b) = bias {
            let bias_expanded = b.view(&[1, channels as i32, 1])?;
            result = result.add(&bias_expanded)?;
        }

        Ok(result)
    }
}

/// 3D batch normalization
///
/// Applies batch normalization over a 5D input (batch, channels, depth, height, width).
/// Normalizes over batch and spatial dimensions for each channel.
///
/// # Arguments
///
/// * `input` - Input tensor of shape [batch, channels, depth, height, width]
/// * `weight` - Optional learnable affine scale parameters (gamma)
/// * `bias` - Optional learnable affine shift parameters (beta)
/// * `running_mean` - Running mean for inference
/// * `running_var` - Running variance for inference
/// * `training` - Whether in training mode
/// * `momentum` - Momentum for running statistics update
/// * `eps` - Small value for numerical stability
pub fn batch_norm_3d(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 5 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let depth = input_shape[2];
    let height = input_shape[3];
    let width = input_shape[4];

    if training {
        // Compute batch statistics
        // Reshape to [batch * depth * height * width, channels]
        let total_spatial = batch_size * depth * height * width;
        let reshaped = input.view(&[total_spatial as i32, channels as i32])?;
        let mean = reshaped.mean(Some(&[0]), false)?;

        // Compute variance
        let centered = reshaped.sub(&mean.unsqueeze(0)?)?;
        let variance = centered.pow_scalar(2.0)?.mean(Some(&[0]), false)?;

        // Add epsilon for numerical stability
        let eps_tensor = torsh_tensor::creation::full(&[channels], eps)?;
        let stable_var = variance.add(&eps_tensor)?;
        let inv_std = stable_var.rsqrt()?;

        // Apply normalization
        let normalized = centered.mul_op(&inv_std.unsqueeze(0)?)?;

        // Reshape back to original shape
        let input_shape_i32: Vec<i32> = input_shape.iter().map(|&x| x as i32).collect();
        let output = normalized.view(&input_shape_i32)?;

        // Apply scale and shift
        let mut result = output;
        if let Some(w) = weight {
            let weight_expanded = w.view(&[1, channels as i32, 1, 1, 1])?;
            result = result.mul_op(&weight_expanded)?;
        }
        if let Some(b) = bias {
            let bias_expanded = b.view(&[1, channels as i32, 1, 1, 1])?;
            result = result.add(&bias_expanded)?;
        }

        // Update running statistics (would be done by the module in practice)
        let _ = (running_mean, running_var, momentum, mean, variance);

        Ok(result)
    } else {
        // Use running statistics for inference
        let default_mean = torsh_tensor::creation::zeros(&[channels])?;
        let default_var = torsh_tensor::creation::ones(&[channels])?;
        let r_mean = running_mean.unwrap_or(&default_mean);
        let r_var = running_var.unwrap_or(&default_var);

        // Apply normalization
        let eps_tensor = torsh_tensor::creation::full(&[channels], eps)?;
        let stable_var = r_var.add(&eps_tensor)?;
        let inv_std = stable_var.rsqrt()?;

        let mean_expanded = r_mean.view(&[1, channels as i32, 1, 1, 1])?;
        let inv_std_expanded = inv_std.view(&[1, channels as i32, 1, 1, 1])?;

        let normalized = input.sub(&mean_expanded)?.mul_op(&inv_std_expanded)?;

        // Apply scale and shift
        let mut result = normalized;
        if let Some(w) = weight {
            let weight_expanded = w.view(&[1, channels as i32, 1, 1, 1])?;
            result = result.mul_op(&weight_expanded)?;
        }
        if let Some(b) = bias {
            let bias_expanded = b.view(&[1, channels as i32, 1, 1, 1])?;
            result = result.add(&bias_expanded)?;
        }

        Ok(result)
    }
}

/// Batch normalization function (generic)
#[allow(clippy::too_many_arguments)]
pub fn batch_norm(
    input: &Tensor,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    momentum: f32,
    eps: f32,
) -> Result<Tensor> {
    // Generic batch normalization that dispatches to the appropriate implementation
    // based on input dimensionality
    let input_shape_obj = input.shape();
    let input_dims = input_shape_obj.dims();

    match input_dims.len() {
        3 => {
            // 1D batch normalization: [batch_size, channels, length]
            batch_norm_1d(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
            )
        }
        4 => {
            // 2D batch normalization: [batch_size, channels, height, width]
            batch_norm_2d(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
            )
        }
        5 => {
            // 3D batch normalization: [batch_size, channels, depth, height, width]
            batch_norm_3d(
                input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
            )
        }
        2 => {
            // For 2D input [batch_size, features], treat as 1D with length=1
            // Reshape to [batch_size, features, 1]
            let batch_size = input_dims[0] as i32;
            let features = input_dims[1] as i32;
            let reshaped_input = input.view(&[batch_size, features, 1])?;

            let result = batch_norm_1d(
                &reshaped_input,
                weight,
                bias,
                running_mean,
                running_var,
                training,
                momentum,
                eps,
            )?;

            // Reshape back to [batch_size, features]
            result.view(&[batch_size, features])
        }
        _ => {
            // For unsupported dimensionality, return error
            Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "batch_norm: expected 2D, 3D, 4D, or 5D input, got {}D",
                input_dims.len()
            )))
        }
    }
}

// =============================================================================
// LAYER NORMALIZATION
// =============================================================================

/// Enhanced layer normalization with SciRS2 numerical stability
pub fn layer_norm_enhanced(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    // Enhanced layer normalization with numerical stability improvements
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    let norm_dims = normalized_shape.len();
    let _norm_size: usize = normalized_shape.iter().product();

    // Create dimension indices for normalization
    let norm_dim_indices: Vec<usize> = (input_shape.len() - norm_dims..input_shape.len()).collect();

    // Compute mean and variance over the normalized dimensions
    let mean = input.mean(Some(&norm_dim_indices), true)?;
    let centered = input.sub(&mean)?;
    let variance = centered
        .pow_scalar(2.0)?
        .mean(Some(&norm_dim_indices), true)?;

    // Add epsilon and compute inverse standard deviation
    let eps_tensor = torsh_tensor::creation::full(&[1], eps)?;
    let stable_var = variance.add(&eps_tensor)?;
    let inv_std = stable_var.rsqrt()?;

    // Apply normalization
    let normalized = centered.mul_op(&inv_std)?;

    // Apply learnable parameters if provided
    let mut result = normalized;
    if let Some(w) = weight {
        result = result.mul_op(w)?;
    }
    if let Some(b) = bias {
        result = result.add(b)?;
    }

    Ok(result)
}

/// Layer normalization function
///
/// Normalizes the input over the last `normalized_shape.len()` dimensions.
/// This is the standard layer normalization used in transformers and other architectures.
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `normalized_shape` - Shape of the dimensions to normalize over (typically the last few dims)
/// * `weight` - Optional learnable affine scale parameters (gamma)
/// * `bias` - Optional learnable affine shift parameters (beta)
/// * `eps` - Small value for numerical stability (default: 1e-5)
///
/// # Example
///
/// For an input of shape [batch, seq_len, hidden_dim] with normalized_shape = \[hidden_dim\],
/// this will normalize over the hidden_dim dimension for each position independently.
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    // Use the enhanced implementation which provides numerical stability
    layer_norm_enhanced(input, normalized_shape, weight, bias, eps)
}

/// Layer normalization with configuration
pub fn layer_norm_configured(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
    config: &FunctionalConfig,
) -> Result<Tensor> {
    validate_inputs!(
        config,
        validation::validate_not_empty(input, "input"),
        validation::validate_positive(eps, "eps")
    );
    func_error!(
        layer_norm_enhanced(input, normalized_shape, weight, bias, eps),
        "Layer normalization"
    )
}

// =============================================================================
// GROUP NORMALIZATION
// =============================================================================

/// Group normalization
///
/// Divides channels into groups and normalizes within each group.
/// GroupNorm is effective for small batch sizes where BatchNorm struggles.
///
/// # Arguments
///
/// * `input` - Input tensor of shape [batch, channels, ...]
/// * `num_groups` - Number of groups to divide channels into
/// * `weight` - Optional learnable affine scale parameters (gamma)
/// * `bias` - Optional learnable affine shift parameters (beta)
/// * `eps` - Small value for numerical stability
pub fn group_norm(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() < 2 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Input must have at least 2 dimensions for group normalization".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];

    if channels % num_groups != 0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Number of channels ({}) must be divisible by number of groups ({})",
            channels, num_groups
        )));
    }

    let channels_per_group = channels / num_groups;

    // Calculate total spatial dimensions
    let spatial_size: usize = input_shape[2..].iter().product();

    // Reshape to [batch, num_groups, channels_per_group, spatial]
    let reshaped = input.view(&[
        batch_size as i32,
        num_groups as i32,
        channels_per_group as i32,
        spatial_size as i32,
    ])?;

    // Normalize over channels_per_group and spatial dimensions (dims 2 and 3)
    // Compute mean and variance for each group
    let group_size = channels_per_group * spatial_size;
    let flattened = reshaped.view(&[(batch_size * num_groups) as i32, group_size as i32])?;

    let mean = flattened.mean(Some(&[1]), true)?;
    let centered = flattened.sub(&mean)?;
    let variance = centered.pow_scalar(2.0)?.mean(Some(&[1]), true)?;

    // Add epsilon for numerical stability
    let eps_tensor = torsh_tensor::creation::full(&[1], eps)?;
    let stable_var = variance.add(&eps_tensor)?;
    let inv_std = stable_var.rsqrt()?;

    // Normalize
    let normalized = centered.mul_op(&inv_std)?;

    // Reshape back to [batch, num_groups, channels_per_group, spatial]
    let normalized = normalized.view(&[
        batch_size as i32,
        num_groups as i32,
        channels_per_group as i32,
        spatial_size as i32,
    ])?;

    // Reshape to original shape
    let input_shape_i32: Vec<i32> = input_shape.iter().map(|&x| x as i32).collect();
    let mut result = normalized.view(&input_shape_i32)?;

    // Apply scale and shift
    if let Some(w) = weight {
        // Reshape weight for broadcasting
        let mut weight_shape = vec![1, channels];
        weight_shape.extend(vec![1; input_shape.len() - 2]);
        let weight_shape_i32: Vec<i32> = weight_shape.iter().map(|&x| x as i32).collect();
        let weight_expanded = w.view(&weight_shape_i32)?;
        result = result.mul_op(&weight_expanded)?;
    }
    if let Some(b) = bias {
        // Reshape bias for broadcasting
        let mut bias_shape = vec![1, channels];
        bias_shape.extend(vec![1; input_shape.len() - 2]);
        let bias_shape_i32: Vec<i32> = bias_shape.iter().map(|&x| x as i32).collect();
        let bias_expanded = b.view(&bias_shape_i32)?;
        result = result.add(&bias_expanded)?;
    }

    Ok(result)
}

// =============================================================================
// INSTANCE NORMALIZATION
// =============================================================================

/// Instance normalization
///
/// Normalizes each channel of each sample independently across spatial dimensions.
/// InstanceNorm is commonly used in style transfer and image generation tasks.
///
/// # Arguments
///
/// * `input` - Input tensor of shape [batch, channels, ...]
/// * `weight` - Optional learnable affine scale parameters (gamma)
/// * `bias` - Optional learnable affine shift parameters (beta)
/// * `eps` - Small value for numerical stability
pub fn instance_norm(
    input: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() < 3 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Input must have at least 3 dimensions for instance normalization".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];

    // Calculate total spatial dimensions
    let spatial_size: usize = input_shape[2..].iter().product();

    // Reshape to [batch * channels, spatial]
    // This allows us to normalize each instance (batch, channel pair) independently
    let reshaped = input.view(&[(batch_size * channels) as i32, spatial_size as i32])?;

    // Compute mean and variance for each instance
    let mean = reshaped.mean(Some(&[1]), true)?;
    let centered = reshaped.sub(&mean)?;
    let variance = centered.pow_scalar(2.0)?.mean(Some(&[1]), true)?;

    // Add epsilon for numerical stability
    let eps_tensor = torsh_tensor::creation::full(&[1], eps)?;
    let stable_var = variance.add(&eps_tensor)?;
    let inv_std = stable_var.rsqrt()?;

    // Normalize
    let normalized = centered.mul_op(&inv_std)?;

    // Reshape back to original shape
    let input_shape_i32: Vec<i32> = input_shape.iter().map(|&x| x as i32).collect();
    let mut result = normalized.view(&input_shape_i32)?;

    // Apply scale and shift
    if let Some(w) = weight {
        // Reshape weight for broadcasting [1, channels, 1, 1, ...]
        let mut weight_shape = vec![1, channels];
        weight_shape.extend(vec![1; input_shape.len() - 2]);
        let weight_shape_i32: Vec<i32> = weight_shape.iter().map(|&x| x as i32).collect();
        let weight_expanded = w.view(&weight_shape_i32)?;
        result = result.mul_op(&weight_expanded)?;
    }
    if let Some(b) = bias {
        // Reshape bias for broadcasting [1, channels, 1, 1, ...]
        let mut bias_shape = vec![1, channels];
        bias_shape.extend(vec![1; input_shape.len() - 2]);
        let bias_shape_i32: Vec<i32> = bias_shape.iter().map(|&x| x as i32).collect();
        let bias_expanded = b.view(&bias_shape_i32)?;
        result = result.add(&bias_expanded)?;
    }

    Ok(result)
}

// =============================================================================
// LOCAL RESPONSE NORMALIZATION
// =============================================================================

/// Local Response Normalization (LRN)
///
/// Implements local response normalization as used in AlexNet.
/// Normalizes across nearby channels at each spatial location.
///
/// # Arguments
/// * `input` - Input tensor [batch, channels, height, width]
/// * `size` - Number of neighboring channels to normalize across
/// * `alpha` - Multiplicative factor for normalization
/// * `beta` - Exponent for normalization
/// * `k` - Additive constant for numerical stability
///
/// # Formula
/// `output[i] = input[i] / (k + alpha * sum(input[j]^2))^beta`
/// where j ranges over \[i - size/2, i + size/2\] clamped to valid channels
pub fn local_response_norm(
    input: &Tensor,
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
) -> Result<Tensor> {
    // LRN: output[b,c,h,w] = input[b,c,h,w] / (k + alpha * sum_adjacent_squares)^beta
    // where sum_adjacent_squares is the sum of squares of neighboring channels

    let input_shape_binding = input.shape();
    let input_shape = input_shape_binding.dims();

    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "LRN requires 4D input [batch, channels, height, width]".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0f32; input_data.len()];

    // Half window size (how many channels on each side to consider)
    let half_size = size / 2;

    // Process each element
    for b in 0..batch_size {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    // Calculate the range of channels to normalize over
                    let c_start = if c >= half_size { c - half_size } else { 0 };
                    let c_end = (c + half_size + 1).min(channels);

                    // Compute sum of squares across neighboring channels
                    let mut sum_squares = 0.0f32;
                    for c_neighbor in c_start..c_end {
                        let idx = b * channels * height * width
                            + c_neighbor * height * width
                            + h * width
                            + w;
                        sum_squares += input_data[idx] * input_data[idx];
                    }

                    // Apply LRN formula
                    let input_idx =
                        b * channels * height * width + c * height * width + h * width + w;
                    let scale = k + alpha * sum_squares;
                    output_data[input_idx] = input_data[input_idx] / scale.powf(beta);
                }
            }
        }
    }

    Tensor::from_vec(output_data, input_shape)
}

// =============================================================================
// SPECTRAL NORMALIZATION
// =============================================================================

/// Spectral normalization for weight matrices
///
/// Normalizes weights by their largest singular value to stabilize GAN training.
/// Uses power iteration to efficiently estimate the largest singular value.
///
/// # Arguments
/// * `weight` - Weight matrix to normalize [out_features, in_features]
/// * `u` - Left singular vector estimate (will be updated)
/// * `n_power_iterations` - Number of power iterations to perform
/// * `eps` - Small epsilon for numerical stability
///
/// # Returns
/// * Normalized weight matrix and updated u vector
pub fn spectral_norm(
    weight: &Tensor,
    u: &Tensor,
    n_power_iterations: usize,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    // Spectral normalization: W_sn = W / sigma(W)
    // where sigma(W) is the largest singular value
    //
    // Power iteration algorithm:
    // For i in 1..n_iterations:
    //   v = W^T @ u / ||W^T @ u||
    //   u = W @ v / ||W @ v||
    // sigma = u^T @ W @ v

    let weight_shape_binding = weight.shape();
    let weight_shape = weight_shape_binding.dims();

    if weight_shape.len() != 2 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Spectral normalization requires 2D weight matrix".to_string(),
        ));
    }

    let out_features = weight_shape[0];
    let in_features = weight_shape[1];

    let weight_data = weight.to_vec()?;
    let mut u_data = u.to_vec()?;

    // Ensure u has correct size
    if u_data.len() != out_features {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "u vector must have size {}, got {}",
            out_features,
            u_data.len()
        )));
    }

    // Power iteration to find dominant singular vector
    for _ in 0..n_power_iterations {
        // v = W^T @ u
        let mut v_data = vec![0.0f32; in_features];
        for j in 0..in_features {
            let mut sum = 0.0f32;
            for i in 0..out_features {
                sum += weight_data[i * in_features + j] * u_data[i];
            }
            v_data[j] = sum;
        }

        // Normalize v: v = v / ||v||
        let v_norm = (v_data.iter().map(|&x| x * x).sum::<f32>()).sqrt() + eps;
        for v in v_data.iter_mut() {
            *v /= v_norm;
        }

        // u = W @ v
        for i in 0..out_features {
            let mut sum = 0.0f32;
            for j in 0..in_features {
                sum += weight_data[i * in_features + j] * v_data[j];
            }
            u_data[i] = sum;
        }

        // Normalize u: u = u / ||u||
        let u_norm = (u_data.iter().map(|&x| x * x).sum::<f32>()).sqrt() + eps;
        for u in u_data.iter_mut() {
            *u /= u_norm;
        }
    }

    // Compute sigma = u^T @ W @ v (final iteration)
    // First compute v = W^T @ u
    let mut v_data = vec![0.0f32; in_features];
    for j in 0..in_features {
        let mut sum = 0.0f32;
        for i in 0..out_features {
            sum += weight_data[i * in_features + j] * u_data[i];
        }
        v_data[j] = sum;
    }

    // Normalize v
    let v_norm = (v_data.iter().map(|&x| x * x).sum::<f32>()).sqrt() + eps;
    for v in v_data.iter_mut() {
        *v /= v_norm;
    }

    // Compute sigma = u^T @ W @ v
    let mut sigma = 0.0f32;
    for i in 0..out_features {
        let mut row_dot_v = 0.0f32;
        for j in 0..in_features {
            row_dot_v += weight_data[i * in_features + j] * v_data[j];
        }
        sigma += u_data[i] * row_dot_v;
    }

    // Normalize weight by sigma: W_sn = W / sigma
    let sigma_with_eps = sigma.max(eps); // Prevent division by zero
    let normalized_weight_data: Vec<f32> =
        weight_data.iter().map(|&w| w / sigma_with_eps).collect();

    let normalized_weight = Tensor::from_vec(normalized_weight_data, weight_shape)?;
    let new_u = Tensor::from_vec(u_data, &[out_features])?;

    Ok((normalized_weight, new_u))
}

// =============================================================================
// WEIGHT NORMALIZATION
// =============================================================================

/// Weight normalization
pub fn weight_norm(weight: &Tensor, g: &Tensor, _dim: i32) -> Result<Tensor> {
    // Weight normalization: w = g * v / ||v||
    // where v is the weight vector and g is a learnable scalar

    // Compute norm along dimension: sqrt(sum(x^2, dim))
    let squared = weight.pow(2.0)?;
    let sum_squared = squared.sum()?;
    let norm = sum_squared.sqrt()?;
    let normalized = weight.div(&norm)?;
    let result = normalized.mul_op(g)?;

    Ok(result)
}

// =============================================================================
// RMS NORMALIZATION
// =============================================================================

/// RMS normalization (Root Mean Square normalization)
pub fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    // RMS normalization: x / sqrt(mean(x^2) + eps) * weight
    let squared = input.pow(2.0)?;
    let last_dim = input.shape().dims().len() - 1;
    let mean_squared = squared.mean(Some(&[last_dim]), true)?;
    let eps_tensor = torsh_tensor::creation::full_like(&mean_squared, eps)?;
    let rms = mean_squared.add(&eps_tensor)?.sqrt()?;
    let normalized = input.div(&rms)?;
    normalized.mul(weight)
}

// =============================================================================
// CONVENIENCE FUNCTIONS WITH STANDARDIZED API
// =============================================================================

/// Convenient normalization functions with standardized API
pub mod configured {
    use super::*;

    /// Batch normalization with configuration
    pub fn batch_norm_configured(
        input: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        running_mean: Option<&Tensor>,
        running_var: Option<&Tensor>,
        training: bool,
        momentum: f32,
        eps: f32,
        config: &FunctionalConfig,
    ) -> Result<Tensor> {
        batch_norm_2d_with_config(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            config,
        )
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Calculate the number of features to normalize over
pub fn get_norm_features(_input_shape: &[usize], normalized_shape: &[usize]) -> usize {
    // For layer norm, this is the product of the normalized dimensions
    normalized_shape.iter().product()
}

/// Validate normalization parameters
pub fn validate_norm_params(
    input_shape: &[usize],
    normalized_shape: &[usize],
    eps: f32,
) -> Result<()> {
    if eps <= 0.0 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Epsilon must be positive".to_string(),
        ));
    }

    if normalized_shape.len() > input_shape.len() {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Normalized shape cannot have more dimensions than input".to_string(),
        ));
    }

    // Check that normalized shape matches the last dimensions of input
    let input_suffix = &input_shape[input_shape.len() - normalized_shape.len()..];
    if input_suffix != normalized_shape {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Normalized shape {:?} doesn't match input shape suffix {:?}",
            normalized_shape, input_suffix
        )));
    }

    Ok(())
}

/// Create affine parameters for normalization layers
pub fn create_affine_params(
    shape: &[usize],
    init_weight: f32,
    init_bias: f32,
) -> Result<(Tensor, Tensor)> {
    let weight = torsh_tensor::creation::full(shape, init_weight)?;
    let bias = torsh_tensor::creation::full(shape, init_bias)?;
    Ok((weight, bias))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_layer_norm_basic() -> Result<()> {
        // Test basic layer normalization with a simple example
        // Input: [2, 3] - 2 samples, 3 features
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // Sample 1
                4.0, 5.0, 6.0, // Sample 2
            ],
            &[2, 3],
        )?;

        // Normalize over the last dimension (features)
        let output = layer_norm(&input, &[3], None, None, 1e-5)?;

        assert_eq!(output.shape().dims(), &[2, 3]);

        // For sample 1: mean = 2.0, std ≈ 0.8165
        // Normalized: [-1.2247, 0, 1.2247]
        // For sample 2: mean = 5.0, std ≈ 0.8165
        // Normalized: [-1.2247, 0, 1.2247]

        let output_data = output.to_vec()?;

        // Check that mean is approximately 0 for each sample
        let sample1_mean = (output_data[0] + output_data[1] + output_data[2]) / 3.0;
        let sample2_mean = (output_data[3] + output_data[4] + output_data[5]) / 3.0;

        assert_relative_eq!(sample1_mean, 0.0, epsilon = 1e-5);
        assert_relative_eq!(sample2_mean, 0.0, epsilon = 1e-5);

        // Check that variance is approximately 1 for each sample
        let sample1_var =
            (output_data[0].powi(2) + output_data[1].powi(2) + output_data[2].powi(2)) / 3.0;
        let sample2_var =
            (output_data[3].powi(2) + output_data[4].powi(2) + output_data[5].powi(2)) / 3.0;

        assert_relative_eq!(sample1_var, 1.0, epsilon = 1e-4);
        assert_relative_eq!(sample2_var, 1.0, epsilon = 1e-4);

        Ok(())
    }

    #[test]
    fn test_layer_norm_with_affine() -> Result<()> {
        // Test layer normalization with learnable affine parameters
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4])?;

        // Create weight (scale) and bias (shift)
        let weight = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[4])?;
        let bias = Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5], &[4])?;

        let output = layer_norm(&input, &[4], Some(&weight), Some(&bias), 1e-5)?;

        let output_data = output.to_vec()?;

        // After normalization (mean=0, var=1), then scale by 2 and shift by 0.5
        // The normalized values should be scaled and shifted
        for &val in output_data.iter() {
            // Values should be centered around 0.5 (bias) with scale factor 2
            assert!(val.abs() < 10.0); // Sanity check
        }

        Ok(())
    }

    #[test]
    fn test_layer_norm_multidimensional() -> Result<()> {
        // Test layer normalization on higher dimensional tensor (like transformer outputs)
        // Shape: [batch=2, seq_len=3, hidden_dim=4]
        let input = Tensor::from_vec(
            vec![
                // Batch 0
                1.0, 2.0, 3.0, 4.0, // Position 0
                2.0, 3.0, 4.0, 5.0, // Position 1
                3.0, 4.0, 5.0, 6.0, // Position 2
                // Batch 1
                4.0, 5.0, 6.0, 7.0, // Position 0
                5.0, 6.0, 7.0, 8.0, // Position 1
                6.0, 7.0, 8.0, 9.0, // Position 2
            ],
            &[2, 3, 4],
        )?;

        // Normalize over the hidden dimension
        let output = layer_norm(&input, &[4], None, None, 1e-5)?;

        assert_eq!(output.shape().dims(), &[2, 3, 4]);

        let output_data = output.to_vec()?;

        // Check that each position has mean ≈ 0 and variance ≈ 1
        for batch in 0..2 {
            for pos in 0..3 {
                let start_idx = (batch * 3 + pos) * 4;
                let slice = &output_data[start_idx..start_idx + 4];

                let mean: f32 = slice.iter().sum::<f32>() / 4.0;
                let var: f32 = slice.iter().map(|x| x.powi(2)).sum::<f32>() / 4.0;

                assert_relative_eq!(mean, 0.0, epsilon = 1e-4);
                assert_relative_eq!(var, 1.0, epsilon = 1e-4);
            }
        }

        Ok(())
    }

    #[test]
    fn test_layer_norm_single_value() -> Result<()> {
        // Edge case: normalize over a single dimension
        let input = Tensor::from_vec(vec![5.0, 10.0, 15.0], &[3, 1])?;

        let output = layer_norm(&input, &[1], None, None, 1e-5)?;

        let output_data = output.to_vec()?;

        // With a single value per sample, the normalized value should be 0
        // (after centering by the mean)
        for &val in output_data.iter() {
            assert_relative_eq!(val, 0.0, epsilon = 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_layer_norm_zeros_input() -> Result<()> {
        // Test with all zeros input
        let input = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[1, 4])?;

        let output = layer_norm(&input, &[4], None, None, 1e-5)?;

        let output_data = output.to_vec()?;

        // With zero variance, output should be zeros (0 - 0) / eps^0.5 ≈ 0
        for &val in output_data.iter() {
            assert!(val.abs() < 1e-2); // Should be very close to 0
        }

        Ok(())
    }

    #[test]
    fn test_layer_norm_consistency() -> Result<()> {
        // Test that layer_norm produces same results as layer_norm_enhanced
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])?;

        let output1 = layer_norm(&input, &[4], None, None, 1e-5)?;
        let output2 = layer_norm_enhanced(&input, &[4], None, None, 1e-5)?;

        let data1 = output1.to_vec()?;
        let data2 = output2.to_vec()?;

        assert_eq!(data1.len(), data2.len());
        for (v1, v2) in data1.iter().zip(data2.iter()) {
            assert_relative_eq!(v1, v2, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_group_norm_basic() -> Result<()> {
        // Test basic group normalization
        // Input shape: [batch=2, channels=4, height=3, width=3]
        // num_groups=2 means 2 channels per group
        let batch_size = 2;
        let channels = 4;
        let height = 3;
        let width = 3;
        let num_groups = 2;
        let channels_per_group = channels / num_groups;

        let input_data: Vec<f32> = (0..batch_size * channels * height * width)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let input = Tensor::from_vec(input_data, &[batch_size, channels, height, width])?;

        let output = group_norm(&input, num_groups, None, None, 1e-5)?;

        assert_eq!(
            output.shape().dims(),
            &[batch_size, channels, height, width]
        );

        // Verify normalization for first group of first sample
        let output_data = output.to_vec()?;

        // For group norm, each group should have mean ≈ 0 and variance ≈ 1
        // First group is channels 0-1, spatial dims 0-8 (9 spatial elements)
        let group_size = channels_per_group * height * width;
        let first_group_start = 0;
        let first_group_end = group_size;
        let first_group: Vec<f32> = output_data[first_group_start..first_group_end].to_vec();

        let mean: f32 = first_group.iter().sum::<f32>() / first_group.len() as f32;
        let variance: f32 =
            first_group.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / first_group.len() as f32;

        assert_relative_eq!(mean, 0.0, epsilon = 1e-4);
        assert_relative_eq!(variance, 1.0, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    fn test_group_norm_with_affine() -> Result<()> {
        // Test group normalization with learnable affine parameters
        let batch_size = 1;
        let channels = 6;
        let spatial_dim = 4;
        let num_groups = 3; // 2 channels per group

        let input_data: Vec<f32> = (0..batch_size * channels * spatial_dim)
            .map(|i| (i as f32) * 0.2 + 1.0)
            .collect();
        let input = Tensor::from_vec(input_data, &[batch_size, channels, spatial_dim])?;

        // Create weight (scale) and bias parameters
        let weight = Tensor::from_vec(vec![2.0, 1.5, 1.0, 0.5, 2.5, 3.0], &[channels])?;
        let bias = Tensor::from_vec(vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3], &[channels])?;

        let output = group_norm(&input, num_groups, Some(&weight), Some(&bias), 1e-5)?;

        assert_eq!(output.shape().dims(), &[batch_size, channels, spatial_dim]);

        // Output should be scaled and shifted according to weight and bias
        let output_data = output.to_vec()?;
        assert!(output_data.len() == batch_size * channels * spatial_dim);

        Ok(())
    }

    #[test]
    fn test_group_norm_invalid_groups() {
        // Test error handling when channels not divisible by num_groups
        let input = Tensor::from_vec(vec![1.0; 2 * 5 * 3 * 3], &[2, 5, 3, 3]).unwrap();

        let result = group_norm(&input, 3, None, None, 1e-5); // 5 channels, 3 groups
        assert!(result.is_err());
    }

    #[test]
    fn test_instance_norm_basic() -> Result<()> {
        // Test basic instance normalization
        // Input shape: [batch=2, channels=3, spatial=4]
        let batch_size = 2;
        let channels = 3;
        let spatial_size = 4;

        let input_data: Vec<f32> = (0..batch_size * channels * spatial_size)
            .map(|i| (i as f32) * 0.5 + 2.0)
            .collect();
        let input = Tensor::from_vec(input_data, &[batch_size, channels, spatial_size])?;

        let output = instance_norm(&input, None, None, 1e-5)?;

        assert_eq!(output.shape().dims(), &[batch_size, channels, spatial_size]);

        // For instance norm, each (batch, channel) instance should have mean ≈ 0, variance ≈ 1
        let output_data = output.to_vec()?;

        // Check first instance (batch=0, channel=0)
        let instance_start = 0;
        let instance_end = spatial_size;
        let first_instance: Vec<f32> = output_data[instance_start..instance_end].to_vec();

        let mean: f32 = first_instance.iter().sum::<f32>() / first_instance.len() as f32;
        let variance: f32 = first_instance
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / first_instance.len() as f32;

        assert_relative_eq!(mean, 0.0, epsilon = 1e-4);
        assert_relative_eq!(variance, 1.0, epsilon = 1e-3);

        Ok(())
    }

    #[test]
    fn test_instance_norm_with_affine() -> Result<()> {
        // Test instance normalization with learnable parameters
        let batch_size = 2;
        let channels = 4;
        let height = 3;
        let width = 3;

        let input_data: Vec<f32> = (0..batch_size * channels * height * width)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let input = Tensor::from_vec(input_data, &[batch_size, channels, height, width])?;

        let weight = Tensor::from_vec(vec![1.5, 2.0, 0.5, 3.0], &[channels])?;
        let bias = Tensor::from_vec(vec![0.2, -0.2, 0.1, -0.1], &[channels])?;

        let output = instance_norm(&input, Some(&weight), Some(&bias), 1e-5)?;

        assert_eq!(
            output.shape().dims(),
            &[batch_size, channels, height, width]
        );

        // Output should be affected by weight and bias
        let output_data = output.to_vec()?;
        assert!(output_data.len() == batch_size * channels * height * width);

        Ok(())
    }

    #[test]
    fn test_instance_norm_2d_image() -> Result<()> {
        // Test instance norm on 2D image data
        let batch_size = 1;
        let channels = 3; // RGB
        let height = 4;
        let width = 4;

        let input_data: Vec<f32> = (0..batch_size * channels * height * width)
            .map(|i| (i as f32) / 10.0)
            .collect();
        let input = Tensor::from_vec(input_data, &[batch_size, channels, height, width])?;

        let output = instance_norm(&input, None, None, 1e-5)?;

        assert_eq!(
            output.shape().dims(),
            &[batch_size, channels, height, width]
        );

        Ok(())
    }

    #[test]
    fn test_instance_norm_invalid_dims() {
        // Test error handling for insufficient dimensions
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = instance_norm(&input, None, None, 1e-5);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_norm_basic() -> Result<()> {
        // Test basic weight normalization
        // Weight norm: w = g * v / ||v||
        let weight = Tensor::from_vec(
            vec![3.0, 4.0], // L2 norm = 5.0
            &[2],
        )?;
        let g = Tensor::from_vec(vec![10.0], &[1])?;

        let output = weight_norm(&weight, &g, 0)?;

        let output_data = output.to_vec()?;

        // Expected: [3.0/5.0 * 10.0, 4.0/5.0 * 10.0] = [6.0, 8.0]
        assert_relative_eq!(output_data[0], 6.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 8.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_weight_norm_matrix() -> Result<()> {
        // Test weight normalization on a matrix
        let weight = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let g = Tensor::from_vec(vec![5.0], &[1])?;

        let output = weight_norm(&weight, &g, 0)?;

        assert_eq!(output.shape().dims(), &[2, 2]);

        // Check normalization occurred
        let output_data = output.to_vec()?;
        assert!(output_data.iter().all(|&x| x != 0.0));

        Ok(())
    }

    #[test]
    fn test_weight_norm_preserve_direction() -> Result<()> {
        // Weight normalization should preserve direction, only normalize magnitude
        let weight = Tensor::from_vec(vec![6.0, 8.0], &[2])?; // L2 norm = 10.0
        let g = Tensor::from_vec(vec![20.0], &[1])?;

        let output = weight_norm(&weight, &g, 0)?;
        let output_data = output.to_vec()?;

        // Normalized: [6/10, 8/10] * 20 = [12.0, 16.0]
        assert_relative_eq!(output_data[0], 12.0, epsilon = 1e-4);
        assert_relative_eq!(output_data[1], 16.0, epsilon = 1e-4);

        // Verify direction is preserved (ratio should be same as input)
        let input_ratio = 6.0 / 8.0;
        let output_ratio = output_data[0] / output_data[1];
        assert_relative_eq!(input_ratio, output_ratio, epsilon = 1e-5);

        Ok(())
    }
}
