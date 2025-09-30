//! Normalization functions for neural networks

use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{stats::StatMode, Tensor};

/// Batch normalization
///
/// Applies batch normalization over a batch of inputs
#[allow(clippy::too_many_arguments)]
pub fn batch_norm(
    input: &Tensor,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> TorshResult<Tensor> {
    // Input can be 2D (N, C), 3D (N, C, L), 4D (N, C, H, W) or 5D (N, C, D, H, W)
    let shape = input.shape().dims().to_vec();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Batch norm requires at least 2D input",
            "batch_norm",
        ));
    }

    let num_features = shape[1];

    // Calculate mean and variance
    let (mean, var) = if training {
        // Calculate batch statistics
        let axes: Vec<usize> = (0..ndim).filter(|&i| i != 1).collect();
        let mean = input.mean(Some(&axes), true)?;
        let var = input.var(Some(&axes), true, StatMode::Population)?;

        // Update running statistics if provided
        if let (Some(running_mean), Some(running_var)) = (running_mean, running_var) {
            // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            let _running_mean_update = running_mean
                .mul_scalar((1.0 - momentum) as f32)?
                .add_op(&mean.mul_scalar(momentum as f32)?)?;
            let _running_var_update = running_var
                .mul_scalar((1.0 - momentum) as f32)?
                .add_op(&var.mul_scalar(momentum as f32)?)?;

            // Note: In practice, these updates should be applied in-place
            // This would require mutable references which we don't have here
        }

        (mean, var)
    } else {
        // Use running statistics
        match (running_mean, running_var) {
            (Some(rm), Some(rv)) => (rm.clone(), rv.clone()),
            _ => {
                return Err(TorshError::invalid_argument_with_context(
                    "Running mean and var required for eval mode",
                    "batch_norm",
                ))
            }
        }
    };

    // Normalize: (x - mean) / sqrt(var + eps)
    let std = var.add_scalar(eps as f32)?.sqrt()?;
    let normalized = input.sub(&mean)?.div(&std)?;

    // Apply affine transformation if weight and bias are provided
    let output = match (weight, bias) {
        (Some(w), Some(b)) => {
            // Reshape weight and bias to match normalized dimensions
            let mut w_shape = vec![1; ndim];
            w_shape[1] = num_features;
            let w_reshaped = w.view(&w_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?;

            let mut b_shape = vec![1; ndim];
            b_shape[1] = num_features;
            let b_reshaped = b.view(&b_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?;

            normalized.mul_op(&w_reshaped)?.add_op(&b_reshaped)?
        }
        (Some(w), None) => {
            let mut w_shape = vec![1; ndim];
            w_shape[1] = num_features;
            let w_reshaped = w.view(&w_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
            normalized.mul_op(&w_reshaped)?
        }
        (None, Some(b)) => {
            let mut b_shape = vec![1; ndim];
            b_shape[1] = num_features;
            let b_reshaped = b.view(&b_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
            normalized.add_op(&b_reshaped)?
        }
        (None, None) => normalized,
    };

    Ok(output)
}

/// Layer normalization
///
/// Applies layer normalization over a mini-batch of inputs
pub fn layer_norm(
    input: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> TorshResult<Tensor> {
    // Normalize over the last len(normalized_shape) dimensions
    let ndim = input.shape().ndim();
    let norm_ndim = normalized_shape.len();

    if norm_ndim > ndim {
        return Err(TorshError::invalid_argument_with_context(
            "Normalized shape dimension count exceeds input dimensions",
            "layer_norm",
        ));
    }

    // Calculate axes to normalize over
    let axes: Vec<usize> = ((ndim - norm_ndim)..ndim).collect();

    // Calculate mean and variance
    let mean = input.mean(Some(&axes), true)?;
    let var = input.var(Some(&axes), true, StatMode::Population)?;

    // Normalize
    let std = var.add_scalar(eps as f32)?.sqrt()?;
    let normalized = input.sub(&mean)?.div(&std)?;

    // Apply affine transformation if provided
    let output = match (weight, bias) {
        (Some(w), Some(b)) => normalized.mul_op(w)?.add_op(b)?,
        (Some(w), None) => normalized.mul_op(w)?,
        (None, Some(b)) => normalized.add_op(b)?,
        (None, None) => normalized,
    };

    Ok(output)
}

/// Instance normalization
///
/// Applies instance normalization over a batch of inputs
#[allow(clippy::too_many_arguments)]
pub fn instance_norm(
    input: &Tensor,
    _running_mean: Option<&Tensor>,
    _running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    _use_input_stats: bool,
    _momentum: f64,
    eps: f64,
) -> TorshResult<Tensor> {
    // Instance norm normalizes each instance separately
    // For 4D input (N, C, H, W), normalize over (H, W) for each (N, C)
    let shape = input.shape().dims().to_vec();
    let ndim = shape.len();

    if ndim < 3 {
        return Err(TorshError::invalid_argument_with_context(
            "Instance norm requires at least 3D input",
            "instance_norm",
        ));
    }

    // Calculate axes to normalize over (spatial dimensions)
    let axes: Vec<usize> = (2..ndim).collect();

    // Calculate mean and variance
    let mean = input.mean(Some(&axes), true)?;
    let var = input.var(Some(&axes), true, StatMode::Population)?;

    // Normalize
    let std = var.add_scalar(eps as f32)?.sqrt()?;
    let normalized = input.sub(&mean)?.div(&std)?;

    // Apply affine transformation if provided
    let output = match (weight, bias) {
        (Some(w), Some(b)) => {
            // Reshape weight and bias for broadcasting
            let w = w.unsqueeze(0)?; // Add batch dimension
            let b = b.unsqueeze(0)?;
            normalized.mul_op(&w)?.add_op(&b)?
        }
        (Some(w), None) => {
            let w = w.unsqueeze(0)?;
            normalized.mul_op(&w)?
        }
        (None, Some(b)) => {
            let b = b.unsqueeze(0)?;
            normalized.add_op(&b)?
        }
        (None, None) => normalized,
    };

    Ok(output)
}

/// Group normalization
///
/// Divides channels into groups and normalizes within each group
pub fn group_norm(
    input: &Tensor,
    num_groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> TorshResult<Tensor> {
    let shape = input.shape().dims().to_vec();
    let ndim = shape.len();

    if ndim < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Group norm requires at least 2D input",
            "group_norm",
        ));
    }

    let batch_size = shape[0];
    let num_channels = shape[1];

    if num_channels % num_groups != 0 {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Number of channels {} must be divisible by num_groups {}",
                num_channels, num_groups
            ),
            "group_norm",
        ));
    }

    let channels_per_group = num_channels / num_groups;

    // Reshape to (N, G, C//G, *spatial)
    let mut new_shape = vec![batch_size, num_groups, channels_per_group];
    new_shape.extend_from_slice(&shape[2..]);

    let reshaped = input.reshape(&new_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?;

    // Normalize over channel and spatial dimensions within each group
    let axes: Vec<usize> = (2..new_shape.len()).collect();
    let mean = reshaped.mean(Some(&axes), true)?;
    let var = reshaped.var(Some(&axes), true, StatMode::Population)?;

    // Normalize
    let std = var.add_scalar(eps as f32)?.sqrt()?;
    let normalized = reshaped.sub(&mean)?.div(&std)?;

    // Reshape back to original dimensions
    let normalized = normalized.reshape(&shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?;

    // Apply affine transformation if provided
    let output = match (weight, bias) {
        (Some(w), Some(b)) => {
            let w = w.unsqueeze(0)?; // Add batch dimension
            let b = b.unsqueeze(0)?;
            normalized.mul_op(&w)?.add_op(&b)?
        }
        (Some(w), None) => {
            let w = w.unsqueeze(0)?;
            normalized.mul_op(&w)?
        }
        (None, Some(b)) => {
            let b = b.unsqueeze(0)?;
            normalized.add_op(&b)?
        }
        (None, None) => normalized,
    };

    Ok(output)
}

/// Local response normalization
///
/// Applies local response normalization over an input signal
pub fn local_response_norm(
    input: &Tensor,
    size: usize,
    alpha: f64,
    beta: f64,
    k: f64,
) -> TorshResult<Tensor> {
    // LRN normalizes over neighboring channels
    // For each position, normalize using channels [c-size/2, c+size/2]

    let shape_obj = input.shape();
    let shape = shape_obj.dims();
    if shape.len() < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Local response norm requires at least 2D input",
            "local_response_norm",
        ));
    }

    let _num_channels = shape[1];

    // Create padded tensor for easier computation
    let _padding = size / 2;

    // Compute squared values
    let squared = input.pow_scalar(2.0)?;

    // For each channel, sum over the neighboring channels
    // This is a simplified implementation - a full implementation would use
    // efficient convolution-like operations for the windowed sum

    // For now, return a placeholder implementation
    // that at least computes a basic normalization
    let sum_sq = squared.clone();

    // Compute denominator: (k + alpha/n * sum(x_i^2))^beta
    let n = size as f32;
    let denominator = sum_sq
        .mul_scalar((alpha / n as f64) as f32)?
        .add_scalar(k as f32)?
        .pow_scalar(beta as f32)?;

    // Normalize
    input.div(&denominator)
}

/// Normalize tensor using Lp norm
pub fn normalize(
    input: &Tensor,
    p: f64,
    dim: i64,
    eps: f64,
    out: Option<&mut Tensor>,
) -> TorshResult<Tensor> {
    // Validate p parameter
    if p <= 0.0 {
        return Err(TorshError::invalid_argument_with_context(
            &format!("normalize: p must be positive, got {}", p),
            "normalize",
        ));
    }

    // Validate dimension
    let ndim = input.ndim() as i64;
    let dim = if dim < 0 { ndim + dim } else { dim };

    if dim < 0 || dim >= ndim {
        return Err(TorshError::InvalidArgument(format!(
            "Dimension {} out of range for tensor with {} dimensions",
            dim, ndim
        )));
    }

    // For now, return a simple implementation for p=2 (L2 norm)
    if p == 2.0 {
        // Compute L2 norm: sqrt(sum(x^2))
        let squared = input.pow_scalar(2.0)?;
        let sum = squared.sum_dim(&[dim as i32], true)?;
        let norm = sum.sqrt()?;

        // Add epsilon to avoid division by zero
        let norm_eps = norm.add_scalar(eps as f32)?;

        // Normalize
        let normalized = input.div(&norm_eps)?;

        if let Some(_out_tensor) = out {
            // Copy to output tensor if provided
            // For now, we don't support in-place operations
            return Err(TorshError::UnsupportedOperation {
                op: "in-place normalize".to_string(),
                dtype: "tensor".to_string(),
            });
        }

        Ok(normalized)
    } else {
        // For other norms, return unsupported error
        Err(TorshError::UnsupportedOperation {
            op: format!("normalize with p={}", p),
            dtype: "tensor".to_string(),
        })
    }
}

/// Weight normalization
///
/// Decouples the magnitude and direction of weight vectors
pub fn weight_norm(weight: &Tensor, dim: i64) -> TorshResult<(Tensor, Tensor)> {
    // Compute the norm over the specified dimension
    let squared = weight.pow_scalar(2.0)?;
    let norm = squared.sum_dim(&[dim as i32], true)?.sqrt()?;

    // Normalized direction
    let direction = weight.div(&norm)?;

    // Squeeze the norm dimension for the magnitude output
    let magnitude = norm.squeeze(dim as i32)?;

    Ok((magnitude, direction))
}

/// Spectral normalization
///
/// Normalizes weight by its spectral norm (largest singular value)
pub fn spectral_norm(
    weight: &Tensor,
    u: Option<&Tensor>,
    _n_power_iterations: usize,
    eps: f64,
) -> TorshResult<(Tensor, Tensor)> {
    // Power iteration to estimate largest singular value
    // This is a placeholder implementation - full implementation would require:
    // 1. Reshape weight to 2D matrix
    // 2. Initialize u vector if not provided
    // 3. Perform power iterations
    // 4. Compute spectral norm
    // 5. Normalize weight

    let shape_obj = weight.shape();
    let shape = shape_obj.dims();
    if shape.len() < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Spectral norm requires at least 2D weight tensor",
            "spectral_norm",
        ));
    }

    // For now, return a simple normalization
    // In a full implementation, this would use power iteration
    let frobenius_norm = weight.pow_scalar(2.0)?.sum()?.sqrt()?;
    let normalized_weight = weight.div_scalar(frobenius_norm.item()? + eps as f32)?;

    // Return normalized weight and a dummy u vector
    let u_vec = if let Some(u_input) = u {
        u_input.clone()
    } else {
        // Create a dummy u vector for now
        torsh_tensor::creation::ones::<f32>(&[shape[0]])?
    };

    Ok((normalized_weight, u_vec))
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_normalize() {
        // Basic parameter validation tests
        let input = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        // Test invalid p value (must be positive)
        let result = normalize(&input, -1.0, 0, 1e-12, None);
        assert!(result.is_err());

        // Test invalid p value (zero)
        let result = normalize(&input, 0.0, 0, 1e-12, None);
        assert!(result.is_err());

        // Test valid p=2 normalization
        let result = normalize(&input, 2.0, 0, 1e-12, None);
        assert!(result.is_ok());

        // Test unsupported p values
        let result = normalize(&input, 1.0, 0, 1e-12, None);
        assert!(result.is_err()); // p=1 not implemented yet
    }
}
