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

    // Compute Lp norm: (sum(|x|^p))^(1/p)
    let norm = if (p - 2.0).abs() < 1e-7 {
        // Optimized path for L2 norm (most common case)
        let squared = input.pow_scalar(2.0)?;
        let sum = squared.sum_dim(&[dim as i32], true)?;
        sum.sqrt()?
    } else if (p - 1.0).abs() < 1e-7 {
        // Optimized path for L1 norm
        let abs_vals = input.abs()?;
        abs_vals.sum_dim(&[dim as i32], true)?
    } else if p.is_infinite() && p.is_sign_positive() {
        // L-infinity norm: max(|x|)
        let abs_vals = input.abs()?;
        abs_vals.max(Some(dim as usize), true)?
    } else {
        // General Lp norm
        let abs_vals = input.abs()?;
        let powered = abs_vals.pow_scalar(p as f32)?;
        let sum = powered.sum_dim(&[dim as i32], true)?;
        sum.pow_scalar((1.0 / p) as f32)?
    };

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
/// Normalizes weight by its spectral norm (largest singular value) using power iteration.
///
/// The spectral norm ||W||_2 is the largest singular value of the weight matrix W.
/// This is computed efficiently using the power iteration method:
///
/// 1. Start with random vector u
/// 2. Iterate: v = W^T u / ||W^T u||, u = W v / ||W v||
/// 3. Spectral norm ≈ u^T W v
///
/// # Arguments
/// * `weight` - Weight tensor (at least 2D)
/// * `u` - Optional initial vector for power iteration
/// * `n_power_iterations` - Number of power iterations (typically 1-5)
/// * `eps` - Small constant for numerical stability
///
/// # Returns
/// * Tuple of (normalized_weight, updated_u_vector)
pub fn spectral_norm(
    weight: &Tensor,
    u: Option<&Tensor>,
    n_power_iterations: usize,
    eps: f64,
) -> TorshResult<(Tensor, Tensor)> {
    let shape_obj = weight.shape();
    let shape = shape_obj.dims();

    if shape.len() < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Spectral norm requires at least 2D weight tensor",
            "spectral_norm",
        ));
    }

    // Reshape weight to 2D: [out_features, in_features]
    // For conv layers: [out_channels, in_channels * kernel_h * kernel_w]
    let out_features = shape[0];
    let in_features: usize = shape[1..].iter().product();
    let weight_mat = weight.view(&[out_features as i32, in_features as i32])?;

    // Initialize u vector if not provided
    let mut u_vec = if let Some(u_input) = u {
        u_input.clone()
    } else {
        // Initialize with random normal values
        use torsh_tensor::creation::randn;
        randn::<f32>(&[out_features])?
    };

    // Normalize u to unit length
    let u_norm = u_vec.pow_scalar(2.0)?.sum()?.sqrt()?;
    u_vec = u_vec.div_scalar(u_norm.item()? + eps as f32)?;

    // Power iteration to find dominant eigenvector
    for _ in 0..n_power_iterations {
        // v = W^T u
        let weight_t = weight_mat.t()?;
        let v = weight_t.matmul(&u_vec.view(&[out_features as i32, 1])?)?;
        let v = v.squeeze(1)?;

        // Normalize v
        let v_norm = v.pow_scalar(2.0)?.sum()?.sqrt()?;
        let v = v.div_scalar(v_norm.item()? + eps as f32)?;

        // u = W v
        let u = weight_mat.matmul(&v.view(&[in_features as i32, 1])?)?;
        u_vec = u.squeeze(1)?;

        // Normalize u
        let u_norm = u_vec.pow_scalar(2.0)?.sum()?.sqrt()?;
        u_vec = u_vec.div_scalar(u_norm.item()? + eps as f32)?;
    }

    // Compute spectral norm: sigma = u^T W v
    // First compute v = W^T u
    let weight_t = weight_mat.t()?;
    let v = weight_t.matmul(&u_vec.view(&[out_features as i32, 1])?)?;
    let v = v.squeeze(1)?;

    // Normalize v
    let v_norm = v.pow_scalar(2.0)?.sum()?.sqrt()?;
    let v = v.div_scalar(v_norm.item()? + eps as f32)?;

    // Compute sigma = u^T W v
    let wv = weight_mat.matmul(&v.view(&[in_features as i32, 1])?)?;
    let wv = wv.squeeze(1)?;

    // u^T (Wv) - dot product
    let u_wv = u_vec.mul(&wv)?.sum()?;
    let sigma = u_wv.item()?;

    // Normalize weight by spectral norm
    let normalized_weight = weight.div_scalar(sigma + eps as f32)?;

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

        // Test valid p=2 normalization (L2 norm)
        let result = normalize(&input, 2.0, 0, 1e-12, None);
        assert!(result.is_ok());

        // Test valid p=1 normalization (L1 norm)
        let result = normalize(&input, 1.0, 0, 1e-12, None);
        assert!(result.is_ok());

        // Test valid p=3 normalization (general p-norm)
        let result = normalize(&input, 3.0, 0, 1e-12, None);
        assert!(result.is_ok());

        // Test L-infinity norm
        let result = normalize(&input, f64::INFINITY, 0, 1e-12, None);
        assert!(result.is_ok());
    }
}
