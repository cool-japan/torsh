//! Dropout and regularization functions for neural networks

use crate::random_ops::rand;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::{creation::rand_like, Tensor};

/// Dropout
///
/// During training, randomly zeroes some elements of the input tensor
/// with probability p using samples from a Bernoulli distribution.
pub fn dropout(input: &Tensor, p: f64, training: bool, inplace: bool) -> TorshResult<Tensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    if !(0.0..=1.0).contains(&p) {
        return Err(TorshError::invalid_argument_with_context(
            &format!("Dropout probability must be between 0 and 1, got {}", p),
            "dropout",
        ));
    }

    // Generate random mask with probability (1-p) of keeping values
    let keep_prob = 1.0 - p;
    let random_tensor = rand_like(input)?;

    // Create binary mask where values < p are set to 0, others to 1/keep_prob
    let scale = 1.0 / keep_prob;
    let random_data = random_tensor.data()?;
    let mask_data: Vec<f32> = random_data
        .iter()
        .map(|&x| if x < p as f32 { 0.0 } else { scale as f32 })
        .collect();

    let mask = Tensor::from_data(mask_data, input.shape().dims().to_vec(), input.device())?;

    // Apply mask
    let output = if inplace {
        // TODO: Implement inplace operations when available
        input.clone().mul_op(&mask)?
    } else {
        input.mul_op(&mask)?
    };

    Ok(output)
}

/// Dropout1d
///
/// Randomly zero out entire channels (a channel is a 1D feature map).
/// Usually used after Conv1d modules.
pub fn dropout1d(input: &Tensor, p: f64, training: bool, inplace: bool) -> TorshResult<Tensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    let shape = input.shape().dims().to_vec();
    if shape.len() != 3 {
        return Err(TorshError::invalid_argument_with_context(
            &format!("Expected 3D input (N, C, L), got {}D", shape.len()),
            "dropout1d",
        ));
    }

    // Generate mask for channels (N, C, 1)
    let mask_shape = vec![shape[0], shape[1], 1];
    let random_tensor = rand(&mask_shape, Some(0.0), Some(1.0), None)?;

    // Create binary mask for channels
    let keep_prob = 1.0 - p;
    let scale = 1.0 / keep_prob;

    // Generate channel mask data
    let mask_data: Vec<f32> = random_tensor
        .to_vec()?
        .iter()
        .map(|&x| if x < p as f32 { 0.0 } else { scale as f32 })
        .collect();

    // Broadcast mask values to full shape
    let mut broadcast_data = vec![0.0f32; shape[0] * shape[1] * shape[2]];
    for n in 0..shape[0] {
        for c in 0..shape[1] {
            let mask_value = mask_data[n * shape[1] + c];
            for l in 0..shape[2] {
                let idx = (n * shape[1] + c) * shape[2] + l;
                broadcast_data[idx] = mask_value;
            }
        }
    }

    let mask = Tensor::from_data(broadcast_data, shape.clone(), input.device())?;

    // Apply mask
    let output = if inplace {
        input.clone().mul_op(&mask)?
    } else {
        input.mul_op(&mask)?
    };

    Ok(output)
}

/// Dropout2d
///
/// Randomly zero out entire channels (a channel is a 2D feature map).
/// Usually used after Conv2d modules.
pub fn dropout2d(input: &Tensor, p: f64, training: bool, inplace: bool) -> TorshResult<Tensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    let shape = input.shape().dims().to_vec();
    if shape.len() != 4 {
        return Err(TorshError::invalid_argument_with_context(
            &format!("Expected 4D input (N, C, H, W), got {}D", shape.len()),
            "dropout2d",
        ));
    }

    // Generate mask for channels (N, C, 1, 1)
    let mask_shape = vec![shape[0], shape[1], 1, 1];
    let random_tensor = rand(&mask_shape, Some(0.0), Some(1.0), None)?;

    // Create binary mask for channels
    let keep_prob = 1.0 - p;
    let scale = 1.0 / keep_prob;

    // Generate channel mask data
    let mask_data: Vec<f32> = random_tensor
        .to_vec()?
        .iter()
        .map(|&x| if x < p as f32 { 0.0 } else { scale as f32 })
        .collect();

    // Broadcast mask values to full shape
    let mut broadcast_data = vec![0.0f32; shape[0] * shape[1] * shape[2] * shape[3]];
    for n in 0..shape[0] {
        for c in 0..shape[1] {
            let mask_value = mask_data[n * shape[1] + c];
            for h in 0..shape[2] {
                for w in 0..shape[3] {
                    let idx = ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
                    broadcast_data[idx] = mask_value;
                }
            }
        }
    }

    let mask = Tensor::from_data(broadcast_data, shape.clone(), input.device())?;

    // Apply mask
    let output = if inplace {
        input.clone().mul_op(&mask)?
    } else {
        input.mul_op(&mask)?
    };

    Ok(output)
}

/// Dropout3d
///
/// Randomly zero out entire channels (a channel is a 3D feature map).
/// Usually used after Conv3d modules.
pub fn dropout3d(input: &Tensor, p: f64, training: bool, inplace: bool) -> TorshResult<Tensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    let shape = input.shape().dims().to_vec();
    if shape.len() != 5 {
        return Err(TorshError::invalid_argument_with_context(
            &format!("Expected 5D input (N, C, D, H, W), got {}D", shape.len()),
            "dropout3d",
        ));
    }

    // Generate mask for channels (N, C, 1, 1, 1)
    let mask_shape = vec![shape[0], shape[1], 1, 1, 1];
    let random_tensor = rand(&mask_shape, Some(0.0), Some(1.0), None)?;

    // Create binary mask for channels
    let keep_prob = 1.0 - p;
    let scale = 1.0 / keep_prob;

    // Generate channel mask data
    let mask_data: Vec<f32> = random_tensor
        .to_vec()?
        .iter()
        .map(|&x| if x < p as f32 { 0.0 } else { scale as f32 })
        .collect();

    // Broadcast mask values to full shape
    let mut broadcast_data = vec![0.0f32; shape[0] * shape[1] * shape[2] * shape[3] * shape[4]];
    for n in 0..shape[0] {
        for c in 0..shape[1] {
            let mask_value = mask_data[n * shape[1] + c];
            for d in 0..shape[2] {
                for h in 0..shape[3] {
                    for w in 0..shape[4] {
                        let idx =
                            (((n * shape[1] + c) * shape[2] + d) * shape[3] + h) * shape[4] + w;
                        broadcast_data[idx] = mask_value;
                    }
                }
            }
        }
    }

    let mask = Tensor::from_data(broadcast_data, shape.clone(), input.device())?;

    // Apply mask
    let output = if inplace {
        input.clone().mul_op(&mask)?
    } else {
        input.mul_op(&mask)?
    };

    Ok(output)
}

/// Alpha dropout
///
/// Applies alpha dropout to the input. Alpha Dropout is a type of Dropout
/// that maintains the self-normalizing property.
pub fn alpha_dropout(input: &Tensor, p: f64, training: bool, inplace: bool) -> TorshResult<Tensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    if !(0.0..=1.0).contains(&p) {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "Alpha dropout probability must be between 0 and 1, got {}",
                p
            ),
            "alpha_dropout",
        ));
    }

    // Alpha dropout parameters for SELU activation
    let alpha = 1.673_263_2_f32;
    let scale = 1.050_701_f32;
    let alpha_p = -alpha * scale;

    // Calculate affine transformation parameters
    let a = ((1.0f32 - p as f32) * (1.0f32 + p as f32 * alpha_p.powi(2))).sqrt();
    let b = -a * alpha_p * p as f32;

    // Generate mask (1.0 where random > p, 0.0 otherwise)
    let random_tensor = rand_like(input)?;
    let random_data = random_tensor.data()?;
    let mask_data: Vec<f32> = random_data
        .iter()
        .map(|&x| if x > p as f32 { 1.0 } else { 0.0 })
        .collect();

    let mask = Tensor::from_data(mask_data, input.shape().dims().to_vec(), input.device())?;

    // Apply alpha dropout

    if inplace {
        // x = x * mask + alpha_p * (1 - mask)
        // x = a * x + b
        let not_mask = mask.neg()?.add_scalar(1.0)?;
        let alpha_term = not_mask.mul_scalar(alpha_p)?;
        let x = input.clone().mul_op(&mask)?.add_op(&alpha_term)?;
        x.mul_scalar(a)?.add_scalar(b)
    } else {
        let not_mask = mask.neg()?.add_scalar(1.0)?;
        let alpha_term = not_mask.mul_scalar(alpha_p)?;
        let x = input.mul_op(&mask)?.add_op(&alpha_term)?;
        x.mul_scalar(a)?.add_scalar(b)
    }
}

/// Feature alpha dropout
///
/// Applies alpha dropout to entire channels.
pub fn feature_alpha_dropout(
    input: &Tensor,
    p: f64,
    training: bool,
    inplace: bool,
) -> TorshResult<Tensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    let shape = input.shape().dims().to_vec();
    if shape.len() < 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Feature alpha dropout requires at least 2D input",
            "feature_alpha_dropout",
        ));
    }

    // Alpha dropout parameters
    let alpha = 1.673_263_2_f32;
    let scale = 1.050_701_f32;
    let alpha_p = -alpha * scale;

    // Calculate affine transformation parameters
    let a = ((1.0f32 - p as f32) * (1.0f32 + p as f32 * alpha_p.powi(2))).sqrt();
    let b = -a * alpha_p * p as f32;

    // Generate mask for channels
    let mut mask_shape = vec![shape[0], shape[1]];
    mask_shape.extend(vec![1; shape.len() - 2]);
    let random_tensor = rand(&mask_shape, Some(0.0), Some(1.0), None)?;

    // Create mask (1.0 where random > p, 0.0 otherwise)
    let mask_data: Vec<f32> = random_tensor
        .to_vec()?
        .iter()
        .map(|&x| if x > p as f32 { 1.0 } else { 0.0 })
        .collect();

    // Broadcast mask to input shape
    let total_size: usize = shape.iter().product();
    let mut broadcast_data = vec![0.0f32; total_size];

    // Calculate strides for broadcasting
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Broadcast the mask
    for i in 0..total_size {
        let mut idx = i;
        let n = idx / strides[0];
        idx %= strides[0];
        let c = idx / strides[1];

        let mask_idx = n * shape[1] + c;
        broadcast_data[i] = mask_data[mask_idx];
    }

    let mask = Tensor::from_data(broadcast_data, shape.clone(), input.device())?;

    // Apply feature alpha dropout

    if inplace {
        let not_mask = mask.neg()?.add_scalar(1.0)?;
        let alpha_term = not_mask.mul_scalar(alpha_p)?;
        let x = input.clone().mul_op(&mask)?.add_op(&alpha_term)?;
        x.mul_scalar(a)?.add_scalar(b)
    } else {
        let not_mask = mask.neg()?.add_scalar(1.0)?;
        let alpha_term = not_mask.mul_scalar(alpha_p)?;
        let x = input.mul_op(&mask)?.add_op(&alpha_term)?;
        x.mul_scalar(a)?.add_scalar(b)
    }
}

/// Fractional max pooling 2d with dropout
///
/// Applies fractional max pooling with random sampling for regularization
pub fn fractional_max_pool2d_with_indices(
    input: &Tensor,
    _kernel_size: (usize, usize),
    _output_size: Option<(usize, usize)>,
    _output_ratio: Option<(f64, f64)>,
    _return_indices: bool,
) -> TorshResult<(Tensor, Option<Tensor>)> {
    // TODO: Implement fractional max pooling
    // For now, return the input unchanged as a placeholder
    Ok((input.clone(), None))
}

/// Gaussian dropout
///
/// Multiplicative noise where each element is multiplied by a sample from
/// a Gaussian distribution with mean 1 and standard deviation sqrt(p/(1-p))
pub fn gaussian_dropout(
    input: &Tensor,
    p: f64,
    training: bool,
    inplace: bool,
) -> TorshResult<Tensor> {
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    if !(0.0..1.0).contains(&p) {
        return Err(TorshError::invalid_argument_with_context(
            &format!("Gaussian dropout probability must be in [0, 1), got {}", p),
            "gaussian_dropout",
        ));
    }

    // Standard deviation of the multiplicative noise
    let std = (p / (1.0 - p)).sqrt();

    // Generate Gaussian noise with mean 0 and std 1, then scale and shift to mean 1
    let randn = torsh_tensor::creation::randn_like(input);
    let noise = randn?.mul_scalar(std as f32)?.add_scalar(1.0)?;

    // Apply multiplicative noise

    if inplace {
        input.clone().mul_op(&noise)
    } else {
        input.mul_op(&noise)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::ones;

    #[test]
    fn test_dropout_probability_validation() -> TorshResult<()> {
        // Test that invalid probabilities are rejected
        let input = ones::<f32>(&[2, 3])?;

        assert!(dropout(&input, -0.1, true, false).is_err());
        assert!(dropout(&input, 1.1, true, false).is_err());

        // Test that p=0 returns input unchanged (doesn't execute dropout logic)
        assert!(dropout(&input, 0.0, true, false).is_ok());

        // Test that training=false returns input unchanged (doesn't execute dropout logic)
        assert!(dropout(&input, 0.5, false, false).is_ok());
        Ok(())
    }
}
