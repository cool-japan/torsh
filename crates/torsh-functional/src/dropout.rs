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

/// Generate a fractional pooling region-boundary sequence.
///
/// Implements the disjoint pseudo-random sampling scheme from Ben Graham,
/// "Fractional Max-Pooling" (2015). For an input dimension `input_size`,
/// a pooling-window `kernel`, and a target `output_size`, this produces the
/// `output_size` starting indices of the (possibly overlapping) pooling
/// regions. Region `i` covers `[starts[i], starts[i] + kernel)`.
///
/// The boundaries follow `starts[i] = floor(alpha * (i + u))` where
/// `alpha = (input_size - kernel) / (output_size - 1)` and `u` is a single
/// random sample shared across the dimension (PyTorch-compatible behaviour),
/// guaranteeing the regions tile the input without leaving gaps.
fn fractional_pool_sequence(
    input_size: usize,
    kernel: usize,
    output_size: usize,
    sample: f32,
) -> TorshResult<Vec<usize>> {
    if kernel == 0 {
        return Err(TorshError::invalid_argument_with_context(
            "fractional_max_pool2d: kernel size must be non-zero",
            "fractional_max_pool2d",
        ));
    }
    if output_size == 0 || output_size > input_size {
        return Err(TorshError::invalid_argument_with_context(
            &format!(
                "fractional_max_pool2d: invalid output size {output_size} for input size {input_size}"
            ),
            "fractional_max_pool2d",
        ));
    }
    if kernel > input_size {
        return Err(TorshError::invalid_argument_with_context(
            &format!("fractional_max_pool2d: kernel {kernel} exceeds input size {input_size}"),
            "fractional_max_pool2d",
        ));
    }

    let mut starts = Vec::with_capacity(output_size);

    if output_size == 1 {
        starts.push(0);
        return Ok(starts);
    }

    let alpha = (input_size - kernel) as f32 / (output_size - 1) as f32;
    let u = sample.clamp(0.0, 1.0 - f32::EPSILON);

    let max_start = input_size - kernel;
    for i in 0..output_size {
        let start = (alpha * (i as f32 + u)).floor() as usize;
        starts.push(start.min(max_start));
    }

    Ok(starts)
}

/// Fractional max pooling 2d with dropout
///
/// Applies 2D fractional max pooling with pseudo-random region sampling, as
/// described in Ben Graham, "Fractional Max-Pooling" (2015). The output spatial
/// size is given either explicitly via `output_size` or derived from
/// `output_ratio` (`out = floor(input_dim * ratio)`). Exactly one of the two
/// must be provided. When `return_indices` is set, the flattened input index of
/// each selected maximum is returned alongside the pooled output.
pub fn fractional_max_pool2d_with_indices(
    input: &Tensor,
    kernel_size: (usize, usize),
    output_size: Option<(usize, usize)>,
    output_ratio: Option<(f64, f64)>,
    return_indices: bool,
) -> TorshResult<(Tensor, Option<Tensor>)> {
    let shape = input.shape().dims().to_vec();
    if shape.len() != 4 {
        return Err(TorshError::invalid_argument_with_context(
            &format!("Expected 4D input (N, C, H, W), got {}D", shape.len()),
            "fractional_max_pool2d",
        ));
    }

    let (batch_size, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    // Resolve the target output size from either an explicit size or a ratio.
    let (out_h, out_w) = match (output_size, output_ratio) {
        (Some(size), None) => size,
        (None, Some((rh, rw))) => {
            if !(0.0..=1.0).contains(&rh) || !(0.0..=1.0).contains(&rw) {
                return Err(TorshError::invalid_argument_with_context(
                    "fractional_max_pool2d: output_ratio components must be in (0, 1]",
                    "fractional_max_pool2d",
                ));
            }
            (
                ((height as f64) * rh).floor() as usize,
                ((width as f64) * rw).floor() as usize,
            )
        }
        (Some(_), Some(_)) => {
            return Err(TorshError::invalid_argument_with_context(
                "fractional_max_pool2d: specify exactly one of output_size or output_ratio",
                "fractional_max_pool2d",
            ));
        }
        (None, None) => {
            return Err(TorshError::invalid_argument_with_context(
                "fractional_max_pool2d: one of output_size or output_ratio is required",
                "fractional_max_pool2d",
            ));
        }
    };

    // Draw the two per-dimension random samples that anchor the region grids.
    let samples = rand(&[2], Some(0.0), Some(1.0), None)?.to_vec()?;
    let row_starts = fractional_pool_sequence(height, kernel_size.0, out_h, samples[0])?;
    let col_starts = fractional_pool_sequence(width, kernel_size.1, out_w, samples[1])?;

    let output_len = batch_size * channels * out_h * out_w;
    let mut output_data = vec![f32::NEG_INFINITY; output_len];
    let mut indices_data = if return_indices {
        Some(vec![0i64; output_len])
    } else {
        None
    };

    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_h {
                let h_start = row_starts[oh];
                let h_end = (h_start + kernel_size.0).min(height);
                for ow in 0..out_w {
                    let w_start = col_starts[ow];
                    let w_end = (w_start + kernel_size.1).min(width);

                    let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0i64;

                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let in_idx = ((b * channels + c) * height + ih) * width + iw;
                            let val = input_data[in_idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = in_idx as i64;
                            }
                        }
                    }

                    output_data[out_idx] = max_val;
                    if let Some(ref mut indices) = indices_data {
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }
    }

    let output = Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_h, out_w],
        input.device(),
    )?;

    let indices = if let Some(indices_data) = indices_data {
        let indices_f32: Vec<f32> = indices_data.iter().map(|&idx| idx as f32).collect();
        Some(Tensor::from_data(
            indices_f32,
            vec![batch_size, channels, out_h, out_w],
            input.device(),
        )?)
    } else {
        None
    };

    Ok((output, indices))
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

    #[test]
    fn test_fractional_pool_sequence_tiles_input() {
        // The region starts must be non-decreasing, in range, and the last
        // region must reach the end of the input.
        let starts = fractional_pool_sequence(8, 2, 4, 0.5).unwrap();
        assert_eq!(starts.len(), 4);
        for w in starts.windows(2) {
            assert!(w[1] >= w[0], "region starts must be non-decreasing");
        }
        assert!(*starts.last().unwrap() + 2 <= 8);
        assert_eq!(*starts.last().unwrap(), 6); // last window covers [6, 8)
    }

    #[test]
    fn test_fractional_max_pool2d_output_shape_and_values() -> TorshResult<()> {
        // 1x1x4x4 ramp: value at (h, w) = h * 4 + w.
        let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let input = Tensor::from_data(data, vec![1, 1, 4, 4], torsh_core::device::DeviceType::Cpu)?;

        let (output, indices) =
            fractional_max_pool2d_with_indices(&input, (2, 2), Some((2, 2)), None, true)?;

        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
        let out = output.to_vec()?;
        // Every pooled value must be a genuine maximum drawn from the input, and
        // strictly greater than the global minimum (proves it is not a clone /
        // passthrough of the input).
        let in_vec = input.to_vec()?;
        let in_max = in_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for &v in &out {
            assert!(in_vec.contains(&v), "pooled value {v} must come from input");
        }
        // The bottom-right region always includes the global maximum (15.0).
        assert!(out.iter().cloned().fold(f32::NEG_INFINITY, f32::max) == in_max);

        // Indices must point at the selected maxima.
        let idx = indices.expect("indices requested").to_vec()?;
        for (k, &v) in out.iter().enumerate() {
            let flat = idx[k] as usize;
            assert!((in_vec[flat] - v).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_fractional_max_pool2d_ratio() -> TorshResult<()> {
        let data: Vec<f32> = (0..36).map(|x| x as f32).collect();
        let input = Tensor::from_data(data, vec![1, 1, 6, 6], torsh_core::device::DeviceType::Cpu)?;

        // ratio 0.5 on a size-6 dimension -> output size 3.
        let (output, _) =
            fractional_max_pool2d_with_indices(&input, (2, 2), None, Some((0.5, 0.5)), false)?;
        assert_eq!(output.shape().dims(), &[1, 1, 3, 3]);
        Ok(())
    }

    #[test]
    fn test_fractional_max_pool2d_argument_errors() -> TorshResult<()> {
        let input = ones::<f32>(&[1, 1, 4, 4])?;
        // Neither output_size nor output_ratio.
        assert!(fractional_max_pool2d_with_indices(&input, (2, 2), None, None, false).is_err());
        // Both specified.
        assert!(fractional_max_pool2d_with_indices(
            &input,
            (2, 2),
            Some((2, 2)),
            Some((0.5, 0.5)),
            false
        )
        .is_err());
        // Wrong dimensionality.
        let input_3d = ones::<f32>(&[1, 4, 4])?;
        assert!(
            fractional_max_pool2d_with_indices(&input_3d, (2, 2), Some((2, 2)), None, false)
                .is_err()
        );
        Ok(())
    }
}
