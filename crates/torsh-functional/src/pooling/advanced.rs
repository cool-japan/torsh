//! Advanced and specialized pooling operations

use crate::utils::{function_context, validate_tensor_dims};
use scirs2_core::random::{Random, Rng};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// LP pooling (Lp norm pooling)
pub fn lp_pool1d(
    input: &Tensor,
    norm_type: f32,
    kernel_size: usize,
    stride: Option<usize>,
    ceil_mode: bool,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("lp_pool1d");
    validate_tensor_dims(input, 3, &context)?;

    if norm_type <= 0.0 {
        return Err(TorshError::config_error_with_context(
            "LP norm must be positive",
            &context,
        ));
    }

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let length = dims[2];

    let out_length = if ceil_mode {
        ((length - kernel_size) as f32 / stride as f32).ceil() as usize + 1
    } else {
        (length - kernel_size) / stride + 1
    };

    let mut output_data = vec![0.0f32; batch_size * channels * out_length];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for ol in 0..out_length {
                let out_idx = (b * channels + c) * out_length + ol;
                let mut sum = 0.0f32;

                for kl in 0..kernel_size {
                    let il = ol * stride + kl;
                    if il < length {
                        let in_idx = (b * channels + c) * length + il;
                        let val = input_data[in_idx].abs();
                        sum += val.powf(norm_type);
                    }
                }

                output_data[out_idx] = sum.powf(1.0 / norm_type);
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_length],
        input.device(),
    )
}

/// Stochastic pooling for regularization
#[allow(clippy::too_many_arguments)]
pub fn stochastic_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    training: bool,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("stochastic_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let out_height = (height + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_width = (width + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

    let output_size = batch_size * channels * out_height * out_width;
    let mut output_data = vec![0.0f32; output_size];
    let input_data = input.to_vec()?;

    if training {
        // Stochastic sampling during training
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;

                        // Collect values and compute probabilities
                        let mut values = Vec::new();
                        let mut total = 0.0f32;

                        for kh in 0..kernel_size.0 {
                            for kw in 0..kernel_size.1 {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;

                                if ih >= padding.0
                                    && ih < height + padding.0
                                    && iw >= padding.1
                                    && iw < width + padding.1
                                {
                                    let real_ih = ih - padding.0;
                                    let real_iw = iw - padding.1;

                                    if real_ih < height && real_iw < width {
                                        let in_idx = ((b * channels + c) * height + real_ih)
                                            * width
                                            + real_iw;
                                        let val = input_data[in_idx].max(0.0); // Use ReLU for positive probabilities
                                        values.push(val);
                                        total += val;
                                    }
                                }
                            }
                        }

                        if total > 0.0 && !values.is_empty() {
                            // Sample according to probabilities
                            let mut rng = Random::default();
                            let rand_val: f32 = rng.random();
                            let mut cumsum = 0.0f32;

                            for &val in &values {
                                cumsum += val / total;
                                if rand_val <= cumsum {
                                    output_data[out_idx] = val;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Expectation during inference (similar to average pooling with probabilities)
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;

                        let mut weighted_sum = 0.0f32;
                        let mut total_weight = 0.0f32;

                        for kh in 0..kernel_size.0 {
                            for kw in 0..kernel_size.1 {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;

                                if ih >= padding.0
                                    && ih < height + padding.0
                                    && iw >= padding.1
                                    && iw < width + padding.1
                                {
                                    let real_ih = ih - padding.0;
                                    let real_iw = iw - padding.1;

                                    if real_ih < height && real_iw < width {
                                        let in_idx = ((b * channels + c) * height + real_ih)
                                            * width
                                            + real_iw;
                                        let val = input_data[in_idx].max(0.0);
                                        weighted_sum += val * val; // Weight by value itself
                                        total_weight += val;
                                    }
                                }
                            }
                        }

                        output_data[out_idx] = if total_weight > 0.0 {
                            weighted_sum / total_weight
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_height, out_width],
        input.device(),
    )
}

/// Spatial pyramid pooling
pub fn spatial_pyramid_pool2d(input: &Tensor, pyramid_levels: &[usize]) -> TorshResult<Tensor> {
    let context = function_context("spatial_pyramid_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let total_bins: usize = pyramid_levels.iter().map(|&level| level * level).sum();
    let mut output_data = vec![0.0f32; batch_size * channels * total_bins];
    let input_data = input.to_vec()?;

    let mut bin_offset = 0;

    for &level in pyramid_levels {
        let bin_height = height / level;
        let bin_width = width / level;

        for b in 0..batch_size {
            for c in 0..channels {
                for py in 0..level {
                    for px in 0..level {
                        let bin_idx =
                            ((b * channels + c) * total_bins + bin_offset) + py * level + px;

                        let start_h = py * bin_height;
                        let end_h = if py == level - 1 {
                            height
                        } else {
                            (py + 1) * bin_height
                        };
                        let start_w = px * bin_width;
                        let end_w = if px == level - 1 {
                            width
                        } else {
                            (px + 1) * bin_width
                        };

                        let mut max_val = f32::NEG_INFINITY;

                        for h in start_h..end_h {
                            for w in start_w..end_w {
                                let in_idx = ((b * channels + c) * height + h) * width + w;
                                let val = input_data[in_idx];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }

                        output_data[bin_idx] = max_val;
                    }
                }
            }
        }

        bin_offset += level * level;
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, total_bins],
        input.device(),
    )
}

/// Learnable pooling with trainable parameters
pub fn learnable_pool2d(
    input: &Tensor,
    pooling_weights: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("learnable_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let out_height = (height + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_width = (width + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

    let output_size = batch_size * channels * out_height * out_width;
    let mut output_data = vec![0.0f32; output_size];
    let input_data = input.to_vec()?;
    let weights_data = pooling_weights.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    let mut weighted_sum = 0.0f32;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;

                            if ih >= padding.0
                                && ih < height + padding.0
                                && iw >= padding.1
                                && iw < width + padding.1
                            {
                                let real_ih = ih - padding.0;
                                let real_iw = iw - padding.1;

                                if real_ih < height && real_iw < width {
                                    let in_idx =
                                        ((b * channels + c) * height + real_ih) * width + real_iw;
                                    let weight_idx = kh * kernel_size.1 + kw;

                                    weighted_sum += input_data[in_idx] * weights_data[weight_idx];
                                }
                            }
                        }
                    }

                    output_data[out_idx] = weighted_sum;
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_height, out_width],
        input.device(),
    )
}

/// Fractional max pooling 2D
///
/// Implements fractional max pooling where the output size can be a fractional
/// multiple of the input size, providing data augmentation through randomized pooling regions.
pub fn fractional_max_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    output_size: Option<(usize, usize)>,
    output_ratio: Option<(f64, f64)>,
    return_indices: bool,
) -> TorshResult<(Tensor, Option<Tensor>)> {
    let context = function_context("fractional_max_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    // Determine output size based on provided parameters
    let (out_height, out_width) = if let Some(size) = output_size {
        size
    } else if let Some(ratio) = output_ratio {
        let out_h = (height as f64 * ratio.0) as usize;
        let out_w = (width as f64 * ratio.1) as usize;
        (out_h.max(1), out_w.max(1))
    } else {
        return Err(TorshError::config_error_with_context(
            "Either output_size or output_ratio must be specified",
            &context,
        ));
    };

    // Generate random but deterministic pooling sequences
    let mut rng = Random::default();

    // Calculate pooling regions using fractional strides
    let alpha_h = (height - kernel_size.0) as f64 / (out_height - 1).max(1) as f64;
    let alpha_w = (width - kernel_size.1) as f64 / (out_width - 1).max(1) as f64;

    let mut output_data = vec![f32::NEG_INFINITY; batch_size * channels * out_height * out_width];
    let mut indices_data = if return_indices {
        Some(vec![0usize; batch_size * channels * out_height * out_width])
    } else {
        None
    };

    let input_data = input.to_vec()?;

    // Generate random sequences for row and column pooling
    let mut row_sequence = vec![0usize; out_height + 1];
    let mut col_sequence = vec![0usize; out_width + 1];

    row_sequence[0] = 0;
    row_sequence[out_height] = height;
    for i in 1..out_height {
        let u: f64 = rng.random();
        row_sequence[i] = ((i as f64 - u) * alpha_h) as usize + kernel_size.0 / 2;
    }

    col_sequence[0] = 0;
    col_sequence[out_width] = width;
    for i in 1..out_width {
        let u: f64 = rng.random();
        col_sequence[i] = ((i as f64 - u) * alpha_w) as usize + kernel_size.1 / 2;
    }

    // Perform pooling
    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;

                    // Determine pooling region boundaries
                    let h_start = if oh == 0 { 0 } else { row_sequence[oh] };
                    let h_end = (h_start + kernel_size.0).min(height);
                    let w_start = if ow == 0 { 0 } else { col_sequence[ow] };
                    let w_end = (w_start + kernel_size.1).min(width);

                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;

                    // Find maximum in the pooling region
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let in_idx = ((b * channels + c) * height + ih) * width + iw;
                            if input_data[in_idx] > max_val {
                                max_val = input_data[in_idx];
                                max_idx = in_idx;
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

    let output_tensor = Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_height, out_width],
        input.device(),
    )?;

    let indices_tensor = if let Some(indices) = indices_data {
        Some(Tensor::from_data(
            indices.into_iter().map(|i| i as f32).collect(),
            vec![batch_size, channels, out_height, out_width],
            input.device(),
        )?)
    } else {
        None
    };

    Ok((output_tensor, indices_tensor))
}

/// LP pooling 2D (Lp norm pooling)
pub fn lp_pool2d(
    input: &Tensor,
    norm_type: f32,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    ceil_mode: bool,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("lp_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    if norm_type <= 0.0 {
        return Err(TorshError::config_error_with_context(
            "LP norm must be positive",
            &context,
        ));
    }

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let out_height = if ceil_mode {
        ((height - kernel_size.0) as f32 / stride.0 as f32).ceil() as usize + 1
    } else {
        (height - kernel_size.0) / stride.0 + 1
    };

    let out_width = if ceil_mode {
        ((width - kernel_size.1) as f32 / stride.1 as f32).ceil() as usize + 1
    } else {
        (width - kernel_size.1) / stride.1 + 1
    };

    let mut output_data = vec![0.0f32; batch_size * channels * out_height * out_width];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;

                            if ih < height && iw < width {
                                let in_idx = ((b * channels + c) * height + ih) * width + iw;
                                sum += input_data[in_idx].abs().powf(norm_type);
                                count += 1;
                            }
                        }
                    }

                    if count > 0 {
                        output_data[out_idx] = sum.powf(1.0 / norm_type);
                    }
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_height, out_width],
        input.device(),
    )
}
