//! Pooling operations for neural networks
//!
//! This module provides comprehensive pooling operations including max pooling,
//! average pooling, global pooling, and adaptive pooling variants.

use torsh_core::error::Result;
use torsh_tensor::Tensor;

// =============================================================================
// MAX POOLING OPERATIONS
// =============================================================================

/// 1D max pooling function
///
/// Applies max pooling over a 1D input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
/// * `kernel_size` - Size of the pooling window
/// * `stride` - Stride of the pooling window (default: kernel_size)
/// * `padding` - Zero padding to add to both sides (default: 0)
/// * `dilation` - Spacing between kernel elements (default: 1)
///
/// # Returns
/// Output tensor of shape [N, C, L_out]
pub fn max_pool1d(
    input: &Tensor,
    kernel_size: usize,
    stride: Option<usize>,
    padding: Option<usize>,
    dilation: Option<usize>,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let stride = stride.unwrap_or(kernel_size);
    let padding = padding.unwrap_or(0);
    let dilation = dilation.unwrap_or(1);

    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];

    // Calculate output length
    let out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    let mut output_data = vec![f32::NEG_INFINITY; batch * channels * out_length];
    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for out_l in 0..out_length {
                let l_start = out_l * stride;
                let mut max_val = f32::NEG_INFINITY;

                for k in 0..kernel_size {
                    let l = l_start + k * dilation;

                    if l >= padding && l < length + padding {
                        // This is actual data (not padding)
                        let input_l = l - padding;
                        let input_idx = b * (channels * length) + c * length + input_l;
                        max_val = max_val.max(input_data[input_idx]);
                    }
                    // Padding positions are implicitly -infinity (won't affect max)
                }

                let output_idx = b * (channels * out_length) + c * out_length + out_l;
                output_data[output_idx] = max_val;
            }
        }
    }

    let output_shape = vec![batch, channels, out_length];
    Tensor::from_vec(output_data, &output_shape)
}

/// 2D max pooling function
///
/// Applies max pooling over a 2D input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `kernel_size` - Size of the pooling window (height, width)
/// * `stride` - Stride of the pooling window (default: kernel_size)
/// * `padding` - Zero padding to add to all sides (default: (0, 0))
/// * `dilation` - Spacing between kernel elements (default: (1, 1))
///
/// # Returns
/// Output tensor of shape [N, C, H_out, W_out]
pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    dilation: Option<(usize, usize)>,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let stride = stride.unwrap_or(kernel_size);
    let padding = padding.unwrap_or((0, 0));
    let dilation = dilation.unwrap_or((1, 1));

    let [batch, channels, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    // Calculate output dimensions
    let out_height = (height + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1;
    let out_width = (width + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1;

    let mut output_data = vec![f32::NEG_INFINITY; batch * channels * out_height * out_width];
    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for out_h in 0..out_height {
                for out_w in 0..out_width {
                    let h_start = out_h * stride.0;
                    let w_start = out_w * stride.1;
                    let mut max_val = f32::NEG_INFINITY;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let h = h_start + kh * dilation.0;
                            let w = w_start + kw * dilation.1;

                            if h >= padding.0
                                && h < height + padding.0
                                && w >= padding.1
                                && w < width + padding.1
                            {
                                // This is actual data (not padding)
                                let input_h = h - padding.0;
                                let input_w = w - padding.1;
                                let input_idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + input_h * width
                                    + input_w;
                                max_val = max_val.max(input_data[input_idx]);
                            }
                            // Padding positions are implicitly -infinity (won't affect max)
                        }
                    }

                    let output_idx = b * (channels * out_height * out_width)
                        + c * (out_height * out_width)
                        + out_h * out_width
                        + out_w;
                    output_data[output_idx] = max_val;
                }
            }
        }
    }

    let output_shape = vec![batch, channels, out_height, out_width];
    Tensor::from_vec(output_data, &output_shape)
}

/// 3D max pooling function
///
/// Applies max pooling over a 3D input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
/// * `kernel_size` - Size of the pooling window (depth, height, width)
/// * `stride` - Stride of the pooling window (default: kernel_size)
/// * `padding` - Zero padding to add to all sides (default: (0, 0, 0))
/// * `dilation` - Spacing between kernel elements (default: (1, 1, 1))
///
/// # Returns
/// Output tensor of shape [N, C, D_out, H_out, W_out]
pub fn max_pool3d(
    input: &Tensor,
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: Option<(usize, usize, usize)>,
    dilation: Option<(usize, usize, usize)>,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 5 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let stride = stride.unwrap_or(kernel_size);
    let padding = padding.unwrap_or((0, 0, 0));
    let dilation = dilation.unwrap_or((1, 1, 1));

    let [batch, channels, depth, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    ];

    // Calculate output dimensions
    let out_depth = (depth + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1;
    let out_height = (height + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1;
    let out_width = (width + 2 * padding.2 - dilation.2 * (kernel_size.2 - 1) - 1) / stride.2 + 1;

    let mut output_data =
        vec![f32::NEG_INFINITY; batch * channels * out_depth * out_height * out_width];
    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for out_d in 0..out_depth {
                for out_h in 0..out_height {
                    for out_w in 0..out_width {
                        let d_start = out_d * stride.0;
                        let h_start = out_h * stride.1;
                        let w_start = out_w * stride.2;
                        let mut max_val = f32::NEG_INFINITY;

                        for kd in 0..kernel_size.0 {
                            for kh in 0..kernel_size.1 {
                                for kw in 0..kernel_size.2 {
                                    let d = d_start + kd * dilation.0;
                                    let h = h_start + kh * dilation.1;
                                    let w = w_start + kw * dilation.2;

                                    if d >= padding.0
                                        && d < depth + padding.0
                                        && h >= padding.1
                                        && h < height + padding.1
                                        && w >= padding.2
                                        && w < width + padding.2
                                    {
                                        // This is actual data (not padding)
                                        let input_d = d - padding.0;
                                        let input_h = h - padding.1;
                                        let input_w = w - padding.2;
                                        let input_idx = b * (channels * depth * height * width)
                                            + c * (depth * height * width)
                                            + input_d * (height * width)
                                            + input_h * width
                                            + input_w;
                                        max_val = max_val.max(input_data[input_idx]);
                                    }
                                    // Padding positions are implicitly -infinity (won't affect max)
                                }
                            }
                        }

                        let output_idx = b * (channels * out_depth * out_height * out_width)
                            + c * (out_depth * out_height * out_width)
                            + out_d * (out_height * out_width)
                            + out_h * out_width
                            + out_w;
                        output_data[output_idx] = max_val;
                    }
                }
            }
        }
    }

    let output_shape = vec![batch, channels, out_depth, out_height, out_width];
    Tensor::from_vec(output_data, &output_shape)
}

// =============================================================================
// AVERAGE POOLING OPERATIONS
// =============================================================================

/// 1D average pooling function
///
/// Applies average pooling over a 1D input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
/// * `kernel_size` - Size of the pooling window
/// * `stride` - Stride of the pooling window (default: kernel_size)
/// * `padding` - Zero padding to add to both sides (default: 0)
/// * `count_include_pad` - If True, include padding zeros in average calculation
///
/// # Returns
/// Output tensor of shape [N, C, L_out] where L_out = (L + 2*padding - kernel_size) / stride + 1
pub fn avg_pool1d(
    input: &Tensor,
    kernel_size: usize,
    stride: Option<usize>,
    padding: Option<usize>,
    count_include_pad: bool,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let stride = stride.unwrap_or(kernel_size);
    let padding = padding.unwrap_or(0);

    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];

    // Calculate output length
    let out_length = (length + 2 * padding - kernel_size) / stride + 1;

    let mut output_data = vec![0.0f32; batch * channels * out_length];
    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for out_l in 0..out_length {
                let l_start = out_l * stride;
                let mut sum = 0.0f32;
                let mut count = 0;

                for k in 0..kernel_size {
                    let l = l_start + k;

                    if l < padding || l >= length + padding {
                        // This is a padding position
                        if count_include_pad {
                            count += 1;
                            // sum += 0.0 (padding value)
                        }
                    } else {
                        // This is an actual data position
                        let input_l = l - padding;
                        let input_idx = b * (channels * length) + c * length + input_l;
                        sum += input_data[input_idx];
                        count += 1;
                    }
                }

                let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                let output_idx = b * (channels * out_length) + c * out_length + out_l;
                output_data[output_idx] = avg;
            }
        }
    }

    let output_shape = vec![batch, channels, out_length];
    Tensor::from_vec(output_data, &output_shape)
}

/// 2D average pooling function
///
/// Applies average pooling over a 2D input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `kernel_size` - Size of the pooling window (height, width)
/// * `stride` - Stride of the pooling window (default: kernel_size)
/// * `padding` - Zero padding to add to all sides (default: (0, 0))
/// * `count_include_pad` - If True, include padding zeros in average calculation
///
/// # Returns
/// Output tensor of shape [N, C, H_out, W_out]
pub fn avg_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    count_include_pad: bool,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();
    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let stride = stride.unwrap_or(kernel_size);
    let padding = padding.unwrap_or((0, 0));

    let [batch, channels, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    // Calculate output dimensions
    let out_height = (height + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_width = (width + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

    let mut output_data = vec![0.0f32; batch * channels * out_height * out_width];
    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for out_h in 0..out_height {
                for out_w in 0..out_width {
                    let h_start = out_h * stride.0;
                    let w_start = out_w * stride.1;
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let h = h_start + kh;
                            let w = w_start + kw;

                            if h < padding.0
                                || h >= height + padding.0
                                || w < padding.1
                                || w >= width + padding.1
                            {
                                // This is a padding position
                                if count_include_pad {
                                    count += 1;
                                    // sum += 0.0 (padding value)
                                }
                            } else {
                                // This is an actual data position
                                let input_h = h - padding.0;
                                let input_w = w - padding.1;
                                let input_idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + input_h * width
                                    + input_w;
                                sum += input_data[input_idx];
                                count += 1;
                            }
                        }
                    }

                    let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                    let output_idx = b * (channels * out_height * out_width)
                        + c * (out_height * out_width)
                        + out_h * out_width
                        + out_w;
                    output_data[output_idx] = avg;
                }
            }
        }
    }

    let output_shape = vec![batch, channels, out_height, out_width];
    Tensor::from_vec(output_data, &output_shape)
}

/// 3D average pooling function
///
/// Applies average pooling over a 3D input tensor.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
/// * `kernel_size` - Size of the pooling window (depth, height, width)
/// * `stride` - Stride of the pooling window (default: kernel_size)
/// * `padding` - Zero padding to add to all sides (default: (0, 0, 0))
/// * `count_include_pad` - If True, include padding zeros in average calculation
///
/// # Returns
/// Output tensor of shape [N, C, D_out, H_out, W_out]
pub fn avg_pool3d(
    input: &Tensor,
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: Option<(usize, usize, usize)>,
    count_include_pad: bool,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 5 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let stride = stride.unwrap_or(kernel_size);
    let padding = padding.unwrap_or((0, 0, 0));

    let [batch, channels, depth, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    ];

    // Calculate output dimensions
    let out_depth = (depth + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_height = (height + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
    let out_width = (width + 2 * padding.2 - kernel_size.2) / stride.2 + 1;

    let mut output_data = vec![0.0f32; batch * channels * out_depth * out_height * out_width];
    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for out_d in 0..out_depth {
                for out_h in 0..out_height {
                    for out_w in 0..out_width {
                        let d_start = out_d * stride.0;
                        let h_start = out_h * stride.1;
                        let w_start = out_w * stride.2;
                        let mut sum = 0.0f32;
                        let mut count = 0;

                        for kd in 0..kernel_size.0 {
                            for kh in 0..kernel_size.1 {
                                for kw in 0..kernel_size.2 {
                                    let d = d_start + kd;
                                    let h = h_start + kh;
                                    let w = w_start + kw;

                                    if d < padding.0
                                        || d >= depth + padding.0
                                        || h < padding.1
                                        || h >= height + padding.1
                                        || w < padding.2
                                        || w >= width + padding.2
                                    {
                                        // This is a padding position
                                        if count_include_pad {
                                            count += 1;
                                            // sum += 0.0 (padding value)
                                        }
                                    } else {
                                        // This is an actual data position
                                        let input_d = d - padding.0;
                                        let input_h = h - padding.1;
                                        let input_w = w - padding.2;
                                        let input_idx = b * (channels * depth * height * width)
                                            + c * (depth * height * width)
                                            + input_d * (height * width)
                                            + input_h * width
                                            + input_w;
                                        sum += input_data[input_idx];
                                        count += 1;
                                    }
                                }
                            }
                        }

                        let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                        let output_idx = b * (channels * out_depth * out_height * out_width)
                            + c * (out_depth * out_height * out_width)
                            + out_d * (out_height * out_width)
                            + out_h * out_width
                            + out_w;
                        output_data[output_idx] = avg;
                    }
                }
            }
        }
    }

    let output_shape = vec![batch, channels, out_depth, out_height, out_width];
    Tensor::from_vec(output_data, &output_shape)
}

// =============================================================================
// ADAPTIVE POOLING OPERATIONS
// =============================================================================

/// 1D adaptive average pooling function
///
/// Adaptively pools the input to a specified output size using average pooling.
/// Each output element is computed as the average of a region of the input.
///
/// # Arguments
/// * `input` - Input tensor of shape [..., L] where L is the length
/// * `output_size` - Target output length
///
/// # Returns
/// Output tensor of shape [..., output_size]
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4])?;
/// let pooled = adaptive_avg_pool1d(&input, 2)?; // Shape: [1, 1, 2]
/// ```
pub fn adaptive_avg_pool1d(input: &Tensor, output_size: usize) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() < 1 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0], // At least 1D
            got: input_shape.to_vec(),
        });
    }

    let input_length = input_shape[input_shape.len() - 1];

    // If output size matches input, just return clone
    if output_size == input_length {
        return Ok(input.clone());
    }

    // Create output shape
    let mut output_shape = input_shape.to_vec();
    let length_idx = output_shape.len() - 1;
    output_shape[length_idx] = output_size;

    let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
    let mut output_data = vec![0.0f32; batch_size * output_size];
    let input_data = input.to_vec()?;

    // For each batch element
    for b in 0..batch_size {
        // For each output position
        for out_l in 0..output_size {
            // Calculate the range of input positions this output pools from
            let start_idx = (out_l * input_length) / output_size;
            let end_idx = ((out_l + 1) * input_length) / output_size;

            let mut sum = 0.0f32;
            let mut count = 0;

            for in_l in start_idx..end_idx {
                let input_idx = b * input_length + in_l;
                sum += input_data[input_idx];
                count += 1;
            }

            let avg = if count > 0 { sum / count as f32 } else { 0.0 };
            let output_idx = b * output_size + out_l;
            output_data[output_idx] = avg;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// 2D adaptive average pooling function
///
/// Adaptively pools 2D input to specified output size using average pooling.
///
/// # Arguments
/// * `input` - Input tensor of shape [..., H, W]
/// * `output_size` - Target output (height, width). None means keep that dimension.
///
/// # Returns
/// Output tensor of shape [..., output_height, output_width]
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0; 16], vec![1, 1, 4, 4])?;
/// let pooled = adaptive_avg_pool2d(&input, (Some(2), Some(2)))?; // [1, 1, 2, 2]
/// ```
pub fn adaptive_avg_pool2d(
    input: &Tensor,
    output_size: (Option<usize>, Option<usize>),
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() < 2 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0], // At least 2D
            got: input_shape.to_vec(),
        });
    }

    let input_height = input_shape[input_shape.len() - 2];
    let input_width = input_shape[input_shape.len() - 1];

    let output_height = output_size.0.unwrap_or(input_height);
    let output_width = output_size.1.unwrap_or(input_width);

    // If output size matches input, just return clone
    if output_height == input_height && output_width == input_width {
        return Ok(input.clone());
    }

    // Create output shape
    let mut output_shape = input_shape.to_vec();
    let height_idx = output_shape.len() - 2;
    let width_idx = output_shape.len() - 1;
    output_shape[height_idx] = output_height;
    output_shape[width_idx] = output_width;

    let batch_size: usize = input_shape[..input_shape.len() - 2].iter().product();
    let mut output_data = vec![0.0f32; batch_size * output_height * output_width];
    let input_data = input.to_vec()?;

    // For each batch element
    for b in 0..batch_size {
        // For each output position
        for out_h in 0..output_height {
            for out_w in 0..output_width {
                // Calculate the range of input positions this output pools from
                let h_start = (out_h * input_height) / output_height;
                let h_end = ((out_h + 1) * input_height) / output_height;
                let w_start = (out_w * input_width) / output_width;
                let w_end = ((out_w + 1) * input_width) / output_width;

                let mut sum = 0.0f32;
                let mut count = 0;

                for in_h in h_start..h_end {
                    for in_w in w_start..w_end {
                        let input_idx =
                            b * (input_height * input_width) + in_h * input_width + in_w;
                        sum += input_data[input_idx];
                        count += 1;
                    }
                }

                let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                let output_idx = b * (output_height * output_width) + out_h * output_width + out_w;
                output_data[output_idx] = avg;
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// 3D adaptive average pooling function
///
/// Adapts the pooling kernel and stride to produce the desired output size.
/// For each output position, computes the average of values in the corresponding
/// input region.
///
/// # Arguments
/// * `input` - Input tensor of shape [..., D, H, W]
/// * `output_size` - Target output size (depth, height, width), None means keep dimension
///
/// # Returns
/// Output tensor of shape [..., output_depth, output_height, output_width]
pub fn adaptive_avg_pool3d(
    input: &Tensor,
    output_size: (Option<usize>, Option<usize>, Option<usize>),
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() < 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let input_depth = input_shape[input_shape.len() - 3];
    let input_height = input_shape[input_shape.len() - 2];
    let input_width = input_shape[input_shape.len() - 1];

    let output_depth = output_size.0.unwrap_or(input_depth);
    let output_height = output_size.1.unwrap_or(input_height);
    let output_width = output_size.2.unwrap_or(input_width);

    if output_depth == input_depth && output_height == input_height && output_width == input_width {
        return Ok(input.clone());
    }

    let mut output_shape = input_shape.to_vec();
    let depth_idx = output_shape.len() - 3;
    let height_idx = output_shape.len() - 2;
    let width_idx = output_shape.len() - 1;
    output_shape[depth_idx] = output_depth;
    output_shape[height_idx] = output_height;
    output_shape[width_idx] = output_width;

    let batch_size: usize = input_shape[..input_shape.len() - 3].iter().product();
    let mut output_data = vec![0.0f32; batch_size * output_depth * output_height * output_width];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for out_d in 0..output_depth {
            for out_h in 0..output_height {
                for out_w in 0..output_width {
                    // Calculate input range for this output position
                    let d_start = (out_d * input_depth) / output_depth;
                    let d_end = ((out_d + 1) * input_depth) / output_depth;
                    let h_start = (out_h * input_height) / output_height;
                    let h_end = ((out_h + 1) * input_height) / output_height;
                    let w_start = (out_w * input_width) / output_width;
                    let w_end = ((out_w + 1) * input_width) / output_width;

                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for in_d in d_start..d_end {
                        for in_h in h_start..h_end {
                            for in_w in w_start..w_end {
                                let input_idx = b * (input_depth * input_height * input_width)
                                    + in_d * (input_height * input_width)
                                    + in_h * input_width
                                    + in_w;
                                sum += input_data[input_idx];
                                count += 1;
                            }
                        }
                    }

                    let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                    let output_idx = b * (output_depth * output_height * output_width)
                        + out_d * (output_height * output_width)
                        + out_h * output_width
                        + out_w;
                    output_data[output_idx] = avg;
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// 1D adaptive max pooling function
///
/// Adapts the pooling kernel and stride to produce the desired output size.
/// For each output position, computes the maximum of values in the corresponding
/// input region.
///
/// # Arguments
/// * `input` - Input tensor of shape [..., L]
/// * `output_size` - Target output length
///
/// # Returns
/// Output tensor of shape [..., output_size]
pub fn adaptive_max_pool1d(input: &Tensor, output_size: usize) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() < 1 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0],
            got: input_shape.to_vec(),
        });
    }

    let input_length = input_shape[input_shape.len() - 1];

    if output_size == input_length {
        return Ok(input.clone());
    }

    let mut output_shape = input_shape.to_vec();
    let length_idx = output_shape.len() - 1;
    output_shape[length_idx] = output_size;

    let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();
    let mut output_data = vec![f32::NEG_INFINITY; batch_size * output_size];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for out_l in 0..output_size {
            // Calculate input range for this output position
            let start_idx = (out_l * input_length) / output_size;
            let end_idx = ((out_l + 1) * input_length) / output_size;

            let mut max_val = f32::NEG_INFINITY;

            for in_l in start_idx..end_idx {
                let input_idx = b * input_length + in_l;
                max_val = max_val.max(input_data[input_idx]);
            }

            let output_idx = b * output_size + out_l;
            output_data[output_idx] = max_val;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// 2D adaptive max pooling function
///
/// Adapts the pooling kernel and stride to produce the desired output size.
/// For each output position, computes the maximum of values in the corresponding
/// input region.
///
/// # Arguments
/// * `input` - Input tensor of shape [..., H, W]
/// * `output_size` - Target output size (height, width), None means keep dimension
///
/// # Returns
/// Output tensor of shape [..., output_height, output_width]
pub fn adaptive_max_pool2d(
    input: &Tensor,
    output_size: (Option<usize>, Option<usize>),
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() < 2 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0],
            got: input_shape.to_vec(),
        });
    }

    let input_height = input_shape[input_shape.len() - 2];
    let input_width = input_shape[input_shape.len() - 1];

    let output_height = output_size.0.unwrap_or(input_height);
    let output_width = output_size.1.unwrap_or(input_width);

    if output_height == input_height && output_width == input_width {
        return Ok(input.clone());
    }

    let mut output_shape = input_shape.to_vec();
    let height_idx = output_shape.len() - 2;
    let width_idx = output_shape.len() - 1;
    output_shape[height_idx] = output_height;
    output_shape[width_idx] = output_width;

    let batch_size: usize = input_shape[..input_shape.len() - 2].iter().product();
    let mut output_data = vec![f32::NEG_INFINITY; batch_size * output_height * output_width];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for out_h in 0..output_height {
            for out_w in 0..output_width {
                // Calculate input range for this output position
                let h_start = (out_h * input_height) / output_height;
                let h_end = ((out_h + 1) * input_height) / output_height;
                let w_start = (out_w * input_width) / output_width;
                let w_end = ((out_w + 1) * input_width) / output_width;

                let mut max_val = f32::NEG_INFINITY;

                for in_h in h_start..h_end {
                    for in_w in w_start..w_end {
                        let input_idx =
                            b * (input_height * input_width) + in_h * input_width + in_w;
                        max_val = max_val.max(input_data[input_idx]);
                    }
                }

                let output_idx = b * (output_height * output_width) + out_h * output_width + out_w;
                output_data[output_idx] = max_val;
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// 3D adaptive max pooling function
///
/// Adapts the pooling kernel and stride to produce the desired output size.
/// For each output position, computes the maximum of values in the corresponding
/// input region.
///
/// # Arguments
/// * `input` - Input tensor of shape [..., D, H, W]
/// * `output_size` - Target output size (depth, height, width), None means keep dimension
///
/// # Returns
/// Output tensor of shape [..., output_depth, output_height, output_width]
pub fn adaptive_max_pool3d(
    input: &Tensor,
    output_size: (Option<usize>, Option<usize>, Option<usize>),
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() < 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let input_depth = input_shape[input_shape.len() - 3];
    let input_height = input_shape[input_shape.len() - 2];
    let input_width = input_shape[input_shape.len() - 1];

    let output_depth = output_size.0.unwrap_or(input_depth);
    let output_height = output_size.1.unwrap_or(input_height);
    let output_width = output_size.2.unwrap_or(input_width);

    if output_depth == input_depth && output_height == input_height && output_width == input_width {
        return Ok(input.clone());
    }

    let mut output_shape = input_shape.to_vec();
    let depth_idx = output_shape.len() - 3;
    let height_idx = output_shape.len() - 2;
    let width_idx = output_shape.len() - 1;
    output_shape[depth_idx] = output_depth;
    output_shape[height_idx] = output_height;
    output_shape[width_idx] = output_width;

    let batch_size: usize = input_shape[..input_shape.len() - 3].iter().product();
    let mut output_data =
        vec![f32::NEG_INFINITY; batch_size * output_depth * output_height * output_width];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for out_d in 0..output_depth {
            for out_h in 0..output_height {
                for out_w in 0..output_width {
                    // Calculate input range for this output position
                    let d_start = (out_d * input_depth) / output_depth;
                    let d_end = ((out_d + 1) * input_depth) / output_depth;
                    let h_start = (out_h * input_height) / output_height;
                    let h_end = ((out_h + 1) * input_height) / output_height;
                    let w_start = (out_w * input_width) / output_width;
                    let w_end = ((out_w + 1) * input_width) / output_width;

                    let mut max_val = f32::NEG_INFINITY;

                    for in_d in d_start..d_end {
                        for in_h in h_start..h_end {
                            for in_w in w_start..w_end {
                                let input_idx = b * (input_depth * input_height * input_width)
                                    + in_d * (input_height * input_width)
                                    + in_h * input_width
                                    + in_w;
                                max_val = max_val.max(input_data[input_idx]);
                            }
                        }
                    }

                    let output_idx = b * (output_depth * output_height * output_width)
                        + out_d * (output_height * output_width)
                        + out_h * output_width
                        + out_w;
                    output_data[output_idx] = max_val;
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

// =============================================================================
// GLOBAL POOLING OPERATIONS
// =============================================================================

/// Global average pooling for 1D
///
/// Computes the average of each channel across the entire length dimension.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
///
/// # Returns
/// Output tensor of shape [N, C, 1] containing the average value for each channel
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])?;
/// let pooled = global_avg_pool1d(&input)?; // Shape: [1, 2, 1]
/// ```
pub fn global_avg_pool1d(input: &Tensor) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];

    // Calculate average across the length dimension for each channel
    let output_shape = vec![batch, channels, 1];
    let mut output_data = vec![0.0f32; batch * channels];

    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            let mut sum = 0.0f32;
            for l in 0..length {
                let idx = b * (channels * length) + c * length + l;
                sum += input_data[idx];
            }
            let avg = sum / length as f32;
            let output_idx = b * channels + c;
            output_data[output_idx] = avg;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Global average pooling for 2D
///
/// Computes the average of each channel across the entire spatial dimensions (H, W).
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
///
/// # Returns
/// Output tensor of shape [N, C, 1, 1] containing the average value for each channel
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2])?;
/// let pooled = global_avg_pool2d(&input)?; // Shape: [1, 1, 1, 1], value: 2.5
/// ```
pub fn global_avg_pool2d(input: &Tensor) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let [batch, channels, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    // Calculate average across spatial dimensions (H x W) for each channel
    let output_shape = vec![batch, channels, 1, 1];
    let mut output_data = vec![0.0f32; batch * channels];

    let input_data = input.to_vec()?;
    let spatial_size = (height * width) as f32;

    for b in 0..batch {
        for c in 0..channels {
            let mut sum = 0.0f32;
            for h in 0..height {
                for w in 0..width {
                    let idx =
                        b * (channels * height * width) + c * (height * width) + h * width + w;
                    sum += input_data[idx];
                }
            }
            let avg = sum / spatial_size;
            let output_idx = b * channels + c;
            output_data[output_idx] = avg;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Global average pooling for 3D
///
/// Computes the average of each channel across the entire spatial dimensions (D, H, W).
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
///
/// # Returns
/// Output tensor of shape [N, C, 1, 1, 1] containing the average value for each channel
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0; 8], vec![1, 1, 2, 2, 2])?;
/// let pooled = global_avg_pool3d(&input)?; // Shape: [1, 1, 1, 1, 1]
/// ```
pub fn global_avg_pool3d(input: &Tensor) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 5 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let [batch, channels, depth, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    ];

    // Calculate average across spatial dimensions (D x H x W) for each channel
    let output_shape = vec![batch, channels, 1, 1, 1];
    let mut output_data = vec![0.0f32; batch * channels];

    let input_data = input.to_vec()?;
    let spatial_size = (depth * height * width) as f32;

    for b in 0..batch {
        for c in 0..channels {
            let mut sum = 0.0f32;
            for d in 0..depth {
                for h in 0..height {
                    for w in 0..width {
                        let idx = b * (channels * depth * height * width)
                            + c * (depth * height * width)
                            + d * (height * width)
                            + h * width
                            + w;
                        sum += input_data[idx];
                    }
                }
            }
            let avg = sum / spatial_size;
            let output_idx = b * channels + c;
            output_data[output_idx] = avg;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Global max pooling for 1D
///
/// Computes the maximum value of each channel across the entire length dimension.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
///
/// # Returns
/// Output tensor of shape [N, C, 1] containing the maximum value for each channel
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 5.0, 3.0, 2.0], vec![1, 2, 2])?;
/// let pooled = global_max_pool1d(&input)?; // Shape: [1, 2, 1], values: [5.0, 3.0]
/// ```
pub fn global_max_pool1d(input: &Tensor) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];

    // Calculate maximum across the length dimension for each channel
    let output_shape = vec![batch, channels, 1];
    let mut output_data = vec![f32::NEG_INFINITY; batch * channels];

    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            let mut max_val = f32::NEG_INFINITY;
            for l in 0..length {
                let idx = b * (channels * length) + c * length + l;
                max_val = max_val.max(input_data[idx]);
            }
            let output_idx = b * channels + c;
            output_data[output_idx] = max_val;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Global max pooling for 2D
///
/// Computes the maximum value of each channel across the entire spatial dimensions (H, W).
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
///
/// # Returns
/// Output tensor of shape [N, C, 1, 1] containing the maximum value for each channel
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 5.0, 3.0, 2.0], vec![1, 1, 2, 2])?;
/// let pooled = global_max_pool2d(&input)?; // Shape: [1, 1, 1, 1], value: 5.0
/// ```
pub fn global_max_pool2d(input: &Tensor) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let [batch, channels, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    // Calculate maximum across spatial dimensions (H x W) for each channel
    let output_shape = vec![batch, channels, 1, 1];
    let mut output_data = vec![f32::NEG_INFINITY; batch * channels];

    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            let mut max_val = f32::NEG_INFINITY;
            for h in 0..height {
                for w in 0..width {
                    let idx =
                        b * (channels * height * width) + c * (height * width) + h * width + w;
                    max_val = max_val.max(input_data[idx]);
                }
            }
            let output_idx = b * channels + c;
            output_data[output_idx] = max_val;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Global max pooling for 3D
///
/// Computes the maximum value of each channel across the entire spatial dimensions (D, H, W).
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
///
/// # Returns
/// Output tensor of shape [N, C, 1, 1, 1] containing the maximum value for each channel
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 5.0, 3.0, 2.0, 8.0, 1.0, 4.0, 6.0], vec![1, 1, 2, 2, 2])?;
/// let pooled = global_max_pool3d(&input)?; // Shape: [1, 1, 1, 1, 1], value: 8.0
/// ```
pub fn global_max_pool3d(input: &Tensor) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 5 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let [batch, channels, depth, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    ];

    // Calculate maximum across spatial dimensions (D x H x W) for each channel
    let output_shape = vec![batch, channels, 1, 1, 1];
    let mut output_data = vec![f32::NEG_INFINITY; batch * channels];

    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..depth {
                for h in 0..height {
                    for w in 0..width {
                        let idx = b * (channels * depth * height * width)
                            + c * (depth * height * width)
                            + d * (height * width)
                            + h * width
                            + w;
                        max_val = max_val.max(input_data[idx]);
                    }
                }
            }
            let output_idx = b * channels + c;
            output_data[output_idx] = max_val;
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

// =============================================================================
// PADDING OPERATIONS
// =============================================================================

/// General padding function
///
/// Pads a tensor according to the specified mode.
///
/// # Arguments
/// * `input` - Input tensor
/// * `padding` - Padding specification as slice of (left/right) or (top/bottom, left/right) tuples
/// * `mode` - Padding mode: "constant", "reflect", "replicate", or "zero"
/// * `value` - Fill value for constant padding (default: 0.0)
///
/// # Returns
/// Padded tensor
///
/// # Supported Modes
/// - "constant": Pads with a constant value (specified by `value`)
/// - "zero": Pads with zeros (equivalent to constant with value=0.0)
/// - "reflect": Pads by reflecting values at boundaries (padding must be < input size)
/// - "replicate": Pads by replicating edge values
///
/// # Example
/// ```ignore
/// // Zero padding for 2D tensor
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2])?;
/// let padded = pad(&input, &[(1, 1), (1, 1)], "zero", None)?;
///
/// // Constant padding with custom value
/// let padded = pad(&input, &[(1, 1), (1, 1)], "constant", Some(5.0))?;
/// ```
pub fn pad(
    input: &Tensor,
    padding: &[(usize, usize)],
    mode: &str,
    value: Option<f32>,
) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    // Determine tensor dimensionality and validate padding specification
    match (input_shape.len(), padding.len()) {
        // 1D tensor padding (shape: [N, C, L])
        (3, 1) => {
            let (pad_left, pad_right) = padding[0];
            match mode {
                "reflect" => reflection_pad1d(input, (pad_left, pad_right)),
                "replicate" => replication_pad1d(input, (pad_left, pad_right)),
                "constant" | "zero" => {
                    // Implement constant padding for 1D
                    let fill_value = if mode == "zero" {
                        0.0
                    } else {
                        value.unwrap_or(0.0)
                    };
                    let [batch, channels, length] =
                        [input_shape[0], input_shape[1], input_shape[2]];
                    let new_length = length + pad_left + pad_right;

                    let output_shape = vec![batch, channels, new_length];
                    let mut output_data = vec![fill_value; batch * channels * new_length];
                    let input_data = input.to_vec()?;

                    // Copy input data to the appropriate position
                    for b in 0..batch {
                        for c in 0..channels {
                            for l in 0..length {
                                let input_idx = b * (channels * length) + c * length + l;
                                let output_idx =
                                    b * (channels * new_length) + c * new_length + (l + pad_left);
                                output_data[output_idx] = input_data[input_idx];
                            }
                        }
                    }

                    Tensor::from_vec(output_data, &output_shape)
                }
                _ => Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Unsupported padding mode: {}",
                    mode
                ))),
            }
        }
        // 2D tensor padding (shape: [N, C, H, W])
        (4, 2) => {
            let (pad_left, pad_right) = padding[1];
            let (pad_top, pad_bottom) = padding[0];
            match mode {
                "reflect" => reflection_pad2d(input, (pad_left, pad_right, pad_top, pad_bottom)),
                "replicate" => replication_pad2d(input, (pad_left, pad_right, pad_top, pad_bottom)),
                "constant" | "zero" => {
                    let fill_value = if mode == "zero" {
                        0.0
                    } else {
                        value.unwrap_or(0.0)
                    };

                    // Use zero_pad2d if padding with zeros, otherwise implement constant padding
                    if fill_value == 0.0 {
                        zero_pad2d(input, (pad_left, pad_right, pad_top, pad_bottom))
                    } else {
                        // Implement constant padding with custom value
                        let [batch, channels, height, width] = [
                            input_shape[0],
                            input_shape[1],
                            input_shape[2],
                            input_shape[3],
                        ];

                        let new_height = height + pad_top + pad_bottom;
                        let new_width = width + pad_left + pad_right;

                        let output_shape = vec![batch, channels, new_height, new_width];
                        let mut output_data =
                            vec![fill_value; batch * channels * new_height * new_width];
                        let input_data = input.to_vec()?;

                        // Copy input data to the appropriate position
                        for b in 0..batch {
                            for c in 0..channels {
                                for h in 0..height {
                                    for w in 0..width {
                                        let input_idx = b * (channels * height * width)
                                            + c * (height * width)
                                            + h * width
                                            + w;

                                        let output_h = h + pad_top;
                                        let output_w = w + pad_left;
                                        let output_idx = b * (channels * new_height * new_width)
                                            + c * (new_height * new_width)
                                            + output_h * new_width
                                            + output_w;

                                        output_data[output_idx] = input_data[input_idx];
                                    }
                                }
                            }
                        }

                        Tensor::from_vec(output_data, &output_shape)
                    }
                }
                _ => Err(torsh_core::error::TorshError::InvalidArgument(format!(
                    "Unsupported padding mode: {}",
                    mode
                ))),
            }
        }
        _ => Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Unsupported combination: tensor shape length {} with padding length {}. \
                 Expected 3D tensor with 1 padding pair or 4D tensor with 2 padding pairs.",
            input_shape.len(),
            padding.len()
        ))),
    }
}

/// Reflection padding 1D
///
/// Pads a 3D tensor by reflecting values at boundaries along the length dimension.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
/// * `padding` - (pad_left, pad_right)
///
/// # Returns
/// Padded tensor of shape [N, C, L+pad_left+pad_right]
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![1, 1, 3])?;
/// let padded = reflection_pad1d(&input, (2, 2))?; // [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
/// ```
pub fn reflection_pad1d(input: &Tensor, padding: (usize, usize)) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let (pad_left, pad_right) = padding;
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];

    // Validate padding doesn't exceed input size for reflection
    if pad_left >= length || pad_right >= length {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Padding size ({}, {}) cannot be >= input length ({}) for reflection padding",
            pad_left, pad_right, length
        )));
    }

    let new_length = length + pad_left + pad_right;

    // Create output tensor
    let output_shape = vec![batch, channels, new_length];
    let mut output_data = vec![0.0f32; batch * channels * new_length];

    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for l in 0..new_length {
                let input_idx = b * (channels * length) + c * length;
                let output_idx = b * (channels * new_length) + c * new_length + l;

                // Determine which value to reflect
                let src_l = if l < pad_left {
                    // Reflect from the left: pad_left - 1 - l
                    pad_left - l
                } else if l >= pad_left + length {
                    // Reflect from the right
                    let offset = l - (pad_left + length);
                    length - 2 - offset
                } else {
                    // Use original value
                    l - pad_left
                };

                output_data[output_idx] = input_data[input_idx + src_l];
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Reflection padding 2D
///
/// Pads a 4D tensor by reflecting values at boundaries along height and width dimensions.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `padding` - (pad_left, pad_right, pad_top, pad_bottom)
///
/// # Returns
/// Padded tensor of shape [N, C, H+pad_top+pad_bottom, W+pad_left+pad_right]
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2])?;
/// let padded = reflection_pad2d(&input, (1, 1, 1, 1))?;
/// ```
pub fn reflection_pad2d(input: &Tensor, padding: (usize, usize, usize, usize)) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let (pad_left, pad_right, pad_top, pad_bottom) = padding;
    let [batch, channels, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    // Validate padding doesn't exceed input size for reflection
    if pad_left >= width || pad_right >= width {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Horizontal padding ({}, {}) cannot be >= input width ({}) for reflection padding",
            pad_left, pad_right, width
        )));
    }
    if pad_top >= height || pad_bottom >= height {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Vertical padding ({}, {}) cannot be >= input height ({}) for reflection padding",
            pad_top, pad_bottom, height
        )));
    }

    let new_height = height + pad_top + pad_bottom;
    let new_width = width + pad_left + pad_right;

    // Create output tensor
    let output_shape = vec![batch, channels, new_height, new_width];
    let mut output_data = vec![0.0f32; batch * channels * new_height * new_width];

    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for h in 0..new_height {
                for w in 0..new_width {
                    // Determine which values to reflect
                    let src_h = if h < pad_top {
                        // Reflect from the top
                        pad_top - h
                    } else if h >= pad_top + height {
                        // Reflect from the bottom
                        let offset = h - (pad_top + height);
                        height - 2 - offset
                    } else {
                        // Use original value
                        h - pad_top
                    };

                    let src_w = if w < pad_left {
                        // Reflect from the left
                        pad_left - w
                    } else if w >= pad_left + width {
                        // Reflect from the right
                        let offset = w - (pad_left + width);
                        width - 2 - offset
                    } else {
                        // Use original value
                        w - pad_left
                    };

                    let input_idx = b * (channels * height * width)
                        + c * (height * width)
                        + src_h * width
                        + src_w;

                    let output_idx = b * (channels * new_height * new_width)
                        + c * (new_height * new_width)
                        + h * new_width
                        + w;

                    output_data[output_idx] = input_data[input_idx];
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Replication padding 1D
///
/// Pads a 3D tensor by replicating edge values along the length dimension.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
/// * `padding` - (pad_left, pad_right)
///
/// # Returns
/// Padded tensor of shape [N, C, L+pad_left+pad_right]
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![1, 1, 3])?;
/// let padded = replication_pad1d(&input, (1, 1))?; // [1.0, 1.0, 2.0, 3.0, 3.0]
/// ```
pub fn replication_pad1d(input: &Tensor, padding: (usize, usize)) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 3 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let (pad_left, pad_right) = padding;
    let [batch, channels, length] = [input_shape[0], input_shape[1], input_shape[2]];

    let new_length = length + pad_left + pad_right;

    // Create output tensor
    let output_shape = vec![batch, channels, new_length];
    let mut output_data = vec![0.0f32; batch * channels * new_length];

    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for l in 0..new_length {
                let input_idx = b * (channels * length) + c * length;

                let output_idx = b * (channels * new_length) + c * new_length + l;

                // Determine which value to replicate
                let src_l = if l < pad_left {
                    0 // Replicate leftmost value
                } else if l >= pad_left + length {
                    length - 1 // Replicate rightmost value
                } else {
                    l - pad_left // Use original value
                };

                output_data[output_idx] = input_data[input_idx + src_l];
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Replication padding 2D
///
/// Pads a 4D tensor by replicating edge values along height and width dimensions.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `padding` - (pad_left, pad_right, pad_top, pad_bottom)
///
/// # Returns
/// Padded tensor of shape [N, C, H+pad_top+pad_bottom, W+pad_left+pad_right]
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2])?;
/// let padded = replication_pad2d(&input, (1, 1, 1, 1))?;
/// ```
pub fn replication_pad2d(input: &Tensor, padding: (usize, usize, usize, usize)) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let (pad_left, pad_right, pad_top, pad_bottom) = padding;
    let [batch, channels, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    let new_height = height + pad_top + pad_bottom;
    let new_width = width + pad_left + pad_right;

    // Create output tensor
    let output_shape = vec![batch, channels, new_height, new_width];
    let mut output_data = vec![0.0f32; batch * channels * new_height * new_width];

    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for h in 0..new_height {
                for w in 0..new_width {
                    // Determine which values to replicate
                    let src_h = if h < pad_top {
                        0 // Replicate top edge
                    } else if h >= pad_top + height {
                        height - 1 // Replicate bottom edge
                    } else {
                        h - pad_top // Use original value
                    };

                    let src_w = if w < pad_left {
                        0 // Replicate left edge
                    } else if w >= pad_left + width {
                        width - 1 // Replicate right edge
                    } else {
                        w - pad_left // Use original value
                    };

                    let input_idx = b * (channels * height * width)
                        + c * (height * width)
                        + src_h * width
                        + src_w;

                    let output_idx = b * (channels * new_height * new_width)
                        + c * (new_height * new_width)
                        + h * new_width
                        + w;

                    output_data[output_idx] = input_data[input_idx];
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

/// Zero padding 2D
///
/// Pads a 4D tensor with zeros around the height and width dimensions.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `padding` - (pad_left, pad_right, pad_top, pad_bottom)
///
/// # Returns
/// Padded tensor of shape [N, C, H+pad_top+pad_bottom, W+pad_left+pad_right]
///
/// # Example
/// ```ignore
/// let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2])?;
/// let padded = zero_pad2d(&input, (1, 1, 1, 1))?; // Shape: [1, 1, 4, 4]
/// ```
pub fn zero_pad2d(input: &Tensor, padding: (usize, usize, usize, usize)) -> Result<Tensor> {
    let input_shape_obj = input.shape();
    let input_shape = input_shape_obj.dims();

    if input_shape.len() != 4 {
        return Err(torsh_core::error::TorshError::ShapeMismatch {
            expected: vec![0, 0, 0, 0],
            got: input_shape.to_vec(),
        });
    }

    let (pad_left, pad_right, pad_top, pad_bottom) = padding;
    let [batch, channels, height, width] = [
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    ];

    let new_height = height + pad_top + pad_bottom;
    let new_width = width + pad_left + pad_right;

    // Create zero-filled output tensor
    let output_shape = vec![batch, channels, new_height, new_width];
    let mut output_data = vec![0.0f32; batch * channels * new_height * new_width];

    // Copy input data to the appropriate position in output
    let input_data = input.to_vec()?;

    for b in 0..batch {
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let input_idx =
                        b * (channels * height * width) + c * (height * width) + h * width + w;

                    let output_h = h + pad_top;
                    let output_w = w + pad_left;
                    let output_idx = b * (channels * new_height * new_width)
                        + c * (new_height * new_width)
                        + output_h * new_width
                        + output_w;

                    output_data[output_idx] = input_data[input_idx];
                }
            }
        }
    }

    Tensor::from_vec(output_data, &output_shape)
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Calculate output size for pooling operation
pub fn pool_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
}

/// Calculate adaptive pooling kernel and stride sizes
pub fn adaptive_pool_params(input_size: usize, output_size: usize) -> (usize, usize) {
    let stride = input_size / output_size;
    let kernel = input_size - (output_size - 1) * stride;
    (kernel, stride)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_pad2d() -> Result<()> {
        // Create a simple 2x2 tensor: [[1, 2], [3, 4]]
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            &[1, 1, 2, 2], // [batch, channels, height, width]
        )?;

        // Pad with 1 on all sides
        let padded = zero_pad2d(&input, (1, 1, 1, 1))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 4, 4]);

        // Check that padding is zeros
        let padded_data = padded.to_vec()?;

        // Expected: [[0, 0, 0, 0],
        //            [0, 1, 2, 0],
        //            [0, 3, 4, 0],
        //            [0, 0, 0, 0]]
        assert_eq!(padded_data[0], 0.0); // top-left corner
        assert_eq!(padded_data[5], 1.0); // original [0,0]
        assert_eq!(padded_data[6], 2.0); // original [0,1]
        assert_eq!(padded_data[9], 3.0); // original [1,0]
        assert_eq!(padded_data[10], 4.0); // original [1,1]

        Ok(())
    }

    #[test]
    fn test_replication_pad1d() -> Result<()> {
        // Create a simple 1D tensor: [1, 2, 3]
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0],
            &[1, 1, 3], // [batch, channels, length]
        )?;

        // Pad with 2 on left, 2 on right
        let padded = replication_pad1d(&input, (2, 2))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 7]);

        // Expected: [1, 1, 1, 2, 3, 3, 3]
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 1.0); // replicated left edge
        assert_eq!(padded_data[1], 1.0); // replicated left edge
        assert_eq!(padded_data[2], 1.0); // original [0]
        assert_eq!(padded_data[3], 2.0); // original [1]
        assert_eq!(padded_data[4], 3.0); // original [2]
        assert_eq!(padded_data[5], 3.0); // replicated right edge
        assert_eq!(padded_data[6], 3.0); // replicated right edge

        Ok(())
    }

    #[test]
    fn test_replication_pad2d() -> Result<()> {
        // Create a simple 2x2 tensor: [[1, 2], [3, 4]]
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        // Pad with 1 on all sides
        let padded = replication_pad2d(&input, (1, 1, 1, 1))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 4, 4]);

        let padded_data = padded.to_vec()?;

        // Expected: [[1, 1, 2, 2],
        //            [1, 1, 2, 2],
        //            [3, 3, 4, 4],
        //            [3, 3, 4, 4]]
        assert_eq!(padded_data[0], 1.0); // top-left corner (replicated)
        assert_eq!(padded_data[5], 1.0); // original [0,0]
        assert_eq!(padded_data[6], 2.0); // original [0,1]
        assert_eq!(padded_data[15], 4.0); // bottom-right corner (replicated)

        Ok(())
    }

    #[test]
    fn test_reflection_pad1d() -> Result<()> {
        // Create a simple 1D tensor: [1, 2, 3, 4]
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4])?;

        // Pad with 2 on left, 2 on right
        let padded = reflection_pad1d(&input, (2, 2))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 8]);

        // Expected: [3, 2, 1, 2, 3, 4, 3, 2]
        //           (reflect left) | (original) | (reflect right)
        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 3.0); // reflected from index 2
        assert_eq!(padded_data[1], 2.0); // reflected from index 1
        assert_eq!(padded_data[2], 1.0); // original [0]
        assert_eq!(padded_data[3], 2.0); // original [1]
        assert_eq!(padded_data[4], 3.0); // original [2]
        assert_eq!(padded_data[5], 4.0); // original [3]
        assert_eq!(padded_data[6], 3.0); // reflected from index 2
        assert_eq!(padded_data[7], 2.0); // reflected from index 1

        Ok(())
    }

    #[test]
    fn test_reflection_pad2d() -> Result<()> {
        // Create a simple 3x3 tensor
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[1, 1, 3, 3],
        )?;

        // Pad with 1 on all sides
        let padded = reflection_pad2d(&input, (1, 1, 1, 1))?;
        let padded_shape = padded.shape();
        assert_eq!(padded_shape.dims(), &[1, 1, 5, 5]);

        let padded_data = padded.to_vec()?;

        // The center 3x3 should be the original data
        // Top-left should be reflected value (5.0 = original[1,1])
        assert_eq!(padded_data[0], 5.0); // reflected top-left

        // Center should match original
        let center_idx = 1 * 5 + 1; // row 1, col 1
        assert_eq!(padded_data[center_idx], 1.0); // original [0,0]

        Ok(())
    }

    #[test]
    fn test_reflection_pad_validation() {
        // Test that padding >= input size fails for reflection padding
        let input = Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 2]).unwrap();

        // Padding size 2 >= input length 2 should fail
        let result = reflection_pad1d(&input, (2, 1));
        assert!(result.is_err());

        let input2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();

        // Padding size 2 >= input width 2 should fail
        let result2d = reflection_pad2d(&input2d, (2, 1, 1, 1));
        assert!(result2d.is_err());
    }

    #[test]
    fn test_general_pad_zero_mode() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        // Test zero padding
        let padded = pad(&input, &[(1, 1), (1, 1)], "zero", None)?;
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 4]);

        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 0.0); // Should be zero

        Ok(())
    }

    #[test]
    fn test_general_pad_constant_mode() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        // Test constant padding with value 5.0
        let padded = pad(&input, &[(1, 1), (1, 1)], "constant", Some(5.0))?;
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 4]);

        let padded_data = padded.to_vec()?;
        assert_eq!(padded_data[0], 5.0); // Should be 5.0

        Ok(())
    }

    #[test]
    fn test_general_pad_reflect_mode() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        // Test reflection padding
        let padded = pad(&input, &[(1, 1), (1, 1)], "reflect", None)?;
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 4]);

        Ok(())
    }

    #[test]
    fn test_general_pad_replicate_mode() -> Result<()> {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        // Test replication padding
        let padded = pad(&input, &[(1, 1), (1, 1)], "replicate", None)?;
        assert_eq!(padded.shape().dims(), &[1, 1, 4, 4]);

        let padded_data = padded.to_vec()?;
        // Top-left corner should be replicated from original[0,0] = 1.0
        assert_eq!(padded_data[0], 1.0);

        Ok(())
    }

    #[test]
    fn test_general_pad_invalid_mode() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();

        // Test invalid mode
        let result = pad(&input, &[(1, 1), (1, 1)], "invalid", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pad_shape_mismatch() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();

        // Test shape mismatch (3D padding spec for 4D tensor)
        let result = pad(&input, &[(1, 1)], "zero", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_pad2d_asymmetric() -> Result<()> {
        // Test asymmetric padding
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        // Pad with different amounts on each side
        let padded = zero_pad2d(&input, (1, 2, 0, 1))?;
        assert_eq!(padded.shape().dims(), &[1, 1, 3, 5]);

        Ok(())
    }

    // =============================================================================
    // GLOBAL POOLING TESTS
    // =============================================================================

    #[test]
    fn test_global_avg_pool1d() -> Result<()> {
        // Create a 1D tensor: [[1, 2, 3, 4], [5, 6, 7, 8]]
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 2, 4], // [batch=1, channels=2, length=4]
        )?;

        let pooled = global_avg_pool1d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1]);

        let pooled_data = pooled.to_vec()?;
        // Channel 0: (1+2+3+4)/4 = 2.5
        // Channel 1: (5+6+7+8)/4 = 6.5
        assert!((pooled_data[0] - 2.5).abs() < 1e-5);
        assert!((pooled_data[1] - 6.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_avg_pool2d() -> Result<()> {
        // Create a 2x2 tensor: [[1, 2], [3, 4]]
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0],
            &[1, 1, 2, 2], // [batch=1, channels=1, height=2, width=2]
        )?;

        let pooled = global_avg_pool2d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Average: (1+2+3+4)/4 = 2.5
        assert!((pooled_data[0] - 2.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_avg_pool2d_multi_channel() -> Result<()> {
        // Create a 2-channel tensor
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, // Channel 0
                5.0, 6.0, 7.0, 8.0, // Channel 1
            ],
            &[1, 2, 2, 2], // [batch=1, channels=2, height=2, width=2]
        )?;

        let pooled = global_avg_pool2d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Channel 0: (1+2+3+4)/4 = 2.5
        // Channel 1: (5+6+7+8)/4 = 6.5
        assert!((pooled_data[0] - 2.5).abs() < 1e-5);
        assert!((pooled_data[1] - 6.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_avg_pool3d() -> Result<()> {
        // Create a 2x2x2 tensor with all values = 3.0
        let input = Tensor::from_vec(
            vec![3.0; 8],
            &[1, 1, 2, 2, 2], // [batch=1, channels=1, depth=2, height=2, width=2]
        )?;

        let pooled = global_avg_pool3d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // All values are 3.0, so average is 3.0
        assert!((pooled_data[0] - 3.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_max_pool1d() -> Result<()> {
        // Create a 1D tensor: [[1, 5, 3, 2], [8, 6, 7, 4]]
        let input = Tensor::from_vec(
            vec![1.0, 5.0, 3.0, 2.0, 8.0, 6.0, 7.0, 4.0],
            &[1, 2, 4], // [batch=1, channels=2, length=4]
        )?;

        let pooled = global_max_pool1d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1]);

        let pooled_data = pooled.to_vec()?;
        // Channel 0: max(1, 5, 3, 2) = 5.0
        // Channel 1: max(8, 6, 7, 4) = 8.0
        assert!((pooled_data[0] - 5.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_max_pool2d() -> Result<()> {
        // Create a 2x2 tensor: [[1, 5], [3, 2]]
        let input = Tensor::from_vec(
            vec![1.0, 5.0, 3.0, 2.0],
            &[1, 1, 2, 2], // [batch=1, channels=1, height=2, width=2]
        )?;

        let pooled = global_max_pool2d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Maximum: max(1, 5, 3, 2) = 5.0
        assert!((pooled_data[0] - 5.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_max_pool2d_multi_channel() -> Result<()> {
        // Create a 2-channel tensor
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, // Channel 0
                9.0, 6.0, 7.0, 8.0, // Channel 1
            ],
            &[1, 2, 2, 2], // [batch=1, channels=2, height=2, width=2]
        )?;

        let pooled = global_max_pool2d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Channel 0: max(1, 2, 3, 4) = 4.0
        // Channel 1: max(9, 6, 7, 8) = 9.0
        assert!((pooled_data[0] - 4.0).abs() < 1e-5);
        assert!((pooled_data[1] - 9.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_max_pool3d() -> Result<()> {
        // Create a 2x2x2 tensor
        let input = Tensor::from_vec(
            vec![1.0, 5.0, 3.0, 2.0, 8.0, 1.0, 4.0, 6.0],
            &[1, 1, 2, 2, 2], // [batch=1, channels=1, depth=2, height=2, width=2]
        )?;

        let pooled = global_max_pool3d(&input)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Maximum: max(1, 5, 3, 2, 8, 1, 4, 6) = 8.0
        assert!((pooled_data[0] - 8.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_pool_with_negative_values() -> Result<()> {
        // Test global max pooling with negative values
        let input = Tensor::from_vec(vec![-5.0, -2.0, -8.0, -1.0], &[1, 1, 2, 2])?;

        let max_pooled = global_max_pool2d(&input)?;
        let max_data = max_pooled.to_vec()?;
        // Maximum of negative values: max(-5, -2, -8, -1) = -1.0
        assert!((max_data[0] - (-1.0)).abs() < 1e-5);

        let avg_pooled = global_avg_pool2d(&input)?;
        let avg_data = avg_pooled.to_vec()?;
        // Average: (-5-2-8-1)/4 = -4.0
        assert!((avg_data[0] - (-4.0)).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_global_pool_shape_validation() {
        // Test that wrong input shapes are rejected

        // 1D pooling requires 3D input
        let input_2d = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        assert!(global_avg_pool1d(&input_2d).is_err());
        assert!(global_max_pool1d(&input_2d).is_err());

        // 2D pooling requires 4D input
        let input_3d = Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 2]).unwrap();
        assert!(global_avg_pool2d(&input_3d).is_err());
        assert!(global_max_pool2d(&input_3d).is_err());

        // 3D pooling requires 5D input
        let input_4d = Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 1, 2]).unwrap();
        assert!(global_avg_pool3d(&input_4d).is_err());
        assert!(global_max_pool3d(&input_4d).is_err());
    }

    // =============================================================================
    // ADAPTIVE POOLING TESTS
    // =============================================================================

    #[test]
    fn test_adaptive_avg_pool1d_downsampling() -> Result<()> {
        // Downsample from length 8 to 4
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 1, 8], // [batch=1, channels=1, length=8]
        )?;

        let pooled = adaptive_avg_pool1d(&input, 4)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 4]);

        let pooled_data = pooled.to_vec()?;
        // Each output position averages 2 input positions
        // [0]: avg(1, 2) = 1.5
        // [1]: avg(3, 4) = 3.5
        // [2]: avg(5, 6) = 5.5
        // [3]: avg(7, 8) = 7.5
        assert!((pooled_data[0] - 1.5).abs() < 1e-5);
        assert!((pooled_data[1] - 3.5).abs() < 1e-5);
        assert!((pooled_data[2] - 5.5).abs() < 1e-5);
        assert!((pooled_data[3] - 7.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_avg_pool1d_same_size() -> Result<()> {
        // Test when input and output sizes match
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4])?;

        let pooled = adaptive_avg_pool1d(&input, 4)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 4]);

        let pooled_data = pooled.to_vec()?;
        // Should be identity when sizes match
        assert!((pooled_data[0] - 1.0).abs() < 1e-5);
        assert!((pooled_data[1] - 2.0).abs() < 1e-5);
        assert!((pooled_data[2] - 3.0).abs() < 1e-5);
        assert!((pooled_data[3] - 4.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_avg_pool2d_downsampling() -> Result<()> {
        // 4x4 -> 2x2
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        )?;

        let pooled = adaptive_avg_pool2d(&input, (Some(2), Some(2)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);

        let pooled_data = pooled.to_vec()?;
        // [0,0]: avg(1,2,5,6) = 3.5
        // [0,1]: avg(3,4,7,8) = 5.5
        // [1,0]: avg(9,10,13,14) = 11.5
        // [1,1]: avg(11,12,15,16) = 13.5
        assert!((pooled_data[0] - 3.5).abs() < 1e-5);
        assert!((pooled_data[1] - 5.5).abs() < 1e-5);
        assert!((pooled_data[2] - 11.5).abs() < 1e-5);
        assert!((pooled_data[3] - 13.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_avg_pool2d_mixed_dimensions() -> Result<()> {
        // Test with None for one dimension (keep original size)
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 1, 2, 4])?;

        let pooled = adaptive_avg_pool2d(&input, (Some(2), Some(2)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);

        Ok(())
    }

    #[test]
    fn test_adaptive_avg_pool3d_downsampling() -> Result<()> {
        // 2x2x2 -> 1x1x1
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 1, 2, 2, 2],
        )?;

        let pooled = adaptive_avg_pool3d(&input, (Some(1), Some(1), Some(1)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Average of all values: (1+2+3+4+5+6+7+8)/8 = 4.5
        assert!((pooled_data[0] - 4.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_max_pool1d_downsampling() -> Result<()> {
        // Downsample from length 8 to 4
        let input = Tensor::from_vec(vec![1.0, 8.0, 3.0, 6.0, 2.0, 9.0, 4.0, 5.0], &[1, 1, 8])?;

        let pooled = adaptive_max_pool1d(&input, 4)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 4]);

        let pooled_data = pooled.to_vec()?;
        // Each output position takes max of 2 input positions
        // [0]: max(1, 8) = 8.0
        // [1]: max(3, 6) = 6.0
        // [2]: max(2, 9) = 9.0
        // [3]: max(4, 5) = 5.0
        assert!((pooled_data[0] - 8.0).abs() < 1e-5);
        assert!((pooled_data[1] - 6.0).abs() < 1e-5);
        assert!((pooled_data[2] - 9.0).abs() < 1e-5);
        assert!((pooled_data[3] - 5.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_max_pool2d_downsampling() -> Result<()> {
        // 4x4 -> 2x2
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        )?;

        let pooled = adaptive_max_pool2d(&input, (Some(2), Some(2)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);

        let pooled_data = pooled.to_vec()?;
        // [0,0]: max(1,2,5,6) = 6.0
        // [0,1]: max(3,4,7,8) = 8.0
        // [1,0]: max(9,10,13,14) = 14.0
        // [1,1]: max(11,12,15,16) = 16.0
        assert!((pooled_data[0] - 6.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);
        assert!((pooled_data[2] - 14.0).abs() < 1e-5);
        assert!((pooled_data[3] - 16.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_max_pool3d_downsampling() -> Result<()> {
        // 2x2x2 -> 1x1x1
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 15.0],
            &[1, 1, 2, 2, 2],
        )?;

        let pooled = adaptive_max_pool3d(&input, (Some(1), Some(1), Some(1)))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Maximum of all values: max(1,2,3,4,5,6,7,15) = 15.0
        assert!((pooled_data[0] - 15.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_pool_with_negative_values() -> Result<()> {
        let input = Tensor::from_vec(vec![-5.0, -2.0, -8.0, -1.0], &[1, 1, 4])?;

        // Adaptive average pooling
        let avg_pooled = adaptive_avg_pool1d(&input, 2)?;
        let avg_data = avg_pooled.to_vec()?;
        // [0]: avg(-5, -2) = -3.5
        // [1]: avg(-8, -1) = -4.5
        assert!((avg_data[0] - (-3.5)).abs() < 1e-5);
        assert!((avg_data[1] - (-4.5)).abs() < 1e-5);

        // Adaptive max pooling
        let max_pooled = adaptive_max_pool1d(&input, 2)?;
        let max_data = max_pooled.to_vec()?;
        // [0]: max(-5, -2) = -2.0
        // [1]: max(-8, -1) = -1.0
        assert!((max_data[0] - (-2.0)).abs() < 1e-5);
        assert!((max_data[1] - (-1.0)).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_pool_identity() -> Result<()> {
        // When output size equals input size, should return identical tensor
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4])?;

        let avg_pooled = adaptive_avg_pool1d(&input, 4)?;
        assert_eq!(avg_pooled.shape().dims(), input.shape().dims());
        let avg_data = avg_pooled.to_vec()?;
        assert_eq!(avg_data, vec![1.0, 2.0, 3.0, 4.0]);

        let max_pooled = adaptive_max_pool1d(&input, 4)?;
        assert_eq!(max_pooled.shape().dims(), input.shape().dims());
        let max_data = max_pooled.to_vec()?;
        assert_eq!(max_data, vec![1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_adaptive_pool_multi_channel() -> Result<()> {
        // Test with multiple channels
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, // Channel 0
                5.0, 6.0, 7.0, 8.0, // Channel 1
            ],
            &[1, 2, 4],
        )?;

        let pooled = adaptive_avg_pool1d(&input, 2)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 2]);

        let pooled_data = pooled.to_vec()?;
        // Channel 0: [avg(1,2)=1.5, avg(3,4)=3.5]
        // Channel 1: [avg(5,6)=5.5, avg(7,8)=7.5]
        assert!((pooled_data[0] - 1.5).abs() < 1e-5);
        assert!((pooled_data[1] - 3.5).abs() < 1e-5);
        assert!((pooled_data[2] - 5.5).abs() < 1e-5);
        assert!((pooled_data[3] - 7.5).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_adaptive_pool_batch() -> Result<()> {
        // Test with multiple batches
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, // Batch 0
                5.0, 6.0, 7.0, 8.0, // Batch 1
            ],
            &[2, 1, 4],
        )?;

        let pooled = adaptive_max_pool1d(&input, 2)?;
        assert_eq!(pooled.shape().dims(), &[2, 1, 2]);

        let pooled_data = pooled.to_vec()?;
        // Batch 0: [max(1,2)=2, max(3,4)=4]
        // Batch 1: [max(5,6)=6, max(7,8)=8]
        assert!((pooled_data[0] - 2.0).abs() < 1e-5);
        assert!((pooled_data[1] - 4.0).abs() < 1e-5);
        assert!((pooled_data[2] - 6.0).abs() < 1e-5);
        assert!((pooled_data[3] - 8.0).abs() < 1e-5);

        Ok(())
    }

    // =============================================================================
    // MAX POOLING TESTS
    // =============================================================================

    #[test]
    fn test_max_pool1d_basic() -> Result<()> {
        // Test basic max pooling 1D: [1,2,8,4,5,3,7,6] with kernel=2, stride=2
        let input = Tensor::from_vec(vec![1.0, 2.0, 8.0, 4.0, 5.0, 3.0, 7.0, 6.0], &[1, 1, 8])?;

        let pooled = max_pool1d(&input, 2, None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 4]);

        let pooled_data = pooled.to_vec()?;
        // [max(1,2)=2, max(8,4)=8, max(5,3)=5, max(7,6)=7]
        assert!((pooled_data[0] - 2.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);
        assert!((pooled_data[2] - 5.0).abs() < 1e-5);
        assert!((pooled_data[3] - 7.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_max_pool1d_with_stride() -> Result<()> {
        // Test with custom stride
        let input = Tensor::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0, 6.0], &[1, 1, 6])?;

        let pooled = max_pool1d(&input, 2, Some(1), None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 5]);

        let pooled_data = pooled.to_vec()?;
        // stride=1: [max(1,5)=5, max(5,2)=5, max(2,8)=8, max(8,3)=8, max(3,6)=6]
        assert!((pooled_data[0] - 5.0).abs() < 1e-5);
        assert!((pooled_data[1] - 5.0).abs() < 1e-5);
        assert!((pooled_data[2] - 8.0).abs() < 1e-5);
        assert!((pooled_data[3] - 8.0).abs() < 1e-5);
        assert!((pooled_data[4] - 6.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_max_pool2d_basic() -> Result<()> {
        // Test 2x2 max pooling on 4x4 input
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0, 9.0, 10.0, 13.0, 14.0, 11.0, 12.0, 15.0,
                16.0,
            ],
            &[1, 1, 4, 4],
        )?;

        let pooled = max_pool2d(&input, (2, 2), None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);

        let pooled_data = pooled.to_vec()?;
        // [max(1,2,3,4)=4, max(5,6,7,8)=8, max(9,10,11,12)=12, max(13,14,15,16)=16]
        assert!((pooled_data[0] - 4.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);
        assert!((pooled_data[2] - 12.0).abs() < 1e-5);
        assert!((pooled_data[3] - 16.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_max_pool2d_multi_channel() -> Result<()> {
        // Test with 2 channels
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, // Channel 0
                5.0, 6.0, 7.0, 8.0, // Channel 1
            ],
            &[1, 2, 2, 2],
        )?;

        let pooled = max_pool2d(&input, (2, 2), None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 2, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Channel 0: max(1,2,3,4) = 4.0
        // Channel 1: max(5,6,7,8) = 8.0
        assert!((pooled_data[0] - 4.0).abs() < 1e-5);
        assert!((pooled_data[1] - 8.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_max_pool3d_basic() -> Result<()> {
        // Test 3D max pooling with 2x2x2 kernel
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 16.0],
            &[1, 1, 2, 2, 2],
        )?;

        let pooled = max_pool3d(&input, (2, 2, 2), None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 1, 1, 1]);

        let pooled_data = pooled.to_vec()?;
        // Maximum of all values: 16.0
        assert!((pooled_data[0] - 16.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_max_pool_with_negative_values() -> Result<()> {
        // Test max pooling handles negative values correctly
        let input = Tensor::from_vec(vec![-5.0, -2.0, -8.0, -1.0], &[1, 1, 4])?;

        let pooled = max_pool1d(&input, 2, None, None, None)?;
        let pooled_data = pooled.to_vec()?;

        // [max(-5,-2)=-2, max(-8,-1)=-1]
        assert!((pooled_data[0] - (-2.0)).abs() < 1e-5);
        assert!((pooled_data[1] - (-1.0)).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_max_pool2d_with_padding() -> Result<()> {
        // Test 2D max pooling with padding
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;

        // With padding=1, kernel=2, stride=2 (default): output = (2+2*1-2)/2+1 = 2
        let pooled = max_pool2d(&input, (2, 2), None, Some((1, 1)), None)?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2, 2]);

        // All outputs should have actual values (not -inf) since each window
        // contains at least one actual data point
        let pooled_data = pooled.to_vec()?;
        for &val in &pooled_data {
            assert!(val > f32::NEG_INFINITY);
        }

        Ok(())
    }

    #[test]
    fn test_max_pool1d_with_dilation() -> Result<()> {
        // Test max pooling with dilation
        let input = Tensor::from_vec(vec![1.0, 5.0, 2.0, 8.0, 3.0, 6.0], &[1, 1, 6])?;

        // kernel_size=2, stride=2, dilation=2
        // Window 1: positions 0,2 -> values 1,2 -> max=2
        // Window 2: positions 2,4 -> values 2,3 -> max=3
        let pooled = max_pool1d(&input, 2, Some(2), None, Some(2))?;
        assert_eq!(pooled.shape().dims(), &[1, 1, 2]);

        let pooled_data = pooled.to_vec()?;
        assert!((pooled_data[0] - 2.0).abs() < 1e-5);
        assert!((pooled_data[1] - 3.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_max_pool_batch() -> Result<()> {
        // Test with multiple batches
        let input = Tensor::from_vec(
            vec![
                1.0, 4.0, 2.0, 3.0, // Batch 0
                5.0, 8.0, 6.0, 7.0, // Batch 1
            ],
            &[2, 1, 4],
        )?;

        let pooled = max_pool1d(&input, 2, None, None, None)?;
        assert_eq!(pooled.shape().dims(), &[2, 1, 2]);

        let pooled_data = pooled.to_vec()?;
        // Batch 0: [max(1,4)=4, max(2,3)=3]
        // Batch 1: [max(5,8)=8, max(6,7)=7]
        assert!((pooled_data[0] - 4.0).abs() < 1e-5);
        assert!((pooled_data[1] - 3.0).abs() < 1e-5);
        assert!((pooled_data[2] - 8.0).abs() < 1e-5);
        assert!((pooled_data[3] - 7.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_max_pool_shape_validation() {
        // Test that wrong input shapes are rejected

        // 1D pooling requires 3D input
        let input_2d = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).unwrap();
        assert!(max_pool1d(&input_2d, 2, None, None, None).is_err());

        // 2D pooling requires 4D input
        let input_3d = Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 2]).unwrap();
        assert!(max_pool2d(&input_3d, (2, 2), None, None, None).is_err());

        // 3D pooling requires 5D input
        let input_4d = Tensor::from_vec(vec![1.0, 2.0], &[1, 1, 1, 2]).unwrap();
        assert!(max_pool3d(&input_4d, (2, 2, 2), None, None, None).is_err());
    }
}
