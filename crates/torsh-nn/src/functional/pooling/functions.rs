//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;
use torsh_tensor::Tensor;

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
                        let input_l = l - padding;
                        let input_idx = b * (channels * length) + c * length + input_l;
                        max_val = max_val.max(input_data[input_idx]);
                    }
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
                                let input_h = h - padding.0;
                                let input_w = w - padding.1;
                                let input_idx = b * (channels * height * width)
                                    + c * (height * width)
                                    + input_h * width
                                    + input_w;
                                max_val = max_val.max(input_data[input_idx]);
                            }
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
                        if count_include_pad {
                            count += 1;
                        }
                    } else {
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
                                if count_include_pad {
                                    count += 1;
                                }
                            } else {
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
                                        if count_include_pad {
                                            count += 1;
                                        }
                                    } else {
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
    let mut output_data = vec![0.0f32; batch_size * output_size];
    let input_data = input.to_vec()?;
    for b in 0..batch_size {
        for out_l in 0..output_size {
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
    let mut output_data = vec![0.0f32; batch_size * output_height * output_width];
    let input_data = input.to_vec()?;
    for b in 0..batch_size {
        for out_h in 0..output_height {
            for out_w in 0..output_width {
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
    match (input_shape.len(), padding.len()) {
        (3, 1) => {
            let (pad_left, pad_right) = padding[0];
            match mode {
                "reflect" => reflection_pad1d(input, (pad_left, pad_right)),
                "replicate" => replication_pad1d(input, (pad_left, pad_right)),
                "constant" | "zero" => {
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
                    if fill_value == 0.0 {
                        zero_pad2d(input, (pad_left, pad_right, pad_top, pad_bottom))
                    } else {
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
    if pad_left >= length || pad_right >= length {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Padding size ({}, {}) cannot be >= input length ({}) for reflection padding",
            pad_left, pad_right, length
        )));
    }
    let new_length = length + pad_left + pad_right;
    let output_shape = vec![batch, channels, new_length];
    let mut output_data = vec![0.0f32; batch * channels * new_length];
    let input_data = input.to_vec()?;
    for b in 0..batch {
        for c in 0..channels {
            for l in 0..new_length {
                let input_idx = b * (channels * length) + c * length;
                let output_idx = b * (channels * new_length) + c * new_length + l;
                let src_l = if l < pad_left {
                    pad_left - l
                } else if l >= pad_left + length {
                    let offset = l - (pad_left + length);
                    length - 2 - offset
                } else {
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
    let output_shape = vec![batch, channels, new_height, new_width];
    let mut output_data = vec![0.0f32; batch * channels * new_height * new_width];
    let input_data = input.to_vec()?;
    for b in 0..batch {
        for c in 0..channels {
            for h in 0..new_height {
                for w in 0..new_width {
                    let src_h = if h < pad_top {
                        pad_top - h
                    } else if h >= pad_top + height {
                        let offset = h - (pad_top + height);
                        height - 2 - offset
                    } else {
                        h - pad_top
                    };
                    let src_w = if w < pad_left {
                        pad_left - w
                    } else if w >= pad_left + width {
                        let offset = w - (pad_left + width);
                        width - 2 - offset
                    } else {
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
    let output_shape = vec![batch, channels, new_length];
    let mut output_data = vec![0.0f32; batch * channels * new_length];
    let input_data = input.to_vec()?;
    for b in 0..batch {
        for c in 0..channels {
            for l in 0..new_length {
                let input_idx = b * (channels * length) + c * length;
                let output_idx = b * (channels * new_length) + c * new_length + l;
                let src_l = if l < pad_left {
                    0
                } else if l >= pad_left + length {
                    length - 1
                } else {
                    l - pad_left
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
    let output_shape = vec![batch, channels, new_height, new_width];
    let mut output_data = vec![0.0f32; batch * channels * new_height * new_width];
    let input_data = input.to_vec()?;
    for b in 0..batch {
        for c in 0..channels {
            for h in 0..new_height {
                for w in 0..new_width {
                    let src_h = if h < pad_top {
                        0
                    } else if h >= pad_top + height {
                        height - 1
                    } else {
                        h - pad_top
                    };
                    let src_w = if w < pad_left {
                        0
                    } else if w >= pad_left + width {
                        width - 1
                    } else {
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
    let output_shape = vec![batch, channels, new_height, new_width];
    let mut output_data = vec![0.0f32; batch * channels * new_height * new_width];
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
