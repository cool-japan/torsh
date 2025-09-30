//! Convolution operations for neural networks

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// 1D convolution over an input signal composed of several input planes.
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> TorshResult<Tensor> {
    // Input shape: (N, C_in, L)
    // Weight shape: (C_out, C_in/groups, kernel_size)
    // Output shape: (N, C_out, L_out)

    input.conv1d(weight, bias, stride, padding, dilation, groups)
}

/// 2D convolution over an input image composed of several input planes.
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> TorshResult<Tensor> {
    // Input shape: (N, C_in, H, W)
    // Weight shape: (C_out, C_in/groups, kernel_h, kernel_w)
    // Output shape: (N, C_out, H_out, W_out)

    input.conv2d(weight, bias, stride, padding, dilation, groups)
}

/// 3D convolution over an input image composed of several input planes.
pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
) -> TorshResult<Tensor> {
    // Input shape: (N, C_in, D, H, W)
    // Weight shape: (C_out, C_in/groups, kernel_d, kernel_h, kernel_w)
    // Output shape: (N, C_out, D_out, H_out, W_out)

    input.conv3d(weight, bias, stride, padding, dilation, groups)
}

/// Transposed 1D convolution (also known as deconvolution).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
    dilation: usize,
) -> TorshResult<Tensor> {
    // Input shape: (N, C_in, L_in)
    // Weight shape: (C_in, C_out/groups, kernel_size)
    // Output shape: (N, C_out, L_out)

    let input_shape = input.shape().dims().to_vec();
    let weight_shape = weight.shape().dims().to_vec();

    if input_shape.len() != 3 {
        return Err(torsh_core::TorshError::dimension_error_with_context(
            "Input must be 3D (N, C_in, L_in)",
            "conv_transpose1d",
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_length = input_shape[2];

    let kernel_size = weight_shape[2];
    let out_channels = weight_shape[1] * groups;

    // Calculate output length
    let output_length = conv_transpose_output_size(
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        dilation,
    );

    // Fallback implementation using conv2d operations
    // This is a simplified approach - transpose conv can be implemented as
    // regular conv with modified stride and padding patterns

    // Create output tensor with proper shape
    let output_shape = vec![batch_size, out_channels, output_length];
    let mut output_data = vec![0.0f32; output_shape.iter().product()];

    // Apply basic transposed convolution logic
    // This is a simplified implementation that would need optimization
    for b in 0..batch_size {
        for out_c in 0..out_channels {
            for in_c in 0..(in_channels / groups) {
                let weight_idx = in_c * out_channels / groups + out_c;

                for i in 0..input_length {
                    for k in 0..kernel_size {
                        let output_pos = i * stride + k * dilation;
                        if output_pos >= padding && output_pos < output_length + padding {
                            let final_pos = output_pos - padding;
                            if final_pos < output_length {
                                // Simplified weight access
                                let input_data = input.data()?;
                                let weight_data = weight.data()?;
                                let input_val = input_data
                                    [b * in_channels * input_length + in_c * input_length + i];
                                let weight_val = weight_data[weight_idx * kernel_size + k];
                                let output_idx = b * out_channels * output_length
                                    + out_c * output_length
                                    + final_pos;
                                output_data[output_idx] += input_val * weight_val;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut result = Tensor::from_data(output_data, output_shape, input.device())?;

    // Add bias if provided
    if let Some(bias_tensor) = bias {
        result = result.add_op(bias_tensor)?;
    }

    Ok(result)
}

/// Transposed 2D convolution (also known as deconvolution).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    groups: usize,
    dilation: (usize, usize),
) -> TorshResult<Tensor> {
    // Input shape: (N, C_in, H_in, W_in)
    // Weight shape: (C_in, C_out/groups, kernel_h, kernel_w)
    // Output shape: (N, C_out, H_out, W_out)

    let input_shape = input.shape().dims().to_vec();
    let weight_shape = weight.shape().dims().to_vec();

    if input_shape.len() != 4 {
        return Err(torsh_core::TorshError::dimension_error_with_context(
            "Input must be 4D (N, C_in, H_in, W_in)",
            "conv_transpose2d",
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];
    let out_channels = weight_shape[1] * groups;

    // Calculate output dimensions
    let output_height = conv_transpose_output_size(
        input_height,
        kernel_height,
        stride.0,
        padding.0,
        output_padding.0,
        dilation.0,
    );
    let output_width = conv_transpose_output_size(
        input_width,
        kernel_width,
        stride.1,
        padding.1,
        output_padding.1,
        dilation.1,
    );

    // Try to use tensor's built-in method first
    if let Ok(result) = input.conv_transpose2d(
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    ) {
        Ok(result)
    } else {
        // Fallback implementation
        // Transposed convolution can be thought of as:
        // 1. Upsampling the input by inserting zeros between elements
        // 2. Applying regular convolution with flipped weights

        let output_shape = vec![batch_size, out_channels, output_height, output_width];
        let mut output_data = vec![0.0f32; output_shape.iter().product()];

        // Simplified transposed convolution implementation
        for b in 0..batch_size {
            for out_c in 0..out_channels {
                for in_c in 0..(in_channels / groups) {
                    for h in 0..input_height {
                        for w in 0..input_width {
                            let input_data = input.data()?;
                            let input_val =
                                input_data[b * in_channels * input_height * input_width
                                    + in_c * input_height * input_width
                                    + h * input_width
                                    + w];

                            // Apply kernel at each position
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let out_h = h * stride.0 + kh * dilation.0;
                                    let out_w = w * stride.1 + kw * dilation.1;

                                    if out_h >= padding.0 && out_w >= padding.1 {
                                        let final_h = out_h - padding.0;
                                        let final_w = out_w - padding.1;

                                        if final_h < output_height && final_w < output_width {
                                            let weight_data = weight.data()?;
                                            let weight_val = weight_data[in_c
                                                * out_channels
                                                * kernel_height
                                                * kernel_width
                                                + out_c * kernel_height * kernel_width
                                                + kh * kernel_width
                                                + kw];

                                            let output_idx =
                                                b * out_channels * output_height * output_width
                                                    + out_c * output_height * output_width
                                                    + final_h * output_width
                                                    + final_w;

                                            output_data[output_idx] += input_val * weight_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut result = Tensor::from_data(output_data, output_shape, input.device())?;

        // Add bias if provided
        if let Some(bias_tensor) = bias {
            // Broadcast bias across spatial dimensions
            let bias_shape = vec![1, out_channels, 1, 1];
            let bias_reshaped =
                bias_tensor.view(&bias_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
            result = result.add_op(&bias_reshaped)?;
        }

        Ok(result)
    }
}

/// Transposed 3D convolution (also known as deconvolution).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
    groups: usize,
    dilation: (usize, usize, usize),
) -> TorshResult<Tensor> {
    // Input shape: (N, C_in, D_in, H_in, W_in)
    // Weight shape: (C_in, C_out/groups, kernel_d, kernel_h, kernel_w)
    // Output shape: (N, C_out, D_out, H_out, W_out)

    let input_shape = input.shape().dims().to_vec();
    let weight_shape = weight.shape().dims().to_vec();

    if input_shape.len() != 5 {
        return Err(torsh_core::TorshError::dimension_error_with_context(
            "Input must be 5D (N, C_in, D_in, H_in, W_in)",
            "conv_transpose3d",
        ));
    }

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_depth = input_shape[2];
    let input_height = input_shape[3];
    let input_width = input_shape[4];

    let kernel_depth = weight_shape[2];
    let kernel_height = weight_shape[3];
    let kernel_width = weight_shape[4];
    let out_channels = weight_shape[1] * groups;

    // Calculate output dimensions
    let output_depth = conv_transpose_output_size(
        input_depth,
        kernel_depth,
        stride.0,
        padding.0,
        output_padding.0,
        dilation.0,
    );
    let output_height = conv_transpose_output_size(
        input_height,
        kernel_height,
        stride.1,
        padding.1,
        output_padding.1,
        dilation.1,
    );
    let output_width = conv_transpose_output_size(
        input_width,
        kernel_width,
        stride.2,
        padding.2,
        output_padding.2,
        dilation.2,
    );

    // Fallback implementation for 3D transposed convolution
    // Note: conv_transpose3d is not yet implemented in tensor crate
    let output_shape = vec![
        batch_size,
        out_channels,
        output_depth,
        output_height,
        output_width,
    ];
    let mut output_data = vec![0.0f32; output_shape.iter().product()];

    // Simplified 3D transposed convolution implementation
    for b in 0..batch_size {
        for out_c in 0..out_channels {
            for in_c in 0..(in_channels / groups) {
                for d in 0..input_depth {
                    for h in 0..input_height {
                        for w in 0..input_width {
                            let input_data = input.data()?;
                            let input_val = input_data[b
                                * in_channels
                                * input_depth
                                * input_height
                                * input_width
                                + in_c * input_depth * input_height * input_width
                                + d * input_height * input_width
                                + h * input_width
                                + w];

                            // Apply kernel at each position
                            for kd in 0..kernel_depth {
                                for kh in 0..kernel_height {
                                    for kw in 0..kernel_width {
                                        let out_d = d * stride.0 + kd * dilation.0;
                                        let out_h = h * stride.1 + kh * dilation.1;
                                        let out_w = w * stride.2 + kw * dilation.2;

                                        if out_d >= padding.0
                                            && out_h >= padding.1
                                            && out_w >= padding.2
                                        {
                                            let final_d = out_d - padding.0;
                                            let final_h = out_h - padding.1;
                                            let final_w = out_w - padding.2;

                                            if final_d < output_depth
                                                && final_h < output_height
                                                && final_w < output_width
                                            {
                                                let weight_data = weight.data()?;
                                                let weight_val = weight_data[in_c
                                                    * out_channels
                                                    * kernel_depth
                                                    * kernel_height
                                                    * kernel_width
                                                    + out_c
                                                        * kernel_depth
                                                        * kernel_height
                                                        * kernel_width
                                                    + kd * kernel_height * kernel_width
                                                    + kh * kernel_width
                                                    + kw];

                                                let output_idx = b
                                                    * out_channels
                                                    * output_depth
                                                    * output_height
                                                    * output_width
                                                    + out_c
                                                        * output_depth
                                                        * output_height
                                                        * output_width
                                                    + final_d * output_height * output_width
                                                    + final_h * output_width
                                                    + final_w;

                                                output_data[output_idx] += input_val * weight_val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut result = Tensor::from_data(output_data, output_shape, input.device())?;

    // Add bias if provided
    if let Some(bias_tensor) = bias {
        // Broadcast bias across spatial dimensions
        let bias_shape = vec![1, out_channels, 1, 1, 1];
        let bias_reshaped =
            bias_tensor.view(&bias_shape.iter().map(|&x| x as i32).collect::<Vec<_>>())?;
        result = result.add_op(&bias_reshaped)?;
    }

    Ok(result)
}

/// Extracts sliding local blocks from a batched input tensor.
pub fn unfold(input: &Tensor, dimension: i64, size: usize, step: usize) -> TorshResult<Tensor> {
    // Creates sliding windows along dimension
    let input_shape = input.shape().dims().to_vec();
    let ndim = input_shape.len() as i64;

    // Normalize dimension to positive value
    let dim = if dimension < 0 {
        (ndim + dimension) as usize
    } else {
        dimension as usize
    };

    if dim >= input_shape.len() {
        return Err(torsh_core::TorshError::dimension_error_with_context(
            &format!(
                "Dimension {} is out of range for tensor with {} dimensions",
                dimension, ndim
            ),
            "unfold",
        ));
    }

    let dim_size = input_shape[dim];
    if size > dim_size {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            &format!(
                "Unfold size {} is larger than dimension size {}",
                size, dim_size
            ),
            "unfold",
        ));
    }

    // Calculate number of windows
    let num_windows = if step == 0 {
        1
    } else {
        ((dim_size - size) / step) + 1
    };

    // Create output shape: original shape with dimension replaced by [num_windows, size]
    let mut output_shape = input_shape.clone();
    output_shape[dim] = num_windows;
    output_shape.insert(dim + 1, size);

    let input_data = input.data()?;
    let mut output_data = vec![0.0f32; output_shape.iter().product()];

    // Calculate strides for input tensor
    let mut input_strides = vec![1; input_shape.len()];
    for i in (0..input_shape.len() - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Calculate strides for output tensor
    let mut output_strides = vec![1; output_shape.len()];
    for i in (0..output_shape.len() - 1).rev() {
        output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }

    // Extract sliding windows
    let total_elements_before_dim: usize = input_shape[..dim].iter().product();
    let total_elements_after_dim: usize = input_shape[dim + 1..].iter().product();

    for before_idx in 0..total_elements_before_dim {
        for after_idx in 0..total_elements_after_dim {
            for window_idx in 0..num_windows {
                for size_idx in 0..size {
                    let input_dim_idx = window_idx * step + size_idx;
                    if input_dim_idx < dim_size {
                        // Calculate input index
                        let mut input_idx = 0;
                        input_idx += before_idx * input_strides[..dim].iter().sum::<usize>();
                        input_idx += input_dim_idx * input_strides[dim];
                        input_idx += after_idx * input_strides[dim + 1..].iter().sum::<usize>();

                        // Calculate output index
                        let mut output_idx = 0;
                        output_idx += before_idx * output_strides[..dim].iter().sum::<usize>();
                        output_idx += window_idx * output_strides[dim];
                        output_idx += size_idx * output_strides[dim + 1];
                        output_idx += after_idx * output_strides[dim + 2..].iter().sum::<usize>();

                        if input_idx < input_data.len() && output_idx < output_data.len() {
                            output_data[output_idx] = input_data[input_idx];
                        }
                    }
                }
            }
        }
    }

    Tensor::from_data(output_data, output_shape, input.device())
}

/// Combines an array of sliding local blocks into a large containing tensor.
pub fn fold(
    input: &Tensor,
    output_size: (usize, usize),
    kernel_size: (usize, usize),
    dilation: (usize, usize),
    padding: (usize, usize),
    stride: (usize, usize),
) -> TorshResult<Tensor> {
    // Inverse of unfold operation for 2D tensors
    // Input shape: (N, C * kernel_h * kernel_w, L) where L is number of sliding windows
    // Output shape: (N, C, output_h, output_w)

    let input_shape = input.shape().dims().to_vec();
    if input_shape.len() != 3 {
        return Err(torsh_core::TorshError::dimension_error_with_context(
            "Fold input must be 3D (N, C * kernel_h * kernel_w, L)",
            "fold",
        ));
    }

    let batch_size = input_shape[0];
    let channels_times_kernel = input_shape[1];
    let num_windows = input_shape[2];

    let kernel_area = kernel_size.0 * kernel_size.1;
    if channels_times_kernel % kernel_area != 0 {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            "Input channel dimension must be divisible by kernel area",
            "fold",
        ));
    }

    let channels = channels_times_kernel / kernel_area;
    let output_height = output_size.0;
    let output_width = output_size.1;

    // Verify that the number of windows matches expected value
    let expected_windows = {
        let h_windows =
            (output_height + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1;
        let w_windows =
            (output_width + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1;
        h_windows * w_windows
    };

    if num_windows != expected_windows {
        return Err(torsh_core::TorshError::invalid_argument_with_context(
            &format!("Expected {} windows, got {}", expected_windows, num_windows),
            "fold",
        ));
    }

    let output_shape = vec![batch_size, channels, output_height, output_width];
    let mut output_data = vec![0.0f32; output_shape.iter().product()];
    let input_data = input.data()?;

    // Number of windows in each dimension
    let h_windows =
        (output_height + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) / stride.0 + 1;
    let w_windows =
        (output_width + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) / stride.1 + 1;

    for b in 0..batch_size {
        for c in 0..channels {
            for h_win in 0..h_windows {
                for w_win in 0..w_windows {
                    let window_idx = h_win * w_windows + w_win;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let kernel_idx = kh * kernel_size.1 + kw;
                            let input_channel_idx = c * kernel_area + kernel_idx;

                            // Calculate output position
                            let out_h = h_win as i32 * stride.0 as i32
                                + kh as i32 * dilation.0 as i32
                                - padding.0 as i32;
                            let out_w = w_win as i32 * stride.1 as i32
                                + kw as i32 * dilation.1 as i32
                                - padding.1 as i32;

                            if out_h >= 0
                                && out_w >= 0
                                && (out_h as usize) < output_height
                                && (out_w as usize) < output_width
                            {
                                let input_idx = b * channels_times_kernel * num_windows
                                    + input_channel_idx * num_windows
                                    + window_idx;

                                let output_idx = b * channels * output_height * output_width
                                    + c * output_height * output_width
                                    + (out_h as usize) * output_width
                                    + (out_w as usize);

                                if input_idx < input_data.len() && output_idx < output_data.len() {
                                    output_data[output_idx] += input_data[input_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor::from_data(output_data, output_shape, input.device())
}

/// Depthwise convolution
pub fn depthwise_conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> TorshResult<Tensor> {
    // Depthwise convolution is a grouped convolution where groups = in_channels
    let in_channels = input.shape().dims()[1];
    conv2d(input, weight, bias, stride, padding, dilation, in_channels)
}

/// Separable convolution (depthwise + pointwise)
pub fn separable_conv2d(
    input: &Tensor,
    depthwise_weight: &Tensor,
    pointwise_weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> TorshResult<Tensor> {
    // First apply depthwise convolution
    let depthwise_out = depthwise_conv2d(input, depthwise_weight, None, stride, padding, dilation)?;

    // Then apply pointwise convolution (1x1 conv)
    conv2d(
        &depthwise_out,
        pointwise_weight,
        bias,
        (1, 1),
        (0, 0),
        (1, 1),
        1,
    )
}

/// Helper function to calculate output size for convolution
pub fn conv_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> usize {
    let kernel_size_dilated = (kernel_size - 1) * dilation + 1;
    ((input_size + 2 * padding - kernel_size_dilated) / stride) + 1
}

/// Helper function to calculate output size for transposed convolution
pub fn conv_transpose_output_size(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> usize {
    let kernel_size_dilated = (kernel_size - 1) * dilation + 1;
    (input_size - 1) * stride - 2 * padding + kernel_size_dilated + output_padding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_output_size() {
        // Test standard convolution output size calculation
        assert_eq!(conv_output_size(32, 3, 1, 1, 1), 32);
        assert_eq!(conv_output_size(32, 3, 2, 1, 1), 16);
        assert_eq!(conv_output_size(32, 5, 1, 2, 1), 32);
        assert_eq!(conv_output_size(32, 3, 1, 1, 2), 30);
    }

    #[test]
    fn test_conv_transpose_output_size() {
        // Test transposed convolution output size calculation
        assert_eq!(conv_transpose_output_size(16, 3, 2, 1, 1, 1), 32);
        assert_eq!(conv_transpose_output_size(16, 4, 2, 1, 0, 1), 32);
    }
}
