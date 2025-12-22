//! # Convolution Operations for Neural Networks
//!
//! This module provides comprehensive convolution operations fundamental to deep learning,
//! including standard, transposed, depthwise, and separable convolutions.
//!
//! ## Mathematical Foundation
//!
//! ### Standard Convolution
//! The discrete convolution operation computes:
//! ```text
//! y[n] = Σ(k) x[n + k] * w[k] + b
//! ```
//! where:
//! - `x` is the input signal
//! - `w` is the convolution kernel (learnable weights)
//! - `b` is the bias term
//! - `k` ranges over the kernel size
//!
//! ### Output Size Calculation
//! For a 1D convolution with stride `s`, padding `p`, dilation `d`, and kernel size `k`:
//! ```text
//! L_out = floor((L_in + 2p - d(k - 1) - 1) / s) + 1
//! ```
//!
//! For 2D convolutions (height and width computed independently):
//! ```text
//! H_out = floor((H_in + 2p_h - d_h(k_h - 1) - 1) / s_h) + 1
//! W_out = floor((W_in + 2p_w - d_w(k_w - 1) - 1) / s_w) + 1
//! ```
//!
//! ### Transposed Convolution (Deconvolution)
//! Transposed convolution reverses the spatial transformation:
//! ```text
//! L_out = (L_in - 1) * s - 2p + d(k - 1) + op + 1
//! ```
//! where `op` is the output padding parameter.
//!
//! ### Grouped Convolution
//! Groups divide channels into independent convolution operations:
//! - Total parameters: `C_out * (C_in / groups) * k`
//! - Computational efficiency: O(1/groups) compared to standard convolution
//! - Depthwise convolution is the special case where `groups = C_in = C_out`
//!
//! ## Performance Characteristics
//!
//! ### Computational Complexity
//! - **Standard Conv2D**: O(N * C_in * C_out * k_h * k_w * H_out * W_out)
//! - **Grouped Conv2D**: O(N * C_in * C_out * k_h * k_w * H_out * W_out / groups)
//! - **Depthwise Conv2D**: O(N * C * k_h * k_w * H_out * W_out) where C = C_in = C_out
//! - **Separable Conv2D**: O(N * C * k * H_out * W_out + N * C_in * C_out * H_out * W_out)
//!
//! ### Memory Usage
//! - **Weights**: C_out * C_in * k_h * k_w * sizeof(dtype)
//! - **Activations**: N * C_out * H_out * W_out * sizeof(dtype)
//! - **Workspace** (for im2col): N * C_in * k_h * k_w * H_out * W_out * sizeof(dtype)
//!
//! ## Examples
//!
//! ### Basic 2D Convolution
//! ```rust,no_run
//! # use torsh_tensor::Tensor;
//! # use torsh_functional::conv::conv2d;
//! # fn example() -> torsh_core::Result<()> {
//! // Input: batch=1, channels=3 (RGB), height=32, width=32
//! let input = randn(&[1, 3, 32, 32])?;
//!
//! // Kernel: 64 output channels, 3 input channels, 3x3 kernel
//! let weight = randn(&[64, 3, 3, 3])?;
//! let bias = Some(Tensor::zeros(&[64])?);
//!
//! // Standard convolution: stride=1, padding=1
//! let output = conv2d(
//!     &input,
//!     &weight,
//!     bias.as_ref(),
//!     (1, 1),  // stride
//!     (1, 1),  // padding (maintains spatial dimensions)
//!     (1, 1),  // dilation
//!     1,       // groups
//! )?;
//!
//! // Output shape: [1, 64, 32, 32]
//! # Ok(())
//! # }
//! ```
//!
//! ### Depthwise Separable Convolution
//! ```rust,no_run
//! # use torsh_tensor::Tensor;
//! # use torsh_functional::conv::{depthwise_conv2d, conv2d};
//! # fn example() -> torsh_core::Result<()> {
//! let input = randn(&[1, 64, 32, 32])?;
//!
//! // Depthwise convolution (spatial filtering)
//! let depthwise_weight = randn(&[64, 1, 3, 3])?;
//! let depthwise = depthwise_conv2d(
//!     &input,
//!     &depthwise_weight,
//!     None,
//!     (1, 1),
//!     (1, 1),
//!     (1, 1),
//! )?;
//!
//! // Pointwise convolution (channel mixing)
//! let pointwise_weight = randn(&[128, 64, 1, 1])?;
//! let output = conv2d(
//!     &depthwise,
//!     &pointwise_weight,
//!     None,
//!     (1, 1),
//!     (0, 0),
//!     (1, 1),
//!     1,
//! )?;
//!
//! // Output shape: [1, 128, 32, 32]
//! // Parameters: 64*(3*3) + 128*64 = 8768 (vs 64*128*3*3 = 73728 for standard)
//! # Ok(())
//! # }
//! ```

use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// 1D convolution over an input signal composed of several input planes.
///
/// # Mathematical Definition
/// ```text
/// out[c_out][l] = bias[c_out] + Σ(c_in, k) weight[c_out][c_in][k] * input[c_in][l*s + k*d - p]
/// ```
///
/// # Arguments
/// * `input` - Input tensor of shape `[N, C_in, L]`
/// * `weight` - Convolution kernel of shape `[C_out, C_in/groups, K]`
/// * `bias` - Optional bias tensor of shape `[C_out]`
/// * `stride` - Stride of the convolution
/// * `padding` - Zero padding added to both sides
/// * `dilation` - Spacing between kernel elements
/// * `groups` - Number of blocked connections from input to output channels
///
/// # Shape
/// - Input: `[N, C_in, L]`
/// - Weight: `[C_out, C_in/groups, K]`
/// - Bias: `[C_out]` (optional)
/// - Output: `[N, C_out, L_out]` where `L_out = floor((L + 2*padding - dilation*(K-1) - 1) / stride) + 1`
///
/// # Examples
/// ```rust,no_run
/// # use torsh_tensor::Tensor;
/// # use torsh_functional::conv::conv1d;
/// # use torsh_functional::random_ops::randn;
/// # fn example() -> torsh_core::Result<()> {
/// let input = randn(&[2, 16, 100])?;  // batch=2, channels=16, length=100
/// let weight = randn(&[32, 16, 5])?;  // 32 output channels, kernel_size=5
/// let bias = Some(Tensor::zeros(&[32])?);
///
/// let output = conv1d(&input, &weight, bias.as_ref(), 1, 2, 1, 1)?;
/// // Output shape: [2, 32, 100] (padding=2 maintains length)
/// # Ok(())
/// # }
/// ```
pub fn conv1d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> TorshResult<Tensor> {
    input.conv1d(weight, bias, stride, padding, dilation, groups)
}

/// 2D convolution over an input image composed of several input planes.
///
/// # Mathematical Definition
/// ```text
/// out[c_out][h][w] = bias[c_out] +
///     Σ(c_in, kh, kw) weight[c_out][c_in][kh][kw] *
///                     input[c_in][h*s_h + kh*d_h - p_h][w*s_w + kw*d_w - p_w]
/// ```
///
/// # Arguments
/// * `input` - Input tensor of shape `[N, C_in, H, W]`
/// * `weight` - Convolution kernel of shape `[C_out, C_in/groups, K_h, K_w]`
/// * `bias` - Optional bias tensor of shape `[C_out]`
/// * `stride` - Stride of the convolution `(stride_h, stride_w)`
/// * `padding` - Zero padding added to both sides `(padding_h, padding_w)`
/// * `dilation` - Spacing between kernel elements `(dilation_h, dilation_w)`
/// * `groups` - Number of blocked connections from input to output channels
///
/// # Shape
/// - Input: `[N, C_in, H, W]`
/// - Weight: `[C_out, C_in/groups, K_h, K_w]`
/// - Bias: `[C_out]` (optional)
/// - Output: `[N, C_out, H_out, W_out]` where:
///   - `H_out = floor((H + 2*padding_h - dilation_h*(K_h-1) - 1) / stride_h) + 1`
///   - `W_out = floor((W + 2*padding_w - dilation_w*(K_w-1) - 1) / stride_w) + 1`
///
/// # Performance Notes
/// - Computational complexity: O(N * C_in * C_out * K_h * K_w * H_out * W_out / groups)
/// - Memory usage scales with batch size and output spatial dimensions
/// - For large kernels (K > 5), consider using FFT-based convolution
/// - Grouped convolutions reduce computation by factor of 1/groups
///
/// # Examples
/// ```rust,no_run
/// # use torsh_tensor::Tensor;
/// # use torsh_functional::conv::conv2d;
/// # use torsh_functional::random_ops::randn;
/// # fn example() -> torsh_core::Result<()> {
/// // Standard convolution for image classification
/// let input = randn(&[8, 3, 224, 224])?;    // ImageNet-like input
/// let weight = randn(&[64, 3, 7, 7])?;      // First layer kernel
/// let bias = Some(Tensor::zeros(&[64])?);
///
/// let output = conv2d(
///     &input,
///     &weight,
///     bias.as_ref(),
///     (2, 2),  // stride=2 reduces spatial dimensions by half
///     (3, 3),  // padding=3 for kernel_size=7
///     (1, 1),  // standard dilation
///     1,       // no grouping
/// )?;
/// // Output shape: [8, 64, 112, 112]
/// # Ok(())
/// # }
/// ```
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> TorshResult<Tensor> {
    input.conv2d(weight, bias, stride, padding, dilation, groups)
}

/// 3D convolution over a volumetric input composed of several input planes.
///
/// # Mathematical Definition
/// ```text
/// out[c_out][d][h][w] = bias[c_out] +
///     Σ(c_in, kd, kh, kw) weight[c_out][c_in][kd][kh][kw] *
///                         input[c_in][d*s_d + kd*dil_d - p_d]
///                                    [h*s_h + kh*dil_h - p_h]
///                                    [w*s_w + kw*dil_w - p_w]
/// ```
///
/// # Arguments
/// * `input` - Input tensor of shape `[N, C_in, D, H, W]`
/// * `weight` - Convolution kernel of shape `[C_out, C_in/groups, K_d, K_h, K_w]`
/// * `bias` - Optional bias tensor of shape `[C_out]`
/// * `stride` - Stride of the convolution `(stride_d, stride_h, stride_w)`
/// * `padding` - Zero padding added to all sides `(padding_d, padding_h, padding_w)`
/// * `dilation` - Spacing between kernel elements `(dilation_d, dilation_h, dilation_w)`
/// * `groups` - Number of blocked connections from input to output channels
///
/// # Shape
/// - Input: `[N, C_in, D, H, W]`
/// - Weight: `[C_out, C_in/groups, K_d, K_h, K_w]`
/// - Bias: `[C_out]` (optional)
/// - Output: `[N, C_out, D_out, H_out, W_out]` where each dimension follows:
///   - `D_out = floor((D + 2*padding_d - dilation_d*(K_d-1) - 1) / stride_d) + 1`
///   - `H_out = floor((H + 2*padding_h - dilation_h*(K_h-1) - 1) / stride_h) + 1`
///   - `W_out = floor((W + 2*padding_w - dilation_w*(K_w-1) - 1) / stride_w) + 1`
///
/// # Applications
/// - **Video processing**: Temporal convolutions across video frames
/// - **Medical imaging**: 3D CT/MRI scan analysis
/// - **Point cloud processing**: Volumetric deep learning
/// - **Action recognition**: Spatio-temporal feature extraction
///
/// # Performance Notes
/// - Computational complexity: O(N * C_in * C_out * K_d * K_h * K_w * D_out * H_out * W_out / groups)
/// - Memory intensive due to 3D spatial dimensions
/// - Consider using (2+1)D convolutions for video: separate spatial and temporal convolutions
/// - Grouped convolutions particularly beneficial for 3D due to high computational cost
///
/// # Examples
/// ```rust,no_run
/// # use torsh_tensor::Tensor;
/// # use torsh_functional::conv::conv3d;
/// # use torsh_functional::random_ops::randn;
/// # fn example() -> torsh_core::Result<()> {
/// // Video classification: 16-frame clips
/// let input = randn(&[4, 3, 16, 112, 112])?;  // batch=4, RGB, 16 frames, 112x112
/// let weight = randn(&[64, 3, 3, 3, 3])?;     // 3x3x3 kernel
/// let bias = Some(Tensor::zeros(&[64])?);
///
/// let output = conv3d(
///     &input,
///     &weight,
///     bias.as_ref(),
///     (1, 1, 1),  // unit stride
///     (1, 1, 1),  // same padding
///     (1, 1, 1),  // standard dilation
///     1,          // no grouping
/// )?;
/// // Output shape: [4, 64, 16, 112, 112]
/// # Ok(())
/// # }
/// ```
pub fn conv3d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
) -> TorshResult<Tensor> {
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
