//! Sparse tensor convolution operations
//!
//! This module provides convolution operations optimized for sparse tensors,
//! including 1D and 2D convolutions with support for padding, stride, and dilation.

use crate::sparse::core::SparseTensor;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Sparse 1D convolution
///
/// Performs 1D convolution on sparse input tensors with dense kernels.
/// This is efficient for sparse inputs as it only processes non-zero elements.
///
/// # Mathematical Formula
/// For input x and kernel w:
/// `y[b, i] = Σ(x[b, i + k*d - p] * w[o, k]) + bias[o]`
/// where b=batch, i=output position, k=kernel position, d=dilation, p=padding, o=output channel
///
/// # Arguments
/// * `input` - Sparse input tensor \[batch_size, input_length\]
/// * `weight` - Dense weight tensor \[out_channels, kernel_size\]
/// * `bias` - Optional bias tensor \[out_channels\]
/// * `stride` - Convolution stride
/// * `padding` - Zero padding
/// * `dilation` - Kernel dilation
///
/// # Returns
/// Sparse output tensor after convolution
pub fn sparse_conv1d(
    input: &SparseTensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> TorshResult<SparseTensor> {
    if input.ndim != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Input must be 2D tensor [batch_size, input_length]",
            "sparse_conv1d",
        ));
    }

    let weight_shape_binding = weight.shape();
    let weight_shape = weight_shape_binding.dims();
    if weight_shape.len() != 2 {
        return Err(TorshError::invalid_argument_with_context(
            "Weight must be 2D tensor [out_channels, kernel_size]",
            "sparse_conv1d",
        ));
    }

    let batch_size = input.shape[0];
    let input_length = input.shape[1];
    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[1];

    // Calculate output length
    let output_length =
        (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    let mut result_values = Vec::new();
    let mut result_indices = Vec::new();

    let input_values = input.values.to_vec()?;
    let input_indices = input.indices.to_vec()?;
    let weight_data = weight.to_vec()?;

    // For each non-zero input element
    for i in 0..input.nnz {
        let batch_idx = input_indices[i] as usize;
        let in_pos = input_indices[input.nnz + i] as usize;
        let input_val = input_values[i];

        // Apply convolution kernel
        for out_ch in 0..out_channels {
            for k in 0..kernel_size {
                let in_idx = in_pos + padding;
                if in_idx >= k * dilation && (in_idx - k * dilation) % stride == 0 {
                    let out_pos = (in_idx - k * dilation) / stride;
                    if out_pos < output_length {
                        let weight_val = weight_data[out_ch * kernel_size + k];
                        let conv_val = input_val * weight_val;

                        if conv_val.abs() > 1e-8 {
                            result_values.push(conv_val);
                            result_indices.push(batch_idx as f32);
                            result_indices.push(out_pos as f32);
                        }
                    }
                }
            }
        }
    }

    // Add bias if present
    if let Some(bias_tensor) = bias {
        let bias_data = bias_tensor.to_vec()?;
        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for pos in 0..output_length {
                    if bias_data[out_ch].abs() > 1e-8 {
                        result_values.push(bias_data[out_ch]);
                        result_indices.push(batch as f32);
                        result_indices.push(pos as f32);
                    }
                }
            }
        }
    }

    let nnz = result_values.len();
    let values = Tensor::from_data(result_values, vec![nnz], input.values.device())?;
    let indices = Tensor::from_data(result_indices, vec![2, nnz], input.indices.device())?;
    let shape = vec![batch_size, output_length];

    let mut result = SparseTensor::new(values, indices, shape)?;
    result.coalesce()?;
    Ok(result)
}

/// Sparse 2D convolution
///
/// Performs 2D convolution on sparse input tensors with dense kernels.
/// Optimized for sparse inputs by only processing non-zero elements.
///
/// # Mathematical Formula
/// For input x and kernel w:
/// `y[b, o, h, w] = Σ(x[b, i, h + kh*dh - ph, w + kw*dw - pw] * w[o, i, kh, kw]) + bias[o]`
/// where b=batch, o=output channel, i=input channel, h,w=spatial positions, kh,kw=kernel positions
///
/// # Arguments
/// * `input` - Sparse input tensor \[batch_size, channels, height, width\]
/// * `weight` - Dense weight tensor \[out_channels, in_channels, kernel_height, kernel_width\]
/// * `bias` - Optional bias tensor \[out_channels\]
/// * `stride` - Convolution stride (height, width)
/// * `padding` - Zero padding (height, width)
/// * `dilation` - Kernel dilation (height, width)
///
/// # Returns
/// Sparse output tensor after convolution
pub fn sparse_conv2d(
    input: &SparseTensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> TorshResult<SparseTensor> {
    if input.ndim != 4 {
        return Err(TorshError::invalid_argument_with_context(
            "Input must be 4D tensor [batch_size, channels, height, width]",
            "sparse_conv2d",
        ));
    }

    let weight_shape_binding = weight.shape();
    let weight_shape = weight_shape_binding.dims();
    if weight_shape.len() != 4 {
        return Err(TorshError::invalid_argument_with_context(
            "Weight must be 4D tensor [out_channels, in_channels, kernel_height, kernel_width]",
            "sparse_conv2d",
        ));
    }

    let batch_size = input.shape[0];
    let in_channels = input.shape[1];
    let in_height = input.shape[2];
    let in_width = input.shape[3];

    let out_channels = weight_shape[0];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    // Calculate output dimensions
    let out_height = (in_height + 2 * padding.0 - dilation.0 * (kernel_h - 1) - 1) / stride.0 + 1;
    let out_width = (in_width + 2 * padding.1 - dilation.1 * (kernel_w - 1) - 1) / stride.1 + 1;

    let mut result_values = Vec::new();
    let mut result_indices = Vec::new();

    let input_values = input.values.to_vec()?;
    let input_indices = input.indices.to_vec()?;
    let weight_data = weight.to_vec()?;

    // For each non-zero input element
    for i in 0..input.nnz {
        let batch_idx = input_indices[i] as usize;
        let in_ch = input_indices[input.nnz + i] as usize;
        let in_h = input_indices[2 * input.nnz + i] as usize;
        let in_w = input_indices[3 * input.nnz + i] as usize;
        let input_val = input_values[i];

        // Apply convolution kernel
        for out_ch in 0..out_channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let h_idx = in_h + padding.0;
                    let w_idx = in_w + padding.1;

                    if h_idx >= kh * dilation.0
                        && w_idx >= kw * dilation.1
                        && (h_idx - kh * dilation.0) % stride.0 == 0
                        && (w_idx - kw * dilation.1) % stride.1 == 0
                    {
                        let out_h = (h_idx - kh * dilation.0) / stride.0;
                        let out_w = (w_idx - kw * dilation.1) / stride.1;

                        if out_h < out_height && out_w < out_width {
                            let weight_idx = out_ch * (in_channels * kernel_h * kernel_w)
                                + in_ch * (kernel_h * kernel_w)
                                + kh * kernel_w
                                + kw;
                            let weight_val = weight_data[weight_idx];
                            let conv_val = input_val * weight_val;

                            if conv_val.abs() > 1e-8 {
                                result_values.push(conv_val);
                                result_indices.push(batch_idx as f32);
                                result_indices.push(out_ch as f32);
                                result_indices.push(out_h as f32);
                                result_indices.push(out_w as f32);
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias if present
    if let Some(bias_tensor) = bias {
        let bias_data = bias_tensor.to_vec()?;
        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        if bias_data[out_ch].abs() > 1e-8 {
                            result_values.push(bias_data[out_ch]);
                            result_indices.push(batch as f32);
                            result_indices.push(out_ch as f32);
                            result_indices.push(h as f32);
                            result_indices.push(w as f32);
                        }
                    }
                }
            }
        }
    }

    let nnz = result_values.len();
    let values = Tensor::from_data(result_values, vec![nnz], input.values.device())?;
    let indices = Tensor::from_data(result_indices, vec![4, nnz], input.indices.device())?;
    let shape = vec![batch_size, out_channels, out_height, out_width];

    let mut result = SparseTensor::new(values, indices, shape)?;
    result.coalesce()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::core::sparse_coo_tensor;

    #[test]
    fn test_sparse_conv1d() -> TorshResult<()> {
        // Create a simple 1D sparse tensor
        let values = Tensor::from_data(vec![1.0, 2.0], vec![2], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 0.0, 1.0, 3.0],
            vec![2, 2],
            torsh_core::DeviceType::Cpu,
        )?;
        let shape = vec![1, 5]; // batch_size=1, length=5

        let sparse_input = sparse_coo_tensor(&indices, &values, &shape)?;

        // Create a simple weight tensor
        let weight = Tensor::from_data(vec![0.5, 0.3], vec![1, 2], torsh_core::DeviceType::Cpu)?;

        // Test convolution
        let result = sparse_conv1d(&sparse_input, &weight, None, 1, 0, 1)?;

        // Verify result shape
        assert_eq!(result.shape(), &[1, 4]); // output length = 5 - 2 + 1 = 4

        Ok(())
    }

    #[test]
    fn test_sparse_conv2d_simple() -> TorshResult<()> {
        // Create a simple 2D sparse tensor [1, 1, 3, 3] with one non-zero element
        let values = Tensor::from_data(vec![1.0], vec![1], torsh_core::DeviceType::Cpu)?;
        let indices = Tensor::from_data(
            vec![0.0, 0.0, 1.0, 1.0], // [batch=0, channel=0, h=1, w=1]
            vec![4, 1],
            torsh_core::DeviceType::Cpu,
        )?;
        let shape = vec![1, 1, 3, 3];

        let sparse_input = sparse_coo_tensor(&indices, &values, &shape)?;

        // Create a simple 2x2 kernel
        let weight = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2], // [out_ch=1, in_ch=1, h=2, w=2]
            torsh_core::DeviceType::Cpu,
        )?;

        // Test convolution with stride=1, padding=0
        let result = sparse_conv2d(&sparse_input, &weight, None, (1, 1), (0, 0), (1, 1))?;

        // Verify result shape: (3-2+1, 3-2+1) = (2, 2)
        assert_eq!(result.shape(), &[1, 1, 2, 2]);

        Ok(())
    }
}
