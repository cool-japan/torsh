//! Unpooling operations for upsampling with sparse indices

use crate::utils::{function_context, validate_tensor_dims};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// 1D max unpooling
pub fn max_unpool1d(
    input: &Tensor,
    indices: &Tensor,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    output_size: Option<usize>,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("max_unpool1d");
    validate_tensor_dims(input, 3, &context)?;
    validate_tensor_dims(indices, 3, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let pooled_length = dims[2];

    // Calculate output size
    let out_length = if let Some(size) = output_size {
        size
    } else {
        (pooled_length - 1) * stride - 2 * padding + kernel_size
    };

    let mut output_data = vec![0.0f32; batch_size * channels * out_length];
    let input_data = input.to_vec()?;
    let indices_data = indices.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for pl in 0..pooled_length {
                let in_idx = (b * channels + c) * pooled_length + pl;
                let val = input_data[in_idx];

                // Convert f32 index back to usize (it was stored as f32 in max_pool)
                let orig_idx = indices_data[in_idx] as usize;

                // Calculate output position from original index
                let orig_spatial_idx = orig_idx % out_length;
                let out_idx = (b * channels + c) * out_length + orig_spatial_idx;

                if out_idx < output_data.len() {
                    output_data[out_idx] = val;
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_length],
        input.device(),
    )
}

/// 2D max unpooling
pub fn max_unpool2d(
    input: &Tensor,
    indices: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    output_size: Option<(usize, usize)>,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("max_unpool2d");
    validate_tensor_dims(input, 4, &context)?;
    validate_tensor_dims(indices, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let pooled_height = dims[2];
    let pooled_width = dims[3];

    // Calculate output size
    let (out_height, out_width) = if let Some(size) = output_size {
        size
    } else {
        (
            (pooled_height - 1) * stride.0 - 2 * padding.0 + kernel_size.0,
            (pooled_width - 1) * stride.1 - 2 * padding.1 + kernel_size.1,
        )
    };

    let output_size = batch_size * channels * out_height * out_width;
    let mut output_data = vec![0.0f32; output_size];
    let input_data = input.to_vec()?;
    let indices_data = indices.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for ph in 0..pooled_height {
                for pw in 0..pooled_width {
                    let in_idx = ((b * channels + c) * pooled_height + ph) * pooled_width + pw;
                    let val = input_data[in_idx];

                    // Convert f32 index back to usize
                    let orig_idx = indices_data[in_idx] as usize;

                    // The original index was a spatial index in the original tensor
                    let spatial_size = out_height * out_width;
                    let orig_spatial_idx = orig_idx % spatial_size;
                    let out_idx = (b * channels + c) * spatial_size + orig_spatial_idx;

                    if out_idx < output_data.len() {
                        output_data[out_idx] = val;
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

/// 3D max unpooling
pub fn max_unpool3d(
    input: &Tensor,
    indices: &Tensor,
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: (usize, usize, usize),
    output_size: Option<(usize, usize, usize)>,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("max_unpool3d");
    validate_tensor_dims(input, 5, &context)?;
    validate_tensor_dims(indices, 5, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let pooled_depth = dims[2];
    let pooled_height = dims[3];
    let pooled_width = dims[4];

    // Calculate output size
    let (out_depth, out_height, out_width) = if let Some(size) = output_size {
        size
    } else {
        (
            (pooled_depth - 1) * stride.0 - 2 * padding.0 + kernel_size.0,
            (pooled_height - 1) * stride.1 - 2 * padding.1 + kernel_size.1,
            (pooled_width - 1) * stride.2 - 2 * padding.2 + kernel_size.2,
        )
    };

    let output_elements = batch_size * channels * out_depth * out_height * out_width;
    let mut output_data = vec![0.0f32; output_elements];
    let input_data = input.to_vec()?;
    let indices_data = indices.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for pd in 0..pooled_depth {
                for ph in 0..pooled_height {
                    for pw in 0..pooled_width {
                        let in_idx = (((b * channels + c) * pooled_depth + pd) * pooled_height
                            + ph)
                            * pooled_width
                            + pw;
                        let val = input_data[in_idx];

                        // Convert f32 index back to usize
                        let orig_idx = indices_data[in_idx] as usize;

                        // The original index was a spatial index in the original tensor
                        let spatial_size = out_depth * out_height * out_width;
                        let orig_spatial_idx = orig_idx % spatial_size;
                        let out_idx = (b * channels + c) * spatial_size + orig_spatial_idx;

                        if out_idx < output_data.len() {
                            output_data[out_idx] = val;
                        }
                    }
                }
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_depth, out_height, out_width],
        input.device(),
    )
}
