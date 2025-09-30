//! Global pooling operations that reduce spatial dimensions to scalars

use crate::utils::function_context;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Global average pooling - averages over all spatial dimensions
pub fn global_avg_pool(input: &Tensor) -> TorshResult<Tensor> {
    let context = function_context("global_avg_pool");

    let shape = input.shape();
    let dims = shape.dims();

    if dims.len() < 3 {
        return Err(TorshError::config_error_with_context(
            "Input must have at least 3 dimensions (batch, channel, spatial)",
            &context,
        ));
    }

    let batch_size = dims[0];
    let channels = dims[1];

    // Calculate the total number of spatial elements
    let spatial_size: usize = dims[2..].iter().product();

    let input_data = input.to_vec()?;
    let mut output_data = vec![0.0f32; batch_size * channels];

    for b in 0..batch_size {
        for c in 0..channels {
            let mut sum = 0.0f32;

            for s in 0..spatial_size {
                let idx = (b * channels + c) * spatial_size + s;
                sum += input_data[idx];
            }

            let out_idx = b * channels + c;
            output_data[out_idx] = sum / spatial_size as f32;
        }
    }

    Tensor::from_data(output_data, vec![batch_size, channels], input.device())
}

/// Global max pooling - takes maximum over all spatial dimensions
pub fn global_max_pool(input: &Tensor) -> TorshResult<Tensor> {
    let context = function_context("global_max_pool");

    let shape = input.shape();
    let dims = shape.dims();

    if dims.len() < 3 {
        return Err(TorshError::config_error_with_context(
            "Input must have at least 3 dimensions (batch, channel, spatial)",
            &context,
        ));
    }

    let batch_size = dims[0];
    let channels = dims[1];

    // Calculate the total number of spatial elements
    let spatial_size: usize = dims[2..].iter().product();

    let input_data = input.to_vec()?;
    let mut output_data = vec![f32::NEG_INFINITY; batch_size * channels];

    for b in 0..batch_size {
        for c in 0..channels {
            let mut max_val = f32::NEG_INFINITY;

            for s in 0..spatial_size {
                let idx = (b * channels + c) * spatial_size + s;
                let val = input_data[idx];
                if val > max_val {
                    max_val = val;
                }
            }

            let out_idx = b * channels + c;
            output_data[out_idx] = max_val;
        }
    }

    Tensor::from_data(output_data, vec![batch_size, channels], input.device())
}
