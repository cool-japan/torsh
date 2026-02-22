//! Global pooling operations that reduce spatial dimensions to scalars

use crate::utils::function_context;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Generic helper for global pooling operations
///
/// Reduces code duplication between global_avg_pool and global_max_pool by
/// providing a common pattern for global spatial reduction.
///
/// # Parameters
/// - `input`: Input tensor with shape [N, C, ...spatial dims...]
/// - `operation_name`: Name of the operation for error context
/// - `init_value`: Initial value for the accumulator
/// - `reduce_fn`: Reduction function applied to each spatial element
/// - `finalize_fn`: Final transformation applied to accumulated value
///
/// # Returns
/// Tensor with shape [N, C] containing reduced values
fn global_pool_generic<F, G>(
    input: &Tensor,
    operation_name: &str,
    init_value: f32,
    reduce_fn: F,
    finalize_fn: G,
) -> TorshResult<Tensor>
where
    F: Fn(f32, f32) -> f32,
    G: Fn(f32, usize) -> f32,
{
    let context = function_context(operation_name);

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
    let spatial_size: usize = dims[2..].iter().product();

    let input_data = input.to_vec()?;
    let mut output_data = vec![init_value; batch_size * channels];

    for b in 0..batch_size {
        for c in 0..channels {
            let mut accumulator = init_value;

            for s in 0..spatial_size {
                let idx = (b * channels + c) * spatial_size + s;
                accumulator = reduce_fn(accumulator, input_data[idx]);
            }

            let out_idx = b * channels + c;
            output_data[out_idx] = finalize_fn(accumulator, spatial_size);
        }
    }

    Tensor::from_data(output_data, vec![batch_size, channels], input.device())
}

/// Global average pooling - averages over all spatial dimensions
///
/// Computes the mean of all spatial dimensions, producing a [N, C] output
/// from a [N, C, ...spatial...] input.
///
/// # Examples
/// ```rust
/// use torsh_functional::pooling::global_avg_pool;
/// use torsh_tensor::creation::randn;
///
/// let input = randn::<f32>(&[2, 3, 28, 28])?; // [N, C, H, W]
/// let output = global_avg_pool(&input)?;       // [N, C]
/// assert_eq!(output.shape().dims(), &[2, 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn global_avg_pool(input: &Tensor) -> TorshResult<Tensor> {
    global_pool_generic(
        input,
        "global_avg_pool",
        0.0,                             // Initial value for sum
        |acc, val| acc + val,            // Accumulate sum
        |sum, count| sum / count as f32, // Divide by count for average
    )
}

/// Global max pooling - takes maximum over all spatial dimensions
///
/// Computes the maximum value across all spatial dimensions, producing a [N, C]
/// output from a [N, C, ...spatial...] input.
///
/// # Examples
/// ```rust
/// use torsh_functional::pooling::global_max_pool;
/// use torsh_tensor::creation::randn;
///
/// let input = randn::<f32>(&[2, 3, 28, 28])?; // [N, C, H, W]
/// let output = global_max_pool(&input)?;       // [N, C]
/// assert_eq!(output.shape().dims(), &[2, 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn global_max_pool(input: &Tensor) -> TorshResult<Tensor> {
    global_pool_generic(
        input,
        "global_max_pool",
        f32::NEG_INFINITY,         // Initial value for max
        |acc, val| acc.max(val),   // Find maximum
        |max_val, _count| max_val, // Return max value unchanged
    )
}
