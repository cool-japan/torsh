//! Multi-threaded execution for large tensor operations
//!
//! This module provides parallel implementations of tensor operations
//! for improved performance on large tensors.
//!
//! **SciRS2 POLICY COMPLIANCE**: Uses `scirs2_core::parallel_ops` for all
//! parallel operations instead of direct rayon imports.

// ✅ SciRS2 POLICY: Use scirs2_core::parallel_ops instead of direct rayon
use scirs2_core::parallel_ops::*;
use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;
// use std::sync::Arc;

// Import error type that's not re-exported by scirs2_core::parallel_ops
// This is a necessary exception for the init_thread_pool function
use rayon::ThreadPoolBuildError;

/// Configuration for parallel execution
#[derive(Clone, Debug)]
pub struct ParallelConfig {
    /// Minimum tensor size to use parallel execution
    pub size_threshold: usize,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Maximum number of threads to use (None = use all available)
    pub max_threads: Option<usize>,
    /// Whether to use adaptive chunk sizing
    pub adaptive_chunking: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            size_threshold: 1000,
            chunk_size: 1000,
            max_threads: None,
            adaptive_chunking: true,
        }
    }
}

/// Initialize the thread pool with custom configuration
///
/// **SciRS2 POLICY**: Uses re-exported types from scirs2_core::parallel_ops
#[allow(dead_code)]
pub fn init_thread_pool(config: &ParallelConfig) -> Result<(), ThreadPoolBuildError> {
    let mut builder = ThreadPoolBuilder::new();

    if let Some(max_threads) = config.max_threads {
        builder = builder.num_threads(max_threads);
    }

    let pool = builder.build_global();
    pool
}

/// Determine optimal chunk size based on data size and thread count
///
/// **SciRS2 POLICY**: Uses re-exported functions from scirs2_core::parallel_ops
fn optimal_chunk_size(data_size: usize, config: &ParallelConfig) -> usize {
    if !config.adaptive_chunking {
        return config.chunk_size;
    }

    let num_threads = current_num_threads();
    let chunks_per_thread = 4; // Aim for 4 chunks per thread for good load balancing
    let optimal_chunks = num_threads * chunks_per_thread;
    let optimal_size = (data_size / optimal_chunks).max(config.chunk_size);

    optimal_size
}

/// Parallel element-wise operation on a single tensor
#[allow(dead_code)]
pub fn parallel_elementwise<F>(
    input: &Tensor,
    operation: F,
    config: Option<ParallelConfig>,
) -> TorshResult<Tensor>
where
    F: Fn(f32) -> f32 + Send + Sync,
{
    let config = config.unwrap_or_default();
    let data = input.data()?;

    if data.len() < config.size_threshold {
        // Use sequential processing for small tensors
        let result_data: Vec<f32> = data.iter().map(|&x| operation(x)).collect();
        return Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device());
    }

    let chunk_size = optimal_chunk_size(data.len(), &config);

    let result_data: Vec<f32> = data
        .par_chunks(chunk_size)
        .flat_map(|chunk| chunk.iter().map(|&x| operation(x)).collect::<Vec<_>>())
        .collect();

    Tensor::from_data(result_data, input.shape().dims().to_vec(), input.device())
}

/// Parallel element-wise operation on two tensors
#[allow(dead_code)]
pub fn parallel_elementwise_binary<F>(
    a: &Tensor,
    b: &Tensor,
    operation: F,
    config: Option<ParallelConfig>,
) -> TorshResult<Tensor>
where
    F: Fn(f32, f32) -> f32 + Send + Sync,
{
    let config = config.unwrap_or_default();

    if a.shape() != b.shape() {
        return Err(torsh_core::TorshError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }

    let data_a = a.data()?;
    let data_b = b.data()?;

    if data_a.len() < config.size_threshold {
        // Use sequential processing for small tensors
        let result_data: Vec<f32> = data_a
            .iter()
            .zip(data_b.iter())
            .map(|(&x, &y)| operation(x, y))
            .collect();
        return Ok(Tensor::from_data(
            result_data,
            a.shape().dims().to_vec(),
            a.device(),
        )?);
    }

    let chunk_size = optimal_chunk_size(data_a.len(), &config);

    let result_data: Vec<f32> = data_a
        .par_chunks(chunk_size)
        .zip(data_b.par_chunks(chunk_size))
        .flat_map(|(chunk_a, chunk_b)| {
            chunk_a
                .iter()
                .zip(chunk_b.iter())
                .map(|(&x, &y)| operation(x, y))
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(Tensor::from_data(
        result_data,
        a.shape().dims().to_vec(),
        a.device(),
    )?)
}

/// Parallel reduction operation
#[allow(dead_code)]
pub fn parallel_reduce<F, R>(
    input: &Tensor,
    identity: R,
    map_op: F,
    reduce_op: fn(R, R) -> R,
    config: Option<ParallelConfig>,
) -> TorshResult<R>
where
    F: Fn(f32) -> R + Send + Sync,
    R: Send + Copy + Sync,
{
    let config = config.unwrap_or_default();
    let data = input.data()?;

    if data.len() < config.size_threshold {
        // Use sequential processing for small tensors
        return Ok(data.iter().map(|&x| map_op(x)).fold(identity, reduce_op));
    }

    let chunk_size = optimal_chunk_size(data.len(), &config);

    let result = data
        .par_chunks(chunk_size)
        .map(|chunk| chunk.iter().map(|&x| map_op(x)).fold(identity, reduce_op))
        .reduce(|| identity, reduce_op);

    Ok(result)
}

/// Parallel matrix multiplication for 2D tensors
#[allow(dead_code)]
pub fn parallel_matmul(
    a: &Tensor,
    b: &Tensor,
    config: Option<ParallelConfig>,
) -> TorshResult<Tensor> {
    let config = config.unwrap_or_default();

    // Check shapes
    let a_shape_ref = a.shape();
    let a_shape = a_shape_ref.dims();
    let b_shape_ref = b.shape();
    let b_shape = b_shape_ref.dims();

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(torsh_core::TorshError::InvalidArgument(
            "parallel_matmul requires 2D tensors".to_string(),
        ));
    }

    if a_shape[1] != b_shape[0] {
        return Err(torsh_core::TorshError::ShapeMismatch {
            expected: vec![a_shape[0], a_shape[1]],
            got: b_shape.to_vec(),
        });
    }

    let m = a_shape[0];
    let n = b_shape[1];
    let k = a_shape[1];

    let data_a = a.data()?;
    let data_b = b.data()?;

    // Use parallel processing for large matrices (simplified)
    if m * n >= config.size_threshold {
        // For now, use a simpler approach that avoids complex nested parallel/sequential iterator mixing
        let mut result_data = vec![0.0f32; m * n];

        result_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += data_a[i * k + kk] * data_b[kk * n + j];
                    }
                    row[j] = sum;
                }
            });

        Ok(Tensor::from_data(result_data, vec![m, n], a.device())?)
    } else {
        // Fall back to standard implementation
        a.matmul(b)
    }
}

/// Parallel convolution operation (simplified)
#[allow(dead_code)]
pub fn parallel_conv2d_simple(
    input: &Tensor,
    weight: &Tensor,
    stride: [usize; 2],
    padding: [usize; 2],
    config: Option<ParallelConfig>,
) -> TorshResult<Tensor> {
    let config = config.unwrap_or_default();

    let input_shape_ref = input.shape();
    let input_shape = input_shape_ref.dims();
    let weight_shape_ref = weight.shape();
    let weight_shape = weight_shape_ref.dims();

    // Simplified shape checks (assuming NCHW format)
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return Err(torsh_core::TorshError::InvalidArgument(
            "parallel_conv2d requires 4D tensors".to_string(),
        ));
    }

    let batch_size = input_shape[0];
    let out_channels = weight_shape[0];
    let in_channels = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];

    let output_h = (input_h + 2 * padding[0] - kernel_h) / stride[0] + 1;
    let output_w = (input_w + 2 * padding[1] - kernel_w) / stride[1] + 1;

    let output_size = batch_size * out_channels * output_h * output_w;

    // Use parallel processing for large convolutions (simplified)
    if output_size >= config.size_threshold {
        // Simplified parallel convolution using chunked processing
        let input_data = input.data()?;
        let weight_data = weight.data()?;

        let mut result_data = vec![0.0f32; output_size];
        let elements_per_batch = out_channels * output_h * output_w;

        result_data
            .par_chunks_mut(elements_per_batch)
            .enumerate()
            .for_each(|(b, batch_output)| {
                for oc in 0..out_channels {
                    for oh in 0..output_h {
                        for ow in 0..output_w {
                            let mut sum = 0.0;
                            for ic in 0..in_channels {
                                for kh in 0..kernel_h {
                                    for kw in 0..kernel_w {
                                        let ih = oh * stride[0] + kh;
                                        let iw = ow * stride[1] + kw;

                                        if ih < padding[0]
                                            || ih >= input_h + padding[0]
                                            || iw < padding[1]
                                            || iw >= input_w + padding[1]
                                        {
                                            continue; // Padding
                                        }

                                        let ih = ih - padding[0];
                                        let iw = iw - padding[1];

                                        let input_idx = b * in_channels * input_h * input_w
                                            + ic * input_h * input_w
                                            + ih * input_w
                                            + iw;
                                        let weight_idx = oc * in_channels * kernel_h * kernel_w
                                            + ic * kernel_h * kernel_w
                                            + kh * kernel_w
                                            + kw;

                                        sum += input_data[input_idx] * weight_data[weight_idx];
                                    }
                                }
                            }
                            let output_idx = oc * output_h * output_w + oh * output_w + ow;
                            batch_output[output_idx] = sum;
                        }
                    }
                }
            });

        Ok(Tensor::from_data(
            result_data,
            vec![batch_size, out_channels, output_h, output_w],
            input.device(),
        )?)
    } else {
        // Fall back to standard implementation
        crate::conv::conv2d(
            input,
            weight,
            None,
            (stride[0], stride[1]),
            (padding[0], padding[1]),
            (1, 1),
            1,
        )
    }
}

/// Parallel softmax computation
#[allow(dead_code)]
pub fn parallel_softmax(
    input: &Tensor,
    dim: i32,
    config: Option<ParallelConfig>,
) -> TorshResult<Tensor> {
    let config = config.unwrap_or_default();
    let shape_ref = input.shape();
    let shape = shape_ref.dims();
    let data = input.data()?;

    // Only parallelize for large tensors
    if data.len() < config.size_threshold {
        return input.softmax(dim);
    }

    // Simplified parallel softmax along last dimension for now
    if dim == -1 || dim as usize == shape.len() - 1 {
        let last_dim_size = shape[shape.len() - 1];
        let _outer_size = data.len() / last_dim_size;

        let mut result_data = vec![0.0f32; data.len()];

        result_data
            .par_chunks_mut(last_dim_size)
            .enumerate()
            .for_each(|(i, output_slice)| {
                let start_idx = i * last_dim_size;
                let input_slice = &data[start_idx..start_idx + last_dim_size];

                // Find max for numerical stability
                let max_val = input_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Compute exp(x - max) and sum
                let mut sum = 0.0f32;
                for (j, &x) in input_slice.iter().enumerate() {
                    let exp_val = (x - max_val).exp();
                    output_slice[j] = exp_val;
                    sum += exp_val;
                }

                // Normalize
                for val in output_slice.iter_mut() {
                    *val /= sum;
                }
            });

        Ok(Tensor::from_data(
            result_data,
            shape.to_vec(),
            input.device(),
        )?)
    } else {
        // Fall back to standard implementation for other dimensions
        input.softmax(dim)
    }
}

/// Parallel activation functions with automatic optimization selection
pub mod parallel_activations {
    use super::*;

    /// Parallel ReLU
    #[allow(dead_code)]
    pub fn relu(input: &Tensor, config: Option<ParallelConfig>) -> TorshResult<Tensor> {
        parallel_elementwise(input, |x| x.max(0.0), config)
    }

    /// Parallel Leaky ReLU
    #[allow(dead_code)]
    pub fn leaky_relu(
        input: &Tensor,
        negative_slope: f32,
        config: Option<ParallelConfig>,
    ) -> TorshResult<Tensor> {
        parallel_elementwise(
            input,
            move |x| if x > 0.0 { x } else { x * negative_slope },
            config,
        )
    }

    /// Parallel ELU
    #[allow(dead_code)]
    pub fn elu(input: &Tensor, alpha: f32, config: Option<ParallelConfig>) -> TorshResult<Tensor> {
        parallel_elementwise(
            input,
            move |x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) },
            config,
        )
    }

    /// Parallel Sigmoid using lookup table for large tensors
    #[allow(dead_code)]
    pub fn sigmoid(input: &Tensor, config: Option<ParallelConfig>) -> TorshResult<Tensor> {
        let config = config.unwrap_or_default();

        // Use lookup table for very large tensors
        if input.numel() > config.size_threshold * 10 {
            crate::activation_lookup::sigmoid_lookup(input, None)
        } else {
            parallel_elementwise(input, |x| 1.0 / (1.0 + (-x).exp()), Some(config))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{ones, rand};

    #[test]
    fn test_parallel_elementwise() {
        let input = rand(&[1000]).unwrap();
        let result = parallel_elementwise(&input, |x| x * 2.0, None).unwrap();

        assert_eq!(input.shape(), result.shape());
    }

    #[test]
    fn test_parallel_binary_op() {
        let a = ones(&[500, 500]).unwrap();
        let b = ones(&[500, 500]).unwrap();
        let result = parallel_elementwise_binary(&a, &b, |x, y| x + y, None).unwrap();

        assert_eq!(a.shape(), result.shape());
    }

    #[test]
    fn test_parallel_reduce() {
        let input = ones(&[1000]).unwrap();
        let sum = parallel_reduce(&input, 0.0, |x| x, |a, b| a + b, None).unwrap();

        assert!((sum - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_optimal_chunk_size() {
        let config = ParallelConfig::default();
        let chunk_size = optimal_chunk_size(10000, &config);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 10000);
    }
}
