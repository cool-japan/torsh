//! Adaptive pooling operations that produce fixed output sizes

use crate::utils::{function_context, validate_tensor_dims};
use torsh_core::Result as TorshResult;
use torsh_tensor::Tensor;

/// Adaptive 1D max pooling
pub fn adaptive_max_pool1d(input: &Tensor, output_size: usize) -> TorshResult<Tensor> {
    let context = function_context("adaptive_max_pool1d");
    validate_tensor_dims(input, 3, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let length = dims[2];

    let mut output_data = vec![f32::NEG_INFINITY; batch_size * channels * output_size];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for ol in 0..output_size {
                let out_idx = (b * channels + c) * output_size + ol;

                // Calculate input range for this output position
                let start = (ol * length) / output_size;
                let end = ((ol + 1) * length) / output_size;

                let mut max_val = f32::NEG_INFINITY;

                for il in start..end {
                    let in_idx = (b * channels + c) * length + il;
                    let val = input_data[in_idx];
                    if val > max_val {
                        max_val = val;
                    }
                }

                output_data[out_idx] = max_val;
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, output_size],
        input.device(),
    )
}

/// Adaptive 2D max pooling
pub fn adaptive_max_pool2d(input: &Tensor, output_size: (usize, usize)) -> TorshResult<Tensor> {
    let context = function_context("adaptive_max_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let (out_height, out_width) = output_size;
    let output_elements = batch_size * channels * out_height * out_width;
    let mut output_data = vec![f32::NEG_INFINITY; output_elements];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;

                    // Calculate input ranges
                    let h_start = (oh * height) / out_height;
                    let h_end = ((oh + 1) * height) / out_height;
                    let w_start = (ow * width) / out_width;
                    let w_end = ((ow + 1) * width) / out_width;

                    let mut max_val = f32::NEG_INFINITY;

                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let in_idx = ((b * channels + c) * height + ih) * width + iw;
                            let val = input_data[in_idx];
                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }

                    output_data[out_idx] = max_val;
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

/// Adaptive 3D max pooling
pub fn adaptive_max_pool3d(
    input: &Tensor,
    output_size: (usize, usize, usize),
) -> TorshResult<Tensor> {
    let context = function_context("adaptive_max_pool3d");
    validate_tensor_dims(input, 5, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let depth = dims[2];
    let height = dims[3];
    let width = dims[4];

    let (out_depth, out_height, out_width) = output_size;
    let output_elements = batch_size * channels * out_depth * out_height * out_width;
    let mut output_data = vec![f32::NEG_INFINITY; output_elements];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_idx = (((b * channels + c) * out_depth + od) * out_height + oh)
                            * out_width
                            + ow;

                        // Calculate input ranges
                        let d_start = (od * depth) / out_depth;
                        let d_end = ((od + 1) * depth) / out_depth;
                        let h_start = (oh * height) / out_height;
                        let h_end = ((oh + 1) * height) / out_height;
                        let w_start = (ow * width) / out_width;
                        let w_end = ((ow + 1) * width) / out_width;

                        let mut max_val = f32::NEG_INFINITY;

                        for id in d_start..d_end {
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let in_idx = (((b * channels + c) * depth + id) * height + ih)
                                        * width
                                        + iw;
                                    let val = input_data[in_idx];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        output_data[out_idx] = max_val;
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

/// Adaptive 1D average pooling
pub fn adaptive_avg_pool1d(input: &Tensor, output_size: usize) -> TorshResult<Tensor> {
    let context = function_context("adaptive_avg_pool1d");
    validate_tensor_dims(input, 3, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let length = dims[2];

    let mut output_data = vec![0.0f32; batch_size * channels * output_size];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for ol in 0..output_size {
                let out_idx = (b * channels + c) * output_size + ol;

                // Calculate input range for this output position
                let start = (ol * length) / output_size;
                let end = ((ol + 1) * length) / output_size;

                let mut sum = 0.0f32;
                let count = end - start;

                for il in start..end {
                    let in_idx = (b * channels + c) * length + il;
                    sum += input_data[in_idx];
                }

                output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }
    }

    Tensor::from_data(
        output_data,
        vec![batch_size, channels, output_size],
        input.device(),
    )
}

/// Adaptive 2D average pooling
pub fn adaptive_avg_pool2d(input: &Tensor, output_size: (usize, usize)) -> TorshResult<Tensor> {
    let context = function_context("adaptive_avg_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let (out_height, out_width) = output_size;
    let output_elements = batch_size * channels * out_height * out_width;
    let mut output_data = vec![0.0f32; output_elements];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;

                    // Calculate input ranges
                    let h_start = (oh * height) / out_height;
                    let h_end = ((oh + 1) * height) / out_height;
                    let w_start = (ow * width) / out_width;
                    let w_end = ((ow + 1) * width) / out_width;

                    let mut sum = 0.0f32;
                    let count = (h_end - h_start) * (w_end - w_start);

                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let in_idx = ((b * channels + c) * height + ih) * width + iw;
                            sum += input_data[in_idx];
                        }
                    }

                    output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
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

/// Adaptive 3D average pooling
pub fn adaptive_avg_pool3d(
    input: &Tensor,
    output_size: (usize, usize, usize),
) -> TorshResult<Tensor> {
    let context = function_context("adaptive_avg_pool3d");
    validate_tensor_dims(input, 5, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let depth = dims[2];
    let height = dims[3];
    let width = dims[4];

    let (out_depth, out_height, out_width) = output_size;
    let output_elements = batch_size * channels * out_depth * out_height * out_width;
    let mut output_data = vec![0.0f32; output_elements];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_idx = (((b * channels + c) * out_depth + od) * out_height + oh)
                            * out_width
                            + ow;

                        // Calculate input ranges
                        let d_start = (od * depth) / out_depth;
                        let d_end = ((od + 1) * depth) / out_depth;
                        let h_start = (oh * height) / out_height;
                        let h_end = ((oh + 1) * height) / out_height;
                        let w_start = (ow * width) / out_width;
                        let w_end = ((ow + 1) * width) / out_width;

                        let mut sum = 0.0f32;
                        let count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);

                        for id in d_start..d_end {
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let in_idx = (((b * channels + c) * depth + id) * height + ih)
                                        * width
                                        + iw;
                                    sum += input_data[in_idx];
                                }
                            }
                        }

                        output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
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
