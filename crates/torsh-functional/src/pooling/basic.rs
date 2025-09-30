//! Basic pooling operations: max and average pooling in 1D, 2D, and 3D

use crate::utils::{calculate_pooling_output_size, function_context, validate_tensor_dims};
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// 1D max pooling
#[allow(clippy::too_many_arguments)]
pub fn max_pool1d(
    input: &Tensor,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    dilation: usize,
    return_indices: bool,
) -> TorshResult<(Tensor, Option<Tensor>)> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("max_pool1d");
    validate_tensor_dims(input, 3, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let length = dims[2];

    let out_length = calculate_pooling_output_size(length, kernel_size, stride, padding, dilation);

    let mut output_data = vec![f32::NEG_INFINITY; batch_size * channels * out_length];
    let mut indices_data = if return_indices {
        Some(vec![0i64; batch_size * channels * out_length])
    } else {
        None
    };

    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for ol in 0..out_length {
                let out_idx = (b * channels + c) * out_length + ol;
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0;

                for kl in 0..kernel_size {
                    let il = ol * stride + kl * dilation;

                    if il >= padding && il < length + padding {
                        let real_il = il - padding;

                        if real_il < length {
                            let in_idx = (b * channels + c) * length + real_il;
                            let val = input_data[in_idx];

                            if val > max_val {
                                max_val = val;
                                max_idx = in_idx as i64;
                            }
                        }
                    }
                }

                output_data[out_idx] = max_val;
                if let Some(ref mut indices) = indices_data {
                    indices[out_idx] = max_idx;
                }
            }
        }
    }

    let output = Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_length],
        input.device(),
    )?;

    let indices = if let Some(indices_data) = indices_data {
        let indices_f32: Vec<f32> = indices_data.iter().map(|&idx| idx as f32).collect();
        Some(Tensor::from_data(
            indices_f32,
            vec![batch_size, channels, out_length],
            input.device(),
        )?)
    } else {
        None
    };

    Ok((output, indices))
}

/// 2D max pooling
#[allow(clippy::too_many_arguments)]
pub fn max_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    dilation: (usize, usize),
    ceil_mode: bool,
    return_indices: bool,
) -> TorshResult<(Tensor, Option<Tensor>)> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("max_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let out_height = if ceil_mode {
        ((height + 2 * padding.0 - dilation.0 * (kernel_size.0 - 1) - 1) as f32 / stride.0 as f32)
            .ceil() as usize
    } else {
        calculate_pooling_output_size(height, kernel_size.0, stride.0, padding.0, dilation.0)
    };

    let out_width = if ceil_mode {
        ((width + 2 * padding.1 - dilation.1 * (kernel_size.1 - 1) - 1) as f32 / stride.1 as f32)
            .ceil() as usize
    } else {
        calculate_pooling_output_size(width, kernel_size.1, stride.1, padding.1, dilation.1)
    };

    let output_size = batch_size * channels * out_height * out_width;
    let mut output_data = vec![f32::NEG_INFINITY; output_size];
    let mut indices_data = if return_indices {
        Some(vec![0i64; output_size])
    } else {
        None
    };

    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh * dilation.0;
                            let iw = ow * stride.1 + kw * dilation.1;

                            if ih >= padding.0
                                && ih < height + padding.0
                                && iw >= padding.1
                                && iw < width + padding.1
                            {
                                let real_ih = ih - padding.0;
                                let real_iw = iw - padding.1;

                                if real_ih < height && real_iw < width {
                                    let in_idx =
                                        ((b * channels + c) * height + real_ih) * width + real_iw;
                                    let val = input_data[in_idx];

                                    if val > max_val {
                                        max_val = val;
                                        max_idx = in_idx as i64;
                                    }
                                }
                            }
                        }
                    }

                    output_data[out_idx] = max_val;
                    if let Some(ref mut indices) = indices_data {
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }
    }

    let output = Tensor::from_data(
        output_data,
        vec![batch_size, channels, out_height, out_width],
        input.device(),
    )?;

    let indices = if let Some(indices_data) = indices_data {
        let indices_f32: Vec<f32> = indices_data.iter().map(|&idx| idx as f32).collect();
        Some(Tensor::from_data(
            indices_f32,
            vec![batch_size, channels, out_height, out_width],
            input.device(),
        )?)
    } else {
        None
    };

    Ok((output, indices))
}

/// 1D average pooling
pub fn avg_pool1d(
    input: &Tensor,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    ceil_mode: bool,
    count_include_pad: bool,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("avg_pool1d");
    validate_tensor_dims(input, 3, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let length = dims[2];

    let out_length = if ceil_mode {
        ((length + 2 * padding - kernel_size) as f32 / stride as f32).ceil() as usize + 1
    } else {
        calculate_pooling_output_size(length, kernel_size, stride, padding, 1)
    };

    let mut output_data = vec![0.0f32; batch_size * channels * out_length];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for ol in 0..out_length {
                let out_idx = (b * channels + c) * out_length + ol;
                let mut sum = 0.0f32;
                let mut count = 0;

                for kl in 0..kernel_size {
                    let il = ol * stride + kl;

                    if il >= padding && il < length + padding {
                        let real_il = il - padding;

                        if real_il < length {
                            let in_idx = (b * channels + c) * length + real_il;
                            sum += input_data[in_idx];
                            count += 1;
                        } else if count_include_pad {
                            count += 1;
                        }
                    } else if count_include_pad {
                        count += 1;
                    }
                }

                if count > 0 {
                    output_data[out_idx] = sum / count as f32;
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

/// 2D average pooling
pub fn avg_pool2d(
    input: &Tensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Option<usize>,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("avg_pool2d");
    validate_tensor_dims(input, 4, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let height = dims[2];
    let width = dims[3];

    let out_height = if ceil_mode {
        ((height + 2 * padding.0 - kernel_size.0) as f32 / stride.0 as f32).ceil() as usize + 1
    } else {
        calculate_pooling_output_size(height, kernel_size.0, stride.0, padding.0, 1)
    };

    let out_width = if ceil_mode {
        ((width + 2 * padding.1 - kernel_size.1) as f32 / stride.1 as f32).ceil() as usize + 1
    } else {
        calculate_pooling_output_size(width, kernel_size.1, stride.1, padding.1, 1)
    };

    let output_size = batch_size * channels * out_height * out_width;
    let mut output_data = vec![0.0f32; output_size];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = ((b * channels + c) * out_height + oh) * out_width + ow;
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;

                            if ih >= padding.0
                                && ih < height + padding.0
                                && iw >= padding.1
                                && iw < width + padding.1
                            {
                                let real_ih = ih - padding.0;
                                let real_iw = iw - padding.1;

                                if real_ih < height && real_iw < width {
                                    let in_idx =
                                        ((b * channels + c) * height + real_ih) * width + real_iw;
                                    sum += input_data[in_idx];
                                    count += 1;
                                } else if count_include_pad {
                                    count += 1;
                                }
                            } else if count_include_pad {
                                count += 1;
                            }
                        }
                    }

                    let divisor = divisor_override.unwrap_or(count) as f32;
                    if divisor > 0.0 {
                        output_data[out_idx] = sum / divisor;
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

/// Max pooling 3D
pub fn max_pool3d(
    input: &Tensor,
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    ceil_mode: bool,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("max_pool3d");
    validate_tensor_dims(input, 5, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let depth = dims[2];
    let height = dims[3];
    let width = dims[4];

    let effective_kernel = (
        (kernel_size.0 - 1) * dilation.0 + 1,
        (kernel_size.1 - 1) * dilation.1 + 1,
        (kernel_size.2 - 1) * dilation.2 + 1,
    );

    let out_depth = if ceil_mode {
        ((depth + 2 * padding.0 - effective_kernel.0) as f32 / stride.0 as f32).ceil() as usize + 1
    } else {
        (depth + 2 * padding.0 - effective_kernel.0) / stride.0 + 1
    };

    let out_height = if ceil_mode {
        ((height + 2 * padding.1 - effective_kernel.1) as f32 / stride.1 as f32).ceil() as usize + 1
    } else {
        (height + 2 * padding.1 - effective_kernel.1) / stride.1 + 1
    };

    let out_width = if ceil_mode {
        ((width + 2 * padding.2 - effective_kernel.2) as f32 / stride.2 as f32).ceil() as usize + 1
    } else {
        (width + 2 * padding.2 - effective_kernel.2) / stride.2 + 1
    };

    let output_size = batch_size * channels * out_depth * out_height * out_width;
    let mut output_data = vec![f32::NEG_INFINITY; output_size];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_idx = ((((b * channels + c) * out_depth + od) * out_height + oh)
                            * out_width
                            + ow);

                        for kd in 0..kernel_size.0 {
                            for kh in 0..kernel_size.1 {
                                for kw in 0..kernel_size.2 {
                                    let id = od * stride.0 + kd * dilation.0;
                                    let ih = oh * stride.1 + kh * dilation.1;
                                    let iw = ow * stride.2 + kw * dilation.2;

                                    if id >= padding.0
                                        && id < depth + padding.0
                                        && ih >= padding.1
                                        && ih < height + padding.1
                                        && iw >= padding.2
                                        && iw < width + padding.2
                                    {
                                        let real_id = id - padding.0;
                                        let real_ih = ih - padding.1;
                                        let real_iw = iw - padding.2;

                                        if real_id < depth && real_ih < height && real_iw < width {
                                            let in_idx = ((((b * channels + c) * depth + real_id)
                                                * height
                                                + real_ih)
                                                * width
                                                + real_iw);
                                            let val = input_data[in_idx];
                                            if val > output_data[out_idx] {
                                                output_data[out_idx] = val;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Handle case where no valid input was found
                        if output_data[out_idx] == f32::NEG_INFINITY {
                            output_data[out_idx] = 0.0;
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

/// Average pooling 3D
pub fn avg_pool3d(
    input: &Tensor,
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: (usize, usize, usize),
    ceil_mode: bool,
    count_include_pad: bool,
) -> TorshResult<Tensor> {
    let stride = stride.unwrap_or(kernel_size);

    let context = function_context("avg_pool3d");
    validate_tensor_dims(input, 5, &context)?;

    let shape = input.shape();
    let dims = shape.dims();
    let batch_size = dims[0];
    let channels = dims[1];
    let depth = dims[2];
    let height = dims[3];
    let width = dims[4];

    let out_depth = if ceil_mode {
        ((depth + 2 * padding.0 - kernel_size.0) as f32 / stride.0 as f32).ceil() as usize + 1
    } else {
        (depth + 2 * padding.0 - kernel_size.0) / stride.0 + 1
    };

    let out_height = if ceil_mode {
        ((height + 2 * padding.1 - kernel_size.1) as f32 / stride.1 as f32).ceil() as usize + 1
    } else {
        (height + 2 * padding.1 - kernel_size.1) / stride.1 + 1
    };

    let out_width = if ceil_mode {
        ((width + 2 * padding.2 - kernel_size.2) as f32 / stride.2 as f32).ceil() as usize + 1
    } else {
        (width + 2 * padding.2 - kernel_size.2) / stride.2 + 1
    };

    let output_size = batch_size * channels * out_depth * out_height * out_width;
    let mut output_data = vec![0.0f32; output_size];
    let input_data = input.to_vec()?;

    for b in 0..batch_size {
        for c in 0..channels {
            for od in 0..out_depth {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_idx = ((((b * channels + c) * out_depth + od) * out_height + oh)
                            * out_width
                            + ow);

                        let mut sum = 0.0f32;
                        let mut count = 0;

                        for kd in 0..kernel_size.0 {
                            for kh in 0..kernel_size.1 {
                                for kw in 0..kernel_size.2 {
                                    let id = od * stride.0 + kd;
                                    let ih = oh * stride.1 + kh;
                                    let iw = ow * stride.2 + kw;

                                    if count_include_pad
                                        || (id >= padding.0
                                            && id < depth + padding.0
                                            && ih >= padding.1
                                            && ih < height + padding.1
                                            && iw >= padding.2
                                            && iw < width + padding.2)
                                    {
                                        if id >= padding.0
                                            && id < depth + padding.0
                                            && ih >= padding.1
                                            && ih < height + padding.1
                                            && iw >= padding.2
                                            && iw < width + padding.2
                                        {
                                            let real_id = id - padding.0;
                                            let real_ih = ih - padding.1;
                                            let real_iw = iw - padding.2;

                                            if real_id < depth
                                                && real_ih < height
                                                && real_iw < width
                                            {
                                                let in_idx = ((((b * channels + c) * depth
                                                    + real_id)
                                                    * height
                                                    + real_ih)
                                                    * width
                                                    + real_iw);
                                                sum += input_data[in_idx];
                                            }
                                        }
                                        count += 1;
                                    }
                                }
                            }
                        }

                        if count > 0 {
                            output_data[out_idx] = sum / count as f32;
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
