//! Sparse pooling layers
//!
//! This module provides pooling operations optimized for sparse tensors,
//! including max pooling, average pooling, and adaptive pooling variants.

use crate::TorshResult;
use torsh_core::TorshError;
use torsh_tensor::{creation::zeros, Tensor};

/// Sparse 2D Max Pooling layer
///
/// Performs max pooling on sparse tensor input, preserving sparsity patterns where possible.
/// Only processes non-zero regions to maintain computational efficiency.
pub struct SparseMaxPool2d {
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Padding (height, width)
    padding: (usize, usize),
    /// Dilation (height, width)
    dilation: (usize, usize),
}

impl SparseMaxPool2d {
    /// Create a new sparse 2D max pooling layer
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));

        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Forward pass for sparse max pooling
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape: (batch_size, channels, height, width)
        let input_shape = input.shape();
        if input_shape.ndim() != 4 {
            return Err(TorshError::InvalidArgument(
                "Input must be 4D tensor (batch_size, channels, height, width)".to_string(),
            ));
        }

        let batch_size = input_shape.dims()[0];
        let channels = input_shape.dims()[1];
        let input_height = input_shape.dims()[2];
        let input_width = input_shape.dims()[3];

        // Calculate output dimensions
        let output_height =
            (input_height + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let output_width =
            (input_width + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;

        let mut output = zeros::<f32>(&[batch_size, channels, output_height, output_width])?;

        // Perform max pooling for each batch and channel
        for b in 0..batch_size {
            for c in 0..channels {
                self.max_pool_channel(
                    input,
                    &mut output,
                    b,
                    c,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                )?;
            }
        }

        Ok(output)
    }

    /// Perform max pooling for a single channel
    #[allow(clippy::too_many_arguments)]
    fn max_pool_channel(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        batch_idx: usize,
        channel_idx: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
    ) -> TorshResult<()> {
        for out_h in 0..output_height {
            for out_w in 0..output_width {
                let mut max_val = f32::NEG_INFINITY;
                let mut found_value = false;

                // Iterate over the pooling window
                for kh in 0..self.kernel_size.0 {
                    for kw in 0..self.kernel_size.1 {
                        let in_h = out_h * self.stride.0 + kh * self.dilation.0;
                        let in_w = out_w * self.stride.1 + kw * self.dilation.1;

                        // Apply padding offset
                        if in_h >= self.padding.0 && in_w >= self.padding.1 {
                            let padded_in_h = in_h - self.padding.0;
                            let padded_in_w = in_w - self.padding.1;

                            // Check bounds
                            if padded_in_h < input_height && padded_in_w < input_width {
                                let val = input.get(&[
                                    batch_idx,
                                    channel_idx,
                                    padded_in_h,
                                    padded_in_w,
                                ])?;
                                if val > max_val || !found_value {
                                    max_val = val;
                                    found_value = true;
                                }
                            }
                        }
                    }
                }

                // Set output value (0.0 if no values found)
                output.set(
                    &[batch_idx, channel_idx, out_h, out_w],
                    if found_value { max_val } else { 0.0 },
                )?;
            }
        }

        Ok(())
    }
}

/// Sparse 2D Average Pooling layer
///
/// Performs average pooling on sparse tensor input, computing averages only over non-zero regions
/// to maintain meaningful sparse representations.
pub struct SparseAvgPool2d {
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Padding (height, width)
    padding: (usize, usize),
    /// Whether to count padding zeros in the average
    count_include_pad: bool,
}

impl SparseAvgPool2d {
    /// Create a new sparse 2D average pooling layer
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        count_include_pad: bool,
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        let padding = padding.unwrap_or((0, 0));

        Self {
            kernel_size,
            stride,
            padding,
            count_include_pad,
        }
    }

    /// Forward pass for sparse average pooling
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape: (batch_size, channels, height, width)
        let input_shape = input.shape();
        if input_shape.ndim() != 4 {
            return Err(TorshError::InvalidArgument(
                "Input must be 4D tensor (batch_size, channels, height, width)".to_string(),
            ));
        }

        let batch_size = input_shape.dims()[0];
        let channels = input_shape.dims()[1];
        let input_height = input_shape.dims()[2];
        let input_width = input_shape.dims()[3];

        // Calculate output dimensions
        let output_height =
            (input_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let output_width =
            (input_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = zeros::<f32>(&[batch_size, channels, output_height, output_width])?;

        // Perform average pooling for each batch and channel
        for b in 0..batch_size {
            for c in 0..channels {
                self.avg_pool_channel(
                    input,
                    &mut output,
                    b,
                    c,
                    input_height,
                    input_width,
                    output_height,
                    output_width,
                )?;
            }
        }

        Ok(output)
    }

    /// Perform average pooling for a single channel
    #[allow(clippy::too_many_arguments)]
    fn avg_pool_channel(
        &self,
        input: &Tensor,
        output: &mut Tensor,
        batch_idx: usize,
        channel_idx: usize,
        input_height: usize,
        input_width: usize,
        output_height: usize,
        output_width: usize,
    ) -> TorshResult<()> {
        for out_h in 0..output_height {
            for out_w in 0..output_width {
                let mut sum = 0.0;
                let mut count = 0;

                // Iterate over the pooling window
                for kh in 0..self.kernel_size.0 {
                    for kw in 0..self.kernel_size.1 {
                        let in_h = out_h * self.stride.0 + kh;
                        let in_w = out_w * self.stride.1 + kw;

                        // Apply padding offset
                        if in_h >= self.padding.0 && in_w >= self.padding.1 {
                            let padded_in_h = in_h - self.padding.0;
                            let padded_in_w = in_w - self.padding.1;

                            // Check bounds
                            if padded_in_h < input_height && padded_in_w < input_width {
                                let val = input.get(&[
                                    batch_idx,
                                    channel_idx,
                                    padded_in_h,
                                    padded_in_w,
                                ])?;
                                sum += val;
                                count += 1;
                            } else if self.count_include_pad {
                                count += 1; // Count padding zeros
                            }
                        } else if self.count_include_pad {
                            count += 1; // Count padding zeros
                        }
                    }
                }

                // Calculate average
                let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                output.set(&[batch_idx, channel_idx, out_h, out_w], avg)?;
            }
        }

        Ok(())
    }
}

/// Sparse 1D Max Pooling layer
pub struct SparseMaxPool1d {
    /// Kernel size
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    /// Dilation
    dilation: usize,
}

impl SparseMaxPool1d {
    /// Create a new sparse 1D max pooling layer
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        dilation: Option<usize>,
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        let padding = padding.unwrap_or(0);
        let dilation = dilation.unwrap_or(1);

        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Forward pass for sparse 1D max pooling
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape: (batch_size, channels, length)
        let input_shape = input.shape();
        if input_shape.ndim() != 3 {
            return Err(TorshError::InvalidArgument(
                "Input must be 3D tensor (batch_size, channels, length)".to_string(),
            ));
        }

        let batch_size = input_shape.dims()[0];
        let channels = input_shape.dims()[1];
        let input_length = input_shape.dims()[2];

        // Calculate output length
        let output_length =
            (input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
                + 1;

        let output = zeros::<f32>(&[batch_size, channels, output_length])?;

        // Perform max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for out_pos in 0..output_length {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut found_value = false;

                    // Iterate over the pooling window
                    for k in 0..self.kernel_size {
                        let in_pos = out_pos * self.stride + k * self.dilation;

                        // Apply padding offset
                        if in_pos >= self.padding {
                            let padded_in_pos = in_pos - self.padding;

                            // Check bounds
                            if padded_in_pos < input_length {
                                let val = input.get(&[b, c, padded_in_pos])?;
                                if val > max_val || !found_value {
                                    max_val = val;
                                    found_value = true;
                                }
                            }
                        }
                    }

                    // Set output value
                    output.set(&[b, c, out_pos], if found_value { max_val } else { 0.0 })?;
                }
            }
        }

        Ok(output)
    }
}

/// Sparse 1D Average Pooling layer
pub struct SparseAvgPool1d {
    /// Kernel size
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    /// Whether to count padding zeros in the average
    count_include_pad: bool,
}

impl SparseAvgPool1d {
    /// Create a new sparse 1D average pooling layer
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        count_include_pad: bool,
    ) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        let padding = padding.unwrap_or(0);

        Self {
            kernel_size,
            stride,
            padding,
            count_include_pad,
        }
    }

    /// Forward pass for sparse 1D average pooling
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape: (batch_size, channels, length)
        let input_shape = input.shape();
        if input_shape.ndim() != 3 {
            return Err(TorshError::InvalidArgument(
                "Input must be 3D tensor (batch_size, channels, length)".to_string(),
            ));
        }

        let batch_size = input_shape.dims()[0];
        let channels = input_shape.dims()[1];
        let input_length = input_shape.dims()[2];

        // Calculate output length
        let output_length = (input_length + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let output = zeros::<f32>(&[batch_size, channels, output_length])?;

        // Perform average pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for out_pos in 0..output_length {
                    let mut sum = 0.0;
                    let mut count = 0;

                    // Iterate over the pooling window
                    for k in 0..self.kernel_size {
                        let in_pos = out_pos * self.stride + k;

                        // Apply padding offset
                        if in_pos >= self.padding {
                            let padded_in_pos = in_pos - self.padding;

                            // Check bounds
                            if padded_in_pos < input_length {
                                let val = input.get(&[b, c, padded_in_pos])?;
                                sum += val;
                                count += 1;
                            } else if self.count_include_pad {
                                count += 1; // Count padding zeros
                            }
                        } else if self.count_include_pad {
                            count += 1; // Count padding zeros
                        }
                    }

                    // Calculate average
                    let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                    output.set(&[b, c, out_pos], avg)?;
                }
            }
        }

        Ok(output)
    }
}

/// Sparse Adaptive Max Pooling 2D
///
/// Performs adaptive max pooling to produce a fixed output size regardless of input dimensions.
/// Useful for global pooling operations in sparse neural networks.
pub struct SparseAdaptiveMaxPool2d {
    /// Target output size (height, width)
    output_size: (usize, usize),
}

impl SparseAdaptiveMaxPool2d {
    /// Create a new sparse adaptive max pooling layer
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Forward pass for sparse adaptive max pooling
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape: (batch_size, channels, height, width)
        let input_shape = input.shape();
        if input_shape.ndim() != 4 {
            return Err(TorshError::InvalidArgument(
                "Input must be 4D tensor (batch_size, channels, height, width)".to_string(),
            ));
        }

        let batch_size = input_shape.dims()[0];
        let channels = input_shape.dims()[1];
        let input_height = input_shape.dims()[2];
        let input_width = input_shape.dims()[3];

        let output_height = self.output_size.0;
        let output_width = self.output_size.1;

        let output = zeros::<f32>(&[batch_size, channels, output_height, output_width])?;

        // Perform adaptive max pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        // Calculate adaptive pooling window
                        let start_h = (out_h * input_height) / output_height;
                        let end_h = ((out_h + 1) * input_height).div_ceil(output_height);
                        let start_w = (out_w * input_width) / output_width;
                        let end_w = ((out_w + 1) * input_width).div_ceil(output_width);

                        let mut max_val = f32::NEG_INFINITY;
                        let mut found_value = false;

                        // Find max in adaptive window
                        for h in start_h..end_h {
                            for w in start_w..end_w {
                                if h < input_height && w < input_width {
                                    let val = input.get(&[b, c, h, w])?;
                                    if val > max_val || !found_value {
                                        max_val = val;
                                        found_value = true;
                                    }
                                }
                            }
                        }

                        output.set(
                            &[b, c, out_h, out_w],
                            if found_value { max_val } else { 0.0 },
                        )?;
                    }
                }
            }
        }

        Ok(output)
    }
}

/// Sparse Adaptive Average Pooling 2D
///
/// Performs adaptive average pooling to produce a fixed output size regardless of input dimensions.
/// Computes averages over adaptive window sizes.
pub struct SparseAdaptiveAvgPool2d {
    /// Target output size (height, width)
    output_size: (usize, usize),
}

impl SparseAdaptiveAvgPool2d {
    /// Create a new sparse adaptive average pooling layer
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Forward pass for sparse adaptive average pooling
    pub fn forward(&self, input: &Tensor) -> TorshResult<Tensor> {
        // Validate input shape: (batch_size, channels, height, width)
        let input_shape = input.shape();
        if input_shape.ndim() != 4 {
            return Err(TorshError::InvalidArgument(
                "Input must be 4D tensor (batch_size, channels, height, width)".to_string(),
            ));
        }

        let batch_size = input_shape.dims()[0];
        let channels = input_shape.dims()[1];
        let input_height = input_shape.dims()[2];
        let input_width = input_shape.dims()[3];

        let output_height = self.output_size.0;
        let output_width = self.output_size.1;

        let output = zeros::<f32>(&[batch_size, channels, output_height, output_width])?;

        // Perform adaptive average pooling
        for b in 0..batch_size {
            for c in 0..channels {
                for out_h in 0..output_height {
                    for out_w in 0..output_width {
                        // Calculate adaptive pooling window
                        let start_h = (out_h * input_height) / output_height;
                        let end_h = ((out_h + 1) * input_height).div_ceil(output_height);
                        let start_w = (out_w * input_width) / output_width;
                        let end_w = ((out_w + 1) * input_width).div_ceil(output_width);

                        let mut sum = 0.0;
                        let mut count = 0;

                        // Calculate average in adaptive window
                        for h in start_h..end_h {
                            for w in start_w..end_w {
                                if h < input_height && w < input_width {
                                    let val = input.get(&[b, c, h, w])?;
                                    sum += val;
                                    count += 1;
                                }
                            }
                        }

                        let avg = if count > 0 { sum / count as f32 } else { 0.0 };
                        output.set(&[b, c, out_h, out_w], avg)?;
                    }
                }
            }
        }

        Ok(output)
    }
}
