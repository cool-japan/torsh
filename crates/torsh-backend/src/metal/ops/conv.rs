//! Convolution operations for Metal backend

use torsh_core::Shape;

use crate::metal::{
    buffer::MetalBuffer,
    error::{MetalError, Result},
    kernels::{kernel_names, KernelManager},
    mps::Conv2dParams,
    ops::execute_and_wait,
};

/// 2D Convolution parameters
pub struct Conv2dConfig {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }
}

/// 2D Convolution forward pass
pub fn conv2d(
    input: &MetalBuffer,
    weight: &MetalBuffer,
    bias: Option<&MetalBuffer>,
    config: Conv2dConfig,
) -> Result<MetalBuffer> {
    let input_shape = input.shape().dims();
    let weight_shape = weight.shape().dims();

    // Validate shapes
    if input_shape.len() != 4 || weight_shape.len() != 4 {
        return Err(MetalError::ShapeMismatch {
            expected: vec![4, 4],
            got: vec![input_shape.len(), weight_shape.len()],
        });
    }

    let (batch_size, in_channels, height, width) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );

    let (out_channels, weight_in_channels, kernel_height, kernel_width) = (
        weight_shape[0],
        weight_shape[1],
        weight_shape[2],
        weight_shape[3],
    );

    if in_channels != weight_in_channels * config.groups {
        return Err(MetalError::ShapeMismatch {
            expected: vec![in_channels],
            got: vec![weight_in_channels * config.groups],
        });
    }

    // Calculate output dimensions
    let _out_height = (height + 2 * config.padding.0 - config.dilation.0 * (kernel_height - 1) - 1)
        / config.stride.0
        + 1;
    let _out_width = (width + 2 * config.padding.1 - config.dilation.1 * (kernel_width - 1) - 1)
        / config.stride.1
        + 1;

    let _device = input.device();

    // Use MPS for better performance
    let params = Conv2dParams {
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_height,
        kernel_width,
        stride_height: config.stride.0,
        stride_width: config.stride.1,
        padding_height: config.padding.0,
        padding_width: config.padding.1,
        dilation_height: config.dilation.0,
        dilation_width: config.dilation.1,
        groups: config.groups,
    };

    // For now, use custom kernel
    conv2d_kernel(input, weight, bias, params)
}

/// Convolution using custom kernel
fn conv2d_kernel(
    input: &MetalBuffer,
    weight: &MetalBuffer,
    bias: Option<&MetalBuffer>,
    params: Conv2dParams,
) -> Result<MetalBuffer> {
    let device = input.device();

    // Calculate output shape
    let out_h = (params.height + 2 * params.padding_height - params.kernel_height)
        / params.stride_height
        + 1;
    let out_w =
        (params.width + 2 * params.padding_width - params.kernel_width) / params.stride_width + 1;

    let output_shape = Shape::from(vec![params.batch_size, params.out_channels, out_h, out_w]);

    let output = MetalBuffer::zeros(&output_shape, &input.dtype(), device)?;
    let kernel_manager = KernelManager::new(device.device_ref())?;

    // Create parameters buffer
    let conv_params = [
        params.batch_size as u32,
        params.in_channels as u32,
        params.height as u32,
        params.width as u32,
        params.out_channels as u32,
        params.kernel_height as u32,
        params.kernel_width as u32,
        params.stride_height as u32,
        params.stride_width as u32,
        params.padding_height as u32,
        params.padding_width as u32,
    ];

    let params_buffer = device.device().new_buffer_with_data(
        conv_params.as_ptr() as *const _,
        (conv_params.len() * std::mem::size_of::<u32>()) as u64,
        device.resource_options(),
    );

    // Create null buffer for bias if not provided
    let null_buffer = device.device().new_buffer(4, device.resource_options());

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(weight.buffer()), 0);
        encoder.set_buffer(
            2,
            bias.map(|b| b.buffer().as_ref()).or(Some(&null_buffer)),
            0,
        );
        encoder.set_buffer(3, Some(output.buffer()), 0);
        encoder.set_buffer(4, Some(&params_buffer), 0);

        let total_work = params.batch_size * params.out_channels * out_h * out_w;
        kernel_manager.dispatch_3d(
            encoder,
            kernel_names::CONV2D_F32,
            total_work,
            params.out_channels,
            params.batch_size,
        )
    })?;

    Ok(output)
}
