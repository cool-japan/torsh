//! Pooling operations for Metal backend

use torsh_core::Shape;

use crate::metal::{
    buffer::MetalBuffer,
    error::{MetalError, Result},
    kernels::{kernel_names, KernelManager},
    ops::execute_and_wait,
};

/// Pooling configuration
pub struct Pool2dConfig {
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Default for Pool2dConfig {
    fn default() -> Self {
        Self {
            kernel_size: (2, 2),
            stride: (2, 2),
            padding: (0, 0),
            dilation: (1, 1),
        }
    }
}

/// Max pooling 2D
pub fn max_pool2d(input: &MetalBuffer, config: Pool2dConfig) -> Result<MetalBuffer> {
    pool2d_kernel(input, config, kernel_names::MAXPOOL2D_F32)
}

/// Average pooling 2D
pub fn avg_pool2d(input: &MetalBuffer, config: Pool2dConfig) -> Result<MetalBuffer> {
    pool2d_kernel(input, config, kernel_names::AVGPOOL2D_F32)
}

/// Generic pooling kernel
fn pool2d_kernel(
    input: &MetalBuffer,
    config: Pool2dConfig,
    kernel_name: &str,
) -> Result<MetalBuffer> {
    let input_shape = input.shape().dims();

    if input_shape.len() != 4 {
        return Err(MetalError::ShapeMismatch {
            expected: vec![4],
            got: vec![input_shape.len()],
        });
    }

    let (batch_size, channels, height, width) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );

    // Calculate output dimensions
    let out_height =
        (height + 2 * config.padding.0 - config.dilation.0 * (config.kernel_size.0 - 1) - 1)
            / config.stride.0
            + 1;
    let out_width =
        (width + 2 * config.padding.1 - config.dilation.1 * (config.kernel_size.1 - 1) - 1)
            / config.stride.1
            + 1;

    let output_shape = Shape::from(vec![batch_size, channels, out_height, out_width]);

    let device = input.device();
    let output = MetalBuffer::zeros(&output_shape, &input.dtype(), device)?;
    let kernel_manager = KernelManager::new(device.device_ref())?;

    // Create parameters buffer
    let pool_params = [
        batch_size as u32,
        channels as u32,
        height as u32,
        width as u32,
        config.kernel_size.0 as u32,
        config.kernel_size.1 as u32,
        config.stride.0 as u32,
        config.stride.1 as u32,
        config.padding.0 as u32,
        config.padding.1 as u32,
    ];

    let params_buffer = device.device().new_buffer_with_data(
        pool_params.as_ptr() as *const _,
        (pool_params.len() * std::mem::size_of::<u32>()) as u64,
        device.resource_options(),
    );

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);
        encoder.set_buffer(2, Some(&params_buffer), 0);

        let total_work = batch_size * channels * out_height * out_width;
        kernel_manager.dispatch_3d(encoder, kernel_name, total_work, channels, batch_size)
    })?;

    Ok(output)
}

/// Adaptive average pooling 2D
pub fn adaptive_avg_pool2d(
    input: &MetalBuffer,
    output_size: (usize, usize),
) -> Result<MetalBuffer> {
    let input_shape = input.shape().dims();

    if input_shape.len() != 4 {
        return Err(MetalError::ShapeMismatch {
            expected: vec![4],
            got: vec![input_shape.len()],
        });
    }

    let (_, _, height, width) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    );

    // Calculate kernel size and stride to achieve desired output size
    let kernel_h = height / output_size.0;
    let kernel_w = width / output_size.1;
    let stride_h = kernel_h;
    let stride_w = kernel_w;

    let config = Pool2dConfig {
        kernel_size: (kernel_h, kernel_w),
        stride: (stride_h, stride_w),
        padding: (0, 0),
        dilation: (1, 1),
    };

    avg_pool2d(input, config)
}
