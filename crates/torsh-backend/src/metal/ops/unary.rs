//! Unary operations for Metal backend

use crate::metal::{
    buffer::MetalBuffer,
    error::Result,
    kernels::{kernel_names, KernelManager},
    ops::execute_and_wait,
};

/// Negate operation
pub fn neg(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_NEG_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// Exponential operation
pub fn exp(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_EXP_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// Natural logarithm operation
pub fn log(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_LOG_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// Square root operation
pub fn sqrt(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_SQRT_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// Hyperbolic tangent operation
pub fn tanh(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_TANH_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// ReLU activation operation
pub fn relu(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_RELU_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// Absolute value operation
pub fn abs(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_ABS_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// Sine operation
pub fn sin(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_SIN_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// Cosine operation
pub fn cos(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_COS_F32, input.shape().numel())
    })?;

    Ok(output)
}

/// Sigmoid activation operation
pub fn sigmoid(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(
            encoder,
            kernel_names::UNARY_SIGMOID_F32,
            input.shape().numel(),
        )
    })?;

    Ok(output)
}

/// GELU activation operation
pub fn gelu(input: &MetalBuffer) -> Result<MetalBuffer> {
    let device = input.device();
    let output = MetalBuffer::zeros(input.shape(), &input.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::UNARY_GELU_F32, input.shape().numel())
    })?;

    Ok(output)
}
