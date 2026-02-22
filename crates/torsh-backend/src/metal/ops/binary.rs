//! Binary operations for Metal backend

use crate::metal::{
    buffer::MetalBuffer,
    error::{MetalError, Result},
    kernels::{kernel_names, KernelManager},
    ops::execute_and_wait,
};

/// Addition operation
pub fn add(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    validate_shapes(a, b)?;

    let device = a.device();
    let output = MetalBuffer::zeros(a.shape(), &a.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(a.buffer()), 0);
        encoder.set_buffer(1, Some(b.buffer()), 0);
        encoder.set_buffer(2, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::BINARY_ADD_F32, a.shape().numel())
    })?;

    Ok(output)
}

/// Subtraction operation
pub fn sub(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    validate_shapes(a, b)?;

    let device = a.device();
    let output = MetalBuffer::zeros(a.shape(), &a.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(a.buffer()), 0);
        encoder.set_buffer(1, Some(b.buffer()), 0);
        encoder.set_buffer(2, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::BINARY_SUB_F32, a.shape().numel())
    })?;

    Ok(output)
}

/// Multiplication operation
pub fn mul(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    validate_shapes(a, b)?;

    let device = a.device();
    let output = MetalBuffer::zeros(a.shape(), &a.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(a.buffer()), 0);
        encoder.set_buffer(1, Some(b.buffer()), 0);
        encoder.set_buffer(2, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::BINARY_MUL_F32, a.shape().numel())
    })?;

    Ok(output)
}

/// Division operation
pub fn div(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    validate_shapes(a, b)?;

    let device = a.device();
    let output = MetalBuffer::zeros(a.shape(), &a.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(a.buffer()), 0);
        encoder.set_buffer(1, Some(b.buffer()), 0);
        encoder.set_buffer(2, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::BINARY_DIV_F32, a.shape().numel())
    })?;

    Ok(output)
}

/// Power operation
pub fn pow(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    validate_shapes(a, b)?;

    let device = a.device();
    let output = MetalBuffer::zeros(a.shape(), &a.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(a.buffer()), 0);
        encoder.set_buffer(1, Some(b.buffer()), 0);
        encoder.set_buffer(2, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::BINARY_POW_F32, a.shape().numel())
    })?;

    Ok(output)
}

/// Element-wise maximum operation
pub fn maximum(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    validate_shapes(a, b)?;

    let device = a.device();
    let output = MetalBuffer::zeros(a.shape(), &a.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(a.buffer()), 0);
        encoder.set_buffer(1, Some(b.buffer()), 0);
        encoder.set_buffer(2, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::BINARY_MAX_F32, a.shape().numel())
    })?;

    Ok(output)
}

/// Element-wise minimum operation
pub fn minimum(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    validate_shapes(a, b)?;

    let device = a.device();
    let output = MetalBuffer::zeros(a.shape(), &a.dtype(), device)?;

    let kernel_manager = KernelManager::new(device.device_ref())?;

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(a.buffer()), 0);
        encoder.set_buffer(1, Some(b.buffer()), 0);
        encoder.set_buffer(2, Some(output.buffer()), 0);

        kernel_manager.dispatch_1d(encoder, kernel_names::BINARY_MIN_F32, a.shape().numel())
    })?;

    Ok(output)
}

/// Validate that two buffers have compatible shapes
fn validate_shapes(a: &MetalBuffer, b: &MetalBuffer) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(MetalError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    }

    if a.dtype() != b.dtype() {
        return Err(MetalError::ConversionError(format!(
            "Data type mismatch: {:?} vs {:?}",
            a.dtype(),
            b.dtype()
        )));
    }

    Ok(())
}
