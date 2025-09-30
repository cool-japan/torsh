//! Reduction operations for Metal backend

use torsh_core::Shape;

use crate::metal::{
    buffer::MetalBuffer,
    error::{MetalError, Result},
    kernels::{kernel_names, KernelManager},
    ops::execute_and_wait,
};

/// Sum reduction
pub fn sum(input: &MetalBuffer, dims: Option<&[usize]>, keepdim: bool) -> Result<MetalBuffer> {
    reduce_op(input, dims, keepdim, kernel_names::REDUCE_SUM_F32)
}

/// Mean reduction
pub fn mean(input: &MetalBuffer, dims: Option<&[usize]>, keepdim: bool) -> Result<MetalBuffer> {
    reduce_op(input, dims, keepdim, kernel_names::REDUCE_MEAN_F32)
}

/// Max reduction
pub fn max(input: &MetalBuffer, dims: Option<&[usize]>, keepdim: bool) -> Result<MetalBuffer> {
    reduce_op(input, dims, keepdim, kernel_names::REDUCE_MAX_F32)
}

/// Min reduction
pub fn min(input: &MetalBuffer, dims: Option<&[usize]>, keepdim: bool) -> Result<MetalBuffer> {
    reduce_op(input, dims, keepdim, kernel_names::REDUCE_MIN_F32)
}

/// Generic reduction operation
fn reduce_op(
    input: &MetalBuffer,
    dims: Option<&[usize]>,
    keepdim: bool,
    kernel_name: &str,
) -> Result<MetalBuffer> {
    let input_shape = input.shape().dims();
    let ndim = input_shape.len();

    // Determine which dimensions to reduce
    let reduce_dims: Vec<usize> = if let Some(dims) = dims {
        dims.iter()
            .map(|&d| if d < ndim { d } else { d - ndim })
            .collect()
    } else {
        // Reduce all dimensions
        (0..ndim).collect()
    };

    // Calculate output shape
    let mut output_shape = Vec::new();
    for (i, &size) in input_shape.iter().enumerate() {
        if reduce_dims.contains(&i) {
            if keepdim {
                output_shape.push(1);
            }
        } else {
            output_shape.push(size);
        }
    }

    if output_shape.is_empty() {
        output_shape.push(1); // Scalar output
    }

    let device = input.device();
    let output = MetalBuffer::zeros(&Shape::from(output_shape), &input.dtype(), device)?;

    // For now, use a simple full reduction kernel
    // Real implementation would handle partial reductions
    if reduce_dims.len() == ndim && !keepdim {
        // Full reduction to scalar
        full_reduce(input, output.clone(), kernel_name)?;
    } else {
        // Partial reduction - would need more sophisticated kernel
        return Err(crate::metal::error::metal_errors::unsupported_operation_error(
            "Partial reductions not yet implemented",
            None,
        ));
    }

    Ok(output)
}

/// Full reduction to scalar
fn full_reduce(input: &MetalBuffer, output: MetalBuffer, kernel_name: &str) -> Result<()> {
    let device = input.device();
    let kernel_manager = KernelManager::new(device.device_ref())?;

    let shape = [input.shape().numel() as u32];
    let shape_buffer = device.device().new_buffer_with_data(
        shape.as_ptr() as *const _,
        (shape.len() * std::mem::size_of::<u32>()) as u64,
        device.resource_options(),
    );

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(input.buffer()), 0);
        encoder.set_buffer(1, Some(output.buffer()), 0);
        encoder.set_buffer(2, Some(&shape_buffer), 0);

        // Simple single-threaded reduction for now
        kernel_manager.dispatch_1d(encoder, kernel_name, 1)
    })
}

/// Softmax operation
pub fn softmax(input: &MetalBuffer, dim: i32) -> Result<MetalBuffer> {
    let ndim = input.shape().dims().len() as i32;
    let dim = if dim < 0 { ndim + dim } else { dim };

    if dim < 0 || dim >= ndim {
        return Err(MetalError::BackendError(format!(
            "Invalid softmax dimension {} for tensor with {} dimensions",
            dim, ndim
        )));
    }

    // Softmax = exp(x - max) / sum(exp(x - max))
    // TODO: This is a simplified implementation - replace with proper Metal compute shader
    // For now, return an error indicating this operation is not yet implemented
    Err(crate::metal::error::metal_errors::unsupported_operation_error(
        "Softmax not yet implemented for Metal backend",
        None,
    ))
}
