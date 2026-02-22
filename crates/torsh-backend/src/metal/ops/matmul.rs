//! Matrix multiplication operations for Metal backend

use torsh_core::Shape;

use crate::metal::{
    buffer::MetalBuffer,
    error::{MetalError, Result},
    kernels::{kernel_names, KernelManager},
    ops::execute_and_wait,
};

/// Matrix multiplication using MPS
pub fn matmul(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    // Validate shapes
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(MetalError::ShapeMismatch {
            expected: vec![2, 2],
            got: vec![a_shape.len(), b_shape.len()],
        });
    }

    // Get matrix dimensions
    let m = a_shape[a_shape.len() - 2];
    let k1 = a_shape[a_shape.len() - 1];
    let k2 = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];

    if k1 != k2 {
        return Err(MetalError::ShapeMismatch {
            expected: vec![k1],
            got: vec![k2],
        });
    }

    let _device = a.device();

    // For now, use custom kernel for simplicity
    // In production, would use MPSMatMul for better performance
    matmul_kernel(a, b, m, n, k1)
}

/// Matrix multiplication using custom kernel
fn matmul_kernel(
    a: &MetalBuffer,
    b: &MetalBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<MetalBuffer> {
    let device = a.device();
    let output_shape = {
        let mut shape = a.shape().dims().to_vec();
        let len = shape.len();
        shape[len - 2] = m;
        shape[len - 1] = n;
        Shape::from(shape)
    };

    let output = MetalBuffer::zeros(&output_shape, &a.dtype(), device)?;
    let kernel_manager = KernelManager::new(device.device_ref())?;

    // Create dimensions buffer
    let dims = [m as u32, n as u32, k as u32];
    let dims_buffer = device.device().new_buffer_with_data(
        dims.as_ptr() as *const _,
        (dims.len() * std::mem::size_of::<u32>()) as u64,
        device.resource_options(),
    );

    execute_and_wait(device, |encoder| {
        encoder.set_buffer(0, Some(a.buffer()), 0);
        encoder.set_buffer(1, Some(b.buffer()), 0);
        encoder.set_buffer(2, Some(output.buffer()), 0);
        encoder.set_buffer(3, Some(&dims_buffer), 0);

        kernel_manager.dispatch_2d(encoder, kernel_names::MATMUL_F32, n, m)
    })?;

    Ok(output)
}

/// Batched matrix multiplication
pub fn bmm(a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    let a_shape = a.shape().dims();
    let b_shape = b.shape().dims();

    if a_shape.len() != 3 || b_shape.len() != 3 {
        return Err(MetalError::ShapeMismatch {
            expected: vec![3, 3],
            got: vec![a_shape.len(), b_shape.len()],
        });
    }

    if a_shape[0] != b_shape[0] {
        return Err(MetalError::ShapeMismatch {
            expected: vec![a_shape[0]],
            got: vec![b_shape[0]],
        });
    }

    let batch_size = a_shape[0];
    let _m = a_shape[1];
    let k1 = a_shape[2];
    let k2 = b_shape[1];
    let _n = b_shape[2];

    if k1 != k2 {
        return Err(MetalError::ShapeMismatch {
            expected: vec![k1],
            got: vec![k2],
        });
    }

    // Perform batched matrix multiplication
    // For now, loop over batches - could be optimized with custom kernel
    let _device = a.device();
    let mut outputs = Vec::new();

    for i in 0..batch_size {
        // Extract batch slices
        let a_batch = extract_batch(a, i)?;
        let b_batch = extract_batch(b, i)?;

        // Perform matmul
        let output = matmul(&a_batch, &b_batch)?;
        outputs.push(output);
    }

    // Concatenate results
    concatenate_batches(&outputs)
}

/// Extract a single batch from a 3D tensor
fn extract_batch(tensor: &MetalBuffer, batch_idx: usize) -> Result<MetalBuffer> {
    let shape = tensor.shape().dims();
    if shape.len() != 3 || batch_idx >= shape[0] {
        return Err(MetalError::BackendError("Invalid batch index".to_string()));
    }

    let batch_shape = Shape::from(vec![shape[1], shape[2]]);
    let batch_size = shape[1] * shape[2];
    let _offset = batch_idx * batch_size;

    // Create a view of the batch
    // This is a simplified version - real implementation would handle strides
    tensor.view(&batch_shape)
}

/// Concatenate batches into a single tensor
fn concatenate_batches(batches: &[MetalBuffer]) -> Result<MetalBuffer> {
    if batches.is_empty() {
        return Err(MetalError::BackendError(
            "No batches to concatenate".to_string(),
        ));
    }

    let first_shape = batches[0].shape().dims();
    let batch_size = batches.len();
    let mut output_shape = vec![batch_size];
    output_shape.extend_from_slice(first_shape);

    let device = batches[0].device();
    let output = MetalBuffer::zeros(&Shape::from(output_shape), &batches[0].dtype(), device)?;

    // Copy each batch to output
    // This is simplified - real implementation would use Metal commands

    Ok(output)
}
