//! Rust bindings for tensor operation kernels

use crate::error::{CudaError, CudaResult};

// Import generated bindings
include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));

/// Launch elementwise addition kernel
pub fn launch_elementwise_add_f32(
    a: *mut f32,
    b: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_elementwise_add_f32(a, b, output, size, stream);
    }
}

/// Launch elementwise multiplication kernel
pub fn launch_elementwise_mul_f32(
    a: *mut f32,
    b: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_elementwise_mul_f32(a, b, output, size, stream);
    }
}

/// Launch elementwise subtraction kernel
pub fn launch_elementwise_sub_f32(
    a: *mut f32,
    b: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_elementwise_sub_f32(a, b, output, size, stream);
    }
}

/// Launch elementwise division kernel
pub fn launch_elementwise_div_f32(
    a: *mut f32,
    b: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_elementwise_div_f32(a, b, output, size, stream);
    }
}

/// Launch ReLU activation kernel
pub fn launch_elementwise_relu_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_elementwise_relu_f32(input, output, size, stream);
    }
}

/// Launch sigmoid activation kernel
pub fn launch_elementwise_sigmoid_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_elementwise_sigmoid_f32(input, output, size, stream);
    }
}

/// Launch tanh activation kernel
pub fn launch_elementwise_tanh_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_elementwise_tanh_f32(input, output, size, stream);
    }
}

/// Launch GELU activation kernel
pub fn launch_elementwise_gelu_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_elementwise_gelu_f32(input, output, size, stream);
    }
}

/// Launch matrix transpose kernel
pub fn launch_transpose_f32(
    input: *mut f32,
    output: *mut f32,
    rows: i32,
    cols: i32,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_transpose_f32(input, output, rows, cols, stream);
    }
}

/// Launch scalar multiplication kernel
pub fn launch_scalar_mul_f32(
    input: *mut f32,
    output: *mut f32,
    scalar: f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_scalar_mul_f32(input, output, scalar, size, stream);
    }
}

/// Launch ReLU activation kernel (alias for compatibility)
pub fn launch_relu_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    launch_elementwise_relu_f32(input, output, size, stream);
}

/// Launch sigmoid activation kernel (alias for compatibility)
pub fn launch_sigmoid_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    launch_elementwise_sigmoid_f32(input, output, size, stream);
}

/// Launch tanh activation kernel (alias for compatibility)
pub fn launch_tanh_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    launch_elementwise_tanh_f32(input, output, size, stream);
}

/// Launch F32 to F16 conversion kernel
pub fn launch_f32_to_f16(
    input: *mut f32,
    output: *mut f16,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_f32_to_f16(input, output, size, stream);
    }
}

/// Launch F16 to F32 conversion kernel
pub fn launch_f16_to_f32(
    input: *mut f16,
    output: *mut f32,
    size: usize,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_f16_to_f32(input, output, size, stream);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kernel_bindings_exist() {
        // Just verify the functions are properly linked
        // Actual functionality testing requires CUDA device
    }
}