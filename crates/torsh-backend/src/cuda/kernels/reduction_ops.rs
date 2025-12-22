//! Rust bindings for reduction operation kernels

#[allow(unused_imports)]
use crate::cuda::error::{CudaError, CudaResult};

/// Launch sum reduction kernel
pub fn launch_sum_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    axis: i32,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_sum_f32(input, output, size, axis, stream);
    }
}

/// Launch mean reduction kernel
pub fn launch_mean_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    axis: i32,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_mean_f32(input, output, size, axis, stream);
    }
}

/// Launch max reduction kernel
pub fn launch_max_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    axis: i32,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_max_f32(input, output, size, axis, stream);
    }
}

/// Launch min reduction kernel
pub fn launch_min_f32(
    input: *mut f32,
    output: *mut f32,
    size: usize,
    axis: i32,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_min_f32(input, output, size, axis, stream);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduction_kernel_bindings_exist() {
        // Verify the functions are properly linked
    }
}
