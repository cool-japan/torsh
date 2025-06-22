//! Rust bindings for neural network operation kernels

use crate::error::{CudaError, CudaResult};

/// Launch 2D convolution kernel
pub fn launch_conv2d_f32(
    input: *mut f32,
    weight: *mut f32,
    bias: *mut f32,
    output: *mut f32,
    batch_size: i32,
    in_channels: i32,
    out_channels: i32,
    input_height: i32,
    input_width: i32,
    kernel_height: i32,
    kernel_width: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_conv2d_f32(
            input, weight, bias, output,
            batch_size, in_channels, out_channels,
            input_height, input_width,
            kernel_height, kernel_width,
            pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, stream,
        );
    }
}

/// Launch 2D max pooling kernel
pub fn launch_maxpool2d_f32(
    input: *mut f32,
    output: *mut f32,
    batch_size: i32,
    channels: i32,
    input_height: i32,
    input_width: i32,
    output_height: i32,
    output_width: i32,
    kernel_height: i32,
    kernel_width: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_maxpool2d_f32(
            input, output,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_height, kernel_width,
            pad_h, pad_w, stride_h, stride_w,
            stream,
        );
    }
}

/// Launch 2D batch normalization kernel
pub fn launch_batchnorm2d_f32(
    input: *mut f32,
    output: *mut f32,
    weight: *mut f32,
    bias: *mut f32,
    running_mean: *mut f32,
    running_var: *mut f32,
    batch_size: i32,
    channels: i32,
    height: i32,
    width: i32,
    eps: f32,
    momentum: f32,
    training: bool,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_batchnorm2d_f32(
            input, output, weight, bias,
            running_mean, running_var,
            batch_size, channels, height, width,
            eps, momentum, training, stream,
        );
    }
}

/// Launch softmax kernel
pub fn launch_softmax_f32(
    input: *mut f32,
    output: *mut f32,
    batch_size: i32,
    classes: i32,
    stream: cuda_sys::CUstream,
) {
    unsafe {
        super::cuda_kernels::launch_softmax_f32(input, output, batch_size, classes, stream);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neural_kernel_bindings_exist() {
        // Verify the functions are properly linked
    }
}