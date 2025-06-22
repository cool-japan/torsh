#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __cplusplus
extern "C" {
#endif

// Tensor operations
void launch_elementwise_add_f32(float* a, float* b, float* out, size_t size, cudaStream_t stream);
void launch_elementwise_mul_f32(float* a, float* b, float* out, size_t size, cudaStream_t stream);
void launch_elementwise_relu_f32(float* input, float* output, size_t size, cudaStream_t stream);
void launch_elementwise_sigmoid_f32(float* input, float* output, size_t size, cudaStream_t stream);
void launch_elementwise_tanh_f32(float* input, float* output, size_t size, cudaStream_t stream);

// Matrix operations
void launch_matmul_f32(
    float* a, float* b, float* c,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    cudaStream_t stream
);

void launch_transpose_f32(
    float* input, float* output,
    int rows, int cols,
    cudaStream_t stream
);

// Reduction operations
void launch_sum_f32(float* input, float* output, size_t size, int axis, cudaStream_t stream);
void launch_mean_f32(float* input, float* output, size_t size, int axis, cudaStream_t stream);
void launch_max_f32(float* input, float* output, size_t size, int axis, cudaStream_t stream);
void launch_min_f32(float* input, float* output, size_t size, int axis, cudaStream_t stream);

// Neural network operations
void launch_conv2d_f32(
    float* input, float* weight, float* bias, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int kernel_height, int kernel_width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
);

void launch_maxpool2d_f32(
    float* input, float* output,
    int batch_size, int channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_height, int kernel_width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    cudaStream_t stream
);

void launch_batchnorm2d_f32(
    float* input, float* output,
    float* weight, float* bias,
    float* running_mean, float* running_var,
    int batch_size, int channels,
    int height, int width,
    float eps, float momentum,
    bool training,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H