#include "cuda_kernels.h"
#include <cmath>

// Convolution 2D kernel (simple implementation)
__global__ void conv2d_f32_kernel(
    float* input, float* weight, float* bias, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_height, int kernel_width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * output_height * output_width;
    
    if (idx < total_outputs) {
        // Compute output position
        int w_out = idx % output_width;
        int h_out = (idx / output_width) % output_height;
        int c_out = (idx / (output_width * output_height)) % out_channels;
        int n = idx / (out_channels * output_height * output_width);
        
        float sum = 0.0f;
        
        // Convolution computation
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int kh = 0; kh < kernel_height; kh++) {
                for (int kw = 0; kw < kernel_width; kw++) {
                    int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                    int w_in = w_out * stride_w - pad_w + kw * dilation_w;
                    
                    if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                        int input_idx = n * in_channels * input_height * input_width +
                                       c_in * input_height * input_width +
                                       h_in * input_width + w_in;
                        
                        int weight_idx = c_out * in_channels * kernel_height * kernel_width +
                                        c_in * kernel_height * kernel_width +
                                        kh * kernel_width + kw;
                        
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        // Add bias if provided
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        
        output[idx] = sum;
    }
}

// Max pooling 2D kernel
__global__ void maxpool2d_f32_kernel(
    float* input, float* output,
    int batch_size, int channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_height, int kernel_width,
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_height * output_width;
    
    if (idx < total_outputs) {
        // Compute output position
        int w_out = idx % output_width;
        int h_out = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int n = idx / (channels * output_height * output_width);
        
        float max_val = -INFINITY;
        
        // Max pooling computation
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = n * channels * input_height * input_width +
                                   c * input_height * input_width +
                                   h_in * input_width + w_in;
                    
                    max_val = fmaxf(max_val, input[input_idx]);
                }
            }
        }
        
        output[idx] = max_val;
    }
}

// Average pooling 2D kernel
__global__ void avgpool2d_f32_kernel(
    float* input, float* output,
    int batch_size, int channels,
    int input_height, int input_width,
    int output_height, int output_width,
    int kernel_height, int kernel_width,  
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_height * output_width;
    
    if (idx < total_outputs) {
        int w_out = idx % output_width;
        int h_out = (idx / output_width) % output_height;
        int c = (idx / (output_width * output_height)) % channels;
        int n = idx / (channels * output_height * output_width);
        
        float sum = 0.0f;
        int count = 0;
        
        for (int kh = 0; kh < kernel_height; kh++) {
            for (int kw = 0; kw < kernel_width; kw++) {
                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;
                
                if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                    int input_idx = n * channels * input_height * input_width +
                                   c * input_height * input_width +
                                   h_in * input_width + w_in;
                    
                    sum += input[input_idx];
                    count++;
                }
            }
        }
        
        output[idx] = count > 0 ? sum / count : 0.0f;
    }
}

// Batch normalization 2D kernel
__global__ void batchnorm2d_f32_kernel(
    float* input, float* output,
    float* weight, float* bias,
    float* running_mean, float* running_var,
    float* batch_mean, float* batch_var,
    int batch_size, int channels,
    int height, int width,
    float eps, float momentum,
    bool training
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx < total_elements) {
        int c = (idx / (height * width)) % channels;
        
        float mean, var;
        if (training) {
            mean = batch_mean[c];
            var = batch_var[c];
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        
        // Normalize
        float normalized = (input[idx] - mean) / sqrtf(var + eps);
        
        // Scale and shift
        if (weight != nullptr && bias != nullptr) {
            output[idx] = weight[c] * normalized + bias[c];
        } else if (weight != nullptr) {
            output[idx] = weight[c] * normalized;
        } else if (bias != nullptr) {
            output[idx] = normalized + bias[c];
        } else {
            output[idx] = normalized;
        }
    }
}

// Softmax kernel (numerically stable)
__global__ void softmax_f32_kernel(float* input, float* output, int batch_size, int classes) {
    int batch_idx = blockIdx.x;
    
    if (batch_idx < batch_size) {
        float* input_row = input + batch_idx * classes;
        float* output_row = output + batch_idx * classes;
        
        // Find maximum for numerical stability
        float max_val = input_row[0];
        for (int i = 1; i < classes; i++) {
            max_val = fmaxf(max_val, input_row[i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < classes; i++) {
            output_row[i] = expf(input_row[i] - max_val);
            sum += output_row[i];
        }
        
        // Normalize
        for (int i = 0; i < classes; i++) {
            output_row[i] /= sum;
        }
    }
}

// Launch functions
extern "C" {
    void launch_conv2d_f32(
        float* input, float* weight, float* bias, float* output,
        int batch_size, int in_channels, int out_channels,
        int input_height, int input_width,
        int kernel_height, int kernel_width,
        int pad_h, int pad_w,
        int stride_h, int stride_w,
        int dilation_h, int dilation_w,
        cudaStream_t stream
    ) {
        int output_height = (input_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
        int output_width = (input_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
        
        int total_outputs = batch_size * out_channels * output_height * output_width;
        int block_size = 256;
        int grid_size = (total_outputs + block_size - 1) / block_size;
        
        conv2d_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            input, weight, bias, output,
            batch_size, in_channels, out_channels,
            input_height, input_width,
            output_height, output_width,
            kernel_height, kernel_width,
            pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w
        );
    }
    
    void launch_maxpool2d_f32(
        float* input, float* output,
        int batch_size, int channels,
        int input_height, int input_width,
        int output_height, int output_width,
        int kernel_height, int kernel_width,
        int pad_h, int pad_w,
        int stride_h, int stride_w,
        cudaStream_t stream
    ) {
        int total_outputs = batch_size * channels * output_height * output_width;
        int block_size = 256;
        int grid_size = (total_outputs + block_size - 1) / block_size;
        
        maxpool2d_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output,
            batch_size, channels,
            input_height, input_width,
            output_height, output_width,
            kernel_height, kernel_width,
            pad_h, pad_w, stride_h, stride_w
        );
    }
    
    void launch_batchnorm2d_f32(
        float* input, float* output,
        float* weight, float* bias,
        float* running_mean, float* running_var,
        int batch_size, int channels,
        int height, int width,
        float eps, float momentum,
        bool training,
        cudaStream_t stream
    ) {
        // For training mode, we need to compute batch statistics first
        // This is a simplified version - a full implementation would compute
        // batch mean and variance in separate kernels
        
        int total_elements = batch_size * channels * height * width;
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        // Placeholder batch statistics (should be computed properly)
        float* batch_mean = running_mean;  // Simplified
        float* batch_var = running_var;    // Simplified
        
        batchnorm2d_f32_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, weight, bias,
            running_mean, running_var,
            batch_mean, batch_var,
            batch_size, channels, height, width,
            eps, momentum, training
        );
    }
    
    void launch_softmax_f32(float* input, float* output, int batch_size, int classes, cudaStream_t stream) {
        softmax_f32_kernel<<<batch_size, 1, 0, stream>>>(input, output, batch_size, classes);
    }
}