#include "cuda_kernels.h"
#include <cmath>

// Elementwise operations
__global__ void elementwise_add_f32_kernel(float* a, float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_mul_f32_kernel(float* a, float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void elementwise_sub_f32_kernel(float* a, float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void elementwise_div_f32_kernel(float* a, float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}

// Activation functions
__global__ void elementwise_relu_f32_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void elementwise_sigmoid_f32_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void elementwise_tanh_f32_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void elementwise_gelu_f32_kernel(float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float cube = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * cube);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// Matrix transpose
__global__ void transpose_f32_kernel(float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        int input_idx = idy * cols + idx;
        int output_idx = idx * rows + idy;
        output[output_idx] = input[input_idx];
    }
}

// Optimized transpose using shared memory
__global__ void transpose_f32_shared_kernel(float* input, float* output, int rows, int cols) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Load into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Write transposed output
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Launch functions
extern "C" {
    void launch_elementwise_add_f32(float* a, float* b, float* out, size_t size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elementwise_add_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, out, size);
    }
    
    void launch_elementwise_mul_f32(float* a, float* b, float* out, size_t size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elementwise_mul_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, out, size);
    }
    
    void launch_elementwise_sub_f32(float* a, float* b, float* out, size_t size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elementwise_sub_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, out, size);
    }
    
    void launch_elementwise_div_f32(float* a, float* b, float* out, size_t size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elementwise_div_f32_kernel<<<grid_size, block_size, 0, stream>>>(a, b, out, size);
    }
    
    void launch_elementwise_relu_f32(float* input, float* output, size_t size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elementwise_relu_f32_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
    }
    
    void launch_elementwise_sigmoid_f32(float* input, float* output, size_t size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elementwise_sigmoid_f32_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
    }
    
    void launch_elementwise_tanh_f32(float* input, float* output, size_t size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elementwise_tanh_f32_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
    }
    
    void launch_elementwise_gelu_f32(float* input, float* output, size_t size, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        elementwise_gelu_f32_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
    }
    
    void launch_transpose_f32(float* input, float* output, int rows, int cols, cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((cols + block_size.x - 1) / block_size.x, 
                       (rows + block_size.y - 1) / block_size.y);
        
        // Use shared memory optimization for larger matrices
        if (rows > 512 && cols > 512) {
            dim3 shared_block(32, 32);
            dim3 shared_grid((cols + shared_block.x - 1) / shared_block.x,
                            (rows + shared_block.y - 1) / shared_block.y);
            transpose_f32_shared_kernel<<<shared_grid, shared_block, 0, stream>>>(input, output, rows, cols);
        } else {
            transpose_f32_kernel<<<grid_size, block_size, 0, stream>>>(input, output, rows, cols);
        }
    }
}