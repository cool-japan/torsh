#include "cuda_kernels.h"
#include <cmath>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Warp-level reduction primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction using shared memory
template<typename Op>
__device__ float block_reduce(float val, Op op) {
    static __shared__ float shared[32]; // One value per warp
    
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    // Warp-level reduction
    if constexpr (std::is_same_v<Op, std::plus<float>>) {
        val = warp_reduce_sum(val);
    } else if constexpr (std::is_same_v<Op, MaxOp>) {
        val = warp_reduce_max(val);
    } else if constexpr (std::is_same_v<Op, MinOp>) {
        val = warp_reduce_min(val);
    }
    
    // Write to shared memory
    if (lane == 0) shared[wid] = val;
    
    __syncthreads();
    
    // Read from shared memory only for first warp
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    
    // Final warp-level reduction
    if (wid == 0) {
        if constexpr (std::is_same_v<Op, std::plus<float>>) {
            val = warp_reduce_sum(val);
        } else if constexpr (std::is_same_v<Op, MaxOp>) {
            val = warp_reduce_max(val);
        } else if constexpr (std::is_same_v<Op, MinOp>) {
            val = warp_reduce_min(val);
        }
    }
    
    return val;
}

// Helper operation types
struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct MinOp {
    __device__ float operator()(float a, float b) const { return fminf(a, b); }
};

// Sum reduction kernel
__global__ void sum_f32_kernel(float* input, float* output, size_t size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Optimized sum reduction using warp shuffles
__global__ void sum_f32_optimized_kernel(float* input, float* output, size_t size) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    while (idx < size) {
        sum += input[idx];
        idx += stride;
    }
    
    // Block-level reduction
    sum = block_reduce(sum, std::plus<float>{});
    
    // Write result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

// Max reduction kernel
__global__ void max_f32_kernel(float* input, float* output, size_t size) {
    float max_val = -INFINITY;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    while (idx < size) {
        max_val = fmaxf(max_val, input[idx]);
        idx += stride;
    }
    
    // Block-level reduction
    max_val = block_reduce(max_val, MaxOp{});
    
    // Write result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = max_val;
    }
}

// Min reduction kernel
__global__ void min_f32_kernel(float* input, float* output, size_t size) {
    float min_val = INFINITY;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    while (idx < size) {
        min_val = fminf(min_val, input[idx]);
        idx += stride;
    }
    
    // Block-level reduction
    min_val = block_reduce(min_val, MinOp{});
    
    // Write result
    if (threadIdx.x == 0) {
        output[blockIdx.x] = min_val;
    }
}

// Multi-dimensional reduction along specific axis
__global__ void reduce_axis_f32_kernel(
    float* input, float* output,
    int* input_shape, int* output_shape,
    int input_ndim, int output_ndim,
    int axis, int op_type
) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total output size
    int output_size = 1;
    for (int i = 0; i < output_ndim; i++) {
        output_size *= output_shape[i];
    }
    
    if (output_idx >= output_size) return;
    
    // Convert flat output index to multi-dimensional coordinates
    int output_coords[8]; // Support up to 8D tensors
    int temp = output_idx;
    for (int i = output_ndim - 1; i >= 0; i--) {
        output_coords[i] = temp % output_shape[i];
        temp /= output_shape[i];
    }
    
    // Iterate over the reduction axis
    float result;
    bool first = true;
    
    for (int axis_idx = 0; axis_idx < input_shape[axis]; axis_idx++) {
        // Build input coordinates
        int input_coords[8];
        int coord_idx = 0;
        
        for (int dim = 0; dim < input_ndim; dim++) {
            if (dim == axis) {
                input_coords[dim] = axis_idx;
            } else {
                input_coords[dim] = output_coords[coord_idx++];
            }
        }
        
        // Convert to flat input index
        int input_idx = 0;
        int stride = 1;
        for (int i = input_ndim - 1; i >= 0; i--) {
            input_idx += input_coords[i] * stride;
            stride *= input_shape[i];
        }
        
        float val = input[input_idx];
        
        if (first) {
            result = val;
            first = false;
        } else {
            switch (op_type) {
                case 0: result += val; break;        // Sum
                case 1: result = fmaxf(result, val); break; // Max
                case 2: result = fminf(result, val); break; // Min
            }
        }
    }
    
    // For mean, divide by the size of the reduction axis
    if (op_type == 3) { // Mean
        result /= input_shape[axis];
    }
    
    output[output_idx] = result;
}

// Launch functions
extern "C" {
    void launch_sum_f32(float* input, float* output, size_t size, int axis, cudaStream_t stream) {
        if (axis < 0) {
            // Global sum reduction
            int block_size = 256;
            int grid_size = min(65535, (int)((size + block_size - 1) / block_size));
            
            // Use optimized kernel for better performance
            sum_f32_optimized_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
            
            // If we have multiple blocks, need to sum their results
            if (grid_size > 1) {
                // Recursively reduce until we have a single value
                size_t remaining = grid_size;
                float* temp_input = output;
                
                while (remaining > 1) {
                    int new_grid = min(65535, (int)((remaining + block_size - 1) / block_size));
                    sum_f32_optimized_kernel<<<new_grid, block_size, 0, stream>>>(
                        temp_input, temp_input, remaining
                    );
                    remaining = new_grid;
                }
            }
        } else {
            // Axis-specific reduction (simplified - would need tensor shape information)
            int block_size = 256;
            int grid_size = (size + block_size - 1) / block_size;
            sum_f32_optimized_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
        }
    }
    
    void launch_mean_f32(float* input, float* output, size_t size, int axis, cudaStream_t stream) {
        // First compute sum
        launch_sum_f32(input, output, size, axis, stream);
        
        // Then divide by size (would need proper axis size in real implementation)
        int block_size = 256;
        int grid_size = 1; // For global mean, output is single value
        
        // Simple kernel to divide by size
        auto divide_kernel = [] __device__ (float* data, size_t count) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx == 0) {
                data[0] /= count;
            }
        };
        
        // Note: This is simplified - real implementation would handle axis properly
    }
    
    void launch_max_f32(float* input, float* output, size_t size, int axis, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = min(65535, (int)((size + block_size - 1) / block_size));
        
        max_f32_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
        
        // Reduce multiple block results if needed
        if (grid_size > 1) {
            size_t remaining = grid_size;
            while (remaining > 1) {
                int new_grid = min(65535, (int)((remaining + block_size - 1) / block_size));
                max_f32_kernel<<<new_grid, block_size, 0, stream>>>(output, output, remaining);
                remaining = new_grid;
            }
        }
    }
    
    void launch_min_f32(float* input, float* output, size_t size, int axis, cudaStream_t stream) {
        int block_size = 256;
        int grid_size = min(65535, (int)((size + block_size - 1) / block_size));
        
        min_f32_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
        
        // Reduce multiple block results if needed
        if (grid_size > 1) {
            size_t remaining = grid_size;
            while (remaining > 1) {
                int new_grid = min(65535, (int)((remaining + block_size - 1) / block_size));
                min_f32_kernel<<<new_grid, block_size, 0, stream>>>(output, output, remaining);
                remaining = new_grid;
            }
        }
    }
}