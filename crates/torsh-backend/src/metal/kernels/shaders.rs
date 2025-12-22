//! Metal shader source code

/// Combined Metal shader source for all kernels
pub const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Unary Operations
// ============================================================================

kernel void unary_neg_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = -input[id];
}

kernel void unary_exp_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = exp(input[id]);
}

kernel void unary_log_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = log(input[id]);
}

kernel void unary_sqrt_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = sqrt(input[id]);
}

kernel void unary_tanh_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = tanh(input[id]);
}

kernel void unary_relu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = max(input[id], 0.0f);
}

kernel void unary_abs_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = abs(input[id]);
}

kernel void unary_sin_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = sin(input[id]);
}

kernel void unary_cos_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = cos(input[id]);
}

kernel void unary_sigmoid_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = 1.0f / (1.0f + exp(-input[id]));
}

kernel void unary_gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.79788456f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    output[id] = 0.5f * x * (1.0f + tanh(inner));
}

// ============================================================================
// Binary Operations
// ============================================================================

kernel void binary_add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] + b[id];
}

kernel void binary_sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] - b[id];
}

kernel void binary_mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] * b[id];
}

kernel void binary_div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] / b[id];
}

kernel void binary_pow_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = pow(a[id], b[id]);
}

kernel void binary_max_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = max(a[id], b[id]);
}

kernel void binary_min_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = min(a[id], b[id]);
}

// ============================================================================
// Reduction Operations
// ============================================================================

kernel void reduce_sum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* shape [[buffer(2)]],
    uint3 id [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]])
{
    // Simple reduction - more sophisticated versions would use shared memory
    uint idx = id.x;
    if (idx == 0) {
        float sum = 0.0f;
        uint total_size = shape[0];
        for (uint i = 0; i < total_size; i++) {
            sum += input[i];
        }
        output[0] = sum;
    }
}

kernel void reduce_mean_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* shape [[buffer(2)]],
    uint3 id [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]])
{
    // Simple reduction - more sophisticated versions would use shared memory
    uint idx = id.x;
    if (idx == 0) {
        float sum = 0.0f;
        uint total_size = shape[0];
        for (uint i = 0; i < total_size; i++) {
            sum += input[i];
        }
        output[0] = sum / float(total_size);
    }
}

kernel void reduce_max_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* shape [[buffer(2)]],
    uint3 id [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]])
{
    // Simple reduction - more sophisticated versions would use shared memory
    uint idx = id.x;
    if (idx == 0) {
        float max_val = -INFINITY;
        uint total_size = shape[0];
        for (uint i = 0; i < total_size; i++) {
            max_val = max(max_val, input[i]);
        }
        output[0] = max_val;
    }
}

kernel void reduce_min_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* shape [[buffer(2)]],
    uint3 id [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]])
{
    // Simple reduction - more sophisticated versions would use shared memory
    uint idx = id.x;
    if (idx == 0) {
        float min_val = INFINITY;
        uint total_size = shape[0];
        for (uint i = 0; i < total_size; i++) {
            min_val = min(min_val, input[i]);
        }
        output[0] = min_val;
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

kernel void matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    device const uint* dims [[buffer(3)]], // [M, N, K]
    uint2 id [[thread_position_in_grid]])
{
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];
    
    uint row = id.y;
    uint col = id.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    
    c[row * N + col] = sum;
}

kernel void transpose_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* dims [[buffer(2)]], // [rows, cols]
    uint2 id [[thread_position_in_grid]])
{
    uint rows = dims[0];
    uint cols = dims[1];
    
    uint row = id.y;
    uint col = id.x;
    
    if (row >= rows || col >= cols) return;
    
    output[col * rows + row] = input[row * cols + col];
}

// ============================================================================
// Softmax Operations
// ============================================================================

kernel void softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const uint* params [[buffer(2)]], // [outer_size, dim_size, inner_size]
    uint id [[thread_position_in_grid]])
{
    uint outer_size = params[0];
    uint dim_size = params[1];
    uint inner_size = params[2];

    uint outer_idx = id / inner_size;
    uint inner_idx = id % inner_size;

    if (outer_idx >= outer_size) return;

    uint base_offset = outer_idx * dim_size * inner_size + inner_idx;

    // Step 1: Find max value for numerical stability
    float max_val = -INFINITY;
    for (uint i = 0; i < dim_size; i++) {
        uint idx = base_offset + i * inner_size;
        max_val = max(max_val, input[idx]);
    }

    // Step 2: Compute exp(x - max) and sum
    float sum = 0.0f;
    for (uint i = 0; i < dim_size; i++) {
        uint idx = base_offset + i * inner_size;
        float exp_val = exp(input[idx] - max_val);
        output[idx] = exp_val;
        sum += exp_val;
    }

    // Step 3: Normalize by sum
    for (uint i = 0; i < dim_size; i++) {
        uint idx = base_offset + i * inner_size;
        output[idx] /= sum;
    }
}

// ============================================================================
// Convolution Operations
// ============================================================================

kernel void conv2d_f32(
    device const float* input [[buffer(0)]],   // NCHW format
    device const float* weight [[buffer(1)]],  // OIHW format
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    device const uint* params [[buffer(4)]],   // [N, C, H, W, O, KH, KW, SH, SW, PH, PW]
    uint3 id [[thread_position_in_grid]])
{
    uint n = id.z;
    uint o = id.y;
    uint out_x = id.x;
    
    // Load parameters
    uint N = params[0];
    uint C = params[1];
    uint H = params[2];
    uint W = params[3];
    uint O = params[4];
    uint KH = params[5];
    uint KW = params[6];
    uint SH = params[7];
    uint SW = params[8];
    uint PH = params[9];
    uint PW = params[10];
    
    uint OH = (H + 2 * PH - KH) / SH + 1;
    uint OW = (W + 2 * PW - KW) / SW + 1;
    
    uint out_y = out_x / OW;
    out_x = out_x % OW;
    
    if (n >= N || o >= O || out_y >= OH || out_x >= OW) return;
    
    float sum = bias ? bias[o] : 0.0f;
    
    for (uint c = 0; c < C; c++) {
        for (uint kh = 0; kh < KH; kh++) {
            for (uint kw = 0; kw < KW; kw++) {
                int in_y = out_y * SH - PH + kh;
                int in_x = out_x * SW - PW + kw;
                
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    uint in_idx = n * C * H * W + c * H * W + in_y * W + in_x;
                    uint w_idx = o * C * KH * KW + c * KH * KW + kh * KW + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    uint out_idx = n * O * OH * OW + o * OH * OW + out_y * OW + out_x;
    output[out_idx] = sum;
}

// ============================================================================
// Pooling Operations
// ============================================================================

kernel void maxpool2d_f32(
    device const float* input [[buffer(0)]],   // NCHW format
    device float* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],   // [N, C, H, W, KH, KW, SH, SW, PH, PW]
    uint3 id [[thread_position_in_grid]])
{
    uint n = id.z;
    uint c = id.y;
    uint out_x = id.x;
    
    // Load parameters
    uint N = params[0];
    uint C = params[1];
    uint H = params[2];
    uint W = params[3];
    uint KH = params[4];
    uint KW = params[5];
    uint SH = params[6];
    uint SW = params[7];
    uint PH = params[8];
    uint PW = params[9];
    
    uint OH = (H + 2 * PH - KH) / SH + 1;
    uint OW = (W + 2 * PW - KW) / SW + 1;
    
    uint out_y = out_x / OW;
    out_x = out_x % OW;
    
    if (n >= N || c >= C || out_y >= OH || out_x >= OW) return;
    
    float max_val = -INFINITY;
    
    for (uint kh = 0; kh < KH; kh++) {
        for (uint kw = 0; kw < KW; kw++) {
            int in_y = out_y * SH - PH + kh;
            int in_x = out_x * SW - PW + kw;
            
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                uint in_idx = n * C * H * W + c * H * W + in_y * W + in_x;
                max_val = max(max_val, input[in_idx]);
            }
        }
    }
    
    uint out_idx = n * C * OH * OW + c * OH * OW + out_y * OW + out_x;
    output[out_idx] = max_val;
}

kernel void avgpool2d_f32(
    device const float* input [[buffer(0)]],   // NCHW format
    device float* output [[buffer(1)]],
    device const uint* params [[buffer(2)]],   // [N, C, H, W, KH, KW, SH, SW, PH, PW]
    uint3 id [[thread_position_in_grid]])
{
    uint n = id.z;
    uint c = id.y;
    uint out_x = id.x;
    
    // Load parameters
    uint N = params[0];
    uint C = params[1];
    uint H = params[2];
    uint W = params[3];
    uint KH = params[4];
    uint KW = params[5];
    uint SH = params[6];
    uint SW = params[7];
    uint PH = params[8];
    uint PW = params[9];
    
    uint OH = (H + 2 * PH - KH) / SH + 1;
    uint OW = (W + 2 * PW - KW) / SW + 1;
    
    uint out_y = out_x / OW;
    out_x = out_x % OW;
    
    if (n >= N || c >= C || out_y >= OH || out_x >= OW) return;
    
    float sum = 0.0f;
    uint count = 0;
    
    for (uint kh = 0; kh < KH; kh++) {
        for (uint kw = 0; kw < KW; kw++) {
            int in_y = out_y * SH - PH + kh;
            int in_x = out_x * SW - PW + kw;
            
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                uint in_idx = n * C * H * W + c * H * W + in_y * W + in_x;
                sum += input[in_idx];
                count++;
            }
        }
    }
    
    uint out_idx = n * C * OH * OW + c * OH * OW + out_y * OW + out_x;
    output[out_idx] = count > 0 ? sum / float(count) : 0.0f;
}
"#;
