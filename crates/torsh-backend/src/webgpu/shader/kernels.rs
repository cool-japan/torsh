//! Pre-compiled WGSL kernels for common operations

#[cfg(feature = "webgpu")]
use crate::webgpu::wgpu;
use crate::webgpu::WebGpuDevice;

/// Optimized elementwise addition kernel with vectorization
pub const ELEMENTWISE_ADD: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> input_b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Vectorized addition for 4 elements at once
    output[index] = input_a[index] + input_b[index];
}
"#;

/// Optimized elementwise multiplication kernel with vectorization
pub const ELEMENTWISE_MUL: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> input_b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Vectorized multiplication for 4 elements at once
    output[index] = input_a[index] * input_b[index];
}
"#;

/// Optimized elementwise subtraction kernel with vectorization
pub const ELEMENTWISE_SUB: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> input_b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Vectorized subtraction for 4 elements at once
    output[index] = input_a[index] - input_b[index];
}
"#;

/// Optimized elementwise division kernel with vectorization and safety
pub const ELEMENTWISE_DIV: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> input_b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Vectorized division with safety checks for divide by zero
    let epsilon = vec4<f32>(1e-8, 1e-8, 1e-8, 1e-8);
    let safe_divisor = max(abs(input_b[index]), epsilon) * sign(input_b[index]);
    output[index] = input_a[index] / safe_divisor;
}
"#;

/// ReLU activation kernel
pub const RELU: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = max(0.0, input[index]);
}
"#;

/// Softmax activation kernel (simplified version)
pub const SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let index = global_id.x;
    let local_index = local_id.x;
    let array_size = arrayLength(&input);
    
    if (index >= array_size) {
        return;
    }
    
    // Load input value
    let value = input[index];
    shared_data[local_index] = exp(value);
    
    workgroupBarrier();
    
    // Simple sum reduction (not optimal but works)
    var sum = 0.0;
    for (var i = 0u; i < min(64u, array_size); i++) {
        sum += shared_data[i];
    }
    
    // Normalize
    output[index] = shared_data[local_index] / sum;
}
"#;

/// Optimized matrix multiplication kernel with tiling and shared memory
pub const MATRIX_MUL: &str = r#"
struct Uniforms {
    M: u32,
    N: u32,
    K: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(2) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrix_c: array<f32>;

// Shared memory tiles for better cache utilization
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    if (row >= uniforms.M || col >= uniforms.N) {
        return;
    }
    
    var sum = 0.0;
    let tile_size = 16u;
    let num_tiles = (uniforms.K + tile_size - 1) / tile_size;
    
    // Tiled matrix multiplication
    for (var t = 0u; t < num_tiles; t++) {
        let tile_k = t * tile_size + local_col;
        let tile_k_b = t * tile_size + local_row;
        
        // Load tiles into shared memory
        if (tile_k < uniforms.K && row < uniforms.M) {
            tile_a[local_row][local_col] = matrix_a[row * uniforms.K + tile_k];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }
        
        if (tile_k_b < uniforms.K && col < uniforms.N) {
            tile_b[local_row][local_col] = matrix_b[tile_k_b * uniforms.N + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial dot product using shared memory
        for (var k = 0u; k < tile_size; k++) {
            sum += tile_a[local_row][k] * tile_b[k][local_col];
        }
        
        workgroupBarrier();
    }
    
    // Store result
    if (row < uniforms.M && col < uniforms.N) {
        let c_idx = row * uniforms.N + col;
        matrix_c[c_idx] = sum;
    }
}
"#;

/// 2D Convolution kernel
pub const CONV2D: &str = r#"
struct ConvUniforms {
    input_height: u32,
    input_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    output_height: u32,
    output_width: u32,
    stride: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ConvUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_y = global_id.y;
    let out_x = global_id.x;
    
    if (out_y >= uniforms.output_height || out_x >= uniforms.output_width) {
        return;
    }
    
    var sum = 0.0;
    
    for (var ky = 0u; ky < uniforms.kernel_height; ky++) {
        for (var kx = 0u; kx < uniforms.kernel_width; kx++) {
            let in_y = out_y * uniforms.stride + ky;
            let in_x = out_x * uniforms.stride + kx;
            
            // Check bounds with padding
            if (in_y >= uniforms.padding && in_x >= uniforms.padding &&
                in_y < uniforms.input_height + uniforms.padding &&
                in_x < uniforms.input_width + uniforms.padding) {
                
                let input_y = in_y - uniforms.padding;
                let input_x = in_x - uniforms.padding;
                
                if (input_y < uniforms.input_height && input_x < uniforms.input_width) {
                    let input_idx = input_y * uniforms.input_width + input_x;
                    let kernel_idx = ky * uniforms.kernel_width + kx;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
    }
    
    let output_idx = out_y * uniforms.output_width + out_x;
    output[output_idx] = sum;
}
"#;

/// Batch normalization kernel
pub const BATCH_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> mean: array<f32>;
@group(0) @binding(2) var<storage, read> variance: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    let eps = 1e-5;
    let normalized = (input[index] - mean[0]) / sqrt(variance[0] + eps);
    output[index] = gamma[0] * normalized + beta[0];
}
"#;

/// Helper functions for creating bind group layouts
pub fn create_binary_op_layout(device: &WebGpuDevice) -> wgpu::BindGroupLayout {
    crate::webgpu::shader::layout_helpers::create_binary_op_layout(device)
}

pub fn create_unary_op_layout(device: &WebGpuDevice) -> wgpu::BindGroupLayout {
    crate::webgpu::shader::layout_helpers::create_unary_op_layout(device)
}

/// Create layout for matrix multiplication
pub fn create_matmul_layout(device: &WebGpuDevice) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Matrix Multiplication Layout"),
        entries: &[
            // Uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Matrix A
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Matrix B
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Matrix C (output)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Optimized reduction sum kernel with shared memory
pub const REDUCE_SUM: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let i = workgroup_id.x * 256u + tid;
    let array_size = arrayLength(&input);
    
    // Load data into shared memory
    shared_data[tid] = if (i < array_size) { input[i] } else { 0.0 };
    workgroupBarrier();
    
    // Reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && i + stride < array_size) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    
    // Write result for this workgroup
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
"#;

/// Optimized transpose kernel with shared memory coalescing
pub const TRANSPOSE: &str = r#"
struct TransposeUniforms {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> uniforms: TransposeUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

var<workgroup> tile: array<array<f32, 17>, 16>; // 17 to avoid bank conflicts

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;
    
    // Load tile from global memory to shared memory
    if (row < uniforms.rows && col < uniforms.cols) {
        tile[local_row][local_col] = input[row * uniforms.cols + col];
    } else {
        tile[local_row][local_col] = 0.0;
    }
    
    workgroupBarrier();
    
    // Compute transposed indices
    let trans_row = global_id.x;
    let trans_col = global_id.y;
    
    // Write transposed data to global memory
    if (trans_row < uniforms.cols && trans_col < uniforms.rows) {
        output[trans_row * uniforms.rows + trans_col] = tile[local_col][local_row];
    }
}
"#;

/// Optimized fused multiply-add kernel
pub const FUSED_MULTIPLY_ADD: &str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> input_b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> input_c: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Fused multiply-add: a * b + c
    output[index] = fma(input_a[index], input_b[index], input_c[index]);
}
"#;

/// Optimized layer normalization kernel
pub const LAYER_NORM: &str = r#"
struct LayerNormUniforms {
    size: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> uniforms: LayerNormUniforms;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_sum_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let batch_idx = global_id.y;
    let base_idx = batch_idx * uniforms.size;
    
    // Compute mean
    var sum = 0.0;
    var i = tid;
    while (i < uniforms.size) {
        sum += input[base_idx + i];
        i += 256u;
    }
    shared_sum[tid] = sum;
    workgroupBarrier();
    
    // Reduce sum
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    
    let mean = shared_sum[0] / f32(uniforms.size);
    workgroupBarrier();
    
    // Compute variance
    var sum_sq = 0.0;
    i = tid;
    while (i < uniforms.size) {
        let diff = input[base_idx + i] - mean;
        sum_sq += diff * diff;
        i += 256u;
    }
    shared_sum_sq[tid] = sum_sq;
    workgroupBarrier();
    
    // Reduce sum of squares
    stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }
    
    let variance = shared_sum_sq[0] / f32(uniforms.size);
    let inv_std = inverseSqrt(variance + uniforms.eps);
    
    // Normalize and scale
    i = tid;
    while (i < uniforms.size) {
        let normalized = (input[base_idx + i] - mean) * inv_std;
        output[base_idx + i] = gamma[i] * normalized + beta[i];
        i += 256u;
    }
}
"#;
