//! WGSL Kernel Library for ToRSh WebGPU Backend

#[cfg(feature = "webgpu")]
use crate::webgpu::wgpu;
//!
//! This module provides a comprehensive collection of WGSL compute shaders
//! for common tensor operations, optimized for wgpu 26.0.1.

use std::collections::HashMap;

/// Collection of WGSL compute shaders for tensor operations
pub struct WgslKernels;

impl WgslKernels {
    /// Get all available kernels as a map
    pub fn all_kernels() -> HashMap<&'static str, &'static str> {
        let mut kernels = HashMap::new();

        // Arithmetic operations
        kernels.insert("add_f32", Self::ADD_F32);
        kernels.insert("mul_f32", Self::MUL_F32);
        kernels.insert("sub_f32", Self::SUB_F32);
        kernels.insert("div_f32", Self::DIV_F32);

        // Unary operations
        kernels.insert("relu_f32", Self::RELU_F32);
        kernels.insert("sigmoid_f32", Self::SIGMOID_F32);
        kernels.insert("tanh_f32", Self::TANH_F32);
        kernels.insert("exp_f32", Self::EXP_F32);
        kernels.insert("log_f32", Self::LOG_F32);
        kernels.insert("sqrt_f32", Self::SQRT_F32);

        // Matrix operations
        kernels.insert("matmul_f32", Self::MATMUL_F32);
        kernels.insert("transpose_f32", Self::TRANSPOSE_F32);

        // Reduction operations
        kernels.insert("sum_f32", Self::SUM_F32);
        kernels.insert("max_f32", Self::MAX_F32);
        kernels.insert("min_f32", Self::MIN_F32);

        // Utility operations
        kernels.insert("fill_f32", Self::FILL_F32);
        kernels.insert("copy_f32", Self::COPY_F32);

        kernels
    }

    /// Element-wise addition of two f32 arrays
    pub const ADD_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = input_a[index] + input_b[index];
}
"#;

    /// Element-wise multiplication of two f32 arrays
    pub const MUL_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = input_a[index] * input_b[index];
}
"#;

    /// Element-wise subtraction of two f32 arrays
    pub const SUB_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = input_a[index] - input_b[index];
}
"#;

    /// Element-wise division of two f32 arrays
    pub const DIV_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = input_a[index] / input_b[index];
}
"#;

    /// ReLU activation function
    pub const RELU_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = max(0.0, input[index]);
}
"#;

    /// Sigmoid activation function
    pub const SIGMOID_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = 1.0 / (1.0 + exp(-input[index]));
}
"#;

    /// Tanh activation function
    pub const TANH_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = tanh(input[index]);
}
"#;

    /// Element-wise exponential
    pub const EXP_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = exp(input[index]);
}
"#;

    /// Element-wise natural logarithm
    pub const LOG_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = log(input[index]);
}
"#;

    /// Element-wise square root
    pub const SQRT_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = sqrt(input[index]);
}
"#;

    /// Matrix multiplication (optimized for small-medium matrices)
    pub const MATMUL_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // [M, N, K, _]

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let M = dims.x;
    let N = dims.y;
    let K = dims.z;

    let row = global_id.y;
    let col = global_id.x;

    if (row >= M || col >= N) {
        return;
    }

    var sum = 0.0;

    for (var tile = 0u; tile < (K + 15u) / 16u; tile++) {
        // Load tiles into workgroup memory
        let a_idx = row * K + tile * 16u + local_id.x;
        let b_idx = (tile * 16u + local_id.y) * N + col;

        if (tile * 16u + local_id.x < K) {
            tile_a[local_id.y][local_id.x] = matrix_a[a_idx];
        } else {
            tile_a[local_id.y][local_id.x] = 0.0;
        }

        if (tile * 16u + local_id.y < K) {
            tile_b[local_id.y][local_id.x] = matrix_b[b_idx];
        } else {
            tile_b[local_id.y][local_id.x] = 0.0;
        }

        workgroupBarrier();

        // Compute partial sum
        for (var k = 0u; k < 16u; k++) {
            sum += tile_a[local_id.y][k] * tile_b[k][local_id.x];
        }

        workgroupBarrier();
    }

    result[row * N + col] = sum;
}
"#;

    /// Matrix transpose
    pub const TRANSPOSE_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>; // [rows, cols]

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rows = dims.x;
    let cols = dims.y;

    let src_row = global_id.y;
    let src_col = global_id.x;

    if (src_row >= rows || src_col >= cols) {
        return;
    }

    let src_idx = src_row * cols + src_col;
    let dst_idx = src_col * rows + src_row;

    output[dst_idx] = input[src_idx];
}
"#;

    /// Sum reduction along the last dimension
    pub const SUM_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory
    if (gid < size) {
        shared_sum[tid] = input[gid];
    } else {
        shared_sum[tid] = 0.0;
    }

    workgroupBarrier();

    // Tree reduction in shared memory
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    // Write result
    if (tid == 0u) {
        output[group_id.x] = shared_sum[0];
    }
}
"#;

    /// Max reduction along the last dimension
    pub const MAX_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

var<workgroup> shared_max: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory
    if (gid < size) {
        shared_max[tid] = input[gid];
    } else {
        shared_max[tid] = -3.40282347e+38; // -f32::MAX
    }

    workgroupBarrier();

    // Tree reduction in shared memory
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        workgroupBarrier();
    }

    // Write result
    if (tid == 0u) {
        output[group_id.x] = shared_max[0];
    }
}
"#;

    /// Min reduction along the last dimension
    pub const MIN_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

var<workgroup> shared_min: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory
    if (gid < size) {
        shared_min[tid] = input[gid];
    } else {
        shared_min[tid] = 3.40282347e+38; // f32::MAX
    }

    workgroupBarrier();

    // Tree reduction in shared memory
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
        }
        workgroupBarrier();
    }

    // Write result
    if (tid == 0u) {
        output[group_id.x] = shared_min[0];
    }
}
"#;

    /// Fill array with a constant value
    pub const FILL_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> value: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = value;
}
"#;

    /// Copy array elements
    pub const COPY_F32: &'static str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&output)) {
        return;
    }
    output[index] = input[index];
}
"#;
}