//! SPIR-V kernel compiler for Vulkan compute shaders.

use super::common_types::{
    GeneratedKernel, KernelDataType, KernelOperation, KernelSpec, ReductionOp,
};
use crate::error::BackendError;

/// SPIR-V kernel compiler for Vulkan compute shaders
pub struct SpirvCompiler {
    glslc_available: bool,
}

impl SpirvCompiler {
    pub fn new() -> Self {
        Self {
            glslc_available: Self::check_glslc_availability(),
        }
    }

    fn check_glslc_availability() -> bool {
        std::process::Command::new("glslc")
            .arg("--version")
            .output()
            .is_ok()
    }

    pub fn generate_kernel(&mut self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        if !self.glslc_available {
            return Err(BackendError::BackendError(
                "glslc compiler not available for SPIR-V generation".to_string(),
            ));
        }
        let start_time = std::time::Instant::now();
        let glsl_source = match &spec.operation {
            KernelOperation::ElementwiseAdd => self.generate_glsl_elementwise_add(&spec)?,
            KernelOperation::ElementwiseMul => self.generate_glsl_elementwise_mul(&spec)?,
            KernelOperation::ElementwiseDiv => self.generate_glsl_elementwise_div(&spec)?,
            KernelOperation::ElementwiseSub => self.generate_glsl_elementwise_sub(&spec)?,
            KernelOperation::MatrixMultiply { m, n, k } => {
                self.generate_glsl_matmul(&spec, *m, *n, *k)?
            }
            KernelOperation::ReLU => self.generate_glsl_relu(&spec)?,
            KernelOperation::GELU => self.generate_glsl_gelu(&spec)?,
            KernelOperation::Softmax { dim } => self.generate_glsl_softmax(&spec, *dim)?,
            KernelOperation::Transpose { dims } => self.generate_glsl_transpose(&spec, dims)?,
            KernelOperation::Reduction { op, dim } => {
                self.generate_glsl_reduction(&spec, op, *dim)?
            }
            _ => {
                return Err(BackendError::NotImplemented(format!(
                    "SPIR-V kernel generation not implemented for {:?}",
                    spec.operation
                )));
            }
        };
        let compilation_time = start_time.elapsed().as_millis() as u64;
        Ok(GeneratedKernel {
            source_code: glsl_source,
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec,
            compilation_time_ms: compilation_time,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        })
    }

    fn generate_glsl_elementwise_add(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"#version 450

layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {{
    {data_type} data[];
}} input_a;

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {{
    {data_type} data[];
}} input_b;

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint size;
}};

void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= size) {{
        return;
    }}

    output_data.data[index] = input_a.data[index] + input_b.data[index];
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_glsl_elementwise_mul(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"#version 450

layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {{
    {data_type} data[];
}} input_a;

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {{
    {data_type} data[];
}} input_b;

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint size;
}};

void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= size) {{
        return;
    }}

    output_data.data[index] = input_a.data[index] * input_b.data[index];
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_glsl_elementwise_div(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"#version 450

layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {{
    {data_type} data[];
}} input_a;

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {{
    {data_type} data[];
}} input_b;

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint size;
}};

void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= size) {{
        return;
    }}

    output_data.data[index] = input_a.data[index] / input_b.data[index];
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_glsl_elementwise_sub(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"#version 450

layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer InputA {{
    {data_type} data[];
}} input_a;

layout(set = 0, binding = 1, std430) restrict readonly buffer InputB {{
    {data_type} data[];
}} input_b;

layout(set = 0, binding = 2, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint size;
}};

void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= size) {{
        return;
    }}

    output_data.data[index] = input_a.data[index] - input_b.data[index];
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_glsl_matmul(
        &self,
        spec: &KernelSpec,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let tile_size = 16;
        let source = format!(
            r#"#version 450

#define TILE_SIZE {tile_size}

layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer MatrixA {{
    {data_type} data[];
}} matrix_a;

layout(set = 0, binding = 1, std430) restrict readonly buffer MatrixB {{
    {data_type} data[];
}} matrix_b;

layout(set = 0, binding = 2, std430) restrict writeonly buffer MatrixC {{
    {data_type} data[];
}} matrix_c;

layout(push_constant) uniform PushConstants {{
    uint M;
    uint N;
    uint K;
}};

shared {data_type} tile_a[TILE_SIZE][TILE_SIZE];
shared {data_type} tile_b[TILE_SIZE][TILE_SIZE];

void main() {{
    uint row = gl_WorkGroupID.y * TILE_SIZE + gl_LocalInvocationID.y;
    uint col = gl_WorkGroupID.x * TILE_SIZE + gl_LocalInvocationID.x;

    {data_type} sum = 0.0;

    for (uint tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {{
        uint a_row = row;
        uint a_col = tile * TILE_SIZE + gl_LocalInvocationID.x;
        uint b_row = tile * TILE_SIZE + gl_LocalInvocationID.y;
        uint b_col = col;

        if (a_row < M && a_col < K) {{
            tile_a[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = matrix_a.data[a_row * K + a_col];
        }} else {{
            tile_a[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
        }}

        if (b_row < K && b_col < N) {{
            tile_b[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = matrix_b.data[b_row * N + b_col];
        }} else {{
            tile_b[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 0.0;
        }}

        barrier();

        for (uint i = 0; i < TILE_SIZE; ++i) {{
            sum += tile_a[gl_LocalInvocationID.y][i] * tile_b[i][gl_LocalInvocationID.x];
        }}

        barrier();
    }}

    if (row < M && col < N) {{
        matrix_c.data[row * N + col] = sum;
    }}
}}
"#,
            data_type = data_type,
            tile_size = tile_size
        );
        Ok(source)
    }

    fn generate_glsl_relu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"#version 450

layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {{
    {data_type} data[];
}} input_data;

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint size;
}};

void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= size) {{
        return;
    }}

    output_data.data[index] = max(input_data.data[index], 0.0);
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_glsl_gelu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"#version 450

layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {{
    {data_type} data[];
}} input_data;

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint size;
}};

void main() {{
    uint index = gl_GlobalInvocationID.x;
    if (index >= size) {{
        return;
    }}

    {data_type} x = input_data.data[index];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    {data_type} sqrt_2_over_pi = 0.7978845608;
    {data_type} a = 0.044715;
    {data_type} inner = sqrt_2_over_pi * (x + a * x * x * x);
    output_data.data[index] = 0.5 * x * (1.0 + tanh(inner));
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_glsl_softmax(
        &self,
        spec: &KernelSpec,
        _dim: usize,
    ) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"#version 450

layout(local_size_x = {}, local_size_y = {}, local_size_z = {}) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {{
    {data_type} data[];
}} input_data;

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint size;
}};

shared {data_type} max_val;
shared {data_type} sum_val;

void main() {{
    uint index = gl_GlobalInvocationID.x;
    uint local_index = gl_LocalInvocationID.x;

    // Initialize shared memory
    if (local_index == 0) {{
        max_val = -3.4028235e+38; // -FLT_MAX
        sum_val = 0.0;
    }}

    barrier();

    // Find maximum for numerical stability
    if (index < size) {{
        atomicMax(max_val, input_data.data[index]);
    }}

    barrier();

    // Compute exponentials and accumulate sum
    {data_type} exp_val = 0.0;
    if (index < size) {{
        exp_val = exp(input_data.data[index] - max_val);
        atomicAdd(sum_val, exp_val);
    }}

    barrier();

    // Store normalized result
    if (index < size) {{
        output_data.data[index] = exp_val / sum_val;
    }}
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_glsl_transpose(
        &self,
        spec: &KernelSpec,
        _dims: &[usize],
    ) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let source = format!(
            r#"#version 450

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {{
    {data_type} data[];
}} input_data;

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint rows;
    uint cols;
}};

void main() {{
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row < rows && col < cols) {{
        output_data.data[col * rows + row] = input_data.data[row * cols + col];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_glsl_reduction(
        &self,
        spec: &KernelSpec,
        op: &ReductionOp,
        _dim: Option<usize>,
    ) -> Result<String, BackendError> {
        let data_type = self.glsl_type(spec.output_type)?;
        let (identity, _combine_op, _atomic_op) = match op {
            ReductionOp::Sum => ("0.0", "+", "atomicAdd"),
            ReductionOp::Max => ("-3.4028235e+38", "max", "atomicMax"),
            ReductionOp::Min => ("3.4028235e+38", "min", "atomicMin"),
            ReductionOp::Mean => ("0.0", "+", "atomicAdd"),
            ReductionOp::Product => ("1.0", "*", "atomicExchange"),
        };
        let source = format!(
            r#"#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer Input {{
    {data_type} data[];
}} input_data;

layout(set = 0, binding = 1, std430) restrict writeonly buffer Output {{
    {data_type} data[];
}} output_data;

layout(push_constant) uniform PushConstants {{
    uint size;
}};

shared {data_type} local_data[256];

void main() {{
    uint index = gl_GlobalInvocationID.x;
    uint local_index = gl_LocalInvocationID.x;

    // Load data into local memory
    {data_type} value = {identity};
    if (index < size) {{
        value = input_data.data[index];
    }}
    local_data[local_index] = value;

    barrier();

    // Parallel reduction within workgroup
    for (uint stride = 128; stride > 0; stride >>= 1) {{
        if (local_index < stride && index + stride < size) {{
            local_data[local_index] = {combine_expr};
        }}
        barrier();
    }}

    // Write result for this workgroup
    if (local_index == 0) {{
        {data_type} result = local_data[0];
        {post_process}
        output_data.data[gl_WorkGroupID.x] = result;
    }}
}}
"#,
            data_type = data_type,
            identity = identity,
            combine_expr = match op {
                ReductionOp::Sum | ReductionOp::Mean =>
                    "local_data[local_index] + local_data[local_index + stride]",
                ReductionOp::Max =>
                    "max(local_data[local_index], local_data[local_index + stride])",
                ReductionOp::Min =>
                    "min(local_data[local_index], local_data[local_index + stride])",
                ReductionOp::Product =>
                    "local_data[local_index] * local_data[local_index + stride]",
            },
            post_process = match op {
                ReductionOp::Mean => "result = result / size;",
                _ => "",
            }
        );
        Ok(source)
    }

    fn glsl_type(&self, data_type: KernelDataType) -> Result<&'static str, BackendError> {
        match data_type {
            KernelDataType::F32 => Ok("float"),
            KernelDataType::F64 => Ok("double"),
            KernelDataType::I32 => Ok("int"),
            KernelDataType::I64 => Err(BackendError::NotImplemented(
                "I64 not widely supported in GLSL compute shaders".to_string(),
            )),
            KernelDataType::U32 => Ok("uint"),
            KernelDataType::U64 => Err(BackendError::NotImplemented(
                "U64 not widely supported in GLSL compute shaders".to_string(),
            )),
            KernelDataType::F16 => Ok("float16_t"),
            KernelDataType::BF16 => Err(BackendError::NotImplemented(
                "BF16 not supported in GLSL".to_string(),
            )),
        }
    }
}
