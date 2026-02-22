//! OpenCL kernel compiler.

use super::common_types::{
    GeneratedKernel, KernelDataType, KernelOperation, KernelSpec, ReductionOp,
};
use crate::error::BackendError;

/// OpenCL kernel compiler
pub struct OpenCLCompiler {
    opencl_available: bool,
}

impl OpenCLCompiler {
    pub fn new() -> Self {
        Self {
            opencl_available: Self::check_opencl_availability(),
        }
    }

    fn check_opencl_availability() -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1").exists()
                || std::path::Path::new("/usr/lib/libOpenCL.so").exists()
                || std::path::Path::new("/opt/intel/opencl/lib64/libOpenCL.so").exists()
        }
        #[cfg(target_os = "windows")]
        {
            std::path::Path::new("C:\\Windows\\System32\\OpenCL.dll").exists()
        }
        #[cfg(target_os = "macos")]
        {
            std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            false
        }
    }

    pub fn generate_kernel(&mut self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        if !self.opencl_available {
            return Err(BackendError::BackendError(
                "OpenCL not available on system".to_string(),
            ));
        }
        let start_time = std::time::Instant::now();
        let source_code = match &spec.operation {
            KernelOperation::ElementwiseAdd => self.generate_opencl_elementwise_add(&spec)?,
            KernelOperation::ElementwiseMul => self.generate_opencl_elementwise_mul(&spec)?,
            KernelOperation::ElementwiseDiv => self.generate_opencl_elementwise_div(&spec)?,
            KernelOperation::ElementwiseSub => self.generate_opencl_elementwise_sub(&spec)?,
            KernelOperation::MatrixMultiply { m, n, k } => {
                self.generate_opencl_matmul(&spec, *m, *n, *k)?
            }
            KernelOperation::ReLU => self.generate_opencl_relu(&spec)?,
            KernelOperation::GELU => self.generate_opencl_gelu(&spec)?,
            KernelOperation::Softmax { dim } => self.generate_opencl_softmax(&spec, *dim)?,
            KernelOperation::Transpose { dims } => self.generate_opencl_transpose(&spec, dims)?,
            KernelOperation::Reduction { op, dim } => {
                self.generate_opencl_reduction(&spec, op, *dim)?
            }
            _ => {
                return Err(BackendError::NotImplemented(format!(
                    "OpenCL kernel generation not implemented for {:?}",
                    spec.operation
                )));
            }
        };
        let compilation_time = start_time.elapsed().as_millis() as u64;
        Ok(GeneratedKernel {
            source_code,
            entry_point: "kernel_main".to_string(),
            compiled_binary: None,
            spec,
            compilation_time_ms: compilation_time,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        })
    }

    fn generate_opencl_elementwise_add(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input_a,
    __global const {data_type}* restrict input_b,
    __global {data_type}* restrict output,
    const int size
) {{
    const int gid = get_global_id(0);
    if (gid < size) {{
        output[gid] = input_a[gid] + input_b[gid];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_opencl_elementwise_mul(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input_a,
    __global const {data_type}* restrict input_b,
    __global {data_type}* restrict output,
    const int size
) {{
    const int gid = get_global_id(0);
    if (gid < size) {{
        output[gid] = input_a[gid] * input_b[gid];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_opencl_elementwise_div(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input_a,
    __global const {data_type}* restrict input_b,
    __global {data_type}* restrict output,
    const int size
) {{
    const int gid = get_global_id(0);
    if (gid < size) {{
        output[gid] = input_a[gid] / input_b[gid];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_opencl_elementwise_sub(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input_a,
    __global const {data_type}* restrict input_b,
    __global {data_type}* restrict output,
    const int size
) {{
    const int gid = get_global_id(0);
    if (gid < size) {{
        output[gid] = input_a[gid] - input_b[gid];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_opencl_matmul(
        &self,
        spec: &KernelSpec,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let tile_size = 16;
        let source = format!(
            r#"
#define TILE_SIZE {tile_size}

__kernel void kernel_main(
    __global const {data_type}* restrict A,
    __global const {data_type}* restrict B,
    __global {data_type}* restrict C,
    const int M,
    const int N,
    const int K
) {{
    __local {data_type} tile_A[TILE_SIZE][TILE_SIZE];
    __local {data_type} tile_B[TILE_SIZE][TILE_SIZE];

    const int row = get_group_id(1) * TILE_SIZE + get_local_id(1);
    const int col = get_group_id(0) * TILE_SIZE + get_local_id(0);

    {data_type} sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {{
        const int a_row = row;
        const int a_col = tile * TILE_SIZE + get_local_id(0);
        const int b_row = tile * TILE_SIZE + get_local_id(1);
        const int b_col = col;

        if (a_row < M && a_col < K) {{
            tile_A[get_local_id(1)][get_local_id(0)] = A[a_row * K + a_col];
        }} else {{
            tile_A[get_local_id(1)][get_local_id(0)] = 0.0f;
        }}

        if (b_row < K && b_col < N) {{
            tile_B[get_local_id(1)][get_local_id(0)] = B[b_row * N + b_col];
        }} else {{
            tile_B[get_local_id(1)][get_local_id(0)] = 0.0f;
        }}

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; ++i) {{
            sum += tile_A[get_local_id(1)][i] * tile_B[i][get_local_id(0)];
        }}

        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#,
            data_type = data_type,
            tile_size = tile_size
        );
        Ok(source)
    }

    fn generate_opencl_relu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input,
    __global {data_type}* restrict output,
    const int size
) {{
    const int gid = get_global_id(0);
    if (gid < size) {{
        output[gid] = max(input[gid], ({data_type})0.0f);
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_opencl_gelu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input,
    __global {data_type}* restrict output,
    const int size
) {{
    const int gid = get_global_id(0);
    if (gid < size) {{
        const {data_type} x = input[gid];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const {data_type} sqrt_2_over_pi = 0.7978845608f;
        const {data_type} a = 0.044715f;
        const {data_type} inner = sqrt_2_over_pi * (x + a * x * x * x);
        output[gid] = 0.5f * x * (1.0f + tanh(inner));
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_opencl_softmax(
        &self,
        spec: &KernelSpec,
        _dim: usize,
    ) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input,
    __global {data_type}* restrict output,
    const int size
) {{
    const int gid = get_global_id(0);

    // Find maximum for numerical stability
    {data_type} max_val = input[0];
    for (int i = 1; i < size; ++i) {{
        max_val = max(max_val, input[i]);
    }}

    // Compute exponentials and sum
    {data_type} sum = 0.0f;
    for (int i = 0; i < size; ++i) {{
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }}

    // Normalize
    if (gid < size) {{
        output[gid] = output[gid] / sum;
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_opencl_transpose(
        &self,
        spec: &KernelSpec,
        _dims: &[usize],
    ) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input,
    __global {data_type}* restrict output,
    const int rows,
    const int cols
) {{
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row < rows && col < cols) {{
        output[col * rows + row] = input[row * cols + col];
    }}
}}
"#,
            data_type = data_type
        );
        Ok(source)
    }

    fn generate_opencl_reduction(
        &self,
        spec: &KernelSpec,
        op: &ReductionOp,
        _dim: Option<usize>,
    ) -> Result<String, BackendError> {
        let data_type = self.opencl_type(spec.output_type)?;
        let (op_name, identity, combine_op) = match op {
            ReductionOp::Sum => ("sum", "0.0f", "+"),
            ReductionOp::Max => ("max", "-INFINITY", "max"),
            ReductionOp::Min => ("min", "INFINITY", "min"),
            ReductionOp::Mean => ("mean", "0.0f", "+"),
            ReductionOp::Product => ("product", "1.0f", "*"),
        };
        let source = format!(
            r#"
__kernel void kernel_main(
    __global const {data_type}* restrict input,
    __global {data_type}* restrict output,
    const int size,
    __local {data_type}* local_data
) {{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int group_size = get_local_size(0);

    // Load data into local memory
    {data_type} value = {identity};
    if (gid < size) {{
        value = input[gid];
    }}
    local_data[lid] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduce within workgroup
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {{
        if (lid < stride && gid + stride < size) {{
            local_data[lid] = {combine_expr};
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}

    // Write result for this workgroup
    if (lid == 0) {{
        {data_type} result = local_data[0];
        {post_process}
        output[get_group_id(0)] = result;
    }}
}}
"#,
            data_type = data_type,
            identity = identity,
            combine_expr = if op_name == "max" || op_name == "min" {
                format!("{}(local_data[lid], local_data[lid + stride])", combine_op)
            } else {
                format!("local_data[lid] {} local_data[lid + stride]", combine_op)
            },
            post_process = if matches!(op, ReductionOp::Mean) {
                "result = result / size;"
            } else {
                ""
            }
        );
        Ok(source)
    }

    pub(crate) fn opencl_type(
        &self,
        data_type: KernelDataType,
    ) -> Result<&'static str, BackendError> {
        match data_type {
            KernelDataType::F32 => Ok("float"),
            KernelDataType::F64 => Ok("double"),
            KernelDataType::I32 => Ok("int"),
            KernelDataType::I64 => Ok("long"),
            KernelDataType::U32 => Ok("uint"),
            KernelDataType::U64 => Ok("ulong"),
            KernelDataType::F16 => Ok("half"),
            KernelDataType::BF16 => Err(BackendError::NotImplemented(
                "BF16 not supported in OpenCL".to_string(),
            )),
        }
    }
}
