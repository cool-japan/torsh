//! Main kernel generator that dispatches to backend-specific compilers.

use super::cache::KernelCache;
use super::common_types::{
    CacheStatistics, CompilationTarget, GeneratedKernel, KernelDataType, KernelOperation,
    KernelSpec, ReductionOp,
};
use super::cpu_compiler::CpuCompiler;
use super::cuda_compiler::CudaCompiler;
use super::opencl_compiler::OpenCLCompiler;
use super::spirv_compiler::SpirvCompiler;
use crate::error::BackendError;

/// Main kernel generator
pub struct KernelGenerator {
    cache: KernelCache,
    cuda_compiler: Option<CudaCompiler>,
    opencl_compiler: Option<OpenCLCompiler>,
    cpu_compiler: Option<CpuCompiler>,
    spirv_compiler: Option<SpirvCompiler>,
}

impl KernelGenerator {
    /// Create a new kernel generator
    pub fn new() -> Self {
        Self {
            cache: KernelCache::new(1000),
            cuda_compiler: Some(CudaCompiler::new()),
            opencl_compiler: Some(OpenCLCompiler::new()),
            cpu_compiler: Some(CpuCompiler::new()),
            spirv_compiler: Some(SpirvCompiler::new()),
        }
    }

    /// Generate and compile a kernel
    pub fn generate_kernel(&mut self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        let cache_key = spec.hash_key();
        if let Some(cached_kernel) = self.cache.get(&cache_key) {
            return Ok(cached_kernel);
        }
        let kernel = match &spec.target {
            CompilationTarget::CUDA { .. } => {
                if let Some(ref mut compiler) = self.cuda_compiler {
                    compiler.generate_kernel(spec)?
                } else {
                    return Err(BackendError::BackendError(
                        "CUDA compiler not available".to_string(),
                    ));
                }
            }
            CompilationTarget::OpenCL { .. } => {
                if let Some(ref mut compiler) = self.opencl_compiler {
                    compiler.generate_kernel(spec)?
                } else {
                    return Err(BackendError::BackendError(
                        "OpenCL compiler not available".to_string(),
                    ));
                }
            }
            CompilationTarget::CPU { .. } => {
                if let Some(ref mut compiler) = self.cpu_compiler {
                    compiler.generate_kernel(spec)?
                } else {
                    return Err(BackendError::BackendError(
                        "CPU compiler not available".to_string(),
                    ));
                }
            }
            CompilationTarget::SPIRV => {
                if let Some(ref mut compiler) = self.spirv_compiler {
                    compiler.generate_kernel(spec)?
                } else {
                    return Err(BackendError::BackendError(
                        "SPIRV compiler not available".to_string(),
                    ));
                }
            }
            CompilationTarget::WebGPU => self.generate_webgpu_kernel(spec)?,
            CompilationTarget::LLVM => {
                return Err(BackendError::NotImplemented(
                    "LLVM compilation not yet implemented".to_string(),
                ));
            }
        };
        self.cache.insert(cache_key, kernel.clone());
        Ok(kernel)
    }

    /// Generate a WebGPU kernel using WGSL
    fn generate_webgpu_kernel(&self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        let start_time = std::time::Instant::now();
        let source_code = match &spec.operation {
            KernelOperation::ElementwiseAdd => self.generate_webgpu_elementwise_add(&spec)?,
            KernelOperation::ElementwiseMul => self.generate_webgpu_elementwise_mul(&spec)?,
            KernelOperation::MatrixMultiply { m, n, k } => {
                self.generate_webgpu_matmul(&spec, *m, *n, *k)?
            }
            KernelOperation::ReLU => self.generate_webgpu_relu(&spec)?,
            KernelOperation::Reduction { op, dim } => {
                self.generate_webgpu_reduction(&spec, op, *dim)?
            }
            _ => {
                return Err(BackendError::NotImplemented(format!(
                    "WebGPU kernel generation not implemented for {:?}",
                    spec.operation
                )));
            }
        };
        let compilation_time = start_time.elapsed().as_millis() as u64;
        Ok(GeneratedKernel {
            source_code,
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec,
            compilation_time_ms: compilation_time,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        })
    }

    /// Generate WebGPU WGSL code for elementwise addition
    fn generate_webgpu_elementwise_add(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_spirv_type();
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"
@group(0) @binding(0) var<storage, read> input_a: array<{data_type}>;
@group(0) @binding(1) var<storage, read> input_b: array<{data_type}>;
@group(0) @binding(2) var<storage, read_write> output: array<{data_type}>;

@compute @workgroup_size({}, {}, {})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    if (index >= arrayLength(&output)) {{
        return;
    }}
    
    output[index] = input_a[index] + input_b[index];
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    /// Generate WebGPU WGSL code for elementwise multiplication
    fn generate_webgpu_elementwise_mul(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_spirv_type();
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"
@group(0) @binding(0) var<storage, read> input_a: array<{data_type}>;
@group(0) @binding(1) var<storage, read> input_b: array<{data_type}>;
@group(0) @binding(2) var<storage, read_write> output: array<{data_type}>;

@compute @workgroup_size({}, {}, {})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    if (index >= arrayLength(&output)) {{
        return;
    }}
    
    output[index] = input_a[index] * input_b[index];
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    /// Generate WebGPU WGSL code for matrix multiplication
    fn generate_webgpu_matmul(
        &self,
        spec: &KernelSpec,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_spirv_type();
        let tile_size = 16;
        let source = format!(
            r#"
@group(0) @binding(0) var<storage, read> matrix_a: array<{data_type}>;
@group(0) @binding(1) var<storage, read> matrix_b: array<{data_type}>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<{data_type}>;

var<workgroup> tile_a: array<array<{data_type}, {tile_size}>, {tile_size}>;
var<workgroup> tile_b: array<array<{data_type}, {tile_size}>, {tile_size}>;

@compute @workgroup_size({tile_size}, {tile_size}, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let row = group_id.y * {tile_size} + local_id.y;
    let col = group_id.x * {tile_size} + local_id.x;
    
    let M = {m}u;
    let N = {n}u;
    let K = {k}u;
    
    var sum: {data_type} = 0.0;
    
    for (var tile = 0u; tile < (K + {tile_size} - 1) / {tile_size}; tile++) {{
        let a_row = row;
        let a_col = tile * {tile_size} + local_id.x;
        let b_row = tile * {tile_size} + local_id.y;
        let b_col = col;
        
        if (a_row < M && a_col < K) {{
            tile_a[local_id.y][local_id.x] = matrix_a[a_row * K + a_col];
        }} else {{
            tile_a[local_id.y][local_id.x] = 0.0;
        }}
        
        if (b_row < K && b_col < N) {{
            tile_b[local_id.y][local_id.x] = matrix_b[b_row * N + b_col];
        }} else {{
            tile_b[local_id.y][local_id.x] = 0.0;
        }}
        
        workgroupBarrier();
        
        for (var i = 0u; i < {tile_size}; i++) {{
            sum += tile_a[local_id.y][i] * tile_b[i][local_id.x];
        }}
        
        workgroupBarrier();
    }}
    
    if (row < M && col < N) {{
        matrix_c[row * N + col] = sum;
    }}
}}
"#,
            data_type = data_type,
            tile_size = tile_size,
            m = m,
            n = n,
            k = k
        );
        Ok(source)
    }

    /// Generate WebGPU WGSL code for ReLU activation
    fn generate_webgpu_relu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_spirv_type();
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let source = format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<{data_type}>;
@group(0) @binding(1) var<storage, read_write> output: array<{data_type}>;

@compute @workgroup_size({}, {}, {})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let index = global_id.x;
    if (index >= arrayLength(&output)) {{
        return;
    }}
    
    output[index] = max(input[index], 0.0);
}}
"#,
            workgroup_size.0,
            workgroup_size.1,
            workgroup_size.2,
            data_type = data_type
        );
        Ok(source)
    }

    /// Generate WebGPU WGSL code for reduction operations
    fn generate_webgpu_reduction(
        &self,
        spec: &KernelSpec,
        op: &ReductionOp,
        _dim: Option<usize>,
    ) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_spirv_type();
        let workgroup_size = spec.workgroup_size.unwrap_or((256, 1, 1));
        let (init_value, reduce_op) = match op {
            ReductionOp::Sum => ("0.0", "result = result + shared_data[i];"),
            ReductionOp::Max => match spec.output_type {
                KernelDataType::F32 | KernelDataType::F64 => {
                    ("-3.402823466e+38", "result = max(result, shared_data[i]);")
                }
                _ => ("0", "result = max(result, shared_data[i]);"),
            },
            ReductionOp::Min => match spec.output_type {
                KernelDataType::F32 | KernelDataType::F64 => {
                    ("3.402823466e+38", "result = min(result, shared_data[i]);")
                }
                _ => ("2147483647", "result = min(result, shared_data[i]);"),
            },
            ReductionOp::Mean => ("0.0", "result = result + shared_data[i];"),
            ReductionOp::Product => ("1.0", "result = result * shared_data[i];"),
        };
        let post_process = if matches!(op, ReductionOp::Mean) {
            "result = result / f32(input_size);"
        } else {
            ""
        };
        let source = format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<{data_type}>;
@group(0) @binding(1) var<storage, read_write> output: array<{data_type}>;
@group(0) @binding(2) var<uniform> input_size: u32;

var<workgroup> shared_data: array<{data_type}, {workgroup_size}>;

@compute @workgroup_size({workgroup_size}, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>
) {{
    let lid = local_id.x;
    let gid = global_id.x;

    // Initialize shared memory with identity value
    var local_result: {data_type} = {init_value};

    // Each thread loads multiple elements if necessary
    var idx = gid;
    while (idx < input_size) {{
        let val = input[idx];
        {reduce_op_load}
        idx += num_groups.x * {workgroup_size};
    }}

    shared_data[lid] = local_result;
    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride = {workgroup_size}u / 2u;
    while (stride > 0u) {{
        if (lid < stride && lid + stride < {workgroup_size}u) {{
            var result = shared_data[lid];
            let i = lid + stride;
            {reduce_op}
            shared_data[lid] = result;
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    // First thread in workgroup writes result
    if (lid == 0u) {{
        var result = shared_data[0];
        {post_process}
        output[group_id.x] = result;
    }}
}}
"#,
            data_type = data_type,
            workgroup_size = workgroup_size.0,
            init_value = init_value,
            reduce_op = reduce_op,
            reduce_op_load = reduce_op
                .replace("shared_data[i]", "val")
                .replace("result = result", "local_result = local_result"),
            post_process = post_process,
        );
        Ok(source)
    }

    /// Get cache statistics
    pub fn cache_statistics(&self) -> CacheStatistics {
        self.cache.statistics()
    }

    /// Clear the kernel cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}
