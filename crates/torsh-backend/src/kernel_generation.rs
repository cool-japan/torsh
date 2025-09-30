//! Custom kernel generation and compilation system
//!
//! This module provides runtime kernel generation capabilities for optimizing
//! tensor operations based on specific input characteristics and hardware features.
//! It supports multiple compilation backends including LLVM, SPIR-V, and OpenCL.

use crate::error::BackendError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Kernel data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelDataType {
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    F16,
    BF16,
}

impl KernelDataType {
    /// Get the size of the data type in bytes
    pub fn size(&self) -> usize {
        match self {
            KernelDataType::F32 | KernelDataType::I32 | KernelDataType::U32 => 4,
            KernelDataType::F64 | KernelDataType::I64 | KernelDataType::U64 => 8,
            KernelDataType::F16 | KernelDataType::BF16 => 2,
        }
    }

    /// Get the C/CUDA type name
    pub fn to_c_type(&self) -> &'static str {
        match self {
            KernelDataType::F32 => "float",
            KernelDataType::F64 => "double",
            KernelDataType::I32 => "int",
            KernelDataType::I64 => "long long",
            KernelDataType::U32 => "unsigned int",
            KernelDataType::U64 => "unsigned long long",
            KernelDataType::F16 => "half",
            KernelDataType::BF16 => "__nv_bfloat16",
        }
    }

    /// Get the SPIR-V type
    pub fn to_spirv_type(&self) -> &'static str {
        match self {
            KernelDataType::F32 => "f32",
            KernelDataType::F64 => "f64",
            KernelDataType::I32 => "i32",
            KernelDataType::I64 => "i64",
            KernelDataType::U32 => "u32",
            KernelDataType::U64 => "u64",
            KernelDataType::F16 => "f16",
            KernelDataType::BF16 => "bf16",
        }
    }
}

/// Kernel operation types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelOperation {
    ElementwiseAdd,
    ElementwiseMul,
    ElementwiseDiv,
    ElementwiseSub,
    MatrixMultiply {
        m: usize,
        n: usize,
        k: usize,
    },
    Convolution2D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    Reduction {
        op: ReductionOp,
        dim: Option<usize>,
    },
    Transpose {
        dims: Vec<usize>,
    },
    Softmax {
        dim: usize,
    },
    LayerNorm {
        normalized_shape: Vec<usize>,
    },
    BatchNorm {
        num_features: usize,
    },
    ReLU,
    GELU,
    Custom {
        name: String,
    },
}

/// Reduction operation types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    Sum,
    Max,
    Min,
    Mean,
    Product,
}

/// Target compilation backend
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationTarget {
    CUDA { compute_capability: (u32, u32) },
    OpenCL { version: String },
    CPU { architecture: String },
    WebGPU,
    SPIRV,
    LLVM,
}

/// Kernel optimization flags
#[derive(Debug, Clone)]
pub struct OptimizationFlags {
    pub vectorization: bool,
    pub loop_unrolling: bool,
    pub memory_coalescing: bool,
    pub shared_memory_usage: bool,
    pub tensor_cores: bool,
    pub auto_tuning: bool,
    pub aggressive_inlining: bool,
    pub math_optimizations: bool,
}

impl Default for OptimizationFlags {
    fn default() -> Self {
        Self {
            vectorization: true,
            loop_unrolling: true,
            memory_coalescing: true,
            shared_memory_usage: true,
            tensor_cores: false,
            auto_tuning: false,
            aggressive_inlining: false,
            math_optimizations: true,
        }
    }
}

/// Kernel specification for generation
#[derive(Debug, Clone)]
pub struct KernelSpec {
    pub operation: KernelOperation,
    pub input_types: Vec<KernelDataType>,
    pub output_type: KernelDataType,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub target: CompilationTarget,
    pub optimization_flags: OptimizationFlags,
    pub workgroup_size: Option<(usize, usize, usize)>,
    pub shared_memory_size: Option<usize>,
}

impl KernelSpec {
    /// Create a new kernel specification
    pub fn new(
        operation: KernelOperation,
        input_types: Vec<KernelDataType>,
        output_type: KernelDataType,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        target: CompilationTarget,
    ) -> Self {
        Self {
            operation,
            input_types,
            output_type,
            input_shapes,
            output_shape,
            target,
            optimization_flags: OptimizationFlags::default(),
            workgroup_size: None,
            shared_memory_size: None,
        }
    }

    /// Enable tensor core usage if available
    pub fn with_tensor_cores(mut self) -> Self {
        self.optimization_flags.tensor_cores = true;
        self
    }

    /// Set custom workgroup size
    pub fn with_workgroup_size(mut self, size: (usize, usize, usize)) -> Self {
        self.workgroup_size = Some(size);
        self
    }

    /// Set shared memory size
    pub fn with_shared_memory(mut self, size: usize) -> Self {
        self.shared_memory_size = Some(size);
        self
    }

    /// Generate a unique hash for caching
    pub fn hash_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", self).hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

/// Generated kernel code and metadata
#[derive(Debug, Clone)]
pub struct GeneratedKernel {
    pub source_code: String,
    pub entry_point: String,
    pub compiled_binary: Option<Vec<u8>>,
    pub spec: KernelSpec,
    pub compilation_time_ms: u64,
    pub estimated_performance: f64,
    pub register_usage: Option<u32>,
    pub shared_memory_usage: Option<u32>,
}

/// Kernel cache for storing compiled kernels
pub struct KernelCache {
    cache: Arc<Mutex<HashMap<String, GeneratedKernel>>>,
    max_size: usize,
    hit_count: Arc<Mutex<u64>>,
    miss_count: Arc<Mutex<u64>>,
}

impl KernelCache {
    /// Create a new kernel cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a kernel from cache
    pub fn get(&self, key: &str) -> Option<GeneratedKernel> {
        let cache = self.cache.lock().unwrap();
        if let Some(kernel) = cache.get(key) {
            *self.hit_count.lock().unwrap() += 1;
            Some(kernel.clone())
        } else {
            *self.miss_count.lock().unwrap() += 1;
            None
        }
    }

    /// Insert a kernel into cache
    pub fn insert(&self, key: String, kernel: GeneratedKernel) {
        let mut cache = self.cache.lock().unwrap();

        // Simple LRU eviction if cache is full
        if cache.len() >= self.max_size {
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }

        cache.insert(key, kernel);
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        let hits = *self.hit_count.lock().unwrap();
        let misses = *self.miss_count.lock().unwrap();
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        CacheStatistics {
            hits,
            misses,
            total_requests: total,
            hit_rate,
            cache_size: self.cache.lock().unwrap().len(),
            max_cache_size: self.max_size,
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
        *self.hit_count.lock().unwrap() = 0;
        *self.miss_count.lock().unwrap() = 0;
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub total_requests: u64,
    pub hit_rate: f64,
    pub cache_size: usize,
    pub max_cache_size: usize,
}

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
            cache: KernelCache::new(1000), // Default cache size
            cuda_compiler: Some(CudaCompiler::new()),
            opencl_compiler: Some(OpenCLCompiler::new()),
            cpu_compiler: Some(CpuCompiler::new()),
            spirv_compiler: Some(SpirvCompiler::new()),
        }
    }

    /// Generate and compile a kernel
    pub fn generate_kernel(&mut self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        let cache_key = spec.hash_key();

        // Check cache first
        if let Some(cached_kernel) = self.cache.get(&cache_key) {
            return Ok(cached_kernel);
        }

        // Generate kernel based on target
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
            CompilationTarget::WebGPU => {
                // WebGPU uses WGSL, which we'll generate directly
                self.generate_webgpu_kernel(spec)?
            }
            CompilationTarget::LLVM => {
                return Err(BackendError::NotImplemented(
                    "LLVM compilation not yet implemented".to_string(),
                ));
            }
        };

        // Cache the generated kernel
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
                )))
            }
        };

        let compilation_time = start_time.elapsed().as_millis() as u64;

        Ok(GeneratedKernel {
            source_code,
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec,
            compilation_time_ms: compilation_time,
            estimated_performance: 1.0, // Placeholder
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
        let tile_size = 16; // Common tile size for GPU matrix multiplication

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

        // Generate the reduction operation code
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

        // For mean reduction, we need to divide by the count
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

impl Default for KernelGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// CUDA kernel compiler
pub struct CudaCompiler {
    #[allow(dead_code)]
    nvcc_path: Option<String>,
}

impl CudaCompiler {
    pub fn new() -> Self {
        Self {
            nvcc_path: Self::find_nvcc(),
        }
    }

    fn find_nvcc() -> Option<String> {
        // Try to find nvcc in common locations
        let paths = [
            "/usr/local/cuda/bin/nvcc",
            "/opt/cuda/bin/nvcc",
            "nvcc", // Assume it's in PATH
        ];

        for path in &paths {
            if std::process::Command::new(path)
                .arg("--version")
                .output()
                .is_ok()
            {
                return Some(path.to_string());
            }
        }
        None
    }

    pub fn generate_kernel(&mut self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        let start_time = std::time::Instant::now();

        let source_code = match &spec.operation {
            KernelOperation::ElementwiseAdd => self.generate_cuda_elementwise_add(&spec)?,
            KernelOperation::ElementwiseMul => self.generate_cuda_elementwise_mul(&spec)?,
            KernelOperation::MatrixMultiply { m, n, k } => {
                self.generate_cuda_matmul(&spec, *m, *n, *k)?
            }
            _ => {
                return Err(BackendError::NotImplemented(format!(
                    "CUDA kernel generation not implemented for {:?}",
                    spec.operation
                )))
            }
        };

        let compilation_time = start_time.elapsed().as_millis() as u64;

        Ok(GeneratedKernel {
            source_code,
            entry_point: "kernel_main".to_string(),
            compiled_binary: None, // Would compile with nvcc in a real implementation
            spec,
            compilation_time_ms: compilation_time,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        })
    }

    fn generate_cuda_elementwise_add(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_c_type();
        let _block_size = spec.workgroup_size.unwrap_or((256, 1, 1)).0;

        let source = format!(
            r#"
extern "C" __global__ void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    int size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input_a[idx] + input_b[idx];
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cuda_elementwise_mul(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_c_type();

        let source = format!(
            r#"
extern "C" __global__ void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    int size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {{
        output[idx] = input_a[idx] * input_b[idx];
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cuda_matmul(
        &self,
        spec: &KernelSpec,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<String, BackendError> {
        let data_type = spec.output_type.to_c_type();
        let tile_size = 16;

        let source = format!(
            r#"
#define TILE_SIZE {tile_size}

extern "C" __global__ void kernel_main(
    const {data_type}* __restrict__ A,
    const {data_type}* __restrict__ B,
    {data_type}* __restrict__ C,
    int M, int N, int K
) {{
    __shared__ {data_type} tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ {data_type} tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    {data_type} sum = 0.0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {{
        int a_row = row;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = col;
        
        if (a_row < M && a_col < K) {{
            tile_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        }} else {{
            tile_A[threadIdx.y][threadIdx.x] = 0.0;
        }}
        
        if (b_row < K && b_col < N) {{
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        }} else {{
            tile_B[threadIdx.y][threadIdx.x] = 0.0;
        }}
        
        __syncthreads();
        
        for (int i = 0; i < TILE_SIZE; ++i) {{
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }}
        
        __syncthreads();
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
}

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
        // Check if OpenCL is available on the system
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
                )))
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

        // Simple 2D transpose for now
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
            ReductionOp::Mean => ("mean", "0.0f", "+"), // Will divide by count later
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

    fn opencl_type(&self, data_type: KernelDataType) -> Result<&'static str, BackendError> {
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

/// CPU kernel compiler with SIMD optimization
pub struct CpuCompiler {
    compiler_available: bool,
    simd_support: CpuSimdSupport,
}

#[derive(Debug, Clone)]
pub struct CpuSimdSupport {
    pub sse2: bool,
    pub sse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool, // ARM NEON
}

impl CpuCompiler {
    pub fn new() -> Self {
        Self {
            compiler_available: Self::check_compiler_availability(),
            simd_support: Self::detect_simd_support(),
        }
    }

    fn check_compiler_availability() -> bool {
        // Check for available C compiler
        let compilers = ["gcc", "clang", "cl", "icc"];

        for compiler in &compilers {
            if std::process::Command::new(compiler)
                .arg("--version")
                .output()
                .is_ok()
            {
                return true;
            }
        }
        false
    }

    fn detect_simd_support() -> CpuSimdSupport {
        let mut support = CpuSimdSupport {
            sse2: false,
            sse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            neon: false,
        };

        // Use std::arch to detect CPU features
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "sse2")]
            {
                support.sse2 = true;
            }
            #[cfg(target_feature = "sse3")]
            {
                support.sse3 = true;
            }
            #[cfg(target_feature = "sse4.1")]
            {
                support.sse4_1 = true;
            }
            #[cfg(target_feature = "sse4.2")]
            {
                support.sse4_2 = true;
            }
            #[cfg(target_feature = "avx")]
            {
                support.avx = true;
            }
            #[cfg(target_feature = "avx2")]
            {
                support.avx2 = true;
            }
            #[cfg(target_feature = "avx512f")]
            {
                support.avx512f = true;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(target_feature = "neon")]
            {
                support.neon = true;
            }
        }

        support
    }

    pub fn generate_kernel(&mut self, spec: KernelSpec) -> Result<GeneratedKernel, BackendError> {
        if !self.compiler_available {
            return Err(BackendError::BackendError(
                "No C compiler available for CPU kernel generation".to_string(),
            ));
        }

        let start_time = std::time::Instant::now();

        let source_code = match &spec.operation {
            KernelOperation::ElementwiseAdd => self.generate_cpu_elementwise_add(&spec)?,
            KernelOperation::ElementwiseMul => self.generate_cpu_elementwise_mul(&spec)?,
            KernelOperation::ElementwiseDiv => self.generate_cpu_elementwise_div(&spec)?,
            KernelOperation::ElementwiseSub => self.generate_cpu_elementwise_sub(&spec)?,
            KernelOperation::MatrixMultiply { m, n, k } => {
                self.generate_cpu_matmul(&spec, *m, *n, *k)?
            }
            KernelOperation::ReLU => self.generate_cpu_relu(&spec)?,
            KernelOperation::GELU => self.generate_cpu_gelu(&spec)?,
            KernelOperation::Softmax { dim } => self.generate_cpu_softmax(&spec, *dim)?,
            KernelOperation::Transpose { dims } => self.generate_cpu_transpose(&spec, dims)?,
            KernelOperation::Reduction { op, dim } => {
                self.generate_cpu_reduction(&spec, op, *dim)?
            }
            _ => {
                return Err(BackendError::NotImplemented(format!(
                    "CPU kernel generation not implemented for {:?}",
                    spec.operation
                )))
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

    fn generate_cpu_elementwise_add(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let (vector_includes, vector_ops) =
            self.generate_simd_operations(&spec.operation, data_type)?;

        let source = format!(
            r#"
{vector_includes}
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    size_t size
) {{
    {vector_ops}

    // Fallback scalar implementation
    #pragma omp parallel for
    for (size_t i = vector_end; i < size; ++i) {{
        output[i] = input_a[i] + input_b[i];
    }}
}}
"#,
            data_type = data_type,
            vector_includes = vector_includes,
            vector_ops = vector_ops
        );

        Ok(source)
    }

    fn generate_cpu_elementwise_mul(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;
        let (vector_includes, vector_ops) =
            self.generate_simd_operations(&spec.operation, data_type)?;

        let source = format!(
            r#"
{vector_includes}
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    size_t size
) {{
    {vector_ops}

    // Fallback scalar implementation
    #pragma omp parallel for
    for (size_t i = vector_end; i < size; ++i) {{
        output[i] = input_a[i] * input_b[i];
    }}
}}
"#,
            data_type = data_type,
            vector_includes = vector_includes,
            vector_ops = vector_ops
        );

        Ok(source)
    }

    fn generate_cpu_elementwise_div(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;

        let source = format!(
            r#"
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    size_t size
) {{
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = input_a[i] / input_b[i];
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cpu_elementwise_sub(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;

        let source = format!(
            r#"
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input_a,
    const {data_type}* __restrict__ input_b,
    {data_type}* __restrict__ output,
    size_t size
) {{
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = input_a[i] - input_b[i];
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cpu_matmul(
        &self,
        spec: &KernelSpec,
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;

        let source = format!(
            r#"
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ A,
    const {data_type}* __restrict__ B,
    {data_type}* __restrict__ C,
    size_t M, size_t N, size_t K
) {{
    const size_t BLOCK_SIZE = 64;

    #pragma omp parallel for collapse(2)
    for (size_t bi = 0; bi < M; bi += BLOCK_SIZE) {{
        for (size_t bj = 0; bj < N; bj += BLOCK_SIZE) {{
            for (size_t bk = 0; bk < K; bk += BLOCK_SIZE) {{
                size_t i_max = (bi + BLOCK_SIZE < M) ? bi + BLOCK_SIZE : M;
                size_t j_max = (bj + BLOCK_SIZE < N) ? bj + BLOCK_SIZE : N;
                size_t k_max = (bk + BLOCK_SIZE < K) ? bk + BLOCK_SIZE : K;

                for (size_t i = bi; i < i_max; ++i) {{
                    for (size_t j = bj; j < j_max; ++j) {{
                        {data_type} sum = C[i * N + j];
                        for (size_t k = bk; k < k_max; ++k) {{
                            sum += A[i * K + k] * B[k * N + j];
                        }}
                        C[i * N + j] = sum;
                    }}
                }}
            }}
        }}
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cpu_relu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;

        let source = format!(
            r#"
#include <omp.h>
#include <algorithm>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t size
) {{
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = std::max(input[i], ({data_type})0.0);
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cpu_gelu(&self, spec: &KernelSpec) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;

        let source = format!(
            r#"
#include <omp.h>
#include <cmath>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t size
) {{
    const {data_type} sqrt_2_over_pi = 0.7978845608028654;
    const {data_type} a = 0.044715;

    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        const {data_type} x = input[i];
        const {data_type} inner = sqrt_2_over_pi * (x + a * x * x * x);
        output[i] = 0.5 * x * (1.0 + std::tanh(inner));
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cpu_softmax(&self, spec: &KernelSpec, _dim: usize) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;

        let source = format!(
            r#"
#include <omp.h>
#include <cmath>
#include <algorithm>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t size
) {{
    // Find maximum for numerical stability
    {data_type} max_val = input[0];
    for (size_t i = 1; i < size; ++i) {{
        max_val = std::max(max_val, input[i]);
    }}

    // Compute exponentials
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = std::exp(input[i] - max_val);
    }}

    // Compute sum
    {data_type} sum = 0.0;
    for (size_t i = 0; i < size; ++i) {{
        sum += output[i];
    }}

    // Normalize
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {{
        output[i] = output[i] / sum;
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cpu_transpose(
        &self,
        spec: &KernelSpec,
        _dims: &[usize],
    ) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;

        // Simple 2D transpose implementation
        let source = format!(
            r#"
#include <omp.h>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t rows,
    size_t cols
) {{
    const size_t BLOCK_SIZE = 32;

    #pragma omp parallel for collapse(2)
    for (size_t bi = 0; bi < rows; bi += BLOCK_SIZE) {{
        for (size_t bj = 0; bj < cols; bj += BLOCK_SIZE) {{
            size_t i_max = (bi + BLOCK_SIZE < rows) ? bi + BLOCK_SIZE : rows;
            size_t j_max = (bj + BLOCK_SIZE < cols) ? bj + BLOCK_SIZE : cols;

            for (size_t i = bi; i < i_max; ++i) {{
                for (size_t j = bj; j < j_max; ++j) {{
                    output[j * rows + i] = input[i * cols + j];
                }}
            }}
        }}
    }}
}}
"#,
            data_type = data_type
        );

        Ok(source)
    }

    fn generate_cpu_reduction(
        &self,
        spec: &KernelSpec,
        op: &ReductionOp,
        _dim: Option<usize>,
    ) -> Result<String, BackendError> {
        let data_type = self.cpu_type(spec.output_type)?;

        let (init_value, _combine_op) = match op {
            ReductionOp::Sum => ("0.0", "+"),
            ReductionOp::Max => match data_type {
                "float" => ("-std::numeric_limits<float>::infinity()", "std::max"),
                "double" => ("-std::numeric_limits<double>::infinity()", "std::max"),
                _ => ("0.0", "std::max"), // Fallback
            },
            ReductionOp::Min => match data_type {
                "float" => ("std::numeric_limits<float>::infinity()", "std::min"),
                "double" => ("std::numeric_limits<double>::infinity()", "std::min"),
                _ => ("0.0", "std::min"), // Fallback
            },
            ReductionOp::Mean => ("0.0", "+"),
            ReductionOp::Product => ("1.0", "*"),
        };

        let source = format!(
            r#"
#include <omp.h>
#include <limits>
#include <algorithm>

extern "C" void kernel_main(
    const {data_type}* __restrict__ input,
    {data_type}* __restrict__ output,
    size_t size
) {{
    {data_type} result = {init_value};

    #pragma omp parallel for reduction({combine_op}:result)
    for (size_t i = 0; i < size; ++i) {{
        {reduction_expr}
    }}

    {post_process}
    output[0] = result;
}}
"#,
            data_type = data_type,
            init_value = init_value,
            combine_op = match op {
                ReductionOp::Sum | ReductionOp::Mean => "+",
                ReductionOp::Product => "*",
                _ => "+", // Use + for max/min since we'll handle them specially
            },
            reduction_expr = match op {
                ReductionOp::Sum | ReductionOp::Mean | ReductionOp::Product =>
                    "result = result {} input[i];".replace(
                        "{}",
                        match op {
                            ReductionOp::Sum | ReductionOp::Mean => "+",
                            ReductionOp::Product => "*",
                            _ => "+",
                        }
                    ),
                ReductionOp::Max => "result = std::max(result, input[i]);".to_string(),
                ReductionOp::Min => "result = std::min(result, input[i]);".to_string(),
            },
            post_process = match op {
                ReductionOp::Mean => "result = result / size;",
                _ => "",
            }
        );

        Ok(source)
    }

    fn generate_simd_operations(
        &self,
        operation: &KernelOperation,
        data_type: &str,
    ) -> Result<(String, String), BackendError> {
        let includes = if self.simd_support.avx2 {
            "#include <immintrin.h>"
        } else if self.simd_support.sse2 {
            "#include <emmintrin.h>"
        } else if self.simd_support.neon {
            "#include <arm_neon.h>"
        } else {
            ""
        };

        let vector_ops = match (operation, data_type) {
            (KernelOperation::ElementwiseAdd, "float") if self.simd_support.avx2 => {
                r#"
    size_t vector_end = (size / 8) * 8;

    for (size_t i = 0; i < vector_end; i += 8) {
        __m256 a = _mm256_load_ps(&input_a[i]);
        __m256 b = _mm256_load_ps(&input_b[i]);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_store_ps(&output[i], result);
    }
"#
            }
            (KernelOperation::ElementwiseMul, "float") if self.simd_support.avx2 => {
                r#"
    size_t vector_end = (size / 8) * 8;

    for (size_t i = 0; i < vector_end; i += 8) {
        __m256 a = _mm256_load_ps(&input_a[i]);
        __m256 b = _mm256_load_ps(&input_b[i]);
        __m256 result = _mm256_mul_ps(a, b);
        _mm256_store_ps(&output[i], result);
    }
"#
            }
            (KernelOperation::ElementwiseAdd, "float") if self.simd_support.sse2 => {
                r#"
    size_t vector_end = (size / 4) * 4;

    for (size_t i = 0; i < vector_end; i += 4) {
        __m128 a = _mm_load_ps(&input_a[i]);
        __m128 b = _mm_load_ps(&input_b[i]);
        __m128 result = _mm_add_ps(a, b);
        _mm_store_ps(&output[i], result);
    }
"#
            }
            (KernelOperation::ElementwiseMul, "float") if self.simd_support.sse2 => {
                r#"
    size_t vector_end = (size / 4) * 4;

    for (size_t i = 0; i < vector_end; i += 4) {
        __m128 a = _mm_load_ps(&input_a[i]);
        __m128 b = _mm_load_ps(&input_b[i]);
        __m128 result = _mm_mul_ps(a, b);
        _mm_store_ps(&output[i], result);
    }
"#
            }
            _ => "size_t vector_end = 0; // No vectorization available",
        };

        Ok((includes.to_string(), vector_ops.to_string()))
    }

    fn cpu_type(&self, data_type: KernelDataType) -> Result<&'static str, BackendError> {
        match data_type {
            KernelDataType::F32 => Ok("float"),
            KernelDataType::F64 => Ok("double"),
            KernelDataType::I32 => Ok("int32_t"),
            KernelDataType::I64 => Ok("int64_t"),
            KernelDataType::U32 => Ok("uint32_t"),
            KernelDataType::U64 => Ok("uint64_t"),
            KernelDataType::F16 => Err(BackendError::NotImplemented(
                "F16 not supported in CPU kernels without specific compiler extensions".to_string(),
            )),
            KernelDataType::BF16 => Err(BackendError::NotImplemented(
                "BF16 not supported in CPU kernels".to_string(),
            )),
        }
    }
}

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
        // Check if glslc (Shaderc) is available for GLSL -> SPIR-V compilation
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

        // Generate GLSL compute shader source
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
                )))
            }
        };

        let compilation_time = start_time.elapsed().as_millis() as u64;

        Ok(GeneratedKernel {
            source_code: glsl_source,
            entry_point: "main".to_string(),
            compiled_binary: None, // Could compile to SPIR-V binary with glslc
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
            ReductionOp::Max => ("-3.4028235e+38", "max", "atomicMax"), // -FLT_MAX
            ReductionOp::Min => ("3.4028235e+38", "min", "atomicMin"),  // FLT_MAX
            ReductionOp::Mean => ("0.0", "+", "atomicAdd"),
            ReductionOp::Product => ("1.0", "*", "atomicExchange"), // No atomic multiply, use exchange
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
            KernelDataType::F16 => Ok("float16_t"), // Requires GL_NV_gpu_shader5 extension
            KernelDataType::BF16 => Err(BackendError::NotImplemented(
                "BF16 not supported in GLSL".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_data_type_properties() {
        assert_eq!(KernelDataType::F32.size(), 4);
        assert_eq!(KernelDataType::F64.size(), 8);
        assert_eq!(KernelDataType::I32.to_c_type(), "int");
        assert_eq!(KernelDataType::F32.to_spirv_type(), "f32");
    }

    #[test]
    fn test_kernel_spec_creation() {
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::CUDA {
                compute_capability: (7, 5),
            },
        );

        assert_eq!(spec.input_types.len(), 2);
        assert_eq!(spec.output_type, KernelDataType::F32);
        assert!(!spec.hash_key().is_empty());
    }

    #[test]
    fn test_kernel_cache() {
        let cache = KernelCache::new(2);

        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100]],
            vec![100],
            CompilationTarget::CUDA {
                compute_capability: (7, 5),
            },
        );

        let kernel = GeneratedKernel {
            source_code: "test".to_string(),
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec: spec.clone(),
            compilation_time_ms: 100,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        };

        let key = "test_key".to_string();
        cache.insert(key.clone(), kernel.clone());

        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());

        let stats = cache.statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.cache_size, 1);
    }

    #[test]
    fn test_webgpu_kernel_generation() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );

        let result = generator.generate_kernel(spec);
        assert!(result.is_ok());

        let kernel = result.unwrap();
        assert!(!kernel.source_code.is_empty());
        assert_eq!(kernel.entry_point, "main");
    }

    #[test]
    fn test_cuda_kernel_generation() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::ElementwiseMul,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::CUDA {
                compute_capability: (7, 5),
            },
        );

        let result = generator.generate_kernel(spec);
        assert!(result.is_ok());

        let kernel = result.unwrap();
        assert!(!kernel.source_code.is_empty());
        assert!(kernel.source_code.contains("__global__"));
    }

    #[test]
    fn test_matrix_multiply_kernel() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::MatrixMultiply {
                m: 128,
                n: 128,
                k: 128,
            },
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![128, 128], vec![128, 128]],
            vec![128, 128],
            CompilationTarget::WebGPU,
        );

        let result = generator.generate_kernel(spec);
        assert!(result.is_ok());

        let kernel = result.unwrap();
        assert!(kernel.source_code.contains("workgroup"));
        assert!(kernel.source_code.contains("matrix"));
    }

    #[test]
    fn test_optimization_flags() {
        let mut flags = OptimizationFlags::default();
        assert!(flags.vectorization);
        assert!(flags.loop_unrolling);

        flags.tensor_cores = true;
        assert!(flags.tensor_cores);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = KernelCache::new(1); // Very small cache

        let kernel1 = GeneratedKernel {
            source_code: "test1".to_string(),
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec: KernelSpec::new(
                KernelOperation::ElementwiseAdd,
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![100]],
                vec![100],
                CompilationTarget::WebGPU,
            ),
            compilation_time_ms: 100,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        };

        let kernel2 = GeneratedKernel {
            source_code: "test2".to_string(),
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec: KernelSpec::new(
                KernelOperation::ElementwiseMul,
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![100]],
                vec![100],
                CompilationTarget::WebGPU,
            ),
            compilation_time_ms: 100,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        };

        cache.insert("key1".to_string(), kernel1);
        cache.insert("key2".to_string(), kernel2); // Should evict key1

        assert!(cache.get("key2").is_some());
        assert_eq!(cache.statistics().cache_size, 1);
    }

    #[test]
    fn test_opencl_kernel_generation() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::OpenCL {
                version: "2.0".to_string(),
            },
        );

        let result = generator.generate_kernel(spec);
        // Should succeed if OpenCL is available, otherwise should return proper error
        match result {
            Ok(kernel) => {
                assert!(!kernel.source_code.is_empty());
                assert!(kernel.source_code.contains("__kernel"));
                assert!(kernel.source_code.contains("__global"));
                assert_eq!(kernel.entry_point, "kernel_main");
            }
            Err(BackendError::BackendError(msg)) => {
                assert!(msg.contains("OpenCL not available"));
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_opencl_matrix_multiply_kernel() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::MatrixMultiply {
                m: 64,
                n: 64,
                k: 64,
            },
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![64, 64], vec![64, 64]],
            vec![64, 64],
            CompilationTarget::OpenCL {
                version: "2.0".to_string(),
            },
        );

        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(kernel.source_code.contains("TILE_SIZE"));
                assert!(kernel.source_code.contains("__local"));
                assert!(kernel.source_code.contains("barrier"));
            }
            Err(BackendError::BackendError(_)) => {
                // OpenCL not available - test passes
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_cpu_kernel_generation() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::ElementwiseMul,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![1000], vec![1000]],
            vec![1000],
            CompilationTarget::CPU {
                architecture: "x86_64".to_string(),
            },
        );

        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(!kernel.source_code.is_empty());
                assert!(kernel.source_code.contains("extern \"C\""));
                assert!(kernel.source_code.contains("__restrict__"));
                assert!(kernel.source_code.contains("#pragma omp"));
                assert_eq!(kernel.entry_point, "kernel_main");
            }
            Err(BackendError::BackendError(msg)) => {
                assert!(msg.contains("No C compiler available"));
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_cpu_simd_support_detection() {
        let cpu_compiler = CpuCompiler::new();
        // Should not panic and should return valid support info
        let _support = &cpu_compiler.simd_support;
    }

    #[test]
    fn test_cpu_advanced_operations() {
        let mut generator = KernelGenerator::new();

        let operations = vec![
            KernelOperation::ReLU,
            KernelOperation::GELU,
            KernelOperation::Softmax { dim: 1 },
            KernelOperation::Transpose { dims: vec![0, 1] },
            KernelOperation::Reduction {
                op: ReductionOp::Sum,
                dim: Some(0),
            },
        ];

        for operation in operations {
            let spec = KernelSpec::new(
                operation,
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![100]],
                vec![100],
                CompilationTarget::CPU {
                    architecture: "x86_64".to_string(),
                },
            );

            let result = generator.generate_kernel(spec);
            match result {
                Ok(kernel) => {
                    assert!(!kernel.source_code.is_empty());
                    assert!(kernel.source_code.contains("extern \"C\""));
                }
                Err(BackendError::BackendError(_)) => {
                    // Compiler not available - acceptable
                }
                _ => panic!("Unexpected error type"),
            }
        }
    }

    #[test]
    fn test_spirv_kernel_generation() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::ElementwiseDiv,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![256], vec![256]],
            vec![256],
            CompilationTarget::SPIRV,
        );

        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(!kernel.source_code.is_empty());
                assert!(kernel.source_code.contains("#version 450"));
                assert!(kernel.source_code.contains("layout("));
                assert!(kernel.source_code.contains("gl_GlobalInvocationID"));
                assert_eq!(kernel.entry_point, "main");
            }
            Err(BackendError::BackendError(msg)) => {
                assert!(msg.contains("glslc compiler not available"));
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_spirv_advanced_operations() {
        let mut generator = KernelGenerator::new();

        let operations = vec![
            (KernelOperation::ReLU, "max"),
            (KernelOperation::GELU, "tanh"),
            (KernelOperation::Softmax { dim: 1 }, "atomicMax"),
            (
                KernelOperation::Transpose { dims: vec![0, 1] },
                "gl_GlobalInvocationID.y",
            ),
        ];

        for (operation, expected_content) in operations {
            let spec = KernelSpec::new(
                operation,
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![100]],
                vec![100],
                CompilationTarget::SPIRV,
            );

            let result = generator.generate_kernel(spec);
            match result {
                Ok(kernel) => {
                    assert!(!kernel.source_code.is_empty());
                    assert!(kernel.source_code.contains("#version 450"));
                    assert!(kernel.source_code.contains(expected_content));
                }
                Err(BackendError::BackendError(_)) => {
                    // glslc not available - acceptable
                }
                _ => panic!("Unexpected error type"),
            }
        }
    }

    #[test]
    fn test_spirv_matrix_multiply_with_shared_memory() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::MatrixMultiply {
                m: 128,
                n: 128,
                k: 128,
            },
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![128, 128], vec![128, 128]],
            vec![128, 128],
            CompilationTarget::SPIRV,
        );

        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(kernel.source_code.contains("shared"));
                assert!(kernel.source_code.contains("TILE_SIZE"));
                assert!(kernel.source_code.contains("barrier()"));
                assert!(kernel.source_code.contains("gl_WorkGroupID"));
            }
            Err(BackendError::BackendError(_)) => {
                // glslc not available - acceptable
            }
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_data_type_conversions() {
        // Test KernelDataType conversions
        assert_eq!(KernelDataType::F32.size(), 4);
        assert_eq!(KernelDataType::F64.size(), 8);
        assert_eq!(KernelDataType::F16.size(), 2);

        assert_eq!(KernelDataType::F32.to_c_type(), "float");
        assert_eq!(KernelDataType::F64.to_c_type(), "double");
        assert_eq!(KernelDataType::I32.to_c_type(), "int");

        assert_eq!(KernelDataType::F32.to_spirv_type(), "f32");
        assert_eq!(KernelDataType::U32.to_spirv_type(), "u32");
    }

    #[test]
    fn test_kernel_spec_with_custom_options() {
        let spec = KernelSpec::new(
            KernelOperation::MatrixMultiply {
                m: 256,
                n: 256,
                k: 256,
            },
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![256, 256], vec![256, 256]],
            vec![256, 256],
            CompilationTarget::CUDA {
                compute_capability: (8, 0),
            },
        )
        .with_tensor_cores()
        .with_workgroup_size((32, 32, 1))
        .with_shared_memory(49152); // 48KB

        assert!(spec.optimization_flags.tensor_cores);
        assert_eq!(spec.workgroup_size, Some((32, 32, 1)));
        assert_eq!(spec.shared_memory_size, Some(49152));
    }

    #[test]
    fn test_optimization_flags_defaults() {
        let flags = OptimizationFlags::default();
        assert!(flags.vectorization);
        assert!(flags.loop_unrolling);
        assert!(flags.memory_coalescing);
        assert!(flags.shared_memory_usage);
        assert!(!flags.tensor_cores);
        assert!(!flags.auto_tuning);
        assert!(!flags.aggressive_inlining);
        assert!(flags.math_optimizations);
    }

    #[test]
    fn test_unsupported_operations() {
        let mut generator = KernelGenerator::new();

        let spec = KernelSpec::new(
            KernelOperation::Custom {
                name: "unsupported_operation".to_string(),
            },
            vec![KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );

        let result = generator.generate_kernel(spec);
        assert!(result.is_err());
        match result {
            Err(BackendError::NotImplemented(_)) => {} // Expected
            _ => panic!("Should return NotImplemented error"),
        }
    }

    #[test]
    fn test_unsupported_data_types() {
        // Test BF16 support across different compilers
        let opencl_compiler = OpenCLCompiler::new();
        let cpu_compiler = CpuCompiler::new();

        assert!(opencl_compiler.opencl_type(KernelDataType::BF16).is_err());
        assert!(cpu_compiler.cpu_type(KernelDataType::BF16).is_err());
    }

    #[test]
    fn test_reduction_operations() {
        let operations = vec![
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
            ReductionOp::Product,
        ];

        let mut generator = KernelGenerator::new();

        for op in operations {
            let spec = KernelSpec::new(
                KernelOperation::Reduction {
                    op: op.clone(),
                    dim: Some(0),
                },
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![1000]],
                vec![1],
                CompilationTarget::WebGPU,
            );

            let result = generator.generate_kernel(spec);
            assert!(result.is_ok());
            let kernel = result.unwrap();
            assert!(!kernel.source_code.is_empty());
        }
    }

    #[test]
    fn test_kernel_hash_consistency() {
        let spec1 = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );

        let spec2 = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );

        let spec3 = KernelSpec::new(
            KernelOperation::ElementwiseMul, // Different operation
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );

        assert_eq!(spec1.hash_key(), spec2.hash_key());
        assert_ne!(spec1.hash_key(), spec3.hash_key());
    }
}
