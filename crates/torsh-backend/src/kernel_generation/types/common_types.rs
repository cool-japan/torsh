//! Common type definitions for kernel generation.
//!
//! Contains all shared data structures, enums, and specifications
//! used across compiler backends.

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

/// Reduction operation types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    Sum,
    Max,
    Min,
    Mean,
    Product,
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

/// CPU SIMD instruction set support
#[derive(Debug, Clone)]
pub struct CpuSimdSupport {
    pub sse2: bool,
    pub sse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
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
