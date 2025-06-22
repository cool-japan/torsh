//! Compute kernel abstraction and management

use crate::Device;
use torsh_core::dtype::DType;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Compute kernel handle
#[derive(Debug)]
pub struct Kernel {
    /// Unique kernel ID
    pub id: usize,

    /// Device this kernel is compiled for
    pub device: Device,

    /// Kernel name
    pub name: String,

    /// Kernel descriptor used for creation
    pub descriptor: KernelDescriptor,

    /// Backend-specific handle
    pub handle: KernelHandle,

    /// Kernel metadata
    pub metadata: KernelMetadata,
}

impl Kernel {
    /// Create a new kernel
    pub fn new(
        id: usize,
        device: Device,
        name: String,
        descriptor: KernelDescriptor,
        handle: KernelHandle,
        metadata: KernelMetadata,
    ) -> Self {
        Self {
            id,
            device,
            name,
            descriptor,
            handle,
            metadata,
        }
    }

    /// Get kernel ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get kernel name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the device this kernel is compiled for
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get kernel metadata
    pub fn metadata(&self) -> &KernelMetadata {
        &self.metadata
    }

    /// Get backend-specific handle
    pub fn handle(&self) -> &KernelHandle {
        &self.handle
    }
}

/// Kernel descriptor for creation
#[derive(Debug, Clone)]
pub struct KernelDescriptor {
    /// Kernel name/entry point
    pub name: String,

    /// Kernel source code or bytecode
    pub source: KernelSource,

    /// Compilation options
    pub compile_options: Vec<String>,

    /// Kernel parameters description
    pub parameters: Vec<KernelParameter>,

    /// Workgroup size hint
    pub workgroup_size_hint: Option<(u32, u32, u32)>,

    /// Whether to cache the compiled kernel
    pub cache: bool,
}

impl KernelDescriptor {
    /// Create a new kernel descriptor
    pub fn new(name: String, source: KernelSource) -> Self {
        Self {
            name,
            source,
            compile_options: Vec::new(),
            parameters: Vec::new(),
            workgroup_size_hint: None,
            cache: true,
        }
    }

    /// Add a compilation option
    pub fn with_compile_option(mut self, option: String) -> Self {
        self.compile_options.push(option);
        self
    }

    /// Add a kernel parameter
    pub fn with_parameter(mut self, param: KernelParameter) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set workgroup size hint
    pub fn with_workgroup_size_hint(mut self, size: (u32, u32, u32)) -> Self {
        self.workgroup_size_hint = Some(size);
        self
    }

    /// Disable kernel caching
    pub fn without_cache(mut self) -> Self {
        self.cache = false;
        self
    }
}

/// Kernel source code or bytecode
#[derive(Debug, Clone)]
pub enum KernelSource {
    /// High-level source code (HLSL, GLSL, etc.)
    Source {
        code: String,
        language: KernelLanguage,
    },

    /// Pre-compiled bytecode
    Bytecode {
        data: Vec<u8>,
        format: BytecodeFormat,
    },

    /// SPIR-V bytecode
    SpirV { data: Vec<u32> },

    /// Platform-specific binary
    Binary { data: Vec<u8>, platform: String },
}

/// Kernel programming language
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelLanguage {
    /// WGSL (WebGPU Shading Language)
    Wgsl,

    /// HLSL (High Level Shading Language)
    Hlsl,

    /// GLSL (OpenGL Shading Language)
    Glsl,

    /// Metal Shading Language
    Metal,

    /// CUDA C++
    Cuda,

    /// OpenCL C
    OpenCl,

    /// Custom language
    Custom(String),
}

/// Bytecode format
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BytecodeFormat {
    /// SPIR-V
    SpirV,

    /// DXIL (DirectX Intermediate Language)
    Dxil,

    /// Metal AIR (Apple Intermediate Representation)
    MetalAir,

    /// CUDA PTX
    Ptx,

    /// Custom format
    Custom(String),
}

/// Kernel parameter description
#[derive(Debug, Clone)]
pub struct KernelParameter {
    /// Parameter name
    pub name: String,

    /// Parameter type
    pub param_type: KernelParameterType,

    /// Parameter binding location
    pub binding: Option<u32>,

    /// Whether parameter is read-only
    pub readonly: bool,
}

impl KernelParameter {
    /// Create a buffer parameter
    pub fn buffer(name: String, dtype: DType, readonly: bool) -> Self {
        Self {
            name,
            param_type: KernelParameterType::Buffer { dtype },
            binding: None,
            readonly,
        }
    }

    /// Create a uniform parameter
    pub fn uniform(name: String, dtype: DType) -> Self {
        Self {
            name,
            param_type: KernelParameterType::Uniform { dtype },
            binding: None,
            readonly: true,
        }
    }

    /// Set binding location
    pub fn with_binding(mut self, binding: u32) -> Self {
        self.binding = Some(binding);
        self
    }
}

/// Kernel parameter type
#[derive(Debug, Clone)]
pub enum KernelParameterType {
    /// Buffer parameter
    Buffer { dtype: DType },

    /// Uniform data parameter
    Uniform { dtype: DType },

    /// Texture/image parameter
    Texture { dimensions: u32, dtype: DType },

    /// Sampler parameter
    Sampler,

    /// Scalar parameter
    Scalar { dtype: DType },
}

/// Backend-specific kernel handle
#[derive(Debug)]
pub enum KernelHandle {
    /// CPU kernel (function pointer)
    Cpu { function: *const () },

    /// CUDA kernel
    #[cfg(feature = "cuda")]
    Cuda { module: u64, function: u64 },

    /// Metal kernel
    #[cfg(feature = "metal")]
    Metal { library_id: u64, function_id: u64 },

    /// WebGPU kernel
    #[cfg(feature = "webgpu")]
    WebGpu {
        shader_module_id: String,
        entry_point: String,
    },

    /// Generic handle for custom backends
    Generic {
        handle: Box<dyn std::any::Any + Send + Sync>,
    },
}

unsafe impl Send for KernelHandle {}
unsafe impl Sync for KernelHandle {}

/// Kernel metadata and compilation information
#[derive(Debug, Clone)]
pub struct KernelMetadata {
    /// Compilation time in milliseconds
    pub compile_time_ms: f64,

    /// Compiled binary size in bytes
    pub binary_size: usize,

    /// Number of registers used per thread
    pub registers_per_thread: Option<u32>,

    /// Shared memory usage in bytes
    pub shared_memory_usage: Option<usize>,

    /// Maximum workgroup size
    pub max_workgroup_size: Option<(u32, u32, u32)>,

    /// Compiler version
    pub compiler_version: String,

    /// Compilation warnings
    pub warnings: Vec<String>,

    /// Performance hints from compiler
    pub performance_hints: Vec<String>,
}

impl Default for KernelMetadata {
    fn default() -> Self {
        Self {
            compile_time_ms: 0.0,
            binary_size: 0,
            registers_per_thread: None,
            shared_memory_usage: None,
            max_workgroup_size: None,
            compiler_version: "Unknown".to_string(),
            warnings: Vec::new(),
            performance_hints: Vec::new(),
        }
    }
}

/// Kernel launch configuration
#[derive(Debug, Clone)]
pub struct KernelLaunchConfig {
    /// Workgroup size (local work size)
    pub workgroup_size: (u32, u32, u32),

    /// Number of workgroups (global work size / workgroup size)
    pub workgroup_count: (u32, u32, u32),

    /// Shared memory size in bytes
    pub shared_memory_size: Option<usize>,

    /// Stream/queue for asynchronous execution
    pub stream_id: Option<usize>,
}

impl KernelLaunchConfig {
    /// Create a 1D launch configuration
    pub fn linear(global_size: u32, workgroup_size: Option<u32>) -> Self {
        let wg_size = workgroup_size.unwrap_or(256);
        let wg_count = global_size.div_ceil(wg_size);

        Self {
            workgroup_size: (wg_size, 1, 1),
            workgroup_count: (wg_count, 1, 1),
            shared_memory_size: None,
            stream_id: None,
        }
    }

    /// Create a 2D launch configuration
    pub fn grid_2d(global_size: (u32, u32), workgroup_size: Option<(u32, u32)>) -> Self {
        let wg_size = workgroup_size.unwrap_or((16, 16));
        let wg_count = (
            global_size.0.div_ceil(wg_size.0),
            global_size.1.div_ceil(wg_size.1),
        );

        Self {
            workgroup_size: (wg_size.0, wg_size.1, 1),
            workgroup_count: (wg_count.0, wg_count.1, 1),
            shared_memory_size: None,
            stream_id: None,
        }
    }

    /// Create a 3D launch configuration
    pub fn grid_3d(global_size: (u32, u32, u32), workgroup_size: Option<(u32, u32, u32)>) -> Self {
        let wg_size = workgroup_size.unwrap_or((8, 8, 8));
        let wg_count = (
            global_size.0.div_ceil(wg_size.0),
            global_size.1.div_ceil(wg_size.1),
            global_size.2.div_ceil(wg_size.2),
        );

        Self {
            workgroup_size: wg_size,
            workgroup_count: wg_count,
            shared_memory_size: None,
            stream_id: None,
        }
    }

    /// Set shared memory size
    pub fn with_shared_memory(mut self, size: usize) -> Self {
        self.shared_memory_size = Some(size);
        self
    }

    /// Set execution stream
    pub fn with_stream(mut self, stream_id: usize) -> Self {
        self.stream_id = Some(stream_id);
        self
    }

    /// Get total number of threads
    pub fn total_threads(&self) -> u64 {
        (self.workgroup_size.0 as u64)
            * (self.workgroup_size.1 as u64)
            * (self.workgroup_size.2 as u64)
            * (self.workgroup_count.0 as u64)
            * (self.workgroup_count.1 as u64)
            * (self.workgroup_count.2 as u64)
    }
}
