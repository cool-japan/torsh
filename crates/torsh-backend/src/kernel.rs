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
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
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

impl Clone for KernelHandle {
    fn clone(&self) -> Self {
        match self {
            KernelHandle::Cpu { function } => KernelHandle::Cpu {
                function: *function,
            },
            #[cfg(feature = "cuda")]
            KernelHandle::Cuda { module, function } => KernelHandle::Cuda {
                module: *module,
                function: *function,
            },
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            KernelHandle::Metal {
                library_id,
                function_id,
            } => KernelHandle::Metal {
                library_id: *library_id,
                function_id: *function_id,
            },
            #[cfg(feature = "webgpu")]
            KernelHandle::WebGpu {
                shader_module_id,
                entry_point,
            } => KernelHandle::WebGpu {
                shader_module_id: shader_module_id.clone(),
                entry_point: entry_point.clone(),
            },
            KernelHandle::Generic { .. } => {
                // For Generic handles, we can't actually clone the Box<dyn Any>
                // This is a limitation - in practice, backends should avoid using Generic handles
                // for kernels that need to be cloned
                panic!("Cannot clone Generic kernel handles")
            }
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceInfo};
    use torsh_core::{device::DeviceType, dtype::DType};

    fn create_test_device() -> Device {
        let info = DeviceInfo::default();
        Device::new(0, DeviceType::Cpu, "Test CPU".to_string(), info)
    }

    #[test]
    fn test_kernel_descriptor_creation() {
        let source = KernelSource::Source {
            code: "void main() {}".to_string(),
            language: KernelLanguage::Hlsl,
        };

        let desc = KernelDescriptor::new("test_kernel".to_string(), source);

        assert_eq!(desc.name, "test_kernel");
        assert!(desc.compile_options.is_empty());
        assert!(desc.parameters.is_empty());
        assert_eq!(desc.workgroup_size_hint, None);
        assert!(desc.cache);
    }

    #[test]
    fn test_kernel_descriptor_builder() {
        let source = KernelSource::Source {
            code: "void main() {}".to_string(),
            language: KernelLanguage::Cuda,
        };

        let param = KernelParameter::buffer("input".to_string(), DType::F32, true);

        let desc = KernelDescriptor::new("complex_kernel".to_string(), source)
            .with_compile_option("-O3".to_string())
            .with_compile_option("--fast-math".to_string())
            .with_parameter(param)
            .with_workgroup_size_hint((256, 1, 1))
            .without_cache();

        assert_eq!(desc.name, "complex_kernel");
        assert_eq!(desc.compile_options.len(), 2);
        assert!(desc.compile_options.contains(&"-O3".to_string()));
        assert!(desc.compile_options.contains(&"--fast-math".to_string()));
        assert_eq!(desc.parameters.len(), 1);
        assert_eq!(desc.workgroup_size_hint, Some((256, 1, 1)));
        assert!(!desc.cache);
    }

    #[test]
    fn test_kernel_source_variants() {
        let source1 = KernelSource::Source {
            code: "vertex main() {}".to_string(),
            language: KernelLanguage::Metal,
        };

        let source2 = KernelSource::Bytecode {
            data: vec![0x12, 0x34, 0x56, 0x78],
            format: BytecodeFormat::SpirV,
        };

        let source3 = KernelSource::SpirV {
            data: vec![0x07230203, 0x00010000],
        };

        let source4 = KernelSource::Binary {
            data: vec![0xCA, 0xFE, 0xBA, 0xBE],
            platform: "cuda".to_string(),
        };

        // Test that all variants can be created
        match source1 {
            KernelSource::Source { language, .. } => assert_eq!(language, KernelLanguage::Metal),
            _ => panic!("Wrong variant"),
        }

        match source2 {
            KernelSource::Bytecode { format, .. } => assert_eq!(format, BytecodeFormat::SpirV),
            _ => panic!("Wrong variant"),
        }

        match source3 {
            KernelSource::SpirV { .. } => {}
            _ => panic!("Wrong variant"),
        }

        match source4 {
            KernelSource::Binary { platform, .. } => assert_eq!(platform, "cuda"),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_kernel_language_variants() {
        let languages = [
            KernelLanguage::Wgsl,
            KernelLanguage::Hlsl,
            KernelLanguage::Glsl,
            KernelLanguage::Metal,
            KernelLanguage::Cuda,
            KernelLanguage::OpenCl,
            KernelLanguage::Custom("MyLang".to_string()),
        ];

        // Ensure all languages are distinct
        for (i, lang1) in languages.iter().enumerate() {
            for (j, lang2) in languages.iter().enumerate() {
                if i != j {
                    assert_ne!(lang1, lang2);
                }
            }
        }
    }

    #[test]
    fn test_bytecode_format_variants() {
        let formats = [
            BytecodeFormat::SpirV,
            BytecodeFormat::Dxil,
            BytecodeFormat::MetalAir,
            BytecodeFormat::Ptx,
            BytecodeFormat::Custom("MyFormat".to_string()),
        ];

        // Ensure all formats are distinct
        for (i, format1) in formats.iter().enumerate() {
            for (j, format2) in formats.iter().enumerate() {
                if i != j {
                    assert_ne!(format1, format2);
                }
            }
        }
    }

    #[test]
    fn test_kernel_parameter_creation() {
        let buffer_param = KernelParameter::buffer("data".to_string(), DType::F32, false);
        assert_eq!(buffer_param.name, "data");
        assert!(!buffer_param.readonly);
        assert_eq!(buffer_param.binding, None);
        match buffer_param.param_type {
            KernelParameterType::Buffer { dtype } => assert_eq!(dtype, DType::F32),
            _ => panic!("Wrong parameter type"),
        }

        let uniform_param = KernelParameter::uniform("scale".to_string(), DType::F32);
        assert_eq!(uniform_param.name, "scale");
        assert!(uniform_param.readonly);
        match uniform_param.param_type {
            KernelParameterType::Uniform { dtype } => assert_eq!(dtype, DType::F32),
            _ => panic!("Wrong parameter type"),
        }

        let bound_param = buffer_param.with_binding(0);
        assert_eq!(bound_param.binding, Some(0));
    }

    #[test]
    fn test_kernel_parameter_types() {
        let buffer_type = KernelParameterType::Buffer { dtype: DType::I32 };
        let uniform_type = KernelParameterType::Uniform { dtype: DType::F64 };
        let texture_type = KernelParameterType::Texture {
            dimensions: 2,
            dtype: DType::F32,
        };
        let sampler_type = KernelParameterType::Sampler;
        let scalar_type = KernelParameterType::Scalar { dtype: DType::U8 };

        // Test that different types are distinct
        assert_ne!(
            std::mem::discriminant(&buffer_type),
            std::mem::discriminant(&uniform_type)
        );
        assert_ne!(
            std::mem::discriminant(&uniform_type),
            std::mem::discriminant(&texture_type)
        );
        assert_ne!(
            std::mem::discriminant(&texture_type),
            std::mem::discriminant(&sampler_type)
        );
        assert_ne!(
            std::mem::discriminant(&sampler_type),
            std::mem::discriminant(&scalar_type)
        );
    }

    #[test]
    fn test_kernel_handle_cpu() {
        let handle = KernelHandle::Cpu {
            function: std::ptr::null(),
        };

        match handle {
            KernelHandle::Cpu { function } => assert!(function.is_null()),
            _ => panic!("Wrong handle type"),
        }
    }

    #[test]
    fn test_kernel_metadata_default() {
        let metadata = KernelMetadata::default();

        assert_eq!(metadata.compile_time_ms, 0.0);
        assert_eq!(metadata.binary_size, 0);
        assert_eq!(metadata.registers_per_thread, None);
        assert_eq!(metadata.shared_memory_usage, None);
        assert_eq!(metadata.max_workgroup_size, None);
        assert_eq!(metadata.compiler_version, "Unknown");
        assert!(metadata.warnings.is_empty());
        assert!(metadata.performance_hints.is_empty());
    }

    #[test]
    fn test_kernel_creation() {
        let device = create_test_device();
        let source = KernelSource::Source {
            code: "void main() {}".to_string(),
            language: KernelLanguage::Hlsl,
        };
        let desc = KernelDescriptor::new("test".to_string(), source);
        let handle = KernelHandle::Cpu {
            function: std::ptr::null(),
        };
        let metadata = KernelMetadata::default();

        let kernel = Kernel::new(
            1,
            device.clone(),
            "test_kernel".to_string(),
            desc,
            handle,
            metadata,
        );

        assert_eq!(kernel.id(), 1);
        assert_eq!(kernel.name(), "test_kernel");
        assert_eq!(kernel.device().id(), device.id());
    }

    #[test]
    fn test_kernel_launch_config_linear() {
        let config = KernelLaunchConfig::linear(1000, Some(64));

        assert_eq!(config.workgroup_size, (64, 1, 1));
        assert_eq!(config.workgroup_count, (16, 1, 1)); // ceil(1000/64) = 16
        assert_eq!(config.shared_memory_size, None);
        assert_eq!(config.stream_id, None);
        assert_eq!(config.total_threads(), 64 * 16); // 1024 total threads

        let config_default = KernelLaunchConfig::linear(1000, None);
        assert_eq!(config_default.workgroup_size, (256, 1, 1));
        assert_eq!(config_default.workgroup_count, (4, 1, 1)); // ceil(1000/256) = 4
    }

    #[test]
    fn test_kernel_launch_config_2d() {
        let config = KernelLaunchConfig::grid_2d((100, 50), Some((10, 5)));

        assert_eq!(config.workgroup_size, (10, 5, 1));
        assert_eq!(config.workgroup_count, (10, 10, 1)); // ceil(100/10), ceil(50/5)
        assert_eq!(config.total_threads(), 10 * 5 * 10 * 10); // 5000 total threads

        let config_default = KernelLaunchConfig::grid_2d((100, 50), None);
        assert_eq!(config_default.workgroup_size, (16, 16, 1));
        assert_eq!(config_default.workgroup_count, (7, 4, 1)); // ceil(100/16), ceil(50/16)
    }

    #[test]
    fn test_kernel_launch_config_3d() {
        let config = KernelLaunchConfig::grid_3d((64, 32, 16), Some((8, 4, 2)));

        assert_eq!(config.workgroup_size, (8, 4, 2));
        assert_eq!(config.workgroup_count, (8, 8, 8)); // ceil(64/8), ceil(32/4), ceil(16/2)
        assert_eq!(config.total_threads(), 8 * 4 * 2 * 8 * 8 * 8); // 32768 total threads

        let config_default = KernelLaunchConfig::grid_3d((64, 32, 16), None);
        assert_eq!(config_default.workgroup_size, (8, 8, 8));
        assert_eq!(config_default.workgroup_count, (8, 4, 2)); // ceil(64/8), ceil(32/8), ceil(16/8)
    }

    #[test]
    fn test_kernel_launch_config_builder() {
        let config = KernelLaunchConfig::linear(1000, Some(128))
            .with_shared_memory(4096)
            .with_stream(1);

        assert_eq!(config.shared_memory_size, Some(4096));
        assert_eq!(config.stream_id, Some(1));
    }
}
