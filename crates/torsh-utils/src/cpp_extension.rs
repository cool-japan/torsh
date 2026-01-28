//! C++ Extension utilities for ToRSh
//!
//! This module provides utilities for building C++ extensions that integrate
//! with the ToRSh framework, similar to PyTorch's cpp_extension module.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// JIT compilation configuration
#[derive(Debug, Clone, Default)]
pub struct JitCompilationConfig {
    /// Enable just-in-time compilation
    pub enabled: bool,
    /// Cache compiled kernels
    pub cache_enabled: bool,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
    /// Optimization level for JIT (0-3)
    pub optimization_level: u8,
    /// Enable CUDA JIT compilation
    pub cuda_jit: bool,
    /// CUDA JIT cache size (in MB)
    pub cuda_cache_size: usize,
    /// Maximum number of registers for CUDA kernels
    pub cuda_max_registers: Option<u32>,
}

/// Custom operation definition
#[derive(Debug, Clone)]
pub struct CustomOpDefinition {
    /// Operation name
    pub name: String,
    /// Operation type (forward, backward, both)
    pub op_type: CustomOpType,
    /// Input tensor shapes (None means dynamic)
    pub input_shapes: Vec<Option<Vec<usize>>>,
    /// Output tensor shapes (None means dynamic)
    pub output_shapes: Vec<Option<Vec<usize>>>,
    /// CPU implementation source
    pub cpu_source: Option<String>,
    /// CUDA implementation source
    pub cuda_source: Option<String>,
    /// Custom compile flags for this operation
    pub compile_flags: Vec<String>,
    /// Operation schema for validation
    pub schema: OpSchema,
}

/// Custom operation type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CustomOpType {
    Forward,
    Backward,
    ForwardBackward,
}

/// Operation schema for validation and optimization
#[derive(Debug, Clone, Default)]
pub struct OpSchema {
    /// Input tensor types
    pub input_types: Vec<TensorType>,
    /// Output tensor types
    pub output_types: Vec<TensorType>,
    /// Whether the operation is elementwise
    pub is_elementwise: bool,
    /// Whether the operation is deterministic
    pub is_deterministic: bool,
    /// Memory requirement estimation
    pub memory_requirement: MemoryRequirement,
}

/// Tensor type information
#[derive(Debug, Clone)]
pub struct TensorType {
    /// Data type (f32, f64, i32, etc.)
    pub dtype: String,
    /// Minimum number of dimensions
    pub min_dims: usize,
    /// Maximum number of dimensions (None means unlimited)
    pub max_dims: Option<usize>,
    /// Whether the tensor can be sparse
    pub supports_sparse: bool,
}

/// Memory requirement estimation
#[derive(Debug, Clone, Default)]
pub enum MemoryRequirement {
    #[default]
    Unknown,
    /// O(1) memory
    Constant,
    /// O(n) memory where n is input size
    Linear,
    /// O(n²) memory
    Quadratic,
    /// Custom memory formula
    Custom(String),
}

/// Cross-platform build configuration
#[derive(Debug, Clone, Default)]
pub struct CrossPlatformConfig {
    /// Target platforms to build for
    pub target_platforms: Vec<TargetPlatform>,
    /// Windows-specific settings
    pub windows: WindowsConfig,
    /// macOS-specific settings
    pub macos: MacOsConfig,
    /// Linux-specific settings
    pub linux: LinuxConfig,
    /// Enable cross-compilation
    pub cross_compile: bool,
    /// Docker-based building
    pub use_docker: bool,
}

/// Target platform specification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TargetPlatform {
    WindowsX64,
    WindowsX86,
    MacOsX64,
    MacOsArm64,
    LinuxX64,
    LinuxArm64,
    LinuxAarch64,
}

/// Windows-specific build configuration
#[derive(Debug, Clone, Default)]
pub struct WindowsConfig {
    /// Visual Studio version to use
    pub vs_version: Option<String>,
    /// Windows SDK version
    pub sdk_version: Option<String>,
    /// Use clang instead of MSVC
    pub use_clang: bool,
    /// Enable Windows-specific optimizations
    pub enable_simd: bool,
}

/// macOS-specific build configuration
#[derive(Debug, Clone, Default)]
pub struct MacOsConfig {
    /// Minimum macOS version
    pub min_version: Option<String>,
    /// Xcode version to use
    pub xcode_version: Option<String>,
    /// Enable Metal Performance Shaders
    pub enable_mps: bool,
    /// Universal binary (x64 + ARM64)
    pub universal_binary: bool,
}

/// Linux-specific build configuration
#[derive(Debug, Clone, Default)]
pub struct LinuxConfig {
    /// GCC/Clang version preference
    pub compiler_preference: CompilerPreference,
    /// Enable Intel MKL
    pub enable_mkl: bool,
    /// Enable OpenMP
    pub enable_openmp: bool,
    /// Distribution-specific packages
    pub distro_packages: Vec<String>,
}

/// Compiler preference on Linux
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CompilerPreference {
    #[default]
    Auto,
    Gcc,
    Clang,
    Intel,
}

/// Configuration for building a C++ extension
#[derive(Debug, Clone)]
pub struct CppExtensionConfig {
    /// Name of the extension module
    pub name: String,
    /// Source files to compile
    pub sources: Vec<PathBuf>,
    /// Include directories
    pub include_dirs: Vec<PathBuf>,
    /// Library directories
    pub library_dirs: Vec<PathBuf>,
    /// Libraries to link
    pub libraries: Vec<String>,
    /// Extra compiler flags
    pub extra_compile_args: Vec<String>,
    /// Extra linker flags
    pub extra_link_args: Vec<String>,
    /// Whether to build with CUDA support
    pub with_cuda: bool,
    /// CUDA architectures to target
    pub cuda_archs: Vec<String>,
    /// Whether to enable debug symbols
    pub debug: bool,
    /// Output directory
    pub build_dir: PathBuf,
    /// JIT compilation settings
    pub jit_config: JitCompilationConfig,
    /// Custom operation definitions
    pub custom_ops: Vec<CustomOpDefinition>,
    /// Cross-platform build settings
    pub cross_platform: CrossPlatformConfig,
}

impl CppExtensionConfig {
    /// Create a new C++ extension configuration
    pub fn new(name: impl Into<String>, sources: Vec<PathBuf>) -> Self {
        let name = name.into();
        let build_dir = env::temp_dir().join("torsh_cpp_extensions").join(&name);

        Self {
            name,
            sources,
            include_dirs: vec![],
            library_dirs: vec![],
            libraries: vec![],
            extra_compile_args: vec![],
            extra_link_args: vec![],
            with_cuda: false,
            cuda_archs: vec![
                "sm_70".to_string(),
                "sm_75".to_string(),
                "sm_80".to_string(),
                "sm_86".to_string(),
                "sm_89".to_string(),
            ],
            debug: false,
            build_dir,
            jit_config: JitCompilationConfig::default(),
            custom_ops: vec![],
            cross_platform: CrossPlatformConfig::default(),
        }
    }

    /// Add include directory
    pub fn include_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.include_dirs.push(dir.as_ref().to_path_buf());
        self
    }

    /// Add library directory
    pub fn library_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.library_dirs.push(dir.as_ref().to_path_buf());
        self
    }

    /// Add library to link
    pub fn library(mut self, lib: impl Into<String>) -> Self {
        self.libraries.push(lib.into());
        self
    }

    /// Add extra compile arguments
    pub fn extra_compile_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_compile_args.push(arg.into());
        self
    }

    /// Add extra link arguments
    pub fn extra_link_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_link_args.push(arg.into());
        self
    }

    /// Enable CUDA support
    pub fn cuda(mut self, cuda_archs: Vec<String>) -> Self {
        self.with_cuda = true;
        self.cuda_archs = cuda_archs;
        self
    }

    /// Enable debug symbols
    pub fn debug(mut self) -> Self {
        self.debug = true;
        self
    }

    /// Set build directory
    pub fn build_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.build_dir = dir.as_ref().to_path_buf();
        self
    }

    /// Enable JIT compilation
    pub fn jit(mut self, config: JitCompilationConfig) -> Self {
        self.jit_config = config;
        self
    }

    /// Add custom operation
    pub fn custom_op(mut self, op: CustomOpDefinition) -> Self {
        self.custom_ops.push(op);
        self
    }

    /// Set cross-platform build configuration
    pub fn cross_platform(mut self, config: CrossPlatformConfig) -> Self {
        self.cross_platform = config;
        self
    }

    /// Enable JIT compilation with default settings
    pub fn enable_jit(mut self) -> Self {
        self.jit_config.enabled = true;
        self.jit_config.cache_enabled = true;
        self.jit_config.optimization_level = 2;
        self
    }

    /// Enable CUDA JIT compilation
    pub fn enable_cuda_jit(mut self) -> Self {
        self.jit_config.cuda_jit = true;
        self.jit_config.cuda_cache_size = 256; // 256 MB default
        self
    }
}

/// Build result containing the path to the compiled extension
#[derive(Debug)]
pub struct BuildResult {
    /// Path to the compiled shared library
    pub library_path: PathBuf,
    /// Include directories for using the extension
    pub include_dirs: Vec<PathBuf>,
    /// JIT compilation results
    pub jit_info: Option<JitBuildInfo>,
    /// Custom operations that were compiled
    pub compiled_ops: Vec<String>,
    /// Cross-platform build artifacts
    pub platform_artifacts: HashMap<TargetPlatform, PathBuf>,
}

/// JIT compilation build information
#[derive(Debug)]
pub struct JitBuildInfo {
    /// JIT cache directory
    pub cache_dir: PathBuf,
    /// Number of kernels compiled
    pub kernel_count: usize,
    /// CUDA JIT compilation info
    pub cuda_info: Option<CudaJitInfo>,
}

/// CUDA JIT compilation information
#[derive(Debug)]
pub struct CudaJitInfo {
    /// PTX cache size in bytes
    pub ptx_cache_size: usize,
    /// Number of CUDA kernels
    pub kernel_count: usize,
    /// GPU compute capability used
    pub compute_capability: Vec<String>,
    /// Runtime compilation cache hits
    pub cache_hits: usize,
    /// Runtime compilation cache misses
    pub cache_misses: usize,
    /// JIT compilation time in milliseconds
    pub compilation_time_ms: f64,
}

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Device index
    pub device_id: u32,
    /// Device name
    pub name: String,
    /// Compute capability (e.g., "8.0")
    pub compute_capability: String,
    /// Total global memory in bytes
    pub total_memory: usize,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum grid dimensions
    pub max_grid_size: [u32; 3],
    /// Maximum block dimensions
    pub max_block_size: [u32; 3],
    /// Warp size
    pub warp_size: u32,
    /// Number of multiprocessors
    pub multiprocessor_count: u32,
    /// Maximum shared memory per block
    pub shared_memory_per_block: usize,
}

/// Advanced CUDA kernel compilation options
#[derive(Debug, Clone)]
pub struct CudaKernelCompilationOptions {
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable fast math operations
    pub fast_math: bool,
    /// Maximum register count per thread
    pub max_registers: Option<u32>,
    /// Use cache for global memory loads
    pub use_cache: bool,
    /// Generate debug information
    pub debug_info: bool,
    /// Compile for specific GPU architecture
    pub target_arch: Option<String>,
    /// Custom compiler flags
    pub custom_flags: Vec<String>,
}

impl Default for CudaKernelCompilationOptions {
    fn default() -> Self {
        Self {
            optimization_level: 2,
            fast_math: false,
            max_registers: None,
            use_cache: true,
            debug_info: false,
            target_arch: None,
            custom_flags: vec![],
        }
    }
}

/// Runtime CUDA kernel management
#[derive(Debug)]
pub struct RuntimeCudaKernel {
    /// Kernel name
    pub name: String,
    /// PTX source code
    pub ptx_source: String,
    /// Compiled module handle (would be CUmodule in real implementation)
    pub module_handle: Option<usize>,
    /// Kernel function handle (would be CUfunction in real implementation)
    pub function_handle: Option<usize>,
    /// Compilation options used
    pub compilation_options: CudaKernelCompilationOptions,
    /// Grid and block configuration
    pub launch_config: CudaLaunchConfig,
}

/// CUDA kernel launch configuration
#[derive(Debug, Clone)]
pub struct CudaLaunchConfig {
    /// Grid dimensions
    pub grid_size: [u32; 3],
    /// Block dimensions
    pub block_size: [u32; 3],
    /// Shared memory size in bytes
    pub shared_memory_size: usize,
    /// CUDA stream handle
    pub stream: Option<usize>,
}

/// Build a C++ extension
pub fn build_cpp_extension(config: &CppExtensionConfig) -> Result<BuildResult, String> {
    // Create build directory
    fs::create_dir_all(&config.build_dir)
        .map_err(|e| format!("Failed to create build directory: {}", e))?;

    // Setup JIT compilation if enabled
    let jit_info = if config.jit_config.enabled {
        Some(setup_jit_compilation(config)?)
    } else {
        None
    };

    // Generate custom operation sources
    let mut generated_sources = vec![];
    let mut compiled_ops = vec![];

    for custom_op in &config.custom_ops {
        let generated_source = generate_custom_op_source(custom_op)?;
        generated_sources.push(generated_source);
        compiled_ops.push(custom_op.name.clone());
    }

    // Build for each target platform
    let mut platform_artifacts = HashMap::new();

    if config.cross_platform.target_platforms.is_empty() {
        // Build for current platform
        let artifact = build_for_platform(config, None, &generated_sources, &jit_info)?;
        platform_artifacts.insert(detect_current_platform(), artifact);
    } else {
        // Build for specified platforms
        for platform in &config.cross_platform.target_platforms {
            let artifact =
                build_for_platform(config, Some(platform), &generated_sources, &jit_info)?;
            platform_artifacts.insert(platform.clone(), artifact);
        }
    }

    // Get the main artifact (current platform or first specified)
    let main_artifact = platform_artifacts
        .get(&detect_current_platform())
        .or_else(|| platform_artifacts.values().next())
        .ok_or("No artifacts built")?
        .clone();

    Ok(BuildResult {
        library_path: main_artifact,
        include_dirs: config.include_dirs.clone(),
        jit_info,
        compiled_ops,
        platform_artifacts,
    })
}

/// Setup JIT compilation
fn setup_jit_compilation(config: &CppExtensionConfig) -> Result<JitBuildInfo, String> {
    let cache_dir = config
        .jit_config
        .cache_dir
        .clone()
        .unwrap_or_else(|| config.build_dir.join("jit_cache"));

    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Failed to create JIT cache directory: {}", e))?;

    let cuda_info = if config.jit_config.cuda_jit && config.with_cuda {
        Some(setup_cuda_jit(config, &cache_dir)?)
    } else {
        None
    };

    Ok(JitBuildInfo {
        cache_dir,
        kernel_count: config.custom_ops.len(),
        cuda_info,
    })
}

/// Setup CUDA JIT compilation
fn setup_cuda_jit(config: &CppExtensionConfig, cache_dir: &Path) -> Result<CudaJitInfo, String> {
    let cuda_cache_dir = cache_dir.join("cuda");
    fs::create_dir_all(&cuda_cache_dir)
        .map_err(|e| format!("Failed to create CUDA cache directory: {}", e))?;

    // Initialize CUDA runtime and query device capabilities
    let device_info = query_cuda_devices()?;
    let available_archs = device_info
        .iter()
        .map(|dev| format!("sm_{}", dev.compute_capability.replace(".", "")))
        .collect::<Vec<_>>();

    // Setup PTX cache structure
    let ptx_cache_dir = cuda_cache_dir.join("ptx");
    let cubin_cache_dir = cuda_cache_dir.join("cubin");
    fs::create_dir_all(&ptx_cache_dir)
        .map_err(|e| format!("Failed to create PTX cache directory: {}", e))?;
    fs::create_dir_all(&cubin_cache_dir)
        .map_err(|e| format!("Failed to create CUBIN cache directory: {}", e))?;

    // Configure JIT compilation options
    configure_cuda_jit_options(config)?;

    // Validate CUDA kernel sources for syntax
    for op in &config.custom_ops {
        if let Some(cuda_source) = &op.cuda_source {
            validate_cuda_kernel_syntax(cuda_source, &op.name)?;
        }
    }

    Ok(CudaJitInfo {
        ptx_cache_size: config.jit_config.cuda_cache_size * 1024 * 1024, // Convert MB to bytes
        kernel_count: config
            .custom_ops
            .iter()
            .filter(|op| op.cuda_source.is_some())
            .count(),
        compute_capability: available_archs,
        cache_hits: 0,
        cache_misses: 0,
        compilation_time_ms: 0.0,
    })
}

/// Generate custom operation source code
fn generate_custom_op_source(op: &CustomOpDefinition) -> Result<PathBuf, String> {
    // Generate C++ source code for the custom operation
    let source_content = match &op.op_type {
        CustomOpType::Forward => generate_forward_op(&op.name, &op.cpu_source, &op.cuda_source)?,
        CustomOpType::Backward => generate_backward_op(&op.name, &op.cpu_source, &op.cuda_source)?,
        CustomOpType::ForwardBackward => {
            generate_forward_backward_op(&op.name, &op.cpu_source, &op.cuda_source)?
        }
    };

    // Write to temporary file
    let temp_file = env::temp_dir().join(format!("{}_custom_op.cpp", op.name));
    fs::write(&temp_file, source_content)
        .map_err(|e| format!("Failed to write custom op source: {}", e))?;

    Ok(temp_file)
}

/// Generate forward operation source
fn generate_forward_op(
    name: &str,
    cpu_source: &Option<String>,
    cuda_source: &Option<String>,
) -> Result<String, String> {
    let mut source = format!(
        r#"// Generated custom operation: {}
#include <torsh/tensor.h>
#include <torsh/autograd.h>

namespace torsh {{
namespace ops {{

"#,
        name
    );

    // Add CPU implementation
    if let Some(cpu_impl) = cpu_source {
        source.push_str(&format!(
            r#"
// CPU implementation
Tensor {}_cpu_forward(const std::vector<Tensor>& inputs) {{
    {}
}}
"#,
            name, cpu_impl
        ));
    }

    // Add CUDA implementation
    if let Some(cuda_impl) = cuda_source {
        source.push_str(&format!(
            r#"
#ifdef TORSH_USE_CUDA
// CUDA implementation
Tensor {}_cuda_forward(const std::vector<Tensor>& inputs) {{
    {}
}}
#endif
"#,
            name, cuda_impl
        ));
    }

    // Add dispatcher
    source.push_str(&format!(
        r#"
// Operation dispatcher
Tensor {}_forward(const std::vector<Tensor>& inputs) {{
#ifdef TORSH_USE_CUDA
    if (inputs[0].is_cuda()) {{
        return {}_cuda_forward(inputs);
    }}
#endif
    return {}_cpu_forward(inputs);
}}

// Register operation
TORSH_REGISTER_OP("{}", {}_forward);

}} // namespace ops
}} // namespace torsh
"#,
        name, name, name, name, name
    ));

    Ok(source)
}

/// Generate backward operation source
fn generate_backward_op(
    name: &str,
    cpu_source: &Option<String>,
    cuda_source: &Option<String>,
) -> Result<String, String> {
    // Similar structure to forward op but for backward pass
    let mut source = format!(
        r#"// Generated custom backward operation: {}
#include <torsh/tensor.h>
#include <torsh/autograd.h>

namespace torsh {{
namespace ops {{
"#,
        name
    );

    if let Some(cpu_impl) = cpu_source {
        source.push_str(&format!(
            r#"
std::vector<Tensor> {}_cpu_backward(const std::vector<Tensor>& grad_outputs, const std::vector<Tensor>& inputs) {{
    {}
}}
"#,
            name, cpu_impl
        ));
    }

    if let Some(cuda_impl) = cuda_source {
        source.push_str(&format!(
            r#"
#ifdef TORSH_USE_CUDA
std::vector<Tensor> {}_cuda_backward(const std::vector<Tensor>& grad_outputs, const std::vector<Tensor>& inputs) {{
    {}
}}
#endif
"#,
            name, cuda_impl
        ));
    }

    source.push_str(&format!(
        r#"
std::vector<Tensor> {}_backward(const std::vector<Tensor>& grad_outputs, const std::vector<Tensor>& inputs) {{
#ifdef TORSH_USE_CUDA
    if (inputs[0].is_cuda()) {{
        return {}_cuda_backward(grad_outputs, inputs);
    }}
#endif
    return {}_cpu_backward(grad_outputs, inputs);
}}

TORSH_REGISTER_BACKWARD_OP("{}", {}_backward);

}} // namespace ops
}} // namespace torsh
"#,
        name, name, name, name, name
    ));

    Ok(source)
}

/// Generate forward and backward operation source
fn generate_forward_backward_op(
    name: &str,
    cpu_source: &Option<String>,
    cuda_source: &Option<String>,
) -> Result<String, String> {
    // Combine forward and backward generation
    let forward_source = generate_forward_op(name, cpu_source, cuda_source)?;
    let backward_source =
        generate_backward_op(&format!("{}_backward", name), cpu_source, cuda_source)?;

    Ok(format!("{}\n\n{}", forward_source, backward_source))
}

/// Build for a specific platform
fn build_for_platform(
    config: &CppExtensionConfig,
    target_platform: Option<&TargetPlatform>,
    generated_sources: &[PathBuf],
    _jit_info: &Option<JitBuildInfo>,
) -> Result<PathBuf, String> {
    // Determine compiler based on platform
    let (compiler, extra_flags) =
        match target_platform {
            Some(TargetPlatform::WindowsX64) | Some(TargetPlatform::WindowsX86) => {
                if config.cross_platform.windows.use_clang {
                    (
                        "clang++".to_string(),
                        vec![
                            "-target".to_string(),
                            get_windows_target(target_platform.expect(
                                "target_platform should be Some for Windows platform branch",
                            )),
                        ],
                    )
                } else {
                    ("cl.exe".to_string(), vec!["/std:c++17".to_string()])
                }
            }
            Some(TargetPlatform::MacOsX64) | Some(TargetPlatform::MacOsArm64) => {
                let target = match target_platform
                    .expect("target_platform should be Some for macOS platform branch")
                {
                    TargetPlatform::MacOsX64 => "x86_64-apple-darwin",
                    TargetPlatform::MacOsArm64 => "arm64-apple-darwin",
                    _ => unreachable!(),
                };
                (
                    "clang++".to_string(),
                    vec!["-target".to_string(), target.to_string()],
                )
            }
            Some(TargetPlatform::LinuxX64)
            | Some(TargetPlatform::LinuxArm64)
            | Some(TargetPlatform::LinuxAarch64) => {
                match config.cross_platform.linux.compiler_preference {
                    CompilerPreference::Clang => ("clang++".to_string(), vec![]),
                    CompilerPreference::Gcc => ("g++".to_string(), vec![]),
                    CompilerPreference::Intel => ("icpc".to_string(), vec![]),
                    CompilerPreference::Auto => (
                        env::var("CXX").unwrap_or_else(|_| "g++".to_string()),
                        vec![],
                    ),
                }
            }
            None => {
                // Current platform
                if config.with_cuda {
                    ("nvcc".to_string(), vec![])
                } else {
                    (
                        env::var("CXX").unwrap_or_else(|_| "c++".to_string()),
                        vec![],
                    )
                }
            }
        };

    // Build compile command
    let mut cmd = Command::new(&compiler);

    // Add platform-specific flags
    cmd.args(&extra_flags);

    // Add include directories
    for include_dir in &config.include_dirs {
        cmd.arg(format!("-I{}", include_dir.display()));
    }

    // Add ToRSh include directory
    if let Ok(torsh_include) = env::var("TORSH_INCLUDE_DIR") {
        cmd.arg(format!("-I{}", torsh_include));
    }

    // Add standard flags (platform-specific)
    if compiler.contains("cl.exe") {
        // MSVC flags
        cmd.arg("/std:c++17");
        if !config.debug {
            cmd.arg("/O2");
            cmd.arg("/DNDEBUG");
        } else {
            cmd.arg("/Od");
            cmd.arg("/Zi");
        }
    } else {
        // GCC/Clang flags
        cmd.arg("-std=c++17");
        cmd.arg("-fPIC");
        if !config.debug {
            cmd.arg("-O3");
            cmd.arg("-DNDEBUG");
        } else {
            cmd.arg("-g");
            cmd.arg("-O0");
        }
    }

    // Add CUDA specific flags
    if config.with_cuda && compiler.contains("nvcc") {
        for arch in &config.cuda_archs {
            cmd.arg(format!(
                "-gencode=arch=compute_{},code={}",
                &arch[3..],
                arch
            ));
        }
        cmd.arg("-x").arg("cu");
    }

    // Add extra compile args
    for arg in &config.extra_compile_args {
        cmd.arg(arg);
    }

    // Add source files (original + generated)
    for source in &config.sources {
        cmd.arg(source);
    }
    for source in generated_sources {
        cmd.arg(source);
    }

    // Output file
    let platform_suffix = target_platform
        .map(|p| format!("_{:?}", p))
        .unwrap_or_default();
    let output_file = config
        .build_dir
        .join(format!("lib{}{}.so", config.name, platform_suffix));

    if compiler.contains("cl.exe") {
        cmd.arg("/Fe:").arg(&output_file);
        cmd.arg("/LD"); // Create DLL
    } else {
        cmd.arg("-shared");
        cmd.arg("-o").arg(&output_file);
    }

    // Add library directories
    for lib_dir in &config.library_dirs {
        if compiler.contains("cl.exe") {
            cmd.arg(format!("/LIBPATH:{}", lib_dir.display()));
        } else {
            cmd.arg(format!("-L{}", lib_dir.display()));
        }
    }

    // Add libraries
    for lib in &config.libraries {
        if compiler.contains("cl.exe") {
            cmd.arg(format!("{}.lib", lib));
        } else {
            cmd.arg(format!("-l{}", lib));
        }
    }

    // Add extra link args
    for arg in &config.extra_link_args {
        cmd.arg(arg);
    }

    // Execute build
    let output = cmd
        .output()
        .map_err(|e| format!("Failed to execute compiler {}: {}", compiler, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Compilation failed for platform {:?}:\n{}",
            target_platform, stderr
        ));
    }

    Ok(output_file)
}

/// Detect current platform
fn detect_current_platform() -> TargetPlatform {
    match env::consts::OS {
        "windows" => match env::consts::ARCH {
            "x86_64" => TargetPlatform::WindowsX64,
            "x86" => TargetPlatform::WindowsX86,
            _ => TargetPlatform::WindowsX64, // Default
        },
        "macos" => match env::consts::ARCH {
            "aarch64" => TargetPlatform::MacOsArm64,
            _ => TargetPlatform::MacOsX64,
        },
        "linux" => match env::consts::ARCH {
            "aarch64" => TargetPlatform::LinuxAarch64,
            "arm64" => TargetPlatform::LinuxArm64,
            _ => TargetPlatform::LinuxX64,
        },
        _ => TargetPlatform::LinuxX64, // Default fallback
    }
}

/// Get Windows target string
fn get_windows_target(platform: &TargetPlatform) -> String {
    match platform {
        TargetPlatform::WindowsX64 => "x86_64-pc-windows-msvc".to_string(),
        TargetPlatform::WindowsX86 => "i686-pc-windows-msvc".to_string(),
        _ => "x86_64-pc-windows-msvc".to_string(), // Default
    }
}

/// Load a C++ extension from a shared library
pub fn load_cpp_extension(library_path: &Path) -> Result<(), String> {
    // This would typically use libloading or similar to dynamically load the library
    // For now, we just verify the file exists
    if !library_path.exists() {
        return Err(format!("Library not found: {}", library_path.display()));
    }

    // In a real implementation, we would:
    // 1. Load the shared library
    // 2. Register any custom operators
    // 3. Initialize any global state

    Ok(())
}

/// Generate a simple C++ extension template
pub fn generate_extension_template(name: &str, output_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Generate header file
    let header_content = format!(
        r#"#pragma once

#include <torsh/tensor.h>
#include <torsh/module.h>

namespace torsh {{
namespace ops {{

// Example custom operation
Tensor {}_forward(const Tensor& input);

}} // namespace ops
}} // namespace torsh
"#,
        name
    );

    let header_path = output_dir.join(format!("{}.h", name));
    fs::write(&header_path, header_content)
        .map_err(|e| format!("Failed to write header file: {}", e))?;

    // Generate source file
    let source_content = format!(
        r#"#include "{}.h"
#include <torsh/autograd.h>
#include <iostream>

namespace torsh {{
namespace ops {{

Tensor {}_forward(const Tensor& input) {{
    // Example implementation
    auto output = input.clone();
    
    // Perform custom operation
    // This is where you would implement your custom logic
    
    return output;
}}

// Register the operation
TORSH_LIBRARY(TORCH_EXTENSION_NAME, m) {{
    m.def("{}_forward", &{}_forward);
}}

}} // namespace ops
}} // namespace torsh
"#,
        name, name, name, name
    );

    let source_path = output_dir.join(format!("{}.cpp", name));
    fs::write(&source_path, source_content)
        .map_err(|e| format!("Failed to write source file: {}", e))?;

    // Generate setup script
    let setup_content = format!(
        r#"use torsh_utils::cpp_extension::{{CppExtensionConfig, build_cpp_extension}};
use std::path::PathBuf;

fn main() {{
    let config = CppExtensionConfig::new("{}", vec![
        PathBuf::from("{}.cpp"),
    ])
    .include_dir(".")
    .extra_compile_arg("-Wall")
    .extra_compile_arg("-Wextra");

    match build_cpp_extension(&config) {{
        Ok(result) => {{
            println!("Extension built successfully!");
            println!("Library: {{:?}}", result.library_path);
        }}
        Err(e) => {{
            eprintln!("Build failed: {{}}", e);
            std::process::exit(1);
        }}
    }}
}}
"#,
        name, name
    );

    let setup_path = output_dir.join("build.rs");
    fs::write(&setup_path, setup_content)
        .map_err(|e| format!("Failed to write setup script: {}", e))?;

    Ok(())
}

/// Check if CUDA is available for building extensions
pub fn cuda_is_available() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Get CUDA architectures available on the system
pub fn get_cuda_arch_list() -> Vec<String> {
    // This would ideally query the actual GPUs on the system
    // For now, return common architectures
    vec![
        "sm_70".to_string(), // V100
        "sm_75".to_string(), // T4, RTX 20xx
        "sm_80".to_string(), // A100
        "sm_86".to_string(), // RTX 30xx
        "sm_89".to_string(), // RTX 40xx
    ]
}

/// Query CUDA devices on the system
fn query_cuda_devices() -> Result<Vec<CudaDeviceInfo>, String> {
    // In a real implementation, this would use CUDA Driver API to query devices
    // For now, return mock data based on CUDA availability
    if !cuda_is_available() {
        return Err("CUDA is not available on this system".to_string());
    }

    // Mock device information (in real implementation, would query actual devices)
    let mock_device = CudaDeviceInfo {
        device_id: 0,
        name: "NVIDIA GPU".to_string(),
        compute_capability: "8.0".to_string(),
        total_memory: 8 * 1024 * 1024 * 1024, // 8GB
        max_threads_per_block: 1024,
        max_grid_size: [65535, 65535, 65535],
        max_block_size: [1024, 1024, 64],
        warp_size: 32,
        multiprocessor_count: 80,
        shared_memory_per_block: 48 * 1024, // 48KB
    };

    Ok(vec![mock_device])
}

/// Configure CUDA JIT compilation options
fn configure_cuda_jit_options(config: &CppExtensionConfig) -> Result<(), String> {
    // In a real implementation, this would configure CUDA driver JIT options
    // Such as:
    // - cuLinkCreate with JIT options
    // - Setting optimization level
    // - Configuring cache behavior
    // - Setting debug/profiling options

    if config.jit_config.cuda_jit {
        // Validate JIT configuration
        if config.jit_config.cuda_cache_size == 0 {
            return Err("CUDA JIT cache size must be greater than 0".to_string());
        }

        if config.jit_config.optimization_level > 3 {
            return Err("CUDA JIT optimization level must be 0-3".to_string());
        }

        // Configure JIT options based on config
        // This is where we would set:
        // - CU_JIT_OPTIMIZATION_LEVEL
        // - CU_JIT_CACHE_MODE
        // - CU_JIT_MAX_REGISTERS
        // - CU_JIT_THREADS_PER_BLOCK
    }

    Ok(())
}

/// Validate CUDA kernel syntax
fn validate_cuda_kernel_syntax(cuda_source: &str, op_name: &str) -> Result<(), String> {
    // Basic syntax validation for CUDA kernel source
    let required_patterns = [
        "__global__", // Kernel function marker
        "__device__", // Or device function marker
        "__host__",   // Or host function marker
    ];

    // Check if at least one CUDA pattern is present
    let has_cuda_pattern = required_patterns
        .iter()
        .any(|pattern| cuda_source.contains(pattern));

    if !has_cuda_pattern {
        return Err(format!(
            "CUDA source for operation '{}' does not contain valid CUDA kernel markers (__global__, __device__, or __host__)",
            op_name
        ));
    }

    // Check for common syntax errors
    let brackets_open = cuda_source.chars().filter(|&c| c == '{').count();
    let brackets_close = cuda_source.chars().filter(|&c| c == '}').count();

    if brackets_open != brackets_close {
        return Err(format!(
            "CUDA source for operation '{}' has mismatched braces ({{ and }})",
            op_name
        ));
    }

    // Check for semicolon at end of statements (basic check)
    let lines: Vec<&str> = cuda_source.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if !trimmed.is_empty()
            && !trimmed.starts_with("//")
            && !trimmed.starts_with("/*")
            && !trimmed.ends_with('{')
            && !trimmed.ends_with('}')
            && !trimmed.ends_with(';')
            && !trimmed.starts_with('#')
        {
            return Err(format!(
                "CUDA source for operation '{}' line {} may be missing semicolon: '{}'",
                op_name,
                i + 1,
                trimmed
            ));
        }
    }

    Ok(())
}

/// Compile CUDA kernel at runtime
pub fn compile_cuda_kernel_runtime(
    kernel_source: &str,
    kernel_name: &str,
    options: &CudaKernelCompilationOptions,
) -> Result<RuntimeCudaKernel, String> {
    // Validate CUDA availability
    if !cuda_is_available() {
        return Err("CUDA is not available for runtime compilation".to_string());
    }

    // Validate kernel source
    validate_cuda_kernel_syntax(kernel_source, kernel_name)?;

    // In a real implementation, this would:
    // 1. Use CUDA Driver API to compile PTX from source
    // 2. Load the compiled module
    // 3. Get kernel function handle
    // 4. Configure launch parameters

    // Generate PTX source (mock)
    let ptx_source = format!(
        r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry {}(
    .param .u64 param_0
)
{{
    // Generated PTX code would go here
    ret;
}}
"#,
        kernel_name
    );

    // Mock launch configuration
    let launch_config = CudaLaunchConfig {
        grid_size: [1, 1, 1],
        block_size: [256, 1, 1],
        shared_memory_size: 0,
        stream: None,
    };

    Ok(RuntimeCudaKernel {
        name: kernel_name.to_string(),
        ptx_source,
        module_handle: Some(1),   // Mock handle
        function_handle: Some(1), // Mock handle
        compilation_options: options.clone(),
        launch_config,
    })
}

/// Launch a runtime-compiled CUDA kernel
pub fn launch_cuda_kernel(
    kernel: &RuntimeCudaKernel,
    args: &[*mut std::ffi::c_void],
) -> Result<(), String> {
    // In a real implementation, this would:
    // 1. Validate kernel is loaded
    // 2. Set kernel parameters
    // 3. Launch kernel with configured grid/block dimensions
    // 4. Handle synchronization if needed

    if kernel.module_handle.is_none() || kernel.function_handle.is_none() {
        return Err(format!("Kernel '{}' is not properly loaded", kernel.name));
    }

    // Validate launch configuration
    if kernel.launch_config.grid_size[0] == 0 || kernel.launch_config.block_size[0] == 0 {
        return Err(format!(
            "Invalid launch configuration for kernel '{}'",
            kernel.name
        ));
    }

    // Mock kernel launch validation
    println!(
        "Launching CUDA kernel '{}' with grid {:?} and block {:?}",
        kernel.name, kernel.launch_config.grid_size, kernel.launch_config.block_size
    );

    // Validate argument count (basic check)
    if args.is_empty() {
        return Err(format!(
            "No arguments provided for kernel '{}'",
            kernel.name
        ));
    }

    Ok(())
}

/// Auto-tune CUDA kernel launch parameters
pub fn auto_tune_cuda_kernel(
    kernel: &mut RuntimeCudaKernel,
    input_sizes: &[usize],
) -> Result<CudaLaunchConfig, String> {
    // Query device properties for optimal configuration
    let devices = query_cuda_devices()?;
    let device = devices
        .first()
        .ok_or("No CUDA devices available for auto-tuning")?;

    // Calculate optimal block size based on kernel complexity and device properties
    let optimal_block_size = if input_sizes.iter().any(|&size| size > 10000) {
        // Large inputs: use larger blocks for better memory coalescing
        device.max_threads_per_block.min(512)
    } else {
        // Small inputs: use smaller blocks to avoid warp underutilization
        device.max_threads_per_block.min(256)
    };

    // Calculate grid size based on input size and block size
    let total_elements = input_sizes.iter().max().copied().unwrap_or(1);
    let optimal_grid_size =
        (total_elements + optimal_block_size as usize - 1) / optimal_block_size as usize;

    // Limit grid size to device maximum
    let clamped_grid_size = (optimal_grid_size as u32).min(device.max_grid_size[0]);

    let optimized_config = CudaLaunchConfig {
        grid_size: [clamped_grid_size, 1, 1],
        block_size: [optimal_block_size, 1, 1],
        shared_memory_size: 0, // Auto-tune shared memory based on kernel requirements
        stream: kernel.launch_config.stream,
    };

    // Update kernel configuration
    kernel.launch_config = optimized_config.clone();

    Ok(optimized_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_cpp_extension_config() {
        let config = CppExtensionConfig::new("test_ext", vec![PathBuf::from("test.cpp")])
            .include_dir("/usr/include")
            .library("torsh")
            .extra_compile_arg("-std=c++17");

        assert_eq!(config.name, "test_ext");
        assert_eq!(config.sources.len(), 1);
        assert_eq!(config.include_dirs.len(), 1);
        assert_eq!(config.libraries.len(), 1);
    }

    #[test]
    fn test_generate_template() {
        let temp_dir = env::temp_dir().join("torsh_test_template");
        let result = generate_extension_template("test_op", &temp_dir);

        assert!(result.is_ok());
        assert!(temp_dir.join("test_op.h").exists());
        assert!(temp_dir.join("test_op.cpp").exists());
        assert!(temp_dir.join("build.rs").exists());

        // Cleanup
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cuda_detection() {
        // This test might fail on systems without CUDA
        let available = cuda_is_available();
        println!("CUDA available: {}", available);

        if available {
            let archs = get_cuda_arch_list();
            assert!(!archs.is_empty());
        }
    }

    #[test]
    fn test_cuda_kernel_compilation_options() {
        let default_options = CudaKernelCompilationOptions::default();
        assert_eq!(default_options.optimization_level, 2);
        assert!(!default_options.fast_math);
        assert!(default_options.use_cache);
        assert!(!default_options.debug_info);

        let custom_options = CudaKernelCompilationOptions {
            optimization_level: 3,
            fast_math: true,
            max_registers: Some(64),
            debug_info: true,
            target_arch: Some("sm_80".to_string()),
            ..Default::default()
        };

        assert_eq!(custom_options.optimization_level, 3);
        assert!(custom_options.fast_math);
        assert_eq!(custom_options.max_registers, Some(64));
        assert!(custom_options.debug_info);
    }

    #[test]
    fn test_cuda_kernel_syntax_validation() {
        // Valid CUDA kernel
        let valid_kernel = r#"
        __global__ void test_kernel(float* input, float* output) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            output[idx] = input[idx] * 2.0f;
        }
        "#;

        assert!(validate_cuda_kernel_syntax(valid_kernel, "test_kernel").is_ok());

        // Invalid kernel (missing __global__)
        let invalid_kernel = r#"
        void test_kernel(float* input, float* output) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            output[idx] = input[idx] * 2.0f;
        }
        "#;

        assert!(validate_cuda_kernel_syntax(invalid_kernel, "test_kernel").is_err());

        // Invalid kernel (mismatched braces)
        let invalid_braces = r#"
        __global__ void test_kernel(float* input, float* output) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            output[idx] = input[idx] * 2.0f;
        // Missing closing brace
        "#;

        assert!(validate_cuda_kernel_syntax(invalid_braces, "test_kernel").is_err());
    }

    #[test]
    fn test_runtime_cuda_kernel_compilation() {
        let kernel_source = r#"
        __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        "#;

        let options = CudaKernelCompilationOptions::default();

        // This should work even without CUDA (returns mock result)
        if cuda_is_available() {
            let result = compile_cuda_kernel_runtime(kernel_source, "vector_add", &options);
            if let Ok(kernel) = result {
                assert_eq!(kernel.name, "vector_add");
                assert!(!kernel.ptx_source.is_empty());
                assert!(kernel.module_handle.is_some());
                assert!(kernel.function_handle.is_some());
            }
        }
    }

    #[test]
    fn test_cuda_launch_config_auto_tuning() {
        if cuda_is_available() {
            let kernel_source = r#"
            __global__ void simple_kernel(float* data) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                data[idx] *= 2.0f;
            }
            "#;

            let options = CudaKernelCompilationOptions::default();
            let result = compile_cuda_kernel_runtime(kernel_source, "simple_kernel", &options);

            if let Ok(mut kernel) = result {
                let input_sizes = vec![1024, 2048, 4096];
                let tuned_config = auto_tune_cuda_kernel(&mut kernel, &input_sizes);

                if let Ok(config) = tuned_config {
                    assert!(config.grid_size[0] > 0);
                    assert!(config.block_size[0] > 0);
                    assert!(config.block_size[0] <= 1024); // Max threads per block
                }
            }
        }
    }

    #[test]
    fn test_custom_op_with_cuda_jit() {
        let custom_op = CustomOpDefinition {
            name: "custom_relu".to_string(),
            op_type: CustomOpType::Forward,
            input_shapes: vec![None], // Dynamic shape
            output_shapes: vec![None],
            cpu_source: Some("return torch::relu(inputs[0]);".to_string()),
            cuda_source: Some(
                r#"
            __global__ void relu_kernel(float* input, float* output, int size) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size) {
                    output[idx] = fmaxf(0.0f, input[idx]);
                }
            }
            "#
                .to_string(),
            ),
            compile_flags: vec!["-O3".to_string()],
            schema: OpSchema {
                input_types: vec![TensorType {
                    dtype: "float32".to_string(),
                    min_dims: 1,
                    max_dims: None,
                    supports_sparse: false,
                }],
                output_types: vec![TensorType {
                    dtype: "float32".to_string(),
                    min_dims: 1,
                    max_dims: None,
                    supports_sparse: false,
                }],
                is_elementwise: true,
                is_deterministic: true,
                memory_requirement: MemoryRequirement::Linear,
            },
        };

        let config = CppExtensionConfig::new("custom_relu_ext", vec![])
            .enable_cuda_jit()
            .custom_op(custom_op);

        assert!(config.jit_config.cuda_jit);
        assert_eq!(config.custom_ops.len(), 1);
        assert_eq!(config.custom_ops[0].name, "custom_relu");
    }
}
