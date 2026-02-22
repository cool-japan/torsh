// Copyright (c) 2025 ToRSh Contributors
//
// WebGPU Compute Shader Integration
//
// This module provides abstractions for WebGPU compute shaders, enabling
// high-performance tensor operations in web browsers and native applications.
//
// # Key Features
//
// - **WGSL Shader Compilation**: Compile and manage WebGPU Shading Language shaders
// - **Compute Pipeline Management**: Efficient pipeline creation and caching
// - **Buffer Management**: Optimized GPU buffer allocation and transfer
// - **Workgroup Optimization**: Automatic workgroup size calculation
// - **Cross-Platform**: Works in browsers (via wasm) and native applications
//
// # Design Principles
//
// 1. **Zero-Copy Transfer**: Minimize data transfer between CPU and GPU
// 2. **Pipeline Caching**: Reuse compiled pipelines for performance
// 3. **Memory Efficiency**: Efficient buffer pooling and management
// 4. **Shader Composition**: Modular shader building blocks
//
// # Examples
//
// ```rust
// use torsh_core::webgpu::{WGSLShader, ComputePipeline, GPUBuffer};
//
// // Define a WGSL compute shader
// let shader = WGSLShader::new(r#"
//     @compute @workgroup_size(64)
//     fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//         // Compute shader code
//     }
// "#);
//
// // Create a compute pipeline
// let pipeline = ComputePipeline::new(shader);
//
// // Allocate GPU buffers
// let input_buffer = GPUBuffer::storage(data.len());
// let output_buffer = GPUBuffer::storage(data.len());
// ```

use core::fmt;

/// WebGPU compute shader in WGSL (WebGPU Shading Language)
///
/// This struct represents a compute shader written in WGSL that can be
/// compiled and executed on the GPU.
#[derive(Debug, Clone)]
pub struct WGSLShader {
    /// Shader source code in WGSL
    source: String,
    /// Entry point function name
    entry_point: String,
    /// Workgroup size (x, y, z)
    workgroup_size: (u32, u32, u32),
}

impl WGSLShader {
    /// Create a new WGSL shader
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            entry_point: "main".to_string(),
            workgroup_size: (64, 1, 1), // Default workgroup size
        }
    }

    /// Create a shader with custom entry point
    pub fn with_entry_point(source: impl Into<String>, entry_point: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            entry_point: entry_point.into(),
            workgroup_size: (64, 1, 1),
        }
    }

    /// Set workgroup size
    pub fn with_workgroup_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroup_size = (x, y, z);
        self
    }

    /// Get shader source
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get entry point
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Get workgroup size
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        self.workgroup_size
    }

    /// Validate shader syntax (basic validation)
    pub fn validate(&self) -> Result<(), ShaderError> {
        if self.source.is_empty() {
            return Err(ShaderError::EmptyShader);
        }
        if !self.source.contains("@compute") {
            return Err(ShaderError::MissingComputeAttribute);
        }
        Ok(())
    }
}

/// Shader compilation and validation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShaderError {
    /// Shader source is empty
    EmptyShader,
    /// Missing @compute attribute
    MissingComputeAttribute,
    /// Syntax error in shader
    SyntaxError(String),
    /// Compilation failed
    CompilationFailed(String),
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderError::EmptyShader => write!(f, "Shader source is empty"),
            ShaderError::MissingComputeAttribute => write!(f, "Missing @compute attribute"),
            ShaderError::SyntaxError(msg) => write!(f, "Syntax error: {}", msg),
            ShaderError::CompilationFailed(msg) => write!(f, "Compilation failed: {}", msg),
        }
    }
}

/// GPU buffer types for WebGPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    /// Storage buffer (read/write)
    Storage,
    /// Uniform buffer (read-only, small data)
    Uniform,
    /// Staging buffer (CPU-GPU transfer)
    Staging,
    /// Vertex buffer
    Vertex,
    /// Index buffer
    Index,
}

/// GPU buffer descriptor
///
/// Describes a GPU buffer for allocation and management.
#[derive(Debug, Clone)]
pub struct GPUBuffer {
    /// Buffer size in bytes
    size: usize,
    /// Buffer usage
    usage: BufferUsage,
    /// Whether buffer is mapped for CPU access
    mapped: bool,
    /// Buffer label for debugging
    label: Option<String>,
}

impl GPUBuffer {
    /// Create a storage buffer
    pub fn storage(size: usize) -> Self {
        Self {
            size,
            usage: BufferUsage::Storage,
            mapped: false,
            label: None,
        }
    }

    /// Create a uniform buffer
    pub fn uniform(size: usize) -> Self {
        Self {
            size,
            usage: BufferUsage::Uniform,
            mapped: false,
            label: None,
        }
    }

    /// Create a staging buffer
    pub fn staging(size: usize) -> Self {
        Self {
            size,
            usage: BufferUsage::Staging,
            mapped: true,
            label: None,
        }
    }

    /// Set buffer label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get buffer usage
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Check if buffer is mapped
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Get buffer label
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

/// Compute pipeline for executing shaders
///
/// Represents a compiled compute pipeline that can be executed on the GPU.
#[derive(Debug, Clone)]
pub struct ComputePipeline {
    /// Associated shader
    shader: WGSLShader,
    /// Bind group layouts
    bind_groups: Vec<BindGroupLayout>,
    /// Pipeline label
    label: Option<String>,
}

impl ComputePipeline {
    /// Create a new compute pipeline
    pub fn new(shader: WGSLShader) -> Self {
        Self {
            shader,
            bind_groups: Vec::new(),
            label: None,
        }
    }

    /// Add a bind group layout
    pub fn with_bind_group(mut self, layout: BindGroupLayout) -> Self {
        self.bind_groups.push(layout);
        self
    }

    /// Set pipeline label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get shader
    pub fn shader(&self) -> &WGSLShader {
        &self.shader
    }

    /// Get bind groups
    pub fn bind_groups(&self) -> &[BindGroupLayout] {
        &self.bind_groups
    }

    /// Get pipeline label
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Calculate optimal dispatch size for a given data size
    pub fn optimal_dispatch_size(&self, data_size: usize) -> (u32, u32, u32) {
        let (wg_x, wg_y, wg_z) = self.shader.workgroup_size();
        let workgroup_count = (data_size as u32 + wg_x - 1) / wg_x;
        (workgroup_count, wg_y, wg_z)
    }
}

/// Bind group layout entry
///
/// Describes a binding in a bind group (buffer, texture, sampler, etc.)
#[derive(Debug, Clone)]
pub struct BindGroupEntry {
    /// Binding index
    binding: u32,
    /// Resource type
    resource_type: ResourceType,
    /// Shader visibility
    visibility: ShaderStage,
}

impl BindGroupEntry {
    /// Create a new bind group entry
    pub fn new(binding: u32, resource_type: ResourceType, visibility: ShaderStage) -> Self {
        Self {
            binding,
            resource_type,
            visibility,
        }
    }

    /// Get binding index
    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Get resource type
    pub fn resource_type(&self) -> ResourceType {
        self.resource_type
    }

    /// Get visibility
    pub fn visibility(&self) -> ShaderStage {
        self.visibility
    }
}

/// Bind group layout
///
/// Describes the layout of bindings in a bind group.
#[derive(Debug, Clone)]
pub struct BindGroupLayout {
    /// Entries in this bind group
    entries: Vec<BindGroupEntry>,
    /// Layout label
    label: Option<String>,
}

impl BindGroupLayout {
    /// Create a new bind group layout
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            label: None,
        }
    }

    /// Add an entry
    pub fn with_entry(mut self, entry: BindGroupEntry) -> Self {
        self.entries.push(entry);
        self
    }

    /// Set label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get entries
    pub fn entries(&self) -> &[BindGroupEntry] {
        &self.entries
    }

    /// Get label
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl Default for BindGroupLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource types for bind group entries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    /// Storage buffer (read/write)
    StorageBuffer,
    /// Uniform buffer (read-only)
    UniformBuffer,
    /// Read-only storage buffer
    ReadOnlyStorageBuffer,
    /// Texture (2D/3D)
    Texture,
    /// Sampler
    Sampler,
}

/// Shader stage visibility
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    /// Vertex shader
    Vertex,
    /// Fragment shader
    Fragment,
    /// Compute shader
    Compute,
    /// All stages
    All,
}

/// Workgroup size optimizer
///
/// Automatically calculates optimal workgroup sizes based on data dimensions
/// and GPU capabilities.
#[derive(Debug, Clone, Copy)]
pub struct WorkgroupOptimizer {
    /// Maximum workgroup size (device-dependent)
    max_workgroup_size: u32,
    /// Preferred workgroup size for 1D operations
    preferred_1d: u32,
    /// Preferred workgroup size for 2D operations
    preferred_2d: (u32, u32),
}

impl WorkgroupOptimizer {
    /// Create a new workgroup optimizer
    pub fn new(max_workgroup_size: u32) -> Self {
        Self {
            max_workgroup_size,
            preferred_1d: 256,
            preferred_2d: (16, 16),
        }
    }

    /// Calculate optimal workgroup size for 1D data
    pub fn optimize_1d(&self, data_size: usize) -> u32 {
        let size = data_size as u32;
        if size <= 64 {
            64
        } else if size <= self.preferred_1d {
            self.preferred_1d
        } else {
            self.max_workgroup_size.min(512)
        }
    }

    /// Calculate optimal workgroup size for 2D data
    pub fn optimize_2d(&self, width: usize, height: usize) -> (u32, u32) {
        let (w, h) = (width as u32, height as u32);
        if w <= 16 && h <= 16 {
            (8, 8)
        } else if w <= 32 && h <= 32 {
            (16, 16)
        } else {
            self.preferred_2d
        }
    }

    /// Calculate optimal workgroup size for 3D data
    pub fn optimize_3d(&self, width: usize, height: usize, depth: usize) -> (u32, u32, u32) {
        let (w, h, d) = (width as u32, height as u32, depth as u32);
        if w <= 8 && h <= 8 && d <= 8 {
            (4, 4, 4)
        } else {
            (8, 8, 4)
        }
    }

    /// Get maximum workgroup size
    pub fn max_workgroup_size(&self) -> u32 {
        self.max_workgroup_size
    }
}

impl Default for WorkgroupOptimizer {
    fn default() -> Self {
        Self::new(256) // Common default
    }
}

/// Pipeline cache for reusing compiled pipelines
///
/// Caches compiled compute pipelines to avoid recompilation.
#[derive(Debug, Clone)]
pub struct PipelineCache {
    /// Cached pipelines (shader source hash -> pipeline)
    cache: Vec<(u64, ComputePipeline)>,
    /// Maximum cache size
    max_size: usize,
}

impl PipelineCache {
    /// Create a new pipeline cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Vec::new(),
            max_size,
        }
    }

    /// Get or create a pipeline
    pub fn get_or_create<F>(&mut self, shader: &WGSLShader, create_fn: F) -> &ComputePipeline
    where
        F: FnOnce(&WGSLShader) -> ComputePipeline,
    {
        let hash = self.hash_shader(shader);

        // Check if already cached
        if let Some(index) = self.cache.iter().position(|(h, _)| *h == hash) {
            return &self.cache[index].1;
        }

        // Create new pipeline
        let pipeline = create_fn(shader);
        self.cache.push((hash, pipeline));

        // Evict oldest if cache is full
        if self.cache.len() > self.max_size {
            self.cache.remove(0);
        }

        &self
            .cache
            .last()
            .expect("cache should have at least one entry after push")
            .1
    }

    /// Simple hash function for shader source
    fn hash_shader(&self, shader: &WGSLShader) -> u64 {
        // Simple hash based on source length and first few characters
        let src = shader.source();
        let mut hash = src.len() as u64;
        for (i, byte) in src.bytes().take(16).enumerate() {
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(byte as u64 * (i as u64 + 1));
        }
        hash
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache size
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Get maximum cache size
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

/// Common WGSL shader templates
pub mod shaders {
    use super::*;

    /// Element-wise addition shader
    pub fn elementwise_add() -> WGSLShader {
        WGSLShader::new(
            r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&input_a)) {
        output[index] = input_a[index] + input_b[index];
    }
}
"#,
        )
    }

    /// Element-wise multiplication shader
    pub fn elementwise_mul() -> WGSLShader {
        WGSLShader::new(
            r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index < arrayLength(&input_a)) {
        output[index] = input_a[index] * input_b[index];
    }
}
"#,
        )
    }

    /// Matrix multiplication shader (simple version)
    pub fn matrix_mul() -> WGSLShader {
        WGSLShader::new(
            r#"
struct Dimensions {
    m: u32,
    n: u32,
    k: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= dims.m || col >= dims.n) {
        return;
    }

    var sum = 0.0;
    for (var i = 0u; i < dims.k; i++) {
        let a_index = row * dims.k + i;
        let b_index = i * dims.n + col;
        sum += matrix_a[a_index] * matrix_b[b_index];
    }

    let out_index = row * dims.n + col;
    output[out_index] = sum;
}
"#,
        )
        .with_workgroup_size(16, 16, 1)
    }

    /// Reduction (sum) shader
    pub fn reduce_sum() -> WGSLShader {
        WGSLShader::new(
            r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let index = global_id.x;

    // Load data into shared memory
    if (index < arrayLength(&input)) {
        shared_data[tid] = input[index];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Reduce in shared memory
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        workgroupBarrier();
    }

    // Write result
    if (tid == 0u) {
        output[global_id.x / 256u] = shared_data[0];
    }
}
"#,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgsl_shader_creation() {
        let shader = WGSLShader::new("@compute @workgroup_size(64) fn main() {}");
        assert!(shader.source().contains("@compute"));
        assert_eq!(shader.entry_point(), "main");
        assert_eq!(shader.workgroup_size(), (64, 1, 1));
    }

    #[test]
    fn test_shader_with_entry_point() {
        let shader = WGSLShader::with_entry_point("code", "custom_entry");
        assert_eq!(shader.entry_point(), "custom_entry");
    }

    #[test]
    fn test_shader_workgroup_size() {
        let shader = WGSLShader::new("code").with_workgroup_size(16, 16, 1);
        assert_eq!(shader.workgroup_size(), (16, 16, 1));
    }

    #[test]
    fn test_shader_validation() {
        let shader = WGSLShader::new("@compute fn main() {}");
        assert!(shader.validate().is_ok());

        let empty_shader = WGSLShader::new("");
        assert_eq!(empty_shader.validate(), Err(ShaderError::EmptyShader));

        let invalid_shader = WGSLShader::new("fn main() {}");
        assert_eq!(
            invalid_shader.validate(),
            Err(ShaderError::MissingComputeAttribute)
        );
    }

    #[test]
    fn test_gpu_buffer_creation() {
        let storage = GPUBuffer::storage(1024);
        assert_eq!(storage.size(), 1024);
        assert_eq!(storage.usage(), BufferUsage::Storage);
        assert!(!storage.is_mapped());

        let uniform = GPUBuffer::uniform(256);
        assert_eq!(uniform.usage(), BufferUsage::Uniform);

        let staging = GPUBuffer::staging(512);
        assert_eq!(staging.usage(), BufferUsage::Staging);
        assert!(staging.is_mapped());
    }

    #[test]
    fn test_buffer_with_label() {
        let buffer = GPUBuffer::storage(1024).with_label("test_buffer");
        assert_eq!(buffer.label(), Some("test_buffer"));
    }

    #[test]
    fn test_compute_pipeline() {
        let shader = WGSLShader::new("@compute fn main() {}");
        let pipeline = ComputePipeline::new(shader.clone());
        assert_eq!(pipeline.shader().source(), shader.source());
        assert_eq!(pipeline.bind_groups().len(), 0);
    }

    #[test]
    fn test_pipeline_with_label() {
        let shader = WGSLShader::new("@compute fn main() {}");
        let pipeline = ComputePipeline::new(shader).with_label("test_pipeline");
        assert_eq!(pipeline.label(), Some("test_pipeline"));
    }

    #[test]
    fn test_optimal_dispatch_size() {
        let shader = WGSLShader::new("code").with_workgroup_size(64, 1, 1);
        let pipeline = ComputePipeline::new(shader);

        let (x, _, _) = pipeline.optimal_dispatch_size(1000);
        assert_eq!(x, 16); // ceil(1000 / 64) = 16
    }

    #[test]
    fn test_bind_group_entry() {
        let entry = BindGroupEntry::new(0, ResourceType::StorageBuffer, ShaderStage::Compute);
        assert_eq!(entry.binding(), 0);
        assert_eq!(entry.resource_type(), ResourceType::StorageBuffer);
        assert_eq!(entry.visibility(), ShaderStage::Compute);
    }

    #[test]
    fn test_bind_group_layout() {
        let entry = BindGroupEntry::new(0, ResourceType::StorageBuffer, ShaderStage::Compute);
        let layout = BindGroupLayout::new().with_entry(entry);
        assert_eq!(layout.entries().len(), 1);
    }

    #[test]
    fn test_workgroup_optimizer() {
        let optimizer = WorkgroupOptimizer::new(512);
        assert_eq!(optimizer.optimize_1d(50), 64); // size <= 64 -> 64
        assert_eq!(optimizer.optimize_1d(100), 256); // size <= 256 -> 256
        assert_eq!(optimizer.optimize_1d(500), 512); // size > 256 -> min(max, 512)

        let (w, h) = optimizer.optimize_2d(10, 10);
        assert_eq!((w, h), (8, 8));

        let (w, h, d) = optimizer.optimize_3d(10, 10, 10);
        assert_eq!((w, h, d), (8, 8, 4));
    }

    #[test]
    fn test_default_workgroup_optimizer() {
        let optimizer = WorkgroupOptimizer::default();
        assert_eq!(optimizer.max_workgroup_size(), 256);
    }

    #[test]
    fn test_pipeline_cache() {
        let mut cache = PipelineCache::new(10);
        assert_eq!(cache.size(), 0);

        let shader = WGSLShader::new("@compute fn main() {}");
        let _pipeline = cache.get_or_create(&shader, |s| ComputePipeline::new(s.clone()));
        assert_eq!(cache.size(), 1);

        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_shader_templates() {
        let add_shader = shaders::elementwise_add();
        assert!(add_shader.source().contains("input_a"));
        assert!(add_shader.validate().is_ok());

        let mul_shader = shaders::elementwise_mul();
        assert!(mul_shader.source().contains("input_b"));
        assert!(mul_shader.validate().is_ok());

        let matmul_shader = shaders::matrix_mul();
        assert!(matmul_shader.source().contains("matrix_a"));
        assert_eq!(matmul_shader.workgroup_size(), (16, 16, 1));
        assert!(matmul_shader.validate().is_ok());

        let reduce_shader = shaders::reduce_sum();
        assert!(reduce_shader.source().contains("shared_data"));
        assert!(reduce_shader.validate().is_ok());
    }

    #[test]
    fn test_resource_types() {
        let _storage = ResourceType::StorageBuffer;
        let _uniform = ResourceType::UniformBuffer;
        let _readonly = ResourceType::ReadOnlyStorageBuffer;
        let _texture = ResourceType::Texture;
        let _sampler = ResourceType::Sampler;
    }

    #[test]
    fn test_shader_stages() {
        let _vertex = ShaderStage::Vertex;
        let _fragment = ShaderStage::Fragment;
        let _compute = ShaderStage::Compute;
        let _all = ShaderStage::All;
    }

    #[test]
    fn test_buffer_usage_types() {
        let _storage = BufferUsage::Storage;
        let _uniform = BufferUsage::Uniform;
        let _staging = BufferUsage::Staging;
        let _vertex = BufferUsage::Vertex;
        let _index = BufferUsage::Index;
    }

    #[test]
    fn test_shader_error_display() {
        let err = ShaderError::EmptyShader;
        assert_eq!(format!("{}", err), "Shader source is empty");

        let err = ShaderError::MissingComputeAttribute;
        assert_eq!(format!("{}", err), "Missing @compute attribute");

        let err = ShaderError::SyntaxError("test".to_string());
        assert_eq!(format!("{}", err), "Syntax error: test");

        let err = ShaderError::CompilationFailed("test".to_string());
        assert_eq!(format!("{}", err), "Compilation failed: test");
    }

    #[test]
    fn test_pipeline_with_bind_group() {
        let shader = WGSLShader::new("@compute fn main() {}");
        let layout = BindGroupLayout::new();
        let pipeline = ComputePipeline::new(shader).with_bind_group(layout);
        assert_eq!(pipeline.bind_groups().len(), 1);
    }

    #[test]
    fn test_bind_group_with_label() {
        let layout = BindGroupLayout::new().with_label("test_layout");
        assert_eq!(layout.label(), Some("test_layout"));
    }

    #[test]
    fn test_default_bind_group_layout() {
        let layout = BindGroupLayout::default();
        assert_eq!(layout.entries().len(), 0);
    }

    #[test]
    fn test_cache_max_size() {
        let cache = PipelineCache::new(5);
        assert_eq!(cache.max_size(), 5);
    }
}
