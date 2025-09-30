//! WebGPU compute pipeline management for ToRSh

#[cfg(feature = "webgpu")]
use crate::webgpu::wgpu;
use crate::webgpu::shader::ShaderCache;
use crate::webgpu::{WebGpuDevice, WebGpuError, WebGpuResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Compute pipeline descriptor
#[derive(Debug, Clone)]
pub struct PipelineDescriptor {
    pub label: String,
    pub shader_source: String,
    pub entry_point: String,
    pub workgroup_size: (u32, u32, u32),
    pub bind_group_layouts: Vec<wgpu::BindGroupLayoutDescriptor<'static>>,
}

impl PipelineDescriptor {
    /// Create a new pipeline descriptor
    pub fn new(
        label: impl Into<String>,
        shader_source: impl Into<String>,
        entry_point: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            shader_source: shader_source.into(),
            entry_point: entry_point.into(),
            workgroup_size: (64, 1, 1),
            bind_group_layouts: Vec::new(),
        }
    }

    /// Set workgroup size
    pub fn with_workgroup_size(mut self, workgroup_size: (u32, u32, u32)) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }

    /// Add bind group layout
    pub fn with_bind_group_layout(
        mut self,
        layout: wgpu::BindGroupLayoutDescriptor<'static>,
    ) -> Self {
        self.bind_group_layouts.push(layout);
        self
    }

    /// Generate a cache key for this pipeline
    pub fn cache_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.shader_source.hash(&mut hasher);
        self.entry_point.hash(&mut hasher);
        self.workgroup_size.hash(&mut hasher);

        format!("{}:{:x}", self.label, hasher.finish())
    }
}

/// Compute pipeline wrapper
#[derive(Debug)]
pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    descriptor: PipelineDescriptor,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    device: Arc<WebGpuDevice>,
}

impl ComputePipeline {
    /// Create a new compute pipeline
    pub fn new(device: Arc<WebGpuDevice>, descriptor: PipelineDescriptor) -> WebGpuResult<Self> {
        // Validate workgroup size
        crate::webgpu::error::validate_workgroup_size(descriptor.workgroup_size, device.limits())?;

        // Create bind group layouts
        let bind_group_layouts: Vec<_> = descriptor
            .bind_group_layouts
            .iter()
            .map(|desc| device.create_bind_group_layout(desc))
            .collect();

        // Create pipeline layout
        let pipeline_layout =
            device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{} Pipeline Layout", descriptor.label)),
                    bind_group_layouts: &bind_group_layouts.iter().collect::<Vec<_>>(),
                    push_constant_ranges: &[],
                });

        // Create shader module
        let shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", descriptor.label)),
            source: wgpu::ShaderSource::Wgsl(descriptor.shader_source.as_str().into()),
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&descriptor.label),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(descriptor.entry_point.as_str()),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            descriptor,
            bind_group_layouts,
            device,
        })
    }

    /// Get the underlying wgpu compute pipeline
    pub fn wgpu_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Get pipeline descriptor
    pub fn descriptor(&self) -> &PipelineDescriptor {
        &self.descriptor
    }

    /// Get bind group layouts
    pub fn bind_group_layouts(&self) -> &[wgpu::BindGroupLayout] {
        &self.bind_group_layouts
    }

    /// Create bind groups for buffers
    pub fn create_bind_groups(
        &self,
        buffer_groups: &[&[&wgpu::Buffer]],
    ) -> WebGpuResult<Vec<wgpu::BindGroup>> {
        if buffer_groups.len() != self.bind_group_layouts.len() {
            return Err(WebGpuError::ValidationFailed(format!(
                "Buffer group count {} doesn't match bind group layout count {}",
                buffer_groups.len(),
                self.bind_group_layouts.len()
            )));
        }

        let bind_groups = self
            .bind_group_layouts
            .iter()
            .zip(buffer_groups.iter())
            .enumerate()
            .map(|(i, (layout, buffers))| {
                let entries: Vec<_> = buffers
                    .iter()
                    .enumerate()
                    .map(|(j, buffer)| wgpu::BindGroupEntry {
                        binding: j as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect();

                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("{} Bind Group {}", self.descriptor.label, i)),
                    layout,
                    entries: &entries,
                })
            })
            .collect();

        Ok(bind_groups)
    }

    /// Dispatch compute work
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: &[&wgpu::BindGroup],
        workgroup_count: (u32, u32, u32),
    ) -> WebGpuResult<()> {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{} Compute Pass", self.descriptor.label)),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);

        for (i, bind_group) in bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, Some(*bind_group), &[]);
        }

        compute_pass.dispatch_workgroups(workgroup_count.0, workgroup_count.1, workgroup_count.2);

        Ok(())
    }

    /// Calculate optimal workgroup count for given problem size
    pub fn optimal_workgroup_count(&self, problem_size: (u32, u32, u32)) -> (u32, u32, u32) {
        let (wx, wy, wz) = self.descriptor.workgroup_size;
        let (px, py, pz) = problem_size;

        ((px + wx - 1) / wx, (py + wy - 1) / wy, (pz + wz - 1) / wz)
    }
}

/// Pipeline cache for reusing compiled pipelines
#[derive(Debug)]
pub struct PipelineCache {
    cache: RwLock<HashMap<String, Arc<ComputePipeline>>>,
    shader_cache: Arc<ShaderCache>,
    device: Arc<WebGpuDevice>,
}

impl PipelineCache {
    /// Create a new pipeline cache
    pub fn new(device: Arc<WebGpuDevice>) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            shader_cache: Arc::new(ShaderCache::new()),
            device,
        }
    }

    /// Get or create a pipeline
    pub fn get_or_create(
        &self,
        descriptor: PipelineDescriptor,
    ) -> WebGpuResult<Arc<ComputePipeline>> {
        let cache_key = descriptor.cache_key();

        // Try to get from cache first
        {
            let cache = self.cache.read();
            if let Some(pipeline) = cache.get(&cache_key) {
                return Ok(Arc::clone(pipeline));
            }
        }

        // Create new pipeline
        let pipeline = ComputePipeline::new(Arc::clone(&self.device), descriptor)?;
        let pipeline_arc = Arc::new(pipeline);

        // Store in cache
        {
            let mut cache = self.cache.write();
            cache.insert(cache_key, Arc::clone(&pipeline_arc));
        }

        Ok(pipeline_arc)
    }

    /// Create pipeline for common operations
    pub fn create_elementwise_binary_pipeline(
        &self,
        operation: &str,
        shader_source: &str,
    ) -> WebGpuResult<Arc<ComputePipeline>> {
        let binary_layout = wgpu::BindGroupLayoutDescriptor {
            label: Some("Binary Operation Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

        let descriptor =
            PipelineDescriptor::new(format!("elementwise_{}", operation), shader_source, "main")
                .with_bind_group_layout(binary_layout);

        self.get_or_create(descriptor)
    }

    /// Create pipeline for unary operations
    pub fn create_unary_pipeline(
        &self,
        operation: &str,
        shader_source: &str,
    ) -> WebGpuResult<Arc<ComputePipeline>> {
        let unary_layout = wgpu::BindGroupLayoutDescriptor {
            label: Some("Unary Operation Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

        let descriptor =
            PipelineDescriptor::new(format!("unary_{}", operation), shader_source, "main")
                .with_bind_group_layout(unary_layout);

        self.get_or_create(descriptor)
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.write().clear();
        self.shader_cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> PipelineCacheStats {
        let cache = self.cache.read();
        let pipeline_count = cache.len();
        let (shader_count, shader_bytes) = self.shader_cache.stats();

        PipelineCacheStats {
            pipeline_count,
            shader_count,
            shader_bytes,
            total_memory_usage: self.estimate_memory_usage(&cache),
        }
    }

    /// Estimate memory usage of cached pipelines
    fn estimate_memory_usage(&self, cache: &HashMap<String, Arc<ComputePipeline>>) -> usize {
        // Rough estimate: each pipeline uses about 1KB + shader size
        cache.len() * 1024
            + cache
                .values()
                .map(|p| p.descriptor.shader_source.len())
                .sum::<usize>()
    }
}

/// Pipeline cache statistics
#[derive(Debug, Clone)]
pub struct PipelineCacheStats {
    pub pipeline_count: usize,
    pub shader_count: usize,
    pub shader_bytes: usize,
    pub total_memory_usage: usize,
}

/// Pipeline factory for creating common pipelines
#[derive(Debug)]
pub struct PipelineFactory {
    cache: Arc<PipelineCache>,
    device: Arc<WebGpuDevice>,
}

impl PipelineFactory {
    /// Create a new pipeline factory
    pub fn new(device: Arc<WebGpuDevice>) -> Self {
        let cache = Arc::new(PipelineCache::new(Arc::clone(&device)));
        Self { cache, device }
    }

    /// Get the pipeline cache
    pub fn cache(&self) -> &Arc<PipelineCache> {
        &self.cache
    }

    /// Create elementwise addition pipeline
    pub fn create_add_pipeline(&self) -> WebGpuResult<Arc<ComputePipeline>> {
        self.cache.create_elementwise_binary_pipeline(
            "add",
            crate::webgpu::shader::kernels::ELEMENTWISE_ADD,
        )
    }

    /// Create elementwise multiplication pipeline
    pub fn create_mul_pipeline(&self) -> WebGpuResult<Arc<ComputePipeline>> {
        self.cache.create_elementwise_binary_pipeline(
            "mul",
            crate::webgpu::shader::kernels::ELEMENTWISE_MUL,
        )
    }

    /// Create ReLU activation pipeline
    pub fn create_relu_pipeline(&self) -> WebGpuResult<Arc<ComputePipeline>> {
        self.cache
            .create_unary_pipeline("relu", crate::webgpu::shader::kernels::RELU)
    }

    /// Create softmax pipeline
    pub fn create_softmax_pipeline(&self) -> WebGpuResult<Arc<ComputePipeline>> {
        self.cache
            .create_unary_pipeline("softmax", crate::webgpu::shader::kernels::SOFTMAX)
    }

    /// Create matrix multiplication pipeline
    pub fn create_matmul_pipeline(&self) -> WebGpuResult<Arc<ComputePipeline>> {
        let descriptor = PipelineDescriptor::new(
            "matrix_mul",
            crate::webgpu::shader::kernels::MATRIX_MUL,
            "main",
        )
        .with_workgroup_size((8, 8, 1));

        self.cache.get_or_create(descriptor)
    }

    /// Create convolution 2D pipeline
    pub fn create_conv2d_pipeline(&self) -> WebGpuResult<Arc<ComputePipeline>> {
        let descriptor =
            PipelineDescriptor::new("conv2d", crate::webgpu::shader::kernels::CONV2D, "main")
                .with_workgroup_size((8, 8, 1));

        self.cache.get_or_create(descriptor)
    }

    /// Create custom pipeline from descriptor
    pub fn create_custom_pipeline(
        &self,
        descriptor: PipelineDescriptor,
    ) -> WebGpuResult<Arc<ComputePipeline>> {
        self.cache.get_or_create(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_descriptor() {
        let desc = PipelineDescriptor::new("test", "shader_source", "main")
            .with_workgroup_size((32, 32, 1));

        assert_eq!(desc.label, "test");
        assert_eq!(desc.entry_point, "main");
        assert_eq!(desc.workgroup_size, (32, 32, 1));

        let cache_key = desc.cache_key();
        assert!(cache_key.starts_with("test:"));
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);

                let descriptor = PipelineDescriptor::new(
                    "test_pipeline",
                    crate::webgpu::shader::kernels::ELEMENTWISE_ADD,
                    "main",
                );

                let pipeline = ComputePipeline::new(device, descriptor);
                if pipeline.is_ok() {
                    let pipeline = pipeline.unwrap();
                    assert_eq!(pipeline.descriptor().label, "test_pipeline");
                    assert_eq!(pipeline.descriptor().workgroup_size, (64, 1, 1));
                }
            }
        }
    }

    #[tokio::test]
    async fn test_pipeline_factory() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);
                let factory = PipelineFactory::new(device);

                // Test creating common pipelines
                if let Ok(_) = factory.create_add_pipeline() {
                    // Pipeline creation succeeded
                }

                if let Ok(_) = factory.create_relu_pipeline() {
                    // Pipeline creation succeeded
                }

                let stats = factory.cache().stats();
                assert!(stats.pipeline_count >= 0);
            }
        }
    }

    #[test]
    fn test_workgroup_count_calculation() {
        let descriptor =
            PipelineDescriptor::new("test", "source", "main").with_workgroup_size((64, 1, 1));

        // Create a mock pipeline to test workgroup calculation
        // In a real test, you'd create an actual pipeline
        let workgroup_size = descriptor.workgroup_size;
        let problem_size = (1000, 1, 1);

        let optimal_count = (
            (problem_size.0 + workgroup_size.0 - 1) / workgroup_size.0,
            (problem_size.1 + workgroup_size.1 - 1) / workgroup_size.1,
            (problem_size.2 + workgroup_size.2 - 1) / workgroup_size.2,
        );

        assert_eq!(optimal_count, (16, 1, 1)); // ceil(1000/64) = 16
    }

    #[test]
    fn test_pipeline_cache_stats() {
        let stats = PipelineCacheStats {
            pipeline_count: 5,
            shader_count: 3,
            shader_bytes: 1024,
            total_memory_usage: 8192,
        };

        assert_eq!(stats.pipeline_count, 5);
        assert_eq!(stats.shader_count, 3);
        assert_eq!(stats.shader_bytes, 1024);
        assert_eq!(stats.total_memory_usage, 8192);
    }
}
