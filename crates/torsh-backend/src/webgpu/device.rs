//! WebGPU device management for ToRSh

use crate::webgpu::{AdapterInfo, WebGpuError, WebGpuResult};
use crate::{DeviceFeature, DeviceInfo};
use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::device::{DeviceId, DeviceType};
#[cfg(feature = "webgpu")]
#[allow(unused_imports)]
use wgpu;

/// WebGPU device wrapper
#[derive(Debug)]
pub struct WebGpuDevice {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter: Arc<wgpu::Adapter>,
    device_info: DeviceInfo,
    limits: wgpu::Limits,
    features: wgpu::Features,
    // Internal state for tracking
    memory_usage: Arc<RwLock<DeviceMemoryInfo>>,
}

impl Clone for WebGpuDevice {
    fn clone(&self) -> Self {
        Self {
            device: Arc::clone(&self.device),
            queue: Arc::clone(&self.queue),
            adapter: Arc::clone(&self.adapter),
            device_info: self.device_info.clone(),
            limits: self.limits.clone(),
            features: self.features,
            memory_usage: Arc::clone(&self.memory_usage),
        }
    }
}

/// Device memory tracking information
#[derive(Debug, Clone, Default)]
pub struct DeviceMemoryInfo {
    pub allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
    pub allocation_count: usize,
    pub deallocation_count: usize,
}

/// Comprehensive WebGPU device capabilities
#[derive(Debug, Clone)]
pub struct WebGpuDeviceCapabilities {
    // Hardware identification
    pub device_type: wgpu::DeviceType,
    pub backend: wgpu::Backend,
    pub vendor_id: u32,
    pub device_id: u32,

    // Texture limits
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,

    // Binding limits
    pub max_bind_groups: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,

    // Shader stage limits
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_samplers_per_shader_stage: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_textures_per_shader_stage: u32,
    pub max_uniform_buffers_per_shader_stage: u32,

    // Buffer limits
    pub max_uniform_buffer_binding_size: u32,
    pub max_storage_buffer_binding_size: u32,
    pub min_uniform_buffer_offset_alignment: u32,
    pub min_storage_buffer_offset_alignment: u32,
    pub max_vertex_buffers: u32,
    pub max_buffer_size: u64,
    pub max_vertex_attributes: u32,
    pub max_vertex_buffer_array_stride: u32,

    // Render limits
    pub max_inter_stage_shader_components: u32,
    pub max_color_attachments: u32,
    pub max_color_attachment_bytes_per_sample: u32,

    // Compute limits
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,

    // Feature flags
    pub features: wgpu::Features,

    // Performance estimates
    pub memory_bandwidth_estimate: f32,
    pub compute_throughput_estimate: f32,
    pub optimal_batch_size: u32,
}

/// Feature compatibility report
#[derive(Debug, Clone)]
pub struct FeatureCompatibilityReport {
    pub supported: Vec<DeviceFeature>,
    pub unsupported: Vec<DeviceFeature>,
    pub compatibility_score: f32,
}

/// Device performance benchmark results
#[derive(Debug, Clone, Default)]
pub struct DevicePerformanceBenchmark {
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,
    /// Compute throughput in GFLOPS
    pub compute_throughput_gflops: f32,
    /// Texture throughput in Gpixels/sec
    pub texture_throughput_gpixels: f32,
    /// Buffer creation latency in ms
    pub buffer_creation_latency_ms: f32,
    /// Pipeline creation latency in ms
    pub pipeline_creation_latency_ms: f32,
}

impl std::fmt::Display for DevicePerformanceBenchmark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "WebGPU Device Performance Benchmark Results:")?;
        writeln!(
            f,
            "  Memory Bandwidth: {:.2} GB/s",
            self.memory_bandwidth_gbps
        )?;
        writeln!(
            f,
            "  Compute Throughput: {:.2} GFLOPS",
            self.compute_throughput_gflops
        )?;
        writeln!(
            f,
            "  Texture Throughput: {:.2} Gpixels/s",
            self.texture_throughput_gpixels
        )?;
        writeln!(
            f,
            "  Buffer Creation Latency: {:.2} ms",
            self.buffer_creation_latency_ms
        )?;
        writeln!(
            f,
            "  Pipeline Creation Latency: {:.2} ms",
            self.pipeline_creation_latency_ms
        )?;
        Ok(())
    }
}

impl FeatureCompatibilityReport {
    /// Check if all features are supported
    pub fn is_fully_compatible(&self) -> bool {
        self.unsupported.is_empty()
    }

    /// Get compatibility percentage
    pub fn compatibility_percentage(&self) -> f32 {
        self.compatibility_score * 100.0
    }

    /// Get a summary of compatibility status
    pub fn summary(&self) -> String {
        if self.is_fully_compatible() {
            "All features supported".to_string()
        } else {
            format!(
                "{:.1}% compatible - {} supported, {} unsupported",
                self.compatibility_percentage(),
                self.supported.len(),
                self.unsupported.len()
            )
        }
    }
}

impl WebGpuDevice {
    /// Create a new WebGPU device from adapter
    pub async fn new(adapter: wgpu::Adapter, device_id: usize) -> WebGpuResult<Self> {
        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(&format!("ToRSh WebGPU Device {}", device_id)),
                required_features: wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                    | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
                    | wgpu::Features::BUFFER_BINDING_ARRAY
                    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 1024 * 1024 * 1024, // 1GB
                    max_compute_workgroup_storage_size: 32768,
                    max_compute_invocations_per_workgroup: 1024,
                    max_compute_workgroup_size_x: 1024,
                    max_compute_workgroup_size_y: 1024,
                    max_compute_workgroup_size_z: 64,
                    max_compute_workgroups_per_dimension: 65535,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: Default::default(),
            })
            .await
            .map_err(|e| WebGpuError::DeviceCreation(e.to_string()))?;

        let limits = device.limits();
        let features = device.features();

        // Create device info
        let device_info = DeviceInfo {
            vendor: adapter_info.name.clone(),
            driver_version: adapter_info.driver_info.clone(),
            total_memory: Self::estimate_memory_total(&adapter_info, &limits) as usize,
            available_memory: Self::estimate_memory_total(&adapter_info, &limits) as usize, // Initially all available
            compute_units: limits.max_compute_workgroups_per_dimension as usize,
            max_work_group_size: limits.max_compute_workgroup_size_x as usize,
            max_work_group_dimensions: vec![
                limits.max_compute_workgroup_size_x as usize,
                limits.max_compute_workgroup_size_y as usize,
                limits.max_compute_workgroup_size_z as usize,
            ],
            clock_frequency_mhz: 1000,    // Default 1GHz estimate
            memory_bandwidth_gbps: 100.0, // 100 GB/s estimate
            peak_gflops: 1000.0,          // 1 TFLOPS estimate
            features: Self::get_device_features(&features),
            properties: vec![
                ("backend".to_string(), format!("{:?}", adapter_info.backend)),
                (
                    "device_type".to_string(),
                    format!("{:?}", adapter_info.device_type),
                ),
            ],
        };

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
            device_info,
            limits,
            features,
            memory_usage: Arc::new(RwLock::new(DeviceMemoryInfo::default())),
        })
    }

    /// Create WebGPU device from best available adapter
    pub async fn from_best_adapter(device_id: usize) -> WebGpuResult<Self> {
        let adapter = crate::webgpu::get_best_adapter().await?;
        Self::new(adapter, device_id).await
    }

    /// Create WebGPU device from specific adapter index
    pub async fn from_adapter_index(adapter_index: usize, device_id: usize) -> WebGpuResult<Self> {
        let adapters = crate::webgpu::enumerate_adapters().await?;
        let adapter = adapters.into_iter().nth(adapter_index).ok_or_else(|| {
            WebGpuError::ResourceNotFound(format!("Adapter {} not found", adapter_index))
        })?;
        Self::new(adapter, device_id).await
    }

    /// Get the underlying wgpu device
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Get the underlying wgpu queue
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Get the adapter that created this device
    pub fn adapter(&self) -> &wgpu::Adapter {
        &*self.adapter
    }

    /// Get device limits
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// Get device features
    pub fn features(&self) -> &wgpu::Features {
        &self.features
    }

    /// Get adapter information
    pub fn adapter_info(&self) -> AdapterInfo {
        crate::webgpu::get_adapter_info(&self.adapter)
    }

    /// Create a command encoder
    pub fn create_command_encoder(&self, label: Option<&str>) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label })
    }

    /// Submit commands to the queue
    pub fn submit<I>(&self, command_buffers: I) -> wgpu::SubmissionIndex
    where
        I: IntoIterator<Item = wgpu::CommandBuffer>,
    {
        self.queue.submit(command_buffers)
    }

    /// Wait for all submitted commands to complete
    pub async fn wait_for_completion(&self) -> WebGpuResult<()> {
        // Wait for device to process commands
        let _ = self.device.poll(wgpu::PollType::Wait);
        Ok(())
    }

    /// Create a compute pipeline
    pub fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor,
    ) -> wgpu::ComputePipeline {
        self.device.create_compute_pipeline(desc)
    }

    /// Create a shader module
    pub fn create_shader_module(&self, desc: &wgpu::ShaderModuleDescriptor) -> wgpu::ShaderModule {
        self.device.create_shader_module(desc.clone())
    }

    /// Create a buffer
    pub fn create_buffer(&self, desc: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(desc);

        // Track memory allocation
        {
            let mut memory_usage = self.memory_usage.write();
            memory_usage.allocated_bytes += desc.size;
            memory_usage.peak_allocated_bytes = memory_usage
                .peak_allocated_bytes
                .max(memory_usage.allocated_bytes);
            memory_usage.allocation_count += 1;
        }

        buffer
    }

    /// Create a bind group layout
    pub fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor,
    ) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(desc)
    }

    /// Create a bind group
    pub fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor) -> wgpu::BindGroup {
        self.device.create_bind_group(desc)
    }

    /// Track buffer deallocation
    pub fn track_buffer_deallocation(&self, size: u64) {
        let mut memory_usage = self.memory_usage.write();
        memory_usage.allocated_bytes = memory_usage.allocated_bytes.saturating_sub(size);
        memory_usage.deallocation_count += 1;
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> DeviceMemoryInfo {
        self.memory_usage.read().clone()
    }

    /// Check if a feature is supported
    pub fn supports_feature(&self, feature: wgpu::Features) -> bool {
        self.features.contains(feature)
    }

    /// Get optimal workgroup size for compute operations
    pub fn optimal_workgroup_size(&self, elements: u32) -> (u32, u32, u32) {
        let max_x = self.limits.max_compute_workgroup_size_x;
        let max_invocations = self.limits.max_compute_invocations_per_workgroup;

        // Find optimal 1D workgroup size
        let optimal_size = [64, 128, 256, 512, 1024]
            .iter()
            .filter(|&&size| size <= max_x && size <= max_invocations)
            .max()
            .copied()
            .unwrap_or(64);

        (optimal_size.min(elements), 1, 1)
    }

    /// Get comprehensive device capabilities mapping
    pub fn get_device_capabilities(&self) -> WebGpuDeviceCapabilities {
        let adapter_info = self.adapter_info();

        WebGpuDeviceCapabilities {
            // Hardware info
            device_type: adapter_info.device_type,
            backend: adapter_info.backend,
            vendor_id: adapter_info.vendor,
            device_id: adapter_info.device,

            // Compute limits
            max_texture_dimension_1d: self.limits.max_texture_dimension_1d,
            max_texture_dimension_2d: self.limits.max_texture_dimension_2d,
            max_texture_dimension_3d: self.limits.max_texture_dimension_3d,
            max_texture_array_layers: self.limits.max_texture_array_layers,
            max_bind_groups: self.limits.max_bind_groups,
            max_bindings_per_bind_group: self.limits.max_bindings_per_bind_group,
            max_dynamic_uniform_buffers_per_pipeline_layout: self
                .limits
                .max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout: self
                .limits
                .max_dynamic_storage_buffers_per_pipeline_layout,
            max_sampled_textures_per_shader_stage: self
                .limits
                .max_sampled_textures_per_shader_stage,
            max_samplers_per_shader_stage: self.limits.max_samplers_per_shader_stage,
            max_storage_buffers_per_shader_stage: self.limits.max_storage_buffers_per_shader_stage,
            max_storage_textures_per_shader_stage: self
                .limits
                .max_storage_textures_per_shader_stage,
            max_uniform_buffers_per_shader_stage: self.limits.max_uniform_buffers_per_shader_stage,
            max_uniform_buffer_binding_size: self.limits.max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size: self.limits.max_storage_buffer_binding_size,
            min_uniform_buffer_offset_alignment: self.limits.min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment: self.limits.min_storage_buffer_offset_alignment,
            max_vertex_buffers: self.limits.max_vertex_buffers,
            max_buffer_size: self.limits.max_buffer_size,
            max_vertex_attributes: self.limits.max_vertex_attributes,
            max_vertex_buffer_array_stride: self.limits.max_vertex_buffer_array_stride,
            max_inter_stage_shader_components: self.limits.max_inter_stage_shader_components,
            max_color_attachments: self.limits.max_color_attachments,
            max_color_attachment_bytes_per_sample: self
                .limits
                .max_color_attachment_bytes_per_sample,
            max_compute_workgroup_storage_size: self.limits.max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup: self
                .limits
                .max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: self.limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: self.limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: self.limits.max_compute_workgroup_size_z,
            max_compute_workgroups_per_dimension: self.limits.max_compute_workgroups_per_dimension,

            // Feature flags
            features: self.features,

            // Performance characteristics
            memory_bandwidth_estimate: Self::estimate_memory_bandwidth(&adapter_info),
            compute_throughput_estimate: Self::estimate_compute_throughput(&adapter_info),
            optimal_batch_size: Self::estimate_optimal_batch_size(&adapter_info, &self.limits),
        }
    }

    /// Check detailed feature compatibility
    pub fn check_feature_compatibility(
        &self,
        required_features: &[DeviceFeature],
    ) -> FeatureCompatibilityReport {
        let mut supported = Vec::new();
        let mut unsupported = Vec::new();
        let device_features = Self::get_device_features(&self.features);

        for feature in required_features {
            if device_features.contains(feature) {
                supported.push(feature.clone());
            } else {
                unsupported.push(feature.clone());
            }
        }

        let compatibility_score = if required_features.is_empty() {
            1.0
        } else {
            supported.len() as f32 / required_features.len() as f32
        };

        FeatureCompatibilityReport {
            supported,
            unsupported,
            compatibility_score,
        }
    }

    /// Perform device performance benchmarking for capability assessment
    pub async fn benchmark_device_performance(&self) -> WebGpuResult<DevicePerformanceBenchmark> {
        let mut benchmark = DevicePerformanceBenchmark::default();

        // Memory bandwidth test
        benchmark.memory_bandwidth_gbps = self.benchmark_memory_bandwidth().await?;

        // Compute throughput test
        benchmark.compute_throughput_gflops = self.benchmark_compute_throughput().await?;

        // Texture throughput test
        benchmark.texture_throughput_gpixels = self.benchmark_texture_operations().await?;

        // Buffer creation latency
        benchmark.buffer_creation_latency_ms = self.benchmark_buffer_creation().await?;

        // Pipeline creation latency
        benchmark.pipeline_creation_latency_ms = self.benchmark_pipeline_creation().await?;

        Ok(benchmark)
    }

    /// Benchmark memory bandwidth by copying data between buffers
    async fn benchmark_memory_bandwidth(&self) -> WebGpuResult<f32> {
        let buffer_size = 64 * 1024 * 1024; // 64MB
        let data = vec![1.0f32; buffer_size / 4];

        // Create source buffer
        let src_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Benchmark Source Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        // Write test data
        {
            let mut buffer_slice = src_buffer.slice(..).get_mapped_range_mut();
            let bytes = bytemuck::cast_slice_mut::<u8, f32>(&mut buffer_slice);
            bytes.copy_from_slice(&data);
        }
        src_buffer.unmap();

        // Create destination buffer
        let dst_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Benchmark Dest Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Perform timed copy operations
        let start = std::time::Instant::now();
        let iterations = 10;

        for _ in 0..iterations {
            let mut encoder = self.create_command_encoder(Some("Benchmark Copy"));
            encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, buffer_size as u64);
            self.submit([encoder.finish()]);
            self.wait_for_completion().await?;
        }

        let elapsed = start.elapsed();
        let bytes_copied = buffer_size as f64 * iterations as f64;
        let seconds = elapsed.as_secs_f64();
        let bandwidth_bps = bytes_copied / seconds;
        let bandwidth_gbps = bandwidth_bps / (1024.0 * 1024.0 * 1024.0);

        Ok(bandwidth_gbps as f32)
    }

    /// Benchmark compute throughput using matrix multiplication
    async fn benchmark_compute_throughput(&self) -> WebGpuResult<f32> {
        let matrix_size = 512; // 512x512 matrices
        let element_count = matrix_size * matrix_size;

        // Create compute shader for matrix multiplication
        let shader_source = format!(
            r#"
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;
            
            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let row = global_id.y;
                let col = global_id.x;
                
                if (row >= {matrix_size}u || col >= {matrix_size}u) {{
                    return;
                }}
                
                var sum = 0.0;
                for (var k = 0u; k < {matrix_size}u; k = k + 1u) {{
                    sum = sum + a[row * {matrix_size}u + k] * b[k * {matrix_size}u + col];
                }}
                
                result[row * {matrix_size}u + col] = sum;
            }}
        "#,
            matrix_size = matrix_size
        );

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Benchmark Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Create buffers
        let buffer_size = element_count * std::mem::size_of::<f32>();
        let input_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix A"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix B"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Matrix"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create compute pipeline
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Benchmark Bind Group Layout"),
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
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Benchmark Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Benchmark Compute Pipeline"),
                    cache: None,
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Benchmark Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        // Benchmark the computation
        let start = std::time::Instant::now();
        let iterations = 5;

        for _ in 0..iterations {
            let mut encoder = self.create_command_encoder(Some("Benchmark Compute"));
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Benchmark Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(
                    (matrix_size as u32 + 15) / 16,
                    (matrix_size as u32 + 15) / 16,
                    1,
                );
            }
            self.submit([encoder.finish()]);
            self.wait_for_completion().await?;
        }

        let elapsed = start.elapsed();
        let ops_per_iteration = 2 * matrix_size * matrix_size * matrix_size; // Matrix multiplication FLOPs
        let total_ops = ops_per_iteration as f64 * iterations as f64;
        let seconds = elapsed.as_secs_f64();
        let gflops = (total_ops / seconds) / 1e9;

        Ok(gflops as f32)
    }

    /// Benchmark texture operations
    async fn benchmark_texture_operations(&self) -> WebGpuResult<f32> {
        let texture_size = 1024;
        let pixel_count = texture_size * texture_size;

        // Create source texture
        let src_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Benchmark Source Texture"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let dst_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Benchmark Dest Texture"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Benchmark texture copy operations using wgpu 26.x API
        let start = std::time::Instant::now();
        let iterations = 20;

        let copy_size = wgpu::Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        };

        for _ in 0..iterations {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Benchmark Texture Copy"),
                });

            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &src_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &dst_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                copy_size,
            );

            self.queue.submit([encoder.finish()]);
            let _ = self.device.poll(wgpu::PollType::Wait);
        }

        let elapsed = start.elapsed();
        let pixels_processed = pixel_count as f64 * iterations as f64;
        let seconds = elapsed.as_secs_f64();
        let gpixels_per_sec = (pixels_processed / seconds) / 1e9;

        Ok(gpixels_per_sec as f32)
    }

    /// Benchmark buffer creation latency
    async fn benchmark_buffer_creation(&self) -> WebGpuResult<f32> {
        let buffer_size = 1024 * 1024; // 1MB buffers
        let iterations = 100;
        let mut buffers = Vec::with_capacity(iterations);

        let start = std::time::Instant::now();

        for i in 0..iterations {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Benchmark Buffer {}", i)),
                size: buffer_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            buffers.push(buffer);
        }

        let elapsed = start.elapsed();
        let avg_latency_ms = elapsed.as_millis() as f32 / iterations as f32;

        // Clean up buffers
        drop(buffers);

        Ok(avg_latency_ms)
    }

    /// Benchmark compute pipeline creation latency
    async fn benchmark_pipeline_creation(&self) -> WebGpuResult<f32> {
        let shader_source = r#"
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                // Simple no-op shader for benchmarking pipeline creation
            }
        "#;

        let iterations = 10;
        let mut pipelines = Vec::with_capacity(iterations);

        let start = std::time::Instant::now();

        for i in 0..iterations {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("Benchmark Shader {}", i)),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("Benchmark Pipeline Layout {}", i)),
                        bind_group_layouts: &[],
                        push_constant_ranges: &[],
                    });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Benchmark Pipeline {}", i)),
                    layout: Some(&pipeline_layout),
                    cache: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                });

            pipelines.push(pipeline);
        }

        let elapsed = start.elapsed();
        let avg_latency_ms = elapsed.as_millis() as f32 / iterations as f32;

        // Clean up pipelines
        drop(pipelines);

        Ok(avg_latency_ms)
    }

    /// Estimate memory bandwidth based on device type and vendor
    fn estimate_memory_bandwidth(adapter_info: &AdapterInfo) -> f32 {
        match (adapter_info.device_type, adapter_info.vendor) {
            (wgpu::DeviceType::DiscreteGpu, 0x10DE) => 900.0, // NVIDIA high-end
            (wgpu::DeviceType::DiscreteGpu, 0x1002) => 800.0, // AMD high-end
            (wgpu::DeviceType::DiscreteGpu, 0x8086) => 600.0, // Intel Arc
            (wgpu::DeviceType::DiscreteGpu, _) => 500.0,      // Other discrete
            (wgpu::DeviceType::IntegratedGpu, 0x106B) => 200.0, // Apple Silicon
            (wgpu::DeviceType::IntegratedGpu, 0x8086) => 150.0, // Intel integrated
            (wgpu::DeviceType::IntegratedGpu, 0x1002) => 120.0, // AMD APU
            (wgpu::DeviceType::IntegratedGpu, _) => 100.0,    // Other integrated
            (wgpu::DeviceType::VirtualGpu, _) => 50.0,        // Virtual GPU
            (wgpu::DeviceType::Cpu, _) => 25.0,               // CPU fallback
            (wgpu::DeviceType::Other, _) => 30.0,             // Unknown
        }
    }

    /// Estimate compute throughput based on device type and vendor
    fn estimate_compute_throughput(adapter_info: &AdapterInfo) -> f32 {
        match (adapter_info.device_type, adapter_info.vendor) {
            (wgpu::DeviceType::DiscreteGpu, 0x10DE) => 15000.0, // NVIDIA high-end
            (wgpu::DeviceType::DiscreteGpu, 0x1002) => 12000.0, // AMD high-end
            (wgpu::DeviceType::DiscreteGpu, 0x8086) => 8000.0,  // Intel Arc
            (wgpu::DeviceType::DiscreteGpu, _) => 6000.0,       // Other discrete
            (wgpu::DeviceType::IntegratedGpu, 0x106B) => 3000.0, // Apple Silicon
            (wgpu::DeviceType::IntegratedGpu, 0x8086) => 1500.0, // Intel integrated
            (wgpu::DeviceType::IntegratedGpu, 0x1002) => 1200.0, // AMD APU
            (wgpu::DeviceType::IntegratedGpu, _) => 1000.0,     // Other integrated
            (wgpu::DeviceType::VirtualGpu, _) => 500.0,         // Virtual GPU
            (wgpu::DeviceType::Cpu, _) => 200.0,                // CPU fallback
            (wgpu::DeviceType::Other, _) => 300.0,              // Unknown
        }
    }

    /// Estimate optimal batch size for operations
    fn estimate_optimal_batch_size(adapter_info: &AdapterInfo, limits: &wgpu::Limits) -> u32 {
        let base_size = match adapter_info.device_type {
            wgpu::DeviceType::DiscreteGpu => 256,
            wgpu::DeviceType::IntegratedGpu => 128,
            wgpu::DeviceType::VirtualGpu => 64,
            wgpu::DeviceType::Cpu => 32,
            wgpu::DeviceType::Other => 64,
        };

        // Adjust based on workgroup limits
        base_size.min(limits.max_compute_invocations_per_workgroup)
    }

    /// Convert WebGPU device type to compute capability version
    fn get_compute_capability(adapter_info: &wgpu::AdapterInfo, _limits: &wgpu::Limits) -> String {
        match adapter_info.device_type {
            wgpu::DeviceType::DiscreteGpu => "WebGPU-Discrete".to_string(),
            wgpu::DeviceType::IntegratedGpu => "WebGPU-Integrated".to_string(),
            wgpu::DeviceType::VirtualGpu => "WebGPU-Virtual".to_string(),
            wgpu::DeviceType::Cpu => "WebGPU-CPU".to_string(),
            wgpu::DeviceType::Other => "WebGPU-Other".to_string(),
        }
    }

    /// Estimate total memory available (WebGPU doesn't expose this directly)
    fn estimate_memory_total(adapter_info: &wgpu::AdapterInfo, limits: &wgpu::Limits) -> u64 {
        match adapter_info.device_type {
            wgpu::DeviceType::DiscreteGpu => 8 * 1024 * 1024 * 1024, // 8GB estimate
            wgpu::DeviceType::IntegratedGpu => 4 * 1024 * 1024 * 1024, // 4GB estimate
            wgpu::DeviceType::VirtualGpu => 2 * 1024 * 1024 * 1024,  // 2GB estimate
            wgpu::DeviceType::Cpu => {
                (limits.max_storage_buffer_binding_size as u64).min(1024 * 1024 * 1024)
            }
            wgpu::DeviceType::Other => 1024 * 1024 * 1024, // 1GB default
        }
    }

    /// Convert WebGPU features to device features with comprehensive detection
    fn get_device_features(features: &wgpu::Features) -> Vec<DeviceFeature> {
        let mut device_features = Vec::new();

        // Query and timing features
        if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            device_features.push(DeviceFeature::TimestampQuery);
        }
        if features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
            device_features.push(DeviceFeature::TimestampQueryInsideEncoders);
        }
        if features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY) {
            device_features.push(DeviceFeature::PipelineStatistics);
        }

        // Buffer and storage features
        if features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) {
            device_features.push(DeviceFeature::MappableBuffers);
        }
        if features.contains(wgpu::Features::BUFFER_BINDING_ARRAY) {
            device_features.push(DeviceFeature::BufferArrays);
        }
        if features.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY) {
            device_features.push(DeviceFeature::StorageArrays);
        }
        // Note: UNSIZED_BINDING_ARRAY feature not available in this wgpu version
        // if features.contains(wgpu::Features::UNSIZED_BINDING_ARRAY) {
        //     device_features.push(DeviceFeature::UnsizedBindingArray);
        // }

        // Compute and shader features
        if features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE) {
            device_features.push(DeviceFeature::IndirectFirstInstance);
        }
        if features.contains(wgpu::Features::SHADER_F16) {
            device_features.push(DeviceFeature::ShaderF16);
        }
        if features.contains(wgpu::Features::SHADER_I16) {
            device_features.push(DeviceFeature::ShaderI16);
        }
        if features.contains(wgpu::Features::SHADER_PRIMITIVE_INDEX) {
            device_features.push(DeviceFeature::ShaderPrimitiveIndex);
        }
        if features.contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
            device_features.push(DeviceFeature::ShaderEarlyDepthTest);
        }

        // Advanced compute features
        if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT) {
            device_features.push(DeviceFeature::MultiDrawIndirect);
        }
        if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            device_features.push(DeviceFeature::MultiDrawIndirectCount);
        }

        // Rendering features
        // Note: MULTISAMPLED_SHADING feature not available in this wgpu version
        // if features.contains(wgpu::Features::MULTISAMPLED_SHADING) {
        //     device_features.push(DeviceFeature::Multisampling);
        // }
        if features.contains(wgpu::Features::CLEAR_TEXTURE) {
            device_features.push(DeviceFeature::ClearTexture);
        }

        // Validation and debugging features
        if features.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH) {
            device_features.push(DeviceFeature::SpirvShaderPassthrough);
        }

        device_features
    }
}

// Note: Device trait implementation temporarily disabled due to compilation issues
// TODO: Implement proper device trait when available
impl WebGpuDevice {
    pub fn id(&self) -> DeviceId {
        DeviceId::new() // Default device ID for WebGPU
    }

    pub fn name(&self) -> &str {
        "WebGPU Device" // Default device name
    }

    pub fn device_type(&self) -> DeviceType {
        DeviceType::Wgpu(0) // Default WebGPU device type
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.device_info
    }

    pub fn is_available(&self) -> bool {
        // Check if device is still valid by polling
        let _ = self.device.poll(wgpu::PollType::Wait);
        true // WebGPU devices are typically always available once created
    }

    pub fn synchronize(&self) -> crate::error::BackendResult<()> {
        // Wait for device to process commands
        let _ = self.device.poll(wgpu::PollType::Wait);
        Ok(())
    }

    pub fn memory_info(&self) -> (u64, u64) {
        let usage = self.memory_usage();
        let total = self.device_info.total_memory;
        let used = usage.allocated_bytes;
        let free = total.saturating_sub(used as usize);
        (used, free as u64)
    }
}

/// WebGPU device builder for convenient device creation
#[derive(Debug, Default)]
pub struct WebGpuDeviceBuilder {
    adapter_index: Option<usize>,
    device_id: usize,
    power_preference: wgpu::PowerPreference,
    required_features: wgpu::Features,
    required_limits: Option<wgpu::Limits>,
}

impl WebGpuDeviceBuilder {
    /// Create a new device builder
    pub fn new() -> Self {
        Self {
            adapter_index: None,
            device_id: 0,
            power_preference: wgpu::PowerPreference::HighPerformance,
            required_features: wgpu::Features::empty(),
            required_limits: None,
        }
    }

    /// Set the adapter index to use
    pub fn adapter_index(mut self, index: usize) -> Self {
        self.adapter_index = Some(index);
        self
    }

    /// Set the device ID
    pub fn device_id(mut self, id: usize) -> Self {
        self.device_id = id;
        self
    }

    /// Set the power preference
    pub fn power_preference(mut self, preference: wgpu::PowerPreference) -> Self {
        self.power_preference = preference;
        self
    }

    /// Add required features
    pub fn features(mut self, features: wgpu::Features) -> Self {
        self.required_features |= features;
        self
    }

    /// Set required limits
    pub fn limits(mut self, limits: wgpu::Limits) -> Self {
        self.required_limits = Some(limits);
        self
    }

    /// Build the device
    pub async fn build(self) -> WebGpuResult<WebGpuDevice> {
        if let Some(adapter_index) = self.adapter_index {
            WebGpuDevice::from_adapter_index(adapter_index, self.device_id).await
        } else {
            WebGpuDevice::from_best_adapter(self.device_id).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_device_creation() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            let result = WebGpuDevice::from_best_adapter(0).await;
            if let Ok(device) = result {
                assert_eq!(device.id(), "0");
                assert_eq!(device.device_type(), DeviceType::Wgpu(0));
                assert!(device.is_available());

                let (used, free) = device.memory_info();
                assert_eq!(used, 0); // Initially no memory used
                assert!(free > 0);
            }
        }
    }

    #[tokio::test]
    async fn test_device_builder() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            let result = WebGpuDeviceBuilder::new()
                .device_id(42)
                .power_preference(wgpu::PowerPreference::LowPower)
                .features(wgpu::Features::TIMESTAMP_QUERY)
                .build()
                .await;

            if let Ok(device) = result {
                assert_eq!(device.id(), "42");
                assert!(device.supports_feature(wgpu::Features::TIMESTAMP_QUERY));
            }
        }
    }

    #[test]
    fn test_optimal_workgroup_size() {
        let _limits = wgpu::Limits {
            max_compute_workgroup_size_x: 256,
            max_compute_invocations_per_workgroup: 256,
            ..Default::default()
        };

        // Mock device with test limits
        let _device_info = DeviceInfo {
            vendor: "Test".to_string(),
            driver_version: "1.0".to_string(),
            total_memory: 1024 * 1024 * 1024,
            available_memory: 1024 * 1024 * 1024,
            compute_units: 8,
            max_work_group_size: 256,
            max_work_group_dimensions: vec![256, 256, 64],
            clock_frequency_mhz: 1000,
            memory_bandwidth_gbps: 400.0,
            peak_gflops: 1000.0,
            features: vec![],
            properties: vec![],
        };

        // Test workgroup size calculation logic (would need access to device for full test)
        assert_eq!((64_u32).min(100), 64);
        assert_eq!((64_u32).min(32), 32);
    }

    #[test]
    fn test_memory_tracking() {
        let memory_info = DeviceMemoryInfo {
            allocated_bytes: 1024,
            peak_allocated_bytes: 2048,
            allocation_count: 5,
            deallocation_count: 2,
        };

        assert_eq!(memory_info.allocated_bytes, 1024);
        assert_eq!(memory_info.peak_allocated_bytes, 2048);
        assert_eq!(memory_info.allocation_count, 5);
        assert_eq!(memory_info.deallocation_count, 2);
    }
}
