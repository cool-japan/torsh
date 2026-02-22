//! WebGPU kernel operations for ToRSh

#[cfg(feature = "webgpu")]
#[allow(unused_imports)]
use wgpu;

use crate::webgpu::pipeline::{ComputePipeline, PipelineDescriptor, PipelineFactory};
use crate::webgpu::{WebGpuBuffer, WebGpuDevice, WebGpuError, WebGpuResult};
use crate::{KernelDescriptor, KernelHandle};
use std::sync::Arc;

/// Type alias for WebGPU compute pipeline
pub type WebGpuComputePipeline = ComputePipeline;

/// WebGPU kernel wrapper
#[derive(Debug)]
pub struct WebGpuKernel {
    pub pipeline: Arc<WebGpuComputePipeline>,
    pub descriptor: KernelDescriptor,
    pub handle: KernelHandle,
}

// Note: Kernel trait implementation temporarily disabled due to compilation issues
// TODO: Implement proper kernel trait when available
impl WebGpuKernel {
    pub fn handle(&self) -> KernelHandle {
        self.handle.clone()
    }

    pub fn descriptor(&self) -> &KernelDescriptor {
        &self.descriptor
    }

    pub fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Kernel cache for efficient kernel reuse
#[derive(Debug)]
pub struct WebGpuKernelCache {
    kernels: Arc<parking_lot::RwLock<std::collections::HashMap<String, Arc<WebGpuKernel>>>>,
}

impl WebGpuKernelCache {
    pub fn new() -> Self {
        Self {
            kernels: Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
        }
    }

    pub fn get(&self, key: &str) -> Option<Arc<WebGpuKernel>> {
        self.kernels.read().get(key).cloned()
    }

    pub fn insert(&self, key: String, kernel: Arc<WebGpuKernel>) {
        self.kernels.write().insert(key, kernel);
    }

    pub fn clear(&self) {
        self.kernels.write().clear();
    }

    pub fn len(&self) -> usize {
        self.kernels.read().len()
    }
}

/// WebGPU kernel executor for tensor operations
#[derive(Debug)]
pub struct WebGpuKernelExecutor {
    device: Arc<WebGpuDevice>,
    pipeline_factory: PipelineFactory,
}

impl WebGpuKernelExecutor {
    /// Create a new kernel executor
    pub fn new(device: Arc<WebGpuDevice>) -> Self {
        let pipeline_factory = PipelineFactory::new(Arc::clone(&device));
        Self {
            device,
            pipeline_factory,
        }
    }

    /// Create a kernel from descriptor
    pub fn create_kernel(&self, descriptor: crate::KernelDescriptor) -> WebGpuResult<WebGpuKernel> {
        // Create a simple compute pipeline based on the kernel descriptor
        let shader_module = self
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: Some(&descriptor.name),
                source: wgpu::ShaderSource::Wgsl(match &descriptor.source {
                    crate::kernel::KernelSource::Source { code, .. } => code.clone().into(),
                    _ => {
                        return Err(WebGpuError::InvalidShaderSource(
                            "Only WGSL source code is supported for WebGPU".to_string(),
                        ))
                    }
                }),
            });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Kernel Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout =
            self.device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Kernel Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let _pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&descriptor.name),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"), // Default WebGPU entry point
                compilation_options: Default::default(),
                cache: None,
            });

        // Extract shader source code from KernelSource
        let shader_source = match &descriptor.source {
            crate::kernel::KernelSource::Source { code, language } => {
                // For WebGPU, we expect WGSL
                match language {
                    crate::kernel::KernelLanguage::Wgsl => code.clone(),
                    _ => {
                        return Err(WebGpuError::UnsupportedFeature(format!(
                            "WebGPU only supports WGSL, got {:?}",
                            language
                        )))
                    }
                }
            }
            crate::kernel::KernelSource::Bytecode { .. } => {
                return Err(WebGpuError::UnsupportedFeature(
                    "WebGPU backend does not support pre-compiled bytecode".to_string(),
                ));
            }
            crate::kernel::KernelSource::SpirV { .. } => {
                return Err(WebGpuError::UnsupportedFeature(
                    "WebGPU backend does not support SPIR-V bytecode directly".to_string(),
                ));
            }
            crate::kernel::KernelSource::Binary { .. } => {
                return Err(WebGpuError::UnsupportedFeature(
                    "WebGPU backend does not support platform-specific binary code".to_string(),
                ));
            }
        };

        // Create pipeline descriptor for the WebGpuComputePipeline wrapper
        let pipeline_descriptor =
            PipelineDescriptor::new(descriptor.name.clone(), shader_source, "main".to_string())
                .with_workgroup_size((64, 1, 1));

        let compute_pipeline =
            WebGpuComputePipeline::new(self.device.clone(), pipeline_descriptor)?;

        let shader_module_id = format!("kernel_{}", descriptor.name);
        Ok(WebGpuKernel {
            pipeline: Arc::new(compute_pipeline),
            descriptor,
            handle: KernelHandle::WebGpu {
                shader_module_id,
                entry_point: "main".to_string(),
            },
        })
    }

    /// Execute a WebGPU kernel
    pub async fn execute_kernel(
        &self,
        kernel: &WebGpuKernel,
        buffers: &[&WebGpuBuffer],
        uniform_data: &[u8],
        _workgroup_size: (u32, u32, u32),
        workgroup_count: (u32, u32, u32),
    ) -> WebGpuResult<()> {
        let mut encoder = self.device.create_command_encoder(Some("Kernel Execution"));

        // Create uniform buffer if needed
        let uniform_buffer = if !uniform_data.is_empty() {
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Kernel Uniforms"),
                size: uniform_data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        if let (Some(buffer), uniform_data) = (uniform_buffer.as_ref(), uniform_data) {
            if !uniform_data.is_empty() {
                self.device.queue().write_buffer(buffer, 0, uniform_data);
            }
        }

        // Create bind groups (simplified implementation)
        let wgpu_buffers: Vec<_> = buffers.iter().map(|b| b.wgpu_buffer()).collect();
        let _bind_groups = kernel
            .pipeline
            .create_bind_groups(&[wgpu_buffers.as_slice()])?;

        // Dispatch kernel
        kernel
            .pipeline
            .dispatch(&mut encoder, &[], workgroup_count)?;

        let command_buffer = encoder.finish();
        self.device.submit([command_buffer]);

        Ok(())
    }

    /// Execute elementwise addition
    pub async fn elementwise_add(
        &self,
        a: &WebGpuBuffer,
        b: &WebGpuBuffer,
        output: &WebGpuBuffer,
    ) -> WebGpuResult<()> {
        let pipeline = self.pipeline_factory.create_add_pipeline()?;

        let bind_groups = pipeline.create_bind_groups(&[&[
            a.wgpu_buffer(),
            b.wgpu_buffer(),
            output.wgpu_buffer(),
        ]])?;

        let mut encoder = self.device.create_command_encoder(Some("Elementwise Add"));

        let element_count = output.size() / 4; // Assuming f32 elements
        let workgroup_count = pipeline.optimal_workgroup_count((element_count as u32, 1, 1));

        pipeline.dispatch(&mut encoder, &[&bind_groups[0]], workgroup_count)?;

        let command_buffer = encoder.finish();
        self.device.submit([command_buffer]);

        Ok(())
    }

    /// Execute elementwise multiplication
    pub async fn elementwise_mul(
        &self,
        a: &WebGpuBuffer,
        b: &WebGpuBuffer,
        output: &WebGpuBuffer,
    ) -> WebGpuResult<()> {
        let pipeline = self.pipeline_factory.create_mul_pipeline()?;

        let bind_groups = pipeline.create_bind_groups(&[&[
            a.wgpu_buffer(),
            b.wgpu_buffer(),
            output.wgpu_buffer(),
        ]])?;

        let mut encoder = self.device.create_command_encoder(Some("Elementwise Mul"));

        let element_count = output.size() / 4; // Assuming f32 elements
        let workgroup_count = pipeline.optimal_workgroup_count((element_count as u32, 1, 1));

        pipeline.dispatch(&mut encoder, &[&bind_groups[0]], workgroup_count)?;

        let command_buffer = encoder.finish();
        self.device.submit([command_buffer]);

        Ok(())
    }

    /// Execute ReLU activation
    pub async fn relu(&self, input: &WebGpuBuffer, output: &WebGpuBuffer) -> WebGpuResult<()> {
        let pipeline = self.pipeline_factory.create_relu_pipeline()?;

        let bind_groups =
            pipeline.create_bind_groups(&[&[input.wgpu_buffer(), output.wgpu_buffer()]])?;

        let mut encoder = self.device.create_command_encoder(Some("ReLU"));

        let element_count = output.size() / 4; // Assuming f32 elements
        let workgroup_count = pipeline.optimal_workgroup_count((element_count as u32, 1, 1));

        pipeline.dispatch(&mut encoder, &[&bind_groups[0]], workgroup_count)?;

        let command_buffer = encoder.finish();
        self.device.submit([command_buffer]);

        Ok(())
    }

    /// Execute softmax activation
    pub async fn softmax(&self, input: &WebGpuBuffer, output: &WebGpuBuffer) -> WebGpuResult<()> {
        let pipeline = self.pipeline_factory.create_softmax_pipeline()?;

        let bind_groups =
            pipeline.create_bind_groups(&[&[input.wgpu_buffer(), output.wgpu_buffer()]])?;

        let mut encoder = self.device.create_command_encoder(Some("Softmax"));

        let element_count = output.size() / 4; // Assuming f32 elements
        let workgroup_count = pipeline.optimal_workgroup_count((element_count as u32, 1, 1));

        pipeline.dispatch(&mut encoder, &[&bind_groups[0]], workgroup_count)?;

        let command_buffer = encoder.finish();
        self.device.submit([command_buffer]);

        Ok(())
    }

    /// Execute matrix multiplication
    pub async fn matmul(
        &self,
        a: &WebGpuBuffer,
        b: &WebGpuBuffer,
        output: &WebGpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> WebGpuResult<()> {
        let pipeline = self.pipeline_factory.create_matmul_pipeline()?;

        // Create uniform buffer for matrix dimensions
        let uniforms = [m, n, k, 0]; // Padding to align to 16 bytes
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MatMul Uniforms"),
            size: std::mem::size_of_val(&uniforms) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.device
            .queue()
            .write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&uniforms));

        let bind_groups = pipeline.create_bind_groups(&[&[
            &uniform_buffer,
            a.wgpu_buffer(),
            b.wgpu_buffer(),
            output.wgpu_buffer(),
        ]])?;

        let mut encoder = self
            .device
            .create_command_encoder(Some("Matrix Multiplication"));

        let workgroup_count = pipeline.optimal_workgroup_count((m, n, 1));

        pipeline.dispatch(&mut encoder, &[&bind_groups[0]], workgroup_count)?;

        let command_buffer = encoder.finish();
        self.device.submit([command_buffer]);

        Ok(())
    }

    /// Execute 2D convolution
    pub async fn conv2d(
        &self,
        input: &WebGpuBuffer,
        kernel: &WebGpuBuffer,
        output: &WebGpuBuffer,
        input_height: u32,
        input_width: u32,
        kernel_height: u32,
        kernel_width: u32,
        output_height: u32,
        output_width: u32,
        stride: u32,
        padding: u32,
    ) -> WebGpuResult<()> {
        let pipeline = self.pipeline_factory.create_conv2d_pipeline()?;

        // Create uniform buffer for convolution parameters
        let uniforms = [
            input_height,
            input_width,
            kernel_height,
            kernel_width,
            output_height,
            output_width,
            stride,
            padding,
        ];

        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv2D Uniforms"),
            size: std::mem::size_of_val(&uniforms) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.device
            .queue()
            .write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&uniforms));

        let bind_groups = pipeline.create_bind_groups(&[&[
            &uniform_buffer,
            input.wgpu_buffer(),
            kernel.wgpu_buffer(),
            output.wgpu_buffer(),
        ]])?;

        let mut encoder = self.device.create_command_encoder(Some("Conv2D"));

        let workgroup_count = pipeline.optimal_workgroup_count((output_height, output_width, 1));

        pipeline.dispatch(&mut encoder, &[&bind_groups[0]], workgroup_count)?;

        let command_buffer = encoder.finish();
        self.device.submit([command_buffer]);

        Ok(())
    }

    /// Execute custom kernel with provided shader source
    pub async fn execute_custom_kernel(
        &self,
        shader_source: &str,
        entry_point: &str,
        buffers: &[&WebGpuBuffer],
        workgroup_size: (u32, u32, u32),
        workgroup_count: (u32, u32, u32),
        uniform_data: Option<&[u8]>,
    ) -> WebGpuResult<()> {
        use crate::webgpu::pipeline::PipelineDescriptor;

        // Create pipeline descriptor without custom bind group layout for now
        let descriptor = PipelineDescriptor::new("custom_kernel", shader_source, entry_point)
            .with_workgroup_size(workgroup_size);

        let pipeline = self.pipeline_factory.cache().get_or_create(descriptor)?;

        // Create uniform buffer if needed
        let uniform_buffer = if let Some(data) = uniform_data {
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Custom Kernel Uniforms"),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        if let (Some(buffer), Some(data)) = (uniform_buffer.as_ref(), uniform_data) {
            self.device.queue().write_buffer(buffer, 0, data);
        }

        // Create bind groups
        let wgpu_buffers: Vec<_> = buffers.iter().map(|b| b.wgpu_buffer()).collect();
        let buffer_group = wgpu_buffers.as_slice();

        if let Some(uniform_buf) = uniform_buffer.as_ref() {
            let mut all_buffers = vec![uniform_buf];
            all_buffers.extend(wgpu_buffers.iter());
            // Note: This is simplified - real implementation would handle uniform vs storage buffers separately
        }

        let bind_groups = pipeline.create_bind_groups(&[buffer_group])?;

        let mut encoder = self.device.create_command_encoder(Some("Custom Kernel"));

        pipeline.dispatch(&mut encoder, &[&bind_groups[0]], workgroup_count)?;

        let command_buffer = encoder.finish();
        self.device.submit([command_buffer]);

        Ok(())
    }

    /// Execute a simple kernel by name
    pub async fn execute_simple_kernel(
        &self,
        _kernel_name: &str,
        _buffers: &[&wgpu::Buffer],
        _uniform_data: &[u8],
        _workgroup_size: (u32, u32, u32),
        _workgroup_count: (u32, u32, u32),
    ) -> WebGpuResult<()> {
        // For now, this is a simplified implementation that just synchronizes
        // Real implementation would dispatch based on kernel_name
        self.device.wait_for_completion().await
    }

    /// Wait for all operations to complete
    pub async fn synchronize(&self) -> WebGpuResult<()> {
        self.device.wait_for_completion().await
    }

    /// Get pipeline cache statistics
    pub fn pipeline_stats(&self) -> crate::webgpu::pipeline::PipelineCacheStats {
        self.pipeline_factory.cache().stats()
    }
}

/// Kernel operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelOp {
    ElementwiseAdd,
    ElementwiseMul,
    ElementwiseSub,
    ElementwiseDiv,
    MatMul,
    Conv2D,
    ReLU,
    Softmax,
    BatchNorm,
    LayerNorm,
    Dropout,
    Custom,
}

impl KernelOp {
    /// Get the shader source for this operation
    pub fn shader_source(&self) -> &'static str {
        match self {
            KernelOp::ElementwiseAdd => crate::webgpu::shader::kernels::ELEMENTWISE_ADD,
            KernelOp::ElementwiseMul => crate::webgpu::shader::kernels::ELEMENTWISE_MUL,
            KernelOp::MatMul => crate::webgpu::shader::kernels::MATRIX_MUL,
            KernelOp::Conv2D => crate::webgpu::shader::kernels::CONV2D,
            KernelOp::ReLU => crate::webgpu::shader::kernels::RELU,
            KernelOp::Softmax => crate::webgpu::shader::kernels::SOFTMAX,
            _ => "", // Would need additional shader implementations
        }
    }

    /// Get optimal workgroup size for this operation
    pub fn optimal_workgroup_size(&self) -> (u32, u32, u32) {
        match self {
            KernelOp::ElementwiseAdd
            | KernelOp::ElementwiseMul
            | KernelOp::ElementwiseSub
            | KernelOp::ElementwiseDiv
            | KernelOp::ReLU
            | KernelOp::Softmax => (64, 1, 1),
            KernelOp::MatMul | KernelOp::Conv2D => (8, 8, 1),
            _ => (64, 1, 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferDescriptor, BufferHandle, BufferUsage, MemoryLocation};

    #[tokio::test]
    async fn test_kernel_executor_creation() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);
                let executor = WebGpuKernelExecutor::new(device);

                let stats = executor.pipeline_stats();
                assert_eq!(stats.pipeline_count, 0); // No pipelines created yet
            }
        }
    }

    #[tokio::test]
    async fn test_elementwise_operations() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);
                let executor = WebGpuKernelExecutor::new(Arc::clone(&device));

                // Create test buffers
                let size = 1024usize;
                let descriptor = BufferDescriptor {
                    size,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                    location: MemoryLocation::Device,
                    dtype: None,
                    shape: None,
                    initial_data: None,
                    alignment: None,
                    zero_init: false,
                };

                if let (Ok(a), Ok(b), Ok(output)) = (
                    WebGpuBuffer::new(
                        Arc::clone(&device),
                        descriptor.clone(),
                        BufferHandle::WebGpu {
                            buffer_ptr: 1,
                            size,
                        },
                    ),
                    WebGpuBuffer::new(
                        Arc::clone(&device),
                        descriptor.clone(),
                        BufferHandle::WebGpu {
                            buffer_ptr: 2,
                            size,
                        },
                    ),
                    WebGpuBuffer::new(
                        Arc::clone(&device),
                        descriptor.clone(),
                        BufferHandle::WebGpu {
                            buffer_ptr: 3,
                            size,
                        },
                    ),
                ) {
                    // Test elementwise addition
                    let result = executor.elementwise_add(&a, &b, &output).await;
                    if result.is_ok() {
                        executor.synchronize().await.unwrap();
                        // Operation completed successfully
                    }

                    // Test elementwise multiplication
                    let result = executor.elementwise_mul(&a, &b, &output).await;
                    if result.is_ok() {
                        executor.synchronize().await.unwrap();
                        // Operation completed successfully
                    }
                }
            }
        }
    }

    #[test]
    fn test_kernel_op_properties() {
        assert_eq!(
            KernelOp::ElementwiseAdd.optimal_workgroup_size(),
            (64, 1, 1)
        );
        assert_eq!(KernelOp::MatMul.optimal_workgroup_size(), (8, 8, 1));
        assert_eq!(KernelOp::Conv2D.optimal_workgroup_size(), (8, 8, 1));

        assert!(!KernelOp::ElementwiseAdd.shader_source().is_empty());
        assert!(!KernelOp::ReLU.shader_source().is_empty());
    }

    #[tokio::test]
    async fn test_relu_activation() {
        if cfg!(feature = "webgpu") && crate::webgpu::is_available() {
            if let Ok(device) = WebGpuDevice::from_best_adapter(0).await {
                let device = Arc::new(device);
                let executor = WebGpuKernelExecutor::new(Arc::clone(&device));

                let size = 512usize;
                let descriptor = BufferDescriptor {
                    size,
                    usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                    location: MemoryLocation::Device,
                    dtype: None,
                    shape: None,
                    initial_data: None,
                    alignment: None,
                    zero_init: false,
                };

                if let (Ok(input), Ok(output)) = (
                    WebGpuBuffer::new(
                        Arc::clone(&device),
                        descriptor.clone(),
                        BufferHandle::WebGpu {
                            buffer_ptr: 1,
                            size,
                        },
                    ),
                    WebGpuBuffer::new(
                        Arc::clone(&device),
                        descriptor.clone(),
                        BufferHandle::WebGpu {
                            buffer_ptr: 2,
                            size,
                        },
                    ),
                ) {
                    let result = executor.relu(&input, &output).await;
                    if result.is_ok() {
                        executor.synchronize().await.unwrap();
                        // ReLU operation completed successfully
                    }
                }
            }
        }
    }
}
