//! Metal shader kernels for tensor operations

use metal::{ComputeCommandEncoderRef, ComputePipelineState, Device, Library, MTLSize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::metal::error::{metal_errors, Result};

pub mod shaders;

/// Kernel manager for Metal compute kernels
pub struct KernelManager {
    device: Device,
    library: Library,
    pipelines: Arc<Mutex<HashMap<String, ComputePipelineState>>>,
}

impl KernelManager {
    /// Create a new kernel manager
    pub fn new(device: &Device) -> Result<Self> {
        // Compile the Metal shader library
        let source = shaders::SHADER_SOURCE;
        let options = metal::CompileOptions::new();

        let library = device
            .new_library_with_source(source, &options)
            .map_err(|e| metal_errors::shader_compilation_error(e.to_string(), None))?;

        Ok(Self {
            device: device.clone(),
            library,
            pipelines: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get or create a compute pipeline for a kernel
    pub fn get_pipeline(&self, kernel_name: &str) -> Result<ComputePipelineState> {
        let mut pipelines = self.pipelines.lock().map_err(|e| {
            metal_errors::metal_api_error(format!("Failed to lock pipelines: {}", e), None)
        })?;

        if let Some(pipeline) = pipelines.get(kernel_name) {
            return Ok(pipeline.clone());
        }

        // Create new pipeline
        let function = self.library.get_function(kernel_name, None).map_err(|_| {
            metal_errors::kernel_execution_error(
                format!("Kernel '{}' not found in library", kernel_name),
                None,
            )
        })?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                metal_errors::shader_compilation_error(
                    format!("Failed to create pipeline for '{}': {}", kernel_name, e),
                    None,
                )
            })?;

        pipelines.insert(kernel_name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Dispatch a 1D compute kernel
    pub fn dispatch_1d(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kernel_name: &str,
        global_size: usize,
    ) -> Result<()> {
        let pipeline = self.get_pipeline(kernel_name)?;
        encoder.set_compute_pipeline_state(&pipeline);

        let thread_group_size = pipeline.thread_execution_width() as usize;
        let thread_groups = global_size.div_ceil(thread_group_size);

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups as u64, 1, 1),
            MTLSize::new(thread_group_size as u64, 1, 1),
        );

        Ok(())
    }

    /// Dispatch a 2D compute kernel
    pub fn dispatch_2d(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kernel_name: &str,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let pipeline = self.get_pipeline(kernel_name)?;
        encoder.set_compute_pipeline_state(&pipeline);

        let w = pipeline.thread_execution_width() as usize;
        let h = (pipeline.max_total_threads_per_threadgroup() as usize) / w;

        let thread_groups_x = width.div_ceil(w);
        let thread_groups_y = height.div_ceil(h);

        encoder.dispatch_thread_groups(
            MTLSize::new(thread_groups_x as u64, thread_groups_y as u64, 1),
            MTLSize::new(w as u64, h as u64, 1),
        );

        Ok(())
    }

    /// Dispatch a 3D compute kernel
    pub fn dispatch_3d(
        &self,
        encoder: &ComputeCommandEncoderRef,
        kernel_name: &str,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Result<()> {
        let pipeline = self.get_pipeline(kernel_name)?;
        encoder.set_compute_pipeline_state(&pipeline);

        // Use reasonable thread group sizes
        let tg_width = 8;
        let tg_height = 8;
        let tg_depth = 4;

        let thread_groups_x = width.div_ceil(tg_width);
        let thread_groups_y = height.div_ceil(tg_height);
        let thread_groups_z = depth.div_ceil(tg_depth);

        encoder.dispatch_thread_groups(
            MTLSize::new(
                thread_groups_x as u64,
                thread_groups_y as u64,
                thread_groups_z as u64,
            ),
            MTLSize::new(tg_width as u64, tg_height as u64, tg_depth as u64),
        );

        Ok(())
    }
}

/// Kernel names
pub mod kernel_names {
    // Unary operations
    pub const UNARY_NEG_F32: &str = "unary_neg_f32";
    pub const UNARY_EXP_F32: &str = "unary_exp_f32";
    pub const UNARY_LOG_F32: &str = "unary_log_f32";
    pub const UNARY_SQRT_F32: &str = "unary_sqrt_f32";
    pub const UNARY_TANH_F32: &str = "unary_tanh_f32";
    pub const UNARY_RELU_F32: &str = "unary_relu_f32";
    pub const UNARY_ABS_F32: &str = "unary_abs_f32";
    pub const UNARY_SIN_F32: &str = "unary_sin_f32";
    pub const UNARY_COS_F32: &str = "unary_cos_f32";
    pub const UNARY_SIGMOID_F32: &str = "unary_sigmoid_f32";
    pub const UNARY_GELU_F32: &str = "unary_gelu_f32";

    // Binary operations
    pub const BINARY_ADD_F32: &str = "binary_add_f32";
    pub const BINARY_SUB_F32: &str = "binary_sub_f32";
    pub const BINARY_MUL_F32: &str = "binary_mul_f32";
    pub const BINARY_DIV_F32: &str = "binary_div_f32";
    pub const BINARY_POW_F32: &str = "binary_pow_f32";
    pub const BINARY_MAX_F32: &str = "binary_max_f32";
    pub const BINARY_MIN_F32: &str = "binary_min_f32";

    // Reduction operations
    pub const REDUCE_SUM_F32: &str = "reduce_sum_f32";
    pub const REDUCE_MEAN_F32: &str = "reduce_mean_f32";
    pub const REDUCE_MAX_F32: &str = "reduce_max_f32";
    pub const REDUCE_MIN_F32: &str = "reduce_min_f32";

    // Softmax
    pub const SOFTMAX_F32: &str = "softmax_f32";

    // Matrix operations
    pub const MATMUL_F32: &str = "matmul_f32";
    pub const TRANSPOSE_F32: &str = "transpose_f32";

    // Convolution
    pub const CONV2D_F32: &str = "conv2d_f32";
    pub const CONV2D_BACKWARD_F32: &str = "conv2d_backward_f32";

    // Pooling
    pub const MAXPOOL2D_F32: &str = "maxpool2d_f32";
    pub const AVGPOOL2D_F32: &str = "avgpool2d_f32";
}
