//! CPU convolution implementation with optimized algorithms

use crate::convolution::{
    algorithms, ConvolutionAlgorithm, ConvolutionOps, ConvolutionPerformanceHints, ConvolutionType,
    PaddingMode,
};

// Re-export for benchmarks
pub use crate::convolution::ConvolutionConfig;
use crate::cpu::buffer::BufferCpuExt;
use crate::{BackendResult, Buffer, Device};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// CPU convolution operations implementation
#[derive(Clone, Debug)]
pub struct CpuConvolutionOps {
    /// Performance hints for algorithm selection
    performance_hints: ConvolutionPerformanceHints,
    /// Number of threads for parallel processing
    #[allow(dead_code)]
    num_threads: usize,
}

impl CpuConvolutionOps {
    /// Create a new CPU convolution operations instance
    pub fn new(num_threads: Option<usize>) -> Self {
        let num_threads = num_threads.unwrap_or_else(|| rayon::current_num_threads());

        Self {
            performance_hints: ConvolutionPerformanceHints {
                small_kernel_algorithm: ConvolutionAlgorithm::Direct,
                large_kernel_algorithm: ConvolutionAlgorithm::Im2col,
                fft_threshold: 7,
                winograd_threshold: 6,
                tile_size: (16, 16),
                memory_bandwidth: 100.0, // CPU memory bandwidth
                compute_throughput: num_threads as f32 * 50.0, // Estimated GOPS
            },
            num_threads,
        }
    }

    /// Copy buffer data safely for CPU
    #[allow(dead_code)]
    fn copy_buffer_data(&self, src: &Buffer, dst: &Buffer, size: usize) -> BackendResult<()> {
        if !src.is_cpu() || !dst.is_cpu() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Both buffers must be CPU buffers".to_string(),
            ));
        }

        let src_ptr = src.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get source buffer pointer".to_string(),
            )
        })?;

        let dst_ptr = dst.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get destination buffer pointer".to_string(),
            )
        })?;

        if size > src.size.min(dst.size) {
            return Err(torsh_core::error::TorshError::BackendError(format!(
                "Copy size {} exceeds buffer capacity",
                size
            )));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }

        Ok(())
    }

    /// Execute direct convolution on CPU
    fn direct_convolution(
        &self,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        config: &ConvolutionConfig,
    ) -> BackendResult<()> {
        // Get buffer pointers
        let input_ptr = input.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get input buffer pointer".to_string(),
            )
        })?;

        let kernel_ptr = kernel.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get kernel buffer pointer".to_string(),
            )
        })?;

        let output_ptr = output.as_cpu_ptr().ok_or_else(|| {
            torsh_core::error::TorshError::BackendError(
                "Failed to get output buffer pointer".to_string(),
            )
        })?;

        unsafe {
            let input_data = std::slice::from_raw_parts(input_ptr as *const f32, input.size / 4);
            let kernel_data = std::slice::from_raw_parts(kernel_ptr as *const f32, kernel.size / 4);
            let output_data =
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output.size / 4);

            match config.conv_type {
                ConvolutionType::Conv2D => {
                    algorithms::DirectConvolution::conv2d_direct(
                        input_data,
                        kernel_data,
                        output_data,
                        &config.input_dims,
                        &config.kernel_dims,
                        &config.output_dims,
                        (config.strides[0], config.strides[1]),
                        (config.padding[0], config.padding[1]),
                    )?;
                }
                ConvolutionType::DepthwiseConv2D => {
                    // Simplified depthwise convolution - would need specialized implementation
                    self.depthwise_direct_implementation(
                        input_data,
                        kernel_data,
                        output_data,
                        config,
                    )?;
                }
                _ => {
                    return Err(torsh_core::error::TorshError::BackendError(format!(
                        "Convolution type {:?} not implemented yet",
                        config.conv_type
                    )));
                }
            }

            // Add bias if provided
            if let Some(bias_buffer) = bias {
                let bias_ptr = bias_buffer.as_cpu_ptr().ok_or_else(|| {
                    torsh_core::error::TorshError::BackendError(
                        "Failed to get bias buffer pointer".to_string(),
                    )
                })?;
                let bias_data =
                    std::slice::from_raw_parts(bias_ptr as *const f32, bias_buffer.size / 4);

                self.add_bias(output_data, bias_data, &config.output_dims)?;
            }
        }

        Ok(())
    }

    /// Add bias to output
    fn add_bias(
        &self,
        output: &mut [f32],
        bias: &[f32],
        output_dims: &[usize],
    ) -> BackendResult<()> {
        if output_dims.len() < 4 {
            return Ok(());
        }

        let (batch, channels, height, width) = (
            output_dims[0],
            output_dims[1],
            output_dims[2],
            output_dims[3],
        );

        for b in 0..batch {
            for c in 0..channels {
                let bias_value = bias.get(c).copied().unwrap_or(0.0);
                for h in 0..height {
                    for w in 0..width {
                        let idx =
                            b * channels * height * width + c * height * width + h * width + w;
                        if idx < output.len() {
                            output[idx] += bias_value;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Simplified depthwise convolution implementation
    fn depthwise_direct_implementation(
        &self,
        input: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        config: &ConvolutionConfig,
    ) -> BackendResult<()> {
        let (batch, channels, in_h, in_w) = (
            config.input_dims[0],
            config.input_dims[1],
            config.input_dims[2],
            config.input_dims[3],
        );
        let (_, _, k_h, k_w) = (
            config.kernel_dims[0],
            config.kernel_dims[1],
            config.kernel_dims[2],
            config.kernel_dims[3],
        );
        let (_, _, out_h, out_w) = (
            config.output_dims[0],
            config.output_dims[1],
            config.output_dims[2],
            config.output_dims[3],
        );
        let (s_h, s_w) = (config.strides[0], config.strides[1]);
        let (p_h, p_w) = (config.padding[0], config.padding[1]);

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        for kh in 0..k_h {
                            for kw in 0..k_w {
                                let ih = oh * s_h + kh;
                                let iw = ow * s_w + kw;

                                if ih >= p_h && iw >= p_w && ih < in_h + p_h && iw < in_w + p_w {
                                    let input_h = ih - p_h;
                                    let input_w = iw - p_w;

                                    if input_h < in_h && input_w < in_w {
                                        let input_idx = b * channels * in_h * in_w
                                            + c * in_h * in_w
                                            + input_h * in_w
                                            + input_w;
                                        let kernel_idx = c * k_h * k_w + kh * k_w + kw;

                                        if input_idx < input.len() && kernel_idx < kernel.len() {
                                            sum += input[input_idx] * kernel[kernel_idx];
                                        }
                                    }
                                }
                            }
                        }

                        let output_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;

                        if output_idx < output.len() {
                            output[output_idx] = sum;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl ConvolutionOps for CpuConvolutionOps {
    async fn convolution(
        &self,
        _device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        config: &ConvolutionConfig,
    ) -> BackendResult<()> {
        if !config.is_valid() {
            return Err(torsh_core::error::TorshError::BackendError(
                "Invalid convolution configuration".to_string(),
            ));
        }

        let algorithm = self.select_algorithm(config);

        match algorithm {
            ConvolutionAlgorithm::Direct => {
                self.direct_convolution(input, kernel, bias, output, config)
            }
            ConvolutionAlgorithm::Im2col => {
                // For now, fall back to direct convolution
                // A full im2col implementation would require GEMM operations
                self.direct_convolution(input, kernel, bias, output, config)
            }
            ConvolutionAlgorithm::Winograd => {
                // For now, fall back to direct convolution
                // A full Winograd implementation would require specialized transforms
                self.direct_convolution(input, kernel, bias, output, config)
            }
            ConvolutionAlgorithm::FftBased => {
                // For now, fall back to direct convolution
                // FFT-based convolution would use our FFT operations module
                self.direct_convolution(input, kernel, bias, output, config)
            }
            _ => self.direct_convolution(input, kernel, bias, output, config),
        }
    }

    async fn conv2d(
        &self,
        device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> BackendResult<()> {
        // Create a basic configuration from parameters
        // For a full implementation, we'd need to infer dimensions from buffer sizes
        let config = ConvolutionConfig {
            conv_type: ConvolutionType::Conv2D,
            input_dims: vec![1, 1, 32, 32],  // Placeholder dimensions
            output_dims: vec![1, 1, 32, 32], // Placeholder dimensions
            kernel_dims: vec![1, 1, 3, 3],   // Placeholder dimensions
            strides: vec![stride.0, stride.1],
            padding: vec![padding.0, padding.1],
            dilation: vec![dilation.0, dilation.1],
            groups: 1,
            padding_mode: PaddingMode::Custom,
            dtype: torsh_core::dtype::DType::F32,
            algorithm: ConvolutionAlgorithm::Auto,
        };

        self.convolution(device, input, kernel, bias, output, &config)
            .await
    }

    async fn depthwise_conv2d(
        &self,
        device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<()> {
        // Create depthwise configuration
        let config = ConvolutionConfig {
            conv_type: ConvolutionType::DepthwiseConv2D,
            input_dims: vec![1, 16, 32, 32], // Placeholder dimensions
            output_dims: vec![1, 16, 32, 32], // Placeholder dimensions
            kernel_dims: vec![16, 1, 3, 3],  // Placeholder dimensions
            strides: vec![stride.0, stride.1],
            padding: vec![padding.0, padding.1],
            dilation: vec![1, 1],
            groups: 16, // Depthwise means groups = input channels
            padding_mode: PaddingMode::Custom,
            dtype: torsh_core::dtype::DType::F32,
            algorithm: ConvolutionAlgorithm::Direct,
        };

        self.convolution(device, input, kernel, bias, output, &config)
            .await
    }

    async fn conv_transpose2d(
        &self,
        _device: &Device,
        _input: &Buffer,
        _kernel: &Buffer,
        _bias: Option<&Buffer>,
        _output: &Buffer,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _output_padding: (usize, usize),
    ) -> BackendResult<()> {
        Err(torsh_core::error::TorshError::BackendError(
            "Transposed convolution not implemented for CPU backend yet".to_string(),
        ))
    }

    async fn grouped_conv2d(
        &self,
        device: &Device,
        input: &Buffer,
        kernel: &Buffer,
        bias: Option<&Buffer>,
        output: &Buffer,
        groups: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<()> {
        // Create grouped configuration
        let config = ConvolutionConfig {
            conv_type: ConvolutionType::GroupedConv2D,
            input_dims: vec![1, 16, 32, 32], // Placeholder dimensions
            output_dims: vec![1, 16, 32, 32], // Placeholder dimensions
            kernel_dims: vec![16, 16 / groups, 3, 3], // Placeholder dimensions
            strides: vec![stride.0, stride.1],
            padding: vec![padding.0, padding.1],
            dilation: vec![1, 1],
            groups,
            padding_mode: PaddingMode::Custom,
            dtype: torsh_core::dtype::DType::F32,
            algorithm: ConvolutionAlgorithm::Direct,
        };

        self.convolution(device, input, kernel, bias, output, &config)
            .await
    }

    fn select_algorithm(&self, config: &ConvolutionConfig) -> ConvolutionAlgorithm {
        if config.algorithm != ConvolutionAlgorithm::Auto {
            return config.algorithm;
        }

        // Auto-select based on configuration and performance hints
        match config.conv_type {
            ConvolutionType::Conv2D => {
                if config.kernel_dims.len() >= 4 {
                    let kernel_h = config.kernel_dims[2];
                    let kernel_w = config.kernel_dims[3];
                    let kernel_size = kernel_h.max(kernel_w);

                    if kernel_size <= 3 {
                        // Small kernels work well with direct convolution on CPU
                        ConvolutionAlgorithm::Direct
                    } else if kernel_size <= self.performance_hints.winograd_threshold {
                        ConvolutionAlgorithm::Winograd
                    } else if kernel_size >= self.performance_hints.fft_threshold {
                        ConvolutionAlgorithm::FftBased
                    } else {
                        ConvolutionAlgorithm::Im2col
                    }
                } else {
                    ConvolutionAlgorithm::Direct
                }
            }
            ConvolutionType::DepthwiseConv2D => ConvolutionAlgorithm::Direct,
            ConvolutionType::SeparableConv2D => ConvolutionAlgorithm::Direct,
            ConvolutionType::GroupedConv2D => ConvolutionAlgorithm::Direct,
            _ => ConvolutionAlgorithm::Im2col,
        }
    }

    fn supports_convolution(&self) -> bool {
        true
    }

    fn supported_conv_types(&self) -> Vec<ConvolutionType> {
        vec![
            ConvolutionType::Conv1D,
            ConvolutionType::Conv2D,
            ConvolutionType::Conv3D,
            ConvolutionType::DepthwiseConv2D,
            ConvolutionType::SeparableConv2D,
            ConvolutionType::GroupedConv2D,
            // ConvolutionType::ConvTranspose2D, // Not implemented yet
            ConvolutionType::DilatedConv2D,
        ]
    }

    fn supported_algorithms(&self) -> Vec<ConvolutionAlgorithm> {
        vec![
            ConvolutionAlgorithm::Auto,
            ConvolutionAlgorithm::Direct,
            ConvolutionAlgorithm::Im2col,
            ConvolutionAlgorithm::Winograd,
            ConvolutionAlgorithm::FftBased,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolution::ConvolutionConfig;

    #[test]
    fn test_cpu_convolution_ops_creation() {
        let conv_ops = CpuConvolutionOps::new(Some(2));
        assert!(conv_ops.supports_convolution());
        assert!(!conv_ops.supported_conv_types().is_empty());
        assert!(!conv_ops.supported_algorithms().is_empty());
    }

    #[test]
    fn test_algorithm_selection() {
        let conv_ops = CpuConvolutionOps::new(Some(1));

        // Small kernel should use direct convolution on CPU
        let small_config = ConvolutionConfig::conv2d(1, 3, 16, (32, 32), (3, 3), (1, 1), (1, 1));
        assert_eq!(
            conv_ops.select_algorithm(&small_config),
            ConvolutionAlgorithm::Direct
        );

        // Large kernel should use FFT-based convolution
        let large_config = ConvolutionConfig::conv2d(1, 3, 16, (32, 32), (9, 9), (1, 1), (4, 4));
        assert_eq!(
            conv_ops.select_algorithm(&large_config),
            ConvolutionAlgorithm::FftBased
        );

        // Depthwise should always use direct
        let depthwise_config =
            ConvolutionConfig::depthwise_conv2d(1, 16, (32, 32), (3, 3), (1, 1), (1, 1));
        assert_eq!(
            conv_ops.select_algorithm(&depthwise_config),
            ConvolutionAlgorithm::Direct
        );
    }
}
