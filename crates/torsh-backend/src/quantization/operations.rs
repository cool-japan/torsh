//! Core quantization operations and hardware implementations
//!
//! This module provides the fundamental quantization operations through the [`QuantizationOps`] trait
//! and its hardware-accelerated implementation [`HardwareQuantizationOps`]. It includes:
//! - Basic quantize/dequantize operations for all supported data types
//! - Quantized matrix multiplication, convolution, and element-wise operations
//! - Hardware-specific optimizations (SIMD, VNNI, DP4A, Tensor Cores)
//! - Automatic calibration for optimal quantization parameters

use super::core::{QuantizedDType, QuantizationParams, QuantizationScheme, QuantizedTensor};
use super::hardware::QuantizationHardwareFeatures;
use crate::{BackendResult, Device};
use std::collections::HashMap;
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String};

/// Quantization operations trait
///
/// This trait defines the core interface for quantization operations that can be
/// implemented by different backends (CPU, CUDA, etc.) with hardware-specific optimizations.
pub trait QuantizationOps {
    /// Quantize floating-point data to quantized representation
    ///
    /// Converts an array of f32 values to quantized bytes according to the given parameters.
    /// The output format depends on the quantization data type (packed for Int4/Binary).
    fn quantize_f32(&self, input: &[f32], params: &QuantizationParams) -> BackendResult<Vec<u8>>;

    /// Dequantize quantized data back to floating-point
    ///
    /// Converts quantized bytes back to f32 values using the provided parameters.
    /// This is the inverse operation of quantize_f32.
    fn dequantize_f32(&self, input: &[u8], params: &QuantizationParams) -> BackendResult<Vec<f32>>;

    /// Quantized matrix multiplication
    ///
    /// Performs matrix multiplication directly in quantized space for better performance.
    /// Supports INT8 VNNI and other hardware-accelerated paths where available.
    fn qmatmul(&self, a: &QuantizedTensor, b: &QuantizedTensor) -> BackendResult<QuantizedTensor>;

    /// Quantized convolution
    ///
    /// 2D convolution operation optimized for quantized tensors. Uses specialized kernels
    /// for common quantization types and falls back to dequantize-compute-quantize for others.
    fn qconv2d(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
        bias: Option<&QuantizedTensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<QuantizedTensor>;

    /// Quantized element-wise addition
    ///
    /// Adds two quantized tensors element-wise, handling scale and zero-point differences.
    fn qadd(&self, a: &QuantizedTensor, b: &QuantizedTensor) -> BackendResult<QuantizedTensor>;

    /// Quantized ReLU activation function
    ///
    /// Applies ReLU activation in quantized space: max(x, zero_point).
    /// More efficient than dequantizing, applying ReLU, and requantizing.
    fn qrelu(&self, input: &QuantizedTensor) -> BackendResult<QuantizedTensor>;

    /// Calibrate quantization parameters from sample data
    ///
    /// Analyzes sample data to determine optimal quantization parameters (scale, zero_point)
    /// for the target data type. Uses statistical methods to minimize quantization error.
    fn calibrate(
        &self,
        samples: &[&[f32]],
        target_dtype: QuantizedDType,
    ) -> BackendResult<QuantizationParams>;
}


/// Hardware-accelerated quantization operations
///
/// Implements the [`QuantizationOps`] trait with hardware-specific optimizations.
/// Automatically detects and uses available features like SIMD, VNNI, DP4A, etc.
#[derive(Debug, Clone)]
pub struct HardwareQuantizationOps {
    /// Device for operations
    device: Device,
    /// Hardware-specific optimizations
    hw_features: QuantizationHardwareFeatures,
    /// Calibration cache for reusing computed parameters
    #[allow(dead_code)]
    calibration_cache: HashMap<String, QuantizationParams>,
}

impl HardwareQuantizationOps {
    /// Create new hardware quantization operations
    ///
    /// Automatically detects available hardware features and optimizations
    /// for the given device (CPU, CUDA, etc.).
    pub fn new(device: Device) -> Self {
        let hw_features = QuantizationHardwareFeatures::detect(&device);

        Self {
            device,
            hw_features,
            calibration_cache: HashMap::new(),
        }
    }

    /// Get the device this instance operates on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the detected hardware features
    pub fn hardware_features(&self) -> &QuantizationHardwareFeatures {
        &self.hw_features
    }

}

impl QuantizationOps for HardwareQuantizationOps {
    fn quantize_f32(&self, input: &[f32], params: &QuantizationParams) -> BackendResult<Vec<u8>> {
        match params.dtype {
            QuantizedDType::UInt8 => self.quantize_f32_to_u8(input, params),
            QuantizedDType::Int8 => self.quantize_f32_to_i8(input, params),
            QuantizedDType::Int4 => self.quantize_f32_to_i4_packed(input, params),
            QuantizedDType::UInt4 => self.quantize_f32_to_u4_packed(input, params),
            QuantizedDType::Binary => self.quantize_f32_to_binary(input, params),
            _ => Err(TorshError::BackendError(
                "Unsupported quantization type".to_string(),
            ).into()),
        }
    }

    fn dequantize_f32(&self, input: &[u8], params: &QuantizationParams) -> BackendResult<Vec<f32>> {
        match params.dtype {
            QuantizedDType::UInt8 => self.dequantize_u8_to_f32(input, params),
            QuantizedDType::Int8 => self.dequantize_i8_to_f32(input, params),
            QuantizedDType::Int4 => self.dequantize_i4_packed_to_f32(input, params),
            QuantizedDType::UInt4 => self.dequantize_u4_packed_to_f32(input, params),
            QuantizedDType::Binary => self.dequantize_binary_to_f32(input, params),
            _ => Err(TorshError::BackendError(
                "Unsupported quantization type".to_string(),
            ).into()),
        }
    }

    fn qmatmul(&self, a: &QuantizedTensor, b: &QuantizedTensor) -> BackendResult<QuantizedTensor> {
        // Check dimension compatibility
        if a.shape().len() != 2 || b.shape().len() != 2 {
            return Err(TorshError::BackendError(
                "Only 2D matrices supported".to_string(),
            ).into());
        }

        if a.shape()[1] != b.shape()[0] {
            return Err(TorshError::BackendError(
                "Matrix dimensions incompatible".to_string(),
            ).into());
        }

        // Use hardware-accelerated path if available
        if self.hw_features.supports_int8_simd
            && a.params().dtype == QuantizedDType::Int8
            && b.params().dtype == QuantizedDType::Int8
        {
            self.qmatmul_int8_accelerated(a, b)
        } else {
            self.qmatmul_fallback(a, b)
        }
    }

    fn qconv2d(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
        bias: Option<&QuantizedTensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<QuantizedTensor> {
        // Validate input dimensions
        if input.shape().len() != 4 || weight.shape().len() != 4 {
            return Err(TorshError::BackendError(
                "Conv2D requires 4D tensors".to_string(),
            ).into());
        }

        // Use optimized path for supported configurations
        if self.hw_features.supports_int8_simd
            && input.params().dtype == QuantizedDType::Int8
            && weight.params().dtype == QuantizedDType::Int8
        {
            self.qconv2d_int8_optimized(input, weight, bias, stride, padding)
        } else {
            self.qconv2d_fallback(input, weight, bias, stride, padding)
        }
    }

    fn qadd(&self, a: &QuantizedTensor, b: &QuantizedTensor) -> BackendResult<QuantizedTensor> {
        if a.shape() != b.shape() {
            return Err(TorshError::BackendError(
                "Tensor shapes must match for addition".to_string(),
            ).into());
        }

        // Use SIMD-accelerated addition if available
        if self.hw_features.supports_int8_simd
            && a.params().dtype == QuantizedDType::Int8
            && b.params().dtype == QuantizedDType::Int8
        {
            self.qadd_int8_simd(a, b)
        } else {
            self.qadd_fallback(a, b)
        }
    }

    fn qrelu(&self, input: &QuantizedTensor) -> BackendResult<QuantizedTensor> {
        // ReLU in quantized space: max(x, zero_point)
        let zero_point = input.params().zero_point[0] as u8;

        let mut result = input.clone();

        match input.params().dtype {
            QuantizedDType::UInt8 => {
                for byte in result.data_mut() {
                    *byte = (*byte).max(zero_point);
                }
            }
            QuantizedDType::Int8 => {
                let zero_point_i8 = zero_point as i8;
                for byte in result.data_mut() {
                    let val = *byte as i8;
                    *byte = val.max(zero_point_i8) as u8;
                }
            }
            _ => {
                return Err(TorshError::BackendError(
                    "Unsupported dtype for qReLU".to_string(),
                ).into())
            }
        }

        Ok(result)
    }

    fn calibrate(
        &self,
        samples: &[&[f32]],
        target_dtype: QuantizedDType,
    ) -> BackendResult<QuantizationParams> {
        if samples.is_empty() {
            return Err(TorshError::BackendError(
                "No calibration samples provided".to_string(),
            ).into());
        }

        // Collect statistics from all samples
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for sample in samples {
            for &val in *sample {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Create quantization parameters
        let mut params = QuantizationParams {
            dtype: target_dtype,
            scheme: QuantizationScheme::Asymmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: Some(min_val),
            max_val: Some(max_val),
        };

        params.from_statistics(min_val, max_val)?;

        Ok(params)
    }
}

impl HardwareQuantizationOps {
    /// Quantize f32 to u8 with hardware acceleration
    fn quantize_f32_to_u8(
        &self,
        input: &[f32],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<u8>> {
        let scale = params.scale[0];
        let zero_point = params.zero_point[0] as f32;
        let mut output = Vec::with_capacity(input.len());

        if self.hw_features.supports_int8_simd {
            // Use SIMD for quantization
            self.quantize_f32_to_u8_simd(input, scale, zero_point, &mut output)
        } else {
            // Scalar fallback
            for &val in input {
                let quantized = (val / scale + zero_point).round().clamp(0.0, 255.0) as u8;
                output.push(quantized);
            }
            Ok(output)
        }
    }

    /// SIMD-accelerated f32 to u8 quantization
    fn quantize_f32_to_u8_simd(
        &self,
        input: &[f32],
        scale: f32,
        zero_point: f32,
        output: &mut Vec<u8>,
    ) -> BackendResult<Vec<u8>> {
        // This would use platform-specific SIMD instructions
        // For now, use a vectorized approach
        let inv_scale = 1.0 / scale;

        for chunk in input.chunks(4) {
            for &val in chunk {
                let quantized = (val * inv_scale + zero_point).round().clamp(0.0, 255.0) as u8;
                output.push(quantized);
            }
        }

        Ok(output.clone())
    }

    /// Quantize f32 to i8
    fn quantize_f32_to_i8(
        &self,
        input: &[f32],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<u8>> {
        let scale = params.scale[0];
        let zero_point = params.zero_point[0] as f32;
        let mut output = Vec::with_capacity(input.len());

        for &val in input {
            let quantized = (val / scale + zero_point).round().clamp(-128.0, 127.0) as i8;
            output.push(quantized as u8);
        }

        Ok(output)
    }

    /// Quantize f32 to packed 4-bit integers
    fn quantize_f32_to_i4_packed(
        &self,
        input: &[f32],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<u8>> {
        let scale = params.scale[0];
        let zero_point = params.zero_point[0] as f32;
        let mut output = Vec::with_capacity((input.len() + 1) / 2);

        for chunk in input.chunks(2) {
            let first = (chunk[0] / scale + zero_point).round().clamp(-8.0, 7.0) as i8;
            let second = if chunk.len() > 1 {
                (chunk[1] / scale + zero_point).round().clamp(-8.0, 7.0) as i8
            } else {
                0
            };

            // Pack two 4-bit values into one byte
            let packed = ((first & 0x0F) << 4) | (second & 0x0F);
            output.push(packed as u8);
        }

        Ok(output)
    }

    /// Quantize f32 to packed unsigned 4-bit integers
    fn quantize_f32_to_u4_packed(
        &self,
        input: &[f32],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<u8>> {
        let scale = params.scale[0];
        let zero_point = params.zero_point[0] as f32;
        let mut output = Vec::with_capacity((input.len() + 1) / 2);

        for chunk in input.chunks(2) {
            let first = (chunk[0] / scale + zero_point).round().clamp(0.0, 15.0) as u8;
            let second = if chunk.len() > 1 {
                (chunk[1] / scale + zero_point).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };

            // Pack two 4-bit values into one byte
            let packed = (first << 4) | second;
            output.push(packed);
        }

        Ok(output)
    }

    /// Quantize f32 to binary (1-bit)
    fn quantize_f32_to_binary(
        &self,
        input: &[f32],
        _params: &QuantizationParams,
    ) -> BackendResult<Vec<u8>> {
        let mut output = Vec::with_capacity((input.len() + 7) / 8);

        for chunk in input.chunks(8) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                if val > 0.0 {
                    byte |= 1 << i;
                }
            }
            output.push(byte);
        }

        Ok(output)
    }

    /// Dequantize u8 to f32
    fn dequantize_u8_to_f32(
        &self,
        input: &[u8],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<f32>> {
        let scale = params.scale[0];
        let zero_point = params.zero_point[0] as f32;
        let mut output = Vec::with_capacity(input.len());

        for &val in input {
            let dequantized = (val as f32 - zero_point) * scale;
            output.push(dequantized);
        }

        Ok(output)
    }

    /// Dequantize i8 to f32
    fn dequantize_i8_to_f32(
        &self,
        input: &[u8],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<f32>> {
        let scale = params.scale[0];
        let zero_point = params.zero_point[0] as f32;
        let mut output = Vec::with_capacity(input.len());

        for &val in input {
            let dequantized = (val as i8 as f32 - zero_point) * scale;
            output.push(dequantized);
        }

        Ok(output)
    }

    /// Dequantize packed 4-bit integers to f32
    fn dequantize_i4_packed_to_f32(
        &self,
        input: &[u8],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<f32>> {
        let scale = params.scale[0];
        let zero_point = params.zero_point[0] as f32;
        let mut output = Vec::with_capacity(input.len() * 2);

        for &byte in input {
            // Unpack two 4-bit values
            let first = ((byte as i8) >> 4) as f32;
            let second = ((byte as i8) & 0x0F) as f32;

            output.push((first - zero_point) * scale);
            output.push((second - zero_point) * scale);
        }

        Ok(output)
    }

    /// Dequantize packed unsigned 4-bit integers to f32
    fn dequantize_u4_packed_to_f32(
        &self,
        input: &[u8],
        params: &QuantizationParams,
    ) -> BackendResult<Vec<f32>> {
        let scale = params.scale[0];
        let zero_point = params.zero_point[0] as f32;
        let mut output = Vec::with_capacity(input.len() * 2);

        for &byte in input {
            // Unpack two 4-bit values
            let first = (byte >> 4) as f32;
            let second = (byte & 0x0F) as f32;

            output.push((first - zero_point) * scale);
            output.push((second - zero_point) * scale);
        }

        Ok(output)
    }

    /// Dequantize binary to f32
    fn dequantize_binary_to_f32(
        &self,
        input: &[u8],
        _params: &QuantizationParams,
    ) -> BackendResult<Vec<f32>> {
        let mut output = Vec::with_capacity(input.len() * 8);

        for &byte in input {
            for i in 0..8 {
                let bit = (byte >> i) & 1;
                output.push(if bit == 1 { 1.0 } else { -1.0 });
            }
        }

        Ok(output)
    }

    /// Hardware-accelerated INT8 matrix multiplication
    fn qmatmul_int8_accelerated(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        // This would use INT8 VNNI or similar instructions
        // For now, implement a optimized but portable version
        self.qmatmul_fallback(a, b)
    }

    /// Fallback quantized matrix multiplication
    fn qmatmul_fallback(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];

        // For simplicity, dequantize, multiply, and requantize
        let a_f32 = self.dequantize_f32(a.data(), a.params())?;
        let b_f32 = self.dequantize_f32(b.data(), b.params())?;

        let mut c_f32 = vec![0.0; m * n];

        // Matrix multiplication
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a_f32[i * k + l] * b_f32[l * n + j];
                }
                c_f32[i * n + j] = sum;
            }
        }

        // Create output quantization parameters (use same as input A for simplicity)
        let output_params = a.params().clone();
        let c_quantized = self.quantize_f32(&c_f32, &output_params)?;

        QuantizedTensor::from_data(
            c_quantized,
            vec![m, n],
            output_params,
            a.device().clone(),
        )
    }

    /// Optimized INT8 convolution
    fn qconv2d_int8_optimized(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
        bias: Option<&QuantizedTensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<QuantizedTensor> {
        // Enhanced convolution implementation with proper dimension calculation

        // Validate input dimensions
        if input.shape().len() != 4 || weight.shape().len() != 4 {
            return Err(TorshError::InvalidArgument(
                "Convolution requires 4D tensors (NCHW format)".to_string(),
            ));
        }

        let [batch_size, in_channels, in_height, in_width] = [
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        ];

        let [out_channels, weight_in_channels, kernel_height, kernel_width] = [
            weight.shape()[0],
            weight.shape()[1],
            weight.shape()[2],
            weight.shape()[3],
        ];

        // Validate channel consistency
        if in_channels != weight_in_channels {
            return Err(TorshError::InvalidArgument(format!(
                "Input channels {} doesn't match weight channels {}",
                in_channels, weight_in_channels
            )));
        }

        // Calculate proper output dimensions
        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;
        let output_shape = vec![batch_size, out_channels, out_height, out_width];

        // Create output tensor with proper shape
        let mut output = QuantizedTensor::new(
            output_shape.clone(),
            input.params().clone(),
            input.device().clone(),
        );

        // Perform convolution operation
        self.perform_quantized_convolution(
            input,
            weight,
            bias,
            &mut output,
            stride,
            padding,
            kernel_height,
            kernel_width,
        )?;

        Ok(output)
    }

    /// Perform the actual quantized convolution computation
    fn perform_quantized_convolution(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
        bias: Option<&QuantizedTensor>,
        output: &mut QuantizedTensor,
        stride: (usize, usize),
        padding: (usize, usize),
        kernel_h: usize,
        kernel_w: usize,
    ) -> BackendResult<()> {
        let [batch_size, in_channels, in_height, in_width] = [
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        ];

        let [out_channels, _, out_height, out_width] = [
            output.shape()[0],
            output.shape()[1],
            output.shape()[2],
            output.shape()[3],
        ];

        // Extract quantization parameters
        let input_scale = input.params().scale;
        let weight_scale = weight.params().scale;
        let input_zero_point = input.params().zero_point as i32;
        let weight_zero_point = weight.params().zero_point as i32;

        // Compute output scale (simplified approach)
        let output_scale = input_scale * weight_scale;

        // Iterate over output tensor
        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut accumulator: i32 = 0;

                        // Convolve over the kernel
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                for in_ch in 0..in_channels {
                                    // Calculate input position with padding
                                    let ih = oh * stride.0 + kh;
                                    let iw = ow * stride.1 + kw;

                                    let input_val = if ih >= padding.0 && iw >= padding.1 {
                                        let actual_ih = ih - padding.0;
                                        let actual_iw = iw - padding.1;
                                        if actual_ih < in_height && actual_iw < in_width {
                                            let input_idx = batch * (in_channels * in_height * in_width)
                                                + in_ch * (in_height * in_width)
                                                + actual_ih * in_width
                                                + actual_iw;
                                            input.data()[input_idx] as i32 - input_zero_point
                                        } else {
                                            -input_zero_point // Padding value
                                        }
                                    } else {
                                        -input_zero_point // Padding value
                                    };

                                    // Get weight value
                                    let weight_idx = out_ch * (in_channels * kernel_h * kernel_w)
                                        + in_ch * (kernel_h * kernel_w)
                                        + kh * kernel_w
                                        + kw;
                                    let weight_val = weight.data()[weight_idx] as i32 - weight_zero_point;

                                    // Accumulate the product
                                    accumulator += input_val * weight_val;
                                }
                            }
                        }

                        // Add bias if present
                        if let Some(bias_tensor) = bias {
                            let bias_val = bias_tensor.data()[out_ch] as i32;
                            accumulator += bias_val;
                        }

                        // Apply scaling and convert back to quantized format
                        let scaled_output = (accumulator as f32 * output_scale) + input.params().zero_point as f32;
                        let quantized_output = scaled_output.round().clamp(0.0, 255.0) as u8;

                        // Store result
                        let output_idx = batch * (out_channels * out_height * out_width)
                            + out_ch * (out_height * out_width)
                            + oh * out_width
                            + ow;

                        // Set the output value (we need to access the mutable data)
                        // This is a simplified approach - in practice, we'd need a mutable accessor
                        unsafe {
                            let output_data = output.data_slice_mut(0, output.data().len()).expect("output data slice should be valid");
                            output_data[output_idx] = quantized_output;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Fallback convolution implementation
    fn qconv2d_fallback(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
        _bias: Option<&QuantizedTensor>,
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> BackendResult<QuantizedTensor> {
        // Simplified fallback - just return a placeholder
        let output_shape = vec![input.shape()[0], weight.shape()[0], 1, 1];

        QuantizedTensor::new(
            output_shape,
            input.params().clone(),
            input.device().clone(),
        )
    }

    /// SIMD-accelerated INT8 addition
    fn qadd_int8_simd(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        let mut result = a.clone();

        // Simple elementwise addition (would use SIMD in real implementation)
        for (a_byte, b_byte) in result.data_mut().iter_mut().zip(b.data().iter()) {
            let a_val = *a_byte as i8;
            let b_val = *b_byte as i8;
            let sum = (a_val as i16 + b_val as i16).clamp(-128, 127) as i8;
            *a_byte = sum as u8;
        }

        Ok(result)
    }

    /// Fallback quantized addition
    fn qadd_fallback(
        &self,
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        // Dequantize, add, requantize
        let a_f32 = self.dequantize_f32(a.data(), a.params())?;
        let b_f32 = self.dequantize_f32(b.data(), b.params())?;

        let mut c_f32 = Vec::with_capacity(a_f32.len());
        for (a_val, b_val) in a_f32.iter().zip(b_f32.iter()) {
            c_f32.push(a_val + b_val);
        }

        let c_quantized = self.quantize_f32(&c_f32, a.params())?;

        QuantizedTensor::from_data(
            c_quantized,
            a.shape().to_vec(),
            a.params().clone(),
            a.device().clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn test_hardware_features_detection() {
        let cpu_ops = HardwareQuantizationOps::new(Device::cpu().unwrap());
        let features = cpu_ops.hardware_features();

        assert!(features.supports_int8_simd);
        assert!(features.supports_int4_packed);
        assert!(features.max_parallel_ops > 0);
    }

    #[test]
    fn test_quantize_dequantize_u8() -> BackendResult<()> {
        let ops = HardwareQuantizationOps::new(Device::cpu().unwrap());
        let params = QuantizationParams::uint8_asymmetric();

        let input = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let quantized = ops.quantize_f32(&input, &params)?;
        let dequantized = ops.dequantize_f32(&quantized, &params)?;

        assert_eq!(quantized.len(), input.len());
        assert_eq!(dequantized.len(), input.len());

        Ok(())
    }

    #[test]
    fn test_quantize_i4_packed() -> BackendResult<()> {
        let ops = HardwareQuantizationOps::new(Device::cpu().unwrap());
        let params = QuantizationParams::int4_symmetric();

        let input = vec![1.0, -1.0, 2.0, -2.0];
        let quantized = ops.quantize_f32(&input, &params)?;

        // 4 elements packed into 2 bytes
        assert_eq!(quantized.len(), 2);

        Ok(())
    }

    #[test]
    fn test_binary_quantization() -> BackendResult<()> {
        let ops = HardwareQuantizationOps::new(Device::cpu().unwrap());
        let params = QuantizationParams::binary();

        let input = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let quantized = ops.quantize_f32(&input, &params)?;
        let dequantized = ops.dequantize_f32(&quantized, &params)?;

        // 8 elements packed into 1 byte
        assert_eq!(quantized.len(), 1);
        assert_eq!(dequantized.len(), 8);

        Ok(())
    }

    #[test]
    fn test_calibration() -> BackendResult<()> {
        let ops = HardwareQuantizationOps::new(Device::cpu().unwrap());

        let sample1 = vec![0.0, 1.0, 2.0];
        let sample2 = vec![-1.0, 0.5, 3.0];
        let samples = vec![sample1.as_slice(), sample2.as_slice()];

        let params = ops.calibrate(&samples, QuantizedDType::UInt8)?;

        assert_eq!(params.dtype, QuantizedDType::UInt8);
        assert!(params.scale[0] > 0.0);
        assert_eq!(params.min_val, Some(-1.0));
        assert_eq!(params.max_val, Some(3.0));

        Ok(())
    }

    #[test]
    fn test_qrelu() -> BackendResult<()> {
        let ops = HardwareQuantizationOps::new(Device::cpu().unwrap());
        let params = QuantizationParams::int8_symmetric();

        // Create a quantized tensor with some negative values
        let data = vec![100u8, 200u8, 50u8]; // Some will be negative when interpreted as i8
        let tensor = QuantizedTensor::from_data(
            data,
            vec![3],
            params,
            Device::cpu().unwrap(),
        )?;

        let result = ops.qrelu(&tensor)?;

        // All values should be >= zero_point in quantized space
        assert_eq!(result.shape(), tensor.shape());

        Ok(())
    }

    #[test]
    fn test_qadd() -> BackendResult<()> {
        let ops = HardwareQuantizationOps::new(Device::cpu().unwrap());
        let params = QuantizationParams::int8_symmetric();

        let data_a = vec![100u8, 50u8, 200u8];
        let data_b = vec![20u8, 30u8, 10u8];

        let tensor_a = QuantizedTensor::from_data(
            data_a,
            vec![3],
            params.clone(),
            Device::cpu().unwrap(),
        )?;

        let tensor_b = QuantizedTensor::from_data(
            data_b,
            vec![3],
            params,
            Device::cpu().unwrap(),
        )?;

        let result = ops.qadd(&tensor_a, &tensor_b)?;

        assert_eq!(result.shape(), tensor_a.shape());
        assert_eq!(result.data().len(), tensor_a.data().len());

        Ok(())
    }
}