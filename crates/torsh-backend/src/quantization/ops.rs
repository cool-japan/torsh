//! Quantization operations trait and basic implementations
//!
//! This module defines the core QuantizationOps trait that provides the interface
//! for all quantization operations, including quantization/dequantization,
//! quantized arithmetic, and calibration. It also includes basic CPU implementations
//! and helper functions for common quantization operations.

use super::params::QuantizationParams;
use super::tensor::QuantizedTensor;
use super::types::{QuantizationScheme, QuantizedDType};
use crate::BackendResult;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Quantization operations trait
///
/// Defines the interface for quantization operations that can be implemented
/// by different backends (CPU, GPU, specialized accelerators). This trait
/// provides both low-level operations (quantize/dequantize) and high-level
/// operations (quantized arithmetic and neural network operations).
pub trait QuantizationOps {
    /// Quantize floating-point data to quantized representation
    ///
    /// Converts an array of floating-point values to quantized integers
    /// using the provided quantization parameters. The output format
    /// depends on the quantization type and may pack multiple values
    /// per byte for sub-byte quantization.
    ///
    /// # Arguments
    ///
    /// * `input` - Floating-point input data
    /// * `params` - Quantization parameters (scale, zero_point, etc.)
    ///
    /// # Returns
    ///
    /// Returns quantized data as a byte vector, or an error if
    /// quantization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # use torsh_backend::quantization::{QuantizationOps, QuantizationParams, CpuQuantizationOps};
    /// let ops = CpuQuantizationOps::new();
    /// let input = vec![0.0, 1.0, 2.0, 3.0];
    /// let params = QuantizationParams::int8_symmetric();
    /// let quantized = ops.quantize_f32(&input, &params).unwrap();
    /// ```
    fn quantize_f32(&self, input: &[f32], params: &QuantizationParams) -> BackendResult<Vec<u8>>;

    /// Dequantize quantized data back to floating-point
    ///
    /// Converts quantized integer data back to floating-point values
    /// using the provided quantization parameters. This is the inverse
    /// operation of `quantize_f32`.
    ///
    /// # Arguments
    ///
    /// * `input` - Quantized input data as bytes
    /// * `params` - Quantization parameters used for the original quantization
    ///
    /// # Returns
    ///
    /// Returns floating-point data, or an error if dequantization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # use torsh_backend::quantization::{QuantizationOps, QuantizationParams, CpuQuantizationOps};
    /// let ops = CpuQuantizationOps::new();
    /// let quantized = vec![0, 64, 128, 192];
    /// let params = QuantizationParams::uint8_asymmetric();
    /// let dequantized = ops.dequantize_f32(&quantized, &params).unwrap();
    /// ```
    fn dequantize_f32(&self, input: &[u8], params: &QuantizationParams) -> BackendResult<Vec<f32>>;

    /// Quantized matrix multiplication
    ///
    /// Performs matrix multiplication on two quantized tensors, returning
    /// a quantized result. This operation may require requantization to
    /// handle the increased precision of the multiplication result.
    ///
    /// # Arguments
    ///
    /// * `a` - Left operand quantized tensor
    /// * `b` - Right operand quantized tensor
    ///
    /// # Returns
    ///
    /// Returns the quantized matrix multiplication result.
    fn qmatmul(&self, a: &QuantizedTensor, b: &QuantizedTensor) -> BackendResult<QuantizedTensor>;

    /// Quantized 2D convolution
    ///
    /// Performs 2D convolution with quantized input, weights, and optional bias.
    /// This is a core operation for quantized convolutional neural networks.
    ///
    /// # Arguments
    ///
    /// * `input` - Input feature maps (quantized)
    /// * `weight` - Convolution weights (quantized)
    /// * `bias` - Optional bias terms (quantized)
    /// * `stride` - Convolution stride (height, width)
    /// * `padding` - Convolution padding (height, width)
    ///
    /// # Returns
    ///
    /// Returns the quantized convolution result.
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
    /// Performs element-wise addition of two quantized tensors.
    /// This may require rescaling if the tensors have different
    /// quantization parameters.
    ///
    /// # Arguments
    ///
    /// * `a` - First operand
    /// * `b` - Second operand
    ///
    /// # Returns
    ///
    /// Returns the quantized addition result.
    fn qadd(&self, a: &QuantizedTensor, b: &QuantizedTensor) -> BackendResult<QuantizedTensor>;

    /// Quantized ReLU activation
    ///
    /// Applies ReLU activation to a quantized tensor. For certain
    /// quantization schemes, this can be implemented very efficiently
    /// by simply clamping values to the zero point.
    ///
    /// # Arguments
    ///
    /// * `input` - Input quantized tensor
    ///
    /// # Returns
    ///
    /// Returns the tensor after applying ReLU activation.
    fn qrelu(&self, input: &QuantizedTensor) -> BackendResult<QuantizedTensor>;

    /// Calibrate quantization parameters from sample data
    ///
    /// Analyzes sample data to determine optimal quantization parameters.
    /// This is typically used for post-training quantization to find
    /// parameters that minimize quantization error.
    ///
    /// # Arguments
    ///
    /// * `samples` - Array of sample data arrays
    /// * `target_dtype` - Target quantization data type
    ///
    /// # Returns
    ///
    /// Returns calibrated quantization parameters.
    fn calibrate(
        &self,
        samples: &[&[f32]],
        target_dtype: QuantizedDType,
    ) -> BackendResult<QuantizationParams>;
}

/// CPU-based quantization operations implementation
///
/// Provides a reference implementation of quantization operations
/// using CPU-only code. This serves as a fallback when specialized
/// hardware acceleration is not available.
#[derive(Debug, Clone, Default)]
pub struct CpuQuantizationOps;

impl CpuQuantizationOps {
    /// Create a new CPU quantization operations instance
    pub fn new() -> Self {
        Self
    }

    /// Helper function to quantize a single value
    fn quantize_value(
        &self,
        value: f32,
        scale: f32,
        zero_point: i32,
        dtype: &QuantizedDType,
    ) -> i32 {
        let (qmin, qmax) = dtype.value_range();
        let quantized = (value / scale).round() + zero_point as f32;
        quantized.clamp(qmin as f32, qmax as f32) as i32
    }

    /// Helper function to dequantize a single value
    fn dequantize_value(&self, quantized: i32, scale: f32, zero_point: i32) -> f32 {
        scale * (quantized - zero_point) as f32
    }

    /// Pack sub-byte values into bytes
    fn pack_sub_byte(&self, values: &[i32], dtype: &QuantizedDType) -> Vec<u8> {
        match dtype {
            QuantizedDType::Int4 | QuantizedDType::UInt4 => {
                let mut result = Vec::new();
                for chunk in values.chunks(2) {
                    let val1 = chunk[0] as u8 & 0x0F;
                    let val2 = if chunk.len() > 1 {
                        (chunk[1] as u8 & 0x0F) << 4
                    } else {
                        0
                    };
                    result.push(val1 | val2);
                }
                result
            }
            QuantizedDType::Binary => {
                let mut result = Vec::new();
                for chunk in values.chunks(8) {
                    let mut byte = 0u8;
                    for (i, &val) in chunk.iter().enumerate() {
                        if val != 0 {
                            byte |= 1 << i;
                        }
                    }
                    result.push(byte);
                }
                result
            }
            _ => {
                // For other types, just cast to bytes
                values.iter().map(|&v| v as u8).collect()
            }
        }
    }

    /// Unpack sub-byte values from bytes
    fn unpack_sub_byte(
        &self,
        data: &[u8],
        num_elements: usize,
        dtype: &QuantizedDType,
    ) -> Vec<i32> {
        match dtype {
            QuantizedDType::Int4 => {
                let mut result = Vec::new();
                for &byte in data {
                    if result.len() < num_elements {
                        let val1 = ((byte & 0x0F) as i8) << 4 >> 4; // Sign extend
                        result.push(val1 as i32);
                    }
                    if result.len() < num_elements {
                        let val2 = ((byte & 0xF0) as i8) >> 4; // Already sign extended
                        result.push(val2 as i32);
                    }
                }
                result.truncate(num_elements);
                result
            }
            QuantizedDType::UInt4 => {
                let mut result = Vec::new();
                for &byte in data {
                    if result.len() < num_elements {
                        result.push((byte & 0x0F) as i32);
                    }
                    if result.len() < num_elements {
                        result.push(((byte & 0xF0) >> 4) as i32);
                    }
                }
                result.truncate(num_elements);
                result
            }
            QuantizedDType::Binary => {
                let mut result = Vec::new();
                for &byte in data {
                    for bit in 0..8 {
                        if result.len() < num_elements {
                            result.push(if (byte >> bit) & 1 != 0 { 1 } else { 0 });
                        }
                    }
                }
                result.truncate(num_elements);
                result
            }
            _ => {
                // For other types, just cast from bytes
                data.iter().take(num_elements).map(|&b| b as i32).collect()
            }
        }
    }
}

impl QuantizationOps for CpuQuantizationOps {
    fn quantize_f32(&self, input: &[f32], params: &QuantizationParams) -> BackendResult<Vec<u8>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Validate parameters
        params.validate()?;

        // Handle per-channel quantization
        let num_channels = params.scale.len();
        let channel_size = if num_channels > 1 {
            input.len() / num_channels
        } else {
            input.len()
        };

        let mut quantized_values = Vec::with_capacity(input.len());

        match params.scheme {
            QuantizationScheme::ChannelWise if num_channels > 1 => {
                // Per-channel quantization
                for (channel, chunk) in input.chunks(channel_size).enumerate() {
                    let scale = params.scale[channel];
                    let zero_point = params.zero_point[channel];

                    for &value in chunk {
                        let q_val = self.quantize_value(value, scale, zero_point, &params.dtype);
                        quantized_values.push(q_val);
                    }
                }
            }
            _ => {
                // Tensor-wide quantization
                let scale = params.scale[0];
                let zero_point = params.zero_point[0];

                for &value in input {
                    let q_val = self.quantize_value(value, scale, zero_point, &params.dtype);
                    quantized_values.push(q_val);
                }
            }
        }

        // Pack into bytes based on data type
        let packed = if params.dtype.is_sub_byte() {
            self.pack_sub_byte(&quantized_values, &params.dtype)
        } else {
            match params.dtype {
                QuantizedDType::Int8 | QuantizedDType::UInt8 => {
                    quantized_values.iter().map(|&v| v as u8).collect()
                }
                QuantizedDType::Int16 | QuantizedDType::UInt16 => {
                    let mut result = Vec::new();
                    for &val in &quantized_values {
                        let bytes = (val as u16).to_le_bytes();
                        result.extend_from_slice(&bytes);
                    }
                    result
                }
                _ => {
                    return Err(torsh_core::error::TorshError::NotImplemented(format!(
                        "Quantization not implemented for {:?}",
                        params.dtype
                    )))
                }
            }
        };

        Ok(packed)
    }

    fn dequantize_f32(&self, input: &[u8], params: &QuantizationParams) -> BackendResult<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Validate parameters
        params.validate()?;

        // Determine number of elements
        let num_elements = match params.dtype {
            QuantizedDType::Int4 | QuantizedDType::UInt4 => input.len() * 2,
            QuantizedDType::Binary => input.len() * 8,
            QuantizedDType::Int8 | QuantizedDType::UInt8 => input.len(),
            QuantizedDType::Int16 | QuantizedDType::UInt16 => input.len() / 2,
            _ => {
                return Err(torsh_core::error::TorshError::NotImplemented(format!(
                    "Dequantization not implemented for {:?}",
                    params.dtype
                )))
            }
        };

        // Unpack quantized values
        let quantized_values = if params.dtype.is_sub_byte() {
            self.unpack_sub_byte(input, num_elements, &params.dtype)
        } else {
            match params.dtype {
                QuantizedDType::Int8 => input.iter().map(|&b| b as i8 as i32).collect(),
                QuantizedDType::UInt8 => input.iter().map(|&b| b as i32).collect(),
                QuantizedDType::Int16 => input
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as i32)
                    .collect(),
                QuantizedDType::UInt16 => input
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as i32)
                    .collect(),
                _ => unreachable!(),
            }
        };

        // Dequantize to floating point
        let num_channels = params.scale.len();
        let mut result = Vec::with_capacity(quantized_values.len());

        if num_channels > 1 && matches!(params.scheme, QuantizationScheme::ChannelWise) {
            // Per-channel dequantization
            let channel_size = quantized_values.len() / num_channels;
            for (channel, chunk) in quantized_values.chunks(channel_size).enumerate() {
                let scale = params.scale[channel];
                let zero_point = params.zero_point[channel];

                for &q_val in chunk {
                    let f_val = self.dequantize_value(q_val, scale, zero_point);
                    result.push(f_val);
                }
            }
        } else {
            // Tensor-wide dequantization
            let scale = params.scale[0];
            let zero_point = params.zero_point[0];

            for &q_val in &quantized_values {
                let f_val = self.dequantize_value(q_val, scale, zero_point);
                result.push(f_val);
            }
        }

        Ok(result)
    }

    fn qmatmul(
        &self,
        _a: &QuantizedTensor,
        _b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        // Placeholder implementation
        Err(torsh_core::error::TorshError::NotImplemented(
            "Quantized matrix multiplication not yet implemented for CPU backend".to_string(),
        ))
    }

    fn qconv2d(
        &self,
        _input: &QuantizedTensor,
        _weight: &QuantizedTensor,
        _bias: Option<&QuantizedTensor>,
        _stride: (usize, usize),
        _padding: (usize, usize),
    ) -> BackendResult<QuantizedTensor> {
        // Placeholder implementation
        Err(torsh_core::error::TorshError::NotImplemented(
            "Quantized convolution not yet implemented for CPU backend".to_string(),
        ))
    }

    fn qadd(&self, _a: &QuantizedTensor, _b: &QuantizedTensor) -> BackendResult<QuantizedTensor> {
        // Placeholder implementation
        Err(torsh_core::error::TorshError::NotImplemented(
            "Quantized addition not yet implemented for CPU backend".to_string(),
        ))
    }

    fn qrelu(&self, _input: &QuantizedTensor) -> BackendResult<QuantizedTensor> {
        // Placeholder implementation
        Err(torsh_core::error::TorshError::NotImplemented(
            "Quantized ReLU not yet implemented for CPU backend".to_string(),
        ))
    }

    fn calibrate(
        &self,
        samples: &[&[f32]],
        target_dtype: QuantizedDType,
    ) -> BackendResult<QuantizationParams> {
        if samples.is_empty() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Cannot calibrate with empty samples".to_string(),
            ));
        }

        // Find global min and max across all samples
        let mut global_min = f32::INFINITY;
        let mut global_max = f32::NEG_INFINITY;

        for sample in samples {
            for &value in *sample {
                if value.is_finite() {
                    global_min = global_min.min(value);
                    global_max = global_max.max(value);
                }
            }
        }

        if !global_min.is_finite() || !global_max.is_finite() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "No finite values found in calibration samples".to_string(),
            ));
        }

        // Create parameters and calculate from statistics
        let mut params = QuantizationParams {
            dtype: target_dtype,
            scheme: QuantizationScheme::Asymmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        };

        params.from_statistics(global_min, global_max)?;
        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::QuantizationParams;

    #[test]
    fn test_cpu_quantization_ops_creation() {
        let ops = CpuQuantizationOps::new();
        assert!(format!("{:?}", ops).contains("CpuQuantizationOps"));
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let ops = CpuQuantizationOps::new();
        let input = vec![0.0, 1.0, -1.0, 2.0];
        let mut params = QuantizationParams::int8_symmetric();
        params.from_statistics(-2.0, 2.0).unwrap();

        let quantized = ops.quantize_f32(&input, &params).unwrap();
        let dequantized = ops.dequantize_f32(&quantized, &params).unwrap();

        // Check that values are approximately preserved
        for (original, recovered) in input.iter().zip(dequantized.iter()) {
            assert!(
                (original - recovered).abs() < 0.1,
                "Original: {}, Recovered: {}",
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_quantize_uint8() {
        let ops = CpuQuantizationOps::new();
        let input = vec![0.0, 0.5, 1.0];
        let mut params = QuantizationParams::uint8_asymmetric();
        params.from_statistics(0.0, 1.0).unwrap();

        let quantized = ops.quantize_f32(&input, &params).unwrap();
        assert_eq!(quantized.len(), 3);

        let dequantized = ops.dequantize_f32(&quantized, &params).unwrap();
        assert_eq!(dequantized.len(), 3);
    }

    #[test]
    fn test_quantize_int4() {
        let ops = CpuQuantizationOps::new();
        let input = vec![0.0, 1.0, -1.0, 2.0]; // 4 elements
        let mut params = QuantizationParams::int4_symmetric();
        params.from_statistics(-2.0, 2.0).unwrap();

        let quantized = ops.quantize_f32(&input, &params).unwrap();
        assert_eq!(quantized.len(), 2); // 4 elements packed into 2 bytes

        let dequantized = ops.dequantize_f32(&quantized, &params).unwrap();
        assert_eq!(dequantized.len(), 4);
    }

    #[test]
    fn test_calibrate() {
        let ops = CpuQuantizationOps::new();
        let sample1 = vec![0.0, 1.0, 2.0];
        let sample2 = vec![-1.0, 0.5, 3.0];
        let samples = vec![sample1.as_slice(), sample2.as_slice()];

        let params = ops.calibrate(&samples, QuantizedDType::Int8).unwrap();
        assert_eq!(params.dtype, QuantizedDType::Int8);
        assert_eq!(params.min_val, Some(-1.0));
        assert_eq!(params.max_val, Some(3.0));
        assert!(params.scale[0] > 0.0);
    }

    #[test]
    fn test_calibrate_empty_samples() {
        let ops = CpuQuantizationOps::new();
        let samples: Vec<&[f32]> = vec![];

        let result = ops.calibrate(&samples, QuantizedDType::Int8);
        assert!(result.is_err());
    }

    #[test]
    fn test_unimplemented_operations() {
        let ops = CpuQuantizationOps::new();
        let params = QuantizationParams::default();
        let tensor = QuantizedTensor::new(vec![2, 2], params, crate::Device::cpu().unwrap());

        assert!(ops.qmatmul(&tensor, &tensor).is_err());
        assert!(ops.qconv2d(&tensor, &tensor, None, (1, 1), (0, 0)).is_err());
        assert!(ops.qadd(&tensor, &tensor).is_err());
        assert!(ops.qrelu(&tensor).is_err());
    }

    #[test]
    fn test_quantize_value_helper() {
        let ops = CpuQuantizationOps::new();

        // Test symmetric int8 quantization
        let result = ops.quantize_value(1.0, 0.5, 0, &QuantizedDType::Int8);
        assert_eq!(result, 2); // 1.0 / 0.5 + 0 = 2

        // Test clamping
        let result = ops.quantize_value(1000.0, 1.0, 0, &QuantizedDType::Int8);
        assert_eq!(result, 127); // Clamped to max value
    }

    #[test]
    fn test_dequantize_value_helper() {
        let ops = CpuQuantizationOps::new();

        let result = ops.dequantize_value(2, 0.5, 0);
        assert_eq!(result, 1.0); // 0.5 * (2 - 0) = 1.0
    }
}
