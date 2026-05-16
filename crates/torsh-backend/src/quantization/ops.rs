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
        a: &QuantizedTensor,
        b: &QuantizedTensor,
    ) -> BackendResult<QuantizedTensor> {
        // Validate shapes for matrix multiplication: a is [M, K], b is [K, N].
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "qmatmul requires 2D tensors".to_string(),
            ));
        }
        let (m, k_a) = (a.shape[0], a.shape[1]);
        let (k_b, n) = (b.shape[0], b.shape[1]);
        if k_a != k_b {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "qmatmul dimension mismatch: a has K={k_a} but b has K={k_b}"
            )));
        }

        // Dequantize-compute-requantize strategy: optimal accuracy for a CPU
        // reference implementation without adding new dependencies.
        let a_float = self.dequantize_f32(&a.data, &a.params)?;
        let b_float = self.dequantize_f32(&b.data, &b.params)?;

        let mut out_float = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for k in 0..k_a {
                    acc += a_float[row * k_a + k] * b_float[k * n + col];
                }
                out_float[row * n + col] = acc;
            }
        }

        // Derive output quantization params from the result range.
        let mut out_params = a.params.clone();
        out_params.from_statistics(
            out_float.iter().copied().fold(f32::INFINITY, f32::min),
            out_float.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )?;

        let out_data = self.quantize_f32(&out_float, &out_params)?;
        QuantizedTensor::from_data(out_data, vec![m, n], out_params, a.device.clone())
    }

    fn qconv2d(
        &self,
        input: &QuantizedTensor,
        weight: &QuantizedTensor,
        bias: Option<&QuantizedTensor>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> BackendResult<QuantizedTensor> {
        // Expect input shape [N, C_in, H, W] and weight shape [C_out, C_in, kH, kW].
        if input.ndim() != 4 || weight.ndim() != 4 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "qconv2d requires 4D tensors [N, C, H, W] for input and [C_out, C_in, kH, kW] for weight".to_string(),
            ));
        }
        let (n, c_in, h_in, w_in) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (c_out, c_in_w, k_h, k_w) = (
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        );
        if c_in != c_in_w {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "qconv2d channel mismatch: input C_in={c_in} but weight C_in={c_in_w}"
            )));
        }

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;
        let h_out = (h_in + 2 * pad_h).saturating_sub(k_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w).saturating_sub(k_w) / stride_w + 1;

        let input_float = self.dequantize_f32(&input.data, &input.params)?;
        let weight_float = self.dequantize_f32(&weight.data, &weight.params)?;

        // Decode optional bias.
        let bias_float: Option<Vec<f32>> = match bias {
            Some(b) => Some(self.dequantize_f32(&b.data, &b.params)?),
            None => None,
        };

        let out_numel = n * c_out * h_out * w_out;
        let mut out_float = vec![0.0f32; out_numel];

        for batch in 0..n {
            for oc in 0..c_out {
                let bias_val = bias_float
                    .as_ref()
                    .map(|bf| bf.get(oc).copied().unwrap_or(0.0))
                    .unwrap_or(0.0);
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = 0.0f32;
                        for ic in 0..c_in {
                            for kh in 0..k_h {
                                let ih = oh * stride_h + kh;
                                let ih_pad = ih.checked_sub(pad_h);
                                for kw in 0..k_w {
                                    let iw = ow * stride_w + kw;
                                    let iw_pad = iw.checked_sub(pad_w);
                                    let in_val = match (ih_pad, iw_pad) {
                                        (Some(ih_r), Some(iw_r))
                                            if ih_r < h_in && iw_r < w_in =>
                                        {
                                            input_float[batch * c_in * h_in * w_in
                                                + ic * h_in * w_in
                                                + ih_r * w_in
                                                + iw_r]
                                        }
                                        _ => 0.0,
                                    };
                                    let w_val = weight_float[oc * c_in * k_h * k_w
                                        + ic * k_h * k_w
                                        + kh * k_w
                                        + kw];
                                    acc += in_val * w_val;
                                }
                            }
                        }
                        let out_idx =
                            batch * c_out * h_out * w_out + oc * h_out * w_out + oh * w_out + ow;
                        out_float[out_idx] = acc + bias_val;
                    }
                }
            }
        }

        let mut out_params = input.params.clone();
        out_params.from_statistics(
            out_float.iter().copied().fold(f32::INFINITY, f32::min),
            out_float.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )?;

        let out_data = self.quantize_f32(&out_float, &out_params)?;
        QuantizedTensor::from_data(
            out_data,
            vec![n, c_out, h_out, w_out],
            out_params,
            input.device.clone(),
        )
    }

    fn qadd(&self, a: &QuantizedTensor, b: &QuantizedTensor) -> BackendResult<QuantizedTensor> {
        if a.shape != b.shape {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "qadd shape mismatch: {:?} vs {:?}",
                a.shape, b.shape
            )));
        }

        let a_float = self.dequantize_f32(&a.data, &a.params)?;
        let b_float = self.dequantize_f32(&b.data, &b.params)?;
        let out_float: Vec<f32> = a_float
            .iter()
            .zip(b_float.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        let mut out_params = a.params.clone();
        out_params.from_statistics(
            out_float.iter().copied().fold(f32::INFINITY, f32::min),
            out_float.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )?;

        let out_data = self.quantize_f32(&out_float, &out_params)?;
        QuantizedTensor::from_data(out_data, a.shape.clone(), out_params, a.device.clone())
    }

    fn qrelu(&self, input: &QuantizedTensor) -> BackendResult<QuantizedTensor> {
        let input_float = self.dequantize_f32(&input.data, &input.params)?;
        let out_float: Vec<f32> = input_float.iter().map(|&v| v.max(0.0)).collect();

        let max_val = out_float
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        // After ReLU the minimum is always 0.
        let mut out_params = input.params.clone();
        out_params.from_statistics(0.0, max_val.max(0.0))?;

        let out_data = self.quantize_f32(&out_float, &out_params)?;
        QuantizedTensor::from_data(
            out_data,
            input.shape.clone(),
            out_params,
            input.device.clone(),
        )
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
        params
            .from_statistics(-2.0, 2.0)
            .expect("construction from statistics should succeed");

        let quantized = ops
            .quantize_f32(&input, &params)
            .expect("f32 quantization should succeed");
        let dequantized = ops
            .dequantize_f32(&quantized, &params)
            .expect("f32 dequantization should succeed");

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
        params
            .from_statistics(0.0, 1.0)
            .expect("construction from statistics should succeed");

        let quantized = ops
            .quantize_f32(&input, &params)
            .expect("f32 quantization should succeed");
        assert_eq!(quantized.len(), 3);

        let dequantized = ops
            .dequantize_f32(&quantized, &params)
            .expect("f32 dequantization should succeed");
        assert_eq!(dequantized.len(), 3);
    }

    #[test]
    fn test_quantize_int4() {
        let ops = CpuQuantizationOps::new();
        let input = vec![0.0, 1.0, -1.0, 2.0]; // 4 elements
        let mut params = QuantizationParams::int4_symmetric();
        params
            .from_statistics(-2.0, 2.0)
            .expect("construction from statistics should succeed");

        let quantized = ops
            .quantize_f32(&input, &params)
            .expect("f32 quantization should succeed");
        assert_eq!(quantized.len(), 2); // 4 elements packed into 2 bytes

        let dequantized = ops
            .dequantize_f32(&quantized, &params)
            .expect("f32 dequantization should succeed");
        assert_eq!(dequantized.len(), 4);
    }

    #[test]
    fn test_calibrate() {
        let ops = CpuQuantizationOps::new();
        let sample1 = vec![0.0, 1.0, 2.0];
        let sample2 = vec![-1.0, 0.5, 3.0];
        let samples = vec![sample1.as_slice(), sample2.as_slice()];

        let params = ops
            .calibrate(&samples, QuantizedDType::Int8)
            .expect("calibration should succeed");
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
    fn test_qmatmul_basic() {
        let ops = CpuQuantizationOps::new();
        let device = crate::Device::cpu().expect("cpu device should be available");

        // Build a 2x2 identity-ish matrix as a quantized tensor.
        // We construct a 2x2 matrix with values [1, 0, 0, 1] quantized to Int8.
        let mut params = QuantizationParams::int8_symmetric();
        params
            .from_statistics(-1.0, 1.0)
            .expect("params from statistics should succeed");
        let data = ops
            .quantize_f32(&[1.0, 0.0, 0.0, 1.0], &params)
            .expect("quantize should succeed");
        let a = QuantizedTensor::from_data(data.clone(), vec![2, 2], params.clone(), device.clone())
            .expect("tensor from data should succeed");
        let b = QuantizedTensor::from_data(data, vec![2, 2], params, device)
            .expect("tensor from data should succeed");

        let result = ops.qmatmul(&a, &b).expect("qmatmul should succeed");
        assert_eq!(result.shape(), &[2, 2]);

        let result_float = ops
            .dequantize_f32(&result.data, &result.params)
            .expect("dequantize should succeed");
        // I * I = I, so diagonal ≈ 1 and off-diagonal ≈ 0.
        assert!(
            (result_float[0] - 1.0).abs() < 0.15,
            "result[0,0]={} expected ≈1",
            result_float[0]
        );
        assert!(
            result_float[1].abs() < 0.15,
            "result[0,1]={} expected ≈0",
            result_float[1]
        );
    }

    #[test]
    fn test_qadd_basic() {
        let ops = CpuQuantizationOps::new();
        let device = crate::Device::cpu().expect("cpu device should be available");

        let mut params = QuantizationParams::int8_symmetric();
        params
            .from_statistics(-4.0, 4.0)
            .expect("params from statistics should succeed");

        let a_data = ops
            .quantize_f32(&[1.0, 2.0, 3.0, 4.0], &params)
            .expect("quantize should succeed");
        let b_data = ops
            .quantize_f32(&[0.5, 0.5, 0.5, 0.5], &params)
            .expect("quantize should succeed");

        let a = QuantizedTensor::from_data(a_data, vec![2, 2], params.clone(), device.clone())
            .expect("tensor from data should succeed");
        let b = QuantizedTensor::from_data(b_data, vec![2, 2], params, device)
            .expect("tensor from data should succeed");

        let result = ops.qadd(&a, &b).expect("qadd should succeed");
        assert_eq!(result.shape(), &[2, 2]);

        let result_float = ops
            .dequantize_f32(&result.data, &result.params)
            .expect("dequantize should succeed");
        // Expected: [1.5, 2.5, 3.5, 4.5]
        assert!(
            (result_float[0] - 1.5).abs() < 0.2,
            "result[0]={} expected ≈1.5",
            result_float[0]
        );
        assert!(
            (result_float[3] - 4.5).abs() < 0.2,
            "result[3]={} expected ≈4.5",
            result_float[3]
        );
    }

    #[test]
    fn test_qrelu_basic() {
        let ops = CpuQuantizationOps::new();
        let device = crate::Device::cpu().expect("cpu device should be available");

        let mut params = QuantizationParams::int8_symmetric();
        params
            .from_statistics(-2.0, 2.0)
            .expect("params from statistics should succeed");

        let data = ops
            .quantize_f32(&[-1.0, 0.0, 1.0, 2.0], &params)
            .expect("quantize should succeed");
        let tensor =
            QuantizedTensor::from_data(data, vec![4], params, device).expect("tensor should work");

        let result = ops.qrelu(&tensor).expect("qrelu should succeed");
        assert_eq!(result.shape(), &[4]);

        let result_float = ops
            .dequantize_f32(&result.data, &result.params)
            .expect("dequantize should succeed");
        // ReLU: [-1, 0, 1, 2] -> [0, 0, 1, 2]
        assert!(result_float[0] >= -0.1, "negative value should be zeroed");
        assert!(result_float[0] <= 0.15, "negative value should be zeroed");
        assert!(
            (result_float[3] - 2.0).abs() < 0.2,
            "positive value should be preserved"
        );
    }

    #[test]
    fn test_qmatmul_shape_mismatch() {
        let ops = CpuQuantizationOps::new();
        let device = crate::Device::cpu().expect("cpu device should be available");
        let mut params = QuantizationParams::int8_symmetric();
        params
            .from_statistics(-1.0, 1.0)
            .expect("params from statistics should succeed");
        let data_a = ops
            .quantize_f32(&[1.0, 0.0, 0.0], &params)
            .expect("quantize should succeed");
        let data_b = ops
            .quantize_f32(&[1.0, 0.0, 0.0, 0.0], &params)
            .expect("quantize should succeed");
        let a = QuantizedTensor::from_data(data_a, vec![1, 3], params.clone(), device.clone())
            .expect("tensor should work");
        let b = QuantizedTensor::from_data(data_b, vec![2, 2], params, device)
            .expect("tensor should work");
        assert!(ops.qmatmul(&a, &b).is_err());
    }

    #[test]
    fn test_qadd_shape_mismatch() {
        let ops = CpuQuantizationOps::new();
        let device = crate::Device::cpu().expect("cpu device should be available");
        let params = QuantizationParams::default();
        let a = QuantizedTensor::new(vec![2, 2], params.clone(), device.clone());
        let b = QuantizedTensor::new(vec![2, 3], params, device);
        assert!(ops.qadd(&a, &b).is_err());
    }

    #[test]
    fn test_qconv2d_basic() {
        let ops = CpuQuantizationOps::new();
        let device = crate::Device::cpu().expect("cpu device should be available");

        // 1x1 input, 1x1 kernel (trivial convolution).
        let mut params = QuantizationParams::int8_symmetric();
        params
            .from_statistics(-2.0, 2.0)
            .expect("params from statistics should succeed");

        // Input [N=1, C=1, H=3, W=3] of all 1s.
        let input_data = ops
            .quantize_f32(&vec![1.0f32; 9], &params)
            .expect("quantize should succeed");
        // Weight [C_out=1, C_in=1, kH=1, kW=1] = [1.0].
        let weight_data = ops
            .quantize_f32(&[1.0f32], &params)
            .expect("quantize should succeed");

        let input =
            QuantizedTensor::from_data(input_data, vec![1, 1, 3, 3], params.clone(), device.clone())
                .expect("tensor should work");
        let weight =
            QuantizedTensor::from_data(weight_data, vec![1, 1, 1, 1], params, device)
                .expect("tensor should work");

        let result = ops
            .qconv2d(&input, &weight, None, (1, 1), (0, 0))
            .expect("qconv2d should succeed");
        // 1x1 kernel on 3x3 input with stride 1, no padding => 3x3 output.
        assert_eq!(result.shape(), &[1, 1, 3, 3]);

        let result_float = ops
            .dequantize_f32(&result.data, &result.params)
            .expect("dequantize should succeed");
        // Each output element = 1 * 1 = 1.
        for &v in &result_float {
            assert!((v - 1.0).abs() < 0.2, "expected ≈1.0 but got {v}");
        }
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
