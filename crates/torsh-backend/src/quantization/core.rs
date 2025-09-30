//! Core quantization types, parameters, and tensor representations
//!
//! This module provides the fundamental building blocks for quantization in ToRSh:
//! - [`QuantizedDType`] - Quantization data types (Int8, UInt8, Int4, etc.)
//! - [`QuantizationScheme`] - Different quantization approaches (Linear, Symmetric, etc.)
//! - [`QuantizationParams`] - Complete parameter set for quantization operations
//! - [`QuantizedTensor`] - Tensor representation with quantization metadata

use crate::{BackendResult, Device};
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec};

/// Quantization data types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizedDType {
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 16-bit unsigned integer
    UInt16,
    /// 4-bit signed integer (packed)
    Int4,
    /// 4-bit unsigned integer (packed)
    UInt4,
    /// 1-bit binary (packed)
    Binary,
    /// Mixed precision with different bits per channel
    Mixed(Vec<u8>),
}

impl QuantizedDType {
    /// Get the number of bits for this quantization type
    pub fn bits(&self) -> u8 {
        match self {
            QuantizedDType::Int8 | QuantizedDType::UInt8 => 8,
            QuantizedDType::Int16 | QuantizedDType::UInt16 => 16,
            QuantizedDType::Int4 | QuantizedDType::UInt4 => 4,
            QuantizedDType::Binary => 1,
            QuantizedDType::Mixed(bits) => bits.iter().max().copied().unwrap_or(8),
        }
    }

    /// Check if this type is signed
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            QuantizedDType::Int8 | QuantizedDType::Int16 | QuantizedDType::Int4
        )
    }

    /// Get the range of values for this type
    pub fn value_range(&self) -> (i64, i64) {
        match self {
            QuantizedDType::Int8 => (-128, 127),
            QuantizedDType::UInt8 => (0, 255),
            QuantizedDType::Int16 => (-32768, 32767),
            QuantizedDType::UInt16 => (0, 65535),
            QuantizedDType::Int4 => (-8, 7),
            QuantizedDType::UInt4 => (0, 15),
            QuantizedDType::Binary => (0, 1),
            QuantizedDType::Mixed(_) => (0, 255), // Conservative estimate
        }
    }

    /// Get the memory footprint per element in bytes
    pub fn bytes_per_element(&self) -> usize {
        match self {
            QuantizedDType::Int8 | QuantizedDType::UInt8 => 1,
            QuantizedDType::Int16 | QuantizedDType::UInt16 => 2,
            QuantizedDType::Int4 | QuantizedDType::UInt4 => 1, // Packed 2 elements per byte
            QuantizedDType::Binary => 1,                       // Packed 8 elements per byte
            QuantizedDType::Mixed(_) => 1, // Assume 1 byte as conservative estimate
        }
    }

    /// Check if this type requires packing (multiple elements per byte)
    pub fn is_packed(&self) -> bool {
        matches!(self, QuantizedDType::Int4 | QuantizedDType::UInt4 | QuantizedDType::Binary)
    }

    /// Get the packing factor (elements per byte) for packed types
    pub fn packing_factor(&self) -> usize {
        match self {
            QuantizedDType::Int4 | QuantizedDType::UInt4 => 2,
            QuantizedDType::Binary => 8,
            _ => 1,
        }
    }
}

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationScheme {
    /// Linear/uniform quantization
    Linear,
    /// Logarithmic quantization for dynamic range
    Logarithmic,
    /// Symmetric quantization (zero point = 0)
    Symmetric,
    /// Asymmetric quantization with custom zero point
    Asymmetric,
    /// Block-wise quantization
    BlockWise,
    /// Channel-wise quantization
    ChannelWise,
}

impl QuantizationScheme {
    /// Check if this scheme supports zero points
    pub fn supports_zero_point(&self) -> bool {
        matches!(self, QuantizationScheme::Asymmetric | QuantizationScheme::Linear)
    }

    /// Check if this scheme requires per-channel parameters
    pub fn is_per_channel(&self) -> bool {
        matches!(self, QuantizationScheme::ChannelWise)
    }

    /// Check if this scheme requires block-wise parameters
    pub fn is_block_wise(&self) -> bool {
        matches!(self, QuantizationScheme::BlockWise)
    }
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    /// Quantization data type
    pub dtype: QuantizedDType,
    /// Quantization scheme
    pub scheme: QuantizationScheme,
    /// Scale factor(s)
    pub scale: Vec<f32>,
    /// Zero point(s)
    pub zero_point: Vec<i32>,
    /// Block size for block-wise quantization
    pub block_size: Option<usize>,
    /// Minimum and maximum values for calibration
    pub min_val: Option<f32>,
    pub max_val: Option<f32>,
}

impl Default for QuantizationParams {
    fn default() -> Self {
        Self {
            dtype: QuantizedDType::UInt8,
            scheme: QuantizationScheme::Linear,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }
}

impl QuantizationParams {
    /// Create parameters for INT8 symmetric quantization
    pub fn int8_symmetric() -> Self {
        Self {
            dtype: QuantizedDType::Int8,
            scheme: QuantizationScheme::Symmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for UINT8 asymmetric quantization
    pub fn uint8_asymmetric() -> Self {
        Self {
            dtype: QuantizedDType::UInt8,
            scheme: QuantizationScheme::Asymmetric,
            scale: vec![1.0],
            zero_point: vec![128],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for INT4 quantization
    pub fn int4_symmetric() -> Self {
        Self {
            dtype: QuantizedDType::Int4,
            scheme: QuantizationScheme::Symmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for binary quantization
    pub fn binary() -> Self {
        Self {
            dtype: QuantizedDType::Binary,
            scheme: QuantizationScheme::Symmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for channel-wise quantization
    pub fn channel_wise(num_channels: usize, dtype: QuantizedDType) -> Self {
        Self {
            dtype,
            scheme: QuantizationScheme::ChannelWise,
            scale: vec![1.0; num_channels],
            zero_point: vec![0; num_channels],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }

    /// Create parameters for block-wise quantization
    pub fn block_wise(block_size: usize, dtype: QuantizedDType) -> Self {
        Self {
            dtype,
            scheme: QuantizationScheme::BlockWise,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: Some(block_size),
            min_val: None,
            max_val: None,
        }
    }

    /// Calculate quantization parameters from input statistics
    pub fn from_statistics(&mut self, min_val: f32, max_val: f32) -> BackendResult<()> {
        self.min_val = Some(min_val);
        self.max_val = Some(max_val);

        let (qmin, qmax) = self.dtype.value_range();
        let qmin = qmin as f32;
        let qmax = qmax as f32;

        match self.scheme {
            QuantizationScheme::Symmetric => {
                let max_range = max_val.abs().max(min_val.abs());
                if max_range == 0.0 {
                    self.scale[0] = 1.0;
                } else {
                    self.scale[0] = (2.0 * max_range) / (qmax - qmin);
                }
                self.zero_point[0] = 0;
            }
            QuantizationScheme::Asymmetric => {
                if max_val == min_val {
                    self.scale[0] = 1.0;
                    self.zero_point[0] = 0;
                } else {
                    self.scale[0] = (max_val - min_val) / (qmax - qmin);
                    self.zero_point[0] = (qmin - min_val / self.scale[0]).round() as i32;
                }
            }
            _ => {
                // For other schemes, use asymmetric as default
                if max_val == min_val {
                    self.scale[0] = 1.0;
                    self.zero_point[0] = 0;
                } else {
                    self.scale[0] = (max_val - min_val) / (qmax - qmin);
                    self.zero_point[0] = (qmin - min_val / self.scale[0]).round() as i32;
                }
            }
        }

        Ok(())
    }

    /// Validate parameter consistency
    pub fn validate(&self) -> BackendResult<()> {
        // Check scale and zero_point length consistency
        if self.scheme.is_per_channel() {
            if self.scale.is_empty() || self.zero_point.is_empty() {
                return Err(TorshError::dimension_error(
                    "Channel-wise quantization requires non-empty scale and zero_point vectors",
                    "validate"
                ).into());
            }
            if self.scale.len() != self.zero_point.len() {
                return Err(TorshError::dimension_error(
                    "Scale and zero_point vectors must have the same length for channel-wise quantization",
                    "validate"
                ).into());
            }
        } else {
            if self.scale.len() != 1 || self.zero_point.len() != 1 {
                return Err(TorshError::dimension_error(
                    "Non-channel-wise quantization requires exactly one scale and zero_point value",
                    "validate"
                ).into());
            }
        }

        // Check block size for block-wise quantization
        if self.scheme.is_block_wise() && self.block_size.is_none() {
            return Err(TorshError::dimension_error(
                "Block-wise quantization requires a block_size",
                "validate"
            ).into());
        }

        // Check scale values are positive
        for &scale in &self.scale {
            if scale <= 0.0 || !scale.is_finite() {
                return Err(TorshError::dimension_error(
                    "Scale values must be positive and finite",
                    "validate"
                ).into());
            }
        }

        Ok(())
    }

    /// Estimate memory overhead for parameters
    pub fn memory_overhead(&self) -> usize {
        let scale_size = self.scale.len() * std::mem::size_of::<f32>();
        let zero_point_size = self.zero_point.len() * std::mem::size_of::<i32>();
        scale_size + zero_point_size + std::mem::size_of::<Self>()
    }

    /// Create new quantization parameters with scale and zero_point
    ///
    /// This is a convenience method for benchmarking purposes
    pub fn new(scale: f32, zero_point: i32) -> Self {
        Self {
            dtype: QuantizedDType::Int8,
            scheme: QuantizationScheme::Asymmetric,
            scale: vec![scale],
            zero_point: vec![zero_point],
            block_size: None,
            min_val: None,
            max_val: None,
        }
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data
    pub data: Vec<u8>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Quantization parameters
    pub params: QuantizationParams,
    /// Device where tensor is stored
    pub device: Device,
}

impl QuantizedTensor {
    /// Create a new quantized tensor
    pub fn new(shape: Vec<usize>, params: QuantizationParams, device: Device) -> BackendResult<Self> {
        // Validate parameters first
        params.validate()?;

        let total_elements: usize = shape.iter().product();
        let bytes_per_element = params.dtype.bytes_per_element();

        let data_size = if params.dtype.is_packed() {
            // For packed types, calculate actual storage needed
            let packing_factor = params.dtype.packing_factor();
            (total_elements + packing_factor - 1) / packing_factor
        } else {
            total_elements * bytes_per_element
        };

        Ok(Self {
            data: vec![0; data_size],
            shape,
            params,
            device,
        })
    }

    /// Create a quantized tensor from raw data
    pub fn from_data(
        data: Vec<u8>,
        shape: Vec<usize>,
        params: QuantizationParams,
        device: Device,
    ) -> BackendResult<Self> {
        params.validate()?;

        let tensor = Self {
            data,
            shape,
            params,
            device,
        };

        // Validate that data size matches expected size
        tensor.validate_data_size()?;

        Ok(tensor)
    }

    /// Get the number of elements in the tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.data.len() + self.params.memory_overhead()
    }

    /// Get the data size in bytes
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Get the compression ratio compared to FP32
    pub fn compression_ratio(&self) -> f32 {
        let fp32_size = self.num_elements() * 4; // 4 bytes per FP32
        let quantized_size = self.data_size();
        fp32_size as f32 / quantized_size as f32
    }

    /// Validate that data size matches expected size for the tensor
    fn validate_data_size(&self) -> BackendResult<()> {
        let total_elements = self.num_elements();
        let expected_size = if self.params.dtype.is_packed() {
            let packing_factor = self.params.dtype.packing_factor();
            (total_elements + packing_factor - 1) / packing_factor
        } else {
            total_elements * self.params.dtype.bytes_per_element()
        };

        if self.data.len() != expected_size {
            return Err(TorshError::dimension_error(
                &format!(
                    "Data size mismatch: expected {} bytes, got {}",
                    expected_size,
                    self.data.len()
                ),
                "validate_memory_layout"
            ).into());
        }

        Ok(())
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get quantization parameters
    pub fn params(&self) -> &QuantizationParams {
        &self.params
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if tensor is on CPU
    pub fn is_cpu(&self) -> bool {
        matches!(self.device.device_type, torsh_core::device::DeviceType::Cpu)
    }

    /// Check if tensor is on GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self.device.device_type, torsh_core::device::DeviceType::Cuda(_) | torsh_core::device::DeviceType::Metal(_))
    }

    /// Get tensor data as slice
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get mutable tensor data
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// Quantize f32 data to i8 using the given parameters
///
/// This is a standalone utility function for benchmarking purposes
pub fn quantize_to_int8(data: &[f32], params: &QuantizationParams) -> Result<Vec<i8>, TorshError> {
    let scale = params.scale[0];
    let zero_point = params.zero_point[0] as i8;

    let quantized = data
        .iter()
        .map(|&x| {
            let scaled = (x / scale).round() as i32 + zero_point as i32;
            scaled.clamp(-128, 127) as i8
        })
        .collect();

    Ok(quantized)
}

/// Dequantize i8 data back to f32 using the given parameters
///
/// This is a standalone utility function for benchmarking purposes
pub fn dequantize_from_int8(data: &[i8], params: &QuantizationParams) -> Result<Vec<f32>, TorshError> {
    let scale = params.scale[0];
    let zero_point = params.zero_point[0] as i8;

    let dequantized = data
        .iter()
        .map(|&x| {
            (x - zero_point) as f32 * scale
        })
        .collect();

    Ok(dequantized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_dtype_bits() {
        assert_eq!(QuantizedDType::Int8.bits(), 8);
        assert_eq!(QuantizedDType::UInt8.bits(), 8);
        assert_eq!(QuantizedDType::Int16.bits(), 16);
        assert_eq!(QuantizedDType::Int4.bits(), 4);
        assert_eq!(QuantizedDType::Binary.bits(), 1);
        assert_eq!(QuantizedDType::Mixed(vec![4, 8, 16]).bits(), 16);
    }

    #[test]
    fn test_quantized_dtype_signed() {
        assert!(QuantizedDType::Int8.is_signed());
        assert!(!QuantizedDType::UInt8.is_signed());
        assert!(QuantizedDType::Int4.is_signed());
        assert!(!QuantizedDType::UInt4.is_signed());
    }

    #[test]
    fn test_quantized_dtype_value_range() {
        assert_eq!(QuantizedDType::Int8.value_range(), (-128, 127));
        assert_eq!(QuantizedDType::UInt8.value_range(), (0, 255));
        assert_eq!(QuantizedDType::Int4.value_range(), (-8, 7));
        assert_eq!(QuantizedDType::Binary.value_range(), (0, 1));
    }

    #[test]
    fn test_quantization_scheme_properties() {
        assert!(QuantizationScheme::Asymmetric.supports_zero_point());
        assert!(!QuantizationScheme::Symmetric.supports_zero_point());
        assert!(QuantizationScheme::ChannelWise.is_per_channel());
        assert!(QuantizationScheme::BlockWise.is_block_wise());
    }

    #[test]
    fn test_quantization_params_creation() {
        let params = QuantizationParams::int8_symmetric();
        assert_eq!(params.dtype, QuantizedDType::Int8);
        assert_eq!(params.scheme, QuantizationScheme::Symmetric);
        assert_eq!(params.zero_point[0], 0);

        let params = QuantizationParams::uint8_asymmetric();
        assert_eq!(params.dtype, QuantizedDType::UInt8);
        assert_eq!(params.scheme, QuantizationScheme::Asymmetric);
        assert_eq!(params.zero_point[0], 128);
    }

    #[test]
    fn test_quantization_params_validation() {
        let mut params = QuantizationParams::default();
        assert!(params.validate().is_ok());

        // Test invalid scale
        params.scale[0] = 0.0;
        assert!(params.validate().is_err());

        params.scale[0] = f32::NAN;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let params = QuantizationParams::uint8_asymmetric();
        let tensor = QuantizedTensor::new(vec![2, 3, 4], params, Device::cpu().unwrap());
        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();
        assert_eq!(tensor.num_elements(), 24);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert!(tensor.is_cpu());
    }

    #[test]
    fn test_compression_ratio() {
        let params = QuantizationParams::uint8_asymmetric();
        let tensor = QuantizedTensor::new(vec![10, 10], params, Device::cpu().unwrap()).unwrap();

        // 100 elements: FP32 = 400 bytes, UInt8 = 100 bytes
        assert_eq!(tensor.compression_ratio(), 4.0);
    }

    #[test]
    fn test_packed_types() {
        assert!(QuantizedDType::Int4.is_packed());
        assert!(QuantizedDType::Binary.is_packed());
        assert!(!QuantizedDType::Int8.is_packed());

        assert_eq!(QuantizedDType::Int4.packing_factor(), 2);
        assert_eq!(QuantizedDType::Binary.packing_factor(), 8);
        assert_eq!(QuantizedDType::Int8.packing_factor(), 1);
    }
}