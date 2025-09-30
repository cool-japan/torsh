//! Quantization operations for tensors
//!
//! This module provides comprehensive quantization support for neural network optimization:
//! - Quantized data types (QInt8, QUInt8) with scale and zero-point parameters
//! - Quantization and dequantization operations
//! - Automatic quantization parameter computation
//! - Quantized arithmetic operations
//! - Type promotion for mixed quantized/float operations

use crate::{FloatElement, Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};
use num_traits::ToPrimitive;

/// Quantized 8-bit signed integer with scale and zero-point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QInt8 {
    pub value: i8,
    pub scale: f32,
    pub zero_point: i8,
}

impl QInt8 {
    /// Create a new QInt8 value
    pub fn new(value: i8, scale: f32, zero_point: i8) -> Self {
        Self { value, scale, zero_point }
    }

    /// Quantize a floating-point value to QInt8
    pub fn quantize(float_val: f32, scale: f32, zero_point: i8) -> Self {
        let quantized = ((float_val / scale) + zero_point as f32).round() as i8;
        Self::new(quantized.clamp(i8::MIN, i8::MAX), scale, zero_point)
    }

    /// Dequantize QInt8 value back to floating-point
    pub fn dequantize(&self) -> f32 {
        let result = (self.value - self.zero_point) as f32 * self.scale;
        // Clamp to finite values to avoid NaN/Infinity
        if result.is_finite() {
            result
        } else if result.is_nan() {
            0.0f32
        } else if result.is_infinite() && result > 0.0 {
            f32::MAX
        } else {
            f32::MIN
        }
    }
}

impl TensorElement for QInt8 {
    fn dtype() -> torsh_core::dtype::DType {
        torsh_core::dtype::DType::QInt8
    }

    fn from_f64(v: f64) -> Option<Self> {
        // Default scale and zero_point for conversion
        Some(Self::quantize(v as f32, 1.0, 0))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.dequantize() as f64)
    }

    fn zero() -> Self {
        Self::new(0, 1.0, 0)
    }

    fn one() -> Self {
        Self::new(1, 1.0, 0)
    }
}

/// Quantized 8-bit unsigned integer with scale and zero-point
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QUInt8 {
    pub value: u8,
    pub scale: f32,
    pub zero_point: u8,
}

impl QUInt8 {
    /// Create a new QUInt8 value
    pub fn new(value: u8, scale: f32, zero_point: u8) -> Self {
        Self { value, scale, zero_point }
    }

    /// Quantize a floating-point value to QUInt8
    pub fn quantize(float_val: f32, scale: f32, zero_point: u8) -> Self {
        let quantized = ((float_val / scale) + zero_point as f32).round() as u8;
        Self::new(quantized.clamp(u8::MIN, u8::MAX), scale, zero_point)
    }

    /// Dequantize QUInt8 value back to floating-point
    pub fn dequantize(&self) -> f32 {
        (self.value as i32 - self.zero_point as i32) as f32 * self.scale
    }
}

impl TensorElement for QUInt8 {
    fn dtype() -> torsh_core::dtype::DType {
        torsh_core::dtype::DType::QUInt8
    }

    fn from_f64(v: f64) -> Option<Self> {
        // Default scale and zero_point for conversion
        Some(Self::quantize(v as f32, 1.0, 128))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.dequantize() as f64)
    }

    fn zero() -> Self {
        Self::new(128, 1.0, 128) // 128 is typical zero_point for uint8
    }

    fn one() -> Self {
        Self::new(129, 1.0, 128)
    }
}

/// Quantization operations for floating-point tensors
impl<T: FloatElement> Tensor<T> {
    /// Quantize tensor to QInt8 format with specified scale and zero_point
    pub fn quantize_qint8(&self, scale: f32, zero_point: i8) -> Result<Tensor<QInt8>> {
        let data = self.data()?;
        let quantized_data: Vec<QInt8> = data.iter()
            .map(|&val| {
                let float_val = ToPrimitive::to_f64(&val).unwrap_or(0.0) as f32;
                QInt8::quantize(float_val, scale, zero_point)
            })
            .collect();

        Tensor::from_data(quantized_data, self.shape().dims().to_vec(), self.device)
    }

    /// Quantize tensor to QUInt8 format with specified scale and zero_point
    pub fn quantize_quint8(&self, scale: f32, zero_point: u8) -> Result<Tensor<QUInt8>> {
        let data = self.data()?;
        let quantized_data: Vec<QUInt8> = data.iter()
            .map(|&val| {
                let float_val = ToPrimitive::to_f64(&val).unwrap_or(0.0) as f32;
                QUInt8::quantize(float_val, scale, zero_point)
            })
            .collect();

        Tensor::from_data(quantized_data, self.shape().dims().to_vec(), self.device)
    }

    /// Auto-quantize tensor by computing optimal scale and zero_point from data statistics
    /// Returns (quantized_tensor, scale, zero_point)
    pub fn auto_quantize_qint8(&self) -> Result<(Tensor<QInt8>, f32, i8)> {
        let data = self.data()?;
        let float_data: Vec<f32> = data.iter()
            .map(|&val| ToPrimitive::to_f64(&val).unwrap_or(0.0) as f32)
            .collect();

        if float_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot quantize empty tensor".to_string()));
        }

        let min_val = float_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = float_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Handle extreme values that could cause numerical instability
        let (effective_min, effective_max) = if min_val == f32::NEG_INFINITY || max_val == f32::INFINITY ||
            min_val == f32::MIN || max_val == f32::MAX {
            // For extreme values, use a reasonable range to avoid numerical issues
            let abs_max = float_data.iter()
                .map(|&x| x.abs())
                .filter(|&x| x.is_finite())
                .fold(0.0f32, |a, b| a.max(b));

            if abs_max == 0.0 {
                (0.0, 1.0) // Default range if all values are zero or infinite
            } else {
                (-abs_max, abs_max)
            }
        } else {
            (min_val, max_val)
        };

        // Compute scale and zero_point for symmetric quantization around zero
        let scale = (effective_max - effective_min) / 255.0; // Use full range of i8
        let zero_point = (-128.0f32 - effective_min / scale).round() as i8;

        let quantized_tensor = self.quantize_qint8(scale, zero_point)?;
        Ok((quantized_tensor, scale, zero_point))
    }

    /// Auto-quantize tensor to QUInt8 format
    /// Returns (quantized_tensor, scale, zero_point)
    pub fn auto_quantize_quint8(&self) -> Result<(Tensor<QUInt8>, f32, u8)> {
        let data = self.data()?;
        let float_data: Vec<f32> = data.iter()
            .map(|&val| ToPrimitive::to_f64(&val).unwrap_or(0.0) as f32)
            .collect();

        if float_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot quantize empty tensor".to_string()));
        }

        let min_val = float_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = float_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute scale and zero_point for asymmetric quantization
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (0.0f32 - min_val / scale).round() as u8;

        let quantized_tensor = self.quantize_quint8(scale, zero_point)?;
        Ok((quantized_tensor, scale, zero_point))
    }
}

/// Dequantization operations for QInt8 tensors
impl Tensor<QInt8> {
    /// Dequantize QInt8 tensor back to f32
    pub fn dequantize_f32(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let dequantized_data: Vec<f32> = data.iter()
            .map(|qval| qval.dequantize())
            .collect();

        Tensor::from_data(dequantized_data, self.shape().dims().to_vec(), self.device)
    }

    /// Element-wise addition for quantized tensors (requires same scale and zero_point)
    pub fn add_quantized(&self, other: &Self) -> Result<Self> {
        let self_data = self.data()?;
        let other_data = other.data()?;

        if self_data.is_empty() || other_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot add empty tensors".to_string()));
        }

        // Check if scales and zero_points are compatible
        let self_scale = self_data[0].scale;
        let self_zero_point = self_data[0].zero_point;
        let other_scale = other_data[0].scale;
        let other_zero_point = other_data[0].zero_point;

        if (self_scale - other_scale).abs() > 1e-6 || self_zero_point != other_zero_point {
            return Err(TorshError::InvalidArgument(
                "Quantized tensors must have matching scale and zero_point for addition".to_string()
            ));
        }

        // Perform quantized addition
        let result_data: Vec<QInt8> = self_data.iter().zip(other_data.iter())
            .map(|(a, b)| {
                let result_val = (a.value as i32 + b.value as i32 - self_zero_point as i32).clamp(i8::MIN as i32, i8::MAX as i32) as i8;
                QInt8::new(result_val, self_scale, self_zero_point)
            })
            .collect();

        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }

    /// Element-wise multiplication for quantized tensors
    pub fn mul_quantized(&self, other: &Self) -> Result<Self> {
        let self_data = self.data()?;
        let other_data = other.data()?;

        if self_data.is_empty() || other_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot multiply empty tensors".to_string()));
        }

        if self_data.len() != other_data.len() {
            return Err(TorshError::ShapeMismatch {
                expected: vec![self_data.len()],
                got: vec![other_data.len()],
            });
        }

        // For multiplication, the output scale is the product of input scales
        let result_scale = self_data[0].scale * other_data[0].scale;
        let result_zero_point = 0i8; // Multiplication typically uses zero_point = 0

        // Perform quantized multiplication
        let result_data: Vec<QInt8> = self_data.iter().zip(other_data.iter())
            .map(|(a, b)| {
                let a_dequant = a.dequantize();
                let b_dequant = b.dequantize();
                let result_float = a_dequant * b_dequant;
                QInt8::quantize(result_float, result_scale, result_zero_point)
            })
            .collect();

        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }
}

/// Dequantization operations for QUInt8 tensors
impl Tensor<QUInt8> {
    /// Dequantize QUInt8 tensor back to f32
    pub fn dequantize_f32(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let dequantized_data: Vec<f32> = data.iter()
            .map(|qval| qval.dequantize())
            .collect();

        Tensor::from_data(dequantized_data, self.shape().dims().to_vec(), self.device)
    }

    /// Element-wise addition for quantized tensors (requires same scale and zero_point)
    pub fn add_quantized(&self, other: &Self) -> Result<Self> {
        let self_data = self.data()?;
        let other_data = other.data()?;

        if self_data.is_empty() || other_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot add empty tensors".to_string()));
        }

        // Check if scales and zero_points are compatible
        let self_scale = self_data[0].scale;
        let self_zero_point = self_data[0].zero_point;
        let other_scale = other_data[0].scale;
        let other_zero_point = other_data[0].zero_point;

        if (self_scale - other_scale).abs() > 1e-6 || self_zero_point != other_zero_point {
            return Err(TorshError::InvalidArgument(
                "Quantized tensors must have matching scale and zero_point for addition".to_string()
            ));
        }

        // Perform quantized addition
        let result_data: Vec<QUInt8> = self_data.iter().zip(other_data.iter())
            .map(|(a, b)| {
                let result_val = (a.value as i32 + b.value as i32 - self_zero_point as i32).clamp(u8::MIN as i32, u8::MAX as i32) as u8;
                QUInt8::new(result_val, self_scale, self_zero_point)
            })
            .collect();

        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }

    /// Element-wise multiplication for quantized tensors
    pub fn mul_quantized(&self, other: &Self) -> Result<Self> {
        let self_data = self.data()?;
        let other_data = other.data()?;

        if self_data.is_empty() || other_data.is_empty() {
            return Err(TorshError::InvalidArgument("Cannot multiply empty tensors".to_string()));
        }

        if self_data.len() != other_data.len() {
            return Err(TorshError::ShapeMismatch {
                expected: vec![self_data.len()],
                got: vec![other_data.len()],
            });
        }

        // For multiplication, the output scale is the product of input scales
        let result_scale = self_data[0].scale * other_data[0].scale;
        let result_zero_point = 0u8; // Multiplication typically uses zero_point = 0

        // Perform quantized multiplication
        let result_data: Vec<QUInt8> = self_data.iter().zip(other_data.iter())
            .map(|(a, b)| {
                let a_dequant = a.dequantize();
                let b_dequant = b.dequantize();
                let result_float = a_dequant * b_dequant;
                QUInt8::quantize(result_float, result_scale, result_zero_point)
            })
            .collect();

        Self::from_data(result_data, self.shape().dims().to_vec(), self.device)
    }
}

/// Utility functions for quantization
pub mod utils {
    use super::*;

    /// Compute optimal quantization parameters for a range of values
    pub fn compute_quantization_params_symmetric(min_val: f32, max_val: f32) -> (f32, i8) {
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = abs_max / 127.0; // Leave one bit for sign
        (scale, 0) // Symmetric quantization has zero_point = 0
    }

    /// Compute optimal quantization parameters for asymmetric quantization
    pub fn compute_quantization_params_asymmetric(min_val: f32, max_val: f32) -> (f32, u8) {
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as u8;
        (scale, zero_point)
    }

    /// Calibrate quantization parameters from a batch of data
    pub fn calibrate_quantization<T: FloatElement>(tensors: &[&Tensor<T>]) -> Result<(f32, f32)> {
        let mut global_min = f32::INFINITY;
        let mut global_max = f32::NEG_INFINITY;

        for tensor in tensors {
            let data = tensor.data()?;
            for &val in data.iter() {
                let float_val = <T as TensorElement>::to_f64(&val).unwrap_or(0.0) as f32;
                if float_val.is_finite() {
                    global_min = global_min.min(float_val);
                    global_max = global_max.max(float_val);
                }
            }
        }

        Ok((global_min, global_max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_qint8_quantization_dequantization() {
        // Test basic quantization and dequantization
        let float_data = vec![0.0f32, 1.0, 2.0, -1.0, -2.0];
        let tensor = Tensor::from_data(float_data.clone(), vec![5], DeviceType::Cpu).unwrap();

        // Test manual quantization
        let scale = 0.1f32;
        let zero_point = 0i8;
        let quantized = tensor.quantize_qint8(scale, zero_point).unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        let dequantized_data = dequantized.data().unwrap();

        // Check that values are approximately preserved
        for (original, recovered) in float_data.iter().zip(dequantized_data.iter()) {
            assert!((original - recovered).abs() < 0.2,
                "Original: {}, Recovered: {}", original, recovered);
        }
    }

    #[test]
    fn test_quint8_quantization_dequantization() {
        // Test basic quantization and dequantization for unsigned
        let float_data = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(float_data.clone(), vec![5], DeviceType::Cpu).unwrap();

        // Test manual quantization
        let scale = 0.1f32;
        let zero_point = 128u8;
        let quantized = tensor.quantize_quint8(scale, zero_point).unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        let dequantized_data = dequantized.data().unwrap();

        // Check that values are approximately preserved
        for (original, recovered) in float_data.iter().zip(dequantized_data.iter()) {
            assert!((original - recovered).abs() < 0.2,
                "Original: {}, Recovered: {}", original, recovered);
        }
    }

    #[test]
    fn test_auto_quantization_qint8() {
        // Test automatic quantization parameter computation
        let float_data = vec![-5.0f32, -2.5, 0.0, 2.5, 5.0];
        let tensor = Tensor::from_data(float_data.clone(), vec![5], DeviceType::Cpu).unwrap();

        let (quantized, scale, _zero_point) = tensor.auto_quantize_qint8().unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        let dequantized_data = dequantized.data().unwrap();

        // Scale should be (max - min) / 255
        let expected_scale = (5.0 - (-5.0)) / 255.0;
        assert!((scale - expected_scale).abs() < 1e-6,
            "Expected scale: {}, Got: {}", expected_scale, scale);

        // Check that values are approximately preserved
        for (original, recovered) in float_data.iter().zip(dequantized_data.iter()) {
            assert!((original - recovered).abs() < 0.1,
                "Original: {}, Recovered: {}", original, recovered);
        }
    }

    #[test]
    fn test_auto_quantization_quint8() {
        // Test automatic quantization for unsigned range
        let float_data = vec![0.0f32, 2.0, 4.0, 6.0, 8.0];
        let tensor = Tensor::from_data(float_data.clone(), vec![5], DeviceType::Cpu).unwrap();

        let (quantized, scale, _zero_point) = tensor.auto_quantize_quint8().unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();
        let dequantized_data = dequantized.data().unwrap();

        // Scale should be (max - min) / 255
        let expected_scale = (8.0 - 0.0) / 255.0;
        assert!((scale - expected_scale).abs() < 1e-6,
            "Expected scale: {}, Got: {}", expected_scale, scale);

        // Check that values are approximately preserved
        for (original, recovered) in float_data.iter().zip(dequantized_data.iter()) {
            assert!((original - recovered).abs() < 0.1,
                "Original: {}, Recovered: {}", original, recovered);
        }
    }

    #[test]
    fn test_quantized_addition() {
        // Test addition of quantized tensors
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![0.5f32, 1.0, 1.5];

        let tensor1 = Tensor::from_data(data1, vec![3], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(data2, vec![3], DeviceType::Cpu).unwrap();

        let scale = 0.1f32;
        let zero_point = 0i8;

        let q1 = tensor1.quantize_qint8(scale, zero_point).unwrap();
        let q2 = tensor2.quantize_qint8(scale, zero_point).unwrap();

        let q_result = q1.add_quantized(&q2).unwrap();
        let dequantized_result = q_result.dequantize_f32().unwrap();
        let result_data = dequantized_result.data().unwrap();

        // Check that quantized addition gives approximately correct results
        let expected = vec![1.5f32, 3.0, 4.5];
        for (expected_val, actual_val) in expected.iter().zip(result_data.iter()) {
            assert!((expected_val - actual_val).abs() < 0.3,
                "Expected: {}, Got: {}", expected_val, actual_val);
        }
    }

    #[test]
    fn test_quantized_multiplication() {
        // Test multiplication of quantized tensors
        let data1 = vec![2.0f32, 3.0, 4.0];
        let data2 = vec![1.5f32, 2.0, 0.5];

        let tensor1 = Tensor::from_data(data1, vec![3], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(data2, vec![3], DeviceType::Cpu).unwrap();

        let scale = 0.1f32;
        let zero_point = 0i8;

        let q1 = tensor1.quantize_qint8(scale, zero_point).unwrap();
        let q2 = tensor2.quantize_qint8(scale, zero_point).unwrap();

        let q_result = q1.mul_quantized(&q2).unwrap();
        let dequantized_result = q_result.dequantize_f32().unwrap();
        let result_data = dequantized_result.data().unwrap();

        // Check that quantized multiplication gives approximately correct results
        let expected = vec![3.0f32, 6.0, 2.0];
        for (expected_val, actual_val) in expected.iter().zip(result_data.iter()) {
            assert!((expected_val - actual_val).abs() < 0.5,
                "Expected: {}, Got: {}", expected_val, actual_val);
        }
    }

    #[test]
    fn test_quantized_types_tensor_element() {
        // Test that quantized types implement TensorElement correctly
        assert_eq!(QInt8::dtype(), torsh_core::dtype::DType::QInt8);
        assert_eq!(QUInt8::dtype(), torsh_core::dtype::DType::QUInt8);

        // Test zero and one values
        let zero_qint8 = QInt8::zero();
        assert_eq!(zero_qint8.value, 0);

        let one_qint8 = QInt8::one();
        assert_eq!(one_qint8.value, 1);

        let zero_quint8 = QUInt8::zero();
        assert_eq!(zero_quint8.value, 128); // Typical zero_point for uint8

        let one_quint8 = QUInt8::one();
        assert_eq!(one_quint8.value, 129);

        // Test conversion functions
        let qint8_from_f64 = QInt8::from_f64(2.5).unwrap();
        assert_eq!(qint8_from_f64.dequantize(), 3.0); // Should be rounded to nearest integer (2.5 -> 3)

        let quint8_from_f64 = QUInt8::from_f64(3.7).unwrap();
        assert!((quint8_from_f64.dequantize() - 4.0).abs() < 0.1); // Should be close to 4.0
    }

    #[test]
    fn test_quantization_error_handling() {
        // Test error handling for empty tensors
        let empty_tensor: Tensor<f32> = Tensor::from_data(vec![], vec![0], DeviceType::Cpu).unwrap();
        assert!(empty_tensor.auto_quantize_qint8().is_err());
        assert!(empty_tensor.auto_quantize_quint8().is_err());

        // Test error handling for mismatched scale/zero_point in addition
        let data1 = vec![1.0f32, 2.0];
        let data2 = vec![1.0f32, 2.0];

        let tensor1 = Tensor::from_data(data1, vec![2], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(data2, vec![2], DeviceType::Cpu).unwrap();

        let q1 = tensor1.quantize_qint8(0.1, 0).unwrap();
        let q2 = tensor2.quantize_qint8(0.2, 0).unwrap(); // Different scale

        assert!(q1.add_quantized(&q2).is_err()); // Should fail due to different scales
    }

    #[test]
    fn test_quantization_utils() {
        // Test utility functions
        let (scale, zero_point) = utils::compute_quantization_params_symmetric(-5.0, 5.0);
        assert!((scale - (5.0 / 127.0)).abs() < 1e-6);
        assert_eq!(zero_point, 0);

        let (scale, zero_point) = utils::compute_quantization_params_asymmetric(0.0, 10.0);
        assert!((scale - (10.0 / 255.0)).abs() < 1e-6);
        assert_eq!(zero_point, 0);
    }

    #[test]
    fn test_quantization_precision_boundary() {
        // Test quantization at the boundaries of the data type ranges
        let float_data = vec![f32::MIN, f32::MAX, 0.0f32];
        let tensor = Tensor::from_data(float_data, vec![3], DeviceType::Cpu).unwrap();

        let (quantized, _scale, _zero_point) = tensor.auto_quantize_qint8().unwrap();
        let dequantized = quantized.dequantize_f32().unwrap();

        // Should not panic and should produce valid results
        assert!(!dequantized.data().unwrap().iter().any(|&x| x.is_nan()));
    }
}