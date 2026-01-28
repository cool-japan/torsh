// Data Types: Quantized Integer Types
//
// This module implements quantized integer types for efficient neural network
// inference and training. Quantization reduces memory usage and computation
// costs by representing floating point values as integers with associated
// scale and zero-point parameters.

use std::fmt;
use std::hash::{Hash, Hasher};

use crate::dtype::core::DType;
use crate::dtype::traits::TensorElement;

/// Quantized 8-bit signed integer with scale and zero-point parameters
///
/// QInt8 represents floating point values using the formula:
/// `real_value = scale * (quantized_value - zero_point)`
///
/// This allows efficient representation of floating point data in 8-bit
/// signed integers while preserving the ability to recover approximate
/// original values through dequantization.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct QInt8 {
    /// The quantized integer value
    pub value: i8,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point offset for asymmetric quantization
    pub zero_point: i8,
}

impl Eq for QInt8 {}

impl Hash for QInt8 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
        self.scale.to_bits().hash(state);
        self.zero_point.hash(state);
    }
}

impl fmt::Display for QInt8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QInt8(value={}, scale={:.6}, zero_point={})",
            self.value, self.scale, self.zero_point
        )
    }
}

impl QInt8 {
    /// Create a new quantized int8 value
    ///
    /// # Arguments
    /// * `value` - The quantized integer value
    /// * `scale` - Scale factor for dequantization (must be positive)
    /// * `zero_point` - Zero point offset for asymmetric quantization
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::quantized::QInt8;
    ///
    /// let q = QInt8::new(100, 0.1, 5);
    /// assert_eq!(q.value, 100);
    /// assert_eq!(q.scale, 0.1);
    /// assert_eq!(q.zero_point, 5);
    /// ```
    pub fn new(value: i8, scale: f32, zero_point: i8) -> Self {
        debug_assert!(scale > 0.0, "Scale must be positive");
        QInt8 {
            value,
            scale,
            zero_point,
        }
    }

    /// Create a QInt8 with symmetric quantization (zero_point = 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::quantized::QInt8;
    ///
    /// let q = QInt8::symmetric(100, 0.1);
    /// assert_eq!(q.zero_point, 0);
    /// ```
    pub fn symmetric(value: i8, scale: f32) -> Self {
        Self::new(value, scale, 0)
    }

    /// Dequantize to f32 using the stored scale and zero-point
    ///
    /// Formula: `real_value = scale * (value - zero_point)`
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::quantized::QInt8;
    ///
    /// let q = QInt8::new(110, 0.1, 10);
    /// let dequantized = q.dequantize();
    /// assert!((dequantized - 10.0).abs() < f32::EPSILON);
    /// ```
    pub fn dequantize(&self) -> f32 {
        self.scale * (self.value as f32 - self.zero_point as f32)
    }

    /// Dequantize to f64 with higher precision
    pub fn dequantize_f64(&self) -> f64 {
        self.scale as f64 * (self.value as f64 - self.zero_point as f64)
    }

    /// Quantize a floating point value to QInt8
    ///
    /// Formula: `quantized_value = round(value / scale + zero_point)`
    /// The result is clamped to the valid i8 range.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_core::dtype::quantized::QInt8;
    ///
    /// let q = QInt8::quantize(2.5, 0.1, 5);
    /// assert_eq!(q.value, 30); // round(2.5 / 0.1 + 5) = round(30)
    /// ```
    pub fn quantize(value: f32, scale: f32, zero_point: i8) -> Self {
        debug_assert!(scale > 0.0, "Scale must be positive");
        let quantized = (value / scale + zero_point as f32).round() as i32;
        let clamped = quantized.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        QInt8::new(clamped, scale, zero_point)
    }

    /// Quantize with automatic scale and zero-point calculation
    ///
    /// Calculates optimal scale and zero-point to represent the given range
    /// [min_value, max_value] using the full i8 range.
    pub fn quantize_range(value: f32, min_value: f32, max_value: f32) -> Self {
        let (scale, zero_point) = calculate_qint8_params(min_value, max_value);
        Self::quantize(value, scale, zero_point)
    }

    /// Check if this is a symmetric quantization (zero_point == 0)
    pub fn is_symmetric(&self) -> bool {
        self.zero_point == 0
    }

    /// Get the effective range this quantization can represent
    pub fn representable_range(&self) -> (f32, f32) {
        let min_q = i8::MIN as f32;
        let max_q = i8::MAX as f32;
        let min_val = self.scale * (min_q - self.zero_point as f32);
        let max_val = self.scale * (max_q - self.zero_point as f32);
        (min_val, max_val)
    }

    /// Calculate the quantization error for a given float value
    pub fn quantization_error(&self, original: f32) -> f32 {
        let quantized = Self::quantize(original, self.scale, self.zero_point);
        let restored = quantized.dequantize();
        (restored - original).abs()
    }

    /// Create a zero value with the same quantization parameters
    pub fn zero_like(&self) -> Self {
        QInt8::new(self.zero_point, self.scale, self.zero_point)
    }

    /// Create a one value with the same quantization parameters
    pub fn one_like(&self) -> Self {
        let one_quantized = (1.0 / self.scale + self.zero_point as f32).round() as i32;
        let clamped = one_quantized.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        QInt8::new(clamped, self.scale, self.zero_point)
    }
}

impl TensorElement for QInt8 {
    fn dtype() -> DType {
        DType::QInt8
    }

    fn from_f64(v: f64) -> Option<Self> {
        // Default scale and zero-point for conversion
        Some(QInt8::quantize(v as f32, 1.0, 0))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.dequantize_f64())
    }

    fn zero() -> Self {
        QInt8::new(0, 1.0, 0)
    }

    fn one() -> Self {
        QInt8::new(1, 1.0, 0)
    }

    fn is_zero(&self) -> bool {
        self.value == 0 || self.dequantize() == 0.0
    }

    fn is_one(&self) -> bool {
        (self.dequantize() - 1.0).abs() < 1e-6
    }
}

/// Quantized 8-bit unsigned integer with scale and zero-point parameters
///
/// QUInt8 represents floating point values using the formula:
/// `real_value = scale * (quantized_value - zero_point)`
///
/// The unsigned variant can represent non-negative ranges more efficiently
/// and is commonly used for activations in neural networks.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct QUInt8 {
    /// The quantized integer value
    pub value: u8,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point offset for asymmetric quantization
    pub zero_point: u8,
}

impl Eq for QUInt8 {}

impl Hash for QUInt8 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
        self.scale.to_bits().hash(state);
        self.zero_point.hash(state);
    }
}

impl fmt::Display for QUInt8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QUInt8(value={}, scale={:.6}, zero_point={})",
            self.value, self.scale, self.zero_point
        )
    }
}

impl QUInt8 {
    /// Create a new quantized uint8 value
    ///
    /// # Arguments
    /// * `value` - The quantized integer value
    /// * `scale` - Scale factor for dequantization (must be positive)
    /// * `zero_point` - Zero point offset for asymmetric quantization
    pub fn new(value: u8, scale: f32, zero_point: u8) -> Self {
        debug_assert!(scale > 0.0, "Scale must be positive");
        QUInt8 {
            value,
            scale,
            zero_point,
        }
    }

    /// Create a QUInt8 with symmetric quantization (zero_point = 0)
    pub fn symmetric(value: u8, scale: f32) -> Self {
        Self::new(value, scale, 0)
    }

    /// Dequantize to f32 using the stored scale and zero-point
    ///
    /// Formula: `real_value = scale * (value - zero_point)`
    pub fn dequantize(&self) -> f32 {
        self.scale * (self.value as i32 - self.zero_point as i32) as f32
    }

    /// Dequantize to f64 with higher precision
    pub fn dequantize_f64(&self) -> f64 {
        self.scale as f64 * (self.value as i32 - self.zero_point as i32) as f64
    }

    /// Quantize a floating point value to QUInt8
    ///
    /// Formula: `quantized_value = round(value / scale + zero_point)`
    /// The result is clamped to the valid u8 range.
    pub fn quantize(value: f32, scale: f32, zero_point: u8) -> Self {
        debug_assert!(scale > 0.0, "Scale must be positive");
        let quantized = (value / scale + zero_point as f32).round() as i32;
        let clamped = quantized.clamp(0, u8::MAX as i32) as u8;
        QUInt8::new(clamped, scale, zero_point)
    }

    /// Quantize with automatic scale and zero-point calculation
    pub fn quantize_range(value: f32, min_value: f32, max_value: f32) -> Self {
        let (scale, zero_point) = calculate_quint8_params(min_value, max_value);
        Self::quantize(value, scale, zero_point)
    }

    /// Check if this is a symmetric quantization (zero_point == 0)
    pub fn is_symmetric(&self) -> bool {
        self.zero_point == 0
    }

    /// Get the effective range this quantization can represent
    pub fn representable_range(&self) -> (f32, f32) {
        let min_q = 0.0;
        let max_q = u8::MAX as f32;
        let min_val = self.scale * (min_q - self.zero_point as f32);
        let max_val = self.scale * (max_q - self.zero_point as f32);
        (min_val, max_val)
    }

    /// Calculate the quantization error for a given float value
    pub fn quantization_error(&self, original: f32) -> f32 {
        let quantized = Self::quantize(original, self.scale, self.zero_point);
        let restored = quantized.dequantize();
        (restored - original).abs()
    }

    /// Create a zero value with the same quantization parameters
    pub fn zero_like(&self) -> Self {
        QUInt8::new(self.zero_point, self.scale, self.zero_point)
    }

    /// Create a one value with the same quantization parameters
    pub fn one_like(&self) -> Self {
        let one_quantized = (1.0 / self.scale + self.zero_point as f32).round() as i32;
        let clamped = one_quantized.clamp(0, u8::MAX as i32) as u8;
        QUInt8::new(clamped, self.scale, self.zero_point)
    }
}

impl TensorElement for QUInt8 {
    fn dtype() -> DType {
        DType::QUInt8
    }

    fn from_f64(v: f64) -> Option<Self> {
        // Default scale and zero-point for conversion
        Some(QUInt8::quantize(v as f32, 1.0, 0))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.dequantize_f64())
    }

    fn zero() -> Self {
        QUInt8::new(0, 1.0, 0)
    }

    fn one() -> Self {
        QUInt8::new(1, 1.0, 0)
    }

    fn is_zero(&self) -> bool {
        self.value == self.zero_point || self.dequantize() == 0.0
    }

    fn is_one(&self) -> bool {
        (self.dequantize() - 1.0).abs() < 1e-6
    }
}

/// Calculate optimal quantization parameters for QInt8
///
/// Returns (scale, zero_point) that maps [min_value, max_value] to the full i8 range.
pub fn calculate_qint8_params(min_value: f32, max_value: f32) -> (f32, i8) {
    assert!(max_value >= min_value, "max_value must be >= min_value");

    if max_value == min_value {
        // Degenerate case - all values are the same
        return (1.0, 0);
    }

    let qmin = i8::MIN as f32;
    let qmax = i8::MAX as f32;

    // Calculate scale to map the range to quantized range
    // Use a slightly larger scale to ensure full coverage after rounding
    let scale = (max_value - min_value) / (qmax - qmin - 1.0); // Reserve one level for safety

    // Calculate zero point
    // We want: min_value ≈ scale * (qmin - zero_point)
    // So: zero_point ≈ qmin - min_value / scale
    let zero_point_real = qmin - min_value / scale;
    let zero_point = zero_point_real.round().clamp(qmin, qmax) as i8;

    (scale, zero_point)
}

/// Calculate optimal quantization parameters for QUInt8
///
/// Returns (scale, zero_point) that maps [min_value, max_value] to the full u8 range.
pub fn calculate_quint8_params(min_value: f32, max_value: f32) -> (f32, u8) {
    assert!(max_value >= min_value, "max_value must be >= min_value");

    if max_value == min_value {
        // Degenerate case - all values are the same
        return (1.0, 0);
    }

    let qmin = 0.0;
    let qmax = u8::MAX as f32;

    // Calculate scale to map the range to quantized range
    let scale = (max_value - min_value) / (qmax - qmin);

    // Calculate zero point
    let zero_point_real = qmin - min_value / scale;
    let zero_point = zero_point_real.round().clamp(qmin, qmax) as u8;

    (scale, zero_point)
}

/// Quantization schemes for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationScheme {
    /// Symmetric quantization with zero_point = 0
    Symmetric,
    /// Asymmetric quantization with calculated zero_point
    Asymmetric,
    /// Per-channel quantization (different params per channel)
    PerChannel,
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Quantization scheme to use
    pub scheme: QuantizationScheme,
    /// Whether to use signed or unsigned quantization
    pub signed: bool,
    /// Number of bits (currently only 8 is supported)
    pub bits: u8,
    /// Observer for calculating quantization parameters
    pub observer: QuantizationObserver,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::Asymmetric,
            signed: true,
            bits: 8,
            observer: QuantizationObserver::default(),
        }
    }
}

/// Observer for calculating quantization parameters from data
#[derive(Debug, Clone)]
pub struct QuantizationObserver {
    min_val: f32,
    max_val: f32,
    count: usize,
}

impl Default for QuantizationObserver {
    fn default() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            count: 0,
        }
    }
}

impl QuantizationObserver {
    /// Create a new observer
    pub fn new() -> Self {
        Self::default()
    }

    /// Update the observer with a new value
    pub fn update(&mut self, value: f32) {
        if value.is_finite() {
            self.min_val = self.min_val.min(value);
            self.max_val = self.max_val.max(value);
            self.count += 1;
        }
    }

    /// Update the observer with multiple values
    pub fn update_batch(&mut self, values: &[f32]) {
        for &value in values {
            self.update(value);
        }
    }

    /// Calculate quantization parameters for QInt8
    pub fn calculate_qint8_params(&self) -> Option<(f32, i8)> {
        if self.count == 0 {
            return None;
        }
        Some(calculate_qint8_params(self.min_val, self.max_val))
    }

    /// Calculate quantization parameters for QUInt8
    pub fn calculate_quint8_params(&self) -> Option<(f32, u8)> {
        if self.count == 0 {
            return None;
        }
        Some(calculate_quint8_params(self.min_val, self.max_val))
    }

    /// Reset the observer
    pub fn reset(&mut self) {
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
        self.count = 0;
    }

    /// Get the observed range
    pub fn range(&self) -> Option<(f32, f32)> {
        if self.count == 0 {
            None
        } else {
            Some((self.min_val, self.max_val))
        }
    }

    /// Calculate quantization parameters for QInt32
    pub fn calculate_qint32_params(&self) -> Option<(f32, i32)> {
        if self.count == 0 {
            return None;
        }
        Some(calculate_qint32_params(self.min_val, self.max_val))
    }
}

/// Quantized 32-bit signed integer with scale and zero-point parameters
///
/// QInt32 provides higher precision quantization compared to QInt8, offering
/// a wider dynamic range while still using less memory than f32/f64.
/// This is useful for intermediate computations in neural networks where
/// higher precision is needed but full floating point is not required.
///
/// Formula: `real_value = scale * (quantized_value - zero_point)`
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct QInt32 {
    /// The quantized integer value
    pub value: i32,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point offset for asymmetric quantization
    pub zero_point: i32,
}

impl Eq for QInt32 {}

impl Hash for QInt32 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
        self.scale.to_bits().hash(state);
        self.zero_point.hash(state);
    }
}

impl fmt::Display for QInt32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QInt32(value={}, scale={:.6}, zero_point={})",
            self.value, self.scale, self.zero_point
        )
    }
}

impl QInt32 {
    /// Create a new quantized int32 value
    ///
    /// # Arguments
    /// * `value` - The quantized integer value
    /// * `scale` - Scale factor for dequantization (must be positive)
    /// * `zero_point` - Zero point offset for asymmetric quantization
    pub fn new(value: i32, scale: f32, zero_point: i32) -> Self {
        debug_assert!(scale > 0.0, "Scale must be positive");
        QInt32 {
            value,
            scale,
            zero_point,
        }
    }

    /// Create a QInt32 with symmetric quantization (zero_point = 0)
    pub fn symmetric(value: i32, scale: f32) -> Self {
        Self::new(value, scale, 0)
    }

    /// Dequantize to f32 using the stored scale and zero-point
    ///
    /// Formula: `real_value = scale * (value - zero_point)`
    pub fn dequantize(&self) -> f32 {
        self.scale * (self.value as f64 - self.zero_point as f64) as f32
    }

    /// Dequantize to f64 with higher precision
    pub fn dequantize_f64(&self) -> f64 {
        self.scale as f64 * (self.value as f64 - self.zero_point as f64)
    }

    /// Quantize a floating point value to QInt32
    ///
    /// Formula: `quantized_value = round(value / scale + zero_point)`
    /// The result is clamped to the valid i32 range.
    pub fn quantize(value: f32, scale: f32, zero_point: i32) -> Self {
        debug_assert!(scale > 0.0, "Scale must be positive");
        let quantized = (value as f64 / scale as f64 + zero_point as f64).round();
        let clamped = quantized.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        QInt32::new(clamped, scale, zero_point)
    }

    /// Quantize with automatic scale and zero-point calculation
    pub fn quantize_range(value: f32, min_value: f32, max_value: f32) -> Self {
        let (scale, zero_point) = calculate_qint32_params(min_value, max_value);
        Self::quantize(value, scale, zero_point)
    }

    /// Check if this is a symmetric quantization (zero_point == 0)
    pub fn is_symmetric(&self) -> bool {
        self.zero_point == 0
    }

    /// Get the effective range this quantization can represent
    pub fn representable_range(&self) -> (f32, f32) {
        let min_q = i32::MIN as f64;
        let max_q = i32::MAX as f64;
        let min_val = (self.scale as f64 * (min_q - self.zero_point as f64)) as f32;
        let max_val = (self.scale as f64 * (max_q - self.zero_point as f64)) as f32;
        (min_val, max_val)
    }

    /// Calculate the quantization error for a given float value
    pub fn quantization_error(&self, original: f32) -> f32 {
        let quantized = Self::quantize(original, self.scale, self.zero_point);
        let restored = quantized.dequantize();
        (restored - original).abs()
    }

    /// Create a zero value with the same quantization parameters
    pub fn zero_like(&self) -> Self {
        QInt32::new(self.zero_point, self.scale, self.zero_point)
    }

    /// Create a one value with the same quantization parameters
    pub fn one_like(&self) -> Self {
        let one_quantized = (1.0 / self.scale as f64 + self.zero_point as f64).round();
        let clamped = one_quantized.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        QInt32::new(clamped, self.scale, self.zero_point)
    }
}

impl TensorElement for QInt32 {
    fn dtype() -> DType {
        DType::QInt32
    }

    fn from_f64(v: f64) -> Option<Self> {
        // Default scale and zero-point for conversion
        Some(QInt32::quantize(v as f32, 1.0, 0))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.dequantize_f64())
    }

    fn zero() -> Self {
        QInt32::new(0, 1.0, 0)
    }

    fn one() -> Self {
        QInt32::new(1, 1.0, 0)
    }

    fn is_zero(&self) -> bool {
        self.value == 0 || self.dequantize() == 0.0
    }

    fn is_one(&self) -> bool {
        (self.dequantize() - 1.0).abs() < 1e-6
    }
}

/// Calculate optimal quantization parameters for QInt32
///
/// Returns (scale, zero_point) that maps [min_value, max_value] to the full i32 range.
pub fn calculate_qint32_params(min_value: f32, max_value: f32) -> (f32, i32) {
    assert!(max_value >= min_value, "max_value must be >= min_value");

    if max_value == min_value {
        // Degenerate case - all values are the same
        return (1.0, 0);
    }

    let qmin = i32::MIN as f64;
    let qmax = i32::MAX as f64;

    // Calculate scale to map the range to quantized range
    let scale = ((max_value - min_value) as f64 / (qmax - qmin)) as f32;

    // Calculate zero point
    let zero_point_real = qmin - (min_value as f64 / scale as f64);
    let zero_point = zero_point_real.round().clamp(qmin, qmax) as i32;

    (scale, zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qint8_basic_operations() {
        let q = QInt8::new(100, 0.1, 5);
        assert_eq!(q.value, 100);
        assert_eq!(q.scale, 0.1);
        assert_eq!(q.zero_point, 5);

        // Test dequantization
        let dequantized = q.dequantize();
        let expected = 0.1 * (100.0 - 5.0); // 9.5
        assert!((dequantized - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_qint8_quantization() {
        let original = 2.5;
        let scale = 0.1;
        let zero_point = 5;

        let quantized = QInt8::quantize(original, scale, zero_point);
        let expected_value = (original / scale + zero_point as f32).round() as i8;
        assert_eq!(quantized.value, expected_value);

        // Test round-trip
        let restored = quantized.dequantize();
        assert!((restored - original).abs() < 0.1); // Allow for quantization error
    }

    #[test]
    fn test_qint8_range_quantization() {
        let min_val = -10.0;
        let max_val = 10.0;
        let test_value = 5.0;

        let quantized = QInt8::quantize_range(test_value, min_val, max_val);
        let restored = quantized.dequantize();

        // Should be close to original
        assert!((restored - test_value).abs() < 0.2);

        // Test range
        let (min_repr, max_repr) = quantized.representable_range();
        assert!(min_repr <= min_val);
        assert!(max_repr >= max_val);
    }

    #[test]
    fn test_quint8_basic_operations() {
        let q = QUInt8::new(150, 0.2, 10);
        assert_eq!(q.value, 150);
        assert_eq!(q.scale, 0.2);
        assert_eq!(q.zero_point, 10);

        // Test dequantization
        let dequantized = q.dequantize();
        let expected = 0.2 * (150 - 10) as f32; // 28.0
        assert!((dequantized - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_quint8_quantization() {
        let original = 5.5;
        let scale = 0.2;
        let zero_point = 10;

        let quantized = QUInt8::quantize(original, scale, zero_point);
        let expected_value = (original / scale + zero_point as f32).round() as u8;
        assert_eq!(quantized.value, expected_value);

        // Test round-trip
        let restored = quantized.dequantize();
        assert!((restored - original).abs() < 0.3); // Allow for quantization error
    }

    #[test]
    fn test_symmetric_quantization() {
        let q_int8 = QInt8::symmetric(50, 0.1);
        assert_eq!(q_int8.zero_point, 0);
        assert!(q_int8.is_symmetric());

        let q_uint8 = QUInt8::symmetric(100, 0.1);
        assert_eq!(q_uint8.zero_point, 0);
        assert!(q_uint8.is_symmetric());
    }

    #[test]
    fn test_parameter_calculation() {
        let min_val = -5.0;
        let max_val = 10.0;

        // Test QInt8 parameter calculation
        let (scale_i8, zero_point_i8) = calculate_qint8_params(min_val, max_val);
        assert!(scale_i8 > 0.0);

        // Test QUInt8 parameter calculation
        let (scale_u8, _zero_point_u8) = calculate_quint8_params(min_val, max_val);
        assert!(scale_u8 > 0.0);

        // Verify the parameters work
        let q_i8_min = QInt8::quantize(min_val, scale_i8, zero_point_i8);
        let q_i8_max = QInt8::quantize(max_val, scale_i8, zero_point_i8);

        assert!((q_i8_min.dequantize() - min_val).abs() < 0.1);
        assert!((q_i8_max.dequantize() - max_val).abs() < 0.1);
    }

    #[test]
    fn test_quantization_observer() {
        let mut observer = QuantizationObserver::new();

        // Update with some values
        let values = vec![-2.5, -1.0, 0.0, 1.5, 3.0];
        observer.update_batch(&values);

        // Check range
        let (min_obs, max_obs) = observer.range().unwrap();
        assert_eq!(min_obs, -2.5);
        assert_eq!(max_obs, 3.0);

        // Calculate parameters
        let (scale, _zero_point) = observer.calculate_qint8_params().unwrap();
        assert!(scale > 0.0);

        // Test reset
        observer.reset();
        assert!(observer.range().is_none());
    }

    #[test]
    fn test_tensor_element_trait() {
        // Test QInt8
        assert_eq!(QInt8::dtype(), DType::QInt8);
        let zero_qint8 = QInt8::zero();
        assert_eq!(zero_qint8.value, 0);
        let one_qint8 = QInt8::one();
        assert_eq!(one_qint8.value, 1);

        // Test conversions
        let from_f64 = QInt8::from_f64(5.5).expect("f64 conversion should succeed");
        let to_f64 = from_f64.to_f64().expect("f64 conversion should succeed");
        assert!((to_f64 - 5.5).abs() < 1.0); // Allow for quantization error

        // Test QUInt8
        assert_eq!(QUInt8::dtype(), DType::QUInt8);
        let zero_quint8 = QUInt8::zero();
        assert_eq!(zero_quint8.value, 0);
        let one_quint8 = QUInt8::one();
        assert_eq!(one_quint8.value, 1);
    }

    #[test]
    fn test_quantization_error() {
        let scale = 0.1;
        let zero_point = 0;
        let original = std::f32::consts::PI;

        let q = QInt8::quantize(original, scale, zero_point);
        let error = q.quantization_error(original);

        // Error should be less than half the quantization step
        assert!(error < scale / 2.0);
    }

    #[test]
    fn test_edge_cases() {
        // Test clamping
        let large_val = 1000.0;
        let q_large = QInt8::quantize(large_val, 1.0, 0);
        assert_eq!(q_large.value, i8::MAX);

        let small_val = -1000.0;
        let q_small = QInt8::quantize(small_val, 1.0, 0);
        assert_eq!(q_small.value, i8::MIN);

        // Test degenerate range
        let (scale, zero_point) = calculate_qint8_params(5.0, 5.0);
        assert_eq!(scale, 1.0);
        assert_eq!(zero_point, 0);
    }

    #[test]
    fn test_zero_one_like() {
        let q = QInt8::new(50, 0.2, 10);

        let zero_like = q.zero_like();
        assert_eq!(zero_like.value, q.zero_point);
        assert_eq!(zero_like.scale, q.scale);

        let one_like = q.one_like();
        assert_eq!(one_like.scale, q.scale);
        assert_eq!(one_like.zero_point, q.zero_point);
        assert!((one_like.dequantize() - 1.0).abs() < 0.3);
    }

    #[test]
    fn test_display_formatting() {
        let q_int8 = QInt8::new(42, 0.125, -5);
        let display_str = format!("{}", q_int8);
        assert!(display_str.contains("QInt8"));
        assert!(display_str.contains("42"));
        assert!(display_str.contains("0.125"));
        assert!(display_str.contains("-5"));

        let q_uint8 = QUInt8::new(200, 0.5, 128);
        let display_str = format!("{}", q_uint8);
        assert!(display_str.contains("QUInt8"));
        assert!(display_str.contains("200"));
        assert!(display_str.contains("0.5"));
        assert!(display_str.contains("128"));
    }

    #[test]
    fn test_hash_and_equality() {
        let q1 = QInt8::new(42, 0.1, 5);
        let q2 = QInt8::new(42, 0.1, 5);
        let q3 = QInt8::new(43, 0.1, 5);

        assert_eq!(q1, q2);
        assert_ne!(q1, q3);

        // Test that equal values have equal hashes
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(q1, "value1");
        map.insert(q2, "value2"); // Should overwrite
        map.insert(q3, "value3");

        assert_eq!(map.len(), 2);
        assert_eq!(map[&q1], "value2"); // q2 overwrote q1's value
    }
}
