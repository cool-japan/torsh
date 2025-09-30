//! Core quantization data types and schemes
//!
//! This module defines the fundamental data types and quantization schemes
//! used throughout the quantization system. It provides the foundation
//! for all quantization operations in the ToRSh backend.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Quantization data types
///
/// Defines the various quantized data types supported by the system,
/// from 1-bit binary quantization to 16-bit integer types, including
/// mixed precision support for advanced quantization strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizedDType {
    /// 8-bit signed integer quantization
    ///
    /// Range: -128 to 127
    /// Most common quantization type for post-training quantization
    Int8,
    /// 8-bit unsigned integer quantization
    ///
    /// Range: 0 to 255
    /// Common for activations and weights with ReLU networks
    UInt8,
    /// 16-bit signed integer quantization
    ///
    /// Range: -32768 to 32767
    /// Higher precision quantization for sensitive models
    Int16,
    /// 16-bit unsigned integer quantization
    ///
    /// Range: 0 to 65535
    /// Used when higher precision is needed without negative values
    UInt16,
    /// 4-bit signed integer quantization (packed)
    ///
    /// Range: -8 to 7
    /// Ultra-low precision for extreme compression
    Int4,
    /// 4-bit unsigned integer quantization (packed)
    ///
    /// Range: 0 to 15
    /// Extreme compression for non-negative values
    UInt4,
    /// 1-bit binary quantization (packed)
    ///
    /// Range: 0 to 1
    /// Maximum compression, used in binary neural networks
    Binary,
    /// Mixed precision with different bits per channel
    ///
    /// Allows different quantization precision for each channel,
    /// enabling fine-grained control over accuracy vs compression trade-offs
    Mixed(Vec<u8>),
}

impl QuantizedDType {
    /// Get the number of bits for this quantization type
    ///
    /// Returns the bit width for the quantization type. For mixed precision,
    /// returns the maximum bits used across all channels.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizedDType;
    ///
    /// assert_eq!(QuantizedDType::Int8.bits(), 8);
    /// assert_eq!(QuantizedDType::Int4.bits(), 4);
    /// assert_eq!(QuantizedDType::Binary.bits(), 1);
    /// ```
    pub fn bits(&self) -> u8 {
        match self {
            QuantizedDType::Int8 | QuantizedDType::UInt8 => 8,
            QuantizedDType::Int16 | QuantizedDType::UInt16 => 16,
            QuantizedDType::Int4 | QuantizedDType::UInt4 => 4,
            QuantizedDType::Binary => 1,
            QuantizedDType::Mixed(bits) => bits.iter().max().copied().unwrap_or(8),
        }
    }

    /// Check if this quantization type uses signed integers
    ///
    /// Returns true for signed integer types (Int8, Int16, Int4),
    /// false for unsigned types and binary quantization.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizedDType;
    ///
    /// assert!(QuantizedDType::Int8.is_signed());
    /// assert!(!QuantizedDType::UInt8.is_signed());
    /// assert!(!QuantizedDType::Binary.is_signed());
    /// ```
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            QuantizedDType::Int8 | QuantizedDType::Int16 | QuantizedDType::Int4
        )
    }

    /// Get the range of representable values for this quantization type
    ///
    /// Returns a tuple (min_value, max_value) representing the full
    /// range of values that can be represented with this quantization type.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizedDType;
    ///
    /// assert_eq!(QuantizedDType::Int8.value_range(), (-128, 127));
    /// assert_eq!(QuantizedDType::UInt8.value_range(), (0, 255));
    /// assert_eq!(QuantizedDType::Binary.value_range(), (0, 1));
    /// ```
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

    /// Get the number of distinct values representable by this type
    ///
    /// Returns the total number of unique values that can be represented.
    /// This is useful for calculating quantization step sizes and analyzing
    /// quantization error.
    ///
    /// # Examples
    ///
    /// ```
    /// use torsh_backend::quantization::QuantizedDType;
    ///
    /// assert_eq!(QuantizedDType::Int8.num_values(), 256);
    /// assert_eq!(QuantizedDType::UInt4.num_values(), 16);
    /// assert_eq!(QuantizedDType::Binary.num_values(), 2);
    /// ```
    pub fn num_values(&self) -> u64 {
        let (min, max) = self.value_range();
        (max - min + 1) as u64
    }

    /// Check if this type supports sub-byte packing
    ///
    /// Returns true for quantization types that use less than 8 bits
    /// and can be packed multiple values per byte for storage efficiency.
    pub fn is_sub_byte(&self) -> bool {
        matches!(
            self,
            QuantizedDType::Int4 | QuantizedDType::UInt4 | QuantizedDType::Binary
        )
    }

    /// Get the storage efficiency factor
    ///
    /// Returns how many values can be packed into a single byte.
    /// For 8-bit and 16-bit types, this is less than 1.
    pub fn values_per_byte(&self) -> f32 {
        8.0 / self.bits() as f32
    }

    /// Check if this type is compatible with another for mixed operations
    ///
    /// Returns true if the two types can be safely mixed in operations
    /// without explicit conversion. Generally, types of the same bit width
    /// and signedness are compatible.
    pub fn is_compatible_with(&self, other: &QuantizedDType) -> bool {
        match (self, other) {
            // Same types are always compatible
            (a, b) if a == b => true,
            // Same bit width and signedness
            (QuantizedDType::Int8, QuantizedDType::UInt8)
            | (QuantizedDType::UInt8, QuantizedDType::Int8) => false, // Different signedness
            (QuantizedDType::Int16, QuantizedDType::UInt16)
            | (QuantizedDType::UInt16, QuantizedDType::Int16) => false, // Different signedness
            (QuantizedDType::Int4, QuantizedDType::UInt4)
            | (QuantizedDType::UInt4, QuantizedDType::Int4) => false, // Different signedness
            // Mixed precision is compatible with anything of same max bits
            (QuantizedDType::Mixed(bits), other) | (other, QuantizedDType::Mixed(bits)) => {
                let max_bits = bits.iter().max().copied().unwrap_or(8);
                max_bits == other.bits()
            }
            _ => false,
        }
    }
}

/// Quantization schemes for different use cases
///
/// Defines various quantization strategies that determine how floating-point
/// values are mapped to quantized integer values. Each scheme offers different
/// trade-offs between accuracy, implementation complexity, and hardware support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationScheme {
    /// Linear/uniform quantization
    ///
    /// Maps floating-point values to quantized values using a linear mapping
    /// with constant step size. This is the most common and hardware-friendly
    /// quantization scheme.
    ///
    /// Formula: q = round((x - zero_point) / scale)
    Linear,

    /// Logarithmic quantization for dynamic range
    ///
    /// Uses logarithmic spacing between quantization levels, which can better
    /// preserve small values while still representing large dynamic ranges.
    /// Useful for weights with exponential distributions.
    Logarithmic,

    /// Symmetric quantization (zero point = 0)
    ///
    /// A special case of linear quantization where the zero floating-point
    /// value maps exactly to zero in the quantized representation. This
    /// simplifies hardware implementation and often provides better accuracy.
    ///
    /// Formula: q = round(x / scale)
    Symmetric,

    /// Asymmetric quantization with custom zero point
    ///
    /// Uses a custom zero point that doesn't necessarily align with zero,
    /// allowing better utilization of the quantization range for data
    /// that doesn't center around zero.
    ///
    /// Formula: q = round(x / scale + zero_point)
    Asymmetric,

    /// Block-wise quantization
    ///
    /// Applies quantization parameters independently to blocks of the tensor,
    /// allowing for better adaptation to local value distributions. Each block
    /// has its own scale and zero point.
    BlockWise,

    /// Channel-wise quantization
    ///
    /// Applies different quantization parameters to each channel (output channel
    /// for weights, feature channel for activations). This provides better
    /// accuracy for models with varying channel sensitivities.
    ChannelWise,
}

impl QuantizationScheme {
    /// Check if this scheme requires per-channel parameters
    ///
    /// Returns true for schemes that need different parameters for each
    /// channel or block, affecting memory requirements and computation.
    pub fn is_per_channel(&self) -> bool {
        matches!(
            self,
            QuantizationScheme::ChannelWise | QuantizationScheme::BlockWise
        )
    }

    /// Check if this scheme uses symmetric quantization
    ///
    /// Returns true for schemes where the quantization is symmetric around zero,
    /// which can enable certain hardware optimizations.
    pub fn is_symmetric(&self) -> bool {
        matches!(self, QuantizationScheme::Symmetric)
    }

    /// Check if this scheme requires zero point parameters
    ///
    /// Returns false only for symmetric schemes where zero point is always zero.
    pub fn requires_zero_point(&self) -> bool {
        !self.is_symmetric()
    }

    /// Get the computational complexity factor for this scheme
    ///
    /// Returns a relative complexity factor (1.0 = baseline linear quantization)
    /// indicating the computational overhead of this quantization scheme.
    pub fn complexity_factor(&self) -> f32 {
        match self {
            QuantizationScheme::Linear | QuantizationScheme::Symmetric => 1.0,
            QuantizationScheme::Asymmetric => 1.1,
            QuantizationScheme::Logarithmic => 1.5,
            QuantizationScheme::BlockWise => 1.3,
            QuantizationScheme::ChannelWise => 1.2,
        }
    }

    /// Check if this scheme is supported on typical hardware accelerators
    ///
    /// Returns true for schemes that have good hardware support across
    /// common inference accelerators and CPUs.
    pub fn has_hardware_support(&self) -> bool {
        matches!(
            self,
            QuantizationScheme::Linear
                | QuantizationScheme::Symmetric
                | QuantizationScheme::Asymmetric
        )
    }
}

impl Default for QuantizationScheme {
    /// Default quantization scheme
    ///
    /// Returns Linear quantization as the default, which provides the best
    /// balance of accuracy, hardware support, and implementation simplicity.
    fn default() -> Self {
        QuantizationScheme::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_dtype_bits() {
        assert_eq!(QuantizedDType::Int8.bits(), 8);
        assert_eq!(QuantizedDType::UInt8.bits(), 8);
        assert_eq!(QuantizedDType::Int16.bits(), 16);
        assert_eq!(QuantizedDType::UInt16.bits(), 16);
        assert_eq!(QuantizedDType::Int4.bits(), 4);
        assert_eq!(QuantizedDType::UInt4.bits(), 4);
        assert_eq!(QuantizedDType::Binary.bits(), 1);

        let mixed = QuantizedDType::Mixed(vec![4, 8, 4]);
        assert_eq!(mixed.bits(), 8);
    }

    #[test]
    fn test_quantized_dtype_signed() {
        assert!(QuantizedDType::Int8.is_signed());
        assert!(QuantizedDType::Int16.is_signed());
        assert!(QuantizedDType::Int4.is_signed());
        assert!(!QuantizedDType::UInt8.is_signed());
        assert!(!QuantizedDType::UInt16.is_signed());
        assert!(!QuantizedDType::UInt4.is_signed());
        assert!(!QuantizedDType::Binary.is_signed());
    }

    #[test]
    fn test_quantized_dtype_value_range() {
        assert_eq!(QuantizedDType::Int8.value_range(), (-128, 127));
        assert_eq!(QuantizedDType::UInt8.value_range(), (0, 255));
        assert_eq!(QuantizedDType::Int16.value_range(), (-32768, 32767));
        assert_eq!(QuantizedDType::UInt16.value_range(), (0, 65535));
        assert_eq!(QuantizedDType::Int4.value_range(), (-8, 7));
        assert_eq!(QuantizedDType::UInt4.value_range(), (0, 15));
        assert_eq!(QuantizedDType::Binary.value_range(), (0, 1));
    }

    #[test]
    fn test_quantized_dtype_num_values() {
        assert_eq!(QuantizedDType::Int8.num_values(), 256);
        assert_eq!(QuantizedDType::UInt8.num_values(), 256);
        assert_eq!(QuantizedDType::Int4.num_values(), 16);
        assert_eq!(QuantizedDType::UInt4.num_values(), 16);
        assert_eq!(QuantizedDType::Binary.num_values(), 2);
    }

    #[test]
    fn test_quantized_dtype_sub_byte() {
        assert!(!QuantizedDType::Int8.is_sub_byte());
        assert!(!QuantizedDType::UInt8.is_sub_byte());
        assert!(!QuantizedDType::Int16.is_sub_byte());
        assert!(QuantizedDType::Int4.is_sub_byte());
        assert!(QuantizedDType::UInt4.is_sub_byte());
        assert!(QuantizedDType::Binary.is_sub_byte());
    }

    #[test]
    fn test_quantized_dtype_values_per_byte() {
        assert_eq!(QuantizedDType::Int8.values_per_byte(), 1.0);
        assert_eq!(QuantizedDType::UInt8.values_per_byte(), 1.0);
        assert_eq!(QuantizedDType::Int4.values_per_byte(), 2.0);
        assert_eq!(QuantizedDType::Binary.values_per_byte(), 8.0);
    }

    #[test]
    fn test_quantization_scheme_properties() {
        assert!(!QuantizationScheme::Linear.is_per_channel());
        assert!(QuantizationScheme::ChannelWise.is_per_channel());
        assert!(QuantizationScheme::BlockWise.is_per_channel());

        assert!(QuantizationScheme::Symmetric.is_symmetric());
        assert!(!QuantizationScheme::Asymmetric.is_symmetric());

        assert!(!QuantizationScheme::Symmetric.requires_zero_point());
        assert!(QuantizationScheme::Asymmetric.requires_zero_point());

        assert!(QuantizationScheme::Linear.has_hardware_support());
        assert!(QuantizationScheme::Symmetric.has_hardware_support());
        assert!(!QuantizationScheme::Logarithmic.has_hardware_support());
    }

    #[test]
    fn test_quantization_scheme_complexity() {
        assert_eq!(QuantizationScheme::Linear.complexity_factor(), 1.0);
        assert_eq!(QuantizationScheme::Symmetric.complexity_factor(), 1.0);
        assert!(QuantizationScheme::Logarithmic.complexity_factor() > 1.0);
    }

    #[test]
    fn test_quantization_scheme_default() {
        assert_eq!(QuantizationScheme::default(), QuantizationScheme::Linear);
    }

    #[test]
    fn test_dtype_compatibility() {
        assert!(QuantizedDType::Int8.is_compatible_with(&QuantizedDType::Int8));
        assert!(!QuantizedDType::Int8.is_compatible_with(&QuantizedDType::UInt8));
        assert!(!QuantizedDType::Int8.is_compatible_with(&QuantizedDType::Int16));

        let mixed8 = QuantizedDType::Mixed(vec![8, 8, 8]);
        assert!(mixed8.is_compatible_with(&QuantizedDType::Int8));
        assert!(mixed8.is_compatible_with(&QuantizedDType::UInt8));
    }
}
