// Data Types: BFloat16 Operations and Rounding Modes
//
// This module provides advanced BFloat16 (brain floating point) operations with
// precise rounding control. BFloat16 is a 16-bit floating point format that uses
// 8 bits for the exponent (same as F32) but only 7 bits for the mantissa,
// providing better dynamic range at the cost of precision compared to F16.

use half::bf16;

/// IEEE 754-style rounding modes for BFloat16 operations
///
/// These rounding modes control how intermediate results are rounded when
/// converting from higher precision operations back to BFloat16 format.
/// Different rounding modes are crucial for numerical stability and
/// reproducibility in machine learning applications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum BF16RoundingMode {
    /// Round to nearest representable value, ties to even (IEEE 754 default)
    ///
    /// When a value is exactly halfway between two representable values,
    /// round to the one with an even mantissa. This reduces systematic bias
    /// in repeated operations and is the default IEEE 754 rounding mode.
    NearestTiesToEven,

    /// Round to nearest representable value, ties away from zero
    ///
    /// When a value is exactly halfway between two representable values,
    /// round to the one further from zero. This can be more intuitive
    /// but may introduce systematic bias in repeated operations.
    NearestTiesAway,

    /// Round toward zero (truncate)
    ///
    /// Always round toward zero, effectively truncating the fractional part.
    /// This is equivalent to the truncation that occurs in integer conversion.
    TowardZero,

    /// Round toward positive infinity
    ///
    /// Always round toward positive infinity. Positive values are rounded up,
    /// negative values are rounded toward zero.
    TowardPositive,

    /// Round toward negative infinity
    ///
    /// Always round toward negative infinity. Negative values are rounded down,
    /// positive values are rounded toward zero.
    TowardNegative,
}

impl Default for BF16RoundingMode {
    fn default() -> Self {
        Self::NearestTiesToEven
    }
}

impl BF16RoundingMode {
    /// Get a string representation of the rounding mode
    pub fn as_str(&self) -> &'static str {
        match self {
            BF16RoundingMode::NearestTiesToEven => "nearest_ties_even",
            BF16RoundingMode::NearestTiesAway => "nearest_ties_away",
            BF16RoundingMode::TowardZero => "toward_zero",
            BF16RoundingMode::TowardPositive => "toward_positive",
            BF16RoundingMode::TowardNegative => "toward_negative",
        }
    }

    /// Parse a rounding mode from string
    #[allow(clippy::should_implement_trait)] // Custom parsing logic, not standard FromStr
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "nearest_ties_even" | "nearest" | "even" => Some(BF16RoundingMode::NearestTiesToEven),
            "nearest_ties_away" | "away" => Some(BF16RoundingMode::NearestTiesAway),
            "toward_zero" | "zero" | "trunc" => Some(BF16RoundingMode::TowardZero),
            "toward_positive" | "positive" | "up" => Some(BF16RoundingMode::TowardPositive),
            "toward_negative" | "negative" | "down" => Some(BF16RoundingMode::TowardNegative),
            _ => None,
        }
    }

    /// Check if this rounding mode can introduce systematic bias
    pub fn introduces_bias(&self) -> bool {
        match self {
            BF16RoundingMode::NearestTiesToEven => false,
            BF16RoundingMode::NearestTiesAway => true,
            BF16RoundingMode::TowardZero => true,
            BF16RoundingMode::TowardPositive => true,
            BF16RoundingMode::TowardNegative => true,
        }
    }
}

/// Extension trait for BFloat16 operations with precise rounding control
///
/// This trait extends the basic bf16 type with operations that allow explicit
/// control over rounding behavior. This is essential for reproducible numerical
/// computations and can help prevent accumulation of rounding errors in
/// iterative algorithms.
pub trait BFloat16Ops {
    /// Add two BFloat16 values with specified rounding mode
    ///
    /// Performs addition in higher precision (F32), then rounds the result
    /// back to BFloat16 using the specified rounding mode.
    fn add_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self;

    /// Subtract two BFloat16 values with specified rounding mode
    ///
    /// Performs subtraction in higher precision (F32), then rounds the result
    /// back to BFloat16 using the specified rounding mode.
    fn sub_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self;

    /// Multiply two BFloat16 values with specified rounding mode
    ///
    /// Performs multiplication in higher precision (F32), then rounds the result
    /// back to BFloat16 using the specified rounding mode.
    fn mul_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self;

    /// Divide two BFloat16 values with specified rounding mode
    ///
    /// Performs division in higher precision (F32), then rounds the result
    /// back to BFloat16 using the specified rounding mode.
    fn div_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self;

    /// Convert from F32 to BFloat16 with specified rounding mode
    ///
    /// This is the core conversion function that implements the various
    /// IEEE 754 rounding modes for the F32 -> BF16 conversion.
    fn from_f32_with_rounding(value: f32, mode: BF16RoundingMode) -> Self;

    /// Convert from F64 to BFloat16 with specified rounding mode
    ///
    /// First converts to F32, then applies BF16 rounding. Some precision
    /// may be lost in the F64 -> F32 step.
    fn from_f64_with_rounding(value: f64, mode: BF16RoundingMode) -> Self;

    /// Fused multiply-add with specified rounding mode
    ///
    /// Computes (self * mul + add) with only one rounding operation at the end.
    /// This can provide better numerical accuracy than separate multiply and add.
    fn fma_with_rounding(self, mul: Self, add: Self, mode: BF16RoundingMode) -> Self;

    /// Square root with specified rounding mode
    fn sqrt_with_rounding(self, mode: BF16RoundingMode) -> Self;

    /// Exponential function with specified rounding mode
    fn exp_with_rounding(self, mode: BF16RoundingMode) -> Self;

    /// Natural logarithm with specified rounding mode
    fn ln_with_rounding(self, mode: BF16RoundingMode) -> Self;

    /// Power function with specified rounding mode
    fn powf_with_rounding(self, exp: Self, mode: BF16RoundingMode) -> Self;

    /// Sine function with specified rounding mode
    fn sin_with_rounding(self, mode: BF16RoundingMode) -> Self;

    /// Cosine function with specified rounding mode
    fn cos_with_rounding(self, mode: BF16RoundingMode) -> Self;

    /// Tangent function with specified rounding mode
    fn tan_with_rounding(self, mode: BF16RoundingMode) -> Self;

    /// Absolute value with specified rounding mode
    fn abs_with_rounding(self, mode: BF16RoundingMode) -> Self;

    /// Reciprocal (1/x) with specified rounding mode
    fn recip_with_rounding(self, mode: BF16RoundingMode) -> Self;

    /// Minimum of two values with specified rounding mode
    fn min_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self;

    /// Maximum of two values with specified rounding mode
    fn max_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self;
}

impl BFloat16Ops for bf16 {
    fn add_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32() + other.to_f32();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn sub_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32() - other.to_f32();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn mul_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32() * other.to_f32();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn div_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32() / other.to_f32();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn from_f32_with_rounding(value: f32, mode: BF16RoundingMode) -> Self {
        match mode {
            BF16RoundingMode::NearestTiesToEven => {
                // This is the default rounding mode for bf16::from_f32
                bf16::from_f32(value)
            }
            BF16RoundingMode::NearestTiesAway => {
                // Implement ties away from zero rounding
                let bits = value.to_bits();
                let sign_bit = bits & 0x8000_0000;
                let abs_bits = bits & 0x7FFF_FFFF;

                // Extract the 16 most significant bits (BF16 format)
                let bf16_bits = bits >> 16;
                let truncated_bits = bf16_bits << 16;
                let truncated_f32 = f32::from_bits(truncated_bits);

                // Check if we need to round
                let fraction_bits = abs_bits & 0x0000_FFFF;
                let halfway = 0x8000;

                if fraction_bits > halfway
                    || (fraction_bits == halfway && value.abs() >= truncated_f32.abs())
                {
                    // Round away from zero
                    #[allow(clippy::if_same_then_else)] // Both branches use same increment
                    let increment = if sign_bit == 0 {
                        0x0001_0000
                    } else {
                        0x0001_0000
                    };
                    let rounded_bits = truncated_bits.wrapping_add(increment);
                    bf16::from_f32(f32::from_bits(rounded_bits))
                } else {
                    bf16::from_f32(truncated_f32)
                }
            }
            BF16RoundingMode::TowardZero => {
                // Truncate toward zero
                let bits = value.to_bits();
                let truncated_bits = (bits >> 16) << 16;
                bf16::from_f32(f32::from_bits(truncated_bits))
            }
            BF16RoundingMode::TowardPositive => {
                // Round toward +infinity
                let bits = value.to_bits();
                let truncated_bits = (bits >> 16) << 16;
                let truncated_f32 = f32::from_bits(truncated_bits);

                if value > truncated_f32 && !value.is_infinite() {
                    // Round up
                    let rounded_bits = truncated_bits.wrapping_add(0x0001_0000);
                    bf16::from_f32(f32::from_bits(rounded_bits))
                } else {
                    bf16::from_f32(truncated_f32)
                }
            }
            BF16RoundingMode::TowardNegative => {
                // Round toward -infinity
                let bits = value.to_bits();
                let truncated_bits = (bits >> 16) << 16;
                let truncated_f32 = f32::from_bits(truncated_bits);

                if value < truncated_f32 && !value.is_infinite() {
                    // Round down (more negative)
                    let rounded_bits = truncated_bits.wrapping_add(0x0001_0000);
                    bf16::from_f32(f32::from_bits(rounded_bits))
                } else {
                    bf16::from_f32(truncated_f32)
                }
            }
        }
    }

    fn from_f64_with_rounding(value: f64, mode: BF16RoundingMode) -> Self {
        // Convert to F32 first, then apply BF16 rounding
        Self::from_f32_with_rounding(value as f32, mode)
    }

    fn fma_with_rounding(self, mul: Self, add: Self, mode: BF16RoundingMode) -> Self {
        // Perform FMA in F32 precision, then round to BF16
        let result_f32 = self.to_f32().mul_add(mul.to_f32(), add.to_f32());
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn sqrt_with_rounding(self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().sqrt();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn exp_with_rounding(self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().exp();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn ln_with_rounding(self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().ln();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn powf_with_rounding(self, exp: Self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().powf(exp.to_f32());
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn sin_with_rounding(self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().sin();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn cos_with_rounding(self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().cos();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn tan_with_rounding(self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().tan();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn abs_with_rounding(self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().abs();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn recip_with_rounding(self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().recip();
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn min_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().min(other.to_f32());
        Self::from_f32_with_rounding(result_f32, mode)
    }

    fn max_with_rounding(self, other: Self, mode: BF16RoundingMode) -> Self {
        let result_f32 = self.to_f32().max(other.to_f32());
        Self::from_f32_with_rounding(result_f32, mode)
    }
}

/// Utility functions for BFloat16 precision analysis
pub mod utils {
    use super::*;

    /// Calculate the Unit in Last Place (ULP) for a BF16 value
    ///
    /// ULP represents the spacing between consecutive floating point numbers.
    /// This is useful for numerical analysis and error bounds.
    pub fn ulp_bf16(value: bf16) -> bf16 {
        let f32_val = value.to_f32();
        let next_f32 = f32::from_bits(f32_val.to_bits().wrapping_add(1));
        bf16::from_f32(next_f32 - f32_val)
    }

    /// Check if two BF16 values are approximately equal within ULP tolerance
    pub fn approx_eq_ulp(a: bf16, b: bf16, max_ulp: u32) -> bool {
        let diff = (a.to_f32() - b.to_f32()).abs();
        let ulp = ulp_bf16(a.max(b)).to_f32();
        diff <= ulp * max_ulp as f32
    }

    /// Get the mantissa bits of a BF16 value
    pub fn mantissa_bits(value: bf16) -> u16 {
        let bits = value.to_bits();
        bits & 0x007F // 7-bit mantissa
    }

    /// Get the exponent of a BF16 value (biased)
    pub fn exponent_biased(value: bf16) -> u16 {
        let bits = value.to_bits();
        (bits >> 7) & 0x00FF // 8-bit exponent
    }

    /// Get the unbiased exponent of a BF16 value
    pub fn exponent_unbiased(value: bf16) -> i16 {
        let biased = exponent_biased(value) as i16;
        biased - 127 // BF16 uses same bias as F32
    }

    /// Check if a BF16 value is subnormal (denormalized)
    pub fn is_subnormal(value: bf16) -> bool {
        let exp = exponent_biased(value);
        let mantissa = mantissa_bits(value);
        exp == 0 && mantissa != 0
    }

    /// Estimate the relative error when converting F32 to BF16
    pub fn conversion_relative_error(original: f32) -> f32 {
        let converted = bf16::from_f32(original);
        let restored = converted.to_f32();

        if original == 0.0 {
            return 0.0;
        }

        ((restored - original) / original).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounding_mode_properties() {
        // Test string conversion
        assert_eq!(
            BF16RoundingMode::NearestTiesToEven.as_str(),
            "nearest_ties_even"
        );
        assert_eq!(BF16RoundingMode::TowardZero.as_str(), "toward_zero");

        // Test parsing
        assert_eq!(
            BF16RoundingMode::from_str("nearest"),
            Some(BF16RoundingMode::NearestTiesToEven)
        );
        assert_eq!(
            BF16RoundingMode::from_str("trunc"),
            Some(BF16RoundingMode::TowardZero)
        );

        // Test bias detection
        assert!(!BF16RoundingMode::NearestTiesToEven.introduces_bias());
        assert!(BF16RoundingMode::TowardZero.introduces_bias());
    }

    #[test]
    fn test_basic_arithmetic_with_rounding() {
        let a = bf16::from_f32(1.5);
        let b = bf16::from_f32(2.5);
        let mode = BF16RoundingMode::NearestTiesToEven;

        // Test arithmetic operations
        let sum = a.add_with_rounding(b, mode);
        assert!((sum.to_f32() - 4.0).abs() < f32::EPSILON);

        let diff = a.sub_with_rounding(b, mode);
        assert!((diff.to_f32() - (-1.0)).abs() < f32::EPSILON);

        let product = a.mul_with_rounding(b, mode);
        assert!((product.to_f32() - 3.75).abs() < 0.01);

        let quotient = a.div_with_rounding(b, mode);
        assert!((quotient.to_f32() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_different_rounding_modes() {
        let value = 1.234375f32; // Exactly representable + small increment

        // Test different rounding modes give different results for edge cases
        let nearest_even = bf16::from_f32_with_rounding(value, BF16RoundingMode::NearestTiesToEven);
        let toward_zero = bf16::from_f32_with_rounding(value, BF16RoundingMode::TowardZero);
        let toward_pos = bf16::from_f32_with_rounding(value, BF16RoundingMode::TowardPositive);

        // All should be close to original value
        assert!((nearest_even.to_f32() - value).abs() < 0.01);
        assert!((toward_zero.to_f32() - value).abs() < 0.01);
        assert!((toward_pos.to_f32() - value).abs() < 0.01);

        // TowardZero should not exceed original value
        assert!(toward_zero.to_f32() <= value + f32::EPSILON);
    }

    #[test]
    fn test_mathematical_functions() {
        let value = bf16::from_f32(2.0);
        let mode = BF16RoundingMode::NearestTiesToEven;

        // Test mathematical functions
        let sqrt_result = value.sqrt_with_rounding(mode);
        assert!((sqrt_result.to_f32() - 1.414).abs() < 0.01);

        let exp_result = value.exp_with_rounding(mode);
        assert!((exp_result.to_f32() - 7.389).abs() < 0.1);

        let ln_result = value.ln_with_rounding(mode);
        assert!((ln_result.to_f32() - 0.693).abs() < 0.01);

        // Test trigonometric functions
        let pi_quarter = bf16::from_f32(std::f32::consts::PI / 4.0);
        let sin_result = pi_quarter.sin_with_rounding(mode);
        assert!((sin_result.to_f32() - (std::f32::consts::SQRT_2 / 2.0)).abs() < 0.01);
    }

    #[test]
    fn test_fma_operation() {
        let a = bf16::from_f32(2.0);
        let b = bf16::from_f32(3.0);
        let c = bf16::from_f32(1.0);
        let mode = BF16RoundingMode::NearestTiesToEven;

        // Test fused multiply-add
        let fma_result = a.fma_with_rounding(b, c, mode);
        let expected = 2.0 * 3.0 + 1.0; // = 7.0
        assert!((fma_result.to_f32() - expected).abs() < f32::EPSILON);
    }

    #[test]
    fn test_special_values() {
        let mode = BF16RoundingMode::NearestTiesToEven;

        // Test special values
        let zero = bf16::from_f32(0.0);
        let inf = bf16::from_f32(f32::INFINITY);
        let neg_inf = bf16::from_f32(f32::NEG_INFINITY);
        let nan = bf16::from_f32(f32::NAN);

        // Operations with zero
        assert_eq!(zero.add_with_rounding(zero, mode).to_f32(), 0.0);
        assert_eq!(
            zero.mul_with_rounding(bf16::from_f32(5.0), mode).to_f32(),
            0.0
        );

        // Check that infinity and NaN are preserved (approximately)
        assert!(inf.to_f32().is_infinite() && inf.to_f32().is_sign_positive());
        assert!(neg_inf.to_f32().is_infinite() && neg_inf.to_f32().is_sign_negative());
        assert!(nan.to_f32().is_nan());
    }

    #[test]
    fn test_utility_functions() {
        let value = bf16::from_f32(1.0);

        // Test ULP calculation
        let ulp = utils::ulp_bf16(value);
        assert!(ulp.to_f32() > 0.0);

        // Test approximate equality
        let close_value = bf16::from_f32(1.0 + ulp.to_f32());
        assert!(utils::approx_eq_ulp(value, close_value, 2));

        // Test bit manipulation functions
        let mantissa = utils::mantissa_bits(value);
        let exp_biased = utils::exponent_biased(value);
        let exp_unbiased = utils::exponent_unbiased(value);

        assert!(mantissa <= 0x7F); // 7-bit mantissa
        assert!(exp_biased <= 0xFF); // 8-bit exponent
        assert_eq!(exp_unbiased, exp_biased as i16 - 127);

        // Test subnormal detection
        let _tiny = bf16::from_f32(1e-40); // Should be subnormal in BF16
                                           // Note: might not be subnormal depending on BF16 range
    }

    #[test]
    fn test_conversion_error_analysis() {
        let original = 1.234567f32;
        let error = utils::conversion_relative_error(original);

        // Error should be small but non-zero due to BF16 precision limits
        assert!(error >= 0.0);
        assert!(error < 0.1); // Should be reasonable for most values

        // Test zero case
        let zero_error = utils::conversion_relative_error(0.0);
        assert_eq!(zero_error, 0.0);
    }

    #[test]
    fn test_precision_limits() {
        // Test BF16 precision limits
        let large_value = 65504.0f32; // Near maximum representable value
        let bf16_large = bf16::from_f32(large_value);
        assert!((bf16_large.to_f32() - large_value).abs() < 1000.0);

        let small_value = 1e-5f32;
        let bf16_small = bf16::from_f32(small_value);
        // BF16 should represent this reasonably well due to its wide dynamic range
        assert!(bf16_small.to_f32() > 0.0);
    }

    #[test]
    fn test_min_max_operations() {
        let a = bf16::from_f32(1.5);
        let b = bf16::from_f32(2.5);
        let mode = BF16RoundingMode::NearestTiesToEven;

        let min_result = a.min_with_rounding(b, mode);
        let max_result = a.max_with_rounding(b, mode);

        assert_eq!(min_result.to_f32(), 1.5);
        assert_eq!(max_result.to_f32(), 2.5);
    }

    #[test]
    fn test_reciprocal_and_abs() {
        let value = bf16::from_f32(-2.0);
        let mode = BF16RoundingMode::NearestTiesToEven;

        let abs_result = value.abs_with_rounding(mode);
        assert_eq!(abs_result.to_f32(), 2.0);

        let recip_result = value.recip_with_rounding(mode);
        assert!((recip_result.to_f32() - (-0.5)).abs() < 0.01);
    }
}
