//! IEEE 754 Compliance Tests and Utilities
//!
//! This module provides comprehensive tests and utilities to ensure IEEE 754 compliance
//! for floating-point operations in ToRSh. IEEE 754 defines standards for:
//! - Floating-point arithmetic
//! - Special values (NaN, Inf, -Inf, -0)
//! - Rounding modes
//! - Exception handling
//! - Comparison semantics

use crate::dtype::{DType, FloatElement};
use crate::error::{Result, TorshError};

/// IEEE 754 special values that must be handled correctly
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialValue {
    /// Positive infinity
    PositiveInfinity,
    /// Negative infinity
    NegativeInfinity,
    /// Not a Number (quiet NaN)
    QuietNaN,
    /// Not a Number (signaling NaN)
    SignalingNaN,
    /// Positive zero
    PositiveZero,
    /// Negative zero
    NegativeZero,
    /// Subnormal (denormalized) numbers near zero
    Subnormal,
}

/// IEEE 754 rounding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default)
    ToNearestTiesToEven,
    /// Round to nearest, ties away from zero
    ToNearestTiesAway,
    /// Round toward zero (truncation)
    TowardZero,
    /// Round toward positive infinity
    TowardPositiveInfinity,
    /// Round toward negative infinity
    TowardNegativeInfinity,
}

/// IEEE 754 exceptions that can occur during operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exception {
    /// Invalid operation (e.g., sqrt of negative number)
    InvalidOperation,
    /// Division by zero
    DivisionByZero,
    /// Result too large to represent
    Overflow,
    /// Result too small to represent
    Underflow,
    /// Result not exact (precision loss)
    Inexact,
}

/// Trait for IEEE 754 compliant floating-point types
pub trait IEEE754Float: FloatElement {
    /// Check if value is positive infinity
    fn is_positive_infinity(self) -> bool;

    /// Check if value is negative infinity
    fn is_negative_infinity(self) -> bool;

    /// Check if value is any infinity
    fn is_infinity(self) -> bool {
        self.is_positive_infinity() || self.is_negative_infinity()
    }

    /// Check if value is NaN (either quiet or signaling)
    fn is_nan_value(self) -> bool;

    /// Check if value is positive zero
    fn is_positive_zero(self) -> bool;

    /// Check if value is negative zero
    fn is_negative_zero(self) -> bool;

    /// Check if value is subnormal (denormalized)
    fn is_subnormal(self) -> bool;

    /// Get the sign bit (true for negative, including -0)
    fn sign_bit(self) -> bool;

    /// Create positive infinity
    fn positive_infinity() -> Self;

    /// Create negative infinity
    fn negative_infinity() -> Self;

    /// Create quiet NaN
    fn quiet_nan() -> Self;

    /// Create positive zero
    fn positive_zero() -> Self;

    /// Create negative zero
    fn negative_zero() -> Self;

    /// Copy sign from another value
    fn copysign(self, sign: Self) -> Self;
}

impl IEEE754Float for f32 {
    fn is_positive_infinity(self) -> bool {
        self == f32::INFINITY
    }

    fn is_negative_infinity(self) -> bool {
        self == f32::NEG_INFINITY
    }

    fn is_nan_value(self) -> bool {
        self.is_nan()
    }

    fn is_positive_zero(self) -> bool {
        self == 0.0 && self.is_sign_positive()
    }

    fn is_negative_zero(self) -> bool {
        self == 0.0 && self.is_sign_negative()
    }

    fn is_subnormal(self) -> bool {
        self.is_finite() && self != 0.0 && self.abs() < f32::MIN_POSITIVE
    }

    fn sign_bit(self) -> bool {
        self.is_sign_negative()
    }

    fn positive_infinity() -> Self {
        f32::INFINITY
    }

    fn negative_infinity() -> Self {
        f32::NEG_INFINITY
    }

    fn quiet_nan() -> Self {
        f32::NAN
    }

    fn positive_zero() -> Self {
        0.0_f32
    }

    fn negative_zero() -> Self {
        -0.0_f32
    }

    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }
}

impl IEEE754Float for f64 {
    fn is_positive_infinity(self) -> bool {
        self == f64::INFINITY
    }

    fn is_negative_infinity(self) -> bool {
        self == f64::NEG_INFINITY
    }

    fn is_nan_value(self) -> bool {
        self.is_nan()
    }

    fn is_positive_zero(self) -> bool {
        self == 0.0 && self.is_sign_positive()
    }

    fn is_negative_zero(self) -> bool {
        self == 0.0 && self.is_sign_negative()
    }

    fn is_subnormal(self) -> bool {
        self.is_finite() && self != 0.0 && self.abs() < f64::MIN_POSITIVE
    }

    fn sign_bit(self) -> bool {
        self.is_sign_negative()
    }

    fn positive_infinity() -> Self {
        f64::INFINITY
    }

    fn negative_infinity() -> Self {
        f64::NEG_INFINITY
    }

    fn quiet_nan() -> Self {
        f64::NAN
    }

    fn positive_zero() -> Self {
        0.0_f64
    }

    fn negative_zero() -> Self {
        -0.0_f64
    }

    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }
}

/// Verify IEEE 754 compliance for a given floating-point type
pub struct ComplianceChecker;

impl ComplianceChecker {
    /// Check special value handling for a float type
    pub fn check_special_values<T: IEEE754Float>() -> Result<()> {
        // Test infinity
        let pos_inf = T::positive_infinity();
        let neg_inf = T::negative_infinity();
        assert!(pos_inf.is_positive_infinity());
        assert!(neg_inf.is_negative_infinity());
        assert!(pos_inf.is_infinity());
        assert!(neg_inf.is_infinity());

        // Test NaN
        let nan = T::quiet_nan();
        assert!(nan.is_nan_value());
        assert!(nan != nan); // NaN != NaN by IEEE 754

        // Test zeros
        let pos_zero = T::positive_zero();
        let neg_zero = T::negative_zero();
        assert!(pos_zero.is_positive_zero());
        assert!(neg_zero.is_negative_zero());
        assert!(pos_zero == neg_zero); // +0 == -0 by IEEE 754

        Ok(())
    }

    /// Check arithmetic operations with special values
    pub fn check_special_arithmetic<T: IEEE754Float>() -> Result<()> {
        let pos_inf = T::positive_infinity();
        let neg_inf = T::negative_infinity();
        let nan = T::quiet_nan();
        let one = T::from(1.0).expect("numeric conversion should succeed");
        let zero = T::from(0.0).expect("numeric conversion should succeed");

        // Infinity + Infinity = Infinity
        assert!((pos_inf + pos_inf).is_positive_infinity());
        assert!((neg_inf + neg_inf).is_negative_infinity());

        // Infinity - Infinity = NaN
        assert!((pos_inf - pos_inf).is_nan_value());

        // Infinity * finite = Infinity (with sign rules)
        assert!((pos_inf * one).is_positive_infinity());
        assert!((neg_inf * one).is_negative_infinity());

        // Infinity * 0 = NaN
        assert!((pos_inf * zero).is_nan_value());

        // finite / 0 = Infinity (with sign rules)
        let pos_zero = T::positive_zero();
        let neg_zero = T::negative_zero();
        assert!((one / pos_zero).is_positive_infinity());
        assert!((one / neg_zero).is_negative_infinity());

        // 0 / 0 = NaN
        assert!((zero / zero).is_nan_value());

        // NaN propagation
        assert!((nan + one).is_nan_value());
        assert!((nan * one).is_nan_value());
        assert!((nan / one).is_nan_value());

        Ok(())
    }

    /// Check comparison operations
    pub fn check_comparisons<T: IEEE754Float>() -> Result<()> {
        let pos_inf = T::positive_infinity();
        let neg_inf = T::negative_infinity();
        let nan = T::quiet_nan();
        let one = T::from(1.0).expect("numeric conversion should succeed");
        let pos_zero = T::positive_zero();
        let neg_zero = T::negative_zero();

        // NaN comparisons always return false except !=
        assert!(!(nan == nan));
        assert!(!(nan < one));
        assert!(!(nan > one));
        assert!(!(nan <= one));
        assert!(!(nan >= one));
        assert!(nan != nan);

        // Infinity comparisons
        assert!(pos_inf > one);
        assert!(neg_inf < one);
        assert!(pos_inf > neg_inf);

        // Zero comparisons
        assert!(pos_zero == neg_zero);
        assert!(!(pos_zero < neg_zero));
        assert!(!(pos_zero > neg_zero));

        Ok(())
    }

    /// Check sign operations
    pub fn check_sign_operations<T: IEEE754Float>() -> Result<()> {
        let one = T::from(1.0).expect("numeric conversion should succeed");
        let neg_one = T::from(-1.0).expect("numeric conversion should succeed");
        let pos_zero = T::positive_zero();
        let neg_zero = T::negative_zero();

        // Sign bit checks
        assert!(!one.sign_bit());
        assert!(neg_one.sign_bit());
        assert!(!pos_zero.sign_bit());
        assert!(neg_zero.sign_bit());

        // Copysign
        assert_eq!(IEEE754Float::copysign(one, neg_one), neg_one);
        assert_eq!(IEEE754Float::copysign(neg_one, one), one);
        assert!(IEEE754Float::copysign(pos_zero, neg_one).is_negative_zero());
        assert!(IEEE754Float::copysign(neg_zero, one).is_positive_zero());

        Ok(())
    }

    /// Check subnormal number handling
    pub fn check_subnormal_handling<T: IEEE754Float>() -> Result<()> {
        // This is a basic check; actual subnormal handling depends on CPU flags
        // Most CPUs support gradual underflow (subnormal numbers)
        let min_positive = if std::mem::size_of::<T>() == 4 {
            T::from(f32::MIN_POSITIVE).expect("numeric conversion should succeed")
        } else {
            T::from(f64::MIN_POSITIVE).expect("numeric conversion should succeed")
        };

        let two = T::from(2.0).expect("numeric conversion should succeed");
        let half_min = min_positive / two;

        // half_min should be subnormal (if supported)
        if half_min != T::from(0.0).expect("numeric conversion should succeed") {
            assert!(IEEE754Float::is_subnormal(half_min));
        }

        Ok(())
    }

    /// Run all IEEE 754 compliance checks
    pub fn run_all_checks<T: IEEE754Float>() -> Result<()> {
        Self::check_special_values::<T>()?;
        Self::check_special_arithmetic::<T>()?;
        Self::check_comparisons::<T>()?;
        Self::check_sign_operations::<T>()?;
        Self::check_subnormal_handling::<T>()?;
        Ok(())
    }
}

/// Check if a DType represents an IEEE 754 compliant floating-point type
pub fn is_ieee754_compliant(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::F32 | DType::F64)
}

/// Validate IEEE 754 compliance for tensor operations
pub fn validate_ieee754_operation(dtype: DType, operation: &str) -> Result<()> {
    if !is_ieee754_compliant(dtype) {
        return Err(TorshError::InvalidArgument(format!(
            "Operation '{}' requires IEEE 754 compliant floating-point type, got {:?}",
            operation, dtype
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_special_values() {
        assert!(ComplianceChecker::check_special_values::<f32>().is_ok());
    }

    #[test]
    fn test_f64_special_values() {
        assert!(ComplianceChecker::check_special_values::<f64>().is_ok());
    }

    #[test]
    fn test_f32_special_arithmetic() {
        assert!(ComplianceChecker::check_special_arithmetic::<f32>().is_ok());
    }

    #[test]
    fn test_f64_special_arithmetic() {
        assert!(ComplianceChecker::check_special_arithmetic::<f64>().is_ok());
    }

    #[test]
    fn test_f32_comparisons() {
        assert!(ComplianceChecker::check_comparisons::<f32>().is_ok());
    }

    #[test]
    fn test_f64_comparisons() {
        assert!(ComplianceChecker::check_comparisons::<f64>().is_ok());
    }

    #[test]
    fn test_f32_sign_operations() {
        assert!(ComplianceChecker::check_sign_operations::<f32>().is_ok());
    }

    #[test]
    fn test_f64_sign_operations() {
        assert!(ComplianceChecker::check_sign_operations::<f64>().is_ok());
    }

    #[test]
    fn test_f32_subnormal_handling() {
        assert!(ComplianceChecker::check_subnormal_handling::<f32>().is_ok());
    }

    #[test]
    fn test_f64_subnormal_handling() {
        assert!(ComplianceChecker::check_subnormal_handling::<f64>().is_ok());
    }

    #[test]
    fn test_f32_full_compliance() {
        assert!(ComplianceChecker::run_all_checks::<f32>().is_ok());
    }

    #[test]
    fn test_f64_full_compliance() {
        assert!(ComplianceChecker::run_all_checks::<f64>().is_ok());
    }

    #[test]
    fn test_is_ieee754_compliant() {
        assert!(is_ieee754_compliant(DType::F32));
        assert!(is_ieee754_compliant(DType::F64));
        assert!(is_ieee754_compliant(DType::F16));

        assert!(!is_ieee754_compliant(DType::I32));
        assert!(!is_ieee754_compliant(DType::I64));
        assert!(!is_ieee754_compliant(DType::Bool));
    }

    #[test]
    fn test_validate_ieee754_operation() {
        assert!(validate_ieee754_operation(DType::F32, "sin").is_ok());
        assert!(validate_ieee754_operation(DType::F64, "cos").is_ok());

        let result = validate_ieee754_operation(DType::I32, "sin");
        assert!(result.is_err());
    }

    #[test]
    fn test_nan_properties() {
        let nan = f32::NAN;

        // NaN is never equal to anything, including itself
        assert!(nan != nan);
        assert!(!(nan == nan));
        assert!(!(nan < 0.0));
        assert!(!(nan > 0.0));
        assert!(!(nan <= 0.0));
        assert!(!(nan >= 0.0));

        // NaN should be identified
        assert!(nan.is_nan_value());
    }

    #[test]
    fn test_infinity_arithmetic() {
        let inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;

        // Infinity arithmetic
        assert!((inf + 1.0) == inf);
        assert!((inf * 2.0) == inf);
        assert!((neg_inf * 2.0) == neg_inf);
        // inf * neg_inf = neg_inf (not NaN, as infinities have consistent sign rules)
        assert!((inf * neg_inf).is_negative_infinity());
        assert!((inf / inf).is_nan());
    }

    #[test]
    fn test_zero_sign() {
        let pos_zero = 0.0_f32;
        let neg_zero = -0.0_f32;

        // Zeros are equal but have different signs
        assert!(pos_zero == neg_zero);
        assert!(pos_zero.is_positive_zero());
        assert!(neg_zero.is_negative_zero());

        // Sign affects division by zero
        assert!((1.0 / pos_zero).is_positive_infinity());
        assert!((1.0 / neg_zero).is_negative_infinity());
    }

    #[test]
    fn test_subnormal_numbers() {
        // Create a subnormal number
        let min_positive = f32::MIN_POSITIVE; // Smallest normal positive f32
        let subnormal = min_positive / 2.0;

        if subnormal != 0.0 {
            // CPU supports subnormal numbers
            assert!(subnormal.is_subnormal());
            assert!(subnormal.is_finite());
            assert!(subnormal > 0.0);
            assert!(subnormal < min_positive);
        }
    }

    #[test]
    fn test_copysign() {
        let x = 3.0_f32;
        let y = -5.0_f32;

        assert_eq!(x.copysign(y), -3.0);
        assert_eq!((-x).copysign(x), 3.0);

        // Works with zeros
        let pos_zero = 0.0_f32;
        let neg_zero = -0.0_f32;
        assert!(pos_zero.copysign(neg_zero).is_negative_zero());
        assert!(neg_zero.copysign(pos_zero).is_positive_zero());
    }

    #[test]
    fn test_rounding_mode_enum() {
        // Just verify enum can be created and compared
        let mode = RoundingMode::ToNearestTiesToEven;
        assert_eq!(mode, RoundingMode::ToNearestTiesToEven);
        assert_ne!(mode, RoundingMode::TowardZero);
    }

    #[test]
    fn test_exception_enum() {
        // Just verify enum can be created and compared
        let exc = Exception::InvalidOperation;
        assert_eq!(exc, Exception::InvalidOperation);
        assert_ne!(exc, Exception::DivisionByZero);
    }

    #[test]
    fn test_special_value_enum() {
        // Just verify enum can be created and compared
        let val = SpecialValue::PositiveInfinity;
        assert_eq!(val, SpecialValue::PositiveInfinity);
        assert_ne!(val, SpecialValue::NegativeInfinity);
    }
}
