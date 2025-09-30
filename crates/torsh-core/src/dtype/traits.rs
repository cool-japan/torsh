// Data Types: Tensor Element Traits and Core Abstractions
//
// This module defines the core traits that determine which types can be used as
// tensor elements in the ToRSh framework. It includes the fundamental TensorElement
// trait and specialized traits for floating point and complex number operations.

use half::{bf16, f16};
use num_complex::Complex;
use num_traits::{Float, NumCast, One, Zero};
use std::any::Any;
use std::fmt;

use crate::dtype::core::DType;

/// Core trait for types that can be stored in tensors
///
/// This trait defines the fundamental requirements for any type that can be used
/// as an element in a ToRSh tensor. All tensor element types must be:
/// - Cloneable and copyable for efficient operations
/// - Send + Sync for thread safety
/// - Have equality comparison for testing and validation
/// - Debuggable for development and troubleshooting
/// - 'static lifetime for type safety
///
/// The trait also provides methods for type identification and zero/one values
/// which are essential for tensor operations.
pub trait TensorElement: Clone + Copy + Send + Sync + PartialEq + fmt::Debug + 'static {
    /// Get the DType corresponding to this element type
    ///
    /// Each tensor element type maps to exactly one DType variant.
    /// This is used for type checking and storage optimization.
    fn dtype() -> DType;

    /// Get the zero value for this type
    ///
    /// Returns the additive identity element (zero) for this type.
    /// This is used for tensor initialization and padding operations.
    fn zero() -> Self;

    /// Get the one value for this type
    ///
    /// Returns the multiplicative identity element (one) for this type.
    /// This is used for tensor operations like identity matrices.
    fn one() -> Self;

    /// Check if this value is zero
    ///
    /// Returns true if this value represents the zero element.
    /// This is used for sparsity detection and optimization.
    fn is_zero(&self) -> bool;

    /// Check if this value is one
    ///
    /// Returns true if this value represents the one element.
    /// This is used for optimization and mathematical operations.
    fn is_one(&self) -> bool;

    /// Convert from f64 if possible
    ///
    /// Attempts to convert a 64-bit floating point value to this type.
    /// Returns None if the conversion is not possible or loses precision.
    fn from_f64(value: f64) -> Option<Self>;

    /// Convert to f64 if possible
    ///
    /// Attempts to convert this value to a 64-bit floating point.
    /// Returns None if the conversion is not possible.
    fn to_f64(&self) -> Option<f64>;

    /// Get type information for dynamic typing
    ///
    /// Returns type information that can be used for runtime type checking
    /// and dynamic dispatch in advanced tensor operations.
    fn type_info() -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// Specialized trait for floating point tensor elements
///
/// This trait extends TensorElement for types that represent floating point numbers.
/// It provides access to floating point specific operations through the Float trait
/// from num-traits, enabling mathematical functions like trigonometry, logarithms, etc.
pub trait FloatElement: TensorElement + Float {
    /// Get machine epsilon for this floating point type
    ///
    /// Returns the difference between 1.0 and the next larger representable value.
    /// This is useful for numerical stability checks and comparisons.
    fn epsilon() -> Self;

    /// Check if the value is finite (not infinite or NaN)
    ///
    /// Returns true if the value is a normal finite number.
    fn is_finite(&self) -> bool {
        Float::is_finite(*self)
    }

    /// Check if the value is infinite
    ///
    /// Returns true if the value represents positive or negative infinity.
    fn is_infinite(&self) -> bool {
        Float::is_infinite(*self)
    }

    /// Check if the value is NaN (Not a Number)
    ///
    /// Returns true if the value is NaN, which occurs from invalid operations
    /// like 0/0 or sqrt(-1) in real arithmetic.
    fn is_nan(&self) -> bool {
        Float::is_nan(*self)
    }

    /// Get positive infinity for this type
    fn infinity() -> Self {
        Float::infinity()
    }

    /// Get negative infinity for this type
    fn neg_infinity() -> Self {
        Float::neg_infinity()
    }

    /// Get NaN (Not a Number) for this type
    fn nan() -> Self {
        Float::nan()
    }
}

/// Helper macro for implementing TensorElement for basic numeric types
///
/// This macro reduces boilerplate by automatically implementing the TensorElement
/// trait for standard numeric types with their corresponding DType variants.
macro_rules! impl_tensor_element {
    ($ty:ty, $dtype:expr) => {
        impl TensorElement for $ty {
            fn dtype() -> DType {
                $dtype
            }

            fn zero() -> Self {
                <$ty as Zero>::zero()
            }

            fn one() -> Self {
                <$ty as One>::one()
            }

            fn is_zero(&self) -> bool {
                *self == <$ty as TensorElement>::zero()
            }

            fn is_one(&self) -> bool {
                *self == <$ty as TensorElement>::one()
            }

            fn from_f64(value: f64) -> Option<Self> {
                NumCast::from(value)
            }

            fn to_f64(&self) -> Option<f64> {
                NumCast::from(*self)
            }
        }
    };
}

// Implement TensorElement for standard integer types
impl_tensor_element!(u8, DType::U8);
impl_tensor_element!(i8, DType::I8);
impl_tensor_element!(i16, DType::I16);
impl_tensor_element!(i32, DType::I32);
impl_tensor_element!(u32, DType::U32);
impl_tensor_element!(i64, DType::I64);
impl_tensor_element!(u64, DType::U64);

// Implement TensorElement for standard floating point types
impl_tensor_element!(f32, DType::F32);
impl_tensor_element!(f64, DType::F64);

/// Custom implementation for bool (doesn't implement standard numeric traits)
impl TensorElement for bool {
    fn dtype() -> DType {
        DType::Bool
    }

    fn zero() -> Self {
        false
    }

    fn one() -> Self {
        true
    }

    fn is_zero(&self) -> bool {
        !*self
    }

    fn is_one(&self) -> bool {
        *self
    }

    fn from_f64(value: f64) -> Option<Self> {
        if value == 0.0 {
            Some(false)
        } else if value == 1.0 {
            Some(true)
        } else {
            None
        }
    }

    fn to_f64(&self) -> Option<f64> {
        Some(if *self { 1.0 } else { 0.0 })
    }
}

/// Implement FloatElement for standard floating point types
impl FloatElement for f32 {
    fn epsilon() -> Self {
        f32::EPSILON
    }
}

impl FloatElement for f64 {
    fn epsilon() -> Self {
        f64::EPSILON
    }
}

/// Custom implementation for f16 (half precision)
impl TensorElement for f16 {
    fn dtype() -> DType {
        DType::F16
    }

    fn zero() -> Self {
        f16::ZERO
    }

    fn one() -> Self {
        f16::ONE
    }

    fn is_zero(&self) -> bool {
        *self == f16::ZERO
    }

    fn is_one(&self) -> bool {
        *self == f16::ONE
    }

    fn from_f64(value: f64) -> Option<Self> {
        Some(f16::from_f64(value))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(f16::to_f64(*self))
    }
}

impl FloatElement for f16 {
    fn epsilon() -> Self {
        f16::EPSILON
    }

    fn is_finite(&self) -> bool {
        (*self).is_finite()
    }

    fn is_infinite(&self) -> bool {
        (*self).is_infinite()
    }

    fn is_nan(&self) -> bool {
        (*self).is_nan()
    }

    fn infinity() -> Self {
        f16::INFINITY
    }

    fn neg_infinity() -> Self {
        f16::NEG_INFINITY
    }

    fn nan() -> Self {
        f16::NAN
    }
}

/// Custom implementation for bf16 (brain floating point)
impl TensorElement for bf16 {
    fn dtype() -> DType {
        DType::BF16
    }

    fn zero() -> Self {
        bf16::ZERO
    }

    fn one() -> Self {
        bf16::ONE
    }

    fn is_zero(&self) -> bool {
        *self == bf16::ZERO
    }

    fn is_one(&self) -> bool {
        *self == bf16::ONE
    }

    fn from_f64(value: f64) -> Option<Self> {
        Some(bf16::from_f64(value))
    }

    fn to_f64(&self) -> Option<f64> {
        Some(bf16::to_f64(*self))
    }
}

impl FloatElement for bf16 {
    fn epsilon() -> Self {
        bf16::EPSILON
    }

    fn is_finite(&self) -> bool {
        (*self).is_finite()
    }

    fn is_infinite(&self) -> bool {
        (*self).is_infinite()
    }

    fn is_nan(&self) -> bool {
        (*self).is_nan()
    }

    fn infinity() -> Self {
        bf16::INFINITY
    }

    fn neg_infinity() -> Self {
        bf16::NEG_INFINITY
    }

    fn nan() -> Self {
        bf16::NAN
    }
}

/// Type aliases for complex numbers
pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

/// Implementation for 32-bit complex numbers (Complex32)
impl TensorElement for Complex32 {
    fn dtype() -> DType {
        DType::C64
    }

    fn zero() -> Self {
        Complex32::new(0.0, 0.0)
    }

    fn one() -> Self {
        Complex32::new(1.0, 0.0)
    }

    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    fn is_one(&self) -> bool {
        self.re == 1.0 && self.im == 0.0
    }

    fn from_f64(value: f64) -> Option<Self> {
        Some(Complex32::new(value as f32, 0.0))
    }

    fn to_f64(&self) -> Option<f64> {
        if self.im == 0.0 {
            Some(self.re as f64)
        } else {
            None // Cannot represent complex number as real
        }
    }
}

/// Implementation for 64-bit complex numbers (Complex64)
impl TensorElement for Complex64 {
    fn dtype() -> DType {
        DType::C128
    }

    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }

    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }

    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    fn is_one(&self) -> bool {
        self.re == 1.0 && self.im == 0.0
    }

    fn from_f64(value: f64) -> Option<Self> {
        Some(Complex64::new(value, 0.0))
    }

    fn to_f64(&self) -> Option<f64> {
        if self.im == 0.0 {
            Some(self.re)
        } else {
            None // Cannot represent complex number as real
        }
    }
}

/// Trait for advanced tensor element operations
///
/// This trait provides additional operations that are useful for tensor
/// computation but not required for basic tensor storage.
pub trait AdvancedTensorElement: TensorElement {
    /// Compute absolute value
    fn abs(&self) -> Self;

    /// Compute square root
    fn sqrt(&self) -> Self;

    /// Compute exponential function
    fn exp(&self) -> Self;

    /// Compute natural logarithm
    fn ln(&self) -> Self;

    /// Compute sine
    fn sin(&self) -> Self;

    /// Compute cosine
    fn cos(&self) -> Self;

    /// Compute power function
    fn powf(&self, exp: Self) -> Self;

    /// Minimum of two values
    fn min(&self, other: Self) -> Self;

    /// Maximum of two values
    fn max(&self, other: Self) -> Self;
}

/// Implement AdvancedTensorElement for floating point types
impl AdvancedTensorElement for f32 {
    fn abs(&self) -> Self {
        (*self).abs()
    }
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }
    fn exp(&self) -> Self {
        (*self).exp()
    }
    fn ln(&self) -> Self {
        (*self).ln()
    }
    fn sin(&self) -> Self {
        (*self).sin()
    }
    fn cos(&self) -> Self {
        (*self).cos()
    }
    fn powf(&self, exp: Self) -> Self {
        (*self).powf(exp)
    }
    fn min(&self, other: Self) -> Self {
        (*self).min(other)
    }
    fn max(&self, other: Self) -> Self {
        (*self).max(other)
    }
}

impl AdvancedTensorElement for f64 {
    fn abs(&self) -> Self {
        (*self).abs()
    }
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }
    fn exp(&self) -> Self {
        (*self).exp()
    }
    fn ln(&self) -> Self {
        (*self).ln()
    }
    fn sin(&self) -> Self {
        (*self).sin()
    }
    fn cos(&self) -> Self {
        (*self).cos()
    }
    fn powf(&self, exp: Self) -> Self {
        (*self).powf(exp)
    }
    fn min(&self, other: Self) -> Self {
        (*self).min(other)
    }
    fn max(&self, other: Self) -> Self {
        (*self).max(other)
    }
}

/// Trait for types that support bitwise operations
///
/// This trait is useful for integer types and boolean operations in tensors.
pub trait BitwiseTensorElement: TensorElement {
    /// Bitwise AND operation
    fn bitand(&self, other: Self) -> Self;

    /// Bitwise OR operation
    fn bitor(&self, other: Self) -> Self;

    /// Bitwise XOR operation
    fn bitxor(&self, other: Self) -> Self;

    /// Bitwise NOT operation
    fn not(&self) -> Self;

    /// Left shift operation
    fn shl(&self, shift: u32) -> Self;

    /// Right shift operation
    fn shr(&self, shift: u32) -> Self;
}

/// Implement BitwiseTensorElement for integer types
macro_rules! impl_bitwise_tensor_element {
    ($ty:ty) => {
        impl BitwiseTensorElement for $ty {
            fn bitand(&self, other: Self) -> Self {
                self & other
            }
            fn bitor(&self, other: Self) -> Self {
                self | other
            }
            fn bitxor(&self, other: Self) -> Self {
                self ^ other
            }
            fn not(&self) -> Self {
                !self
            }
            fn shl(&self, shift: u32) -> Self {
                self << shift
            }
            fn shr(&self, shift: u32) -> Self {
                self >> shift
            }
        }
    };
}

impl_bitwise_tensor_element!(u8);
impl_bitwise_tensor_element!(i8);
impl_bitwise_tensor_element!(i16);
impl_bitwise_tensor_element!(i32);
impl_bitwise_tensor_element!(u32);
impl_bitwise_tensor_element!(i64);
impl_bitwise_tensor_element!(u64);

impl BitwiseTensorElement for bool {
    fn bitand(&self, other: Self) -> Self {
        *self && other
    }
    fn bitor(&self, other: Self) -> Self {
        *self || other
    }
    fn bitxor(&self, other: Self) -> Self {
        *self != other
    }
    fn not(&self) -> Self {
        !*self
    }
    fn shl(&self, _shift: u32) -> Self {
        *self
    }
    fn shr(&self, _shift: u32) -> Self {
        *self
    }
}

/// Utility trait for type erasure and dynamic dispatch
///
/// This trait provides dyn-compatible operations for tensor elements.
/// Separated from TensorElement to avoid dyn-compatibility issues with PartialEq.
pub trait AnyTensorElement: Any + Send + Sync + fmt::Debug {
    /// Get the type as Any for downcasting
    fn as_any(&self) -> &dyn Any;

    /// Clone this value into a Box<dyn AnyTensorElement>
    fn clone_boxed(&self) -> Box<dyn AnyTensorElement>;

    /// Get a string representation of this value
    fn to_string(&self) -> String;

    /// Get the DType for this element
    fn dtype(&self) -> DType;

    /// Convert to f64 if possible
    fn to_f64_dyn(&self) -> Option<f64>;
}

/// Blanket implementation for all TensorElement types
impl<T: TensorElement + Any> AnyTensorElement for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn AnyTensorElement> {
        Box::new(*self)
    }

    fn to_string(&self) -> String {
        format!("{:?}", self)
    }

    fn dtype(&self) -> DType {
        T::dtype()
    }

    fn to_f64_dyn(&self) -> Option<f64> {
        self.to_f64()
    }
}

/// Helper function to check if two tensor element types are compatible
///
/// Returns true if the two types can be used together in tensor operations
/// without explicit conversion.
pub fn types_compatible<T1: TensorElement, T2: TensorElement>() -> bool {
    T1::dtype() == T2::dtype()
}

/// Helper function to get the promoted type for two tensor element types
///
/// Returns the DType that should be used when combining two different types
/// in tensor operations.
pub fn promote_types<T1: TensorElement, T2: TensorElement>() -> DType {
    let dtype1 = T1::dtype();
    let dtype2 = T2::dtype();

    // If types are the same, no promotion needed
    if dtype1 == dtype2 {
        return dtype1;
    }

    // Simple promotion rules (can be expanded)
    match (dtype1, dtype2) {
        // Float types always promote to the larger precision
        (DType::F32, DType::F64) | (DType::F64, DType::F32) => DType::F64,
        (DType::F16, DType::F32) | (DType::F32, DType::F16) => DType::F32,
        (DType::F16, DType::F64) | (DType::F64, DType::F16) => DType::F64,

        // Integer to float promotions
        (dt1, dt2) if dt1.is_int() && dt2.is_float() => dt2,
        (dt1, dt2) if dt1.is_float() && dt2.is_int() => dt1,

        // Integer promotions - promote to larger size
        (DType::I8, DType::I16) | (DType::I16, DType::I8) => DType::I16,
        (DType::I8, DType::I32) | (DType::I32, DType::I8) => DType::I32,
        (DType::I16, DType::I32) | (DType::I32, DType::I16) => DType::I32,
        (DType::I32, DType::I64) | (DType::I64, DType::I32) => DType::I64,

        // Default: promote to F32 for mixed operations
        _ => DType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_element_basic_types() {
        // Test that basic types implement TensorElement correctly
        assert_eq!(<f32 as TensorElement>::dtype(), DType::F32);
        assert_eq!(<f64 as TensorElement>::dtype(), DType::F64);
        assert_eq!(<i32 as TensorElement>::dtype(), DType::I32);
        assert_eq!(<bool as TensorElement>::dtype(), DType::Bool);

        // Test zero and one values
        assert_eq!(<f32 as TensorElement>::zero(), 0.0f32);
        assert_eq!(<f32 as TensorElement>::one(), 1.0f32);
        assert_eq!(<i32 as TensorElement>::zero(), 0i32);
        assert_eq!(<i32 as TensorElement>::one(), 1i32);
        assert!(!<bool as TensorElement>::zero());
        assert!(<bool as TensorElement>::one());
    }

    #[test]
    fn test_float_element_special_values() {
        // Test special floating point values
        assert!(<f32 as FloatElement>::nan().is_nan());
        assert!(<f32 as FloatElement>::infinity().is_infinite());
        assert!(<f32 as FloatElement>::neg_infinity().is_infinite());

        assert!(<f64 as FloatElement>::nan().is_nan());
        assert!(<f64 as FloatElement>::infinity().is_infinite());
        assert!(<f64 as FloatElement>::neg_infinity().is_infinite());
    }

    #[test]
    fn test_complex_types() {
        // Test complex number implementations
        let c32_zero = <Complex<f32> as TensorElement>::zero();
        let c32_one = <Complex<f32> as TensorElement>::one();

        assert!(TensorElement::is_zero(&c32_zero));
        assert!(TensorElement::is_one(&c32_one));
        assert_eq!(<Complex<f32> as TensorElement>::dtype(), DType::C64);

        let c64_zero = <Complex<f64> as TensorElement>::zero();
        let c64_one = <Complex<f64> as TensorElement>::one();

        assert!(TensorElement::is_zero(&c64_zero));
        assert!(TensorElement::is_one(&c64_one));
        assert_eq!(<Complex<f64> as TensorElement>::dtype(), DType::C128);
    }

    #[test]
    fn test_half_precision_types() {
        // Test f16 implementation
        let f16_zero = <half::f16 as TensorElement>::zero();
        let f16_one = <half::f16 as TensorElement>::one();

        assert!(TensorElement::is_zero(&f16_zero));
        assert!(TensorElement::is_one(&f16_one));
        assert_eq!(<half::f16 as TensorElement>::dtype(), DType::F16);

        // Test bf16 implementation
        let bf16_zero = <half::bf16 as TensorElement>::zero();
        let bf16_one = <half::bf16 as TensorElement>::one();

        assert!(TensorElement::is_zero(&bf16_zero));
        assert!(TensorElement::is_one(&bf16_one));
        assert_eq!(<half::bf16 as TensorElement>::dtype(), DType::BF16);
    }

    #[test]
    fn test_conversion_methods() {
        // Test f64 conversion
        assert_eq!(42i32.to_f64(), Some(42.0));
        assert_eq!(std::f32::consts::PI.to_f64(), Some(3.1415927410125732)); // f32 precision
        assert_eq!(i32::from_f64(42.0), Some(42i32));
        assert_eq!(bool::from_f64(1.0), Some(true));
        assert_eq!(bool::from_f64(0.0), Some(false));
        assert_eq!(bool::from_f64(0.5), None); // Invalid bool conversion
    }

    #[test]
    fn test_advanced_tensor_element() {
        // Test advanced operations on f32
        let x = 4.0f32;
        assert_eq!(x.sqrt(), 2.0);
        assert_eq!(x.abs(), 4.0);
        assert_eq!((-x).abs(), 4.0);

        let y = 2.0f32;
        assert_eq!(x.max(y), 4.0);
        assert_eq!(x.min(y), 2.0);
    }

    #[test]
    fn test_bitwise_operations() {
        // Test bitwise operations on integers
        let a = 5u8; // 0101
        let b = 3u8; // 0011

        assert_eq!(a.bitand(b), 1u8); // 0001
        assert_eq!(a.bitor(b), 7u8); // 0111
        assert_eq!(a.bitxor(b), 6u8); // 0110
        assert_eq!(a.not(), !5u8); // 11111010

        // Test bitwise operations on booleans
        assert!(!true.bitand(false));
        assert!(true.bitor(false));
        assert!(true.bitxor(false));
        assert!(!true.not());
    }

    #[test]
    fn test_type_compatibility() {
        // Test type compatibility checks
        assert!(types_compatible::<f32, f32>());
        assert!(!types_compatible::<f32, f64>());
        assert!(!types_compatible::<i32, f32>());
    }

    #[test]
    fn test_type_promotion() {
        // Test type promotion rules
        assert_eq!(promote_types::<f32, f64>(), DType::F64);
        assert_eq!(promote_types::<f16, f32>(), DType::F32);
        assert_eq!(promote_types::<i32, f32>(), DType::F32);
        assert_eq!(promote_types::<i8, i32>(), DType::I32);
        assert_eq!(promote_types::<i32, i32>(), DType::I32); // Same type
    }

    #[test]
    fn test_type_info() {
        // Test type information
        assert!(f32::type_info().contains("f32"));
        assert!(i32::type_info().contains("i32"));
        assert!(bool::type_info().contains("bool"));
    }
}
