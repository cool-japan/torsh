// Data Types: Type Promotion and Conversion System
//
// This module implements sophisticated type promotion and conversion logic for the ToRSh
// tensor framework. It handles automatic type promotion during tensor operations,
// providing safe and predictable behavior when mixing different numeric types.

use crate::dtype::core::DType;
use std::collections::HashMap;

/// Trait for automatic type promotion during tensor operations
///
/// This trait defines how types should be promoted when combined in operations.
/// The promotion system ensures that no precision is lost and follows common
/// mathematical conventions for numeric type hierarchies.
pub trait TypePromotion {
    /// Promote two types to a common type that can represent both
    ///
    /// Returns the promoted type that can safely represent values from both
    /// input types without loss of precision.
    fn promote_types(type1: DType, type2: DType) -> DType;

    /// Check if a type can be safely converted to another type
    ///
    /// Returns true if the conversion would not lose precision or change
    /// the mathematical meaning of values.
    fn can_promote_to(from: DType, to: DType) -> bool;

    /// Get the promotion precedence of a type
    ///
    /// Lower numbers indicate higher precedence in promotion decisions.
    /// Types with higher precedence tend to be promoted to during operations.
    fn promotion_precedence(dtype: DType) -> u8;

    /// Find the result type for a sequence of operations
    ///
    /// Given multiple types that will be combined in operations,
    /// determines the final result type.
    fn result_type(types: &[DType]) -> DType;

    /// Check if implicit conversion is allowed
    ///
    /// Some conversions may be mathematically valid but should require
    /// explicit user intent (e.g., float to integer).
    fn allows_implicit_conversion(from: DType, to: DType) -> bool;

    /// Find the common type for a collection of types
    ///
    /// Returns the type that can represent all the given types without
    /// loss of precision, or None if no such type exists.
    fn common_type(types: &[DType]) -> Option<DType> {
        if types.is_empty() {
            return None;
        }
        Some(Self::result_type(types))
    }
}

impl TypePromotion for DType {
    fn promote_types(type1: DType, type2: DType) -> DType {
        if type1 == type2 {
            return type1;
        }

        // Special case: Boolean promotion
        if type1 == DType::Bool {
            #[allow(clippy::if_same_then_else)] // Type promotion logic - prefer non-boolean type
            return if type2.is_int() {
                type2
            } else if type2.is_complex() {
                type2
            } else {
                DType::F32
            };
        }
        if type2 == DType::Bool {
            #[allow(clippy::if_same_then_else)] // Type promotion logic - prefer non-boolean type
            return if type1.is_int() {
                type1
            } else if type1.is_complex() {
                type1
            } else {
                DType::F32
            };
        }

        // Complex number promotion rules
        if type1.is_complex() || type2.is_complex() {
            return promote_complex_types(type1, type2);
        }

        // Quantized type promotion rules
        if type1.is_quantized() || type2.is_quantized() {
            return promote_quantized_types(type1, type2);
        }

        // Floating point promotion rules
        if type1.is_float() || type2.is_float() {
            return promote_float_types(type1, type2);
        }

        // Integer promotion rules
        if type1.is_int() && type2.is_int() {
            return promote_integer_types(type1, type2);
        }

        // Default: promote to F32 for mixed cases
        DType::F32
    }

    fn can_promote_to(from: DType, to: DType) -> bool {
        if from == to {
            return true;
        }

        match (from, to) {
            // Boolean can promote to any type
            (DType::Bool, _) => true,

            // Integer widening promotions
            (DType::I8, DType::I16 | DType::I32 | DType::I64) => true,
            (DType::I16, DType::I32 | DType::I64) => true,
            (DType::I32, DType::I64) => true,
            (DType::U8, DType::U32 | DType::U64 | DType::I16 | DType::I32 | DType::I64) => true,
            (DType::U32, DType::U64 | DType::I64) => true,

            // Integer to float promotions
            (dt_from, dt_to) if dt_from.is_int() && dt_to.is_float() => {
                // Check if the float type can represent the full range
                match (dt_from, dt_to) {
                    (DType::I64 | DType::U64, DType::F16 | DType::BF16) => false, // Too much precision loss
                    (DType::I32 | DType::U32, DType::F16 | DType::BF16) => false, // Too much precision loss
                    _ => true,
                }
            }

            // Float promotions
            (DType::F16, DType::F32 | DType::F64 | DType::BF16) => true,
            (DType::BF16, DType::F32 | DType::F64) => true,
            (DType::F32, DType::F64) => true,

            // Complex promotions
            (DType::C64, DType::C128) => true,
            (dt_from, dt_to) if !dt_from.is_complex() && dt_to.is_complex() => {
                // Non-complex can promote to complex if the component type works
                match dt_to {
                    DType::C64 => Self::can_promote_to(dt_from, DType::F32),
                    DType::C128 => Self::can_promote_to(dt_from, DType::F64),
                    _ => false,
                }
            }

            _ => false,
        }
    }

    fn promotion_precedence(dtype: DType) -> u8 {
        match dtype {
            // Boolean has lowest precedence (gets promoted easily)
            DType::Bool => 0,

            // Integers by size and signedness
            DType::I8 => 10,
            DType::U8 => 11,
            DType::I16 => 20,
            DType::I32 => 30,
            DType::U32 => 31,
            DType::I64 => 40,
            DType::U64 => 41,

            // Quantized types (special handling)
            DType::QInt8 => 15,
            DType::QUInt8 => 16,
            DType::QInt32 => 35, // Higher precision quantized

            // Floating point types by precision
            DType::F16 => 50,
            DType::BF16 => 51,
            DType::F32 => 60,
            DType::F64 => 70,

            // Complex types have highest precedence
            DType::C64 => 80,
            DType::C128 => 90,
        }
    }

    fn result_type(types: &[DType]) -> DType {
        if types.is_empty() {
            return DType::F32; // Default
        }

        if types.len() == 1 {
            return types[0];
        }

        // Find the type with highest precedence
        let mut result = types[0];
        for &dtype in &types[1..] {
            result = Self::promote_types(result, dtype);
        }

        result
    }

    fn allows_implicit_conversion(from: DType, to: DType) -> bool {
        // Only allow implicit conversions that don't lose precision
        match (from, to) {
            // Same type is always allowed
            (a, b) if a == b => true,

            // Boolean to any numeric type
            (DType::Bool, _) if to.is_int() || to.is_float() => true,

            // Integer widening
            (DType::I8, DType::I16 | DType::I32 | DType::I64) => true,
            (DType::I16, DType::I32 | DType::I64) => true,
            (DType::I32, DType::I64) => true,
            (DType::U8, DType::U32 | DType::U64 | DType::I16 | DType::I32 | DType::I64) => true,
            (DType::U32, DType::U64) => true,

            // Float widening
            (DType::F16, DType::F32 | DType::F64) => true,
            (DType::BF16, DType::F32 | DType::F64) => true,
            (DType::F32, DType::F64) => true,

            // Integer to float (with precision check)
            (DType::I8 | DType::U8, DType::F32 | DType::F64) => true,
            (DType::I16, DType::F32 | DType::F64) => true,
            (DType::I32, DType::F64) => true, // F32 might lose precision for large I32

            // Real to complex
            (dt_from, DType::C64) if !dt_from.is_complex() => {
                Self::allows_implicit_conversion(dt_from, DType::F32)
            }
            (dt_from, DType::C128) if !dt_from.is_complex() => {
                Self::allows_implicit_conversion(dt_from, DType::F64)
            }

            // Complex widening
            (DType::C64, DType::C128) => true,

            // Disallow potentially lossy conversions
            _ => false,
        }
    }
}

/// Promote two complex types or promote non-complex to complex
fn promote_complex_types(type1: DType, type2: DType) -> DType {
    match (type1.is_complex(), type2.is_complex()) {
        (true, true) => {
            // Both complex - promote to higher precision
            match (type1, type2) {
                (DType::C64, DType::C128) | (DType::C128, DType::C64) => DType::C128,
                (DType::C64, DType::C64) => DType::C64,
                (DType::C128, DType::C128) => DType::C128,
                _ => DType::C128, // Default to highest precision
            }
        }
        (true, false) => {
            // type1 is complex, type2 is not
            match type1 {
                DType::C64 => {
                    if DType::can_promote_to(type2, DType::F32) {
                        DType::C64
                    } else {
                        DType::C128
                    }
                }
                DType::C128 => DType::C128,
                _ => DType::C128,
            }
        }
        (false, true) => {
            // type2 is complex, type1 is not
            promote_complex_types(type2, type1)
        }
        (false, false) => {
            // Neither is complex - shouldn't happen in this function
            DType::C64
        }
    }
}

/// Promote quantized types with special handling
fn promote_quantized_types(type1: DType, type2: DType) -> DType {
    match (type1.is_quantized(), type2.is_quantized()) {
        (true, true) => {
            // Both quantized - promote to wider quantized type or float
            match (type1, type2) {
                (DType::QInt8, DType::QUInt8) | (DType::QUInt8, DType::QInt8) => DType::F32,
                (DType::QInt8, DType::QInt8) => DType::QInt8,
                (DType::QUInt8, DType::QUInt8) => DType::QUInt8,
                _ => DType::F32,
            }
        }
        (true, false) => {
            // One quantized, one not - promote to appropriate float
            if type2.is_float() {
                type2
            } else {
                DType::F32
            }
        }
        (false, true) => {
            // One quantized, one not
            promote_quantized_types(type2, type1)
        }
        (false, false) => {
            // Neither quantized - shouldn't happen
            DType::F32
        }
    }
}

/// Promote floating point types to higher precision
fn promote_float_types(type1: DType, type2: DType) -> DType {
    let float_precedence = |dt: DType| -> u8 {
        match dt {
            DType::F16 => 1,
            DType::BF16 => 2,
            DType::F32 => 3,
            DType::F64 => 4,
            _ => 0, // Non-float types get lowest precedence
        }
    };

    let prec1 = float_precedence(type1);
    let prec2 = float_precedence(type2);

    match prec1.max(prec2) {
        4 => DType::F64,
        3 => DType::F32,
        2 => DType::BF16,
        1 => DType::F16,
        _ => DType::F32, // Default for mixed cases
    }
}

/// Promote integer types based on size and signedness
fn promote_integer_types(type1: DType, type2: DType) -> DType {
    let int_info = |dt: DType| -> (u8, bool) {
        // (size_bytes, is_signed)
        match dt {
            DType::I8 => (1, true),
            DType::U8 => (1, false),
            DType::I16 => (2, true),
            DType::I32 => (4, true),
            DType::U32 => (4, false),
            DType::I64 => (8, true),
            DType::U64 => (8, false),
            _ => (0, false),
        }
    };

    let (size1, signed1) = int_info(type1);
    let (size2, signed2) = int_info(type2);

    // Promote to larger size
    let target_size = size1.max(size2);

    // Special case: if one is signed and other is unsigned of same size, promote to next size
    if size1 == size2 && signed1 != signed2 {
        match target_size {
            1 => return DType::I16, // I8 + U8 -> I16
            4 => return DType::I64, // I32 + U32 -> I64
            8 => return DType::F64, // I64 + U64 -> F64 (no larger integer type)
            _ => {}
        }
    }

    // If either is signed and sizes are equal, prefer signed
    let prefer_signed = (signed1 || signed2) && (size1 == size2);

    match (target_size, prefer_signed || (signed1 && signed2)) {
        (1, true) => DType::I8,
        (1, false) => DType::U8,
        (2, _) => DType::I16, // I16 can represent all U8 and I8 values
        (4, true) => DType::I32,
        (4, false) => DType::U32,
        (8, true) => DType::I64,
        (8, false) => DType::U64,
        _ => DType::I32, // Fallback
    }
}

/// Trait for automatic promotion with specific types
///
/// This trait allows for more specific promotion rules when working
/// with concrete type parameters.
pub trait AutoPromote<T> {
    /// Get the promoted type when combining with type T
    fn auto_promote() -> DType;
}

impl AutoPromote<DType> for DType {
    fn auto_promote() -> DType {
        DType::F32 // Default promotion target
    }
}

/// Macro for implementing AutoPromote for specific type pairs
macro_rules! impl_auto_promote {
    ($from:ty, $to:expr) => {
        impl AutoPromote<$from> for DType {
            fn auto_promote() -> DType {
                $to
            }
        }
    };
}

// Implement common promotion targets
impl_auto_promote!(f32, DType::F32);
impl_auto_promote!(f64, DType::F64);
impl_auto_promote!(i32, DType::I32);
impl_auto_promote!(i64, DType::I64);
impl_auto_promote!(bool, DType::Bool);

/// Type promotion matrix for efficient lookup
pub struct PromotionMatrix {
    matrix: HashMap<(DType, DType), DType>,
}

impl PromotionMatrix {
    /// Create a new promotion matrix with precomputed results
    pub fn new() -> Self {
        let mut matrix = HashMap::new();

        // Precompute common promotion pairs for performance
        let types = [
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::I32,
            DType::U32,
            DType::I64,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
            DType::C64,
            DType::C128,
            DType::QInt8,
            DType::QUInt8,
        ];

        for &type1 in &types {
            for &type2 in &types {
                let promoted = DType::promote_types(type1, type2);
                matrix.insert((type1, type2), promoted);
            }
        }

        Self { matrix }
    }

    /// Fast lookup of promotion result
    pub fn promote(&self, type1: DType, type2: DType) -> DType {
        self.matrix
            .get(&(type1, type2))
            .copied()
            .unwrap_or_else(|| DType::promote_types(type1, type2))
    }

    /// Check if promotion is symmetric
    pub fn is_symmetric(&self, type1: DType, type2: DType) -> bool {
        self.promote(type1, type2) == self.promote(type2, type1)
    }
}

impl Default for PromotionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// Common type promotion utilities
pub mod utils {
    use super::*;

    /// Find the common type for a slice of DTypes
    pub fn common_type(types: &[DType]) -> Option<DType> {
        if types.is_empty() {
            return None;
        }

        Some(DType::result_type(types))
    }

    /// Check if all types in a slice can be promoted to a target type
    pub fn all_promotable_to(types: &[DType], target: DType) -> bool {
        types.iter().all(|&dt| DType::can_promote_to(dt, target))
    }

    /// Find the minimal type that can represent all given types
    pub fn minimal_common_type(types: &[DType]) -> Option<DType> {
        if types.is_empty() {
            return None;
        }

        // Start with the first type and find minimal promotion
        let mut result = types[0];
        for &dtype in &types[1..] {
            result = DType::promote_types(result, dtype);
        }

        Some(result)
    }

    /// Check if a promotion chain is valid
    pub fn valid_promotion_chain(chain: &[DType]) -> bool {
        if chain.len() < 2 {
            return true;
        }

        for window in chain.windows(2) {
            if !DType::can_promote_to(window[0], window[1]) {
                return false;
            }
        }

        true
    }

    /// Get the precision level of a type (higher = more precise)
    pub fn precision_level(dtype: DType) -> u8 {
        match dtype {
            DType::Bool => 1,
            DType::I8 | DType::U8 => 8,
            DType::I16 => 16,
            DType::F16 | DType::BF16 => 16,
            DType::I32 | DType::U32 | DType::F32 => 32,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 64,
            DType::C128 => 128,
            DType::QInt8 | DType::QUInt8 => 8, // Quantized types have effective 8-bit precision
            DType::QInt32 => 32,               // QInt32 has effective 32-bit precision
        }
    }

    /// Check if promotion would lose precision
    pub fn loses_precision(from: DType, to: DType) -> bool {
        precision_level(from) > precision_level(to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_promotion() {
        // Same types
        assert_eq!(DType::promote_types(DType::F32, DType::F32), DType::F32);

        // Integer promotions
        assert_eq!(DType::promote_types(DType::I8, DType::I16), DType::I16);
        assert_eq!(DType::promote_types(DType::I16, DType::I32), DType::I32);
        assert_eq!(DType::promote_types(DType::I32, DType::I64), DType::I64);

        // Float promotions
        assert_eq!(DType::promote_types(DType::F16, DType::F32), DType::F32);
        assert_eq!(DType::promote_types(DType::F32, DType::F64), DType::F64);

        // Integer to float
        assert_eq!(DType::promote_types(DType::I32, DType::F32), DType::F32);
        assert_eq!(DType::promote_types(DType::I64, DType::F64), DType::F64);
    }

    #[test]
    fn test_complex_promotion() {
        // Real to complex
        assert_eq!(DType::promote_types(DType::F32, DType::C64), DType::C64);
        assert_eq!(DType::promote_types(DType::F64, DType::C128), DType::C128);

        // Complex to complex
        assert_eq!(DType::promote_types(DType::C64, DType::C128), DType::C128);
    }

    #[test]
    fn test_boolean_promotion() {
        // Boolean to integer
        assert_eq!(DType::promote_types(DType::Bool, DType::I32), DType::I32);
        assert_eq!(DType::promote_types(DType::Bool, DType::U8), DType::U8);

        // Boolean to float
        assert_eq!(DType::promote_types(DType::Bool, DType::F32), DType::F32);
    }

    #[test]
    fn test_quantized_promotion() {
        // Quantized to float
        assert_eq!(DType::promote_types(DType::QInt8, DType::F32), DType::F32);

        // Mixed quantized
        assert_eq!(
            DType::promote_types(DType::QInt8, DType::QUInt8),
            DType::F32
        );
    }

    #[test]
    fn test_can_promote_to() {
        // Valid promotions
        assert!(DType::can_promote_to(DType::I8, DType::I16));
        assert!(DType::can_promote_to(DType::F32, DType::F64));
        assert!(DType::can_promote_to(DType::Bool, DType::I32));

        // Invalid promotions (precision loss)
        assert!(!DType::can_promote_to(DType::F64, DType::F32));
        assert!(!DType::can_promote_to(DType::I32, DType::I16));
    }

    #[test]
    fn test_promotion_precedence() {
        assert!(DType::promotion_precedence(DType::Bool) < DType::promotion_precedence(DType::I32));
        assert!(DType::promotion_precedence(DType::I32) < DType::promotion_precedence(DType::F32));
        assert!(DType::promotion_precedence(DType::F32) < DType::promotion_precedence(DType::C64));
    }

    #[test]
    fn test_result_type() {
        let types = vec![DType::I8, DType::F32, DType::I16];
        assert_eq!(DType::result_type(&types), DType::F32);

        let complex_types = vec![DType::F32, DType::C64];
        assert_eq!(DType::result_type(&complex_types), DType::C64);
    }

    #[test]
    fn test_implicit_conversion() {
        // Allowed implicit conversions
        assert!(DType::allows_implicit_conversion(DType::I8, DType::I16));
        assert!(DType::allows_implicit_conversion(DType::F32, DType::F64));
        assert!(DType::allows_implicit_conversion(DType::Bool, DType::F32));

        // Disallowed implicit conversions
        assert!(!DType::allows_implicit_conversion(DType::F32, DType::I32));
        assert!(!DType::allows_implicit_conversion(DType::F64, DType::F32));
    }

    #[test]
    fn test_promotion_matrix() {
        let matrix = PromotionMatrix::new();

        // Test fast lookup
        assert_eq!(matrix.promote(DType::I32, DType::F32), DType::F32);
        assert_eq!(matrix.promote(DType::F16, DType::F64), DType::F64);

        // Test symmetry for commutative operations
        assert!(matrix.is_symmetric(DType::I32, DType::F32));
        assert!(matrix.is_symmetric(DType::F16, DType::F32));
    }

    #[test]
    fn test_utils_functions() {
        let types = vec![DType::I8, DType::I16, DType::I32];

        // Test common type
        assert_eq!(utils::common_type(&types), Some(DType::I32));

        // Test all promotable
        assert!(utils::all_promotable_to(&types, DType::I64));
        assert!(!utils::all_promotable_to(&types, DType::I8));

        // Test minimal common type
        assert_eq!(utils::minimal_common_type(&types), Some(DType::I32));

        // Test promotion chain
        let valid_chain = vec![DType::I8, DType::I16, DType::I32];
        assert!(utils::valid_promotion_chain(&valid_chain));

        let invalid_chain = vec![DType::I32, DType::I16, DType::I8];
        assert!(!utils::valid_promotion_chain(&invalid_chain));
    }

    #[test]
    fn test_precision_levels() {
        assert!(utils::precision_level(DType::F64) > utils::precision_level(DType::F32));
        assert!(utils::precision_level(DType::I32) > utils::precision_level(DType::I16));
        assert!(utils::precision_level(DType::C128) > utils::precision_level(DType::C64));

        // Test precision loss detection
        assert!(utils::loses_precision(DType::F64, DType::F32));
        assert!(!utils::loses_precision(DType::F32, DType::F64));
    }

    #[test]
    fn test_edge_cases() {
        // Mixed signed/unsigned of same size
        assert_eq!(DType::promote_types(DType::I32, DType::U32), DType::I64);

        // Very different types
        assert_eq!(DType::promote_types(DType::Bool, DType::C128), DType::C128);

        // Empty type list
        assert_eq!(utils::common_type(&[]), None);
        assert_eq!(utils::minimal_common_type(&[]), None);
    }
}
