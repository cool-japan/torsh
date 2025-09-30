// Data Types: Unified Module Interface
//
// This module provides a unified interface for the ToRSh data type system,
// orchestrating core types, tensor element traits, specialized numeric types,
// type promotion logic, and custom type extensions.
//
// ## Architecture
//
// The data type system is organized into seven specialized modules:
//
// - **core**: Basic DType enum and fundamental operations
// - **traits**: TensorElement trait and core abstractions
// - **bfloat16**: BFloat16 operations with precise rounding control
// - **quantized**: Quantized integer types (QInt8, QUInt8)
// - **complex**: Complex number extensions and operations
// - **promotion**: Type promotion and conversion system
// - **custom**: Extensible custom type system and registry
//
// ## Usage
//
// ```rust
// use torsh_core::dtype::{DType, TensorElement, TypePromotion};
//
// // Basic type operations
// let size = DType::F32.size();
// let is_float = DType::F64.is_float();
//
// // Type promotion
// let promoted = DType::promote_types(DType::I32, DType::F32);
//
// // Custom types
// use torsh_core::dtype::{CustomDTypeRegistry, CustomTensorElement};
// CustomDTypeRegistry::register::<MyCustomType>()?;
// ```

// Re-export all public APIs from specialized modules
#[allow(unused_imports)] // Public API re-exports
pub use self::bfloat16::*;
#[allow(unused_imports)] // Public API re-exports
pub use self::complex::*;
#[allow(unused_imports)] // Public API re-exports
pub use self::core::*;
#[allow(unused_imports)] // Public API re-exports
pub use self::custom::*;
#[allow(unused_imports)] // Public API re-exports
pub use self::promotion::*;
#[allow(unused_imports)] // Public API re-exports
pub use self::quantized::*;
#[allow(unused_imports)] // Public API re-exports
pub use self::traits::*;

// Module declarations
pub mod bfloat16;
pub mod complex;
pub mod core;
pub mod custom;
pub mod promotion;
pub mod quantized;
pub mod traits;

// Re-export commonly used external types for convenience
pub use half::{bf16, f16};
pub use num_complex::Complex;

/// Convenience type aliases for common use cases
pub type Float16 = f16;
pub type BFloat16 = bf16;
pub type Float32 = f32;
pub type Float64 = f64;
pub type Int8 = i8;
pub type Int16 = i16;
pub type Int32 = i32;
pub type Int64 = i64;
pub type UInt8 = u8;
pub type UInt16 = u16;
pub type UInt32 = u32;
pub type UInt64 = u64;

/// Global type system manager
///
/// This structure provides a unified interface for managing the entire type system,
/// including standard types, custom types, and type conversions.
pub struct TypeSystem {
    promotion_matrix: promotion::PromotionMatrix,
    #[allow(dead_code)] // Custom type manager - future implementation
    custom_manager: custom::CustomTypeManager,
}

impl TypeSystem {
    /// Create a new type system instance
    pub fn new() -> Self {
        Self {
            promotion_matrix: promotion::PromotionMatrix::new(),
            custom_manager: custom::CustomTypeManager::new(),
        }
    }

    /// Get the default global type system instance
    pub fn global() -> &'static TypeSystem {
        static INSTANCE: std::sync::OnceLock<TypeSystem> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(TypeSystem::new)
    }

    /// Register a custom type in the global registry
    pub fn register_custom_type<T: CustomTensorElement>() -> Result<(), crate::error::TorshError> {
        CustomDTypeRegistry::register::<T>()
    }

    /// Promote two types using the optimized matrix
    pub fn promote_types(&self, type1: DType, type2: DType) -> DType {
        self.promotion_matrix.promote(type1, type2)
    }

    /// Check type compatibility
    ///
    /// Types are compatible if they are the same or can be promoted within the same category.
    /// Bool is not compatible with float/integer types even though promotion is possible.
    pub fn are_compatible(&self, type1: DType, type2: DType) -> bool {
        if type1 == type2 {
            return true;
        }

        // Bool is only compatible with bool
        if type1 == DType::Bool || type2 == DType::Bool {
            return false;
        }

        // Otherwise check if promotion is possible
        DType::can_promote_to(type1, type2) || DType::can_promote_to(type2, type1)
    }

    /// Get the result type for multiple types
    pub fn result_type(&self, types: &[DType]) -> Option<DType> {
        if types.is_empty() {
            return None;
        }

        let mut result = types[0];
        for &dtype in &types[1..] {
            result = self.promote_types(result, dtype);
        }

        Some(result)
    }

    /// Check if a type supports specific operations
    pub fn supports_operation(&self, dtype: DType, operation: &str) -> bool {
        match operation {
            "arithmetic" => dtype.is_int() || dtype.is_float() || dtype.is_complex(),
            "comparison" => dtype != DType::C64 && dtype != DType::C128, // Complex numbers don't support ordering
            "bitwise" => dtype.is_int() || dtype == DType::Bool,
            "serialization" => true, // All standard types support serialization
            _ => false,
        }
    }

    /// Get type hierarchy information
    pub fn type_hierarchy(&self, dtype: DType) -> TypeHierarchy {
        TypeHierarchy::new(dtype)
    }
}

impl Default for TypeSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a type's position in the type hierarchy
#[derive(Debug, Clone)]
pub struct TypeHierarchy {
    pub dtype: DType,
    pub category: TypeCategory,
    pub precision: u8,
    pub can_promote_to: Vec<DType>,
    pub can_accept_from: Vec<DType>,
}

impl TypeHierarchy {
    fn new(dtype: DType) -> Self {
        let category = TypeCategory::from_dtype(dtype);
        let precision = promotion::utils::precision_level(dtype);

        // Find types this can promote to
        let all_types = [
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

        let can_promote_to: Vec<DType> = all_types
            .iter()
            .filter(|&&target| DType::can_promote_to(dtype, target))
            .copied()
            .collect();

        let can_accept_from: Vec<DType> = all_types
            .iter()
            .filter(|&&source| DType::can_promote_to(source, dtype))
            .copied()
            .collect();

        Self {
            dtype,
            category,
            precision,
            can_promote_to,
            can_accept_from,
        }
    }
}

/// Categories of data types
#[derive(Debug, Clone, PartialEq)]
pub enum TypeCategory {
    Boolean,
    SignedInteger,
    UnsignedInteger,
    FloatingPoint,
    Complex,
    Quantized,
    Custom,
}

impl TypeCategory {
    fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Bool => TypeCategory::Boolean,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 => TypeCategory::SignedInteger,
            DType::U8 | DType::U32 | DType::U64 => TypeCategory::UnsignedInteger,
            DType::F16 | DType::BF16 | DType::F32 | DType::F64 => TypeCategory::FloatingPoint,
            DType::C64 | DType::C128 => TypeCategory::Complex,
            DType::QInt8 | DType::QUInt8 => TypeCategory::Quantized,
        }
    }
}

/// Comprehensive type information for debugging and introspection
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub dtype: DType,
    pub name: String,
    pub size_bytes: usize,
    pub alignment: usize,
    pub category: TypeCategory,
    pub precision: u8,
    pub properties: TypeProperties,
    pub hierarchy: TypeHierarchy,
}

/// Properties of a data type
#[derive(Debug, Clone)]
pub struct TypeProperties {
    pub is_numeric: bool,
    pub is_signed: bool,
    pub is_integer: bool,
    pub is_floating_point: bool,
    pub is_complex: bool,
    pub is_quantized: bool,
    pub supports_infinity: bool,
    pub supports_nan: bool,
    pub supports_arithmetic: bool,
    pub supports_comparison: bool,
    pub supports_bitwise: bool,
}

impl TypeInfo {
    /// Get comprehensive type information
    pub fn for_dtype(dtype: DType) -> Self {
        let hierarchy = TypeHierarchy::new(dtype);
        let category = TypeCategory::from_dtype(dtype);

        let properties = TypeProperties {
            is_numeric: dtype.is_int() || dtype.is_float() || dtype.is_complex(),
            is_signed: dtype.is_signed(),
            is_integer: dtype.is_int(),
            is_floating_point: dtype.is_float(),
            is_complex: dtype.is_complex(),
            is_quantized: dtype.is_quantized(),
            supports_infinity: dtype.is_float(),
            supports_nan: dtype.is_float(),
            supports_arithmetic: dtype.is_int() || dtype.is_float() || dtype.is_complex(),
            supports_comparison: dtype != DType::C64 && dtype != DType::C128,
            supports_bitwise: dtype.is_int() || dtype == DType::Bool,
        };

        Self {
            dtype,
            name: dtype.name().to_string(),
            size_bytes: dtype.size(),
            alignment: dtype.size(), // Simplified - would be more sophisticated
            category,
            precision: promotion::utils::precision_level(dtype),
            properties,
            hierarchy,
        }
    }
}

/// Convenience functions for common type operations
pub mod utils {
    use super::*;

    /// Check if two types can be used together in operations
    pub fn are_compatible(type1: DType, type2: DType) -> bool {
        TypeSystem::global().are_compatible(type1, type2)
    }

    /// Find the best common type for a set of types
    pub fn find_common_type(types: &[DType]) -> Option<DType> {
        TypeSystem::global().result_type(types)
    }

    /// Get a human-readable description of a type
    pub fn type_description(dtype: DType) -> String {
        let info = TypeInfo::for_dtype(dtype);
        format!(
            "{} ({} bytes, {:?}, precision: {})",
            info.name, info.size_bytes, info.category, info.precision
        )
    }

    /// Check if a value can be safely represented in a target type
    pub fn can_represent_value<T: TensorElement>(value: T, target_type: DType) -> bool {
        // Convert to f64 and check if it can be represented in target type
        if let Some(f64_val) = value.to_f64() {
            // Check range for integer types
            match target_type {
                DType::I8 => f64_val >= i8::MIN as f64 && f64_val <= i8::MAX as f64,
                DType::U8 => f64_val >= 0.0 && f64_val <= u8::MAX as f64,
                DType::I16 => f64_val >= i16::MIN as f64 && f64_val <= i16::MAX as f64,
                DType::I32 => f64_val >= i32::MIN as f64 && f64_val <= i32::MAX as f64,
                DType::U32 => f64_val >= 0.0 && f64_val <= u32::MAX as f64,
                DType::I64 => f64_val >= i64::MIN as f64 && f64_val <= i64::MAX as f64,
                DType::U64 => f64_val >= 0.0 && f64_val <= u64::MAX as f64,
                DType::F16 => f64_val.is_finite() && f64_val.abs() <= f16::MAX.to_f64(),
                DType::BF16 => f64_val.is_finite() && f64_val.abs() <= bf16::MAX.to_f64(),
                DType::F32 => f64_val.is_finite() && f64_val.abs() <= f32::MAX as f64,
                DType::F64 => f64_val.is_finite(),
                DType::Bool => f64_val == 0.0 || f64_val == 1.0,
                _ => true, // For complex and quantized types, assume representable
            }
        } else {
            false
        }
    }

    /// Get the "safest" type for mixed operations
    pub fn safest_type(types: &[DType]) -> DType {
        // Priority: Complex > Float > Integer > Bool
        let has_complex = types.iter().any(|t| t.is_complex());
        if has_complex {
            return DType::C128; // Highest precision complex
        }

        let has_float = types.iter().any(|t| t.is_float());
        if has_float {
            return DType::F64; // Highest precision float
        }

        let has_large_int = types.iter().any(|t| matches!(t, DType::I64 | DType::U64));
        if has_large_int {
            return DType::I64; // Highest precision integer
        }

        DType::I32 // Default safe integer type
    }

    /// Convert a string representation to DType
    pub fn parse_dtype(s: &str) -> Result<DType, String> {
        s.parse()
    }

    /// Get all supported data types
    pub fn all_dtypes() -> Vec<DType> {
        vec![
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
        ]
    }

    /// Get memory alignment requirements for a type
    pub fn alignment_for_dtype(dtype: DType) -> usize {
        // Simplified alignment - in practice would be more sophisticated
        match dtype {
            DType::Bool | DType::I8 | DType::U8 | DType::QInt8 | DType::QUInt8 => 1,
            DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::I32 | DType::U32 | DType::F32 => 4,
            DType::I64 | DType::U64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use super::{
        BFloat16Ops, ComplexElement, CustomDTypeRegistry, CustomTensorElement, DType, FloatElement,
        QInt8, QUInt8, TensorElement, TypeCategory, TypeInfo, TypePromotion, TypeSystem,
    };

    pub use super::utils::{
        all_dtypes, are_compatible, can_represent_value, find_common_type, parse_dtype,
        safest_type, type_description,
    };
}

// Backward compatibility re-exports
// These ensure that existing code using the original dtype.rs continues to work
pub use bfloat16::{BF16RoundingMode, BFloat16Ops};
pub use complex::{Complex32, Complex64, ComplexElement};
pub use core::DType;
pub use custom::{CustomDTypeInfo, CustomDTypeRegistry, CustomTensorElement, ExtendedDType};
pub use promotion::{AutoPromote, PromotionMatrix, TypePromotion};
pub use quantized::{QInt8, QUInt8};
pub use traits::{FloatElement, TensorElement};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_system() {
        let type_system = TypeSystem::new();

        // Test promotion
        let promoted = type_system.promote_types(DType::I32, DType::F32);
        assert_eq!(promoted, DType::F32);

        // Test compatibility
        assert!(type_system.are_compatible(DType::I8, DType::I16));
        assert!(!type_system.are_compatible(DType::F32, DType::Bool));

        // Test result type
        let types = vec![DType::I8, DType::F32, DType::I16];
        let result = type_system.result_type(&types);
        assert_eq!(result, Some(DType::F32));
    }

    #[test]
    fn test_type_hierarchy() {
        let hierarchy = TypeHierarchy::new(DType::I32);

        assert_eq!(hierarchy.dtype, DType::I32);
        assert_eq!(hierarchy.category, TypeCategory::SignedInteger);
        assert!(hierarchy.can_promote_to.contains(&DType::I64));
        assert!(hierarchy.can_accept_from.contains(&DType::I8));
    }

    #[test]
    fn test_type_info() {
        let info = TypeInfo::for_dtype(DType::F32);

        assert_eq!(info.name, "f32");
        assert_eq!(info.size_bytes, 4);
        assert_eq!(info.category, TypeCategory::FloatingPoint);
        assert!(info.properties.is_numeric);
        assert!(info.properties.is_floating_point);
        assert!(info.properties.supports_arithmetic);
        assert!(!info.properties.is_integer);
    }

    #[test]
    fn test_utils_functions() {
        // Test compatibility
        assert!(utils::are_compatible(DType::I32, DType::I64));

        // Test common type finding
        let types = vec![DType::I8, DType::I16, DType::I32];
        assert_eq!(utils::find_common_type(&types), Some(DType::I32));

        // Test type description
        let desc = utils::type_description(DType::F64);
        assert!(desc.contains("f64"));
        assert!(desc.contains("8 bytes"));

        // Test value representation
        assert!(utils::can_represent_value(42i32, DType::I64));
        assert!(!utils::can_represent_value(300i32, DType::I8));

        // Test safest type
        let mixed_types = vec![DType::I32, DType::F32, DType::Bool];
        assert_eq!(utils::safest_type(&mixed_types), DType::F64);

        // Test dtype parsing
        assert_eq!(utils::parse_dtype("f32").unwrap(), DType::F32);
        assert!(utils::parse_dtype("invalid").is_err());

        // Test all dtypes
        let all = utils::all_dtypes();
        assert!(all.contains(&DType::F32));
        assert!(all.contains(&DType::I64));
        assert!(all.len() >= 16);

        // Test alignment
        assert_eq!(utils::alignment_for_dtype(DType::F64), 8);
        assert_eq!(utils::alignment_for_dtype(DType::I16), 2);
    }

    #[test]
    fn test_type_categories() {
        assert_eq!(
            TypeCategory::from_dtype(DType::I32),
            TypeCategory::SignedInteger
        );
        assert_eq!(
            TypeCategory::from_dtype(DType::F64),
            TypeCategory::FloatingPoint
        );
        assert_eq!(TypeCategory::from_dtype(DType::C64), TypeCategory::Complex);
        assert_eq!(TypeCategory::from_dtype(DType::Bool), TypeCategory::Boolean);
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that all original APIs still work
        let dtype = DType::F32;
        assert_eq!(dtype.size(), 4);
        assert!(dtype.is_float());
        assert!(!dtype.is_complex());
        assert_eq!(dtype.name(), "f32");

        // Test type promotion
        let promoted = DType::promote_types(DType::I32, DType::F32);
        assert_eq!(promoted, DType::F32);

        // Test tensor element
        assert_eq!(<f32 as TensorElement>::dtype(), DType::F32);
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
    }

    #[test]
    fn test_prelude_imports() {
        use super::prelude::*;

        // Test that all commonly used items are available
        let dtype = DType::F32;
        let _type_system = TypeSystem::new();
        let _info = TypeInfo::for_dtype(dtype);

        assert!(are_compatible(DType::I32, DType::I64));
        let _desc = type_description(DType::F64);
    }
}
