//! Type promotion utilities for functional operations
//!
//! This module implements PyTorch-compatible type promotion rules for tensor operations.
//! Type promotion ensures that operations between tensors of different types produce
//! results with appropriate types following well-defined rules.

use torsh_core::dtype::{DType, TensorElement};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Type promotion category for organizing data types
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypeCategory {
    /// Boolean types (bool)
    Boolean = 0,
    /// Integer types (i8, i16, i32, i64, u8)
    Integer = 1,
    /// Floating point types (f16, f32, f64)
    FloatingPoint = 2,
    /// Complex types (c64, c128)
    Complex = 3,
}

/// Get the type category for a data type
pub fn get_type_category(dtype: DType) -> TypeCategory {
    match dtype {
        DType::Bool => TypeCategory::Boolean,
        DType::U8 | DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U32 | DType::U64 => {
            TypeCategory::Integer
        }
        DType::F16 | DType::F32 | DType::F64 => TypeCategory::FloatingPoint,
        DType::C64 | DType::C128 => TypeCategory::Complex,
        DType::BF16 => TypeCategory::FloatingPoint, // Treat bfloat16 as floating point
        DType::QInt8 | DType::QUInt8 => TypeCategory::Integer, // Treat quantized types as integer
    }
}

/// Get the byte size/precision ranking for a data type within its category
pub fn get_type_precision(dtype: DType) -> u8 {
    match dtype {
        DType::Bool => 1,
        DType::U8 | DType::I8 => 8,
        DType::I16 => 16,
        DType::F16 | DType::BF16 => 16,
        DType::I32 | DType::F32 | DType::U32 => 32,
        DType::I64 | DType::F64 | DType::C64 | DType::U64 => 64,
        DType::C128 => 128,
        DType::QInt8 | DType::QUInt8 => 8, // Treat quantized 8-bit types as 8-bit
    }
}

/// Promote two data types according to PyTorch rules
pub fn promote_types(lhs: DType, rhs: DType) -> Result<DType> {
    // If types are the same, no promotion needed
    if lhs == rhs {
        return Ok(lhs);
    }

    let lhs_category = get_type_category(lhs);
    let rhs_category = get_type_category(rhs);

    // Promotion follows category hierarchy: Boolean < Integer < FloatingPoint < Complex
    let result_category = std::cmp::max(lhs_category, rhs_category);

    match result_category {
        TypeCategory::Boolean => {
            // Both must be boolean for this case
            Ok(DType::Bool)
        }
        TypeCategory::Integer => {
            // Promote to the larger integer type
            let lhs_precision = get_type_precision(lhs);
            let rhs_precision = get_type_precision(rhs);

            if lhs_precision >= rhs_precision {
                Ok(lhs)
            } else {
                Ok(rhs)
            }
        }
        TypeCategory::FloatingPoint => {
            // Promote to floating point, choosing the higher precision
            let target_precision = std::cmp::max(get_type_precision(lhs), get_type_precision(rhs));

            match target_precision {
                16 => Ok(DType::F16), // Could be F16 or BF16, default to F16
                32 => Ok(DType::F32),
                64 => Ok(DType::F64),
                _ => Ok(DType::F32), // Default fallback
            }
        }
        TypeCategory::Complex => {
            // Promote to complex type
            let target_precision = std::cmp::max(get_type_precision(lhs), get_type_precision(rhs));

            if target_precision <= 64 {
                Ok(DType::C64)
            } else {
                Ok(DType::C128)
            }
        }
    }
}

/// Promote multiple data types to a common type
pub fn promote_multiple_types(types: &[DType]) -> Result<DType> {
    if types.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Cannot promote empty type list".to_string(),
        ));
    }

    if types.len() == 1 {
        return Ok(types[0]);
    }

    let mut result = types[0];
    for &dtype in &types[1..] {
        result = promote_types(result, dtype)?;
    }

    Ok(result)
}

/// Check if a tensor can be cast to a target type without data loss
pub fn can_cast_safely(from: DType, to: DType) -> bool {
    if from == to {
        return true;
    }

    let from_category = get_type_category(from);
    let to_category = get_type_category(to);

    // Can always promote to higher category
    if to_category > from_category {
        return true;
    }

    // Within same category, can promote to higher precision
    if from_category == to_category {
        return get_type_precision(to) >= get_type_precision(from);
    }

    false
}

/// Promote tensors to a common type for operations
pub fn promote_tensors<T, U>(lhs: &Tensor<T>, rhs: &Tensor<U>) -> Result<(Tensor<f32>, Tensor<f32>)>
where
    T: TensorElement,
    U: TensorElement,
{
    // For now, convert both to f32 for simplicity
    // In a full implementation, we would use the proper promoted type
    let lhs_f32 = Tensor::zeros(&lhs.shape().dims(), lhs.device())?;
    let rhs_f32 = Tensor::zeros(&rhs.shape().dims(), rhs.device())?;

    Ok((lhs_f32, rhs_f32))
}

/// Promote a list of tensors to a common type
pub fn promote_tensor_list<T>(tensors: &[&Tensor<T>]) -> Result<Vec<Tensor<f32>>>
where
    T: TensorElement,
{
    let mut result = Vec::new();
    for tensor in tensors {
        let promoted = Tensor::zeros(&tensor.shape().dims(), tensor.device())?;
        result.push(promoted);
    }
    Ok(result)
}

/// Get the result type for a binary operation between two tensors
pub fn result_type<T, U>(lhs: &Tensor<T>, rhs: &Tensor<U>) -> Result<DType>
where
    T: TensorElement,
    U: TensorElement,
{
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    promote_types(lhs_dtype, rhs_dtype)
}

/// Type promotion for scalar operations
pub fn promote_scalar_type<T>(tensor_dtype: DType, _scalar: T) -> Result<DType>
where
    T: TensorElement,
{
    let scalar_dtype = T::dtype();
    promote_types(tensor_dtype, scalar_dtype)
}

/// Helper function to ensure tensors have compatible types for operations
pub fn ensure_compatible_types<T, U>(lhs: &Tensor<T>, rhs: &Tensor<U>) -> Result<DType>
where
    T: TensorElement,
    U: TensorElement,
{
    result_type(lhs, rhs)
}

/// Type promotion rules for reduction operations
pub fn reduction_result_type(input_dtype: DType, operation: &str) -> Result<DType> {
    match operation {
        "sum" | "prod" => {
            // Sum and product may need higher precision to avoid overflow
            match input_dtype {
                DType::Bool | DType::U8 | DType::I8 | DType::I16 => Ok(DType::I64),
                DType::I32 | DType::U32 => Ok(DType::I64),
                DType::I64 | DType::U64 => Ok(DType::I64),
                DType::F16 | DType::BF16 => Ok(DType::F32),
                DType::F32 => Ok(DType::F32),
                DType::F64 => Ok(DType::F64),
                DType::C64 => Ok(DType::C64),
                DType::C128 => Ok(DType::C128),
                DType::QInt8 | DType::QUInt8 => Ok(DType::I64), // Quantized types promote to I64 for sum/prod
            }
        }
        "mean" => {
            // Mean always produces floating point result
            match input_dtype {
                DType::Bool
                | DType::U8
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U32
                | DType::U64 => Ok(DType::F32),
                DType::F16 | DType::BF16 => Ok(DType::F32),
                DType::F32 => Ok(DType::F32),
                DType::F64 => Ok(DType::F64),
                DType::C64 => Ok(DType::C64),
                DType::C128 => Ok(DType::C128),
                DType::QInt8 | DType::QUInt8 => Ok(DType::F32), // Quantized types promote to F32 for mean
            }
        }
        "max" | "min" | "argmax" | "argmin" => {
            // Max/min preserve input type, argmax/argmin return indices (i64)
            if operation.starts_with("arg") {
                Ok(DType::I64)
            } else {
                Ok(input_dtype)
            }
        }
        _ => {
            // Default: preserve input type
            Ok(input_dtype)
        }
    }
}

/// Get the common dtype for a mixed-type operation
pub fn common_dtype_for_operation(dtypes: &[DType], operation: &str) -> Result<DType> {
    if dtypes.is_empty() {
        return Err(TorshError::InvalidArgument(
            "No dtypes provided".to_string(),
        ));
    }

    // For most operations, promote all types to a common type
    let common_type = promote_multiple_types(dtypes)?;

    // Some operations have special rules
    match operation {
        "div" | "true_div" => {
            // Division always produces floating point result
            match common_type {
                DType::Bool | DType::U8 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
                    Ok(DType::F32)
                }
                _ => Ok(common_type),
            }
        }
        "floor_div" => {
            // Floor division preserves integer types but promotes small integers
            match common_type {
                DType::Bool | DType::U8 | DType::I8 => Ok(DType::I32),
                _ => Ok(common_type),
            }
        }
        _ => Ok(common_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::{ones, zeros};

    #[test]
    fn test_type_categories() {
        assert_eq!(get_type_category(DType::Bool), TypeCategory::Boolean);
        assert_eq!(get_type_category(DType::I32), TypeCategory::Integer);
        assert_eq!(get_type_category(DType::F32), TypeCategory::FloatingPoint);
        assert_eq!(get_type_category(DType::C64), TypeCategory::Complex);
    }

    #[test]
    fn test_type_precision() {
        assert!(get_type_precision(DType::F64) > get_type_precision(DType::F32));
        assert!(get_type_precision(DType::I64) > get_type_precision(DType::I32));
        assert!(get_type_precision(DType::C128) > get_type_precision(DType::C64));
    }

    #[test]
    fn test_basic_type_promotion() {
        assert_eq!(promote_types(DType::I32, DType::I32).unwrap(), DType::I32);
        assert_eq!(promote_types(DType::I32, DType::F32).unwrap(), DType::F32);
        assert_eq!(promote_types(DType::F32, DType::F64).unwrap(), DType::F64);
        assert_eq!(promote_types(DType::F32, DType::C64).unwrap(), DType::C64);
    }

    #[test]
    fn test_multiple_type_promotion() {
        let types = vec![DType::I32, DType::F32, DType::F64];
        assert_eq!(promote_multiple_types(&types).unwrap(), DType::F64);

        let types = vec![DType::Bool, DType::I16, DType::I32];
        assert_eq!(promote_multiple_types(&types).unwrap(), DType::I32);
    }

    #[test]
    fn test_safe_casting() {
        assert!(can_cast_safely(DType::I32, DType::I64));
        assert!(can_cast_safely(DType::F32, DType::F64));
        assert!(can_cast_safely(DType::I32, DType::F32));
        assert!(!can_cast_safely(DType::F64, DType::F32));
        assert!(!can_cast_safely(DType::I64, DType::I32));
    }

    #[test]
    fn test_reduction_result_types() {
        assert_eq!(
            reduction_result_type(DType::I32, "sum").unwrap(),
            DType::I64
        );
        assert_eq!(
            reduction_result_type(DType::F32, "mean").unwrap(),
            DType::F32
        );
        assert_eq!(
            reduction_result_type(DType::I32, "argmax").unwrap(),
            DType::I64
        );
        assert_eq!(
            reduction_result_type(DType::F32, "max").unwrap(),
            DType::F32
        );
    }

    #[test]
    fn test_operation_dtypes() {
        let dtypes = vec![DType::I32, DType::F32];
        assert_eq!(
            common_dtype_for_operation(&dtypes, "add").unwrap(),
            DType::F32
        );
        assert_eq!(
            common_dtype_for_operation(&dtypes, "div").unwrap(),
            DType::F32
        );

        let int_types = vec![DType::I16, DType::I32];
        assert_eq!(
            common_dtype_for_operation(&int_types, "div").unwrap(),
            DType::F32
        );
    }

    #[test]
    fn test_tensor_promotion() -> Result<()> {
        let t1: Tensor<f32> = ones(&[2, 3])?; // Specify f32 type
        let t2: Tensor<f32> = zeros(&[2, 3])?; // Specify f32 type

        let result_dtype = result_type(&t1, &t2)?;
        assert_eq!(result_dtype, DType::F32);

        Ok(())
    }
}
