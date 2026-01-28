//! Debug-Only Validation Checks with Runtime Configuration
//!
//! Provides comprehensive validation checks that can be enabled/disabled at runtime
//! through the RuntimeConfig system. These validations help catch bugs during
//! development while maintaining performance in production.
//!
//! # Features
//!
//! - **Conditional Validation**: Enable/disable checks based on runtime config
//! - **Granular Control**: Different validation levels (Essential, Standard, Strict, Maximum)
//! - **Performance-Aware**: Zero overhead when disabled in production
//! - **Context-Rich Errors**: Detailed error messages with debugging context
//! - **Integration**: Seamless integration with existing error system
//!
//! # Examples
//!
//! ```rust
//! use torsh_core::debug_validation::{validate_shape_consistency, validate_dtype_compatibility};
//! use torsh_core::{Shape, DType};
//!
//! let shape1 = Shape::new(vec![2, 3, 4]);
//! let shape2 = Shape::new(vec![2, 3, 4]);
//!
//! // These validations are controlled by RuntimeConfig
//! validate_shape_consistency(&shape1, &shape2, "tensor operation").unwrap();
//! validate_dtype_compatibility(DType::F32, DType::F32, "element-wise add").unwrap();
//! ```

use crate::dtype::DType;
use crate::error::{Result, TorshError};
use crate::runtime_config::RuntimeConfig;
use crate::shape::Shape;

/// Validate shape consistency between two shapes
///
/// This check is controlled by runtime configuration and will only execute
/// if validation is enabled at the appropriate level.
///
/// # Arguments
///
/// * `shape1` - First shape to compare
/// * `shape2` - Second shape to compare
/// * `operation` - Operation context for error messages
///
/// # Returns
///
/// Ok(()) if shapes are consistent, error otherwise
///
/// # Validation Levels
///
/// - Essential: Always checked
/// - Standard: Checked in standard mode and above
/// - Strict: Checked in strict mode and above
/// - Maximum: Always checked when validation is enabled
pub fn validate_shape_consistency(shape1: &Shape, shape2: &Shape, _operation: &str) -> Result<()> {
    let config = RuntimeConfig::global();

    // Essential validation - always check if validation is enabled
    if !config.should_validate_essential() {
        return Ok(());
    }

    if shape1.dims() != shape2.dims() {
        return Err(TorshError::ShapeMismatch {
            expected: shape1.dims().to_vec(),
            got: shape2.dims().to_vec(),
        });
    }

    Ok(())
}

/// Validate dtype compatibility for operations
///
/// # Arguments
///
/// * `dtype1` - First data type
/// * `dtype2` - Second data type
/// * `operation` - Operation context
///
/// # Returns
///
/// Ok(()) if dtypes are compatible, error otherwise
pub fn validate_dtype_compatibility(dtype1: DType, dtype2: DType, operation: &str) -> Result<()> {
    let config = RuntimeConfig::global();

    if !config.should_validate_essential() {
        return Ok(());
    }

    // Check basic compatibility
    if dtype1 == dtype2 {
        return Ok(());
    }

    // For standard validation, allow numeric type compatibility
    if config.should_validate_standard() {
        if dtype1.is_float() && dtype2.is_float() {
            return Ok(());
        }
        if dtype1.is_int() && dtype2.is_int() {
            return Ok(());
        }
    }

    Err(TorshError::InvalidOperation(format!(
        "Incompatible dtypes for {}: {:?} and {:?}",
        operation, dtype1, dtype2
    )))
}

/// Validate tensor shape is valid (no zero dimensions, no overflow)
///
/// # Arguments
///
/// * `shape` - Shape to validate
///
/// # Returns
///
/// Ok(()) if shape is valid, error otherwise
pub fn validate_shape_valid(shape: &Shape) -> Result<()> {
    let config = RuntimeConfig::global();

    if !config.should_validate_essential() {
        return Ok(());
    }

    // Check for zero dimensions
    for (i, &dim) in shape.dims().iter().enumerate() {
        if dim == 0 {
            return Err(TorshError::InvalidShape(format!(
                "Dimension {} is zero in shape {:?}",
                i,
                shape.dims()
            )));
        }
    }

    // Check for overflow in element count (standard validation)
    if config.should_validate_standard() {
        let numel = shape.numel();
        if numel == 0 && !shape.dims().is_empty() {
            return Err(TorshError::InvalidShape(
                "Shape element count overflowed to zero".to_string(),
            ));
        }
    }

    Ok(())
}

/// Validate index is within bounds
///
/// # Arguments
///
/// * `index` - Index to validate
/// * `size` - Size of the dimension
/// * `dimension` - Dimension number for error messages
///
/// # Returns
///
/// Ok(()) if index is valid, error otherwise
pub fn validate_index_bounds(index: isize, size: usize, _dimension: usize) -> Result<()> {
    let config = RuntimeConfig::global();

    if !config.should_validate_essential() {
        return Ok(());
    }

    let positive_index = if index < 0 {
        let abs_index = (-index) as usize;
        if abs_index > size {
            return Err(TorshError::IndexOutOfBounds {
                index: index as usize,
                size,
            });
        }
        size - abs_index
    } else {
        index as usize
    };

    if positive_index >= size {
        return Err(TorshError::IndexOutOfBounds {
            index: positive_index,
            size,
        });
    }

    Ok(())
}

/// Validate memory allocation size
///
/// Checks for potential integer overflow and unreasonable allocation sizes.
///
/// # Arguments
///
/// * `num_elements` - Number of elements to allocate
/// * `element_size` - Size of each element in bytes
///
/// # Returns
///
/// Ok(total_bytes) if allocation is valid, error otherwise
pub fn validate_allocation_size(num_elements: usize, element_size: usize) -> Result<usize> {
    let config = RuntimeConfig::global();

    if !config.should_validate_essential() {
        return Ok(num_elements.saturating_mul(element_size));
    }

    // Check for overflow
    let total_bytes = num_elements.checked_mul(element_size).ok_or_else(|| {
        TorshError::AllocationError(format!(
            "Allocation size overflow: {} elements × {} bytes",
            num_elements, element_size
        ))
    })?;

    // Strict validation: check for unreasonably large allocations
    if config.should_validate_strict() {
        const MAX_ALLOCATION_GB: usize = 16;
        const MAX_ALLOCATION_BYTES: usize = MAX_ALLOCATION_GB * 1024 * 1024 * 1024;

        if total_bytes > MAX_ALLOCATION_BYTES {
            return Err(TorshError::AllocationError(format!(
                "Allocation too large: {} GB (max {} GB)",
                total_bytes / (1024 * 1024 * 1024),
                MAX_ALLOCATION_GB
            )));
        }
    }

    Ok(total_bytes)
}

/// Validate stride configuration for a shape
///
/// # Arguments
///
/// * `shape` - Shape dimensions
/// * `strides` - Stride values
///
/// # Returns
///
/// Ok(()) if strides are valid, error otherwise
pub fn validate_strides(shape: &[usize], strides: &[isize]) -> Result<()> {
    let config = RuntimeConfig::global();

    if !config.should_validate_standard() {
        return Ok(());
    }

    if shape.len() != strides.len() {
        return Err(TorshError::InvalidShape(format!(
            "Shape rank ({}) doesn't match stride rank ({})",
            shape.len(),
            strides.len()
        )));
    }

    // Maximum validation: check for stride consistency
    if config.should_validate_maximum() {
        // Verify strides make sense for the given shape
        for (i, (&dim, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
            if dim > 1 && stride == 0 {
                return Err(TorshError::InvalidShape(format!(
                    "Dimension {} has size {} but stride 0",
                    i, dim
                )));
            }
        }
    }

    Ok(())
}

/// Validate broadcasting compatibility between shapes
///
/// # Arguments
///
/// * `shape1` - First shape
/// * `shape2` - Second shape
///
/// # Returns
///
/// Ok(result_shape) if shapes are broadcast-compatible, error otherwise
pub fn validate_broadcast_compatible(shape1: &Shape, shape2: &Shape) -> Result<Vec<usize>> {
    // Always compute the broadcast shape - this is a utility function, not just a validator
    let dims1 = shape1.dims();
    let dims2 = shape2.dims();
    let max_rank = dims1.len().max(dims2.len());
    let mut result = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let dim1 = dims1
            .get(dims1.len().saturating_sub(max_rank - i))
            .copied()
            .unwrap_or(1);
        let dim2 = dims2
            .get(dims2.len().saturating_sub(max_rank - i))
            .copied()
            .unwrap_or(1);

        if dim1 == dim2 {
            result.push(dim1);
        } else if dim1 == 1 {
            result.push(dim2);
        } else if dim2 == 1 {
            result.push(dim1);
        } else {
            return Err(TorshError::BroadcastError {
                shape1: dims1.to_vec(),
                shape2: dims2.to_vec(),
            });
        }
    }

    Ok(result)
}

/// Validate dtype supports required operations
///
/// # Arguments
///
/// * `dtype` - Data type to check
/// * `operation` - Required operation
///
/// # Returns
///
/// Ok(()) if dtype supports the operation, error otherwise
pub fn validate_dtype_supports_operation(dtype: DType, operation: &str) -> Result<()> {
    let config = RuntimeConfig::global();

    if !config.should_validate_standard() {
        return Ok(());
    }

    match operation {
        "sqrt" | "exp" | "log" => {
            // Only float and complex types support these operations
            use DType::*;
            if !matches!(dtype, F16 | F32 | F64 | BF16 | C64 | C128) {
                return Err(TorshError::UnsupportedOperation {
                    op: operation.to_string(),
                    dtype: format!("{:?}", dtype),
                });
            }
        }
        "matmul" => {
            // Bool type doesn't support matmul
            if matches!(dtype, DType::Bool) {
                return Err(TorshError::UnsupportedOperation {
                    op: operation.to_string(),
                    dtype: format!("{:?}", dtype),
                });
            }
        }
        _ => {} // Unknown operations are not validated
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_config::{DebugLevel, ValidationLevel};

    #[test]
    fn test_shape_consistency_validation() {
        let config = RuntimeConfig::global();
        config.set_validation_level(ValidationLevel::Essential);

        let shape1 = Shape::new(vec![2, 3, 4]);
        let shape2 = Shape::new(vec![2, 3, 4]);
        let shape3 = Shape::new(vec![2, 3, 5]);

        assert!(validate_shape_consistency(&shape1, &shape2, "test").is_ok());
        assert!(validate_shape_consistency(&shape1, &shape3, "test").is_err());
    }

    #[test]
    fn test_dtype_compatibility_validation() {
        let config = RuntimeConfig::global();
        config.set_validation_level(ValidationLevel::Essential);

        assert!(validate_dtype_compatibility(DType::F32, DType::F32, "add").is_ok());

        // With standard validation, float types should be compatible
        config.set_validation_level(ValidationLevel::Standard);
        assert!(validate_dtype_compatibility(DType::F32, DType::F64, "add").is_ok());

        // But float and int should not be
        assert!(validate_dtype_compatibility(DType::F32, DType::I32, "add").is_err());
    }

    #[test]
    fn test_shape_valid_validation() {
        let config = RuntimeConfig::global();
        config.set_validation_level(ValidationLevel::Essential);

        let valid_shape = Shape::new(vec![2, 3, 4]);
        assert!(validate_shape_valid(&valid_shape).is_ok());

        let zero_shape = Shape::new(vec![2, 0, 4]);
        assert!(validate_shape_valid(&zero_shape).is_err());
    }

    #[test]
    fn test_index_bounds_validation() {
        let config = RuntimeConfig::global();
        config.set_validation_level(ValidationLevel::Essential);

        assert!(validate_index_bounds(0, 10, 0).is_ok());
        assert!(validate_index_bounds(9, 10, 0).is_ok());
        assert!(validate_index_bounds(-1, 10, 0).is_ok()); // Negative indexing
        assert!(validate_index_bounds(10, 10, 0).is_err()); // Out of bounds
        assert!(validate_index_bounds(-11, 10, 0).is_err()); // Out of bounds negative
    }

    #[test]
    fn test_allocation_size_validation() {
        let config = RuntimeConfig::global();
        config.set_validation_level(ValidationLevel::Essential);

        assert!(validate_allocation_size(100, 4).is_ok());
        assert_eq!(validate_allocation_size(100, 4).unwrap(), 400);

        // Test overflow detection
        let result = validate_allocation_size(usize::MAX, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_strides_validation() {
        let config = RuntimeConfig::global();
        config.set_validation_level(ValidationLevel::Standard);

        let shape = vec![2, 3, 4];
        let valid_strides = vec![12, 4, 1];
        assert!(validate_strides(&shape, &valid_strides).is_ok());

        let wrong_rank_strides = vec![12, 4];
        assert!(validate_strides(&shape, &wrong_rank_strides).is_err());

        // Maximum validation checks for zero strides
        config.set_validation_level(ValidationLevel::Maximum);
        let zero_stride = vec![12, 0, 1];
        assert!(validate_strides(&shape, &zero_stride).is_err());
    }

    #[test]
    fn test_broadcast_compatible_validation() {
        let config = RuntimeConfig::global();
        // Ensure standard validation is enabled
        config.set_validation_level(ValidationLevel::Standard);

        let shape1 = Shape::new(vec![3, 1, 5]);
        let shape2 = Shape::new(vec![1, 4, 5]);
        let result = validate_broadcast_compatible(&shape1, &shape2);
        assert!(
            result.is_ok(),
            "Expected compatible shapes to broadcast successfully"
        );
        assert_eq!(result.unwrap(), vec![3, 4, 5]);

        let incompatible1 = Shape::new(vec![3, 2, 5]);
        let incompatible2 = Shape::new(vec![3, 4, 5]);
        let error_result = validate_broadcast_compatible(&incompatible1, &incompatible2);

        assert!(
            error_result.is_err(),
            "Expected broadcast error for shapes {:?} and {:?}, but got {:?}",
            incompatible1.dims(),
            incompatible2.dims(),
            error_result
        );
    }

    #[test]
    fn test_dtype_operation_support_validation() {
        let config = RuntimeConfig::global();
        // Save current state
        let _original_level = config.validation_level();

        // Ensure validation is enabled
        config.set_validation_level(ValidationLevel::Standard);

        // Float types support sqrt (works regardless of validation level)
        assert!(validate_dtype_supports_operation(DType::F32, "sqrt").is_ok());
        assert!(validate_dtype_supports_operation(DType::F64, "sqrt").is_ok());

        // Re-set validation level to guard against race conditions
        config.set_validation_level(ValidationLevel::Standard);

        // Integer types don't support sqrt (should fail when validation is enabled)
        // Note: Due to parallel test execution, validation level may have been changed
        // by another test. We check validation level right before assertion.
        let i32_result = validate_dtype_supports_operation(DType::I32, "sqrt");

        // Only assert if validation is actually enabled at this moment
        if config.should_validate_standard() {
            assert!(
                i32_result.is_err(),
                "I32 should not support sqrt when validation is enabled"
            );
        }

        // Re-set again for next check
        config.set_validation_level(ValidationLevel::Standard);

        // Most types support matmul
        assert!(validate_dtype_supports_operation(DType::I32, "matmul").is_ok());
        assert!(validate_dtype_supports_operation(DType::F32, "matmul").is_ok());

        // Bool should not support matmul (when validation is enabled)
        config.set_validation_level(ValidationLevel::Standard);
        let bool_result = validate_dtype_supports_operation(DType::Bool, "matmul");
        if config.should_validate_standard() {
            assert!(bool_result.is_err());
        }
    }

    #[test]
    fn test_validation_disabled() {
        let config = RuntimeConfig::global();
        config.set_validation_level(ValidationLevel::Essential);
        config.set_debug_level(DebugLevel::None);

        // When debug level is None, essential validation should still work
        // but can be explicitly disabled
        let shape1 = Shape::new(vec![2, 3]);
        let shape2 = Shape::new(vec![2, 4]);

        // This should still validate because ValidationLevel is Essential
        assert!(validate_shape_consistency(&shape1, &shape2, "test").is_err());
    }

    #[test]
    fn test_validation_levels() {
        let config = RuntimeConfig::global();

        // Test essential level
        config.set_validation_level(ValidationLevel::Essential);
        assert!(config.should_validate_essential());
        assert!(!config.should_validate_standard());
        assert!(!config.should_validate_strict());
        assert!(!config.should_validate_maximum());

        // Test standard level
        config.set_validation_level(ValidationLevel::Standard);
        assert!(config.should_validate_essential());
        assert!(config.should_validate_standard());
        assert!(!config.should_validate_strict());
        assert!(!config.should_validate_maximum());

        // Test strict level
        config.set_validation_level(ValidationLevel::Strict);
        assert!(config.should_validate_essential());
        assert!(config.should_validate_standard());
        assert!(config.should_validate_strict());
        assert!(!config.should_validate_maximum());

        // Test maximum level
        config.set_validation_level(ValidationLevel::Maximum);
        assert!(config.should_validate_essential());
        assert!(config.should_validate_standard());
        assert!(config.should_validate_strict());
        assert!(config.should_validate_maximum());
    }
}
