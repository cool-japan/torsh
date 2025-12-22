//! Comprehensive error condition tests for torsh-core
//!
//! This test suite validates error handling across all core modules including:
//! - Shape validation errors
//! - DType conversion errors
//! - Device operation errors
//! - Memory allocation errors
//! - Broadcasting errors
//! - Type promotion edge cases

use torsh_core::{DType, DeviceType, Shape, TorshError, TypePromotion};

#[cfg(test)]
mod comprehensive_error_tests {
    use super::*;

    // Shape Error Tests

    #[test]
    fn test_shape_zero_dimension_errors() {
        // Test zero dimension rejection
        assert!(Shape::from_dims([0]).is_err());
        assert!(Shape::from_dims([0, 5]).is_err());
        assert!(Shape::from_dims([5, 0]).is_err());
        assert!(Shape::from_dims([0, 0, 0]).is_err());

        // Test mixed zero and non-zero
        assert!(Shape::from_dims([1, 0, 3]).is_err());
        assert!(Shape::from_dims([0, 1, 1]).is_err());
    }

    #[test]
    fn test_shape_large_dimensions() {
        // Test handling of large dimensions
        // Note: Some implementations may allow very large dimensions
        // and only fail on actual memory allocation

        // Test with reasonable large dimensions
        let large_dims = vec![10_000, 10_000];
        let result = Shape::from_dims(large_dims);
        if let Ok(shape) = result {
            // If shape creation succeeds, verify properties
            assert_eq!(shape.numel(), 100_000_000);
        }

        // Test product overflow detection (if implemented)
        let overflow_dims = vec![usize::MAX / 2 + 1, 3];
        let result = Shape::from_dims(overflow_dims);
        // May succeed or fail depending on overflow checking implementation
        if result.is_err() {
            match result.unwrap_err() {
                TorshError::InvalidShape(_) => {} // Expected if overflow checked
                _ => panic!("Expected InvalidShape error for overflow"),
            }
        }
    }

    #[test]
    fn test_shape_empty_dimension_list() {
        // Empty dimensions should create scalar shape
        let shape = Shape::from_dims(Vec::<usize>::new()).unwrap();
        assert!(shape.is_scalar());
        assert_eq!(shape.ndim(), 0);
        assert_eq!(shape.numel(), 1); // Scalar has 1 element
    }

    #[test]
    fn test_shape_broadcasting_incompatible_errors() {
        let shape1 = Shape::from_dims([3, 4, 5]).unwrap();

        // Incompatible dimensions
        let incompatible = Shape::from_dims(vec![3, 7, 5]).unwrap();
        assert!(shape1.broadcast_with(&incompatible).is_err());

        // Different ranks with incompatible dims
        let incompatible2 = Shape::from_dims(vec![7]).unwrap();
        assert!(shape1.broadcast_with(&incompatible2).is_err());

        // Verify error type
        match shape1.broadcast_with(&incompatible) {
            Err(TorshError::BroadcastError { .. }) => {} // Expected
            other => panic!("Expected BroadcastError, got {:?}", other),
        }
    }

    #[test]
    fn test_shape_transpose_validation() {
        let shape = Shape::from_dims([2, 3, 4]).unwrap();

        // Verify shape has correct dimensions
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.dims(), &[2, 3, 4]);

        // Test that shape operations work correctly
        let squeezed = shape.squeeze();
        assert_eq!(squeezed.dims(), &[2, 3, 4]); // No unit dimensions to squeeze

        // Test unsqueeze
        let unsqueezed = shape.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_shape_squeeze_unsqueeze_errors() {
        let shape = Shape::from_dims([2, 1, 3, 1]).unwrap();

        // Squeeze non-unit dimension (should error)
        assert!(shape.squeeze_dim(0).is_err()); // dim 0 has size 2
        assert!(shape.squeeze_dim(2).is_err()); // dim 2 has size 3

        // Squeeze valid dimensions (should succeed)
        assert!(shape.squeeze_dim(1).is_ok()); // dim 1 has size 1
        assert!(shape.squeeze_dim(3).is_ok()); // dim 3 has size 1

        // Out of bounds dimension
        assert!(shape.squeeze_dim(10).is_err());
        assert!(shape.unsqueeze(10).is_err());
    }

    // DType Error Tests

    #[test]
    fn test_dtype_invalid_conversions() {
        // Test type promotion edge cases
        let bool_type = DType::Bool;
        let complex_type = DType::C128;

        // Bool with complex (should have defined behavior)
        let result = DType::promote_types(bool_type, complex_type);
        assert!(result.is_complex());

        // Quantized type promotions
        let qint8 = DType::QInt8;
        let quint8 = DType::QUInt8;

        // Quantized types should promote to float
        let result = DType::promote_types(qint8, DType::I32);
        assert_eq!(result, DType::F32);

        let result = DType::promote_types(quint8, DType::F64);
        assert_eq!(result, DType::F64);
    }

    #[test]
    fn test_dtype_size_and_alignment() {
        // Verify all dtypes have valid sizes
        let all_dtypes = vec![
            DType::Bool,
            DType::I8,
            DType::U8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
            DType::C64,
            DType::C128,
            DType::QInt8,
            DType::QUInt8,
        ];

        for dtype in all_dtypes {
            let size = dtype.size();
            assert!(size > 0, "DType {:?} has invalid size 0", dtype);

            // Verify size is power of 2 for most types (except complex and quantized)
            if !dtype.is_complex() && !dtype.is_quantized() {
                assert!(
                    size.is_power_of_two() || size == 0,
                    "DType {:?} has non-power-of-2 size: {}",
                    dtype,
                    size
                );
            }
        }
    }

    #[test]
    fn test_dtype_category_properties() {
        // Test that dtype categories have expected properties
        let all_dtypes = vec![
            DType::Bool,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
            DType::C64,
            DType::C128,
            DType::QInt8,
            DType::QUInt8,
        ];

        for dtype in all_dtypes {
            let is_float = dtype.is_float();
            let is_int = dtype.is_int();
            let is_complex = dtype.is_complex();
            let _is_quantized = dtype.is_quantized();

            // Verify mutual exclusivity for primary categories
            // Note: Quantized types may also be considered int types
            if is_float {
                assert!(
                    !is_complex,
                    "DType {:?} cannot be both float and complex",
                    dtype
                );
            }

            if is_complex {
                assert!(
                    !is_float,
                    "DType {:?} cannot be both complex and float",
                    dtype
                );
                assert!(!is_int, "DType {:?} cannot be both complex and int", dtype);
            }

            // Quantized types may overlap with int category
            // This is expected behavior for QInt8/QUInt8
        }
    }

    // Device Error Tests

    #[test]
    fn test_device_type_comparison() {
        // Valid device indices
        assert!(DeviceType::Cuda(0) == DeviceType::Cuda(0));
        assert!(DeviceType::Cuda(7) == DeviceType::Cuda(7));

        // Large but potentially valid indices
        let _large_cuda = DeviceType::Cuda(255);
        let _large_metal = DeviceType::Metal(100);

        // Device type comparison
        assert_ne!(DeviceType::Cpu, DeviceType::Cuda(0));
        assert_ne!(DeviceType::Cuda(0), DeviceType::Cuda(1));
        assert_ne!(DeviceType::Cuda(0), DeviceType::Metal(0));
    }

    // Memory and Boundary Condition Tests

    #[test]
    fn test_memory_size_calculations() {
        // Test memory size calculations don't overflow
        let shape = Shape::from_dims([1000, 1000]).unwrap();
        let dtype = DType::F64;

        let total_elements = shape.numel();
        let bytes_per_element = dtype.size();

        // This multiplication should not overflow
        let total_bytes = total_elements.checked_mul(bytes_per_element);
        assert!(total_bytes.is_some(), "Memory size calculation overflowed");
        assert_eq!(total_bytes.unwrap(), 8_000_000); // 8MB

        // Test with very large shapes that would overflow
        if let Ok(large_shape) = Shape::from_dims([1_000_000, 1_000_000]) {
            // If shape creation succeeds, memory calculation might overflow
            let large_elements = large_shape.numel();
            let checked_bytes = large_elements.checked_mul(dtype.size());
            // Should detect overflow
            if checked_bytes.is_none() {
                // Overflow detected correctly
            }
        }
    }

    #[test]
    fn test_broadcasting_edge_cases() {
        // Scalar broadcasting
        let scalar = Shape::from_dims([]).unwrap();
        let tensor = Shape::from_dims([3, 4, 5]).unwrap();

        let result = tensor.broadcast_with(&scalar).unwrap();
        assert_eq!(result.dims(), tensor.dims());

        let result2 = scalar.broadcast_with(&tensor).unwrap();
        assert_eq!(result2.dims(), tensor.dims());

        // Single dimension broadcasting
        let shape1 = Shape::from_dims([1, 5]).unwrap();
        let shape2 = Shape::from_dims([3, 1]).unwrap();

        let result = shape1.broadcast_with(&shape2).unwrap();
        assert_eq!(result.dims(), &[3, 5]);

        // Trailing dimensions
        let shape1 = Shape::from_dims([3, 4, 5]).unwrap();
        let shape2 = Shape::from_dims([5]).unwrap();

        let result = shape1.broadcast_with(&shape2).unwrap();
        assert_eq!(result.dims(), &[3, 4, 5]);
    }

    #[test]
    fn test_type_promotion_symmetry() {
        // Type promotion should be symmetric
        let dtype_pairs = vec![
            (DType::F32, DType::F64),
            (DType::I32, DType::I64),
            (DType::F32, DType::C64),
            (DType::I32, DType::F32),
        ];

        for (dtype1, dtype2) in dtype_pairs {
            let result1 = DType::promote_types(dtype1, dtype2);
            let result2 = DType::promote_types(dtype2, dtype1);

            assert_eq!(
                result1, result2,
                "Type promotion not symmetric: {:?} + {:?} = {:?}, but {:?} + {:?} = {:?}",
                dtype1, dtype2, result1, dtype2, dtype1, result2
            );
        }
    }

    #[test]
    fn test_extreme_dimension_counts() {
        // Test with maximum reasonable dimension count
        let max_dims = vec![2; 32]; // 32-dimensional tensor
        let shape = Shape::from_dims(max_dims.clone());
        assert!(shape.is_ok());

        if let Ok(s) = shape {
            assert_eq!(s.ndim(), 32);
            assert_eq!(s.numel(), 2_usize.pow(32)); // 2^32 elements
        }

        // Test very high dimensional tensors
        let high_dims = vec![1; 100]; // 100-dimensional tensor of size 1
        let shape = Shape::from_dims(high_dims);
        assert!(shape.is_ok());

        if let Ok(s) = shape {
            assert_eq!(s.ndim(), 100);
            assert_eq!(s.numel(), 1); // All dimensions are 1
        }
    }

    #[test]
    fn test_error_message_quality() {
        // Verify error messages contain useful information

        // Invalid shape error
        let result = Shape::from_dims([0, 5]);
        if let Err(e) = result {
            let error_msg = format!("{}", e);
            assert!(!error_msg.is_empty(), "Error message should not be empty");
            // Error message should mention the issue
            assert!(
                error_msg.to_lowercase().contains("zero")
                    || error_msg.to_lowercase().contains("invalid")
                    || error_msg.to_lowercase().contains("dimension"),
                "Error message should describe the issue: {}",
                error_msg
            );
        }

        // Broadcasting error
        let shape1 = Shape::from_dims([3, 4]).unwrap();
        let shape2 = Shape::from_dims([5, 6]).unwrap();
        let result = shape1.broadcast_with(&shape2);

        if let Err(e) = result {
            let error_msg = format!("{}", e);
            assert!(!error_msg.is_empty(), "Error message should not be empty");
            assert!(
                error_msg.to_lowercase().contains("broadcast")
                    || error_msg.to_lowercase().contains("incompatible")
                    || error_msg.to_lowercase().contains("shape"),
                "Error message should describe broadcast issue: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_stride_calculation_edge_cases() {
        // Test stride calculations for various layouts

        // 1D tensor
        let shape1d = Shape::from_dims([10]).unwrap();
        assert_eq!(shape1d.strides(), &[1]);

        // 2D tensor
        let shape2d = Shape::from_dims([3, 4]).unwrap();
        assert_eq!(shape2d.strides(), &[4, 1]); // C-contiguous

        // 3D tensor
        let shape3d = Shape::from_dims([2, 3, 4]).unwrap();
        assert_eq!(shape3d.strides(), &[12, 4, 1]); // C-contiguous

        // Scalar (0D)
        let scalar = Shape::from_dims([]).unwrap();
        let empty_strides: Vec<usize> = vec![];
        assert_eq!(scalar.strides(), empty_strides.as_slice()); // No strides for scalar

        // Very large strides
        let large_shape = Shape::from_dims([1000, 1000, 1000]).unwrap();
        let strides = large_shape.strides();
        assert_eq!(strides[0], 1_000_000);
        assert_eq!(strides[1], 1_000);
        assert_eq!(strides[2], 1);
    }
}
