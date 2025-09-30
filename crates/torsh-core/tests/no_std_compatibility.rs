//! No-std compatibility tests for torsh-core
//!
//! These tests verify that the core functionality works without the standard library
//! when the no_std feature is enabled.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};

use torsh_core::{
    device::CpuDevice,
    dtype::{ComplexElement, FloatElement, TensorElement},
    DType, Device, DeviceType, Shape, TorshError,
};

#[cfg(feature = "std")]
use std::vec::Vec;

/// Test basic shape operations in no_std environment
#[test]
fn test_no_std_shape_operations() {
    // Test shape creation
    let shape = Shape::new(vec![2, 3, 4]);
    assert_eq!(shape.ndim(), 3);
    assert_eq!(shape.numel(), 24);
    assert_eq!(shape.dims(), &[2, 3, 4]);

    // Test scalar shape
    let scalar = Shape::scalar();
    assert_eq!(scalar.ndim(), 0);
    assert_eq!(scalar.numel(), 1);

    // Test shape creation from dims
    let built_shape = Shape::from_dims(vec![5, 6]).expect("Shape from_dims should work in no_std");
    assert_eq!(built_shape.dims(), &[5, 6]);

    // Test shape operations
    assert!(!shape.is_scalar());
    assert!(scalar.is_scalar());
    assert!(!shape.is_empty());

    // Test strides calculation
    let strides = shape.strides();
    assert_eq!(strides.len(), 3);
    assert_eq!(strides, vec![12, 4, 1]);
}

/// Test data type operations in no_std environment
#[test]
fn test_no_std_dtype_operations() {
    // Test basic data types
    let f32_dtype = DType::F32;
    let f64_dtype = DType::F64;
    let i32_dtype = DType::I32;
    let i64_dtype = DType::I64;

    // Test type properties
    assert_eq!(f32_dtype.size_bytes(), 4);
    assert_eq!(f64_dtype.size_bytes(), 8);
    assert_eq!(i32_dtype.size_bytes(), 4);
    assert_eq!(i64_dtype.size_bytes(), 8);

    assert!(f32_dtype.is_float());
    assert!(f64_dtype.is_float());
    assert!(!i32_dtype.is_float());
    assert!(!i64_dtype.is_float());

    assert!(!f32_dtype.is_int());
    assert!(!f64_dtype.is_int());
    assert!(i32_dtype.is_int());
    assert!(i64_dtype.is_int());

    // Test complex types if available
    #[cfg(feature = "half")]
    {
        let c64_dtype = DType::C64;
        let c128_dtype = DType::C128;

        assert!(c64_dtype.is_complex());
        assert!(c128_dtype.is_complex());
        assert_eq!(c64_dtype.size_bytes(), 8);
        assert_eq!(c128_dtype.size_bytes(), 16);
    }
}

/// Test device operations in no_std environment
#[test]
fn test_no_std_device_operations() {
    // Test CPU device
    let cpu_device = CpuDevice::new();
    assert_eq!(cpu_device.device_type(), DeviceType::Cpu);

    // Test device equality
    let cpu_device2 = CpuDevice::new();
    assert_eq!(cpu_device.device_type(), cpu_device2.device_type());

    // Test device display (using format! which should work in no_std)
    let device_string = format!("{cpu_device:?}");
    assert!(device_string.contains("CpuDevice"));

    // Test device debug
    let device_debug = format!("{cpu_device:?}");
    assert!(device_debug.contains("Cpu"));
}

/// Test error handling in no_std environment
#[test]
fn test_no_std_error_handling() {
    // Test error creation
    let shape_error = TorshError::InvalidShape("test error".to_string());

    // Test error display
    let error_string = format!("{shape_error}");
    assert!(error_string.contains("test error"));

    // Test error debug
    let error_debug = format!("{shape_error:?}");
    assert!(error_debug.contains("InvalidShape"));

    // Test shape builder error
    let builder_result = Shape::from_dims(vec![0]);

    assert!(builder_result.is_err());

    if let Err(error) = builder_result {
        let error_string = format!("{error}");
        assert!(!error_string.is_empty());
    }
}

/// Test tensor element traits in no_std environment
#[test]
fn test_no_std_tensor_elements() {
    // Test that basic numeric types implement TensorElement
    fn test_tensor_element<T: TensorElement>() {
        let dtype = T::dtype();
        assert!(dtype.size_bytes() > 0);
    }

    test_tensor_element::<f32>();
    test_tensor_element::<f64>();
    test_tensor_element::<i32>();
    test_tensor_element::<i64>();
    test_tensor_element::<i16>();
    test_tensor_element::<i8>();
    test_tensor_element::<u8>();

    // Test float element trait
    fn test_float_element<T: FloatElement>() {
        let dtype = T::dtype();
        assert!(dtype.is_float());
    }

    test_float_element::<f32>();
    test_float_element::<f64>();

    // Test complex element trait if available
    #[cfg(feature = "half")]
    {
        use torsh_core::dtype::{Complex32, Complex64};

        fn test_complex_element<T: ComplexElement>() {
            let dtype = T::dtype();
            assert!(dtype.is_complex());
        }

        test_complex_element::<Complex32>();
        test_complex_element::<Complex64>();
    }
}

/// Test shape broadcasting in no_std environment
#[test]
fn test_no_std_broadcasting() {
    let shape1 = Shape::new(vec![1, 3, 1]);
    let shape2 = Shape::new(vec![2, 1, 4]);

    // Test broadcast compatibility
    assert!(shape1.broadcast_with(&shape2).is_ok());

    // Test broadcast shape calculation
    let broadcast_result = shape1.broadcast_with(&shape2);
    assert!(broadcast_result.is_ok());

    if let Ok(result_shape) = broadcast_result {
        assert_eq!(result_shape.dims(), &[2, 3, 4]);
    }

    // Test broadcast_with
    let shape_array = Shape::new(vec![2, 1, 4]);
    let broadcast_with_result = shape1.broadcast_with(&shape_array);
    assert!(broadcast_with_result.is_ok());

    if let Ok(result_shape) = broadcast_with_result {
        assert_eq!(result_shape.dims(), &[2, 3, 4]);
    }

    // Test incompatible broadcasting
    let shape3 = Shape::new(vec![3, 2]);
    let shape4 = Shape::new(vec![4, 5]);

    assert!(shape3.broadcast_with(&shape4).is_err());
    assert!(shape3.broadcast_with(&shape4).is_err());
}

/// Test memory operations that should work in no_std
#[test]
fn test_no_std_memory_operations() {
    // Test vector operations (using alloc)
    let mut dims = vec![1, 2, 3];
    dims.push(4);
    assert_eq!(dims.len(), 4);

    let shape = Shape::new(dims);
    assert_eq!(shape.ndim(), 4);

    // Test cloning
    let cloned_shape = shape.clone();
    assert_eq!(shape, cloned_shape);

    // Test strides allocation
    let strides = shape.strides();
    assert_eq!(strides.len(), 4);

    // Test shape operations that allocate
    let default_strides = shape.default_strides();
    assert_eq!(default_strides.len(), 4);
}

/// Test macro functionality in no_std environment
#[test]
fn test_no_std_macros() {
    // Test shape creation with new
    let shape = Shape::new(vec![2, 3, 4]);
    assert_eq!(shape.dims(), &[2, 3, 4]);

    // Test scalar shape
    let scalar = Shape::scalar();
    assert!(scalar.is_scalar());

    // Test stride calculation
    let strides = shape.default_strides();
    assert_eq!(strides, vec![12, 4, 1]);
}

/// Test core constants and version info in no_std
#[test]
fn test_no_std_constants() {
    // Test version constants are accessible and have reasonable values
    // Version constants should be defined (implicit by compilation success)
    let _major = torsh_core::VERSION_MAJOR;
    let _minor = torsh_core::VERSION_MINOR;
    let _patch = torsh_core::VERSION_PATCH;

    // Test version string
    let version = torsh_core::VERSION;
    assert!(!version.is_empty());
}

/// Compilation test for no_std feature combinations
#[cfg(test)]
mod feature_combinations {
    use super::*;

    #[test]
    fn test_minimal_no_std() {
        // Test that core functionality works with minimal features
        let shape = Shape::new(vec![2, 3]);
        assert_eq!(shape.numel(), 6);

        let device = CpuDevice::new();
        assert_eq!(device.device_type(), DeviceType::Cpu);

        let dtype = DType::F32;
        assert_eq!(dtype.size_bytes(), 4);
    }

    #[cfg(feature = "half")]
    #[test]
    fn test_no_std_with_half_precision() {
        // Test half-precision types in no_std
        let f16_dtype = DType::F16;
        let bf16_dtype = DType::BF16;

        assert!(f16_dtype.is_float());
        assert!(bf16_dtype.is_float());
        assert_eq!(f16_dtype.size_bytes(), 2);
        assert_eq!(bf16_dtype.size_bytes(), 2);
    }

    #[test]
    fn test_no_std_error_chain() {
        // Test that error handling works properly in no_std
        let result: Result<Shape, TorshError> = Shape::from_dims(vec![0]);

        match result {
            Ok(_) => panic!("Expected error"),
            Err(error) => {
                // Error should be displayable in no_std
                let error_string = format!("{error}");
                assert!(!error_string.is_empty(), "Error string should not be empty");
            }
        }
    }
}

/// Performance tests that should work in no_std (without timing)
#[cfg(test)]
mod no_std_performance {
    use super::*;

    #[test]
    fn test_no_std_shape_creation_performance() {
        // Test that we can create many shapes without issues
        let mut shapes = Vec::new();

        for i in 1..=100 {
            let shape = Shape::new(vec![i, i + 1, i + 2]);
            shapes.push(shape);
        }

        assert_eq!(shapes.len(), 100);

        // Test that all shapes are valid
        for (i, shape) in shapes.iter().enumerate() {
            let expected_dims = [i + 1, i + 2, i + 3];
            assert_eq!(shape.dims(), &expected_dims);
        }
    }

    #[test]
    fn test_no_std_broadcasting_performance() {
        // Test broadcasting operations in bulk
        let base_shape = Shape::new(vec![1, 1, 1]);

        for i in 1..=50 {
            let other_shape = Shape::new(vec![i, i + 1, i + 2]);
            assert!(base_shape.broadcast_with(&other_shape).is_ok());

            let result = base_shape.broadcast_with(&other_shape).unwrap();
            assert_eq!(result.dims(), &[i, i + 1, i + 2]);
        }
    }
}
