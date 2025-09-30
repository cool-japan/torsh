use torsh_core::{DType, DeviceType, Shape, TorshError, TypePromotion};

#[cfg(test)]
mod scirs2_integration_tests {
    use super::*;

    #[test]
    fn test_scirs2_tensor_creation_compatibility() {
        // Test compatibility with SciRS2 tensor creation patterns
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        // Verify ToRSh can work with compatible shapes
        let torsh_shape = Shape::from_dims(shape.clone()).unwrap();
        assert_eq!(torsh_shape.dims(), &shape);
        assert_eq!(torsh_shape.numel(), data.len());

        // Verify dtype compatibility
        let torsh_dtype = DType::F32;
        assert_eq!(torsh_dtype.size(), std::mem::size_of::<f32>());
        assert!(torsh_dtype.is_float());
    }

    #[test]
    fn test_scirs2_shape_conversion() {
        // Test shape compatibility for SciRS2 integration
        let dims = vec![2, 3, 4];

        // Create ToRSh shape
        let torsh_shape = Shape::from_dims(dims.clone()).unwrap();

        // Verify compatibility
        assert_eq!(torsh_shape.dims(), &dims);
        assert_eq!(torsh_shape.ndim(), dims.len());
        assert_eq!(torsh_shape.numel(), dims.iter().product::<usize>());

        // Verify shape operations work correctly
        let broadcast_shape = Shape::from_dims(vec![1, 3, 1]).unwrap();
        let broadcasted = torsh_shape.broadcast_with(&broadcast_shape);
        assert!(broadcasted.is_ok());

        let result_shape = broadcasted.unwrap();
        assert_eq!(result_shape.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_scirs2_dtype_mapping() {
        // Test dtype compatibility for SciRS2 integration
        let test_cases = vec![
            DType::F32,
            DType::F64,
            DType::I32,
            DType::I64,
            DType::Bool,
            DType::C64,
            DType::C128,
        ];

        for torsh_dtype in test_cases {
            // Verify dtype properties
            assert!(!torsh_dtype.name().is_empty());
            assert!(torsh_dtype.size() > 0);

            // Verify type categorization
            match torsh_dtype {
                DType::F32 | DType::F64 => assert!(torsh_dtype.is_float()),
                DType::I32 | DType::I64 => assert!(torsh_dtype.is_int()),
                DType::Bool => {
                    // Bool type is valid
                    assert_eq!(torsh_dtype.size(), 1);
                }
                DType::C64 | DType::C128 => assert!(torsh_dtype.is_complex()),
                _ => {}
            }
        }
    }

    #[test]
    fn test_scirs2_backend_compatibility() {
        // Test device type compatibility for SciRS2 integration
        let device_type = DeviceType::Cpu;

        // Verify device properties
        assert_eq!(device_type, DeviceType::Cpu);

        // Test device type characteristics
        match device_type {
            DeviceType::Cpu => {
                // CPU device is always supported
            }
            DeviceType::Cuda(_) => {
                // CUDA device index handling
            }
            DeviceType::Metal(_) => {
                // Metal device support
            }
            DeviceType::Wgpu(_) => {
                // WebGPU device support
            }
        }
    }

    #[test]
    fn test_scirs2_tensor_operations() {
        // Test tensor operations that might use SciRS2 backend
        let shape1 = Shape::from_dims([2, 3]).unwrap();
        let shape2 = Shape::from_dims([3, 4]).unwrap();

        // Test matrix multiplication shape inference
        let result_shape = Shape::from_dims([2, 4]).unwrap();

        // Verify shapes are compatible for matrix multiplication
        assert_eq!(shape1.dims()[1], shape2.dims()[0]);
        assert_eq!(result_shape.dims()[0], shape1.dims()[0]);
        assert_eq!(result_shape.dims()[1], shape2.dims()[1]);

        // Test broadcasting compatibility
        let broadcast_shape = Shape::from_dims(vec![1, 3]).unwrap();
        let broadcasted = shape1.broadcast_with(&broadcast_shape);
        assert!(broadcasted.is_ok());

        let result = broadcasted.unwrap();
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_scirs2_error_handling() {
        // Test error handling compatibility with SciRS2

        // Test invalid shape creation
        let result = Shape::from_dims([0, 5]);
        assert!(result.is_err());

        if let Err(TorshError::InvalidShape(_)) = result {
            // Expected error type
        } else {
            panic!("Expected InvalidShape error");
        }

        // Test invalid broadcasting
        let shape1 = Shape::from_dims([2, 3]).unwrap();
        let invalid_shape = Shape::from_dims(vec![4, 5]).unwrap();

        let result = shape1.broadcast_with(&invalid_shape);
        assert!(result.is_err());

        if let Err(TorshError::BroadcastError { .. }) = result {
            // Expected error type
        } else {
            panic!("Expected BroadcastError");
        }
    }

    #[test]
    fn test_scirs2_memory_layout() {
        // Test memory layout compatibility
        let shape = Shape::from_dims([2, 3, 4]).unwrap();

        // Test strides calculation
        let strides = shape.strides();
        assert_eq!(strides.len(), 3);

        // Verify C-contiguous layout (row-major)
        assert_eq!(strides[0], 3 * 4); // 12
        assert_eq!(strides[1], 4); // 4
        assert_eq!(strides[2], 1); // 1

        // Test contiguity
        assert!(shape.is_contiguous());

        // Test memory requirements
        let dtype = DType::F32;
        let total_bytes = shape.numel() * dtype.size();
        assert_eq!(total_bytes, 2 * 3 * 4 * 4); // 96 bytes
    }

    #[test]
    fn test_scirs2_advanced_operations() {
        // Test advanced shape operations that would be used with SciRS2 backend

        // Test multi-dimensional shape creation
        let input_shape = Shape::from_dims([1, 3, 32, 32]).unwrap(); // NCHW
        let kernel_shape = Shape::from_dims([64, 3, 3, 3]).unwrap(); // OIHW

        // Verify shapes are valid
        assert_eq!(input_shape.ndim(), 4);
        assert_eq!(kernel_shape.ndim(), 4);
        assert_eq!(input_shape.dims()[1], kernel_shape.dims()[1]); // Channel compatibility

        // Test matrix operations shape compatibility
        let matrix_a = Shape::from_dims([10, 128]).unwrap();
        let matrix_b = Shape::from_dims([128, 64]).unwrap();

        // Verify matrix multiplication compatibility
        assert_eq!(matrix_a.dims()[1], matrix_b.dims()[0]);

        // Test tensor operations
        let linear_input = Shape::from_dims([10, 128]).unwrap();
        assert_eq!(linear_input.numel(), 10 * 128);

        // Test shape creation for different configurations
        let reshaped_dims = [2, 5, 128];
        let new_shape = Shape::from_dims(reshaped_dims).unwrap();
        assert_eq!(new_shape.dims(), &reshaped_dims);
        assert_eq!(new_shape.numel(), 2 * 5 * 128);
    }

    #[test]
    fn test_scirs2_type_promotion() {
        // Test type promotion rules that might be used by SciRS2

        // Test float promotion
        let result = DType::promote_types(DType::F32, DType::F64);
        assert_eq!(result, DType::F64);

        // Test int to float promotion
        let result = DType::promote_types(DType::I32, DType::F32);
        assert_eq!(result, DType::F32);

        // Test complex promotion
        let result = DType::promote_types(DType::F32, DType::C64);
        assert_eq!(result, DType::C64);

        let result = DType::promote_types(DType::F64, DType::C64);
        assert_eq!(result, DType::C128);

        // Test quantized type behavior
        let qint8 = DType::QInt8;
        let result = DType::promote_types(qint8, DType::I32);
        assert_eq!(result, DType::F32); // Quantized promotes to float
    }

    #[test]
    fn test_scirs2_device_synchronization() {
        // Test device type enumeration for SciRS2 integration
        let device_types = vec![
            DeviceType::Cpu,
            DeviceType::Cuda(0),
            DeviceType::Metal(0),
            DeviceType::Wgpu(0),
        ];

        for device_type in device_types {
            // Test device type properties
            match device_type {
                DeviceType::Cpu => {
                    // CPU is always available - no assertion needed
                }
                DeviceType::Cuda(index) => {
                    // CUDA device with index
                    assert!(index < 256); // Reasonable device limit
                }
                DeviceType::Metal(index) => {
                    // Metal device with index
                    assert!(index < 256); // Reasonable device limit
                }
                DeviceType::Wgpu(index) => {
                    // WebGPU device with index
                    assert!(index < 256); // Reasonable device limit
                }
            }
        }
    }

    #[test]
    fn test_scirs2_performance_integration() {
        // Test performance-related integration with SciRS2

        // Test large tensor operations
        let large_shape = Shape::from_dims([1000, 1000]).unwrap();
        assert_eq!(large_shape.numel(), 1_000_000);

        // Test broadcasting with large tensors
        let broadcast_shape = Shape::from_dims(vec![1, 1000]).unwrap();
        let result = large_shape.broadcast_with(&broadcast_shape);
        assert!(result.is_ok());

        // Test memory requirements for large tensors
        let dtype = DType::F32;
        let bytes_needed = large_shape.numel() * dtype.size();
        assert_eq!(bytes_needed, 4_000_000); // 4MB

        // Test stride calculation for large tensors
        let strides = large_shape.strides();
        assert_eq!(strides, &[1000, 1]);
    }
}
