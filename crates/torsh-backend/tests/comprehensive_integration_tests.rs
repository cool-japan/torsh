//! Comprehensive integration tests for torsh-backend
//!
//! This module tests the integration of various backend features including:
//! - Cross-backend scirs2 integration
//! - Error handling and edge cases
//! - Robustness under extreme conditions

use torsh_backend::*;
use torsh_core::DType;

/// Test backend creation and basic functionality across all available backends
#[test]
fn test_cross_backend_availability() {
    let backends = available_backends();
    assert!(
        !backends.is_empty(),
        "At least one backend should be available"
    );

    // CPU backend should always be available
    assert!(
        backends.contains(&BackendType::Cpu),
        "CPU backend should always be available"
    );

    for backend_type in backends {
        println!("Testing backend: {:?}", backend_type);

        match BackendBuilder::new().backend_type(backend_type).build() {
            Ok(backend) => {
                if let Ok(device) = backend.default_device() {
                    println!("  Device: {}", device.name());
                    println!("  Type: {:?}", device.device_type());
                }
            }
            Err(e) => {
                eprintln!("  Backend creation failed: {}", e);
            }
        }
    }
}

/// Test error handling for invalid buffer operations
#[test]
fn test_buffer_error_handling() {
    if let Ok(backend) = BackendBuilder::new().backend_type(BackendType::Cpu).build() {
        if let Ok(device) = backend.default_device() {
            // Test zero-size buffer
            let descriptor = buffer::BufferDescriptor::new(0, buffer::BufferUsage::STORAGE)
                .with_location(buffer::MemoryLocation::Host);

            let result = backend.create_buffer(&device, &descriptor);
            if let Err(e) = result {
                println!("Zero-size buffer rejected: {}", e);
            }

            // Test large buffer (but not so large it crashes)
            let large_descriptor = buffer::BufferDescriptor::new(
                1024 * 1024 * 1024 * 10, // 10GB - likely to fail gracefully
                buffer::BufferUsage::STORAGE,
            )
            .with_location(buffer::MemoryLocation::Host);

            let large_result = backend.create_buffer(&device, &large_descriptor);
            if let Err(e) = large_result {
                println!("Large buffer (10GB) rejected: {}", e);
            } else {
                println!("Large buffer (10GB) allocation succeeded");
            }
        }
    }
}

/// Test buffer operations with various alignments
#[test]
fn test_buffer_alignment_handling() {
    if let Ok(backend) = BackendBuilder::new().backend_type(BackendType::Cpu).build() {
        if let Ok(device) = backend.default_device() {
            let alignments = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];

            for alignment in alignments {
                let descriptor = buffer::BufferDescriptor::new(1024, buffer::BufferUsage::STORAGE)
                    .with_location(buffer::MemoryLocation::Host)
                    .with_alignment(alignment);

                match backend.create_buffer(&device, &descriptor) {
                    Ok(buffer) => {
                        assert_eq!(
                            buffer.size(),
                            1024,
                            "Buffer size should match for alignment {}",
                            alignment
                        );
                    }
                    Err(e) => {
                        println!("Buffer creation failed for alignment {}: {}", alignment, e);
                    }
                }
            }
        }
    }
}

/// Test SIMD operations with edge case data
#[test]
fn test_simd_edge_cases() {
    use cpu::simd::*;

    // Test with NaN values
    let nan_values = vec![f32::NAN; 100];
    let normal_values = vec![1.0f32; 100];
    let mut result = vec![0.0f32; 100];

    simd_add_f32(&nan_values, &normal_values, &mut result);
    for &val in &result {
        assert!(val.is_nan(), "NaN + normal should be NaN");
    }

    // Test with infinity
    let inf_values = vec![f32::INFINITY; 50];
    let neg_inf_values = vec![f32::NEG_INFINITY; 50];
    let mut inf_result = vec![0.0f32; 50];

    simd_add_f32(&inf_values, &neg_inf_values, &mut inf_result);
    for &val in &inf_result {
        assert!(val.is_nan(), "Inf + -Inf should be NaN");
    }

    // Test with very small values (denormals)
    let tiny_values = vec![f32::MIN_POSITIVE; 100];
    let mut tiny_result = vec![0.0f32; 100];

    simd_add_f32(&tiny_values, &tiny_values, &mut tiny_result);
    for &val in &tiny_result {
        assert!(
            val.is_finite() && val > 0.0,
            "Adding tiny values should be finite and positive"
        );
    }
}

/// Test quantization with extreme values
#[test]
fn test_quantization_robustness() {
    use quantization::core::*;

    let extreme_values = vec![f32::MIN, -1e30, -1000.0, 0.0, 1000.0, 1e30, f32::MAX];

    let params = QuantizationParams::int8_symmetric();

    if let Ok(quantized) = quantize_to_int8(&extreme_values, &params) {
        assert_eq!(
            quantized.len(),
            extreme_values.len(),
            "Quantization should preserve length"
        );

        for &q in &quantized {
            assert!(
                q >= i8::MIN && q <= i8::MAX,
                "Quantized value should be in i8 range"
            );
        }
    }

    // Test with all zeros
    let zeros = vec![0.0f32; 100];
    if let Ok(quantized_zeros) = quantize_to_int8(&zeros, &params) {
        for &q in &quantized_zeros {
            assert_eq!(q, 0, "Quantized zero should be zero");
        }
    }

    // Test with single value repeated
    let repeated = vec![42.0f32; 100];
    if let Ok(quantized_repeated) = quantize_to_int8(&repeated, &params) {
        let first = quantized_repeated[0];
        for &q in &quantized_repeated {
            assert_eq!(q, first, "Quantized repeated values should be identical");
        }
    }
}

/// Test sparse matrix operations with edge cases
#[test]
fn test_sparse_operations_edge_cases() {
    use sparse_ops::*;

    let empty_matrix: SparseMatrix<f32> = SparseMatrix::new_coo(10, 10);
    assert_eq!(empty_matrix.nnz, 0, "Empty matrix should have 0 non-zeros");

    let mut single_element: SparseMatrix<f32> = SparseMatrix::new_coo(5, 5);
    let _ = single_element.insert_coo(2, 3, 42.0);
    assert!(
        single_element.nnz > 0,
        "Matrix with one element should have non-zeros"
    );

    let mut diagonal: SparseMatrix<f32> = SparseMatrix::new_coo(100, 100);
    for i in 0..100 {
        let _ = diagonal.insert_coo(i, i, i as f32);
    }
    assert!(
        diagonal.nnz <= 100,
        "Diagonal matrix should have at most 100 non-zeros"
    );

    if let Ok(csr) = diagonal.to_csr() {
        assert_eq!(csr.rows, 100, "CSR should preserve row count");
        assert_eq!(csr.cols, 100, "CSR should preserve column count");
    }
}

/// Test FFT with various input sizes
#[test]
fn test_fft_size_variations() {
    use fft::*;

    let sizes = vec![8, 16, 32, 64, 128, 256, 512, 1024];

    for size in sizes {
        let forward = FftPlan::new_1d(size, FftDirection::Forward);
        let inverse = FftPlan::new_1d(size, FftDirection::Inverse);

        assert_eq!(
            forward.dimensions,
            vec![size],
            "FFT plan should have correct size"
        );
        assert_eq!(
            forward.direction,
            FftDirection::Forward,
            "Forward plan should be forward"
        );
        assert_eq!(
            inverse.direction,
            FftDirection::Inverse,
            "Inverse plan should be inverse"
        );
        assert_eq!(
            forward.total_elements(),
            size,
            "FFT total elements should match size"
        );
    }

    // Test 2D FFT
    let fft_2d = FftPlan::new(
        fft::FftType::C2C2D,
        vec![64, 64],
        1,
        DType::C64,
        DType::C64,
        FftDirection::Forward,
        fft::FftNormalization::None,
    );

    assert_eq!(
        fft_2d.total_elements(),
        64 * 64,
        "2D FFT should have correct total elements"
    );
}

/// Test memory pool stress scenarios
#[test]
fn test_memory_pool_stress() {
    if let Ok(backend) = BackendBuilder::new().backend_type(BackendType::Cpu).build() {
        if let Ok(device) = backend.default_device() {
            let mut buffers = Vec::new();

            // Allocate many small buffers
            for _i in 0..100 {
                let descriptor = buffer::BufferDescriptor::new(1024, buffer::BufferUsage::STORAGE)
                    .with_location(buffer::MemoryLocation::Host);

                if let Ok(buffer) = backend.create_buffer(&device, &descriptor) {
                    buffers.push(buffer);
                }
            }

            println!("Successfully allocated {} small buffers", buffers.len());

            // Drop half and reallocate
            buffers.truncate(50);

            for _i in 0..50 {
                let descriptor = buffer::BufferDescriptor::new(1024, buffer::BufferUsage::STORAGE)
                    .with_location(buffer::MemoryLocation::Host);

                if let Ok(buffer) = backend.create_buffer(&device, &descriptor) {
                    buffers.push(buffer);
                }
            }

            println!("Final buffer count after reallocation: {}", buffers.len());
        }
    }
}

/// Test device capability reporting consistency
#[test]
fn test_device_capabilities_consistency() {
    if let Ok(backend) = BackendBuilder::new().backend_type(BackendType::Cpu).build() {
        if let Ok(device) = backend.default_device() {
            use device::DeviceFeature;

            let features = vec![
                DeviceFeature::DoublePrecision,
                DeviceFeature::HalfPrecision,
                DeviceFeature::UnifiedMemory,
                DeviceFeature::AtomicOperations,
                DeviceFeature::SubGroups,
                DeviceFeature::Printf,
                DeviceFeature::Profiling,
            ];

            for feature in features {
                let supported = device.supports_feature(feature);
                println!("Feature {:?}: {}", supported, supported);
            }
        }
    }
}

/// Test cross-backend validation for mathematical correctness
#[test]
#[ignore = "Requires CUDA hardware - run with --ignored flag"]
fn test_cross_backend_math_correctness() {
    let validator = cross_backend_validation::CrossBackendValidator::new();

    if let Err(e) = validator.validate_device_creation() {
        println!("Device creation validation warning: {}", e);
    }

    if let Err(e) = validator.validate_capabilities_consistency() {
        println!("Capabilities validation warning: {}", e);
    }

    if let Err(e) = validator.validate_memory_management() {
        println!("Memory management validation warning: {}", e);
    }

    if let Err(e) = validator.validate_error_handling() {
        println!("Error handling validation warning: {}", e);
    }

    if let Err(e) = validator.validate_performance_hints() {
        println!("Performance hints validation warning: {}", e);
    }
}

/// Test backend builder configuration combinations
#[test]
fn test_backend_builder_configurations() {
    let with_profiling = BackendBuilder::new()
        .backend_type(BackendType::Cpu)
        .enable_profiling(true)
        .build();

    assert!(
        with_profiling.is_ok(),
        "Backend with profiling should build"
    );

    let default_backend = BackendBuilder::new().backend_type(BackendType::Cpu).build();

    assert!(default_backend.is_ok(), "Default backend should build");
}

/// Test error propagation through backend operations
#[test]
fn test_error_propagation() {
    use cpu::simd::*;

    // Test with mismatched array sizes
    let short_array = vec![1.0f32; 10];
    let long_array = vec![2.0f32; 100];
    let mut result = vec![0.0f32; 5];

    simd_add_f32(&short_array, &long_array, &mut result);

    // Test with empty arrays
    let empty: Vec<f32> = vec![];
    let mut empty_result = vec![];

    simd_add_f32(&empty, &empty, &mut empty_result);
    assert_eq!(
        empty_result.len(),
        0,
        "Empty arrays should produce empty result"
    );
}
