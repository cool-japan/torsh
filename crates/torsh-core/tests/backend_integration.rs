//! Integration tests for different backend combinations
//!
//! These tests verify that torsh-core works correctly with various backend configurations
//! and that the backend detection and feature systems work as expected.

#![allow(clippy::uninlined_format_args)]

use torsh_core::{
    backend_detection::BackendFeatureDetector,
    device::{CpuDevice, Device, DeviceType},
    memory_monitor::{MemoryPressure, SystemMemoryMonitor},
    DType, Shape, TorshError,
};

/// Test backend detection functionality
#[test]
fn test_backend_detection() {
    let detector = BackendFeatureDetector::new().expect("Failed to create detector");

    // Test that CPU features are detected
    assert!(
        detector.runtime_features.cpu_features.physical_cores > 0,
        "Should detect physical cores"
    );
    assert!(
        detector.runtime_features.cpu_features.logical_cores > 0,
        "Should detect logical cores"
    );
    assert!(
        !detector
            .runtime_features
            .cpu_features
            .architecture
            .is_empty(),
        "Should detect architecture"
    );

    // Test device availability
    assert!(
        !detector.available_devices.is_empty(),
        "Should detect at least CPU device"
    );

    // Test CPU device specifically
    let cpu_devices: Vec<_> = detector
        .available_devices
        .iter()
        .filter(|d| matches!(d.device_type, DeviceType::Cpu))
        .collect();
    assert!(!cpu_devices.is_empty(), "Should detect CPU device");
}

/// Test device capability queries across different device types
#[test]
fn test_device_capabilities() {
    let cpu_device = CpuDevice::new();

    // Test basic device properties
    assert_eq!(cpu_device.device_type(), DeviceType::Cpu);
    assert!(
        cpu_device.is_available().unwrap_or(false),
        "CPU device should be available"
    );

    // Test device capabilities
    let _capabilities = cpu_device
        .capabilities()
        .expect("Should get CPU capabilities");

    // CPU should support basic operations
    assert!(
        cpu_device
            .supports_feature(DType::F32.name())
            .unwrap_or(false),
        "CPU should support F32"
    );
    assert!(
        cpu_device
            .supports_feature(DType::F64.name())
            .unwrap_or(false),
        "CPU should support F64"
    );
    assert!(
        cpu_device
            .supports_feature(DType::I32.name())
            .unwrap_or(false),
        "CPU should support I32"
    );
    assert!(
        cpu_device
            .supports_feature(DType::I64.name())
            .unwrap_or(false),
        "CPU should support I64"
    );

    // Test memory info
    let memory_info = cpu_device.memory_info().expect("Should get memory info");
    assert!(memory_info.total > 0, "CPU should have some total memory");
    assert!(
        memory_info.available <= memory_info.total,
        "Available memory should be <= total memory"
    );

    // Test device name
    assert_eq!(cpu_device.name(), "CPU");
}

/// Test backend feature combinations
#[test]
fn test_backend_feature_combinations() {
    let detector = BackendFeatureDetector::new().expect("Should create detector");
    let runtime_features = &detector.runtime_features;

    // Test CPU features
    assert!(
        runtime_features.cpu_features.physical_cores > 0,
        "Should detect physical cores"
    );
    assert!(
        runtime_features.cpu_features.logical_cores > 0,
        "Should detect logical cores"
    );
    assert!(
        !runtime_features.cpu_features.architecture.is_empty(),
        "Should detect architecture"
    );

    // Test SIMD availability
    let simd = &runtime_features.cpu_features.simd;
    let has_simd = simd.sse || simd.avx || simd.avx2 || simd.neon || simd.fma;

    if has_simd {
        println!("SIMD features detected: {simd:?}");
    }

    // Test system features
    let system_features = &runtime_features.system_features;
    assert!(system_features.page_size > 0, "Should detect page size");
}

/// Test performance tier classification
#[test]
fn test_performance_tier_classification() {
    let detector = BackendFeatureDetector::new().expect("Should create detector");
    let summary = &detector.backend_summary;

    // Test that we have detected some performance characteristics
    println!(
        "Performance tier: {performance_tier:?}",
        performance_tier = summary.performance_tier
    );

    // Test memory classification (from system features)
    assert!(
        detector.runtime_features.system_features.total_memory > 0,
        "Should detect system memory"
    );

    // Test device count (from available devices)
    assert!(
        !detector.available_devices.is_empty(),
        "Should detect at least one device"
    );

    // Test that we have at least one available device
    assert!(
        summary.recommended_device.is_some(),
        "Should have a recommended device"
    );
}

/// Test memory monitoring integration
#[test]
fn test_memory_monitoring_integration() {
    let monitor = SystemMemoryMonitor::new().expect("Should create memory monitor");

    // Test basic memory statistics
    let stats = monitor.current_stats();
    assert!(stats.total_physical > 0, "Should detect total memory");
    assert!(
        stats.available_physical <= stats.total_physical,
        "Available should be <= total"
    );

    // Test memory pressure detection (from stats)
    let pressure = stats.pressure;

    match pressure {
        MemoryPressure::Normal => {
            println!("Normal memory pressure detected");
        }
        MemoryPressure::Moderate => {
            println!("Moderate memory pressure detected");
        }
        MemoryPressure::High => {
            println!("High memory pressure detected");
        }
        MemoryPressure::Critical => {
            println!("Critical memory pressure detected");
        }
    }

    println!("Memory stats: {stats:?}");
    println!("Memory pressure: {pressure:?}");
}

/// Test backend compatibility with different data types
#[test]
fn test_backend_dtype_compatibility() {
    let detector = BackendFeatureDetector::new().expect("Should create detector");
    let devices = &detector.available_devices;

    let test_dtypes = [
        DType::F32,
        DType::F64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::Bool,
    ];

    #[cfg(feature = "half")]
    let extended_dtypes = [DType::F16, DType::BF16, DType::C64, DType::C128];

    for device_info in devices {
        // For now, only test CPU device
        if matches!(device_info.device_type, DeviceType::Cpu) {
            let cpu_device = CpuDevice::new();

            println!(
                "Testing device: {device_type:?}",
                device_type = device_info.device_type
            );

            // Test basic data types
            for &dtype in &test_dtypes {
                let supported = cpu_device.supports_feature(dtype.name()).unwrap_or(false);
                println!("  {dtype:?}: {}", if supported { "✅" } else { "❌" });

                // CPU should support all basic types
                assert!(supported, "CPU should support {dtype:?}");
            }

            // Test extended data types if available
            #[cfg(feature = "half")]
            {
                for &dtype in &extended_dtypes {
                    let supported = cpu_device.supports_feature(dtype.name()).unwrap_or(false);
                    println!("  {dtype:?}: {}", if supported { "✅" } else { "❌" });
                }
            }
        }
    }
}

/// Test shape operations across different backends
#[test]
fn test_shape_operations_backend_compatibility() {
    let test_shapes = [
        Shape::new(vec![1]),          // 1D
        Shape::new(vec![2, 3]),       // 2D
        Shape::new(vec![2, 3, 4]),    // 3D
        Shape::new(vec![2, 3, 4, 5]), // 4D
        Shape::new(vec![1, 1, 1, 1]), // All ones
        Shape::scalar(),              // Scalar
    ];

    let detector = BackendFeatureDetector::new().expect("Should create detector");
    let devices = &detector.available_devices;

    for device_info in devices {
        println!(
            "Testing shape operations on device: {device_type:?}",
            device_type = device_info.device_type
        );

        for shape in &test_shapes {
            // Test basic shape properties
            assert_eq!(shape.ndim(), shape.dims().len());

            // Test stride calculation
            let strides = shape.default_strides();
            assert_eq!(strides.len(), shape.ndim());

            // Test element count
            let numel = shape.numel();
            let expected_numel: usize = shape.dims().iter().product();
            assert_eq!(numel, expected_numel);

            // Test broadcasting with scalar
            let scalar = Shape::scalar();
            assert!(shape.broadcast_with(&scalar).is_ok());
            assert!(scalar.broadcast_with(shape).is_ok());

            // Test broadcasting result
            let broadcast_result = shape.broadcast_with(&scalar);
            assert!(broadcast_result.is_ok());
            if let Ok(result) = broadcast_result {
                assert_eq!(result.dims(), shape.dims());
            }

            println!("  Shape {shape:?}: ✅");
        }
    }
}

/// Test error handling across different backends
#[test]
fn test_backend_error_handling() {
    // Test invalid shape creation should work the same on all backends
    let invalid_shape_result = Shape::from_dims(vec![0]);

    assert!(invalid_shape_result.is_err());

    if let Err(error) = invalid_shape_result {
        match error {
            TorshError::InvalidShape(_) => {
                println!("  Invalid shape error: ✅");
            }
            _ => {
                panic!("Expected InvalidShape error, got: {:?}", error);
            }
        }
    }

    // Test invalid broadcasting
    let shape1 = Shape::new(vec![3, 4]);
    let shape2 = Shape::new(vec![5, 6]);

    assert!(shape1.broadcast_with(&shape2).is_err());

    let broadcast_result = shape1.broadcast_with(&shape2);
    assert!(broadcast_result.is_err());

    if let Err(error) = broadcast_result {
        match error {
            TorshError::BroadcastError { .. } => {
                println!("  Invalid broadcast error: ✅");
            }
            _ => {
                panic!("Expected BroadcastError for broadcasting, got: {:?}", error);
            }
        }
    }
}

/// Test concurrent operations across backends
#[test]
fn test_concurrent_backend_operations() {
    use std::sync::Arc;
    use std::thread;

    let detector = Arc::new(BackendFeatureDetector::new().expect("Should create detector"));

    const NUM_THREADS: usize = 4;
    const OPERATIONS_PER_THREAD: usize = 100;

    let mut handles = Vec::new();

    for thread_id in 0..NUM_THREADS {
        let detector_clone = Arc::clone(&detector);

        let handle = thread::spawn(move || {
            for i in 0..OPERATIONS_PER_THREAD {
                // Test device operations
                let cpu_device = CpuDevice::new();
                assert_eq!(cpu_device.device_type(), DeviceType::Cpu);
                assert!(cpu_device
                    .supports_feature(DType::F32.name())
                    .unwrap_or(false));

                // Test shape operations
                let shape = Shape::new(vec![thread_id + 1, i + 1]);
                assert_eq!(shape.numel(), (thread_id + 1) * (i + 1));

                // Test backend detection properties
                assert!(!detector_clone.available_devices.is_empty());
                assert!(detector_clone.runtime_features.cpu_features.physical_cores > 0);
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    println!("Concurrent backend operations completed successfully");
}

/// Test backend feature detection consistency
#[test]
fn test_backend_detection_consistency() {
    // Run detection multiple times and ensure consistency
    let mut detectors = Vec::new();

    for _ in 0..5 {
        let detector = BackendFeatureDetector::new().expect("Should create detector");
        detectors.push(detector);
    }

    // Check that key properties are consistent across runs
    let first_detector = &detectors[0];

    for detector in &detectors[1..] {
        assert_eq!(
            detector.available_devices.len(),
            first_detector.available_devices.len(),
            "Device count should be consistent"
        );

        assert_eq!(
            detector.backend_summary.performance_tier,
            first_detector.backend_summary.performance_tier,
            "Performance tier should be consistent"
        );

        assert_eq!(
            detector.runtime_features.cpu_features.physical_cores,
            first_detector.runtime_features.cpu_features.physical_cores,
            "Physical core count should be consistent"
        );

        // Memory might vary slightly due to system activity, but should be close
        let memory_diff = detector
            .runtime_features
            .system_features
            .total_memory
            .abs_diff(first_detector.runtime_features.system_features.total_memory);

        let memory_variance = memory_diff as f64
            / first_detector.runtime_features.system_features.total_memory as f64;
        assert!(
            memory_variance < 0.1,
            "Memory detection should be reasonably consistent (variance: {:.2}%)",
            memory_variance * 100.0
        );
    }

    println!("Backend detection consistency verified");
}
