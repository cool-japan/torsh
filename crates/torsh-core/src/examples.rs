//! Comprehensive examples for torsh-core modules
//!
//! This module provides real-world examples and tutorials for using
//! the core functionality of the ToRSh tensor library.

use crate::{
    backend_detection::BackendFeatureDetector,
    device::CpuDevice,
    memory_monitor::{MemoryPressure, SystemMemoryMonitor},
    ConversionUtils, DType, Device, InteropDocs, NumpyArrayInfo, Result, Shape,
};

/// Examples for device operations
pub struct DeviceExamples;

impl DeviceExamples {
    /// Basic device creation and usage
    pub fn basic_device_usage() -> Result<()> {
        // Create a CPU device
        let cpu_device = CpuDevice::new();
        println!("Created CPU device: {:?}", cpu_device.device_type());

        // Check device availability
        if cpu_device.is_available().unwrap_or(false) {
            println!("Device is available for computation");
        }

        // Synchronize device operations
        cpu_device.synchronize()?;
        println!("Device synchronized successfully");

        Ok(())
    }

    /// Device capability detection
    pub fn device_capabilities() -> Result<()> {
        let detector = BackendFeatureDetector::new()?;

        // Access system capabilities
        let runtime_features = &detector.runtime_features;
        println!("Runtime features: {runtime_features:#?}");

        // Access available devices
        let available_devices = &detector.available_devices;
        println!("Available devices: {available_devices:#?}");

        // Check specific capabilities
        if runtime_features.cpu_features.simd.avx2 {
            println!("AVX2 is available for accelerated operations");
        }

        if runtime_features.cpu_features.simd.neon {
            println!("ARM NEON is available for vectorized operations");
        }

        Ok(())
    }

    /// Device synchronization patterns
    pub fn synchronization_patterns() -> Result<()> {
        let device = CpuDevice::new();

        // Basic synchronization
        device.synchronize()?;

        // Synchronization with timeout
        device.synchronize()?; // Timeout version not available

        // Wait for device to become idle
        device.synchronize()?; // Wait for idle version not available

        println!("All synchronization patterns completed successfully");
        Ok(())
    }
}

/// Examples for shape operations
pub struct ShapeExamples;

impl ShapeExamples {
    /// Basic shape creation and manipulation
    pub fn basic_shape_operations() -> Result<()> {
        // Create shapes
        let shape1 = Shape::new(vec![2, 3, 4]);
        let shape2 = Shape::new(vec![24]);

        println!("Shape 1: {:?}, elements: {}", shape1.dims(), shape1.numel());
        println!("Shape 2: {:?}, elements: {}", shape2.dims(), shape2.numel());

        // Check shape properties
        if shape1.is_contiguous() {
            println!("Shape 1 is contiguous");
        }

        if shape2.is_scalar() {
            println!("Shape 2 is scalar");
        } else {
            println!("Shape 2 is not scalar");
        }

        Ok(())
    }

    /// Shape broadcasting examples
    pub fn broadcasting_examples() -> Result<()> {
        let shape1 = Shape::new(vec![3, 1, 4]);
        let shape2 = Shape::new(vec![1, 2, 1]);

        // Check if shapes are broadcastable
        match shape1.broadcast_with(&shape2) {
            Ok(result_shape) => {
                println!(
                    "Broadcasting {:?} with {:?} gives {:?}",
                    shape1.dims(),
                    shape2.dims(),
                    result_shape.dims()
                );
            }
            Err(e) => {
                println!("Broadcasting failed: {e}");
            }
        }

        Ok(())
    }

    /// Advanced shape operations
    pub fn advanced_shape_operations() -> Result<()> {
        let shape = Shape::new(vec![2, 3, 4, 5]);

        // Get shape information
        println!("Shape: {:?}", shape.dims());
        println!("Number of dimensions: {}", shape.ndim());
        println!("Total elements: {}", shape.numel());
        println!("Is contiguous: {}", shape.is_contiguous());

        // Get strides
        let strides = shape.strides();
        println!("Strides: {strides:?}");

        // Create reshaped views (using new shape with compatible numel)
        let reshaped = Shape::new(vec![6, 20]);
        if shape.numel() == reshaped.numel() {
            println!("Reshaped to: {:?}", reshaped.dims());
        } else {
            println!(
                "Cannot reshape {} to {} - incompatible element count",
                shape.numel(),
                reshaped.numel()
            );
        }

        Ok(())
    }
}

/// Examples for data type operations
pub struct DTypeExamples;

impl DTypeExamples {
    /// Basic data type usage
    pub fn basic_dtype_operations() {
        // Different data types
        let dtypes = vec![
            DType::F32,
            DType::F64,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::Bool,
            DType::C64,
            DType::C128,
        ];

        for dtype in dtypes {
            let name = dtype.name();
            let size = dtype.size_bytes();
            let is_float = dtype.is_float();
            let is_int = dtype.is_int();
            let is_complex = dtype.is_complex();
            println!("Type: {name}, Size: {size} bytes, Float: {is_float}, Int: {is_int}, Complex: {is_complex}");
        }
    }

    /// Type promotion examples
    pub fn type_promotion_examples() {
        use crate::dtype::TypePromotion;

        // Promote types for mixed operations
        let common_type = <DType as TypePromotion>::common_type(&[DType::I32, DType::F32]);
        println!("Common type of I32 and F32: {common_type:?}");

        let common_type = <DType as TypePromotion>::common_type(&[DType::F32, DType::F64]);
        println!("Common type of F32 and F64: {common_type:?}");

        // Complex type promotion
        let common_type = <DType as TypePromotion>::common_type(&[DType::F32, DType::C64]);
        println!("Common type of F32 and C64: {common_type:?}");
    }

    /// Quantized type examples
    pub fn quantized_types() {
        use crate::dtype::{QInt8, QUInt8};

        // Create quantized types
        let qint8 = QInt8 {
            value: -100,
            scale: 0.5,
            zero_point: 0,
        };
        let quint8 = QUInt8 {
            value: 50,
            scale: 0.25,
            zero_point: 128,
        };

        println!("QInt8: value={}, scale={}", qint8.value, qint8.scale);
        println!("QUInt8: value={}, scale={}", quint8.value, quint8.scale);

        // Convert to/from float
        let float_val = qint8.value as f32 * qint8.scale;
        let back_to_qint8 = QInt8 {
            value: (float_val / 0.5) as i8,
            scale: 0.5,
            zero_point: 0,
        };

        println!("QInt8 -> f32: {float_val}");
        println!("f32 -> QInt8: value={}", back_to_qint8.value);
    }
}

/// Examples for memory management
pub struct MemoryExamples;

impl MemoryExamples {
    /// Memory pool usage
    pub fn memory_pool_usage() -> Result<()> {
        use crate::storage::{allocate_pooled, deallocate_pooled};

        // Allocate memory from pool
        let size = 1024;
        let _alignment = 64;
        let ptr: Vec<f32> = allocate_pooled(size);

        println!("Allocated {size} bytes from memory pool");

        // Check pool statistics
        println!("Pool allocation successful");

        // Deallocate memory
        deallocate_pooled(ptr);
        println!("Memory deallocated");

        Ok(())
    }

    /// System memory monitoring
    pub fn memory_monitoring() -> Result<()> {
        let monitor = SystemMemoryMonitor::new()?;

        // Get system memory stats
        let stats = monitor.current_stats();
        println!(
            "System memory: {} MB total, {} MB available",
            stats.total_physical / 1024 / 1024,
            stats.available_physical / 1024 / 1024
        );

        // Check memory pressure
        let pressure = stats.pressure;
        match pressure {
            crate::memory_monitor::MemoryPressure::Normal => println!("Memory pressure: Normal"),
            crate::memory_monitor::MemoryPressure::Moderate => {
                println!("Memory pressure: Moderate")
            }
            crate::memory_monitor::MemoryPressure::High => println!("Memory pressure: High"),
            crate::memory_monitor::MemoryPressure::Critical => {
                println!("Memory pressure: Critical")
            }
        }

        Ok(())
    }
}

/// Examples for interoperability features
pub struct InteropExamples;

impl InteropExamples {
    /// NumPy compatibility examples
    pub fn numpy_compatibility() {
        // Create NumPy-compatible array info
        let numpy_info = NumpyArrayInfo::new(vec![10, 20, 30], DType::F32);

        println!("NumPy array info:");
        println!("  Shape: {:?}", numpy_info.shape);
        println!("  Strides: {:?}", numpy_info.strides);
        println!("  C-contiguous: {}", numpy_info.c_contiguous);
        println!("  F-contiguous: {}", numpy_info.f_contiguous);
        println!("  Size in bytes: {}", numpy_info.nbytes);

        // Check layout efficiency
        let efficiency = ConversionUtils::layout_efficiency_score(
            &numpy_info.shape,
            &numpy_info.strides,
            numpy_info.dtype.size(),
        );
        println!("  Layout efficiency: {efficiency:.2}");
    }

    /// ONNX type conversion examples
    pub fn onnx_conversion() {
        use crate::interop::{OnnxDataType, OnnxTensorInfo};

        // Convert ToRSh types to ONNX
        let torsh_types = vec![DType::F32, DType::I64, DType::Bool, DType::C64];

        for dtype in torsh_types {
            let onnx_type = OnnxDataType::from(dtype);
            println!("ToRSh {dtype:?} -> ONNX {onnx_type:?}");

            // Convert back
            let back_to_torsh = DType::try_from(onnx_type).unwrap();
            assert_eq!(dtype, back_to_torsh);
        }

        // Create ONNX tensor info
        let tensor_info = OnnxTensorInfo {
            elem_type: OnnxDataType::Float,
            shape: vec![Some(10), None, Some(20)], // None for dynamic dimensions
            name: Some("example_tensor".to_string()),
        };

        println!("ONNX tensor info: {tensor_info:#?}");
    }

    /// Arrow integration examples
    pub fn arrow_integration() {
        use crate::interop::{ArrowDataType, ArrowTypeInfo};
        use std::collections::HashMap;

        // Convert ToRSh types to Arrow
        let dtype = DType::F64;
        let arrow_type = ArrowDataType::from(dtype);
        println!("ToRSh {dtype:?} -> Arrow {arrow_type:?}");

        // Complex type conversion
        let complex_dtype = DType::C128;
        let arrow_complex = ArrowDataType::from(complex_dtype);
        println!("ToRSh {complex_dtype:?} -> Arrow {arrow_complex:?}");

        // Create Arrow type info with metadata
        let mut metadata = HashMap::new();
        metadata.insert("origin".to_string(), "torsh".to_string());
        metadata.insert("version".to_string(), "0.1.0".to_string());

        let arrow_info = ArrowTypeInfo {
            data_type: arrow_type,
            metadata,
        };

        println!("Arrow type info: {arrow_info:#?}");
    }
}

/// Complete workflow examples
pub struct WorkflowExamples;

impl WorkflowExamples {
    /// Basic tensor creation workflow
    pub fn basic_tensor_workflow() -> Result<()> {
        println!("=== Basic Tensor Workflow ===");

        // 1. Create device
        let device = CpuDevice::new();
        println!("1. Created device: {:?}", device.device_type());

        // 2. Define shape and dtype
        let shape = Shape::new(vec![3, 4, 5]);
        let dtype = DType::F32;
        println!("2. Defined shape: {:?}, dtype: {:?}", shape.dims(), dtype);

        // 3. Check memory requirements
        let bytes_needed = shape.numel() * dtype.size_bytes();
        println!("3. Memory needed: {bytes_needed} bytes");

        // 4. Create NumPy-compatible info
        let numpy_info = NumpyArrayInfo::new(shape.dims().to_vec(), dtype);
        println!(
            "4. NumPy compatible: C-order={}, size={} bytes",
            numpy_info.c_contiguous, numpy_info.nbytes
        );

        // 5. Synchronize device
        device.synchronize()?;
        println!("5. Device synchronized");

        Ok(())
    }

    /// Memory-aware tensor processing
    pub fn memory_aware_processing() -> Result<()> {
        println!("=== Memory-Aware Processing ===");

        // 1. Check system memory
        let monitor = SystemMemoryMonitor::new()?;
        let stats = monitor.current_stats();

        println!(
            "1. System memory: {:.1} GB available",
            stats.available_physical as f64 / 1024.0 / 1024.0 / 1024.0
        );
        println!("   Memory pressure: {:?}", stats.pressure);

        // 2. Decide on tensor size based on available memory
        let max_elements = match stats.pressure {
            MemoryPressure::Normal => 1_000_000,
            MemoryPressure::Moderate => 500_000,
            MemoryPressure::High => 100_000,
            MemoryPressure::Critical => 10_000,
        };

        // 3. Create appropriately sized tensor
        let shape = if max_elements >= 1_000_000 {
            Shape::new(vec![100, 100, 100])
        } else if max_elements >= 100_000 {
            Shape::new(vec![50, 50, 40])
        } else {
            Shape::new(vec![20, 20, 25])
        };

        println!(
            "2. Selected shape: {:?} ({} elements)",
            shape.dims(),
            shape.numel()
        );

        // 4. Allocate using memory pool
        let size = shape.numel();
        let data: Vec<f32> = vec![0.0; size];
        println!("3. Allocated {size} elements");

        // 5. Process (simulated)
        println!("4. Processing tensor...");

        // 6. Clean up (automatic when Vec<f32> goes out of scope)
        drop(data);
        println!("5. Memory deallocated");

        Ok(())
    }

    /// Cross-platform compatibility workflow
    pub fn cross_platform_workflow() -> Result<()> {
        println!("=== Cross-Platform Compatibility ===");

        // 1. Detect platform capabilities
        let detector = BackendFeatureDetector::new()?;
        let features = &detector.runtime_features;

        println!("1. Platform detection:");
        println!("   Architecture: {:?}", features.cpu_features.architecture);
        println!(
            "   SIMD: AVX2={}, NEON={}",
            features.cpu_features.simd.avx2, features.cpu_features.simd.neon
        );

        // 2. Choose optimal data types
        let dtype = if features.cpu_features.simd.avx512f {
            println!("2. Using F32 (AVX-512 available)");
            DType::F32
        } else if features.cpu_features.simd.avx2 {
            println!("2. Using F32 (AVX2 available)");
            DType::F32
        } else {
            println!("2. Using F64 (fallback for precision)");
            DType::F64
        };

        // 3. Create tensors with optimal layout
        let shape = Shape::new(vec![32, 32, 32]); // Power of 2 for better SIMD
        let numpy_info = NumpyArrayInfo::new(shape.dims().to_vec(), dtype);

        println!("3. Created tensor:");
        println!("   Shape: {:?}", numpy_info.shape);
        println!(
            "   Layout efficiency: {:.2}",
            ConversionUtils::layout_efficiency_score(
                &numpy_info.shape,
                &numpy_info.strides,
                dtype.size()
            )
        );

        // 4. Show interoperability info
        println!("4. Interoperability:");
        let onnx_type = crate::interop::OnnxDataType::from(dtype);
        let arrow_type = crate::interop::ArrowDataType::from(dtype);
        println!("   ONNX type: {onnx_type:?}");
        println!("   Arrow type: {arrow_type:?}");

        Ok(())
    }
}

/// Performance optimization examples
pub struct PerformanceExamples;

impl PerformanceExamples {
    /// Memory layout optimization
    pub fn memory_layout_optimization() {
        println!("=== Memory Layout Optimization ===");

        let shapes_and_layouts = vec![
            ("C-contiguous", vec![1000, 1000], vec![4000, 4]),
            ("F-contiguous", vec![1000, 1000], vec![4, 4000]),
            ("Strided", vec![1000, 1000], vec![8000, 8]),
        ];

        for (name, shape, strides) in shapes_and_layouts {
            let efficiency = ConversionUtils::layout_efficiency_score(&shape, &strides, 4);
            println!("{name}: efficiency = {efficiency:.3}");

            if efficiency > 0.9 {
                println!("  ✓ Excellent layout for performance");
            } else if efficiency > 0.7 {
                println!("  ⚠ Good layout, some optimization possible");
            } else {
                println!("  ⚡ Consider layout optimization");
            }
        }
    }

    /// SIMD optimization guidance
    pub fn simd_optimization_guidance() -> Result<()> {
        println!("=== SIMD Optimization Guidance ===");

        let detector = BackendFeatureDetector::new()?;
        let features = &detector.runtime_features;

        // Vector widths for different SIMD instruction sets
        let vector_widths = if features.cpu_features.simd.avx512f {
            println!("AVX-512 detected: 16 floats per vector");
            16
        } else if features.cpu_features.simd.avx2 {
            println!("AVX2 detected: 8 floats per vector");
            8
        } else if features.cpu_features.simd.neon {
            println!("NEON detected: 4 floats per vector");
            4
        } else {
            println!("No SIMD detected: scalar operations");
            1
        };

        // Recommend tensor sizes
        let recommended_sizes = vec![vector_widths * 32, vector_widths * 64, vector_widths * 128];

        println!("Recommended tensor sizes for optimal SIMD usage:");
        for size in recommended_sizes {
            println!(
                "  {} elements ({}x vector width)",
                size,
                size / vector_widths
            );
        }

        Ok(())
    }
}

/// Documentation and help utilities
pub struct DocumentationExamples;

impl DocumentationExamples {
    /// Print comprehensive help
    pub fn print_help() {
        println!("{}", InteropDocs::supported_conversions());
        println!("{}", InteropDocs::conversion_examples());
    }

    /// Print API overview
    pub fn api_overview() {
        println!(
            r#"
ToRSh Core API Overview
======================

Core Modules:
• device    - Device abstraction and management
• dtype     - Data type definitions and operations
• shape     - Shape and stride utilities
• storage   - Memory management and allocation
• interop   - Interoperability with other libraries
• error     - Error handling and reporting

Key Types:
• Device    - Hardware device abstraction
• DType     - Tensor data types (F32, I64, C64, etc.)
• Shape     - Tensor dimensions and layout
• TorshError - Comprehensive error handling

Getting Started:
1. Create a device: let device = CpuDevice::new();
2. Define shape: let shape = Shape::new(vec![2, 3, 4]);
3. Choose dtype: let dtype = DType::F32;
4. Check interop: let numpy_info = NumpyArrayInfo::new(...);

For detailed examples, see the examples module.
"#
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_examples() {
        assert!(DeviceExamples::basic_device_usage().is_ok());
        assert!(DeviceExamples::device_capabilities().is_ok());
        assert!(DeviceExamples::synchronization_patterns().is_ok());
    }

    #[test]
    fn test_shape_examples() {
        assert!(ShapeExamples::basic_shape_operations().is_ok());
        assert!(ShapeExamples::broadcasting_examples().is_ok());
        assert!(ShapeExamples::advanced_shape_operations().is_ok());
    }

    #[test]
    fn test_dtype_examples() {
        DTypeExamples::basic_dtype_operations();
        DTypeExamples::type_promotion_examples();
        DTypeExamples::quantized_types();
    }

    #[test]
    fn test_memory_examples() {
        assert!(MemoryExamples::memory_pool_usage().is_ok());
        assert!(MemoryExamples::memory_monitoring().is_ok());
    }

    #[test]
    fn test_interop_examples() {
        InteropExamples::numpy_compatibility();
        InteropExamples::onnx_conversion();
        InteropExamples::arrow_integration();
    }

    #[test]
    fn test_workflow_examples() {
        assert!(WorkflowExamples::basic_tensor_workflow().is_ok());
        assert!(WorkflowExamples::memory_aware_processing().is_ok());
        assert!(WorkflowExamples::cross_platform_workflow().is_ok());
    }

    #[test]
    fn test_performance_examples() {
        PerformanceExamples::memory_layout_optimization();
        assert!(PerformanceExamples::simd_optimization_guidance().is_ok());
    }

    #[test]
    fn test_documentation_examples() {
        DocumentationExamples::print_help();
        DocumentationExamples::api_overview();
    }
}
