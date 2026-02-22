//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::types::{
        CompilationTarget, CpuCompiler, GeneratedKernel, KernelCache, KernelDataType,
        KernelGenerator, KernelOperation, KernelSpec, OpenCLCompiler, OptimizationFlags,
        ReductionOp,
    };
    use crate::error::BackendError;
    #[test]
    fn test_kernel_data_type_properties() {
        assert_eq!(KernelDataType::F32.size(), 4);
        assert_eq!(KernelDataType::F64.size(), 8);
        assert_eq!(KernelDataType::I32.to_c_type(), "int");
        assert_eq!(KernelDataType::F32.to_spirv_type(), "f32");
    }
    #[test]
    fn test_kernel_spec_creation() {
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::CUDA {
                compute_capability: (7, 5),
            },
        );
        assert_eq!(spec.input_types.len(), 2);
        assert_eq!(spec.output_type, KernelDataType::F32);
        assert!(!spec.hash_key().is_empty());
    }
    #[test]
    fn test_kernel_cache() {
        let cache = KernelCache::new(2);
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100]],
            vec![100],
            CompilationTarget::CUDA {
                compute_capability: (7, 5),
            },
        );
        let kernel = GeneratedKernel {
            source_code: "test".to_string(),
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec: spec.clone(),
            compilation_time_ms: 100,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        };
        let key = "test_key".to_string();
        cache.insert(key.clone(), kernel.clone());
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        let stats = cache.statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.cache_size, 1);
    }
    #[test]
    fn test_webgpu_kernel_generation() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );
        let result = generator.generate_kernel(spec);
        assert!(result.is_ok());
        let kernel = result.unwrap();
        assert!(!kernel.source_code.is_empty());
        assert_eq!(kernel.entry_point, "main");
    }
    #[test]
    fn test_cuda_kernel_generation() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseMul,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::CUDA {
                compute_capability: (7, 5),
            },
        );
        let result = generator.generate_kernel(spec);
        assert!(result.is_ok());
        let kernel = result.unwrap();
        assert!(!kernel.source_code.is_empty());
        assert!(kernel.source_code.contains("__global__"));
    }
    #[test]
    fn test_matrix_multiply_kernel() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::MatrixMultiply {
                m: 128,
                n: 128,
                k: 128,
            },
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![128, 128], vec![128, 128]],
            vec![128, 128],
            CompilationTarget::WebGPU,
        );
        let result = generator.generate_kernel(spec);
        assert!(result.is_ok());
        let kernel = result.unwrap();
        assert!(kernel.source_code.contains("workgroup"));
        assert!(kernel.source_code.contains("matrix"));
    }
    #[test]
    fn test_optimization_flags() {
        let mut flags = OptimizationFlags::default();
        assert!(flags.vectorization);
        assert!(flags.loop_unrolling);
        flags.tensor_cores = true;
        assert!(flags.tensor_cores);
    }
    #[test]
    fn test_cache_eviction() {
        let cache = KernelCache::new(1);
        let kernel1 = GeneratedKernel {
            source_code: "test1".to_string(),
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec: KernelSpec::new(
                KernelOperation::ElementwiseAdd,
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![100]],
                vec![100],
                CompilationTarget::WebGPU,
            ),
            compilation_time_ms: 100,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        };
        let kernel2 = GeneratedKernel {
            source_code: "test2".to_string(),
            entry_point: "main".to_string(),
            compiled_binary: None,
            spec: KernelSpec::new(
                KernelOperation::ElementwiseMul,
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![100]],
                vec![100],
                CompilationTarget::WebGPU,
            ),
            compilation_time_ms: 100,
            estimated_performance: 1.0,
            register_usage: None,
            shared_memory_usage: None,
        };
        cache.insert("key1".to_string(), kernel1);
        cache.insert("key2".to_string(), kernel2);
        assert!(cache.get("key2").is_some());
        assert_eq!(cache.statistics().cache_size, 1);
    }
    #[test]
    fn test_opencl_kernel_generation() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::OpenCL {
                version: "2.0".to_string(),
            },
        );
        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(!kernel.source_code.is_empty());
                assert!(kernel.source_code.contains("__kernel"));
                assert!(kernel.source_code.contains("__global"));
                assert_eq!(kernel.entry_point, "kernel_main");
            }
            Err(BackendError::BackendError(msg)) => {
                assert!(msg.contains("OpenCL not available"));
            }
            _ => panic!("Unexpected error type"),
        }
    }
    #[test]
    fn test_opencl_matrix_multiply_kernel() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::MatrixMultiply {
                m: 64,
                n: 64,
                k: 64,
            },
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![64, 64], vec![64, 64]],
            vec![64, 64],
            CompilationTarget::OpenCL {
                version: "2.0".to_string(),
            },
        );
        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(kernel.source_code.contains("TILE_SIZE"));
                assert!(kernel.source_code.contains("__local"));
                assert!(kernel.source_code.contains("barrier"));
            }
            Err(BackendError::BackendError(_)) => {}
            _ => panic!("Unexpected error type"),
        }
    }
    #[test]
    fn test_cpu_kernel_generation() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseMul,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![1000], vec![1000]],
            vec![1000],
            CompilationTarget::CPU {
                architecture: "x86_64".to_string(),
            },
        );
        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(!kernel.source_code.is_empty());
                assert!(kernel.source_code.contains("extern \"C\""));
                assert!(kernel.source_code.contains("__restrict__"));
                assert!(kernel.source_code.contains("#pragma omp"));
                assert_eq!(kernel.entry_point, "kernel_main");
            }
            Err(BackendError::BackendError(msg)) => {
                assert!(msg.contains("No C compiler available"));
            }
            _ => panic!("Unexpected error type"),
        }
    }
    #[test]
    fn test_cpu_simd_support_detection() {
        let cpu_compiler = CpuCompiler::new();
        let _support = &cpu_compiler.simd_support;
    }
    #[test]
    fn test_cpu_advanced_operations() {
        let mut generator = KernelGenerator::new();
        let operations = vec![
            KernelOperation::ReLU,
            KernelOperation::GELU,
            KernelOperation::Softmax { dim: 1 },
            KernelOperation::Transpose { dims: vec![0, 1] },
            KernelOperation::Reduction {
                op: ReductionOp::Sum,
                dim: Some(0),
            },
        ];
        for operation in operations {
            let spec = KernelSpec::new(
                operation,
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![100]],
                vec![100],
                CompilationTarget::CPU {
                    architecture: "x86_64".to_string(),
                },
            );
            let result = generator.generate_kernel(spec);
            match result {
                Ok(kernel) => {
                    assert!(!kernel.source_code.is_empty());
                    assert!(kernel.source_code.contains("extern \"C\""));
                }
                Err(BackendError::BackendError(_)) => {}
                _ => panic!("Unexpected error type"),
            }
        }
    }
    #[test]
    fn test_spirv_kernel_generation() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::ElementwiseDiv,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![256], vec![256]],
            vec![256],
            CompilationTarget::SPIRV,
        );
        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(!kernel.source_code.is_empty());
                assert!(kernel.source_code.contains("#version 450"));
                assert!(kernel.source_code.contains("layout("));
                assert!(kernel.source_code.contains("gl_GlobalInvocationID"));
                assert_eq!(kernel.entry_point, "main");
            }
            Err(BackendError::BackendError(msg)) => {
                assert!(msg.contains("glslc compiler not available"));
            }
            _ => panic!("Unexpected error type"),
        }
    }
    #[test]
    fn test_spirv_advanced_operations() {
        let mut generator = KernelGenerator::new();
        let operations = vec![
            (KernelOperation::ReLU, "max"),
            (KernelOperation::GELU, "tanh"),
            (KernelOperation::Softmax { dim: 1 }, "atomicMax"),
            (
                KernelOperation::Transpose { dims: vec![0, 1] },
                "gl_GlobalInvocationID.y",
            ),
        ];
        for (operation, expected_content) in operations {
            let spec = KernelSpec::new(
                operation,
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![100]],
                vec![100],
                CompilationTarget::SPIRV,
            );
            let result = generator.generate_kernel(spec);
            match result {
                Ok(kernel) => {
                    assert!(!kernel.source_code.is_empty());
                    assert!(kernel.source_code.contains("#version 450"));
                    assert!(kernel.source_code.contains(expected_content));
                }
                Err(BackendError::BackendError(_)) => {}
                _ => panic!("Unexpected error type"),
            }
        }
    }
    #[test]
    fn test_spirv_matrix_multiply_with_shared_memory() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::MatrixMultiply {
                m: 128,
                n: 128,
                k: 128,
            },
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![128, 128], vec![128, 128]],
            vec![128, 128],
            CompilationTarget::SPIRV,
        );
        let result = generator.generate_kernel(spec);
        match result {
            Ok(kernel) => {
                assert!(kernel.source_code.contains("shared"));
                assert!(kernel.source_code.contains("TILE_SIZE"));
                assert!(kernel.source_code.contains("barrier()"));
                assert!(kernel.source_code.contains("gl_WorkGroupID"));
            }
            Err(BackendError::BackendError(_)) => {}
            _ => panic!("Unexpected error type"),
        }
    }
    #[test]
    fn test_data_type_conversions() {
        assert_eq!(KernelDataType::F32.size(), 4);
        assert_eq!(KernelDataType::F64.size(), 8);
        assert_eq!(KernelDataType::F16.size(), 2);
        assert_eq!(KernelDataType::F32.to_c_type(), "float");
        assert_eq!(KernelDataType::F64.to_c_type(), "double");
        assert_eq!(KernelDataType::I32.to_c_type(), "int");
        assert_eq!(KernelDataType::F32.to_spirv_type(), "f32");
        assert_eq!(KernelDataType::U32.to_spirv_type(), "u32");
    }
    #[test]
    fn test_kernel_spec_with_custom_options() {
        let spec = KernelSpec::new(
            KernelOperation::MatrixMultiply {
                m: 256,
                n: 256,
                k: 256,
            },
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![256, 256], vec![256, 256]],
            vec![256, 256],
            CompilationTarget::CUDA {
                compute_capability: (8, 0),
            },
        )
        .with_tensor_cores()
        .with_workgroup_size((32, 32, 1))
        .with_shared_memory(49152);
        assert!(spec.optimization_flags.tensor_cores);
        assert_eq!(spec.workgroup_size, Some((32, 32, 1)));
        assert_eq!(spec.shared_memory_size, Some(49152));
    }
    #[test]
    fn test_optimization_flags_defaults() {
        let flags = OptimizationFlags::default();
        assert!(flags.vectorization);
        assert!(flags.loop_unrolling);
        assert!(flags.memory_coalescing);
        assert!(flags.shared_memory_usage);
        assert!(!flags.tensor_cores);
        assert!(!flags.auto_tuning);
        assert!(!flags.aggressive_inlining);
        assert!(flags.math_optimizations);
    }
    #[test]
    fn test_unsupported_operations() {
        let mut generator = KernelGenerator::new();
        let spec = KernelSpec::new(
            KernelOperation::Custom {
                name: "unsupported_operation".to_string(),
            },
            vec![KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );
        let result = generator.generate_kernel(spec);
        assert!(result.is_err());
        match result {
            Err(BackendError::NotImplemented(_)) => {}
            _ => panic!("Should return NotImplemented error"),
        }
    }
    #[test]
    fn test_unsupported_data_types() {
        let opencl_compiler = OpenCLCompiler::new();
        let cpu_compiler = CpuCompiler::new();
        assert!(opencl_compiler.opencl_type(KernelDataType::BF16).is_err());
        assert!(cpu_compiler.cpu_type(KernelDataType::BF16).is_err());
    }
    #[test]
    fn test_reduction_operations() {
        let operations = vec![
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::Mean,
            ReductionOp::Product,
        ];
        let mut generator = KernelGenerator::new();
        for op in operations {
            let spec = KernelSpec::new(
                KernelOperation::Reduction {
                    op: op.clone(),
                    dim: Some(0),
                },
                vec![KernelDataType::F32],
                KernelDataType::F32,
                vec![vec![1000]],
                vec![1],
                CompilationTarget::WebGPU,
            );
            let result = generator.generate_kernel(spec);
            assert!(result.is_ok());
            let kernel = result.unwrap();
            assert!(!kernel.source_code.is_empty());
        }
    }
    #[test]
    fn test_kernel_hash_consistency() {
        let spec1 = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );
        let spec2 = KernelSpec::new(
            KernelOperation::ElementwiseAdd,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );
        let spec3 = KernelSpec::new(
            KernelOperation::ElementwiseMul,
            vec![KernelDataType::F32, KernelDataType::F32],
            KernelDataType::F32,
            vec![vec![100], vec![100]],
            vec![100],
            CompilationTarget::WebGPU,
        );
        assert_eq!(spec1.hash_key(), spec2.hash_key());
        assert_ne!(spec1.hash_key(), spec3.hash_key());
    }
}
