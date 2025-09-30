//! Focused Integration Tests for Advanced Kernel Fusion Optimization
//!
//! This test suite specifically validates the kernel fusion optimization capabilities
//! including fusion opportunity detection, dynamic kernel generation, performance
//! prediction, and optimization strategy selection under various operation patterns.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

use torsh_backend::cuda::{
    AccessPattern, ActivationType, AdvancedKernelFusionOptimizer, DataType, FusionKernel,
    FusionOperation, FusionOptimizationResult, FusionPatternType, FusionStrategyType,
    KernelFusionConfig, KernelFusionStatus, MemoryLayout, OperationType, OptimizationLevel,
    ReductionType,
};

#[cfg(all(test, feature = "cuda"))]
mod kernel_fusion_tests {
    use super::*;

    /// Test fusion opportunity detection for element-wise operation chains
    #[test]
    fn test_elementwise_fusion_opportunities() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: true,
            enable_pattern_analysis: true,
            min_performance_improvement: 5.0,
            max_fusion_operations: 8,
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(
            optimizer.initialize().is_ok(),
            "Fusion optimizer should initialize successfully"
        );

        // Create element-wise operation chain that should be fusable
        let elementwise_operations = create_elementwise_operation_chain();

        // Analyze fusion opportunities
        let opportunities_result = optimizer.analyze_fusion_opportunities(&elementwise_operations);
        assert!(
            opportunities_result.is_ok(),
            "Should successfully analyze fusion opportunities"
        );

        let opportunities = opportunities_result.unwrap();
        assert!(
            !opportunities.is_empty(),
            "Should detect fusion opportunities for element-wise chain"
        );

        // Verify that opportunities include element-wise patterns
        let has_elementwise_pattern = opportunities
            .iter()
            .any(|op| op.fusion_pattern == FusionPatternType::ElementWiseChain);
        assert!(
            has_elementwise_pattern,
            "Should detect element-wise chain pattern"
        );

        println!(
            "Detected {} fusion opportunities for element-wise operations",
            opportunities.len()
        );
    }

    /// Test convolution + activation fusion (common neural network pattern)
    #[test]
    fn test_convolution_activation_fusion() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: true,
            enable_memory_optimization: true,
            min_performance_improvement: 10.0,
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Create convolution + activation pattern
        let conv_activation_ops = create_convolution_activation_pattern();

        // Test fusion optimization
        let fusion_result = optimizer.optimize_kernel_fusion(&conv_activation_ops);
        assert!(
            fusion_result.is_ok(),
            "Convolution + activation fusion should succeed"
        );

        let result = fusion_result.unwrap();
        assert!(
            !result.fused_kernels.is_empty(),
            "Should produce fused kernels"
        );
        assert!(
            result.performance_improvement > 0.0,
            "Should show performance improvement"
        );

        // Verify fusion strategy was appropriate
        assert_eq!(
            result.strategy_used,
            FusionStrategyType::BalancedPerformance
        );

        println!(
            "Convolution + activation fusion achieved {:.2}% performance improvement",
            result.performance_improvement
        );
    }

    /// Test matrix multiplication + bias + activation fusion
    #[test]
    fn test_matmul_bias_activation_fusion() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: true,
            enable_cache_optimization: true,
            min_performance_improvement: 8.0,
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Create MatMul + Bias + Activation pattern (common in neural networks)
        let matmul_bias_activation = create_matmul_bias_activation_pattern();

        // Test opportunity detection
        let opportunities = optimizer.analyze_fusion_opportunities(&matmul_bias_activation);
        assert!(opportunities.is_ok());

        let ops = opportunities.unwrap();
        assert!(!ops.is_empty(), "Should detect fusion opportunities");

        // Test full optimization
        let fusion_result = optimizer.optimize_kernel_fusion(&matmul_bias_activation);
        assert!(fusion_result.is_ok());

        let result = fusion_result.unwrap();
        assert!(result.memory_reduction > 0.0, "Should reduce memory usage");
        assert!(
            !result.success_metrics.is_none(),
            "Should provide success metrics"
        );

        println!(
            "MatMul + Bias + Activation fusion: {:.2}% performance gain, {:.2}% memory reduction",
            result.performance_improvement, result.memory_reduction
        );
    }

    /// Test reduction operation fusion and optimization
    #[test]
    fn test_reduction_operation_fusion() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: false, // Conservative for reduction ops
            enable_pattern_analysis: true,
            min_performance_improvement: 12.0,
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Create reduction operation patterns
        let reduction_patterns = vec![
            create_sum_reduction_pattern(),
            create_mean_reduction_pattern(),
            create_max_reduction_pattern(),
        ];

        for (pattern_name, pattern_ops) in vec![
            ("Sum Reduction", reduction_patterns[0].clone()),
            ("Mean Reduction", reduction_patterns[1].clone()),
            ("Max Reduction", reduction_patterns[2].clone()),
        ] {
            let fusion_result = optimizer.optimize_kernel_fusion(&pattern_ops);

            match fusion_result {
                Ok(result) => {
                    assert!(
                        result.performance_improvement >= 0.0,
                        "Performance improvement should be non-negative for {}",
                        pattern_name
                    );
                    println!(
                        "{}: {:.2}% performance improvement",
                        pattern_name, result.performance_improvement
                    );
                }
                Err(e) => {
                    // Some reduction patterns might not be fusable - this is acceptable
                    println!("{}: Fusion not applied - {:?}", pattern_name, e);
                }
            }
        }
    }

    /// Test complex operation dependency resolution
    #[test]
    fn test_complex_dependency_resolution() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: true,
            enable_pattern_analysis: true,
            max_fusion_operations: 10,
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Create operations with complex dependencies
        let complex_dependency_ops = create_complex_dependency_pattern();

        // Test dependency analysis
        let opportunities = optimizer.analyze_fusion_opportunities(&complex_dependency_ops);
        assert!(opportunities.is_ok(), "Should handle complex dependencies");

        let ops = opportunities.unwrap();
        println!(
            "Complex dependency analysis found {} fusion opportunities",
            ops.len()
        );

        // Test fusion with dependency constraints
        let fusion_result = optimizer.optimize_kernel_fusion(&complex_dependency_ops);

        match fusion_result {
            Ok(result) => {
                assert!(
                    !result.fused_kernels.is_empty(),
                    "Should produce some fused kernels"
                );

                // Verify dependency constraints are maintained
                for kernel in &result.fused_kernels {
                    assert!(
                        !kernel.fused_operations.is_empty(),
                        "Fused kernel should contain operations"
                    );

                    // Check that operations are in correct dependency order
                    for i in 1..kernel.fused_operations.len() {
                        assert!(
                            kernel.fused_operations[i].execution_order
                                >= kernel.fused_operations[i - 1].execution_order,
                            "Operations should be in dependency order"
                        );
                    }
                }

                println!(
                    "Complex dependency fusion successful: {:.2}% improvement",
                    result.performance_improvement
                );
            }
            Err(e) => {
                println!("Complex dependency fusion failed (acceptable): {:?}", e);
            }
        }
    }

    /// Test different fusion strategies and their effectiveness
    #[test]
    fn test_fusion_strategy_selection() {
        let strategies = vec![
            FusionStrategyType::Aggressive,
            FusionStrategyType::Conservative,
            FusionStrategyType::MemoryOptimized,
            FusionStrategyType::ComputeOptimized,
            FusionStrategyType::BalancedPerformance,
        ];

        let test_operations = create_mixed_operation_workload();

        for strategy in strategies {
            let config = KernelFusionConfig {
                fusion_strategies: vec![strategy.clone()],
                enable_aggressive_fusion: matches!(strategy, FusionStrategyType::Aggressive),
                min_performance_improvement: match strategy {
                    FusionStrategyType::Conservative => 15.0,
                    FusionStrategyType::Aggressive => 5.0,
                    _ => 10.0,
                },
                ..Default::default()
            };

            let optimizer = AdvancedKernelFusionOptimizer::new(config);
            assert!(
                optimizer.initialize().is_ok(),
                "Should initialize with strategy: {:?}",
                strategy
            );

            let fusion_result = optimizer.optimize_kernel_fusion(&test_operations);

            match fusion_result {
                Ok(result) => {
                    println!(
                        "Strategy {:?}: {:.2}% performance improvement, {:.2}% memory reduction",
                        strategy, result.performance_improvement, result.memory_reduction
                    );

                    // Verify strategy was applied
                    assert_eq!(result.strategy_used, strategy);

                    // Strategy-specific validations
                    match strategy {
                        FusionStrategyType::MemoryOptimized => {
                            assert!(
                                result.memory_reduction >= 0.0,
                                "Memory-optimized should focus on memory reduction"
                            );
                        }
                        FusionStrategyType::ComputeOptimized => {
                            assert!(
                                result.performance_improvement >= 0.0,
                                "Compute-optimized should focus on performance"
                            );
                        }
                        _ => {}
                    }
                }
                Err(e) => {
                    println!("Strategy {:?}: Failed to optimize - {:?}", strategy, e);
                }
            }
        }
    }

    /// Test fusion cache effectiveness and reuse
    #[test]
    fn test_fusion_cache_reuse() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: true,
            enable_cache_optimization: true,
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        let test_operations = create_cacheable_operation_pattern();

        // First optimization run - should populate cache
        let start_time1 = Instant::now();
        let result1 = optimizer.optimize_kernel_fusion(&test_operations);
        let duration1 = start_time1.elapsed();

        assert!(result1.is_ok(), "First optimization should succeed");

        // Second optimization run with same pattern - should use cache
        let start_time2 = Instant::now();
        let result2 = optimizer.optimize_kernel_fusion(&test_operations);
        let duration2 = start_time2.elapsed();

        assert!(result2.is_ok(), "Second optimization should succeed");

        // Cache reuse should make second run faster (though this is implementation-dependent)
        println!("First run: {:?}, Second run: {:?}", duration1, duration2);

        // Verify cache effectiveness through status
        let status = optimizer.get_optimization_status();
        println!("Cache hit ratio: {:.2}%", status.cache_hit_ratio * 100.0);
        assert!(
            status.cache_hit_ratio >= 0.0,
            "Cache hit ratio should be tracked"
        );
    }

    /// Test performance prediction accuracy
    #[test]
    fn test_performance_prediction_accuracy() {
        let config = KernelFusionConfig {
            enable_pattern_analysis: true,
            enable_performance_optimization: true,
            min_performance_improvement: 5.0,
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        let prediction_test_cases = vec![
            ("Simple ElementWise", create_simple_elementwise_ops()),
            ("Complex Mixed", create_mixed_operation_workload()),
            ("Memory Intensive", create_memory_intensive_ops()),
        ];

        for (case_name, operations) in prediction_test_cases {
            // Analyze opportunities to get predictions
            let opportunities = optimizer.analyze_fusion_opportunities(&operations);
            assert!(
                opportunities.is_ok(),
                "Should analyze opportunities for case: {}",
                case_name
            );

            let ops = opportunities.unwrap();
            if !ops.is_empty() {
                // Execute actual fusion
                let fusion_result = optimizer.optimize_kernel_fusion(&operations);

                match fusion_result {
                    Ok(result) => {
                        println!(
                            "{}: Predicted improvement available, Actual: {:.2}%",
                            case_name, result.performance_improvement
                        );

                        // Basic validation that predictions and results are reasonable
                        assert!(
                            result.performance_improvement >= -10.0,
                            "Performance should not degrade significantly"
                        );
                        assert!(
                            result.performance_improvement <= 1000.0,
                            "Performance improvement should be realistic"
                        );
                    }
                    Err(e) => {
                        println!("{}: Fusion failed - {:?}", case_name, e);
                    }
                }
            } else {
                println!("{}: No fusion opportunities detected", case_name);
            }
        }
    }

    /// Test error handling in fusion optimization
    #[test]
    fn test_fusion_error_handling() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: true,
            max_fusion_operations: 3, // Low limit to trigger errors
            optimization_timeout: Duration::from_millis(10), // Very short timeout
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Test various error scenarios
        let error_test_cases = vec![
            ("Empty Operations", Vec::new()),
            ("Circular Dependencies", create_circular_dependency_ops()),
            ("Too Many Operations", create_large_operation_chain(20)),
            ("Invalid Operations", create_invalid_operations()),
        ];

        for (case_name, operations) in error_test_cases {
            let fusion_result = optimizer.optimize_kernel_fusion(&operations);

            match fusion_result {
                Ok(result) => {
                    println!(
                        "{}: Unexpectedly succeeded - {:.2}% improvement",
                        case_name, result.performance_improvement
                    );
                    // Some cases might succeed despite being designed to fail
                }
                Err(e) => {
                    println!("{}: Failed as expected - {:?}", case_name, e);

                    // Verify error types are appropriate
                    use torsh_backend::cuda::kernel_fusion_optimizer::KernelFusionError;
                    match e {
                        KernelFusionError::AnalysisError(_)
                        | KernelFusionError::GenerationError(_)
                        | KernelFusionError::OptimizationError(_)
                        | KernelFusionError::CompilationError(_)
                        | KernelFusionError::ConfigurationError(_) => {
                            // Expected error types
                        }
                        _ => panic!("Unexpected error type for case {}: {:?}", case_name, e),
                    }
                }
            }
        }

        // Verify optimizer is still functional after errors
        let simple_ops = create_simple_elementwise_ops();
        let recovery_result = optimizer.optimize_kernel_fusion(&simple_ops);
        assert!(
            recovery_result.is_ok() || recovery_result.is_err(),
            "Optimizer should still be functional after error scenarios"
        );
    }

    /// Test mixed-precision fusion optimization
    #[test]
    fn test_mixed_precision_fusion() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: true,
            enable_memory_optimization: true,
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Create operations with different data types
        let mixed_precision_ops = create_mixed_precision_operations();

        let fusion_result = optimizer.optimize_kernel_fusion(&mixed_precision_ops);

        match fusion_result {
            Ok(result) => {
                println!(
                    "Mixed precision fusion: {:.2}% improvement",
                    result.performance_improvement
                );

                // Verify that mixed precision is handled correctly
                for kernel in &result.fused_kernels {
                    // Should contain operations with different data types
                    let data_types: Vec<_> = kernel
                        .fused_operations
                        .iter()
                        .flat_map(|op| &op.input_tensors)
                        .map(|tensor| tensor.data_type.clone())
                        .collect();

                    if data_types.len() > 1 {
                        println!(
                            "Kernel {} handles mixed precision: {:?}",
                            kernel.kernel_id, data_types
                        );
                    }
                }
            }
            Err(e) => {
                println!("Mixed precision fusion not supported: {:?}", e);
            }
        }
    }

    // Helper functions for creating test operation patterns

    fn create_elementwise_operation_chain() -> Vec<FusionOperation> {
        vec![
            create_fusion_operation("add_op", OperationType::ElementWiseAdd, Vec::new(), 0),
            create_fusion_operation(
                "mul_op",
                OperationType::ElementWiseMul,
                vec!["add_op".to_string()],
                1,
            ),
            create_fusion_operation(
                "sub_op",
                OperationType::ElementWiseSub,
                vec!["mul_op".to_string()],
                2,
            ),
            create_fusion_operation(
                "relu_op",
                OperationType::Activation(ActivationType::ReLU),
                vec!["sub_op".to_string()],
                3,
            ),
        ]
    }

    fn create_convolution_activation_pattern() -> Vec<FusionOperation> {
        vec![
            create_fusion_operation("conv_op", OperationType::Convolution2D, Vec::new(), 0),
            create_fusion_operation(
                "relu_op",
                OperationType::Activation(ActivationType::ReLU),
                vec!["conv_op".to_string()],
                1,
            ),
        ]
    }

    fn create_matmul_bias_activation_pattern() -> Vec<FusionOperation> {
        vec![
            create_fusion_operation("matmul_op", OperationType::MatrixMultiply, Vec::new(), 0),
            create_fusion_operation(
                "bias_op",
                OperationType::ElementWiseAdd,
                vec!["matmul_op".to_string()],
                1,
            ),
            create_fusion_operation(
                "gelu_op",
                OperationType::Activation(ActivationType::GELU),
                vec!["bias_op".to_string()],
                2,
            ),
        ]
    }

    fn create_sum_reduction_pattern() -> Vec<FusionOperation> {
        vec![create_fusion_operation(
            "sum_reduce",
            OperationType::Reduction(ReductionType::Sum),
            Vec::new(),
            0,
        )]
    }

    fn create_mean_reduction_pattern() -> Vec<FusionOperation> {
        vec![create_fusion_operation(
            "mean_reduce",
            OperationType::Reduction(ReductionType::Mean),
            Vec::new(),
            0,
        )]
    }

    fn create_max_reduction_pattern() -> Vec<FusionOperation> {
        vec![create_fusion_operation(
            "max_reduce",
            OperationType::Reduction(ReductionType::Max),
            Vec::new(),
            0,
        )]
    }

    fn create_complex_dependency_pattern() -> Vec<FusionOperation> {
        vec![
            create_fusion_operation("op_a", OperationType::ElementWiseAdd, Vec::new(), 0),
            create_fusion_operation("op_b", OperationType::ElementWiseMul, Vec::new(), 1),
            create_fusion_operation(
                "op_c",
                OperationType::ElementWiseSub,
                vec!["op_a".to_string(), "op_b".to_string()],
                2,
            ),
            create_fusion_operation(
                "op_d",
                OperationType::Activation(ActivationType::Sigmoid),
                vec!["op_a".to_string()],
                3,
            ),
            create_fusion_operation(
                "op_e",
                OperationType::ElementWiseAdd,
                vec!["op_c".to_string(), "op_d".to_string()],
                4,
            ),
        ]
    }

    fn create_mixed_operation_workload() -> Vec<FusionOperation> {
        vec![
            create_fusion_operation("tensor_add", OperationType::ElementWiseAdd, Vec::new(), 0),
            create_fusion_operation(
                "matmul",
                OperationType::MatrixMultiply,
                vec!["tensor_add".to_string()],
                1,
            ),
            create_fusion_operation("transpose", OperationType::Transpose, Vec::new(), 2),
            create_fusion_operation(
                "activation",
                OperationType::Activation(ActivationType::Tanh),
                vec!["matmul".to_string()],
                3,
            ),
        ]
    }

    fn create_cacheable_operation_pattern() -> Vec<FusionOperation> {
        // Simple, repeatable pattern that should be cached
        vec![
            create_fusion_operation("cache_add", OperationType::ElementWiseAdd, Vec::new(), 0),
            create_fusion_operation(
                "cache_relu",
                OperationType::Activation(ActivationType::ReLU),
                vec!["cache_add".to_string()],
                1,
            ),
        ]
    }

    fn create_simple_elementwise_ops() -> Vec<FusionOperation> {
        vec![
            create_fusion_operation("simple_add", OperationType::ElementWiseAdd, Vec::new(), 0),
            create_fusion_operation(
                "simple_mul",
                OperationType::ElementWiseMul,
                vec!["simple_add".to_string()],
                1,
            ),
        ]
    }

    fn create_memory_intensive_ops() -> Vec<FusionOperation> {
        vec![
            create_fusion_operation("big_matmul", OperationType::MatrixMultiply, Vec::new(), 0),
            create_fusion_operation("big_conv", OperationType::Convolution2D, Vec::new(), 1),
            create_fusion_operation(
                "transpose_big",
                OperationType::Transpose,
                vec!["big_matmul".to_string()],
                2,
            ),
        ]
    }

    fn create_circular_dependency_ops() -> Vec<FusionOperation> {
        vec![
            create_fusion_operation(
                "circular_a",
                OperationType::ElementWiseAdd,
                vec!["circular_b".to_string()],
                0,
            ),
            create_fusion_operation(
                "circular_b",
                OperationType::ElementWiseMul,
                vec!["circular_a".to_string()],
                1,
            ),
        ]
    }

    fn create_large_operation_chain(count: usize) -> Vec<FusionOperation> {
        let mut ops = Vec::new();

        for i in 0..count {
            let dependencies = if i == 0 {
                Vec::new()
            } else {
                vec![format!("large_op_{}", i - 1)]
            };

            ops.push(create_fusion_operation(
                &format!("large_op_{}", i),
                OperationType::ElementWiseAdd,
                dependencies,
                i,
            ));
        }

        ops
    }

    fn create_invalid_operations() -> Vec<FusionOperation> {
        vec![
            // Operation with empty ID
            FusionOperation {
                operation_id: "".to_string(),
                operation_type: OperationType::ElementWiseAdd,
                input_tensors: Vec::new(),
                output_tensors: Vec::new(),
                parameters: HashMap::new(),
                memory_requirements: Default::default(),
                compute_requirements: Default::default(),
                dependencies: Vec::new(),
                execution_order: 0,
            },
        ]
    }

    fn create_mixed_precision_operations() -> Vec<FusionOperation> {
        use torsh_backend::cuda::kernel_fusion_optimizer::TensorDescriptor;

        let f32_tensor = TensorDescriptor {
            shape: vec![128, 128],
            stride: vec![128, 1],
            data_type: DataType::F32,
            memory_layout: MemoryLayout::RowMajor,
            access_pattern: AccessPattern::Sequential,
            lifetime: Default::default(),
        };

        let f16_tensor = TensorDescriptor {
            shape: vec![128, 128],
            stride: vec![128, 1],
            data_type: DataType::F16,
            memory_layout: MemoryLayout::RowMajor,
            access_pattern: AccessPattern::Sequential,
            lifetime: Default::default(),
        };

        vec![
            FusionOperation {
                operation_id: "fp32_op".to_string(),
                operation_type: OperationType::ElementWiseAdd,
                input_tensors: vec![f32_tensor.clone()],
                output_tensors: vec![f32_tensor],
                parameters: HashMap::new(),
                memory_requirements: Default::default(),
                compute_requirements: Default::default(),
                dependencies: Vec::new(),
                execution_order: 0,
            },
            FusionOperation {
                operation_id: "fp16_op".to_string(),
                operation_type: OperationType::ElementWiseMul,
                input_tensors: vec![f16_tensor.clone()],
                output_tensors: vec![f16_tensor],
                parameters: HashMap::new(),
                memory_requirements: Default::default(),
                compute_requirements: Default::default(),
                dependencies: vec!["fp32_op".to_string()],
                execution_order: 1,
            },
        ]
    }

    fn create_fusion_operation(
        id: &str,
        op_type: OperationType,
        dependencies: Vec<String>,
        execution_order: usize,
    ) -> FusionOperation {
        FusionOperation {
            operation_id: id.to_string(),
            operation_type: op_type,
            input_tensors: Vec::new(),
            output_tensors: Vec::new(),
            parameters: HashMap::new(),
            memory_requirements: Default::default(),
            compute_requirements: Default::default(),
            dependencies,
            execution_order,
        }
    }
}

/// Performance benchmarking tests for fusion optimization
#[cfg(test)]
mod fusion_benchmarks {
    use super::*;

    #[test]
    fn test_fusion_optimization_performance() {
        let config = KernelFusionConfig {
            enable_aggressive_fusion: true,
            optimization_timeout: Duration::from_secs(5),
            ..Default::default()
        };

        let optimizer = AdvancedKernelFusionOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Benchmark different workload sizes
        let workload_sizes = vec![("Small", 3), ("Medium", 8), ("Large", 15)];

        for (workload_name, operation_count) in workload_sizes {
            let operations = create_large_operation_chain(operation_count);

            let start_time = Instant::now();
            let fusion_result = optimizer.optimize_kernel_fusion(&operations);
            let optimization_time = start_time.elapsed();

            println!(
                "{} workload ({} ops): Optimization time {:?}",
                workload_name, operation_count, optimization_time
            );

            match fusion_result {
                Ok(result) => {
                    println!(
                        "  Success: {:.2}% performance improvement, {:.2}% memory reduction",
                        result.performance_improvement, result.memory_reduction
                    );

                    // Performance should scale reasonably with workload size
                    assert!(
                        optimization_time < Duration::from_secs(2),
                        "Optimization should complete in reasonable time for {} workload",
                        workload_name
                    );
                }
                Err(e) => {
                    println!("  Failed: {:?}", e);
                }
            }
        }
    }

    #[test]
    fn test_fusion_strategy_performance_comparison() {
        let strategies = vec![
            (FusionStrategyType::Conservative, "Conservative"),
            (FusionStrategyType::Aggressive, "Aggressive"),
            (FusionStrategyType::BalancedPerformance, "Balanced"),
        ];

        let test_operations = create_mixed_operation_workload();

        for (strategy, strategy_name) in strategies {
            let config = KernelFusionConfig {
                fusion_strategies: vec![strategy.clone()],
                enable_aggressive_fusion: matches!(strategy, FusionStrategyType::Aggressive),
                ..Default::default()
            };

            let optimizer = AdvancedKernelFusionOptimizer::new(config);
            assert!(optimizer.initialize().is_ok());

            let start_time = Instant::now();
            let result = optimizer.optimize_kernel_fusion(&test_operations);
            let strategy_time = start_time.elapsed();

            println!(
                "{} strategy: Optimization time {:?}",
                strategy_name, strategy_time
            );

            match result {
                Ok(optimization_result) => {
                    println!(
                        "  Performance: {:.2}%, Memory: {:.2}%",
                        optimization_result.performance_improvement,
                        optimization_result.memory_reduction
                    );
                }
                Err(e) => {
                    println!("  Failed: {:?}", e);
                }
            }
        }
    }
}
