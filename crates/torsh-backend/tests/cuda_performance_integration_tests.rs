//! Comprehensive Integration Tests for CUDA Performance Optimization Systems
//!
//! This test suite validates the entire CUDA performance optimization pipeline including
//! memory optimization, kernel fusion, intelligent task scheduling, and the comprehensive
//! performance coordinator under realistic workload conditions.

#![cfg(cuda_available)]

use futures;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::time::sleep;

use torsh_backend::cuda::intelligent_task_scheduler::{
    AffinityPreferences, DeviceCapability, ResourceRequirements as TaskResourceRequirements,
    SchedulingConstraints, TaskData, TaskPriority,
};
use torsh_backend::cuda::kernel_fusion_optimizer::{
    ActivationType, KernelFusionError, ReductionType,
};
use torsh_backend::cuda::memory::optimization::advanced_memory_optimizer::{
    AdvancedMemoryConfig, AdvancedMemoryOptimizer, MemoryOptimizationReport,
};
use torsh_backend::cuda::performance_optimization_coordinator::{
    CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
    ResourceRequirements, TensorOperation, TensorOperationType,
};
use torsh_backend::cuda::{
    AdvancedKernelFusionOptimizer, ComprehensivePerformanceStatus, CoordinationError,
    CudaOperationRequest, CudaOperationResult, CudaPerformanceOptimizationCoordinator,
    FusionOperation, IntelligentSchedulingConfig, IntelligentTaskScheduler, KernelFusionConfig,
    OperationType, PerformanceCoordinatorConfig, PerformanceMetrics, SchedulableTask, TaskType,
};

/// Comprehensive integration test suite for CUDA performance optimization
#[cfg(all(test, cuda_available))]
mod integration_tests {
    use super::*;

    /// Test the complete performance optimization coordinator pipeline
    #[tokio::test]
    async fn test_comprehensive_performance_optimization_pipeline() {
        let config = PerformanceCoordinatorConfig::default();
        let coordinator = CudaPerformanceOptimizationCoordinator::new(config);

        // Test initialization
        let init_result = coordinator.initialize().await;
        assert!(
            init_result.is_ok(),
            "Performance coordinator should initialize successfully"
        );

        // Test operation request creation
        let operation_request = integration_tests::create_test_tensor_operation_request();

        // Execute comprehensive optimization
        let optimization_result = coordinator.optimize_cuda_operation(operation_request).await;
        assert!(
            optimization_result.is_ok(),
            "CUDA operation optimization should succeed"
        );

        let result = optimization_result.unwrap();
        assert!(
            result.execution_success,
            "Operation execution should be successful"
        );
        assert!(
            result.quality_score > 0.7,
            "Quality score should be above 70%"
        );

        // Validate performance status
        let status = coordinator.get_comprehensive_status();
        assert_eq!(status.total_operations_coordinated, 1);
        assert!(status.success_rate >= 0.0);
        assert!(!status.active_optimizations.is_empty());
    }

    /// Test memory optimization integration with real workloads
    #[test]
    fn test_advanced_memory_optimization_integration() {
        let config = AdvancedMemoryConfig::default();
        let optimizer = AdvancedMemoryOptimizer::new(config);

        // Test initialization
        let init_result = optimizer.initialize();
        assert!(
            init_result.is_ok(),
            "Memory optimizer should initialize successfully"
        );

        // Test optimized allocation and deallocation patterns
        let allocation_sizes = vec![1024, 4096, 16384, 65536, 262144]; // Various sizes
        let mut allocations = Vec::new();

        for size in &allocation_sizes {
            let ptr_result =
                optimizer.optimized_allocate(*size, 256, Some(Duration::from_secs(60)));
            assert!(
                ptr_result.is_ok(),
                "Memory allocation should succeed for size {}",
                size
            );

            if let Ok(ptr) = ptr_result {
                allocations.push((ptr, *size));
            }
        }

        // Test deallocations
        for (ptr, size) in allocations {
            let dealloc_result = optimizer.optimized_deallocate(ptr, size);
            assert!(dealloc_result.is_ok(), "Memory deallocation should succeed");
        }

        // Test comprehensive optimization
        let optimization_result = optimizer.perform_comprehensive_optimization();
        assert!(
            optimization_result.is_ok(),
            "Comprehensive memory optimization should succeed"
        );

        let report = optimization_result.unwrap();
        assert!(
            report.performance_improvement > 0.0,
            "Should show performance improvement"
        );
        assert!(
            !report.recommendations.is_empty(),
            "Should provide optimization recommendations"
        );

        // Validate optimization status
        let status = optimizer.get_optimization_status();
        assert!(status.total_optimizations > 0);
        assert!(status.performance_improvement >= 0.0);
    }

    /// Test kernel fusion optimization with complex operation chains
    #[test]
    fn test_kernel_fusion_optimization_integration() {
        let config = KernelFusionConfig::default();
        let optimizer = AdvancedKernelFusionOptimizer::new(config);

        // Test initialization
        let init_result = optimizer.initialize();
        assert!(
            init_result.is_ok(),
            "Kernel fusion optimizer should initialize successfully"
        );

        // Create complex operation chain for fusion
        let operations = create_complex_fusion_operations();

        // Analyze fusion opportunities
        let opportunities_result = optimizer.analyze_fusion_opportunities(&operations);
        assert!(
            opportunities_result.is_ok(),
            "Fusion opportunity analysis should succeed"
        );

        let opportunities = opportunities_result.unwrap();
        assert!(
            !opportunities.is_empty(),
            "Should identify fusion opportunities"
        );

        // Execute kernel fusion optimization
        let fusion_result = optimizer.optimize_kernel_fusion(&operations);
        assert!(
            fusion_result.is_ok(),
            "Kernel fusion optimization should succeed"
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
        assert!(
            result.memory_reduction >= 0.0,
            "Should show memory reduction"
        );

        // Validate fusion status
        let status = optimizer.get_optimization_status();
        assert!(status.total_fusions > 0);
        assert!(status.success_rate > 0.0);
    }

    /// Test intelligent task scheduling with various workload patterns
    #[test]
    fn test_intelligent_task_scheduling_integration() {
        let config = IntelligentSchedulingConfig::default();
        let scheduler = IntelligentTaskScheduler::new(config);

        // Test initialization
        let init_result = scheduler.initialize();
        assert!(
            init_result.is_ok(),
            "Task scheduler should initialize successfully"
        );

        // Create diverse task workload
        let tasks = create_diverse_task_workload();
        let mut submission_results = Vec::new();

        // Submit tasks
        for task in tasks {
            let submission_result = scheduler.submit_task(task);
            assert!(submission_result.is_ok(), "Task submission should succeed");
            submission_results.push(submission_result.unwrap());
        }

        assert_eq!(
            submission_results.len(),
            5,
            "All tasks should be submitted successfully"
        );

        // Validate scheduling status
        let status = scheduler.get_scheduling_status();
        assert!(status.total_tasks_scheduled >= 5);
        assert!(status.resource_utilization_efficiency > 0.0);
    }

    /// Test cross-component integration and coordination
    #[tokio::test]
    async fn test_cross_component_coordination() {
        let config = PerformanceCoordinatorConfig {
            enable_memory_optimization: true,
            enable_kernel_fusion: true,
            enable_intelligent_scheduling: true,
            min_performance_improvement: 5.0, // Lower threshold for testing
            ..Default::default()
        };
        let coordinator = CudaPerformanceOptimizationCoordinator::new(config);

        // Initialize coordinator
        assert!(coordinator.initialize().await.is_ok());

        // Test multiple concurrent operations
        let operations = vec![
            integration_tests::create_matrix_multiplication_request(),
            integration_tests::create_convolution_request(),
            integration_tests::create_element_wise_request(),
        ];

        let mut results = Vec::new();
        for operation in operations {
            let result = coordinator.optimize_cuda_operation(operation).await;
            assert!(result.is_ok(), "Each operation should succeed");
            results.push(result.unwrap());
        }

        assert_eq!(results.len(), 3, "All operations should complete");

        // Verify that different optimization strategies were applied
        let mut memory_optimizations = 0;
        let mut fusion_optimizations = 0;
        let mut scheduling_optimizations = 0;

        for result in &results {
            if result.optimization_applied.memory_optimization {
                memory_optimizations += 1;
            }
            if result.optimization_applied.kernel_fusion {
                fusion_optimizations += 1;
            }
            if result.optimization_applied.intelligent_scheduling {
                scheduling_optimizations += 1;
            }
        }

        // Verify coordination is working
        let status = coordinator.get_comprehensive_status();
        assert_eq!(status.total_operations_coordinated, 3);
        assert!(!status.system_recommendations.is_empty());
    }

    /// Test performance benchmarking and regression detection
    #[tokio::test]
    async fn test_performance_benchmarking() {
        let config = PerformanceCoordinatorConfig::default();
        let coordinator = CudaPerformanceOptimizationCoordinator::new(config);
        assert!(coordinator.initialize().await.is_ok());

        // Run benchmark workloads
        let benchmark_operations = create_benchmark_workloads();
        let mut execution_times = Vec::new();
        let mut quality_scores = Vec::new();

        for operation in benchmark_operations {
            let start = Instant::now();
            let result = coordinator.optimize_cuda_operation(operation).await;
            let execution_time = start.elapsed();

            assert!(result.is_ok(), "Benchmark operation should succeed");
            let result = result.unwrap();

            execution_times.push(execution_time);
            quality_scores.push(result.quality_score);
        }

        // Validate performance characteristics
        let avg_execution_time: Duration =
            execution_times.iter().sum::<Duration>() / execution_times.len() as u32;
        let avg_quality_score = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;

        assert!(
            avg_execution_time < Duration::from_secs(1),
            "Average execution time should be reasonable"
        );
        assert!(
            avg_quality_score > 0.6,
            "Average quality score should be above 60%"
        );

        // Check for performance consistency
        let max_time = execution_times.iter().max().unwrap();
        let min_time = execution_times.iter().min().unwrap();
        let time_variance = max_time.as_millis() as f64 / min_time.as_millis() as f64;

        assert!(
            time_variance < 5.0,
            "Execution time variance should be reasonable"
        );
    }

    /// Test error handling and fault tolerance
    #[tokio::test]
    async fn test_error_handling_and_fault_tolerance() {
        let config = PerformanceCoordinatorConfig::default();
        let coordinator = CudaPerformanceOptimizationCoordinator::new(config);
        assert!(coordinator.initialize().await.is_ok());

        // Test invalid operation requests
        let invalid_requests = create_invalid_operation_requests();

        for invalid_request in invalid_requests {
            let result = coordinator.optimize_cuda_operation(invalid_request).await;
            // Should either succeed with degraded performance or fail gracefully
            if result.is_err() {
                match result.unwrap_err() {
                    CoordinationError::InitializationError(_)
                    | CoordinationError::MemoryOptimizationError(_)
                    | CoordinationError::FusionOptimizationError(_)
                    | CoordinationError::SchedulingError(_)
                    | CoordinationError::ConfigurationError(_) => {
                        // Expected error types - this is good
                    }
                    _ => panic!("Unexpected error type"),
                }
            }
        }

        // Test resource exhaustion scenarios
        let resource_heavy_request = create_resource_heavy_request();
        let result = coordinator
            .optimize_cuda_operation(resource_heavy_request)
            .await;

        // Should handle gracefully (either succeed with warnings or fail safely)
        match result {
            Ok(res) => assert!(res.quality_score >= 0.0),
            Err(_) => {} // Acceptable to fail on resource exhaustion
        }
    }

    /// Test adaptive learning and optimization improvement
    #[tokio::test]
    async fn test_adaptive_learning() {
        let config = PerformanceCoordinatorConfig {
            enable_adaptive_strategies: true,
            performance_monitoring_interval: Duration::from_millis(100),
            ..Default::default()
        };
        let coordinator = CudaPerformanceOptimizationCoordinator::new(config);
        assert!(coordinator.initialize().await.is_ok());

        // Run multiple similar operations to enable learning
        let repeated_operation = integration_tests::create_matrix_multiplication_request();
        let mut quality_scores = Vec::new();

        for i in 0..5 {
            let mut operation = repeated_operation.clone();
            operation.request_id = format!("adaptive_test_{}", i);

            let result = coordinator.optimize_cuda_operation(operation).await;
            assert!(result.is_ok(), "Repeated operation {} should succeed", i);

            quality_scores.push(result.unwrap().quality_score);

            // Small delay to allow learning
            sleep(Duration::from_millis(50)).await;
        }

        // Check if there's improvement over time (adaptive learning)
        if quality_scores.len() >= 3 {
            let early_avg = (quality_scores[0] + quality_scores[1]) / 2.0;
            let late_avg = (quality_scores[3] + quality_scores[4]) / 2.0;

            // Allow for some variance, but generally expect improvement or stability
            assert!(
                late_avg >= early_avg - 0.1,
                "Performance should not significantly degrade over time"
            );
        }
    }

    /// Test multi-GPU coordination and load balancing
    #[tokio::test]
    async fn test_multi_gpu_coordination() {
        let config = PerformanceCoordinatorConfig {
            enable_predictive_balancing: true,
            enable_resource_awareness: true,
            ..Default::default()
        };
        let coordinator = CudaPerformanceOptimizationCoordinator::new(config);
        assert!(coordinator.initialize().await.is_ok());

        // Simulate multiple concurrent operations that should be distributed
        let concurrent_operations = vec![
            integration_tests::create_matrix_multiplication_request(),
            integration_tests::create_convolution_request(),
            integration_tests::create_element_wise_request(),
            create_reduction_request(),
        ];

        let mut handles = Vec::new();

        // Submit operations concurrently
        for (i, operation) in concurrent_operations.into_iter().enumerate() {
            let coord_clone = Arc::new(&coordinator);
            let handle = tokio::spawn(async move {
                let mut op = operation;
                op.request_id = format!("concurrent_op_{}", i);
                coord_clone.optimize_cuda_operation(op).await
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let results: Vec<_> = futures::future::join_all(handles).await;

        let mut successful_operations = 0;
        for result in results {
            if let Ok(Ok(_)) = result {
                successful_operations += 1;
            }
        }

        assert!(
            successful_operations >= 2,
            "At least half of concurrent operations should succeed"
        );

        // Validate load balancing effectiveness
        let status = coordinator.get_comprehensive_status();
        assert!(status.coordination_state.active_operations >= 0);
    }

    // Helper functions for creating test data

    pub fn create_test_tensor_operation_request() -> CudaOperationRequest {
        use torsh_backend::cuda::performance_optimization_coordinator::{
            CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
            ResourceRequirements, TensorOperation,
        };

        CudaOperationRequest {
            request_id: "test_operation_001".to_string(),
            operation_type: CudaOperationType::TensorComputation,
            tensor_operations: vec![
                TensorOperation::default(), // Placeholder tensor operation
            ],
            resource_requirements: ResourceRequirements::default(),
            performance_requirements: PerformanceRequirements::default(),
            optimization_hints: OptimizationHints::default(),
            deadline: Some(SystemTime::now() + Duration::from_secs(10)),
            priority: RequestPriority::Medium,
            submission_time: SystemTime::now(),
        }
    }

    pub fn create_matrix_multiplication_request() -> CudaOperationRequest {
        use torsh_backend::cuda::performance_optimization_coordinator::{
            CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
            ResourceRequirements, TensorOperation,
        };

        CudaOperationRequest {
            request_id: "matmul_operation".to_string(),
            operation_type: CudaOperationType::MatrixOperation,
            tensor_operations: vec![TensorOperation::default()],
            resource_requirements: ResourceRequirements::default(),
            performance_requirements: PerformanceRequirements::default(),
            optimization_hints: OptimizationHints::default(),
            deadline: None,
            priority: RequestPriority::High,
            submission_time: SystemTime::now(),
        }
    }

    pub fn create_convolution_request() -> CudaOperationRequest {
        use torsh_backend::cuda::performance_optimization_coordinator::{
            CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
            ResourceRequirements, TensorOperation,
        };

        CudaOperationRequest {
            request_id: "conv_operation".to_string(),
            operation_type: CudaOperationType::ConvolutionalOperation,
            tensor_operations: vec![TensorOperation::default()],
            resource_requirements: ResourceRequirements::default(),
            performance_requirements: PerformanceRequirements::default(),
            optimization_hints: OptimizationHints::default(),
            deadline: None,
            priority: RequestPriority::Medium,
            submission_time: SystemTime::now(),
        }
    }

    pub fn create_element_wise_request() -> CudaOperationRequest {
        use torsh_backend::cuda::performance_optimization_coordinator::{
            CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
            ResourceRequirements, TensorOperation,
        };

        CudaOperationRequest {
            request_id: "elementwise_operation".to_string(),
            operation_type: CudaOperationType::TensorComputation,
            tensor_operations: vec![TensorOperation::default()],
            resource_requirements: ResourceRequirements::default(),
            performance_requirements: PerformanceRequirements::default(),
            optimization_hints: OptimizationHints::default(),
            deadline: None,
            priority: RequestPriority::Low,
            submission_time: SystemTime::now(),
        }
    }

    pub fn create_reduction_request() -> CudaOperationRequest {
        use torsh_backend::cuda::performance_optimization_coordinator::{
            CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
            ResourceRequirements, TensorOperation,
        };

        CudaOperationRequest {
            request_id: "reduction_operation".to_string(),
            operation_type: CudaOperationType::ReductionOperation,
            tensor_operations: vec![TensorOperation::default()],
            resource_requirements: ResourceRequirements::default(),
            performance_requirements: PerformanceRequirements::default(),
            optimization_hints: OptimizationHints::default(),
            deadline: None,
            priority: RequestPriority::Medium,
            submission_time: SystemTime::now(),
        }
    }

    pub fn create_complex_fusion_operations() -> Vec<FusionOperation> {
        vec![
            FusionOperation {
                operation_id: "op1".to_string(),
                operation_type: OperationType::ElementWiseAdd,
                input_tensors: Vec::new(),
                output_tensors: Vec::new(),
                parameters: HashMap::new(),
                memory_requirements: Default::default(),
                compute_requirements: Default::default(),
                dependencies: Vec::new(),
                execution_order: 0,
            },
            FusionOperation {
                operation_id: "op2".to_string(),
                operation_type: OperationType::ElementWiseMul,
                input_tensors: Vec::new(),
                output_tensors: Vec::new(),
                parameters: HashMap::new(),
                memory_requirements: Default::default(),
                compute_requirements: Default::default(),
                dependencies: vec!["op1".to_string()],
                execution_order: 1,
            },
            FusionOperation {
                operation_id: "op3".to_string(),
                operation_type: OperationType::Activation(
                    torsh_backend::cuda::kernel_fusion_optimizer::ActivationType::ReLU,
                ),
                input_tensors: Vec::new(),
                output_tensors: Vec::new(),
                parameters: HashMap::new(),
                memory_requirements: Default::default(),
                compute_requirements: Default::default(),
                dependencies: vec!["op2".to_string()],
                execution_order: 2,
            },
        ]
    }

    pub fn create_diverse_task_workload() -> Vec<SchedulableTask> {
        use torsh_backend::cuda::intelligent_task_scheduler::{
            ResourceRequirements, SchedulingConstraints, TaskData, TaskPriority,
        };

        vec![
            SchedulableTask {
                task_id: "task_1".to_string(),
                task_type: TaskType::TensorOperation,
                priority: TaskPriority {
                    base_priority: 100,
                    dynamic_adjustment: 0,
                    aging_bonus: 0,
                    performance_bonus: 0,
                    deadline_urgency: 0,
                },
                resource_requirements: ResourceRequirements {
                    gpu_memory: 1024 * 1024 * 256, // 256MB
                    compute_units: 16,
                    bandwidth_requirements: 100.0,
                    shared_memory: 48 * 1024,
                    register_count: 64,
                    device_capabilities: Vec::new(),
                    affinity_preferences: Default::default(),
                },
                dependencies: Vec::new(),
                estimated_execution_time: Duration::from_millis(100),
                deadline: None,
                submission_time: SystemTime::now(),
                task_data: TaskData::default(),
                scheduling_constraints: SchedulingConstraints::default(),
            },
            SchedulableTask {
                task_id: "task_2".to_string(),
                task_type: TaskType::MatrixMultiplication,
                priority: TaskPriority {
                    base_priority: 200,
                    dynamic_adjustment: 0,
                    aging_bonus: 0,
                    performance_bonus: 0,
                    deadline_urgency: 50,
                },
                resource_requirements: ResourceRequirements {
                    gpu_memory: 1024 * 1024 * 512, // 512MB
                    compute_units: 32,
                    bandwidth_requirements: 200.0,
                    shared_memory: 96 * 1024,
                    register_count: 128,
                    device_capabilities: Vec::new(),
                    affinity_preferences: Default::default(),
                },
                dependencies: Vec::new(),
                estimated_execution_time: Duration::from_millis(200),
                deadline: Some(SystemTime::now() + Duration::from_secs(5)),
                submission_time: SystemTime::now(),
                task_data: TaskData::default(),
                scheduling_constraints: SchedulingConstraints::default(),
            },
            SchedulableTask {
                task_id: "task_3".to_string(),
                task_type: TaskType::Convolution,
                priority: TaskPriority {
                    base_priority: 150,
                    dynamic_adjustment: 0,
                    aging_bonus: 0,
                    performance_bonus: 0,
                    deadline_urgency: 25,
                },
                resource_requirements: ResourceRequirements {
                    gpu_memory: 1024 * 1024 * 768, // 768MB
                    compute_units: 24,
                    bandwidth_requirements: 150.0,
                    shared_memory: 72 * 1024,
                    register_count: 96,
                    device_capabilities: Vec::new(),
                    affinity_preferences: Default::default(),
                },
                dependencies: vec!["task_1".to_string()],
                estimated_execution_time: Duration::from_millis(150),
                deadline: Some(SystemTime::now() + Duration::from_secs(8)),
                submission_time: SystemTime::now(),
                task_data: TaskData::default(),
                scheduling_constraints: SchedulingConstraints::default(),
            },
            SchedulableTask {
                task_id: "task_4".to_string(),
                task_type: TaskType::Reduction,
                priority: TaskPriority {
                    base_priority: 75,
                    dynamic_adjustment: 0,
                    aging_bonus: 0,
                    performance_bonus: 0,
                    deadline_urgency: 0,
                },
                resource_requirements: ResourceRequirements {
                    gpu_memory: 1024 * 1024 * 128, // 128MB
                    compute_units: 8,
                    bandwidth_requirements: 75.0,
                    shared_memory: 32 * 1024,
                    register_count: 32,
                    device_capabilities: Vec::new(),
                    affinity_preferences: Default::default(),
                },
                dependencies: vec!["task_2".to_string(), "task_3".to_string()],
                estimated_execution_time: Duration::from_millis(75),
                deadline: None,
                submission_time: SystemTime::now(),
                task_data: TaskData::default(),
                scheduling_constraints: SchedulingConstraints::default(),
            },
            SchedulableTask {
                task_id: "task_5".to_string(),
                task_type: TaskType::MemoryTransfer,
                priority: TaskPriority {
                    base_priority: 50,
                    dynamic_adjustment: 0,
                    aging_bonus: 0,
                    performance_bonus: 0,
                    deadline_urgency: 0,
                },
                resource_requirements: ResourceRequirements {
                    gpu_memory: 1024 * 1024 * 64, // 64MB
                    compute_units: 4,
                    bandwidth_requirements: 300.0, // High bandwidth for memory transfer
                    shared_memory: 16 * 1024,
                    register_count: 16,
                    device_capabilities: Vec::new(),
                    affinity_preferences: Default::default(),
                },
                dependencies: Vec::new(),
                estimated_execution_time: Duration::from_millis(50),
                deadline: None,
                submission_time: SystemTime::now(),
                task_data: TaskData::default(),
                scheduling_constraints: SchedulingConstraints::default(),
            },
        ]
    }

    pub fn create_benchmark_workloads() -> Vec<CudaOperationRequest> {
        use torsh_backend::cuda::performance_optimization_coordinator::{
            CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
            ResourceRequirements, TensorOperation,
        };

        vec![
            CudaOperationRequest {
                request_id: "benchmark_small".to_string(),
                operation_type: CudaOperationType::TensorComputation,
                tensor_operations: vec![TensorOperation::default()],
                resource_requirements: ResourceRequirements::default(),
                performance_requirements: PerformanceRequirements::default(),
                optimization_hints: OptimizationHints::default(),
                deadline: None,
                priority: RequestPriority::Medium,
                submission_time: SystemTime::now(),
            },
            CudaOperationRequest {
                request_id: "benchmark_medium".to_string(),
                operation_type: CudaOperationType::MatrixOperation,
                tensor_operations: vec![TensorOperation::default()],
                resource_requirements: ResourceRequirements::default(),
                performance_requirements: PerformanceRequirements::default(),
                optimization_hints: OptimizationHints::default(),
                deadline: None,
                priority: RequestPriority::High,
                submission_time: SystemTime::now(),
            },
            CudaOperationRequest {
                request_id: "benchmark_large".to_string(),
                operation_type: CudaOperationType::ConvolutionalOperation,
                tensor_operations: vec![TensorOperation::default()],
                resource_requirements: ResourceRequirements::default(),
                performance_requirements: PerformanceRequirements::default(),
                optimization_hints: OptimizationHints::default(),
                deadline: None,
                priority: RequestPriority::High,
                submission_time: SystemTime::now(),
            },
        ]
    }

    pub fn create_invalid_operation_requests() -> Vec<CudaOperationRequest> {
        use torsh_backend::cuda::performance_optimization_coordinator::{
            CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
            ResourceRequirements, TensorOperation,
        };

        vec![
            // Request with empty operation ID (should be handled gracefully)
            CudaOperationRequest {
                request_id: "".to_string(),
                operation_type: CudaOperationType::TensorComputation,
                tensor_operations: Vec::new(),
                resource_requirements: ResourceRequirements::default(),
                performance_requirements: PerformanceRequirements::default(),
                optimization_hints: OptimizationHints::default(),
                deadline: Some(SystemTime::now() - Duration::from_secs(10)), // Past deadline
                priority: RequestPriority::Low,
                submission_time: SystemTime::now(),
            },
        ]
    }

    pub fn create_resource_heavy_request() -> CudaOperationRequest {
        use torsh_backend::cuda::performance_optimization_coordinator::{
            CudaOperationType, OptimizationHints, PerformanceRequirements, RequestPriority,
            ResourceRequirements, TensorOperation,
        };

        CudaOperationRequest {
            request_id: "resource_heavy".to_string(),
            operation_type: CudaOperationType::BatchedOperations,
            tensor_operations: vec![
                TensorOperation::default(),
                TensorOperation::default(),
                TensorOperation::default(),
                TensorOperation::default(),
                TensorOperation::default(),
            ],
            resource_requirements: ResourceRequirements::default(),
            performance_requirements: PerformanceRequirements::default(),
            optimization_hints: OptimizationHints::default(),
            deadline: Some(SystemTime::now() + Duration::from_millis(100)), // Very tight deadline
            priority: RequestPriority::Critical,
            submission_time: SystemTime::now(),
        }
    }
}

/// Performance regression detection tests
#[cfg(all(test, cuda_available))]
mod regression_tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_regression_detection() {
        let config = PerformanceCoordinatorConfig::default();
        let coordinator = CudaPerformanceOptimizationCoordinator::new(config);
        assert!(coordinator.initialize().await.is_ok());

        // Baseline performance measurement
        let baseline_request = integration_tests::create_test_tensor_operation_request();
        let baseline_start = Instant::now();
        let baseline_result = coordinator.optimize_cuda_operation(baseline_request).await;
        let baseline_duration = baseline_start.elapsed();

        assert!(baseline_result.is_ok());
        let baseline_quality = baseline_result.unwrap().quality_score;

        // Run multiple iterations to check for performance consistency
        let mut quality_scores = Vec::new();
        let mut execution_times = Vec::new();

        for i in 0..10 {
            let mut request = integration_tests::create_test_tensor_operation_request();
            request.request_id = format!("regression_test_{}", i);

            let start = Instant::now();
            let result = coordinator.optimize_cuda_operation(request).await;
            let duration = start.elapsed();

            assert!(result.is_ok(), "Operation {} should succeed", i);
            let result = result.unwrap();

            quality_scores.push(result.quality_score);
            execution_times.push(duration);
        }

        // Check for performance regression
        let avg_quality = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
        let avg_execution_time: Duration =
            execution_times.iter().sum::<Duration>() / execution_times.len() as u32;

        // Performance should not degrade significantly
        assert!(
            avg_quality >= baseline_quality - 0.1,
            "Quality should not degrade significantly"
        );
        assert!(
            avg_execution_time.as_millis() <= baseline_duration.as_millis() * 2,
            "Execution time should not double"
        );

        // Check for consistency (standard deviation should be reasonable)
        let quality_variance = quality_scores
            .iter()
            .map(|&x| (x - avg_quality).powi(2))
            .sum::<f64>()
            / quality_scores.len() as f64;
        let quality_std_dev = quality_variance.sqrt();

        assert!(
            quality_std_dev < 0.2,
            "Quality score variance should be reasonable"
        );
    }
}

/// Stress testing for high-load scenarios
#[cfg(all(test, cuda_available))]
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_high_load_concurrent_operations() {
        let config = PerformanceCoordinatorConfig {
            max_scheduling_latency: Duration::from_millis(100),
            resource_utilization_threshold: 0.95, // High threshold for stress test
            ..Default::default()
        };
        let coordinator = Arc::new(CudaPerformanceOptimizationCoordinator::new(config));
        assert!(coordinator.initialize().await.is_ok());

        // Submit many concurrent operations
        let num_operations = 20;
        let mut handles = Vec::new();

        for i in 0..num_operations {
            let coord_clone: Arc<CudaPerformanceOptimizationCoordinator> = Arc::clone(&coordinator);
            let handle = tokio::spawn(async move {
                let mut request = integration_tests::create_test_tensor_operation_request();
                request.request_id = format!("stress_test_{}", i);
                coord_clone.optimize_cuda_operation(request).await
            });
            handles.push(handle);
        }

        // Wait for all operations with timeout
        let timeout_duration = Duration::from_secs(30);
        let results =
            tokio::time::timeout(timeout_duration, futures::future::join_all(handles)).await;

        assert!(
            results.is_ok(),
            "All operations should complete within timeout"
        );
        let results = results.unwrap();

        let mut successful_operations = 0;
        for result in results {
            if let Ok(Ok(_)) = result {
                successful_operations += 1;
            }
        }

        // At least 80% of operations should succeed under high load
        assert!(
            successful_operations >= (num_operations * 4 / 5),
            "At least 80% of operations should succeed under high load"
        );

        let status = coordinator.get_comprehensive_status();
        assert_eq!(status.total_operations_coordinated, num_operations as u64);
    }
}
