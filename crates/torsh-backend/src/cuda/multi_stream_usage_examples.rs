//! Usage examples for the advanced multi-stream execution system
//!
//! This module provides comprehensive examples of how to use the multi-stream
//! execution system for various deep learning workloads and scenarios.

#[cfg(feature = "cuda")]
mod examples {
    use crate::cuda::{
        intelligent_scheduler::{
            MemoryAccessPattern, SchedulingStrategy, SynchronizationRequirements,
            WorkloadCharacteristics,
        },
        multi_stream_orchestrator::{MultiStreamOrchestrator, OrchestratorConfig},
        stream_advanced::WorkloadType,
        CudaStream,
    };
    use crate::error::CudaResult;
    use std::time::Duration;

    /// Example: Training a neural network with multi-stream execution
    pub fn neural_network_training_example() -> CudaResult<()> {
        // Create orchestrator with optimized configuration for training
        let config = OrchestratorConfig {
            max_concurrent_operations: 16,
            graph_capture_threshold: Duration::from_millis(5), // Capture graphs for ops > 5ms
            enable_auto_optimization: true,
            ..Default::default()
        };

        let mut orchestrator = MultiStreamOrchestrator::new(config)?;

        // Define workload characteristics for different operations
        let conv_characteristics = WorkloadCharacteristics {
            workload_type: WorkloadType::Compute,
            estimated_compute_time: Duration::from_millis(15),
            estimated_memory_bandwidth: 500_000_000_000, // 500 GB/s
            parallel_potential: 0.9,
            memory_access_pattern: MemoryAccessPattern::Sequential,
            synchronization_requirements: SynchronizationRequirements {
                requires_barrier: false,
                dependencies: vec![],
                provides_outputs: vec!["conv_output".to_string()],
            },
        };

        let batch_norm_characteristics = WorkloadCharacteristics {
            workload_type: WorkloadType::Memory,
            estimated_compute_time: Duration::from_millis(3),
            estimated_memory_bandwidth: 200_000_000_000, // 200 GB/s
            parallel_potential: 0.6,
            memory_access_pattern: MemoryAccessPattern::Broadcast,
            synchronization_requirements: SynchronizationRequirements {
                requires_barrier: false,
                dependencies: vec!["conv_output".to_string()],
                provides_outputs: vec!["normalized_output".to_string()],
            },
        };

        let activation_characteristics = WorkloadCharacteristics {
            workload_type: WorkloadType::Compute,
            estimated_compute_time: Duration::from_millis(2),
            estimated_memory_bandwidth: 300_000_000_000, // 300 GB/s
            parallel_potential: 0.95,
            memory_access_pattern: MemoryAccessPattern::Sequential,
            synchronization_requirements: SynchronizationRequirements {
                requires_barrier: false,
                dependencies: vec!["normalized_output".to_string()],
                provides_outputs: vec!["activated_output".to_string()],
            },
        };

        // Execute forward pass operations
        let operations = vec![
            ("conv_forward".to_string(), conv_characteristics),
            ("batch_norm".to_string(), batch_norm_characteristics),
            ("activation".to_string(), activation_characteristics),
        ];

        let results = orchestrator.execute_batch(operations, |op_name| {
            Box::new(move |stream: &CudaStream| {
                match op_name {
                    "conv_forward" => {
                        // Simulate convolution kernel launch
                        simulate_kernel_launch(stream, "conv2d_kernel", Duration::from_millis(15))
                    }
                    "batch_norm" => {
                        // Simulate batch normalization
                        simulate_kernel_launch(
                            stream,
                            "batch_norm_kernel",
                            Duration::from_millis(3),
                        )
                    }
                    "activation" => {
                        // Simulate activation function
                        simulate_kernel_launch(stream, "relu_kernel", Duration::from_millis(2))
                    }
                    _ => Ok(()),
                }
            })
        })?;

        println!("Forward pass completed successfully!");
        println!("Total operations: {}", results.len());

        // Get performance metrics
        let metrics = orchestrator.get_metrics();
        println!(
            "Scheduler accuracy: {:.2}%",
            metrics.scheduler_metrics.prediction_accuracy * 100.0
        );
        println!(
            "Stream utilization: {:.2}%",
            metrics.scheduler_metrics.stream_utilization * 100.0
        );

        Ok(())
    }

    /// Example: Matrix multiplication with different strategies
    pub fn matrix_multiplication_comparison() -> CudaResult<()> {
        let mut orchestrators = vec![
            ("latency_optimized", SchedulingStrategy::MinimizeLatency),
            (
                "throughput_optimized",
                SchedulingStrategy::MaximizeThroughput,
            ),
            ("balanced", SchedulingStrategy::Balanced),
            ("load_balanced", SchedulingStrategy::LoadBalance),
        ];

        for (name, strategy) in orchestrators.iter() {
            println!("\n=== Testing {} strategy ===", name);

            let config = OrchestratorConfig {
                max_concurrent_operations: 8,
                ..Default::default()
            };

            let mut orchestrator = MultiStreamOrchestrator::new(config)?;

            // Large matrix multiplication characteristics
            let matmul_characteristics = WorkloadCharacteristics {
                workload_type: WorkloadType::Compute,
                estimated_compute_time: Duration::from_millis(50),
                estimated_memory_bandwidth: 1_000_000_000_000, // 1 TB/s
                parallel_potential: 0.95,
                memory_access_pattern: MemoryAccessPattern::Strided { stride: 1024 },
                synchronization_requirements: SynchronizationRequirements {
                    requires_barrier: false,
                    dependencies: vec![],
                    provides_outputs: vec!["matmul_result".to_string()],
                },
            };

            let result = orchestrator.execute_operation(
                "large_matmul".to_string(),
                matmul_characteristics,
                |stream| simulate_kernel_launch(stream, "gemm_kernel", Duration::from_millis(50)),
            )?;

            println!("Execution time: {:?}", result.execution_time);
            println!("Used graph capture: {}", result.used_graph_capture);
            println!(
                "Memory bandwidth: {} GB/s",
                result.memory_bandwidth / 1_000_000_000
            );
        }

        Ok(())
    }

    /// Example: Repeating workload optimization
    pub fn repeating_workload_optimization() -> CudaResult<()> {
        let config = OrchestratorConfig {
            graph_capture_threshold: Duration::from_millis(1), // Aggressive graph capture
            enable_auto_optimization: true,
            ..Default::default()
        };

        let mut orchestrator = MultiStreamOrchestrator::new(config)?;

        // Define a typical inference workload
        let inference_characteristics = WorkloadCharacteristics {
            workload_type: WorkloadType::Mixed,
            estimated_compute_time: Duration::from_millis(8),
            estimated_memory_bandwidth: 400_000_000_000, // 400 GB/s
            parallel_potential: 0.8,
            memory_access_pattern: MemoryAccessPattern::Sequential,
            synchronization_requirements: SynchronizationRequirements {
                requires_barrier: false,
                dependencies: vec![],
                provides_outputs: vec!["inference_result".to_string()],
            },
        };

        println!("Running repeating inference workload...");

        let result = orchestrator.execute_repeating_workload(
            "inference_batch".to_string(),
            inference_characteristics,
            100, // 100 iterations
            |stream| {
                // Simulate inference operations
                simulate_kernel_launch(stream, "inference_kernel", Duration::from_millis(8))?;
                simulate_kernel_launch(stream, "postprocess_kernel", Duration::from_millis(2))?;
                Ok(())
            },
        )?;

        println!("Repeating workload results:");
        println!("Total iterations: {}", result.total_iterations);
        println!("Total execution time: {:?}", result.total_execution_time);
        println!(
            "Average execution time: {:?}",
            result.average_execution_time
        );
        println!("Graph capture used: {}", result.graph_capture_used);
        println!(
            "Performance improvement: {:.2}%",
            result.performance_improvement * 100.0
        );

        // Show execution time progression
        println!("\nExecution time progression (first 10 iterations):");
        for (i, &time) in result.execution_times.iter().take(10).enumerate() {
            println!("Iteration {}: {:?}", i + 1, time);
        }

        Ok(())
    }

    /// Example: Multi-GPU data parallel training
    pub fn multi_gpu_data_parallel_example() -> CudaResult<()> {
        let config = OrchestratorConfig {
            max_concurrent_operations: 32, // Support multiple GPUs
            enable_auto_optimization: true,
            ..Default::default()
        };

        let mut orchestrator = MultiStreamOrchestrator::new(config)?;

        // Simulate data parallel training across multiple streams
        let mut operations = Vec::new();

        for gpu_id in 0..4 {
            // Forward pass on each GPU
            let forward_characteristics = WorkloadCharacteristics {
                workload_type: WorkloadType::Compute,
                estimated_compute_time: Duration::from_millis(20),
                estimated_memory_bandwidth: 600_000_000_000, // 600 GB/s
                parallel_potential: 0.9,
                memory_access_pattern: MemoryAccessPattern::Sequential,
                synchronization_requirements: SynchronizationRequirements {
                    requires_barrier: false,
                    dependencies: vec![],
                    provides_outputs: vec![format!("forward_gpu_{}", gpu_id)],
                },
            };

            // Backward pass on each GPU
            let backward_characteristics = WorkloadCharacteristics {
                workload_type: WorkloadType::Compute,
                estimated_compute_time: Duration::from_millis(25),
                estimated_memory_bandwidth: 700_000_000_000, // 700 GB/s
                parallel_potential: 0.85,
                memory_access_pattern: MemoryAccessPattern::Sequential,
                synchronization_requirements: SynchronizationRequirements {
                    requires_barrier: false,
                    dependencies: vec![format!("forward_gpu_{}", gpu_id)],
                    provides_outputs: vec![format!("gradients_gpu_{}", gpu_id)],
                },
            };

            operations.push((format!("forward_gpu_{}", gpu_id), forward_characteristics));
            operations.push((format!("backward_gpu_{}", gpu_id), backward_characteristics));
        }

        // Gradient synchronization (all-reduce)
        let allreduce_characteristics = WorkloadCharacteristics {
            workload_type: WorkloadType::Memory,
            estimated_compute_time: Duration::from_millis(10),
            estimated_memory_bandwidth: 100_000_000_000, // 100 GB/s (communication bound)
            parallel_potential: 0.3,
            memory_access_pattern: MemoryAccessPattern::Reduction,
            synchronization_requirements: SynchronizationRequirements {
                requires_barrier: true,
                dependencies: (0..4).map(|i| format!("gradients_gpu_{}", i)).collect(),
                provides_outputs: vec!["synchronized_gradients".to_string()],
            },
        };

        operations.push(("allreduce_gradients".to_string(), allreduce_characteristics));

        println!("Executing multi-GPU data parallel training step...");

        let results = orchestrator.execute_batch(operations, |op_name| {
            Box::new(move |stream: &CudaStream| {
                if op_name.starts_with("forward_") {
                    simulate_kernel_launch(stream, "forward_kernel", Duration::from_millis(20))
                } else if op_name.starts_with("backward_") {
                    simulate_kernel_launch(stream, "backward_kernel", Duration::from_millis(25))
                } else if op_name.starts_with("allreduce_") {
                    simulate_kernel_launch(stream, "allreduce_kernel", Duration::from_millis(10))
                } else {
                    Ok(())
                }
            })
        })?;

        println!("Multi-GPU training step completed!");
        println!("Total operations executed: {}", results.len());

        // Synchronize all streams
        orchestrator.synchronize_all()?;

        // Get final metrics
        let metrics = orchestrator.get_metrics();
        println!("\nFinal performance metrics:");
        println!("Total operations: {}", metrics.total_operations_executed);
        println!(
            "Success rate: {:.1}%",
            (metrics.successful_operations as f64 / metrics.total_operations_executed as f64)
                * 100.0
        );
        println!(
            "Average execution time: {:?}",
            metrics.average_execution_time
        );
        println!(
            "Peak concurrent operations: {}",
            metrics.peak_concurrent_operations
        );

        Ok(())
    }

    /// Example: Memory-intensive workload optimization
    pub fn memory_intensive_workload_example() -> CudaResult<()> {
        let config = OrchestratorConfig {
            memory_pressure_threshold: 0.7, // More conservative memory usage
            enable_auto_optimization: true,
            ..Default::default()
        };

        let mut orchestrator = MultiStreamOrchestrator::new(config)?;

        // Large tensor operations
        let operations = vec![
            (
                "large_tensor_copy",
                WorkloadCharacteristics {
                    workload_type: WorkloadType::Memory,
                    estimated_compute_time: Duration::from_millis(30),
                    estimated_memory_bandwidth: 1_200_000_000_000, // 1.2 TB/s
                    parallel_potential: 0.4,
                    memory_access_pattern: MemoryAccessPattern::Sequential,
                    synchronization_requirements: SynchronizationRequirements {
                        requires_barrier: false,
                        dependencies: vec![],
                        provides_outputs: vec!["copied_tensor".to_string()],
                    },
                },
            ),
            (
                "tensor_reshape",
                WorkloadCharacteristics {
                    workload_type: WorkloadType::Memory,
                    estimated_compute_time: Duration::from_millis(5),
                    estimated_memory_bandwidth: 800_000_000_000, // 800 GB/s
                    parallel_potential: 0.7,
                    memory_access_pattern: MemoryAccessPattern::Strided { stride: 512 },
                    synchronization_requirements: SynchronizationRequirements {
                        requires_barrier: false,
                        dependencies: vec!["copied_tensor".to_string()],
                        provides_outputs: vec!["reshaped_tensor".to_string()],
                    },
                },
            ),
            (
                "memory_reduction",
                WorkloadCharacteristics {
                    workload_type: WorkloadType::Mixed,
                    estimated_compute_time: Duration::from_millis(12),
                    estimated_memory_bandwidth: 600_000_000_000, // 600 GB/s
                    parallel_potential: 0.8,
                    memory_access_pattern: MemoryAccessPattern::Reduction,
                    synchronization_requirements: SynchronizationRequirements {
                        requires_barrier: false,
                        dependencies: vec!["reshaped_tensor".to_string()],
                        provides_outputs: vec!["reduced_result".to_string()],
                    },
                },
            ),
        ];

        println!("Executing memory-intensive workload...");

        let results = orchestrator.execute_batch(
            operations
                .into_iter()
                .map(|(name, chars)| (name.to_string(), chars))
                .collect(),
            |op_name| {
                Box::new(move |stream: &CudaStream| match op_name {
                    "large_tensor_copy" => {
                        simulate_kernel_launch(stream, "memcpy_kernel", Duration::from_millis(30))
                    }
                    "tensor_reshape" => {
                        simulate_kernel_launch(stream, "reshape_kernel", Duration::from_millis(5))
                    }
                    "memory_reduction" => {
                        simulate_kernel_launch(stream, "reduce_kernel", Duration::from_millis(12))
                    }
                    _ => Ok(()),
                })
            },
        )?;

        println!("Memory-intensive workload completed!");
        for result in &results {
            println!(
                "Operation - Execution time: {:?}, Bandwidth: {} GB/s",
                result.execution_time,
                result.memory_bandwidth / 1_000_000_000
            );
        }

        Ok(())
    }

    // Helper function to simulate kernel launches
    fn simulate_kernel_launch(
        _stream: &CudaStream,
        kernel_name: &str,
        duration: Duration,
    ) -> CudaResult<()> {
        // In a real implementation, this would launch an actual CUDA kernel
        println!(
            "Launching kernel: {} (simulated for {:?})",
            kernel_name, duration
        );

        // Simulate execution time
        std::thread::sleep(std::time::Duration::from_micros(100)); // Very brief simulation

        Ok(())
    }

    /// Example: Performance analysis and optimization
    pub fn performance_analysis_example() -> CudaResult<()> {
        let mut orchestrator = MultiStreamOrchestrator::new(OrchestratorConfig::default())?;

        // Run various workloads to collect performance data
        let workloads = vec![
            (
                "small_compute",
                WorkloadType::Compute,
                Duration::from_millis(5),
            ),
            (
                "large_compute",
                WorkloadType::Compute,
                Duration::from_millis(50),
            ),
            (
                "memory_bound",
                WorkloadType::Memory,
                Duration::from_millis(20),
            ),
            (
                "mixed_workload",
                WorkloadType::Mixed,
                Duration::from_millis(15),
            ),
        ];

        for (name, workload_type, duration) in workloads {
            let characteristics = WorkloadCharacteristics {
                workload_type,
                estimated_compute_time: duration,
                estimated_memory_bandwidth: 500_000_000_000, // 500 GB/s
                parallel_potential: 0.8,
                memory_access_pattern: MemoryAccessPattern::Sequential,
                synchronization_requirements: SynchronizationRequirements {
                    requires_barrier: false,
                    dependencies: vec![],
                    provides_outputs: vec![format!("{}_output", name)],
                },
            };

            let _result =
                orchestrator.execute_operation(name.to_string(), characteristics, |stream| {
                    simulate_kernel_launch(stream, name, duration)
                })?;
        }

        // Optimize configuration based on collected data
        let optimization_result = orchestrator.optimize_configuration()?;

        println!("Performance optimization results:");
        println!(
            "Optimizations applied: {}",
            optimization_result.optimizations_applied
        );
        println!(
            "Performance improvement: {:.2}%",
            optimization_result.performance_improvement * 100.0
        );
        if let Some(new_strategy) = optimization_result.new_strategy {
            println!("New optimal strategy: {:?}", new_strategy);
        }

        // Get comprehensive performance report
        let profiling_report = orchestrator.get_profiling_report();
        println!("\nProfiling report:");
        println!("Total streams profiled: {}", profiling_report.total_streams);

        for stream_report in &profiling_report.streams {
            println!(
                "Stream {}: {} operations, {:?} total time",
                stream_report.stream_id, stream_report.operation_count, stream_report.total_time
            );
        }

        // Get graph performance summary
        let graph_performance = orchestrator.get_graph_performance();
        println!("\nGraph performance summary:");
        for (graph_name, summary) in graph_performance {
            println!(
                "Graph '{}': {} executions, trend: {:?}",
                graph_name, summary.execution_stats.execution_count, summary.performance_trend
            );
        }

        Ok(())
    }

    /// Run all examples
    pub fn run_all_examples() -> CudaResult<()> {
        println!("=== Multi-Stream Execution Examples ===\n");

        if crate::cuda::is_available() {
            println!("1. Neural Network Training Example");
            neural_network_training_example()?;

            println!("\n2. Matrix Multiplication Comparison");
            matrix_multiplication_comparison()?;

            println!("\n3. Repeating Workload Optimization");
            repeating_workload_optimization()?;

            println!("\n4. Multi-GPU Data Parallel Training");
            multi_gpu_data_parallel_example()?;

            println!("\n5. Memory-Intensive Workload");
            memory_intensive_workload_example()?;

            println!("\n6. Performance Analysis");
            performance_analysis_example()?;

            println!("\n=== All examples completed successfully! ===");
        } else {
            println!("CUDA not available - skipping examples");
        }

        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use examples::*;

#[cfg(not(feature = "cuda"))]
pub fn run_all_examples() -> crate::error::CudaResult<()> {
    println!("CUDA feature not enabled - multi-stream examples not available");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::examples::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_examples_compilation() {
        // Just test that the examples compile
        // Actual execution would require CUDA hardware
        if crate::cuda::is_available() {
            // Could run lightweight examples here
        }
    }

    #[test]
    fn test_example_availability() {
        #[cfg(feature = "cuda")]
        {
            // CUDA examples should be available
            assert!(true);
        }

        #[cfg(not(feature = "cuda"))]
        {
            // Should compile even without CUDA
            let result = run_all_examples();
            assert!(result.is_ok());
        }
    }
}
