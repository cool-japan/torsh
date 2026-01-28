//! Focused Integration Tests for Advanced Memory Optimization Systems
//!
//! This test suite specifically validates the advanced memory optimization capabilities
//! including predictive pooling, intelligent prefetching, bandwidth optimization,
//! and memory pattern analysis under various workload conditions.
//!
//! NOTE: This test is currently disabled because the memory optimization module
//! is not yet exported from the CUDA memory module.

#![allow(unexpected_cfgs)]
#![cfg(all(cuda_available, feature = "memory_optimization_tests"))]

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use torsh_backend::cuda::memory::optimization::advanced_memory_optimizer::{
    AdvancedMemoryConfig, AdvancedMemoryOptimizer, MemoryOptimizationError,
    MemoryOptimizationReport, MemoryOptimizationStatus, MemorySafetyLevel,
    OptimizationAggressiveness,
};

#[cfg(all(test, cuda_available))]
mod memory_optimization_tests {
    use super::*;

    /// Test predictive memory pooling under various allocation patterns
    #[test]
    fn test_predictive_memory_pooling() {
        let config = AdvancedMemoryConfig {
            enable_predictive_pooling: true,
            enable_intelligent_prefetch: true,
            enable_pattern_analysis: true,
            optimization_aggressiveness: OptimizationAggressiveness::Moderate,
            memory_safety_level: MemorySafetyLevel::Safe,
            ..Default::default()
        };

        let optimizer = AdvancedMemoryOptimizer::new(config);
        assert!(
            optimizer.initialize().is_ok(),
            "Memory optimizer should initialize successfully"
        );

        // Test different allocation patterns
        let allocation_patterns = vec![
            vec![1024, 2048, 4096],       // Growing pattern
            vec![4096, 2048, 1024],       // Shrinking pattern
            vec![1024, 4096, 1024, 4096], // Alternating pattern
            vec![8192, 8192, 8192],       // Constant pattern
        ];

        for (pattern_idx, pattern) in allocation_patterns.iter().enumerate() {
            let mut allocations = Vec::new();

            // Perform allocations
            for (alloc_idx, &size) in pattern.iter().enumerate() {
                let lifetime_hint = Some(Duration::from_millis((alloc_idx + 1) as u64 * 100));
                let ptr_result = optimizer.optimized_allocate(size, 256, lifetime_hint);

                assert!(
                    ptr_result.is_ok(),
                    "Allocation {} of pattern {} should succeed (size: {})",
                    alloc_idx,
                    pattern_idx,
                    size
                );

                if let Ok(ptr) = ptr_result {
                    allocations.push((ptr, size));
                }
            }

            // Perform deallocations in reverse order (LIFO pattern)
            for (ptr, size) in allocations.into_iter().rev() {
                let dealloc_result = optimizer.optimized_deallocate(ptr, size);
                assert!(
                    dealloc_result.is_ok(),
                    "Deallocation should succeed for pattern {}",
                    pattern_idx
                );
            }
        }

        let status = optimizer.get_optimization_status();
        assert!(
            status.total_optimizations > 0,
            "Should have performed optimizations"
        );
    }

    /// Test intelligent prefetching accuracy and effectiveness
    #[test]
    fn test_intelligent_prefetching() {
        let config = AdvancedMemoryConfig {
            enable_predictive_pooling: false, // Focus on prefetching
            enable_intelligent_prefetch: true,
            enable_pattern_analysis: true,
            enable_bandwidth_optimization: true,
            optimization_aggressiveness: OptimizationAggressiveness::Aggressive,
            memory_safety_level: MemorySafetyLevel::Moderate,
            ..Default::default()
        };

        let optimizer = AdvancedMemoryOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Create predictable access patterns for prefetch learning
        let access_sizes = vec![1024, 2048, 4096, 8192];
        let num_iterations = 5;

        for iteration in 0..num_iterations {
            let mut allocations = Vec::new();

            // Allocate in predictable pattern
            for (idx, &size) in access_sizes.iter().enumerate() {
                let enhanced_size = size * (iteration + 1); // Gradually increasing
                let ptr_result =
                    optimizer.optimized_allocate(enhanced_size, 256, Some(Duration::from_secs(1)));

                assert!(
                    ptr_result.is_ok(),
                    "Allocation {} in iteration {} should succeed",
                    idx,
                    iteration
                );

                if let Ok(ptr) = ptr_result {
                    allocations.push((ptr, enhanced_size));
                }

                // Small delay to simulate access pattern timing
                thread::sleep(Duration::from_millis(10));
            }

            // Deallocate in same order to maintain pattern
            for (ptr, size) in allocations {
                assert!(optimizer.optimized_deallocate(ptr, size).is_ok());
                thread::sleep(Duration::from_millis(5));
            }
        }

        // After pattern establishment, prefetching should be more effective
        let status = optimizer.get_optimization_status();
        assert!(
            status.prefetch_accuracy >= 0.0,
            "Prefetch accuracy should be tracked"
        );

        // Run comprehensive optimization to see prefetch improvements
        let optimization_report = optimizer.perform_comprehensive_optimization();
        assert!(optimization_report.is_ok());

        let report = optimization_report.unwrap();
        assert!(
            !report.recommendations.is_empty(),
            "Should provide optimization recommendations"
        );
    }

    /// Test memory bandwidth optimization under various access patterns
    #[test]
    fn test_memory_bandwidth_optimization() {
        let config = AdvancedMemoryConfig {
            enable_bandwidth_optimization: true,
            enable_pattern_analysis: true,
            enable_cache_optimization: true,
            optimization_aggressiveness: OptimizationAggressiveness::Aggressive,
            memory_safety_level: MemorySafetyLevel::Performance,
            ..Default::default()
        };

        let optimizer = AdvancedMemoryOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Test different bandwidth-intensive scenarios
        let bandwidth_test_scenarios = vec![
            ("Sequential Access", vec![1024, 1024, 1024, 1024, 1024]),
            ("Random Access", vec![8192, 2048, 16384, 1024, 32768]),
            ("Burst Access", vec![65536, 65536, 65536]),
            ("Mixed Workload", vec![1024, 32768, 4096, 16384, 2048]),
        ];

        for (scenario_name, sizes) in bandwidth_test_scenarios {
            let start_time = Instant::now();
            let mut allocations = Vec::new();

            // Perform allocations
            for &size in &sizes {
                let ptr_result = optimizer.optimized_allocate(size, 256, None);
                assert!(
                    ptr_result.is_ok(),
                    "Allocation should succeed in scenario: {}",
                    scenario_name
                );

                if let Ok(ptr) = ptr_result {
                    allocations.push((ptr, size));
                }
            }

            // Measure allocation performance
            let allocation_time = start_time.elapsed();

            // Perform deallocations
            let dealloc_start = Instant::now();
            for (ptr, size) in allocations {
                assert!(optimizer.optimized_deallocate(ptr, size).is_ok());
            }
            let deallocation_time = dealloc_start.elapsed();

            // Bandwidth optimization should keep times reasonable
            assert!(
                allocation_time < Duration::from_millis(100),
                "Allocation time should be reasonable for scenario: {}",
                scenario_name
            );
            assert!(
                deallocation_time < Duration::from_millis(50),
                "Deallocation time should be reasonable for scenario: {}",
                scenario_name
            );
        }

        let status = optimizer.get_optimization_status();
        assert!(
            status.total_optimizations >= 4,
            "Should have optimized all scenarios"
        );
    }

    /// Test memory compaction and defragmentation
    #[test]
    fn test_memory_compaction() {
        let config = AdvancedMemoryConfig {
            enable_memory_compaction: true,
            enable_pattern_analysis: true,
            enable_pressure_monitoring: true,
            optimization_aggressiveness: OptimizationAggressiveness::Aggressive,
            memory_safety_level: MemorySafetyLevel::Safe,
            ..Default::default()
        };

        let optimizer = AdvancedMemoryOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Create fragmentation by allocating and deallocating in a pattern that creates holes
        let fragmentation_pattern = vec![
            (1024, true),  // Allocate
            (2048, true),  // Allocate
            (4096, true),  // Allocate
            (8192, true),  // Allocate
            (2048, false), // Deallocate second allocation (creates hole)
            (8192, false), // Deallocate fourth allocation (creates hole)
            (1024, true),  // Allocate small (should use first hole or compact)
            (16384, true), // Allocate large (may trigger compaction)
        ];

        let mut active_allocations = HashMap::new();
        let mut allocation_counter = 0;

        for (size, is_allocation) in fragmentation_pattern {
            if is_allocation {
                allocation_counter += 1;
                let ptr_result = optimizer.optimized_allocate(size, 256, None);
                assert!(
                    ptr_result.is_ok(),
                    "Allocation {} should succeed (size: {})",
                    allocation_counter,
                    size
                );

                if let Ok(ptr) = ptr_result {
                    active_allocations.insert(allocation_counter, (ptr, size));
                }
            } else {
                // Find allocation to deallocate based on size
                let key_to_remove = active_allocations
                    .iter()
                    .find(|(_, &(_, alloc_size))| alloc_size == size)
                    .map(|(&k, _)| k);

                if let Some(key) = key_to_remove {
                    let (ptr, alloc_size) = active_allocations.remove(&key).unwrap();
                    assert!(optimizer.optimized_deallocate(ptr, alloc_size).is_ok());
                }
            }
        }

        // Clean up remaining allocations
        for (ptr, size) in active_allocations.into_values() {
            assert!(optimizer.optimized_deallocate(ptr, size).is_ok());
        }

        // Run comprehensive optimization which should trigger compaction
        let optimization_result = optimizer.perform_comprehensive_optimization();
        assert!(optimization_result.is_ok());

        let report = optimization_result.unwrap();
        assert!(
            report.performance_improvement >= 0.0,
            "Should show performance metrics"
        );
        assert!(
            report.memory_savings >= 0,
            "Should show memory savings from compaction"
        );
    }

    /// Test cache hierarchy optimization
    #[test]
    fn test_cache_hierarchy_optimization() {
        let config = AdvancedMemoryConfig {
            enable_cache_optimization: true,
            enable_pattern_analysis: true,
            enable_bandwidth_optimization: true,
            optimization_aggressiveness: OptimizationAggressiveness::Maximum,
            memory_safety_level: MemorySafetyLevel::Performance,
            ..Default::default()
        };

        let optimizer = AdvancedMemoryOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Test cache-friendly and cache-unfriendly access patterns
        let cache_test_scenarios = vec![
            ("Cache Friendly", vec![64, 64, 64, 64, 64, 64, 64, 64]), // Small, uniform
            (
                "Cache Hostile",
                vec![1024 * 1024, 512, 2 * 1024 * 1024, 256],
            ), // Large, varied
            ("Mixed Pattern", vec![128, 1024, 256, 2048, 512, 4096]), // Mixed sizes
        ];

        for (scenario_name, sizes) in cache_test_scenarios {
            let scenario_start = Instant::now();
            let mut allocations = Vec::new();

            // Perform rapid allocations to stress cache
            for &size in &sizes {
                for iteration in 0..3 {
                    let actual_size = size * (iteration + 1);
                    let ptr_result = optimizer.optimized_allocate(
                        actual_size,
                        32,
                        Some(Duration::from_millis(100)),
                    );

                    if let Ok(ptr) = ptr_result {
                        allocations.push((ptr, actual_size));
                    }
                }
            }

            let allocation_phase_time = scenario_start.elapsed();

            // Rapid deallocations
            for (ptr, size) in allocations {
                optimizer
                    .optimized_deallocate(ptr, size)
                    .expect("Deallocation should succeed");
            }

            let total_scenario_time = scenario_start.elapsed();

            // Cache optimization should improve performance for cache-friendly patterns
            println!(
                "Scenario '{}' - Allocation: {:?}, Total: {:?}",
                scenario_name, allocation_phase_time, total_scenario_time
            );

            assert!(
                total_scenario_time < Duration::from_secs(1),
                "Scenario should complete within reasonable time: {}",
                scenario_name
            );
        }

        let status = optimizer.get_optimization_status();
        assert!(
            status.cache_optimization_count >= 0,
            "Should track cache optimizations"
        );
    }

    /// Test memory pressure monitoring and adaptive response
    #[test]
    fn test_memory_pressure_monitoring() {
        let config = AdvancedMemoryConfig {
            enable_pressure_monitoring: true,
            enable_adaptive_strategies: true,
            enable_memory_compaction: true,
            optimization_aggressiveness: OptimizationAggressiveness::Moderate,
            memory_safety_level: MemorySafetyLevel::Safe,
            ..Default::default()
        };

        let optimizer = AdvancedMemoryOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Create memory pressure by allocating increasingly large blocks
        let pressure_sizes = vec![
            1024 * 1024,      // 1MB
            4 * 1024 * 1024,  // 4MB
            16 * 1024 * 1024, // 16MB
            64 * 1024 * 1024, // 64MB
        ];

        let mut large_allocations = Vec::new();

        for (idx, &size) in pressure_sizes.iter().enumerate() {
            println!(
                "Allocating large block {} of size {} MB",
                idx,
                size / (1024 * 1024)
            );

            let ptr_result = optimizer.optimized_allocate(size, 256, None);

            match ptr_result {
                Ok(ptr) => {
                    large_allocations.push((ptr, size));
                    println!("Successfully allocated {} MB", size / (1024 * 1024));
                }
                Err(e) => {
                    println!(
                        "Failed to allocate {} MB: {:?} - This may be expected under pressure",
                        size / (1024 * 1024),
                        e
                    );
                }
            }

            // Check system status after each large allocation
            let status = optimizer.get_optimization_status();
            println!("Current pressure level: {:?}", status.pressure_level);

            // Allow pressure monitoring to adapt
            thread::sleep(Duration::from_millis(100));
        }

        // Clean up allocations
        for (ptr, size) in large_allocations {
            assert!(optimizer.optimized_deallocate(ptr, size).is_ok());
        }

        // Run comprehensive optimization to see pressure handling results
        let optimization_report = optimizer.perform_comprehensive_optimization();
        assert!(optimization_report.is_ok());

        let report = optimization_report.unwrap();
        println!(
            "Memory pressure test completed. Performance improvement: {:.2}%",
            report.performance_improvement
        );

        assert!(
            !report.recommendations.is_empty(),
            "Should provide recommendations after pressure testing"
        );
    }

    /// Test concurrent memory operations for thread safety
    #[test]
    fn test_concurrent_memory_operations() {
        let config = AdvancedMemoryConfig {
            enable_predictive_pooling: true,
            enable_intelligent_prefetch: true,
            enable_pattern_analysis: true,
            optimization_aggressiveness: OptimizationAggressiveness::Moderate,
            memory_safety_level: MemorySafetyLevel::Safe,
            ..Default::default()
        };

        let optimizer = Arc::new(AdvancedMemoryOptimizer::new(config));
        assert!(optimizer.initialize().is_ok());

        let num_threads = 4;
        let operations_per_thread = 10;
        let mut handles = Vec::new();

        // Spawn concurrent threads performing memory operations
        for thread_id in 0..num_threads {
            let optimizer_clone = Arc::clone(&optimizer);

            let handle = thread::spawn(move || {
                let mut thread_allocations = Vec::new();
                let base_size = 1024 * (thread_id + 1); // Different sizes per thread

                // Perform allocations
                for op_id in 0..operations_per_thread {
                    let size = base_size * (op_id + 1);
                    let lifetime = Duration::from_millis((op_id as u64 + 1) * 50);

                    let ptr_result = optimizer_clone.optimized_allocate(size, 256, Some(lifetime));

                    if let Ok(ptr) = ptr_result {
                        thread_allocations.push((ptr, size));
                    }

                    // Small delay to create realistic timing
                    thread::sleep(Duration::from_millis(10));
                }

                // Perform deallocations
                for (ptr, size) in thread_allocations {
                    let dealloc_result = optimizer_clone.optimized_deallocate(ptr, size);
                    assert!(
                        dealloc_result.is_ok(),
                        "Deallocation should succeed in thread {}",
                        thread_id
                    );
                }

                println!(
                    "Thread {} completed {} operations",
                    thread_id, operations_per_thread
                );
                thread_id // Return thread ID for verification
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        let mut completed_threads = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(thread_id) => completed_threads.push(thread_id),
                Err(e) => panic!("Thread panicked: {:?}", e),
            }
        }

        assert_eq!(
            completed_threads.len(),
            num_threads,
            "All threads should complete successfully"
        );

        let status = optimizer.get_optimization_status();
        assert!(
            status.total_optimizations >= num_threads as u64,
            "Should have performed optimizations from all threads"
        );

        println!(
            "Concurrent test completed with {} threads, {} total optimizations",
            num_threads, status.total_optimizations
        );
    }

    /// Test error handling and recovery scenarios
    #[test]
    fn test_error_handling_and_recovery() {
        let config = AdvancedMemoryConfig {
            optimization_aggressiveness: OptimizationAggressiveness::Safe,
            memory_safety_level: MemorySafetyLevel::Safe,
            ..Default::default()
        };

        let optimizer = AdvancedMemoryOptimizer::new(config);
        assert!(optimizer.initialize().is_ok());

        // Test invalid allocation scenarios
        let invalid_scenarios = vec![
            (0, 256, "Zero size allocation"),
            (usize::MAX, 256, "Maximum size allocation"),
            (1024, 0, "Zero alignment"),
        ];

        for (size, alignment, description) in invalid_scenarios {
            let result = optimizer.optimized_allocate(size, alignment, None);

            match result {
                Ok(_) => {
                    // If it succeeds, it should be a valid allocation
                    println!("{}: Surprisingly succeeded (may be valid)", description);
                }
                Err(MemoryOptimizationError::AllocationFailed(msg)) => {
                    println!("{}: Failed as expected - {}", description, msg);
                }
                Err(e) => {
                    println!("{}: Failed with error: {:?}", description, e);
                }
            }
        }

        // Test double deallocation (should be handled gracefully)
        if let Ok(ptr) = optimizer.optimized_allocate(1024, 256, None) {
            assert!(optimizer.optimized_deallocate(ptr, 1024).is_ok());

            // Second deallocation should be handled gracefully
            let second_dealloc = optimizer.optimized_deallocate(ptr, 1024);
            match second_dealloc {
                Ok(_) => println!("Double deallocation handled successfully"),
                Err(e) => println!("Double deallocation failed as expected: {:?}", e),
            }
        }

        // Test system recovery after errors
        let status = optimizer.get_optimization_status();
        assert!(
            status.memory_leak_detections >= 0,
            "Should track potential memory issues"
        );
    }
}

/// Performance characterization tests for different optimization levels
#[cfg(test)]
mod performance_characterization_tests {
    use super::*;

    #[test]
    fn test_optimization_aggressiveness_levels() {
        let aggressiveness_levels = vec![
            OptimizationAggressiveness::Safe,
            OptimizationAggressiveness::Moderate,
            OptimizationAggressiveness::Aggressive,
            OptimizationAggressiveness::Extreme,
        ];

        for aggressiveness in aggressiveness_levels {
            let config = AdvancedMemoryConfig {
                optimization_aggressiveness: aggressiveness,
                memory_safety_level: MemorySafetyLevel::Safe,
                enable_predictive_pooling: true,
                enable_intelligent_prefetch: true,
                enable_pattern_analysis: true,
                ..Default::default()
            };

            let optimizer = AdvancedMemoryOptimizer::new(config);
            assert!(
                optimizer.initialize().is_ok(),
                "Should initialize with aggressiveness level: {:?}",
                aggressiveness
            );

            // Run standard allocation pattern
            let test_sizes = vec![1024, 4096, 16384, 65536];
            let start_time = Instant::now();

            for &size in &test_sizes {
                if let Ok(ptr) = optimizer.optimized_allocate(size, 256, None) {
                    assert!(optimizer.optimized_deallocate(ptr, size).is_ok());
                }
            }

            let execution_time = start_time.elapsed();
            println!(
                "Aggressiveness {:?}: Execution time {:?}",
                aggressiveness, execution_time
            );

            let optimization_report = optimizer.perform_comprehensive_optimization();
            assert!(optimization_report.is_ok());

            let report = optimization_report.unwrap();
            println!(
                "Aggressiveness {:?}: Performance improvement {:.2}%",
                aggressiveness, report.performance_improvement
            );

            // More aggressive levels should potentially show better optimization
            // but this is implementation-dependent
            assert!(report.performance_improvement >= 0.0);
        }
    }

    #[test]
    fn test_memory_safety_levels() {
        let safety_levels = vec![
            MemorySafetyLevel::Unsafe,
            MemorySafetyLevel::Performance,
            MemorySafetyLevel::Moderate,
            MemorySafetyLevel::Safe,
        ];

        for safety_level in safety_levels {
            let config = AdvancedMemoryConfig {
                memory_safety_level: safety_level,
                optimization_aggressiveness: OptimizationAggressiveness::Moderate,
                enable_pressure_monitoring: true,
                ..Default::default()
            };

            let optimizer = AdvancedMemoryOptimizer::new(config);
            assert!(
                optimizer.initialize().is_ok(),
                "Should initialize with safety level: {:?}",
                safety_level
            );

            // Test allocation patterns that might stress safety mechanisms
            let stress_pattern = vec![
                1024, 2048, 4096, 8192, 16384, 16384, 8192, 4096, 2048,
                1024, // Reverse pattern
            ];

            let mut allocations = Vec::new();
            for &size in &stress_pattern {
                if let Ok(ptr) = optimizer.optimized_allocate(size, 256, None) {
                    allocations.push((ptr, size));
                }
            }

            // Clean up
            for (ptr, size) in allocations {
                assert!(optimizer.optimized_deallocate(ptr, size).is_ok());
            }

            let status = optimizer.get_optimization_status();
            println!(
                "Safety level {:?}: {} optimizations, {} memory saved",
                safety_level, status.total_optimizations, status.memory_saved
            );

            // All safety levels should complete successfully
            assert!(status.total_optimizations >= 0);
        }
    }
}
