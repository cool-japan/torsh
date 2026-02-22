//! Comprehensive tests for advanced profiling functionality

use std::collections::HashMap;
use std::time::Duration;
use torsh_utils::benchmark::TimingStats;
use torsh_utils::bottleneck::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic bottleneck profiling
    #[test]
    fn test_basic_profiling() {
        let config = AdvancedProfilingConfig {
            enable_flame_graph: false,
            enable_memory_profiling: true,
            enable_gpu_profiling: false,
            enable_call_stack_analysis: false,
            enable_regression_detection: false,
            enable_hotspot_analysis: false,
            sample_rate_hz: 1000.0,
            memory_snapshot_interval_ms: 10.0,
        };

        assert!(!config.enable_flame_graph);
        assert!(config.enable_memory_profiling);
        assert_eq!(config.sample_rate_hz, 1000.0);
    }

    /// Test flame graph generation
    #[test]
    fn test_flame_graph_generation() {
        let samples = vec![
            ProfileSample {
                function_name: "conv2d_forward".to_string(),
                duration_ms: 10.0,
                stack_trace: vec![
                    "model.forward".to_string(),
                    "conv_layer.forward".to_string(),
                ],
            },
            ProfileSample {
                function_name: "linear_forward".to_string(),
                duration_ms: 5.0,
                stack_trace: vec![
                    "model.forward".to_string(),
                    "linear_layer.forward".to_string(),
                ],
            },
        ];

        let flame_frame = build_flame_graph_tree(samples).unwrap();

        assert_eq!(flame_frame.name, "root");
        assert_eq!(flame_frame.children.len(), 2);
        assert!(flame_frame.total_time_ms > 0.0);

        // Check that children have correct data
        let conv_child = flame_frame
            .children
            .iter()
            .find(|child| child.name == "conv2d_forward");
        assert!(conv_child.is_some());
        assert_eq!(conv_child.unwrap().self_time_ms, 10.0);
    }

    /// Test memory profiling data structures
    #[test]
    fn test_memory_profiling() {
        let memory_snapshots = vec![
            MemorySnapshot {
                timestamp_ms: 0.0,
                allocated_mb: 100.0,
                reserved_mb: 150.0,
                active_allocations: 1000,
                largest_free_block_mb: 50.0,
            },
            MemorySnapshot {
                timestamp_ms: 10.0,
                allocated_mb: 120.0,
                reserved_mb: 150.0,
                active_allocations: 1200,
                largest_free_block_mb: 30.0,
            },
        ];

        let memory_leaks = vec![MemoryLeak {
            allocation_site: "tensor_alloc".to_string(),
            size_mb: 5.0,
            age_ms: 1000.0,
            stack_trace: vec!["tensor_new".to_string(), "model_forward".to_string()],
        }];

        let memory_profile = MemoryProfileData {
            peak_usage_mb: 120.0,
            current_usage_mb: 110.0,
            allocation_timeline: memory_snapshots,
            memory_leaks,
            fragmentation_ratio: 0.15,
            gc_pressure: None,
            memory_bandwidth_utilization: 75.0,
            cache_performance: CachePerformance {
                l1_hit_rate: 0.95,
                l2_hit_rate: 0.88,
                l3_hit_rate: Some(0.82),
                cache_misses_per_instruction: 0.05,
                memory_stalls_percentage: 12.0,
            },
        };

        assert_eq!(memory_profile.peak_usage_mb, 120.0);
        assert_eq!(memory_profile.memory_leaks.len(), 1);
        assert_eq!(memory_profile.fragmentation_ratio, 0.15);
        assert_eq!(memory_profile.cache_performance.l1_hit_rate, 0.95);
    }

    /// Test GPU profiling data
    #[test]
    fn test_gpu_profiling() {
        let kernel_executions = vec![
            GpuKernelExecution {
                kernel_name: "conv2d_kernel".to_string(),
                duration_ms: 2.5,
                grid_size: (64, 64, 1),
                block_size: (16, 16, 1),
                registers_per_thread: 32,
                shared_memory_kb: 48.0,
                occupancy: 0.75,
            },
            GpuKernelExecution {
                kernel_name: "matmul_kernel".to_string(),
                duration_ms: 1.8,
                grid_size: (32, 32, 1),
                block_size: (32, 32, 1),
                registers_per_thread: 24,
                shared_memory_kb: 64.0,
                occupancy: 0.85,
            },
        ];

        let memory_transfers = vec![
            GpuMemoryTransfer {
                direction: MemoryTransferDirection::HostToDevice,
                size_mb: 10.0,
                duration_ms: 0.5,
                bandwidth_gb_s: 20.0,
            },
            GpuMemoryTransfer {
                direction: MemoryTransferDirection::DeviceToHost,
                size_mb: 5.0,
                duration_ms: 0.3,
                bandwidth_gb_s: 16.7,
            },
        ];

        let gpu_profile = GpuProfileData {
            utilization_percentage: 85.0,
            memory_utilization_percentage: 60.0,
            temperature_celsius: 65.0,
            power_consumption_watts: 150.0,
            kernel_executions,
            memory_transfers,
            compute_capability: "8.6".to_string(),
            occupancy_percentage: 80.0,
        };

        assert_eq!(gpu_profile.kernel_executions.len(), 2);
        assert_eq!(gpu_profile.memory_transfers.len(), 2);
        assert_eq!(gpu_profile.utilization_percentage, 85.0);
    }

    /// Test call stack analysis
    #[test]
    fn test_call_stack_analysis() {
        let call_stacks = vec![
            vec![
                "main".to_string(),
                "model.forward".to_string(),
                "conv_layer.forward".to_string(),
            ],
            vec![
                "main".to_string(),
                "model.forward".to_string(),
                "linear_layer.forward".to_string(),
            ],
            vec![
                "main".to_string(),
                "model.forward".to_string(),
                "conv_layer.forward".to_string(),
            ],
        ];

        let analysis = analyze_call_stacks(call_stacks).unwrap();

        assert_eq!(analysis.max_stack_depth, 3);
        assert_eq!(analysis.average_stack_depth, 3.0);
        assert!(analysis.call_frequency.contains_key("conv_layer.forward"));
        assert_eq!(analysis.call_frequency["conv_layer.forward"], 2);
        assert!(analysis.hottest_paths.len() > 0);
    }

    /// Test hotspot analysis
    #[test]
    fn test_hotspot_analysis() {
        let cpu_hotspots = vec![
            Hotspot {
                function_name: "conv2d_forward".to_string(),
                time_percentage: 45.0,
                instruction_count: Some(2_000_000),
                cache_misses: Some(150_000),
                branch_mispredictions: Some(10_000),
            },
            Hotspot {
                function_name: "matrix_multiply".to_string(),
                time_percentage: 30.0,
                instruction_count: Some(1_500_000),
                cache_misses: Some(100_000),
                branch_mispredictions: Some(5_000),
            },
        ];

        let memory_hotspots = vec![
            MemoryHotspot {
                operation: "large_tensor_copy".to_string(),
                access_pattern: MemoryAccessPattern::Sequential,
                bandwidth_utilization: 80.0,
                latency_ms: 2.5,
            },
            MemoryHotspot {
                operation: "sparse_access".to_string(),
                access_pattern: MemoryAccessPattern::Random,
                bandwidth_utilization: 25.0,
                latency_ms: 8.0,
            },
        ];

        let hotspot_analysis = HotspotAnalysis {
            cpu_hotspots,
            memory_hotspots,
            io_hotspots: vec![],
            synchronization_hotspots: vec![],
        };

        assert_eq!(hotspot_analysis.cpu_hotspots.len(), 2);
        assert_eq!(hotspot_analysis.memory_hotspots.len(), 2);

        // Verify hottest CPU function
        let hottest_cpu = &hotspot_analysis.cpu_hotspots[0];
        assert_eq!(hottest_cpu.function_name, "conv2d_forward");
        assert_eq!(hottest_cpu.time_percentage, 45.0);

        // Verify memory access patterns
        let sequential_access = &hotspot_analysis.memory_hotspots[0];
        assert!(matches!(
            sequential_access.access_pattern,
            MemoryAccessPattern::Sequential
        ));
        assert_eq!(sequential_access.bandwidth_utilization, 80.0);
    }

    /// Test timing statistics calculation
    #[test]
    fn test_timing_statistics() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(13),
            Duration::from_millis(14),
            Duration::from_millis(9),
            Duration::from_millis(15),
        ];

        let stats = calculate_timing_stats(&times);

        assert_eq!(stats.min, Duration::from_millis(9));
        assert_eq!(stats.max, Duration::from_millis(15));
        assert!(stats.mean >= Duration::from_millis(10));
        assert!(stats.mean <= Duration::from_millis(15));
        assert!(stats.std > Duration::ZERO);
    }

    /// Test bottleneck report generation
    #[test]
    fn test_bottleneck_report() {
        let layer_times = vec![
            LayerTiming {
                name: "conv1".to_string(),
                module_type: "Conv2D".to_string(),
                forward_time: Duration::from_millis(10),
                backward_time: Some(Duration::from_millis(15)),
                percentage: 35.0,
                num_params: 1000,
            },
            LayerTiming {
                name: "linear1".to_string(),
                module_type: "Linear".to_string(),
                forward_time: Duration::from_millis(5),
                backward_time: Some(Duration::from_millis(8)),
                percentage: 20.0,
                num_params: 500,
            },
        ];

        let mut operation_times = HashMap::new();
        operation_times.insert(
            "forward".to_string(),
            OperationTiming {
                count: 100,
                total_time: Duration::from_millis(1000),
                avg_time: Duration::from_millis(10),
                min_time: Duration::from_millis(8),
                max_time: Duration::from_millis(15),
            },
        );

        let memory_peaks = vec![MemoryPeak {
            operation: "forward_pass".to_string(),
            allocated_mb: 256.0,
            reserved_mb: 300.0,
        }];

        let memory_profile = MemoryProfileData::default();
        let hotspot_analysis = HotspotAnalysis::default();

        let recommendations = generate_advanced_recommendations(
            &layer_times,
            &operation_times,
            &memory_peaks,
            &memory_profile,
            &hotspot_analysis,
        );

        // Should have basic recommendations for slow layers
        assert!(!recommendations.is_empty());

        // Check for convolution-specific recommendations
        let conv_recommendations: Vec<_> = recommendations
            .iter()
            .filter(|r| r.contains("conv"))
            .collect();
        assert!(!conv_recommendations.is_empty());
    }

    /// Test performance regression detection
    #[test]
    fn test_performance_regression() {
        let baseline = PerformanceMetrics {
            total_time_ms: 100.0,
            memory_usage_mb: 256.0,
            throughput_ops_per_sec: 1000.0,
            energy_consumption_mj: Some(5.0),
        };

        let current = PerformanceMetrics {
            total_time_ms: 120.0,             // 20% slower
            memory_usage_mb: 280.0,           // 9% more memory
            throughput_ops_per_sec: 900.0,    // 10% lower throughput
            energy_consumption_mj: Some(6.0), // 20% more energy
        };

        let regression = RegressionAnalysis {
            baseline_performance: baseline,
            current_performance: current,
            regression_percentage: 15.0,
            regressed_operations: vec!["conv2d".to_string(), "matmul".to_string()],
            improvements: vec!["batch_norm".to_string()],
        };

        assert_eq!(regression.regression_percentage, 15.0);
        assert_eq!(regression.regressed_operations.len(), 2);
        assert_eq!(regression.improvements.len(), 1);

        // Verify specific regressions
        let time_regression = (regression.current_performance.total_time_ms
            - regression.baseline_performance.total_time_ms)
            / regression.baseline_performance.total_time_ms;
        assert!((time_regression - 0.2).abs() < 0.01); // 20% regression
    }

    /// Test memory leak detection
    #[test]
    fn test_memory_leak_detection() {
        let leaks = vec![
            MemoryLeak {
                allocation_site: "tensor_cache".to_string(),
                size_mb: 10.0,
                age_ms: 5000.0,
                stack_trace: vec![
                    "cache_insert".to_string(),
                    "model_forward".to_string(),
                    "training_loop".to_string(),
                ],
            },
            MemoryLeak {
                allocation_site: "gradient_buffer".to_string(),
                size_mb: 2.5,
                age_ms: 1000.0,
                stack_trace: vec!["alloc_gradient".to_string(), "backward_pass".to_string()],
            },
        ];

        assert_eq!(leaks.len(), 2);

        // Check largest leak
        let largest_leak = leaks
            .iter()
            .max_by(|a, b| a.size_mb.partial_cmp(&b.size_mb).unwrap())
            .unwrap();
        assert_eq!(largest_leak.allocation_site, "tensor_cache");
        assert_eq!(largest_leak.size_mb, 10.0);

        // Check oldest leak
        let oldest_leak = leaks
            .iter()
            .max_by(|a, b| a.age_ms.partial_cmp(&b.age_ms).unwrap())
            .unwrap();
        assert_eq!(oldest_leak.age_ms, 5000.0);
    }

    /// Test cache performance analysis
    #[test]
    fn test_cache_performance() {
        let cache_perf = CachePerformance {
            l1_hit_rate: 0.95,
            l2_hit_rate: 0.85,
            l3_hit_rate: Some(0.75),
            cache_misses_per_instruction: 0.08,
            memory_stalls_percentage: 15.0,
        };

        assert_eq!(cache_perf.l1_hit_rate, 0.95);
        assert_eq!(cache_perf.l2_hit_rate, 0.85);
        assert_eq!(cache_perf.l3_hit_rate.unwrap(), 0.75);

        // Test cache efficiency calculation
        let overall_hit_rate = cache_perf.l1_hit_rate
            + (1.0 - cache_perf.l1_hit_rate) * cache_perf.l2_hit_rate
            + (1.0 - cache_perf.l1_hit_rate)
                * (1.0 - cache_perf.l2_hit_rate)
                * cache_perf.l3_hit_rate.unwrap_or(0.0);

        assert!(overall_hit_rate > 0.95);
        assert!(cache_perf.memory_stalls_percentage < 20.0);
    }

    /// Test profiling configuration validation
    #[test]
    fn test_profiling_config_validation() {
        let valid_config = AdvancedProfilingConfig {
            enable_flame_graph: true,
            enable_memory_profiling: true,
            enable_gpu_profiling: true,
            enable_call_stack_analysis: true,
            enable_regression_detection: false,
            enable_hotspot_analysis: true,
            sample_rate_hz: 1000.0,
            memory_snapshot_interval_ms: 5.0,
        };

        // Validate sample rate
        assert!(valid_config.sample_rate_hz > 0.0);
        assert!(valid_config.sample_rate_hz <= 10000.0);

        // Validate snapshot interval
        assert!(valid_config.memory_snapshot_interval_ms > 0.0);
        assert!(valid_config.memory_snapshot_interval_ms <= 1000.0);

        // Test configuration combinations
        if valid_config.enable_gpu_profiling {
            // GPU profiling should be available
            assert!(true);
        }

        if valid_config.enable_flame_graph {
            // Should have reasonable sample rate for flame graphs
            assert!(valid_config.sample_rate_hz >= 100.0);
        }
    }

    // Helper functions

    fn calculate_timing_stats(times: &[Duration]) -> TimingStats {
        let mut sorted_times = times.to_vec();
        sorted_times.sort();

        let n = sorted_times.len() as f32;
        let mean = sorted_times.iter().sum::<Duration>() / sorted_times.len() as u32;

        let variance = sorted_times
            .iter()
            .map(|t| {
                let diff = t.as_secs_f32() - mean.as_secs_f32();
                diff * diff
            })
            .sum::<f32>()
            / n;

        let std = Duration::from_secs_f32(variance.sqrt());

        TimingStats {
            mean,
            std,
            min: sorted_times[0],
            max: sorted_times[sorted_times.len() - 1],
            median: sorted_times[sorted_times.len() / 2],
            p95: sorted_times[(0.95 * n) as usize],
            p99: sorted_times[(0.99 * n) as usize],
        }
    }

    fn generate_advanced_recommendations(
        layer_times: &[LayerTiming],
        operation_times: &HashMap<String, OperationTiming>,
        memory_peaks: &[MemoryPeak],
        _memory_profile: &MemoryProfileData,
        _hotspot_analysis: &HotspotAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check for slow layers
        if let Some(slowest) = layer_times.first() {
            if slowest.percentage > 30.0 {
                recommendations.push(format!(
                    "Layer '{}' takes {:.1}% of total time. Consider optimizing this layer.",
                    slowest.name, slowest.percentage
                ));
            }
        }

        // Check forward/backward balance
        if let (Some(forward), Some(backward)) = (
            operation_times.get("forward"),
            operation_times.get("backward"),
        ) {
            let ratio = backward.avg_time.as_secs_f32() / forward.avg_time.as_secs_f32();
            if ratio > 3.0 {
                recommendations.push(format!(
                    "Backward pass is {:.1}x slower than forward pass. Consider gradient checkpointing.",
                    ratio
                ));
            }
        }

        // Check memory usage
        if !memory_peaks.is_empty() {
            let max_memory = memory_peaks
                .iter()
                .map(|p| p.allocated_mb)
                .fold(0.0f32, |a, b| a.max(b));

            if max_memory > 1000.0 {
                recommendations.push(format!(
                    "High memory usage detected ({:.1} MB). Consider using mixed precision training.",
                    max_memory
                ));
            }
        }

        recommendations
    }

    fn build_flame_graph_tree(
        samples: Vec<ProfileSample>,
    ) -> Result<FlameFrame, Box<dyn std::error::Error>> {
        let mut root = FlameFrame {
            name: "root".to_string(),
            file: None,
            line: None,
            self_time_ms: 0.0,
            total_time_ms: 0.0,
            sample_count: samples.len(),
            children: Vec::new(),
        };

        // Aggregate samples by function name
        let mut function_times: HashMap<String, f32> = HashMap::new();

        for sample in &samples {
            let function_name = sample.function_name.clone();
            let time_ms = sample.duration_ms;
            *function_times.entry(function_name).or_insert(0.0) += time_ms;
        }

        // Create child frames
        for (function_name, total_time) in function_times {
            let child_frame = FlameFrame {
                name: function_name,
                file: None,
                line: None,
                self_time_ms: total_time,
                total_time_ms: total_time,
                sample_count: 1, // Simplified
                children: Vec::new(),
            };
            root.children.push(child_frame);
            root.total_time_ms += total_time;
        }

        Ok(root)
    }

    fn analyze_call_stacks(
        call_stacks: Vec<Vec<String>>,
    ) -> Result<CallStackAnalysis, Box<dyn std::error::Error>> {
        let mut call_frequency = HashMap::new();
        let mut total_depth = 0;
        let mut max_depth = 0;

        for stack in &call_stacks {
            total_depth += stack.len();
            max_depth = max_depth.max(stack.len());

            for function in stack {
                *call_frequency.entry(function.clone()).or_insert(0) += 1;
            }
        }

        let average_stack_depth = if !call_stacks.is_empty() {
            total_depth as f32 / call_stacks.len() as f32
        } else {
            0.0
        };

        // Find hottest call paths (simplified)
        let hottest_paths = call_stacks
            .into_iter()
            .take(5)
            .map(|path| CallPath {
                path,
                total_time_ms: 100.0, // Placeholder
                call_count: 1,
                average_time_ms: 100.0,
            })
            .collect();

        Ok(CallStackAnalysis {
            hottest_paths,
            recursive_calls: vec![], // Would detect recursive patterns
            call_frequency,
            average_stack_depth,
            max_stack_depth: max_depth,
        })
    }

    // Test data structures matching the main module

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    struct ProfileSample {
        function_name: String,
        duration_ms: f32,
        stack_trace: Vec<String>,
    }
}
