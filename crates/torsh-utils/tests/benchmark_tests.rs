//! Comprehensive tests for benchmarking functionality

use std::collections::HashMap;
use std::time::Duration;
use torsh_utils::benchmark::*;
use torsh_utils::mobile_optimizer::{CpuInfo, MemoryInfo, MobilePlatform, PlatformBenchmarkInfo};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test basic benchmark configuration
    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();

        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.benchmark_iterations, 100);
        assert_eq!(config.batch_sizes, vec![1, 8, 16, 32]);
        assert_eq!(config.input_shapes, vec![vec![3, 224, 224]]);
        assert!(config.profile_memory);
        assert!(config.profile_backward);
    }

    /// Test mobile benchmark configuration
    #[test]
    fn test_mobile_benchmark_config() {
        let platform_info = PlatformBenchmarkInfo {
            platform: MobilePlatform::iOS {
                chip: "A15".to_string(),
                neural_engine: true,
            },
            device_model: "iPhone 13".to_string(),
            os_version: "15.0".to_string(),
            cpu_info: CpuInfo {
                cores_performance: 2,
                cores_efficiency: 4,
                max_frequency_ghz: 3.23,
                cache_l1_kb: 128,
                cache_l2_kb: 12288,
                cache_l3_kb: None,
            },
            memory_info: MemoryInfo {
                total_mb: 6144,
                bandwidth_gb_s: 68.25,
                memory_type: "LPDDR5".to_string(),
            },
            thermal_design_power: Some(15.0),
        };

        let mobile_config = MobileBenchmarkConfig {
            platform_info,
            monitor_thermal: true,
            measure_power: true,
            test_frequency_scaling: false,
            test_memory_pressure: true,
            stress_test_duration_minutes: Some(5),
            latency_thresholds: LatencyThresholds::default(),
            energy_targets: Some(EnergyTargets::default()),
        };

        assert!(mobile_config.monitor_thermal);
        assert!(mobile_config.measure_power);
        assert_eq!(mobile_config.stress_test_duration_minutes, Some(5));
    }

    /// Test latency thresholds
    #[test]
    fn test_latency_thresholds() {
        let thresholds = LatencyThresholds::default();

        assert_eq!(thresholds.realtime_ms, 16.67); // 60 FPS
        assert_eq!(thresholds.interactive_ms, 100.0);
        assert_eq!(thresholds.batch_ms, 1000.0);

        // Test custom thresholds
        let custom_thresholds = LatencyThresholds {
            realtime_ms: 8.33, // 120 FPS
            interactive_ms: 50.0,
            batch_ms: 500.0,
        };

        assert_eq!(custom_thresholds.realtime_ms, 8.33);
        assert!(custom_thresholds.interactive_ms < thresholds.interactive_ms);
    }

    /// Test energy targets
    #[test]
    fn test_energy_targets() {
        let targets = EnergyTargets::default();

        assert_eq!(targets.inferences_per_joule, 1000.0);
        assert_eq!(targets.max_power_watts, 5.0);
        assert_eq!(targets.target_battery_hours, 8.0);

        // Test efficiency calculation
        let power_consumption_mw = 2000.0; // 2 watts
        let inference_rate_fps = 60.0;
        let efficiency = inference_rate_fps / (power_consumption_mw / 1000.0);

        assert_eq!(efficiency, 30.0); // 30 inferences per watt
    }

    /// Test benchmark result structure
    #[test]
    fn test_benchmark_result() {
        let mut results_by_batch = HashMap::new();

        results_by_batch.insert(
            1,
            BatchResult {
                batch_size: 1,
                forward_time: TimingStats {
                    mean: Duration::from_millis(10),
                    std: Duration::from_millis(1),
                    min: Duration::from_millis(8),
                    max: Duration::from_millis(12),
                    median: Duration::from_millis(10),
                    p95: Duration::from_millis(11),
                    p99: Duration::from_millis(12),
                },
                backward_time: Some(TimingStats {
                    mean: Duration::from_millis(15),
                    std: Duration::from_millis(2),
                    min: Duration::from_millis(12),
                    max: Duration::from_millis(18),
                    median: Duration::from_millis(15),
                    p95: Duration::from_millis(17),
                    p99: Duration::from_millis(18),
                }),
                total_time: TimingStats {
                    mean: Duration::from_millis(25),
                    std: Duration::from_millis(2),
                    min: Duration::from_millis(22),
                    max: Duration::from_millis(28),
                    median: Duration::from_millis(25),
                    p95: Duration::from_millis(27),
                    p99: Duration::from_millis(28),
                },
                throughput: 40.0, // 1000ms / 25ms = 40 samples/sec
                memory_stats: Some(MemoryStats {
                    peak_allocated_mb: 128.0,
                    peak_reserved_mb: 256.0,
                    avg_allocated_mb: 96.0,
                }),
            },
        );

        let summary = BenchmarkSummary {
            best_batch_size: 1,
            best_throughput: 40.0,
            optimal_memory_batch: 1,
            recommendations: vec![
                "Batch size 1 provides optimal throughput".to_string(),
                "Memory usage is within acceptable limits".to_string(),
            ],
        };

        let benchmark_result = BenchmarkResult {
            model_name: "TestModel".to_string(),
            total_params: 1000000,
            results_by_batch,
            summary,
            mobile_results: None,
            validation_results: None,
        };

        assert_eq!(benchmark_result.model_name, "TestModel");
        assert_eq!(benchmark_result.total_params, 1000000);
        assert_eq!(benchmark_result.results_by_batch.len(), 1);
        assert_eq!(benchmark_result.summary.best_batch_size, 1);
    }

    /// Test validation results
    #[test]
    fn test_validation_results() {
        let platform_validation = PlatformValidationResults {
            ios_app_store_compliant: Some(true),
            android_performance_class: None,
            device_compatibility_score: 85.0,
            device_support_percentage: 95.0,
        };

        let validation_results = ValidationResults {
            meets_realtime_latency: true,
            meets_interactive_latency: true,
            meets_energy_targets: false,
            thermal_throttling_detected: false,
            memory_pressure_impact: Some(5.0), // 5% impact
            sustained_performance_degradation: Some(2.0), // 2% degradation
            platform_validation,
            recommendations: vec![
                "Energy efficiency below target. Consider model compression.".to_string(),
                "Thermal management is adequate for continuous operation.".to_string(),
            ],
        };

        assert!(validation_results.meets_realtime_latency);
        assert!(validation_results.meets_interactive_latency);
        assert!(!validation_results.meets_energy_targets);
        assert_eq!(validation_results.memory_pressure_impact, Some(5.0));
        assert_eq!(validation_results.recommendations.len(), 2);
    }

    /// Test mobile platform detection
    #[test]
    fn test_mobile_platform() {
        let ios_platform = MobilePlatform::iOS {
            chip: "A15".to_string(),
            neural_engine: true,
        };

        let android_platform = MobilePlatform::Android {
            soc: "Snapdragon 8 Gen 1".to_string(),
            npu_available: true,
        };

        let other_platform = MobilePlatform::Other("Custom SoC".to_string());

        // Test iOS platform
        match ios_platform {
            MobilePlatform::iOS {
                chip,
                neural_engine,
            } => {
                assert_eq!(chip, "A15");
                assert!(neural_engine);
            }
            _ => panic!("Expected iOS platform"),
        }

        // Test Android platform
        match android_platform {
            MobilePlatform::Android { soc, npu_available } => {
                assert_eq!(soc, "Snapdragon 8 Gen 1");
                assert!(npu_available);
            }
            _ => panic!("Expected Android platform"),
        }

        // Test other platform
        match other_platform {
            MobilePlatform::Other(name) => {
                assert_eq!(name, "Custom SoC");
            }
            _ => panic!("Expected Other platform"),
        }
    }

    /// Test CPU info structure
    #[test]
    fn test_cpu_info() {
        let cpu_info = CpuInfo {
            cores_performance: 4,
            cores_efficiency: 4,
            max_frequency_ghz: 3.0,
            cache_l1_kb: 64,
            cache_l2_kb: 256,
            cache_l3_kb: Some(8192),
        };

        assert_eq!(cpu_info.cores_performance, 4);
        assert_eq!(cpu_info.cores_efficiency, 4);
        assert_eq!(cpu_info.max_frequency_ghz, 3.0);
        assert_eq!(cpu_info.cache_l3_kb, Some(8192));

        // Test total cores calculation
        let total_cores = cpu_info.cores_performance + cpu_info.cores_efficiency;
        assert_eq!(total_cores, 8);

        // Test cache hierarchy
        assert!(cpu_info.cache_l1_kb < cpu_info.cache_l2_kb);
        assert!(cpu_info.cache_l2_kb < cpu_info.cache_l3_kb.unwrap_or(0));
    }

    /// Test memory info structure
    #[test]
    fn test_memory_info() {
        let memory_info = MemoryInfo {
            total_mb: 8192,
            bandwidth_gb_s: 68.25,
            memory_type: "LPDDR5".to_string(),
        };

        assert_eq!(memory_info.total_mb, 8192);
        assert_eq!(memory_info.bandwidth_gb_s, 68.25);
        assert_eq!(memory_info.memory_type, "LPDDR5");

        // Test memory calculations
        let total_gb = memory_info.total_mb as f32 / 1024.0;
        assert!((total_gb - 8.0).abs() < 0.1);

        let bandwidth_mb_s = memory_info.bandwidth_gb_s * 1024.0;
        assert!((bandwidth_mb_s - 69888.0).abs() < 1.0);
    }

    /// Test timing statistics calculation
    #[test]
    fn test_timing_stats() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(13),
            Duration::from_millis(14),
        ];

        let stats = calculate_timing_stats(&times);

        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(14));
        assert_eq!(stats.median, Duration::from_millis(12));

        // Check mean is reasonable
        let expected_mean = Duration::from_millis(12); // (10+11+12+13+14)/5 = 12
        assert_eq!(stats.mean, expected_mean);

        // Check percentiles
        let p95_expected = Duration::from_millis(14); // 95th percentile
        assert_eq!(stats.p95, p95_expected);
    }

    /// Test memory statistics calculation
    #[test]
    fn test_memory_stats() {
        let samples = vec![
            (100.0, 150.0), // (allocated, reserved)
            (120.0, 150.0),
            (110.0, 150.0),
            (130.0, 200.0),
            (105.0, 160.0),
        ];

        let stats = calculate_memory_stats(&samples);

        assert_eq!(stats.peak_allocated_mb, 130.0);
        assert_eq!(stats.peak_reserved_mb, 200.0);

        let expected_avg = (100.0 + 120.0 + 110.0 + 130.0 + 105.0) / 5.0;
        assert_eq!(stats.avg_allocated_mb, expected_avg);
    }

    /// Test benchmark summary generation
    #[test]
    fn test_benchmark_summary() {
        let mut results = HashMap::new();

        results.insert(
            1,
            BatchResult {
                batch_size: 1,
                forward_time: create_timing_stats(10),
                backward_time: Some(create_timing_stats(15)),
                total_time: create_timing_stats(25),
                throughput: 40.0,
                memory_stats: Some(MemoryStats {
                    peak_allocated_mb: 100.0,
                    peak_reserved_mb: 150.0,
                    avg_allocated_mb: 80.0,
                }),
            },
        );

        results.insert(
            8,
            BatchResult {
                batch_size: 8,
                forward_time: create_timing_stats(80),
                backward_time: Some(create_timing_stats(120)),
                total_time: create_timing_stats(200),
                throughput: 40.0, // 8 samples in 200ms = 40 samples/sec
                memory_stats: Some(MemoryStats {
                    peak_allocated_mb: 800.0,
                    peak_reserved_mb: 1200.0,
                    avg_allocated_mb: 640.0,
                }),
            },
        );

        let summary = generate_summary(&results);

        // Both batch sizes have same throughput, so either could be "best"
        assert!(summary.best_batch_size == 1 || summary.best_batch_size == 8);
        assert_eq!(summary.best_throughput, 40.0);

        // Both batch sizes have same memory efficiency (100MB per sample), so either could be optimal
        assert!(summary.optimal_memory_batch == 1 || summary.optimal_memory_batch == 8);

        assert!(!summary.recommendations.is_empty());
    }

    /// Test performance regression detection
    #[test]
    fn test_performance_regression_detection() {
        let baseline_result = BatchResult {
            batch_size: 1,
            forward_time: create_timing_stats(10),
            backward_time: Some(create_timing_stats(15)),
            total_time: create_timing_stats(25),
            throughput: 40.0,
            memory_stats: Some(MemoryStats {
                peak_allocated_mb: 100.0,
                peak_reserved_mb: 150.0,
                avg_allocated_mb: 80.0,
            }),
        };

        let current_result = BatchResult {
            batch_size: 1,
            forward_time: create_timing_stats(12), // 20% slower
            backward_time: Some(create_timing_stats(18)), // 20% slower
            total_time: create_timing_stats(30),   // 20% slower
            throughput: 33.33,                     // Correspondingly lower
            memory_stats: Some(MemoryStats {
                peak_allocated_mb: 120.0, // 20% more memory
                peak_reserved_mb: 180.0,
                avg_allocated_mb: 96.0,
            }),
        };

        let regression_threshold = 0.1; // 10%

        // Check timing regression
        let timing_regression = (current_result.total_time.mean.as_millis() as f32
            - baseline_result.total_time.mean.as_millis() as f32)
            / baseline_result.total_time.mean.as_millis() as f32;

        assert!(timing_regression > regression_threshold);

        // Check memory regression
        let memory_regression = (current_result
            .memory_stats
            .as_ref()
            .unwrap()
            .peak_allocated_mb
            - baseline_result
                .memory_stats
                .as_ref()
                .unwrap()
                .peak_allocated_mb)
            / baseline_result
                .memory_stats
                .as_ref()
                .unwrap()
                .peak_allocated_mb;

        assert!(memory_regression > regression_threshold);

        // Check throughput regression
        let throughput_regression =
            (baseline_result.throughput - current_result.throughput) / baseline_result.throughput;

        assert!(throughput_regression > regression_threshold);
    }

    /// Test stress testing scenarios
    #[test]
    fn test_stress_testing() {
        let stress_config = StressTestConfig {
            duration_minutes: 5,
            thermal_monitoring: true,
            memory_pressure: true,
            frequency_scaling: false,
            continuous_inference: true,
            target_utilization: 0.8,
        };

        assert_eq!(stress_config.duration_minutes, 5);
        assert!(stress_config.thermal_monitoring);
        assert!(stress_config.memory_pressure);
        assert_eq!(stress_config.target_utilization, 0.8);

        // Simulate stress test results
        let stress_results = StressTestResults {
            initial_throughput: 100.0,
            final_throughput: 95.0, // 5% degradation
            thermal_throttling_events: 2,
            memory_pressure_events: 1,
            stability_score: 0.95,
            sustained_performance_ratio: 0.95,
        };

        assert_eq!(stress_results.thermal_throttling_events, 2);
        assert_eq!(stress_results.memory_pressure_events, 1);

        let performance_degradation = (stress_results.initial_throughput
            - stress_results.final_throughput)
            / stress_results.initial_throughput;
        assert!((performance_degradation - 0.05).abs() < 0.01);
    }

    // Helper functions

    fn create_timing_stats(mean_ms: u64) -> TimingStats {
        let mean = Duration::from_millis(mean_ms);
        let std = Duration::from_millis(mean_ms / 20); // 5% std deviation
        let min = Duration::from_millis((mean_ms as f32 * 0.8) as u64);
        let max = Duration::from_millis((mean_ms as f32 * 1.2) as u64);

        TimingStats {
            mean,
            std,
            min,
            max,
            median: mean,
            p95: Duration::from_millis((mean_ms as f32 * 1.1) as u64),
            p99: Duration::from_millis((mean_ms as f32 * 1.15) as u64),
        }
    }

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

    fn calculate_memory_stats(samples: &[(f32, f32)]) -> MemoryStats {
        let peak_allocated = samples.iter().map(|(a, _)| *a).fold(0.0f32, f32::max);
        let peak_reserved = samples.iter().map(|(_, r)| *r).fold(0.0f32, f32::max);
        let avg_allocated = samples.iter().map(|(a, _)| *a).sum::<f32>() / samples.len() as f32;

        MemoryStats {
            peak_allocated_mb: peak_allocated,
            peak_reserved_mb: peak_reserved,
            avg_allocated_mb: avg_allocated,
        }
    }

    fn generate_summary(results: &HashMap<usize, BatchResult>) -> BenchmarkSummary {
        let mut best_throughput = 0.0;
        let mut best_batch_size = 0;
        let mut optimal_memory_batch = 0;
        let mut min_memory_per_sample = f32::INFINITY;

        for (batch_size, result) in results {
            if result.throughput > best_throughput {
                best_throughput = result.throughput;
                best_batch_size = *batch_size;
            }

            if let Some(mem) = &result.memory_stats {
                let memory_per_sample = mem.peak_allocated_mb / *batch_size as f32;
                if memory_per_sample < min_memory_per_sample {
                    min_memory_per_sample = memory_per_sample;
                    optimal_memory_batch = *batch_size;
                }
            }
        }

        let mut recommendations = Vec::new();

        recommendations.push(format!(
            "Best throughput: {:.1} samples/sec at batch size {}",
            best_throughput, best_batch_size
        ));

        if optimal_memory_batch > 0 {
            recommendations.push(format!(
                "Most memory efficient: batch size {} ({:.1} MB/sample)",
                optimal_memory_batch, min_memory_per_sample
            ));
        }

        BenchmarkSummary {
            best_batch_size,
            best_throughput,
            optimal_memory_batch,
            recommendations,
        }
    }

    // Test data structures

    #[derive(Debug)]
    #[allow(dead_code)]
    struct StressTestConfig {
        duration_minutes: u32,
        thermal_monitoring: bool,
        memory_pressure: bool,
        frequency_scaling: bool,
        continuous_inference: bool,
        target_utilization: f32,
    }

    #[derive(Debug)]
    #[allow(dead_code)]
    struct StressTestResults {
        initial_throughput: f32,
        final_throughput: f32,
        thermal_throttling_events: u32,
        memory_pressure_events: u32,
        stability_score: f32,
        sustained_performance_ratio: f32,
    }
}
