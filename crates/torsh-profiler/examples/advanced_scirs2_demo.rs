//! Advanced SciRS2-Enhanced Profiling Demo
//!
//! This example demonstrates the comprehensive SciRS2-enhanced profiling capabilities
//! of the torsh-profiler crate, showcasing the full utilization of SciRS2-core features
//! including advanced metrics, benchmarking, validation, and comprehensive profiling.

use std::{thread, time::Duration};
use torsh_profiler::{
    benchmark_scirs2, collect_scirs2_metrics, profile_scirs2_comprehensive,
    profile_scirs2_sampling, AdvancedProfilingConfig, PerformanceTargets, SamplingStrategy,
    ScirS2EnhancedProfiler, ValidationLevel,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced SciRS2-Enhanced Profiling Demo");
    println!("==========================================");
    println!("Showcasing comprehensive SciRS2-core integration");
    println!();

    // Initialize the enhanced profiler with comprehensive SciRS2 features
    println!("🔧 Initializing SciRS2-Enhanced Profiler...");
    let mut profiler = ScirS2EnhancedProfiler::new()?;
    profiler.start_enhanced_profiling()?;
    println!("   ✅ Advanced profiler initialized with comprehensive SciRS2 features");
    println!();

    // Demonstrate comprehensive metrics profiling
    println!("📊 Comprehensive Metrics Profiling Demo:");
    let result = profile_scirs2_comprehensive!(profiler, "matrix_computation", {
        simulate_matrix_computation();
        "Matrix computation completed"
    })?;
    println!("   ✅ Result: {}", result);
    println!();

    // Demonstrate sampling-based profiling with SciRS2 random generation
    println!("🎲 SciRS2 Sampling-Based Profiling:");
    for i in 0..5 {
        let sample_rate = 0.8; // 80% sampling rate
        let operation_name = format!("sampled_operation_{}", i);

        match profile_scirs2_sampling!(profiler, &operation_name, sample_rate, {
            simulate_variable_workload(i);
            format!("Sampled operation {} result", i)
        })? {
            Some(result) => println!("   ✅ Sampled: {}", result),
            None => println!("   ⏭️  Operation {} skipped by sampling", i),
        }
    }
    println!();

    // Demonstrate comprehensive benchmarking
    println!("⏱️  Advanced SciRS2 Benchmarking:");
    let benchmark_operation = || {
        simulate_cpu_intensive_task();
    };

    let benchmark_results = benchmark_scirs2!(profiler, "cpu_benchmark", 10, benchmark_operation)?;
    display_benchmark_results(&benchmark_results);
    println!();

    // Demonstrate validation with SciRS2 constants
    println!("🔍 SciRS2 Validation and Constants Demo:");
    let test_values = vec![100.0, 3.14159, f64::NAN, f64::INFINITY, -42.5];

    for (i, value) in test_values.iter().enumerate() {
        let operation_name = format!("validation_test_{}", i);
        let analysis = profiler.analyze_with_constants(&operation_name, *value);

        println!(
            "   📈 Analysis for {}: raw={:.3}, math_norm={:.6}, phys_norm={:.3e}",
            operation_name,
            analysis.raw_value,
            analysis.math_normalized,
            analysis.physics_normalized
        );
    }
    println!();

    // Demonstrate memory-intensive operations with enhanced tracking
    println!("💾 Memory-Enhanced Profiling:");
    {
        let result = profiler.profile_with_comprehensive_metrics("memory_simulation", || {
            simulate_memory_intensive_task()
        })?;
        println!("   ✅ Memory simulation result: {}", result);
    }
    println!();

    // Collect and display comprehensive metrics
    println!("📊 Comprehensive Metrics Summary:");
    let metrics = collect_scirs2_metrics!(profiler);
    display_metrics_summary(&metrics);
    println!();

    // Demonstrate advanced configuration
    println!("⚙️  Advanced Profiling Configuration:");
    let advanced_config = AdvancedProfilingConfig {
        enable_simd_acceleration: true,
        enable_parallel_processing: true,
        enable_gpu_acceleration: false,
        enable_memory_optimization: true,
        enable_advanced_metrics: true,
        sampling_strategy: SamplingStrategy::Adaptive(0.1, 0.9),
        validation_level: ValidationLevel::Comprehensive,
        performance_targets: PerformanceTargets {
            max_latency_ns: Some(1_000_000), // 1ms
            min_throughput_ops_per_sec: Some(1000.0),
            max_memory_usage_bytes: Some(100_000_000), // 100MB
            target_cpu_utilization: Some(0.8),
        },
    };
    println!("   ✅ Configuration: {:?}", advanced_config);
    println!();

    // Export comprehensive profiling data
    println!("💾 Exporting Enhanced SciRS2 Data:");
    let export_path = std::env::temp_dir().join("advanced_scirs2_demo.json");
    let export_str = export_path.display().to_string();
    profiler.export_scirs2_format(&export_str)?;
    println!("   ✅ Enhanced data exported to {}", export_str);
    println!();

    println!("🎯 Advanced SciRS2-Enhanced Profiling Complete!");
    println!("   This demo showcased:");
    println!("   • Comprehensive metrics collection (timers, counters, gauges, histograms)");
    println!("   • Advanced benchmarking with statistical analysis");
    println!("   • SciRS2 random sampling for performance optimization");
    println!("   • Input validation using SciRS2 validation features");
    println!("   • Mathematical normalization with SciRS2 constants");
    println!("   • Memory-aware profiling with SciRS2 memory management");
    println!("   • Advanced configuration and performance targets");
    println!();
    println!("   All features follow the SciRS2 integration policy for");
    println!("   maximum utilization of SciRS2-core capabilities!");

    Ok(())
}

fn simulate_matrix_computation() {
    // Simulate matrix computation workload
    thread::sleep(Duration::from_millis(5));

    // Simulate some mathematical operations
    let mut sum = 0.0;
    for i in 0..1000 {
        sum += (i as f64).sin() * (i as f64).cos();
    }

    // Prevent compiler optimization
    std::hint::black_box(sum);
}

fn simulate_variable_workload(iteration: usize) {
    // Variable workload based on iteration
    let sleep_duration = Duration::from_millis(1 + (iteration as u64 * 2));
    thread::sleep(sleep_duration);

    // Some computation
    let mut result = 1.0;
    for i in 1..=iteration * 100 {
        result *= (i as f64).sqrt();
    }

    std::hint::black_box(result);
}

fn simulate_cpu_intensive_task() {
    // CPU-intensive computation
    let mut factorial = 1u64;
    for i in 1..=10 {
        factorial = factorial.saturating_mul(i);
    }

    // Some floating point operations
    let mut result = factorial as f64;
    for _ in 0..100 {
        result = result.sqrt().sin().exp().ln();
    }

    std::hint::black_box(result);
}

fn simulate_memory_intensive_task() -> String {
    // Allocate and manipulate memory
    let mut data = Vec::with_capacity(1000);

    for i in 0..1000 {
        data.push(format!("data_item_{}_with_some_content", i));
    }

    // Process the data
    let processed: Vec<String> = data.into_iter().map(|s| s.to_uppercase()).collect();

    format!("Processed {} items", processed.len())
}

fn display_benchmark_results(results: &torsh_profiler::BenchmarkResults) {
    println!("   📊 Benchmark Results for '{}':", results.benchmark_name);
    println!("      Iterations: {}", results.iterations);
    println!("      Mean duration: {:.2} ns", results.mean_duration_ns);
    println!("      Std deviation: {:.2} ns", results.std_deviation_ns);
    println!("      Min duration: {:.2} ns", results.min_duration_ns);
    println!("      Max duration: {:.2} ns", results.max_duration_ns);
    println!("      Total duration: {:.2} ns", results.total_duration_ns);
    println!(
        "      Throughput: {:.2} ops/sec",
        results.throughput_ops_per_sec
    );
}

fn display_metrics_summary(metrics: &torsh_profiler::MetricsSummary) {
    println!("   📈 Metrics Summary:");
    println!("      Total operations: {}", metrics.total_operations);
    println!("      Active timers: {}", metrics.active_timers.len());
    println!("      Counters tracked: {}", metrics.counter_values.len());
    println!("      Gauges tracked: {}", metrics.gauge_values.len());
    println!(
        "      Histograms tracked: {}",
        metrics.histogram_stats.len()
    );

    // Display top counters
    if !metrics.counter_values.is_empty() {
        println!("      Top counters:");
        let mut counters: Vec<_> = metrics.counter_values.iter().collect();
        counters.sort_by_key(|(_, &count)| std::cmp::Reverse(count));

        for (name, count) in counters.iter().take(3) {
            println!("        • {}: {} times", name, count);
        }
    }

    // Display histogram statistics
    if !metrics.histogram_stats.is_empty() {
        println!("      Histogram statistics:");
        for (name, stats) in metrics.histogram_stats.iter() {
            println!(
                "        • {}: {} samples, mean={:.2}",
                name, stats.count, stats.mean
            );
        }
    }
}
