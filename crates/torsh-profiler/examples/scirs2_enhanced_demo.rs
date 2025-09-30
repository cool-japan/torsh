//! SciRS2-Enhanced Profiling Demo
//!
//! This example demonstrates the SciRS2-enhanced performance analysis capabilities
//! of the torsh-profiler crate, following the SciRS2 integration policy.

use std::{thread, time::Duration};
use torsh_profiler::{
    export_global_json, profile_scope, scirs2_enhanced_performance_analysis, start_profiling,
    stop_profiling, ProfileEvent,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SciRS2-Enhanced Profiling Demo");
    println!("==================================");
    println!();

    // Start profiling
    start_profiling();

    // Simulate some computational work
    simulate_workload();

    stop_profiling();

    // Create sample events for SciRS2 analysis
    let sample_events = create_sample_events();

    println!("📊 Running SciRS2-Enhanced Performance Analysis...");
    println!();

    // ✅ Demonstrate SciRS2-enhanced analysis
    match scirs2_enhanced_performance_analysis(&sample_events) {
        Ok(analysis_result) => {
            display_scirs2_results(&analysis_result);
        }
        Err(e) => {
            eprintln!("Analysis failed: {}", e);
        }
    }

    // Export results
    println!("💾 Exporting profiling data...");
    if let Err(e) = export_global_json("/tmp/scirs2_enhanced_demo.json") {
        eprintln!("Export failed: {}", e);
    } else {
        println!("   ✅ Exported to /tmp/scirs2_enhanced_demo.json");
    }

    println!();
    println!("🎯 SciRS2-Enhanced Analysis Complete!");
    println!("   This analysis used SciRS2's random number generation");
    println!("   instead of direct rand usage, following the SciRS2 policy.");

    Ok(())
}

fn simulate_workload() {
    // Simulate various computational patterns
    {
        profile_scope!("initialization");
        thread::sleep(Duration::from_millis(10));
    }

    for i in 0..5 {
        let scope_name = format!("computation_{}", i);
        profile_scope!(&scope_name);
        thread::sleep(Duration::from_millis(5 + i * 2));
    }

    {
        profile_scope!("cleanup");
        thread::sleep(Duration::from_millis(8));
    }
}

fn create_sample_events() -> Vec<ProfileEvent> {
    vec![
        ProfileEvent {
            name: "matrix_multiplication".to_string(),
            category: "compute".to_string(),
            start_us: 0,
            duration_us: 1500,
            thread_id: 1,
            operation_count: Some(1000000),
            flops: Some(2000000000),
            bytes_transferred: Some(8192),
            stack_trace: None,
        },
        ProfileEvent {
            name: "memory_allocation".to_string(),
            category: "memory".to_string(),
            start_us: 2000,
            duration_us: 800,
            thread_id: 1,
            operation_count: Some(100),
            flops: Some(0),
            bytes_transferred: Some(1048576),
            stack_trace: None,
        },
        ProfileEvent {
            name: "data_transfer".to_string(),
            category: "io".to_string(),
            start_us: 3000,
            duration_us: 2200,
            thread_id: 2,
            operation_count: Some(50),
            flops: Some(0),
            bytes_transferred: Some(4194304),
            stack_trace: None,
        },
        ProfileEvent {
            name: "simd_operations".to_string(),
            category: "compute".to_string(),
            start_us: 5500,
            duration_us: 450,
            thread_id: 1,
            operation_count: Some(500000),
            flops: Some(1000000000),
            bytes_transferred: Some(2048),
            stack_trace: None,
        },
    ]
}

fn display_scirs2_results(result: &torsh_profiler::SciRS2AnalysisResult) {
    println!("🔬 SciRS2 Analysis Results");
    println!("-------------------------");

    // Display SIMD-enhanced statistics
    println!("📈 SIMD-Enhanced Statistics:");
    println!(
        "   Mean duration: {:.2}μs",
        result.simd_statistics.simd_accelerated_mean
    );
    println!(
        "   Variance: {:.2}",
        result.simd_statistics.simd_accelerated_variance
    );
    println!(
        "   Skewness: {:.3}",
        result.simd_statistics.simd_accelerated_skewness
    );
    println!(
        "   Kurtosis: {:.3}",
        result.simd_statistics.simd_accelerated_kurtosis
    );
    println!(
        "   Vectorization efficiency: {:.1}%",
        result.simd_statistics.vectorization_efficiency * 100.0
    );
    println!(
        "   Cache hit ratio: {:.1}%",
        result.simd_statistics.cache_hit_ratio * 100.0
    );
    println!();

    // Display parallel analysis results
    println!("🚀 Parallel Processing Analysis:");
    println!(
        "   Parallel efficiency: {:.1}%",
        result.parallel_analysis.parallel_efficiency * 100.0
    );
    println!(
        "   Load balance score: {:.1}%",
        result.parallel_analysis.load_balance_score * 100.0
    );
    println!(
        "   Memory efficiency: {:.1}%",
        result.parallel_analysis.memory_efficiency * 100.0
    );
    println!(
        "   CPU utilization: {:.1}%",
        result.parallel_analysis.cpu_utilization * 100.0
    );
    println!(
        "   Chunks processed: {}",
        result.parallel_analysis.chunks_processed
    );
    println!();

    // Display benchmark metrics
    println!("⏱️ Benchmark Metrics:");
    println!(
        "   Analysis duration: {}ns",
        result.benchmark_metrics.analysis_duration_ns
    );
    println!(
        "   Throughput: {:.0} ops/sec",
        result.benchmark_metrics.throughput_ops_per_sec
    );
    println!(
        "   Peak memory: {:.2}MB",
        result.benchmark_metrics.memory_usage_peak_mb
    );
    println!(
        "   CPU cycles: {}",
        result.benchmark_metrics.cpu_cycles_consumed
    );
    println!();

    // Display performance score
    println!("🎯 Performance Score: {:.1}/100", result.performance_score);
    println!();

    // Display SciRS2 optimization recommendations
    println!("💡 SciRS2 Optimization Recommendations:");
    for (i, optimization) in result.optimization_recommendations.iter().enumerate() {
        println!(
            "   {}. {} (+{:.0}% improvement)",
            i + 1,
            optimization.optimization_type,
            optimization.expected_improvement_percent
        );
        println!("      Description: {}", optimization.description);
        println!(
            "      Complexity: {}",
            optimization.implementation_complexity
        );
        println!(
            "      SciRS2 Features: {:?}",
            optimization.scirs2_features_used
        );
        println!();
    }
}
