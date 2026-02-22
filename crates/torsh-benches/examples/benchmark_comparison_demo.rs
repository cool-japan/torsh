//! Comprehensive example demonstrating benchmark comparison features
//!
//! This example shows how to compare benchmark results from different implementations,
//! configurations, or optimization passes to identify improvements and regressions.

use std::collections::HashMap;
use torsh_benches::benchmark_comparison::{BenchmarkComparator, ComparisonVerdict};
use torsh_benches::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Benchmark Comparison Demo ===\n");

    // Create a comparator
    let mut comparator = BenchmarkComparator::new();

    println!("1. Simulating baseline benchmark results...\n");
    let baseline_results = create_baseline_results();
    display_results("Baseline (Unoptimized)", &baseline_results);

    println!("\n2. Simulating optimized benchmark results...\n");
    let optimized_results = create_optimized_results();
    display_results("Optimized (SIMD + Parallel)", &optimized_results);

    // Add results to comparator
    comparator.add_baseline(&baseline_results, "baseline_v1.0");
    comparator.add_candidate(&optimized_results, "optimized_v2.0");

    println!("\n3. Performing comparison analysis...\n");
    let comparisons = comparator.compare();

    // Display detailed comparison
    println!("Detailed Comparison Results:");
    println!("{:-<100}", "");
    for comparison in comparisons {
        let verdict_symbol = match comparison.verdict {
            ComparisonVerdict::MajorImprovement => "🚀",
            ComparisonVerdict::Improvement => "✅",
            ComparisonVerdict::NoChange => "➖",
            ComparisonVerdict::Regression => "⚠️",
            ComparisonVerdict::MajorRegression => "🔴",
        };

        println!(
            "{} {} | Speedup: {:.2}x | Change: {:+.1}% | Verdict: {:?}",
            verdict_symbol,
            comparison.benchmark_name,
            comparison.speedup,
            comparison.improvement_percentage,
            comparison.verdict
        );
    }

    // Get summary statistics
    println!("\n4. Summary Statistics:\n");
    let summary = comparator.get_summary();
    println!("Total Comparisons: {}", summary.total_comparisons);
    println!("Improvements: {}", summary.improvements);
    println!("Regressions: {}", summary.regressions);
    println!("Major Improvements: {}", summary.major_improvements);
    println!("Major Regressions: {}", summary.major_regressions);
    println!("Average Speedup: {:.2}x", summary.average_speedup);
    println!(
        "Geometric Mean Speedup: {:.2}x",
        summary.geometric_mean_speedup
    );

    // Generate markdown report
    println!("\n5. Generating Markdown Report...\n");
    let markdown_report = comparator.generate_markdown_report();
    println!("{}", markdown_report);

    // Export to JSON
    println!("\n6. Exporting to JSON...\n");
    let json_export = comparator.export_json()?;
    println!("JSON export size: {} bytes", json_export.len());
    println!("JSON preview (first 500 chars):");
    println!("{}", &json_export[..500.min(json_export.len())]);

    // Save reports to files
    std::fs::write("comparison_report.md", markdown_report)?;
    std::fs::write("comparison_results.json", json_export)?;
    println!("\n✅ Reports saved to:");
    println!("  - comparison_report.md");
    println!("  - comparison_results.json");

    Ok(())
}

/// Create baseline (unoptimized) benchmark results
fn create_baseline_results() -> Vec<BenchResult> {
    vec![
        create_benchmark_result("matmul", 256, 1000.0, 50.0),
        create_benchmark_result("matmul", 512, 8000.0, 400.0),
        create_benchmark_result("matmul", 1024, 64000.0, 3200.0),
        create_benchmark_result("conv2d", 64, 500.0, 25.0),
        create_benchmark_result("conv2d", 128, 4000.0, 200.0),
        create_benchmark_result("conv2d", 256, 32000.0, 1600.0),
        create_benchmark_result("reduction_sum", 1024, 100.0, 5.0),
        create_benchmark_result("reduction_sum", 4096, 400.0, 20.0),
        create_benchmark_result("reduction_sum", 16384, 1600.0, 80.0),
    ]
}

/// Create optimized benchmark results
fn create_optimized_results() -> Vec<BenchResult> {
    vec![
        // matmul: 25% improvement
        create_benchmark_result("matmul", 256, 750.0, 37.5),
        create_benchmark_result("matmul", 512, 6000.0, 300.0),
        create_benchmark_result("matmul", 1024, 48000.0, 2400.0),
        // conv2d: 40% improvement (major)
        create_benchmark_result("conv2d", 64, 300.0, 15.0),
        create_benchmark_result("conv2d", 128, 2400.0, 120.0),
        create_benchmark_result("conv2d", 256, 19200.0, 960.0),
        // reduction_sum: 15% improvement
        create_benchmark_result("reduction_sum", 1024, 85.0, 4.25),
        create_benchmark_result("reduction_sum", 4096, 340.0, 17.0),
        create_benchmark_result("reduction_sum", 16384, 1360.0, 68.0),
    ]
}

/// Helper function to create a benchmark result
fn create_benchmark_result(name: &str, size: usize, mean_ns: f64, std_ns: f64) -> BenchResult {
    use torsh_core::dtype::DType;

    let throughput = if mean_ns > 0.0 {
        Some(1_000_000_000.0 / mean_ns)
    } else {
        Some(0.0)
    };

    BenchResult {
        name: name.to_string(),
        size,
        dtype: DType::F32,
        mean_time_ns: mean_ns,
        std_dev_ns: std_ns,
        throughput,
        memory_usage: Some(size * size * 4), // Assuming f32
        peak_memory: Some(size * size * 8),  // Peak memory estimate
        metrics: HashMap::new(),
    }
}

/// Display benchmark results in a formatted table
fn display_results(label: &str, results: &[BenchResult]) {
    println!("{}:", label);
    println!("{:-<100}", "");
    println!(
        "{:<20} | {:>10} | {:>15} | {:>20}",
        "Benchmark", "Size", "Mean Time", "Throughput"
    );
    println!("{:-<100}", "");

    for result in results {
        let time_str = if result.mean_time_ns < 1000.0 {
            format!("{:.1} ns", result.mean_time_ns)
        } else if result.mean_time_ns < 1_000_000.0 {
            format!("{:.1} μs", result.mean_time_ns / 1000.0)
        } else if result.mean_time_ns < 1_000_000_000.0 {
            format!("{:.1} ms", result.mean_time_ns / 1_000_000.0)
        } else {
            format!("{:.1} s", result.mean_time_ns / 1_000_000_000.0)
        };

        let throughput_ops_per_sec = result.throughput.unwrap_or(0.0);
        let throughput_str = if throughput_ops_per_sec < 1_000.0 {
            format!("{:.1} ops/s", throughput_ops_per_sec)
        } else if throughput_ops_per_sec < 1_000_000.0 {
            format!("{:.1} Kops/s", throughput_ops_per_sec / 1000.0)
        } else if throughput_ops_per_sec < 1_000_000_000.0 {
            format!("{:.1} Mops/s", throughput_ops_per_sec / 1_000_000.0)
        } else {
            format!("{:.1} Gops/s", throughput_ops_per_sec / 1_000_000_000.0)
        };

        println!(
            "{:<20} | {:>10} | {:>15} | {:>20}",
            result.name, result.size, time_str, throughput_str
        );
    }
}
