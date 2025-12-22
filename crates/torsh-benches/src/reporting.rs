//! General benchmark reporting and analysis utilities for ToRSh
//!
//! This module provides comprehensive reporting functions that aggregate
//! results from various benchmark suites and generate unified analysis reports.

use crate::core::{ComparisonRunner, PerformanceAnalyzer};

// Re-export the benchmark runner functions from ndarray_comparisons
// to maintain backward compatibility while avoiding duplication
pub use crate::ndarray_comparisons::{run_comparison_benchmarks, run_extended_benchmarks};

/// Comprehensive benchmark suite with analysis
pub fn benchmark_and_analyze() -> std::io::Result<()> {
    // Ensure target directory exists
    std::fs::create_dir_all("target")?;

    let runner = run_extended_benchmarks();

    // Generate basic comparison report
    runner.generate_report("target/comparison_report.md")?;

    // Perform detailed analysis
    let mut analyzer = PerformanceAnalyzer::new();
    analyzer.add_results(runner.results());

    // Analyze key operations
    let operations = ["matrix_multiplication", "elementwise_addition", "conv2d"];

    let mut analysis_file = std::fs::File::create("target/performance_analysis.md")?;
    use std::io::Write;

    writeln!(analysis_file, "# ToRSh Performance Analysis\n")?;

    for operation in &operations {
        let analysis = analyzer.analyze_operation(operation);

        writeln!(analysis_file, "## {}\n", operation)?;

        // Write library statistics
        for (library, stats) in &analysis.library_stats {
            writeln!(analysis_file, "### {}\n", library)?;
            writeln!(
                analysis_file,
                "- Average time: {:.2} μs",
                stats.mean_time_ns / 1000.0
            )?;
            if let Some(throughput) = stats.mean_throughput {
                writeln!(
                    analysis_file,
                    "- Throughput: {:.2} GFLOPS",
                    throughput / 1e9
                )?;
            }
            writeln!(analysis_file, "- Samples: {}\n", stats.sample_count)?;
        }

        // Write recommendations
        if !analysis.recommendations.is_empty() {
            writeln!(analysis_file, "### Recommendations\n")?;
            for rec in &analysis.recommendations {
                writeln!(analysis_file, "- {}", rec)?;
            }
            writeln!(analysis_file)?;
        }
    }

    println!("Extended benchmarks completed!");
    println!("Results saved to target/comparison_report.md");
    println!("Analysis saved to target/performance_analysis.md");

    Ok(())
}

/// Legacy function for backward compatibility
pub fn benchmark_and_compare() -> std::io::Result<()> {
    benchmark_and_analyze()
}

/// Generate master comparison report with all available benchmark suites
pub fn generate_master_comparison_report() -> std::io::Result<()> {
    use std::io::Write;

    // Ensure target directory exists
    std::fs::create_dir_all("target")?;

    let mut report_file = std::fs::File::create("target/master_comparison_report.md")?;

    writeln!(
        report_file,
        "# ToRSh Master Performance Comparison Report\n"
    )?;
    writeln!(
        report_file,
        "Generated on: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;

    writeln!(report_file, "## Overview\n")?;
    writeln!(
        report_file,
        "This master report aggregates performance comparisons from all available benchmark suites."
    )?;
    writeln!(
        report_file,
        "It provides a comprehensive view of ToRSh performance across different operations and libraries.\n"
    )?;

    // Basic ToRSh vs ndarray comparisons
    writeln!(report_file, "## Core Library Comparisons\n")?;
    let basic_runner = crate::ndarray_comparisons::run_comparison_benchmarks();
    write_comparison_section(&mut report_file, &basic_runner, "Core ToRSh vs ndarray")?;

    // PyTorch comparisons (if available)
    #[cfg(feature = "pytorch")]
    {
        writeln!(report_file, "\n## PyTorch Comparisons\n")?;
        let pytorch_runner = crate::pytorch_comparisons::run_pytorch_comparison_benchmarks();
        write_comparison_section(&mut report_file, &pytorch_runner, "ToRSh vs PyTorch")?;
    }

    // TensorFlow comparisons (if available)
    #[cfg(feature = "tensorflow")]
    {
        writeln!(report_file, "\n## TensorFlow Comparisons\n")?;
        let tensorflow_runner = crate::tensorflow_comparisons::run_tensorflow_comparison_suite();
        write_comparison_section(&mut report_file, &tensorflow_runner, "ToRSh vs TensorFlow")?;
    }

    // JAX comparisons (if available)
    #[cfg(feature = "jax")]
    {
        writeln!(report_file, "\n## JAX Comparisons\n")?;
        let jax_runner = crate::jax_comparisons::run_jax_comparison_suite();
        write_comparison_section(&mut report_file, &jax_runner, "ToRSh vs JAX")?;
    }

    // NumPy baseline comparisons (if available)
    #[cfg(feature = "numpy_baseline")]
    {
        writeln!(report_file, "\n## NumPy Baseline Comparisons\n")?;
        let numpy_runner = crate::numpy_comparisons::run_numpy_comparison_suite();
        write_comparison_section(&mut report_file, &numpy_runner, "ToRSh vs NumPy")?;
    }

    writeln!(report_file, "\n## Summary\n")?;
    writeln!(
        report_file,
        "This master report provides a comprehensive view of ToRSh performance across all available comparison libraries."
    )?;
    writeln!(
        report_file,
        "Each comparison suite targets specific use cases and performance characteristics:"
    )?;
    writeln!(
        report_file,
        "- **ndarray**: Rust ecosystem baseline performance"
    )?;
    writeln!(
        report_file,
        "- **PyTorch**: Deep learning framework comparison"
    )?;
    writeln!(
        report_file,
        "- **TensorFlow**: Production ML framework comparison"
    )?;
    writeln!(
        report_file,
        "- **JAX**: High-performance research framework comparison"
    )?;
    writeln!(
        report_file,
        "- **NumPy**: Scientific computing foundation baseline"
    )?;

    println!("📈 Master comparison report generated!");
    println!("   📄 Report: target/master_comparison_report.md");

    Ok(())
}

/// Helper function to write a comparison section
fn write_comparison_section(
    file: &mut std::fs::File,
    runner: &ComparisonRunner,
    section_title: &str,
) -> std::io::Result<()> {
    use std::io::Write;

    writeln!(file, "### {}\n", section_title)?;

    if runner.results().is_empty() {
        writeln!(file, "No results available.\n")?;
        return Ok(());
    }

    // Group results by operation
    let mut operations = std::collections::HashMap::new();
    for result in runner.results() {
        operations
            .entry(&result.operation)
            .or_insert_with(Vec::new)
            .push(result);
    }

    for (operation, results) in &operations {
        writeln!(file, "#### {}\n", operation.replace('_', " "))?;

        writeln!(file, "| Library | Size | Time (μs) | Throughput (GFLOPS) |")?;
        writeln!(file, "|---------|------|-----------|---------------------|")?;

        for result in results {
            let throughput_str = if let Some(tp) = result.throughput {
                format!("{:.2}", tp / 1e9)
            } else {
                "N/A".to_string()
            };

            writeln!(
                file,
                "| {} | {} | {:.2} | {} |",
                result.library,
                result.size,
                result.time_ns / 1000.0,
                throughput_str
            )?;
        }
        writeln!(file)?;
    }

    Ok(())
}

/// Run all available comparison suites in sequence
pub fn run_all_comparison_suites() -> std::io::Result<()> {
    println!("🚀 Running all available comparison suites...\n");

    // Core ndarray comparisons
    println!("📊 Running core ndarray comparisons...");
    let _ndarray_runner = crate::ndarray_comparisons::run_comparison_benchmarks();

    // PyTorch comparisons (if available)
    #[cfg(feature = "pytorch")]
    {
        println!("🔥 Running PyTorch comparisons...");
        let _pytorch_runner = crate::pytorch_comparisons::run_pytorch_comparison_benchmarks();
    }

    // TensorFlow comparisons (if available)
    #[cfg(feature = "tensorflow")]
    {
        println!("🌊 Running TensorFlow comparisons...");
        let _tensorflow_runner = crate::tensorflow_comparisons::run_tensorflow_comparison_suite();
    }

    // JAX comparisons (if available)
    #[cfg(feature = "jax")]
    {
        println!("⚡ Running JAX comparisons...");
        let _jax_runner = crate::jax_comparisons::run_jax_comparison_suite();
    }

    // NumPy baseline comparisons (if available)
    #[cfg(feature = "numpy_baseline")]
    {
        println!("📐 Running NumPy baseline comparisons...");
        let _numpy_runner = crate::numpy_comparisons::run_numpy_comparison_suite();
    }

    // Generate master report
    generate_master_comparison_report()?;

    println!("✅ All comparison suites completed!");
    println!("📄 Master report: target/master_comparison_report.md");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_comparison_benchmarks() {
        let runner = run_comparison_benchmarks();
        assert!(!runner.results().is_empty());
    }

    #[test]
    fn test_run_extended_benchmarks() {
        let runner = run_extended_benchmarks();
        assert!(!runner.results().is_empty());
    }

    #[test]
    fn test_benchmark_and_analyze() {
        // Test that the analysis function can run without errors
        assert!(benchmark_and_analyze().is_ok());
    }

    #[test]
    fn test_generate_master_report() {
        // Test that master report generation can run without errors
        assert!(generate_master_comparison_report().is_ok());
    }
}
