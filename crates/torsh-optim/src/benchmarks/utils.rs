//! Utility functions for benchmarking optimizers
//!
//! This module provides convenient utility functions for running common
//! benchmarking scenarios and quick performance evaluations.

use super::optimizer::OptimizerBenchmarks;
use crate::{adam::Adam, rmsprop::RMSprop, sgd::SGD, Optimizer, OptimizerResult};
use parking_lot::RwLock;
use std::sync::Arc;
use torsh_core::device::DeviceType;
use torsh_tensor::creation;

/// Utility function to run quick benchmarks on common optimizers
///
/// This function provides a convenient way to quickly evaluate the performance
/// of the most commonly used optimizers (Adam, SGD, RMSprop) on a standard
/// set of benchmarks.
///
/// # Returns
///
/// Returns Ok(()) if all benchmarks complete successfully, or an error if
/// any benchmark fails.
///
/// # Example
///
/// ```rust,no_run
/// # use torsh_core::error::Result;
/// # fn main() -> Result<()> {
/// use torsh_optim::benchmarks::utils::run_quick_benchmark_suite;
///
/// // Run benchmarks on common optimizers (this may take some time)
/// run_quick_benchmark_suite()?;
/// # Ok(())
/// # }
/// ```
pub fn run_quick_benchmark_suite() -> OptimizerResult<()> {
    let benchmarks = OptimizerBenchmarks::new();
    let device = DeviceType::Cpu;

    println!("Running quick benchmark suite...");

    // Test Adam
    let adam_tensor = creation::randn::<f32>(&[1000])?;
    let adam_params = vec![Arc::new(RwLock::new(adam_tensor))];
    let adam = Adam::new(adam_params, Some(0.001), None, None, None, false);
    println!("\nAdam Optimizer:");
    let adam_results = benchmarks.run_comprehensive_benchmarks(adam)?;
    benchmarks.print_results(&adam_results);

    // Test SGD
    let sgd_tensor = creation::randn::<f32>(&[1000])?;
    let sgd_params = vec![Arc::new(RwLock::new(sgd_tensor))];
    let sgd = SGD::new(sgd_params, 0.01, None, None, None, false);
    println!("\nSGD Optimizer:");
    let sgd_results = benchmarks.run_comprehensive_benchmarks(sgd)?;
    benchmarks.print_results(&sgd_results);

    // Test RMSprop
    let rmsprop_tensor = creation::randn::<f32>(&[1000])?;
    let rmsprop_params = vec![Arc::new(RwLock::new(rmsprop_tensor))];
    let rmsprop = RMSprop::new(rmsprop_params, Some(0.01), None, None, None, None, false);
    println!("\nRMSprop Optimizer:");
    let rmsprop_results = benchmarks.run_comprehensive_benchmarks(rmsprop)?;
    benchmarks.print_results(&rmsprop_results);

    Ok(())
}

/// Run a quick comparison between multiple optimizers
///
/// This utility function creates a quick side-by-side comparison of
/// different optimizers using default parameters.
///
/// # Arguments
///
/// * `problem_size` - Size of the optimization problem for testing
/// * `num_iterations` - Number of benchmark iterations to run
///
/// # Returns
///
/// Returns Ok(()) if the comparison completes successfully.
pub fn run_quick_optimizer_comparison(
    problem_size: usize,
    num_iterations: usize,
) -> OptimizerResult<()> {
    use super::comparison::OptimizerComparison;
    use super::core::BenchmarkConfig;

    let config = BenchmarkConfig {
        num_iterations,
        warmup_iterations: num_iterations / 10,
        max_time_seconds: 30.0,
        ..Default::default()
    };

    println!(
        "Running quick optimizer comparison (problem size: {}, iterations: {})",
        problem_size, num_iterations
    );

    // Run comparison for Adam optimizers
    let adam_comparison = OptimizerComparison::<Adam>::new()
        .with_config(config.clone())
        .add_optimizer("Adam", move || {
            let tensor = creation::randn::<f32>(&[problem_size])?;
            let params = vec![Arc::new(RwLock::new(tensor))];
            Ok(Adam::new(params, Some(0.001), None, None, None, false))
        });

    let adam_results = adam_comparison.run_comparison_suite()?;
    adam_comparison.print_comparison_table(&adam_results);

    // Run comparison for SGD optimizers
    let sgd_comparison = OptimizerComparison::<SGD>::new()
        .with_config(config)
        .add_optimizer("SGD", move || {
            let tensor = creation::randn::<f32>(&[problem_size])?;
            let params = vec![Arc::new(RwLock::new(tensor))];
            Ok(SGD::new(params, 0.01, Some(0.9), None, None, false))
        });

    let sgd_results = sgd_comparison.run_comparison_suite()?;
    sgd_comparison.print_comparison_table(&sgd_results);

    Ok(())
}

/// Run domain-specific benchmarks for all optimizers
///
/// This utility function runs Computer Vision and NLP benchmarks
/// for a set of optimizers to help users understand performance
/// in specific domains.
pub fn run_domain_benchmarks() -> OptimizerResult<()> {
    use super::domain_specific::{CVBenchmarks, NLPBenchmarks};

    println!("Running domain-specific benchmarks...");

    let cv_benchmarks = CVBenchmarks::new();
    let nlp_benchmarks = NLPBenchmarks::new();

    // Test optimizers on CV tasks
    println!("\n{:=<80}", "");
    println!("{:^80}", "COMPUTER VISION BENCHMARKS");
    println!("{:=<80}", "");

    // Test Adam optimizer
    println!("\n--- Adam on ResNet Training ---");
    {
        let tensor = creation::randn::<f32>(&[1000])?;
        let params = vec![Arc::new(RwLock::new(tensor))];
        let optimizer = Adam::new(params, Some(0.001), None, None, None, false);
        let result = cv_benchmarks.benchmark_resnet_training(optimizer)?;
        println!(
            "Completed {} iterations in {:.3?} (avg: {:.3?} per iteration)",
            result.iterations_completed, result.total_time, result.avg_time_per_iteration
        );
    }

    // Test SGD optimizer
    println!("\n--- SGD on ResNet Training ---");
    {
        let tensor = creation::randn::<f32>(&[1000])?;
        let params = vec![Arc::new(RwLock::new(tensor))];
        let optimizer = SGD::new(params, 0.01, Some(0.9), None, None, false);
        let result = cv_benchmarks.benchmark_resnet_training(optimizer)?;
        println!(
            "Completed {} iterations in {:.3?} (avg: {:.3?} per iteration)",
            result.iterations_completed, result.total_time, result.avg_time_per_iteration
        );
    }

    // Test optimizers on NLP tasks
    println!("\n{:=<80}", "");
    println!("{:^80}", "NATURAL LANGUAGE PROCESSING BENCHMARKS");
    println!("{:=<80}", "");

    // Test Adam optimizer on NLP
    println!("\n--- Adam on Transformer Training ---");
    {
        let tensor = creation::randn::<f32>(&[1000])?;
        let params = vec![Arc::new(RwLock::new(tensor))];
        let optimizer = Adam::new(params, Some(0.001), None, None, None, false);
        let result = nlp_benchmarks.benchmark_transformer_training(optimizer)?;
        println!(
            "Completed {} iterations in {:.3?} (avg: {:.3?} per iteration)",
            result.iterations_completed, result.total_time, result.avg_time_per_iteration
        );
    }

    // Test SGD optimizer on NLP
    println!("\n--- SGD on Transformer Training ---");
    {
        let tensor = creation::randn::<f32>(&[1000])?;
        let params = vec![Arc::new(RwLock::new(tensor))];
        let optimizer = SGD::new(params, 0.01, Some(0.9), None, None, false);
        let result = nlp_benchmarks.benchmark_transformer_training(optimizer)?;
        println!(
            "Completed {} iterations in {:.3?} (avg: {:.3?} per iteration)",
            result.iterations_completed, result.total_time, result.avg_time_per_iteration
        );
    }

    Ok(())
}

/// Create a simple benchmark configuration for testing
///
/// This utility function creates a lightweight benchmark configuration
/// suitable for quick tests and development.
pub fn create_test_config() -> super::core::BenchmarkConfig {
    super::core::BenchmarkConfig {
        num_iterations: 100,
        warmup_iterations: 10,
        max_time_seconds: 10.0,
        device: DeviceType::Cpu,
        profile_memory: false,
    }
}

/// Create a comprehensive benchmark configuration
///
/// This utility function creates a full benchmark configuration
/// suitable for thorough performance evaluation.
pub fn create_comprehensive_config() -> super::core::BenchmarkConfig {
    super::core::BenchmarkConfig {
        num_iterations: 10000,
        warmup_iterations: 1000,
        max_time_seconds: 300.0, // 5 minutes
        device: DeviceType::Cpu,
        profile_memory: true,
    }
}

/// Utility to benchmark a single optimizer on multiple problem sizes
///
/// This function helps evaluate how an optimizer's performance scales
/// with problem size.
pub fn benchmark_scaling<O: Clone>(
    optimizer_factory: impl Fn(usize) -> OptimizerResult<O>,
    sizes: &[usize],
) -> OptimizerResult<()>
where
    O: crate::Optimizer,
{
    let benchmarks = OptimizerBenchmarks::new();

    println!("Benchmarking optimizer scaling across problem sizes...");
    println!(
        "{:<15} {:>12} {:>15} {:>15}",
        "Problem Size", "Iterations", "Total Time", "Avg Time"
    );
    println!("{:-<60}", "");

    for &size in sizes {
        let optimizer = optimizer_factory(size)?;
        let result = benchmarks.benchmark_step_performance(optimizer, size)?;

        println!(
            "{:<15} {:>12} {:>15.3?} {:>15.3?}",
            size, result.iterations_completed, result.total_time, result.avg_time_per_iteration
        );
    }

    Ok(())
}

/// Generate a summary report of benchmark results
///
/// This utility function takes benchmark results and generates
/// a formatted summary report.
///
/// # Notes
///
/// Writing to a String should never fail in practice, but we handle
/// all errors properly to maintain COOLJAPAN policy compliance.
pub fn generate_summary_report(results: &[super::core::BenchmarkResult]) -> String {
    use std::fmt::Write;

    let mut report = String::new();

    // Note: Writing to String should never fail, but we handle errors to avoid unwrap()
    let _ = writeln!(report, "BENCHMARK SUMMARY REPORT");
    let _ = writeln!(report, "{:=<50}", "");

    let total_benchmarks = results.len();
    let total_iterations: usize = results.iter().map(|r| r.iterations_completed).sum();
    let total_time: std::time::Duration = results.iter().map(|r| r.total_time).sum();

    let _ = writeln!(report, "Total Benchmarks: {}", total_benchmarks);
    let _ = writeln!(report, "Total Iterations: {}", total_iterations);
    let _ = writeln!(report, "Total Time: {:.3?}", total_time);

    if total_benchmarks > 0 {
        let avg_time_per_benchmark = total_time / total_benchmarks as u32;
        let _ = writeln!(
            report,
            "Average Time per Benchmark: {:.3?}",
            avg_time_per_benchmark
        );
    }

    // Find fastest and slowest benchmarks
    if let Some(fastest) = results.iter().min_by_key(|r| r.avg_time_per_iteration) {
        let _ = writeln!(
            report,
            "Fastest: {} ({:.3?} per iteration)",
            fastest.name, fastest.avg_time_per_iteration
        );
    }

    if let Some(slowest) = results.iter().max_by_key(|r| r.avg_time_per_iteration) {
        let _ = writeln!(
            report,
            "Slowest: {} ({:.3?} per iteration)",
            slowest.name, slowest.avg_time_per_iteration
        );
    }

    // Convergence analysis
    let converged_benchmarks: Vec<_> = results
        .iter()
        .filter(|r| r.final_loss.map_or(false, |loss| loss < 1e-6))
        .collect();

    if !converged_benchmarks.is_empty() {
        let _ = writeln!(
            report,
            "\nConverged Benchmarks: {}",
            converged_benchmarks.len()
        );
        for benchmark in converged_benchmarks {
            let _ = writeln!(
                report,
                "  - {} (final loss: {:.2e})",
                benchmark.name,
                benchmark.final_loss.unwrap_or(0.0)
            );
        }
    }

    let _ = writeln!(report, "{:=<50}", "");

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_config() {
        let config = create_test_config();
        assert_eq!(config.num_iterations, 100);
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.max_time_seconds, 10.0);
        assert!(!config.profile_memory);
    }

    #[test]
    fn test_create_comprehensive_config() {
        let config = create_comprehensive_config();
        assert_eq!(config.num_iterations, 10000);
        assert_eq!(config.warmup_iterations, 1000);
        assert_eq!(config.max_time_seconds, 300.0);
        assert!(config.profile_memory);
    }

    #[test]
    fn test_generate_summary_report() {
        use super::super::core::BenchmarkResult;
        use std::time::Duration;

        let results = vec![
            BenchmarkResult {
                name: "test1".to_string(),
                iterations_completed: 100,
                total_time: Duration::from_secs(1),
                avg_time_per_iteration: Duration::from_millis(10),
                min_time_per_iteration: Duration::from_millis(8),
                max_time_per_iteration: Duration::from_millis(15),
                time_std_dev: Duration::from_millis(2),
                final_loss: Some(1e-8),
                memory_stats: None,
                convergence_rate: Some(0.1),
            },
            BenchmarkResult {
                name: "test2".to_string(),
                iterations_completed: 200,
                total_time: Duration::from_secs(2),
                avg_time_per_iteration: Duration::from_millis(10),
                min_time_per_iteration: Duration::from_millis(9),
                max_time_per_iteration: Duration::from_millis(12),
                time_std_dev: Duration::from_millis(1),
                final_loss: Some(1e-4),
                memory_stats: None,
                convergence_rate: Some(0.05),
            },
        ];

        let report = generate_summary_report(&results);

        assert!(report.contains("BENCHMARK SUMMARY REPORT"));
        assert!(report.contains("Total Benchmarks: 2"));
        assert!(report.contains("Total Iterations: 300"));
        assert!(report.contains("Converged Benchmarks: 1"));
        assert!(report.contains("test1 (final loss: 1.00e-8)"));
    }
}
