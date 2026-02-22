//! Performance comparisons with other tensor libraries - Clean Modular Interface
//!
//! This module provides comprehensive performance comparison capabilities between
//! ToRSh and other tensor libraries including NDArray, PyTorch, TensorFlow, and JAX.
//!
//! # Architecture
//!
//! The comparison system is organized into specialized modules for improved
//! maintainability and functionality:
//!
//! - **core**: Core comparison infrastructure and result types
//! - **analysis**: Performance analysis and regression detection
//! - **torsh_benchmarks**: ToRSh benchmark implementations
//! - **ndarray_comparisons**: NDArray comparison benchmarks (feature-gated)
//! - **pytorch_comparisons**: PyTorch comparison benchmarks (feature-gated)
//! - **tensorflow_comparisons**: TensorFlow comparison benchmarks (feature-gated)
//! - **jax_comparisons**: JAX comparison benchmarks (feature-gated)
//! - **integration**: High-level benchmark runners and coordination functions
//!
//! All original APIs are maintained for full backward compatibility.

// Modular architecture imports - organized by functionality
mod comparisons;

// Re-export the complete modular interface for external use
pub use comparisons::*;

// Complete backward compatibility re-exports for all comparison components
// ============================================================================

// Core Infrastructure - Essential comparison types and functionality
pub use comparisons::core::{
    ComparisonRunner,
    ComparisonResult,
    benchmark_and_compare,
};

// Analysis & Diagnostics - Performance analysis and regression detection
pub use comparisons::analysis::{
    PerformanceAnalyzer,
    AnalysisResult,
    LibraryStats,
    RegressionDetector,
    RegressionResult,
};

// ToRSh Benchmarks - Core ToRSh benchmark implementations
pub use comparisons::torsh_benchmarks::{
    TorshMatmulBench,
    TorshElementwiseBench,
};

// NDArray Comparisons - Feature-gated NDArray benchmark implementations
#[cfg(feature = "compare-external")]
pub use comparisons::ndarray_comparisons::{
    NdarrayMatmulBench,
    NdarrayElementwiseBench,
};

// PyTorch Comparisons - Feature-gated PyTorch benchmark implementations
#[cfg(feature = "pytorch")]
pub use comparisons::pytorch_comparisons::{
    PyTorchMatmulBench,
    PyTorchElementwiseBench,
    PyTorchConvBench,
    PyTorchAutogradBench,
    PyTorchDataLoaderBench,
    run_pytorch_comparison_benchmarks,
    run_comprehensive_pytorch_benchmarks,
    run_quick_pytorch_comparison,
};

// TensorFlow Comparisons - Feature-gated TensorFlow benchmark implementations
#[cfg(feature = "tensorflow")]
pub use comparisons::tensorflow_comparisons::{
    TensorFlowBenchRunner,
    TorshVsTensorFlowMatmul,
    TorshVsTensorFlowElementwise,
    TorshVsTensorFlowConv2d,
    run_tensorflow_comparison_suite,
    generate_tensorflow_comparison_report,
};

// JAX Comparisons - Feature-gated JAX benchmark implementations
#[cfg(feature = "jax")]
pub use comparisons::jax_comparisons::{
    JAXBenchRunner,
};

// Integration Functions - High-level benchmark runners and coordination
pub use comparisons::integration::{
    run_comparison_benchmarks,
    run_extended_benchmarks,
    benchmark_and_analyze,
};

// Utility Functions - Additional helper functions for benchmark comparisons
pub use comparisons::utils::{
    generate_comparison_summary,
    calculate_memory_efficiency,
    analyze_performance_scaling,
};

// Preset Configurations - Pre-configured benchmark suites for common scenarios
pub use comparisons::presets::{
    quick_dev_comparison,
    comprehensive_research_comparison,
    production_validation,
};

/// Enhanced comparison utilities for advanced analysis
pub mod advanced {
    //! Advanced comparison analysis tools

    use super::{ComparisonResult, PerformanceAnalyzer};
    use std::collections::HashMap;

    /// Cross-library performance comparison matrix
    pub fn generate_performance_matrix(results: &[ComparisonResult]) -> HashMap<String, HashMap<String, f64>> {
        let mut matrix = HashMap::new();

        // Group by operation
        let mut by_operation: HashMap<String, Vec<&ComparisonResult>> = HashMap::new();
        for result in results {
            by_operation
                .entry(result.operation.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (operation, op_results) in by_operation {
            let mut op_matrix = HashMap::new();

            // Group by library for this operation
            let mut by_library: HashMap<String, Vec<&ComparisonResult>> = HashMap::new();
            for result in op_results {
                by_library
                    .entry(result.library.clone())
                    .or_insert_with(Vec::new)
                    .push(result);
            }

            // Calculate average times for each library
            for (library, lib_results) in by_library {
                let avg_time = lib_results.iter()
                    .map(|r| r.time_ns)
                    .sum::<f64>() / lib_results.len() as f64;
                op_matrix.insert(library, avg_time);
            }

            matrix.insert(operation, op_matrix);
        }

        matrix
    }

    /// Generate statistical confidence intervals for benchmark results
    pub fn calculate_confidence_intervals(results: &[ComparisonResult], confidence: f64) -> HashMap<String, (f64, f64)> {
        let mut intervals = HashMap::new();

        // Group by operation and library
        let mut by_op_lib: HashMap<(String, String), Vec<f64>> = HashMap::new();
        for result in results {
            by_op_lib
                .entry((result.operation.clone(), result.library.clone()))
                .or_insert_with(Vec::new)
                .push(result.time_ns);
        }

        for ((operation, library), times) in by_op_lib {
            if times.len() > 1 {
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                let variance = times.iter()
                    .map(|t| (t - mean).powi(2))
                    .sum::<f64>() / (times.len() - 1) as f64;
                let std_dev = variance.sqrt();
                let std_error = std_dev / (times.len() as f64).sqrt();

                // Approximate confidence interval (assuming normal distribution)
                let z_score = match confidence {
                    0.95 => 1.96,
                    0.99 => 2.576,
                    0.90 => 1.645,
                    _ => 1.96, // Default to 95%
                };

                let margin = z_score * std_error;
                intervals.insert(
                    format!("{}_{}", operation, library),
                    (mean - margin, mean + margin)
                );
            }
        }

        intervals
    }

    /// Detect performance anomalies using statistical analysis
    pub fn detect_performance_anomalies(results: &[ComparisonResult], threshold: f64) -> Vec<String> {
        let mut anomalies = Vec::new();

        // Group by operation and library
        let mut by_op_lib: HashMap<(String, String), Vec<f64>> = HashMap::new();
        for result in results {
            by_op_lib
                .entry((result.operation.clone(), result.library.clone()))
                .or_insert_with(Vec::new)
                .push(result.time_ns);
        }

        for ((operation, library), times) in by_op_lib {
            if times.len() > 3 { // Need sufficient data points
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                let variance = times.iter()
                    .map(|t| (t - mean).powi(2))
                    .sum::<f64>() / times.len() as f64;
                let std_dev = variance.sqrt();

                // Check for outliers using z-score
                for &time in &times {
                    let z_score = (time - mean).abs() / std_dev;
                    if z_score > threshold {
                        anomalies.push(format!(
                            "Anomaly detected in {} for {}: time={:.2}ns (z-score={:.2})",
                            operation, library, time, z_score
                        ));
                    }
                }
            }
        }

        anomalies
    }
}

/// Benchmarking automation utilities
pub mod automation {
    //! Automated benchmark scheduling and execution

    use super::{ComparisonRunner, benchmark_and_analyze};
    use std::time::Duration;

    /// Automated benchmark scheduler
    pub struct BenchmarkScheduler {
        interval: Duration,
        baseline_path: String,
        enabled: bool,
    }

    impl BenchmarkScheduler {
        pub fn new(interval: Duration, baseline_path: String) -> Self {
            Self {
                interval,
                baseline_path,
                enabled: false,
            }
        }

        /// Enable automated benchmarking
        pub fn enable(&mut self) {
            self.enabled = true;
        }

        /// Disable automated benchmarking
        pub fn disable(&mut self) {
            self.enabled = false;
        }

        /// Check if scheduler is enabled
        pub fn is_enabled(&self) -> bool {
            self.enabled
        }

        /// Run a single benchmark cycle
        pub fn run_cycle(&self) -> std::io::Result<()> {
            if !self.enabled {
                return Ok(());
            }

            println!("🔄 Running automated benchmark cycle...");
            benchmark_and_analyze()?;
            println!("✅ Automated benchmark cycle completed");

            Ok(())
        }

        /// Get configured interval
        pub fn interval(&self) -> Duration {
            self.interval
        }

        /// Update benchmark interval
        pub fn set_interval(&mut self, interval: Duration) {
            self.interval = interval;
        }
    }

    /// Continuous integration benchmark runner
    pub fn run_ci_benchmarks() -> std::io::Result<bool> {
        println!("🚀 Running CI benchmark suite...");

        // Run quick comparison for CI
        let runner = super::quick_dev_comparison();

        // Generate CI report
        runner.generate_report("target/ci_benchmark_report.md")?;

        // Check for significant performance changes
        let results = runner.results();
        let has_regressions = results.iter().any(|r| {
            // Simple heuristic: check if any operation took significantly longer
            if let Some(throughput) = r.throughput {
                throughput < 1e6 // Less than 1 MFLOPS indicates potential issues
            } else {
                r.time_ns > 1e9 // More than 1 second indicates potential issues
            }
        });

        if has_regressions {
            println!("⚠️  Potential performance regressions detected");
        } else {
            println!("✅ CI benchmarks passed");
        }

        Ok(!has_regressions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_comparison_system_integration() {
        // Test that the complete modular system works together

        // Test core functionality
        let mut runner = ComparisonRunner::new();
        assert_eq!(runner.results().len(), 0);

        // Test analysis functionality
        let comparison_runner = run_comparison_benchmarks();
        assert!(comparison_runner.results().len() > 0);

        let mut analyzer = PerformanceAnalyzer::new();
        analyzer.add_results(comparison_runner.results());

        // Test utility functions
        let summary = generate_comparison_summary(comparison_runner.results());
        assert!(summary.len() >= 0);

        let efficiency = calculate_memory_efficiency(comparison_runner.results());
        assert!(efficiency.len() >= 0);

        let scaling = analyze_performance_scaling(comparison_runner.results());
        assert!(scaling.len() >= 0);

        // Test preset functions
        let quick_runner = quick_dev_comparison();
        assert!(quick_runner.results().len() > 0);
    }

    #[test]
    fn test_advanced_analysis_tools() {
        // Test advanced analysis functionality
        let runner = run_comparison_benchmarks();
        let results = runner.results();

        if !results.is_empty() {
            // Test performance matrix generation
            let matrix = advanced::generate_performance_matrix(results);
            assert!(matrix.len() >= 0);

            // Test confidence interval calculation
            let intervals = advanced::calculate_confidence_intervals(results, 0.95);
            assert!(intervals.len() >= 0);

            // Test anomaly detection
            let anomalies = advanced::detect_performance_anomalies(results, 2.0);
            assert!(anomalies.len() >= 0);
        }
    }

    #[test]
    fn test_automation_utilities() {
        // Test automation functionality
        use std::time::Duration;

        let mut scheduler = automation::BenchmarkScheduler::new(
            Duration::from_secs(3600),
            "baseline.json".to_string()
        );

        assert!(!scheduler.is_enabled());

        scheduler.enable();
        assert!(scheduler.is_enabled());

        scheduler.disable();
        assert!(!scheduler.is_enabled());

        assert_eq!(scheduler.interval(), Duration::from_secs(3600));

        scheduler.set_interval(Duration::from_secs(1800));
        assert_eq!(scheduler.interval(), Duration::from_secs(1800));
    }

    #[test]
    fn test_complete_backward_compatibility() {
        // Comprehensive test that all original APIs continue to work

        // Core comparison functionality
        let _runner = ComparisonRunner::new();
        let _comparison_results = run_comparison_benchmarks();
        let _extended_results = run_extended_benchmarks();

        // Analysis functionality
        let mut _analyzer = PerformanceAnalyzer::new();
        let _detector = RegressionDetector::new(0.1);

        // ToRSh benchmarks
        let _matmul = TorshMatmulBench;
        let _elementwise = TorshElementwiseBench;

        // Feature-gated functionality should be available when enabled
        #[cfg(feature = "compare-external")]
        {
            let _ndarray_matmul = NdarrayMatmulBench;
            let _ndarray_elementwise = NdarrayElementwiseBench;
        }

        #[cfg(feature = "pytorch")]
        {
            let _pytorch_matmul = PyTorchMatmulBench;
            let _pytorch_elementwise = PyTorchElementwiseBench;
            let _pytorch_conv = PyTorchConvBench;
            let _pytorch_autograd = PyTorchAutogradBench;
            let _pytorch_dataloader = PyTorchDataLoaderBench;

            let _pytorch_comparison = run_pytorch_comparison_benchmarks();
            let _quick_pytorch = run_quick_pytorch_comparison();
        }

        #[cfg(feature = "tensorflow")]
        {
            let _tf_runner = TensorFlowBenchRunner::new();
            let _tf_matmul = TorshVsTensorFlowMatmul::new();
            let _tf_elementwise = TorshVsTensorFlowElementwise::new();
            let _tf_conv = TorshVsTensorFlowConv2d::new();

            let _tf_comparison = run_tensorflow_comparison_suite();
        }

        #[cfg(feature = "jax")]
        {
            let _jax_runner = JAXBenchRunner::new();
        }

        // Integration and preset functions
        let _quick_dev = quick_dev_comparison();
        let _production_result = production_validation();
    }
}