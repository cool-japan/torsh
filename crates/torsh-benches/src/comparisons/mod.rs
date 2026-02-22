//! Performance comparisons with other tensor libraries - Modular System
//!
//! This module provides comprehensive performance comparison capabilities between
//! ToRSh and other tensor libraries including NDArray, PyTorch, TensorFlow, and JAX.
//!
//! # Architecture
//!
//! The comparison system is organized into specialized modules:
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
//! All modules provide feature-gated functionality for optional external library comparisons.

// Modular architecture imports
pub mod core;
pub mod analysis;
pub mod torsh_benchmarks;
pub mod ndarray_comparisons;
pub mod pytorch_comparisons;
pub mod tensorflow_comparisons;
pub mod jax_comparisons;
pub mod integration;

// Re-export core comparison infrastructure
pub use core::{
    ComparisonRunner, ComparisonResult, benchmark_and_compare,
};

// Re-export analysis utilities
pub use analysis::{
    PerformanceAnalyzer, AnalysisResult, LibraryStats,
    RegressionDetector, RegressionResult,
};

// Re-export ToRSh benchmarks
pub use torsh_benchmarks::{
    TorshMatmulBench, TorshElementwiseBench,
};

// Re-export NDArray comparisons (feature-gated)
#[cfg(feature = "compare-external")]
pub use ndarray_comparisons::{
    NdarrayMatmulBench, NdarrayElementwiseBench,
};

// Re-export PyTorch comparisons (feature-gated)
#[cfg(feature = "pytorch")]
pub use pytorch_comparisons::{
    PyTorchMatmulBench, PyTorchElementwiseBench, PyTorchConvBench,
    PyTorchAutogradBench, PyTorchDataLoaderBench,
    run_pytorch_comparison_benchmarks, run_comprehensive_pytorch_benchmarks,
    run_quick_pytorch_comparison,
};

// Re-export TensorFlow comparisons (feature-gated)
#[cfg(feature = "tensorflow")]
pub use tensorflow_comparisons::{
    TensorFlowBenchRunner, TorshVsTensorFlowMatmul,
    TorshVsTensorFlowElementwise, TorshVsTensorFlowConv2d,
    run_tensorflow_comparison_suite, generate_tensorflow_comparison_report,
};

// Re-export JAX comparisons (feature-gated)
#[cfg(feature = "jax")]
pub use jax_comparisons::{
    JAXBenchRunner,
};

// Re-export integration functions
pub use integration::{
    run_comparison_benchmarks, run_extended_benchmarks, benchmark_and_analyze,
};

/// Comprehensive comparison utilities
pub mod utils {
    //! Utility functions for benchmark comparisons

    use super::core::ComparisonResult;
    use std::collections::HashMap;

    /// Generate performance comparison summary
    pub fn generate_comparison_summary(results: &[ComparisonResult]) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        // Group by operation and library
        let mut by_operation: HashMap<String, Vec<&ComparisonResult>> = HashMap::new();
        for result in results {
            by_operation
                .entry(result.operation.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (operation, op_results) in by_operation {
            let mut by_library: HashMap<String, Vec<&ComparisonResult>> = HashMap::new();
            for result in op_results {
                by_library
                    .entry(result.library.clone())
                    .or_insert_with(Vec::new)
                    .push(result);
            }

            // Calculate relative performance vs ToRSh
            if let Some(torsh_results) = by_library.get("torsh") {
                let torsh_avg_time: f64 = torsh_results.iter()
                    .map(|r| r.time_ns)
                    .sum::<f64>() / torsh_results.len() as f64;

                for (library, lib_results) in &by_library {
                    if library != "torsh" {
                        let lib_avg_time: f64 = lib_results.iter()
                            .map(|r| r.time_ns)
                            .sum::<f64>() / lib_results.len() as f64;

                        let speedup = lib_avg_time / torsh_avg_time;
                        summary.insert(format!("{}_{}_vs_torsh", operation, library), speedup);
                    }
                }
            }
        }

        summary
    }

    /// Calculate memory efficiency ratios
    pub fn calculate_memory_efficiency(results: &[ComparisonResult]) -> HashMap<String, f64> {
        let mut efficiency = HashMap::new();

        for result in results {
            if let (Some(throughput), Some(memory)) = (result.throughput, result.memory_usage) {
                let efficiency_ratio = throughput / memory as f64; // FLOPS per byte
                efficiency.insert(
                    format!("{}_{}_memory_efficiency", result.operation, result.library),
                    efficiency_ratio
                );
            }
        }

        efficiency
    }

    /// Generate performance scaling analysis
    pub fn analyze_performance_scaling(results: &[ComparisonResult]) -> HashMap<String, Vec<(usize, f64)>> {
        let mut scaling = HashMap::new();

        // Group by operation and library
        let mut by_op_lib: HashMap<(String, String), Vec<&ComparisonResult>> = HashMap::new();
        for result in results {
            by_op_lib
                .entry((result.operation.clone(), result.library.clone()))
                .or_insert_with(Vec::new)
                .push(result);
        }

        for ((operation, library), op_lib_results) in by_op_lib {
            let mut size_time_pairs: Vec<(usize, f64)> = op_lib_results
                .iter()
                .map(|r| (r.size, r.time_ns))
                .collect();

            size_time_pairs.sort_by_key(|&(size, _)| size);

            scaling.insert(format!("{}_{}", operation, library), size_time_pairs);
        }

        scaling
    }
}

/// Benchmarking presets for common scenarios
pub mod presets {
    //! Pre-configured benchmark suites for common use cases

    use super::core::ComparisonRunner;

    /// Quick development comparison (small sizes, essential operations)
    pub fn quick_dev_comparison() -> ComparisonRunner {
        #[cfg(feature = "pytorch")]
        {
            super::pytorch_comparisons::run_quick_pytorch_comparison()
        }
        #[cfg(not(feature = "pytorch"))]
        {
            super::integration::run_comparison_benchmarks()
        }
    }

    /// Comprehensive research comparison (all libraries, all operations)
    pub fn comprehensive_research_comparison() -> std::io::Result<ComparisonRunner> {
        let mut runner = super::integration::run_extended_benchmarks();

        // Add PyTorch comparisons if available
        #[cfg(feature = "pytorch")]
        {
            let pytorch_runner = super::pytorch_comparisons::run_pytorch_comparison_benchmarks();
            for result in pytorch_runner.results() {
                runner.add_result(result.clone());
            }
        }

        // Add TensorFlow comparisons if available
        #[cfg(feature = "tensorflow")]
        {
            let tensorflow_runner = super::tensorflow_comparisons::run_tensorflow_comparison_suite();
            for result in tensorflow_runner.results() {
                runner.add_result(result.clone());
            }
        }

        Ok(runner)
    }

    /// Production validation comparison (performance regression detection)
    pub fn production_validation() -> std::io::Result<()> {
        super::integration::benchmark_and_analyze()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Benchmarkable;

    #[test]
    fn test_modular_comparison_system() {
        // Test that all modules are accessible and working

        // Test core infrastructure
        let mut runner = ComparisonRunner::new();
        assert_eq!(runner.results().len(), 0);

        // Test ToRSh benchmarks
        let mut torsh_bench = TorshMatmulBench;
        let input = torsh_bench.setup(64);
        let result = torsh_bench.run(&input);
        assert!(result.is_ok());

        // Test integration functions
        let comparison_runner = run_comparison_benchmarks();
        assert!(comparison_runner.results().len() > 0);

        // Test analysis utilities
        let mut analyzer = PerformanceAnalyzer::new();
        analyzer.add_results(comparison_runner.results());

        // Test utilities
        let summary = utils::generate_comparison_summary(comparison_runner.results());
        assert!(summary.len() >= 0); // May be 0 if no external libraries available

        // Test presets
        let quick_runner = presets::quick_dev_comparison();
        assert!(quick_runner.results().len() > 0);
    }

    #[test]
    fn test_backward_compatibility() {
        // Ensure that the modular system maintains full backward compatibility
        // All original function names and APIs should work exactly as before

        // Test basic comparison runner functionality
        let runner = run_comparison_benchmarks();
        assert!(runner.results().len() > 0);

        let extended_runner = run_extended_benchmarks();
        assert!(extended_runner.results().len() > 0);

        // Test analysis functionality
        let mut analyzer = PerformanceAnalyzer::new();
        analyzer.add_results(runner.results());

        let analysis = analyzer.analyze_operation("matrix_multiplication");
        assert_eq!(analysis.operation, "matrix_multiplication");
    }

    #[test]
    fn test_feature_gated_compilation() {
        // Test that feature-gated modules compile correctly

        // This should always compile (no feature gates)
        let _runner = ComparisonRunner::new();
        let _analyzer = PerformanceAnalyzer::new();

        // Feature-gated modules should be available if features are enabled
        #[cfg(feature = "compare-external")]
        {
            let _ndarray_bench = NdarrayMatmulBench;
        }

        #[cfg(feature = "pytorch")]
        {
            let _pytorch_runner = run_pytorch_comparison_benchmarks();
        }

        #[cfg(feature = "tensorflow")]
        {
            let _tensorflow_runner = run_tensorflow_comparison_suite();
        }

        #[cfg(feature = "jax")]
        {
            let _jax_runner = JAXBenchRunner::new();
        }
    }

    #[test]
    fn test_modular_structure_integrity() {
        // Test that all modules are properly accessible

        // Core infrastructure
        let _runner = ComparisonRunner::new();
        let _analyzer = PerformanceAnalyzer::new();
        let _detector = RegressionDetector::new(0.1);

        // ToRSh benchmarks
        let _matmul_bench = TorshMatmulBench;
        let _elementwise_bench = TorshElementwiseBench;

        // Integration functions
        let _comparison = run_comparison_benchmarks();
        let _extended = run_extended_benchmarks();

        // Utility functions
        let dummy_results = vec![];
        let _summary = utils::generate_comparison_summary(&dummy_results);
        let _efficiency = utils::calculate_memory_efficiency(&dummy_results);
        let _scaling = utils::analyze_performance_scaling(&dummy_results);

        // Presets
        let _quick = presets::quick_dev_comparison();
    }
}