//! Comprehensive benchmarking suite for optimizers
//!
//! This module provides a complete toolkit for evaluating and comparing optimizer performance
//! across diverse optimization scenarios. It's designed for both research and production
//! use cases to help you choose the best optimizer for your specific needs.
//!
//! ## Features
//!
//! ### Core Benchmarks
//! - **Step Performance**: Raw computational speed measurement
//! - **Convergence Speed**: How quickly optimizers reach optimal solutions
//! - **Memory Usage**: Memory consumption and scaling characteristics
//! - **Sparse Gradients**: Performance with different sparsity patterns
//!
//! ### Advanced Analytics
//! - **Side-by-side Comparisons**: Statistical comparison between optimizers
//! - **Convergence Analysis**: Detailed convergence behavior analysis
//! - **Hardware Utilization**: CPU/Memory usage tracking (when available)
//! - **Domain-specific Benchmarks**: Tests tailored for CV, NLP, and RL
//!
//! ### Export & Visualization
//! - **JSON/CSV Export**: Data export for external analysis
//! - **Statistical Tests**: Determine if performance differences are significant
//! - **Benchmark Reports**: Comprehensive HTML reports with visualizations
//! - **Performance Regression**: Track optimizer performance over time
//!
//! ## Quick Start
//!
//! ```rust
//! use torsh_optim::benchmarks::*;
//! use torsh_optim::{Adam, SGD, AdamW};
//!
//! // Create benchmark suite
//! let benchmarks = OptimizerBenchmarks::new();
//!
//! // Compare optimizers side-by-side
//! let comparison = OptimizerComparison::new()
//!     .add_optimizer("Adam", || Adam::new(params.clone(), None, None, None, None, false))
//!     .add_optimizer("AdamW", || AdamW::new(params.clone(), None, None, None, Some(0.01), false))
//!     .add_optimizer("SGD", || SGD::new(params.clone(), 0.01, Some(0.9), None, None, false));
//!
//! // Run comprehensive benchmarks
//! let results = comparison.run_comparison_suite()?;
//! comparison.print_comparison_table(&results);
//! ```
//!
//! ## Domain-Specific Benchmarks
//!
//! ### Computer Vision
//! ```rust
//! let cv_bench = CVBenchmarks::new();
//! let results = cv_bench.benchmark_resnet_training(optimizer)?;
//! ```
//!
//! ### Natural Language Processing
//! ```rust
//! let nlp_bench = NLPBenchmarks::new();
//! let results = nlp_bench.benchmark_transformer_training(optimizer)?;
//! ```
//!
//! ### Reinforcement Learning
//! ```rust
//! let rl_bench = RLBenchmarks::new();
//! let results = rl_bench.benchmark_policy_gradient(optimizer)?;
//! ```
//!
//! ## Performance Analysis
//!
//! The benchmark suite provides detailed statistical analysis:
//!
//! - **Execution Time**: Mean, median, std dev, confidence intervals
//! - **Convergence Metrics**: Convergence rate, final loss, iteration count
//! - **Memory Analysis**: Peak usage, growth patterns, efficiency scores
//! - **Statistical Significance**: P-values, effect sizes, confidence intervals
//!
//! ## Exporting Results
//!
//! ```rust
//! // Export to JSON for analysis
//! results.export_json("benchmark_results.json")?;
//!
//! // Export to CSV for spreadsheet analysis
//! results.export_csv("benchmark_results.csv")?;
//!
//! // Generate comprehensive HTML report
//! results.generate_html_report("benchmark_report.html")?;
//! ```

pub mod comparison;
pub mod core;
pub mod domain_specific;
pub mod optimizer;
pub mod utils;

// Re-export the main types for convenience
pub use core::{BenchmarkConfig, BenchmarkResult, MemoryStats, StatisticalAnalysis};

pub use optimizer::OptimizerBenchmarks;

pub use comparison::{ComparisonResult, OptimizerComparison};

pub use domain_specific::{CVBenchmarks, NLPBenchmarks, RLBenchmarks};

pub use utils::{
    benchmark_scaling, create_comprehensive_config, create_test_config, generate_summary_report,
    run_domain_benchmarks, run_quick_benchmark_suite, run_quick_optimizer_comparison,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Test that we can create instances of the main types
        let _benchmarks = OptimizerBenchmarks::new();
        let _comparison = OptimizerComparison::<crate::sgd::SGD>::new();
        let _cv_bench = CVBenchmarks::new();
        let _nlp_bench = NLPBenchmarks::new();
        let _rl_bench = RLBenchmarks::new();

        // Test utility functions
        let _test_config = create_test_config();
        let _comp_config = create_comprehensive_config();
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.num_iterations, 1000);
        assert_eq!(config.warmup_iterations, 100);
        assert_eq!(config.max_time_seconds, 60.0);
        assert!(!config.profile_memory);
    }
}
