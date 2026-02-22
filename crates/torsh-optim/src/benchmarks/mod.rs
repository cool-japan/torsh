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
//! ```rust,no_run
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::benchmarks::*;
//! use torsh_optim::Adam;
//!
//! // Create some parameters
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//! let params_clone1 = params.clone();
//! let params_clone2 = params.clone();
//! let params_clone3 = params.clone();
//!
//! // Create benchmark suite
//! let benchmarks = OptimizerBenchmarks::new();
//!
//! // Compare different configurations of the same optimizer
//! let comparison = OptimizerComparison::new()
//!     .add_optimizer("Adam-1e-3", move || Ok(Adam::new(params_clone1.clone(), Some(0.001), None, None, None, false)))
//!     .add_optimizer("Adam-1e-4", move || Ok(Adam::new(params_clone2.clone(), Some(0.0001), None, None, None, false)))
//!     .add_optimizer("Adam-AMSGrad", move || Ok(Adam::new(params_clone3.clone(), Some(0.001), None, None, None, true)));
//!
//! // Run comprehensive benchmarks
//! let results = comparison.run_comparison_suite()?;
//! comparison.print_comparison_table(&results);
//! # Ok(())
//! # }
//! ```
//!
//! ## Domain-Specific Benchmarks
//!
//! ### Computer Vision
//! ```rust,no_run
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::benchmarks::domain_specific::CVBenchmarks;
//! use torsh_optim::Adam;
//!
//! // Create some parameters and optimizer
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//! let optimizer = Adam::new(params, None, None, None, None, false);
//!
//! let cv_bench = CVBenchmarks::new();
//! let results = cv_bench.benchmark_resnet_training(optimizer)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Natural Language Processing
//! ```rust,no_run
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::benchmarks::domain_specific::NLPBenchmarks;
//! use torsh_optim::AdamW;
//!
//! // Create some parameters and optimizer
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//! let optimizer = AdamW::new(params, Some(5e-5), None, None, Some(0.01), false);
//!
//! let nlp_bench = NLPBenchmarks::new();
//! let results = nlp_bench.benchmark_transformer_training(optimizer)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Reinforcement Learning
//! ```rust,no_run
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::benchmarks::domain_specific::RLBenchmarks;
//! use torsh_optim::Adam;
//!
//! // Create some parameters and optimizer
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//! let optimizer = Adam::new(params, Some(3e-4), None, None, None, false);
//!
//! let rl_bench = RLBenchmarks::new();
//! let results = rl_bench.benchmark_policy_gradient(optimizer)?;
//! # Ok(())
//! # }
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
//! ```rust,no_run
//! # use torsh_tensor::creation::randn;
//! # use torsh_core::error::Result;
//! # use parking_lot::RwLock;
//! # use std::sync::Arc;
//! # fn main() -> Result<()> {
//! use torsh_optim::benchmarks::*;
//! use torsh_optim::Adam;
//!
//! // Create some parameters and run benchmarks
//! let param1 = Arc::new(RwLock::new(randn::<f32>(&[10, 20])?));
//! let params = vec![param1];
//! let params_clone = params.clone();
//! let comparison = OptimizerComparison::new()
//!     .add_optimizer("Adam", move || Ok(Adam::new(params_clone.clone(), None, None, None, None, false)));
//! let results = comparison.run_comparison_suite()?;
//!
//! // Export results (examples - actual paths may vary)
//! // results.export_json("benchmark_results.json")?;
//! // results.export_csv("benchmark_results.csv")?;
//! // results.generate_html_report("benchmark_report.html")?;
//! # Ok(())
//! # }
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
