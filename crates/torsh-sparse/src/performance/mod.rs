//! # Comprehensive Performance Analysis Framework for Sparse Tensors
//!
//! This module provides a complete suite of performance analysis, benchmarking, and optimization
//! tools specifically designed for sparse tensor operations. The framework is organized into
//! specialized modules that work together to deliver comprehensive performance insights.
//!
//! ## Architecture Overview
//!
//! The performance analysis system is built around four core modules:
//!
//! - **core**: Fundamental types and utilities for measurements and timing
//! - **benchmarking**: Comprehensive profiling tools for sparse operations
//! - **memory_analysis**: Memory usage analysis and reporting capabilities
//! - **auto_tuning**: Intelligent format selection and optimization
//!
//! ## Quick Start
//!
//! ### Basic Performance Profiling
//!
//! ```rust
//! use torsh_sparse::performance::{SparseProfiler, BenchmarkConfig};
//! use torsh_sparse::{CooTensor, SparseFormat};
//! use torsh_core::Shape;
//!
//! // Create a profiler with fast benchmark configuration
//! let mut profiler = SparseProfiler::new(BenchmarkConfig::fast());
//!
//! // Create a test sparse tensor
//! let rows = vec![0, 0, 1, 2];
//! let cols = vec![0, 2, 1, 2];
//! let vals = vec![1.0, 2.0, 3.0, 4.0];
//! let shape = Shape::new(vec![3, 3]);
//! let coo = CooTensor::new(rows, cols, vals, shape)?;
//!
//! // Benchmark format conversion
//! let measurement = profiler.benchmark_format_conversion(&coo, SparseFormat::Csr)?;
//! println!("Conversion took: {:?}", measurement.duration);
//! ```
//!
//! ### Memory Analysis
//!
//! ```rust
//! use torsh_sparse::performance::MemoryAnalysis;
//! use torsh_sparse::SparseFormat;
//!
//! // Analyze memory usage for different formats
//! let coo_analysis = MemoryAnalysis::new(SparseFormat::Coo, 1000, (100, 100));
//! let csr_analysis = MemoryAnalysis::new(SparseFormat::Csr, 1000, (100, 100));
//!
//! println!("COO compression ratio: {:.2}x", coo_analysis.compression_ratio);
//! println!("CSR compression ratio: {:.2}x", csr_analysis.compression_ratio);
//!
//! let comparison = coo_analysis.compare_with(&csr_analysis);
//! println!("Better format: {:?}", comparison.better_format);
//! ```
//!
//! ### Auto-Tuning
//!
//! ```rust
//! use torsh_sparse::performance::AutoTuner;
//! use torsh_tensor::Tensor;
//!
//! let mut tuner = AutoTuner::new();
//! let dense = Tensor::randn(&[1000, 1000])?;
//!
//! // Automatically find optimal format for matrix multiplication
//! let optimal_format = tuner.find_optimal_format(&dense, "matmul", 1e-6)?;
//! println!("Optimal format: {:?}", optimal_format);
//!
//! // Get performance recommendations
//! let recommendations = tuner.get_recommendations();
//! for rec in recommendations {
//!     println!("Recommendation: {}", rec);
//! }
//! ```
//!
//! ### Comprehensive Performance Report
//!
//! ```rust
//! use torsh_sparse::performance::{SparseProfiler, PerformanceReport};
//!
//! let mut profiler = SparseProfiler::new(BenchmarkConfig::default());
//!
//! // Run multiple benchmarks...
//! // profiler.benchmark_*(...);
//!
//! // Generate comprehensive report
//! let report = profiler.generate_report();
//! println!("{}", report);
//!
//! // Find optimal operations
//! if let Some(fastest) = report.find_fastest_operation("conversion") {
//!     println!("Fastest conversion: {}", fastest.operation);
//! }
//! ```
//!
//! ## Performance Optimization Guidelines
//!
//! ### Format Selection
//!
//! | Use Case | Recommended Format | Rationale |
//! |----------|-------------------|-----------|
//! | Matrix multiplication | CSR | Efficient row access patterns |
//! | Transpose operations | CSC | Efficient column access patterns |
//! | Construction/modification | COO | Simple triplet format |
//! | Memory-constrained | Auto-tune | Let the system decide |
//!
//! ### Benchmarking Best Practices
//!
//! 1. **Warm-up iterations**: Always include warm-up runs to account for JIT compilation
//! 2. **Multiple measurements**: Use statistical analysis across multiple runs
//! 3. **Memory tracking**: Enable memory collection for comprehensive analysis
//! 4. **Realistic data**: Use representative matrix sizes and sparsity patterns
//! 5. **Operation context**: Consider the full operation pipeline, not just individual ops
//!
//! ### Auto-Tuning Configuration
//!
//! Choose the appropriate tuning configuration based on your use case:
//!
//! - **Conservative**: Thorough benchmarking with high confidence thresholds
//! - **Aggressive**: Fast tuning with lower improvement thresholds
//! - **Custom**: Tailored configuration for specific workloads
//!
//! ## Advanced Features
//!
//! ### Custom Metrics
//!
//! The performance framework supports custom metrics for domain-specific analysis:
//!
//! ```rust
//! use torsh_sparse::performance::PerformanceMeasurement;
//! use std::collections::HashMap;
//!
//! let mut measurement = PerformanceMeasurement::new("custom_op".to_string());
//! measurement.add_metric("flops".to_string(), 1000000.0);
//! measurement.add_metric("efficiency".to_string(), 0.85);
//!
//! if let Some(ops_per_sec) = measurement.ops_per_second() {
//!     println!("Operations per second: {:.2}", ops_per_sec);
//! }
//! ```
//!
//! ### Memory Efficiency Analysis
//!
//! ```rust
//! use torsh_sparse::performance::MemoryAnalysis;
//!
//! let analysis = MemoryAnalysis::new(format, nnz, (rows, cols));
//!
//! println!("Memory efficiency: {:.2}%", analysis.memory_efficiency() * 100.0);
//! println!("Sparsity level: {:.2}%", analysis.sparsity_level() * 100.0);
//! println!("Efficiently compressed: {}", analysis.is_efficiently_compressed());
//! ```
//!
//! ## Integration with Sparse Tensor Operations
//!
//! The performance framework integrates seamlessly with sparse tensor operations:
//!
//! ```rust
//! use torsh_sparse::performance::{AutoTuner, SparseProfiler};
//! use torsh_sparse::{SparseTensor, SparseFormat};
//!
//! // Auto-tune format for specific operation
//! let mut tuner = AutoTuner::new();
//! let optimal_format = tuner.find_optimal_format(&dense_matrix, "matmul", 1e-6)?;
//!
//! // Convert to optimal format
//! let sparse = match optimal_format {
//!     SparseFormat::Coo => dense_matrix.to_sparse_coo(1e-6)?,
//!     SparseFormat::Csr => dense_matrix.to_sparse_csr(1e-6)?,
//!     SparseFormat::Csc => dense_matrix.to_sparse_csc(1e-6)?,
//! };
//!
//! // Profile the actual operation
//! let mut profiler = SparseProfiler::new(BenchmarkConfig::default());
//! let measurement = profiler.benchmark_sparse_matmul(&sparse, &other_sparse)?;
//! ```
//!
//! ## Thread Safety and Concurrency
//!
//! All performance analysis tools are designed to be thread-safe where appropriate:
//!
//! - **PerformanceMeasurement**: Immutable after creation, safe to share
//! - **BenchmarkConfig**: Read-only configuration, safe to share
//! - **MemoryAnalysis**: Immutable analysis results, safe to share
//! - **SparseProfiler**: Not thread-safe, use separate instances per thread
//! - **AutoTuner**: Contains mutable cache, requires synchronization for concurrent access

// Core module exports
pub mod core;
pub mod benchmarking;
pub mod memory_analysis;
pub mod auto_tuning;

// Re-export core types and utilities
pub use core::{
    PerformanceMeasurement, BenchmarkConfig,
    memory::{MemoryTracker},
    timing::{Timer, measure_time, measure_time_async},
};

// Re-export benchmarking functionality
pub use benchmarking::SparseProfiler;

// Re-export memory analysis types
pub use memory_analysis::{
    MemoryAnalysis, MemoryComparison, OperationStatistics,
    PerformanceReport, ReportSummary,
};

// Re-export auto-tuning capabilities
pub use auto_tuning::{
    AutoTuner, TuningConfig, TuningReport,
};

/// Convenience function to create a comprehensive performance analyzer
///
/// Creates a fully configured performance analysis setup with sensible defaults
/// for most use cases. This includes a profiler, auto-tuner, and report generator.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::create_performance_analyzer;
///
/// let (mut profiler, mut tuner) = create_performance_analyzer();
///
/// // Use profiler for detailed benchmarking
/// // Use tuner for automatic format optimization
/// ```
pub fn create_performance_analyzer() -> (SparseProfiler, AutoTuner) {
    let config = BenchmarkConfig::default();
    let profiler = SparseProfiler::new(config.clone());

    let tuning_config = TuningConfig {
        benchmark_config: config,
        ..TuningConfig::default()
    };
    let tuner = AutoTuner::with_config(tuning_config);

    (profiler, tuner)
}

/// Convenience function to create a fast performance analyzer for quick analysis
///
/// Uses fast benchmark configurations suitable for development and quick testing.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::create_fast_analyzer;
///
/// let (mut profiler, mut tuner) = create_fast_analyzer();
/// // Performs faster but less comprehensive analysis
/// ```
pub fn create_fast_analyzer() -> (SparseProfiler, AutoTuner) {
    let config = BenchmarkConfig::fast();
    let profiler = SparseProfiler::new(config.clone());

    let tuning_config = TuningConfig {
        benchmark_config: config,
        ..TuningConfig::aggressive()
    };
    let tuner = AutoTuner::with_config(tuning_config);

    (profiler, tuner)
}

/// Convenience function to create a comprehensive performance analyzer
///
/// Uses thorough benchmark configurations suitable for production optimization
/// and detailed performance analysis.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::create_comprehensive_analyzer;
///
/// let (mut profiler, mut tuner) = create_comprehensive_analyzer();
/// // Performs thorough but slower analysis
/// ```
pub fn create_comprehensive_analyzer() -> (SparseProfiler, AutoTuner) {
    let config = BenchmarkConfig::comprehensive();
    let profiler = SparseProfiler::new(config.clone());

    let tuning_config = TuningConfig {
        benchmark_config: config,
        ..TuningConfig::conservative()
    };
    let tuner = AutoTuner::with_config(tuning_config);

    (profiler, tuner)
}

/// Analyze memory usage for a specific sparse format and matrix characteristics
///
/// This is a convenience function for quick memory analysis without setting up
/// the full performance framework.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::analyze_memory_usage;
/// use torsh_sparse::SparseFormat;
///
/// let analysis = analyze_memory_usage(SparseFormat::Csr, 10000, (1000, 1000));
/// println!("Compression ratio: {:.2}x", analysis.compression_ratio);
/// ```
pub fn analyze_memory_usage(
    format: crate::SparseFormat,
    nnz: usize,
    dimensions: (usize, usize),
) -> MemoryAnalysis {
    MemoryAnalysis::new(format, nnz, dimensions)
}

/// Compare memory usage between different sparse formats
///
/// Convenience function to quickly compare memory characteristics of different
/// sparse formats for the same matrix.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance::compare_format_memory;
/// use torsh_sparse::SparseFormat;
///
/// let comparison = compare_format_memory(
///     SparseFormat::Coo,
///     SparseFormat::Csr,
///     1000,
///     (100, 100)
/// );
///
/// println!("Better format: {:?}", comparison.better_format);
/// ```
pub fn compare_format_memory(
    format_a: crate::SparseFormat,
    format_b: crate::SparseFormat,
    nnz: usize,
    dimensions: (usize, usize),
) -> MemoryComparison {
    let analysis_a = MemoryAnalysis::new(format_a, nnz, dimensions);
    let analysis_b = MemoryAnalysis::new(format_b, nnz, dimensions);
    analysis_a.compare_with(&analysis_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convenience_functions() {
        // Test analyzer creation functions
        let (profiler, tuner) = create_performance_analyzer();
        assert_eq!(profiler.config().warmup_iterations, 3); // Default config

        let (fast_profiler, fast_tuner) = create_fast_analyzer();
        assert_eq!(fast_profiler.config().warmup_iterations, 1); // Fast config

        let (comp_profiler, comp_tuner) = create_comprehensive_analyzer();
        assert_eq!(comp_profiler.config().warmup_iterations, 10); // Comprehensive config
    }

    #[test]
    fn test_memory_analysis_convenience() {
        let analysis = analyze_memory_usage(crate::SparseFormat::Coo, 1000, (100, 100));
        assert_eq!(analysis.nnz, 1000);
        assert_eq!(analysis.matrix_dimensions, (100, 100));
        assert!(analysis.compression_ratio > 1.0);
    }

    #[test]
    fn test_format_memory_comparison() {
        let comparison = compare_format_memory(
            crate::SparseFormat::Coo,
            crate::SparseFormat::Csr,
            1000,
            (100, 100),
        );

        assert_eq!(comparison.format_a, crate::SparseFormat::Coo);
        assert_eq!(comparison.format_b, crate::SparseFormat::Csr);
        assert!(comparison.memory_ratio > 0.0);
    }

    #[test]
    fn test_module_re_exports() {
        // Test that all major types are accessible through the module interface
        let _measurement = PerformanceMeasurement::new("test".to_string());
        let _config = BenchmarkConfig::default();
        let _profiler = SparseProfiler::new(BenchmarkConfig::fast());
        let _tuner = AutoTuner::new();
        let _analysis = MemoryAnalysis::new(crate::SparseFormat::Coo, 100, (10, 10));
        let _report = PerformanceReport::new();
        let _timer = Timer::start();
        let _tracker = MemoryTracker::new();
    }

    #[test]
    fn test_analyzer_configurations() {
        let (default_profiler, _) = create_performance_analyzer();
        let (fast_profiler, _) = create_fast_analyzer();
        let (comp_profiler, _) = create_comprehensive_analyzer();

        // Verify different configurations
        assert!(fast_profiler.config().warmup_iterations < default_profiler.config().warmup_iterations);
        assert!(default_profiler.config().warmup_iterations < comp_profiler.config().warmup_iterations);
    }
}