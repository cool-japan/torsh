//! # Unified Performance Tools Interface
//!
//! This module provides a comprehensive performance analysis and profiling system for sparse
//! tensor operations. The system is built on a modular architecture with specialized components
//! for different aspects of performance analysis.
//!
//! ## Architecture Overview
//!
//! The performance tools system is organized into 5 specialized modules:
//!
//! - **core**: Foundation profiling infrastructure with measurement and benchmarking
//! - **memory**: Memory analysis, compression tracking, and cache performance
//! - **reporting**: Statistics aggregation, report generation, and performance analysis
//! - **optimization**: Auto-tuning, hardware optimization, and intelligent recommendations
//! - **export**: Data export, visualization, and TensorBoard integration
//!
//! ## Quick Start
//!
//! ```rust
//! use torsh_sparse::performance_tools::{SparseProfiler, BenchmarkConfig};
//!
//! // Create a profiler with custom configuration
//! let config = BenchmarkConfig::thorough();
//! let mut profiler = SparseProfiler::new(config);
//!
//! // Profile sparse format conversion
//! let measurements = profiler.benchmark_format_conversion(&dense_matrix)?;
//!
//! // Generate comprehensive report
//! let report = PerformanceReport::default() // TODO: Replace with actual report generation;
//! println!("{}", report);
//! ```
//!
//! ## Advanced Usage
//!
//! ```rust
//! use torsh_sparse::performance_tools::{
//!     SparseProfiler, AutoTuner, TrendAnalyzer, PerformanceExporter,
//!     OptimizationStrategy, InputCharacteristics, DistributionPattern
//! };
//!
//! // Auto-tune sparse format selection
//! let mut tuner = AutoTuner::new().with_strategy(OptimizationStrategy::Speed);
//! let characteristics = InputCharacteristics {
//!     dimensions: (10000, 10000),
//!     sparsity: 0.95,
//!     distribution_pattern: DistributionPattern::Random,
//!     operation_types: vec![OperationType::MatrixVector],
//!     memory_budget: Some(1024 * 1024 * 1024), // 1GB
//! };
//! let tuning_result = tuner.find_optimal_format(&characteristics)?;
//!
//! // Analyze performance trends
//! let mut trend_analyzer = TrendAnalyzer::new();
//! let regressions = trend_analyzer.detect_regressions(10.0); // 10% threshold
//!
//! // Export to various formats
//! let csv_data = PerformanceExporter::to_csv(&report)?;
//! let json_data = PerformanceExporter::to_json(&report)?;
//! ```
//!
//! ## Performance Considerations
//!
//! - **Memory-efficient profiling**: Uses lazy evaluation and streaming for large datasets
//! - **Hardware optimization**: Leverages SIMD, cache optimization, and NUMA awareness
//! - **Adaptive algorithms**: Auto-tuning based on data characteristics and hardware capabilities
//! - **Minimal overhead**: Optimized measurement infrastructure with configurable precision
//! - **Scalable analysis**: Supports both single-threaded and distributed profiling scenarios

// Re-export all modules for internal organization
pub mod core;
pub mod export;
pub mod memory;
pub mod optimization;
pub mod reporting;

// Core profiling infrastructure
pub use core::{BenchmarkConfig, PerformanceMeasurement, SparseProfiler};

// Memory analysis capabilities
pub use memory::{
    analyze_sparse_memory, benchmark_cache_performance, CachePerformanceResult,
    MemoryAccessPattern, MemoryAnalysis, MemoryBreakdown, MemoryTracker, MemoryUsageResult,
};

// Reporting and statistics
pub use reporting::{
    MemoryStatistics, MetricStatistics, OperationStatistics, PerformanceReport, PerformanceSummary,
    StatisticsCollector,
};

// Auto-tuning and optimization
pub use optimization::{
    AutoTuner, CacheInfo, CpuInfo, DistributionPattern, HardwareBenchmark, InputCharacteristics,
    MemoryInfo, OperationType, OptimizationStrategy, SystemCapabilityReport, SystemInfo,
    TuningResult,
};

// Export and visualization
pub use export::{
    PerformanceExporter, PlotData, TensorBoardExporter, TrendAnalysis, TrendAnalyzer,
    TrendDirection,
};

use crate::{SparseTensor, TorshError, TorshResult};
use std::collections::HashMap;

/// Convenience function to create a default profiler with balanced configuration
///
/// This is the recommended starting point for most performance analysis tasks.
/// It uses a balanced configuration that provides good accuracy without excessive overhead.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance_tools::default_profiler;
///
/// let mut profiler = default_profiler();
/// // Use profiler for benchmarking...
/// ```
pub fn default_profiler() -> SparseProfiler {
    SparseProfiler::new(BenchmarkConfig::default())
}

/// Create a fast profiler for quick performance checks
///
/// This configuration minimizes overhead and is suitable for quick performance
/// checks during development or CI/CD pipelines.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance_tools::fast_profiler;
///
/// let mut profiler = fast_profiler();
/// // Quick performance check...
/// ```
pub fn fast_profiler() -> SparseProfiler {
    SparseProfiler::new(BenchmarkConfig::fast())
}

/// Create a thorough profiler for detailed analysis
///
/// This configuration provides maximum detail and accuracy for comprehensive
/// performance analysis and optimization work.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance_tools::thorough_profiler;
///
/// let mut profiler = thorough_profiler();
/// // Detailed performance analysis...
/// ```
pub fn thorough_profiler() -> SparseProfiler {
    SparseProfiler::new(BenchmarkConfig::thorough())
}

/// Create a memory-focused profiler for memory analysis
///
/// This configuration emphasizes memory usage tracking and analysis,
/// suitable for memory optimization tasks.
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance_tools::memory_profiler;
///
/// let mut profiler = memory_profiler();
/// // Memory-focused analysis...
/// ```
pub fn memory_profiler() -> SparseProfiler {
    SparseProfiler::new(BenchmarkConfig::memory_focused())
}

/// Comprehensive performance analysis function
///
/// This function provides a one-stop solution for complete performance analysis
/// of sparse tensor operations, combining profiling, auto-tuning, and reporting.
///
/// # Arguments
///
/// * `sparse_tensors` - Collection of sparse tensors to analyze
/// * `operations` - Operations to benchmark
/// * `export_path` - Optional path for exporting results
///
/// # Returns
///
/// Comprehensive analysis report with recommendations
///
/// # Examples
///
/// ```rust
/// use torsh_sparse::performance_tools::comprehensive_analysis;
///
/// let tensors = vec![&coo_tensor, &csr_tensor];
/// let operations = vec!["matmul", "transpose"];
/// let analysis = comprehensive_analysis(&tensors, &operations, Some("results/"))?;
/// ```
pub fn comprehensive_analysis(
    sparse_tensors: &[&dyn SparseTensor],
    operations: &[&str],
    export_path: Option<&str>,
) -> TorshResult<ComprehensiveAnalysisReport> {
    let mut profiler = default_profiler();
    let mut auto_tuner = AutoTuner::new();
    let mut hardware_benchmark = HardwareBenchmark::new();
    let mut trend_analyzer = TrendAnalyzer::new();
    let mut exporter = TensorBoardExporter::new();

    // 1. Hardware capability analysis
    let system_capabilities = hardware_benchmark.analyze_system_capabilities()?;

    // 2. Memory analysis for each tensor
    let mut memory_analyses = Vec::new();
    for tensor in sparse_tensors {
        let analysis = analyze_sparse_memory(*tensor)?;
        memory_analyses.push(analysis);
    }

    // 3. Performance profiling
    let mut all_measurements = Vec::new();
    for tensor in sparse_tensors {
        // Auto-tune format selection
        let characteristics = infer_characteristics(*tensor);
        let tuning_result = auto_tuner.find_optimal_format(&characteristics)?;

        // Profile operations based on auto-tuning results
        for &operation in operations {
            let measurements = match operation {
                "matmul" => {
                    if sparse_tensors.len() >= 2 {
                        profiler.benchmark_sparse_matmul(*tensor, sparse_tensors[1])?
                    } else {
                        continue;
                    }
                }
                "format_conversion" => {
                    // Create a dummy dense matrix for conversion benchmarking
                    let shape = tensor.shape();
                    let dummy_dense = create_dummy_dense_matrix(shape.dims()[0], shape.dims()[1])?;
                    profiler.benchmark_format_conversion(&dummy_dense)?
                }
                "dense_to_sparse" => {
                    let shape = tensor.shape();
                    let dummy_dense = create_dummy_dense_matrix(shape.dims()[0], shape.dims()[1])?;
                    profiler.benchmark_dense_to_sparse(&dummy_dense, 0.01)?
                }
                _ => continue,
            };
            all_measurements.extend(measurements);
        }
    }

    // 4. Generate comprehensive report
    let mut report = PerformanceReport::new();
    report.add_measurements(&profiler.measurements);

    // Add memory analyses to the report
    for analysis in &memory_analyses {
        report.add_memory_analysis(analysis.clone());
    }

    // Add metadata about the analysis
    report.add_metadata("analysis_type".to_string(), "comprehensive".to_string());
    report.add_metadata(
        "tensors_analyzed".to_string(),
        sparse_tensors.len().to_string(),
    );
    report.add_metadata("operations_profiled".to_string(), operations.join(","));
    report.add_metadata(
        "memory_analyses_count".to_string(),
        memory_analyses.len().to_string(),
    );

    // 5. Export results if path provided
    if let Some(path) = export_path {
        // Export comprehensive report to CSV
        let csv_data = PerformanceExporter::to_csv(&report)?;
        std::fs::write(format!("{}/performance_report.csv", path), csv_data)
            .map_err(|e| TorshError::IoError(format!("Failed to write CSV report: {}", e)))?;

        // Export comprehensive report to JSON
        let json_data = PerformanceExporter::to_json(&report)?;
        std::fs::write(format!("{}/performance_report.json", path), json_data)
            .map_err(|e| TorshError::IoError(format!("Failed to write JSON report: {}", e)))?;

        // Export detailed analysis report
        exporter.export_report(&report, path)?;

        // Export memory analyses separately for detailed analysis
        for (i, analysis) in memory_analyses.iter().enumerate() {
            let analysis_path = format!("{}/memory_analysis_{}.txt", path, i);
            let analysis_text = format!("{:#?}", analysis);
            std::fs::write(&analysis_path, analysis_text).map_err(|e| {
                TorshError::IoError(format!("Failed to write memory analysis: {}", e))
            })?;
        }
    }

    // 6. Generate optimization recommendations
    let auto_tuning_recommendations = auto_tuner.get_recommendations();
    let hardware_recommendations = system_capabilities.recommendations.clone();
    let performance_recommendations = report.get_recommendations();

    // Generate enhanced memory-aware recommendations
    let memory_budget = None; // Could be passed as parameter in future
    let operations_strings: Vec<String> = operations.iter().map(|s| s.to_string()).collect();
    let memory_recommendations =
        crate::performance_tools::memory::generate_memory_optimization_recommendations(
            &memory_analyses,
            memory_budget,
            &operations_strings,
        );

    // Combine all recommendations into comprehensive analysis
    let mut all_recommendations = Vec::new();
    all_recommendations.extend(auto_tuning_recommendations.clone());
    all_recommendations.extend(hardware_recommendations.clone());
    all_recommendations.extend(performance_recommendations.clone());
    all_recommendations.extend(memory_recommendations.clone());

    // Create comprehensive analysis report
    let comprehensive_report = ComprehensiveAnalysisReport {
        performance_report: report,
        memory_analyses,
        system_capabilities,
        auto_tuning_recommendations,
        hardware_recommendations,
        performance_recommendations,
        export_path: export_path.map(|p| p.to_string()),
    };

    Ok(comprehensive_report)
}

/// Result of comprehensive performance analysis
#[derive(Debug, Clone)]
pub struct ComprehensiveAnalysisReport {
    /// Detailed performance report with operation statistics
    pub performance_report: PerformanceReport,
    /// Memory analysis results for each tensor
    pub memory_analyses: Vec<MemoryAnalysis>,
    /// System capability analysis
    pub system_capabilities: SystemCapabilityReport,
    /// Auto-tuning recommendations
    pub auto_tuning_recommendations: Vec<String>,
    /// Hardware-specific recommendations
    pub hardware_recommendations: Vec<String>,
    /// Performance optimization recommendations
    pub performance_recommendations: Vec<String>,
    /// Export path if results were exported
    pub export_path: Option<String>,
}

impl ComprehensiveAnalysisReport {
    /// Get all recommendations in a single list
    pub fn all_recommendations(&self) -> Vec<String> {
        let mut all_recs = Vec::new();
        all_recs.extend(self.auto_tuning_recommendations.clone());
        all_recs.extend(self.hardware_recommendations.clone());
        all_recs.extend(self.performance_recommendations.clone());
        all_recs
    }

    /// Get a summary of the analysis
    pub fn summary(&self) -> String {
        format!(
            "Performance Analysis Summary:\n\
             - Total measurements: {}\n\
             - Operations analyzed: {}\n\
             - Memory analyses: {}\n\
             - Recommendations: {}\n\
             - Export path: {}",
            self.performance_report.total_measurements,
            self.performance_report.operation_count,
            self.memory_analyses.len(),
            self.all_recommendations().len(),
            self.export_path.as_deref().unwrap_or("None")
        )
    }

    /// Get the most critical recommendations
    ///
    /// Returns up to 5 most important recommendations based on potential impact
    pub fn top_recommendations(&self) -> Vec<String> {
        let mut all_recs = self.all_recommendations();

        // Prioritize hardware and auto-tuning recommendations as they typically have higher impact
        let mut prioritized = Vec::new();

        // Add hardware recommendations first (usually most impactful)
        prioritized.extend(self.hardware_recommendations.iter().take(3).cloned());

        // Add auto-tuning recommendations
        prioritized.extend(self.auto_tuning_recommendations.iter().take(2).cloned());

        // Fill remaining slots with performance recommendations if space available
        let remaining_slots = 5usize.saturating_sub(prioritized.len());
        prioritized.extend(
            self.performance_recommendations
                .iter()
                .take(remaining_slots)
                .cloned(),
        );

        prioritized
    }

    /// Get performance bottlenecks analysis
    ///
    /// Analyzes performance data to identify the most significant bottlenecks
    pub fn bottleneck_analysis(&self) -> Vec<String> {
        let mut bottlenecks = Vec::new();

        // Analyze operation statistics to find bottlenecks
        if let Some((slowest_op, _)) = self
            .performance_report
            .operation_statistics
            .iter()
            .max_by_key(|(_, stats)| stats.max_time)
        {
            bottlenecks.push(format!(
                "Slowest operation: {} (consider optimization)",
                slowest_op
            ));
        }

        // Analyze memory efficiency
        let avg_compression_ratio: f32 = self
            .memory_analyses
            .iter()
            .map(|analysis| analysis.compression_ratio)
            .sum::<f32>()
            / self.memory_analyses.len() as f32;

        if avg_compression_ratio < 2.0 {
            bottlenecks.push(
                "Low compression ratio detected (consider different sparse format)".to_string(),
            );
        }

        // Check for high memory overhead
        let high_overhead_count = self
            .memory_analyses
            .iter()
            .filter(|analysis| analysis.overhead_per_nnz > 16.0) // More than 16 bytes per nnz is high
            .count();

        if high_overhead_count > 0 {
            bottlenecks.push(format!(
                "High memory overhead detected in {} tensor(s) (consider format optimization)",
                high_overhead_count
            ));
        }

        bottlenecks
    }

    /// Generate optimization priority matrix
    ///
    /// Returns recommendations sorted by potential impact and implementation difficulty
    pub fn optimization_priorities(&self) -> Vec<(String, String, String)> {
        let mut priorities = Vec::new();

        // Hardware recommendations - usually high impact, medium difficulty
        for rec in &self.hardware_recommendations {
            priorities.push((rec.clone(), "High".to_string(), "Medium".to_string()));
        }

        // Auto-tuning recommendations - medium impact, low difficulty
        for rec in &self.auto_tuning_recommendations {
            priorities.push((rec.clone(), "Medium".to_string(), "Low".to_string()));
        }

        // Performance recommendations - varies
        for rec in &self.performance_recommendations {
            priorities.push((rec.clone(), "Medium".to_string(), "Medium".to_string()));
        }

        priorities
    }
}

/// Quick performance check for a single sparse tensor
///
/// This function provides a simplified interface for quick performance
/// assessment of a single sparse tensor.
///
/// # Arguments
///
/// * `sparse_tensor` - Sparse tensor to analyze
///
/// # Returns
///
/// Quick performance summary
pub fn quick_performance_check(
    sparse_tensor: &dyn SparseTensor,
) -> TorshResult<QuickPerformanceResult> {
    let mut profiler = fast_profiler();

    // Memory analysis
    let memory_analysis = analyze_sparse_memory(sparse_tensor)?;

    // Basic profiling
    let shape = sparse_tensor.shape();
    let dummy_dense = create_dummy_dense_matrix(shape.dims()[0], shape.dims()[1])?;
    let format_measurements = profiler.benchmark_format_conversion(&dummy_dense)?;

    // Generate basic report
    let report = PerformanceReport::default(); // TODO: Replace with actual report generation
    let summary = report.performance_summary();

    // Generate recommendations before moving memory_analysis
    let recommendations = generate_quick_recommendations(&memory_analysis, &summary);

    Ok(QuickPerformanceResult {
        memory_analysis,
        performance_summary: summary,
        format_conversion_time: format_measurements
            .iter()
            .find(|m| {
                m.operation
                    .contains(&format!("{:?}", sparse_tensor.format()))
            })
            .map(|m| m.duration)
            .unwrap_or_default(),
        recommendations,
    })
}

/// Result of quick performance check
#[derive(Debug, Clone)]
pub struct QuickPerformanceResult {
    /// Memory analysis of the tensor
    pub memory_analysis: MemoryAnalysis,
    /// Performance summary
    pub performance_summary: PerformanceSummary,
    /// Format conversion time
    pub format_conversion_time: std::time::Duration,
    /// Quick recommendations
    pub recommendations: Vec<String>,
}

impl std::fmt::Display for QuickPerformanceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Quick Performance Check ===")?;
        writeln!(f, "Format: {:?}", self.memory_analysis.format)?;
        writeln!(
            f,
            "Compression Ratio: {:.1}x",
            self.memory_analysis.compression_ratio
        )?;
        writeln!(
            f,
            "Memory Efficiency: {}",
            self.memory_analysis.memory_efficiency_rating()
        )?;
        writeln!(
            f,
            "Performance Grade: {}",
            self.performance_summary.performance_grade()
        )?;
        writeln!(
            f,
            "Format Conversion: {:.3}ms",
            self.format_conversion_time.as_secs_f64() * 1000.0
        )?;
        writeln!(f)?;
        writeln!(f, "Recommendations:")?;
        for rec in &self.recommendations {
            writeln!(f, "  • {}", rec)?;
        }
        Ok(())
    }
}

// Helper functions

/// Infer characteristics from a sparse tensor for auto-tuning
fn infer_characteristics(sparse_tensor: &dyn SparseTensor) -> InputCharacteristics {
    let shape = sparse_tensor.shape();
    let nnz = sparse_tensor.nnz();
    let total_elements = shape.dims()[0] * shape.dims()[1];
    let sparsity = 1.0 - (nnz as f64 / total_elements as f64);

    // Infer distribution pattern based on sparsity and format
    let distribution_pattern = match sparse_tensor.format() {
        crate::SparseFormat::Csr => {
            if sparsity > 0.95 {
                DistributionPattern::Random
            } else {
                DistributionPattern::RowClustered
            }
        }
        crate::SparseFormat::Csc => DistributionPattern::ColumnClustered,
        crate::SparseFormat::Coo => DistributionPattern::Random,
        _ => DistributionPattern::Random, // Default case
    };

    // Default operation types
    let operation_types = vec![OperationType::MatrixVector, OperationType::ElementWise];

    InputCharacteristics {
        dimensions: (shape.dims()[0], shape.dims()[1]),
        sparsity,
        distribution_pattern,
        operation_types,
        memory_budget: None,
    }
}

/// Create a dummy dense matrix for benchmarking
fn create_dummy_dense_matrix(rows: usize, cols: usize) -> TorshResult<crate::Tensor> {
    use torsh_core::{DType, Device, Shape};

    // Create a simple dense matrix filled with ones
    let shape = Shape::new(vec![rows.min(100), cols.min(100)]); // Limit size for performance
    let device = torsh_core::DeviceType::Cpu;
    let dtype = DType::F32;

    // This is a simplified implementation - in practice, you'd use the actual
    // tensor creation APIs from torsh-tensor
    crate::Tensor::ones(shape.dims(), device)
}

/// Generate quick recommendations based on analysis
fn generate_quick_recommendations(
    memory_analysis: &MemoryAnalysis,
    performance_summary: &PerformanceSummary,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // Memory-based recommendations
    if memory_analysis.compression_ratio < 2.0 {
        recommendations
            .push("Consider using a denser format or different sparsity threshold".to_string());
    } else if memory_analysis.compression_ratio > 10.0 {
        recommendations
            .push("Excellent compression achieved - current format is optimal".to_string());
    }

    // Performance-based recommendations
    match performance_summary.performance_grade().as_str() {
        "A" => recommendations.push("Excellent performance - no optimization needed".to_string()),
        "B" => recommendations.push("Good performance - minor optimizations possible".to_string()),
        "C" => recommendations.push("Fair performance - consider format optimization".to_string()),
        "D" | "F" => {
            recommendations.push("Poor performance - significant optimization needed".to_string())
        }
        _ => {}
    }

    // Consistency recommendations
    if performance_summary.consistency_score < 0.7 {
        recommendations
            .push("Performance inconsistency detected - investigate system load".to_string());
    }

    if recommendations.is_empty() {
        recommendations.push("Performance appears optimal".to_string());
    }

    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CooTensor, SparseFormat};
    use torsh_core::{DType, Device, Shape};

    fn create_test_sparse_tensor() -> CooTensor {
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(vec![3, 3]);
        CooTensor::new(row_indices, col_indices, values, shape).unwrap()
    }

    #[test]
    fn test_default_profiler() {
        let profiler = default_profiler();
        assert_eq!(profiler.config.warmup_iterations, 3);
        assert_eq!(profiler.config.measured_iterations, 10);
    }

    #[test]
    fn test_fast_profiler() {
        let profiler = fast_profiler();
        assert_eq!(profiler.config.warmup_iterations, 1);
        assert_eq!(profiler.config.measured_iterations, 3);
        assert!(!profiler.config.collect_memory);
    }

    #[test]
    fn test_thorough_profiler() {
        let profiler = thorough_profiler();
        assert_eq!(profiler.config.warmup_iterations, 5);
        assert_eq!(profiler.config.measured_iterations, 20);
        assert!(profiler.config.collect_memory);
    }

    #[test]
    fn test_memory_profiler() {
        let profiler = memory_profiler();
        assert!(profiler.config.collect_memory);
        assert!(profiler.config.gc_between_iterations);
    }

    #[test]
    fn test_quick_performance_check() {
        let tensor = create_test_sparse_tensor();
        let result = quick_performance_check(&tensor);

        assert!(result.is_ok());
        let quick_result = result.unwrap();

        assert_eq!(quick_result.memory_analysis.format, SparseFormat::Coo);
        assert_eq!(quick_result.memory_analysis.nnz, 3);
        assert!(!quick_result.recommendations.is_empty());
    }

    #[test]
    fn test_quick_performance_result_display() {
        let tensor = create_test_sparse_tensor();
        let result = quick_performance_check(&tensor).unwrap();

        let display_string = format!("{}", result);
        assert!(display_string.contains("Quick Performance Check"));
        assert!(display_string.contains("Format:"));
        assert!(display_string.contains("Compression Ratio:"));
        assert!(display_string.contains("Recommendations:"));
    }

    #[test]
    fn test_infer_characteristics() {
        let tensor = create_test_sparse_tensor();
        let characteristics = infer_characteristics(&tensor);

        assert_eq!(characteristics.dimensions, (3, 3));
        assert!(characteristics.sparsity > 0.5); // 3 non-zeros out of 9 = 66.7% sparse
        assert!(characteristics
            .operation_types
            .contains(&OperationType::MatrixVector));
    }

    #[test]
    fn test_generate_quick_recommendations() {
        let memory_analysis = MemoryAnalysis::new(SparseFormat::Coo, 100, (1000, 1000));
        let performance_summary = PerformanceSummary {
            total_operations: 10,
            total_time: std::time::Duration::from_millis(100),
            avg_throughput: 100.0,
            avg_memory_efficiency: 0.8,
            consistency_score: 0.9,
        };

        let recommendations =
            generate_quick_recommendations(&memory_analysis, &performance_summary);
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_comprehensive_analysis_report() {
        let report = ComprehensiveAnalysisReport {
            performance_report: PerformanceReport::new(),
            memory_analyses: vec![MemoryAnalysis::new(SparseFormat::Coo, 100, (1000, 1000))],
            system_capabilities: SystemCapabilityReport {
                system_info: SystemInfo::detect(),
                capability_scores: HashMap::new(),
                recommendations: vec!["Test recommendation".to_string()],
                benchmark_timestamp: std::time::SystemTime::now(),
            },
            auto_tuning_recommendations: vec!["Auto-tune rec".to_string()],
            hardware_recommendations: vec!["Hardware rec".to_string()],
            performance_recommendations: vec!["Performance rec".to_string()],
            export_path: Some("/tmp/test".to_string()),
        };

        let all_recs = report.all_recommendations();
        assert_eq!(all_recs.len(), 3);

        let summary = report.summary();
        assert!(summary.contains("Performance Analysis Summary"));
        assert!(summary.contains("Total measurements: 0"));
        assert!(summary.contains("Export path: /tmp/test"));
    }

    #[test]
    fn test_module_re_exports() {
        // Test that all major types are accessible through the unified interface
        let _config = BenchmarkConfig::default();
        let _profiler = SparseProfiler::new(_config);
        let _tuner = AutoTuner::new();
        let _analyzer = TrendAnalyzer::new();
        let _exporter = TensorBoardExporter::new();

        // Test enum variants
        let _strategy = OptimizationStrategy::Speed;
        let _pattern = DistributionPattern::Random;
        let _operation = OperationType::MatrixVector;
        let _trend = TrendDirection::Improving;
    }

    #[test]
    fn test_convenience_functions() {
        // Test all convenience profiler creation functions
        let _default = default_profiler();
        let _fast = fast_profiler();
        let _thorough = thorough_profiler();
        let _memory = memory_profiler();

        // Verify they have different configurations
        assert_ne!(
            _fast.config.measured_iterations,
            _thorough.config.measured_iterations
        );
        assert_ne!(_fast.config.collect_memory, _memory.config.collect_memory);
    }
}
