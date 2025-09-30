//! # Auto-tuning and Hardware Optimization Module
//!
//! This module provides advanced performance optimization capabilities including
//! auto-tuning algorithms, hardware-specific benchmarking, and intelligent
//! recommendations for optimal sparse tensor operations.
//!
//! ## Key Components
//!
//! - **AutoTuner**: Automatic performance optimization and format selection
//! - **HardwareBenchmark**: Hardware-specific performance profiling
//! - **SystemInfo**: System capability detection and analysis
//! - **Optimization strategies**: Various algorithms for performance optimization
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_sparse::performance_tools::optimization::{AutoTuner, HardwareBenchmark};
//!
//! // Auto-tune sparse format selection
//! let tuner = AutoTuner::new();
//! let optimal_format = tuner.find_optimal_format(&matrix_data)?;
//!
//! // Benchmark hardware capabilities
//! let benchmark = HardwareBenchmark::new();
//! let system_info = benchmark.analyze_system_capabilities()?;
//! ```

pub mod auto_tuner;
pub mod hardware;
pub mod types;

// Re-export public API
pub use auto_tuner::AutoTuner;
pub use hardware::{
    CacheInfo, CpuInfo, HardwareBenchmark, MemoryInfo, SystemCapabilityReport, SystemInfo,
};
pub use types::{
    DistributionPattern, InputCharacteristics, OperationType, OptimizationStrategy, TuningResult,
};

// Tests that apply to the whole module
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_characteristics() -> InputCharacteristics {
        InputCharacteristics {
            dimensions: (1000, 1000),
            sparsity: 0.9,
            distribution_pattern: DistributionPattern::Random,
            operation_types: vec![OperationType::MatrixVector],
            memory_budget: Some(1024 * 1024), // 1MB
        }
    }

    #[test]
    fn test_auto_tuner_creation() {
        let tuner = AutoTuner::new();
        // Test that with_strategy works (can't access private field directly)
        let _tuner_with_strategy = tuner.with_strategy(OptimizationStrategy::Balanced);
    }

    #[test]
    fn test_auto_tuner_with_strategy() {
        let tuner = AutoTuner::new().with_strategy(OptimizationStrategy::Speed);
        // We can't access private fields, so just test that creation works
        let _ = tuner;
    }

    #[test]
    fn test_find_optimal_format() {
        let mut tuner = AutoTuner::new();
        let characteristics = create_test_characteristics();

        let result = tuner.find_optimal_format(&characteristics);
        assert!(result.is_ok());

        let tuning_result = result.unwrap();
        assert!(tuning_result.performance_score > 0.0);
        assert!(tuning_result.confidence >= 0.0 && tuning_result.confidence <= 1.0);
        assert!(!tuning_result.reasoning.is_empty());
    }

    #[test]
    fn test_get_recommendations() {
        let tuner = AutoTuner::new().with_strategy(OptimizationStrategy::Speed);
        let recommendations = tuner.get_recommendations();

        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.contains("speed") || r.contains("Speed")));
    }

    #[test]
    fn test_hardware_benchmark_creation() {
        let benchmark = HardwareBenchmark::new();
        // We can't access private fields, so just test that creation works
        let _ = benchmark;
    }

    #[test]
    fn test_system_info_detection() {
        let system_info = SystemInfo::detect();
        assert!(system_info.cpu_info.logical_cores > 0);
        assert!(system_info.cpu_info.physical_cores > 0);
        assert!(system_info.memory_info.total_memory > 0);
        assert!(!system_info.os_info.is_empty());
    }

    #[test]
    fn test_cpu_benchmark() {
        let mut benchmark = HardwareBenchmark::new();
        let score = benchmark.benchmark_cpu_compute();
        assert!(score.is_ok());
        let score_val = score.unwrap();
        assert!(score_val > 0.0);

        // Test caching
        let score2 = benchmark.benchmark_cpu_compute();
        assert!(score2.is_ok());
        assert_eq!(score_val, score2.unwrap());
    }

    #[test]
    fn test_memory_benchmark() {
        let mut benchmark = HardwareBenchmark::new();
        let score = benchmark.benchmark_memory_bandwidth();
        assert!(score.is_ok());
        assert!(score.unwrap() > 0.0);
    }

    #[test]
    fn test_cache_benchmark() {
        let mut benchmark = HardwareBenchmark::new();
        let score = benchmark.benchmark_cache_efficiency();
        assert!(score.is_ok());
        assert!(score.unwrap() > 0.0);
    }

    #[test]
    fn test_system_capability_analysis() {
        let mut benchmark = HardwareBenchmark::new();
        let report = benchmark.analyze_system_capabilities();

        assert!(report.is_ok());
        let report = report.unwrap();

        assert!(!report.capability_scores.is_empty());
        assert!(!report.recommendations.is_empty());
        assert!(report.capability_scores.contains_key("cpu_compute_score"));
        assert!(report
            .capability_scores
            .contains_key("memory_bandwidth_score"));
    }

    #[test]
    fn test_distribution_patterns() {
        use DistributionPattern::*;

        let random = Random;
        let block = Block {
            block_size: (10, 10),
        };
        let banded = Banded { bandwidth: 5 };
        let diagonal = Diagonal;

        assert_eq!(random, Random);
        assert_eq!(
            block,
            Block {
                block_size: (10, 10)
            }
        );
        assert_eq!(banded, Banded { bandwidth: 5 });
        assert_eq!(diagonal, Diagonal);
    }

    #[test]
    fn test_operation_types() {
        use OperationType::*;

        let matrix_vector = MatrixVector;
        let matrix_matrix = MatrixMatrix;
        let transpose = Transpose;

        assert_eq!(matrix_vector, MatrixVector);
        assert_eq!(matrix_matrix, MatrixMatrix);
        assert_eq!(transpose, Transpose);
    }

    #[test]
    fn test_optimization_strategies() {
        let speed = OptimizationStrategy::Speed;
        let memory = OptimizationStrategy::Memory;
        let balanced = OptimizationStrategy::Balanced;
        let custom = OptimizationStrategy::Custom {
            speed_weight: 0.5,
            memory_weight: 0.3,
            cache_weight: 0.2,
        };

        assert!(matches!(speed, OptimizationStrategy::Speed));
        assert!(matches!(memory, OptimizationStrategy::Memory));
        assert!(matches!(balanced, OptimizationStrategy::Balanced));
        assert!(matches!(custom, OptimizationStrategy::Custom { .. }));
    }
}
