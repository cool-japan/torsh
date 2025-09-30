//! Performance profiling and benchmarking framework
//!
//! This module provides comprehensive tools for profiling functional operations,
//! memory usage analysis, performance benchmarking, and regression testing.

use std::sync::{Mutex, OnceLock};

// Module declarations
pub mod benchmarking;
pub mod core;
pub mod regression;

// Re-exports for public API
pub use benchmarking::{benchmark, profile_operation, BenchmarkConfig, BenchmarkResults};
pub use core::{OperationMetrics, OperationSummary, Profiler};
pub use regression::{
    run_performance_regression_test, BaselineSummary, PerformanceBaseline,
    PerformanceRegressionTester, RegressionTestConfig, RegressionTestResult, SystemInfo,
};

/// Global profiler instance for convenience
static GLOBAL_PROFILER: OnceLock<Mutex<Profiler>> = OnceLock::new();

/// Get the global profiler instance
pub fn global_profiler() -> &'static Mutex<Profiler> {
    GLOBAL_PROFILER.get_or_init(|| Mutex::new(Profiler::new()))
}

/// Macro for easy profiling of operations
#[macro_export]
macro_rules! profile {
    ($name:expr, $inputs:expr, $operation:expr) => {{
        let mut profiler = $crate::profiling::global_profiler().lock().unwrap();
        profiler.start_operation($name, $inputs)?;
        let result = $operation;
        let output_refs: Vec<_> = [&result].iter().map(|t| *t).collect();
        profiler.finish_operation(&output_refs)?;
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::Result as TorshResult;
    use torsh_tensor::creation::{ones, zeros};

    #[test]
    fn test_profiler_basic() -> TorshResult<()> {
        let mut profiler = Profiler::new();

        let input = zeros(&[10, 10])?;
        let inputs = vec![&input];

        profiler.start_operation("test_op", &inputs)?;
        let output = ones(&[10, 10])?;
        let outputs = vec![&output];
        profiler.finish_operation(&outputs)?;

        assert_eq!(profiler.metrics.len(), 1);
        assert_eq!(profiler.metrics[0].name, "test_op");

        Ok(())
    }

    #[test]
    fn test_benchmark_function() -> TorshResult<()> {
        let input = zeros(&[100, 100])?;
        let inputs = vec![&input];

        let config = BenchmarkConfig {
            warmup_iters: 2,
            bench_iters: 5,
            min_duration: 0.1,
            max_duration: 1.0,
            detailed_metrics: false,
        };

        let results = benchmark(
            "zeros_benchmark",
            |_inputs| -> TorshResult<Vec<torsh_tensor::Tensor>> { Ok(vec![zeros(&[100, 100])?]) },
            &inputs,
            config,
        )?;

        assert_eq!(results.operation_name, "zeros_benchmark");
        assert!(!results.metrics.is_empty());
        assert!(results.summary.mean_duration > 0.0);

        Ok(())
    }

    #[test]
    fn test_flops_estimation() {
        use super::core::estimate_flops;
        let input_shapes = vec![vec![128, 256], vec![256, 512]];
        let output_shapes = vec![vec![128, 512]];

        let flops = estimate_flops("matmul", &input_shapes, &output_shapes);
        let expected_flops = 2 * 128 * 256 * 512; // 2 * M * K * N
        assert_eq!(flops, expected_flops as u64);
    }

    #[test]
    fn test_global_profiler() {
        let profiler_ref = global_profiler();
        let _profiler = profiler_ref.lock().unwrap();
        // Just test that we can get the global profiler without panicking
    }
}
