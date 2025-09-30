//! Benchmarking framework for functional operations
//!
//! This module provides comprehensive benchmarking capabilities with
//! warmup iterations, statistical analysis, and detailed metrics collection.

use super::core::{OperationMetrics, OperationSummary, Profiler};
use std::time::Instant;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iters: usize,
    /// Number of benchmark iterations
    pub bench_iters: usize,
    /// Minimum benchmark duration in seconds
    pub min_duration: f64,
    /// Maximum benchmark duration in seconds
    pub max_duration: f64,
    /// Whether to collect detailed metrics
    pub detailed_metrics: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iters: 5,
            bench_iters: 100,
            min_duration: 1.0,
            max_duration: 60.0,
            detailed_metrics: true,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub operation_name: String,
    pub config: BenchmarkConfig,
    pub metrics: Vec<OperationMetrics>,
    pub summary: OperationSummary,
}

/// Benchmark a function with given inputs
pub fn benchmark<F, R>(
    name: &str,
    mut operation: F,
    inputs: &[&Tensor],
    config: BenchmarkConfig,
) -> TorshResult<BenchmarkResults>
where
    F: FnMut(&[&Tensor]) -> TorshResult<R>,
    R: AsRef<[Tensor]>,
{
    let mut profiler = Profiler::new();
    if config.detailed_metrics {
        profiler.enable_memory_tracking();
        profiler.enable_flops_counting();
    }

    // Warmup iterations
    for _ in 0..config.warmup_iters {
        let _ = operation(inputs)?;
    }

    // Benchmark iterations
    let start_time = Instant::now();
    let mut iteration = 0;

    while iteration < config.bench_iters {
        let elapsed = start_time.elapsed().as_secs_f64();
        if elapsed > config.max_duration {
            break;
        }
        if iteration > 0 && elapsed > config.min_duration {
            break;
        }

        profiler.start_operation(name, inputs)?;
        let result = operation(inputs)?;
        let output_refs: Vec<&Tensor> = result.as_ref().iter().collect();
        profiler.finish_operation(&output_refs)?;

        iteration += 1;
    }

    let summary = profiler
        .get_summary(name)
        .ok_or_else(|| TorshError::Other("Failed to generate benchmark summary".to_string()))?;

    Ok(BenchmarkResults {
        operation_name: name.to_string(),
        config,
        metrics: profiler.metrics,
        summary,
    })
}

/// Profile a single operation
pub fn profile_operation<F, R>(
    name: &str,
    mut operation: F,
    inputs: &[&Tensor],
) -> TorshResult<OperationMetrics>
where
    F: FnMut(&[&Tensor]) -> TorshResult<R>,
    R: AsRef<[Tensor]>,
{
    let mut profiler = Profiler::new();
    profiler.enable_memory_tracking();
    profiler.enable_flops_counting();

    profiler.start_operation(name, inputs)?;
    let result = operation(inputs)?;
    let output_refs: Vec<&Tensor> = result.as_ref().iter().collect();
    profiler.finish_operation(&output_refs)?;

    Ok(profiler.metrics.into_iter().next().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::randn;

    #[test]
    fn test_benchmark_basic() -> TorshResult<()> {
        let input = randn(&[128, 128])?;
        let inputs = vec![&input];

        let config = BenchmarkConfig {
            warmup_iters: 1,
            bench_iters: 3,
            min_duration: 0.1,
            max_duration: 1.0,
            detailed_metrics: false,
        };

        let results = benchmark(
            "test_operation",
            |inputs| -> TorshResult<Vec<Tensor>> { Ok(vec![inputs[0].clone()]) },
            &inputs,
            config,
        )?;

        assert_eq!(results.operation_name, "test_operation");
        assert!(results.metrics.len() <= 3);
        Ok(())
    }

    #[test]
    fn test_profile_operation() -> TorshResult<()> {
        let input = randn(&[64, 64])?;
        let inputs = vec![&input];

        let metrics = profile_operation(
            "test_profile",
            |inputs| -> TorshResult<Vec<Tensor>> { Ok(vec![inputs[0].clone()]) },
            &inputs,
        )?;

        assert_eq!(metrics.name, "test_profile");
        assert!(!metrics.input_shapes.is_empty());
        assert!(!metrics.output_shapes.is_empty());
        Ok(())
    }
}
