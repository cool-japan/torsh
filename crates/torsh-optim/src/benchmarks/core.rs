//! Core benchmarking functionality and types
//!
//! This module provides the fundamental types and core benchmarking operations
//! for evaluating optimizer performance. It includes basic performance tests,
//! memory usage analysis, and convergence benchmarks.

use crate::{Optimizer, OptimizerResult};
use std::time::{Duration, Instant};
use torsh_core::device::DeviceType;
use torsh_tensor::{creation, Tensor};

/// Statistical analysis of benchmark results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct StatisticalAnalysis {
    /// Mean execution time
    pub mean_time: Duration,
    /// Median execution time
    pub median_time: Duration,
    /// Standard deviation of execution times
    pub std_dev: Duration,
    /// Confidence interval (lower, upper)
    pub confidence_interval: (Duration, Duration),
    /// Effect size for statistical significance
    pub effect_size: Option<f64>,
    /// P-value for statistical significance
    pub p_value: Option<f64>,
}

/// Benchmark configuration
///
/// Controls the behavior of benchmark runs including iteration counts,
/// time limits, and profiling options.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of benchmark iterations to run
    pub num_iterations: usize,
    /// Number of warmup iterations (excluded from timing)
    pub warmup_iterations: usize,
    /// Maximum time to run each benchmark (in seconds)
    pub max_time_seconds: f32,
    /// Device to run benchmarks on
    pub device: DeviceType,
    /// Whether to include memory profiling
    pub profile_memory: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_iterations: 1000,
            warmup_iterations: 100,
            max_time_seconds: 60.0,
            device: DeviceType::Cpu,
            profile_memory: false,
        }
    }
}

/// Results from a single benchmark run
///
/// Contains comprehensive timing, convergence, and memory statistics
/// from a benchmark execution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Number of iterations completed
    pub iterations_completed: usize,
    /// Total elapsed time
    pub total_time: Duration,
    /// Average time per iteration
    pub avg_time_per_iteration: Duration,
    /// Minimum time per iteration
    pub min_time_per_iteration: Duration,
    /// Maximum time per iteration
    pub max_time_per_iteration: Duration,
    /// Standard deviation of iteration times
    pub time_std_dev: Duration,
    /// Final convergence metric (if applicable)
    pub final_loss: Option<f32>,
    /// Memory usage statistics
    pub memory_stats: Option<MemoryStats>,
    /// Convergence rate (loss reduction per iteration)
    pub convergence_rate: Option<f32>,
}

/// Memory usage statistics during benchmark execution
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Memory usage at start
    pub initial_memory_bytes: usize,
    /// Memory usage at end
    pub final_memory_bytes: usize,
    /// Average memory usage
    pub avg_memory_bytes: usize,
}

/// Core optimizer benchmark suite
///
/// Provides fundamental benchmarking operations including step performance,
/// convergence analysis, memory scaling, and sparse gradient handling.
pub struct OptimizerBenchmarks {
    config: BenchmarkConfig,
}

impl OptimizerBenchmarks {
    /// Create a new benchmark suite with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create a new benchmark suite with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: BenchmarkConfig) {
        self.config = config;
    }

    /// Estimate memory usage for parameters and optimizer state
    fn estimate_memory_usage<O: Optimizer>(params: &[Tensor], optimizer: &O) -> usize {
        let mut total_bytes = 0;

        // Calculate memory for parameters
        for param in params {
            let shape = param.shape();
            let element_count = shape.dims().iter().product::<usize>();
            // Assume f32 elements (4 bytes each)
            total_bytes += element_count * 4;

            // Add memory for gradients if present
            if param.has_grad() {
                total_bytes += element_count * 4;
            }
        }

        // Estimate optimizer state memory
        // This is a rough approximation based on common optimizer patterns
        let state_multiplier = match optimizer.get_lr().len() {
            // Simple optimizers like SGD might have minimal state
            1 => 1.2,
            // More complex optimizers like Adam have momentum and squared gradients
            _ => 3.0,
        };

        let optimizer_state_bytes = (total_bytes as f64 * state_multiplier) as usize;
        total_bytes + optimizer_state_bytes
    }

    /// Benchmark optimizer step performance
    ///
    /// Measures the raw computational performance of the optimizer's step operation
    /// across multiple iterations to get reliable timing statistics.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Optimizer to benchmark
    /// * `problem_size` - Number of parameters in the optimization problem
    ///
    /// # Returns
    ///
    /// Benchmark results with timing statistics
    pub fn benchmark_step_performance<O: Optimizer>(
        &self,
        mut optimizer: O,
        problem_size: usize,
    ) -> OptimizerResult<BenchmarkResult> {
        let mut params = creation::randn::<f32>(&[problem_size])?;
        let mut iteration_times = Vec::new();

        // Warmup iterations to stabilize timing
        for _ in 0..self.config.warmup_iterations {
            let grads = creation::randn::<f32>(&[problem_size])?;
            params.set_grad(Some(grads));
            optimizer.step()?;
        }

        let start_time = Instant::now();
        let mut iterations_completed = 0;

        // Main benchmark loop
        for _i in 0..self.config.num_iterations {
            // Check time limit
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            let grads = creation::randn::<f32>(&[problem_size])?;
            params.set_grad(Some(grads));

            let iter_start = Instant::now();
            optimizer.step()?;
            let iter_time = iter_start.elapsed();

            iteration_times.push(iter_time);
            iterations_completed += 1;
        }

        let total_time = start_time.elapsed();

        // Calculate statistics
        let avg_time = total_time / iterations_completed as u32;
        let min_time = iteration_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_time = iteration_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        // Calculate standard deviation
        let mean_nanos = avg_time.as_nanos() as f64;
        let variance = iteration_times
            .iter()
            .map(|t| (t.as_nanos() as f64 - mean_nanos).powi(2))
            .sum::<f64>()
            / iterations_completed as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        Ok(BenchmarkResult {
            name: format!("step_performance_size_{}", problem_size),
            iterations_completed,
            total_time,
            avg_time_per_iteration: avg_time,
            min_time_per_iteration: min_time,
            max_time_per_iteration: max_time,
            time_std_dev: std_dev,
            final_loss: None,
            memory_stats: None,
            convergence_rate: None,
        })
    }

    /// Benchmark convergence on quadratic function
    ///
    /// Tests how quickly the optimizer converges on a simple quadratic optimization
    /// problem. This provides insight into convergence behavior and rate.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Optimizer to benchmark
    /// * `dimension` - Dimensionality of the optimization problem
    ///
    /// # Returns
    ///
    /// Benchmark results with convergence metrics
    pub fn benchmark_quadratic_convergence<O: Optimizer>(
        &self,
        mut optimizer: O,
        dimension: usize,
    ) -> OptimizerResult<BenchmarkResult> {
        let mut params = creation::randn::<f32>(&[dimension])?;
        let target = creation::zeros::<f32>(&[dimension])?;

        let mut losses = Vec::new();
        let mut iteration_times = Vec::new();

        // Initial loss: ||params - target||^2
        let initial_loss = params.sub(&target)?.pow(2.0)?.sum()?.item()?;
        losses.push(initial_loss);

        let start_time = Instant::now();
        let mut iterations_completed = 0;

        for _i in 0..self.config.num_iterations {
            // Check time limit
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            // Compute gradients for quadratic loss: grad = 2 * (params - target)
            let grads = params.sub(&target)?.mul_scalar(2.0)?;

            let iter_start = Instant::now();
            params.set_grad(Some(grads));
            optimizer.step()?;
            let iter_time = iter_start.elapsed();

            iteration_times.push(iter_time);

            // Compute loss
            let loss = params.sub(&target)?.pow(2.0)?.sum()?.item()?;
            losses.push(loss);

            iterations_completed += 1;

            // Early stopping if converged
            if loss < 1e-8 {
                break;
            }
        }

        let total_time = start_time.elapsed();
        let final_loss = losses.last().copied().unwrap_or(f32::INFINITY);

        // Calculate convergence rate (log reduction per iteration)
        let convergence_rate = if losses.len() > 1 {
            let log_reduction = (initial_loss.ln() - final_loss.ln()).max(0.0);
            Some(log_reduction / iterations_completed as f32)
        } else {
            None
        };

        // Calculate timing statistics
        let avg_time = total_time / iterations_completed as u32;
        let min_time = iteration_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_time = iteration_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        let mean_nanos = avg_time.as_nanos() as f64;
        let variance = iteration_times
            .iter()
            .map(|t| (t.as_nanos() as f64 - mean_nanos).powi(2))
            .sum::<f64>()
            / iterations_completed as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        Ok(BenchmarkResult {
            name: format!("quadratic_convergence_dim_{}", dimension),
            iterations_completed,
            total_time,
            avg_time_per_iteration: avg_time,
            min_time_per_iteration: min_time,
            max_time_per_iteration: max_time,
            time_std_dev: std_dev,
            final_loss: Some(final_loss),
            memory_stats: None,
            convergence_rate,
        })
    }

    /// Benchmark sparse gradient handling
    ///
    /// Tests optimizer performance when dealing with sparse gradients,
    /// which is common in many machine learning scenarios.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Optimizer to benchmark
    /// * `total_params` - Total number of parameters
    /// * `sparsity` - Fraction of gradients that are non-zero (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Benchmark results for sparse gradient performance
    pub fn benchmark_sparse_gradients<O: Optimizer>(
        &self,
        mut optimizer: O,
        total_params: usize,
        sparsity: f32,
    ) -> OptimizerResult<BenchmarkResult> {
        let mut params = creation::randn::<f32>(&[total_params])?;

        let mut iteration_times = Vec::new();

        // Warmup with sparse gradients
        for _ in 0..self.config.warmup_iterations {
            let mut grads = creation::zeros::<f32>(&[total_params])?;

            // Set sparse gradients
            for i in 0..total_params {
                if (i as f32 / total_params as f32) < sparsity {
                    let grad_val = ((i as f32 * 0.1) % 2.0) - 1.0;
                    grads.set(&[i], grad_val)?;
                }
            }

            params.set_grad(Some(grads));
            optimizer.step()?;
        }

        let start_time = Instant::now();
        let mut iterations_completed = 0;

        for _ in 0..self.config.num_iterations {
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            let mut grads = creation::zeros::<f32>(&[total_params])?;

            // Set sparse gradients
            for i in 0..total_params {
                if (i as f32 / total_params as f32) < sparsity {
                    let grad_val = ((i as f32 * 0.1) % 2.0) - 1.0;
                    grads.set(&[i], grad_val)?;
                }
            }

            let iter_start = Instant::now();
            params.set_grad(Some(grads));
            optimizer.step()?;
            let iter_time = iter_start.elapsed();

            iteration_times.push(iter_time);
            iterations_completed += 1;
        }

        let total_time = start_time.elapsed();
        let avg_time = total_time / iterations_completed as u32;
        let min_time = iteration_times
            .iter()
            .min()
            .copied()
            .unwrap_or(Duration::ZERO);
        let max_time = iteration_times
            .iter()
            .max()
            .copied()
            .unwrap_or(Duration::ZERO);

        let mean_nanos = avg_time.as_nanos() as f64;
        let variance = iteration_times
            .iter()
            .map(|t| (t.as_nanos() as f64 - mean_nanos).powi(2))
            .sum::<f64>()
            / iterations_completed as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        Ok(BenchmarkResult {
            name: format!(
                "sparse_gradients_params_{}_sparsity_{:.2}",
                total_params, sparsity
            ),
            iterations_completed,
            total_time,
            avg_time_per_iteration: avg_time,
            min_time_per_iteration: min_time,
            max_time_per_iteration: max_time,
            time_std_dev: std_dev,
            final_loss: None,
            memory_stats: None,
            convergence_rate: None,
        })
    }

    /// Run a comprehensive set of core benchmarks
    ///
    /// Executes multiple benchmark scenarios to provide a complete performance
    /// profile of the optimizer including step performance, convergence, and
    /// sparse gradient handling.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - Optimizer to benchmark (must be cloneable)
    ///
    /// # Returns
    ///
    /// Vector of benchmark results covering different scenarios
    pub fn run_comprehensive_benchmarks<O: Optimizer + Clone>(
        &self,
        optimizer: O,
    ) -> OptimizerResult<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        // Step performance benchmarks for different problem sizes
        for &size in &[100, 1000, 10000] {
            results.push(self.benchmark_step_performance(optimizer.clone(), size)?);
        }

        // Convergence benchmarks
        for &dim in &[10, 100, 1000] {
            results.push(self.benchmark_quadratic_convergence(optimizer.clone(), dim)?);
        }

        // Sparse gradient benchmarks
        for &sparsity in &[0.1, 0.01, 0.001] {
            results.push(self.benchmark_sparse_gradients(optimizer.clone(), 10000, sparsity)?);
        }

        Ok(results)
    }

    /// Print benchmark results in a formatted table
    ///
    /// Displays benchmark results in an easy-to-read tabular format
    /// with timing, convergence, and performance metrics.
    pub fn print_results(&self, results: &[BenchmarkResult]) {
        println!("\n{:=<100}", "");
        println!("{:^100}", "OPTIMIZER BENCHMARK RESULTS");
        println!("{:=<100}", "");

        println!(
            "{:<40} {:>12} {:>12} {:>12} {:>12} {:>10}",
            "Benchmark", "Iterations", "Total Time", "Avg Time", "Min Time", "Max Time"
        );
        println!("{:-<100}", "");

        for result in results {
            println!(
                "{:<40} {:>12} {:>12.3?} {:>12.3?} {:>12.3?} {:>12.3?}",
                result.name,
                result.iterations_completed,
                result.total_time,
                result.avg_time_per_iteration,
                result.min_time_per_iteration,
                result.max_time_per_iteration
            );

            if let Some(loss) = result.final_loss {
                println!("{:<40} Final Loss: {:.6e}", "", loss);
            }

            if let Some(rate) = result.convergence_rate {
                println!("{:<40} Convergence Rate: {:.6e}", "", rate);
            }
        }

        println!("{:=<100}", "");
    }
}

impl Default for OptimizerBenchmarks {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.num_iterations, 1000);
        assert_eq!(config.warmup_iterations, 100);
        assert_eq!(config.max_time_seconds, 60.0);
    }

    #[test]
    fn test_benchmark_result_creation() {
        let result = BenchmarkResult {
            name: "test_benchmark".to_string(),
            iterations_completed: 100,
            total_time: Duration::from_secs(1),
            avg_time_per_iteration: Duration::from_millis(10),
            min_time_per_iteration: Duration::from_millis(5),
            max_time_per_iteration: Duration::from_millis(15),
            time_std_dev: Duration::from_millis(2),
            final_loss: Some(0.1),
            memory_stats: None,
            convergence_rate: Some(0.05),
        };

        assert_eq!(result.name, "test_benchmark");
        assert_eq!(result.iterations_completed, 100);
        assert_eq!(result.final_loss, Some(0.1));
    }

    #[test]
    fn test_memory_stats() {
        let stats = MemoryStats {
            peak_memory_bytes: 1000,
            initial_memory_bytes: 500,
            final_memory_bytes: 800,
            avg_memory_bytes: 750,
        };

        assert_eq!(stats.peak_memory_bytes, 1000);
        assert_eq!(stats.avg_memory_bytes, 750);
    }
}
