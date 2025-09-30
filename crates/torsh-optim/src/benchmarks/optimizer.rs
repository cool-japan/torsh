//! Core optimizer benchmarking implementation
//!
//! This module contains the main OptimizerBenchmarks struct and its implementation
//! for comprehensive optimizer performance evaluation.

use super::core::{BenchmarkConfig, BenchmarkResult, MemoryStats};
use crate::{Optimizer, OptimizerResult};
use std::time::{Duration, Instant};
use torsh_core::device::DeviceType;
use torsh_tensor::{creation, Tensor};

/// Optimizer benchmark suite
pub struct OptimizerBenchmarks {
    config: BenchmarkConfig,
}

impl OptimizerBenchmarks {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create a new benchmark suite with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
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

        // Base memory for optimizer structure itself (at least 1KB)
        let base_optimizer_memory = 1024;
        let optimizer_state_bytes = if total_bytes > 0 {
            (total_bytes as f64 * state_multiplier) as usize
        } else {
            base_optimizer_memory
        };

        total_bytes + optimizer_state_bytes
    }

    /// Benchmark optimizer step performance
    pub fn benchmark_step_performance<O: Optimizer>(
        &self,
        mut optimizer: O,
        problem_size: usize,
    ) -> OptimizerResult<BenchmarkResult> {
        let mut params = creation::randn::<f32>(&[problem_size])?;
        let mut iteration_times = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let grads = creation::randn::<f32>(&[problem_size])?;
            params.set_grad(Some(grads));
            optimizer.step()?;
        }

        let start_time = Instant::now();
        let mut iterations_completed = 0;

        // Benchmark loop
        for i in 0..self.config.num_iterations {
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
    pub fn benchmark_quadratic_convergence<O: Optimizer>(
        &self,
        mut optimizer: O,
        dimension: usize,
    ) -> OptimizerResult<BenchmarkResult> {
        let device = self.config.device;
        let mut params = creation::randn::<f32>(&[dimension])?;
        let target = Tensor::zeros(&[dimension], DeviceType::Cpu)?;

        let mut losses = Vec::new();
        let mut iteration_times = Vec::new();

        // Initial loss
        let initial_loss = params.sub(&target)?.pow(2.0)?.sum()?.item()?;
        losses.push(initial_loss);

        let start_time = Instant::now();
        let mut iterations_completed = 0;

        for i in 0..self.config.num_iterations {
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

        // Calculate convergence rate
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

    /// Benchmark memory usage scaling
    pub fn benchmark_memory_scaling<O: Optimizer + Clone>(
        &self,
        optimizer_factory: impl Fn(Vec<Tensor>) -> OptimizerResult<O>,
    ) -> OptimizerResult<Vec<BenchmarkResult>> {
        let device = self.config.device;
        let sizes = vec![100, 1000, 10000, 100000];
        let mut results = Vec::new();

        for &size in &sizes {
            let params = vec![creation::randn::<f32>(&[size])?];
            let mut optimizer = optimizer_factory(params.clone())?;

            // Warmup
            for _ in 0..10 {
                let grads = creation::randn::<f32>(&[size])?;
                params[0].set_grad(Some(grads));
                optimizer.step()?;
            }

            // Measure memory usage during optimization
            let start_time = Instant::now();
            let mut iteration_times = Vec::new();
            let iterations = 100.min(self.config.num_iterations);

            // Track memory usage if enabled
            let memory_stats = if self.config.profile_memory {
                let initial_memory = Self::estimate_memory_usage(&params, &optimizer);
                let mut memory_samples = Vec::new();
                let mut peak_memory = initial_memory;

                memory_samples.push(initial_memory);

                for _ in 0..iterations {
                    let grads = creation::randn::<f32>(&[size])?;

                    let iter_start = Instant::now();
                    params[0].set_grad(Some(grads));
                    optimizer.step()?;
                    let iter_time = iter_start.elapsed();

                    iteration_times.push(iter_time);

                    // Sample memory usage periodically (every 10 iterations)
                    if iteration_times.len() % 10 == 0 {
                        let current_memory = Self::estimate_memory_usage(&params, &optimizer);
                        memory_samples.push(current_memory);
                        peak_memory = peak_memory.max(current_memory);
                    }
                }

                let final_memory = Self::estimate_memory_usage(&params, &optimizer);
                memory_samples.push(final_memory);
                peak_memory = peak_memory.max(final_memory);

                let avg_memory = memory_samples.iter().sum::<usize>() / memory_samples.len().max(1);

                Some(MemoryStats {
                    peak_memory_bytes: peak_memory,
                    initial_memory_bytes: initial_memory,
                    final_memory_bytes: final_memory,
                    avg_memory_bytes: avg_memory,
                })
            } else {
                // If memory profiling is disabled, still collect timing data
                for _ in 0..iterations {
                    let grads = creation::randn::<f32>(&[size])?;

                    let iter_start = Instant::now();
                    params[0].set_grad(Some(grads));
                    optimizer.step()?;
                    let iter_time = iter_start.elapsed();

                    iteration_times.push(iter_time);
                }

                None
            };

            let total_time = start_time.elapsed();
            let avg_time = total_time / iterations as u32;
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
                / iterations as f64;
            let std_dev = Duration::from_nanos(variance.sqrt() as u64);

            results.push(BenchmarkResult {
                name: format!("memory_scaling_size_{}", size),
                iterations_completed: iterations,
                total_time,
                avg_time_per_iteration: avg_time,
                min_time_per_iteration: min_time,
                max_time_per_iteration: max_time,
                time_std_dev: std_dev,
                final_loss: None,
                memory_stats,
                convergence_rate: None,
            });
        }

        Ok(results)
    }

    /// Benchmark sparse gradient handling
    pub fn benchmark_sparse_gradients<O: Optimizer>(
        &self,
        mut optimizer: O,
        total_params: usize,
        sparsity: f32,
    ) -> OptimizerResult<BenchmarkResult> {
        let mut params = creation::randn::<f32>(&[total_params])?;

        let mut iteration_times = Vec::new();

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let mut grads = Tensor::zeros(&[total_params], DeviceType::Cpu)?;

            // Set sparse gradients
            for i in 0..total_params {
                if (i as f32 / total_params as f32) < sparsity {
                    let grad_val = ((i as f32 * 0.1) % 2.0) - 1.0;
                    let _ = grads.set(&[i], grad_val);
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

            let mut grads = Tensor::zeros(&[total_params], DeviceType::Cpu)?;

            // Set sparse gradients
            for i in 0..total_params {
                if (i as f32 / total_params as f32) < sparsity {
                    let grad_val = ((i as f32 * 0.1) % 2.0) - 1.0;
                    let _ = grads.set(&[i], grad_val);
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

    /// Run comprehensive benchmark suite
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

    /// Benchmark noisy quadratic convergence (more realistic optimization scenario)
    pub fn benchmark_noisy_quadratic_convergence<O: Optimizer>(
        &self,
        mut optimizer: O,
        dimension: usize,
        noise_level: f32,
    ) -> OptimizerResult<BenchmarkResult> {
        let mut params = creation::randn::<f32>(&[dimension])?;
        let target = Tensor::zeros(&[dimension], DeviceType::Cpu)?;

        let mut losses = Vec::new();
        let mut iteration_times = Vec::new();

        // Initial loss
        let initial_loss = params.sub(&target)?.pow(2.0)?.sum()?.item()?;
        losses.push(initial_loss);

        let start_time = Instant::now();
        let mut iterations_completed = 0;

        for _ in 0..self.config.num_iterations {
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            // Compute gradients for quadratic loss with noise: grad = 2 * (params - target) + noise
            let clean_grads = params.sub(&target)?.mul_scalar(2.0)?;
            let noise = creation::randn::<f32>(&[dimension])?.mul_scalar(noise_level)?;
            let grads = clean_grads.add(&noise)?;

            let iter_start = Instant::now();
            params.set_grad(Some(grads));
            optimizer.step()?;
            let iter_time = iter_start.elapsed();

            iteration_times.push(iter_time);

            // Compute loss
            let loss = params.sub(&target)?.pow(2.0)?.sum()?.item()?;
            losses.push(loss);

            iterations_completed += 1;

            // Early stopping if converged (more lenient due to noise)
            if loss < 1e-6 {
                break;
            }
        }

        let total_time = start_time.elapsed();
        let final_loss = losses.last().copied().unwrap_or(f32::INFINITY);

        // Calculate convergence rate
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
            name: format!("noisy_quadratic_dim_{}_noise_{:.3}", dimension, noise_level),
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

    /// Benchmark Rosenbrock function optimization (classic non-convex benchmark)
    pub fn benchmark_rosenbrock_optimization<O: Optimizer>(
        &self,
        mut optimizer: O,
        dimension: usize,
    ) -> OptimizerResult<BenchmarkResult> {
        let mut params = creation::randn::<f32>(&[dimension])?;

        let mut losses = Vec::new();
        let mut iteration_times = Vec::new();

        let start_time = Instant::now();
        let mut iterations_completed = 0;

        for _ in 0..self.config.num_iterations {
            if start_time.elapsed().as_secs_f32() > self.config.max_time_seconds {
                break;
            }

            // Compute Rosenbrock function gradients
            // f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2) for i = 0..n-2
            let mut grads = Tensor::zeros(&[dimension], DeviceType::Cpu)?;

            // This is a simplified approximation of Rosenbrock gradients
            // for the purpose of benchmarking optimizer performance
            for i in 0..dimension {
                let x_i = params.get(&[i])?;
                let grad_val = if i < dimension - 1 {
                    let x_next = params.get(&[i + 1])?;
                    // Approximate gradient: -400*x[i]*(x[i+1] - x[i]^2) - 2*(1 - x[i])
                    -400.0 * x_i * (x_next - x_i * x_i) - 2.0 * (1.0 - x_i)
                } else {
                    // Last element: 200*(x[i] - x[i-1]^2)
                    if i > 0 {
                        let x_prev = params.get(&[i - 1])?;
                        200.0 * (x_i - x_prev * x_prev)
                    } else {
                        0.0
                    }
                };
                grads.set(&[i], grad_val);
            }

            let iter_start = Instant::now();
            params.set_grad(Some(grads));
            optimizer.step()?;
            let iter_time = iter_start.elapsed();

            iteration_times.push(iter_time);

            // Compute approximate Rosenbrock loss
            let mut loss = 0.0;
            for i in 0..(dimension - 1) {
                let x_i = params.get(&[i])?;
                let x_next = params.get(&[i + 1])?;
                loss += 100.0 * (x_next - x_i * x_i).powf(2.0) + (1.0 - x_i).powf(2.0);
            }
            losses.push(loss);

            iterations_completed += 1;

            // Early stopping if converged
            if loss < 1e-4 {
                break;
            }
        }

        let total_time = start_time.elapsed();
        let final_loss = losses.last().copied().unwrap_or(f32::INFINITY);
        let initial_loss = losses.first().copied().unwrap_or(0.0);

        // Calculate convergence rate
        let convergence_rate = if losses.len() > 1 && initial_loss > 0.0 {
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
            name: format!("rosenbrock_optimization_dim_{}", dimension),
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
}

impl Default for OptimizerBenchmarks {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sgd::SGD;
    use parking_lot::RwLock;
    use std::sync::Arc;

    #[test]
    fn test_optimizer_benchmarks_creation() {
        let benchmarks = OptimizerBenchmarks::new();
        assert_eq!(benchmarks.config.num_iterations, 1000);
        assert_eq!(benchmarks.config.warmup_iterations, 100);
        assert_eq!(benchmarks.config.max_time_seconds, 60.0);
    }

    #[test]
    fn test_custom_config() {
        let custom_config = BenchmarkConfig {
            num_iterations: 500,
            warmup_iterations: 50,
            max_time_seconds: 30.0,
            ..Default::default()
        };

        let benchmarks = OptimizerBenchmarks::with_config(custom_config);
        assert_eq!(benchmarks.config.num_iterations, 500);
        assert_eq!(benchmarks.config.warmup_iterations, 50);
        assert_eq!(benchmarks.config.max_time_seconds, 30.0);
    }

    #[test]
    fn test_memory_estimation() {
        // This is a basic test - in practice we'd need actual tensors and optimizers
        // For now, just ensure the function doesn't panic
        let params: Vec<Tensor> = Vec::new();
        let sgd_params = vec![Arc::new(RwLock::new(
            creation::zeros::<f32>(&[10]).unwrap(),
        ))];
        let sgd = SGD::new(sgd_params, 0.01, None, None, None, false);

        let memory = OptimizerBenchmarks::estimate_memory_usage(&params, &sgd);
        assert!(memory > 0); // Should account for optimizer state even with no params
    }
}
