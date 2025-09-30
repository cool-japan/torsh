//! Multi-optimizer comparison functionality
//!
//! This module provides comprehensive comparison capabilities for evaluating
//! multiple optimizers side-by-side with statistical analysis.

use super::core::{BenchmarkConfig, BenchmarkResult, MemoryStats, StatisticalAnalysis};
use crate::{Optimizer, OptimizerResult};
use std::time::Duration;

/// Comparison results between multiple optimizers
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
pub struct ComparisonResult {
    /// Optimizer name
    pub optimizer_name: String,
    /// Individual benchmark results
    pub benchmark_results: Vec<BenchmarkResult>,
    /// Statistical analysis
    pub statistical_analysis: StatisticalAnalysis,
    /// Performance ranking (1 = best)
    pub performance_rank: usize,
    /// Convergence ranking (1 = best)
    pub convergence_rank: usize,
    /// Memory efficiency ranking (1 = best)
    pub memory_rank: usize,
}

/// Multi-optimizer comparison suite
pub struct OptimizerComparison<O: Optimizer + Clone> {
    optimizers: Vec<(String, Box<dyn Fn() -> OptimizerResult<O>>)>,
    config: BenchmarkConfig,
}

impl<O: Optimizer + Clone> OptimizerComparison<O> {
    /// Create a new optimizer comparison suite
    pub fn new() -> Self {
        Self {
            optimizers: Vec::new(),
            config: BenchmarkConfig::default(),
        }
    }

    /// Add an optimizer to the comparison
    pub fn add_optimizer<F>(mut self, name: &str, factory: F) -> Self
    where
        F: Fn() -> OptimizerResult<O> + 'static,
    {
        self.optimizers.push((name.to_string(), Box::new(factory)));
        self
    }

    /// Set custom benchmark configuration
    pub fn with_config(mut self, config: BenchmarkConfig) -> Self {
        self.config = config;
        self
    }

    /// Run comprehensive comparison between all optimizers
    pub fn run_comparison_suite(&self) -> OptimizerResult<Vec<ComparisonResult>> {
        use super::optimizer::OptimizerBenchmarks;

        let benchmarks = OptimizerBenchmarks::with_config(self.config.clone());
        let mut comparison_results = Vec::new();

        for (name, factory) in &self.optimizers {
            let optimizer = factory()?;
            let results = benchmarks.run_comprehensive_benchmarks(optimizer)?;

            // Calculate statistical analysis
            let execution_times: Vec<Duration> =
                results.iter().map(|r| r.avg_time_per_iteration).collect();

            let stats = Self::calculate_statistical_analysis(&execution_times);

            comparison_results.push(ComparisonResult {
                optimizer_name: name.clone(),
                benchmark_results: results,
                statistical_analysis: stats,
                performance_rank: 0, // Will be calculated after all results are collected
                convergence_rank: 0,
                memory_rank: 0,
            });
        }

        // Calculate rankings
        self.calculate_rankings(&mut comparison_results);

        Ok(comparison_results)
    }

    /// Calculate statistical analysis from timing data
    fn calculate_statistical_analysis(times: &[Duration]) -> StatisticalAnalysis {
        if times.is_empty() {
            return StatisticalAnalysis {
                mean_time: Duration::ZERO,
                median_time: Duration::ZERO,
                std_dev: Duration::ZERO,
                confidence_interval: (Duration::ZERO, Duration::ZERO),
                effect_size: None,
                p_value: None,
            };
        }

        let total_nanos: u64 = times.iter().map(|t| t.as_nanos() as u64).sum();
        let mean_nanos = total_nanos / times.len() as u64;
        let mean_time = Duration::from_nanos(mean_nanos);

        let mut sorted_times = times.to_vec();
        sorted_times.sort();
        let median_time = sorted_times[sorted_times.len() / 2];

        // Calculate standard deviation
        let variance: f64 = times
            .iter()
            .map(|t| {
                let diff = t.as_nanos() as f64 - mean_nanos as f64;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        // Calculate 95% confidence interval (assuming normal distribution)
        let margin_error = 1.96 * (std_dev.as_nanos() as f64) / (times.len() as f64).sqrt();
        let lower_bound = Duration::from_nanos((mean_nanos as f64 - margin_error) as u64);
        let upper_bound = Duration::from_nanos((mean_nanos as f64 + margin_error) as u64);

        StatisticalAnalysis {
            mean_time,
            median_time,
            std_dev,
            confidence_interval: (lower_bound, upper_bound),
            effect_size: None,
            p_value: None,
        }
    }

    /// Calculate performance rankings
    fn calculate_rankings(&self, results: &mut [ComparisonResult]) {
        // Performance ranking (based on mean execution time - lower is better)
        let mut perf_indices: Vec<usize> = (0..results.len()).collect();
        perf_indices.sort_by(|&a, &b| {
            results[a]
                .statistical_analysis
                .mean_time
                .cmp(&results[b].statistical_analysis.mean_time)
        });
        for (rank, &idx) in perf_indices.iter().enumerate() {
            results[idx].performance_rank = rank + 1;
        }

        // Convergence ranking (based on convergence rate - higher is better)
        let mut conv_indices: Vec<usize> = (0..results.len()).collect();
        conv_indices.sort_by(|&a, &b| {
            let conv_a = results[a]
                .benchmark_results
                .iter()
                .filter_map(|r| r.convergence_rate)
                .fold(0.0, |acc, x| acc + x);
            let conv_b = results[b]
                .benchmark_results
                .iter()
                .filter_map(|r| r.convergence_rate)
                .fold(0.0, |acc, x| acc + x);
            conv_b
                .partial_cmp(&conv_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (rank, &idx) in conv_indices.iter().enumerate() {
            results[idx].convergence_rank = rank + 1;
        }

        // Memory ranking (based on peak memory usage - lower is better)
        let mut mem_indices: Vec<usize> = (0..results.len()).collect();
        mem_indices.sort_by(|&a, &b| {
            let mem_a = results[a]
                .benchmark_results
                .iter()
                .filter_map(|r| r.memory_stats.as_ref())
                .map(|m| m.peak_memory_bytes)
                .fold(0, |acc, x| acc + x);
            let mem_b = results[b]
                .benchmark_results
                .iter()
                .filter_map(|r| r.memory_stats.as_ref())
                .map(|m| m.peak_memory_bytes)
                .fold(0, |acc, x| acc + x);
            mem_a.cmp(&mem_b)
        });
        for (rank, &idx) in mem_indices.iter().enumerate() {
            results[idx].memory_rank = rank + 1;
        }
    }

    /// Print detailed comparison table
    pub fn print_comparison_table(&self, results: &[ComparisonResult]) {
        println!("\n{:=<120}", "");
        println!("{:^120}", "OPTIMIZER COMPARISON RESULTS");
        println!("{:=<120}", "");

        println!(
            "{:<20} {:>12} {:>15} {:>15} {:>15} {:>12} {:>12} {:>12}",
            "Optimizer",
            "Perf Rank",
            "Mean Time",
            "Median Time",
            "Std Dev",
            "Conv Rank",
            "Mem Rank",
            "Total Score"
        );
        println!("{:-<120}", "");

        for result in results {
            let total_score =
                result.performance_rank + result.convergence_rank + result.memory_rank;
            println!(
                "{:<20} {:>12} {:>15.3?} {:>15.3?} {:>15.3?} {:>12} {:>12} {:>12}",
                result.optimizer_name,
                result.performance_rank,
                result.statistical_analysis.mean_time,
                result.statistical_analysis.median_time,
                result.statistical_analysis.std_dev,
                result.convergence_rank,
                result.memory_rank,
                total_score
            );
        }

        println!("{:-<120}", "");
        println!("Lower ranks are better. Total Score = Performance + Convergence + Memory ranks");
        println!("{:=<120}", "");
    }

    /// Export comparison results to JSON
    #[cfg(feature = "serde")]
    pub fn export_json(&self, results: &[ComparisonResult], filename: &str) -> OptimizerResult<()> {
        use std::fs::File;
        use std::io::Write;

        let json = serde_json::to_string_pretty(results)
            .map_err(|e| crate::OptimizerError::SerializationError(e.to_string()))?;

        let mut file = File::create(filename).map_err(|e| crate::OptimizerError::IoError(e))?;

        file.write_all(json.as_bytes())
            .map_err(|e| crate::OptimizerError::IoError(e))?;

        println!("Comparison results exported to {}", filename);
        Ok(())
    }
}

impl<O: Optimizer + Clone> Default for OptimizerComparison<O> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_comparison_result() {
        let result = ComparisonResult {
            optimizer_name: "TestOptimizer".to_string(),
            benchmark_results: Vec::new(),
            statistical_analysis: StatisticalAnalysis {
                mean_time: Duration::from_millis(10),
                median_time: Duration::from_millis(9),
                std_dev: Duration::from_millis(2),
                confidence_interval: (Duration::from_millis(8), Duration::from_millis(12)),
                effect_size: Some(0.5),
                p_value: Some(0.05),
            },
            performance_rank: 1,
            convergence_rank: 2,
            memory_rank: 1,
        };

        assert_eq!(result.optimizer_name, "TestOptimizer");
        assert_eq!(result.performance_rank, 1);
        assert_eq!(result.convergence_rank, 2);
        assert_eq!(result.memory_rank, 1);
    }

    #[test]
    fn test_statistical_analysis_empty() {
        let stats = OptimizerComparison::<crate::sgd::SGD>::calculate_statistical_analysis(&[]);
        assert_eq!(stats.mean_time, Duration::ZERO);
        assert_eq!(stats.median_time, Duration::ZERO);
        assert_eq!(stats.std_dev, Duration::ZERO);
    }

    #[test]
    fn test_statistical_analysis_single_value() {
        let times = vec![Duration::from_millis(10)];
        let stats = OptimizerComparison::<crate::sgd::SGD>::calculate_statistical_analysis(&times);
        assert_eq!(stats.mean_time, Duration::from_millis(10));
        assert_eq!(stats.median_time, Duration::from_millis(10));
        assert_eq!(stats.std_dev, Duration::ZERO);
    }
}
