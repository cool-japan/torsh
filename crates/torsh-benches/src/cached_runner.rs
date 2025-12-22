//! Cached benchmark runner utilities
//!
//! This module provides utilities for running benchmarks with automatic caching,
//! integrating the BenchmarkCache with the existing benchmark infrastructure.

use crate::benchmark_cache::BenchmarkCache;
use crate::{BenchResult, Benchmarkable};
use std::time::{Duration, Instant};

/// Cached benchmark runner
///
/// Provides automatic caching for benchmark results, avoiding expensive re-runs
/// when code hasn't changed.
pub struct CachedBenchRunner {
    cache: BenchmarkCache,
    default_iterations: u32,
    default_warmup: u32,
}

impl CachedBenchRunner {
    /// Create a new cached benchmark runner
    pub fn new(cache_dir: impl AsRef<std::path::Path>) -> Self {
        Self {
            cache: BenchmarkCache::new(cache_dir),
            default_iterations: 10,
            default_warmup: 3,
        }
    }

    /// Create a cached runner with custom TTL
    pub fn with_ttl(cache_dir: impl AsRef<std::path::Path>, ttl: Duration) -> Self {
        Self {
            cache: BenchmarkCache::with_ttl(cache_dir, ttl),
            default_iterations: 10,
            default_warmup: 3,
        }
    }

    /// Set default iterations
    pub fn set_iterations(&mut self, iterations: u32) -> &mut Self {
        self.default_iterations = iterations;
        self
    }

    /// Set default warmup iterations
    pub fn set_warmup(&mut self, warmup: u32) -> &mut Self {
        self.default_warmup = warmup;
        self
    }

    /// Disable git validation (useful for CI)
    pub fn without_git_validation(mut self) -> Self {
        self.cache = self.cache.without_git_validation();
        self
    }

    /// Run a benchmark with caching
    ///
    /// Returns the cached result if available and valid, otherwise runs the benchmark
    /// and caches the result.
    pub fn run<B: Benchmarkable>(
        &mut self,
        benchmark_name: &str,
        size: usize,
        benchmark: B,
    ) -> BenchResult
    where
        B::Output: Default,
    {
        self.run_with_config(
            benchmark_name,
            size,
            benchmark,
            self.default_iterations,
            self.default_warmup,
        )
    }

    /// Run a benchmark with custom iteration counts
    pub fn run_with_config<B: Benchmarkable>(
        &mut self,
        benchmark_name: &str,
        size: usize,
        mut benchmark: B,
        iterations: u32,
        warmup: u32,
    ) -> BenchResult
    where
        B::Output: Default,
    {
        // Check cache first
        if let Some(cached) = self.cache.get(benchmark_name, size) {
            return cached;
        }

        // Run benchmark
        let input = benchmark.setup(size);

        // Warmup
        for _ in 0..warmup {
            let _ = benchmark.run(&input);
        }

        // Measure
        let mut total_time = 0.0;
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = benchmark.run(&input);
            total_time += start.elapsed().as_nanos() as f64;
        }

        let avg_time = total_time / iterations as f64;
        let throughput = if benchmark.flops(size) > 0 {
            Some(benchmark.flops(size) as f64 / avg_time * 1e9)
        } else {
            None
        };

        let result = BenchResult {
            name: format!("{}_{}", benchmark_name, size),
            size,
            dtype: torsh_core::dtype::DType::F32, // Default to F32
            mean_time_ns: avg_time,
            std_dev_ns: 0.0, // Single measurement, no std dev
            throughput,
            memory_usage: None,
            peak_memory: None,
            metrics: std::collections::HashMap::new(),
        };

        // Cache the result
        let _ = self
            .cache
            .put(benchmark_name, size, result.clone(), iterations, warmup);

        result
    }

    /// Run benchmarks across multiple sizes with caching
    pub fn run_sizes<B: Benchmarkable>(
        &mut self,
        benchmark_name: &str,
        sizes: &[usize],
        mut create_benchmark: impl FnMut() -> B,
    ) -> Vec<BenchResult>
    where
        B::Output: Default,
    {
        sizes
            .iter()
            .map(|&size| {
                let bench = create_benchmark();
                self.run(benchmark_name, size, bench)
            })
            .collect()
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> crate::benchmark_cache::CacheStats {
        self.cache.stats()
    }

    /// Prune invalid cache entries
    pub fn prune_cache(&mut self) -> std::io::Result<usize> {
        self.cache.prune()
    }

    /// Clear all cached results
    pub fn clear_cache(&mut self) -> std::io::Result<()> {
        self.cache.clear()
    }
}

/// Batch benchmark runner with caching
///
/// Runs multiple benchmarks across multiple sizes with intelligent caching
pub struct BatchCachedRunner {
    runner: CachedBenchRunner,
    results: Vec<(String, Vec<BenchResult>)>,
}

impl BatchCachedRunner {
    /// Create a new batch runner
    pub fn new(cache_dir: impl AsRef<std::path::Path>) -> Self {
        Self {
            runner: CachedBenchRunner::new(cache_dir),
            results: Vec::new(),
        }
    }

    /// Add a benchmark to the batch
    pub fn add_benchmark<B: Benchmarkable>(
        &mut self,
        name: &str,
        sizes: &[usize],
        create_benchmark: impl FnMut() -> B,
    ) where
        B::Output: Default,
    {
        let results = self.runner.run_sizes(name, sizes, create_benchmark);
        self.results.push((name.to_string(), results));
    }

    /// Run all benchmarks
    pub fn run_all(&mut self) {
        // Benchmarks are run when added, so this is a no-op
        // This method exists for API consistency
    }

    /// Get all results
    pub fn results(&self) -> &[(String, Vec<BenchResult>)] {
        &self.results
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut report = String::from("# Benchmark Summary\n\n");

        for (name, results) in &self.results {
            report.push_str(&format!("## {}\n\n", name));
            report.push_str("| Size | Time | Throughput |\n");
            report.push_str("|------|------|------------|\n");

            for result in results {
                let throughput_str = if let Some(tp) = result.throughput {
                    format!("{:.2} GFLOPS", tp / 1e9)
                } else {
                    "N/A".to_string()
                };

                report.push_str(&format!(
                    "| {} | {:.2} ms | {} |\n",
                    result.name.split('_').last().unwrap_or("?"),
                    result.mean_time_ns / 1_000_000.0,
                    throughput_str
                ));
            }
            report.push('\n');
        }

        // Add cache statistics
        let stats = self.runner.cache_stats();
        report.push_str("## Cache Statistics\n\n");
        report.push_str(&format!("- Total entries: {}\n", stats.total_entries));
        report.push_str(&format!("- Valid entries: {}\n", stats.valid_entries));
        report.push_str(&format!("- Hit rate: {:.1}%\n", stats.hit_rate() * 100.0));

        report
    }

    /// Save summary to file
    pub fn save_summary(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock benchmark for testing
    struct MockBench;

    impl Benchmarkable for MockBench {
        type Input = usize;
        type Output = usize;

        fn setup(&mut self, size: usize) -> Self::Input {
            size
        }

        fn run(&mut self, input: &Self::Input) -> Self::Output {
            // Simulate some work
            std::thread::sleep(std::time::Duration::from_micros(10));
            *input
        }

        fn flops(&self, size: usize) -> usize {
            size * 100
        }
    }

    #[test]
    fn test_cached_runner_creation() {
        let temp_dir = std::env::temp_dir().join("torsh_cached_runner_test");
        let runner = CachedBenchRunner::new(&temp_dir);
        assert_eq!(runner.default_iterations, 10);
        assert_eq!(runner.default_warmup, 3);
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cached_runner_config() {
        let temp_dir = std::env::temp_dir().join("torsh_cached_runner_config");
        let mut runner = CachedBenchRunner::new(&temp_dir);
        runner.set_iterations(20).set_warmup(5);
        assert_eq!(runner.default_iterations, 20);
        assert_eq!(runner.default_warmup, 5);
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cached_runner_run() {
        let temp_dir = std::env::temp_dir().join("torsh_cached_runner_run");
        let mut runner = CachedBenchRunner::new(&temp_dir).without_git_validation();
        runner.set_iterations(2).set_warmup(1);

        let result = runner.run("mock_bench", 100, MockBench);
        assert!(result.mean_time_ns > 0.0);
        assert!(result.throughput.is_some());

        // Second run should be cached (faster)
        let cached_result = runner.run("mock_bench", 100, MockBench);
        assert_eq!(cached_result.mean_time_ns, result.mean_time_ns);

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_batch_runner() {
        let temp_dir = std::env::temp_dir().join("torsh_batch_runner_test");
        let mut batch = BatchCachedRunner::new(&temp_dir);

        batch.add_benchmark("test1", &[10, 20], || MockBench);
        batch.add_benchmark("test2", &[30, 40], || MockBench);

        let results = batch.results();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1.len(), 2);
        assert_eq!(results[1].1.len(), 2);

        let summary = batch.summary();
        assert!(summary.contains("Benchmark Summary"));
        assert!(summary.contains("test1"));
        assert!(summary.contains("test2"));

        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
