//! Example demonstrating the benchmark caching system
//!
//! This example shows how to use the benchmark cache to avoid re-running
//! expensive benchmarks when the code hasn't changed.

use std::time::Duration;
use torsh_benches::benchmark_cache::BenchmarkCache;
use torsh_benches::{BenchResult, Benchmarkable};
use torsh_tensor::Tensor;

/// Simple matrix multiplication benchmark
struct MatmulBench;

impl Benchmarkable for MatmulBench {
    type Input = (Tensor<f32>, Tensor<f32>);
    type Output = Result<Tensor<f32>, torsh_core::error::TorshError>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let a = torsh_tensor::creation::rand::<f32>(&[size, size]).unwrap();
        let b = torsh_tensor::creation::rand::<f32>(&[size, size]).unwrap();
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        input.0.matmul(&input.1)
    }

    fn flops(&self, size: usize) -> usize {
        2 * size * size * size
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Benchmark Caching System Demo\n");

    // Create cache with 1-day TTL
    let cache_dir = std::env::temp_dir().join("torsh_bench_cache_demo");
    let mut cache = BenchmarkCache::with_ttl(&cache_dir, Duration::from_secs(24 * 60 * 60));

    println!("📁 Cache directory: {:?}\n", cache_dir);

    // Benchmark sizes to test
    let sizes = vec![64, 128, 256, 512];
    let benchmark_name = "matrix_multiplication";

    println!("Running benchmarks with caching...\n");

    for &size in &sizes {
        println!("🔍 Size: {}x{}", size, size);

        // Try to get cached result
        if let Some(cached_result) = cache.get(benchmark_name, size) {
            println!("  ✅ Using cached result:");
            println!(
                "     Time: {:.2} ms",
                cached_result.mean_time_ns / 1_000_000.0
            );
            if let Some(throughput) = cached_result.throughput {
                println!("     Throughput: {:.2} GFLOPS", throughput / 1e9);
            }
            println!();
            continue;
        }

        println!("  🔄 Cache miss - running benchmark...");

        // Run benchmark
        let mut bench = MatmulBench;
        let input = bench.setup(size);

        let warmup_iterations = 3;
        for _ in 0..warmup_iterations {
            let _ = bench.run(&input);
        }

        let iterations = 10;
        let mut total_time = 0.0;

        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = bench.run(&input);
            total_time += start.elapsed().as_nanos() as f64;
        }

        let avg_time = total_time / iterations as f64;
        let flops = bench.flops(size) as f64;
        let throughput = flops / avg_time * 1e9;

        let result = BenchResult {
            name: format!("{}_{}", benchmark_name, size),
            size,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: avg_time,
            std_dev_ns: 0.0,
            throughput: Some(throughput),
            memory_usage: None,
            peak_memory: None,
            metrics: std::collections::HashMap::new(),
        };

        // Cache the result
        cache.put(
            benchmark_name,
            size,
            result.clone(),
            iterations as u32,
            warmup_iterations as u32,
        )?;

        println!("  ✅ Benchmark completed:");
        println!("     Time: {:.2} ms", result.mean_time_ns / 1_000_000.0);
        println!("     Throughput: {:.2} GFLOPS", throughput / 1e9);
        println!("     Cached for future use");
        println!();
    }

    // Show cache statistics
    println!("\n📊 Cache Statistics:");
    let stats = cache.stats();
    println!("  Total entries: {}", stats.total_entries);
    println!("  Valid entries: {}", stats.valid_entries);
    println!("  Expired entries: {}", stats.expired_entries);
    println!("  Invalid system: {}", stats.invalid_system);
    println!("  Invalid git: {}", stats.invalid_git);
    println!("  Hit rate: {:.1}%", stats.hit_rate() * 100.0);

    println!("\n💡 Try running this example again to see cached results!");
    println!("   Modify the code and the cache will automatically invalidate.");

    // Demonstrate cache pruning
    println!("\n🧹 Pruning invalid cache entries...");
    let pruned = cache.prune()?;
    println!("  Removed {} invalid entries", pruned);

    // Cleanup example: clear the cache
    // Uncomment to clear cache between runs:
    // cache.clear()?;
    // println!("\n🗑️  Cache cleared");

    Ok(())
}
