//! Real model benchmarking with actual tensor operations
//!
//! This module provides comprehensive benchmarking capabilities using real
//! torsh-tensor operations to measure actual model performance.

// Infrastructure module - functions designed for CLI command integration
#![allow(dead_code)]

use anyhow::Result;
use std::time::Instant;
use tracing::{debug, info};

// ✅ SciRS2 POLICY COMPLIANT: Use scirs2-core unified access patterns
use scirs2_core::random::{thread_rng, Distribution, Normal};

// ToRSh integration
use torsh::core::device::DeviceType;
use torsh::tensor::Tensor;

use super::tensor_integration::forward_pass;
use super::types::{TimingResult, TorshModel};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Batch size for inference
    pub batch_size: usize,
    /// Device to run benchmarks on
    pub device: DeviceType,
    /// Whether to measure memory usage
    pub measure_memory: bool,
    /// Whether to collect detailed timing statistics
    pub collect_detailed_stats: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            batch_size: 1,
            device: DeviceType::Cpu,
            measure_memory: true,
            collect_detailed_stats: true,
        }
    }
}

/// Detailed benchmark results
#[derive(Debug, Clone)]
pub struct DetailedBenchmarkResults {
    /// Individual iteration timings (ms)
    pub iteration_timings: Vec<f64>,
    /// Warmup phase duration (ms)
    pub warmup_duration: f64,
    /// Total measurement duration (ms)
    pub measurement_duration: f64,
    /// Memory usage samples (MB)
    pub memory_samples: Vec<f64>,
    /// Peak memory usage (MB)
    pub peak_memory: f64,
    /// Average memory usage (MB)
    pub avg_memory: f64,
    /// Throughput (samples/sec)
    pub throughput: f64,
    /// Latency statistics
    pub latency_stats: LatencyStatistics,
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStatistics {
    /// Mean latency (ms)
    pub mean: f64,
    /// Median latency (ms)
    pub median: f64,
    /// Standard deviation (ms)
    pub std_dev: f64,
    /// Minimum latency (ms)
    pub min: f64,
    /// Maximum latency (ms)
    pub max: f64,
    /// 95th percentile (ms)
    pub p95: f64,
    /// 99th percentile (ms)
    pub p99: f64,
}

/// Run comprehensive model benchmark with real tensor operations
pub fn benchmark_model_real(
    model: &TorshModel,
    config: &BenchmarkConfig,
) -> Result<DetailedBenchmarkResults> {
    info!(
        "Starting real model benchmark with {} warmup and {} measurement iterations",
        config.warmup_iterations, config.measurement_iterations
    );

    // Create input tensor for the model
    let input_shape = model
        .layers
        .first()
        .map(|l| l.input_shape.clone())
        .unwrap_or_else(|| vec![784]);

    let input = create_input_tensor(&input_shape, config.batch_size, config.device)?;

    // Warmup phase
    debug!("Running warmup phase...");
    let warmup_start = Instant::now();
    for i in 0..config.warmup_iterations {
        let _ = forward_pass(model, &input)?;

        if i % 10 == 0 {
            debug!("Warmup iteration {}/{}", i, config.warmup_iterations);
        }
    }
    let warmup_duration = warmup_start.elapsed().as_secs_f64() * 1000.0;

    debug!("Warmup completed in {:.2} ms", warmup_duration);

    // Measurement phase
    debug!("Running measurement phase...");
    let mut iteration_timings = Vec::with_capacity(config.measurement_iterations);
    let mut memory_samples = Vec::new();

    let measurement_start = Instant::now();

    for i in 0..config.measurement_iterations {
        // Measure single iteration
        let iter_start = Instant::now();
        let _ = forward_pass(model, &input)?;
        let iter_duration = iter_start.elapsed().as_secs_f64() * 1000.0;

        iteration_timings.push(iter_duration);

        // Measure memory if requested (every 10 iterations)
        if config.measure_memory && i % 10 == 0 {
            let memory_mb = estimate_memory_usage(model);
            memory_samples.push(memory_mb);
        }

        if i % 20 == 0 {
            debug!(
                "Measurement iteration {}/{} - {:.2} ms",
                i, config.measurement_iterations, iter_duration
            );
        }
    }

    let measurement_duration = measurement_start.elapsed().as_secs_f64() * 1000.0;

    debug!("Measurement completed in {:.2} ms", measurement_duration);

    // Calculate statistics
    let latency_stats = calculate_latency_statistics(&iteration_timings)?;

    let peak_memory = memory_samples
        .iter()
        .copied()
        .max_by(|a, b| {
            a.partial_cmp(b)
                .expect("memory sample values should be comparable")
        })
        .unwrap_or(0.0);

    let avg_memory = if !memory_samples.is_empty() {
        memory_samples.iter().sum::<f64>() / memory_samples.len() as f64
    } else {
        0.0
    };

    let throughput = if latency_stats.mean > 0.0 {
        (1000.0 / latency_stats.mean) * config.batch_size as f64
    } else {
        0.0
    };

    Ok(DetailedBenchmarkResults {
        iteration_timings,
        warmup_duration,
        measurement_duration,
        memory_samples,
        peak_memory,
        avg_memory,
        throughput,
        latency_stats,
    })
}

/// Convert detailed results to timing result format
pub fn to_timing_result(results: &DetailedBenchmarkResults) -> TimingResult {
    TimingResult {
        throughput_fps: results.throughput,
        latency_ms: results.latency_stats.mean,
        memory_mb: results.peak_memory,
        warmup_time_ms: results.warmup_duration,
        avg_inference_time_ms: results.latency_stats.mean,
        min_inference_time_ms: results.latency_stats.min,
        max_inference_time_ms: results.latency_stats.max,
        std_dev_ms: results.latency_stats.std_dev,
        device_utilization: None, // Would need device-specific profiling
    }
}

/// Calculate latency statistics from timing samples
fn calculate_latency_statistics(timings: &[f64]) -> Result<LatencyStatistics> {
    if timings.is_empty() {
        anyhow::bail!("No timing samples available for statistics");
    }

    let mut sorted_timings = timings.to_vec();
    sorted_timings.sort_by(|a, b| {
        a.partial_cmp(b)
            .expect("timing values should be comparable")
    });

    let mean = sorted_timings.iter().sum::<f64>() / sorted_timings.len() as f64;

    let variance = sorted_timings
        .iter()
        .map(|&t| (t - mean).powi(2))
        .sum::<f64>()
        / sorted_timings.len() as f64;

    let std_dev = variance.sqrt();

    let median = if sorted_timings.len() % 2 == 0 {
        let mid = sorted_timings.len() / 2;
        (sorted_timings[mid - 1] + sorted_timings[mid]) / 2.0
    } else {
        sorted_timings[sorted_timings.len() / 2]
    };

    let min = sorted_timings[0];
    let max = sorted_timings[sorted_timings.len() - 1];

    let p95_idx = ((sorted_timings.len() as f64 * 0.95) as usize).min(sorted_timings.len() - 1);
    let p95 = sorted_timings[p95_idx];

    let p99_idx = ((sorted_timings.len() as f64 * 0.99) as usize).min(sorted_timings.len() - 1);
    let p99 = sorted_timings[p99_idx];

    Ok(LatencyStatistics {
        mean,
        median,
        std_dev,
        min,
        max,
        p95,
        p99,
    })
}

/// Create input tensor for benchmarking
fn create_input_tensor(
    shape: &[usize],
    batch_size: usize,
    device: DeviceType,
) -> Result<Tensor<f32>> {
    let mut full_shape = vec![batch_size];
    full_shape.extend_from_slice(shape);

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0)?;

    let num_elements: usize = full_shape.iter().product();
    let data: Vec<f32> = (0..num_elements)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();

    Ok(Tensor::from_data(data, full_shape, device)?)
}

/// Estimate memory usage for a model
fn estimate_memory_usage(model: &TorshModel) -> f64 {
    let param_count: u64 = model.layers.iter().map(|l| l.parameters).sum();

    // Estimate total memory (parameters + activations + gradients)
    // Assuming f32 (4 bytes) and some overhead
    let memory_bytes = param_count * 4 * 2; // params + activations

    memory_bytes as f64 / (1024.0 * 1024.0)
}

/// Compare performance across different batch sizes
pub fn benchmark_batch_sizes(
    model: &TorshModel,
    batch_sizes: &[usize],
    device: DeviceType,
) -> Result<Vec<(usize, DetailedBenchmarkResults)>> {
    let mut results = Vec::new();

    for &batch_size in batch_sizes {
        info!("Benchmarking with batch size: {}", batch_size);

        let config = BenchmarkConfig {
            batch_size,
            device,
            ..Default::default()
        };

        let bench_results = benchmark_model_real(model, &config)?;
        results.push((batch_size, bench_results));
    }

    Ok(results)
}

/// Compare performance across different devices
pub fn benchmark_devices(
    model: &TorshModel,
    devices: &[DeviceType],
) -> Result<Vec<(DeviceType, DetailedBenchmarkResults)>> {
    let mut results = Vec::new();

    for &device in devices {
        info!("Benchmarking on device: {:?}", device);

        let config = BenchmarkConfig {
            device,
            ..Default::default()
        };

        let bench_results = benchmark_model_real(model, &config)?;
        results.push((device, bench_results));
    }

    Ok(results)
}

/// Format benchmark results for display
pub fn format_benchmark_results(results: &DetailedBenchmarkResults) -> String {
    format!(
        r#"
Benchmark Results:
==================
Throughput: {:.2} samples/sec
Latency (avg): {:.2} ms
Latency (median): {:.2} ms
Latency (min): {:.2} ms
Latency (max): {:.2} ms
Latency (p95): {:.2} ms
Latency (p99): {:.2} ms
Latency (std dev): {:.2} ms

Memory:
-------
Peak: {:.2} MB
Average: {:.2} MB

Timing:
-------
Warmup: {:.2} ms
Total Measurement: {:.2} ms
Iterations: {}
"#,
        results.throughput,
        results.latency_stats.mean,
        results.latency_stats.median,
        results.latency_stats.min,
        results.latency_stats.max,
        results.latency_stats.p95,
        results.latency_stats.p99,
        results.latency_stats.std_dev,
        results.peak_memory,
        results.avg_memory,
        results.warmup_duration,
        results.measurement_duration,
        results.iteration_timings.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::super::tensor_integration::create_real_model;
    use super::*;

    #[test]
    fn test_latency_statistics() {
        let timings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = calculate_latency_statistics(&timings).unwrap();

        assert!((stats.mean - 5.5).abs() < 0.1);
        assert!((stats.median - 5.5).abs() < 0.1);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.measurement_iterations, 100);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_create_input_tensor() {
        let tensor = create_input_tensor(&[3, 224, 224], 2, DeviceType::Cpu).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3, 224, 224]);
    }

    #[test]
    #[ignore = "Flaky test - passes individually but may fail in full suite"]
    fn test_benchmark_model_real() {
        let model = create_real_model("test", 2, DeviceType::Cpu).unwrap();
        let config = BenchmarkConfig {
            warmup_iterations: 2,
            measurement_iterations: 5,
            batch_size: 1,
            device: DeviceType::Cpu,
            measure_memory: true,
            collect_detailed_stats: true,
        };

        let results = benchmark_model_real(&model, &config).unwrap();
        assert_eq!(results.iteration_timings.len(), 5);
        assert!(results.throughput > 0.0);
        assert!(results.peak_memory > 0.0);
    }
}
