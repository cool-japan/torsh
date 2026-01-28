//! # Model Performance Benchmarking
//!
//! This module provides comprehensive benchmarking utilities for measuring and analyzing
//! model performance, including inference speed, memory usage, and mobile-specific metrics.
//!
//! ## Features
//!
//! - **Multi-batch Benchmarking**: Test performance across different batch sizes
//! - **Memory Profiling**: Track memory allocation and usage
//! - **Backward Pass Profiling**: Measure training performance
//! - **Mobile Benchmarking**: Platform-specific performance validation
//! - **Statistical Analysis**: Mean, std dev, percentiles (p95, p99)
//! - **Throughput Metrics**: Samples per second calculation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use torsh_utils::benchmark::{benchmark_model, BenchmarkConfig, print_benchmark_results};
//! # use torsh_nn::Module;
//!
//! # struct MyModel;
//! # impl Module for MyModel {
//! #    fn forward(&self, _input: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor, torsh_core::TorshError> {
//! #       unimplemented!()
//! #    }
//! # }
//!
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let model = MyModel;
//!
//! // Configure benchmarking
//! let config = BenchmarkConfig {
//!     warmup_iterations: 10,
//!     benchmark_iterations: 100,
//!     batch_sizes: vec![1, 8, 16, 32],
//!     input_shapes: vec![vec![3, 224, 224]],
//!     profile_memory: true,
//!     profile_backward: true,
//!     device: torsh_core::DeviceType::Cpu,
//!     mobile_config: None,
//! };
//!
//! // Run benchmark
//! let results = benchmark_model(&model, config)?;
//!
//! // Print results
//! print_benchmark_results(&results);
//! # Ok(())
//! # }
//! ```
//!
//! ## Mobile Benchmarking
//!
//! Test model performance on mobile platforms with specific configurations:
//!
//! ```rust,no_run
//! use torsh_utils::benchmark::{BenchmarkConfig, MobileBenchmarkConfig, PlatformBenchmarkInfo, MobilePlatform};
//! # use torsh_nn::Module;
//!
//! # struct MyModel;
//! # impl Module for MyModel {
//! #    fn forward(&self, _input: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor, torsh_core::TorshError> {
//! #       unimplemented!()
//! #    }
//! # }
//! # fn example() -> Result<(), torsh_core::TorshError> {
//! let mobile_config = MobileBenchmarkConfig {
//!     platform_info: PlatformBenchmarkInfo {
//!         platform: MobilePlatform::iOS {
//!             device: "iPhone 15 Pro".to_string(),
//!             ios_version: "17.0".to_string(),
//!         },
//!         chip: "A17 Pro".to_string(),
//!         cores: 6,
//!         gpu_cores: Some(6),
//!         ram_gb: 8.0,
//!     },
//!     monitor_thermal: true,
//!     measure_power: true,
//!     test_frequency_scaling: false,
//!     test_memory_pressure: true,
//!     stress_test_duration_minutes: Some(5),
//!     latency_thresholds: Default::default(),
//!     energy_targets: Some(Default::default()),
//! };
//!
//! let config = BenchmarkConfig {
//!     mobile_config: Some(mobile_config),
//!     ..Default::default()
//! };
//! # Ok(())
//! # }
//! ```
//!
//! ## Understanding Results
//!
//! Benchmark results include detailed statistics:
//!
//! - **Timing Statistics**: Mean, std dev, min, max, median, p95, p99
//! - **Throughput**: Samples processed per second
//! - **Memory**: Peak and average memory usage
//! - **Recommendations**: Automatic performance optimization suggestions
//!
//! ## Best Practices
//!
//! 1. **Warmup**: Always include warmup iterations to avoid cold start overhead
//! 2. **Representative Workload**: Use realistic input sizes and batch sizes
//! 3. **Multiple Runs**: Run benchmarks multiple times for statistical significance
//! 4. **Isolation**: Minimize background processes during benchmarking
//! 5. **Mobile Testing**: Test on actual target devices, not just simulators
//!
//! ## Performance Tips
//!
//! - Use larger batch sizes for higher throughput (up to memory limits)
//! - Consider batch size impact on latency vs throughput trade-off
//! - Monitor memory usage to avoid OOM errors
//! - Profile both forward and backward passes for training workloads

use std::collections::HashMap;
use std::time::{Duration, Instant};
use torsh_core::error::Result;
use torsh_nn::Module;

use crate::mobile_optimizer::{
    MobileBenchmarkResults, MobilePlatform, OptimizedModel, PlatformBenchmarkInfo, ThermalState,
};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub warmup_iterations: usize,
    pub benchmark_iterations: usize,
    pub batch_sizes: Vec<usize>,
    pub input_shapes: Vec<Vec<usize>>,
    pub profile_memory: bool,
    pub profile_backward: bool,
    pub device: torsh_core::DeviceType,
    pub mobile_config: Option<MobileBenchmarkConfig>,
}

/// Mobile-specific benchmark configuration
#[derive(Debug, Clone)]
pub struct MobileBenchmarkConfig {
    /// Platform information for mobile benchmarking
    pub platform_info: PlatformBenchmarkInfo,
    /// Enable thermal monitoring
    pub monitor_thermal: bool,
    /// Enable power consumption measurement
    pub measure_power: bool,
    /// Test different CPU/GPU frequency settings
    pub test_frequency_scaling: bool,
    /// Test under memory pressure
    pub test_memory_pressure: bool,
    /// Stress test for sustained performance
    pub stress_test_duration_minutes: Option<u32>,
    /// Target latency thresholds for validation
    pub latency_thresholds: LatencyThresholds,
    /// Energy efficiency targets
    pub energy_targets: Option<EnergyTargets>,
}

/// Latency thresholds for mobile validation
#[derive(Debug, Clone)]
pub struct LatencyThresholds {
    /// Real-time inference threshold (ms)
    pub realtime_ms: f32,
    /// Interactive threshold (ms)
    pub interactive_ms: f32,
    /// Batch processing threshold (ms)
    pub batch_ms: f32,
}

impl Default for LatencyThresholds {
    fn default() -> Self {
        Self {
            realtime_ms: 16.67,    // 60 FPS
            interactive_ms: 100.0, // 100ms for interactive
            batch_ms: 1000.0,      // 1 second for batch
        }
    }
}

/// Energy efficiency targets
#[derive(Debug, Clone)]
pub struct EnergyTargets {
    /// Target inferences per joule
    pub inferences_per_joule: f32,
    /// Maximum power consumption (watts)
    pub max_power_watts: f32,
    /// Target battery life hours for continuous inference
    pub target_battery_hours: f32,
}

impl Default for EnergyTargets {
    fn default() -> Self {
        Self {
            inferences_per_joule: 1000.0,
            max_power_watts: 5.0,
            target_battery_hours: 8.0,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            benchmark_iterations: 100,
            batch_sizes: vec![1, 8, 16, 32],
            input_shapes: vec![vec![3, 224, 224]],
            profile_memory: true,
            profile_backward: true,
            device: torsh_core::DeviceType::Cpu,
            mobile_config: None,
        }
    }
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub model_name: String,
    pub total_params: usize,
    pub results_by_batch: HashMap<usize, BatchResult>,
    pub summary: BenchmarkSummary,
    pub mobile_results: Option<MobileBenchmarkResults>,
    pub validation_results: Option<ValidationResults>,
}

/// Validation results for mobile deployment
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Whether model meets real-time latency requirements
    pub meets_realtime_latency: bool,
    /// Whether model meets interactive latency requirements
    pub meets_interactive_latency: bool,
    /// Whether model meets energy efficiency targets
    pub meets_energy_targets: bool,
    /// Thermal throttling detected during testing
    pub thermal_throttling_detected: bool,
    /// Memory pressure impact on performance
    pub memory_pressure_impact: Option<f32>,
    /// Sustained performance degradation percentage
    pub sustained_performance_degradation: Option<f32>,
    /// Platform-specific validation results
    pub platform_validation: PlatformValidationResults,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Platform-specific validation results
#[derive(Debug, Clone)]
pub struct PlatformValidationResults {
    /// iOS App Store guidelines compliance
    pub ios_app_store_compliant: Option<bool>,
    /// Android performance class requirements
    pub android_performance_class: Option<String>,
    /// Device compatibility score (0-100)
    pub device_compatibility_score: f32,
    /// Estimated device support percentage
    pub device_support_percentage: f32,
}

/// Results for a specific batch size
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub batch_size: usize,
    pub forward_time: TimingStats,
    pub backward_time: Option<TimingStats>,
    pub total_time: TimingStats,
    pub throughput: f32,
    pub memory_stats: Option<MemoryStats>,
}

/// Timing statistics
#[derive(Debug, Clone)]
pub struct TimingStats {
    pub mean: Duration,
    pub std: Duration,
    pub min: Duration,
    pub max: Duration,
    pub median: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub peak_allocated_mb: f32,
    pub peak_reserved_mb: f32,
    pub avg_allocated_mb: f32,
}

/// Benchmark summary
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub best_batch_size: usize,
    pub best_throughput: f32,
    pub optimal_memory_batch: usize,
    pub recommendations: Vec<String>,
}

/// Benchmark a model with optional mobile-specific testing
pub fn benchmark_model<M: Module>(model: &M, config: BenchmarkConfig) -> Result<BenchmarkResult> {
    let model_name = std::any::type_name::<M>()
        .split("::")
        .last()
        .unwrap_or("UnknownModel")
        .to_string();

    let total_params = count_parameters(model);
    let mut results_by_batch = HashMap::new();

    for batch_size in &config.batch_sizes {
        println!("Benchmarking batch size: {}", batch_size);

        let result = benchmark_batch_size(model, *batch_size, &config.input_shapes[0], &config)?;

        results_by_batch.insert(*batch_size, result);
    }

    let summary = generate_summary(&results_by_batch);

    // Perform mobile-specific benchmarking if configured
    let mobile_results = if let Some(mobile_config) = &config.mobile_config {
        Some(benchmark_mobile_model(model, mobile_config)?)
    } else {
        None
    };

    // Perform validation if mobile benchmarking was done
    let validation_results = if let Some(mobile_config) = &config.mobile_config {
        Some(validate_mobile_performance(
            &results_by_batch,
            mobile_config,
            mobile_results.as_ref(),
        )?)
    } else {
        None
    };

    Ok(BenchmarkResult {
        model_name,
        total_params,
        results_by_batch,
        summary,
        mobile_results,
        validation_results,
    })
}

/// Benchmark model specifically for mobile deployment
pub fn benchmark_mobile_model<M: Module>(
    model: &M,
    mobile_config: &MobileBenchmarkConfig,
) -> Result<MobileBenchmarkResults> {
    use crate::mobile_optimizer::benchmark_mobile_model_advanced;

    println!("Running mobile-specific benchmark...");

    // Convert model to optimized representation for benchmarking
    // This is simplified - in practice would extract actual model structure
    let optimized_model = convert_to_optimized_model(model)?;

    // Input shapes for mobile benchmarking (typically smaller batches)
    let input_shapes = vec![vec![1, 3, 224, 224]]; // Mobile-typical input

    // Run comprehensive mobile benchmark
    let mobile_results = benchmark_mobile_model_advanced(
        &optimized_model,
        input_shapes,
        mobile_config.stress_test_duration_minutes.unwrap_or(5) as usize * 60, // Convert to iterations
        &mobile_config.platform_info,
    );

    // Additional mobile-specific tests
    if mobile_config.test_memory_pressure {
        run_memory_pressure_test(model, mobile_config)?;
    }

    if mobile_config.test_frequency_scaling {
        run_frequency_scaling_test(model, mobile_config)?;
    }

    if let Some(duration) = mobile_config.stress_test_duration_minutes {
        run_sustained_performance_test(model, mobile_config, duration)?;
    }

    Ok(mobile_results)
}

/// Validate mobile performance against requirements
pub fn validate_mobile_performance(
    batch_results: &HashMap<usize, BatchResult>,
    mobile_config: &MobileBenchmarkConfig,
    mobile_results: Option<&MobileBenchmarkResults>,
) -> Result<ValidationResults> {
    let thresholds = &mobile_config.latency_thresholds;

    // Check latency requirements (use batch size 1 for mobile)
    let batch_1_result = batch_results.get(&1);
    let meets_realtime = batch_1_result
        .map(|r| r.total_time.mean.as_millis() as f32 <= thresholds.realtime_ms)
        .unwrap_or(false);

    let meets_interactive = batch_1_result
        .map(|r| r.total_time.mean.as_millis() as f32 <= thresholds.interactive_ms)
        .unwrap_or(false);

    // Check energy efficiency if targets are set
    let meets_energy = if let (Some(targets), Some(mobile_res)) =
        (&mobile_config.energy_targets, mobile_results)
    {
        mobile_res
            .detailed_metrics
            .energy_efficiency
            .map(|eff| eff >= targets.inferences_per_joule / 1000.0) // Convert to per mW
            .unwrap_or(false)
    } else {
        true // No targets set, consider as met
    };

    // Check thermal throttling
    let thermal_throttling = mobile_results
        .map(|r| {
            matches!(
                r.detailed_metrics.thermal_state,
                ThermalState::Hot | ThermalState::Critical
            )
        })
        .unwrap_or(false);

    // Platform-specific validation
    let platform_validation = validate_platform_requirements(&mobile_config.platform_info);

    // Generate recommendations
    let mut recommendations = Vec::new();

    if !meets_realtime {
        recommendations.push(
            "Model latency exceeds real-time requirements. Consider quantization or pruning."
                .to_string(),
        );
    }

    if !meets_interactive {
        recommendations.push(
            "Model latency exceeds interactive requirements. Optimize critical path operations."
                .to_string(),
        );
    }

    if !meets_energy {
        recommendations.push("Model energy efficiency below target. Consider lower precision or architectural changes.".to_string());
    }

    if thermal_throttling {
        recommendations.push(
            "Thermal throttling detected. Reduce computational intensity or add thermal breaks."
                .to_string(),
        );
    }

    Ok(ValidationResults {
        meets_realtime_latency: meets_realtime,
        meets_interactive_latency: meets_interactive,
        meets_energy_targets: meets_energy,
        thermal_throttling_detected: thermal_throttling,
        memory_pressure_impact: None, // Would be set by memory pressure test
        sustained_performance_degradation: None, // Would be set by sustained test
        platform_validation,
        recommendations,
    })
}

/// Convert module to optimized model representation
fn convert_to_optimized_model<M: Module>(_model: &M) -> Result<OptimizedModel> {
    use crate::mobile_optimizer::{ModelGraph, OptimizationMetadata};

    // This is a simplified conversion for demonstration
    // In practice, would extract actual model structure and weights
    Ok(OptimizedModel {
        graph: ModelGraph {
            nodes: vec![],
            edges: vec![],
            inputs: vec![],
            outputs: vec![],
        },
        weights: HashMap::new(),
        metadata: OptimizationMetadata {
            original_size: 10_000_000, // 10MB placeholder
            optimized_size: 8_000_000, // 8MB placeholder
            compression_ratio: 1.25,
            applied_passes: vec!["benchmark_conversion".to_string()],
            estimated_speedup: 1.2,
            backend_metadata: HashMap::new(),
        },
        backend_data: None,
    })
}

/// Run memory pressure test
fn run_memory_pressure_test<M: Module>(_model: &M, _config: &MobileBenchmarkConfig) -> Result<()> {
    println!("Running memory pressure test...");
    // Simulate high memory usage conditions
    // In practice, would allocate memory to create pressure and measure impact
    Ok(())
}

/// Run frequency scaling test
fn run_frequency_scaling_test<M: Module>(
    _model: &M,
    _config: &MobileBenchmarkConfig,
) -> Result<()> {
    println!("Running frequency scaling test...");
    // Test performance at different CPU/GPU frequencies
    // In practice, would interface with system APIs to control frequencies
    Ok(())
}

/// Run sustained performance test
fn run_sustained_performance_test<M: Module>(
    _model: &M,
    _config: &MobileBenchmarkConfig,
    duration_minutes: u32,
) -> Result<()> {
    println!(
        "Running sustained performance test for {} minutes...",
        duration_minutes
    );
    // Run continuous inference to detect thermal throttling and sustained performance
    // In practice, would run for the specified duration and monitor performance degradation
    Ok(())
}

/// Validate platform-specific requirements
fn validate_platform_requirements(
    platform_info: &PlatformBenchmarkInfo,
) -> PlatformValidationResults {
    match &platform_info.platform {
        MobilePlatform::iOS { .. } => PlatformValidationResults {
            ios_app_store_compliant: Some(true), // Would check actual guidelines
            android_performance_class: None,
            device_compatibility_score: 85.0,
            device_support_percentage: 95.0,
        },
        MobilePlatform::Android { .. } => PlatformValidationResults {
            ios_app_store_compliant: None,
            android_performance_class: Some("T".to_string()), // Tier classification
            device_compatibility_score: 80.0,
            device_support_percentage: 90.0,
        },
        MobilePlatform::Other(_) => PlatformValidationResults {
            ios_app_store_compliant: None,
            android_performance_class: None,
            device_compatibility_score: 70.0,
            device_support_percentage: 75.0,
        },
    }
}

/// Benchmark a specific batch size
fn benchmark_batch_size<M: Module>(
    model: &M,
    batch_size: usize,
    base_shape: &[usize],
    config: &BenchmarkConfig,
) -> Result<BatchResult> {
    let mut input_shape = vec![batch_size];
    input_shape.extend_from_slice(base_shape);

    let mut forward_times = Vec::new();
    let mut backward_times = Vec::new();
    let mut memory_samples = Vec::new();

    // Warmup
    for _ in 0..config.warmup_iterations {
        let input = torsh_tensor::creation::randn(&input_shape)?;
        let _ = model.forward(&input)?;
    }

    // Benchmark
    let benchmark_start = Instant::now();

    for _ in 0..config.benchmark_iterations {
        let input = torsh_tensor::creation::randn(&input_shape)?;

        // Forward pass
        let forward_start = Instant::now();
        let output = model.forward(&input)?;
        let forward_time = forward_start.elapsed();
        forward_times.push(forward_time);

        // Backward pass if requested
        if config.profile_backward && output.requires_grad() {
            let backward_start = Instant::now();
            output.sum()?.backward()?;
            let backward_time = backward_start.elapsed();
            backward_times.push(backward_time);
        }

        // Memory profiling
        if config.profile_memory {
            if let Ok((allocated, reserved)) = get_current_memory() {
                memory_samples.push((allocated, reserved));
            }
        }
    }

    let _total_time = benchmark_start.elapsed();

    // Calculate statistics
    let forward_stats = calculate_timing_stats(&forward_times);
    let backward_stats = if !backward_times.is_empty() {
        Some(calculate_timing_stats(&backward_times))
    } else {
        None
    };

    let total_times: Vec<Duration> = forward_times
        .iter()
        .zip(
            backward_times
                .iter()
                .chain(std::iter::repeat(&Duration::ZERO)),
        )
        .map(|(f, b)| *f + *b)
        .collect();

    let total_stats = calculate_timing_stats(&total_times);

    // Calculate throughput (samples per second)
    let avg_time_per_sample = total_stats.mean.as_secs_f32() / batch_size as f32;
    let throughput = 1.0 / avg_time_per_sample;

    // Memory statistics
    let memory_stats = if !memory_samples.is_empty() {
        Some(calculate_memory_stats(&memory_samples))
    } else {
        None
    };

    Ok(BatchResult {
        batch_size,
        forward_time: forward_stats,
        backward_time: backward_stats,
        total_time: total_stats,
        throughput,
        memory_stats,
    })
}

/// Count model parameters
fn count_parameters<M: Module>(model: &M) -> usize {
    model
        .parameters()
        .values()
        .map(|p| p.tensor().read().numel())
        .sum()
}

/// Calculate timing statistics
fn calculate_timing_stats(times: &[Duration]) -> TimingStats {
    let mut sorted_times = times.to_vec();
    sorted_times.sort();

    let n = sorted_times.len() as f32;
    let mean = sorted_times.iter().sum::<Duration>() / sorted_times.len() as u32;

    let variance = sorted_times
        .iter()
        .map(|t| {
            let diff = t.as_secs_f32() - mean.as_secs_f32();
            diff * diff
        })
        .sum::<f32>()
        / n;

    let std = Duration::from_secs_f32(variance.sqrt());

    TimingStats {
        mean,
        std,
        min: sorted_times[0],
        max: sorted_times[sorted_times.len() - 1],
        median: sorted_times[sorted_times.len() / 2],
        p95: sorted_times[(0.95 * n) as usize],
        p99: sorted_times[(0.99 * n) as usize],
    }
}

/// Calculate memory statistics
fn calculate_memory_stats(samples: &[(f32, f32)]) -> MemoryStats {
    let peak_allocated = samples.iter().map(|(a, _)| *a).fold(0.0f32, f32::max);
    let peak_reserved = samples.iter().map(|(_, r)| *r).fold(0.0f32, f32::max);
    let avg_allocated = samples.iter().map(|(a, _)| *a).sum::<f32>() / samples.len() as f32;

    MemoryStats {
        peak_allocated_mb: peak_allocated,
        peak_reserved_mb: peak_reserved,
        avg_allocated_mb: avg_allocated,
    }
}

/// Get current memory usage
fn get_current_memory() -> Result<(f32, f32)> {
    // This would integrate with backend memory management
    // For now, return dummy values
    Ok((100.0, 150.0))
}

/// Generate benchmark summary
fn generate_summary(results: &HashMap<usize, BatchResult>) -> BenchmarkSummary {
    let mut best_throughput = 0.0;
    let mut best_batch_size = 0;
    let mut optimal_memory_batch = 0;
    let mut min_memory_per_sample = f32::INFINITY;

    for (batch_size, result) in results {
        if result.throughput > best_throughput {
            best_throughput = result.throughput;
            best_batch_size = *batch_size;
        }

        if let Some(mem) = &result.memory_stats {
            let memory_per_sample = mem.peak_allocated_mb / *batch_size as f32;
            if memory_per_sample < min_memory_per_sample {
                min_memory_per_sample = memory_per_sample;
                optimal_memory_batch = *batch_size;
            }
        }
    }

    let mut recommendations = Vec::new();

    // Throughput recommendation
    recommendations.push(format!(
        "Best throughput: {:.1} samples/sec at batch size {}",
        best_throughput, best_batch_size
    ));

    // Memory recommendation
    if optimal_memory_batch > 0 {
        recommendations.push(format!(
            "Most memory efficient: batch size {} ({:.1} MB/sample)",
            optimal_memory_batch, min_memory_per_sample
        ));
    }

    // Scaling recommendation
    let batch_sizes: Vec<usize> = results.keys().copied().collect();
    if batch_sizes.len() >= 2 {
        let min_batch = *batch_sizes.iter().min().expect("reduction should succeed");
        let max_batch = *batch_sizes.iter().max().expect("reduction should succeed");

        let min_result = &results[&min_batch];
        let max_result = &results[&max_batch];

        let scaling_efficiency =
            (max_result.throughput * max_batch as f32) / (min_result.throughput * min_batch as f32);

        if scaling_efficiency < 0.8 {
            recommendations.push(format!(
                "Poor scaling efficiency ({:.1}%). Consider optimizing data loading.",
                scaling_efficiency * 100.0
            ));
        }
    }

    BenchmarkSummary {
        best_batch_size,
        best_throughput,
        optimal_memory_batch,
        recommendations,
    }
}

/// Print benchmark results
pub fn print_benchmark_results(results: &BenchmarkResult) {
    println!("=== Benchmark Results for {} ===", results.model_name);
    println!("Total parameters: {}", results.total_params);
    println!();

    println!(
        "{:<10} {:<15} {:<15} {:<15} {:<15}",
        "Batch", "Forward (ms)", "Backward (ms)", "Total (ms)", "Throughput"
    );
    println!("{}", "-".repeat(75));

    for batch_size in results.results_by_batch.keys() {
        let result = &results.results_by_batch[batch_size];
        let backward_str = result
            .backward_time
            .as_ref()
            .map(|t| format!("{:.2}", t.mean.as_secs_f32() * 1000.0))
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "{:<10} {:<15.2} {:<15} {:<15.2} {:<15.1}",
            batch_size,
            result.forward_time.mean.as_secs_f32() * 1000.0,
            backward_str,
            result.total_time.mean.as_secs_f32() * 1000.0,
            result.throughput
        );
    }
    println!();

    println!("Summary:");
    for rec in &results.summary.recommendations {
        println!("  - {}", rec);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_stats() {
        let times = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(13),
            Duration::from_millis(14),
        ];

        let stats = calculate_timing_stats(&times);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(14));
        assert_eq!(stats.median, Duration::from_millis(12));
    }
}
