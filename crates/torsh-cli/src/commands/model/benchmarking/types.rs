//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use scirs2_core::ndarray::{Array2, Array3};
use std::time::Instant;
use tracing::info;

/// Kernel execution efficiency metrics
#[derive(Debug, Clone, serde::Serialize)]
pub struct KernelEfficiency {
    /// GPU occupancy (for CUDA/Metal)
    pub occupancy_percent: Option<f64>,
    /// Warp/wave efficiency
    pub warp_efficiency: Option<f64>,
    /// Cache hit rate
    pub cache_hit_rate: Option<f64>,
    /// Register usage efficiency
    pub register_efficiency: Option<f64>,
}
/// Regression comparison modes
#[derive(Debug, Clone)]
pub enum RegressionComparisonMode {
    /// Compare against previous version
    PreviousVersion,
    /// Compare against specific baseline
    Baseline,
    /// Compare against running average
    RunningAverage,
}
/// Comparison result between baseline and current benchmark
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkComparison {
    /// Whether this represents a significant regression
    pub is_regression: bool,
    /// Whether this represents a significant improvement
    pub is_improvement: bool,
    /// Latency comparison
    pub latency_change_percent: f64,
    /// Throughput comparison
    pub throughput_change_percent: f64,
    /// Memory usage comparison
    pub memory_change_percent: f64,
    /// Summary of degradation (if any)
    pub degradation_summary: String,
    /// Summary of improvement (if any)
    pub improvement_summary: String,
    /// Statistical significance
    pub statistical_significance: f64,
}
/// Profiler for benchmarking with torsh-profiler integration
#[derive(Debug)]
pub struct BenchmarkProfiler {
    /// Device being profiled
    pub device: String,
    /// Whether to profile memory
    pub profile_memory: bool,
    /// Collected metrics
    pub iteration_metrics: Vec<InferenceMetrics>,
    /// Start time of current iteration
    pub current_iteration_start: Option<Instant>,
    /// Peak memory usage observed
    pub peak_memory_mb: f64,
    /// Total FLOPS performed
    pub total_flops: u64,
}
impl BenchmarkProfiler {
    /// Create a new benchmark profiler
    pub fn new(device: String, profile_memory: bool) -> Result<Self> {
        info!("Initializing benchmark profiler for device: {}", device);
        Ok(Self {
            device,
            profile_memory,
            iteration_metrics: Vec::new(),
            current_iteration_start: None,
            peak_memory_mb: 0.0,
            total_flops: 0,
        })
    }
    /// Start profiling an iteration
    pub fn start_iteration(&mut self) {
        self.current_iteration_start = Some(Instant::now());
    }
    /// End profiling an iteration
    pub fn end_iteration(&mut self) {
        self.current_iteration_start = None;
    }
    /// Record metrics for an iteration
    pub fn record_metrics(&mut self, metrics: InferenceMetrics) {
        if metrics.memory_usage_mb > self.peak_memory_mb {
            self.peak_memory_mb = metrics.memory_usage_mb;
        }
        self.total_flops += metrics.flops;
        self.iteration_metrics.push(metrics);
    }
    /// Get profiling summary
    pub fn get_summary(&self) -> ProfilingSummary {
        let avg_device_utilization = if !self.iteration_metrics.is_empty() {
            let total_utilization: f64 = self
                .iteration_metrics
                .iter()
                .filter_map(|m| m.device_utilization)
                .sum();
            let count = self
                .iteration_metrics
                .iter()
                .filter(|m| m.device_utilization.is_some())
                .count();
            if count > 0 {
                Some(total_utilization / count as f64)
            } else {
                None
            }
        } else {
            None
        };
        let total_time_seconds: f64 = self
            .iteration_metrics
            .iter()
            .map(|m| m.computation_time_ms / 1000.0)
            .sum();
        let avg_flops_per_sec = if total_time_seconds > 0.0 {
            self.total_flops as f64 / total_time_seconds
        } else {
            0.0
        };
        let theoretical_peak_flops = match self.device.as_str() {
            "cuda" | "gpu" => 10_000_000_000.0,
            "metal" => 8_000_000_000.0,
            "cpu" => 100_000_000.0,
            _ => 1_000_000_000.0,
        };
        let computational_efficiency = (avg_flops_per_sec / theoretical_peak_flops).clamp(0.0, 1.0);
        ProfilingSummary {
            peak_memory_mb: self.peak_memory_mb,
            avg_device_utilization,
            total_flops: self.total_flops,
            avg_flops_per_sec,
            computational_efficiency,
        }
    }
}
/// Advanced performance metrics for detailed analysis
#[derive(Debug, Clone, serde::Serialize)]
pub struct AdvancedPerformanceMetrics {
    /// Percentile latencies (p50, p90, p95, p99)
    pub latency_percentiles: LatencyPercentiles,
    /// Thermal performance characteristics
    pub thermal_characteristics: ThermalCharacteristics,
    /// Memory bandwidth utilization
    pub memory_bandwidth: MemoryBandwidth,
    /// Arithmetic intensity
    pub arithmetic_intensity: f64,
    /// Kernel efficiency metrics
    pub kernel_efficiency: KernelEfficiency,
    /// Performance consistency score (0.0 to 1.0)
    pub performance_consistency: f64,
}
/// Memory bandwidth characteristics
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryBandwidth {
    /// Effective memory bandwidth in GB/s
    pub effective_bandwidth_gbs: f64,
    /// Theoretical peak bandwidth in GB/s
    pub peak_bandwidth_gbs: f64,
    /// Bandwidth utilization (0.0 to 1.0)
    pub utilization: f64,
    /// Memory access pattern efficiency
    pub access_pattern_efficiency: f64,
}
/// Benchmark input container
#[derive(Debug, Clone)]
pub struct BenchmarkInputs {
    /// Input tensors using SciRS2
    pub inputs: Vec<Array3<f32>>,
    /// Batch size
    pub batch_size: usize,
    /// Input shape
    pub shape: Vec<usize>,
}
/// Summary of profiling results
#[derive(Debug, Clone)]
pub struct ProfilingSummary {
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Average device utilization
    pub avg_device_utilization: Option<f64>,
    /// Total FLOPS performed
    pub total_flops: u64,
    /// Average FLOPS per second
    pub avg_flops_per_sec: f64,
    /// Computational efficiency
    pub computational_efficiency: f64,
}
/// Regression testing configuration
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    /// Acceptable performance degradation percentage
    pub max_degradation_percent: f64,
    /// Number of iterations for statistical significance
    pub statistical_iterations: usize,
    /// Confidence level for statistical testing
    pub confidence_level: f64,
    /// Baseline comparison mode
    pub comparison_mode: RegressionComparisonMode,
}
/// Custom benchmark definition
#[derive(Debug, Clone)]
pub struct CustomBenchmarkDefinition {
    /// Benchmark name
    pub name: String,
    /// Model path
    pub model_path: std::path::PathBuf,
    /// Input configurations
    pub input_configs: Vec<BenchmarkInputConfig>,
    /// Expected performance thresholds
    pub thresholds: PerformanceThresholds,
}
/// Model container for benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkModel {
    /// Model parameters using SciRS2
    pub parameters: Vec<Array2<f32>>,
    /// Total parameter count
    pub parameter_count: usize,
    /// Model architecture
    pub architecture: String,
    /// Device the model is on
    pub device: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
}
/// Thermal performance characteristics
#[derive(Debug, Clone, serde::Serialize)]
pub struct ThermalCharacteristics {
    /// Whether thermal throttling was detected
    pub throttling_detected: bool,
    /// Estimated performance degradation due to thermal (percentage)
    pub thermal_degradation_percent: f64,
    /// Performance stability score (0.0 to 1.0)
    pub stability_score: f64,
}
/// Performance thresholds for validation
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum acceptable latency in milliseconds
    pub max_latency_ms: f64,
    /// Minimum required throughput in FPS
    pub min_throughput_fps: f64,
    /// Maximum memory usage in MB
    pub max_memory_mb: f64,
    /// Minimum device utilization percentage
    pub min_device_utilization: Option<f64>,
}
/// Benchmark input configuration
#[derive(Debug, Clone)]
pub struct BenchmarkInputConfig {
    /// Batch size
    pub batch_size: usize,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Data type
    pub precision: String,
    /// Device target
    pub device: String,
}
/// Benchmark suite configuration for torsh-benches integration
#[derive(Debug, Clone)]
pub struct BenchmarkSuiteConfig {
    /// Standard benchmarks to run
    pub standard_benchmarks: Vec<String>,
    /// Custom benchmark definitions
    pub custom_benchmarks: Vec<CustomBenchmarkDefinition>,
    /// Regression testing configuration
    pub regression_config: Option<RegressionConfig>,
    /// Baseline results for comparison
    pub baseline_path: Option<std::path::PathBuf>,
}
/// Latency percentiles for detailed performance analysis
#[derive(Debug, Clone, serde::Serialize)]
pub struct LatencyPercentiles {
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}
/// Inference metrics for a single iteration
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Device utilization percentage
    pub device_utilization: Option<f64>,
    /// Computation time in milliseconds
    pub computation_time_ms: f64,
    /// FLOPS performed
    pub flops: u64,
}
