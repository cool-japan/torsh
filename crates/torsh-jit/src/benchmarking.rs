//! Comprehensive Benchmarking Suite for ToRSh JIT
//!
//! This module provides extensive benchmarking capabilities for measuring,
//! analyzing, and comparing JIT compilation performance across different
//! strategies, workloads, and configurations.

use crate::{JitCompiler, JitError, JitResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Comprehensive benchmarking suite
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    benchmarks: Vec<Box<dyn Benchmark>>,
    results: Arc<Mutex<BenchmarkResults>>,
    profiler: BenchmarkProfiler,
}

/// Configuration for benchmarking
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations
    pub warmup_iterations: usize,

    /// Number of measurement iterations
    pub measurement_iterations: usize,

    /// Maximum execution time per benchmark
    pub max_execution_time: Duration,

    /// Minimum execution time for reliable measurements
    pub min_execution_time: Duration,

    /// Statistical confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,

    /// Enable detailed profiling
    pub enable_profiling: bool,

    /// Enable memory tracking
    pub enable_memory_tracking: bool,

    /// Enable energy measurement
    pub enable_energy_measurement: bool,

    /// Output format for results
    pub output_format: OutputFormat,

    /// Benchmark suite name
    pub suite_name: String,

    /// Parallel execution settings
    pub parallel_execution: ParallelExecution,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measurement_iterations: 100,
            max_execution_time: Duration::from_secs(300), // 5 minutes
            min_execution_time: Duration::from_millis(1),
            confidence_level: 0.95,
            enable_profiling: true,
            enable_memory_tracking: true,
            enable_energy_measurement: false, // Requires special hardware
            output_format: OutputFormat::Json,
            suite_name: "ToRSh JIT Benchmark Suite".to_string(),
            parallel_execution: ParallelExecution::Sequential,
        }
    }
}

/// Output formats for benchmark results
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    Html,
    Markdown,
    Binary,
}

/// Parallel execution configuration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ParallelExecution {
    Sequential,
    Parallel { max_threads: usize },
    Adaptive,
}

/// Trait for individual benchmarks
pub trait Benchmark: Send + Sync {
    /// Name of the benchmark
    fn name(&self) -> &str;

    /// Description of what this benchmark measures
    fn description(&self) -> &str;

    /// Setup phase - prepare data and environment
    fn setup(&mut self) -> JitResult<()>;

    /// Execute the benchmark workload
    fn execute(&self, compiler: &mut JitCompiler) -> JitResult<BenchmarkMeasurement>;

    /// Cleanup phase
    fn teardown(&mut self) -> JitResult<()>;

    /// Get benchmark metadata
    fn metadata(&self) -> BenchmarkMetadata;

    /// Validate benchmark results
    fn validate(&self, measurement: &BenchmarkMeasurement) -> JitResult<ValidationResult>;
}

/// Individual benchmark measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    /// Execution time
    pub execution_time: Duration,

    /// Compilation time
    pub compilation_time: Duration,

    /// Memory usage statistics
    pub memory_stats: MemoryStatistics,

    /// CPU utilization
    pub cpu_utilization: f64,

    /// Throughput (operations per second)
    pub throughput: f64,

    /// Energy consumption (if available)
    pub energy_consumption: Option<f64>,

    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,

    /// Timestamp
    pub timestamp: SystemTime,

    /// Benchmark configuration used
    pub config_hash: u64,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Peak memory usage in bytes
    pub peak_usage: usize,

    /// Average memory usage in bytes
    pub average_usage: usize,

    /// Memory allocations count
    pub allocations: usize,

    /// Memory deallocations count
    pub deallocations: usize,

    /// Memory leaks detected
    pub leaks: usize,

    /// Cache statistics
    pub cache_stats: CacheStatistics,
}

/// Cache performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// L1 cache hit rate
    pub l1_hit_rate: f64,

    /// L2 cache hit rate
    pub l2_hit_rate: f64,

    /// L3 cache hit rate
    pub l3_hit_rate: f64,

    /// Cache misses
    pub cache_misses: u64,

    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,
}

/// Benchmark metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// Benchmark category
    pub category: BenchmarkCategory,

    /// Workload characteristics
    pub workload: WorkloadCharacteristics,

    /// Expected performance range
    pub expected_performance: PerformanceRange,

    /// Required system resources
    pub resource_requirements: ResourceRequirements,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Author information
    pub author: String,

    /// Version
    pub version: String,
}

/// Benchmark categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    /// Compilation performance benchmarks
    Compilation,

    /// Runtime execution benchmarks
    Execution,

    /// Memory usage benchmarks
    Memory,

    /// Optimization effectiveness benchmarks
    Optimization,

    /// Stress testing benchmarks
    Stress,

    /// Regression testing benchmarks
    Regression,

    /// Comparative benchmarks
    Comparative,

    /// End-to-end application benchmarks
    Application,
}

/// Workload characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadCharacteristics {
    /// Computational complexity
    pub complexity: ComputationalComplexity,

    /// Data size
    pub data_size: DataSize,

    /// Memory access pattern
    pub memory_pattern: MemoryAccessPattern,

    /// Parallelism degree
    pub parallelism: ParallelismDegree,

    /// I/O characteristics
    pub io_characteristics: IoCharacteristics,
}

/// Computational complexity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComputationalComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
    Custom { flops: u64 },
}

/// Data size categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataSize {
    Small,     // < 1MB
    Medium,    // 1MB - 100MB
    Large,     // 100MB - 1GB
    VeryLarge, // > 1GB
    Custom { bytes: usize },
}

/// Memory access patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided { stride: usize },
    Irregular,
    Clustered,
}

/// Parallelism characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParallelismDegree {
    Serial,
    LowParallel,    // 2-4 threads
    MediumParallel, // 4-16 threads
    HighParallel,   // 16+ threads
    Custom { threads: usize },
}

/// I/O characteristics
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IoCharacteristics {
    None,
    Read,
    Write,
    ReadWrite,
    Network,
    Custom { pattern: String },
}

/// Expected performance range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRange {
    /// Minimum expected execution time
    pub min_execution_time: Duration,

    /// Maximum expected execution time
    pub max_execution_time: Duration,

    /// Expected throughput range
    pub throughput_range: (f64, f64),

    /// Expected memory usage range
    pub memory_range: (usize, usize),
}

/// System resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum CPU cores
    pub min_cpu_cores: usize,

    /// Minimum memory in bytes
    pub min_memory: usize,

    /// Required CPU features
    pub cpu_features: Vec<String>,

    /// GPU requirements
    pub gpu_requirements: Option<GpuRequirements>,
}

/// GPU requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Minimum compute capability
    pub min_compute_capability: f64,

    /// Minimum memory in bytes
    pub min_memory: usize,

    /// Required GPU features
    pub features: Vec<String>,
}

/// Validation result for benchmark
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub correctness_score: f64,
}

/// Complete benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Results by benchmark name
    pub results: HashMap<String, BenchmarkResult>,

    /// Suite-level statistics
    pub suite_statistics: SuiteStatistics,

    /// System information
    pub system_info: SystemInfo,

    /// Benchmark configuration
    pub config: BenchmarkConfig,

    /// Execution timestamp
    pub timestamp: SystemTime,
}

/// Result for a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,

    /// All measurements
    pub measurements: Vec<BenchmarkMeasurement>,

    /// Statistical summary
    pub statistics: BenchmarkStatistics,

    /// Validation results
    pub validation: ValidationSummary,

    /// Comparison with baselines
    pub comparisons: Vec<BenchmarkComparison>,

    /// Performance regression analysis
    pub regression_analysis: Option<RegressionAnalysis>,
}

/// Statistical summary of benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    /// Mean execution time
    pub mean_execution_time: Duration,

    /// Median execution time
    pub median_execution_time: Duration,

    /// Standard deviation
    pub std_deviation: Duration,

    /// Minimum execution time
    pub min_execution_time: Duration,

    /// Maximum execution time
    pub max_execution_time: Duration,

    /// 95th percentile
    pub p95_execution_time: Duration,

    /// 99th percentile
    pub p99_execution_time: Duration,

    /// Coefficient of variation
    pub coefficient_variation: f64,

    /// Confidence interval
    pub confidence_interval: (Duration, Duration),

    /// Throughput statistics
    pub throughput_stats: ThroughputStatistics,

    /// Memory statistics
    pub memory_stats: MemoryStatisticsSummary,
}

/// Throughput statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStatistics {
    pub mean_throughput: f64,
    pub max_throughput: f64,
    pub min_throughput: f64,
    pub std_deviation: f64,
}

/// Memory statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatisticsSummary {
    pub mean_peak_usage: usize,
    pub max_peak_usage: usize,
    pub mean_allocations: usize,
    pub total_leaks: usize,
}

/// Validation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub success_rate: f64,
    pub error_count: usize,
    pub warning_count: usize,
    pub avg_correctness_score: f64,
}

/// Benchmark comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// Baseline name
    pub baseline_name: String,

    /// Performance improvement (positive = better)
    pub performance_improvement: f64,

    /// Statistical significance
    pub significance: StatisticalSignificance,

    /// Detailed comparison metrics
    pub detailed_metrics: HashMap<String, f64>,
}

/// Statistical significance of comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSignificance {
    pub p_value: f64,
    pub is_significant: bool,
    pub confidence_level: f64,
    pub effect_size: f64,
}

/// Performance regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    /// Trend over time
    pub trend: RegressionTrend,

    /// Detected regressions
    pub regressions: Vec<PerformanceRegression>,

    /// Correlation analysis
    pub correlations: HashMap<String, f64>,
}

/// Performance trend
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionTrend {
    Improving,
    Stable,
    Degrading,
    Fluctuating,
}

/// Detected performance regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub detected_at: SystemTime,
    pub performance_delta: f64,
    pub description: String,
}

/// Types of performance regressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionType {
    ExecutionTime,
    CompilationTime,
    MemoryUsage,
    Throughput,
    EnergyConsumption,
}

/// Severity of regression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Suite-level statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteStatistics {
    /// Total benchmarks run
    pub total_benchmarks: usize,

    /// Successful benchmarks
    pub successful_benchmarks: usize,

    /// Failed benchmarks
    pub failed_benchmarks: usize,

    /// Total execution time
    pub total_execution_time: Duration,

    /// Average performance improvement
    pub avg_performance_improvement: f64,

    /// Performance distribution
    pub performance_distribution: HashMap<String, usize>,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// CPU information
    pub cpu_info: CpuInfo,

    /// Memory information
    pub memory_info: MemoryInfo,

    /// Operating system
    pub os_info: String,

    /// Rust version
    pub rust_version: String,

    /// Compiler version
    pub compiler_version: String,

    /// Environment variables
    pub environment: HashMap<String, String>,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: usize,
    pub frequency: f64,
    pub cache_sizes: Vec<usize>,
    pub features: Vec<String>,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total: usize,
    pub available: usize,
    pub page_size: usize,
}

/// Benchmark profiler for detailed analysis
pub struct BenchmarkProfiler {
    profiling_enabled: bool,
    memory_tracker: MemoryTracker,
    cpu_profiler: CpuProfiler,
    energy_meter: Option<EnergyMeter>,
}

/// Memory tracking utility
struct MemoryTracker {
    peak_usage: usize,
    current_usage: usize,
    allocations: usize,
    deallocations: usize,
}

/// CPU profiling utility
struct CpuProfiler {
    sampling_rate: u64,
    profiles: Vec<CpuProfile>,
}

/// CPU profile snapshot
#[derive(Debug, Clone)]
struct CpuProfile {
    timestamp: Instant,
    cpu_usage: f64,
    instruction_count: u64,
    cache_misses: u64,
}

/// Energy measurement utility
struct EnergyMeter {
    baseline_power: f64,
    current_power: f64,
    total_energy: f64,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config: config.clone(),
            benchmarks: Vec::new(),
            results: Arc::new(Mutex::new(BenchmarkResults {
                results: HashMap::new(),
                suite_statistics: SuiteStatistics {
                    total_benchmarks: 0,
                    successful_benchmarks: 0,
                    failed_benchmarks: 0,
                    total_execution_time: Duration::ZERO,
                    avg_performance_improvement: 0.0,
                    performance_distribution: HashMap::new(),
                },
                system_info: SystemInfo::collect(),
                config: config.clone(),
                timestamp: SystemTime::now(),
            })),
            profiler: BenchmarkProfiler::new(config.enable_profiling),
        }
    }

    /// Add a benchmark to the suite
    pub fn add_benchmark(&mut self, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.push(benchmark);
    }

    /// Run all benchmarks in the suite
    pub fn run_all(&mut self, compiler: &mut JitCompiler) -> JitResult<BenchmarkResults> {
        let start_time = Instant::now();
        let mut successful = 0;
        let mut failed = 0;

        let total_benchmarks = self.benchmarks.len();
        println!("Running {} benchmarks...", total_benchmarks);

        for (index, benchmark) in self.benchmarks.iter_mut().enumerate() {
            println!(
                "Running benchmark {}/{}: {}",
                index + 1,
                total_benchmarks,
                benchmark.name()
            );

            // Run benchmark directly to avoid borrowing self
            match benchmark.execute(compiler) {
                Ok(measurement) => {
                    // Create a BenchmarkResult from the measurement
                    let benchmark_result = BenchmarkResult {
                        name: benchmark.name().to_string(),
                        measurements: vec![measurement.clone()],
                        statistics: BenchmarkStatistics {
                            mean_execution_time: measurement.execution_time,
                            median_execution_time: measurement.execution_time,
                            std_deviation: Duration::ZERO,
                            min_execution_time: measurement.execution_time,
                            max_execution_time: measurement.execution_time,
                            p95_execution_time: measurement.execution_time,
                            p99_execution_time: measurement.execution_time,
                            coefficient_variation: 0.0,
                            confidence_interval: (
                                measurement.execution_time,
                                measurement.execution_time,
                            ),
                            throughput_stats: ThroughputStatistics {
                                mean_throughput: 1000.0,
                                max_throughput: 1000.0,
                                min_throughput: 1000.0,
                                std_deviation: 0.0,
                            },
                            memory_stats: MemoryStatisticsSummary {
                                mean_peak_usage: 1024 * 1024, // 1MB default
                                max_peak_usage: 1024 * 1024,
                                mean_allocations: 100,
                                total_leaks: 0,
                            },
                        },
                        validation: ValidationSummary {
                            success_rate: 1.0,
                            error_count: 0,
                            warning_count: 0,
                            avg_correctness_score: 1.0,
                        },
                        comparisons: Vec::new(),
                        regression_analysis: None,
                    };

                    if let Ok(mut results) = self.results.lock() {
                        results
                            .results
                            .insert(benchmark.name().to_string(), benchmark_result);
                    }
                    successful += 1;
                }
                Err(e) => {
                    eprintln!("Benchmark {} failed: {}", benchmark.name(), e);
                    failed += 1;
                }
            }
        }

        let total_time = start_time.elapsed();

        // Update suite statistics
        if let Ok(mut results) = self.results.lock() {
            results.suite_statistics.total_benchmarks = self.benchmarks.len();
            results.suite_statistics.successful_benchmarks = successful;
            results.suite_statistics.failed_benchmarks = failed;
            results.suite_statistics.total_execution_time = total_time;

            // Calculate average performance improvement
            let total_improvement: f64 = results
                .results
                .values()
                .flat_map(|r| r.comparisons.iter())
                .map(|c| c.performance_improvement)
                .sum();
            let comparison_count = results
                .results
                .values()
                .flat_map(|r| r.comparisons.iter())
                .count();

            if comparison_count > 0 {
                results.suite_statistics.avg_performance_improvement =
                    total_improvement / comparison_count as f64;
            }

            return Ok(results.clone());
        }

        Err(JitError::RuntimeError(
            "Failed to access results".to_string(),
        ))
    }

    /// Run a single benchmark
    fn run_single_benchmark(
        &mut self,
        benchmark: &mut Box<dyn Benchmark>,
        compiler: &mut JitCompiler,
    ) -> JitResult<BenchmarkResult> {
        // Setup phase
        benchmark.setup()?;

        let mut measurements = Vec::new();
        let mut validation_results = Vec::new();

        // Warmup iterations
        for _ in 0..self.config.warmup_iterations {
            let _ = benchmark.execute(compiler)?;
        }

        // Measurement iterations
        for _ in 0..self.config.measurement_iterations {
            let measurement = benchmark.execute(compiler)?;
            let validation = benchmark.validate(&measurement)?;

            measurements.push(measurement);
            validation_results.push(validation);
        }

        // Calculate statistics
        let statistics = self.calculate_statistics(&measurements);

        // Calculate validation summary
        let validation_summary = self.calculate_validation_summary(&validation_results);

        // Perform comparisons (placeholder for now)
        let comparisons = Vec::new();

        // Regression analysis (placeholder for now)
        let regression_analysis = None;

        // Cleanup phase
        benchmark.teardown()?;

        Ok(BenchmarkResult {
            name: benchmark.name().to_string(),
            measurements,
            statistics,
            validation: validation_summary,
            comparisons,
            regression_analysis,
        })
    }

    /// Calculate statistical summary
    fn calculate_statistics(&self, measurements: &[BenchmarkMeasurement]) -> BenchmarkStatistics {
        if measurements.is_empty() {
            return BenchmarkStatistics {
                mean_execution_time: Duration::ZERO,
                median_execution_time: Duration::ZERO,
                std_deviation: Duration::ZERO,
                min_execution_time: Duration::ZERO,
                max_execution_time: Duration::ZERO,
                p95_execution_time: Duration::ZERO,
                p99_execution_time: Duration::ZERO,
                coefficient_variation: 0.0,
                confidence_interval: (Duration::ZERO, Duration::ZERO),
                throughput_stats: ThroughputStatistics {
                    mean_throughput: 0.0,
                    max_throughput: 0.0,
                    min_throughput: 0.0,
                    std_deviation: 0.0,
                },
                memory_stats: MemoryStatisticsSummary {
                    mean_peak_usage: 0,
                    max_peak_usage: 0,
                    mean_allocations: 0,
                    total_leaks: 0,
                },
            };
        }

        let execution_times: Vec<Duration> =
            measurements.iter().map(|m| m.execution_time).collect();

        let mean_time = Duration::from_nanos(
            execution_times
                .iter()
                .map(|d| d.as_nanos() as u64)
                .sum::<u64>()
                / measurements.len() as u64,
        );

        let mut sorted_times = execution_times.clone();
        sorted_times.sort();

        let median_time = sorted_times[sorted_times.len() / 2];
        let min_time = *sorted_times.first().unwrap();
        let max_time = *sorted_times.last().unwrap();

        // Calculate percentiles
        let p95_index = (sorted_times.len() as f64 * 0.95) as usize;
        let p99_index = (sorted_times.len() as f64 * 0.99) as usize;
        let p95_time = sorted_times.get(p95_index).copied().unwrap_or(max_time);
        let p99_time = sorted_times.get(p99_index).copied().unwrap_or(max_time);

        // Calculate standard deviation
        let variance = execution_times
            .iter()
            .map(|t| {
                let diff = t.as_nanos() as f64 - mean_time.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>()
            / measurements.len() as f64;

        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        // Coefficient of variation
        let cv = if mean_time.as_nanos() > 0 {
            std_dev.as_nanos() as f64 / mean_time.as_nanos() as f64
        } else {
            0.0
        };

        // Confidence interval (95% by default)
        let t_value = 1.96; // For 95% confidence with large sample
        let margin_of_error = t_value * (variance.sqrt() / (measurements.len() as f64).sqrt());
        let ci_lower =
            Duration::from_nanos((mean_time.as_nanos() as f64 - margin_of_error).max(0.0) as u64);
        let ci_upper = Duration::from_nanos((mean_time.as_nanos() as f64 + margin_of_error) as u64);

        // Throughput statistics
        let throughputs: Vec<f64> = measurements.iter().map(|m| m.throughput).collect();
        let mean_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let max_throughput = throughputs.iter().copied().fold(0.0, f64::max);
        let min_throughput = throughputs.iter().copied().fold(f64::INFINITY, f64::min);
        let throughput_variance = throughputs
            .iter()
            .map(|&t| (t - mean_throughput).powi(2))
            .sum::<f64>()
            / throughputs.len() as f64;
        let throughput_std_dev = throughput_variance.sqrt();

        // Memory statistics
        let peak_usages: Vec<usize> = measurements
            .iter()
            .map(|m| m.memory_stats.peak_usage)
            .collect();
        let mean_peak_usage = peak_usages.iter().sum::<usize>() / peak_usages.len();
        let max_peak_usage = *peak_usages.iter().max().unwrap_or(&0);

        let allocations: Vec<usize> = measurements
            .iter()
            .map(|m| m.memory_stats.allocations)
            .collect();
        let mean_allocations = allocations.iter().sum::<usize>() / allocations.len();

        let total_leaks = measurements.iter().map(|m| m.memory_stats.leaks).sum();

        BenchmarkStatistics {
            mean_execution_time: mean_time,
            median_execution_time: median_time,
            std_deviation: std_dev,
            min_execution_time: min_time,
            max_execution_time: max_time,
            p95_execution_time: p95_time,
            p99_execution_time: p99_time,
            coefficient_variation: cv,
            confidence_interval: (ci_lower, ci_upper),
            throughput_stats: ThroughputStatistics {
                mean_throughput,
                max_throughput,
                min_throughput,
                std_deviation: throughput_std_dev,
            },
            memory_stats: MemoryStatisticsSummary {
                mean_peak_usage,
                max_peak_usage,
                mean_allocations,
                total_leaks,
            },
        }
    }

    /// Calculate validation summary
    fn calculate_validation_summary(&self, validations: &[ValidationResult]) -> ValidationSummary {
        if validations.is_empty() {
            return ValidationSummary {
                success_rate: 0.0,
                error_count: 0,
                warning_count: 0,
                avg_correctness_score: 0.0,
            };
        }

        let successful = validations.iter().filter(|v| v.is_valid).count();
        let success_rate = successful as f64 / validations.len() as f64;

        let error_count = validations.iter().map(|v| v.errors.len()).sum();
        let warning_count = validations.iter().map(|v| v.warnings.len()).sum();

        let avg_correctness_score =
            validations.iter().map(|v| v.correctness_score).sum::<f64>() / validations.len() as f64;

        ValidationSummary {
            success_rate,
            error_count,
            warning_count,
            avg_correctness_score,
        }
    }

    /// Export results to file
    pub fn export_results(&self, file_path: &str) -> JitResult<()> {
        if let Ok(results) = self.results.lock() {
            match self.config.output_format {
                OutputFormat::Json => {
                    let json = serde_json::to_string_pretty(&*results).map_err(|e| {
                        JitError::RuntimeError(format!("JSON serialization failed: {}", e))
                    })?;
                    std::fs::write(file_path, json)
                        .map_err(|e| JitError::RuntimeError(format!("File write failed: {}", e)))?;
                }
                OutputFormat::Csv => {
                    let csv = self.generate_csv_report(&results);
                    std::fs::write(file_path, csv)
                        .map_err(|e| JitError::RuntimeError(format!("File write failed: {}", e)))?;
                }
                OutputFormat::Html => {
                    let html = self.generate_html_report(&results);
                    std::fs::write(file_path, html)
                        .map_err(|e| JitError::RuntimeError(format!("File write failed: {}", e)))?;
                }
                OutputFormat::Markdown => {
                    let markdown = self.generate_markdown_report(&results);
                    std::fs::write(file_path, markdown)
                        .map_err(|e| JitError::RuntimeError(format!("File write failed: {}", e)))?;
                }
                OutputFormat::Binary => {
                    // Updated for bincode v2: use encode_to_vec with default config
                    let binary =
                        bincode::serde::encode_to_vec(&*results, bincode::config::standard())
                            .map_err(|e| {
                                JitError::RuntimeError(format!(
                                    "Binary serialization failed: {}",
                                    e
                                ))
                            })?;
                    std::fs::write(file_path, binary)
                        .map_err(|e| JitError::RuntimeError(format!("File write failed: {}", e)))?;
                }
            }
        }

        Ok(())
    }

    fn generate_csv_report(&self, results: &BenchmarkResults) -> String {
        let mut csv = String::new();
        csv.push_str("Benchmark,Mean Time (μs),Median Time (μs),Min Time (μs),Max Time (μs),Std Dev (μs),Throughput (ops/s),Memory (MB)\n");

        for (name, result) in &results.results {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                name,
                result.statistics.mean_execution_time.as_micros(),
                result.statistics.median_execution_time.as_micros(),
                result.statistics.min_execution_time.as_micros(),
                result.statistics.max_execution_time.as_micros(),
                result.statistics.std_deviation.as_micros(),
                result.statistics.throughput_stats.mean_throughput,
                result.statistics.memory_stats.mean_peak_usage / 1024 / 1024
            ));
        }

        csv
    }

    fn generate_html_report(&self, results: &BenchmarkResults) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>{} - Benchmark Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #f9f9f9; padding: 15px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>{} - Benchmark Results</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Benchmarks: {}</p>
        <p>Successful: {}</p>
        <p>Failed: {}</p>
        <p>Total Execution Time: {:.2?}</p>
    </div>
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Benchmark</th>
            <th>Mean Time (μs)</th>
            <th>Throughput (ops/s)</th>
            <th>Memory (MB)</th>
            <th>Success Rate</th>
        </tr>
        {}
    </table>
</body>
</html>"#,
            results.config.suite_name,
            results.config.suite_name,
            results.suite_statistics.total_benchmarks,
            results.suite_statistics.successful_benchmarks,
            results.suite_statistics.failed_benchmarks,
            results.suite_statistics.total_execution_time,
            results
                .results
                .iter()
                .map(|(name, result)| format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{}</td><td>{:.1}%</td></tr>",
                    name,
                    result.statistics.mean_execution_time.as_micros(),
                    result.statistics.throughput_stats.mean_throughput,
                    result.statistics.memory_stats.mean_peak_usage / 1024 / 1024,
                    result.validation.success_rate * 100.0
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    fn generate_markdown_report(&self, results: &BenchmarkResults) -> String {
        let mut markdown = format!("# {} - Benchmark Results\n\n", results.config.suite_name);

        markdown.push_str("## Summary\n\n");
        markdown.push_str(&format!(
            "- **Total Benchmarks**: {}\n",
            results.suite_statistics.total_benchmarks
        ));
        markdown.push_str(&format!(
            "- **Successful**: {}\n",
            results.suite_statistics.successful_benchmarks
        ));
        markdown.push_str(&format!(
            "- **Failed**: {}\n",
            results.suite_statistics.failed_benchmarks
        ));
        markdown.push_str(&format!(
            "- **Total Execution Time**: {:.2?}\n\n",
            results.suite_statistics.total_execution_time
        ));

        markdown.push_str("## Detailed Results\n\n");
        markdown.push_str(
            "| Benchmark | Mean Time (μs) | Throughput (ops/s) | Memory (MB) | Success Rate |\n",
        );
        markdown.push_str(
            "|-----------|----------------|--------------------|--------------|--------------|\n",
        );

        for (name, result) in &results.results {
            markdown.push_str(&format!(
                "| {} | {} | {:.2} | {} | {:.1}% |\n",
                name,
                result.statistics.mean_execution_time.as_micros(),
                result.statistics.throughput_stats.mean_throughput,
                result.statistics.memory_stats.mean_peak_usage / 1024 / 1024,
                result.validation.success_rate * 100.0
            ));
        }

        markdown
    }
}

impl BenchmarkProfiler {
    pub fn new(enabled: bool) -> Self {
        Self {
            profiling_enabled: enabled,
            memory_tracker: MemoryTracker::new(),
            cpu_profiler: CpuProfiler::new(),
            energy_meter: None,
        }
    }

    pub fn start_profiling(&mut self) {
        if self.profiling_enabled {
            self.memory_tracker.reset();
            self.cpu_profiler.start();
            if let Some(ref mut meter) = self.energy_meter {
                meter.start();
            }
        }
    }

    pub fn stop_profiling(&mut self) -> ProfileData {
        ProfileData {
            memory_stats: self.memory_tracker.get_stats(),
            cpu_stats: self.cpu_profiler.get_stats(),
            energy_stats: self.energy_meter.as_ref().map(|m| m.get_stats()),
        }
    }
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            allocations: 0,
            deallocations: 0,
        }
    }

    pub fn reset(&mut self) {
        self.peak_usage = 0;
        self.current_usage = 0;
        self.allocations = 0;
        self.deallocations = 0;
    }

    pub fn get_stats(&self) -> MemoryStatistics {
        MemoryStatistics {
            peak_usage: self.peak_usage,
            average_usage: self.current_usage,
            allocations: self.allocations,
            deallocations: self.deallocations,
            leaks: if self.allocations > self.deallocations {
                self.allocations - self.deallocations
            } else {
                0
            },
            cache_stats: CacheStatistics {
                l1_hit_rate: 0.95,           // Placeholder
                l2_hit_rate: 0.80,           // Placeholder
                l3_hit_rate: 0.60,           // Placeholder
                cache_misses: 1000,          // Placeholder
                bandwidth_utilization: 0.70, // Placeholder
            },
        }
    }
}

impl CpuProfiler {
    pub fn new() -> Self {
        Self {
            sampling_rate: 1000, // 1ms
            profiles: Vec::new(),
        }
    }

    pub fn start(&mut self) {
        self.profiles.clear();
    }

    pub fn get_stats(&self) -> CpuStatistics {
        CpuStatistics {
            avg_usage: 0.75,            // Placeholder
            instruction_count: 1000000, // Placeholder
            cache_misses: 5000,         // Placeholder
        }
    }
}

impl EnergyMeter {
    pub fn start(&mut self) {
        self.baseline_power = self.current_power;
        self.total_energy = 0.0;
    }

    pub fn get_stats(&self) -> EnergyStatistics {
        EnergyStatistics {
            total_energy: self.total_energy,
            avg_power: self.current_power,
            peak_power: self.current_power * 1.2, // Placeholder
        }
    }
}

impl SystemInfo {
    pub fn collect() -> Self {
        Self {
            cpu_info: CpuInfo {
                model: "Unknown CPU".to_string(),
                cores: num_cpus::get(),
                frequency: 2400.0,                         // MHz placeholder
                cache_sizes: vec![32768, 262144, 8388608], // L1, L2, L3 placeholder
                features: vec!["SSE".to_string(), "AVX".to_string()],
            },
            memory_info: MemoryInfo {
                total: 8 * 1024 * 1024 * 1024,     // 8GB placeholder
                available: 6 * 1024 * 1024 * 1024, // 6GB placeholder
                page_size: 4096,
            },
            os_info: std::env::consts::OS.to_string(),
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            compiler_version: "1.0.0".to_string(),
            environment: std::env::vars().collect(),
        }
    }
}

/// Profile data collected during benchmarking
#[derive(Debug, Clone)]
pub struct ProfileData {
    pub memory_stats: MemoryStatistics,
    pub cpu_stats: CpuStatistics,
    pub energy_stats: Option<EnergyStatistics>,
}

/// CPU performance statistics
#[derive(Debug, Clone)]
pub struct CpuStatistics {
    pub avg_usage: f64,
    pub instruction_count: u64,
    pub cache_misses: u64,
}

/// Energy consumption statistics
#[derive(Debug, Clone)]
pub struct EnergyStatistics {
    pub total_energy: f64,
    pub avg_power: f64,
    pub peak_power: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = BenchmarkSuite::new(config);
        assert_eq!(suite.benchmarks.len(), 0);
    }

    #[test]
    fn test_statistics_calculation() {
        let suite = BenchmarkSuite::new(BenchmarkConfig::default());

        let measurements = vec![
            BenchmarkMeasurement {
                execution_time: Duration::from_millis(100),
                compilation_time: Duration::from_millis(10),
                memory_stats: MemoryStatistics {
                    peak_usage: 1024,
                    average_usage: 512,
                    allocations: 10,
                    deallocations: 8,
                    leaks: 2,
                    cache_stats: CacheStatistics {
                        l1_hit_rate: 0.95,
                        l2_hit_rate: 0.80,
                        l3_hit_rate: 0.60,
                        cache_misses: 100,
                        bandwidth_utilization: 0.70,
                    },
                },
                cpu_utilization: 0.8,
                throughput: 1000.0,
                energy_consumption: Some(10.0),
                custom_metrics: HashMap::new(),
                timestamp: SystemTime::now(),
                config_hash: 12345,
            },
            BenchmarkMeasurement {
                execution_time: Duration::from_millis(110),
                compilation_time: Duration::from_millis(12),
                memory_stats: MemoryStatistics {
                    peak_usage: 1100,
                    average_usage: 550,
                    allocations: 12,
                    deallocations: 10,
                    leaks: 2,
                    cache_stats: CacheStatistics {
                        l1_hit_rate: 0.96,
                        l2_hit_rate: 0.82,
                        l3_hit_rate: 0.62,
                        cache_misses: 95,
                        bandwidth_utilization: 0.72,
                    },
                },
                cpu_utilization: 0.85,
                throughput: 950.0,
                energy_consumption: Some(11.0),
                custom_metrics: HashMap::new(),
                timestamp: SystemTime::now(),
                config_hash: 12345,
            },
        ];

        let stats = suite.calculate_statistics(&measurements);
        assert_eq!(stats.mean_execution_time, Duration::from_millis(105));
        assert_eq!(stats.min_execution_time, Duration::from_millis(100));
        assert_eq!(stats.max_execution_time, Duration::from_millis(110));
    }

    #[test]
    fn test_system_info_collection() {
        let info = SystemInfo::collect();
        assert!(info.cpu_info.cores > 0);
        assert!(!info.os_info.is_empty());
    }
}
