//! Benchmarking suite for ToRSh
//!
//! This crate provides comprehensive benchmarks for ToRSh to measure
//! performance against other tensor libraries and track regressions.
//!
//! # Architecture
//!
//! The benchmarking suite is organized into specialized modules:
//!
//! - **core**: Core comparison infrastructure and performance analysis
//! - **ndarray_comparisons**: ToRSh vs ndarray baseline comparisons
//! - **pytorch_comparisons**: PyTorch integration benchmarks (feature-gated)
//! - **tensorflow_comparisons**: TensorFlow comparison suite (feature-gated)
//! - **jax_comparisons**: JAX performance benchmarks (feature-gated)
//! - **numpy_comparisons**: NumPy baseline comparisons (feature-gated)
//! - **reporting**: Comprehensive reporting and analysis utilities
//!
//! Additional specialized benchmarks include model-specific tests, hardware
//! optimization benchmarks, edge deployment tests, and SciRS2 integration.

#![cfg_attr(
    not(any(feature = "tensorflow", feature = "pytorch", feature = "jax")),
    allow(unexpected_cfgs)
)]

// Core comparison infrastructure
pub mod core;

// Specialized comparison modules
pub mod ndarray_comparisons;
pub mod reporting;

// Feature-gated external library comparisons
#[cfg(feature = "pytorch")]
pub mod pytorch_comparisons;

#[cfg(feature = "tensorflow")]
pub mod tensorflow_comparisons;

#[cfg(feature = "jax")]
pub mod jax_comparisons;

#[cfg(feature = "numpy_baseline")]
pub mod numpy_comparisons;

// Legacy comparison module (now clean interface)
pub mod comparisons {
    //! Legacy comparison interface that delegates to specialized modules
    //!
    //! This module maintains backward compatibility while providing access
    //! to the new modular comparison architecture.

    // Re-export core infrastructure
    pub use crate::core::*;

    // Re-export ndarray comparisons (always available)
    pub use crate::ndarray_comparisons::*;

    // Re-export feature-gated comparisons
    #[cfg(feature = "pytorch")]
    pub use crate::pytorch_comparisons::*;

    #[cfg(feature = "tensorflow")]
    pub use crate::tensorflow_comparisons::*;

    #[cfg(feature = "jax")]
    pub use crate::jax_comparisons::*;

    #[cfg(feature = "numpy_baseline")]
    pub use crate::numpy_comparisons::*;

    // Re-export reporting utilities
    pub use crate::reporting::*;
}

// Specialized benchmark modules
pub mod advanced_analysis;
pub mod benchmark_analysis;
pub mod benchmark_cache;
pub mod benchmark_comparison;
pub mod benchmark_validation;
pub mod benchmarks;
pub mod cached_runner;
pub mod ci_integration;
pub mod custom_ops_benchmarks;
pub mod distributed_training;
pub mod edge_deployment;
pub mod hardware_benchmarks;
pub mod html_reporting;
pub mod metrics;
pub mod mobile_benchmarks;
pub mod model_benchmarks;
pub mod performance_dashboards;
pub mod precision_benchmarks;
pub mod regression_detection;
pub mod scalability;
pub mod scirs2_benchmarks;
pub mod system_info;
pub mod utils;
pub mod visualization;
pub mod wasm_benchmarks;

// Core benchmarks
pub use benchmarks::{
    AdvancedSystemsBenchmarkSuite, AutoTuningBench, BackwardPassBench, DataLoaderThroughputBench,
    ErrorDiagnosticsBench, GradientComputeBench, MatmulBench, SIMDGNNBench, TensorArithmeticBench,
    TensorCreationBench, VectorizedMetricsBench,
};

// Core comparison framework
pub use core::{ComparisonResult, ComparisonRunner, PerformanceAnalyzer};

// Reporting utilities
pub use reporting::{
    benchmark_and_analyze, benchmark_and_compare, generate_master_comparison_report,
    run_all_comparison_suites, run_comparison_benchmarks, run_extended_benchmarks,
};

// Metrics and utilities
pub use metrics::{CpuStats, MemoryStats, MetricsCollector, PerformanceReport, SystemMetrics};
pub use utils::{
    DataGenerator,
    Distribution,
    EnhancedBenchResult,
    EnhancedBenchSuite,
    Environment,
    EnvironmentInfo,
    Formatter,
    MemoryMonitor,
    ParallelBenchRunner,
    Timer,
    // Enhanced utilities
    TimingStats,
    ValidationResult,
    Validator,
};

// Model benchmarks
pub use model_benchmarks::{ModelBenchmarkSuite, ResNetBlockBench, TransformerBlockBench};

// Scalability and hardware tests
pub use hardware_benchmarks::{
    CPUGPUComparisonBench, MemoryBandwidthBench, MultiGPUBench, ThermalThrottlingBench,
};
pub use scalability::ScalabilityTestSuite;

// Precision and optimization benchmarks
pub use precision_benchmarks::{MixedPrecisionTrainingBench, PruningBench, QuantizationBench};

// Advanced benchmarks
pub use custom_ops_benchmarks::{ConvolutionOperation, CustomOpBench, FFTOperation};
pub use edge_deployment::{BatteryLifeBench, EdgeInferenceBench, EdgeMemoryBench};
pub use mobile_benchmarks::{ARMOptimizationBench, MobileGPUBench, MobilePlatformBench};
pub use wasm_benchmarks::{BrowserSpecificBench, WASMPerformanceBench, WebDeploymentBench};

// SciRS2 integration benchmarks
pub use scirs2_benchmarks::{
    AdvancedNeuralNetworkBench, AdvancedOptimizerBench, GraphNeuralNetworkBench,
    SciRS2BenchmarkSuite, SciRS2MathBench, SciRS2RandomBench, SpatialVisionBench,
    TimeSeriesAnalysisBench,
};

// Reporting and analysis
pub use advanced_analysis::{AdaptiveBenchmarking, AdvancedAnalyzer};
pub use benchmark_analysis::{
    BenchmarkAnalyzer, BottleneckAnalysis, PerformanceAnalysis, PerformanceRating,
};
pub use benchmark_validation::{BenchmarkValidator, NumericalAccuracy, ValidationConfig};
pub use ci_integration::{CIBenchmarkRunner, CIConfig, NotificationConfig};
pub use html_reporting::{HtmlReportGenerator, Theme};
pub use performance_dashboards::{DashboardConfig, PerformanceDashboard};
pub use regression_detection::{AdvancedRegressionDetector, RegressionAnalysis};
pub use system_info::{BenchmarkEnvironment, SystemInfo, SystemInfoCollector};
pub use visualization::{ChartType, VisualizationGenerator};

use criterion::{BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Trait for benchmarkable operations
pub trait Benchmarkable {
    type Input;
    type Output;

    /// Setup the benchmark input for a given size
    fn setup(&mut self, size: usize) -> Self::Input;

    /// Run the benchmark operation on the input
    fn run(&mut self, input: &Self::Input) -> Self::Output;

    /// Calculate the number of floating-point operations for a given size
    fn flops(&self, size: usize) -> usize {
        // Default implementation - can be overridden by specific benchmarks
        size
    }

    /// Calculate the number of bytes accessed for a given size
    fn bytes_accessed(&self, size: usize) -> usize {
        // Default implementation - can be overridden by specific benchmarks
        size * std::mem::size_of::<f32>()
    }
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Benchmark name
    pub name: String,

    /// Input sizes to test
    pub sizes: Vec<usize>,

    /// Data types to test
    pub dtypes: Vec<torsh_core::dtype::DType>,

    /// Number of warmup iterations
    pub warmup_time: Duration,

    /// Measurement time per size
    pub measurement_time: Duration,

    /// Whether to include memory usage metrics
    pub measure_memory: bool,

    /// Whether to include throughput metrics
    pub measure_throughput: bool,

    /// Custom metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            sizes: vec![64, 256, 1024, 4096],
            dtypes: vec![torsh_core::dtype::DType::F32],
            warmup_time: Duration::from_millis(100),
            measurement_time: Duration::from_secs(1),
            measure_memory: false,
            measure_throughput: true,
            metadata: std::collections::HashMap::new(),
        }
    }
}

impl BenchConfig {
    /// Create a new benchmark configuration
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Set input sizes
    pub fn with_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.sizes = sizes;
        self
    }

    /// Set data types
    pub fn with_dtypes(mut self, dtypes: Vec<torsh_core::dtype::DType>) -> Self {
        self.dtypes = dtypes;
        self
    }

    /// Enable memory measurement
    pub fn with_memory_measurement(mut self) -> Self {
        self.measure_memory = true;
        self
    }

    /// Set timing configuration
    pub fn with_timing(mut self, warmup: Duration, measurement: Duration) -> Self {
        self.warmup_time = warmup;
        self.measurement_time = measurement;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Benchmark result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchResult {
    /// Benchmark name
    pub name: String,

    /// Input size
    pub size: usize,

    /// Data type
    pub dtype: torsh_core::dtype::DType,

    /// Mean execution time in nanoseconds
    pub mean_time_ns: f64,

    /// Standard deviation of execution time
    pub std_dev_ns: f64,

    /// Throughput (operations per second)
    pub throughput: Option<f64>,

    /// Memory usage in bytes
    pub memory_usage: Option<usize>,

    /// Peak memory usage in bytes
    pub peak_memory: Option<usize>,

    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f64>,
}

impl BenchResult {
    /// Get throughput in GFLOPS
    pub fn gflops(&self, flops_per_op: usize) -> Option<f64> {
        self.throughput.map(|tps| tps * flops_per_op as f64 / 1e9)
    }

    /// Get memory bandwidth in GB/s
    pub fn memory_bandwidth_gbps(&self, bytes_per_op: usize) -> Option<f64> {
        self.throughput.map(|tps| tps * bytes_per_op as f64 / 1e9)
    }
}

/// Macro to create a simple benchmark
#[macro_export]
macro_rules! benchmark {
    ($name:expr, $setup:expr, $run:expr) => {{
        struct SimpleBench<S, R> {
            setup_fn: S,
            run_fn: R,
        }

        impl<S, R, I, O> $crate::Benchmarkable for SimpleBench<S, R>
        where
            S: FnMut(usize) -> I,
            R: FnMut(&I) -> O,
        {
            type Input = I;
            type Output = O;

            fn setup(&mut self, size: usize) -> <Self as $crate::Benchmarkable>::Input {
                (self.setup_fn)(size)
            }

            fn run(
                &mut self,
                input: &<Self as $crate::Benchmarkable>::Input,
            ) -> <Self as $crate::Benchmarkable>::Output {
                (self.run_fn)(input)
            }
        }

        SimpleBench {
            setup_fn: $setup,
            run_fn: $run,
        }
    }};
}

/// Benchmark runner
pub struct BenchRunner {
    criterion: Criterion,
    configs: Vec<BenchConfig>,
    results: Vec<BenchResult>,
}

impl BenchRunner {
    /// Create a new benchmark runner
    pub fn new() -> Self {
        Self {
            criterion: Criterion::default()
                .warm_up_time(Duration::from_millis(100))
                .measurement_time(Duration::from_secs(1)),
            configs: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Add a benchmark configuration
    pub fn add_config(mut self, config: BenchConfig) -> Self {
        self.configs.push(config);
        self
    }

    /// Run a benchmarkable operation
    pub fn run_benchmark<B: Benchmarkable>(&mut self, mut bench: B, config: &BenchConfig) {
        let mut group = self.criterion.benchmark_group(&config.name);
        group.warm_up_time(config.warmup_time);
        group.measurement_time(config.measurement_time);

        for &size in &config.sizes {
            for &dtype in &config.dtypes {
                let bench_id = BenchmarkId::new(format!("{}_{:?}", config.name, dtype), size);

                if config.measure_throughput {
                    let bytes_per_op = bench.bytes_accessed(size);
                    group.throughput(Throughput::Bytes(bytes_per_op as u64));
                }

                group.bench_with_input(bench_id, &size, |b, &size| {
                    let input = bench.setup(size);

                    b.iter(|| bench.run(&input));
                });
            }
        }

        group.finish();
    }

    /// Get benchmark results
    pub fn results(&self) -> &[BenchResult] {
        &self.results
    }

    /// Export results to CSV
    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(
            file,
            "name,size,dtype,mean_time_ns,std_dev_ns,throughput,memory_usage"
        )?;

        for result in &self.results {
            writeln!(
                file,
                "{},{},{:?},{},{},{:?},{:?}",
                result.name,
                result.size,
                result.dtype,
                result.mean_time_ns,
                result.std_dev_ns,
                result.throughput,
                result.memory_usage
            )?;
        }

        Ok(())
    }

    /// Generate HTML report
    pub fn generate_report(&self, output_dir: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(output_dir)?;

        let report_path = format!("{}/benchmark_report.html", output_dir);
        let mut file = std::fs::File::create(report_path)?;

        use std::io::Write;
        writeln!(file, "<!DOCTYPE html>")?;
        writeln!(
            file,
            "<html><head><title>ToRSh Benchmark Report</title></head><body>"
        )?;
        writeln!(file, "<h1>ToRSh Benchmark Report</h1>")?;

        // Add results table
        writeln!(file, "<table border='1'>")?;
        writeln!(file, "<tr><th>Benchmark</th><th>Size</th><th>Type</th><th>Time (μs)</th><th>Throughput</th></tr>")?;

        for result in &self.results {
            writeln!(
                file,
                "<tr><td>{}</td><td>{}</td><td>{:?}</td><td>{:.2}</td><td>{:.2}</td></tr>",
                result.name,
                result.size,
                result.dtype,
                result.mean_time_ns / 1000.0,
                result.throughput.unwrap_or(0.0)
            )?;
        }

        writeln!(file, "</table>")?;
        writeln!(file, "</body></html>")?;

        Ok(())
    }
}

impl Default for BenchRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Common benchmark utilities
pub mod prelude {
    pub use super::{benchmark, BenchConfig, BenchResult, BenchRunner};
    pub use super::{BenchmarkAnalyzer, SystemInfoCollector};
    pub use crate::benchmark_analysis::{BottleneckAnalysis, PerformanceAnalysis};
    pub use crate::core::{ComparisonResult, ComparisonRunner, PerformanceAnalyzer};
    pub use crate::system_info::{BenchmarkEnvironment, SystemInfo};
    pub use crate::Benchmarkable;
    pub use criterion::{black_box, BenchmarkId, Criterion, Throughput};

    // Enhanced utilities
    pub use crate::utils::{
        Distribution, EnhancedBenchResult, EnhancedBenchSuite, EnvironmentInfo, Formatter,
        MemoryMonitor, ParallelBenchRunner, Timer, TimingStats, ValidationResult, Validator,
    };
}
