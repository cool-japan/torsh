//! Benchmarking suite for ToRSh
//!
//! This crate provides comprehensive benchmarks for ToRSh to measure
//! performance against other tensor libraries and track regressions.

pub mod benchmarks;
pub mod comparisons;
pub mod metrics;
pub mod utils;

pub use benchmarks::*;
pub use comparisons::*;
pub use metrics::*;
pub use utils::*;

use criterion::{BenchmarkId, Criterion, Throughput};
use std::time::Duration;

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
#[derive(Debug, Clone)]
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

/// Trait for benchmarkable operations
pub trait Benchmarkable {
    type Input;
    type Output;

    /// Setup the benchmark with given input size
    fn setup(&mut self, size: usize) -> Self::Input;

    /// Run the benchmark operation
    fn run(&mut self, input: &Self::Input) -> Self::Output;

    /// Cleanup after benchmark
    fn cleanup(&mut self, input: Self::Input, output: Self::Output) {}

    /// Get the number of FLOPS for this operation
    fn flops(&self, size: usize) -> usize {
        size // Default: assume linear complexity
    }

    /// Get the number of bytes accessed for this operation
    fn bytes_accessed(&self, size: usize) -> usize {
        size * std::mem::size_of::<f32>() // Default: assume f32 elements
    }
}

/// Macro to create a simple benchmark
#[macro_export]
macro_rules! benchmark {
    ($name:expr, $setup:expr, $run:expr) => {
        struct SimpleBench<S, R> {
            setup_fn: S,
            run_fn: R,
        }

        impl<S, R, I, O> Benchmarkable for SimpleBench<S, R>
        where
            S: FnMut(usize) -> I,
            R: FnMut(&I) -> O,
        {
            type Input = I;
            type Output = O;

            fn setup(&mut self, size: usize) -> Self::Input {
                (self.setup_fn)(size)
            }

            fn run(&mut self, input: &Self::Input) -> Self::Output {
                (self.run_fn)(input)
            }
        }

        SimpleBench {
            setup_fn: $setup,
            run_fn: $run,
        }
    };
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
    pub use super::{benchmark, BenchConfig, BenchResult, BenchRunner, Benchmarkable};
    pub use criterion::{black_box, BenchmarkId, Criterion, Throughput};
}
