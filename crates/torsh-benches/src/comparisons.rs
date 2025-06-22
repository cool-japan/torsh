//! Performance comparisons with other tensor libraries

use crate::{BenchConfig, BenchResult, Benchmarkable};
use criterion::black_box;

#[cfg(feature = "compare-external")]
use ndarray::{Array, Array2};
#[cfg(feature = "compare-external")]
use ndarray_rand::RandomExt;

/// Comparison benchmark runner
pub struct ComparisonRunner {
    results: Vec<ComparisonResult>,
}

impl ComparisonRunner {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Add comparison results
    pub fn add_result(&mut self, result: ComparisonResult) {
        self.results.push(result);
    }

    /// Get all comparison results
    pub fn results(&self) -> &[ComparisonResult] {
        &self.results
    }

    /// Generate comparison report
    pub fn generate_report(&self, output_path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(output_path)?;

        writeln!(file, "# ToRSh Performance Comparison Report\n")?;

        // Group results by operation
        let mut grouped: std::collections::HashMap<String, Vec<&ComparisonResult>> =
            std::collections::HashMap::new();

        for result in &self.results {
            grouped
                .entry(result.operation.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (operation, results) in grouped {
            writeln!(file, "## {}\n", operation)?;
            writeln!(file, "| Library | Size | Time (μs) | Speedup vs ToRSh |")?;
            writeln!(file, "|---------|------|-----------|------------------|")?;

            for result in &results {
                let speedup = if result.library == "torsh" {
                    1.0
                } else {
                    // Find corresponding ToRSh result
                    if let Some(torsh_result) = results
                        .iter()
                        .find(|r| r.library == "torsh" && r.size == result.size)
                    {
                        torsh_result.time_ns / result.time_ns
                    } else {
                        1.0
                    }
                };

                writeln!(
                    file,
                    "| {} | {} | {:.2} | {:.2}x |",
                    result.library,
                    result.size,
                    result.time_ns / 1000.0,
                    speedup
                )?;
            }
            writeln!(file)?;
        }

        Ok(())
    }
}

impl Default for ComparisonRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison result
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub operation: String,
    pub library: String,
    pub size: usize,
    pub time_ns: f64,
    pub throughput: Option<f64>,
    pub memory_usage: Option<usize>,
}

/// ToRSh matrix multiplication benchmark
pub struct TorshMatmulBench;

impl Benchmarkable for TorshMatmulBench {
    type Input = (torsh_tensor::Tensor<f32>, torsh_tensor::Tensor<f32>);
    type Output = Result<torsh_tensor::Tensor<f32>, torsh_core::error::TorshError>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let a = torsh_tensor::creation::rand::<f32>(&[size, size]);
        let b = torsh_tensor::creation::rand::<f32>(&[size, size]);
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        input.0.matmul(&input.1)
    }

    fn flops(&self, size: usize) -> usize {
        2 * size * size * size
    }
}

/// NDArray matrix multiplication benchmark (for comparison)
#[cfg(feature = "compare-external")]
pub struct NdarrayMatmulBench;

#[cfg(feature = "compare-external")]
impl Benchmarkable for NdarrayMatmulBench {
    type Input = (Array2<f32>, Array2<f32>);
    type Output = Array2<f32>;

    fn setup(&mut self, size: usize) -> Self::Input {
        use ndarray::Array;
        let a = Array::random((size, size), rand::distributions::Standard);
        let b = Array::random((size, size), rand::distributions::Standard);
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        black_box(input.0.dot(&input.1))
    }

    fn flops(&self, size: usize) -> usize {
        2 * size * size * size
    }
}

/// Element-wise operations comparison
pub struct TorshElementwiseBench;

impl Benchmarkable for TorshElementwiseBench {
    type Input = (torsh_tensor::Tensor<f32>, torsh_tensor::Tensor<f32>);
    type Output = Result<torsh_tensor::Tensor<f32>, torsh_core::error::TorshError>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let a = torsh_tensor::creation::rand::<f32>(&[size]);
        let b = torsh_tensor::creation::rand::<f32>(&[size]);
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        input.0.add(&input.1)
    }

    fn flops(&self, size: usize) -> usize {
        size
    }
}

#[cfg(feature = "compare-external")]
pub struct NdarrayElementwiseBench;

#[cfg(feature = "compare-external")]
impl Benchmarkable for NdarrayElementwiseBench {
    type Input = (
        Array<f32, ndarray::Dim<[usize; 1]>>,
        Array<f32, ndarray::Dim<[usize; 1]>>,
    );
    type Output = Array<f32, ndarray::Dim<[usize; 1]>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        use ndarray::Array;
        let a = Array::random(size, rand::distributions::Standard);
        let b = Array::random(size, rand::distributions::Standard);
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        black_box(&input.0 + &input.1)
    }

    fn flops(&self, size: usize) -> usize {
        size
    }
}

/// Run comparison benchmarks
pub fn run_comparison_benchmarks() -> ComparisonRunner {
    let mut runner = ComparisonRunner::new();

    let sizes = vec![64, 128, 256, 512, 1024];

    // Matrix multiplication comparisons
    for &size in &sizes {
        // ToRSh benchmark
        let mut torsh_bench = TorshMatmulBench;
        let input = torsh_bench.setup(size);

        let start = std::time::Instant::now();
        let _ = torsh_bench.run(&input);
        let torsh_time = start.elapsed().as_nanos() as f64;

        runner.add_result(ComparisonResult {
            operation: "matrix_multiplication".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(torsh_bench.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: None,
        });

        // NDArray benchmark
        #[cfg(feature = "compare-external")]
        {
            let mut ndarray_bench = NdarrayMatmulBench;
            let input = ndarray_bench.setup(size);

            let start = std::time::Instant::now();
            let _ = ndarray_bench.run(&input);
            let ndarray_time = start.elapsed().as_nanos() as f64;

            runner.add_result(ComparisonResult {
                operation: "matrix_multiplication".to_string(),
                library: "ndarray".to_string(),
                size,
                time_ns: ndarray_time,
                throughput: Some(ndarray_bench.flops(size) as f64 / ndarray_time * 1e9),
                memory_usage: None,
            });
        }
    }

    // Element-wise operation comparisons
    for &size in &sizes {
        // ToRSh benchmark
        let mut torsh_bench = TorshElementwiseBench;
        let input = torsh_bench.setup(size);

        let start = std::time::Instant::now();
        let _ = torsh_bench.run(&input);
        let torsh_time = start.elapsed().as_nanos() as f64;

        runner.add_result(ComparisonResult {
            operation: "elementwise_addition".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(torsh_bench.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: None,
        });

        // NDArray benchmark
        #[cfg(feature = "compare-external")]
        {
            let mut ndarray_bench = NdarrayElementwiseBench;
            let input = ndarray_bench.setup(size);

            let start = std::time::Instant::now();
            let _ = ndarray_bench.run(&input);
            let ndarray_time = start.elapsed().as_nanos() as f64;

            runner.add_result(ComparisonResult {
                operation: "elementwise_addition".to_string(),
                library: "ndarray".to_string(),
                size,
                time_ns: ndarray_time,
                throughput: Some(ndarray_bench.flops(size) as f64 / ndarray_time * 1e9),
                memory_usage: None,
            });
        }
    }

    runner
}

/// Benchmark all libraries and generate comparison report
pub fn benchmark_and_compare() -> std::io::Result<()> {
    let runner = run_comparison_benchmarks();
    runner.generate_report("target/comparison_report.md")?;

    println!("Comparison benchmarks completed!");
    println!("Results saved to target/comparison_report.md");

    Ok(())
}

/// Performance regression detection
pub struct RegressionDetector {
    baseline_results: Vec<BenchResult>,
    threshold: f64, // Performance degradation threshold (e.g., 0.1 = 10%)
}

impl RegressionDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            baseline_results: Vec::new(),
            threshold,
        }
    }

    /// Load baseline results from file
    pub fn load_baseline(&mut self, path: &str) -> std::io::Result<()> {
        // TODO: Implement loading from JSON/CSV
        Ok(())
    }

    /// Save current results as new baseline
    pub fn save_baseline(&self, results: &[BenchResult], path: &str) -> std::io::Result<()> {
        // TODO: Implement saving to JSON/CSV
        Ok(())
    }

    /// Check for performance regressions
    pub fn check_regression(&self, current_results: &[BenchResult]) -> Vec<RegressionResult> {
        let mut regressions = Vec::new();

        for current in current_results {
            if let Some(baseline) = self
                .baseline_results
                .iter()
                .find(|b| b.name == current.name && b.size == current.size)
            {
                let slowdown = current.mean_time_ns / baseline.mean_time_ns;
                if slowdown > (1.0 + self.threshold) {
                    regressions.push(RegressionResult {
                        benchmark: current.name.clone(),
                        size: current.size,
                        baseline_time: baseline.mean_time_ns,
                        current_time: current.mean_time_ns,
                        slowdown_factor: slowdown,
                        is_regression: true,
                    });
                }
            }
        }

        regressions
    }
}

/// Regression detection result
#[derive(Debug, Clone)]
pub struct RegressionResult {
    pub benchmark: String,
    pub size: usize,
    pub baseline_time: f64,
    pub current_time: f64,
    pub slowdown_factor: f64,
    pub is_regression: bool,
}
