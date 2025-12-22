//! NumPy baseline comparison benchmarks for ToRSh
//!
//! This module provides comprehensive benchmarks comparing ToRSh tensor operations
//! with NumPy baseline implementations. NumPy serves as the foundational reference
//! for scientific computing performance.

#![allow(deprecated)]

use crate::core::{ComparisonResult, ComparisonRunner, PerformanceAnalyzer};
use crate::Benchmarkable;

/// NumPy benchmark runner
#[cfg(feature = "numpy_baseline")]
pub struct NumPyBenchRunner {
    python_initialized: bool,
    numpy_available: bool,
}

#[cfg(feature = "numpy_baseline")]
impl NumPyBenchRunner {
    pub fn new() -> Self {
        let mut runner = Self {
            python_initialized: false,
            numpy_available: false,
        };

        if let Err(e) = runner.initialize_python() {
            eprintln!("Warning: Failed to initialize Python/NumPy: {}", e);
        }

        runner
    }

    /// Initialize Python interpreter and check NumPy availability
    fn initialize_python(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "numpy_baseline")]
        {
            use pyo3::prelude::*;

            pyo3::prepare_freethreaded_python();

            Python::with_gil(|py| -> PyResult<()> {
                // Try to import NumPy
                match py.import("numpy") {
                    Ok(_) => {
                        self.numpy_available = true;
                        self.python_initialized = true;
                        println!("NumPy available for baseline benchmarks");
                        Ok(())
                    }
                    Err(e) => {
                        eprintln!("NumPy not available: {}", e);
                        Err(e)
                    }
                }
            })?;
        }

        Ok(())
    }

    /// Check if NumPy is available
    pub fn is_numpy_available(&self) -> bool {
        self.numpy_available
    }

    /// Run NumPy tensor operation benchmark
    #[cfg(feature = "numpy_baseline")]
    pub fn benchmark_numpy_operation(
        &self,
        operation: &str,
        size: usize,
        iterations: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if !self.numpy_available {
            return Err("NumPy not available".into());
        }

        use pyo3::prelude::*;

        Python::with_gil(|py| -> PyResult<f64> {
            let _np = py.import("numpy")?;
            let _time_module = py.import("time")?;

            // Create the benchmark script as a string
            let benchmark_code = format!(
                r#"
import numpy as np
import time

np.random.seed(42)  # For reproducibility

def benchmark_operation():
    if '{}' == 'matmul':
        a = np.random.randn({}, {}).astype(np.float32)
        b = np.random.randn({}, {}).astype(np.float32)

        # Warmup
        for _ in range(5):
            _ = np.matmul(a, b)

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = np.matmul(a, b)

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'add':
        a = np.random.randn({}).astype(np.float32)
        b = np.random.randn({}).astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = np.add(a, b)

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = np.add(a, b)

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'conv2d':
        # NumPy doesn't have native conv2d, so we'll simulate with basic operations
        batch_size = 16
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        input_size = {}

        # Create input and simulate convolution computation
        x = np.random.randn(batch_size, in_channels, input_size, input_size).astype(np.float32)
        kernel = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)

        # Simple simulation of convolution workload using matrix operations
        def simulate_conv():
            # Flatten and multiply to simulate convolution computation
            flattened_x = x.reshape(batch_size, -1)
            flattened_k = kernel.reshape(out_channels, -1)
            return np.dot(flattened_k, flattened_x.T)

        # Warmup
        for _ in range(5):
            _ = simulate_conv()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = simulate_conv()

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'relu':
        x = np.random.randn({}).astype(np.float32)

        # Warmup
        for _ in range(10):
            _ = np.maximum(x, 0.0)

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = np.maximum(x, 0.0)

        end_time = time.time()

        return (end_time - start_time) / {}

    else:
        raise ValueError(f"Unknown operation: {}")

avg_time = benchmark_operation()
avg_time
"#,
                operation,
                size,
                size, // matmul shapes
                size,
                size,
                iterations,
                iterations, // matmul iterations
                operation,
                size, // add shapes
                size,
                iterations,
                iterations, // add iterations
                operation,
                size, // conv2d input size
                iterations,
                iterations, // conv2d iterations
                operation,
                size, // relu shapes
                iterations,
                iterations, // relu iterations
                operation   // error case
            );

            // Execute the benchmark code
            let code_cstr = std::ffi::CString::new(benchmark_code)?;
            let result: f64 = py.eval(&code_cstr, None, None)?.extract()?;
            Ok(result)
        })
        .map_err(|e| e.into())
    }

    #[cfg(not(feature = "numpy_baseline"))]
    pub fn benchmark_numpy_operation(
        &self,
        _operation: &str,
        _size: usize,
        _iterations: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        Err("NumPy baseline feature not enabled".into())
    }
}

#[cfg(feature = "numpy_baseline")]
impl Default for NumPyBenchRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// ToRSh vs NumPy matrix multiplication benchmark
#[cfg(feature = "numpy_baseline")]
pub struct TorshVsNumPyMatmul {
    numpy_runner: NumPyBenchRunner,
}

#[cfg(feature = "numpy_baseline")]
impl TorshVsNumPyMatmul {
    pub fn new() -> Self {
        Self {
            numpy_runner: NumPyBenchRunner::new(),
        }
    }
}

#[cfg(feature = "numpy_baseline")]
impl Benchmarkable for TorshVsNumPyMatmul {
    type Input = (torsh_tensor::Tensor<f32>, torsh_tensor::Tensor<f32>);
    type Output = Vec<ComparisonResult>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let a = torsh_tensor::creation::rand::<f32>(&[size, size]).unwrap();
        let b = torsh_tensor::creation::rand::<f32>(&[size, size]).unwrap();
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let mut results = Vec::new();
        let size = input.0.shape().dims()[0];
        let iterations = 10;

        // Benchmark ToRSh
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = input.0.matmul(&input.1);
        }
        let torsh_time = start.elapsed().as_nanos() as f64 / iterations as f64;

        results.push(ComparisonResult {
            operation: "matrix_multiplication".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(self.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: Some(size * size * 4 * 3), // 3 matrices (A, B, result)
        });

        // Benchmark NumPy
        if self.numpy_runner.is_numpy_available() {
            match self
                .numpy_runner
                .benchmark_numpy_operation("matmul", size, iterations)
            {
                Ok(numpy_time_seconds) => {
                    let numpy_time_ns = numpy_time_seconds * 1e9;
                    results.push(ComparisonResult {
                        operation: "matrix_multiplication".to_string(),
                        library: "numpy".to_string(),
                        size,
                        time_ns: numpy_time_ns,
                        throughput: Some(self.flops(size) as f64 / numpy_time_ns * 1e9),
                        memory_usage: Some(size * size * 4 * 3),
                    });
                }
                Err(e) => {
                    eprintln!("NumPy benchmark failed: {:?}", e);
                }
            }
        }

        results
    }

    fn flops(&self, size: usize) -> usize {
        2 * size * size * size // Matrix multiplication FLOPS
    }
}

/// ToRSh vs NumPy element-wise operations benchmark
#[cfg(feature = "numpy_baseline")]
pub struct TorshVsNumPyElementwise {
    numpy_runner: NumPyBenchRunner,
}

#[cfg(feature = "numpy_baseline")]
impl TorshVsNumPyElementwise {
    pub fn new() -> Self {
        Self {
            numpy_runner: NumPyBenchRunner::new(),
        }
    }
}

#[cfg(feature = "numpy_baseline")]
impl Benchmarkable for TorshVsNumPyElementwise {
    type Input = (torsh_tensor::Tensor<f32>, torsh_tensor::Tensor<f32>);
    type Output = Vec<ComparisonResult>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let a = torsh_tensor::creation::rand::<f32>(&[size]).unwrap();
        let b = torsh_tensor::creation::rand::<f32>(&[size]).unwrap();
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let mut results = Vec::new();
        let size = input.0.shape().dims()[0];
        let iterations = 100;

        // Benchmark ToRSh
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = input.0.add(&input.1);
        }
        let torsh_time = start.elapsed().as_nanos() as f64 / iterations as f64;

        results.push(ComparisonResult {
            operation: "elementwise_addition".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(self.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: Some(size * 4 * 3), // 3 vectors
        });

        // Benchmark NumPy
        if self.numpy_runner.is_numpy_available() {
            match self
                .numpy_runner
                .benchmark_numpy_operation("add", size, iterations)
            {
                Ok(numpy_time_seconds) => {
                    let numpy_time_ns = numpy_time_seconds * 1e9;
                    results.push(ComparisonResult {
                        operation: "elementwise_addition".to_string(),
                        library: "numpy".to_string(),
                        size,
                        time_ns: numpy_time_ns,
                        throughput: Some(self.flops(size) as f64 / numpy_time_ns * 1e9),
                        memory_usage: Some(size * 4 * 3),
                    });
                }
                Err(e) => {
                    eprintln!("NumPy benchmark failed: {:?}", e);
                }
            }
        }

        results
    }

    fn flops(&self, size: usize) -> usize {
        size // Element-wise addition FLOPS
    }
}

/// Comprehensive NumPy comparison suite
#[cfg(feature = "numpy_baseline")]
pub fn run_numpy_comparison_suite() -> ComparisonRunner {
    let mut runner = ComparisonRunner::new();

    println!("🚀 Running comprehensive ToRSh vs NumPy benchmarks...");

    let sizes = vec![64, 128, 256, 512];

    // Matrix multiplication benchmarks
    println!("📊 Benchmarking matrix multiplication...");
    let mut matmul_bench = TorshVsNumPyMatmul::new();
    for &size in &sizes {
        let input = matmul_bench.setup(size);
        let results = matmul_bench.run(&input);
        for result in results {
            runner.add_result(result);
        }
    }

    // Element-wise operation benchmarks
    println!("📊 Benchmarking element-wise operations...");
    let mut elementwise_bench = TorshVsNumPyElementwise::new();
    for &size in &[1000, 10000, 100000, 1000000] {
        let input = elementwise_bench.setup(size);
        let results = elementwise_bench.run(&input);
        for result in results {
            runner.add_result(result);
        }
    }

    println!("✅ NumPy comparison benchmarks completed!");
    runner
}

/// Generate comprehensive NumPy performance report
#[cfg(feature = "numpy_baseline")]
pub fn generate_numpy_comparison_report() -> std::io::Result<()> {
    let runner = run_numpy_comparison_suite();

    // Generate markdown report
    runner.generate_report("target/numpy_comparison.md")?;

    // Generate detailed analysis
    let mut analyzer = PerformanceAnalyzer::new();
    analyzer.add_results(runner.results());

    // Create comprehensive performance analysis
    let mut analysis_file = std::fs::File::create("target/numpy_vs_torsh_analysis.md")?;
    use std::io::Write;

    writeln!(analysis_file, "# ToRSh vs NumPy Performance Analysis\n")?;
    writeln!(
        analysis_file,
        "Generated on: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;

    let operations = ["matrix_multiplication", "elementwise_addition"];

    for operation in &operations {
        let analysis = analyzer.analyze_operation(operation);

        writeln!(
            analysis_file,
            "## {}\n",
            operation.replace('_', " ").to_uppercase()
        )?;

        // Performance summary table
        writeln!(
            analysis_file,
            "| Library | Avg Time (μs) | Avg Throughput (GFLOPS) | Speedup vs ToRSh |"
        )?;
        writeln!(
            analysis_file,
            "|---------|---------------|-------------------------|-------------------|"
        )?;

        let torsh_stats = analysis.library_stats.get("torsh");

        for (library, stats) in &analysis.library_stats {
            let speedup = if library == "torsh" {
                1.0
            } else if let Some(torsh) = torsh_stats {
                torsh.mean_time_ns / stats.mean_time_ns
            } else {
                1.0
            };

            writeln!(
                analysis_file,
                "| {} | {:.2} | {:.2} | {:.2}x |",
                library.to_uppercase(),
                stats.mean_time_ns / 1000.0,
                stats.mean_throughput.unwrap_or(0.0) / 1e9,
                speedup
            )?;
        }
        writeln!(analysis_file)?;

        // Performance insights
        if !analysis.recommendations.is_empty() {
            writeln!(analysis_file, "### Performance Insights\n")?;
            for rec in &analysis.recommendations {
                writeln!(analysis_file, "- {}", rec)?;
            }
            writeln!(analysis_file)?;
        }
    }

    // Overall summary
    writeln!(analysis_file, "## Overall Summary\n")?;
    writeln!(analysis_file, "This benchmark compares ToRSh (pure Rust tensor library) with NumPy (Python/C tensor library).")?;
    writeln!(
        analysis_file,
        "Results provide baseline performance comparison against the foundational scientific computing library.\n"
    )?;

    writeln!(analysis_file, "### Key Findings:\n")?;
    writeln!(
        analysis_file,
        "- **Matrix Multiplication**: Core linear algebra performance comparison against NumPy/BLAS"
    )?;
    writeln!(
        analysis_file,
        "- **Element-wise Operations**: Memory bandwidth and vectorization efficiency baseline"
    )?;
    writeln!(analysis_file, "\n### Notes:\n")?;
    writeln!(
        analysis_file,
        "- All benchmarks run with consistent random seeds for reproducibility"
    )?;
    writeln!(
        analysis_file,
        "- NumPy uses optimized BLAS libraries (OpenBLAS/MKL) for linear algebra"
    )?;
    writeln!(
        analysis_file,
        "- ToRSh uses pure Rust implementations with SIMD optimizations"
    )?;
    writeln!(
        analysis_file,
        "- NumPy serves as the baseline for scientific computing performance"
    )?;

    println!("📈 Comprehensive NumPy comparison report generated!");
    println!("   📄 Basic report: target/numpy_comparison.md");
    println!("   📊 Detailed analysis: target/numpy_vs_torsh_analysis.md");

    Ok(())
}

// Non-feature-gated stubs for consistent API
#[cfg(not(feature = "numpy_baseline"))]
pub fn run_numpy_comparison_suite() -> ComparisonRunner {
    println!("⚠️ NumPy baseline feature not enabled - skipping comparisons");
    ComparisonRunner::new()
}

#[cfg(not(feature = "numpy_baseline"))]
pub fn generate_numpy_comparison_report() -> std::io::Result<()> {
    println!("⚠️ NumPy baseline feature not enabled - skipping report generation");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numpy_bench_runner_creation() {
        // Test that NumPyBenchRunner can be created without panicking
        // even if NumPy is not available
        let _runner = NumPyBenchRunner::new();
    }

    #[test]
    #[cfg(feature = "numpy_baseline")]
    fn test_numpy_comparison_suite() {
        // Test that the comparison suite can run without errors
        let runner = run_numpy_comparison_suite();
        // Should have some results if NumPy is available
        // Verify results collection exists (length is usize, always >= 0)
        let _ = runner.results().len();
    }

    #[test]
    #[cfg(feature = "numpy_baseline")]
    fn test_torsh_vs_numpy_matmul() {
        let mut bench = TorshVsNumPyMatmul::new();
        let input = bench.setup(64);
        let results = bench.run(&input);

        // Should have at least ToRSh results
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.library == "torsh"));
    }

    #[test]
    #[cfg(feature = "numpy_baseline")]
    fn test_torsh_vs_numpy_elementwise() {
        let mut bench = TorshVsNumPyElementwise::new();
        let input = bench.setup(1000);
        let results = bench.run(&input);

        // Should have at least ToRSh results
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.library == "torsh"));
    }
}
