//! JAX library comparisons
//!
//! This module provides benchmarks comparing ToRSh performance against JAX,
//! Google's high-performance machine learning research framework featuring
//! JIT compilation, automatic differentiation, and XLA optimization.

#![allow(deprecated)]

use crate::{core::ComparisonResult, Benchmarkable};

#[cfg(feature = "jax")]
use std::ffi::CString;

/// JAX benchmark runner with Python integration
///
/// Manages Python interpreter initialization and JAX availability detection,
/// supporting JIT compilation and XLA optimization with GPU acceleration.
pub struct JAXBenchRunner {
    python_initialized: bool,
    jax_available: bool,
    device: String,
}

#[cfg(feature = "jax")]
impl JAXBenchRunner {
    /// Create a new JAX benchmark runner
    pub fn new() -> Self {
        let mut runner = Self {
            python_initialized: false,
            jax_available: false,
            device: "cpu".to_string(),
        };

        if let Err(e) = runner.initialize_python() {
            eprintln!("Warning: Failed to initialize Python/JAX: {}", e);
        }

        runner
    }

    /// Initialize Python interpreter and check JAX availability
    fn initialize_python(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use pyo3::prelude::*;

        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| -> PyResult<()> {
            // Try to import JAX
            match py.import("jax") {
                Ok(jax) => {
                    self.jax_available = true;
                    self.python_initialized = true;

                    // Check available devices
                    let devices = jax.call_method0("devices")?;
                    let device_list: Vec<String> = devices.extract()?;

                    // Check for GPU devices
                    let has_gpu = device_list
                        .iter()
                        .any(|d| d.contains("gpu") || d.contains("GPU"));

                    if has_gpu {
                        self.device = "gpu".to_string();
                        println!("JAX GPU available, using GPU for benchmarks");
                    } else {
                        println!("JAX CPU only, using CPU for benchmarks");
                    }

                    Ok(())
                }
                Err(e) => {
                    eprintln!("JAX not available: {}", e);
                    Err(e)
                }
            }
        })?;

        Ok(())
    }

    /// Check if JAX is available
    pub fn is_jax_available(&self) -> bool {
        self.jax_available
    }

    /// Run JAX tensor operation benchmark with JIT compilation
    pub fn benchmark_jax_operation(
        &self,
        operation: &str,
        size: usize,
        iterations: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if !self.jax_available {
            return Err("JAX not available".into());
        }

        #[cfg(feature = "jax")]
        {
            use pyo3::prelude::*;

            Python::with_gil(|py| -> PyResult<f64> {
                let _jax = py.import("jax")?;
                let _jnp = py.import("jax.numpy")?;
                let _time_module = py.import("time")?;

                // Create the benchmark script as a string
                let benchmark_code = format!(
                    r#"
import jax
import jax.numpy as jnp
import time
import numpy as np

# Set up device
device = '{}'
if device == 'gpu':
    # Try to use GPU if available
    try:
        # This will fail if no GPU, falling back to CPU
        jax.device_put(jnp.array([1.0]), jax.devices('gpu')[0])
    except:
        device = 'cpu'

jax.config.update('jax_platform_name', device)
key = jax.random.PRNGKey(42)  # For reproducibility

def benchmark_operation():
    if '{}' == 'matmul':
        key1, key2 = jax.random.split(key)
        a = jax.random.normal(key1, ({}, {}))
        b = jax.random.normal(key2, ({}, {}))

        # Compile the function
        matmul_fn = jax.jit(jnp.matmul)

        # Warmup
        for _ in range(5):
            _ = matmul_fn(a, b).block_until_ready()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = matmul_fn(a, b)
            result.block_until_ready()  # Ensure computation is complete

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'add':
        key1, key2 = jax.random.split(key)
        a = jax.random.normal(key1, ({},))
        b = jax.random.normal(key2, ({},))

        # Compile the function
        add_fn = jax.jit(jnp.add)

        # Warmup
        for _ in range(10):
            _ = add_fn(a, b).block_until_ready()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = add_fn(a, b)
            result.block_until_ready()

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'conv2d':
        batch_size = 16
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        input_size = {}

        # Create input and kernel (JAX uses NHWC format)
        key1, key2 = jax.random.split(key)
        x = jax.random.normal(key1, (batch_size, input_size, input_size, in_channels))
        kernel = jax.random.normal(key2, (kernel_size, kernel_size, in_channels, out_channels))

        # Define convolution function
        def conv_fn(x, kernel):
            return jax.lax.conv_general_dilated(
                x, kernel,
                window_strides=(1, 1),
                padding='SAME',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC')
            )

        # Compile the function
        compiled_conv = jax.jit(conv_fn)

        # Warmup
        for _ in range(5):
            _ = compiled_conv(x, kernel).block_until_ready()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = compiled_conv(x, kernel)
            result.block_until_ready()

        end_time = time.time()

        return (end_time - start_time) / {}

    elif '{}' == 'relu':
        key1 = jax.random.split(key)[0]
        x = jax.random.normal(key1, ({},))

        # Compile the function
        relu_fn = jax.jit(jnp.maximum, static_argnums=())

        # Warmup
        for _ in range(10):
            _ = relu_fn(x, 0.0).block_until_ready()

        # Benchmark
        start_time = time.time()

        for _ in range({}):
            result = relu_fn(x, 0.0)
            result.block_until_ready()

        end_time = time.time()

        return (end_time - start_time) / {}

    else:
        raise ValueError(f"Unknown operation: {}")

avg_time = benchmark_operation()
avg_time
"#,
                    self.device,
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
                let code_cstr = CString::new(benchmark_code)?;
                let result: f64 = py.eval(&code_cstr, None, None)?.extract()?;
                Ok(result)
            })
            .map_err(|e| e.into())
        }

        #[cfg(not(feature = "jax"))]
        {
            let _ = (operation, size, iterations);
            Err("JAX feature not enabled".into())
        }
    }
}

#[cfg(not(feature = "jax"))]
impl JAXBenchRunner {
    pub fn new() -> Self {
        Self {
            python_initialized: false,
            jax_available: false,
            device: "cpu".to_string(),
        }
    }

    pub fn is_jax_available(&self) -> bool {
        false
    }

    pub fn benchmark_jax_operation(
        &self,
        _operation: &str,
        _size: usize,
        _iterations: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        Err("JAX feature not enabled".into())
    }
}

#[cfg(feature = "jax")]
impl Default for JAXBenchRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// ToRSh vs JAX matrix multiplication benchmark
///
/// Compares matrix multiplication performance between ToRSh's Rust implementation
/// and JAX's JIT-compiled XLA kernels with GPU acceleration support.
pub struct TorshVsJAXMatmul {
    jax_runner: JAXBenchRunner,
}

impl TorshVsJAXMatmul {
    /// Create a new matrix multiplication comparison benchmark
    pub fn new() -> Self {
        Self {
            jax_runner: JAXBenchRunner::new(),
        }
    }
}

impl Benchmarkable for TorshVsJAXMatmul {
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

        // Benchmark JAX
        if self.jax_runner.is_jax_available() {
            match self
                .jax_runner
                .benchmark_jax_operation("matmul", size, iterations)
            {
                Ok(jax_time_seconds) => {
                    let jax_time_ns = jax_time_seconds * 1e9;
                    results.push(ComparisonResult {
                        operation: "matrix_multiplication".to_string(),
                        library: "jax".to_string(),
                        size,
                        time_ns: jax_time_ns,
                        throughput: Some(self.flops(size) as f64 / jax_time_ns * 1e9),
                        memory_usage: Some(size * size * 4 * 3),
                    });
                }
                Err(e) => {
                    eprintln!("JAX benchmark failed: {:?}", e);
                }
            }
        }

        results
    }

    fn flops(&self, size: usize) -> usize {
        2 * size * size * size // Matrix multiplication FLOPS
    }
}

/// ToRSh vs JAX element-wise operations benchmark
///
/// Compares element-wise addition performance between ToRSh and JAX,
/// focusing on JIT compilation benefits and vectorization efficiency.
pub struct TorshVsJAXElementwise {
    jax_runner: JAXBenchRunner,
}

impl TorshVsJAXElementwise {
    /// Create a new element-wise operations comparison benchmark
    pub fn new() -> Self {
        Self {
            jax_runner: JAXBenchRunner::new(),
        }
    }
}

impl Benchmarkable for TorshVsJAXElementwise {
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

        // Benchmark JAX
        if self.jax_runner.is_jax_available() {
            match self
                .jax_runner
                .benchmark_jax_operation("add", size, iterations)
            {
                Ok(jax_time_seconds) => {
                    let jax_time_ns = jax_time_seconds * 1e9;
                    results.push(ComparisonResult {
                        operation: "elementwise_addition".to_string(),
                        library: "jax".to_string(),
                        size,
                        time_ns: jax_time_ns,
                        throughput: Some(self.flops(size) as f64 / jax_time_ns * 1e9),
                        memory_usage: Some(size * 4 * 3),
                    });
                }
                Err(e) => {
                    eprintln!("JAX benchmark failed: {:?}", e);
                }
            }
        }

        results
    }

    fn flops(&self, size: usize) -> usize {
        size // Element-wise addition FLOPS
    }
}

/// Comprehensive JAX comparison suite
///
/// Executes full benchmark suite comparing ToRSh and JAX performance
/// across matrix operations and element-wise operations with JIT compilation.
pub fn run_jax_comparison_suite() -> crate::core::ComparisonRunner {
    let mut runner = crate::core::ComparisonRunner::new();

    println!("🚀 Running comprehensive ToRSh vs JAX benchmarks...");

    let sizes = vec![64, 128, 256, 512];

    // Matrix multiplication benchmarks
    println!("📊 Benchmarking matrix multiplication...");
    let mut matmul_bench = TorshVsJAXMatmul::new();
    for &size in &sizes {
        let input = matmul_bench.setup(size);
        let results = matmul_bench.run(&input);
        for result in results {
            runner.add_result(result);
        }
    }

    // Element-wise operation benchmarks
    println!("📊 Benchmarking element-wise operations...");
    let mut elementwise_bench = TorshVsJAXElementwise::new();
    for &size in &[1000, 10000, 100000, 1000000] {
        let input = elementwise_bench.setup(size);
        let results = elementwise_bench.run(&input);
        for result in results {
            runner.add_result(result);
        }
    }

    println!("✅ JAX comparison benchmarks completed!");
    runner
}

/// Generate comprehensive JAX performance report
///
/// Creates detailed analysis and reports comparing ToRSh and JAX performance
/// with focus on JIT compilation benefits and XLA optimization insights.
pub fn generate_jax_comparison_report() -> std::io::Result<()> {
    let runner = run_jax_comparison_suite();

    // Generate markdown report
    runner.generate_report("target/jax_comparison.md")?;

    // Generate detailed analysis
    let mut analyzer = crate::core::PerformanceAnalyzer::new();
    analyzer.add_results(runner.results());

    // Create comprehensive performance analysis
    let mut analysis_file = std::fs::File::create("target/jax_vs_torsh_analysis.md")?;
    use std::io::Write;

    writeln!(analysis_file, "# ToRSh vs JAX Performance Analysis\n")?;
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
    writeln!(analysis_file, "This benchmark compares ToRSh (pure Rust tensor library) with JAX (Python/XLA tensor library).")?;
    writeln!(
        analysis_file,
        "Results include both CPU and GPU measurements where available.\n"
    )?;

    writeln!(analysis_file, "### Key Findings:\n")?;
    writeln!(
        analysis_file,
        "- **Matrix Multiplication**: Core linear algebra performance comparison"
    )?;
    writeln!(
        analysis_file,
        "- **Element-wise Operations**: Memory bandwidth and vectorization efficiency"
    )?;
    writeln!(analysis_file, "\n### Notes:\n")?;
    writeln!(
        analysis_file,
        "- All benchmarks run with consistent random seeds for reproducibility"
    )?;
    writeln!(
        analysis_file,
        "- JAX uses JIT compilation and XLA optimization, with CUDA when available"
    )?;
    writeln!(
        analysis_file,
        "- ToRSh uses pure Rust implementations with SIMD optimizations"
    )?;

    println!("📈 Comprehensive JAX comparison report generated!");
    println!("   📄 Basic report: target/jax_comparison.md");
    println!("   📊 Detailed analysis: target/jax_vs_torsh_analysis.md");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jax_bench_runner() {
        let runner = JAXBenchRunner::new();
        // JAX availability depends on environment
        // Just test that runner can be created
        assert!(runner.device == "cpu" || runner.device == "gpu");
    }

    #[test]
    fn test_torsh_vs_jax_matmul() {
        let mut bench = TorshVsJAXMatmul::new();
        let input = bench.setup(4);
        let results = bench.run(&input);

        // Should always have ToRSh results
        assert!(!results.is_empty());
        let torsh_result = results.iter().find(|r| r.library == "torsh");
        assert!(torsh_result.is_some());

        assert_eq!(bench.flops(4), 2 * 4 * 4 * 4);
    }

    #[test]
    fn test_torsh_vs_jax_elementwise() {
        let mut bench = TorshVsJAXElementwise::new();
        let input = bench.setup(100);
        let results = bench.run(&input);

        // Should always have ToRSh results
        assert!(!results.is_empty());
        let torsh_result = results.iter().find(|r| r.library == "torsh");
        assert!(torsh_result.is_some());

        assert_eq!(bench.flops(100), 100);
    }

    #[test]
    fn test_jax_comparison_suite() {
        let runner = run_jax_comparison_suite();
        assert!(!runner.results().is_empty());

        // Should have ToRSh results at minimum
        let torsh_results: Vec<_> = runner
            .results()
            .iter()
            .filter(|r| r.library == "torsh")
            .collect();
        assert!(!torsh_results.is_empty());

        // Check operation types
        let operations: std::collections::HashSet<_> =
            runner.results().iter().map(|r| &r.operation).collect();
        assert!(operations.contains(&"matrix_multiplication".to_string()));
        assert!(operations.contains(&"elementwise_addition".to_string()));
    }
}
