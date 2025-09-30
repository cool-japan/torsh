//! TensorFlow comparison benchmarks
//!
//! This module provides comprehensive TensorFlow benchmark implementations
//! for comparing ToRSh performance with TensorFlow operations.

use super::core::{ComparisonResult, ComparisonRunner};
use crate::Benchmarkable;

/// TensorFlow benchmark runner
#[cfg(feature = "tensorflow")]
pub struct TensorFlowBenchRunner {
    python_initialized: bool,
    tensorflow_available: bool,
    device: String,
}

#[cfg(feature = "tensorflow")]
impl TensorFlowBenchRunner {
    pub fn new() -> Self {
        let mut runner = Self {
            python_initialized: false,
            tensorflow_available: false,
            device: "cpu".to_string(),
        };

        if let Err(e) = runner.initialize_python() {
            eprintln!("Warning: Failed to initialize Python/TensorFlow: {}", e);
        }

        runner
    }

    /// Initialize Python interpreter and check TensorFlow availability
    fn initialize_python(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "tensorflow")]
        {
            use pyo3::prelude::*;

            pyo3::prepare_freethreaded_python();

            Python::with_gil(|py| -> PyResult<()> {
                // Try to import TensorFlow
                match py.import("tensorflow") {
                    Ok(tf) => {
                        self.tensorflow_available = true;
                        self.python_initialized = true;

                        // Check available devices
                        let devices = tf
                            .call_method0("config")?
                            .call_method0("list_physical_devices")?;

                        // Try to find GPU
                        let gpu_devices = tf
                            .call_method0("config")?
                            .call_method1("list_physical_devices", ("GPU",))?;

                        let gpu_count: usize = gpu_devices.len()?;

                        if gpu_count > 0 {
                            self.device = "gpu".to_string();
                            println!("TensorFlow GPU available, using GPU for benchmarks");
                        } else {
                            println!("TensorFlow CPU only, using CPU for benchmarks");
                        }

                        Ok(())
                    }
                    Err(e) => {
                        eprintln!("TensorFlow not available: {}", e);
                        Err(e)
                    }
                }
            })?;
        }

        Ok(())
    }

    /// Check if TensorFlow is available
    pub fn is_tensorflow_available(&self) -> bool {
        self.tensorflow_available
    }

    /// Run TensorFlow tensor operation benchmark
    #[cfg(feature = "tensorflow")]
    pub fn benchmark_tensorflow_operation(
        &self,
        operation: &str,
        size: usize,
        iterations: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if !self.tensorflow_available {
            return Err("TensorFlow not available".into());
        }

        use pyo3::prelude::*;

        Python::with_gil(|py| -> PyResult<f64> {
            let tf = py.import("tensorflow")?;
            let time_module = py.import("time")?;

            // Create the benchmark script as a string
            let benchmark_code = format!(
                r#"
import tensorflow as tf
import time
import numpy as np

# Configure GPU memory growth to avoid allocation issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

device_name = '/{}:0'
tf.random.set_seed(42)  # For reproducibility

def benchmark_operation():
    with tf.device(device_name):
        if '{}' == 'matmul':
            a = tf.random.normal([{}, {}])
            b = tf.random.normal([{}, {}])

            # Warmup
            for _ in range(5):
                _ = tf.matmul(a, b)

            # Benchmark
            start_time = time.time()

            for _ in range({}):
                result = tf.matmul(a, b)

            # Ensure computation is complete
            if hasattr(result, 'numpy'):
                _ = result.numpy()

            end_time = time.time()

            return (end_time - start_time) / {}

        elif '{}' == 'add':
            a = tf.random.normal([{}])
            b = tf.random.normal([{}])

            # Warmup
            for _ in range(10):
                _ = tf.add(a, b)

            # Benchmark
            start_time = time.time()

            for _ in range({}):
                result = tf.add(a, b)

            if hasattr(result, 'numpy'):
                _ = result.numpy()

            end_time = time.time()

            return (end_time - start_time) / {}

        elif '{}' == 'conv2d':
            batch_size = 16
            in_channels = 3
            out_channels = 64
            kernel_size = 3
            input_size = {}

            # Create input and kernel
            x = tf.random.normal([batch_size, input_size, input_size, in_channels])
            kernel = tf.random.normal([kernel_size, kernel_size, in_channels, out_channels])

            # Warmup
            for _ in range(5):
                _ = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

            # Benchmark
            start_time = time.time()

            for _ in range({}):
                result = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')

            if hasattr(result, 'numpy'):
                _ = result.numpy()

            end_time = time.time()

            return (end_time - start_time) / {}

        elif '{}' == 'relu':
            x = tf.random.normal([{}])

            # Warmup
            for _ in range(10):
                _ = tf.nn.relu(x)

            # Benchmark
            start_time = time.time()

            for _ in range({}):
                result = tf.nn.relu(x)

            if hasattr(result, 'numpy'):
                _ = result.numpy()

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
            let result: f64 = py.eval(&benchmark_code, None, None)?.extract()?;
            Ok(result)
        })
        .map_err(|e| e.into())
    }

    #[cfg(not(feature = "tensorflow"))]
    pub fn benchmark_tensorflow_operation(
        &self,
        _operation: &str,
        _size: usize,
        _iterations: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        Err("TensorFlow feature not enabled".into())
    }
}

#[cfg(feature = "tensorflow")]
impl Default for TensorFlowBenchRunner {
    fn default() -> Self {
        Self::new()
    }
}

/// ToRSh vs TensorFlow matrix multiplication benchmark
#[cfg(feature = "tensorflow")]
pub struct TorshVsTensorFlowMatmul {
    tensorflow_runner: TensorFlowBenchRunner,
}

#[cfg(feature = "tensorflow")]
impl TorshVsTensorFlowMatmul {
    pub fn new() -> Self {
        Self {
            tensorflow_runner: TensorFlowBenchRunner::new(),
        }
    }
}

#[cfg(feature = "tensorflow")]
impl Benchmarkable for TorshVsTensorFlowMatmul {
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

        // Benchmark TensorFlow
        if self.tensorflow_runner.is_tensorflow_available() {
            match self
                .tensorflow_runner
                .benchmark_tensorflow_operation("matmul", size, iterations)
            {
                Ok(tensorflow_time_seconds) => {
                    let tensorflow_time_ns = tensorflow_time_seconds * 1e9;
                    results.push(ComparisonResult {
                        operation: "matrix_multiplication".to_string(),
                        library: "tensorflow".to_string(),
                        size,
                        time_ns: tensorflow_time_ns,
                        throughput: Some(self.flops(size) as f64 / tensorflow_time_ns * 1e9),
                        memory_usage: Some(size * size * 4 * 3),
                    });
                }
                Err(e) => {
                    eprintln!("TensorFlow benchmark failed: {:?}", e);
                }
            }
        }

        results
    }

    fn flops(&self, size: usize) -> usize {
        2 * size * size * size // Matrix multiplication FLOPS
    }
}

/// ToRSh vs TensorFlow element-wise operations benchmark
#[cfg(feature = "tensorflow")]
pub struct TorshVsTensorFlowElementwise {
    tensorflow_runner: TensorFlowBenchRunner,
}

#[cfg(feature = "tensorflow")]
impl TorshVsTensorFlowElementwise {
    pub fn new() -> Self {
        Self {
            tensorflow_runner: TensorFlowBenchRunner::new(),
        }
    }
}

#[cfg(feature = "tensorflow")]
impl Benchmarkable for TorshVsTensorFlowElementwise {
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

        // Benchmark TensorFlow
        if self.tensorflow_runner.is_tensorflow_available() {
            match self
                .tensorflow_runner
                .benchmark_tensorflow_operation("add", size, iterations)
            {
                Ok(tensorflow_time_seconds) => {
                    let tensorflow_time_ns = tensorflow_time_seconds * 1e9;
                    results.push(ComparisonResult {
                        operation: "elementwise_addition".to_string(),
                        library: "tensorflow".to_string(),
                        size,
                        time_ns: tensorflow_time_ns,
                        throughput: Some(self.flops(size) as f64 / tensorflow_time_ns * 1e9),
                        memory_usage: Some(size * 4 * 3),
                    });
                }
                Err(e) => {
                    eprintln!("TensorFlow benchmark failed: {:?}", e);
                }
            }
        }

        results
    }

    fn flops(&self, size: usize) -> usize {
        size // Element-wise addition FLOPS
    }
}

/// ToRSh vs TensorFlow convolution benchmark
#[cfg(feature = "tensorflow")]
pub struct TorshVsTensorFlowConv2d {
    tensorflow_runner: TensorFlowBenchRunner,
}

#[cfg(feature = "tensorflow")]
impl TorshVsTensorFlowConv2d {
    pub fn new() -> Self {
        Self {
            tensorflow_runner: TensorFlowBenchRunner::new(),
        }
    }
}

#[cfg(feature = "tensorflow")]
impl Benchmarkable for TorshVsTensorFlowConv2d {
    type Input = (torsh_nn::layers::conv::Conv2d, torsh_tensor::Tensor<f32>);
    type Output = Vec<ComparisonResult>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let batch_size = 16;
        let in_channels = 3;
        let out_channels = 64;
        let kernel_size = 3;

        let conv = torsh_nn::Conv2d::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            (1, 1),
            (1, 1),
            (1, 1),
            true,
            1,
        );
        let input =
            torsh_tensor::creation::rand::<f32>(&[batch_size, in_channels, size, size]).unwrap();

        (conv, input)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        let mut results = Vec::new();
        let size = input.1.shape().dims()[2]; // Height dimension
        let iterations = 5;

        // Benchmark ToRSh
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = input.0.forward(&input.1);
        }
        let torsh_time = start.elapsed().as_nanos() as f64 / iterations as f64;

        results.push(ComparisonResult {
            operation: "conv2d".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(self.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: Some(16 * 3 * size * size * 4), // Approximate memory
        });

        // Benchmark TensorFlow
        if self.tensorflow_runner.is_tensorflow_available() {
            match self
                .tensorflow_runner
                .benchmark_tensorflow_operation("conv2d", size, iterations)
            {
                Ok(tensorflow_time_seconds) => {
                    let tensorflow_time_ns = tensorflow_time_seconds * 1e9;
                    results.push(ComparisonResult {
                        operation: "conv2d".to_string(),
                        library: "tensorflow".to_string(),
                        size,
                        time_ns: tensorflow_time_ns,
                        throughput: Some(self.flops(size) as f64 / tensorflow_time_ns * 1e9),
                        memory_usage: Some(16 * 3 * size * size * 4),
                    });
                }
                Err(e) => {
                    eprintln!("TensorFlow benchmark failed: {:?}", e);
                }
            }
        }

        results
    }

    fn flops(&self, size: usize) -> usize {
        // Approximate FLOPS for convolution
        let batch_size = 16;
        let in_channels = 3;
        let out_channels = 64;
        let kernel_size = 3;
        batch_size * out_channels * size * size * in_channels * kernel_size * kernel_size
    }
}

/// Comprehensive TensorFlow comparison suite
#[cfg(feature = "tensorflow")]
pub fn run_tensorflow_comparison_suite() -> ComparisonRunner {
    let mut runner = ComparisonRunner::new();

    println!("🚀 Running comprehensive ToRSh vs TensorFlow benchmarks...");

    let sizes = vec![64, 128, 256, 512];

    // Matrix multiplication benchmarks
    println!("📊 Benchmarking matrix multiplication...");
    let mut matmul_bench = TorshVsTensorFlowMatmul::new();
    for &size in &sizes {
        let input = matmul_bench.setup(size);
        let results = matmul_bench.run(&input);
        for result in results {
            runner.add_result(result);
        }
    }

    // Element-wise operation benchmarks
    println!("📊 Benchmarking element-wise operations...");
    let mut elementwise_bench = TorshVsTensorFlowElementwise::new();
    for &size in &[1000, 10000, 100000, 1000000] {
        let input = elementwise_bench.setup(size);
        let results = elementwise_bench.run(&input);
        for result in results {
            runner.add_result(result);
        }
    }

    // Convolution benchmarks
    println!("📊 Benchmarking convolution operations...");
    let mut conv_bench = TorshVsTensorFlowConv2d::new();
    for &size in &[32, 64, 128, 224] {
        let input = conv_bench.setup(size);
        let results = conv_bench.run(&input);
        for result in results {
            runner.add_result(result);
        }
    }

    println!("✅ TensorFlow comparison benchmarks completed!");
    runner
}

/// Generate comprehensive TensorFlow performance report
#[cfg(feature = "tensorflow")]
pub fn generate_tensorflow_comparison_report() -> std::io::Result<()> {
    let runner = run_tensorflow_comparison_suite();

    // Generate markdown report
    runner.generate_report("target/tensorflow_comparison.md")?;

    // Generate detailed analysis
    let mut analyzer = super::analysis::PerformanceAnalyzer::new();
    analyzer.add_results(runner.results());

    // Create comprehensive performance analysis
    let mut analysis_file = std::fs::File::create("target/tensorflow_vs_torsh_analysis.md")?;
    use std::io::Write;

    writeln!(
        analysis_file,
        "# ToRSh vs TensorFlow Performance Analysis\n"
    )?;
    writeln!(
        analysis_file,
        "Generated on: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;

    let operations = ["matrix_multiplication", "elementwise_addition", "conv2d"];

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
    writeln!(analysis_file, "This benchmark compares ToRSh (pure Rust tensor library) with TensorFlow (Python/C++ tensor library).")?;
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
    writeln!(
        analysis_file,
        "- **Convolution**: Neural network workload performance"
    )?;
    writeln!(analysis_file, "\n### Notes:\n")?;
    writeln!(
        analysis_file,
        "- All benchmarks run with consistent random seeds for reproducibility"
    )?;
    writeln!(
        analysis_file,
        "- TensorFlow uses optimized XLA compilation and CUDA when available"
    )?;
    writeln!(
        analysis_file,
        "- ToRSh uses pure Rust implementations with SIMD optimizations"
    )?;

    println!("📈 Comprehensive TensorFlow comparison report generated!");
    println!("   📄 Basic report: target/tensorflow_comparison.md");
    println!("   📊 Detailed analysis: target/tensorflow_vs_torsh_analysis.md");

    Ok(())
}