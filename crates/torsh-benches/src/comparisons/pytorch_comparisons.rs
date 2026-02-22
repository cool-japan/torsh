//! PyTorch comparison benchmarks
//!
//! This module provides comprehensive PyTorch benchmark implementations
//! for comparing ToRSh performance with PyTorch operations.

use super::core::{ComparisonResult, ComparisonRunner};
use super::torsh_benchmarks::{TorshElementwiseBench, TorshMatmulBench};

#[cfg(feature = "pytorch")]
use numpy::PyArray2;
#[cfg(feature = "pytorch")]
use pyo3::prelude::*;
#[cfg(feature = "pytorch")]
use pyo3::types::PyModule;

/// PyTorch matrix multiplication benchmark
#[cfg(feature = "pytorch")]
pub struct PyTorchMatmulBench;

#[cfg(feature = "pytorch")]
impl PyTorchMatmulBench {
    pub fn setup(&mut self, size: usize) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;

            // Create random tensors
            let a = torch.call_method1("randn", (size, size))?;
            let b = torch.call_method1("randn", (size, size))?;

            Ok((a.into(), b.into()))
        })
    }

    pub fn run(&mut self, input: &(Py<PyAny>, Py<PyAny>)) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;
            let result = torch.call_method1("matmul", (&input.0, &input.1))?;
            Ok(result.into())
        })
    }

    pub fn flops(&self, size: usize) -> usize {
        2 * size * size * size
    }
}

/// PyTorch element-wise addition benchmark
#[cfg(feature = "pytorch")]
pub struct PyTorchElementwiseBench;

#[cfg(feature = "pytorch")]
impl PyTorchElementwiseBench {
    pub fn setup(&mut self, size: usize) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;

            let a = torch.call_method1("randn", (size,))?;
            let b = torch.call_method1("randn", (size,))?;

            Ok((a.into(), b.into()))
        })
    }

    pub fn run(&mut self, input: &(Py<PyAny>, Py<PyAny>)) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = input.0.call_method1(py, "__add__", (&input.1,))?;
            Ok(result)
        })
    }

    pub fn flops(&self, size: usize) -> usize {
        size
    }
}

/// PyTorch convolution benchmark
#[cfg(feature = "pytorch")]
pub struct PyTorchConvBench;

#[cfg(feature = "pytorch")]
impl PyTorchConvBench {
    pub fn setup(
        &mut self,
        batch_size: usize,
        in_channels: usize,
        height: usize,
        width: usize,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;
            let nn = PyModule::import(py, "torch.nn")?;

            // Create input tensor
            let input = torch.call_method1("randn", (batch_size, in_channels, height, width))?;

            // Create conv layer
            let conv = nn.call_method("Conv2d", (in_channels, 64, 3), None)?;

            Ok((input.into(), conv.into()))
        })
    }

    pub fn run(&mut self, input: &(Py<PyAny>, Py<PyAny>)) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let result = input.1.call_method1(py, "__call__", (&input.0,))?;
            Ok(result)
        })
    }

    pub fn flops(
        &self,
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    ) -> usize {
        batch_size * out_channels * height * width * in_channels * kernel_size * kernel_size
    }
}

/// PyTorch autograd/backward benchmark
#[cfg(feature = "pytorch")]
pub struct PyTorchAutogradBench;

#[cfg(feature = "pytorch")]
impl PyTorchAutogradBench {
    pub fn setup(&mut self, size: usize) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;

            // Create tensor with gradient tracking
            let x = torch.call_method1("randn", (size, size))?;
            x.call_method1("requires_grad_", (true,))?;

            // Perform some operations
            let y = x.call_method1("matmul", (x,))?;
            let loss = y.call_method0("sum")?;

            Ok(loss.into())
        })
    }

    pub fn run(&mut self, input: &Py<PyAny>) -> PyResult<()> {
        Python::with_gil(|py| {
            input.call_method0(py, "backward")?;
            Ok(())
        })
    }

    pub fn flops(&self, size: usize) -> usize {
        4 * size * size * size // Approximate for forward + backward
    }
}

/// PyTorch data loading benchmark
#[cfg(feature = "pytorch")]
pub struct PyTorchDataLoaderBench;

#[cfg(feature = "pytorch")]
impl PyTorchDataLoaderBench {
    pub fn setup(&mut self, num_samples: usize, batch_size: usize) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;
            let torch_utils = PyModule::import(py, "torch.utils.data")?;

            // Create random dataset
            let data = torch.call_method1("randn", (num_samples, 32, 32))?;
            let targets = torch.call_method1("randint", (0, 10, (num_samples,)))?;

            let dataset = torch_utils.call_method1("TensorDataset", (data, targets))?;
            let dataloader = torch_utils.call_method(
                "DataLoader",
                (dataset,),
                Some([("batch_size", batch_size), ("shuffle", true)].into_py_dict(py)),
            )?;

            Ok(dataloader.into())
        })
    }

    pub fn run(&mut self, input: &Py<PyAny>) -> PyResult<usize> {
        Python::with_gil(|py| {
            let mut count = 0;
            for _batch in input.iter(py)? {
                count += 1;
                if count >= 10 {
                    // Limit to 10 batches for benchmarking
                    break;
                }
            }
            Ok(count)
        })
    }
}

/// Run comprehensive PyTorch comparison benchmarks
#[cfg(feature = "pytorch")]
pub fn run_pytorch_comparison_benchmarks() -> ComparisonRunner {
    let mut runner = ComparisonRunner::new();

    let sizes = vec![64, 128, 256, 512, 1024];

    // Matrix multiplication comparisons
    for &size in &sizes {
        // ToRSh benchmark
        let mut torsh_bench = TorshMatmulBench;
        let torsh_input = torsh_bench.setup(size);

        let start = std::time::Instant::now();
        let _ = torsh_bench.run(&torsh_input);
        let torsh_time = start.elapsed().as_nanos() as f64;

        runner.add_result(ComparisonResult {
            operation: "matrix_multiplication".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(torsh_bench.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: Some(2 * size * size * 4), // Two matrices
        });

        // PyTorch benchmark
        let mut pytorch_bench = PyTorchMatmulBench;
        if let Ok(pytorch_input) = pytorch_bench.setup(size) {
            let start = std::time::Instant::now();
            let _ = pytorch_bench.run(&pytorch_input);
            let pytorch_time = start.elapsed().as_nanos() as f64;

            runner.add_result(ComparisonResult {
                operation: "matrix_multiplication".to_string(),
                library: "pytorch".to_string(),
                size,
                time_ns: pytorch_time,
                throughput: Some(pytorch_bench.flops(size) as f64 / pytorch_time * 1e9),
                memory_usage: Some(2 * size * size * 4),
            });
        }
    }

    // Element-wise operation comparisons
    for &size in &sizes {
        // ToRSh benchmark
        let mut torsh_bench = TorshElementwiseBench;
        let torsh_input = torsh_bench.setup(size);

        let start = std::time::Instant::now();
        let _ = torsh_bench.run(&torsh_input);
        let torsh_time = start.elapsed().as_nanos() as f64;

        runner.add_result(ComparisonResult {
            operation: "elementwise_addition".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(torsh_bench.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: Some(size * 4),
        });

        // PyTorch benchmark
        let mut pytorch_bench = PyTorchElementwiseBench;
        if let Ok(pytorch_input) = pytorch_bench.setup(size) {
            let start = std::time::Instant::now();
            let _ = pytorch_bench.run(&pytorch_input);
            let pytorch_time = start.elapsed().as_nanos() as f64;

            runner.add_result(ComparisonResult {
                operation: "elementwise_addition".to_string(),
                library: "pytorch".to_string(),
                size,
                time_ns: pytorch_time,
                throughput: Some(pytorch_bench.flops(size) as f64 / pytorch_time * 1e9),
                memory_usage: Some(size * 4),
            });
        }
    }

    // Autograd/backward pass comparisons
    for &size in &[32, 64, 128, 256] {
        // ToRSh autograd benchmark
        let mut torsh_bench = crate::benchmarks::BackwardPassBench;
        let torsh_input = torsh_bench.setup(size);

        let start = std::time::Instant::now();
        let _ = torsh_bench.run(&torsh_input);
        let torsh_time = start.elapsed().as_nanos() as f64;

        runner.add_result(ComparisonResult {
            operation: "autograd_backward".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(torsh_bench.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: Some(size * size * 4),
        });

        // PyTorch autograd benchmark
        let mut pytorch_bench = PyTorchAutogradBench;
        if let Ok(pytorch_input) = pytorch_bench.setup(size) {
            let start = std::time::Instant::now();
            let _ = pytorch_bench.run(&pytorch_input);
            let pytorch_time = start.elapsed().as_nanos() as f64;

            runner.add_result(ComparisonResult {
                operation: "autograd_backward".to_string(),
                library: "pytorch".to_string(),
                size,
                time_ns: pytorch_time,
                throughput: Some(pytorch_bench.flops(size) as f64 / pytorch_time * 1e9),
                memory_usage: Some(size * size * 4),
            });
        }
    }

    // Data loading comparisons
    for &num_samples in &[100, 500, 1000] {
        let batch_size = 32;

        // ToRSh data loading benchmark
        let mut torsh_bench = crate::benchmarks::DataLoaderThroughputBench;
        let torsh_input = torsh_bench.setup(num_samples / 10); // Adjust for size parameter

        let start = std::time::Instant::now();
        let _ = torsh_bench.run(&torsh_input);
        let torsh_time = start.elapsed().as_nanos() as f64;

        runner.add_result(ComparisonResult {
            operation: "data_loading".to_string(),
            library: "torsh".to_string(),
            size: num_samples,
            time_ns: torsh_time,
            throughput: Some(num_samples as f64 / torsh_time * 1e9), // Samples per second
            memory_usage: Some(num_samples * 32 * 32 * 4),
        });

        // PyTorch data loading benchmark
        let mut pytorch_bench = PyTorchDataLoaderBench;
        if let Ok(pytorch_input) = pytorch_bench.setup(num_samples, batch_size) {
            let start = std::time::Instant::now();
            let _ = pytorch_bench.run(&pytorch_input);
            let pytorch_time = start.elapsed().as_nanos() as f64;

            runner.add_result(ComparisonResult {
                operation: "data_loading".to_string(),
                library: "pytorch".to_string(),
                size: num_samples,
                time_ns: pytorch_time,
                throughput: Some(num_samples as f64 / pytorch_time * 1e9),
                memory_usage: Some(num_samples * 32 * 32 * 4),
            });
        }
    }

    runner
}

/// Comprehensive comparison suite including PyTorch
#[cfg(feature = "pytorch")]
pub fn run_comprehensive_pytorch_benchmarks() -> std::io::Result<()> {
    println!("Running comprehensive PyTorch comparison benchmarks...");

    let runner = run_pytorch_comparison_benchmarks();

    // Generate detailed comparison report
    runner.generate_report("target/pytorch_comparison_report.md")?;

    // Perform analysis
    let mut analyzer = super::analysis::PerformanceAnalyzer::new();
    analyzer.add_results(runner.results());

    let operations = [
        "matrix_multiplication",
        "elementwise_addition",
        "autograd_backward",
        "data_loading",
    ];

    let mut analysis_file = std::fs::File::create("target/pytorch_analysis.md")?;
    use std::io::Write;

    writeln!(analysis_file, "# ToRSh vs PyTorch Performance Analysis\n")?;
    writeln!(
        analysis_file,
        "This report compares ToRSh performance against PyTorch across key operations.\n"
    )?;

    for operation in &operations {
        let analysis = analyzer.analyze_operation(operation);

        writeln!(
            analysis_file,
            "## {}\n",
            operation.replace('_', " ").to_uppercase()
        )?;

        // Calculate performance ratios
        let torsh_stats = analysis.library_stats.get("torsh");
        let pytorch_stats = analysis.library_stats.get("pytorch");

        if let (Some(torsh), Some(pytorch)) = (torsh_stats, pytorch_stats) {
            let speedup = pytorch.mean_time_ns / torsh.mean_time_ns;

            if speedup > 1.0 {
                writeln!(
                    analysis_file,
                    "**ToRSh is {:.2}x faster than PyTorch for this operation**\n",
                    speedup
                )?;
            } else {
                writeln!(
                    analysis_file,
                    "**PyTorch is {:.2}x faster than ToRSh for this operation**\n",
                    1.0 / speedup
                )?;
            }

            writeln!(
                analysis_file,
                "| Library | Avg Time (μs) | Throughput | Samples |"
            )?;
            writeln!(
                analysis_file,
                "|---------|---------------|------------|---------|"
            )?;
            writeln!(
                analysis_file,
                "| ToRSh   | {:.2}         | {:.2}      | {}      |",
                torsh.mean_time_ns / 1000.0,
                torsh.mean_throughput.unwrap_or(0.0) / 1e6,
                torsh.sample_count
            )?;
            writeln!(
                analysis_file,
                "| PyTorch | {:.2}         | {:.2}      | {}      |",
                pytorch.mean_time_ns / 1000.0,
                pytorch.mean_throughput.unwrap_or(0.0) / 1e6,
                pytorch.sample_count
            )?;
            writeln!(analysis_file)?;
        }

        // Add recommendations
        if !analysis.recommendations.is_empty() {
            writeln!(analysis_file, "### Recommendations\n")?;
            for rec in &analysis.recommendations {
                writeln!(analysis_file, "- {}", rec)?;
            }
            writeln!(analysis_file)?;
        }
    }

    // Add summary section
    writeln!(analysis_file, "## Summary\n")?;
    writeln!(analysis_file, "This benchmark suite demonstrates ToRSh's performance characteristics compared to PyTorch.")?;
    writeln!(
        analysis_file,
        "Key areas for further optimization and competitive advantages are highlighted above.\n"
    )?;

    println!("PyTorch comparison benchmarks completed!");
    println!("Results saved to target/pytorch_comparison_report.md");
    println!("Analysis saved to target/pytorch_analysis.md");

    Ok(())
}

/// Quick PyTorch comparison for CI/development
#[cfg(feature = "pytorch")]
pub fn run_quick_pytorch_comparison() -> ComparisonRunner {
    let mut runner = ComparisonRunner::new();

    // Quick comparison with smaller sizes
    let sizes = vec![64, 128, 256];

    for &size in &sizes {
        // Matrix multiplication comparison
        let mut torsh_bench = TorshMatmulBench;
        let torsh_input = torsh_bench.setup(size);

        let start = std::time::Instant::now();
        let _ = torsh_bench.run(&torsh_input);
        let torsh_time = start.elapsed().as_nanos() as f64;

        runner.add_result(ComparisonResult {
            operation: "quick_matmul".to_string(),
            library: "torsh".to_string(),
            size,
            time_ns: torsh_time,
            throughput: Some(torsh_bench.flops(size) as f64 / torsh_time * 1e9),
            memory_usage: None,
        });

        // PyTorch comparison
        let mut pytorch_bench = PyTorchMatmulBench;
        if let Ok(pytorch_input) = pytorch_bench.setup(size) {
            let start = std::time::Instant::now();
            let _ = pytorch_bench.run(&pytorch_input);
            let pytorch_time = start.elapsed().as_nanos() as f64;

            runner.add_result(ComparisonResult {
                operation: "quick_matmul".to_string(),
                library: "pytorch".to_string(),
                size,
                time_ns: pytorch_time,
                throughput: Some(pytorch_bench.flops(size) as f64 / pytorch_time * 1e9),
                memory_usage: None,
            });
        }
    }

    runner
}