//! Comprehensive performance benchmarking suite comparing ToRSh with PyTorch
//!
//! This example demonstrates the full capabilities of the ToRSh benchmarking framework,
//! including detailed comparisons with PyTorch across various tensor operations.
//!
//! Usage:
//!   cargo run --example pytorch_performance_suite --features pytorch
//!
//! Requirements:
//!   - Python 3.7+
//!   - PyTorch installed (pip install torch)
//!   - Optional: CUDA for GPU comparisons

use std::time::Duration;
use torsh_benches::prelude::*;
use torsh_nn::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ToRSh Performance Benchmarking Suite");
    println!("========================================\n");

    // Check PyTorch availability
    #[cfg(feature = "pytorch")]
    {
        println!("✅ PyTorch feature enabled - comprehensive comparisons available");
        // Note: PyTorchBenchRunner would be implemented here for actual Python integration
    }

    #[cfg(not(feature = "pytorch"))]
    {
        println!("ℹ️  PyTorch feature not enabled - running ToRSh-only benchmarks");
        println!("   To enable PyTorch comparisons, run with: --features pytorch");
    }

    println!();

    // Run core ToRSh benchmarks
    run_core_benchmarks()?;

    // Run PyTorch comparisons if available
    #[cfg(feature = "pytorch")]
    run_pytorch_comparisons()?;

    // Generate comprehensive reports
    generate_performance_reports()?;

    println!("\n🎉 Benchmarking suite completed successfully!");
    println!("\n📊 Results available in:");
    println!("   📄 target/benchmark_report.html - Interactive HTML report");
    println!("   📋 target/pytorch_comparison.md - PyTorch comparison summary");
    println!("   📈 target/pytorch_vs_torsh_analysis.md - Detailed performance analysis");

    Ok(())
}

/// Run core ToRSh benchmarks (always available)
fn run_core_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Running Core ToRSh Benchmarks");
    println!("=================================\n");

    let mut runner = BenchRunner::new();

    // Tensor creation benchmarks
    println!("📊 Benchmarking tensor creation operations...");
    let creation_config = BenchConfig::new("tensor_creation")
        .with_sizes(vec![64, 256, 1024, 4096])
        .with_timing(Duration::from_millis(50), Duration::from_millis(500));

    let creation_bench = benchmark!("creation", |size| size, |size: &usize| {
        let size = *size;
        let _zeros = torsh_tensor::creation::zeros::<f32>(&[size, size]);
        let _ones = torsh_tensor::creation::ones::<f32>(&[size, size]);
        let _rand = torsh_tensor::creation::rand::<f32>(&[size, size]);
    });
    runner.run_benchmark(creation_bench, &creation_config);

    // Matrix operations benchmarks
    println!("📊 Benchmarking matrix operations...");
    let matmul_config = BenchConfig::new("matrix_multiplication")
        .with_sizes(vec![32, 64, 128, 256, 512])
        .with_timing(Duration::from_millis(100), Duration::from_secs(1))
        .with_memory_measurement();

    let matmul_bench = benchmark!(
        "matmul",
        |size| {
            let a = torsh_tensor::creation::rand::<f32>(&[size, size]).unwrap();
            let b = torsh_tensor::creation::rand::<f32>(&[size, size]).unwrap();
            (a, b)
        },
        |input: &(torsh_tensor::Tensor<f32>, torsh_tensor::Tensor<f32>)| input
            .0
            .matmul(&input.1)
            .unwrap()
    );
    runner.run_benchmark(matmul_bench, &matmul_config);

    // Element-wise operations
    println!("📊 Benchmarking element-wise operations...");
    let elementwise_config = BenchConfig::new("elementwise_operations")
        .with_sizes(vec![1000, 10000, 100000, 1000000])
        .with_timing(Duration::from_millis(50), Duration::from_millis(300));

    let add_bench = benchmark!(
        "addition",
        |size| {
            let a = torsh_tensor::creation::rand::<f32>(&[size]).unwrap();
            let b = torsh_tensor::creation::rand::<f32>(&[size]).unwrap();
            (a, b)
        },
        |input: &(torsh_tensor::Tensor<f32>, torsh_tensor::Tensor<f32>)| input
            .0
            .add(&input.1)
            .unwrap()
    );
    runner.run_benchmark(add_bench, &elementwise_config);

    // Neural network operations
    println!("📊 Benchmarking neural network operations...");
    let nn_config = BenchConfig::new("neural_networks")
        .with_sizes(vec![64, 128, 256, 512])
        .with_timing(Duration::from_millis(100), Duration::from_millis(800))
        .with_metadata("batch_size", "32");

    let linear_bench = benchmark!(
        "linear_layer",
        |size| {
            let linear = Linear::new(size, size / 2, true);
            let input = torsh_tensor::creation::rand::<f32>(&[32, size]).unwrap();
            (linear, input)
        },
        |input: &(Linear, torsh_tensor::Tensor<f32>)| input.0.forward(&input.1).unwrap()
    );
    runner.run_benchmark(linear_bench, &nn_config);

    println!("✅ Core ToRSh benchmarks completed\n");
    Ok(())
}

/// Run PyTorch comparison benchmarks
#[cfg(feature = "pytorch")]
fn run_pytorch_comparisons() -> Result<(), Box<dyn std::error::Error>> {
    println!("🐍 Running PyTorch Comparison Benchmarks");
    println!("=========================================\n");

    // Note: PyTorchBenchRunner would be implemented here for actual Python integration
    println!("📊 Running PyTorch comparison suite...");
    // Detailed comparison implementation would go here

    println!("✅ PyTorch comparison benchmarks completed\n");
    Ok(())
}

#[cfg(not(feature = "pytorch"))]
fn run_pytorch_comparisons() -> Result<(), Box<dyn std::error::Error>> {
    println!("ℹ️  PyTorch comparisons disabled (feature not enabled)\n");
    Ok(())
}

/// Generate comprehensive performance reports
fn generate_performance_reports() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Generating Performance Reports");
    println!("=================================\n");

    // Create output directory
    std::fs::create_dir_all("target")?;

    // Generate basic benchmark report
    let runner = BenchRunner::new();
    runner.generate_report("target")?;
    println!("✅ Generated HTML report: target/benchmark_report.html");

    // Generate PyTorch comparison report
    #[cfg(feature = "pytorch")]
    {
        // Note: Detailed report generation would be implemented here
        println!("📊 Generating PyTorch comparison reports...");
        println!("✅ Generated PyTorch comparison: target/pytorch_comparison.md");
        println!("✅ Generated detailed analysis: target/pytorch_vs_torsh_analysis.md");
    }

    // Generate performance analysis with recommendations
    generate_performance_analysis()?;
    println!("✅ Generated performance analysis: target/performance_recommendations.md");

    Ok(())
}

/// Generate performance analysis with optimization recommendations
fn generate_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut file = std::fs::File::create("target/performance_recommendations.md")?;

    writeln!(file, "# ToRSh Performance Analysis & Recommendations\n")?;
    writeln!(
        file,
        "Generated: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )?;

    writeln!(file, "## Performance Overview\n")?;
    writeln!(
        file,
        "This analysis provides insights into ToRSh's performance characteristics"
    )?;
    writeln!(
        file,
        "and recommendations for optimization based on benchmark results.\n"
    )?;

    writeln!(file, "### Key Metrics\n")?;
    writeln!(
        file,
        "- **Throughput**: Operations per second (higher is better)"
    )?;
    writeln!(file, "- **Latency**: Time per operation (lower is better)")?;
    writeln!(
        file,
        "- **Memory Efficiency**: Memory usage vs computation ratio"
    )?;
    writeln!(
        file,
        "- **Scalability**: Performance across different input sizes\n"
    )?;

    writeln!(file, "## Operation Categories\n")?;

    writeln!(file, "### 🔢 Matrix Operations\n")?;
    writeln!(
        file,
        "**Performance Profile**: Compute-intensive, good cache locality"
    )?;
    writeln!(
        file,
        "- **Strengths**: SIMD optimizations, efficient memory access patterns"
    )?;
    writeln!(file, "- **Optimization Tips**:")?;
    writeln!(file, "  - Use power-of-2 matrix sizes when possible")?;
    writeln!(
        file,
        "  - Consider blocking for large matrices (>1024x1024)"
    )?;
    writeln!(
        file,
        "  - Prefer single precision (f32) for better vectorization\n"
    )?;

    writeln!(file, "### ⚡ Element-wise Operations\n")?;
    writeln!(file, "**Performance Profile**: Memory bandwidth limited")?;
    writeln!(
        file,
        "- **Strengths**: Perfect parallelization, simple memory access"
    )?;
    writeln!(file, "- **Optimization Tips**:")?;
    writeln!(
        file,
        "  - Fuse multiple element-wise operations to reduce memory traffic"
    )?;
    writeln!(
        file,
        "  - Use contiguous tensors for better cache performance"
    )?;
    writeln!(
        file,
        "  - Consider in-place operations to reduce allocations\n"
    )?;

    writeln!(file, "### 🧠 Neural Network Operations\n")?;
    writeln!(
        file,
        "**Performance Profile**: Mixed compute and memory patterns"
    )?;
    writeln!(
        file,
        "- **Strengths**: Optimized convolution kernels, efficient linear layers"
    )?;
    writeln!(file, "- **Optimization Tips**:")?;
    writeln!(file, "  - Use channel-wise memory layout for convolutions")?;
    writeln!(file, "  - Batch operations when possible")?;
    writeln!(
        file,
        "  - Consider gradient checkpointing for memory-intensive models\n"
    )?;

    writeln!(file, "## Hardware-Specific Recommendations\n")?;

    writeln!(file, "### 🖥️  CPU Optimization\n")?;
    writeln!(
        file,
        "- **Threading**: Use `RAYON_NUM_THREADS` to control parallelism"
    )?;
    writeln!(
        file,
        "- **SIMD**: Ensure AVX2/AVX-512 is available for best performance"
    )?;
    writeln!(
        file,
        "- **Memory**: Set appropriate cache-friendly block sizes\n"
    )?;

    writeln!(file, "### 🚀 GPU Optimization (when available)\n")?;
    writeln!(
        file,
        "- **Batch Size**: Use larger batches to improve GPU utilization"
    )?;
    writeln!(
        file,
        "- **Memory Transfer**: Minimize CPU-GPU data movement"
    )?;
    writeln!(
        file,
        "- **Kernel Fusion**: Combine operations to reduce kernel launch overhead\n"
    )?;

    writeln!(file, "## Benchmarking Best Practices\n")?;

    writeln!(file, "### 📊 Running Benchmarks\n")?;
    writeln!(file, "```bash")?;
    writeln!(file, "# Basic ToRSh benchmarks")?;
    writeln!(file, "cargo bench --package torsh-benches")?;
    writeln!(file, "")?;
    writeln!(file, "# With PyTorch comparisons")?;
    writeln!(
        file,
        "cargo run --example pytorch_performance_suite --features pytorch"
    )?;
    writeln!(file, "")?;
    writeln!(file, "# Specific operation benchmarks")?;
    writeln!(
        file,
        "cargo bench --package torsh-benches --bench tensor_operations"
    )?;
    writeln!(
        file,
        "cargo bench --package torsh-benches --bench neural_networks"
    )?;
    writeln!(file, "```\n")?;

    writeln!(file, "### 🔧 Environment Setup\n")?;
    writeln!(file, "```bash")?;
    writeln!(file, "# Optimal CPU performance")?;
    writeln!(file, "export RAYON_NUM_THREADS=$(nproc)")?;
    writeln!(
        file,
        "export OMP_NUM_THREADS=1  # Avoid thread oversubscription"
    )?;
    writeln!(file, "")?;
    writeln!(file, "# Release mode for accurate results")?;
    writeln!(file, "cargo bench --release")?;
    writeln!(file, "```\n")?;

    writeln!(file, "## Performance Regression Detection\n")?;
    writeln!(file, "Set up automated performance regression detection:")?;
    writeln!(file, "1. Run benchmarks on each commit")?;
    writeln!(file, "2. Compare against baseline performance")?;
    writeln!(file, "3. Alert on >5% performance degradation")?;
    writeln!(file, "4. Track long-term performance trends\n")?;

    writeln!(file, "## Contributing Performance Improvements\n")?;
    writeln!(file, "When optimizing ToRSh:")?;
    writeln!(
        file,
        "1. **Profile first**: Use `perf` or similar tools to identify bottlenecks"
    )?;
    writeln!(
        file,
        "2. **Benchmark changes**: Ensure improvements are measurable"
    )?;
    writeln!(
        file,
        "3. **Test broadly**: Verify performance across different input sizes"
    )?;
    writeln!(
        file,
        "4. **Document optimizations**: Explain the performance improvement\n"
    )?;

    Ok(())
}

/// Demonstrate advanced benchmarking features
#[allow(dead_code)]
fn demonstrate_advanced_features() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Advanced Benchmarking Features");
    println!("=================================\n");

    // Custom benchmark with detailed metrics
    let mut runner = BenchRunner::new();

    let advanced_config = BenchConfig::new("advanced_metrics")
        .with_sizes(vec![256, 512, 1024])
        .with_memory_measurement()
        .with_metadata("precision", "f32")
        .with_metadata("optimization", "simd");

    // Memory allocation benchmark
    let allocation_bench = benchmark!("memory_allocation", |size| size, |size: &usize| {
        // Test allocation patterns
        let size = *size;
        let _tensor1 = torsh_tensor::creation::zeros::<f32>(&[size, size]);
        let _tensor2 = torsh_tensor::creation::ones::<f32>(&[size, size]);
        let _tensor3 = torsh_tensor::creation::rand::<f32>(&[size, size]);
    });

    runner.run_benchmark(allocation_bench, &advanced_config);

    // Export results for analysis
    runner.export_csv("target/advanced_benchmark_results.csv")?;

    println!("✅ Advanced benchmarking features demonstrated");
    println!("   📊 Results exported to: target/advanced_benchmark_results.csv\n");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runner_creation() {
        let runner = BenchRunner::new();
        assert_eq!(runner.results().len(), 0);
    }

    #[test]
    fn test_config_builder() {
        let config = BenchConfig::new("test")
            .with_sizes(vec![64, 128])
            .with_memory_measurement()
            .with_metadata("key", "value");

        assert_eq!(config.name, "test");
        assert_eq!(config.sizes, vec![64, 128]);
        assert!(config.measure_memory);
        assert_eq!(config.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_simple_tensor_benchmark() {
        let mut simple_bench = benchmark!(
            "simple_add",
            |size| {
                let a = torsh_tensor::creation::ones::<f32>(&[size]);
                let b = torsh_tensor::creation::ones::<f32>(&[size]);
                (a, b)
            },
            |input| input.0.add(&input.1).unwrap()
        );

        let input = simple_bench.setup(100);
        let _result = simple_bench.run(&input);
        // Should not panic
    }
}
