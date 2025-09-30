# ToRSh Benchmarking Guide

## Overview

This guide provides comprehensive instructions for using the ToRSh benchmarking suite to measure performance, identify bottlenecks, and optimize your machine learning workloads.

## Quick Start

### Running Basic Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark category
cargo bench --bench tensor_operations
cargo bench --bench neural_networks
cargo bench --bench autograd

# Run with specific filter
cargo bench "matmul"
cargo bench "conv2d"
```

### Basic Example

```rust
use torsh_benches::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a benchmark runner
    let mut runner = BenchRunner::new(BenchConfig::default());
    
    // Add benchmarks
    runner.add_benchmark(Box::new(MatMulBench::new(512, 512, 512)));
    runner.add_benchmark(Box::new(Conv2dBench::new(32, 3, 224, 224)));
    
    // Run and generate report
    let results = runner.run_all()?;
    let report = HtmlReportGenerator::new().generate(&results)?;
    report.save("benchmark_results.html")?;
    
    Ok(())
}
```

## Benchmark Categories

### 1. Tensor Operations

**Matrix Operations**
- `MatMulBench`: Matrix multiplication with configurable dimensions
- `EinSumBench`: Einstein summation operations
- `BatchedMatMulBench`: Batched matrix operations

**Element-wise Operations**
- `ElementWiseBench`: Addition, multiplication, division operations
- `UnaryOpsBench`: Activation functions, trigonometric operations
- `BroadcastingBench`: Broadcasting semantics performance

**Reduction Operations**
- `ReductionBench`: Sum, mean, max, min operations along dimensions
- `NormBench`: L1, L2, and custom norm calculations

### 2. Neural Network Layers

**Core Layers**
- `LinearBench`: Dense/fully-connected layers
- `Conv2dBench`: 2D convolution operations
- `BatchNormBench`: Batch normalization layers

**Advanced Layers**
- `AttentionBench`: Multi-head attention mechanisms
- `TransformerBench`: Complete transformer blocks
- `ResNetBench`: ResNet architectures (ResNet18, ResNet50, ResNet152)

### 3. Autograd System

**Gradient Computation**
- `BackwardPassBench`: Backward propagation timing
- `GradientComputationBench`: Gradient calculation overhead
- `HigherOrderDerivativeBench`: Second and higher-order derivatives

**Memory Management**
- `AutogradMemoryBench`: Memory usage during gradient computation
- `CheckpointingBench`: Gradient checkpointing performance

### 4. Model Architectures

**Computer Vision**
- `ResNetBench`: ResNet family benchmarks
- `VGGBench`: VGG architecture performance
- `EfficientNetBench`: EfficientNet models

**Natural Language Processing**
- `TransformerBench`: BERT, GPT-style models
- `LSTMBench`: LSTM and GRU networks
- `EmbeddingBench`: Embedding layer performance

**Generative Models**
- `GANBench`: Generator and discriminator networks
- `VAEBench`: Variational autoencoder performance

### 5. Hardware-Specific Benchmarks

**CPU Optimizations**
- `SIMDBench`: SIMD instruction utilization
- `CacheBench`: Cache-friendly access patterns
- `ParallelismBench`: Multi-threading efficiency

**GPU Acceleration**
- `CUDABench`: CUDA kernel performance
- `MemoryBandwidthBench`: GPU memory throughput
- `KernelFusionBench`: Fused operation performance

**Mobile/Edge**
- `MobileBench`: ARM optimization benchmarks
- `QuantizationBench`: INT8/INT16 quantized operations
- `PruningBench`: Sparse model performance

## Configuration Options

### Benchmark Configuration

```rust
use torsh_benches::BenchConfig;

let config = BenchConfig {
    // Timing configuration
    warm_up_iterations: 10,
    measurement_iterations: 100,
    timeout: Duration::from_secs(300),
    
    // Hardware configuration
    device: Device::cuda_if_available(),
    dtype: DType::F32,
    
    // Memory configuration
    memory_tracking: true,
    gc_between_runs: true,
    
    // Output configuration
    detailed_timing: true,
    memory_profiling: true,
    power_monitoring: true,
};
```

### Device Selection

```rust
// Automatic device selection
let config = BenchConfig::default();

// Force CPU execution
let config = BenchConfig::cpu_only();

// Force GPU execution (if available)
let config = BenchConfig::gpu_only();

// Mixed precision
let config = BenchConfig::mixed_precision();
```

## Advanced Usage

### Custom Benchmarks

```rust
use torsh_benches::{Benchmark, BenchmarkResult, BenchConfig};
use torsh_tensor::Tensor;

struct CustomOpBench {
    input_size: usize,
}

impl Benchmark for CustomOpBench {
    fn name(&self) -> &str { "custom_operation" }
    
    fn setup(&mut self, config: &BenchConfig) -> Result<()> {
        // Initialize tensors, models, etc.
        Ok(())
    }
    
    fn run_iteration(&mut self, iteration: usize) -> Result<BenchmarkResult> {
        let start = Instant::now();
        
        // Your custom operation here
        let input = Tensor::randn(vec![self.input_size, self.input_size]);
        let output = input.custom_operation()?;
        
        // Ensure computation completes
        output.synchronize()?;
        
        let duration = start.elapsed();
        
        Ok(BenchmarkResult {
            duration,
            memory_used: output.memory_usage()?,
            throughput: Some(calculate_throughput(&input, duration)),
            metadata: HashMap::new(),
        })
    }
    
    fn teardown(&mut self) -> Result<()> {
        // Cleanup resources
        Ok(())
    }
}
```

### Comparison Benchmarks

```rust
use torsh_benches::comparison::*;

// Compare with PyTorch
let pytorch_bench = PyTorchMatMulBench::new(512, 512, 512);
let torsh_bench = MatMulBench::new(512, 512, 512);

let comparison = CrossFrameworkComparison::new()
    .add_framework("pytorch", pytorch_bench)
    .add_framework("torsh", torsh_bench)
    .run()?;

// Generate comparison report
let report = ComparisonReportGenerator::new()
    .with_speedup_analysis()
    .with_memory_comparison()
    .generate(&comparison)?;
```

### Profiling Integration

```rust
use torsh_benches::profiling::*;

let mut runner = BenchRunner::new(config);

// Enable detailed profiling
runner.enable_profiling(ProfilingConfig {
    cpu_profiling: true,
    memory_profiling: true,
    gpu_profiling: true,
    flame_graph: true,
});

let results = runner.run_all()?;

// Generate profiling report
let profiling_report = ProfilingReportGenerator::new()
    .with_flame_graphs()
    .with_bottleneck_analysis()
    .generate(&results)?;
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Run Benchmarks
      run: |
        cargo bench --package torsh-benches -- --output-format json > benchmark_results.json
        
    - name: Generate Performance Report
      run: |
        cargo run --bin benchmark-ci-reporter -- \
          --input benchmark_results.json \
          --output performance_report.html \
          --baseline main
          
    - name: Upload Results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: |
          benchmark_results.json
          performance_report.html
```

### Regression Detection

```rust
use torsh_benches::regression::*;

let detector = RegressionDetector::new()
    .with_baseline_from_git("main")
    .with_threshold(0.05) // 5% regression threshold
    .with_statistical_analysis();

let current_results = run_benchmarks()?;
let regression_analysis = detector.analyze(&current_results)?;

if regression_analysis.has_regressions() {
    println!("Performance regressions detected:");
    for regression in regression_analysis.regressions() {
        println!("  {}: {:.2}% slower", regression.benchmark, regression.slowdown * 100.0);
    }
    std::process::exit(1);
}
```

## Report Generation

### HTML Reports

```rust
use torsh_benches::reporting::*;

let results = run_benchmarks()?;

let html_report = HtmlReportGenerator::new()
    .with_theme(ReportTheme::Dark)
    .with_interactive_charts()
    .with_comparison_tables()
    .with_performance_trends()
    .generate(&results)?;

html_report.save("performance_report.html")?;
```

### JSON Export

```rust
let json_report = JsonReportGenerator::new()
    .with_detailed_metrics()
    .with_system_info()
    .generate(&results)?;

json_report.save("benchmark_results.json")?;
```

### CSV Export

```rust
let csv_report = CsvReportGenerator::new()
    .with_summary_statistics()
    .with_raw_timings()
    .generate(&results)?;

csv_report.save("benchmark_data.csv")?;
```

## Performance Tips

### Memory Management

1. **Pre-allocate tensors** when possible to avoid allocation overhead
2. **Use appropriate batch sizes** for your hardware
3. **Enable memory pooling** for reduced fragmentation
4. **Profile memory usage** to identify leaks

### GPU Optimization

1. **Warm up the GPU** before timing operations
2. **Use appropriate precision** (FP16 vs FP32)
3. **Optimize data transfer** between CPU and GPU
4. **Leverage tensor cores** on compatible hardware

### CPU Optimization

1. **Use SIMD instructions** where available
2. **Optimize cache usage** with appropriate access patterns
3. **Leverage multi-threading** for parallel operations
4. **Consider NUMA topology** on multi-socket systems

## Troubleshooting

### Common Issues

**Inconsistent Results**
- Ensure system is idle during benchmarking
- Disable CPU frequency scaling
- Use sufficient warm-up iterations

**Memory Issues**
- Reduce batch sizes for large models
- Enable gradient checkpointing
- Use memory-efficient attention

**GPU Issues**
- Check CUDA driver compatibility
- Verify sufficient GPU memory
- Monitor thermal throttling

### Debugging

```rust
use torsh_benches::debug::*;

let debug_runner = DebugBenchRunner::new()
    .with_verbose_logging()
    .with_memory_tracking()
    .with_error_reporting();

let results = debug_runner.run_with_debugging(benchmarks)?;
```

## Best Practices

1. **Consistent Environment**: Run benchmarks on dedicated hardware
2. **Statistical Significance**: Use sufficient iterations for reliable results
3. **Baseline Comparisons**: Always compare against known baselines
4. **Documentation**: Document benchmark configurations and environments
5. **Automation**: Integrate benchmarks into CI/CD pipelines
6. **Monitoring**: Track performance trends over time

## Extending the Benchmark Suite

### Adding New Benchmarks

1. Implement the `Benchmark` trait
2. Add configuration options
3. Include in the benchmark runner
4. Add tests and documentation

### Contributing Guidelines

1. Follow existing code patterns
2. Include comprehensive tests
3. Document configuration options
4. Provide usage examples
5. Update this guide with new features

## Resources

- [ToRSh Documentation](https://docs.torsh.dev)
- [Performance Optimization Guide](OPTIMIZATION_TIPS.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [API Reference](https://docs.rs/torsh-benches)