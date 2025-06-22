# ToRSh Benchmarks

This crate provides comprehensive benchmarking utilities for ToRSh, designed to measure performance across different operations, compare with other tensor libraries, and track performance regressions.

## Features

- **Tensor Operations**: Benchmarks for creation, arithmetic, matrix multiplication, and reductions
- **Neural Networks**: Performance tests for layers, activations, loss functions, and optimizers
- **Memory Operations**: Memory allocation, copying, and management benchmarks
- **Comparisons**: Side-by-side performance comparisons with other libraries
- **System Metrics**: CPU, memory, and performance profiling
- **Regression Detection**: Track performance changes over time

## Usage

### Running Benchmarks

Run all benchmarks:
```bash
cargo bench
```

Run specific benchmark suites:
```bash
cargo bench tensor_operations
cargo bench neural_networks
cargo bench memory_operations
```

### Custom Benchmarks

```rust
use torsh_benches::prelude::*;
use torsh_tensor::creation::*;

// Create a simple benchmark
let mut runner = BenchRunner::new();

let config = BenchConfig::new("my_benchmark")
    .with_sizes(vec![64, 128, 256])
    .with_dtypes(vec![DType::F32]);

let bench = benchmark!(
    "tensor_addition",
    |size| {
        let a = rand::<f32>(&[size, size]);
        let b = rand::<f32>(&[size, size]);
        (a, b)
    },
    |(a, b)| a.add(b).unwrap()
);

runner.run_benchmark(bench, &config);
```

### Performance Comparisons

Enable external library comparisons:
```bash
cargo bench --features compare-external
```

### System Metrics Collection

```rust
use torsh_benches::metrics::*;

let mut collector = MetricsCollector::new();
collector.start();

// Run your code here

let metrics = collector.stop();
println!("Peak memory usage: {:.2} MB", metrics.memory_stats.peak_usage_mb);
println!("CPU utilization: {:.1}%", metrics.cpu_stats.average_usage_percent);
```

### Performance Profiling

```rust
use torsh_benches::metrics::*;

let mut profiler = PerformanceProfiler::new();

profiler.begin_event("matrix_multiply");
// Your code here
profiler.end_event("matrix_multiply");

let report = profiler.generate_report();
profiler.export_chrome_trace("profile.json").unwrap();
```

## Benchmark Categories

### Tensor Operations
- **Creation**: `zeros()`, `ones()`, `rand()`, `eye()`
- **Arithmetic**: element-wise operations, broadcasting
- **Linear Algebra**: matrix multiplication, decompositions
- **Reductions**: sum, mean, min, max, norm
- **Indexing**: slicing, gathering, scattering

### Neural Network Operations
- **Layers**: Linear, Conv2d, BatchNorm, Dropout
- **Activations**: ReLU, Sigmoid, Tanh, Softmax
- **Loss Functions**: MSE, Cross-entropy, BCE
- **Optimizers**: SGD, Adam, RMSprop

### Memory Operations
- **Allocation**: buffer creation and management
- **Transfer**: host-to-device, device-to-host
- **Synchronization**: device synchronization overhead

### Data Loading
- **Dataset**: reading and preprocessing
- **DataLoader**: batching and shuffling
- **Transforms**: image and tensor transformations

## Configuration

Benchmark behavior can be customized through `BenchConfig`:

```rust
let config = BenchConfig::new("custom_benchmark")
    .with_sizes(vec![32, 64, 128, 256, 512, 1024])
    .with_dtypes(vec![DType::F16, DType::F32, DType::F64])
    .with_timing(
        Duration::from_millis(500),  // warmup time
        Duration::from_secs(2),      // measurement time
    )
    .with_memory_measurement()
    .with_metadata("device", "cpu");
```

## Output Formats

Benchmark results can be exported in multiple formats:

- **HTML Reports**: Interactive charts and tables
- **CSV**: Raw data for further analysis
- **JSON**: Structured data for automation
- **Chrome Tracing**: Performance profiling visualization

## Performance Comparison

When the `compare-external` feature is enabled, benchmarks will run against:

- **ndarray**: Rust tensor library
- **nalgebra**: Linear algebra library
- Additional libraries can be added

Results show relative performance:
```
| Operation | Library | Size | Time (μs) | Speedup vs ToRSh |
|-----------|---------|------|-----------|------------------|
| MatMul    | torsh   | 512  | 234.5     | 1.00x            |
| MatMul    | ndarray | 512  | 289.1     | 0.81x            |
```

## Regression Detection

Track performance over time:

```rust
let mut detector = RegressionDetector::new(0.1); // 10% threshold
detector.load_baseline("baseline_results.json").unwrap();

let regressions = detector.check_regression(&current_results);
for regression in regressions {
    println!("Regression detected in {}: {:.2}x slower", 
             regression.benchmark, regression.slowdown_factor);
}
```

## Environment Setup

For consistent benchmarking results:

```rust
use torsh_benches::utils::Environment;

Environment::setup_for_benchmarking();
// Run benchmarks
Environment::restore_environment();
```

This will:
- Set high process priority
- Disable CPU frequency scaling (if possible)
- Set CPU affinity for consistent results

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Run benchmarks
  run: cargo bench --features compare-external

- name: Upload benchmark results
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'criterion'
    output-file-path: target/criterion/reports/index.html
```

## Development

To add new benchmarks:

1. Implement the `Benchmarkable` trait for your operation
2. Add benchmark configurations
3. Create criterion benchmark functions
4. Update the benchmark group

See existing benchmarks in `benches/` for examples.

## Performance Tips

- Use `black_box()` to prevent compiler optimizations
- Pre-allocate test data outside timing loops
- Use consistent input sizes across runs
- Monitor system load during benchmarking
- Use statistical analysis for noisy measurements

## Troubleshooting

### Inconsistent Results
- Ensure system is idle during benchmarking
- Check CPU thermal throttling
- Use fixed CPU frequency if possible
- Increase measurement time for noisy benchmarks

### Memory Issues
- Monitor available memory during large benchmarks
- Use memory profiling to detect leaks
- Consider batch size limitations

### Compilation Errors
- Ensure all ToRSh crates are up to date
- Check feature flag compatibility
- Verify external library versions