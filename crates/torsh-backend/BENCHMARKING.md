# ToRSh Backend Performance Benchmarking Guide

This document describes the comprehensive performance benchmarking suite for the ToRSh backend system.

## Overview

The ToRSh backend includes extensive performance benchmarks to measure and validate the performance characteristics of different backend operations, optimizations, and algorithms.

## Running Benchmarks

### Prerequisites

Ensure you have the latest stable Rust toolchain installed:

```bash
rustup update stable
```

### Basic Benchmark Execution

Run all benchmarks:
```bash
cargo bench
```

Run specific benchmark suites:
```bash
# Run general backend benchmarks
cargo bench --bench backend_benchmarks

# Run CPU-specific benchmarks  
cargo bench --bench cpu_benchmarks
```

Run specific benchmark groups:
```bash
# Memory allocation benchmarks
cargo bench --bench backend_benchmarks memory_allocation

# SIMD operation benchmarks
cargo bench --bench cpu_benchmarks cpu_simd

# Auto-tuning benchmarks
cargo bench --bench backend_benchmarks autotuning
```

### Advanced Benchmark Options

Generate detailed HTML reports:
```bash
cargo bench -- --output-format html
```

Run benchmarks with custom sample sizes:
```bash
cargo bench -- --sample-size 100
```

Run benchmarks for specific durations:
```bash
cargo bench -- --measurement-time 30
```

Save baseline for performance regression testing:
```bash
cargo bench -- --save-baseline main
```

Compare against baseline:
```bash
cargo bench -- --baseline main
```

## Benchmark Categories

### 1. General Backend Benchmarks (`backend_benchmarks`)

#### Memory Allocation
- Tests allocation performance for different buffer sizes (1KB - 1MB)
- Measures allocation latency and throughput
- Validates memory pool efficiency

#### Device Operations
- CPU device creation performance
- Device feature query latency
- Device capability reporting overhead

#### Kernel Operations  
- Kernel creation and compilation time
- Kernel metadata processing
- Entry point resolution

#### SIMD Operations
- SIMD threshold detection performance
- Vectorization decision overhead
- SIMD utilization efficiency

#### Cross-Backend Validation
- Validator creation overhead
- Floating-point comparison performance
- Cross-backend consistency checking

#### Auto-tuning System
- Configuration generation performance
- Performance measurement overhead
- Cache hit/miss ratios

#### Quantization Operations
- INT8 quantization/dequantization performance  
- Parameter optimization efficiency
- Quantization accuracy vs. speed trade-offs

#### FFT Operations
- FFT plan creation for different sizes
- 1D/2D/3D transform performance
- Memory usage optimization

#### Sparse Operations
- Sparse matrix creation and format conversion
- SpMV (Sparse Matrix-Vector) multiplication
- CSR/COO format efficiency

#### Profiler Operations
- Event creation and completion overhead
- Profiling data collection efficiency
- Report generation performance

### 2. CPU-Specific Benchmarks (`cpu_benchmarks`)

#### SIMD Operations
- Element-wise operations (add, multiply, dot product)
- Throughput measurements for different array sizes
- SIMD vs. scalar performance comparison

#### Platform Optimization
- CPU information detection performance
- Microarchitecture-specific optimizations
- Feature detection overhead

#### Memory Operations
- Memory allocation with different strategies
- Cache-aligned access patterns
- NUMA-aware allocation performance

#### Feature Detection
- CPU feature detection latency
- Architecture information gathering
- Capability enumeration performance

#### Convolution Operations
- Convolution configuration performance
- Algorithm selection efficiency
- Optimization strategy evaluation

#### RNN Operations
- RNN configuration overhead
- Weight buffer size calculation
- Layer setup performance

#### Optimized Kernels
- Hand-optimized vs. generic implementations
- Matrix multiplication performance
- Parallel reduction efficiency

## Performance Targets

### Memory Operations
- **Allocation Latency**: < 10μs for buffers up to 1MB
- **Throughput**: > 1GB/s for sequential memory access
- **Cache Efficiency**: > 95% hit rate for repeated allocations

### SIMD Operations
- **Vectorization Speedup**: 2-8x over scalar operations
- **Threshold Detection**: < 100ns overhead
- **Memory Bandwidth**: > 80% of theoretical peak

### Device Operations
- **Device Creation**: < 1ms for CPU devices
- **Feature Query**: < 100ns per query
- **Capability Reporting**: < 10μs complete enumeration

### Kernel Operations
- **Kernel Creation**: < 5ms for simple kernels
- **Compilation Cache**: 99%+ hit rate for repeated kernels
- **Metadata Processing**: < 1ms per kernel

## Interpreting Results

### Key Metrics

1. **Latency**: Time taken for single operations
2. **Throughput**: Operations or data processed per second  
3. **Memory Usage**: Peak and average memory consumption
4. **Cache Efficiency**: Hit/miss ratios for cached operations
5. **Scalability**: Performance scaling with data size/thread count

### Performance Analysis

#### Good Performance Indicators
- Consistent latency across runs (low variance)
- Linear or better scaling with data size
- High cache hit rates (>95%)
- Efficient memory utilization
- Low overhead for small operations

#### Performance Issues to Investigate
- High latency variance (>20%)
- Sublinear scaling with data size
- Low cache efficiency (<80%)
- Memory leaks or excessive allocation
- High overhead for simple operations

### Regression Detection

Establish performance baselines:
```bash
# Save current performance as baseline
cargo bench -- --save-baseline current

# After changes, compare performance
cargo bench -- --baseline current
```

Monitor for:
- >5% regression in critical path operations
- >10% increase in memory usage
- Significant variance increases
- Cache efficiency degradation

## Continuous Integration

### Automated Benchmarking

Include benchmark validation in CI:
```yaml
- name: Run Performance Benchmarks
  run: |
    cargo bench --bench backend_benchmarks -- --output-format json > bench_results.json
    cargo bench --bench cpu_benchmarks -- --output-format json >> bench_results.json
```

### Performance Monitoring

Set up automated regression detection:
- Run benchmarks on every PR
- Compare against main branch baseline
- Flag significant performance regressions
- Track performance trends over time

## Optimization Guidelines

### General Principles
1. Measure before optimizing
2. Focus on critical path operations
3. Optimize for common use cases
4. Validate optimizations with benchmarks
5. Consider memory vs. compute trade-offs

### Backend-Specific Optimizations
- **CPU**: Maximize SIMD utilization and cache efficiency
- **GPU**: Optimize memory access patterns and occupancy
- **Memory**: Minimize allocations and fragmentation
- **Cross-backend**: Reduce synchronization overhead

## Troubleshooting

### Common Issues

#### Inconsistent Results
- Ensure stable system load during benchmarking
- Disable CPU frequency scaling
- Close unnecessary applications
- Run multiple iterations to establish confidence intervals

#### Memory Issues
- Monitor for memory leaks during long-running benchmarks
- Validate proper cleanup in benchmark teardown
- Check for excessive temporary allocations

#### Performance Degradation
- Profile to identify bottlenecks
- Check for unintended algorithmic changes
- Validate compiler optimizations are enabled
- Verify SIMD instructions are being generated

### Debug Mode
Run benchmarks with additional debug information:
```bash
RUST_LOG=debug cargo bench --bench backend_benchmarks
```

### Profiling Integration
Combine with profiling tools for detailed analysis:
```bash
# Profile with perf
perf record --call-graph=dwarf cargo bench --bench cpu_benchmarks
perf report

# Profile with Instruments (macOS)
xcrun xctrace record --template "Time Profiler" --launch -- cargo bench
```

## Contributing

### Adding New Benchmarks

1. Create benchmark functions following existing patterns
2. Use appropriate measurement durations and sample sizes
3. Include multiple data sizes for scalability testing
4. Add throughput measurements where applicable
5. Document expected performance characteristics

### Benchmark Best Practices

1. Use `black_box()` to prevent compiler optimizations
2. Set appropriate measurement times for stable results
3. Include warmup iterations for JIT-compiled code
4. Test multiple input sizes and patterns
5. Validate benchmark correctness with unit tests

### Performance Regression Policy

1. All performance regressions >5% require explanation
2. Critical path regressions >10% block merge
3. Memory usage increases >20% require optimization
4. New features must include relevant benchmarks
5. Optimizations must be validated with before/after benchmarks

## Resources

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
- [AMD μProf](https://developer.amd.com/amd-uprof/)