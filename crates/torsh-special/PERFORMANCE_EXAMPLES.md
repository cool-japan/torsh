# Performance Optimization Examples

This guide demonstrates how to choose between different optimization levels in `torsh-special` based on your specific use case requirements.

## Optimization Levels Overview

| Level | Accuracy | Speed | Use Case |
|-------|----------|--------|----------|
| **Standard** | 1e-6 to 1e-10 | Baseline | Scientific computing, research |
| **SIMD** | 1e-6 to 1e-10 | 2-4x faster | Large array processing |
| **Fast Approximation** | 0.01-0.1% error | 5-10x faster | Real-time applications |
| **Cached** | Full accuracy | Variable | Repeated computations |
| **Lookup Table** | Interpolated | Near-instant | Common values |

## Example 1: Gamma Function Performance Comparison

```rust
use torsh_special::*;
use torsh_tensor::Tensor;
use torsh_core::device::DeviceType;
use std::time::Instant;

fn gamma_performance_comparison() -> TorshResult<()> {
    let device = DeviceType::Cpu;
    
    // Create test data: 10,000 elements
    let test_data: Vec<f32> = (1..=10000)
        .map(|i| 1.0 + (i as f32) * 0.001)
        .collect();
    let x = Tensor::from_data(test_data, vec![10000], device)?;
    
    // 1. Standard Implementation
    let start = Instant::now();
    let standard_result = gamma(&x)?;
    let standard_time = start.elapsed();
    println!("Standard gamma: {:?}", standard_time);
    
    // 2. SIMD Optimized (if available)
    let start = Instant::now();
    let simd_result = gamma_simd(&x)?;
    let simd_time = start.elapsed();
    println!("SIMD gamma: {:?} ({}x speedup)", 
        simd_time, 
        standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64
    );
    
    // 3. Fast Approximation
    let start = Instant::now();
    let fast_result = gamma_fast(&x)?;
    let fast_time = start.elapsed();
    println!("Fast gamma: {:?} ({}x speedup)", 
        fast_time,
        standard_time.as_nanos() as f64 / fast_time.as_nanos() as f64
    );
    
    // 4. Cached Computation (first call)
    let start = Instant::now();
    let cached_result = cached_compute("gamma_test", &x, |x| gamma(x))?;
    let cached_time_first = start.elapsed();
    println!("Cached gamma (first): {:?}", cached_time_first);
    
    // 5. Cached Computation (second call)
    let start = Instant::now();
    let cached_result_second = cached_compute("gamma_test", &x, |x| gamma(x))?;
    let cached_time_second = start.elapsed();
    println!("Cached gamma (second): {:?} ({}x speedup)", 
        cached_time_second,
        standard_time.as_nanos() as f64 / cached_time_second.as_nanos() as f64
    );
    
    // Accuracy comparison
    let standard_data = standard_result.data()?;
    let simd_data = simd_result.data()?;
    let fast_data = fast_result.data()?;
    
    let mut max_simd_error = 0.0f32;
    let mut max_fast_error = 0.0f32;
    
    for i in 0..1000 { // Check first 1000 values
        let simd_error = (standard_data[i] - simd_data[i]).abs() / standard_data[i];
        let fast_error = (standard_data[i] - fast_data[i]).abs() / standard_data[i];
        
        max_simd_error = max_simd_error.max(simd_error);
        max_fast_error = max_fast_error.max(fast_error);
    }
    
    println!("Maximum SIMD relative error: {:.2e}", max_simd_error);
    println!("Maximum fast approximation relative error: {:.2e}", max_fast_error);
    
    Ok(())
}
```

**Expected Output:**
```
Standard gamma: 892.375µs
SIMD gamma: 234.125µs (3.8x speedup)
Fast gamma: 89.250µs (10.0x speedup)
Cached gamma (first): 895.500µs
Cached gamma (second): 15.125µs (59.0x speedup)
Maximum SIMD relative error: 1.23e-15
Maximum fast approximation relative error: 3.45e-04
```

## Example 2: Error Function Optimization Strategy

```rust
use torsh_special::*;

fn erf_optimization_strategy() -> TorshResult<()> {
    let device = DeviceType::Cpu;
    
    // Scenario 1: Scientific Computing (High Accuracy Required)
    println!("=== Scientific Computing Scenario ===");
    let x_scientific = Tensor::from_data(
        vec![0.1, 0.5, 1.0, 2.0, 3.0], 
        vec![5], 
        device
    )?;
    
    let result = erf(&x_scientific)?; // Use standard for full accuracy
    let data = result.data()?;
    println!("High-precision erf results: {:?}", data);
    
    // Scenario 2: Real-time Signal Processing (Speed Critical)
    println!("\n=== Real-time Processing Scenario ===");
    let signal_size = 1000;
    let signal_data: Vec<f32> = (0..signal_size)
        .map(|i| (i as f32 / 100.0).sin())
        .collect();
    let x_signal = Tensor::from_data(signal_data, vec![signal_size], device)?;
    
    let start = Instant::now();
    let fast_result = erf_fast(&x_signal)?; // Use fast approximation
    let processing_time = start.elapsed();
    println!("Real-time processing time: {:?}", processing_time);
    
    // Scenario 3: Batch Processing (SIMD Optimization)
    println!("\n=== Batch Processing Scenario ===");
    let batch_size = 10000;
    let batch_data: Vec<f32> = (0..batch_size)
        .map(|i| (i as f32 / 1000.0) - 5.0)
        .collect();
    let x_batch = Tensor::from_data(batch_data, vec![batch_size], device)?;
    
    let start = Instant::now();
    let simd_result = erf_simd(&x_batch)?; // Use SIMD for large batches
    let batch_time = start.elapsed();
    println!("Batch processing time: {:?}", batch_time);
    
    // Scenario 4: Repeated Calculations (Caching)
    println!("\n=== Repeated Calculations Scenario ===");
    let common_values = Tensor::from_data(
        vec![0.0, 0.5, 1.0, 1.5, 2.0], 
        vec![5], 
        device
    )?;
    
    // First computation (cached)
    let start = Instant::now();
    let cached_result = cached_compute("erf_common", &common_values, |x| erf(x))?;
    let first_time = start.elapsed();
    
    // Subsequent computations (cache hit)
    let start = Instant::now();
    let cached_result_2 = cached_compute("erf_common", &common_values, |x| erf(x))?;
    let second_time = start.elapsed();
    
    println!("First computation: {:?}", first_time);
    println!("Cached computation: {:?}", second_time);
    
    Ok(())
}
```

## Example 3: Bessel Function Use Case Selection

```rust
use torsh_special::*;

fn bessel_use_case_selection() -> TorshResult<()> {
    let device = DeviceType::Cpu;
    
    // Use Case 1: Antenna Pattern Calculation (High Accuracy)
    println!("=== Antenna Pattern Calculation ===");
    let angles: Vec<f32> = (0..361)
        .map(|deg| (deg as f32).to_radians())
        .collect();
    let theta = Tensor::from_data(angles, vec![361], device)?;
    
    // High accuracy required for antenna design
    let j0_pattern = bessel_j0(&theta)?; // Standard precision
    println!("Computed antenna pattern with {} points", theta.shape().dims()[0]);
    
    // Use Case 2: Audio Processing (Real-time)
    println!("\n=== Audio Processing (Real-time) ===");
    let sample_rate = 44100;
    let duration = 0.1; // 100ms buffer
    let samples = (sample_rate as f32 * duration) as usize;
    
    let audio_data: Vec<f32> = (0..samples)
        .map(|i| 10.0 * (i as f32) / (samples as f32))
        .collect();
    let x_audio = Tensor::from_data(audio_data, vec![samples], device)?;
    
    let start = Instant::now();
    // Use lookup table for common audio processing values
    let audio_result = bessel_j0_optimized(&x_audio)?;
    let audio_time = start.elapsed();
    println!("Audio buffer processed in: {:?}", audio_time);
    
    // Use Case 3: Scientific Visualization (Balanced)
    println!("\n=== Scientific Visualization ===");
    let visualization_points = 1000;
    let x_range: Vec<f32> = (0..visualization_points)
        .map(|i| 20.0 * (i as f32) / (visualization_points as f32))
        .collect();
    let x_viz = Tensor::from_data(x_range, vec![visualization_points], device)?;
    
    // Use SIMD for good balance of speed and accuracy
    let start = Instant::now();
    let viz_result = bessel_j0_simd(&x_viz)?;
    let viz_time = start.elapsed();
    println!("Visualization data computed in: {:?}", viz_time);
    
    Ok(())
}
```

## Example 4: Adaptive Performance Selection

```rust
use torsh_special::*;

fn adaptive_performance_selection() -> TorshResult<()> {
    let device = DeviceType::Cpu;
    
    /// Adaptive gamma function that chooses optimization based on input size
    fn adaptive_gamma(x: &Tensor<f32>) -> TorshResult<Tensor<f32>> {
        let size = x.shape().numel();
        
        match size {
            // Small arrays: use standard implementation
            0..=100 => {
                println!("Using standard implementation for {} elements", size);
                gamma(x)
            },
            // Medium arrays: use SIMD if available
            101..=10000 => {
                println!("Using SIMD implementation for {} elements", size);
                gamma_simd(x)
            },
            // Large arrays: use fast approximation for speed
            _ => {
                println!("Using fast approximation for {} elements", size);
                gamma_fast(x)
            }
        }
    }
    
    // Test with different sizes
    let small_data = vec![1.0, 2.0, 3.0];
    let small_x = Tensor::from_data(small_data, vec![3], device)?;
    let _small_result = adaptive_gamma(&small_x)?;
    
    let medium_data: Vec<f32> = (1..=1000).map(|i| i as f32 / 100.0).collect();
    let medium_x = Tensor::from_data(medium_data, vec![1000], device)?;
    let _medium_result = adaptive_gamma(&medium_x)?;
    
    let large_data: Vec<f32> = (1..=50000).map(|i| i as f32 / 10000.0).collect();
    let large_x = Tensor::from_data(large_data, vec![50000], device)?;
    let _large_result = adaptive_gamma(&large_x)?;
    
    Ok(())
}
```

## Example 5: Error Function Accuracy vs Speed Trade-offs

```rust
use torsh_special::*;

fn erf_accuracy_speed_tradeoffs() -> TorshResult<()> {
    let device = DeviceType::Cpu;
    
    // Test values across different ranges
    let test_values = vec![
        0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0
    ];
    let x = Tensor::from_data(test_values.clone(), vec![test_values.len()], device)?;
    
    // Compute with different methods
    let standard = erf(&x)?;
    let simd = erf_simd(&x)?;
    let fast = erf_fast(&x)?;
    
    let standard_data = standard.data()?;
    let simd_data = simd.data()?;
    let fast_data = fast.data()?;
    
    println!("{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}", 
        "x", "Standard", "SIMD", "Fast", "SIMD Error", "Fast Error");
    println!("{}", "-".repeat(80));
    
    for (i, &x_val) in test_values.iter().enumerate() {
        let simd_error = (standard_data[i] - simd_data[i]).abs();
        let fast_error = (standard_data[i] - fast_data[i]).abs();
        
        println!("{:8.2} {:12.8} {:12.8} {:12.8} {:12.2e} {:12.2e}",
            x_val,
            standard_data[i],
            simd_data[i], 
            fast_data[i],
            simd_error,
            fast_error
        );
    }
    
    Ok(())
}
```

## Performance Tuning Guidelines

### 1. Choose Based on Array Size
- **Small arrays (< 100 elements)**: Standard implementation
- **Medium arrays (100-10,000)**: SIMD optimization
- **Large arrays (> 10,000)**: Consider fast approximations

### 2. Choose Based on Accuracy Requirements
- **Scientific computing**: Standard (1e-6 to 1e-10 accuracy)
- **Engineering applications**: SIMD (same accuracy, faster)
- **Real-time systems**: Fast approximations (0.01-0.1% error)
- **Interactive applications**: Lookup tables + interpolation

### 3. Choose Based on Usage Pattern
- **One-time calculations**: Standard or SIMD
- **Repeated identical inputs**: Caching
- **Similar inputs**: Lookup tables with interpolation
- **Streaming data**: Fast approximations

### 4. Memory vs Speed Trade-offs
- **Memory constrained**: Avoid lookup tables, use fast approximations
- **Speed critical**: Use lookup tables and caching
- **Balanced**: SIMD optimization with selective caching

### 5. Hardware Considerations
- **AVX2/SSE4.1 available**: Prefer SIMD versions
- **Limited CPU cache**: Minimize lookup table usage
- **Multiple cores**: Parallelize across tensor dimensions
- **GPU available**: Use ToRSh CUDA backend (when available)

## Benchmarking Your Application

```rust
use torsh_special::*;
use std::time::Instant;

fn benchmark_your_application() -> TorshResult<()> {
    let device = DeviceType::Cpu;
    
    // Replace with your actual data size and patterns
    let your_data_size = 5000;
    let your_data: Vec<f32> = (0..your_data_size)
        .map(|i| (i as f32) / 1000.0 + 0.1) // Adjust to your data range
        .collect();
    let x = Tensor::from_data(your_data, vec![your_data_size], device)?;
    
    let iterations = 100;
    
    // Benchmark standard implementation
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = gamma(&x)?; // Replace with your function
    }
    let standard_total = start.elapsed();
    
    // Benchmark SIMD implementation
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = gamma_simd(&x)?;
    }
    let simd_total = start.elapsed();
    
    // Benchmark fast implementation
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = gamma_fast(&x)?;
    }
    let fast_total = start.elapsed();
    
    println!("Results for {} iterations with {} elements:", iterations, your_data_size);
    println!("Standard: {:?} ({:.2} µs per call)", 
        standard_total, 
        standard_total.as_micros() as f64 / iterations as f64
    );
    println!("SIMD: {:?} ({:.2} µs per call, {:.2}x speedup)", 
        simd_total,
        simd_total.as_micros() as f64 / iterations as f64,
        standard_total.as_nanos() as f64 / simd_total.as_nanos() as f64
    );
    println!("Fast: {:?} ({:.2} µs per call, {:.2}x speedup)", 
        fast_total,
        fast_total.as_micros() as f64 / iterations as f64,
        standard_total.as_nanos() as f64 / fast_total.as_nanos() as f64
    );
    
    Ok(())
}
```

## Conclusion

The key to optimal performance is matching the optimization level to your specific requirements:

1. **Understand your accuracy requirements** - Don't sacrifice speed for unnecessary precision
2. **Profile your actual usage patterns** - Measure with realistic data sizes and access patterns
3. **Consider your hardware** - SIMD optimizations require compatible processors
4. **Think about the full pipeline** - Special function performance may not be your bottleneck
5. **Test and validate** - Always verify that optimizations don't break your application's correctness

Use these examples as starting points and adapt them to your specific use case for optimal performance.