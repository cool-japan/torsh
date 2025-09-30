//! Interactive examples demonstrating different optimization levels in torsh-special
//!
//! This example shows how to use different optimization strategies:
//! 1. Standard functions - balanced accuracy and performance
//! 2. SIMD optimized functions - high performance for large tensors
//! 3. Fast approximations - maximum speed with controlled accuracy loss
//! 4. Smart caching - automatic optimization for repeated computations

use std::time::Instant;
use torsh_core::device::DeviceType;
use torsh_special::*;
use torsh_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TorSh Special Functions - Optimization Level Examples\n");

    let device = DeviceType::Cpu;

    // Create test data
    let small_data = vec![0.1, 0.5, 1.0, 2.0, 5.0];
    let large_data: Vec<f32> = (0..10000).map(|i| (i as f32) * 0.001).collect();

    let small_tensor = Tensor::from_data(small_data.clone(), vec![5], device)?;
    let large_tensor = Tensor::from_data(large_data, vec![10000], device)?;

    println!("📊 Test Data:");
    println!("  Small tensor: 5 elements [0.1, 0.5, 1.0, 2.0, 5.0]");
    println!("  Large tensor: 10,000 elements [0.0, 0.001, 0.002, ..., 9.999]\n");

    // 1. Standard Functions
    println!("🔧 1. STANDARD FUNCTIONS (Default - Balanced accuracy/performance)");
    demo_standard_functions(&small_tensor)?;

    // 2. SIMD Optimized Functions
    println!("\n⚡ 2. SIMD OPTIMIZED FUNCTIONS (High performance for large data)");
    demo_simd_functions(&large_tensor)?;

    // 3. Fast Approximations
    println!("\n🏃 3. FAST APPROXIMATIONS (Maximum speed, controlled accuracy loss)");
    demo_fast_approximations(&small_tensor)?;

    // 4. Smart Caching
    println!("\n🧠 4. SMART CACHING (Automatic optimization for repeated computations)");
    demo_smart_caching(&small_tensor)?;

    // 5. Performance Comparison
    println!("\n📈 5. PERFORMANCE COMPARISON");
    performance_comparison(&large_tensor)?;

    // 6. When to Use Each Level
    println!("\n💡 6. OPTIMIZATION LEVEL SELECTION GUIDE");
    optimization_guide();

    Ok(())
}

fn demo_standard_functions(tensor: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Using standard implementations with full precision:");

    // Gamma function
    let gamma_result = gamma::gamma(tensor)?;
    let gamma_data = gamma_result.data()?;
    println!(
        "  γ(x): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        gamma_data[0], gamma_data[1], gamma_data[2], gamma_data[3], gamma_data[4]
    );

    // Error function
    let erf_result = error_functions::erf(tensor)?;
    let erf_data = erf_result.data()?;
    println!(
        "  erf(x): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        erf_data[0], erf_data[1], erf_data[2], erf_data[3], erf_data[4]
    );

    // Bessel function
    let bessel_result = bessel::bessel_j0(tensor)?;
    let bessel_data = bessel_result.data()?;
    println!(
        "  J₀(x): [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
        bessel_data[0], bessel_data[1], bessel_data[2], bessel_data[3], bessel_data[4]
    );

    Ok(())
}

fn demo_simd_functions(tensor: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Using SIMD-accelerated implementations for large tensors:");

    let start = Instant::now();

    // SIMD Gamma (if available)
    #[cfg(feature = "simd")]
    {
        use torsh_special::simd_optimizations::*;

        let gamma_result = gamma_simd(tensor)?;
        let gamma_time = start.elapsed();
        println!(
            "  ✓ SIMD γ(x): computed 10,000 values in {:.2?}",
            gamma_time
        );

        // SIMD Error function
        let start_erf = Instant::now();
        let erf_result = erf_simd(tensor)?;
        let erf_time = start_erf.elapsed();
        println!(
            "  ✓ SIMD erf(x): computed 10,000 values in {:.2?}",
            erf_time
        );

        // Show first few results
        let gamma_data = gamma_result.data()?;
        let erf_data = erf_result.data()?;
        println!(
            "  Sample results - γ: [{:.4}, {:.4}, {:.4}...], erf: [{:.4}, {:.4}, {:.4}...]",
            gamma_data[0], gamma_data[1], gamma_data[2], erf_data[0], erf_data[1], erf_data[2]
        );
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("  ⚠️ SIMD features not available in this build");
        println!("  To enable: cargo build --features simd");

        // Fallback to standard functions with timing
        let _gamma_result = gamma::gamma(tensor)?;
        let gamma_time = start.elapsed();
        println!(
            "  Standard γ(x): computed 10,000 values in {:.2?}",
            gamma_time
        );
    }

    Ok(())
}

fn demo_fast_approximations(tensor: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Using fast approximations (0.01-0.1% accuracy trade-off):");

    use torsh_special::fast_approximations::*;

    let start = Instant::now();

    // Fast gamma approximation
    let gamma_fast_result = gamma_fast(tensor)?;
    let gamma_fast_time = start.elapsed();
    let gamma_fast_data = gamma_fast_result.data()?;

    // Compare with standard
    let gamma_std_result = gamma::gamma(tensor)?;
    let gamma_std_data = gamma_std_result.data()?;

    println!("  Fast γ(x): computed in {:.2?}", gamma_fast_time);
    println!("  Accuracy comparison:");
    for i in 0..5 {
        let error = ((gamma_fast_data[i] - gamma_std_data[i]) / gamma_std_data[i] * 100.0).abs();
        println!(
            "    x={:.1}: fast={:.4}, std={:.4}, error={:.2}%",
            tensor.data()?[i],
            gamma_fast_data[i],
            gamma_std_data[i],
            error
        );
    }

    // Fast error function
    let start_erf = Instant::now();
    let _erf_fast_result = erf_fast(tensor)?;
    let erf_fast_time = start_erf.elapsed();
    println!("  Fast erf(x): computed in {:.2?}", erf_fast_time);

    Ok(())
}

fn demo_smart_caching(tensor: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Using smart caching for repeated expensive computations:");

    use torsh_special::smart_caching::*;

    // Clear cache for clean demo
    clear_cache();

    let data = tensor.data()?;

    // First computation (cache miss)
    let start1 = Instant::now();
    let result1 = cached_compute(data[0] as f64, function_ids::GAMMA, || {
        let single_tensor = Tensor::from_data(vec![data[0]], vec![1], tensor.device()).unwrap();
        let gamma_result = gamma::gamma(&single_tensor).unwrap();
        gamma_result.data().unwrap()[0] as f64
    });
    let time1 = start1.elapsed();

    // Second computation (cache hit)
    let start2 = Instant::now();
    let result2 = cached_compute(data[0] as f64, function_ids::GAMMA, || {
        let single_tensor = Tensor::from_data(vec![data[0]], vec![1], tensor.device()).unwrap();
        let gamma_result = gamma::gamma(&single_tensor).unwrap();
        gamma_result.data().unwrap()[0] as f64
    });
    let time2 = start2.elapsed();

    // Third computation with different function
    let start3 = Instant::now();
    let result3 = cached_compute(data[0] as f64, function_ids::BESSEL_J0, || {
        let single_tensor = Tensor::from_data(vec![data[0]], vec![1], tensor.device()).unwrap();
        let bessel_result = bessel::bessel_j0(&single_tensor).unwrap();
        bessel_result.data().unwrap()[0] as f64
    });
    let time3 = start3.elapsed();

    println!("  First γ(x) call (cache miss): {:.2?}", time1);
    println!("  Second γ(x) call (cache hit): {:.2?}", time2);
    println!("  J₀(x) call (new function): {:.2?}", time3);
    println!(
        "  Results: γ({:.1})={:.4}, γ({:.1})={:.4}, J₀({:.1})={:.4}",
        data[0], result1, data[0], result2, data[0], result3
    );
    println!(
        "  Speedup: {:.1}x faster on cache hit",
        time1.as_nanos() as f64 / time2.as_nanos() as f64
    );

    // Show cache statistics
    let stats = cache_stats();
    println!(
        "  Cache statistics: {} hits, {} misses",
        stats.hits, stats.misses
    );

    Ok(())
}

fn performance_comparison(tensor: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("  Comparing performance across optimization levels:");

    let iterations = 10;

    // Standard function timing
    let start_std = Instant::now();
    for _ in 0..iterations {
        let _ = gamma::gamma(tensor)?;
    }
    let time_std = start_std.elapsed() / iterations;

    // Fast approximation timing
    let start_fast = Instant::now();
    for _ in 0..iterations {
        let _ = fast_approximations::gamma_fast(tensor)?;
    }
    let time_fast = start_fast.elapsed() / iterations;

    // SIMD timing (if available)
    #[cfg(feature = "simd")]
    let time_simd = {
        let start_simd = Instant::now();
        for _ in 0..iterations {
            let _ = simd_optimizations::gamma_simd(tensor)?;
        }
        start_simd.elapsed() / iterations
    };

    println!("  Average time for γ(x) on 10,000 elements:");
    println!("    Standard:        {:.2?} (baseline)", time_std);
    println!(
        "    Fast approx:     {:.2?} ({:.1}x faster)",
        time_fast,
        time_std.as_nanos() as f64 / time_fast.as_nanos() as f64
    );

    #[cfg(feature = "simd")]
    println!(
        "    SIMD optimized:  {:.2?} ({:.1}x faster)",
        time_simd,
        time_std.as_nanos() as f64 / time_simd.as_nanos() as f64
    );

    #[cfg(not(feature = "simd"))]
    println!("    SIMD optimized:  (not available - build with --features simd)");

    Ok(())
}

fn optimization_guide() {
    println!("  Choose the right optimization level for your use case:\n");

    println!("  🔧 STANDARD FUNCTIONS - Use when:");
    println!("     • You need maximum accuracy");
    println!("     • Working with small to medium datasets");
    println!("     • Numerical precision is critical");
    println!("     • Example: Scientific computing, financial calculations\n");

    println!("  ⚡ SIMD OPTIMIZED - Use when:");
    println!("     • Processing large tensors (>1000 elements)");
    println!("     • Performance is critical");
    println!("     • Modern CPU with SIMD support available");
    println!("     • Example: Machine learning inference, signal processing\n");

    println!("  🏃 FAST APPROXIMATIONS - Use when:");
    println!("     • Maximum speed is required");
    println!("     • 0.01-0.1% accuracy loss is acceptable");
    println!("     • Real-time applications");
    println!("     • Example: Graphics rendering, game physics\n");

    println!("  🧠 SMART CACHING - Use when:");
    println!("     • Repeated computations with same inputs");
    println!("     • Expensive functions (gamma, bessel, etc.)");
    println!("     • Memory is available for cache");
    println!("     • Example: Monte Carlo simulations, iterative algorithms\n");

    println!("  📝 SELECTION MATRIX:");
    println!("     ┌─────────────────┬─────────┬─────────┬─────────┐");
    println!("     │ Use Case        │ Small   │ Large   │ Repeat  │");
    println!("     │                 │ Data    │ Data    │ Compute │");
    println!("     ├─────────────────┼─────────┼─────────┼─────────┤");
    println!("     │ High Accuracy   │ Std     │ SIMD    │ Cache   │");
    println!("     │ Fast Computing  │ Fast    │ SIMD    │ Cache   │");
    println!("     │ Real-time       │ Fast    │ Fast    │ Fast    │");
    println!("     └─────────────────┴─────────┴─────────┴─────────┘");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_examples_run() -> Result<(), Box<dyn std::error::Error>> {
        // Test that examples can run without errors
        let device = DeviceType::Cpu;
        let test_data = vec![0.5, 1.0, 2.0];
        let tensor = Tensor::from_data(test_data, vec![3], device)?;

        demo_standard_functions(&tensor)?;
        demo_fast_approximations(&tensor)?;
        demo_smart_caching(&tensor)?;

        Ok(())
    }
}
