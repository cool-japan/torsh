//! Cross-Platform Profiling Demo
//!
//! This example demonstrates cross-platform profiling capabilities across x86_64, ARM64,
//! RISC-V, and WebAssembly, with platform-specific optimizations.

use torsh_profiler::cross_platform::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Profiler: Cross-Platform Profiling Demo ===\n");

    // ========================================
    // Part 1: Platform Detection
    // ========================================
    println!("1. Platform Detection");
    println!("   Detecting current architecture and capabilities\n");

    let arch = PlatformArch::detect();
    println!("   Current Architecture: {}", arch);
    println!("      Is ARM: {}", arch.is_arm());
    println!("      Is RISC-V: {}", arch.is_riscv());
    println!("      Is WebAssembly: {}", arch.is_wasm());
    println!(
        "      Supports HW Counters: {}",
        arch.supports_hardware_counters()
    );

    // ========================================
    // Part 2: Platform Capabilities
    // ========================================
    println!("\n2. Platform Capabilities Detection");
    println!("   Analyzing platform-specific features\n");

    let caps = PlatformCapabilities::detect();
    println!("   Platform Features:");
    println!("   {}", "-".repeat(60));
    println!("      Architecture: {}", caps.arch);
    println!(
        "      RDTSC Support: {}",
        if caps.has_rdtsc { "✓" } else { "✗" }
    );
    println!(
        "      PMU Support: {}",
        if caps.has_pmu { "✓" } else { "✗" }
    );
    println!(
        "      SIMD Support: {}",
        if caps.has_simd { "✓" } else { "✗" }
    );
    println!("      SIMD Width: {} bits", caps.simd_width);
    println!("      Cache Line Size: {} bytes", caps.cache_line_size);
    println!(
        "      Atomic Operations: {}",
        if caps.supports_atomics { "✓" } else { "✗" }
    );
    println!(
        "      Threading Support: {}",
        if caps.supports_threads { "✓" } else { "✗" }
    );
    println!(
        "      Timer Resolution: {} nanoseconds",
        caps.timer_resolution_ns
    );
    println!("   {}", "-".repeat(60));

    // ========================================
    // Part 3: Cross-Platform Timer
    // ========================================
    println!("\n3. Cross-Platform High-Resolution Timer");
    println!("   Testing timer accuracy across platforms\n");

    let mut timer = CrossPlatformTimer::new();

    // Test 1: Basic timing
    timer.start();
    simulate_work(100);
    let elapsed_us = timer.elapsed_us();
    let elapsed_ns = timer.elapsed_ns();

    println!("   Basic Timing Test:");
    println!("      Elapsed: {} μs ({} ns)", elapsed_us, elapsed_ns);
    println!(
        "      Accuracy: ~{} ns resolution",
        caps.timer_resolution_ns
    );

    // Test 2: Cycle counter (if available)
    if let Some(cycles) = timer.get_cycle_count() {
        println!("\n   Hardware Cycle Counter:");
        println!("      Cycles: {}", cycles);
        println!("      Note: Available on x86_64 with RDTSC");
    } else {
        println!("\n   Hardware Cycle Counter: Not available on this platform");
    }

    // ========================================
    // Part 4: Platform-Specific Profiling
    // ========================================
    println!("\n4. Platform-Specific Profiling Features");
    println!("   {}", "=".repeat(60));

    #[cfg(target_arch = "aarch64")]
    {
        println!("\n   ARM64 (AArch64) Features:");
        let neon = arm64::NeonInfo::detect();
        println!(
            "      NEON SIMD: {}",
            if neon.available { "✓" } else { "✗" }
        );
        println!("      Register Width: {} bits", neon.register_width);
        println!("      Number of Registers: {}", neon.num_registers);

        #[cfg(target_os = "macos")]
        {
            if arm64::apple_silicon::is_apple_silicon() {
                println!("\n   Apple Silicon Detected:");
                println!(
                    "      Performance Cores: {}",
                    arm64::apple_silicon::performance_core_count()
                );
                println!(
                    "      Efficiency Cores: {}",
                    arm64::apple_silicon::efficiency_core_count()
                );
                println!("      Total Cores: {}", num_cpus::get());
            }
        }

        println!("\n   ARM64 Performance Counters:");
        println!("      • Cycle Count");
        println!("      • Instruction Count");
        println!("      • L1/L2 Cache Misses");
        println!("      • Branch Misses");
    }

    #[cfg(target_arch = "x86_64")]
    {
        println!("\n   x86_64 Features:");
        println!("      RDTSC: ✓ (High-precision timing)");
        println!("      PMU: ✓ (Performance monitoring unit)");
        println!("      SIMD: AVX2 (256-bit vectors)");
        println!("\n   x86_64 Performance Counters:");
        println!("      • CPU Cycles (RDTSC)");
        println!("      • Instructions Retired");
        println!("      • Cache References/Misses");
        println!("      • Branch Predictions/Misses");
        println!("      • TLB Misses");
    }

    #[cfg(target_arch = "riscv64")]
    {
        println!("\n   RISC-V Features:");
        let rvv = riscv::RVVInfo::detect();
        println!(
            "      Vector Extension: {}",
            if rvv.available { "✓" } else { "✗" }
        );
        if rvv.available {
            println!("      VLEN: {} bits", rvv.vlen);
        }

        println!("\n   RISC-V Performance Counters:");
        println!("      • Cycle Count (RDCYCLE)");
        println!("      • Instruction Count (RDINSTRET)");
        println!("      • Timer (RDTIME)");
    }

    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    {
        println!("\n   WebAssembly Features:");
        let runtime = wasm::WasmRuntime::detect();
        println!("      Runtime: {:?}", runtime);

        let simd = wasm::WasmSimdInfo::detect();
        println!("      SIMD128: {}", if simd.simd128 { "✓" } else { "✗" });

        let mem_profiler = wasm::WasmMemoryProfiler::new();
        println!("\n   WASM Memory:");
        println!(
            "      Current Pages: {}",
            mem_profiler.current_memory_pages()
        );
        println!("      Memory Size: {} bytes", mem_profiler.memory_bytes());
        println!("      Max Pages: {:?}", mem_profiler.max_pages);
    }

    println!("\n   {}", "=".repeat(60));

    // ========================================
    // Part 5: Cross-Platform Profiler
    // ========================================
    println!("\n5. Cross-Platform Profiler");
    println!("   Creating unified profiler with platform optimizations\n");

    let mut profiler = CrossPlatformProfiler::new();
    println!("{}", profiler.platform_info());

    println!("\n   Recommended Profiling Strategy:");
    let strategy = profiler.recommended_strategy();
    println!("      Strategy: {}", strategy);
    match strategy {
        ProfilingStrategy::HardwareCounters => {
            println!("      Description: Use CPU performance counters for detailed metrics");
            println!("      Overhead: Very Low (<1%)");
            println!("      Accuracy: Cycle-accurate");
        }
        ProfilingStrategy::Hybrid => {
            println!("      Description: Combine software and hardware profiling");
            println!("      Overhead: Low (1-3%)");
            println!("      Accuracy: High");
        }
        ProfilingStrategy::Sampling => {
            println!("      Description: Statistical sampling profiling");
            println!("      Overhead: Very Low (<1%)");
            println!("      Accuracy: Statistical");
        }
        ProfilingStrategy::Lightweight => {
            println!("      Description: Minimal instrumentation profiling");
            println!("      Overhead: Ultra-low (<0.5%)");
            println!("      Accuracy: Moderate");
        }
        ProfilingStrategy::Basic => {
            println!("      Description: Basic timing-only profiling");
            println!("      Overhead: Minimal");
            println!("      Accuracy: Basic");
        }
    }

    // ========================================
    // Part 6: Benchmark Across Platforms
    // ========================================
    println!("\n6. Cross-Platform Performance Benchmark");
    println!("   Measuring profiling overhead\n");

    let iterations = 1000;

    // Benchmark without profiling
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        simulate_work(10);
    }
    let baseline_time = start.elapsed();

    // Benchmark with profiling
    let profiled_start = std::time::Instant::now();
    profiler.start();
    for i in 0..iterations {
        if i % 100 == 0 {
            let _elapsed = profiler.stop();
            profiler.start();
            if i > 0 {
                // Just measuring, not printing every iteration
            }
        }
        simulate_work(10);
    }
    let profiled_time = profiled_start.elapsed();

    let overhead = if baseline_time.as_micros() > 0 {
        (profiled_time.as_micros() as f64 - baseline_time.as_micros() as f64)
            / baseline_time.as_micros() as f64
            * 100.0
    } else {
        0.0
    };

    println!("   Benchmark Results ({} iterations):", iterations);
    println!("      Baseline (no profiling): {:?}", baseline_time);
    println!("      With profiling: {:?}", baseline_time + profiled_time);
    println!("      Overhead: {:.2}%", overhead);

    // ========================================
    // Part 7: Platform Recommendations
    // ========================================
    println!("\n7. Platform-Specific Recommendations");
    println!("   {}", "=".repeat(60));

    match arch {
        PlatformArch::X86_64 => {
            println!("\n   x86_64 Optimization Tips:");
            println!("      • Use RDTSC for high-precision timing");
            println!("      • Enable PMU counters for detailed metrics");
            println!("      • Use AVX2/AVX-512 for vectorized operations");
            println!("      • Consider Intel VTune for advanced profiling");
            println!("      • Use perf for system-wide profiling");
        }
        PlatformArch::ARM64 => {
            println!("\n   ARM64 Optimization Tips:");
            println!("      • Use NEON for SIMD operations");
            println!("      • Leverage Apple Instruments on macOS");
            println!("      • Consider P-core vs E-core placement");
            println!("      • Use ARM Performance Libraries (APL)");
            println!("      • Profile cache behavior carefully");
        }
        PlatformArch::RISCV64 => {
            println!("\n   RISC-V Optimization Tips:");
            println!("      • Use vector extension (RVV) when available");
            println!("      • Profile memory access patterns");
            println!("      • Consider RISC-V specific performance counters");
            println!("      • Use compressed instructions for code density");
        }
        PlatformArch::WASM32 | PlatformArch::WASM64 => {
            println!("\n   WebAssembly Optimization Tips:");
            println!("      • Use SIMD128 for performance");
            println!("      • Minimize memory allocations");
            println!("      • Use SharedArrayBuffer for threading");
            println!("      • Profile with browser DevTools");
            println!("      • Consider WASM profiling tools");
        }
        _ => {
            println!("\n   General Optimization Tips:");
            println!("      • Use standard timing mechanisms");
            println!("      • Minimize profiling overhead");
            println!("      • Use sampling when possible");
        }
    }

    println!("\n   {}", "=".repeat(60));

    println!("\n=== Demo Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("  ✓ Automatic platform detection (x86_64, ARM64, RISC-V, WASM)");
    println!("  ✓ Platform capability discovery");
    println!("  ✓ Cross-platform high-resolution timing");
    println!("  ✓ Architecture-specific optimizations");
    println!("  ✓ Recommended profiling strategies");
    println!("  ✓ Overhead measurement");
    println!("  ✓ Platform-specific recommendations");
    println!("\nThe cross-platform profiler adapts automatically to your architecture!");

    Ok(())
}

// Helper function to simulate work
fn simulate_work(duration_us: u64) {
    let start = std::time::Instant::now();
    while start.elapsed().as_micros() < duration_us as u128 {
        // Busy wait
        std::hint::spin_loop();
    }
}
