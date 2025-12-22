//! Advanced Model Profiling Example
//!
//! This example demonstrates how to use the comprehensive profiling capabilities
//! of torsh-hub to analyze model performance, memory usage, and identify bottlenecks.
//!
//! Run with: cargo run --example advanced_profiling

use std::time::Duration;
use torsh_hub::profiling::{ModelProfiler, ProfilerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Model Profiling Example ===\n");

    // Step 1: Configure the profiler
    println!("Step 1: Configuring profiler...");
    let config = ProfilerConfig {
        enable_memory_profiling: true,
        enable_layer_timing: true,
        enable_shape_tracking: true,
        enable_gradient_tracking: true,
        memory_sample_interval: Duration::from_millis(50),
        max_profile_history: 100,
        profile_dir: std::env::temp_dir().join("torsh_profiles"),
        enable_call_stack: true,
        enable_op_profiling: true,
    };

    let _profiler = ModelProfiler::new(config)?;
    println!("✓ Profiler configured\n");

    println!("=== Profiling Features ===");
    println!("The ModelProfiler provides:");
    println!("  ✓ Layer-wise timing analysis");
    println!("  ✓ Memory profiling and tracking");
    println!("  ✓ Tensor shape tracking");
    println!("  ✓ Gradient computation profiling");
    println!("  ✓ Operation-level profiling");
    println!("  ✓ Call stack tracking");
    println!("  ✓ Bottleneck identification");
    println!("  ✓ Optimization recommendations");

    println!("\n=== Example Profiling Workflow ===");
    println!("1. Create profiler with custom configuration");
    println!("2. Start a profiling session");
    println!("3. Run your model forward/backward passes");
    println!("4. Record layer executions and operations");
    println!("5. Take memory snapshots at key points");
    println!("6. End the session and analyze results");
    println!("7. Get optimization recommendations");
    println!("8. Export results for further analysis");

    println!("\n=== Performance Metrics ===");
    println!("  - Total execution time");
    println!("  - Per-layer execution time");
    println!("  - Memory usage (peak, average, timeline)");
    println!("  - CPU utilization");
    println!("  - GPU utilization (if available)");
    println!("  - I/O statistics");
    println!("  - Context switches");

    println!("\n=== Memory Analysis ===");
    println!("  - Memory allocation tracking");
    println!("  - Peak memory detection");
    println!("  - Memory leak identification");
    println!("  - Garbage collection monitoring");
    println!("  - Per-layer memory usage");

    println!("\n=== Bottleneck Detection ===");
    println!("  - Automatic identification of slow layers");
    println!("  - Severity classification");
    println!("  - Percentage of total time analysis");
    println!("  - Actionable optimization suggestions");

    println!("\n=== Resource Monitoring ===");
    println!("  - CPU: Usage, context switches, frequency");
    println!("  - Memory: Timeline, allocations, GC stats");
    println!("  - GPU: Utilization, memory, temperature, power");
    println!("  - I/O: Disk stats, network stats");

    println!("\n=== Export Formats ===");
    println!("  - JSON: For programmatic analysis");
    println!("  - CSV: For spreadsheet analysis");
    println!("  - Binary: For fast loading/saving");
    println!("  - HTML: For visual reports");

    println!("\n=== Optimization Recommendations ===");
    println!("The profiler provides automatic suggestions:");
    println!("  - Layer fusion opportunities");
    println!("  - Memory optimization strategies");
    println!("  - Batch size recommendations");
    println!("  - Parallelization opportunities");
    println!("  - Hardware utilization improvements");

    println!("\n=== Comparative Analysis ===");
    println!("  - Compare multiple profiling sessions");
    println!("  - Track performance improvements");
    println!("  - Regression detection");
    println!("  - A/B testing different configurations");

    println!("\n=== Integration ===");
    println!("  - Seamless integration with training loops");
    println!("  - Minimal overhead");
    println!("  - Configurable sampling rates");
    println!("  - Context-aware profiling");

    println!("\n=== Profiling Complete ===");
    println!("For detailed API usage, see the ModelProfiler documentation.");
    println!(
        "Profile data stored in: {:?}",
        std::env::temp_dir().join("torsh_profiles")
    );

    Ok(())
}
