//! Profiler demonstration example

use std::thread;
use std::time::Duration;
use torsh_profiler::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting profiler demo...");

    // Start global profiling
    start_profiling();

    // Simulate some work with scoped profiling
    {
        let _scope = ScopeGuard::new("main_computation");

        // Simulate matrix multiplication
        {
            let _scope = ScopeGuard::with_category("matrix_mul", "compute");
            thread::sleep(Duration::from_millis(50));
        }

        // Simulate convolution
        {
            let _scope = ScopeGuard::with_category("conv2d", "neural_net");
            thread::sleep(Duration::from_millis(30));
        }
    }

    // Manual event recording
    let profiler = global_profiler();
    let mut prof = profiler.lock();
    prof.add_event(ProfileEvent {
        name: "custom_operation".to_string(),
        category: "manual".to_string(),
        start_us: 0,
        duration_us: 25000, // 25ms
        thread_id: 1,
        operation_count: Some(1000000), // 1M operations
        flops: Some(2000000000),        // 2G FLOPS
        bytes_transferred: Some(4096),  // 4KB transferred
        stack_trace: None,
    });
    drop(prof);

    // Stop profiling
    stop_profiling();

    // Get statistics
    if let Ok((ops, flops, bytes, flops_per_sec, bandwidth)) = get_global_stats() {
        println!("Profile Statistics:");
        println!("  Total Operations: {ops}");
        println!("  Total FLOPS: {flops}");
        println!("  Total Bytes: {bytes}");
        println!("  FLOPS/sec: {flops_per_sec:.2}");
        println!("  Bandwidth (GB/s): {bandwidth:.2}");
    }

    // Export to different formats
    println!("\nExporting profile data...");

    // Chrome trace format
    if let Err(e) = export_global_trace("/tmp/profile_demo.json") {
        println!("Chrome trace export failed: {e}");
    } else {
        println!("Chrome trace exported to /tmp/profile_demo.json");
    }

    // JSON format
    if let Err(e) = export_global_json("/tmp/profile_demo_events.json") {
        println!("JSON export failed: {e}");
    } else {
        println!("JSON exported to /tmp/profile_demo_events.json");
    }

    // CSV format
    if let Err(e) = export_global_csv("/tmp/profile_demo.csv") {
        println!("CSV export failed: {e}");
    } else {
        println!("CSV exported to /tmp/profile_demo.csv");
    }

    // TensorBoard format
    if let Err(e) = export_global_tensorboard("/tmp/profile_demo_tensorboard") {
        println!("TensorBoard export failed: {e}");
    } else {
        println!("TensorBoard logs exported to /tmp/profile_demo_tensorboard_*.log");
    }

    println!("\nProfiler demo completed!");

    Ok(())
}
