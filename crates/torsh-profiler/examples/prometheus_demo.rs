//! Prometheus metrics integration demo
//!
//! This example demonstrates how to export profiling metrics to Prometheus format
//! for monitoring and alerting.

use torsh_profiler::{profile_scope, start_profiling, stop_profiling, PrometheusExporter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Prometheus Metrics Integration Demo ===\n");

    // Start profiling
    start_profiling();

    // Simulate some operations
    simulate_workload();

    // Stop profiling
    stop_profiling();

    // Create Prometheus exporter
    println!("Creating Prometheus exporter...");
    let exporter = PrometheusExporter::new()?;

    // Update metrics from global profiler
    println!("Updating metrics from profiler events...");
    let profiler = torsh_profiler::global_profiler();
    let profiler_guard = profiler.lock();
    exporter.update_from_profiler(&profiler_guard)?;
    drop(profiler_guard);

    // Export metrics in Prometheus text format
    println!("\n=== Prometheus Metrics Export ===\n");
    let metrics_text = exporter.export_text()?;
    println!("{}", metrics_text);

    // Demonstrate custom metrics
    println!("\n=== Custom Metrics Demo ===\n");
    demonstrate_custom_metrics()?;

    // Demonstrate metric updates
    println!("\n=== Real-time Metric Updates Demo ===\n");
    demonstrate_realtime_updates()?;

    println!("\n✅ Prometheus integration demo completed!");
    println!("\nMetrics can be scraped by Prometheus at /metrics endpoint");
    println!("Example Prometheus configuration:");
    println!("  scrape_configs:");
    println!("    - job_name: 'torsh-profiler'");
    println!("      static_configs:");
    println!("        - targets: ['localhost:9090']");

    Ok(())
}

fn simulate_workload() {
    println!("Running simulated workload...\n");

    // Simulate matrix multiplication
    {
        profile_scope!("matrix_multiply");
        let size = 100;
        let _result: Vec<Vec<f64>> = (0..size)
            .map(|i| (0..size).map(|j| (i * j) as f64).collect())
            .collect();
    }

    // Simulate data transfer
    {
        profile_scope!("data_transfer");
        let data: Vec<u8> = vec![0; 1024 * 1024]; // 1MB
        let _checksum: u64 = data.iter().map(|&x| x as u64).sum();
    }

    // Simulate GPU kernel
    {
        profile_scope!("gpu_kernel");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // Simulate memory allocation
    {
        profile_scope!("memory_allocation");
        let _large_vec: Vec<f64> = vec![0.0; 1000000];
    }

    println!("Workload completed!\n");
}

fn demonstrate_custom_metrics() -> Result<(), Box<dyn std::error::Error>> {
    let exporter = PrometheusExporter::new()?;

    // Create custom counter
    println!("Creating custom counter metric...");
    let training_steps = exporter.create_counter(
        "torsh_training_steps_total",
        "Total number of training steps",
        &["model", "dataset"],
    )?;

    training_steps
        .with_label_values(&["resnet50", "imagenet"])
        .inc_by(100.0);
    training_steps
        .with_label_values(&["bert", "wikipedia"])
        .inc_by(50.0);

    // Create custom gauge
    println!("Creating custom gauge metric...");
    let model_accuracy = exporter.create_gauge(
        "torsh_model_accuracy",
        "Current model accuracy",
        &["model", "metric"],
    )?;

    model_accuracy
        .with_label_values(&["resnet50", "top1"])
        .set(0.756);
    model_accuracy
        .with_label_values(&["resnet50", "top5"])
        .set(0.928);

    // Create custom histogram
    println!("Creating custom histogram metric...");
    let batch_size_dist = exporter.create_histogram(
        "torsh_batch_size_distribution",
        "Distribution of batch sizes",
        &["model"],
        vec![16.0, 32.0, 64.0, 128.0, 256.0],
    )?;

    batch_size_dist
        .with_label_values(&["resnet50"])
        .observe(64.0);
    batch_size_dist
        .with_label_values(&["resnet50"])
        .observe(128.0);

    // Export and display
    let metrics_text = exporter.export_text()?;
    println!("\nCustom Metrics:\n{}", metrics_text);

    Ok(())
}

fn demonstrate_realtime_updates() -> Result<(), Box<dyn std::error::Error>> {
    let exporter = PrometheusExporter::new()?;

    // Simulate real-time metric updates
    println!("Simulating real-time metric updates...\n");

    for i in 1..=5 {
        // Record memory allocation
        exporter.set_memory_allocated("training_loop", (i * 1024 * 1024) as f64);

        // Record active operations
        exporter.set_active_operations("forward_pass", i as f64);

        // Record profiling overhead
        exporter.record_overhead(i as f64 * 10.0);

        println!(
            "Update {}: memory={} MB, active_ops={}, overhead={} µs",
            i,
            i,
            i,
            i * 10
        );

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Export final state
    let metrics_text = exporter.export_text()?;
    println!("\nFinal Metrics State:\n{}", metrics_text);

    Ok(())
}
