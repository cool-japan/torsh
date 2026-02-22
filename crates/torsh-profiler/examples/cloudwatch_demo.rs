//! AWS CloudWatch metrics integration demo
//!
//! This example demonstrates how to publish profiling metrics to AWS CloudWatch
//! for monitoring and alerting.

use torsh_profiler::{
    profile_scope, start_profiling, stop_profiling, CloudWatchPublisher,
    CloudWatchPublisherBuilder, CloudWatchUnit, Dimension,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== AWS CloudWatch Metrics Integration Demo ===\n");

    // Create CloudWatch publisher
    println!("1. Creating CloudWatch publisher...");
    let mut publisher = CloudWatchPublisherBuilder::new("ToRSh/Profiling")
        .region("us-west-2")
        .dimension("Environment", "development")
        .dimension("Application", "demo")
        .buffer_size(20)
        .build();

    println!("   ✓ Publisher created for namespace: ToRSh/Profiling");
    println!("   ✓ Region: us-west-2");
    println!("   ✓ Default dimensions: Environment=development, Application=demo\n");

    // Run profiled workload
    println!("2. Running profiled workload...");
    start_profiling();
    simulate_workload();
    stop_profiling();

    // Publish metrics from profiling events
    println!("\n3. Publishing metrics to CloudWatch...");
    let profiler = torsh_profiler::global_profiler();
    let profiler_guard = profiler.lock();
    let events = profiler_guard.events();

    publisher.publish_from_events(&events)?;
    println!("   ✓ Published metrics for {} operations", events.len());

    // Publish custom metrics
    println!("\n4. Publishing custom metrics...");
    publish_custom_metrics(&mut publisher)?;

    // Show buffered metrics
    println!("\n5. Buffered metrics (ready for CloudWatch):");
    display_buffered_metrics(&publisher);

    // Export as JSON for inspection
    println!("\n6. Exporting metrics as JSON...");
    let json = publisher.export_json()?;
    let preview_len = json.len().min(500);
    println!("   JSON preview (first {} chars):", preview_len);
    println!("   {}", &json[..preview_len]);

    // Flush to CloudWatch (simulated)
    println!("\n7. Flushing metrics to CloudWatch...");
    publisher.flush()?;
    println!("   ✓ Metrics flushed successfully");

    println!("\n✅ CloudWatch integration demo completed!");
    println!("\nIn production, metrics would be sent to AWS CloudWatch:");
    println!("  - Namespace: ToRSh/Profiling");
    println!("  - Region: us-west-2");
    println!("  - Metrics available in CloudWatch console and API");
    println!("  - Can create CloudWatch alarms based on thresholds");
    println!("  - Integrate with CloudWatch dashboards");

    Ok(())
}

fn simulate_workload() {
    println!("   Running matrix multiplication...");
    {
        profile_scope!("matrix_multiply");
        let size = 100;
        let _result: Vec<Vec<f64>> = (0..size)
            .map(|i| (0..size).map(|j| (i * j) as f64).collect())
            .collect();
    }

    println!("   Running data transfer simulation...");
    {
        profile_scope!("data_transfer");
        let data: Vec<u8> = vec![0; 1024 * 1024]; // 1MB
        let _checksum: u64 = data.iter().map(|&x| x as u64).sum();
    }

    println!("   Running GPU kernel simulation...");
    {
        profile_scope!("gpu_kernel");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    println!("   Workload completed");
}

fn publish_custom_metrics(
    publisher: &mut CloudWatchPublisher,
) -> Result<(), Box<dyn std::error::Error>> {
    // Model training metrics
    publisher.put_metric(
        "ModelAccuracy",
        0.9542,
        CloudWatchUnit::Percent,
        vec![
            Dimension {
                name: "Model".to_string(),
                value: "ResNet50".to_string(),
            },
            Dimension {
                name: "Dataset".to_string(),
                value: "ImageNet".to_string(),
            },
        ],
    )?;
    println!("   ✓ Published ModelAccuracy metric");

    // Training throughput
    publisher.put_metric(
        "TrainingThroughput",
        1542.0,
        CloudWatchUnit::CountPerSecond,
        vec![
            Dimension {
                name: "Model".to_string(),
                value: "ResNet50".to_string(),
            },
            Dimension {
                name: "BatchSize".to_string(),
                value: "64".to_string(),
            },
        ],
    )?;
    println!("   ✓ Published TrainingThroughput metric");

    // Memory utilization
    publisher.put_metric(
        "MemoryUtilization",
        85.5,
        CloudWatchUnit::Percent,
        vec![Dimension {
            name: "MemoryType".to_string(),
            value: "GPU".to_string(),
        }],
    )?;
    println!("   ✓ Published MemoryUtilization metric");

    // GPU utilization
    publisher.put_metric(
        "GPUUtilization",
        92.3,
        CloudWatchUnit::Percent,
        vec![
            Dimension {
                name: "Device".to_string(),
                value: "GPU:0".to_string(),
            },
            Dimension {
                name: "Operation".to_string(),
                value: "Training".to_string(),
            },
        ],
    )?;
    println!("   ✓ Published GPUUtilization metric");

    // Inference latency with statistics
    use torsh_profiler::StatisticSet;
    let latency_stats = StatisticSet {
        sample_count: 1000.0,
        sum: 15420.0,
        minimum: 10.5,
        maximum: 25.8,
    };

    publisher.put_metric_statistics(
        "InferenceLatency",
        latency_stats,
        CloudWatchUnit::Milliseconds,
        vec![
            Dimension {
                name: "Model".to_string(),
                value: "BERT".to_string(),
            },
            Dimension {
                name: "BatchSize".to_string(),
                value: "32".to_string(),
            },
        ],
    )?;
    println!("   ✓ Published InferenceLatency statistics");

    Ok(())
}

fn display_buffered_metrics(publisher: &CloudWatchPublisher) {
    let metrics = publisher.get_buffered_metrics();
    println!("   Total buffered: {} metrics\n", metrics.len());

    for (i, metric) in metrics.iter().take(10).enumerate() {
        println!("   Metric #{}: {}", i + 1, metric.metric_name);
        println!("     Value: {:.2}", metric.value);
        println!("     Unit: {:?}", metric.unit);
        println!("     Dimensions: {:?}", metric.dimensions);
        if let Some(ref stats) = metric.statistic_values {
            println!("     Statistics:");
            println!("       Sample Count: {}", stats.sample_count);
            println!("       Sum: {:.2}", stats.sum);
            println!("       Min: {:.2}", stats.minimum);
            println!("       Max: {:.2}", stats.maximum);
        }
        println!();
    }

    if metrics.len() > 10 {
        println!("   ... and {} more metrics", metrics.len() - 10);
    }
}
