//! Integrated Profiler Demo
//!
//! This example demonstrates the integrated profiler system that combines:
//! - Online learning and anomaly detection
//! - Cross-platform optimization
//! - Cloud provider integration
//! - Performance prediction
//! - Automatic optimization recommendations

use torsh_profiler::{integrated_profiler::*, ProfileEvent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Profiler: Integrated Profiler Demo ===\n");

    // ========================================
    // Part 1: Initialize Integrated Profiler
    // ========================================
    println!("1. Initializing Integrated Profiler");
    println!("   Setting up unified profiling system with all features\n");

    let mut profiler = IntegratedProfiler::new()?;
    println!("   ✓ Integrated profiler created successfully");
    println!("   Platform: {}", profiler.get_stats().platform_arch);
    if let Some(cloud) = &profiler.get_stats().cloud_provider {
        println!("   Cloud Provider: {}", cloud);
    }

    // Start profiling
    profiler.start()?;
    println!("   ✓ Profiler started\n");

    // ========================================
    // Part 2: Simulate Normal Operations
    // ========================================
    println!("2. Processing Normal Operations");
    println!("   Simulating typical training workload\n");

    let thread_id = format!("{:?}", std::thread::current().id())
        .chars()
        .filter(|c| c.is_numeric())
        .collect::<String>()
        .parse::<usize>()
        .unwrap_or(1);

    // Process 100 normal operations
    for i in 0..100 {
        let event = ProfileEvent {
            name: format!("forward_pass_{}", i),
            category: "training".to_string(),
            thread_id,
            start_us: i * 1000,
            duration_us: 100 + (i % 10) as u64, // Normal variance
            operation_count: Some(128),         // Batch size
            flops: Some(10_000_000),            // 10M FLOPs
            bytes_transferred: Some(512_000),   // 500 KB
            stack_trace: None,
        };

        let result = profiler.process_event(&event)?;

        if i == 50 {
            println!("   Processed 50 events:");
            println!("      Cluster assignment: {}", result.cluster);
            println!(
                "      Predicted duration: {:.2}μs",
                result.predicted_duration
            );
            println!("      Prediction error: {:.2}%", result.prediction_error);
        }
    }

    let stats = profiler.get_stats();
    println!("\n   Normal operation statistics:");
    println!("      Total events: {}", stats.total_events);
    println!("      Anomalies detected: {}", stats.total_anomalies);
    println!(
        "      Avg prediction error: {:.2}%",
        stats.avg_prediction_error_percent
    );

    // ========================================
    // Part 3: Inject Anomalies
    // ========================================
    println!("\n3. Detecting Anomalies");
    println!("   Injecting performance anomalies\n");

    // Memory spike anomaly
    let memory_spike = ProfileEvent {
        name: "memory_intensive_op".to_string(),
        category: "training".to_string(),
        thread_id,
        start_us: 100_000,
        duration_us: 150,
        operation_count: Some(128),
        flops: Some(10_000_000),
        bytes_transferred: Some(5_120_000), // 10x normal memory
        stack_trace: None,
    };

    let result = profiler.process_event(&memory_spike)?;
    println!("   Memory Spike Event:");
    println!("      Anomalies detected: {}", result.anomalies.len());
    for anomaly in &result.anomalies {
        println!("      • {}", anomaly.explanation);
        println!("        Severity: {:.2}", anomaly.severity);
    }

    // Duration spike anomaly
    let duration_spike = ProfileEvent {
        name: "slow_computation".to_string(),
        category: "training".to_string(),
        thread_id,
        start_us: 101_000,
        duration_us: 1000, // 10x slower
        operation_count: Some(128),
        flops: Some(10_000_000),
        bytes_transferred: Some(512_000),
        stack_trace: None,
    };

    let result = profiler.process_event(&duration_spike)?;
    println!("\n   Duration Spike Event:");
    println!("      Anomalies detected: {}", result.anomalies.len());
    for anomaly in &result.anomalies {
        println!("      • {}", anomaly.explanation);
        println!("        Severity: {:.2}", anomaly.severity);
    }

    // ========================================
    // Part 4: Optimization Recommendations
    // ========================================
    println!("\n4. Optimization Recommendations");
    println!("   Analyzing performance patterns and generating recommendations\n");

    let recommendations = profiler.get_top_recommendations(10);
    println!("   Top {} Recommendations:", recommendations.len());
    println!("   {}", "=".repeat(60));

    for (i, rec) in recommendations.iter().enumerate() {
        println!(
            "\n   {}. {} (Priority: {})",
            i + 1,
            rec.rec_type,
            rec.priority
        );
        println!("      Description: {}", rec.description);
        println!(
            "      Expected Improvement: {:.1}%",
            rec.expected_improvement_percent
        );
        println!("      Complexity: {}", rec.complexity);
        println!("      Actions:");
        for action in &rec.actions {
            println!("        • {}", action);
        }
    }

    println!("\n   {}", "=".repeat(60));

    // ========================================
    // Part 5: Performance Trends
    // ========================================
    println!("\n5. Performance Trends Analysis");
    println!("   Analyzing performance over time\n");

    // Process more events to build history
    for i in 100..200 {
        let event = ProfileEvent {
            name: format!("operation_{}", i),
            category: "training".to_string(),
            thread_id,
            start_us: i * 1000,
            duration_us: 100 + (i % 20) as u64,
            operation_count: Some(128),
            flops: Some(10_000_000),
            bytes_transferred: Some(512_000),
            stack_trace: None,
        };
        profiler.process_event(&event)?;
    }

    let history = profiler.get_performance_history();
    println!("   Performance History:");
    println!("      Snapshots recorded: {}", history.len());

    if let Some(latest) = history.back() {
        println!("      Latest snapshot:");
        println!("        Average duration: {:.2}μs", latest.avg_duration_us);
        println!(
            "        Average memory: {:.2} MB",
            latest.avg_memory_bytes / 1_048_576.0
        );
        println!("        Anomaly count: {}", latest.anomaly_count);
    }

    // ========================================
    // Part 6: Comprehensive Report
    // ========================================
    println!("\n6. Comprehensive Profiling Report");
    println!("   Generating integrated analysis report\n");

    let report = profiler.export_report()?;

    println!("   Overall Statistics:");
    println!(
        "      Total events processed: {}",
        report.stats.total_events
    );
    println!(
        "      Total anomalies detected: {}",
        report.stats.total_anomalies
    );
    println!(
        "      Recommendations generated: {}",
        report.stats.total_recommendations
    );
    println!(
        "      Average prediction error: {:.2}%",
        report.stats.avg_prediction_error_percent
    );
    println!("      Platform: {}", report.stats.platform_arch);
    if let Some(cloud) = &report.stats.cloud_provider {
        println!("      Cloud Provider: {}", cloud);
    }

    println!("\n   Anomaly Detection Summary:");
    println!(
        "      Anomaly rate: {:.2}%",
        report.anomaly_summary.anomaly_rate * 100.0
    );
    println!(
        "      Duration mean: {:.2}μs",
        report.anomaly_summary.duration_mean
    );
    println!(
        "      Duration std dev: {:.2}μs",
        report.anomaly_summary.duration_std
    );
    println!(
        "      Memory mean: {:.2} KB",
        report.anomaly_summary.memory_mean / 1024.0
    );

    println!("\n   Performance Predictor:");
    println!(
        "      Training samples: {}",
        report.predictor_stats.sample_count
    );
    println!(
        "      Average loss: {:.4}",
        report.predictor_stats.average_loss
    );
    println!("      Model weights: {:?}", report.predictor_stats.weights);

    println!("\n   Performance Trends:");
    println!(
        "      Average duration: {:.2}μs",
        report.performance_trends.avg_duration_us
    );
    println!(
        "      Duration trend: {:.2}%",
        report.performance_trends.duration_trend_percent
    );
    println!(
        "      Stability score: {:.2}",
        report.performance_trends.stability_score
    );

    println!("\n   Platform Information:");
    let platform_lines: Vec<_> = report.platform_info.lines().collect();
    for line in platform_lines.iter().take(8) {
        println!("      {}", line);
    }

    // ========================================
    // Part 7: Cloud Integration (if available)
    // ========================================
    if report.cloud_info.is_some() {
        println!("\n7. Cloud Integration");
        println!("   Cloud-specific profiling insights\n");

        if let Some(ref cloud) = report.cloud_info {
            println!("   Running on: {}", cloud);
            println!("   Cloud-specific recommendations available");
        }
    }

    // ========================================
    // Part 8: Export to JSON
    // ========================================
    println!("\n8. Exporting Report");

    let json_report = serde_json::to_string_pretty(&report)?;
    println!("   Report exported to JSON ({} bytes)", json_report.len());
    println!("   Sample output (first 500 characters):");
    println!("   {}", "-".repeat(60));
    for line in json_report.lines().take(20) {
        println!("   {}", line);
    }
    println!("   {} [truncated]", "-".repeat(60));

    // Stop profiler
    profiler.stop()?;
    println!("\n   ✓ Profiler stopped");

    // ========================================
    // Summary
    // ========================================
    println!("\n=== Demo Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("  ✓ Unified profiling system combining all advanced features");
    println!("  ✓ Real-time anomaly detection with online learning");
    println!("  ✓ Performance prediction using online gradient descent");
    println!("  ✓ Automatic clustering of operation patterns");
    println!("  ✓ Intelligent optimization recommendations");
    println!("  ✓ Performance trend analysis");
    println!("  ✓ Cross-platform awareness");
    println!("  ✓ Cloud provider integration");
    println!("  ✓ Comprehensive reporting and export");
    println!("\nThe integrated profiler provides a complete solution for production profiling!");

    Ok(())
}
