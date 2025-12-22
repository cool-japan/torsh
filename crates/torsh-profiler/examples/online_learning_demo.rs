//! Online Learning Demo
//!
//! This example demonstrates the online learning capabilities of the torsh-profiler,
//! including real-time anomaly detection, streaming K-means clustering, and online prediction.

use torsh_profiler::{online_learning::*, start_profiling, stop_profiling, ProfileEvent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Profiler: Online Learning Demo ===\n");

    // Start profiling
    start_profiling();

    // ========================================
    // Part 1: Online Anomaly Detection
    // ========================================
    println!("1. Online Anomaly Detection");
    println!("   Detecting anomalies in real-time using incremental statistics\n");

    let config = OnlineLearningConfig {
        window_size: 100,
        learning_rate: 0.01,
        ewma_decay: 0.9,
        num_clusters: 5,
        anomaly_threshold: 3.0,
        min_samples: 30,
        drift_detection: true,
        drift_sensitivity: 0.05,
    };

    let mut anomaly_detector = OnlineAnomalyDetector::new(config.clone());

    // Get thread_id as usize
    let thread_id = format!("{:?}", std::thread::current().id())
        .chars()
        .filter(|c| c.is_numeric())
        .collect::<String>()
        .parse::<usize>()
        .unwrap_or(1);

    // Simulate normal operations
    for i in 0..50 {
        let event = ProfileEvent {
            name: format!("normal_op_{}", i),
            category: "training".to_string(),
            thread_id,
            start_us: i as u64 * 1000,
            duration_us: 100 + (i % 10) as u64, // Normal duration with small variance
            operation_count: Some(100),
            flops: Some(10000),
            bytes_transferred: Some(1024),
            stack_trace: None,
        };

        let anomalies = anomaly_detector.process_event(&event)?;
        if !anomalies.is_empty() {
            for anomaly in anomalies {
                println!(
                    "   Anomaly detected: {} (severity: {:.2})",
                    anomaly.anomaly_type, anomaly.severity
                );
            }
        }
    }

    // Inject anomalous events
    println!("   Injecting anomalous events...");
    for i in 50..53 {
        let anomalous_event = ProfileEvent {
            name: format!("anomalous_op_{}", i),
            category: "training".to_string(),
            thread_id,
            start_us: i as u64 * 1000,
            duration_us: 1000, // 10x longer than normal
            operation_count: Some(100),
            flops: Some(10000),
            bytes_transferred: Some(10240), // 10x more data
            stack_trace: None,
        };

        let anomalies = anomaly_detector.process_event(&anomalous_event)?;
        for anomaly in &anomalies {
            println!("   ⚠️  {} detected!", anomaly.anomaly_type);
            println!(
                "      Expected: {:.2}, Actual: {:.2}",
                anomaly.expected_value, anomaly.actual_value
            );
            println!(
                "      Z-score: {:.2}, Severity: {:.2}",
                anomaly.z_score, anomaly.severity
            );
            println!("      {}", anomaly.explanation);
        }
    }

    let stats = anomaly_detector.get_stats();
    println!("\n   Anomaly Detection Statistics:");
    println!("      Total samples: {}", stats.total_samples);
    println!("      Anomalies detected: {}", stats.anomaly_count);
    println!("      Anomaly rate: {:.2}%", stats.anomaly_rate * 100.0);
    println!(
        "      Duration mean: {:.2}μs (±{:.2})",
        stats.duration_mean, stats.duration_std
    );

    // ========================================
    // Part 2: Streaming K-means Clustering
    // ========================================
    println!("\n2. Streaming K-means Clustering");
    println!("   Clustering performance profiles in real-time\n");

    let mut kmeans = StreamingKMeans::new(3, 2); // 3 clusters, 2 dimensions

    // Simulate different operation types with different characteristics
    let operations = vec![
        // Fast, low memory operations (Cluster 1)
        vec![50.0, 512.0],
        vec![55.0, 600.0],
        vec![48.0, 500.0],
        vec![52.0, 550.0],
        // Medium speed, medium memory (Cluster 2)
        vec![200.0, 2048.0],
        vec![210.0, 2100.0],
        vec![195.0, 2000.0],
        vec![205.0, 2050.0],
        // Slow, high memory operations (Cluster 3)
        vec![800.0, 8192.0],
        vec![820.0, 8500.0],
        vec![790.0, 8000.0],
        vec![810.0, 8300.0],
    ];

    println!("   Training streaming K-means with operation profiles...");
    for (i, features) in operations.iter().enumerate() {
        let cluster = kmeans.update(features)?;
        println!(
            "      Op {}: [{:.0}μs, {:.0}B] → Cluster {}",
            i, features[0], features[1], cluster
        );
    }

    // Test prediction
    println!("\n   Testing cluster prediction:");
    let test_cases = vec![
        vec![52.0, 520.0],   // Should be Cluster 1 (fast)
        vec![205.0, 2060.0], // Should be Cluster 2 (medium)
        vec![815.0, 8400.0], // Should be Cluster 3 (slow)
    ];

    for (i, features) in test_cases.iter().enumerate() {
        let cluster = kmeans.predict(features)?;
        println!(
            "      Test {}: [{:.0}μs, {:.0}B] → Cluster {}",
            i, features[0], features[1], cluster
        );
    }

    // ========================================
    // Part 3: Online Prediction
    // ========================================
    println!("\n3. Online Performance Prediction");
    println!("   Predicting operation duration using online gradient descent\n");

    let mut predictor = OnlinePredictor::new(3, 0.01, 100);

    // Train predictor with simple relationship:
    // duration = 10 + 0.5 * data_size + 2.0 * complexity + 0.001 * iterations
    println!("   Training online predictor...");
    for i in 0..100 {
        let data_size = (i % 100) as f64;
        let complexity = ((i / 10) % 5) as f64;
        let iterations = (i * 10) as f64;

        let features = vec![data_size, complexity, iterations];
        let actual_duration = 10.0 + 0.5 * data_size + 2.0 * complexity + 0.001 * iterations;

        let loss = predictor.update(&features, actual_duration)?;
        if i % 20 == 0 {
            println!("      Iteration {}: Loss = {:.4}", i, loss);
        }
    }

    let predictor_stats = predictor.get_stats();
    println!("\n   Predictor Statistics:");
    println!("      Samples: {}", predictor_stats.sample_count);
    println!("      Average loss: {:.4}", predictor_stats.average_loss);
    println!("      Learned weights: {:?}", predictor_stats.weights);
    println!("      Bias: {:.4}", predictor_stats.bias);

    // Make predictions
    println!("\n   Making predictions:");
    let test_predictions = vec![
        (
            vec![50.0, 2.0, 500.0],
            10.0 + 0.5 * 50.0 + 2.0 * 2.0 + 0.001 * 500.0,
        ),
        (
            vec![80.0, 4.0, 800.0],
            10.0 + 0.5 * 80.0 + 2.0 * 4.0 + 0.001 * 800.0,
        ),
    ];

    for (i, (features, expected)) in test_predictions.iter().enumerate() {
        let predicted = predictor.predict(features)?;
        println!(
            "      Test {}: Expected = {:.2}μs, Predicted = {:.2}μs, Error = {:.2}%",
            i,
            expected,
            predicted,
            ((predicted - expected).abs() / expected * 100.0)
        );
    }

    // ========================================
    // Part 4: Concept Drift Detection
    // ========================================
    println!("\n4. Concept Drift Detection");
    println!("   Detecting changes in performance characteristics\n");

    let mut drift_detector = DriftDetector::new(0.05);

    // Simulate stable performance
    println!("   Simulating stable performance (duration ~100μs)...");
    for i in 0..30 {
        drift_detector.add_value(100.0 + (i % 5) as f64);
    }
    println!(
        "      Drift detected: {}",
        drift_detector.is_drift_detected()
    );

    // Simulate performance degradation
    println!("\n   Simulating performance degradation (duration →200μs)...");
    for i in 0..30 {
        let drift = drift_detector.add_value(200.0 + (i % 5) as f64);
        if drift {
            println!("      ⚠️  Concept drift detected at iteration {}!", i);
            if let Some(time) = drift_detector.last_drift_time() {
                println!("         Last drift: {}", time);
            }
            break;
        }
    }

    // ========================================
    // Part 5: EWMA Trend Analysis
    // ========================================
    println!("\n5. EWMA Trend Analysis");
    println!("   Tracking performance trends with exponentially weighted moving average\n");

    let mut ewma = EWMA::new(0.9);

    let durations = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, // Stable
        105.0, 110.0, 115.0, 120.0, 125.0, // Increasing trend
        120.0, 115.0, 110.0, 105.0, 100.0, // Decreasing trend
    ];

    println!("   Duration | EWMA Value | Trend");
    println!("   ---------|------------|-------");
    for (i, &duration) in durations.iter().enumerate() {
        let prev_ewma = ewma.value();
        ewma.update(duration);
        let trend = if i == 0 {
            "baseline".to_string()
        } else {
            let change = ewma.value() - prev_ewma;
            if change > 0.5 {
                "↑ increasing".to_string()
            } else if change < -0.5 {
                "↓ decreasing".to_string()
            } else {
                "→ stable".to_string()
            }
        };
        println!(
            "   {:.2}μs    | {:.2}       | {}",
            duration,
            ewma.value(),
            trend
        );
    }

    // ========================================
    // Part 6: Export Results
    // ========================================
    println!("\n6. Exporting Results");

    let anomaly_json = anomaly_detector.export_json()?;
    println!(
        "   Anomaly detection stats exported to JSON ({} bytes)",
        anomaly_json.len()
    );

    // Stop profiling
    stop_profiling();

    println!("\n=== Demo Complete ===");
    println!("\nKey Takeaways:");
    println!("  • Online anomaly detection enables real-time performance monitoring");
    println!("  • Streaming K-means clusters operations without storing full dataset");
    println!("  • Online predictors adapt to changing performance characteristics");
    println!("  • Drift detection identifies significant performance changes");
    println!("  • EWMA provides smooth trend analysis with low memory overhead");
    println!("\nThese techniques enable production profiling with minimal overhead!");

    Ok(())
}
