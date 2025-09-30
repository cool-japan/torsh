//! Advanced Analytics Demo - Regression Detection, ML Analysis, and Pattern Recognition
//!
//! This example demonstrates the advanced analytics capabilities of the ToRSh profiler,
//! including regression detection, machine learning-based performance analysis,
//! anomaly detection, and pattern recognition.

use std::{thread, time::Duration};
use torsh_profiler::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Advanced Analytics Demo - ToRSh Profiler");
    println!("==========================================\n");

    // 1. Regression Detection Demo
    demo_regression_detection()?;

    // 2. ML-based Performance Analysis Demo
    demo_ml_analysis()?;

    // 3. Anomaly Detection Demo
    demo_anomaly_detection()?;

    // 4. Pattern Recognition Demo
    demo_pattern_recognition()?;

    // 5. Correlation Analysis Demo
    demo_correlation_analysis()?;

    println!("✅ All analytics demos completed successfully!");
    Ok(())
}

fn demo_regression_detection() -> TorshResult<()> {
    println!("📈 1. Regression Detection Demo");
    println!("------------------------------");

    let mut detector = RegressionDetector::with_defaults();

    // Simulate baseline performance data (good performance)
    println!("   📊 Establishing performance baseline...");
    let matrix_samples: Vec<f64> = (0..10).map(|i| 100.0 + (i as f64 * 2.0)).collect();
    let convolution_samples: Vec<f64> = (0..10).map(|i| 50.0 + (i as f64 * 1.0)).collect();

    detector.update_baseline("matrix_multiply", "ml_operations", matrix_samples)?;
    detector.update_baseline("convolution", "ml_operations", convolution_samples)?;

    // Create test events for regression detection
    let mut test_events = Vec::new();

    // Simulate normal performance (should not trigger regression)
    println!("   ✅ Testing normal performance...");
    for i in 10..15 {
        let event = ProfileEvent {
            name: "matrix_multiply".to_string(),
            category: "ml_operations".to_string(),
            start_us: i * 1000,
            duration_us: (105.0 + (i as f64 * 2.0)) as u64,
            thread_id: 1,
            operation_count: Some(1),
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };
        test_events.push(event);
    }

    // Check for regressions with normal performance
    let normal_regressions = detector.detect_regressions(&test_events)?;
    if normal_regressions.is_empty() {
        println!("     ✅ No regressions detected with normal performance");
    } else {
        println!(
            "     ⚠️  Unexpected regressions: {}",
            normal_regressions.len()
        );
    }

    // Simulate performance regression
    println!("   🚨 Testing performance regression...");
    let mut regression_events = Vec::new();
    for i in 15..20 {
        let event = ProfileEvent {
            name: "matrix_multiply".to_string(),
            category: "ml_operations".to_string(),
            start_us: i * 1000,
            duration_us: (200.0 + (i as f64 * 5.0)) as u64,
            thread_id: 1,
            operation_count: Some(1),
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };
        regression_events.push(event);
    }

    let regressions = detector.detect_regressions(&regression_events)?;
    for regression in &regressions {
        println!(
            "     🚨 Regression detected: {:?} ({}% change)",
            regression.severity, regression.change_percent
        );
    }

    // Export regression baselines
    detector.save_baselines("/tmp/regression_baselines.json")?;
    println!("   📁 Exported regression baselines to /tmp/regression_baselines.json");

    println!();
    Ok(())
}

fn demo_ml_analysis() -> TorshResult<()> {
    println!("🤖 2. ML-based Performance Analysis Demo");
    println!("---------------------------------------");

    let config = MLAnalysisConfig::default();
    let mut analyzer = MLAnalyzer::new(config);

    // Generate diverse performance data for clustering
    println!("   📊 Generating performance data for ML analysis...");

    // Create ProfileEvent data for analysis
    let mut events = Vec::new();

    // Fast operations cluster
    for i in 0..15 {
        events.push(ProfileEvent {
            name: format!("fast_op_{i}"),
            category: "fast".to_string(),
            duration_us: (10000.0 + (i as f64 * 500.0)) as u64,
            start_us: (i as u64 * 1000),
            thread_id: 1,
            stack_trace: Some("main::fast_op".to_string()),
            operation_count: Some(1),
            flops: Some(1000),
            bytes_transferred: Some(50 * 1024 * 1024 + (i as u64 * 2 * 1024 * 1024)),
        });
    }

    // Slow operations cluster
    for i in 0..15 {
        events.push(ProfileEvent {
            name: format!("slow_op_{i}"),
            category: "slow".to_string(),
            duration_us: (100000.0 + (i as f64 * 5000.0)) as u64,
            start_us: (i as u64 * 10000),
            thread_id: 2,
            stack_trace: Some("main::slow_op".to_string()),
            operation_count: Some(1),
            flops: Some(500),
            bytes_transferred: Some(200 * 1024 * 1024 + (i as u64 * 10 * 1024 * 1024)),
        });
    }

    // Medium operations cluster
    for i in 0..10 {
        events.push(ProfileEvent {
            name: format!("medium_op_{i}"),
            category: "medium".to_string(),
            duration_us: (50000.0 + (i as f64 * 2000.0)) as u64,
            start_us: (i as u64 * 5000),
            thread_id: 3,
            stack_trace: Some("main::medium_op".to_string()),
            operation_count: Some(1),
            flops: Some(750),
            bytes_transferred: Some(100 * 1024 * 1024 + (i as u64 * 5 * 1024 * 1024)),
        });
    }

    // Extract features from events
    println!("   🔍 Extracting features for ML analysis...");
    let features = analyzer.extract_features(&events)?;
    analyzer.add_features(features)?;

    // Perform clustering analysis
    println!("   🔍 Performing K-means clustering analysis...");
    let _ = analyzer.train_clustering();
    let clusters = analyzer.get_clusters();
    println!("     📊 Found {} clusters", clusters.len());

    for (i, cluster) in clusters.iter().enumerate() {
        println!("     Cluster {}: {} samples", i, cluster.samples.len());
    }

    // Test anomaly detection
    println!("   🚨 Testing anomaly detection...");
    let anomaly_features = analyzer.extract_features(&events[..5])?;
    let anomaly_result = analyzer.detect_anomaly(&anomaly_features)?;
    println!(
        "     Anomaly score: {:.2} (is_anomaly: {})",
        anomaly_result.anomaly_score, anomaly_result.is_anomaly
    );

    // Get optimization suggestions
    let suggestions = analyzer.get_optimization_suggestions(&anomaly_features);
    println!("   💡 Optimization suggestions:");
    for suggestion in suggestions.iter().take(3) {
        println!("     • {suggestion}");
    }

    println!("   📁 ML analysis complete!");

    println!();
    Ok(())
}

fn demo_anomaly_detection() -> TorshResult<()> {
    println!("🚨 3. Anomaly Detection Demo");
    println!("---------------------------");

    start_profiling();

    // Generate normal performance data
    println!("   📊 Generating normal performance baseline...");
    for i in 0..20 {
        let _scope = ProfileScope::simple(format!("normal_op_{i}"), "baseline".to_string());
        thread::sleep(Duration::from_millis(10 + (i % 5))); // Normal variation
    }

    // Inject some anomalies
    println!("   🚨 Injecting performance anomalies...");

    // Memory anomaly (unusually high memory usage)
    {
        let _scope = ProfileScope::simple("memory_hog".to_string(), "anomaly".to_string());
        thread::sleep(Duration::from_millis(50)); // Much longer than normal
    }

    // CPU anomaly (very fast operation)
    {
        let _scope = ProfileScope::simple("cpu_burst".to_string(), "anomaly".to_string());
        thread::sleep(Duration::from_millis(1)); // Much faster than normal
    }

    // Duration anomaly (very slow operation)
    {
        let _scope = ProfileScope::simple("slow_anomaly".to_string(), "anomaly".to_string());
        thread::sleep(Duration::from_millis(100)); // Much slower than normal
    }

    stop_profiling();

    // Run anomaly detection analysis
    println!("   🔍 Running anomaly detection analysis...");

    let profiler = global_profiler();
    let prof = profiler.lock();
    let events = prof.events();

    let anomaly_analysis = detect_global_anomalies();

    println!("   📊 Anomaly Detection Results:");
    println!("     Total events analyzed: {}", events.len());
    println!(
        "     Performance anomalies: {}",
        anomaly_analysis.performance_anomalies.len()
    );
    println!(
        "     Memory anomalies: {}",
        anomaly_analysis.memory_anomalies.len()
    );
    println!(
        "     Throughput anomalies: {}",
        anomaly_analysis.throughput_anomalies.len()
    );
    println!(
        "     Temporal anomalies: {}",
        anomaly_analysis.temporal_anomalies.len()
    );

    // Show the most severe performance anomalies
    let mut performance_anomalies = anomaly_analysis.performance_anomalies.clone();
    performance_anomalies.sort_by(|a, b| b.severity.cmp(&a.severity));

    println!("   🚨 Most severe performance anomalies:");
    for anomaly in performance_anomalies.iter().take(3) {
        println!(
            "     • {} - {} (confidence: {:.2})",
            anomaly.event_name, anomaly.description, anomaly.confidence
        );
    }

    // Show memory anomalies separately
    if !anomaly_analysis.memory_anomalies.is_empty() {
        println!("   💾 Memory anomalies:");
        for anomaly in anomaly_analysis.memory_anomalies.iter().take(2) {
            println!(
                "     • Memory anomaly detected (confidence: {:.2})",
                anomaly.confidence
            );
        }
    }

    // Export anomaly analysis
    export_global_anomaly_analysis("/tmp/anomaly_analysis.json")?;
    println!("   📁 Exported anomaly analysis to /tmp/anomaly_analysis.json");

    println!();
    Ok(())
}

fn demo_pattern_recognition() -> TorshResult<()> {
    println!("🔍 4. Pattern Recognition Demo");
    println!("-----------------------------");

    start_profiling();

    // Generate patterns for detection
    println!("   📊 Generating recognizable patterns...");

    // Cyclic pattern
    for cycle in 0..3 {
        for i in 0..5 {
            let _scope = ProfileScope::simple(format!("cycle_{cycle}_{i}"), "cyclic".to_string());
            thread::sleep(Duration::from_millis(5 + (i * 2))); // Increasing pattern
        }
    }

    // Burst pattern
    for burst in 0..2 {
        // Quiet period
        thread::sleep(Duration::from_millis(10));

        // Burst activity
        for i in 0..10 {
            let _scope = ProfileScope::simple(format!("burst_{burst}_{i}"), "burst".to_string());
            thread::sleep(Duration::from_millis(1));
        }
    }

    // Gradual degradation pattern
    for i in 0..10 {
        let _scope = ProfileScope::simple(format!("degrading_{i}"), "degradation".to_string());
        thread::sleep(Duration::from_millis(5 + (i * 3))); // Getting slower
    }

    stop_profiling();

    // Run pattern detection
    println!("   🔍 Running pattern detection analysis...");

    let profiler = global_profiler();
    let prof = profiler.lock();
    let _events = prof.events();

    let pattern_analysis = detect_global_patterns();

    println!("   📊 Pattern Detection Results:");
    println!(
        "     Performance patterns: {}",
        pattern_analysis.performance_patterns.len()
    );
    println!(
        "     Bottleneck patterns: {}",
        pattern_analysis.bottleneck_patterns.len()
    );
    println!(
        "     Resource patterns: {}",
        pattern_analysis.resource_patterns.len()
    );
    println!(
        "     Temporal patterns: {}",
        pattern_analysis.temporal_patterns.len()
    );
    println!(
        "     Optimization patterns: {}",
        pattern_analysis.optimization_patterns.len()
    );

    // Show detected patterns
    println!("   🎯 Detected patterns:");
    for pattern in pattern_analysis.performance_patterns.iter().take(3) {
        println!(
            "     • {} - {} (confidence: {:.2})",
            pattern.pattern_type, pattern.description, pattern.confidence_score
        );
    }

    // Show optimization opportunities
    if !pattern_analysis.optimization_patterns.is_empty() {
        println!("   💡 Optimization opportunities:");
        for opt in pattern_analysis.optimization_patterns.iter().take(3) {
            println!(
                "     • {} - improvement: {:.2}% (complexity: {:?})",
                opt.optimization_type, opt.potential_improvement, opt.implementation_complexity
            );
        }
    }

    // Export pattern analysis
    export_global_pattern_analysis("/tmp/pattern_analysis.json")?;
    println!("   📁 Exported pattern analysis to /tmp/pattern_analysis.json");

    println!();
    Ok(())
}

fn demo_correlation_analysis() -> TorshResult<()> {
    println!("🔗 5. Correlation Analysis Demo");
    println!("------------------------------");

    start_profiling();

    // Generate correlated operations
    println!("   📊 Generating correlated operations...");

    for i in 0..10 {
        // These operations should be correlated (always happen together)
        {
            let _scope = ProfileScope::simple(format!("setup_{i}"), "correlated".to_string());
            thread::sleep(Duration::from_millis(5));
        }

        {
            let _scope = ProfileScope::simple(format!("process_{i}"), "correlated".to_string());
            thread::sleep(Duration::from_millis(10 + i));
        }

        {
            let _scope = ProfileScope::simple(format!("cleanup_{i}"), "correlated".to_string());
            thread::sleep(Duration::from_millis(3));
        }

        // Independent operation (should not be highly correlated)
        if i % 3 == 0 {
            let _scope =
                ProfileScope::simple(format!("independent_{i}"), "independent".to_string());
            thread::sleep(Duration::from_millis(8));
        }
    }

    stop_profiling();

    // Run correlation analysis
    println!("   🔍 Running correlation analysis...");

    let profiler = global_profiler();
    let prof = profiler.lock();
    let _events = prof.events();

    let correlation_analysis = analyze_global_correlations();

    println!("   📊 Correlation Analysis Results:");
    println!(
        "     Operation correlations: {}",
        correlation_analysis.operation_correlations.len()
    );
    println!(
        "     Performance correlations: {}",
        correlation_analysis.performance_correlations.len()
    );
    println!(
        "     Memory correlations: {}",
        correlation_analysis.memory_correlations.len()
    );
    println!(
        "     Temporal correlations: {}",
        correlation_analysis.temporal_correlations.len()
    );

    // Show strongest correlations
    let mut all_correlations = correlation_analysis.operation_correlations.clone();
    all_correlations.sort_by(|a, b| {
        b.correlation_strength
            .partial_cmp(&a.correlation_strength)
            .unwrap()
    });

    println!("   🔗 Strongest correlations:");
    for corr in all_correlations.iter().take(3) {
        println!(
            "     • {} ↔ {} (strength: {:.3})",
            corr.operation_a, corr.operation_b, corr.correlation_strength
        );
        if !corr.insights.is_empty() {
            println!("       💡 Insight: {}", corr.insights[0]);
        }
    }

    // Export correlation analysis
    export_global_correlation_analysis("/tmp/correlation_analysis.json")?;
    println!("   📁 Exported correlation analysis to /tmp/correlation_analysis.json");

    println!();
    Ok(())
}
