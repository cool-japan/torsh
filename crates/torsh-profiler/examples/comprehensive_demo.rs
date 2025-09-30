//! Comprehensive Demo - Full ToRSh Profiler Feature Showcase
//!
//! This example demonstrates the complete feature set of the ToRSh profiler,
//! including profiling, analytics, dashboard, alerts, regression detection,
//! and reporting capabilities.

use std::{thread, time::Duration};
use torsh_profiler::ci_cd::CiCdIntegration;
use torsh_profiler::reporting::{ReportFrequency, ReportGenerator};
use torsh_profiler::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Comprehensive ToRSh Profiler Demo");
    println!("===================================\n");

    // 1. Setup comprehensive profiling environment
    setup_profiling_environment()?;

    // 2. Demonstrate core profiling features
    demonstrate_core_profiling()?;

    // 3. Show memory profiling capabilities
    demonstrate_memory_profiling()?;

    // 4. Advanced analytics and ML analysis
    demonstrate_analytics()?;

    // 5. Dashboard and real-time monitoring
    demonstrate_dashboard()?;

    // 6. Alerts and automated monitoring
    demonstrate_alerts()?;

    // 7. Regression detection
    demonstrate_regression_detection()?;

    // 8. Reporting and visualization
    demonstrate_reporting()?;

    // 9. CI/CD integration example
    demonstrate_cicd_integration()?;

    // 10. Export data in multiple formats
    demonstrate_data_export()?;

    println!("✅ Comprehensive demo completed successfully!");
    println!("📁 Check /tmp/ directory for generated reports and data files");

    Ok(())
}

fn setup_profiling_environment() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Setting up profiling environment...");

    // Enable global profiling
    start_profiling();

    // Initialize optimized profiling for production-like performance
    init_optimized_profiling();

    println!("✅ Profiling environment ready\n");
    Ok(())
}

fn demonstrate_core_profiling() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demonstrating core profiling features...");

    // Basic function profiling
    {
        let _scope =
            ProfileScope::simple("data_preprocessing".to_string(), "ml_pipeline".to_string());
        simulate_data_preprocessing();
    }

    // Nested profiling with detailed operations
    {
        let _scope = ProfileScope::simple("model_training".to_string(), "ml_pipeline".to_string());

        for epoch in 0..5 {
            let _epoch_scope =
                ProfileScope::simple(format!("training_epoch_{epoch}"), "ml_pipeline".to_string());

            // Forward pass
            {
                let _scope =
                    ProfileScope::simple("forward_pass".to_string(), "neural_network".to_string());
                simulate_forward_pass();
            }

            // Backward pass
            {
                let _scope =
                    ProfileScope::simple("backward_pass".to_string(), "neural_network".to_string());
                simulate_backward_pass();
            }

            // Optimizer step
            {
                let _scope = ProfileScope::simple(
                    "optimizer_step".to_string(),
                    "neural_network".to_string(),
                );
                simulate_optimizer_step();
            }
        }
    }

    // Profile GPU operations (simulated)
    {
        let _scope =
            ProfileScope::simple("gpu_computation".to_string(), "gpu_operations".to_string());
        simulate_gpu_computation();
    }

    println!("✅ Core profiling demonstration complete\n");
    Ok(())
}

fn demonstrate_memory_profiling() -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Demonstrating memory profiling...");

    let mut memory_profiler = MemoryProfiler::new();
    memory_profiler.enable();
    memory_profiler.set_leak_detection_enabled(true);
    memory_profiler.set_timeline_enabled(true);

    // Simulate memory allocations
    for i in 0..10 {
        let size = (i + 1) * 1024 * 1024; // Increasing allocation sizes
        memory_profiler.record_allocation(i * 1000, size)?;

        thread::sleep(Duration::from_millis(50));

        // Simulate some deallocations
        if i % 3 == 0 {
            memory_profiler.record_deallocation(i * 1000)?;
        }
    }

    // Analyze memory usage
    let memory_stats = memory_profiler.get_stats()?;
    println!("   📈 Memory Statistics:");
    println!(
        "      Current usage: {:.2} MB",
        memory_stats.allocated as f64 / (1024.0 * 1024.0)
    );
    println!(
        "      Peak usage: {:.2} MB",
        memory_stats.peak as f64 / (1024.0 * 1024.0)
    );
    println!("      Total allocations: {}", memory_stats.allocations);
    println!("      Total deallocations: {}", memory_stats.deallocations);

    // Check for potential memory leaks
    let leak_results = memory_profiler.detect_leaks()?;
    if !leak_results.potential_leaks.is_empty() {
        println!(
            "   ⚠️ Found {} potential memory leaks",
            leak_results.leak_count
        );
        println!(
            "      Total leaked: {:.2} MB",
            leak_results.total_leaked_bytes as f64 / (1024.0 * 1024.0)
        );
    } else {
        println!("   ✅ No memory leaks detected");
    }

    println!("✅ Memory profiling demonstration complete\n");
    Ok(())
}

fn demonstrate_analytics() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Demonstrating analytics and ML features...");

    let profiler = global_profiler();
    let binding = profiler.lock();
    let events = binding.events();

    if !events.is_empty() {
        // Machine learning analysis
        let config = MLAnalysisConfig::default();
        let mut analyzer = MLAnalyzer::new(config);

        let features = analyzer.extract_features(events)?;
        analyzer.add_features(features.clone())?;

        // Perform clustering
        if analyzer.train_clustering().is_ok() {
            let clusters = analyzer.get_clusters();
            println!("   📊 Performance clustering results:");
            println!("      Found {} performance clusters", clusters.len());

            for (i, cluster) in clusters.iter().enumerate() {
                println!("      Cluster {}: {} samples", i, cluster.samples.len());
            }
        }

        // Anomaly detection
        let anomaly_result = analyzer.detect_anomaly(&features)?;
        println!("   🔍 Anomaly detection:");
        println!("      Anomaly score: {:.2}", anomaly_result.anomaly_score);
        println!("      Is anomaly: {}", anomaly_result.is_anomaly);

        // Get optimization suggestions
        let suggestions = analyzer.get_optimization_suggestions(&features);
        if !suggestions.is_empty() {
            println!("   💡 Optimization suggestions:");
            for suggestion in suggestions.iter().take(3) {
                println!("      • {suggestion}");
            }
        }
    }

    println!("✅ Analytics demonstration complete\n");
    Ok(())
}

fn demonstrate_dashboard() -> Result<(), Box<dyn std::error::Error>> {
    println!("🖥️ Demonstrating dashboard features...");

    let _profiler = global_profiler();
    let _memory_profiler = MemoryProfiler::new();

    // Create dashboard with custom configuration
    let dashboard_config = DashboardConfig {
        port: 8080,
        refresh_interval: 5,
        real_time_updates: true,
        max_data_points: 500,
        enable_stack_traces: true,
        custom_css: Some(
            "
            .dashboard { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .card { box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
        "
            .to_string(),
        ),
        websocket_config: WebSocketConfig::default(),
    };

    let dashboard = create_dashboard_with_config(dashboard_config);

    // Add sample alerts
    let alerts = vec![
        DashboardAlert {
            id: "high_latency".to_string(),
            severity: DashboardAlertSeverity::Warning,
            title: "High Latency Detected".to_string(),
            message: "Operation latency exceeded 100ms threshold".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            resolved: false,
        },
        DashboardAlert {
            id: "memory_usage".to_string(),
            severity: DashboardAlertSeverity::Info,
            title: "Memory Usage Notification".to_string(),
            message: "Memory usage is within normal parameters".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            resolved: false,
        },
    ];

    for alert in alerts {
        dashboard.add_alert(alert)?;
    }

    // Generate dashboard HTML
    let dashboard_html = dashboard.generate_dashboard_html()?;
    std::fs::write("/tmp/comprehensive_dashboard.html", dashboard_html)?;
    println!("   📄 Dashboard HTML generated: /tmp/comprehensive_dashboard.html");

    // Export dashboard data
    dashboard.export_data_json("/tmp/dashboard_data.json")?;
    println!("   📊 Dashboard data exported: /tmp/dashboard_data.json");

    // Show current metrics
    if let Some(current_data) = dashboard.get_current_data()? {
        println!("   📈 Current metrics:");
        println!(
            "      Operations: {}",
            current_data.performance_metrics.total_operations
        );
        println!(
            "      Avg duration: {:.2} ms",
            current_data.performance_metrics.average_duration_ms
        );
        println!(
            "      Memory usage: {:.1} MB",
            current_data.memory_metrics.current_usage_mb
        );
    }

    println!("✅ Dashboard demonstration complete\n");
    Ok(())
}

fn demonstrate_alerts() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚨 Demonstrating alerts system...");

    // Create alert configuration
    let alert_config = AlertConfig {
        duration_threshold: Duration::from_millis(50),
        memory_threshold: 100 * 1024 * 1024, // 100MB
        throughput_threshold: 100.0,
        enable_anomaly_detection: true,
        sigma_threshold: 2.0,
        notification_channels: vec![
            NotificationChannel::Console,
            NotificationChannel::Log {
                level: "warn".to_string(),
                format: "json".to_string(),
            },
        ],
        rate_limit_seconds: 60,
    };

    let mut alert_manager = create_alert_manager_with_config(alert_config);

    // Simulate alert conditions
    println!("   🔍 Simulating alert conditions...");

    // Create a slow operation event that should trigger duration threshold alert
    let slow_event = ProfileEvent {
        name: "slow_operation".to_string(),
        category: "test".to_string(),
        start_us: 0,
        duration_us: 75_000, // 75ms - should exceed 50ms threshold
        thread_id: 1,
        operation_count: Some(1),
        flops: None,
        bytes_transferred: None,
        stack_trace: None,
    };

    // Process the event for alert detection
    let alerts = alert_manager.process_event(&slow_event)?;
    if !alerts.is_empty() {
        println!(
            "      🚨 Triggered {} alert(s) for slow operation",
            alerts.len()
        );
    }

    // Get alert statistics
    let alert_stats = alert_manager.get_statistics();
    println!("   📊 Alert statistics:");
    println!("      Total alerts: {}", alert_stats.total_alerts);
    println!("      Active alerts: {}", alert_stats.active_alerts);
    println!("      Resolved alerts: {}", alert_stats.resolved_alerts);

    println!("✅ Alerts demonstration complete\n");
    Ok(())
}

fn demonstrate_regression_detection() -> Result<(), Box<dyn std::error::Error>> {
    println!("📉 Demonstrating regression detection...");

    let mut detector = RegressionDetector::with_defaults();

    // Establish baseline performance
    println!("   📊 Establishing performance baseline...");
    let mut baseline_samples = Vec::new();
    for i in 0..20 {
        let baseline_time = 100.0 + (i as f64 * 0.5); // Gradual improvement
        baseline_samples.push(baseline_time);
    }

    detector.update_baseline("matrix_multiply", "cpu", baseline_samples)?;

    // Test for normal performance (should not trigger regression)
    println!("   ✅ Testing normal performance...");
    // Create mock events for testing
    let normal_events = vec![ProfileEvent {
        name: "matrix_multiply".to_string(),
        category: "cpu".to_string(),
        start_us: 0,
        duration_us: 102,
        thread_id: 0,
        operation_count: None,
        flops: None,
        bytes_transferred: None,
        stack_trace: None,
    }];
    let normal_results = detector.detect_regressions(&normal_events)?;
    if normal_results.is_empty() {
        println!("      No regression detected for normal performance");
    }

    // Test for performance regression
    println!("   🚨 Testing performance regression...");
    let regression_events = vec![ProfileEvent {
        name: "matrix_multiply".to_string(),
        category: "cpu".to_string(),
        start_us: 0,
        duration_us: 150,
        thread_id: 0,
        operation_count: None,
        flops: None,
        bytes_transferred: None,
        stack_trace: None,
    }];
    let regression_results = detector.detect_regressions(&regression_events)?;
    if !regression_results.is_empty() {
        let regression = &regression_results[0];
        println!("      Regression detected!");
        println!("      Severity: {:?}", regression.severity);
        println!("      Change: {:.1}%", regression.change_percent);
        println!("      P-value: {:.4}", regression.p_value);

        if !regression.recommendation.is_empty() {
            println!("      Recommendation: {}", regression.recommendation);
        }
    }

    // Save regression baselines
    detector.save_baselines("/tmp/regression_baselines.json")?;
    println!("   📁 Regression baselines saved: /tmp/regression_baselines.json");

    println!("✅ Regression detection demonstration complete\n");
    Ok(())
}

fn demonstrate_reporting() -> Result<(), Box<dyn std::error::Error>> {
    println!("📄 Demonstrating reporting capabilities...");

    let profiler = global_profiler();
    let _memory_profiler = MemoryProfiler::new();

    // Create comprehensive report configuration
    let reporting_config = ReportConfig {
        name: "Comprehensive Report".to_string(),
        description: "Complete performance analysis report".to_string(),
        report_type: ReportType::Detailed,
        format: ReportFormat::Html,
        frequency: ReportFrequency::OnDemand,
        output_path: "/tmp/comprehensive_report.html".to_string(),
        template_path: None,
        include_charts: true,
        include_raw_data: true,
        time_range: None,
        filters: vec![],
        recipients: vec![],
        enabled: true,
    };

    let reporter = ReportGenerator::new(reporting_config);

    // Generate different types of reports
    println!("   📊 Generating performance report...");
    let events = profiler.lock().events().to_vec();
    let alerts = vec![]; // Empty alerts for now
    let report = reporter.generate_report(&events, &alerts)?;
    let exported_content = reporter.export_report(&report)?;
    std::fs::write("/tmp/performance_report.html", exported_content)?;

    println!("   📁 Report generated and exported to /tmp/performance_report.html");

    println!("✅ Reporting demonstration complete\n");
    Ok(())
}

fn demonstrate_cicd_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Demonstrating CI/CD integration...");

    let ci_config = CiCdConfig {
        platform: CiCdPlatform::GitHub,
        baseline_path: "performance_baseline.json".to_string(),
        report_path: "performance_report.json".to_string(),
        fail_on_regression: false,   // Don't fail for demo
        regression_threshold: 0.15,  // 15% regression threshold
        improvement_threshold: 0.05, // 5% improvement threshold
        enable_comments: true,
        comment_template: Some("Performance analysis: {}% change detected".to_string()),
        artifact_retention_days: 30,
    };

    let mut ci_integration = CiCdIntegration::new(ci_config);

    // Simulate CI/CD pipeline steps
    println!("   🏗️ Simulating CI/CD pipeline...");

    // Collect build environment info
    let build_info = ci_integration.detect_build_info()?;
    println!("      Build ID: {}", build_info.build_id);
    println!("      Branch: {}", build_info.branch);

    // Generate performance report
    println!("   ⚡ Generating performance report...");
    let events = torsh_profiler::global_profiler().lock().events().to_vec();
    let report = ci_integration.generate_report(&events)?;

    // Save CI report
    println!("   📄 Saving CI report...");
    ci_integration.save_report(&report)?;
    println!("   ✅ CI report saved successfully");

    println!("✅ CI/CD integration demonstration complete\n");
    Ok(())
}

fn demonstrate_data_export() -> Result<(), Box<dyn std::error::Error>> {
    println!("📤 Demonstrating data export capabilities...");

    let profiler = global_profiler();
    let memory_profiler = MemoryProfiler::new();

    // Export to Chrome Tracing format
    println!("   🌐 Exporting to Chrome Tracing format...");
    let binding = profiler.lock();
    let events = binding.events();
    export(events, "/tmp/chrome_trace.json")?;

    // Export to TensorBoard format
    println!("   📊 Exporting to TensorBoard format...");
    export_tensorboard_scalars(events, "/tmp/tensorboard_scalars")?;
    export_tensorboard_histograms(events, "/tmp/tensorboard_histograms")?;

    // Export custom formats
    println!("   🔧 Exporting custom formats...");

    let csv_format = CustomExportFormat {
        name: "csv".to_string(),
        description: "Custom CSV export format".to_string(),
        file_extension: "csv".to_string(),
        schema: ExportSchema::Csv {
            columns: vec![
                CsvColumn {
                    name: "Name".to_string(),
                    field: "name".to_string(),
                    formatter: None,
                },
                CsvColumn {
                    name: "Duration (ms)".to_string(),
                    field: "duration_us".to_string(),
                    formatter: Some(CsvFormatter::Duration(DurationFormat::Milliseconds)),
                },
                CsvColumn {
                    name: "Category".to_string(),
                    field: "category".to_string(),
                    formatter: None,
                },
            ],
            delimiter: ',',
            include_header: true,
        },
    };

    let mut exporter = CustomExporter::new();
    exporter.register_format(csv_format);
    let binding2 = profiler.lock();
    let events2 = binding2.events();
    exporter.export(events2, "csv", "/tmp/custom_export.csv")?;

    // Generate visualizations
    println!("   📈 Generating visualizations...");
    export_performance_trend_chart(&profiler.lock(), "/tmp/performance_trends.html")?;
    export_operation_frequency_chart(&profiler.lock(), "/tmp/operation_frequency.html")?;
    export_memory_scatter_plot(&memory_profiler, "/tmp/memory_scatter.html")?;
    export_duration_histogram(&profiler.lock(), "/tmp/duration_histogram.html")?;

    println!("✅ Data export demonstration complete\n");
    Ok(())
}

// Simulation functions for demonstration
fn simulate_data_preprocessing() {
    thread::sleep(Duration::from_millis(20));
}

fn simulate_forward_pass() {
    thread::sleep(Duration::from_millis(15));
}

fn simulate_backward_pass() {
    thread::sleep(Duration::from_millis(25));
}

fn simulate_optimizer_step() {
    thread::sleep(Duration::from_millis(10));
}

fn simulate_gpu_computation() {
    thread::sleep(Duration::from_millis(30));
}
