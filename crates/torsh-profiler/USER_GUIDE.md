# ToRSh Profiler User Guide

Welcome to the ToRSh Profiler - a comprehensive performance profiling library for the ToRSh deep learning framework. This guide will help you get started with profiling your applications and optimizing performance.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Profiling](#basic-profiling)
3. [Advanced Features](#advanced-features)
4. [Dashboard and Visualization](#dashboard-and-visualization)
5. [Integration with CI/CD](#integration-with-cicd)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

Add the ToRSh profiler to your `Cargo.toml`:

```toml
[dependencies]
torsh-profiler = "0.1.0-alpha.1"
```

### Basic Setup

```rust
use torsh_profiler::*;

fn main() {
    // Start global profiling
    start_profiling();
    
    // Your application code here
    your_application_code();
    
    // Stop profiling and export results
    stop_profiling();
    
    // Export results
    let profiler = global_profiler();
    let memory_profiler = MemoryProfiler::new();
    
    // Export to various formats
    export_performance_dashboard(&profiler.lock().unwrap(), &memory_profiler, "dashboard.html").unwrap();
}
```

## Basic Profiling

### Function Profiling

Use the `ProfileScope` to profile functions automatically:

```rust
fn my_function() {
    let _scope = ProfileScope::simple("my_function".to_string(), "computation".to_string());
    
    // Your function code here
    expensive_computation();
}
```

### Using Profiling Macros

The profiler provides convenient macros for different profiling scenarios:

```rust
use torsh_profiler::*;

// Profile a block of code
profile_block!("matrix_multiply", {
    let result = matrix_a * matrix_b;
    result
});

// Profile a function call
profile_function!("data_loading", load_data_from_disk());

// Profile with custom metadata
profile_with_metadata!("gpu_operation", {
    operation_count: 1000,
    bytes_transferred: 1024 * 1024,
    flops: 2_000_000
}, {
    gpu_kernel_launch();
});

// Profile asynchronously
profile_async!("async_download", {
    async_download_data().await
});
```

### Memory Profiling

Track memory allocations and detect leaks:

```rust
let mut memory_profiler = MemoryProfiler::new();
memory_profiler.enable();

// Enable leak detection
memory_profiler.set_leak_detection_enabled(true);

// Your code that allocates memory
let data = vec![0u8; 1024 * 1024]; // 1MB allocation

// Check for leaks
let leak_results = memory_profiler.detect_leaks().unwrap();
if !leak_results.potential_leaks.is_empty() {
    println!("Found {} potential memory leaks", leak_results.leak_count);
}
```

### GPU Profiling

For CUDA operations:

```rust
let mut cuda_profiler = CudaProfiler::new();
cuda_profiler.enable();

// Profile GPU operations
{
    let _event = cuda_profiler.start_timing("kernel_launch").unwrap();
    // Launch CUDA kernel
    cuda_kernel_launch();
}

// Get GPU statistics
let stats = cuda_profiler.get_stats();
println!("Total kernel launches: {}", stats.kernel_launches);
```

## Advanced Features

### Machine Learning-based Analysis

Use ML algorithms to analyze performance patterns:

```rust
let config = MLAnalysisConfig::default();
let mut analyzer = MLAnalyzer::new(config);

// Create performance data from profiling events
let events = profiler.lock().unwrap().get_events();
let features = analyzer.extract_features(&events).unwrap();
analyzer.add_features(features).unwrap();

// Perform clustering to identify performance patterns
analyzer.train_clustering().unwrap();
let clusters = analyzer.get_clusters();

// Detect anomalies
let anomaly_result = analyzer.detect_anomaly(&features).unwrap();
if anomaly_result.is_anomaly {
    println!("Performance anomaly detected! Score: {}", anomaly_result.anomaly_score);
}

// Get optimization suggestions
let suggestions = analyzer.get_optimization_suggestions(&features);
for suggestion in suggestions {
    println!("💡 {}", suggestion);
}
```

### Regression Detection

Monitor performance over time and detect regressions:

```rust
let mut detector = RegressionDetector::new();

// Add baseline measurements
for i in 0..100 {
    detector.add_measurement("matrix_multiply", 100.0 + (i as f64)).unwrap();
}

// Update baselines
detector.update_baselines().unwrap();

// Check for regressions
let result = detector.check_regression("matrix_multiply", 150.0).unwrap();
if let Some(regression) = result {
    println!("Regression detected: {:?} ({}% slower)", 
             regression.severity, regression.percentage_change);
    
    for recommendation in regression.recommendations {
        println!("📋 {}", recommendation);
    }
}
```

### Distributed Profiling

For multi-node distributed applications:

```rust
// Initialize distributed profiling
init_distributed_profiling().unwrap();

// Add cluster nodes
add_cluster_node("worker1", "192.168.1.100:8080").unwrap();
add_cluster_node("worker2", "192.168.1.101:8080").unwrap();

// Record distributed events
record_distributed_event("data_transfer", "worker1", "worker2", 1024).unwrap();

// Analyze distributed performance
let analysis = analyze_distributed_performance().unwrap();
println!("Network efficiency: {:.2}%", analysis.network_analysis.efficiency);
```

### Custom Tool Integration

Integrate with external profiling tools:

```rust
// NVIDIA Nsight integration
let nsight_config = NsightConfig::default();
let mut nsight_profiler = create_nsight_profiler_with_config(nsight_config);

nsight_profiler.start_profiling().unwrap();
// Your GPU code here
nsight_profiler.stop_profiling().unwrap();

// Intel VTune integration
let vtune_config = VTuneConfig::default();
let mut vtune_profiler = create_vtune_profiler_with_config(vtune_config);

vtune_profiler.start_hotspot_analysis().unwrap();
// Your CPU-intensive code here
vtune_profiler.stop_analysis().unwrap();
```

## Dashboard and Visualization

### Real-time Dashboard

Create a web-based dashboard for real-time monitoring:

```rust
let dashboard_config = DashboardConfig {
    port: 8080,
    refresh_interval: 5,
    real_time_updates: true,
    max_data_points: 1000,
    enable_stack_traces: true,
    custom_css: Some("/* Your custom CSS */".to_string()),
};

let dashboard = create_dashboard_with_config(dashboard_config);

// Start the dashboard server
dashboard.start(profiler.clone(), memory_profiler.clone()).unwrap();

println!("Dashboard available at: http://localhost:8080");

// Add custom alerts
let alert = DashboardAlert {
    id: "high_memory".to_string(),
    severity: DashboardAlertSeverity::Warning,
    title: "High Memory Usage".to_string(),
    message: "Memory usage exceeded 80% threshold".to_string(),
    timestamp: std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs(),
    resolved: false,
};

dashboard.add_alert(alert).unwrap();
```

### Static Reports

Generate comprehensive reports in multiple formats:

```rust
let reporting_config = ReportingConfig {
    include_performance_analysis: true,
    include_memory_analysis: true,
    include_recommendations: true,
    chart_generation: true,
    export_raw_data: true,
};

let mut reporter = create_reporting_engine_with_config(reporting_config);

// Generate different types of reports
reporter.generate_performance_report(&profiler.lock().unwrap(), "performance_report.html").unwrap();
reporter.generate_memory_report(&memory_profiler, "memory_report.html").unwrap();

// Export to different formats
reporter.export_to_pdf("performance_report.html", "report.pdf").unwrap();
reporter.export_to_json(&profiler.lock().unwrap(), "data.json").unwrap();
```

### Visualization

Create interactive charts and visualizations:

```rust
// Generate performance trend charts
export_performance_trend_chart(&profiler.lock().unwrap(), "trend_chart.html").unwrap();

// Create memory usage scatter plots
export_memory_scatter_plot(&memory_profiler, "memory_plot.html").unwrap();

// Generate operation frequency charts
export_operation_frequency_chart(&profiler.lock().unwrap(), "frequency_chart.html").unwrap();

// Create duration histograms
export_duration_histogram(&profiler.lock().unwrap(), "duration_histogram.html").unwrap();
```

## Integration with CI/CD

### Automated Performance Testing

Integrate profiling into your CI/CD pipeline:

```rust
let ci_config = CICDConfig {
    platform: CICDPlatform::GitHubActions,
    regression_threshold: 10.0, // 10% regression threshold
    baseline_branch: "main".to_string(),
    report_format: ReportFormat::Html,
    fail_on_regression: true,
    generate_pr_comments: true,
};

let mut ci_integration = create_cicd_integration_with_config(ci_config);

// Run performance tests
ci_integration.run_performance_tests().unwrap();

// Check for regressions
let regression_results = ci_integration.check_regressions().unwrap();
if !regression_results.is_empty() {
    for regression in regression_results {
        println!("❌ Regression in {}: {}% slower", 
                 regression.operation, regression.percentage_change);
    }
    
    // Generate PR comment
    ci_integration.generate_pr_comment().unwrap();
}
```

### Automated Alerts

Set up automated alerts for performance issues:

```rust
let alert_config = AlertConfig {
    duration_threshold: Duration::from_millis(100),
    memory_threshold: 1024 * 1024 * 1024, // 1GB
    throughput_threshold: 1000.0, // operations per second
    enable_anomaly_detection: true,
    notification_channels: vec![
        NotificationChannel::Slack { webhook_url: "https://hooks.slack.com/...".to_string() },
        NotificationChannel::Email { recipients: vec!["team@company.com".to_string()] },
    ],
};

let mut alert_manager = create_alert_manager_with_config(alert_config);
alert_manager.start_monitoring(&profiler, &memory_profiler).unwrap();
```

## Performance Optimization

### Bottleneck Detection

Automatically identify performance bottlenecks:

```rust
let bottleneck_results = detect_bottlenecks(&profiler.lock().unwrap(), &memory_profiler).unwrap();

for bottleneck in bottleneck_results.bottlenecks {
    println!("🚨 Bottleneck detected in {}: {}", bottleneck.operation, bottleneck.description);
    println!("   Severity: {:?}", bottleneck.severity);
    println!("   Impact: {:.1}% of total time", bottleneck.impact_percentage);
    
    for recommendation in bottleneck.recommendations {
        println!("   💡 {}", recommendation);
    }
}
```

### Efficiency Analysis

Analyze overall system efficiency:

```rust
let efficiency_results = analyze_efficiency(&profiler.lock().unwrap(), &memory_profiler).unwrap();

println!("System Efficiency Report:");
println!("  CPU Efficiency: {:.1}%", efficiency_results.cpu_efficiency);
println!("  Memory Efficiency: {:.1}%", efficiency_results.memory_efficiency);
println!("  Cache Performance: {:.1}%", efficiency_results.cache_performance);
println!("  Overall Score: {:.1}/100", efficiency_results.overall_score);

for suggestion in efficiency_results.optimization_suggestions {
    println!("  📈 {}", suggestion);
}
```

### Workload Characterization

Understand your application's workload characteristics:

```rust
let workload_analysis = analyze_workload(&profiler.lock().unwrap()).unwrap();

println!("Workload Analysis:");
println!("  Type: {:?}", workload_analysis.workload_type);
println!("  Compute Intensity: {:.2}", workload_analysis.compute_characteristics.arithmetic_intensity);
println!("  Memory Bandwidth Utilization: {:.1}%", workload_analysis.memory_patterns.bandwidth_utilization);
println!("  Parallelization Efficiency: {:.1}%", workload_analysis.parallelism_analysis.efficiency);

for recommendation in workload_analysis.optimization_recommendations {
    println!("  🎯 {} (Priority: {:?})", recommendation.description, recommendation.priority);
}
```

## Troubleshooting

### Common Issues

#### High Profiling Overhead

```rust
// Use sampling to reduce overhead
let sampling_config = SamplingConfig {
    sampling_rate: 0.1, // Sample 10% of operations
    adaptive_sampling: true,
    min_duration_threshold: Duration::from_micros(100),
};

let optimized_profiler = create_optimized_profiler_with_config(sampling_config);
```

#### Memory Issues

```rust
// Enable memory pooling for reduced allocations
let mut memory_profiler = MemoryProfiler::new();
memory_profiler.set_memory_pool_size(1024 * 1024 * 1024); // 1GB pool

// Use compact events to reduce memory usage
init_optimized_profiling().unwrap();
```

#### Large Data Exports

```rust
// Use streaming export for large datasets
let export_config = CustomExportFormat {
    format: ExportFormat::Csv,
    compression: Some(CompressionType::Gzip),
    streaming: true,
    batch_size: 10000,
};

export_with_config(&profiler.lock().unwrap(), "large_data.csv.gz", export_config).unwrap();
```

### Performance Tips

1. **Use Scoped Profiling**: Always use RAII-based profiling (`ProfileScope`) to ensure proper cleanup.

2. **Enable Profiling Conditionally**: Use feature flags or environment variables to enable profiling only when needed.

3. **Monitor Overhead**: Use the built-in overhead tracking to ensure profiling doesn't impact performance significantly.

4. **Batch Operations**: When possible, batch multiple small operations to reduce profiling overhead.

5. **Use Appropriate Sampling**: For high-frequency operations, use sampling to reduce overhead.

### Debugging

Enable debug logging to troubleshoot issues:

```rust
// Set environment variable
std::env::set_var("TORSH_PROFILER_LOG", "debug");

// Or use tracing directly
use tracing::{info, debug, error};

debug!("Profiling operation: {}", operation_name);
```

## Best Practices

1. **Start Simple**: Begin with basic function profiling before moving to advanced features.

2. **Measure First**: Always establish baseline performance before optimization.

3. **Profile in Production**: Use low-overhead profiling in production environments.

4. **Automate Analysis**: Set up automated regression detection and alerts.

5. **Document Results**: Keep track of optimizations and their impact.

6. **Team Collaboration**: Share profiling results and dashboards with your team.

## Examples

See the `examples/` directory for complete examples:

- `profiler_demo.rs` - Basic profiling example
- `analytics_demo.rs` - Advanced analytics and ML analysis
- `dashboard_demo.rs` - Real-time dashboard example
- `advanced_profiling_demo.rs` - Comprehensive profiling features

## Support

For issues and questions:

- Check the [troubleshooting section](#troubleshooting)
- Review the examples in the repository
- File issues on the GitHub repository
- Refer to the API documentation

## Next Steps

1. Start with basic profiling in your application
2. Set up a dashboard for real-time monitoring
3. Integrate with your CI/CD pipeline
4. Use ML-based analysis for optimization insights
5. Share results with your team using reports and visualizations

Happy profiling! 🚀