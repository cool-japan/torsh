//! Enhanced Distributed Training Features Demonstration
//!
//! This example demonstrates the advanced features implemented in torsh-distributed:
//! - Enhanced gradient compression with performance optimizations
//! - Network-aware adaptive compression
//! - Comprehensive benchmarking suite
//! - Distributed monitoring and analytics
//! - Enhanced fault tolerance mechanisms
//! - Distributed memory optimization
//! - Training analytics dashboard with real-time insights
//!
//! Run with: cargo run --package torsh-distributed --example enhanced_features_demo

use std::sync::Arc;
use std::time::Instant;
use torsh_distributed::{
    distributed_memory_optimization::{DistributedMemoryOptimizer, MemoryOptimizationConfig},
    distributed_monitoring::{DistributedMonitor, MonitoringConfig},
    enhanced_benchmarks::{BenchmarkConfig, EnhancedBenchmarkSuite},
    enhanced_fault_tolerance::{EnhancedFaultTolerance, FaultToleranceConfig},
    gradient_compression::{CompressionConfig, CompressionMethod},
    gradient_compression_enhanced::EnhancedGradientCompressor,
    network_aware_compression::{
        AdaptiveCompressionConfig, NetworkAwareCompressor, TrainingMetrics,
    },
    training_analytics_dashboard::{DashboardConfig, TrainingAnalyticsDashboard},
    TorshResult,
};
use torsh_tensor::creation::randn;
// use tracing::{info, Level};
// Note: tracing_subscriber would need to be added as dependency for full logging

#[tokio::main]
async fn main() -> TorshResult<()> {
    // Initialize logging (simplified for demo)
    // tracing_subscriber::fmt()
    //     .with_max_level(Level::INFO)
    //     .init();

    println!("🚀 Enhanced Distributed Training Features Demonstration");
    println!("======================================================\n");

    // Demo 1: Enhanced Gradient Compression
    demo_enhanced_compression().await?;

    // Demo 2: Network-Aware Adaptive Compression
    demo_network_aware_compression().await?;

    // Demo 3: Comprehensive Benchmarking Suite
    demo_benchmarking_suite().await?;

    // Demo 4: Performance Comparison
    demo_performance_comparison().await?;

    // Demo 5: Distributed Monitoring System
    demo_distributed_monitoring().await?;

    // Demo 6: Enhanced Fault Tolerance
    demo_enhanced_fault_tolerance().await?;

    // Demo 7: Distributed Memory Optimization
    demo_distributed_memory_optimization().await?;

    // Demo 8: Training Analytics Dashboard
    demo_training_analytics_dashboard().await?;

    println!("\n🎉 All demonstrations completed successfully!");
    println!("======================================================");

    Ok(())
}

/// Demonstrate enhanced gradient compression capabilities
async fn demo_enhanced_compression() -> TorshResult<()> {
    println!("📊 Demo 1: Enhanced Gradient Compression");
    println!("----------------------------------------");

    // Create a configuration for enhanced compression
    let config = CompressionConfig {
        method: CompressionMethod::TopK { k: 0.1 },
        compression_ratio: 0.1,
        error_feedback: true,
        memory_efficient: true,
        ..CompressionConfig::default()
    };

    // Initialize enhanced compressor
    let mut compressor = EnhancedGradientCompressor::new(config)?;

    // Create test gradients of different sizes
    let test_gradients = vec![
        ("Small Model (1M params)", randn::<f32>(&[1000, 1000])?),
        ("Medium Model (10M params)", randn::<f32>(&[3162, 3162])?),
        ("Large Model (100M params)", randn::<f32>(&[10000, 10000])?),
    ];

    println!("Testing enhanced compression on different model sizes:");

    for (model_name, gradient) in test_gradients {
        let start_time = Instant::now();
        let (compressed, metrics) = compressor.compress_gradient_enhanced(&gradient, model_name)?;
        let total_time = start_time.elapsed();

        println!("\n  {} Results:", model_name);
        println!("    Compression Ratio: {:.3}", metrics.compression_ratio);
        println!(
            "    Compression Time: {:.2}ms",
            metrics.compression_time_us as f64 / 1000.0
        );
        println!(
            "    Memory Saved: {:.2}MB",
            metrics.memory_saved as f64 / 1_048_576.0
        );
        println!("    Throughput: {:.2}MB/s", metrics.throughput_mbps);
        println!("    Compression Error: {:.6}", metrics.compression_error);
        println!("    Optimized Ops: {}", metrics.optimized_ops_count);
        println!("    Total Time: {:.2}ms", total_time.as_millis());

        // Verify decompression works
        let decompressed = compressor.decompress_gradient_enhanced(&compressed)?;
        println!(
            "    ✓ Decompression successful (shape: {:?})",
            decompressed.shape().dims()
        );
    }

    println!("\n✅ Enhanced compression demo completed");
    Ok(())
}

/// Demonstrate network-aware adaptive compression
async fn demo_network_aware_compression() -> TorshResult<()> {
    println!("\n🌐 Demo 2: Network-Aware Adaptive Compression");
    println!("-----------------------------------------------");

    // Create base compression configuration
    let base_config = CompressionConfig {
        method: CompressionMethod::TopK { k: 0.2 },
        compression_ratio: 0.2,
        error_feedback: true,
        ..CompressionConfig::default()
    };

    // Create adaptive configuration
    let adaptive_config = AdaptiveCompressionConfig {
        target_bandwidth_utilization: 0.8,
        min_compression_ratio: 0.01,
        max_compression_ratio: 0.9,
        adaptation_sensitivity: 0.4,
        convergence_quality_weight: 0.7,
        communication_efficiency_weight: 0.3,
        ..AdaptiveCompressionConfig::default()
    };

    // Initialize network-aware compressor
    let mut network_compressor = NetworkAwareCompressor::new(base_config, adaptive_config)?;

    // Simulate different training scenarios
    let scenarios = vec![
        (
            "Early Training (High Loss)",
            TrainingMetrics {
                loss: 2.5,
                gradient_norm: 10.0,
                learning_rate: 0.01,
            },
        ),
        (
            "Mid Training (Moderate Loss)",
            TrainingMetrics {
                loss: 1.2,
                gradient_norm: 3.0,
                learning_rate: 0.005,
            },
        ),
        (
            "Late Training (Low Loss)",
            TrainingMetrics {
                loss: 0.3,
                gradient_norm: 0.5,
                learning_rate: 0.001,
            },
        ),
        (
            "Fine-tuning (Very Low Loss)",
            TrainingMetrics {
                loss: 0.1,
                gradient_norm: 0.1,
                learning_rate: 0.0001,
            },
        ),
    ];

    let test_gradient = randn::<f32>(&[2000, 2000])?;

    println!("Testing adaptive compression across training phases:");

    for (phase_name, training_metrics) in scenarios {
        let start_time = Instant::now();
        let (_compressed, metrics) = network_compressor
            .compress_gradient_adaptive(&test_gradient, Some(training_metrics.clone()))?;
        let total_time = start_time.elapsed();

        println!("\n  {} Results:", phase_name);
        println!("    Training Loss: {:.3}", training_metrics.loss);
        println!("    Gradient Norm: {:.3}", training_metrics.gradient_norm);
        println!("    Learning Rate: {:.6}", training_metrics.learning_rate);
        println!(
            "    Adapted Compression Ratio: {:.3}",
            metrics.compression_ratio
        );
        println!(
            "    Compression Time: {:.2}ms",
            metrics.compression_time_us as f64 / 1000.0
        );
        println!("    Throughput: {:.2}MB/s", metrics.throughput_mbps);
        println!(
            "    Memory Saved: {:.2}MB",
            metrics.memory_saved as f64 / 1_048_576.0
        );
        println!("    Total Time: {:.2}ms", total_time.as_millis());
    }

    // Get network performance metrics
    if let Some(network_metrics) = network_compressor.get_network_metrics()? {
        println!("\n  Current Network Performance:");
        println!("    Bandwidth: {:.2}MB/s", network_metrics.bandwidth_mbps);
        println!("    Latency: {:.2}ms", network_metrics.latency_ms);
        println!(
            "    Packet Loss: {:.4}%",
            network_metrics.packet_loss * 100.0
        );
        println!(
            "    Stability Score: {:.3}",
            network_metrics.stability_score
        );
    }

    // Get compression statistics
    let stats = network_compressor.get_compression_statistics()?;
    println!("\n  Compression Statistics:");
    println!("    Average Ratio: {:.3}", stats.average_compression_ratio);
    println!(
        "    Average Time: {:.2}μs",
        stats.average_compression_time_us
    );
    println!(
        "    Average Throughput: {:.2}MB/s",
        stats.average_throughput_mbps
    );
    println!("    Average Error: {:.6}", stats.average_compression_error);
    println!("    Total Compressions: {}", stats.total_compressions);

    println!("\n✅ Network-aware compression demo completed");
    Ok(())
}

/// Demonstrate comprehensive benchmarking suite
async fn demo_benchmarking_suite() -> TorshResult<()> {
    println!("\n📈 Demo 3: Comprehensive Benchmarking Suite");
    println!("--------------------------------------------");

    // Create benchmark configuration (reduced for demo)
    let benchmark_config = BenchmarkConfig {
        iterations: 20, // Reduced for faster demo
        tensor_sizes: vec![
            vec![500, 500],   // Medium tensor
            vec![1000, 1000], // Large tensor
        ],
        compression_methods: vec![
            CompressionMethod::TopK { k: 0.1 },
            CompressionMethod::Quantization { bits: 8 },
            CompressionMethod::SignSGD,
        ],
        compression_ratios: vec![0.05, 0.1, 0.2],
        include_warmup: true,
        warmup_iterations: 5,
        detailed_metrics: true,
    };

    // Create and run benchmark suite
    let mut benchmark_suite = EnhancedBenchmarkSuite::new(benchmark_config);

    println!("Running comprehensive benchmark suite...");
    println!("(This may take a few moments)");

    let start_time = Instant::now();
    let summary = benchmark_suite.run_complete_suite()?;
    let total_benchmark_time = start_time.elapsed();

    // Display results
    summary.print_summary();

    println!(
        "Benchmark suite completed in {:.2} seconds",
        total_benchmark_time.as_secs_f32()
    );

    // Export results to JSON (for analysis)
    let results_json = benchmark_suite.export_results_json()?;
    println!(
        "📄 Detailed results available in JSON format ({} characters)",
        results_json.len()
    );

    println!("\n✅ Benchmarking suite demo completed");
    Ok(())
}

/// Demonstrate performance comparison between different approaches
async fn demo_performance_comparison() -> TorshResult<()> {
    println!("\n⚡ Demo 4: Performance Comparison");
    println!("----------------------------------");

    let test_gradient = randn::<f32>(&[1500, 1500])?;

    // Test standard compression
    println!("Testing standard gradient compression:");
    let standard_config = CompressionConfig {
        method: CompressionMethod::TopK { k: 0.1 },
        compression_ratio: 0.1,
        ..CompressionConfig::default()
    };

    let mut standard_compressor =
        torsh_distributed::gradient_compression::GradientCompressor::new(standard_config.clone());
    let standard_start = Instant::now();
    let _standard_result = standard_compressor.compress(&test_gradient, "comparison")?;
    let standard_time = standard_start.elapsed();

    println!(
        "  Standard compression time: {:.2}ms",
        standard_time.as_millis()
    );

    // Test enhanced compression
    println!("\nTesting enhanced gradient compression:");
    let mut enhanced_compressor = EnhancedGradientCompressor::new(standard_config.clone())?;
    let enhanced_start = Instant::now();
    let (_enhanced_result, enhanced_metrics) =
        enhanced_compressor.compress_gradient_enhanced(&test_gradient, "comparison")?;
    let enhanced_time = enhanced_start.elapsed();

    println!(
        "  Enhanced compression time: {:.2}ms",
        enhanced_time.as_millis()
    );
    println!(
        "  Enhanced throughput: {:.2}MB/s",
        enhanced_metrics.throughput_mbps
    );
    println!(
        "  Enhanced optimized ops: {}",
        enhanced_metrics.optimized_ops_count
    );

    // Test network-aware compression
    println!("\nTesting network-aware adaptive compression:");
    let adaptive_config = AdaptiveCompressionConfig::default();
    let mut network_compressor = NetworkAwareCompressor::new(standard_config, adaptive_config)?;
    let training_metrics = TrainingMetrics {
        loss: 1.0,
        gradient_norm: 2.0,
        learning_rate: 0.001,
    };

    let network_start = Instant::now();
    let (_network_result, network_metrics) =
        network_compressor.compress_gradient_adaptive(&test_gradient, Some(training_metrics))?;
    let network_time = network_start.elapsed();

    println!(
        "  Network-aware compression time: {:.2}ms",
        network_time.as_millis()
    );
    println!(
        "  Network-aware throughput: {:.2}MB/s",
        network_metrics.throughput_mbps
    );
    println!(
        "  Adaptive compression ratio: {:.3}",
        network_metrics.compression_ratio
    );

    // Performance comparison
    println!("\n📊 Performance Summary:");
    println!(
        "  Standard:     {:.2}ms (baseline)",
        standard_time.as_millis()
    );
    println!(
        "  Enhanced:     {:.2}ms ({:.1}% {})",
        enhanced_time.as_millis(),
        ((standard_time.as_millis() as f64 - enhanced_time.as_millis() as f64)
            / standard_time.as_millis() as f64
            * 100.0)
            .abs(),
        if enhanced_time < standard_time {
            "faster"
        } else {
            "slower"
        }
    );
    println!(
        "  Network-aware: {:.2}ms ({:.1}% {})",
        network_time.as_millis(),
        ((standard_time.as_millis() as f64 - network_time.as_millis() as f64)
            / standard_time.as_millis() as f64
            * 100.0)
            .abs(),
        if network_time < standard_time {
            "faster"
        } else {
            "slower"
        }
    );

    println!("\n✅ Performance comparison demo completed");
    Ok(())
}

/// Demonstrate distributed monitoring system capabilities
async fn demo_distributed_monitoring() -> TorshResult<()> {
    println!("\n📡 Demo 5: Distributed Monitoring System");
    println!("----------------------------------------");

    // Create monitoring configuration
    let monitor_config = MonitoringConfig {
        collection_interval: std::time::Duration::from_millis(500),
        enable_gpu_monitoring: true,
        enable_comm_analysis: true,
        enable_anomaly_detection: true,
        ..MonitoringConfig::default()
    };

    // Initialize distributed monitor
    let monitor = Arc::new(DistributedMonitor::new(monitor_config, true));

    println!("Starting distributed monitoring system...");

    // Simulate some training activity for a few seconds
    println!("Simulating training activity and collecting metrics...");
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Get current metrics
    if let Ok(Some(node_metrics)) = monitor.get_current_metrics() {
        println!("\n  📊 Current Node Metrics:");
        println!(
            "    CPU Utilization: {:.1}%",
            node_metrics.system_metrics.cpu_utilization
        );
        println!(
            "    GPU Utilization: {:.1}%",
            node_metrics.system_metrics.gpu_utilization
        );
        println!(
            "    Memory Usage: {:.1}MB",
            node_metrics.system_metrics.memory_usage_mb
        );
        println!(
            "    Training Epoch: {}",
            node_metrics.training_metrics.epoch
        );
        println!(
            "    Current Loss: {:.6}",
            node_metrics.training_metrics.loss
        );
        println!(
            "    Throughput: {:.2} samples/sec",
            node_metrics.training_metrics.throughput_samples_per_sec
        );
        println!(
            "    Gradient Norm: {:.6}",
            node_metrics.training_metrics.gradient_norm
        );
    }

    // Get cluster summary
    if let Ok(cluster_summary) = monitor.get_cluster_summary() {
        println!("\n  🌐 Cluster Summary:");
        println!("    Total Nodes: {}", cluster_summary.total_nodes);
        println!("    Healthy Nodes: {}", cluster_summary.healthy_nodes);
        println!(
            "    Average CPU: {:.1}%",
            cluster_summary.avg_cpu_utilization
        );
        println!(
            "    Average GPU: {:.1}%",
            cluster_summary.avg_gpu_utilization
        );
        println!(
            "    Total Throughput: {:.2} samples/sec",
            cluster_summary.total_throughput
        );
    }

    // Check for active alerts (includes anomaly detection)
    let alerts = monitor.get_active_alerts()?;
    if !alerts.is_empty() {
        println!("\n  🚨 Active Alerts:");
        for alert in alerts.iter().take(3) {
            println!(
                "    - {}: {} (severity: {})",
                alert.metric_name, alert.message, alert.severity
            );
        }
    } else {
        println!("\n  ✅ No alerts - system operating normally");
    }

    // Monitor is automatically stopped when dropped

    println!("\n✅ Distributed monitoring demo completed");
    Ok(())
}

/// Demonstrate enhanced fault tolerance mechanisms
async fn demo_enhanced_fault_tolerance() -> TorshResult<()> {
    println!("\n🛡️ Demo 6: Enhanced Fault Tolerance");
    println!("------------------------------------");

    // Create fault tolerance configuration
    let ft_config = FaultToleranceConfig {
        enable_predictive_detection: true,
        enable_automatic_recovery: true,
        max_recovery_attempts: 3,
        node_timeout: std::time::Duration::from_secs(10),
        communication_timeout: std::time::Duration::from_secs(5),
        ..FaultToleranceConfig::default()
    };

    // Create monitoring for fault tolerance
    let monitor_config = MonitoringConfig::default();
    let monitor = Arc::new(DistributedMonitor::new(monitor_config, false));

    // Initialize enhanced fault tolerance
    let fault_tolerance = Arc::new(EnhancedFaultTolerance::new(ft_config, monitor.clone()));

    println!("Initializing enhanced fault tolerance system...");

    // Simulate some operations
    println!("Simulating training operations with fault monitoring...");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Get fault tolerance status
    let status = fault_tolerance.get_status()?;
    println!("\n  📊 Fault Tolerance Status:");
    println!("    System Health Score: {:.3}", status.system_health_score);
    println!("    Total Nodes: {}", status.total_nodes);
    println!("    Healthy Nodes: {}", status.healthy_nodes);
    println!("    Active Incidents: {}", status.active_incidents);
    println!("    Recovering Incidents: {}", status.recovering_incidents);

    // Demonstrate failure detection capabilities
    println!("\n  🔍 Checking failure detection capabilities...");
    let failures = fault_tolerance.detect_failures()?;
    if failures.is_empty() {
        println!("    ✅ No failures detected - system operating normally");
    } else {
        println!("    🚨 Detected {} potential failure(s)", failures.len());
        for failure in failures.iter().take(3) {
            println!("    - {}", failure);
        }
    }

    println!("\n✅ Enhanced fault tolerance demo completed");
    Ok(())
}

/// Demonstrate distributed memory optimization
async fn demo_distributed_memory_optimization() -> TorshResult<()> {
    println!("\n💾 Demo 7: Distributed Memory Optimization");
    println!("-------------------------------------------");

    // Create memory optimization configuration
    let mem_config = MemoryOptimizationConfig {
        enable_cross_node_balancing: true,
        enable_predictive_management: true,
        pressure_threshold: 0.85,
        prediction_window: std::time::Duration::from_secs(600), // 10 minutes
        max_concurrent_optimizations: 3,
        optimization_interval: std::time::Duration::from_secs(5),
        ..MemoryOptimizationConfig::default()
    };

    // Create monitoring for memory optimization
    let monitor_config = MonitoringConfig::default();
    let monitor = Arc::new(DistributedMonitor::new(monitor_config, false));

    // Initialize distributed memory optimizer
    let memory_optimizer = Arc::new(DistributedMemoryOptimizer::new(mem_config, monitor.clone()));

    println!("Initializing distributed memory optimization system...");

    // Simulate memory-intensive operations
    println!("Simulating memory-intensive training operations...");

    // Create large tensors to simulate memory usage
    let large_tensors = [
        randn::<f32>(&[1000, 1000])?,
        randn::<f32>(&[1500, 1500])?,
        randn::<f32>(&[800, 800])?,
    ];

    for (i, tensor) in large_tensors.iter().enumerate() {
        let node_id = format!("node_{}", i);
        let size_mb = (tensor.numel() * 4) as u64 / 1_048_576; // Convert bytes to MB
        memory_optimizer.track_allocation(node_id, size_mb, format!("tensor_{}", i), true)?;
    }

    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Get optimization status
    let status = memory_optimizer.get_optimization_status()?;
    println!("\n  📊 Memory Optimization Status:");
    println!(
        "    Average Memory Utilization: {:.1}%",
        status.avg_memory_utilization
    );
    println!(
        "    Average Pressure Score: {:.3}",
        status.avg_pressure_score
    );
    println!("    Active Optimizations: {}", status.active_optimizations);
    println!("    Total Memory (MB): {:.2}", status.total_memory_mb);
    println!(
        "    Optimization Efficiency: {:.1}%",
        status.optimization_efficiency * 100.0
    );

    // Analyze optimization opportunities
    println!("\n  🚀 Analyzing memory optimization opportunities...");
    let optimization_actions = memory_optimizer.analyze_optimization_opportunities()?;
    if !optimization_actions.is_empty() {
        println!(
            "    Found {} optimization opportunities",
            optimization_actions.len()
        );
        for action in optimization_actions.iter().take(3) {
            println!(
                "    - {:?} on {} (Priority: {})",
                action.technique, action.target_node, action.priority
            );
        }
    } else {
        println!("    ✅ No immediate optimization opportunities found");
    }

    println!("\n✅ Distributed memory optimization demo completed");
    Ok(())
}

/// Demonstrate training analytics dashboard
async fn demo_training_analytics_dashboard() -> TorshResult<()> {
    println!("\n📊 Demo 8: Training Analytics Dashboard");
    println!("---------------------------------------");

    // Create dashboard configuration
    let dashboard_config = DashboardConfig {
        update_interval: std::time::Duration::from_secs(2),
        retention_period: std::time::Duration::from_secs(3600),
        enable_predictions: true,
        enable_recommendations: true,
        aggregation_window: std::time::Duration::from_secs(60),
        ..DashboardConfig::default()
    };

    // Create supporting systems
    let monitor_config = MonitoringConfig::default();
    let monitor = Arc::new(DistributedMonitor::new(monitor_config, false));

    let ft_config = FaultToleranceConfig::default();
    let fault_tolerance = Arc::new(EnhancedFaultTolerance::new(ft_config, monitor.clone()));

    let mem_config = MemoryOptimizationConfig::default();
    let memory_optimizer = Arc::new(DistributedMemoryOptimizer::new(mem_config, monitor.clone()));

    // Initialize training analytics dashboard
    let dashboard = TrainingAnalyticsDashboard::new(
        dashboard_config,
        monitor.clone(),
        fault_tolerance.clone(),
        memory_optimizer.clone(),
    );

    println!("Initializing training analytics dashboard...");

    // Initialize supporting systems (systems are automatically initialized)

    // Simulate training activity
    println!("Simulating training activity and gathering analytics...");
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    // Update analytics
    dashboard.update_analytics()?;

    // Get current analytics
    if let Some(analytics) = dashboard.get_current_analytics()? {
        println!("\n  📊 Training Performance Analytics:");
        println!("    Current Epoch: {}", analytics.performance.current_epoch);
        println!("    Average Loss: {:.6}", analytics.performance.avg_loss);
        println!("    Loss Trend: {:.6}", analytics.performance.loss_trend);
        println!(
            "    Cluster Throughput: {:.2} samples/sec",
            analytics.performance.cluster_throughput
        );
        println!(
            "    Throughput Efficiency: {:.1}%",
            analytics.performance.throughput_efficiency * 100.0
        );
        println!(
            "    Training Stability: {:.3}",
            analytics.performance.training_stability
        );

        println!("\n  💾 Resource Utilization Analytics:");
        println!(
            "    CPU Utilization: {:.1}%",
            analytics.resource_utilization.avg_cpu_utilization
        );
        println!(
            "    GPU Utilization: {:.1}%",
            analytics.resource_utilization.avg_gpu_utilization
        );
        println!(
            "    Memory Utilization: {:.1}%",
            analytics.resource_utilization.avg_memory_utilization
        );
        println!(
            "    Resource Efficiency: {:.3}",
            analytics.resource_utilization.resource_efficiency
        );
        println!(
            "    Primary Bottleneck: {}",
            analytics.resource_utilization.primary_bottleneck
        );

        println!("\n  📡 Communication Analytics:");
        println!(
            "    Average Latency: {}μs",
            analytics.communication.avg_latency_us
        );
        println!(
            "    Bandwidth Utilization: {:.1}%",
            analytics.communication.bandwidth_utilization * 100.0
        );
        println!(
            "    Communication Efficiency: {:.3}",
            analytics.communication.efficiency_score
        );
        println!(
            "    Congestion Level: {:.1}%",
            analytics.communication.congestion_level * 100.0
        );

        println!("\n  ❤️  System Health Analytics:");
        println!(
            "    Cluster Health Score: {:.3}",
            analytics.system_health.cluster_health_score
        );
        println!(
            "    Healthy Nodes: {}",
            analytics.system_health.healthy_nodes
        );
        println!("    Failed Nodes: {}", analytics.system_health.failed_nodes);
        println!(
            "    Active Incidents: {}",
            analytics.system_health.active_incidents
        );
        println!(
            "    Failure Probability: {:.1}%",
            analytics.system_health.failure_probability * 100.0
        );

        println!("\n  📈 Convergence Analytics:");
        println!(
            "    Convergence Rate: {:.6}",
            analytics.convergence.convergence_rate
        );
        println!(
            "    Training Progress: {:.1}%",
            analytics.convergence.training_progress * 100.0
        );
        println!(
            "    Convergence Confidence: {:.3}",
            analytics.convergence.convergence_confidence
        );
        println!(
            "    Overfitting Risk: {:.1}%",
            analytics.convergence.overfitting_risk * 100.0
        );

        println!("\n  ⚡ Efficiency Analytics:");
        println!(
            "    Overall Efficiency: {:.3}",
            analytics.efficiency.overall_efficiency
        );
        println!(
            "    Compute Efficiency: {:.3}",
            analytics.efficiency.compute_efficiency
        );
        println!(
            "    Communication Efficiency: {:.3}",
            analytics.efficiency.communication_efficiency
        );
        println!(
            "    Memory Efficiency: {:.3}",
            analytics.efficiency.memory_efficiency
        );

        // Show optimization recommendations
        if !analytics.efficiency.recommendations.is_empty() {
            println!("\n  💡 Optimization Recommendations:");
            for rec in analytics.efficiency.recommendations.iter().take(3) {
                println!(
                    "    - {} (Category: {}, Priority: {}, Impact: {:.1}%)",
                    rec.title,
                    rec.category,
                    rec.priority,
                    rec.expected_impact * 100.0
                );
                println!("      {}", rec.description);
            }
        }
    }

    // Generate training summary
    if let Ok(summary) = dashboard.generate_training_summary() {
        println!("\n  📋 Training Summary Report:");
        println!(
            "    Total Runtime: {:.2} seconds",
            summary.total_runtime.as_secs_f32()
        );
        println!("    Average Efficiency: {:.3}", summary.avg_efficiency);
        println!(
            "    Peak Throughput: {:.2} samples/sec",
            summary.peak_throughput
        );
        println!("    Total Incidents: {}", summary.total_incidents);
        println!("    Convergence Rate: {:.6}", summary.convergence_rate);
    }

    // Export dashboard data
    if let Ok(export) = dashboard.export_dashboard_data() {
        println!("\n  📄 Dashboard Data Export:");
        println!(
            "    Analytics History Length: {}",
            export.analytics_history.len()
        );
        println!(
            "    Export Size: {} KB",
            serde_json::to_string(&export)
                .map(|s| s.len() / 1024)
                .unwrap_or(0)
        );
    }

    // Clean up (systems are automatically cleaned up when dropped)

    println!("\n✅ Training analytics dashboard demo completed");
    Ok(())
}
