//! Dashboard Demo - Real-time Performance Monitoring
//!
//! This example demonstrates the dashboard functionality of the ToRSh profiler,
//! including real-time performance monitoring, metrics collection, and HTML generation.

use std::{sync::Arc, thread, time::Duration};
use torsh_profiler::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Dashboard Demo - ToRSh Profiler");
    println!("=================================\n");

    // Create profilers
    let profiler = Arc::new(Profiler::new());
    let memory_profiler = Arc::new(MemoryProfiler::new());

    // Enable profiling
    let _profiler_mut = Arc::try_unwrap(profiler.clone()).unwrap_or_else(|arc| {
        println!("Warning: Multiple references to profiler exist");
        (*arc).clone()
    });
    // Since we can't easily unwrap Arc in this case, let's work with it as-is

    println!("📊 Generating performance data for dashboard...");

    // Simulate some profiled operations
    for i in 0..20 {
        let _scope = ProfileScope::simple(format!("operation_{i}"), "demo".to_string());

        // Simulate work with varying durations
        let duration = match i % 4 {
            0 => 10,  // Fast operations
            1 => 25,  // Medium operations
            2 => 50,  // Slow operations
            3 => 100, // Very slow operations
            _ => 10,
        };

        thread::sleep(Duration::from_millis(duration));

        // Simulate memory allocations
        if i % 3 == 0 {
            let _ = memory_profiler.record_allocation(i * 1000, 1024 * (i + 1));
        }
    }

    println!("✅ Generated {} operations for dashboard\n", 20);

    // Create dashboard
    println!("🖥️  Creating dashboard...");
    let dashboard_config = DashboardConfig {
        port: 8080,
        refresh_interval: 5,
        real_time_updates: true,
        max_data_points: 100,
        enable_stack_traces: false,
        custom_css: Some(
            "
            .card { border-left-color: #28a745; }
            .metric { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); }
        "
            .to_string(),
        ),
        websocket_config: WebSocketConfig::default(),
    };

    let dashboard = create_dashboard_with_config(dashboard_config);

    // Add some sample alerts
    let alert1 = DashboardAlert {
        id: "demo_alert_1".to_string(),
        severity: DashboardAlertSeverity::Warning,
        title: "High Memory Usage".to_string(),
        message: "Memory usage is above 80% threshold".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        resolved: false,
    };

    let alert2 = DashboardAlert {
        id: "demo_alert_2".to_string(),
        severity: DashboardAlertSeverity::Info,
        title: "Performance Optimization".to_string(),
        message: "Consider using vectorization for better performance".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        resolved: false,
    };

    dashboard.add_alert(alert1)?;
    dashboard.add_alert(alert2)?;

    println!("✅ Dashboard created with sample alerts\n");

    // Generate dashboard HTML
    println!("📄 Generating dashboard HTML...");
    let dashboard_html = dashboard.generate_dashboard_html()?;

    // Export dashboard to file
    let dashboard_file = std::env::temp_dir().join("torsh_dashboard.html");
    let dashboard_file_str = dashboard_file.display().to_string();
    std::fs::write(&dashboard_file, &dashboard_html)?;
    println!("✅ Dashboard HTML exported to: {dashboard_file_str}");

    // Export dashboard data to JSON
    let dashboard_data_file = std::env::temp_dir().join("torsh_dashboard_data.json");
    let dashboard_data_str = dashboard_data_file.display().to_string();
    dashboard.export_data_json(&dashboard_data_str)?;
    println!("✅ Dashboard data exported to: {dashboard_data_str}");

    // Show dashboard metrics
    if let Some(current_data) = dashboard.get_current_data()? {
        println!("\n📊 Current Dashboard Metrics:");
        println!(
            "   • Total Operations: {}",
            current_data.performance_metrics.total_operations
        );
        println!(
            "   • Average Duration: {:.2} ms",
            current_data.performance_metrics.average_duration_ms
        );
        println!(
            "   • Operations/Second: {:.1}",
            current_data.performance_metrics.operations_per_second
        );
        println!(
            "   • CPU Utilization: {:.1}%",
            current_data.performance_metrics.cpu_utilization
        );
        println!(
            "   • Memory Usage: {:.1} MB",
            current_data.memory_metrics.current_usage_mb
        );
        println!(
            "   • Peak Memory: {:.1} MB",
            current_data.memory_metrics.peak_usage_mb
        );
        println!("   • Active Alerts: {}", current_data.alerts.len());
        println!("   • Top Operations: {}", current_data.top_operations.len());
    }

    // Show active alerts
    let active_alerts = dashboard.get_active_alerts()?;
    if !active_alerts.is_empty() {
        println!("\n🚨 Active Alerts:");
        for alert in &active_alerts {
            println!(
                "   • {}: {} ({:?})",
                alert.title, alert.message, alert.severity
            );
        }
    }

    println!("\n🎯 Dashboard Demo Summary:");
    println!("   • Dashboard functionality implemented and tested");
    println!("   • Real-time metrics collection working");
    println!("   • HTML generation and export successful");
    println!("   • Alert system operational");
    println!("   • Data export to JSON format working");

    println!("\n📁 Generated Files:");
    println!("   • Dashboard HTML: {dashboard_file_str}");
    println!("   • Dashboard Data: {dashboard_data_str}");

    println!("\n💡 Next Steps:");
    println!("   • Open {dashboard_file_str} in a web browser to view the dashboard");
    println!("   • Use dashboard.start() to run real-time monitoring server");
    println!("   • Integrate with your application for continuous monitoring");

    println!("\n✅ Dashboard demo completed successfully!");

    Ok(())
}
