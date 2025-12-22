//! Grafana dashboard integration demo
//!
//! This example demonstrates how to create Grafana dashboards for
//! visualizing profiling metrics.

use torsh_profiler::{DashboardTemplates, GrafanaDashboardGenerator, GridPos};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Grafana Dashboard Integration Demo ===\n");

    // Create a custom dashboard
    println!("1. Creating custom dashboard...");
    create_custom_dashboard()?;

    // Create pre-built dashboards
    println!("\n2. Creating profiling overview dashboard...");
    create_profiling_dashboard()?;

    println!("\n3. Creating memory profiling dashboard...");
    create_memory_dashboard()?;

    println!("\n4. Creating performance metrics dashboard...");
    create_performance_dashboard()?;

    // Demonstrate advanced dashboard features
    println!("\n5. Creating advanced custom dashboard...");
    create_advanced_dashboard()?;

    println!("\n✅ Grafana dashboard generation completed!");
    println!("\nGenerated dashboards can be imported into Grafana:");
    println!("  1. Open Grafana UI");
    println!("  2. Navigate to Dashboards → Import");
    println!("  3. Upload the JSON file or paste the JSON content");
    println!("  4. Configure data source (Prometheus)");

    Ok(())
}

fn create_custom_dashboard() -> Result<(), Box<dyn std::error::Error>> {
    let mut dashboard = GrafanaDashboardGenerator::new("ToRSh Custom Dashboard")
        .with_uid("torsh-custom")
        .with_tags(vec!["custom".to_string(), "demo".to_string()])
        .with_time_range("now-5m", "now")
        .with_refresh("5s");

    // Add operation count graph
    dashboard.add_graph_panel(
        "Total Operations",
        "sum(torsh_operation_total)",
        "Total",
        GridPos {
            h: 8,
            w: 12,
            x: 0,
            y: 0,
        },
    );

    // Add memory usage graph
    dashboard.add_graph_panel(
        "Memory Usage",
        "torsh_memory_allocated_bytes - torsh_memory_deallocated_bytes",
        "Net Memory",
        GridPos {
            h: 8,
            w: 12,
            x: 12,
            y: 0,
        },
    );

    // Add FLOPS gauge
    dashboard.add_gauge_panel(
        "FLOPS",
        "sum(rate(torsh_flops_total[1m]))",
        "ops",
        0.0,
        1e9,
        GridPos {
            h: 6,
            w: 8,
            x: 0,
            y: 8,
        },
    );

    // Add overhead stat
    dashboard.add_stat_panel(
        "Profiling Overhead",
        "avg(torsh_profiling_overhead_microseconds)",
        "µs",
        GridPos {
            h: 6,
            w: 8,
            x: 8,
            y: 8,
        },
    );

    // Add active threads stat
    dashboard.add_stat_panel(
        "Active Threads",
        "count(torsh_thread_activity)",
        "short",
        GridPos {
            h: 6,
            w: 8,
            x: 16,
            y: 8,
        },
    );

    // Export to file
    let filename = "/tmp/torsh_custom_dashboard.json";
    dashboard.export_to_file(filename)?;
    println!("✓ Custom dashboard saved to: {}", filename);

    // Print preview
    let json = dashboard.export_json()?;
    println!(
        "Dashboard preview (first 500 chars):\n{}",
        &json[..json.len().min(500)]
    );

    Ok(())
}

fn create_profiling_dashboard() -> Result<(), Box<dyn std::error::Error>> {
    let dashboard = DashboardTemplates::create_profiling_dashboard();
    let filename = "/tmp/torsh_profiling_overview.json";
    dashboard.export_to_file(filename)?;
    println!("✓ Profiling overview dashboard saved to: {}", filename);

    let panel_count = dashboard.dashboard().panels.len();
    println!("  - {} panels configured", panel_count);
    println!("  - Includes: duration graphs, rate graphs, memory gauges, FLOPS metrics");

    Ok(())
}

fn create_memory_dashboard() -> Result<(), Box<dyn std::error::Error>> {
    let dashboard = DashboardTemplates::create_memory_dashboard();
    let filename = "/tmp/torsh_memory_profiling.json";
    dashboard.export_to_file(filename)?;
    println!("✓ Memory profiling dashboard saved to: {}", filename);

    let panel_count = dashboard.dashboard().panels.len();
    println!("  - {} panels configured", panel_count);
    println!("  - Includes: allocation/deallocation graphs, net memory usage, rates");

    Ok(())
}

fn create_performance_dashboard() -> Result<(), Box<dyn std::error::Error>> {
    let dashboard = DashboardTemplates::create_performance_dashboard();
    let filename = "/tmp/torsh_performance_metrics.json";
    dashboard.export_to_file(filename)?;
    println!("✓ Performance metrics dashboard saved to: {}", filename);

    let panel_count = dashboard.dashboard().panels.len();
    println!("  - {} panels configured", panel_count);
    println!("  - Includes: FLOPS, throughput, latency percentiles, operation duration");

    Ok(())
}

fn create_advanced_dashboard() -> Result<(), Box<dyn std::error::Error>> {
    let mut dashboard = GrafanaDashboardGenerator::new("ToRSh Advanced Monitoring")
        .with_uid("torsh-advanced")
        .with_tags(vec![
            "advanced".to_string(),
            "monitoring".to_string(),
            "analytics".to_string(),
        ])
        .with_time_range("now-1h", "now")
        .with_refresh("10s");

    // Add operation variable for filtering
    dashboard.add_variable(
        "operation",
        "label_values(torsh_operation_total, operation)",
        true,
        true,
    );

    // Add thread variable
    dashboard.add_variable(
        "thread",
        "label_values(torsh_operation_duration_microseconds, thread_id)",
        true,
        false,
    );

    // Row 1: Operation metrics
    dashboard.add_graph_panel(
        "Operation Duration (P50, P95, P99)",
        r#"histogram_quantile(0.50, sum(rate(torsh_operation_duration_microseconds_bucket{operation=~"$operation"}[5m])) by (le, operation))"#,
        "P50 - {{operation}}",
        GridPos { h: 8, w: 8, x: 0, y: 0 },
    );

    dashboard.add_graph_panel(
        "Operation Rate by Thread",
        r#"sum(rate(torsh_operation_total{operation=~"$operation"}[5m])) by (thread_id)"#,
        "Thread {{thread_id}}",
        GridPos {
            h: 8,
            w: 8,
            x: 8,
            y: 0,
        },
    );

    dashboard.add_graph_panel(
        "Bytes Transferred by Direction",
        r#"sum(rate(torsh_bytes_transferred_total{operation=~"$operation"}[5m])) by (direction)"#,
        "{{direction}}",
        GridPos {
            h: 8,
            w: 8,
            x: 16,
            y: 0,
        },
    );

    // Row 2: Resource utilization
    dashboard.add_heatmap_panel(
        "Operation Duration Distribution",
        r#"sum(rate(torsh_operation_duration_microseconds_bucket{operation=~"$operation"}[5m])) by (le)"#,
        GridPos { h: 8, w: 12, x: 0, y: 8 },
    );

    dashboard.add_graph_panel(
        "Memory Efficiency",
        r#"rate(torsh_flops_total{operation=~"$operation"}[5m]) / (torsh_memory_allocated_bytes{operation=~"$operation"} + 1)"#,
        "{{operation}}",
        GridPos { h: 8, w: 12, x: 12, y: 8 },
    );

    // Row 3: Performance indicators
    dashboard.add_gauge_panel(
        "System Throughput",
        "sum(rate(torsh_operation_total[1m]))",
        "ops/s",
        0.0,
        10000.0,
        GridPos {
            h: 6,
            w: 6,
            x: 0,
            y: 16,
        },
    );

    dashboard.add_gauge_panel(
        "Avg FLOPS",
        "avg(rate(torsh_flops_total[1m]))",
        "flops",
        0.0,
        1e8,
        GridPos {
            h: 6,
            w: 6,
            x: 6,
            y: 16,
        },
    );

    dashboard.add_stat_panel(
        "Total Operations (24h)",
        "sum(increase(torsh_operation_total[24h]))",
        "short",
        GridPos {
            h: 6,
            w: 6,
            x: 12,
            y: 16,
        },
    );

    dashboard.add_stat_panel(
        "Profiling Overhead %",
        "(sum(torsh_profiling_overhead_microseconds) / sum(torsh_operation_duration_microseconds)) * 100",
        "percent",
        GridPos { h: 6, w: 6, x: 18, y: 16 },
    );

    // Export
    let filename = "/tmp/torsh_advanced_monitoring.json";
    dashboard.export_to_file(filename)?;
    println!("✓ Advanced monitoring dashboard saved to: {}", filename);

    let panel_count = dashboard.dashboard().panels.len();
    let var_count = dashboard.dashboard().templating.list.len();
    println!("  - {} panels configured", panel_count);
    println!("  - {} variables (filters): operation, thread", var_count);
    println!("  - Includes: percentiles, heatmaps, efficiency metrics, throughput gauges");

    Ok(())
}
