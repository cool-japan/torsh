//! Dashboard HTML generation and web interface functionality
//!
//! This module provides comprehensive HTML generation capabilities for the ToRSh
//! performance dashboard, including responsive layouts, interactive features,
//! and customizable themes.

use crate::{MemoryProfiler, Profiler, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::TorshError;

use super::types::{
    DashboardAlert, DashboardAlertSeverity, DashboardConfig, DashboardData, DashboardTheme,
    MemoryMetrics, OperationSummary, PerformanceMetrics, SystemMetrics,
};
use super::{create_dashboard, create_dashboard_with_config, Dashboard};

// =============================================================================
// Core HTML Generation
// =============================================================================

/// Generate complete dashboard HTML interface
pub fn generate_dashboard_html(dashboard: &Dashboard) -> TorshResult<String> {
    dashboard.generate_dashboard_html()
}

/// Create default dashboard data for empty state
fn create_default_data() -> DashboardData {
    DashboardData {
        timestamp: 0,
        performance_metrics: PerformanceMetrics {
            total_operations: 0,
            average_duration_ms: 0.0,
            operations_per_second: 0.0,
            total_flops: 0,
            gflops_per_second: 0.0,
            cpu_utilization: 0.0,
            thread_count: 0,
        },
        memory_metrics: MemoryMetrics {
            current_usage_mb: 0.0,
            peak_usage_mb: 0.0,
            total_allocations: 0,
            total_deallocations: 0,
            active_allocations: 0,
            fragmentation_ratio: 0.0,
            allocation_rate: 0.0,
        },
        system_metrics: SystemMetrics {
            uptime_seconds: 0,
            load_average: 0.0,
            available_memory_mb: 0.0,
            disk_usage_percent: 0.0,
            network_io_mbps: 0.0,
        },
        alerts: Vec::new(),
        top_operations: Vec::new(),
    }
}

/// Build complete HTML document structure
pub fn build_html_document(
    data: &DashboardData,
    alerts: &[DashboardAlert],
    config: &DashboardConfig,
    theme: &DashboardTheme,
) -> TorshResult<String> {
    let head_section = generate_head_section(theme, config);
    let header_section = generate_header_section();
    let dashboard_section = generate_dashboard_section(data, alerts);
    let scripts_section = generate_scripts_section(config);
    let footer_section = generate_footer_section(data);

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
{head_section}
<body>
    {header_section}
    {dashboard_section}
    {footer_section}
    {scripts_section}
</body>
</html>"#
    );

    Ok(html)
}

// =============================================================================
// HTML Section Generators
// =============================================================================

/// Generate HTML head section with styles and metadata
fn generate_head_section(theme: &DashboardTheme, config: &DashboardConfig) -> String {
    let base_styles = generate_base_css(theme);
    let custom_css = config.custom_css.as_deref().unwrap_or("");

    format!(
        r#"<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="ToRSh Performance Dashboard - Real-time monitoring">
    <title>ToRSh Performance Dashboard</title>
    <style>
        {base_styles}
        {custom_css}
    </style>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚀</text></svg>">
</head>"#
    )
}

/// Generate dashboard header section
fn generate_header_section() -> String {
    r#"<div class="header">
        <h1>🚀 ToRSh Performance Dashboard</h1>
        <p>Real-time performance monitoring and analytics</p>
        <div class="header-controls">
            <button class="refresh-button" onclick="refreshDashboard()">🔄 Refresh</button>
            <button class="settings-button" onclick="toggleSettings()">⚙️ Settings</button>
            <button class="fullscreen-button" onclick="toggleFullscreen()">⛶ Fullscreen</button>
        </div>
    </div>"#
        .to_string()
}

/// Generate main dashboard content section
fn generate_dashboard_section(data: &DashboardData, alerts: &[DashboardAlert]) -> String {
    let performance_card = generate_performance_card(&data.performance_metrics);
    let memory_card = generate_memory_card(&data.memory_metrics);
    let system_card = generate_system_card(&data.system_metrics);
    let alerts_card = generate_alerts_card(alerts);
    let operations_card = generate_operations_card(&data.top_operations);
    let charts_card = generate_charts_card(data);

    format!(
        r#"<div class="dashboard">
        {performance_card}
        {memory_card}
        {system_card}
        {alerts_card}
        {operations_card}
        {charts_card}
    </div>"#
    )
}

/// Generate JavaScript section for interactivity
fn generate_scripts_section(config: &DashboardConfig) -> String {
    let websocket_config = if config.websocket_config.enabled {
        format!(
            "const WEBSOCKET_URL = 'ws://localhost:{}/ws';",
            config.websocket_config.port
        )
    } else {
        "const WEBSOCKET_URL = null;".to_string()
    };

    format!(
        r#"<script>
        {websocket_config}
        const REFRESH_INTERVAL = {};

        {}
    </script>"#,
        config.refresh_interval * 1000,
        include_str!("../../assets/dashboard.js") // Would contain dashboard JavaScript
    )
}

/// Generate footer section with timestamp
fn generate_footer_section(data: &DashboardData) -> String {
    let timestamp = chrono::DateTime::from_timestamp(data.timestamp as i64, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "Unknown".to_string());

    format!(
        r#"<div class="footer">
        <div class="status-indicator">
            <span class="status-dot active"></span>
            <span>Live</span>
        </div>
        <div class="timestamp">Last updated: {timestamp}</div>
        <div class="attribution">Powered by ToRSh Dashboard</div>
    </div>"#
    )
}

// =============================================================================
// Card Generators
// =============================================================================

/// Generate performance metrics card
fn generate_performance_card(metrics: &PerformanceMetrics) -> String {
    format!(
        r#"<div class="card performance-card">
            <h3><span class="card-icon">⚡</span>Performance Metrics</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">Total Operations</span>
                    <span class="metric-value">{}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Average Duration</span>
                    <span class="metric-value">{:.2} ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Operations/Second</span>
                    <span class="metric-value">{:.1}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">GFLOPS/Second</span>
                    <span class="metric-value">{:.2}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CPU Utilization</span>
                    <span class="metric-value">{:.1}%</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {:.1}%"></div>
                    </div>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Threads</span>
                    <span class="metric-value">{}</span>
                </div>
            </div>
        </div>"#,
        metrics.total_operations,
        metrics.average_duration_ms,
        metrics.operations_per_second,
        metrics.gflops_per_second,
        metrics.cpu_utilization,
        metrics.cpu_utilization.min(100.0),
        metrics.thread_count
    )
}

/// Generate memory metrics card
fn generate_memory_card(metrics: &MemoryMetrics) -> String {
    let usage_percentage = if metrics.peak_usage_mb > 0.0 {
        (metrics.current_usage_mb / metrics.peak_usage_mb * 100.0).min(100.0)
    } else {
        0.0
    };

    format!(
        r#"<div class="card memory-card">
            <h3><span class="card-icon">💾</span>Memory Metrics</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">Current Usage</span>
                    <span class="metric-value">{:.1} MB</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {:.1}%"></div>
                    </div>
                </div>
                <div class="metric">
                    <span class="metric-label">Peak Usage</span>
                    <span class="metric-value">{:.1} MB</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Allocations</span>
                    <span class="metric-value">{}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Active Allocations</span>
                    <span class="metric-value">{}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Fragmentation</span>
                    <span class="metric-value">{:.1}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Allocation Rate</span>
                    <span class="metric-value">{:.2}/s</span>
                </div>
            </div>
        </div>"#,
        metrics.current_usage_mb,
        usage_percentage,
        metrics.peak_usage_mb,
        metrics.total_allocations,
        metrics.active_allocations,
        metrics.fragmentation_ratio * 100.0,
        metrics.allocation_rate
    )
}

/// Generate system metrics card
fn generate_system_card(metrics: &SystemMetrics) -> String {
    let uptime_formatted = format_duration(metrics.uptime_seconds);

    format!(
        r#"<div class="card system-card">
            <h3><span class="card-icon">🖥️</span>System Metrics</h3>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value">{}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Load Average</span>
                    <span class="metric-value">{:.2}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Available Memory</span>
                    <span class="metric-value">{:.1} MB</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Disk Usage</span>
                    <span class="metric-value">{:.1}%</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {:.1}%"></div>
                    </div>
                </div>
                <div class="metric">
                    <span class="metric-label">Network I/O</span>
                    <span class="metric-value">{:.2} MB/s</span>
                </div>
            </div>
        </div>"#,
        uptime_formatted,
        metrics.load_average,
        metrics.available_memory_mb,
        metrics.disk_usage_percent,
        metrics.disk_usage_percent.min(100.0),
        metrics.network_io_mbps
    )
}

/// Generate alerts card
fn generate_alerts_card(alerts: &[DashboardAlert]) -> String {
    let alerts_content = if alerts.is_empty() {
        "<div class=\"no-alerts\">✅ No active alerts</div>".to_string()
    } else {
        alerts
            .iter()
            .map(|alert| {
                let (class, icon) = match alert.severity {
                    DashboardAlertSeverity::Emergency => ("alert-emergency", "🚨"),
                    DashboardAlertSeverity::Critical => ("alert-critical", "🔴"),
                    DashboardAlertSeverity::Warning => ("alert-warning", "⚠️"),
                    DashboardAlertSeverity::Info => ("alert-info", "ℹ️"),
                };

                format!(
                    r#"<div class="alert {class}">
                        <div class="alert-header">
                            <span class="alert-icon">{icon}</span>
                            <span class="alert-title">{}</span>
                            <span class="alert-time">{}</span>
                        </div>
                        <div class="alert-message">{}</div>
                    </div>"#,
                    alert.title,
                    format_timestamp(alert.timestamp),
                    alert.message
                )
            })
            .collect::<Vec<_>>()
            .join("")
    };

    format!(
        r#"<div class="card alerts-card">
            <h3><span class="card-icon">🚨</span>Active Alerts <span class="alert-count">{}</span></h3>
            <div class="alerts-container">
                {alerts_content}
            </div>
        </div>"#,
        alerts.len()
    )
}

/// Generate top operations card
fn generate_operations_card(operations: &[OperationSummary]) -> String {
    let operations_content = if operations.is_empty() {
        "<div class=\"no-operations\">No operations recorded</div>".to_string()
    } else {
        operations
            .iter()
            .take(10)
            .enumerate()
            .map(|(index, op)| {
                format!(
                    r#"<div class="operation" data-category="{}">
                        <div class="operation-rank">#{}</div>
                        <div class="operation-details">
                            <div class="operation-name">{}</div>
                            <div class="operation-category">{}</div>
                        </div>
                        <div class="operation-metrics">
                            <div class="operation-duration">{:.2} ms</div>
                            <div class="operation-percentage">{:.1}%</div>
                            <div class="operation-count">{} ops</div>
                        </div>
                        <div class="operation-bar">
                            <div class="operation-fill" style="width: {:.1}%"></div>
                        </div>
                    </div>"#,
                    op.category,
                    index + 1,
                    op.name,
                    op.category,
                    op.average_duration_ms,
                    op.percentage_of_total,
                    op.count,
                    op.percentage_of_total.min(100.0)
                )
            })
            .collect::<Vec<_>>()
            .join("")
    };

    format!(
        r#"<div class="card operations-card">
            <h3><span class="card-icon">📊</span>Top Operations</h3>
            <div class="operations-container">
                {operations_content}
            </div>
        </div>"#
    )
}

/// Generate charts and visualizations card
fn generate_charts_card(data: &DashboardData) -> String {
    format!(
        r#"<div class="card charts-card">
            <h3><span class="card-icon">📈</span>Performance Charts</h3>
            <div class="charts-container">
                <div class="chart-placeholder" id="performance-chart">
                    <div class="chart-title">Performance Over Time</div>
                    <div class="chart-content">Chart will be rendered here</div>
                </div>
                <div class="chart-placeholder" id="memory-chart">
                    <div class="chart-title">Memory Usage</div>
                    <div class="chart-content">Chart will be rendered here</div>
                </div>
                <div class="chart-placeholder" id="operations-chart">
                    <div class="chart-title">Operation Distribution</div>
                    <div class="chart-content">Chart will be rendered here</div>
                </div>
            </div>
        </div>"#
    )
}

// =============================================================================
// CSS Generation
// =============================================================================

/// Generate base CSS styles for the dashboard
fn generate_base_css(theme: &DashboardTheme) -> String {
    format!(
        r#"
        :root {{
            --primary-color: {};
            --secondary-color: {};
            --background-color: {};
            --text-color: {};
            --accent-color: {};
            --font-family: {};
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: var(--font-family);
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }}

        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 1rem;
        }}

        .header-controls {{
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }}

        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }}

        .card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1), 0 3px 6px rgba(0,0,0,0.05);
        }}

        .card h3 {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-color);
        }}

        .card-icon {{
            font-size: 1.2em;
        }}

        .metrics-grid {{
            display: grid;
            gap: 0.75rem;
        }}

        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: #f8f9fa;
            border-radius: 8px;
            transition: background-color 0.2s ease;
        }}

        .metric:hover {{
            background: #e9ecef;
        }}

        .metric-label {{
            font-weight: 500;
            color: #666;
        }}

        .metric-value {{
            font-weight: 700;
            font-size: 1.1em;
            color: var(--primary-color);
        }}

        .progress-bar {{
            width: 100%;
            height: 6px;
            background: #e9ecef;
            border-radius: 3px;
            margin-top: 0.5rem;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            border-radius: 3px;
            transition: width 0.3s ease;
        }}

        .alert {{
            margin: 0.5rem 0;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid;
        }}

        .alert-info {{
            background: #e3f2fd;
            border-color: #2196f3;
            color: #0d47a1;
        }}

        .alert-warning {{
            background: #fff3e0;
            border-color: #ff9800;
            color: #e65100;
        }}

        .alert-critical {{
            background: #ffebee;
            border-color: #f44336;
            color: #b71c1c;
        }}

        .alert-emergency {{
            background: #fce4ec;
            border-color: #e91e63;
            color: #880e4f;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}

        .alert-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }}

        .alert-title {{
            font-weight: 600;
            flex: 1;
        }}

        .alert-time {{
            font-size: 0.85em;
            opacity: 0.7;
        }}

        .operation {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.75rem;
            margin: 0.5rem 0;
            background: #f8f9fa;
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }}

        .operation-rank {{
            font-weight: 700;
            color: var(--primary-color);
            min-width: 2rem;
        }}

        .operation-details {{
            flex: 1;
        }}

        .operation-name {{
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}

        .operation-category {{
            font-size: 0.85em;
            color: #666;
        }}

        .operation-metrics {{
            display: flex;
            gap: 1rem;
            font-size: 0.9em;
        }}

        .operation-bar {{
            position: absolute;
            bottom: 0;
            left: 0;
            height: 3px;
            width: 100%;
            background: rgba(0,0,0,0.1);
        }}

        .operation-fill {{
            height: 100%;
            background: var(--primary-color);
            transition: width 0.3s ease;
        }}

        .charts-container {{
            display: grid;
            gap: 1rem;
        }}

        .chart-placeholder {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}

        .chart-title {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-color);
        }}

        .footer {{
            background: #f8f9fa;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 1px solid #dee2e6;
            font-size: 0.9em;
            color: #666;
        }}

        .status-indicator {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
        }}

        .status-dot.active {{
            animation: blink 2s infinite;
        }}

        @keyframes blink {{
            0%, 50% {{ opacity: 1; }}
            51%, 100% {{ opacity: 0.3; }}
        }}

        .refresh-button, .settings-button, .fullscreen-button {{
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s ease;
        }}

        .refresh-button:hover, .settings-button:hover, .fullscreen-button:hover {{
            background: rgba(255,255,255,0.3);
            transform: translateY(-1px);
        }}

        .no-alerts, .no-operations {{
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 2rem;
        }}

        .alert-count {{
            background: var(--accent-color);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 0.5rem;
        }}

        @media (max-width: 768px) {{
            .dashboard {{
                grid-template-columns: 1fr;
                padding: 1rem;
            }}

            .header {{
                padding: 1rem;
            }}

            .header h1 {{
                font-size: 2rem;
            }}

            .operation-metrics {{
                flex-direction: column;
                gap: 0.25rem;
            }}
        }}
        "#,
        theme.primary_color,
        theme.secondary_color,
        theme.background_color,
        theme.text_color,
        theme.accent_color,
        theme.font_family
    )
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Format duration in human-readable format
fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else if seconds < 86400 {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    } else {
        format!("{}d {}h", seconds / 86400, (seconds % 86400) / 3600)
    }
}

/// Format timestamp for display
fn format_timestamp(timestamp: u64) -> String {
    chrono::DateTime::from_timestamp(timestamp as i64, 0)
        .map(|dt| dt.format("%H:%M:%S").to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

// =============================================================================
// Export Functions
// =============================================================================

/// Export dashboard HTML to file
pub fn export_dashboard_html(
    profiler: &Profiler,
    memory_profiler: &MemoryProfiler,
    file_path: &str,
) -> TorshResult<()> {
    let dashboard = create_dashboard();
    let html = generate_dashboard_html(&dashboard)?;
    std::fs::write(file_path, html)
        .map_err(|e| TorshError::IoError(format!("Failed to write dashboard HTML: {e}")))?;
    Ok(())
}

/// Export dashboard HTML with custom configuration
pub fn export_dashboard_html_with_config(
    profiler: &Profiler,
    memory_profiler: &MemoryProfiler,
    config: DashboardConfig,
    file_path: &str,
) -> TorshResult<()> {
    let dashboard = create_dashboard_with_config(config);
    let html = generate_dashboard_html(&dashboard)?;
    std::fs::write(file_path, html)
        .map_err(|e| TorshError::IoError(format!("Failed to write dashboard HTML: {e}")))?;
    Ok(())
}

/// Generate dashboard HTML with custom theme
pub fn generate_dashboard_html_with_theme(
    dashboard: &Dashboard,
    theme: &DashboardTheme,
) -> TorshResult<String> {
    let current_data = dashboard
        .get_current_data()?
        .unwrap_or_else(create_default_data);
    let active_alerts = dashboard.get_active_alerts()?;

    let html = build_html_document(&current_data, &active_alerts, &dashboard.config, theme)?;
    Ok(html)
}

/// Generate static dashboard HTML (without real-time features)
pub fn generate_static_dashboard_html(
    data: &DashboardData,
    alerts: &[DashboardAlert],
) -> TorshResult<String> {
    let config = DashboardConfig {
        real_time_updates: false,
        websocket_config: super::types::WebSocketConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let theme = DashboardTheme::default();

    let html = build_html_document(data, alerts, &config, &theme)?;
    Ok(html)
}

// =============================================================================
// Template System (for future extension)
// =============================================================================

/// Dashboard template configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTemplate {
    pub name: String,
    pub layout: LayoutType,
    pub components: Vec<ComponentConfig>,
    pub theme: DashboardTheme,
}

/// Layout types for dashboard templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutType {
    Grid,
    Sidebar,
    Tabbed,
    Single,
}

/// Component configuration for templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConfig {
    pub component_type: String,
    pub position: (u32, u32),
    pub size: (u32, u32),
    pub config: HashMap<String, String>,
}

// =============================================================================
// DashboardRenderer - Missing Implementation
// =============================================================================

/// Dashboard HTML renderer with template system
pub struct DashboardRenderer {
    /// Renderer configuration
    pub config: DashboardRendererConfig,
}

/// Configuration for the dashboard renderer
#[derive(Debug, Clone)]
pub struct DashboardRendererConfig {
    /// Enable template caching
    pub enable_caching: bool,
    /// Enable minification
    pub enable_minification: bool,
    /// Include debug information
    pub include_debug_info: bool,
}

impl Default for DashboardRendererConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            enable_minification: false,
            include_debug_info: false,
        }
    }
}

impl DashboardRenderer {
    /// Create a new dashboard renderer
    pub fn new() -> Self {
        Self {
            config: DashboardRendererConfig::default(),
        }
    }

    /// Create a new dashboard renderer with custom configuration
    pub fn new_with_config(config: DashboardRendererConfig) -> Self {
        Self { config }
    }

    /// Generate complete dashboard HTML interface
    pub fn generate_dashboard_html(
        &self,
        data: &DashboardData,
        alerts: &[DashboardAlert],
        config: &DashboardConfig,
        theme: &DashboardTheme,
    ) -> TorshResult<String> {
        build_html_document(data, alerts, config, theme)
    }

    /// Generate HTML for specific component
    pub fn generate_component_html(
        &self,
        component_type: &str,
        data: &DashboardData,
    ) -> TorshResult<String> {
        match component_type {
            "performance" => Ok(generate_performance_card(&data.performance_metrics)),
            "memory" => Ok(generate_memory_card(&data.memory_metrics)),
            "system" => Ok(generate_system_card(&data.system_metrics)),
            "alerts" => Ok(generate_alerts_card(&data.alerts)),
            "operations" => Ok(generate_operations_card(&data.top_operations)),
            "charts" => Ok(generate_charts_card(data)),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown component type: {component_type}"
            ))),
        }
    }
}

impl Default for DashboardRenderer {
    fn default() -> Self {
        Self::new()
    }
}
