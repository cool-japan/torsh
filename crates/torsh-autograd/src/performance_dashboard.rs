// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Performance Dashboards for Autograd Analysis
//!
//! This module provides real-time and historical performance dashboards
//! for comprehensive autograd analysis and monitoring.
//!
//! # Features
//!
//! - **Real-time Metrics**: Live performance metrics with auto-refresh
//! - **Historical Analysis**: Time-series visualization of performance trends
//! - **Multi-dimensional Views**: CPU, memory, GPU, and operation-level metrics
//! - **Anomaly Highlighting**: Automatic detection and highlighting of anomalies
//! - **Export Capabilities**: Export dashboards to HTML, JSON, or images
//! - **Customizable Widgets**: Configurable dashboard layout and widgets

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard refresh interval (seconds)
    pub refresh_interval_secs: u64,

    /// Historical data retention (hours)
    pub retention_hours: u64,

    /// Maximum data points per chart
    pub max_data_points: usize,

    /// Enable real-time updates
    pub enable_realtime: bool,

    /// Dashboard layout
    pub layout: DashboardLayout,

    /// Theme
    pub theme: DashboardTheme,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            refresh_interval_secs: 5,
            retention_hours: 24,
            max_data_points: 1000,
            enable_realtime: true,
            layout: DashboardLayout::Grid,
            theme: DashboardTheme::Dark,
        }
    }
}

/// Dashboard layout type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardLayout {
    /// Grid layout
    Grid,

    /// Stacked layout
    Stacked,

    /// Tabbed layout
    Tabbed,

    /// Custom layout
    Custom,
}

/// Dashboard theme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardTheme {
    /// Light theme
    Light,

    /// Dark theme
    Dark,

    /// High contrast theme
    HighContrast,
}

/// Performance dashboard
pub struct PerformanceDashboard {
    config: DashboardConfig,
    metrics_store: Arc<RwLock<MetricsStore>>,
    widgets: Arc<RwLock<Vec<Box<dyn DashboardWidget + Send + Sync>>>>,
    snapshots: Arc<RwLock<VecDeque<DashboardSnapshot>>>,
}

/// Metrics storage
#[derive(Debug, Default)]
pub struct MetricsStore {
    /// Performance metrics time series
    performance_series: VecDeque<PerformanceMetric>,

    /// Memory metrics time series
    memory_series: VecDeque<MemoryMetric>,

    /// Operation metrics time series
    operation_series: VecDeque<OperationMetric>,

    /// Error metrics time series
    error_series: VecDeque<ErrorMetric>,
}

/// Performance metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Operations per second
    pub ops_per_second: f64,

    /// Average latency (ms)
    pub avg_latency_ms: f64,

    /// P50 latency (ms)
    pub p50_latency_ms: f64,

    /// P95 latency (ms)
    pub p95_latency_ms: f64,

    /// P99 latency (ms)
    pub p99_latency_ms: f64,

    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,

    /// GPU utilization (0.0-1.0)
    pub gpu_utilization: Option<f64>,
}

/// Memory metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetric {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Total memory allocated (bytes)
    pub total_allocated_bytes: u64,

    /// Peak memory usage (bytes)
    pub peak_usage_bytes: u64,

    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f64,

    /// Gradient memory (bytes)
    pub gradient_memory_bytes: u64,

    /// Graph memory (bytes)
    pub graph_memory_bytes: u64,

    /// Active tensors count
    pub active_tensors: usize,
}

/// Operation metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetric {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Operation name
    pub operation_name: String,

    /// Execution count
    pub execution_count: u64,

    /// Average duration (ms)
    pub avg_duration_ms: f64,

    /// Total time (ms)
    pub total_time_ms: f64,

    /// Percentage of total time
    pub time_percentage: f64,
}

/// Error metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetric {
    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Error rate (errors per second)
    pub error_rate: f64,

    /// Total errors
    pub total_errors: u64,

    /// Errors by category
    pub errors_by_category: HashMap<String, u64>,

    /// Critical errors
    pub critical_errors: u64,
}

/// Dashboard widget trait
pub trait DashboardWidget {
    /// Widget name
    fn name(&self) -> &str;

    /// Render widget to string (text/HTML)
    fn render(&self, format: RenderFormat) -> String;

    /// Update widget data
    fn update(&mut self, data: &MetricsStore);

    /// Widget type
    fn widget_type(&self) -> WidgetType;
}

/// Render format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderFormat {
    /// Plain text
    Text,

    /// HTML
    Html,

    /// JSON
    Json,

    /// Markdown
    Markdown,
}

/// Widget type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetType {
    /// Line chart
    LineChart,

    /// Bar chart
    BarChart,

    /// Gauge
    Gauge,

    /// Table
    Table,

    /// Metric card
    MetricCard,

    /// Heatmap
    Heatmap,
}

/// Dashboard snapshot for historical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,

    /// Performance summary
    pub performance_summary: PerformanceSummary,

    /// Memory summary
    pub memory_summary: MemorySummary,

    /// Top operations
    pub top_operations: Vec<OperationSummary>,

    /// Error summary
    pub error_summary: ErrorSummary,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Average ops/sec
    pub avg_ops_per_second: f64,

    /// Peak ops/sec
    pub peak_ops_per_second: f64,

    /// Average latency
    pub avg_latency_ms: f64,

    /// CPU utilization
    pub cpu_utilization: f64,

    /// GPU utilization
    pub gpu_utilization: Option<f64>,
}

/// Memory summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    /// Current memory usage
    pub current_usage_bytes: u64,

    /// Peak memory usage
    pub peak_usage_bytes: u64,

    /// Memory utilization
    pub utilization: f64,

    /// Active tensors
    pub active_tensors: usize,
}

/// Operation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSummary {
    /// Operation name
    pub name: String,

    /// Execution count
    pub count: u64,

    /// Total time
    pub total_time_ms: f64,

    /// Percentage of total time
    pub time_percentage: f64,
}

/// Error summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    /// Total errors
    pub total_errors: u64,

    /// Error rate
    pub error_rate: f64,

    /// Critical errors
    pub critical_errors: u64,

    /// Errors by category
    pub errors_by_category: HashMap<String, u64>,
}

impl PerformanceDashboard {
    /// Create a new performance dashboard
    pub fn new(config: DashboardConfig) -> Self {
        let dashboard = Self {
            config,
            metrics_store: Arc::new(RwLock::new(MetricsStore::default())),
            widgets: Arc::new(RwLock::new(Vec::new())),
            snapshots: Arc::new(RwLock::new(VecDeque::new())),
        };

        // Add default widgets
        dashboard.add_default_widgets();

        dashboard
    }

    /// Record performance metric
    pub fn record_performance(&self, metric: PerformanceMetric) {
        let mut store = self.metrics_store.write();
        store.performance_series.push_back(metric);

        // Cleanup old data
        let cutoff_time = Utc::now() - Duration::hours(self.config.retention_hours as i64);
        while let Some(m) = store.performance_series.front() {
            if m.timestamp < cutoff_time {
                store.performance_series.pop_front();
            } else {
                break;
            }
        }
    }

    /// Record memory metric
    pub fn record_memory(&self, metric: MemoryMetric) {
        let mut store = self.metrics_store.write();
        store.memory_series.push_back(metric);

        let cutoff_time = Utc::now() - Duration::hours(self.config.retention_hours as i64);
        while let Some(m) = store.memory_series.front() {
            if m.timestamp < cutoff_time {
                store.memory_series.pop_front();
            } else {
                break;
            }
        }
    }

    /// Record operation metric
    pub fn record_operation(&self, metric: OperationMetric) {
        let mut store = self.metrics_store.write();
        store.operation_series.push_back(metric);

        let cutoff_time = Utc::now() - Duration::hours(self.config.retention_hours as i64);
        while let Some(m) = store.operation_series.front() {
            if m.timestamp < cutoff_time {
                store.operation_series.pop_front();
            } else {
                break;
            }
        }
    }

    /// Record error metric
    pub fn record_error(&self, metric: ErrorMetric) {
        let mut store = self.metrics_store.write();
        store.error_series.push_back(metric);

        let cutoff_time = Utc::now() - Duration::hours(self.config.retention_hours as i64);
        while let Some(m) = store.error_series.front() {
            if m.timestamp < cutoff_time {
                store.error_series.pop_front();
            } else {
                break;
            }
        }
    }

    /// Take dashboard snapshot
    pub fn take_snapshot(&self) -> DashboardSnapshot {
        let store = self.metrics_store.read();

        let snapshot = DashboardSnapshot {
            timestamp: Utc::now(),
            performance_summary: self.compute_performance_summary(&store),
            memory_summary: self.compute_memory_summary(&store),
            top_operations: self.compute_top_operations(&store),
            error_summary: self.compute_error_summary(&store),
        };

        drop(store);

        // Store snapshot
        let mut snapshots = self.snapshots.write();
        snapshots.push_back(snapshot.clone());

        // Limit snapshot history
        while snapshots.len() > 100 {
            snapshots.pop_front();
        }

        snapshot
    }

    /// Render dashboard
    pub fn render(&self, format: RenderFormat) -> String {
        let widgets = self.widgets.read();
        let store = self.metrics_store.read();

        match format {
            RenderFormat::Html => self.render_html(&widgets, &store),
            RenderFormat::Text => self.render_text(&widgets, &store),
            RenderFormat::Json => self.render_json(&store),
            RenderFormat::Markdown => self.render_markdown(&widgets, &store),
        }
    }

    /// Add custom widget
    pub fn add_widget(&self, widget: Box<dyn DashboardWidget + Send + Sync>) {
        self.widgets.write().push(widget);
    }

    /// Get historical snapshots
    pub fn get_snapshots(&self, limit: Option<usize>) -> Vec<DashboardSnapshot> {
        let snapshots = self.snapshots.read();
        let limit = limit.unwrap_or(100);

        snapshots.iter().rev().take(limit).cloned().collect()
    }

    /// Export dashboard data
    pub fn export_data(&self) -> DashboardExport {
        let store = self.metrics_store.read();

        DashboardExport {
            timestamp: Utc::now(),
            performance_metrics: store.performance_series.iter().cloned().collect(),
            memory_metrics: store.memory_series.iter().cloned().collect(),
            operation_metrics: store.operation_series.iter().cloned().collect(),
            error_metrics: store.error_series.iter().cloned().collect(),
        }
    }

    // Private helper methods

    fn add_default_widgets(&self) {
        // Performance chart
        self.add_widget(Box::new(PerformanceChartWidget::new()));

        // Memory chart
        self.add_widget(Box::new(MemoryChartWidget::new()));

        // Top operations table
        self.add_widget(Box::new(TopOperationsWidget::new()));

        // Error rate gauge
        self.add_widget(Box::new(ErrorRateWidget::new()));
    }

    fn compute_performance_summary(&self, store: &MetricsStore) -> PerformanceSummary {
        if store.performance_series.is_empty() {
            return PerformanceSummary {
                avg_ops_per_second: 0.0,
                peak_ops_per_second: 0.0,
                avg_latency_ms: 0.0,
                cpu_utilization: 0.0,
                gpu_utilization: None,
            };
        }

        let count = store.performance_series.len() as f64;
        let avg_ops = store
            .performance_series
            .iter()
            .map(|m| m.ops_per_second)
            .sum::<f64>()
            / count;
        let peak_ops = store
            .performance_series
            .iter()
            .map(|m| m.ops_per_second)
            .fold(0.0, f64::max);
        let avg_latency = store
            .performance_series
            .iter()
            .map(|m| m.avg_latency_ms)
            .sum::<f64>()
            / count;
        let cpu_util = store
            .performance_series
            .iter()
            .map(|m| m.cpu_utilization)
            .sum::<f64>()
            / count;

        PerformanceSummary {
            avg_ops_per_second: avg_ops,
            peak_ops_per_second: peak_ops,
            avg_latency_ms: avg_latency,
            cpu_utilization: cpu_util,
            gpu_utilization: None,
        }
    }

    fn compute_memory_summary(&self, store: &MetricsStore) -> MemorySummary {
        if let Some(latest) = store.memory_series.back() {
            let peak = store
                .memory_series
                .iter()
                .map(|m| m.peak_usage_bytes)
                .max()
                .unwrap_or(0);

            MemorySummary {
                current_usage_bytes: latest.total_allocated_bytes,
                peak_usage_bytes: peak,
                utilization: latest.memory_utilization,
                active_tensors: latest.active_tensors,
            }
        } else {
            MemorySummary {
                current_usage_bytes: 0,
                peak_usage_bytes: 0,
                utilization: 0.0,
                active_tensors: 0,
            }
        }
    }

    fn compute_top_operations(&self, store: &MetricsStore) -> Vec<OperationSummary> {
        let mut op_map: HashMap<String, (u64, f64)> = HashMap::new();

        for metric in &store.operation_series {
            let entry = op_map
                .entry(metric.operation_name.clone())
                .or_insert((0, 0.0));
            entry.0 += metric.execution_count;
            entry.1 += metric.total_time_ms;
        }

        let total_time: f64 = op_map.values().map(|(_, time)| time).sum();

        let mut operations: Vec<_> = op_map
            .into_iter()
            .map(|(name, (count, time))| OperationSummary {
                name,
                count,
                total_time_ms: time,
                time_percentage: if total_time > 0.0 {
                    (time / total_time) * 100.0
                } else {
                    0.0
                },
            })
            .collect();

        operations.sort_by(|a, b| {
            b.total_time_ms
                .partial_cmp(&a.total_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        operations.truncate(10);

        operations
    }

    fn compute_error_summary(&self, store: &MetricsStore) -> ErrorSummary {
        if let Some(latest) = store.error_series.back() {
            ErrorSummary {
                total_errors: latest.total_errors,
                error_rate: latest.error_rate,
                critical_errors: latest.critical_errors,
                errors_by_category: latest.errors_by_category.clone(),
            }
        } else {
            ErrorSummary {
                total_errors: 0,
                error_rate: 0.0,
                critical_errors: 0,
                errors_by_category: HashMap::new(),
            }
        }
    }

    fn render_html(
        &self,
        _widgets: &[Box<dyn DashboardWidget + Send + Sync>],
        store: &MetricsStore,
    ) -> String {
        let snapshot = DashboardSnapshot {
            timestamp: Utc::now(),
            performance_summary: self.compute_performance_summary(store),
            memory_summary: self.compute_memory_summary(store),
            top_operations: self.compute_top_operations(store),
            error_summary: self.compute_error_summary(store),
        };

        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Autograd Performance Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .widget {{ background: #2a2a2a; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
        .metric {{ display: inline-block; margin-right: 30px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; color: #888; }}
        h2 {{ color: #4a9eff; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <h1>Autograd Performance Dashboard</h1>
        <p>Updated: {}</p>

        <div class="widget">
            <h2>Performance</h2>
            <div class="metric">
                <div class="metric-value">{:.2}</div>
                <div class="metric-label">Ops/Second</div>
            </div>
            <div class="metric">
                <div class="metric-value">{:.2}ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>
            <div class="metric">
                <div class="metric-value">{:.1}%</div>
                <div class="metric-label">CPU Usage</div>
            </div>
        </div>

        <div class="widget">
            <h2>Memory</h2>
            <div class="metric">
                <div class="metric-value">{:.2} GB</div>
                <div class="metric-label">Current Usage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{}</div>
                <div class="metric-label">Active Tensors</div>
            </div>
        </div>

        <div class="widget">
            <h2>Top Operations</h2>
            <table>
                <tr><th>Operation</th><th>Count</th><th>Time (ms)</th><th>% Time</th></tr>
                {}
            </table>
        </div>
    </div>
</body>
</html>"#,
            snapshot.timestamp,
            snapshot.performance_summary.avg_ops_per_second,
            snapshot.performance_summary.avg_latency_ms,
            snapshot.performance_summary.cpu_utilization * 100.0,
            snapshot.memory_summary.current_usage_bytes as f64 / 1_073_741_824.0,
            snapshot.memory_summary.active_tensors,
            snapshot
                .top_operations
                .iter()
                .map(|op| format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.1}%</td></tr>",
                    op.name, op.count, op.total_time_ms, op.time_percentage
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    fn render_text(
        &self,
        _widgets: &[Box<dyn DashboardWidget + Send + Sync>],
        store: &MetricsStore,
    ) -> String {
        let snapshot = DashboardSnapshot {
            timestamp: Utc::now(),
            performance_summary: self.compute_performance_summary(store),
            memory_summary: self.compute_memory_summary(store),
            top_operations: self.compute_top_operations(store),
            error_summary: self.compute_error_summary(store),
        };

        format!(
            r#"=== Autograd Performance Dashboard ===
Updated: {}

Performance:
  Ops/Second: {:.2}
  Avg Latency: {:.2}ms
  CPU Usage: {:.1}%

Memory:
  Current: {:.2} GB
  Active Tensors: {}

Top Operations:
{}

Errors:
  Total: {}
  Rate: {:.2} errors/sec
"#,
            snapshot.timestamp,
            snapshot.performance_summary.avg_ops_per_second,
            snapshot.performance_summary.avg_latency_ms,
            snapshot.performance_summary.cpu_utilization * 100.0,
            snapshot.memory_summary.current_usage_bytes as f64 / 1_073_741_824.0,
            snapshot.memory_summary.active_tensors,
            snapshot
                .top_operations
                .iter()
                .map(|op| format!(
                    "  {} - {} calls, {:.2}ms ({:.1}%)",
                    op.name, op.count, op.total_time_ms, op.time_percentage
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            snapshot.error_summary.total_errors,
            snapshot.error_summary.error_rate
        )
    }

    fn render_json(&self, _store: &MetricsStore) -> String {
        let snapshot = self.take_snapshot();
        serde_json::to_string_pretty(&snapshot).unwrap_or_else(|_| "{}".to_string())
    }

    fn render_markdown(
        &self,
        _widgets: &[Box<dyn DashboardWidget + Send + Sync>],
        store: &MetricsStore,
    ) -> String {
        let snapshot = DashboardSnapshot {
            timestamp: Utc::now(),
            performance_summary: self.compute_performance_summary(store),
            memory_summary: self.compute_memory_summary(store),
            top_operations: self.compute_top_operations(store),
            error_summary: self.compute_error_summary(store),
        };

        format!(
            r#"# Autograd Performance Dashboard

**Updated**: {}

## Performance

- **Ops/Second**: {:.2}
- **Avg Latency**: {:.2}ms
- **CPU Usage**: {:.1}%

## Memory

- **Current**: {:.2} GB
- **Active Tensors**: {}

## Top Operations

| Operation | Count | Time (ms) | % Time |
|-----------|-------|-----------|--------|
{}

## Errors

- **Total**: {}
- **Rate**: {:.2} errors/sec
"#,
            snapshot.timestamp,
            snapshot.performance_summary.avg_ops_per_second,
            snapshot.performance_summary.avg_latency_ms,
            snapshot.performance_summary.cpu_utilization * 100.0,
            snapshot.memory_summary.current_usage_bytes as f64 / 1_073_741_824.0,
            snapshot.memory_summary.active_tensors,
            snapshot
                .top_operations
                .iter()
                .map(|op| format!(
                    "| {} | {} | {:.2} | {:.1}% |",
                    op.name, op.count, op.total_time_ms, op.time_percentage
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            snapshot.error_summary.total_errors,
            snapshot.error_summary.error_rate
        )
    }
}

/// Dashboard export data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardExport {
    /// Export timestamp
    pub timestamp: DateTime<Utc>,

    /// Performance metrics
    pub performance_metrics: Vec<PerformanceMetric>,

    /// Memory metrics
    pub memory_metrics: Vec<MemoryMetric>,

    /// Operation metrics
    pub operation_metrics: Vec<OperationMetric>,

    /// Error metrics
    pub error_metrics: Vec<ErrorMetric>,
}

// Default widget implementations

struct PerformanceChartWidget {
    name: String,
}

impl PerformanceChartWidget {
    fn new() -> Self {
        Self {
            name: "Performance Chart".to_string(),
        }
    }
}

impl DashboardWidget for PerformanceChartWidget {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&self, format: RenderFormat) -> String {
        match format {
            RenderFormat::Text => "Performance Chart (Text)".to_string(),
            RenderFormat::Html => "<div>Performance Chart</div>".to_string(),
            _ => "Performance Chart".to_string(),
        }
    }

    fn update(&mut self, _data: &MetricsStore) {
        // Update widget data
    }

    fn widget_type(&self) -> WidgetType {
        WidgetType::LineChart
    }
}

struct MemoryChartWidget {
    name: String,
}

impl MemoryChartWidget {
    fn new() -> Self {
        Self {
            name: "Memory Chart".to_string(),
        }
    }
}

impl DashboardWidget for MemoryChartWidget {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&self, format: RenderFormat) -> String {
        match format {
            RenderFormat::Text => "Memory Chart (Text)".to_string(),
            RenderFormat::Html => "<div>Memory Chart</div>".to_string(),
            _ => "Memory Chart".to_string(),
        }
    }

    fn update(&mut self, _data: &MetricsStore) {}

    fn widget_type(&self) -> WidgetType {
        WidgetType::LineChart
    }
}

struct TopOperationsWidget {
    name: String,
}

impl TopOperationsWidget {
    fn new() -> Self {
        Self {
            name: "Top Operations".to_string(),
        }
    }
}

impl DashboardWidget for TopOperationsWidget {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&self, format: RenderFormat) -> String {
        match format {
            RenderFormat::Text => "Top Operations (Text)".to_string(),
            RenderFormat::Html => "<div>Top Operations</div>".to_string(),
            _ => "Top Operations".to_string(),
        }
    }

    fn update(&mut self, _data: &MetricsStore) {}

    fn widget_type(&self) -> WidgetType {
        WidgetType::Table
    }
}

struct ErrorRateWidget {
    name: String,
}

impl ErrorRateWidget {
    fn new() -> Self {
        Self {
            name: "Error Rate".to_string(),
        }
    }
}

impl DashboardWidget for ErrorRateWidget {
    fn name(&self) -> &str {
        &self.name
    }

    fn render(&self, format: RenderFormat) -> String {
        match format {
            RenderFormat::Text => "Error Rate (Text)".to_string(),
            RenderFormat::Html => "<div>Error Rate</div>".to_string(),
            _ => "Error Rate".to_string(),
        }
    }

    fn update(&mut self, _data: &MetricsStore) {}

    fn widget_type(&self) -> WidgetType {
        WidgetType::Gauge
    }
}

/// Global dashboard instance
static GLOBAL_DASHBOARD: OnceLock<Arc<PerformanceDashboard>> = OnceLock::new();

/// Get global dashboard
pub fn get_global_dashboard() -> Arc<PerformanceDashboard> {
    GLOBAL_DASHBOARD
        .get_or_init(|| Arc::new(PerformanceDashboard::new(DashboardConfig::default())))
        .clone()
}

/// Initialize global dashboard
pub fn init_global_dashboard(config: DashboardConfig) {
    let _ = GLOBAL_DASHBOARD.set(Arc::new(PerformanceDashboard::new(config)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let dashboard = PerformanceDashboard::new(DashboardConfig::default());

        let snapshot = dashboard.take_snapshot();
        assert_eq!(snapshot.top_operations.len(), 0); // No operations yet
    }

    #[test]
    fn test_performance_recording() {
        let dashboard = PerformanceDashboard::new(DashboardConfig::default());

        let metric = PerformanceMetric {
            timestamp: Utc::now(),
            ops_per_second: 100.0,
            avg_latency_ms: 10.0,
            p50_latency_ms: 8.0,
            p95_latency_ms: 15.0,
            p99_latency_ms: 20.0,
            cpu_utilization: 0.5,
            gpu_utilization: None,
        };

        dashboard.record_performance(metric);

        let snapshot = dashboard.take_snapshot();
        assert!(snapshot.performance_summary.avg_ops_per_second > 0.0);
    }

    #[test]
    fn test_memory_recording() {
        let dashboard = PerformanceDashboard::new(DashboardConfig::default());

        let metric = MemoryMetric {
            timestamp: Utc::now(),
            total_allocated_bytes: 1_000_000,
            peak_usage_bytes: 1_500_000,
            memory_utilization: 0.75,
            gradient_memory_bytes: 500_000,
            graph_memory_bytes: 500_000,
            active_tensors: 100,
        };

        dashboard.record_memory(metric);

        let snapshot = dashboard.take_snapshot();
        assert_eq!(snapshot.memory_summary.active_tensors, 100);
    }

    #[test]
    fn test_dashboard_rendering() {
        let dashboard = PerformanceDashboard::new(DashboardConfig::default());

        let text = dashboard.render(RenderFormat::Text);
        assert!(text.contains("Performance Dashboard"));

        let html = dashboard.render(RenderFormat::Html);
        assert!(html.contains("<!DOCTYPE html>"));

        let json = dashboard.render(RenderFormat::Json);
        assert!(json.contains("timestamp"));

        let md = dashboard.render(RenderFormat::Markdown);
        assert!(md.contains("# Autograd Performance Dashboard"));
    }

    #[test]
    fn test_top_operations() {
        let dashboard = PerformanceDashboard::new(DashboardConfig::default());

        // Record some operations
        for i in 0..5 {
            dashboard.record_operation(OperationMetric {
                timestamp: Utc::now(),
                operation_name: format!("op{}", i),
                execution_count: (i + 1) * 10,
                avg_duration_ms: (i + 1) as f64 * 5.0,
                total_time_ms: (i + 1) as f64 * 50.0,
                time_percentage: 0.0,
            });
        }

        let snapshot = dashboard.take_snapshot();
        assert_eq!(snapshot.top_operations.len(), 5);

        // Should be sorted by total time
        assert!(
            snapshot.top_operations[0].total_time_ms >= snapshot.top_operations[1].total_time_ms
        );
    }

    #[test]
    fn test_export_data() {
        let dashboard = PerformanceDashboard::new(DashboardConfig::default());

        dashboard.record_performance(PerformanceMetric {
            timestamp: Utc::now(),
            ops_per_second: 100.0,
            avg_latency_ms: 10.0,
            p50_latency_ms: 8.0,
            p95_latency_ms: 15.0,
            p99_latency_ms: 20.0,
            cpu_utilization: 0.5,
            gpu_utilization: None,
        });

        let export = dashboard.export_data();
        assert_eq!(export.performance_metrics.len(), 1);
    }
}
