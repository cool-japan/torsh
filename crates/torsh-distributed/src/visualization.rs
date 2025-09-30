//! Visualization tools for distributed training monitoring
//!
//! This module provides comprehensive visualization capabilities for distributed training
//! including real-time dashboards, performance charts, communication graphs, and bottleneck analysis.

use crate::metrics::get_global_metrics_collector;
use crate::profiling::get_global_profiler;
use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::UNIX_EPOCH;

/// Chart types supported by the visualization system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart for time series data
    Line,
    /// Bar chart for categorical data
    Bar,
    /// Pie chart for proportional data
    Pie,
    /// Scatter plot for correlation analysis
    Scatter,
    /// Heat map for matrix data
    Heatmap,
    /// Network graph for communication patterns
    Network,
}

/// Color schemes for visualizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Default blue theme
    Default,
    /// Dark theme
    Dark,
    /// High contrast theme
    HighContrast,
    /// Performance-oriented colors (green/yellow/red)
    Performance,
    /// Categorical colors
    Categorical,
}

impl ColorScheme {
    /// Get primary colors for the scheme
    pub fn colors(&self) -> Vec<&'static str> {
        match self {
            ColorScheme::Default => vec!["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"],
            ColorScheme::Dark => vec!["#34495e", "#2c3e50", "#e67e22", "#e74c3c", "#95a5a6"],
            ColorScheme::HighContrast => {
                vec!["#000000", "#ffffff", "#ff0000", "#00ff00", "#0000ff"]
            }
            ColorScheme::Performance => vec!["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"],
            ColorScheme::Categorical => vec!["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        }
    }

    /// Get background color for the scheme
    pub fn background_color(&self) -> &'static str {
        match self {
            ColorScheme::Default | ColorScheme::Performance | ColorScheme::Categorical => "#ffffff",
            ColorScheme::Dark => "#2c3e50",
            ColorScheme::HighContrast => "#ffffff",
        }
    }

    /// Get text color for the scheme
    pub fn text_color(&self) -> &'static str {
        match self {
            ColorScheme::Default
            | ColorScheme::Performance
            | ColorScheme::Categorical
            | ColorScheme::HighContrast => "#333333",
            ColorScheme::Dark => "#ecf0f1",
        }
    }
}

/// Configuration for visualization generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Width of generated charts
    pub chart_width: u32,
    /// Height of generated charts
    pub chart_height: u32,
    /// Color scheme to use
    pub color_scheme: ColorScheme,
    /// Whether to include interactive features
    pub interactive: bool,
    /// Maximum number of data points to show
    pub max_data_points: usize,
    /// Update interval for real-time charts (seconds)
    pub update_interval_secs: u64,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            chart_width: 800,
            chart_height: 400,
            color_scheme: ColorScheme::Default,
            interactive: true,
            max_data_points: 100,
            update_interval_secs: 5,
        }
    }
}

/// Data point for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// X-axis value (usually timestamp)
    pub x: f64,
    /// Y-axis value
    pub y: f64,
    /// Optional label
    pub label: Option<String>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl DataPoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y,
            label: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Chart data series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartSeries {
    /// Series name
    pub name: String,
    /// Data points
    pub data: Vec<DataPoint>,
    /// Series color
    pub color: String,
    /// Series type (overrides chart type if specified)
    pub chart_type: Option<ChartType>,
}

impl ChartSeries {
    pub fn new(name: String, color: String) -> Self {
        Self {
            name,
            data: Vec::new(),
            color,
            chart_type: None,
        }
    }

    pub fn add_point(&mut self, point: DataPoint) {
        self.data.push(point);
    }

    pub fn with_type(mut self, chart_type: ChartType) -> Self {
        self.chart_type = Some(chart_type);
        self
    }
}

/// Complete chart specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chart {
    /// Chart title
    pub title: String,
    /// Chart type
    pub chart_type: ChartType,
    /// X-axis label
    pub x_label: String,
    /// Y-axis label
    pub y_label: String,
    /// Data series
    pub series: Vec<ChartSeries>,
    /// Configuration
    pub config: VisualizationConfig,
}

impl Chart {
    pub fn new(title: String, chart_type: ChartType) -> Self {
        Self {
            title,
            chart_type,
            x_label: "X".to_string(),
            y_label: "Y".to_string(),
            series: Vec::new(),
            config: VisualizationConfig::default(),
        }
    }

    pub fn with_labels(mut self, x_label: String, y_label: String) -> Self {
        self.x_label = x_label;
        self.y_label = y_label;
        self
    }

    pub fn add_series(&mut self, series: ChartSeries) {
        self.series.push(series);
    }

    pub fn with_config(mut self, config: VisualizationConfig) -> Self {
        self.config = config;
        self
    }
}

/// Dashboard layout specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    /// Dashboard title
    pub title: String,
    /// Charts in the dashboard
    pub charts: Vec<Chart>,
    /// Layout configuration
    pub layout: DashboardLayout,
    /// Global configuration
    pub config: VisualizationConfig,
}

/// Dashboard layout options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    /// Number of columns
    pub columns: u32,
    /// Chart spacing in pixels
    pub spacing: u32,
    /// Whether to use responsive layout
    pub responsive: bool,
}

impl Default for DashboardLayout {
    fn default() -> Self {
        Self {
            columns: 2,
            spacing: 20,
            responsive: true,
        }
    }
}

impl Dashboard {
    pub fn new(title: String) -> Self {
        Self {
            title,
            charts: Vec::new(),
            layout: DashboardLayout::default(),
            config: VisualizationConfig::default(),
        }
    }

    pub fn add_chart(&mut self, chart: Chart) {
        self.charts.push(chart);
    }

    pub fn with_layout(mut self, layout: DashboardLayout) -> Self {
        self.layout = layout;
        self
    }

    pub fn with_config(mut self, config: VisualizationConfig) -> Self {
        self.config = config;
        self
    }
}

/// Visualization generator for distributed training data
pub struct VisualizationGenerator {
    /// Configuration
    config: VisualizationConfig,
}

impl VisualizationGenerator {
    /// Create a new visualization generator
    pub fn new() -> Self {
        Self::with_config(VisualizationConfig::default())
    }

    /// Create a new visualization generator with custom configuration
    pub fn with_config(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate performance metrics dashboard
    pub fn generate_performance_dashboard(&self) -> TorshResult<Dashboard> {
        let mut dashboard = Dashboard::new("Distributed Training Performance".to_string())
            .with_config(self.config.clone());

        // System metrics chart
        if let Ok(system_chart) = self.create_system_metrics_chart() {
            dashboard.add_chart(system_chart);
        }

        // Communication metrics chart
        if let Ok(comm_chart) = self.create_communication_metrics_chart() {
            dashboard.add_chart(comm_chart);
        }

        // Training progress chart
        if let Ok(training_chart) = self.create_training_progress_chart() {
            dashboard.add_chart(training_chart);
        }

        // Bottleneck analysis chart
        if let Ok(bottleneck_chart) = self.create_bottleneck_chart() {
            dashboard.add_chart(bottleneck_chart);
        }

        Ok(dashboard)
    }

    /// Create system metrics chart (CPU, memory, GPU usage)
    fn create_system_metrics_chart(&self) -> TorshResult<Chart> {
        let metrics_collector = get_global_metrics_collector();
        let system_history = metrics_collector.get_system_history()?;

        let mut chart = Chart::new("System Resource Usage".to_string(), ChartType::Line)
            .with_labels("Time".to_string(), "Usage (%)".to_string())
            .with_config(self.config.clone());

        let colors = self.config.color_scheme.colors();

        // CPU usage series
        let mut cpu_series = ChartSeries::new("CPU Usage".to_string(), colors[0].to_string());
        for point in &system_history {
            let timestamp = point
                .timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            cpu_series.add_point(DataPoint::new(timestamp, point.value.cpu_usage_pct));
        }
        chart.add_series(cpu_series);

        // Memory usage series
        let mut memory_series = ChartSeries::new("Memory Usage".to_string(), colors[1].to_string());
        for point in &system_history {
            let timestamp = point
                .timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            memory_series.add_point(DataPoint::new(timestamp, point.value.memory_usage_pct));
        }
        chart.add_series(memory_series);

        // GPU usage series (if available)
        if system_history
            .iter()
            .any(|p| p.value.gpu_usage_pct.is_some())
        {
            let mut gpu_series = ChartSeries::new("GPU Usage".to_string(), colors[2].to_string());
            for point in &system_history {
                if let Some(gpu_usage) = point.value.gpu_usage_pct {
                    let timestamp = point
                        .timestamp
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    gpu_series.add_point(DataPoint::new(timestamp, gpu_usage));
                }
            }
            chart.add_series(gpu_series);
        }

        Ok(chart)
    }

    /// Create communication metrics chart
    fn create_communication_metrics_chart(&self) -> TorshResult<Chart> {
        let metrics_collector = get_global_metrics_collector();
        let comm_history = metrics_collector.get_communication_history()?;

        let mut chart = Chart::new("Communication Performance".to_string(), ChartType::Line)
            .with_labels("Time".to_string(), "Value".to_string())
            .with_config(self.config.clone());

        let colors = self.config.color_scheme.colors();

        // Latency series
        let mut latency_series =
            ChartSeries::new("Avg Latency (ms)".to_string(), colors[0].to_string());
        for point in &comm_history {
            let timestamp = point
                .timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            latency_series.add_point(DataPoint::new(timestamp, point.value.avg_latency_ms));
        }
        chart.add_series(latency_series);

        // Bandwidth series
        let mut bandwidth_series =
            ChartSeries::new("Avg Bandwidth (MB/s)".to_string(), colors[1].to_string());
        for point in &comm_history {
            let timestamp = point
                .timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            bandwidth_series.add_point(DataPoint::new(timestamp, point.value.avg_bandwidth_mbps));
        }
        chart.add_series(bandwidth_series);

        // Operations per second series
        let mut ops_series = ChartSeries::new("Operations/sec".to_string(), colors[2].to_string());
        for point in &comm_history {
            let timestamp = point
                .timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            ops_series.add_point(DataPoint::new(timestamp, point.value.ops_per_second));
        }
        chart.add_series(ops_series);

        Ok(chart)
    }

    /// Create training progress chart
    fn create_training_progress_chart(&self) -> TorshResult<Chart> {
        let metrics_collector = get_global_metrics_collector();
        let training_history = metrics_collector.get_training_history()?;

        let mut chart = Chart::new("Training Progress".to_string(), ChartType::Line)
            .with_labels("Step".to_string(), "Loss".to_string())
            .with_config(self.config.clone());

        let colors = self.config.color_scheme.colors();

        // Training loss series
        let mut train_loss_series =
            ChartSeries::new("Training Loss".to_string(), colors[0].to_string());
        for point in &training_history {
            if let Some(loss) = point.value.training_loss {
                train_loss_series.add_point(DataPoint::new(point.value.current_step as f64, loss));
            }
        }
        if !train_loss_series.data.is_empty() {
            chart.add_series(train_loss_series);
        }

        // Validation loss series
        let mut val_loss_series =
            ChartSeries::new("Validation Loss".to_string(), colors[1].to_string());
        for point in &training_history {
            if let Some(loss) = point.value.validation_loss {
                val_loss_series.add_point(DataPoint::new(point.value.current_step as f64, loss));
            }
        }
        if !val_loss_series.data.is_empty() {
            chart.add_series(val_loss_series);
        }

        // If no loss data, show samples per second
        if chart.series.is_empty() {
            let mut throughput_series =
                ChartSeries::new("Samples/sec".to_string(), colors[2].to_string());
            for point in &training_history {
                if point.value.samples_per_second > 0.0 {
                    let timestamp = point
                        .timestamp
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    throughput_series
                        .add_point(DataPoint::new(timestamp, point.value.samples_per_second));
                }
            }
            if !throughput_series.data.is_empty() {
                chart.y_label = "Samples/sec".to_string();
                chart.x_label = "Time".to_string();
                chart.add_series(throughput_series);
            }
        }

        Ok(chart)
    }

    /// Create bottleneck analysis chart
    fn create_bottleneck_chart(&self) -> TorshResult<Chart> {
        crate::bottleneck_detection::with_global_bottleneck_detector(|detector| {
            let history = detector.get_bottleneck_history();

            let mut chart = Chart::new("Bottleneck Analysis".to_string(), ChartType::Bar)
                .with_labels("Bottleneck Type".to_string(), "Count".to_string())
                .with_config(self.config.clone());

            // Count bottlenecks by type
            let mut bottleneck_counts: HashMap<String, u32> = HashMap::new();
            for bottleneck in history {
                *bottleneck_counts
                    .entry(bottleneck.bottleneck_type.to_string())
                    .or_insert(0) += 1;
            }

            if !bottleneck_counts.is_empty() {
                let colors = self.config.color_scheme.colors();
                let mut series =
                    ChartSeries::new("Bottleneck Count".to_string(), colors[0].to_string());

                for (i, (bottleneck_type, count)) in bottleneck_counts.iter().enumerate() {
                    series.add_point(
                        DataPoint::new(i as f64, *count as f64).with_label(bottleneck_type.clone()),
                    );
                }

                chart.add_series(series);
            }

            Ok(chart)
        })
    }

    /// Create communication pattern network graph
    pub fn create_communication_network_graph(&self) -> TorshResult<Chart> {
        let profiler = get_global_profiler();
        let events = profiler.get_all_events()?;

        let mut chart = Chart::new("Communication Network".to_string(), ChartType::Network)
            .with_labels("Rank".to_string(), "Communication Volume".to_string())
            .with_config(self.config.clone());

        // Analyze communication patterns between ranks
        let mut rank_comm: HashMap<(u32, u32), f64> = HashMap::new();
        let mut all_ranks: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for event in &events {
            all_ranks.insert(event.rank);
            // For simplicity, create self-communication entries
            let key = (event.rank, event.rank);
            *rank_comm.entry(key).or_insert(0.0) += event.data_size_bytes as f64;
        }

        // Create data points for the network graph
        let colors = self.config.color_scheme.colors();
        let mut series =
            ChartSeries::new("Communication Volume".to_string(), colors[0].to_string());

        for ((src, dst), volume) in rank_comm.iter() {
            series.add_point(
                DataPoint::new(*src as f64, *dst as f64)
                    .with_metadata("volume".to_string(), volume.to_string())
                    .with_metadata("src_rank".to_string(), src.to_string())
                    .with_metadata("dst_rank".to_string(), dst.to_string()),
            );
        }

        chart.add_series(series);
        Ok(chart)
    }

    /// Generate SVG chart
    pub fn generate_svg_chart(&self, chart: &Chart) -> TorshResult<String> {
        let mut svg = String::new();

        // SVG header
        svg.push_str(&format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="{}"/>
            "#,
            chart.config.chart_width,
            chart.config.chart_height,
            chart.config.color_scheme.background_color()
        ));

        // Title
        svg.push_str(&format!(
            r#"<text x="{}" y="30" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="{}">{}</text>"#,
            chart.config.chart_width / 2,
            chart.config.color_scheme.text_color(),
            chart.title
        ));

        match chart.chart_type {
            ChartType::Line => self.generate_line_chart_svg(&mut svg, chart)?,
            ChartType::Bar => self.generate_bar_chart_svg(&mut svg, chart)?,
            ChartType::Pie => self.generate_pie_chart_svg(&mut svg, chart)?,
            _ => {} // Other chart types not implemented in SVG
        }

        svg.push_str("</svg>");
        Ok(svg)
    }

    /// Generate line chart SVG
    fn generate_line_chart_svg(&self, svg: &mut String, chart: &Chart) -> TorshResult<()> {
        let margin = 60;
        let chart_width = chart.config.chart_width - 2 * margin;
        let chart_height = chart.config.chart_height - 2 * margin - 40; // 40 for title

        for (series_idx, series) in chart.series.iter().enumerate() {
            if series.data.is_empty() {
                continue;
            }

            // Find data ranges
            let x_min = series
                .data
                .iter()
                .map(|p| p.x)
                .fold(f64::INFINITY, f64::min);
            let x_max = series
                .data
                .iter()
                .map(|p| p.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let y_min = series
                .data
                .iter()
                .map(|p| p.y)
                .fold(f64::INFINITY, f64::min);
            let y_max = series
                .data
                .iter()
                .map(|p| p.y)
                .fold(f64::NEG_INFINITY, f64::max);

            let x_range = if x_max > x_min { x_max - x_min } else { 1.0 };
            let y_range = if y_max > y_min { y_max - y_min } else { 1.0 };

            // Generate path data
            let mut path_data = String::new();
            for (i, point) in series.data.iter().enumerate() {
                let x = margin as f64 + ((point.x - x_min) / x_range) * chart_width as f64;
                let y = (margin + 40) as f64 + chart_height as f64
                    - ((point.y - y_min) / y_range) * chart_height as f64;

                if i == 0 {
                    path_data.push_str(&format!("M{},{}", x, y));
                } else {
                    path_data.push_str(&format!(" L{},{}", x, y));
                }
            }

            // Add path
            svg.push_str(&format!(
                r#"<path d="{}" stroke="{}" stroke-width="2" fill="none"/>"#,
                path_data, series.color
            ));

            // Add legend
            let legend_y = 40 + (series_idx as u32) * 20 + 10;
            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="15" height="15" fill="{}"/>
                <text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">{}</text>"#,
                chart.config.chart_width - 150,
                legend_y,
                series.color,
                chart.config.chart_width - 130,
                legend_y + 12,
                chart.config.color_scheme.text_color(),
                series.name
            ));
        }

        Ok(())
    }

    /// Generate bar chart SVG
    fn generate_bar_chart_svg(&self, svg: &mut String, chart: &Chart) -> TorshResult<()> {
        let margin = 60;
        let chart_width = chart.config.chart_width - 2 * margin;
        let chart_height = chart.config.chart_height - 2 * margin - 40;

        for series in &chart.series {
            if series.data.is_empty() {
                continue;
            }

            let max_y = series
                .data
                .iter()
                .map(|p| p.y)
                .fold(f64::NEG_INFINITY, f64::max);
            let bar_width = chart_width as f64 / series.data.len() as f64 * 0.8;

            for (i, point) in series.data.iter().enumerate() {
                let x = margin as f64
                    + (i as f64 + 0.1) * (chart_width as f64 / series.data.len() as f64);
                let bar_height = if max_y > 0.0 {
                    (point.y / max_y) * chart_height as f64
                } else {
                    0.0
                };
                let y = (margin + 40) as f64 + chart_height as f64 - bar_height;

                svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
                    x, y, bar_width, bar_height, series.color
                ));

                // Add label if available
                if let Some(label) = &point.label {
                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{}">{}</text>"#,
                        x + bar_width / 2.0,
                        (margin + 40 + chart_height) as f64 + 15.0,
                        chart.config.color_scheme.text_color(),
                        label
                    ));
                }
            }
        }

        Ok(())
    }

    /// Generate pie chart SVG  
    fn generate_pie_chart_svg(&self, svg: &mut String, chart: &Chart) -> TorshResult<()> {
        for series in &chart.series {
            if series.data.is_empty() {
                continue;
            }

            let center_x = chart.config.chart_width as f64 / 2.0;
            let center_y = (chart.config.chart_height as f64) / 2.0;
            let radius =
                ((chart.config.chart_width.min(chart.config.chart_height)) as f64 / 2.0) - 50.0;

            let total: f64 = series.data.iter().map(|p| p.y).sum();
            let mut current_angle: f64 = 0.0;

            let colors = chart.config.color_scheme.colors();

            for (i, point) in series.data.iter().enumerate() {
                let slice_angle = (point.y / total) * 2.0 * std::f64::consts::PI;

                let start_x = center_x + radius * current_angle.cos();
                let start_y = center_y + radius * current_angle.sin();

                current_angle += slice_angle;

                let end_x = center_x + radius * current_angle.cos();
                let end_y = center_y + radius * current_angle.sin();

                let large_arc_flag = if slice_angle > std::f64::consts::PI {
                    1
                } else {
                    0
                };

                let color = colors[i % colors.len()];

                svg.push_str(&format!(
                    r#"<path d="M{},{} L{},{} A{},{} 0 {},{} {},{} Z" fill="{}"/>"#,
                    center_x,
                    center_y,
                    start_x,
                    start_y,
                    radius,
                    radius,
                    large_arc_flag,
                    1,
                    end_x,
                    end_y,
                    color
                ));

                // Add label
                if let Some(label) = &point.label {
                    let label_angle = current_angle - slice_angle / 2.0;
                    let label_x = center_x + (radius + 20.0) * label_angle.cos();
                    let label_y = center_y + (radius + 20.0) * label_angle.sin();

                    svg.push_str(&format!(
                        r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{}">{}</text>"#,
                        label_x, label_y,
                        chart.config.color_scheme.text_color(),
                        label
                    ));
                }
            }
        }

        Ok(())
    }

    /// Generate HTML dashboard
    pub fn generate_html_dashboard(&self, dashboard: &Dashboard) -> TorshResult<String> {
        let mut html = String::new();

        // HTML header
        html.push_str(&format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: {};
            color: {};
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: repeat({}, 1fr);
            gap: {}px;
        }}
        .chart-container {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background-color: {};
        }}
        h1 {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .chart {{
            width: 100%;
            height: auto;
        }}
        {}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div class="dashboard">
"#,
            dashboard.title,
            dashboard.config.color_scheme.background_color(),
            dashboard.config.color_scheme.text_color(),
            dashboard.layout.columns,
            dashboard.layout.spacing,
            dashboard.config.color_scheme.background_color(),
            if dashboard.layout.responsive {
                "@media (max-width: 768px) { .dashboard { grid-template-columns: 1fr; } }"
            } else {
                ""
            },
            dashboard.title
        ));

        // Add charts
        for chart in &dashboard.charts {
            html.push_str(r#"        <div class="chart-container">"#);
            let svg = self.generate_svg_chart(chart)?;
            html.push_str(&format!(r#"            <div class="chart">{}</div>"#, svg));
            html.push_str(r#"        </div>"#);
        }

        // HTML footer
        html.push_str(
            r#"    </div>
</body>
</html>"#,
        );

        Ok(html)
    }

    /// Export dashboard data as JSON
    pub fn export_dashboard_json(&self, dashboard: &Dashboard) -> TorshResult<String> {
        serde_json::to_string_pretty(dashboard).map_err(|e| {
            TorshDistributedError::BackendError {
                backend: "json".to_string(),
                message: format!("JSON serialization failed: {}", e),
            }
            .into()
        })
    }
}

impl Default for VisualizationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a complete monitoring dashboard HTML file
pub fn generate_monitoring_dashboard() -> TorshResult<String> {
    let generator = VisualizationGenerator::new();
    let dashboard = generator.generate_performance_dashboard()?;
    generator.generate_html_dashboard(&dashboard)
}

/// Generate a communication network visualization
pub fn generate_communication_network_html() -> TorshResult<String> {
    let generator = VisualizationGenerator::new();
    let chart = generator.create_communication_network_graph()?;

    let mut dashboard = Dashboard::new("Communication Network Analysis".to_string());
    dashboard.add_chart(chart);

    generator.generate_html_dashboard(&dashboard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_scheme() {
        let scheme = ColorScheme::Default;
        let colors = scheme.colors();
        assert!(!colors.is_empty());
        assert_eq!(scheme.background_color(), "#ffffff");
        assert_eq!(scheme.text_color(), "#333333");
    }

    #[test]
    fn test_data_point_creation() {
        let point = DataPoint::new(1.0, 2.0)
            .with_label("Test".to_string())
            .with_metadata("key".to_string(), "value".to_string());

        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.label, Some("Test".to_string()));
        assert_eq!(point.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_chart_creation() {
        let mut chart = Chart::new("Test Chart".to_string(), ChartType::Line)
            .with_labels("X Axis".to_string(), "Y Axis".to_string());

        let mut series = ChartSeries::new("Test Series".to_string(), "#ff0000".to_string());
        series.add_point(DataPoint::new(1.0, 10.0));
        series.add_point(DataPoint::new(2.0, 20.0));

        chart.add_series(series);

        assert_eq!(chart.title, "Test Chart");
        assert_eq!(chart.chart_type, ChartType::Line);
        assert_eq!(chart.series.len(), 1);
        assert_eq!(chart.series[0].data.len(), 2);
    }

    #[test]
    fn test_dashboard_creation() {
        let mut dashboard = Dashboard::new("Test Dashboard".to_string());
        let chart = Chart::new("Test Chart".to_string(), ChartType::Bar);
        dashboard.add_chart(chart);

        assert_eq!(dashboard.title, "Test Dashboard");
        assert_eq!(dashboard.charts.len(), 1);
    }

    #[test]
    fn test_visualization_generator() {
        let generator = VisualizationGenerator::new();
        assert_eq!(generator.config.chart_width, 800);
        assert_eq!(generator.config.chart_height, 400);
    }

    #[test]
    fn test_svg_generation() {
        let generator = VisualizationGenerator::new();
        let mut chart = Chart::new("Test".to_string(), ChartType::Line);

        let mut series = ChartSeries::new("Data".to_string(), "#0000ff".to_string());
        series.add_point(DataPoint::new(0.0, 0.0));
        series.add_point(DataPoint::new(1.0, 1.0));
        chart.add_series(series);

        let svg = generator.generate_svg_chart(&chart).unwrap();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Test"));
    }

    #[test]
    fn test_html_dashboard_generation() {
        let generator = VisualizationGenerator::new();
        let dashboard = Dashboard::new("Test Dashboard".to_string());

        let html = generator.generate_html_dashboard(&dashboard).unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Dashboard"));
        assert!(html.contains("</html>"));
    }
}
