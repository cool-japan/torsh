//! Advanced Visualization Tools for ToRSh Benchmarks
//!
//! This module provides comprehensive visualization capabilities for benchmark data,
//! including interactive charts, performance trend analysis, heatmaps, and statistical plots.

use crate::{
    performance_dashboards::{PerformancePoint, RegressionSeverity},
    regression_detection::AdvancedRegressionResult,
    BenchResult,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Chart width in pixels
    pub width: u32,
    /// Chart height in pixels
    pub height: u32,
    /// Chart theme
    pub theme: ChartTheme,
    /// Enable interactive features
    pub interactive: bool,
    /// Animation duration in milliseconds
    pub animation_duration: u32,
    /// Color palette
    pub color_palette: Vec<String>,
    /// Font family
    pub font_family: String,
    /// Font size
    pub font_size: u32,
    /// Export format
    pub export_format: ExportFormat,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            theme: ChartTheme::Light,
            interactive: true,
            animation_duration: 750,
            color_palette: vec![
                "#3498db".to_string(),
                "#e74c3c".to_string(),
                "#2ecc71".to_string(),
                "#f39c12".to_string(),
                "#9b59b6".to_string(),
                "#1abc9c".to_string(),
                "#34495e".to_string(),
                "#e67e22".to_string(),
            ],
            font_family: "Arial, sans-serif".to_string(),
            font_size: 12,
            export_format: ExportFormat::Html,
        }
    }
}

/// Chart theme options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartTheme {
    Light,
    Dark,
    Minimal,
    Professional,
    Colorful,
}

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    Html,
    Svg,
    Png,
    Pdf,
    Json,
}

/// Chart type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Scatter,
    Heatmap,
    Box,
    Violin,
    Histogram,
    Radar,
    Treemap,
    Sunburst,
}

/// Visualization data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationPoint {
    /// X-axis value
    pub x: f64,
    /// Y-axis value
    pub y: f64,
    /// Label for the point
    pub label: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Color override
    pub color: Option<String>,
    /// Size override
    pub size: Option<f64>,
}

/// Chart series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartSeries {
    /// Series name
    pub name: String,
    /// Data points
    pub data: Vec<VisualizationPoint>,
    /// Chart type for this series
    pub chart_type: ChartType,
    /// Color override
    pub color: Option<String>,
    /// Line width or bar width
    pub width: Option<f64>,
    /// Fill opacity
    pub fill_opacity: Option<f64>,
    /// Dash pattern for lines
    pub dash_pattern: Option<Vec<f64>>,
}

/// Chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chart {
    /// Chart title
    pub title: String,
    /// Chart subtitle
    pub subtitle: Option<String>,
    /// X-axis configuration
    pub x_axis: AxisConfig,
    /// Y-axis configuration
    pub y_axis: AxisConfig,
    /// Chart series
    pub series: Vec<ChartSeries>,
    /// Chart type (default for all series)
    pub chart_type: ChartType,
    /// Legend configuration
    pub legend: LegendConfig,
    /// Tooltip configuration
    pub tooltip: TooltipConfig,
    /// Annotations
    pub annotations: Vec<Annotation>,
}

/// Axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    /// Axis title
    pub title: String,
    /// Axis type
    pub axis_type: AxisType,
    /// Minimum value
    pub min: Option<f64>,
    /// Maximum value
    pub max: Option<f64>,
    /// Tick interval
    pub tick_interval: Option<f64>,
    /// Logarithmic scale
    pub log_scale: bool,
    /// Grid lines
    pub grid: bool,
    /// Axis label format
    pub label_format: String,
}

/// Axis type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxisType {
    Linear,
    Logarithmic,
    DateTime,
    Category,
}

/// Legend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegendConfig {
    /// Show legend
    pub show: bool,
    /// Legend position
    pub position: LegendPosition,
    /// Legend orientation
    pub orientation: LegendOrientation,
}

/// Legend position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegendPosition {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Legend orientation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegendOrientation {
    Horizontal,
    Vertical,
}

/// Tooltip configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TooltipConfig {
    /// Show tooltips
    pub show: bool,
    /// Tooltip trigger
    pub trigger: TooltipTrigger,
    /// Custom tooltip format
    pub format: Option<String>,
}

/// Tooltip trigger
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TooltipTrigger {
    Hover,
    Click,
    Focus,
}

/// Chart annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Annotation type
    pub annotation_type: AnnotationType,
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Text content
    pub text: String,
    /// Color
    pub color: Option<String>,
    /// Font size
    pub font_size: Option<u32>,
}

/// Annotation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationType {
    Text,
    Arrow,
    Line,
    Rectangle,
    Circle,
}

/// Advanced visualization generator
pub struct VisualizationGenerator {
    /// Configuration
    config: VisualizationConfig,
    /// Chart templates
    templates: HashMap<String, Chart>,
}

impl VisualizationGenerator {
    /// Create a new visualization generator
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            templates: HashMap::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(VisualizationConfig::default())
    }

    /// Generate performance trend chart
    pub fn generate_performance_trend(&self, points: &[PerformancePoint]) -> Chart {
        let mut data = Vec::new();

        for point in points {
            data.push(VisualizationPoint {
                x: point.timestamp.timestamp_millis() as f64,
                y: point.mean_time_ns / 1_000_000.0, // Convert to milliseconds
                label: format!("{} ({})", point.benchmark_name, point.size),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("benchmark".to_string(), point.benchmark_name.clone());
                    meta.insert("size".to_string(), point.size.to_string());
                    meta.insert("dtype".to_string(), point.dtype.clone());
                    meta.insert(
                        "throughput".to_string(),
                        point
                            .throughput
                            .map(|t| t.to_string())
                            .unwrap_or("N/A".to_string()),
                    );
                    meta
                },
                color: None,
                size: None,
            });
        }

        Chart {
            title: "Performance Trends Over Time".to_string(),
            subtitle: Some("Execution time trends for benchmark operations".to_string()),
            x_axis: AxisConfig {
                title: "Time".to_string(),
                axis_type: AxisType::DateTime,
                min: None,
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "%Y-%m-%d %H:%M".to_string(),
            },
            y_axis: AxisConfig {
                title: "Execution Time (ms)".to_string(),
                axis_type: AxisType::Linear,
                min: Some(0.0),
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "{:.2f}".to_string(),
            },
            series: vec![ChartSeries {
                name: "Performance".to_string(),
                data,
                chart_type: ChartType::Line,
                color: Some(self.config.color_palette[0].clone()),
                width: Some(2.0),
                fill_opacity: None,
                dash_pattern: None,
            }],
            chart_type: ChartType::Line,
            legend: LegendConfig {
                show: true,
                position: LegendPosition::TopRight,
                orientation: LegendOrientation::Vertical,
            },
            tooltip: TooltipConfig {
                show: true,
                trigger: TooltipTrigger::Hover,
                format: None,
            },
            annotations: Vec::new(),
        }
    }

    /// Generate throughput comparison chart
    pub fn generate_throughput_comparison(&self, results: &[BenchResult]) -> Chart {
        let mut series_map: HashMap<String, Vec<VisualizationPoint>> = HashMap::new();

        for result in results {
            if let Some(throughput) = result.throughput {
                let series_name = format!("{:?}", result.dtype);
                let entry = series_map.entry(series_name).or_insert_with(Vec::new);

                entry.push(VisualizationPoint {
                    x: result.size as f64,
                    y: throughput,
                    label: format!("{} ({})", result.name, result.size),
                    metadata: {
                        let mut meta = HashMap::new();
                        meta.insert("benchmark".to_string(), result.name.clone());
                        meta.insert("size".to_string(), result.size.to_string());
                        meta.insert(
                            "mean_time".to_string(),
                            (result.mean_time_ns / 1_000_000.0).to_string(),
                        );
                        meta
                    },
                    color: None,
                    size: None,
                });
            }
        }

        let mut series = Vec::new();
        for (i, (name, data)) in series_map.into_iter().enumerate() {
            series.push(ChartSeries {
                name,
                data,
                chart_type: ChartType::Bar,
                color: Some(self.config.color_palette[i % self.config.color_palette.len()].clone()),
                width: Some(0.8),
                fill_opacity: Some(0.8),
                dash_pattern: None,
            });
        }

        Chart {
            title: "Throughput Comparison".to_string(),
            subtitle: Some("Operations per second across different data types".to_string()),
            x_axis: AxisConfig {
                title: "Input Size".to_string(),
                axis_type: AxisType::Linear,
                min: None,
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "{:.0}".to_string(),
            },
            y_axis: AxisConfig {
                title: "Throughput (ops/sec)".to_string(),
                axis_type: AxisType::Linear,
                min: Some(0.0),
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "{:.0}".to_string(),
            },
            series,
            chart_type: ChartType::Bar,
            legend: LegendConfig {
                show: true,
                position: LegendPosition::TopRight,
                orientation: LegendOrientation::Vertical,
            },
            tooltip: TooltipConfig {
                show: true,
                trigger: TooltipTrigger::Hover,
                format: None,
            },
            annotations: Vec::new(),
        }
    }

    /// Generate performance heatmap
    pub fn generate_performance_heatmap(&self, results: &[BenchResult]) -> Chart {
        let mut data = Vec::new();
        let mut benchmarks = Vec::new();
        let mut sizes = Vec::new();

        // Collect unique benchmarks and sizes
        for result in results {
            if !benchmarks.contains(&result.name) {
                benchmarks.push(result.name.clone());
            }
            if !sizes.contains(&result.size) {
                sizes.push(result.size);
            }
        }

        benchmarks.sort();
        sizes.sort();

        // Create heatmap data
        for (i, benchmark) in benchmarks.iter().enumerate() {
            for (j, &size) in sizes.iter().enumerate() {
                if let Some(result) = results
                    .iter()
                    .find(|r| r.name == *benchmark && r.size == size)
                {
                    data.push(VisualizationPoint {
                        x: j as f64,
                        y: i as f64,
                        label: format!("{} ({})", benchmark, size),
                        metadata: {
                            let mut meta = HashMap::new();
                            meta.insert("benchmark".to_string(), benchmark.clone());
                            meta.insert("size".to_string(), size.to_string());
                            meta.insert(
                                "value".to_string(),
                                (result.mean_time_ns / 1_000_000.0).to_string(),
                            );
                            meta
                        },
                        color: None,
                        size: Some(result.mean_time_ns / 1_000_000.0),
                    });
                }
            }
        }

        Chart {
            title: "Performance Heatmap".to_string(),
            subtitle: Some("Execution time across benchmarks and input sizes".to_string()),
            x_axis: AxisConfig {
                title: "Input Size".to_string(),
                axis_type: AxisType::Category,
                min: None,
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: false,
                label_format: "{:.0}".to_string(),
            },
            y_axis: AxisConfig {
                title: "Benchmark".to_string(),
                axis_type: AxisType::Category,
                min: None,
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: false,
                label_format: "{}".to_string(),
            },
            series: vec![ChartSeries {
                name: "Execution Time".to_string(),
                data,
                chart_type: ChartType::Heatmap,
                color: None,
                width: None,
                fill_opacity: Some(0.8),
                dash_pattern: None,
            }],
            chart_type: ChartType::Heatmap,
            legend: LegendConfig {
                show: true,
                position: LegendPosition::Right,
                orientation: LegendOrientation::Vertical,
            },
            tooltip: TooltipConfig {
                show: true,
                trigger: TooltipTrigger::Hover,
                format: Some("Value: {:.2f} ms".to_string()),
            },
            annotations: Vec::new(),
        }
    }

    /// Generate regression analysis chart
    pub fn generate_regression_analysis(&self, regressions: &[AdvancedRegressionResult]) -> Chart {
        let mut data = Vec::new();

        for (i, regression) in regressions.iter().enumerate() {
            data.push(VisualizationPoint {
                x: i as f64,
                y: regression.effect_size * 100.0,
                label: regression.benchmark_id.clone(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("benchmark".to_string(), regression.benchmark_id.clone());
                    meta.insert("p_value".to_string(), regression.p_value.to_string());
                    meta.insert(
                        "effect_size".to_string(),
                        regression.effect_size.to_string(),
                    );
                    meta.insert("severity".to_string(), format!("{:?}", regression.severity));
                    meta
                },
                color: Some(match regression.severity {
                    RegressionSeverity::Minor => "#f39c12".to_string(),
                    RegressionSeverity::Moderate => "#e67e22".to_string(),
                    RegressionSeverity::Major => "#d35400".to_string(),
                    RegressionSeverity::Critical => "#c0392b".to_string(),
                }),
                size: Some(regression.p_value / 10.0),
            });
        }

        Chart {
            title: "Performance Regression Analysis".to_string(),
            subtitle: Some("Performance changes with severity indicators".to_string()),
            x_axis: AxisConfig {
                title: "Benchmark Index".to_string(),
                axis_type: AxisType::Linear,
                min: None,
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "{:.0}".to_string(),
            },
            y_axis: AxisConfig {
                title: "Performance Change (%)".to_string(),
                axis_type: AxisType::Linear,
                min: None,
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "{:.1f}%".to_string(),
            },
            series: vec![ChartSeries {
                name: "Regressions".to_string(),
                data,
                chart_type: ChartType::Scatter,
                color: None,
                width: None,
                fill_opacity: Some(0.7),
                dash_pattern: None,
            }],
            chart_type: ChartType::Scatter,
            legend: LegendConfig {
                show: false,
                position: LegendPosition::TopRight,
                orientation: LegendOrientation::Vertical,
            },
            tooltip: TooltipConfig {
                show: true,
                trigger: TooltipTrigger::Hover,
                format: Some("Change: {:.1f}%, Confidence: {:.1f}%".to_string()),
            },
            annotations: vec![
                Annotation {
                    annotation_type: AnnotationType::Line,
                    x: 0.0,
                    y: 0.0,
                    text: "No Change".to_string(),
                    color: Some("#7f8c8d".to_string()),
                    font_size: None,
                },
                Annotation {
                    annotation_type: AnnotationType::Line,
                    x: 0.0,
                    y: 10.0,
                    text: "10% Regression Threshold".to_string(),
                    color: Some("#e74c3c".to_string()),
                    font_size: None,
                },
            ],
        }
    }

    /// Generate memory usage analysis chart
    pub fn generate_memory_analysis(&self, results: &[BenchResult]) -> Chart {
        let mut execution_data = Vec::new();
        let mut memory_data = Vec::new();

        for result in results {
            if let Some(memory) = result.memory_usage {
                execution_data.push(VisualizationPoint {
                    x: result.size as f64,
                    y: result.mean_time_ns / 1_000_000.0,
                    label: format!("{} ({})", result.name, result.size),
                    metadata: HashMap::new(),
                    color: None,
                    size: None,
                });

                memory_data.push(VisualizationPoint {
                    x: result.size as f64,
                    y: memory as f64 / 1_048_576.0, // Convert to MB
                    label: format!("{} ({})", result.name, result.size),
                    metadata: HashMap::new(),
                    color: None,
                    size: None,
                });
            }
        }

        Chart {
            title: "Memory Usage vs Execution Time".to_string(),
            subtitle: Some("Correlation between memory usage and performance".to_string()),
            x_axis: AxisConfig {
                title: "Input Size".to_string(),
                axis_type: AxisType::Linear,
                min: None,
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "{:.0}".to_string(),
            },
            y_axis: AxisConfig {
                title: "Value".to_string(),
                axis_type: AxisType::Linear,
                min: Some(0.0),
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "{:.2f}".to_string(),
            },
            series: vec![
                ChartSeries {
                    name: "Execution Time (ms)".to_string(),
                    data: execution_data,
                    chart_type: ChartType::Line,
                    color: Some(self.config.color_palette[0].clone()),
                    width: Some(2.0),
                    fill_opacity: None,
                    dash_pattern: None,
                },
                ChartSeries {
                    name: "Memory Usage (MB)".to_string(),
                    data: memory_data,
                    chart_type: ChartType::Bar,
                    color: Some(self.config.color_palette[1].clone()),
                    width: Some(0.6),
                    fill_opacity: Some(0.7),
                    dash_pattern: None,
                },
            ],
            chart_type: ChartType::Line,
            legend: LegendConfig {
                show: true,
                position: LegendPosition::TopLeft,
                orientation: LegendOrientation::Vertical,
            },
            tooltip: TooltipConfig {
                show: true,
                trigger: TooltipTrigger::Hover,
                format: None,
            },
            annotations: Vec::new(),
        }
    }

    /// Generate statistical distribution chart
    pub fn generate_distribution_chart(&self, results: &[BenchResult]) -> Chart {
        let mut data = Vec::new();
        let mut benchmarks = Vec::new();

        // Group results by benchmark name
        let mut benchmark_groups: HashMap<String, Vec<&BenchResult>> = HashMap::new();
        for result in results {
            benchmark_groups
                .entry(result.name.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (i, (benchmark, group)) in benchmark_groups.iter().enumerate() {
            benchmarks.push(benchmark.clone());

            // Calculate statistics for box plot
            let mut times: Vec<f64> = group.iter().map(|r| r.mean_time_ns / 1_000_000.0).collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let min = times[0];
            let max = times[times.len() - 1];
            let median = times[times.len() / 2];
            let q1 = times[times.len() / 4];
            let q3 = times[3 * times.len() / 4];

            data.push(VisualizationPoint {
                x: i as f64,
                y: median,
                label: benchmark.clone(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("min".to_string(), min.to_string());
                    meta.insert("q1".to_string(), q1.to_string());
                    meta.insert("median".to_string(), median.to_string());
                    meta.insert("q3".to_string(), q3.to_string());
                    meta.insert("max".to_string(), max.to_string());
                    meta
                },
                color: None,
                size: None,
            });
        }

        Chart {
            title: "Performance Distribution".to_string(),
            subtitle: Some("Statistical distribution of benchmark execution times".to_string()),
            x_axis: AxisConfig {
                title: "Benchmark".to_string(),
                axis_type: AxisType::Category,
                min: None,
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: false,
                label_format: "{}".to_string(),
            },
            y_axis: AxisConfig {
                title: "Execution Time (ms)".to_string(),
                axis_type: AxisType::Linear,
                min: Some(0.0),
                max: None,
                tick_interval: None,
                log_scale: false,
                grid: true,
                label_format: "{:.2f}".to_string(),
            },
            series: vec![ChartSeries {
                name: "Distribution".to_string(),
                data,
                chart_type: ChartType::Box,
                color: Some(self.config.color_palette[0].clone()),
                width: Some(0.8),
                fill_opacity: Some(0.7),
                dash_pattern: None,
            }],
            chart_type: ChartType::Box,
            legend: LegendConfig {
                show: false,
                position: LegendPosition::TopRight,
                orientation: LegendOrientation::Vertical,
            },
            tooltip: TooltipConfig {
                show: true,
                trigger: TooltipTrigger::Hover,
                format: Some("Median: {:.2f} ms".to_string()),
            },
            annotations: Vec::new(),
        }
    }

    /// Export chart to HTML
    pub fn export_to_html(&self, chart: &Chart, output_path: &str) -> std::io::Result<()> {
        let html = self.generate_html_chart(chart);
        std::fs::write(output_path, html)?;
        Ok(())
    }

    /// Generate HTML representation of chart
    fn generate_html_chart(&self, chart: &Chart) -> String {
        let chart_json = serde_json::to_string_pretty(chart).unwrap_or_default();
        let theme_css = self.get_theme_css();

        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: {};
            margin: 20px;
            background: {};
        }}
        .chart-container {{
            width: {}px;
            height: {}px;
            margin: 20px auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .chart-title {{
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .chart-subtitle {{
            text-align: center;
            font-size: 16px;
            color: #7f8c8d;
            margin-bottom: 20px;
        }}
        {}
    </style>
</head>
<body>
    <div class="chart-container">
        <div class="chart-title">{}</div>
        {}
        <div id="chart"></div>
    </div>
    
    <script>
        const chartData = {};
        
        // Convert chart data to Plotly format
        const plotlyData = chartData.series.map(series => ({{
            x: series.data.map(point => point.x),
            y: series.data.map(point => point.y),
            type: getPlotlyType(series.chart_type),
            name: series.name,
            marker: {{ color: series.color }},
            mode: 'lines+markers'
        }}));
        
        const layout = {{
            title: chartData.title,
            xaxis: {{ title: chartData.x_axis.title }},
            yaxis: {{ title: chartData.y_axis.title }},
            showlegend: chartData.legend.show,
            font: {{ family: '{}', size: {} }},
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            autosize: true,
            responsive: true
        }};
        
        const config = {{
            responsive: true,
            displayModeBar: {},
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        }};
        
        Plotly.newPlot('chart', plotlyData, layout, config);
        
        function getPlotlyType(chartType) {{
            switch (chartType) {{
                case 'Line': return 'scatter';
                case 'Bar': return 'bar';
                case 'Scatter': return 'scatter';
                case 'Heatmap': return 'heatmap';
                case 'Box': return 'box';
                case 'Histogram': return 'histogram';
                default: return 'scatter';
            }}
        }}
    </script>
</body>
</html>
        "#,
            chart.title,
            self.config.font_family,
            self.get_background_color(),
            self.config.width,
            self.config.height,
            theme_css,
            chart.title,
            chart
                .subtitle
                .as_ref()
                .map(|s| format!("<div class=\"chart-subtitle\">{}</div>", s))
                .unwrap_or_default(),
            chart_json,
            self.config.font_family,
            self.config.font_size,
            self.config.interactive.to_string()
        )
    }

    /// Get theme-specific CSS
    fn get_theme_css(&self) -> String {
        match self.config.theme {
            ChartTheme::Dark => r#"
                body { background: #2c3e50; color: white; }
                .chart-container { background: #34495e; }
                .chart-title { color: #ecf0f1; }
                .chart-subtitle { color: #bdc3c7; }
            "#
            .to_string(),
            ChartTheme::Minimal => r#"
                body { background: #fafafa; }
                .chart-container { box-shadow: none; border: 1px solid #e0e0e0; }
            "#
            .to_string(),
            ChartTheme::Professional => r#"
                body { background: #f8f9fa; }
                .chart-container { box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .chart-title { color: #343a40; }
            "#
            .to_string(),
            _ => String::new(),
        }
    }

    /// Get background color for theme
    fn get_background_color(&self) -> &'static str {
        match self.config.theme {
            ChartTheme::Dark => "#2c3e50",
            ChartTheme::Minimal => "#fafafa",
            ChartTheme::Professional => "#f8f9fa",
            _ => "#f5f5f5",
        }
    }

    /// Save chart template
    pub fn save_template(&mut self, name: &str, chart: Chart) {
        self.templates.insert(name.to_string(), chart);
    }

    /// Load chart template
    pub fn load_template(&self, name: &str) -> Option<&Chart> {
        self.templates.get(name)
    }

    /// Generate comprehensive dashboard
    pub fn generate_dashboard(
        &self,
        results: &[BenchResult],
        points: &[PerformancePoint],
        regressions: &[AdvancedRegressionResult],
        output_dir: &str,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // Generate individual charts
        let trend_chart = self.generate_performance_trend(points);
        self.export_to_html(&trend_chart, &format!("{}/trend.html", output_dir))?;

        let throughput_chart = self.generate_throughput_comparison(results);
        self.export_to_html(
            &throughput_chart,
            &format!("{}/throughput.html", output_dir),
        )?;

        let heatmap_chart = self.generate_performance_heatmap(results);
        self.export_to_html(&heatmap_chart, &format!("{}/heatmap.html", output_dir))?;

        let regression_chart = self.generate_regression_analysis(regressions);
        self.export_to_html(
            &regression_chart,
            &format!("{}/regressions.html", output_dir),
        )?;

        let memory_chart = self.generate_memory_analysis(results);
        self.export_to_html(&memory_chart, &format!("{}/memory.html", output_dir))?;

        let distribution_chart = self.generate_distribution_chart(results);
        self.export_to_html(
            &distribution_chart,
            &format!("{}/distribution.html", output_dir),
        )?;

        // Generate main dashboard index
        let dashboard_html = self.generate_dashboard_index();
        std::fs::write(format!("{}/index.html", output_dir), dashboard_html)?;

        Ok(())
    }

    /// Generate dashboard index page
    fn generate_dashboard_index(&self) -> String {
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ToRSh Benchmark Visualization Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            text-align: center;
            background: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .card h3 {
            color: #2c3e50;
            margin-top: 0;
        }
        .card p {
            color: #7f8c8d;
            margin-bottom: 20px;
        }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #2980b9;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 ToRSh Benchmark Visualization Dashboard</h1>
        <p>Comprehensive performance analysis and visualization tools</p>
    </div>

    <div class="grid">
        <div class="card">
            <h3>📈 Performance Trends</h3>
            <p>Track performance changes over time with interactive trend analysis</p>
            <a href="trend.html" class="btn">View Trends</a>
        </div>
        
        <div class="card">
            <h3>🏃 Throughput Comparison</h3>
            <p>Compare throughput across different data types and operations</p>
            <a href="throughput.html" class="btn">View Throughput</a>
        </div>
        
        <div class="card">
            <h3>🔥 Performance Heatmap</h3>
            <p>Visualize performance patterns across benchmarks and sizes</p>
            <a href="heatmap.html" class="btn">View Heatmap</a>
        </div>
        
        <div class="card">
            <h3>⚠️ Regression Analysis</h3>
            <p>Identify and analyze performance regressions with severity levels</p>
            <a href="regressions.html" class="btn">View Regressions</a>
        </div>
        
        <div class="card">
            <h3>💾 Memory Analysis</h3>
            <p>Analyze memory usage patterns and correlations with performance</p>
            <a href="memory.html" class="btn">View Memory</a>
        </div>
        
        <div class="card">
            <h3>📊 Statistical Distribution</h3>
            <p>Explore statistical distributions of benchmark results</p>
            <a href="distribution.html" class="btn">View Distribution</a>
        </div>
    </div>

    <div class="footer">
        <p>Generated by ToRSh Benchmark Visualization System</p>
    </div>
</body>
</html>
        "#
        .to_string()
    }
}

/// Visualization builder for easy chart creation
pub struct VisualizationBuilder {
    config: VisualizationConfig,
}

impl VisualizationBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: VisualizationConfig::default(),
        }
    }

    /// Set chart dimensions
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    /// Set chart theme
    pub fn theme(mut self, theme: ChartTheme) -> Self {
        self.config.theme = theme;
        self
    }

    /// Set interactive mode
    pub fn interactive(mut self, interactive: bool) -> Self {
        self.config.interactive = interactive;
        self
    }

    /// Set color palette
    pub fn colors(mut self, colors: Vec<String>) -> Self {
        self.config.color_palette = colors;
        self
    }

    /// Build the visualization generator
    pub fn build(self) -> VisualizationGenerator {
        VisualizationGenerator::new(self.config)
    }
}

impl Default for VisualizationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchResult;

    #[test]
    fn test_visualization_config() {
        let config = VisualizationConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.theme, ChartTheme::Light);
        assert!(config.interactive);
    }

    #[test]
    fn test_visualization_builder() {
        let generator = VisualizationBuilder::new()
            .dimensions(1024, 768)
            .theme(ChartTheme::Dark)
            .interactive(false)
            .build();

        assert_eq!(generator.config.width, 1024);
        assert_eq!(generator.config.height, 768);
        assert_eq!(generator.config.theme, ChartTheme::Dark);
        assert!(!generator.config.interactive);
    }

    #[test]
    fn test_throughput_chart_generation() {
        let generator = VisualizationGenerator::default();
        let results = vec![BenchResult {
            name: "test_benchmark".to_string(),
            size: 1024,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: 1000.0,
            std_dev_ns: 100.0,
            throughput: Some(1000.0),
            memory_usage: Some(1024),
            peak_memory: Some(2048),
            metrics: HashMap::new(),
        }];

        let chart = generator.generate_throughput_comparison(&results);
        assert_eq!(chart.title, "Throughput Comparison");
        assert!(!chart.series.is_empty());
        assert_eq!(chart.chart_type, ChartType::Bar);
    }

    #[test]
    fn test_performance_trend_generation() {
        let generator = VisualizationGenerator::default();
        let points = vec![PerformancePoint {
            timestamp: Utc::now(),
            benchmark_name: "test".to_string(),
            size: 1024,
            dtype: "F32".to_string(),
            mean_time_ns: 1000.0,
            std_dev_ns: 100.0,
            throughput: Some(1000.0),
            memory_usage: Some(1024),
            peak_memory: Some(2048),
            git_commit: None,
            build_config: "release".to_string(),
            metadata: HashMap::new(),
        }];

        let chart = generator.generate_performance_trend(&points);
        assert_eq!(chart.title, "Performance Trends Over Time");
        assert!(!chart.series.is_empty());
        assert_eq!(chart.chart_type, ChartType::Line);
    }
}
