//! Advanced Visualization Export
//!
//! This module provides export capabilities for advanced visualization libraries:
//! - Plotly.js for interactive charts
//! - D3.js for custom visualizations
//! - Vega/Vega-Lite for declarative graphics
//! - Chart.js for simple charts
//! - Custom HTML dashboards

use crate::integrated_profiler::{IntegratedReport, PerformanceSnapshot};
use crate::ProfileEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::{Result as TorshResult, TorshError};

/// Visualization library type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationLibrary {
    Plotly,
    D3,
    VegaLite,
    ChartJs,
    Custom,
}

impl std::fmt::Display for VisualizationLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plotly => write!(f, "Plotly.js"),
            Self::D3 => write!(f, "D3.js"),
            Self::VegaLite => write!(f, "Vega-Lite"),
            Self::ChartJs => write!(f, "Chart.js"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Plotly chart specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyChart {
    /// Data traces
    pub data: Vec<PlotlyTrace>,
    /// Layout configuration
    pub layout: PlotlyLayout,
    /// Configuration options
    pub config: PlotlyConfig,
}

/// Plotly trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyTrace {
    /// X axis data
    pub x: Vec<serde_json::Value>,
    /// Y axis data
    pub y: Vec<serde_json::Value>,
    /// Trace type (scatter, bar, line, etc.)
    #[serde(rename = "type")]
    pub trace_type: String,
    /// Trace mode (lines, markers, lines+markers)
    pub mode: Option<String>,
    /// Trace name
    pub name: String,
    /// Marker configuration
    pub marker: Option<PlotlyMarker>,
    /// Line configuration
    pub line: Option<PlotlyLine>,
}

/// Plotly marker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyMarker {
    /// Marker color
    pub color: Option<String>,
    /// Marker size
    pub size: Option<f64>,
}

/// Plotly line configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyLine {
    /// Line color
    pub color: Option<String>,
    /// Line width
    pub width: Option<f64>,
}

/// Plotly layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyLayout {
    /// Chart title
    pub title: String,
    /// X axis configuration
    pub xaxis: PlotlyAxis,
    /// Y axis configuration
    pub yaxis: PlotlyAxis,
    /// Show legend
    pub showlegend: bool,
    /// Height
    pub height: Option<usize>,
    /// Width
    pub width: Option<usize>,
}

/// Plotly axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyAxis {
    /// Axis title
    pub title: String,
}

/// Plotly configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlotlyConfig {
    /// Responsive sizing
    pub responsive: bool,
    /// Display mode bar
    #[serde(rename = "displayModeBar")]
    pub display_mode_bar: bool,
}

impl Default for PlotlyConfig {
    fn default() -> Self {
        Self {
            responsive: true,
            display_mode_bar: true,
        }
    }
}

/// D3.js visualization specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Visualization {
    /// Data array
    pub data: Vec<HashMap<String, serde_json::Value>>,
    /// Visualization type
    pub vis_type: String,
    /// Configuration
    pub config: D3Config,
}

/// D3 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Config {
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
    /// Margins
    pub margin: D3Margin,
    /// Colors
    pub colors: Vec<String>,
}

/// D3 margins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D3Margin {
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
    pub left: usize,
}

impl Default for D3Config {
    fn default() -> Self {
        Self {
            width: 960,
            height: 500,
            margin: D3Margin {
                top: 20,
                right: 20,
                bottom: 30,
                left: 50,
            },
            colors: vec![
                "#1f77b4".to_string(),
                "#ff7f0e".to_string(),
                "#2ca02c".to_string(),
                "#d62728".to_string(),
                "#9467bd".to_string(),
            ],
        }
    }
}

/// Vega-Lite specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegaLiteSpec {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub description: String,
    pub data: VegaData,
    pub mark: String,
    pub encoding: VegaEncoding,
    pub width: Option<usize>,
    pub height: Option<usize>,
}

/// Vega data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegaData {
    pub values: Vec<HashMap<String, serde_json::Value>>,
}

/// Vega encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegaEncoding {
    pub x: Option<VegaChannel>,
    pub y: Option<VegaChannel>,
    pub color: Option<VegaChannel>,
}

/// Vega channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VegaChannel {
    pub field: String,
    #[serde(rename = "type")]
    pub channel_type: String,
    pub title: Option<String>,
}

/// Advanced visualization exporter
pub struct AdvancedVisualizationExporter;

impl AdvancedVisualizationExporter {
    /// Generate Plotly performance timeline
    pub fn plotly_performance_timeline(
        history: &[PerformanceSnapshot],
    ) -> TorshResult<PlotlyChart> {
        let timestamps: Vec<_> = history
            .iter()
            .map(|s| serde_json::Value::String(s.timestamp.to_rfc3339()))
            .collect();

        let durations: Vec<_> = history
            .iter()
            .map(|s| {
                serde_json::Value::Number(serde_json::Number::from_f64(s.avg_duration_us).unwrap())
            })
            .collect();

        let memory_values: Vec<_> = history
            .iter()
            .map(|s| {
                serde_json::Value::Number(
                    serde_json::Number::from_f64(s.avg_memory_bytes / 1_048_576.0).unwrap(),
                )
            })
            .collect();

        Ok(PlotlyChart {
            data: vec![
                PlotlyTrace {
                    x: timestamps.clone(),
                    y: durations,
                    trace_type: "scatter".to_string(),
                    mode: Some("lines+markers".to_string()),
                    name: "Duration (μs)".to_string(),
                    marker: Some(PlotlyMarker {
                        color: Some("#1f77b4".to_string()),
                        size: Some(6.0),
                    }),
                    line: Some(PlotlyLine {
                        color: Some("#1f77b4".to_string()),
                        width: Some(2.0),
                    }),
                },
                PlotlyTrace {
                    x: timestamps,
                    y: memory_values,
                    trace_type: "scatter".to_string(),
                    mode: Some("lines+markers".to_string()),
                    name: "Memory (MB)".to_string(),
                    marker: Some(PlotlyMarker {
                        color: Some("#ff7f0e".to_string()),
                        size: Some(6.0),
                    }),
                    line: Some(PlotlyLine {
                        color: Some("#ff7f0e".to_string()),
                        width: Some(2.0),
                    }),
                },
            ],
            layout: PlotlyLayout {
                title: "Performance Timeline".to_string(),
                xaxis: PlotlyAxis {
                    title: "Time".to_string(),
                },
                yaxis: PlotlyAxis {
                    title: "Value".to_string(),
                },
                showlegend: true,
                height: Some(500),
                width: Some(900),
            },
            config: PlotlyConfig::default(),
        })
    }

    /// Generate D3 force-directed graph of operation dependencies
    pub fn d3_operation_graph(events: &[ProfileEvent]) -> TorshResult<D3Visualization> {
        let mut nodes = HashMap::new();
        let mut node_id = 0;

        // Create nodes from unique operations
        for event in events {
            if !nodes.contains_key(&event.name) {
                nodes.insert(event.name.clone(), node_id);
                node_id += 1;
            }
        }

        // Create data array
        let mut data = Vec::new();
        for (name, id) in &nodes {
            let mut node = HashMap::new();
            node.insert("id".to_string(), serde_json::Value::Number((*id).into()));
            node.insert("name".to_string(), serde_json::Value::String(name.clone()));
            data.push(node);
        }

        Ok(D3Visualization {
            data,
            vis_type: "force-directed".to_string(),
            config: D3Config::default(),
        })
    }

    /// Generate Vega-Lite anomaly scatter plot
    pub fn vegalite_anomaly_plot(report: &IntegratedReport) -> TorshResult<VegaLiteSpec> {
        let mut values = Vec::new();

        // Create data points (simplified - would use actual anomaly data)
        for i in 0..10 {
            let mut point = HashMap::new();
            point.insert(
                "duration".to_string(),
                serde_json::Value::Number((100 + i * 10).into()),
            );
            point.insert(
                "memory".to_string(),
                serde_json::Value::Number((1000 + i * 100).into()),
            );
            point.insert("anomaly".to_string(), serde_json::Value::Bool(i > 7));
            values.push(point);
        }

        Ok(VegaLiteSpec {
            schema: "https://vega.github.io/schema/vega-lite/v5.json".to_string(),
            description: "Anomaly Detection Scatter Plot".to_string(),
            data: VegaData { values },
            mark: "point".to_string(),
            encoding: VegaEncoding {
                x: Some(VegaChannel {
                    field: "duration".to_string(),
                    channel_type: "quantitative".to_string(),
                    title: Some("Duration (μs)".to_string()),
                }),
                y: Some(VegaChannel {
                    field: "memory".to_string(),
                    channel_type: "quantitative".to_string(),
                    title: Some("Memory (bytes)".to_string()),
                }),
                color: Some(VegaChannel {
                    field: "anomaly".to_string(),
                    channel_type: "nominal".to_string(),
                    title: Some("Anomaly".to_string()),
                }),
            },
            width: Some(600),
            height: Some(400),
        })
    }

    /// Generate interactive HTML dashboard
    pub fn generate_html_dashboard(report: &IntegratedReport) -> TorshResult<String> {
        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>ToRSh Profiler Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .chart {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ToRSh Profiler Dashboard</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Events</div>
                <div class="stat-value">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Anomalies Detected</div>
                <div class="stat-value">{}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Prediction Error</div>
                <div class="stat-value">{:.2}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Platform</div>
                <div class="stat-value">{}</div>
            </div>
        </div>

        <h2>📊 Performance Metrics</h2>
        <div class="chart" id="performanceChart"></div>

        <h2>💡 Top Recommendations</h2>
        {}

        <h2>📈 Performance Trends</h2>
        <div class="chart">
            <p><strong>Average Duration:</strong> {:.2}μs</p>
            <p><strong>Duration Trend:</strong> {:.2}%</p>
            <p><strong>Stability Score:</strong> {:.2}/1.0</p>
        </div>
    </div>

    <script>
        // Placeholder for Plotly chart
        var trace = {{
            x: [1, 2, 3, 4, 5],
            y: [100, 105, 103, 110, 108],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Duration'
        }};
        var layout = {{
            title: 'Performance Over Time',
            xaxis: {{ title: 'Sample' }},
            yaxis: {{ title: 'Duration (μs)' }}
        }};
        Plotly.newPlot('performanceChart', [trace], layout);
    </script>
</body>
</html>"#,
            report.stats.total_events,
            report.stats.total_anomalies,
            report.stats.avg_prediction_error_percent,
            report.stats.platform_arch,
            report
                .top_recommendations
                .iter()
                .take(5)
                .map(|r| format!(
                    r#"<div class="recommendation">
                <strong>{}</strong>: {} (Expected improvement: {:.1}%)
            </div>"#,
                    r.rec_type, r.description, r.expected_improvement_percent
                ))
                .collect::<Vec<_>>()
                .join("\n        "),
            report.performance_trends.avg_duration_us,
            report.performance_trends.duration_trend_percent,
            report.performance_trends.stability_score,
        );

        Ok(html)
    }

    /// Export to JSON for any library
    pub fn export_json<T: Serialize>(data: &T) -> TorshResult<String> {
        serde_json::to_string_pretty(data)
            .map_err(|e| TorshError::operation_error(&format!("JSON export failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrated_profiler::IntegratedProfiler;

    #[test]
    fn test_plotly_chart_generation() {
        let snapshots = vec![PerformanceSnapshot {
            timestamp: chrono::Utc::now(),
            avg_duration_us: 100.0,
            avg_memory_bytes: 1_048_576.0,
            flops_rate: 1_000_000.0,
            throughput_ops_per_sec: 100.0,
            anomaly_count: 0,
            active_cluster: Some(0),
        }];

        let chart = AdvancedVisualizationExporter::plotly_performance_timeline(&snapshots);
        assert!(chart.is_ok());

        let chart = chart.unwrap();
        assert_eq!(chart.data.len(), 2); // Duration and memory traces
        assert_eq!(chart.layout.title, "Performance Timeline");
    }

    #[test]
    fn test_d3_graph_generation() {
        let events = vec![
            ProfileEvent {
                name: "op1".to_string(),
                category: "test".to_string(),
                thread_id: 1,
                start_us: 0,
                duration_us: 100,
                operation_count: None,
                flops: None,
                bytes_transferred: None,
                stack_trace: None,
            },
            ProfileEvent {
                name: "op2".to_string(),
                category: "test".to_string(),
                thread_id: 1,
                start_us: 100,
                duration_us: 150,
                operation_count: None,
                flops: None,
                bytes_transferred: None,
                stack_trace: None,
            },
        ];

        let graph = AdvancedVisualizationExporter::d3_operation_graph(&events);
        assert!(graph.is_ok());

        let graph = graph.unwrap();
        assert_eq!(graph.data.len(), 2);
        assert_eq!(graph.vis_type, "force-directed");
    }

    #[test]
    fn test_vegalite_spec_generation() {
        let mut profiler = IntegratedProfiler::new().unwrap();
        profiler.start().unwrap();
        let report = profiler.export_report().unwrap();

        let spec = AdvancedVisualizationExporter::vegalite_anomaly_plot(&report);
        assert!(spec.is_ok());

        let spec = spec.unwrap();
        assert!(spec.schema.contains("vega-lite"));
        assert_eq!(spec.mark, "point");
    }

    #[test]
    fn test_html_dashboard_generation() {
        let mut profiler = IntegratedProfiler::new().unwrap();
        profiler.start().unwrap();
        let report = profiler.export_report().unwrap();

        let html = AdvancedVisualizationExporter::generate_html_dashboard(&report);
        assert!(html.is_ok());

        let html = html.unwrap();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("ToRSh Profiler Dashboard"));
        assert!(html.contains("Plotly"));
    }

    #[test]
    fn test_json_export() {
        let chart = PlotlyChart {
            data: vec![],
            layout: PlotlyLayout {
                title: "Test".to_string(),
                xaxis: PlotlyAxis {
                    title: "X".to_string(),
                },
                yaxis: PlotlyAxis {
                    title: "Y".to_string(),
                },
                showlegend: true,
                height: None,
                width: None,
            },
            config: PlotlyConfig::default(),
        };

        let json = AdvancedVisualizationExporter::export_json(&chart);
        assert!(json.is_ok());

        let json = json.unwrap();
        assert!(json.contains("\"title\""));
        assert!(json.contains("\"Test\""));
    }
}
