//! Grafana dashboard integration for torsh-profiler
//!
//! This module provides Grafana dashboard generation and management functionality,
//! allowing automatic creation of monitoring dashboards from profiling data.

use crate::{TorshError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Grafana dashboard generator
pub struct GrafanaDashboardGenerator {
    dashboard: Dashboard,
}

/// Complete Grafana dashboard structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub title: String,
    pub uid: Option<String>,
    pub tags: Vec<String>,
    pub timezone: String,
    pub editable: bool,
    pub panels: Vec<Panel>,
    pub templating: Templating,
    pub time: TimeRange,
    pub refresh: String,
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    pub version: u32,
}

/// Grafana panel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Panel {
    pub id: u32,
    pub title: String,
    #[serde(rename = "type")]
    pub panel_type: String,
    pub datasource: String,
    pub targets: Vec<Target>,
    #[serde(rename = "gridPos")]
    pub grid_pos: GridPos,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_config: Option<FieldConfig>,
}

/// Prometheus query target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    pub expr: String,
    #[serde(rename = "legendFormat")]
    pub legend_format: String,
    #[serde(rename = "refId")]
    pub ref_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval: Option<String>,
}

/// Panel grid position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPos {
    pub h: u32, // height
    pub w: u32, // width
    pub x: u32, // x position
    pub y: u32, // y position
}

/// Dashboard templating configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Templating {
    pub list: Vec<Variable>,
}

/// Dashboard variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    #[serde(rename = "type")]
    pub var_type: String,
    pub datasource: String,
    pub query: String,
    pub multi: bool,
    #[serde(rename = "includeAll")]
    pub include_all: bool,
}

/// Time range configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub from: String,
    pub to: String,
}

/// Field configuration for panels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConfig {
    pub defaults: FieldDefaults,
    pub overrides: Vec<serde_json::Value>,
}

/// Default field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefaults {
    pub unit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thresholds: Option<Thresholds>,
}

/// Threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thresholds {
    pub mode: String,
    pub steps: Vec<ThresholdStep>,
}

/// Individual threshold step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdStep {
    pub value: f64,
    pub color: String,
}

impl GrafanaDashboardGenerator {
    /// Create a new Grafana dashboard generator
    pub fn new(title: &str) -> Self {
        Self {
            dashboard: Dashboard {
                title: title.to_string(),
                uid: None,
                tags: vec!["torsh".to_string(), "profiling".to_string()],
                timezone: "browser".to_string(),
                editable: true,
                panels: Vec::new(),
                templating: Templating { list: Vec::new() },
                time: TimeRange {
                    from: "now-1h".to_string(),
                    to: "now".to_string(),
                },
                refresh: "10s".to_string(),
                schema_version: 36,
                version: 1,
            },
        }
    }

    /// Set dashboard UID
    pub fn with_uid(mut self, uid: &str) -> Self {
        self.dashboard.uid = Some(uid.to_string());
        self
    }

    /// Add tags to the dashboard
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.dashboard.tags.extend(tags);
        self
    }

    /// Set time range
    pub fn with_time_range(mut self, from: &str, to: &str) -> Self {
        self.dashboard.time.from = from.to_string();
        self.dashboard.time.to = to.to_string();
        self
    }

    /// Set refresh interval
    pub fn with_refresh(mut self, refresh: &str) -> Self {
        self.dashboard.refresh = refresh.to_string();
        self
    }

    /// Add a graph panel
    pub fn add_graph_panel(
        &mut self,
        title: &str,
        query: &str,
        legend: &str,
        grid_pos: GridPos,
    ) -> &mut Self {
        let panel = Panel {
            id: self.dashboard.panels.len() as u32 + 1,
            title: title.to_string(),
            panel_type: "timeseries".to_string(),
            datasource: "Prometheus".to_string(),
            targets: vec![Target {
                expr: query.to_string(),
                legend_format: legend.to_string(),
                ref_id: "A".to_string(),
                interval: None,
            }],
            grid_pos,
            options: None,
            field_config: Some(FieldConfig {
                defaults: FieldDefaults {
                    unit: "short".to_string(),
                    min: None,
                    max: None,
                    thresholds: None,
                },
                overrides: Vec::new(),
            }),
        };
        self.dashboard.panels.push(panel);
        self
    }

    /// Add a heatmap panel
    pub fn add_heatmap_panel(&mut self, title: &str, query: &str, grid_pos: GridPos) -> &mut Self {
        let panel = Panel {
            id: self.dashboard.panels.len() as u32 + 1,
            title: title.to_string(),
            panel_type: "heatmap".to_string(),
            datasource: "Prometheus".to_string(),
            targets: vec![Target {
                expr: query.to_string(),
                legend_format: "".to_string(),
                ref_id: "A".to_string(),
                interval: None,
            }],
            grid_pos,
            options: None,
            field_config: None,
        };
        self.dashboard.panels.push(panel);
        self
    }

    /// Add a gauge panel
    pub fn add_gauge_panel(
        &mut self,
        title: &str,
        query: &str,
        unit: &str,
        min: f64,
        max: f64,
        grid_pos: GridPos,
    ) -> &mut Self {
        let panel = Panel {
            id: self.dashboard.panels.len() as u32 + 1,
            title: title.to_string(),
            panel_type: "gauge".to_string(),
            datasource: "Prometheus".to_string(),
            targets: vec![Target {
                expr: query.to_string(),
                legend_format: "".to_string(),
                ref_id: "A".to_string(),
                interval: None,
            }],
            grid_pos,
            options: None,
            field_config: Some(FieldConfig {
                defaults: FieldDefaults {
                    unit: unit.to_string(),
                    min: Some(min),
                    max: Some(max),
                    thresholds: Some(Thresholds {
                        mode: "absolute".to_string(),
                        steps: vec![
                            ThresholdStep {
                                value: min,
                                color: "green".to_string(),
                            },
                            ThresholdStep {
                                value: (max - min) * 0.7 + min,
                                color: "yellow".to_string(),
                            },
                            ThresholdStep {
                                value: (max - min) * 0.9 + min,
                                color: "red".to_string(),
                            },
                        ],
                    }),
                },
                overrides: Vec::new(),
            }),
        };
        self.dashboard.panels.push(panel);
        self
    }

    /// Add a stat panel
    pub fn add_stat_panel(
        &mut self,
        title: &str,
        query: &str,
        unit: &str,
        grid_pos: GridPos,
    ) -> &mut Self {
        let panel = Panel {
            id: self.dashboard.panels.len() as u32 + 1,
            title: title.to_string(),
            panel_type: "stat".to_string(),
            datasource: "Prometheus".to_string(),
            targets: vec![Target {
                expr: query.to_string(),
                legend_format: "".to_string(),
                ref_id: "A".to_string(),
                interval: None,
            }],
            grid_pos,
            options: None,
            field_config: Some(FieldConfig {
                defaults: FieldDefaults {
                    unit: unit.to_string(),
                    min: None,
                    max: None,
                    thresholds: None,
                },
                overrides: Vec::new(),
            }),
        };
        self.dashboard.panels.push(panel);
        self
    }

    /// Add a variable to the dashboard
    pub fn add_variable(
        &mut self,
        name: &str,
        query: &str,
        multi: bool,
        include_all: bool,
    ) -> &mut Self {
        let variable = Variable {
            name: name.to_string(),
            var_type: "query".to_string(),
            datasource: "Prometheus".to_string(),
            query: query.to_string(),
            multi,
            include_all,
        };
        self.dashboard.templating.list.push(variable);
        self
    }

    /// Get a reference to the dashboard
    pub fn dashboard(&self) -> &Dashboard {
        &self.dashboard
    }

    /// Build the dashboard
    pub fn build(self) -> Dashboard {
        self.dashboard
    }

    /// Export dashboard as JSON
    pub fn export_json(&self) -> TorshResult<String> {
        serde_json::to_string_pretty(&self.dashboard).map_err(|e| {
            TorshError::operation_error(&format!("Failed to serialize dashboard: {}", e))
        })
    }

    /// Export dashboard to file
    pub fn export_to_file(&self, path: &str) -> TorshResult<()> {
        let json = self.export_json()?;
        std::fs::write(path, json).map_err(|e| {
            TorshError::operation_error(&format!("Failed to write dashboard file: {}", e))
        })
    }
}

/// Pre-built dashboard templates
pub struct DashboardTemplates;

impl DashboardTemplates {
    /// Create a comprehensive profiling dashboard
    pub fn create_profiling_dashboard() -> GrafanaDashboardGenerator {
        let mut dashboard = GrafanaDashboardGenerator::new("ToRSh Profiling Overview")
            .with_uid("torsh-profiling-overview")
            .with_tags(vec!["performance".to_string(), "monitoring".to_string()])
            .with_time_range("now-15m", "now")
            .with_refresh("5s");

        // Add operation duration graph
        dashboard.add_graph_panel(
            "Operation Duration (P95)",
            r#"histogram_quantile(0.95, sum(rate(torsh_operation_duration_microseconds_bucket[5m])) by (le, operation))"#,
            "{{operation}}",
            GridPos { h: 8, w: 12, x: 0, y: 0 },
        );

        // Add operation rate graph
        dashboard.add_graph_panel(
            "Operation Rate",
            "rate(torsh_operation_total[5m])",
            "{{operation}}",
            GridPos {
                h: 8,
                w: 12,
                x: 12,
                y: 0,
            },
        );

        // Add memory allocation gauge
        dashboard.add_gauge_panel(
            "Memory Allocated",
            "sum(torsh_memory_allocated_bytes) by (operation)",
            "bytes",
            0.0,
            1e9, // 1GB
            GridPos {
                h: 6,
                w: 6,
                x: 0,
                y: 8,
            },
        );

        // Add FLOPS gauge
        dashboard.add_gauge_panel(
            "FLOPS",
            "rate(torsh_flops_total[1m])",
            "ops",
            0.0,
            1e9, // 1 GFLOPS
            GridPos {
                h: 6,
                w: 6,
                x: 6,
                y: 8,
            },
        );

        // Add thread activity stat
        dashboard.add_stat_panel(
            "Active Threads",
            "count(torsh_thread_activity)",
            "short",
            GridPos {
                h: 6,
                w: 6,
                x: 12,
                y: 8,
            },
        );

        // Add profiling overhead stat
        dashboard.add_stat_panel(
            "Profiling Overhead (Avg)",
            "avg(torsh_profiling_overhead_microseconds)",
            "µs",
            GridPos {
                h: 6,
                w: 6,
                x: 18,
                y: 8,
            },
        );

        // Add operation duration heatmap
        dashboard.add_heatmap_panel(
            "Operation Duration Heatmap",
            "sum(rate(torsh_operation_duration_microseconds_bucket[5m])) by (le)",
            GridPos {
                h: 8,
                w: 24,
                x: 0,
                y: 14,
            },
        );

        // Add operation variable
        dashboard.add_variable(
            "operation",
            "label_values(torsh_operation_total, operation)",
            true,
            true,
        );

        dashboard
    }

    /// Create a memory profiling dashboard
    pub fn create_memory_dashboard() -> GrafanaDashboardGenerator {
        let mut dashboard = GrafanaDashboardGenerator::new("ToRSh Memory Profiling")
            .with_uid("torsh-memory-profiling")
            .with_tags(vec!["memory".to_string(), "profiling".to_string()])
            .with_time_range("now-30m", "now")
            .with_refresh("10s");

        // Memory allocated over time
        dashboard.add_graph_panel(
            "Memory Allocated Over Time",
            "torsh_memory_allocated_bytes",
            "{{operation}}",
            GridPos {
                h: 8,
                w: 12,
                x: 0,
                y: 0,
            },
        );

        // Memory deallocated over time
        dashboard.add_graph_panel(
            "Memory Deallocated Over Time",
            "torsh_memory_deallocated_bytes",
            "{{operation}}",
            GridPos {
                h: 8,
                w: 12,
                x: 12,
                y: 0,
            },
        );

        // Net memory usage
        dashboard.add_graph_panel(
            "Net Memory Usage",
            "torsh_memory_allocated_bytes - torsh_memory_deallocated_bytes",
            "{{operation}}",
            GridPos {
                h: 8,
                w: 24,
                x: 0,
                y: 8,
            },
        );

        // Memory allocation rate
        dashboard.add_stat_panel(
            "Allocation Rate",
            "rate(torsh_memory_allocated_bytes[5m])",
            "Bps",
            GridPos {
                h: 6,
                w: 12,
                x: 0,
                y: 16,
            },
        );

        // Memory deallocation rate
        dashboard.add_stat_panel(
            "Deallocation Rate",
            "rate(torsh_memory_deallocated_bytes[5m])",
            "Bps",
            GridPos {
                h: 6,
                w: 12,
                x: 12,
                y: 16,
            },
        );

        dashboard
    }

    /// Create a performance metrics dashboard
    pub fn create_performance_dashboard() -> GrafanaDashboardGenerator {
        let mut dashboard = GrafanaDashboardGenerator::new("ToRSh Performance Metrics")
            .with_uid("torsh-performance-metrics")
            .with_tags(vec!["performance".to_string(), "metrics".to_string()])
            .with_time_range("now-1h", "now")
            .with_refresh("5s");

        // FLOPS over time
        dashboard.add_graph_panel(
            "FLOPS Over Time",
            "rate(torsh_flops_total[1m])",
            "{{operation}}",
            GridPos {
                h: 8,
                w: 12,
                x: 0,
                y: 0,
            },
        );

        // Bytes transferred over time
        dashboard.add_graph_panel(
            "Bytes Transferred",
            "rate(torsh_bytes_transferred_total[1m])",
            "{{operation}} - {{direction}}",
            GridPos {
                h: 8,
                w: 12,
                x: 12,
                y: 0,
            },
        );

        // Operation latency percentiles
        dashboard.add_graph_panel(
            "Operation Latency Percentiles",
            r#"histogram_quantile(0.50, sum(rate(torsh_operation_duration_microseconds_bucket[5m])) by (le, operation))"#,
            "P50",
            GridPos { h: 8, w: 24, x: 0, y: 8 },
        );

        // Throughput gauge
        dashboard.add_gauge_panel(
            "Throughput",
            "sum(rate(torsh_operation_total[1m]))",
            "ops",
            0.0,
            1000.0,
            GridPos {
                h: 6,
                w: 8,
                x: 0,
                y: 16,
            },
        );

        // Avg operation duration
        dashboard.add_gauge_panel(
            "Avg Operation Duration",
            "avg(torsh_operation_duration_microseconds)",
            "µs",
            0.0,
            10000.0,
            GridPos {
                h: 6,
                w: 8,
                x: 8,
                y: 16,
            },
        );

        // Max operation duration
        dashboard.add_gauge_panel(
            "Max Operation Duration",
            "max(torsh_operation_duration_microseconds)",
            "µs",
            0.0,
            100000.0,
            GridPos {
                h: 6,
                w: 8,
                x: 16,
                y: 16,
            },
        );

        dashboard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let dashboard = GrafanaDashboardGenerator::new("Test Dashboard");
        assert_eq!(dashboard.dashboard.title, "Test Dashboard");
        assert!(dashboard.dashboard.panels.is_empty());
    }

    #[test]
    fn test_add_graph_panel() {
        let mut dashboard = GrafanaDashboardGenerator::new("Test");
        dashboard.add_graph_panel(
            "Test Graph",
            "up",
            "{{job}}",
            GridPos {
                h: 8,
                w: 12,
                x: 0,
                y: 0,
            },
        );
        assert_eq!(dashboard.dashboard.panels.len(), 1);
        assert_eq!(dashboard.dashboard.panels[0].title, "Test Graph");
    }

    #[test]
    fn test_export_json() {
        let dashboard = GrafanaDashboardGenerator::new("Test");
        let json = dashboard.export_json();
        assert!(json.is_ok());
        assert!(json.unwrap().contains("Test"));
    }

    #[test]
    fn test_profiling_dashboard_template() {
        let dashboard = DashboardTemplates::create_profiling_dashboard();
        assert!(dashboard.dashboard.panels.len() > 0);
        assert_eq!(
            dashboard.dashboard.uid,
            Some("torsh-profiling-overview".to_string())
        );
    }

    #[test]
    fn test_memory_dashboard_template() {
        let dashboard = DashboardTemplates::create_memory_dashboard();
        assert!(dashboard.dashboard.panels.len() > 0);
        assert!(dashboard.dashboard.title.contains("Memory"));
    }

    #[test]
    fn test_performance_dashboard_template() {
        let dashboard = DashboardTemplates::create_performance_dashboard();
        assert!(dashboard.dashboard.panels.len() > 0);
        assert!(dashboard.dashboard.title.contains("Performance"));
    }
}
