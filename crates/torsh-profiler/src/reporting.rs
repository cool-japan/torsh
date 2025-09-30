//! Comprehensive reporting system for performance profiling
//!
//! This module provides advanced reporting capabilities including automated report generation,
//! scheduled reports, multi-format export, and customizable templates.

use crate::{
    alerts::{Alert, AlertStats},
    regression::RegressionResult,
    ProfileEvent,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Report types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportType {
    Performance,
    Memory,
    Alerts,
    Regression,
    Summary,
    Detailed,
    Custom(String),
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportFormat {
    Html,
    Pdf,
    Json,
    Csv,
    Markdown,
    Excel,
    Xml,
}

/// Report frequency for scheduled reports
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReportFrequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    OnDemand,
    OnThreshold,
}

/// Report configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub name: String,
    pub description: String,
    pub report_type: ReportType,
    pub format: ReportFormat,
    pub frequency: ReportFrequency,
    pub output_path: String,
    pub template_path: Option<String>,
    pub include_charts: bool,
    pub include_raw_data: bool,
    pub time_range: Option<Duration>,
    pub filters: Vec<ReportFilter>,
    pub recipients: Vec<String>,
    pub enabled: bool,
}

/// Report filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFilter {
    OperationName(String),
    MinDuration(u64),
    MaxDuration(u64),
    ThreadId(String),
    Severity(String),
    TimeRange {
        start: SystemTime,
        end: SystemTime,
    },
    Custom {
        field: String,
        operator: String,
        value: String,
    },
}

/// Complete performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub metadata: ReportMetadata,
    pub summary: ReportSummary,
    pub performance_analysis: PerformanceAnalysis,
    pub memory_analysis: MemoryAnalysis,
    pub alert_analysis: AlertAnalysis,
    pub regression_analysis: Option<Vec<RegressionResult>>,
    pub recommendations: Vec<Recommendation>,
    pub charts: Vec<ChartData>,
    pub raw_data: Option<Vec<ProfileEvent>>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    pub id: String,
    pub name: String,
    pub generated_at: SystemTime,
    pub time_range: TimeRange,
    pub total_events: usize,
    pub report_type: ReportType,
    pub format: ReportFormat,
    pub version: String,
}

/// Time range for report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: SystemTime,
    pub end: SystemTime,
    pub duration: Duration,
}

/// Report summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_operations: u64,
    pub total_duration_ns: u64,
    pub average_duration_ns: u64,
    pub min_duration_ns: u64,
    pub max_duration_ns: u64,
    pub total_memory_bytes: u64,
    pub peak_memory_bytes: u64,
    pub total_flops: u64,
    pub operations_per_second: f64,
    pub gflops_per_second: f64,
    pub throughput_mbps: f64,
}

/// Performance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    pub slowest_operations: Vec<OperationSummary>,
    pub fastest_operations: Vec<OperationSummary>,
    pub most_frequent_operations: Vec<OperationSummary>,
    pub performance_trends: Vec<TrendData>,
    pub bottlenecks: Vec<BottleneckInfo>,
    pub efficiency_score: f64,
}

/// Memory analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    pub peak_usage: u64,
    pub average_usage: u64,
    pub allocation_rate: f64,
    pub deallocation_rate: f64,
    pub fragmentation_ratio: f64,
    pub memory_leaks: Vec<MemoryLeakInfo>,
    pub memory_trends: Vec<TrendData>,
}

/// Alert analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAnalysis {
    pub total_alerts: u64,
    pub alerts_by_severity: HashMap<String, u64>,
    pub alerts_by_operation: HashMap<String, u64>,
    pub alert_trends: Vec<TrendData>,
    pub mean_time_to_resolution: Duration,
    pub false_positive_rate: f64,
}

/// Operation summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSummary {
    pub name: String,
    pub count: u64,
    pub total_duration_ns: u64,
    pub average_duration_ns: u64,
    pub min_duration_ns: u64,
    pub max_duration_ns: u64,
    pub std_deviation_ns: u64,
    pub percentile_95_ns: u64,
    pub percentile_99_ns: u64,
}

/// Trend data for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendData {
    pub timestamp: SystemTime,
    pub value: f64,
    pub label: String,
}

/// Bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckInfo {
    pub operation: String,
    pub bottleneck_type: String,
    pub severity: String,
    pub impact_percentage: f64,
    pub description: String,
    pub recommendations: Vec<String>,
}

/// Memory leak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeakInfo {
    pub location: String,
    pub leaked_bytes: u64,
    pub allocation_count: u64,
    pub first_seen: SystemTime,
    pub last_seen: SystemTime,
}

/// Recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub priority: String,
    pub category: String,
    pub title: String,
    pub description: String,
    pub potential_impact: String,
    pub implementation_effort: String,
}

/// Chart data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    pub chart_type: String,
    pub title: String,
    pub x_axis: String,
    pub y_axis: String,
    pub data: Vec<ChartPoint>,
}

/// Chart point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartPoint {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
}

/// Report generator
pub struct ReportGenerator {
    config: ReportConfig,
    template_engine: TemplateEngine,
}

impl ReportGenerator {
    pub fn new(config: ReportConfig) -> Self {
        Self {
            template_engine: TemplateEngine::new(),
            config,
        }
    }

    /// Generate a complete performance report
    pub fn generate_report(
        &self,
        events: &[ProfileEvent],
        alerts: &[Alert],
    ) -> Result<PerformanceReport> {
        let filtered_events = self.apply_filters(events);

        let metadata = self.generate_metadata(&filtered_events);
        let summary = self.generate_summary(&filtered_events);
        let performance_analysis = self.analyze_performance(&filtered_events);
        let memory_analysis = self.analyze_memory(&filtered_events);
        let alert_analysis = self.analyze_alerts(alerts);
        let recommendations =
            self.generate_recommendations(&performance_analysis, &memory_analysis, &alert_analysis);
        let charts = self.generate_charts(&filtered_events, &performance_analysis);
        let raw_data = if self.config.include_raw_data {
            Some(filtered_events)
        } else {
            None
        };

        Ok(PerformanceReport {
            metadata,
            summary,
            performance_analysis,
            memory_analysis,
            alert_analysis,
            regression_analysis: None, // Would be populated if available
            recommendations,
            charts,
            raw_data,
        })
    }

    fn apply_filters(&self, events: &[ProfileEvent]) -> Vec<ProfileEvent> {
        let mut filtered = events.to_vec();

        for filter in &self.config.filters {
            filtered.retain(|event| {
                match filter {
                    ReportFilter::OperationName(name) => event.name.contains(name),
                    ReportFilter::MinDuration(min) => (event.duration_us * 1000) >= *min,
                    ReportFilter::MaxDuration(max) => (event.duration_us * 1000) <= *max,
                    ReportFilter::ThreadId(id) => format!("{:?}", event.thread_id).contains(id),
                    ReportFilter::TimeRange { start, end } => {
                        let event_time =
                            SystemTime::UNIX_EPOCH + Duration::from_micros(event.start_us);
                        event_time >= *start && event_time <= *end
                    }
                    _ => true, // Skip unsupported filters for now
                }
            });
        }

        // Apply time range if specified
        if let Some(time_range) = self.config.time_range {
            let cutoff = SystemTime::now() - time_range;
            filtered.retain(|event| {
                let event_time = SystemTime::UNIX_EPOCH + Duration::from_micros(event.start_us);
                event_time >= cutoff
            });
        }

        filtered
    }

    fn generate_metadata(&self, events: &[ProfileEvent]) -> ReportMetadata {
        let start_time = events
            .iter()
            .map(|e| SystemTime::UNIX_EPOCH + Duration::from_micros(e.start_us))
            .min()
            .unwrap_or(SystemTime::now());
        let end_time = events
            .iter()
            .map(|e| SystemTime::UNIX_EPOCH + Duration::from_micros(e.start_us))
            .max()
            .unwrap_or(SystemTime::now());
        let duration = end_time
            .duration_since(start_time)
            .unwrap_or(Duration::ZERO);

        ReportMetadata {
            id: format!(
                "report_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            name: self.config.name.clone(),
            generated_at: SystemTime::now(),
            time_range: TimeRange {
                start: start_time,
                end: end_time,
                duration,
            },
            total_events: events.len(),
            report_type: self.config.report_type.clone(),
            format: self.config.format.clone(),
            version: "1.0.0".to_string(),
        }
    }

    fn generate_summary(&self, events: &[ProfileEvent]) -> ReportSummary {
        let total_operations = events.len() as u64;
        let total_duration_ns: u64 = events.iter().map(|e| e.duration_us * 1000).sum(); // Convert to ns
        let average_duration_ns = if total_operations > 0 {
            total_duration_ns / total_operations
        } else {
            0
        };
        let min_duration_ns = events
            .iter()
            .map(|e| e.duration_us * 1000)
            .min()
            .unwrap_or(0);
        let max_duration_ns = events
            .iter()
            .map(|e| e.duration_us * 1000)
            .max()
            .unwrap_or(0);

        let total_memory_bytes: u64 = events.iter().filter_map(|e| e.bytes_transferred).sum();
        let peak_memory_bytes = events
            .iter()
            .filter_map(|e| e.bytes_transferred)
            .max()
            .unwrap_or(0);
        let total_flops: u64 = events.iter().filter_map(|e| e.flops).sum();

        let total_duration_seconds = total_duration_ns as f64 / 1_000_000_000.0;
        let operations_per_second = if total_duration_seconds > 0.0 {
            total_operations as f64 / total_duration_seconds
        } else {
            0.0
        };
        let gflops_per_second = if total_duration_seconds > 0.0 {
            total_flops as f64 / total_duration_seconds / 1_000_000_000.0
        } else {
            0.0
        };
        let throughput_mbps = if total_duration_seconds > 0.0 {
            total_memory_bytes as f64 / total_duration_seconds / 1_048_576.0
        } else {
            0.0
        };

        ReportSummary {
            total_operations,
            total_duration_ns,
            average_duration_ns,
            min_duration_ns,
            max_duration_ns,
            total_memory_bytes,
            peak_memory_bytes,
            total_flops,
            operations_per_second,
            gflops_per_second,
            throughput_mbps,
        }
    }

    fn analyze_performance(&self, events: &[ProfileEvent]) -> PerformanceAnalysis {
        let mut operation_stats: HashMap<String, Vec<u64>> = HashMap::new();

        for event in events {
            operation_stats
                .entry(event.name.clone())
                .or_default()
                .push(event.duration_us * 1000); // Convert to ns
        }

        let mut operation_summaries: Vec<OperationSummary> = operation_stats
            .iter()
            .map(|(name, durations)| {
                let count = durations.len() as u64;
                let total_duration_ns: u64 = durations.iter().sum();
                let average_duration_ns = total_duration_ns / count;
                let min_duration_ns = *durations.iter().min().unwrap_or(&0);
                let max_duration_ns = *durations.iter().max().unwrap_or(&0);

                // Calculate standard deviation
                let variance = durations
                    .iter()
                    .map(|d| {
                        let diff = *d as f64 - average_duration_ns as f64;
                        diff * diff
                    })
                    .sum::<f64>()
                    / count as f64;
                let std_deviation_ns = variance.sqrt() as u64;

                // Calculate percentiles
                let mut sorted_durations = durations.clone();
                sorted_durations.sort();
                let percentile_95_ns = sorted_durations
                    .get((count as f64 * 0.95) as usize)
                    .copied()
                    .unwrap_or(0);
                let percentile_99_ns = sorted_durations
                    .get((count as f64 * 0.99) as usize)
                    .copied()
                    .unwrap_or(0);

                OperationSummary {
                    name: name.clone(),
                    count,
                    total_duration_ns,
                    average_duration_ns,
                    min_duration_ns,
                    max_duration_ns,
                    std_deviation_ns,
                    percentile_95_ns,
                    percentile_99_ns,
                }
            })
            .collect();

        // Sort by different criteria
        let mut slowest = operation_summaries.clone();
        slowest.sort_by(|a, b| b.average_duration_ns.cmp(&a.average_duration_ns));
        slowest.truncate(10);

        let mut fastest = operation_summaries.clone();
        fastest.sort_by(|a, b| a.average_duration_ns.cmp(&b.average_duration_ns));
        fastest.truncate(10);

        let mut most_frequent = operation_summaries.clone();
        most_frequent.sort_by(|a, b| b.count.cmp(&a.count));
        most_frequent.truncate(10);

        let performance_trends = self.calculate_performance_trends(events);
        let bottlenecks = self.identify_bottlenecks(&operation_summaries);
        let efficiency_score = self.calculate_efficiency_score(&operation_summaries);

        PerformanceAnalysis {
            slowest_operations: slowest,
            fastest_operations: fastest,
            most_frequent_operations: most_frequent,
            performance_trends,
            bottlenecks,
            efficiency_score,
        }
    }

    fn analyze_memory(&self, events: &[ProfileEvent]) -> MemoryAnalysis {
        let memory_events: Vec<u64> = events.iter().filter_map(|e| e.bytes_transferred).collect();

        let peak_usage = memory_events.iter().max().copied().unwrap_or(0);
        let average_usage = if !memory_events.is_empty() {
            memory_events.iter().sum::<u64>() / memory_events.len() as u64
        } else {
            0
        };

        let allocation_rate = if events.len() > 1 {
            let last_time =
                SystemTime::UNIX_EPOCH + Duration::from_micros(events.last().unwrap().start_us);
            let first_time =
                SystemTime::UNIX_EPOCH + Duration::from_micros(events.first().unwrap().start_us);
            let time_range = last_time
                .duration_since(first_time)
                .unwrap_or(Duration::from_secs(1));
            memory_events.len() as f64 / time_range.as_secs_f64()
        } else {
            0.0
        };

        let memory_trends = self.calculate_memory_trends(events);

        MemoryAnalysis {
            peak_usage,
            average_usage,
            allocation_rate,
            deallocation_rate: allocation_rate * 0.8, // Approximation
            fragmentation_ratio: 0.1,                 // Would need more sophisticated analysis
            memory_leaks: vec![],                     // Would need leak detection data
            memory_trends,
        }
    }

    fn analyze_alerts(&self, alerts: &[Alert]) -> AlertAnalysis {
        let total_alerts = alerts.len() as u64;

        let mut alerts_by_severity = HashMap::new();
        let mut alerts_by_operation = HashMap::new();

        for alert in alerts {
            let severity = alert.severity.to_string();
            *alerts_by_severity.entry(severity).or_insert(0) += 1;

            if let Some(operation) = alert.metadata.get("operation") {
                *alerts_by_operation.entry(operation.clone()).or_insert(0) += 1;
            }
        }

        let resolved_alerts: Vec<_> = alerts.iter().filter(|a| a.resolved).collect();
        let mean_time_to_resolution = if !resolved_alerts.is_empty() {
            let total_resolution_time: Duration = resolved_alerts
                .iter()
                .filter_map(|a| a.resolved_at?.duration_since(a.timestamp).ok())
                .sum();
            total_resolution_time / resolved_alerts.len() as u32
        } else {
            Duration::ZERO
        };

        AlertAnalysis {
            total_alerts,
            alerts_by_severity,
            alerts_by_operation,
            alert_trends: vec![], // Would calculate from historical data
            mean_time_to_resolution,
            false_positive_rate: 0.05, // Would need more sophisticated analysis
        }
    }

    fn generate_recommendations(
        &self,
        performance: &PerformanceAnalysis,
        memory: &MemoryAnalysis,
        alerts: &AlertAnalysis,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        if performance.efficiency_score < 0.7 {
            recommendations.push(Recommendation {
                priority: "High".to_string(),
                category: "Performance".to_string(),
                title: "Optimize slow operations".to_string(),
                description: "Several operations are performing below optimal levels".to_string(),
                potential_impact: "20-40% performance improvement".to_string(),
                implementation_effort: "Medium".to_string(),
            });
        }

        // Memory recommendations
        if memory.fragmentation_ratio > 0.2 {
            recommendations.push(Recommendation {
                priority: "Medium".to_string(),
                category: "Memory".to_string(),
                title: "Address memory fragmentation".to_string(),
                description: "High memory fragmentation detected".to_string(),
                potential_impact: "10-20% memory efficiency improvement".to_string(),
                implementation_effort: "Low".to_string(),
            });
        }

        // Alert recommendations
        if alerts.total_alerts > 100 {
            recommendations.push(Recommendation {
                priority: "Medium".to_string(),
                category: "Alerts".to_string(),
                title: "Review alert thresholds".to_string(),
                description: "High number of alerts may indicate threshold tuning needed"
                    .to_string(),
                potential_impact: "Reduced noise and better signal detection".to_string(),
                implementation_effort: "Low".to_string(),
            });
        }

        recommendations
    }

    fn generate_charts(
        &self,
        events: &[ProfileEvent],
        performance: &PerformanceAnalysis,
    ) -> Vec<ChartData> {
        let mut charts = Vec::new();

        if self.config.include_charts {
            // Performance trend chart
            charts.push(ChartData {
                chart_type: "line".to_string(),
                title: "Performance Trend".to_string(),
                x_axis: "Time".to_string(),
                y_axis: "Duration (ns)".to_string(),
                data: performance
                    .performance_trends
                    .iter()
                    .map(|t| ChartPoint {
                        x: t.timestamp
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64(),
                        y: t.value,
                        label: Some(t.label.clone()),
                    })
                    .collect(),
            });

            // Operation frequency chart
            charts.push(ChartData {
                chart_type: "bar".to_string(),
                title: "Most Frequent Operations".to_string(),
                x_axis: "Operation".to_string(),
                y_axis: "Count".to_string(),
                data: performance
                    .most_frequent_operations
                    .iter()
                    .enumerate()
                    .map(|(i, op)| ChartPoint {
                        x: i as f64,
                        y: op.count as f64,
                        label: Some(op.name.clone()),
                    })
                    .collect(),
            });
        }

        charts
    }

    fn calculate_performance_trends(&self, events: &[ProfileEvent]) -> Vec<TrendData> {
        let mut trends = Vec::new();

        // Group events by time windows (e.g., per minute)
        let mut time_buckets: HashMap<u64, Vec<u64>> = HashMap::new();

        for event in events {
            let event_time = SystemTime::UNIX_EPOCH + Duration::from_micros(event.start_us);
            let bucket = event_time
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs()
                / 60; // 1-minute buckets
            time_buckets
                .entry(bucket)
                .or_default()
                .push(event.duration_us * 1000); // Convert to ns
        }

        for (bucket, durations) in time_buckets {
            let avg_duration = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
            let timestamp = SystemTime::UNIX_EPOCH + Duration::from_secs(bucket * 60);

            trends.push(TrendData {
                timestamp,
                value: avg_duration,
                label: "Average Duration".to_string(),
            });
        }

        trends.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        trends
    }

    fn calculate_memory_trends(&self, events: &[ProfileEvent]) -> Vec<TrendData> {
        let mut trends = Vec::new();

        for event in events {
            if let Some(bytes) = event.bytes_transferred {
                trends.push(TrendData {
                    timestamp: SystemTime::UNIX_EPOCH + Duration::from_micros(event.start_us),
                    value: bytes as f64,
                    label: "Memory Usage".to_string(),
                });
            }
        }

        trends
    }

    fn identify_bottlenecks(&self, operations: &[OperationSummary]) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        for op in operations {
            // Identify operations with high variance (inconsistent performance)
            if op.std_deviation_ns > op.average_duration_ns / 2 {
                bottlenecks.push(BottleneckInfo {
                    operation: op.name.clone(),
                    bottleneck_type: "Performance Variance".to_string(),
                    severity: "Medium".to_string(),
                    impact_percentage: 15.0,
                    description: "Operation shows high performance variance".to_string(),
                    recommendations: vec!["Investigate resource contention".to_string()],
                });
            }

            // Identify extremely slow operations
            if op.average_duration_ns > 1_000_000_000 {
                // > 1 second
                bottlenecks.push(BottleneckInfo {
                    operation: op.name.clone(),
                    bottleneck_type: "Slow Operation".to_string(),
                    severity: "High".to_string(),
                    impact_percentage: 30.0,
                    description: "Operation is extremely slow".to_string(),
                    recommendations: vec!["Profile and optimize this operation".to_string()],
                });
            }
        }

        bottlenecks
    }

    fn calculate_efficiency_score(&self, operations: &[OperationSummary]) -> f64 {
        if operations.is_empty() {
            return 1.0;
        }

        let total_time: u64 = operations.iter().map(|op| op.total_duration_ns).sum();
        let efficient_time: u64 = operations
            .iter()
            .map(|op| op.min_duration_ns * op.count) // Theoretical optimal time
            .sum();

        if total_time > 0 {
            efficient_time as f64 / total_time as f64
        } else {
            1.0
        }
    }

    /// Export report to specified format
    pub fn export_report(&self, report: &PerformanceReport) -> Result<String> {
        match self.config.format {
            ReportFormat::Json => self.export_json(report),
            ReportFormat::Html => self.export_html(report),
            ReportFormat::Csv => self.export_csv(report),
            ReportFormat::Markdown => self.export_markdown(report),
            _ => Err(anyhow::anyhow!(
                "Unsupported format: {:?}",
                self.config.format
            )),
        }
    }

    fn export_json(&self, report: &PerformanceReport) -> Result<String> {
        let json = serde_json::to_string_pretty(report)?;
        fs::write(&self.config.output_path, &json)?;
        Ok(json)
    }

    fn export_html(&self, report: &PerformanceReport) -> Result<String> {
        let html = self.template_engine.render_html_report(report);
        fs::write(&self.config.output_path, &html)?;
        Ok(html)
    }

    fn export_csv(&self, report: &PerformanceReport) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("operation,count,avg_duration_ns,min_duration_ns,max_duration_ns\n");

        for op in &report.performance_analysis.slowest_operations {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                op.name, op.count, op.average_duration_ns, op.min_duration_ns, op.max_duration_ns
            ));
        }

        fs::write(&self.config.output_path, &csv)?;
        Ok(csv)
    }

    fn export_markdown(&self, report: &PerformanceReport) -> Result<String> {
        let markdown = self.template_engine.render_markdown_report(report);
        fs::write(&self.config.output_path, &markdown)?;
        Ok(markdown)
    }
}

/// Template engine for generating reports
pub struct TemplateEngine {
    templates: HashMap<String, String>,
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateEngine {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn load_template(&mut self, name: &str, template: String) {
        self.templates.insert(name.to_string(), template);
    }

    pub fn render_html_report(&self, report: &PerformanceReport) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Performance Report - {}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e7f3ff; padding: 15px; border-radius: 5px; }}
        .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Performance Report: {}</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Operations:</strong> {}</p>
        <p><strong>Average Duration:</strong> {:.2} ms</p>
        <p><strong>Peak Memory:</strong> {} MB</p>
        <p><strong>Efficiency Score:</strong> {:.2}%</p>
    </div>
    
    <h2>Slowest Operations</h2>
    <table>
        <tr><th>Operation</th><th>Count</th><th>Avg Duration (ms)</th><th>Max Duration (ms)</th></tr>
        {}
    </table>
    
    <h2>Recommendations</h2>
    {}
    
    <p><em>Generated at: {:?}</em></p>
</body>
</html>
"#,
            report.metadata.name,
            report.metadata.name,
            report.summary.total_operations,
            report.summary.average_duration_ns as f64 / 1_000_000.0,
            report.summary.peak_memory_bytes / 1_048_576,
            report.performance_analysis.efficiency_score * 100.0,
            report
                .performance_analysis
                .slowest_operations
                .iter()
                .map(|op| format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.2}</td></tr>",
                    op.name,
                    op.count,
                    op.average_duration_ns as f64 / 1_000_000.0,
                    op.max_duration_ns as f64 / 1_000_000.0
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            report
                .recommendations
                .iter()
                .map(|rec| format!(
                    "<div class=\"recommendation\"><strong>{}:</strong> {}</div>",
                    rec.title, rec.description
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            report.metadata.generated_at
        )
    }

    pub fn render_markdown_report(&self, report: &PerformanceReport) -> String {
        format!(
            r#"# Performance Report: {}

## Summary

- **Total Operations:** {}
- **Average Duration:** {:.2} ms
- **Peak Memory:** {} MB
- **Efficiency Score:** {:.2}%

## Slowest Operations

| Operation | Count | Avg Duration (ms) | Max Duration (ms) |
|-----------|-------|-------------------|-------------------|
{}

## Recommendations

{}

---
*Generated at: {:?}*
"#,
            report.metadata.name,
            report.summary.total_operations,
            report.summary.average_duration_ns as f64 / 1_000_000.0,
            report.summary.peak_memory_bytes / 1_048_576,
            report.performance_analysis.efficiency_score * 100.0,
            report
                .performance_analysis
                .slowest_operations
                .iter()
                .map(|op| format!(
                    "| {} | {} | {:.2} | {:.2} |",
                    op.name,
                    op.count,
                    op.average_duration_ns as f64 / 1_000_000.0,
                    op.max_duration_ns as f64 / 1_000_000.0
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            report
                .recommendations
                .iter()
                .map(|rec| format!("- **{}:** {}", rec.title, rec.description))
                .collect::<Vec<_>>()
                .join("\n"),
            report.metadata.generated_at
        )
    }
}

/// Convenience functions
/// Create a performance report configuration
pub fn create_performance_report_config(
    name: String,
    output_path: String,
    format: ReportFormat,
) -> ReportConfig {
    ReportConfig {
        name,
        description: "Performance analysis report".to_string(),
        report_type: ReportType::Performance,
        format,
        frequency: ReportFrequency::OnDemand,
        output_path,
        template_path: None,
        include_charts: true,
        include_raw_data: false,
        time_range: None,
        filters: vec![],
        recipients: vec![],
        enabled: true,
    }
}

/// Generate a quick performance report
pub fn generate_quick_report(
    events: &[ProfileEvent],
    alerts: &[Alert],
    output_path: String,
    format: ReportFormat,
) -> Result<PerformanceReport> {
    let config = create_performance_report_config(
        "Quick Performance Report".to_string(),
        output_path,
        format,
    );

    let generator = ReportGenerator::new(config);
    let report = generator.generate_report(events, alerts)?;
    generator.export_report(&report)?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_report_config_creation() {
        let config = create_performance_report_config(
            "Test Report".to_string(),
            "/tmp/test_report.html".to_string(),
            ReportFormat::Html,
        );

        assert_eq!(config.name, "Test Report");
        assert_eq!(config.format, ReportFormat::Html);
        assert_eq!(config.report_type, ReportType::Performance);
    }

    #[test]
    fn test_report_generation() {
        let events = vec![ProfileEvent {
            name: "test_operation".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1000,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(100),
            bytes_transferred: Some(1024),
            stack_trace: Some("test trace".to_string()),
        }];

        let alerts = vec![];

        let config = create_performance_report_config(
            "Test Report".to_string(),
            "/tmp/test_report.json".to_string(),
            ReportFormat::Json,
        );

        let generator = ReportGenerator::new(config);
        let report = generator.generate_report(&events, &alerts).unwrap();

        assert_eq!(report.summary.total_operations, 1);
        assert_eq!(report.summary.average_duration_ns, 1000000);
        assert!(!report.performance_analysis.slowest_operations.is_empty());
    }

    #[test]
    fn test_html_template_rendering() {
        let template_engine = TemplateEngine::new();

        let report = PerformanceReport {
            metadata: ReportMetadata {
                id: "test".to_string(),
                name: "Test Report".to_string(),
                generated_at: SystemTime::now(),
                time_range: TimeRange {
                    start: SystemTime::now(),
                    end: SystemTime::now(),
                    duration: Duration::from_secs(1),
                },
                total_events: 1,
                report_type: ReportType::Performance,
                format: ReportFormat::Html,
                version: "1.0.0".to_string(),
            },
            summary: ReportSummary {
                total_operations: 10,
                total_duration_ns: 10000000,
                average_duration_ns: 1000000,
                min_duration_ns: 500000,
                max_duration_ns: 2000000,
                total_memory_bytes: 1024,
                peak_memory_bytes: 2048,
                total_flops: 1000,
                operations_per_second: 100.0,
                gflops_per_second: 0.1,
                throughput_mbps: 1.0,
            },
            performance_analysis: PerformanceAnalysis {
                slowest_operations: vec![],
                fastest_operations: vec![],
                most_frequent_operations: vec![],
                performance_trends: vec![],
                bottlenecks: vec![],
                efficiency_score: 0.85,
            },
            memory_analysis: MemoryAnalysis {
                peak_usage: 2048,
                average_usage: 1024,
                allocation_rate: 10.0,
                deallocation_rate: 8.0,
                fragmentation_ratio: 0.1,
                memory_leaks: vec![],
                memory_trends: vec![],
            },
            alert_analysis: AlertAnalysis {
                total_alerts: 5,
                alerts_by_severity: HashMap::new(),
                alerts_by_operation: HashMap::new(),
                alert_trends: vec![],
                mean_time_to_resolution: Duration::from_secs(300),
                false_positive_rate: 0.05,
            },
            regression_analysis: None,
            recommendations: vec![],
            charts: vec![],
            raw_data: None,
        };

        let html = template_engine.render_html_report(&report);
        assert!(html.contains("Test Report"));
        assert!(html.contains("Total Operations"));
        assert!(html.contains("85.00%"));
    }

    #[test]
    fn test_filter_application() {
        let events = vec![
            ProfileEvent {
                name: "fast_op".to_string(),
                category: "test".to_string(),
                start_us: 0,
                duration_us: 500,
                thread_id: 0,
                operation_count: Some(1),
                flops: Some(10),
                bytes_transferred: Some(100),
                stack_trace: Some("test trace".to_string()),
            },
            ProfileEvent {
                name: "slow_op".to_string(),
                category: "test".to_string(),
                start_us: 0,
                duration_us: 2000,
                thread_id: 0,
                operation_count: Some(1),
                flops: Some(10),
                bytes_transferred: Some(100),
                stack_trace: Some("test trace".to_string()),
            },
        ];

        let mut config = create_performance_report_config(
            "Filtered Report".to_string(),
            "/tmp/filtered_report.json".to_string(),
            ReportFormat::Json,
        );

        config.filters = vec![ReportFilter::MinDuration(1000000)];

        let generator = ReportGenerator::new(config);
        let filtered_events = generator.apply_filters(&events);

        assert_eq!(filtered_events.len(), 1);
        assert_eq!(filtered_events[0].name, "slow_op");
    }
}
