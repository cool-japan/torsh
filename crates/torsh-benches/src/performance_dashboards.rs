//! Performance Dashboards for ToRSh Benchmarks
//!
//! This module provides comprehensive performance monitoring and dashboard capabilities
//! for tracking ToRSh benchmark performance over time, detecting regressions,
//! and providing real-time insights into performance metrics.

use crate::BenchResult;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Performance metric point for time-series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePoint {
    /// Timestamp when the metric was recorded
    pub timestamp: DateTime<Utc>,
    /// Benchmark name
    pub benchmark_name: String,
    /// Input size
    pub size: usize,
    /// Data type
    pub dtype: String,
    /// Mean execution time in nanoseconds
    pub mean_time_ns: f64,
    /// Standard deviation of execution time
    pub std_dev_ns: f64,
    /// Throughput (operations per second)
    pub throughput: Option<f64>,
    /// Memory usage in bytes
    pub memory_usage: Option<usize>,
    /// Peak memory usage in bytes
    pub peak_memory: Option<usize>,
    /// Git commit hash (optional)
    pub git_commit: Option<String>,
    /// Build configuration (debug/release)
    pub build_config: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl PerformancePoint {
    /// Create a new performance point from a benchmark result
    pub fn from_result(result: &BenchResult, metadata: Option<HashMap<String, String>>) -> Self {
        Self {
            timestamp: Utc::now(),
            benchmark_name: result.name.clone(),
            size: result.size,
            dtype: format!("{:?}", result.dtype),
            mean_time_ns: result.mean_time_ns,
            std_dev_ns: result.std_dev_ns,
            throughput: result.throughput,
            memory_usage: result.memory_usage,
            peak_memory: result.peak_memory,
            git_commit: get_git_commit(),
            build_config: if cfg!(debug_assertions) {
                "debug".to_string()
            } else {
                "release".to_string()
            },
            metadata: metadata.unwrap_or_default(),
        }
    }

    /// Get performance efficiency score (0-100)
    pub fn efficiency_score(&self) -> f64 {
        let base_score = 50.0; // Baseline score
        let time_factor = 1.0 / (self.mean_time_ns / 1_000_000.0); // Convert to ms and invert
        let throughput_factor = self.throughput.unwrap_or(1.0) / 1000.0; // Normalize throughput

        let score = base_score + (time_factor * 10.0) + (throughput_factor * 10.0);
        score.min(100.0).max(0.0)
    }
}

/// Performance regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetection {
    /// Benchmark name
    pub benchmark_name: String,
    /// Whether a regression was detected
    pub is_regression: bool,
    /// Percentage change in performance (negative = regression)
    pub performance_change: f64,
    /// Confidence level (0-100)
    pub confidence: f64,
    /// Baseline performance point
    pub baseline: PerformancePoint,
    /// Current performance point
    pub current: PerformancePoint,
    /// Regression severity
    pub severity: RegressionSeverity,
}

/// Severity levels for performance regressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// Less than 5% performance degradation
    Minor,
    /// 5-15% performance degradation
    Moderate,
    /// 15-30% performance degradation
    Major,
    /// More than 30% performance degradation
    Critical,
}

impl RegressionSeverity {
    /// Get severity from performance change percentage
    pub fn from_change(change: f64) -> Self {
        let abs_change = change.abs();
        if abs_change < 5.0 {
            Self::Minor
        } else if abs_change < 15.0 {
            Self::Moderate
        } else if abs_change < 30.0 {
            Self::Major
        } else {
            Self::Critical
        }
    }
}

/// Performance dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Maximum number of data points to store per benchmark
    pub max_history_points: usize,
    /// Regression detection threshold (percentage)
    pub regression_threshold: f64,
    /// Number of recent points to compare for regression detection
    pub regression_window: usize,
    /// Enable real-time monitoring
    pub enable_real_time: bool,
    /// Dashboard refresh interval in seconds
    pub refresh_interval: u64,
    /// Database file path for persistent storage
    pub database_path: String,
    /// Enable email alerts for regressions
    pub enable_alerts: bool,
    /// Email configuration for alerts
    pub alert_config: Option<AlertConfig>,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            max_history_points: 1000,
            regression_threshold: 10.0,
            regression_window: 5,
            enable_real_time: true,
            refresh_interval: 30,
            database_path: "performance_history.db".to_string(),
            enable_alerts: false,
            alert_config: None,
        }
    }
}

/// Alert configuration for performance regressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// SMTP server address
    pub smtp_server: String,
    /// SMTP port
    pub smtp_port: u16,
    /// Email username
    pub username: String,
    /// Email password
    pub password: String,
    /// Recipients for alerts
    pub recipients: Vec<String>,
    /// Subject prefix for alert emails
    pub subject_prefix: String,
}

/// Performance dashboard for monitoring benchmarks
pub struct PerformanceDashboard {
    /// Configuration
    config: DashboardConfig,
    /// Performance history storage
    history: HashMap<String, Vec<PerformancePoint>>,
    /// Regression detection results
    regressions: Vec<RegressionDetection>,
    /// Dashboard metrics
    metrics: DashboardMetrics,
}

/// Dashboard metrics and statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DashboardMetrics {
    /// Total benchmarks tracked
    pub total_benchmarks: usize,
    /// Total performance points recorded
    pub total_points: usize,
    /// Number of active regressions
    pub active_regressions: usize,
    /// Average performance trend (positive = improving)
    pub performance_trend: f64,
    /// System health score (0-100)
    pub health_score: f64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

impl PerformanceDashboard {
    /// Create a new performance dashboard
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
            regressions: Vec::new(),
            metrics: DashboardMetrics::default(),
        }
    }

    /// Create dashboard with default configuration
    pub fn default() -> Self {
        Self::new(DashboardConfig::default())
    }

    /// Add a performance point to the dashboard
    pub fn add_point(&mut self, point: PerformancePoint) {
        let key = format!(
            "{}_{}_{}_{}",
            point.benchmark_name, point.size, point.dtype, point.build_config
        );

        let points = self.history.entry(key.clone()).or_insert_with(Vec::new);
        points.push(point.clone());

        // Limit history size
        if points.len() > self.config.max_history_points {
            points.remove(0);
        }

        // Update metrics
        self.update_metrics();

        // Check for regressions
        if let Some(regression) = self.detect_regression(&key, &point) {
            self.regressions.push(regression.clone());

            // Send alert if enabled
            if self.config.enable_alerts {
                self.send_alert(&regression);
            }
        }
    }

    /// Add multiple performance points from benchmark results
    pub fn add_results(
        &mut self,
        results: &[BenchResult],
        metadata: Option<HashMap<String, String>>,
    ) {
        for result in results {
            let point = PerformancePoint::from_result(result, metadata.clone());
            self.add_point(point);
        }
    }

    /// Detect performance regression
    fn detect_regression(
        &self,
        key: &str,
        current: &PerformancePoint,
    ) -> Option<RegressionDetection> {
        let points = self.history.get(key)?;

        if points.len() < self.config.regression_window + 1 {
            return None;
        }

        // Get baseline (average of previous N points)
        let baseline_points =
            &points[points.len() - self.config.regression_window - 1..points.len() - 1];
        let baseline_time = baseline_points.iter().map(|p| p.mean_time_ns).sum::<f64>()
            / baseline_points.len() as f64;

        // Calculate performance change
        let change = ((current.mean_time_ns - baseline_time) / baseline_time) * 100.0;

        // Check if regression exceeds threshold
        if change > self.config.regression_threshold {
            let confidence = calculate_confidence(baseline_points, current);
            let severity = RegressionSeverity::from_change(change);

            Some(RegressionDetection {
                benchmark_name: current.benchmark_name.clone(),
                is_regression: true,
                performance_change: change,
                confidence,
                baseline: baseline_points.last().unwrap().clone(),
                current: current.clone(),
                severity,
            })
        } else {
            None
        }
    }

    /// Update dashboard metrics
    fn update_metrics(&mut self) {
        self.metrics.total_benchmarks = self.history.len();
        self.metrics.total_points = self.history.values().map(|v| v.len()).sum();
        self.metrics.active_regressions = self.regressions.len();
        self.metrics.last_update = Utc::now();

        // Calculate performance trend
        let mut trend_sum = 0.0;
        let mut trend_count = 0;

        for points in self.history.values() {
            if points.len() >= 2 {
                let recent = &points[points.len().saturating_sub(5)..];
                let trend = calculate_trend(recent);
                trend_sum += trend;
                trend_count += 1;
            }
        }

        self.metrics.performance_trend = if trend_count > 0 {
            trend_sum / trend_count as f64
        } else {
            0.0
        };

        // Calculate health score
        self.metrics.health_score = self.calculate_health_score();
    }

    /// Calculate overall system health score
    fn calculate_health_score(&self) -> f64 {
        let mut score = 100.0;

        // Penalize for active regressions
        let regression_penalty = self.metrics.active_regressions as f64 * 10.0;
        score -= regression_penalty;

        // Reward for positive performance trends
        if self.metrics.performance_trend > 0.0 {
            score += self.metrics.performance_trend * 5.0;
        } else {
            score += self.metrics.performance_trend * 2.0;
        }

        score.min(100.0).max(0.0)
    }

    /// Send alert for performance regression
    fn send_alert(&self, regression: &RegressionDetection) {
        if let Some(_alert_config) = &self.config.alert_config {
            // In a real implementation, this would send an email
            eprintln!(
                "PERFORMANCE ALERT: Regression detected in {}: {:.1}% degradation",
                regression.benchmark_name, regression.performance_change
            );
        }
    }

    /// Get performance history for a specific benchmark
    pub fn get_history(
        &self,
        benchmark_name: &str,
        size: usize,
        dtype: &str,
    ) -> Option<&[PerformancePoint]> {
        let key = format!(
            "{}_{}_{}_{}",
            benchmark_name,
            size,
            dtype,
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        );
        self.history.get(&key).map(|v| v.as_slice())
    }

    /// Get all active regressions
    pub fn get_regressions(&self) -> &[RegressionDetection] {
        &self.regressions
    }

    /// Get dashboard metrics
    pub fn get_metrics(&self) -> &DashboardMetrics {
        &self.metrics
    }

    /// Clear old regressions
    pub fn clear_old_regressions(&mut self, age_threshold: Duration) {
        let cutoff = Utc::now()
            - chrono::Duration::from_std(age_threshold).unwrap_or(chrono::Duration::hours(24));
        self.regressions.retain(|r| r.current.timestamp > cutoff);
    }

    /// Export dashboard data as JSON
    pub fn export_json(&self, path: &str) -> std::io::Result<()> {
        let data = serde_json::json!({
            "metrics": self.metrics,
            "regressions": self.regressions,
            "history_summary": self.get_history_summary()
        });

        std::fs::write(path, serde_json::to_string_pretty(&data)?)?;
        Ok(())
    }

    /// Get summary of performance history
    pub fn get_history_summary(&self) -> HashMap<String, usize> {
        self.history
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }

    /// Generate performance dashboard HTML
    pub fn generate_dashboard_html(&self, output_path: &str) -> std::io::Result<()> {
        let html = self.create_dashboard_html();
        std::fs::write(output_path, html)?;
        Ok(())
    }

    /// Create HTML dashboard content
    fn create_dashboard_html(&self) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ToRSh Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .regression-alert {{ background: #e74c3c; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .regression-item {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #e74c3c; margin-bottom: 10px; }}
        .health-score {{ font-size: 3em; font-weight: bold; text-align: center; }}
        .health-excellent {{ color: #27ae60; }}
        .health-good {{ color: #f39c12; }}
        .health-poor {{ color: #e74c3c; }}
        .refresh-info {{ text-align: center; color: #7f8c8d; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 ToRSh Performance Dashboard</h1>
            <p>Real-time performance monitoring and regression detection</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Total Benchmarks</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Performance Points</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Active Regressions</div>
            </div>
            <div class="metric-card">
                <div class="health-score {}">{}%</div>
                <div class="metric-label">System Health</div>
            </div>
        </div>

        {}

        <div class="chart-container">
            <h2>Performance Trends</h2>
            <canvas id="performanceChart" width="400" height="200"></canvas>
        </div>

        <div class="refresh-info">
            <p>Last updated: {} | Auto-refresh: {}s</p>
        </div>
    </div>

    <script>
        // Initialize charts
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: [],
                datasets: [{{
                    label: 'Performance Trend',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.1
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Performance Score'
                        }}
                    }}
                }}
            }}
        }});

        // Auto-refresh functionality
        setInterval(() => {{
            location.reload();
        }}, {} * 1000);
    </script>
</body>
</html>
        "#,
            self.metrics.total_benchmarks,
            self.metrics.total_points,
            self.metrics.active_regressions,
            self.get_health_class(self.metrics.health_score),
            self.metrics.health_score as u32,
            self.format_regressions_html(),
            self.metrics.last_update.format("%Y-%m-%d %H:%M:%S UTC"),
            self.config.refresh_interval,
            self.config.refresh_interval
        )
    }

    /// Get CSS class for health score
    fn get_health_class(&self, score: f64) -> &'static str {
        if score >= 80.0 {
            "health-excellent"
        } else if score >= 60.0 {
            "health-good"
        } else {
            "health-poor"
        }
    }

    /// Format regressions for HTML display
    fn format_regressions_html(&self) -> String {
        if self.regressions.is_empty() {
            return String::new();
        }

        let mut html = String::from(
            r#"<div class="regression-alert">
            <h3>⚠️ Performance Regressions Detected</h3>
        </div>"#,
        );

        for regression in &self.regressions {
            html.push_str(&format!(
                r#"
                <div class="regression-item">
                    <strong>{}</strong> - {:.1}% degradation ({:?})
                    <br>
                    <small>Size: {}, Type: {}, Confidence: {:.1}%</small>
                </div>
            "#,
                regression.benchmark_name,
                regression.performance_change,
                regression.severity,
                regression.current.size,
                regression.current.dtype,
                regression.confidence
            ));
        }

        html
    }
}

/// Calculate statistical confidence for regression detection
fn calculate_confidence(baseline_points: &[PerformancePoint], current: &PerformancePoint) -> f64 {
    if baseline_points.is_empty() {
        return 0.0;
    }

    let baseline_mean =
        baseline_points.iter().map(|p| p.mean_time_ns).sum::<f64>() / baseline_points.len() as f64;
    let baseline_std = {
        let variance = baseline_points
            .iter()
            .map(|p| (p.mean_time_ns - baseline_mean).powi(2))
            .sum::<f64>()
            / baseline_points.len() as f64;
        variance.sqrt()
    };

    // Simple confidence calculation based on standard deviations
    let z_score = (current.mean_time_ns - baseline_mean).abs() / baseline_std;
    let confidence = (1.0 - (-z_score.powi(2) / 2.0).exp()) * 100.0;

    confidence.min(99.0).max(0.0)
}

/// Calculate performance trend from time series data
fn calculate_trend(points: &[PerformancePoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    let first = points.first().unwrap();
    let last = points.last().unwrap();

    // Negative trend means performance is getting worse (higher execution time)
    ((first.mean_time_ns - last.mean_time_ns) / first.mean_time_ns) * 100.0
}

/// Get git commit hash
fn get_git_commit() -> Option<String> {
    // In a real implementation, this would use git2 or similar
    // For now, return a placeholder
    Some("unknown".to_string())
}

/// Performance dashboard builder for easy configuration
pub struct DashboardBuilder {
    config: DashboardConfig,
}

impl DashboardBuilder {
    /// Create a new dashboard builder
    pub fn new() -> Self {
        Self {
            config: DashboardConfig::default(),
        }
    }

    /// Set maximum history points
    pub fn max_history_points(mut self, max: usize) -> Self {
        self.config.max_history_points = max;
        self
    }

    /// Set regression threshold
    pub fn regression_threshold(mut self, threshold: f64) -> Self {
        self.config.regression_threshold = threshold;
        self
    }

    /// Enable real-time monitoring
    pub fn enable_real_time(mut self, enabled: bool) -> Self {
        self.config.enable_real_time = enabled;
        self
    }

    /// Set refresh interval
    pub fn refresh_interval(mut self, interval: u64) -> Self {
        self.config.refresh_interval = interval;
        self
    }

    /// Set database path
    pub fn database_path(mut self, path: &str) -> Self {
        self.config.database_path = path.to_string();
        self
    }

    /// Enable email alerts
    pub fn enable_alerts(mut self, config: AlertConfig) -> Self {
        self.config.enable_alerts = true;
        self.config.alert_config = Some(config);
        self
    }

    /// Build the dashboard
    pub fn build(self) -> PerformanceDashboard {
        PerformanceDashboard::new(self.config)
    }
}

impl Default for DashboardBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchResult;

    #[test]
    fn test_dashboard_creation() {
        let dashboard = PerformanceDashboard::default();
        assert_eq!(dashboard.metrics.total_benchmarks, 0);
        assert_eq!(dashboard.metrics.total_points, 0);
    }

    #[test]
    fn test_add_performance_point() {
        let mut dashboard = PerformanceDashboard::default();
        let result = BenchResult {
            name: "test_benchmark".to_string(),
            size: 1024,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: 1000.0,
            std_dev_ns: 100.0,
            throughput: Some(1000.0),
            memory_usage: Some(1024),
            peak_memory: Some(2048),
            metrics: HashMap::new(),
        };

        let point = PerformancePoint::from_result(&result, None);
        dashboard.add_point(point);

        assert_eq!(dashboard.metrics.total_benchmarks, 1);
        assert_eq!(dashboard.metrics.total_points, 1);
    }

    #[test]
    fn test_regression_detection() {
        let mut dashboard = PerformanceDashboard::default();
        let mut config = DashboardConfig::default();
        config.regression_threshold = 5.0;
        config.regression_window = 2;
        dashboard.config = config;

        // Add baseline points
        for _i in 0..3 {
            let result = BenchResult {
                name: "test_benchmark".to_string(),
                size: 1024,
                dtype: torsh_core::dtype::DType::F32,
                mean_time_ns: 1000.0,
                std_dev_ns: 100.0,
                throughput: Some(1000.0),
                memory_usage: Some(1024),
                peak_memory: Some(2048),
                metrics: HashMap::new(),
            };
            let point = PerformancePoint::from_result(&result, None);
            dashboard.add_point(point);
        }

        // Add regression point
        let regression_result = BenchResult {
            name: "test_benchmark".to_string(),
            size: 1024,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: 1200.0, // 20% slower
            std_dev_ns: 100.0,
            throughput: Some(800.0),
            memory_usage: Some(1024),
            peak_memory: Some(2048),
            metrics: HashMap::new(),
        };
        let regression_point = PerformancePoint::from_result(&regression_result, None);
        dashboard.add_point(regression_point);

        assert!(!dashboard.regressions.is_empty());
        assert!(dashboard.regressions[0].is_regression);
        assert!(dashboard.regressions[0].performance_change > 5.0);
    }

    #[test]
    fn test_dashboard_builder() {
        let dashboard = DashboardBuilder::new()
            .max_history_points(500)
            .regression_threshold(15.0)
            .enable_real_time(false)
            .build();

        assert_eq!(dashboard.config.max_history_points, 500);
        assert_eq!(dashboard.config.regression_threshold, 15.0);
        assert!(!dashboard.config.enable_real_time);
    }

    #[test]
    fn test_efficiency_score() {
        let result = BenchResult {
            name: "test_benchmark".to_string(),
            size: 1024,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: 1000.0,
            std_dev_ns: 100.0,
            throughput: Some(1000.0),
            memory_usage: Some(1024),
            peak_memory: Some(2048),
            metrics: HashMap::new(),
        };

        let point = PerformancePoint::from_result(&result, None);
        let score = point.efficiency_score();
        assert!(score >= 0.0 && score <= 100.0);
    }
}
