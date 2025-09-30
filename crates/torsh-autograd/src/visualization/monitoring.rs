//! # Gradient Flow Monitoring and Trend Analysis
//!
//! This module provides real-time monitoring and historical trend analysis capabilities
//! for gradient flows in neural network training. It tracks performance over time and
//! identifies patterns, regressions, and improvements.
//!
//! ## Key Components
//!
//! - **GradientFlowMonitor**: Real-time monitoring with historical analysis storage
//! - **GradientTrendAnalysis**: Time-series analysis of gradient flow metrics
//! - **PerformanceTracker**: Detailed performance metric tracking over time
//! - **AlertingSystem**: Automated detection of performance anomalies
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_autograd::visualization::monitoring::GradientFlowMonitor;
//! use torsh_autograd::context::AutogradContext;
//!
//! let mut monitor = GradientFlowMonitor::new();
//! let ctx = AutogradContext::new();
//!
//! // Monitor gradient flow during training
//! for epoch in 0..100 {
//!     // ... training step ...
//!     monitor.analyze_and_store(&ctx)?;
//!
//!     // Check for trends every 10 epochs
//!     if epoch % 10 == 0 {
//!         if let Some(trend) = monitor.get_trend_analysis() {
//!             println!("Trend analysis: {:?}", trend);
//!         }
//!     }
//! }
//!
//! // Generate monitoring report
//! let report = monitor.generate_monitoring_report()?;
//! ```

use super::core::GradientFlowAnalysis;
use super::visualizer::GradientVisualizer;
use crate::context::AutogradContext;
use std::collections::VecDeque;
use std::fmt::Write;
use std::time::{Duration, SystemTime};
use torsh_core::error::{Result, TorshError};
use tracing::{debug, info, warn};

/// Real-time gradient flow monitor with historical analysis and trend detection
///
/// The `GradientFlowMonitor` continuously tracks gradient flow characteristics
/// during training, storing historical data and providing trend analysis to
/// detect performance improvements, regressions, and anomalies.
#[derive(Debug)]
pub struct GradientFlowMonitor {
    /// Visualizer for performing gradient flow analysis
    visualizer: GradientVisualizer,
    /// History of gradient flow analyses
    analysis_history: VecDeque<TimestampedAnalysis>,
    /// Whether monitoring is currently enabled
    monitoring_enabled: bool,
    /// Configuration for monitoring behavior
    config: MonitoringConfig,
    /// Performance tracker for detailed metrics
    performance_tracker: PerformanceTracker,
    /// Alerting system for anomaly detection
    alerting_system: AlertingSystem,
}

/// Configuration for gradient flow monitoring
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Maximum number of analyses to keep in history
    pub max_history_size: usize,
    /// Minimum number of analyses required for trend detection
    pub min_trend_samples: usize,
    /// Time window for trend analysis (in analyses)
    pub trend_window_size: usize,
    /// Threshold for detecting significant changes (percentage)
    pub change_threshold: f64,
    /// Whether to enable automatic alerting
    pub enable_alerting: bool,
    /// Interval between performance snapshots
    pub snapshot_interval: Duration,
}

/// Timestamped gradient flow analysis for historical tracking
#[derive(Debug, Clone)]
struct TimestampedAnalysis {
    /// Timestamp when the analysis was performed
    timestamp: SystemTime,
    /// The gradient flow analysis results
    analysis: GradientFlowAnalysis,
    /// Training epoch or step number (if available)
    epoch: Option<usize>,
    /// Additional metadata
    metadata: AnalysisMetadata,
}

/// Additional metadata for analysis tracking
#[derive(Debug, Clone, Default)]
struct AnalysisMetadata {
    /// Learning rate at time of analysis
    learning_rate: Option<f32>,
    /// Batch size used
    batch_size: Option<usize>,
    /// Current loss value
    loss_value: Option<f32>,
    /// Custom tags for categorization
    tags: Vec<String>,
}

/// Comprehensive trend analysis of gradient flows over time
#[derive(Debug, Clone)]
pub struct GradientTrendAnalysis {
    /// Trend in number of operations (positive = increasing)
    pub operations_trend: TrendDirection,
    /// Trend in memory usage (positive = increasing)
    pub memory_trend: TrendDirection,
    /// Trend in number of bottlenecks (positive = increasing)
    pub bottlenecks_trend: TrendDirection,
    /// Trend in gradient magnitude (positive = increasing)
    pub gradient_magnitude_trend: TrendDirection,
    /// Overall gradient health trend
    pub health_trend: TrendDirection,
    /// Time period covered by this analysis
    pub time_period: Duration,
    /// Number of data points used
    pub sample_count: usize,
    /// Confidence in trend analysis (0-1)
    pub confidence: f64,
    /// Detected anomalies
    pub anomalies: Vec<AnomalyReport>,
}

/// Direction and magnitude of a trend
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Strongly increasing (> 10% change)
    StronglyIncreasing(f64),
    /// Increasing (5-10% change)
    Increasing(f64),
    /// Stable (< 5% change)
    Stable(f64),
    /// Decreasing (-10 to -5% change)
    Decreasing(f64),
    /// Strongly decreasing (< -10% change)
    StronglyDecreasing(f64),
    /// Insufficient data for trend analysis
    Unknown,
}

impl TrendDirection {
    /// Create trend direction from percentage change
    fn from_percentage_change(change: f64) -> Self {
        match change {
            c if c > 10.0 => TrendDirection::StronglyIncreasing(c),
            c if c > 5.0 => TrendDirection::Increasing(c),
            c if c > -5.0 => TrendDirection::Stable(c),
            c if c > -10.0 => TrendDirection::Decreasing(c),
            c => TrendDirection::StronglyDecreasing(c),
        }
    }

    /// Check if trend indicates improvement for the given metric
    pub fn is_improvement(&self, metric_type: MetricType) -> bool {
        match metric_type {
            MetricType::GradientHealth | MetricType::MemoryEfficiency => {
                matches!(
                    self,
                    TrendDirection::Increasing(_) | TrendDirection::StronglyIncreasing(_)
                )
            }
            MetricType::Bottlenecks | MetricType::MemoryUsage => {
                matches!(
                    self,
                    TrendDirection::Decreasing(_) | TrendDirection::StronglyDecreasing(_)
                )
            }
            MetricType::Operations => {
                matches!(self, TrendDirection::Stable(_))
            }
        }
    }

    /// Get the percentage change value
    pub fn change_percentage(&self) -> f64 {
        match self {
            TrendDirection::StronglyIncreasing(c) => *c,
            TrendDirection::Increasing(c) => *c,
            TrendDirection::Stable(c) => *c,
            TrendDirection::Decreasing(c) => *c,
            TrendDirection::StronglyDecreasing(c) => *c,
            TrendDirection::Unknown => 0.0,
        }
    }
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrendDirection::StronglyIncreasing(c) => write!(f, "Strongly Increasing ({:+.1}%)", c),
            TrendDirection::Increasing(c) => write!(f, "Increasing ({:+.1}%)", c),
            TrendDirection::Stable(c) => write!(f, "Stable ({:+.1}%)", c),
            TrendDirection::Decreasing(c) => write!(f, "Decreasing ({:+.1}%)", c),
            TrendDirection::StronglyDecreasing(c) => write!(f, "Strongly Decreasing ({:+.1}%)", c),
            TrendDirection::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Type of metric for trend analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricType {
    Operations,
    MemoryUsage,
    MemoryEfficiency,
    Bottlenecks,
    GradientHealth,
}

/// Performance tracker for detailed metrics over time
#[derive(Debug)]
struct PerformanceTracker {
    /// Time-series data for various metrics
    metrics_history: std::collections::HashMap<String, VecDeque<MetricPoint>>,
    /// Configuration for tracking
    config: TrackingConfig,
}

/// Configuration for performance tracking
#[derive(Debug, Clone)]
struct TrackingConfig {
    /// Maximum number of metric points to keep
    max_metric_points: usize,
    /// Metrics to track
    tracked_metrics: Vec<String>,
}

/// Individual metric data point
#[derive(Debug, Clone)]
struct MetricPoint {
    /// Timestamp of the measurement
    timestamp: SystemTime,
    /// Metric value
    value: f64,
    /// Optional additional context
    context: Option<String>,
}

/// Alerting system for anomaly detection
#[derive(Debug)]
struct AlertingSystem {
    /// Configuration for alerting
    config: AlertingConfig,
    /// History of generated alerts
    alert_history: VecDeque<Alert>,
}

/// Configuration for alerting system
#[derive(Debug, Clone)]
struct AlertingConfig {
    /// Enable gradient magnitude alerts
    pub enable_gradient_alerts: bool,
    /// Enable memory usage alerts
    pub enable_memory_alerts: bool,
    /// Enable bottleneck alerts
    pub enable_bottleneck_alerts: bool,
    /// Threshold for gradient magnitude alerts
    pub gradient_magnitude_threshold: f32,
    /// Threshold for memory usage alerts (MB)
    pub memory_threshold_mb: f64,
    /// Threshold for bottleneck count alerts
    pub bottleneck_count_threshold: usize,
}

/// Alert generated by the monitoring system
#[derive(Debug, Clone)]
pub struct Alert {
    /// Type of alert
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Timestamp when alert was generated
    pub timestamp: SystemTime,
    /// Associated analysis (if any)
    pub analysis_id: Option<usize>,
}

/// Type of monitoring alert
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    /// Gradient magnitude issues
    GradientMagnitude,
    /// Memory usage concerns
    MemoryUsage,
    /// Performance bottlenecks
    Bottlenecks,
    /// Overall performance regression
    PerformanceRegression,
    /// Training instability detected
    TrainingInstability,
}

/// Severity level for alerts
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational - no action needed
    Info,
    /// Warning - monitoring recommended
    Warning,
    /// Critical - immediate attention required
    Critical,
}

/// Anomaly report for unusual behavior
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Severity of the anomaly
    pub severity: f64,
    /// Description of the anomaly
    pub description: String,
    /// Timestamp when detected
    pub timestamp: SystemTime,
    /// Suggested actions
    pub suggestions: Vec<String>,
}

/// Type of detected anomaly
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    /// Sudden spike in gradient magnitudes
    GradientSpike,
    /// Sudden drop in gradient magnitudes
    GradientVanishing,
    /// Unexpected memory usage increase
    MemoryAnomaly,
    /// Performance degradation
    PerformanceDrop,
    /// Training oscillation
    Oscillation,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            min_trend_samples: 5,
            trend_window_size: 20,
            change_threshold: 5.0,
            enable_alerting: true,
            snapshot_interval: Duration::from_secs(60),
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enable_gradient_alerts: true,
            enable_memory_alerts: true,
            enable_bottleneck_alerts: true,
            gradient_magnitude_threshold: 10.0,
            memory_threshold_mb: 1000.0,
            bottleneck_count_threshold: 5,
        }
    }
}

impl GradientFlowMonitor {
    /// Create a new gradient flow monitor with default configuration
    pub fn new() -> Self {
        Self::with_config(MonitoringConfig::default())
    }

    /// Create a gradient flow monitor with custom configuration
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            visualizer: GradientVisualizer::new(),
            analysis_history: VecDeque::with_capacity(config.max_history_size),
            monitoring_enabled: true,
            performance_tracker: PerformanceTracker::new(),
            alerting_system: AlertingSystem::new(),
            config,
        }
    }

    /// Enable or disable gradient flow monitoring
    pub fn set_monitoring(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
        if enabled {
            info!("Gradient flow monitoring enabled");
        } else {
            info!("Gradient flow monitoring disabled");
        }
    }

    /// Check if monitoring is currently enabled
    pub fn is_monitoring_enabled(&self) -> bool {
        self.monitoring_enabled
    }

    /// Perform gradient flow analysis and store in history
    ///
    /// This method analyzes the current state of the autograd context and
    /// adds the results to the monitoring history for trend analysis.
    pub fn analyze_and_store(&mut self, ctx: &AutogradContext) -> Result<()> {
        self.analyze_and_store_with_metadata(ctx, None, AnalysisMetadata::default())
    }

    /// Perform analysis with additional metadata
    pub fn analyze_and_store_with_metadata(
        &mut self,
        ctx: &AutogradContext,
        epoch: Option<usize>,
        metadata: AnalysisMetadata,
    ) -> Result<()> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        debug!("Performing gradient flow analysis for monitoring");

        let analysis = self.visualizer.analyze_gradient_flow(ctx)?;
        let timestamped_analysis = TimestampedAnalysis {
            timestamp: SystemTime::now(),
            analysis: analysis.clone(),
            epoch,
            metadata,
        };

        // Store analysis in history
        self.analysis_history.push_back(timestamped_analysis);

        // Maintain history size limit
        while self.analysis_history.len() > self.config.max_history_size {
            self.analysis_history.pop_front();
        }

        // Update performance tracker
        self.performance_tracker.record_analysis(&analysis)?;

        // Check for alerts
        if self.config.enable_alerting {
            self.alerting_system.check_for_alerts(&analysis)?;
        }

        debug!(
            "Analysis stored, history size: {}",
            self.analysis_history.len()
        );

        Ok(())
    }

    /// Get trend analysis from recent history
    ///
    /// Analyzes the recent gradient flow history to identify trends,
    /// improvements, and potential issues.
    pub fn get_trend_analysis(&self) -> Option<GradientTrendAnalysis> {
        if self.analysis_history.len() < self.config.min_trend_samples {
            return None;
        }

        let window_size = self
            .config
            .trend_window_size
            .min(self.analysis_history.len());
        let recent_analyses: Vec<_> = self
            .analysis_history
            .iter()
            .rev()
            .take(window_size)
            .collect();

        if recent_analyses.len() < 2 {
            return None;
        }

        let oldest = &recent_analyses[recent_analyses.len() - 1].analysis;
        let newest = &recent_analyses[0].analysis;

        // Calculate trends for different metrics
        let operations_trend = self.calculate_trend(
            oldest.total_operations as f64,
            newest.total_operations as f64,
        );

        let memory_trend = self.calculate_trend(
            oldest.memory_breakdown.total_memory() as f64,
            newest.memory_breakdown.total_memory() as f64,
        );

        let bottlenecks_trend = self.calculate_trend(
            oldest.gradient_bottlenecks.len() as f64,
            newest.gradient_bottlenecks.len() as f64,
        );

        let gradient_magnitude_trend = self.calculate_trend(
            oldest.gradient_stats.mean_magnitude as f64,
            newest.gradient_stats.mean_magnitude as f64,
        );

        let health_trend = self.calculate_trend(
            oldest.gradient_health_score(),
            newest.gradient_health_score(),
        );

        // Calculate time period
        let time_period = newest.timestamp.duration_since(oldest.timestamp);

        // Calculate confidence based on data availability
        let confidence =
            (recent_analyses.len() as f64 / self.config.trend_window_size as f64).min(1.0);

        // Detect anomalies
        let anomalies = self.detect_anomalies(&recent_analyses);

        Some(GradientTrendAnalysis {
            operations_trend,
            memory_trend,
            bottlenecks_trend,
            gradient_magnitude_trend,
            health_trend,
            time_period,
            sample_count: recent_analyses.len(),
            confidence,
            anomalies,
        })
    }

    /// Generate a comprehensive monitoring report
    pub fn generate_monitoring_report(&self) -> Result<String> {
        let mut report = String::new();
        self.write_monitoring_report(&mut report)?;
        Ok(report)
    }

    fn write_monitoring_report(&self, report: &mut String) -> Result<()> {
        use std::fmt::Write;

        // Helper to convert fmt::Error to TorshError
        let write_fmt = |f: std::result::Result<(), std::fmt::Error>| -> Result<()> {
            f.map_err(|e| {
                torsh_core::TorshError::invalid_operation(&format!("Format error: {}", e))
            })
        };

        write_fmt(writeln!(
            report,
            "╔══════════════════════════════════════════════════════════════╗"
        ))?;
        write_fmt(writeln!(
            report,
            "║                 Gradient Flow Monitoring Report              ║"
        ))?;
        write_fmt(writeln!(
            report,
            "╚══════════════════════════════════════════════════════════════╝"
        ))?;
        write_fmt(writeln!(report))?;

        // Overview
        write_fmt(writeln!(report, "📊 Monitoring Overview"))?;
        write_fmt(writeln!(
            report,
            "  Status: {}",
            if self.monitoring_enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        ))?;
        write_fmt(writeln!(
            report,
            "  History Size: {} analyses",
            self.analysis_history.len()
        ))?;
        write_fmt(writeln!(
            report,
            "  Active Alerts: {}",
            self.alerting_system.active_alert_count()
        ))?;

        if let Some(latest) = self.analysis_history.back() {
            write_fmt(writeln!(
                report,
                "  Latest Analysis: {:?}",
                latest.timestamp
            ))?;
            if let Some(epoch) = latest.epoch {
                write_fmt(writeln!(report, "  Current Epoch: {}", epoch))?;
            }
        }
        write_fmt(writeln!(report))?;

        // Trend analysis
        if let Some(trend) = self.get_trend_analysis() {
            write_fmt(writeln!(
                report,
                "📈 Trend Analysis (Last {} samples)",
                trend.sample_count
            ))?;
            write_fmt(writeln!(report, "  Operations: {}", trend.operations_trend))?;
            write_fmt(writeln!(report, "  Memory Usage: {}", trend.memory_trend))?;
            write_fmt(writeln!(
                report,
                "  Bottlenecks: {}",
                trend.bottlenecks_trend
            ))?;
            write_fmt(writeln!(
                report,
                "  Gradient Magnitude: {}",
                trend.gradient_magnitude_trend
            ))?;
            write_fmt(writeln!(report, "  Health Score: {}", trend.health_trend))?;
            write_fmt(writeln!(
                report,
                "  Confidence: {:.1}%",
                trend.confidence * 100.0
            ))?;
            write_fmt(writeln!(report))?;

            // Anomalies
            if !trend.anomalies.is_empty() {
                write_fmt(writeln!(report, "⚠️  Detected Anomalies"))?;
                for anomaly in &trend.anomalies {
                    write_fmt(writeln!(
                        report,
                        "  • {} (Severity: {:.2})",
                        anomaly.description, anomaly.severity
                    ))?;
                }
                write_fmt(writeln!(report))?;
            }
        }

        // Recent alerts
        let recent_alerts = self.alerting_system.get_recent_alerts(10);
        if !recent_alerts.is_empty() {
            write_fmt(writeln!(report, "🚨 Recent Alerts"))?;
            for alert in recent_alerts {
                write_fmt(writeln!(
                    report,
                    "  • [{:?}] {}",
                    alert.severity, alert.message
                ))?;
            }
            write_fmt(writeln!(report))?;
        }

        // Performance summary
        if let Some(latest) = self.analysis_history.back() {
            write_fmt(writeln!(report, "📋 Current Performance Summary"))?;
            let summary = latest.analysis.summary();
            write_fmt(writeln!(
                report,
                "  Total Operations: {}",
                summary.total_operations
            ))?;
            write_fmt(writeln!(
                report,
                "  Health Score: {:.3}",
                summary.gradient_health_score
            ))?;
            write_fmt(writeln!(
                report,
                "  Memory Efficiency: {:.1}%",
                summary.memory_efficiency * 100.0
            ))?;
            write_fmt(writeln!(
                report,
                "  Bottlenecks: {}",
                summary.bottleneck_count
            ))?;
        }

        Ok(())
    }

    /// Get historical data for a specific metric
    pub fn get_metric_history(&self, metric: &str) -> Vec<(SystemTime, f64)> {
        self.analysis_history
            .iter()
            .map(|entry| {
                let value = match metric {
                    "operations" => entry.analysis.total_operations as f64,
                    "memory" => entry.analysis.memory_breakdown.total_memory() as f64,
                    "bottlenecks" => entry.analysis.gradient_bottlenecks.len() as f64,
                    "health" => entry.analysis.gradient_health_score(),
                    "gradient_magnitude" => entry.analysis.gradient_stats.mean_magnitude as f64,
                    _ => 0.0,
                };
                (entry.timestamp, value)
            })
            .collect()
    }

    /// Clear monitoring history
    pub fn clear_history(&mut self) {
        self.analysis_history.clear();
        self.performance_tracker.clear();
        self.alerting_system.clear_alerts();
        info!("Monitoring history cleared");
    }

    /// Get the latest analysis
    pub fn latest_analysis(&self) -> Option<&GradientFlowAnalysis> {
        self.analysis_history.back().map(|entry| &entry.analysis)
    }

    /// Calculate trend between two values
    fn calculate_trend(&self, old_value: f64, new_value: f64) -> TrendDirection {
        if old_value == 0.0 {
            return if new_value > 0.0 {
                TrendDirection::StronglyIncreasing(f64::INFINITY)
            } else {
                TrendDirection::Stable(0.0)
            };
        }

        let percentage_change = ((new_value - old_value) / old_value) * 100.0;
        TrendDirection::from_percentage_change(percentage_change)
    }

    /// Detect anomalies in recent analyses
    fn detect_anomalies(&self, analyses: &[&TimestampedAnalysis]) -> Vec<AnomalyReport> {
        let mut anomalies = Vec::new();

        if analyses.len() < 3 {
            return anomalies;
        }

        // Check for gradient spikes or vanishing
        self.detect_gradient_anomalies(analyses, &mut anomalies);

        // Check for memory anomalies
        self.detect_memory_anomalies(analyses, &mut anomalies);

        // Check for performance drops
        self.detect_performance_anomalies(analyses, &mut anomalies);

        anomalies
    }

    /// Detect gradient-related anomalies
    fn detect_gradient_anomalies(
        &self,
        analyses: &[&TimestampedAnalysis],
        anomalies: &mut Vec<AnomalyReport>,
    ) {
        let magnitudes: Vec<f32> = analyses
            .iter()
            .map(|a| a.analysis.gradient_stats.mean_magnitude)
            .collect();

        // Simple anomaly detection based on sudden changes
        for i in 1..magnitudes.len() {
            let prev = magnitudes[i - 1];
            let curr = magnitudes[i];

            if prev > 0.0 {
                let change_ratio = curr / prev;
                if change_ratio > 5.0 {
                    anomalies.push(AnomalyReport {
                        anomaly_type: AnomalyType::GradientSpike,
                        severity: (change_ratio - 1.0) as f64,
                        description: format!(
                            "Gradient magnitude spike: {:.2e} -> {:.2e}",
                            prev, curr
                        ),
                        timestamp: analyses[i].timestamp,
                        suggestions: vec![
                            "Check learning rate".to_string(),
                            "Monitor gradient clipping".to_string(),
                        ],
                    });
                } else if change_ratio < 0.1 {
                    anomalies.push(AnomalyReport {
                        anomaly_type: AnomalyType::GradientVanishing,
                        severity: (1.0 - change_ratio) as f64,
                        description: format!(
                            "Gradient magnitude drop: {:.2e} -> {:.2e}",
                            prev, curr
                        ),
                        timestamp: analyses[i].timestamp,
                        suggestions: vec![
                            "Check for vanishing gradients".to_string(),
                            "Consider gradient scaling".to_string(),
                        ],
                    });
                }
            }
        }
    }

    /// Detect memory-related anomalies
    fn detect_memory_anomalies(
        &self,
        analyses: &[&TimestampedAnalysis],
        anomalies: &mut Vec<AnomalyReport>,
    ) {
        let memory_usage: Vec<usize> = analyses
            .iter()
            .map(|a| a.analysis.memory_breakdown.total_memory())
            .collect();

        for i in 1..memory_usage.len() {
            let prev = memory_usage[i - 1] as f64;
            let curr = memory_usage[i] as f64;

            if prev > 0.0 {
                let change_ratio = curr / prev;
                if change_ratio > 2.0 {
                    anomalies.push(AnomalyReport {
                        anomaly_type: AnomalyType::MemoryAnomaly,
                        severity: change_ratio - 1.0,
                        description: format!(
                            "Memory usage spike: {:.1}MB -> {:.1}MB",
                            prev / (1024.0 * 1024.0),
                            curr / (1024.0 * 1024.0)
                        ),
                        timestamp: analyses[i].timestamp,
                        suggestions: vec![
                            "Check for memory leaks".to_string(),
                            "Monitor batch size".to_string(),
                        ],
                    });
                }
            }
        }
    }

    /// Detect performance-related anomalies
    fn detect_performance_anomalies(
        &self,
        analyses: &[&TimestampedAnalysis],
        anomalies: &mut Vec<AnomalyReport>,
    ) {
        let health_scores: Vec<f64> = analyses
            .iter()
            .map(|a| a.analysis.gradient_health_score())
            .collect();

        for i in 1..health_scores.len() {
            let prev = health_scores[i - 1];
            let curr = health_scores[i];

            let change = curr - prev;
            if change < -0.2 {
                anomalies.push(AnomalyReport {
                    anomaly_type: AnomalyType::PerformanceDrop,
                    severity: (-change) * 5.0, // Scale to 0-1 range
                    description: format!("Health score drop: {:.3} -> {:.3}", prev, curr),
                    timestamp: analyses[i].timestamp,
                    suggestions: vec![
                        "Review recent changes".to_string(),
                        "Check data quality".to_string(),
                    ],
                });
            }
        }
    }
}

impl Default for GradientFlowMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation for helper structs

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            metrics_history: std::collections::HashMap::new(),
            config: TrackingConfig {
                max_metric_points: 1000,
                tracked_metrics: vec![
                    "operations".to_string(),
                    "memory".to_string(),
                    "health".to_string(),
                ],
            },
        }
    }

    fn record_analysis(&mut self, analysis: &GradientFlowAnalysis) -> Result<()> {
        let timestamp = SystemTime::now();

        // Record tracked metrics
        for metric in &self.config.tracked_metrics {
            let value = match metric.as_str() {
                "operations" => analysis.total_operations as f64,
                "memory" => analysis.memory_breakdown.total_memory() as f64,
                "health" => analysis.gradient_health_score(),
                _ => continue,
            };

            let metric_point = MetricPoint {
                timestamp,
                value,
                context: None,
            };

            let history = self
                .metrics_history
                .entry(metric.clone())
                .or_insert_with(|| VecDeque::with_capacity(self.config.max_metric_points));

            history.push_back(metric_point);

            // Maintain size limit
            while history.len() > self.config.max_metric_points {
                history.pop_front();
            }
        }

        Ok(())
    }

    fn clear(&mut self) {
        self.metrics_history.clear();
    }
}

impl AlertingSystem {
    fn new() -> Self {
        Self {
            config: AlertingConfig::default(),
            alert_history: VecDeque::with_capacity(100),
        }
    }

    fn check_for_alerts(&mut self, analysis: &GradientFlowAnalysis) -> Result<()> {
        // Check gradient magnitude alerts
        if self.config.enable_gradient_alerts {
            if analysis.gradient_stats.max_magnitude > self.config.gradient_magnitude_threshold {
                self.generate_alert(Alert {
                    alert_type: AlertType::GradientMagnitude,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "High gradient magnitude detected: {:.2e}",
                        analysis.gradient_stats.max_magnitude
                    ),
                    timestamp: SystemTime::now(),
                    analysis_id: None,
                });
            }
        }

        // Check memory alerts
        if self.config.enable_memory_alerts {
            let memory_mb = analysis.memory_breakdown.total_memory() as f64 / (1024.0 * 1024.0);
            if memory_mb > self.config.memory_threshold_mb {
                self.generate_alert(Alert {
                    alert_type: AlertType::MemoryUsage,
                    severity: AlertSeverity::Warning,
                    message: format!("High memory usage: {:.1} MB", memory_mb),
                    timestamp: SystemTime::now(),
                    analysis_id: None,
                });
            }
        }

        // Check bottleneck alerts
        if self.config.enable_bottleneck_alerts {
            if analysis.gradient_bottlenecks.len() > self.config.bottleneck_count_threshold {
                self.generate_alert(Alert {
                    alert_type: AlertType::Bottlenecks,
                    severity: AlertSeverity::Critical,
                    message: format!(
                        "Multiple bottlenecks detected: {}",
                        analysis.gradient_bottlenecks.len()
                    ),
                    timestamp: SystemTime::now(),
                    analysis_id: None,
                });
            }
        }

        Ok(())
    }

    fn generate_alert(&mut self, alert: Alert) {
        self.alert_history.push_back(alert);

        // Maintain history size
        while self.alert_history.len() > 100 {
            self.alert_history.pop_front();
        }
    }

    fn active_alert_count(&self) -> usize {
        // Count alerts from the last hour
        let one_hour_ago = SystemTime::now() - Duration::from_secs(3600);
        self.alert_history
            .iter()
            .filter(|alert| alert.timestamp > one_hour_ago)
            .count()
    }

    fn get_recent_alerts(&self, count: usize) -> Vec<&Alert> {
        self.alert_history.iter().rev().take(count).collect()
    }

    fn clear_alerts(&mut self) {
        self.alert_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::AutogradContext;

    #[test]
    fn test_monitor_creation() {
        let monitor = GradientFlowMonitor::new();
        assert!(monitor.is_monitoring_enabled());
        assert_eq!(monitor.analysis_history.len(), 0);
    }

    #[test]
    fn test_monitoring_enable_disable() {
        let mut monitor = GradientFlowMonitor::new();

        assert!(monitor.is_monitoring_enabled());

        monitor.set_monitoring(false);
        assert!(!monitor.is_monitoring_enabled());

        monitor.set_monitoring(true);
        assert!(monitor.is_monitoring_enabled());
    }

    #[test]
    fn test_analysis_storage() {
        let mut monitor = GradientFlowMonitor::new();
        let ctx = AutogradContext::new();

        // Initially no history
        assert!(monitor.get_trend_analysis().is_none());

        // Add some analyses
        monitor.analyze_and_store(&ctx).unwrap();
        monitor.analyze_and_store(&ctx).unwrap();

        assert_eq!(monitor.analysis_history.len(), 2);
    }

    #[test]
    fn test_trend_analysis() {
        let mut monitor = GradientFlowMonitor::new();
        let ctx = AutogradContext::new();

        // Need multiple analyses for trend
        for _ in 0..5 {
            monitor.analyze_and_store(&ctx).unwrap();
        }

        let trend = monitor.get_trend_analysis();
        assert!(trend.is_some());

        let trend_analysis = trend.unwrap();
        assert_eq!(trend_analysis.sample_count, 5);
        assert!(trend_analysis.confidence > 0.0);
    }

    #[test]
    fn test_trend_direction() {
        let trend = TrendDirection::from_percentage_change(15.0);
        assert!(matches!(trend, TrendDirection::StronglyIncreasing(_)));

        let trend = TrendDirection::from_percentage_change(7.0);
        assert!(matches!(trend, TrendDirection::Increasing(_)));

        let trend = TrendDirection::from_percentage_change(2.0);
        assert!(matches!(trend, TrendDirection::Stable(_)));

        let trend = TrendDirection::from_percentage_change(-7.0);
        assert!(matches!(trend, TrendDirection::Decreasing(_)));

        let trend = TrendDirection::from_percentage_change(-15.0);
        assert!(matches!(trend, TrendDirection::StronglyDecreasing(_)));
    }

    #[test]
    fn test_trend_improvement() {
        let increasing = TrendDirection::Increasing(10.0);
        let decreasing = TrendDirection::Decreasing(-10.0);

        // For health metrics, increasing is good
        assert!(increasing.is_improvement(MetricType::GradientHealth));
        assert!(!decreasing.is_improvement(MetricType::GradientHealth));

        // For bottlenecks, decreasing is good
        assert!(!increasing.is_improvement(MetricType::Bottlenecks));
        assert!(decreasing.is_improvement(MetricType::Bottlenecks));
    }

    #[test]
    fn test_metric_history() {
        let mut monitor = GradientFlowMonitor::new();
        let ctx = AutogradContext::new();

        monitor.analyze_and_store(&ctx).unwrap();
        monitor.analyze_and_store(&ctx).unwrap();

        let history = monitor.get_metric_history("operations");
        assert_eq!(history.len(), 2);

        let health_history = monitor.get_metric_history("health");
        assert_eq!(health_history.len(), 2);
    }

    #[test]
    fn test_monitoring_report() {
        let mut monitor = GradientFlowMonitor::new();
        let ctx = AutogradContext::new();

        monitor.analyze_and_store(&ctx).unwrap();

        let report = monitor.generate_monitoring_report();
        assert!(report.is_ok());

        let report_text = report.unwrap();
        assert!(report_text.contains("Gradient Flow Monitoring Report"));
        assert!(report_text.contains("Monitoring Overview"));
    }

    #[test]
    fn test_clear_history() {
        let mut monitor = GradientFlowMonitor::new();
        let ctx = AutogradContext::new();

        monitor.analyze_and_store(&ctx).unwrap();
        assert_eq!(monitor.analysis_history.len(), 1);

        monitor.clear_history();
        assert_eq!(monitor.analysis_history.len(), 0);
    }

    #[test]
    fn test_latest_analysis() {
        let mut monitor = GradientFlowMonitor::new();
        let ctx = AutogradContext::new();

        assert!(monitor.latest_analysis().is_none());

        monitor.analyze_and_store(&ctx).unwrap();
        assert!(monitor.latest_analysis().is_some());
    }

    #[test]
    fn test_alert_types() {
        use AlertType::*;
        let alerts = vec![
            GradientMagnitude,
            MemoryUsage,
            Bottlenecks,
            PerformanceRegression,
            TrainingInstability,
        ];

        // Test that all alert types are distinct
        for (i, alert1) in alerts.iter().enumerate() {
            for (j, alert2) in alerts.iter().enumerate() {
                if i != j {
                    assert_ne!(alert1, alert2);
                }
            }
        }
    }

    #[test]
    fn test_alert_severity_ordering() {
        use AlertSeverity::*;
        assert!(Info < Warning);
        assert!(Warning < Critical);
        assert!(Info < Critical);
    }

    #[test]
    fn test_anomaly_types() {
        use AnomalyType::*;
        let anomalies = vec![
            GradientSpike,
            GradientVanishing,
            MemoryAnomaly,
            PerformanceDrop,
            Oscillation,
        ];

        // Test that all anomaly types are distinct
        for (i, anomaly1) in anomalies.iter().enumerate() {
            for (j, anomaly2) in anomalies.iter().enumerate() {
                if i != j {
                    assert_ne!(anomaly1, anomaly2);
                }
            }
        }
    }
}
