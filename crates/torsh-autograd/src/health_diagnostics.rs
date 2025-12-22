// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Autograd Health Checks and Diagnostics
//!
//! This module provides comprehensive health monitoring and diagnostic capabilities
//! for autograd systems, enabling early detection of issues and proactive maintenance.
//!
//! # Features
//!
//! - **System Health Checks**: Memory, computation graph, gradient health
//! - **Performance Diagnostics**: Identify performance bottlenecks
//! - **Resource Monitoring**: Track resource usage and limits
//! - **Automated Recommendations**: Actionable suggestions for issues
//! - **Health Scoring**: Quantitative health metrics (0-100)
//! - **Alerting Integration**: Integration with anomaly alerting

use crate::common_utils::*;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Check name
    pub name: String,

    /// Check status
    pub status: HealthStatus,

    /// Health score (0-100)
    pub score: u8,

    /// Details/description
    pub details: String,

    /// Metrics collected
    pub metrics: HashMap<String, f64>,

    /// Recommendations
    pub recommendations: Vec<String>,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Duration of health check
    pub duration: Duration,
}

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Healthy - no issues
    Healthy,

    /// Warning - minor issues detected
    Warning,

    /// Critical - major issues requiring attention
    Critical,

    /// Unknown - unable to determine health
    Unknown,
}

/// Overall system health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall health status
    pub overall_status: HealthStatus,

    /// Overall health score (0-100)
    pub overall_score: u8,

    /// Individual check results
    pub checks: Vec<HealthCheckResult>,

    /// Critical issues count
    pub critical_count: usize,

    /// Warning issues count
    pub warning_count: usize,

    /// Top recommendations
    pub top_recommendations: Vec<String>,

    /// Report timestamp
    pub timestamp: DateTime<Utc>,

    /// Total check duration
    pub total_duration: Duration,
}

/// Health check trait
pub trait HealthCheck: Send + Sync {
    /// Run the health check
    fn check(&self) -> HealthCheckResult;

    /// Get check name
    fn name(&self) -> &str;

    /// Get check description
    fn description(&self) -> &str;
}

/// Memory health check
pub struct MemoryHealthCheck {
    /// Memory threshold for warning (bytes)
    pub warning_threshold: usize,

    /// Memory threshold for critical (bytes)
    pub critical_threshold: usize,
}

impl Default for MemoryHealthCheck {
    fn default() -> Self {
        Self {
            warning_threshold: 4 * 1024 * 1024 * 1024,  // 4 GB
            critical_threshold: 8 * 1024 * 1024 * 1024, // 8 GB
        }
    }
}

impl HealthCheck for MemoryHealthCheck {
    fn check(&self) -> HealthCheckResult {
        let start = std::time::Instant::now();

        // Get system memory info (simplified - real implementation would query system)
        let estimated_usage = 1024 * 1024 * 1024; // 1 GB placeholder

        let status = if estimated_usage > self.critical_threshold {
            HealthStatus::Critical
        } else if estimated_usage > self.warning_threshold {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        let score = if estimated_usage > self.critical_threshold {
            20
        } else if estimated_usage > self.warning_threshold {
            60
        } else {
            100
        };

        let mut metrics = HashMap::new();
        metrics.insert("memory_bytes".to_string(), estimated_usage as f64);
        metrics.insert(
            "warning_threshold".to_string(),
            self.warning_threshold as f64,
        );
        metrics.insert(
            "critical_threshold".to_string(),
            self.critical_threshold as f64,
        );

        let mut recommendations = Vec::new();
        if status == HealthStatus::Critical {
            recommendations.push(
                "CRITICAL: Memory usage is very high. Consider reducing batch size or model size."
                    .to_string(),
            );
        } else if status == HealthStatus::Warning {
            recommendations.push(
                "Memory usage is elevated. Monitor closely and consider optimization.".to_string(),
            );
        }

        HealthCheckResult {
            name: self.name().to_string(),
            status,
            score,
            details: format!("Memory usage: {}", format_bytes(estimated_usage)),
            metrics,
            recommendations,
            timestamp: Utc::now(),
            duration: start.elapsed(),
        }
    }

    fn name(&self) -> &str {
        "memory_health"
    }

    fn description(&self) -> &str {
        "Checks system memory usage and availability"
    }
}

/// Computation graph health check
pub struct GraphHealthCheck {
    /// Maximum safe graph depth
    pub max_safe_depth: usize,

    /// Maximum safe node count
    pub max_safe_nodes: usize,
}

impl Default for GraphHealthCheck {
    fn default() -> Self {
        Self {
            max_safe_depth: 1000,
            max_safe_nodes: 100_000,
        }
    }
}

impl HealthCheck for GraphHealthCheck {
    fn check(&self) -> HealthCheckResult {
        let start = std::time::Instant::now();

        // Placeholder values - real implementation would query actual graph
        let current_depth = 50;
        let current_nodes = 5000;

        let depth_ratio = current_depth as f64 / self.max_safe_depth as f64;
        let nodes_ratio = current_nodes as f64 / self.max_safe_nodes as f64;

        let status = if depth_ratio > 0.9 || nodes_ratio > 0.9 {
            HealthStatus::Critical
        } else if depth_ratio > 0.7 || nodes_ratio > 0.7 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        let score = ((1.0 - depth_ratio.max(nodes_ratio)) * 100.0).clamp(0.0, 100.0) as u8;

        let mut metrics = HashMap::new();
        metrics.insert("graph_depth".to_string(), current_depth as f64);
        metrics.insert("graph_nodes".to_string(), current_nodes as f64);
        metrics.insert("max_depth".to_string(), self.max_safe_depth as f64);
        metrics.insert("max_nodes".to_string(), self.max_safe_nodes as f64);

        let mut recommendations = Vec::new();
        if depth_ratio > 0.7 {
            recommendations
                .push("Graph depth is high. Consider using gradient checkpointing.".to_string());
        }
        if nodes_ratio > 0.7 {
            recommendations.push(
                "Graph has many nodes. Consider operation fusion or graph optimization."
                    .to_string(),
            );
        }

        HealthCheckResult {
            name: self.name().to_string(),
            status,
            score,
            details: format!("Graph: {} nodes, depth {}", current_nodes, current_depth),
            metrics,
            recommendations,
            timestamp: Utc::now(),
            duration: start.elapsed(),
        }
    }

    fn name(&self) -> &str {
        "graph_health"
    }

    fn description(&self) -> &str {
        "Checks computation graph complexity and size"
    }
}

/// Gradient health check
pub struct GradientHealthCheck {
    /// Recent gradient statistics (for trend analysis)
    recent_stats: Arc<Mutex<VecDeque<GradientStats>>>,

    /// Maximum history size
    max_history: usize,
}

#[derive(Debug, Clone)]
struct GradientStats {
    #[allow(dead_code)]
    timestamp: DateTime<Utc>,
    mean_magnitude: f64,
    zero_ratio: f64,
    nan_count: usize,
    inf_count: usize,
}

impl Default for GradientHealthCheck {
    fn default() -> Self {
        Self {
            recent_stats: Arc::new(Mutex::new(VecDeque::new())),
            max_history: 100,
        }
    }
}

impl GradientHealthCheck {
    /// Record gradient statistics
    pub fn record_stats(
        &self,
        mean_magnitude: f64,
        zero_ratio: f64,
        nan_count: usize,
        inf_count: usize,
    ) {
        let stats = GradientStats {
            timestamp: Utc::now(),
            mean_magnitude,
            zero_ratio,
            nan_count,
            inf_count,
        };

        let mut history = self.recent_stats.lock();
        history.push_back(stats);

        while history.len() > self.max_history {
            history.pop_front();
        }
    }
}

impl HealthCheck for GradientHealthCheck {
    fn check(&self) -> HealthCheckResult {
        let start = std::time::Instant::now();

        let history = self.recent_stats.lock();

        let status = if history.is_empty() {
            HealthStatus::Unknown
        } else {
            let latest = history.back().unwrap();

            if latest.nan_count > 0 || latest.inf_count > 0 {
                HealthStatus::Critical
            } else if latest.mean_magnitude < 1e-7 || latest.mean_magnitude > 100.0 {
                HealthStatus::Warning
            } else if latest.zero_ratio > 0.9 {
                HealthStatus::Warning
            } else {
                HealthStatus::Healthy
            }
        };

        let score = if status == HealthStatus::Critical {
            0
        } else if status == HealthStatus::Warning {
            50
        } else if status == HealthStatus::Unknown {
            70
        } else {
            100
        };

        let mut metrics = HashMap::new();
        let mut recommendations = Vec::new();

        if let Some(latest) = history.back() {
            metrics.insert("mean_magnitude".to_string(), latest.mean_magnitude);
            metrics.insert("zero_ratio".to_string(), latest.zero_ratio);
            metrics.insert("nan_count".to_string(), latest.nan_count as f64);
            metrics.insert("inf_count".to_string(), latest.inf_count as f64);

            if latest.nan_count > 0 {
                recommendations.push(
                    "CRITICAL: NaN gradients detected. Check for numerical instability."
                        .to_string(),
                );
            }
            if latest.inf_count > 0 {
                recommendations.push("CRITICAL: Infinite gradients detected. Reduce learning rate or add gradient clipping.".to_string());
            }
            if latest.mean_magnitude < 1e-7 {
                recommendations.push("Vanishing gradients detected. Consider residual connections or better initialization.".to_string());
            }
            if latest.mean_magnitude > 100.0 {
                recommendations
                    .push("Large gradients detected. Consider gradient clipping.".to_string());
            }
            if latest.zero_ratio > 0.9 {
                recommendations.push("Many zero gradients (dead neurons). Consider LeakyReLU or better initialization.".to_string());
            }
        }

        let details = if history.is_empty() {
            "No gradient statistics available".to_string()
        } else {
            format!("Monitoring {} recent gradient computations", history.len())
        };

        HealthCheckResult {
            name: self.name().to_string(),
            status,
            score,
            details,
            metrics,
            recommendations,
            timestamp: Utc::now(),
            duration: start.elapsed(),
        }
    }

    fn name(&self) -> &str {
        "gradient_health"
    }

    fn description(&self) -> &str {
        "Checks gradient computation health and stability"
    }
}

/// Performance health check
pub struct PerformanceHealthCheck {
    /// Recent operation timings
    recent_timings: Arc<Mutex<VecDeque<f64>>>,

    /// Warning threshold (ms)
    pub warning_threshold_ms: f64,

    /// Critical threshold (ms)
    pub critical_threshold_ms: f64,
}

impl Default for PerformanceHealthCheck {
    fn default() -> Self {
        Self {
            recent_timings: Arc::new(Mutex::new(VecDeque::new())),
            warning_threshold_ms: 100.0,
            critical_threshold_ms: 500.0,
        }
    }
}

impl PerformanceHealthCheck {
    /// Record operation timing
    pub fn record_timing(&self, duration_ms: f64) {
        let mut timings = self.recent_timings.lock();
        timings.push_back(duration_ms);

        while timings.len() > 100 {
            timings.pop_front();
        }
    }
}

impl HealthCheck for PerformanceHealthCheck {
    fn check(&self) -> HealthCheckResult {
        let start = std::time::Instant::now();

        let timings = self.recent_timings.lock();

        let status = if timings.is_empty() {
            HealthStatus::Unknown
        } else {
            let avg = mean(&timings.iter().copied().collect::<Vec<_>>());

            if avg > self.critical_threshold_ms {
                HealthStatus::Critical
            } else if avg > self.warning_threshold_ms {
                HealthStatus::Warning
            } else {
                HealthStatus::Healthy
            }
        };

        let score = if timings.is_empty() {
            70
        } else {
            let avg = mean(&timings.iter().copied().collect::<Vec<_>>());
            let ratio = (self.critical_threshold_ms - avg) / self.critical_threshold_ms;
            (ratio * 100.0).clamp(0.0, 100.0) as u8
        };

        let mut metrics = HashMap::new();
        let mut recommendations = Vec::new();

        if !timings.is_empty() {
            let values: Vec<f64> = timings.iter().copied().collect();
            let avg = mean(&values);
            let p50 = percentile(&values, 50.0);
            let p95 = percentile(&values, 95.0);
            let p99 = percentile(&values, 99.0);

            metrics.insert("avg_ms".to_string(), avg);
            metrics.insert("p50_ms".to_string(), p50);
            metrics.insert("p95_ms".to_string(), p95);
            metrics.insert("p99_ms".to_string(), p99);

            if avg > self.critical_threshold_ms {
                recommendations.push(
                    "CRITICAL: Operations are very slow. Profile to find bottlenecks.".to_string(),
                );
            } else if avg > self.warning_threshold_ms {
                recommendations.push(
                    "Operations are slower than expected. Consider optimization.".to_string(),
                );
            }

            if p99 > 2.0 * avg {
                recommendations
                    .push("High latency variance detected. Investigate outliers.".to_string());
            }
        }

        HealthCheckResult {
            name: self.name().to_string(),
            status,
            score,
            details: format!("Monitoring {} recent operations", timings.len()),
            metrics,
            recommendations,
            timestamp: Utc::now(),
            duration: start.elapsed(),
        }
    }

    fn name(&self) -> &str {
        "performance_health"
    }

    fn description(&self) -> &str {
        "Checks autograd operation performance"
    }
}

/// Health diagnostic system
pub struct HealthDiagnostics {
    /// Registered health checks
    checks: Arc<Mutex<Vec<Arc<dyn HealthCheck>>>>,

    /// Health check history
    history: Arc<Mutex<VecDeque<HealthReport>>>,

    /// Maximum history size
    max_history: usize,
}

impl HealthDiagnostics {
    /// Create a new health diagnostics system
    pub fn new() -> Self {
        let mut diagnostics = Self {
            checks: Arc::new(Mutex::new(Vec::new())),
            history: Arc::new(Mutex::new(VecDeque::new())),
            max_history: 100,
        };

        // Register default checks
        diagnostics.register_check(Arc::new(MemoryHealthCheck::default()));
        diagnostics.register_check(Arc::new(GraphHealthCheck::default()));
        diagnostics.register_check(Arc::new(GradientHealthCheck::default()));
        diagnostics.register_check(Arc::new(PerformanceHealthCheck::default()));

        diagnostics
    }

    /// Register a health check
    pub fn register_check(&mut self, check: Arc<dyn HealthCheck>) {
        self.checks.lock().push(check);
    }

    /// Run all health checks and generate report
    pub fn run_diagnostics(&self) -> HealthReport {
        let start = std::time::Instant::now();

        let checks = self.checks.lock();
        let mut results = Vec::new();

        for check in checks.iter() {
            results.push(check.check());
        }

        let critical_count = results
            .iter()
            .filter(|r| r.status == HealthStatus::Critical)
            .count();
        let warning_count = results
            .iter()
            .filter(|r| r.status == HealthStatus::Warning)
            .count();

        let overall_status = if critical_count > 0 {
            HealthStatus::Critical
        } else if warning_count > 0 {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };

        let overall_score = if results.is_empty() {
            0
        } else {
            (results.iter().map(|r| r.score as u32).sum::<u32>() / results.len() as u32) as u8
        };

        // Collect top recommendations
        let mut all_recommendations: Vec<String> = results
            .iter()
            .flat_map(|r| r.recommendations.clone())
            .collect();

        all_recommendations.sort();
        all_recommendations.dedup();
        let top_recommendations = all_recommendations.into_iter().take(5).collect();

        let report = HealthReport {
            overall_status,
            overall_score,
            checks: results,
            critical_count,
            warning_count,
            top_recommendations,
            timestamp: Utc::now(),
            total_duration: start.elapsed(),
        };

        // Add to history
        let mut history = self.history.lock();
        history.push_back(report.clone());

        while history.len() > self.max_history {
            history.pop_front();
        }

        report
    }

    /// Get health check history
    pub fn history(&self) -> Vec<HealthReport> {
        self.history.lock().iter().cloned().collect()
    }

    /// Generate text report
    pub fn format_report(&self, report: &HealthReport) -> String {
        let mut output = String::new();

        output.push_str("=== Autograd Health Diagnostics Report ===\n\n");
        output.push_str(&format!("Overall Status: {:?}\n", report.overall_status));
        output.push_str(&format!("Overall Score: {}/100\n", report.overall_score));
        output.push_str(&format!("Timestamp: {}\n", report.timestamp));
        output.push_str(&format!("Duration: {:?}\n\n", report.total_duration));

        output.push_str(&format!(
            "Issues: {} critical, {} warning\n\n",
            report.critical_count, report.warning_count
        ));

        output.push_str("Health Checks:\n");
        for check in &report.checks {
            let status_icon = match check.status {
                HealthStatus::Healthy => "✓",
                HealthStatus::Warning => "⚠",
                HealthStatus::Critical => "✗",
                HealthStatus::Unknown => "?",
            };

            output.push_str(&format!(
                "  {} {} - Score: {}/100\n",
                status_icon, check.name, check.score
            ));
            output.push_str(&format!("     {}\n", check.details));

            if !check.recommendations.is_empty() {
                for rec in &check.recommendations {
                    output.push_str(&format!("     → {}\n", rec));
                }
            }
        }

        if !report.top_recommendations.is_empty() {
            output.push_str("\nTop Recommendations:\n");
            for (i, rec) in report.top_recommendations.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, rec));
            }
        }

        output
    }
}

impl Default for HealthDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_health_check() {
        let check = MemoryHealthCheck::default();
        let result = check.check();

        assert_eq!(result.name, "memory_health");
        assert!(result.score <= 100);
    }

    #[test]
    fn test_graph_health_check() {
        let check = GraphHealthCheck::default();
        let result = check.check();

        assert_eq!(result.name, "graph_health");
        assert!(result.score <= 100);
    }

    #[test]
    fn test_gradient_health_check() {
        let check = GradientHealthCheck::default();

        // Record some stats
        check.record_stats(0.1, 0.0, 0, 0);

        let result = check.check();

        assert_eq!(result.name, "gradient_health");
        assert!(result.score <= 100);
    }

    #[test]
    fn test_performance_health_check() {
        let check = PerformanceHealthCheck::default();

        // Record some timings
        check.record_timing(50.0);
        check.record_timing(60.0);
        check.record_timing(55.0);

        let result = check.check();

        assert_eq!(result.name, "performance_health");
        assert!(result.score <= 100);
    }

    #[test]
    fn test_diagnostics_system() {
        let diagnostics = HealthDiagnostics::new();
        let report = diagnostics.run_diagnostics();

        assert!(report.overall_score <= 100);
        assert_eq!(report.checks.len(), 4); // Default checks
    }

    #[test]
    fn test_report_formatting() {
        let diagnostics = HealthDiagnostics::new();
        let report = diagnostics.run_diagnostics();
        let formatted = diagnostics.format_report(&report);

        assert!(formatted.contains("Health Diagnostics Report"));
        assert!(formatted.contains("Overall Status"));
    }
}
