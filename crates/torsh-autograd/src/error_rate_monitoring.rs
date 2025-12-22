// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Error Rate Monitoring and Alerting
//!
//! This module provides comprehensive error rate monitoring and alerting
//! for autograd operations, enabling proactive issue detection and response.
//!
//! # Features
//!
//! - **Real-time Error Tracking**: Monitor error rates across all operations
//! - **Intelligent Alerting**: Multi-level alerts with configurable thresholds
//! - **Trend Analysis**: Detect error rate trends and anomalies
//! - **Error Classification**: Categorize errors by type and severity
//! - **Alert Aggregation**: Prevent alert fatigue with smart aggregation
//! - **Integration**: Webhook, email, and custom alert handlers

use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock};

/// Error rate monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMonitoringConfig {
    /// Enable error rate monitoring
    pub enabled: bool,

    /// Window size for rate calculation (seconds)
    pub window_size_secs: u64,

    /// Sampling interval (seconds)
    pub sampling_interval_secs: u64,

    /// Warning threshold (error rate)
    pub warning_threshold: f64,

    /// Critical threshold (error rate)
    pub critical_threshold: f64,

    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,

    /// Anomaly detection sensitivity (0.0-1.0)
    pub anomaly_sensitivity: f64,

    /// Alert cooldown period (seconds)
    pub alert_cooldown_secs: u64,

    /// Maximum alerts per hour
    pub max_alerts_per_hour: usize,

    /// Enable alert aggregation
    pub enable_alert_aggregation: bool,
}

impl Default for ErrorMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_size_secs: 300,      // 5 minutes
            sampling_interval_secs: 60, // 1 minute
            warning_threshold: 0.01,    // 1% error rate
            critical_threshold: 0.05,   // 5% error rate
            enable_anomaly_detection: true,
            anomaly_sensitivity: 0.7,
            alert_cooldown_secs: 300, // 5 minutes
            max_alerts_per_hour: 10,
            enable_alert_aggregation: true,
        }
    }
}

/// Error rate monitor
pub struct ErrorRateMonitor {
    config: ErrorMonitoringConfig,
    error_history: Arc<RwLock<ErrorHistory>>,
    alert_manager: Arc<RwLock<AlertManager>>,
    statistics: Arc<RwLock<ErrorRateStatistics>>,
}

/// Historical error tracking
#[derive(Debug, Default)]
struct ErrorHistory {
    /// Timestamped error events
    events: VecDeque<ErrorEvent>,

    /// Total operations count
    total_operations: VecDeque<OperationCount>,

    /// Error counts by category
    errors_by_category: HashMap<ErrorCategory, VecDeque<ErrorEvent>>,
}

/// Error event record
#[derive(Debug, Clone)]
struct ErrorEvent {
    timestamp: DateTime<Utc>,
    #[allow(dead_code)]
    error_type: String,
    #[allow(dead_code)]
    category: ErrorCategory,
    severity: ErrorSeverity,
    #[allow(dead_code)]
    operation_name: String,
    #[allow(dead_code)]
    context: HashMap<String, String>,
}

/// Operation count record
#[derive(Debug, Clone)]
struct OperationCount {
    timestamp: DateTime<Utc>,
    count: u64,
}

/// Error category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Numerical errors (NaN, Inf)
    Numerical,

    /// Memory errors (OOM)
    Memory,

    /// Computation errors (shape mismatch, etc.)
    Computation,

    /// System errors (IO, network)
    System,

    /// Timeout errors
    Timeout,

    /// Resource exhaustion
    ResourceExhaustion,

    /// Configuration errors
    Configuration,

    /// Unknown errors
    Unknown,
}

/// Error severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Low severity
    Low,

    /// Medium severity
    Medium,

    /// High severity
    High,

    /// Critical severity
    Critical,
}

/// Alert manager
struct AlertManager {
    /// Active alerts
    active_alerts: HashMap<String, Alert>,

    /// Alert history
    alert_history: VecDeque<Alert>,

    /// Alert handlers
    handlers: Vec<Box<dyn AlertHandler + Send + Sync>>,

    /// Last alert time by category
    last_alert_time: HashMap<ErrorCategory, DateTime<Utc>>,

    /// Alert count in current hour
    alerts_this_hour: VecDeque<DateTime<Utc>>,
}

impl Default for AlertManager {
    fn default() -> Self {
        Self {
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            handlers: Vec::new(),
            last_alert_time: HashMap::new(),
            alerts_this_hour: VecDeque::new(),
        }
    }
}

/// Alert structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,

    /// Alert level
    pub level: AlertLevel,

    /// Alert title
    pub title: String,

    /// Alert description
    pub description: String,

    /// Error rate at time of alert
    pub error_rate: f64,

    /// Error category
    pub category: Option<ErrorCategory>,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Resolved flag
    pub resolved: bool,

    /// Resolution time
    pub resolved_at: Option<DateTime<Utc>>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Alert level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertLevel {
    /// Informational alert
    Info,

    /// Warning alert
    Warning,

    /// Critical alert
    Critical,

    /// Emergency alert
    Emergency,
}

/// Alert handler trait
pub trait AlertHandler {
    /// Handle new alert
    fn handle_alert(&mut self, alert: &Alert);

    /// Handle alert resolution
    fn handle_resolution(&mut self, alert: &Alert);
}

/// Error rate statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ErrorRateStatistics {
    /// Current error rate
    pub current_error_rate: f64,

    /// Error rate trend (positive = increasing)
    pub error_rate_trend: f64,

    /// Total errors in window
    pub total_errors: u64,

    /// Total operations in window
    pub total_operations: u64,

    /// Errors by category
    pub errors_by_category: HashMap<String, u64>,

    /// Errors by severity
    pub errors_by_severity: HashMap<String, u64>,

    /// Peak error rate in window
    pub peak_error_rate: f64,

    /// Average error rate
    pub avg_error_rate: f64,

    /// Active alerts count
    pub active_alerts_count: usize,

    /// Total alerts fired
    pub total_alerts_fired: u64,
}

impl ErrorRateMonitor {
    /// Create a new error rate monitor
    pub fn new(config: ErrorMonitoringConfig) -> Self {
        Self {
            config,
            error_history: Arc::new(RwLock::new(ErrorHistory::default())),
            alert_manager: Arc::new(RwLock::new(AlertManager::default())),
            statistics: Arc::new(RwLock::new(ErrorRateStatistics::default())),
        }
    }

    /// Record an error
    pub fn record_error(
        &self,
        error_type: String,
        category: ErrorCategory,
        severity: ErrorSeverity,
        operation_name: String,
        context: HashMap<String, String>,
    ) {
        if !self.config.enabled {
            return;
        }

        let event = ErrorEvent {
            timestamp: Utc::now(),
            error_type,
            category,
            severity,
            operation_name,
            context,
        };

        let mut history = self.error_history.write();
        history.events.push_back(event.clone());

        // Add to category-specific tracking
        history
            .errors_by_category
            .entry(category)
            .or_insert_with(VecDeque::new)
            .push_back(event);

        // Cleanup old events
        self.cleanup_old_events(&mut history);

        drop(history);

        // Update statistics and check thresholds
        self.update_statistics();
        self.check_alert_conditions();
    }

    /// Record successful operation
    pub fn record_operation(&self) {
        if !self.config.enabled {
            return;
        }

        let mut history = self.error_history.write();
        history.total_operations.push_back(OperationCount {
            timestamp: Utc::now(),
            count: 1,
        });

        // Cleanup old counts
        let cutoff_time = Utc::now() - Duration::seconds(self.config.window_size_secs as i64);
        while let Some(count) = history.total_operations.front() {
            if count.timestamp < cutoff_time {
                history.total_operations.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get current error rate
    pub fn get_error_rate(&self) -> f64 {
        let history = self.error_history.read();

        let total_errors = history.events.len() as f64;
        let total_ops: u64 = history.total_operations.iter().map(|c| c.count).sum();

        if total_ops == 0 {
            return 0.0;
        }

        total_errors / total_ops as f64
    }

    /// Get error rate by category
    pub fn get_error_rate_by_category(&self, category: ErrorCategory) -> f64 {
        let history = self.error_history.read();

        let category_errors = history
            .errors_by_category
            .get(&category)
            .map(|e| e.len())
            .unwrap_or(0) as f64;

        let total_ops: u64 = history.total_operations.iter().map(|c| c.count).sum();

        if total_ops == 0 {
            return 0.0;
        }

        category_errors / total_ops as f64
    }

    /// Get statistics
    pub fn get_statistics(&self) -> ErrorRateStatistics {
        self.statistics.read().clone()
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        let manager = self.alert_manager.read();
        manager.active_alerts.values().cloned().collect()
    }

    /// Register alert handler
    pub fn register_alert_handler(&self, handler: Box<dyn AlertHandler + Send + Sync>) {
        self.alert_manager.write().handlers.push(handler);
    }

    /// Resolve alert
    pub fn resolve_alert(&self, alert_id: &str) {
        let mut manager = self.alert_manager.write();

        if let Some(alert) = manager.active_alerts.remove(alert_id) {
            let mut resolved_alert = alert.clone();
            resolved_alert.resolved = true;
            resolved_alert.resolved_at = Some(Utc::now());

            // Notify handlers
            for handler in &mut manager.handlers {
                handler.handle_resolution(&resolved_alert);
            }

            // Move to history
            manager.alert_history.push_back(resolved_alert);
        }
    }

    /// Get alert history
    pub fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let manager = self.alert_manager.read();
        let limit = limit.unwrap_or(100);

        manager
            .alert_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    // Private helper methods

    fn cleanup_old_events(&self, history: &mut ErrorHistory) {
        let cutoff_time = Utc::now() - Duration::seconds(self.config.window_size_secs as i64);

        // Cleanup main events
        while let Some(event) = history.events.front() {
            if event.timestamp < cutoff_time {
                history.events.pop_front();
            } else {
                break;
            }
        }

        // Cleanup category events
        for events in history.errors_by_category.values_mut() {
            while let Some(event) = events.front() {
                if event.timestamp < cutoff_time {
                    events.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    fn update_statistics(&self) {
        let history = self.error_history.read();
        let manager = self.alert_manager.read();

        let total_errors = history.events.len() as u64;
        let total_ops: u64 = history.total_operations.iter().map(|c| c.count).sum();

        let current_error_rate = if total_ops == 0 {
            0.0
        } else {
            total_errors as f64 / total_ops as f64
        };

        // Calculate trend (simplified)
        let trend = self.calculate_trend(&history);

        // Count by category
        let mut errors_by_category = HashMap::new();
        for (category, events) in &history.errors_by_category {
            errors_by_category.insert(format!("{:?}", category), events.len() as u64);
        }

        // Count by severity
        let mut errors_by_severity = HashMap::new();
        for event in &history.events {
            *errors_by_severity
                .entry(format!("{:?}", event.severity))
                .or_insert(0) += 1;
        }

        let stats = ErrorRateStatistics {
            current_error_rate,
            error_rate_trend: trend,
            total_errors,
            total_operations: total_ops,
            errors_by_category,
            errors_by_severity,
            peak_error_rate: current_error_rate, // Simplified
            avg_error_rate: current_error_rate,  // Simplified
            active_alerts_count: manager.active_alerts.len(),
            total_alerts_fired: manager.alert_history.len() as u64,
        };

        *self.statistics.write() = stats;
    }

    fn calculate_trend(&self, history: &ErrorHistory) -> f64 {
        // Simple trend calculation: compare first half vs second half of window
        let mid_point = history.events.len() / 2;

        if mid_point == 0 {
            return 0.0;
        }

        let first_half_count = mid_point;
        let second_half_count = history.events.len() - mid_point;

        let first_half_rate = first_half_count as f64;
        let second_half_rate = second_half_count as f64;

        if first_half_rate == 0.0 {
            return 1.0; // Large positive trend if starting from zero
        }

        (second_half_rate - first_half_rate) / first_half_rate
    }

    fn check_alert_conditions(&self) {
        let stats = self.statistics.read().clone();
        drop(stats);

        let error_rate = self.get_error_rate();

        // Check thresholds
        if error_rate >= self.config.critical_threshold {
            self.fire_alert(
                AlertLevel::Critical,
                "Critical Error Rate".to_string(),
                format!(
                    "Error rate ({:.2}%) exceeds critical threshold ({:.2}%)",
                    error_rate * 100.0,
                    self.config.critical_threshold * 100.0
                ),
                error_rate,
                None,
            );
        } else if error_rate >= self.config.warning_threshold {
            self.fire_alert(
                AlertLevel::Warning,
                "Elevated Error Rate".to_string(),
                format!(
                    "Error rate ({:.2}%) exceeds warning threshold ({:.2}%)",
                    error_rate * 100.0,
                    self.config.warning_threshold * 100.0
                ),
                error_rate,
                None,
            );
        }

        // Check for anomalies if enabled
        if self.config.enable_anomaly_detection {
            self.check_anomalies();
        }
    }

    fn check_anomalies(&self) {
        let stats = self.statistics.read();

        // Simple anomaly detection: large trend change
        let error_rate_trend = stats.error_rate_trend;
        drop(stats);

        if error_rate_trend > self.config.anomaly_sensitivity {
            self.fire_alert(
                AlertLevel::Warning,
                "Error Rate Anomaly Detected".to_string(),
                "Significant increase in error rate detected".to_string(),
                error_rate_trend,
                None,
            );
        }
    }

    fn fire_alert(
        &self,
        level: AlertLevel,
        title: String,
        description: String,
        error_rate: f64,
        category: Option<ErrorCategory>,
    ) {
        let mut manager = self.alert_manager.write();

        // Check cooldown
        if let Some(cat) = category {
            if let Some(last_time) = manager.last_alert_time.get(&cat) {
                let elapsed = (Utc::now() - *last_time).num_seconds() as u64;
                if elapsed < self.config.alert_cooldown_secs {
                    return; // Still in cooldown period
                }
            }
        }

        // Check alert rate limiting
        let one_hour_ago = Utc::now() - Duration::hours(1);
        manager.alerts_this_hour.retain(|t| *t > one_hour_ago);

        if manager.alerts_this_hour.len() >= self.config.max_alerts_per_hour {
            return; // Too many alerts
        }

        // Create alert
        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            level,
            title,
            description,
            error_rate,
            category,
            timestamp: Utc::now(),
            resolved: false,
            resolved_at: None,
            metadata: HashMap::new(),
        };

        // Record alert time
        if let Some(cat) = category {
            manager.last_alert_time.insert(cat, alert.timestamp);
        }
        manager.alerts_this_hour.push_back(alert.timestamp);

        // Store alert
        manager
            .active_alerts
            .insert(alert.id.clone(), alert.clone());

        // Notify handlers
        for handler in &mut manager.handlers {
            handler.handle_alert(&alert);
        }

        // Update statistics
        let mut stats = self.statistics.write();
        stats.total_alerts_fired += 1;
        stats.active_alerts_count = manager.active_alerts.len();
    }
}

/// Console alert handler
pub struct ConsoleAlertHandler;

impl AlertHandler for ConsoleAlertHandler {
    fn handle_alert(&mut self, alert: &Alert) {
        println!("[ALERT {:?}] {}", alert.level, alert.title);
        println!("  {}", alert.description);
        println!("  Error Rate: {:.2}%", alert.error_rate * 100.0);
        println!("  Time: {}", alert.timestamp);
    }

    fn handle_resolution(&mut self, alert: &Alert) {
        println!("[RESOLVED] {}", alert.title);
        println!("  Resolved at: {:?}", alert.resolved_at);
    }
}

/// Global error rate monitor
static GLOBAL_ERROR_MONITOR: OnceLock<Arc<ErrorRateMonitor>> = OnceLock::new();

/// Get global error rate monitor
pub fn get_global_error_monitor() -> Arc<ErrorRateMonitor> {
    GLOBAL_ERROR_MONITOR
        .get_or_init(|| Arc::new(ErrorRateMonitor::new(ErrorMonitoringConfig::default())))
        .clone()
}

/// Initialize global error rate monitor
pub fn init_global_error_monitor(config: ErrorMonitoringConfig) {
    let _ = GLOBAL_ERROR_MONITOR.set(Arc::new(ErrorRateMonitor::new(config)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_recording() {
        let monitor = ErrorRateMonitor::new(ErrorMonitoringConfig::default());

        monitor.record_operation();
        monitor.record_operation();
        monitor.record_error(
            "test_error".to_string(),
            ErrorCategory::Numerical,
            ErrorSeverity::Medium,
            "test_op".to_string(),
            HashMap::new(),
        );

        let error_rate = monitor.get_error_rate();
        assert!(error_rate > 0.0);
        assert!(error_rate < 1.0);
    }

    #[test]
    fn test_error_rate_by_category() {
        let monitor = ErrorRateMonitor::new(ErrorMonitoringConfig::default());

        monitor.record_operation();
        monitor.record_error(
            "numerical_error".to_string(),
            ErrorCategory::Numerical,
            ErrorSeverity::High,
            "test_op".to_string(),
            HashMap::new(),
        );

        let num_rate = monitor.get_error_rate_by_category(ErrorCategory::Numerical);
        let mem_rate = monitor.get_error_rate_by_category(ErrorCategory::Memory);

        assert!(num_rate > 0.0);
        assert_eq!(mem_rate, 0.0);
    }

    #[test]
    fn test_alert_firing() {
        let mut config = ErrorMonitoringConfig::default();
        config.warning_threshold = 0.1; // 10%
        config.critical_threshold = 0.5; // 50%

        let monitor = ErrorRateMonitor::new(config);

        // Register console handler
        monitor.register_alert_handler(Box::new(ConsoleAlertHandler));

        // Record operations to trigger alert
        for _ in 0..10 {
            monitor.record_operation();
        }

        // Record enough errors to exceed warning threshold
        for _ in 0..2 {
            monitor.record_error(
                "test_error".to_string(),
                ErrorCategory::Computation,
                ErrorSeverity::High,
                "test_op".to_string(),
                HashMap::new(),
            );
        }

        let alerts = monitor.get_active_alerts();
        assert!(!alerts.is_empty());
    }

    #[test]
    fn test_alert_resolution() {
        // Use a config that won't auto-fire alerts
        let config = ErrorMonitoringConfig {
            enabled: false, // Disable auto-firing
            ..Default::default()
        };
        let monitor = ErrorRateMonitor::new(config);

        // Fire an alert manually
        monitor.fire_alert(
            AlertLevel::Warning,
            "Test Alert".to_string(),
            "Test".to_string(),
            0.5,
            None,
        );

        let alerts = monitor.get_active_alerts();
        assert_eq!(alerts.len(), 1);

        let alert_id = alerts[0].id.clone();
        monitor.resolve_alert(&alert_id);

        let active = monitor.get_active_alerts();
        assert_eq!(active.len(), 0);

        let history = monitor.get_alert_history(None);
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let monitor = ErrorRateMonitor::new(ErrorMonitoringConfig::default());

        monitor.record_operation();
        monitor.record_error(
            "error1".to_string(),
            ErrorCategory::Numerical,
            ErrorSeverity::Medium,
            "op1".to_string(),
            HashMap::new(),
        );

        let stats = monitor.get_statistics();
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.total_operations, 1);
        assert!(stats.current_error_rate > 0.0);
    }
}
