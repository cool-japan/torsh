// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Anomaly Alerting System for Autograd
//!
//! This module provides a comprehensive alerting system for autograd anomalies,
//! enabling real-time notifications, configurable alert policies, and multi-channel
//! alert delivery.
//!
//! # Features
//!
//! - **Real-time Detection**: Immediate anomaly detection and alerting
//! - **Configurable Policies**: Customizable alert thresholds and conditions
//! - **Multi-channel Delivery**: Support for logging, callbacks, webhooks, and more
//! - **Alert Aggregation**: Prevent alert fatigue through intelligent aggregation
//! - **Alert History**: Track and analyze alert patterns over time
//! - **Severity Levels**: Critical, Warning, Info for prioritized response

use crate::error_handling::{AutogradError, AutogradResult};

/// Type of anomaly detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    /// NaN value detected
    NaN,
    /// Infinity value detected
    Infinity,
    /// Gradient explosion
    GradientExplosion,
    /// Gradient vanishing
    GradientVanishing,
    /// Numerical instability
    NumericalInstability,
    /// Other anomaly
    Other,
}

/// Tensor statistics at time of anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorStatistics {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
}
use chrono::{DateTime, Duration, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert - attention needed
    Warning,
    /// Critical alert - immediate action required
    Critical,
}

/// Autograd anomaly alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlert {
    /// Unique alert ID
    pub id: u64,

    /// Alert timestamp
    pub timestamp: DateTime<Utc>,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Anomaly type that triggered the alert
    pub anomaly_type: AnomalyType,

    /// Operation that caused the anomaly
    pub operation: String,

    /// Tensor ID (if applicable)
    pub tensor_id: Option<String>,

    /// Anomaly details
    pub details: String,

    /// Tensor statistics at time of anomaly
    pub statistics: Option<TensorStatistics>,

    /// Alert metadata
    pub metadata: HashMap<String, String>,

    /// Whether this alert has been acknowledged
    pub acknowledged: bool,
}

impl AnomalyAlert {
    /// Create a new alert
    pub fn new(
        id: u64,
        severity: AlertSeverity,
        anomaly_type: AnomalyType,
        operation: String,
    ) -> Self {
        Self {
            id,
            timestamp: Utc::now(),
            severity,
            anomaly_type,
            operation,
            tensor_id: None,
            details: String::new(),
            statistics: None,
            metadata: HashMap::new(),
            acknowledged: false,
        }
    }

    /// Set alert details
    pub fn with_details(mut self, details: String) -> Self {
        self.details = details;
        self
    }

    /// Set tensor ID
    pub fn with_tensor_id(mut self, tensor_id: String) -> Self {
        self.tensor_id = Some(tensor_id);
        self
    }

    /// Set tensor statistics
    pub fn with_statistics(mut self, statistics: TensorStatistics) -> Self {
        self.statistics = Some(statistics);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Acknowledge the alert
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }

    /// Format alert as human-readable string
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("🚨 ALERT #{} [{:?}]\n", self.id, self.severity));
        output.push_str(&format!("Time: {}\n", self.timestamp));
        output.push_str(&format!("Type: {:?}\n", self.anomaly_type));
        output.push_str(&format!("Operation: {}\n", self.operation));

        if let Some(tensor_id) = &self.tensor_id {
            output.push_str(&format!("Tensor: {}\n", tensor_id));
        }

        if !self.details.is_empty() {
            output.push_str(&format!("Details: {}\n", self.details));
        }

        if let Some(stats) = &self.statistics {
            output.push_str("Statistics:\n");
            output.push_str(&format!("  Min: {:.6e}\n", stats.min));
            output.push_str(&format!("  Max: {:.6e}\n", stats.max));
            output.push_str(&format!("  Mean: {:.6e}\n", stats.mean));
            output.push_str(&format!("  Std: {:.6e}\n", stats.std));
        }

        if !self.metadata.is_empty() {
            output.push_str("Metadata:\n");
            for (key, value) in &self.metadata {
                output.push_str(&format!("  {}: {}\n", key, value));
            }
        }

        output
    }
}

/// Alert policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertPolicy {
    /// Whether alerting is enabled
    pub enabled: bool,

    /// Minimum severity level to trigger alerts
    pub min_severity: AlertSeverity,

    /// Alert rate limiting (max alerts per time window)
    pub rate_limit: Option<RateLimit>,

    /// Alert aggregation window
    pub aggregation_window: Option<Duration>,

    /// Whether to deduplicate similar alerts
    pub deduplicate: bool,

    /// Maximum alert history size
    pub max_history_size: usize,

    /// Specific anomaly types to alert on (empty = all types)
    pub anomaly_types: Vec<AnomalyType>,
}

impl Default for AlertPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            min_severity: AlertSeverity::Warning,
            rate_limit: Some(RateLimit {
                max_alerts: 100,
                time_window: Duration::minutes(1),
            }),
            aggregation_window: Some(Duration::seconds(5)),
            deduplicate: true,
            max_history_size: 1000,
            anomaly_types: Vec::new(), // Alert on all types
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Maximum number of alerts in time window
    pub max_alerts: usize,

    /// Time window for rate limiting
    pub time_window: Duration,
}

/// Alert channel for delivering alerts
pub trait AlertChannel: Send + Sync {
    /// Send an alert through this channel
    fn send(&self, alert: &AnomalyAlert) -> AutogradResult<()>;

    /// Get channel name
    fn name(&self) -> &str;
}

/// Log-based alert channel
pub struct LogAlertChannel {
    name: String,
}

impl LogAlertChannel {
    /// Create a new log alert channel
    pub fn new() -> Self {
        Self {
            name: "log".to_string(),
        }
    }
}

impl Default for LogAlertChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertChannel for LogAlertChannel {
    fn send(&self, alert: &AnomalyAlert) -> AutogradResult<()> {
        match alert.severity {
            AlertSeverity::Critical => {
                tracing::error!("{}", alert.format());
            }
            AlertSeverity::Warning => {
                tracing::warn!("{}", alert.format());
            }
            AlertSeverity::Info => {
                tracing::info!("{}", alert.format());
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Callback-based alert channel
pub struct CallbackAlertChannel {
    name: String,
    callback: Arc<dyn Fn(&AnomalyAlert) + Send + Sync>,
}

impl CallbackAlertChannel {
    /// Create a new callback alert channel
    pub fn new<F>(name: String, callback: F) -> Self
    where
        F: Fn(&AnomalyAlert) + Send + Sync + 'static,
    {
        Self {
            name,
            callback: Arc::new(callback),
        }
    }
}

impl AlertChannel for CallbackAlertChannel {
    fn send(&self, alert: &AnomalyAlert) -> AutogradResult<()> {
        (self.callback)(alert);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Alert statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStatistics {
    /// Total number of alerts generated
    pub total_alerts: usize,

    /// Alerts by severity
    pub alerts_by_severity: HashMap<AlertSeverity, usize>,

    /// Alerts by anomaly type
    pub alerts_by_type: HashMap<String, usize>,

    /// Number of acknowledged alerts
    pub acknowledged_alerts: usize,

    /// Number of suppressed alerts (rate limiting)
    pub suppressed_alerts: usize,

    /// Number of aggregated alerts
    pub aggregated_alerts: usize,
}

impl Default for AlertStatistics {
    fn default() -> Self {
        Self {
            total_alerts: 0,
            alerts_by_severity: HashMap::new(),
            alerts_by_type: HashMap::new(),
            acknowledged_alerts: 0,
            suppressed_alerts: 0,
            aggregated_alerts: 0,
        }
    }
}

/// Anomaly alert manager
pub struct AnomalyAlertManager {
    /// Alert policy
    policy: Arc<RwLock<AlertPolicy>>,

    /// Alert channels
    channels: Arc<RwLock<Vec<Arc<dyn AlertChannel>>>>,

    /// Alert history
    history: Arc<Mutex<VecDeque<AnomalyAlert>>>,

    /// Alert statistics
    statistics: Arc<Mutex<AlertStatistics>>,

    /// Next alert ID
    next_alert_id: Arc<Mutex<u64>>,

    /// Pending alerts for aggregation
    pending_alerts: Arc<Mutex<Vec<AnomalyAlert>>>,

    /// Rate limiting tracker
    rate_limit_tracker: Arc<Mutex<VecDeque<DateTime<Utc>>>>,
}

impl AnomalyAlertManager {
    /// Create a new alert manager
    pub fn new(policy: AlertPolicy) -> Self {
        let manager = Self {
            policy: Arc::new(RwLock::new(policy)),
            channels: Arc::new(RwLock::new(Vec::new())),
            history: Arc::new(Mutex::new(VecDeque::new())),
            statistics: Arc::new(Mutex::new(AlertStatistics::default())),
            next_alert_id: Arc::new(Mutex::new(1)), // Start IDs from 1
            pending_alerts: Arc::new(Mutex::new(Vec::new())),
            rate_limit_tracker: Arc::new(Mutex::new(VecDeque::new())),
        };

        // Add default log channel
        manager.add_channel(Arc::new(LogAlertChannel::new()));

        manager
    }

    /// Add an alert channel
    pub fn add_channel(&self, channel: Arc<dyn AlertChannel>) {
        self.channels.write().push(channel);
    }

    /// Remove an alert channel by name
    pub fn remove_channel(&self, name: &str) {
        self.channels.write().retain(|c| c.name() != name);
    }

    /// Update alert policy
    pub fn update_policy(&self, policy: AlertPolicy) {
        *self.policy.write() = policy;
    }

    /// Generate an alert
    pub fn alert(
        &self,
        severity: AlertSeverity,
        anomaly_type: AnomalyType,
        operation: String,
        details: String,
    ) -> AutogradResult<u64> {
        let policy = self.policy.read();

        // Check if alerting is enabled
        if !policy.enabled {
            return Ok(0);
        }

        // Check severity threshold
        if severity < policy.min_severity {
            return Ok(0);
        }

        // Check anomaly type filter
        if !policy.anomaly_types.is_empty() && !policy.anomaly_types.contains(&anomaly_type) {
            return Ok(0);
        }

        // Check rate limiting
        if let Some(rate_limit) = &policy.rate_limit {
            if !self.check_rate_limit(rate_limit) {
                let mut stats = self.statistics.lock();
                stats.suppressed_alerts += 1;
                return Ok(0);
            }
        }

        // Generate alert ID
        let alert_id = {
            let mut next_id = self.next_alert_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Create alert
        let alert =
            AnomalyAlert::new(alert_id, severity, anomaly_type, operation).with_details(details);

        // Handle aggregation
        if let Some(window) = policy.aggregation_window {
            self.aggregate_alert(alert, window)?;
        } else {
            self.send_alert(alert)?;
        }

        Ok(alert_id)
    }

    /// Check rate limiting
    fn check_rate_limit(&self, rate_limit: &RateLimit) -> bool {
        let mut tracker = self.rate_limit_tracker.lock();
        let now = Utc::now();

        // Remove old entries
        let cutoff = now - rate_limit.time_window;
        while let Some(&timestamp) = tracker.front() {
            if timestamp < cutoff {
                tracker.pop_front();
            } else {
                break;
            }
        }

        // Check if under limit
        if tracker.len() >= rate_limit.max_alerts {
            return false;
        }

        // Add new entry
        tracker.push_back(now);
        true
    }

    /// Aggregate similar alerts
    fn aggregate_alert(&self, alert: AnomalyAlert, window: Duration) -> AutogradResult<()> {
        let mut pending = self.pending_alerts.lock();

        // Check for similar recent alerts
        if self.policy.read().deduplicate {
            if let Some(_similar) = pending.iter().find(|a| {
                a.anomaly_type == alert.anomaly_type
                    && a.operation == alert.operation
                    && (alert.timestamp - a.timestamp) < window
            }) {
                // Similar alert found - increment aggregation counter
                let mut stats = self.statistics.lock();
                stats.aggregated_alerts += 1;
                return Ok(());
            }
        }

        pending.push(alert.clone());

        // Flush old pending alerts
        let cutoff = Utc::now() - window;
        let mut to_send = Vec::new();
        let mut i = 0;
        while i < pending.len() {
            if pending[i].timestamp < cutoff {
                to_send.push(pending.remove(i));
            } else {
                i += 1;
            }
        }
        drop(pending);

        for alert in to_send {
            self.send_alert(alert)?;
        }

        Ok(())
    }

    /// Send alert through all channels
    fn send_alert(&self, alert: AnomalyAlert) -> AutogradResult<()> {
        // Update statistics
        {
            let mut stats = self.statistics.lock();
            stats.total_alerts += 1;
            *stats.alerts_by_severity.entry(alert.severity).or_insert(0) += 1;
            *stats
                .alerts_by_type
                .entry(format!("{:?}", alert.anomaly_type))
                .or_insert(0) += 1;
        }

        // Add to history
        {
            let mut history = self.history.lock();
            let max_size = self.policy.read().max_history_size;

            history.push_back(alert.clone());

            while history.len() > max_size {
                history.pop_front();
            }
        }

        // Send through all channels
        let channels = self.channels.read();
        for channel in channels.iter() {
            if let Err(e) = channel.send(&alert) {
                tracing::warn!(
                    "Failed to send alert through channel {}: {}",
                    channel.name(),
                    e
                );
            }
        }

        Ok(())
    }

    /// Get alert history
    pub fn history(&self) -> Vec<AnomalyAlert> {
        self.history.lock().iter().cloned().collect()
    }

    /// Get recent alerts (last N)
    pub fn recent_alerts(&self, count: usize) -> Vec<AnomalyAlert> {
        let history = self.history.lock();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Get alerts by severity
    pub fn alerts_by_severity(&self, severity: AlertSeverity) -> Vec<AnomalyAlert> {
        self.history
            .lock()
            .iter()
            .filter(|a| a.severity == severity)
            .cloned()
            .collect()
    }

    /// Get unacknowledged alerts
    pub fn unacknowledged_alerts(&self) -> Vec<AnomalyAlert> {
        self.history
            .lock()
            .iter()
            .filter(|a| !a.acknowledged)
            .cloned()
            .collect()
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&self, alert_id: u64) -> AutogradResult<()> {
        let mut history = self.history.lock();
        if let Some(alert) = history.iter_mut().find(|a| a.id == alert_id) {
            alert.acknowledge();
            let mut stats = self.statistics.lock();
            stats.acknowledged_alerts += 1;
            Ok(())
        } else {
            Err(AutogradError::Configuration {
                parameter: "alert_id".to_string(),
                value: alert_id.to_string(),
                reason: "Alert not found".to_string(),
                valid_range: None,
            })
        }
    }

    /// Acknowledge all alerts
    pub fn acknowledge_all(&self) {
        let mut history = self.history.lock();
        let mut stats = self.statistics.lock();

        for alert in history.iter_mut() {
            if !alert.acknowledged {
                alert.acknowledge();
                stats.acknowledged_alerts += 1;
            }
        }
    }

    /// Get alert statistics
    pub fn statistics(&self) -> AlertStatistics {
        self.statistics.lock().clone()
    }

    /// Clear alert history
    pub fn clear_history(&self) {
        self.history.lock().clear();
        *self.statistics.lock() = AlertStatistics::default();
    }

    /// Generate alert summary
    pub fn summary(&self) -> String {
        let stats = self.statistics.lock();
        let mut output = String::new();

        output.push_str("=== Anomaly Alert Summary ===\n\n");
        output.push_str(&format!("Total alerts: {}\n", stats.total_alerts));
        output.push_str(&format!("Acknowledged: {}\n", stats.acknowledged_alerts));
        output.push_str(&format!("Suppressed: {}\n", stats.suppressed_alerts));
        output.push_str(&format!("Aggregated: {}\n\n", stats.aggregated_alerts));

        output.push_str("Alerts by severity:\n");
        let mut severity_counts: Vec<_> = stats.alerts_by_severity.iter().collect();
        severity_counts.sort_by_key(|(s, _)| *s);
        for (severity, count) in severity_counts {
            output.push_str(&format!("  {:?}: {}\n", severity, count));
        }

        output.push_str("\nAlerts by type:\n");
        let mut type_counts: Vec<_> = stats.alerts_by_type.iter().collect();
        type_counts.sort_by_key(|(_, count)| -(**count as i64));
        for (atype, count) in type_counts.iter().take(10) {
            output.push_str(&format!("  {}: {}\n", atype, count));
        }

        output
    }
}

/// Global anomaly alert manager
static GLOBAL_ALERT_MANAGER: once_cell::sync::Lazy<AnomalyAlertManager> =
    once_cell::sync::Lazy::new(|| AnomalyAlertManager::new(AlertPolicy::default()));

/// Get the global alert manager
pub fn global_alert_manager() -> &'static AnomalyAlertManager {
    &GLOBAL_ALERT_MANAGER
}

/// Convenience function to generate an alert
pub fn alert(
    severity: AlertSeverity,
    anomaly_type: AnomalyType,
    operation: String,
    details: String,
) -> AutogradResult<u64> {
    global_alert_manager().alert(severity, anomaly_type, operation, details)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = AnomalyAlert::new(
            1,
            AlertSeverity::Warning,
            AnomalyType::NaN,
            "test_op".to_string(),
        )
        .with_details("Test alert".to_string());

        assert_eq!(alert.id, 1);
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.operation, "test_op");
        assert!(!alert.acknowledged);
    }

    #[test]
    fn test_alert_manager() {
        let policy = AlertPolicy {
            aggregation_window: None, // Disable aggregation for immediate delivery
            ..Default::default()
        };
        let manager = AnomalyAlertManager::new(policy);

        let alert_id = manager
            .alert(
                AlertSeverity::Critical,
                AnomalyType::NaN,
                "matmul".to_string(),
                "NaN detected in gradient".to_string(),
            )
            .unwrap();

        assert!(alert_id > 0);

        let history = manager.history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].operation, "matmul");
    }

    #[test]
    fn test_alert_acknowledgement() {
        let policy = AlertPolicy {
            aggregation_window: None, // Disable aggregation for immediate delivery
            ..Default::default()
        };
        let manager = AnomalyAlertManager::new(policy);

        let alert_id = manager
            .alert(
                AlertSeverity::Warning,
                AnomalyType::Infinity,
                "add".to_string(),
                "Infinity detected".to_string(),
            )
            .unwrap();

        let unack = manager.unacknowledged_alerts();
        assert_eq!(unack.len(), 1);

        manager.acknowledge_alert(alert_id).unwrap();

        let unack_after = manager.unacknowledged_alerts();
        assert_eq!(unack_after.len(), 0);
    }

    #[test]
    fn test_alert_filtering() {
        let policy = AlertPolicy {
            aggregation_window: None, // Disable aggregation for immediate delivery
            min_severity: AlertSeverity::Info, // Allow all severity levels
            ..Default::default()
        };
        let manager = AnomalyAlertManager::new(policy);

        manager
            .alert(
                AlertSeverity::Info,
                AnomalyType::NaN,
                "op1".to_string(),
                "Info alert".to_string(),
            )
            .unwrap();

        manager
            .alert(
                AlertSeverity::Warning,
                AnomalyType::Infinity,
                "op2".to_string(),
                "Warning alert".to_string(),
            )
            .unwrap();

        manager
            .alert(
                AlertSeverity::Critical,
                AnomalyType::GradientExplosion,
                "op3".to_string(),
                "Critical alert".to_string(),
            )
            .unwrap();

        let critical_alerts = manager.alerts_by_severity(AlertSeverity::Critical);
        assert_eq!(critical_alerts.len(), 1);

        let warning_alerts = manager.alerts_by_severity(AlertSeverity::Warning);
        assert_eq!(warning_alerts.len(), 1);
    }

    #[test]
    fn test_callback_channel() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let channel = Arc::new(CallbackAlertChannel::new(
            "test_callback".to_string(),
            move |_alert| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            },
        ));

        let policy = AlertPolicy {
            aggregation_window: None, // Disable aggregation for immediate delivery
            ..Default::default()
        };
        let manager = AnomalyAlertManager::new(policy);
        manager.add_channel(channel);

        manager
            .alert(
                AlertSeverity::Warning,
                AnomalyType::NaN,
                "test".to_string(),
                "Test".to_string(),
            )
            .unwrap();

        // Give a moment for async processing
        std::thread::sleep(std::time::Duration::from_millis(10));

        assert!(counter.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_alert_statistics() {
        let policy = AlertPolicy {
            aggregation_window: None, // Disable aggregation for immediate delivery
            ..Default::default()
        };
        let manager = AnomalyAlertManager::new(policy);

        for i in 0..5 {
            manager
                .alert(
                    AlertSeverity::Warning,
                    AnomalyType::NaN,
                    format!("op_{}", i),
                    "Test".to_string(),
                )
                .unwrap();
        }

        let stats = manager.statistics();
        assert_eq!(stats.total_alerts, 5);
        assert_eq!(
            *stats
                .alerts_by_severity
                .get(&AlertSeverity::Warning)
                .unwrap(),
            5
        );
    }
}
