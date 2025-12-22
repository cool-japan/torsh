//! Real-time Alerting System for Distributed Training
//!
//! This module provides a comprehensive alerting system that monitors distributed training
//! metrics and triggers alerts based on configurable rules and thresholds.
//!
//! ## Features
//!
//! - **Configurable Alert Rules**: Define custom alert conditions for any metric
//! - **Multiple Severity Levels**: Warning, Error, and Critical alerts
//! - **Alert History**: Track all triggered alerts with timestamps
//! - **Alert Acknowledgment**: Mark alerts as acknowledged to prevent duplicate notifications
//! - **Flexible Conditions**: Support for threshold, rate-of-change, and anomaly-based alerts
//! - **Integration**: Seamless integration with AdvancedMonitor and Prometheus exporter
//! - **Extensible Notifications**: Built-in logging, extensible for email/Slack/PagerDuty
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use torsh_distributed::alerting::{AlertManager, AlertRule, AlertCondition, AlertSeverity};
//! use torsh_distributed::advanced_monitoring::AdvancedMonitor;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let monitor = Arc::new(AdvancedMonitor::new(process_group));
//!     let mut alert_manager = AlertManager::new(monitor.clone());
//!
//!     // Configure alert rules
//!     alert_manager.add_rule(AlertRule {
//!         name: "high_gpu_memory".to_string(),
//!         description: "GPU memory usage exceeds 90%".to_string(),
//!         condition: AlertCondition::Threshold {
//!             metric: "gpu_memory_usage_percent".to_string(),
//!             operator: ">".to_string(),
//!             value: 90.0,
//!         },
//!         severity: AlertSeverity::Warning,
//!         cooldown_secs: 300,
//!     })?;
//!
//!     // Start monitoring
//!     alert_manager.start().await?;
//!
//!     Ok(())
//! }
//! ```

use crate::advanced_monitoring::{AdvancedMetrics, AdvancedMonitor};
use crate::{TorshDistributedError, TorshResult};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Maximum number of alerts to keep in history
const MAX_ALERT_HISTORY: usize = 1000;

/// Default check interval for alert conditions (seconds)
const DEFAULT_CHECK_INTERVAL_SECS: u64 = 10;

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert - no action required
    Info,
    /// Warning alert - should be investigated
    Warning,
    /// Error alert - requires attention
    Error,
    /// Critical alert - immediate action required
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Error => write!(f, "ERROR"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Alert condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Threshold-based condition (metric > value, metric < value, etc.)
    Threshold {
        metric: String,
        operator: String, // ">", "<", ">=", "<=", "==", "!="
        value: f64,
    },
    /// Rate of change condition
    RateOfChange {
        metric: String,
        operator: String,
        rate_per_sec: f64,
        window_secs: u64,
    },
    /// Anomaly detection condition
    Anomaly {
        metric: String,
        sensitivity: f64, // Z-score threshold
    },
    /// Custom lambda condition
    Custom { name: String, description: String },
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Unique name for the rule
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// Condition that triggers the alert
    pub condition: AlertCondition,

    /// Alert severity level
    pub severity: AlertSeverity,

    /// Cooldown period in seconds to prevent alert spam
    pub cooldown_secs: u64,
}

/// Triggered alert instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Rule name that triggered the alert
    pub rule_name: String,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Detailed message
    pub message: String,

    /// Timestamp when alert was triggered (skipped from serialization)
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,

    /// Affected rank (if applicable)
    pub rank: Option<u32>,

    /// Metric value that triggered the alert
    pub metric_value: Option<f64>,

    /// Whether the alert has been acknowledged
    #[serde(skip, default)]
    pub acknowledged: bool,
}

/// Alert notification handler trait
pub trait AlertNotifier: Send + Sync {
    /// Handle a triggered alert
    fn notify(&self, alert: &Alert) -> TorshResult<()>;
}

/// Built-in logging notifier
pub struct LoggingNotifier;

impl AlertNotifier for LoggingNotifier {
    fn notify(&self, alert: &Alert) -> TorshResult<()> {
        let rank_info = alert
            .rank
            .map(|r| format!(" [Rank {}]", r))
            .unwrap_or_default();

        let value_info = alert
            .metric_value
            .map(|v| format!(" (value: {:.2})", v))
            .unwrap_or_default();

        match alert.severity {
            AlertSeverity::Info => info!(
                "🔔 ALERT [{}]{}: {}{}",
                alert.rule_name, rank_info, alert.message, value_info
            ),
            AlertSeverity::Warning => warn!(
                "⚠️  ALERT [{}]{}: {}{}",
                alert.rule_name, rank_info, alert.message, value_info
            ),
            AlertSeverity::Error => error!(
                "❌ ALERT [{}]{}: {}{}",
                alert.rule_name, rank_info, alert.message, value_info
            ),
            AlertSeverity::Critical => error!(
                "🚨 CRITICAL ALERT [{}]{}: {}{}",
                alert.rule_name, rank_info, alert.message, value_info
            ),
        }

        Ok(())
    }
}

/// Alert manager for distributed training
pub struct AlertManager {
    /// Reference to the monitoring system
    monitor: Arc<AdvancedMonitor>,

    /// Configured alert rules
    rules: Arc<RwLock<HashMap<String, AlertRule>>>,

    /// Alert history
    history: Arc<RwLock<VecDeque<Alert>>>,

    /// Last trigger time for each rule (for cooldown)
    last_trigger: Arc<RwLock<HashMap<String, Instant>>>,

    /// Alert notifiers
    notifiers: Arc<RwLock<Vec<Box<dyn AlertNotifier>>>>,

    /// Check interval in seconds
    check_interval_secs: u64,

    /// Whether the alert manager is running
    running: Arc<RwLock<bool>>,

    /// Statistics
    stats: Arc<RwLock<AlertStats>>,
}

#[derive(Debug, Default)]
struct AlertStats {
    total_alerts: u64,
    alerts_by_severity: HashMap<String, u64>,
    alerts_by_rule: HashMap<String, u64>,
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(monitor: Arc<AdvancedMonitor>) -> Self {
        let notifiers: Vec<Box<dyn AlertNotifier>> = vec![Box::new(LoggingNotifier)];

        Self {
            monitor,
            rules: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            last_trigger: Arc::new(RwLock::new(HashMap::new())),
            notifiers: Arc::new(RwLock::new(notifiers)),
            check_interval_secs: DEFAULT_CHECK_INTERVAL_SECS,
            running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(AlertStats::default())),
        }
    }

    /// Add a new alert rule
    pub fn add_rule(&mut self, rule: AlertRule) -> TorshResult<()> {
        let mut rules = self.rules.write();
        if rules.contains_key(&rule.name) {
            return Err(TorshDistributedError::configuration_error(format!(
                "Alert rule '{}' already exists",
                rule.name
            )));
        }
        let rule_name = rule.name.clone();
        rules.insert(rule.name.clone(), rule);
        info!("Added alert rule: {}", rule_name);
        Ok(())
    }

    /// Remove an alert rule
    pub fn remove_rule(&mut self, name: &str) -> TorshResult<()> {
        let mut rules = self.rules.write();
        rules.remove(name).ok_or_else(|| {
            TorshDistributedError::configuration_error(format!("Alert rule '{}' not found", name))
        })?;
        info!("Removed alert rule: {}", name);
        Ok(())
    }

    /// Set check interval
    pub fn set_check_interval(&mut self, secs: u64) {
        self.check_interval_secs = secs;
    }

    /// Add a custom notifier
    pub fn add_notifier(&mut self, notifier: Box<dyn AlertNotifier>) {
        self.notifiers.write().push(notifier);
    }

    /// Start the alert manager
    pub async fn start(&self) -> TorshResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(TorshDistributedError::configuration_error(
                    "Alert manager is already running",
                ));
            }
            *running = true;
        }

        info!(
            "Alert manager started with {} rules",
            self.rules.read().len()
        );

        let manager = self.clone_for_task();
        tokio::spawn(async move {
            manager.run().await;
        });

        Ok(())
    }

    /// Stop the alert manager
    pub fn stop(&self) {
        *self.running.write() = false;
        info!("Alert manager stopped");
    }

    /// Clone for async task
    fn clone_for_task(&self) -> Self {
        Self {
            monitor: Arc::clone(&self.monitor),
            rules: Arc::clone(&self.rules),
            history: Arc::clone(&self.history),
            last_trigger: Arc::clone(&self.last_trigger),
            notifiers: Arc::clone(&self.notifiers),
            check_interval_secs: self.check_interval_secs,
            running: Arc::clone(&self.running),
            stats: Arc::clone(&self.stats),
        }
    }

    /// Main monitoring loop
    async fn run(&self) {
        let interval = Duration::from_secs(self.check_interval_secs);

        while *self.running.read() {
            if let Err(e) = self.check_all_rules().await {
                error!("Error checking alert rules: {}", e);
            }

            tokio::time::sleep(interval).await;
        }
    }

    /// Check all alert rules
    async fn check_all_rules(&self) -> TorshResult<()> {
        let rules = self.rules.read().clone();

        for (name, rule) in rules.iter() {
            // Check cooldown
            if let Some(last_time) = self.last_trigger.read().get(name) {
                if last_time.elapsed().as_secs() < rule.cooldown_secs {
                    continue;
                }
            }

            // Evaluate condition
            if let Some(alert) = self.evaluate_rule(rule).await? {
                self.trigger_alert(alert)?;
            }
        }

        Ok(())
    }

    /// Evaluate a single rule
    async fn evaluate_rule(&self, rule: &AlertRule) -> TorshResult<Option<Alert>> {
        match &rule.condition {
            AlertCondition::Threshold {
                metric,
                operator,
                value,
            } => {
                self.evaluate_threshold(rule, metric, operator, *value)
                    .await
            }
            AlertCondition::RateOfChange {
                metric,
                operator,
                rate_per_sec,
                window_secs,
            } => {
                self.evaluate_rate_of_change(rule, metric, operator, *rate_per_sec, *window_secs)
                    .await
            }
            AlertCondition::Anomaly {
                metric,
                sensitivity,
            } => self.evaluate_anomaly(rule, metric, *sensitivity).await,
            AlertCondition::Custom { .. } => {
                // Custom conditions would be handled by user-defined logic
                Ok(None)
            }
        }
    }

    /// Evaluate threshold condition
    async fn evaluate_threshold(
        &self,
        rule: &AlertRule,
        metric: &str,
        operator: &str,
        threshold: f64,
    ) -> TorshResult<Option<Alert>> {
        let metrics = self.monitor.get_latest_metrics().await?;

        for (rank, metric_data) in metrics {
            if let Some(value) = self.extract_metric_value(metric, &metric_data) {
                let triggered = match operator {
                    ">" => value > threshold,
                    "<" => value < threshold,
                    ">=" => value >= threshold,
                    "<=" => value <= threshold,
                    "==" => (value - threshold).abs() < 1e-6,
                    "!=" => (value - threshold).abs() >= 1e-6,
                    _ => false,
                };

                if triggered {
                    return Ok(Some(Alert {
                        rule_name: rule.name.clone(),
                        severity: rule.severity,
                        message: format!(
                            "{}: {} {} {} (current: {:.2})",
                            rule.description, metric, operator, threshold, value
                        ),
                        timestamp: Instant::now(),
                        rank: Some(rank),
                        metric_value: Some(value),
                        acknowledged: false,
                    }));
                }
            }
        }

        Ok(None)
    }

    /// Evaluate rate of change condition
    async fn evaluate_rate_of_change(
        &self,
        rule: &AlertRule,
        metric: &str,
        operator: &str,
        rate_threshold: f64,
        window_secs: u64,
    ) -> TorshResult<Option<Alert>> {
        // Get historical metrics for rate calculation
        let metrics = self.monitor.get_latest_metrics().await?;

        for (rank, _current_data) in metrics {
            // Get historical data for this rank
            if let Some(history) = self.monitor.get_rank_history(rank) {
                if history.len() < 2 {
                    continue;
                }

                // Calculate rate of change over the window
                let window_duration = Duration::from_secs(window_secs);

                let recent_values: Vec<(Duration, f64)> = history
                    .iter()
                    .rev()
                    .filter_map(|m| {
                        // m.timestamp is a Duration since start of monitoring
                        if let Some(latest_timestamp) = history.last().map(|h| h.timestamp) {
                            let age = latest_timestamp.saturating_sub(m.timestamp);
                            if age <= window_duration {
                                self.extract_metric_value(metric, m).map(|v| (age, v))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect();

                if recent_values.len() >= 2 {
                    let oldest = recent_values.last().unwrap();
                    let newest = recent_values.first().unwrap();
                    let time_diff = (newest.0.as_secs_f64() - oldest.0.as_secs_f64()).max(1.0);
                    let value_diff = newest.1 - oldest.1;
                    let rate = value_diff / time_diff;

                    let triggered = match operator {
                        ">" => rate > rate_threshold,
                        "<" => rate < rate_threshold,
                        ">=" => rate >= rate_threshold,
                        "<=" => rate <= rate_threshold,
                        _ => false,
                    };

                    if triggered {
                        return Ok(Some(Alert {
                            rule_name: rule.name.clone(),
                            severity: rule.severity,
                            message: format!(
                                "{}: {} rate {} {} per second (current: {:.2}/s)",
                                rule.description, metric, operator, rate_threshold, rate
                            ),
                            timestamp: Instant::now(),
                            rank: Some(rank),
                            metric_value: Some(rate),
                            acknowledged: false,
                        }));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Evaluate anomaly detection condition
    async fn evaluate_anomaly(
        &self,
        rule: &AlertRule,
        _metric: &str,
        sensitivity: f64,
    ) -> TorshResult<Option<Alert>> {
        let anomalies = self.monitor.get_recent_anomalies(10);

        // Check if any recent anomalies match our criteria
        for anomaly in anomalies {
            if anomaly.severity >= (sensitivity * 10.0) as u8 {
                return Ok(Some(Alert {
                    rule_name: rule.name.clone(),
                    severity: rule.severity,
                    message: format!("{}: {}", rule.description, anomaly.description),
                    timestamp: Instant::now(),
                    rank: None, // Anomalies don't have a specific rank
                    metric_value: Some(anomaly.current_value),
                    acknowledged: false,
                }));
            }
        }

        Ok(None)
    }

    /// Extract metric value from AdvancedMetrics
    fn extract_metric_value(&self, metric: &str, data: &AdvancedMetrics) -> Option<f64> {
        match metric {
            // Compute metrics
            "forward_time_ms" => Some(data.compute.forward_time_ms),
            "backward_time_ms" => Some(data.compute.backward_time_ms),
            "optimizer_time_ms" => Some(data.compute.optimizer_time_ms),
            "gpu_utilization" => Some(data.compute.gpu_utilization),
            "cpu_utilization" => Some(data.compute.cpu_utilization),
            "gflops" => Some(data.compute.gflops),

            // Communication metrics
            "all_reduce_time_ms" => Some(data.communication.all_reduce_time_ms),
            "broadcast_time_ms" => Some(data.communication.broadcast_time_ms),
            "bandwidth_mbps" => Some(data.communication.bandwidth_mbps),
            "comm_comp_ratio" => Some(data.communication.comm_comp_ratio),

            // Memory metrics
            "gpu_memory_used_mb" => Some(data.memory.gpu_memory_used_mb),
            "cpu_memory_used_mb" => Some(data.memory.cpu_memory_used_mb),
            "peak_memory_mb" => Some(data.memory.peak_memory_mb),
            "gpu_memory_usage_percent" => Some(
                (data.memory.gpu_memory_used_mb / data.memory.gpu_memory_total_mb.max(1.0)) * 100.0,
            ),

            // I/O metrics
            "data_load_time_ms" => Some(data.io.data_load_time_ms),
            "disk_read_mbps" => Some(data.io.disk_read_mbps),
            "disk_write_mbps" => Some(data.io.disk_write_mbps),

            // Custom metrics
            _ => data.custom.get(metric).copied(),
        }
    }

    /// Trigger an alert
    fn trigger_alert(&self, alert: Alert) -> TorshResult<()> {
        // Update last trigger time
        self.last_trigger
            .write()
            .insert(alert.rule_name.clone(), Instant::now());

        // Add to history
        let mut history = self.history.write();
        history.push_back(alert.clone());
        if history.len() > MAX_ALERT_HISTORY {
            history.pop_front();
        }

        // Update statistics
        let mut stats = self.stats.write();
        stats.total_alerts += 1;
        *stats
            .alerts_by_severity
            .entry(alert.severity.to_string())
            .or_insert(0) += 1;
        *stats
            .alerts_by_rule
            .entry(alert.rule_name.clone())
            .or_insert(0) += 1;

        // Notify all handlers
        let notifiers = self.notifiers.read();
        for notifier in notifiers.iter() {
            if let Err(e) = notifier.notify(&alert) {
                error!("Failed to send alert notification: {}", e);
            }
        }

        Ok(())
    }

    /// Get alert history
    pub fn get_history(&self) -> Vec<Alert> {
        self.history.read().iter().cloned().collect()
    }

    /// Get recent alerts
    pub fn get_recent_alerts(&self, count: usize) -> Vec<Alert> {
        self.history
            .read()
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Get alerts by severity
    pub fn get_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<Alert> {
        self.history
            .read()
            .iter()
            .filter(|a| a.severity == severity)
            .cloned()
            .collect()
    }

    /// Get unacknowledged alerts
    pub fn get_unacknowledged_alerts(&self) -> Vec<Alert> {
        self.history
            .read()
            .iter()
            .filter(|a| !a.acknowledged)
            .cloned()
            .collect()
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&self, rule_name: &str, timestamp: Instant) {
        let mut history = self.history.write();
        for alert in history.iter_mut() {
            if alert.rule_name == rule_name && alert.timestamp == timestamp {
                alert.acknowledged = true;
                debug!("Acknowledged alert: {}", rule_name);
                break;
            }
        }
    }

    /// Get alert statistics
    pub fn get_statistics(&self) -> HashMap<String, u64> {
        let stats = self.stats.read();
        let mut result = HashMap::new();
        result.insert("total_alerts".to_string(), stats.total_alerts);

        for (severity, count) in &stats.alerts_by_severity {
            result.insert(format!("alerts_{}", severity.to_lowercase()), *count);
        }

        result
    }

    /// Clear alert history
    pub fn clear_history(&self) {
        self.history.write().clear();
        debug!("Cleared alert history");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced_monitoring::{
        CommunicationMetrics, ComputeMetrics, IoMetrics, MemoryMetrics,
    };
    use crate::backend::BackendType;
    use crate::init_process_group;

    #[tokio::test]
    async fn test_alert_rule_creation() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let monitor = Arc::new(AdvancedMonitor::new(Arc::new(pg)));
        let mut manager = AlertManager::new(monitor);

        let rule = AlertRule {
            name: "test_rule".to_string(),
            description: "Test rule".to_string(),
            condition: AlertCondition::Threshold {
                metric: "gpu_utilization".to_string(),
                operator: ">".to_string(),
                value: 90.0,
            },
            severity: AlertSeverity::Warning,
            cooldown_secs: 60,
        };

        assert!(manager.add_rule(rule).is_ok());
        assert_eq!(manager.rules.read().len(), 1);
    }

    #[tokio::test]
    async fn test_threshold_alert_trigger() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let monitor = Arc::new(AdvancedMonitor::new(Arc::new(pg)));

        // Record metrics that should trigger alert
        let metrics = AdvancedMetrics {
            timestamp: Duration::from_secs(0),
            compute: ComputeMetrics {
                forward_time_ms: 10.0,
                backward_time_ms: 15.0,
                optimizer_time_ms: 2.0,
                gpu_utilization: 95.0, // Above threshold
                cpu_utilization: 60.0,
                tensor_core_utilization: 75.0,
                gflops: 100.0,
            },
            communication: CommunicationMetrics {
                all_reduce_time_ms: 8.0,
                broadcast_time_ms: 3.0,
                all_gather_time_ms: 1.0,
                bandwidth_mbps: 1024.0,
                comm_comp_ratio: 0.3,
                num_operations: 100,
                avg_message_size: 10240,
            },
            memory: MemoryMetrics {
                gpu_memory_used_mb: 512.0,
                gpu_memory_total_mb: 1024.0,
                cpu_memory_used_mb: 2048.0,
                memory_bandwidth_gbps: 10.0,
                num_allocations: 50,
                peak_memory_mb: 768.0,
            },
            io: IoMetrics {
                data_load_time_ms: 20.0,
                disk_read_mbps: 100.0,
                disk_write_mbps: 50.0,
                preprocessing_time_ms: 5.0,
            },
            custom: HashMap::new(),
        };

        monitor.record_metrics(metrics).unwrap();

        let mut manager = AlertManager::new(monitor);

        let rule = AlertRule {
            name: "high_gpu_util".to_string(),
            description: "GPU utilization is high".to_string(),
            condition: AlertCondition::Threshold {
                metric: "gpu_utilization".to_string(),
                operator: ">".to_string(),
                value: 90.0,
            },
            severity: AlertSeverity::Warning,
            cooldown_secs: 0, // No cooldown for testing
        };

        manager.add_rule(rule).unwrap();

        // Check rules
        manager.check_all_rules().await.unwrap();

        // Should have triggered an alert
        let history = manager.get_history();
        assert!(!history.is_empty());
        assert_eq!(history[0].rule_name, "high_gpu_util");
        assert_eq!(history[0].severity, AlertSeverity::Warning);
    }

    #[tokio::test]
    async fn test_alert_statistics() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let monitor = Arc::new(AdvancedMonitor::new(Arc::new(pg)));
        let manager = AlertManager::new(monitor);

        // Trigger some alerts manually
        let alert1 = Alert {
            rule_name: "test1".to_string(),
            severity: AlertSeverity::Warning,
            message: "Test alert 1".to_string(),
            timestamp: Instant::now(),
            rank: Some(0),
            metric_value: Some(50.0),
            acknowledged: false,
        };

        let alert2 = Alert {
            rule_name: "test2".to_string(),
            severity: AlertSeverity::Error,
            message: "Test alert 2".to_string(),
            timestamp: Instant::now(),
            rank: Some(0),
            metric_value: Some(100.0),
            acknowledged: false,
        };

        manager.trigger_alert(alert1).unwrap();
        manager.trigger_alert(alert2).unwrap();

        let stats = manager.get_statistics();
        assert_eq!(*stats.get("total_alerts").unwrap(), 2);
        assert_eq!(*stats.get("alerts_warning").unwrap(), 1);
        assert_eq!(*stats.get("alerts_error").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_alert_acknowledgment() {
        let pg = init_process_group(BackendType::Gloo, 0, 1, "127.0.0.1", 29500)
            .await
            .unwrap();
        let monitor = Arc::new(AdvancedMonitor::new(Arc::new(pg)));
        let manager = AlertManager::new(monitor);

        let timestamp = Instant::now();
        let alert = Alert {
            rule_name: "test".to_string(),
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            timestamp,
            rank: Some(0),
            metric_value: Some(50.0),
            acknowledged: false,
        };

        manager.trigger_alert(alert).unwrap();

        let unack = manager.get_unacknowledged_alerts();
        assert_eq!(unack.len(), 1);

        manager.acknowledge_alert("test", timestamp);

        let unack_after = manager.get_unacknowledged_alerts();
        assert_eq!(unack_after.len(), 0);
    }
}
