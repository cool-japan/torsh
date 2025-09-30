//! Alert system for performance monitoring
//!
//! This module provides a comprehensive alerting system for performance monitoring,
//! including threshold-based alerts, trend analysis, and multiple notification channels.

use crate::ProfileEvent;
use anyhow::Result;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Global alert manager instance
static ALERT_MANAGER: Lazy<Arc<Mutex<AlertManager>>> =
    Lazy::new(|| Arc::new(Mutex::new(AlertManager::new())));

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
            AlertSeverity::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

/// Alert trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Threshold-based conditions
    DurationThreshold {
        operation: String,
        max_duration_ns: u64,
    },
    MemoryThreshold {
        max_memory_bytes: u64,
    },
    ThroughputThreshold {
        operation: String,
        min_throughput: f64,
    },
    ErrorRateThreshold {
        operation: String,
        max_error_rate: f64,
    },

    /// Trend-based conditions
    PerformanceDegradation {
        operation: String,
        degradation_threshold: f64,
        window_size: usize,
    },
    MemoryLeak {
        growth_threshold: f64,
        window_size: usize,
    },

    /// Anomaly detection
    StatisticalAnomaly {
        operation: String,
        sigma_threshold: f64,
    },

    /// Custom condition
    Custom {
        name: String,
        expression: String,
    },
}

/// Alert notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email {
        recipients: Vec<String>,
        smtp_config: SmtpConfig,
    },
    Slack {
        webhook_url: String,
        channel: String,
    },
    Discord {
        webhook_url: String,
    },
    PagerDuty {
        integration_key: String,
    },
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    Log {
        level: String,
        format: String,
    },
    Console,
}

/// SMTP configuration for email alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtpConfig {
    pub server: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub use_tls: bool,
}

/// Alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: String,
    pub name: String,
    pub description: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub channels: Vec<NotificationChannel>,
    pub enabled: bool,
    pub cooldown_duration: Duration,
    pub max_alerts_per_hour: u32,
    pub tags: HashMap<String, String>,
}

/// Alert instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub rule_id: String,
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub timestamp: SystemTime,
    pub resolved: bool,
    pub resolved_at: Option<SystemTime>,
    pub metadata: HashMap<String, String>,
    pub related_events: Vec<String>,
}

/// Alert statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStats {
    pub total_alerts: u64,
    pub alerts_by_severity: HashMap<AlertSeverity, u64>,
    pub alerts_by_rule: HashMap<String, u64>,
    pub active_alerts: u64,
    pub resolved_alerts: u64,
    pub false_positives: u64,
    pub last_alert_time: Option<SystemTime>,
    pub mean_time_to_resolution: Duration,
}

/// Alert configuration for simplified setup
#[derive(Debug, Clone)]
pub struct AlertConfig {
    pub duration_threshold: Duration,
    pub memory_threshold: u64,
    pub throughput_threshold: f64,
    pub enable_anomaly_detection: bool,
    pub sigma_threshold: f64,
    pub notification_channels: Vec<NotificationChannel>,
    pub rate_limit_seconds: u64,
}

/// Alert manager
pub struct AlertManager {
    rules: HashMap<String, AlertRule>,
    active_alerts: HashMap<String, Alert>,
    alert_history: Vec<Alert>,
    stats: AlertStats,
    performance_history: Vec<(SystemTime, f64)>,
    memory_history: Vec<(SystemTime, u64)>,
    last_alert_times: HashMap<String, SystemTime>,
    alert_counts: HashMap<String, u32>,
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            active_alerts: HashMap::new(),
            alert_history: Vec::new(),
            stats: AlertStats {
                total_alerts: 0,
                alerts_by_severity: HashMap::new(),
                alerts_by_rule: HashMap::new(),
                active_alerts: 0,
                resolved_alerts: 0,
                false_positives: 0,
                last_alert_time: None,
                mean_time_to_resolution: Duration::from_secs(0),
            },
            performance_history: Vec::new(),
            memory_history: Vec::new(),
            last_alert_times: HashMap::new(),
            alert_counts: HashMap::new(),
        }
    }

    /// Add an alert rule
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.insert(rule.id.clone(), rule);
    }

    /// Remove an alert rule
    pub fn remove_rule(&mut self, rule_id: &str) -> Option<AlertRule> {
        self.rules.remove(rule_id)
    }

    /// Enable/disable an alert rule
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) {
        if let Some(rule) = self.rules.get_mut(rule_id) {
            rule.enabled = enabled;
        }
    }

    /// Process a profile event and check for alerts
    pub fn process_event(&mut self, event: &ProfileEvent) -> Result<Vec<Alert>> {
        let mut triggered_alerts = Vec::new();

        // Update performance history
        let timestamp = SystemTime::UNIX_EPOCH + Duration::from_micros(event.start_us);
        self.performance_history
            .push((timestamp, event.duration_us as f64 * 1000.0)); // Convert to ns
        if let Some(bytes) = event.bytes_transferred {
            self.memory_history.push((timestamp, bytes));
        }

        // Keep only recent history (last 1000 events)
        if self.performance_history.len() > 1000 {
            self.performance_history.drain(0..100);
        }
        if self.memory_history.len() > 1000 {
            self.memory_history.drain(0..100);
        }

        // Collect all alerts to trigger (to avoid borrow checker issues)
        let mut alerts_to_trigger = Vec::new();

        // Check each rule
        for rule in self.rules.values().filter(|r| r.enabled) {
            if self.should_skip_rule(rule) {
                continue;
            }

            if let Some(alert) = self.evaluate_rule(rule, event)? {
                alerts_to_trigger.push(alert);
            }
        }

        // Now trigger all alerts (after iterator is done)
        for alert in alerts_to_trigger {
            triggered_alerts.push(alert.clone());
            self.trigger_alert(alert)?;
        }

        Ok(triggered_alerts)
    }

    fn should_skip_rule(&self, rule: &AlertRule) -> bool {
        // Check cooldown period
        if let Some(last_time) = self.last_alert_times.get(&rule.id) {
            if last_time.elapsed().unwrap_or(Duration::MAX) < rule.cooldown_duration {
                return true;
            }
        }

        // Check rate limiting
        if let Some(count) = self.alert_counts.get(&rule.id) {
            if *count >= rule.max_alerts_per_hour {
                return true;
            }
        }

        false
    }

    fn evaluate_rule(&self, rule: &AlertRule, event: &ProfileEvent) -> Result<Option<Alert>> {
        let should_trigger = match &rule.condition {
            AlertCondition::DurationThreshold {
                operation,
                max_duration_ns,
            } => event.name == *operation && (event.duration_us * 1000) > *max_duration_ns,
            AlertCondition::MemoryThreshold { max_memory_bytes } => {
                if let Some(bytes) = event.bytes_transferred {
                    bytes > *max_memory_bytes
                } else {
                    false
                }
            }
            AlertCondition::ThroughputThreshold {
                operation,
                min_throughput,
            } => {
                if event.name == *operation {
                    if let Some(ops) = event.operation_count {
                        let throughput = ops as f64 / (event.duration_us as f64 / 1_000_000.0);
                        throughput < *min_throughput
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            AlertCondition::PerformanceDegradation {
                operation,
                degradation_threshold,
                window_size,
            } => {
                self.detect_performance_degradation(operation, *degradation_threshold, *window_size)
            }
            AlertCondition::MemoryLeak {
                growth_threshold,
                window_size,
            } => self.detect_memory_leak(*growth_threshold, *window_size),
            AlertCondition::StatisticalAnomaly {
                operation,
                sigma_threshold,
            } => self.detect_statistical_anomaly(operation, *sigma_threshold, event),
            AlertCondition::Custom {
                name: _,
                expression: _,
            } => {
                // Custom expression evaluation would go here
                false
            }
            _ => false,
        };

        if should_trigger {
            let alert = Alert {
                id: format!(
                    "{}_{}",
                    rule.id,
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ),
                rule_id: rule.id.clone(),
                severity: rule.severity,
                title: format!("Alert: {}", rule.name),
                description: self.generate_alert_description(rule, event),
                timestamp: SystemTime::now(),
                resolved: false,
                resolved_at: None,
                metadata: self.generate_alert_metadata(rule, event),
                related_events: vec![format!("{}_{}", event.name, event.start_us)],
            };
            Ok(Some(alert))
        } else {
            Ok(None)
        }
    }

    fn detect_performance_degradation(
        &self,
        operation: &str,
        threshold: f64,
        window_size: usize,
    ) -> bool {
        let relevant_events: Vec<_> = self
            .performance_history
            .iter()
            .rev()
            .take(window_size)
            .collect();

        if relevant_events.len() < window_size {
            return false;
        }

        let recent_avg = relevant_events
            .iter()
            .take(window_size / 2)
            .map(|(_, v)| v)
            .sum::<f64>()
            / (window_size / 2) as f64;
        let older_avg = relevant_events
            .iter()
            .skip(window_size / 2)
            .map(|(_, v)| v)
            .sum::<f64>()
            / (window_size / 2) as f64;

        if older_avg > 0.0 {
            let degradation = (recent_avg - older_avg) / older_avg;
            degradation > threshold
        } else {
            false
        }
    }

    fn detect_memory_leak(&self, growth_threshold: f64, window_size: usize) -> bool {
        let relevant_events: Vec<_> = self.memory_history.iter().rev().take(window_size).collect();

        if relevant_events.len() < window_size {
            return false;
        }

        let recent_avg = relevant_events
            .iter()
            .take(window_size / 2)
            .map(|(_, v)| *v as f64)
            .sum::<f64>()
            / (window_size / 2) as f64;
        let older_avg = relevant_events
            .iter()
            .skip(window_size / 2)
            .map(|(_, v)| *v as f64)
            .sum::<f64>()
            / (window_size / 2) as f64;

        if older_avg > 0.0 {
            let growth = (recent_avg - older_avg) / older_avg;
            growth > growth_threshold
        } else {
            false
        }
    }

    fn detect_statistical_anomaly(
        &self,
        operation: &str,
        sigma_threshold: f64,
        event: &ProfileEvent,
    ) -> bool {
        if event.name != operation {
            return false;
        }

        let durations: Vec<f64> = self
            .performance_history
            .iter()
            .map(|(_, duration)| *duration)
            .collect();

        if durations.len() < 30 {
            return false; // Need enough data for statistical analysis
        }

        let mean = durations.iter().sum::<f64>() / durations.len() as f64;
        let variance =
            durations.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / durations.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            let z_score = (event.duration_us as f64 * 1000.0 - mean) / std_dev;
            z_score.abs() > sigma_threshold
        } else {
            false
        }
    }

    fn generate_alert_description(&self, rule: &AlertRule, event: &ProfileEvent) -> String {
        match &rule.condition {
            AlertCondition::DurationThreshold {
                operation,
                max_duration_ns,
            } => {
                format!(
                    "Operation '{}' took {}ns, exceeding threshold of {}ns",
                    operation,
                    event.duration_us * 1000,
                    max_duration_ns
                )
            }
            AlertCondition::MemoryThreshold { max_memory_bytes } => {
                format!(
                    "Memory usage of {} bytes exceeds threshold of {} bytes",
                    event.bytes_transferred.unwrap_or(0),
                    max_memory_bytes
                )
            }
            AlertCondition::ThroughputThreshold {
                operation,
                min_throughput,
            } => {
                let actual_throughput = if let Some(ops) = event.operation_count {
                    ops as f64 / (event.duration_us as f64 / 1_000_000.0)
                } else {
                    0.0
                };
                format!("Operation '{operation}' throughput {actual_throughput:.2} ops/sec is below threshold of {min_throughput:.2} ops/sec")
            }
            AlertCondition::PerformanceDegradation {
                operation,
                degradation_threshold,
                ..
            } => {
                format!(
                    "Performance degradation detected for '{}' (>{:.1}% slower)",
                    operation,
                    degradation_threshold * 100.0
                )
            }
            AlertCondition::MemoryLeak {
                growth_threshold, ..
            } => {
                format!(
                    "Memory leak detected (>{:.1}% growth)",
                    growth_threshold * 100.0
                )
            }
            AlertCondition::StatisticalAnomaly {
                operation,
                sigma_threshold,
            } => {
                format!("Statistical anomaly detected for '{operation}' (>{sigma_threshold:.1}σ from mean)")
            }
            AlertCondition::Custom { name, .. } => {
                format!("Custom condition '{name}' triggered")
            }
            _ => "Alert condition triggered".to_string(),
        }
    }

    fn generate_alert_metadata(
        &self,
        rule: &AlertRule,
        event: &ProfileEvent,
    ) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("rule_name".to_string(), rule.name.clone());
        metadata.insert("operation".to_string(), event.name.clone());
        metadata.insert(
            "duration_ns".to_string(),
            (event.duration_us * 1000).to_string(),
        );
        metadata.insert("thread_id".to_string(), event.thread_id.to_string());

        if let Some(ops) = event.operation_count {
            metadata.insert("operation_count".to_string(), ops.to_string());
        }
        if let Some(flops) = event.flops {
            metadata.insert("flops".to_string(), flops.to_string());
        }
        if let Some(bytes) = event.bytes_transferred {
            metadata.insert("bytes_transferred".to_string(), bytes.to_string());
        }

        metadata
    }

    fn trigger_alert(&mut self, alert: Alert) -> Result<()> {
        // Update statistics
        self.stats.total_alerts += 1;
        *self
            .stats
            .alerts_by_severity
            .entry(alert.severity)
            .or_insert(0) += 1;
        *self
            .stats
            .alerts_by_rule
            .entry(alert.rule_id.clone())
            .or_insert(0) += 1;
        self.stats.active_alerts += 1;
        self.stats.last_alert_time = Some(alert.timestamp);

        // Update rate limiting
        self.last_alert_times
            .insert(alert.rule_id.clone(), alert.timestamp);
        *self.alert_counts.entry(alert.rule_id.clone()).or_insert(0) += 1;

        // Store alert
        self.active_alerts.insert(alert.id.clone(), alert.clone());
        self.alert_history.push(alert.clone());

        // Send notifications
        if let Some(rule) = self.rules.get(&alert.rule_id) {
            for channel in &rule.channels {
                self.send_notification(channel, &alert)?;
            }
        }

        Ok(())
    }

    fn send_notification(&self, channel: &NotificationChannel, alert: &Alert) -> Result<()> {
        match channel {
            NotificationChannel::Console => {
                println!(
                    "[{}] {}: {}",
                    alert.severity, alert.title, alert.description
                );
            }
            NotificationChannel::Log { level, format } => {
                let log_message = format
                    .replace("{title}", &alert.title)
                    .replace("{description}", &alert.description)
                    .replace("{severity}", &alert.severity.to_string());
                match level.as_str() {
                    "error" => log::error!("{log_message}"),
                    "warn" => log::warn!("{log_message}"),
                    "info" => log::info!("{log_message}"),
                    _ => log::debug!("{log_message}"),
                }
            }
            NotificationChannel::Slack {
                webhook_url,
                channel,
            } => {
                self.send_slack_notification(webhook_url, channel, alert)?;
            }
            NotificationChannel::Discord { webhook_url } => {
                self.send_discord_notification(webhook_url, alert)?;
            }
            NotificationChannel::Email {
                recipients,
                smtp_config,
            } => {
                self.send_email_notification(recipients, smtp_config, alert)?;
            }
            NotificationChannel::Webhook { url, headers } => {
                self.send_webhook_notification(url, headers, alert)?;
            }
            NotificationChannel::PagerDuty { integration_key } => {
                self.send_pagerduty_notification(integration_key, alert)?;
            }
        }
        Ok(())
    }

    fn send_slack_notification(
        &self,
        webhook_url: &str,
        channel: &str,
        alert: &Alert,
    ) -> Result<()> {
        let payload = serde_json::json!({
            "channel": channel,
            "text": format!("{}: {}", alert.title, alert.description),
            "attachments": [{
                "color": match alert.severity {
                    AlertSeverity::Emergency => "danger",
                    AlertSeverity::Critical => "danger",
                    AlertSeverity::Warning => "warning",
                    AlertSeverity::Info => "good",
                },
                "fields": [
                    {
                        "title": "Severity",
                        "value": alert.severity.to_string(),
                        "short": true
                    },
                    {
                        "title": "Timestamp",
                        "value": format!("{:?}", alert.timestamp),
                        "short": true
                    }
                ]
            }]
        });

        // In a real implementation, we would use reqwest or similar to send the HTTP request
        println!("Would send Slack notification to {webhook_url}: {payload}");
        Ok(())
    }

    fn send_discord_notification(&self, webhook_url: &str, alert: &Alert) -> Result<()> {
        let payload = serde_json::json!({
            "content": format!("**{}**\n{}", alert.title, alert.description),
            "embeds": [{
                "title": "Alert Details",
                "color": match alert.severity {
                    AlertSeverity::Emergency => 0xFF0000,
                    AlertSeverity::Critical => 0xFF0000,
                    AlertSeverity::Warning => 0xFFFF00,
                    AlertSeverity::Info => 0x00FF00,
                },
                "fields": [
                    {
                        "name": "Severity",
                        "value": alert.severity.to_string(),
                        "inline": true
                    },
                    {
                        "name": "Timestamp",
                        "value": format!("{:?}", alert.timestamp),
                        "inline": true
                    }
                ]
            }]
        });

        println!("Would send Discord notification to {webhook_url}: {payload}");
        Ok(())
    }

    fn send_email_notification(
        &self,
        recipients: &[String],
        smtp_config: &SmtpConfig,
        alert: &Alert,
    ) -> Result<()> {
        let subject = format!("[{}] {}", alert.severity, alert.title);
        let body = format!(
            "{}\n\nTimestamp: {:?}\nSeverity: {}\nAlert ID: {}",
            alert.description, alert.timestamp, alert.severity, alert.id
        );

        println!(
            "Would send email to {:?} via {}:{}: {}",
            recipients, smtp_config.server, smtp_config.port, subject
        );
        Ok(())
    }

    fn send_webhook_notification(
        &self,
        url: &str,
        headers: &HashMap<String, String>,
        alert: &Alert,
    ) -> Result<()> {
        let payload = serde_json::to_string(alert)?;
        println!("Would send webhook to {url} with headers {headers:?}: {payload}");
        Ok(())
    }

    fn send_pagerduty_notification(&self, integration_key: &str, alert: &Alert) -> Result<()> {
        let payload = serde_json::json!({
            "routing_key": integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "severity": match alert.severity {
                    AlertSeverity::Emergency => "critical",
                    AlertSeverity::Critical => "critical",
                    AlertSeverity::Warning => "warning",
                    AlertSeverity::Info => "info",
                },
                "source": "torsh-profiler",
                "custom_details": alert.metadata
            }
        });

        println!("Would send PagerDuty notification with key {integration_key}: {payload}");
        Ok(())
    }

    /// Resolve an alert
    pub fn resolve_alert(&mut self, alert_id: &str) -> Result<()> {
        if let Some(alert) = self.active_alerts.remove(alert_id) {
            let mut resolved_alert = alert;
            resolved_alert.resolved = true;
            resolved_alert.resolved_at = Some(SystemTime::now());

            self.stats.active_alerts -= 1;
            self.stats.resolved_alerts += 1;

            // Update mean time to resolution
            if let Ok(resolution_time) = resolved_alert
                .resolved_at
                .unwrap()
                .duration_since(resolved_alert.timestamp)
            {
                let total_resolution_time = self.stats.mean_time_to_resolution.as_secs()
                    * self.stats.resolved_alerts
                    + resolution_time.as_secs();
                self.stats.mean_time_to_resolution =
                    Duration::from_secs(total_resolution_time / self.stats.resolved_alerts);
            }
        }
        Ok(())
    }

    /// Get alert statistics
    pub fn get_statistics(&self) -> AlertStats {
        self.stats.clone()
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.values().cloned().collect()
    }

    /// Get alert history
    pub fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let mut history = self.alert_history.clone();
        history.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        if let Some(limit) = limit {
            history.truncate(limit);
        }

        history
    }

    /// Clear old alerts from history
    pub fn cleanup_old_alerts(&mut self, older_than: Duration) {
        let cutoff_time = SystemTime::now() - older_than;
        self.alert_history
            .retain(|alert| alert.timestamp >= cutoff_time);
    }

    /// Reset hourly alert counts
    pub fn reset_hourly_counts(&mut self) {
        self.alert_counts.clear();
    }
}

/// Public API functions
/// Get the global alert manager
pub fn get_alert_manager() -> Arc<Mutex<AlertManager>> {
    ALERT_MANAGER.clone()
}

/// Add an alert rule
pub fn add_alert_rule(rule: AlertRule) {
    ALERT_MANAGER.lock().add_rule(rule);
}

/// Remove an alert rule
pub fn remove_alert_rule(rule_id: &str) -> Option<AlertRule> {
    ALERT_MANAGER.lock().remove_rule(rule_id)
}

/// Process a profile event for alerts
pub fn process_alert_event(event: &ProfileEvent) -> Result<Vec<Alert>> {
    ALERT_MANAGER.lock().process_event(event)
}

/// Resolve an alert
pub fn resolve_alert(alert_id: &str) -> Result<()> {
    ALERT_MANAGER.lock().resolve_alert(alert_id)
}

/// Get alert statistics
pub fn get_alert_statistics() -> AlertStats {
    ALERT_MANAGER.lock().get_statistics()
}

/// Get active alerts
pub fn get_active_alerts() -> Vec<Alert> {
    ALERT_MANAGER.lock().get_active_alerts()
}

/// Get alert history
pub fn get_alert_history(limit: Option<usize>) -> Vec<Alert> {
    ALERT_MANAGER.lock().get_alert_history(limit)
}

/// Create a simple duration threshold alert rule
pub fn create_duration_alert_rule(
    id: String,
    operation: String,
    max_duration_ns: u64,
    severity: AlertSeverity,
    channels: Vec<NotificationChannel>,
) -> AlertRule {
    AlertRule {
        id,
        name: format!("Duration threshold for {operation}"),
        description: format!("Alert when {operation} takes longer than {max_duration_ns}ns"),
        condition: AlertCondition::DurationThreshold {
            operation,
            max_duration_ns,
        },
        severity,
        channels,
        enabled: true,
        cooldown_duration: Duration::from_secs(300), // 5 minutes
        max_alerts_per_hour: 10,
        tags: HashMap::new(),
    }
}

/// Create a memory threshold alert rule
pub fn create_memory_alert_rule(
    id: String,
    max_memory_bytes: u64,
    severity: AlertSeverity,
    channels: Vec<NotificationChannel>,
) -> AlertRule {
    AlertRule {
        id,
        name: "Memory threshold".to_string(),
        description: format!("Alert when memory usage exceeds {max_memory_bytes} bytes"),
        condition: AlertCondition::MemoryThreshold { max_memory_bytes },
        severity,
        channels,
        enabled: true,
        cooldown_duration: Duration::from_secs(300),
        max_alerts_per_hour: 10,
        tags: HashMap::new(),
    }
}

/// Create an alert manager with simplified configuration
pub fn create_alert_manager_with_config(config: AlertConfig) -> AlertManager {
    let mut manager = AlertManager::new();

    // Add duration threshold rule
    let duration_rule = AlertRule {
        id: "duration_threshold".to_string(),
        name: "Duration Threshold".to_string(),
        description: format!(
            "Alert when operation duration exceeds {:?}",
            config.duration_threshold
        ),
        condition: AlertCondition::DurationThreshold {
            operation: "*".to_string(),
            max_duration_ns: config.duration_threshold.as_nanos() as u64,
        },
        severity: AlertSeverity::Warning,
        channels: config.notification_channels.clone(),
        enabled: true,
        cooldown_duration: Duration::from_secs(config.rate_limit_seconds),
        max_alerts_per_hour: 10,
        tags: HashMap::new(),
    };
    manager.add_rule(duration_rule);

    // Add memory threshold rule
    let memory_rule = AlertRule {
        id: "memory_threshold".to_string(),
        name: "Memory Threshold".to_string(),
        description: format!(
            "Alert when memory usage exceeds {} bytes",
            config.memory_threshold
        ),
        condition: AlertCondition::MemoryThreshold {
            max_memory_bytes: config.memory_threshold,
        },
        severity: AlertSeverity::Critical,
        channels: config.notification_channels.clone(),
        enabled: true,
        cooldown_duration: Duration::from_secs(config.rate_limit_seconds),
        max_alerts_per_hour: 5,
        tags: HashMap::new(),
    };
    manager.add_rule(memory_rule);

    if config.enable_anomaly_detection {
        // Add statistical anomaly detection rule
        let anomaly_rule = AlertRule {
            id: "anomaly_detection".to_string(),
            name: "Statistical Anomaly".to_string(),
            description: format!(
                "Alert when performance deviates more than {} sigma",
                config.sigma_threshold
            ),
            condition: AlertCondition::StatisticalAnomaly {
                operation: "*".to_string(),
                sigma_threshold: config.sigma_threshold,
            },
            severity: AlertSeverity::Warning,
            channels: config.notification_channels,
            enabled: true,
            cooldown_duration: Duration::from_secs(config.rate_limit_seconds),
            max_alerts_per_hour: 20,
            tags: HashMap::new(),
        };
        manager.add_rule(anomaly_rule);
    }

    manager
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_alert_manager_creation() {
        let manager = AlertManager::new();
        assert_eq!(manager.rules.len(), 0);
        assert_eq!(manager.active_alerts.len(), 0);
        assert_eq!(manager.stats.total_alerts, 0);
    }

    #[test]
    fn test_add_remove_rule() {
        let mut manager = AlertManager::new();
        let rule = create_duration_alert_rule(
            "test_rule".to_string(),
            "test_op".to_string(),
            1000000,
            AlertSeverity::Warning,
            vec![NotificationChannel::Console],
        );

        manager.add_rule(rule);
        assert_eq!(manager.rules.len(), 1);

        let removed = manager.remove_rule("test_rule");
        assert!(removed.is_some());
        assert_eq!(manager.rules.len(), 0);
    }

    #[test]
    fn test_duration_threshold_alert() {
        let mut manager = AlertManager::new();
        let rule = create_duration_alert_rule(
            "duration_test".to_string(),
            "slow_operation".to_string(),
            1000000, // 1ms
            AlertSeverity::Warning,
            vec![NotificationChannel::Console],
        );

        manager.add_rule(rule);

        let event = ProfileEvent {
            name: "slow_operation".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 2000, // 2ms - should trigger alert
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(10),
            bytes_transferred: Some(100),
            stack_trace: Some("test trace".to_string()),
        };

        let alerts = manager.process_event(&event).unwrap();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);
        assert_eq!(manager.stats.total_alerts, 1);
    }

    #[test]
    fn test_memory_threshold_alert() {
        let mut manager = AlertManager::new();
        let rule = create_memory_alert_rule(
            "memory_test".to_string(),
            1024, // 1KB
            AlertSeverity::Critical,
            vec![NotificationChannel::Console],
        );

        manager.add_rule(rule);

        let event = ProfileEvent {
            name: "memory_operation".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1000,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(10),
            bytes_transferred: Some(2048), // 2KB - should trigger alert
            stack_trace: Some("test trace".to_string()),
        };

        let alerts = manager.process_event(&event).unwrap();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_resolution() {
        let mut manager = AlertManager::new();
        let rule = create_duration_alert_rule(
            "resolve_test".to_string(),
            "test_op".to_string(),
            1000000,
            AlertSeverity::Warning,
            vec![NotificationChannel::Console],
        );

        manager.add_rule(rule);

        let event = ProfileEvent {
            name: "test_op".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 2000,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(10),
            bytes_transferred: Some(100),
            stack_trace: Some("test trace".to_string()),
        };

        let alerts = manager.process_event(&event).unwrap();
        assert_eq!(alerts.len(), 1);
        assert_eq!(manager.stats.active_alerts, 1);

        manager.resolve_alert(&alerts[0].id).unwrap();
        assert_eq!(manager.stats.active_alerts, 0);
        assert_eq!(manager.stats.resolved_alerts, 1);
    }

    #[test]
    fn test_alert_cooldown() {
        let mut manager = AlertManager::new();
        let mut rule = create_duration_alert_rule(
            "cooldown_test".to_string(),
            "test_op".to_string(),
            1000000,
            AlertSeverity::Warning,
            vec![NotificationChannel::Console],
        );
        rule.cooldown_duration = Duration::from_secs(1);

        manager.add_rule(rule);

        let event = ProfileEvent {
            name: "test_op".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 2000,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(10),
            bytes_transferred: Some(100),
            stack_trace: Some("test trace".to_string()),
        };

        // First alert should trigger
        let alerts1 = manager.process_event(&event).unwrap();
        assert_eq!(alerts1.len(), 1);

        // Second alert should be blocked by cooldown
        let alerts2 = manager.process_event(&event).unwrap();
        assert_eq!(alerts2.len(), 0);
    }

    #[test]
    fn test_alert_serialization() {
        let alert = Alert {
            id: "test_alert".to_string(),
            rule_id: "test_rule".to_string(),
            severity: AlertSeverity::Warning,
            title: "Test Alert".to_string(),
            description: "This is a test alert".to_string(),
            timestamp: SystemTime::now(),
            resolved: false,
            resolved_at: None,
            metadata: HashMap::new(),
            related_events: vec!["event1".to_string()],
        };

        let json = serde_json::to_string(&alert).unwrap();
        let deserialized: Alert = serde_json::from_str(&json).unwrap();

        assert_eq!(alert.id, deserialized.id);
        assert_eq!(alert.rule_id, deserialized.rule_id);
        assert_eq!(alert.severity, deserialized.severity);
        assert_eq!(alert.title, deserialized.title);
        assert_eq!(alert.description, deserialized.description);
        assert_eq!(alert.resolved, deserialized.resolved);
    }

    #[test]
    fn test_global_api_functions() {
        let rule = create_duration_alert_rule(
            "global_test".to_string(),
            "global_op".to_string(),
            1000000,
            AlertSeverity::Info,
            vec![NotificationChannel::Console],
        );

        add_alert_rule(rule);

        let event = ProfileEvent {
            name: "global_op".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 2000,
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(10),
            bytes_transferred: Some(100),
            stack_trace: Some("test trace".to_string()),
        };

        let alerts = process_alert_event(&event).unwrap();
        assert_eq!(alerts.len(), 1);

        let stats = get_alert_statistics();
        assert_eq!(stats.total_alerts, 1);

        let active_alerts = get_active_alerts();
        assert_eq!(active_alerts.len(), 1);

        resolve_alert(&alerts[0].id).unwrap();

        let resolved_stats = get_alert_statistics();
        assert_eq!(resolved_stats.active_alerts, 0);
        assert_eq!(resolved_stats.resolved_alerts, 1);
    }
}
