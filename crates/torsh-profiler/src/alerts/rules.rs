//! Alert rules and rule management
//!
//! This module contains the alert rule definitions and helper functions
//! for creating common alert rules.

use super::core::{AlertCondition, AlertSeverity};
use super::notifications::NotificationChannel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

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

/// Create a duration-based alert rule
pub fn create_duration_alert_rule(
    operation: String,
    max_duration_ms: f64,
    severity: AlertSeverity,
    channels: Vec<NotificationChannel>,
) -> AlertRule {
    AlertRule {
        id: format!("duration_{}_{}", operation, max_duration_ms),
        name: format!("Duration threshold for {}", operation),
        description: format!(
            "Alert when operation '{}' exceeds {}ms duration",
            operation, max_duration_ms
        ),
        condition: AlertCondition::DurationThreshold {
            operation,
            max_duration_ns: (max_duration_ms * 1_000_000.0) as u64,
        },
        severity,
        channels,
        enabled: true,
        cooldown_duration: Duration::from_secs(300),
        max_alerts_per_hour: 10,
        tags: HashMap::new(),
    }
}

/// Create a memory-based alert rule
pub fn create_memory_alert_rule(
    max_memory_mb: u64,
    severity: AlertSeverity,
    channels: Vec<NotificationChannel>,
) -> AlertRule {
    AlertRule {
        id: format!("memory_{}", max_memory_mb),
        name: format!("Memory threshold {}MB", max_memory_mb),
        description: format!("Alert when memory usage exceeds {}MB", max_memory_mb),
        condition: AlertCondition::MemoryThreshold {
            max_memory_bytes: max_memory_mb * 1024 * 1024,
        },
        severity,
        channels,
        enabled: true,
        cooldown_duration: Duration::from_secs(300),
        max_alerts_per_hour: 10,
        tags: HashMap::new(),
    }
}
