//! Core alert types and enums
//!
//! This module contains the fundamental data structures used throughout
//! the alert system, including severity levels, conditions, and alert instances.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

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
    pub notification_channels: Vec<super::notifications::NotificationChannel>,
    pub rate_limit_seconds: u64,
}
