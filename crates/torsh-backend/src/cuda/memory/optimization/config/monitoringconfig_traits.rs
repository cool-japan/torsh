//! # MonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `MonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;
use super::types::{AlertingConfig, AnomalyConfig, CorrelationConfig, DashboardConfig, HealthConfig, LogConfig, MetricsConfig, MonitoringConfig, ResourceConfig, StateMonitorConfig, TracingConfig, TrendConfig};

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(10),
            system_monitor_config: StateMonitorConfig::default(),
            metrics_config: MetricsConfig::default(),
            alerting_config: AlertingConfig::default(),
            dashboard_config: DashboardConfig::default(),
            log_config: LogConfig::default(),
            anomaly_config: AnomalyConfig::default(),
            trend_config: TrendConfig::default(),
            health_config: HealthConfig::default(),
            resource_config: ResourceConfig::default(),
            correlation_config: CorrelationConfig::default(),
            tracing_config: TracingConfig::default(),
        }
    }
}

