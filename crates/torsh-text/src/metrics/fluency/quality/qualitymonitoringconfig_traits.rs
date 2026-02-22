//! # QualityMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `QualityMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap};

use super::types::QualityMonitoringConfig;

impl Default for QualityMonitoringConfig {
    fn default() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("critical_drop".to_string(), 0.10);
        alert_thresholds.insert("significant_decline".to_string(), 0.05);
        alert_thresholds.insert("consistency_violation".to_string(), 0.15);
        Self {
            enable_real_time_monitoring: true,
            alert_thresholds,
            monitoring_frequency: 100,
            quality_degradation_sensitivity: 0.05,
        }
    }
}

