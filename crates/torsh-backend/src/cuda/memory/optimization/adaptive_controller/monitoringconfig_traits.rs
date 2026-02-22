//! # MonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `MonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::types::MonitoringConfig;

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            frequency: Duration::from_secs(10),
            collection_interval: Duration::from_secs(5),
            retention_period: Duration::from_secs(24 * 3600),
            detailed_logging: true,
            alert_thresholds: HashMap::new(),
        }
    }
}

