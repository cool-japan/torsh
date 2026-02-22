//! # QualityAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `QualityAnalyzer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{QualityAnalyzer, QualityMonitoringConfig, QualityStandards, QualityThresholds, QualityWeights};

impl Default for QualityAnalyzer {
    fn default() -> Self {
        Self {
            weights: QualityWeights::default(),
            thresholds: QualityThresholds::default(),
            standards: QualityStandards::default(),
            monitoring_config: QualityMonitoringConfig::default(),
        }
    }
}

