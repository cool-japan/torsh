//! # AnomalyDetectionResult - Trait Implementations
//!
//! This module contains trait implementations for `AnomalyDetectionResult`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AnomalyDetectionResult;

impl Default for AnomalyDetectionResult {
    fn default() -> Self {
        Self {
            anomaly_detected: false,
            confidence: 0.0,
            anomaly_type: None,
            suggested_action: None,
        }
    }
}
