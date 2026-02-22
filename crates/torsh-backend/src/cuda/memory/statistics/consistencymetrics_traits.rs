//! # ConsistencyMetrics - Trait Implementations
//!
//! This module contains trait implementations for `ConsistencyMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ConsistencyMetrics;

impl Default for ConsistencyMetrics {
    fn default() -> Self {
        Self {
            latency_coefficient_variation: 0.0,
            throughput_stability: 1.0,
            predictability_score: 1.0,
            anomaly_score: 0.0,
        }
    }
}
