//! # TransferPerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `TransferPerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LatencyStatistics, TransferPerformanceMetrics};

impl Default for TransferPerformanceMetrics {
    fn default() -> Self {
        Self {
            average_bandwidth: 0.0,
            peak_bandwidth: 0.0,
            bandwidth_consistency: 1.0,
            latency_stats: LatencyStatistics::default(),
            efficiency_vs_max: 0.0,
        }
    }
}
