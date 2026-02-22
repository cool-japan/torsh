//! # LatencyStatistics - Trait Implementations
//!
//! This module contains trait implementations for `LatencyStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{LatencyStatistics, PercentileStats, PerformanceTrend};

impl Default for LatencyStatistics {
    fn default() -> Self {
        Self {
            mean_latency: Duration::from_secs(0),
            percentiles: PercentileStats::default(),
            variance: 0.0,
            max_spikes: Vec::new(),
            trend: PerformanceTrend::Stable,
        }
    }
}
