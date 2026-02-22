//! # SystemLatencyCharacteristics - Trait Implementations
//!
//! This module contains trait implementations for `SystemLatencyCharacteristics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{LoadLatencyCharacteristics, PercentileStats, SystemLatencyCharacteristics};

impl Default for SystemLatencyCharacteristics {
    fn default() -> Self {
        Self {
            average_latency: Duration::from_secs(0),
            percentiles: PercentileStats::default(),
            consistency: 1.0,
            load_latency: LoadLatencyCharacteristics::default(),
        }
    }
}
