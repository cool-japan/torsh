//! # PerformanceSnapshot - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceSnapshot`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::PerformanceSnapshot;

impl Default for PerformanceSnapshot {
    fn default() -> Self {
        Self {
            average_latency: Duration::from_secs(0),
            throughput: 0.0,
            efficiency: 1.0,
            error_rate: 0.0,
        }
    }
}
