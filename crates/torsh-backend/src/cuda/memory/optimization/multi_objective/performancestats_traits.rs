//! # PerformanceStats - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant};

use super::types::PerformanceStats;

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            avg_generation_time: Duration::from_secs(0),
            total_time: Duration::from_secs(0),
            evaluations_per_second: 0.0,
            peak_memory_usage: 0,
            cpu_utilization: 0.0,
            convergence_rate: 0.0,
        }
    }
}

