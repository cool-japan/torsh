//! # AllocationPerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `AllocationPerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::AllocationPerformanceMetrics;

impl Default for AllocationPerformanceMetrics {
    fn default() -> Self {
        Self {
            average_allocation_time: Duration::from_secs(0),
            allocation_throughput: 0.0,
            success_rate: 1.0,
            memory_efficiency: 1.0,
        }
    }
}
