//! # ThroughputMetrics - Trait Implementations
//!
//! This module contains trait implementations for `ThroughputMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ThroughputMetrics;

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            operations_per_second: 0.0,
            bytes_per_second: 0.0,
            peak_throughput: 0.0,
            consistency_score: 1.0,
            efficiency_vs_theoretical: 0.0,
        }
    }
}
