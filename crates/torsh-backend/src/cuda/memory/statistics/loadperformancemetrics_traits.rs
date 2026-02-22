//! # LoadPerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `LoadPerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LoadPerformanceMetrics, LoadScalingCharacteristics, PerformanceSnapshot};

impl Default for LoadPerformanceMetrics {
    fn default() -> Self {
        Self {
            low_load_performance: PerformanceSnapshot::default(),
            medium_load_performance: PerformanceSnapshot::default(),
            high_load_performance: PerformanceSnapshot::default(),
            scaling_characteristics: LoadScalingCharacteristics::default(),
        }
    }
}
