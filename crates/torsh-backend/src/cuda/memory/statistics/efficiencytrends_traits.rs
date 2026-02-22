//! # EfficiencyTrends - Trait Implementations
//!
//! This module contains trait implementations for `EfficiencyTrends`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EfficiencyTrend, EfficiencyTrends};

impl Default for EfficiencyTrends {
    fn default() -> Self {
        Self {
            memory_trend: EfficiencyTrend::Stable,
            bandwidth_trend: EfficiencyTrend::Stable,
            compute_trend: EfficiencyTrend::Stable,
            overall_trend: EfficiencyTrend::Stable,
        }
    }
}
