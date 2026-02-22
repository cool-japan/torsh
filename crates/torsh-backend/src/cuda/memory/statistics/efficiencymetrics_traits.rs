//! # EfficiencyMetrics - Trait Implementations
//!
//! This module contains trait implementations for `EfficiencyMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EfficiencyMetrics, EfficiencyTrend};

impl Default for EfficiencyMetrics {
    fn default() -> Self {
        Self {
            utilization_efficiency: 1.0,
            allocation_efficiency: 1.0,
            waste_percentage: 0.0,
            pool_efficiency: 1.0,
            overall_efficiency: 1.0,
            efficiency_trend: EfficiencyTrend::Stable,
        }
    }
}
