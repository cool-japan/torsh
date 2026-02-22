//! # LifetimeStatistics - Trait Implementations
//!
//! This module contains trait implementations for `LifetimeStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{LifetimeStatistics, PercentileStats};

impl Default for LifetimeStatistics {
    fn default() -> Self {
        Self {
            average_lifetime: Duration::from_secs(0),
            lifetime_distribution: PercentileStats::default(),
            short_lived_percentage: 0.0,
            long_lived_percentage: 0.0,
            size_lifetime_correlation: 0.0,
            prediction_accuracy: 0.0,
        }
    }
}
