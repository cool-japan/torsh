//! # VolumeStatistics - Trait Implementations
//!
//! This module contains trait implementations for `VolumeStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{SizeDistribution, VolumeStatistics};

impl Default for VolumeStatistics {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            average_size: 0.0,
            size_distribution: SizeDistribution::default(),
            peak_rate: 0.0,
        }
    }
}
