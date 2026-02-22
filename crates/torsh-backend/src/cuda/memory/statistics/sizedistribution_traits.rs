//! # SizeDistribution - Trait Implementations
//!
//! This module contains trait implementations for `SizeDistribution`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::{
    DistributionCharacteristics, OutlierAnalysis, PercentileStats, SizeDistribution,
};

impl Default for SizeDistribution {
    fn default() -> Self {
        Self {
            buckets: BTreeMap::new(),
            percentiles: PercentileStats::default(),
            characteristics: DistributionCharacteristics::default(),
            outliers: OutlierAnalysis::default(),
        }
    }
}
