//! # DistributionCharacteristics - Trait Implementations
//!
//! This module contains trait implementations for `DistributionCharacteristics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DistributionCharacteristics, DistributionType};

impl Default for DistributionCharacteristics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            standard_deviation: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            distribution_type: DistributionType::Unknown,
            goodness_of_fit: 0.0,
        }
    }
}
