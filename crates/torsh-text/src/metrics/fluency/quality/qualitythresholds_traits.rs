//! # QualityThresholds - Trait Implementations
//!
//! This module contains trait implementations for `QualityThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QualityThresholds;

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            excellent_threshold: 0.90,
            good_threshold: 0.80,
            fair_threshold: 0.70,
            poor_threshold: 0.60,
            critical_threshold: 0.50,
        }
    }
}

