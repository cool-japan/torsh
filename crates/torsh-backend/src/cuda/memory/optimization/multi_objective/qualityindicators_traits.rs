//! # QualityIndicators - Trait Implementations
//!
//! This module contains trait implementations for `QualityIndicators`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::QualityIndicators;

impl Default for QualityIndicators {
    fn default() -> Self {
        Self {
            additive_epsilon: 0.0,
            multiplicative_epsilon: 0.0,
            r2_indicator: 0.0,
            igd_plus: 0.0,
            modified_igd: 0.0,
        }
    }
}

