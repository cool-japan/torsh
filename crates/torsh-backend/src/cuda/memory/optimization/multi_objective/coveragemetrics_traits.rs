//! # CoverageMetrics - Trait Implementations
//!
//! This module contains trait implementations for `CoverageMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CoverageMetrics;

impl Default for CoverageMetrics {
    fn default() -> Self {
        Self {
            c_metric: 0.0,
            set_coverage: 0.0,
            epsilon_indicator: 0.0,
            binary_epsilon: 0.0,
        }
    }
}

