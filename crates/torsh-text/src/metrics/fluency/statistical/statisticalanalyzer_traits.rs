//! # StatisticalAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalAnalyzer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StatisticalAnalyzer;

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            bootstrap_samples: 1000,
            outlier_threshold: 2.0,
        }
    }
}

