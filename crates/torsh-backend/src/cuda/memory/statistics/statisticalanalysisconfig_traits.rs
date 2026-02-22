//! # StatisticalAnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalAnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StatisticalAnalysisConfig;

impl Default for StatisticalAnalysisConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 100,
            confidence_interval: 0.95,
            enable_outlier_detection: true,
            significance_threshold: 0.05,
        }
    }
}
