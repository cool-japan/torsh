//! # OutlierAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `OutlierAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{OutlierAnalysis, OutlierImpact};

impl Default for OutlierAnalysis {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            outlier_count: 0,
            outlier_percentage: 0.0,
            impact_assessment: OutlierImpact::Minimal,
        }
    }
}
