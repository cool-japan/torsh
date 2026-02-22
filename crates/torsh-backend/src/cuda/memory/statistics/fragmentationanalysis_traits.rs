//! # FragmentationAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `FragmentationAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{FragmentationAnalysis, FragmentationImpact};

impl Default for FragmentationAnalysis {
    fn default() -> Self {
        Self {
            internal_fragmentation: 0.0,
            external_fragmentation: 0.0,
            overall_fragmentation: 0.0,
            fragmentation_trend: Vec::new(),
            performance_impact: FragmentationImpact::Negligible,
            defragmentation_recommendations: Vec::new(),
        }
    }
}
