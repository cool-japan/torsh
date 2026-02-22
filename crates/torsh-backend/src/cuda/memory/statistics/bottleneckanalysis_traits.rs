//! # BottleneckAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `BottleneckAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BottleneckAnalysis, BottleneckImpactAssessment};

impl Default for BottleneckAnalysis {
    fn default() -> Self {
        Self {
            bottlenecks: Vec::new(),
            primary_bottleneck: None,
            impact_assessment: BottleneckImpactAssessment::default(),
            recommendations: Vec::new(),
        }
    }
}
