//! # BottleneckImpactAssessment - Trait Implementations
//!
//! This module contains trait implementations for `BottleneckImpactAssessment`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{BottleneckImpactAssessment, BusinessImpactLevel, UserImpactLevel};

impl Default for BottleneckImpactAssessment {
    fn default() -> Self {
        Self {
            performance_degradation: 0.0,
            resource_waste: 0.0,
            user_impact: UserImpactLevel::None,
            business_impact: BusinessImpactLevel::None,
        }
    }
}
