//! # ResourceUtilizationEfficiency - Trait Implementations
//!
//! This module contains trait implementations for `ResourceUtilizationEfficiency`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EfficiencyTrends, ResourceUtilizationEfficiency};

impl Default for ResourceUtilizationEfficiency {
    fn default() -> Self {
        Self {
            overall_efficiency: 1.0,
            memory_efficiency: 1.0,
            bandwidth_efficiency: 1.0,
            compute_efficiency: 1.0,
            trends: EfficiencyTrends::default(),
            improvement_opportunities: Vec::new(),
        }
    }
}
