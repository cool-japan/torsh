//! # MappingEffectiveness - Trait Implementations
//!
//! This module contains trait implementations for `MappingEffectiveness`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MappingEffectiveness;

impl Default for MappingEffectiveness {
    fn default() -> Self {
        Self {
            performance_benefit: 0.0,
            efficiency_improvement: 0.0,
            cost_benefit_ratio: 0.0,
            utilization_rate: 0.0,
        }
    }
}
