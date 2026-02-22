//! # MappingStatistics - Trait Implementations
//!
//! This module contains trait implementations for `MappingStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MappingEffectiveness, MappingOverheadAnalysis, MappingStatistics};

impl Default for MappingStatistics {
    fn default() -> Self {
        Self {
            total_mappings: 0,
            active_mappings: 0,
            success_rate: 1.0,
            overhead_analysis: MappingOverheadAnalysis::default(),
            effectiveness: MappingEffectiveness::default(),
        }
    }
}
