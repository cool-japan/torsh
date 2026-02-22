//! # OptimizationMetrics - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::OptimizationMetrics;

impl Default for OptimizationMetrics {
    fn default() -> Self {
        Self {
            total_optimizations: 0,
            successful_optimizations: 0,
            average_improvement: 0.0,
            effectiveness_by_type: HashMap::new(),
            cumulative_improvement: 0.0,
        }
    }
}
