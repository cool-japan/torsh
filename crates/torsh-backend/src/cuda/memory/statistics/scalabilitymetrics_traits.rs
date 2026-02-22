//! # ScalabilityMetrics - Trait Implementations
//!
//! This module contains trait implementations for `ScalabilityMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ScalabilityMetrics, ScalabilityPattern};

impl Default for ScalabilityMetrics {
    fn default() -> Self {
        Self {
            pattern: ScalabilityPattern::Linear,
            max_scalable_load: 1.0,
            efficiency: 1.0,
            bottlenecks: Vec::new(),
        }
    }
}
