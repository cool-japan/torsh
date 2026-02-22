//! # DiversityStrategy - Trait Implementations
//!
//! This module contains trait implementations for `DiversityStrategy`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DiversityStrategy, DiversityStrategyType};

impl Default for DiversityStrategy {
    fn default() -> Self {
        Self {
            strategy_type: DiversityStrategyType::CrowdingDistance,
            diversity_threshold: 0.1,
            niche_radius: 0.1,
            clustering_config: None,
        }
    }
}

