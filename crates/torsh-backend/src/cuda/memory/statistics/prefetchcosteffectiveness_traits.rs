//! # PrefetchCostEffectiveness - Trait Implementations
//!
//! This module contains trait implementations for `PrefetchCostEffectiveness`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PrefetchCostEffectiveness;

impl Default for PrefetchCostEffectiveness {
    fn default() -> Self {
        Self {
            prefetch_cost: 0.0,
            benefits: 0.0,
            net_benefit: 0.0,
            cost_per_success: 0.0,
        }
    }
}
