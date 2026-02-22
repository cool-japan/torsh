//! # MigrationCostBenefit - Trait Implementations
//!
//! This module contains trait implementations for `MigrationCostBenefit`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MigrationCostBenefit;

impl Default for MigrationCostBenefit {
    fn default() -> Self {
        Self {
            total_cost: 0.0,
            total_benefit: 0.0,
            net_benefit: 0.0,
            return_on_investment: 0.0,
        }
    }
}
