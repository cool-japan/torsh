//! # MigrationStatistics - Trait Implementations
//!
//! This module contains trait implementations for `MigrationStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MigrationCostBenefit, MigrationStatistics};

impl Default for MigrationStatistics {
    fn default() -> Self {
        Self {
            total_migrations: 0,
            migration_frequency: 0.0,
            migration_overhead: 0.0,
            effectiveness_score: 1.0,
            patterns: Vec::new(),
            cost_benefit: MigrationCostBenefit::default(),
        }
    }
}
