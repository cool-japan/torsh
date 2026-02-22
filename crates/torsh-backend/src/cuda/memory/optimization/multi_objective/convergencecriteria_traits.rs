//! # ConvergenceCriteria - Trait Implementations
//!
//! This module contains trait implementations for `ConvergenceCriteria`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ConvergenceCriteria, ConvergenceMetric};

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_generations: 500,
            target_hypervolume: None,
            tolerance: 1e-6,
            stagnation_limit: 50,
            min_improvement: 1e-4,
            convergence_metrics: vec![ConvergenceMetric::Hypervolume],
            early_stopping: None,
        }
    }
}

