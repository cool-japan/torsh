//! # DeadCodeEliminationPass - Trait Implementations
//!
//! This module contains trait implementations for `DeadCodeEliminationPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{DeadCodeEliminationPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dead-code-elimination"
    }
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::new();
        if computation.nodes.is_empty() {
            return Ok(stats);
        }
        let mut reachable = vec![false; computation.nodes.len()];
        Self::mark_reachable(computation, computation.root, &mut reachable);
        let dead_count = reachable.iter().filter(|&&r| !r).count();
        if dead_count > 0 {
            stats.nodes_removed = dead_count;
        }
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 1
    }
}
