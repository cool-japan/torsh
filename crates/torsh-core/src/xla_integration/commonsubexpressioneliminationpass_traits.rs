//! # CommonSubexpressionEliminationPass - Trait Implementations
//!
//! This module contains trait implementations for `CommonSubexpressionEliminationPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{CommonSubexpressionEliminationPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for CommonSubexpressionEliminationPass {
    fn name(&self) -> &str {
        "common-subexpression-elimination"
    }
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::new();
        let mut duplicates = 0;
        for i in 0..computation.nodes.len() {
            for j in (i + 1)..computation.nodes.len() {
                if Self::nodes_equivalent(&computation.nodes[i], &computation.nodes[j]) {
                    duplicates += 1;
                }
            }
        }
        if duplicates > 0 {
            stats.nodes_removed = duplicates;
        }
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 2
    }
}
