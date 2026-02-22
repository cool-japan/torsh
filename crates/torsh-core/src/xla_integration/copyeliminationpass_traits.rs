//! # CopyEliminationPass - Trait Implementations
//!
//! This module contains trait implementations for `CopyEliminationPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{CopyEliminationPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for CopyEliminationPass {
    fn name(&self) -> &str {
        "copy-elimination"
    }
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::default();
        if !self.should_run(&computation.config) {
            return Ok(stats);
        }
        let eliminable_copies = Self::count_eliminable_copies(computation);
        if eliminable_copies > 0 {
            stats.nodes_removed = eliminable_copies;
        }
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 1
    }
}
