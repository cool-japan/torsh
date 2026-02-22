//! # LayoutOptimizationPass - Trait Implementations
//!
//! This module contains trait implementations for `LayoutOptimizationPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{LayoutOptimizationPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for LayoutOptimizationPass {
    fn name(&self) -> &str {
        "layout-optimization"
    }
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::default();
        if !self.should_run(&computation.config) {
            return Ok(stats);
        }
        let layout_optimization_opportunities = Self::count_layout_opportunities(computation);
        if layout_optimization_opportunities > 0 {
            stats.nodes_modified = layout_optimization_opportunities;
        }
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 2
    }
}
