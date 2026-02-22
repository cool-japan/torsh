//! # MemoryAllocationOptimizationPass - Trait Implementations
//!
//! This module contains trait implementations for `MemoryAllocationOptimizationPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{MemoryAllocationOptimizationPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for MemoryAllocationOptimizationPass {
    fn name(&self) -> &str {
        "memory-allocation-optimization"
    }
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::default();
        if !self.should_run(&computation.config) {
            return Ok(stats);
        }
        let reuse_opportunities = Self::count_buffer_reuse_opportunities(computation);
        let inplace_opportunities = Self::count_inplace_opportunities(computation);
        if reuse_opportunities > 0 || inplace_opportunities > 0 {
            stats.nodes_modified = reuse_opportunities + inplace_opportunities;
        }
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 1
    }
}
