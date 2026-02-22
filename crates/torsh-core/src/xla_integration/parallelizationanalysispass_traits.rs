//! # ParallelizationAnalysisPass - Trait Implementations
//!
//! This module contains trait implementations for `ParallelizationAnalysisPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{ParallelizationAnalysisPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for ParallelizationAnalysisPass {
    fn name(&self) -> &str {
        "parallelization-analysis"
    }
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::default();
        if !self.should_run(&computation.config) {
            return Ok(stats);
        }
        let independent_groups = Self::count_independent_operation_groups(computation);
        let batch_parallel_ops = Self::count_batch_parallelizable_ops(computation);
        if independent_groups > 0 || batch_parallel_ops > 0 {
            stats.nodes_modified = independent_groups + batch_parallel_ops;
        }
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.optimization_level >= 2
    }
}
