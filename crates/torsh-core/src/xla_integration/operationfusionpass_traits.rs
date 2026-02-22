//! # OperationFusionPass - Trait Implementations
//!
//! This module contains trait implementations for `OperationFusionPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{OperationFusionPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for OperationFusionPass {
    fn name(&self) -> &str {
        "operation-fusion"
    }
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::new();
        if !computation.config.enable_fusion {
            return Ok(stats);
        }
        let fusion_candidates = Self::find_fusion_candidates(computation);
        if !fusion_candidates.is_empty() {
            stats.nodes_removed = fusion_candidates.len();
            stats.nodes_added = fusion_candidates.len();
        }
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.enable_fusion && config.optimization_level >= 2
    }
}
