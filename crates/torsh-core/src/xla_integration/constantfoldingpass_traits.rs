//! # ConstantFoldingPass - Trait Implementations
//!
//! This module contains trait implementations for `ConstantFoldingPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{ConstantFoldingPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "constant-folding"
    }
    fn run(&self, _computation: &mut XlaComputation) -> Result<PassStatistics> {
        let stats = PassStatistics::new();
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.enable_algebraic_simplification && config.optimization_level >= 1
    }
}
