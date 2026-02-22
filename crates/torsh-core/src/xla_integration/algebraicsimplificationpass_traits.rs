//! # AlgebraicSimplificationPass - Trait Implementations
//!
//! This module contains trait implementations for `AlgebraicSimplificationPass`.
//!
//! ## Implemented Traits
//!
//! - `XlaPass`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;

use super::functions::XlaPass;
use super::types::{AlgebraicSimplificationPass, PassStatistics, XlaComputation, XlaConfig};

impl XlaPass for AlgebraicSimplificationPass {
    fn name(&self) -> &str {
        "algebraic-simplification"
    }
    fn run(&self, computation: &mut XlaComputation) -> Result<PassStatistics> {
        let mut stats = PassStatistics::new();
        let simplifications = Self::find_simplifications(computation);
        if simplifications > 0 {
            stats.nodes_modified = simplifications;
        }
        Ok(stats)
    }
    fn should_run(&self, config: &XlaConfig) -> bool {
        config.enable_algebraic_simplification && config.optimization_level >= 1
    }
}
