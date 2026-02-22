//! # XlaPassManager - Trait Implementations
//!
//! This module contains trait implementations for `XlaPassManager`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AlgebraicSimplificationPass, CommonSubexpressionEliminationPass, ConstantFoldingPass,
    CopyEliminationPass, DeadCodeEliminationPass, LayoutOptimizationPass,
    MemoryAllocationOptimizationPass, OperationFusionPass, ParallelizationAnalysisPass,
    XlaPassManager,
};

impl Default for XlaPassManager {
    fn default() -> Self {
        let mut manager = Self {
            passes: Vec::new(),
            run_until_fixed_point: true,
            max_iterations: 10,
        };
        manager.add_pass(Box::new(ConstantFoldingPass));
        manager.add_pass(Box::new(AlgebraicSimplificationPass));
        manager.add_pass(Box::new(CopyEliminationPass));
        manager.add_pass(Box::new(CommonSubexpressionEliminationPass));
        manager.add_pass(Box::new(OperationFusionPass));
        manager.add_pass(Box::new(LayoutOptimizationPass));
        manager.add_pass(Box::new(MemoryAllocationOptimizationPass));
        manager.add_pass(Box::new(ParallelizationAnalysisPass));
        manager.add_pass(Box::new(DeadCodeEliminationPass));
        manager
    }
}
