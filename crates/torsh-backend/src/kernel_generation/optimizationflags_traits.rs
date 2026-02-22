//! # OptimizationFlags - Trait Implementations
//!
//! This module contains trait implementations for `OptimizationFlags`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::OptimizationFlags;

impl Default for OptimizationFlags {
    fn default() -> Self {
        Self {
            vectorization: true,
            loop_unrolling: true,
            memory_coalescing: true,
            shared_memory_usage: true,
            tensor_cores: false,
            auto_tuning: false,
            aggressive_inlining: false,
            math_optimizations: true,
        }
    }
}
