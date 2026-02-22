//! # XlaConfig - Trait Implementations
//!
//! This module contains trait implementations for `XlaConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{XlaConfig, XlaTarget};

impl Default for XlaConfig {
    fn default() -> Self {
        Self {
            target: XlaTarget::Cpu,
            enable_fusion: true,
            enable_layout_optimization: true,
            enable_algebraic_simplification: true,
            optimization_level: 2,
        }
    }
}
