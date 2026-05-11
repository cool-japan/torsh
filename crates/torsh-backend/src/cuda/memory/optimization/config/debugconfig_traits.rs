//! # DebugConfig - Trait Implementations
//!
//! This module contains trait implementations for `DebugConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DebugConfig;

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enable_debug_logging: false,
            enable_performance_profiling: false,
        }
    }
}

