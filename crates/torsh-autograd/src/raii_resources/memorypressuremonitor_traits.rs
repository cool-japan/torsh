//! # MemoryPressureMonitor - Trait Implementations
//!
//! This module contains trait implementations for `MemoryPressureMonitor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MemoryPressureMonitor;

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self::new()
    }
}
