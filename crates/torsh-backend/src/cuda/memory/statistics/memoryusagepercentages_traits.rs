//! # MemoryUsagePercentages - Trait Implementations
//!
//! This module contains trait implementations for `MemoryUsagePercentages`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MemoryUsagePercentages;

impl Default for MemoryUsagePercentages {
    fn default() -> Self {
        Self {
            device_percentage: 0.0,
            unified_percentage: 0.0,
            pinned_percentage: 0.0,
        }
    }
}
