//! # MemoryUsageBreakdown - Trait Implementations
//!
//! This module contains trait implementations for `MemoryUsageBreakdown`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MemoryUsageBreakdown, MemoryUsagePercentages};

impl Default for MemoryUsageBreakdown {
    fn default() -> Self {
        Self {
            device_memory: 0,
            unified_memory: 0,
            pinned_memory: 0,
            total_usage: 0,
            usage_percentages: MemoryUsagePercentages::default(),
        }
    }
}
