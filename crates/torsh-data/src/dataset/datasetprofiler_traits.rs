//! # DatasetProfiler - Trait Implementations
//!
//! This module contains trait implementations for `DatasetProfiler`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DatasetProfiler;

#[cfg(feature = "std")]
impl Default for DatasetProfiler {
    fn default() -> Self {
        Self::new()
    }
}
