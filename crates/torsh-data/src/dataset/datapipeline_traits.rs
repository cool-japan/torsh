//! # DataPipeline - Trait Implementations
//!
//! This module contains trait implementations for `DataPipeline`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DataPipeline;

impl<T> Default for DataPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}
