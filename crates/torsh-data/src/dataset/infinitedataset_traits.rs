//! # InfiniteDataset - Trait Implementations
//!
//! This module contains trait implementations for `InfiniteDataset`.
//!
//! ## Implemented Traits
//!
//! - `StreamingDataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::StreamingDataset;
use super::types::{InfiniteDataset, InfiniteDatasetIter};

impl<F, T> StreamingDataset for InfiniteDataset<F, T>
where
    F: Fn() -> Result<T> + Send + Sync + Clone,
{
    type Item = T;
    type Stream = InfiniteDatasetIter<F, T>;
    fn stream(&self) -> Self::Stream {
        InfiniteDatasetIter {
            generator: self.generator.clone(),
        }
    }
    fn has_more(&self) -> bool {
        true
    }
}
