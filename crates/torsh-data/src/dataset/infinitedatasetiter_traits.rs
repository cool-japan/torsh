//! # InfiniteDatasetIter - Trait Implementations
//!
//! This module contains trait implementations for `InfiniteDatasetIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::types::InfiniteDatasetIter;

impl<F, T> Iterator for InfiniteDatasetIter<F, T>
where
    F: Fn() -> Result<T> + Send + Sync,
{
    type Item = Result<T>;
    fn next(&mut self) -> Option<Self::Item> {
        Some((self.generator)())
    }
}
