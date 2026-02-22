//! # CachedDataset - Trait Implementations
//!
//! This module contains trait implementations for `CachedDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Result;

use super::functions::Dataset;
use super::types::CachedDataset;

impl<D: Dataset> Dataset for CachedDataset<D>
where
    D::Item: Clone,
{
    type Item = D::Item;
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        self.inner.get_item(index)
    }
}
