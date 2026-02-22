//! # DatasetIterator - Trait Implementations
//!
//! This module contains trait implementations for `DatasetIterator`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Result;

use super::functions::Dataset;
use super::types::DatasetIterator;

impl<'a, D: Dataset> Iterator for DatasetIterator<'a, D> {
    type Item = Result<D::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.dataset.len() {
            let item = self.dataset.get_item(self.index);
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}
