//! # BatchIterator - Trait Implementations
//!
//! This module contains trait implementations for `BatchIterator`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Result;

use super::functions::Dataset;
use super::types::BatchIterator;

impl<'a, D: Dataset> Iterator for BatchIterator<'a, D>
where
    D::Item: Clone,
{
    type Item = Result<Vec<D::Item>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.dataset.len() {
            return None;
        }
        let end = (self.index + self.batch_size).min(self.dataset.len());
        let mut batch = Vec::with_capacity(end - self.index);
        for i in self.index..end {
            match self.dataset.get_item(i) {
                Ok(item) => batch.push(item),
                Err(e) => return Some(Err(e)),
            }
        }
        self.index = end;
        Some(Ok(batch))
    }
}
