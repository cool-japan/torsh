//! # ChainDatasetIter - Trait Implementations
//!
//! This module contains trait implementations for `ChainDatasetIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::IterableDataset;
use super::types::ChainDatasetIter;

impl<D: IterableDataset + Clone> Iterator for ChainDatasetIter<D> {
    type Item = Result<D::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut iter) = self.current_iter {
                if let Some(item) = iter.next() {
                    return Some(item);
                }
            }
            self.current_index += 1;
            if self.current_index >= self.datasets.len() {
                return None;
            }
            self.current_iter = Some(self.datasets[self.current_index].iter());
        }
    }
}
