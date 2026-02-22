//! # DatasetToStreamingIter - Trait Implementations
//!
//! This module contains trait implementations for `DatasetToStreamingIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::Dataset;
use super::types::DatasetToStreamingIter;

impl<D: Dataset> Iterator for DatasetToStreamingIter<D> {
    type Item = Result<D::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.dataset.len() {
            if self.repeat {
                self.current_index = 0;
            } else {
                return None;
            }
        }
        let result = self.dataset.get(self.current_index);
        self.current_index += 1;
        Some(result)
    }
}
