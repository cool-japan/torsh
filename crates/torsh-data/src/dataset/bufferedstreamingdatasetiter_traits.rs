//! # BufferedStreamingDatasetIter - Trait Implementations
//!
//! This module contains trait implementations for `BufferedStreamingDatasetIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::StreamingDataset;
use super::types::BufferedStreamingDatasetIter;

impl<S: StreamingDataset> Iterator for BufferedStreamingDatasetIter<S> {
    type Item = Result<S::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.prefetch {
            while self.buffer.len() < self.buffer_size {
                if let Some(item) = self.source_iter.next() {
                    self.buffer.push_back(item);
                } else {
                    break;
                }
            }
        }
        if let Some(item) = self.buffer.pop_front() {
            Some(item)
        } else {
            self.source_iter.next()
        }
    }
}
