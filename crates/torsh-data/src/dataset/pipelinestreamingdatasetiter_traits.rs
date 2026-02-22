//! # PipelineStreamingDatasetIter - Trait Implementations
//!
//! This module contains trait implementations for `PipelineStreamingDatasetIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::StreamingDataset;
use super::types::PipelineStreamingDatasetIter;

impl<S: StreamingDataset<Item = T>, T> Iterator for PipelineStreamingDatasetIter<S, T> {
    type Item = Result<T>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.source_iter.next()? {
            Ok(item) => match self.pipeline.apply(item) {
                Ok(transformed) => Some(Ok(transformed)),
                Err(e) => Some(Err(e)),
            },
            Err(e) => Some(Err(e)),
        }
    }
}
