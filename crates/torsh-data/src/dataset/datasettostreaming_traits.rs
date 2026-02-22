//! # DatasetToStreaming - Trait Implementations
//!
//! This module contains trait implementations for `DatasetToStreaming`.
//!
//! ## Implemented Traits
//!
//! - `StreamingDataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::{Dataset, StreamingDataset};
use super::types::{DatasetToStreaming, DatasetToStreamingIter};

impl<D: Dataset + Clone> StreamingDataset for DatasetToStreaming<D> {
    type Item = D::Item;
    type Stream = DatasetToStreamingIter<D>;
    fn stream(&self) -> Self::Stream {
        DatasetToStreamingIter {
            dataset: self.dataset.clone(),
            current_index: 0,
            repeat: self.repeat,
        }
    }
    fn has_more(&self) -> bool {
        self.repeat || self.dataset.len() > 0
    }
    fn reset(&self) -> Result<()> {
        Ok(())
    }
}
