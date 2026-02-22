//! # PipelineStreamingDataset - Trait Implementations
//!
//! This module contains trait implementations for `PipelineStreamingDataset`.
//!
//! ## Implemented Traits
//!
//! - `StreamingDataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::StreamingDataset;
use super::types::{PipelineStreamingDataset, PipelineStreamingDatasetIter};

impl<S: StreamingDataset<Item = T>, T> StreamingDataset for PipelineStreamingDataset<S, T> {
    type Item = T;
    type Stream = PipelineStreamingDatasetIter<S, T>;
    fn stream(&self) -> Self::Stream {
        PipelineStreamingDatasetIter {
            source_iter: self.source.stream(),
            pipeline: self.pipeline.clone(),
        }
    }
    fn has_more(&self) -> bool {
        self.source.has_more()
    }
    fn reset(&self) -> Result<()> {
        self.source.reset()
    }
}
