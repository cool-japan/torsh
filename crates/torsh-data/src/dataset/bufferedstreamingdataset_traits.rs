//! # BufferedStreamingDataset - Trait Implementations
//!
//! This module contains trait implementations for `BufferedStreamingDataset`.
//!
//! ## Implemented Traits
//!
//! - `StreamingDataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::StreamingDataset;
use super::types::{BufferedStreamingDataset, BufferedStreamingDatasetIter};

impl<S: StreamingDataset> StreamingDataset for BufferedStreamingDataset<S>
where
    S::Item: Send,
{
    type Item = S::Item;
    type Stream = BufferedStreamingDatasetIter<S>;
    fn stream(&self) -> Self::Stream {
        BufferedStreamingDatasetIter {
            source_iter: self.source.stream(),
            buffer: std::collections::VecDeque::with_capacity(self.buffer_size),
            buffer_size: self.buffer_size,
            prefetch: self.prefetch,
        }
    }
    fn has_more(&self) -> bool {
        self.source.has_more()
    }
    fn reset(&self) -> Result<()> {
        self.source.reset()
    }
}
