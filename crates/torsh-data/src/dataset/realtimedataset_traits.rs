//! # RealTimeDataset - Trait Implementations
//!
//! This module contains trait implementations for `RealTimeDataset`.
//!
//! ## Implemented Traits
//!
//! - `StreamingDataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::StreamingDataset;
use super::types::{RealTimeDataset, RealTimeDatasetIter};

impl<T: Send + Sync + 'static> StreamingDataset for RealTimeDataset<T> {
    type Item = T;
    type Stream = RealTimeDatasetIter<T>;
    fn stream(&self) -> Self::Stream {
        let (_, receiver) = std::sync::mpsc::channel();
        RealTimeDatasetIter { receiver }
    }
    fn has_more(&self) -> bool {
        true
    }
}
