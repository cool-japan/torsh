//! # RealTimeDatasetIter - Trait Implementations
//!
//! This module contains trait implementations for `RealTimeDatasetIter`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::types::RealTimeDatasetIter;

impl<T> Iterator for RealTimeDatasetIter<T> {
    type Item = Result<T>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.try_recv() {
            Ok(item) => Some(Ok(item)),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
        }
    }
}
