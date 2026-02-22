//! # ConsolidatedDataset - Trait Implementations
//!
//! This module contains trait implementations for `ConsolidatedDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::{ConsolidatedDataset, DataItem};

impl Dataset for ConsolidatedDataset {
    type Item = DataItem;
    fn len(&self) -> usize {
        self.items.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        self.items
            .get(index)
            .cloned()
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
    }
}
