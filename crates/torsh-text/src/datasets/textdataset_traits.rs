//! # TextDataset - Trait Implementations
//!
//! This module contains trait implementations for `TextDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::TextDataset;

impl Dataset for TextDataset {
    type Item = String;
    fn len(&self) -> usize {
        self.data.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        self.data
            .get(index)
            .cloned()
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
    }
}

impl Default for TextDataset {
    fn default() -> Self {
        Self::new()
    }
}
