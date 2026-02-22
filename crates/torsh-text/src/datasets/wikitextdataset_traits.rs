//! # WikiTextDataset - Trait Implementations
//!
//! This module contains trait implementations for `WikiTextDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::WikiTextDataset;

impl Dataset for WikiTextDataset {
    type Item = String;
    fn len(&self) -> usize {
        self.articles.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        self.articles
            .get(index)
            .cloned()
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
    }
}
