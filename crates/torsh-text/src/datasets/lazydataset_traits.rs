//! # LazyDataset - Trait Implementations
//!
//! This module contains trait implementations for `LazyDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::LazyDataset;

impl Dataset for LazyDataset {
    type Item = String;
    fn len(&self) -> usize {
        self.total_lines
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        let content = std::fs::read_to_string(&self.file_path)?;
        content
            .lines()
            .nth(index)
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
            .map(|s| s.to_string())
    }
}
