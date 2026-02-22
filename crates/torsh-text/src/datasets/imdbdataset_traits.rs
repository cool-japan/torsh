//! # ImdbDataset - Trait Implementations
//!
//! This module contains trait implementations for `ImdbDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::ImdbDataset;

impl Dataset for ImdbDataset {
    type Item = (String, bool);
    fn len(&self) -> usize {
        self.reviews.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.reviews.len() {
            Ok((self.reviews[index].clone(), self.labels[index]))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
}
