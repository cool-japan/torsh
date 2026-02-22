//! # AgNewsDataset - Trait Implementations
//!
//! This module contains trait implementations for `AgNewsDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::AgNewsDataset;

impl Dataset for AgNewsDataset {
    type Item = (String, usize);
    fn len(&self) -> usize {
        self.texts.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.texts.len() {
            Ok((self.texts[index].clone(), self.labels[index]))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
}
