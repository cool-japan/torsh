//! # TranslationDataset - Trait Implementations
//!
//! This module contains trait implementations for `TranslationDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::TranslationDataset;

impl Dataset for TranslationDataset {
    type Item = (String, String);
    fn len(&self) -> usize {
        self.source_texts.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.source_texts.len() {
            Ok((
                self.source_texts[index].clone(),
                self.target_texts[index].clone(),
            ))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
}
