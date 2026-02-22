//! # Multi30kDataset - Trait Implementations
//!
//! This module contains trait implementations for `Multi30kDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::Multi30kDataset;

impl Dataset for Multi30kDataset {
    type Item = (String, String);
    fn len(&self) -> usize {
        self.english_sentences.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.english_sentences.len() {
            Ok((
                self.english_sentences[index].clone(),
                self.german_sentences[index].clone(),
            ))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
}
