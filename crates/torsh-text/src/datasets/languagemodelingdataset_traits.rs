//! # LanguageModelingDataset - Trait Implementations
//!
//! This module contains trait implementations for `LanguageModelingDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::LanguageModelingDataset;

impl Dataset for LanguageModelingDataset {
    type Item = String;
    fn len(&self) -> usize {
        let full_text = self.texts.join(" ");
        let text_len = full_text.chars().count();
        if text_len < self.sequence_length {
            0
        } else {
            (text_len - self.sequence_length) / self.stride + 1
        }
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        let sequences = self.get_sequences();
        sequences
            .get(index)
            .cloned()
            .ok_or_else(|| TextError::DatasetError(format!("Index {} out of bounds", index)))
    }
}
