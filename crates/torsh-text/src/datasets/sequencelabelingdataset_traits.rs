//! # SequenceLabelingDataset - Trait Implementations
//!
//! This module contains trait implementations for `SequenceLabelingDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{Result, TextError};

use super::functions::Dataset;
use super::types::SequenceLabelingDataset;

impl Dataset for SequenceLabelingDataset {
    type Item = (Vec<String>, Vec<String>);
    fn len(&self) -> usize {
        self.sequences.len()
    }
    fn get_item(&self, index: usize) -> Result<Self::Item> {
        if index < self.sequences.len() {
            Ok((self.sequences[index].clone(), self.labels[index].clone()))
        } else {
            Err(TextError::DatasetError(format!(
                "Index {} out of bounds",
                index
            )))
        }
    }
}
