//! # Subset - Trait Implementations
//!
//! This module contains trait implementations for `Subset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::Dataset;
use super::types::Subset;

impl<D: Dataset> Dataset for Subset<D> {
    type Item = D::Item;
    fn len(&self) -> usize {
        self.indices.len()
    }
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.indices.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }
        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}
