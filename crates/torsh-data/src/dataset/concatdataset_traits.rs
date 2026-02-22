//! # ConcatDataset - Trait Implementations
//!
//! This module contains trait implementations for `ConcatDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::Dataset;
use super::types::ConcatDataset;

impl<D: Dataset> Dataset for ConcatDataset<D> {
    type Item = D::Item;
    fn len(&self) -> usize {
        self.cumulative_sizes.last().copied().unwrap_or(0)
    }
    fn get(&self, index: usize) -> Result<Self::Item> {
        if let Some((dataset_idx, sample_idx)) = self.dataset_idx(index) {
            self.datasets[dataset_idx].get(sample_idx)
        } else {
            Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            })
        }
    }
}
