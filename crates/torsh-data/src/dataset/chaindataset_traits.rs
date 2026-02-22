//! # ChainDataset - Trait Implementations
//!
//! This module contains trait implementations for `ChainDataset`.
//!
//! ## Implemented Traits
//!
//! - `IterableDataset`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::IterableDataset;
use super::types::{ChainDataset, ChainDatasetIter};

impl<D: IterableDataset + Clone> IterableDataset for ChainDataset<D> {
    type Item = D::Item;
    type Iter = ChainDatasetIter<D>;
    fn iter(&self) -> Self::Iter {
        let current_iter = if !self.datasets.is_empty() {
            Some(self.datasets[0].iter())
        } else {
            None
        };
        ChainDatasetIter {
            datasets: self.datasets.clone(),
            current_index: 0,
            current_iter,
        }
    }
}

impl<D: IterableDataset + Clone> Clone for ChainDataset<D> {
    fn clone(&self) -> Self {
        Self {
            datasets: self.datasets.clone(),
        }
    }
}
