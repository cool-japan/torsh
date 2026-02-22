//! # ProfiledDataset - Trait Implementations
//!
//! This module contains trait implementations for `ProfiledDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::Dataset;
use super::types::ProfiledDataset;

#[cfg(feature = "std")]
impl<D: Dataset> Dataset for ProfiledDataset<D> {
    type Item = D::Item;
    fn len(&self) -> usize {
        self.dataset.len()
    }
    fn get(&self, index: usize) -> Result<Self::Item> {
        let start = std::time::Instant::now();
        let result = self.dataset.get(index);
        let duration = start.elapsed();
        self.profiler.record_access(index, duration);
        result
    }
}
