//! # CachedDataset - Trait Implementations
//!
//! This module contains trait implementations for `CachedDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;

use super::functions::Dataset;
use super::types::CachedDataset;

impl<D: Dataset> Dataset for CachedDataset<D>
where
    D::Item: Clone + Send + Sync,
{
    type Item = D::Item;
    fn len(&self) -> usize {
        self.dataset.len()
    }
    fn get(&self, index: usize) -> Result<Self::Item> {
        {
            let mut access_count = self
                .access_count
                .write()
                .expect("lock should not be poisoned");
            *access_count.entry(index).or_insert(0) += 1;
        }
        {
            let cache = self.cache.read().expect("lock should not be poisoned");
            if let Some(item) = cache.get(&index) {
                return Ok(item.clone());
            }
        }
        let item = self.dataset.get(index)?;
        {
            self.evict_lru();
            let mut cache = self.cache.write().expect("lock should not be poisoned");
            cache.insert(index, item.clone());
        }
        Ok(item)
    }
}
