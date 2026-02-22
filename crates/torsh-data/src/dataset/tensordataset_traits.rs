//! # TensorDataset - Trait Implementations
//!
//! This module contains trait implementations for `TensorDataset`.
//!
//! ## Implemented Traits
//!
//! - `Dataset`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use torsh_core::error::Result;
use torsh_tensor::Tensor;

use super::functions::Dataset;
use super::types::TensorDataset;

impl<T: torsh_core::dtype::TensorElement> Dataset for TensorDataset<T> {
    type Item = Vec<Tensor<T>>;
    fn len(&self) -> usize {
        if self.tensors.is_empty() {
            0
        } else {
            self.tensors[0].size(0).unwrap_or(0)
        }
    }
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }
        let mut items = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            let selected = tensor.select(0, index as i64)?;
            items.push(selected);
        }
        Ok(items)
    }
}
