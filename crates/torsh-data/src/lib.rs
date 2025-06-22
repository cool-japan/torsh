//! Data loading and preprocessing utilities for ToRSh
//!
//! This crate provides PyTorch-compatible data loading functionality,
//! including datasets, data loaders, and common transformations.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod dataset;
pub mod dataloader;
pub mod sampler;
pub mod collate;
pub mod transforms;

#[cfg(feature = "image-support")]
pub mod vision;

#[cfg(feature = "dataframe")]
pub mod tabular;

#[cfg(feature = "audio-support")]
pub mod audio;

pub use dataset::{Dataset, IterableDataset, TensorDataset};
pub use dataloader::{DataLoader, DataLoaderBuilder};
pub use sampler::{Sampler, SequentialSampler, RandomSampler, BatchSampler};
pub use collate::{collate_fn, Collate};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::dataset::*;
    pub use crate::dataloader::*;
    pub use crate::sampler::*;
    pub use crate::collate::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_imports() {
        // Basic smoke test
        let _ = SequentialSampler::new(10);
    }
}