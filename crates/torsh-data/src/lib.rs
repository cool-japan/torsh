//! Data loading and preprocessing utilities for ToRSh
//!
//! This crate provides PyTorch-compatible data loading functionality,
//! including datasets, data loaders, and common transformations.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod collate;
pub mod dataloader;
pub mod dataset;
pub mod sampler;
pub mod transforms;

#[cfg(feature = "image-support")]
pub mod vision;

#[cfg(feature = "dataframe")]
pub mod tabular;

#[cfg(feature = "audio-support")]
pub mod audio;

pub use collate::{collate_fn, Collate};
pub use dataloader::{DataLoader, DataLoaderBuilder};
pub use dataset::{Dataset, IterableDataset, TensorDataset};
pub use sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::collate::*;
    pub use crate::dataloader::*;
    pub use crate::dataset::*;
    pub use crate::sampler::*;
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
