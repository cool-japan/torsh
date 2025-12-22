//! Core DataLoader implementation
//!
//! This module contains the fundamental DataLoader functionality including the main
//! DataLoader struct, its iterator, builder pattern, and core traits.

use crate::{
    collate::{Collate, DefaultCollate},
    dataset::Dataset,
    sampler::{BatchSampler, BatchingSampler, RandomSampler, SequentialSampler},
};
// ✅ SciRS2 POLICY: Use scirs2_core::parallel_ops instead of rayon::prelude
use scirs2_core::parallel_ops::*;
use torsh_core::error::Result;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Trait for DataLoader functionality
pub trait DataLoaderTrait<D: Dataset, C: Collate<D::Item>> {
    /// Get the number of batches
    fn len(&self) -> usize;

    /// Check if the dataloader is empty
    fn is_empty(&self) -> bool;
}

/// DataLoader for batching and iterating over datasets
///
/// The DataLoader provides an efficient way to iterate over datasets in batches,
/// with support for parallel loading, shuffling, and various optimization strategies.
///
/// # Type Parameters
///
/// - `D`: Dataset type implementing the Dataset trait
/// - `S`: Sampler type implementing the BatchSampler trait
/// - `C`: Collate function type implementing the Collate trait
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::core::DataLoader;
/// use torsh_data::dataset::TensorDataset;
///
/// let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
/// let dataloader = DataLoader::builder(dataset)
///     .batch_size(2)
///     .num_workers(4)
///     .build()?;
///
/// for batch in dataloader.iter() {
///     // Process batch
/// }
/// ```
pub struct DataLoader<D, S, C> {
    dataset: D,
    sampler: S,
    collate_fn: C,
    num_workers: usize,
    #[allow(dead_code)]
    pin_memory: bool,
    #[allow(dead_code)]
    drop_last: bool,
    #[allow(dead_code)]
    timeout: Option<std::time::Duration>,
}

impl<D: Dataset> DataLoader<D, (), ()> {
    /// Create a new DataLoader builder
    ///
    /// This provides a fluent API for configuring DataLoader options.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to iterate over
    ///
    /// # Returns
    ///
    /// A DataLoaderBuilder for configuring the DataLoader
    pub fn builder(dataset: D) -> DataLoaderBuilder<D> {
        DataLoaderBuilder::new(dataset)
    }
}

impl<D, S, C> DataLoader<D, S, C>
where
    D: Dataset,
    S: BatchSampler,
    C: Collate<D::Item>,
{
    /// Create an iterator over the dataset
    ///
    /// Returns a DataLoaderIterator that will yield batches according to
    /// the configured sampler and collation function.
    pub fn iter(&self) -> DataLoaderIterator<'_, D, S, C> {
        DataLoaderIterator {
            dataset: &self.dataset,
            sampler_iter: self.sampler.iter(),
            collate_fn: &self.collate_fn,
            num_workers: self.num_workers,
        }
    }

    /// Get the number of batches
    ///
    /// Returns the total number of batches that will be produced by this DataLoader,
    /// based on the underlying sampler's batch count.
    pub fn len(&self) -> usize {
        self.sampler.len()
    }

    /// Check if the dataloader is empty
    ///
    /// Returns true if the DataLoader will produce zero batches.
    pub fn is_empty(&self) -> bool {
        self.sampler.is_empty()
    }

    /// Get the dataset
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Get the sampler
    pub fn sampler(&self) -> &S {
        &self.sampler
    }

    /// Get the collate function
    pub fn collate_fn(&self) -> &C {
        &self.collate_fn
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
}

impl<D, S, C> DataLoaderTrait<D, C> for DataLoader<D, S, C>
where
    D: Dataset + Sync,
    S: BatchSampler + Sync,
    C: Collate<D::Item> + Sync,
    D::Item: Send,
    C::Output: Send,
    S::Iter: Iterator<Item = Vec<usize>>,
{
    fn len(&self) -> usize {
        self.sampler.len()
    }

    fn is_empty(&self) -> bool {
        self.sampler.is_empty()
    }
}

/// Iterator for DataLoader
///
/// This iterator handles the actual batch loading process, including parallel
/// processing when multiple workers are configured.
pub struct DataLoaderIterator<'a, D, S, C>
where
    D: Dataset,
    S: BatchSampler,
    C: Collate<D::Item>,
{
    dataset: &'a D,
    sampler_iter: S::Iter,
    collate_fn: &'a C,
    num_workers: usize,
}

impl<D, S, C> Iterator for DataLoaderIterator<'_, D, S, C>
where
    D: Dataset + Sync,
    D::Item: Send,
    S: BatchSampler,
    S::Iter: Iterator<Item = Vec<usize>>,
    C: Collate<D::Item> + Sync,
    C::Output: Send,
{
    type Item = Result<C::Output>;

    fn next(&mut self) -> Option<Self::Item> {
        let indices = self.sampler_iter.next()?;

        let batch_result = if self.num_workers > 1 {
            // Parallel loading using Rayon
            let samples: Result<Vec<_>> = indices
                .into_par_iter()
                .map(|idx| self.dataset.get(idx))
                .collect();

            match samples {
                Ok(samples) => self.collate_fn.collate(samples),
                Err(e) => return Some(Err(e)),
            }
        } else {
            // Sequential loading
            let mut samples = Vec::with_capacity(indices.len());
            for idx in indices {
                match self.dataset.get(idx) {
                    Ok(sample) => samples.push(sample),
                    Err(e) => return Some(Err(e)),
                }
            }
            self.collate_fn.collate(samples)
        };

        // Apply memory pinning if enabled
        match batch_result {
            Ok(batch) => {
                // Note: Memory pinning implementation would be applied here
                // For now, this is a placeholder for the pin_memory flag
                Some(Ok(batch))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

/// Builder for DataLoader
///
/// Provides a fluent API for configuring DataLoader instances with various options
/// such as batch size, shuffling, number of workers, and memory pinning.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::core::DataLoaderBuilder;
/// use torsh_data::dataset::TensorDataset;
///
/// let dataset = TensorDataset::new(vec![1, 2, 3, 4, 5]);
/// let dataloader = DataLoaderBuilder::new(dataset)
///     .batch_size(32)
///     .shuffle(true)
///     .num_workers(4)
///     .pin_memory(true)
///     .drop_last(true)
///     .build()?;
/// ```
pub struct DataLoaderBuilder<D: Dataset> {
    dataset: D,
    batch_size: Option<usize>,
    shuffle: bool,
    num_workers: usize,
    pin_memory: bool,
    drop_last: bool,
    timeout: Option<std::time::Duration>,
    generator: Option<u64>,
}

impl<D: Dataset> DataLoaderBuilder<D> {
    /// Create a new builder
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to create a DataLoader for
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: None,
            shuffle: false,
            num_workers: 0,
            pin_memory: false,
            drop_last: false,
            timeout: None,
            generator: None,
        }
    }

    /// Set batch size
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples per batch
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set whether to shuffle the data
    ///
    /// # Arguments
    ///
    /// * `shuffle` - Whether to randomly shuffle the dataset
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the number of worker threads
    ///
    /// # Arguments
    ///
    /// * `num_workers` - Number of worker threads for parallel data loading
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Set whether to pin memory
    ///
    /// # Arguments
    ///
    /// * `pin_memory` - Whether to pin memory for faster GPU transfers
    pub fn pin_memory(mut self, pin_memory: bool) -> Self {
        self.pin_memory = pin_memory;
        self
    }

    /// Set whether to drop the last incomplete batch
    ///
    /// # Arguments
    ///
    /// * `drop_last` - Whether to drop the last batch if it's smaller than batch_size
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set timeout for collecting a batch
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait for batch collection
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set random generator seed
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for reproducible shuffling
    pub fn generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Build the DataLoader with sequential sampling
    ///
    /// Creates a DataLoader that processes the dataset in sequential order.
    /// This is the default behavior when shuffle is false or not specified.
    pub fn build(
        self,
    ) -> Result<DataLoader<D, BatchingSampler<SequentialSampler>, DefaultCollate>> {
        let batch_size = self.batch_size.unwrap_or(1);
        let base_sampler = SequentialSampler::new(self.dataset.len());
        let batch_sampler = BatchingSampler::new(base_sampler, batch_size, self.drop_last);

        Ok(DataLoader {
            dataset: self.dataset,
            sampler: batch_sampler,
            collate_fn: DefaultCollate,
            num_workers: self.num_workers,
            pin_memory: self.pin_memory,
            drop_last: self.drop_last,
            timeout: self.timeout,
        })
    }

    /// Build the DataLoader with random sampling (shuffled)
    ///
    /// Creates a DataLoader that randomly shuffles the dataset order.
    /// Useful for training scenarios where data order should be randomized.
    pub fn build_with_random_sampling(
        self,
    ) -> Result<DataLoader<D, BatchingSampler<RandomSampler>, DefaultCollate>> {
        let batch_size = self.batch_size.unwrap_or(1);
        let mut base_sampler = RandomSampler::new(self.dataset.len(), None, false);

        if let Some(seed) = self.generator {
            base_sampler = base_sampler.with_generator(seed);
        }

        let batch_sampler = BatchingSampler::new(base_sampler, batch_size, self.drop_last);

        Ok(DataLoader {
            dataset: self.dataset,
            sampler: batch_sampler,
            collate_fn: DefaultCollate,
            num_workers: self.num_workers,
            pin_memory: self.pin_memory,
            drop_last: self.drop_last,
            timeout: self.timeout,
        })
    }

    /// Build the DataLoader with auto-selected sampling strategy
    ///
    /// Automatically chooses between sequential and random sampling based on
    /// the shuffle setting configured in the builder.
    pub fn build_auto(self) -> Result<Box<dyn DataLoaderTrait<D, DefaultCollate> + Send + Sync>>
    where
        D: Send + Sync + 'static,
        D::Item: Send + Sync + 'static,
        DefaultCollate: Collate<D::Item>,
        <DefaultCollate as Collate<D::Item>>::Output: Send,
    {
        if self.shuffle {
            Ok(Box::new(self.build_with_random_sampling()?))
        } else {
            Ok(Box::new(self.build()?))
        }
    }
}

/// Simplified DataLoader type for common use cases
///
/// This type alias provides a convenient shorthand for the most common DataLoader
/// configuration using sequential sampling and default collation.
pub type SimpleDataLoader<D> = DataLoader<D, BatchingSampler<SequentialSampler>, DefaultCollate>;

/// Simplified DataLoader type for random sampling use cases
///
/// This type alias provides a convenient shorthand for DataLoader with random sampling.
pub type RandomDataLoader<D> = DataLoader<D, BatchingSampler<RandomSampler>, DefaultCollate>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;

    #[test]
    fn test_dataloader_builder() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let builder = DataLoaderBuilder::new(dataset);

        assert_eq!(builder.batch_size, None);
        assert!(!builder.shuffle);
        assert_eq!(builder.num_workers, 0);
        assert!(!builder.pin_memory);
        assert!(!builder.drop_last);
    }

    #[test]
    fn test_dataloader_builder_configuration() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let builder = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .shuffle(true)
            .num_workers(4)
            .pin_memory(true)
            .drop_last(true);

        assert_eq!(builder.batch_size, Some(2));
        assert!(builder.shuffle);
        assert_eq!(builder.num_workers, 4);
        assert!(builder.pin_memory);
        assert!(builder.drop_last);
    }

    #[test]
    fn test_dataloader_sequential_build() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .build()
            .unwrap();

        assert_eq!(dataloader.len(), 3); // 5 items, batch size 2 = 3 batches (last with 1 item)
        assert!(!dataloader.is_empty());
    }

    #[test]
    fn test_dataloader_random_build() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .generator(42)
            .build_with_random_sampling()
            .unwrap();

        assert_eq!(dataloader.len(), 3);
        assert!(!dataloader.is_empty());
    }

    #[test]
    fn test_dataloader_iteration() {
        // Create a tensor with 4 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[4]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .build()
            .unwrap();

        let mut iter = dataloader.iter();
        let batch1 = iter.next().unwrap().unwrap();
        let batch2 = iter.next().unwrap().unwrap();
        assert!(iter.next().is_none());

        // Verify batch contents (each batch should have 1 stacked tensor)
        assert_eq!(batch1.len(), 1);
        assert_eq!(batch2.len(), 1);

        // Verify the stacked tensor shape: [batch_size, sample_features]
        // Original tensor is [4], each sample is [1], batched becomes [2, 1]
        assert_eq!(batch1[0].shape().dims(), &[2, 1]); // 2 samples batched
        assert_eq!(batch2[0].shape().dims(), &[2, 1]); // 2 samples batched
    }

    #[test]
    fn test_dataloader_drop_last() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .drop_last(true)
            .build()
            .unwrap();

        assert_eq!(dataloader.len(), 2); // 5 items, batch size 2, drop_last = 2 complete batches
    }

    #[test]
    fn test_dataloader_trait_implementation() {
        // Create a tensor with 5 samples (first dimension is number of samples)
        let tensor = torsh_tensor::creation::ones::<f32>(&[5]).unwrap();
        let dataset = TensorDataset::from_tensor(tensor);
        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .build()
            .unwrap();

        // Test trait methods
        assert_eq!(DataLoaderTrait::len(&dataloader), 3);
        assert!(!DataLoaderTrait::is_empty(&dataloader));
    }

    #[test]
    fn test_empty_dataloader() {
        let tensors: Vec<torsh_tensor::Tensor<f32>> = vec![];
        let dataset = TensorDataset::new(tensors);
        let dataloader = DataLoaderBuilder::new(dataset)
            .batch_size(2)
            .build()
            .unwrap();

        assert_eq!(dataloader.len(), 0);
        assert!(dataloader.is_empty());

        let mut iter = dataloader.iter();
        assert!(iter.next().is_none());
    }
}
