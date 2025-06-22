//! DataLoader implementation for efficient data loading

use crate::{
    collate::{Collate, DefaultCollate},
    dataset::Dataset,
    sampler::{BatchSampler, BatchSamplerTrait, SequentialSampler},
};
use rayon::prelude::*;
use torsh_core::error::Result;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// DataLoader for batching and iterating over datasets
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
    pub fn builder(dataset: D) -> DataLoaderBuilder<D> {
        DataLoaderBuilder::new(dataset)
    }
}

impl<D, S, C> DataLoader<D, S, C>
where
    D: Dataset,
    S: BatchSamplerTrait,
    C: Collate<D::Item>,
{
    /// Create an iterator over the dataset
    pub fn iter(&self) -> DataLoaderIterator<D, S, C> {
        DataLoaderIterator {
            dataset: &self.dataset,
            sampler_iter: self.sampler.iter(),
            collate_fn: &self.collate_fn,
            num_workers: self.num_workers,
        }
    }

    /// Get the number of batches
    pub fn len(&self) -> usize {
        self.sampler.len()
    }

    /// Check if the dataloader is empty
    pub fn is_empty(&self) -> bool {
        self.sampler.is_empty()
    }
}

/// Iterator for DataLoader
pub struct DataLoaderIterator<'a, D, S, C>
where
    D: Dataset,
    S: BatchSamplerTrait,
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
    S: BatchSamplerTrait,
    S::Iter: Iterator<Item = Vec<usize>>,
    C: Collate<D::Item> + Sync,
    C::Output: Send,
{
    type Item = Result<C::Output>;

    fn next(&mut self) -> Option<Self::Item> {
        let indices = self.sampler_iter.next()?;

        let batch_result = if self.num_workers > 1 {
            // Parallel loading
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

        Some(batch_result)
    }
}

/// Builder for DataLoader
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
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Set whether to shuffle the data
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the number of worker threads
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Set whether to pin memory
    pub fn pin_memory(mut self, pin_memory: bool) -> Self {
        self.pin_memory = pin_memory;
        self
    }

    /// Set whether to drop the last incomplete batch
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set timeout for collecting a batch
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set random generator seed
    pub fn generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Build the DataLoader with default settings
    pub fn build(self) -> Result<DataLoader<D, BatchSampler<SequentialSampler>, DefaultCollate>> {
        let batch_size = self.batch_size.unwrap_or(1);

        // For simplicity, always use SequentialSampler for now
        // TODO: Add proper shuffle support with different return types
        let base_sampler = SequentialSampler::new(self.dataset.len());
        let batch_sampler = BatchSampler::new(base_sampler, batch_size, self.drop_last);

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
}

/// Simplified DataLoader type for common use cases
pub type SimpleDataLoader<D> = DataLoader<D, BatchSampler<SequentialSampler>, DefaultCollate>;

/// Create a simple DataLoader with basic settings
pub fn simple_dataloader<D: Dataset>(
    dataset: D,
    batch_size: usize,
    shuffle: bool,
) -> Result<SimpleDataLoader<D>> {
    let base_sampler = if shuffle {
        // We need concrete types here, so use SequentialSampler as placeholder
        SequentialSampler::new(dataset.len())
    } else {
        SequentialSampler::new(dataset.len())
    };

    let batch_sampler = BatchSampler::new(base_sampler, batch_size, false);

    Ok(DataLoader {
        dataset,
        sampler: batch_sampler,
        collate_fn: DefaultCollate,
        num_workers: 0,
        pin_memory: false,
        drop_last: false,
        timeout: None,
    })
}

/// Prefetch iterator for performance optimization
pub struct PrefetchIterator<T> {
    receiver: crossbeam::channel::Receiver<Option<T>>,
    _handle: std::thread::JoinHandle<()>,
}

impl<T> PrefetchIterator<T>
where
    T: Send + 'static,
{
    /// Create a new prefetch iterator
    pub fn new<I>(inner: I, buffer_size: usize) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
    {
        let (sender, receiver) = crossbeam::channel::bounded(buffer_size);

        let handle = std::thread::spawn(move || {
            for item in inner {
                if sender.send(Some(item)).is_err() {
                    // Receiver has been dropped, stop producing
                    break;
                }
            }
            // Send None to signal end of iteration
            let _ = sender.send(None);
        });

        Self {
            receiver,
            _handle: handle,
        }
    }
}

impl<T> Iterator for PrefetchIterator<T>
where
    T: Send + 'static,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(Some(item)) => Some(item),
            Ok(None) | Err(_) => None,
        }
    }
}

/// Extension trait for adding prefetching to iterators
pub trait PrefetchExt<T>: Iterator<Item = T> + Sized + Send + 'static
where
    T: Send + 'static,
{
    /// Add prefetching to the iterator
    fn prefetch(self, buffer_size: usize) -> PrefetchIterator<T> {
        PrefetchIterator::new(self, buffer_size)
    }
}

impl<I, T> PrefetchExt<T> for I
where
    I: Iterator<Item = T> + Send + 'static,
    T: Send + 'static,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TensorDataset;
    use torsh_tensor::creation::*;

    #[test]
    fn test_dataloader_builder() {
        let data = ones::<f32>(&[10, 3]);
        let dataset = TensorDataset::from_tensor(data);

        let dataloader = DataLoader::builder(dataset)
            .batch_size(2)
            .shuffle(false)
            .num_workers(1)
            .build()
            .unwrap();

        assert_eq!(dataloader.len(), 5); // 10 samples / 2 batch_size
    }

    #[test]
    fn test_simple_dataloader() {
        let data = ones::<f32>(&[10, 3]);
        let dataset = TensorDataset::from_tensor(data);

        let dataloader = simple_dataloader(dataset, 3, false).unwrap();
        assert_eq!(dataloader.len(), 4); // ceil(10 / 3)
    }

    #[test]
    fn test_dataloader_iteration() {
        let data = ones::<f32>(&[6, 2]);
        let dataset = TensorDataset::from_tensor(data);

        let dataloader = simple_dataloader(dataset, 2, false).unwrap();

        let mut batch_count = 0;
        for batch_result in dataloader.iter() {
            assert!(batch_result.is_ok());
            batch_count += 1;
        }

        assert_eq!(batch_count, 3); // 6 samples / 2 batch_size
    }

    #[test]
    fn test_prefetch_iterator() {
        let data = vec![1, 2, 3, 4, 5];
        let iter = data.into_iter();

        let mut prefetch_iter = iter.prefetch(2);
        let mut collected = Vec::new();

        while let Some(item) = prefetch_iter.next() {
            collected.push(item);
        }

        assert_eq!(collected, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_parallel_dataloader() {
        let data = ones::<f32>(&[8, 2]);
        let dataset = TensorDataset::from_tensor(data);

        let dataloader = DataLoader::builder(dataset)
            .batch_size(2)
            .num_workers(2) // Use 2 workers for parallel loading
            .build()
            .unwrap();

        let mut batch_count = 0;
        for batch_result in dataloader.iter() {
            assert!(batch_result.is_ok());
            batch_count += 1;
        }

        assert_eq!(batch_count, 4); // 8 samples / 2 batch_size
    }
}
