//! Batch sampling functionality
//!
//! This module provides utilities for converting individual samplers into
//! batch samplers that yield batches of indices instead of individual indices.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::core::{BatchSampler, Sampler};

/// Wrapper that converts any sampler into a batch sampler
///
/// This sampler takes an underlying sampler and groups its output into batches
/// of a specified size. The last batch may be smaller than the batch size
/// unless `drop_last` is set to true.
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::{SequentialSampler, BatchingSampler, BatchSampler};
///
/// let base_sampler = SequentialSampler::new(10);
/// let batch_sampler = BatchingSampler::new(base_sampler, 3, false);
///
/// let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
/// assert_eq!(batches.len(), 4); // [0,1,2], [3,4,5], [6,7,8], [9]
/// ```
#[derive(Debug, Clone)]
pub struct BatchingSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchingSampler<S> {
    /// Create a new batching sampler
    ///
    /// # Arguments
    ///
    /// * `sampler` - The underlying sampler to batch
    /// * `batch_size` - Size of each batch
    /// * `drop_last` - Whether to drop the last incomplete batch
    ///
    /// # Panics
    ///
    /// Panics if `batch_size` is 0
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        assert!(batch_size > 0, "Batch size must be positive");
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Check if dropping last incomplete batch
    pub fn drop_last(&self) -> bool {
        self.drop_last
    }

    /// Get a reference to the underlying sampler
    pub fn sampler(&self) -> &S {
        &self.sampler
    }

    /// Get the underlying sampler by value
    pub fn into_sampler(self) -> S {
        self.sampler
    }

    /// Convert this batching sampler into a distributed version
    ///
    /// This creates a distributed wrapper around the underlying sampler
    /// and then wraps it with a new BatchingSampler.
    ///
    /// # Arguments
    ///
    /// * `num_replicas` - Total number of processes
    /// * `rank` - Current process rank (0-based)
    pub fn into_distributed(
        self,
        num_replicas: usize,
        rank: usize,
    ) -> BatchingSampler<super::distributed::DistributedWrapper<S>> {
        let distributed_sampler = self.sampler.into_distributed(num_replicas, rank);
        BatchingSampler::new(distributed_sampler, self.batch_size, self.drop_last)
    }
}

impl<S: Sampler> BatchSampler for BatchingSampler<S> {
    type Iter = BatchSamplerIter<S::Iter>;

    fn iter(&self) -> Self::Iter {
        BatchSamplerIter::new(self.sampler.iter(), self.batch_size, self.drop_last)
    }

    fn num_batches(&self) -> usize {
        let total_samples = self.sampler.len();
        if total_samples == 0 {
            return 0;
        }

        if self.drop_last {
            total_samples / self.batch_size
        } else {
            (total_samples + self.batch_size - 1) / self.batch_size
        }
    }
}

/// Iterator that groups indices from an underlying iterator into batches
#[derive(Debug)]
pub struct BatchSamplerIter<I: Iterator<Item = usize>> {
    inner: I,
    batch_size: usize,
    drop_last: bool,
}

impl<I: Iterator<Item = usize>> BatchSamplerIter<I> {
    /// Create a new batch sampler iterator
    pub fn new(inner: I, batch_size: usize, drop_last: bool) -> Self {
        Self {
            inner,
            batch_size,
            drop_last,
        }
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Check if dropping last incomplete batch
    pub fn drop_last(&self) -> bool {
        self.drop_last
    }
}

impl<I: Iterator<Item = usize>> Iterator for BatchSamplerIter<I> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);

        // Collect items for this batch
        for _ in 0..self.batch_size {
            if let Some(item) = self.inner.next() {
                batch.push(item);
            } else {
                break;
            }
        }

        if batch.is_empty() {
            None
        } else if batch.len() < self.batch_size && self.drop_last {
            None
        } else {
            Some(batch)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.inner.size_hint();

        let lower_batches = if self.drop_last {
            lower / self.batch_size
        } else {
            (lower + self.batch_size - 1) / self.batch_size
        };

        let upper_batches = upper.map(|u| {
            if self.drop_last {
                u / self.batch_size
            } else {
                (u + self.batch_size - 1) / self.batch_size
            }
        });

        (lower_batches, upper_batches)
    }
}

/// Create a batch sampler from any sampler
///
/// Convenience function for creating a batch sampler.
///
/// # Arguments
///
/// * `sampler` - The underlying sampler
/// * `batch_size` - Size of each batch
/// * `drop_last` - Whether to drop the last incomplete batch
pub fn batch<S: Sampler>(sampler: S, batch_size: usize, drop_last: bool) -> BatchingSampler<S> {
    BatchingSampler::new(sampler, batch_size, drop_last)
}

/// Create a batch sampler that keeps the last incomplete batch
///
/// Convenience function for creating a batch sampler that doesn't drop
/// the last batch even if it's incomplete.
///
/// # Arguments
///
/// * `sampler` - The underlying sampler
/// * `batch_size` - Size of each batch
pub fn batch_keep_last<S: Sampler>(sampler: S, batch_size: usize) -> BatchingSampler<S> {
    BatchingSampler::new(sampler, batch_size, false)
}

/// Create a batch sampler that drops the last incomplete batch
///
/// Convenience function for creating a batch sampler that drops
/// the last batch if it's incomplete.
///
/// # Arguments
///
/// * `sampler` - The underlying sampler
/// * `batch_size` - Size of each batch
pub fn batch_drop_last<S: Sampler>(sampler: S, batch_size: usize) -> BatchingSampler<S> {
    BatchingSampler::new(sampler, batch_size, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::basic::SequentialSampler;

    #[test]
    fn test_batching_sampler_basic() {
        let base_sampler = SequentialSampler::new(10);
        let batch_sampler = BatchingSampler::new(base_sampler, 3, false);

        assert_eq!(batch_sampler.batch_size(), 3);
        assert!(!batch_sampler.drop_last());
        assert_eq!(batch_sampler.num_batches(), 4); // 10 items, 3 per batch = 4 batches

        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
        assert_eq!(batches[3], vec![9]); // Last incomplete batch
    }

    #[test]
    fn test_batching_sampler_drop_last() {
        let base_sampler = SequentialSampler::new(10);
        let batch_sampler = BatchingSampler::new(base_sampler, 3, true);

        assert!(batch_sampler.drop_last());
        assert_eq!(batch_sampler.num_batches(), 3); // Drops last incomplete batch

        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
        // Last batch [9] is dropped
    }

    #[test]
    fn test_batching_sampler_exact_division() {
        let base_sampler = SequentialSampler::new(9);
        let batch_sampler = BatchingSampler::new(base_sampler, 3, true);

        assert_eq!(batch_sampler.num_batches(), 3);

        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
    }

    #[test]
    fn test_batching_sampler_empty() {
        let base_sampler = SequentialSampler::new(0);
        let batch_sampler = BatchingSampler::new(base_sampler, 3, false);

        assert_eq!(batch_sampler.num_batches(), 0);
        assert!(batch_sampler.is_empty());

        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 0);
    }

    #[test]
    fn test_batching_sampler_single_item() {
        let base_sampler = SequentialSampler::new(1);
        let batch_sampler = BatchingSampler::new(base_sampler, 3, false);

        assert_eq!(batch_sampler.num_batches(), 1);

        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], vec![0]);
    }

    #[test]
    fn test_batching_sampler_single_item_drop_last() {
        let base_sampler = SequentialSampler::new(1);
        let batch_sampler = BatchingSampler::new(base_sampler, 3, true);

        assert_eq!(batch_sampler.num_batches(), 0);

        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 0);
    }

    #[test]
    #[should_panic(expected = "Batch size must be positive")]
    fn test_batching_sampler_zero_batch_size() {
        let base_sampler = SequentialSampler::new(10);
        BatchingSampler::new(base_sampler, 0, false);
    }

    #[test]
    fn test_batch_sampler_iter_size_hint() {
        let base_sampler = SequentialSampler::new(10);
        let batch_sampler = BatchingSampler::new(base_sampler, 3, false);

        let iter = batch_sampler.iter();
        assert_eq!(iter.size_hint(), (4, Some(4)));

        let batch_sampler_drop = BatchingSampler::new(SequentialSampler::new(10), 3, true);
        let iter_drop = batch_sampler_drop.iter();
        assert_eq!(iter_drop.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_batching_sampler_into_sampler() {
        let base_sampler = SequentialSampler::new(5);
        let batch_sampler = BatchingSampler::new(base_sampler, 2, false);

        let recovered_sampler = batch_sampler.into_sampler();
        assert_eq!(recovered_sampler.len(), 5);
    }

    #[test]
    fn test_convenience_functions() {
        let base_sampler = SequentialSampler::new(10);

        let batch_keep = batch_keep_last(base_sampler.clone(), 3);
        assert!(!batch_keep.drop_last());
        assert_eq!(batch_keep.num_batches(), 4);

        let batch_drop = batch_drop_last(base_sampler.clone(), 3);
        assert!(batch_drop.drop_last());
        assert_eq!(batch_drop.num_batches(), 3);

        let batch_generic = batch(base_sampler, 3, true);
        assert!(batch_generic.drop_last());
        assert_eq!(batch_generic.num_batches(), 3);
    }

    #[test]
    fn test_batch_sampler_iter_properties() {
        let base_sampler = SequentialSampler::new(7);
        let batch_sampler = BatchingSampler::new(base_sampler, 3, false);

        let mut iter = batch_sampler.iter();
        assert_eq!(iter.batch_size(), 3);
        assert!(!iter.drop_last());

        // Test collecting batches one by one
        let batch1 = iter.next().unwrap();
        assert_eq!(batch1, vec![0, 1, 2]);

        let batch2 = iter.next().unwrap();
        assert_eq!(batch2, vec![3, 4, 5]);

        let batch3 = iter.next().unwrap();
        assert_eq!(batch3, vec![6]);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_batch_sizes() {
        // Test various batch sizes
        let test_cases = vec![
            (10, 1, false, 10), // Each item is its own batch
            (10, 10, false, 1), // Single batch with all items
            (10, 15, false, 1), // Batch size larger than dataset
            (0, 5, false, 0),   // Empty dataset
        ];

        for (dataset_size, batch_size, drop_last, expected_batches) in test_cases {
            if dataset_size == 0 && batch_size > 0 {
                // Skip invalid combinations handled by SequentialSampler
                continue;
            }

            let base_sampler = SequentialSampler::new(dataset_size);
            let batch_sampler = BatchingSampler::new(base_sampler, batch_size, drop_last);

            assert_eq!(
                batch_sampler.num_batches(),
                expected_batches,
                "Failed for dataset_size={}, batch_size={}, drop_last={}",
                dataset_size,
                batch_size,
                drop_last
            );

            let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
            assert_eq!(
                batches.len(),
                expected_batches,
                "Actual batch count doesn't match for dataset_size={}, batch_size={}, drop_last={}",
                dataset_size,
                batch_size,
                drop_last
            );
        }
    }

    #[test]
    fn test_edge_case_large_batch_size() {
        let base_sampler = SequentialSampler::new(3);
        let batch_sampler = BatchingSampler::new(base_sampler, 100, false);

        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], vec![0, 1, 2]);
    }
}
