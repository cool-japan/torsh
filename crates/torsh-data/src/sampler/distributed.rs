//! Distributed sampling for multi-process training
//!
//! This module provides samplers for distributed training scenarios where
//! multiple processes need to sample different subsets of the data to avoid
//! overlap and ensure balanced workloads.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::random::{Random, Rng};

use super::core::{Sampler, SamplerIterator};

/// Wrapper that makes any sampler work in a distributed setting
///
/// This wrapper takes an underlying sampler and distributes its indices across
/// multiple replicas (processes). Each replica gets a disjoint subset of the
/// indices, ensuring no overlap between processes.
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::{SequentialSampler, DistributedWrapper, Sampler};
///
/// let base_sampler = SequentialSampler::new(10);
/// // Process 0 of 2 processes
/// let distributed = DistributedWrapper::new(base_sampler, 2, 0);
///
/// let indices: Vec<usize> = distributed.iter().collect();
/// // Process 0 gets: [0, 2, 4, 6, 8]
/// // Process 1 would get: [1, 3, 5, 7, 9]
/// ```
#[derive(Debug, Clone)]
pub struct DistributedWrapper<S: Sampler> {
    sampler: S,
    num_replicas: usize,
    rank: usize,
    shuffle: bool,
    generator: Option<u64>,
}

impl<S: Sampler> DistributedWrapper<S> {
    /// Create a new distributed wrapper
    ///
    /// # Arguments
    ///
    /// * `sampler` - The underlying sampler to distribute
    /// * `num_replicas` - Total number of processes
    /// * `rank` - Current process rank (0-based)
    ///
    /// # Panics
    ///
    /// Panics if `num_replicas` is 0 or `rank` >= `num_replicas`
    pub fn new(sampler: S, num_replicas: usize, rank: usize) -> Self {
        assert!(num_replicas > 0, "Number of replicas must be positive");
        assert!(rank < num_replicas, "Rank must be less than num_replicas");

        Self {
            sampler,
            num_replicas,
            rank,
            shuffle: true,
            generator: None,
        }
    }

    /// Enable or disable shuffling
    ///
    /// When shuffling is enabled, the indices are shuffled before distribution.
    /// This ensures different ordering across epochs.
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the random number generator seed
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for reproducible shuffling
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Get the number of replicas
    pub fn num_replicas(&self) -> usize {
        self.num_replicas
    }

    /// Get the current rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Check if shuffling is enabled
    pub fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Get the generator seed if set
    pub fn generator(&self) -> Option<u64> {
        self.generator
    }

    /// Get a reference to the underlying sampler
    pub fn sampler(&self) -> &S {
        &self.sampler
    }

    /// Get the underlying sampler by value
    pub fn into_sampler(self) -> S {
        self.sampler
    }

    /// Calculate the number of samples this replica will receive
    fn calculate_num_samples(&self) -> usize {
        let total_samples = self.sampler.len();
        // Each replica gets roughly equal number of samples
        // If total doesn't divide evenly, some replicas get one extra sample
        let base_samples = total_samples / self.num_replicas;
        let extra_samples = total_samples % self.num_replicas;

        if self.rank < extra_samples {
            base_samples + 1
        } else {
            base_samples
        }
    }
}

impl<S: Sampler> Sampler for DistributedWrapper<S> {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        // Get all indices from the underlying sampler
        let mut all_indices: Vec<usize> = self.sampler.iter().collect();

        // Shuffle if enabled
        if self.shuffle {
            // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
            let mut rng = match self.generator {
                Some(seed) => Random::seed(seed),
                None => Random::seed(42),
            };

            // Fisher-Yates shuffle
            for i in (1..all_indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                all_indices.swap(i, j);
            }
        }

        // Distribute indices across replicas
        let replica_indices: Vec<usize> = all_indices
            .into_iter()
            .enumerate()
            .filter_map(|(i, idx)| {
                if i % self.num_replicas == self.rank {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        SamplerIterator::new(replica_indices)
    }

    fn len(&self) -> usize {
        self.calculate_num_samples()
    }
}

/// Distributed sampler for balanced data distribution
///
/// Unlike DistributedWrapper which wraps an existing sampler, DistributedSampler
/// is a standalone sampler designed specifically for distributed training.
/// It provides more control over the distribution strategy.
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::{DistributedSampler, Sampler};
///
/// // Process 1 of 4 processes, working with dataset of size 100
/// let sampler = DistributedSampler::new(100, 4, 1, true).with_generator(42);
///
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 25); // 100 / 4 = 25 samples per process
/// ```
#[derive(Debug, Clone)]
pub struct DistributedSampler {
    dataset_size: usize,
    num_replicas: usize,
    rank: usize,
    shuffle: bool,
    generator: Option<u64>,
    drop_last: bool,
}

impl DistributedSampler {
    /// Create a new distributed sampler
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Total size of the dataset
    /// * `num_replicas` - Total number of processes
    /// * `rank` - Current process rank (0-based)
    /// * `shuffle` - Whether to shuffle indices
    ///
    /// # Panics
    ///
    /// Panics if `dataset_size` is 0, `num_replicas` is 0, or `rank` >= `num_replicas`
    pub fn new(dataset_size: usize, num_replicas: usize, rank: usize, shuffle: bool) -> Self {
        assert!(dataset_size > 0, "Dataset size must be positive");
        assert!(num_replicas > 0, "Number of replicas must be positive");
        assert!(rank < num_replicas, "Rank must be less than num_replicas");

        Self {
            dataset_size,
            num_replicas,
            rank,
            shuffle,
            generator: None,
            drop_last: false,
        }
    }

    /// Set the random number generator seed
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Set whether to drop the last samples to make dataset evenly divisible
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Get the dataset size
    pub fn dataset_size(&self) -> usize {
        self.dataset_size
    }

    /// Get the number of replicas
    pub fn num_replicas(&self) -> usize {
        self.num_replicas
    }

    /// Get the current rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Check if shuffling is enabled
    pub fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Check if dropping last samples
    pub fn drop_last(&self) -> bool {
        self.drop_last
    }

    /// Get the generator seed if set
    pub fn generator(&self) -> Option<u64> {
        self.generator
    }

    /// Calculate the effective dataset size after potential padding
    fn effective_dataset_size(&self) -> usize {
        if self.drop_last {
            // Drop samples to make evenly divisible
            (self.dataset_size / self.num_replicas) * self.num_replicas
        } else {
            // Pad with duplicates to make evenly divisible
            let samples_per_replica =
                (self.dataset_size + self.num_replicas - 1) / self.num_replicas;
            samples_per_replica * self.num_replicas
        }
    }

    /// Calculate the number of samples this replica will receive
    fn calculate_num_samples(&self) -> usize {
        if self.drop_last {
            self.dataset_size / self.num_replicas
        } else {
            (self.dataset_size + self.num_replicas - 1) / self.num_replicas
        }
    }
}

impl Sampler for DistributedSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let effective_size = self.effective_dataset_size();
        let samples_per_replica = self.calculate_num_samples();

        // Create base indices
        let mut indices: Vec<usize> = if self.drop_last {
            (0..effective_size).collect()
        } else {
            // Pad with duplicates if needed
            (0..effective_size).map(|i| i % self.dataset_size).collect()
        };

        // Shuffle if enabled
        if self.shuffle {
            // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
            let mut rng = match self.generator {
                Some(seed) => Random::seed(seed),
                None => Random::seed(42),
            };

            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        // Extract this replica's portion
        let start_idx = self.rank * samples_per_replica;
        let end_idx = start_idx + samples_per_replica;
        let replica_indices = indices[start_idx..end_idx.min(indices.len())].to_vec();

        SamplerIterator::new(replica_indices)
    }

    fn len(&self) -> usize {
        self.calculate_num_samples()
    }
}

/// Create a distributed wrapper for any sampler
///
/// Convenience function for creating a distributed wrapper.
///
/// # Arguments
///
/// * `sampler` - The underlying sampler
/// * `num_replicas` - Total number of processes
/// * `rank` - Current process rank
pub fn distributed<S: Sampler>(
    sampler: S,
    num_replicas: usize,
    rank: usize,
) -> DistributedWrapper<S> {
    DistributedWrapper::new(sampler, num_replicas, rank)
}

/// Create a distributed sampler
///
/// Convenience function for creating a distributed sampler.
///
/// # Arguments
///
/// * `dataset_size` - Total size of the dataset
/// * `num_replicas` - Total number of processes
/// * `rank` - Current process rank
/// * `shuffle` - Whether to shuffle indices
pub fn distributed_sampler(
    dataset_size: usize,
    num_replicas: usize,
    rank: usize,
    shuffle: bool,
) -> DistributedSampler {
    DistributedSampler::new(dataset_size, num_replicas, rank, shuffle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::basic::SequentialSampler;

    #[test]
    fn test_distributed_wrapper_basic() {
        let base_sampler = SequentialSampler::new(10);
        let distributed = DistributedWrapper::new(base_sampler, 2, 0).with_shuffle(false);

        assert_eq!(distributed.num_replicas(), 2);
        assert_eq!(distributed.rank(), 0);
        assert!(!distributed.shuffle());
        assert_eq!(distributed.len(), 5); // 10 / 2 = 5 samples per process

        let indices: Vec<usize> = distributed.iter().collect();
        assert_eq!(indices, vec![0, 2, 4, 6, 8]); // Even indices for rank 0
    }

    #[test]
    fn test_distributed_wrapper_rank_1() {
        let base_sampler = SequentialSampler::new(10);
        let distributed = DistributedWrapper::new(base_sampler, 2, 1).with_shuffle(false);

        assert_eq!(distributed.rank(), 1);
        assert_eq!(distributed.len(), 5);

        let indices: Vec<usize> = distributed.iter().collect();
        assert_eq!(indices, vec![1, 3, 5, 7, 9]); // Odd indices for rank 1
    }

    #[test]
    fn test_distributed_wrapper_uneven_split() {
        let base_sampler = SequentialSampler::new(7); // 7 doesn't divide evenly by 3

        let dist0 = DistributedWrapper::new(base_sampler.clone(), 3, 0).with_shuffle(false);
        let dist1 = DistributedWrapper::new(base_sampler.clone(), 3, 1).with_shuffle(false);
        let dist2 = DistributedWrapper::new(base_sampler, 3, 2).with_shuffle(false);

        // First rank gets extra sample: 7 / 3 = 2 remainder 1
        assert_eq!(dist0.len(), 3); // 2 + 1 = 3
        assert_eq!(dist1.len(), 2); // 2
        assert_eq!(dist2.len(), 2); // 2

        let indices0: Vec<usize> = dist0.iter().collect();
        let indices1: Vec<usize> = dist1.iter().collect();
        let indices2: Vec<usize> = dist2.iter().collect();

        assert_eq!(indices0, vec![0, 3, 6]);
        assert_eq!(indices1, vec![1, 4]);
        assert_eq!(indices2, vec![2, 5]);

        // Verify all indices are covered
        let mut all_indices = indices0;
        all_indices.extend(indices1);
        all_indices.extend(indices2);
        all_indices.sort();
        assert_eq!(all_indices, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_distributed_wrapper_with_shuffle() {
        let base_sampler = SequentialSampler::new(10);
        let distributed = DistributedWrapper::new(base_sampler, 2, 0)
            .with_shuffle(true)
            .with_generator(42);

        let indices: Vec<usize> = distributed.iter().collect();
        assert_eq!(indices.len(), 5);

        // Verify indices are from original dataset
        for &idx in &indices {
            assert!(idx < 10);
        }

        // Should be deterministic with same seed
        let distributed2 = DistributedWrapper::new(SequentialSampler::new(10), 2, 0)
            .with_shuffle(true)
            .with_generator(42);
        let indices2: Vec<usize> = distributed2.iter().collect();
        assert_eq!(indices, indices2);
    }

    #[test]
    #[should_panic(expected = "Number of replicas must be positive")]
    fn test_distributed_wrapper_zero_replicas() {
        let base_sampler = SequentialSampler::new(10);
        DistributedWrapper::new(base_sampler, 0, 0);
    }

    #[test]
    #[should_panic(expected = "Rank must be less than num_replicas")]
    fn test_distributed_wrapper_invalid_rank() {
        let base_sampler = SequentialSampler::new(10);
        DistributedWrapper::new(base_sampler, 2, 2);
    }

    #[test]
    fn test_distributed_sampler_basic() {
        let sampler = DistributedSampler::new(12, 3, 1, false);

        assert_eq!(sampler.dataset_size(), 12);
        assert_eq!(sampler.num_replicas(), 3);
        assert_eq!(sampler.rank(), 1);
        assert!(!sampler.shuffle());
        assert!(!sampler.drop_last());
        assert_eq!(sampler.len(), 4); // 12 / 3 = 4

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices, vec![4, 5, 6, 7]); // Rank 1 gets indices 4-7
    }

    #[test]
    fn test_distributed_sampler_with_padding() {
        let sampler = DistributedSampler::new(10, 3, 0, false); // 10 doesn't divide by 3

        // With padding, each replica gets 4 samples (total 12, padded from 10)
        assert_eq!(sampler.len(), 4);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 4);

        // All indices should be valid (0-9, with possible duplicates due to padding)
        for &idx in &indices {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_distributed_sampler_drop_last() {
        let sampler = DistributedSampler::new(10, 3, 0, false).with_drop_last(true);

        // With drop_last, 10 / 3 = 3 samples per replica (drops 1 sample)
        assert_eq!(sampler.len(), 3);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_distributed_sampler_shuffle() {
        let sampler = DistributedSampler::new(12, 3, 0, true).with_generator(42);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 4);

        // Should be deterministic with same seed
        let sampler2 = DistributedSampler::new(12, 3, 0, true).with_generator(42);
        let indices2: Vec<usize> = sampler2.iter().collect();
        assert_eq!(indices, indices2);
    }

    #[test]
    fn test_convenience_functions() {
        let base_sampler = SequentialSampler::new(8);
        let dist_wrapper = distributed(base_sampler, 2, 0);
        assert_eq!(dist_wrapper.len(), 4);

        let dist_sampler = distributed_sampler(8, 2, 1, false);
        assert_eq!(dist_sampler.len(), 4);
    }

    #[test]
    fn test_distributed_sampler_edge_cases() {
        // Single replica (should get all data)
        let sampler = DistributedSampler::new(10, 1, 0, false);
        assert_eq!(sampler.len(), 10);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices, (0..10).collect::<Vec<_>>());

        // More replicas than data points
        let sampler = DistributedSampler::new(2, 5, 3, false);
        assert_eq!(sampler.len(), 1);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 1);
        assert!(indices[0] < 2);
    }

    #[test]
    fn test_into_sampler() {
        let base_sampler = SequentialSampler::new(5);
        let distributed = DistributedWrapper::new(base_sampler, 2, 0);

        let recovered = distributed.into_sampler();
        assert_eq!(recovered.len(), 5);
    }
}
