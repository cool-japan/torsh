//! Basic sampling strategies
//!
//! This module provides fundamental sampling implementations including
//! sequential and random sampling patterns.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::random::Random;

use super::core::{Sampler, SamplerIterator};

/// Sequential sampler that yields indices in order
///
/// This sampler produces indices from 0 to dataset_size-1 in sequential order.
/// Useful for deterministic iteration over datasets.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::sampler::{SequentialSampler, Sampler};
///
/// let sampler = SequentialSampler::new(5);
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices, vec![0, 1, 2, 3, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    dataset_size: usize,
}

impl SequentialSampler {
    /// Create a new sequential sampler
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Number of samples in the dataset (can be 0 for empty datasets)
    pub fn new(dataset_size: usize) -> Self {
        Self { dataset_size }
    }

    /// Get the dataset size
    pub fn dataset_size(&self) -> usize {
        self.dataset_size
    }
}

impl Sampler for SequentialSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        SamplerIterator::from_range(0, self.dataset_size)
    }

    fn len(&self) -> usize {
        self.dataset_size
    }
}

/// Random sampler that yields indices in random order
///
/// This sampler shuffles the indices and yields them in random order.
/// Can optionally sample with or without replacement and control the
/// number of samples returned.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::sampler::{RandomSampler, Sampler};
///
/// // Sample all indices in random order
/// let sampler = RandomSampler::new(5, None, false).with_generator(42);
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 5);
///
/// // Sample 3 indices without replacement
/// let sampler = RandomSampler::new(10, Some(3), false).with_generator(42);
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct RandomSampler {
    dataset_size: usize,
    num_samples: Option<usize>,
    replacement: bool,
    generator: Option<u64>,
}

impl RandomSampler {
    /// Create a new random sampler
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Number of samples in the dataset
    /// * `num_samples` - Number of samples to yield (None for all)
    /// * `replacement` - Whether to sample with replacement
    ///
    /// # Panics
    ///
    /// Panics if `dataset_size` is 0 or if sampling without replacement
    /// but `num_samples` > `dataset_size`
    pub fn new(dataset_size: usize, num_samples: Option<usize>, replacement: bool) -> Self {
        let actual_num_samples = num_samples.unwrap_or(dataset_size);

        super::core::utils::validate_sampling_params(
            dataset_size,
            Some(actual_num_samples),
            replacement,
        )
        .expect("Invalid sampling parameters");

        Self {
            dataset_size,
            num_samples,
            replacement,
            generator: None,
        }
    }

    /// Create a simple random sampler with default settings (no replacement, all samples)
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Number of samples in the dataset
    ///
    /// # Panics
    ///
    /// Panics if `dataset_size` is 0
    pub fn simple(dataset_size: usize) -> Self {
        Self::new(dataset_size, None, false)
    }

    /// Create a random sampler with specific replacement setting
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Number of samples in the dataset
    /// * `replacement` - Whether to sample with replacement
    /// * `num_samples` - Number of samples to yield (None for all)
    ///
    /// # Panics
    ///
    /// Panics if `dataset_size` is 0
    pub fn with_replacement(
        dataset_size: usize,
        replacement: bool,
        num_samples: Option<usize>,
    ) -> Self {
        Self::new(dataset_size, num_samples, replacement)
    }

    /// Set the random number generator seed
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for reproducible sampling
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Get the dataset size
    pub fn dataset_size(&self) -> usize {
        self.dataset_size
    }

    /// Get the number of samples that will be yielded
    pub fn num_samples(&self) -> usize {
        self.num_samples.unwrap_or(self.dataset_size)
    }

    /// Check if sampling with replacement
    pub fn replacement(&self) -> bool {
        self.replacement
    }

    /// Get the generator seed if set
    pub fn generator(&self) -> Option<u64> {
        self.generator
    }
}

impl Sampler for RandomSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let num_samples = self.num_samples();

        if self.replacement {
            self.iter_with_replacement(num_samples)
        } else {
            self.iter_without_replacement(num_samples)
        }
    }

    fn len(&self) -> usize {
        self.num_samples()
    }
}

impl RandomSampler {
    /// Generate iterator for sampling with replacement
    fn iter_with_replacement(&self, num_samples: usize) -> SamplerIterator {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = match self.generator {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let indices: Vec<usize> = (0..num_samples)
            .map(|_| rng.gen_range(0..self.dataset_size))
            .collect();

        SamplerIterator::new(indices)
    }

    /// Generate iterator for sampling without replacement
    fn iter_without_replacement(&self, num_samples: usize) -> SamplerIterator {
        if num_samples == self.dataset_size {
            // Return all indices shuffled
            let indices: Vec<usize> = (0..self.dataset_size).collect();
            SamplerIterator::shuffled(indices, self.generator)
        } else {
            // Use utility function for efficient sampling
            let indices =
                super::core::utils::random_indices(self.dataset_size, num_samples, self.generator);
            SamplerIterator::new(indices)
        }
    }
}

/// Create a sequential sampler
///
/// Convenience function for creating a sequential sampler.
///
/// # Arguments
///
/// * `dataset_size` - Number of samples in the dataset
pub fn sequential(dataset_size: usize) -> SequentialSampler {
    SequentialSampler::new(dataset_size)
}

/// Create a random sampler
///
/// Convenience function for creating a random sampler that yields all
/// indices in random order without replacement.
///
/// # Arguments
///
/// * `dataset_size` - Number of samples in the dataset
/// * `seed` - Optional random seed for reproducible sampling
pub fn random(dataset_size: usize, seed: Option<u64>) -> RandomSampler {
    let mut sampler = RandomSampler::new(dataset_size, None, false);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a random sampler with replacement
///
/// Convenience function for creating a random sampler that samples
/// with replacement.
///
/// # Arguments
///
/// * `dataset_size` - Number of samples in the dataset
/// * `num_samples` - Number of samples to yield
/// * `seed` - Optional random seed for reproducible sampling
pub fn random_with_replacement(
    dataset_size: usize,
    num_samples: usize,
    seed: Option<u64>,
) -> RandomSampler {
    let mut sampler = RandomSampler::new(dataset_size, Some(num_samples), true);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a random subset sampler
///
/// Convenience function for creating a random sampler that yields
/// a subset of indices without replacement.
///
/// # Arguments
///
/// * `dataset_size` - Number of samples in the dataset
/// * `num_samples` - Number of samples to yield
/// * `seed` - Optional random seed for reproducible sampling
pub fn random_subset(dataset_size: usize, num_samples: usize, seed: Option<u64>) -> RandomSampler {
    let mut sampler = RandomSampler::new(dataset_size, Some(num_samples), false);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let sampler = SequentialSampler::new(5);
        assert_eq!(sampler.len(), 5);
        assert_eq!(sampler.dataset_size(), 5);
        assert!(!sampler.is_empty());

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sequential_sampler_zero_size() {
        // Zero-size datasets are now allowed for empty datasets
        let sampler = SequentialSampler::new(0);
        assert_eq!(sampler.dataset_size(), 0);

        // Empty sampler should produce no indices
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_random_sampler_all_indices() {
        let sampler = RandomSampler::new(5, None, false).with_generator(42);
        assert_eq!(sampler.len(), 5);
        assert_eq!(sampler.dataset_size(), 5);
        assert_eq!(sampler.num_samples(), 5);
        assert!(!sampler.replacement());
        assert_eq!(sampler.generator(), Some(42));

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 5);

        // All indices 0-4 should be present
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        assert_eq!(sorted_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_sampler_subset() {
        let sampler = RandomSampler::new(10, Some(3), false).with_generator(42);
        assert_eq!(sampler.len(), 3);
        assert_eq!(sampler.num_samples(), 3);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 3);

        // All indices should be unique and in range
        let mut unique_indices = indices.clone();
        unique_indices.sort();
        unique_indices.dedup();
        assert_eq!(unique_indices.len(), 3);

        for &idx in &indices {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_random_sampler_with_replacement() {
        let sampler = RandomSampler::new(3, Some(10), true).with_generator(42);
        assert_eq!(sampler.len(), 10);
        assert_eq!(sampler.num_samples(), 10);
        assert!(sampler.replacement());

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 10);

        // All indices should be in range (but may be duplicated)
        for &idx in &indices {
            assert!(idx < 3);
        }
    }

    #[test]
    #[should_panic(expected = "Invalid sampling parameters")]
    fn test_random_sampler_invalid_no_replacement() {
        RandomSampler::new(5, Some(10), false);
    }

    #[test]
    fn test_random_sampler_reproducible() {
        let sampler1 = RandomSampler::new(10, Some(5), false).with_generator(42);
        let sampler2 = RandomSampler::new(10, Some(5), false).with_generator(42);

        let indices1: Vec<usize> = sampler1.iter().collect();
        let indices2: Vec<usize> = sampler2.iter().collect();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_convenience_functions() {
        let seq = sequential(5);
        assert_eq!(seq.len(), 5);

        let rand = random(5, Some(42));
        assert_eq!(rand.len(), 5);
        assert!(!rand.replacement());

        let rand_repl = random_with_replacement(3, 10, Some(42));
        assert_eq!(rand_repl.len(), 10);
        assert!(rand_repl.replacement());

        let subset = random_subset(10, 3, Some(42));
        assert_eq!(subset.len(), 3);
        assert!(!subset.replacement());
    }

    #[test]
    fn test_random_sampler_clone() {
        let sampler = RandomSampler::new(5, Some(3), false).with_generator(42);
        let cloned = sampler.clone();

        assert_eq!(sampler.len(), cloned.len());
        assert_eq!(sampler.dataset_size(), cloned.dataset_size());
        assert_eq!(sampler.replacement(), cloned.replacement());
        assert_eq!(sampler.generator(), cloned.generator());
    }

    #[test]
    fn test_edge_cases() {
        // Single element dataset
        let seq = SequentialSampler::new(1);
        let indices: Vec<usize> = seq.iter().collect();
        assert_eq!(indices, vec![0]);

        let rand = RandomSampler::new(1, None, false);
        let indices: Vec<usize> = rand.iter().collect();
        assert_eq!(indices, vec![0]);

        // Sample 0 items (should be allowed for with replacement)
        let rand_zero = RandomSampler::new(5, Some(0), true);
        assert_eq!(rand_zero.len(), 0);
        assert!(rand_zero.is_empty());

        let indices: Vec<usize> = rand_zero.iter().collect();
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_iterator_properties() {
        let sampler = RandomSampler::new(5, Some(3), false).with_generator(42);
        let mut iter = sampler.iter();

        // Test size hint
        assert_eq!(iter.size_hint(), (3, Some(3)));

        // Test exact size
        assert_eq!(iter.len(), 3);

        // Consume one item
        iter.next();
        assert_eq!(iter.len(), 2);
    }
}
