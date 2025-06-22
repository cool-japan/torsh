//! Sampling strategies for datasets

// use torsh_core::error::Result;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// Trait for sampling from a dataset
pub trait Sampler: Send {
    /// Iterator type returned by the sampler
    type Iter: Iterator<Item = usize> + Send;

    /// Create an iterator over indices
    fn iter(&self) -> Self::Iter;

    /// Total number of samples that will be yielded
    fn len(&self) -> usize;

    /// Check if sampler is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Sequential sampler that yields indices in order
#[derive(Clone, Debug)]
pub struct SequentialSampler {
    num_samples: usize,
}

impl SequentialSampler {
    /// Create a new sequential sampler
    pub fn new(num_samples: usize) -> Self {
        Self { num_samples }
    }
}

impl Sampler for SequentialSampler {
    type Iter = std::ops::Range<usize>;

    fn iter(&self) -> Self::Iter {
        0..self.num_samples
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

/// Random sampler that yields indices in random order
pub struct RandomSampler {
    num_samples: usize,
    replacement: bool,
    generator: Option<u64>,
}

impl RandomSampler {
    /// Create a new random sampler
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            replacement: false,
            generator: None,
        }
    }

    /// Set whether to sample with replacement
    pub fn with_replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }

    /// Set random generator seed
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }
}

/// Iterator for RandomSampler
pub struct RandomSamplerIter {
    indices: Vec<usize>,
    current: usize,
}

impl Iterator for RandomSamplerIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

impl Sampler for RandomSampler {
    type Iter = RandomSamplerIter;

    fn iter(&self) -> Self::Iter {
        use rand::rngs::StdRng;
        use rand::{seq::SliceRandom, Rng, SeedableRng};

        let indices: Vec<usize> = if self.replacement {
            // Sample with replacement
            let mut rng = if let Some(seed) = self.generator {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_entropy()
            };

            (0..self.num_samples)
                .map(|_| rng.gen_range(0..self.num_samples))
                .collect()
        } else {
            // Sample without replacement (shuffle)
            let mut indices: Vec<usize> = (0..self.num_samples).collect();
            let mut rng = if let Some(seed) = self.generator {
                StdRng::seed_from_u64(seed)
            } else {
                StdRng::from_entropy()
            };
            indices.shuffle(&mut rng);
            indices
        };

        RandomSamplerIter {
            indices,
            current: 0,
        }
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

/// Batch sampler that yields batches of indices
pub struct BatchSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    /// Create a new batch sampler
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        assert!(batch_size > 0, "batch_size must be positive");
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }
}

/// Iterator for BatchSampler
pub struct BatchSamplerIter<I: Iterator<Item = usize>> {
    iter: I,
    batch_size: usize,
    drop_last: bool,
}

impl<I: Iterator<Item = usize>> Iterator for BatchSamplerIter<I> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            match self.iter.next() {
                Some(idx) => batch.push(idx),
                None => break,
            }
        }

        if batch.is_empty() || (batch.len() < self.batch_size && self.drop_last) {
            None
        } else {
            Some(batch)
        }
    }
}

/// Trait for batch samplers that yield batches of indices
pub trait BatchSamplerTrait: Send {
    /// Iterator type returned by the batch sampler
    type Iter: Iterator<Item = Vec<usize>> + Send;

    /// Create an iterator over batches of indices
    fn iter(&self) -> Self::Iter;

    /// Total number of batches that will be yielded
    fn len(&self) -> usize;

    /// Check if batch sampler is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S: Sampler> BatchSamplerTrait for BatchSampler<S> {
    type Iter = BatchSamplerIter<S::Iter>;

    fn iter(&self) -> Self::Iter {
        BatchSamplerIter {
            iter: self.sampler.iter(),
            batch_size: self.batch_size,
            drop_last: self.drop_last,
        }
    }

    fn len(&self) -> usize {
        if self.drop_last {
            self.sampler.len() / self.batch_size
        } else {
            self.sampler.len().div_ceil(self.batch_size)
        }
    }
}

/// Weighted random sampler for imbalanced datasets
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    num_samples: usize,
    #[allow(dead_code)]
    replacement: bool,
    generator: Option<u64>,
}

impl WeightedRandomSampler {
    /// Create a new weighted random sampler
    pub fn new(weights: Vec<f64>, num_samples: usize, replacement: bool) -> Self {
        assert!(!weights.is_empty(), "weights cannot be empty");
        assert!(
            weights.iter().all(|&w| w >= 0.0),
            "weights must be non-negative"
        );

        Self {
            weights,
            num_samples,
            replacement,
            generator: None,
        }
    }

    /// Set random generator seed
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }
}

/// Iterator for WeightedRandomSampler
pub struct WeightedRandomSamplerIter {
    indices: Vec<usize>,
    current: usize,
}

impl Iterator for WeightedRandomSamplerIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.indices.len() {
            let idx = self.indices[self.current];
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

impl Sampler for WeightedRandomSampler {
    type Iter = WeightedRandomSamplerIter;

    fn iter(&self) -> Self::Iter {
        use rand::rngs::StdRng;
        use rand::{
            distributions::{Distribution, WeightedIndex},
            SeedableRng,
        };

        let mut rng = if let Some(seed) = self.generator {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        let dist = WeightedIndex::new(&self.weights).unwrap();
        let indices: Vec<usize> = (0..self.num_samples)
            .map(|_| dist.sample(&mut rng))
            .collect();

        WeightedRandomSamplerIter {
            indices,
            current: 0,
        }
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let sampler = SequentialSampler::new(5);
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(sampler.len(), 5);
    }

    #[test]
    fn test_random_sampler() {
        let sampler = RandomSampler::new(5).with_generator(42);
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 5);

        // Check all indices are unique (no replacement)
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_batch_sampler() {
        let base_sampler = SequentialSampler::new(10);

        // Test without dropping last
        let batch_sampler = BatchSampler::new(base_sampler.clone(), 3, false);
        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[3], vec![9]);

        // Test with dropping last
        let batch_sampler = BatchSampler::new(base_sampler, 3, true);
        let batches: Vec<Vec<usize>> = batch_sampler.iter().collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[2], vec![6, 7, 8]);
    }

    #[test]
    fn test_weighted_sampler() {
        let weights = vec![0.1, 0.1, 0.1, 0.1, 0.6]; // Last element has higher weight
        let sampler = WeightedRandomSampler::new(weights, 100, true).with_generator(42);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 100);

        // Count occurrences
        let mut counts = vec![0; 5];
        for &idx in &indices {
            counts[idx] += 1;
        }

        // Last element should appear more frequently
        assert!(counts[4] > counts[0]);
    }
}
