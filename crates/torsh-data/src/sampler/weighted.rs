//! Weighted and subset sampling functionality
//!
//! This module provides sampling strategies that work with weighted datasets
//! and subset selections, useful for imbalanced datasets and custom data selection.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::random::{Random, Rng};

use super::core::{rng_utils, Sampler, SamplerIterator};

/// Weighted random sampler for imbalanced datasets
///
/// This sampler allows sampling indices according to specified weights,
/// which is particularly useful for handling imbalanced datasets or
/// implementing custom sampling strategies.
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::{WeightedRandomSampler, Sampler};
///
/// // Higher weight for the last class (imbalanced dataset)
/// let weights = vec![0.1, 0.1, 0.1, 0.1, 0.6];
/// let sampler = WeightedRandomSampler::new(weights, 100, true).with_generator(42);
///
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 100);
/// ```
#[derive(Debug, Clone)]
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    num_samples: usize,
    replacement: bool,
    generator: Option<u64>,
}

impl WeightedRandomSampler {
    /// Create a new weighted random sampler
    ///
    /// # Arguments
    ///
    /// * `weights` - Sampling weights for each index
    /// * `num_samples` - Number of samples to generate
    /// * `replacement` - Whether to sample with replacement
    ///
    /// # Panics
    ///
    /// Panics if weights are empty, contain negative values, or don't sum to a positive finite value
    pub fn new(weights: Vec<f64>, num_samples: usize, replacement: bool) -> Self {
        assert!(!weights.is_empty(), "weights cannot be empty");
        assert!(
            weights.iter().all(|&w| w >= 0.0),
            "weights must be non-negative"
        );
        let weight_sum: f64 = weights.iter().sum();
        assert!(
            weight_sum > 0.0 && weight_sum.is_finite(),
            "weights must sum to a positive finite value, got {weight_sum}"
        );

        Self {
            weights,
            num_samples,
            replacement,
            generator: None,
        }
    }

    /// Set random generator seed
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for reproducible sampling
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Get the weights
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Check if sampling with replacement
    pub fn replacement(&self) -> bool {
        self.replacement
    }

    /// Get the generator seed if set
    pub fn generator(&self) -> Option<u64> {
        self.generator
    }

    /// Sample indices according to weights with replacement
    fn sample_with_replacement(&self) -> Vec<usize> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = rng_utils::create_rng(self.generator);

        // Normalize weights to create cumulative distribution
        let weight_sum: f64 = self.weights.iter().sum();
        let mut cumulative_weights = Vec::with_capacity(self.weights.len());
        let mut cumsum = 0.0;

        for &weight in &self.weights {
            cumsum += weight / weight_sum;
            cumulative_weights.push(cumsum);
        }

        // Ensure the last value is exactly 1.0 to handle floating point precision
        if let Some(last) = cumulative_weights.last_mut() {
            *last = 1.0;
        }

        // Sample using inverse transform sampling
        (0..self.num_samples)
            .map(|_| {
                let rand_val: f64 = rng.gen();
                // Binary search for the first cumulative weight >= rand_val
                cumulative_weights
                    .binary_search_by(|&x| {
                        x.partial_cmp(&rand_val)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or_else(|i| i)
                    .min(self.weights.len() - 1)
            })
            .collect()
    }

    /// Sample indices according to weights without replacement
    fn sample_without_replacement(&self) -> Vec<usize> {
        if self.num_samples >= self.weights.len() {
            // Return all indices if we need more samples than available
            return (0..self.weights.len()).collect();
        }

        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = rng_utils::create_rng(self.generator);

        // Use weighted reservoir sampling for sampling without replacement
        let mut selected_indices = Vec::new();
        let mut remaining_weights = self.weights.clone();
        let mut remaining_indices: Vec<usize> = (0..self.weights.len()).collect();

        for _ in 0..self.num_samples {
            if remaining_indices.is_empty() {
                break;
            }

            // Normalize remaining weights
            let weight_sum: f64 = remaining_weights.iter().sum();
            if weight_sum <= 0.0 {
                break;
            }

            let mut cumsum = 0.0;
            let rand_val: f64 = rng.gen::<f64>() * weight_sum;

            let mut selected_idx = 0;
            for (i, &weight) in remaining_weights.iter().enumerate() {
                cumsum += weight;
                if cumsum >= rand_val {
                    selected_idx = i;
                    break;
                }
            }

            // Add the selected index to results
            selected_indices.push(remaining_indices[selected_idx]);

            // Remove the selected index and weight
            remaining_indices.remove(selected_idx);
            remaining_weights.remove(selected_idx);
        }

        selected_indices
    }
}

impl Sampler for WeightedRandomSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let indices = if self.replacement {
            self.sample_with_replacement()
        } else {
            self.sample_without_replacement()
        };

        SamplerIterator::new(indices)
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

/// Random sampler that samples from a subset of indices
///
/// This sampler takes a predefined subset of indices and samples from them
/// in random order. Useful for creating custom data splits or working with
/// filtered datasets.
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::{SubsetRandomSampler, Sampler};
///
/// // Sample from odd indices only
/// let subset_indices = vec![1, 3, 5, 7, 9];
/// let sampler = SubsetRandomSampler::new(subset_indices).with_generator(42);
///
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct SubsetRandomSampler {
    indices: Vec<usize>,
    generator: Option<u64>,
}

impl SubsetRandomSampler {
    /// Create a new subset random sampler
    ///
    /// # Arguments
    ///
    /// * `indices` - The subset of indices to sample from
    pub fn new(indices: Vec<usize>) -> Self {
        Self {
            indices,
            generator: None,
        }
    }

    /// Set random generator seed
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for reproducible sampling
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Get the subset indices
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get the generator seed if set
    pub fn generator(&self) -> Option<u64> {
        self.generator
    }
}

impl Sampler for SubsetRandomSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let mut shuffled_indices = self.indices.clone();
        rng_utils::shuffle_indices(&mut shuffled_indices, self.generator);
        SamplerIterator::new(shuffled_indices)
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

/// Create a weighted random sampler
///
/// Convenience function for creating a weighted random sampler.
///
/// # Arguments
///
/// * `weights` - Sampling weights for each index
/// * `num_samples` - Number of samples to generate
/// * `replacement` - Whether to sample with replacement
/// * `seed` - Optional random seed for reproducible sampling
pub fn weighted_random(
    weights: Vec<f64>,
    num_samples: usize,
    replacement: bool,
    seed: Option<u64>,
) -> WeightedRandomSampler {
    let mut sampler = WeightedRandomSampler::new(weights, num_samples, replacement);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a subset random sampler
///
/// Convenience function for creating a subset random sampler.
///
/// # Arguments
///
/// * `indices` - The subset of indices to sample from
/// * `seed` - Optional random seed for reproducible sampling
pub fn subset_random(indices: Vec<usize>, seed: Option<u64>) -> SubsetRandomSampler {
    let mut sampler = SubsetRandomSampler::new(indices);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a balanced weighted sampler for class imbalance
///
/// Creates weights that are inversely proportional to class frequencies,
/// providing balanced sampling for imbalanced datasets.
///
/// # Arguments
///
/// * `class_counts` - Number of samples per class
/// * `num_samples` - Total number of samples to generate
/// * `seed` - Optional random seed for reproducible sampling
pub fn balanced_weighted(
    class_counts: &[usize],
    num_samples: usize,
    seed: Option<u64>,
) -> WeightedRandomSampler {
    let total_samples: usize = class_counts.iter().sum();
    let num_classes = class_counts.len();

    // Calculate inverse frequency weights
    let weights: Vec<f64> = class_counts
        .iter()
        .map(|&count| {
            if count > 0 {
                total_samples as f64 / (num_classes as f64 * count as f64)
            } else {
                0.0
            }
        })
        .collect();

    weighted_random(weights, num_samples, true, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_sampler_basic() {
        let weights = vec![0.1, 0.1, 0.1, 0.1, 0.6]; // Last element has higher weight
        let sampler = WeightedRandomSampler::new(weights.clone(), 100, true).with_generator(42);

        assert_eq!(sampler.len(), 100);
        assert_eq!(sampler.weights(), &weights);
        assert_eq!(sampler.num_samples(), 100);
        assert!(sampler.replacement());
        assert_eq!(sampler.generator(), Some(42));

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 100);

        // All indices should be in valid range
        for &idx in &indices {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_weighted_sampler_without_replacement() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sampler = WeightedRandomSampler::new(weights, 3, false).with_generator(42);

        assert!(!sampler.replacement());

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 3);

        // All indices should be unique
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        sorted_indices.dedup();
        assert_eq!(sorted_indices.len(), 3);

        // All indices should be in valid range
        for &idx in &indices {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_weighted_sampler_uniform_weights() {
        let weights = vec![1.0; 10];
        let sampler = WeightedRandomSampler::new(weights, 50, true).with_generator(42);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 50);

        // With uniform weights, should get reasonably balanced distribution
        let mut counts = [0; 10];
        for &idx in &indices {
            counts[idx] += 1;
        }

        // Each index should appear at least once in 50 samples
        for count in counts {
            assert!(count > 0);
        }
    }

    #[test]
    fn test_weighted_sampler_extreme_weights() {
        let weights = vec![0.0, 0.0, 0.0, 1.0]; // Only last index has weight
        let sampler = WeightedRandomSampler::new(weights, 10, true).with_generator(42);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 10);

        // All samples should be from index 3
        for &idx in &indices {
            assert_eq!(idx, 3);
        }
    }

    #[test]
    #[should_panic(expected = "weights cannot be empty")]
    fn test_weighted_sampler_empty_weights() {
        WeightedRandomSampler::new(vec![], 10, true);
    }

    #[test]
    #[should_panic(expected = "weights must be non-negative")]
    fn test_weighted_sampler_negative_weights() {
        WeightedRandomSampler::new(vec![1.0, -1.0, 1.0], 10, true);
    }

    #[test]
    #[should_panic(expected = "weights must sum to a positive finite value")]
    fn test_weighted_sampler_zero_sum() {
        WeightedRandomSampler::new(vec![0.0, 0.0, 0.0], 10, true);
    }

    #[test]
    fn test_subset_random_sampler() {
        // Test with a subset of indices
        let subset_indices = vec![1, 3, 5, 7, 9];
        let sampler = SubsetRandomSampler::new(subset_indices.clone()).with_generator(42);

        assert_eq!(sampler.len(), 5);
        assert_eq!(sampler.indices(), &subset_indices);
        assert_eq!(sampler.generator(), Some(42));

        let sampled_indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(sampled_indices.len(), 5);

        // Check that all sampled indices are from the original subset
        for idx in &sampled_indices {
            assert!(subset_indices.contains(idx));
        }

        // Check that we have all indices (just shuffled)
        let mut sorted_sampled = sampled_indices.clone();
        sorted_sampled.sort();
        let mut sorted_original = subset_indices;
        sorted_original.sort();
        assert_eq!(sorted_sampled, sorted_original);
    }

    #[test]
    fn test_subset_random_sampler_empty() {
        let sampler = SubsetRandomSampler::new(vec![]);
        assert_eq!(sampler.len(), 0);
        assert!(sampler.is_empty());

        let indices: Vec<usize> = sampler.iter().collect();
        assert!(indices.is_empty());
    }

    #[test]
    fn test_subset_random_sampler_single() {
        let sampler = SubsetRandomSampler::new(vec![42]);
        assert_eq!(sampler.len(), 1);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices, vec![42]);
    }

    #[test]
    fn test_subset_random_sampler_reproducible() {
        let subset_indices = vec![10, 20, 30, 40, 50];
        let sampler1 = SubsetRandomSampler::new(subset_indices.clone()).with_generator(123);
        let sampler2 = SubsetRandomSampler::new(subset_indices).with_generator(123);

        let indices1: Vec<usize> = sampler1.iter().collect();
        let indices2: Vec<usize> = sampler2.iter().collect();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_convenience_functions() {
        // Test weighted_random convenience function
        let weights = vec![1.0, 2.0, 3.0];
        let weighted = weighted_random(weights.clone(), 10, true, Some(42));
        assert_eq!(weighted.weights(), &weights);
        assert_eq!(weighted.num_samples(), 10);
        assert!(weighted.replacement());
        assert_eq!(weighted.generator(), Some(42));

        // Test subset_random convenience function
        let indices = vec![1, 3, 5];
        let subset = subset_random(indices.clone(), Some(42));
        assert_eq!(subset.indices(), &indices);
        assert_eq!(subset.generator(), Some(42));

        // Test balanced_weighted convenience function
        let class_counts = vec![100, 50, 25]; // Imbalanced classes
        let balanced = balanced_weighted(&class_counts, 30, Some(42));
        assert_eq!(balanced.num_samples(), 30);
        assert!(balanced.replacement());

        // Verify that balanced weights are inversely proportional to class counts
        let weights = balanced.weights();
        assert!(weights[2] > weights[1]); // Smallest class has highest weight
        assert!(weights[1] > weights[0]); // Medium class has medium weight
    }

    #[test]
    fn test_balanced_weighted_edge_cases() {
        // Test with zero counts
        let class_counts = vec![100, 0, 50];
        let balanced = balanced_weighted(&class_counts, 20, Some(42));
        let weights = balanced.weights();

        assert!(weights[0] > 0.0);
        assert_eq!(weights[1], 0.0); // Zero count should give zero weight
        assert!(weights[2] > 0.0);
        assert!(weights[2] > weights[0]); // Smaller class should have higher weight

        // Test with single class
        let class_counts = vec![100];
        let balanced = balanced_weighted(&class_counts, 10, Some(42));
        assert_eq!(balanced.weights().len(), 1);
        assert!(balanced.weights()[0] > 0.0);
    }

    #[test]
    fn test_weighted_sampler_clone() {
        let weights = vec![1.0, 2.0, 3.0];
        let sampler = WeightedRandomSampler::new(weights.clone(), 10, true).with_generator(42);
        let cloned = sampler.clone();

        assert_eq!(sampler.weights(), cloned.weights());
        assert_eq!(sampler.num_samples(), cloned.num_samples());
        assert_eq!(sampler.replacement(), cloned.replacement());
        assert_eq!(sampler.generator(), cloned.generator());
    }

    #[test]
    fn test_subset_sampler_clone() {
        let indices = vec![1, 3, 5, 7];
        let sampler = SubsetRandomSampler::new(indices.clone()).with_generator(42);
        let cloned = sampler.clone();

        assert_eq!(sampler.indices(), cloned.indices());
        assert_eq!(sampler.generator(), cloned.generator());
    }
}
