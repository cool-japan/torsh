//! Stratified sampling functionality
//!
//! This module provides stratified sampling strategies that ensure proportional
//! representation of different classes or strata in the sampled data.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::collections::HashMap;

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::rand_prelude::SliceRandom;

use super::core::{rng_utils, Sampler, SamplerIterator};

/// Stratified sampler that samples proportionally from different strata/classes
///
/// This sampler ensures that samples from each stratum are drawn proportionally
/// to their size in the population, which is useful for maintaining class balance
/// in machine learning datasets.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::sampler::{StratifiedSampler, Sampler};
///
/// // Dataset with 3 classes: [0,0,0,1,1,1,2,2,2]
/// let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
/// let sampler = StratifiedSampler::new(&labels, 6, false).with_generator(42);
///
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 6); // 2 samples from each class
/// ```
#[derive(Debug, Clone)]
pub struct StratifiedSampler {
    strata: Vec<Vec<usize>>,
    num_samples: usize,
    replacement: bool,
    generator: Option<u64>,
}

impl StratifiedSampler {
    /// Create a new stratified sampler
    ///
    /// # Arguments
    ///
    /// * `labels` - Labels for each sample in the dataset
    /// * `num_samples` - Total number of samples to draw
    /// * `replacement` - Whether to sample with replacement
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::StratifiedSampler;
    ///
    /// let labels = vec![0, 0, 1, 1, 2, 2];
    /// let sampler = StratifiedSampler::new(&labels, 3, false);
    /// assert_eq!(sampler.num_strata(), 3);
    /// ```
    pub fn new(labels: &[usize], num_samples: usize, replacement: bool) -> Self {
        // Group indices by their labels
        let mut strata: HashMap<usize, Vec<usize>> = HashMap::new();

        for (idx, &label) in labels.iter().enumerate() {
            strata.entry(label).or_default().push(idx);
        }

        // Convert to Vec and sort by label to ensure deterministic ordering
        let mut strata_pairs: Vec<(usize, Vec<usize>)> = strata.into_iter().collect();
        strata_pairs.sort_unstable_by_key(|(label, _)| *label);
        let strata: Vec<Vec<usize>> = strata_pairs
            .into_iter()
            .map(|(_, indices)| indices)
            .collect();

        Self {
            strata,
            num_samples,
            replacement,
            generator: None,
        }
    }

    /// Create a stratified sampler from pre-grouped strata
    ///
    /// # Arguments
    ///
    /// * `strata` - Pre-grouped indices for each stratum
    /// * `num_samples` - Total number of samples to draw
    /// * `replacement` - Whether to sample with replacement
    pub fn from_strata(strata: Vec<Vec<usize>>, num_samples: usize, replacement: bool) -> Self {
        Self {
            strata,
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

    /// Get the number of strata
    pub fn num_strata(&self) -> usize {
        self.strata.len()
    }

    /// Get the strata
    pub fn strata(&self) -> &[Vec<usize>] {
        &self.strata
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

    /// Get the proportional sample count for each stratum
    ///
    /// This method calculates how many samples should be drawn from each stratum
    /// to maintain proportional representation in the final sample.
    pub fn get_stratum_sample_counts(&self) -> Vec<usize> {
        let total_population: usize = self.strata.iter().map(|s| s.len()).sum();

        if total_population == 0 {
            return vec![0; self.strata.len()];
        }

        let mut counts = Vec::with_capacity(self.strata.len());
        let mut allocated = 0;

        // Calculate proportional samples for each stratum
        for (i, stratum) in self.strata.iter().enumerate() {
            let count = if i == self.strata.len() - 1 {
                // Last stratum gets the remainder to ensure exact total
                self.num_samples.saturating_sub(allocated)
            } else {
                let proportion = stratum.len() as f64 / total_population as f64;
                (self.num_samples as f64 * proportion).round() as usize
            };

            counts.push(count);
            allocated += count;
        }

        counts
    }

    /// Get stratum sizes
    pub fn stratum_sizes(&self) -> Vec<usize> {
        self.strata.iter().map(|s| s.len()).collect()
    }

    /// Calculate the total population size across all strata
    pub fn total_population(&self) -> usize {
        self.strata.iter().map(|s| s.len()).sum()
    }

    /// Check if the sampler is valid (has non-empty strata)
    pub fn is_valid(&self) -> bool {
        !self.strata.is_empty() && self.total_population() > 0
    }
}

impl Sampler for StratifiedSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        if !self.is_valid() {
            return SamplerIterator::new(vec![]);
        }

        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = rng_utils::create_rng(self.generator);
        let stratum_counts = self.get_stratum_sample_counts();
        let mut all_indices = Vec::with_capacity(self.num_samples);

        // Sample from each stratum
        for (stratum, &count) in self.strata.iter().zip(stratum_counts.iter()) {
            if count == 0 || stratum.is_empty() {
                continue;
            }

            let stratum_samples: Vec<usize> = if self.replacement || count <= stratum.len() {
                if self.replacement {
                    // Sample with replacement
                    (0..count)
                        .map(|_| stratum[rng_utils::gen_range(&mut rng, 0..stratum.len())])
                        .collect()
                } else {
                    // Sample without replacement
                    let mut shuffled = stratum.clone();
                    shuffled.shuffle(&mut rng);
                    shuffled.into_iter().take(count).collect()
                }
            } else {
                // Need more samples than available in stratum - sample with replacement
                (0..count)
                    .map(|_| stratum[rng_utils::gen_range(&mut rng, 0..stratum.len())])
                    .collect()
            };

            all_indices.extend(stratum_samples);
        }

        // Shuffle the final combined indices to avoid grouping by stratum
        all_indices.shuffle(&mut rng);

        SamplerIterator::new(all_indices)
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

/// Create a stratified sampler from labels
///
/// Convenience function for creating a stratified sampler.
///
/// # Arguments
///
/// * `labels` - Labels for each sample in the dataset
/// * `num_samples` - Total number of samples to draw
/// * `replacement` - Whether to sample with replacement
/// * `seed` - Optional random seed for reproducible sampling
pub fn stratified(
    labels: &[usize],
    num_samples: usize,
    replacement: bool,
    seed: Option<u64>,
) -> StratifiedSampler {
    let mut sampler = StratifiedSampler::new(labels, num_samples, replacement);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a balanced stratified sampler
///
/// Creates a stratified sampler that draws equal numbers of samples from each stratum,
/// regardless of their original sizes.
///
/// # Arguments
///
/// * `labels` - Labels for each sample in the dataset
/// * `samples_per_stratum` - Number of samples to draw from each stratum
/// * `replacement` - Whether to sample with replacement
/// * `seed` - Optional random seed for reproducible sampling
pub fn balanced_stratified(
    labels: &[usize],
    samples_per_stratum: usize,
    replacement: bool,
    seed: Option<u64>,
) -> StratifiedSampler {
    // Group indices by labels
    let mut strata: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        strata.entry(label).or_default().push(idx);
    }

    let strata: Vec<Vec<usize>> = strata.into_values().collect();
    let num_samples = strata.len() * samples_per_stratum;

    let mut sampler = StratifiedSampler::from_strata(strata, num_samples, replacement);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a stratified train-test split
///
/// Splits the data into training and testing sets while maintaining
/// the proportion of samples from each class.
///
/// # Arguments
///
/// * `labels` - Labels for each sample in the dataset
/// * `test_ratio` - Proportion of data to use for testing (0.0 to 1.0)
/// * `seed` - Optional random seed for reproducible splits
///
/// # Returns
///
/// A tuple of (train_sampler, test_sampler)
pub fn stratified_train_test_split(
    labels: &[usize],
    test_ratio: f64,
    seed: Option<u64>,
) -> (StratifiedSampler, StratifiedSampler) {
    assert!(
        (0.0..=1.0).contains(&test_ratio),
        "test_ratio must be between 0.0 and 1.0"
    );

    // Group indices by labels
    let mut strata: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        strata.entry(label).or_default().push(idx);
    }

    let mut train_strata = Vec::new();
    let mut test_strata = Vec::new();

    // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
    let mut rng = rng_utils::create_rng(seed);

    for (_, mut stratum) in strata {
        // Shuffle the stratum
        stratum.shuffle(&mut rng);

        // Split into train and test
        let test_size = ((stratum.len() as f64) * test_ratio).round() as usize;
        let test_size = test_size.min(stratum.len());

        let (train_indices, test_indices) = stratum.split_at(stratum.len() - test_size);

        if !train_indices.is_empty() {
            train_strata.push(train_indices.to_vec());
        }
        if !test_indices.is_empty() {
            test_strata.push(test_indices.to_vec());
        }
    }

    let train_size = train_strata.iter().map(|s| s.len()).sum();
    let test_size = test_strata.iter().map(|s| s.len()).sum();

    let train_sampler = StratifiedSampler::from_strata(train_strata, train_size, false);
    let test_sampler = StratifiedSampler::from_strata(test_strata, test_size, false);

    (train_sampler, test_sampler)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratified_sampler_basic() {
        // Test with balanced classes: [0,0,0,1,1,1,2,2,2]
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let sampler = StratifiedSampler::new(&labels, 6, false).with_generator(42);

        assert_eq!(sampler.len(), 6);
        assert_eq!(sampler.num_strata(), 3);
        assert_eq!(sampler.num_samples(), 6);
        assert!(!sampler.replacement());
        assert_eq!(sampler.generator(), Some(42));
        assert!(sampler.is_valid());

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 6);

        // Check that all indices are valid
        for &idx in &indices {
            assert!(idx < labels.len());
        }

        // Count samples per class
        let mut class_counts = [0; 3];
        for &idx in &indices {
            class_counts[labels[idx]] += 1;
        }

        // Should be roughly proportional (2 samples per class for balanced classes)
        assert_eq!(class_counts[0], 2);
        assert_eq!(class_counts[1], 2);
        assert_eq!(class_counts[2], 2);
    }

    #[test]
    fn test_stratified_sampler_imbalanced() {
        // Test with imbalanced classes: [0,0,0,0,0,1,1,2]
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 2];
        let sampler = StratifiedSampler::new(&labels, 8, false).with_generator(42);

        assert_eq!(sampler.len(), 8);
        assert_eq!(sampler.num_strata(), 3);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 8);

        // Count samples per class
        let mut class_counts = [0; 3];
        for &idx in &indices {
            class_counts[labels[idx]] += 1;
        }

        // Should be proportional: class 0 (5/8 = 62.5%), class 1 (2/8 = 25%), class 2 (1/8 = 12.5%)
        // Expected: class 0 → 5 samples, class 1 → 2 samples, class 2 → 1 sample
        assert_eq!(class_counts[0], 5);
        assert_eq!(class_counts[1], 2);
        assert_eq!(class_counts[2], 1);
    }

    #[test]
    fn test_stratified_sampler_with_replacement() {
        let labels = vec![0, 1, 2];
        let sampler = StratifiedSampler::new(&labels, 9, true).with_generator(42);

        assert_eq!(sampler.len(), 9);
        assert!(sampler.replacement());

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 9);

        // Count samples per class
        let mut class_counts = [0; 3];
        for &idx in &indices {
            class_counts[labels[idx]] += 1;
        }

        // Should be roughly equal (3 samples per class)
        assert_eq!(class_counts[0], 3);
        assert_eq!(class_counts[1], 3);
        assert_eq!(class_counts[2], 3);
    }

    #[test]
    fn test_stratified_sampler_empty() {
        let labels: Vec<usize> = vec![];
        let sampler = StratifiedSampler::new(&labels, 5, false);

        assert_eq!(sampler.len(), 5);
        assert_eq!(sampler.num_strata(), 0);
        assert!(!sampler.is_valid());

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn test_stratified_sampler_single_stratum() {
        let labels = vec![0, 0, 0, 0, 0];
        let sampler = StratifiedSampler::new(&labels, 3, false).with_generator(42);

        assert_eq!(sampler.len(), 3);
        assert_eq!(sampler.num_strata(), 1);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 3);

        // All indices should be valid and from class 0
        for &idx in &indices {
            assert!(idx < 5);
            assert_eq!(labels[idx], 0);
        }
    }

    #[test]
    fn test_stratified_sampler_oversample() {
        // Test when requesting more samples than available
        let labels = vec![0, 1];
        let sampler = StratifiedSampler::new(&labels, 10, true).with_generator(42);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 10);

        // Should have samples from both classes
        let mut class_counts = [0; 2];
        for &idx in &indices {
            class_counts[labels[idx]] += 1;
        }

        assert!(class_counts[0] > 0);
        assert!(class_counts[1] > 0);
        assert_eq!(class_counts[0] + class_counts[1], 10);
    }

    #[test]
    fn test_stratified_sampler_from_strata() {
        let strata = vec![
            vec![0, 1, 2],    // First stratum
            vec![3, 4],       // Second stratum
            vec![5, 6, 7, 8], // Third stratum
        ];
        let sampler = StratifiedSampler::from_strata(strata.clone(), 6, false).with_generator(42);

        assert_eq!(sampler.len(), 6);
        assert_eq!(sampler.num_strata(), 3);
        assert_eq!(sampler.strata(), &strata);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 6);

        // All indices should be from the original strata
        for &idx in &indices {
            let found = strata.iter().any(|stratum| stratum.contains(&idx));
            assert!(found);
        }
    }

    #[test]
    fn test_stratified_sampler_properties() {
        let labels = vec![0, 1, 2, 0, 1, 2];
        let sampler = StratifiedSampler::new(&labels, 4, false);

        assert_eq!(sampler.stratum_sizes(), vec![2, 2, 2]);
        assert_eq!(sampler.total_population(), 6);

        let counts = sampler.get_stratum_sample_counts();
        assert_eq!(counts.iter().sum::<usize>(), 4); // Should sum to num_samples
    }

    #[test]
    fn test_convenience_functions() {
        let labels = vec![0, 0, 1, 1, 2, 2];

        // Test stratified convenience function
        let sampler = stratified(&labels, 4, false, Some(42));
        assert_eq!(sampler.len(), 4);
        assert_eq!(sampler.generator(), Some(42));

        // Test balanced_stratified convenience function
        let balanced = balanced_stratified(&labels, 2, false, Some(42));
        assert_eq!(balanced.len(), 6); // 3 strata * 2 samples each
        assert_eq!(balanced.generator(), Some(42));

        let indices: Vec<usize> = balanced.iter().collect();
        assert_eq!(indices.len(), 6);

        // Count samples per class - should be exactly 2 each
        let mut class_counts = [0; 3];
        for &idx in &indices {
            class_counts[labels[idx]] += 1;
        }
        assert_eq!(class_counts[0], 2);
        assert_eq!(class_counts[1], 2);
        assert_eq!(class_counts[2], 2);
    }

    #[test]
    fn test_stratified_train_test_split() {
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let (train_sampler, test_sampler) = stratified_train_test_split(&labels, 0.25, Some(42));

        // Total samples should equal original dataset size
        assert_eq!(train_sampler.len() + test_sampler.len(), labels.len());

        // Should maintain proportion in both sets
        let train_indices: Vec<usize> = train_sampler.iter().collect();
        let test_indices: Vec<usize> = test_sampler.iter().collect();

        // Count classes in train set
        let mut train_class_counts = [0; 3];
        for &idx in &train_indices {
            train_class_counts[labels[idx]] += 1;
        }

        // Count classes in test set
        let mut test_class_counts = [0; 3];
        for &idx in &test_indices {
            test_class_counts[labels[idx]] += 1;
        }

        // Each class should appear in both train and test sets
        for i in 0..3 {
            assert!(train_class_counts[i] > 0);
            assert!(test_class_counts[i] > 0);
            assert_eq!(train_class_counts[i] + test_class_counts[i], 4); // 4 samples per class
        }
    }

    #[test]
    #[should_panic(expected = "test_ratio must be between 0.0 and 1.0")]
    fn test_stratified_train_test_split_invalid_ratio() {
        let labels = vec![0, 1, 2];
        stratified_train_test_split(&labels, 1.5, None);
    }

    #[test]
    fn test_stratified_sampler_clone() {
        let labels = vec![0, 1, 2, 0, 1, 2];
        let sampler = StratifiedSampler::new(&labels, 4, false).with_generator(42);
        let cloned = sampler.clone();

        assert_eq!(sampler.len(), cloned.len());
        assert_eq!(sampler.num_strata(), cloned.num_strata());
        assert_eq!(sampler.replacement(), cloned.replacement());
        assert_eq!(sampler.generator(), cloned.generator());
        assert_eq!(sampler.strata(), cloned.strata());
    }

    #[test]
    fn test_stratified_sampler_reproducible() {
        let labels = vec![0, 0, 1, 1, 2, 2];
        let sampler1 = StratifiedSampler::new(&labels, 4, false).with_generator(123);
        let sampler2 = StratifiedSampler::new(&labels, 4, false).with_generator(123);

        let indices1: Vec<usize> = sampler1.iter().collect();
        let indices2: Vec<usize> = sampler2.iter().collect();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_edge_cases() {
        // Test with zero samples requested
        let labels = vec![0, 1, 2];
        let sampler = StratifiedSampler::new(&labels, 0, false);
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 0);

        // Test with large number of samples
        let labels = vec![0, 1];
        let sampler = StratifiedSampler::new(&labels, 1000, true);
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 1000);
    }
}
