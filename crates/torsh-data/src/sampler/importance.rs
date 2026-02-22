//! Importance sampling functionality
//!
//! This module provides importance sampling strategies that adjust sampling probabilities
//! to correct for dataset bias or to emphasize certain types of samples during training.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::random::Rng;

use super::core::{rng_utils, Sampler, SamplerIterator};

/// Importance sampling for biased data handling
///
/// This sampler adjusts sampling probabilities to correct for dataset bias
/// or to emphasize certain types of samples during training. It allows for
/// sophisticated control over the sampling distribution through importance weights.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::sampler::{ImportanceSampler, Sampler};
///
/// // Create importance weights (higher = more important)
/// let importance_weights = vec![0.1, 0.5, 1.0, 0.3, 0.8];
/// let sampler = ImportanceSampler::new(importance_weights, 3, true)
///     .with_temperature(0.5)
///     .with_generator(42);
///
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct ImportanceSampler {
    importance_weights: Vec<f64>,
    num_samples: usize,
    replacement: bool,
    temperature: f64,
    generator: Option<u64>,
}

impl ImportanceSampler {
    /// Create a new importance sampler
    ///
    /// # Arguments
    ///
    /// * `importance_weights` - Importance weights for each sample (higher = more important)
    /// * `num_samples` - Number of samples to select
    /// * `replacement` - Whether to sample with replacement
    ///
    /// # Panics
    ///
    /// Panics if importance_weights is empty or contains negative values.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::ImportanceSampler;
    ///
    /// let weights = vec![1.0, 2.0, 0.5, 3.0]; // Index 3 is most important
    /// let sampler = ImportanceSampler::new(weights, 2, false);
    /// ```
    pub fn new(importance_weights: Vec<f64>, num_samples: usize, replacement: bool) -> Self {
        // Validate importance weights
        assert!(
            !importance_weights.is_empty() || num_samples == 0,
            "importance_weights cannot be empty when num_samples > 0"
        );
        assert!(
            importance_weights.iter().all(|&w| w >= 0.0),
            "importance_weights must be non-negative"
        );

        // Skip weight sum validation for empty weights (when num_samples is 0)
        if !importance_weights.is_empty() {
            let weight_sum: f64 = importance_weights.iter().sum();
            assert!(
                weight_sum > 0.0 && weight_sum.is_finite(),
                "importance_weights must sum to a positive finite value"
            );
        }

        // Clamp num_samples to maximum available when sampling without replacement
        let clamped_num_samples = if !replacement {
            num_samples.min(importance_weights.len())
        } else {
            num_samples
        };

        Self {
            importance_weights,
            num_samples: clamped_num_samples,
            replacement,
            temperature: 1.0,
            generator: None,
        }
    }

    /// Set temperature for softmax scaling of importance weights
    ///
    /// Temperature controls the sharpness of the importance distribution:
    /// - Higher temperature (> 1.0) = more uniform sampling
    /// - Lower temperature (< 1.0) = more biased toward high importance samples
    /// - Temperature = 1.0 = no scaling
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature value (must be positive)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::ImportanceSampler;
    ///
    /// let weights = vec![1.0, 2.0, 3.0];
    /// let sampler = ImportanceSampler::new(weights, 2, true)
    ///     .with_temperature(0.5); // More emphasis on high importance
    /// ```
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        assert!(temperature > 0.0, "temperature must be positive");
        self.temperature = temperature;
        self
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

    /// Get the importance weights
    pub fn importance_weights(&self) -> &[f64] {
        &self.importance_weights
    }

    /// Get the number of samples
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Check if sampling with replacement
    pub fn replacement(&self) -> bool {
        self.replacement
    }

    /// Get the temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get the generator seed if set
    pub fn generator(&self) -> Option<u64> {
        self.generator
    }

    /// Update importance weights
    ///
    /// # Arguments
    ///
    /// * `new_weights` - New importance weights
    ///
    /// # Panics
    ///
    /// Panics if new_weights has different length than original weights.
    pub fn update_weights(&mut self, new_weights: Vec<f64>) {
        assert_eq!(
            new_weights.len(),
            self.importance_weights.len(),
            "New weights must have same length as original weights"
        );
        assert!(
            new_weights.iter().all(|&w| w >= 0.0),
            "importance_weights must be non-negative"
        );

        let weight_sum: f64 = new_weights.iter().sum();
        assert!(
            weight_sum > 0.0 && weight_sum.is_finite(),
            "importance_weights must sum to a positive finite value"
        );

        self.importance_weights = new_weights;
    }

    /// Apply temperature scaling to importance weights
    fn get_scaled_weights(&self) -> Vec<f64> {
        if (self.temperature - 1.0).abs() < f64::EPSILON {
            return self.importance_weights.clone();
        }

        // Apply temperature scaling (like softmax)
        let max_weight = self
            .importance_weights
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        self.importance_weights
            .iter()
            .map(|&w| ((w - max_weight) / self.temperature).exp())
            .collect()
    }

    /// Sample indices with importance weighting and replacement
    fn sample_with_replacement(&self) -> Vec<usize> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = rng_utils::create_rng(self.generator);

        let scaled_weights = self.get_scaled_weights();
        let weight_sum: f64 = scaled_weights.iter().sum();

        // Create cumulative distribution
        let mut cumulative_weights = Vec::with_capacity(scaled_weights.len());
        let mut cumsum = 0.0;

        for &weight in &scaled_weights {
            cumsum += weight / weight_sum;
            cumulative_weights.push(cumsum);
        }

        // Ensure the last value is exactly 1.0
        if let Some(last) = cumulative_weights.last_mut() {
            *last = 1.0;
        }

        // Sample using inverse transform sampling
        (0..self.num_samples)
            .map(|_| {
                let rand_val: f64 = rng.random();
                cumulative_weights
                    .binary_search_by(|&x| {
                        x.partial_cmp(&rand_val)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or_else(|i| i)
                    .min(self.importance_weights.len() - 1)
            })
            .collect()
    }

    /// Sample indices with importance weighting without replacement
    fn sample_without_replacement(&self) -> Vec<usize> {
        if self.num_samples >= self.importance_weights.len() {
            // Return all indices if we need more samples than available
            return (0..self.importance_weights.len()).collect();
        }

        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = rng_utils::create_rng(self.generator);

        let scaled_weights = self.get_scaled_weights();
        let mut selected_indices = Vec::new();
        let mut remaining_weights = scaled_weights;
        let mut remaining_indices: Vec<usize> = (0..self.importance_weights.len()).collect();

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
            let rand_val: f64 = rng.random::<f64>() * weight_sum;

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

    /// Get sampling statistics
    pub fn sampling_stats(&self) -> ImportanceStats {
        let scaled_weights = self.get_scaled_weights();
        let weight_sum: f64 = scaled_weights.iter().sum();
        let mean_weight = weight_sum / scaled_weights.len() as f64;

        let variance = scaled_weights
            .iter()
            .map(|&w| (w - mean_weight).powi(2))
            .sum::<f64>()
            / scaled_weights.len() as f64;

        let max_weight = scaled_weights
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_weight = scaled_weights.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        ImportanceStats {
            num_samples: self.num_samples,
            total_items: self.importance_weights.len(),
            replacement: self.replacement,
            temperature: self.temperature,
            mean_weight,
            weight_variance: variance,
            weight_range: max_weight - min_weight,
            weight_ratio: if min_weight > 0.0 {
                max_weight / min_weight
            } else {
                f64::INFINITY
            },
        }
    }
}

impl Sampler for ImportanceSampler {
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
        if self.replacement {
            self.num_samples
        } else {
            self.num_samples.min(self.importance_weights.len())
        }
    }
}

/// Statistics about importance sampling
#[derive(Debug, Clone, PartialEq)]
pub struct ImportanceStats {
    /// Number of samples to be drawn
    pub num_samples: usize,
    /// Total number of items in the dataset
    pub total_items: usize,
    /// Whether sampling with replacement
    pub replacement: bool,
    /// Temperature scaling factor
    pub temperature: f64,
    /// Mean importance weight (after scaling)
    pub mean_weight: f64,
    /// Variance in importance weights
    pub weight_variance: f64,
    /// Range of importance weights (max - min)
    pub weight_range: f64,
    /// Ratio of max to min importance weights
    pub weight_ratio: f64,
}

/// Create an importance sampler with uniform weights
///
/// Convenience function for creating an importance sampler with equal weights
/// for all samples (equivalent to uniform sampling).
///
/// # Arguments
///
/// * `dataset_size` - Size of the dataset
/// * `num_samples` - Number of samples to select
/// * `replacement` - Whether to sample with replacement
/// * `seed` - Optional random seed for reproducible sampling
pub fn uniform_importance_sampler(
    dataset_size: usize,
    num_samples: usize,
    replacement: bool,
    seed: Option<u64>,
) -> ImportanceSampler {
    let weights = vec![1.0; dataset_size];
    let mut sampler = ImportanceSampler::new(weights, num_samples, replacement);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create an importance sampler from class frequencies for class balancing
///
/// Creates importance weights that are inversely proportional to class frequencies,
/// helping to balance training for imbalanced datasets.
///
/// # Arguments
///
/// * `class_labels` - Class label for each sample
/// * `num_samples` - Number of samples to select
/// * `replacement` - Whether to sample with replacement
/// * `seed` - Optional random seed for reproducible sampling
pub fn class_balanced_importance_sampler(
    class_labels: &[usize],
    num_samples: usize,
    replacement: bool,
    seed: Option<u64>,
) -> ImportanceSampler {
    // Count frequency of each class
    let max_class = class_labels.iter().max().copied().unwrap_or(0);
    let mut class_counts = vec![0usize; max_class + 1];

    for &label in class_labels {
        if label <= max_class {
            class_counts[label] += 1;
        }
    }

    // Calculate inverse frequency weights
    let total_samples = class_labels.len() as f64;
    let num_classes = class_counts.len() as f64;

    let weights: Vec<f64> = class_labels
        .iter()
        .map(|&label| {
            let class_count = class_counts[label];
            if class_count > 0 {
                total_samples / (num_classes * class_count as f64)
            } else {
                1.0
            }
        })
        .collect();

    let mut sampler = ImportanceSampler::new(weights, num_samples, replacement);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create an importance sampler based on loss values
///
/// Creates importance weights based on training losses, emphasizing
/// samples with higher losses (harder samples).
///
/// # Arguments
///
/// * `losses` - Loss value for each sample
/// * `num_samples` - Number of samples to select
/// * `replacement` - Whether to sample with replacement
/// * `power` - Power to raise losses to (higher = more emphasis on hard samples)
/// * `seed` - Optional random seed for reproducible sampling
pub fn loss_based_importance_sampler(
    losses: &[f64],
    num_samples: usize,
    replacement: bool,
    power: f64,
    seed: Option<u64>,
) -> ImportanceSampler {
    let weights: Vec<f64> = losses
        .iter()
        .map(|&loss| loss.max(1e-6).powf(power)) // Avoid zero weights
        .collect();

    let mut sampler = ImportanceSampler::new(weights, num_samples, replacement);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create an importance sampler with exponential weighting
///
/// Creates importance weights using exponential scaling, which can provide
/// smoother transitions between importance levels.
///
/// # Arguments
///
/// * `scores` - Importance scores for each sample
/// * `num_samples` - Number of samples to select
/// * `replacement` - Whether to sample with replacement
/// * `scale` - Exponential scaling factor
/// * `seed` - Optional random seed for reproducible sampling
pub fn exponential_importance_sampler(
    scores: &[f64],
    num_samples: usize,
    replacement: bool,
    scale: f64,
    seed: Option<u64>,
) -> ImportanceSampler {
    let weights: Vec<f64> = scores.iter().map(|&score| (score * scale).exp()).collect();

    let mut sampler = ImportanceSampler::new(weights, num_samples, replacement);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_sampler_basic() {
        let importance_weights = vec![0.1, 0.5, 1.0, 0.3, 0.8];
        let sampler = ImportanceSampler::new(importance_weights.clone(), 3, true)
            .with_temperature(1.0)
            .with_generator(42);

        assert_eq!(sampler.importance_weights(), &importance_weights);
        assert_eq!(sampler.num_samples(), 3);
        assert!(sampler.replacement());
        assert_eq!(sampler.temperature(), 1.0);
        assert_eq!(sampler.generator(), Some(42));

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 3);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_importance_sampler_without_replacement() {
        let importance_weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sampler = ImportanceSampler::new(importance_weights, 3, false).with_generator(42);

        assert!(!sampler.replacement());
        assert_eq!(sampler.len(), 3);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 3);

        // All indices should be unique
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        sorted_indices.dedup();
        assert_eq!(sorted_indices.len(), 3);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_importance_sampler_temperature_scaling() {
        let importance_weights = vec![1.0, 10.0]; // Very different weights

        // Low temperature should emphasize differences
        let low_temp_sampler = ImportanceSampler::new(importance_weights.clone(), 10, true)
            .with_temperature(0.1)
            .with_generator(42);

        // High temperature should make more uniform
        let high_temp_sampler = ImportanceSampler::new(importance_weights.clone(), 10, true)
            .with_temperature(10.0)
            .with_generator(42);

        let low_temp_indices: Vec<usize> = low_temp_sampler.iter().collect();
        let high_temp_indices: Vec<usize> = high_temp_sampler.iter().collect();

        // Count occurrences of index 1 (higher weight)
        let low_temp_high_weight_count = low_temp_indices.iter().filter(|&&i| i == 1).count();
        let high_temp_high_weight_count = high_temp_indices.iter().filter(|&&i| i == 1).count();

        // Low temperature should favor high weight index more
        assert!(low_temp_high_weight_count >= high_temp_high_weight_count);
    }

    #[test]
    fn test_importance_sampler_edge_cases() {
        // Single sample
        let single_weight = vec![1.0];
        let single_sampler = ImportanceSampler::new(single_weight, 1, false);
        let indices: Vec<usize> = single_sampler.iter().collect();
        assert_eq!(indices, vec![0]);

        // Zero samples
        let zero_sampler = ImportanceSampler::new(vec![1.0, 2.0], 0, true);
        assert_eq!(zero_sampler.len(), 0);
        let indices: Vec<usize> = zero_sampler.iter().collect();
        assert!(indices.is_empty());

        // More samples than available (without replacement)
        let limited_sampler = ImportanceSampler::new(vec![1.0, 2.0], 5, false);
        assert_eq!(limited_sampler.len(), 2); // Should be clamped
        let indices: Vec<usize> = limited_sampler.iter().collect();
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_importance_sampler_extreme_weights() {
        // One weight much higher than others
        let extreme_weights = vec![0.001, 0.001, 1000.0, 0.001];
        let sampler = ImportanceSampler::new(extreme_weights, 20, true).with_generator(42);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 20);

        // Should heavily favor index 2
        let high_weight_count = indices.iter().filter(|&&i| i == 2).count();
        assert!(high_weight_count > 10); // Should be most of the samples
    }

    #[test]
    fn test_update_weights() {
        let mut sampler = ImportanceSampler::new(vec![1.0, 2.0, 3.0], 2, true);

        let new_weights = vec![3.0, 1.0, 2.0];
        sampler.update_weights(new_weights.clone());

        assert_eq!(sampler.importance_weights(), &new_weights);
    }

    #[test]
    fn test_sampling_stats() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sampler = ImportanceSampler::new(weights, 3, true);

        let stats = sampler.sampling_stats();
        assert_eq!(stats.num_samples, 3);
        assert_eq!(stats.total_items, 5);
        assert!(stats.replacement);
        assert_eq!(stats.temperature, 1.0);
        assert!(stats.mean_weight > 0.0);
        assert!(stats.weight_variance >= 0.0);
        assert!(stats.weight_range >= 0.0);
        assert!(stats.weight_ratio >= 1.0);
    }

    #[test]
    fn test_convenience_functions() {
        // Test uniform_importance_sampler
        let uniform = uniform_importance_sampler(10, 5, true, Some(42));
        assert_eq!(uniform.importance_weights().len(), 10);
        assert!(uniform.importance_weights().iter().all(|&w| w == 1.0));

        // Test class_balanced_importance_sampler
        let class_labels = vec![0, 0, 0, 1, 1, 2]; // Imbalanced: 3, 2, 1
        let balanced = class_balanced_importance_sampler(&class_labels, 4, true, Some(42));
        let weights = balanced.importance_weights();

        // Class 2 (1 sample) should have highest weight
        // Class 0 (3 samples) should have lowest weight
        assert!(weights[5] > weights[3]); // Class 2 > Class 1
        assert!(weights[3] > weights[0]); // Class 1 > Class 0

        // Test loss_based_importance_sampler
        let losses = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let loss_based = loss_based_importance_sampler(&losses, 3, true, 1.0, Some(42));
        let weights = loss_based.importance_weights();

        // Higher loss should give higher weight
        assert!(weights[3] > weights[2]); // Loss 0.9 > 0.3
        assert!(weights[1] > weights[0]); // Loss 0.8 > 0.1

        // Test exponential_importance_sampler
        let scores = vec![1.0, 2.0, 3.0];
        let exponential = exponential_importance_sampler(&scores, 2, true, 1.0, Some(42));
        let weights = exponential.importance_weights();

        // Should follow exponential relationship
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_scaled_weights() {
        let weights = vec![1.0, 2.0, 3.0];
        let sampler = ImportanceSampler::new(weights.clone(), 2, true);

        // Temperature = 1.0 should not change weights
        let scaled_1 = sampler.get_scaled_weights();
        assert_eq!(scaled_1, weights);

        // Lower temperature should increase differences
        let sampler_low = sampler.clone().with_temperature(0.5);
        let scaled_low = sampler_low.get_scaled_weights();

        // Higher temperature should decrease differences
        let sampler_high = sampler.clone().with_temperature(2.0);
        let scaled_high = sampler_high.get_scaled_weights();

        // Check that scaling affects the distribution
        assert_ne!(scaled_low, weights);
        assert_ne!(scaled_high, weights);
    }

    #[test]
    fn test_reproducibility() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sampler1 = ImportanceSampler::new(weights.clone(), 3, true).with_generator(123);
        let sampler2 = ImportanceSampler::new(weights, 3, true).with_generator(123);

        let indices1: Vec<usize> = sampler1.iter().collect();
        let indices2: Vec<usize> = sampler2.iter().collect();

        assert_eq!(indices1, indices2);
    }

    #[test]
    #[should_panic(expected = "importance_weights cannot be empty")]
    fn test_empty_weights() {
        ImportanceSampler::new(vec![], 5, true);
    }

    #[test]
    #[should_panic(expected = "importance_weights must be non-negative")]
    fn test_negative_weights() {
        ImportanceSampler::new(vec![1.0, -1.0, 2.0], 3, true);
    }

    #[test]
    #[should_panic(expected = "importance_weights must sum to a positive finite value")]
    fn test_zero_sum_weights() {
        ImportanceSampler::new(vec![0.0, 0.0, 0.0], 2, true);
    }

    #[test]
    fn test_invalid_no_replacement() {
        // Requesting more samples than available without replacement should clamp to available
        let sampler = ImportanceSampler::new(vec![1.0, 2.0], 5, false);
        assert_eq!(sampler.len(), 2); // Should be clamped to available items
    }

    #[test]
    #[should_panic(expected = "temperature must be positive")]
    fn test_invalid_temperature() {
        ImportanceSampler::new(vec![1.0, 2.0], 1, true).with_temperature(0.0);
    }

    #[test]
    #[should_panic(expected = "New weights must have same length")]
    fn test_update_weights_wrong_size() {
        let mut sampler = ImportanceSampler::new(vec![1.0, 2.0, 3.0], 2, true);
        sampler.update_weights(vec![1.0, 2.0]); // Wrong size
    }

    #[test]
    fn test_class_balanced_edge_cases() {
        // Empty labels
        let balanced_empty = class_balanced_importance_sampler(&[], 0, true, None);
        assert!(balanced_empty.importance_weights().is_empty());

        // Single class
        let single_class = vec![0, 0, 0];
        let balanced_single = class_balanced_importance_sampler(&single_class, 2, true, None);
        let weights = balanced_single.importance_weights();
        assert!(weights.iter().all(|&w| w > 0.0));
        // All weights should be equal for single class
        assert!((weights[0] - weights[1]).abs() < f64::EPSILON);

        // Large class numbers
        let large_classes = vec![0, 100, 5];
        let balanced_large = class_balanced_importance_sampler(&large_classes, 2, true, None);
        assert_eq!(balanced_large.importance_weights().len(), 3);
    }

    #[test]
    fn test_loss_based_edge_cases() {
        // Zero losses
        let zero_losses = vec![0.0, 0.0, 0.0];
        let loss_sampler = loss_based_importance_sampler(&zero_losses, 2, true, 1.0, None);
        let weights = loss_sampler.importance_weights();
        assert!(weights.iter().all(|&w| w > 0.0)); // Should have minimum weights

        // Extreme losses
        let extreme_losses = vec![1e-10, 1e10];
        let extreme_sampler = loss_based_importance_sampler(&extreme_losses, 1, true, 1.0, None);
        let weights = extreme_sampler.importance_weights();
        assert!(weights[1] > weights[0]); // High loss should have higher weight
    }

    #[test]
    fn test_exponential_edge_cases() {
        // Negative scores
        let negative_scores = vec![-1.0, 0.0, 1.0];
        let exp_sampler = exponential_importance_sampler(&negative_scores, 2, true, 1.0, None);
        let weights = exp_sampler.importance_weights();
        assert!(weights.iter().all(|&w| w > 0.0)); // All weights should be positive
        assert!(weights[2] > weights[1]); // Higher score should have higher weight
        assert!(weights[1] > weights[0]); // exp(0) > exp(-1)

        // Large scale factor
        let scores = vec![1.0, 2.0];
        let large_scale = exponential_importance_sampler(&scores, 1, true, 10.0, None);
        let weights = large_scale.importance_weights();
        assert!(weights[1] > weights[0]); // Should maintain order
    }
}
