//! Adaptive sampling functionality
//!
//! This module provides adaptive sampling strategies that dynamically adjust sampling
//! behavior based on training progress, model performance, and sample characteristics.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::random::Rng;

use super::core::{rng_utils, Sampler, SamplerIterator};

/// Adaptive sampling strategies
///
/// These strategies define different approaches for selecting samples based on
/// their characteristics such as difficulty, frequency, or uncertainty.
#[derive(Clone, Debug, PartialEq)]
pub enum AdaptiveStrategy {
    /// Focus on hard samples (high loss)
    ///
    /// Prioritizes samples with high training loss, which are typically
    /// harder for the model to learn.
    ///
    /// # Arguments
    ///
    /// * `intensity` - Controls how strongly to weight hard samples (typically 0.5-2.0)
    HardSampling { intensity: f64 },

    /// Focus on easy samples (low loss)
    ///
    /// Prioritizes samples with low training loss, which can help with
    /// stable training or curriculum learning.
    ///
    /// # Arguments
    ///
    /// * `intensity` - Controls how strongly to weight easy samples (typically 0.5-2.0)
    EasySampling { intensity: f64 },

    /// Uniform sampling for exploration
    ///
    /// Samples all data points with equal probability, providing a baseline
    /// for comparison and ensuring exploration.
    Uniform,

    /// Uncertainty-based sampling
    ///
    /// Focuses on samples where the model predictions are most uncertain.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Controls the sharpness of uncertainty weighting
    Uncertainty { temperature: f64 },

    /// Frequency-based inverse sampling
    ///
    /// Prioritizes samples that have been seen less frequently during training,
    /// helping to balance the training distribution.
    ///
    /// # Arguments
    ///
    /// * `power` - Controls the strength of inverse frequency weighting
    InverseFrequency { power: f64 },

    /// Gradient-based importance
    ///
    /// Focuses on samples that produce gradients above a certain threshold,
    /// indicating they contribute significantly to learning.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum gradient magnitude for inclusion
    GradientMagnitude { threshold: f64 },
}

impl Default for AdaptiveStrategy {
    fn default() -> Self {
        AdaptiveStrategy::Uniform
    }
}

/// Adaptive sampler that dynamically adjusts sampling strategy based on training progress
///
/// This sampler combines multiple sampling strategies and adapts the sampling distribution
/// based on model performance, loss patterns, and sample difficulty over time.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::sampler::{AdaptiveSampler, AdaptiveStrategy, Sampler};
///
/// let mut sampler = AdaptiveSampler::new(1000, 64)
///     .with_adaptation_rate(0.1)
///     .with_warmup_epochs(5)
///     .with_generator(42);
///
/// // Add custom strategy
/// sampler = sampler.add_strategy(
///     AdaptiveStrategy::InverseFrequency { power: 1.0 },
///     0.2
/// );
///
/// // During training, update with sample losses
/// let sample_indices = vec![0, 1, 2, 3, 4];
/// let losses = vec![0.5, 0.8, 0.3, 0.9, 0.2];
/// sampler.update_sample_losses(&sample_indices, &losses);
///
/// // Set current epoch for adaptation
/// sampler.set_epoch(10);
///
/// // Get adaptive samples
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 64);
/// ```
#[derive(Clone)]
pub struct AdaptiveSampler {
    dataset_size: usize,
    num_samples: usize,
    strategies: Vec<AdaptiveStrategy>,
    strategy_weights: Vec<f64>,
    sample_losses: Vec<f64>,
    sample_difficulties: Vec<f64>,
    sample_frequencies: Vec<usize>,
    adaptation_rate: f64,
    smoothing_factor: f64,
    current_epoch: usize,
    warmup_epochs: usize,
    generator: Option<u64>,
}

impl AdaptiveSampler {
    /// Create a new adaptive sampler
    ///
    /// Creates a sampler with default strategies: hard sampling, uniform, and uncertainty.
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Total size of the dataset
    /// * `num_samples` - Number of samples to select per iteration
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::AdaptiveSampler;
    ///
    /// let sampler = AdaptiveSampler::new(1000, 32);
    /// assert_eq!(sampler.len(), 32);
    /// ```
    pub fn new(dataset_size: usize, num_samples: usize) -> Self {
        let strategies = vec![
            AdaptiveStrategy::HardSampling { intensity: 1.0 },
            AdaptiveStrategy::Uniform,
            AdaptiveStrategy::Uncertainty { temperature: 1.0 },
        ];

        let strategy_weights = vec![0.4, 0.3, 0.3];

        Self {
            dataset_size,
            num_samples,
            strategies,
            strategy_weights,
            sample_losses: vec![0.0; dataset_size],
            sample_difficulties: vec![0.0; dataset_size],
            sample_frequencies: vec![0; dataset_size],
            adaptation_rate: 0.1,
            smoothing_factor: 0.9,
            current_epoch: 0,
            warmup_epochs: 5,
            generator: None,
        }
    }

    /// Add a custom sampling strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - The adaptive strategy to add
    /// * `weight` - Initial weight for this strategy (will be normalized)
    pub fn add_strategy(mut self, strategy: AdaptiveStrategy, weight: f64) -> Self {
        self.strategies.push(strategy);
        self.strategy_weights.push(weight);
        self.normalize_strategy_weights();
        self
    }

    /// Set adaptation rate for strategy weight updates
    ///
    /// # Arguments
    ///
    /// * `rate` - Adaptation rate (typically 0.01-0.2)
    pub fn with_adaptation_rate(mut self, rate: f64) -> Self {
        self.adaptation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set smoothing factor for exponential moving average of losses
    ///
    /// # Arguments
    ///
    /// * `factor` - Smoothing factor (0.0-1.0, higher values = more smoothing)
    pub fn with_smoothing_factor(mut self, factor: f64) -> Self {
        self.smoothing_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Set number of warmup epochs before adaptation begins
    ///
    /// # Arguments
    ///
    /// * `epochs` - Number of warmup epochs
    pub fn with_warmup_epochs(mut self, epochs: usize) -> Self {
        self.warmup_epochs = epochs;
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

    /// Get the current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Get the warmup epochs
    pub fn warmup_epochs(&self) -> usize {
        self.warmup_epochs
    }

    /// Get the adaptation rate
    pub fn adaptation_rate(&self) -> f64 {
        self.adaptation_rate
    }

    /// Get the smoothing factor
    pub fn smoothing_factor(&self) -> f64 {
        self.smoothing_factor
    }

    /// Get the current strategy weights
    pub fn strategy_weights(&self) -> &[f64] {
        &self.strategy_weights
    }

    /// Get the current strategies
    pub fn strategies(&self) -> &[AdaptiveStrategy] {
        &self.strategies
    }

    /// Get sample losses
    pub fn sample_losses(&self) -> &[f64] {
        &self.sample_losses
    }

    /// Get sample difficulties
    pub fn sample_difficulties(&self) -> &[f64] {
        &self.sample_difficulties
    }

    /// Get sample frequencies
    pub fn sample_frequencies(&self) -> &[usize] {
        &self.sample_frequencies
    }

    /// Check if the sampler is in warmup phase
    pub fn is_warming_up(&self) -> bool {
        self.current_epoch < self.warmup_epochs
    }

    /// Update sample losses from training
    ///
    /// This method should be called after each training batch to update
    /// the sampler's understanding of sample difficulty.
    ///
    /// # Arguments
    ///
    /// * `sample_indices` - Indices of samples in the batch
    /// * `losses` - Corresponding loss values for each sample
    ///
    /// # Panics
    ///
    /// Panics if the lengths of `sample_indices` and `losses` don't match.
    pub fn update_sample_losses(&mut self, sample_indices: &[usize], losses: &[f64]) {
        assert_eq!(sample_indices.len(), losses.len());

        for (&idx, &loss) in sample_indices.iter().zip(losses.iter()) {
            if idx < self.dataset_size {
                // Exponential moving average for loss smoothing
                self.sample_losses[idx] = self.smoothing_factor * self.sample_losses[idx]
                    + (1.0 - self.smoothing_factor) * loss;

                // Track sample frequency
                self.sample_frequencies[idx] += 1;
            }
        }

        self.update_sample_difficulties();
        self.adapt_strategy_weights();
    }

    /// Set current epoch for adaptation tracking
    ///
    /// # Arguments
    ///
    /// * `epoch` - Current training epoch
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    /// Reset sampler state
    pub fn reset(&mut self) {
        self.sample_losses.fill(0.0);
        self.sample_difficulties.fill(0.0);
        self.sample_frequencies.fill(0);
        self.current_epoch = 0;
    }

    /// Get statistics about the current adaptive sampling state
    pub fn adaptive_stats(&self) -> AdaptiveStats {
        let hard_samples = self
            .sample_difficulties
            .iter()
            .filter(|&&d| d > 0.5)
            .count();
        let max_freq = self.sample_frequencies.iter().max().copied().unwrap_or(0);
        let min_freq = self.sample_frequencies.iter().min().copied().unwrap_or(0);
        let mean_loss = self.sample_losses.iter().sum::<f64>() / self.sample_losses.len() as f64;

        AdaptiveStats {
            current_epoch: self.current_epoch,
            warmup_epochs: self.warmup_epochs,
            is_warming_up: self.is_warming_up(),
            hard_samples_count: hard_samples,
            hard_samples_ratio: hard_samples as f64 / self.dataset_size as f64,
            frequency_imbalance: if min_freq > 0 {
                max_freq as f64 / min_freq as f64
            } else {
                0.0
            },
            mean_loss,
            adaptation_rate: self.adaptation_rate,
            num_strategies: self.strategies.len(),
        }
    }

    /// Update sample difficulties based on current losses
    fn update_sample_difficulties(&mut self) {
        if self.sample_losses.is_empty() {
            return;
        }

        let mean_loss = self.sample_losses.iter().sum::<f64>() / self.sample_losses.len() as f64;
        let variance = self
            .sample_losses
            .iter()
            .map(|&loss| (loss - mean_loss).powi(2))
            .sum::<f64>()
            / self.sample_losses.len() as f64;
        let std_dev = variance.sqrt();

        for (i, &loss) in self.sample_losses.iter().enumerate() {
            // Normalize difficulty score
            self.sample_difficulties[i] = if std_dev > 0.0 {
                (loss - mean_loss) / std_dev
            } else {
                0.0
            };
        }
    }

    /// Adapt strategy weights based on current training state
    fn adapt_strategy_weights(&mut self) {
        if self.is_warming_up() {
            return;
        }

        // Calculate strategy performance metrics
        let hard_samples_ratio = self
            .sample_difficulties
            .iter()
            .filter(|&&d| d > 0.5)
            .count() as f64
            / self.dataset_size as f64;

        let frequency_imbalance = {
            let max_freq = self.sample_frequencies.iter().max().unwrap_or(&1);
            let min_freq = self.sample_frequencies.iter().min().unwrap_or(&1);
            (*max_freq as f64 / (*min_freq as f64).max(1.0)).ln()
        };

        // Adjust weights based on current training state
        let mut new_weights = self.strategy_weights.clone();

        // If too many hard samples, reduce hard sampling
        if hard_samples_ratio > 0.3 {
            for (i, strategy) in self.strategies.iter().enumerate() {
                match strategy {
                    AdaptiveStrategy::HardSampling { .. } => {
                        new_weights[i] *= 1.0 - self.adaptation_rate;
                    }
                    AdaptiveStrategy::EasySampling { .. } => {
                        new_weights[i] *= 1.0 + self.adaptation_rate;
                    }
                    _ => {}
                }
            }
        }

        // If high frequency imbalance, increase inverse frequency sampling
        if frequency_imbalance > 1.0 {
            for (i, strategy) in self.strategies.iter().enumerate() {
                if let AdaptiveStrategy::InverseFrequency { .. } = strategy {
                    new_weights[i] *= 1.0 + self.adaptation_rate;
                }
            }
        }

        self.strategy_weights = new_weights;
        self.normalize_strategy_weights();
    }

    /// Normalize strategy weights to sum to 1.0
    fn normalize_strategy_weights(&mut self) {
        let sum: f64 = self.strategy_weights.iter().sum();
        if sum > 0.0 {
            for weight in &mut self.strategy_weights {
                *weight /= sum;
            }
        } else {
            // If all weights are zero, reset to uniform
            let uniform_weight = 1.0 / self.strategy_weights.len() as f64;
            self.strategy_weights.fill(uniform_weight);
        }
    }

    /// Get sampling weights for a specific strategy
    fn get_strategy_weights(&self, strategy: &AdaptiveStrategy) -> Vec<f64> {
        match strategy {
            AdaptiveStrategy::HardSampling { intensity } => self
                .sample_difficulties
                .iter()
                .map(|&d| (d * intensity).exp())
                .collect(),
            AdaptiveStrategy::EasySampling { intensity } => self
                .sample_difficulties
                .iter()
                .map(|&d| (-d * intensity).exp())
                .collect(),
            AdaptiveStrategy::Uniform => {
                vec![1.0; self.dataset_size]
            }
            AdaptiveStrategy::Uncertainty { temperature } => self
                .sample_losses
                .iter()
                .map(|&loss| (loss / temperature).exp())
                .collect(),
            AdaptiveStrategy::InverseFrequency { power } => self
                .sample_frequencies
                .iter()
                .map(|&freq| 1.0 / (freq as f64 + 1.0).powf(*power))
                .collect(),
            AdaptiveStrategy::GradientMagnitude { threshold } => self
                .sample_losses
                .iter()
                .map(|&loss| if loss > *threshold { loss } else { 0.1 })
                .collect(),
        }
    }

    /// Combine weights from all strategies
    fn get_combined_weights(&self) -> Vec<f64> {
        let mut combined = vec![0.0; self.dataset_size];

        for (strategy, &weight) in self.strategies.iter().zip(self.strategy_weights.iter()) {
            let strategy_weights = self.get_strategy_weights(strategy);
            for (i, &w) in strategy_weights.iter().enumerate() {
                combined[i] += weight * w;
            }
        }

        // Ensure all weights are positive
        for w in &mut combined {
            *w = w.max(1e-6);
        }

        combined
    }

    /// Sample indices using weighted sampling with replacement
    fn sample_with_replacement(&self, weights: &[f64]) -> Vec<usize> {
        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = rng_utils::create_rng(self.generator);

        // Create cumulative distribution
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum <= 0.0 {
            // Fallback to uniform sampling
            return (0..self.num_samples)
                .map(|_| rng_utils::gen_range(&mut rng, 0..self.dataset_size))
                .collect();
        }

        let mut cumulative_weights = Vec::with_capacity(weights.len());
        let mut cumsum = 0.0;

        for &weight in weights {
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
                    .min(self.dataset_size - 1)
            })
            .collect()
    }
}

impl Sampler for AdaptiveSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let weights = self.get_combined_weights();
        let indices = self.sample_with_replacement(&weights);
        SamplerIterator::new(indices)
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

/// Statistics about the current adaptive sampling state
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveStats {
    /// Current training epoch
    pub current_epoch: usize,
    /// Number of warmup epochs
    pub warmup_epochs: usize,
    /// Whether currently in warmup phase
    pub is_warming_up: bool,
    /// Number of samples classified as hard
    pub hard_samples_count: usize,
    /// Ratio of hard samples to total samples
    pub hard_samples_ratio: f64,
    /// Imbalance in sample frequencies (max/min)
    pub frequency_imbalance: f64,
    /// Mean loss across all samples
    pub mean_loss: f64,
    /// Current adaptation rate
    pub adaptation_rate: f64,
    /// Number of active strategies
    pub num_strategies: usize,
}

/// Create an adaptive sampler with hard sampling focus
///
/// Convenience function for creating an adaptive sampler that emphasizes hard samples.
///
/// # Arguments
///
/// * `dataset_size` - Total size of the dataset
/// * `num_samples` - Number of samples to select per iteration
/// * `intensity` - Intensity of hard sampling focus
/// * `seed` - Optional random seed for reproducible sampling
pub fn hard_adaptive_sampler(
    dataset_size: usize,
    num_samples: usize,
    intensity: f64,
    seed: Option<u64>,
) -> AdaptiveSampler {
    let mut sampler = AdaptiveSampler::new(dataset_size, num_samples)
        .add_strategy(AdaptiveStrategy::HardSampling { intensity }, 0.7);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create an adaptive sampler with frequency balancing
///
/// Convenience function for creating an adaptive sampler that balances sample frequencies.
///
/// # Arguments
///
/// * `dataset_size` - Total size of the dataset
/// * `num_samples` - Number of samples to select per iteration
/// * `power` - Power for inverse frequency weighting
/// * `seed` - Optional random seed for reproducible sampling
pub fn frequency_balanced_sampler(
    dataset_size: usize,
    num_samples: usize,
    power: f64,
    seed: Option<u64>,
) -> AdaptiveSampler {
    let mut sampler = AdaptiveSampler::new(dataset_size, num_samples)
        .add_strategy(AdaptiveStrategy::InverseFrequency { power }, 0.6);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create an adaptive sampler with uncertainty focus
///
/// Convenience function for creating an adaptive sampler that emphasizes uncertain samples.
///
/// # Arguments
///
/// * `dataset_size` - Total size of the dataset
/// * `num_samples` - Number of samples to select per iteration
/// * `temperature` - Temperature for uncertainty weighting
/// * `seed` - Optional random seed for reproducible sampling
pub fn uncertainty_adaptive_sampler(
    dataset_size: usize,
    num_samples: usize,
    temperature: f64,
    seed: Option<u64>,
) -> AdaptiveSampler {
    let mut sampler = AdaptiveSampler::new(dataset_size, num_samples)
        .add_strategy(AdaptiveStrategy::Uncertainty { temperature }, 0.8);
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_sampler_basic() {
        let dataset_size = 100;
        let num_samples = 50;
        let sampler = AdaptiveSampler::new(dataset_size, num_samples).with_generator(42);

        assert_eq!(sampler.len(), num_samples);
        assert_eq!(sampler.current_epoch(), 0);
        assert_eq!(sampler.warmup_epochs(), 5);
        assert!(sampler.is_warming_up());
        assert_eq!(sampler.strategies().len(), 3); // Default strategies
        assert_eq!(sampler.strategy_weights().len(), 3);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), num_samples);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < dataset_size);
        }
    }

    #[test]
    fn test_adaptive_sampler_with_losses() {
        let dataset_size = 10;
        let num_samples = 5;
        let mut sampler = AdaptiveSampler::new(dataset_size, num_samples).with_generator(42);

        // Initially all difficulties should be zero
        assert!(sampler.sample_difficulties().iter().all(|&d| d == 0.0));

        // Simulate training with some sample losses
        let sample_indices = vec![0, 2, 4, 6, 8];
        let losses = vec![0.1, 0.8, 0.2, 0.9, 0.3]; // Indices 2 and 6 have high losses

        sampler.update_sample_losses(&sample_indices, &losses);

        // Check that losses were updated (with smoothing factor 0.9: new_loss = 0.9 * 0.0 + 0.1 * input)
        assert!((sampler.sample_losses()[0] - 0.01).abs() < 1e-10); // 0.1 * 0.1 = 0.01
        assert!((sampler.sample_losses()[2] - 0.08).abs() < 1e-10); // 0.1 * 0.8 = 0.08
        assert!((sampler.sample_losses()[6] - 0.09).abs() < 1e-10); // 0.1 * 0.9 = 0.09

        // Check that frequencies were updated
        assert_eq!(sampler.sample_frequencies()[0], 1);
        assert_eq!(sampler.sample_frequencies()[2], 1);
        assert_eq!(sampler.sample_frequencies()[1], 0); // Not sampled

        // Sample should still work
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), num_samples);
    }

    #[test]
    fn test_adaptive_sampler_strategy_adaptation() {
        let dataset_size = 20;
        let num_samples = 10;
        let mut sampler = AdaptiveSampler::new(dataset_size, num_samples)
            .with_warmup_epochs(2)
            .with_generator(42);

        // Add custom strategy
        sampler = sampler.add_strategy(AdaptiveStrategy::InverseFrequency { power: 1.0 }, 0.2);
        assert_eq!(sampler.strategies().len(), 4);

        // Initial weights should be normalized
        let initial_sum: f64 = sampler.strategy_weights().iter().sum();
        assert!((initial_sum - 1.0).abs() < f64::EPSILON);

        // During warmup, weights shouldn't change
        sampler.set_epoch(1);
        assert!(sampler.is_warming_up());

        let sample_indices: Vec<usize> = (0..10).collect();
        let losses = vec![0.5; 10];
        sampler.update_sample_losses(&sample_indices, &losses);

        let _weights_during_warmup = sampler.strategy_weights().to_vec();

        // After warmup, weights can adapt
        sampler.set_epoch(3);
        assert!(!sampler.is_warming_up());

        sampler.update_sample_losses(&sample_indices, &losses);
        // Weights might have changed (depending on adaptation logic)

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), num_samples);
    }

    #[test]
    fn test_adaptive_strategies() {
        let dataset_size = 10;
        let num_samples = 5;

        let strategies = vec![
            AdaptiveStrategy::HardSampling { intensity: 1.0 },
            AdaptiveStrategy::EasySampling { intensity: 1.0 },
            AdaptiveStrategy::Uniform,
            AdaptiveStrategy::Uncertainty { temperature: 1.0 },
            AdaptiveStrategy::InverseFrequency { power: 1.0 },
            AdaptiveStrategy::GradientMagnitude { threshold: 0.5 },
        ];

        for strategy in strategies {
            let sampler = AdaptiveSampler::new(dataset_size, num_samples)
                .add_strategy(strategy, 0.5)
                .with_generator(42);

            let indices: Vec<usize> = sampler.iter().collect();
            assert_eq!(indices.len(), num_samples);

            // All indices should be valid
            for &idx in &indices {
                assert!(idx < dataset_size);
            }
        }
    }

    #[test]
    fn test_adaptive_sampler_difficulty_calculation() {
        let dataset_size = 5;
        let num_samples = 3;
        let mut sampler = AdaptiveSampler::new(dataset_size, num_samples).with_generator(42);

        // Update with specific losses to test difficulty calculation
        let sample_indices = vec![0, 1, 2, 3, 4];
        let losses = vec![0.1, 0.2, 0.8, 0.9, 0.3]; // Clear difficulty pattern

        sampler.update_sample_losses(&sample_indices, &losses);

        let difficulties = sampler.sample_difficulties();

        // Indices 2 and 3 should have higher difficulty (positive scores)
        assert!(difficulties[2] > difficulties[0]);
        assert!(difficulties[3] > difficulties[1]);
        assert!(difficulties[2] > 0.0);
        assert!(difficulties[3] > 0.0);
    }

    #[test]
    fn test_adaptive_sampler_methods() {
        let mut sampler = AdaptiveSampler::new(20, 10)
            .with_adaptation_rate(0.2)
            .with_smoothing_factor(0.8)
            .with_warmup_epochs(3)
            .with_generator(42);

        assert_eq!(sampler.adaptation_rate(), 0.2);
        assert_eq!(sampler.smoothing_factor(), 0.8);
        assert_eq!(sampler.warmup_epochs(), 3);

        // Test epoch setting
        sampler.set_epoch(5);
        assert_eq!(sampler.current_epoch(), 5);
        assert!(!sampler.is_warming_up());

        // Test reset
        sampler.update_sample_losses(&[0, 1, 2], &[0.5, 0.6, 0.7]);
        assert!(sampler.sample_losses().iter().any(|&l| l > 0.0));
        assert!(sampler.sample_frequencies().iter().any(|&f| f > 0));

        sampler.reset();
        assert!(sampler.sample_losses().iter().all(|&l| l == 0.0));
        assert!(sampler.sample_frequencies().iter().all(|&f| f == 0));
        assert_eq!(sampler.current_epoch(), 0);
    }

    #[test]
    fn test_adaptive_stats() {
        let mut sampler = AdaptiveSampler::new(100, 32);

        let stats = sampler.adaptive_stats();
        assert_eq!(stats.current_epoch, 0);
        assert_eq!(stats.warmup_epochs, 5);
        assert!(stats.is_warming_up);
        assert_eq!(stats.hard_samples_count, 0);
        assert_eq!(stats.hard_samples_ratio, 0.0);
        assert_eq!(stats.mean_loss, 0.0);
        assert_eq!(stats.num_strategies, 3);

        // Update with losses and check stats
        let sample_indices: Vec<usize> = (0..20).collect();
        let losses: Vec<f64> = (0..20).map(|i| if i > 15 { 0.8 } else { 0.2 }).collect();
        sampler.update_sample_losses(&sample_indices, &losses);

        let stats = sampler.adaptive_stats();
        assert!(stats.mean_loss > 0.0);
        assert!(stats.hard_samples_count > 0);
        assert!(stats.hard_samples_ratio > 0.0);
    }

    #[test]
    fn test_convenience_functions() {
        // Test hard_adaptive_sampler
        let hard_sampler = hard_adaptive_sampler(100, 32, 1.5, Some(42));
        assert_eq!(hard_sampler.len(), 32);
        assert!(hard_sampler.strategies().len() > 3); // Should have added hard sampling

        // Test frequency_balanced_sampler
        let freq_sampler = frequency_balanced_sampler(100, 32, 1.0, Some(42));
        assert_eq!(freq_sampler.len(), 32);

        // Test uncertainty_adaptive_sampler
        let uncertainty_sampler = uncertainty_adaptive_sampler(100, 32, 0.8, Some(42));
        assert_eq!(uncertainty_sampler.len(), 32);
    }

    #[test]
    fn test_weight_normalization() {
        let mut sampler = AdaptiveSampler::new(10, 5);

        // Add strategies with arbitrary weights
        sampler = sampler
            .add_strategy(AdaptiveStrategy::HardSampling { intensity: 1.0 }, 2.0)
            .add_strategy(AdaptiveStrategy::EasySampling { intensity: 1.0 }, 3.0);

        // Weights should be normalized
        let sum: f64 = sampler.strategy_weights().iter().sum();
        assert!((sum - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_strategy_weights() {
        let sampler = AdaptiveSampler::new(5, 3);

        // Test uniform strategy
        let uniform_weights = sampler.get_strategy_weights(&AdaptiveStrategy::Uniform);
        assert!(uniform_weights.iter().all(|&w| w == 1.0));

        // Create sampler with some loss data
        let mut sampler_with_data = AdaptiveSampler::new(5, 3);
        sampler_with_data.update_sample_losses(&[0, 1, 2], &[0.1, 0.5, 0.9]);

        // Test hard sampling strategy
        let hard_weights = sampler_with_data
            .get_strategy_weights(&AdaptiveStrategy::HardSampling { intensity: 1.0 });
        assert_eq!(hard_weights.len(), 5);

        // Test easy sampling strategy
        let easy_weights = sampler_with_data
            .get_strategy_weights(&AdaptiveStrategy::EasySampling { intensity: 1.0 });
        assert_eq!(easy_weights.len(), 5);
    }

    #[test]
    fn test_edge_cases() {
        // Empty num_samples
        let empty_sampler = AdaptiveSampler::new(10, 0);
        assert_eq!(empty_sampler.len(), 0);
        let indices: Vec<usize> = empty_sampler.iter().collect();
        assert!(indices.is_empty());

        // Single sample
        let single_sampler = AdaptiveSampler::new(10, 1);
        let indices: Vec<usize> = single_sampler.iter().collect();
        assert_eq!(indices.len(), 1);

        // Large dataset
        let large_sampler = AdaptiveSampler::new(10000, 64);
        assert_eq!(large_sampler.len(), 64);

        // Invalid sample indices (should be ignored)
        let mut sampler = AdaptiveSampler::new(5, 3);
        sampler.update_sample_losses(&[0, 10, 2], &[0.1, 0.5, 0.3]); // Index 10 is out of bounds
                                                                     // Should not panic and should ignore invalid index

        // Zero weights should fallback to uniform
        let mut zero_weight_sampler = AdaptiveSampler::new(5, 3);
        zero_weight_sampler.strategy_weights = vec![0.0, 0.0, 0.0];
        zero_weight_sampler.normalize_strategy_weights();
        let sum: f64 = zero_weight_sampler.strategy_weights().iter().sum();
        assert!((sum - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adaptive_strategy_equality() {
        assert_eq!(
            AdaptiveStrategy::HardSampling { intensity: 1.0 },
            AdaptiveStrategy::HardSampling { intensity: 1.0 }
        );
        assert_eq!(AdaptiveStrategy::Uniform, AdaptiveStrategy::Uniform);
        assert_ne!(
            AdaptiveStrategy::HardSampling { intensity: 1.0 },
            AdaptiveStrategy::EasySampling { intensity: 1.0 }
        );
    }

    #[test]
    fn test_adaptive_strategy_default() {
        assert_eq!(AdaptiveStrategy::default(), AdaptiveStrategy::Uniform);
    }

    #[test]
    fn test_parameter_clamping() {
        let sampler = AdaptiveSampler::new(10, 5)
            .with_adaptation_rate(1.5) // Should be clamped to 1.0
            .with_smoothing_factor(-0.1); // Should be clamped to 0.0

        assert_eq!(sampler.adaptation_rate(), 1.0);
        assert_eq!(sampler.smoothing_factor(), 0.0);
    }

    #[test]
    fn test_reproducibility() {
        let mut sampler1 = AdaptiveSampler::new(20, 10).with_generator(123);
        let mut sampler2 = AdaptiveSampler::new(20, 10).with_generator(123);

        // Update both with same data
        let sample_indices = vec![0, 1, 2, 3, 4];
        let losses = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        sampler1.update_sample_losses(&sample_indices, &losses);
        sampler2.update_sample_losses(&sample_indices, &losses);

        let indices1: Vec<usize> = sampler1.iter().collect();
        let indices2: Vec<usize> = sampler2.iter().collect();

        assert_eq!(indices1, indices2);
    }
}
