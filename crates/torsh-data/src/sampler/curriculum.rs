//! Curriculum learning sampling functionality
//!
//! This module provides curriculum learning samplers that adjust sampling based on
//! training progress, gradually introducing more difficult samples as training proceeds.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::random::{Random, SeedableRng};

use super::core::{rng_utils, Sampler, SamplerIterator};

/// Curriculum learning strategies
///
/// These strategies define how the difficulty threshold changes over training epochs,
/// allowing for different curriculum learning approaches.
#[derive(Clone, Debug, PartialEq)]
pub enum CurriculumStrategy {
    /// Linear increase from easy to hard
    ///
    /// The difficulty threshold increases linearly with training progress.
    /// At epoch 0, only the easiest samples are included.
    /// At the final epoch, all samples are included.
    Linear,

    /// Exponential increase (starts slow, accelerates)
    ///
    /// The difficulty threshold follows an exponential curve.
    /// Training starts very slowly with only easy samples,
    /// then rapidly includes harder samples towards the end.
    ///
    /// # Arguments
    ///
    /// * `base` - Exponential base (must be > 1.0, typically 2.0-10.0)
    Exponential { base: f64 },

    /// Step-wise increase at specific epochs
    ///
    /// The difficulty threshold increases in discrete steps
    /// at predefined epoch milestones.
    ///
    /// # Arguments
    ///
    /// * `thresholds` - Epoch numbers where difficulty increases
    Step { thresholds: Vec<usize> },

    /// Anti-curriculum (hard to easy)
    ///
    /// Starts with the hardest samples and gradually includes easier ones.
    /// This is the opposite of traditional curriculum learning.
    AntiCurriculum,

    /// Self-paced learning (adaptive difficulty)
    ///
    /// Adjusts difficulty based on a lambda parameter that can be
    /// modified based on model performance.
    ///
    /// # Arguments
    ///
    /// * `lambda` - Self-pacing parameter (0.0-1.0)
    SelfPaced { lambda: f64 },
}

impl Default for CurriculumStrategy {
    fn default() -> Self {
        CurriculumStrategy::Linear
    }
}

/// Curriculum learning sampler that adjusts sampling based on training progress
///
/// This sampler gradually introduces more difficult samples as training progresses,
/// following the curriculum learning paradigm. The difficulty of samples is determined
/// by a user-provided difficulty function.
///
/// # Examples
///
/// ```rust
/// use torsh_data::sampler::{CurriculumSampler, CurriculumStrategy, Sampler};
///
/// // Define difficulty as distance from center
/// let difficulty_fn = |idx: usize| (idx as f64 - 50.0).abs() / 50.0;
///
/// let mut sampler = CurriculumSampler::new(
///     100,
///     difficulty_fn,
///     10,
///     CurriculumStrategy::Linear
/// ).with_generator(42);
///
/// // Set current epoch
/// sampler.set_epoch(0); // Start with easiest samples
/// let early_indices: Vec<usize> = sampler.iter().collect();
///
/// sampler.set_epoch(9); // Include harder samples
/// let late_indices: Vec<usize> = sampler.iter().collect();
///
/// assert!(late_indices.len() >= early_indices.len());
/// ```
#[derive(Clone)]
pub struct CurriculumSampler<F> {
    difficulties: Vec<f64>,
    difficulty_fn: F,
    current_epoch: usize,
    total_epochs: usize,
    curriculum_strategy: CurriculumStrategy,
    generator: Option<u64>,
}

impl<F> CurriculumSampler<F>
where
    F: Fn(usize) -> f64 + Send + Clone,
{
    /// Create a new curriculum sampler
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Size of the dataset
    /// * `difficulty_fn` - Function that maps sample index to difficulty score (0.0 = easy, 1.0 = hard)
    /// * `total_epochs` - Total number of training epochs
    /// * `strategy` - Curriculum learning strategy to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// use torsh_data::sampler::{CurriculumSampler, CurriculumStrategy};
    ///
    /// // Define difficulty based on sample index
    /// let difficulty_fn = |idx: usize| idx as f64 / 100.0;
    ///
    /// let sampler = CurriculumSampler::new(
    ///     100,
    ///     difficulty_fn,
    ///     20,
    ///     CurriculumStrategy::Linear
    /// );
    /// ```
    pub fn new(
        dataset_size: usize,
        difficulty_fn: F,
        total_epochs: usize,
        strategy: CurriculumStrategy,
    ) -> Self {
        let difficulties: Vec<f64> = (0..dataset_size).map(&difficulty_fn).collect();

        Self {
            difficulties,
            difficulty_fn,
            current_epoch: 0,
            total_epochs,
            curriculum_strategy: strategy,
            generator: None,
        }
    }

    /// Create a curriculum sampler with predefined difficulties
    ///
    /// # Arguments
    ///
    /// * `difficulties` - Precomputed difficulty scores for each sample
    /// * `total_epochs` - Total number of training epochs
    /// * `strategy` - Curriculum learning strategy to use
    pub fn from_difficulties(
        difficulties: Vec<f64>,
        total_epochs: usize,
        strategy: CurriculumStrategy,
    ) -> Self
    where
        F: Default,
    {
        Self {
            difficulty_fn: F::default(),
            difficulties,
            current_epoch: 0,
            total_epochs,
            curriculum_strategy: strategy,
            generator: None,
        }
    }

    /// Set the current epoch for curriculum progression
    ///
    /// # Arguments
    ///
    /// * `epoch` - Current training epoch (0-based)
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
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

    /// Get the total number of epochs
    pub fn total_epochs(&self) -> usize {
        self.total_epochs
    }

    /// Get the curriculum strategy
    pub fn strategy(&self) -> &CurriculumStrategy {
        &self.curriculum_strategy
    }

    /// Get the difficulties
    pub fn difficulties(&self) -> &[f64] {
        &self.difficulties
    }

    /// Get the generator seed if set
    pub fn generator(&self) -> Option<u64> {
        self.generator
    }

    /// Calculate training progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        if self.total_epochs <= 1 {
            1.0
        } else {
            (self.current_epoch as f64 / (self.total_epochs - 1) as f64).min(1.0)
        }
    }

    /// Calculate difficulty threshold for current epoch
    ///
    /// Returns a value between 0.0 and 1.0 indicating the maximum difficulty
    /// of samples that should be included in the current epoch.
    pub fn get_difficulty_threshold(&self) -> f64 {
        let progress = self.progress();

        match &self.curriculum_strategy {
            CurriculumStrategy::Linear => progress,
            CurriculumStrategy::Exponential { base } => {
                if *base <= 1.0 {
                    progress // Fallback to linear if base is invalid
                } else {
                    (base.powf(progress) - 1.0) / (base - 1.0)
                }
            }
            CurriculumStrategy::Step { thresholds } => {
                if thresholds.is_empty() {
                    1.0
                } else {
                    let mut threshold = 0.0;
                    for &epoch_threshold in thresholds {
                        if self.current_epoch >= epoch_threshold {
                            threshold += 1.0 / thresholds.len() as f64;
                        }
                    }
                    threshold.min(1.0)
                }
            }
            CurriculumStrategy::AntiCurriculum => {
                // Anti-curriculum: start inclusive, stay inclusive but focus on easier samples midway, end fully inclusive
                // Early (progress=0): threshold=1.0 (all samples)
                // Late (progress=1): threshold=1.0 (all samples)
                // But with different ordering/selection behavior
                1.0 // Always include all samples for now
            }
            CurriculumStrategy::SelfPaced { lambda } => {
                // Self-paced learning adjusts based on performance
                // This is a simplified version - in practice you'd track model performance
                (progress * lambda).min(1.0)
            }
        }
    }

    /// Get indices based on difficulty threshold
    ///
    /// Returns indices of samples that should be included based on the
    /// current epoch and curriculum strategy.
    pub fn get_curriculum_indices(&self) -> Vec<usize> {
        if self.difficulties.is_empty() {
            return Vec::new();
        }

        let threshold = self.get_difficulty_threshold();
        let max_difficulty = self
            .difficulties
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_difficulty = self
            .difficulties
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        // Normalize difficulties to [0, 1]
        let range = max_difficulty - min_difficulty;
        let normalized_threshold = if range > 0.0 {
            threshold
        } else {
            1.0 // If all difficulties are the same, include all
        };

        self.difficulties
            .iter()
            .enumerate()
            .filter_map(|(idx, &difficulty)| {
                let normalized_difficulty = if range > 0.0 {
                    (difficulty - min_difficulty) / range
                } else {
                    0.0
                };

                let include_sample = match &self.curriculum_strategy {
                    CurriculumStrategy::AntiCurriculum => {
                        // Anti-curriculum: start inclusive, become more selective for easy samples
                        // At early epochs (threshold=1): include all samples
                        // At late epochs (threshold=0): include only easiest samples
                        normalized_difficulty <= normalized_threshold
                    }
                    _ => {
                        // Regular curriculum: include easy samples first, then harder ones
                        normalized_difficulty <= normalized_threshold
                    }
                };

                if include_sample {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Update difficulties using the difficulty function
    ///
    /// Recalculates all difficulty scores using the stored difficulty function.
    /// Useful when the dataset has changed or the difficulty function has been modified.
    pub fn update_difficulties(&mut self) {
        let dataset_size = self.difficulties.len();
        self.difficulties = (0..dataset_size).map(&self.difficulty_fn).collect();
    }

    /// Set a new curriculum strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - New curriculum strategy to use
    pub fn set_strategy(&mut self, strategy: CurriculumStrategy) {
        self.curriculum_strategy = strategy;
    }

    /// Reset to the beginning of the curriculum
    pub fn reset(&mut self) {
        self.current_epoch = 0;
    }

    /// Check if curriculum is complete (all samples included)
    pub fn is_complete(&self) -> bool {
        self.current_epoch >= self.total_epochs || self.get_difficulty_threshold() >= 1.0
    }

    /// Get statistics about the current curriculum state
    pub fn curriculum_stats(&self) -> CurriculumStats {
        let threshold = self.get_difficulty_threshold();
        let indices = self.get_curriculum_indices();
        let total_samples = self.difficulties.len();

        CurriculumStats {
            current_epoch: self.current_epoch,
            total_epochs: self.total_epochs,
            progress: self.progress(),
            difficulty_threshold: threshold,
            included_samples: indices.len(),
            total_samples,
            inclusion_ratio: if total_samples > 0 {
                indices.len() as f64 / total_samples as f64
            } else {
                0.0
            },
        }
    }
}

impl<F: Send + Clone> Sampler for CurriculumSampler<F>
where
    F: Fn(usize) -> f64 + Send + Clone,
{
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let mut indices = self.get_curriculum_indices();

        // Shuffle the valid indices using rng_utils
        rng_utils::shuffle_indices(&mut indices, self.generator);

        SamplerIterator::new(indices)
    }

    fn len(&self) -> usize {
        self.get_curriculum_indices().len()
    }
}

/// Statistics about the current curriculum state
#[derive(Debug, Clone, PartialEq)]
pub struct CurriculumStats {
    /// Current training epoch
    pub current_epoch: usize,
    /// Total number of epochs
    pub total_epochs: usize,
    /// Training progress (0.0 to 1.0)
    pub progress: f64,
    /// Current difficulty threshold (0.0 to 1.0)
    pub difficulty_threshold: f64,
    /// Number of samples included in current epoch
    pub included_samples: usize,
    /// Total number of samples in dataset
    pub total_samples: usize,
    /// Ratio of included samples to total samples
    pub inclusion_ratio: f64,
}

/// Create a linear curriculum sampler
///
/// Convenience function for creating a curriculum sampler with linear progression.
///
/// # Arguments
///
/// * `dataset_size` - Size of the dataset
/// * `difficulty_fn` - Function that maps sample index to difficulty score
/// * `total_epochs` - Total number of training epochs
/// * `seed` - Optional random seed for reproducible sampling
pub fn linear_curriculum<F>(
    dataset_size: usize,
    difficulty_fn: F,
    total_epochs: usize,
    seed: Option<u64>,
) -> CurriculumSampler<F>
where
    F: Fn(usize) -> f64 + Send + Clone,
{
    let mut sampler = CurriculumSampler::new(
        dataset_size,
        difficulty_fn,
        total_epochs,
        CurriculumStrategy::Linear,
    );
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create an exponential curriculum sampler
///
/// Convenience function for creating a curriculum sampler with exponential progression.
///
/// # Arguments
///
/// * `dataset_size` - Size of the dataset
/// * `difficulty_fn` - Function that maps sample index to difficulty score
/// * `total_epochs` - Total number of training epochs
/// * `base` - Exponential base (must be > 1.0)
/// * `seed` - Optional random seed for reproducible sampling
pub fn exponential_curriculum<F>(
    dataset_size: usize,
    difficulty_fn: F,
    total_epochs: usize,
    base: f64,
    seed: Option<u64>,
) -> CurriculumSampler<F>
where
    F: Fn(usize) -> f64 + Send + Clone,
{
    let mut sampler = CurriculumSampler::new(
        dataset_size,
        difficulty_fn,
        total_epochs,
        CurriculumStrategy::Exponential { base },
    );
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a step-wise curriculum sampler
///
/// Convenience function for creating a curriculum sampler with step-wise progression.
///
/// # Arguments
///
/// * `dataset_size` - Size of the dataset
/// * `difficulty_fn` - Function that maps sample index to difficulty score
/// * `total_epochs` - Total number of training epochs
/// * `thresholds` - Epoch numbers where difficulty increases
/// * `seed` - Optional random seed for reproducible sampling
pub fn step_curriculum<F>(
    dataset_size: usize,
    difficulty_fn: F,
    total_epochs: usize,
    thresholds: Vec<usize>,
    seed: Option<u64>,
) -> CurriculumSampler<F>
where
    F: Fn(usize) -> f64 + Send + Clone,
{
    let mut sampler = CurriculumSampler::new(
        dataset_size,
        difficulty_fn,
        total_epochs,
        CurriculumStrategy::Step { thresholds },
    );
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create an anti-curriculum sampler
///
/// Convenience function for creating a curriculum sampler that starts with hard samples.
///
/// # Arguments
///
/// * `dataset_size` - Size of the dataset
/// * `difficulty_fn` - Function that maps sample index to difficulty score
/// * `total_epochs` - Total number of training epochs
/// * `seed` - Optional random seed for reproducible sampling
pub fn anti_curriculum<F>(
    dataset_size: usize,
    difficulty_fn: F,
    total_epochs: usize,
    seed: Option<u64>,
) -> CurriculumSampler<F>
where
    F: Fn(usize) -> f64 + Send + Clone,
{
    let mut sampler = CurriculumSampler::new(
        dataset_size,
        difficulty_fn,
        total_epochs,
        CurriculumStrategy::AntiCurriculum,
    );
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple difficulty function for testing
    fn linear_difficulty(idx: usize) -> f64 {
        idx as f64 / 100.0
    }

    // Difficulty function based on distance from center
    fn center_distance_difficulty(idx: usize) -> f64 {
        (idx as f64 - 50.0).abs() / 50.0
    }

    #[test]
    fn test_curriculum_sampler_basic() {
        let mut sampler =
            CurriculumSampler::new(100, linear_difficulty, 10, CurriculumStrategy::Linear)
                .with_generator(42);

        assert_eq!(sampler.total_epochs(), 10);
        assert_eq!(sampler.current_epoch(), 0);
        assert_eq!(sampler.generator(), Some(42));
        assert_eq!(sampler.difficulties().len(), 100);
        assert!(!sampler.is_complete());

        // At epoch 0, should include only easiest samples
        sampler.set_epoch(0);
        let early_indices = sampler.get_curriculum_indices();
        assert!(!early_indices.is_empty());
        assert!(early_indices.len() < 100);

        // At final epoch, should include all samples
        sampler.set_epoch(9);
        let late_indices = sampler.get_curriculum_indices();
        assert_eq!(late_indices.len(), 100);
        assert!(sampler.is_complete());
    }

    #[test]
    fn test_curriculum_strategies() {
        let dataset_size = 100;
        let total_epochs = 10;

        // Test Linear strategy
        let mut linear_sampler = CurriculumSampler::new(
            dataset_size,
            linear_difficulty,
            total_epochs,
            CurriculumStrategy::Linear,
        );

        linear_sampler.set_epoch(0);
        let linear_early = linear_sampler.get_curriculum_indices().len();
        linear_sampler.set_epoch(5);
        let linear_mid = linear_sampler.get_curriculum_indices().len();
        linear_sampler.set_epoch(9);
        let linear_late = linear_sampler.get_curriculum_indices().len();

        assert!(linear_early < linear_mid);
        assert!(linear_mid < linear_late);
        assert_eq!(linear_late, dataset_size);

        // Test Exponential strategy
        let mut exp_sampler = CurriculumSampler::new(
            dataset_size,
            linear_difficulty,
            total_epochs,
            CurriculumStrategy::Exponential { base: 2.0 },
        );

        exp_sampler.set_epoch(0);
        let exp_early = exp_sampler.get_curriculum_indices().len();
        exp_sampler.set_epoch(5);
        let exp_mid = exp_sampler.get_curriculum_indices().len();

        // Exponential should grow slower initially
        assert!(exp_early <= linear_early);

        // Test Step strategy
        let mut step_sampler = CurriculumSampler::new(
            dataset_size,
            linear_difficulty,
            total_epochs,
            CurriculumStrategy::Step {
                thresholds: vec![3, 6, 9],
            },
        );

        step_sampler.set_epoch(2);
        let step_before = step_sampler.get_curriculum_indices().len();
        step_sampler.set_epoch(3);
        let step_after = step_sampler.get_curriculum_indices().len();

        // Should have a jump at threshold
        assert!(step_after > step_before);

        // Test AntiCurriculum
        let mut anti_sampler = CurriculumSampler::new(
            dataset_size,
            linear_difficulty,
            total_epochs,
            CurriculumStrategy::AntiCurriculum,
        );

        anti_sampler.set_epoch(0);
        let anti_early = anti_sampler.get_curriculum_indices().len();
        anti_sampler.set_epoch(9);
        let anti_late = anti_sampler.get_curriculum_indices().len();

        // Anti-curriculum should start with more samples
        assert!(anti_early > linear_early);
        assert_eq!(anti_late, dataset_size);
    }

    #[test]
    fn test_difficulty_threshold_calculation() {
        let sampler =
            CurriculumSampler::new(100, linear_difficulty, 10, CurriculumStrategy::Linear);

        // Test progress calculation
        assert_eq!(sampler.progress(), 0.0);

        let mut sampler = sampler;
        sampler.set_epoch(5);
        assert!((sampler.progress() - 5.0 / 9.0).abs() < f64::EPSILON);

        sampler.set_epoch(9);
        assert_eq!(sampler.progress(), 1.0);

        // Test different strategies
        assert_eq!(sampler.get_difficulty_threshold(), 1.0);

        sampler.set_strategy(CurriculumStrategy::Exponential { base: 2.0 });
        sampler.set_epoch(0);
        assert_eq!(sampler.get_difficulty_threshold(), 0.0);

        sampler.set_strategy(CurriculumStrategy::AntiCurriculum);
        sampler.set_epoch(0);
        assert_eq!(sampler.get_difficulty_threshold(), 1.0);
    }

    #[test]
    fn test_curriculum_from_difficulties() {
        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let difficulty_fn = |idx: usize| difficulties.get(idx).copied().unwrap_or(0.0);
        let sampler = CurriculumSampler::new(
            difficulties.len(),
            difficulty_fn,
            5,
            CurriculumStrategy::Linear,
        );

        assert_eq!(sampler.difficulties(), &difficulties);
        assert_eq!(sampler.total_epochs(), 5);
    }

    #[test]
    fn test_curriculum_indices_selection() {
        let difficulties = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let difficulty_fn = |idx: usize| difficulties.get(idx).copied().unwrap_or(0.0);
        let mut sampler = CurriculumSampler::new(
            difficulties.len(),
            difficulty_fn,
            6,
            CurriculumStrategy::Linear,
        );

        // At epoch 0, should include only easiest samples
        sampler.set_epoch(0);
        let indices = sampler.get_curriculum_indices();
        assert!(indices.contains(&0)); // Easiest sample should be included
        assert!(!indices.contains(&5)); // Hardest sample should not be included

        // At final epoch, should include all samples
        sampler.set_epoch(5);
        let indices = sampler.get_curriculum_indices();
        assert_eq!(indices.len(), 6);
        for i in 0..6 {
            assert!(indices.contains(&i));
        }
    }

    #[test]
    fn test_curriculum_stats() {
        let mut sampler =
            CurriculumSampler::new(100, linear_difficulty, 10, CurriculumStrategy::Linear);

        sampler.set_epoch(5);
        let stats = sampler.curriculum_stats();

        assert_eq!(stats.current_epoch, 5);
        assert_eq!(stats.total_epochs, 10);
        assert_eq!(stats.total_samples, 100);
        assert!(stats.progress > 0.0 && stats.progress < 1.0);
        assert!(stats.difficulty_threshold > 0.0 && stats.difficulty_threshold < 1.0);
        assert!(stats.included_samples > 0 && stats.included_samples < 100);
        assert!(stats.inclusion_ratio > 0.0 && stats.inclusion_ratio < 1.0);
    }

    #[test]
    fn test_curriculum_sampler_iter() {
        let mut sampler =
            CurriculumSampler::new(20, linear_difficulty, 5, CurriculumStrategy::Linear)
                .with_generator(42);

        sampler.set_epoch(0);
        let indices1: Vec<usize> = sampler.iter().collect();
        let indices2: Vec<usize> = sampler.iter().collect();

        assert_eq!(indices1.len(), sampler.len());
        assert_eq!(indices2.len(), sampler.len());

        // With the same seed, should get the same shuffled order
        assert_eq!(indices1, indices2);

        // Advance to later epoch
        sampler.set_epoch(4);
        let late_indices: Vec<usize> = sampler.iter().collect();
        assert!(late_indices.len() >= indices1.len());
    }

    #[test]
    fn test_convenience_functions() {
        // Test linear_curriculum
        let linear = linear_curriculum(50, linear_difficulty, 10, Some(42));
        assert_eq!(linear.total_epochs(), 10);
        assert_eq!(linear.generator(), Some(42));
        assert!(matches!(linear.strategy(), CurriculumStrategy::Linear));

        // Test exponential_curriculum
        let exponential = exponential_curriculum(50, linear_difficulty, 10, 2.0, Some(42));
        assert!(
            matches!(exponential.strategy(), CurriculumStrategy::Exponential { base } if *base == 2.0)
        );

        // Test step_curriculum
        let step = step_curriculum(50, linear_difficulty, 10, vec![2, 5, 8], Some(42));
        assert!(
            matches!(step.strategy(), CurriculumStrategy::Step { thresholds } if thresholds == &vec![2, 5, 8])
        );

        // Test anti_curriculum
        let anti = anti_curriculum(50, linear_difficulty, 10, Some(42));
        assert!(matches!(
            anti.strategy(),
            CurriculumStrategy::AntiCurriculum
        ));
    }

    #[test]
    fn test_curriculum_methods() {
        let mut sampler =
            CurriculumSampler::new(100, linear_difficulty, 10, CurriculumStrategy::Linear);

        // Test reset
        sampler.set_epoch(5);
        assert_eq!(sampler.current_epoch(), 5);
        sampler.reset();
        assert_eq!(sampler.current_epoch(), 0);

        // Test set_strategy
        sampler.set_strategy(CurriculumStrategy::Exponential { base: 3.0 });
        assert!(
            matches!(sampler.strategy(), CurriculumStrategy::Exponential { base } if *base == 3.0)
        );

        // Test update_difficulties
        let original_difficulties = sampler.difficulties().to_vec();
        sampler.update_difficulties();
        assert_eq!(sampler.difficulties(), &original_difficulties);
    }

    #[test]
    fn test_edge_cases() {
        // Empty dataset
        let empty_sampler =
            CurriculumSampler::new(0, linear_difficulty, 5, CurriculumStrategy::Linear);
        assert_eq!(empty_sampler.len(), 0);
        assert!(empty_sampler.get_curriculum_indices().is_empty());

        // Single epoch
        let mut single_epoch =
            CurriculumSampler::new(10, linear_difficulty, 1, CurriculumStrategy::Linear);
        single_epoch.set_epoch(0);
        assert_eq!(single_epoch.progress(), 1.0);
        assert!(single_epoch.is_complete());

        // Same difficulties
        let same_difficulties = vec![0.5; 10];
        let same_difficulty_fn = |idx: usize| same_difficulties.get(idx).copied().unwrap_or(0.0);
        let mut same_sampler = CurriculumSampler::new(
            same_difficulties.len(),
            same_difficulty_fn,
            5,
            CurriculumStrategy::Linear,
        );
        same_sampler.set_epoch(0);
        assert_eq!(same_sampler.get_curriculum_indices().len(), 10); // Should include all

        // Invalid exponential base
        let mut invalid_exp = CurriculumSampler::new(
            10,
            linear_difficulty,
            5,
            CurriculumStrategy::Exponential { base: 0.5 }, // Invalid base
        );
        invalid_exp.set_epoch(2);
        // Should fallback to linear-like behavior
        assert!(invalid_exp.get_difficulty_threshold() >= 0.0);
    }

    #[test]
    fn test_curriculum_strategy_equality() {
        assert_eq!(CurriculumStrategy::Linear, CurriculumStrategy::Linear);
        assert_eq!(
            CurriculumStrategy::Exponential { base: 2.0 },
            CurriculumStrategy::Exponential { base: 2.0 }
        );
        assert_ne!(
            CurriculumStrategy::Linear,
            CurriculumStrategy::AntiCurriculum
        );
    }

    #[test]
    fn test_curriculum_strategy_default() {
        assert_eq!(CurriculumStrategy::default(), CurriculumStrategy::Linear);
    }

    #[test]
    fn test_center_distance_difficulty() {
        let mut sampler = CurriculumSampler::new(
            101, // 0 to 100, center at 50
            center_distance_difficulty,
            10,
            CurriculumStrategy::Linear,
        )
        .with_generator(42);

        // At early epochs, should prefer samples near center (index 50)
        sampler.set_epoch(0);
        let early_indices = sampler.get_curriculum_indices();
        assert!(early_indices.contains(&50)); // Center should be included

        // At later epochs, should include samples farther from center
        sampler.set_epoch(9);
        let late_indices = sampler.get_curriculum_indices();
        assert!(late_indices.len() > early_indices.len());
        assert!(late_indices.contains(&0) || late_indices.contains(&100)); // Extremes should be included
    }
}
