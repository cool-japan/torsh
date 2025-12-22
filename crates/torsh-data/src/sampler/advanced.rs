//! Advanced sampling strategies for specialized machine learning scenarios.
//!
//! This module provides sophisticated sampling techniques that go beyond basic
//! random and sequential sampling. These strategies are particularly useful for
//! handling imbalanced datasets, implementing importance sampling, and creating
//! structured sampling patterns for specific machine learning applications.
//!
//! # Key Components
//!
//! ## Weighted Sampling
//! - [`WeightedRandomSampler`] - Probability-weighted random sampling
//! - Support for unnormalized weights and automatic normalization
//! - Efficient implementation using alias method for O(1) sampling
//!
//! ## Grouped Sampling
//! - [`GroupedSampler`] - Groups samples by user-defined criteria
//! - Configurable shuffling within and between groups
//! - Useful for batch sampling with specific constraints
//!
//! ## Stratified Sampling
//! - [`StratifiedSampler`] - Maintains proportional representation across strata
//! - Automatic balancing and class-aware sampling
//! - Essential for classification with imbalanced datasets
//!
//! ## Importance Sampling
//! - [`ImportanceSampler`] - Samples based on importance scores
//! - Adaptive importance weight calculation
//! - Critical for active learning and hard negative mining
//!
//! # Examples
//!
//! ## Weighted Random Sampling
//! ```rust,ignore
//! use torsh_data::sampler::{Sampler, WeightedRandomSampler};
//!
//! // Sample with higher probability for larger weights
//! let weights = vec![0.1, 0.3, 0.2, 0.4];
//! let sampler = WeightedRandomSampler::new(weights, true)
//!     .with_generator(42);
//!
//! let indices: Vec<usize> = sampler.iter().take(10).collect();
//! // Index 3 (weight 0.4) will appear more frequently
//! ```
//!
//! ## Grouped Sampling
//! ```rust,ignore
//! use torsh_data::sampler::{Sampler, GroupedSampler};
//!
//! // Group samples by some criterion (e.g., class label)
//! let group_fn = |idx: usize| idx % 3; // 3 groups
//! # struct DummyDataset { len: usize }
//! # impl crate::dataset::Dataset for DummyDataset {
//! #     type Item = usize;
//! #     fn get(&self, index: usize) -> Option<Self::Item> { Some(index) }
//! #     fn len(&self) -> usize { self.len }
//! # }
//! let dataset = DummyDataset { len: 12 };
//!
//! let sampler = GroupedSampler::new(&dataset, group_fn)
//!     .with_shuffle_groups(true)
//!     .with_shuffle_within_groups(true);
//!
//! // Samples will be grouped together but in random order
//! ```
//!
//! ## Stratified Sampling
//! ```rust,ignore
//! use torsh_data::sampler::{Sampler, StratifiedSampler};
//!
//! // Ensure balanced representation across classes
//! let class_labels = vec![0, 0, 1, 1, 1, 2, 2, 2, 2];
//! let sampler = StratifiedSampler::new(class_labels)
//!     .with_proportional(true)
//!     .with_generator(123);
//!
//! // Each class will be represented proportionally
//! ```
//!
//! ## Importance Sampling
//! ```rust,ignore
//! use torsh_data::sampler::{Sampler, ImportanceSampler};
//!
//! // Sample based on importance scores (e.g., loss values)
//! let importance_scores = vec![0.1, 0.8, 0.3, 0.9, 0.2];
//! let sampler = ImportanceSampler::new(importance_scores)
//!     .with_temperature(2.0)  // Higher temp = more uniform
//!     .with_generator(456);
//!
//! // High-importance samples will be selected more frequently
//! ```

#[cfg(not(feature = "std"))]
use alloc::{collections::HashMap, vec, vec::Vec};
#[cfg(feature = "std")]
use std::collections::HashMap;

use super::core::{rng_utils, Sampler, SamplerIterator};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{Random, Rng};

/// Weighted random sampler for probability-based sampling.
///
/// This sampler allows you to specify different probabilities for each sample
/// in the dataset. Samples with higher weights are more likely to be selected.
/// This is essential for handling imbalanced datasets or implementing custom
/// sampling distributions.
///
/// # Implementation Details
///
/// The sampler uses the alias method for efficient O(1) sampling after O(n)
/// preprocessing. This makes it suitable for large datasets where you need
/// to draw many samples.
///
/// # Performance Characteristics
///
/// - **Preprocessing**: O(n) time and space to build alias table
/// - **Sampling**: O(1) per sample after preprocessing
/// - **Memory**: O(n) for alias table storage
/// - **Numerical Stability**: Handles unnormalized weights robustly
#[derive(Debug, Clone)]
pub struct WeightedRandomSampler {
    weights: Vec<f32>,
    replacement: bool,
    generator: Option<u64>,
    alias_table: Option<AliasTable>,
}

impl WeightedRandomSampler {
    /// Create a new weighted random sampler.
    ///
    /// # Arguments
    ///
    /// * `weights` - Vector of weights for each sample (will be normalized)
    /// * `replacement` - Whether to sample with replacement
    ///
    /// # Panics
    ///
    /// Panics if weights vector is empty or contains only zeros.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::{Sampler, WeightedRandomSampler};
    ///
    /// let weights = vec![1.0, 2.0, 3.0]; // Unnormalized weights
    /// let sampler = WeightedRandomSampler::new(weights, true);
    ///
    /// // Sample probabilities will be [1/6, 2/6, 3/6]
    /// let indices: Vec<usize> = sampler.iter().take(100).collect();
    /// // Index 2 should appear most frequently
    /// ```
    pub fn new(weights: Vec<f32>, replacement: bool) -> Self {
        assert!(!weights.is_empty(), "Weights vector cannot be empty");
        assert!(
            weights.iter().any(|&w| w > 0.0),
            "At least one weight must be positive"
        );

        Self {
            weights,
            replacement,
            generator: None,
            alias_table: None,
        }
    }

    /// Set random generator seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed for deterministic sampling
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::{Sampler, WeightedRandomSampler};
    ///
    /// let weights = vec![1.0, 2.0, 3.0];
    /// let sampler = WeightedRandomSampler::new(weights, true)
    ///     .with_generator(42);
    ///
    /// // Sampling will be deterministic
    /// ```
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Get the weights used by this sampler.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Check if sampling is done with replacement.
    pub fn uses_replacement(&self) -> bool {
        self.replacement
    }

    /// Get the generator seed if set.
    pub fn generator_seed(&self) -> Option<u64> {
        self.generator
    }

    /// Build the alias table for efficient sampling.
    fn build_alias_table(&mut self) {
        if self.alias_table.is_none() {
            self.alias_table = Some(AliasTable::new(&self.weights));
        }
    }

    /// Generate weighted random indices.
    fn generate_indices(&mut self, count: usize) -> Vec<usize> {
        self.build_alias_table();
        let alias_table = self.alias_table.as_ref().unwrap();

        let mut rng = rng_utils::create_rng(self.generator);
        let mut indices = Vec::with_capacity(count);

        for _ in 0..count {
            let idx = alias_table.sample(&mut rng);
            indices.push(idx);
        }

        indices
    }
}

impl Sampler for WeightedRandomSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let count = if self.replacement {
            self.weights.len() // With replacement, sample as many as we have
        } else {
            self.weights.len() // Without replacement, sample each once
        };

        let mut sampler = self.clone();
        let indices = if self.replacement {
            sampler.generate_indices(count)
        } else {
            // Without replacement: weighted shuffle
            let mut weighted_indices: Vec<(usize, f32)> = self
                .weights
                .iter()
                .enumerate()
                .map(|(i, &w)| (i, w))
                .collect();

            let mut rng = rng_utils::create_rng(self.generator);

            // Fisher-Yates shuffle with weights
            for i in (1..weighted_indices.len()).rev() {
                let total_weight: f32 = weighted_indices[..=i].iter().map(|(_, w)| w).sum();
                let mut target_weight = rng.random::<f32>() * total_weight;

                let mut selected_idx = 0;
                for (j, (_, weight)) in weighted_indices[..=i].iter().enumerate() {
                    target_weight -= weight;
                    if target_weight <= 0.0 {
                        selected_idx = j;
                        break;
                    }
                }

                weighted_indices.swap(i, selected_idx);
            }

            weighted_indices.into_iter().map(|(idx, _)| idx).collect()
        };

        SamplerIterator::new(indices)
    }

    fn len(&self) -> usize {
        self.weights.len()
    }
}

/// Efficient alias table implementation for O(1) weighted sampling.
///
/// The alias method allows constant-time sampling from a discrete probability
/// distribution by preprocessing the weights into a lookup table.
#[derive(Debug, Clone)]
struct AliasTable {
    prob: Vec<f32>,
    alias: Vec<usize>,
}

impl AliasTable {
    /// Build an alias table from unnormalized weights.
    fn new(weights: &[f32]) -> Self {
        let n = weights.len();
        let sum: f32 = weights.iter().sum();

        assert!(sum > 0.0, "Total weight must be positive");

        let mut prob = vec![0.0; n];
        let mut alias = vec![0; n];

        // Normalize weights to probabilities
        let normalized: Vec<f32> = weights.iter().map(|&w| w * n as f32 / sum).collect();

        // Separate into small and large probability buckets
        let mut small = Vec::new();
        let mut large = Vec::new();

        for (i, &p) in normalized.iter().enumerate() {
            if p < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        prob.copy_from_slice(&normalized);

        // Build alias table
        while let (Some(l), Some(g)) = (small.pop(), large.pop()) {
            alias[l] = g;
            prob[g] = prob[g] + prob[l] - 1.0;

            if prob[g] < 1.0 {
                small.push(g);
            } else {
                large.push(g);
            }
        }

        // Handle remaining large probabilities
        while let Some(g) = large.pop() {
            prob[g] = 1.0;
        }

        // Handle remaining small probabilities
        while let Some(l) = small.pop() {
            prob[l] = 1.0;
        }

        Self { prob, alias }
    }

    /// Sample an index using the alias table.
    fn sample(&self, rng: &mut Random<scirs2_core::rngs::StdRng>) -> usize {
        let i = rng.gen_range(0..self.prob.len());
        let coin_flip = rng.random::<f32>();

        if coin_flip < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }
}

/// Sampler that groups indices by a key function and samples groups together.
///
/// This sampler allows you to define custom grouping criteria and control
/// how samples within and between groups are ordered. This is useful for
/// scenarios where you want to process related samples together.
///
/// # Use Cases
///
/// - **Sequence Data**: Group by sequence ID to process complete sequences
/// - **Hierarchical Data**: Group by category for structured processing
/// - **Batch Constraints**: Ensure certain samples appear in the same batch
/// - **Memory Efficiency**: Group similar samples for better cache locality
#[derive(Debug)]
pub struct GroupedSampler<F> {
    groups: Vec<Vec<usize>>,
    shuffle_groups: bool,
    shuffle_within_groups: bool,
    generator: Option<u64>,
    _phantom: std::marker::PhantomData<F>,
}

impl<F> GroupedSampler<F>
where
    F: Fn(usize) -> usize + Send,
{
    /// Create a new grouped sampler.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset to sample from
    /// * `group_fn` - Function that maps sample index to group ID
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::{Sampler, GroupedSampler};
    ///
    /// # struct DummyDataset { len: usize }
    /// # impl crate::dataset::Dataset for DummyDataset {
    /// #     type Item = usize;
    /// #     fn get(&self, index: usize) -> Option<Self::Item> { Some(index) }
    /// #     fn len(&self) -> usize { self.len }
    /// # }
    /// let dataset = DummyDataset { len: 10 };
    ///
    /// // Group by class (assuming 3 classes)
    /// let group_by_class = |idx: usize| idx % 3;
    /// let sampler = GroupedSampler::new(&dataset, group_by_class);
    /// ```
    pub fn new<D>(dataset: &D, group_fn: F) -> Self
    where
        D: crate::dataset::Dataset,
    {
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();

        // Group indices by the group function
        for idx in 0..dataset.len() {
            let group_key = group_fn(idx);
            groups.entry(group_key).or_default().push(idx);
        }

        // Convert to sorted vector of groups
        let mut group_list: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
        group_list.sort_by_key(|(key, _)| *key);
        let groups: Vec<Vec<usize>> = group_list.into_iter().map(|(_, indices)| indices).collect();

        Self {
            groups,
            shuffle_groups: false,
            shuffle_within_groups: false,
            generator: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set whether to shuffle the order of groups.
    ///
    /// # Arguments
    ///
    /// * `shuffle` - Whether to randomize group order
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use torsh_data::sampler::GroupedSampler;
    /// # struct DummyDataset { len: usize }
    /// # impl crate::dataset::Dataset for DummyDataset {
    /// #     type Item = usize;
    /// #     fn get(&self, index: usize) -> Option<Self::Item> { Some(index) }
    /// #     fn len(&self) -> usize { self.len }
    /// # }
    /// let dataset = DummyDataset { len: 10 };
    /// let sampler = GroupedSampler::new(&dataset, |idx| idx % 3)
    ///     .with_shuffle_groups(true);
    /// ```
    pub fn with_shuffle_groups(mut self, shuffle: bool) -> Self {
        self.shuffle_groups = shuffle;
        self
    }

    /// Set whether to shuffle within each group.
    ///
    /// # Arguments
    ///
    /// * `shuffle` - Whether to randomize order within groups
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// # use torsh_data::sampler::GroupedSampler;
    /// # struct DummyDataset { len: usize }
    /// # impl crate::dataset::Dataset for DummyDataset {
    /// #     type Item = usize;
    /// #     fn get(&self, index: usize) -> Option<Self::Item> { Some(index) }
    /// #     fn len(&self) -> usize { self.len }
    /// # }
    /// let dataset = DummyDataset { len: 10 };
    /// let sampler = GroupedSampler::new(&dataset, |idx| idx % 3)
    ///     .with_shuffle_within_groups(true);
    /// ```
    pub fn with_shuffle_within_groups(mut self, shuffle: bool) -> Self {
        self.shuffle_within_groups = shuffle;
        self
    }

    /// Set random generator seed.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed for deterministic shuffling
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Get the number of groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Get the sizes of all groups.
    pub fn group_sizes(&self) -> Vec<usize> {
        self.groups.iter().map(|group| group.len()).collect()
    }

    /// Check if groups will be shuffled.
    pub fn shuffles_groups(&self) -> bool {
        self.shuffle_groups
    }

    /// Check if samples within groups will be shuffled.
    pub fn shuffles_within_groups(&self) -> bool {
        self.shuffle_within_groups
    }
}

impl<F: Send> Sampler for GroupedSampler<F> {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let mut rng = rng_utils::create_rng(self.generator);
        let mut groups = self.groups.clone();

        // Shuffle within groups if requested
        if self.shuffle_within_groups {
            for group in &mut groups {
                group.shuffle(&mut rng);
            }
        }

        // Shuffle the order of groups if requested
        if self.shuffle_groups {
            groups.shuffle(&mut rng);
        }

        // Flatten all groups into a single list of indices
        let indices: Vec<usize> = groups.into_iter().flatten().collect();

        SamplerIterator::new(indices)
    }

    fn len(&self) -> usize {
        self.groups.iter().map(|group| group.len()).sum()
    }
}

/// Stratified sampler for balanced representation across strata.
///
/// This sampler ensures that each stratum (class/category) is represented
/// proportionally in the sample. This is essential for classification tasks
/// with imbalanced datasets where you want to maintain class balance.
///
/// # Key Features
///
/// - **Proportional Sampling**: Maintains original class proportions
/// - **Balanced Sampling**: Equal samples per class (when specified)
/// - **Minimum Guarantees**: Ensures each class gets at least one sample
/// - **Reproducible**: Deterministic when seeded
#[derive(Debug, Clone)]
pub struct StratifiedSampler {
    strata: HashMap<usize, Vec<usize>>,
    proportional: bool,
    min_samples_per_stratum: usize,
    generator: Option<u64>,
}

impl StratifiedSampler {
    /// Create a new stratified sampler.
    ///
    /// # Arguments
    ///
    /// * `class_labels` - Vector mapping sample index to class/stratum
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::{Sampler, StratifiedSampler};
    ///
    /// let labels = vec![0, 0, 1, 1, 1, 2]; // 2 class 0, 3 class 1, 1 class 2
    /// let sampler = StratifiedSampler::new(labels);
    ///
    /// // Will maintain proportional representation
    /// ```
    pub fn new(class_labels: Vec<usize>) -> Self {
        let mut strata: HashMap<usize, Vec<usize>> = HashMap::new();

        // Group indices by class label
        for (idx, &class) in class_labels.iter().enumerate() {
            strata.entry(class).or_default().push(idx);
        }

        Self {
            strata,
            proportional: true,
            min_samples_per_stratum: 1,
            generator: None,
        }
    }

    /// Create stratified sampler from pre-grouped strata.
    ///
    /// # Arguments
    ///
    /// * `strata` - Map from stratum ID to vector of sample indices
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::collections::HashMap;
    /// use torsh_data::sampler::StratifiedSampler;
    ///
    /// let mut strata = HashMap::new();
    /// strata.insert(0, vec![0, 1, 2]);    // Stratum 0: indices 0, 1, 2
    /// strata.insert(1, vec![3, 4, 5, 6]); // Stratum 1: indices 3, 4, 5, 6
    ///
    /// let sampler = StratifiedSampler::from_strata(strata);
    /// ```
    pub fn from_strata(strata: HashMap<usize, Vec<usize>>) -> Self {
        Self {
            strata,
            proportional: true,
            min_samples_per_stratum: 1,
            generator: None,
        }
    }

    /// Set whether to maintain proportional representation.
    ///
    /// When true (default), the number of samples per stratum is proportional
    /// to the stratum size. When false, each stratum gets equal samples.
    ///
    /// # Arguments
    ///
    /// * `proportional` - Whether to use proportional sampling
    pub fn with_proportional(mut self, proportional: bool) -> Self {
        self.proportional = proportional;
        self
    }

    /// Set minimum samples per stratum.
    ///
    /// Ensures each stratum gets at least this many samples, even if
    /// proportional sampling would give it fewer.
    ///
    /// # Arguments
    ///
    /// * `min_samples` - Minimum samples per stratum
    pub fn with_min_samples_per_stratum(mut self, min_samples: usize) -> Self {
        self.min_samples_per_stratum = min_samples;
        self
    }

    /// Set random generator seed.
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Get the number of strata.
    pub fn num_strata(&self) -> usize {
        self.strata.len()
    }

    /// Get the size of each stratum.
    pub fn stratum_sizes(&self) -> HashMap<usize, usize> {
        self.strata.iter().map(|(&k, v)| (k, v.len())).collect()
    }

    /// Check if proportional sampling is enabled.
    pub fn uses_proportional(&self) -> bool {
        self.proportional
    }

    /// Calculate how many samples each stratum should contribute.
    fn calculate_stratum_samples(&self, total_samples: usize) -> HashMap<usize, usize> {
        let total_stratum_size: usize = self.strata.values().map(|v| v.len()).sum();
        let mut stratum_samples = HashMap::new();

        if self.proportional {
            // Proportional to stratum size
            for (&stratum_id, indices) in &self.strata {
                let proportional_samples = (indices.len() * total_samples) / total_stratum_size;
                let final_samples = proportional_samples.max(self.min_samples_per_stratum);
                stratum_samples.insert(stratum_id, final_samples);
            }
        } else {
            // Equal samples per stratum
            let samples_per_stratum = total_samples / self.strata.len();
            for &stratum_id in self.strata.keys() {
                stratum_samples.insert(
                    stratum_id,
                    samples_per_stratum.max(self.min_samples_per_stratum),
                );
            }
        }

        stratum_samples
    }
}

impl Sampler for StratifiedSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let total_samples: usize = self.strata.values().map(|v| v.len()).sum();
        let stratum_samples = self.calculate_stratum_samples(total_samples);

        let mut rng = rng_utils::create_rng(self.generator);
        let mut all_indices = Vec::new();

        // Sample from each stratum
        for (&stratum_id, indices) in &self.strata {
            let target_samples = stratum_samples[&stratum_id];
            let mut stratum_indices = indices.clone();
            stratum_indices.shuffle(&mut rng);

            // Take samples with replacement if needed
            if target_samples <= indices.len() {
                all_indices.extend(&stratum_indices[..target_samples]);
            } else {
                // Need sampling with replacement
                all_indices.extend(&stratum_indices);
                for _ in indices.len()..target_samples {
                    let idx = rng.gen_range(0..indices.len());
                    all_indices.push(indices[idx]);
                }
            }
        }

        // Final shuffle to mix strata
        all_indices.shuffle(&mut rng);

        SamplerIterator::new(all_indices)
    }

    fn len(&self) -> usize {
        let total_samples: usize = self.strata.values().map(|v| v.len()).sum();
        let stratum_samples = self.calculate_stratum_samples(total_samples);
        stratum_samples.values().sum()
    }
}

/// Importance sampler for adaptive sample selection.
///
/// This sampler selects samples based on importance scores, which can represent
/// various metrics like loss values, prediction confidence, or gradient norms.
/// High-importance samples are selected more frequently, making this ideal for
/// active learning and hard negative mining.
///
/// # Applications
///
/// - **Active Learning**: Sample uncertain or informative examples
/// - **Hard Negative Mining**: Focus on difficult examples
/// - **Curriculum Learning**: Gradually increase sample difficulty
/// - **Online Learning**: Adapt to changing data distributions
#[derive(Debug, Clone)]
pub struct ImportanceSampler {
    importance_scores: Vec<f32>,
    temperature: f32,
    generator: Option<u64>,
    adaptive: bool,
    update_rate: f32,
}

impl ImportanceSampler {
    /// Create a new importance sampler.
    ///
    /// # Arguments
    ///
    /// * `importance_scores` - Vector of importance values for each sample
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::{Sampler, ImportanceSampler};
    ///
    /// // Higher scores = more important
    /// let scores = vec![0.1, 0.8, 0.3, 0.9, 0.2];
    /// let sampler = ImportanceSampler::new(scores);
    ///
    /// // Samples 1 and 3 will be selected more frequently
    /// ```
    pub fn new(importance_scores: Vec<f32>) -> Self {
        assert!(
            !importance_scores.is_empty(),
            "Importance scores cannot be empty"
        );

        Self {
            importance_scores,
            temperature: 1.0,
            generator: None,
            adaptive: false,
            update_rate: 0.1,
        }
    }

    /// Set the temperature for importance sampling.
    ///
    /// Higher temperature makes sampling more uniform, lower temperature
    /// makes it more focused on high-importance samples.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature parameter (> 0.0)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::ImportanceSampler;
    ///
    /// let scores = vec![0.1, 0.8, 0.3];
    /// let sampler = ImportanceSampler::new(scores)
    ///     .with_temperature(2.0); // More uniform sampling
    /// ```
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive");
        self.temperature = temperature;
        self
    }

    /// Enable adaptive importance updates.
    ///
    /// When enabled, importance scores can be updated based on recent
    /// sampling feedback to adapt to changing data characteristics.
    ///
    /// # Arguments
    ///
    /// * `adaptive` - Whether to enable adaptive updates
    /// * `update_rate` - Rate of adaptation (0.0 to 1.0)
    pub fn with_adaptive(mut self, adaptive: bool, update_rate: f32) -> Self {
        assert!(
            update_rate >= 0.0 && update_rate <= 1.0,
            "Update rate must be in [0, 1]"
        );
        self.adaptive = adaptive;
        self.update_rate = update_rate;
        self
    }

    /// Set random generator seed.
    pub fn with_generator(mut self, seed: u64) -> Self {
        self.generator = Some(seed);
        self
    }

    /// Get the importance scores.
    pub fn importance_scores(&self) -> &[f32] {
        &self.importance_scores
    }

    /// Get the temperature parameter.
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Check if adaptive updates are enabled.
    pub fn is_adaptive(&self) -> bool {
        self.adaptive
    }

    /// Update importance scores (for adaptive sampling).
    ///
    /// # Arguments
    ///
    /// * `new_scores` - Updated importance scores
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::ImportanceSampler;
    ///
    /// let mut sampler = ImportanceSampler::new(vec![0.1, 0.5, 0.3])
    ///     .with_adaptive(true, 0.1);
    ///
    /// // Update based on new loss values
    /// let new_losses = vec![0.2, 0.8, 0.1];
    /// sampler.update_importance_scores(new_losses);
    /// ```
    pub fn update_importance_scores(&mut self, new_scores: Vec<f32>) {
        if self.adaptive && new_scores.len() == self.importance_scores.len() {
            for (old, &new) in self.importance_scores.iter_mut().zip(new_scores.iter()) {
                *old = (1.0 - self.update_rate) * *old + self.update_rate * new;
            }
        }
    }

    /// Convert importance scores to sampling probabilities.
    fn compute_probabilities(&self) -> Vec<f32> {
        // Apply temperature scaling
        let scaled_scores: Vec<f32> = self
            .importance_scores
            .iter()
            .map(|&score| (score / self.temperature).exp())
            .collect();

        // Normalize to probabilities
        let total: f32 = scaled_scores.iter().sum();
        if total > 0.0 {
            scaled_scores.iter().map(|&score| score / total).collect()
        } else {
            // Fallback to uniform if all scores are zero
            vec![1.0 / self.importance_scores.len() as f32; self.importance_scores.len()]
        }
    }
}

impl Sampler for ImportanceSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let probabilities = self.compute_probabilities();
        let mut weighted_sampler = WeightedRandomSampler::new(probabilities, false);

        if let Some(seed) = self.generator {
            weighted_sampler = weighted_sampler.with_generator(seed);
        }

        weighted_sampler.iter()
    }

    fn len(&self) -> usize {
        self.importance_scores.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock dataset for testing
    struct MockDataset {
        size: usize,
    }

    impl crate::dataset::Dataset for MockDataset {
        type Item = usize;

        fn get(&self, index: usize) -> torsh_core::error::Result<Self::Item> {
            if index < self.size {
                Ok(index)
            } else {
                Err(torsh_core::error::TorshError::IndexOutOfBounds {
                    index,
                    size: self.size,
                })
            }
        }

        fn len(&self) -> usize {
            self.size
        }
    }

    #[test]
    fn test_weighted_random_sampler() {
        let weights = vec![0.1, 0.3, 0.6]; // Unnormalized weights
        let sampler = WeightedRandomSampler::new(weights.clone(), true).with_generator(42);

        assert_eq!(sampler.len(), 3);
        assert_eq!(sampler.weights(), &weights);
        assert!(sampler.uses_replacement());
        assert_eq!(sampler.generator_seed(), Some(42));

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 3);
        assert!(indices.iter().all(|&i| i < 3));
    }

    #[test]
    fn test_weighted_sampler_deterministic() {
        let weights = vec![1.0, 2.0, 3.0];
        let sampler1 = WeightedRandomSampler::new(weights.clone(), true).with_generator(123);
        let sampler2 = WeightedRandomSampler::new(weights, true).with_generator(123);

        let indices1: Vec<usize> = sampler1.iter().collect();
        let indices2: Vec<usize> = sampler2.iter().collect();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_alias_table() {
        let weights = vec![1.0, 2.0, 3.0];
        let table = AliasTable::new(&weights);

        assert_eq!(table.prob.len(), 3);
        assert_eq!(table.alias.len(), 3);

        let mut rng = rng_utils::create_rng(Some(42));

        // Sample multiple times to check basic functionality
        let mut counts = vec![0; 3];
        for _ in 0..1000 {
            let sample = table.sample(&mut rng);
            assert!(sample < 3);
            counts[sample] += 1;
        }

        // Higher weights should have higher counts (approximately)
        assert!(counts[2] > counts[1]); // Weight 3 > Weight 2
        assert!(counts[1] > counts[0]); // Weight 2 > Weight 1
    }

    #[test]
    fn test_grouped_sampler() {
        let dataset = MockDataset { size: 12 };
        let group_fn = |idx: usize| idx % 3; // 3 groups

        let sampler = GroupedSampler::new(&dataset, group_fn)
            .with_shuffle_groups(false)
            .with_shuffle_within_groups(false);

        assert_eq!(sampler.len(), 12);
        assert_eq!(sampler.num_groups(), 3);
        assert_eq!(sampler.group_sizes(), vec![4, 4, 4]); // 12 / 3 = 4 each

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 12);

        // Without shuffling, should maintain group order
        // Group 0: [0, 3, 6, 9], Group 1: [1, 4, 7, 10], Group 2: [2, 5, 8, 11]
    }

    #[test]
    fn test_grouped_sampler_with_shuffling() {
        let dataset = MockDataset { size: 9 };
        let group_fn = |idx: usize| idx % 3;

        let sampler = GroupedSampler::new(&dataset, group_fn)
            .with_shuffle_groups(true)
            .with_shuffle_within_groups(true)
            .with_generator(42);

        let indices1: Vec<usize> = sampler.iter().collect();
        let indices2: Vec<usize> = sampler.iter().collect();

        // Should be deterministic with same seed
        assert_eq!(indices1, indices2);
        assert_eq!(indices1.len(), 9);

        // Should contain all original indices
        let mut sorted_indices = indices1;
        sorted_indices.sort();
        assert_eq!(sorted_indices, (0..9).collect::<Vec<_>>());
    }

    #[test]
    fn test_stratified_sampler() {
        let class_labels = vec![0, 0, 1, 1, 1, 2]; // 2 class 0, 3 class 1, 1 class 2
        let sampler = StratifiedSampler::new(class_labels)
            .with_proportional(true)
            .with_generator(42);

        assert_eq!(sampler.num_strata(), 3);
        assert!(sampler.uses_proportional());

        let stratum_sizes = sampler.stratum_sizes();
        assert_eq!(stratum_sizes[&0], 2);
        assert_eq!(stratum_sizes[&1], 3);
        assert_eq!(stratum_sizes[&2], 1);

        let indices: Vec<usize> = sampler.iter().collect();
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_stratified_sampler_balanced() {
        let class_labels = vec![0, 0, 1, 1, 1, 2];
        let sampler = StratifiedSampler::new(class_labels)
            .with_proportional(false) // Equal samples per stratum
            .with_min_samples_per_stratum(2)
            .with_generator(42);

        assert!(!sampler.uses_proportional());

        let indices: Vec<usize> = sampler.iter().collect();
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_stratified_sampler_from_strata() {
        let mut strata = HashMap::new();
        strata.insert(0, vec![0, 1]);
        strata.insert(1, vec![2, 3, 4]);
        strata.insert(2, vec![5]);

        let sampler = StratifiedSampler::from_strata(strata);
        assert_eq!(sampler.num_strata(), 3);

        let indices: Vec<usize> = sampler.iter().collect();
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_importance_sampler() {
        let scores = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let sampler = ImportanceSampler::new(scores.clone())
            .with_temperature(1.0)
            .with_generator(42);

        assert_eq!(sampler.len(), 5);
        assert_eq!(sampler.importance_scores(), &scores);
        assert_eq!(sampler.temperature(), 1.0);
        assert!(!sampler.is_adaptive());

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 5);
        assert!(indices.iter().all(|&i| i < 5));
    }

    #[test]
    fn test_importance_sampler_temperature() {
        let scores = vec![0.1, 1.0, 0.1]; // One very high score

        // Low temperature - should heavily favor high-importance sample
        let low_temp_sampler = ImportanceSampler::new(scores.clone())
            .with_temperature(0.1)
            .with_generator(42);

        // High temperature - should be more uniform
        let high_temp_sampler = ImportanceSampler::new(scores)
            .with_temperature(10.0)
            .with_generator(42);

        // Both should work without panicking
        let _low_indices: Vec<usize> = low_temp_sampler.iter().collect();
        let _high_indices: Vec<usize> = high_temp_sampler.iter().collect();
    }

    #[test]
    fn test_importance_sampler_adaptive() {
        let scores = vec![0.1, 0.5, 0.3];
        let mut sampler = ImportanceSampler::new(scores)
            .with_adaptive(true, 0.2)
            .with_generator(42);

        assert!(sampler.is_adaptive());

        let original_scores = sampler.importance_scores().to_vec();

        // Update scores
        let new_scores = vec![0.2, 0.8, 0.1];
        sampler.update_importance_scores(new_scores);

        let updated_scores = sampler.importance_scores().to_vec();
        assert_ne!(original_scores, updated_scores);

        // The updated scores should be a blend of old and new
        for i in 0..3 {
            assert!(updated_scores[i] != original_scores[i]);
        }
    }

    #[test]
    #[should_panic(expected = "Weights vector cannot be empty")]
    fn test_weighted_sampler_empty_weights() {
        WeightedRandomSampler::new(vec![], true);
    }

    #[test]
    #[should_panic(expected = "At least one weight must be positive")]
    fn test_weighted_sampler_zero_weights() {
        WeightedRandomSampler::new(vec![0.0, 0.0, 0.0], true);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_importance_sampler_zero_temperature() {
        let scores = vec![0.1, 0.2, 0.3];
        ImportanceSampler::new(scores).with_temperature(0.0);
    }

    #[test]
    #[should_panic(expected = "Importance scores cannot be empty")]
    fn test_importance_sampler_empty_scores() {
        ImportanceSampler::new(vec![]);
    }

    #[test]
    fn test_importance_sampler_probabilities() {
        let scores = vec![1.0, 2.0, 3.0];
        let sampler = ImportanceSampler::new(scores).with_temperature(1.0);

        let probabilities = sampler.compute_probabilities();
        assert_eq!(probabilities.len(), 3);

        // Probabilities should sum to 1 (approximately)
        let sum: f32 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher scores should have higher probabilities
        assert!(probabilities[2] > probabilities[1]);
        assert!(probabilities[1] > probabilities[0]);
    }
}
