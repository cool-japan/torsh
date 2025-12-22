//! Active learning sampling functionality
//!
//! This module provides active learning samplers that prioritize uncertain or informative
//! samples to maximize learning efficiency with minimal labeling effort.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use std::collections::HashSet;

// ✅ SciRS2 Policy Compliant - Using scirs2_core for all random operations
use scirs2_core::rand_prelude::SliceRandom;

use super::core::{rng_utils, Sampler, SamplerIterator};

/// Active learning acquisition strategies
///
/// These strategies determine how samples are selected for labeling based on
/// different criteria such as uncertainty, information gain, or diversity.
#[derive(Clone, Debug, PartialEq)]
pub enum AcquisitionStrategy {
    /// Select samples with highest uncertainty
    ///
    /// Selects samples where the model is most uncertain about predictions.
    /// This is the most common active learning strategy.
    UncertaintySampling,

    /// Select samples that maximize expected information gain
    ///
    /// Chooses samples that are expected to provide the most information
    /// about the underlying data distribution.
    ExpectedInformationGain,

    /// Select diverse samples using clustering
    ///
    /// Ensures diversity in selected samples by partitioning into clusters
    /// and sampling from each cluster.
    ///
    /// # Arguments
    ///
    /// * `num_clusters` - Number of clusters to create for diversity
    DiversitySampling { num_clusters: usize },

    /// Combine uncertainty and diversity
    ///
    /// Balances between uncertain samples and diverse samples using a
    /// weighted combination approach.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Weight for uncertainty vs diversity (0.0-1.0)
    ///   - 1.0 = pure uncertainty sampling
    ///   - 0.0 = pure diversity sampling
    UncertaintyDiversity { alpha: f64 },

    /// Query by committee (variance across models)
    ///
    /// Selects samples where multiple models disagree the most.
    /// Requires ensemble predictions or committee of models.
    QueryByCommittee,

    /// Expected model change
    ///
    /// Selects samples that are expected to cause the largest change
    /// in model parameters when added to training set.
    ExpectedModelChange,
}

impl Default for AcquisitionStrategy {
    fn default() -> Self {
        AcquisitionStrategy::UncertaintySampling
    }
}

/// Active learning sampler that prioritizes uncertain or informative samples
///
/// This sampler selects samples based on uncertainty estimates or other
/// information-theoretic criteria to maximize learning efficiency with
/// minimal labeling effort.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::sampler::{ActiveLearningSampler, AcquisitionStrategy, Sampler};
///
/// let mut sampler = ActiveLearningSampler::new(
///     1000,
///     AcquisitionStrategy::UncertaintySampling,
///     10
/// ).with_generator(42);
///
/// // Update with uncertainty scores from model
/// let uncertainties = vec![0.5; 1000]; // Mock uncertainties
/// sampler.update_uncertainties(uncertainties);
///
/// // Get samples to label
/// let indices: Vec<usize> = sampler.iter().collect();
/// assert_eq!(indices.len(), 10);
///
/// // Add labeled samples
/// sampler.add_labeled_samples(&indices);
/// ```
#[derive(Clone)]
pub struct ActiveLearningSampler {
    uncertainties: Vec<f64>,
    acquisition_strategy: AcquisitionStrategy,
    num_samples: usize,
    budget_per_round: usize,
    current_round: usize,
    labeled_indices: HashSet<usize>,
    generator: Option<u64>,
}

impl ActiveLearningSampler {
    /// Create a new active learning sampler
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Total size of the dataset
    /// * `acquisition_strategy` - Strategy for selecting samples
    /// * `budget_per_round` - Number of samples to select per round
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::sampler::{ActiveLearningSampler, AcquisitionStrategy};
    ///
    /// let sampler = ActiveLearningSampler::new(
    ///     1000,
    ///     AcquisitionStrategy::UncertaintySampling,
    ///     20
    /// );
    /// ```
    pub fn new(
        dataset_size: usize,
        acquisition_strategy: AcquisitionStrategy,
        budget_per_round: usize,
    ) -> Self {
        Self {
            uncertainties: vec![0.0; dataset_size],
            acquisition_strategy,
            num_samples: dataset_size,
            budget_per_round,
            current_round: 0,
            labeled_indices: HashSet::new(),
            generator: None,
        }
    }

    /// Create an active learning sampler with initial labeled samples
    ///
    /// # Arguments
    ///
    /// * `dataset_size` - Total size of the dataset
    /// * `acquisition_strategy` - Strategy for selecting samples
    /// * `budget_per_round` - Number of samples to select per round
    /// * `initial_labeled` - Initially labeled sample indices
    pub fn with_initial_labeled(
        dataset_size: usize,
        acquisition_strategy: AcquisitionStrategy,
        budget_per_round: usize,
        initial_labeled: &[usize],
    ) -> Self {
        let mut sampler = Self::new(dataset_size, acquisition_strategy, budget_per_round);
        for &idx in initial_labeled {
            sampler.labeled_indices.insert(idx);
        }
        sampler
    }

    /// Update uncertainty scores for all samples
    ///
    /// # Arguments
    ///
    /// * `uncertainties` - Uncertainty scores for each sample (higher = more uncertain)
    ///
    /// # Panics
    ///
    /// Panics if the length of uncertainties doesn't match the dataset size.
    pub fn update_uncertainties(&mut self, uncertainties: Vec<f64>) {
        assert!(uncertainties.len() == self.num_samples, "assertion failed");
        self.uncertainties = uncertainties;
    }

    /// Add newly labeled samples
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of newly labeled samples
    pub fn add_labeled_samples(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.num_samples {
                self.labeled_indices.insert(idx);
            }
        }
        self.current_round += 1;
    }

    /// Remove samples from labeled set (useful for experimental scenarios)
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices to remove from labeled set
    pub fn remove_labeled_samples(&mut self, indices: &[usize]) {
        for &idx in indices {
            self.labeled_indices.remove(&idx);
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

    /// Get the current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get the budget per round
    pub fn budget_per_round(&self) -> usize {
        self.budget_per_round
    }

    /// Get the acquisition strategy
    pub fn strategy(&self) -> &AcquisitionStrategy {
        &self.acquisition_strategy
    }

    /// Get the number of labeled samples
    pub fn num_labeled(&self) -> usize {
        self.labeled_indices.len()
    }

    /// Get the number of unlabeled samples
    pub fn num_unlabeled(&self) -> usize {
        self.num_samples - self.labeled_indices.len()
    }

    /// Get the labeled sample indices
    pub fn labeled_indices(&self) -> Vec<usize> {
        self.labeled_indices.iter().copied().collect()
    }

    /// Get unlabeled sample indices
    pub fn get_unlabeled_indices(&self) -> Vec<usize> {
        (0..self.num_samples)
            .filter(|idx| !self.labeled_indices.contains(idx))
            .collect()
    }

    /// Check if a sample is labeled
    pub fn is_labeled(&self, index: usize) -> bool {
        self.labeled_indices.contains(&index)
    }

    /// Set a new acquisition strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - New acquisition strategy to use
    pub fn set_strategy(&mut self, strategy: AcquisitionStrategy) {
        self.acquisition_strategy = strategy;
    }

    /// Set a new budget per round
    ///
    /// # Arguments
    ///
    /// * `budget` - New budget per round
    pub fn set_budget(&mut self, budget: usize) {
        self.budget_per_round = budget;
    }

    /// Reset the sampler to initial state
    pub fn reset(&mut self) {
        self.labeled_indices.clear();
        self.current_round = 0;
    }

    /// Get statistics about the current active learning state
    pub fn active_learning_stats(&self) -> ActiveLearningStats {
        let unlabeled_count = self.num_unlabeled();
        let available_budget = self.budget_per_round.min(unlabeled_count);

        ActiveLearningStats {
            current_round: self.current_round,
            num_labeled: self.num_labeled(),
            num_unlabeled: unlabeled_count,
            total_samples: self.num_samples,
            budget_per_round: self.budget_per_round,
            available_budget,
            labeling_ratio: self.num_labeled() as f64 / self.num_samples as f64,
        }
    }

    /// Select samples based on acquisition strategy
    fn select_samples(&self) -> Vec<usize> {
        let unlabeled = self.get_unlabeled_indices();
        let budget = self.budget_per_round.min(unlabeled.len());

        if budget == 0 {
            return Vec::new();
        }

        match &self.acquisition_strategy {
            AcquisitionStrategy::UncertaintySampling => {
                self.uncertainty_sampling(&unlabeled, budget)
            }
            AcquisitionStrategy::ExpectedInformationGain => {
                self.information_gain_sampling(&unlabeled, budget)
            }
            AcquisitionStrategy::DiversitySampling { num_clusters } => {
                self.diversity_sampling(&unlabeled, budget, *num_clusters)
            }
            AcquisitionStrategy::UncertaintyDiversity { alpha } => {
                self.uncertainty_diversity_sampling(&unlabeled, budget, *alpha)
            }
            AcquisitionStrategy::QueryByCommittee => self.query_by_committee(&unlabeled, budget),
            AcquisitionStrategy::ExpectedModelChange => {
                self.expected_model_change(&unlabeled, budget)
            }
        }
    }

    /// Uncertainty sampling: select most uncertain samples
    fn uncertainty_sampling(&self, unlabeled: &[usize], budget: usize) -> Vec<usize> {
        let mut scored: Vec<_> = unlabeled
            .iter()
            .map(|&idx| (idx, self.uncertainties[idx]))
            .collect();

        // Sort by uncertainty (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(budget)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Information gain sampling (simplified version)
    fn information_gain_sampling(&self, unlabeled: &[usize], budget: usize) -> Vec<usize> {
        // This is a simplified implementation
        // In practice, you'd calculate expected information gain more rigorously
        let mut scored: Vec<_> = unlabeled
            .iter()
            .map(|&idx| {
                let ig = self.uncertainties[idx] * (1.0 + (idx as f64).ln());
                (idx, ig)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(budget)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Diversity sampling using simple clustering
    fn diversity_sampling(
        &self,
        unlabeled: &[usize],
        budget: usize,
        num_clusters: usize,
    ) -> Vec<usize> {
        if num_clusters == 0 {
            return self.uncertainty_sampling(unlabeled, budget);
        }

        // ✅ SciRS2 Policy Compliant - Using scirs2_core for random operations
        let mut rng = rng_utils::create_rng(self.generator);

        // Simplified diversity sampling: randomly partition into clusters and sample from each
        let mut indices = unlabeled.to_vec();
        indices.shuffle(&mut rng);

        let cluster_size = (unlabeled.len() / num_clusters).max(1);
        let base_samples_per_cluster = budget / num_clusters;
        let extra_samples = budget % num_clusters;

        let mut selected = Vec::new();
        let mut cluster_idx = 0;

        for cluster_start in (0..indices.len()).step_by(cluster_size) {
            let cluster_end = (cluster_start + cluster_size).min(indices.len());
            let cluster = &indices[cluster_start..cluster_end];

            // Calculate samples for this cluster (distribute remainder to first clusters)
            let cluster_samples_count = if cluster_idx < extra_samples {
                base_samples_per_cluster + 1
            } else {
                base_samples_per_cluster
            };

            if cluster_samples_count == 0 {
                cluster_idx += 1;
                continue;
            }

            // Sample from this cluster based on uncertainty
            let mut cluster_scored: Vec<_> = cluster
                .iter()
                .map(|&idx| (idx, self.uncertainties[idx]))
                .collect();

            cluster_scored
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let cluster_samples = cluster_scored
                .into_iter()
                .take(cluster_samples_count)
                .map(|(idx, _)| idx);

            selected.extend(cluster_samples);
            cluster_idx += 1;

            if selected.len() >= budget {
                break;
            }
        }

        selected.truncate(budget);
        selected
    }

    /// Combine uncertainty and diversity
    fn uncertainty_diversity_sampling(
        &self,
        unlabeled: &[usize],
        budget: usize,
        alpha: f64,
    ) -> Vec<usize> {
        let alpha = alpha.clamp(0.0, 1.0);

        // Simplified combined approach
        let uncertainty_count = (budget as f64 * alpha) as usize;
        let diversity_count = budget - uncertainty_count;

        let mut selected = self.uncertainty_sampling(unlabeled, uncertainty_count);

        if diversity_count > 0 {
            // Remove already selected from unlabeled for diversity sampling
            let remaining: Vec<usize> = unlabeled
                .iter()
                .filter(|idx| !selected.contains(idx))
                .copied()
                .collect();

            let diversity_samples = self.diversity_sampling(&remaining, diversity_count, 3);
            selected.extend(diversity_samples);
        }

        selected
    }

    /// Query by committee (simplified)
    fn query_by_committee(&self, unlabeled: &[usize], budget: usize) -> Vec<usize> {
        // In practice, you'd have multiple models and compute variance
        // For now, use uncertainty as a proxy for committee disagreement
        self.uncertainty_sampling(unlabeled, budget)
    }

    /// Expected model change (simplified)
    fn expected_model_change(&self, unlabeled: &[usize], budget: usize) -> Vec<usize> {
        // Simplified: assume uncertainty correlates with model change
        let mut scored: Vec<_> = unlabeled
            .iter()
            .map(|&idx| {
                let change_score =
                    self.uncertainties[idx] * (1.0 + idx as f64 / unlabeled.len() as f64);
                (idx, change_score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(budget)
            .map(|(idx, _)| idx)
            .collect()
    }
}

impl Sampler for ActiveLearningSampler {
    type Iter = SamplerIterator;

    fn iter(&self) -> Self::Iter {
        let indices = self.select_samples();
        SamplerIterator::new(indices)
    }

    fn len(&self) -> usize {
        let unlabeled_count = self.get_unlabeled_indices().len();
        self.budget_per_round.min(unlabeled_count)
    }
}

/// Statistics about the current active learning state
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveLearningStats {
    /// Current active learning round
    pub current_round: usize,
    /// Number of labeled samples
    pub num_labeled: usize,
    /// Number of unlabeled samples
    pub num_unlabeled: usize,
    /// Total number of samples
    pub total_samples: usize,
    /// Budget per round
    pub budget_per_round: usize,
    /// Available budget for current round
    pub available_budget: usize,
    /// Ratio of labeled to total samples
    pub labeling_ratio: f64,
}

/// Create an uncertainty sampling active learner
///
/// Convenience function for creating an active learning sampler with uncertainty sampling.
///
/// # Arguments
///
/// * `dataset_size` - Total size of the dataset
/// * `budget_per_round` - Number of samples to select per round
/// * `seed` - Optional random seed for reproducible sampling
pub fn uncertainty_sampler(
    dataset_size: usize,
    budget_per_round: usize,
    seed: Option<u64>,
) -> ActiveLearningSampler {
    let mut sampler = ActiveLearningSampler::new(
        dataset_size,
        AcquisitionStrategy::UncertaintySampling,
        budget_per_round,
    );
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a diversity sampling active learner
///
/// Convenience function for creating an active learning sampler with diversity sampling.
///
/// # Arguments
///
/// * `dataset_size` - Total size of the dataset
/// * `budget_per_round` - Number of samples to select per round
/// * `num_clusters` - Number of clusters for diversity
/// * `seed` - Optional random seed for reproducible sampling
pub fn diversity_sampler(
    dataset_size: usize,
    budget_per_round: usize,
    num_clusters: usize,
    seed: Option<u64>,
) -> ActiveLearningSampler {
    let mut sampler = ActiveLearningSampler::new(
        dataset_size,
        AcquisitionStrategy::DiversitySampling { num_clusters },
        budget_per_round,
    );
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

/// Create a combined uncertainty-diversity active learner
///
/// Convenience function for creating an active learning sampler that combines
/// uncertainty and diversity sampling.
///
/// # Arguments
///
/// * `dataset_size` - Total size of the dataset
/// * `budget_per_round` - Number of samples to select per round
/// * `alpha` - Weight for uncertainty vs diversity (0.0-1.0)
/// * `seed` - Optional random seed for reproducible sampling
pub fn uncertainty_diversity_sampler(
    dataset_size: usize,
    budget_per_round: usize,
    alpha: f64,
    seed: Option<u64>,
) -> ActiveLearningSampler {
    let mut sampler = ActiveLearningSampler::new(
        dataset_size,
        AcquisitionStrategy::UncertaintyDiversity { alpha },
        budget_per_round,
    );
    if let Some(s) = seed {
        sampler = sampler.with_generator(s);
    }
    sampler
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_learning_sampler_basic() {
        let mut sampler =
            ActiveLearningSampler::new(100, AcquisitionStrategy::UncertaintySampling, 10)
                .with_generator(42);

        assert_eq!(sampler.num_samples, 100);
        assert_eq!(sampler.budget_per_round(), 10);
        assert_eq!(sampler.current_round(), 0);
        assert_eq!(sampler.num_labeled(), 0);
        assert_eq!(sampler.num_unlabeled(), 100);
        assert_eq!(sampler.generator, Some(42));

        // Update uncertainties
        let uncertainties: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        sampler.update_uncertainties(uncertainties);

        // Should select highest uncertainty samples
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 10);

        // With uncertainty sampling, should select samples with highest uncertainty (90-99)
        for &idx in &indices {
            assert!(idx >= 90); // Highest uncertainty samples
        }

        // Add labeled samples
        sampler.add_labeled_samples(&indices);
        assert_eq!(sampler.num_labeled(), 10);
        assert_eq!(sampler.num_unlabeled(), 90);
        assert_eq!(sampler.current_round(), 1);
    }

    #[test]
    fn test_acquisition_strategies() {
        let dataset_size = 50;
        let budget = 5;

        // Test different strategies
        let strategies = vec![
            AcquisitionStrategy::UncertaintySampling,
            AcquisitionStrategy::ExpectedInformationGain,
            AcquisitionStrategy::DiversitySampling { num_clusters: 3 },
            AcquisitionStrategy::UncertaintyDiversity { alpha: 0.5 },
            AcquisitionStrategy::QueryByCommittee,
            AcquisitionStrategy::ExpectedModelChange,
        ];

        for strategy in strategies {
            let mut sampler = ActiveLearningSampler::new(dataset_size, strategy.clone(), budget)
                .with_generator(42);

            // Set up uncertainties with clear pattern
            let uncertainties: Vec<f64> = (0..dataset_size)
                .map(|i| if i < 10 { 0.9 } else { 0.1 })
                .collect();
            sampler.update_uncertainties(uncertainties);

            let indices: Vec<usize> = sampler.iter().collect();
            assert_eq!(indices.len(), budget);

            // All strategies should prefer high uncertainty samples to some degree
            // (except pure diversity which might sample differently)
            match strategy {
                AcquisitionStrategy::DiversitySampling { .. } => {
                    // Diversity sampling might select from anywhere
                    assert!(indices.iter().all(|&idx| idx < dataset_size));
                }
                _ => {
                    // Other strategies should prefer high uncertainty (indices 0-9)
                    let high_uncertainty_count = indices.iter().filter(|&&idx| idx < 10).count();
                    assert!(high_uncertainty_count > 0);
                }
            }
        }
    }

    #[test]
    fn test_active_learning_with_initial_labeled() {
        let initial_labeled = vec![0, 1, 2, 3, 4];
        let mut sampler = ActiveLearningSampler::with_initial_labeled(
            100,
            AcquisitionStrategy::UncertaintySampling,
            5,
            &initial_labeled,
        );

        assert_eq!(sampler.num_labeled(), 5);
        assert_eq!(sampler.num_unlabeled(), 95);

        for &idx in &initial_labeled {
            assert!(sampler.is_labeled(idx));
        }

        // Update uncertainties
        let uncertainties = vec![0.5; 100];
        sampler.update_uncertainties(uncertainties);

        // Should not select already labeled samples
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 5);

        for &idx in &indices {
            assert!(!initial_labeled.contains(&idx));
        }
    }

    #[test]
    fn test_uncertainty_diversity_sampling() {
        let mut sampler = ActiveLearningSampler::new(
            20,
            AcquisitionStrategy::UncertaintyDiversity { alpha: 0.6 },
            10,
        )
        .with_generator(42);

        // Set up uncertainties with clear pattern
        let uncertainties: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        sampler.update_uncertainties(uncertainties);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 10);

        // With alpha=0.6, should get 6 uncertainty samples + 4 diversity samples
        // The exact composition depends on implementation details, but should include mix
        let high_uncertainty_count = indices.iter().filter(|&&idx| idx >= 15).count();
        assert!(high_uncertainty_count > 0); // Should have some high uncertainty
        assert!(high_uncertainty_count < indices.len()); // But not all
    }

    #[test]
    fn test_diversity_sampling() {
        let mut sampler = ActiveLearningSampler::new(
            30,
            AcquisitionStrategy::DiversitySampling { num_clusters: 3 },
            9,
        )
        .with_generator(42);

        let uncertainties = vec![0.5; 30]; // Equal uncertainties
        sampler.update_uncertainties(uncertainties);

        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices.len(), 9);

        // With equal uncertainties, diversity sampling should spread across the range
        // (exact behavior depends on clustering implementation)
        assert!(indices.iter().all(|&idx| idx < 30));
    }

    #[test]
    fn test_active_learning_stats() {
        let mut sampler =
            ActiveLearningSampler::new(100, AcquisitionStrategy::UncertaintySampling, 15);

        let stats = sampler.active_learning_stats();
        assert_eq!(stats.current_round, 0);
        assert_eq!(stats.num_labeled, 0);
        assert_eq!(stats.num_unlabeled, 100);
        assert_eq!(stats.total_samples, 100);
        assert_eq!(stats.budget_per_round, 15);
        assert_eq!(stats.available_budget, 15);
        assert_eq!(stats.labeling_ratio, 0.0);

        // Add some labeled samples
        sampler.add_labeled_samples(&[0, 1, 2, 3, 4]);

        let stats = sampler.active_learning_stats();
        assert_eq!(stats.current_round, 1);
        assert_eq!(stats.num_labeled, 5);
        assert_eq!(stats.num_unlabeled, 95);
        assert_eq!(stats.labeling_ratio, 0.05);
    }

    #[test]
    fn test_sampler_methods() {
        let mut sampler =
            ActiveLearningSampler::new(50, AcquisitionStrategy::UncertaintySampling, 10);

        // Test labeled/unlabeled tracking
        assert_eq!(sampler.get_unlabeled_indices().len(), 50);
        assert!(sampler.labeled_indices().is_empty());

        sampler.add_labeled_samples(&[5, 15, 25]);
        assert_eq!(sampler.num_labeled(), 3);
        assert!(sampler.is_labeled(5));
        assert!(sampler.is_labeled(15));
        assert!(sampler.is_labeled(25));
        assert!(!sampler.is_labeled(0));

        let labeled = sampler.labeled_indices();
        assert_eq!(labeled.len(), 3);
        assert!(labeled.contains(&5));
        assert!(labeled.contains(&15));
        assert!(labeled.contains(&25));

        // Test remove labeled samples
        sampler.remove_labeled_samples(&[15]);
        assert_eq!(sampler.num_labeled(), 2);
        assert!(!sampler.is_labeled(15));

        // Test strategy change
        sampler.set_strategy(AcquisitionStrategy::DiversitySampling { num_clusters: 4 });
        assert!(matches!(
            sampler.strategy(),
            AcquisitionStrategy::DiversitySampling { num_clusters: 4 }
        ));

        // Test budget change
        sampler.set_budget(5);
        assert_eq!(sampler.budget_per_round(), 5);

        // Test reset
        sampler.reset();
        assert_eq!(sampler.num_labeled(), 0);
        assert_eq!(sampler.current_round(), 0);
    }

    #[test]
    fn test_convenience_functions() {
        // Test uncertainty_sampler
        let uncertainty = uncertainty_sampler(100, 10, Some(42));
        assert!(matches!(
            uncertainty.strategy(),
            AcquisitionStrategy::UncertaintySampling
        ));
        assert_eq!(uncertainty.budget_per_round(), 10);

        // Test diversity_sampler
        let diversity = diversity_sampler(100, 10, 5, Some(42));
        assert!(matches!(
            diversity.strategy(),
            AcquisitionStrategy::DiversitySampling { num_clusters: 5 }
        ));

        // Test uncertainty_diversity_sampler
        let combined = uncertainty_diversity_sampler(100, 10, 0.7, Some(42));
        assert!(matches!(
            combined.strategy(),
            AcquisitionStrategy::UncertaintyDiversity { alpha } if (*alpha - 0.7).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn test_edge_cases() {
        // Empty budget
        let mut sampler =
            ActiveLearningSampler::new(10, AcquisitionStrategy::UncertaintySampling, 0);
        assert_eq!(sampler.len(), 0);
        let indices: Vec<usize> = sampler.iter().collect();
        assert!(indices.is_empty());

        // All samples labeled
        sampler.set_budget(5);
        sampler.add_labeled_samples(&(0..10).collect::<Vec<_>>());
        assert_eq!(sampler.num_unlabeled(), 0);
        assert_eq!(sampler.len(), 0);

        // Budget larger than unlabeled
        let large_budget =
            ActiveLearningSampler::new(5, AcquisitionStrategy::UncertaintySampling, 10);
        assert_eq!(large_budget.len(), 5); // Should be clamped to available

        // Invalid alpha should be clamped
        let mut clamped = ActiveLearningSampler::new(
            10,
            AcquisitionStrategy::UncertaintyDiversity { alpha: 1.5 },
            5,
        );
        let uncertainties = vec![0.5; 10];
        clamped.update_uncertainties(uncertainties);
        let indices: Vec<usize> = clamped.iter().collect();
        assert_eq!(indices.len(), 5); // Should still work

        // Zero clusters in diversity sampling
        let mut zero_clusters = ActiveLearningSampler::new(
            10,
            AcquisitionStrategy::DiversitySampling { num_clusters: 0 },
            3,
        );
        let uncertainties = vec![0.5; 10];
        zero_clusters.update_uncertainties(uncertainties);
        let indices: Vec<usize> = zero_clusters.iter().collect();
        assert_eq!(indices.len(), 3); // Should fallback to uncertainty sampling
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_update_uncertainties_wrong_size() {
        let mut sampler =
            ActiveLearningSampler::new(10, AcquisitionStrategy::UncertaintySampling, 5);
        // Wrong size should panic
        sampler.update_uncertainties(vec![0.5; 5]);
    }

    #[test]
    fn test_acquisition_strategy_equality() {
        assert_eq!(
            AcquisitionStrategy::UncertaintySampling,
            AcquisitionStrategy::UncertaintySampling
        );
        assert_eq!(
            AcquisitionStrategy::DiversitySampling { num_clusters: 3 },
            AcquisitionStrategy::DiversitySampling { num_clusters: 3 }
        );
        assert_ne!(
            AcquisitionStrategy::UncertaintySampling,
            AcquisitionStrategy::ExpectedInformationGain
        );
    }

    #[test]
    fn test_acquisition_strategy_default() {
        assert_eq!(
            AcquisitionStrategy::default(),
            AcquisitionStrategy::UncertaintySampling
        );
    }

    #[test]
    fn test_reproducibility() {
        let mut sampler1 = ActiveLearningSampler::new(
            20,
            AcquisitionStrategy::DiversitySampling { num_clusters: 3 },
            5,
        )
        .with_generator(123);

        let mut sampler2 = ActiveLearningSampler::new(
            20,
            AcquisitionStrategy::DiversitySampling { num_clusters: 3 },
            5,
        )
        .with_generator(123);

        let uncertainties = vec![0.5; 20];
        sampler1.update_uncertainties(uncertainties.clone());
        sampler2.update_uncertainties(uncertainties);

        let indices1: Vec<usize> = sampler1.iter().collect();
        let indices2: Vec<usize> = sampler2.iter().collect();

        assert_eq!(indices1, indices2);
    }
}
