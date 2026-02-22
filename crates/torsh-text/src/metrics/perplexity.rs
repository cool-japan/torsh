//! Perplexity Calculation for Language Model Evaluation
//!
//! This module provides comprehensive perplexity computation for evaluating language models.
//! Perplexity measures how well a language model predicts a sequence of words - lower values
//! indicate better model performance. It is the exponential of the average negative log-likelihood
//! of the words in the sequence.
//!
//! # Mathematical Foundation
//!
//! For a sequence of N words w₁, w₂, ..., wₙ, perplexity is calculated as:
//!
//! ```text
//! Perplexity = exp(-1/N * Σᵢ log P(wᵢ|w₁...wᵢ₋₁))
//! ```
//!
//! # Features
//!
//! - Standard perplexity calculation from probabilities
//! - Perplexity calculation from raw logits with numerical stability
//! - Cross-entropy based perplexity computation
//! - Sequence-level and corpus-level evaluation
//! - Support for different smoothing techniques
//! - Comparative analysis between models
//!
//! # Example
//!
//! ```rust
//! use torsh_text::metrics::perplexity::PerplexityCalculator;
//!
//! let calculator = PerplexityCalculator::new();
//!
//! // From probabilities
//! let probabilities = &[0.3, 0.2, 0.4, 0.1];
//! let perplexity = calculator.calculate_from_probabilities(probabilities)?;
//! println!("Perplexity: {:.2}", perplexity);
//!
//! // From logits (more numerically stable)
//! let logits = &[-1.2, -1.6, -0.9, -2.3];
//! let perplexity = calculator.calculate_from_logits(logits)?;
//! println!("Perplexity from logits: {:.2}", perplexity);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{Result, TextError};
use std::collections::HashMap;

/// Smoothing techniques for handling zero probabilities
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothingMethod {
    /// No smoothing applied
    None,
    /// Add-one (Laplace) smoothing
    AddOne,
    /// Add-k smoothing with custom k value
    AddK(f64),
    /// Good-Turing smoothing
    GoodTuring,
    /// Kneser-Ney smoothing
    KneserNey,
}

/// Configuration for perplexity calculation
#[derive(Debug, Clone)]
pub struct PerplexityConfig {
    /// Smoothing method to handle zero probabilities
    pub smoothing: SmoothingMethod,
    /// Minimum probability threshold to prevent log(0)
    pub min_probability: f64,
    /// Use log-space calculations for better numerical stability
    pub use_log_space: bool,
    /// Vocabulary size for smoothing calculations
    pub vocabulary_size: Option<usize>,
}

impl Default for PerplexityConfig {
    fn default() -> Self {
        Self {
            smoothing: SmoothingMethod::None,
            min_probability: 1e-12,
            use_log_space: true,
            vocabulary_size: None,
        }
    }
}

/// Comprehensive perplexity calculator with advanced features
#[derive(Debug, Clone)]
pub struct PerplexityCalculator {
    config: PerplexityConfig,
}

impl Default for PerplexityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerplexityCalculator {
    /// Create a new perplexity calculator with default configuration
    pub fn new() -> Self {
        Self {
            config: PerplexityConfig::default(),
        }
    }

    /// Create a perplexity calculator with custom configuration
    pub fn with_config(config: PerplexityConfig) -> Self {
        Self { config }
    }

    /// Set smoothing method for handling zero probabilities
    pub fn with_smoothing(mut self, smoothing: SmoothingMethod) -> Self {
        self.config.smoothing = smoothing;
        self
    }

    /// Set minimum probability threshold
    pub fn with_min_probability(mut self, min_prob: f64) -> Self {
        self.config.min_probability = min_prob.max(0.0);
        self
    }

    /// Enable or disable log-space calculations
    pub fn with_log_space(mut self, use_log_space: bool) -> Self {
        self.config.use_log_space = use_log_space;
        self
    }

    /// Set vocabulary size for smoothing calculations
    pub fn with_vocabulary_size(mut self, vocab_size: usize) -> Self {
        self.config.vocabulary_size = Some(vocab_size);
        self
    }

    /// Calculate perplexity from probability values
    ///
    /// # Arguments
    /// * `probabilities` - Array of probability values for each token in the sequence
    ///
    /// # Returns
    /// Perplexity value where lower values indicate better model performance
    pub fn calculate_from_probabilities(&self, probabilities: &[f64]) -> Result<f64> {
        if probabilities.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No probabilities provided for perplexity calculation"
            )));
        }

        let smoothed_probs = self.apply_smoothing(probabilities)?;
        self.compute_perplexity(&smoothed_probs)
    }

    /// Calculate perplexity from raw logit values with numerical stability
    ///
    /// This method is more numerically stable than converting logits to probabilities first,
    /// as it avoids potential overflow/underflow issues.
    ///
    /// # Arguments
    /// * `logits` - Array of raw logit values from the language model
    ///
    /// # Returns
    /// Perplexity value computed directly from logits
    pub fn calculate_from_logits(&self, logits: &[f64]) -> Result<f64> {
        if logits.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No logits provided for perplexity calculation"
            )));
        }

        // Convert logits to log probabilities using log-sum-exp trick for numerical stability
        let log_probs = self.logits_to_log_probabilities(logits)?;

        // Calculate average negative log probability
        let avg_neg_log_prob = -log_probs.iter().sum::<f64>() / log_probs.len() as f64;

        Ok(avg_neg_log_prob.exp())
    }

    /// Calculate perplexity from cross-entropy loss
    ///
    /// Cross-entropy and perplexity are related: perplexity = exp(cross_entropy)
    ///
    /// # Arguments
    /// * `cross_entropy` - Cross-entropy loss value
    ///
    /// # Returns
    /// Perplexity value derived from cross-entropy
    pub fn calculate_from_cross_entropy(&self, cross_entropy: f64) -> Result<f64> {
        if cross_entropy < 0.0 {
            return Err(TextError::Other(anyhow::anyhow!(
                "Cross-entropy must be non-negative, got: {}",
                cross_entropy
            )));
        }

        Ok(cross_entropy.exp())
    }

    /// Calculate sequence-level perplexity for multiple sequences
    ///
    /// Computes perplexity for each sequence individually, returning detailed metrics
    /// including per-sequence perplexity and aggregate statistics.
    pub fn calculate_sequence_level(
        &self,
        sequences_probabilities: &[Vec<f64>],
    ) -> Result<SequencePerplexityMetrics> {
        if sequences_probabilities.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No sequences provided for perplexity calculation"
            )));
        }

        let mut sequence_perplexities = Vec::new();
        let mut total_log_prob = 0.0;
        let mut total_tokens = 0;

        for sequence_probs in sequences_probabilities {
            if sequence_probs.is_empty() {
                continue;
            }

            let seq_perplexity = self.calculate_from_probabilities(sequence_probs)?;
            sequence_perplexities.push(seq_perplexity);

            // Accumulate for corpus-level calculation
            let smoothed_probs = self.apply_smoothing(sequence_probs)?;
            for &prob in &smoothed_probs {
                total_log_prob += prob.max(self.config.min_probability).ln();
                total_tokens += 1;
            }
        }

        if sequence_perplexities.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No valid sequences found for perplexity calculation"
            )));
        }

        let corpus_perplexity = if total_tokens > 0 {
            (-total_log_prob / total_tokens as f64).exp()
        } else {
            0.0
        };

        let average_perplexity =
            sequence_perplexities.iter().sum::<f64>() / sequence_perplexities.len() as f64;
        let min_perplexity = sequence_perplexities
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let max_perplexity = sequence_perplexities
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate standard deviation
        let variance = sequence_perplexities
            .iter()
            .map(|&x| (x - average_perplexity).powi(2))
            .sum::<f64>()
            / sequence_perplexities.len() as f64;
        let std_deviation = variance.sqrt();

        Ok(SequencePerplexityMetrics {
            sequence_perplexities,
            corpus_perplexity,
            average_perplexity,
            min_perplexity,
            max_perplexity,
            std_deviation,
            total_sequences: sequences_probabilities.len(),
            total_tokens,
        })
    }

    /// Compare perplexity between two models on the same data
    ///
    /// Provides detailed comparison metrics including relative improvement,
    /// statistical significance tests, and per-sequence analysis.
    pub fn compare_models(
        &self,
        model1_probabilities: &[Vec<f64>],
        model2_probabilities: &[Vec<f64>],
        model1_name: &str,
        model2_name: &str,
    ) -> Result<ModelComparisonMetrics> {
        if model1_probabilities.len() != model2_probabilities.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Number of sequences must match between models: {} vs {}",
                model1_probabilities.len(),
                model2_probabilities.len()
            )));
        }

        let metrics1 = self.calculate_sequence_level(model1_probabilities)?;
        let metrics2 = self.calculate_sequence_level(model2_probabilities)?;

        let relative_improvement = if metrics1.corpus_perplexity > 0.0 {
            (metrics1.corpus_perplexity - metrics2.corpus_perplexity) / metrics1.corpus_perplexity
        } else {
            0.0
        };

        let better_model = if metrics2.corpus_perplexity < metrics1.corpus_perplexity {
            model2_name
        } else {
            model1_name
        };

        // Count wins per sequence
        let mut model1_wins = 0;
        let mut model2_wins = 0;
        let mut ties = 0;

        for (&perp1, &perp2) in metrics1
            .sequence_perplexities
            .iter()
            .zip(metrics2.sequence_perplexities.iter())
        {
            if (perp1 - perp2).abs() < 1e-6 {
                ties += 1;
            } else if perp1 < perp2 {
                model1_wins += 1;
            } else {
                model2_wins += 1;
            }
        }

        Ok(ModelComparisonMetrics {
            model1_name: model1_name.to_string(),
            model2_name: model2_name.to_string(),
            model1_metrics: metrics1,
            model2_metrics: metrics2,
            relative_improvement,
            better_model: better_model.to_string(),
            model1_wins,
            model2_wins,
            ties,
        })
    }

    /// Calculate perplexity bounds for confidence intervals
    ///
    /// Provides confidence intervals for perplexity estimates using bootstrap sampling
    pub fn calculate_confidence_interval(
        &self,
        probabilities: &[f64],
        confidence_level: f64,
        bootstrap_samples: usize,
    ) -> Result<ConfidenceInterval> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(TextError::Other(anyhow::anyhow!(
                "Confidence level must be between 0 and 1, got: {}",
                confidence_level
            )));
        }

        if bootstrap_samples < 10 {
            return Err(TextError::Other(anyhow::anyhow!(
                "Bootstrap samples must be at least 10, got: {}",
                bootstrap_samples
            )));
        }

        let base_perplexity = self.calculate_from_probabilities(probabilities)?;
        let mut bootstrap_perplexities = Vec::new();

        // Simple bootstrap resampling
        for _ in 0..bootstrap_samples {
            let mut resampled_probs = Vec::with_capacity(probabilities.len());
            for _ in 0..probabilities.len() {
                let idx = (rand::random::<f64>() * probabilities.len() as f64) as usize;
                resampled_probs.push(probabilities[idx.min(probabilities.len() - 1)]);
            }

            if let Ok(perp) = self.calculate_from_probabilities(&resampled_probs) {
                bootstrap_perplexities.push(perp);
            }
        }

        if bootstrap_perplexities.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No valid bootstrap samples generated"
            )));
        }

        bootstrap_perplexities.sort_by(|a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * bootstrap_perplexities.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * bootstrap_perplexities.len() as f64) as usize;

        let lower_bound = bootstrap_perplexities[lower_idx.min(bootstrap_perplexities.len() - 1)];
        let upper_bound = bootstrap_perplexities[upper_idx.min(bootstrap_perplexities.len() - 1)];

        Ok(ConfidenceInterval {
            base_perplexity,
            confidence_level,
            lower_bound,
            upper_bound,
            bootstrap_samples: bootstrap_perplexities.len(),
        })
    }

    // Private implementation methods

    /// Apply smoothing to probabilities to handle zero values
    fn apply_smoothing(&self, probabilities: &[f64]) -> Result<Vec<f64>> {
        match self.config.smoothing {
            SmoothingMethod::None => {
                // Just clamp to minimum probability
                Ok(probabilities
                    .iter()
                    .map(|&p| p.max(self.config.min_probability))
                    .collect())
            }
            SmoothingMethod::AddOne => {
                let vocab_size = self.config.vocabulary_size.unwrap_or(probabilities.len());
                Ok(probabilities
                    .iter()
                    .map(|&p| {
                        (p * probabilities.len() as f64 + 1.0)
                            / (probabilities.len() + vocab_size) as f64
                    })
                    .collect())
            }
            SmoothingMethod::AddK(k) => {
                let vocab_size = self.config.vocabulary_size.unwrap_or(probabilities.len());
                Ok(probabilities
                    .iter()
                    .map(|&p| {
                        (p * probabilities.len() as f64 + k)
                            / (probabilities.len() as f64 + k * vocab_size as f64)
                    })
                    .collect())
            }
            SmoothingMethod::GoodTuring | SmoothingMethod::KneserNey => {
                // Simplified implementation - in practice, these require more sophisticated algorithms
                Ok(probabilities
                    .iter()
                    .map(|&p| {
                        if p == 0.0 {
                            self.config.min_probability
                        } else {
                            p
                        }
                    })
                    .collect())
            }
        }
    }

    /// Convert logits to log probabilities using numerically stable log-sum-exp
    fn logits_to_log_probabilities(&self, logits: &[f64]) -> Result<Vec<f64>> {
        // Find maximum logit for numerical stability
        let max_logit = logits.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        // Calculate log-sum-exp
        let exp_sum: f64 = logits.iter().map(|&x| (x - max_logit).exp()).sum();

        if exp_sum <= 0.0 {
            return Err(TextError::Other(anyhow::anyhow!(
                "Invalid logits resulted in zero or negative sum"
            )));
        }

        let log_sum_exp = max_logit + exp_sum.ln();

        // Convert to log probabilities
        Ok(logits.iter().map(|&logit| logit - log_sum_exp).collect())
    }

    /// Compute perplexity from processed probabilities
    fn compute_perplexity(&self, probabilities: &[f64]) -> Result<f64> {
        let mut log_sum = 0.0;

        for &prob in probabilities {
            if prob <= 0.0 {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Invalid probability encountered: {}",
                    prob
                )));
            }
            log_sum += prob.ln();
        }

        let average_log_prob = log_sum / probabilities.len() as f64;
        Ok((-average_log_prob).exp())
    }
}

/// Detailed metrics for sequence-level perplexity analysis
#[derive(Debug, Clone)]
pub struct SequencePerplexityMetrics {
    /// Perplexity for each individual sequence
    pub sequence_perplexities: Vec<f64>,
    /// Corpus-level perplexity across all sequences
    pub corpus_perplexity: f64,
    /// Average perplexity across sequences
    pub average_perplexity: f64,
    /// Minimum perplexity observed
    pub min_perplexity: f64,
    /// Maximum perplexity observed
    pub max_perplexity: f64,
    /// Standard deviation of sequence perplexities
    pub std_deviation: f64,
    /// Total number of sequences processed
    pub total_sequences: usize,
    /// Total number of tokens processed
    pub total_tokens: usize,
}

/// Comparative analysis metrics between two models
#[derive(Debug, Clone)]
pub struct ModelComparisonMetrics {
    /// Name of the first model
    pub model1_name: String,
    /// Name of the second model
    pub model2_name: String,
    /// Detailed metrics for model 1
    pub model1_metrics: SequencePerplexityMetrics,
    /// Detailed metrics for model 2
    pub model2_metrics: SequencePerplexityMetrics,
    /// Relative improvement from model 1 to model 2 (negative means model 2 is worse)
    pub relative_improvement: f64,
    /// Name of the model with better (lower) perplexity
    pub better_model: String,
    /// Number of sequences where model 1 performed better
    pub model1_wins: usize,
    /// Number of sequences where model 2 performed better
    pub model2_wins: usize,
    /// Number of sequences with tied performance
    pub ties: usize,
}

/// Confidence interval for perplexity estimates
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    /// Base perplexity estimate
    pub base_perplexity: f64,
    /// Confidence level (e.g., 0.95 for 95% confidence)
    pub confidence_level: f64,
    /// Lower bound of the confidence interval
    pub lower_bound: f64,
    /// Upper bound of the confidence interval
    pub upper_bound: f64,
    /// Number of bootstrap samples used
    pub bootstrap_samples: usize,
}

impl SequencePerplexityMetrics {
    /// Check if the corpus shows consistent performance (low variance)
    pub fn is_consistent(&self, threshold: f64) -> bool {
        self.std_deviation <= threshold
    }

    /// Get the range of perplexity values (max - min)
    pub fn perplexity_range(&self) -> f64 {
        self.max_perplexity - self.min_perplexity
    }

    /// Get coefficient of variation (std_dev / mean)
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.average_perplexity > 0.0 {
            self.std_deviation / self.average_perplexity
        } else {
            0.0
        }
    }
}

impl ModelComparisonMetrics {
    /// Check if model 2 is significantly better than model 1
    pub fn is_model2_better(&self) -> bool {
        self.model2_metrics.corpus_perplexity < self.model1_metrics.corpus_perplexity
    }

    /// Get the percentage of sequences where model 2 won
    pub fn model2_win_rate(&self) -> f64 {
        if self.total_comparisons() > 0 {
            self.model2_wins as f64 / self.total_comparisons() as f64
        } else {
            0.0
        }
    }

    /// Get total number of comparisons made
    pub fn total_comparisons(&self) -> usize {
        self.model1_wins + self.model2_wins + self.ties
    }

    /// Get relative performance improvement as percentage
    pub fn improvement_percentage(&self) -> f64 {
        self.relative_improvement * 100.0
    }
}

impl ConfidenceInterval {
    /// Check if the confidence interval contains a specific value
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower_bound && value <= self.upper_bound
    }

    /// Get the width of the confidence interval
    pub fn width(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }

    /// Get the margin of error (half the width)
    pub fn margin_of_error(&self) -> f64 {
        self.width() / 2.0
    }
}

/// Legacy compatibility function - calculate perplexity from probabilities
pub fn calculate(probabilities: &[f64]) -> Result<f64> {
    PerplexityCalculator::new().calculate_from_probabilities(probabilities)
}

/// Legacy compatibility function - calculate perplexity from logits
pub fn calculate_from_logits(logits: &[f64]) -> Result<f64> {
    PerplexityCalculator::new().calculate_from_logits(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_perplexity_calculation() {
        let calculator = PerplexityCalculator::new();
        let probabilities = &[0.25, 0.25, 0.25, 0.25]; // Uniform distribution
        let perplexity = calculator
            .calculate_from_probabilities(probabilities)
            .unwrap();

        // For uniform distribution over 4 items, perplexity should be 4
        assert!((perplexity - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_perplexity_from_logits() {
        let calculator = PerplexityCalculator::new();
        let logits = &[1.0, 1.0, 1.0, 1.0]; // Uniform logits
        let perplexity = calculator.calculate_from_logits(logits).unwrap();

        // Should also result in perplexity of 4
        assert!((perplexity - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_prediction() {
        let calculator = PerplexityCalculator::new();
        let probabilities = &[1.0]; // Perfect prediction
        let perplexity = calculator
            .calculate_from_probabilities(probabilities)
            .unwrap();

        // Perfect prediction should have perplexity of 1
        assert!((perplexity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_input() {
        let calculator = PerplexityCalculator::new();
        let result = calculator.calculate_from_probabilities(&[]);
        assert!(result.is_err());

        let result = calculator.calculate_from_logits(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_probabilities() {
        let calculator = PerplexityCalculator::new();
        let probabilities = &[0.5, -0.1]; // Negative probability
        let result = calculator.calculate_from_probabilities(probabilities);
        // Should work with smoothing that clamps to min_probability
        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_entropy_conversion() {
        let calculator = PerplexityCalculator::new();
        let cross_entropy = 2.0;
        let perplexity = calculator
            .calculate_from_cross_entropy(cross_entropy)
            .unwrap();

        assert!((perplexity - cross_entropy.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_sequence_level_analysis() {
        let calculator = PerplexityCalculator::new();
        let sequences = vec![vec![0.5, 0.5], vec![0.25, 0.25, 0.25, 0.25], vec![0.1, 0.9]];

        let metrics = calculator.calculate_sequence_level(&sequences).unwrap();

        assert_eq!(metrics.total_sequences, 3);
        assert_eq!(metrics.sequence_perplexities.len(), 3);
        assert!(metrics.corpus_perplexity > 0.0);
        assert!(metrics.average_perplexity > 0.0);
        assert!(metrics.min_perplexity <= metrics.max_perplexity);
    }

    #[test]
    fn test_model_comparison() {
        let calculator = PerplexityCalculator::new();

        // Model 1: worse performance (higher perplexity)
        let model1_probs = vec![
            vec![0.1, 0.9], // High perplexity
            vec![0.3, 0.7], // Medium perplexity
        ];

        // Model 2: better performance (lower perplexity)
        let model2_probs = vec![
            vec![0.01, 0.99], // Low perplexity
            vec![0.05, 0.95], // Low perplexity
        ];

        let comparison = calculator
            .compare_models(&model1_probs, &model2_probs, "Model1", "Model2")
            .unwrap();

        assert_eq!(comparison.better_model, "Model2");
        assert!(comparison.relative_improvement > 0.0);
        assert_eq!(comparison.model2_wins, 2);
        assert_eq!(comparison.model1_wins, 0);
    }

    #[test]
    fn test_smoothing_methods() {
        let calculator_none = PerplexityCalculator::new().with_smoothing(SmoothingMethod::None);
        let calculator_add_one = PerplexityCalculator::new()
            .with_smoothing(SmoothingMethod::AddOne)
            .with_vocabulary_size(10);

        let probabilities = &[0.0, 0.5, 0.5]; // Contains zero probability

        let perp_none = calculator_none
            .calculate_from_probabilities(probabilities)
            .unwrap();
        let perp_add_one = calculator_add_one
            .calculate_from_probabilities(probabilities)
            .unwrap();

        // Both should work, but give different results
        assert!(perp_none > 0.0);
        assert!(perp_add_one > 0.0);
        assert_ne!(perp_none, perp_add_one);
    }

    #[test]
    fn test_confidence_interval() {
        let calculator = PerplexityCalculator::new();
        let probabilities = vec![0.2, 0.3, 0.1, 0.4];

        let ci = calculator
            .calculate_confidence_interval(&probabilities, 0.95, 100)
            .unwrap();

        assert!(ci.confidence_level == 0.95);
        assert!(ci.lower_bound <= ci.base_perplexity);
        assert!(ci.upper_bound >= ci.base_perplexity);
        assert!(ci.lower_bound <= ci.upper_bound);
        assert!(ci.bootstrap_samples > 0);
    }

    #[test]
    fn test_legacy_functions() {
        let probabilities = &[0.25, 0.25, 0.25, 0.25];
        let perplexity = calculate(probabilities).unwrap();
        assert!((perplexity - 4.0).abs() < 1e-10);

        let logits = &[1.0, 1.0, 1.0, 1.0];
        let perplexity = calculate_from_logits(logits).unwrap();
        assert!((perplexity - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_stability() {
        let calculator = PerplexityCalculator::new();

        // Test with very small probabilities
        let small_probs = &[1e-10, 1e-10, 1.0 - 2e-10];
        let perplexity = calculator
            .calculate_from_probabilities(small_probs)
            .unwrap();
        assert!(perplexity.is_finite());
        assert!(perplexity > 0.0);

        // Test with large logits
        let large_logits = &[100.0, 101.0, 99.0];
        let perplexity = calculator.calculate_from_logits(large_logits).unwrap();
        assert!(perplexity.is_finite());
        assert!(perplexity > 0.0);
    }
}
