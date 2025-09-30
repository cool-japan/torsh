//! Coherence Metrics Module
//!
//! This module provides comprehensive statistical analysis and measurement functionality
//! for coherence analysis. It includes metric calculation, statistical aggregation,
//! confidence scoring, and comparative analysis capabilities.

use std::collections::{HashMap, HashSet};
use thiserror::Error;

use super::{
    discourse_coherence::DiscourseCoherenceResult, entity_coherence::EntityCoherenceResult,
    lexical_coherence::LexicalCoherenceResult, structural_coherence::StructuralCoherenceResult,
    topic_coherence::TopicCoherenceResult,
};

/// Errors that can occur during coherence metrics calculation
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CoherenceMetricsError {
    #[error("Invalid input data: {message}")]
    InvalidInput { message: String },
    #[error(
        "Insufficient data for statistical analysis: {required} items required, {found} found"
    )]
    InsufficientData { required: usize, found: usize },
    #[error("Statistical calculation failed: {operation}")]
    StatisticalError { operation: String },
    #[error("Configuration error: {parameter} = {value}")]
    ConfigurationError { parameter: String, value: String },
}

/// Configuration for coherence metrics calculation
#[derive(Debug, Clone)]
pub struct CoherenceMetricsConfig {
    pub window_size: usize,
    pub confidence_threshold: f64,
    pub enable_statistical_analysis: bool,
    pub enable_distribution_analysis: bool,
    pub enable_comparative_analysis: bool,
    pub metric_weights: MetricWeights,
    pub statistical_parameters: StatisticalParameters,
}

/// Weight configuration for different coherence metrics
#[derive(Debug, Clone)]
pub struct MetricWeights {
    pub entity_weight: f64,
    pub lexical_weight: f64,
    pub discourse_weight: f64,
    pub topic_weight: f64,
    pub structural_weight: f64,
}

/// Parameters for statistical analysis
#[derive(Debug, Clone)]
pub struct StatisticalParameters {
    pub use_robust_statistics: bool,
    pub confidence_level: f64,
    pub outlier_threshold: f64,
    pub smoothing_factor: f64,
}

impl Default for CoherenceMetricsConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            confidence_threshold: 0.7,
            enable_statistical_analysis: true,
            enable_distribution_analysis: true,
            enable_comparative_analysis: true,
            metric_weights: MetricWeights::default(),
            statistical_parameters: StatisticalParameters::default(),
        }
    }
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self {
            entity_weight: 0.25,
            lexical_weight: 0.20,
            discourse_weight: 0.20,
            topic_weight: 0.20,
            structural_weight: 0.15,
        }
    }
}

impl Default for StatisticalParameters {
    fn default() -> Self {
        Self {
            use_robust_statistics: true,
            confidence_level: 0.95,
            outlier_threshold: 2.0,
            smoothing_factor: 0.1,
        }
    }
}

/// Comprehensive coherence metrics result
#[derive(Debug, Clone, PartialEq)]
pub struct CoherenceMetricsResult {
    pub overall_coherence: f64,
    pub confidence_score: f64,
    pub coherence_distribution: Vec<f64>,
    pub coherence_transitions: Vec<f64>,
    pub local_coherence: f64,
    pub global_coherence: f64,
    pub coherence_breakdown: HashMap<String, f64>,
    pub statistical_analysis: Option<StatisticalAnalysis>,
    pub distribution_analysis: Option<DistributionAnalysis>,
    pub quality_indicators: QualityIndicators,
}

/// Statistical analysis of coherence metrics
#[derive(Debug, Clone, PartialEq)]
pub struct StatisticalAnalysis {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub percentiles: HashMap<u8, f64>,
    pub outliers: Vec<usize>,
    pub robust_mean: f64,
    pub robust_std_dev: f64,
}

/// Distribution analysis of coherence scores
#[derive(Debug, Clone, PartialEq)]
pub struct DistributionAnalysis {
    pub histogram: HashMap<String, usize>,
    pub distribution_type: DistributionType,
    pub goodness_of_fit: f64,
    pub entropy: f64,
    pub uniformity: f64,
    pub clustering_coefficient: f64,
    pub distribution_summary: DistributionSummary,
}

/// Types of statistical distributions
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionType {
    Normal,
    Skewed,
    Bimodal,
    Uniform,
    Exponential,
    Unknown,
}

/// Summary statistics for distribution
#[derive(Debug, Clone, PartialEq)]
pub struct DistributionSummary {
    pub range: f64,
    pub interquartile_range: f64,
    pub coefficient_of_variation: f64,
    pub modal_class: String,
    pub concentration: f64,
}

/// Quality indicators for coherence assessment
#[derive(Debug, Clone, PartialEq)]
pub struct QualityIndicators {
    pub reliability_score: f64,
    pub consistency_score: f64,
    pub coverage_score: f64,
    pub stability_score: f64,
    pub discriminant_validity: f64,
}

/// Comparison result between coherence analyses
#[derive(Debug, Clone, PartialEq)]
pub struct CoherenceComparisonResult {
    pub text1_coherence: f64,
    pub text2_coherence: f64,
    pub coherence_difference: f64,
    pub better_text: u8,
    pub detailed_comparison: HashMap<String, f64>,
    pub significance_test: Option<SignificanceTest>,
    pub effect_size: f64,
}

/// Statistical significance test result
#[derive(Debug, Clone, PartialEq)]
pub struct SignificanceTest {
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_significant: bool,
    pub test_type: String,
}

/// Main coherence metrics calculator
pub struct CoherenceMetricsCalculator {
    config: CoherenceMetricsConfig,
}

impl CoherenceMetricsCalculator {
    /// Create new metrics calculator with configuration
    pub fn new(config: CoherenceMetricsConfig) -> Self {
        Self { config }
    }

    /// Create calculator with default configuration
    pub fn with_default_config() -> Self {
        Self {
            config: CoherenceMetricsConfig::default(),
        }
    }

    /// Calculate comprehensive coherence metrics from individual results
    pub fn calculate_comprehensive_metrics(
        &self,
        entity_coherence: &EntityCoherenceResult,
        lexical_coherence: &LexicalCoherenceResult,
        discourse_coherence: &DiscourseCoherenceResult,
        topic_coherence: &TopicCoherenceResult,
        structural_coherence: &StructuralCoherenceResult,
        sentences: &[String],
        paragraphs: &[String],
    ) -> Result<CoherenceMetricsResult, CoherenceMetricsError> {
        if sentences.is_empty() {
            return Err(CoherenceMetricsError::InvalidInput {
                message: "No sentences provided".to_string(),
            });
        }

        // Calculate overall coherence
        let overall_coherence = self.calculate_overall_coherence(
            entity_coherence,
            lexical_coherence,
            discourse_coherence,
            topic_coherence,
            structural_coherence,
        )?;

        // Calculate distribution and transitions
        let coherence_distribution = self.calculate_coherence_distribution(sentences)?;
        let coherence_transitions = self.calculate_coherence_transitions(sentences)?;

        // Calculate local and global coherence
        let local_coherence = self.calculate_local_coherence(sentences)?;
        let global_coherence = self.calculate_global_coherence(sentences, paragraphs)?;

        // Create coherence breakdown
        let coherence_breakdown = self.create_coherence_breakdown(
            entity_coherence,
            lexical_coherence,
            discourse_coherence,
            topic_coherence,
            structural_coherence,
        )?;

        // Calculate confidence score
        let confidence_score =
            self.calculate_confidence_score(overall_coherence, &coherence_distribution)?;

        // Perform statistical analysis if enabled
        let statistical_analysis = if self.config.enable_statistical_analysis {
            Some(self.perform_statistical_analysis(&coherence_distribution)?)
        } else {
            None
        };

        // Perform distribution analysis if enabled
        let distribution_analysis = if self.config.enable_distribution_analysis {
            Some(self.perform_distribution_analysis(&coherence_distribution)?)
        } else {
            None
        };

        // Calculate quality indicators
        let quality_indicators = self.calculate_quality_indicators(
            overall_coherence,
            &coherence_distribution,
            &coherence_transitions,
        )?;

        Ok(CoherenceMetricsResult {
            overall_coherence,
            confidence_score,
            coherence_distribution,
            coherence_transitions,
            local_coherence,
            global_coherence,
            coherence_breakdown,
            statistical_analysis,
            distribution_analysis,
            quality_indicators,
        })
    }

    /// Calculate overall coherence score from individual metrics
    pub fn calculate_overall_coherence(
        &self,
        entity_coherence: &EntityCoherenceResult,
        lexical_coherence: &LexicalCoherenceResult,
        discourse_coherence: &DiscourseCoherenceResult,
        topic_coherence: &TopicCoherenceResult,
        structural_coherence: &StructuralCoherenceResult,
    ) -> Result<f64, CoherenceMetricsError> {
        let weights = &self.config.metric_weights;

        let entity_score = (entity_coherence.entity_grid_coherence
            + entity_coherence.entity_transition_coherence
            + entity_coherence.coreference_coherence)
            / 3.0;

        let lexical_score = (lexical_coherence.lexical_chain_coherence
            + lexical_coherence.semantic_field_coherence
            + lexical_coherence.vocabulary_consistency)
            / 3.0;

        let discourse_score = (discourse_coherence.discourse_marker_coherence
            + discourse_coherence.transition_coherence
            + discourse_coherence.cohesion_score)
            / 3.0;

        let topic_score = (topic_coherence.topic_consistency
            + topic_coherence.topic_shift_coherence
            + topic_coherence.thematic_unity)
            / 3.0;

        let structural_score = (structural_coherence.paragraph_coherence
            + structural_coherence.organizational_coherence
            + structural_coherence.hierarchical_coherence)
            / 3.0;

        let overall = (entity_score * weights.entity_weight)
            + (lexical_score * weights.lexical_weight)
            + (discourse_score * weights.discourse_weight)
            + (topic_score * weights.topic_weight)
            + (structural_score * weights.structural_weight);

        Ok(overall.clamp(0.0, 1.0))
    }

    /// Calculate coherence distribution across text segments
    pub fn calculate_coherence_distribution(
        &self,
        sentences: &[String],
    ) -> Result<Vec<f64>, CoherenceMetricsError> {
        if sentences.len() < self.config.window_size {
            return Err(CoherenceMetricsError::InsufficientData {
                required: self.config.window_size,
                found: sentences.len(),
            });
        }

        let mut distribution = Vec::new();
        let window_size = self.config.window_size;

        for i in 0..=sentences.len().saturating_sub(window_size) {
            let window_end = (i + window_size).min(sentences.len());
            let window_sentences = &sentences[i..window_end];
            let local_coherence = self.calculate_local_coherence(window_sentences)?;
            distribution.push(local_coherence);
        }

        Ok(distribution)
    }

    /// Calculate coherence transitions between adjacent sentences
    pub fn calculate_coherence_transitions(
        &self,
        sentences: &[String],
    ) -> Result<Vec<f64>, CoherenceMetricsError> {
        if sentences.len() < 2 {
            return Err(CoherenceMetricsError::InsufficientData {
                required: 2,
                found: sentences.len(),
            });
        }

        let mut transitions = Vec::new();

        for i in 0..sentences.len().saturating_sub(1) {
            let transition_quality =
                self.calculate_sentence_lexical_overlap(&sentences[i], &sentences[i + 1]);
            transitions.push(transition_quality);
        }

        Ok(transitions)
    }

    /// Calculate local coherence for a set of sentences
    pub fn calculate_local_coherence(
        &self,
        sentences: &[String],
    ) -> Result<f64, CoherenceMetricsError> {
        if sentences.len() < 2 {
            return Ok(1.0);
        }

        let mut coherence_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..sentences.len() - 1 {
            let current = &sentences[i];
            let next = &sentences[i + 1];

            let lexical_overlap = self.calculate_sentence_lexical_overlap(current, next);
            let semantic_continuity = self.calculate_semantic_continuity(current, next);
            let structural_continuity = self.calculate_structural_continuity(current, next);

            let local_coherence = (lexical_overlap * 0.4)
                + (semantic_continuity * 0.4)
                + (structural_continuity * 0.2);
            coherence_sum += local_coherence;
            comparisons += 1;
        }

        if comparisons > 0 {
            Ok(coherence_sum / comparisons as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate global coherence across entire text
    pub fn calculate_global_coherence(
        &self,
        sentences: &[String],
        paragraphs: &[String],
    ) -> Result<f64, CoherenceMetricsError> {
        let sentence_coherence = self.calculate_sentence_level_global_coherence(sentences)?;
        let paragraph_coherence = self.calculate_paragraph_level_global_coherence(paragraphs)?;
        let document_coherence = self.calculate_document_level_coherence(sentences, paragraphs)?;

        Ok((sentence_coherence * 0.4) + (paragraph_coherence * 0.4) + (document_coherence * 0.2))
    }

    /// Calculate confidence score for coherence assessment
    pub fn calculate_confidence_score(
        &self,
        overall_coherence: f64,
        distribution: &[f64],
    ) -> Result<f64, CoherenceMetricsError> {
        if distribution.is_empty() {
            return Ok(0.5);
        }

        let mean = distribution.iter().sum::<f64>() / distribution.len() as f64;
        let variance = distribution.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / distribution.len() as f64;
        let std_dev = variance.sqrt();

        let consistency_score = (1.0 - std_dev).max(0.0);
        let strength_score = overall_coherence;
        let coverage_score = ((distribution.len() as f64).ln() / 10.0).min(1.0);

        Ok((consistency_score * 0.4) + (strength_score * 0.4) + (coverage_score * 0.2))
    }

    /// Create detailed breakdown of coherence components
    pub fn create_coherence_breakdown(
        &self,
        entity_coherence: &EntityCoherenceResult,
        lexical_coherence: &LexicalCoherenceResult,
        discourse_coherence: &DiscourseCoherenceResult,
        topic_coherence: &TopicCoherenceResult,
        structural_coherence: &StructuralCoherenceResult,
    ) -> Result<HashMap<String, f64>, CoherenceMetricsError> {
        let mut breakdown = HashMap::new();

        // Entity coherence components
        breakdown.insert(
            "entity_grid".to_string(),
            entity_coherence.entity_grid_coherence,
        );
        breakdown.insert(
            "entity_transitions".to_string(),
            entity_coherence.entity_transition_coherence,
        );
        breakdown.insert(
            "coreference".to_string(),
            entity_coherence.coreference_coherence,
        );
        breakdown.insert(
            "entity_density".to_string(),
            entity_coherence.entity_density,
        );

        // Lexical coherence components
        breakdown.insert(
            "lexical_chains".to_string(),
            lexical_coherence.lexical_chain_coherence,
        );
        breakdown.insert(
            "semantic_fields".to_string(),
            lexical_coherence.semantic_field_coherence,
        );
        breakdown.insert(
            "lexical_repetition".to_string(),
            lexical_coherence.lexical_repetition_score,
        );
        breakdown.insert(
            "vocabulary_consistency".to_string(),
            lexical_coherence.vocabulary_consistency,
        );

        // Discourse coherence components
        breakdown.insert(
            "discourse_markers".to_string(),
            discourse_coherence.discourse_marker_coherence,
        );
        breakdown.insert(
            "transition_coherence".to_string(),
            discourse_coherence.transition_coherence,
        );
        breakdown.insert(
            "cohesion_score".to_string(),
            discourse_coherence.cohesion_score,
        );

        // Topic coherence components
        breakdown.insert(
            "topic_consistency".to_string(),
            topic_coherence.topic_consistency,
        );
        breakdown.insert(
            "topic_shift_coherence".to_string(),
            topic_coherence.topic_shift_coherence,
        );
        breakdown.insert("thematic_unity".to_string(), topic_coherence.thematic_unity);

        // Structural coherence components
        breakdown.insert(
            "paragraph_coherence".to_string(),
            structural_coherence.paragraph_coherence,
        );
        breakdown.insert(
            "section_coherence".to_string(),
            structural_coherence.section_coherence,
        );
        breakdown.insert(
            "organizational_coherence".to_string(),
            structural_coherence.organizational_coherence,
        );
        breakdown.insert(
            "hierarchical_coherence".to_string(),
            structural_coherence.hierarchical_coherence,
        );
        breakdown.insert(
            "structural_consistency".to_string(),
            structural_coherence.structural_consistency,
        );

        Ok(breakdown)
    }

    /// Perform comprehensive statistical analysis
    pub fn perform_statistical_analysis(
        &self,
        distribution: &[f64],
    ) -> Result<StatisticalAnalysis, CoherenceMetricsError> {
        if distribution.len() < 3 {
            return Err(CoherenceMetricsError::InsufficientData {
                required: 3,
                found: distribution.len(),
            });
        }

        let mut sorted_data = distribution.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = distribution.iter().sum::<f64>() / distribution.len() as f64;
        let median = self.calculate_median(&sorted_data);
        let variance = distribution.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / distribution.len() as f64;
        let std_dev = variance.sqrt();

        let skewness = self.calculate_skewness(distribution, mean, std_dev)?;
        let kurtosis = self.calculate_kurtosis(distribution, mean, std_dev)?;
        let percentiles = self.calculate_percentiles(&sorted_data)?;
        let outliers = self.detect_outliers(distribution, mean, std_dev)?;
        let (robust_mean, robust_std_dev) = self.calculate_robust_statistics(&sorted_data)?;

        Ok(StatisticalAnalysis {
            mean,
            median,
            std_dev,
            variance,
            skewness,
            kurtosis,
            percentiles,
            outliers,
            robust_mean,
            robust_std_dev,
        })
    }

    /// Perform distribution analysis
    pub fn perform_distribution_analysis(
        &self,
        distribution: &[f64],
    ) -> Result<DistributionAnalysis, CoherenceMetricsError> {
        let histogram = self.calculate_histogram(distribution)?;
        let distribution_type = self.classify_distribution(distribution)?;
        let goodness_of_fit = self.calculate_goodness_of_fit(distribution, &distribution_type)?;
        let entropy = self.calculate_entropy(&histogram);
        let uniformity = self.calculate_uniformity(distribution)?;
        let clustering_coefficient = self.calculate_clustering_coefficient(distribution)?;
        let distribution_summary = self.calculate_distribution_summary(distribution)?;

        Ok(DistributionAnalysis {
            histogram,
            distribution_type,
            goodness_of_fit,
            entropy,
            uniformity,
            clustering_coefficient,
            distribution_summary,
        })
    }

    /// Calculate quality indicators for coherence assessment
    pub fn calculate_quality_indicators(
        &self,
        overall_coherence: f64,
        distribution: &[f64],
        transitions: &[f64],
    ) -> Result<QualityIndicators, CoherenceMetricsError> {
        let reliability_score = self.calculate_reliability_score(distribution)?;
        let consistency_score = self.calculate_consistency_score(transitions)?;
        let coverage_score = self.calculate_coverage_score(distribution.len())?;
        let stability_score = self.calculate_stability_score(distribution)?;
        let discriminant_validity =
            self.calculate_discriminant_validity(overall_coherence, distribution)?;

        Ok(QualityIndicators {
            reliability_score,
            consistency_score,
            coverage_score,
            stability_score,
            discriminant_validity,
        })
    }

    /// Compare coherence between two texts
    pub fn compare_coherence(
        &self,
        result1: &CoherenceMetricsResult,
        result2: &CoherenceMetricsResult,
    ) -> Result<CoherenceComparisonResult, CoherenceMetricsError> {
        let coherence_difference = result1.overall_coherence - result2.overall_coherence;
        let better_text = if coherence_difference > 0.0 { 1 } else { 2 };

        let mut detailed_comparison = HashMap::new();
        for (key, value1) in &result1.coherence_breakdown {
            if let Some(value2) = result2.coherence_breakdown.get(key) {
                detailed_comparison.insert(key.clone(), value1 - value2);
            }
        }

        let significance_test = if self.config.enable_comparative_analysis {
            Some(self.perform_significance_test(
                &result1.coherence_distribution,
                &result2.coherence_distribution,
            )?)
        } else {
            None
        };

        let effect_size = self.calculate_effect_size(
            &result1.coherence_distribution,
            &result2.coherence_distribution,
        )?;

        Ok(CoherenceComparisonResult {
            text1_coherence: result1.overall_coherence,
            text2_coherence: result2.overall_coherence,
            coherence_difference,
            better_text,
            detailed_comparison,
            significance_test,
            effect_size,
        })
    }

    // Private helper methods

    fn calculate_sentence_lexical_overlap(&self, sentence1: &str, sentence2: &str) -> f64 {
        let words1: HashSet<String> = sentence1
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase())
            .filter(|w| !w.is_empty() && w.len() > 2)
            .collect();

        let words2: HashSet<String> = sentence2
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase())
            .filter(|w| !w.is_empty() && w.len() > 2)
            .collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    fn calculate_semantic_continuity(&self, sentence1: &str, sentence2: &str) -> f64 {
        // Simplified semantic continuity calculation
        // In a full implementation, this would use semantic similarity models
        let words1: Vec<String> = sentence1
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase())
            .filter(|w| !w.is_empty())
            .collect();

        let words2: Vec<String> = sentence2
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase())
            .filter(|w| !w.is_empty())
            .collect();

        let mut similarity_sum = 0.0;
        let mut comparisons = 0;

        for word1 in &words1 {
            for word2 in &words2 {
                let similarity = self.calculate_word_similarity(word1, word2);
                similarity_sum += similarity;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            similarity_sum / comparisons as f64
        } else {
            0.0
        }
    }

    fn calculate_structural_continuity(&self, sentence1: &str, sentence2: &str) -> f64 {
        // Simplified structural continuity based on sentence patterns
        let punct1 = sentence1
            .chars()
            .filter(|c| c.is_ascii_punctuation())
            .count();
        let punct2 = sentence2
            .chars()
            .filter(|c| c.is_ascii_punctuation())
            .count();

        let len1 = sentence1.split_whitespace().count();
        let len2 = sentence2.split_whitespace().count();

        let punct_similarity =
            1.0 - (punct1 as isize - punct2 as isize).abs() as f64 / (punct1 + punct2 + 1) as f64;
        let length_similarity =
            1.0 - (len1 as isize - len2 as isize).abs() as f64 / (len1 + len2 + 1) as f64;

        (punct_similarity + length_similarity) / 2.0
    }

    fn calculate_word_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        // Simple edit distance-based similarity
        let distance = self.levenshtein_distance(word1, word2);
        let max_len = word1.len().max(word2.len());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    fn calculate_sentence_level_global_coherence(
        &self,
        sentences: &[String],
    ) -> Result<f64, CoherenceMetricsError> {
        if sentences.len() < 3 {
            return Ok(1.0);
        }

        let mut global_coherence = 0.0;
        let mut comparisons = 0;

        for i in 0..sentences.len() {
            for j in (i + 2)..sentences.len().min(i + self.config.window_size + 2) {
                let coherence =
                    self.calculate_sentence_lexical_overlap(&sentences[i], &sentences[j]);
                let distance_penalty = 1.0 / (j - i) as f64;
                global_coherence += coherence * distance_penalty;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            Ok(global_coherence / comparisons as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_paragraph_level_global_coherence(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, CoherenceMetricsError> {
        if paragraphs.len() < 2 {
            return Ok(1.0);
        }

        let mut coherence_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..paragraphs.len() - 1 {
            let coherence =
                self.calculate_sentence_lexical_overlap(&paragraphs[i], &paragraphs[i + 1]);
            coherence_sum += coherence;
            comparisons += 1;
        }

        if comparisons > 0 {
            Ok(coherence_sum / comparisons as f64)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_document_level_coherence(
        &self,
        sentences: &[String],
        paragraphs: &[String],
    ) -> Result<f64, CoherenceMetricsError> {
        let sentence_count = sentences.len() as f64;
        let paragraph_count = paragraphs.len() as f64;

        let structure_score = if paragraph_count > 0.0 {
            (sentence_count / paragraph_count).min(10.0) / 10.0
        } else {
            0.0
        };

        Ok(structure_score)
    }

    fn calculate_median(&self, sorted_data: &[f64]) -> f64 {
        let len = sorted_data.len();
        if len % 2 == 0 {
            (sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0
        } else {
            sorted_data[len / 2]
        }
    }

    fn calculate_skewness(
        &self,
        data: &[f64],
        mean: f64,
        std_dev: f64,
    ) -> Result<f64, CoherenceMetricsError> {
        if std_dev == 0.0 {
            return Ok(0.0);
        }

        let skew_sum = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>();

        Ok(skew_sum / data.len() as f64)
    }

    fn calculate_kurtosis(
        &self,
        data: &[f64],
        mean: f64,
        std_dev: f64,
    ) -> Result<f64, CoherenceMetricsError> {
        if std_dev == 0.0 {
            return Ok(0.0);
        }

        let kurt_sum = data
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>();

        Ok((kurt_sum / data.len() as f64) - 3.0)
    }

    fn calculate_percentiles(
        &self,
        sorted_data: &[f64],
    ) -> Result<HashMap<u8, f64>, CoherenceMetricsError> {
        let mut percentiles = HashMap::new();
        let percentile_values = vec![10, 25, 50, 75, 90];

        for p in percentile_values {
            let index = ((p as f64 / 100.0) * (sorted_data.len() - 1) as f64) as usize;
            percentiles.insert(p, sorted_data[index.min(sorted_data.len() - 1)]);
        }

        Ok(percentiles)
    }

    fn detect_outliers(
        &self,
        data: &[f64],
        mean: f64,
        std_dev: f64,
    ) -> Result<Vec<usize>, CoherenceMetricsError> {
        let threshold = self.config.statistical_parameters.outlier_threshold;
        let outliers = data
            .iter()
            .enumerate()
            .filter(|(_, &value)| ((value - mean) / std_dev).abs() > threshold)
            .map(|(index, _)| index)
            .collect();

        Ok(outliers)
    }

    fn calculate_robust_statistics(
        &self,
        sorted_data: &[f64],
    ) -> Result<(f64, f64), CoherenceMetricsError> {
        if !self.config.statistical_parameters.use_robust_statistics {
            let mean = sorted_data.iter().sum::<f64>() / sorted_data.len() as f64;
            let variance = sorted_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / sorted_data.len() as f64;
            return Ok((mean, variance.sqrt()));
        }

        // Trimmed mean (remove 10% from each end)
        let trim_amount = (sorted_data.len() as f64 * 0.1) as usize;
        let trimmed_data = &sorted_data[trim_amount..sorted_data.len() - trim_amount];

        let robust_mean = trimmed_data.iter().sum::<f64>() / trimmed_data.len() as f64;

        // Median Absolute Deviation (MAD)
        let median = self.calculate_median(sorted_data);
        let mut deviations: Vec<f64> = sorted_data.iter().map(|x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = self.calculate_median(&deviations);
        let robust_std_dev = 1.4826 * mad; // Scale factor for normal distribution

        Ok((robust_mean, robust_std_dev))
    }

    fn calculate_histogram(
        &self,
        data: &[f64],
    ) -> Result<HashMap<String, usize>, CoherenceMetricsError> {
        let mut histogram = HashMap::new();
        let bin_count = 10;
        let min_val = data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let bin_width = (max_val - min_val) / bin_count as f64;

        for &value in data {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bin_count - 1);
            let bin_label = format!(
                "[{:.2}, {:.2})",
                min_val + bin_index as f64 * bin_width,
                min_val + (bin_index + 1) as f64 * bin_width
            );
            *histogram.entry(bin_label).or_insert(0) += 1;
        }

        Ok(histogram)
    }

    fn classify_distribution(
        &self,
        data: &[f64],
    ) -> Result<DistributionType, CoherenceMetricsError> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let median = {
            let mut sorted = data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.calculate_median(&sorted)
        };

        let skewness = self.calculate_skewness(
            data,
            mean,
            (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt(),
        )?;

        // Simple classification based on skewness and other properties
        if skewness.abs() < 0.5 && (mean - median).abs() < 0.1 {
            Ok(DistributionType::Normal)
        } else if skewness.abs() > 1.0 {
            Ok(DistributionType::Skewed)
        } else {
            Ok(DistributionType::Unknown)
        }
    }

    fn calculate_goodness_of_fit(
        &self,
        _data: &[f64],
        _dist_type: &DistributionType,
    ) -> Result<f64, CoherenceMetricsError> {
        // Simplified goodness of fit calculation
        // In a full implementation, this would use proper statistical tests
        Ok(0.8)
    }

    fn calculate_entropy(&self, histogram: &HashMap<String, usize>) -> f64 {
        let total: usize = histogram.values().sum();
        if total == 0 {
            return 0.0;
        }

        let entropy = histogram
            .values()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total as f64;
                -p * p.ln()
            })
            .sum();

        entropy
    }

    fn calculate_uniformity(&self, data: &[f64]) -> Result<f64, CoherenceMetricsError> {
        let histogram = self.calculate_histogram(data)?;
        let total: usize = histogram.values().sum();
        let expected_count = total as f64 / histogram.len() as f64;

        let chi_square: f64 = histogram
            .values()
            .map(|&count| (count as f64 - expected_count).powi(2) / expected_count)
            .sum();

        // Convert chi-square to uniformity score (0-1)
        Ok((-chi_square / total as f64).exp())
    }

    fn calculate_clustering_coefficient(&self, data: &[f64]) -> Result<f64, CoherenceMetricsError> {
        // Simplified clustering coefficient based on local variance
        if data.len() < 3 {
            return Ok(0.0);
        }

        let window_size = 3;
        let mut local_variances = Vec::new();

        for i in 0..=data.len().saturating_sub(window_size) {
            let window = &data[i..i + window_size];
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            local_variances.push(variance);
        }

        let avg_local_variance = local_variances.iter().sum::<f64>() / local_variances.len() as f64;
        Ok(1.0 - avg_local_variance.min(1.0))
    }

    fn calculate_distribution_summary(
        &self,
        data: &[f64],
    ) -> Result<DistributionSummary, CoherenceMetricsError> {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let range = sorted.last().unwrap() - sorted.first().unwrap();
        let q1_index = (0.25 * (sorted.len() - 1) as f64) as usize;
        let q3_index = (0.75 * (sorted.len() - 1) as f64) as usize;
        let interquartile_range = sorted[q3_index] - sorted[q1_index];

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let std_dev =
            (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
        let coefficient_of_variation = if mean != 0.0 { std_dev / mean } else { 0.0 };

        let histogram = self.calculate_histogram(data)?;
        let modal_class = histogram
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(key, _)| key.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        let concentration = 1.0 - (range / (data.len() as f64));

        Ok(DistributionSummary {
            range,
            interquartile_range,
            coefficient_of_variation,
            modal_class,
            concentration,
        })
    }

    fn calculate_reliability_score(
        &self,
        distribution: &[f64],
    ) -> Result<f64, CoherenceMetricsError> {
        if distribution.len() < 2 {
            return Ok(0.5);
        }

        let mean = distribution.iter().sum::<f64>() / distribution.len() as f64;
        let variance = distribution.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / distribution.len() as f64;

        // Reliability based on consistency (inverse of coefficient of variation)
        let cv = if mean != 0.0 {
            variance.sqrt() / mean
        } else {
            1.0
        };
        Ok((1.0 / (1.0 + cv)).min(1.0))
    }

    fn calculate_consistency_score(
        &self,
        transitions: &[f64],
    ) -> Result<f64, CoherenceMetricsError> {
        if transitions.is_empty() {
            return Ok(1.0);
        }

        let mean = transitions.iter().sum::<f64>() / transitions.len() as f64;
        let variance =
            transitions.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / transitions.len() as f64;

        Ok((1.0 - variance.sqrt()).max(0.0))
    }

    fn calculate_coverage_score(&self, sample_size: usize) -> Result<f64, CoherenceMetricsError> {
        // Coverage based on sample size adequacy
        let adequate_size = 30.0;
        Ok((sample_size as f64 / adequate_size).min(1.0))
    }

    fn calculate_stability_score(
        &self,
        distribution: &[f64],
    ) -> Result<f64, CoherenceMetricsError> {
        if distribution.len() < 4 {
            return Ok(0.5);
        }

        // Split distribution in half and compare means
        let mid = distribution.len() / 2;
        let first_half = &distribution[..mid];
        let second_half = &distribution[mid..];

        let mean1 = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let mean2 = second_half.iter().sum::<f64>() / second_half.len() as f64;

        Ok(1.0 - (mean1 - mean2).abs())
    }

    fn calculate_discriminant_validity(
        &self,
        overall_coherence: f64,
        distribution: &[f64],
    ) -> Result<f64, CoherenceMetricsError> {
        // Simplified discriminant validity based on score range utilization
        if distribution.is_empty() {
            return Ok(0.5);
        }

        let min_val = distribution
            .iter()
            .fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = distribution
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let range_utilization = max_val - min_val;

        Ok(range_utilization.min(1.0))
    }

    fn perform_significance_test(
        &self,
        data1: &[f64],
        data2: &[f64],
    ) -> Result<SignificanceTest, CoherenceMetricsError> {
        // Simplified t-test implementation
        if data1.len() < 2 || data2.len() < 2 {
            return Err(CoherenceMetricsError::InsufficientData {
                required: 2,
                found: data1.len().min(data2.len()),
            });
        }

        let mean1 = data1.iter().sum::<f64>() / data1.len() as f64;
        let mean2 = data2.iter().sum::<f64>() / data2.len() as f64;

        let var1 =
            data1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (data1.len() - 1) as f64;
        let var2 =
            data2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (data2.len() - 1) as f64;

        let pooled_se = ((var1 / data1.len() as f64) + (var2 / data2.len() as f64)).sqrt();
        let t_statistic = if pooled_se != 0.0 {
            (mean1 - mean2) / pooled_se
        } else {
            0.0
        };

        // Simplified p-value calculation (normally would use proper t-distribution)
        let p_value = 2.0 * (1.0 - (1.0 / (1.0 + t_statistic.abs())));
        let is_significant = p_value < 0.05;

        Ok(SignificanceTest {
            test_statistic: t_statistic,
            p_value,
            is_significant,
            test_type: "Two-sample t-test".to_string(),
        })
    }

    fn calculate_effect_size(
        &self,
        data1: &[f64],
        data2: &[f64],
    ) -> Result<f64, CoherenceMetricsError> {
        if data1.is_empty() || data2.is_empty() {
            return Ok(0.0);
        }

        let mean1 = data1.iter().sum::<f64>() / data1.len() as f64;
        let mean2 = data2.iter().sum::<f64>() / data2.len() as f64;

        let var1 = data1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / data1.len() as f64;
        let var2 = data2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / data2.len() as f64;

        let pooled_std = ((var1 + var2) / 2.0).sqrt();

        if pooled_std != 0.0 {
            Ok((mean1 - mean2) / pooled_std)
        } else {
            Ok(0.0)
        }
    }
}

/// Simple coherence calculation functions for convenience
pub fn calculate_overall_coherence_simple(text: &str) -> Result<f64, CoherenceMetricsError> {
    // This would require the full analyzer - simplified version
    let sentences: Vec<String> = text
        .split(". ")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if sentences.is_empty() {
        return Ok(0.0);
    }

    let calculator = CoherenceMetricsCalculator::with_default_config();
    calculator.calculate_local_coherence(&sentences)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_calculator_creation() {
        let calculator = CoherenceMetricsCalculator::with_default_config();
        assert_eq!(calculator.config.window_size, 5);
        assert_eq!(calculator.config.confidence_threshold, 0.7);
    }

    #[test]
    fn test_coherence_distribution_calculation() -> Result<(), CoherenceMetricsError> {
        let calculator = CoherenceMetricsCalculator::with_default_config();
        let sentences = vec![
            "The cat sat on the mat.".to_string(),
            "The cat was comfortable.".to_string(),
            "It decided to sleep.".to_string(),
            "The weather was nice.".to_string(),
            "Everyone enjoyed the day.".to_string(),
            "The sun was shining.".to_string(),
        ];

        let distribution = calculator.calculate_coherence_distribution(&sentences)?;
        assert!(distribution.len() > 0);
        assert!(distribution
            .iter()
            .all(|&score| score >= 0.0 && score <= 1.0));

        Ok(())
    }

    #[test]
    fn test_confidence_score_calculation() -> Result<(), CoherenceMetricsError> {
        let calculator = CoherenceMetricsCalculator::with_default_config();
        let distribution = vec![0.8, 0.7, 0.9, 0.6, 0.8, 0.7];

        let confidence = calculator.calculate_confidence_score(0.75, &distribution)?;
        assert!(confidence >= 0.0 && confidence <= 1.0);

        Ok(())
    }

    #[test]
    fn test_statistical_analysis() -> Result<(), CoherenceMetricsError> {
        let calculator = CoherenceMetricsCalculator::with_default_config();
        let distribution = vec![0.2, 0.4, 0.6, 0.8, 0.5, 0.7, 0.3, 0.9, 0.1, 0.6];

        let analysis = calculator.perform_statistical_analysis(&distribution)?;
        assert!(analysis.mean > 0.0);
        assert!(analysis.std_dev >= 0.0);
        assert!(analysis.percentiles.contains_key(&50));

        Ok(())
    }

    #[test]
    fn test_local_coherence_calculation() -> Result<(), CoherenceMetricsError> {
        let calculator = CoherenceMetricsCalculator::with_default_config();
        let sentences = vec![
            "The dog ran quickly.".to_string(),
            "The quick dog ran.".to_string(),
            "Dogs run very fast.".to_string(),
        ];

        let local_coherence = calculator.calculate_local_coherence(&sentences)?;
        assert!(local_coherence >= 0.0 && local_coherence <= 1.0);

        Ok(())
    }

    #[test]
    fn test_simple_coherence_calculation() -> Result<(), CoherenceMetricsError> {
        let text = "The cat sat on the mat. The cat was comfortable. It decided to sleep.";
        let coherence = calculate_overall_coherence_simple(text)?;
        assert!(coherence >= 0.0 && coherence <= 1.0);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let calculator = CoherenceMetricsCalculator::with_default_config();

        // Test insufficient data error
        let empty_sentences: Vec<String> = vec![];
        let result = calculator.calculate_coherence_distribution(&empty_sentences);
        assert!(matches!(
            result,
            Err(CoherenceMetricsError::InsufficientData { .. })
        ));
    }

    #[test]
    fn test_lexical_overlap_calculation() {
        let calculator = CoherenceMetricsCalculator::with_default_config();

        let overlap1 = calculator.calculate_sentence_lexical_overlap(
            "The cat sat on the mat",
            "The cat was on the floor",
        );
        assert!(overlap1 > 0.0);

        let overlap2 = calculator.calculate_sentence_lexical_overlap("Hello world", "Goodbye moon");
        assert!(overlap2 >= 0.0);
    }

    #[test]
    fn test_distribution_analysis() -> Result<(), CoherenceMetricsError> {
        let calculator = CoherenceMetricsCalculator::with_default_config();
        let distribution = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        let analysis = calculator.perform_distribution_analysis(&distribution)?;
        assert!(!analysis.histogram.is_empty());
        assert!(analysis.entropy >= 0.0);
        assert!(analysis.uniformity >= 0.0 && analysis.uniformity <= 1.0);

        Ok(())
    }
}
