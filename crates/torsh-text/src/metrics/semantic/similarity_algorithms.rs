//! Similarity Algorithms Module
//!
//! This module provides comprehensive similarity computation algorithms for semantic analysis.
//! It implements multiple similarity approaches including cosine similarity, Jaccard similarity,
//! soft semantic matching, weighted similarity, hierarchical analysis, and contextual similarity.
//!
//! # Algorithm Categories
//!
//! ## Vector-Based Algorithms
//! - **Cosine Similarity**: Measures angle between feature vectors
//! - **Weighted Similarity**: Importance-weighted semantic matching
//!
//! ## Set-Based Algorithms
//! - **Jaccard Similarity**: Set-based concept overlap measurement
//! - **Soft Semantic**: Fuzzy matching with concept expansion
//!
//! ## Advanced Algorithms
//! - **Hierarchical**: Multi-level semantic analysis
//! - **Contextual**: Context-aware similarity with disambiguation
//!
//! # Usage Examples
//!
//! ```rust
//! use torsh_text::metrics::semantic::similarity_algorithms::{SimilarityAlgorithmEngine, SimilarityAlgorithm};
//!
//! let engine = SimilarityAlgorithmEngine::new();
//! let similarity = engine.compute_similarity(
//!     SimilarityAlgorithm::Cosine,
//!     &features1,
//!     &features2
//! )?;
//! ```

use std::collections::{HashMap, HashSet};
use thiserror::Error;

use super::feature_extraction::SemanticFeatureVector;

/// Errors that can occur during similarity computation
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SimilarityAlgorithmError {
    #[error("Invalid feature vectors: {message}")]
    InvalidFeatures { message: String },
    #[error("Algorithm computation failed: {algorithm} - {reason}")]
    ComputationError { algorithm: String, reason: String },
    #[error("Unsupported algorithm: {algorithm}")]
    UnsupportedAlgorithm { algorithm: String },
    #[error("Feature vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Available similarity algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimilarityAlgorithm {
    /// Cosine similarity on semantic feature vectors
    Cosine,
    /// Jaccard similarity on concept sets
    Jaccard,
    /// Soft semantic similarity with fuzzy matching
    SoftSemantic,
    /// Weighted similarity based on word importance
    Weighted,
    /// Hierarchical similarity at multiple levels
    Hierarchical,
    /// Contextual similarity with disambiguation
    Contextual,
}

impl std::fmt::Display for SimilarityAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimilarityAlgorithm::Cosine => write!(f, "Cosine"),
            SimilarityAlgorithm::Jaccard => write!(f, "Jaccard"),
            SimilarityAlgorithm::SoftSemantic => write!(f, "SoftSemantic"),
            SimilarityAlgorithm::Weighted => write!(f, "Weighted"),
            SimilarityAlgorithm::Hierarchical => write!(f, "Hierarchical"),
            SimilarityAlgorithm::Contextual => write!(f, "Contextual"),
        }
    }
}

/// Configuration for similarity algorithm computation
#[derive(Debug, Clone)]
pub struct SimilarityAlgorithmConfig {
    pub jaccard_threshold: f64,
    pub soft_semantic_domain_weight: f64,
    pub hierarchical_level_weights: Vec<f64>,
    pub contextual_domain_boost: f64,
    pub weighted_normalization: bool,
}

impl Default for SimilarityAlgorithmConfig {
    fn default() -> Self {
        Self {
            jaccard_threshold: 0.1,
            soft_semantic_domain_weight: 0.5,
            hierarchical_level_weights: vec![0.4, 0.3, 0.3],
            contextual_domain_boost: 0.1,
            weighted_normalization: true,
        }
    }
}

/// Result of similarity computation with detailed breakdown
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityResult {
    pub similarity: f64,
    pub algorithm: SimilarityAlgorithm,
    pub confidence: f64,
    pub component_scores: HashMap<String, f64>,
    pub metadata: SimilarityMetadata,
}

/// Metadata about similarity computation
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityMetadata {
    pub feature_dimensions: usize,
    pub shared_domains: usize,
    pub computation_time_ms: u64,
    pub normalization_applied: bool,
}

/// Main engine for similarity algorithm computation
pub struct SimilarityAlgorithmEngine {
    config: SimilarityAlgorithmConfig,
    domain_vocabularies: HashMap<String, HashSet<String>>,
}

impl SimilarityAlgorithmEngine {
    /// Create new similarity algorithm engine with default configuration
    pub fn new() -> Self {
        Self {
            config: SimilarityAlgorithmConfig::default(),
            domain_vocabularies: Self::initialize_domain_vocabularies(),
        }
    }

    /// Create engine with custom configuration
    pub fn with_config(config: SimilarityAlgorithmConfig) -> Self {
        Self {
            config,
            domain_vocabularies: Self::initialize_domain_vocabularies(),
        }
    }

    /// Compute similarity using specified algorithm
    pub fn compute_similarity(
        &self,
        algorithm: SimilarityAlgorithm,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<f64, SimilarityAlgorithmError> {
        let start_time = std::time::Instant::now();

        // Validate input features
        self.validate_features(features1, features2)?;

        let similarity = match algorithm {
            SimilarityAlgorithm::Cosine => {
                self.compute_cosine_similarity(&features1.features, &features2.features)?
            }
            SimilarityAlgorithm::Jaccard => {
                self.compute_jaccard_similarity(features1, features2)?
            }
            SimilarityAlgorithm::SoftSemantic => {
                self.compute_soft_semantic_similarity(features1, features2)?
            }
            SimilarityAlgorithm::Weighted => {
                self.compute_weighted_similarity(features1, features2)?
            }
            SimilarityAlgorithm::Hierarchical => {
                self.compute_hierarchical_similarity(features1, features2)?
            }
            SimilarityAlgorithm::Contextual => {
                self.compute_contextual_similarity(features1, features2)?
            }
        };

        Ok(similarity)
    }

    /// Compute detailed similarity analysis with breakdown
    pub fn analyze_similarity(
        &self,
        algorithm: SimilarityAlgorithm,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<SimilarityResult, SimilarityAlgorithmError> {
        let start_time = std::time::Instant::now();

        // Validate input features
        self.validate_features(features1, features2)?;

        let similarity = self.compute_similarity(algorithm, features1, features2)?;
        let confidence = self.calculate_confidence(algorithm, similarity, features1, features2)?;
        let component_scores = self.compute_component_breakdown(algorithm, features1, features2)?;

        let shared_domains = self.count_shared_domains(features1, features2);
        let computation_time = start_time.elapsed().as_millis() as u64;

        Ok(SimilarityResult {
            similarity,
            algorithm,
            confidence,
            component_scores,
            metadata: SimilarityMetadata {
                feature_dimensions: features1.features.len(),
                shared_domains,
                computation_time_ms: computation_time,
                normalization_applied: true,
            },
        })
    }

    /// Compare multiple algorithms on the same feature pairs
    pub fn compare_algorithms(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
        algorithms: &[SimilarityAlgorithm],
    ) -> Result<HashMap<SimilarityAlgorithm, f64>, SimilarityAlgorithmError> {
        let mut results = HashMap::new();

        for &algorithm in algorithms {
            let similarity = self.compute_similarity(algorithm, features1, features2)?;
            results.insert(algorithm, similarity);
        }

        Ok(results)
    }

    // Individual similarity algorithm implementations

    /// Compute cosine similarity between feature vectors
    fn compute_cosine_similarity(
        &self,
        vec1: &[f64],
        vec2: &[f64],
    ) -> Result<f64, SimilarityAlgorithmError> {
        if vec1.len() != vec2.len() {
            return Err(SimilarityAlgorithmError::DimensionMismatch {
                expected: vec1.len(),
                actual: vec2.len(),
            });
        }

        if vec1.is_empty() {
            return Ok(1.0); // Empty vectors are considered identical
        }

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            Ok(0.0)
        } else {
            let similarity = (dot_product / (norm1 * norm2)).max(0.0).min(1.0);
            Ok(similarity)
        }
    }

    /// Compute Jaccard similarity on semantic features
    fn compute_jaccard_similarity(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<f64, SimilarityAlgorithmError> {
        let threshold = self.config.jaccard_threshold;

        // Convert features to binary indicators
        let set1: HashSet<usize> = features1
            .features
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > threshold)
            .map(|(i, _)| i)
            .collect();

        let set2: HashSet<usize> = features2
            .features
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > threshold)
            .map(|(i, _)| i)
            .collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            Ok(1.0) // Both sets empty - perfect match
        } else {
            Ok(intersection as f64 / union as f64)
        }
    }

    /// Compute soft semantic similarity with fuzzy matching
    fn compute_soft_semantic_similarity(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<f64, SimilarityAlgorithmError> {
        let mut similarity = 0.0;
        let mut total_weight = 0.0;

        // Domain similarity with soft matching
        let domain_weight = self.config.soft_semantic_domain_weight;
        for domain in &features1.domain_scores {
            let score1 = *domain.1;
            let score2 = features2.domain_scores.get(domain.0).unwrap_or(&0.0);

            let domain_sim = 1.0 - (score1 - score2).abs();
            let weight = (score1 + score2) / 2.0 + 0.1; // Ensure non-zero weight

            similarity += domain_sim * weight * domain_weight;
            total_weight += weight * domain_weight;
        }

        // Feature vector similarity
        let feature_sim =
            self.compute_cosine_similarity(&features1.features, &features2.features)?;
        similarity += feature_sim * (1.0 - domain_weight);
        total_weight += 1.0 - domain_weight;

        if total_weight == 0.0 {
            Ok(0.0)
        } else {
            Ok(similarity / total_weight)
        }
    }

    /// Compute weighted similarity based on word importance
    fn compute_weighted_similarity(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<f64, SimilarityAlgorithmError> {
        let mut weighted_similarity = 0.0;
        let mut total_weight = 0.0;

        // Weight by word importance
        for (word, &weight1) in &features1.word_weights {
            if let Some(&weight2) = features2.word_weights.get(word) {
                let avg_weight = (weight1 + weight2) / 2.0;
                weighted_similarity += avg_weight;
                total_weight += avg_weight;
            }
        }

        // Add base cosine similarity
        let cosine_sim =
            self.compute_cosine_similarity(&features1.features, &features2.features)?;
        let base_weight = total_weight.max(1.0);
        weighted_similarity += cosine_sim * base_weight;
        total_weight += base_weight;

        if total_weight == 0.0 {
            Ok(0.0)
        } else {
            Ok(weighted_similarity / total_weight)
        }
    }

    /// Compute hierarchical similarity at multiple levels
    fn compute_hierarchical_similarity(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<f64, SimilarityAlgorithmError> {
        let mut similarities = Vec::new();

        // Feature-level similarity
        let feature_sim =
            self.compute_cosine_similarity(&features1.features, &features2.features)?;
        similarities.push(feature_sim);

        // Domain-level similarity
        let mut domain_similarities = Vec::new();
        for domain in &features1.domain_scores {
            let score1 = *domain.1;
            let score2 = features2.domain_scores.get(domain.0).unwrap_or(&0.0);

            if score1 + score2 > 0.0 {
                // Harmonic mean for domain similarity
                let domain_sim =
                    2.0 * (score1 * score2) / (score1 * score1 + score2 * score2 + 1e-8);
                domain_similarities.push(domain_sim);
            }
        }

        if !domain_similarities.is_empty() {
            let avg_domain_sim =
                domain_similarities.iter().sum::<f64>() / domain_similarities.len() as f64;
            similarities.push(avg_domain_sim);
        }

        // Topic-level similarity (if available)
        if let (Some(topics1), Some(topics2)) =
            (&features1.topic_distribution, &features2.topic_distribution)
        {
            let topic_sim = self.compute_cosine_similarity(topics1, topics2)?;
            similarities.push(topic_sim);
        }

        // Weight similarities by configured weights
        let weights = &self.config.hierarchical_level_weights;
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, &sim) in similarities.iter().enumerate() {
            let weight = weights.get(i).copied().unwrap_or(1.0);
            weighted_sum += sim * weight;
            weight_sum += weight;
        }

        if weight_sum == 0.0 {
            Ok(0.0)
        } else {
            Ok(weighted_sum / weight_sum)
        }
    }

    /// Compute contextual similarity with disambiguation
    fn compute_contextual_similarity(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<f64, SimilarityAlgorithmError> {
        // Base similarity using cosine
        let base_sim = self.compute_cosine_similarity(&features1.features, &features2.features)?;

        // Context adjustment based on domain overlap
        let mut context_factor = 1.0;
        let shared_domains = self.count_shared_domains(features1, features2);

        if shared_domains > 0 {
            context_factor += self.config.contextual_domain_boost * shared_domains as f64;
        }

        // Sentiment context adjustment
        if let (Some(sent1), Some(sent2)) =
            (&features1.sentiment_scores, &features2.sentiment_scores)
        {
            let sentiment_alignment = self.compute_sentiment_similarity(sent1, sent2);
            context_factor += sentiment_alignment * 0.1; // Small boost for sentiment alignment
        }

        Ok((base_sim * context_factor).min(1.0))
    }

    // Helper methods

    fn validate_features(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<(), SimilarityAlgorithmError> {
        if features1.features.is_empty() || features2.features.is_empty() {
            return Err(SimilarityAlgorithmError::InvalidFeatures {
                message: "Feature vectors cannot be empty".to_string(),
            });
        }

        if features1.features.len() != features2.features.len() {
            return Err(SimilarityAlgorithmError::DimensionMismatch {
                expected: features1.features.len(),
                actual: features2.features.len(),
            });
        }

        Ok(())
    }

    fn calculate_confidence(
        &self,
        algorithm: SimilarityAlgorithm,
        similarity: f64,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<f64, SimilarityAlgorithmError> {
        let mut confidence_factors = Vec::new();

        // Feature vector magnitude confidence
        let norm1: f64 = features1.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = features2.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude_confidence = ((norm1 * norm2).ln() + 1.0).min(1.0).max(0.0);
        confidence_factors.push(magnitude_confidence);

        // Domain overlap confidence
        let shared_domains = self.count_shared_domains(features1, features2);
        let total_domains = (features1.domain_scores.len() + features2.domain_scores.len()).max(1);
        let domain_confidence = (2.0 * shared_domains as f64 / total_domains as f64).min(1.0);
        confidence_factors.push(domain_confidence);

        // Algorithm-specific confidence adjustments
        match algorithm {
            SimilarityAlgorithm::Cosine => {
                // High confidence for extreme values
                confidence_factors.push(if similarity > 0.8 || similarity < 0.2 {
                    0.9
                } else {
                    0.6
                });
            }
            SimilarityAlgorithm::Jaccard => {
                // Confidence based on set sizes
                let set_size_factor =
                    (features1.word_weights.len() + features2.word_weights.len()) as f64 / 20.0;
                confidence_factors.push(set_size_factor.min(1.0));
            }
            _ => {
                confidence_factors.push(0.7); // Default confidence for other algorithms
            }
        }

        // Average confidence factors
        let confidence = confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64;
        Ok(confidence.max(0.0).min(1.0))
    }

    fn compute_component_breakdown(
        &self,
        algorithm: SimilarityAlgorithm,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> Result<HashMap<String, f64>, SimilarityAlgorithmError> {
        let mut breakdown = HashMap::new();

        // Always include base feature similarity
        let feature_sim =
            self.compute_cosine_similarity(&features1.features, &features2.features)?;
        breakdown.insert("feature_similarity".to_string(), feature_sim);

        // Domain similarities
        for domain in &features1.domain_scores {
            let score1 = *domain.1;
            let score2 = features2.domain_scores.get(domain.0).unwrap_or(&0.0);
            let domain_sim = 1.0 - (score1 - score2).abs();
            breakdown.insert(format!("domain_{}", domain.0), domain_sim);
        }

        // Algorithm-specific components
        match algorithm {
            SimilarityAlgorithm::Weighted => {
                let word_overlap = self.compute_word_overlap_ratio(features1, features2);
                breakdown.insert("word_overlap".to_string(), word_overlap);
            }
            SimilarityAlgorithm::Hierarchical => {
                if let (Some(topics1), Some(topics2)) =
                    (&features1.topic_distribution, &features2.topic_distribution)
                {
                    let topic_sim = self.compute_cosine_similarity(topics1, topics2)?;
                    breakdown.insert("topic_similarity".to_string(), topic_sim);
                }
            }
            _ => {}
        }

        Ok(breakdown)
    }

    fn count_shared_domains(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> usize {
        features1
            .domain_scores
            .keys()
            .filter(|domain| {
                let score1 = features1.domain_scores.get(*domain).unwrap_or(&0.0);
                let score2 = features2.domain_scores.get(*domain).unwrap_or(&0.0);
                *score1 > 0.1 && *score2 > 0.1
            })
            .count()
    }

    fn compute_word_overlap_ratio(
        &self,
        features1: &SemanticFeatureVector,
        features2: &SemanticFeatureVector,
    ) -> f64 {
        let words1: HashSet<_> = features1.word_weights.keys().collect();
        let words2: HashSet<_> = features2.word_weights.keys().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn compute_sentiment_similarity(
        &self,
        sentiment1: &HashMap<String, f64>,
        sentiment2: &HashMap<String, f64>,
    ) -> f64 {
        let pos1 = sentiment1.get("positive").unwrap_or(&0.0);
        let pos2 = sentiment2.get("positive").unwrap_or(&0.0);
        let neg1 = sentiment1.get("negative").unwrap_or(&0.0);
        let neg2 = sentiment2.get("negative").unwrap_or(&0.0);

        let pos_sim = 1.0 - (pos1 - pos2).abs();
        let neg_sim = 1.0 - (neg1 - neg2).abs();

        (pos_sim + neg_sim) / 2.0
    }

    fn initialize_domain_vocabularies() -> HashMap<String, HashSet<String>> {
        let mut domains = HashMap::new();

        // Technology domain
        let tech_words: HashSet<String> = vec![
            "computer",
            "software",
            "algorithm",
            "data",
            "network",
            "internet",
            "programming",
            "code",
            "system",
            "application",
            "digital",
            "technology",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        domains.insert("Technology".to_string(), tech_words);

        // Science domain
        let science_words: HashSet<String> = vec![
            "research",
            "study",
            "experiment",
            "analysis",
            "hypothesis",
            "theory",
            "method",
            "result",
            "conclusion",
            "evidence",
            "observation",
            "measurement",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        domains.insert("Science".to_string(), science_words);

        // Business domain
        let business_words: HashSet<String> = vec![
            "company",
            "market",
            "profit",
            "customer",
            "product",
            "service",
            "strategy",
            "business",
            "management",
            "revenue",
            "growth",
            "investment",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        domains.insert("Business".to_string(), business_words);

        domains
    }
}

impl Default for SimilarityAlgorithmEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for simple similarity computation

/// Compute cosine similarity between two feature vectors
pub fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> Result<f64, SimilarityAlgorithmError> {
    let engine = SimilarityAlgorithmEngine::new();
    engine.compute_cosine_similarity(vec1, vec2)
}

/// Compute Jaccard similarity between two feature vectors
pub fn jaccard_similarity(
    features1: &SemanticFeatureVector,
    features2: &SemanticFeatureVector,
) -> Result<f64, SimilarityAlgorithmError> {
    let engine = SimilarityAlgorithmEngine::new();
    engine.compute_jaccard_similarity(features1, features2)
}

/// Compute soft semantic similarity
pub fn soft_semantic_similarity(
    features1: &SemanticFeatureVector,
    features2: &SemanticFeatureVector,
) -> Result<f64, SimilarityAlgorithmError> {
    let engine = SimilarityAlgorithmEngine::new();
    engine.compute_soft_semantic_similarity(features1, features2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_feature_vector(
        features: Vec<f64>,
        domains: Vec<(&str, f64)>,
    ) -> SemanticFeatureVector {
        let mut domain_scores = HashMap::new();
        for (domain, score) in domains {
            domain_scores.insert(domain.to_string(), score);
        }

        SemanticFeatureVector {
            features,
            domain_scores,
            word_weights: HashMap::new(),
            sentiment_scores: None,
            topic_distribution: None,
        }
    }

    #[test]
    fn test_cosine_similarity() -> Result<(), SimilarityAlgorithmError> {
        let engine = SimilarityAlgorithmEngine::new();

        // Test identical vectors
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let similarity = engine.compute_cosine_similarity(&vec1, &vec2)?;
        assert!((similarity - 1.0).abs() < 1e-10);

        // Test orthogonal vectors
        let vec1 = vec![1.0, 0.0];
        let vec2 = vec![0.0, 1.0];
        let similarity = engine.compute_cosine_similarity(&vec1, &vec2)?;
        assert!((similarity - 0.0).abs() < 1e-10);

        // Test similar vectors
        let vec1 = vec![1.0, 1.0];
        let vec2 = vec![1.0, 0.8];
        let similarity = engine.compute_cosine_similarity(&vec1, &vec2)?;
        assert!(similarity > 0.8);

        Ok(())
    }

    #[test]
    fn test_jaccard_similarity() -> Result<(), SimilarityAlgorithmError> {
        let engine = SimilarityAlgorithmEngine::new();

        let features1 =
            create_test_feature_vector(vec![0.5, 0.2, 0.0, 0.8], vec![("Technology", 0.7)]);

        let features2 =
            create_test_feature_vector(vec![0.6, 0.0, 0.1, 0.7], vec![("Technology", 0.8)]);

        let similarity = engine.compute_jaccard_similarity(&features1, &features2)?;
        assert!(similarity >= 0.0 && similarity <= 1.0);

        Ok(())
    }

    #[test]
    fn test_soft_semantic_similarity() -> Result<(), SimilarityAlgorithmError> {
        let engine = SimilarityAlgorithmEngine::new();

        let features1 =
            create_test_feature_vector(vec![1.0, 0.5], vec![("Technology", 0.8), ("Science", 0.3)]);

        let features2 =
            create_test_feature_vector(vec![0.8, 0.6], vec![("Technology", 0.7), ("Science", 0.4)]);

        let similarity = engine.compute_soft_semantic_similarity(&features1, &features2)?;
        assert!(similarity >= 0.0 && similarity <= 1.0);

        Ok(())
    }

    #[test]
    fn test_algorithm_comparison() -> Result<(), SimilarityAlgorithmError> {
        let engine = SimilarityAlgorithmEngine::new();

        let features1 = create_test_feature_vector(vec![1.0, 0.5, 0.3], vec![("Technology", 0.8)]);

        let features2 = create_test_feature_vector(vec![0.8, 0.6, 0.4], vec![("Technology", 0.7)]);

        let algorithms = vec![
            SimilarityAlgorithm::Cosine,
            SimilarityAlgorithm::Jaccard,
            SimilarityAlgorithm::SoftSemantic,
        ];

        let results = engine.compare_algorithms(&features1, &features2, &algorithms)?;

        assert_eq!(results.len(), 3);
        for (_, similarity) in results {
            assert!(similarity >= 0.0 && similarity <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_similarity_analysis() -> Result<(), SimilarityAlgorithmError> {
        let engine = SimilarityAlgorithmEngine::new();

        let features1 = create_test_feature_vector(vec![1.0, 0.5], vec![("Technology", 0.8)]);

        let features2 = create_test_feature_vector(vec![0.9, 0.6], vec![("Technology", 0.7)]);

        let result =
            engine.analyze_similarity(SimilarityAlgorithm::Cosine, &features1, &features2)?;

        assert!(result.similarity >= 0.0 && result.similarity <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert_eq!(result.algorithm, SimilarityAlgorithm::Cosine);
        assert!(result.metadata.feature_dimensions > 0);
        assert!(!result.component_scores.is_empty());

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let engine = SimilarityAlgorithmEngine::new();

        // Test dimension mismatch
        let features1 = create_test_feature_vector(vec![1.0, 0.5], vec![]);
        let features2 = create_test_feature_vector(vec![1.0, 0.5, 0.3], vec![]);

        let result = engine.compute_similarity(SimilarityAlgorithm::Cosine, &features1, &features2);

        assert!(matches!(
            result,
            Err(SimilarityAlgorithmError::DimensionMismatch { .. })
        ));

        // Test empty features
        let empty_features = create_test_feature_vector(vec![], vec![]);
        let result =
            engine.compute_similarity(SimilarityAlgorithm::Cosine, &empty_features, &features1);

        assert!(matches!(
            result,
            Err(SimilarityAlgorithmError::InvalidFeatures { .. })
        ));
    }

    #[test]
    fn test_convenience_functions() -> Result<(), SimilarityAlgorithmError> {
        let vec1 = vec![1.0, 0.5, 0.3];
        let vec2 = vec![0.8, 0.6, 0.4];

        let similarity = cosine_similarity(&vec1, &vec2)?;
        assert!(similarity >= 0.0 && similarity <= 1.0);

        let features1 = create_test_feature_vector(vec1, vec![("Technology", 0.8)]);
        let features2 = create_test_feature_vector(vec2, vec![("Technology", 0.7)]);

        let jaccard_sim = jaccard_similarity(&features1, &features2)?;
        assert!(jaccard_sim >= 0.0 && jaccard_sim <= 1.0);

        let soft_sim = soft_semantic_similarity(&features1, &features2)?;
        assert!(soft_sim >= 0.0 && soft_sim <= 1.0);

        Ok(())
    }
}
