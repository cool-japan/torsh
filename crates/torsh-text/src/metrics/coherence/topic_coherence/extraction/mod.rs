//! Topic extraction algorithms module
//!
//! This module provides a clean interface for different topic extraction algorithms.
//! Each algorithm implements the TopicExtractor trait, allowing for easy swapping
//! and configuration of different extraction approaches.

use crate::metrics::coherence::topic_coherence::{
    config::{TopicExtractionConfig, TopicModelingApproach},
    results::{SemanticProfile, Topic, TopicEvolution, TopicQualityMetrics, TopicRelationship},
    similarity::SimilarityCalculator,
};
use std::collections::{HashMap, HashSet};

pub mod cooccurrence;
pub mod dynamic;
pub mod hierarchical;
pub mod keyword_clustering;
pub mod latent_semantic;
pub mod tfidf;

/// Error types for topic extraction
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("Insufficient content for topic extraction")]
    InsufficientContent,
    #[error("Algorithm error: {0}")]
    AlgorithmError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Trait for topic extraction algorithms
pub trait TopicExtractor: Send + Sync {
    /// Extract topics from the given sentences
    fn extract_topics(&self, sentences: &[String]) -> Result<Vec<Topic>, ExtractionError>;

    /// Get the name of the extraction algorithm
    fn algorithm_name(&self) -> &'static str;

    /// Get algorithm-specific parameters
    fn get_parameters(&self) -> HashMap<String, String>;

    /// Validate that the algorithm can process the given content
    fn validate_content(&self, sentences: &[String]) -> Result<(), ExtractionError> {
        if sentences.is_empty() {
            return Err(ExtractionError::InsufficientContent);
        }

        let total_words: usize = sentences.iter().map(|s| s.split_whitespace().count()).sum();

        if total_words < 10 {
            return Err(ExtractionError::InsufficientContent);
        }

        Ok(())
    }

    /// Post-process topics with common enhancements
    fn post_process_topics(&self, mut topics: Vec<Topic>, sentences: &[String]) -> Vec<Topic> {
        // Add hierarchical levels based on topic prominence
        topics.sort_by(|a, b| {
            b.prominence
                .partial_cmp(&a.prominence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, topic) in topics.iter_mut().enumerate() {
            topic.hierarchical_level = if i < 3 { 0 } else { 1 };
        }

        // Add basic relationships between topics
        for i in 0..topics.len() {
            let mut relationships = Vec::new();
            for j in 0..topics.len() {
                if i != j {
                    let similarity = self.calculate_topic_similarity(&topics[i], &topics[j]);
                    if similarity > 0.3 {
                        relationships.push(TopicRelationship {
                            related_topic_id: topics[j].topic_id.clone(),
                            relationship_type: "semantic".to_string(),
                            strength: similarity,
                            confidence: similarity * 0.8,
                            directionality: None,
                        });
                    }
                }
            }
            topics[i].relationships = relationships;
        }

        topics
    }

    /// Calculate similarity between two topics
    fn calculate_topic_similarity(&self, topic1: &Topic, topic2: &Topic) -> f64 {
        let mut common_keywords = 0;
        let mut total_keywords = HashSet::new();

        for keyword in &topic1.keywords {
            total_keywords.insert(keyword);
            if topic2.keywords.contains(keyword) {
                common_keywords += 1;
            }
        }

        for keyword in &topic2.keywords {
            total_keywords.insert(keyword);
        }

        if total_keywords.is_empty() {
            0.0
        } else {
            common_keywords as f64 / total_keywords.len() as f64
        }
    }
}

/// Topic extraction factory for creating appropriate extractors
pub struct TopicExtractionFactory;

impl TopicExtractionFactory {
    /// Create a topic extractor based on configuration
    pub fn create_extractor(
        config: &TopicExtractionConfig,
        similarity_calculator: SimilarityCalculator,
    ) -> Box<dyn TopicExtractor> {
        match config.approach {
            TopicModelingApproach::KeywordClustering => {
                Box::new(keyword_clustering::KeywordClusteringExtractor::new(
                    config.clone(),
                    similarity_calculator,
                ))
            }
            TopicModelingApproach::TfIdf => Box::new(tfidf::TfIdfExtractor::new(
                config.clone(),
                similarity_calculator,
            )),
            TopicModelingApproach::LatentSemantic => {
                Box::new(latent_semantic::LatentSemanticExtractor::new(
                    config.clone(),
                    similarity_calculator,
                ))
            }
            TopicModelingApproach::CoOccurrence => Box::new(
                cooccurrence::CoOccurrenceExtractor::new(config.clone(), similarity_calculator),
            ),
            TopicModelingApproach::Hierarchical => Box::new(
                hierarchical::HierarchicalExtractor::new(config.clone(), similarity_calculator),
            ),
            TopicModelingApproach::Dynamic => Box::new(dynamic::DynamicExtractor::new(
                config.clone(),
                similarity_calculator,
            )),
        }
    }
}

/// Utilities shared across extraction algorithms
pub struct ExtractionUtils;

impl ExtractionUtils {
    /// Extract content words from sentences (remove stopwords)
    pub fn extract_content_words(sentences: &[String]) -> Vec<String> {
        let stopwords = Self::get_stopwords();
        let mut content_words = Vec::new();

        for sentence in sentences {
            for word in sentence.split_whitespace() {
                let word_lower = word
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .collect::<String>();

                if word_lower.len() >= 3 && !stopwords.contains(&word_lower) {
                    content_words.push(word_lower);
                }
            }
        }

        content_words
    }

    /// Build a basic semantic profile for keywords
    pub fn build_semantic_profile(keywords: &[String]) -> SemanticProfile {
        let semantic_fields = Self::identify_semantic_fields(keywords);
        let conceptual_clusters = Self::create_conceptual_clusters(keywords);
        let semantic_coherence = Self::calculate_semantic_coherence(keywords);

        SemanticProfile {
            semantic_fields,
            conceptual_clusters,
            semantic_coherence,
            abstractness_level: Self::calculate_abstractness(keywords),
            semantic_diversity: Self::calculate_semantic_diversity(keywords),
            conceptual_density: keywords.len() as f64 / 10.0, // Simplified
        }
    }

    /// Calculate basic quality metrics for a topic
    pub fn calculate_quality_metrics(
        keywords: &[String],
        all_content_words: &[String],
    ) -> TopicQualityMetrics {
        let internal_coherence = Self::calculate_internal_coherence(keywords);
        let distinctiveness = Self::calculate_distinctiveness(keywords, all_content_words);

        TopicQualityMetrics {
            internal_coherence,
            distinctiveness,
            focus: Self::calculate_focus(keywords),
            coverage: Self::calculate_coverage(keywords, all_content_words),
            stability: 0.7, // Simplified - would require temporal analysis
            interpretability: internal_coherence * distinctiveness,
            complexity: keywords.len() as f64 / 5.0,
        }
    }

    /// Analyze topic evolution (simplified version)
    pub fn analyze_topic_evolution(keywords: &[String], sentences: &[String]) -> TopicEvolution {
        let trajectory = Self::calculate_intensity_trajectory(keywords, sentences);
        let peak_position = trajectory
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        TopicEvolution {
            evolution_pattern: "linear".to_string(), // Simplified
            intensity_trajectory: trajectory.clone(),
            development_stages: Vec::new(), // Simplified
            peak_position,
            consistency_score: Self::calculate_trajectory_consistency(&trajectory),
            lifespan_ratio: 1.0, // Simplified
        }
    }

    // Private helper methods

    fn get_stopwords() -> HashSet<String> {
        let words = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may", "might", "can", "this",
            "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
        ];

        words.into_iter().map(|s| s.to_string()).collect()
    }

    fn identify_semantic_fields(keywords: &[String]) -> Vec<String> {
        // Simplified semantic field identification
        let mut fields = Vec::new();

        let tech_indicators = vec!["computer", "software", "data", "algorithm", "system"];
        let science_indicators = vec!["research", "study", "analysis", "theory", "experiment"];
        let business_indicators = vec!["market", "company", "profit", "strategy", "customer"];

        let has_tech = keywords
            .iter()
            .any(|k| tech_indicators.contains(&k.as_str()));
        let has_science = keywords
            .iter()
            .any(|k| science_indicators.contains(&k.as_str()));
        let has_business = keywords
            .iter()
            .any(|k| business_indicators.contains(&k.as_str()));

        if has_tech {
            fields.push("technology".to_string());
        }
        if has_science {
            fields.push("science".to_string());
        }
        if has_business {
            fields.push("business".to_string());
        }

        if fields.is_empty() {
            fields.push("general".to_string());
        }

        fields
    }

    fn create_conceptual_clusters(
        keywords: &[String],
    ) -> Vec<crate::metrics::coherence::topic_coherence::results::ConceptualCluster> {
        // Simplified clustering - group by first letter
        let mut clusters = HashMap::new();

        for keyword in keywords {
            if let Some(first_char) = keyword.chars().next() {
                let cluster_name = format!("cluster_{}", first_char);
                clusters
                    .entry(cluster_name.clone())
                    .or_insert_with(Vec::new)
                    .push(keyword.clone());
            }
        }

        clusters
            .into_iter()
            .filter(|(_, words)| words.len() > 1)
            .map(|(cluster_name, words)| {
                crate::metrics::coherence::topic_coherence::results::ConceptualCluster {
                    cluster_name,
                    coherence: 0.7,  // Simplified
                    centrality: 0.5, // Simplified
                    semantic_weight: words.len() as f64 / keywords.len() as f64,
                    words,
                }
            })
            .collect()
    }

    fn calculate_semantic_coherence(keywords: &[String]) -> f64 {
        // Simplified semantic coherence based on word similarity
        if keywords.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..keywords.len() {
            for j in (i + 1)..keywords.len() {
                // Very basic similarity - common prefixes/suffixes
                let sim = if keywords[i].len() > 2 && keywords[j].len() > 2 {
                    let prefix_match = keywords[i][..2] == keywords[j][..2];
                    if prefix_match {
                        0.6
                    } else {
                        0.2
                    }
                } else {
                    0.1
                };
                total_similarity += sim;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            0.0
        }
    }

    fn calculate_abstractness(keywords: &[String]) -> f64 {
        // Simplified abstractness calculation
        let abstract_indicators = vec!["concept", "idea", "theory", "principle", "approach"];
        let concrete_indicators = vec!["tool", "object", "device", "machine", "product"];

        let abstract_count = keywords
            .iter()
            .filter(|k| abstract_indicators.iter().any(|a| k.contains(a)))
            .count();

        let concrete_count = keywords
            .iter()
            .filter(|k| concrete_indicators.iter().any(|c| k.contains(c)))
            .count();

        if abstract_count + concrete_count > 0 {
            abstract_count as f64 / (abstract_count + concrete_count) as f64
        } else {
            0.5 // Neutral
        }
    }

    fn calculate_semantic_diversity(keywords: &[String]) -> f64 {
        // Simplified diversity based on unique first letters
        let unique_starts: HashSet<char> =
            keywords.iter().filter_map(|k| k.chars().next()).collect();

        unique_starts.len() as f64 / keywords.len() as f64
    }

    fn calculate_internal_coherence(keywords: &[String]) -> f64 {
        Self::calculate_semantic_coherence(keywords)
    }

    fn calculate_distinctiveness(keywords: &[String], all_words: &[String]) -> f64 {
        if all_words.is_empty() {
            return 0.0;
        }

        let unique_in_topic = keywords
            .iter()
            .filter(|k| {
                let count_in_all = all_words.iter().filter(|w| w == k).count();
                count_in_all <= 2 // Word appears rarely in overall text
            })
            .count();

        unique_in_topic as f64 / keywords.len() as f64
    }

    fn calculate_focus(keywords: &[String]) -> f64 {
        // Higher focus for fewer, more specific keywords
        let max_focus_keywords = 5.0;
        (max_focus_keywords - keywords.len() as f64).max(0.0) / max_focus_keywords
    }

    fn calculate_coverage(keywords: &[String], all_words: &[String]) -> f64 {
        if all_words.is_empty() {
            return 0.0;
        }

        let covered_words = all_words.iter().filter(|w| keywords.contains(w)).count();

        covered_words as f64 / all_words.len() as f64
    }

    fn calculate_intensity_trajectory(keywords: &[String], sentences: &[String]) -> Vec<f64> {
        let window_size = (sentences.len() / 10).max(1);
        let mut trajectory = Vec::new();

        for i in 0..sentences.len() {
            let window_start = i.saturating_sub(window_size);
            let window_end = (i + window_size).min(sentences.len());
            let window = &sentences[window_start..window_end];

            let intensity = Self::calculate_window_intensity(keywords, window);
            trajectory.push(intensity);
        }

        trajectory
    }

    fn calculate_window_intensity(keywords: &[String], window: &[String]) -> f64 {
        let window_text = window.join(" ").to_lowercase();
        let matches = keywords
            .iter()
            .map(|k| window_text.matches(&k.to_lowercase()).count())
            .sum::<usize>();

        let total_words = window
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>();

        if total_words > 0 {
            matches as f64 / total_words as f64
        } else {
            0.0
        }
    }

    fn calculate_trajectory_consistency(trajectory: &[f64]) -> f64 {
        if trajectory.len() < 2 {
            return 1.0;
        }

        let mean = trajectory.iter().sum::<f64>() / trajectory.len() as f64;
        let variance =
            trajectory.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / trajectory.len() as f64;

        1.0 / (1.0 + variance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_content_words() {
        let sentences = vec![
            "The computer is running software.".to_string(),
            "This is a test sentence.".to_string(),
        ];

        let content_words = ExtractionUtils::extract_content_words(&sentences);
        assert!(content_words.contains(&"computer".to_string()));
        assert!(content_words.contains(&"software".to_string()));
        assert!(!content_words.contains(&"the".to_string()));
        assert!(!content_words.contains(&"is".to_string()));
    }

    #[test]
    fn test_build_semantic_profile() {
        let keywords = vec![
            "computer".to_string(),
            "software".to_string(),
            "programming".to_string(),
        ];

        let profile = ExtractionUtils::build_semantic_profile(&keywords);
        assert!(!profile.semantic_fields.is_empty());
        assert!(profile.semantic_coherence > 0.0);
    }

    #[test]
    fn test_calculate_quality_metrics() {
        let keywords = vec!["computer".to_string(), "software".to_string()];
        let all_words = vec![
            "computer".to_string(),
            "software".to_string(),
            "hardware".to_string(),
            "programming".to_string(),
        ];

        let metrics = ExtractionUtils::calculate_quality_metrics(&keywords, &all_words);
        assert!(metrics.internal_coherence >= 0.0);
        assert!(metrics.distinctiveness >= 0.0);
        assert!(metrics.coverage >= 0.0);
    }
}
