//! Keyword clustering topic extraction algorithm
//!
//! This module implements topic extraction using keyword clustering based on
//! similarity calculations. It groups related words together to form topics
//! and provides comprehensive analysis of each topic.

use super::{ExtractionError, ExtractionUtils, TopicExtractor};
use crate::metrics::coherence::topic_coherence::{
    config::TopicExtractionConfig,
    results::{SemanticProfile, Topic, TopicEvolution, TopicQualityMetrics, TopicRelationship},
    similarity::SimilarityCalculator,
};
use std::collections::{HashMap, HashSet};

/// Keyword clustering-based topic extractor
pub struct KeywordClusteringExtractor {
    config: TopicExtractionConfig,
    similarity_calculator: SimilarityCalculator,
}

impl KeywordClusteringExtractor {
    /// Create a new keyword clustering extractor
    pub fn new(config: TopicExtractionConfig, similarity_calculator: SimilarityCalculator) -> Self {
        Self {
            config,
            similarity_calculator,
        }
    }

    /// Cluster related words based on similarity scores
    fn cluster_related_words(
        &self,
        content_words: &[String],
    ) -> Result<Vec<Vec<String>>, ExtractionError> {
        if content_words.is_empty() {
            return Ok(Vec::new());
        }

        let mut clusters = Vec::new();
        let mut used_words = HashSet::new();

        // Calculate similarity matrix
        let similarity_matrix = self.build_similarity_matrix(content_words);

        // Greedy clustering algorithm
        for (i, word) in content_words.iter().enumerate() {
            if used_words.contains(&i) {
                continue;
            }

            let mut cluster = vec![word.clone()];
            used_words.insert(i);

            // Find similar words to add to this cluster
            for (j, other_word) in content_words.iter().enumerate() {
                if i != j && !used_words.contains(&j) {
                    let similarity = similarity_matrix[i][j];
                    if similarity >= self.config.topic_threshold {
                        cluster.push(other_word.clone());
                        used_words.insert(j);
                    }
                }
            }

            if cluster.len() >= self.config.min_topic_size {
                clusters.push(cluster);
            }
        }

        // Sort clusters by size (largest first)
        clusters.sort_by(|a, b| b.len().cmp(&a.len()));

        // Limit to max_topics
        if clusters.len() > self.config.max_topics {
            clusters.truncate(self.config.max_topics);
        }

        Ok(clusters)
    }

    /// Build similarity matrix for all word pairs
    fn build_similarity_matrix(&self, words: &[String]) -> Vec<Vec<f64>> {
        let n = words.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                if i == j {
                    matrix[i][j] = 1.0;
                } else {
                    let similarity = self
                        .similarity_calculator
                        .calculate_similarity(&words[i], &words[j]);
                    matrix[i][j] = similarity;
                    matrix[j][i] = similarity; // Symmetric matrix
                }
            }
        }

        matrix
    }

    /// Calculate topic coherence score for a cluster of keywords
    fn calculate_topic_coherence_score(&self, keywords: &[String]) -> f64 {
        self.similarity_calculator.topic_coherence(keywords)
    }

    /// Calculate the text span covered by a topic
    fn calculate_topic_span(&self, keywords: &[String], sentences: &[String]) -> (usize, usize) {
        let mut first_occurrence = sentences.len();
        let mut last_occurrence = 0;

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let sentence_lower = sentence.to_lowercase();
            let has_keyword = keywords
                .iter()
                .any(|keyword| sentence_lower.contains(&keyword.to_lowercase()));

            if has_keyword {
                first_occurrence = first_occurrence.min(sent_idx);
                last_occurrence = last_occurrence.max(sent_idx);
            }
        }

        if first_occurrence <= last_occurrence {
            (first_occurrence, last_occurrence)
        } else {
            (0, 0) // No occurrences found
        }
    }

    /// Calculate topic prominence based on keyword frequency and distribution
    fn calculate_topic_prominence(&self, keywords: &[String], sentences: &[String]) -> f64 {
        if sentences.is_empty() {
            return 0.0;
        }

        let total_sentences = sentences.len();
        let mut sentences_with_topic = 0;
        let mut total_keyword_occurrences = 0;
        let mut total_words = 0;

        for sentence in sentences {
            let sentence_lower = sentence.to_lowercase();
            let words_in_sentence: Vec<&str> = sentence_lower.split_whitespace().collect();
            total_words += words_in_sentence.len();

            let mut has_keyword = false;
            for keyword in keywords {
                let keyword_lower = keyword.to_lowercase();
                let occurrences = words_in_sentence
                    .iter()
                    .filter(|word| **word == keyword_lower)
                    .count();

                if occurrences > 0 {
                    has_keyword = true;
                    total_keyword_occurrences += occurrences;
                }
            }

            if has_keyword {
                sentences_with_topic += 1;
            }
        }

        // Combine sentence coverage and word frequency
        let sentence_coverage = sentences_with_topic as f64 / total_sentences as f64;
        let word_frequency = if total_words > 0 {
            total_keyword_occurrences as f64 / total_words as f64
        } else {
            0.0
        };

        // Weighted combination
        (sentence_coverage * 0.7) + (word_frequency * 10.0 * 0.3)
    }

    /// Calculate topic density (keyword concentration)
    fn calculate_topic_density(
        &self,
        keywords: &[String],
        all_content_words: &[String],
        sentences: &[String],
    ) -> f64 {
        let (start, end) = self.calculate_topic_span(keywords, sentences);

        if start > end || end >= sentences.len() {
            return 0.0;
        }

        // Count keywords in the topic's span
        let mut keyword_count = 0;
        let mut total_word_count = 0;

        for i in start..=end {
            if i < sentences.len() {
                let sentence_words: Vec<&str> = sentences[i].split_whitespace().collect();
                total_word_count += sentence_words.len();

                for word in sentence_words {
                    let word_lower = word.to_lowercase();
                    if keywords.contains(&word_lower) {
                        keyword_count += 1;
                    }
                }
            }
        }

        if total_word_count > 0 {
            keyword_count as f64 / total_word_count as f64
        } else {
            0.0
        }
    }

    /// Filter clusters based on prominence and quality
    fn filter_clusters(
        &self,
        clusters: Vec<Vec<String>>,
        sentences: &[String],
    ) -> Vec<Vec<String>> {
        clusters
            .into_iter()
            .filter(|cluster| {
                let prominence = self.calculate_topic_prominence(cluster, sentences);
                prominence >= self.config.min_topic_prominence
            })
            .collect()
    }

    /// Refine clusters by removing weak connections
    fn refine_clusters(&self, clusters: Vec<Vec<String>>) -> Vec<Vec<String>> {
        clusters
            .into_iter()
            .map(|cluster| {
                if cluster.len() <= 3 {
                    return cluster; // Small clusters don't need refinement
                }

                // Remove words with low average similarity to cluster
                let mut refined_cluster = Vec::new();

                for word in &cluster {
                    let mut total_similarity = 0.0;
                    let mut count = 0;

                    for other_word in &cluster {
                        if word != other_word {
                            total_similarity += self
                                .similarity_calculator
                                .calculate_similarity(word, other_word);
                            count += 1;
                        }
                    }

                    let avg_similarity = if count > 0 {
                        total_similarity / count as f64
                    } else {
                        0.0
                    };

                    // Keep words with above-threshold similarity
                    if avg_similarity >= (self.config.topic_threshold * 0.8) {
                        refined_cluster.push(word.clone());
                    }
                }

                if refined_cluster.len() >= self.config.min_topic_size {
                    refined_cluster
                } else {
                    cluster // Keep original if refinement made it too small
                }
            })
            .filter(|cluster| cluster.len() >= self.config.min_topic_size)
            .collect()
    }
}

impl TopicExtractor for KeywordClusteringExtractor {
    fn extract_topics(&self, sentences: &[String]) -> Result<Vec<Topic>, ExtractionError> {
        self.validate_content(sentences)?;

        // Extract content words
        let content_words = ExtractionUtils::extract_content_words(sentences);

        if content_words.is_empty() {
            return Ok(Vec::new());
        }

        // Cluster related words
        let mut clusters = self.cluster_related_words(&content_words)?;

        // Filter and refine clusters
        clusters = self.filter_clusters(clusters, sentences);
        clusters = self.refine_clusters(clusters);

        // Convert clusters to topics
        let mut topics = Vec::new();

        for (topic_id, keywords) in clusters
            .into_iter()
            .enumerate()
            .take(self.config.max_topics)
        {
            if keywords.len() < self.config.min_topic_size {
                continue;
            }

            // Calculate topic metrics
            let coherence_score = self.calculate_topic_coherence_score(&keywords);
            let span = self.calculate_topic_span(&keywords, sentences);
            let prominence = self.calculate_topic_prominence(&keywords, sentences);
            let density = self.calculate_topic_density(&keywords, &content_words, sentences);

            // Generate comprehensive topic analysis
            let evolution = ExtractionUtils::analyze_topic_evolution(&keywords, sentences);
            let semantic_profile = ExtractionUtils::build_semantic_profile(&keywords);
            let quality_metrics =
                ExtractionUtils::calculate_quality_metrics(&keywords, &content_words);

            let topic = Topic {
                topic_id: format!("cluster_topic_{}", topic_id),
                keywords,
                coherence_score,
                span,
                prominence,
                density,
                evolution,
                semantic_profile,
                quality_metrics,
                hierarchical_level: 0,     // Will be set in post-processing
                relationships: Vec::new(), // Will be set in post-processing
            };

            topics.push(topic);
        }

        // Post-process topics for relationships and hierarchy
        let processed_topics = self.post_process_topics(topics, sentences);

        Ok(processed_topics)
    }

    fn algorithm_name(&self) -> &'static str {
        "Keyword Clustering"
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "topic_threshold".to_string(),
            self.config.topic_threshold.to_string(),
        );
        params.insert(
            "min_topic_size".to_string(),
            self.config.min_topic_size.to_string(),
        );
        params.insert("max_topics".to_string(), self.config.max_topics.to_string());
        params.insert(
            "min_topic_prominence".to_string(),
            self.config.min_topic_prominence.to_string(),
        );
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::coherence::topic_coherence::{
        config::{SimilarityConfig, TopicModelingApproach},
        similarity::SimilarityCalculator,
    };

    fn create_test_extractor() -> KeywordClusteringExtractor {
        let config = TopicExtractionConfig {
            approach: TopicModelingApproach::KeywordClustering,
            min_topic_size: 2,
            max_topics: 5,
            topic_threshold: 0.5,
            keyword_sensitivity: 0.7,
            min_topic_prominence: 0.1,
        };

        let similarity_config = SimilarityConfig::default();
        let similarity_calculator = SimilarityCalculator::new(similarity_config);

        KeywordClusteringExtractor::new(config, similarity_calculator)
    }

    #[test]
    fn test_extract_topics_basic() {
        let extractor = create_test_extractor();
        let sentences = vec![
            "Computer software development requires programming skills.".to_string(),
            "Software engineering involves computer programming and design.".to_string(),
            "The cat sat on the mat peacefully.".to_string(),
            "Animals like cats need comfortable places to rest.".to_string(),
        ];

        let result = extractor.extract_topics(&sentences);
        assert!(result.is_ok());

        let topics = result.unwrap();
        assert!(!topics.is_empty());

        // Should have found at least one topic
        for topic in &topics {
            assert!(!topic.keywords.is_empty());
            assert!(topic.coherence_score >= 0.0);
            assert!(topic.prominence >= 0.0);
        }
    }

    #[test]
    fn test_extract_topics_empty_input() {
        let extractor = create_test_extractor();
        let sentences = vec![];

        let result = extractor.extract_topics(&sentences);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExtractionError::InsufficientContent
        ));
    }

    #[test]
    fn test_extract_topics_insufficient_content() {
        let extractor = create_test_extractor();
        let sentences = vec!["Short.".to_string()];

        let result = extractor.extract_topics(&sentences);
        assert!(result.is_err());
    }

    #[test]
    fn test_cluster_related_words() {
        let extractor = create_test_extractor();
        let words = vec![
            "computer".to_string(),
            "software".to_string(),
            "programming".to_string(),
            "cat".to_string(),
            "animal".to_string(),
        ];

        let result = extractor.cluster_related_words(&words);
        assert!(result.is_ok());

        let clusters = result.unwrap();
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_calculate_topic_span() {
        let extractor = create_test_extractor();
        let keywords = vec!["computer".to_string(), "software".to_string()];
        let sentences = vec![
            "Hello world".to_string(),
            "Computer software is useful".to_string(),
            "More text here".to_string(),
            "Software development".to_string(),
        ];

        let (start, end) = extractor.calculate_topic_span(&keywords, &sentences);
        assert_eq!(start, 1);
        assert_eq!(end, 3);
    }

    #[test]
    fn test_calculate_topic_prominence() {
        let extractor = create_test_extractor();
        let keywords = vec!["computer".to_string()];
        let sentences = vec![
            "The computer is running".to_string(),
            "Software development".to_string(),
            "Computer programming".to_string(),
        ];

        let prominence = extractor.calculate_topic_prominence(&keywords, &sentences);
        assert!(prominence > 0.0);
    }

    #[test]
    fn test_algorithm_name() {
        let extractor = create_test_extractor();
        assert_eq!(extractor.algorithm_name(), "Keyword Clustering");
    }

    #[test]
    fn test_get_parameters() {
        let extractor = create_test_extractor();
        let params = extractor.get_parameters();

        assert!(params.contains_key("topic_threshold"));
        assert!(params.contains_key("min_topic_size"));
        assert!(params.contains_key("max_topics"));
    }
}
