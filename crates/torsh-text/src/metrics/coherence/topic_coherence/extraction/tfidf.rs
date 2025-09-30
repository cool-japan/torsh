//! TF-IDF based topic extraction algorithm
//!
//! This module implements topic extraction using TF-IDF (Term Frequency-Inverse Document Frequency)
//! analysis to identify important terms and group them into coherent topics.

use super::{ExtractionError, ExtractionUtils, TopicExtractor};
use crate::metrics::coherence::topic_coherence::{
    config::TopicExtractionConfig, results::Topic, similarity::SimilarityCalculator,
};
use std::collections::HashMap;

/// TF-IDF based topic extractor
pub struct TfIdfExtractor {
    config: TopicExtractionConfig,
    similarity_calculator: SimilarityCalculator,
}

impl TfIdfExtractor {
    pub fn new(config: TopicExtractionConfig, similarity_calculator: SimilarityCalculator) -> Self {
        Self {
            config,
            similarity_calculator,
        }
    }

    fn calculate_tfidf(&self, sentences: &[String]) -> HashMap<String, f64> {
        let content_words = ExtractionUtils::extract_content_words(sentences);
        let mut word_doc_freq = HashMap::new();
        let mut word_total_freq = HashMap::new();

        // Calculate document frequency and total frequency
        for sentence in sentences {
            let sentence_words: Vec<String> = sentence
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            let unique_words: std::collections::HashSet<String> =
                sentence_words.iter().cloned().collect();

            for word in &unique_words {
                if content_words.contains(word) {
                    *word_doc_freq.entry(word.clone()).or_insert(0) += 1;
                }
            }

            for word in &sentence_words {
                if content_words.contains(word) {
                    *word_total_freq.entry(word.clone()).or_insert(0) += 1;
                }
            }
        }

        // Calculate TF-IDF scores
        let mut tfidf_scores = HashMap::new();
        let total_docs = sentences.len() as f64;

        for word in &content_words {
            let tf = *word_total_freq.get(word).unwrap_or(&0) as f64;
            let df = *word_doc_freq.get(word).unwrap_or(&0) as f64;

            if df > 0.0 {
                let idf = (total_docs / df).ln();
                let tfidf = tf * idf;
                tfidf_scores.insert(word.clone(), tfidf);
            }
        }

        tfidf_scores
    }
}

impl TopicExtractor for TfIdfExtractor {
    fn extract_topics(&self, sentences: &[String]) -> Result<Vec<Topic>, ExtractionError> {
        self.validate_content(sentences)?;

        let tfidf_scores = self.calculate_tfidf(sentences);

        // Sort words by TF-IDF score
        let mut scored_words: Vec<_> = tfidf_scores.into_iter().collect();
        scored_words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Group top words into topics using similarity
        let mut topics = Vec::new();
        let mut used_words = std::collections::HashSet::new();

        for (word, _score) in scored_words.iter().take(self.config.max_topics * 5) {
            if used_words.contains(word) {
                continue;
            }

            let mut topic_keywords = vec![word.clone()];
            used_words.insert(word.clone());

            // Find similar words for this topic
            for (other_word, _) in scored_words.iter() {
                if topic_keywords.len() >= 8 {
                    break;
                }
                if used_words.contains(other_word) {
                    continue;
                }

                let similarity = self
                    .similarity_calculator
                    .calculate_similarity(word, other_word);
                if similarity >= self.config.topic_threshold {
                    topic_keywords.push(other_word.clone());
                    used_words.insert(other_word.clone());
                }
            }

            if topic_keywords.len() >= self.config.min_topic_size {
                let coherence_score = self.similarity_calculator.topic_coherence(&topic_keywords);
                let span = (0, sentences.len().saturating_sub(1));
                let evolution =
                    ExtractionUtils::analyze_topic_evolution(&topic_keywords, sentences);
                let semantic_profile = ExtractionUtils::build_semantic_profile(&topic_keywords);
                let quality_metrics = ExtractionUtils::calculate_quality_metrics(
                    &topic_keywords,
                    &ExtractionUtils::extract_content_words(sentences),
                );

                topics.push(Topic {
                    topic_id: format!("tfidf_topic_{}", topics.len()),
                    keywords: topic_keywords,
                    coherence_score,
                    span,
                    prominence: 0.8, // High for TF-IDF selected terms
                    density: 0.6,
                    evolution,
                    semantic_profile,
                    quality_metrics,
                    hierarchical_level: 0,
                    relationships: Vec::new(),
                });

                if topics.len() >= self.config.max_topics {
                    break;
                }
            }
        }

        Ok(self.post_process_topics(topics, sentences))
    }

    fn algorithm_name(&self) -> &'static str {
        "TF-IDF Analysis"
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("approach".to_string(), "TF-IDF".to_string());
        params.insert("max_topics".to_string(), self.config.max_topics.to_string());
        params
    }
}
