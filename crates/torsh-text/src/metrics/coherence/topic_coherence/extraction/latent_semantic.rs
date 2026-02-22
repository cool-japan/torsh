//! Latent Semantic Analysis (LSA) topic extraction algorithm

use super::{ExtractionError, ExtractionUtils, TopicExtractor};
use crate::metrics::coherence::topic_coherence::{
    config::TopicExtractionConfig, results::Topic, similarity::SimilarityCalculator,
};
use std::collections::HashMap;

/// LSA-based topic extractor
pub struct LatentSemanticExtractor {
    config: TopicExtractionConfig,
    similarity_calculator: SimilarityCalculator,
}

impl LatentSemanticExtractor {
    pub fn new(config: TopicExtractionConfig, similarity_calculator: SimilarityCalculator) -> Self {
        Self {
            config,
            similarity_calculator,
        }
    }
}

impl TopicExtractor for LatentSemanticExtractor {
    fn extract_topics(&self, sentences: &[String]) -> Result<Vec<Topic>, ExtractionError> {
        self.validate_content(sentences)?;
        // Simplified LSA implementation - in practice would use SVD
        let content_words = ExtractionUtils::extract_content_words(sentences);
        let mut topics = Vec::new();

        // Group words by semantic similarity (simplified LSA)
        let mut clusters = Vec::new();
        let mut used_words = std::collections::HashSet::new();

        for word in &content_words {
            if used_words.contains(word) {
                continue;
            }

            let mut cluster = vec![word.clone()];
            used_words.insert(word.clone());

            for other_word in &content_words {
                if !used_words.contains(other_word) {
                    let sim = self
                        .similarity_calculator
                        .semantic_similarity(word, other_word);
                    if sim >= 0.6 {
                        cluster.push(other_word.clone());
                        used_words.insert(other_word.clone());
                    }
                }
            }

            if cluster.len() >= self.config.min_topic_size {
                clusters.push(cluster);
            }
        }

        for (i, keywords) in clusters
            .into_iter()
            .take(self.config.max_topics)
            .enumerate()
        {
            topics.push(Topic {
                topic_id: format!("lsa_topic_{}", i),
                keywords: keywords.clone(),
                coherence_score: self.similarity_calculator.topic_coherence(&keywords),
                span: (0, sentences.len().saturating_sub(1)),
                prominence: 0.7,
                density: 0.5,
                evolution: ExtractionUtils::analyze_topic_evolution(&keywords, sentences),
                semantic_profile: ExtractionUtils::build_semantic_profile(&keywords),
                quality_metrics: ExtractionUtils::calculate_quality_metrics(
                    &keywords,
                    &content_words,
                ),
                hierarchical_level: 0,
                relationships: Vec::new(),
            });
        }

        Ok(self.post_process_topics(topics, sentences))
    }

    fn algorithm_name(&self) -> &'static str {
        "Latent Semantic Analysis"
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("approach".to_string(), "LSA".to_string());
        params
    }
}
