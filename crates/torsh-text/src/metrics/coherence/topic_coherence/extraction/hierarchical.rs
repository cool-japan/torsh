//! Hierarchical topic extraction algorithm

use super::{ExtractionError, ExtractionUtils, TopicExtractor};
use crate::metrics::coherence::topic_coherence::{
    config::TopicExtractionConfig, results::Topic, similarity::SimilarityCalculator,
};
use std::collections::HashMap;

pub struct HierarchicalExtractor {
    config: TopicExtractionConfig,
    similarity_calculator: SimilarityCalculator,
}

impl HierarchicalExtractor {
    pub fn new(config: TopicExtractionConfig, similarity_calculator: SimilarityCalculator) -> Self {
        Self {
            config,
            similarity_calculator,
        }
    }
}

impl TopicExtractor for HierarchicalExtractor {
    fn extract_topics(&self, sentences: &[String]) -> Result<Vec<Topic>, ExtractionError> {
        self.validate_content(sentences)?;
        let content_words = ExtractionUtils::extract_content_words(sentences);
        let mut topics = Vec::new();

        // Simplified hierarchical clustering
        let mut main_clusters = Vec::new();
        let mut used = std::collections::HashSet::new();

        // First level clustering
        for word in &content_words {
            if used.contains(word) {
                continue;
            }
            let mut cluster = vec![word.clone()];
            used.insert(word.clone());

            for other in &content_words {
                if !used.contains(other)
                    && self.similarity_calculator.calculate_similarity(word, other) >= 0.7
                {
                    cluster.push(other.clone());
                    used.insert(other.clone());
                }
            }

            if cluster.len() >= self.config.min_topic_size {
                main_clusters.push((cluster, 0)); // Level 0
            }
        }

        for (i, (keywords, level)) in main_clusters
            .into_iter()
            .take(self.config.max_topics)
            .enumerate()
        {
            topics.push(Topic {
                topic_id: format!("hier_topic_{}", i),
                keywords: keywords.clone(),
                coherence_score: self.similarity_calculator.topic_coherence(&keywords),
                span: (0, sentences.len().saturating_sub(1)),
                prominence: 0.8 - (level as f64 * 0.2),
                density: 0.7,
                evolution: ExtractionUtils::analyze_topic_evolution(&keywords, sentences),
                semantic_profile: ExtractionUtils::build_semantic_profile(&keywords),
                quality_metrics: ExtractionUtils::calculate_quality_metrics(
                    &keywords,
                    &content_words,
                ),
                hierarchical_level: level,
                relationships: Vec::new(),
            });
        }

        Ok(self.post_process_topics(topics, sentences))
    }

    fn algorithm_name(&self) -> &'static str {
        "Hierarchical Clustering"
    }
    fn get_parameters(&self) -> HashMap<String, String> {
        HashMap::from([("approach".to_string(), "Hierarchical".to_string())])
    }
}
