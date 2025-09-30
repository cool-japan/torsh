//! Co-occurrence based topic extraction algorithm

use super::{ExtractionError, ExtractionUtils, TopicExtractor};
use crate::metrics::coherence::topic_coherence::{
    config::TopicExtractionConfig, results::Topic, similarity::SimilarityCalculator,
};
use std::collections::HashMap;

pub struct CoOccurrenceExtractor {
    config: TopicExtractionConfig,
    similarity_calculator: SimilarityCalculator,
}

impl CoOccurrenceExtractor {
    pub fn new(config: TopicExtractionConfig, similarity_calculator: SimilarityCalculator) -> Self {
        Self {
            config,
            similarity_calculator,
        }
    }
}

impl TopicExtractor for CoOccurrenceExtractor {
    fn extract_topics(&self, sentences: &[String]) -> Result<Vec<Topic>, ExtractionError> {
        self.validate_content(sentences)?;
        let content_words = ExtractionUtils::extract_content_words(sentences);
        let mut topics = Vec::new();

        // Simple co-occurrence clustering
        let mut clusters = Vec::new();
        let mut used = std::collections::HashSet::new();

        for word in &content_words {
            if used.contains(word) {
                continue;
            }
            let mut cluster = vec![word.clone()];
            used.insert(word.clone());

            for other in &content_words {
                if !used.contains(other) {
                    let cooc_sim = self
                        .similarity_calculator
                        .cooccurrence_similarity(word, other);
                    if cooc_sim >= 0.5 {
                        cluster.push(other.clone());
                        used.insert(other.clone());
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
                topic_id: format!("cooc_topic_{}", i),
                keywords: keywords.clone(),
                coherence_score: self.similarity_calculator.topic_coherence(&keywords),
                span: (0, sentences.len().saturating_sub(1)),
                prominence: 0.6,
                density: 0.4,
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
        "Co-occurrence Analysis"
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        HashMap::from([("approach".to_string(), "Co-occurrence".to_string())])
    }
}
