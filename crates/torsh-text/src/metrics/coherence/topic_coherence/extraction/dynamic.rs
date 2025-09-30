//! Dynamic topic extraction algorithm

use super::{ExtractionError, ExtractionUtils, TopicExtractor};
use crate::metrics::coherence::topic_coherence::{
    config::TopicExtractionConfig, results::Topic, similarity::SimilarityCalculator,
};
use std::collections::HashMap;

pub struct DynamicExtractor {
    config: TopicExtractionConfig,
    similarity_calculator: SimilarityCalculator,
}

impl DynamicExtractor {
    pub fn new(config: TopicExtractionConfig, similarity_calculator: SimilarityCalculator) -> Self {
        Self {
            config,
            similarity_calculator,
        }
    }
}

impl TopicExtractor for DynamicExtractor {
    fn extract_topics(&self, sentences: &[String]) -> Result<Vec<Topic>, ExtractionError> {
        self.validate_content(sentences)?;
        let content_words = ExtractionUtils::extract_content_words(sentences);
        let mut topics = Vec::new();

        // Dynamic topic modeling - track evolution over text segments
        let segment_size = (sentences.len() / 5).max(1);
        let mut dynamic_clusters = HashMap::new();

        for (seg_idx, segment_start) in (0..sentences.len()).step_by(segment_size).enumerate() {
            let segment_end = (segment_start + segment_size).min(sentences.len());
            let segment = &sentences[segment_start..segment_end];
            let seg_words = ExtractionUtils::extract_content_words(segment);

            // Track word evolution
            for word in &seg_words {
                dynamic_clusters
                    .entry(word.clone())
                    .or_insert_with(Vec::new)
                    .push(seg_idx);
            }
        }

        // Create topics from persistent words
        let mut clusters = Vec::new();
        let mut used = std::collections::HashSet::new();

        for (word, segments) in dynamic_clusters {
            if used.contains(&word) || segments.len() < 2 {
                continue;
            }

            let mut cluster = vec![word.clone()];
            used.insert(word.clone());

            // Find related words that appear in similar segments
            for (other_word, other_segments) in &dynamic_clusters {
                if !used.contains(other_word) && cluster.len() < 8 {
                    let overlap = segments
                        .iter()
                        .filter(|s| other_segments.contains(s))
                        .count();
                    if overlap >= 1
                        && self
                            .similarity_calculator
                            .calculate_similarity(&word, other_word)
                            >= 0.4
                    {
                        cluster.push(other_word.clone());
                        used.insert(other_word.clone());
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
            let mut evolution = ExtractionUtils::analyze_topic_evolution(&keywords, sentences);
            evolution.evolution_pattern = "dynamic".to_string();

            topics.push(Topic {
                topic_id: format!("dyn_topic_{}", i),
                keywords: keywords.clone(),
                coherence_score: self.similarity_calculator.topic_coherence(&keywords),
                span: (0, sentences.len().saturating_sub(1)),
                prominence: 0.75,
                density: 0.6,
                evolution,
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
        "Dynamic Topic Modeling"
    }
    fn get_parameters(&self) -> HashMap<String, String> {
        HashMap::from([("approach".to_string(), "Dynamic".to_string())])
    }
}
