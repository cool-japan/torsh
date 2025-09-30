//! Topic coherence metrics calculation module
//!
//! This module provides comprehensive metrics calculation for topic coherence analysis.
//! It includes all core metrics like consistency, shift coherence, development, and
//! thematic unity, with statistical analysis and quality assessment.

use crate::metrics::coherence::topic_coherence::{
    config::MetricsConfig,
    results::{
        DetailedTopicMetrics, ThematicProgressionPattern, Topic, TopicRelationshipAnalysis,
        TopicTransition, TopicTransitionType,
    },
    similarity::SimilarityCalculator,
};
use std::collections::HashMap;

/// Comprehensive topic coherence metrics calculator
pub struct TopicCoherenceMetricsCalculator {
    config: MetricsConfig,
    similarity_calculator: SimilarityCalculator,
}

impl TopicCoherenceMetricsCalculator {
    /// Create a new metrics calculator
    pub fn new(config: MetricsConfig, similarity_calculator: SimilarityCalculator) -> Self {
        Self {
            config,
            similarity_calculator,
        }
    }

    /// Calculate overall topic consistency score
    pub fn calculate_topic_consistency(&self, topics: &[Topic]) -> f64 {
        if topics.is_empty() {
            return 0.0;
        }

        let coherence_scores: Vec<f64> = topics.iter().map(|topic| topic.coherence_score).collect();

        let average_coherence =
            coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64;

        // Calculate consistency based on variance in coherence scores
        let mean = average_coherence;
        let variance = coherence_scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / coherence_scores.len() as f64;

        let consistency = 1.0 / (1.0 + variance);

        // Weighted combination of average coherence and consistency
        (average_coherence * 0.7) + (consistency * 0.3)
    }

    /// Calculate topic shift coherence based on transitions
    pub fn calculate_topic_shift_coherence(&self, transitions: &[TopicTransition]) -> f64 {
        if transitions.is_empty() {
            return 1.0; // Perfect coherence if no transitions
        }

        let transition_qualities: Vec<f64> = transitions
            .iter()
            .map(|transition| transition.transition_quality)
            .collect();

        let average_quality =
            transition_qualities.iter().sum::<f64>() / transition_qualities.len() as f64;

        // Bonus for smooth transitions
        let smooth_transitions = transitions
            .iter()
            .filter(|t| {
                t.transition_type == TopicTransitionType::Smooth
                    || t.transition_type == TopicTransitionType::Gradual
            })
            .count();

        let smooth_ratio = smooth_transitions as f64 / transitions.len() as f64;

        // Weighted combination
        (average_quality * 0.8) + (smooth_ratio * 0.2)
    }

    /// Calculate topic development quality
    pub fn calculate_topic_development(&self, topics: &[Topic], sentences: &[String]) -> f64 {
        if topics.is_empty() || sentences.is_empty() {
            return 0.0;
        }

        let mut total_development = 0.0;
        let mut development_count = 0;

        for topic in topics {
            let development_score = self.calculate_single_topic_development(topic, sentences);
            if development_score > 0.0 {
                total_development += development_score;
                development_count += 1;
            }
        }

        if development_count > 0 {
            total_development / development_count as f64
        } else {
            0.0
        }
    }

    /// Calculate thematic unity across all topics
    pub fn calculate_thematic_unity(&self, topics: &[Topic]) -> f64 {
        if topics.len() < 2 {
            return 1.0; // Perfect unity with single topic
        }

        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        // Calculate pairwise similarities between topics
        for i in 0..topics.len() {
            for j in (i + 1)..topics.len() {
                let similarity = self
                    .similarity_calculator
                    .keyword_set_similarity(&topics[i].keywords, &topics[j].keywords);
                total_similarity += similarity;
                pair_count += 1;
            }
        }

        let average_similarity = if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            0.0
        };

        // Unity is moderate similarity (not too high, not too low)
        let optimal_similarity = 0.4;
        let deviation = (average_similarity - optimal_similarity).abs();
        1.0 - (deviation * 2.0).min(1.0)
    }

    /// Calculate topic distribution across text
    pub fn calculate_topic_distribution(&self, topics: &[Topic]) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();

        if topics.is_empty() {
            return distribution;
        }

        let total_prominence: f64 = topics.iter().map(|t| t.prominence).sum();

        for topic in topics {
            let proportion = if total_prominence > 0.0 {
                topic.prominence / total_prominence
            } else {
                1.0 / topics.len() as f64
            };

            distribution.insert(topic.topic_id.clone(), proportion);
        }

        distribution
    }

    /// Calculate coherence score for each individual topic
    pub fn calculate_coherence_per_topic(&self, topics: &[Topic]) -> HashMap<String, f64> {
        let mut coherence_map = HashMap::new();

        for topic in topics {
            coherence_map.insert(topic.topic_id.clone(), topic.coherence_score);
        }

        coherence_map
    }

    /// Generate detailed metrics with statistical analysis
    pub fn generate_detailed_metrics(
        &self,
        topics: &[Topic],
        sentences: &[String],
        transitions: &[TopicTransition],
    ) -> DetailedTopicMetrics {
        let coherence_scores: Vec<f64> = topics.iter().map(|t| t.coherence_score).collect();
        let transition_qualities: Vec<f64> =
            transitions.iter().map(|t| t.transition_quality).collect();

        let average_topic_coherence = if !coherence_scores.is_empty() {
            coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
        } else {
            0.0
        };

        let topic_coherence_variance = if coherence_scores.len() > 1 {
            let mean = average_topic_coherence;
            coherence_scores
                .iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f64>()
                / coherence_scores.len() as f64
        } else {
            0.0
        };

        let average_transition_quality = if !transition_qualities.is_empty() {
            transition_qualities.iter().sum::<f64>() / transition_qualities.len() as f64
        } else {
            0.0
        };

        let high_quality_transitions = transitions
            .iter()
            .filter(|t| t.transition_quality >= 0.7)
            .count();

        let total_text_length = sentences.iter().map(|s| s.len()).sum::<usize>();
        let covered_length: usize = topics
            .iter()
            .map(|t| {
                let (start, end) = t.span;
                if end < sentences.len() && start <= end {
                    sentences[start..=end]
                        .iter()
                        .map(|s| s.len())
                        .sum::<usize>()
                } else {
                    0
                }
            })
            .sum();

        let topic_coverage_ratio = if total_text_length > 0 {
            covered_length as f64 / total_text_length as f64
        } else {
            0.0
        };

        let average_topic_lifespan = if !topics.is_empty() {
            topics
                .iter()
                .map(|t| t.evolution.lifespan_ratio)
                .sum::<f64>()
                / topics.len() as f64
        } else {
            0.0
        };

        let topic_overlap_ratio = self.calculate_topic_overlap_ratio(topics);
        let semantic_diversity = self.calculate_semantic_diversity(topics);

        DetailedTopicMetrics {
            average_topic_coherence,
            topic_coherence_variance,
            average_transition_quality,
            high_quality_transitions,
            topic_coverage_ratio,
            average_topic_lifespan,
            topic_overlap_ratio,
            semantic_diversity,
        }
    }

    /// Analyze relationships between topics
    pub fn analyze_topic_relationships(&self, topics: &[Topic]) -> TopicRelationshipAnalysis {
        let total_possible_connections = if topics.len() > 1 {
            topics.len() * (topics.len() - 1) / 2
        } else {
            1
        };

        let mut total_connections = 0;
        let mut total_strength = 0.0;
        let mut relationship_types = std::collections::HashSet::new();

        for topic in topics {
            total_connections += topic.relationships.len();
            for relationship in &topic.relationships {
                total_strength += relationship.strength;
                relationship_types.insert(relationship.relationship_type.clone());
            }
        }

        let network_density = total_connections as f64 / total_possible_connections as f64;
        let average_relationship_strength = if total_connections > 0 {
            total_strength / total_connections as f64
        } else {
            0.0
        };

        // Identify central topics (those with most connections)
        let mut topic_connection_counts: Vec<_> = topics
            .iter()
            .map(|t| (t.topic_id.clone(), t.relationships.len()))
            .collect();
        topic_connection_counts.sort_by(|a, b| b.1.cmp(&a.1));

        let central_topics: Vec<String> = topic_connection_counts
            .into_iter()
            .take(3)
            .map(|(id, _)| id)
            .collect();

        // Simple clustering based on relationships
        let topic_clusters = self.identify_topic_clusters(topics);

        TopicRelationshipAnalysis {
            network_density,
            central_topics,
            topic_clusters,
            average_relationship_strength,
            relationship_types_count: relationship_types.len(),
        }
    }

    /// Calculate topic transitions between adjacent topics
    pub fn calculate_topic_transitions(
        &self,
        topics: &[Topic],
        sentences: &[String],
    ) -> Vec<TopicTransition> {
        let mut transitions = Vec::new();

        if topics.len() < 2 {
            return transitions;
        }

        // Sort topics by their starting position
        let mut sorted_topics = topics.to_vec();
        sorted_topics.sort_by(|a, b| a.span.0.cmp(&b.span.0));

        for i in 0..(sorted_topics.len() - 1) {
            let current_topic = &sorted_topics[i];
            let next_topic = &sorted_topics[i + 1];

            // Check if topics overlap or are adjacent
            if current_topic.span.1 + 1 >= next_topic.span.0 {
                let transition_position = (current_topic.span.1 + next_topic.span.0) / 2;

                let transition_quality =
                    self.calculate_transition_quality(current_topic, next_topic, sentences);

                let transition_type =
                    self.classify_transition_type(current_topic, next_topic, transition_quality);

                let smoothness = transition_quality;
                let bridging_elements =
                    self.find_bridging_elements(current_topic, next_topic, sentences);

                transitions.push(TopicTransition {
                    from_topic: current_topic.topic_id.clone(),
                    to_topic: next_topic.topic_id.clone(),
                    position: transition_position,
                    transition_quality,
                    transition_type,
                    smoothness,
                    bridging_elements,
                });
            }
        }

        transitions
    }

    // Private helper methods

    fn calculate_single_topic_development(&self, topic: &Topic, sentences: &[String]) -> f64 {
        let (start, end) = topic.span;

        if start > end || end >= sentences.len() {
            return 0.0;
        }

        // Analyze keyword distribution across topic span
        let topic_sentences = &sentences[start..=end];
        let segments = Self::split_into_segments(topic_sentences, 3);

        if segments.len() < 2 {
            return 0.5; // Neutral development for short topics
        }

        let mut development_scores = Vec::new();

        for i in 0..(segments.len() - 1) {
            let current_intensity = self.calculate_segment_intensity(&topic.keywords, &segments[i]);
            let next_intensity =
                self.calculate_segment_intensity(&topic.keywords, &segments[i + 1]);

            // Development is positive change or maintenance of high intensity
            let development = if current_intensity > 0.3 && next_intensity > 0.3 {
                0.8 // Sustained development
            } else if next_intensity > current_intensity {
                0.9 // Growing development
            } else if current_intensity > 0.5 {
                0.6 // Declining but was strong
            } else {
                0.3 // Weak development
            };

            development_scores.push(development);
        }

        development_scores.iter().sum::<f64>() / development_scores.len() as f64
    }

    fn split_into_segments(sentences: &[String], num_segments: usize) -> Vec<Vec<String>> {
        let segment_size = (sentences.len() / num_segments).max(1);
        let mut segments = Vec::new();

        for i in 0..num_segments {
            let start = i * segment_size;
            let end = ((i + 1) * segment_size).min(sentences.len());

            if start < sentences.len() {
                segments.push(sentences[start..end].to_vec());
            }
        }

        segments
    }

    fn calculate_segment_intensity(&self, keywords: &[String], segment: &[String]) -> f64 {
        if segment.is_empty() {
            return 0.0;
        }

        let segment_text = segment.join(" ").to_lowercase();
        let matches: usize = keywords
            .iter()
            .map(|keyword| segment_text.matches(&keyword.to_lowercase()).count())
            .sum();

        let total_words: usize = segment.iter().map(|s| s.split_whitespace().count()).sum();

        if total_words > 0 {
            matches as f64 / total_words as f64
        } else {
            0.0
        }
    }

    fn calculate_topic_overlap_ratio(&self, topics: &[Topic]) -> f64 {
        if topics.len() < 2 {
            return 0.0;
        }

        let mut overlaps = 0;
        let mut total_comparisons = 0;

        for i in 0..topics.len() {
            for j in (i + 1)..topics.len() {
                let common_keywords: std::collections::HashSet<_> = topics[i]
                    .keywords
                    .iter()
                    .filter(|k| topics[j].keywords.contains(k))
                    .collect();

                if !common_keywords.is_empty() {
                    overlaps += 1;
                }
                total_comparisons += 1;
            }
        }

        if total_comparisons > 0 {
            overlaps as f64 / total_comparisons as f64
        } else {
            0.0
        }
    }

    fn calculate_semantic_diversity(&self, topics: &[Topic]) -> f64 {
        let mut all_semantic_fields = std::collections::HashSet::new();
        let mut total_fields = 0;

        for topic in topics {
            for field in &topic.semantic_profile.semantic_fields {
                all_semantic_fields.insert(field.clone());
            }
            total_fields += topic.semantic_profile.semantic_fields.len();
        }

        if total_fields > 0 {
            all_semantic_fields.len() as f64 / total_fields as f64
        } else {
            0.0
        }
    }

    fn identify_topic_clusters(&self, topics: &[Topic]) -> Vec<Vec<String>> {
        let mut clusters = Vec::new();
        let mut used_topics = std::collections::HashSet::new();

        for topic in topics {
            if used_topics.contains(&topic.topic_id) {
                continue;
            }

            let mut cluster = vec![topic.topic_id.clone()];
            used_topics.insert(topic.topic_id.clone());

            // Find strongly related topics
            for other_topic in topics {
                if !used_topics.contains(&other_topic.topic_id) {
                    let similarity = self
                        .similarity_calculator
                        .keyword_set_similarity(&topic.keywords, &other_topic.keywords);

                    if similarity >= 0.5 {
                        cluster.push(other_topic.topic_id.clone());
                        used_topics.insert(other_topic.topic_id.clone());
                    }
                }
            }

            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }

        clusters
    }

    fn calculate_transition_quality(
        &self,
        from_topic: &Topic,
        to_topic: &Topic,
        sentences: &[String],
    ) -> f64 {
        // Calculate semantic similarity between topics
        let semantic_similarity = self
            .similarity_calculator
            .keyword_set_similarity(&from_topic.keywords, &to_topic.keywords);

        // Check for bridging content between topics
        let bridge_start = from_topic.span.1.saturating_sub(1);
        let bridge_end = (to_topic.span.0 + 1).min(sentences.len());

        let bridging_quality = if bridge_start < bridge_end {
            let bridge_content = sentences[bridge_start..bridge_end].join(" ");
            let bridge_words: Vec<&str> = bridge_content.split_whitespace().collect();

            let from_matches = from_topic
                .keywords
                .iter()
                .map(|k| bridge_content.matches(k).count())
                .sum::<usize>();

            let to_matches = to_topic
                .keywords
                .iter()
                .map(|k| bridge_content.matches(k).count())
                .sum::<usize>();

            let total_matches = from_matches + to_matches;
            let bridge_ratio = if !bridge_words.is_empty() {
                total_matches as f64 / bridge_words.len() as f64
            } else {
                0.0
            };

            bridge_ratio.min(1.0)
        } else {
            0.0
        };

        // Weighted combination
        (semantic_similarity * 0.7) + (bridging_quality * 0.3)
    }

    fn classify_transition_type(
        &self,
        from_topic: &Topic,
        to_topic: &Topic,
        quality: f64,
    ) -> TopicTransitionType {
        let similarity = self
            .similarity_calculator
            .keyword_set_similarity(&from_topic.keywords, &to_topic.keywords);

        if quality >= 0.8 {
            TopicTransitionType::Smooth
        } else if quality >= 0.6 {
            TopicTransitionType::Gradual
        } else if similarity >= 0.4 {
            TopicTransitionType::Return
        } else if quality < 0.3 {
            TopicTransitionType::Abrupt
        } else {
            TopicTransitionType::Gradual
        }
    }

    fn find_bridging_elements(
        &self,
        from_topic: &Topic,
        to_topic: &Topic,
        sentences: &[String],
    ) -> Vec<String> {
        let mut bridging_elements = Vec::new();

        // Find words that appear in both topics
        for from_keyword in &from_topic.keywords {
            if to_topic.keywords.contains(from_keyword) {
                bridging_elements.push(from_keyword.clone());
            }
        }

        // Find transition phrases near the boundary
        let boundary = (from_topic.span.1 + to_topic.span.0) / 2;
        if boundary < sentences.len() {
            let transition_sentence = &sentences[boundary];
            let transition_phrases = vec![
                "however",
                "moreover",
                "furthermore",
                "in addition",
                "on the other hand",
                "similarly",
                "likewise",
                "in contrast",
                "meanwhile",
                "subsequently",
            ];

            for phrase in transition_phrases {
                if transition_sentence.to_lowercase().contains(phrase) {
                    bridging_elements.push(phrase.to_string());
                }
            }
        }

        bridging_elements
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::coherence::topic_coherence::{
        config::SimilarityConfig,
        results::{SemanticProfile, Topic, TopicEvolution, TopicQualityMetrics},
    };

    fn create_test_calculator() -> TopicCoherenceMetricsCalculator {
        let config = MetricsConfig::default();
        let similarity_config = SimilarityConfig::default();
        let similarity_calculator = SimilarityCalculator::new(similarity_config);
        TopicCoherenceMetricsCalculator::new(config, similarity_calculator)
    }

    fn create_test_topic(id: &str, keywords: Vec<String>, coherence: f64) -> Topic {
        Topic {
            topic_id: id.to_string(),
            keywords,
            coherence_score: coherence,
            span: (0, 10),
            prominence: 0.5,
            density: 0.4,
            evolution: TopicEvolution {
                evolution_pattern: "linear".to_string(),
                intensity_trajectory: vec![0.5, 0.6, 0.7],
                development_stages: Vec::new(),
                peak_position: 1,
                consistency_score: 0.8,
                lifespan_ratio: 0.9,
            },
            semantic_profile: SemanticProfile {
                semantic_fields: vec!["technology".to_string()],
                conceptual_clusters: Vec::new(),
                semantic_coherence: coherence,
                abstractness_level: 0.5,
                semantic_diversity: 0.6,
                conceptual_density: 0.7,
            },
            quality_metrics: TopicQualityMetrics {
                internal_coherence: coherence,
                distinctiveness: 0.7,
                focus: 0.8,
                coverage: 0.6,
                stability: 0.7,
                interpretability: 0.8,
                complexity: 0.5,
            },
            hierarchical_level: 0,
            relationships: Vec::new(),
        }
    }

    #[test]
    fn test_calculate_topic_consistency() {
        let calculator = create_test_calculator();
        let topics = vec![
            create_test_topic("1", vec!["computer".to_string()], 0.8),
            create_test_topic("2", vec!["software".to_string()], 0.7),
        ];

        let consistency = calculator.calculate_topic_consistency(&topics);
        assert!(consistency > 0.0 && consistency <= 1.0);
    }

    #[test]
    fn test_calculate_topic_distribution() {
        let calculator = create_test_calculator();
        let topics = vec![
            create_test_topic("1", vec!["computer".to_string()], 0.8),
            create_test_topic("2", vec!["software".to_string()], 0.6),
        ];

        let distribution = calculator.calculate_topic_distribution(&topics);
        assert_eq!(distribution.len(), 2);
        assert!(distribution.values().all(|&v| v > 0.0 && v <= 1.0));
    }

    #[test]
    fn test_empty_topics() {
        let calculator = create_test_calculator();
        let topics = vec![];

        let consistency = calculator.calculate_topic_consistency(&topics);
        assert_eq!(consistency, 0.0);

        let distribution = calculator.calculate_topic_distribution(&topics);
        assert!(distribution.is_empty());
    }
}
